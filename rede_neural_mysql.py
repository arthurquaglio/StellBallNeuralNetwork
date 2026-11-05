# -*- coding: utf-8 -*-
"""
Motor da Rede Neural e Evolução Genética 100% persistido e conectado ao MySQL Server (MySQL Workbench).

Opera exclusivamente sobre o banco de dados relacional `sbr_neural_db` (sem arquivos locais e sem fallback offline):
  - Topologia MLP: 18 Entradas (+1 Bias) -> 6 Ocultos (+1 Bias) -> 4 Saídas (142 genes por indivíduo)
  - Ativação: ReLU truncada em [0.0, 10000.0]
  - Campeão da Geração contabilizado EXCLUSIVAMENTE quando o cavalo conclui 100% da corrida (`concluiu_corrida = 1`)
  - Evolução Genética executada dentro do MySQL via `CALL sp_sbr_evolve_generation(...)`
  - Inferência e Telemetria de Ativações via `CALL sp_sbr_batch_forward_pass(...)`
"""

import os
import threading
import time
import numpy as np
import mysql.connector

from config import (
    MYSQL_CONFIG,
    ARCHITECTURE_NAME,
    DATABASE_DIR,
    SBR_BRAIN_QTD_LAYERS,
    SBR_BRAIN_QTD_INPUT,
    SBR_BRAIN_QTD_HIDE,
    SBR_BRAIN_QTD_OUTPUT,
    ELITE_STEP,
    TAMANHO_DNA,
)


def relu_clamped(x: np.ndarray) -> np.ndarray:
    """Função de ativação ReLU limitada ao intervalo [0.0, 10000.0]."""
    return np.clip(x, 0.0, 10000.0)


def _consume_all_results(cur):
    """Consome todos os result sets de uma Stored Procedure no mysql-connector."""
    if cur.with_rows:
        cur.fetchall()
    while cur.nextset():
        if cur.with_rows:
            cur.fetchall()


class NeuralMySQLManager:
    def __init__(self, pop_size: int):
        self.pop_size = pop_size
        self.n_in = SBR_BRAIN_QTD_INPUT
        self.n_hid = SBR_BRAIN_QTD_HIDE
        self.n_out = SBR_BRAIN_QTD_OUTPUT
        self.elite_step = ELITE_STEP
        self.dna_size = TAMANHO_DNA

        self.geracao = 0
        self.range_random = float(self.dna_size)
        self.distancia_recorde = 0.0
        self.melhor_fitness_geracao = 0.0
        self.media_fitness_geracao = 0.0

        # Rastreamento oficial de Campeão da Geração (SOMENTE quem completou 100% da corrida!)
        self.qtd_concluiram_geracao_ant = 0
        self.campeao_geracao_anterior_str = "Nenhum (0 concluíram a corrida)"
        self.ultimo_campeao_oficial_str = "Nenhum (Ninguém concluiu a corrida ainda)"

        # Matrizes de pesos sincronizadas diretamente das tabelas sbr_genome_weight e sbr_genome_bias do MySQL
        self.W1 = np.zeros((pop_size, self.n_in, self.n_hid), dtype=np.float64)
        self.B1 = np.zeros((pop_size, self.n_hid), dtype=np.float64)
        self.W2 = np.zeros((pop_size, self.n_hid, self.n_out), dtype=np.float64)
        self.B2 = np.zeros((pop_size, self.n_out), dtype=np.float64)

        # Ativações da última inferência para desenhar o HUD da Rede Neural em tempo real
        self.last_in = np.zeros((pop_size, self.n_in), dtype=np.float64)
        self.last_hid = np.zeros((pop_size, self.n_hid), dtype=np.float64)
        self.last_out = np.zeros((pop_size, self.n_out), dtype=np.float64)

        # Metadados reais de cada indivíduo vindos da tabela sbr_genome no MySQL
        self.genome_ids = np.arange(1, pop_size + 1, dtype=np.int32)
        self.origens = ["aleatorio"] * pop_size
        self.parent_ids = [None] * pop_size
        self.in_neuron_ids = []  # Cache dos IDs dos 18 neurônios de entrada no MySQL

        # Estado da conexão com o MySQL Server (Sem fallback local: pausa se desconectado)
        self.mysql_online = False
        self.mysql_version = ""
        self.mysql_status_msg = "Conectando ao MySQL Server (127.0.0.1:3306)..."
        self.mysql_last_latency_ms = 0.0
        self._last_telemetry_ts = 0.0
        self._bg_lock = threading.Lock()
        self._bg_thread = None

        self.try_connect_and_init_mysql()

    # ------------------------------------------------------------------------
    # CONVERSÃO DNA LINEAR <-> CAMADAS
    # ------------------------------------------------------------------------
    def get_dna_vector(self, idx: int) -> np.ndarray:
        w1_flat = self.W1[idx].T.reshape(-1)
        b1_flat = self.B1[idx].reshape(-1)
        w2_flat = self.W2[idx].T.reshape(-1)
        b2_flat = self.B2[idx].reshape(-1)
        return np.concatenate([w1_flat, b1_flat, w2_flat, b2_flat])

    def set_dna_vector(self, idx: int, dna: np.ndarray):
        c1 = self.n_hid * self.n_in
        c2 = c1 + self.n_hid
        c3 = c2 + (self.n_out * self.n_hid)
        self.W1[idx] = dna[0:c1].reshape((self.n_hid, self.n_in)).T
        self.B1[idx] = dna[c1:c2]
        self.W2[idx] = dna[c2:c3].reshape((self.n_out, self.n_hid)).T
        self.B2[idx] = dna[c3:c3 + self.n_out]

    # ------------------------------------------------------------------------
    # CONEXÃO E BOOTSTRAP DO MYSQL SERVER
    # ------------------------------------------------------------------------
    def _open_conn(self, use_database: bool = True):
        cfg = dict(MYSQL_CONFIG)
        if not use_database:
            cfg.pop("database", None)
        cfg["connect_timeout"] = 5
        return mysql.connector.connect(**cfg)

    def _ensure_champion_columns(self):
        """Garante que as colunas de conclusão de corrida existam no banco atual sem perder dados."""
        conn = self._open_conn(use_database=True)
        cur = conn.cursor()
        alter_statements = [
            "ALTER TABLE sbr_genome ADD COLUMN concluiu_corrida TINYINT(1) NOT NULL DEFAULT 0 AFTER eliminado_laser",
            "ALTER TABLE sbr_genome ADD COLUMN ticks_conclusao INT NULL DEFAULT NULL AFTER concluiu_corrida",
            "ALTER TABLE sbr_generation_summary ADD COLUMN qtd_concluiram INT NOT NULL DEFAULT 0 AFTER range_random",
        ]
        for stmt in alter_statements:
            try:
                cur.execute(stmt)
                conn.commit()
            except Exception:
                pass
        cur.close()
        conn.close()

    def try_connect_and_init_mysql(self) -> bool:
        """
        Conecta obrigatoriamente ao MySQL Server (127.0.0.1:3306) e carrega os genomas reais.
        Se falhar, marca `mysql_online = False` para que o simulador fique pausado até reconectar.
        """
        t0 = time.perf_counter()
        try:
            with self._bg_lock:
                conn = self._open_conn(use_database=False)
                cur = conn.cursor()
                cur.execute("SELECT VERSION()")
                self.mysql_version = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name = %s",
                    (MYSQL_CONFIG["database"],),
                )
                has_db = cur.fetchone()[0] > 0
                cur.close()
                conn.close()

                if not has_db:
                    self._install_sql_scripts_to_server()
                else:
                    self._ensure_champion_columns()
                    self._apply_single_sql_file("05_genetic_evolution.sql")
                    self._apply_single_sql_file("06_workbench_views.sql")

                conn = self._open_conn(use_database=True)
                cur = conn.cursor()
                cur.execute(
                    "CALL sp_sbr_create_architecture(%s, %s, %s, %s, %s)",
                    (ARCHITECTURE_NAME, SBR_BRAIN_QTD_LAYERS, self.n_in, self.n_hid, self.n_out),
                )
                _consume_all_results(cur)
                conn.commit()

                # Cacheia os IDs dos 18 neurônios de entrada
                cur.execute("""
                    SELECT n.id
                    FROM sbr_neuron n
                    JOIN sbr_layer l ON l.id = n.layer_id
                    JOIN sbr_architecture a ON a.id = l.architecture_id
                    WHERE a.nome = %s AND l.ordem = 0
                    ORDER BY n.posicao ASC
                """, (ARCHITECTURE_NAME,))
                self.in_neuron_ids = [int(r[0]) for r in cur.fetchall()]

                # Verifica a última geração existente no banco MySQL
                cur.execute("""
                    SELECT COALESCE(MAX(g.geracao), -1)
                    FROM sbr_genome g
                    JOIN sbr_architecture a ON a.id = g.architecture_id
                    WHERE a.nome = %s
                """, (ARCHITECTURE_NAME,))
                max_gen = int(cur.fetchone()[0])

                if max_gen < 0:
                    cur.execute("CALL sp_sbr_init_population(%s, %s, 0)", (ARCHITECTURE_NAME, self.pop_size))
                    _consume_all_results(cur)
                    conn.commit()
                    self.geracao = 0
                else:
                    self.geracao = max_gen

                cur.close()
                conn.close()

                self._load_generation_from_mysql(self.geracao)
                self.mysql_online = True
                self.mysql_last_latency_ms = (time.perf_counter() - t0) * 1000.0
                self.mysql_status_msg = (
                    f"MySQL ONLINE ({self.mysql_version}) | DB: {MYSQL_CONFIG['database']} | "
                    f"Sync: {self.mysql_last_latency_ms:.0f}ms"
                )
                return True

        except Exception as exc:
            self.mysql_online = False
            self.mysql_status_msg = f"ERRO MYSQL (PAUSADO): {str(exc)[:55]} (F5 Reconectar)"
            print(f"[ERRO MYSQL] {exc}")
            return False

    def resetar_populacao_mysql(self) -> bool:
        """Reinicia o treinamento no MySQL do zero (Geração 0) via tecla F6."""
        try:
            with self._bg_lock:
                conn = self._open_conn(use_database=True)
                cur = conn.cursor()
                cur.execute("CALL sp_sbr_init_population(%s, %s, 1)", (ARCHITECTURE_NAME, self.pop_size))
                _consume_all_results(cur)
                conn.commit()
                cur.close()
                conn.close()
                self.geracao = 0
                self.distancia_recorde = 0.0
                self.melhor_fitness_geracao = 0.0
                self.media_fitness_geracao = 0.0
                self.qtd_concluiram_geracao_ant = 0
                self.campeao_geracao_anterior_str = "Nenhum (0 concluíram a corrida)"
                self.ultimo_campeao_oficial_str = "Nenhum (Ninguém concluiu a corrida ainda)"
                self._load_generation_from_mysql(0)
                self.mysql_online = True
                self.mysql_status_msg = (
                    f"MySQL ONLINE ({self.mysql_version}) | DB: {MYSQL_CONFIG['database']} | Reset G0 OK"
                )
                return True
        except Exception as exc:
            self.mysql_online = False
            self.mysql_status_msg = f"ERRO AO RESETAR MYSQL: {str(exc)[:50]}"
            return False

    def carregar_campeao_mysql(self) -> bool:
        """
        Carrega diretamente do MySQL (tabela `sbr_genome`, `sbr_genome_weight`, `sbr_genome_bias`)
        o melhor Campeão Oficial que CONCLUIU a corrida (`concluiu_corrida = 1`) para o Cavalo #0.
        """
        if not self.mysql_online:
            return False
        try:
            with self._bg_lock:
                conn = self._open_conn(use_database=True)
                cur = conn.cursor()
                cur.execute("""
                    SELECT g.id, g.geracao, g.fitness, g.rider_name, g.ticks_conclusao
                    FROM sbr_genome g
                    JOIN sbr_architecture a ON a.id = g.architecture_id
                    WHERE a.nome = %s AND g.concluiu_corrida = 1
                    ORDER BY g.fitness DESC, COALESCE(g.ticks_conclusao, 999999) ASC, g.geracao DESC
                    LIMIT 1
                """, (ARCHITECTURE_NAME,))
                row = cur.fetchone()
                if not row:
                    cur.close()
                    conn.close()
                    self.mysql_status_msg = (
                        f"MySQL ONLINE ({self.mysql_version}) | DB: {MYSQL_CONFIG['database']} | "
                        f"Ainda não há Campeão (ninguém concluiu a corrida)"
                    )
                    return False

                champ_id, champ_gen, champ_fit, champ_rider, champ_ticks = (
                    int(row[0]), int(row[1]), float(row[2]), str(row[3]), row[4]
                )

                cur.execute("""
                    SELECT s.layer_transition, n_from.posicao, n_to.posicao, w.peso
                    FROM sbr_genome_weight w
                    JOIN sbr_synapse s ON s.id = w.synapse_id
                    JOIN sbr_neuron n_from ON n_from.id = s.from_neuron_id
                    JOIN sbr_neuron n_to ON n_to.id = s.to_neuron_id
                    WHERE w.genome_id = %s
                """, (champ_id,))
                for trans, p_from, p_to, peso in cur.fetchall():
                    if trans == 0:
                        self.W1[0, p_from, p_to] = peso
                    else:
                        self.W2[0, p_from, p_to] = peso

                cur.execute("""
                    SELECT l.ordem, n.posicao, b.bias
                    FROM sbr_genome_bias b
                    JOIN sbr_neuron n ON n.id = b.neuron_id
                    JOIN sbr_layer l ON l.id = n.layer_id
                    WHERE b.genome_id = %s
                """, (champ_id,))
                for l_ordem, pos, bias in cur.fetchall():
                    if l_ordem == 1:
                        self.B1[0, pos] = bias
                    elif l_ordem == 2:
                        self.B2[0, pos] = bias

                cur.close()
                conn.close()

                t_str = f"{champ_ticks}t" if champ_ticks else f"{champ_fit:.0f}px"
                self.mysql_status_msg = (
                    f"MySQL ONLINE ({self.mysql_version}) | DB: {MYSQL_CONFIG['database']} | "
                    f"Campeão {champ_rider} (G{champ_gen} #{champ_id} - {t_str}) no Slot #0"
                )
                return True
        except Exception as exc:
            print(f"[ERRO MYSQL AO CARREGAR CAMPEAO] {exc}")
            return False

    def _apply_single_sql_file(self, fname: str):
        fpath = os.path.join(DATABASE_DIR, fname)
        if not os.path.exists(fpath):
            return
        conn = self._open_conn(use_database=True)
        cur = conn.cursor()
        with open(fpath, "r", encoding="utf-8") as fp:
            raw_sql = fp.read()
        sep = "$$"
        cleaned = raw_sql.replace("DELIMITER $$", "").replace("DELIMITER ;", "")
        blocks = cleaned.split(sep) if sep in raw_sql else [cleaned]
        for b in blocks:
            s = b.strip()
            if not s:
                continue
            if "CREATE PROCEDURE" in s.upper():
                idx = s.upper().find("CREATE PROCEDURE")
                pre = s[:idx].strip()
                proc = s[idx:].strip()
                for sub in pre.split(";"):
                    if sub.strip():
                        cur.execute(sub.strip())
                cur.execute(proc)
            else:
                for sub in s.split(";"):
                    if sub.strip():
                        cur.execute(sub.strip())
        conn.commit()
        cur.close()
        conn.close()

    def _install_sql_scripts_to_server(self):
        conn = self._open_conn(use_database=False)
        cur = conn.cursor()
        with open(os.path.join(DATABASE_DIR, "01_schema.sql"), "r", encoding="utf-8") as fp:
            for sub in fp.read().split(";"):
                if sub.strip():
                    cur.execute(sub.strip())
        conn.commit()
        cur.close()
        conn.close()

        for fname in [
            "02_seed_architecture.sql",
            "03_population_init.sql",
            "04_neural_inference.sql",
            "05_genetic_evolution.sql",
            "06_workbench_views.sql",
        ]:
            self._apply_single_sql_file(fname)

    def _load_generation_from_mysql(self, geracao: int):
        """Carrega todos os 250 genomas, pesos, biases e último Campeão Oficial do MySQL."""
        conn = self._open_conn(use_database=True)
        cur = conn.cursor()

        cur.execute("""
            SELECT g.id, g.slot_index, g.origem, g.parent_genome_id
            FROM sbr_genome g
            JOIN sbr_architecture a ON a.id = g.architecture_id
            WHERE a.nome = %s AND g.geracao = %s
            ORDER BY g.slot_index ASC
        """, (ARCHITECTURE_NAME, geracao))
        rows = cur.fetchall()

        if len(rows) != self.pop_size:
            cur.execute("CALL sp_sbr_init_population(%s, %s, 1)", (ARCHITECTURE_NAME, self.pop_size))
            _consume_all_results(cur)
            conn.commit()
            geracao = 0
            self.geracao = 0
            cur.execute("""
                SELECT g.id, g.slot_index, g.origem, g.parent_genome_id
                FROM sbr_genome g
                JOIN sbr_architecture a ON a.id = g.architecture_id
                WHERE a.nome = %s AND g.geracao = 0
                ORDER BY g.slot_index ASC
            """, (ARCHITECTURE_NAME,))
            rows = cur.fetchall()

        slot_map = {}
        for gid, slot_idx, origem, parent_id in rows:
            if 0 <= slot_idx < self.pop_size:
                self.genome_ids[slot_idx] = gid
                self.origens[slot_idx] = origem
                self.parent_ids[slot_idx] = parent_id
                slot_map[gid] = slot_idx

        # Carrega pesos sinápticos
        cur.execute("""
            SELECT w.genome_id, s.layer_transition, n_from.posicao, n_to.posicao, w.peso
            FROM sbr_genome_weight w
            JOIN sbr_genome g ON g.id = w.genome_id
            JOIN sbr_architecture a ON a.id = g.architecture_id
            JOIN sbr_synapse s ON s.id = w.synapse_id
            JOIN sbr_neuron n_from ON n_from.id = s.from_neuron_id
            JOIN sbr_neuron n_to ON n_to.id = s.to_neuron_id
            WHERE a.nome = %s AND g.geracao = %s
        """, (ARCHITECTURE_NAME, geracao))
        for gid, trans, p_from, p_to, peso in cur.fetchall():
            idx = slot_map.get(gid)
            if idx is not None:
                if trans == 0:
                    self.W1[idx, p_from, p_to] = peso
                else:
                    self.W2[idx, p_from, p_to] = peso

        # Carrega biases
        cur.execute("""
            SELECT b.genome_id, l.ordem, n.posicao, b.bias
            FROM sbr_genome_bias b
            JOIN sbr_genome g ON g.id = b.genome_id
            JOIN sbr_architecture a ON a.id = g.architecture_id
            JOIN sbr_neuron n ON n.id = b.neuron_id
            JOIN sbr_layer l ON l.id = n.layer_id
            WHERE a.nome = %s AND g.geracao = %s
        """, (ARCHITECTURE_NAME, geracao))
        for gid, l_ordem, pos, bias in cur.fetchall():
            idx = slot_map.get(gid)
            if idx is not None:
                if l_ordem == 1:
                    self.B1[idx, pos] = bias
                elif l_ordem == 2:
                    self.B2[idx, pos] = bias

        # Carrega estatísticas históricas do MySQL
        cur.execute("""
            SELECT COALESCE(s.range_random, %s), COALESCE(MAX(s2.distancia_recorde), 0.0)
            FROM sbr_generation_summary s
            JOIN sbr_architecture a ON a.id = s.architecture_id
            LEFT JOIN sbr_generation_summary s2 ON s2.architecture_id = a.id
            WHERE a.nome = %s AND s.geracao = %s
            GROUP BY s.range_random
        """, (float(self.dna_size), ARCHITECTURE_NAME, geracao))
        srow = cur.fetchone()
        if srow:
            self.range_random = float(srow[0])
            self.distancia_recorde = max(self.distancia_recorde, float(srow[1]))

        # Carrega o último Campeão Oficial que concluiu a corrida no histórico do MySQL
        cur.execute("""
            SELECT s.geracao, s.melhor_genome_id, s.melhor_rider_name, s.qtd_concluiram, g.ticks_conclusao
            FROM sbr_generation_summary s
            JOIN sbr_architecture a ON a.id = s.architecture_id
            JOIN sbr_genome g ON g.id = s.melhor_genome_id
            WHERE a.nome = %s AND s.qtd_concluiram > 0 AND s.melhor_genome_id IS NOT NULL
            ORDER BY s.geracao DESC
            LIMIT 1
        """, (ARCHITECTURE_NAME,))
        crow = cur.fetchone()
        if crow:
            c_gen, c_id, c_name, c_qtd, c_ticks = int(crow[0]), int(crow[1]), str(crow[2]), int(crow[3]), crow[4]
            t_info = f" em {c_ticks} ticks" if c_ticks else ""
            self.ultimo_campeao_oficial_str = f"G{c_gen}: {c_name} (#{c_id}{t_info} | {c_qtd} finalistas)"

        cur.close()
        conn.close()

    # ------------------------------------------------------------------------
    # FORWARD PASS E TELEMETRIA EM TEMPO REAL NO MYSQL
    # ------------------------------------------------------------------------
    def calcular_saida_populacao(self, entradas: np.ndarray, alive_mask: np.ndarray, leader_idx: int) -> np.ndarray:
        self.last_in[:] = entradas
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            return self.last_out

        inp_alive = entradas[alive_indices]
        hid_pre = np.einsum("ki,kih->kh", inp_alive, self.W1[alive_indices]) + self.B1[alive_indices]
        hid_act = relu_clamped(hid_pre)

        out_pre = np.einsum("kh,kho->ko", hid_act, self.W2[alive_indices]) + self.B2[alive_indices]
        out_act = relu_clamped(out_pre)

        self.last_hid[alive_indices] = hid_act
        self.last_out[alive_indices] = out_act

        # Atualiza sbr_live_input / sbr_live_activation / sbr_live_decision no MySQL a cada 250ms
        now = time.perf_counter()
        if self.mysql_online and (now - self._last_telemetry_ts) >= 0.25:
            self._last_telemetry_ts = now
            self._trigger_async_mysql_forward_pass(leader_idx, entradas[leader_idx].copy())

        return self.last_out

    def _trigger_async_mysql_forward_pass(self, leader_idx: int, leader_inputs: np.ndarray):
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._bg_thread = threading.Thread(
            target=self._exec_mysql_forward_pass_leader,
            args=(leader_idx, leader_inputs),
            daemon=True,
        )
        self._bg_thread.start()

    def _exec_mysql_forward_pass_leader(self, leader_idx: int, leader_inputs: np.ndarray):
        if not self._bg_lock.acquire(blocking=False):
            return
        t0 = time.perf_counter()
        try:
            conn = self._open_conn(use_database=True)
            cur = conn.cursor()
            gid = int(self.genome_ids[leader_idx])
            cur.execute("DELETE FROM sbr_live_input")
            if len(self.in_neuron_ids) == self.n_in:
                params = [(gid, self.in_neuron_ids[pos], float(leader_inputs[pos])) for pos in range(self.n_in)]
                cur.executemany(
                    "INSERT INTO sbr_live_input (genome_id, neuron_id, valor) VALUES (%s, %s, %s)",
                    params,
                )
                cur.execute("CALL sp_sbr_batch_forward_pass(%s)", (ARCHITECTURE_NAME,))
                _consume_all_results(cur)
                conn.commit()
            cur.close()
            conn.close()
            self.mysql_last_latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            pass
        finally:
            self._bg_lock.release()

    # ------------------------------------------------------------------------
    # EVOLUÇÃO GENÉTICA 100% NO MYSQL SERVER (sp_sbr_evolve_generation)
    # ------------------------------------------------------------------------
    def evoluir_geracao(
        self,
        fitness_array: np.ndarray,
        dist_restante_array: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        colidiu: np.ndarray,
        queimado: np.ndarray,
        concluiu_corrida: np.ndarray,
        ticks_conclusao: np.ndarray,
        rider_indices: np.ndarray,
        riders_info: list[dict],
    ) -> bool:
        """
        Grava os resultados reais de todos os 250 cavalos na tabela `sbr_genome` (incluindo
        `concluiu_corrida` e `ticks_conclusao`), identifica o Campeão da Geração APENAS se
        `concluiu_corrida == 1` e executa `CALL sp_sbr_evolve_generation(...)` no MySQL.
        """
        if not self.mysql_online:
            return False

        best_idx = int(np.argmax(fitness_array))
        self.melhor_fitness_geracao = float(fitness_array[best_idx])
        self.media_fitness_geracao = float(np.mean(fitness_array))
        if self.melhor_fitness_geracao > self.distancia_recorde:
            self.distancia_recorde = self.melhor_fitness_geracao

        # Verifica quem realmente concluiu a corrida nesta geração!
        fin_indices = np.where(concluiu_corrida == 1)[0]
        self.qtd_concluiram_geracao_ant = int(len(fin_indices))
        if self.qtd_concluiram_geracao_ant > 0:
            # Entre os que concluíram a corrida, o Campeão é o de maior fitness (menor tempo em ticks)
            best_fin_local = int(np.argmax(fitness_array[fin_indices]))
            champ_slot = int(fin_indices[best_fin_local])
            champ_gid = int(self.genome_ids[champ_slot])
            champ_name = riders_info[int(rider_indices[champ_slot])]["name"]
            champ_ticks = int(ticks_conclusao[champ_slot])
            self.campeao_geracao_anterior_str = (
                f"{champ_name} (#{champ_gid} em {champ_ticks}t) [{self.qtd_concluiram_geracao_ant} concluíram]"
            )
            self.ultimo_campeao_oficial_str = (
                f"G{self.geracao}: {champ_name} (#{champ_gid} em {champ_ticks}t)"
            )
        else:
            self.campeao_geracao_anterior_str = "Nenhum (0 concluíram a corrida)"

        t0 = time.perf_counter()
        try:
            with self._bg_lock:
                conn = self._open_conn(use_database=True)
                cur = conn.cursor()
                update_rows = [
                    (
                        float(fitness_array[i]),
                        int(dist_restante_array[i]),
                        float(xs[i]),
                        float(ys[i]),
                        int(colidiu[i]),
                        int(queimado[i]),
                        int(concluiu_corrida[i]),
                        int(ticks_conclusao[i]) if int(concluiu_corrida[i]) == 1 else None,
                        int(self.genome_ids[i]),
                    )
                    for i in range(self.pop_size)
                ]
                cur.executemany("""
                    UPDATE sbr_genome
                    SET fitness = %s,
                        distancia_restante = %s,
                        pos_x = %s,
                        pos_y = %s,
                        colidiu = %s,
                        eliminado_laser = %s,
                        concluiu_corrida = %s,
                        ticks_conclusao = %s
                    WHERE id = %s
                """, update_rows)
                conn.commit()

                # Executa a evolução genética completa dentro do MySQL!
                cur.execute(
                    "CALL sp_sbr_evolve_generation(%s, %s, %s)",
                    (ARCHITECTURE_NAME, self.geracao, self.elite_step),
                )
                _consume_all_results(cur)
                conn.commit()
                cur.close()
                conn.close()

                self.geracao += 1
                self._load_generation_from_mysql(self.geracao)

            self.mysql_last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.mysql_status_msg = (
                f"MySQL ONLINE ({self.mysql_version}) | DB: {MYSQL_CONFIG['database']} | "
                f"Evolução G{self.geracao}: {self.mysql_last_latency_ms:.0f}ms"
            )
            return True

        except Exception as exc:
            self.mysql_online = False
            self.mysql_status_msg = f"PAUSADO (Erro MySQL): {str(exc)[:50]} (Aperte F5)"
            print(f"[ERRO MYSQL NA EVOLUÇÃO] {exc}")
            return False
