-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 01_schema.sql
-- DESCRIÇÃO: Modelagem completa da Rede Neural (MLP 18 -> 6 -> 4 com ReLU),
--            Genomas (DNA de pesos e biases), Telemetria em Tempo Real e
--            Histórico Evolutivo para execução e estudo no MySQL Workbench.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS sbr_neural_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE sbr_neural_db;

-- ----------------------------------------------------------------------------
-- 1. ESTRUTURA DA REDE NEURAL (Topologia da MLP)
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sbr_architecture (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(64) NOT NULL,
    qtd_escondidas INT NOT NULL DEFAULT 1,
    qtd_entradas INT NOT NULL DEFAULT 18,
    qtd_neuronios_escondida INT NOT NULL DEFAULT 6,
    qtd_saidas INT NOT NULL DEFAULT 4,
    funcao_ativacao ENUM('relu', 'sigmoid', 'tanh') NOT NULL DEFAULT 'relu',
    criado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_sbr_arch_nome (nome)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sbr_layer (
    id INT AUTO_INCREMENT PRIMARY KEY,
    architecture_id INT NOT NULL,
    ordem INT NOT NULL COMMENT '0 = Entrada (18 sensores), 1 = Escondida (6 neurônios), 2 = Saída (4 ações)',
    tipo ENUM('input', 'hidden', 'output') NOT NULL,
    qtd_neuronios INT NOT NULL,
    FOREIGN KEY (architecture_id) REFERENCES sbr_architecture(id) ON DELETE CASCADE,
    UNIQUE KEY uq_sbr_layer_ordem (architecture_id, ordem)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sbr_neuron (
    id INT AUTO_INCREMENT PRIMARY KEY,
    layer_id INT NOT NULL,
    posicao INT NOT NULL COMMENT 'Índice 0..N-1 dentro da camada',
    funcao_ativacao ENUM('linear', 'relu', 'sigmoid', 'tanh') NOT NULL DEFAULT 'relu',
    label VARCHAR(64) NOT NULL COMMENT 'Ex: SENSOR_ANG_-90, VELOCIDADE, HIDDEN_0, GALOPAR, FREAR, VIRAR_ESQ, VIRAR_DIR',
    FOREIGN KEY (layer_id) REFERENCES sbr_layer(id) ON DELETE CASCADE,
    UNIQUE KEY uq_sbr_neuron_pos (layer_id, posicao)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sbr_synapse (
    id INT AUTO_INCREMENT PRIMARY KEY,
    architecture_id INT NOT NULL,
    layer_transition INT NOT NULL COMMENT '0 = Entrada->Escondida, 1 = Escondida->Saída',
    from_neuron_id INT NOT NULL,
    to_neuron_id INT NOT NULL,
    gene_index INT NOT NULL COMMENT 'Índice linear no vetor DNA',
    FOREIGN KEY (architecture_id) REFERENCES sbr_architecture(id) ON DELETE CASCADE,
    FOREIGN KEY (from_neuron_id) REFERENCES sbr_neuron(id) ON DELETE CASCADE,
    FOREIGN KEY (to_neuron_id) REFERENCES sbr_neuron(id) ON DELETE CASCADE,
    UNIQUE KEY uq_sbr_synapse_pair (from_neuron_id, to_neuron_id),
    UNIQUE KEY uq_sbr_synapse_gene (architecture_id, gene_index),
    INDEX idx_sbr_synapse_to (to_neuron_id),
    INDEX idx_sbr_synapse_arch_trans (architecture_id, layer_transition)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 2. GENOMAS (DNA dos Cavalos / Competidores da Steel Ball Run)
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sbr_genome (
    id INT AUTO_INCREMENT PRIMARY KEY,
    architecture_id INT NOT NULL,
    geracao INT NOT NULL DEFAULT 0,
    slot_index INT NOT NULL COMMENT 'Índice do cavalo na bateria (0 .. QTD_CAVALOS-1)',
    rider_name VARCHAR(64) NOT NULL COMMENT 'Gyro Zeppeli, Johnny Joestar, Diego Brando, Pocoloco, Funny Valentine',
    rider_sprite VARCHAR(64) NOT NULL COMMENT 'gyro.png, jonny.png, dio.png, pocoloco.png, valentine.png',
    fitness DOUBLE NOT NULL DEFAULT 0.0 COMMENT 'maiorDistancia - distancias[X][Y] (+ bônus se concluiu)',
    distancia_restante INT NOT NULL DEFAULT 999999,
    pos_x DOUBLE NOT NULL DEFAULT 0.0,
    pos_y DOUBLE NOT NULL DEFAULT 0.0,
    angulo DOUBLE NOT NULL DEFAULT 0.0,
    velocidade DOUBLE NOT NULL DEFAULT 0.0,
    colidiu TINYINT(1) NOT NULL DEFAULT 0,
    eliminado_laser TINYINT(1) NOT NULL DEFAULT 0,
    concluiu_corrida TINYINT(1) NOT NULL DEFAULT 0 COMMENT '1 = Cruzou a linha de chegada completando a volta',
    ticks_conclusao INT NULL DEFAULT NULL COMMENT 'Quantidade de ticks gastos para concluir a corrida (NULL se não concluiu)',
    origem ENUM('aleatorio', 'elite_intacto', 'clone_mutado', 'crossover') NOT NULL DEFAULT 'aleatorio',
    parent_genome_id INT NULL,
    qtd_mutacoes INT NOT NULL DEFAULT 0,
    criado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (architecture_id) REFERENCES sbr_architecture(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_genome_id) REFERENCES sbr_genome(id) ON DELETE SET NULL,
    UNIQUE KEY uq_sbr_genome_gen_slot (architecture_id, geracao, slot_index),
    INDEX idx_sbr_genome_gen_fit (architecture_id, geracao, fitness DESC)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sbr_genome_weight (
    genome_id INT NOT NULL,
    synapse_id INT NOT NULL,
    peso DOUBLE NOT NULL,
    PRIMARY KEY (genome_id, synapse_id),
    FOREIGN KEY (genome_id) REFERENCES sbr_genome(id) ON DELETE CASCADE,
    FOREIGN KEY (synapse_id) REFERENCES sbr_synapse(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sbr_genome_bias (
    genome_id INT NOT NULL,
    neuron_id INT NOT NULL,
    bias DOUBLE NOT NULL COMMENT 'Peso do neurônio BIAS (saída fixa = 1.0)',
    PRIMARY KEY (genome_id, neuron_id),
    FOREIGN KEY (genome_id) REFERENCES sbr_genome(id) ON DELETE CASCADE,
    FOREIGN KEY (neuron_id) REFERENCES sbr_neuron(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 3. HISTÓRICO EVOLUTIVO E TELEMETRIA DE GERAÇÕES
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sbr_generation_summary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    architecture_id INT NOT NULL,
    geracao INT NOT NULL,
    tamanho_populacao INT NOT NULL,
    elite_step INT NOT NULL DEFAULT 5,
    melhor_fitness DOUBLE NOT NULL DEFAULT 0.0,
    media_fitness DOUBLE NOT NULL DEFAULT 0.0,
    pior_fitness DOUBLE NOT NULL DEFAULT 0.0,
    distancia_recorde DOUBLE NOT NULL DEFAULT 0.0,
    range_random DOUBLE NOT NULL DEFAULT 142.0 COMMENT 'Decai * 0.999 por geração até 15.0',
    qtd_concluiram INT NOT NULL DEFAULT 0 COMMENT 'Quantos cavalos cruzaram a linha de chegada nesta geração',
    melhor_genome_id INT NULL COMMENT 'ID do Campeão da Geração (SOMENTE se concluiu_corrida = 1, senão NULL)',
    melhor_rider_name VARCHAR(64) NULL COMMENT 'Nome do Campeão da Geração (SOMENTE se concluiu_corrida = 1)',
    criado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (architecture_id) REFERENCES sbr_architecture(id) ON DELETE CASCADE,
    FOREIGN KEY (melhor_genome_id) REFERENCES sbr_genome(id) ON DELETE SET NULL,
    UNIQUE KEY uq_sbr_gen_summary (architecture_id, geracao)
) ENGINE=InnoDB;

-- ----------------------------------------------------------------------------
-- 4. TABELAS EM MEMÓRIA (ENGINE=MEMORY) PARA FORWARD PASS EM TEMPO REAL
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sbr_live_input (
    genome_id INT NOT NULL,
    neuron_id INT NOT NULL,
    valor DOUBLE NOT NULL,
    PRIMARY KEY (genome_id, neuron_id)
) ENGINE=MEMORY;

CREATE TABLE IF NOT EXISTS sbr_live_activation (
    genome_id INT NOT NULL,
    neuron_id INT NOT NULL,
    layer_ordem INT NOT NULL,
    posicao INT NOT NULL,
    valor DOUBLE NOT NULL,
    PRIMARY KEY (genome_id, neuron_id),
    INDEX idx_sbr_live_act_layer (genome_id, layer_ordem)
) ENGINE=MEMORY;

CREATE TABLE IF NOT EXISTS sbr_live_decision (
    genome_id INT NOT NULL PRIMARY KEY,
    out_galopar DOUBLE NOT NULL DEFAULT 0.0,
    out_frear DOUBLE NOT NULL DEFAULT 0.0,
    out_esquerda DOUBLE NOT NULL DEFAULT 0.0,
    out_direita DOUBLE NOT NULL DEFAULT 0.0,
    acao_principal VARCHAR(32) NOT NULL DEFAULT 'COSTEAR',
    atualizado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=MEMORY;


-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 02_seed_architecture.sql
-- DESCRIÇÃO: Cria a procedure sp_sbr_create_architecture e instancia a
--            topologia padrão sbr_mlp_18_6_4 (18 Sensores -> 6 Ocultos -> 4 Saídas)
--            com função de ativação ReLU truncada [0, 10000].
-- ============================================================================

USE sbr_neural_db;

DELIMITER $$

DROP PROCEDURE IF EXISTS sp_sbr_create_architecture$$

CREATE PROCEDURE sp_sbr_create_architecture(
    IN p_nome VARCHAR(64),
    IN p_qtd_escondidas INT,
    IN p_qtd_entradas INT,
    IN p_qtd_neuronios_escondida INT,
    IN p_qtd_saidas INT
)
BEGIN
    DECLARE v_arch_id INT DEFAULT NULL;
    DECLARE v_layer_in_id INT;
    DECLARE v_layer_hid_id INT;
    DECLARE v_layer_out_id INT;
    DECLARE v_i INT DEFAULT 0;
    DECLARE v_j INT DEFAULT 0;
    DECLARE v_gene_idx INT DEFAULT 0;
    DECLARE v_angulo DOUBLE;
    DECLARE v_label VARCHAR(64);

    SELECT id INTO v_arch_id FROM sbr_architecture WHERE nome = p_nome;

    IF v_arch_id IS NULL THEN
        INSERT INTO sbr_architecture (
            nome, qtd_escondidas, qtd_entradas, qtd_neuronios_escondida, qtd_saidas, funcao_ativacao
        ) VALUES (
            p_nome, p_qtd_escondidas, p_qtd_entradas, p_qtd_neuronios_escondida, p_qtd_saidas, 'relu'
        );
        SET v_arch_id = LAST_INSERT_ID();

        -- 1. Camada de Entrada (Ordem 0)
        INSERT INTO sbr_layer (architecture_id, ordem, tipo, qtd_neuronios)
        VALUES (v_arch_id, 0, 'input', p_qtd_entradas);
        SET v_layer_in_id = LAST_INSERT_ID();

        SET v_i = 0;
        WHILE v_i < p_qtd_entradas DO
            IF v_i < p_qtd_entradas - 1 THEN
                SET v_angulo = -90.0 + (v_i * 180.0 / (p_qtd_entradas - 2));
                SET v_label = CONCAT('SENSOR_', FORMAT(v_angulo, 1), '_DEG');
            ELSE
                SET v_label = 'VELOCIDADE_ATUAL';
            END IF;

            INSERT INTO sbr_neuron (layer_id, posicao, funcao_ativacao, label)
            VALUES (v_layer_in_id, v_i, 'linear', v_label);
            SET v_i = v_i + 1;
        END WHILE;

        -- 2. Camada Escondida (Ordem 1)
        INSERT INTO sbr_layer (architecture_id, ordem, tipo, qtd_neuronios)
        VALUES (v_arch_id, 1, 'hidden', p_qtd_neuronios_escondida);
        SET v_layer_hid_id = LAST_INSERT_ID();

        SET v_i = 0;
        WHILE v_i < p_qtd_neuronios_escondida DO
            INSERT INTO sbr_neuron (layer_id, posicao, funcao_ativacao, label)
            VALUES (v_layer_hid_id, v_i, 'relu', CONCAT('HIDDEN_SPIN_', v_i));
            SET v_i = v_i + 1;
        END WHILE;

        -- 3. Camada de Saída (Ordem 2)
        INSERT INTO sbr_layer (architecture_id, ordem, tipo, qtd_neuronios)
        VALUES (v_arch_id, 2, 'output', p_qtd_saidas);
        SET v_layer_out_id = LAST_INSERT_ID();

        INSERT INTO sbr_neuron (layer_id, posicao, funcao_ativacao, label) VALUES
            (v_layer_out_id, 0, 'relu', 'GALOPAR'),
            (v_layer_out_id, 1, 'relu', 'FREAR'),
            (v_layer_out_id, 2, 'relu', 'VIRAR_ESQUERDA'),
            (v_layer_out_id, 3, 'relu', 'VIRAR_DIREITA');

        -- 4. Sinapses Entrada -> Escondida (18 * 6 = 108 sinapses)
        INSERT INTO sbr_synapse (architecture_id, layer_transition, from_neuron_id, to_neuron_id, gene_index)
        SELECT
            v_arch_id,
            0,
            n_in.id,
            n_hid.id,
            (n_hid.posicao * p_qtd_entradas) + n_in.posicao
        FROM sbr_neuron n_hid
        CROSS JOIN sbr_neuron n_in
        WHERE n_hid.layer_id = v_layer_hid_id
          AND n_in.layer_id = v_layer_in_id
        ORDER BY n_hid.posicao, n_in.posicao;

        -- 5. Sinapses Escondida -> Saída (6 * 4 = 24 sinapses)
        SET v_gene_idx = p_qtd_entradas * p_qtd_neuronios_escondida;

        INSERT INTO sbr_synapse (architecture_id, layer_transition, from_neuron_id, to_neuron_id, gene_index)
        SELECT
            v_arch_id,
            1,
            n_hid.id,
            n_out.id,
            v_gene_idx + (n_out.posicao * p_qtd_neuronios_escondida) + n_hid.posicao
        FROM sbr_neuron n_out
        CROSS JOIN sbr_neuron n_hid
        WHERE n_out.layer_id = v_layer_out_id
          AND n_hid.layer_id = v_layer_hid_id
        ORDER BY n_out.posicao, n_hid.posicao;
    END IF;
END$$

DELIMITER ;

-- Instancia a arquitetura oficial Steel Ball Run MLP [1, 18, 6, 4]
CALL sp_sbr_create_architecture('sbr_mlp_18_6_4', 1, 18, 6, 4);



-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 03_population_init.sql
-- DESCRIÇÃO: Cria a população inicial (Geração 0) no MySQL com pesos e biases
--            sorteados em [-1000.0, +1000.0] (idêntico a getRandomValue() do
--            [-1000, +1000]) e distribui os 5 personagens da Steel Ball Run:
--            Gyro Zeppeli, Johnny Joestar, Diego Brando, Pocoloco e Valentine.
-- ============================================================================

USE sbr_neural_db;

DELIMITER $$

DROP PROCEDURE IF EXISTS sp_sbr_init_population$$

CREATE PROCEDURE sp_sbr_init_population(
    IN p_architecture_name VARCHAR(64),
    IN p_population_size INT,
    IN p_reset_existing TINYINT(1)
)
BEGIN
    DECLARE v_arch_id INT DEFAULT NULL;
    DECLARE v_existing_count INT DEFAULT 0;
    DECLARE v_slot INT DEFAULT 0;
    DECLARE v_genome_id INT;
    DECLARE v_rider_idx INT;
    DECLARE v_rider_name VARCHAR(64);
    DECLARE v_rider_sprite VARCHAR(64);
    DECLARE v_total_genes INT DEFAULT 142;

    SELECT id INTO v_arch_id
    FROM sbr_architecture
    WHERE nome = p_architecture_name;

    IF v_arch_id IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Arquitetura não encontrada em sbr_architecture';
    END IF;

    IF p_population_size < 5 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'A população precisa ter pelo menos 5 cavalos (Step = 5 do elitismo)';
    END IF;

    IF p_reset_existing = 1 THEN
        DELETE FROM sbr_live_input;
        DELETE FROM sbr_live_activation;
        DELETE FROM sbr_live_decision;
        DELETE FROM sbr_generation_summary WHERE architecture_id = v_arch_id;
        DELETE FROM sbr_genome WHERE architecture_id = v_arch_id;
    END IF;

    SELECT COUNT(*) INTO v_existing_count
    FROM sbr_genome
    WHERE architecture_id = v_arch_id AND geracao = 0;

    IF v_existing_count = 0 THEN
        START TRANSACTION;

        SET v_slot = 0;
        WHILE v_slot < p_population_size DO
            SET v_rider_idx = MOD(v_slot, 5);
            CASE v_rider_idx
                WHEN 0 THEN
                    SET v_rider_name = 'Gyro Zeppeli (Valkyrie)';
                    SET v_rider_sprite = 'gyro.png';
                WHEN 1 THEN
                    SET v_rider_name = 'Johnny Joestar (Slow Dancer)';
                    SET v_rider_sprite = 'jonny.png';
                WHEN 2 THEN
                    SET v_rider_name = 'Diego Brando (Silver Bullet)';
                    SET v_rider_sprite = 'dio.png';
                WHEN 3 THEN
                    SET v_rider_name = 'Pocoloco (Hey Ya!)';
                    SET v_rider_sprite = 'pocoloco.png';
                ELSE
                    SET v_rider_name = 'Funny Valentine (D4C)';
                    SET v_rider_sprite = 'valentine.png';
            END CASE;

            INSERT INTO sbr_genome (
                architecture_id, geracao, slot_index, rider_name, rider_sprite,
                fitness, origem
            ) VALUES (
                v_arch_id, 0, v_slot, v_rider_name, v_rider_sprite,
                0.0, 'aleatorio'
            );
            SET v_genome_id = LAST_INSERT_ID();

            -- Pesos das 132 sinapses em [-1000.0, +1000.0]
            INSERT INTO sbr_genome_weight (genome_id, synapse_id, peso)
            SELECT
                v_genome_id,
                s.id,
                ROUND((RAND() * 2000.0) - 1000.0, 4)
            FROM sbr_synapse s
            WHERE s.architecture_id = v_arch_id;

            -- Biases dos 10 neurônios não-entrada (6 ocultos + 4 saídas) em [-1000.0, +1000.0]
            INSERT INTO sbr_genome_bias (genome_id, neuron_id, bias)
            SELECT
                v_genome_id,
                n.id,
                ROUND((RAND() * 2000.0) - 1000.0, 4)
            FROM sbr_neuron n
            JOIN sbr_layer l ON l.id = n.layer_id
            WHERE l.architecture_id = v_arch_id
              AND l.tipo IN ('hidden', 'output');

            SET v_slot = v_slot + 1;
        END WHILE;

        -- Calcula tamanho do DNA (pesos + biases)
        SELECT
            (SELECT COUNT(*) FROM sbr_synapse WHERE architecture_id = v_arch_id) +
            (SELECT COUNT(*) FROM sbr_neuron n JOIN sbr_layer l ON l.id = n.layer_id
             WHERE l.architecture_id = v_arch_id AND l.tipo IN ('hidden', 'output'))
        INTO v_total_genes;

        INSERT INTO sbr_generation_summary (
            architecture_id, geracao, tamanho_populacao, elite_step,
            melhor_fitness, media_fitness, pior_fitness, distancia_recorde, range_random
        ) VALUES (
            v_arch_id, 0, p_population_size, 5,
            0.0, 0.0, 0.0, 0.0, v_total_genes
        );

        COMMIT;
    END IF;
END$$

DELIMITER ;



-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 04_neural_inference.sql
-- DESCRIÇÃO: Stored Procedures que executam o Forward Pass (Propagação Direta)
--            da Rede Neural (Camada de Entrada -> Escondida -> Saída) usando
--            a função ReLU truncada [0, 10000] idêntica a RNA_CalcularSaida()
--            da rede neural MLP [1, 18, 6, 4] da Steel Ball Run.
-- ============================================================================

USE sbr_neural_db;

DELIMITER $$

DROP PROCEDURE IF EXISTS sp_sbr_batch_forward_pass$$

CREATE PROCEDURE sp_sbr_batch_forward_pass(
    IN p_architecture_name VARCHAR(64)
)
BEGIN
    DECLARE v_arch_id INT DEFAULT NULL;
    DECLARE v_layer_in_id INT;
    DECLARE v_layer_hid_id INT;
    DECLARE v_layer_out_id INT;

    SELECT id INTO v_arch_id
    FROM sbr_architecture
    WHERE nome = p_architecture_name;

    IF v_arch_id IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Arquitetura não encontrada para inferência';
    END IF;

    SELECT id INTO v_layer_in_id FROM sbr_layer WHERE architecture_id = v_arch_id AND ordem = 0;
    SELECT id INTO v_layer_hid_id FROM sbr_layer WHERE architecture_id = v_arch_id AND ordem = 1;
    SELECT id INTO v_layer_out_id FROM sbr_layer WHERE architecture_id = v_arch_id AND ordem = 2;

    -- Limpa ativações e decisões do tick anterior na memória
    DELETE FROM sbr_live_activation;
    DELETE FROM sbr_live_decision;

    -- 1. Registra ativações da camada de entrada (18 sensores)
    INSERT INTO sbr_live_activation (genome_id, neuron_id, layer_ordem, posicao, valor)
    SELECT
        inp.genome_id,
        inp.neuron_id,
        0,
        n.posicao,
        inp.valor
    FROM sbr_live_input inp
    JOIN sbr_neuron n ON n.id = inp.neuron_id
    WHERE n.layer_id = v_layer_in_id;

    -- 2. Forward Pass: Camada de Entrada (0) -> Camada Escondida (1) com ReLU [0, 10000]
    -- Somatorio = SUM(Entrada[j] * Peso[i,j]) + Bias[i] * 1.0
    INSERT INTO sbr_live_activation (genome_id, neuron_id, layer_ordem, posicao, valor)
    SELECT
        act_in.genome_id,
        n_hid.id,
        1,
        n_hid.posicao,
        LEAST(10000.0, GREATEST(0.0,
            SUM(act_in.valor * w.peso) + COALESCE(b.bias, 0.0)
        )) AS saida_relu
    FROM sbr_live_activation act_in
    JOIN sbr_synapse s
      ON s.from_neuron_id = act_in.neuron_id
     AND s.architecture_id = v_arch_id
     AND s.layer_transition = 0
    JOIN sbr_neuron n_hid
      ON n_hid.id = s.to_neuron_id
     AND n_hid.layer_id = v_layer_hid_id
    JOIN sbr_genome_weight w
      ON w.genome_id = act_in.genome_id
     AND w.synapse_id = s.id
    LEFT JOIN sbr_genome_bias b
      ON b.genome_id = act_in.genome_id
     AND b.neuron_id = n_hid.id
    WHERE act_in.layer_ordem = 0
    GROUP BY act_in.genome_id, n_hid.id, n_hid.posicao, b.bias;

    -- 3. Forward Pass: Camada Escondida (1) -> Camada de Saída (2) com ReLU [0, 10000]
    INSERT INTO sbr_live_activation (genome_id, neuron_id, layer_ordem, posicao, valor)
    SELECT
        act_hid.genome_id,
        n_out.id,
        2,
        n_out.posicao,
        LEAST(10000.0, GREATEST(0.0,
            SUM(act_hid.valor * w.peso) + COALESCE(b.bias, 0.0)
        )) AS saida_relu
    FROM sbr_live_activation act_hid
    JOIN sbr_synapse s
      ON s.from_neuron_id = act_hid.neuron_id
     AND s.architecture_id = v_arch_id
     AND s.layer_transition = 1
    JOIN sbr_neuron n_out
      ON n_out.id = s.to_neuron_id
     AND n_out.layer_id = v_layer_out_id
    JOIN sbr_genome_weight w
      ON w.genome_id = act_hid.genome_id
     AND w.synapse_id = s.id
    LEFT JOIN sbr_genome_bias b
      ON b.genome_id = act_hid.genome_id
     AND b.neuron_id = n_out.id
    WHERE act_hid.layer_ordem = 1
    GROUP BY act_hid.genome_id, n_out.id, n_out.posicao, b.bias;

    -- 4. Consolida as 4 saídas de decisão por cavalo
    INSERT INTO sbr_live_decision (
        genome_id, out_galopar, out_frear, out_esquerda, out_direita, acao_principal
    )
    SELECT
        act.genome_id,
        MAX(CASE WHEN act.posicao = 0 THEN act.valor ELSE 0.0 END) AS galopar,
        MAX(CASE WHEN act.posicao = 1 THEN act.valor ELSE 0.0 END) AS frear,
        MAX(CASE WHEN act.posicao = 2 THEN act.valor ELSE 0.0 END) AS esquerda,
        MAX(CASE WHEN act.posicao = 3 THEN act.valor ELSE 0.0 END) AS direita,
        CONCAT_WS('+',
            CASE WHEN MAX(CASE WHEN act.posicao = 0 THEN act.valor ELSE 0.0 END) > 0 THEN 'GALOPAR' END,
            CASE WHEN MAX(CASE WHEN act.posicao = 1 THEN act.valor ELSE 0.0 END) > 0 THEN 'FREAR' END,
            CASE WHEN MAX(CASE WHEN act.posicao = 2 THEN act.valor ELSE 0.0 END) > 0 THEN 'ESQ' END,
            CASE WHEN MAX(CASE WHEN act.posicao = 3 THEN act.valor ELSE 0.0 END) > 0 THEN 'DIR' END
        )
    FROM sbr_live_activation act
    WHERE act.layer_ordem = 2
    GROUP BY act.genome_id;

    -- Retorna todas as decisões calculadas para o simulador aplicar na física
    SELECT
        genome_id,
        out_galopar,
        out_frear,
        out_esquerda,
        out_direita,
        acao_principal
    FROM sbr_live_decision
    ORDER BY genome_id;
END$$

DELIMITER ;



-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 05_genetic_evolution.sql
-- DESCRIÇÃO: Implementa o algoritmo genético neuroevolutivo da Steel Ball Run
--            diretamente em SQL dentro do MySQL:
--            1. Ordena todos os cavalos da geração pelo Fitness (maiorDistancia - dist).
--            2. Registra o Campeão da Geração APENAS se `concluiu_corrida = 1`.
--            3. Seleciona a Elite (Top Step = 5 melhores cavalos).
--            4. Clona os Top 5 em round-robin para toda a nova geração (G + 1).
--            5. Nos clones (slots >= Step), aplica entre 1 e RangeRandom mutações
--               usando os 3 operadores estocásticos de mutação:
--               - Tipo 0: Reset aleatório em [-1000, +1000]
--               - Tipo 1: Multiplicação aleatória por fator em [0.5, 1.5]
--               - Tipo 2: Soma aleatória fina em [-10.0, +10.0]
--            6. Atualiza RangeRandom = GREATEST(15.0, RangeRandom * 0.999).
-- ============================================================================

USE sbr_neural_db;

DELIMITER $$

DROP PROCEDURE IF EXISTS sp_sbr_evolve_generation$$

CREATE PROCEDURE sp_sbr_evolve_generation(
    IN p_architecture_name VARCHAR(64),
    IN p_source_generation INT,
    IN p_elite_step INT
)
BEGIN
    DECLARE v_arch_id INT DEFAULT NULL;
    DECLARE v_pop_size INT DEFAULT 0;
    DECLARE v_range_random DOUBLE DEFAULT 142.0;
    DECLARE v_next_range_random DOUBLE DEFAULT 142.0;
    DECLARE v_total_genes INT DEFAULT 142;
    DECLARE v_best_fitness DOUBLE DEFAULT 0.0;
    DECLARE v_avg_fitness DOUBLE DEFAULT 0.0;
    DECLARE v_worst_fitness DOUBLE DEFAULT 0.0;
    DECLARE v_record_fitness DOUBLE DEFAULT 0.0;
    DECLARE v_qtd_concluiram INT DEFAULT 0;
    DECLARE v_best_genome_id INT DEFAULT NULL;
    DECLARE v_best_rider VARCHAR(64) DEFAULT 'Nenhum (Não Concluiu)';

    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        DROP TEMPORARY TABLE IF EXISTS tmp_sbr_ranked;
        DROP TEMPORARY TABLE IF EXISTS tmp_sbr_elite;
        ROLLBACK;
        RESIGNAL;
    END;

    SELECT id INTO v_arch_id
    FROM sbr_architecture
    WHERE nome = p_architecture_name;

    IF v_arch_id IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Arquitetura não encontrada para evolução genética';
    END IF;

    IF p_elite_step IS NULL OR p_elite_step < 1 THEN
        SET p_elite_step = 5;
    END IF;

    SELECT COUNT(*) INTO v_pop_size
    FROM sbr_genome
    WHERE architecture_id = v_arch_id AND geracao = p_source_generation;

    IF v_pop_size < p_elite_step THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'População da geração menor que o número de elites (Step)';
    END IF;

    -- Remove geração destino caso já exista
    DELETE FROM sbr_genome
    WHERE architecture_id = v_arch_id AND geracao = p_source_generation + 1;
    DELETE FROM sbr_generation_summary
    WHERE architecture_id = v_arch_id AND geracao = p_source_generation + 1;

    -- Calcula estatísticas da geração atual e quantos concluíram a corrida
    SELECT
        MAX(fitness),
        AVG(fitness),
        MIN(fitness),
        COALESCE(SUM(CASE WHEN concluiu_corrida = 1 THEN 1 ELSE 0 END), 0)
    INTO v_best_fitness, v_avg_fitness, v_worst_fitness, v_qtd_concluiram
    FROM sbr_genome
    WHERE architecture_id = v_arch_id AND geracao = p_source_generation;

    -- SÓ declara Campeão da Geração se pelo menos 1 cavalo completou a corrida (concluiu_corrida = 1)!
    IF v_qtd_concluiram > 0 THEN
        SELECT id, rider_name
        INTO v_best_genome_id, v_best_rider
        FROM sbr_genome
        WHERE architecture_id = v_arch_id
          AND geracao = p_source_generation
          AND concluiu_corrida = 1
        ORDER BY fitness DESC, COALESCE(ticks_conclusao, 999999) ASC, id ASC
        LIMIT 1;
    ELSE
        SET v_best_genome_id = NULL;
        SET v_best_rider = 'Nenhum (Não Concluiu)';
    END IF;

    SELECT COALESCE(MAX(distancia_recorde), 0.0)
    INTO v_record_fitness
    FROM sbr_generation_summary
    WHERE architecture_id = v_arch_id;

    IF v_best_fitness > v_record_fitness THEN
        SET v_record_fitness = v_best_fitness;
    END IF;

    SELECT
        (SELECT COUNT(*) FROM sbr_synapse WHERE architecture_id = v_arch_id) +
        (SELECT COUNT(*) FROM sbr_neuron n JOIN sbr_layer l ON l.id = n.layer_id
         WHERE l.architecture_id = v_arch_id AND l.tipo IN ('hidden', 'output'))
    INTO v_total_genes;

    SELECT COALESCE(range_random, v_total_genes)
    INTO v_range_random
    FROM sbr_generation_summary
    WHERE architecture_id = v_arch_id AND geracao = p_source_generation;

    START TRANSACTION;

    -- Atualiza resumo da geração que acabou de correr
    INSERT INTO sbr_generation_summary (
        architecture_id, geracao, tamanho_populacao, elite_step,
        melhor_fitness, media_fitness, pior_fitness, distancia_recorde,
        range_random, qtd_concluiram, melhor_genome_id, melhor_rider_name
    ) VALUES (
        v_arch_id, p_source_generation, v_pop_size, p_elite_step,
        v_best_fitness, v_avg_fitness, v_worst_fitness, v_record_fitness,
        v_range_random, v_qtd_concluiram, v_best_genome_id, v_best_rider
    )
    ON DUPLICATE KEY UPDATE
        melhor_fitness = VALUES(melhor_fitness),
        media_fitness = VALUES(media_fitness),
        pior_fitness = VALUES(pior_fitness),
        distancia_recorde = VALUES(distancia_recorde),
        qtd_concluiram = VALUES(qtd_concluiram),
        melhor_genome_id = VALUES(melhor_genome_id),
        melhor_rider_name = VALUES(melhor_rider_name);

    -- Ranking 0..N-1 ordenado por Fitness DESC
    DROP TEMPORARY TABLE IF EXISTS tmp_sbr_ranked;
    DROP TEMPORARY TABLE IF EXISTS tmp_sbr_elite;

    CREATE TEMPORARY TABLE tmp_sbr_ranked (
        rank_idx INT PRIMARY KEY,
        old_genome_id INT NOT NULL,
        rider_name VARCHAR(64) NOT NULL,
        rider_sprite VARCHAR(64) NOT NULL,
        fitness DOUBLE NOT NULL
    ) ENGINE=MEMORY;

    CREATE TEMPORARY TABLE tmp_sbr_elite (
        rank_idx INT PRIMARY KEY,
        old_genome_id INT NOT NULL,
        rider_name VARCHAR(64) NOT NULL,
        rider_sprite VARCHAR(64) NOT NULL,
        fitness DOUBLE NOT NULL
    ) ENGINE=MEMORY;

    INSERT INTO tmp_sbr_ranked (rank_idx, old_genome_id, rider_name, rider_sprite, fitness)
    SELECT
        ROW_NUMBER() OVER (ORDER BY g.fitness DESC, g.id ASC) - 1 AS r_idx,
        g.id,
        g.rider_name,
        g.rider_sprite,
        g.fitness
    FROM sbr_genome g
    WHERE g.architecture_id = v_arch_id AND g.geracao = p_source_generation;

    INSERT INTO tmp_sbr_elite (rank_idx, old_genome_id, rider_name, rider_sprite, fitness)
    SELECT rank_idx, old_genome_id, rider_name, rider_sprite, fitness
    FROM tmp_sbr_ranked
    WHERE rank_idx < p_elite_step;

    -- Cria os indivíduos da nova geração (p_source_generation + 1)
    INSERT INTO sbr_genome (
        architecture_id, geracao, slot_index, rider_name, rider_sprite,
        fitness, origem, parent_genome_id, qtd_mutacoes
    )
    SELECT
        v_arch_id,
        p_source_generation + 1,
        r.rank_idx AS novo_slot,
        elite.rider_name,
        elite.rider_sprite,
        0.0,
        CASE WHEN r.rank_idx < p_elite_step THEN 'elite_intacto' ELSE 'clone_mutado' END,
        elite.old_genome_id,
        CASE WHEN r.rank_idx < p_elite_step THEN 0 ELSE FLOOR(1 + (RAND() * v_range_random)) END
    FROM tmp_sbr_ranked r
    JOIN tmp_sbr_elite elite
      ON elite.rank_idx = MOD(r.rank_idx, p_elite_step)
    ORDER BY r.rank_idx;

    -- Clona e muta os pesos sinápticos (Set-Based para toda a população de uma vez!)
    INSERT INTO sbr_genome_weight (genome_id, synapse_id, peso)
    SELECT
        child.id AS genome_id,
        pw.synapse_id,
        CASE
            WHEN child.slot_index < p_elite_step THEN pw.peso
            WHEN RAND() < (child.qtd_mutacoes / v_total_genes) THEN
                CASE FLOOR(RAND() * 3)
                    WHEN 0 THEN ROUND((RAND() * 2000.0) - 1000.0, 4)
                    WHEN 1 THEN ROUND(pw.peso * ((RAND() * 1.0) + 0.5), 4)
                    ELSE ROUND(pw.peso + ((RAND() * 20.0) - 10.0), 4)
                END
            ELSE pw.peso
        END AS novo_peso
    FROM sbr_genome child
    JOIN sbr_genome_weight pw ON pw.genome_id = child.parent_genome_id
    WHERE child.architecture_id = v_arch_id
      AND child.geracao = p_source_generation + 1;

    -- Clona e muta os biases com os 3 operadores estocásticos de mutação
    INSERT INTO sbr_genome_bias (genome_id, neuron_id, bias)
    SELECT
        child.id AS genome_id,
        pb.neuron_id,
        CASE
            WHEN child.slot_index < p_elite_step THEN pb.bias
            WHEN RAND() < (child.qtd_mutacoes / v_total_genes) THEN
                CASE FLOOR(RAND() * 3)
                    WHEN 0 THEN ROUND((RAND() * 2000.0) - 1000.0, 4)
                    WHEN 1 THEN ROUND(pb.bias * ((RAND() * 1.0) + 0.5), 4)
                    ELSE ROUND(pb.bias + ((RAND() * 20.0) - 10.0), 4)
                END
            ELSE pb.bias
        END AS novo_bias
    FROM sbr_genome child
    JOIN sbr_genome_bias pb ON pb.genome_id = child.parent_genome_id
    WHERE child.architecture_id = v_arch_id
      AND child.geracao = p_source_generation + 1;

    -- Limpa os pesos das gerações antigas para manter o banco leve e rápido,
    -- preservando os pesos de todos os Campeões de Geração (concluiu_corrida = 1) e da geração atual!
    IF p_source_generation >= 2 THEN
        DELETE w FROM sbr_genome_weight w
        JOIN sbr_genome g ON g.id = w.genome_id
        WHERE g.architecture_id = v_arch_id
          AND g.geracao < p_source_generation
          AND g.concluiu_corrida = 0
          AND g.id NOT IN (
              SELECT melhor_genome_id FROM sbr_generation_summary
              WHERE architecture_id = v_arch_id AND melhor_genome_id IS NOT NULL
          );
        DELETE b FROM sbr_genome_bias b
        JOIN sbr_genome g ON g.id = b.genome_id
        WHERE g.architecture_id = v_arch_id
          AND g.geracao < p_source_generation
          AND g.concluiu_corrida = 0
          AND g.id NOT IN (
              SELECT melhor_genome_id FROM sbr_generation_summary
              WHERE architecture_id = v_arch_id AND melhor_genome_id IS NOT NULL
          );
    END IF;

    -- Decaimento do RangeRandom = GREATEST(15.0, RangeRandom * 0.999)
    SET v_next_range_random = GREATEST(15.0, v_range_random * 0.999);

    INSERT INTO sbr_generation_summary (
        architecture_id, geracao, tamanho_populacao, elite_step,
        melhor_fitness, media_fitness, pior_fitness, distancia_recorde, range_random,
        qtd_concluiram, melhor_genome_id, melhor_rider_name
    ) VALUES (
        v_arch_id, p_source_generation + 1, v_pop_size, p_elite_step,
        0.0, 0.0, 0.0, v_record_fitness, v_next_range_random,
        0, NULL, 'Em andamento...'
    );

    DROP TEMPORARY TABLE IF EXISTS tmp_sbr_ranked;
    DROP TEMPORARY TABLE IF EXISTS tmp_sbr_elite;
    COMMIT;
END$$

DELIMITER ;


-- ============================================================================
-- PROJETO: Steel Ball Run - Neural Network Simulator (MySQL Workbench Edition)
-- ARQUIVO: 06_workbench_views.sql
-- DESCRIÇÃO: Views e Consultas Prontas para Estudo e Inspeção da Rede Neural
--            e da Evolução Genética diretamente no MySQL Workbench.
-- ============================================================================

USE sbr_neural_db;

-- 1. View de Evolução das Gerações (Campeão da Geração contado APENAS se concluiu a corrida)
CREATE OR REPLACE VIEW vw_sbr_generation_evolution AS
SELECT
    s.geracao AS geracao,
    s.tamanho_populacao AS populacao,
    ROUND(s.melhor_fitness, 2) AS melhor_fitness_px,
    ROUND(s.media_fitness, 2) AS media_fitness_px,
    ROUND(s.distancia_recorde, 2) AS recorde_historico_px,
    LEAST(100.0, ROUND((s.melhor_fitness / 6063.0) * 100.0, 1)) AS progresso_pista_pct,
    s.qtd_concluiram AS concluiram_corrida,
    s.melhor_genome_id AS campeao_genome_id,
    COALESCE(s.melhor_rider_name, 'Nenhum (Não Concluiu)') AS campeao_da_geracao,
    ROUND(s.range_random, 2) AS range_mutacao_atual,
    s.criado_em
FROM sbr_generation_summary s
ORDER BY s.geracao DESC;

-- 2. View Exclusiva dos Campeões de Geração (Apenas quem realmente concluiu a corrida!)
CREATE OR REPLACE VIEW vw_sbr_generation_champions AS
SELECT
    s.geracao AS geracao,
    s.melhor_genome_id AS campeao_genome_id,
    s.melhor_rider_name AS campeao_cavaleiro,
    g.slot_index AS numero_cavalo,
    g.ticks_conclusao AS tempo_ticks,
    ROUND(s.melhor_fitness, 2) AS fitness_com_bonus,
    s.qtd_concluiram AS total_finalistas_geracao,
    s.criado_em
FROM sbr_generation_summary s
JOIN sbr_genome g ON g.id = s.melhor_genome_id
WHERE s.qtd_concluiram > 0 AND s.melhor_genome_id IS NOT NULL
ORDER BY s.geracao DESC;

-- 3. View do Placar (Leaderboard) de Todos os Cavalos da Última Geração
CREATE OR REPLACE VIEW vw_sbr_current_leaderboard AS
SELECT
    g.geracao,
    g.slot_index AS numero_cavalo,
    g.id AS genome_id,
    g.rider_name AS cavaleiro,
    ROUND(g.fitness, 1) AS fitness_distancia,
    LEAST(100.0, ROUND((g.fitness / 6063.0) * 100.0, 1)) AS progresso_volta_pct,
    g.concluiu_corrida,
    g.ticks_conclusao AS tempo_ticks,
    g.origem AS tipo_genetico,
    g.parent_genome_id AS id_pai_elite,
    g.qtd_mutacoes AS genes_mutados,
    CASE
        WHEN g.concluiu_corrida = 1 THEN 'CONCLUIU A CORRIDA (FINALISTA)'
        WHEN g.colidiu = 0 THEN 'CORRENDO (VIVO)'
        WHEN g.eliminado_laser = 1 THEN 'ELIMINADO (TEMPESTADE)'
        ELSE 'COLIDIU (PAREDE/OBSTÁCULO)'
    END AS status_corrida
FROM sbr_genome g
WHERE g.geracao = (SELECT MAX(geracao) FROM sbr_genome)
ORDER BY g.concluiu_corrida DESC, g.fitness DESC, g.slot_index ASC;

-- 4. View de Inspeção dos Pesos Sinápticos e Biases do Melhor Campeão que Concluiu a Corrida
CREATE OR REPLACE VIEW vw_sbr_champion_brain AS
SELECT
    g.geracao,
    g.id AS genome_id,
    g.rider_name AS cavaleiro,
    g.concluiu_corrida,
    g.ticks_conclusao,
    ROUND(g.fitness, 1) AS fitness,
    CASE s.layer_transition
        WHEN 0 THEN '0: ENTRADA (Sensores) -> ESCONDIDA (Spin)'
        ELSE '1: ESCONDIDA (Spin) -> SAÍDA (Ações)'
    END AS camada_conexao,
    n_from.label AS neuronio_origem,
    n_to.label AS neuronio_destino,
    ROUND(w.peso, 4) AS peso_sinapse,
    ROUND(COALESCE(b.bias, 0.0), 4) AS bias_neuronio_destino
FROM sbr_genome g
JOIN sbr_genome_weight w ON w.genome_id = g.id
JOIN sbr_synapse s ON s.id = w.synapse_id
JOIN sbr_neuron n_from ON n_from.id = s.from_neuron_id
JOIN sbr_neuron n_to ON n_to.id = s.to_neuron_id
LEFT JOIN sbr_genome_bias b ON b.genome_id = g.id AND b.neuron_id = n_to.id
WHERE g.id = (
    SELECT id FROM sbr_genome
    ORDER BY concluiu_corrida DESC, fitness DESC, geracao DESC, id ASC
    LIMIT 1
)
ORDER BY s.layer_transition, n_to.posicao, n_from.posicao;

-- 5. View das Ativações em Tempo Real (Forward Pass no MySQL)
CREATE OR REPLACE VIEW vw_sbr_live_brain_activations AS
SELECT
    a.genome_id,
    g.rider_name AS cavaleiro,
    CASE a.layer_ordem
        WHEN 0 THEN '0_ENTRADA_SENSOR'
        WHEN 1 THEN '1_OCULTA_SPIN'
        WHEN 2 THEN '2_SAIDA_ACAO'
    END AS camada,
    a.posicao,
    n.label AS neuronio,
    ROUND(a.valor, 3) AS ativacao_relu
FROM sbr_live_activation a
JOIN sbr_neuron n ON n.id = a.neuron_id
JOIN sbr_genome g ON g.id = a.genome_id
ORDER BY a.genome_id, a.layer_ordem, a.posicao;
