# -*- coding: utf-8 -*-
"""
================================================================================
STEEL BALL RUN — NEURAL NETWORK SIMULATOR (MySQL Workbench + Pygame-CE)
================================================================================
Simulador Neuroevolutivo de Corrida de Cavalos ambientado em
JoJo's Bizarre Adventure: Steel Ball Run.

- Múltiplos Mapas com Troca Instantânea (Tecla [TAB], teclas [1/2/3] ou clique no HUD)
  + Suporte Automático a novos mapas na pasta `Sprites/mapas_custom/`
- Campeão da Geração contabilizado EXCLUSIVAMENTE se concluir 100% da corrida
- Matriz Geodésica de Distâncias (Flood-Fill BFS 8-vizinhos)
- Matriz de Colisão dos Personagens: `Sprites/processed/character_collision_matrices.npz`
  (Gyro Zeppeli, Johnny Joestar, Diego Brando, Pocoloco e Funny Valentine)
- Rede Neural MLP [1, 18, 6, 4] com ReLU truncada e Evolução Genética 100% persistida
  e processada no MySQL Server (`database/00_INSTALAR_TUDO_WORKBENCH.sql`).
================================================================================
"""

import math
import os
import sys
import numpy as np
import pygame

from config import (
    PROCESSED_SPRITES_DIR,
    LARGURA_TELA,
    ALTURA_TELA,
    FPS_ALVO,
    QTD_CAVALOS_PADRAO,
    SBR_BRAIN_QTD_INPUT,
    SBR_BRAIN_QTD_HIDE,
    SBR_BRAIN_QTD_OUTPUT,
    START_X,
    START_Y,
    START_ANGLE,
    RIDERS_INFO,
)
from gerador_mapas import obter_catalogo_mapas, gerar_obstaculos_e_zonas_por_distancias
from rede_neural_mysql import NeuralMySQLManager


class SteelBallRunSimulator:
    def __init__(self, qtd_cavalos: int = QTD_CAVALOS_PADRAO):
        pygame.init()
        pygame.display.set_caption("Steel Ball Run — Neural Network Simulator (MySQL Workbench Edition)")
        self.screen = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA), pygame.RESIZABLE)
        self.win_w, self.win_h = LARGURA_TELA, ALTURA_TELA
        self.clock = pygame.time.Clock()

        # Fontes
        self.font_sm = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_md = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_lg = pygame.font.SysFont("Segoe UI", 19, bold=True)
        self.font_dist = pygame.font.SysFont("consolas", 11, bold=True)

        # 1. Catálogo de Mapas e Estado de Zonas/Obstáculos
        self.catalogo_mapas = obter_catalogo_mapas()
        self.mapa_atual_idx = 0
        self.obstaculos_ativos = True
        self.start_x = START_X
        self.start_y = START_Y
        self.start_angle = START_ANGLE
        self._btn_trocar_mapa_rect = pygame.Rect(0, 0, 0, 0)

        # Cache de superfície do mapa escalado pelo zoom para renderização a 60 FPS
        self._cached_zoom = None
        self._cached_map_surf = None
        self._cached_mask_surf = None

        # Carrega o Mapa Inicial (Stage 1: Cânion do Arizona)
        self._carregar_mapa_por_indice(0, reiniciar_partida=False)

        # 2. Carrega Sprites dos 5 Personagens e suas Matrizes de Colisão
        self._carregar_sprites_e_mascaras_personagens()

        # 3. Câmera (Começa enquadrando o mapa inteiro na tela)
        self.camera_zoom = min(self.win_w / float(self.map_w), self.win_h / float(self.map_h)) * 0.95
        self.camera_x = self.map_w / 2.0
        self.camera_y = self.map_h / 2.0
        self.seguir_lider = False
        self.mostrar_sensores = True
        self.mostrar_mascara_colisao = False
        self.mostrar_matriz_distancias = False
        self.mostrar_hud_rede = True
        self.passos_por_frame = 2  # Velocidade da simulação (ajustável com N e M: 1x, 2x, 4x, 8x, 16x)

        # 4. Aloca População e Gerenciador Neural MySQL
        self.qtd_cavalos = qtd_cavalos
        self.neural_db = NeuralMySQLManager(self.qtd_cavalos)
        self.lista_explosoes = []  # Efeitos visuais quando um cavalo colide ou é eliminado

        self._alocar_arrays_populacao()
        self.inicializar_nova_partida()

    # ========================================================================
    # 1. GERENCIAMENTO DE MÚLTIPLOS MAPAS, MATRIZ DE COLISÃO E FLOOD-FILL
    # ========================================================================
    def _carregar_mapa_por_indice(self, idx: int, reiniciar_partida: bool = True):
        """Troca instantaneamente para o mapa de índice `idx` do catálogo."""
        self.catalogo_mapas = obter_catalogo_mapas()
        self.mapa_atual_idx = idx % len(self.catalogo_mapas)
        info_mapa = self.catalogo_mapas[self.mapa_atual_idx]

        self.nome_mapa_atual = info_mapa["nome"]
        self.start_x = float(info_mapa["start_x"])
        self.start_y = float(info_mapa["start_y"])
        self.start_angle = float(info_mapa["start_angle"])

        mapa_path = info_mapa["mapa_file"]
        mascara_path = info_mapa["mascara_file"]
        npz_path = info_mapa["npz_file"]

        self.mapa_surface = pygame.image.load(mapa_path).convert()
        self.map_w, self.map_h = self.mapa_surface.get_size()

        data = np.load(npz_path)
        self.cenario_base = data["cenario"].astype(np.uint8)
        self.distancias = data["distancias"].astype(np.int32)
        self.maior_distancia = int(data["maior_distancia"])
        self.cenario = self.cenario_base.copy()

        if os.path.exists(mascara_path):
            self.mascara_surface = pygame.image.load(mascara_path).convert()
        else:
            rgb_mask = np.stack([self.cenario_base * 255] * 3, axis=-1)
            self.mascara_surface = pygame.surfarray.make_surface(np.transpose(rgb_mask, (1, 0, 2)))

        # Invalida o cache de zoom para renderizar a superfície do novo mapa e centraliza câmera
        self._cached_zoom = None
        if hasattr(self, "win_w") and hasattr(self, "win_h"):
            self.camera_zoom = min(self.win_w / float(self.map_w), self.win_h / float(self.map_h)) * 0.95
            self.camera_x = self.map_w / 2.0
            self.camera_y = self.map_h / 2.0

        # Configura Zonas e Obstáculos para o mapa selecionado
        self._inicializar_zonas_e_obstaculos(auto_mode=bool(info_mapa.get("auto_obstacles", False)))

        print(
            f"[Mapa SBR ({self.mapa_atual_idx + 1}/{len(self.catalogo_mapas)})] {self.nome_mapa_atual} | "
            f"{self.map_w}x{self.map_h} | Maior Distância Flood-Fill: {self.maior_distancia} px"
        )

        if reiniciar_partida and hasattr(self, "X"):
            self.inicializar_nova_partida()

    def trocar_proximo_mapa(self):
        """Alterna para o próximo mapa disponível (Tecla TAB ou clique no HUD)."""
        prox = (self.mapa_atual_idx + 1) % len(self.catalogo_mapas)
        self._carregar_mapa_por_indice(prox, reiniciar_partida=True)

    # ========================================================================
    # 2. CARREGAMENTO DOS PERSONAGENS E SUAS MATRIZES DE COLISÃO
    # ========================================================================
    def _carregar_sprites_e_mascaras_personagens(self):
        npz_chars = os.path.join(PROCESSED_SPRITES_DIR, "character_collision_matrices.npz")
        char_data = np.load(npz_chars) if os.path.exists(npz_chars) else None

        self.rider_surfaces = []
        self.rider_fallen_surfaces = []
        self.rider_mask_surfaces = []
        self.rider_perimeters = []  # Pontos (x, y) do contorno da matriz de colisão de cada personagem

        for info in RIDERS_INFO:
            key = info["key"]
            race_png = os.path.join(PROCESSED_SPRITES_DIR, f"{key}_race.png")
            mask_png = os.path.join(PROCESSED_SPRITES_DIR, f"{key}_mask.png")

            surf = pygame.image.load(race_png).convert_alpha()
            msurf = pygame.image.load(mask_png).convert_alpha()
            colored_mask = msurf.copy()
            colored_mask.fill((*info["color"], 220), special_flags=pygame.BLEND_RGBA_MULT)

            fallen = surf.copy()
            fallen.fill((90, 75, 65, 160), special_flags=pygame.BLEND_RGBA_MULT)

            self.rider_surfaces.append(surf)
            self.rider_fallen_surfaces.append(fallen)
            self.rider_mask_surfaces.append(colored_mask)

            if char_data is not None and f"{key}_perimeter" in char_data:
                perim = char_data[f"{key}_perimeter"].astype(np.float64)
            else:
                perim = np.array([[0.0, 0.0], [18.0, 0.0], [-18.0, 0.0]], dtype=np.float64)
            self.rider_perimeters.append(perim)

        # Pré-calcula os 17 ângulos relativos dos sensores Raycast (-90° a +90°, passos de 11.25°)
        self.sensor_rel_angles_deg = np.array(
            [-90.0 + (i * 180.0 / (SBR_BRAIN_QTD_INPUT - 2)) for i in range(SBR_BRAIN_QTD_INPUT - 1)],
            dtype=np.float64,
        )
        self.ray_steps_coarse = np.arange(4.0, 360.0, 4.0, dtype=np.float64)

    # ========================================================================
    # 3. ZONAS DE TERRENO (TURBO / AREIA MOVEDIÇA) E OBSTÁCULOS GIRATÓRIOS
    # ========================================================================
    def _inicializar_zonas_e_obstaculos(self, auto_mode: bool = False):
        """
        Configura os modificadores de pista e obstáculos dinâmicos da Steel Ball Run:
        - No Stage 1 usa as coordenadas calibradas do Cânion do Arizona;
        - Nos demais Stages (e em mapas customizados) posiciona automaticamente via Flood-Fill.
        """
        if not auto_mode:
            self.zonas = [
                (1020, 365, 50, +0.45, "Golden Spin Boost"),
                (2280, 310, 56, -1.00, "Devil's Palm (Areia)"),
                (1890, 745, 50, -1.00, "Devil's Palm (Areia)"),
                (2250, 1290, 56, -1.00, "Devil's Palm (Areia)"),
                (1530, 1060, 50, +0.45, "Golden Spin Boost"),
                (380, 1210, 56, -1.00, "Devil's Palm (Areia)"),
                (940, 675, 50, -1.00, "Devil's Palm (Areia)"),
                (460, 310, 50, +0.45, "Golden Spin Boost"),
            ]
            self.obstaculos = [
                [1150.0, 340.0, 27.0,   0.0,  6.0],
                [1402.0, 535.0, 28.0,  45.0, -6.0],
                [1652.0, 310.0, 27.0,  90.0,  6.0],
                [1900.0, 330.0, 25.0, 135.0, -6.0],
                [2360.0, 440.0, 28.0, 180.0,  6.0],
                [2270.0, 620.0, 26.0, 225.0, -6.0],
                [1951.0, 745.0, 27.0, 270.0,  6.0],
                [2205.0, 925.0, 26.0, 315.0, -6.0],
                [2410.0, 1218.0, 28.0,  0.0,  6.0],
                [1906.0, 1190.0, 27.0, 45.0, -6.0],
                [1656.0, 1095.0, 27.0, 90.0,  6.0],
                [1156.0, 1040.0, 27.0, 135.0, -6.0],
                [661.0,  1365.0, 26.0, 180.0, 6.0],
                [250.0,  1190.0, 28.0, 225.0, -6.0],
                [716.0,  870.0,  26.0, 270.0, 6.0],
                [930.0,  685.0,  27.0, 315.0, -6.0],
                [332.0,  462.0,  28.0,   0.0, 6.0],
            ]
        else:
            self.obstaculos, self.zonas = gerar_obstaculos_e_zonas_por_distancias(
                self.cenario_base, self.distancias, self.maior_distancia
            )

        self.matriz_boost = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self._atualizar_cenario_obstaculos()

    def _atualizar_cenario_obstaculos(self):
        self.cenario = self.cenario_base.copy()
        self.matriz_boost.fill(0.0)

        if not self.obstaculos_ativos:
            return

        yy, xx = np.ogrid[:self.map_h, :self.map_w]
        for zx, zy, zraio, zval, _ in self.zonas:
            mask_z = (xx - zx) ** 2 + (yy - zy) ** 2 <= zraio ** 2
            self.matriz_boost[mask_z] = zval

        for ox, oy, oraio, _, _ in self.obstaculos:
            mask_o = (xx - ox) ** 2 + (yy - oy) ** 2 <= oraio ** 2
            self.cenario[mask_o] = 0

    # ========================================================================
    # 4. ALOCAÇÃO E INICIALIZAÇÃO DA POPULAÇÃO DE CAVALOS
    # ========================================================================
    def _alocar_arrays_populacao(self):
        n = self.qtd_cavalos
        self.X = np.zeros(n, dtype=np.float64)
        self.Y = np.zeros(n, dtype=np.float64)
        self.VX = np.zeros(n, dtype=np.float64)
        self.VY = np.zeros(n, dtype=np.float64)
        self.Angulo = np.zeros(n, dtype=np.float64)
        self.Velocidade = np.zeros(n, dtype=np.float64)
        self.Colidiu = np.zeros(n, dtype=np.uint8)
        self.Queimado = np.zeros(n, dtype=np.uint8)
        self.ConcluiuCorrida = np.zeros(n, dtype=np.uint8)
        self.TicksConclusao = np.zeros(n, dtype=np.int32)
        self.RiderIdx = np.array([i % len(RIDERS_INFO) for i in range(n)], dtype=np.int32)
        self.DistanciaSensores = np.zeros((n, SBR_BRAIN_QTD_INPUT - 1), dtype=np.float64)
        self.Entradas = np.zeros((n, SBR_BRAIN_QTD_INPUT), dtype=np.float64)
        self.Fitness = np.zeros(n, dtype=np.float64)

        min_pts = min(len(p) for p in self.rider_perimeters)
        self.perim_points = np.zeros((n, min_pts + 3, 2), dtype=np.float64)
        for i in range(n):
            r_idx = self.RiderIdx[i]
            pts = self.rider_perimeters[r_idx][:min_pts]
            self.perim_points[i, :min_pts, :] = pts
            self.perim_points[i, min_pts + 0, :] = [0.0, 0.0]
            self.perim_points[i, min_pts + 1, :] = [18.0, 0.0]
            self.perim_points[i, min_pts + 2, :] = [-18.0, 0.0]

    def inicializar_nova_partida(self):
        """Posiciona todos os cavalos na faixa de largada do mapa atual e reinicia a corrida."""
        self.cavalos_colididos = 0
        self.valor_desfazer_pista = float(self.maior_distancia)
        self.velocidade_laser = 1.15
        self.ticks_partida = 0
        self.lista_explosoes.clear()

        n = self.qtd_cavalos
        perp_rad = math.radians(self.start_angle + 90.0)
        fwd_rad = math.radians(self.start_angle)
        lateral_offsets = np.random.uniform(-22.0, 22.0, size=n)
        forward_offsets = np.random.uniform(-12.0, 16.0, size=n)

        self.X[:] = self.start_x + lateral_offsets * math.cos(perp_rad) + forward_offsets * math.cos(fwd_rad)
        self.Y[:] = self.start_y + lateral_offsets * math.sin(perp_rad) + forward_offsets * math.sin(fwd_rad)
        self.VX.fill(0.0)
        self.VY.fill(0.0)
        self.Angulo.fill(self.start_angle)
        self.Velocidade.fill(0.0)
        self.Colidiu.fill(0)
        self.Queimado.fill(0)
        self.ConcluiuCorrida.fill(0)
        self.TicksConclusao.fill(0)
        self.Fitness.fill(0.0)

    # ========================================================================
    # 5. BUSCA DO MELHOR CAVALO VIVO, RAYCASTING, REDE NEURAL E FÍSICA
    # ========================================================================
    def buscar_melhor_cavalo_vivo(self) -> int:
        """Retorna o índice do cavalo vivo com maior Fitness geodésico validado."""
        vivos = np.where(self.Colidiu == 0)[0]
        if len(vivos) == 0:
            return int(np.argmax(self.Fitness))
        best_local = int(np.argmax(self.Fitness[vivos]))
        return int(vivos[best_local])

    def aplicar_sensores_populacao(self, alive_indices: np.ndarray):
        """
        Raycasting 2D Vetorizado em 2 estágios (NumPy):
        Dispara 17 sensores de -90° a +90° para cada cavalo vivo até atingir `cenario[y, x] == 0`,
        e preenche a 18ª entrada com a Velocidade escalar atual do cavalo.
        """
        if len(alive_indices) == 0:
            return

        ax = self.X[alive_indices]
        ay = self.Y[alive_indices]
        a_ang = self.Angulo[alive_indices]

        angles_deg = a_ang[:, None] + self.sensor_rel_angles_deg[None, :]
        angles_rad = np.deg2rad(angles_deg)
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)

        steps = self.ray_steps_coarse
        rx = (ax[:, None, None] + cos_a[:, :, None] * steps[None, None, :]).astype(np.int32)
        ry = (ay[:, None, None] + sin_a[:, :, None] * steps[None, None, :]).astype(np.int32)

        np.clip(rx, 0, self.map_w - 1, out=rx)
        np.clip(ry, 0, self.map_h - 1, out=ry)

        hits = (self.cenario[ry, rx] == 0)
        any_hit = np.any(hits, axis=2)
        first_idx = np.argmax(hits, axis=2)
        coarse_dist = np.where(any_hit, steps[first_idx], steps[-1])

        base_d = np.maximum(1.0, coarse_dist - 3.0)
        fine_offsets = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        fine_d = base_d[:, :, None] + fine_offsets[None, None, :]

        fx = np.clip((ax[:, None, None] + cos_a[:, :, None] * fine_d).astype(np.int32), 0, self.map_w - 1)
        fy = np.clip((ay[:, None, None] + sin_a[:, :, None] * fine_d).astype(np.int32), 0, self.map_h - 1)
        fine_hits = (self.cenario[fy, fx] == 0)
        fine_any = np.any(fine_hits, axis=2)
        fine_first = np.argmax(fine_hits, axis=2)
        exact_dist = np.where(fine_any, np.take_along_axis(fine_d, fine_first[:, :, None], axis=2)[:, :, 0], coarse_dist)

        self.DistanciaSensores[alive_indices] = exact_dist
        self.Entradas[alive_indices, :SBR_BRAIN_QTD_INPUT - 1] = exact_dist
        self.Entradas[alive_indices, SBR_BRAIN_QTD_INPUT - 1] = self.Velocidade[alive_indices]

    def controlar_cavalos(self):
        alive_mask = (self.Colidiu == 0)
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            return

        self.aplicar_sensores_populacao(alive_indices)
        leader_idx = self.buscar_melhor_cavalo_vivo()
        saidas = self.neural_db.calcular_saida_populacao(self.Entradas, alive_mask, leader_idx)

        s_alive = saidas[alive_indices]
        vel = self.Velocidade[alive_indices]
        ang = self.Angulo[alive_indices]

        vel = np.where(s_alive[:, 0] > 0.0, np.minimum(13.5, vel + 0.2), vel)
        vel = np.where(s_alive[:, 1] > 0.0, np.maximum(-4.0, vel - 0.2), vel)
        ang = np.where(s_alive[:, 2] > 0.0, ang - 5.0, ang)
        ang = np.where(s_alive[:, 3] > 0.0, ang + 5.0, ang)

        self.Velocidade[alive_indices] = vel
        self.Angulo[alive_indices] = ang

    def movimentar_cavalos(self):
        alive_indices = np.where(self.Colidiu == 0)[0]
        if len(alive_indices) == 0:
            return

        ax = self.X[alive_indices]
        ay = self.Y[alive_indices]
        vel = self.Velocidade[alive_indices].copy()
        ang_rad = np.deg2rad(self.Angulo[alive_indices])

        if self.obstaculos_ativos:
            ix = np.clip(ax.astype(np.int32), 0, self.map_w - 1)
            iy = np.clip(ay.astype(np.int32), 0, self.map_h - 1)
            boost = self.matriz_boost[iy, ix]
            temp_vel = np.where(boost < 0.0, vel / 4.0, vel + np.maximum(0.0, boost))
            vel = np.where(boost > 0.0, np.minimum(15.5, vel + boost), vel)
        else:
            temp_vel = vel

        cos_a = np.cos(ang_rad)
        sin_a = np.sin(ang_rad)
        vx = temp_vel * cos_a
        vy = temp_vel * sin_a

        nx = ax + vx
        ny = ay + vy

        perim = self.perim_points[alive_indices] * 0.85
        px = (nx[:, None] + perim[:, :, 0] * cos_a[:, None] - perim[:, :, 1] * sin_a[:, None]).astype(np.int32)
        py = (ny[:, None] + perim[:, :, 0] * sin_a[:, None] + perim[:, :, 1] * cos_a[:, None]).astype(np.int32)

        fora_mapa = (px < 0) | (px >= self.map_w) | (py < 0) | (py >= self.map_h)
        px_safe = np.clip(px, 0, self.map_w - 1)
        py_safe = np.clip(py, 0, self.map_h - 1)

        bateu_parede = fora_mapa | (self.cenario[py_safe, px_safe] == 0)
        colidiu_agora = np.any(bateu_parede, axis=1)

        ok_mask = ~colidiu_agora
        ok_indices = alive_indices[ok_mask]
        self.X[ok_indices] = nx[ok_mask]
        self.Y[ok_indices] = ny[ok_mask]
        self.VX[ok_indices] = vx[ok_mask]
        self.VY[ok_indices] = vy[ok_mask]

        vel_ok = vel[ok_mask]
        vel_ok = np.where(vel_ok > 0.0, np.maximum(0.0, vel_ok - 0.1), vel_ok)
        vel_ok = np.where(vel_ok < 0.0, np.minimum(0.0, vel_ok + 0.1), vel_ok)
        self.Velocidade[ok_indices] = vel_ok

        hit_indices = alive_indices[colidiu_agora]
        if len(hit_indices) > 0:
            self.Colidiu[hit_indices] = 1
            self.cavalos_colididos = int(np.sum(self.Colidiu))
            self._atualizar_fitness_indices(hit_indices)

        # Atualiza o Fitness em tempo real e verifica se algum cavalo CONCLUIU A CORRIDA!
        if len(ok_indices) > 0:
            self._atualizar_fitness_indices(ok_indices)
            ix_ok = np.clip(self.X[ok_indices].astype(np.int32), 0, self.map_w - 1)
            iy_ok = np.clip(self.Y[ok_indices].astype(np.int32), 0, self.map_h - 1)
            d_ok = self.distancias[iy_ok, ix_ok]
            vencedores = ok_indices[
                (d_ok >= 0) & (d_ok <= 35) & (self.Fitness[ok_indices] > self.maior_distancia * 0.85)
            ]
            if len(vencedores) > 0:
                bonus_tempo = max(0.0, (2500.0 - float(self.ticks_partida)) * 0.1)
                self.Fitness[vencedores] = float(self.maior_distancia) + bonus_tempo
                self.ConcluiuCorrida[vencedores] = 1
                self.TicksConclusao[vencedores] = self.ticks_partida
                self.Colidiu[vencedores] = 1
                self.cavalos_colididos = int(np.sum(self.Colidiu))

    def _atualizar_fitness_indices(self, indices: np.ndarray):
        ix = np.clip(self.X[indices].astype(np.int32), 0, self.map_w - 1)
        iy = np.clip(self.Y[indices].astype(np.int32), 0, self.map_h - 1)
        d = self.distancias[iy, ix]
        valid = d >= 0
        novos_fit = np.where(valid, float(self.maior_distancia) - d.astype(np.float64), self.Fitness[indices])

        salto_contramao = (novos_fit - self.Fitness[indices]) > 800.0
        if np.any(salto_contramao):
            infratores = indices[salto_contramao]
            self.Colidiu[infratores] = 1
            self.Queimado[infratores] = 1
            self.Fitness[infratores] = 0.0
            self.cavalos_colididos = int(np.sum(self.Colidiu))

        valid_progress = ~salto_contramao
        if np.any(valid_progress):
            ok_idx = indices[valid_progress]
            self.Fitness[ok_idx] = np.maximum(self.Fitness[ok_idx], novos_fit[valid_progress])

    def laser_destruidor(self):
        alive_indices = np.where(self.Colidiu == 0)[0]
        if len(alive_indices) > 0:
            ix = np.clip(self.X[alive_indices].astype(np.int32), 0, self.map_w - 1)
            iy = np.clip(self.Y[alive_indices].astype(np.int32), 0, self.map_h - 1)
            d = self.distancias[iy, ix]
            atingidos = alive_indices[(d >= self.valor_desfazer_pista) | (d < 0)]
            if len(atingidos) > 0:
                self.Colidiu[atingidos] = 1
                self.Queimado[atingidos] = 1
                self._atualizar_fitness_indices(atingidos)
                for idx in atingidos[:15]:
                    self.criar_explosao(self.X[idx], self.Y[idx])
                self.cavalos_colididos = int(np.sum(self.Colidiu))

        if self.valor_desfazer_pista > 0:
            self.valor_desfazer_pista -= self.velocidade_laser
            self.velocidade_laser += 0.0045

    def criar_explosao(self, x: float, y: float):
        self.lista_explosoes.append({"x": x, "y": y, "frame": 0, "max_frames": 22})

    def verificar_fim_de_partida(self):
        """Encerra a rodada quando todos os cavalos param e dispara a evolução genética no MySQL."""
        if self.cavalos_colididos >= self.qtd_cavalos or self.valor_desfazer_pista <= 0:
            vivos = np.where(self.Colidiu == 0)[0]
            if len(vivos) > 0:
                self._atualizar_fitness_indices(vivos)
                self.Colidiu[vivos] = 1
            ix = np.clip(self.X.astype(np.int32), 0, self.map_w - 1)
            iy = np.clip(self.Y.astype(np.int32), 0, self.map_h - 1)
            dist_rest = np.where(self.distancias[iy, ix] >= 0, self.distancias[iy, ix], self.maior_distancia)

            sucesso = self.neural_db.evoluir_geracao(
                self.Fitness,
                dist_rest,
                self.X,
                self.Y,
                self.Colidiu,
                self.Queimado,
                self.ConcluiuCorrida,
                self.TicksConclusao,
                self.RiderIdx,
                RIDERS_INFO,
            )
            if sucesso:
                self.inicializar_nova_partida()

    # ========================================================================
    # 6. TRANSFORMAÇÃO DE CÂMERA E RENDERIZAÇÃO 2D
    # ========================================================================
    def world_to_screen(self, wx: float, wy: float):
        sx = (wx - self.camera_x) * self.camera_zoom + (self.win_w / 2.0)
        sy = (wy - self.camera_y) * self.camera_zoom + (self.win_h / 2.0)
        return int(sx), int(sy)

    def screen_to_world(self, sx: int, sy: int):
        wx = (sx - (self.win_w / 2.0)) / self.camera_zoom + self.camera_x
        wy = (sy - (self.win_h / 2.0)) / self.camera_zoom + self.camera_y
        return wx, wy

    def desenhar_rede_neural(self, x_orig: int, y_orig: int, largura: int, altura: int, leader_idx: int):
        panel = pygame.Surface((largura, altura), pygame.SRCALPHA)
        pygame.draw.rect(panel, (15, 12, 18, 215), (0, 0, largura, altura), border_radius=12)
        pygame.draw.rect(panel, (212, 175, 55, 220), (0, 0, largura, altura), width=2, border_radius=12)
        self.screen.blit(panel, (x_orig, y_orig))

        rider = RIDERS_INFO[self.RiderIdx[leader_idx]]
        gid = self.neural_db.genome_ids[leader_idx]
        title = self.font_md.render(
            f"CÉREBRO DO LÍDER: {rider['name']} (Genoma MySQL #{gid})", True, (245, 215, 110)
        )
        self.screen.blit(title, (x_orig + 14, y_orig + 8))

        in_vals = self.neural_db.last_in[leader_idx]
        hid_vals = self.neural_db.last_hid[leader_idx]
        out_vals = self.neural_db.last_out[leader_idx]
        w1 = self.neural_db.W1[leader_idx]
        w2 = self.neural_db.W2[leader_idx]

        col_in_x = x_orig + 40
        col_hid_x = x_orig + 195
        col_out_x = x_orig + 350

        top_y = y_orig + 42
        usable_h = altura - 56

        in_ys = [int(top_y + i * (usable_h / max(1, SBR_BRAIN_QTD_INPUT - 1))) for i in range(SBR_BRAIN_QTD_INPUT)]
        hid_ys = [int(top_y + 28 + j * ((usable_h - 56) / max(1, SBR_BRAIN_QTD_HIDE - 1))) for j in range(SBR_BRAIN_QTD_HIDE)]
        out_ys = [int(top_y + 48 + k * ((usable_h - 96) / max(1, SBR_BRAIN_QTD_OUTPUT - 1))) for k in range(SBR_BRAIN_QTD_OUTPUT)]

        for i in range(SBR_BRAIN_QTD_INPUT):
            s_val = in_vals[i]
            for j in range(SBR_BRAIN_QTD_HIDE):
                peso = w1[i, j]
                ativo = (peso * s_val) > 0.0
                cor = (225, 55, 55) if ativo else (65, 65, 75)
                pygame.draw.line(self.screen, cor, (col_in_x, in_ys[i]), (col_hid_x, hid_ys[j]), 1)

        for j in range(SBR_BRAIN_QTD_HIDE):
            s_val = hid_vals[j]
            for k in range(SBR_BRAIN_QTD_OUTPUT):
                peso = w2[j, k]
                ativo = (peso * s_val) > 0.0
                cor = (240, 190, 40) if ativo else (70, 70, 80)
                pygame.draw.line(self.screen, cor, (col_hid_x, hid_ys[j]), (col_out_x, out_ys[k]), 2 if ativo else 1)

        for i in range(SBR_BRAIN_QTD_INPUT):
            if i == SBR_BRAIN_QTD_INPUT - 1:
                op = min(1.0, abs(in_vals[i]) / 10.0)
                cor_n = (int(50 + 205 * op), int(180 * op), 255)
            else:
                op = max(0.0, min(1.0, (250.0 - in_vals[i]) / 250.0))
                v = int(40 + 215 * op)
                cor_n = (v, v, v)
            pygame.draw.circle(self.screen, (240, 240, 240), (col_in_x, in_ys[i]), 6)
            pygame.draw.circle(self.screen, cor_n, (col_in_x, in_ys[i]), 4)

        for j in range(SBR_BRAIN_QTD_HIDE):
            ativo = hid_vals[j] > 0.0
            cor_n = (255, 240, 150) if ativo else (25, 25, 30)
            pygame.draw.circle(self.screen, (230, 230, 230), (col_hid_x, hid_ys[j]), 9)
            pygame.draw.circle(self.screen, cor_n, (col_hid_x, hid_ys[j]), 7)

        labels_saida = ["Galopar (Acelerar)", "Frear / Ré", "Rédea Esquerda", "Rédea Direita"]
        for k in range(SBR_BRAIN_QTD_OUTPUT):
            ativo = out_vals[k] > 0.0
            cor_n = (80, 255, 120) if ativo else (25, 25, 30)
            pygame.draw.circle(self.screen, (255, 255, 255), (col_out_x, out_ys[k]), 10)
            pygame.draw.circle(self.screen, cor_n, (col_out_x, out_ys[k]), 8)
            lbl_color = (120, 255, 150) if ativo else (175, 175, 185)
            lbl_surf = self.font_sm.render(labels_saida[k], True, lbl_color)
            self.screen.blit(lbl_surf, (col_out_x + 16, out_ys[k] - 7))

    def desenhar_hud_telemetria(self, leader_idx: int):
        """Painel de telemetria da corrida, Campeão Oficial (só se concluir), seletor de mapas e MySQL."""
        pw, ph = 525, 290
        px, py = self.win_w - pw - 16, 16
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(panel, (15, 12, 18, 225), (0, 0, pw, ph), border_radius=12)
        pygame.draw.rect(panel, (212, 175, 55, 220), (0, 0, pw, ph), width=2, border_radius=12)
        self.screen.blit(panel, (px, py))

        vivos = self.qtd_cavalos - self.cavalos_colididos
        concluiram_agora = int(np.sum(self.ConcluiuCorrida))
        fit_lider = float(self.Fitness[leader_idx])
        pct_lider = max(0.0, min(100.0, (fit_lider / float(self.maior_distancia)) * 100.0))
        pct_recorde = max(0.0, min(100.0, (self.neural_db.distancia_recorde / float(self.maior_distancia)) * 100.0))

        teve_campeao_ant = self.neural_db.qtd_concluiram_geracao_ant > 0
        cor_campeao_ant = (255, 230, 80) if teve_campeao_ant else (165, 165, 175)

        linhas = [
            (f"STEEL BALL RUN — GERAÇÃO {self.neural_db.geracao}", (245, 215, 110), self.font_lg),
            (
                f"Mapa [{self.mapa_atual_idx + 1}/{len(self.catalogo_mapas)}] (TAB): {self.nome_mapa_atual[:38]}",
                (255, 195, 110),
                self.font_sm,
            ),
            (
                f"Vivos: {vivos}/{self.qtd_cavalos} | Concluíram Volta: {concluiram_agora} | Sim: {self.passos_por_frame}x ({int(self.clock.get_fps())} FPS)",
                (230, 230, 235),
                self.font_sm,
            ),
            (f"Líder na Pista: {RIDERS_INFO[self.RiderIdx[leader_idx]]['name']}", (130, 225, 255), self.font_sm),
            (f"Progresso Líder: {fit_lider:.0f} / {self.maior_distancia} px ({pct_lider:.1f}%)", (120, 255, 140), self.font_sm),
            (f"Campeão Ger. Ant. (Concluiu): {self.neural_db.campeao_geracao_anterior_str[:36]}", cor_campeao_ant, self.font_sm),
            (f"Último Campeão Oficial: {self.neural_db.ultimo_campeao_oficial_str[:42]}", (255, 215, 90), self.font_sm),
            (
                f"Melhor Dist. Ant.: {self.neural_db.melhor_fitness_geracao:.0f} px | Recorde: {self.neural_db.distancia_recorde:.0f} px ({pct_recorde:.1f}%)",
                (200, 200, 210),
                self.font_sm,
            ),
            (
                f"Range Mutação: {self.neural_db.range_random:.1f}/{self.neural_db.dna_size} | Obstáculos (O): {'ATIVOS' if self.obstaculos_ativos else 'OFF'}",
                (215, 180, 255),
                self.font_sm,
            ),
            (self.neural_db.mysql_status_msg, (100, 255, 140) if self.neural_db.mysql_online else (255, 210, 100), self.font_sm),
        ]

        cy = py + 10
        for idx_l, (texto, cor, fnt) in enumerate(linhas):
            if idx_l == 1:
                # Destaca a linha do Mapa como um botão clicável além do atalho [TAB]
                self._btn_trocar_mapa_rect = pygame.Rect(px + 10, cy - 2, pw - 20, 21)
                pygame.draw.rect(self.screen, (55, 42, 22), self._btn_trocar_mapa_rect, border_radius=5)
                pygame.draw.rect(self.screen, (212, 175, 55), self._btn_trocar_mapa_rect, width=1, border_radius=5)
            surf = fnt.render(texto, True, cor)
            self.screen.blit(surf, (px + 14, cy))
            cy += 25 if fnt == self.font_lg else 22

        # Barra de atalhos inferior
        bar_h = 30
        bar = pygame.Surface((self.win_w, bar_h), pygame.SRCALPHA)
        bar.fill((12, 10, 15, 225))
        self.screen.blit(bar, (0, self.win_h - bar_h))
        help_txt = (
            "[TAB / 1,2,3] Trocar Mapa | [WASD] Câmera | [Q/E/Scroll] Zoom | [R] Visão Geral | [G] Seguir Líder | "
            "[C] Sensores | [Z] Colisão | [X] Distâncias | [N/M] Vel | [O] Obstáculos | [F2] Campeão | [F5] Sync | [F6] Reset G0"
        )
        h_surf = self.font_sm.render(help_txt, True, (215, 205, 180))
        self.screen.blit(h_surf, (12, self.win_h - bar_h + 7))

    def desenhar(self):
        self.screen.fill((22, 15, 12))
        leader_idx = self.buscar_melhor_cavalo_vivo()

        if self.seguir_lider:
            self.camera_x = self.X[leader_idx]
            self.camera_y = self.Y[leader_idx]

        # 1. Desenha o Mapa (ou a Matriz de Colisão se Tecla Z estiver ativa)
        scaled_w = max(10, int(self.map_w * self.camera_zoom))
        scaled_h = max(10, int(self.map_h * self.camera_zoom))
        if self._cached_zoom != self.camera_zoom:
            self._cached_zoom = self.camera_zoom
            self._cached_map_surf = pygame.transform.smoothscale(self.mapa_surface, (scaled_w, scaled_h))
            self._cached_mask_surf = pygame.transform.scale(self.mascara_surface, (scaled_w, scaled_h))

        map_sx, map_sy = self.world_to_screen(0, 0)
        base_surf = self._cached_mask_surf if self.mostrar_mascara_colisao else self._cached_map_surf
        self.screen.blit(base_surf, (map_sx, map_sy))

        # 2. Desenha Zonas de Terreno e Obstáculos se ativos
        if self.obstaculos_ativos:
            for zx, zy, zraio, zval, _ in self.zonas:
                sx, sy = self.world_to_screen(zx, zy)
                sr = max(4, int(zraio * self.camera_zoom))
                cor_z = (255, 215, 40) if zval > 0 else (140, 75, 30)
                pygame.draw.circle(self.screen, cor_z, (sx, sy), sr, width=3)
            for ox, oy, oraio, oang, _ in self.obstaculos:
                sx, sy = self.world_to_screen(ox, oy)
                sr = max(4, int(oraio * self.camera_zoom))
                pygame.draw.circle(self.screen, (60, 210, 110), (sx, sy), sr)
                pygame.draw.circle(self.screen, (240, 255, 200), (sx, sy), sr, width=2)

        # 3. Desenha a Onda Eliminatória (Fronteira Geodésica de Desclassificação)
        target_d = int(self.valor_desfazer_pista)
        if 0 < target_d <= self.maior_distancia:
            sub_dist = self.distancias[::4, ::4]
            ly, lx = np.where(np.abs(sub_dist - target_d) <= 4)
            for y4, x4 in zip(ly, lx):
                sx, sy = self.world_to_screen(x4 * 4, y4 * 4)
                if 0 <= sx < self.win_w and 0 <= sy < self.win_h:
                    pygame.draw.rect(self.screen, (0, 255, 120), (sx, sy, 4, 4))

        # 4. Desenha Cavalos Colididos/Caídos primeiro (ao fundo)
        sprite_scale = max(0.22, min(2.0, self.camera_zoom * 1.15))
        dead_indices = np.where(self.Colidiu == 1)[0]
        for i in dead_indices:
            sx, sy = self.world_to_screen(self.X[i], self.Y[i])
            if -60 <= sx <= self.win_w + 60 and -60 <= sy <= self.win_h + 60:
                r_idx = self.RiderIdx[i]
                base_sp = self.rider_mask_surfaces[r_idx] if self.mostrar_mascara_colisao else self.rider_fallen_surfaces[r_idx]
                rot_sp = pygame.transform.rotozoom(base_sp, -self.Angulo[i], sprite_scale)
                rect = rot_sp.get_rect(center=(sx, sy))
                self.screen.blit(rot_sp, rect)

        # 5. Desenha Cavalos Vivos por cima
        alive_indices = np.where(self.Colidiu == 0)[0]
        for i in alive_indices:
            sx, sy = self.world_to_screen(self.X[i], self.Y[i])
            if -60 <= sx <= self.win_w + 60 and -60 <= sy <= self.win_h + 60:
                r_idx = self.RiderIdx[i]
                base_sp = self.rider_mask_surfaces[r_idx] if self.mostrar_mascara_colisao else self.rider_surfaces[r_idx]
                rot_sp = pygame.transform.rotozoom(base_sp, -self.Angulo[i], sprite_scale)
                rect = rot_sp.get_rect(center=(sx, sy))
                self.screen.blit(rot_sp, rect)

                if i == leader_idx:
                    pygame.draw.circle(self.screen, (255, 220, 60), (sx, sy), max(12, int(28 * sprite_scale)), width=2)

        # 6. Desenha os 17 Sensores Raycast do Cavalo Líder (Tecla C)
        if self.mostrar_sensores and len(alive_indices) > 0:
            lx, ly = self.X[leader_idx], self.Y[leader_idx]
            lsx, lsy = self.world_to_screen(lx, ly)
            l_ang = self.Angulo[leader_idx]
            dists = self.DistanciaSensores[leader_idx]
            for s_i in range(SBR_BRAIN_QTD_INPUT - 1):
                ang_rad = math.radians(l_ang + self.sensor_rel_angles_deg[s_i])
                ex = lx + dists[s_i] * math.cos(ang_rad)
                ey = ly + dists[s_i] * math.sin(ang_rad)
                esx, esy = self.world_to_screen(ex, ey)
                if s_i < SBR_BRAIN_QTD_INPUT // 4 or s_i > (3 * SBR_BRAIN_QTD_INPUT) // 4:
                    cor_r = (255, 215, 40)
                else:
                    cor_r = (50, 240, 255)
                pygame.draw.line(self.screen, cor_r, (lsx, lsy), (esx, esy), 2)
                pygame.draw.circle(self.screen, cor_r, (esx, esy), 4)

        # 7. Desenha Matriz de Distâncias ao redor do Mouse (Tecla X)
        if self.mostrar_matriz_distancias:
            mx, my = pygame.mouse.get_pos()
            wx, wy = self.screen_to_world(mx, my)
            c_ix, c_iy = int(wx), int(wy)
            passo = max(8, int(22 / max(0.25, self.camera_zoom)))
            for dy in range(-4 * passo, 5 * passo, passo):
                for dx in range(-4 * passo, 5 * passo, passo):
                    px_i, py_i = c_ix + dx, c_iy + dy
                    if 0 <= px_i < self.map_w and 0 <= py_i < self.map_h:
                        d_val = self.distancias[py_i, px_i]
                        sx, sy = self.world_to_screen(px_i, py_i)
                        cor_d = (80, 255, 120) if d_val >= 0 else (255, 80, 80)
                        txt_s = self.font_dist.render(str(d_val), True, cor_d)
                        self.screen.blit(txt_s, (sx - 12, sy - 6))

        # 8. Desenha Explosões de Eliminação
        novas_exp = []
        for exp in self.lista_explosoes:
            exp["frame"] += 1
            if exp["frame"] <= exp["max_frames"]:
                sx, sy = self.world_to_screen(exp["x"], exp["y"])
                raio = int((exp["frame"] / exp["max_frames"]) * 38 * max(0.5, self.camera_zoom))
                if raio > 2:
                    pygame.draw.circle(self.screen, (255, 160, 40), (sx, sy), raio, width=2)
                novas_exp.append(exp)
        self.lista_explosoes = novas_exp

        # 9. Desenha HUD da Rede Neural e Telemetria MySQL
        if self.mostrar_hud_rede:
            self.desenhar_rede_neural(16, 16, 485, 310, leader_idx)
        self.desenhar_hud_telemetria(leader_idx)

        # 10. Se o MySQL estiver desconectado, exibe aviso de PAUSA (sem fallback local!)
        if not self.neural_db.mysql_online:
            bw, bh = 760, 88
            bx, by = (self.win_w - bw) // 2, (self.win_h - bh) // 2
            bpanel = pygame.Surface((bw, bh), pygame.SRCALPHA)
            pygame.draw.rect(bpanel, (45, 10, 10, 235), (0, 0, bw, bh), border_radius=12)
            pygame.draw.rect(bpanel, (255, 80, 80, 255), (0, 0, bw, bh), width=3, border_radius=12)
            self.screen.blit(bpanel, (bx, by))
            t1 = self.font_lg.render("SIMULAÇÃO PAUSADA — AGUARDANDO CONEXÃO COM O MYSQL SERVER", True, (255, 220, 220))
            t2 = self.font_md.render(f"{self.neural_db.mysql_status_msg} | Pressione [F5] para Reconectar", True, (255, 180, 120))
            self.screen.blit(t1, (bx + (bw - t1.get_width()) // 2, by + 16))
            self.screen.blit(t2, (bx + (bw - t2.get_width()) // 2, by + 48))

        pygame.display.flip()

    # ========================================================================
    # 7. CONTROLES DO USUÁRIO E LOOP PRINCIPAL
    # ========================================================================
    def processar_eventos_e_teclado(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.VIDEORESIZE:
                self.win_w, self.win_h = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
            elif ev.type == pygame.MOUSEWHEEL:
                fator = 1.12 if ev.y > 0 else (1.0 / 1.12)
                self.camera_zoom = max(0.25, min(3.5, self.camera_zoom * fator))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self._btn_trocar_mapa_rect.collidepoint(ev.pos):
                    self.trocar_proximo_mapa()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                elif ev.key == pygame.K_TAB:
                    self.trocar_proximo_mapa()
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx_m = ev.key - pygame.K_1
                    if idx_m < len(self.catalogo_mapas):
                        self._carregar_mapa_por_indice(idx_m, reiniciar_partida=True)
                elif ev.key == pygame.K_g:
                    self.seguir_lider = not self.seguir_lider
                elif ev.key == pygame.K_c:
                    self.mostrar_sensores = not self.mostrar_sensores
                elif ev.key == pygame.K_z:
                    self.mostrar_mascara_colisao = not self.mostrar_mascara_colisao
                elif ev.key == pygame.K_x:
                    self.mostrar_matriz_distancias = not self.mostrar_matriz_distancias
                elif ev.key == pygame.K_h:
                    self.mostrar_hud_rede = not self.mostrar_hud_rede
                elif ev.key == pygame.K_o:
                    self.obstaculos_ativos = not self.obstaculos_ativos
                    self._atualizar_cenario_obstaculos()
                elif ev.key == pygame.K_r:
                    self.seguir_lider = False
                    self.camera_zoom = min(self.win_w / float(self.map_w), self.win_h / float(self.map_h)) * 0.95
                    self.camera_x = self.map_w / 2.0
                    self.camera_y = self.map_h / 2.0
                elif ev.key == pygame.K_n:
                    self.passos_por_frame = min(16, self.passos_por_frame * 2)
                elif ev.key == pygame.K_m:
                    self.passos_por_frame = max(1, self.passos_por_frame // 2)
                elif ev.key == pygame.K_F2:
                    self.neural_db.carregar_campeao_mysql()
                elif ev.key == pygame.K_F5:
                    self.neural_db.try_connect_and_init_mysql()
                elif ev.key == pygame.K_F6:
                    if self.neural_db.resetar_populacao_mysql():
                        self.inicializar_nova_partida()

        keys = pygame.key.get_pressed()
        pan_speed = 14.0 / max(0.25, self.camera_zoom)
        if keys[pygame.K_w]:
            self.seguir_lider = False
            self.camera_y -= pan_speed
        if keys[pygame.K_s]:
            self.seguir_lider = False
            self.camera_y += pan_speed
        if keys[pygame.K_a]:
            self.seguir_lider = False
            self.camera_x -= pan_speed
        if keys[pygame.K_d]:
            self.seguir_lider = False
            self.camera_x += pan_speed
        if keys[pygame.K_q]:
            self.camera_zoom = min(3.5, self.camera_zoom * 1.03)
        if keys[pygame.K_e]:
            self.camera_zoom = max(0.25, self.camera_zoom / 1.03)

        if self.Colidiu[0] == 0:
            if keys[pygame.K_UP]:
                self.Velocidade[0] = min(10.0, self.Velocidade[0] + 0.25)
            if keys[pygame.K_DOWN]:
                self.Velocidade[0] = max(-5.0, self.Velocidade[0] - 0.25)
            if keys[pygame.K_LEFT]:
                self.Angulo[0] -= 4.8
            if keys[pygame.K_RIGHT]:
                self.Angulo[0] += 4.8

        return True

    def executar_tick_simulacao(self):
        if not self.neural_db.mysql_online:
            return
        self.ticks_partida += 1
        self.controlar_cavalos()
        self.movimentar_cavalos()
        if self.obstaculos_ativos:
            for obs in self.obstaculos:
                obs[3] = (obs[3] + obs[4]) % 360.0
        self.laser_destruidor()
        self.verificar_fim_de_partida()

    def rodar(self):
        rodando = True
        while rodando:
            rodando = self.processar_eventos_e_teclado()
            for _ in range(self.passos_por_frame):
                self.executar_tick_simulacao()
            self.desenhar()
            self.clock.tick(FPS_ALVO)
        pygame.quit()


if __name__ == "__main__":
    pop = QTD_CAVALOS_PADRAO
    if len(sys.argv) > 1:
        try:
            pop = max(5, int(sys.argv[1]))
        except ValueError:
            pass
    sim = SteelBallRunSimulator(qtd_cavalos=pop)
    sim.rodar()
