# -*- coding: utf-8 -*-
"""
Gerenciador e Construtor Automático de Mapas para o Steel Ball Run Neural Simulator.

O usuário fornece os próprios sprites de mapas colocando-os em:
  - `Sprites/` (ex: `mapa2.jpg`, `mapa_deserto.png`, etc.) ou
  - `Sprites/mapas/` (ou `Sprites/mapas_custom/`)

Para cada imagem de mapa encontrada, o módulo gera automaticamente (se ainda não existir):
  1. A Matriz de Colisão Binária (`<nome>_mascara.png` / `cenario[H, W]`)
  2. A Matriz Geodésica de Distâncias via BFS Flood-Fill de 8 vizinhos (`<nome>_distancias.npz`)
  3. O Ponto/Ângulo de Largada e os 17 Obstáculos Giratórios + 8 Zonas de Terreno sobre a pista.
"""

import os
import math
from collections import deque
import numpy as np
from PIL import Image, ImageFilter

from config import BASE_DIR, SPRITES_DIR, RIDERS_INFO

CUSTOM_MAPS_DIR = os.path.join(SPRITES_DIR, "mapas_custom")
os.makedirs(CUSTOM_MAPS_DIR, exist_ok=True)

# Nomes de arquivos de personagens que NÃO são mapas na pasta Sprites/
CHARACTER_FILENAMES = {info["file"].lower() for info in RIDERS_INFO}


def compute_bfs_flood_fill(cenario: np.ndarray, start_x: int, start_y: int, start_angle_deg: float) -> tuple[np.ndarray, int]:
    """
    Executa o algoritmo BFS Flood-Fill de 8 vizinhos no sentido anti-horário da chegada até a largada,
    criando uma barreira unidirecional na linha de largada para evitar atalhos em contramão.
    """
    h, w = cenario.shape
    distancias = np.full((h, w), -1, dtype=np.int32)

    rad = math.radians(start_angle_deg)
    fwd_x, fwd_y = math.cos(rad), math.sin(rad)
    perp_x, perp_y = -fwd_y, fwd_x

    cenario_work = cenario.copy()
    seed_points = []

    for s in range(-130, 131):
        for thickness in range(-3, 4):
            bx = int(round(start_x - fwd_x * 35 + perp_x * s + fwd_x * thickness))
            by = int(round(start_y - fwd_y * 35 + perp_y * s + fwd_y * thickness))
            if 0 <= bx < w and 0 <= by < h:
                cenario_work[by, bx] = 0

        sx = int(round(start_x - fwd_x * 43 + perp_x * s))
        sy = int(round(start_y - fwd_y * 43 + perp_y * s))
        if 0 <= sx < w and 0 <= sy < h and cenario_work[sy, sx] == 1:
            distancias[sy, sx] = 0
            seed_points.append((sy, sx))

    if not seed_points:
        distancias[start_y, start_x] = 0
        seed_points.append((start_y, start_x))

    queue = deque(seed_points)
    vizinhos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        cy, cx = queue.popleft()
        cd = distancias[cy, cx] + 1
        for dy, dx in vizinhos:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                if cenario_work[ny, nx] == 1 and distancias[ny, nx] == -1:
                    distancias[ny, nx] = cd
                    queue.append((ny, nx))

    maior_dist = int(distancias.max()) if distancias.max() > 0 else 1

    for s in range(-130, 131):
        for thickness in range(-3, 4):
            bx = int(round(start_x - fwd_x * 35 + perp_x * s + fwd_x * thickness))
            by = int(round(start_y - fwd_y * 35 + perp_y * s + fwd_y * thickness))
            if 0 <= bx < w and 0 <= by < h and cenario[by, bx] == 1:
                distancias[by, bx] = maior_dist - 15

    return distancias, maior_dist


def gerar_obstaculos_e_zonas_por_distancias(cenario: np.ndarray, distancias: np.ndarray, maior_dist: int):
    """
    Posiciona automaticamente 17 obstáculos em slalom (alternando borda superior/inferior/centro)
    e 8 zonas de terreno (Golden Spin Boost e Devil's Palm) ao longo da matriz `distancias`.
    """
    obstaculos = []
    zonas = []

    fracs_obs = np.linspace(0.07, 0.93, 17)
    offsets_slalom = [-0.38, +0.38, -0.35, +0.35, 0.0]

    for idx, frac in enumerate(fracs_obs):
        target_d = int(maior_dist * (1.0 - frac))
        ys, xs = np.where((distancias >= target_d - 6) & (distancias <= target_d + 6) & (cenario == 1))
        if len(xs) < 10:
            continue
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        dx, dy = xs - cx, ys - cy
        cov = np.cov(dx, dy)
        _, eigvecs = np.linalg.eigh(cov)
        perp_axis = eigvecs[:, 1]
        proj = dx * perp_axis[0] + dy * perp_axis[1]
        half_width = max(20.0, float(np.percentile(np.abs(proj), 90)))

        slalom_factor = offsets_slalom[idx % len(offsets_slalom)]
        shift = slalom_factor * half_width
        ox = cx + perp_axis[0] * shift
        oy = cy + perp_axis[1] * shift
        # Garante que o centro do obstáculo caia sempre dentro da pista (cenario == 1)
        best_i = int(np.argmin((xs - ox) ** 2 + (ys - oy) ** 2))
        ox, oy = float(xs[best_i]), float(ys[best_i])
        raio = 26.0 if half_width < 85 else 28.0
        vel_rot = 6.0 if idx % 2 == 0 else -6.0
        obstaculos.append([round(ox, 1), round(oy, 1), raio, float((idx * 45) % 360), vel_rot])

    fracs_zonas = [0.05, 0.19, 0.31, 0.44, 0.56, 0.69, 0.81, 0.95]
    tipos_zonas = [
        (+0.45, "Golden Spin Boost"),
        (-1.00, "Devil's Palm (Areia)"),
        (-1.00, "Devil's Palm (Areia)"),
        (-1.00, "Devil's Palm (Areia)"),
        (+0.45, "Golden Spin Boost"),
        (-1.00, "Devil's Palm (Areia)"),
        (-1.00, "Devil's Palm (Areia)"),
        (+0.45, "Golden Spin Boost"),
    ]
    for idx, frac in enumerate(fracs_zonas):
        target_d = int(maior_dist * (1.0 - frac))
        ys, xs = np.where((distancias >= target_d - 8) & (distancias <= target_d + 8) & (cenario == 1))
        if len(xs) < 10:
            continue
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        best_i = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
        zx, zy = int(xs[best_i]), int(ys[best_i])
        val, nome = tipos_zonas[idx]
        zonas.append((zx, zy, 52, val, nome))

    return obstaculos, zonas


# Mapas calibrados manualmente com máscaras cirúrgicas e Flood-Fill validado
CALIBRATED_MAPS = {
    "mapa.jpg": {
        "id": "mapa_1",
        "nome": "Mapa 1: Cânion Steel Ball Run (mapa.jpg)",
        "mascara_name": "mapa_mascara.png",
        "npz_name": "matrizDistancias.npz",
        "start_x": 845.0,
        "start_y": 225.0,
        "start_angle": 23.0,
        "auto_obstacles": False,
    },
    "mapa2.jpg": {
        "id": "mapa_2",
        "nome": "Mapa 2: Desfiladeiro da Ferrovia (mapa2.jpg)",
        "mascara_name": "mapa2_mascara.png",
        "npz_name": "matrizDistancias_mapa2.npz",
        "start_x": 1620.0,
        "start_y": 1285.0,
        "start_angle": -175.0,
        "auto_obstacles": True,
    },
    "mapa3.jpg": {
        "id": "mapa_3",
        "nome": "Mapa 3: Vale dos Monumentos (mapa3.jpg)",
        "mascara_name": "mapa3_mascara.png",
        "npz_name": "matrizDistancias_mapa3.npz",
        "start_x": 1220.0,
        "start_y": 1290.0,
        "start_angle": 180.0,
        "auto_obstacles": True,
    },
    "mapa4.jpg": {
        "id": "mapa_4",
        "nome": "Mapa 4: Circuito Cruzado do Deserto (mapa4.jpg)",
        "mascara_name": "mapa4_mascara.png",
        "npz_name": "matrizDistancias_mapa4.npz",
        "start_x": 585.0,
        "start_y": 855.0,
        "start_angle": -38.0,
        "auto_obstacles": True,
    },
}


def processar_mapa_do_usuario(img_path: str) -> dict:
    """
    Dado qualquer arquivo de imagem de mapa fornecido pelo usuário (em `Sprites/` ou `Sprites/mapas_custom/`),
    carrega a calibração manual se existir ou gera a máscara de colisão (`_mascara.png`) e Flood-Fill (`_distancias.npz`).
    """
    fname_lower = os.path.basename(img_path).lower()
    if fname_lower in CALIBRATED_MAPS:
        cal = CALIBRATED_MAPS[fname_lower]
        return {
            "id": cal["id"],
            "nome": cal["nome"],
            "mapa_file": img_path,
            "mascara_file": os.path.join(SPRITES_DIR, cal["mascara_name"]),
            "npz_file": os.path.join(BASE_DIR, cal["npz_name"]),
            "start_x": cal["start_x"],
            "start_y": cal["start_y"],
            "start_angle": cal["start_angle"],
            "auto_obstacles": cal["auto_obstacles"],
        }

    pasta = os.path.dirname(img_path)
    nome_base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(pasta, f"{nome_base}_mascara.png")
    npz_path = os.path.join(BASE_DIR, f"matrizDistancias_{nome_base}.npz")

    if not os.path.exists(npz_path):
        print(f"[Gerador de Mapas] Extraindo matriz de colisão e Flood-Fill para novo mapa: {nome_base}...")
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.int32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Filtro cromático calibrado para separar a areia bege da pista do solo laranja-avermelhado do deserto
        is_sand_track = (r >= 175) & (g >= 132) & (b >= 68) & ((r - g) <= 88) & ((g - b) <= 95)
        is_gray_road = (np.abs(r - 110) <= 18) & (np.abs(g - 110) <= 18) & (np.abs(b - 110) <= 18)
        raw_mask = np.where(is_sand_track | is_gray_road, 255, 0).astype(np.uint8)

        pil_m = Image.fromarray(raw_mask)
        pil_m = pil_m.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
        cenario = (np.array(pil_m) > 127).astype(np.uint8)

        ys, xs = np.where(cenario == 1)
        if len(xs) > 0:
            idx_top = int(np.argmin(ys + np.abs(xs - cenario.shape[1] // 3)))
            sx, sy = int(xs[idx_top]), int(ys[idx_top]) + 25
        else:
            sx, sy = cenario.shape[1] // 3, cenario.shape[0] // 4

        distancias, maior_dist = compute_bfs_flood_fill(cenario, sx, sy, 0.0)
        Image.fromarray(cenario * 255).save(mask_path)
        np.savez_compressed(
            npz_path,
            cenario=cenario,
            distancias=distancias,
            maior_distancia=maior_dist,
            start_x=sx,
            start_y=sy,
            start_angle=0.0,
        )

    data = np.load(npz_path)
    sx = float(data["start_x"]) if "start_x" in data else 845.0
    sy = float(data["start_y"]) if "start_y" in data else 225.0
    sang = float(data["start_angle"]) if "start_angle" in data else 0.0

    return {
        "id": nome_base,
        "nome": nome_base.replace("_", " ").title(),
        "mapa_file": img_path,
        "mascara_file": mask_path,
        "npz_file": npz_path,
        "start_x": sx,
        "start_y": sy,
        "start_angle": sang,
        "auto_obstacles": True,
    }


def obter_catalogo_mapas() -> list[dict]:
    """
    Retorna a lista de mapas disponíveis:
      1. O mapa principal `Sprites/mapa.jpg`
      2. Os mapas `mapa2.jpg`, `mapa3.jpg`, `mapa4.jpg` e qualquer outro colocado em `Sprites/` ou `Sprites/mapas_custom/`!
    """
    catalogo = [
        processar_mapa_do_usuario(os.path.join(SPRITES_DIR, "mapa.jpg"))
    ]

    # 1. Procura novos mapas colocados diretamente na pasta `Sprites/` (ex: mapa2.jpg, mapa3.png, etc.)
    if os.path.isdir(SPRITES_DIR):
        for fname in sorted(os.listdir(SPRITES_DIR)):
            fl = fname.lower()
            if not fl.endswith((".png", ".jpg", ".jpeg")):
                continue
            if fl in CHARACTER_FILENAMES or fl == "mapa.jpg" or fl.endswith("_mascara.png") or fl.endswith("_mask.png"):
                continue
            full_p = os.path.join(SPRITES_DIR, fname)
            if os.path.isfile(full_p):
                try:
                    catalogo.append(processar_mapa_do_usuario(full_p))
                except Exception as exc:
                    print(f"[Aviso] Erro ao carregar mapa {fname}: {exc}")

    # 2. Procura novos mapas colocados na subpasta `Sprites/mapas_custom/`
    if os.path.isdir(CUSTOM_MAPS_DIR):
        for fname in sorted(os.listdir(CUSTOM_MAPS_DIR)):
            fl = fname.lower()
            if fl.endswith((".png", ".jpg", ".jpeg")) and not fl.endswith("_mascara.png"):
                full_p = os.path.join(CUSTOM_MAPS_DIR, fname)
                if os.path.isfile(full_p):
                    try:
                        catalogo.append(processar_mapa_do_usuario(full_p))
                    except Exception as exc:
                        print(f"[Aviso] Erro ao carregar mapa customizado {fname}: {exc}")

    return catalogo
