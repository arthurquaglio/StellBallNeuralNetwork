# -*- coding: utf-8 -*-
"""
Configurações globais do simulador Steel Ball Run — Neural Network & Genetic Evolution Edition.
Arquitetura neuroevolutiva 100% persistida e processada no MySQL Server (sbr_neural_db).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITES_DIR = os.path.join(BASE_DIR, "Sprites")
PROCESSED_SPRITES_DIR = os.path.join(SPRITES_DIR, "processed")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

# Carrega variáveis locais do arquivo .env (ignorado pelo Git) se existir
_ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.isfile(_ENV_PATH):
    with open(_ENV_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ============================================================================
# CONFIGURAÇÃO DO BANCO DE DADOS MYSQL (MySQL Workbench - serverPrincipal)
# ============================================================================
MYSQL_CONFIG = {
    "host": os.environ.get("SBR_DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("SBR_DB_PORT", "3306")),
    "user": os.environ.get("SBR_DB_USER", "root"),
    "password": os.environ.get("SBR_DB_PASSWORD", ""),
    "database": os.environ.get("SBR_DB_NAME", "sbr_neural_db"),
}

ARCHITECTURE_NAME = "sbr_mlp_18_6_4"

# ============================================================================
# HIPERPARÂMETROS DA REDE NEURAL E ALGORITMO GENÉTICO (Steel Ball Run)
# ============================================================================
QTD_CAVALOS_PADRAO = 250       # População simultânea fluida a 60 FPS
SBR_BRAIN_QTD_LAYERS = 1       # 1 camada oculta
SBR_BRAIN_QTD_INPUT = 18       # 17 sensores Raycast (-90° a +90°) + 1 Velocidade
SBR_BRAIN_QTD_HIDE = 6         # 6 neurônios na camada escondida (+ 1 BIAS)
SBR_BRAIN_QTD_OUTPUT = 4       # 4 saídas: [0]=Galopar, [1]=Frear, [2]=Virar Esq, [3]=Virar Dir
ELITE_STEP = 5                 # Top 5 indivíduos clonados a cada geração (Elitismo Step = 5)

# Total de parâmetros do DNA: (18 * 6 pesos entrada->oculta) + (6 biases ocultos)
#                           + (6 * 4 pesos oculta->saída)   + (4 biases saída) = 142 genes ativos
TAMANHO_DNA = (
    (SBR_BRAIN_QTD_INPUT * SBR_BRAIN_QTD_HIDE)
    + SBR_BRAIN_QTD_HIDE
    + (SBR_BRAIN_QTD_HIDE * SBR_BRAIN_QTD_OUTPUT)
    + SBR_BRAIN_QTD_OUTPUT
)

# ============================================================================
# PARÂMETROS DO CENÁRIO E FÍSICA DA CORRIDA (Mapa Steel Ball Run 2752x1536)
# ============================================================================
LARGURA_TELA = 1440
ALTURA_TELA = 840
FPS_ALVO = 60

# Posição de largada logo após o pórtico "START" superior esquerdo (sentido horário)
START_X = 845.0
START_Y = 225.0
START_ANGLE = 23.0             # Graus (apontando para a descida do cânion superior)

# Personagens disponíveis na pasta Sprites/
RIDERS_INFO = [
    {"key": "gyro",      "name": "Gyro Zeppeli (Valkyrie)",      "file": "gyro.png",      "color": (155, 89, 182)},
    {"key": "jonny",     "name": "Johnny Joestar (Slow Dancer)", "file": "jonny.png",     "color": (52, 152, 219)},
    {"key": "dio",       "name": "Diego Brando (Silver Bullet)", "file": "dio.png",       "color": (241, 196, 15)},
    {"key": "pocoloco",  "name": "Pocoloco (Hey Ya!)",           "file": "pocoloco.png",  "color": (230, 126, 34)},
    {"key": "valentine", "name": "Funny Valentine (D4C)",        "file": "valentine.png", "color": (231, 76, 60)},
]
