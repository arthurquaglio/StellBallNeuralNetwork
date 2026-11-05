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
