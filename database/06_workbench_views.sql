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
