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
