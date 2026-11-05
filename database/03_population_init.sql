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

