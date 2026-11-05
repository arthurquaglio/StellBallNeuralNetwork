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

