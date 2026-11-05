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

