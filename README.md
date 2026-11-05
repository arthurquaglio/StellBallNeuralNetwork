# Steel Ball Run — Neural Network Simulator (MySQL Workbench Edition)

Simulador de Neuroevolução e Algoritmos Genéticos ambientado na corrida transcontinental de **JoJo's Bizarre Adventure: Steel Ball Run**, com 100% da persistência de genomas, pesos sinápticos, biases e evolução genética executados dentro do **MySQL Server 8.0 (`sbr_neural_db`)** via **MySQL Workbench**.

---

## 1. Como Executar o Simulador (1 Clique)

Basta dar duplo clique em **`INICIAR_SIMULADOR.bat`** ou rodar no terminal dentro da pasta `D:\Aplicativos_Desenvolvimento\StellBallNeuralNetwork`:

```powershell
python main.py 250
```
*(Você pode passar na linha de comando a quantidade de cavalos da população, ex: `python main.py 300` ou `python main.py 500`.)*

---

## 2. Como Adicionar e Trocar de Mapas

Você mesmo coloca os sprites dos mapas! Basta adicionar qualquer imagem de mapa (`.jpg` ou `.png`, ex: `mapa2.jpg`, `mapa3.png`) diretamente na pasta **`Sprites/`** (ou em **`Sprites/mapas_custom/`**).

O módulo [`gerador_mapas.py`](file:///D:/Aplicativos_Desenvolvimento/StellBallNeuralNetwork/gerador_mapas.py) detecta automaticamente os mapas que você colocar na pasta e gera sozinho:
- A máscara binária de colisão (`<nome>_mascara.png`),
- A matriz geodésica de distâncias via BFS Flood-Fill de 8 vizinhos (`matrizDistancias_<nome>.npz`),
- Os 17 obstáculos em slalom e 8 zonas de terreno ao longo da trilha!

### Como trocar de mapa durante o jogo:
* Pressione a tecla **`TAB`** para alternar para o próximo mapa da sua pasta `Sprites/`.
* Pressione as teclas **`1`**, **`2`**, **`3`**, etc. para selecionar diretamente o mapa pelo número.
* Ou **clique com o mouse** diretamente sobre o botão dourado **`Mapa (TAB)`** no painel superior direito do HUD.

---

## 3. Regra Oficial do Campeão da Geração (100% no MySQL Workbench)

Um cavalo **só é declarado Campeão da Geração se realmente concluir 100% da corrida (`concluiu_corrida = 1`)**:
- Caso nenhum cavalo da geração alcance a linha de chegada, tanto o HUD quanto o MySQL registram: **`Nenhum (Não Concluiu)`** (`campeao_genome_id = NULL`, `concluiu_corrida = 0`).
- Quando 1 ou mais cavalos cruzam a linha de chegada, o **Campeão da Geração** é aquele que completou o circuito no menor tempo em ticks (`ticks_conclusao`), exibindo no HUD e gravando no banco quem venceu, seu ID de genoma, seu tempo em ticks e quantos cavalos concluíram aquela bateria.

### Consultas Prontas para Estudo no MySQL Workbench:
```sql
USE sbr_neural_db;

-- 1. Evolução de todas as gerações (mostra quantos concluíram e o Campeão APENAS se concluiu)
SELECT * FROM vw_sbr_generation_evolution;

-- 2. Galeria Exclusiva dos Campeões de Geração (lista apenas gerações onde houve vencedor!)
SELECT * FROM vw_sbr_generation_champions;

-- 3. Placar (Leaderboard) dos cavalos da geração atual (com status CONCLUIU A CORRIDA)
SELECT * FROM vw_sbr_current_leaderboard LIMIT 30;

-- 4. Inspecionar os 132 pesos sinápticos e biases do cérebro do Campeão
SELECT * FROM vw_sbr_champion_brain;

-- 5. Ver os neurônios ativados em tempo real (Forward Pass no MySQL)
SELECT * FROM vw_sbr_live_brain_activations;
```

---

## 4. Atalhos do Simulador

| Tecla | Função |
| :--- | :--- |
| **`TAB`** ou **`1, 2, 3...`** | **Trocar de Mapa instantaneamente** (ou clique no botão de Mapa no HUD) |
| **`W, A, S, D`** | Movimentar a câmera pelo mapa |
| **`Q / E`** ou **Scroll** | Zoom In / Zoom Out |
| **`R`** | Resetar câmera para visão geral do mapa inteiro |
| **`G`** | Travar câmera automaticamente no **Melhor Cavalo Vivo (Líder)** |
| **`C`** | Mostrar / Ocultar os **17 Sensores Raycast** do cavalo líder |
| **`Z`** | Alternar entre **Mapa Colorido** e **Matriz de Colisão (`_mascara.png` + Silhuetas)** |
| **`X`** | Inspecionar os valores numéricos da **`matrizDistancias` (Flood-Fill)** ao redor do mouse |
| **`N / M`** | Aumentar (`2x`, `4x`, `8x`, `16x`) ou diminuir a velocidade da simulação |
| **`O`** | Ativar / Desativar Zonas (*Golden Spin Boost* / *Devil's Palm*) e Obstáculos (*Steel Balls*) |
| **`H`** | Mostrar / Ocultar o painel da Rede Neural |
| **`Setas`** | Controlar manualmente o Cavalo #0 para correr contra as Redes Neurais |
| **`F2`** | Carregar diretamente do MySQL o melhor **Campeão Oficial (que concluiu a corrida)** no Slot #0 |
| **`F5`** | Reconectar / Sincronizar imediatamente com o MySQL Server (`127.0.0.1:3306`) |
| **`F6`** | Resetar o banco MySQL (`sbr_neural_db`) para a **Geração 0** |
