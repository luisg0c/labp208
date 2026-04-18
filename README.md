# Lab P2-08 - Alinhamento com DPO

Pipeline de alinhamento do adapter do Lab P1-07 usando Direct Preference Optimization.
O dataset contem 35 pares de preferencia sobre solicitacoes inadequadas no dominio de musica eletronica (pirataria, falsificacao, sabotagem, assedio). O modelo aprende a recusar essas solicitacoes.

## Sobre o beta

O hiperparametro `beta` (β = 0.1) controla o quanto a politica treinada (π_θ) pode se afastar do modelo de referencia (π_ref) durante o treino. Matematicamente ele aparece como coeficiente da divergencia KL implicita no loss do DPO:

    L = -log σ(β · (log π_θ(chosen|x)/π_ref(chosen|x) - log π_θ(rejected|x)/π_ref(rejected|x)))

Funciona como um "imposto" sobre a distancia entre o modelo ator e o de referencia. Beta alto mantem o modelo preso ao comportamento do SFT original — preserva fluencia mas aprende pouco da preferencia. Beta baixo da liberdade total, com risco de o modelo colapsar em respostas repetitivas ou perder coerencia. O valor 0.1 eh o padrao do paper original (Rafailov et al. 2023) — meio-termo que permite mudanca de comportamento sem destruir a capacidade generativa herdada do Lab P1-07.

## Dependencias

    pip install torch transformers peft trl datasets bitsandbytes openai

## Como rodar

Gerar dataset (precisa de OPENAI_API_KEY no .env):

    python gera_dpo.py

Rodar alinhamento (precisa do adapter do Lab P1-07 em `../labp107/adapter/`):

    python dpo.py

## Uso de IA

Ferramenta usada: Claude Sonnet 4.6

- Revisao da explicacao matematica do beta no README
- Debug do erro de padding_side no DPOTrainer (tem que ser "left", nao "right" como no SFT)
