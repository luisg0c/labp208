# Lab P2-08 - Alinhamento com DPO

## Sobre o beta

O hiperparametro `beta` (β = 0.1) controla o quanto a politica treinada (π_θ) pode se afastar do modelo de referencia (π_ref) durante o treino. Matematicamente ele aparece como coeficiente da divergencia KL implicita no loss do DPO:

    L = -log σ(β · (log π_θ(chosen|x)/π_ref(chosen|x) - log π_θ(rejected|x)/π_ref(rejected|x)))

Funciona como um "imposto" sobre a distancia entre o modelo ator e o de referencia. Beta alto mantem o modelo preso ao comportamento do SFT original — preserva fluencia mas aprende pouco da preferencia. Beta baixo da liberdade total, com risco de o modelo colapsar em respostas repetitivas ou perder coerencia. O valor 0.1 eh o padrao do paper original (Rafailov et al. 2023) — meio-termo que permite mudanca de comportamento sem destruir a capacidade generativa herdada do Lab P1-07.
