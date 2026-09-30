# Resumo para portfólio

**Intelligent Maintenance Decision System — MDP & Reinforcement Learning**

Projeto de tomada de decisão de manutenção orientada por degradação utilizando NASA C-MAPSS. O pipeline transforma sinais multivariados de sensores em um Health Index por normalização, seleção de variáveis e PCA; discretiza a condição do equipamento em estados; estima probabilidades de transição; e compara uma política baseline com Value Iteration e Q-Learning.

No cenário-base, Value Iteration e Q-Learning convergiram para a mesma política operacional e superaram a baseline no retorno descontado modelado nos quatro subconjuntos FD001–FD004. O projeto também evidencia o trade-off entre menor frequência de manutenção e maior exposição a falhas.

**Stack:** Python, Pandas, NumPy, SciPy, scikit-learn, PCA, K-Means, MDP, Value Iteration, Q-Learning, Jupyter.
