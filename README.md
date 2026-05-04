# Heart Failure Mortality Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Modelo com Machine Learning desenvolvido para analisar registros clínicos e prever a probabilidade de mortalidade em pacientes com insuficiência cardíaca.

---

## 🔬 Visão Geral

A insuficiência cardíaca é um evento clínico complexo, e a identificação precoce de pacientes em risco é fundamental para o direcionamento médico assertivo. Este projeto utiliza um conjunto de dados clínicos com 12 variáveis (como fração de ejeção, creatinina sérica, idade e pressão arterial) para treinar e avaliar modelos preditivos de classificação.

O objetivo principal é demonstrar a aplicação de técnicas de ciência de dados na extração de padrões de saúde e no suporte à decisão baseada em evidências, focando em métricas críticas para o contexto hospitalar.

## 🛠️ Metodologia e Funcionalidades

Diferente de abordagens genéricas, este projeto prioriza a identificação correta de pacientes em risco (sensibilidade), visando minimizar falsos negativos em um cenário clínico.

* **Análise Exploratória de Dados (EDA):** Mapeamento de distribuições, análise de covariância e identificação da correlação entre marcadores biológicos e o desfecho de sobrevivência.
* **Engenharia de Atributos:** Limpeza de dados, tratamento de outliers e normalização de variáveis contínuas (ex: níveis de plaquetas e creatinina).
* **Modelagem Preditiva:** Implementação e comparação de performance entre múltiplos algoritmos:
    * Regressão Logística (Baseline)
    * Support Vector Machines (SVM)
    * K-Nearest Neighbors (KNN)
* **Avaliação de Desempenho:** Foco em métricas de triagem clínica, utilizando *Recall* (Sensibilidade), *Precision* e *F1-score*, além da análise detalhada via matriz de confusão.

## 💻 Stack de tecnologias utilizadas

| Ferramenta | Aplicação no Projeto |
| :--- | :--- |
| **Python** | Linguagem base do pipeline de desenvolvimento |
| **Pandas / NumPy** | Estruturação, limpeza e cálculos vetoriais |
| **Scikit-learn** | Construção, treinamento e validação dos modelos ML |
| **Matplotlib / Seaborn** | Geração de gráficos estatísticos e heatmaps de correlação |

## 📁 Estrutura do Repositório

```text
.
├── data/
│   └── heart_failure_clinical_records_dataset.csv  # Dataset original
├── notebooks/
│   └── eda_and_prototyping.ipynb                   # Análises e prototipagem
├── src/
│   └── main.py                                     # Script principal de execução
├── requirements.txt                                # Dependências do projeto
└── README.md
```

Projeto baseado no repositório do usuário [tkarim45](https://github.com/tkarim45/Beginner-Data-Science-Projects).
