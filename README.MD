# MVP

Dashboard para:

- Visualizar lotes con estado de frescura
- Calcular el riesgo de caducidad (%) por lote
  > - Simular restrasos de vuelos y cambios de consumo
- Muestra alertas (verde / amarrillo / rojo) y prioridades

## File structure
DigitalTwin_FreshnessAI/
│
├── data/
│   ├── expirations.xlsx   # Dataset original
│   ├── data_processed.csv                                        # Dataset limpio y enriquecido
│   └── zones_mapping.csv                                         # (Opcional) Mapeo de zonas simuladas
│
├── src/
│   ├── data_preparation.py       # Limpieza y creación de columnas derivadas
│   ├── freshness_model.py        # Cálculo de riesgo o predicción
    │   ├── simulation.py             # Lógica "What if..." (retrasos, consumo)
│   ├── db_service.py             # Gestión de datos (carga/guardado en CSV o SQLite)
│   └── __init__.py
│
├── dashboard/
│   ├── app.py                    # Dashboard principal (Streamlit / Gradio / etc.)
│   ├── components/               # Gráficos o módulos reutilizables
│   │   ├── charts.py
│   │   └── kpi_cards.py
│   └── assets/                   # Logos, colores, íconos
│       ├── gategroup_logo.png
│       └── style.css
│
├── notebooks/
    │   ├── exploration.ipynb         # Exploración inicial del dataset
│   └── model_tuning.ipynb        # Ajuste del score o modelo predictivo
│
├── docs/
│   ├── architecture_diagram.png  # Diagrama de flujo de datos
│   ├── pitch_slides.pdf          # Diapositivas de presentación
│   └── README.md                 # Guía rápida para correr el demo
│
├── requirements.txt              # Librerías necesarias
├── .gitignore                    # Ignorar datasets pesados o archivos temporales
└── run.sh                        # Script para ejecutar todo (opcional)
