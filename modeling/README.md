# High-Level Modeling (HLM) Module

Este módulo implementa un sistema de modelado de alto nivel para simular el comportamiento térmico y energético de clústeres HPC sin necesidad de ejecutar la infraestructura real.

## 🎯 Objetivo

El módulo HLM permite:
- **Simular** el comportamiento energético y térmico del clúster HPC
- **Predecir** el consumo energético de trabajos antes de ejecutarlos
- **Analizar** patrones térmicos y optimizar la eficiencia energética
- **Validar** modelos contra datos reales del sistema instrumentado
- **Generar** reportes y visualizaciones para análisis

## 📁 Estructura del Módulo

```
modeling/
├── __init__.py                 # Punto de entrada del módulo
├── main.py                     # Script principal de ejecución
├── modeling_config.yaml        # Configuración del sistema
├── README.md                   # Esta documentación
│
├── core/                       # Componentes principales
│   ├── __init__.py
│   ├── data_loader.py          # Carga de datos históricos
│   └── simulation_engine.py    # Motor de simulación
│
├── models/                     # Modelos predictivos
│   ├── __init__.py
│   ├── energy_predictor.py     # Predicción de energía
│   └── thermal_predictor.py    # Predicción térmica
│
├── utils/                      # Utilidades
│   ├── __init__.py
│   └── config.py               # Gestión de configuración
│
└── validation/                 # Validación y métricas
    ├── __init__.py
    └── validator.py            # Validación de modelos
```

## 🚀 Uso Rápido

### 1. Configuración

Edita el archivo `modeling_config.yaml` con los parámetros de tu entorno:

```yaml
database:
  host: "localhost"
  port: 5432
  database: "hpc_energy_db"
  username: "tu_usuario"
  password: "tu_password"

simulation:
  num_nodes: 10
  cores_per_node: 16
  memory_per_node_gb: 64
```

### 2. Ejecución Básica

```bash
# Desde el directorio raíz del proyecto
cd modeling

# Ejecutar pipeline completo
python main.py --config modeling_config.yaml

# Solo validación
python main.py --config modeling_config.yaml --validation-only

# Simulación personalizada
python main.py --config modeling_config.yaml --num-jobs 100 --duration 3600
```

### 3. Uso Programático

```python
from modeling import ModelingConfig, HistoricalDataLoader, HPCClusterSimulator
from modeling.models import EnergyPredictor, ThermalPredictor
from modeling.validation import ModelValidator

# Cargar configuración
config = ModelingConfig.from_yaml('modeling_config.yaml')

# Cargar datos históricos
data_loader = HistoricalDataLoader(config)
job_data = data_loader.load_job_metrics()

# Entrenar modelos
energy_model = EnergyPredictor(config)
energy_model.train(job_data)

thermal_model = ThermalPredictor(config)
thermal_model.train(job_data)

# Crear simulador
simulator = HPCClusterSimulator(config, energy_model, thermal_model)

# Ejecutar simulación
results = simulator.run_simulation(duration_hours=24)

# Validar resultados
validator = ModelValidator(config)
validation_report = validator.validate_simulation_results(
    real_data=job_data,
    simulated_data=results
)
```

## 🔧 Componentes Principales

### 1. Data Loader (`core/data_loader.py`)

- Conecta a la base de datos TimescaleDB
- Carga métricas históricas de trabajos y nodos
- Preprocesa y limpia los datos
- Exporta datos en múltiples formatos

### 2. Simulation Engine (`core/simulation_engine.py`)

- Simula el comportamiento del clúster usando SimPy
- Modela recursos, scheduling y ejecución de trabajos
- Implementa modelos térmicos y de potencia
- Genera métricas de simulación

### 3. Energy Predictor (`models/energy_predictor.py`)

- Predice consumo energético basado en características del trabajo
- Soporta múltiples algoritmos (Random Forest, Gradient Boosting, etc.)
- Incluye análisis de importancia de características
- Proporciona intervalos de confianza

### 4. Thermal Predictor (`models/thermal_predictor.py`)

- Predice comportamiento térmico de los nodos
- Modela temperatura promedio, pico y varianza
- Analiza eficiencia térmica
- Genera perfiles de temperatura temporales

### 5. Model Validator (`validation/validator.py`)

- Valida modelos contra datos reales
- Calcula métricas estadísticas (MAE, RMSE, R², etc.)
- Genera reportes de validación
- Crea visualizaciones comparativas

## 📊 Salidas del Sistema

### Reportes Generados

1. **Reporte de Entrenamiento** (`training_report.json`)
   - Métricas de rendimiento de modelos
   - Importancia de características
   - Parámetros optimizados

2. **Reporte de Simulación** (`simulation_report.json`)
   - Resultados de la simulación
   - Métricas de utilización de recursos
   - Patrones energéticos y térmicos

3. **Reporte de Validación** (`validation_report.json`)
   - Comparación modelo vs. realidad
   - Métricas de precisión
   - Análisis de errores

4. **Reporte Markdown** (`comprehensive_report.md`)
   - Resumen ejecutivo
   - Análisis detallado
   - Recomendaciones

### Visualizaciones

- Gráficos de predicción vs. realidad
- Distribuciones de errores
- Importancia de características
- Perfiles temporales de energía y temperatura
- Mapas de calor de utilización

### Datos Exportados

- Predicciones de energía (`energy_predictions.parquet`)
- Predicciones térmicas (`thermal_predictions.parquet`)
- Métricas de simulación (`simulation_metrics.parquet`)
- Datos de validación (`validation_data.parquet`)

## ⚙️ Configuración Avanzada

### Parámetros de Simulación

```yaml
simulation:
  # Especificaciones del clúster
  num_nodes: 10
  cores_per_node: 16
  memory_per_node_gb: 64
  
  # Parámetros temporales
  time_step_seconds: 60
  warmup_time_minutes: 30
  
  # Condiciones ambientales
  ambient_temperature_c: 22.0
  cooling_efficiency: 0.85
  
  # Modelado térmico
  thermal_time_constant: 300
  max_safe_temperature: 85.0
```

### Configuración de Modelos

```yaml
model:
  # Algoritmos a evaluar
  algorithms:
    - "random_forest"
    - "gradient_boosting"
    - "linear"
    - "ridge"
  
  # Optimización de hiperparámetros
  hyperparameter_tuning: true
  tuning_iterations: 50
  
  # Umbrales de rendimiento
  min_r2_score: 0.7
  max_mape_percent: 15.0
```

### Configuración de Salidas

```yaml
output:
  # Formatos de archivo
  data_format: "parquet"
  plot_format: "png"
  report_format: "markdown"
  
  # Configuración de logging
  log_level: "INFO"
  log_to_file: true
  
  # Visualizaciones
  plot_style: "seaborn"
  figure_size: [12, 8]
  dpi: 300
```

## 🔍 Validación y Métricas

### Métricas Estadísticas

- **MAE** (Mean Absolute Error): Error absoluto promedio
- **RMSE** (Root Mean Square Error): Raíz del error cuadrático medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto promedio
- **R²** (Coefficient of Determination): Coeficiente de determinación
- **Correlación de Pearson**: Correlación lineal
- **Test de Kolmogorov-Smirnov**: Similitud de distribuciones

### Umbrales de Calidad

- **Excelente**: R² ≥ 0.9, MAPE ≤ 5%
- **Bueno**: R² ≥ 0.8, MAPE ≤ 10%
- **Aceptable**: R² ≥ 0.7, MAPE ≤ 15%
- **Pobre**: R² < 0.7, MAPE > 15%

## 🛠️ Dependencias

### Librerías Principales

```python
# Análisis de datos
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Simulación
import simpy

# Base de datos
import psycopg2
import sqlalchemy

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Utilidades
import yaml
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
```

### Instalación de Dependencias

```bash
# Instalar dependencias específicas del módulo HLM
pip install simpy psycopg2-binary sqlalchemy pandas numpy scikit-learn matplotlib seaborn pyyaml joblib
```

## 🚨 Consideraciones Importantes

### Limitaciones

1. **Calidad de Datos**: Los modelos son tan buenos como los datos históricos
2. **Representatividad**: Los datos de entrenamiento deben ser representativos
3. **Deriva del Modelo**: Los modelos pueden degradarse con el tiempo
4. **Complejidad Computacional**: Simulaciones grandes requieren recursos significativos

### Mejores Prácticas

1. **Validación Regular**: Re-entrenar modelos periódicamente
2. **Monitoreo**: Supervisar la precisión de las predicciones
3. **Datos de Calidad**: Asegurar datos limpios y completos
4. **Configuración Apropiada**: Ajustar parámetros según el entorno

### Troubleshooting

#### Error de Conexión a Base de Datos
```bash
# Verificar conectividad
psql -h localhost -p 5432 -U postgres -d hpc_energy_db
```

#### Memoria Insuficiente
```yaml
# Reducir chunk_size en configuración
advanced:
  chunk_size: 5000
  memory_limit_gb: 4
```

#### Modelos con Baja Precisión
```yaml
# Ajustar parámetros de entrenamiento
model:
  hyperparameter_tuning: true
  tuning_iterations: 100
  cross_validation_folds: 10
```

## 📈 Casos de Uso

### 1. Planificación de Capacidad

```python
# Simular diferentes configuraciones de clúster
configs = [
    {'num_nodes': 10, 'cores_per_node': 16},
    {'num_nodes': 20, 'cores_per_node': 8},
    {'num_nodes': 15, 'cores_per_node': 12}
]

for config in configs:
    simulator = HPCClusterSimulator(config)
    results = simulator.run_simulation(duration_hours=168)  # 1 semana
    print(f"Configuración {config}: Eficiencia energética = {results['energy_efficiency']}")
```

### 2. Optimización de Scheduling

```python
# Comparar algoritmos de scheduling
scheduling_algorithms = ['fifo', 'backfill', 'fair_share']

for algorithm in scheduling_algorithms:
    config.simulation.scheduling_algorithm = algorithm
    simulator = HPCClusterSimulator(config)
    results = simulator.run_simulation(duration_hours=24)
    print(f"Algoritmo {algorithm}: Tiempo promedio de espera = {results['avg_wait_time']}")
```

### 3. Análisis de Eficiencia Térmica

```python
# Analizar diferentes temperaturas ambientales
ambient_temps = [18, 20, 22, 24, 26]

for temp in ambient_temps:
    config.simulation.ambient_temperature_c = temp
    simulator = HPCClusterSimulator(config)
    results = simulator.run_simulation(duration_hours=12)
    print(f"Temp ambiente {temp}°C: Temp promedio CPU = {results['avg_cpu_temp']}°C")
```

## 🤝 Integración con el Sistema Existente

El módulo HLM está diseñado para ser **completamente no intrusivo**:

- ✅ **No modifica** la infraestructura existente
- ✅ **No interfiere** con el flujo de recolección de métricas
- ✅ **Utiliza** los mismos datos que genera el sistema real
- ✅ **Complementa** el sistema actual sin reemplazarlo
- ✅ **Se ejecuta** independientemente del clúster real

### Flujo de Datos

```
Clúster HPC Real → TimescaleDB → Módulo HLM → Reportes y Predicciones
     ↓                ↑
Métricas Reales   Datos Históricos
```

## 📚 Referencias y Recursos

- [SimPy Documentation](https://simpy.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [HPC Energy Modeling Best Practices](https://www.energy.gov/eere/buildings/downloads/best-practices-hpc-energy-modeling)

---

**Nota**: Este módulo es parte del proyecto HPC Energy Model y está diseñado para trabajar con los datos generados por el sistema de monitoreo existente. Para soporte técnico o preguntas, consulta la documentación principal del proyecto.