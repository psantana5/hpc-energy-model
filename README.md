# Modelado y Predicción del Consumo Energético en Clústeres HPC

## Descripción del Proyecto

Este proyecto de Trabajo de Fin de Grado (TFG) en Ingeniería Informática tiene como objetivo desarrollar un sistema completo para el modelado y predicción del consumo energético en clústeres de computación de alto rendimiento (HPC) mediante el análisis de patrones térmicos y carga de trabajo.

### Objetivos Principales

- **Recolección de métricas**: Capturar datos térmicos, de carga de trabajo y consumo energético de jobs HPC
- **Modelado predictivo**: Desarrollar algoritmos de machine learning para predecir el consumo energético por job
- **Visualización avanzada**: Crear dashboards interactivos con Grafana para análisis de correlaciones
- **Optimización energética**: Proporcionar recomendaciones inteligentes para la colocación eficiente de jobs

## Arquitectura del Sistema

### Infraestructura Base
- **Proxmox**: Plataforma de virtualización con CPU passthrough
- **Slurm**: Gestor de colas y recursos HPC
- **Prometheus**: Sistema de monitorización y alertas
- **TimescaleDB**: Base de datos optimizada para series temporales
- **Grafana**: Plataforma de visualización y dashboards

### Componentes del Clúster
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Slurm Head    │    │  Compute Node 1 │    │  Compute Node 2 │
│   Controller    │    │                 │    │                 │
│                 │    │  - Node Exporter│    │  - Node Exporter│
│ - Prometheus    │    │  - Thermal Mon. │    │  - Thermal Mon. │
│ - TimescaleDB   │    │  - Job Scripts  │    │  - Job Scripts  │
│ - Grafana       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Dataset y Métricas

Cada job ejecutado genera las siguientes métricas:

| Métrica | Descripción | Unidad |
|---------|-------------|--------|
| `timestamp` | Marca temporal de ejecución | Unix timestamp |
| `job_id` | Identificador único del job | String |
| `job_type` | Tipo de carga (CPU/IO/Mixed) | Enum |
| `node_temp_avg` | Temperatura media del nodo | °C |
| `node_temp_peak` | Pico térmico durante ejecución | °C |
| `duration` | Duración total del job | Segundos |
| `cpu_freq_avg` | Frecuencia media de CPU | MHz |
| `energy_consumption` | Consumo energético estimado | Watts |
| `memory_usage` | Uso de memoria promedio | MB |
| `cpu_utilization` | Utilización de CPU promedio | % |

## Estructura del Proyecto

```
hpc-energy-model/
├── README.md
├── LICENSE
├── requirements.txt
├── docker-compose.yml
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   └── api-reference.md
├── infrastructure/
│   ├── proxmox/
│   │   ├── vm-templates/
│   │   └── network-config/
│   ├── slurm/
│   │   ├── slurm.conf
│   │   ├── slurmdbd.conf
│   │   └── job-templates/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── monitoring/
│   ├── exporters/
│   │   ├── thermal-exporter/
│   │   ├── job-exporter/
│   │   └── energy-exporter/
│   └── collectors/
├── workloads/
│   ├── cpu-intensive/
│   ├── io-intensive/
│   ├── mixed-workloads/
│   └── benchmark-suite/
├── ml-models/
│   ├── data-preprocessing/
│   ├── feature-engineering/
│   ├── training/
│   └── inference/
├── api/
│   ├── energy-predictor/
│   ├── job-scheduler/
│   └── recommendation-engine/
├── scripts/
│   ├── setup/
│   ├── data-collection/
│   └── automation/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
└── results/
    ├── datasets/
    ├── models/
    └── reports/
```

## Instalación y Configuración

### Prerrequisitos
- Proxmox VE 7.0+
- Python 3.9+
- Docker y Docker Compose
- Al menos 16GB RAM y 4 cores CPU

### Configuración Rápida

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/psantana5/hpc-energy-model.git
   cd hpc-energy-model
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar infraestructura**:
   ```bash
   ./scripts/setup/deploy-infrastructure.sh
   ```

4. **Inicializar servicios**:
   ```bash
   docker-compose up -d
   ```

## Metodología de Desarrollo

### Fase 1: Preparación del Entorno (Semanas 1-3)
- [ ] Configuración de VMs en Proxmox
- [ ] Instalación y configuración de Slurm
- [ ] Despliegue de stack de monitorización
- [ ] Desarrollo de exportadores personalizados

### Fase 2: Recolección de Datos (Semanas 4-6)
- [ ] Implementación de workloads sintéticos
- [ ] Automatización de ejecución de jobs
- [ ] Validación de métricas recolectadas
- [ ] Generación de dataset inicial

### Fase 3: Análisis y Modelado (Semanas 7-10)
- [ ] Análisis exploratorio de datos
- [ ] Ingeniería de características
- [ ] Entrenamiento de modelos predictivos
- [ ] Validación y optimización de modelos

### Fase 4: Visualización y Recomendaciones (Semanas 11-12)
- [ ] Desarrollo de dashboards en Grafana
- [ ] Implementación de motor de recomendaciones
- [ ] API para predicciones en tiempo real
- [ ] Documentación y presentación final

## Tecnologías Utilizadas

- **Orquestación**: Slurm, Proxmox
- **Monitorización**: Prometheus, Node Exporter, exportadores personalizados
- **Base de Datos**: TimescaleDB, PostgreSQL
- **Visualización**: Grafana, Jupyter Notebooks
- **Machine Learning**: scikit-learn, pandas, numpy
- **Desarrollo**: Python, Bash, Docker
- **Testing**: pytest, unittest

## Resultados Esperados

- **Dataset completo** con +10,000 jobs ejecutados
- **Modelo predictivo** con precisión >85% en consumo energético
- **Dashboard interactivo** para análisis en tiempo real
- **Motor de recomendaciones** para optimización de scheduling
- **Documentación técnica** completa del sistema

## Contribución

Este proyecto está desarrollado como TFG. Para sugerencias o mejoras:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Autor

**[Pau Santana]** - Estudiante de Ingeniería Informática  
Universidad: [UOC (Universitat Oberta de Catalunya)]  
Email: [tu-email@universidad.edu]

## Agradecimientos

- Director/a del TFG: [Nombre del director]
- Departamento de [Nombre del departamento]
- Comunidad open source de Slurm, Prometheus y Grafana

---

*Proyecto desarrollado como Trabajo de Fin de Grado en Ingeniería Informática*