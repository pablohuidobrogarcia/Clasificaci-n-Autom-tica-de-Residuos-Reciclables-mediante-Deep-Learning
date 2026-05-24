# Clasificación Automática de Residuos Reciclables mediante Deep Learning

**Trabajo Fin de Grado - Grado en Business Analytics**  
**Universidad Francisco de Vitoria - Curso Académico 2025-26**  
**Autor: Pablo Huidobro García**

---

## Descripción

Sistema de clasificación automática de envases de bebidas reciclables (latas de aluminio, botellas PET y botellas de vidrio) mediante técnicas de Deep Learning, orientado a su integración en máquinas de depósito de Sistemas de Depósito, Devolución y Retorno (SDDR) en el marco de la Ley 7/2022.

El proyecto está estructurado en tres pilares según la normativa del Grado en Business Analytics:

- **Pilar 1 - Ingeniería del Dato (20%):** Recopilación, preprocesamiento y preparación del dataset
- **Pilar 2 - Análisis de los Datos (20%):** Entrenamiento y comparación de modelos predictivos
- **Pilar 3 - Análisis de Negocio (20%):** Insights estratégicos, viabilidad económica y plan de implementación

---

## Estado del Proyecto

**Progreso General: 75% completado**

| Componente | Estado | Notas |
|------------|--------|-------|
| Pilar 1 - Ingeniería del Dato | ✅ Completo | Dataset preparado, scripts funcionales, análisis exploratorio |
| Pilar 2 - Análisis de Datos | ✅ Completo | 3 modelos entrenados y evaluados con métricas completas |
| Pilar 3 - Análisis de Negocio | ✅ Completo | Viabilidad técnica, económica y plan de implementación |
| Código completo integrado | ✅ Completo | Pipeline funcional que integra los 3 pilares |
| Documentos de calidad | ✅ Completo | Informe Turnitin y Taxonomía de uso de IA |
| Memoria de cada Pilar individualTFG | ✅ Completo 
| Memoria TFG | ✅ Completo
| Presentación defensa | ⏳ Pendiente | Pendiente de desarrollo |


---

## Estructura del Proyecto

```
TFG/
│
├── 01_Documentacion_academica/
│   ├── Clasificación Automática de Residuos Reciclables mediante Deep Learning Anteproyecto.pdf
│   ├── Memoria_TFG_FINAL.docx               
│   ├── Memoria_TFG_FINAL.pdf                
│   ├── Presentacion_defensa.pptx            [PENDIENTE]
│
├── 02_Ingenieria_de_Dato/                    [PILAR 1 - ✅ COMPLETO]
│   ├── Datos brutos/
│   │   ├── LATAS/
│   │   ├── PET/
│   │   └── VIDRIO/
│   ├── Splits/                               [Generado por scripts/Splits.py]
│   │   ├── train/
│   │   │   ├── latas/
│   │   │   ├── pet/
│   │   │   └── vidrio/
│   │   ├── validation/
│   │   │   ├── latas/
│   │   │   ├── pet/
│   │   │   └── vidrio/
│   │   ├── test/
│   │   │   ├── latas/
│   │   │   ├── pet/
│   │   │   └── vidrio/
│   │   └── split_metadata.json
│   ├── scripts/
│   │   └── Splits.py                         [Script de división del dataset]
│   ├── notebooks/
│   │   └── pilar1_ingenieria_dato.ipynb
│   ├── Resultados/
│   │   ├── distribucion_dataset.png
│   │   ├── augmentation_efecto.png
│   │   ├── resoluciones_originales.png  
│   │   ├── estadisticas_rgb.png
│   │   └── ejemplos_dataset.png
│   ├── split_indices.json                    [Índices del split — generado por Pilar 1]
│   ├── rgb_stats.json 
│   └── dataset_metadata.json
│
├── 03_Analisis_de_Datos_y_Modelado/          [PILAR 2 - ✅ COMPLETO]
│   ├── Pilar2_Analisis_Datos_Modelado.docx
│   ├── modelos/
│   │   ├── ResNet50_best.pth
│   │   ├── EfficientNet_B3_best.pth
│   │   ├── ConvNeXt_Tiny_best.pth
│   │   ├── historia_ResNet50.json
│   │   ├── historia_EfficientNet_B3.json
│   │   └── historia_ConvNeXt_Tiny.json
│   ├── notebook/
│   │   └── pilar2_analisis_datos.ipynb
│   └── resultados/
│       ├── comparativa_modelos.csv
│       ├── comparativa_final.png
│       ├── confusion_ResNet50.png
│       ├── confusion_EfficientNet_B3.png
│       ├── confusion_ConvNeXt_Tiny.png
│       ├── curvas_ResNet50.png
│       ├── curvas_EfficientNet_B3.png
│       ├── curvas_ConvNeXt_Tiny.png
│       └── resultados_completos.json
│       └── split_indices.json
│
├── 04_Normativas_tfg/
│   ├── Normativa TFG BA 2526 (1).pdf
│   ├── Plantilla TFG_Curso 2425.doc
│   └── Rúbricas.pdf
│
├── 05_Analisis_de_Negocio/                   [PILAR 3 - ✅ COMPLETO]
│   ├── Pilar3_Analisis_de_Negocio.docx
│   ├── Graficas/
│   │   ├── analisis_economico.png
│   │   ├── comparativa_multicriterio.png
│   │   ├── kpis_operativos.png
│   │   ├── matriz_riesgos.png
│   │   └── plan_implementacion_fases.png
│   ├── notebook/
│   │   └── pilar3_analisis_negocio1.ipynb
│   └── decision_modelo_produccion.json
│
├── 06_Codigo_Completo/                       [✅ COMPLETO]
│   ├── codigo_completo_final.py              [Script principal — integra 3 pilares]
│   ├── config.json
│   └── requirements.txt
│
└── 07_taxonomia_y_calidad/                   [✅ COMPLETO]
    ├── Informe_turnitin.pdf
    └── Taxonomia_uso_IA.pdf
```

---

## Requisitos del Sistema

- Python 3.10 o superior
- GPU con CUDA recomendada para entrenamiento (el entrenamiento se realizó en Kaggle con GPU T4)
- 8 GB RAM mínimo
- 10 GB espacio en disco (incluyendo modelos entrenados)

---

## Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <repositorio>
cd TFG
```

### 2. Instalar dependencias

```bash
# Instalación básica
pip install -r 06_Codigo_Completo/requirements.txt

# Instalación con soporte GPU (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r 06_Codigo_Completo/requirements.txt
```

---

## Ejecución

### Orden de ejecución obligatorio

Los pilares deben ejecutarse en orden ya que el Pilar 2 depende de los índices del split generados por el Pilar 1, y el Pilar 3 depende de los resultados generados por el Pilar 2.

```
Pilar 1  →  genera split_indices.json
Pilar 2  →  lee split_indices.json, genera resultados_completos.json
Pilar 3  →  lee resultados_completos.json, genera gráficas y análisis
```

### Paso 1: Ejecutar Pilar 1

```bash
# Desde la raíz del proyecto
jupyter notebook 02_Ingenieria_de_Dato/notebooks/pilar1_ingenieria_dato.ipynb
```

Este notebook:
- Analiza y visualiza la distribución del dataset
- Configura los transforms de entrenamiento y validación/test
- Genera el split reproducible (seed=42) y guarda los índices en `split_indices.json`
- Prepara los DataLoaders sin data leakage

### Paso 2: Ejecutar Pilar 2 (requiere GPU — recomendado en Kaggle)

```bash
jupyter notebook 03_Analisis_de_Datos_y_Modelado/notebook/pilar2_analisis_datos.ipynb
```

Este notebook:
- Carga los índices del split generados por el Pilar 1
- Entrena ResNet50, EfficientNet-B3 y ConvNeXt-Tiny con transfer learning desde ImageNet
- Evalúa los tres modelos en el conjunto de test
- Genera matrices de confusión, curvas de aprendizaje y comparativa de métricas
- Guarda los pesos de los modelos (`.pth`) y `resultados_completos.json`

> **Nota Kaggle:** El entrenamiento se realizó en Kaggle con GPU T4. Si usas Kaggle, utiliza el script unificado `06_Codigo_Completo/codigo_completo_final.py` y descarga los outputs desde `/kaggle/working/` al terminar.

### Paso 3: Ejecutar Pilar 3 (en local, no requiere GPU)

```bash
jupyter notebook 05_Analisis_de_Negocio/notebook/pilar3_analisis_negocio.ipynb
```

Este notebook:
- Carga `resultados_completos.json` del Pilar 2
- Traduce métricas técnicas a implicaciones de negocio
- Genera análisis de viabilidad técnica, económica y regulatoria
- Produce análisis de riesgos, plan de implementación por fases y KPIs operativos
- Guarda todas las gráficas en `05_Analisis_de_Negocio/Graficas/`

### Outputs generados

- Modelos entrenados: `03_Analisis_de_Datos_y_Modelado/modelos/`
- Resultados y métricas: `03_Analisis_de_Datos_y_Modelado/resultados/`
- Gráficas de negocio: `05_Analisis_de_Negocio/Graficas/`
- Decisión de modelo: `05_Analisis_de_Negocio/decision_modelo_produccion.json`

---

## Dataset

El dataset está compuesto por imágenes propias capturadas en condiciones variables para garantizar robustez en escenarios reales:

| Clase  | Descripción              | Ubicación                                    |
|--------|--------------------------|----------------------------------------------|
| LATAS  | Latas de aluminio        | `02_Ingenieria_de_Dato/Datos brutos/LATAS/` |
| PET    | Botellas de plástico PET | `02_Ingenieria_de_Dato/Datos brutos/PET/`   |
| VIDRIO | Botellas de vidrio       | `02_Ingenieria_de_Dato/Datos brutos/VIDRIO/`|

**Características del dataset:**
- Variabilidad de iluminación (frontal, lateral, mixta)
- Fondos diversos (uniforme, texturizados, complejos)
- Ángulos variados (frontal, oblicuo, superior)
- Estados de conservación (nuevo, usado, deteriorado)

**División del dataset (seed=42, reproducible):**
- Entrenamiento: 70%
- Validación: 15%
- Test: 15%

---

## Modelos Evaluados

| Modelo          | Arquitectura                          | Transfer Learning | Parámetros |
|-----------------|---------------------------------------|-------------------|------------|
| ResNet50        | Red residual de 50 capas              | ImageNet          | 25.6M      |
| EfficientNet-B3 | Escalado compuesto optimizado         | ImageNet          | 12M      |
| ConvNeXt-Tiny   | CNN modernizada inspirada en ViT      | ImageNet          | 28.6M      |

**Configuración de entrenamiento:**
- Optimizador: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-5
- Batch size: 32
- Épocas máximas: 50
- Early stopping: patience=10 épocas
- Métrica principal: F1-Score (weighted)
- Data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing

**Decisiones metodológicas clave:**
- Split generado una sola vez en el Pilar 1 y compartido con el Pilar 2 via `split_indices.json` para garantizar que ambos notebooks usan exactamente el mismo conjunto de test
- Tres `ImageFolder` independientes por split para evitar data leakage entre entrenamiento y validación/test
- `RandomErasing` aplicado después de `ToTensor()` y `Normalize()` (requiere tensor, no imagen PIL)
- Fine-tuning completo de todos los parámetros (no solo la capa final) dado que el dominio de residuos difiere significativamente de ImageNet

---

## Resultados

### Resultados en conjunto de test

| Modelo            | Accuracy | F1-Score | Precision | Recall | Latencia (ms) |
|-------------------|----------|----------|-----------|--------|---------------|
| **ConvNeXt-Tiny** | **99.71%** | **0.9971** | **0.9971** | **0.9971** | 14.13 |
| ResNet50          | 98.24%   | 0.9824   | 0.9825    | 0.9824 | 3.66  |
| EfficientNet-B3   | 98.24%   | 0.9824   | 0.9824    | 0.9824 | 3.03  |

> Resultados obtenidos experimentalmente. Entrenamiento realizado en Kaggle con GPU T4.

### Modelo recomendado para producción: ConvNeXt-Tiny (selección multicriterio)

La selección del modelo **no se realiza de forma automática por F1 máximo** — ese sería un error metodológico en el contexto de deployment en hardware embebido. La decisión es multicriterio:

**Modelo recomendado: ConvNeXt-Tiny**
- F1-Score de 0.9971 → ~29 errores por 10.000 envases
- Reducción del 83.5% en errores absolutos frente a ResNet50 (~176 errores/10k)
- Latencia en laboratorio (GPU T4): 14.13 ms — cumple el umbral SDDR de 200 ms con margen ×14
- Mayor defensibilidad regulatoria ante auditorías Ley 7/2022
- TensorRT soportado desde v8.5+

**⚠ Condición obligatoria:** Validar latencia en Jetson Orin Nano antes del deployment. Los tiempos medidos son de GPU T4 (Kaggle) y no incluyen overhead de pipeline — la extrapolación a hardware edge introduce incertidumbre ×3–×12.

**Contingencia: ResNet50** — activar si ConvNeXt-Tiny supera 200 ms en benchmark real sobre Jetson.
- F1-Score: 0.9824 — Latencia lab: 3.66 ms — Mayor margen de robustez en edge
- Ecosistema ONNX/TensorRT más maduro — arquitectura residual optimizada de forma más predecible

**Alternativa hardware severo: EfficientNet-B3** — si las restricciones de hardware resultan más severas, dada su mayor eficiencia de parámetros (12.0M) y menor latencia (3.03 ms).

---

## Análisis de Negocio (Pilar 3)

### Viabilidad económica del módulo de visión

| Componente | Coste estimado |
|---|---|
| NVIDIA Jetson Orin Nano Super | ~€249 |
| Cámara RGB HD | ~€80 |
| Integración software | ~€50 |
| **Total módulo** | **~€379** |

Rango de mercado para módulos de visión industrial equivalentes: €500–€2.000. La solución propuesta supone un ahorro del 69.7% respecto al promedio de mercado.

> **Nota:** El coste es independiente del modelo seleccionado (ConvNeXt-Tiny o contingencia ResNet50). El hardware de €379 soporta ambas opciones.

### Plan de implementación

| Fase | Duración | Criterio de paso |
|---|---|---|
| Fase 1 — Prototipo | 2 meses | F1 >= 0.90 en condiciones controladas |
| Fase 2 — Piloto | 2 meses | F1 >= 0.90 sostenido 4 semanas + uptime >= 98% |
| Fase 3 — Escalado | 2 meses | Cobertura 100% de la red operativa |

Duración total: 6 meses. Compatible con el deadline de noviembre 2026 establecido por la Ley 7/2022 si el inicio se produce en Q1 2026.

---

## Marco Regulatorio

- **Ley 7/2022** — Ley de Residuos y Suelos Contaminados para una Economía Circular (BOE núm. 85)
- **PPWR** — Packaging and Packaging Waste Regulation (Unión Europea, diciembre 2024)
- **Plazo de implementación SDDR en España:** Noviembre 2026

---

## Configuración Experimental

La configuración del experimento se encuentra en `06_Codigo_Completo/config.json`:

```json
{
  "seed": 42,
  "image_size": 224,
  "batch_size": 32,
  "epochs": 50,
  "early_stopping_patience": 10,
  "learning_rate": 1e-4,
  "weight_decay": 1e-5,
  "num_workers": 2
}
```

---

## Notebooks Jupyter

| Pilar | Notebook | Requiere GPU |
|---|---|---|
| Pilar 1 | `02_Ingenieria_de_Dato/notebooks/pilar1_ingenieria_dato.ipynb` | No |
| Pilar 2 | `03_Analisis_de_Datos_y_Modelado/notebook/pilar2_analisis_datos.ipynb` | Sí (Kaggle recomendado) |
| Pilar 3 | `05_Analisis_de_Negocio/notebook/pilar3_analisis_negocio.ipynb` | No |

---

## Notas Técnicas

1. **Reproducibilidad:** Todos los scripts usan semilla fija (seed=42) para garantizar resultados reproducibles
2. **Rutas relativas:** El código utiliza rutas relativas a la raíz del proyecto, compatible con cualquier sistema operativo
3. **Orden de ejecución:** Pilar 1 debe ejecutarse antes que Pilar 2 para generar `split_indices.json`. Pilar 2 debe ejecutarse antes que Pilar 3 para generar `resultados_completos.json`
4. **Sin data leakage:** El split se genera una sola vez con índices fijos. Validación y test usan transforms sin augmentation, independientes del conjunto de entrenamiento
5. **GPU vs CPU:** El entrenamiento funciona en CPU pero el tiempo será significativamente mayor. Se recomienda Kaggle (GPU gratuita) para el Pilar 2
6. **Modelos preentrenados:** Los archivos `.pth` ocupan 100-300 MB cada uno. No incluir en control de versiones — usar Git LFS o almacenamiento externo
7. **augmentation_factor eliminado:** El `CustomDataset` con factor multiplicador fue eliminado por ser metodológicamente incorrecto (repetía imágenes sin nueva variabilidad). El augmentation real ocurre on-the-fly en cada época gracias a las transformaciones aleatorias de `transform_train`
8. **Selección de modelo — criterio multicriterio:** La selección del modelo para producción NO se realiza automáticamente por F1 máximo. El Pilar 3 aplica un análisis multicriterio que incluye latencia en hardware embebido, madurez del ecosistema y riesgo de deployment. Las métricas de latencia son de GPU T4 (Kaggle) y requieren validación en Jetson Orin Nano antes del deployment en producción
9. **Contingencia de modelo:** Si el benchmark de ConvNeXt-Tiny en Jetson Orin Nano supera el umbral de 200 ms, el modelo de contingencia es ResNet50, cuya conversión ONNX/TensorRT está documentada en el Pilar 3

---

## Estructura de Archivos de Salida

### Metadatos
- `02_Ingenieria_de_Dato/dataset_metadata.json` — Información del dataset original
- `02_Ingenieria_de_Dato/split_indices.json` — Índices del split train/val/test (generado por Pilar 1)

### Modelos
- `03_Analisis_de_Datos_y_Modelado/modelos/*.pth` — Pesos de modelos entrenados
- `03_Analisis_de_Datos_y_Modelado/modelos/historia_*.json` — Histórico de entrenamiento por época

### Resultados de Evaluación
- `03_Analisis_de_Datos_y_Modelado/resultados/comparativa_modelos.csv` — Comparación de métricas
- `03_Analisis_de_Datos_y_Modelado/resultados/resultados_completos.json` — Resultados detallados por modelo
- `03_Analisis_de_Datos_y_Modelado/resultados/*.png` — Matrices de confusión y curvas de aprendizaje

### Análisis de Negocio
- `05_Analisis_de_Negocio/decision_modelo_produccion.json` — Justificación de selección del modelo
- `05_Analisis_de_Negocio/Graficas/*.png` — Visualizaciones del análisis de negocio

---

## Documentación Académica

La carpeta `01_Documentacion_academica/` contiene:

- **Anteproyecto (aprobado):** Propuesta inicial del TFG aprobada por la universidad
- **Memoria TFG (en desarrollo):** Documento completo del trabajo en formato DOCX y PDF
- **Presentación (pendiente):** Diapositivas para la defensa oral
- **Resumen ejecutivo (pendiente):** Síntesis del proyecto en formato ejecutivo

---

## Control de Calidad

La carpeta `07_taxonomia_y_calidad/` incluye:

- **Informe Turnitin:** Verificación de originalidad del documento
- **Taxonomía uso IA:** Declaración de uso de herramientas de inteligencia artificial durante el desarrollo del TFG

---

## Contacto

**Autor:** Pablo Huidobro García  
**Universidad:** Francisco de Vitoria  
**Grado:** Business Analytics  
**Curso Académico:** 2025-26

---

## Licencia

Este proyecto constituye un Trabajo Fin de Grado académico de la Universidad Francisco de Vitoria. Todos los derechos reservados.
