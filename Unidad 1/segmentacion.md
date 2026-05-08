# Segmentación con Inteligencia Artificial

## ¿Qué es la segmentación?

La **segmentación** es la tarea de asignar una etiqueta a **cada píxel** de una imagen, dividiendo la escena en regiones con significado semántico. A diferencia de la clasificación (una etiqueta por imagen) o la detección (una caja por objeto), la segmentación produce una **máscara densa** del mismo tamaño que la entrada.

<img src="../_static/images/segmentacion1.png" width="600" />


---

## Tipos de segmentación

### Segmentación Semántica

Asigna una clase a cada píxel **sin distinguir instancias individuales**. Todos los objetos de la misma clase reciben la misma etiqueta.

```
Imagen → Máscara por píxel → clase(píxel) ∈ {fondo, persona, coche, árbol, …}
```
<img src="../_static/images/segmentacion2.png" width="600" />


**Problema**: dos personas adyacentes se funden en una sola región.

---

### Segmentación de Instancias

Detecta y segmenta **cada objeto individualmente**, incluso si son de la misma clase y se solapan.

```
Imagen → {instancia_1: máscara_1, instancia_2: máscara_2, …}
```

<img src="../_static/images/segmentacion3.png" width="600" />

---

### Segmentación Panóptica

Combina segmentación semántica e instancias en una sola salida coherente:

- **Things** (objetos contables): personas, coches → instancias individuales.
- **Stuff** (regiones no contables): cielo, carretera, césped → semántica.

<img src="../_static/images/segmentacion4.png" width="600" />

---

### Segmentación de Partes (*Part Segmentation*)

Segmenta las partes internas de un objeto: en una persona → cabeza, torso, brazos, piernas.

<img src="../_static/images/segmentacion5.png" width="600" />

---

### Segmentación Interactiva

El usuario guía la segmentación con prompts (clics, cajas, trazos):

```
Imagen + {clic en objeto} → Máscara del objeto seleccionado
```

Base de SAM (Segment Anything Model).

---

### Segmentación de Video

Propaga máscaras de segmentación a lo largo de los frames de un video, manteniendo la identidad de cada objeto.

<img src="../_static/images/segmentacion6.png" width="600" />

---

## Evolución histórica

<img src="../_static/images/segmentacion7.png" width="600" />


| Año | Modelo | Aporte |
|-----|--------|--------|
| 2014 | **FCN** (Fully Convolutional Network) | Primera red totalmente convolucional para segmentación densa. |
| 2015 | **U-Net** | Skip connections para preservar detalles espaciales finos. |
| 2016 | **SegNet** | Encoder-decoder con índices de pooling para upsampling. |
| 2017 | **DeepLab v3** | Dilated convolutions + ASPP para múltiples escalas. |
| 2017 | **Mask R-CNN** | Segmentación de instancias sobre Faster R-CNN. |
| 2018 | **Panoptic FPN** | Segmentación panóptica unificada. |
| 2020 | **SETR** | Primeros Transformers aplicados a segmentación. |
| 2021 | **SegFormer** | Transformer eficiente para segmentación semántica. |
| 2021 | **Swin Transformer** | Backbone jerárquico con atención por ventanas. |
| 2023 | **SAM** | Segmentación zero-shot guiada por prompts. |
| 2024 | **SAM 2** | SAM extendido a video en tiempo real. |

---

## Arquitecturas fundamentales

### FCN — Fully Convolutional Network

Primera arquitectura en sustituir las capas densas finales de una CNN por capas convolucionales, permitiendo entrada de cualquier tamaño y salida densa.

```
Imagen → [VGG/AlexNet backbone] → mapas de características → Conv 1×1 → Upsample → Máscara
```


**Problema**: el upsampling directo produce máscaras borrosas sin detalles finos.

---

### U-Net

Arquitectura encoder-decoder con **conexiones de salto (skip connections)** que concatenan mapas del encoder directamente al decoder en el mismo nivel de resolución.

```
Encoder (contracción):           Decoder (expansión):
  x     → Conv → Pool             UpSample → Conv + [skip] → Conv
  x/2   → Conv → Pool             UpSample → Conv + [skip] → Conv
  x/4   → Conv → Pool             UpSample → Conv + [skip] → Conv
  x/8   → Conv (bottleneck)
```
<img src="../_static/images/segmentacion8.png" width="600" />


**Por qué funciona**: el bottleneck captura contexto global; las skip connections recuperan los detalles espaciales perdidos en el pooling.

**Variantes modernas**:
- **U-Net++**: skip connections anidadas y densas.
- **Attention U-Net**: mecanismo de atención en las skip connections.
- **TransUNet**: encoder con ViT + decoder CNN.
- **Swin-UNet**: U-Net completamente basado en Swin Transformer.

---

### SegNet

Encoder-decoder donde el decoder usa los **índices de MaxPooling** del encoder para hacer upsampling, evitando parámetros adicionales.



---

### DeepLab v3 / v3+

Introduce dos innovaciones clave para capturar **múltiples escalas** sin perder resolución:

**Dilated (Atrous) Convolutions**: convolucionan con "agujeros" para aumentar el campo receptivo sin reducir la resolución espacial.

```
Dilatación 1: campo receptivo normal
Dilatación 2: campo receptivo 2× mayor, misma cantidad de parámetros
Dilatación 4: campo receptivo 4× mayor
```

**ASPP (Atrous Spatial Pyramid Pooling)**: aplica varias convoluciones dilatadas en paralelo con distintas tasas + pooling global.

```
Mapa de características → [ASPP: d=6, d=12, d=18, AvgPool] → Concat → Conv 1×1 → Máscara
```

**DeepLab v3+** añade un decoder ligero (similar a U-Net) para refinar los bordes.

---

### PSPNet — Pyramid Scene Parsing Network

Usa un **Pyramid Pooling Module (PPM)** para capturar contexto global a múltiples escalas:

```
Mapa de características → [Pool 1×1, 2×2, 3×3, 6×6] → Upsample → Concat → Conv → Máscara
```


---

### Mask R-CNN — Segmentación de instancias

Extiende Faster R-CNN con una **tercera cabeza paralela** que predice una máscara binaria por región de interés (ROI):

```
Imagen → Backbone + FPN → RPN → ROI Align → [Clasificación + BBox + Máscara]
```

**ROI Align** (vs. ROI Pooling): interpolación bilineal para preservar la alineación espacial → máscaras más precisas.


---

### SegFormer

Arquitectura totalmente basada en Transformers, eficiente y sin necesidad de positional encoding fijo:

```
Imagen → [Mix Transformer Encoder (MiT)] → representaciones multi-escala
       → [Decoder MLP ligero] → Máscara semántica
```

<img src="../_static/images/segmentacion9.png" width="600" />

**Ventajas**: supera a DeepLab en mIoU con menor costo computacional.

---

### SAM — Segment Anything Model

Meta AI (2023). Modelo de segmentación interactiva **zero-shot** entrenado en más de 1 000 millones de máscaras.

**Arquitectura**:
```
Imagen → [Image Encoder (ViT-H)] → Image Embedding
Prompt → [Prompt Encoder] → Prompt Embedding
        → [Mask Decoder (Transformer)] → Máscaras + scores de calidad
```

**Tipos de prompt soportados**:
- **Punto**: clic en un píxel de interés.
- **Caja**: bounding box que delimita el objeto.
- **Máscara**: máscara gruesa como guía.
- **Texto**: descripción en lenguaje natural (SAM 2 + Grounding DINO).

**SAM 2** (2024): extiende SAM a **video**, propagando máscaras frame a frame con un memory bank que mantiene la identidad de los objetos.

---

### Grounded SAM

Combina **Grounding DINO** (detección open-vocabulary) + **SAM** (segmentación):

```
"Todas las personas con casco"
        ↓
[Grounding DINO] → Bounding boxes
        ↓
[SAM] → Máscaras precisas por instancia
```


Permite segmentación **completamente guiada por lenguaje natural** sin fine-tuning.

---

## Dilated Convolutions en profundidad

Las convoluciones dilatadas son fundamentales en segmentación para mantener la resolución espacial:

```
Convolucion normal (d=1):        Convolucion dilatada (d=2):
  ■ ■ ■                             ■ · ■ · ■
  ■ ■ ■                             · · · · ·
  ■ ■ ■                             ■ · ■ · ■
                                     · · · · ·
                                     ■ · ■ · ■
```

- Misma cantidad de parámetros, mayor campo receptivo.
- No reduce la resolución espacial (sin stride ni pooling).
- Permiten al modelo ver contexto amplio sin perder detalles de borde.
<img src="../_static/images/segmentacion10.png" width="600" />


---

## Feature Pyramid Networks (FPN)

Las FPN permiten detectar y segmentar objetos de **múltiples escalas** en una sola pasada:

```
Backbone (bottom-up):          FPN (top-down + lateral connections):
  C2 (1/4)   ←──────────────── P2 (alta resolución, semántica media)
  C3 (1/8)   ←──────────────── P3
  C4 (1/16)  ←──────────────── P4
  C5 (1/32)  ←──────────────── P5 (baja resolución, alta semántica)
```


Cada nivel P_i combina la semántica de los niveles superiores con los detalles espaciales del nivel correspondiente del backbone.

---

## Funciones de pérdida para segmentación

### Cross-Entropy por píxel

```
L_CE = -(1/N) Σ_i Σ_c y_{i,c} · log(ŷ_{i,c})
```

Estándar para clases balanceadas. Cada píxel contribuye igual.

### Weighted Cross-Entropy

Asigna pesos inversamente proporcionales a la frecuencia de cada clase:

```
L_WCE = -(1/N) Σ_i Σ_c w_c · y_{i,c} · log(ŷ_{i,c})
```

Útil cuando el fondo domina (ej. 95% fondo, 5% objeto de interés).

### Dice Loss

```
L_Dice = 1 - (2 · |A ∩ B|) / (|A| + |B|)
```

Optimiza directamente la superposición entre predicción y ground truth. Ideal para regiones pequeñas y datasets desbalanceados.

### IoU Loss (Jaccard Loss)

```
L_IoU = 1 - |A ∩ B| / |A ∪ B|
```

Similar a Dice, optimiza directamente la métrica de evaluación estándar.

### Focal Loss

```
L_FL = -(1 - ŷ)^γ · log(ŷ)
```

Reduce el peso de los píxeles fáciles (bien clasificados) y enfoca el entrenamiento en los difíciles. Parámetro γ controla el enfoque (típicamente γ=2).

### Tversky Loss

Generalización de Dice que pondera diferente FP y FN:

```
L_Tversky = 1 - TP / (TP + α·FP + β·FN)
```

Con α < β se penaliza más los Falsos Negativos → útil cuando perder un objeto es más costoso que una falsa alarma.

### Pérdida combinada (práctica habitual)

```
L_total = L_CE + λ · L_Dice
```

Combina estabilidad de CE con optimización directa de la métrica.

---

## Métricas de evaluación

### Pixel Accuracy

```
PA = Σ_c TP_c / N_total
```

Porcentaje de píxeles correctamente clasificados. **Engañosa con clases desbalanceadas**.

### Mean Pixel Accuracy (MPA)

```
MPA = (1/C) Σ_c TP_c / N_c
```

Promedia la accuracy por clase, corrigiendo el desbalance.

### IoU (Intersection over Union) por clase

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

<!-- IMAGEN: visualización geométrica de la intersección y unión entre máscara predicha y ground truth -->

### Mean IoU (mIoU)

```
mIoU = (1/C) Σ_c IoU_c
```

**Estándar en benchmarks** de segmentación. Mide el rendimiento promedio sobre todas las clases.
<img src="../_static/images/segmentacion11.png" width="600" />


### Dice Coefficient / F1 por región

```
Dice_c = 2·TP_c / (2·TP_c + FP_c + FN_c)
```

Equivalente al F1-score aplicado a regiones. Muy usado en imágenes médicas.

### Boundary F1 Score (BF)

Mide la precisión específicamente en los **bordes** de los segmentos, capturando fineza de los contornos.

### Panoptic Quality (PQ)

Para segmentación panóptica:

```
PQ = SQ · RQ

SQ (Segmentation Quality) = IoU promedio de instancias emparejadas
RQ (Recognition Quality)  = F1 de emparejamiento instancia-ground truth
```

---

## Data Augmentation para segmentación

La augmentación debe aplicarse **sincrónicamente** a la imagen y su máscara:

| Transformación | Imagen | Máscara |
|----------------|--------|---------|
| Flip horizontal | ✅ | ✅ (misma dirección) |
| Rotación | ✅ | ✅ (misma rotación) |
| Crop aleatorio | ✅ | ✅ (mismo crop) |
| Escala | ✅ | ✅ |
| Cambio de brillo/contraste | ✅ | ❌ (no afecta etiquetas) |
| Ruido gaussiano | ✅ | ❌ |
| Mezcla de colores | ✅ | ❌ |
| Elastic deformation | ✅ | ✅ (misma deformación) |
| MixUp / CutMix | ✅ | ✅ (mezcla ponderada) |

**Librería recomendada**: `albumentations` — aplica transformaciones geométricas de forma sincronizada.

---

## Transfer Learning en segmentación

Aprovechar backbones preentrenados en ImageNet o COCO acelera drásticamente el entrenamiento:

```
Backbone preentrenado (ResNet, EfficientNet, ViT…)
        +
Decoder / cabeza de segmentación (inicializada aleatoriamente)
        ↓
Fine-tuning con datos de segmentación propios
```

**Estrategias de fine-tuning**:
- **Congelar backbone**: entrenar solo el decoder (pocos datos).
- **Fine-tuning parcial**: descongelar últimas capas del backbone.
- **Fine-tuning completo**: descongelar todo (muchos datos).

---

## Segmentación en dominio médico

Dominio con características especiales que requieren consideraciones específicas:

### Desafíos
- Datos escasos y costosos de anotar (requieren expertos).
- Clases muy desbalanceadas (lesión pequeña en fondo grande).
- Variabilidad inter-anotador.
- Modalidades diversas: MRI, CT, ultrasonido, histología, endoscopía.

### Arquitecturas más usadas
- **U-Net y variantes**: estándar de facto en imágenes médicas.
- **nnU-Net**: framework auto-configurado que adapta U-Net a cualquier dataset médico.
- **TransUNet / Swin-UNet**: Transformers para capturar dependencias de largo alcance.

### Técnicas especiales
- **Anotación débil**: usar cajas o puntos en lugar de máscaras densas.
- **Semi-supervisión**: pocas imágenes anotadas + muchas sin anotar.
- **Self-supervised pretraining**: preentrenar en datos no etiquetados.


---

## Segmentación en teledetección y satélite


Desafíos específicos:
- Imágenes de muy alta resolución (gigapíxeles).
- Múltiples bandas espectrales (RGB + NIR + SWIR…).
- Objetos de escalas muy distintas (edificios, campos de cultivo, ríos).

Arquitecturas comunes: U-Net adaptado, SegFormer, DeepLab v3+.

---

## Segmentación en conducción autónoma

Requiere:
- **Tiempo real**: inferencia a 30+ FPS.
- **Alta precisión en bordes**: fundamental para seguridad.
- **Robustez**: distintas condiciones de luz, lluvia, noche.

Datasets: **Cityscapes**, **ADE20K**, **BDD100K**.

---

## Benchmarks y datasets de referencia

| Dataset | Clases | Imágenes | Dominio |
|---------|--------|----------|---------|
| **PASCAL VOC 2012** | 21 | 11K | Objetos cotidianos |
| **MS COCO** | 80 | 330K | Instancias y panóptica |
| **Cityscapes** | 19 | 5K (densamente anotadas) | Conducción urbana |
| **ADE20K** | 150 | 25K | Escenas interiores/exteriores |
| **BDD100K** | 40 | 100K | Conducción diversa |
| **Medical Segmentation Decathlon** | 10 tareas | Variable | Imágenes médicas |
| **ISIC** | 2 | 25K+ | Dermatología (melanoma) |

---

## Flujo completo de un proyecto de segmentación

```
1. Definir el problema
   └── ¿Semántica, instancias o panóptica?
   └── ¿Cuántas clases? ¿Están balanceadas?

2. Recopilar y anotar datos
   └── Herramientas: LabelMe, CVAT, Roboflow, Scale AI
   └── Formato: COCO JSON, Pascal VOC XML, máscaras PNG

3. Preprocesamiento y augmentación
   └── Normalización, resize, albumentations

4. Seleccionar arquitectura
   └── Pocos datos + dominio médico → U-Net / nnU-Net
   └── Tiempo real → SegFormer pequeño, DeepLab MobileNet
   └── Alta precisión → Mask R-CNN, SegFormer-B5
   └── Zero-shot → SAM + Grounding DINO

5. Transfer Learning
   └── Backbone preentrenado en ImageNet / COCO

6. Definir función de pérdida
   └── Balanceado → CE + Dice
   └── Muy desbalanceado → Focal + Tversky

7. Entrenamiento y monitoreo
   └── Métricas: mIoU en validación por época

8. Evaluación final
   └── mIoU, Dice, Boundary F1 en test set

9. Optimización para despliegue
   └── ONNX, TensorRT, quantización INT8

10. Despliegue e inferencia
    └── API REST, edge device, tiempo real
```

---

## Frameworks y herramientas

| Herramienta | Uso |
|-------------|-----|
| `segmentation-models-pytorch` | U-Net, FPN, PSPNet, DeepLab con backbones preentrenados. |
| `MMSegmentation` | Colección completa de modelos de segmentación en PyTorch. |
| `Detectron2` (Meta) | Mask R-CNN, Panoptic FPN, PointRend. |
| `nnU-Net` | U-Net auto-configurado para imágenes médicas. |
| `Hugging Face Transformers` | SegFormer, SAM, Mask2Former, OneFormer. |
| `albumentations` | Augmentación sincronizada imagen + máscara. |
| `MONAI` | Framework médico completo (segmentación 2D y 3D). |
| `Roboflow` | Anotación, augmentación y despliegue en la nube. |
| `LabelMe / CVAT` | Herramientas de anotación de máscaras. |
| `TensorRT / ONNX` | Optimización para inferencia en producción. |