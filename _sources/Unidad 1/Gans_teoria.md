# 1.4 Redes Generativas Adversariales (GANs)

## ¿Qué es una GAN?

Una **Generative Adversarial Network (GAN)** (Goodfellow et al., 2014) es un framework de aprendizaje profundo compuesto por **dos redes neuronales que compiten entre sí** en un juego de suma cero:

- El **Generador (G)** aprende a crear datos sintéticos indistinguibles de los reales.
- El **Discriminador (D)** aprende a distinguir datos reales de sintéticos.

La competencia entre ambas redes impulsa a cada una a mejorar continuamente hasta que G produce muestras de altísima calidad.

<img src="../_static/images/gans1.png" width="600" />


---

## Intuición: el falsificador y el detective

La analogía más usada para entender GANs:

| Red | Rol | Objetivo |
|-----|-----|----------|
| **Generador G** | Falsificador de arte | Crear cuadros tan buenos que el detective no los detecte. |
| **Discriminador D** | Detective / experto | Distinguir cuadros originales de falsificaciones. |

Ambos mejoran al mismo tiempo: el detective obliga al falsificador a ser mejor, y el falsificador obliga al detective a ser más preciso.

---

## Arquitectura

### Generador G

- **Entrada**: vector de ruido aleatorio `z` muestreado de una distribución (normalmente N(0,1)).
- **Salida**: muestra sintética `G(z)` en el espacio de los datos (imagen, audio, texto…).
- **Objetivo**: engañar al discriminador → maximizar `log D(G(z))`.

```
z ~ N(0, I)  →  [G]  →  G(z)  (imagen sintética)
```

### Discriminador D

- **Entrada**: muestra real `x` o muestra sintética `G(z)`.
- **Salida**: escalar en [0, 1] → probabilidad de que la muestra sea real.
- **Objetivo**: clasificar correctamente → maximizar `log D(x) + log(1 - D(G(z)))`.

```
x (real)    →  [D]  →  D(x)   ≈ 1
G(z) (falso) →  [D]  →  D(G(z)) ≈ 0
```

<img src="../_static/images/gans2.png" width="600" />

---

## Función de pérdida original (minimax)

El entrenamiento de una GAN es un **juego minimax**:

```
min_G  max_D  V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

- **D** maximiza: clasifica bien reales e impostores.
- **G** minimiza: hace que D(G(z)) sea lo más cercano a 1 posible.

En la práctica, para evitar gradientes saturados en las primeras etapas, G maximiza `log D(G(z))` en lugar de minimizar `log(1 - D(G(z)))`.

---

## Proceso de entrenamiento

El entrenamiento alterna entre actualizar D y G:

```
Para cada iteración:

  ── Paso 1: Entrenar el Discriminador ──
  1. Tomar un batch de datos reales x
  2. Generar un batch de datos falsos G(z)
  3. Calcular: L_D = -[log D(x) + log(1 - D(G(z)))]
  4. Actualizar pesos de D (G se congela)

  ── Paso 2: Entrenar el Generador ──
  1. Generar un batch de datos falsos G(z)
  2. Calcular: L_G = -log D(G(z))
  3. Actualizar pesos de G (D se congela)
```

<img src="../_static/images/gans3.png" width="600" />

> El equilibrio teórico (Nash) se alcanza cuando G reproduce la distribución real y D no puede hacer mejor que 50% de acierto.

---

## Tipos de GANs

### GAN Condicional (cGAN)

Condiciona tanto G como D en una etiqueta o señal adicional `y`:

```
G(z, y) → muestra del tipo y
D(x, y) → ¿es x una muestra real del tipo y?
```

<img src="../_static/images/gans4.png" width="600" />

**Permite controlar** qué tipo de muestra se genera (ej. generar solo el dígito "7", o una imagen de perro).

**Usos**: generación controlada, síntesis condicionada, transferencia de estilo.

---

### DCGAN — Deep Convolutional GAN

Arquitectura estabilizadora que introdujo convoluciones transpuestas en G y convoluciones strided en D:

**Generador**:
```
z (100) → FC → Reshape → ConvTranspose2D × 4 → imagen (64×64×3)
```

**Discriminador**:
```
imagen → Conv2D (stride 2) × 4 → FC → sigmoid
```

Reglas clave que estabilizan el entrenamiento:
- Batch Normalization en todas las capas (excepto entrada de D y salida de G).
- Activación **ReLU** en G, **LeakyReLU** en D.
- Eliminar capas de pooling → usar stride en su lugar.

<img src="../_static/images/gans5.png" width="600" />

---

### Pix2Pix — GAN para traducción imagen-a-imagen

Traduce una imagen de un dominio a otro usando pares de imágenes de entrenamiento:

```
Imagen fuente (A)  →  [G tipo U-Net]  →  Imagen objetivo (B)
                              ↓
                        [D tipo PatchGAN]
                       ¿El par (A, B) es real o falso?
```

**PatchGAN**: el discriminador evalúa parches locales (ej. 70×70 px) en lugar de la imagen completa → captura mejor texturas y detalles finos.

**Pérdida combinada**:
```
L = L_adversarial + λ · L_L1
```

<img src="../_static/images/gans6.png" width="600" />

**Aplicaciones**: sketch a foto, mapa a satélite, día a noche, blanco y negro a color, plano a fachada.

---

### CycleGAN — Traducción sin pares de datos

Aprende a traducir entre dominios A y B **sin imágenes pareadas**:

```
G: A → B   (traduce de A a B)
F: B → A   (traduce de B a A)
```

**Pérdida de consistencia de ciclo**:
```
F(G(x)) ≈ x   (ciclo A → B → A)
G(F(y)) ≈ y   (ciclo B → A → B)
```

<img src="../_static/images/gans7.png" width="600" />

**Aplicaciones**: caballo ↔ cebra, verano ↔ invierno, foto ↔ pintura, manzana ↔ naranja.

---

### StyleGAN / StyleGAN2 / StyleGAN3

Arquitectura de NVIDIA que separa el contenido del estilo mediante una red de mapeo:

```
z → [Mapping Network f] → w (vector de estilo)
w → [Síntesis por AdaIN en cada capa] → imagen de alta resolución
```

<img src="../_static/images/gans8.png" width="600" />

**Innovaciones**:
- **AdaIN (Adaptive Instance Normalization)**: inyecta el estilo en cada capa.
- **Progressive Growing**: entrena primero en baja resolución, escala progresivamente.
- **Stochastic Variation**: ruido por capa para detalles estocásticos (poros, pelo).

**Aplicaciones**: generación de caras fotorrealistas, generación de arte, avatares digitales, síntesis de datos médicos.

---

### WGAN — Wasserstein GAN

Reemplaza la función de pérdida minimax por la **distancia de Wasserstein** (Earth Mover's Distance):

```
min_G  max_{||D||_L ≤ 1}  E[D(x)] - E[D(G(z))]
```

El discriminador se convierte en un **crítico** (ya no produce probabilidades, sino scores reales).

**Ventajas sobre GAN original**:
- Gradientes más estables durante el entrenamiento.
- La pérdida del crítico correlaciona con la calidad de las imágenes generadas → se puede monitorear.
- Menos susceptible a mode collapse.

**Variante WGAN-GP**: usa **gradient penalty** en lugar de weight clipping para imponer la restricción de Lipschitz.

---

### ProGAN — Progressive Growing GAN

Entrena la GAN de forma **progresiva**: comienza generando imágenes de 4×4 px y va aumentando la resolución añadiendo capas.


**Ventajas**:
- Estabilidad: primero aprende estructura global (baja resolución), luego detalles finos.
- Permite generar imágenes de alta resolución (1024×1024) con calidad fotorrealista.

---

### BigGAN

Escala las GANs a resoluciones y datasets grandes (ImageNet 128×128):

- Batch sizes mucho mayores → mayor estabilidad.
- Condicionamiento de clase en cada capa mediante **class embeddings**.
- Introduce el parámetro **truncation trick**: controla el trade-off diversidad/calidad.

<img src="../_static/images/gans9.png" width="600" />

---

### GAN para Video — VideoGAN / MoCoGAN / SORA

Extienden las GANs al dominio temporal:

- **Contenido**: qué aparece en el video (estático).
- **Movimiento**: cómo cambia a lo largo del tiempo.

```
z_contenido + z_movimiento(t) → [G] → frame_t
```

**SORA** (OpenAI, 2024) va más allá: genera video de alta calidad condicionado por texto usando un modelo de difusión + Transformers, superando las limitaciones de las GANs clásicas para video.

---

### Text-to-Image GANs — StackGAN / AttnGAN

Generan imágenes a partir de descripciones en texto:

```
"Un pájaro rojo con alas azules" → [GAN + CLIP/BERT] → imagen
```

<img src="../_static/images/gans10.png" width="600" />

Superadas hoy por modelos de difusión (Stable Diffusion, DALL-E), pero son el antecedente directo.

---

## Problemas comunes en el entrenamiento de GANs

### Mode Collapse

El generador aprende a producir **solo unas pocas variaciones** (o una sola), ignorando la diversidad real de los datos.

```
Datos reales: 10 dígitos distintos
Generador colapsado: produce solo "1" con diferentes estilos
```

<img src="../_static/images/gans11.png" width="600" />

**Soluciones**:
- Minibatch discrimination: penaliza similitud dentro del batch.
- Unrolled GANs.
- WGAN / WGAN-GP.
- Aumentar la diversidad del ruido de entrada.

### Inestabilidad en el entrenamiento

D y G no convergen al mismo ritmo → uno domina al otro.

**Síntomas**:
- Pérdida de G sube mientras D llega a 0 (D demasiado poderoso).
- Imágenes generadas se degradan repentinamente.

**Soluciones**:
- Tasas de aprendizaje distintas para G y D.
- Actualizar D varias veces por cada actualización de G (o viceversa).
- Label smoothing: en vez de etiquetas 0/1, usar 0.9/0.1.
- Spectral Normalization en D.

### Vanishing Gradients

Cuando D es perfecto, el gradiente que llega a G se anula → G no puede aprender.

**Solución**: usar pérdida de Wasserstein (WGAN) o Feature Matching.

### Memorización

G memoriza los datos de entrenamiento en lugar de generalizar.

**Solución**: más datos, regularización, aumentación.

---

## Métricas de evaluación

| Métrica | Descripción | Qué mide |
|---------|-------------|----------|
| **FID (Fréchet Inception Distance)** | Distancia entre distribuciones de características reales y sintéticas. | Calidad + diversidad. Menor es mejor. |
| **IS (Inception Score)** | Mide nitidez y diversidad usando un clasificador Inception. | Mayor es mejor. |
| **LPIPS** | Similitud perceptual entre imagen real y generada. | Calidad de reconstrucción. |
| **Precision & Recall (GANs)** | Precision = fidelidad; Recall = diversidad. | Trade-off calidad/diversidad. |
| **KID (Kernel Inception Distance)** | Similar a FID pero más robusto con datasets pequeños. | Calidad + diversidad. |

<img src="../_static/images/gans12.png" width="600" />

> **FID** es la métrica estándar en investigación. Valores < 10 indican calidad muy alta; < 50 calidad aceptable.

---

## GANs vs. Modelos de Difusión

Los **modelos de difusión** (Stable Diffusion, DALL-E 3, Midjourney) han superado a las GANs en muchas tareas generativas:

| Criterio | GANs | Difusión |
|----------|------|----------|
| Calidad de imagen | Muy alta | Estado del arte |
| Diversidad | Media (mode collapse) | Alta |
| Estabilidad de entrenamiento | Difícil | Estable |
| Velocidad de inferencia | Muy rápida (una pasada) | Más lenta (múltiples pasos) |
| Control semántico | Moderado | Alto (con texto) |
| Datos necesarios | Muchos | Muchos |
| Interpretabilidad del latente | Media | Baja |

> Las GANs siguen siendo relevantes cuando la **velocidad de inferencia** es crítica (ej. tiempo real) o cuando se trabaja con arquitecturas de edición imagen-a-imagen.

---

## Aplicaciones de las GANs

| Dominio | Aplicación |
|---------|------------|
| **Arte y creatividad** | Generación de obras de arte, avatares, diseño gráfico. |
| **Medicina** | Síntesis de imágenes médicas (MRI, CT) para aumentar datasets. |
| **Moda** | Generación de prendas virtuales, prueba de ropa virtual. |
| **Videojuegos** | Generación de texturas, personajes y escenarios. |
| **Cine / VFX** | Deepfakes, envejecimiento digital, intercambio de rostros. |
| **Ciencias** | Simulación de moléculas, diseño de proteínas, física de partículas. |
| **Datos sintéticos** | Aumentación de datasets escasos (médicos, industriales). |
| **Superresolución** | SRGAN: aumentar resolución de imágenes de baja calidad. |
| **Restauración** | Colorización, eliminación de artefactos, inpainting. |
| **Voz** | Síntesis de voz realista, conversión de voz entre personas. |

---

## Frameworks y recursos

| Herramienta | Uso |
|-------------|-----|
| `PyTorch` | Implementación de cualquier arquitectura GAN. |
| `TensorFlow / Keras` | APIs de alto nivel, tutoriales oficiales de GANs. |
| `Hugging Face Diffusers` | GANs + Difusión (StyleGAN, GAN-based inpainting). |
| `MONAI` | GANs para imágenes médicas. |
| `MMGeneration` | Colección de GANs de investigación en PyTorch. |
| `StyleGAN2-ADA` (NVIDIA) | Implementación oficial de StyleGAN con aumentación adaptativa. |
| `Runway ML` | Plataforma no-code para usar GANs en producción. |