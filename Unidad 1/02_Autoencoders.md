# 1.2 Autoencoders

## ¿Qué es un Autoencoder?

Un **autoencoder** es una red neuronal que aprende a **comprimir** una entrada en una representación interna de menor dimensión y luego **reconstruirla** lo más fielmente posible. No requiere etiquetas externas — la propia entrada es el objetivo → **aprendizaje no supervisado / auto-supervisado**.

<img src="../_static/images/autoencoder1.png" width="600" />

La idea central: si la red logra reconstruir bien la entrada pasando por un cuello de botella estrecho, ese cuello de botella ha capturado lo **esencial** de los datos.

---

## Arquitectura base

Un autoencoder tiene tres componentes:

```
  Entrada x          Espacio latente z         Reconstrucción x̂
  (dim alta)  →  [ENCODER f]  →  (dim baja)  →  [DECODER g]  →  (dim alta)
```

### Encoder

Transforma la entrada `x` en una representación comprimida `z`:

```
z = f(x)
```

- Reduce progresivamente la dimensionalidad.
- Aprende a retener la información más relevante.
- Puede estar compuesto de capas densas, convolucionales, recurrentes, etc.

### Espacio latente (Bottleneck)

- Vector de dimensión `d_z << d_x`.
- Cada dimensión captura un **factor de variación** latente de los datos.
- La geometría de este espacio determina qué tan útil es la representación aprendida.

### Decoder

Reconstruye la entrada original a partir de `z`:

```
x̂ = g(z)
```

- Simetría respecto al encoder (capas inversas).
- Aprende a "descomprimir" la representación latente.

---

## Función de pérdida

El objetivo es minimizar el **error de reconstrucción** entre `x` y `x̂`:

### Para datos continuos — MSE (Mean Squared Error)

```
L = (1/n) Σ ||x - x̂||²
```

Penaliza diferencias absolutas entre píxeles o valores numéricos.

### Para datos binarios / probabilísticos — Binary Cross-Entropy

```
L = -Σ [x · log(x̂) + (1 - x) · log(1 - x̂)]
```

Útil cuando los valores de entrada están en [0, 1].

---

## Entrenamiento

```
1. Alimentar la red con un batch de ejemplos x
2. El encoder comprime: z = f(x)
3. El decoder reconstruye: x̂ = g(z)
4. Calcular la pérdida: L(x, x̂)
5. Retropropagación: actualizar pesos de encoder y decoder
6. Repetir hasta convergencia
```

Los pesos del encoder y decoder se actualizan **conjuntamente** con retropropagación estándar. No hay etiquetas externas — el gradiente proviene del error de reconstrucción.

<img src="../_static/images/autoencoder2.png" width="600" />

---

## Tipos de Autoencoders

### 1. Autoencoder Undercomplete (básico)

El tipo más simple. El espacio latente tiene **menor dimensión** que la entrada, forzando a aprender una representación comprimida.

```
Entrada: 784 dim → Encoder → 32 dim → Decoder → 784 dim
```

**Riesgo**: si el modelo es demasiado poderoso (sin regularización), puede aprender a "copiar" la entrada sin extraer estructura útil.

**Usos**: reducción de dimensionalidad, inicialización de redes más grandes.

---

### 2. Autoencoder Convolucional

Usa **capas convolucionales** en lugar de capas densas, preservando la estructura espacial de las imágenes.

```
Encoder:
  Conv2D(32, 3) + ReLU → MaxPool(2)   → (H/2, W/2, 32)
  Conv2D(64, 3) + ReLU → MaxPool(2)   → (H/4, W/4, 64)

Decoder:
  Conv2DTranspose(64, 3, stride=2)    → (H/2, W/2, 64)
  Conv2DTranspose(32, 3, stride=2)    → (H, W, 32)
  Conv2D(C, 1, sigmoid)               → (H, W, C)
```

<img src="../_static/images/autoencoder3.png" width="600" />


**Usos**: compresión de imágenes, eliminación de ruido, extracción de características visuales.

---

### 3. Denoising Autoencoder (DAE)

Se introduce **ruido artificialmente** en la entrada y se entrena al modelo para reconstruir la versión limpia original.

```
x  →  x̃ (entrada corrompida)  →  Encoder  →  z  →  Decoder  →  x̂ ≈ x (limpia)
```

Tipos de corrupción usados:
- Ruido gaussiano aditivo.
- Enmascaramiento aleatorio (*dropout* de entradas).
- Sal y pimienta.

<img src="../_static/images/autoencoder4.png" width="600" />


**Ventaja clave**: fuerza al modelo a aprender representaciones **robustas** que no dependen de los píxeles individuales, sino de la estructura global.

**Usos**: restauración de imágenes, preprocesamiento médico, robustez ante datos corruptos.

---

### 4. Sparse Autoencoder

Agrega una **penalización de dispersión (sparsity)** a las activaciones del espacio latente, forzando a que solo pocas neuronas estén activas simultáneamente.

```
L_total = L_reconstrucción + β · L_sparsity
```

Donde `L_sparsity` puede ser la divergencia KL entre la activación media y un objetivo de activación ρ (típicamente ~0.05).

<img src="../_static/images/autoencoder5.png" width="600" />

**Interpretación**: cada neurona activa representa una "característica" específica y distinguible de los datos.

**Usos**: extracción de características interpretables, análisis de señales biomédicas, aprendizaje de diccionarios.

---

### 5. Contractive Autoencoder (CAE)

Penaliza la sensibilidad del encoder a pequeñas variaciones en la entrada, usando la **norma de Frobenius del Jacobiano**:

```
L_total = L_reconstrucción + λ · ||∂f(x)/∂x||²_F
```

Aprende representaciones que son **insensibles a perturbaciones** locales pero robustas a variaciones globales.

**Usos**: aprendizaje de variedades (*manifold learning*), representaciones robustas para clasificación.

---

### 6. Variational Autoencoder (VAE)

El más influyente de los autoencoders generativos. En lugar de mapear `x` a un vector fijo `z`, el encoder aprende una **distribución de probabilidad**:

```
Encoder → μ (media) y σ (desviación estándar)
z ~ N(μ, σ²)  ← muestreo con reparametrización: z = μ + ε·σ, ε ~ N(0,1)
Decoder → x̂
```

<img src="../_static/images/autoencoder6.png" width="600" />

#### Función de pérdida del VAE

```
L_VAE = L_reconstrucción + β · KL(q(z|x) || p(z))
```

- **L_reconstrucción**: fidelidad de la salida.
- **KL divergence**: regulariza el espacio latente para que sea continuo y siga N(0,1).

#### Propiedades del espacio latente del VAE

- **Continuidad**: puntos cercanos en `z` generan salidas similares.
- **Completitud**: cualquier punto del espacio latente produce una salida válida.
- Permite **interpolación suave** entre ejemplos.
- Permite **generar** nuevas muestras muestreando `z ~ N(0,1)`.

<img src="../_static/images/autoencoder7.png" width="600" />

**Usos**: generación de imágenes, síntesis de datos, representación disentangled, drug discovery.

---

### 7. β-VAE

Extensión del VAE con un hiperparámetro `β > 1` que aumenta el peso de la regularización KL:

```
L = L_reconstrucción + β · KL(...)
```

Fuerza al modelo a aprender representaciones **disentangled**: cada dimensión del espacio latente controla un factor de variación independiente (color, forma, rotación, etc.).

<img src="../_static/images/autoencoder8.png" width="600" />


---

### 8. Autoencoder Secuencial (Seq2Seq Autoencoder)

Usa **redes recurrentes (LSTM / GRU)** en encoder y decoder para comprimir secuencias de longitud variable.

```
Secuencia x₁,x₂,...,xₙ → LSTM Encoder → z → LSTM Decoder → x̂₁,x̂₂,...,x̂ₙ
```

<img src="../_static/images/autoencoder9.png" width="600" />

**Usos**: compresión de series temporales, preentrenamiento de modelos de lenguaje, anomaly detection en señales temporales.

---

### 9. Masked Autoencoder (MAE)

Inspirado en BERT pero para imágenes. Enmascara un alto porcentaje de parches de la imagen (75–90%) y entrena al modelo para reconstruirlos.

<img src="../_static/images/autoencoder10.png" width="600" />

- El encoder solo procesa los parches visibles.
- El decoder reconstruye los parches enmascarados.
- Muy eficiente computacionalmente y produce representaciones de alta calidad.

**Base de modelos**: MAE de Meta, BEiT, SimMIM.

**Usos**: preentrenamiento de Vision Transformers (ViT), transferencia a tareas downstream con pocos datos.

---

## El Espacio Latente: corazón del autoencoder

El espacio latente `z` es la representación aprendida de los datos. Sus propiedades determinan para qué sirve el autoencoder:

| Propiedad | Autoencoder | Efecto |
|-----------|-------------|--------|
| Comprimido (d_z << d_x) | Todos | Elimina redundancia |
| Disperso | Sparse AE | Interpretabilidad |
| Suave y continuo | VAE | Generación e interpolación |
| Robusto al ruido | DAE, CAE | Representaciones robustas |
| Disentangled | β-VAE | Control de factores latentes |

---

## Aplicaciones de los Autoencoders

### Reducción de dimensionalidad

Alternativa no lineal a PCA. El espacio latente captura estructuras complejas que PCA no puede.


### Detección de anomalías

Entrenar el autoencoder **solo con datos normales**. Al intentar reconstruir una anomalía, el error de reconstrucción será alto → señal de anomalía.

```
Error de reconstrucción alto → Anomalía
Error de reconstrucción bajo → Normal
```

**Usos**: fraude financiero, defectos industriales, intrusiones en redes, patología médica rara.

### Eliminación de ruido (Denoising)

Restaurar imágenes, señales de audio o series temporales corrompidas.

### Compresión de datos

El encoder actúa como compresor y el decoder como descompresor. Aprendido end-to-end para minimizar el error de reconstrucción.

### Generación de datos sintéticos

Los VAE pueden generar nuevas muestras muestreando del espacio latente. Útil para aumentar datasets pequeños.

### Preentrenamiento de representaciones

El encoder preentrenado se usa como **extractor de características** para tareas supervisadas downstream (clasificación, detección…), especialmente útil con pocos datos etiquetados.

### Sistemas de recomendación

Comprimir el historial de interacciones de un usuario en un vector latente para hacer recomendaciones personalizadas.

### Procesamiento de lenguaje natural

Seq2Seq autoencoders comprimen oraciones en vectores semánticos usados para búsqueda, clustering de textos y transferencia de estilo.

---

## Autoencoders vs. otras técnicas de representación

| Técnica | Lineal | Generativa | Escalable | Representación no lineal |
|---------|--------|------------|-----------|--------------------------|
| **PCA** | ✅ | ❌ | ✅ | ❌ |
| **Autoencoder** | ❌ | ❌ | ✅ | ✅ |
| **VAE** | ❌ | ✅ | ✅ | ✅ |
| **GAN** | ❌ | ✅ | ✅ | ✅ |
| **t-SNE / UMAP** | ❌ | ❌ | ❌ (solo visualización) | ✅ |

---

## Hiperparámetros clave

| Hiperparámetro | Efecto |
|----------------|--------|
| **Dimensión del espacio latente** | Muy pequeño → pérdida de info. Muy grande → no aprende a comprimir. |
| **Profundidad del encoder/decoder** | Mayor profundidad → representaciones más abstractas. |
| **Función de activación** | ReLU para capas intermedias; Sigmoid/Tanh para salida según dominio. |
| **β (en VAE)** | Mayor β → más disentanglement, pero peor reconstrucción. |
| **Tasa de corrupción (DAE)** | Mayor ruido → representaciones más robustas, entrenamiento más difícil. |

---

## Diagnóstico: ¿qué puede salir mal?

| Problema | Causa | Solución |
|----------|-------|----------|
| Reconstrucción borrosa | Pérdida MSE promedia modos | Usar pérdida perceptual o adversarial |
| Posterior collapse (VAE) | KL domina desde el inicio | KL annealing (aumentar β gradualmente) |
| Sobreajuste del espacio latente | Sin regularización | Añadir dropout, sparse penalty, KL |
| Espacio latente no estructurado | AE básico sin restricciones | Cambiar a VAE o agregar regularización |

---

## Frameworks y recursos

| Herramienta | Uso |
|-------------|-----|
| `PyTorch` | Implementación flexible de cualquier variante. |
| `Keras / TensorFlow` | APIs de alto nivel para AE y VAE. |
| `Hugging Face` | MAE, BEiT preentrenados para ViT. |
| `PyTorch Lightning` | Entrenamiento estructurado y reproducible. |
| `scikit-learn` | PCA como baseline para comparar con AE. |