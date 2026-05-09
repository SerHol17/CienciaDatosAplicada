# 1.10 Modelos de Lenguaje de Gran Escala (LLMs)

## Indice

1. [Arquitectura base: Transformer decoder](#1-arquitectura-base-transformer-decoder)
2. [Preentrenamiento a escala](#2-preentrenamiento-a-escala)
3. [Leyes de escala (Scaling Laws)](#3-leyes-de-escala)
4. [Tokenizacion avanzada](#4-tokenizacion-avanzada)
5. [Mejoras arquitectonicas modernas](#5-mejoras-arquitectonicas-modernas)
6. [Ajuste fino supervisado (SFT)](#6-ajuste-fino-supervisado)
7. [RLHF y alineacion](#7-rlhf-y-alineacion)
8. [Inferencia eficiente](#8-inferencia-eficiente)
9. [Adaptacion eficiente en parametros (PEFT)](#9-adaptacion-eficiente-en-parametros)
10. [Capacidades emergentes](#10-capacidades-emergentes)
11. [Evaluacion de LLMs](#11-evaluacion-de-llms)
12. [Referencias](#12-referencias)

---

## 1. Arquitectura Base: Transformer Decoder

### 1.1 Formulacion del modelo de lenguaje autorregresivo

Un LLM define una distribucion sobre secuencias de tokens mediante la regla de la cadena:

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \ldots, x_{t-1}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

El objetivo de entrenamiento es maximizar la log-verosimilitud del corpus $\mathcal{D}$:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \right]$$

Equivalentemente, minimizar la **cross-entropy** media por token:

$$\mathcal{L}_{CE} = -\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

### 1.2 Transformer decoder con Pre-LayerNorm

La arquitectura GPT usa **Pre-LN**: LayerNorm se aplica antes de cada sub-capa, no despues:

$$\mathbf{x}_l' = \mathbf{x}_l + \text{MHA}(\text{LN}(\mathbf{x}_l))$$
$$\mathbf{x}_{l+1} = \mathbf{x}_l' + \text{FFN}(\text{LN}(\mathbf{x}_l'))$$

donde $l \in \{1, \ldots, L\}$ es el indice de capa.

**Ventajas de Pre-LN sobre Post-LN:**
- Gradientes mas estables en capas profundas.
- Permite entrenar sin calentamiento del learning rate (warmup).
- La norma del gradiente no explota al inicio.

**Post-LN** (Transformer original):

$$\mathbf{x}_l' = \text{LN}(\mathbf{x}_l + \text{MHA}(\mathbf{x}_l))$$
$$\mathbf{x}_{l+1} = \text{LN}(\mathbf{x}_l' + \text{FFN}(\mathbf{x}_l'))$$

### 1.3 Scaled Dot-Product Attention con mascara causal

Dadas las proyecciones $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, $\mathbf{V} = \mathbf{X}\mathbf{W}^V$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

La **mascara causal** $\mathbf{M}$ es triangular superior:

$$M_{ij} = \begin{cases} 0 & \text{si } i \geq j \\ -\infty & \text{si } i < j \end{cases}$$

Al sumar $-\infty$ antes del softmax, los scores de posiciones futuras ($j > i$) se anulan exactamente: $\exp(-\infty) = 0$. Esto garantiza que la prediccion en la posicion $i$ solo use tokens anteriores.

**Motivo del escalado por $\sqrt{d_k}$:** si $q_i, k_i \overset{iid}{\sim} \mathcal{N}(0, 1)$, entonces:

$$\text{Var}\left[\mathbf{q}^\top \mathbf{k}\right] = \text{Var}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

Sin escalar, la varianza crece con $d_k$, llevando a la saturacion del softmax y gradientes nulos. Al dividir por $\sqrt{d_k}$, la varianza se normaliza a 1.

### 1.4 Multi-Head Attention

Con $h$ cabezas, dimension por cabeza $d_k = d_v = d_{model}/h$:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)$$

$$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

con $\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$.

**Parametros totales de MHA:** $4d_{model}^2$ (tres proyecciones de entrada + proyeccion de salida, cada una $d_{model} \times d_{model}$).

### 1.5 Feed-Forward Network

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

con $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $d_{ff} = 4 d_{model}$ tipicamente.

**Activacion GeLU** (usada en GPT-2/3, BERT):

$$\text{GeLU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

**Activacion SwiGLU** (LLaMA, PaLM):

$$\text{SwiGLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}) = \text{Swish}_1(\mathbf{x}\mathbf{W}) \odot (\mathbf{x}\mathbf{V})$$

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

SwiGLU usa tres matrices ($\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$) con $d_{ff} = \frac{2}{3} \cdot 4 d_{model}$ para mantener el conteo de parametros.

### 1.6 Layer Normalization

$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$

**RMSNorm** (T5, LLaMA): elimina el centrado, solo escala:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

Mas eficiente computacionalmente y empiricamente equivalente a LN en LLMs.

### 1.7 Weight Tying

La matriz de la cabeza de prediccion $\mathbf{W}_{out} \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$ comparte pesos con la tabla de embeddings $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$:

$$\mathbf{W}_{out} = \mathbf{E}$$

Esto reduce el numero de parametros en $|\mathcal{V}| \cdot d_{model}$ (tipicamente 50K × 4096 ≈ 200M params para modelos grandes) y mejora la generalizacion al forzar consistencia entre la representacion de entrada y salida de cada token.

### 1.8 Conteo de parametros

Para un Transformer decoder con $L$ capas, $d_{model}$, $h$ cabezas, $d_{ff} = 4d_{model}$, vocabulario $|\mathcal{V}|$:

| Componente | Parametros |
|-----------|-----------|
| Tabla de embeddings | $|\mathcal{V}| \cdot d_{model}$ |
| MHA por capa ($\times L$) | $4 d_{model}^2$ |
| FFN por capa ($\times L$) | $2 d_{model} \cdot d_{ff} = 8 d_{model}^2$ |
| LN por capa ($\times 2L$) | $2 d_{model}$ |
| LN final | $d_{model}$ |
| **Total aprox.** | $|\mathcal{V}| d_{model} + 12 L d_{model}^2$ |

Para GPT-3 ($L=96$, $d_{model}=12288$, $|\mathcal{V}|=50257$): $\approx 175 \times 10^9$ parametros.

---

## 2. Preentrenamiento a Escala

### 2.1 Objetivo de modelado de lenguaje

El objetivo autorregresivo (next-token prediction) maximiza:

$$\mathcal{L}_{LM} = \frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

La prediccion del token $x_t$ usa la representacion de la ultima capa en la posicion $t-1$:

$$p_\theta(x_t \mid x_{<t}) = \text{softmax}(\mathbf{h}_{t-1}^L \mathbf{W}_{out}^\top)_{x_t}$$

### 2.2 Optimizacion: AdamW

AdamW (Adam con weight decay desacoplado) es el optimizador estandar para LLMs:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \alpha \lambda \theta_t$$

donde el ultimo termino es el weight decay **desacoplado** del gradiente (a diferencia de L2 regularization clasica que lo acopla). Valores tipicos: $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$, $\lambda \in [0.1, 0.01]$.

### 2.3 Learning rate scheduling

**Warmup + Cosine Decay** (estandar en LLMs):

$$\alpha(t) = \begin{cases}
\alpha_{max} \cdot \dfrac{t}{t_{warmup}} & \text{si } t \leq t_{warmup} \\[8pt]
\alpha_{min} + \dfrac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\!\left(\pi \cdot \dfrac{t - t_{warmup}}{T_{total} - t_{warmup}}\right)\right) & \text{si } t > t_{warmup}
\end{cases}$$

El warmup evita que los primeros pasos, donde los momentos de Adam no estan calibrados, destruyan la inicializacion.

### 2.4 Gradient Clipping

Para evitar la explosion del gradiente:

$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{si } \|\mathbf{g}\|_2 \leq \tau \\
\dfrac{\tau}{\|\mathbf{g}\|_2} \mathbf{g} & \text{si } \|\mathbf{g}\|_2 > \tau
\end{cases}$$

con $\tau = 1.0$ tipicamente. Preserva la direccion del gradiente, solo controla su magnitud.

### 2.5 Procesamiento del corpus

**Concatenacion y segmentacion:** el corpus se concatena en una unica secuencia con tokens de separacion de documentos (`<|endoftext|>`) y se divide en chunks de longitud fija $T_{max}$ (contexto del modelo):

$$\text{chunks} = \text{split}(\text{concat}(d_1, \langle\text{eos}\rangle, d_2, \langle\text{eos}\rangle, \ldots), T_{max})$$

**Data mixing:** multiples fuentes de datos se mezclan con pesos asignados. Ejemplo aproximado de Llama-3:

| Fuente | Proporcion |
|--------|-----------|
| Common Crawl (web) | ~80% |
| Wikipedia / Wikidata | ~5% |
| Libros / dominio especifico | ~10% |
| Codigo (GitHub) | ~5% |

---

## 3. Leyes de Escala

### 3.1 Ley de potencias de Kaplan et al. (2020)

La perdida de test en un LLM sigue una ley de potencias respecto al numero de parametros $N$, tokens de entrenamiento $D$, y compute $C$:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

donde $N_c$, $D_c$, $C_c$ son constantes de normalizacion. La relacion entre compute, parametros y tokens es:

$$C \approx 6ND \quad \text{[FLOPs]}$$

El factor 6 proviene de: $\approx 2ND$ en forward pass (multiplicacion y suma) y $\approx 4ND$ en backward pass.

### 3.2 Ley de Chinchilla (Hoffmann et al., 2022)

Kaplan et al. recomendaban escalar $N$ mas agresivamente que $D$ con compute fijo. Hoffmann et al. demostraron que el optimo **computa-fijo** se alcanza cuando:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

Es decir, parametros y tokens deben escalar **proporcionalmente**. La regla empirica de Chinchilla:

$$D_{opt} \approx 20 \cdot N$$

Ejemplo: un modelo de 70B parametros se entrena de forma optima con $\approx 1.4T$ tokens.

### 3.3 Perdida conjunta

La perdida como funcion de $N$ y $D$ simultaneamente:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

donde:
- $E \approx 1.69$: entropia irreducible del lenguaje natural.
- $A, B, \alpha, \beta$: constantes ajustadas empiricamente.
- $A/N^\alpha$: sesgo por capacidad finita del modelo.
- $B/D^\beta$: sesgo por datos de entrenamiento insuficientes.

Con compute $C = 6ND$ fijo, la asignacion optima se obtiene minimizando $L(N, D)$ sujeto a $C = 6ND$:

$$\frac{\partial L}{\partial N}\bigg|_{D=C/6N} = 0 \implies \frac{\alpha A}{N^{\alpha+1}} = \frac{\beta B}{D^{\beta+1}}$$

---

## 4. Tokenizacion Avanzada

### 4.1 Byte-Pair Encoding (BPE)

Algoritmo iterativo de construccion del vocabulario:

1. Inicializar con todos los caracteres UTF-8 como simbolos base.
2. Calcular la frecuencia de todos los pares de simbolos consecutivos.
3. Fusionar el par mas frecuente $(a, b) \to ab$ y agregar $ab$ al vocabulario.
4. Repetir hasta alcanzar el tamaño de vocabulario objetivo $|\mathcal{V}|$.

**Propiedad clave:** las palabras frecuentes quedan como tokens completos; las raras se segmentan en subpalabras conocidas, eliminando el problema de tokens fuera de vocabulario (OOV).

### 4.2 WordPiece (BERT)

Similar a BPE pero el criterio de fusion maximiza la verosimilitud del corpus en lugar de la frecuencia del par:

$$\text{score}(a, b) = \frac{f(ab)}{f(a) \cdot f(b)}$$

Fusiona el par cuya ocurrencia conjunta es mas sorprendente dado el producto de sus frecuencias individuales.

### 4.3 SentencePiece

Tokeniza directamente sobre texto crudo sin presegmentacion por espacios (necesario para lenguas como chino, japones, tailandes). Modela el espacio como un caracter especial `▁`. Usa BPE o modelos de lenguaje de unigrama como algoritmo subyacente.

**Modelo de unigrama:** asigna a cada segmentacion posible $\mathbf{x} = (x_1, \ldots, x_n)$ la probabilidad:

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i)$$

y selecciona la segmentacion de mayor probabilidad (Viterbi).

### 4.4 Fertilidad

La **fertilidad** de un tokenizador mide cuantos tokens genera por palabra:

$$\text{fertility}(\mathcal{V}, \mathcal{D}) = \frac{\text{n tokens}(\mathcal{D})}{\text{n palabras}(\mathcal{D})}$$

Vocabularios mas grandes tienen menor fertilidad (mas palabras completas como tokens) pero mayor tamaño de la tabla de embeddings.

---

## 5. Mejoras Arquitectonicas Modernas

### 5.1 Rotary Positional Embedding (RoPE)

RoPE (Su et al., 2021) codifica la posicion rotando los vectores Q y K antes del producto interno. Para un par de dimensiones $(2i, 2i+1)$, la rotacion de un vector en la posicion $m$ es:

$$\mathbf{R}_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2i/d_k}$$

El producto interno entre query en posicion $m$ y key en posicion $n$ resulta:

$$(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$$

Es decir, el score de atencion solo depende de la **posicion relativa** $n - m$, no de las posiciones absolutas. Esto da al modelo conciencia de distancias relativas y permite extrapolacion a longitudes no vistas en entrenamiento.

### 5.2 Grouped Query Attention (GQA)

En la atencion estandar, cada cabeza tiene sus propias matrices $\mathbf{W}^K$ y $\mathbf{W}^V$.

**Multi-Query Attention (MQA):** una sola cabeza de K y V compartida por todas las cabezas de Q. Reduce la memoria del KV-Cache en un factor $h$ pero puede degradar la calidad.

**Grouped Query Attention:** $G$ grupos de cabezas Q, cada grupo comparte una cabeza de K y V. Es un compromiso entre MHA y MQA:

$$\text{head}_{i,g} = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_g, \mathbf{V}_g), \quad g = \lceil ih/G \rceil$$

Llama-3 70B usa $h=64$ cabezas Q y $G=8$ cabezas KV.

**Reduccion de memoria:** el KV-Cache se reduce de $O(h \cdot d_k \cdot T)$ a $O(G \cdot d_k \cdot T)$.

### 5.3 KV-Cache

Durante la inferencia autorregresiva, en cada paso $t$ se computan nuevas proyecciones K y V solo para el token nuevo $x_t$. Los K y V de posiciones anteriores se almacenan en cache:

$$\mathbf{K}^{(t)} = [\mathbf{K}^{(t-1)}; \mathbf{k}_t], \quad \mathbf{V}^{(t)} = [\mathbf{V}^{(t-1)}; \mathbf{v}_t]$$

**Reduccion de computo:** sin cache, generar $T$ tokens requiere $O(T^2)$ multiplicaciones; con cache, $O(T)$. La memoria aumenta linealmente en $T$.

**Tamaño del KV-Cache** para un batch de tamaño $B$, longitud de contexto $T$, $L$ capas, $G$ cabezas KV:

$$\text{KV-Cache} = 2 \cdot B \cdot T \cdot L \cdot G \cdot d_k \cdot \text{bytes\_por\_elemento}$$

En float16 (2 bytes), Llama-3 70B con $T=8192$: $\approx 8$ GB solo de KV-Cache.

### 5.4 Flash Attention

Rossen et al. (2022) reordenan los computos para evitar materializar la matriz de atencion completa $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k} \in \mathbb{R}^{T \times T}$ en memoria HBM (High Bandwidth Memory):

$$\text{Complejidad estandar: } O(T^2) \text{ en HBM}$$
$$\text{Flash Attention: } O(T^2/M) \text{ accesos HBM}, \quad M = \text{SRAM size}$$

Usa **tiling** (bloques de la matriz) y el **log-sum-exp trick** para calcular el softmax de forma numericamente estable en bloques:

$$\text{LSE}(x_1, \ldots, x_n) = \log \sum_{i=1}^n e^{x_i} = c + \log \sum_{i=1}^n e^{x_i - c}, \quad c = \max_i x_i$$

Flash Attention 2/3 es 2-3x mas rapido que la implementacion naive y reduce el uso de memoria de $O(T^2)$ a $O(T)$ (excluyendo Q, K, V).

### 5.5 Extended Context: Ventanas de Atencion y ALiBi

**Sparse Attention:** en lugar de atender a todos los $T$ tokens, cada token atiende solo a un subconjunto estructurado (ventana local + atencion global selectiva). Longformer usa una ventana de tamano $w$:

$$\text{atencion local: } O(T \cdot w) \text{ en lugar de } O(T^2)$$

**ALiBi (Press et al., 2021):** en lugar de sumar embeddings posicionales, se resta un bias lineal proporcional a la distancia:

$$\text{score}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

donde $m_h$ es una pendiente especifica por cabeza. El modelo aprende a penalizar atenciones a larga distancia. Permite extrapolacion a longitudes mayores que las de entrenamiento sin degradacion severa.

---

## 6. Ajuste Fino Supervisado

### 6.1 Instruction Tuning (SFT)

Dado un dataset de pares (instruccion, respuesta) $\{(x^{(i)}, y^{(i)})\}$, el ajuste fino minimiza:

$$\mathcal{L}_{SFT}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y^{(i)}|} \log p_\theta(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)})$$

**Punto clave:** la perdida se calcula solo sobre los tokens de la respuesta $y$, no sobre los tokens de la instruccion $x$. Los tokens de instruccion se enmascaran (loss mask = 0).

El formato de prompt sigue una plantilla estandar, por ejemplo para Llama:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{assistant_response}<|eot_id|>
```

### 6.2 Catastrophic Forgetting

El ajuste fino en un dataset pequeno puede degradar las capacidades generales adquiridas en el preentrenamiento. Estrategias de mitigacion:

**Elastic Weight Consolidation (EWC):**

$$\mathcal{L}_{EWC}(\theta) = \mathcal{L}_{SFT}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

donde $F_i$ es la diagonal de la **matriz de informacion de Fisher**:

$$F_i = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{pre}}\left[\left(\frac{\partial \log p_\theta(x)}{\partial \theta_i}\right)^2\right]$$

Penaliza cambios en parametros importantes para el preentrenamiento, ponderados por su relevancia (estimada por la curvatura del loss).

---

## 7. RLHF y Alineacion

### 7.1 Reward Modeling

Dado un conjunto de comparaciones humanas $\{(x, y_w, y_l)\}$ donde $y_w$ es la respuesta preferida y $y_l$ la rechazada, se entrena el modelo de recompensa $r_\phi(x, y)$ con el modelo de **Bradley-Terry**:

$$p(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

La perdida de entrenamiento es la log-verosimilitud negativa de las preferencias:

$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

### 7.2 PPO para LLMs

Proximal Policy Optimization (Schulman et al., 2017) optimiza la politica $\pi_\theta$ maximizando la recompensa con restriccion KL:

$$\mathcal{J}_{RLHF}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}\left[r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta(y \mid x) \| \pi_{ref}(y \mid x))\right]$$

El termino KL desacoplado se estima token por token:

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{ref}(y_t \mid x, y_{<t})}$$

La funcion objetivo de PPO con clipping:

$$\mathcal{L}_{PPO}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

donde $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$ es el ratio de probabilidades y $\hat{A}_t$ es la funcion de ventaja estimada.

**Ventaja generalizada (GAE):**

$$\hat{A}_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 7.3 DPO (Direct Preference Optimization)

Rafailov et al. (2023) derivaron que la solucion optima del problema de RL con restriccion KL tiene forma cerrada:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\left(\frac{1}{\beta} r^*(x, y)\right)$$

donde $Z(x) = \sum_y \pi_{ref}(y \mid x) \exp(r^*(x, y)/\beta)$ es la funcion de particion.

Invirtiendo esta relacion, la recompensa optima expresada en terminos de las politicas es:

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x)$$

Sustituyendo en el modelo de Bradley-Terry (donde $Z(x)$ se cancela):

$$p^*(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$$

La perdida DPO entrena directamente $\pi_\theta$ sin necesidad de un modelo de recompensa separado:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]$$

**Gradiente de DPO:** la actualizacion aumenta la probabilidad de las respuestas preferidas y reduce la de las rechazadas, ponderado por cuanto el modelo actual ya las distingue:

$$\nabla_\theta \mathcal{L}_{DPO} \propto -\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))\left[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\right]$$

donde $\hat{r}_\theta(x, y) = \beta \log \pi_\theta(y|x)/\pi_{ref}(y|x)$ es la recompensa implicita.

### 7.4 Constitutional AI (CAI)

Anthropic (2022) propone usar el propio LLM para autocrticar y revisar sus respuestas segun un conjunto de principios ($\mathcal{C}$, la "constitucion"):

1. **Critique:** $y_c = \text{LLM}(x, y_0, \text{"critica segun } \mathcal{C}\text{"})$
2. **Revision:** $y_1 = \text{LLM}(x, y_0, y_c, \text{"revisa segun } \mathcal{C}\text{"})$
3. **RL-CAI:** usar preferencias generadas por el modelo como señal de entrenamiento.

---

## 8. Inferencia Eficiente

### 8.1 Cuantizacion

La cuantizacion reduce la precision numerica de los pesos para disminuir memoria y acelerar el computo.

**Cuantizacion uniforme:** mapea pesos de punto flotante $w \in [\alpha, \beta]$ a enteros de $b$ bits:

$$w_q = \text{round}\!\left(\frac{w - \alpha}{\Delta}\right), \quad \Delta = \frac{\beta - \alpha}{2^b - 1}$$

**Dequantizacion:** $\tilde{w} = w_q \cdot \Delta + \alpha$.

**Error de cuantizacion:** $\|w - \tilde{w}\|^2 \leq \Delta^2 / 4$.

**GPTQ** (Frantar et al., 2022): cuantizacion post-entrenamiento de 4 bits que minimiza el error de reconstruccion capa por capa:

$$\min_{\hat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_F^2$$

usando el Hessiano del error para compensar los errores de cuantizacion de forma iterativa.

**NF4 (NormalFloat-4):** cuantizacion de 4 bits para distribuciones normales (tipica de pesos de redes neuronales). Los niveles de cuantizacion se distribuyen segun los cuantiles de $\mathcal{N}(0,1)$, minimizando el error esperado para pesos gaussianos.

### 8.2 Especulative Decoding

Usa un modelo pequeño (draft model $M_q$, $\approx 7B$) para generar $K$ tokens candidatos, y el modelo grande (target $M_p$, $\approx 70B$) los verifica en paralelo:

1. $M_q$ genera $K$ tokens autoregresivamente.
2. $M_p$ procesa los $K$ tokens en una sola pasada (paralelizable).
3. Se aceptan los tokens donde $M_p(x) \geq M_q(x)$; el primero rechazado se remuestrea.

**Aceptacion con rechazo corregido:** para garantizar que la distribucion de salida sea exactamente $M_p$:

$$\alpha_t = \min\!\left(1, \frac{p_t}{q_t}\right)$$

donde $p_t = M_p(x_t \mid x_{<t})$ y $q_t = M_q(x_t \mid x_{<t})$.

El **speedup** teorico con $K$ tokens draft y tasa de aceptacion $\alpha$:

$$\text{speedup} = \frac{K + 1}{1 + (1 - \alpha^K) / (1 - \alpha)} \approx \frac{1}{1 - \alpha} \quad \text{si } \alpha \to 1$$

### 8.3 Continuous Batching

En la inferencia autorregresiva, distintas secuencias terminan en momentos diferentes. El **static batching** mantiene el batch hasta que la secuencia mas larga termina, desperdiciando computo.

**Continuous batching:** cuando una secuencia termina, su slot en el batch se reutiliza inmediatamente para una nueva secuencia, sin esperar al resto del batch. Aumenta el throughput en 2-4x en produccion.

---

## 9. Adaptacion Eficiente en Parametros (PEFT)

### 9.1 LoRA (Low-Rank Adaptation)

La hipotesis de LoRA (Hu et al., 2021): las actualizaciones de pesos durante el ajuste fino tienen **rango intrisecamente bajo**. Por tanto, se parametriza la actualizacion $\Delta \mathbf{W}$ como:

$$\mathbf{W}' = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

con $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ congelada, $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

**Inicializacion:** $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$, $\mathbf{B} = \mathbf{0}$, de modo que $\Delta \mathbf{W} = \mathbf{0}$ al inicio. Esto garantiza que el modelo comienza el fine-tuning desde el comportamiento original del modelo preentrenado.

**Escala:** la actualizacion se escala por $\alpha/r$:

$$\mathbf{W}' = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$$

donde $\alpha$ es un hiperparametro (tipicamente $\alpha = r$ o $\alpha = 2r$).

**Reduccion de parametros:** para $d = k = 4096$, $r = 16$:

$$\text{Parametros LoRA} = r(d + k) = 16 \times 8192 = 131{,}072$$
$$\text{Parametros originales} = d \times k = 4096^2 = 16{,}777{,}216$$
$$\text{Reduccion} = 128\times$$

**Aplicacion tipica:** LoRA se aplica a las matrices $\mathbf{W}^Q$ y $\mathbf{W}^V$ de cada capa de atencion.

### 9.2 QLoRA

Dettmers et al. (2023) combinan cuantizacion y LoRA:

1. Cuantizar el modelo base a 4 bits (NF4).
2. Agregar adaptadores LoRA en BFloat16.
3. Durante el forward, dequantizar los pesos a BFloat16 para el computo.
4. El gradiente fluye solo por los adaptadores LoRA (los pesos base permanecen congelados y cuantizados).

Permite ajustar un modelo de 65B en una GPU de 48GB con degradacion minima de calidad respecto al SFT completo.

**Memoria:**

$$\text{Memoria}_{QLoRA} \approx \frac{N \cdot 4}{8} + \text{Activaciones} + r(d+k) \cdot L \cdot \text{sizeof(bf16)}$$

donde el primer termino es el modelo a 4 bits.

### 9.3 Prompt Tuning y Prefix Tuning

**Prompt Tuning** (Lester et al., 2021): añadir $k$ tokens "blandos" (soft tokens) entrenables al inicio de la secuencia, manteniendo el modelo congelado:

$$p_\theta(y \mid [\mathbf{P}; x]) \quad \mathbf{P} \in \mathbb{R}^{k \times d_{model}}$$

Solo se actualizan $k \cdot d_{model}$ parametros (tipicamente $k \in [1, 100]$).

**Prefix Tuning** (Li & Liang, 2021): prepone vectores entrenables a las K y V de cada capa:

$$\mathbf{K}_l' = [\mathbf{P}_l^K; \mathbf{K}_l], \quad \mathbf{V}_l' = [\mathbf{P}_l^V; \mathbf{V}_l]$$

Mas expresivo que prompt tuning pero con mas parametros: $2L \cdot k \cdot d_{model}$.

---

## 10. Capacidades Emergentes

### 10.1 In-Context Learning

Los LLMs pueden adaptarse a nuevas tareas viendo ejemplos en el prompt sin actualizar pesos:

$$p_\theta(y \mid x, \mathcal{C}) = p_\theta(y \mid \underbrace{(x_1, y_1), \ldots, (x_k, y_k)}_{\mathcal{C}}, x)$$

Donde $\mathcal{C}$ son las $k$ demostraciones en el contexto. Esta capacidad **emerge** a cierta escala y no es explicable como interpolacion de la distribucion de entrenamiento.

### 10.2 Chain-of-Thought (CoT)

Wei et al. (2022) demostraron que pedir al modelo que genere pasos intermedios mejora drasticamente el razonamiento en tareas aritmeticas y logicas:

**Zero-shot CoT:** añadir "Pensemos paso a paso" al prompt.

**Few-shot CoT:** incluir ejemplos con razonamiento explicito:

```
Q: Roger tiene 5 pelotas. Compra 2 cajas de 3. ¿Cuantas pelotas tiene?
A: Roger comienza con 5. 2 x 3 = 6 pelotas nuevas. 5 + 6 = 11. La respuesta es 11.
```

**Fundamento matematico:** el CoT permite al modelo realizar computos distribuidos en multiples pasos de generacion, aumentando efectivamente el "computo" disponible de $O(d_{ff})$ por token a $O(T_{CoT} \cdot d_{ff})$.

### 10.3 Emergencia y Cambios de Fase

Algunas capacidades de los LLMs aparecen abruptamente al superar cierto umbral de escala, no de forma gradual. Wei et al. (2022) documentaron mas de 100 capacidades emergentes.

**Hipotesis de la metrica:** muchas emergencias son artefactos de metricas no lineales (ej. accuracy exacta vs. probabilidad parcial). Schaeffer et al. (2023) argumentan que bajo metricas continuas, las curvas son suaves.

### 10.4 Retrieval-Augmented Generation (RAG)

Extiende el LLM con acceso a una base de conocimiento externa $\mathcal{K}$:

$$p_\theta(y \mid x) = \sum_{z \in \mathcal{K}} p_\theta(y \mid x, z) \cdot p_\eta(z \mid x)$$

donde $p_\eta(z \mid x)$ es la probabilidad de recuperar el documento $z$ dado el query $x$.

En la practica se usa **recuperacion densa** (Dense Passage Retrieval):

$$\text{score}(q, z) = \text{sim}(\mathbf{E}_q(q), \mathbf{E}_z(z)) = \frac{\mathbf{E}_q(q)^\top \mathbf{E}_z(z)}{\|\mathbf{E}_q(q)\| \|\mathbf{E}_z(z)\|}$$

donde $\mathbf{E}_q$ y $\mathbf{E}_z$ son encoders de query y documentos (tipicamente BERT o similar).

---

## 11. Evaluacion de LLMs

### 11.1 Perplejidad

$$\text{PPL}(\mathcal{D}_{test}) = \exp\!\left(-\frac{1}{\sum_i T_i} \sum_i \sum_{t=1}^{T_i} \log p_\theta(x_t^{(i)} \mid x_{<t}^{(i)})\right)$$

Metrica intrinseca que mide la incertidumbre promedio del modelo por token. No mide directamente utilidad en tareas downstream.

### 11.2 Benchmarks Estandar

| Benchmark | Tarea | Metrica |
|-----------|-------|---------|
| **MMLU** (Hendrycks et al.) | 57 materias academicas | Accuracy |
| **HellaSwag** (Zellers et al.) | Completacion de oraciones | Accuracy |
| **HumanEval** (Chen et al.) | Generacion de codigo | pass@k |
| **GSM8K** (Cobbe et al.) | Matematicas de primaria | Accuracy |
| **MATH** (Hendrycks et al.) | Matematicas olimpiadas | Accuracy |
| **BIG-Bench Hard** | 23 tareas dificiles | Varios |
| **MT-Bench** (Zheng et al.) | Dialogo multi-turno | GPT-4 score |
| **GPQA** (Rein et al.) | Preguntas expertas | Accuracy |

### 11.3 pass@k para codigo

La metrica `pass@k` estima la probabilidad de que al menos una de $k$ muestras generadas pase los tests:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

donde $n$ es el numero total de muestras generadas por problema y $c$ es el numero de muestras correctas. Se usa $n > k$ para reducir la varianza del estimador.

### 11.4 LLM-as-Judge

Usar GPT-4 u otro LLM potente para evaluar la calidad de respuestas. La correlacion con juicios humanos es alta ($\rho > 0.8$) para tareas de respuesta abierta donde la evaluacion automatica clasica falla.

**Sesgos conocidos del LLM-judge:**
- **Sesgo de posicion:** prefiere respuestas en la primera posicion.
- **Sesgo de verbosidad:** prefiere respuestas mas largas.
- **Sesgo de auto-preferencia:** GPT-4 prefiere respuestas estilo GPT-4.

Mitigacion: evaluar en ambos ordenes y promediar; normalizar por longitud.

### 11.5 RLHF Reward Model como proxy de alineacion

El **reward hacking** ocurre cuando la politica maximiza la recompensa del RM sin mejorar la calidad real:

$$r_{human}(\pi) \not\approx r_{RM}(\pi) \quad \text{si } \pi \text{ se aleja mucho de } \pi_{SFT}$$

El termino KL en RLHF actua como regularizador para evitar este problema. El coeficiente $\beta$ controla el compromiso entre maximizar recompensa y mantenerse cerca de la politica de referencia:

$$\pi^* = \arg\max_\pi \mathbb{E}_y[r(x,y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

---

## 12. Referencias

- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS 2020*.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. *arXiv:2001.08361*.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). *NeurIPS 2022*.
- Su, J., et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv:2104.09864*.
- Ainslie, J., et al. (2023). GQA: Training generalized multi-query transformer models. *EMNLP 2023*.
- Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention. *NeurIPS 2022*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.
- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*.
- Rafailov, R., et al. (2023). Direct preference optimization. *NeurIPS 2023*.
- Hu, E., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*.
- Frantar, E., et al. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. *arXiv:2210.17323*.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.
- Wei, J., et al. (2022). Emergent abilities of large language models. *TMLR 2022*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Leviathan, Y., et al. (2023). Fast inference from transformers via speculative decoding. *ICML 2023*.
- Press, O., et al. (2021). Train short, test long: Attention with linear biases (ALiBi). *ICLR 2022*.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Sennrich, R., et al. (2016). Neural machine translation of rare words with subword units. *ACL 2016*.
- Lester, B., et al. (2021). The power of scale for parameter-efficient prompt tuning. *EMNLP 2021*.
- Li, X.L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. *ACL 2021*.

---

<!-- IMAGEN: Diagrama de la arquitectura GPT completa: embedding + positional encoding + L bloques decoder (MHA causal + FFN + Pre-LN + residual) + cabeza de prediccion con weight tying -->

<!-- IMAGEN: Curvas de scaling laws: loss vs. N (parametros), D (tokens) y C (compute) en escala log-log mostrando las leyes de potencias -->

<!-- IMAGEN: Diagrama del pipeline RLHF completo: SFT -> Reward Model -> PPO con las cuatro fases y flujo de datos -->

<!-- IMAGEN: Comparativa de MHA vs MQA vs GQA mostrando las cabezas Q, K, V y su agrupacion -->

<!-- IMAGEN: Diagrama de LoRA: matriz W0 congelada con la rama paralela B*A de bajo rango, mostrando el flujo del gradiente -->

<!-- IMAGEN: Visualizacion de RoPE: rotacion de vectores Q y K en el plano complejo por posicion, mostrando que el producto interno solo depende de la diferencia de posiciones -->
