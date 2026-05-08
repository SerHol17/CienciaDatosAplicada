# 1. Deep Learning Avanzado

## Introduccion

El Deep Learning es la disciplina que ha redefinido los limites de lo que las maquinas pueden aprender. A diferencia del machine learning clasico, donde un ingeniero debia disenar manualmente las caracteristicas relevantes de los datos, las redes neuronales profundas aprenden esas representaciones directamente desde los datos crudos, capa por capa, con una profundidad que permite capturar estructuras de creciente complejidad y abstraccion.

Esta unidad presenta los bloques fundamentales del Deep Learning moderno: desde las redes convolucionales que revolucionaron la vision artificial, hasta los modelos multimodales que integran texto, imagen y audio en un mismo espacio de representacion. Se recorren ademas las principales familias de arquitecturas generativas, los modelos de segmentacion, los detectores de objetos en tiempo real y los modelos de lenguaje basados en Transformers.

---

## Deep Learning: fundamentos y evolucion

Una red neuronal profunda es una funcion parametrizada $f_\theta$ compuesta por multiples capas de transformaciones no lineales:

$$f_\theta(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

Cada capa $f_l$ aprende a transformar su entrada en una representacion mas util para la tarea objetivo. El entrenamiento ajusta los parametros $\theta$ mediante **retropropagacion del gradiente** y **descenso por gradiente estocastico (SGD)**, minimizando una funcion de perdida $\mathcal{L}$.

La profundidad es la clave: las primeras capas detectan patrones simples (bordes, frecuencias), las capas intermedias combinan esos patrones en estructuras (formas, texturas) y las capas finales construyen conceptos de alto nivel (objetos, semantica). Esta jerarquia de representaciones emerge del entrenamiento, sin ser programada explicitamente.

<img src="_static/images/intro1.png" width="600" />


### Por que el Deep Learning funciona ahora

Tres factores convergieron para hacer viable el Deep Learning a gran escala:

- **Datos**: la disponibilidad de datasets masivos (ImageNet, Common Crawl, LAION) proporciona el volumen necesario para entrenar modelos con miles de millones de parametros.
- **Computo**: las GPUs y TPUs permiten paralelizar las operaciones matriciales que dominan el entrenamiento y la inferencia.
- **Algoritmos**: avances en funciones de activacion (ReLU), normalizacion (Batch Normalization), regularizacion (Dropout) y arquitecturas (residual connections) resolvieron problemas que bloqueaban el entrenamiento profundo.

---

## Redes Neuronales Convolucionales (CNNs)

Las **Redes Neuronales Convolucionales** son la arquitectura dominante en vision artificial. Su principio es la **convolucion**: en lugar de conectar cada neurona con todos los pixeles de la imagen (ineficiente y sobreajustado), cada filtro se desplaza por la imagen compartiendo pesos, lo que produce un **mapa de activacion** que registra donde aparece el patron que el filtro detecta.

$$\text{Salida}[i,j] = \sum_{u,v} W[u,v] \cdot X[i+u, j+v] + b$$

Las propiedades que hacen a las CNNs poderosas para imagenes son:

- **Comparticion de pesos**: el mismo filtro detecta un borde horizontal en cualquier posicion de la imagen.
- **Invarianza a la traslacion**: gracias al pooling, la red reconoce un objeto independientemente de donde aparezca.
- **Jerarquia local**: las capas sucesivas combinan patrones locales en estructuras globales.

<img src="_static/images/intro2.png" width="600" />

Los hitos arquitectonicos que marcaron el campo incluyen LeNet (1998), AlexNet (2012, que inicio el boom del Deep Learning moderno), VGGNet (uniformidad y profundidad), ResNet (2015, conexiones residuales que permiten entrenar redes de cientos de capas), y EfficientNet (escalado sistematico de anchura, profundidad y resolucion).

Las CNNs son la base sobre la que se construyen la mayoria de las arquitecturas de esta unidad.

---

## Multimodalidad

Los modelos multimodales procesan y relacionan informacion de **multiples tipos de datos simultaneamente**: texto, imagen, audio, video, profundidad, codigo. La hipotesis central es que las distintas modalidades son proyecciones de una misma realidad, y que alinearlas en un espacio de representacion compartido permite una comprension mas rica que la de cada modalidad por separado.

El modelo **CLIP** (Contrastive Language-Image Pretraining, OpenAI 2021) demostro que es posible alinear texto e imagen entrenando con pares de la forma (imagen, descripcion) extraidos de internet. La imagen y su descripcion se acercan en el espacio latente; la imagen y una descripcion incorrecta se alejan. Este entrenamiento contrastivo produjo representaciones visuales transferibles a decenas de tareas sin fine-tuning especifico.

<img src="_static/images/intro3.png" width="600" />

Los modelos fundacionales actuales como GPT-4o, Gemini y Claude son inherentemente multimodales: reciben texto, imagenes, audio y documentos, y producen respuestas que integran el contenido de todas las modalidades. Esta convergencia multimodal es una de las tendencias mas significativas del Deep Learning contemporaneo.

---

## Autoencoders

Un **autoencoder** es una red neuronal que aprende a comprimir una entrada en una representacion de menor dimension (el **espacio latente**) y a reconstruirla desde esa representacion comprimida. No requiere etiquetas: la propia entrada es el objetivo de aprendizaje.

$$z = f_\theta(x) \qquad \hat{x} = g_\phi(z) \qquad \mathcal{L} = \|x - \hat{x}\|^2$$

El cuello de botella fuerza a la red a descartar redundancia y retener solo la informacion estructuralmente relevante. Lo que emerge en el espacio latente es una representacion compacta que captura los factores de variacion principales de los datos.

<img src="_static/images/autoencoder1.png" width="600" />

Las variantes mas importantes son el **Variational Autoencoder (VAE)**, que aprende una distribucion de probabilidad sobre el espacio latente permitiendo generar muestras nuevas, y el **Denoising Autoencoder**, que entrena la red para recuperar imagenes limpias a partir de entradas corrompidas con ruido, produciendo representaciones mas robustas.

Las aplicaciones de los autoencoders van desde la reduccion de dimensionalidad y la deteccion de anomalias (un dato que no se reconstruye bien es probablemente una anomalia) hasta la sintesis de datos y el preentrenamiento de representaciones para tareas supervisadas.

---

## Redes Generativas Adversariales (GANs)

Las **GANs** (Goodfellow et al., 2014) formulan la generacion como un juego competitivo entre dos redes:

- El **Generador** $G$ recibe un vector de ruido $z$ y produce una muestra sintetica $G(z)$.
- El **Discriminador** $D$ recibe una muestra (real o sintetica) y estima la probabilidad de que sea real.

Ambas redes se entrenan simultaneamente: $G$ aprende a engañar a $D$, y $D$ aprende a no ser engañado. La presion mutua impulsa a $G$ a producir muestras de calidad creciente.

$$\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

<img src="_static/images/gans1.png" width="600" />

La familia de GANs se ha expandido considerablemente. **DCGAN** introdujo arquitecturas convolucionales estables. **Pix2Pix** aplico el principio adversarial a la traduccion imagen a imagen. **CycleGAN** extendio esa idea a dominios sin pares de entrenamiento. **StyleGAN** llevo la generacion de imagenes de alta resolucion al nivel fotorrealista. **WGAN-GP** resolvio los problemas de inestabilidad y mode collapse mediante la distancia de Wasserstein.

Aunque los modelos de difusion han superado a las GANs en calidad de generacion de imagenes, las GANs siguen siendo relevantes en aplicaciones donde la velocidad de inferencia es critica.

---

## Segmentacion con IA

La **segmentacion semantica** asigna una etiqueta de clase a cada pixel de una imagen, produciendo una comprension densa de la escena a nivel de pixel. Es la tarea de percepcion visual mas exigente: requiere que la red comprenda tanto el contenido global de la imagen como la localizacion exacta de cada elemento.

La arquitectura **U-Net** (Ronneberger et al., 2015) se convirtio en el estandar del campo mediante una innovacion estructural: las **skip connections** que conectan directamente cada nivel del encoder con el nivel simetrico del decoder.

<!-- IMAGEN: arquitectura U-Net con las skip connections entre encoder y decoder resaltadas -->

Sin estas conexiones, el decoder debe reconstruir los detalles espaciales finos (bordes, contornos exactos) solo desde el bottleneck comprimido, lo que produce mascaras borrosas. Con ellas, los detalles de alta resolucion del encoder se combinan con el contexto semantico del decoder, produciendo mascaras precisas incluso con datasets pequeños.

Arquitecturas posteriores como **DeepLab v3+** (convolucion dilatada para capturar multiples escalas), **SegFormer** (Transformers para segmentacion eficiente) y **SAM** (Segment Anything Model, Meta 2023, segmentacion zero-shot guiada por prompts de lenguaje natural) han extendido las capacidades del campo a dominios y condiciones cada vez mas generales.

---

## Deteccion de Objetos: YOLO

La **deteccion de objetos** localiza y clasifica simultaneamente multiples objetos en una imagen, produciendo cajas delimitadoras (*bounding boxes*) con clase y puntuacion de confianza para cada objeto detectado.

**YOLO** (You Only Look Once, Redmon et al., 2016) redefinio el campo introduciendo la deteccion en una sola pasada por la red: la imagen se divide en una cuadricula $S \times S$ y cada celda predice directamente las cajas y clases de los objetos que contiene, sin etapas separadas de propuesta de regiones.

<!-- IMAGEN: diagrama de la cuadricula YOLO sobre una imagen con cajas predichas y clases por celda -->

Esta formulacion hace a YOLO extremadamente rapida: puede procesar video en tiempo real a 30 o mas fotogramas por segundo, lo que lo convierte en la arquitectura de referencia para aplicaciones de produccion. La familia YOLO ha evolucionado continuamente (v3, v5, v8, v11) mejorando la precision y añadiendo capacidades de segmentacion y estimacion de pose sobre la misma arquitectura base.

Los detectores de dos etapas como **Faster R-CNN** ofrecen mayor precision a costa de velocidad, mientras que **DETR** (Detection Transformer) elimino los componentes de diseño manual (anclas, NMS) mediante atención.

---

## NLP y Transformers

El procesamiento del lenguaje natural experimento una transformacion radical con la introduccion de la arquitectura **Transformer** (Vaswani et al., 2017). Su mecanismo central, la **atencion multi-cabeza**, permite a cada elemento de una secuencia relacionarse directamente con cualquier otro elemento, independientemente de la distancia entre ellos:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

<!-- IMAGEN: diagrama del mecanismo de atencion con matrices Q, K, V y el mapa de atencion resultante -->

Esta propiedad resolvio el problema de las redes recurrentes (LSTM, GRU) con dependencias de largo alcance, donde la informacion al principio de una secuencia se degradaba antes de influir en el final. Los Transformers procesan toda la secuencia en paralelo, lo que ademas los hace altamente eficientes en hardware moderno.

El preentrenamiento a gran escala sobre texto no etiquetado produjo modelos de proposito general de capacidad creciente. **BERT** (2018) aprendio representaciones bidireccionales optimizadas para comprension. **GPT** (2018 en adelante) aprendio a predecir el siguiente token, emergiendo capacidades de generacion, razonamiento y seguimiento de instrucciones a medida que el modelo escala. Los modelos actuales como GPT-4, Claude y Gemini tienen cientos de miles de millones de parametros y muestran capacidades que antes se consideraban exclusivas de la inteligencia humana.

Los Transformers son hoy la arquitectura dominante no solo en NLP sino en vision (Vision Transformer, ViT), audio, codigo y modelos multimodales, unificando el Deep Learning bajo un mismo paradigma arquitectonico.

---

## Estructura de la unidad

| Modulo | Contenido |
|--------|-----------|
| **Autoencoders** | Arquitecturas, espacio latente, VAE, aplicaciones |
| **GANs** | GAN vanilla, DCGAN, cGAN, WGAN-GP, evaluacion |
| **Segmentacion** | FCN, U-Net, funciones de perdida, metricas, augmentacion |
| **Deteccion de objetos** | YOLO, Faster R-CNN, DETR, metricas de deteccion |
| **NLP y Transformers** | Atencion, BERT, GPT, LLMs, fine-tuning, RAG |

Cada modulo incluye fundamentos teoricos, implementaciones en PyTorch y experimentos practicos sobre datasets reproducibles.