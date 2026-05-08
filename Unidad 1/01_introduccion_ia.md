# Tópicos Emergentes de la IA — 2026

## Contenidos

- [Módulo 1 — Introducción a la Inteligencia Artificial](#módulo-1-introducción-a-la-inteligencia-artificial)
- [Módulo 2 — Inteligencia Artificial Tradicional](#módulo-2-inteligencia-artificial-tradicional)
- [Módulo 3 — Deep Learning](#módulo-3-deep-learning)
- [Módulo 4 — Machine y Deep Learning en la Práctica](#módulo-4-machine-y-deep-learning-en-la-práctica)
- [Módulo 5 — Inteligencia Artificial Generativa](#módulo-5-inteligencia-artificial-generativa)
- [Módulo 6 — Modelos Grandes del Lenguaje (LLMs)](#módulo-6-modelos-grandes-del-lenguaje-llms)
- [Módulo 7 — Herramientas de IA Generativa](#módulo-7-herramientas-de-ia-generativa)
- [Módulo 8 — Contextualización de Modelos](#módulo-8-contextualización-de-modelos)
- [Módulo 9 — Técnicas de Prompting](#módulo-9-técnicas-de-prompting)
- [Módulo 10 — Retrieval Augmented Generation (RAG)](#módulo-10-retrieval-augmented-generation-rag)
- [Módulo 11 — Fine-Tuning para Ajuste de Modelos](#módulo-11-fine-tuning-para-ajuste-de-modelos)
- [Módulo 12 — Agentes de IA y Automatización Low-Code](#módulo-12-agentes-de-ia-y-automatización-low-code)

---

# Módulo 1 — Introducción a la Inteligencia Artificial

## ¿Qué es la Inteligencia Artificial?

La IA es cualquier técnica que permite a los ordenadores **imitar el comportamiento humano**.

<!-- IMAGEN: diagrama de círculos concéntricos AI > ML > DL -->

---

## Línea temporal de la IA

| Época | Paradigma | Descripción |
|-------|-----------|-------------|
| 1950's | **Inteligencia Artificial** | Cualquier técnica que permite a los ordenadores imitar el comportamiento humano. |
| 1980's | **Machine Learning** | Técnicas de IA que otorgan a los ordenadores la habilidad de aprender sin ser explícitamente programados para hacerlo. |
| 2010's | **Deep Learning** | Un subconjunto del ML que hace viable el cálculo de redes neuronales multicapa. |
| 2020's | **IA Generativa / Modelos Fundacionales** | Modelos altamente adaptables que proporcionan una estructura base para múltiples aplicaciones. |

---

## ¿Qué se necesita para hacer IA?

Los tres pilares fundamentales son:

```
        Algoritmos
           / \
          /   \
      Datos — Poder de cómputo
```

- **Datos**: materia prima del aprendizaje automático.
- **Algoritmos**: las instrucciones que transforman datos en modelos.
- **Poder de cómputo**: la capacidad de hardware para entrenar y ejecutar modelos.

<!-- IMAGEN: triángulo con los tres pilares -->

---

# Módulo 2 — Inteligencia Artificial Tradicional

## Flujo de un sistema con IA

Un proyecto de IA sigue un ciclo estructurado:

```
Problema a resolver + Datos
         |
   PREPROCESAMIENTO
         |
   PROCESAMIENTO  ←→  Algoritmos / Inteligencia Artificial
         |
    DESARROLLO   →   Aplicación
         |
 POST-PROCESAMIENTO  →  Soporte
```

> **Durante el desarrollo**: preprocesamiento, procesamiento, desarrollo.  
> **Después del desarrollo**: despliegue, monitoreo y soporte.

---

## De observaciones a predicciones mediante modelos

El objetivo del ML tradicional es aprender una función que mapee observaciones a predicciones:

| Observación (X) | Predicción (Y) |
|-----------------|----------------|
| Imagen del paciente | ¿Tiene patología? |
| Fragmento de texto | Sentimiento |
| Series temporales de finanzas | Tendencia del valor |
| Descripción de la casa | Valor de mercado |
| Perfil del cliente | Producto a comprar |

La terminología equivalente:

| X | Y |
|---|---|
| Variables predictoras | Variables objetivo |
| Variable independiente | Variable dependiente |
| Características (*features*) | Etiquetas (*labels*) |

---

## Proceso de construcción de un modelo

```
Datos de ejemplo (X → y)
        |
  [Modelo plantilla]
        |
  Calibración del modelo
        |
   MODELO entrenado
        |
   X nuevos → y predicciones
```

### Datos de entrenamiento vs. datos de validación

- **Datos de entrenamiento**: se usan para calibrar (ajustar) el modelo.
- **Datos de validación**: X, y **no vistos** durante el entrenamiento → miden la capacidad de generalización.

| | Rend. entrenamiento | Rend. validación |
|-|---------------------|------------------|
| Escenario ideal | BIEN | BIEN |
| Sobreajuste (*overfitting*) | BIEN | MAL |
| Subajuste (*underfitting*) | MAL | MAL |

> **Objetivo**: crear modelos que generalicen bien a datos nunca vistos.  
> El modelado debe enfrentarse a **problemas mal definidos** del mundo real.

---

## ML: Técnicas y Oportunidades

<!-- IMAGEN: mapa mental de algoritmos ML -->

### Aprendizaje supervisado
- **Clasificación**: predice una categoría discreta.
- **Regresión**: predice un valor continuo.

### Aprendizaje no supervisado
- **Clustering**: agrupa observaciones sin etiquetas.
- **Reducción de dimensionalidad**: comprime el espacio de características.

### Aprendizaje semi-supervisado
Combina datos etiquetados (pocos) y no etiquetados (muchos) para entrenar modelos más robustos.

<!-- IMAGEN: comparación supervisado vs semi-supervisado vs no supervisado -->

---

# Módulo 3 — Deep Learning

## ¿Por qué y cuándo usar Deep Learning?

<!-- IMAGEN: curva de rendimiento DL vs ML clásico en función del volumen de datos -->

El Deep Learning supera al ML clásico cuando:
- El **volumen de datos es grande**.
- Las relaciones entre variables son **altamente no lineales**.
- El dominio involucra señales complejas: imágenes, audio, texto, video.

> Referencia: [Deep Learning Specialization – Coursera](https://www.coursera.org/specializations/deep-learning)

---

## Tareas resueltas con Deep Learning

- **Clasificación de imágenes** a nivel casi humano (ej. esteganálisis).
- **Traducción automática** mejorada (ej. DeepL).
- **Conversión de texto a voz** de alta calidad.
- **Asistentes digitales** (ej. Alexa de Amazon).
- **Conducción autónoma** a nivel humano.
- **Segmentación de anuncios** (Google, Bing).
- **Respuesta a preguntas** en lenguaje natural.

---

## Contexto biológico: la neurona artificial

El DL modela la relación entre señales inspirándose en el funcionamiento del cerebro:

| Especie | Neuronas aprox. |
|---------|-----------------|
| Humano | 85 000 millones |
| Gato | 1 000 millones |
| Ratón | 75 millones |

---

## El Perceptrón

<!-- IMAGEN: diagrama de perceptrón simple (entradas → pesos → suma → activación → salida) -->

Una neurona artificial (perceptrón) modela una neurona biológica:
1. Recibe un conjunto de **entradas**.
2. Asigna un **peso** a cada entrada.
3. Calcula una función sobre las entradas ponderadas.
4. Produce una **salida**.

---

## Perceptrón Multicapa (MLP)

<!-- IMAGEN: red con capa de entrada, capas ocultas y capa de salida -->

Sistema bio-inspirado que apila múltiples perceptrones en capas:
- **Capa de entrada**: recibe las características.
- **Capas ocultas**: aprenden representaciones intermedias.
- **Capa de salida**: produce la predicción.

---

## Ventajas y desventajas de las redes neuronales

### Ventajas
- Sirven tanto para **clasificación** como para **regresión**.
- Modelan patrones más complejos que casi cualquier otro algoritmo.
- **No asumen** nada sobre las relaciones subyacentes del dominio.

### Desventajas
- El entrenamiento es **extremadamente intensivo en cómputo**.
- Es necesario monitorear el **sobreajuste** (*overfitting*).
- Resultan en una **caja negra** difícil de interpretar.

---

## Progreso histórico del Deep Learning

| Año | Hito |
|-----|------|
| 1957 | **Perceptrón** – Frank Rosenblatt |
| 1986 | **Retropropagación** – Rumelhart, Hinton & Williams |
| 1998 | **CNN / LeNet-5** – Yann LeCun et al. |
| 2012 | **AlexNet** – Krizhevsky, Sutskever, Hinton → *boom* de las CNNs |
| 2014 | **GANs** – Ian Goodfellow et al. |
| 2015 | **ResNet** – Kaiming He et al. |
| 2017 | **Transformers** – Vaswani et al. |
| 2022 | **ChatGPT** (GPT-3.5) – OpenAI |
| 2023 | Gemini (multimodal), Mixtral (local/open), SAM, SORA, LLaMA 3, GPT-4o… |

---

## Proyección: Hype Cycle de Gartner

El **Hype Cycle** de Gartner describe las etapas típicas de adopción tecnológica:

1. **Lanzamiento**: un avance potencial pone en marcha las expectativas.
2. **Pico de expectativas sobredimensionadas**: historias de éxito y muchos fracasos.
3. **Abismo de desilusión**: el interés se desvanece ante implementaciones fallidas.
4. **Rampa de consolidación**: casos de uso reales comienzan a materializarse.
5. **Meseta de productividad**: adopción generalizada y criterios de evaluación claros.

<!-- IMAGEN: Hype Cycle for Artificial Intelligence 2025 (Gartner) -->
<!-- IMAGEN: Hype Cycle for Emerging Technologies 2025 (Gartner) -->

> Fuente: [Gartner – Hype Cycle for Artificial Intelligence](https://www.gartner.com/en/articles/hype-cycle-for-artificial-intelligence)

---

# Módulo 4 — Machine y Deep Learning en la Práctica

## Métricas de evaluación

La **matriz de confusión** es la base para calcular las métricas de clasificación:

|  | Predicho: Positivo | Predicho: Negativo |
|--|--------------------|--------------------|
| **Real: Positivo** | Verdadero Positivo (VP / TP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP) | Verdadero Negativo (VN / TN) |

### Definiciones

- **Accuracy**: fracción de predicciones correctas sobre el total.
  ```
  Accuracy = (VP + VN) / (VP + VN + FP + FN)
  ```

- **Precision**: de todos los casos que predije como positivos, ¿cuántos realmente lo eran?  
  *(calidad de la respuesta)*
  ```
  Precision = VP / (VP + FP)
  ```

- **Recall (Tasa de Verdaderos Positivos)**: de todos los casos positivos reales, ¿cuántos detecté?  
  *(cantidad de las respuestas)*
  ```
  Recall = VP / (VP + FN)
  ```

- **F1-Score**: media armónica de Precision y Recall — útil cuando ambas métricas son igual de importantes.
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

---

## Ejemplo: Detección de plomo en casas

| | Predicho: Tiene plomo | Predicho: No tiene plomo |
|--|----------------------|--------------------------|
| **Real: Tiene plomo** | 2 (TP) | 0 (FN) |
| **Real: No tiene plomo** | 1 (FP) | 1 (TN) |

```
Precision = 2/3  = 66.6 %
Recall    = 2/2  = 100 %
Accuracy  = 3/4  = 75 %
```

---

## Ejemplo comparativo: plomo vs. COVID

| Métrica | Plomo (n=12 000) | COVID (n=4 510) |
|---------|-----------------|-----------------|
| TP | 500 | 2 000 |
| FN | 1 000 | 10 |
| FP | 500 | 500 |
| TN | 10 000 | 2 000 |
| **Precision** | 50 % | 80 % |
| **Recall** | 33.3 % | 99.5 % |
| **Accuracy** | 87.5 % | 88.7 % |

> La **accuracy puede ser engañosa** con clases desbalanceadas. Siempre analiza precision y recall por separado.

---

## Práctica de ML — Notebook 1

**Contenido:**
- Carga de la base de datos.
- Eliminación de características por conocimiento experto.
- Análisis exploratorio de datos (EDA).
- División de datos en entrenamiento y testing.
- Método K-Nearest Neighbors (KNN).
- Métricas y métricas gráficas.
- KNN con preprocesamiento de *features*.
- Balance de clases.

---

## Práctica de ML — Notebook 2

**Contenido:**
- Carga de la base de datos.
- División en entrenamiento y testing.
- Aprendizaje supervisado: Clasificación.
- Aprendizaje supervisado: Regresión.
- Importancia de características.

---

## Práctica de DL

**Contenido:**
- Librerías necesarias.
- Carga de la base de datos.
- Red Neuronal Convolucional (CNN 2D).
- Red Neuronal Artificial (ANN).
- Transfer Learning.
- Despliegue de modelos de DL.

---

# Módulo 5 — Inteligencia Artificial Generativa

## ¿Qué significa "IA Generativa"?

La **Inteligencia Artificial Generativa** (IA Generativa) crea **nuevo contenido** —texto, imágenes, música, código— a partir de datos existentes, a diferencia de la IA discriminativa que solo clasifica o predice.

Tipos de contenido que puede generar:
- Texto
- Imágenes
- Música y audio
- Video
- Código

---

## Modelos en tiempo real — Omni

<!-- IMAGEN: captura o diagrama de modelos multimodales omni (audio + video + texto en tiempo real) -->

> Demo interactiva: [AI Studio de Google](https://aistudio.google.com)

---

## Arquitecturas para IA Generativa

### GANs — Redes Generativas Adversariales

<!-- IMAGEN: diagrama generador vs. discriminador -->

Dos redes compiten:
- **Generador**: crea muestras falsas.
- **Discriminador**: intenta distinguir lo real de lo falso.

### Transformers pre-entrenados

<!-- IMAGEN: diagrama de arquitectura Transformer con atención -->

Base de los LLMs modernos (GPT, BERT, Claude, Gemini…).

> Visualizador interactivo: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

### Autoencoders

<!-- IMAGEN: diagrama encoder → espacio latente → decoder -->

Aprenden representaciones comprimidas para luego reconstruir o generar variaciones.

---

## IA para crear imágenes

Herramientas destacadas:

| Herramienta | Descripción |
|-------------|-------------|
| **Midjourney** | Generación artística de alta calidad por prompts. |
| **DALL-E** | Modelo de OpenAI integrado en ChatGPT. |
| **Gemini** | Modelo multimodal de Google. |
| **Leonardo.ai** | Plataforma enfocada en contenido visual para creativos. |

---

## IA para audio y video

| Herramienta | Tipo |
|-------------|------|
| **ElevenLabs** | Síntesis y clonación de voz. |
| **SORA** | Generación de video por texto (OpenAI). |
| **Veo 3** | Generación de video (Google DeepMind). |

---

## Inteligencia Artificial General (AGI)

> *"Creemos que nuestra investigación eventualmente conducirá a la inteligencia artificial general, un sistema que puede resolver problemas a nivel humano. Construir AGI segura y beneficiosa es nuestra misión."*

<!-- IMAGEN: diagrama de la escala IA Estrecha → AGI → ASI -->

### Modelos de lenguaje relevantes (2025–2026)

| Modelo | Organización | Características |
|--------|-------------|-----------------|
| GPT-5.3-codex | OpenAI | Razonamiento y generación cercana a nivel experto humano. |
| Claude 4 (Opus / Sonnet / Haiku) | Anthropic | Destacado en tareas de codificación. |
| Gemini 3.0 Pro / Flash | Google | Multimodal, mejoras significativas de rendimiento. |
| LLaMA 4 | Meta | Código abierto, hasta 405 B de parámetros. |
| gpt-oss (120B / 20B) | OpenAI | Modelos de código abierto lanzados en agosto 2025. |
| DeepSeek-Coder V2 | DeepSeek | Especializado en programación, código abierto. |
| Grok 4 / Grok 4 Fast | xAI | Acceso a información en tiempo real. |
| o3-mini | OpenAI | Enfocado en razonamiento profundo y lógico. |

---

# Módulo 6 — Modelos Grandes del Lenguaje (LLMs)

## ¿Qué es un LLM?

Un **Large Language Model (LLM)** es un modelo de IA entrenado sobre enormes cantidades de texto con el objetivo de **predecir el siguiente token** en una secuencia.

<!-- IMAGEN: diagrama conceptual de un LLM (corpus → preentrenamiento → modelo → inferencia) -->

---

## Los LLMs son el "autocompletar" del celular… pero a escala

<!-- IMAGEN: analogía teclado de celular sugiriendo palabras vs. LLM completando párrafos -->

La diferencia está en la **escala**: miles de millones de parámetros, entrenados sobre prácticamente todo el texto de internet.

---

## ¿Cómo funcionan los LLMs?

<!-- IMAGEN: flujo de tokenización → embeddings → atención → decodificación → token de salida -->

El proceso simplificado:

1. **Tokenización**: el texto se descompone en *tokens* (palabras, subpalabras).
2. **Embeddings**: cada token se convierte en un vector numérico.
3. **Atención (*Attention*)**: el modelo pondera las relaciones entre todos los tokens del contexto.
4. **Decodificación**: se genera el token siguiente con mayor probabilidad.
5. El proceso se repite hasta completar la respuesta.

---

## Benchmarks y razonamiento

<!-- IMAGEN: tabla o gráfica de benchmarks comparativos entre modelos (MMLU, HumanEval, etc.) -->

Los modelos modernos incorporan **cómputo en tiempo de inferencia** (*test-time compute*) para mejorar el razonamiento:
- El modelo "piensa" más antes de responder.
- Permite resolver problemas que requieren múltiples pasos lógicos.

---

## MoE: Mixture of Experts

<!-- IMAGEN: diagrama de MoE con router → expertos → suma ponderada -->

La arquitectura **Mixture of Experts** permite escalar modelos de forma eficiente:
- El modelo contiene múltiples sub-redes ("expertos").
- Un **router** decide qué expertos activar para cada token.
- Solo se activa una fracción del total de parámetros por inferencia → **menor costo computacional**.

> Referencia: [Requisitos de hardware para LLMs](https://llamaimodel.com/requirements/)

---

# Módulo 7 — Herramientas de IA Generativa

## Pro-tips al usar LLMs en proveedores

Las plataformas de LLMs (ChatGPT, Claude, Gemini, etc.) ofrecen funcionalidades más allá del chat:

- 🖼️ **Crear imágenes**: generación visual integrada.
- 🎙️ **Generación de audio en tiempo real**: voz sintética y conversación oral.
- 💻 **Ejecución de código**: intérpretes Python integrados (*Code Interpreter*).
- 📚 **Carga de "bases de conocimiento"**: subir documentos propios como contexto.
- 🌐 **Búsqueda en internet**: acceso a información actualizada.
- 🔒 **Privacidad de la información**: revisar siempre las políticas de datos antes de subir información sensible.

---

## IA en manos de todos

<!-- IMAGEN: collage de interfaces de ChatGPT, Claude, Gemini, Copilot en móvil y escritorio -->

El acceso a modelos de lenguaje potentes se ha democratizado. Hoy cualquier persona puede usar IA de forma gratuita o con bajo costo desde un navegador o una app.

---

## Deep Research: IA para investigación profunda

La función **Deep Research** permite a los modelos realizar búsquedas multi-paso, sintetizar fuentes y elaborar informes extendidos de forma autónoma.

Plataformas con esta funcionalidad:

| Plataforma | Notas |
|------------|-------|
| **ChatGPT** | Disponible en versión Pro. |
| **Gemini** | Integrado con búsqueda de Google. |
| **Perplexity** | Especializado en búsqueda con fuentes. |
| **Grok** | Acceso a Twitter/X en tiempo real. |

### Ejemplo de prompt para Deep Research

```
Prompt: Dame los avances de los últimos 6 meses en IA generativa
```

<!-- IMAGEN: capturas comparativas de respuestas de ChatGPT, Gemini, Perplexity y Grok al mismo prompt -->

---

## Herramientas para investigación académica

| Herramienta | Uso principal |
|-------------|---------------|
| **Elicit** | Búsqueda y síntesis de papers científicos con IA. |
| **Consensus** | Respuestas basadas en evidencia académica. |

---

## Herramientas para productividad y creación

| Herramienta | Uso principal |
|-------------|---------------|
| **Napkin** | Convierte texto en diagramas y visualizaciones. |
| **NotebookLM** | Analiza documentos propios y genera podcasts/resúmenes. |
| **Lovable** | Crea aplicaciones web desde una descripción en lenguaje natural. |
| **LMArena** (LMSYS) | Compara modelos de lenguaje de forma anónima (*chatbot arena*). |

---

# Módulo 8 — Contextualización de Modelos

## ¿Cómo evito que un LLM alucine y sepa de mi negocio?

Los LLMs generales no conocen información privada ni actualizada de tu organización. Existen tres estrategias para solucionarlo:

```
                ┌─────────────┐
                │  Prompting  │
                └──────┬──────┘
                       │ (sin datos nuevos)
          ┌────────────┼────────────┐
          ▼                         ▼
    ┌───────────┐           ┌──────────────┐
    │    RAG    │           │ Fine-Tuning  │
    └───────────┘           └──────────────┘
  (recupera contexto         (adapta los pesos
   en tiempo real)            del modelo)
```

| Técnica | Cuándo usarla |
|---------|---------------|
| **Prompting** | Información breve que cabe en el contexto. |
| **RAG** | Documentos extensos o bases de conocimiento dinámicas. |
| **Fine-Tuning** | Estilo, formato o dominio muy específico que no cambia frecuentemente. |

---

# Módulo 9 — Técnicas de Prompting

## ¿Cómo sacarle provecho a los modelos de lenguaje?

<!-- IMAGEN: diagrama de un prompt bien estructurado vs. uno genérico y sus resultados -->

Un buen prompt no es solo una pregunta — es una **instrucción estructurada** que guía al modelo hacia la respuesta que necesitas.

---

## Anatomía de un buen prompt

**Componentes**: Rol · Objetivo · Audiencia · Contexto · Límites · Salida (opcional)

### Ejemplo: "Explícame la PGU"

❌ **Prompt vago**
```
Explícame la PGU
```

✅ **Prompt estructurado**
```
Rol: Actúa como un experto en comunicación del IPS en Chile,
     especializado en lenguaje ciudadano.

Objetivo: Redacta el borrador de un correo electrónico para
          explicar de forma sencilla por qué se rechazó una
          solicitud de PGU.

Audiencia: Un adulto mayor de 70 años que no está familiarizado
           con términos técnicos.

Contexto: El motivo del rechazo es que el postulante no pertenece
          al 90% más vulnerable de la población de 65 años o más,
          según el Registro Social de Hogares.

Límites: Máximo 150 palabras, tono empático y no técnico.
         Terminar sugiriendo que puede actualizar su RSH si su
         situación ha cambiado.
```

---

## Anatomía de un prompt para modelos razonadores

<!-- IMAGEN: diagrama de prompt para modelos o1/o3/DeepSeek-R1 con énfasis en problema + restricciones -->

Los modelos razonadores (o1, o3, DeepSeek-R1) responden mejor cuando el prompt:
- Define el **problema** con precisión.
- Lista **restricciones** y criterios de éxito.
- Evita sobre-guiar el razonamiento intermedio (el modelo lo construye solo).

---

## Técnica 1: Zero-Shot Prompting

La técnica más común: pedir a la IA que realice una tarea **sin ejemplos previos**.

> Útil para tareas directas: resumir, traducir, redactar.

❌ **Sin estructura**
```
Resume el último cambio a la ley del Subsidio Único Familiar.
```

✅ **Zero-shot bien estructurado**
```
Actúa como un analista legal del IPS en Chile. Resume en 3 puntos
clave (usando viñetas) los cambios más recientes a la Ley 18.020
sobre el Subsidio Familiar. El resumen es para una circular interna
dirigida a los jefes de sucursal.
```

---

## Técnica 2: One-Shot Prompting

Proporcionar **un único ejemplo** para que el modelo entienda el patrón esperado.

> Ideal para clasificación simple o formatos difíciles de describir solo con palabras.

```
Tu tarea es clasificar correos de ciudadanos en una de las
siguientes categorías: 'Consulta de Beneficios', 'Problema de Pago',
'Actualización de Datos'.

Ejemplo:
Correo: 'Quería saber si me corresponde el Bono de Invierno.'
→ Categoría: 'Consulta de Beneficios'.

Ahora, clasifica este:
Correo: 'Hola, mi nombre es Juan Soto y no he recibido el pago del
Aporte Familiar Permanente de este año.'
→ Categoría:
```

---

## Técnica 3: Few-Shot Prompting

Proporcionar **varios ejemplos** para patrones más complejos o con matices sutiles.

> Útil para extracción de datos de texto no estructurado o lógica de clasificación compleja.

```
Extrae el RUT y el Beneficio consultado, con el formato:
'RUT: [número], Beneficio: [nombre]'.

Ejemplo 1:
Texto: 'Hola, soy Ana del 15.111.222-3, quería saber del Aporte
Familiar.'
→ RUT: 15.111.222-3, Beneficio: Aporte Familiar Permanente.

Ejemplo 2:
Texto: 'Llamo por mi abuela, RUT 5.888.999-K, para ver si le toca el
Bono de Invierno.'
→ RUT: 5.888.999-K, Beneficio: Bono de Invierno.

Ejemplo 3:
Texto: 'Mi consulta es por el Subsidio de Cesantía, mi cédula es
18.777.666-1.'
→ RUT: 18.777.666-1, Beneficio: Subsidio de Cesantía.

Ahora extrae los datos de:
Texto: 'Buenas tardes, mi RUT es 12.345.678-9 y mi consulta es sobre
la PGU de mi madre.'
→
```

---

## Técnica 4: Salidas en formato específico

Instruir al modelo sobre **cómo estructurar** su respuesta.

Formatos comunes: tabla, lista, JSON, viñetas, HTML, Markdown.

```
Actúa como un analista de políticas públicas del IPS.
Crea una tabla en formato Markdown que compare los requisitos
principales de la PGU y el Aporte Familiar Permanente.

Columnas:
- Beneficio
- Requisito de Edad
- Requisito de Focalización (RSH)
- Forma de Pago

Asegúrate de que la información esté actualizada a la normativa
vigente en Chile.
```

---

## Técnica 5: Chain-of-Thought (CoT) — Cadenas de Pensamiento

Instruir al modelo para que **desglose el razonamiento paso a paso** antes de concluir.

> Precursor de los modelos razonadores. Mejora drásticamente la precisión en problemas complejos.

```
Actúa como un experto previsional del IPS. Analiza paso a paso si a
la siguiente usuaria le corresponde el Bono por Hijo. Evalúa cada
requisito principal y al final concluye si es elegible o no.

Datos:
- Edad: 66 años
- Nacionalidad: Chilena, con residencia en Chile
- Hijos: 2, ambos nacidos vivos
- Situación previsional: Pensionada a los 65 años, afiliada a AFP
```

### Frases clave para activar CoT

| Frase | Efecto |
|-------|--------|
| `"Piensa paso a paso..."` | Razonamiento secuencial. |
| `"Desglosa tu razonamiento..."` | Transparencia del proceso. |
| `"Analiza evaluando primero [A], luego [B], finalmente [C]."` | Razonamiento estructurado. |
| `"Antes de la respuesta final, explica la lógica."` | Justificación explícita. |
| `"Genera un plan detallado en orden cronológico..."` | Planificación paso a paso. |

---

## Notebooks de práctica

- Ejemplos con OpenAI API
- Ejemplos con Gemini API

---

# Módulo 10 — Retrieval Augmented Generation (RAG)

## ¿Qué es RAG?

**RAG** combina la capacidad generativa de un LLM con la recuperación de información relevante desde una base de conocimiento externa, reduciendo alucinaciones y permitiendo responder con datos actualizados o privados.

<!-- IMAGEN: diagrama general RAG (pregunta → retriever → documentos relevantes → LLM → respuesta) -->

---

## Flujo general de RAG

<!-- IMAGEN: diagrama de flujo RAG con las fases: indexación → retrieval → augmentation → generation -->

```
Usuario hace una pregunta
         │
         ▼
   [RETRIEVAL]
   Buscar documentos relevantes
   en la base de conocimiento
         │
         ▼
   [AUGMENTATION]
   Añadir los documentos al
   contexto del prompt
         │
         ▼
   [GENERATION]
   El LLM genera una respuesta
   fundamentada en los documentos
         │
         ▼
   Respuesta al usuario
```

---

## Fase 1: Indexación

<!-- IMAGEN: diagrama de indexación (documentos → chunking → embeddings → vector store) -->

Pasos:
1. **Carga de documentos**: PDF, Word, páginas web, bases de datos…
2. **División en fragmentos (*chunking*)**: segmentar en trozos de tamaño manejable.
3. **Generación de embeddings**: convertir cada fragmento en un vector numérico.
4. **Almacenamiento en vector store**: guardar los vectores en una base de datos vectorial (FAISS, Chroma, Pinecone, etc.).

---

## Fase 2: Retrieval (Recuperación)

<!-- IMAGEN: diagrama de búsqueda semántica (query → embedding → similitud coseno → top-k fragmentos) -->

Cuando el usuario hace una pregunta:
1. La pregunta se convierte en un **embedding**.
2. Se buscan los fragmentos más **similares semánticamente** en el vector store.
3. Se recuperan los **top-k fragmentos** más relevantes.

---

## Fase 3: Augmentation & Generation

<!-- IMAGEN: diagrama de construcción del prompt aumentado y generación de respuesta -->

Los fragmentos recuperados se **inyectan en el prompt** como contexto adicional:

```
[SYSTEM]
Eres un asistente experto. Usa únicamente la siguiente información
para responder.

[CONTEXTO RECUPERADO]
... fragmentos del vector store ...

[PREGUNTA DEL USUARIO]
...
```

El LLM genera su respuesta **basándose en el contexto provisto**, no solo en su entrenamiento.

---

## Variaciones de RAG

<!-- IMAGEN: diagrama comparativo de variantes (Naive RAG → Advanced RAG → Modular RAG) -->

| Variante | Características |
|----------|----------------|
| **Naive RAG** | Flujo básico: chunk → embed → retrieve → generate. |
| **Advanced RAG** | Reranking, query expansion, recuperación híbrida. |
| **Modular RAG** | Componentes intercambiables, agentes de recuperación. |
| **Self-RAG** | El modelo decide cuándo recuperar y evalúa la relevancia. |
| **Graph RAG** | Recuperación sobre grafos de conocimiento. |

> Notebook de práctica con ejemplos de variaciones de RAG disponible en el repositorio del curso.

---

# Módulo 11 — Fine-Tuning para Ajuste de Modelos

## Tamaño de los modelos de lenguaje

<!-- IMAGEN: comparativa de tamaño (número de parámetros) de los principales LLMs -->

Los LLMs varían enormemente en tamaño:
- Modelos pequeños: ~1–7 B parámetros (ejecutables en CPU o GPU de consumo).
- Modelos medianos: ~13–70 B parámetros (requieren GPU de gama alta).
- Modelos grandes: ~405 B+ parámetros (requieren clústeres de GPUs).

> Referencia de hardware: [llamaimodel.com/requirements](https://llamaimodel.com/requirements/)

---

## Precisión de los parámetros

<!-- IMAGEN: comparativa de precisiones numéricas (FP32 → FP16 → INT8 → INT4) con su impacto en memoria -->

Los parámetros de un modelo se almacenan con distinta precisión numérica:

| Precisión | Bits por parámetro | Memoria para 7B params |
|-----------|--------------------|------------------------|
| FP32 | 32 bits | ~28 GB |
| FP16 / BF16 | 16 bits | ~14 GB |
| INT8 | 8 bits | ~7 GB |
| INT4 | 4 bits | ~3.5 GB |

---

## Quantization — Cuantización de modelos

<!-- IMAGEN: diagrama de cuantización (reducción de precisión de pesos con mínima pérdida de calidad) -->

La **cuantización** reduce la precisión de los parámetros para disminuir el uso de memoria y aumentar la velocidad de inferencia, con una pérdida de calidad aceptable.

- Permite ejecutar modelos grandes en hardware de consumo.
- Es la base de formatos como **GGUF** (llama.cpp) y **GPTQ**.

---

## ¿Qué necesitamos para hacer Fine-Tuning?

<!-- IMAGEN: diagrama de requisitos (modelo base + dataset etiquetado + GPU + framework) -->

Componentes necesarios:

1. **Modelo base pre-entrenado**: el punto de partida (LLaMA, Mistral, Gemma…).
2. **Dataset de ajuste**: pares (instrucción → respuesta) específicos del dominio.
3. **Cómputo**: GPU con suficiente VRAM (o servicios en la nube).
4. **Framework**: Hugging Face Transformers, Unsloth, Axolotl, LLaMA-Factory…

---

## LoRA: Low-Rank Adaptation

<!-- IMAGEN: diagrama LoRA mostrando matrices A y B de bajo rango inyectadas en la capa de atención -->

**LoRA** adapta el modelo **sin modificar sus pesos originales**:
- Se añaden matrices de bajo rango (A, B) a las capas de atención.
- Solo se entrenan esas matrices adicionales (~1–2% de los parámetros totales).
- El modelo base se congela → **mucho menos cómputo y memoria**.

> Visualizador de arquitectura Transformer: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

---

## QLoRA: Quantized LoRA

<!-- IMAGEN: diagrama QLoRA = modelo base en 4 bits + adaptadores LoRA en 16 bits -->

**QLoRA** combina cuantización + LoRA:
1. El modelo base se carga en **4 bits** (usando NF4).
2. Los adaptadores LoRA se entrenan en **16 bits**.
3. Resultado: fine-tuning de modelos de 70B+ en una sola GPU de 24 GB.

---

## Pasos para hacer Fine-Tuning

<!-- IMAGEN: diagrama de flujo del proceso de fine-tuning -->

```
1. Seleccionar modelo base
2. Preparar y formatear el dataset
3. Configurar LoRA / QLoRA
4. Entrenamiento
5. Evaluación en conjunto de validación
6. Fusión de pesos (merge) o uso con adaptadores
7. Despliegue
```

---

## ¿Cómo evaluamos que el modelo está bien?

<!-- IMAGEN: gráfica de curvas de pérdida train vs. validation durante el entrenamiento -->

Métricas de evaluación para modelos ajustados:

| Métrica | Uso |
|---------|-----|
| **Perplexity** | Mide qué tan bien predice el modelo el texto de validación. |
| **ROUGE / BLEU** | Comparación con respuestas de referencia (resumen, traducción). |
| **Evaluación humana** | Juicio cualitativo sobre coherencia, precisión, tono. |
| **LLM-as-judge** | Usar otro LLM para puntuar las respuestas generadas. |

---

## Fine-Tuning vs. RAG: ¿cuándo usar cada uno?

<!-- IMAGEN: diagrama comparativo Fine-Tuning vs. RAG -->

| Criterio | RAG | Fine-Tuning |
|----------|-----|-------------|
| Conocimiento actualizable | ✅ Fácil | ❌ Requiere re-entrenamiento |
| Información privada / extensa | ✅ | ✅ |
| Adaptar estilo o formato | ❌ | ✅ |
| Costo computacional | Bajo (inferencia) | Alto (entrenamiento) |
| Citas y trazabilidad | ✅ Fuentes recuperables | ❌ |
| Latencia en producción | Mayor (búsqueda + LLM) | Menor |

> En muchos casos, **RAG + Fine-Tuning** se combinan para obtener lo mejor de ambos enfoques.

---

# Módulo 12 — Agentes de IA y Automatización Low-Code

## Prompt vs. Workflow vs. Agente

<!-- IMAGEN: diagrama comparativo de las tres capas (prompt simple → workflow encadenado → agente autónomo) -->

| Nivel | Descripción | Ejemplo |
|-------|-------------|---------|
| **Prompt** | Una sola instrucción al LLM, respuesta directa. | "Resume este documento." |
| **Workflow** | Secuencia predefinida de pasos con LLMs y herramientas. | Cadena: resumir → traducir → enviar por email. |
| **Agente** | El LLM decide qué acciones tomar de forma autónoma, en bucle. | Investigar un tema, navegar webs, redactar un informe. |

> **La IA generativa entrega valor real cuando se combina con buenos datos y casos de negocio claros.**

---

## Agentes de IA

<!-- IMAGEN: diagrama de arquitectura de un agente (LLM + memoria + herramientas + planeación + acción) -->

Un **agente de IA** es un sistema donde el LLM actúa como el "cerebro" que:
1. **Percibe** el entorno (contexto, herramientas disponibles).
2. **Planifica** una secuencia de acciones para alcanzar el objetivo.
3. **Actúa** ejecutando herramientas (búsqueda web, código, APIs…).
4. **Observa** el resultado y ajusta el plan.

<!-- IMAGEN: bucle percepción → planificación → acción → observación -->

---

## Frameworks para construir agentes

<!-- IMAGEN: logos/comparativa de frameworks (LangChain, CrewAI, AutoGen, n8n, LangGraph) -->

### CrewAI — Agentes colaborativos

<!-- IMAGEN: diagrama CrewAI con roles (Investigador, Escritor, Revisor) colaborando en una tarea -->

CrewAI permite definir **equipos de agentes** con roles especializados:
- Cada agente tiene un rol, un objetivo y herramientas asignadas.
- Los agentes se coordinan para completar tareas complejas.

---

## Human in the Loop

<!-- IMAGEN: diagrama de ciclo agente con punto de intervención humana antes de ejecutar acción crítica -->

No todos los agentes deben actuar de forma completamente autónoma. El patrón **Human in the Loop** inserta puntos de aprobación humana:

- Antes de acciones irreversibles (enviar email, borrar datos…).
- Cuando la confianza del modelo es baja.
- En decisiones que requieren juicio ético o de negocio.

---

## MCP: Model Context Protocol

<!-- IMAGEN: diagrama de arquitectura MCP (host → cliente MCP → servidor MCP → herramientas/recursos) -->

El **Model Context Protocol (MCP)** es un estándar abierto (Anthropic, 2024) que define cómo los LLMs se conectan a herramientas y fuentes de datos externas de forma uniforme.

### ¿Cómo funciona?

```
┌──────────────┐      MCP       ┌─────────────────────┐
│  LLM / Host  │ ←────────────→ │  Servidor MCP        │
│  (Claude,    │                │  (acceso a archivos, │
│   GPT, etc.) │                │   APIs, BD, código…) │
└──────────────┘                └─────────────────────┘
```

### Beneficios
- Un solo protocolo para conectar cualquier LLM con cualquier herramienta.
- Servidores MCP reutilizables entre distintos clientes.
- Ecosistema creciente de servidores MCP (GitHub, Notion, bases de datos…).

<!-- IMAGEN: ecosistema de servidores MCP disponibles -->

---

## Vibe Coding: prototipado y desarrollo de apps

**Vibe coding** es el flujo de trabajo donde describes en lenguaje natural lo que quieres y un agente de IA genera la aplicación completa.

> Herramienta destacada: **Lovable** — genera aplicaciones web funcionales desde una descripción.

<!-- IMAGEN: captura de pantalla de Lovable generando una app desde un prompt -->

Casos de uso:
- MVPs y prototipos rápidos.
- Automatización de procesos internos sin equipo de desarrollo.
- Dashboards y formularios a medida.

---

## Agentes de código en tu PC

<!-- IMAGEN: diagrama de agentes de código locales (Claude Code, Cursor, Copilot Workspace) orquestando archivos y procesos -->

Los agentes de código permiten **orquestar procesos completos** directamente desde el entorno de desarrollo:
- Leer, escribir y ejecutar archivos.
- Correr tests y corregir errores en bucle.
- Interactuar con APIs y servicios locales.

Herramientas destacadas: **Claude Code**, Cursor Agent, GitHub Copilot Workspace.