# Unidad 2: Visualizacion de Datos

## Presentacion

Los datos sin visualizacion son listas de numeros. La visualizacion es el proceso de convertir datos en representaciones graficas que permiten al cerebro humano detectar patrones, tendencias y anomalias que serian invisibles en una tabla. Un analista que no sabe visualizar datos puede tener el modelo correcto y comunicar el resultado incorrecto; uno que visualiza bien puede comunicar hallazgos complejos en segundos.

Esta unidad cubre el ecosistema completo de visualizacion de datos para ciencia de datos e inteligencia de negocios: desde las librerias de codigo abierto en Python hasta las herramientas de BI empresariales, y desde los principios perceptuales que fundamentan toda buena grafica hasta la decision practica de que tipo de grafica usar en cada situacion.

---

## Objetivos de Aprendizaje

Al finalizar esta unidad el estudiante sera capaz de:

- Construir graficas estaticas de publicacion con **Matplotlib**, controlando cada elemento del grafico mediante la API orientada a objetos.
- Generar visualizaciones estadisticas con **Seaborn**, aprovechando su integracion con pandas y sus calculos automaticos de distribuciones e intervalos de confianza.
- Crear visualizaciones interactivas con **Plotly**, incluyendo graficas financieras, geograficas, animaciones y dashboards con Dash.
- Disenar reportes e indicadores empresariales con **Power BI**, modelar datos con esquema en estrella y escribir medidas en DAX.
- Aplicar los principios de percepcion visual, uso del color y reduccion del ruido para construir graficas que comuniquen con claridad y precision.
- Seleccionar el tipo de grafica correcto segun la naturaleza del dato y la tarea del lector.

---

## Contenido de la Unidad

| Modulo | Descripcion |
|--------|-------------|
| **Matplotlib** | Arquitectura Figure/Axes, graficas de linea, barras, histogramas, scatter, boxplots, heatmaps, subplots con GridSpec, animaciones y exportacion. |
| **Seaborn** | Tidy data, graficas relacionales, distribuciones (histplot, kdeplot, ecdfplot), graficas categoricas, regresion, matrices, FacetGrid y PairGrid. |
| **Plotly** | API Express y Graph Objects, graficas interactivas, financieras (candlestick, Bollinger), geograficas, 3D, animaciones, hover personalizado y Dash. |
| **Power BI** | Motor Vertipaq, Power Query (M), modelado en estrella, DAX (CALCULATE, inteligencia de tiempo, RANKX), visualizaciones, RLS y Power BI Service. |
| **Buenas Practicas** | Jerarquia de percepcion de Cleveland-McGill, Data-Ink Ratio, atributos preatentivos, color, tipografia, escalas, accesibilidad y narrativa con datos. |
| **Que Grafica Escoger** | Arbol de decision por tarea: comparacion, tendencia, distribucion, relacion, composicion, geografico, flujo y alta dimension. |

---

## Relacion entre los Modulos

```
DATOS
  │
  ├── Python
  │     ├── Matplotlib  ──► Control total, publicaciones, figuras estaticas
  │     ├── Seaborn     ──► Estadistica automatica, tidy data, exploracion
  │     └── Plotly      ──► Interactividad, dashboards, finanzas, mapas
  │
  └── BI Empresarial
        └── Power BI    ──► Modelado, DAX, reportes de negocio, Service

TODOS ↑ guiados por:
  ├── Buenas Practicas  ──► Como hacerlo bien (percepcion, color, narrativa)
  └── Que Escoger       ──► Cual usar segun los datos y la tarea
```

La eleccion entre Matplotlib, Seaborn y Plotly no es de exclusion mutua: en la practica se usan juntas. Matplotlib provee el control de bajo nivel; Seaborn lo complementa para analisis estadistico rapido; Plotly entra cuando se necesita interactividad o se trabaja en un entorno web. Power BI opera en un contexto diferente: es la herramienta dominante en equipos de negocio donde los consumidores del reporte no son programadores.

---

## Requisitos Previos

- Python basico: DataFrames de pandas, arrays de NumPy, funciones.
- Estadistica descriptiva: media, mediana, desviacion estandar, percentiles, correlacion.
- Para Power BI: acceso a Power BI Desktop (descarga gratuita en microsoft.com/powerbi).

---

## Herramientas y Configuracion

```bash
# Instalar las librerias de Python necesarias para esta unidad
pip install matplotlib seaborn plotly dash kaleido scipy scikit-learn

# Verificar versiones
python -c "import matplotlib, seaborn, plotly; \
           print(matplotlib.__version__, seaborn.__version__, plotly.__version__)"
```

**Power BI Desktop:** descarga gratuita desde [https://powerbi.microsoft.com/desktop](https://powerbi.microsoft.com/desktop). No requiere cuenta para usar en modo local; si requiere cuenta Microsoft para publicar en Power BI Service.


---

## Referencias Clave

- Tufte, E. R. (1983). *The Visual Display of Quantitative Information*. Graphics Press.
- Knaflic, C. N. (2015). *Storytelling with Data*. Wiley.
- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. Disponible en: https://clauswilke.com/dataviz/
- Cleveland, W. S., & McGill, R. (1984). Graphical perception. *JASA*, 79(387), 531–554.
- Russo, M., & Ferrari, A. (2022). *The Definitive Guide to DAX* (2nd ed.). Microsoft Press.
- Documentacion Matplotlib: https://matplotlib.org/stable/
- Documentacion Seaborn: https://seaborn.pydata.org
- Documentacion Plotly: https://plotly.com/python/
- Documentacion Power BI: https://learn.microsoft.com/power-bi/