# 2.13 Que Grafica Escoger

## Indice

1. [El arbol de decision](#1-el-arbol-de-decision)
2. [Comparacion](#2-comparacion)
3. [Tendencia temporal](#3-tendencia-temporal)
4. [Distribucion](#4-distribucion)
5. [Relacion entre variables](#5-relacion-entre-variables)
6. [Composicion parte-todo](#6-composicion-parte-todo)
7. [Geografico y espacial](#7-geografico-y-espacial)
8. [Flujo y proceso](#8-flujo-y-proceso)
9. [Multiples variables simultaneas](#9-multiples-variables-simultaneas)
10. [Tabla de seleccion rapida](#10-tabla-de-seleccion-rapida)

---

## 1. El Arbol de Decision

```
¿Cual es la tarea del lector?
│
├── COMPARAR valores entre categorias ──────────────────────────── Sección 2
│
├── MOSTRAR TENDENCIA en el tiempo ──────────────────────────────  Sección 3
│
├── MOSTRAR DISTRIBUCION de una variable ────────────────────────  Sección 4
│
├── MOSTRAR RELACION entre dos o mas variables ──────────────────  Sección 5
│
├── MOSTRAR COMPOSICION (parte de un todo) ─────────────────────  Sección 6
│
├── MOSTRAR DATOS GEOGRAFICOS o espaciales ──────────────────────  Sección 7
│
├── MOSTRAR FLUJO o proceso ─────────────────────────────────────  Sección 8
│
└── EXPLORAR MUCHAS VARIABLES simultaneamente ───────────────────  Sección 9
```

---

## 2. Comparacion

**Pregunta clave:** ¿Cuanto vale cada categoria? ¿Cual es la mayor? ¿Cuanto difieren?

### 2.1 Pocas categorias (< 7), un periodo

**Grafica de barras verticales (columnas)**

```
Cuando usarla:
  - Comparar valores de 2 a 6 categorias.
  - Las etiquetas de categoria son cortas.
  - El orden de las barras no importa o es natural (meses, dias).

Ejemplo: ventas por trimestre, nota media por asignatura.

Codigo Python (Matplotlib):
  ax.bar(categorias, valores, color='steelblue', edgecolor='white')

Codigo Python (Seaborn):
  sns.barplot(data=df, x='categoria', y='valor', estimator='sum')
```

**Grafica de barras horizontales**

```
Cuando usarla:
  - Etiquetas de categoria son largas (nombres de paises, productos...).
  - Mas de 5-6 categorias (hasta ~15 con comodidad).
  - Quieres ordenar de mayor a menor sin rotar etiquetas.

Ejemplo: top 10 paises por PIB, satisfaccion por departamento.

Codigo Python:
  ax.barh(categorias, valores)              # Matplotlib
  sns.barplot(data=df, x='valor', y='cat', orient='h')  # Seaborn
```

### 2.2 Muchas categorias (> 7)

**Lollipop chart (grafica de piruleta)**

```
Cuando usarla:
  - Muchas categorias donde las barras quedan muy finas y densas.
  - Se quiere reducir el ruido visual de barras llenas.
  - Equivalente a barras pero mas limpio visualmente.

Codigo Python:
  ax.hlines(y=categorias, xmin=0, xmax=valores, color='gray', linewidth=1)
  ax.plot(valores, categorias, 'o', color='steelblue', ms=8)
```

**Tabla ordenada con barras de datos (sparkbars)**

```
Cuando usarla:
  - Mas de 15 categorias.
  - El lector necesita buscar un elemento especifico por nombre.
  - Se necesitan varios valores por categoria (tabla + mini-grafica).
```

### 2.3 Comparar multiples grupos por categoria

**Barras agrupadas**

```
Cuando usarla:
  - Comparar dos o tres series entre varias categorias.
  - La comparacion DENTRO de cada categoria es tan importante
    como la comparacion ENTRE categorias.
  - No mas de 3-4 grupos o la grafica se satura.

Ejemplo: ventas por trimestre, separadas por region.
```

**Grafica de puntos (dot plot o Cleveland dot plot)**

```
Cuando usarla:
  - Comparar dos valores por categoria (antes y despues, grupos A y B).
  - Alternativa mas limpia a las barras agrupadas para dos grupos.
  - El cambio entre los dos puntos es el mensaje.

Ejemplo: salario medio hombres vs mujeres por sector.

Codigo Python:
  # Dos puntos conectados por una linea
  for cat in categorias:
      ax.plot([val_a[cat], val_b[cat]], [cat, cat], 'gray', lw=1.5)
  ax.plot(list(val_a.values()), categorias, 'o', color='steelblue', ms=8)
  ax.plot(list(val_b.values()), categorias, 's', color='darkorange', ms=8)
```

**Barras apiladas (100%)**

```
Cuando usarla:
  - Comparar la PROPORCION de categorias entre grupos.
  - El total de cada grupo es 100% (o se normaliza a 100%).
  - Solo cuando hay 2-3 sub-categorias por barra; con mas, es confuso.

Ejemplo: porcentaje de clientes por nivel de satisfaccion, por producto.

NO usar cuando:
  El total absoluto tambien importa (usar barras apiladas sin normalizar).
  Hay mas de 4-5 sub-categorias (las barras del medio son imposibles de comparar).
```

---

## 3. Tendencia Temporal

**Pregunta clave:** ¿Como cambia este valor en el tiempo? ¿Cual es la tendencia?

### 3.1 Serie temporal continua

**Grafica de lineas**

```
Cuando usarla:
  - Datos con frecuencia regular (diario, mensual, anual).
  - El cambio entre periodos es continuo y significativo.
  - Comparar la tendencia de 2-5 series simultaneas.

Ejemplo: temperatura diaria, precio de acciones, ventas mensuales.

Codigo Python:
  ax.plot(fechas, valores, lw=2, color='steelblue', label='Serie A')
  # Con intervalo de confianza:
  ax.fill_between(fechas, lower, upper, alpha=0.2, color='steelblue')

Seaborn (con IC automatico si hay multiples observaciones por fecha):
  sns.lineplot(data=df, x='fecha', y='valor', hue='grupo', errorbar='ci')
```

**Grafica de area**

```
Cuando usarla:
  - La magnitud acumulada es tan importante como la tendencia.
  - Una sola serie (el area rellena no funciona bien con multiples series superpuestas).
  - Para mostrar que el valor nunca es negativo y enfatizar el volumen.

Evitar cuando:
  Hay multiples series: las areas se solapan y confunden.
  Usar en su lugar lineas + transparencia o areas apiladas.

Codigo Python:
  ax.fill_between(fechas, 0, valores, alpha=0.4, color='steelblue')
  ax.plot(fechas, valores, lw=2, color='steelblue')
```

**Area apilada**

```
Cuando usarla:
  - Mostrar como contribuye cada componente al total en el tiempo.
  - El total y las partes son igualmente importantes.
  - No mas de 4-5 componentes o se vuelve ilegible.

Evitar cuando:
  Los componentes tienen valores muy volatiles que se cruzan frecuentemente.

Codigo Python:
  ax.stackplot(fechas, serie_a, serie_b, serie_c,
               labels=['A','B','C'], alpha=0.8)
```

### 3.2 Pocos puntos temporales (< 6-8)

**Barras verticales**

```
Cuando usarla:
  - Menos de 6-8 periodos donde la comparacion de valor exacto importa mas
    que la tendencia continua.
  - Los periodos no son frecuentes (años, decadas).

Ejemplo: PIB anual de los ultimos 5 años, resultados por trimestre.
```

**Grafica de pendiente (slope chart)**

```
Cuando usarla:
  - Comparar exactamente DOS periodos entre multiples categorias.
  - El cambio (aumento / disminucion) es el mensaje.

Ejemplo: ranking de paises en 2010 vs 2024.

Codigo Python:
  for i, fila in df.iterrows():
      ax.plot([0, 1], [fila['2010'], fila['2024']], 'o-', color='steelblue')
      ax.text(-0.1, fila['2010'], fila['pais'], ha='right')
      ax.text(1.1, fila['2024'], fila['pais'], ha='left')
```

### 3.3 Deteccion de estacionalidad y ciclos

**Heatmap de calendario**

```
Cuando usarla:
  - Datos diarios o semanales durante uno o mas años.
  - Detectar patrones estacionales (dias de la semana, meses).

Ejemplo: ventas diarias en un año (filas = semanas, columnas = dias).

Codigo Python:
  pivot = df.pivot(index='semana', columns='dia', values='ventas')
  sns.heatmap(pivot, cmap='YlOrRd', annot=False)
```

**Grafica de puntos circular (radial / polar)**

```
Cuando usarla:
  - Datos con ciclo natural de 12 meses o 24 horas.
  - El patron ciclico es el mensaje principal.

Cuando NO usar:
  - Comparaciones de magnitud exacta (el angulo es menos preciso que la longitud).
  - Para la mayoria de datos de negocio: preferir lineas o barras.
```

---

## 4. Distribucion

**Pregunta clave:** ¿Como se distribuyen los valores? ¿Que rango tienen? ¿Hay outliers?

### 4.1 Una variable, todos los datos

**Histograma**

```
Cuando usarla:
  - Explorar la forma de la distribucion (simetrica, sesgada, bimodal).
  - Muchos datos (> 30 observaciones).
  - El rango de la variable es continuo.

Parametros criticos: numero de bins (usar Sturges o Freedman-Diaconis).
  k = ceil(log2(n)) + 1   (Sturges)
  h = 2 * IQR * n^(-1/3) (Freedman-Diaconis, mas robusto)

Seaborn:
  sns.histplot(data=df, x='variable', bins=30, kde=True)
```

**KDE (Kernel Density Estimate)**

```
Cuando usarla:
  - La distribucion suave importa mas que los conteos exactos.
  - Comparar distribuciones de multiples grupos superpuestos.
  - Complemento del histograma.

Seaborn:
  sns.kdeplot(data=df, x='variable', hue='grupo', fill=True, alpha=0.3)
```

**ECDF (Empirical Cumulative Distribution Function)**

```
Cuando usarla:
  - Comparar distribuciones de multiples grupos sin elegir el numero de bins.
  - Leer percentiles directamente del grafico.
  - Verificar si los datos siguen una distribucion teorica.

Seaborn:
  sns.ecdfplot(data=df, x='variable', hue='grupo')
```

### 4.2 Distribucion por grupos (comparar grupos)

**Boxplot**

```
Cuando usarla:
  - Comparar la distribucion de 3 o mas grupos.
  - Detectar outliers.
  - Mostrar el resumen de cinco numeros (min, Q1, mediana, Q3, max).

Limitacion: oculta la forma de la distribucion.
            Una distribucion bimodal y una uniforme pueden tener el mismo boxplot.

Seaborn:
  sns.boxplot(data=df, x='grupo', y='variable', palette='pastel')
  # Añadir puntos individuales sobre el boxplot:
  sns.stripplot(data=df, x='grupo', y='variable', color='black', alpha=0.3, size=3)
```

**Violin plot**

```
Cuando usarla:
  - Quieres ver la forma completa de la distribucion (no solo el resumen).
  - Tienes suficientes datos para que el KDE sea significativo (n > 30 por grupo).
  - Comparar 2-6 grupos.

Seaborn:
  sns.violinplot(data=df, x='grupo', y='variable', inner='quart', palette='husl')
```

**Ridgeline plot (Joy plot)**

```
Cuando usarla:
  - Comparar distribuciones de muchos grupos (7-20) donde los boxplots
    se saturan pero los violines son demasiado estrechos.
  - El orden de los grupos tiene significado (tiempo, geografico, ranking).

Libreria: ridgeplot (pip install ridgeplot) o implementacion manual con Matplotlib.
```

### 4.3 Pocos datos (< 30 observaciones)

**Jitter plot / Stripplot**

```
Cuando usarla:
  - Pocos datos donde un histograma tiene sentido estadistico.
  - Mostrar todos los datos individuales sin agrupar.

Seaborn:
  sns.stripplot(data=df, x='grupo', y='variable', jitter=True, size=6)
```

**Beeswarm plot (Swarmplot)**

```
Cuando usarla:
  - Similar al stripplot pero los puntos no se solapan.
  - Muestra la distribucion real de los datos sin usar bins.
  - Limitado a < 500 puntos por grupo (lento con muchos datos).

Seaborn:
  sns.swarmplot(data=df, x='grupo', y='variable', size=5)
```

---

## 5. Relacion entre Variables

**Pregunta clave:** ¿Estan correlacionadas estas variables? ¿Como se afectan mutuamente?

### 5.1 Dos variables numericas

**Grafica de dispersion (scatter)**

```
Cuando usarla:
  - Explorar la relacion entre dos variables cuantitativas continuas.
  - Detectar correlacion, outliers, clusters o patrones.
  - Con linea de tendencia para mostrar la relacion lineal.

Seaborn:
  sns.scatterplot(data=df, x='var_x', y='var_y', hue='grupo', alpha=0.7)
  sns.regplot(data=df, x='var_x', y='var_y', ci=95)  # con regresion lineal

Plotly (interactivo con hover):
  px.scatter(df, x='var_x', y='var_y', color='grupo', trendline='ols')
```

**Hexbin plot o Histogram 2D**

```
Cuando usarla:
  - Muchos datos (> 5000 puntos) donde el scatter satura con puntos superpuestos.
  - La densidad conjunta es mas importante que los puntos individuales.

Matplotlib:
  ax.hexbin(x, y, gridsize=40, cmap='Blues')

Seaborn:
  sns.histplot(data=df, x='var_x', y='var_y', bins=30, cmap='Blues')
```

**KDE bivariado (contornos)**

```
Cuando usarla:
  - Mostrar la densidad conjunta de dos variables como curvas de nivel.
  - Comparar la region de mayor concentracion entre grupos.

Seaborn:
  sns.kdeplot(data=df, x='var_x', y='var_y', hue='grupo',
              fill=True, alpha=0.3, levels=5)
```

### 5.2 Una categorica, una numerica

Ver seccion 2 (Comparacion) y seccion 4 (Distribucion por grupos).

### 5.3 Dos variables categoricas

**Heatmap de tabla de contingencia**

```
Cuando usarla:
  - Mostrar la frecuencia o valor en cada combinacion de dos categorias.
  - La matriz de correlacion entre variables numericas.

Seaborn:
  tabla = pd.crosstab(df['cat_a'], df['cat_b'])
  sns.heatmap(tabla, annot=True, fmt='d', cmap='Blues')
```

**Grafico de burbujas categorico**

```
Cuando usarla:
  - Tabla de contingencia donde el tamaño de la burbuja = frecuencia.
  - Alternativa visual mas rica al heatmap.
```

### 5.4 Mas de dos variables numericas

**Pairplot (matriz de dispersiones)**

```
Cuando usarla:
  - Explorar las relaciones entre todas las pares de variables de un dataset.
  - Dataset con 3-7 variables numericas.
  - Fase exploratoria del analisis, no para presentacion final.

Seaborn:
  sns.pairplot(df, hue='grupo', vars=['var1','var2','var3'],
               diag_kind='kde', plot_kws={'alpha':0.5})
```

**Grafica de coordenadas paralelas**

```
Cuando usarla:
  - Comparar perfiles de multiples grupos en varias dimensiones simultaneas.
  - Detectar patrones en datasets con muchas variables.
  - Datasets de clasificacion donde se quiere ver que distingue cada clase.

Plotly:
  px.parallel_coordinates(df, color='clase',
                           dimensions=['var1','var2','var3','var4'])
```

**Grafica de radar (spider chart)**

```
Cuando usarla:
  - Comparar pocos perfiles (2-4) en 4-8 dimensiones.
  - Las dimensiones son comparables en escala (o estan normalizadas).
  - Tipico en comparativas de rendimiento, evaluaciones 360.

Limitacion: las areas dependen del orden de los ejes, que es arbitrario.
            Puede ser engañosa; preferir coordenadas paralelas o heatmap.
```

---

## 6. Composicion Parte-Todo

**Pregunta clave:** ¿Como se divide el total? ¿Que proporcion representa cada parte?

### 6.1 Un punto en el tiempo

**Grafica de barras apiladas al 100%**

```
Cuando usarla:
  - Mostrar la proporcion de 2-5 categorias dentro de un total.
  - Comparar proporciones entre varios grupos.
  - Cuando el total absoluto no importa, solo la proporcion.

Ventaja sobre torta: permite comparar la misma categoria entre multiples grupos.
```

**Grafica de torta**

```
Cuando usarla (criterios estrictos, cumplir TODOS):
  - Solo 2-5 categorias.
  - Las diferencias entre categorias son visualmente claras (> 5% de diferencia).
  - El mensaje es "X es claramente la mayoria".
  - La audiencia esta familiarizada con graficas de torta.

NO usar cuando:
  - Hay mas de 5-6 categorias.
  - Las proporciones son similares (dificil comparar angulos parecidos).
  - Se necesita precision: usar barras.
  - Se comparan multiples grupos (usar barras apiladas 100%).
```

**Grafica de anillo (donut)**

```
Cuando usarla:
  - Igual que la torta, pero el espacio central permite añadir un valor destacado.
  - Ligeramente mas moderna visualmente que la torta.
  - Mismas limitaciones que la torta.
```

### 6.2 Jerarquia de proporciones

**Treemap**

```
Cuando usarla:
  - La composicion tiene estructura jerarquica (categoria > subcategoria).
  - Hay muchas categorias (15-50) donde la torta es imposible.
  - El area proporcional al valor es el mensaje.

Ejemplo: composicion del presupuesto por departamento > proyecto.

Plotly:
  px.treemap(df, path=['departamento', 'proyecto'], values='presupuesto',
             color='crecimiento', color_continuous_scale='RdYlGn')
```

**Sunburst (diagrama solar)**

```
Cuando usarla:
  - Igual que el treemap pero en formato circular.
  - Dos o tres niveles de jerarquia.
  - La forma circular facilita ver la relacion parte-todo.

Plotly:
  px.sunburst(df, path=['nivel1','nivel2','nivel3'], values='valor')
```

**Grafica de Waffle**

```
Cuando usarla:
  - Mostrar proporciones de forma intuitiva para audiencia general.
  - Cada cuadrado = 1% (cuadricula de 10x10).
  - Muy efectiva para comunicar "1 de cada X".

Libreria Python: pywaffle
  from pywaffle import Waffle
  fig = plt.figure(FigureClass=Waffle, rows=10, values=[35, 40, 25])
```

### 6.3 Composicion en el tiempo

**Grafica de area apilada**

```
Cuando usarla:
  - Mostrar como cambia la composicion del total en el tiempo.
  - 2-5 componentes.
  - El crecimiento del total es tan importante como la composicion.

NO usar cuando:
  - Los componentes tienen valores similares que se cruzan constantemente.
```

**Grafica de barras apiladas 100% en el tiempo**

```
Cuando usarla:
  - La proporcion de cada componente en el tiempo es el mensaje.
  - El total absoluto no importa.

Ejemplo: cuota de mercado de plataformas a lo largo de los años.
```

**Streamgraph**

```
Cuando usarla:
  - Alternativa estetica al area apilada para presentaciones.
  - El flujo y cambio de proporciones es el mensaje.
  - No se necesita precision de lectura.
```

---

## 7. Geografico y Espacial

**Pregunta clave:** ¿Donde ocurre? ¿Como varia el fenomeno en el espacio?

### 7.1 Valores por region administrativa

**Mapa coropletico (Choropleth)**

```
Cuando usarla:
  - Mostrar un valor cuantitativo por region administrativa
    (paises, estados, municipios).
  - La variacion geografica es el mensaje.

Limitacion critica: las regiones grandes dominan visualmente aunque
tengan pocos datos. Un pais grande con valor bajo llama mas la atencion
que una ciudad pequeña con valor muy alto.

Plotly:
  px.choropleth(df, locations='iso_alpha', color='valor',
                color_continuous_scale='Blues')

Cuando NO usar:
  - Cuando el area de la region no es proporcional al fenomeno
    (usar mapa de burbujas en su lugar).
```

**Mapa de burbujas o puntos**

```
Cuando usarla:
  - El fenomeno ocurre en locaciones especificas, no en regiones.
  - Se quiere mostrar magnitud (tamaño de la burbuja) en el mapa.
  - La ubicacion es mas granular que una region administrativa.

Plotly:
  px.scatter_mapbox(df, lat='lat', lon='lon', size='valor',
                    color='categoria', mapbox_style='carto-positron')
```

**Mapa de calor geografico (heatmap espacial)**

```
Cuando usarla:
  - Muchos puntos individuales que se agrupan en zonas densas.
  - Se quiere ver la densidad de ocurrencias en el espacio.
  - Incidentes de crimen, accidentes de trafico, clicks en una pagina web.
```

### 7.2 Conexiones y flujos

**Mapa de flujo (flow map)**

```
Cuando usarla:
  - Mostrar movimiento entre origenes y destinos (migraciones, envios, trafico).
  - El grosor de la linea = magnitud del flujo.

Ejemplo: flujos de migracion entre paises, rutas de envio.
```

---

## 8. Flujo y Proceso

**Pregunta clave:** ¿Como se mueve o transforma algo a traves de etapas?

### 8.1 Flujo entre categorias

**Diagrama de Sankey**

```
Cuando usarla:
  - Mostrar como fluye una cantidad a traves de varias etapas o categorias.
  - El ancho de cada banda es proporcional a la cantidad que fluye.
  - Tipico en: presupuesto (ingresos > gastos), flujo de usuarios (paginas),
    energia (fuentes > usos), comercio (exportaciones > destinos).

Plotly:
  fig = go.Figure(go.Sankey(
      node=dict(label=['A','B','C']),
      link=dict(source=[0,0,1], target=[1,2,2], value=[10,5,8])
  ))
```

**Grafica de embudo (Funnel)**

```
Cuando usarla:
  - Mostrar la reduccion progresiva a traves de etapas ordenadas.
  - Tipico en: conversion de ventas, captacion de usuarios,
    proceso de solicitud (muchos entran, pocos llegan al final).

Plotly:
  px.funnel(df, x='cantidad', y='etapa', title='Embudo de conversion')
```

### 8.2 Cambio entre dos estados

**Grafica de mariposa (Butterfly chart)**

```
Cuando usarla:
  - Comparar dos grupos simetricos en las mismas categorias.
  - Tipico en: piramide de poblacion (hombres vs mujeres por edad),
    comparacion de dos periodos o dos regiones.

Codigo Python:
  ax.barh(edades, hombres, color='steelblue', label='Hombres')
  ax.barh(edades, [-m for m in mujeres], color='coral', label='Mujeres')
  ax.set_xticks(ax.get_xticks())
  ax.set_xticklabels([str(abs(int(x))) for x in ax.get_xticks()])
```

---

## 9. Multiples Variables Simultaneas

**Pregunta clave:** ¿Que patrones o grupos emergen cuando considero muchas variables a la vez?

### 9.1 Datasets multivariados

**Heatmap de correlacion**

```
Cuando usarla:
  - Explorar las correlaciones entre todas las pares de variables numericas.
  - Dataset con 5-20 variables.
  - Identificar variables altamente correlacionadas o independientes.

Seaborn:
  corr = df.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
              cmap='RdBu_r', vmin=-1, vmax=1)
```

**Pairplot**

```
Cuando usarla:
  - Explorar relaciones entre 3-7 variables con distribucion en la diagonal.
  - Colorear por variable categorica para ver separacion entre grupos.

Seaborn:
  sns.pairplot(df, hue='clase', diag_kind='kde', corner=True)
```

**Coordenadas paralelas**

```
Cuando usarla:
  - 5-20 variables numericas o mixtas.
  - Identificar perfiles de grupos (cada linea = una observacion).
  - Comparar clusters o clases.

Plotly:
  px.parallel_coordinates(df, color='clase',
                           color_continuous_scale='Viridis')
```

**t-SNE o UMAP (proyeccion de alta dimension a 2D)**

```
Cuando usarla:
  - Datasets de alta dimension (> 10 variables, cientos o miles de observaciones).
  - Visualizar clusters en el espacio original de alta dimension.
  - El resultado es un scatter 2D donde la proximidad indica similitud.

NO interpretar como: los ejes tienen significado; las distancias entre
clusters son arbitrarias en t-SNE. Solo la proximidad local importa.

Python:
  from sklearn.manifold import TSNE
  X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
  plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap='tab10')
```

---

## 10. Tabla de Seleccion Rapida

| Tarea | Tipo de dato | Grafica recomendada | Alternativa |
|-------|-------------|---------------------|-------------|
| Comparar 2-6 categorias | Categorica + numerica | Barras verticales | Barras horizontales |
| Comparar > 6 categorias | Categorica + numerica | Barras horizontales ordenadas | Lollipop chart |
| Comparar 2 grupos por categoria | 2 categoricas + numerica | Dot plot (antes/despues) | Barras agrupadas |
| Comparar 3-4 grupos por categoria | 2 categoricas + numerica | Barras agrupadas | Grafica de lineas |
| Tendencia continua en el tiempo | Temporal + numerica | Grafica de lineas | Area chart |
| Pocos periodos temporales | Temporal discreta + numerica | Barras verticales | Puntos + linea |
| Dos periodos de comparacion | Temporal + categorica + numerica | Slope chart | Barras agrupadas |
| Estacionalidad diaria/semanal | Temporal + numerica | Heatmap calendario | Lineas multiples |
| Distribucion de una variable | Numerica continua | Histograma + KDE | Violin |
| Distribucion de 2-5 grupos | Numerica + categorica | Violin o Boxplot | Ridgeline |
| Distribucion todos los puntos | Numerica + categorica (<500) | Swarmplot | Beeswarm + boxplot |
| Relacion dos variables continuas | 2 numericas | Scatter | Hexbin (muchos datos) |
| Relacion con muchos datos (>5000) | 2 numericas | Hexbin / Heatmap 2D | KDE 2D |
| Relacion 2 categoricas | 2 categoricas | Heatmap tabla | Grafica de burbujas |
| Todas las relaciones del dataset | Multiples numericas | Pairplot | Heatmap correlacion |
| Composicion simple (< 5 partes) | Categorica + proporcion | Torta / Anillo | Barras apiladas 100% |
| Composicion multiple comparada | Categorica + proporcion | Barras apiladas 100% | - |
| Composicion jerarquica | Jerarquia + numerica | Treemap | Sunburst |
| Composicion en el tiempo | Temporal + categorica + numerica | Area apilada | Barras apiladas |
| Flujo entre etapas | Etapas + flujo cuantitativo | Sankey | Embudo |
| Conversion por etapas | Etapas + reduccion | Embudo | Barras decrecientes |
| Distribucion geografica | Coordenadas + numerica | Mapa de burbujas | Choropleth |
| Valor por region administrativa | Region + numerica | Choropleth | Cartograma |
| Comparacion dos grupos simetricos | 2 grupos + categorica ordenada | Mariposa | Barras agrupadas |
| Perfiles multivariados | Multiples numericas | Coordenadas paralelas | Radar |
| Alta dimension reducida | Alta dimension | t-SNE / UMAP (scatter) | PCA (scatter) |

---

## Reglas de Oro

```
1. Si dudas entre barras y torta: elige barras.
   La longitud es mas precisa que el angulo.

2. Si dudas entre barras y lineas:
   Datos categoricos (sin orden) -> barras.
   Datos con orden natural (tiempo, escala) -> lineas.

3. Si la grafica necesita mas de 30 segundos de explicacion: cambiarla.

4. Si el lector necesita mirar la leyenda mas de una vez: etiqueta directamente.

5. Si todos los colores son distintos pero no hay razon para ello: usar un solo color.

6. Si hay mas de 7 categorias en una torta: usar barras horizontales.

7. Si el eje Y de las barras no empieza en cero: la grafica engaña.

8. Si se usan dos graficas para mostrar lo mismo: eliminar una.

9. Si una tabla muestra los datos igualmente bien que la grafica: usar la tabla.

10. La mejor grafica es la que comunica el mensaje con la menor cantidad
    de elementos visuales posibles.
```

---

## Referencias

- Schwabish, J. (2021). *Better Data Visualizations: A Guide for Scholars, Researchers, and Wonks*. Columbia University Press.
- Cairo, A. (2019). *How Charts Lie: Getting Smarter about Visual Information*. W.W. Norton.
- Knaflic, C. N. (2015). *Storytelling with Data*. Wiley.
- Tufte, E. R. (1983). *The Visual Display of Quantitative Information*. Graphics Press.
- Few, S. (2009). *Now You See It: Simple Visualization Techniques for Quantitative Analysis*. Analytics Press.
- Wilke, C. O. (2019). *Fundamentals of Data Visualization*. O'Reilly. (Disponible gratis en: https://clauswilke.com/dataviz/)
- Healy, K. (2018). *Data Visualization: A Practical Introduction*. Princeton University Press. (Disponible gratis en: https://socviz.co)
- Cleveland, W. S., & McGill, R. (1984). Graphical perception. *JASA*, 79(387), 531-554.
- From Data to Viz: https://www.data-to-viz.com
- The Data Visualisation Catalogue: https://datavizcatalogue.com