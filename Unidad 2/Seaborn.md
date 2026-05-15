# 2.4 Visualizacion con Seaborn

## Indice

1. [Arquitectura y relacion con Matplotlib](#1-arquitectura-y-relacion-con-matplotlib)
2. [Datasets y estructura de datos](#2-datasets-y-estructura-de-datos)
3. [Graficas relacionales](#3-graficas-relacionales)
4. [Graficas de distribuciones](#4-graficas-de-distribuciones)
5. [Graficas categoricas](#5-graficas-categoricas)
6. [Graficas de regresion](#6-graficas-de-regresion)
7. [Matrices y heatmaps](#7-matrices-y-heatmaps)
8. [Figure-level vs Axes-level](#8-figure-level-vs-axes-level)
9. [Temas y paletas de color](#9-temas-y-paletas-de-color)
10. [FacetGrid y PairGrid](#10-facetgrid-y-pairgrid)
11. [Combinacion con Matplotlib](#11-combinacion-con-matplotlib)
12. [Referencias](#12-referencias)

---

## 1. Arquitectura y Relacion con Matplotlib

Seaborn es una libreria de visualizacion estadistica construida **sobre** Matplotlib. Añade:

- Una API de alto nivel orientada a DataFrames de pandas.
- Estadisticas automaticas (intervalos de confianza, regresion, KDE).
- Temas y paletas de color cuidadosamente diseñadas.
- Graficas estadisticas compuestas en una sola llamada.

### 1.1 La relacion de capas

```
Datos (DataFrame pandas)
        ↓
    Seaborn                   ← API de alto nivel, estadisticas automaticas
        ↓
    Matplotlib                ← Motor de renderizado
        ↓
    Backend (Agg, TkAgg...)   ← Pantalla o archivo
```

Todo objeto que Seaborn crea es un objeto Matplotlib. Puedes acceder al `Figure` y `Axes` subyacentes para personalizacion fina.

### 1.2 Instalacion y configuracion

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Configuracion del tema global
# style : 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
# context: 'paper', 'notebook', 'talk', 'poster' (escala el tamano de texto y lineas)
# palette: paleta de colores por defecto
sns.set_theme(
    style='whitegrid',
    context='notebook',
    palette='deep',
    font='sans-serif',
    font_scale=1.1,
    rc={'axes.spines.top': False, 'axes.spines.right': False}
)

print(f'Seaborn version: {sns.__version__}')
```

### 1.3 El concepto de "tidy data"

Seaborn esta optimizado para datos en formato **tidy** (Wickham, 2014):

- Cada **variable** es una columna.
- Cada **observacion** es una fila.
- Cada **tabla** contiene un tipo de entidad.

```python
# Formato WIDE (no optimo para Seaborn)
df_wide = pd.DataFrame({
    'mes'    : ['Ene', 'Feb', 'Mar'],
    'norte'  : [120, 135, 142],
    'sur'    : [98,  105, 112],
    'este'   : [145, 152, 160],
})

# Formato LONG / TIDY (optimo para Seaborn)
# pd.melt convierte de wide a long
df_long = df_wide.melt(
    id_vars='mes',         # columna que identifica la observacion
    var_name='region',     # nombre de la nueva columna de categorias
    value_name='ventas',   # nombre de la nueva columna de valores
)
# Resultado:
#    mes region  ventas
#    Ene  norte     120
#    Ene    sur      98
#    ...

print(df_long.head(6))
```

### 1.4 Parametros comunes a todas las funciones

| Parametro | Descripcion |
|-----------|-------------|
| `data` | DataFrame de pandas |
| `x`, `y` | Columnas para los ejes |
| `hue` | Variable para codificar color (separar grupos) |
| `size` | Variable para codificar tamaño |
| `style` | Variable para codificar marcador/linea |
| `col`, `row` | Variables para crear facetas (subplots) |
| `palette` | Paleta de colores |
| `ax` | Axes de Matplotlib donde dibujar |
| `estimator` | Funcion de agregacion (mean, median, etc.) |
| `errorbar` | Tipo de barra de error ('ci', 'pi', 'se', 'sd', None) |

---



<img src="../_static/images/seaborn1.png" width="600" />

## 2. Datasets y Estructura de Datos

```python
# Seaborn incluye datasets de ejemplo para practicar
print(sns.get_dataset_names())

# Cargar datasets clasicos
tips     = sns.load_dataset('tips')        # propinas en restaurante
iris     = sns.load_dataset('iris')        # medidas de flores
penguins = sns.load_dataset('penguins')    # medidas de pinguinos
titanic  = sns.load_dataset('titanic')     # supervivencia en el Titanic
flights  = sns.load_dataset('flights')     # pasajeros aereos 1949-1960
fmri     = sns.load_dataset('fmri')        # señales cerebrales fMRI
mpg      = sns.load_dataset('mpg')         # consumo de autos

print(tips.head())
print(tips.describe())
print(tips.dtypes)
```

---

## 3. Graficas Relacionales

Las graficas relacionales exploran la **relacion entre dos variables numericas**, opcionalmente segmentadas por variables categoricas.

### 3.1 `sns.scatterplot()` — Dispersion

```python
tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins').dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Scatter basico con hue y size ─────────────────────
# hue  : codifica el DIA en color (paleta cualitativa automatica)
# size : codifica el TAMAÑO de la cuenta en el tamaño del punto
# style: codifica el SEXO en el marcador
sns.scatterplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='day',         # variable categorica -> color
    size='size',       # variable numerica -> tamaño del punto
    style='sex',       # variable categorica -> marcador
    palette='deep',
    sizes=(30, 200),   # rango de tamaños [min, max] en puntos
    alpha=0.8,
    ax=axes[0],
)
axes[0].set_title('Propinas vs. Cuenta total')
axes[0].set_xlabel('Cuenta total ($)')
axes[0].set_ylabel('Propina ($)')

# ── Scatter con regresion y CI ─────────────────────────
# Seaborn puede superponer una linea de regresion
# pero eso es funcion de regresplot/lmplot (ver seccion 6)
sns.scatterplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species',
    style='island',
    palette='Set2',
    s=80,
    alpha=0.9,
    ax=axes[1],
)
axes[1].set_title('Pinguinos: largo vs. profundidad del pico')
axes[1].set_xlabel('Largo del pico (mm)')
axes[1].set_ylabel('Profundidad del pico (mm)')

plt.tight_layout()
plt.show()
```

### 3.2 `sns.lineplot()` — Lineas con intervalos de confianza

La funcion clave: `lineplot` calcula automaticamente la **media** y el **intervalo de confianza al 95%** cuando hay multiples observaciones por punto en X.

**Intervalo de confianza bootstrap al 95%:**

$$IC_{95\%} = \left[\bar{x} - 1.96 \frac{s}{\sqrt{n}},\; \bar{x} + 1.96 \frac{s}{\sqrt{n}}\right]$$

donde $\bar{x}$ es la media, $s$ la desviacion estandar y $n$ el numero de observaciones.

```python
fmri = sns.load_dataset('fmri')
flights = sns.load_dataset('flights')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Lineplot con IC automatico ─────────────────────────
# fmri tiene multiples sujetos por timepoint.
# Seaborn calcula automaticamente mean ± CI por cada valor de 'timepoint'.
# El area sombreada es el IC 95% por bootstrap.
sns.lineplot(
    data=fmri,
    x='timepoint',
    y='signal',
    hue='region',      # separar por region cerebral
    style='event',     # estilo de linea por evento
    errorbar='ci',     # 'ci'=intervalo confianza, 'sd'=desv.std, 'se'=error estandar, None=sin barra
    palette='muted',
    linewidth=2.5,
    ax=axes[0],
)
axes[0].set_title('Señal fMRI: media ± IC 95% por region')
axes[0].set_xlabel('Timepoint'); axes[0].set_ylabel('Señal BOLD')

# ── Lineas multiples por hue ──────────────────────────
# Pivotar flights para grafico de lineas por mes
flights_pivot = flights.pivot(index='year', columns='month', values='passengers')
flights_norm  = flights_pivot.div(flights_pivot.mean())  # normalizar por la media del mes

for mes in ['January', 'April', 'July', 'October']:
    axes[1].plot(flights_pivot.index, flights_pivot[mes],
                 marker='o', ms=4, lw=1.5, label=mes)

axes[1].set_title('Pasajeros aereos por mes (1949-1960)')
axes[1].set_xlabel('Año'); axes[1].set_ylabel('Pasajeros')
axes[1].legend(title='Mes', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.3 `sns.relplot()` — Funcion a nivel de figura con facetas

`relplot` es la funcion **figure-level** para graficas relacionales. Permite crear grillas de subplots con `col` y `row`.

```python
# Crear facetas: un subplot por cada momento del dia
g = sns.relplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='sex',
    col='time',          # una columna (subplot) por 'time' (Lunch, Dinner)
    row='smoker',        # una fila por 'smoker' (Yes, No)
    kind='scatter',      # 'scatter' o 'line'
    palette='husl',
    height=4,            # altura de cada subplot en pulgadas
    aspect=1.2,          # relacion ancho/alto de cada subplot
    s=60,
    alpha=0.7,
)

# g es un FacetGrid; se puede personalizar con g.set_axis_labels(), g.set_titles(), etc.
g.set_axis_labels('Cuenta total ($)', 'Propina ($)')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.figure.suptitle('Propinas por momento del dia y fumador', y=1.02, fontsize=13)
plt.show()
```

---

## 4. Graficas de Distribuciones

Seaborn ofrece herramientas muy completas para visualizar distribuciones univariadas y bivariadas.

### 4.1 `sns.histplot()` — Histograma estadistico

```python
penguins = sns.load_dataset('penguins').dropna()
tips = sns.load_dataset('tips')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Histograma con KDE superpuesto ────────────────────
# stat='density': normalizar para que el area = 1 (estima la PDF)
# kde=True: superponer la estimacion KDE
sns.histplot(
    data=penguins,
    x='flipper_length_mm',
    hue='species',         # histograma separado por especie
    kde=True,              # superponer curva KDE
    stat='density',        # normalizacion: 'count', 'frequency', 'density', 'probability'
    common_norm=False,     # cada grupo normalizado independientemente
    alpha=0.4,
    palette='Set2',
    binwidth=3,            # ancho de bin en las unidades de x
    ax=axes[0, 0],
)
axes[0, 0].set_title('Distribucion de longitud de aleta por especie')
axes[0, 0].set_xlabel('Longitud de aleta (mm)')

# ── Histograma 2D (bivariado) ─────────────────────────
sns.histplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    bins=20,
    pthresh=0.05,          # ocultar celdas con densidad < 5% del maximo
    cmap='Blues',
    cbar=True,             # mostrar barra de color
    ax=axes[0, 1],
)
axes[0, 1].set_title('Histograma 2D: largo vs. profundidad de pico')

# ── Histograma con multiples estadisticas ─────────────
sns.histplot(
    data=tips,
    x='total_bill',
    stat='count',
    bins=20,
    color='steelblue',
    edgecolor='white',
    linewidth=0.8,
    ax=axes[1, 0],
)
# Superponer lineas de media y mediana
media   = tips['total_bill'].mean()
mediana = tips['total_bill'].median()
axes[1, 0].axvline(media,   color='red',   lw=2, linestyle='--', label=f'Media = {media:.1f}')
axes[1, 0].axvline(mediana, color='green', lw=2, linestyle=':',  label=f'Mediana = {mediana:.1f}')
axes[1, 0].legend()
axes[1, 0].set_title('Cuenta total con media y mediana')
axes[1, 0].set_xlabel('Cuenta total ($)')

# ── Histograma acumulado (ECDF via histplot) ──────────
sns.histplot(
    data=tips,
    x='tip',
    cumulative=True,       # histograma acumulado
    stat='density',
    fill=False,
    element='step',        # tipo: 'bars', 'step', 'poly'
    linewidth=2,
    color='darkorange',
    ax=axes[1, 1],
)
axes[1, 1].set_title('Distribucion acumulada de propinas')
axes[1, 1].set_xlabel('Propina ($)')

plt.tight_layout()
plt.show()
```

### 4.2 `sns.kdeplot()` — Estimacion de densidad por kernel

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── KDE univariado ─────────────────────────────────────
# fill=True: area bajo la curva rellena
# bw_adjust: multiplicador del ancho de banda optimo de Silverman
#   bw_h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)   (regla de Silverman)
# bw_adjust=1 usa el ancho optimo; < 1 mas detalle; > 1 mas suavizado
sns.kdeplot(
    data=penguins,
    x='flipper_length_mm',
    hue='species',
    fill=True,
    alpha=0.3,
    bw_adjust=0.8,
    linewidth=2,
    palette='Set2',
    common_norm=False,
    ax=axes[0],
)
axes[0].set_title('KDE univariado por especie')
axes[0].set_xlabel('Longitud de aleta (mm)')

# ── KDE bivariado ──────────────────────────────────────
# En 2D, el KDE estima la densidad conjunta f(x, y)
# Los contornos son curvas de nivel de la funcion de densidad
sns.kdeplot(
    data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species',
    fill=True,
    alpha=0.3,
    levels=5,          # numero de niveles de contorno
    bw_adjust=1.2,
    palette='husl',
    ax=axes[1],
)
axes[1].set_title('KDE bivariado por especie')

# ── KDE con multiple anchura de banda ─────────────────
np.random.seed(42)
datos_demo = np.concatenate([np.random.normal(3, 1, 300), np.random.normal(8, 1.5, 200)])
df_demo = pd.DataFrame({'valor': datos_demo})

for bw, color, lbl in [(0.3,'red','bw=0.3'), (1.0,'green','bw=1.0'), (2.0,'blue','bw=2.0')]:
    sns.kdeplot(data=df_demo, x='valor', bw_adjust=bw,
                color=color, lw=2, label=lbl, ax=axes[2])
axes[2].set_title('Efecto del ancho de banda en KDE')
axes[2].legend()
axes[2].set_xlabel('Valor')

plt.tight_layout()
plt.show()
```

### 4.3 `sns.ecdfplot()` — Funcion de distribucion empirica acumulada

La ECDF (Empirical Cumulative Distribution Function) estima $F(x) = P(X \leq x)$ sin asumir ninguna forma parametrica:

$$\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[X_i \leq x]$$

```python
fig, ax = plt.subplots(figsize=(9, 5))

sns.ecdfplot(
    data=penguins,
    x='body_mass_g',
    hue='species',
    palette='Set2',
    linewidth=2.5,
    ax=ax,
)

ax.set_xlabel('Masa corporal (g)')
ax.set_ylabel('Probabilidad acumulada F(x)')
ax.set_title('ECDF: distribucion acumulada de masa corporal por especie')

# Anotar el percentil 50 (mediana) de cada especie
for especie, color in zip(['Adelie', 'Chinstrap', 'Gentoo'], sns.color_palette('Set2')):
    datos_esp = penguins[penguins['species'] == especie]['body_mass_g'].dropna()
    mediana   = datos_esp.median()
    ax.axvline(mediana, color=color, linestyle='--', alpha=0.6, linewidth=1.2)
    ax.text(mediana + 30, 0.08, f'{especie}\n{mediana:.0f}g',
            color=color, fontsize=8, va='bottom')

plt.tight_layout()
plt.show()
```

### 4.4 `sns.displot()` — Funcion a nivel de figura para distribuciones

```python
# displot es la version figure-level que envuelve histplot, kdeplot y ecdfplot

# Histograma con facetas por sexo
g = sns.displot(
    data=penguins,
    x='flipper_length_mm',
    hue='species',
    col='sex',             # una columna por sexo
    kind='hist',           # 'hist', 'kde', 'ecdf'
    kde=True,
    stat='density',
    common_norm=False,
    palette='Set2',
    height=4,
    aspect=1.1,
)
g.set_axis_labels('Longitud de aleta (mm)', 'Densidad')
g.set_titles(col_template='Sexo: {col_name}')
g.figure.suptitle('Distribucion de aletas por especie y sexo', y=1.02)
plt.show()
```

---

## 5. Graficas Categoricas

Las graficas categoricas comparan distribuciones o valores estadisticos entre grupos definidos por variables categoricas.

### 5.1 `sns.boxplot()` y `sns.violinplot()`

```python
tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins').dropna()

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

# ── Boxplot ────────────────────────────────────────────
# Recordatorio de la matematica:
# Caja: Q1 a Q3 (IQR = Q3 - Q1)
# Linea central: mediana Q2
# Bigotes: hasta 1.5 * IQR desde Q1 y Q3
# Puntos: outliers fuera de los bigotes
sns.boxplot(
    data=tips,
    x='day',
    y='tip',
    hue='sex',
    palette='pastel',
    width=0.6,
    linewidth=1.5,
    fliersize=4,       # tamaño de los puntos outlier
    order=['Thur', 'Fri', 'Sat', 'Sun'],   # orden de las categorias
    ax=axes[0],
)
axes[0].set_title('Propinas por dia y sexo (boxplot)')
axes[0].set_xlabel('Dia'); axes[0].set_ylabel('Propina ($)')

# ── Violin plot ────────────────────────────────────────
# El violin combina boxplot + KDE a ambos lados
# inner: que mostrar dentro del violin
#   'box' = caja, 'quart' = cuartiles, 'point' = puntos, 'stick' = ticks, None
sns.violinplot(
    data=penguins,
    x='species',
    y='flipper_length_mm',
    hue='sex',
    split=True,        # dividir el violin por hue (mitad izq / mitad der)
    inner='quart',     # mostrar cuartiles dentro del violin
    palette='husl',
    linewidth=1.2,
    ax=axes[1],
)
axes[1].set_title('Longitud de aleta: violin dividido por sexo')
axes[1].set_xlabel('Especie'); axes[1].set_ylabel('Longitud de aleta (mm)')

# ── Boxplot + stripplot superpuestos ──────────────────
# Tecnica recomendada: mostrar datos individuales sobre el boxplot
sns.boxplot(
    data=tips, x='day', y='total_bill',
    order=['Thur', 'Fri', 'Sat', 'Sun'],
    palette='pastel', width=0.5,
    flierprops={'alpha': 0},   # ocultar outliers del boxplot (los mostrara stripplot)
    ax=axes[2],
)
sns.stripplot(
    data=tips, x='day', y='total_bill',
    order=['Thur', 'Fri', 'Sat', 'Sun'],
    color='black', alpha=0.3, size=3, jitter=True,
    ax=axes[2],
)
axes[2].set_title('Boxplot + datos individuales (stripplot)')
axes[2].set_xlabel('Dia'); axes[2].set_ylabel('Cuenta total ($)')

plt.tight_layout()
plt.show()
```

### 5.2 `sns.barplot()` — Barras con estadisticas

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Barplot con CI ─────────────────────────────────────
# Seaborn calcula automaticamente:
#   - La media de 'tip' para cada valor de 'day'
#   - El IC 95% por bootstrap (2000 muestras por defecto)
# errorbar='ci': intervalo de confianza (bootstrap)
# errorbar='sd': desviacion estandar
# errorbar='se': error estandar
# errorbar=('ci', 95): CI al 95%
sns.barplot(
    data=tips,
    x='day',
    y='tip',
    hue='sex',
    order=['Thur', 'Fri', 'Sat', 'Sun'],
    estimator='mean',      # funcion de agregacion: 'mean', 'median', np.std, etc.
    errorbar=('ci', 95),   # IC 95% por bootstrap
    palette='pastel',
    capsize=0.08,          # tamano del capuchon de la barra de error
    ax=axes[0],
)
axes[0].set_title('Propina media ± IC 95% por dia y sexo')
axes[0].set_xlabel('Dia'); axes[0].set_ylabel('Propina media ($)')

# ── Barplot horizontal con estimador personalizado ────
sns.barplot(
    data=tips,
    y='day',
    x='total_bill',
    order=['Sun', 'Sat', 'Fri', 'Thur'],
    estimator='median',    # usar la mediana en lugar de la media
    errorbar=None,         # sin barras de error
    color='steelblue',
    orient='h',            # orientacion horizontal
    ax=axes[1],
)
axes[1].set_title('Cuenta mediana por dia')
axes[1].set_ylabel('Dia'); axes[1].set_xlabel('Cuenta total mediana ($)')

plt.tight_layout()
plt.show()
```

### 5.3 `sns.pointplot()` — Puntos con lineas de conexion

```python
fig, ax = plt.subplots(figsize=(10, 5))

# pointplot conecta los estimadores con lineas,
# util para ver tendencias entre categorias ordenadas
sns.pointplot(
    data=tips,
    x='day',
    y='tip',
    hue='sex',
    order=['Thur', 'Fri', 'Sat', 'Sun'],
    dodge=True,          # separar horizontalmente los grupos
    markers=['o', 's'],  # marcadores distintos por grupo
    linestyles=['-','--'],
    errorbar='se',       # barras de error = error estandar
    palette='Set1',
    ax=ax,
)
ax.set_title('Propina media ± SE por dia (pointplot)')
ax.set_xlabel('Dia'); ax.set_ylabel('Propina media ($)')
plt.tight_layout()
plt.show()
```

### 5.4 `sns.stripplot()` y `sns.swarmplot()`

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Stripplot: puntos con jitter aleatorio ────────────
# jitter=True: desplazamiento aleatorio en X para evitar solapamiento
# Muestra TODOS los datos individuales, sin agregacion
sns.stripplot(
    data=penguins,
    x='species',
    y='body_mass_g',
    hue='sex',
    dodge=True,      # separar por hue
    jitter=0.25,     # cantidad de jitter
    alpha=0.7,
    size=5,
    palette='Set2',
    ax=axes[0],
)
axes[0].set_title('Masa corporal por especie (stripplot)')
axes[0].set_xlabel('Especie'); axes[0].set_ylabel('Masa corporal (g)')

# ── Swarmplot: puntos sin solapamiento (no-overlapping) ─
# Usa el algoritmo beeswarm para posicionar los puntos
# sin que se solapen, manteniendo la distribucion real.
# Limitacion: lento con muchos datos (>500 puntos)
sns.swarmplot(
    data=penguins,
    x='species',
    y='body_mass_g',
    hue='sex',
    dodge=True,
    size=4,
    palette='Set2',
    ax=axes[1],
)
axes[1].set_title('Masa corporal por especie (swarmplot - sin solapamiento)')
axes[1].set_xlabel('Especie'); axes[1].set_ylabel('Masa corporal (g)')

plt.tight_layout()
plt.show()
```

### 5.5 `sns.countplot()` — Conteo de categorias

```python
titanic = sns.load_dataset('titanic')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Conteo de supervivientes por clase y sexo
sns.countplot(
    data=titanic,
    x='class',
    hue='survived',       # colorear por supervivencia (0=No, 1=Si)
    palette={0: '#e74c3c', 1: '#2ecc71'},
    order=['First', 'Second', 'Third'],
    ax=axes[0],
)
axes[0].set_title('Supervivencia por clase')
axes[0].set_xlabel('Clase'); axes[0].set_ylabel('Conteo')
# Cambiar etiquetas de la leyenda
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, ['No sobrevivio', 'Sobrevivio'], title='Supervivencia')

# Countplot horizontal
sns.countplot(
    data=titanic,
    y='embark_town',
    hue='sex',
    palette='Set2',
    ax=axes[1],
)
axes[1].set_title('Embarque por puerto y sexo')
axes[1].set_ylabel('Puerto de embarque')
axes[1].set_xlabel('Conteo')

plt.tight_layout()
plt.show()
```

### 5.6 `sns.catplot()` — Funcion a nivel de figura

```python
# catplot es el equivalente figure-level de todas las graficas categoricas
g = sns.catplot(
    data=tips,
    x='day',
    y='total_bill',
    hue='sex',
    col='time',            # columnas por 'time' (Lunch, Dinner)
    kind='violin',         # 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count'
    split=True,
    inner='quart',
    palette='husl',
    height=5,
    aspect=0.85,
    order=['Thur', 'Fri', 'Sat', 'Sun'],
)
g.set_axis_labels('Dia', 'Cuenta total ($)')
g.set_titles(col_template='{col_name}')
g.figure.suptitle('Distribucion de cuentas por dia, sexo y momento', y=1.03)
plt.show()
```

---

## 6. Graficas de Regresion

Seaborn facilita la visualizacion de relaciones lineales y no lineales entre variables, calculando automaticamente el ajuste y el intervalo de confianza.

### 6.1 `sns.regplot()` — Regresion con IC

La regresion lineal simple estima $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$ minimizando el error cuadratico:

$$\hat{\beta}_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

El intervalo de confianza del 95% para la linea de regresion:

$$\hat{y} \pm t_{n-2, 0.025} \cdot s_e \sqrt{\frac{1}{n} + \frac{(x - \bar{x})^2}{\sum_i (x_i - \bar{x})^2}}$$

```python
tips = sns.load_dataset('tips')
penguins = sns.load_dataset('penguins').dropna()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ── Regresion lineal simple ────────────────────────────
sns.regplot(
    data=tips,
    x='total_bill',
    y='tip',
    scatter_kws={'alpha': 0.5, 's': 40, 'color': 'steelblue'},
    line_kws={'color': 'red', 'lw': 2.5},
    ci=95,             # nivel del IC (95% por bootstrap)
    ax=axes[0],
)
# Calcular R² para anotarlo
r, p = stats.pearsonr(tips['total_bill'], tips['tip'])
axes[0].set_title(f'Regresion lineal (R²={r**2:.3f}, p={p:.2e})')
axes[0].set_xlabel('Cuenta total ($)'); axes[0].set_ylabel('Propina ($)')

# ── Regresion polinomial ───────────────────────────────
# order=2: ajuste cuadratico y = b0 + b1*x + b2*x²
sns.regplot(
    data=penguins,
    x='flipper_length_mm',
    y='body_mass_g',
    order=2,           # grado del polinomio
    scatter_kws={'alpha': 0.4, 's': 30},
    line_kws={'color': 'darkorange', 'lw': 2.5},
    ci=95,
    ax=axes[1],
)
axes[1].set_title('Regresion polinomial (grado 2)')
axes[1].set_xlabel('Longitud de aleta (mm)'); axes[1].set_ylabel('Masa corporal (g)')

# ── Regresion logistica ────────────────────────────────
# logistic=True: ajuste de regresion logistica P(Y=1|X)
# Util para variables binarias
sns.regplot(
    data=tips,
    x='total_bill',
    y=(tips['tip'] > 3).astype(int),   # variable binaria: propina > 3$
    logistic=True,
    scatter_kws={'alpha': 0.3, 's': 40},
    line_kws={'color': 'seagreen', 'lw': 2.5},
    ci=95,
    y_jitter=0.05,     # jitter vertical para ver los puntos binarios
    ax=axes[2],
)
axes[2].set_title('Regresion logistica: P(propina > 3$)')
axes[2].set_xlabel('Cuenta total ($)'); axes[2].set_ylabel('P(tip > 3$)')

plt.tight_layout()
plt.show()
```

### 6.2 `sns.lmplot()` — Regresion con facetas

```python
# lmplot es la version figure-level de regplot
# Permite crear grillas de subplots con col y row
g = sns.lmplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='smoker',      # linea de regresion separada por fumador
    col='time',        # columnas por momento del dia
    palette='Set1',
    scatter_kws={'alpha': 0.5, 's': 40},
    line_kws={'lw': 2},
    ci=95,
    height=4,
    aspect=1.1,
)
g.set_axis_labels('Cuenta total ($)', 'Propina ($)')
g.set_titles(col_template='{col_name}')
g.figure.suptitle('Regresion propina ~ cuenta por momento y tabaquismo', y=1.03)
plt.show()
```

### 6.3 `sns.residplot()` — Diagnostico de residuos

Los residuos del modelo $e_i = y_i - \hat{y}_i$ deben cumplir:
- Distribucion aproximadamente normal con media 0.
- Varianza constante (homocedasticidad).
- Independencia.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Grafica de residuos
sns.residplot(
    data=tips,
    x='total_bill',
    y='tip',
    lowess=True,         # superponer suavizado LOWESS para detectar patrones
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 2},
    ax=axes[0],
)
axes[0].axhline(0, color='black', lw=1, linestyle='--')
axes[0].set_title('Residuos de la regresion propina ~ cuenta')
axes[0].set_xlabel('Cuenta total ($)'); axes[0].set_ylabel('Residuo')

# QQ-plot para normalidad de residuos
from scipy import stats as scipy_stats
beta1, beta0, _, _, _ = scipy_stats.linregress(tips['total_bill'], tips['tip'])
residuos = tips['tip'] - (beta0 + beta1 * tips['total_bill'])
scipy_stats.probplot(residuos, dist='norm', plot=axes[1])
axes[1].set_title('QQ-plot de residuos (normalidad)')

plt.tight_layout()
plt.show()
```

---

<img src="../_static/images/seaborn2.png" width="600" />
## 7. Matrices y Heatmaps

### 7.1 `sns.heatmap()` — Mapa de calor avanzado

```python
# Generar datos de correlacion
np.random.seed(42)
variables = ['Ventas', 'Precio', 'Clientes', 'Satisfaccion', 'Margen', 'Devolucion']
n = len(variables)

datos_gen = np.random.randn(500, n)
datos_gen[:, 1] = -0.75 * datos_gen[:, 0] + 0.3 * np.random.randn(500)
datos_gen[:, 2] =  0.82 * datos_gen[:, 0] + 0.2 * np.random.randn(500)
datos_gen[:, 4] =  0.60 * datos_gen[:, 0] - 0.4 * datos_gen[:, 1] + 0.3 * np.random.randn(500)
datos_gen[:, 5] = -0.55 * datos_gen[:, 2] + 0.4 * np.random.randn(500)

df_gen   = pd.DataFrame(datos_gen, columns=variables)
corr_mat = df_gen.corr()   # Pearson por defecto

# Mascara para el triangulo superior (evitar redundancia)
# La matriz de correlacion es simetrica: solo mostrar la mitad inferior
mascara = np.triu(np.ones_like(corr_mat, dtype=bool))

fig, ax = plt.subplots(figsize=(8, 7))

sns.heatmap(
    corr_mat,
    mask=mascara,          # ocultar triangulo superior
    annot=True,            # mostrar valores numericos en cada celda
    fmt='.2f',             # formato de los numeros
    cmap='RdBu_r',         # paleta divergente centrada en 0
    vmin=-1, vmax=1,       # rango fijo de la escala de color
    square=True,           # celdas cuadradas
    linewidths=0.5,        # separacion entre celdas
    linecolor='white',
    cbar_kws={'label': 'Coeficiente de Pearson', 'shrink': 0.8},
    annot_kws={'fontsize': 10, 'fontweight': 'bold'},
    ax=ax,
)

ax.set_title('Matriz de Correlacion (triangulo inferior)', fontsize=13, pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.show()
```

### 7.2 `sns.clustermap()` — Heatmap con clustering jerarquico

El clustering jerarquico reorganiza las filas y columnas de la matriz para agrupar variables similares. Usa la distancia euclidea y el metodo de Ward por defecto:

$$d_{\text{Ward}}(A, B) = \frac{|A| \cdot |B|}{|A| + |B|} \|\bar{x}_A - \bar{x}_B\|^2$$

```python
# Pivotear flights para crear una matriz de tiempo x mes
flights = sns.load_dataset('flights')
flights_pivot = flights.pivot(index='year', columns='month', values='passengers')

# clustermap reorganiza filas y columnas por similitud
# usando clustering jerarquico (dendrograma)
g = sns.clustermap(
    flights_pivot,
    figsize=(12, 8),
    cmap='YlOrRd',
    standard_scale=1,      # normalizar por columna (z-score por mes)
                           # 0=filas, 1=columnas, None=sin normalizar
    method='ward',         # metodo de enlace: 'ward', 'complete', 'average', 'single'
    metric='euclidean',    # distancia: 'euclidean', 'correlation', 'cosine'
    row_cluster=True,      # activar clustering de filas
    col_cluster=True,      # activar clustering de columnas
    linewidths=0.3,
    cbar_pos=(0.02, 0.83, 0.03, 0.15),  # posicion de la barra de color
    dendrogram_ratio=0.15, # fraccion del espacio para el dendrograma
    annot=True,
    fmt='.0f',
    annot_kws={'fontsize': 7},
)
g.figure.suptitle('Clustermap: pasajeros aereos con agrupamiento jerarquico', y=1.01)
plt.show()
```

---

## 8. Figure-level vs Axes-level

Esta distincion es fundamental para entender la API de Seaborn.

### 8.1 Diferencias clave

| Aspecto | Axes-level | Figure-level |
|---------|-----------|-------------|
| Funciones | `histplot`, `scatterplot`, `boxplot`, etc. | `displot`, `relplot`, `catplot`, `lmplot` |
| Retorna | `Axes` de Matplotlib | `FacetGrid` de Seaborn |
| Parametro `ax` | Si, para especificar donde dibujar | No |
| Facetas (`col`, `row`) | No | Si |
| Integracion con plt | Total | Limitada |
| Control fino | Mayor | Menor |

```python
# ── AXES-LEVEL: integra con plt.subplots ──────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Se especifica ax= para controlar donde va
sns.histplot(data=tips, x='tip', ax=axes[0])
sns.boxplot(data=tips, x='day', y='tip', ax=axes[1])

plt.suptitle('Control total con axes-level')
plt.tight_layout()
plt.show()

# ── FIGURE-LEVEL: maneja su propio figure ─────────────
# NO usar plt.subplots() antes; la funcion crea su propia figura
g = sns.displot(
    data=tips,
    x='tip',
    col='time',      # crea automaticamente 2 subplots
    kind='hist',
    kde=True,
    height=4,
)
g.set_axis_labels('Propina ($)', 'Conteo')

# Acceder a los ejes subyacentes si es necesario
for ax in g.axes.flat:
    ax.axvline(tips['tip'].mean(), color='red', linestyle='--', lw=1.5)

plt.show()
```

---


<img src="../_static/images/seaborn3.png" width="600" />
## 9. Temas y Paletas de Color

### 9.1 Temas (styles)

```python
# Los 5 estilos de Seaborn
estilos = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
x = np.linspace(0, 2*np.pi, 100)

for ax, estilo in zip(axes, estilos):
    with sns.axes_style(estilo):         # context manager: solo para esta grafica
        ax.plot(x, np.sin(x), lw=2, color='steelblue')
        ax.plot(x, np.cos(x), lw=2, color='darkorange')
        ax.set_title(f"'{estilo}'", fontsize=10)

plt.suptitle('Los 5 estilos de Seaborn', y=1.05, fontsize=13)
plt.tight_layout()
plt.show()
```

<img src="../_static/images/seaborn4.png" width="600" />
### 9.2 Contextos (contexts)

Los contextos escalan el tamaño de todos los elementos de texto y lineas para distintos medios.

```python
# 'paper' < 'notebook' < 'talk' < 'poster'
contextos = ['paper', 'notebook', 'talk', 'poster']

for contexto in contextos:
    with sns.plotting_context(contexto):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, np.sin(x), lw=2)
        ax.set_title(f"Contexto: '{contexto}'")
        plt.tight_layout()
        plt.show()
```

### 9.3 Paletas de color

```python
# Tipos de paletas en Seaborn:
# - Cualitativas: para categorias sin orden natural
# - Secuenciales : para datos con orden de mayor a menor
# - Divergentes  : para datos con punto medio significativo (ej. correlaciones)

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

paletas_cualitativas = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
paletas_secuenciales = ['Blues', 'Greens', 'Oranges', 'viridis', 'plasma', 'magma']
paletas_divergentes  = ['RdBu', 'RdYlGn', 'coolwarm', 'vlag', 'icefire', 'Spectral']

todas_paletas = [
    ('Cualitativa: deep', 'deep', 'qual'),
    ('Cualitativa: pastel', 'pastel', 'qual'),
    ('Cualitativa: colorblind', 'colorblind', 'qual'),
    ('Secuencial: Blues', 'Blues', 'seq'),
    ('Secuencial: viridis', 'viridis', 'seq'),
    ('Secuencial: magma', 'magma', 'seq'),
    ('Divergente: RdBu', 'RdBu', 'div'),
    ('Divergente: coolwarm', 'coolwarm', 'div'),
    ('Divergente: vlag', 'vlag', 'div'),
    ('Husl: espaciado uniforme', 'husl', 'qual'),
    ('Set2: print-friendly', 'Set2', 'qual'),
    ('Hls: saturacion', 'hls', 'qual'),
]

datos_pal = pd.DataFrame({'x': range(10), 'y': np.random.randn(10).cumsum()})

for ax, (titulo, paleta, tipo) in zip(axes, todas_paletas):
    colores = sns.color_palette(paleta, 8)
    for i, color in enumerate(colores):
        ax.barh(i, i + 1, color=color, edgecolor='white')
    ax.set_xlim(0, 8)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')

plt.suptitle('Paletas de Seaborn: cualitativas, secuenciales y divergentes', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
```

### 9.4 Crear y usar paletas personalizadas

```python
# Paleta a partir de colores especificos
mi_paleta = sns.color_palette(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'])

# Paleta secuencial personalizada (de un color claro a oscuro)
paleta_seq = sns.light_palette('#2c7bb6', n_colors=8)
paleta_rev = sns.dark_palette('#00796B', n_colors=8, reverse=True)

# Paleta divergente personalizada
paleta_div = sns.diverging_palette(
    220, 20,        # tonos Hue para los dos extremos (grados en el circulo de color)
    n=11,           # numero de colores
    s=60,           # saturacion
    l=50,           # luminosidad
    center='light', # color del centro ('light' o 'dark')
)

# Ver cualquier paleta
sns.palplot(paleta_div)
plt.title('Paleta divergente personalizada')
plt.show()

# Usar en una grafica
with sns.color_palette(mi_paleta):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=tips, x='day', y='tip',
                order=['Thur', 'Fri', 'Sat', 'Sun'], ax=ax)
    ax.set_title('Boxplot con paleta personalizada')
    plt.tight_layout()
    plt.show()
```

---

## 10. FacetGrid y PairGrid

### 10.1 `sns.FacetGrid` — Grilla de subplots personalizada

`FacetGrid` es la clase base que usan `relplot`, `displot` y `catplot`. Se puede usar directamente para mayor control.

```python
tips = sns.load_dataset('tips')

# Crear la grilla de facetas
g = sns.FacetGrid(
    tips,
    col='time',          # columnas
    row='smoker',        # filas
    hue='sex',           # color
    height=3.5,
    aspect=1.2,
    margin_titles=True,  # titulos en el margen (mas limpio)
    palette='Set2',
)

# Mapear una funcion a cada subplot de la grilla
# map() aplica la funcion con los argumentos dados a cada celda
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.6, s=40)

# Añadir elementos a cada subplot
g.map(sns.regplot, 'total_bill', 'tip',
      scatter=False,    # no redibujar puntos
      ci=None,
      color='black',
      line_kws={'lw': 1.5, 'linestyle': '--'})

g.set_axis_labels('Cuenta total ($)', 'Propina ($)')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.add_legend(title='Sexo')
g.figure.suptitle('FacetGrid: scatterplot por tiempo, fumador y sexo', y=1.03)
plt.show()
```

### 10.2 `sns.pairplot()` — Matriz de graficas de pares

El pairplot muestra la relacion entre todas las pares de variables numericas de un DataFrame. La diagonal muestra la distribucion de cada variable individualmente.

```python
penguins = sns.load_dataset('penguins').dropna()

# pairplot completo con KDE en la diagonal
g = sns.pairplot(
    penguins,
    hue='species',           # colorear por especie
    vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
    diag_kind='kde',         # 'kde' o 'hist' en la diagonal
    plot_kws={'alpha': 0.5, 's': 30},
    diag_kws={'fill': True, 'alpha': 0.4, 'bw_adjust': 0.8},
    palette='husl',
    corner=False,            # True: solo el triangulo inferior
)

# Ajustar las etiquetas de los ejes
etiquetas = ['Largo pico\n(mm)', 'Prof. pico\n(mm)', 'Largo aleta\n(mm)', 'Masa\n(g)']
for i, etiq in enumerate(etiquetas):
    g.axes[i, 0].set_ylabel(etiq, fontsize=9)
    g.axes[-1, i].set_xlabel(etiq, fontsize=9)

g.figure.suptitle('Pairplot: relaciones entre variables de pinguinos', y=1.02, fontsize=13)
plt.show()
```

### 10.3 `sns.PairGrid` — Control total de la grilla de pares

```python
penguins = sns.load_dataset('penguins').dropna()

# PairGrid da control completo:
# - diagonal: que mostrar en la diagonal
# - upper: triangulo superior
# - lower: triangulo inferior
g = sns.PairGrid(
    penguins,
    vars=['bill_length_mm', 'flipper_length_mm', 'body_mass_g'],
    hue='species',
    palette='Set2',
    diag_sharey=False,
)

# Diagonal: histograma
g.map_diag(sns.histplot, kde=True, alpha=0.5, binwidth=3)

# Triangulo inferior: scatter
g.map_lower(sns.scatterplot, alpha=0.5, s=25)

# Triangulo superior: KDE 2D
g.map_upper(sns.kdeplot, fill=True, alpha=0.3, levels=4)

g.add_legend(title='Especie')
g.figure.suptitle('PairGrid: scatter (abajo), KDE 2D (arriba), histograma (diagonal)', y=1.02)
plt.show()
```

---

## 11. Combinacion con Matplotlib

Como Seaborn es una capa sobre Matplotlib, se puede combinar libremente.

```python
tips = sns.load_dataset('tips')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Grafica Seaborn en axes[0] ─────────────────────────
sns.violinplot(
    data=tips, x='day', y='tip', hue='sex',
    split=True, inner='quart',
    palette='husl', linewidth=1.2,
    order=['Thur', 'Fri', 'Sat', 'Sun'],
    ax=axes[0],
)

# Personalizacion pura de Matplotlib sobre la grafica de Seaborn
axes[0].set_title('Propinas por dia y sexo', fontsize=13, pad=12)
axes[0].set_xlabel('Dia de la semana', fontsize=11)
axes[0].set_ylabel('Propina ($)', fontsize=11)
axes[0].set_ylim(0, 12)
axes[0].axhline(tips['tip'].mean(), color='black', lw=1.5, ls='--',
                label=f'Media global: ${tips["tip"].mean():.2f}')
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# ── Grafica hibrida en axes[1] ─────────────────────────
# Seaborn para el scatter y Matplotlib para anotaciones
sns.scatterplot(
    data=tips, x='total_bill', y='tip',
    hue='size', palette='viridis_r',
    s=60, alpha=0.7, ax=axes[1],
)

# Añadir regresion con Matplotlib directamente
m, b = np.polyfit(tips['total_bill'], tips['tip'], 1)
x_fit = np.linspace(tips['total_bill'].min(), tips['total_bill'].max(), 100)
axes[1].plot(x_fit, m*x_fit + b, 'r--', lw=2, zorder=5,
             label=f'y = {m:.3f}x + {b:.3f}')

# Anotar el punto de mayor propina con Matplotlib
idx_max = tips['tip'].idxmax()
axes[1].annotate(
    f'Mayor propina\n${tips.loc[idx_max,"tip"]:.1f}',
    xy=(tips.loc[idx_max, 'total_bill'], tips.loc[idx_max, 'tip']),
    xytext=(30, 8),
    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
    fontsize=9, color='darkred',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
)
axes[1].set_title('Scatter Seaborn + anotaciones Matplotlib')
axes[1].set_xlabel('Cuenta total ($)')
axes[1].set_ylabel('Propina ($)')
axes[1].legend(title='Tamaño de grupo', fontsize=9)

plt.tight_layout()
plt.show()
```

---

## 12. Referencia Rapida

### Funciones principales por categoria

| Categoria | Funcion axes-level | Funcion figure-level |
|-----------|-------------------|---------------------|
| Relacional | `scatterplot`, `lineplot` | `relplot` |
| Distribucion | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` | `displot` |
| Categorica | `stripplot`, `swarmplot`, `boxplot`, `violinplot`, `barplot`, `pointplot`, `countplot` | `catplot` |
| Regresion | `regplot`, `residplot` | `lmplot` |
| Matrices | `heatmap`, `clustermap` | — |
| Pares | — | `pairplot`, `PairGrid` |
| Facetas | — | `FacetGrid` |

### Cuando usar cada tipo de grafica

| Situacion | Grafica recomendada |
|-----------|---------------------|
| Distribucion de una variable | `histplot` + KDE o `kdeplot` |
| Comparar distribuciones entre grupos | `violinplot` o `boxplot` + `stripplot` |
| Relacion entre dos numericas | `scatterplot` + `regplot` |
| Tendencia temporal | `lineplot` con CI |
| Comparar categorias (medias) | `barplot` con CI o `pointplot` |
| Matriz de correlacion | `heatmap` con `annot=True` |
| Todas las relaciones entre variables | `pairplot` o `PairGrid` |
| Explorar con facetas | `relplot`, `displot`, `catplot` |
| Probabilidad acumulada | `ecdfplot` |

---

## Referencias

- Waskom, M. L. (2021). Seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
- Documentacion oficial: https://seaborn.pydata.org
- Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1–23.
- Seaborn tutorial: https://seaborn.pydata.org/tutorial.html
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.

---

<!-- IMAGEN: Diagrama del modelo de objetos de Seaborn: FacetGrid envolviendo multiples Axes de Matplotlib, con flechas mostrando la relacion con Figure -->

<!-- IMAGEN: Galeria de las funciones figure-level vs axes-level organizadas por tipo: relational, distributions, categorical, regression -->

<!-- IMAGEN: Comparativa visual de los 5 estilos de Seaborn (darkgrid, whitegrid, dark, white, ticks) con la misma grafica en cada uno -->

<!-- IMAGEN: Rueda de paletas de Seaborn: cualitativas (deep, muted, pastel, colorblind), secuenciales y divergentes, con sus casos de uso -->

<!-- IMAGEN: PairGrid con los tres tipos de graficas: KDE en diagonal, scatter en triangulo inferior, KDE 2D en triangulo superior -->