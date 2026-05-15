# 2. 11 Buenas Practicas de Visualizacion de Datos

## Indice

1. [Principios fundamentales](#1-principios-fundamentales)
2. [Percepcion visual y preatencion](#2-percepcion-visual-y-preatencion)
3. [Eleccion y uso del color](#3-eleccion-y-uso-del-color)
4. [Tipografia y texto](#4-tipografia-y-texto)
5. [Ejes, escalas y proporciones](#5-ejes-escalas-y-proporciones)
6. [Reduccion del ruido visual](#6-reduccion-del-ruido-visual)
7. [Accesibilidad](#7-accesibilidad)
8. [Narrativa con datos](#8-narrativa-con-datos)
9. [Errores comunes](#9-errores-comunes)
10. [Lista de verificacion](#10-lista-de-verificacion)

---

## 1. Principios Fundamentales

### 1.1 La jerarquia de precision perceptual

Cleveland y McGill (1984) demostraron experimentalmente que los humanos percibimos los encodings visuales con distinta precision. De mayor a menor exactitud:

```
1. Posicion en escala comun          ← mas preciso
2. Posicion en escalas no alineadas
3. Longitud
4. Angulo / pendiente
5. Area
6. Volumen
7. Densidad / saturacion de color
8. Tono de color                     ← menos preciso
```

**Implicacion directa:** cuando la comparacion precisa de valores es importante, usar posicion (graficas de barras, puntos en eje comun) en lugar de area (burbujas) o angulo (torta).

### 1.2 La relacion datos-tinta (Data-Ink Ratio)

Tufte (1983) define la relacion datos-tinta como:

$$\text{Data-Ink Ratio} = \frac{\text{Tinta usada para representar datos}}{\text{Tinta total de la grafica}}$$

El objetivo es **maximizar este ratio**: eliminar todo elemento visual que no aporte informacion sobre los datos.

```
Elementos que reducen el Data-Ink Ratio (eliminar si no aportan):
  - Bordes de las barras cuando el color ya las delimita
  - Cuadricula densa cuando los valores exactos estan anotados
  - Fondo de color cuando el blanco sirve igual
  - Sombras y efectos 3D en graficas 2D
  - Leyenda cuando se puede etiquetar directamente
  - Repeticion de etiquetas que ya aparecen en el eje
  - Marcadores en cada punto de una linea continua densa
```

### 1.3 Carga cognitiva

La memoria de trabajo humana es limitada (Miller, 1956: 7 ± 2 elementos simultaneos). Una visualizacion efectiva minimiza la carga cognitiva:

```
Principio de proximidad:   elementos relacionados, cerca entre si.
Principio de similitud:    elementos del mismo tipo, mismo color o forma.
Principio de continuidad:  el ojo sigue lineas y curvas naturalmente.
Principio de cierre:       el ojo completa formas incompletas.
Principio de figura-fondo: el objeto de atencion destaca del fondo.
```

### 1.4 El proposito de la visualizacion

Antes de construir cualquier grafica, responder:

```
1. Quien es la audiencia?
   Tecnica (cientificos, ingenieros): tolera mas complejidad y densidad.
   Ejecutiva (directivos): prefiere simpleza, KPIs y narrativa clara.
   General (publico amplio): necesita maxima claridad, minima jerga.

2. Que decision o accion debe facilitar?
   Comparar, identificar tendencia, mostrar distribucion, detectar outlier,
   mostrar relacion, mostrar composicion, mostrar flujo geografico...

3. Que medio la mostrara?
   Pantalla interactiva: mas elementos, tooltips, zoom.
   Presentacion (proyector): tamanos grandes, alto contraste, pocos elementos.
   Impreso en papel: sin interactividad, tamano fijo, colores limitados.
   Dashboard web: actualizacion en tiempo real, multiples vistas.
```

---

## 2. Percepcion Visual y Preatencion

### 2.1 Atributos preatentivos

Los atributos preatentivos son propiedades visuales que el cerebro procesa en menos de 250ms, **antes** de que la atencion consciente se enfoque. Usarlos correctamente permite guiar la atencion del lector sin esfuerzo.

| Atributo | Eficaz para | Ejemplo de uso |
|----------|-------------|----------------|
| Color (tono) | Categorias, resaltar uno | Una barra roja entre barras grises |
| Intensidad de color | Magnitud ordinal | Heatmap de correlacion |
| Tamano | Magnitud cuantitativa | Grafica de burbujas |
| Forma | Categorias (max 5-6) | Marcadores distintos en scatter |
| Orientacion | Direccion, anomalia | Flecha apuntando en sentido contrario |
| Longitud | Magnitud cuantitativa | Barras de distinto largo |
| Posicion | Magnitud cuantitativa, relacion | Punto en eje |
| Movimiento | Llamar atencion (animaciones) | Punto parpadeante |
| Borde / contorno | Agrupacion, enfasis | Rectangulo alrededor de un grupo |

**Regla de oro:** usar un solo atributo preatentivo para resaltar lo mas importante. Usar dos o mas simultaneously anula el efecto.

### 2.2 Codificacion multiple

Cuando el mismo dato se codifica con dos atributos simultaneamente, la percepcion mejora. Ejemplo canonico: en un grafico de barras de categorias, usar **posicion** (longitud de barra) Y **color** para codificar las mismas categorias.

```
Codificacion simple (solo posicion):
  Barras de igual color -> distinguibles por longitud.

Codificacion doble (posicion + color):
  Barras de distinto color -> distinguibles por longitud Y color.
  Mas rapido de leer, especialmente cuando hay leyenda separada.
```

### 2.3 Efecto de figura y fondo

El fondo debe ser neutral para no competir con los datos. Las reglas:

```
Fondo blanco o gris muy claro: optimo para publicaciones e informes.
Fondo oscuro (dark mode): util en dashboards nocturnos o de control,
                          requiere paleta de colores adaptada.
Fondo con cuadricula visible: la cuadricula debe ser mas clara que los datos.
Fondo con imagen: casi siempre reduce la legibilidad. Evitar.
```

---

## 3. Eleccion y Uso del Color

### 3.1 Los tres tipos de paletas

**Paleta cualitativa (categorica):**
- Para variables nominales sin orden (paises, categorias de producto, especies).
- Los colores deben ser igualmente distinguibles; ninguno debe destacar sobre otro.
- Maximo 8-10 categorias con color; mas de eso es inlegible.

```
Buenas paletas cualitativas:
  ColorBrewer Set1, Set2, Set3
  Tableau 10, Tableau 20
  Plotly D3, G10, Bold
  Seaborn "deep", "muted", "colorblind"
```

**Paleta secuencial:**
- Para variables cuantitativas con un minimo y un maximo.
- Progresa de un color claro (valor bajo) a uno oscuro o saturado (valor alto).
- El lector infiere naturalmente que mas oscuro = mas valor.

```
Buenas paletas secuenciales:
  Viridis, Plasma, Inferno, Magma, Cividis (perceptualmente uniformes)
  Blues, Greens, Oranges, YlOrRd, BuPu
  Evitar: arcoiris (no perceptualmente uniforme, problemas de daltonismo)
```

**Paleta divergente:**
- Para variables cuantitativas con un punto medio significativo (cero, media).
- Progresa de un color en un extremo, pasa por un color neutro en el centro, y llega a otro color en el extremo opuesto.

```
Buenas paletas divergentes:
  RdBu (rojo-blanco-azul)
  RdYlGn (rojo-amarillo-verde, problemas con daltonismo rojo-verde)
  Coolwarm, Spectral, PiYG
  Uso tipico: correlaciones, diferencias vs media, anomalias
```

### 3.2 Principios de uso del color

```
UN SOLO COLOR para graficas de una sola serie:
  Una grafica de barras con todas las barras del mismo color es mas limpia
  y eficiente que barras de distintos colores sin significado.

COLOR CON SIGNIFICADO, no con decoracion:
  Cada color debe codificar informacion. Si todas las barras son de distinto
  color pero no hay leyenda ni diferencia conceptual, el color es ruido.

RESALTAR CON COLOR SATURADO, apagar el resto:
  La tecnica mas efectiva: un elemento en color saturado (rojo, naranja),
  el resto en gris. El ojo va directamente al elemento destacado.

CONSISTENCIA en todo el dashboard o informe:
  Si "Norte" es azul en una grafica, debe ser azul en todas las demas.
  Cambiar el significado del color entre graficas confunde al lector.

SEMANTICA DEL COLOR cuando existe convencion:
  Rojo = negativo, error, perdida / Verde = positivo, ok, ganancia.
  Azul = agua, frio / Naranja = calor, advertencia.
  No invertir estas convenciones salvo razon muy clara.
```

### 3.3 Contraste y legibilidad

El ratio de contraste entre texto y fondo debe cumplir WCAG 2.1:

```
Nivel AA (minimo recomendado):
  Texto normal (< 18pt): ratio minimo 4.5:1
  Texto grande (>= 18pt o negrita >= 14pt): ratio minimo 3:1

Nivel AAA (optimo):
  Texto normal: ratio minimo 7:1
  Texto grande: ratio minimo 4.5:1

Ejemplos:
  Negro (#000000) sobre blanco (#FFFFFF): ratio 21:1 (maximo posible)
  Gris oscuro (#595959) sobre blanco:     ratio 7:1 (AAA texto normal)
  Gris claro (#767676) sobre blanco:      ratio 4.5:1 (AA texto normal)
  Blanco sobre azul oscuro (#003f88):     ratio ~8:1

Herramienta: https://webaim.org/resources/contrastchecker/
```

### 3.4 Numero de colores

```
1-2 colores: ideal para graficas simples de una o dos series.
3-5 colores: maximo comodo para el ojo humano sin esfuerzo.
6-8 colores: limite practico en graficas categoricas.
> 8 colores: considerar agrupar categorias, usar facetas, o cambiar el tipo de grafica.

Alternativas cuando hay muchas categorias:
  - Agrupar los menores en "Otros".
  - Usar facetas (un subplot por categoria).
  - Usar interactividad (highlight on hover).
  - Usar etiquetas directas en lugar de leyenda con muchos colores.
```

---

## 4. Tipografia y Texto

### 4.1 Jerarquia tipografica

```
Jerarquia recomendada (de mayor a menor):
  1. Titulo principal del informe / dashboard:  18-24pt, negrita
  2. Titulo de grafica:                         13-16pt, negrita o semibold
  3. Etiquetas de eje:                          10-12pt, regular
  4. Valores / etiquetas de datos:              9-11pt, regular
  5. Anotaciones y notas al pie:                8-10pt, regular o italic

Regla: no usar mas de 2 familias tipograficas en un mismo informe.
Regla: no usar mas de 3 tamaños distintos sin razon justificada.
```

### 4.2 Tipografias recomendadas

```
Para pantalla:
  Sans-serif: Roboto, Inter, Source Sans Pro, Lato, Open Sans, Segoe UI
  Monoespaciada (valores numericos en tablas): Roboto Mono, Source Code Pro

Para publicaciones impresas:
  Serif: Georgia, Palatino, Merriweather (cuerpo de texto con datos)
  Sans-serif: Helvetica, Arial, Gill Sans (titulos, etiquetas)

Evitar en visualizacion:
  Comic Sans, Papyrus, Curlz (no profesionales)
  Fuentes decorativas de un solo peso (poca versatilidad)
  Fuentes de muy bajo contraste entre fino y grueso en tamaños pequeños
```

### 4.3 Etiquetas de datos

```
CUANDO INCLUIR VALORES EN LA GRAFICA:
  Si hay pocos puntos (< 10-15): anotar directamente.
  Si la precision exacta importa mas que la tendencia: siempre anotar.
  Si el lector no puede leer el eje con precision suficiente: anotar.

CUANDO NO INCLUIR VALORES:
  Graficas de tendencia con muchos puntos (valores saturan el espacio).
  Cuando el eje es suficientemente legible.
  Cuando los valores exactos no son el objetivo (la forma importa mas).

FORMATO DE NUMEROS:
  Enteros grandes: usar separador de miles -> 1,250,000 (no 1250000).
  Decimales: no mas de 2 decimales salvo que la precision sea critica.
  Porcentajes: siempre con simbolo % y 1 decimal -> 12.5%.
  Moneda: simbolo delante, sin decimales si son enteros -> $1,250.
  Millones: abreviar cuando el espacio es limitado -> $1.25M.
```

### 4.4 Titulos y subtitulos efectivos

```
TITULO DESCRIPTIVO (que muestra):
  "Ventas mensuales 2024"
  -> Describe lo que se muestra, no lo que significa.

TITULO NARRATIVO (que dice):
  "Las ventas cayeron 18% en Q3 por la interrupcion de suministro"
  -> Comunica el hallazgo. Mas efectivo para presentaciones e informes.

SUBTITULO / CAPTION:
  Explicar la fuente, el periodo, las unidades, o la metodologia.
  "Fuente: Sistema ERP. Datos al 31/12/2024. Ventas en USD."

REGLA: El titulo debe ser lo primero que el lector lea y debe
       orientar inmediatamente sobre lo que esta viendo.
```

---

## 5. Ejes, Escalas y Proporciones

### 5.1 El eje Y y el cero

**Graficas de barras:** el eje Y DEBE comenzar en cero. La longitud de la barra es el encoding; truncar el eje distorsiona la percepcion de magnitud relativa.

```
CORRECTO (barra):            INCORRECTO (barra):
  100 |                        98 |
      | ██                        | ██ ██
      | ██ ██                     | ██ ██
    0 +─────                   96 +─────
         A  B                       A  B

  A parece el doble de B.     A parece 10 veces mayor que B (engañoso).
```

**Graficas de lineas:** el eje Y no necesita comenzar en cero. La pendiente de la linea es el encoding, y un rango ajustado a los datos muestra la variacion con mas detalle.

**Graficas de dispersion:** el rango de los ejes debe ajustarse a los datos con un margen moderado. No es necesario incluir el cero salvo que tenga significado para los datos.

### 5.2 Escalas logaritmicas

Usar escala logaritmica cuando:

```
- Los datos abarcan varios ordenes de magnitud (ej: 100 a 1,000,000).
- El crecimiento es porcentual (tasa de cambio constante = linea recta en log).
- Se quiere comparar la tasa de crecimiento, no el valor absoluto.
- Hay datos muy sesgados donde los valores grandes aplanan los pequeños.

Anotar claramente cuando se usa escala log:
  "Eje Y en escala logaritmica (base 10)"
  Mostrar las marcas del eje como valores reales (10, 100, 1000), no los exponentes.
```

### 5.3 Eje secundario

```
El eje secundario (doble eje Y) es controvertido:
  - Puede facilitar comparar dos series con escalas muy diferentes.
  - Puede manipular la percepcion ajustando las escalas independientemente.

Usar con cuidado:
  ✓ Cuando las dos series son conceptualmente relacionadas (ej: barras de volumen
    y linea de precio, o linea de temperatura y barras de precipitacion).
  ✓ Cuando queda claro que cada serie usa su propio eje.

Evitar cuando:
  ✗ La escala del eje secundario puede manipular la percepcion de correlacion.
  ✗ Hay mas de dos series (considerar facetas o normalizar los datos).
```

### 5.4 Aspect ratio (relacion de aspecto)

La pendiente percibida de una linea depende del aspect ratio. La regla de **banking to 45 degrees** (Cleveland, 1993) propone que el grafico debe tener un tamano tal que las lineas tengan una pendiente media de 45 grados, maximizando la percepcion de cambio.

```
Graficas de linea muy anchas y bajas:
  La linea parece casi plana; los cambios parecen insignificantes.

Graficas de linea muy altas y estrechas:
  La linea parece vertical; los cambios parecen dramaticos.

Regla practica: para series temporales, una relacion ancho:alto de 2:1 a 3:1
suele funcionar bien para la mayoria de datos.
```

### 5.5 Truncar graficas de area

Las graficas de area y de barras apiladas deben comenzar siempre en cero. El area es el encoding; cualquier base distinta de cero hace que el area no sea proporcional al valor.

---

## 6. Reduccion del Ruido Visual

### 6.1 El principio de eliminacion progresiva

Para limpiar una visualizacion, aplicar en orden:

```
1. Eliminar el borde del grafico (la caja exterior).
2. Eliminar o atenuar la cuadricula.
3. Eliminar las marcas (ticks) del eje.
4. Eliminar el fondo de color si no aporta.
5. Eliminar las etiquetas del eje si estan anotados los valores.
6. Reemplazar la leyenda por etiquetas directas en las series.
7. Eliminar el titulo del eje si el titulo de la grafica lo explica.
8. Verificar que cada elemento restante justifica su presencia.
```

### 6.2 Cuadricula

```
CUADRICULA PRINCIPAL (lineas mayores):
  Color gris claro (#DDDDDD a #EEEEEE).
  Grosor 0.5-1pt. Nunca mas oscura o gruesa que los datos.
  Solo horizontal en graficas de barras verticales.
  Solo vertical en graficas de barras horizontales.
  Ambas en scatter plots (ayuda a leer coordenadas).

CUADRICULA MENOR (lineas menores):
  Solo cuando se necesita leer valores intermedios con precision.
  Color aun mas claro que la cuadricula mayor.
  Tipicamente innecesaria si los valores estan anotados.

SIN CUADRICULA:
  Apropiado cuando los valores exactos no importan y la forma es el mensaje.
  Graficas de proporcion, diagramas de flujo, mapas.
```

### 6.3 Leyenda vs. Etiquetas directas

Las leyendas requieren movimiento de ojos entre la leyenda y los datos, aumentando la carga cognitiva. Las etiquetas directas son superiores cuando el espacio lo permite.

```
ETIQUETAS DIRECTAS (preferir):
  - La etiqueta va junto a la serie o barra.
  - El lector no necesita buscar en la leyenda.
  - Funciona bien con 2-5 series.

LEYENDA (cuando la etiqueta directa no es practica):
  - Muchas series superpuestas donde las etiquetas se solapan.
  - Espacio insuficiente junto a los elementos.
  - Ubicar la leyenda encima o debajo (no a la derecha donde compite con los datos).
  - Ordenar la leyenda en el mismo orden en que aparecen las series.
```

### 6.4 Sobreannotacion

```
ANOTACIONES UTILES:
  Señalar el punto mas alto y mas bajo de una serie.
  Marcar eventos importantes (lanzamientos, cambios de politica, crisis).
  Explicar un outlier o anomalia visible.
  Indicar la tendencia o el cambio porcentual clave.

ANOTACIONES A EVITAR:
  Anotar todos y cada uno de los valores (usar una tabla en su lugar).
  Anotaciones que repiten lo que ya dice el eje.
  Flechas o llamadas sin proposito claro.
  Texto de mas de 2 lineas en el canvas del grafico.
```

---

## 7. Accesibilidad

### 7.1 Daltonismo

Aproximadamente el 8% de los hombres y el 0.5% de las mujeres tienen algun grado de daltonismo. Los tipos mas comunes:

```
Deuteranopia / deuteranomalia (mas comun): confusion rojo-verde.
Protanopia / protanomalia:                 confusion rojo-verde (rojo muy oscuro).
Tritanopia / tritanomalia (rara):          confusion azul-amarillo.

Colores problematicos: combinaciones rojo-verde puras (#FF0000 + #00FF00).

Colores seguros para daltonismo rojo-verde:
  Azul (#0072B2), naranja (#E69F00), verde azulado (#009E73),
  amarillo (#F0E442), azul cielo (#56B4E9), rojo naranja (#D55E00),
  rosa (#CC79A7), negro (#000000).
  -> Esta es la paleta de Wong (2011), disenada para accesibilidad.

Paletas seguras predefinidas:
  ColorBrewer: marcar "colorblind safe" en el selector.
  Seaborn: palette="colorblind".
  Matplotlib: estilo "tableau-colorblind10".
```

### 7.2 Redundancia visual para accesibilidad

No depender solo del color para codificar informacion. Usar siempre un segundo encoding:

```
Color + forma del marcador (scatter).
Color + patron de relleno (barras: solido, rayado, punteado).
Color + etiqueta directa (lineas).
Color + posicion (con facetas).
Color + grosor de linea.
```

### 7.3 Tamanos minimos

```
Texto en pantalla:
  Cuerpo y etiquetas: minimo 11px (8pt).
  Tooltips y notas: minimo 10px (7.5pt).
  Nada por debajo de 9px es legible en pantalla a distancia normal.

Texto en presentaciones (proyectadas):
  Minimo 18pt para que sea legible desde el fondo de la sala.
  Titulos: 28-36pt.

Elementos graficos:
  Puntos en scatter: minimo 4px de diametro.
  Lineas: minimo 1.5px de grosor para ser legibles con claridad.
  Barras: minimo 4px de ancho.
```

### 7.4 Texto alternativo y descripciones

En visualizaciones web y dashboards accesibles:

```
Añadir texto alternativo (alt text) a cada imagen de grafica.
Formato recomendado del alt text:
  "[Tipo de grafica] de [variable] por [variable]. [Hallazgo principal]."
  
Ejemplo:
  "Grafico de barras de ventas mensuales 2024. Las ventas alcanzaron su
   maximo en diciembre con 2.3M USD, un 18% sobre el promedio anual."

En Power BI: Panel de formato → General → Texto alternativo.
En Plotly: fig.update_layout(annotations=[...]) o metadatos HTML.
```

---

## 8. Narrativa con Datos

### 8.1 La estructura de la historia

Una visualizacion efectiva sigue una estructura narrativa:

```
CONTEXTO:    Que estamos mirando? (titulo, subtitulo, fuente)
TENSION:     Cual es el problema o pregunta? (lo que el lector quiere saber)
RESOLUCION:  Cual es la respuesta? (el hallazgo destacado visualmente)
ACCION:      Que debe hacer el lector? (llamada a la accion o conclusion)
```

### 8.2 Destacar lo que importa

```
Tecnica 1 — Color selectivo:
  Todo en gris excepto el elemento mas importante, que va en color saturado.

Tecnica 2 — Anotacion directa:
  Un cuadro de texto o flecha que dice exactamente lo que el grafico muestra.
  "Las ventas cayeron 23% en este periodo."

Tecnica 3 — Orden:
  Ordenar barras de mayor a menor (o viceversa) para que la jerarquia
  sea inmediatamente visible. No usar orden alfabetico salvo que el lector
  necesite buscar un elemento especifico.

Tecnica 4 — Eliminacion:
  Mostrar solo los N elementos mas relevantes. Si hay 50 categorias,
  mostrar las 10 mas grandes y agrupar el resto en "Otros".

Tecnica 5 — Comparacion de referencia:
  Añadir una linea de promedio, meta o periodo anterior como referencia.
  El lector necesita un punto de comparacion para juzgar si algo es bueno o malo.
```

### 8.3 El orden de lectura

Los lectores de culturas occidentales siguen el patron **F** o **Z** al escanear:

```
Patron Z (dashboards y slides):
  [Inicio] → → → → → → [Fin]
     ↓                ↙
     ↓             ↙
  [Siguiente linea] → → [Fin]

Colocar el dato mas importante en la esquina superior izquierda.
El segundo elemento mas importante, en la superior derecha.
El resumen o conclusion, en la parte inferior.
```

### 8.4 Contexto es obligatorio

```
Siempre incluir:
  Unidades de medida en los ejes o en el titulo ("USD", "miles de usuarios", "kg").
  El periodo de tiempo cubierto ("Enero-Diciembre 2024").
  La fuente de los datos ("Fuente: Sistema CRM, actualizado 2024-12-31").
  Que significa cada color o forma (leyenda o etiquetas directas).
  N o tamano de muestra en analisis estadisticos.

Un numero sin contexto no comunica nada:
  "47%" -> de que? en que periodo? comparado con que?
  "47% de los clientes compraron mas de una vez en 2024,
   frente al 38% en 2023" -> ahora si comunica.
```

---

## 9. Errores Comunes

### 9.1 Graficas de torta con demasiadas categorias

```
PROBLEMA:
  Una torta con 10 o mas sectores es inlegible. Los angulos pequeños
  son casi imposibles de comparar (jerarquia de percepcion: angulo
  esta en el puesto 4, por debajo de posicion y longitud).

SOLUCION:
  Usar barras horizontales ordenadas de mayor a menor.
  Si se necesita parte-todo, usar solo con < 5 categorias y cuando
  los porcentajes son claramente distintos (no comparar 24% vs 26%).
```

### 9.2 Graficas 3D sin justificacion

```
PROBLEMA:
  El 3D añade perspectiva que distorsiona las proporciones.
  En una torta 3D, los sectores frontales parecen mas grandes que los traseros
  aunque sean iguales. En barras 3D, la profundidad es ruido visual puro.

SOLUCION:
  Usar 3D solo cuando los datos son realmente tridimensionales
  (superficie matematica, scatter 3D con una tercera variable real).
  Nunca usar 3D para embellecer datos que son 2D o 1D.
```

### 9.3 Doble eje Y mal usado

```
PROBLEMA:
  Dos series con escalas independientes pueden hacer que cualquier
  par de series parezca correlacionado ajustando las escalas.
  Ejemplo clasico: correlacion espuria entre el consumo de helados
  y la temperatura, en el mismo grafico que las ventas de paraguas.

SOLUCION:
  Normalizar ambas series (z-score o indexar a 100 en t=0).
  Usar paneles separados (facetas) con el mismo eje Y si es posible.
  Si se usa doble eje, que quede muy claro cuales datos usan cual eje.
```

### 9.4 Saturacion de informacion

```
PROBLEMA:
  Un dashboard con 20 graficas en una pagina abruma al lector.
  Nadie sabe donde mirar; ninguna metrica destaca.

SOLUCION:
  Regla de los 5-7 elementos: no mas de 5-7 graficas por pagina.
  Organizar por jerarquia: KPIs clave arriba, detalles abajo.
  Usar paginas o pestanas para organizar distintos niveles de detalle.
  Preguntar: si el lector solo puede ver un dato, cual seria?
             Ese dato debe ser el mas prominente.
```

### 9.5 Ordenacion incorrecta

```
PROBLEMA:
  Barras ordenadas alfabeticamente cuando el orden de magnitud es lo que importa.
  El ojo tiene que leer todos los valores para encontrar el mayor.

SOLUCION:
  Ordenar barras de mayor a menor (o viceversa) salvo que:
    - Las categorias tienen un orden natural (meses, dias, edad).
    - El lector necesita buscar un elemento especifico por nombre.
    - El orden temporal es el mensaje (evolucion en el tiempo).
```

### 9.6 Lineas en datos no continuos

```
PROBLEMA:
  Conectar con lineas datos que no son continuos implica que los valores
  intermedios existen e interpolan entre los puntos.
  Ejemplo: conectar con linea los ingresos de distintos departamentos
           implica que hay un continuo entre departamentos.

SOLUCION:
  Datos continuos (tiempo, temperatura): grafica de lineas.
  Datos discretos / categoricos: barras, puntos sin lineas.
```

### 9.7 Escala truncada en barras

```
PROBLEMA:
  Un grafico de barras que no empieza en 0 exagera visualmente
  las diferencias entre barras.

  Ejemplo: barras que van de 95% a 100% parecen que una es
           el doble de otra cuando en realidad difieren en 5 puntos.

SOLUCION:
  Barras siempre desde 0.
  Si el rango de interes es 95-100%, usar una grafica de puntos
  (dot plot) o anotar la diferencia textualmente.
```

---

## 10. Lista de Verificacion

### Antes de construir la grafica

```
[ ] He identificado la audiencia y su nivel tecnico.
[ ] He definido la pregunta o decision que la grafica debe facilitar.
[ ] Se el tipo de dato que tengo (continuo, discreto, categorico, temporal...).
[ ] He elegido el tipo de grafica adecuado para el tipo de dato y la tarea.
```

### Al construir la grafica

```
[ ] El eje Y empieza en 0 (si es barra o area).
[ ] Las escalas no estan truncadas de forma engañosa.
[ ] Uso maximo 5-6 colores con significado claro.
[ ] Los colores son accesibles para daltonismo.
[ ] El color no es el unico encoding (hay redundancia con forma, posicion o etiqueta).
[ ] La cuadricula es mas suave que los datos.
[ ] He eliminado bordes, sombras y efectos decorativos innecesarios.
[ ] Las etiquetas de eje son legibles (fuente, tamaño, angulo).
[ ] El titulo describe o comunica el hallazgo principal.
[ ] Las unidades estan indicadas.
[ ] La fuente y el periodo estan indicados.
```

### Al revisar la grafica

```
[ ] Puedo leer el mensaje principal en menos de 5 segundos.
[ ] Lo mas importante visualmente ES lo mas importante informativamente.
[ ] No hay elementos que compitan con el mensaje principal.
[ ] Si imprimo en blanco y negro, la grafica sigue siendo legible.
[ ] Si muestro solo el titulo y la grafica sin contexto adicional, se entiende sola.
[ ] Alguien que no conoce los datos puede entenderla sin explicacion oral.
[ ] He verificado que los datos representados son correctos.
```

---

## Referencias

- Tufte, E. R. (1983). *The Visual Display of Quantitative Information*. Graphics Press.
- Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application. *Journal of the American Statistical Association*, 79(387), 531-554.
- Cleveland, W. S. (1993). *Visualizing Data*. Hobart Press.
- Cairo, A. (2012). *The Functional Art: An Introduction to Information Graphics and Visualization*. New Riders.
- Knaflic, C. N. (2015). *Storytelling with Data*. Wiley.
- Few, S. (2012). *Show Me the Numbers: Designing Tables and Graphs to Enlighten*. Analytics Press.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.
- Wong, B. (2011). Color blindness. *Nature Methods*, 8(6), 441.
- Ware, C. (2004). *Information Visualization: Perception for Design*. Morgan Kaufmann.
- WCAG 2.1: https://www.w3.org/TR/WCAG21/
- ColorBrewer: https://colorbrewer2.org