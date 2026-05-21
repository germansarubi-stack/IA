# Algoritmos de Visión Artificial y Redes Neuronales 👁️🧠

Este repositorio contiene un conjunto de herramientas y scripts en **Python** enfocados en la resolución de problemas clásicos de 
visión por computadora y memoria asociativa. Incluye implementaciones desde cero de la **Transformada de Hough** 
(tanto para detección de rectas como de circunferencias de radio conocido) y una **Red Neuronal de Hopfield** para la recuperación de 
patrones con ruido.

---

## 📋 Contenido del Repositorio

A continuación se detallan los módulos principales incluidos en este proyecto:

### 1. Detección de Circunferencias (`HoughCircunferencia.py`)
Implementación de la Transformada de Hough para detectar circunferencias con un **radio predefinido (específico)** en imágenes. 
* **Caso de uso común:** Ideal para aplicaciones de robótica o automatización donde se requiere localizar objetos cilíndricos, 
monedas o ruedas de tamaño constante.
* **Flujo interno:**
  1. Suavizado de imagen mediante Desenfoque Gaussiano.
  2. Detección de bordes con el algoritmo de Canny.
  3. Acumulación de votos en un plano 2D que mapea los posibles centros $(x_0, y_0)$.
  4. Filtrado de picos cercanos para evitar detecciones redundantes.

### 2. Detección de Líneas (`HoughRectas.py`)
Implementación de la Transformada de Hough lineal estándar utilizando la parametrización de coordenadas polares ($\rho, \theta$).
* **Caso de uso común:** Detección de carriles, pasillos, estructuras arquitectónicas o bordes rectilineos en entornos cerrados.
* **Flujo interno:**
  1. Extracción de bordes con Canny.
  2. Mapeo de cada píxel de borde al espacio de Hough en formato de matriz discretizada de $(\rho, \theta)$.
  3. Búsqueda de máximos locales (picos) que superan un umbral de votación para trazar las líneas verdes sobre la imagen original.

### 3. Red de Hopfield (`redHop.py`)
Una implementación de una red neuronal autoasociativa recurrente basada en la regla de aprendizaje de Hebb (*Hebbian Learning*).
* **Propósito:** Memorizar un patrón binario (matriz de $10 \times 10$ que representa un aro) y reconstruirlo de manera asíncrona a 
partir de una versión contaminada con ruido aleatorio.
* **Visualización:** Muestra en la terminal mediante caracteres (`#` para píxeles activos y `.` para inactivos) la evolución iterativa 
de la red hasta alcanzar la convergencia biestable.

---

## 🛠️ Tecnologías y Librerías Utilizadas

Los scripts están desarrollados en Python y dependen de las siguientes librerías estándar en procesamiento de imágenes y computación 
científica:

* **OpenCV (`cv2`):** Para la lectura de imágenes, conversión a escala de grises, desenfoques y el algoritmo de Canny.
* **NumPy (`np`):** Para el procesamiento eficiente de matrices, cálculo de productos externos en el entrenamiento Hebbiano y funciones 
trigonométricas.
* **Matplotlib (`plt`):** Para renderizar las comparativas entre las imágenes procesadas y sus respectivos espacios de acumuladores de 
Hough.

---

## 🚀 Instalación y Requisitos

Para clonar y ejecutar este proyecto de forma local, asegúrate de tener instalado Python 3 y las dependencias correspondientes.

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO
### 2. Instalar dependencias
Puedes instalarlas fácilmente usando pip:pip install opencv-python numpy matplotlib

## 💻 Modos de Uso y Configuración
Cada script cuenta con una sección editable en la cabecera para ajustar los hiperparámetros según la imagen o el patrón a analizar:

Ajustes en HoughCircunferencia.py
RADIO_CONOCIDO = 55    # Radio exacto de la circunferencia a buscar (en píxeles)
VOTES_THRESHOLD = 28   # Sensibilidad: votos mínimos para validar un centro
ANGULO_STEP = 5       # Precisión angular (en grados) para el bucle de votación

Para ejecutarlo: Asegúrate de contar con una imagen llamada imagen.png en el mismo directorio (o cambia la ruta al final del archivo) y 
ejecuta:python HoughCircunferencia.py

Ajustes en HoughRectas.py
THETA_MAX = 180        # Rango angular máximo (0 a 180°)
THETA_STEP = 1         # Resolución angular (pasos de 1 grado)
VOTES_THRESHOLD = 150  # Cantidad de píxeles colineales mínimos para trazar la recta

Para ejecutarlo: Coloca una imagen de prueba (ej. pasillo.png) y corre:python HoughRectas.py

Ejecución de redHop.py
Este script genera su propio patrón interno y le inyecta un ruido de 15 píxeles invertidos al azar de forma automática. 
No requiere archivos externos:python redHop.py

## 📊 Ejemplos de Salida Esperada
Para Hough (Líneas/Circunferencias): Se abrirá una ventana de matplotlib mostrando dos paneles: la imagen original con las geometrías 
detectadas en verde y el mapa térmico (heatmap) del espacio del acumulador, facilitando la interpretación visual de los picos.

Para la Red de Hopfield: Verás el paso a paso en la consola mostrando cómo el patrón ruidoso se va limpiando dinámicamente celda por 
celda hasta recuperar el aro perfecto original.
Patron con ruido:
..##.##...
.#..#..#..
#....#...#
... (etc)
--------------------
Iteración 1:
...
Convergencia alcanzada.

