import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACION DE LA TRANSFORMACION ---
RADIO_CONOCIDO = 55 # Radio (R) conocido en pixeles
VOTES_THRESHOLD = 28 # Umbral minimo de votos para considerar un centro
ANGULO_STEP = 5 # Paso angular para el voto, en grados

def hough_transform_circulo(img_path):
    # DetecciOn de Bordes
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicamos un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Usamos Canny para la detección de bordes
    bordes = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    
    # Dimensiones de la imagen
    alto, ancho = bordes.shape
    
    # Acumulador de Hough
    # Plano 2D que representa los posibles centros (x0, y0)
    # Su dimensión es igual a la imagen original.
    acumulador = np.zeros((alto, ancho), dtype=np.uint16)

    # Coordenadas de los pixeles de borde (x, y)
    indices_y, indices_x = np.where(bordes > 0)
    
    # Rango de angulos para votar
    angulos = np.deg2rad(np.arange(0, 360, ANGULO_STEP)) # 0 a 360 grados, ya que el centro puede estar en cualquier dirección

    # Acumulacion de votos 
    # Iteramos sobre cada pixel de borde (x, y)
    for x, y in zip(indices_x, indices_y):
        # Para cada pixel de borde, vota por un círculo de posibles centros (x0, y0)
        for angulo in angulos:
            # Calculo de las coordenadas del centro (x0, y0) usando la fórmula:
            # x0 = x - R * cos(a)
            # y0 = y - R * sin(a)
            x0 = int(x - RADIO_CONOCIDO * np.cos(angulo))
            y0 = int(y - RADIO_CONOCIDO * np.sin(angulo))
            
            # Incremetamos el voto si el centro cae dentro de los límites de la imagen
            if 0 <= x0 < ancho and 0 <= y0 < alto:
                acumulador[y0, x0] += 1

    # Deteccion de Picos en el Acumulador
    centros_detectados = []
    
    # Encontramos los picos (acumulaciones por encima del umbral)
    indices_pico_y, indices_pico_x = np.where(acumulador > VOTES_THRESHOLD)

    for x0, y0 in zip(indices_pico_x, indices_pico_y):
        # Filtro para eliminar picos muy cercanos
        es_nuevo_centro = True
        for cx, cy in centros_detectados:
            if np.sqrt((x0 - cx)**2 + (y0 - cy)**2) < (RADIO_CONOCIDO / 5):
                es_nuevo_centro = False
                break
        
        if es_nuevo_centro:
            centros_detectados.append((x0, y0))
            
            # Dibujamos el círculo detectado en verde y su centro en rojo
            cv2.circle(img, (x0, y0), RADIO_CONOCIDO, (0, 255, 0), 2)
            cv2.circle(img, (x0, y0), 3, (0, 0, 255), -1) # Centro en rojo

    # Mostramos resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagen con Circunferencias Detectadas'), plt.axis('off')
    plt.subplot(122), plt.imshow(acumulador, cmap='hot', aspect='auto')
    plt.title('Espacio de Acumuladores (Centros x0, y0)'), plt.xlabel('X (x0)'), plt.ylabel('Y (y0)')
    plt.show()

    print(f"\n--- Resultados ---")
    print(f"Radio Conocido (R): {RADIO_CONOCIDO} píxeles")
    print(f"Centros Detectados (X, Y): {centros_detectados}")
    
    # Retorna las coordenadas del centro para ser usadas por el robot
    return centros_detectados

# Ejemplo de uso
hough_transform_circulo('imagen.png')