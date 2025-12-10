import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACION DE LA TRANSFORMCION ---
THETA_MAX = 180 # Rango de angulos en grados
THETA_STEP = 1  # Paso de 1 grado para Theta
RHO_STEP = 1    # Paso de 1 pixel para Rho, distancia
VOTES_THRESHOLD = 150 # Umbral minimo de votos para considerar una recta

def hough_transform_linea(img_path):
    # Deteccion de Bordes
    # Leemos la imagen seleccionada, la convertimos a escala de grises y aplicamos Canny
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Usamos Canny para la detección de bordes
    bordes = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    
    # Dimensiones de la imagen
    alto, ancho = bordes.shape
    
    # Acumulador de Hough
    # RHO_MAX es la diagonal de la imagen, distancia máxima del origen al borde
    RHO_MAX = int(np.sqrt(alto**2 + ancho**2))
    
    # Definimos el espacio theta
    thetas = np.deg2rad(np.arange(0, THETA_MAX, THETA_STEP)) # convierte a radianes
    num_thetas = len(thetas)
    
    # Definimos el espacio rho, desde -RHO_MAX hasta RHO_MAX
    rhos = np.arange(-RHO_MAX, RHO_MAX + 1, RHO_STEP)
    num_rhos = len(rhos)
    
    # Inicializar la matriz acumuladora
    acumulador = np.zeros((num_rhos, num_thetas), dtype=np.uint8)

    # Coordenadas de los pixeles de borde (x, y)
    indices_y, indices_x = np.where(bordes > 0)

    # Acumulacion de votos
    # Iteramos sobre cada pixel de borde (x, y) y votar en el espacio (rho, theta)
    for x, y in zip(indices_x, indices_y):
        for theta_idx, theta in enumerate(thetas):
            # Calculo de la formula central de Hough
            # rho = x * cos(theta) + y * sin(theta)
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            
            # Mapear el valor de rho (que puede ser negativo) a un índice de array (positivo)
            rho_idx = int((rho + RHO_MAX) / RHO_STEP)
            
            # Incremento el voto
            if 0 <= rho_idx < num_rhos:
                acumulador[rho_idx, theta_idx] += 1

    # Deteccion de Picos en el Acumulador
    rectas_detectadas = []
    
    # Encontrar los picos (acumulaciones por encima del umbral)
    for rho_idx in range(num_rhos):
        for theta_idx in range(num_thetas):
            if acumulador[rho_idx, theta_idx] > VOTES_THRESHOLD:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                rectas_detectadas.append((rho, theta))
                
                # Dibujar las líneas detectadas sobre la imagen original
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # Puntos para dibujar la línea
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Dibujar la línea verde
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Mostrar resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagen con Rectas Detectadas'), plt.axis('off')
    plt.subplot(122), plt.imshow(acumulador, aspect='auto', extent=[0, THETA_MAX, RHO_MAX, -RHO_MAX])
    plt.title('Espacio de Acumuladores (Hough)'), plt.xlabel(r'$\theta$ (grados)'), plt.ylabel(r'$\rho$ (píxeles)')
    plt.show()

    print(f"\n--- Resultados ---")
    print(f"Número de Rectas Detectadas (con umbral {VOTES_THRESHOLD}): {len(rectas_detectadas)}")
    # La salida incluye la imagen con las líneas dibujadas y el mapa de acumuladores

# Ejecutamos la función con una imagen de ejemplo
hough_transform_linea("pasillo.png") 