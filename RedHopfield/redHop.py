import numpy as np

def display_pattern(pattern):
    #Patron 1D como una matriz 2D usando caracteres
    size = int(np.sqrt(len(pattern)))
    for i in range(size):
        line = ""
        for j in range(size):
            # Usa '#' para píxel activo (+1) y '.' para inactivo (-1)
            line += '#' if pattern[i * size + j] == 1 else '.'
        print(line)
    print("-" * size)

class HopfieldNetwork:
    def __init__(self, size):
        #Inicializar la red con un tamaño N (N = altura * ancho)
        self.size = size
        # La matriz de pesos se inicializa en ceros
        self.weights = np.zeros((size, size))

    def train_hebbian(self, pattern):
        #Entrenar la red con un unico patron usando la regla de Hebb
        # El patrón debe ser un vector de -1s y +1s
        # W = P * P^T
        self.weights = np.outer(pattern, pattern)
        # La diagonal de la matriz de pesos debe ser cero
        np.fill_diagonal(self.weights, 0)

    def predict(self, noisy_pattern, max_iter=100):
        #Intentamos reconstruir un patrón a partir de una version ruidosa
        current_pattern = np.copy(noisy_pattern)
        
        print("Iniciando reconstrucción...")
        print("Patron con ruido:")
        display_pattern(current_pattern)

        for iteration in range(max_iter):
            last_pattern = np.copy(current_pattern)
            
            # Actualizacion asincrona, se actualiza una neurona a la vez en orden aleatorio
            update_order = np.random.permutation(self.size)
            
            for i in update_order:
                # Calcular la activacion, h_i = Σ(W_ij * s_j)
                activation = np.dot(self.weights[i, :], current_pattern)
                # Aplicamos la funcion de signo
                current_pattern[i] = 1 if activation >= 0 else -1

            print(f"Iteración {iteration + 1}:")
            display_pattern(current_pattern)
            
            # Si el patron ya no cambia, ha convergido
            if np.array_equal(current_pattern, last_pattern):
                print(f"Convergencia alcanzada en la iteración {iteration + 1}.")
                return current_pattern
        
        print("Se alcanzó el máximo de iteraciones. El patrón final es:")
        display_pattern(current_pattern)
        return current_pattern

# --- Ejemplo de Uso ---

# 1. Patron original (un "aro" en una grilla de 10x10)
# -1 representa el fondo (.), +1 representa la figura (#)
aro_pattern_2d = np.array([
    [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
    [-1,-1, 1,-1,-1,-1,-1, 1,-1,-1],
    [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
    [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
    [-1, 1,-1,-1,-1,-1,-1,-1, 1,-1],
    [-1,-1, 1,-1,-1,-1,-1, 1,-1,-1],
    [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1]
])

# Aplanamos el patron a un vector 1D
aro_pattern_1d = aro_pattern_2d.flatten()

# 2. Creamos una versión con ruido del patron
noisy_pattern = np.copy(aro_pattern_1d)
# Invertirmos el valor de 15 píxeles al azar para simular ruido
noise_indices = np.random.choice(len(noisy_pattern), 15, replace=False)
noisy_pattern[noise_indices] *= -1

# 3. Crear, entrenar y usar la red de Hopfield
network_size = 10 * 10
hopfield_net = HopfieldNetwork(network_size)

print("Patrón original a memorizar:")
display_pattern(aro_pattern_1d)

hopfield_net.train_hebbian(aro_pattern_1d)

# 4. Intentamos reconstruir el patron original desde la versión con ruido
reconstructed_pattern = hopfield_net.predict(noisy_pattern)