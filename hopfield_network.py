import numpy as np
import matplotlib.pyplot as plt
import random


class HopfieldNetwork:
    """
    Implementación de una red de Hopfield para reconocimiento de patrones binarios.
    """

    def __init__(self, size):
        """
        Inicializa la red con una matriz de pesos de tamaño dado.

        :param size: Número de neuronas (tamaño del patrón a memorizar).
        """
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Entrena la red con una lista de patrones binarios.

        :param patterns: Lista de patrones (arrays 1D) a memorizar.
        """
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.size, 1))
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_pattern, iterations=10):
        """
        Ejecuta la red para recuperar un patrón a partir de una entrada ruidosa.

        :param input_pattern: Patrón de entrada (array 1D).
        :param iterations: Número de iteraciones de actualización.
        :return: Lista de estados del patrón en cada iteración.
        """
        state = input_pattern.copy()
        states = [state.copy()]
        for _ in range(iterations):
            for _ in range(self.size):
                i = random.randint(0, self.size - 1)
                state[i] = np.sign(np.dot(self.weights[i], state))
            states.append(state.copy())
        return states


def generate_noisy_pattern(pattern, noise_level=0.3):
    """
    Genera una versión ruidosa de un patrón binario.

    :param pattern: Patrón original (array 1D).
    :param noise_level: Proporción de bits a invertir.
    :return: Patrón ruidoso.
    """
    noisy_pattern = pattern.copy()
    num_noisy = int(noise_level * len(pattern))
    noisy_indices = random.sample(range(len(pattern)), num_noisy)
    for i in noisy_indices:
        noisy_pattern[i] *= -1
    return noisy_pattern


def visualize_evolution(original, processed_states):
    """
    Visualiza la evolución del patrón a lo largo de las iteraciones.

    :param original: Patrón ruidoso inicial.
    :param processed_states: Lista de patrones generados por la red.
    """
    fig, axs = plt.subplots(1, len(processed_states) + 1, figsize=(15, 5))
    axs[0].imshow(original.reshape((10, 10)), cmap='gray')
    axs[0].set_title('Imagen Original')
    axs[0].axis('off')

    for i, state in enumerate(processed_states):
        axs[i + 1].imshow(state.reshape((10, 10)), cmap='gray')
        axs[i + 1].set_title(f'Iteración {i}')
        axs[i + 1].axis('off')
        axs[i + 1].plot(4.5, 4.5, 'ro')  # Punto A
        axs[i + 1].text(5, 5, 'Punto A', color='red', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    clean_pattern = np.array([
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
        [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]).flatten()

    noisy_pattern = generate_noisy_pattern(clean_pattern, noise_level=0.3)

    hopfield = HopfieldNetwork(100)
    hopfield.train([clean_pattern])

    processed_states = hopfield.predict(noisy_pattern, iterations=5)

    visualize_evolution(noisy_pattern, processed_states)
