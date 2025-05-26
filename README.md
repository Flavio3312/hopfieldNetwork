# 🧠 Reconocimiento de Patrones con Redes de Hopfield

Este proyecto implementa una **red neuronal de Hopfield** para el reconocimiento y reconstrucción de patrones binarios, específicamente aplicado a la detección de la forma de un cilindro en una imagen ruidosa. El objetivo es que el sistema pueda identificar correctamente la forma y determinar el **punto de perforación (Punto A)** en coordenadas físicas (centímetros).

## 🚀 Propósito del Código

- Entrenar una red de Hopfield con un patrón limpio (imagen de un cilindro).
- Introducir ruido al patrón para simular condiciones reales de distorsión.
- Utilizar la red para recuperar el patrón original a través de iteraciones.
- Visualizar la evolución del patrón en cada iteración.
- Mostrar la posición del punto de perforación en centímetros.

## 🧩 Características del Modelo

- Red completamente conectada y simétrica.
- Capacidad de memoria asociativa para recuperar patrones conocidos.
- Actualización asincrónica de neuronas para simular evolución natural.
- Visualización paso a paso del proceso de convergencia.

## 📈 Aplicación

Este enfoque es útil en sistemas de visión artificial para tareas como:
- Reconocimiento de formas en entornos industriales.
- Corrección de imágenes dañadas o incompletas.
- Localización precisa de puntos de interés en patrones conocidos.

## ⚠️ Limitaciones

- Capacidad limitada de almacenamiento de patrones.
- Riesgo de convergencia a mínimos locales si el ruido es muy alto.
- No apto para imágenes de alta resolución sin modificaciones.

## 📌 Requisitos

- Python 3.x
- `numpy`
- `matplotlib`


