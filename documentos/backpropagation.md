# Backpropagation explicado visualmente

## El problema que resuelve backpropagation

### Lección 1.1.1 — Por qué ajustar pesos es difícil
Cuando entrenamos una red neuronal, cada neurona tiene pesos que influyen en la salida.
El problema es que un pequeño cambio en un peso puede afectar a muchas capas posteriores.
Esto hace que ajustar los pesos manualmente sea inviable en redes grandes.

### Cómo se calculan los gradientes
Para saber cómo modificar los pesos, necesitamos calcular gradientes.
Los gradientes indican en qué dirección y cuánto cambiar cada peso para reducir el error.
Aquí entra en juego el cálculo diferencial y la regla de la cadena.

## Cómo funciona el algoritmo

### Flujo hacia delante y hacia atrás
El algoritmo de backpropagation se divide en dos fases:
- forward pass: se calcula la salida
- backward pass: se propaga el error hacia atrás
Durante el backward pass se calculan los gradientes de cada peso.

### Ejemplo simple de backpropagation
En una red con una sola neurona, el gradiente se puede calcular de forma directa.
En redes profundas, el proceso se repite capa a capa.
