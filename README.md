# Evaluación de filtros digitales sobre una señal ECG sintética

Proyecto didáctico en Python que genera una señal de electrocardiograma sintética, agrega deriva de línea base, interferencia de 60 Hz y ruido blanco, y compara tres filtros digitales: pasa bajos Butterworth, pasa altos Chebyshev II y pasa bandas FIR con ventana Kaiser.

## Requisitos

Python 3.10 o posterior y las bibliotecas listadas en `requirements.txt`.

## Ejecución

```bash
python -m pip install -r requirements.txt
python analisis_ecg.py
```

El programa guarda cuatro gráficas en `figuras/` y las métricas en `resultados/`. La semilla aleatoria está fijada, por lo que la señal y las cifras se pueden reproducir.

## Contenido

- `analisis_ecg.py`: generación de la señal, diseño, filtrado, medición y visualización.
- `figuras/`: entrada, respuesta en frecuencia, trazas de salida y espectros.
- `resultados/`: tabla CSV y resumen JSON.

La forma de onda es simulada con fines de aprendizaje; no debe interpretarse como señal clínica ni utilizarse para diagnóstico.
