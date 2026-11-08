"""Diseña y evalúa filtros digitales sobre un electrocardiograma sintético con interferencias.

La señal es didáctica y no corresponde a una medición clínica. Ejecuta desde la raíz
el comando `python analisis_ecg.py` para regenerar gráficas y tablas.
"""
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figuras"
OUT = ROOT / "resultados"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Registro reproducible: frecuencia de muestreo, duración y semilla se fijan aquí.
FS = 500.0
DURATION = 8.0
SEED = 20261106
rng = np.random.default_rng(SEED)
t = np.arange(0, DURATION, 1 / FS)


def gaussian(x, center, width, amplitude):
    """Pulso gaussiano para formar las ondas P, Q, R, S y T del latido."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def make_ecg(time):
    """Construye ciclos periódicos de ECG de 75 latidos por minuto."""
    period = 0.8
    phase = np.mod(time, period)
    ecg = np.zeros_like(time)
    for center, width, amp in [
        (0.18, 0.035, 0.12),   # onda P
        (0.365, 0.012, -0.16), # onda Q
        (0.400, 0.010, 1.00),  # complejo R
        (0.432, 0.014, -0.24), # onda S
        (0.610, 0.070, 0.30),  # onda T
    ]:
        ecg += gaussian(phase, center, width, amp)
    return ecg


# Separación entre referencia limpia y perturbaciones para medir la recuperación.
clean = make_ecg(t)
baseline = 0.24 * np.sin(2 * np.pi * 0.25 * t + 0.3)
line_noise = 0.11 * np.sin(2 * np.pi * 60 * t)
white_noise = rng.normal(0, 0.075, size=t.size)
noisy = clean + baseline + line_noise + white_noise

# Tres familias: suavizado de altas frecuencias, eliminación de deriva y banda útil ECG.
sos_lp = signal.butter(5, 40.0, btype="lowpass", fs=FS, output="sos")
sos_hp = signal.cheby2(5, 30, 0.7, btype="highpass", fs=FS, output="sos")
taps_bp = signal.firwin(
    251, [0.7, 40.0], pass_zero=False, fs=FS, window=("kaiser", 7.5)
)
filtered = {
    "Pasa bajos Butterworth": signal.sosfiltfilt(sos_lp, noisy),
    "Pasa altos Chebyshev II": signal.sosfiltfilt(sos_hp, noisy),
    "Pasa bandas FIR Kaiser": signal.filtfilt(taps_bp, [1.0], noisy),
}
filters = {
    "Pasa bajos Butterworth": ("IIR Butterworth", sos_lp),
    "Pasa altos Chebyshev II": ("IIR Chebyshev II", sos_hp),
    "Pasa bandas FIR Kaiser": ("FIR Kaiser", taps_bp),
}

# Evita los extremos del registro, donde el relleno del filtrado bidireccional influye.
valid = (t >= 0.6) & (t <= DURATION - 0.6)

def snr_db(reference, estimate):
    err = reference - estimate
    return float(10 * np.log10(np.mean(reference**2) / np.mean(err**2)))

metrics = {
    "fs_hz": FS,
    "duracion_s": DURATION,
    "semilla": SEED,
    "rms_entrada": float(np.sqrt(np.mean(noisy[valid] ** 2))),
    "snr_entrada_db": snr_db(clean[valid], noisy[valid]),
    "filtros": {},
}
for name, y in filtered.items():
    metrics["filtros"][name] = {
        "rms_salida": float(np.sqrt(np.mean(y[valid] ** 2))),
        "snr_salida_db": snr_db(clean[valid], y[valid]),
        "mejora_snr_db": snr_db(clean[valid], y[valid]) - snr_db(clean[valid], noisy[valid]),
        "rmse": float(np.sqrt(np.mean((clean[valid] - y[valid]) ** 2))),
    }

# Figura de entrada: una ventana de dos segundos y densidad espectral de potencia.
fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
view = t <= 2.0
axes[0].plot(t[view], clean[view], color="#202B35", lw=1.2, label="ECG de referencia")
axes[0].plot(t[view], noisy[view], color="#D66B4F", lw=0.7, alpha=0.82, label="Con interferencia")
axes[0].set(title="Registro ECG sintético", xlabel="Tiempo (s)", ylabel="Amplitud (u.a.)")
axes[0].legend(frameon=False, ncol=2)
axes[0].grid(alpha=0.22)
f, pxx = signal.welch(noisy, fs=FS, nperseg=1024)
axes[1].plot(f, 10 * np.log10(pxx + 1e-15), color="#275D78", lw=1.0)
axes[1].set(xlim=(0, 100), title="Densidad espectral antes del filtrado", xlabel="Frecuencia (Hz)", ylabel="Potencia (dB/Hz)")
axes[1].axvline(0.25, color="#D66B4F", ls="--", lw=0.9, label="Deriva 0.25 Hz")
axes[1].axvline(60, color="#8E6C9E", ls="--", lw=0.9, label="Red 60 Hz")
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.22)
fig.savefig(FIG / "entrada_ecg.png", dpi=180)
plt.close(fig)

# Respuestas de una pasada; el filtrado hacia adelante y atrás eleva al cuadrado la magnitud.
fig, ax = plt.subplots(figsize=(10, 5.3), constrained_layout=True)
for name, (kind, coeff) in filters.items():
    if "FIR" in kind:
        freq, h = signal.freqz(coeff, worN=4096, fs=FS)
    else:
        freq, h = signal.sosfreqz(coeff, worN=4096, fs=FS)
    ax.plot(freq, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), lw=1.25, label=name)
ax.set(xlim=(0, 100), ylim=(-100, 4), title="Respuesta de magnitud de los diseños", xlabel="Frecuencia (Hz)", ylabel="Magnitud (dB)")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.savefig(FIG / "respuesta_filtros.png", dpi=180)
plt.close(fig)

# Comparación temporal en un solo latido representativo.
fig, axes = plt.subplots(4, 1, figsize=(10, 8.1), sharex=True, constrained_layout=True)
start, stop = 1.6, 2.4
sel = (t >= start) & (t <= stop)
series = [("Entrada con ruido", noisy, "#D66B4F"),
          ("Pasa bajos Butterworth", filtered["Pasa bajos Butterworth"], "#275D78"),
          ("Pasa altos Chebyshev II", filtered["Pasa altos Chebyshev II"], "#8E6C9E"),
          ("Pasa bandas FIR Kaiser", filtered["Pasa bandas FIR Kaiser"], "#3C806E")]
for ax, (label, y, color) in zip(axes, series):
    ax.plot(t[sel], y[sel], color=color, lw=1.0, label=label)
    if label != "Entrada con ruido":
        ax.plot(t[sel], clean[sel], color="#202B35", lw=0.75, alpha=0.55, ls="--", label="Referencia")
    ax.set_ylabel("Amplitud")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.2)
axes[-1].set_xlabel("Tiempo (s)")
fig.suptitle("Comparación de salidas alrededor de un latido")
fig.savefig(FIG / "comparacion_tiempo.png", dpi=180)
plt.close(fig)

# Densidad espectral de cada salida para observar las perturbaciones que sobreviven.
fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
for label, y, color in [("Entrada", noisy, "#A1A4A8"),
                        ("Pasa bajos", filtered["Pasa bajos Butterworth"], "#275D78"),
                        ("Pasa altos", filtered["Pasa altos Chebyshev II"], "#8E6C9E"),
                        ("Pasa bandas", filtered["Pasa bandas FIR Kaiser"], "#3C806E")]:
    f, p = signal.welch(y, fs=FS, nperseg=1024)
    ax.plot(f, 10 * np.log10(p + 1e-15), color=color, lw=1.0, label=label)
ax.set(xlim=(0, 100), title="Espectro antes y después del filtrado", xlabel="Frecuencia (Hz)", ylabel="Potencia (dB/Hz)")
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.22)
fig.savefig(FIG / "espectros_salida.png", dpi=180)
plt.close(fig)

# CSV con métricas listo para abrir en una hoja de cálculo.
with (OUT / "metricas_filtros.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filtro", "RMS salida", "SNR salida (dB)", "mejora SNR (dB)", "RMSE"])
    for name, m in metrics["filtros"].items():
        writer.writerow([name, f"{m['rms_salida']:.6f}", f"{m['snr_salida_db']:.3f}",
                         f"{m['mejora_snr_db']:.3f}", f"{m['rmse']:.6f}"])
(OUT / "resumen.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
