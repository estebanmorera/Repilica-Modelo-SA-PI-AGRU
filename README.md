# Réplica Modelo SA-PI-AGRU

Versión reducida del modelo de JP. El flujo conserva cuatro scripts: configuración,
preprocesamiento, modelo y entrenamiento.

## Uso

Desde la carpeta `Scripts`:

```powershell
python preprocessing.py
python train.py
```

La evaluación genera solamente una gráfica por batería y el CSV utilizado para
dibujarla:

```text
outputs/C1_B0005_seed58/evaluation_endpoint/
├── B0005.png
├── B0005_full_cycles.csv
├── B0006.png
├── B0006_cycles.csv
├── B0007.png
└── B0007_cycles.csv
```

Cada CSV contiene únicamente `cycle`, `soh_true` y `soh_pred`. Los checkpoints
se guardan aparte dentro de `outputs/C1_B0005_seed58` para poder reanudar una
corrida interrumpida con `python train.py --resume`.
