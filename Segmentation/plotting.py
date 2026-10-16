import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from pathlib import Path
import re
import nd2
import tifffile as tiff
import torch
from PIL import Image
from anhs_watershed import Background

# --------------------------------------------------
# CONFIGURATION & PATHS
# --------------------------------------------------
with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

data = {}

background = Background(darkframe_folder= str(Path.cwd()) + "/Data")
datapath = list(Path(str(Path.cwd()) + content[1]).glob("*.nd2"))

for _, file in enumerate(datapath):
    raw = nd2.imread(file).astype(np.float32)
    new_img = background.darkfield_assignment(str(file))[0]["raw_minus_dark"]
    
    data[str(file).split("/")[-1].split(".nd2")[0]] = new_img

channels = [0, 0]  # Grayscale / single channel

# Hauptordner für die Ergebnisse
base_output_dir = Path.cwd() / "cellposeresult/cyto3_500epoche"
base_output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DEVICE SELECTION & MODEL LOADING
# --------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    use_gpu = True
else:
    device = torch.device("cpu")
    use_gpu = False

print("Device:", device)

MODEL_PATH = str(Path.cwd()) + "/cellpose_models/models/cellpose_cyto3_500Epoche_1791718660.745401"

model = models.CellposeModel(
                    gpu=use_gpu,
                    device=device,
                    pretrained_model=MODEL_PATH)

print("Model class:", type(model))
print("GPU enabled:", use_gpu)

# --------------------------------------------------
# ITERATE OVER ALL RECORDINGS & FRAMES
# --------------------------------------------------
for keys, recording in data.items():
    print(f"\nVerarbeite Recording: {keys} (Gesamt Frames: {len(recording)})")
    
    # Erstelle einen eigenen Ordner für dieses Recording
    rec_output_dir = base_output_dir / keys
    rec_output_dir.mkdir(parents=True, exist_ok=True)
    
    gif_frames = []
    stack_list = []
    
    # Gehe alle Frames des Recordings durch
    for index in range(len(recording)):
        # 1. Cellpose Vorhersage (Liefert die Maske als Label-Array)
        masks, flows, styles = model.eval(recording[index], channels=channels)
        
        # Konvertiere die Maske in uint16 (Standard für Cellpose-Instanzmasken)
        mask_uint16 = masks.astype(np.uint16)
        stack_list.append(mask_uint16)
        
        # Speichere die Einzelmaske als .tif im Ordner ab
        tif_filename = rec_output_dir / f"frame_{index:04d}_instances.tif"
        tiff.imwrite(tif_filename, mask_uint16, compression="zlib")
        
        # 2. Erstelle das GIF-Bild *nur* aus der gelabelten Maske (ohne Zellbild im Hintergrund)
        # 0-Werte (Hintergrund) werden transparent/schwarz dargestellt, Zellen farbig
        labels_masked = np.ma.masked_where(mask_uint16 == 0, mask_uint16)
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        
        # Wir plotten nur die Maske mit nipy_spectral auf schwarzem Hintergrund
        ax.imshow(
            labels_masked,
            cmap="nipy_spectral",
            interpolation="nearest"
        )
        
        ax.set_title(f"{keys} - Frame {index}")
        ax.axis("off")
        
        plt.tight_layout()
        
        # Plot in ein PIL-Bild umwandeln für das GIF
        fig.canvas.draw()
        rgba_image = np.asarray(fig.canvas.buffer_rgba())
        pil_img = Image.fromarray(rgba_image).convert("RGB")
        gif_frames.append(pil_img)
        
        plt.close(fig)
        
        if (index + 1) % 50 == 0 or (index + 1) == len(recording):
            print(f" -> Frame {index + 1}/{len(recording)} verarbeitet.")

    # 3. Speichere den gesamten Masken-Stack als multi-page .tif Datei ab
    multi_tif_path = rec_output_dir / f"{keys}_masks_stack.tif"
    tiff.imwrite(multi_tif_path, np.stack(stack_list, axis=0), bigtiff=True, compression="zlib")

    # 4. Speichere das animierte GIF im selben Ordner ab (nur Masken)
    if gif_frames:
        gif_path = rec_output_dir / f"{keys}_masks_animation.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,  # Millisekunden pro Frame
            loop=0         # Endlosschleife
        )
        print(f"Ordner erstellt & Daten gespeichert unter: {rec_output_dir}")

print("\nFertig! Alle Masken (.tif) und reinen Masken-GIFs wurden erfolgreich generiert.")