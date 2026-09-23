#!/usr/bin/env python3
"""
Incline des coupes (NIfTI .nii/.nii.gz ou PNG/JPG) façon Fig 12 de Skibbe et al. 2023.

UTILISATION SIMPLE :
  1. Modifie les PARAMÈTRES juste en dessous (au moins DOSSIER_COUPES).
  2. Lance :  python tilted-images.py
  -> toutes les coupes du dossier sont inclinées et enregistrées dans DOSSIER_SORTIE.

(Optionnel) Tu peux aussi donner des fichiers en ligne de commande :
  python tilted-images.py coupe1.nii.gz coupe2.nii.gz --pile
"""

# ============================ PARAMÈTRES ============================
DOSSIER_COUPES = "~/Bureau/coupes"           # dossier contenant tes coupes
DOSSIER_SORTIE = "~/Bureau/coupes_inclinees" # où enregistrer les résultats

UNE_IMAGE_PAR_COUPE = True   # True : chaque coupe inclinée dans son propre fichier
                             # False : toutes les coupes empilées dans un seul schéma
SENS = "vertical"            # "vertical" (haut -> bas) ou "horizontal" (gauche -> droite)
ANGLE = -40                  # inclinaison en degrés (essaie 40 pour l'autre côté)
ECRASE = 0.3                 # 0.2 = très plat, 0.5 = peu écrasé
CMAP = "gray"                # couleurs : "gray", "magma", "hot", "viridis"...
ROT = 1                      # 0, 1, 2 ou 3 : tourne la coupe de 90° x N si elle est de travers

# seulement pour le schéma empilé (UNE_IMAGE_PAR_COUPE = False)
VIDES = 0                    # nb de coupes bleues vides entre deux coupes
PAS = 0.38                   # distance entre les coupes
COULEUR_VIDES = "#8fa9c9"
ESPACEMENT = "50 µm"         # texte de l'accolade ("" pour ne rien afficher)
# ====================================================================

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from matplotlib.patches import Polygon
from PIL import Image


def transfo_coupe(mode, ecrase, angle, dx, dy):
    """Écrase + incline une coupe unité, puis la décale."""
    t = mt.Affine2D()
    if mode == "vertical":            # coupe « couchée », pile de haut en bas
        t.scale(1.0, ecrase).skew_deg(angle, 0)
    else:                             # coupe « debout », pile de gauche à droite
        t.scale(ecrase, 1.0).skew_deg(0, angle)
    return t.translate(dx, dy)


def normaliser(a, cmap):
    """Tableau 2D ou RGB -> image PIL RGBA (contraste 1-99 percentiles)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        lo, hi = np.percentile(a[np.isfinite(a)], [1, 99])
        a = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
        rgba = plt.get_cmap(cmap)(a)
    else:  # RGB
        a = a[..., :3]
        a = a / (a.max() + 1e-12)
        rgba = np.dstack([a, np.ones(a.shape[:2])])
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


def carre_noir(img):
    """Complète l'image en carré (fond noir) pour ne pas la déformer."""
    w, h = img.size
    c = max(w, h)
    fond = Image.new("RGBA", (c, c), (0, 0, 0, 255))
    fond.paste(img, ((c - w) // 2, (c - h) // 2))
    return fond


def charger(chemin, coupes, axe, cmap, rot):
    """Renvoie une liste d'images PIL à partir d'un PNG/JPG ou d'un NIfTI."""
    if not chemin.endswith((".nii", ".nii.gz")):
        return [carre_noir(Image.open(chemin).convert("RGBA"))]

    import nibabel as nib
    data = np.asanyarray(nib.load(chemin).dataobj)
    if data.dtype.names:                       # NIfTI RGB24 (champs R, G, B)
        data = np.stack([data[c] for c in data.dtype.names[:3]], axis=-1)
    data = np.squeeze(data)
    rgb = data.shape[-1] in (3, 4) and data.ndim >= 3

    if data.ndim == 2 or (rgb and data.ndim == 3):
        tranches = [data]                      # déjà une coupe 2D
    else:                                      # volume 3D
        n = data.shape[axe]
        idx = coupes if coupes else [n // 4, n // 2, 3 * n // 4]
        tranches = [np.take(data, i, axis=axe) for i in idx]

    images = []
    for t in tranches:
        t = np.rot90(t, k=rot)                 # NIfTI (x, y) -> affichage
        images.append(carre_noir(normaliser(t, cmap)))
    return images


def deformer_image(img, t, px_par_unite=800):
    """Applique la transfo affine t à l'image (fond transparent).
    Renvoie l'image déformée et son extent [xmin, xmax, ymin, ymax]."""
    W, H = img.size
    coins = t.transform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    xmin, ymin = coins.min(0)
    xmax, ymax = coins.max(0)
    ow = int((xmax - xmin) * px_par_unite)
    oh = int((ymax - ymin) * px_par_unite)
    inv = t.inverted()

    def source(u, v):  # pixel de sortie -> pixel source
        x = xmin + u / px_par_unite
        y = ymax - v / px_par_unite
        lx, ly = inv.transform([[x, y]])[0]
        return lx * W, (1 - ly) * H

    p00, p10, p01 = source(0, 0), source(1, 0), source(0, 1)
    coef = (p10[0] - p00[0], p01[0] - p00[0], p00[0],
            p10[1] - p00[1], p01[1] - p00[1], p00[1])
    out = img.transform((ow, oh), Image.AFFINE, coef,
                        resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    return np.asarray(out), [xmin, xmax, ymin, ymax]


def dessiner(imgs, a, sortie):
    """Dessine les images inclinées (empilées si plusieurs) et enregistre."""
    fig, ax = plt.subplots(figsize=(6, 6))
    carre = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    seq = []
    for i, im in enumerate(imgs):
        seq.append(im)
        if i < len(imgs) - 1:
            seq += [None] * a.vides

    positions = []
    for k, item in enumerate(seq):
        dx, dy = (0, -k * a.pas) if a.mode == "vertical" else (k * a.pas, 0)
        t = transfo_coupe(a.mode, a.ecrase, a.angle, dx, dy)
        z = len(seq) - k if a.mode == "vertical" else k
        if item is None:
            ax.add_patch(Polygon(t.transform(carre), closed=True, facecolor=a.couleur,
                                 edgecolor="#4a6a90", lw=0.6, zorder=z))
        else:
            warped, ext = deformer_image(item, t)
            ax.imshow(warped, extent=ext, origin="upper",
                      interpolation="bilinear", zorder=z)
            ax.add_patch(Polygon(t.transform(carre), closed=True, fill=False,
                                 edgecolor="#2f5f9e", lw=1.2, zorder=z))
        positions.append(t.transform(carre))

    # accolade entre les deux premières coupes
    if len(seq) > 1 and a.espacement:
        p0, p1 = positions[0], positions[1]
        if a.mode == "vertical":
            x = max(p0[:, 0].max(), p1[:, 0].max()) + 0.05
            y0, y1 = p1[:, 1].min(), p0[:, 1].min()
            ax.plot([x, x + 0.03, x + 0.03, x], [y1, y1, y0, y0], color="k", lw=1)
            ax.text(x + 0.06, (y0 + y1) / 2, a.espacement, va="center", fontsize=13,
                    family="serif", style="italic")
        else:
            y = min(p0[:, 1].min(), p1[:, 1].min()) - 0.05
            x0, x1 = p0[:, 0].min(), p1[:, 0].min()
            ax.plot([x0, x0, x1, x1], [y, y - 0.03, y - 0.03, y], color="k", lw=1)
            ax.text((x0 + x1) / 2, y - 0.07, a.espacement, ha="center", va="top",
                    fontsize=13, family="serif", style="italic")

    ax.set_aspect("equal")
    pts = np.vstack(positions)
    m = 0.1
    ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m + 0.3 * (len(seq) > 1))
    ax.set_ylim(pts[:, 1].min() - m - 0.2 * (len(seq) > 1), pts[:, 1].max() + m)
    ax.axis("off")
    fig.savefig(sortie + ".svg", bbox_inches="tight", transparent=True)
    fig.savefig(sortie + ".png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("  ->", sortie + ".png  (+ .svg)")


def nom_sans_ext(chemin):
    n = os.path.basename(chemin)
    for ext in (".nii.gz", ".nii", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return n


def main():
    p = argparse.ArgumentParser(description="Incline des coupes (voir PARAMÈTRES en haut du fichier).")
    p.add_argument("images", nargs="*", help="(optionnel) fichiers de coupes ; sinon DOSSIER_COUPES")
    p.add_argument("--pile", action="store_true", help="empiler toutes les coupes dans un seul schéma")
    p.add_argument("--coupes", type=int, nargs="+", help="indices à extraire si volume 3D")
    p.add_argument("--axe", type=int, default=2, choices=[0, 1, 2])
    a = p.parse_args()

    # valeurs venant du bloc PARAMÈTRES
    a.mode, a.angle, a.ecrase, a.cmap, a.rot = SENS, ANGLE, ECRASE, CMAP, ROT
    a.vides, a.pas, a.couleur, a.espacement = VIDES, PAS, COULEUR_VIDES, ESPACEMENT

    if a.images:
        fichiers = a.images
    else:
        dossier = os.path.expanduser(DOSSIER_COUPES)
        fichiers = []
        for ext in ("*.nii.gz", "*.nii", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            fichiers += glob.glob(os.path.join(dossier, ext))
        fichiers = sorted(set(fichiers))
        if not fichiers:
            raise SystemExit(f"Aucune coupe trouvée dans {dossier} — vérifie DOSSIER_COUPES.")

    sortie = os.path.expanduser(DOSSIER_SORTIE)
    os.makedirs(sortie, exist_ok=True)
    print(f"{len(fichiers)} fichier(s) trouvé(s). Résultats dans : {sortie}")

    if UNE_IMAGE_PAR_COUPE and not a.pile:
        for f in fichiers:
            for j, im in enumerate(charger(f, a.coupes, a.axe, a.cmap, a.rot)):
                suffixe = f"_{j}" if j else ""
                dessiner([im], a, os.path.join(sortie, "incline_" + nom_sans_ext(f) + suffixe))
    else:
        imgs = []
        for f in fichiers:
            imgs += charger(f, a.coupes, a.axe, a.cmap, a.rot)
        dessiner(imgs, a, os.path.join(sortie, "coupes_empilees"))
    print("Terminé.")


if __name__ == "__main__":
    main()