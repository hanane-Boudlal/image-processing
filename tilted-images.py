#!/usr/bin/env python3
"""
Schéma de coupes inclinées et empilées (style Fig 12 de Skibbe et al. 2023).

Exemples :
  # à partir d'un volume NIfTI 3D : coupes 40, 80 et 120 le long de l'axe 2
  python coupes_inclinees.py volume.nii.gz --coupes 40 80 120 --axe 2 --mode horizontal

  # à partir de plusieurs coupes 2D NIfTI (ou PNG/JPG)
  python coupes_inclinees.py coupe1.nii.gz coupe2.nii.gz coupe3.nii.gz --mode vertical

  # empilées de haut en bas (comme le papier)
  python coupes_inclinees.py c1.png c2.png c3.png --mode vertical --vides 2 --espacement "50 µm"

  # empilées de gauche à droite
  python coupes_inclinees.py c1.png c2.png c3.png --mode horizontal --vides 2

Sortie : un SVG (à retoucher dans Inkscape) + un PNG 300 dpi.
"""
import argparse
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+",
                   help="coupes : .nii / .nii.gz (2D ou volume 3D) ou png/jpg")
    p.add_argument("--coupes", type=int, nargs="+",
                   help="indices des coupes à extraire d'un volume 3D")
    p.add_argument("--axe", type=int, default=2, choices=[0, 1, 2],
                   help="axe de coupe du volume (0=x, 1=y, 2=z)")
    p.add_argument("--cmap", default="gray", help="colormap (gray, magma, hot...)")
    p.add_argument("--rot", type=int, default=1,
                   help="rotation de 90° x N pour remettre la coupe à l'endroit")
    p.add_argument("--mode", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--vides", type=int, default=1,
                   help="nb de coupes vides (colorées) entre deux images")
    p.add_argument("--ecrase", type=float, default=0.3, help="écrasement (0.2-0.4)")
    p.add_argument("--angle", type=float, default=-40, help="inclinaison en degrés")
    p.add_argument("--pas", type=float, default=0.38, help="distance entre coupes")
    p.add_argument("--couleur", default="#8fa9c9", help="couleur des coupes vides")
    p.add_argument("--espacement", default="50 µm", help="texte de l'accolade")
    p.add_argument("--titre", default="", help="titre au-dessus du schéma")
    p.add_argument("--out", default="coupes_inclinees")
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(6, 6))
    carre = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # construire la séquence : image, vides, image, vides, ...
    imgs = []
    for f in a.images:
        imgs += charger(f, a.coupes, a.axe, a.cmap, a.rot)
    seq = []
    for i, im in enumerate(imgs):
        seq.append(im)
        if i < len(imgs) - 1:
            seq += [None] * a.vides

    positions = []
    for k, item in enumerate(seq):
        if a.mode == "vertical":
            dx, dy = 0, -k * a.pas            # descend
        else:
            dx, dy = k * a.pas, 0             # va vers la droite
        t = transfo_coupe(a.mode, a.ecrase, a.angle, dx, dy)
        z = len(seq) - k if a.mode == "vertical" else k   # ordre d'affichage
        if item is None:
            poly = Polygon(t.transform(carre), closed=True, facecolor=a.couleur,
                           edgecolor="#4a6a90", lw=0.6, zorder=z)
            ax.add_patch(poly)
        else:
            warped, ext = deformer_image(item, t)
            ax.imshow(warped, extent=ext, origin="upper",
                      interpolation="bilinear", zorder=z)
            ax.add_patch(Polygon(t.transform(carre), closed=True, fill=False,
                                 edgecolor="#2f5f9e", lw=1.2, zorder=z))
        positions.append(t.transform(carre))

    # accolade entre les deux premières coupes consécutives
    if len(seq) > 1:
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

    if a.titre:
        ax.set_title(a.titre, family="serif", fontsize=14)
    ax.set_aspect("equal")
    pts = np.vstack(positions)
    m = 0.25
    ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
    ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
    ax.axis("off")
    fig.savefig(a.out + ".svg", bbox_inches="tight")
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    print("Écrit :", a.out + ".svg", "et", a.out + ".png")


if __name__ == "__main__":
    main()