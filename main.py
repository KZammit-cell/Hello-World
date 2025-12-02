"""
Streamlit app that renders an interactive 3D "Hello World" text using Plotly.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import plotly
from PIL import Image, ImageDraw, ImageFont




import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go


def text_to_voxel_volume(
    text="Hello World",
    img_width=400,
    img_height=120,
    font_size=80,
    depth=10,
):
    """
    Render text into a 2D bitmap (Pillow), then extrude into a 3D voxel volume.
    Returns a 3D numpy array of shape (z, y, x) with values 0 or 1.
    """
    # Create blank grayscale image
    img = Image.new("L", (img_width, img_height), color=0)
    draw = ImageDraw.Draw(img)

    # Try to load a truetype font; fall back to default if not found
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Compute text bounding box and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0 = (img_width - text_w) // 2
    y0 = (img_height - text_h) // 2

    # Draw white text on black background
    draw.text((x0, y0), text, fill=255, font=font)

    # Convert to numpy and threshold
    arr_2d = np.array(img)
    mask = arr_2d > 0  # boolean mask where text is present

    # Stack along z to create a volume (z, y, x)
    volume = np.stack([mask.astype(np.float32)] * depth, axis=0)
    return volume


def create_3d_text_figure(
    text="Hello World",
    font_size=80,
    depth=10,
    img_width=400,
    img_height=120,
):
    """
    Build a Plotly Volume figure from text voxels.
    """
    volume = text_to_voxel_volume(
        text=text,
        img_width=img_width,
        img_height=img_height,
        font_size=font_size,
        depth=depth,
    )
    z_dim, y_dim, x_dim = volume.shape

    # Coordinate grid
    z, y, x = np.mgrid[0:z_dim, 0:y_dim, 0:x_dim]

    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=0.5,        # show only voxels with value ~1
            isomax=1.0,
            opacity=0.15,      # overall transparency
            surface_count=8,   # number of isosurfaces
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        title=f'3D Voxel Text: "{text}"',
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


if __name__ == "__main__":
    fig = create_3d_text_figure(
        text="Hello World",
        font_size=80,
        depth=10,
    )
    fig.show()
