"""
Streamlit app that renders an interactive 3D "Hello World" text using Plotly.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font if available, otherwise fall back to default."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_text_volume(
    text: str, *, depth: int = 6, font_size: int = 120, padding: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple voxel volume representing the supplied text.

    The text is rendered to a grayscale mask with Pillow, then extruded along the
    Z-axis to produce a 3D volume suitable for interactive Plotly visualization.
    """

    font = _load_font(font_size)
    # Determine required canvas size for the text.
    dummy_img = Image.new("L", (1, 1), color=0)
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    width = text_bbox[2] - text_bbox[0] + padding * 2
    height = text_bbox[3] - text_bbox[1] + padding * 2

    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    draw.text((padding, padding), text, fill=255, font=font)

    mask = np.array(image) > 0
    volume = np.repeat(mask[np.newaxis, :, :], depth, axis=0)

    z_idx, y_idx, x_idx = np.indices(volume.shape)
    values = volume.astype(int)
    filled = values.flatten() > 0

    return (
        x_idx.flatten()[filled],
        # Flip Y so the text reads upright in the Plotly coordinate system.
        (height - y_idx.flatten()[filled]).astype(float),
        z_idx.flatten()[filled].astype(float),
        values.flatten()[filled].astype(float),
    )


def build_figure(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, values: np.ndarray, *, text: str
) -> go.Figure:
    """Create a Plotly figure for the 3D voxel text."""

    fig = go.Figure(
        data=[
            go.Volume(
                x=x,
                y=y,
                z=z,
                value=values,
                colorscale="Viridis",
                opacity=0.15,
                surface_count=16,
                showscale=False,
            )
        ]
    )
    fig.update_layout(
        title=f"3D Text: {text}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth",
            aspectmode="data",
            bgcolor="rgba(240,240,240,0.8)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main() -> None:
    """Streamlit entry point."""

    st.set_page_config(page_title="3D Hello World", page_icon="👋", layout="wide")
    st.title("Interactive 3D Text")
    st.write(
        "Use the controls to change how the 'Hello World' text is extruded into a 3D"
        " voxel volume."
    )

    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Extrusion depth (Z layers)", min_value=2, max_value=20, value=8)
    with col2:
        font_size = st.slider("Font size", min_value=60, max_value=200, value=130)

    x, y, z, values = build_text_volume(
        "Hello World", depth=depth, font_size=font_size, padding=20
    )
    figure = build_figure(x, y, z, values, text="Hello World")

    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": True})
    st.caption(
        "The visualization is driven by a simple voxel grid generated from a Pillow"
        " text mask and rendered with Plotly's interactive volume trace."
    )


if __name__ == "__main__":
    main()
