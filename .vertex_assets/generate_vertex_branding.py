#!/usr/bin/env python3
"""Generate Vertex Game Engine branding assets (splash screens, app icon, V logo).

Replaces the user-facing Godot boot splash and app icon with a clean Vertex
identity: a large centered "V" on a dark background. Does NOT touch internal
class names or third-party assets.
"""
import math
import os
from PIL import Image, ImageDraw, ImageFilter

# Brand palette
V_COLOR = (108, 196, 255, 255)        # vibrant cyan-blue accent
V_COLOR_EDITOR = (120, 200, 255, 255) # slightly brighter on editor bg
BG_BOOT = (36, 36, 39, 255)          # neutral dark (matches boot_splash_bg_color 0.14)
BG_EDITOR = (32, 37, 49, 255)        # editor theme bg (matches 0.125,0.145,0.192)
WHITE = (235, 238, 244, 255)


def draw_v(draw, cx, cy, size, color, thickness_ratio=0.14):
    """Draw a clean, modern 'V' as two filled strokes meeting at a point."""
    h = size
    w = size * 0.78
    left_top = (cx - w / 2, cy - h / 2)
    right_top = (cx + w / 2, cy - h / 2)
    bottom = (cx, cy + h / 2)
    t = max(2.0, h * thickness_ratio)

    # Left stroke: trapezoid from top-left down to bottom point.
    # Compute perpendicular offsets for stroke thickness.
    def offset(p1, p2, dist):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L == 0:
            return p1
        nx, ny = -dy / L, dx / L
        return (p1[0] + nx * dist, p1[1] + ny * dist)

    # Left stroke outer/inner edges
    lo = offset(left_top, bottom, t / 2)
    li = offset(left_top, bottom, -t / 2)
    ro = offset(right_top, bottom, -t / 2)
    ri = offset(right_top, bottom, t / 2)

    # Left bar: quad (left_top_outer-ish). Build polygon for left stroke.
    left_poly = [
        offset(left_top, bottom, t / 2),
        offset(left_top, bottom, -t / 2),
        (bottom[0], bottom[1]),
    ]
    # Simpler: draw each stroke as a thick line (polygon) from top to bottom.
    draw.line([left_top, bottom], fill=color, width=int(t), joint="curve")
    draw.line([right_top, bottom], fill=color, width=int(t), joint="curve")


def make_splash(path, width, height, bg, v_color, with_text=False):
    img = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(img)
    cx, cy = width / 2, height / 2
    v_size = min(width, height) * 0.42

    # Soft glow behind the V
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    draw_v(gdraw, cx, cy, v_size, (*v_color[:3], 120))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=min(width, height) * 0.06))
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # The crisp V
    draw_v(draw, cx, cy, v_size, v_color)

    if with_text:
        # "Vertex Game Development" below the V
        from PIL import ImageFont
        txt = "Vertex Game Development"
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                int(height * 0.05),
            )
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (width - tw) / 2
        ty = cy + v_size / 2 + int(height * 0.04)
        draw.text((tx, ty), txt, fill=WHITE, font=font)

    img.save(path)
    print("wrote", path, img.size)


def make_icon(path, size, bg, v_color):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # rounded square background
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    r = int(size * 0.18)
    mdraw.rounded_rectangle([2, 2, size - 2, size - 2], radius=r, fill=255)
    bg_layer = Image.new("RGBA", (size, size), bg)
    img.paste(bg_layer, (0, 0), mask)
    draw = ImageDraw.Draw(img)
    draw_v(draw, size / 2, size / 2, size * 0.58, v_color)
    img.save(path)
    print("wrote", path, img.size)


def make_logo_svg(path):
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <rect width="256" height="256" rx="48" fill="{ '#202531' }"/>
  <g fill="none" stroke="{ '#78C8FF' }" stroke-width="26" stroke-linecap="round" stroke-linejoin="round">
    <path d="M78 64 L128 188"/>
    <path d="M178 64 L128 188"/>
  </g>
</svg>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", path)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main = os.path.join(root, "main")
    os.makedirs(os.path.join(root, ".vertex_assets"), exist_ok=True)
    make_splash(os.path.join(main, "splash.png"), 800, 600, BG_BOOT, V_COLOR, with_text=True)
    make_splash(os.path.join(main, "splash_editor.png"), 800, 600, BG_EDITOR, V_COLOR_EDITOR, with_text=False)
    make_icon(os.path.join(main, "app_icon.png"), 128, (32, 37, 49, 255), V_COLOR_EDITOR)
    make_logo_svg(os.path.join(root, ".vertex_assets", "vertex_logo.svg"))
    print("done")
