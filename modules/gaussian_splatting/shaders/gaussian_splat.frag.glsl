#[fragment]

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "includes/painterly_common.glsl"
#include "includes/painterly_features.glsl"

layout(location = 0) in vec2 uv_coord;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 conic;
layout(location = 3) in float opacity;
layout(location = 4) in vec3 painterly_view_dir;
layout(location = 5) in vec3 painterly_normal_vs;
layout(location = 6) in vec2 stylization_seed;

layout(location = 0) out vec4 frag_color;

// Fragment entry point that resolves the final Gaussian splat color.
void main() {
    float power = painterly_gaussian_power(uv_coord, conic);

    if (power > 0.0) {
        discard;
    }

    float alpha = painterly_gaussian_alpha(opacity, power);

#ifdef PAINTERLY_ENABLE_BRUSH
    float brush_modulation = painterly_apply_brush_modulation(uv_coord, stylization_seed);
    alpha *= brush_modulation;
#endif

    if (alpha < 1.0 / 255.0) {
        discard;
    }

    vec3 final_color = color.rgb;

#ifdef PAINTERLY_ENABLE_PALETTE
    final_color = painterly_apply_palette_quantization(final_color, stylization_seed);
#endif

#ifdef PAINTERLY_ENABLE_LIGHTING
    final_color = painterly_apply_stylized_lighting(final_color, painterly_normal_vs, painterly_view_dir);
#endif

    frag_color = vec4(final_color, alpha);
}
