// Painterly resolve compute shader used for regression compilation checks.
// Applies stylised tone mapping and stroke simulation using configurable macros.

#[compute]

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D input_image;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D output_image;

layout(push_constant, std430) uniform PainterlyParams {
    float time;
    float exposure;
    float blend_factor;
    float pad;
} params;

const vec2 GOLDEN_RATIO = vec2(1.61803398875, 0.61803398875);

float hash(vec2 uv) {
    uv = fract(uv * GOLDEN_RATIO);
    uv += dot(uv, uv + 19.19);
    return fract(uv.x * uv.y);
}

float layered_noise(vec2 uv, int octaves) {
    float amplitude = 0.5;
    float frequency = 1.0;
    float value = 0.0;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * hash(uv * frequency);
        frequency *= 2.03;
        amplitude *= 0.65;
    }
    return value;
}

vec3 apply_palette(vec3 color) {
#ifdef PAINTERLY_STYLE_WATERCOLOR
    color = pow(color, vec3(1.25));
    color += vec3(0.02, 0.05, 0.07);
#elif defined(PAINTERLY_STYLE_INK)
    float luminance = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luminance), color, 0.25);
#elif defined(PAINTERLY_STYLE_BRUSH)
    color = mix(color, vec3(1.05, 0.95, 0.82), 0.15);
    color += vec3(0.03, 0.02, 0.01);
#elif defined(PAINTERLY_STYLE_CHARCOAL)
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = vec3(lum) * vec3(0.8, 0.82, 0.86);
#else
    color = mix(color, vec3(0.93, 0.85, 0.78), 0.08);
#endif
    return color;
}

vec3 apply_density_response(vec3 color, float alpha) {
#ifdef DENSE_SPLATS
    color *= 1.15 + alpha * 0.1;
    alpha = clamp(alpha * 1.2, 0.0, 1.0);
#elif defined(SPARSE_SPLATS)
    color *= 0.95 + alpha * 0.05;
    alpha *= 0.85;
#else
    color *= 1.0 + alpha * 0.08;
#endif
    return color;
}

vec4 evaluate_painterly(vec2 uv, vec4 base_sample) {
    vec4 result = base_sample;
    result.rgb = apply_palette(result.rgb);
    result.rgb = apply_density_response(result.rgb, result.a);

#ifdef ENABLE_TRANSPARENCY_FALLOFF
    result.a = pow(result.a, 1.4);
#endif

#ifdef ENABLE_NOISE_LAYERS
    int octaves = 3;
#ifdef STROKE_VARIATION
    octaves = max(1, STROKE_VARIATION);
#endif
    float noise = layered_noise(uv + params.time, octaves);
    result.rgb += noise * 0.08;
    result.a = clamp(result.a + noise * 0.05, 0.0, 1.0);
#endif

#ifdef ENABLE_PAPER_GRAIN
    float grain = hash(uv * 4.0 + params.time);
    result.rgb -= grain * 0.04;
#endif

#ifdef ENABLE_SOFT_GLOW
    float glow = layered_noise(uv * 0.25 + params.time * 0.5, 2);
    result.rgb += glow * 0.12 * result.a;
#endif

#ifdef ENABLE_TEMPORAL_BLEND
    float ping = sin(params.time * 6.28318) * 0.5 + 0.5;
    result.rgb = mix(result.rgb, vec3(ping), params.blend_factor * 0.1);
    result.a = mix(result.a, 1.0, params.blend_factor * 0.05);
#endif

#ifdef ENABLE_OUTLINE
    vec2 texel = 1.0 / vec2(imageSize(input_image));
    float outline = 0.0;
    outline += hash(uv + texel.xy);
    outline += hash(uv - texel.xy);
    outline += hash(uv + vec2(texel.x, -texel.y));
    outline += hash(uv + vec2(-texel.x, texel.y));
    outline *= 0.125;
    result.rgb = mix(result.rgb, vec3(0.05, 0.07, 0.09), outline * result.a);
#endif

    result.rgb = clamp(result.rgb, 0.0, 1.0);
    result.a = clamp(result.a, 0.0, 1.0);
    return result;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);
    if (coord.x >= size.x || coord.y >= size.y) {
        return;
    }

    vec4 sample = imageLoad(input_image, coord);
    vec2 uv = (vec2(coord) + 0.5) / vec2(size);

    vec4 painterly = evaluate_painterly(uv, sample);
    painterly.rgb *= params.exposure;

    imageStore(output_image, coord, painterly);
}
