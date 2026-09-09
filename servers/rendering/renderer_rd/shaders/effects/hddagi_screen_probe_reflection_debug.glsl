#[vertex]

#version 450

#VERSION_DEFINES

#ifdef USE_MULTIVIEW
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
layout(location = 0) out vec3 uv_interp;
#else
layout(location = 0) out vec2 uv_interp;
#endif

void main() {
	vec2 position = vec2(-1.0, -1.0);
	if (gl_VertexIndex == 1) {
		position = vec2(-1.0, 3.0);
	} else if (gl_VertexIndex == 2) {
		position = vec2(3.0, -1.0);
	}
	gl_Position = vec4(position, 0.0, 1.0);
	uv_interp.xy = position * 0.5 + 0.5;
#ifdef USE_MULTIVIEW
	uv_interp.z = float(ViewIndex);
#endif
}

#[fragment]

#version 450

#VERSION_DEFINES

const uint DEBUG_FLAG_REFLECTION_VALID = 1u << 0u;
const uint DEBUG_FLAG_GUIDES_VALID = 1u << 1u;
const uint DEBUG_FLAG_HISTORY_VIEW_0_VALID = 1u << 2u;
const uint DEBUG_FLAG_HISTORY_VIEW_1_VALID = 1u << 3u;

const float HISTORY_LENGTH_STORAGE_SCALE = 255.0;
const float SPECULAR_MAX_HISTORY_LENGTH = 64.0;
const float SPECULAR_MOMENT_LUMINANCE_SCALE = 16.0;
const float SPECULAR_DENOISER_INV_TONEMAP_RANGE = 0.1;
const float SPECULAR_DENOISER_MAX_LUMINANCE = 8.0;

#ifdef USE_MULTIVIEW
layout(location = 0) in vec3 uv_interp;
layout(set = 0, binding = 0) uniform sampler2DArray raw_signal;
layout(set = 0, binding = 1) uniform sampler2DArray denoised_signal;
layout(set = 0, binding = 2) uniform sampler2DArray current_normal_roughness;
layout(set = 0, binding = 3) uniform sampler2DArray current_motion;
#else
layout(location = 0) in vec2 uv_interp;
layout(set = 0, binding = 0) uniform sampler2D raw_signal;
layout(set = 0, binding = 1) uniform sampler2D denoised_signal;
layout(set = 0, binding = 2) uniform sampler2D current_normal_roughness;
layout(set = 0, binding = 3) uniform sampler2D current_motion;
#endif

layout(set = 0, binding = 4) uniform sampler2D temporal_signal_0;
layout(set = 0, binding = 5) uniform sampler2D temporal_signal_1;
layout(set = 0, binding = 6) uniform sampler2D moments_0;
layout(set = 0, binding = 7) uniform sampler2D moments_1;
layout(set = 0, binding = 8) uniform sampler2D history_normal_roughness_0;
layout(set = 0, binding = 9) uniform sampler2D history_normal_roughness_1;

layout(location = 0) out vec4 frag_color;

layout(push_constant, std430) uniform Params {
	vec2 resolution;
	uint flags;
	uint denoised_view_mask;
}
params;

uint debug_view_index() {
#ifdef USE_MULTIVIEW
	return uint(uv_interp.z + 0.5);
#else
	return 0u;
#endif
}

bool history_valid() {
	uint flag = debug_view_index() == 0u ? DEBUG_FLAG_HISTORY_VIEW_0_VALID : DEBUG_FLAG_HISTORY_VIEW_1_VALID;
	return (params.flags & flag) != 0u;
}

bool denoised_valid() {
	return (params.denoised_view_mask & (1u << debug_view_index())) != 0u;
}

vec4 sample_raw(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(raw_signal, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(raw_signal, uv, 0.0);
#endif
}

vec4 sample_denoised(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(denoised_signal, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(denoised_signal, uv, 0.0);
#endif
}

vec4 sample_guide(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(current_normal_roughness, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(current_normal_roughness, uv, 0.0);
#endif
}

vec4 sample_motion(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(current_motion, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(current_motion, uv, 0.0);
#endif
}

vec4 sample_temporal(vec2 uv) {
	return debug_view_index() == 0u ? textureLod(temporal_signal_0, uv, 0.0) : textureLod(temporal_signal_1, uv, 0.0);
}

vec2 sample_moments(vec2 uv) {
	return debug_view_index() == 0u ? textureLod(moments_0, uv, 0.0).xy : textureLod(moments_1, uv, 0.0).xy;
}

vec4 sample_history_normal_roughness(vec2 uv) {
	return debug_view_index() == 0u ? textureLod(history_normal_roughness_0, uv, 0.0) : textureLod(history_normal_roughness_1, uv, 0.0);
}

bool finite_vec2(vec2 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool finite_vec3(vec3 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

bool finite_vec4(vec4 value) {
	return !any(isnan(value)) && !any(isinf(value));
}

float luminance(vec3 value) {
	return max(dot(max(value, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722)), 0.0);
}

vec3 project_denoiser_space(vec3 value) {
	value = max(value, vec3(0.0));
	float signal_luminance = luminance(value);
	if (signal_luminance > SPECULAR_DENOISER_MAX_LUMINANCE) {
		value *= SPECULAR_DENOISER_MAX_LUMINANCE / signal_luminance;
	}
	return value;
}

vec3 from_denoiser_space(vec3 value) {
	value = project_denoiser_space(value);
	float inverse_weight = 1.0 / max(1.0 - luminance(value) * SPECULAR_DENOISER_INV_TONEMAP_RANGE, 1e-3);
	return value * inverse_weight;
}

vec3 unavailable_color(vec2 pixel) {
	float checker = mod(floor(pixel.x / 12.0) + floor(pixel.y / 12.0), 2.0);
	return mix(vec3(0.055), vec3(0.13), checker);
}

vec3 debug_tonemap(vec3 radiance) {
	if (!finite_vec3(radiance)) {
		return vec3(1.0, 0.0, 1.0);
	}
	radiance = max(radiance, vec3(0.0));
	return radiance / (vec3(1.0) + radiance);
}

vec3 heatmap(float value) {
	value = clamp(value, 0.0, 1.0);
	return vec3(
			clamp(1.5 - abs(4.0 * value - 3.0), 0.0, 1.0),
			clamp(1.5 - abs(4.0 * value - 2.0), 0.0, 1.0),
			clamp(1.5 - abs(4.0 * value - 1.0), 0.0, 1.0));
}

vec3 relative_difference_color(vec3 a, vec3 b) {
	if (!finite_vec3(a) || !finite_vec3(b)) {
		return vec3(1.0, 0.0, 1.0);
	}
	float difference = max(max(abs(a.r - b.r), abs(a.g - b.g)), abs(a.b - b.b));
	float reference = max(max(max(a.r, a.g), a.b), max(max(max(b.r, b.g), b.b), 0.1));
	return heatmap(difference / reference);
}

vec3 history_age_color(vec2 uv) {
	float age = sample_history_normal_roughness(uv).a * HISTORY_LENGTH_STORAGE_SCALE;
	if (isnan(age) || isinf(age)) {
		return vec3(1.0, 0.0, 1.0);
	}
	return heatmap(age / SPECULAR_MAX_HISTORY_LENGTH);
}

vec3 variance_color(vec2 uv) {
	vec2 moments = sample_moments(uv);
	float age = sample_history_normal_roughness(uv).a * HISTORY_LENGTH_STORAGE_SCALE;
	if (!finite_vec2(moments) || isnan(age) || isinf(age)) {
		return vec3(1.0, 0.0, 1.0);
	}
	float variance = max(moments.y - moments.x * moments.x, 0.0) * max(1.0, 4.0 / (age + 1.0));
	float standard_deviation = sqrt(variance) / SPECULAR_MOMENT_LUMINANCE_SCALE;
	float mean = max(moments.x / SPECULAR_MOMENT_LUMINANCE_SCALE, 0.1);
	return heatmap(standard_deviation / mean);
}

vec3 hit_distance_color(float hit_distance) {
	if (isnan(hit_distance) || isinf(hit_distance) || hit_distance < 0.0) {
		return vec3(1.0, 0.0, 1.0);
	}
	if (hit_distance == 0.0) {
		return vec3(0.0);
	}
	if (hit_distance >= 65000.0) {
		return vec3(0.75, 0.08, 0.95);
	}
	return heatmap(log2(1.0 + hit_distance) / 16.0);
}

vec3 source_color(vec2 uv, vec2 pixel) {
	vec4 guide = sample_guide(uv);
	vec4 motion = sample_motion(uv);
	if (!finite_vec4(guide) || !finite_vec4(motion)) {
		return vec3(1.0, 0.0, 1.0);
	}
	float packed = floor(clamp(guide.a, 0.0, 1.0) * 255.0 + 0.5);
	bool screen_source = packed >= 128.0;
	float roughness = (packed - (screen_source ? 128.0 : 0.0)) / 127.0;
	vec3 color = screen_source ? vec3(0.06, 0.92, 0.22) : vec3(0.08, 0.32, 1.0);
	color = mix(color * 0.35, color, 0.25 + roughness * 0.75);
	if (motion.w > 1.0) {
		float stripe = step(0.5, fract((pixel.x + pixel.y) / 8.0));
		color = mix(color, vec3(1.0, 0.05, 0.8), 0.35 + stripe * 0.35);
	}
	return color;
}

uint glyph_row(uint character, int row) {
	if (row < 0 || row >= 5) {
		return 0u;
	}
	if (character == 65u) {
		uint rows[5] = uint[](6u, 9u, 15u, 9u, 9u);
		return rows[row];
	}
	if (character == 68u) {
		uint rows[5] = uint[](14u, 9u, 9u, 9u, 14u);
		return rows[row];
	}
	if (character == 69u) {
		uint rows[5] = uint[](15u, 8u, 14u, 8u, 15u);
		return rows[row];
	}
	if (character == 70u) {
		uint rows[5] = uint[](15u, 8u, 14u, 8u, 8u);
		return rows[row];
	}
	if (character == 71u) {
		uint rows[5] = uint[](7u, 8u, 11u, 9u, 7u);
		return rows[row];
	}
	if (character == 72u) {
		uint rows[5] = uint[](9u, 9u, 15u, 9u, 9u);
		return rows[row];
	}
	if (character == 73u) {
		uint rows[5] = uint[](15u, 6u, 6u, 6u, 15u);
		return rows[row];
	}
	if (character == 77u) {
		uint rows[5] = uint[](9u, 15u, 15u, 9u, 9u);
		return rows[row];
	}
	if (character == 78u) {
		uint rows[5] = uint[](9u, 13u, 11u, 9u, 9u);
		return rows[row];
	}
	if (character == 80u) {
		uint rows[5] = uint[](14u, 9u, 14u, 8u, 8u);
		return rows[row];
	}
	if (character == 82u) {
		uint rows[5] = uint[](14u, 9u, 14u, 10u, 9u);
		return rows[row];
	}
	if (character == 83u) {
		uint rows[5] = uint[](7u, 8u, 6u, 1u, 14u);
		return rows[row];
	}
	if (character == 84u) {
		uint rows[5] = uint[](15u, 6u, 6u, 6u, 6u);
		return rows[row];
	}
	if (character == 86u) {
		uint rows[5] = uint[](9u, 9u, 9u, 9u, 6u);
		return rows[row];
	}
	if (character == 87u) {
		uint rows[5] = uint[](9u, 9u, 15u, 15u, 9u);
		return rows[row];
	}
	return 0u;
}

uint panel_label_character(int panel, int index) {
	if (panel == 0) {
		uint label[3] = uint[](82u, 65u, 87u);
		return label[index];
	}
	if (panel == 1) {
		uint label[3] = uint[](84u, 77u, 80u);
		return label[index];
	}
	if (panel == 2) {
		uint label[3] = uint[](70u, 73u, 78u);
		return label[index];
	}
	if (panel == 3) {
		uint label[3] = uint[](82u, 84u, 68u);
		return label[index];
	}
	if (panel == 4) {
		uint label[3] = uint[](84u, 70u, 68u);
		return label[index];
	}
	if (panel == 5) {
		uint label[3] = uint[](65u, 71u, 69u);
		return label[index];
	}
	if (panel == 6) {
		uint label[3] = uint[](86u, 65u, 82u);
		return label[index];
	}
	if (panel == 7) {
		uint label[3] = uint[](72u, 73u, 84u);
		return label[index];
	}
	uint label[3] = uint[](83u, 82u, 67u);
	return label[index];
}

bool panel_label_pixel(int panel, vec2 tile_pixel, vec2 tile_size) {
	int scale = max(int(floor(min(tile_size.x, tile_size.y) / 150.0)), 1);
	ivec2 pixel = ivec2(tile_pixel) - ivec2(6 * scale, 4 * scale);
	if (pixel.x < 0 || pixel.y < 0) {
		return false;
	}
	int character_index = pixel.x / (5 * scale);
	if (character_index < 0 || character_index >= 3) {
		return false;
	}
	int character_x = (pixel.x - character_index * 5 * scale) / scale;
	int character_y = pixel.y / scale;
	if (character_x < 0 || character_x >= 4 || character_y < 0 || character_y >= 5) {
		return false;
	}
	uint row_bits = glyph_row(panel_label_character(panel, character_index), character_y);
	return ((row_bits >> uint(3 - character_x)) & 1u) != 0u;
}

void main() {
	vec2 grid_position = uv_interp.xy * 3.0;
	ivec2 tile = clamp(ivec2(floor(grid_position)), ivec2(0), ivec2(2));
	int panel = tile.y * 3 + tile.x;
	vec2 source_uv = fract(grid_position);
	vec2 tile_size = params.resolution / 3.0;
	vec2 tile_pixel = source_uv * tile_size;
	bool reflection_available = (params.flags & DEBUG_FLAG_REFLECTION_VALID) != 0u;
	bool guides_available = (params.flags & DEBUG_FLAG_GUIDES_VALID) != 0u;
	bool temporal_available = reflection_available && history_valid();

	vec3 color = unavailable_color(tile_pixel);
	vec4 raw = reflection_available ? sample_raw(source_uv) : vec4(0.0);
	vec3 temporal = temporal_available ? from_denoiser_space(sample_temporal(source_uv).rgb) : vec3(0.0);
	vec3 final_signal = reflection_available ? (denoised_valid() ? sample_denoised(source_uv).rgb : raw.rgb) : vec3(0.0);
	if (panel == 0 && reflection_available) {
		color = debug_tonemap(raw.rgb);
	} else if (panel == 1 && temporal_available) {
		color = debug_tonemap(temporal);
	} else if (panel == 2 && reflection_available) {
		color = debug_tonemap(final_signal);
	} else if (panel == 3 && temporal_available) {
		color = relative_difference_color(raw.rgb, temporal);
	} else if (panel == 4 && temporal_available) {
		color = relative_difference_color(temporal, final_signal);
	} else if (panel == 5 && temporal_available) {
		color = history_age_color(source_uv);
	} else if (panel == 6 && temporal_available) {
		color = variance_color(source_uv);
	} else if (panel == 7 && reflection_available) {
		color = hit_distance_color(raw.a);
	} else if (panel == 8 && guides_available) {
		color = source_color(source_uv, tile_pixel);
	}

	int font_scale = max(int(floor(min(tile_size.x, tile_size.y) / 150.0)), 1);
	if (tile_pixel.y < float(font_scale * 13)) {
		vec3 label_background = panel == 2 && reflection_available && !denoised_valid() ? vec3(0.42, 0.12, 0.015) : vec3(0.12, 0.2, 0.28);
		color = mix(color, label_background, 0.82);
		if (panel_label_pixel(panel, tile_pixel, tile_size)) {
			color = vec3(1.0);
		}
	}
	float border = min(min(tile_pixel.x, tile_pixel.y), min(tile_size.x - tile_pixel.x, tile_size.y - tile_pixel.y));
	if (border < 1.5) {
		color = panel == 2 && reflection_available && !denoised_valid() ? vec3(1.0, 0.24, 0.02) : vec3(0.0);
	}
	frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
