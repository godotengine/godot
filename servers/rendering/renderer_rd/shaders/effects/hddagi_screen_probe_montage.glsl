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

const uint MONTAGE_FLAG_SELECTED_OUTPUT_VALID = 1u << 0u;
const uint MONTAGE_FLAG_HIZ_VALID = 1u << 1u;
const uint MONTAGE_FLAG_NORMAL_ROUGHNESS_VALID = 1u << 2u;
const uint MONTAGE_FLAG_VELOCITY_VALID = 1u << 3u;
const uint MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID = 1u << 4u;
const uint MONTAGE_FLAG_DIRECTIONAL_ADAPTIVE_VALID = 1u << 5u;

const uint TRACE_DEBUG_VALID_SURFACE = 1u << 0u;
const uint TRACE_DEBUG_DETAIL_HIT = 1u << 1u;
const uint TRACE_DEBUG_DETAIL_MISS = 1u << 2u;
const uint TRACE_DEBUG_DETAIL_REJECTED = 1u << 3u;
const uint TRACE_DEBUG_SOURCE_SCREEN_HDDAGI = 1u << 4u;
const uint TRACE_DEBUG_SOURCE_HDDA_VOXEL = 1u << 5u;
const uint TRACE_DEBUG_SOURCE_SKY = 1u << 6u;
const uint TRACE_DEBUG_SCREEN_RADIANCE_FALLBACK = 1u << 7u;
const uint TRACE_DEBUG_INVALID = 1u << 8u;
const uint TRACE_DEBUG_SOURCE_SCREEN_IRRADIANCE_CACHE = 1u << 9u;
const uint TRACE_DEBUG_SOURCE_HDDA_IRRADIANCE_CACHE = 1u << 10u;

const float DIRECTIONAL_PROBE_TILE_SIZE = 8.0;

#ifdef USE_MULTIVIEW
layout(location = 0) in vec3 uv_interp;
layout(set = 0, binding = 0) uniform sampler2DArray resolved_radiance;
layout(set = 0, binding = 1) uniform sampler2DArray selected_radiance;
layout(set = 0, binding = 2) uniform usampler2DArray probe_surface;
layout(set = 0, binding = 3) uniform usampler2DArray trace_debug;
layout(set = 0, binding = 4) uniform sampler2DArray shared_hiz;
layout(set = 0, binding = 5) uniform sampler2DArray normal_roughness;
layout(set = 0, binding = 6) uniform sampler2DArray velocity;
layout(set = 0, binding = 7) uniform sampler2DArray directional_radiance;
layout(set = 0, binding = 8) uniform sampler2DArray directional_filtered;
layout(set = 0, binding = 9) uniform sampler2DArray directional_irradiance;
layout(set = 0, binding = 10) uniform usampler2DArray directional_adaptive_tile_data;
layout(set = 0, binding = 11) uniform usampler2DArray directional_history_age;
layout(set = 0, binding = 12) uniform usampler2DArray directional_adaptive_counter;
#else
layout(location = 0) in vec2 uv_interp;
layout(set = 0, binding = 0) uniform sampler2D resolved_radiance;
layout(set = 0, binding = 1) uniform sampler2D selected_radiance;
layout(set = 0, binding = 2) uniform usampler2D probe_surface;
layout(set = 0, binding = 3) uniform usampler2D trace_debug;
layout(set = 0, binding = 4) uniform sampler2D shared_hiz;
layout(set = 0, binding = 5) uniform sampler2D normal_roughness;
layout(set = 0, binding = 6) uniform sampler2D velocity;
layout(set = 0, binding = 7) uniform sampler2D directional_radiance;
layout(set = 0, binding = 8) uniform sampler2D directional_filtered;
layout(set = 0, binding = 9) uniform sampler2D directional_irradiance;
layout(set = 0, binding = 10) uniform usampler2D directional_adaptive_tile_data;
layout(set = 0, binding = 11) uniform usampler2D directional_history_age;
layout(set = 0, binding = 12) uniform usampler2D directional_adaptive_counter;
#endif

layout(location = 0) out vec4 frag_color;

layout(push_constant, std430) uniform Params {
	vec2 resolution;
	float selected_radiance_scale;
	uint flags;
	uint surface_layer_stride;
	uint surface_history_slot;
	uint hiz_mip_count;
	uint pad;
}
params;

vec4 sample_resolved(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(resolved_radiance, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(resolved_radiance, uv, 0.0);
#endif
}

vec4 sample_selected(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(selected_radiance, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(selected_radiance, uv, 0.0);
#endif
}

vec4 sample_normal_roughness(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(normal_roughness, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(normal_roughness, uv, 0.0);
#endif
}

vec4 sample_velocity(vec2 uv) {
#ifdef USE_MULTIVIEW
	return textureLod(velocity, vec3(uv, uv_interp.z), 0.0);
#else
	return textureLod(velocity, uv, 0.0);
#endif
}

float sample_hiz(vec2 uv, float lod) {
#ifdef USE_MULTIVIEW
	return textureLod(shared_hiz, vec3(uv, uv_interp.z), lod).r;
#else
	return textureLod(shared_hiz, uv, lod).r;
#endif
}

ivec2 probe_surface_size() {
#ifdef USE_MULTIVIEW
	return textureSize(probe_surface, 0).xy;
#else
	return textureSize(probe_surface, 0);
#endif
}

uvec4 load_probe_surface(vec2 uv) {
	ivec2 size = probe_surface_size();
	ivec2 position = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
#ifdef USE_MULTIVIEW
	uint layer = uint(uv_interp.z) * params.surface_layer_stride + params.surface_history_slot;
	return texelFetch(probe_surface, ivec3(position, int(layer)), 0);
#else
	return texelFetch(probe_surface, position, 0);
#endif
}

uint load_trace_debug(vec2 uv) {
#ifdef USE_MULTIVIEW
	ivec2 size = textureSize(trace_debug, 0).xy;
	ivec2 position = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
	return texelFetch(trace_debug, ivec3(position, int(uv_interp.z)), 0).r;
#else
	ivec2 size = textureSize(trace_debug, 0);
	ivec2 position = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
	return texelFetch(trace_debug, position, 0).r;
#endif
}

vec3 unavailable_color(vec2 pixel) {
	float checker = mod(floor(pixel.x / 12.0) + floor(pixel.y / 12.0), 2.0);
	return mix(vec3(0.055), vec3(0.13), checker);
}

bool non_finite(vec3 value) {
	return any(isnan(value)) || any(isinf(value));
}

vec3 debug_tonemap(vec3 radiance) {
	if (non_finite(radiance)) {
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

vec3 trace_debug_color(uint debug_word, vec2 pixel) {
	if ((debug_word & TRACE_DEBUG_INVALID) != 0u) {
		return vec3(1.0, 0.0, 1.0);
	}
	if ((debug_word & TRACE_DEBUG_VALID_SURFACE) == 0u) {
		return vec3(0.025);
	}
	vec3 outcome = vec3(0.12);
	if ((debug_word & TRACE_DEBUG_DETAIL_HIT) != 0u) {
		outcome = vec3(0.05, 0.85, 0.16);
	} else if ((debug_word & TRACE_DEBUG_DETAIL_REJECTED) != 0u) {
		outcome = vec3(1.0, 0.38, 0.02);
	} else if ((debug_word & TRACE_DEBUG_DETAIL_MISS) != 0u) {
		outcome = vec3(0.05, 0.28, 1.0);
	}
	vec3 source = vec3(0.0);
	float source_count = 0.0;
	if ((debug_word & TRACE_DEBUG_SOURCE_SCREEN_HDDAGI) != 0u) {
		source += vec3(0.15, 1.0, 0.2);
		source_count += 1.0;
	}
	if ((debug_word & TRACE_DEBUG_SOURCE_HDDA_VOXEL) != 0u) {
		source += vec3(0.1, 0.32, 1.0);
		source_count += 1.0;
	}
	if ((debug_word & TRACE_DEBUG_SOURCE_SKY) != 0u) {
		source += vec3(0.35, 0.62, 0.95);
		source_count += 1.0;
	}
	if ((debug_word & TRACE_DEBUG_SOURCE_SCREEN_IRRADIANCE_CACHE) != 0u) {
		source += vec3(1.0, 0.72, 0.05);
		source_count += 1.0;
	}
	if ((debug_word & TRACE_DEBUG_SOURCE_HDDA_IRRADIANCE_CACHE) != 0u) {
		source += vec3(1.0, 0.18, 0.52);
		source_count += 1.0;
	}
	vec3 color = source_count > 0.0 ? mix(outcome, source / source_count, 0.55) : outcome;
	if ((debug_word & TRACE_DEBUG_SCREEN_RADIANCE_FALLBACK) != 0u) {
		float stripe = step(0.5, fract((pixel.x + pixel.y) / 8.0));
		color = mix(color, vec3(1.0, 0.9, 0.05), stripe * 0.7);
	}
	return color;
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
	return heatmap(log2(1.0 + hit_distance) / log2(65.0));
}

vec3 oct_to_vec3(vec2 octahedral) {
	vec3 normal = vec3(octahedral, 1.0 - abs(octahedral.x) - abs(octahedral.y));
	float correction = clamp(-normal.z, 0.0, 1.0);
	normal.xy += mix(vec2(correction), vec2(-correction), greaterThanEqual(normal.xy, vec2(0.0)));
	return normalize(normal);
}

vec3 surface_debug_color(vec2 uv) {
	uvec4 surface = load_probe_surface(uv);
	if (surface.x == 0xffffffffu) {
		return vec3(0.015);
	}
	uint packed = surface.w;
	vec2 octahedral = vec2(float(packed & 0xffffu) / 65535.0, float((packed >> 16u) & 0x7fffu) / 32767.0);
	vec3 color = oct_to_vec3(octahedral * 2.0 - 1.0) * 0.5 + 0.5;
	if ((packed & 0x80000000u) != 0u) {
		color = mix(color, vec3(1.0, 0.05, 0.8), 0.32);
	}
	return color;
}

vec3 hiz_debug_color(vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_HIZ_VALID) == 0u || params.hiz_mip_count == 0u) {
		return unavailable_color(pixel);
	}
	vec2 quadrant_position = uv * 2.0;
	ivec2 quadrant = min(ivec2(quadrant_position), ivec2(1));
	uint quadrant_index = uint(quadrant.y * 2 + quadrant.x);
	uint mip = min(quadrant_index * 2u, params.hiz_mip_count - 1u);
	float depth = sample_hiz(fract(quadrant_position), float(mip));
	if (isnan(depth) || isinf(depth)) {
		return vec3(1.0, 0.0, 1.0);
	}
	float value = pow(clamp(depth, 0.0, 1.0), 0.16);
	vec3 tints[4] = vec3[](vec3(0.55, 0.8, 1.0), vec3(0.55, 1.0, 0.62), vec3(1.0, 0.82, 0.38), vec3(1.0, 0.48, 0.55));
	return vec3(value) * tints[quadrant_index];
}

vec3 normal_debug_color(vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_NORMAL_ROUGHNESS_VALID) == 0u) {
		return unavailable_color(pixel);
	}
	vec3 normal = sample_normal_roughness(uv).xyz * 2.0 - 1.0;
	float length_squared = dot(normal, normal);
	if (non_finite(normal) || !(length_squared > 0.001)) {
		return vec3(1.0, 0.0, 1.0);
	}
	return normal * inversesqrt(length_squared) * 0.5 + 0.5;
}

vec3 velocity_debug_color(vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_VELOCITY_VALID) == 0u) {
		return unavailable_color(pixel);
	}
	vec2 motion = sample_velocity(uv).xy * params.resolution;
	if (any(isnan(motion)) || any(isinf(motion))) {
		return vec3(1.0, 0.0, 1.0);
	}
	float magnitude = length(motion);
	vec2 direction = magnitude > 1e-5 ? motion / magnitude : vec2(0.0);
	float intensity = clamp(log2(1.0 + magnitude) / 6.0, 0.0, 1.0);
	return mix(vec3(0.04), vec3(direction * 0.5 + 0.5, intensity), max(intensity, 0.18));
}

ivec2 directional_atlas_size(int atlas_index) {
#ifdef USE_MULTIVIEW
	if (atlas_index == 0) {
		return textureSize(directional_radiance, 0).xy;
	}
	if (atlas_index == 1) {
		return textureSize(directional_filtered, 0).xy;
	}
	return textureSize(directional_irradiance, 0).xy;
#else
	if (atlas_index == 0) {
		return textureSize(directional_radiance, 0);
	}
	if (atlas_index == 1) {
		return textureSize(directional_filtered, 0);
	}
	return textureSize(directional_irradiance, 0);
#endif
}

vec4 load_directional_atlas(int atlas_index, ivec2 position) {
#ifdef USE_MULTIVIEW
	uint history_layer = uint(uv_interp.z) * params.surface_layer_stride + params.surface_history_slot;
	if (atlas_index == 0) {
		return texelFetch(directional_radiance, ivec3(position, int(history_layer)), 0);
	}
	if (atlas_index == 1) {
		return texelFetch(directional_filtered, ivec3(position, int(history_layer)), 0);
	}
	return texelFetch(directional_irradiance, ivec3(position, int(uv_interp.z)), 0);
#else
	if (atlas_index == 0) {
		return texelFetch(directional_radiance, position, 0);
	}
	if (atlas_index == 1) {
		return texelFetch(directional_filtered, position, 0);
	}
	return texelFetch(directional_irradiance, position, 0);
#endif
}

ivec2 directional_tile_data_size() {
#ifdef USE_MULTIVIEW
	return textureSize(directional_adaptive_tile_data, 0).xy;
#else
	return textureSize(directional_adaptive_tile_data, 0);
#endif
}

uvec4 load_directional_tile_data(ivec2 position) {
#ifdef USE_MULTIVIEW
	uint history_layer = uint(uv_interp.z) * params.surface_layer_stride + params.surface_history_slot;
	return texelFetch(directional_adaptive_tile_data, ivec3(position, int(history_layer)), 0);
#else
	return texelFetch(directional_adaptive_tile_data, position, 0);
#endif
}

ivec2 directional_history_age_size() {
#ifdef USE_MULTIVIEW
	return textureSize(directional_history_age, 0).xy;
#else
	return textureSize(directional_history_age, 0);
#endif
}

uint load_directional_history_age(ivec2 position) {
#ifdef USE_MULTIVIEW
	uint history_layer = uint(uv_interp.z) * params.surface_layer_stride + params.surface_history_slot;
	return texelFetch(directional_history_age, ivec3(position, int(history_layer)), 0).r;
#else
	return texelFetch(directional_history_age, position, 0).r;
#endif
}

uint load_directional_adaptive_counter() {
#ifdef USE_MULTIVIEW
	return texelFetch(directional_adaptive_counter, ivec3(ivec2(0), int(uv_interp.z)), 0).r;
#else
	return texelFetch(directional_adaptive_counter, ivec2(0), 0).r;
#endif
}

float directional_grid_factor(vec2 grid_position) {
	vec2 width = abs(dFdx(grid_position)) + abs(dFdy(grid_position));
	vec2 distance = min(fract(grid_position), 1.0 - fract(grid_position));
	vec2 interior = smoothstep(width * 0.45, width * 1.35, distance);
	return mix(0.16, 1.0, min(interior.x, interior.y));
}

vec3 directional_atlas_debug_color(int atlas_index, vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) == 0u) {
		return unavailable_color(pixel);
	}
	ivec2 size = directional_atlas_size(atlas_index);
	ivec2 position = clamp(ivec2(uv * vec2(size)), ivec2(0), size - ivec2(1));
	vec3 color = debug_tonemap(load_directional_atlas(atlas_index, position).rgb);
	color *= directional_grid_factor(uv * vec2(size) / DIRECTIONAL_PROBE_TILE_SIZE);
	if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ADAPTIVE_VALID) != 0u) {
		float split_y = clamp(float(directional_tile_data_size().y) * DIRECTIONAL_PROBE_TILE_SIZE / float(max(size.y, 1)), 0.0, 1.0);
		float split_width = max(fwidth(uv.y) * 1.25, 1.0 / float(max(size.y, 1)));
		float split_line = 1.0 - smoothstep(split_width, split_width * 2.0, abs(uv.y - split_y));
		color = mix(color, vec3(1.0, 0.72, 0.03), split_line * float(split_y < 1.0));
	}
	return color;
}

bool directional_tile_data_valid(uvec4 data, uint capacity) {
	return data.y <= 8u && data.z <= data.y && data.y <= data.w && data.w <= 8u &&
			(data.y == 0u || data.x < capacity) && data.y <= capacity - min(data.x, capacity);
}

vec3 directional_adaptive_debug_color(vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ADAPTIVE_VALID) == 0u) {
		return unavailable_color(pixel);
	}
	ivec2 tile_data_size = directional_tile_data_size();
	ivec2 tile_position = clamp(ivec2(uv * vec2(tile_data_size)), ivec2(0), tile_data_size - ivec2(1));
	uvec4 data = load_directional_tile_data(tile_position);
	uint capacity = uint(tile_data_size.x * tile_data_size.y) / 2u;
	if (!directional_tile_data_valid(data, capacity)) {
		return vec3(1.0, 0.0, 1.0);
	}
	ivec2 age_size = directional_history_age_size();
	uint base_probe_count = uint(tile_data_size.x * tile_data_size.y);
	float age_sum = 0.0;
	uint newborn_count = 0u;
	for (uint offset = 0u; offset < data.y; offset++) {
		uint physical_index = base_probe_count + data.x + offset;
		ivec2 age_position = ivec2(int(physical_index % uint(age_size.x)), int(physical_index / uint(age_size.x)));
		if (any(greaterThanEqual(age_position, age_size))) {
			return vec3(1.0, 0.0, 1.0);
		}
		uint age = load_directional_history_age(age_position);
		age_sum += float(min(age, 8u));
		newborn_count += uint(age <= 1u);
	}
	float density = float(data.y) / 8.0;
	float retained = data.y > 0u ? float(data.z) / float(data.y) : 0.0;
	float maturity = data.y > 0u ? age_sum / (float(data.y) * 8.0) : 0.0;
	float churn = data.y > 0u ? float(newborn_count) / float(data.y) : 0.0;
	vec3 color = mix(vec3(0.008), mix(vec3(1.0, 0.74, 0.015), vec3(1.0, 0.055, 0.015), density), min(density * 4.0, 1.0));
	color = mix(color, mix(vec3(0.04, 0.28, 1.0), vec3(0.04, 1.0, 0.18), maturity), retained * 0.58);
	float crosshatch = step(0.72, fract((pixel.x - pixel.y) / 7.0));
	color = mix(color, vec3(0.9, 0.0, 1.0), crosshatch * churn * 0.72);
	if (data.w > data.y) {
		vec2 fraction = fract(uv * vec2(tile_data_size));
		float edge = min(min(fraction.x, fraction.y), min(1.0 - fraction.x, 1.0 - fraction.y));
		color = mix(color, vec3(1.0, 0.0, 0.0), 1.0 - smoothstep(0.025, 0.07, edge));
	}
	return color * directional_grid_factor(uv * vec2(tile_data_size));
}

vec3 directional_capacity_debug_color(vec2 uv, vec2 pixel) {
	if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ADAPTIVE_VALID) == 0u) {
		return unavailable_color(pixel);
	}
	ivec2 tile_data_size = directional_tile_data_size();
	ivec2 tile_position = clamp(ivec2(uv * vec2(tile_data_size)), ivec2(0), tile_data_size - ivec2(1));
	uvec4 data = load_directional_tile_data(tile_position);
	uint capacity = uint(tile_data_size.x * tile_data_size.y) / 2u;
	if (!directional_tile_data_valid(data, capacity)) {
		return vec3(1.0, 0.0, 1.0);
	}
	vec3 color = vec3(0.008);
	if (data.w > 0u) {
		float demand = float(min(data.w, 8u)) / 8.0;
		color = data.y == data.w ? mix(vec3(0.015, 0.16, 0.035), vec3(0.04, 1.0, 0.18), demand) : (data.y > 0u ? mix(vec3(0.2, 0.08, 0.01), vec3(1.0, 0.7, 0.015), demand) : mix(vec3(0.2, 0.01, 0.01), vec3(1.0, 0.025, 0.01), demand));
	}
	color *= directional_grid_factor(uv * vec2(tile_data_size));
	if (uv.y >= 0.88) {
		float pressure = float(load_directional_adaptive_counter()) / float(max(capacity, 1u));
		vec3 pressure_color = pressure < 0.7 ? vec3(0.04, 1.0, 0.18) : (pressure <= 1.0 ? vec3(1.0, 0.72, 0.015) : vec3(1.0, 0.02, 0.01));
		color = uv.x <= min(pressure, 1.0) ? pressure_color : vec3(0.015);
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
	if (character == 67u) {
		uint rows[5] = uint[](7u, 8u, 8u, 8u, 7u);
		return rows[row];
	}
	if (character == 68u) {
		uint rows[5] = uint[](14u, 9u, 9u, 9u, 14u);
		return rows[row];
	}
	if (character == 70u) {
		uint rows[5] = uint[](15u, 8u, 14u, 8u, 8u);
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
	if (character == 76u) {
		uint rows[5] = uint[](8u, 8u, 8u, 8u, 15u);
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
	if (character == 79u) {
		uint rows[5] = uint[](6u, 9u, 9u, 9u, 6u);
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
	if (character == 85u) {
		uint rows[5] = uint[](9u, 9u, 9u, 9u, 6u);
		return rows[row];
	}
	if (character == 87u) {
		uint rows[5] = uint[](9u, 9u, 15u, 15u, 9u);
		return rows[row];
	}
	if (character == 90u) {
		uint rows[5] = uint[](15u, 1u, 2u, 4u, 15u);
		return rows[row];
	}
	return 0u;
}

uint panel_label_character(int panel, int index) {
	if (panel == 0) {
		uint label[3] = uint[](70u, 73u, 78u);
		return label[index];
	}
	if (panel == 1) {
		if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u) {
			uint label[3] = uint[](82u, 65u, 68u);
			return label[index];
		}
		uint label[3] = uint[](82u, 65u, 87u);
		return label[index];
	}
	if (panel == 2) {
		if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u) {
			uint label[3] = uint[](70u, 76u, 84u);
			return label[index];
		}
		uint label[3] = uint[](68u, 73u, 70u);
		return label[index];
	}
	if (panel == 3) {
		if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u) {
			uint label[3] = uint[](73u, 82u, 82u);
			return label[index];
		}
		uint label[3] = uint[](70u, 84u, 82u);
		return label[index];
	}
	if (panel == 4) {
		uint label[3] = uint[](72u, 73u, 84u);
		return label[index];
	}
	if (panel == 5) {
		uint label[3] = uint[](83u, 85u, 82u);
		return label[index];
	}
	if (panel == 6) {
		uint label[3] = uint[](72u, 73u, 90u);
		return label[index];
	}
	if (panel == 7) {
		if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u) {
			uint label[3] = uint[](65u, 68u, 80u);
			return label[index];
		}
		uint label[3] = uint[](78u, 82u, 77u);
		return label[index];
	}
	if ((params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u) {
		uint label[3] = uint[](67u, 65u, 80u);
		return label[index];
	}
	uint label[3] = uint[](77u, 79u, 84u);
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
	bool directional = (params.flags & MONTAGE_FLAG_DIRECTIONAL_ATLAS_VALID) != 0u;

	vec3 color;
	if (panel == 0) {
		color = (params.flags & MONTAGE_FLAG_SELECTED_OUTPUT_VALID) != 0u ? debug_tonemap(sample_selected(source_uv).rgb * params.selected_radiance_scale) : debug_tonemap(sample_resolved(source_uv).rgb);
	} else if (panel == 1) {
		color = directional ? directional_atlas_debug_color(0, source_uv, tile_pixel) : debug_tonemap(sample_resolved(source_uv).rgb);
	} else if (panel == 2) {
		if (directional) {
			color = directional_atlas_debug_color(1, source_uv, tile_pixel);
		} else if ((params.flags & MONTAGE_FLAG_SELECTED_OUTPUT_VALID) == 0u) {
			color = unavailable_color(tile_pixel);
		} else {
			vec3 difference = abs(sample_selected(source_uv).rgb * params.selected_radiance_scale - sample_resolved(source_uv).rgb);
			color = non_finite(difference) ? vec3(1.0, 0.0, 1.0) : heatmap(max(max(difference.r, difference.g), difference.b) * 4.0);
		}
	} else if (panel == 3) {
		color = directional ? directional_atlas_debug_color(2, source_uv, tile_pixel) : trace_debug_color(load_trace_debug(source_uv), tile_pixel);
	} else if (panel == 4) {
		color = hit_distance_color(sample_resolved(source_uv).a);
	} else if (panel == 5) {
		color = surface_debug_color(source_uv);
	} else if (panel == 6) {
		color = hiz_debug_color(source_uv, tile_pixel);
	} else if (panel == 7) {
		color = directional ? directional_adaptive_debug_color(source_uv, tile_pixel) : normal_debug_color(source_uv, tile_pixel);
	} else {
		color = directional ? directional_capacity_debug_color(source_uv, tile_pixel) : velocity_debug_color(source_uv, tile_pixel);
	}

	int font_scale = max(int(floor(min(tile_size.x, tile_size.y) / 150.0)), 1);
	if (tile_pixel.y < float(font_scale * 13)) {
		color = mix(color, vec3(0.12, 0.2, 0.28), 0.82);
		if (panel_label_pixel(panel, tile_pixel, tile_size)) {
			color = vec3(1.0);
		}
	}
	float border = min(min(tile_pixel.x, tile_pixel.y), min(tile_size.x - tile_pixel.x, tile_size.y - tile_pixel.y));
	if (border < 1.5) {
		color = vec3(0.0);
	}
	frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
