#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(r8, set = 0, binding = 1) uniform restrict writeonly image2D output_shadow;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 light_offset;
	vec4 light_coordinate;
	float bilinear_threshold;
	float shadow_contrast;
	float surface_thickness;
	int ignore_edge_pixels;
	float depth_begin;
	float depth_end;
}
params;

#define WAVE_SIZE 64
#ifndef READ_COUNT
#define READ_COUNT 2
#endif
#define SAMPLE_COUNT ((READ_COUNT - 1) * WAVE_SIZE - 1)
#define HARD_SHADOW_SAMPLES 4
#define FADE_OUT_SAMPLES 8

// Reverse-Z
#define DEPTH_NEAR 1.0
#define DEPTH_FAR 0.0
#define Z_SIGN -1.0

shared float DepthData[READ_COUNT * WAVE_SIZE];

void organise_groups(vec4 light, in float converging, int group_id, ivec2 group_offset, int thread_id, out bool x_major_axis, out vec2 group_start, out vec2 group_end, out vec2 group_delta, out float pixel_distance, out vec2 pixel_pos) {
	vec2 light_xy = floor(light.xy) + 0.5;
	vec2 light_xy_fraction = light.xy - light_xy;

	ivec2 xy = group_offset * WAVE_SIZE + params.light_offset;
	ivec2 sign_xy = ivec2(sign(vec2(xy)));

	// Determine group orientation and direction
	bool horizontal = abs(xy.y) > abs(xy.x) || xy.x == xy.y;
	ivec2 axis = horizontal ? ivec2(-sign_xy.y, 0.0) : ivec2(0.0, sign_xy.x);

	xy += axis * group_id;
	vec2 xy_f = vec2(xy);

	x_major_axis = abs(xy_f.x) > abs(xy_f.y);
	float main_axis = x_major_axis ? xy_f.x : xy_f.y;

	float ma_light_frac = x_major_axis ? light_xy_fraction.x : light_xy_fraction.y;
	ma_light_frac = main_axis > 0.0 ? -ma_light_frac : ma_light_frac;

	float main_axis_start = abs(main_axis) + ma_light_frac;
	float main_axis_end = max(main_axis_start - float(WAVE_SIZE), 0.0);

	group_start = xy_f + light_xy;
	group_end = mix(light.xy, group_start, main_axis_end / main_axis_start);

	// Bake direction into delta so caller can use: pixel_pos += out_delta (same as custom version)
	float thread_step = float(thread_id ^ (converging > 0.0 ? (WAVE_SIZE - 1) : 0));

	pixel_pos = mix(group_start, group_end, thread_step / float(WAVE_SIZE));
	pixel_distance = main_axis_start - thread_step;
	group_delta = converging * (group_start - group_end);
}

void organise_groups_orthogonal(vec4 light, int group_id, ivec2 group_offset, int thread_id, out bool x_major_axis, out vec2 group_start, out vec2 group_end, out vec2 group_delta, out float pixel_distance, out vec2 pixel_pos) {
	ivec2 xy = group_offset * WAVE_SIZE + params.light_offset;

	// Determine group orientation and direction
	bool horizontal = abs(light.x) > abs(light.y);
	ivec2 axis = horizontal ? ivec2(0.0, 1.0) : ivec2(1.0, 0.0);
	x_major_axis = horizontal;

	// Offset groups
	xy += axis * group_id;
	vec2 xy_f = vec2(xy);

	group_start = xy_f;
	group_end = group_start + WAVE_SIZE * light.xy;

	// Bake direction into delta so caller can use: pixel_pos += out_delta (same as custom version)
	pixel_pos = mix(group_start, group_end, thread_id / float(WAVE_SIZE));
	pixel_distance = thread_id;
	group_delta = group_end - group_start;
}

void main() {
	int group_id = int(gl_WorkGroupID.x);
	int thread_id = int(gl_LocalInvocationID.x);
	ivec2 group_offset = ivec2(gl_WorkGroupID.yz);
	vec4 light = params.light_coordinate;

	float converging = -(light.w); // 1 converging rays, -1 diverging rays, 0 parrella rays (orthogonal)
	vec2 group_start, group_end;
	vec2 pixel_pos;
	vec2 group_delta;
	float pixel_distance;
	bool x_major_axis;
	if (converging != 0) {
		organise_groups(light, converging, group_id, group_offset, thread_id, x_major_axis, group_start, group_end, group_delta, pixel_distance, pixel_pos);
	} else {
		organise_groups_orthogonal(light, group_id, group_offset, thread_id, x_major_axis, group_start, group_end, group_delta, pixel_distance, pixel_pos);
	}

	bvec2 group_below_zero = lessThan(max(group_start, group_end), vec2(0.0));
	bvec2 group_above_screen = greaterThan(min(group_start, group_end), params.screen_size);

	// Groups entirely offscreen can exit early
	if (any(group_below_zero) || any(group_above_screen)) {
		return;
	}

	float sampling_depth[READ_COUNT];
	float shadowing_depth[READ_COUNT];
	float sample_distance[READ_COUNT];
	float depth_thickness_scale[READ_COUNT];

	bool is_edge = false;
	ivec2 write_xy = ivec2(pixel_pos);
	const float edge_skip = 1e20;
	const bool ignore_edge_pixels = params.ignore_edge_pixels > 0;
	for (int i = 0; i < READ_COUNT; i++) {
		vec2 read_xy = floor(pixel_pos);
		float minor_axis = x_major_axis ? pixel_pos.y : pixel_pos.x;
		float bilinear = fract(minor_axis) - 0.5;
		int neighbor_bias = bilinear > 0 ? 1 : -1;
		ivec2 neighbor_offset = x_major_axis ? ivec2(0, neighbor_bias) : ivec2(neighbor_bias, 0);

		vec2 depths;
		vec2 uv = (read_xy + 0.5) / vec2(params.screen_size);
		depths.x = textureLod(depth_buffer, uv, 0.0).x;

		vec2 uv_neighbor = (vec2(ivec2(read_xy) + neighbor_offset) + 0.5) / vec2(params.screen_size);
		depths.y = textureLod(depth_buffer, uv_neighbor, 0.0).x;

		depth_thickness_scale[i] = abs(DEPTH_FAR - depths.x);
		bool use_point_filter = abs(depths.x - depths.y) > depth_thickness_scale[i] * params.bilinear_threshold;
		if (i == 0) {
			is_edge = use_point_filter;
		}

		// The pixel starts sampling at this depth
		sampling_depth[i] = depths.x;

		float edge_depth = ignore_edge_pixels ? edge_skip : depths.x;
		// Any sample in this wavefront is possibly interpolated towards the bilinear sample
		// So use should use a shadowing depth that is further away, based on the difference between the two samples
		float shadow_depth = depths.x + abs(depths.x - depths.y) * Z_SIGN;

		// Shadows cast from this depth
		shadowing_depth[i] = use_point_filter ? edge_depth : shadow_depth;

		float stored_depth;
		if (converging != 0) {
			sample_distance[i] = pixel_distance + (WAVE_SIZE * i) * converging;
			stored_depth = (shadowing_depth[i] - light.z) / sample_distance[i];
		} else {
			sample_distance[i] = pixel_distance + (WAVE_SIZE * i);
			stored_depth = shadowing_depth[i] - light.z * sample_distance[i];
		}

		if (i != 0) {
			stored_depth = sample_distance[i] > 0.0 ? stored_depth : 1e10;
		}

		int idx = (i * WAVE_SIZE) + thread_id;
		DepthData[idx] = stored_depth;
		pixel_pos += group_delta;
	}

	memoryBarrierShared();
	barrier();

	// Check if pixel is within depth bounds (fade for smooth transition)
	float near_fade_start = params.depth_begin * 1.05;
	float far_fade_start = params.depth_end - params.depth_end * 0.05 - 0.0001;
	if (sampling_depth[0] <= DEPTH_FAR || sampling_depth[0] <= far_fade_start || sampling_depth[0] >= near_fade_start) {
		return;
	}
	float near_fade = 1.0 - smoothstep(params.depth_begin, near_fade_start, sampling_depth[0]);
	float far_fade = smoothstep(far_fade_start, params.depth_end, sampling_depth[0]);
	float depth_fade = near_fade * far_fade;

	float depth_scale;
	float start_depth = sampling_depth[0];
	if (converging != 0) {
		depth_scale = (1.0 / params.surface_thickness) * sample_distance[0] / depth_thickness_scale[0];
		start_depth = (start_depth - light.z) / sample_distance[0];
		start_depth = start_depth * depth_scale - Z_SIGN;
	} else {
		depth_scale = (1.0 / params.surface_thickness) / depth_thickness_scale[0];
		start_depth = (start_depth - light.z * sample_distance[0]) * depth_scale - Z_SIGN;
	}

	int sample_index = thread_id + 1;
	vec4 shadow_value = vec4(1.0);
	float hard_shadow = 1.0;

	for (int i = 0; i < HARD_SHADOW_SAMPLES; i++) {
		float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
		hard_shadow = min(hard_shadow, depth_delta);
	}

	// Main averaged samples — accumulated into 4 buckets to soften aliasing
	for (int i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++) {
		float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
		shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta);
	}

	// Fade out samples — push distant shadow edge toward lit
	for (int i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++) {
		float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
		float fade_out = float(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES)) / float(FADE_OUT_SAMPLES + 1) * 0.75;
		shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta + fade_out);
	}

	// Apply contrast boost
	shadow_value = clamp(shadow_value * params.shadow_contrast + (1.0 - params.shadow_contrast), 0.0, 1.0);
	hard_shadow = clamp(hard_shadow * params.shadow_contrast + (1.0 - params.shadow_contrast), 0.0, 1.0);

	// Average the 4 buckets, then take the harder of hard_shadow and averaged result
	float shadow = min(hard_shadow, dot(shadow_value, vec4(0.25)));
	shadow = mix(1.0, shadow, depth_fade);
	imageStore(output_shadow, write_xy, vec4(shadow, 0.0, 0.0, 0.0));
}
