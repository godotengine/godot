#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(r8, set = 0, binding = 1) uniform restrict writeonly image2D output_shadow;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D output_debug;

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	ivec2 light_offset;
	vec4 light_coordinate;
	int max_steps;
	int debug_enabled;
	int debug_mode;
	float bilinear_threshold;
	float shadow_contrast;
	float surface_thickness;
	int use_precision_offset;
	int ignore_edge_pixels;
	int bilinear_sampling_offset_mode;
}
params;

#define WAVE_SIZE 64
#define SAMPLE_COUNT 60
#define READ_COUNT (SAMPLE_COUNT / WAVE_SIZE + 2)
#define HARD_SHADOW_SAMPLES 4
#define FADE_OUT_SAMPLES 8
// Reverse-Z
#define DEPTH_NEAR 1.0
#define DEPTH_FAR 0.0
#define Z_SIGN -1.0

shared float DepthData[READ_COUNT * WAVE_SIZE];

void main() {
	int group_id = int(gl_WorkGroupID.x);
	int thread_id = int(gl_LocalInvocationID.x);
	vec4 light = params.light_coordinate;

	vec2 light_xy = floor(light.xy) + 0.5;
	vec2 light_xy_fraction = light.xy - light_xy;

	float direction = -light.w;

	ivec2 group_offset = ivec2(gl_WorkGroupID.yz);
	ivec2 xy = group_offset * WAVE_SIZE + params.light_offset;
	ivec2 sign_xy = ivec2(sign(vec2(xy)));

	// Determine group orientation and direction
	bool horizontal = abs(xy.y) > abs(xy.x) || xy.x == xy.y;
	ivec2 axis = horizontal ? ivec2(-sign_xy.y, 0.0) : ivec2(0.0, sign_xy.x);

	xy += axis * group_id;
	vec2 xy_f = vec2(xy);

	//	bool x_major_axis = abs(xy_f.x) > abs(xy_f.y);
	bool x_major_axis = !horizontal;
	float main_axis = x_major_axis ? xy_f.x : xy_f.y;

	float ma_light_frac = x_major_axis ? light_xy_fraction.x : light_xy_fraction.y;
	ma_light_frac = main_axis > 0.0 ? -ma_light_frac : ma_light_frac;

	float main_axis_start = abs(main_axis) + ma_light_frac;
	float main_axis_end = main_axis_start - float(WAVE_SIZE);

	vec2 group_start = xy_f + light_xy;
	vec2 group_end = mix(light.xy, group_start, main_axis_end / main_axis_start);

	bool start_off = any(lessThan(group_start, vec2(0.0))) || any(greaterThanEqual(group_start, vec2(params.screen_size)));
	bool end_off = any(lessThan(group_end, vec2(0.0))) || any(greaterThanEqual(group_end, vec2(params.screen_size)));
	if (start_off && end_off) {
		return;
	}
	if (xy == ivec2(0, 0)) {
		return;
	}

	// Bake direction into delta so caller can use: pixel_pos += out_delta (same as custom version)
	float thread_step = float(thread_id ^ (direction > 0.0 ? (WAVE_SIZE - 1) : 0));

	vec2 pixel_pos = mix(group_start, group_end, thread_step / float(WAVE_SIZE));
	float pixel_distance = main_axis_start - thread_step;

	vec2 group_delta = direction * (group_start - group_end);

	float sampling_depth[READ_COUNT];
	float shadowing_depth[READ_COUNT];
	float sample_distance[READ_COUNT];
	float depth_thickness_scale[READ_COUNT];

	bool is_edge = false;
	ivec2 write_xy = ivec2(pixel_pos);
	const float edge_skip = 1e20;
	const bool ignore_edge_pixels = params.ignore_edge_pixels > 0;
	const bool bilinear_sampling_offset_mode = params.bilinear_sampling_offset_mode > 0;
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

		if (params.bilinear_sampling_offset_mode > 0) {
			bilinear = use_point_filter ? 0 : bilinear;
			//both shadow depth and starting depth are the same in this mode, unless shadow skipping edges
			sampling_depth[i] = mix(depths.x, depths.y, abs(bilinear));
			shadowing_depth[i] = (ignore_edge_pixels && use_point_filter) ? edge_skip : sampling_depth[i];
		} else {
			// The pixel starts sampling at this depth
			sampling_depth[i] = depths.x;

			float edge_depth = ignore_edge_pixels ? edge_skip : depths.x;
			// Any sample in this wavefront is possibly interpolated towards the bilinear sample
			// So use should use a shadowing depth that is further away, based on the difference between the two samples
			float shadow_depth = depths.x + abs(depths.x - depths.y) * Z_SIGN;

			// Shadows cast from this depth
			shadowing_depth[i] = use_point_filter ? edge_depth : shadow_depth;
		}

		sample_distance[i] = pixel_distance + (WAVE_SIZE * i) * direction;

		float stored_depth = (shadowing_depth[i] - light.z) / sample_distance[i];
		if (i != 0) {
			stored_depth = sample_distance[i] > 0.0 ? stored_depth : 1e10;
		}

		int idx = (i * WAVE_SIZE) + thread_id;
		DepthData[idx] = stored_depth;
		pixel_pos += group_delta;
	}

	memoryBarrierShared();
	barrier();

	float depth_scale = min(sample_distance[0] + direction, 1.0 / params.surface_thickness) * sample_distance[0] / depth_thickness_scale[0];

	float start_depth = sampling_depth[0];
	if (params.use_precision_offset != 0) {
		start_depth = mix(start_depth, DEPTH_FAR, -1.0 / float(0xFFFF));
	}

	start_depth = (start_depth - light.z) / sample_distance[0];
	start_depth = start_depth * depth_scale - Z_SIGN;

	int sample_index = thread_id + 1;
	vec4 shadow_value = vec4(1.0);
	float hard_shadow = 1.0;

	// Hard shadow samples — one sample can fully shadow the pixel
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

	imageStore(output_shadow, write_xy, vec4(shadow, 0.0, 0.0, 0.0));

	float debug_result = shadow;
	switch (params.debug_mode) {
		case 3: // Raw shadow
			debug_result = shadow;
			break;
		case 2: // Groups
			debug_result = fract(float(gl_WorkGroupID.x) / float(WAVE_SIZE));
			break;
		case 1: // Threads
			debug_result = float(thread_id) / float(WAVE_SIZE);
			break;
		case 0: // Raw depth buffer
			debug_result = sampling_depth[0];
			break;
		default:
			break;
	}
	imageStore(output_debug, write_xy, vec4(debug_result, debug_result, debug_result, 1.0));
}
