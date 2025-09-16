#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D source_last_frame;
layout(set = 0, binding = 1) uniform sampler2D source_hiz;
layout(set = 0, binding = 2) uniform sampler2D source_normal_roughness;
layout(rgba16f, set = 0, binding = 3) uniform restrict writeonly image2D output_color;

layout(set = 0, binding = 4, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	mat4 reprojection[2];
	vec4 eye_offset[2];
}
scene_data;

#ifdef MODE_ROUGH
layout(r8, set = 1, binding = 0) uniform restrict writeonly image2D output_mip_level;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 screen_size;
	int mipmaps;
	int num_steps;
	float distance_fade;
	float curve_fade_in;
	float depth_tolerance;
	bool orthogonal;
	int view_index;
}
params;

vec2 compute_cell_count(int level) {
	int cell_count_x = max(1, params.screen_size.x >> level);
	int cell_count_y = max(1, params.screen_size.y >> level);
	return vec2(cell_count_x, cell_count_y);
}

float linearize_depth(float depth) {
	vec4 pos = vec4(0.0, 0.0, depth, 1.0);
	pos = scene_data.inv_projection[params.view_index] * pos;
	return pos.z / pos.w;
}

#define M_PI 3.14159265359

void main() {
	ivec2 pixel_pos = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(pixel_pos, params.screen_size))) {
		return;
	}

	vec4 color = vec4(0.0);
#ifdef MODE_ROUGH
	float mip_level = 0.0;
#endif

	vec3 screen_pos;
	screen_pos.xy = (pixel_pos + 0.5) / params.screen_size;
	screen_pos.z = texelFetch(source_hiz, pixel_pos, 0).x;

	bool should_trace = screen_pos.z != 0.0;
	if (should_trace) {
		vec4 pos;
		pos.xy = screen_pos.xy * 2.0 - 1.0;
		pos.z = screen_pos.z;
		pos.w = 1.0;
		pos = scene_data.inv_projection[params.view_index] * pos;
		pos.xyz /= pos.w;

		vec4 normal_roughness = texelFetch(source_normal_roughness, pixel_pos * 2, 0);
		vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
		float roughness = normal_roughness.w;
		if (roughness > 0.5) {
			roughness = 1.0 - roughness;
		}
		roughness /= (127.0 / 255.0);

		// The roughness cutoff of 0.6 is chosen to match the roughness fadeout from GH-69828.
		if (roughness > 0.6) {
			// Do not compute SSR for rough materials to improve performance at the cost of
			// subtle artifacting.
			imageStore(output_color, pixel_pos, vec4(0.0));
#ifdef MODE_ROUGH
			imageStore(output_mip_level, pixel_pos, vec4(0.0));
#endif
			return;
		}

		vec3 view_dir = params.orthogonal ? vec3(0.0, 0.0, -1.0) : normalize(pos.xyz + scene_data.eye_offset[params.view_index].xyz);
		vec3 ray_dir = normalize(reflect(view_dir, normal));
		vec3 end_pos = pos.xyz + ray_dir;

		vec4 screen_end_pos = scene_data.projection[params.view_index] * vec4(end_pos, 1.0);
		screen_end_pos.xyz /= screen_end_pos.w;
		screen_end_pos.xy = screen_end_pos.xy * 0.5 + 0.5;

		// Normalize Z to -1.0 or +1.0 and do parametric T tracing as suggested here:
		// https://hacksoflife.blogspot.com/2020/10/a-tip-for-hiz-ssr-parametric-t-tracing.html
		vec3 screen_ray_dir = screen_end_pos.xyz - screen_pos;
		screen_ray_dir /= abs(screen_ray_dir.z);

		bool facing_camera = screen_ray_dir.z >= 0.0;

		// Find the screen edge point where we will stop tracing.
		vec2 t0 = (vec2(0.0) - screen_pos.xy) / screen_ray_dir.xy;
		vec2 t1 = (vec2(1.0) - screen_pos.xy) / screen_ray_dir.xy;
		vec2 t2 = max(t0, t1);
		float t_max = min(t2.x, t2.y);

		vec2 cell_step = vec2(screen_ray_dir.x < 0.0 ? -1.0 : 1.0, screen_ray_dir.y < 0.0 ? -1.0 : 1.0);

		const int START_LEVEL = 0;

		int cur_level = START_LEVEL;
		int cur_iteration = params.num_steps;

		// Advance the start point to the closest next cell to prevent immediate self intersection.
		float t;
		{
			vec2 cell_count = compute_cell_count(cur_level);
			vec2 cell_index = floor(screen_pos.xy * cell_count);
			vec2 new_cell_index = cell_index + clamp(cell_step, vec2(0.0), vec2(1.0));
			vec2 new_cell_pos = (new_cell_index / cell_count) + cell_step * 0.000001;
			vec2 pos_t = (new_cell_pos - screen_pos.xy) / screen_ray_dir.xy;
			float edge_t = min(pos_t.x, pos_t.y);

			t = edge_t;
		}

		while (cur_level >= 0 && cur_iteration > 0 && t < t_max) {
			vec3 cur_screen_pos = screen_pos + screen_ray_dir * t;

			vec2 cell_count = compute_cell_count(cur_level);
			vec2 cell_index = floor(cur_screen_pos.xy * cell_count);
			float cell_depth = texelFetch(source_hiz, ivec2(cell_index), cur_level).x;
			float depth_t = (cell_depth - screen_pos.z) * screen_ray_dir.z; // Z is either -1.0 or 1.0 so we don't need to do a divide.

			vec2 new_cell_index = cell_index + clamp(cell_step, vec2(0.0), vec2(1.0));
			vec2 new_cell_pos = (new_cell_index / cell_count) + cell_step * 0.000001;
			vec2 pos_t = (new_cell_pos - screen_pos.xy) / screen_ray_dir.xy;
			float edge_t = min(pos_t.x, pos_t.y);

			bool hit = facing_camera ? (depth_t >= edge_t) : (depth_t <= edge_t);
			int mip_offset = hit ? -1 : +1;

			if (cur_level == 0) {
				float z0 = linearize_depth(cell_depth);
				float z1 = linearize_depth(cur_screen_pos.z);

				if ((z0 - z1) > params.depth_tolerance) {
					hit = false;
					mip_offset = 0; // Keep the mip index the same to prevent it from decreasing and increasing in repeat.
				}
			}

			if (hit) {
				t = facing_camera ? min(t, depth_t) : max(t, depth_t);
			} else {
				t = edge_t;
			}

			cur_level = min(cur_level + mip_offset, params.mipmaps - 1);
			--cur_iteration;
		}

		vec3 cur_screen_pos = screen_pos + screen_ray_dir * t;

		vec4 reprojected_pos;
		reprojected_pos.xy = cur_screen_pos.xy * 2.0 - 1.0;
		reprojected_pos.z = cur_screen_pos.z;
		reprojected_pos.w = 1.0;
		reprojected_pos = scene_data.reprojection[params.view_index] * reprojected_pos;
		reprojected_pos.xy = reprojected_pos.xy / reprojected_pos.w * 0.5 + 0.5;

		color = vec4(textureLod(source_last_frame, reprojected_pos.xy, 0).xyz, 1.0);

		// Instead of hard rejecting samples, write sample validity to the alpha channel.
		// This allows invalid samples to write mip levels to let valid samples have smoother roughness transitions.

		// Hit validation logic is referenced from here:
		// https://github.com/GPUOpen-Effects/FidelityFX-SSSR/blob/master/ffx-sssr/ffx_sssr.h

		float hit_depth = texelFetch(source_hiz, ivec2(cur_screen_pos.xy * params.screen_size), 0).x;
		if (t >= t_max || all(lessThanEqual(abs(cur_screen_pos.xy - screen_pos.xy), 2.0 / params.screen_size)) || hit_depth == 0.0) {
			color.a = 0.0;
		}

		float delta = abs(linearize_depth(cur_screen_pos.z) - linearize_depth(hit_depth));
		float confidence = 1.0 - smoothstep(0.0, params.depth_tolerance, delta);
		color.a *= clamp(confidence * confidence, 0.0, 1.0);

		float margin_blend = 1.0;
		vec2 hit_pos = reprojected_pos.xy * params.screen_size;

		vec2 margin = vec2((params.screen_size.x + params.screen_size.y) * 0.05); // Make a uniform margin.
		{
			// Blend fading out towards inner margin.
			// 0.5 = midpoint of reflection
			vec2 margin_grad = mix(params.screen_size - hit_pos, hit_pos, lessThan(hit_pos, params.screen_size * 0.5));
			margin_blend = smoothstep(0.0, margin.x * margin.y, margin_grad.x * margin_grad.y);
		}

		float ray_len = length(screen_ray_dir.xy * t);

		// Fade In / Fade Out
		float grad = ray_len;
		float initial_fade = params.curve_fade_in == 0.0 ? 1.0 : pow(clamp(grad, 0.0, 1.0), params.curve_fade_in);
		float fade = pow(clamp(1.0 - grad, 0.0, 1.0), params.distance_fade) * initial_fade;

		// Ensure that precision errors do not introduce any fade. Even if it is just slightly below 1.0,
		// strong specular light can leak through the reflection.
		if (fade > 0.999) {
			fade = 1.0;
		}

		color.a *= fade * margin_blend;

#ifdef MODE_ROUGH
		// Tone map the SSR color to have smoother roughness filtering across samples with varying luminance.
		color.rgb /= 1.0 + dot(color.rgb, vec3(0.299, 0.587, 0.114));

		if (roughness > 0.001) {
			float cone_angle = min(roughness, 0.999) * M_PI * 0.5;
			float cone_len = ray_len;
			float op_len = 2.0 * tan(cone_angle) * cone_len; // Opposite side of iso triangle.
			float blur_radius;
			{
				// Fit to sphere inside cone (sphere ends at end of cone), something like this:
				// ___
				// \O/
				//  V
				//
				// as it avoids bleeding from beyond the reflection as much as possible. As a plus
				// it also makes the rough reflection more elongated.
				float a = op_len;
				float h = cone_len;
				float a2 = a * a;
				float fh2 = 4.0 * h * h;
				blur_radius = (a * (sqrt(a2 + fh2) - a)) / (4.0 * h);
			}

			// The division by 15.0 here makes it match with the old roughness blurring implementation.
			mip_level = clamp(log2(blur_radius * max(params.screen_size.x, params.screen_size.y) / 15.0), 0, params.mipmaps - 1);
		}

		// Because we still write mip level for invalid pixels to allow for smooth roughness transitions,
		// this sometimes ends up creating a pyramid-like shape at very rough levels.
		// We can fade the mip level near the end to make it significantly less visible.
		mip_level *= pow(clamp(1.25 - ray_len, 0.0, 1.0), 0.2);
#endif

		color.rgb *= color.a;
	}

	imageStore(output_color, pixel_pos, color);
#ifdef MODE_ROUGH
	imageStore(output_mip_level, pixel_pos, vec4(mip_level / 14.0, 0.0, 0.0, 0.0));
#endif
}
