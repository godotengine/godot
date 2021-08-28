#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform restrict readonly image2D source_diffuse;
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D source_depth;
layout(rgba16f, set = 1, binding = 0) uniform restrict writeonly image2D ssr_image;
#ifdef MODE_ROUGH
layout(r8, set = 1, binding = 1) uniform restrict writeonly image2D blur_radius_image;
#endif
layout(rgba8, set = 2, binding = 0) uniform restrict readonly image2D source_normal_roughness;
layout(set = 3, binding = 0) uniform sampler2D source_metallic;

layout(push_constant, binding = 2, std430) uniform Params {
	vec4 proj_info;

	ivec2 screen_size;
	float camera_z_near;
	float camera_z_far;

	int num_steps;
	float depth_tolerance;
	float distance_fade;
	float curve_fade_in;

	bool orthogonal;
	float filter_mipmap_levels;
	bool use_half_res;
	uint metallic_mask;

	mat4 projection;
}
params;

vec2 view_to_screen(vec3 view_pos, out float w) {
	vec4 projected = params.projection * vec4(view_pos, 1.0);
	projected.xyz /= projected.w;
	projected.xy = projected.xy * 0.5 + 0.5;
	w = projected.w;
	return projected.xy;
}

#define M_PI 3.14159265359

vec3 reconstructCSPosition(vec2 S, float z) {
	if (params.orthogonal) {
		return vec3((S.xy * params.proj_info.xy + params.proj_info.zw), z);
	} else {
		return vec3((S.xy * params.proj_info.xy + params.proj_info.zw) * z, z);
	}
}

void main() {
	// Pixel being shaded
	ivec2 ssC = ivec2(gl_GlobalInvocationID.xy);

	if (any(greaterThanEqual(ssC, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 pixel_size = 1.0 / vec2(params.screen_size);
	vec2 uv = vec2(ssC) * pixel_size;

	uv += pixel_size * 0.5;

	float base_depth = imageLoad(source_depth, ssC).r;

	// World space point being shaded
	vec3 vertex = reconstructCSPosition(uv * vec2(params.screen_size), base_depth);

	vec4 normal_roughness = imageLoad(source_normal_roughness, ssC);
	vec3 normal = normal_roughness.xyz * 2.0 - 1.0;
	normal = normalize(normal);
	normal.y = -normal.y; //because this code reads flipped

	vec3 view_dir = normalize(vertex);
	vec3 ray_dir = normalize(reflect(view_dir, normal));

	if (dot(ray_dir, normal) < 0.001) {
		imageStore(ssr_image, ssC, vec4(0.0));
		return;
	}
	//ray_dir = normalize(view_dir - normal * dot(normal,view_dir) * 2.0);
	//ray_dir = normalize(vec3(1.0, 1.0, -1.0));

	////////////////

	// make ray length and clip it against the near plane (don't want to trace beyond visible)
	float ray_len = (vertex.z + ray_dir.z * params.camera_z_far) > -params.camera_z_near ? (-params.camera_z_near - vertex.z) / ray_dir.z : params.camera_z_far;
	vec3 ray_end = vertex + ray_dir * ray_len;

	float w_begin;
	vec2 vp_line_begin = view_to_screen(vertex, w_begin);
	float w_end;
	vec2 vp_line_end = view_to_screen(ray_end, w_end);
	vec2 vp_line_dir = vp_line_end - vp_line_begin;

	// we need to interpolate w along the ray, to generate perspective correct reflections
	w_begin = 1.0 / w_begin;
	w_end = 1.0 / w_end;

	float z_begin = vertex.z * w_begin;
	float z_end = ray_end.z * w_end;

	vec2 line_begin = vp_line_begin / pixel_size;
	vec2 line_dir = vp_line_dir / pixel_size;
	float z_dir = z_end - z_begin;
	float w_dir = w_end - w_begin;

	// clip the line to the viewport edges

	float scale_max_x = min(1.0, 0.99 * (1.0 - vp_line_begin.x) / max(1e-5, vp_line_dir.x));
	float scale_max_y = min(1.0, 0.99 * (1.0 - vp_line_begin.y) / max(1e-5, vp_line_dir.y));
	float scale_min_x = min(1.0, 0.99 * vp_line_begin.x / max(1e-5, -vp_line_dir.x));
	float scale_min_y = min(1.0, 0.99 * vp_line_begin.y / max(1e-5, -vp_line_dir.y));
	float line_clip = min(scale_max_x, scale_max_y) * min(scale_min_x, scale_min_y);
	line_dir *= line_clip;
	z_dir *= line_clip;
	w_dir *= line_clip;

	// clip z and w advance to line advance
	vec2 line_advance = normalize(line_dir); // down to pixel
	float step_size = length(line_advance) / length(line_dir);
	float z_advance = z_dir * step_size; // adapt z advance to line advance
	float w_advance = w_dir * step_size; // adapt w advance to line advance

	// make line advance faster if direction is closer to pixel edges (this avoids sampling the same pixel twice)
	float advance_angle_adj = 1.0 / max(abs(line_advance.x), abs(line_advance.y));
	line_advance *= advance_angle_adj; // adapt z advance to line advance
	z_advance *= advance_angle_adj;
	w_advance *= advance_angle_adj;

	vec2 pos = line_begin;
	float z = z_begin;
	float w = w_begin;
	float z_from = z / w;
	float z_to = z_from;
	float depth;
	vec2 prev_pos = pos;

	bool found = false;

	float steps_taken = 0.0;

	for (int i = 0; i < params.num_steps; i++) {
		pos += line_advance;
		z += z_advance;
		w += w_advance;

		// convert to linear depth

		depth = imageLoad(source_depth, ivec2(pos - 0.5)).r;

		z_from = z_to;
		z_to = z / w;

		if (depth > z_to) {
			// if depth was surpassed
			if (depth <= max(z_to, z_from) + params.depth_tolerance && -depth < params.camera_z_far) {
				// check the depth tolerance and far clip
				// check that normal is valid
				found = true;
			}
			break;
		}

		steps_taken += 1.0;
		prev_pos = pos;
	}

	if (found) {
		float margin_blend = 1.0;

		vec2 margin = vec2((params.screen_size.x + params.screen_size.y) * 0.5 * 0.05); // make a uniform margin
		if (any(bvec4(lessThan(pos, -margin), greaterThan(pos, params.screen_size + margin)))) {
			// clip outside screen + margin
			imageStore(ssr_image, ssC, vec4(0.0));
			return;
		}

		{
			//blend fading out towards external margin
			vec2 margin_grad = mix(pos - params.screen_size, -pos, lessThan(pos, vec2(0.0)));
			margin_blend = 1.0 - smoothstep(0.0, margin.x, max(margin_grad.x, margin_grad.y));
			//margin_blend = 1.0;
		}

		vec2 final_pos;
		float grad;
		grad = steps_taken / float(params.num_steps);
		float initial_fade = params.curve_fade_in == 0.0 ? 1.0 : pow(clamp(grad, 0.0, 1.0), params.curve_fade_in);
		float fade = pow(clamp(1.0 - grad, 0.0, 1.0), params.distance_fade) * initial_fade;
		final_pos = pos;

		vec4 final_color;

#ifdef MODE_ROUGH

		// if roughness is enabled, do screen space cone tracing
		float blur_radius = 0.0;
		float roughness = normal_roughness.w;

		if (roughness > 0.001) {
			float cone_angle = min(roughness, 0.999) * M_PI * 0.5;
			float cone_len = length(final_pos - line_begin);
			float op_len = 2.0 * tan(cone_angle) * cone_len; // opposite side of iso triangle
			{
				// fit to sphere inside cone (sphere ends at end of cone), something like this:
				// ___
				// \O/
				//  V
				//
				// as it avoids bleeding from beyond the reflection as much as possible. As a plus
				// it also makes the rough reflection more elongated.
				float a = op_len;
				float h = cone_len;
				float a2 = a * a;
				float fh2 = 4.0f * h * h;
				blur_radius = (a * (sqrt(a2 + fh2) - a)) / (4.0f * h);
			}
		}

		final_color = imageLoad(source_diffuse, ivec2((final_pos - 0.5) * pixel_size));

		imageStore(blur_radius_image, ssC, vec4(blur_radius / 255.0)); //stored in r8

#endif

		final_color = vec4(imageLoad(source_diffuse, ivec2(final_pos - 0.5)).rgb, fade * margin_blend);
		//change blend by metallic
		vec4 metallic_mask = unpackUnorm4x8(params.metallic_mask);
		final_color.a *= dot(metallic_mask, texelFetch(source_metallic, ssC << 1, 0));

		imageStore(ssr_image, ssC, final_color);

	} else {
#ifdef MODE_ROUGH
		imageStore(blur_radius_image, ssC, vec4(0.0));
#endif
		imageStore(ssr_image, ssC, vec4(0.0));
	}
}
