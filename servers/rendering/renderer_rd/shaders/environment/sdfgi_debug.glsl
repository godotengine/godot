#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define MAX_CASCADES 8

layout(set = 0, binding = 1) uniform texture3D sdf_cascades[MAX_CASCADES];
layout(set = 0, binding = 2) uniform texture3D light_cascades[MAX_CASCADES];
layout(set = 0, binding = 3) uniform texture3D aniso0_cascades[MAX_CASCADES];
layout(set = 0, binding = 4) uniform texture3D aniso1_cascades[MAX_CASCADES];
layout(set = 0, binding = 5) uniform texture3D occlusion_texture;

layout(set = 0, binding = 8) uniform sampler linear_sampler;

struct CascadeData {
	vec3 offset; //offset of (0,0,0) in world coordinates
	float to_cell; // 1/bounds * grid_size
	ivec3 probe_world_offset;
	uint pad;
	vec4 pad2;
};

layout(set = 0, binding = 9, std140) uniform Cascades {
	CascadeData data[MAX_CASCADES];
}
cascades;

layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D screen_buffer;

layout(set = 0, binding = 11) uniform texture2DArray lightprobe_texture;

layout(push_constant, std430) uniform Params {
	vec3 grid_size;
	uint max_cascades;

	ivec2 screen_size;
	float y_mult;

	float z_near;

	mat3x4 inv_projection;
	// We pack these more tightly than mat3 and vec3, which will require some reconstruction trickery.
	float cam_basis[3][3];
	float cam_origin[3];
}
params;

vec3 linear_to_srgb(vec3 color) {
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

void main() {
	// Pixel being shaded
	ivec2 screen_pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(screen_pos, params.screen_size))) { //too large, do nothing
		return;
	}

	vec3 ray_pos;
	vec3 ray_dir;
	{
		ray_pos = vec3(params.cam_origin[0], params.cam_origin[1], params.cam_origin[2]);

		ray_dir.xy = ((vec2(screen_pos) / vec2(params.screen_size)) * 2.0 - 1.0);
		ray_dir.z = params.z_near;

		ray_dir = (vec4(ray_dir, 1.0) * mat4(params.inv_projection)).xyz;

		mat3 cam_basis;
		{
			vec3 c0 = vec3(params.cam_basis[0][0], params.cam_basis[0][1], params.cam_basis[0][2]);
			vec3 c1 = vec3(params.cam_basis[1][0], params.cam_basis[1][1], params.cam_basis[1][2]);
			vec3 c2 = vec3(params.cam_basis[2][0], params.cam_basis[2][1], params.cam_basis[2][2]);
			cam_basis = mat3(c0, c1, c2);
		}
		ray_dir = normalize(cam_basis * ray_dir);
	}

	ray_pos.y *= params.y_mult;
	ray_dir.y *= params.y_mult;
	ray_dir = normalize(ray_dir);

	vec3 pos_to_uvw = 1.0 / params.grid_size;

	vec3 light = vec3(0.0);
	float blend = 0.0;

#if 1
	// No interpolation

	vec3 inv_dir = 1.0 / ray_dir;

	float rough = 0.5;
	bool hit = false;

	for (uint i = 0; i < params.max_cascades; i++) {
		//convert to local bounds
		vec3 pos = ray_pos - cascades.data[i].offset;
		pos *= cascades.data[i].to_cell;

		// Should never happen for debug, since we start mostly at the bounds center,
		// but add anyway.
		//if (any(lessThan(pos,vec3(0.0))) || any(greaterThanEqual(pos,params.grid_size))) {
		//	continue; //already past bounds for this cascade, goto next
		//}

		//find maximum advance distance (until reaching bounds)
		vec3 t0 = -pos * inv_dir;
		vec3 t1 = (params.grid_size - pos) * inv_dir;
		vec3 tmax = max(t0, t1);
		float max_advance = min(tmax.x, min(tmax.y, tmax.z));

		float advance = 0.0;
		vec3 uvw;
		hit = false;

		while (advance < max_advance) {
			//read how much to advance from SDF
			uvw = (pos + ray_dir * advance) * pos_to_uvw;

			float distance = texture(sampler3D(sdf_cascades[i], linear_sampler), uvw).r * 255.0 - 1.7;

			if (distance < 0.001) {
				//consider hit
				hit = true;
				break;
			}

			advance += distance;
		}

		if (!hit) {
			pos += ray_dir * min(advance, max_advance);
			pos /= cascades.data[i].to_cell;
			pos += cascades.data[i].offset;
			ray_pos = pos;
			continue;
		}

		//compute albedo, emission and normal at hit point

		const float EPSILON = 0.001;
		vec3 hit_normal = normalize(vec3(
				texture(sampler3D(sdf_cascades[i], linear_sampler), uvw + vec3(EPSILON, 0.0, 0.0)).r - texture(sampler3D(sdf_cascades[i], linear_sampler), uvw - vec3(EPSILON, 0.0, 0.0)).r,
				texture(sampler3D(sdf_cascades[i], linear_sampler), uvw + vec3(0.0, EPSILON, 0.0)).r - texture(sampler3D(sdf_cascades[i], linear_sampler), uvw - vec3(0.0, EPSILON, 0.0)).r,
				texture(sampler3D(sdf_cascades[i], linear_sampler), uvw + vec3(0.0, 0.0, EPSILON)).r - texture(sampler3D(sdf_cascades[i], linear_sampler), uvw - vec3(0.0, 0.0, EPSILON)).r));

		vec3 hit_light = texture(sampler3D(light_cascades[i], linear_sampler), uvw).rgb;
		vec4 aniso0 = texture(sampler3D(aniso0_cascades[i], linear_sampler), uvw);
		vec3 hit_aniso0 = aniso0.rgb;
		vec3 hit_aniso1 = vec3(aniso0.a, texture(sampler3D(aniso1_cascades[i], linear_sampler), uvw).rg);

		hit_light *= (dot(max(vec3(0.0), (hit_normal * hit_aniso0)), vec3(1.0)) + dot(max(vec3(0.0), (-hit_normal * hit_aniso1)), vec3(1.0)));

		light = hit_light;

		break;
	}

#endif

	imageStore(screen_buffer, screen_pos, vec4(clamp(linear_to_srgb(light), vec3(0.0), vec3(1.0)), 1.0));
}
