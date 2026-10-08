#version 450

#extension GL_EXT_ray_query : require

// Hybrid ray-traced ambient occlusion. Replaces screen-space AO when the device supports ray query.
// Normals are reconstructed from the depth buffer, so no extra G-buffer input is needed.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform texture2D depth_texture;
layout(set = 0, binding = 1) uniform accelerationStructureEXT scene_tlas;
layout(r8, set = 0, binding = 2) uniform restrict writeonly image2D ao_image;

// Push constants are kept within the 128-byte minimum guaranteed by Vulkan (124 bytes).
// world_from_view_rows holds the 3x4 world-from-view transform: xyz is the basis row, w is the origin component.
layout(push_constant, std430) uniform Params {
	mat4 inv_projection;
	vec4 world_from_view_rows[3];
	float radius;
	uint sample_count;
	uint frame;
}
params;

vec3 to_world_point(vec3 v) {
	return vec3(dot(params.world_from_view_rows[0].xyz, v) + params.world_from_view_rows[0].w,
			dot(params.world_from_view_rows[1].xyz, v) + params.world_from_view_rows[1].w,
			dot(params.world_from_view_rows[2].xyz, v) + params.world_from_view_rows[2].w);
}

vec3 to_world_direction(vec3 v) {
	return vec3(dot(params.world_from_view_rows[0].xyz, v),
			dot(params.world_from_view_rows[1].xyz, v),
			dot(params.world_from_view_rows[2].xyz, v));
}

uint pcg_hash(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float random_float(inout uint seed) {
	seed = pcg_hash(seed);
	return float(seed) / 4294967295.0;
}

vec3 view_position(ivec2 pixel) {
	ivec2 size = textureSize(depth_texture, 0);
	ivec2 clamped = clamp(pixel, ivec2(0), size - ivec2(1));
	vec2 uv = (vec2(clamped) + 0.5) / vec2(size);
	float depth = texelFetch(depth_texture, clamped, 0).r;
	vec4 view = params.inv_projection * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return view.xyz / view.w;
}

vec3 cosine_hemisphere(vec2 xi, vec3 n) {
	float r = sqrt(xi.x);
	float phi = 6.28318530718 * xi.y;
	vec3 d = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
	vec3 t = normalize(abs(n.x) > 0.5 ? cross(n, vec3(0.0, 1.0, 0.0)) : cross(n, vec3(1.0, 0.0, 0.0)));
	vec3 b = cross(n, t);
	return normalize(t * d.x + b * d.y + n * d.z);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = textureSize(depth_texture, 0);
	if (pixel.x >= size.x || pixel.y >= size.y) {
		return;
	}

	float depth = texelFetch(depth_texture, pixel, 0).r;
	if (depth >= 1.0) {
		// Sky or empty pixel: no occlusion.
		imageStore(ao_image, pixel, vec4(1.0));
		return;
	}

	vec3 p = view_position(pixel);
	vec3 px = view_position(pixel + ivec2(1, 0));
	vec3 py = view_position(pixel + ivec2(0, 1));
	vec3 n_view = normalize(cross(px - p, py - p));
	if (dot(n_view, p) > 0.0) {
		n_view = -n_view; // Face the camera.
	}

	vec3 origin = to_world_point(p);
	vec3 normal = normalize(to_world_direction(n_view));
	origin += normal * 0.01; // Small offset against self-intersection.

	uint seed = pcg_hash(uint(pixel.x) + uint(pixel.y) * 65536u + params.frame * 16777619u);
	float occluded = 0.0;
	uint samples = max(params.sample_count, 1u);
	for (uint i = 0u; i < samples; i++) {
		vec2 xi = vec2(random_float(seed), random_float(seed));
		vec3 dir = cosine_hemisphere(xi, normal);

		rayQueryEXT rq;
		rayQueryInitializeEXT(rq, scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.001, dir, params.radius);
		while (rayQueryProceedEXT(rq)) {
		}
		if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
			occluded += 1.0;
		}
	}

	float ao = 1.0 - occluded / float(samples);
	imageStore(ao_image, pixel, vec4(ao));
}
