#[compute]

#version 450

VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(push_constant, binding = 1, std430) uniform Params {
	ivec2 screen_size;
	ivec2 offset;
	float bias;
	float z_far;
	float z_near;
	bool z_flip;
}
params;

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D depth_buffer;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThan(pos, params.screen_size))) { //too large, do nothing
		return;
	}

	vec2 pixel_size = 1.0 / vec2(params.screen_size);
	vec2 uv = (vec2(pos) + 0.5) * pixel_size;

	vec3 normal = vec3(uv * 2.0 - 1.0, 0.0);

	normal.z = 0.5 - 0.5 * ((normal.x * normal.x) + (normal.y * normal.y));
	normal = normalize(normal);

	normal.y = -normal.y; //needs to be flipped to match projection matrix
	if (!params.z_flip) {
		normal.z = -normal.z;
	}

	float depth = texture(source_cube, normal).r;

	// absolute values for direction cosines, bigger value equals closer to basis axis
	vec3 unorm = abs(normal);

	if ((unorm.x >= unorm.y) && (unorm.x >= unorm.z)) {
		// x code
		unorm = normal.x > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(-1.0, 0.0, 0.0);
	} else if ((unorm.y > unorm.x) && (unorm.y >= unorm.z)) {
		// y code
		unorm = normal.y > 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
	} else if ((unorm.z > unorm.x) && (unorm.z > unorm.y)) {
		// z code
		unorm = normal.z > 0.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
	} else {
		// oh-no we messed up code
		// has to be
		unorm = vec3(1.0, 0.0, 0.0);
	}

	float depth_fix = 1.0 / dot(normal, unorm);

	depth = 2.0 * depth - 1.0;
	float linear_depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	depth = (linear_depth * depth_fix) / params.z_far;

	imageStore(depth_buffer, pos + params.offset, vec4(depth));
}
