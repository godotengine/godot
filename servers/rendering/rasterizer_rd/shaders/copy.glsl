/* clang-format off */
[vertex]

#version 450

VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;
/* clang-format on */

void main() {

	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];

	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);
}

/* clang-format off */
[fragment]

#version 450

VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;
/* clang-format on */

#ifdef MODE_CUBE_TO_DP

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(push_constant, binding = 0, std430) uniform Params {
	float bias;
	float z_far;
	float z_near;
	bool z_flip;
}
params;

layout(location = 0) out float depth_buffer;

#endif

void main() {

#ifdef MODE_CUBE_TO_DP

	vec3 normal = vec3(uv_interp * 2.0 - 1.0, 0.0);

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
	depth_buffer = (linear_depth * depth_fix + params.bias) / params.z_far;

#endif
}
