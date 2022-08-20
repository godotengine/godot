#[vertex]

#version 450

#VERSION_DEFINES

layout(push_constant, std430) uniform Params {
	float z_far;
	float z_near;
	vec2 texel_size;
	vec4 screen_rect;
}
params;

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];
	vec2 screen_pos = uv_interp * params.screen_rect.zw + params.screen_rect.xy;
	gl_Position = vec4(screen_pos * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform samplerCube source_cube;

layout(push_constant, std430) uniform Params {
	float z_far;
	float z_near;
	vec2 texel_size;
	vec4 screen_rect;
}
params;

void main() {
	vec2 uv = uv_interp;
	vec2 texel_size = abs(params.texel_size);

	uv = clamp(uv * (1.0 + 2.0 * texel_size) - texel_size, vec2(0.0), vec2(1.0));

	vec3 normal = vec3(uv * 2.0 - 1.0, 0.0);
	normal.z = 0.5 * (1.0 - dot(normal.xy, normal.xy)); // z = 1/2 - 1/2 * (x^2 + y^2)
	normal = normalize(normal);

	normal.y = -normal.y; //needs to be flipped to match projection matrix
	if (params.texel_size.x >= 0.0) { // Sign is used to encode Z flip
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

	gl_FragDepth = depth;
}
