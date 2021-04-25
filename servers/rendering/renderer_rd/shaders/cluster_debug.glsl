#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const vec3 usage_gradient[33] = vec3[]( // 1 (none) + 32
		vec3(0.14, 0.17, 0.23),
		vec3(0.24, 0.44, 0.83),
		vec3(0.23, 0.57, 0.84),
		vec3(0.22, 0.71, 0.84),
		vec3(0.22, 0.85, 0.83),
		vec3(0.21, 0.85, 0.72),
		vec3(0.21, 0.85, 0.57),
		vec3(0.20, 0.85, 0.42),
		vec3(0.20, 0.85, 0.27),
		vec3(0.27, 0.86, 0.19),
		vec3(0.51, 0.85, 0.19),
		vec3(0.57, 0.86, 0.19),
		vec3(0.62, 0.85, 0.19),
		vec3(0.67, 0.86, 0.20),
		vec3(0.73, 0.85, 0.20),
		vec3(0.78, 0.85, 0.20),
		vec3(0.83, 0.85, 0.20),
		vec3(0.85, 0.82, 0.20),
		vec3(0.85, 0.76, 0.20),
		vec3(0.85, 0.81, 0.20),
		vec3(0.85, 0.65, 0.20),
		vec3(0.84, 0.60, 0.21),
		vec3(0.84, 0.56, 0.21),
		vec3(0.84, 0.51, 0.21),
		vec3(0.84, 0.46, 0.21),
		vec3(0.84, 0.41, 0.21),
		vec3(0.84, 0.36, 0.21),
		vec3(0.84, 0.31, 0.21),
		vec3(0.84, 0.27, 0.21),
		vec3(0.83, 0.22, 0.22),
		vec3(0.83, 0.22, 0.27),
		vec3(0.83, 0.22, 0.32),
		vec3(1.00, 0.63, 0.70));
layout(push_constant, binding = 0, std430) uniform Params {
	uvec2 screen_size;
	uvec2 cluster_screen_size;

	uint cluster_shift;
	uint cluster_type;
	float z_near;
	float z_far;

	bool orthogonal;
	uint max_cluster_element_count_div_32;
	uint pad1;
	uint pad2;
}
params;

layout(set = 0, binding = 1, std430) buffer restrict readonly ClusterData {
	uint data[];
}
cluster_data;

layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D screen_buffer;
layout(set = 0, binding = 3) uniform texture2D depth_buffer;
layout(set = 0, binding = 4) uniform sampler depth_buffer_sampler;

void main() {
	uvec2 screen_pos = gl_GlobalInvocationID.xy;
	if (any(greaterThanEqual(screen_pos, params.screen_size))) {
		return;
	}

	uvec2 cluster_pos = screen_pos >> params.cluster_shift;

	uint offset = cluster_pos.y * params.cluster_screen_size.x + cluster_pos.x;
	offset += params.cluster_screen_size.x * params.cluster_screen_size.y * params.cluster_type;
	offset *= (params.max_cluster_element_count_div_32 + 32);

	//depth buffers generally can't be accessed via image API
	float depth = texelFetch(sampler2D(depth_buffer, depth_buffer_sampler), ivec2(screen_pos), 0).r * 2.0 - 1.0;

	if (params.orthogonal) {
		depth = ((depth + (params.z_far + params.z_near) / (params.z_far - params.z_near)) * (params.z_far - params.z_near)) / 2.0;
	} else {
		depth = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near - depth * (params.z_far - params.z_near));
	}
	depth /= params.z_far;

	uint slice = uint(clamp(floor(depth * 32.0), 0.0, 31.0));
	uint slice_minmax = cluster_data.data[offset + params.max_cluster_element_count_div_32 + slice];
	uint item_min = slice_minmax & 0xFFFF;
	uint item_max = slice_minmax >> 16;

	uint item_count = 0;
	for (uint i = 0; i < params.max_cluster_element_count_div_32; i++) {
		uint slice_bits = cluster_data.data[offset + i];
		while (slice_bits != 0) {
			uint bit = findLSB(slice_bits);
			uint item = i * 32 + bit;
			if ((item >= item_min && item < item_max)) {
				item_count++;
			}
			slice_bits &= ~(1 << bit);
		}
	}

	item_count = min(item_count, 32);

	vec3 color = usage_gradient[item_count];

	color = mix(color * 1.2, color * 0.3, float(slice) / 31.0);

	imageStore(screen_buffer, ivec2(screen_pos), vec4(color, 1.0));
}
