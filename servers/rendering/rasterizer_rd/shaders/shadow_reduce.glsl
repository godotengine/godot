#[compute]

#version 450

VERSION_DEFINES

#define BLOCK_SIZE 8

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

#ifdef MODE_REDUCE

shared float tmp_data[BLOCK_SIZE * BLOCK_SIZE];
const uint swizzle_table[BLOCK_SIZE] = uint[](0, 4, 2, 6, 1, 5, 3, 7);
const uint unswizzle_table[BLOCK_SIZE] = uint[](0, 0, 0, 1, 0, 2, 1, 3);

#endif

layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source_depth;
layout(r32f, set = 0, binding = 1) uniform restrict writeonly image2D dst_depth;

layout(push_constant, binding = 1, std430) uniform Params {
	ivec2 source_size;
	ivec2 source_offset;
	uint min_size;
	uint gaussian_kernel_version;
	ivec2 filter_dir;
}
params;

void main() {
#ifdef MODE_REDUCE

	uvec2 pos = gl_LocalInvocationID.xy;

	ivec2 image_offset = params.source_offset;
	ivec2 image_pos = image_offset + ivec2(gl_GlobalInvocationID.xy);
	uint dst_t = swizzle_table[pos.y] * BLOCK_SIZE + swizzle_table[pos.x];
	tmp_data[dst_t] = imageLoad(source_depth, min(image_pos, params.source_size - ivec2(1))).r;
	ivec2 image_size = params.source_size;

	uint t = pos.y * BLOCK_SIZE + pos.x;

	//neighbours
	uint size = BLOCK_SIZE;

	do {
		groupMemoryBarrier();
		barrier();

		size >>= 1;
		image_size >>= 1;
		image_offset >>= 1;

		if (all(lessThan(pos, uvec2(size)))) {
			uint nx = t + size;
			uint ny = t + (BLOCK_SIZE * size);
			uint nxy = ny + size;

			tmp_data[t] += tmp_data[nx];
			tmp_data[t] += tmp_data[ny];
			tmp_data[t] += tmp_data[nxy];
			tmp_data[t] /= 4.0;
		}

	} while (size > params.min_size);

	if (all(lessThan(pos, uvec2(size)))) {
		image_pos = ivec2(unswizzle_table[size + pos.x], unswizzle_table[size + pos.y]);
		image_pos += image_offset + ivec2(gl_WorkGroupID.xy) * int(size);

		image_size = max(ivec2(1), image_size); //in case image size became 0

		if (all(lessThan(image_pos, uvec2(image_size)))) {
			imageStore(dst_depth, image_pos, vec4(tmp_data[t]));
		}
	}
#endif

#ifdef MODE_FILTER

	ivec2 image_pos = params.source_offset + ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(image_pos, params.source_size))) {
		return;
	}

	ivec2 clamp_min = ivec2(params.source_offset);
	ivec2 clamp_max = ivec2(params.source_size) - 1;

	//gaussian kernel, size 9, sigma 4
	const int kernel_size = 9;
	const float gaussian_kernel[kernel_size * 3] = float[](
			0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229,
			0.028532, 0.067234, 0.124009, 0.179044, 0.20236, 0.179044, 0.124009, 0.067234, 0.028532,
			0.081812, 0.101701, 0.118804, 0.130417, 0.134535, 0.130417, 0.118804, 0.101701, 0.081812);
	float accum = 0.0;
	for (int i = 0; i < kernel_size; i++) {
		ivec2 ofs = clamp(image_pos + params.filter_dir * (i - kernel_size / 2), clamp_min, clamp_max);
		accum += imageLoad(source_depth, ofs).r * gaussian_kernel[params.gaussian_kernel_version + i];
	}

	imageStore(dst_depth, image_pos, vec4(accum));

#endif
}
