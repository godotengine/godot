#[compute]

#version 450

#VERSION_DEFINES

#define BLOCK_SIZE 16

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

// Use for reading from screen and from previous luminance.
layout(set = 0, binding = 0) uniform sampler2D source_texture;

#ifdef READ_TEXTURE
layout(set = 1, binding = 0) buffer restrict writeonly Histogram {
	uint data[256];
}
histogram;
shared uint tmp_data[BLOCK_SIZE * BLOCK_SIZE];
#else

layout(set = 1, binding = 0) buffer restrict Histogram {
	uint data[256];
}
histogram;
layout(r32f, set = 2, binding = 0) uniform restrict writeonly image2D dest_luminance;
shared float tmp_data[BLOCK_SIZE * BLOCK_SIZE];
#endif

layout(push_constant, std430) uniform Params {
	ivec2 source_size;
	float max_luminance;
	float min_luminance;
	float exposure_adjust;
	float pad[3];
}
params;

uint color_to_histogram_bin(vec3 p_color) {
	float lum = dot(p_color, vec3(0.2125, 0.7154, 0.0721));

	// Purposfully use big epsilon to capture values near zero and round down.
	if (lum < 0.005) {
		return 0;
	}

	float log_lum = clamp((log2(lum) - params.min_luminance) * params.max_luminance, 0.0, 1.0);

	// Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
	return uint(log_lum * 254.0 + 1.0);
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

#ifdef READ_TEXTURE
	tmp_data[gl_LocalInvocationIndex] = 0;

	groupMemoryBarrier();
	barrier();

	if (any(lessThan(pos, params.source_size))) {
		vec3 col = texelFetch(source_texture, pos, 0).rgb;
		uint idx = color_to_histogram_bin(col);
		atomicAdd(tmp_data[idx], 1);
	}

	groupMemoryBarrier();
	barrier();

	atomicAdd(histogram.data[gl_LocalInvocationIndex], tmp_data[gl_LocalInvocationIndex]);
#else
	uint count = histogram.data[gl_LocalInvocationIndex];
	tmp_data[gl_LocalInvocationIndex] = float(count) * float(gl_LocalInvocationIndex); // assign more weight to higher values

	groupMemoryBarrier();
	barrier();

	for (uint size = (BLOCK_SIZE * BLOCK_SIZE) >> 1; size >= 1; size >>= 1) {
		if (gl_LocalInvocationIndex < size) {
			tmp_data[gl_LocalInvocationIndex] += tmp_data[gl_LocalInvocationIndex + size];
		}
		groupMemoryBarrier();
		barrier();
	}

	histogram.data[gl_LocalInvocationIndex] = 0;

	if (gl_LocalInvocationIndex == 0) {
		float weighted_log_average = (tmp_data[0] / max((params.source_size.x * params.source_size.y) - float(count), 1.0)) - 1.0;

		// Map from our histogram space to actual luminance
		float weighted_average = exp2(((weighted_log_average / 254.0) * params.max_luminance) + params.min_luminance);

		float prev_lum = texelFetch(source_texture, ivec2(0, 0), 0).r; //1 pixel previous exposure
		weighted_average = prev_lum + (weighted_average - prev_lum) * params.exposure_adjust;
		imageStore(dest_luminance, ivec2(0, 0), vec4(weighted_average));
	}
#endif
}
