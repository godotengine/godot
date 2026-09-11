#[compute]

#version 450

#VERSION_DEFINES

// https://www.alextardif.com/HistogramLuminance.html
// https://bruop.github.io/exposure/

#define BLOCK_SIZE 16
#define GROUP_SIZE (BLOCK_SIZE * BLOCK_SIZE)

#ifdef HISTOGRAM_PASS
layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;
#else
layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 source_size;
	float min_log_lum;
	float log_lum_range;
	float exposure_adjust;
	float pad[3];
}
params;

layout(set = 0, binding = 0) uniform sampler2D source_texture;

#ifdef HISTOGRAM_PASS
layout(set = 1, binding = 0) restrict writeonly buffer HistogramBuffer {
	uint data[GROUP_SIZE];
}
histogram;

shared uint histogram_shared[GROUP_SIZE];
#else
layout(set = 1, binding = 0) restrict buffer HistogramBuffer {
	uint data[GROUP_SIZE];
}
histogram;

layout(r32f, set = 2, binding = 0) uniform restrict writeonly image2D dest_luminance;

shared float histogram_shared[GROUP_SIZE];
#endif

uint color_to_bin(vec3 hdr_color) {
	// REC-709 luminance weights.
	float lum = dot(hdr_color, vec3(0.2126, 0.7152, 0.0722));

	// Avoid taking the log of zero
	if (lum < 0.005) {
		return 0;
	}

	// Calculate the log_2 luminance and express it as a value in [0.0, 1.0]
	// where 0.0 represents the minimum luminance, and 1.0 represents the max.
	float log_lum = clamp((log2(lum) - params.min_log_lum) / params.log_lum_range, 0.0, 1.0);

	// Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
	return uint(log_lum * 254.0 + 1.0);
}

void main() {
#ifdef HISTOGRAM_PASS
	histogram_shared[gl_LocalInvocationIndex] = 0;

	groupMemoryBarrier();
	barrier();

	ivec2 tex_coords = ivec2(gl_GlobalInvocationID.xy);
	if (all(lessThan(tex_coords, params.source_size))) {
		vec3 source_color = texelFetch(source_texture, tex_coords, 0).rgb;
		uint bin_index = color_to_bin(source_color);
		atomicAdd(histogram_shared[bin_index], 1);
	}

	groupMemoryBarrier();
	barrier();

	atomicAdd(histogram.data[gl_LocalInvocationIndex], histogram_shared[gl_LocalInvocationIndex]);
#else
	float count_for_this_bin = float(histogram.data[gl_LocalInvocationIndex]);
	histogram_shared[gl_LocalInvocationIndex] = count_for_this_bin * float(gl_LocalInvocationIndex);

	groupMemoryBarrier();
	barrier();

	histogram.data[gl_LocalInvocationIndex] = 0;

	// Perform a weighted count of the luminance range.
	for (uint cutoff = (GROUP_SIZE >> 1); cutoff > 0; cutoff >>= 1) {
		if (uint(gl_LocalInvocationIndex) < cutoff) {
			histogram_shared[gl_LocalInvocationIndex] += histogram_shared[gl_LocalInvocationIndex + cutoff];
		}
		groupMemoryBarrier();
		barrier();
	}

	if (gl_LocalInvocationIndex == 0) {
		float pixel_count = float(params.source_size.x * params.source_size.y);
		float weighted_log_average = (histogram_shared[0] / max(pixel_count - count_for_this_bin, 1.0)) - 1.0;

		// Map from our histogram space to actual luminance.
		float weighted_avg_lum = exp2((weighted_log_average / 254.0) * params.log_lum_range + params.min_log_lum);

		// The new stored value will be interpolated using the last frame's value to prevent sudden shifts in the exposure.
		float prev_lum = texelFetch(source_texture, ivec2(0, 0), 0).x;
		float adapted_lum = prev_lum + (weighted_avg_lum - prev_lum) * params.exposure_adjust;
		imageStore(dest_luminance, ivec2(0, 0), vec4(adapted_lum));
	}
#endif
}
