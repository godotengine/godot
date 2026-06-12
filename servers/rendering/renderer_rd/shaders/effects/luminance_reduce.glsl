#[compute]

#version 450

#VERSION_DEFINES

#extension GL_KHR_shader_subgroup_arithmetic : enable

#define BLOCK_SIZE 8u

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

shared float tmp_data[BLOCK_SIZE * BLOCK_SIZE];

shared float accumulator[BLOCK_SIZE * BLOCK_SIZE];

#ifdef READ_TEXTURE

//use for main texture
layout(set = 0, binding = 0) uniform sampler2D source_texture;

#else

//use for intermediate textures
layout(r32f, set = 0, binding = 0) uniform restrict readonly image2D source_luminance;

#endif

layout(r32f, set = 1, binding = 0) uniform restrict writeonly image2D dest_luminance;

#ifdef WRITE_LUMINANCE
layout(set = 2, binding = 0) uniform sampler2D prev_luminance;
#endif

layout(push_constant, std430) uniform Params {
	ivec2 source_size;
	float max_luminance;
	float min_luminance;
	float exposure_adjust;
	float pad[3];
}
params;

float sharedmem_reduction(float fetch, ivec2 pos) {
	// Ensure that subgroup == 0 with subgroupinvocation == 0 will return summation
	const uint subgroup_size = (BLOCK_SIZE * BLOCK_SIZE) / gl_NumSubgroups;
	uint t = gl_SubgroupInvocationID + (gl_SubgroupID * subgroup_size);

	tmp_data[t] = fetch;
	barrier();

#pragma unroll
	for (uint size = (BLOCK_SIZE * BLOCK_SIZE) >> 1; size > 0u; size >>= 1) {
		if (t < size) {
			tmp_data[t] += tmp_data[t + size];
		}
		barrier();
	}

	return tmp_data[0];
}

float subgroup_reduction(float fetch, ivec2 pos) {
	const uint subgroup_size = (BLOCK_SIZE * BLOCK_SIZE) / gl_NumSubgroups;

	// Failsafe max to avoid infinite loops
	const uint shift = uint(max(findMSB(subgroup_size), 1));

	// Ensure subgroup has finished fetch before subgroup op
	subgroupBarrier();
	float avg_numerator = subgroupAdd(fetch);

	// Shared memory fetch location ensuring group 0 fetches [0,...N-1]
	const uint accumulate_index = gl_SubgroupInvocationID + (gl_SubgroupID * subgroup_size);

	// While more than 1 group has values to share
	for (uint remaining_size = (BLOCK_SIZE * BLOCK_SIZE) >> shift; remaining_size > 1u; remaining_size >>= shift) {
		// Write subgroup's value to shared memory -- elect worker 0
		if (gl_SubgroupInvocationID == 0u) {
			accumulator[gl_SubgroupID] = avg_numerator;
		}
		// Wait for memory sync within workgroup
		barrier();

		// Zero all work items
		avg_numerator = 0.;
		// Read into active workitems only
		if (accumulate_index < remaining_size) {
			avg_numerator = accumulator[accumulate_index];
		}

		// Wait for subgroup to finish reading
		subgroupBarrier();
		// Accumulate within the subgroups
		avg_numerator = subgroupAdd(avg_numerator);

		// Wait for all work items to finish read before writing
		// to shared memory -- if necessary
		if (remaining_size > subgroup_size) {
			barrier();
		}
	}

	return avg_numerator;
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	float avg_numerator = 0.;

	// Initialize to zero
	float fetch = 0.;
	// Update if active
	if (all(lessThan(pos, params.source_size))) {
#ifdef READ_TEXTURE
		vec3 v = texelFetch(source_texture, pos, 0).rgb;
		fetch = max(v.r, max(v.g, v.b));
#else
		fetch = imageLoad(source_luminance, pos).r;
#endif
	}

	// Do binary tree reduction in shared memory if subgroup size is 1 or 2
	// otherwise do register shuffle plan
	if (gl_NumSubgroups > ((BLOCK_SIZE * BLOCK_SIZE) >> 2u)) {
		avg_numerator = sharedmem_reduction(fetch, pos);
	} else {
		avg_numerator = subgroup_reduction(fetch, pos);
	}

	// Now subgroup 0 is guaranteed to have avg_numerator = sum of pixel luminances
	if ((gl_SubgroupID == 0u) && (gl_SubgroupInvocationID == 0u)) {
		// compute rect size w.r.t. work group corner (no guarantee worker 0 executes here)
		ivec2 pos_0 = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy);
		ivec2 rect_size = min(params.source_size - pos_0, ivec2(BLOCK_SIZE));
		float avg = avg_numerator / float(rect_size.x * rect_size.y);
		pos /= ivec2(BLOCK_SIZE);
#ifdef WRITE_LUMINANCE
		float prev_lum = texelFetch(prev_luminance, ivec2(0, 0), 0).r; //1 pixel previous exposure
		avg = clamp(prev_lum + (avg - prev_lum) * params.exposure_adjust, params.min_luminance, params.max_luminance);
#endif
		imageStore(dest_luminance, pos, vec4(avg));
	}
}
