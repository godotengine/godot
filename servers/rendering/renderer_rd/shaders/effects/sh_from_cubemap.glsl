#[compute]

#version 450

#VERSION_DEFINES

// Uncomment this to use the slow version, which will compute everything in one thread.
// This is useful to debug problems when you suspect the issue is in the parallel reduction algorithm.
// #define SH_DEBUG_MODE

// IMPORTANT: Increasing the sample count beyond 1024 can cause out-of-shared LDS memory on weak hardware.
// To increase this value beyond 1024, small changes would be required (compute more samples per thread,
// use multiple compute dispatches, or some clever thread group synchronization.
// Basically beyond 1024 there are performance tradeoffs to consider).
#define SAMPLE_COUNT 1024u

#define NUM_THREADS SAMPLE_COUNT

#define M_PI 3.14159265359

#ifdef SH_DEBUG_MODE
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = SAMPLE_COUNT, local_size_y = 1, local_size_z = 1) in;
#endif

layout(set = 0, binding = 0) uniform samplerCube source_cubemap;
layout(set = 0, binding = 1, std430) restrict writeonly buffer SHCoeffs {
#ifdef REFERENCE_SH_IMPL
	// Reference implementation. Not used.
	vec4 coeffs[3];
#else
	// Optimized version, baking as much as possible for evaluate_sh_l1_geomerics().
	vec4 coeffs[5];
#endif
}
sh;

// For SAMPLE_COUNT = 1024u, 12KB of memory. Fits on every GPU model (smallest is 16KB).
// The data is encoded in 16-bit SNORM to save space.
shared uint shared_lds[NUM_THREADS / 2u][6u];

// Fibonacci Sphere (Sampling).
vec3 to_vector(uint sample_idx_uint) {
	const mediump float sample_idx = float(sample_idx_uint);
	const mediump float offset = 2.0 / SAMPLE_COUNT;
	const mediump float increment = 2.39996322972865332223155550663361f; // Golden Angle.
	mediump vec3 dir;
	dir.y = (sample_idx * offset) - 1.0f + offset * 0.5f;
	const mediump float r = sqrt(1.0 - dir.y * dir.y);
	const mediump float phi = sample_idx * increment;
	dir.x = cos(phi) * r;
	dir.z = sin(phi) * r;
	return dir;
}

#define SAVE_TO_LDS(idx)                                          \
	shared_lds[idx][0] = packSnorm2x16(resl0.xy);                 \
	shared_lds[idx][1] = packSnorm2x16(vec2(resl0.z, resl1n.x));  \
	shared_lds[idx][2] = packSnorm2x16(resl1n.yz);                \
	shared_lds[idx][3] = packSnorm2x16(resl10.xy);                \
	shared_lds[idx][4] = packSnorm2x16(vec2(resl10.z, resl1p.x)); \
	shared_lds[idx][5] = packSnorm2x16(resl1p.yz)

#define LOAD_FROM_LDS(idx)                           \
	vec2 tmp0, tmp1;                                 \
	tmp0 = unpackSnorm2x16(shared_lds[idx][0]);      \
	tmp1 = unpackSnorm2x16(shared_lds[idx][1]);      \
	const vec3 other_resl0 = vec3(tmp0.xy, tmp1.x);  \
	tmp0 = unpackSnorm2x16(shared_lds[idx][2]);      \
	const vec3 other_resl1n = vec3(tmp1.y, tmp0.xy); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][3]);      \
	tmp1 = unpackSnorm2x16(shared_lds[idx][4]);      \
	const vec3 other_resl10 = vec3(tmp0.xy, tmp1.x); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][5]);      \
	const vec3 other_resl1p = vec3(tmp1.y, tmp0.xy)

void store_data(vec3 resl0, vec3 resl1n, vec3 resl10, vec3 resl1p) {
	// We need to undo the normalization to reconstruct the original colour's brightness from
	// the original cubemap implementation.
	// TODO: Should we really denormalize? A normalized-cube means it's energy-conserving
	// (though this SH has been normalized over a sphere, not hemisphere).
	resl0 *= 1.0 / 0.28209479177387814;
	resl1n *= 1.0 / 0.4886025119029199;
	resl10 *= 1.0 / 0.4886025119029199;
	resl1p *= 1.0 / 0.4886025119029199;

#ifdef REFERENCE_SH_IMPL
	// Is this the best encoding? For scalar (Desktop) GPUs it doesn't matter, so this arrangement
	// makes perfect use of contiguous 48 bytes.
	//
	// However VLIW archs (like found on not-so-older Android) could benefit from using
	// vec4 coeffs[4] and waste coeffs[i].w.
	sh.coeffs[0].x = resl0.x;
	sh.coeffs[1].x = resl0.y;
	sh.coeffs[2].x = resl0.z;

	sh.coeffs[0].y = resl1n.x;
	sh.coeffs[1].y = resl1n.y;
	sh.coeffs[2].y = resl1n.z;

	sh.coeffs[0].z = resl10.x;
	sh.coeffs[1].z = resl10.y;
	sh.coeffs[2].z = resl10.z;

	sh.coeffs[0].w = resl1p.x;
	sh.coeffs[1].w = resl1p.y;
	sh.coeffs[2].w = resl1p.z;
#else
	const mediump vec3 R0 = resl0;
	mediump vec3 R1[3]; // R1[channel].
	R1[0] = 0.5f * vec3(resl1p[0], resl1n[0], resl10[0]);
	R1[1] = 0.5f * vec3(resl1p[1], resl1n[1], resl10[1]);
	R1[2] = 0.5f * vec3(resl1p[2], resl1n[2], resl10[2]);
	const mediump vec3 lenR1 = vec3(length(R1[0]), length(R1[1]), length(R1[2]));
	const mediump vec3 p = 1.0f + 2.0f * lenR1 / R0;
	const mediump vec3 a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

	R1[0] = R1[0] / lenR1[0];
	R1[1] = R1[1] / lenR1[1];
	R1[2] = R1[2] / lenR1[2];

	sh.coeffs[0].xyzw = vec4(R0.xyz, R1[0].x);
	sh.coeffs[1].xyzw = vec4(R1[0].yz, R1[1].xy);
	sh.coeffs[2].xyzw = vec4(R1[1].z, R1[2].xyz);
	sh.coeffs[3].xyzw = vec4(p.xyz, a.x);
	sh.coeffs[4].xyzw = vec4(a.yz, 0.0f, 0.0f);
#endif
}

void main() {
#ifdef SH_DEBUG_MODE
	vec3 resl0 = vec3(0.0);
	vec3 resl1n = vec3(0.0);
	vec3 resl10 = vec3(0.0);
	vec3 resl1p = vec3(0.0);

	for (uint _thread_id = 0u; _thread_id < SAMPLE_COUNT; ++_thread_id) {
		const uint thread_idx = _thread_id;
		const vec3 dir = to_vector(thread_idx);
		const vec3 col = textureLod(source_cubemap, dir, 0.0).xyz;
		resl0 += col * 0.28209479177387814;
		resl1n += col * 0.4886025119029199 * dir.y;
		resl10 += col * 0.4886025119029199 * dir.z;
		resl1p += col * 0.4886025119029199 * dir.x;
	}

	resl0 /= SAMPLE_COUNT;
	resl1n /= SAMPLE_COUNT;
	resl10 /= SAMPLE_COUNT;
	resl1p /= SAMPLE_COUNT;

	store_data(resl0, resl1n, resl10, resl1p);
#else
	const uint thread_idx = gl_LocalInvocationIndex;
	const vec3 dir = to_vector(thread_idx);

	const vec3 col = textureLod(source_cubemap, dir, 0.0).xyz;

	vec3 resl0 = col * 0.28209479177387814;
	vec3 resl1n = col * 0.4886025119029199 * dir.y;
	vec3 resl10 = col * 0.4886025119029199 * dir.z;
	vec3 resl1p = col * 0.4886025119029199 * dir.x;

	// -----------------------------------
	// Now perform parallel sum reduction.
	// -----------------------------------

	// The first iteration must be done by hand because otherwise, shared_lds[]
	// would need to be twice its size.

	if (thread_idx >= NUM_THREADS / 2u) {
		const uint dst_idx = thread_idx - NUM_THREADS / 2u;
		SAVE_TO_LDS(dst_idx);
	}

	// memoryBarrierShared ensures our write is visible to everyone else (must be done BEFORE the barrier).
	// barrier ensures every thread's execution reached here.
	memoryBarrierShared();
	barrier();

	if (thread_idx < NUM_THREADS / 2u) {
		const uint src_idx = thread_idx;
		LOAD_FROM_LDS(src_idx);

		// We multiply by 0.5 in every iteration instead of dividing by SAMPLE_COUNT
		// at the end because otherwise, we'd be out of snorm range. And dividing at the
		// beginning would cause serious quantization issues. This is the 'least bad' option.
		resl0 = (resl0 + other_resl0) * 0.5;
		resl1n = (resl1n + other_resl1n) * 0.5;
		resl10 = (resl10 + other_resl10) * 0.5;
		resl1p = (resl1p + other_resl1p) * 0.5;

		SAVE_TO_LDS(src_idx);
	}

	// Repeat the same, generically.
#pragma unroll
	for (uint s = NUM_THREADS / 4u; s > 1u; s >>= 1u) {
		const uint dst_idx = thread_idx;
		const uint src_idx = thread_idx + s;

		memoryBarrierShared();
		barrier();

		if (dst_idx < s) {
			LOAD_FROM_LDS(src_idx);

			resl0 = (resl0 + other_resl0) * 0.5;
			resl1n = (resl1n + other_resl1n) * 0.5;
			resl10 = (resl10 + other_resl10) * 0.5;
			resl1p = (resl1p + other_resl1p) * 0.5;

			SAVE_TO_LDS(dst_idx);
		}
	}

	// The last step is also done by hand to avoid a SAVE_TO_LDS() call.
	// TODO: We could also improve this by finishing a few steps earlier and use subgroups.
	//
	// Using subgroups is dangerous because the multiple if statements may not maximally reconverge
	// unless explicitly using that feature. We also need to account for the different subgroup sizes
	// and make sure all data ends up in a contiguous wavefront (which supposedly we already do).
	{
		const uint dst_idx = thread_idx;
		const uint src_idx = thread_idx + 1u;

		memoryBarrierShared();
		barrier();

		if (dst_idx == 0u) {
			LOAD_FROM_LDS(src_idx);

			resl0 = (resl0 + other_resl0) * 0.5;
			resl1n = (resl1n + other_resl1n) * 0.5;
			resl10 = (resl10 + other_resl10) * 0.5;
			resl1p = (resl1p + other_resl1p) * 0.5;

			store_data(resl0, resl1n, resl10, resl1p);
		}
	}
#endif
}
