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
#define NUM_THREADS 1024u
#define SAMPLES_PER_THREAD 1u

#define SAMPLE_COUNT (NUM_THREADS * SAMPLES_PER_THREAD)

#define M_PI 3.14159265359

#ifdef SH_DEBUG_MODE
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
#else
layout(local_size_x = NUM_THREADS, local_size_y = 1, local_size_z = 1) in;
#endif

layout(push_constant, std430) uniform Constants {
	uint compute_geomerics_l1;
}
constants;

layout(set = 0, binding = 0) uniform samplerCube source_cubemap;
layout(set = 0, binding = 1, std430) restrict writeonly buffer SHCoeffs {
	// Always big enough to cover L2 version, even if using L1.
	// Reference L1 version needs coeffs[3].
	// Optimized version, baking as much as possible for evaluate_sh_l1_geomerics() needs coeffs[5].
	vec4 coeffs[7];
}
sh;

// For NUM_THREADS = 1024u, 12KB of memory for L0 + L1 and 16kb for L1.
// Fits on every GPU model (smallest is 16KB). The data is encoded in 16-bit SNORM to save space.
shared uint shared_lds[NUM_THREADS / 2u][8u];

// Fibonacci Sphere (Sampling).
vec3 to_vector(uint sample_idx_uint) {
	const float sample_idx = float(sample_idx_uint);
	const float offset = 2.0 / SAMPLE_COUNT;
	const float increment = 2.39996322972865332223155550663361; // Golden Angle.
	vec3 dir;
	dir.y = (sample_idx * offset) - 1.0 + offset * 0.5;
	const float r = sqrt(1.0 - dir.y * dir.y);
	const float phi = sample_idx * increment;
	dir.x = cos(phi) * r;
	dir.z = sin(phi) * r;
	return dir;
}

#define SAVE_TO_LDS_L1(idx)                                       \
	shared_lds[idx][0] = packSnorm2x16(resl0.xy);                 \
	shared_lds[idx][1] = packSnorm2x16(vec2(resl0.z, resl1n.x));  \
	shared_lds[idx][2] = packSnorm2x16(resl1n.yz);                \
	shared_lds[idx][3] = packSnorm2x16(resl10.xy);                \
	shared_lds[idx][4] = packSnorm2x16(vec2(resl10.z, resl1p.x)); \
	shared_lds[idx][5] = packSnorm2x16(resl1p.yz)

#define LOAD_FROM_LDS_L1(idx)                        \
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

// We multiply by 0.5 in every iteration instead of dividing by SAMPLE_COUNT
// at the end because otherwise, we'd be out of snorm range. And dividing at the
// beginning would cause serious quantization issues. This is the 'least bad' option.
#define MERGE_LDS_L1()                      \
	resl0 = (resl0 + other_resl0) * 0.5;    \
	resl1n = (resl1n + other_resl1n) * 0.5; \
	resl10 = (resl10 + other_resl10) * 0.5; \
	resl1p = (resl1p + other_resl1p) * 0.5

#define SAVE_TO_LDS_L2(idx)                                         \
	shared_lds[idx][0] = packSnorm2x16(resl2n2.xy);                 \
	shared_lds[idx][1] = packSnorm2x16(vec2(resl2n2.z, resl2n1.x)); \
	shared_lds[idx][2] = packSnorm2x16(resl2n1.yz);                 \
	shared_lds[idx][3] = packSnorm2x16(resl200.xy);                 \
	shared_lds[idx][4] = packSnorm2x16(vec2(resl200.z, resl2p1.x)); \
	shared_lds[idx][5] = packSnorm2x16(resl2p1.yz);                 \
	shared_lds[idx][6] = packSnorm2x16(resl2p2.xy);                 \
	shared_lds[idx][7] = packSnorm2x16(vec2(resl2p2.z, 0.0));

#define LOAD_FROM_LDS_L2(idx)                         \
	vec2 tmp0, tmp1;                                  \
	tmp0 = unpackSnorm2x16(shared_lds[idx][0]);       \
	tmp1 = unpackSnorm2x16(shared_lds[idx][1]);       \
	const vec3 other_resl2n2 = vec3(tmp0.xy, tmp1.x); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][2]);       \
	const vec3 other_resl2n1 = vec3(tmp1.y, tmp0.xy); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][3]);       \
	tmp1 = unpackSnorm2x16(shared_lds[idx][4]);       \
	const vec3 other_resl200 = vec3(tmp0.xy, tmp1.x); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][5]);       \
	const vec3 other_resl2p1 = vec3(tmp1.y, tmp0.xy); \
	tmp0 = unpackSnorm2x16(shared_lds[idx][6]);       \
	tmp1 = unpackSnorm2x16(shared_lds[idx][7]);       \
	const vec3 other_resl2p2 = vec3(tmp0.xy, tmp1.x)

#define MERGE_LDS_L2()                         \
	resl2n2 = (resl2n2 + other_resl2n2) * 0.5; \
	resl2n1 = (resl2n1 + other_resl2n1) * 0.5; \
	resl200 = (resl200 + other_resl200) * 0.5; \
	resl2p1 = (resl2p1 + other_resl2p1) * 0.5; \
	resl2p2 = (resl2p2 + other_resl2p2) * 0.5

void store_data_l1(vec3 resl0, vec3 resl1n, vec3 resl10, vec3 resl1p) {
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
	// NOTE: R0 can't be negative (unless input colors are).
	// R1 can be negative. lenR1 cannot be negative (because it's a length).
	// We use 1e-6 to prevent divisions by zero. This is handled properly in Desktop GPUs, but mobile
	// is anyone's guess since full IEEE-754 compliance is not mandated.
	const vec3 R0 = resl0;
	const vec3 safe_R0 = max(resl0, vec3(1e-6));
	vec3 R1[3]; // R1[channel].
	R1[0] = 0.5 * vec3(resl1p[0], resl1n[0], resl10[0]);
	R1[1] = 0.5 * vec3(resl1p[1], resl1n[1], resl10[1]);
	R1[2] = 0.5 * vec3(resl1p[2], resl1n[2], resl10[2]);
	const vec3 lenR1 = max(vec3(length(R1[0]), length(R1[1]), length(R1[2])), vec3(1e-6));
	const vec3 p = 1.0 + 2.0 * lenR1 / safe_R0;
	const vec3 a = (1.0 - lenR1 / safe_R0) / (1.0 + lenR1 / safe_R0);

	R1[0] = R1[0] / lenR1[0];
	R1[1] = R1[1] / lenR1[1];
	R1[2] = R1[2] / lenR1[2];

	sh.coeffs[0].xyzw = vec4(R0.xyz, R1[0].x);
	sh.coeffs[1].xyzw = vec4(R1[0].yz, R1[1].xy);
	sh.coeffs[2].xyzw = vec4(R1[1].z, R1[2].xyz);
	sh.coeffs[3].xyzw = vec4(p.xyz, a.x);
	sh.coeffs[4].xyzw = vec4(a.yz, 0.0, 0.0);
#endif
}

void store_data_l2_l1(vec3 resl0, vec3 resl1n, vec3 resl10, vec3 resl1p) {
	sh.coeffs[0].xyzw = vec4(resl0.xyz, resl1p.x);
	sh.coeffs[1].xyzw = vec4(resl1n.xyz, resl1p.y);
	sh.coeffs[2].xyzw = vec4(resl10.xyz, resl1p.z);
}

void store_data_l2_l2(vec3 resl2n2, vec3 resl2n1, vec3 resl200, vec3 resl2p1, vec3 resl2p2) {
	sh.coeffs[3].xyzw = vec4(resl2n2.xyz, resl2p1.x);
	sh.coeffs[4].xyzw = vec4(resl2n1.xyz, resl2p1.y);
	sh.coeffs[5].xyzw = vec4(resl200.xyz, resl2p1.z);
	sh.coeffs[6].xyzw = vec4(resl2p2.xyz, 0.0);
}

void main() {
	const float MDISABLE = 1.0;

#ifdef SH_DEBUG_MODE
	vec3 resl0 = vec3(0.0);
	vec3 resl1n = vec3(0.0);
	vec3 resl10 = vec3(0.0);
	vec3 resl1p = vec3(0.0);

	vec3 resl2n2 = vec3(0.0);
	vec3 resl2n1 = vec3(0.0);
	vec3 resl200 = vec3(0.0);
	vec3 resl2p1 = vec3(0.0);
	vec3 resl2p2 = vec3(0.0);

	// Debug mode processes everything in one thread. It's slow, but easy to debug & understand.
	for (uint _thread_id = 0u; _thread_id < NUM_THREADS; ++_thread_id) {
		vec3 tmp_resl0 = vec3(0.0);
		vec3 tmp_resl1n = vec3(0.0);
		vec3 tmp_resl10 = vec3(0.0);
		vec3 tmp_resl1p = vec3(0.0);
		vec3 tmp_resl2n2 = vec3(0.0);
		vec3 tmp_resl2n1 = vec3(0.0);
		vec3 tmp_resl200 = vec3(0.0);
		vec3 tmp_resl2p1 = vec3(0.0);
		vec3 tmp_resl2p2 = vec3(0.0);

		for (uint s = 0u; s < SAMPLES_PER_THREAD; ++s) {
			const uint sample_idx = _thread_id * SAMPLES_PER_THREAD + s;
			const vec3 dir = to_vector(sample_idx);
			const vec3 col = textureLod(source_cubemap, dir * vec3(-1.0, -1.0, 1.0), 0.0).xyz;
			// L0 + L1.
			tmp_resl0 += col;
			tmp_resl1n += col * dir.y;
			tmp_resl10 += col * dir.z;
			tmp_resl1p += col * dir.x;

			// L2.
			tmp_resl2n2 += col * dir.x * dir.y * MDISABLE;
			tmp_resl2n1 += col * dir.y * dir.z * MDISABLE;
			tmp_resl200 += col * (3.0 * dir.z * dir.z - 1.0) * MDISABLE;
			tmp_resl2p1 += col * dir.x * dir.z * MDISABLE;
			tmp_resl2p2 += col * ((dir.x * dir.x) - (dir.y * dir.y)) * MDISABLE;
		}
		// L0 + L1.
		resl0 += tmp_resl0 * (1.0 / SAMPLES_PER_THREAD);
		resl1n += tmp_resl1n * (1.0 / SAMPLES_PER_THREAD);
		resl10 += tmp_resl10 * (1.0 / SAMPLES_PER_THREAD);
		resl1p += tmp_resl1p * (1.0 / SAMPLES_PER_THREAD);

		// L2.
		resl2n2 += tmp_resl2n2 * (1.0 / SAMPLES_PER_THREAD);
		resl2n1 += tmp_resl2n1 * (1.0 / SAMPLES_PER_THREAD);
		resl200 += tmp_resl200 * (1.0 / SAMPLES_PER_THREAD);
		resl2p1 += tmp_resl2p1 * (1.0 / SAMPLES_PER_THREAD);
		resl2p2 += tmp_resl2p2 * (1.0 / SAMPLES_PER_THREAD);
	}

	resl0 *= M_PI / NUM_THREADS * 0.28209479177387814;
	resl1n *= M_PI / NUM_THREADS * -0.4886025119029199;
	resl10 *= M_PI / NUM_THREADS * 0.4886025119029199;
	resl1p *= M_PI / NUM_THREADS * -0.4886025119029199;

	resl2n2 *= M_PI / NUM_THREADS * 1.09254843059208;
	resl2n1 *= M_PI / NUM_THREADS * -1.09254843059208;
	resl200 *= M_PI / NUM_THREADS * 0.31539156525252;
	resl2p1 *= M_PI / NUM_THREADS * -1.09254843059208;
	resl2p2 *= M_PI / NUM_THREADS * 0.54627421529604;

	if (constants.compute_geomerics_l1 != 0u) {
		store_data_l1(resl0, resl1n, resl10, resl1p);
	} else {
		store_data_l2_l1(resl0, resl1n, resl10, resl1p);
		store_data_l2_l2(resl2n2, resl2n1, resl200, resl2p1, resl2p2);
	}
#else // !SH_DEBUG_MODE
	// The real implementation works like this:
	//
	// GPUs have a limit of 1024 threads per threadgroup (also, we run into shared LDS limits too).
	// Having that in mind...
	//
	// For L0+L1 we launch a threadgroup w/ 1024 threads. Each one processes SAMPLES_PER_THREAD samples.
	// Then uses parallel-sum to combine all those 1024 results into just 1 value (9 to be exact).
	//
	// For L2 we also launch 1024 threads. And perform exactly the same as we did with L0+L1, except
	// that the results will be written to another region of memory that doesn't overlap (it might
	// cause false cache sharing if such thing can happen on GPUs, but it's only for the last 2 threads
	// at the end. It shouldn't be a problem).
	//
	// We launch one dispatch with 2 threadgroups. We identify the threadgroup that processes
	// L0+L1 when gl_WorkGroupID.x == 0 and the one that processes L2 when gl_WorkGroupID.x == 1.
	//
	// NOTE: Parallel reduction makes heavy use of barrier(). All threads in the same threadgroup
	// must hit the barrier or else it's undefined behavior. Be mindful of this when using loops
	// or branches, they can't diverge.  This is not a problem for L0+L1 vs L2 code paths because all
	// threadgroups take the same paths.

	const uint thread_idx = gl_LocalInvocationIndex;
	const bool l2_group = gl_WorkGroupID.x != 0u;

	if (!l2_group) {
		// L0 + L1.
		vec3 resl0 = vec3(0.0);
		vec3 resl1n = vec3(0.0);
		vec3 resl10 = vec3(0.0);
		vec3 resl1p = vec3(0.0);

		for (uint s = 0u; s < SAMPLES_PER_THREAD; ++s) {
			const uint sample_idx = thread_idx * SAMPLES_PER_THREAD + s;
			const vec3 dir = to_vector(sample_idx);
			const vec3 col = textureLod(source_cubemap, dir * vec3(1.0, 1.0, 1.0), 0.0).xyz;
			resl0 += col;
			resl1n += col * dir.y;
			resl10 += col * dir.z;
			resl1p += col * dir.x;
		}

		// We must keep these values within the SNORM range [-1.0; 1.0] because of our LDS.
		resl0 *= (1.0 / SAMPLES_PER_THREAD);
		resl1n *= (1.0 / SAMPLES_PER_THREAD);
		resl10 *= (1.0 / SAMPLES_PER_THREAD);
		resl1p *= (1.0 / SAMPLES_PER_THREAD);

		// -----------------------------------
		// Now perform parallel sum reduction.
		// -----------------------------------

		// The first iteration must be done by hand because otherwise, shared_lds[]
		// would need to be twice its size.

		if (thread_idx >= NUM_THREADS / 2u) {
			const uint dst_idx = thread_idx - NUM_THREADS / 2u;
			SAVE_TO_LDS_L1(dst_idx);
		}

		// memoryBarrierShared ensures our write is visible to everyone else (must be done BEFORE the barrier).
		// barrier ensures every thread's execution reached here.
		memoryBarrierShared();
		barrier();

		if (thread_idx < NUM_THREADS / 2u) {
			const uint src_idx = thread_idx;
			LOAD_FROM_LDS_L1(src_idx);
			MERGE_LDS_L1();
			SAVE_TO_LDS_L1(src_idx);
		}

		// Repeat the same, generically.
#pragma unroll
		for (uint s = NUM_THREADS / 4u; s > 1u; s >>= 1u) {
			const uint dst_idx = thread_idx;
			const uint src_idx = thread_idx + s;

			memoryBarrierShared();
			barrier();

			if (dst_idx < s) {
				LOAD_FROM_LDS_L1(src_idx);
				MERGE_LDS_L1();
				SAVE_TO_LDS_L1(dst_idx);
			}
		}

		// The last step is also done by hand to avoid a SAVE_TO_LDS_L1() call.
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
				LOAD_FROM_LDS_L1(src_idx);
				MERGE_LDS_L1();

				resl0 *= M_PI * 0.28209479177387814;
				resl1n *= M_PI * 0.4886025119029199;
				resl10 *= M_PI * 0.4886025119029199;
				resl1p *= M_PI * 0.4886025119029199;

				if (constants.compute_geomerics_l1 != 0u) {
					store_data_l1(resl0, resl1n, resl10, resl1p);
				} else {
					store_data_l2_l1(resl0, resl1n, resl10, resl1p);
				}
			}
		}
	} else {
		// L2.
		vec3 resl2n2 = vec3(0.0);
		vec3 resl2n1 = vec3(0.0);
		vec3 resl200 = vec3(0.0);
		vec3 resl2p1 = vec3(0.0);
		vec3 resl2p2 = vec3(0.0);

		for (uint s = 0u; s < SAMPLES_PER_THREAD; ++s) {
			const uint sample_idx = thread_idx * SAMPLES_PER_THREAD + s;
			const vec3 dir = to_vector(sample_idx);
			const vec3 col = textureLod(source_cubemap, dir * vec3(1.0, 1.0, 1.0), 0.0).xyz;

			resl2n2 += col * dir.x * dir.y * MDISABLE;
			resl2n1 += col * dir.y * dir.z * MDISABLE;
			resl200 += col * (3.0 * dir.z * dir.z - 1.0) * MDISABLE;
			resl2p1 += col * dir.x * dir.z * MDISABLE;
			resl2p2 += col * ((dir.x * dir.x) - (dir.y * dir.y)) * MDISABLE;
		}

		// We must keep these values within the SNORM range [-1.0; 1.0] because of our LDS.
		resl2n2 *= (1.0 / SAMPLES_PER_THREAD);
		resl2n1 *= (1.0 / SAMPLES_PER_THREAD);
		resl200 *= (1.0 / SAMPLES_PER_THREAD);
		resl2p1 *= (1.0 / SAMPLES_PER_THREAD);
		resl2p2 *= (1.0 / SAMPLES_PER_THREAD);

		// resl200 is in range [2; -1]. We need it in SNORM.
		resl200 = clamp((resl200 - 0.5) * (1.0 / 1.5), -1.0, 1.0);

		// -----------------------------------
		// Now perform parallel sum reduction.
		// -----------------------------------

		if (thread_idx >= NUM_THREADS / 2u) {
			const uint dst_idx = thread_idx - NUM_THREADS / 2u;
			SAVE_TO_LDS_L2(dst_idx);
		}

		// memoryBarrierShared ensures our write is visible to everyone else (must be done BEFORE the barrier).
		// barrier ensures every thread's execution reached here.
		memoryBarrierShared();
		barrier();

		if (thread_idx < NUM_THREADS / 2u) {
			const uint src_idx = thread_idx;
			LOAD_FROM_LDS_L2(src_idx);
			MERGE_LDS_L2();
			SAVE_TO_LDS_L2(src_idx);
		}

		// Repeat the same, generically.
#pragma unroll
		for (uint s = NUM_THREADS / 4u; s > 1u; s >>= 1u) {
			const uint dst_idx = thread_idx;
			const uint src_idx = thread_idx + s;

			memoryBarrierShared();
			barrier();

			if (dst_idx < s) {
				LOAD_FROM_LDS_L2(src_idx);
				MERGE_LDS_L2();
				SAVE_TO_LDS_L2(dst_idx);
			}
		}

		// The last step is also done by hand to avoid a SAVE_TO_LDS_L2() call.
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
				LOAD_FROM_LDS_L2(src_idx);
				MERGE_LDS_L2();

				// Restore resl200 to its range [2; -1].
				resl200 = resl200 * 1.5 + 0.5;

				resl2n2 *= M_PI * 1.09254843059208;
				resl2n1 *= M_PI * 1.09254843059208;
				resl200 *= M_PI * 0.31539156525252;
				resl2p1 *= M_PI * 1.09254843059208;
				resl2p2 *= M_PI * 0.54627421529604;

				store_data_l2_l2(resl2n2, resl2n1, resl200, resl2p1, resl2p2);
			}
		}
	}
#endif
}
