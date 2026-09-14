// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2022 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#ifdef JPH_USE_NEON

// Constructing NEON values
#ifdef JPH_COMPILER_MSVC
	#define JPH_NEON_INT32x4(v1, v2, v3, v4) { int64_t(v1) + (int64_t(v2) << 32), int64_t(v3) + (int64_t(v4) << 32) }
	#define JPH_NEON_UINT32x4(v1, v2, v3, v4) { uint64_t(v1) + (uint64_t(v2) << 32), uint64_t(v3) + (uint64_t(v4) << 32) }
	#define JPH_NEON_INT8x16(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) { int64_t(v1) + (int64_t(v2) << 8) + (int64_t(v3) << 16) + (int64_t(v4) << 24) + (int64_t(v5) << 32) + (int64_t(v6) << 40) + (int64_t(v7) << 48) + (int64_t(v8) << 56), int64_t(v9) + (int64_t(v10) << 8) + (int64_t(v11) << 16) + (int64_t(v12) << 24) + (int64_t(v13) << 32) + (int64_t(v14) << 40) + (int64_t(v15) << 48) + (int64_t(v16) << 56) }
	#define JPH_NEON_UINT8x16(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) { uint64_t(v1) + (uint64_t(v2) << 8) + (uint64_t(v3) << 16) + (uint64_t(v4) << 24) + (uint64_t(v5) << 32) + (uint64_t(v6) << 40) + (uint64_t(v7) << 48) + (uint64_t(v8) << 56), uint64_t(v9) + (uint64_t(v10) << 8) + (uint64_t(v11) << 16) + (uint64_t(v12) << 24) + (uint64_t(v13) << 32) + (uint64_t(v14) << 40) + (uint64_t(v15) << 48) + (uint64_t(v16) << 56) }
#else
	#define JPH_NEON_INT32x4(v1, v2, v3, v4) { v1, v2, v3, v4 }
	#define JPH_NEON_UINT32x4(v1, v2, v3, v4) { v1, v2, v3, v4 }
	#define JPH_NEON_INT8x16(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) { v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16 }
	#define JPH_NEON_UINT8x16(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) { v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16 }
#endif

// MSVC and GCC prior to version 12 don't define __builtin_shufflevector
#if defined(JPH_COMPILER_MSVC) || (defined(JPH_COMPILER_GCC) && __GNUC__ < 12)
	JPH_NAMESPACE_BEGIN

	// Generic shuffle vector template
	template <unsigned I1, unsigned I2, unsigned I3, unsigned I4>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4(float32x4_t inV1, float32x4_t inV2)
	{
		float32x2_t lo = vcopy_laneq_f32(vdup_n_f32(0), 0, I1 >= 4? inV2 : inV1, I1 & 0b11);
		lo = vcopy_laneq_f32(lo, 1, I2 >= 4? inV2 : inV1, I2 & 0b11);

		float32x2_t hi = vcopy_laneq_f32(vdup_n_f32(0), 0, I3 >= 4? inV2 : inV1, I3 & 0b11);
		hi = vcopy_laneq_f32(hi, 1, I4 >= 4? inV2 : inV1, I4 & 0b11);

		return vcombine_f32(lo, hi);
	}

	// Specializations
	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 0, 0, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		return vdupq_laneq_f32(inV1, 0);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 0, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vget_low_f32(inV1), vdup_lane_f32(vget_low_f32(inV1), 0));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(inV1, 3, inV1, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 2, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return inV1;
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vget_low_f32(inV1), vrev64_f32(vget_high_f32(inV1)));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 3, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(inV1, 2, inV1, 3);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 1, 4, 5>(float32x4_t inV1, float32x4_t inV2)
	{
		return vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(inV1), vreinterpretq_f64_f32(inV2)));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 1, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp1q_f32(inV1, vdupq_laneq_f32(inV1, 1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 1, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp1q_f32(inV1, vrev64q_f32(inV1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp1q_f32(inV1, vdupq_laneq_f32(inV1, 2));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 2, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(inV1, 1, inV1, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(vuzp1q_f32(inV1, inV1), 2, inV1, 3);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 3, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp1q_f32(inV1, vdupq_laneq_f32(inV1, 3));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 2, 4, 6>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp1q_f32(inV1, inV2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<0, 3, 1, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vzip1q_f32(inV1, vextq_f32(inV1, vdupq_laneq_f32(inV1, 2), 3));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 0, 0, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vrev64_f32(vget_low_f32(inV1)), vdup_lane_f32(vget_low_f32(inV1), 0));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 0, 0, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vrev64_f32(vget_low_f32(inV1)), vzip1_f32(vget_low_f32(inV1), vget_high_f32(inV1)));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 0, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vrev64q_f32(inV1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 1, 1, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		return vdupq_laneq_f32(inV1, 1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 1, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x4_t t = vextq_f32(inV1, inV1, 1);
		return vzip1q_f32(t, t);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 1, 3, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vtrn2q_f32(inV1, inV1);
	}

	// Used extensively by cross product
	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 0, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(vextq_f32(inV1, inV1, 1), 2, inV1, 0);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 0, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(vextq_f32(inV1, inV1, 3), inV1, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 0, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(vuzp1q_f32(inV1, inV1), 0, inV1, 1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vext_f32(vget_low_f32(inV1), vget_high_f32(inV1), 1), vdup_lane_f32(vget_high_f32(inV1), 0));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(inV1, vdupq_laneq_f32(inV1, 2), 1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 2, 3, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(inV1, vdupq_laneq_f32(inV1, 3), 1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 3, 0, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp2q_f32(inV1, vrev64q_f32(inV1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<1, 3, 5, 7>(float32x4_t inV1, float32x4_t inV2)
	{
		return vuzp2q_f32(inV1, inV2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 0, 1, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(vzip1q_f32(inV1, inV1), 0, inV1, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 0, 1, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(vuzp1q_f32(inV1, inV1), inV1, 3);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 1, 0, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x4_t t = vextq_f32(vuzp1q_f32(inV1, inV1), inV1, 3);
		return vuzp1q_f32(t, vuzp1q_f32(inV1, inV1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 1, 0, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x4_t t = vrev64q_f32(inV1);
		return vextq_f32(t, t, 3);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 2, 1, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(vtrn1q_f32(inV1, inV1), vrev64q_f32(inV1), 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 2, 1, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x4_t t = vcopyq_laneq_f32(inV1, 3, inV1, 1);
		return vzip2q_f32(t, t);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 2, 1, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcopyq_laneq_f32(vdupq_laneq_f32(inV1, 2), 2, inV1, 1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 2, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vdupq_laneq_f32(inV1, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 3, 0, 1>(float32x4_t inV1, float32x4_t inV2)
	{
		return vcombine_f32(vget_high_f32(inV1), vget_low_f32(inV1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 3, 1, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(inV1, vextq_f32(inV1, inV1, 1), 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 3, 2, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(inV1, vdupq_laneq_f32(inV1, 2), 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 3, 2, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vreinterpretq_f32_f64(vdupq_laneq_f64(vreinterpretq_f64_f32(inV1), 1));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<2, 3, 6, 7>(float32x4_t inV1, float32x4_t inV2)
	{
		return vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(inV1), vreinterpretq_f64_f32(inV2)));
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<3, 0, 1, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vextq_f32(inV1, inV1, 3);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<3, 0, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		return vtrn1q_f32(vdupq_laneq_f32(inV1, 3), inV1);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<3, 2, 1, 0>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x4_t t = vrev64q_f32(inV1);
		return vextq_f32(t, t, 2);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<3, 2, 3, 2>(float32x4_t inV1, float32x4_t inV2)
	{
		float32x2_t zy = vrev64_f32(vget_high_f32(inV1));
		return vcombine_f32(zy, zy);
	}

	template <>
	JPH_INLINE float32x4_t NeonShuffleFloat32x4<3, 3, 3, 3>(float32x4_t inV1, float32x4_t inV2)
	{
		return vdupq_laneq_f32(inV1, 3);
	}

	// Shuffle a vector
	#define JPH_NEON_SHUFFLE_F32x4(vec1, vec2, index1, index2, index3, index4) NeonShuffleFloat32x4<index1, index2, index3, index4>(vec1, vec2)
	#define JPH_NEON_SHUFFLE_U32x4(vec1, vec2, index1, index2, index3, index4) vreinterpretq_u32_f32((NeonShuffleFloat32x4<index1, index2, index3, index4>(vreinterpretq_f32_u32(vec1), vreinterpretq_f32_u32(vec2))))

	JPH_NAMESPACE_END
#else
	// Shuffle a vector
	#define JPH_NEON_SHUFFLE_F32x4(vec1, vec2, index1, index2, index3, index4) __builtin_shufflevector(vec1, vec2, index1, index2, index3, index4)
	#define JPH_NEON_SHUFFLE_U32x4(vec1, vec2, index1, index2, index3, index4) __builtin_shufflevector(vec1, vec2, index1, index2, index3, index4)
#endif

#endif // JPH_USE_NEON
