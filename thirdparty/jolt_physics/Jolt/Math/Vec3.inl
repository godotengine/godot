// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Math/Vec4.h>
#include <Jolt/Math/UVec4.h>
#include <Jolt/Core/HashCombine.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <random>
JPH_SUPPRESS_WARNINGS_STD_END

// Create a std::hash/JPH::Hash for Vec3
JPH_MAKE_HASHABLE(JPH::Vec3, t.GetX(), t.GetY(), t.GetZ())

JPH_NAMESPACE_BEGIN

void Vec3::CheckW() const
{
#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
	// Avoid asserts when both components are NaN
	JPH_ASSERT(reinterpret_cast<const uint32 *>(mF32)[2] == reinterpret_cast<const uint32 *>(mF32)[3]);
#endif // JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
}

JPH_INLINE Vec3::Type Vec3::sFixW(Type inValue)
{
#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
	#if defined(JPH_USE_SSE)
		return _mm_shuffle_ps(inValue, inValue, _MM_SHUFFLE(2, 2, 1, 0));
	#elif defined(JPH_USE_NEON)
		return JPH_NEON_SHUFFLE_F32x4(inValue, inValue, 0, 1, 2, 2);
	#elif defined(JPH_USE_RVV)
		Type value;
		const vfloat32m1_t v = __riscv_vle32_v_f32m1(inValue.mData, 3);
		__riscv_vse32_v_f32m1(value.mData, v, 3);
		value.mData[3] = value.mData[2];
		return value;
	#else
		Type value;
		value.mData[0] = inValue.mData[0];
		value.mData[1] = inValue.mData[1];
		value.mData[2] = inValue.mData[2];
		value.mData[3] = inValue.mData[2];
		return value;
	#endif
#else
	return inValue;
#endif // JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
}

Vec3::Vec3(Vec4Arg inRHS) :
	mValue(sFixW(inRHS.mValue))
{
}

Vec3::Vec3(const Float3 &inV)
{
#if defined(JPH_USE_SSE)
	Type x = _mm_load_ss(&inV.x);
	Type y = _mm_load_ss(&inV.y);
	Type z = _mm_load_ss(&inV.z);
	Type xy = _mm_unpacklo_ps(x, y);
	mValue = _mm_shuffle_ps(xy, z, _MM_SHUFFLE(0, 0, 1, 0)); // Assure Z and W are the same
#elif defined(JPH_USE_NEON)
	float32x2_t xy = vld1_f32(&inV.x);
	float32x2_t zz = vdup_n_f32(inV.z); // Assure Z and W are the same
	mValue = vcombine_f32(xy, zz);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(&inV.x, 3);
	__riscv_vse32_v_f32m1(mF32, v, 3);
	mF32[3] = inV.z;
#else
	mF32[0] = inV.x;
	mF32[1] = inV.y;
	mF32[2] = inV.z;
	mF32[3] = inV.z; // Not strictly needed when JPH_FLOATING_POINT_EXCEPTIONS_ENABLED is off but prevents warnings about uninitialized variables
#endif
}

Vec3::Vec3(float inX, float inY, float inZ)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_set_ps(inZ, inZ, inY, inX);
#elif defined(JPH_USE_NEON)
	uint32x2_t xy = vcreate_u32(static_cast<uint64>(BitCast<uint32>(inX)) | (static_cast<uint64>(BitCast<uint32>(inY)) << 32));
	uint32x2_t zz = vreinterpret_u32_f32(vdup_n_f32(inZ));
	mValue = vreinterpretq_f32_u32(vcombine_u32(xy, zz));
#elif defined(JPH_USE_RVV)
	const float aggregated[4] = { inX, inY, inZ, inZ };
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(aggregated, 4);
	__riscv_vse32_v_f32m1(mF32, v, 4);
#else
	mF32[0] = inX;
	mF32[1] = inY;
	mF32[2] = inZ;
	mF32[3] = inZ; // Not strictly needed when JPH_FLOATING_POINT_EXCEPTIONS_ENABLED is off but prevents warnings about uninitialized variables
#endif
}

template<uint32 SwizzleX, uint32 SwizzleY, uint32 SwizzleZ>
Vec3 Vec3::Swizzle() const
{
	static_assert(SwizzleX <= 3, "SwizzleX template parameter out of range");
	static_assert(SwizzleY <= 3, "SwizzleY template parameter out of range");
	static_assert(SwizzleZ <= 3, "SwizzleZ template parameter out of range");

#if defined(JPH_USE_SSE)
	return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(SwizzleZ, SwizzleZ, SwizzleY, SwizzleX)); // Assure Z and W are the same
#elif defined(JPH_USE_NEON)
	return JPH_NEON_SHUFFLE_F32x4(mValue, mValue, SwizzleX, SwizzleY, SwizzleZ, SwizzleZ);
#elif defined(JPH_USE_RVV)
	Vec3 v;
	const vfloat32m1_t data = __riscv_vle32_v_f32m1(mF32, 4);
	const uint32 stored_indices[4] = { SwizzleX, SwizzleY, SwizzleZ, SwizzleZ };
	const vuint32m1_t index = __riscv_vle32_v_u32m1(stored_indices, 4);
	const vfloat32m1_t swizzled = __riscv_vrgather_vv_f32m1(data, index, 4);
	__riscv_vse32_v_f32m1(v.mF32, swizzled, 4);
	return v;
#else
	return Vec3(mF32[SwizzleX], mF32[SwizzleY], mF32[SwizzleZ]);
#endif
}

Vec3 Vec3::sZero()
{
#if defined(JPH_USE_SSE)
	return _mm_setzero_ps();
#elif defined(JPH_USE_NEON)
	return vdupq_n_f32(0);
#elif defined(JPH_USE_RVV)
	Vec3 v;
	const vfloat32m1_t zero_vec = __riscv_vfmv_v_f_f32m1(0.0f, 3);
	__riscv_vse32_v_f32m1(v.mF32, zero_vec, 3);
	return v;
#else
	return Vec3(0, 0, 0);
#endif
}

Vec3 Vec3::sReplicate(float inV)
{
#if defined(JPH_USE_SSE)
	return _mm_set1_ps(inV);
#elif defined(JPH_USE_NEON)
	return vdupq_n_f32(inV);
#elif defined(JPH_USE_RVV)
	Vec3 vec;
	const vfloat32m1_t v = __riscv_vfmv_v_f_f32m1(inV, 3);
	__riscv_vse32_v_f32m1(vec.mF32, v, 3);
	return vec;
#else
	return Vec3(inV, inV, inV);
#endif
}

Vec3 Vec3::sOne()
{
	return sReplicate(1.0f);
}

Vec3 Vec3::sNaN()
{
	return sReplicate(numeric_limits<float>::quiet_NaN());
}

Vec3 Vec3::sLoadFloat3Unsafe(const Float3 &inV)
{
#if defined(JPH_USE_SSE)
	Type v = _mm_loadu_ps(&inV.x);
#elif defined(JPH_USE_NEON)
	Type v = vld1q_f32(&inV.x);
#elif defined(JPH_USE_RVV)
	Type v;
	const vfloat32m1_t rvv = __riscv_vle32_v_f32m1(&inV.x, 3);
	__riscv_vse32_v_f32m1(v.mData, rvv, 3);
#else
	Type v = { inV.x, inV.y, inV.z };
#endif
	return sFixW(v);
}

Vec3 Vec3::sMin(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_min_ps(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vminq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t min = __riscv_vfmin_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, min, 3);
	return res;
#else
	return Vec3(min(inV1.mF32[0], inV2.mF32[0]),
				min(inV1.mF32[1], inV2.mF32[1]),
				min(inV1.mF32[2], inV2.mF32[2]));
#endif
}

Vec3 Vec3::sMax(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_max_ps(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vmaxq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t max = __riscv_vfmax_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, max, 3);
	return res;
#else
	return Vec3(max(inV1.mF32[0], inV2.mF32[0]),
				max(inV1.mF32[1], inV2.mF32[1]),
				max(inV1.mF32[2], inV2.mF32[2]));
#endif
}

Vec3 Vec3::sClamp(Vec3Arg inV, Vec3Arg inMin, Vec3Arg inMax)
{
	return sMax(sMin(inV, inMax), inMin);
}

UVec4 Vec3::sEquals(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_castps_si128(_mm_cmpeq_ps(inV1.mValue, inV2.mValue));
#elif defined(JPH_USE_NEON)
	return vceqq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vbool32_t mask = __riscv_vmfeq_vv_f32m1_b32(v1, v2, 3);
	const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0x0, 3);
	const vuint32m1_t merged = __riscv_vmerge_vxm_u32m1(zeros, 0xFFFFFFFF, mask, 3);
	__riscv_vse32_v_u32m1(res.mU32, merged, 3);
	res.mU32[3] = res.mU32[2];
	return res;
#else
	uint32 z = inV1.mF32[2] == inV2.mF32[2]? 0xffffffffu : 0;
	return UVec4(inV1.mF32[0] == inV2.mF32[0]? 0xffffffffu : 0,
				 inV1.mF32[1] == inV2.mF32[1]? 0xffffffffu : 0,
				 z,
				 z);
#endif
}

UVec4 Vec3::sLess(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_castps_si128(_mm_cmplt_ps(inV1.mValue, inV2.mValue));
#elif defined(JPH_USE_NEON)
	return vcltq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(v1, v2, 3);
	const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0x0, 3);
	const vuint32m1_t merged = __riscv_vmerge_vxm_u32m1(zeros, 0xFFFFFFFF, mask, 3);
	__riscv_vse32_v_u32m1(res.mU32, merged, 3);
	res.mU32[3] = res.mU32[2];
	return res;
#else
	uint32 z = inV1.mF32[2] < inV2.mF32[2]? 0xffffffffu : 0;
	return UVec4(inV1.mF32[0] < inV2.mF32[0]? 0xffffffffu : 0,
				 inV1.mF32[1] < inV2.mF32[1]? 0xffffffffu : 0,
				 z,
				 z);
#endif
}

UVec4 Vec3::sLessOrEqual(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_castps_si128(_mm_cmple_ps(inV1.mValue, inV2.mValue));
#elif defined(JPH_USE_NEON)
	return vcleq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vbool32_t mask = __riscv_vmfle_vv_f32m1_b32(v1, v2, 3);
	const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0x0, 3);
	const vuint32m1_t merged = __riscv_vmerge_vxm_u32m1(zeros, 0xFFFFFFFF, mask, 3);
	__riscv_vse32_v_u32m1(res.mU32, merged, 3);
	res.mU32[3] = res.mU32[2];
	return res;
#else
	uint32 z = inV1.mF32[2] <= inV2.mF32[2]? 0xffffffffu : 0;
	return UVec4(inV1.mF32[0] <= inV2.mF32[0]? 0xffffffffu : 0,
				 inV1.mF32[1] <= inV2.mF32[1]? 0xffffffffu : 0,
				 z,
				 z);
#endif
}

UVec4 Vec3::sGreater(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_castps_si128(_mm_cmpgt_ps(inV1.mValue, inV2.mValue));
#elif defined(JPH_USE_NEON)
	return vcgtq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vbool32_t mask = __riscv_vmfgt_vv_f32m1_b32(v1, v2, 3);
	const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0x0, 3);
	const vuint32m1_t merged = __riscv_vmerge_vxm_u32m1(zeros, 0xFFFFFFFF, mask, 3);
	__riscv_vse32_v_u32m1(res.mU32, merged, 3);
	res.mU32[3] = res.mU32[2];
	return res;
#else
	uint32 z = inV1.mF32[2] > inV2.mF32[2]? 0xffffffffu : 0;
	return UVec4(inV1.mF32[0] > inV2.mF32[0]? 0xffffffffu : 0,
				 inV1.mF32[1] > inV2.mF32[1]? 0xffffffffu : 0,
				 z,
				 z);
#endif
}

UVec4 Vec3::sGreaterOrEqual(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_castps_si128(_mm_cmpge_ps(inV1.mValue, inV2.mValue));
#elif defined(JPH_USE_NEON)
	return vcgeq_f32(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV1.mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vbool32_t mask = __riscv_vmfge_vv_f32m1_b32(v1, v2, 3);
	const vuint32m1_t zeros = __riscv_vmv_v_x_u32m1(0x0, 3);
	const vuint32m1_t merged = __riscv_vmerge_vxm_u32m1(zeros, 0xFFFFFFFF, mask, 3);
	__riscv_vse32_v_u32m1(res.mU32, merged, 3);
	res.mU32[3] = res.mU32[2];
	return res;
#else
	uint32 z = inV1.mF32[2] >= inV2.mF32[2]? 0xffffffffu : 0;
	return UVec4(inV1.mF32[0] >= inV2.mF32[0]? 0xffffffffu : 0,
				 inV1.mF32[1] >= inV2.mF32[1]? 0xffffffffu : 0,
				 z,
				 z);
#endif
}

Vec3 Vec3::sFusedMultiplyAdd(Vec3Arg inMul1, Vec3Arg inMul2, Vec3Arg inAdd)
{
#ifdef JPH_USE_FMADD
	#ifdef JPH_USE_SSE
		return _mm_fmadd_ps(inMul1.mValue, inMul2.mValue, inAdd.mValue);
	#elif defined(JPH_USE_NEON)
		return vmlaq_f32(inAdd.mValue, inMul1.mValue, inMul2.mValue);
	#elif defined(JPH_USE_RVV)
		Vec3 res;
		const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inMul1.mF32, 3);
		const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inMul2.mF32, 3);
		const vfloat32m1_t rvv_add = __riscv_vle32_v_f32m1(inAdd.mF32, 3);
		const vfloat32m1_t fmadd = __riscv_vfmacc_vv_f32m1(rvv_add, v1, v2, 3);
		__riscv_vse32_v_f32m1(res.mF32, fmadd, 3);
		return res;
	#else
		return inMul1 * inMul2 + inAdd;
	#endif
#else
	return inMul1 * inMul2 + inAdd;
#endif
}

Vec3 Vec3::sSelect(Vec3Arg inNotSet, Vec3Arg inSet, UVec4Arg inControl)
{
#if defined(JPH_USE_SSE4_1) && !defined(JPH_PLATFORM_WASM) // _mm_blendv_ps has problems on FireFox
	Type v = _mm_blendv_ps(inNotSet.mValue, inSet.mValue, _mm_castsi128_ps(inControl.mValue));
	return sFixW(v);
#elif defined(JPH_USE_SSE)
	__m128 is_set = _mm_castsi128_ps(_mm_srai_epi32(inControl.mValue, 31));
	Type v = _mm_or_ps(_mm_and_ps(is_set, inSet.mValue), _mm_andnot_ps(is_set, inNotSet.mValue));
	return sFixW(v);
#elif defined(JPH_USE_NEON)
	Type v = vbslq_f32(vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(inControl.mValue), 31)), inSet.mValue, inNotSet.mValue);
	return sFixW(v);
#elif defined(JPH_USE_RVV)
	Vec3 masked;
	const vuint32m1_t control = __riscv_vle32_v_u32m1(inControl.mU32, 3);
	const vfloat32m1_t not_set = __riscv_vle32_v_f32m1(inNotSet.mF32, 3);
	const vfloat32m1_t set = __riscv_vle32_v_f32m1(inSet.mF32, 3);

	// Generate RVV bool mask from UVec4
	const vuint32m1_t r = __riscv_vand_vx_u32m1(control, 0x80000000u, 3);
	const vbool32_t rvv_mask = __riscv_vmsne_vx_u32m1_b32(r, 0x0, 3);
	const vfloat32m1_t merged = __riscv_vmerge_vvm_f32m1(not_set, set, rvv_mask, 3);
	__riscv_vse32_v_f32m1(masked.mF32, merged, 3);
	return masked;
#else
	Vec3 result;
	for (int i = 0; i < 3; i++)
		result.mF32[i] = (inControl.mU32[i] & 0x80000000u) ? inSet.mF32[i] : inNotSet.mF32[i];
#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
	result.mF32[3] = result.mF32[2];
#endif // JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
	return result;
#endif
}

Vec3 Vec3::sOr(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_or_ps(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(inV1.mValue), vreinterpretq_u32_f32(inV2.mValue)));
#elif defined(JPH_USE_RVV)
	Vec3 or_result;
	const vuint32m1_t v1 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV1.mF32), 3);
	const vuint32m1_t v2 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV2.mF32), 3);
	const vuint32m1_t res = __riscv_vor_vv_u32m1(v1, v2, 3);
	__riscv_vse32_v_u32m1(reinterpret_cast<uint32 *>(or_result.mF32), res, 3);
	return or_result;
#else
	return Vec3(UVec4::sOr(inV1.ReinterpretAsInt(), inV2.ReinterpretAsInt()).ReinterpretAsFloat());
#endif
}

Vec3 Vec3::sXor(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_xor_ps(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(inV1.mValue), vreinterpretq_u32_f32(inV2.mValue)));
#elif defined(JPH_USE_RVV)
	Vec3 xor_result;
	const vuint32m1_t v1 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV1.mF32), 3);
	const vuint32m1_t v2 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV2.mF32), 3);
	const vuint32m1_t res = __riscv_vxor_vv_u32m1(v1, v2, 3);
	__riscv_vse32_v_u32m1(reinterpret_cast<uint32 *>(xor_result.mF32), res, 3);
	return xor_result;
#else
	return Vec3(UVec4::sXor(inV1.ReinterpretAsInt(), inV2.ReinterpretAsInt()).ReinterpretAsFloat());
#endif
}

Vec3 Vec3::sAnd(Vec3Arg inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_and_ps(inV1.mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(inV1.mValue), vreinterpretq_u32_f32(inV2.mValue)));
#elif defined(JPH_USE_RVV)
	Vec3 and_result;
	const vuint32m1_t v1 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV1.mF32), 3);
	const vuint32m1_t v2 = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32 *>(inV2.mF32), 3);
	const vuint32m1_t res = __riscv_vand_vv_u32m1(v1, v2, 3);
	__riscv_vse32_v_u32m1(reinterpret_cast<uint32 *>(and_result.mF32), res, 3);
	return and_result;
#else
	return Vec3(UVec4::sAnd(inV1.ReinterpretAsInt(), inV2.ReinterpretAsInt()).ReinterpretAsFloat());
#endif
}

Vec3 Vec3::sUnitSpherical(float inTheta, float inPhi)
{
	Vec4 s, c;
	Vec4(inTheta, inPhi, 0, 0).SinCos(s, c);
	return Vec3(s.GetX() * c.GetY(), s.GetX() * s.GetY(), c.GetX());
}

template <class Random>
Vec3 Vec3::sRandom(Random &inRandom)
{
	// Generating uniform unit random vectors in Rn - Andersen Ang
	// See: https://angms.science/doc/RM/randUnitVec.pdf
	float z = -1.0f + 2.0f * float(inRandom() - inRandom.min()) / float(inRandom.max() - inRandom.min());
	float r = JPH::Sqrt(1.0f - Square(z));
	float theta = 2.0f * JPH_PI * float(inRandom() - inRandom.min()) / float(inRandom.max() - inRandom.min());
	Vec4 s, c;
	Vec4::sReplicate(theta).SinCos(s, c);
	return Vec3(r * s.GetX(), r * c.GetX(), z);
}

bool Vec3::operator == (Vec3Arg inV2) const
{
	return sEquals(*this, inV2).TestAllXYZTrue();
}

bool Vec3::IsClose(Vec3Arg inV2, float inMaxDistSq) const
{
	return (inV2 - *this).LengthSq() <= inMaxDistSq;
}

bool Vec3::IsNearZero(float inMaxDistSq) const
{
	return LengthSq() <= inMaxDistSq;
}

Vec3 Vec3::operator * (Vec3Arg inV2) const
{
#if defined(JPH_USE_SSE)
	return _mm_mul_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vmulq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, mul, 3);
	return res;
#else
	return Vec3(mF32[0] * inV2.mF32[0], mF32[1] * inV2.mF32[1], mF32[2] * inV2.mF32[2]);
#endif
}

Vec3 Vec3::operator * (float inV2) const
{
#if defined(JPH_USE_SSE)
	return _mm_mul_ps(mValue, _mm_set1_ps(inV2));
#elif defined(JPH_USE_NEON)
	return vmulq_n_f32(mValue, inV2);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t src = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t mul = __riscv_vfmul_vf_f32m1(src, inV2, 3);
	__riscv_vse32_v_f32m1(res.mF32, mul, 3);
	return res;
#else
	return Vec3(mF32[0] * inV2, mF32[1] * inV2, mF32[2] * inV2);
#endif
}

Vec3 operator * (float inV1, Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	return _mm_mul_ps(_mm_set1_ps(inV1), inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vmulq_n_f32(inV2.mValue, inV1);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t mul = __riscv_vfmul_vf_f32m1(v1, inV1, 3);
	__riscv_vse32_v_f32m1(res.mF32, mul, 3);
	return res;
#else
	return Vec3(inV1 * inV2.mF32[0], inV1 * inV2.mF32[1], inV1 * inV2.mF32[2]);
#endif
}

Vec3 Vec3::operator / (float inV2) const
{
#if defined(JPH_USE_SSE)
	return _mm_div_ps(mValue, _mm_set1_ps(inV2));
#elif defined(JPH_USE_NEON)
	return vdivq_f32(mValue, vdupq_n_f32(inV2));
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t div = __riscv_vfdiv_vf_f32m1(v1, inV2, 3);
	__riscv_vse32_v_f32m1(res.mF32, div, 3);
	return res;
#else
	return Vec3(mF32[0] / inV2, mF32[1] / inV2, mF32[2] / inV2);
#endif
}

Vec3 &Vec3::operator *= (float inV2)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_mul_ps(mValue, _mm_set1_ps(inV2));
#elif defined(JPH_USE_NEON)
	mValue = vmulq_n_f32(mValue, inV2);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t res = __riscv_vfmul_vf_f32m1(v1, inV2, 3);
	__riscv_vse32_v_f32m1(mF32, res, 3);
#else
	for (int i = 0; i < 3; ++i)
		mF32[i] *= inV2;
	#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
		mF32[3] = mF32[2];
	#endif
#endif
	return *this;
}

Vec3 &Vec3::operator *= (Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_mul_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	mValue = vmulq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_res = __riscv_vfmul_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(mF32, rvv_res, 3);
#else
	for (int i = 0; i < 3; ++i)
		mF32[i] *= inV2.mF32[i];
	#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
		mF32[3] = mF32[2];
	#endif
#endif
	return *this;
}

Vec3 &Vec3::operator /= (float inV2)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_div_ps(mValue, _mm_set1_ps(inV2));
#elif defined(JPH_USE_NEON)
	mValue = vdivq_f32(mValue, vdupq_n_f32(inV2));
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t res = __riscv_vfdiv_vf_f32m1(v, inV2, 3);
	__riscv_vse32_v_f32m1(mF32, res, 3);
#else
	for (int i = 0; i < 3; ++i)
		mF32[i] /= inV2;
	#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
		mF32[3] = mF32[2];
	#endif
#endif
	return *this;
}

Vec3 Vec3::operator + (Vec3Arg inV2) const
{
#if defined(JPH_USE_SSE)
	return _mm_add_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vaddq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_add = __riscv_vfadd_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_add, 3);
	return res;
#else
	return Vec3(mF32[0] + inV2.mF32[0], mF32[1] + inV2.mF32[1], mF32[2] + inV2.mF32[2]);
#endif
}

Vec3 &Vec3::operator += (Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_add_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	mValue = vaddq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_add = __riscv_vfadd_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(mF32, rvv_add, 3);
#else
	for (int i = 0; i < 3; ++i)
		mF32[i] += inV2.mF32[i];
	#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
		mF32[3] = mF32[2];
	#endif
#endif
	return *this;
}

Vec3 Vec3::operator - () const
{
#if defined(JPH_USE_SSE)
	return _mm_sub_ps(_mm_setzero_ps(), mValue);
#elif defined(JPH_USE_NEON)
	#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
		return vsubq_f32(vdupq_n_f32(0), mValue);
	#else
		return vnegq_f32(mValue);
	#endif
#elif defined(JPH_USE_RVV)
	#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
		Vec3 res;
		const vfloat32m1_t rvv_zero = __riscv_vfmv_v_f_f32m1(0.0f, 3);
		const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
		const vfloat32m1_t rvv_neg = __riscv_vfsub_vv_f32m1(rvv_zero, v, 3);
		__riscv_vse32_v_f32m1(res.mF32, rvv_neg, 3);
		return res;
	#else
		Vec3 res;
		const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
		const vfloat32m1_t rvv_neg = __riscv_vfsgnjn_vv_f32m1(v, v, 3);
		__riscv_vse32_v_f32m1(res.mF32, rvv_neg, 3);
		return res;
	#endif
#else
	#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
		return Vec3(0.0f - mF32[0], 0.0f - mF32[1], 0.0f - mF32[2]);
	#else
		return Vec3(-mF32[0], -mF32[1], -mF32[2]);
	#endif
#endif
}

Vec3 Vec3::operator - (Vec3Arg inV2) const
{
#if defined(JPH_USE_SSE)
	return _mm_sub_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vsubq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_sub = __riscv_vfsub_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_sub, 3);
	return res;
#else
	return Vec3(mF32[0] - inV2.mF32[0], mF32[1] - inV2.mF32[1], mF32[2] - inV2.mF32[2]);
#endif
}

Vec3 &Vec3::operator -= (Vec3Arg inV2)
{
#if defined(JPH_USE_SSE)
	mValue = _mm_sub_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	mValue = vsubq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_sub = __riscv_vfsub_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(mF32, rvv_sub, 3);
#else
	for (int i = 0; i < 3; ++i)
		mF32[i] -= inV2.mF32[i];
	#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
		mF32[3] = mF32[2];
	#endif
#endif
	return *this;
}

Vec3 Vec3::operator / (Vec3Arg inV2) const
{
	inV2.CheckW(); // Check W equals Z to avoid div by zero
#if defined(JPH_USE_SSE)
	return _mm_div_ps(mValue, inV2.mValue);
#elif defined(JPH_USE_NEON)
	return vdivq_f32(mValue, inV2.mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v2 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	const vfloat32m1_t rvv_div = __riscv_vfdiv_vv_f32m1(v1, v2, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_div, 3);
	return res;
#else
	return Vec3(mF32[0] / inV2.mF32[0], mF32[1] / inV2.mF32[1], mF32[2] / inV2.mF32[2]);
#endif
}

Vec4 Vec3::SplatX() const
{
#if defined(JPH_USE_SSE)
	return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(0, 0, 0, 0));
#elif defined(JPH_USE_NEON)
	return vdupq_laneq_f32(mValue, 0);
#elif defined(JPH_USE_RVV)
	Vec4 vec;
	const vfloat32m1_t splat = __riscv_vfmv_v_f_f32m1(mF32[0], 4);
	__riscv_vse32_v_f32m1(vec.mF32, splat, 4);
	return vec;
#else
	return Vec4(mF32[0], mF32[0], mF32[0], mF32[0]);
#endif
}

Vec4 Vec3::SplatY() const
{
#if defined(JPH_USE_SSE)
	return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(1, 1, 1, 1));
#elif defined(JPH_USE_NEON)
	return vdupq_laneq_f32(mValue, 1);
#elif defined(JPH_USE_RVV)
	Vec4 vec;
	const vfloat32m1_t splat = __riscv_vfmv_v_f_f32m1(mF32[1], 4);
	__riscv_vse32_v_f32m1(vec.mF32, splat, 4);
	return vec;
#else
	return Vec4(mF32[1], mF32[1], mF32[1], mF32[1]);
#endif
}

Vec4 Vec3::SplatZ() const
{
#if defined(JPH_USE_SSE)
	return _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(2, 2, 2, 2));
#elif defined(JPH_USE_NEON)
	return vdupq_laneq_f32(mValue, 2);
#elif defined(JPH_USE_RVV)
	Vec4 vec;
	const vfloat32m1_t splat = __riscv_vfmv_v_f_f32m1(mF32[2], 4);
	__riscv_vse32_v_f32m1(vec.mF32, splat, 4);
	return vec;
#else
	return Vec4(mF32[2], mF32[2], mF32[2], mF32[2]);
#endif
}

int Vec3::GetLowestComponentIndex() const
{
	return GetX() < GetY() ? (GetZ() < GetX() ? 2 : 0) : (GetZ() < GetY() ? 2 : 1);
}

int Vec3::GetHighestComponentIndex() const
{
	return GetX() > GetY() ? (GetZ() > GetX() ? 2 : 0) : (GetZ() > GetY() ? 2 : 1);
}

Vec3 Vec3::Abs() const
{
#if defined(JPH_USE_AVX512)
	return _mm_range_ps(mValue, mValue, 0b1000);
#elif defined(JPH_USE_SSE)
	return _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), mValue), mValue);
#elif defined(JPH_USE_NEON)
	return vabsq_f32(mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t rvv_abs = __riscv_vfsgnj_vf_f32m1(v, 1.0, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_abs, 3);
	return res;
#else
	return Vec3(abs(mF32[0]), abs(mF32[1]), abs(mF32[2]));
#endif
}

Vec3 Vec3::Reciprocal() const
{
	return sOne() / mValue;
}

Vec3 Vec3::sDifferenceOfProducts(Vec3Arg inA, Vec3Arg inB, Vec3Arg inC, Vec3Arg inD)
{
#ifdef JPH_USE_FMADD
	Vec3 cd = inC * inD;
	Vec3 err = Vec3::sFusedMultiplyAdd(-inC, inD, cd);
	Vec3 dop = Vec3::sFusedMultiplyAdd(inA, inB, -cd);
	return dop + err;
#else
	return inA * inB - inC * inD;
#endif
}

Vec3 Vec3::Cross(Vec3Arg inV2) const
{
#if defined(JPH_USE_SSE)
	Type t1 = _mm_shuffle_ps(inV2.mValue, inV2.mValue, _MM_SHUFFLE(0, 0, 2, 1)); // Assure Z and W are the same
	t1 = _mm_mul_ps(t1, mValue);
	Type t2 = _mm_shuffle_ps(mValue, mValue, _MM_SHUFFLE(0, 0, 2, 1)); // Assure Z and W are the same
	t2 = _mm_mul_ps(t2, inV2.mValue);
	Type t3 = _mm_sub_ps(t1, t2);
	return _mm_shuffle_ps(t3, t3, _MM_SHUFFLE(0, 0, 2, 1)); // Assure Z and W are the same
#elif defined(JPH_USE_NEON)
	Type t1 = JPH_NEON_SHUFFLE_F32x4(inV2.mValue, inV2.mValue, 1, 2, 0, 0); // Assure Z and W are the same
	t1 = vmulq_f32(t1, mValue);
	Type t2 = JPH_NEON_SHUFFLE_F32x4(mValue, mValue, 1, 2, 0, 0); // Assure Z and W are the same
	t2 = vmulq_f32(t2, inV2.mValue);
	Type t3 = vsubq_f32(t1, t2);
	return JPH_NEON_SHUFFLE_F32x4(t3, t3, 1, 2, 0, 0); // Assure Z and W are the same
#elif defined(JPH_USE_RVV)
	const uint32 indices[3] = { 1, 2, 0 };
	const vuint32m1_t gather_indices = __riscv_vle32_v_u32m1(indices, 3);
	const vfloat32m1_t v0 = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t v1 = __riscv_vle32_v_f32m1(inV2.mF32, 3);
	vfloat32m1_t t0 = __riscv_vrgather_vv_f32m1(v1, gather_indices, 3);
	t0 = __riscv_vfmul_vv_f32m1(t0, v0, 3);
	vfloat32m1_t t1 = __riscv_vrgather_vv_f32m1(v0, gather_indices, 3);
	t1 = __riscv_vfmul_vv_f32m1(t1, v1, 3);
	const vfloat32m1_t sub = __riscv_vfsub_vv_f32m1(t0, t1, 3);
	const vfloat32m1_t cross = __riscv_vrgather_vv_f32m1(sub, gather_indices, 3);

	Vec3 cross_result;
	__riscv_vse32_v_f32m1(cross_result.mF32, cross, 3);
	return cross_result;
#else
	return Vec3(mF32[1] * inV2.mF32[2] - mF32[2] * inV2.mF32[1],
				mF32[2] * inV2.mF32[0] - mF32[0] * inV2.mF32[2],
				mF32[0] * inV2.mF32[1] - mF32[1] * inV2.mF32[0]);
#endif
}

Vec3 Vec3::CrossPrecise(Vec3Arg inV2) const
{
	return sDifferenceOfProducts(*this, inV2.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>(), inV2).Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
}

float Vec3::ReduceSum() const
{
	// Ensure that we handle -0.0f correctly when cross platform deterministic behavior is required.
#if defined(JPH_USE_SSE4_1)
	#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
		Type val = _mm_blend_ps(mValue, _mm_setzero_ps(), 0x8); // [x, y, z, 0]
		Type shuf = _mm_movehdup_ps(val); // [y, y, 0, 0]
		Type sums = _mm_add_ps(val, shuf); // [x + y, y + y, z + 0, 0]
		shuf = _mm_movehl_ps(shuf, sums); // [z + 0, 0, 0, 0]
	#else
		Type shuf = _mm_movehdup_ps(mValue); // [y, y, w, w]
		Type sums = _mm_add_ps(mValue, shuf); // [x + y, y + y, z + w, w + w]
		shuf = _mm_movehl_ps(mValue, mValue); // [z, w, z, w]
	#endif
	sums = _mm_add_ps(sums, shuf); // Deterministic: [(x + y) + (z + 0), ...], non-deterministic: [(x + y) + z, ...]
	return _mm_cvtss_f32(sums);
#elif defined(JPH_USE_NEON)
	Type v = vsetq_lane_f32(0, mValue, 3);
	return vaddvq_f32(v);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t zeros = __riscv_vfmv_v_f_f32m1(0.0f, 3);
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t sum = __riscv_vfredosum_vs_f32m1_f32m1(v, zeros, 3);
	return __riscv_vfmv_f_s_f32m1_f32(sum);
#else
	#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
		return (mF32[0] + mF32[1]) + (mF32[2] + 0.0f);
	#else
		return mF32[0] + mF32[1] + mF32[2];
	#endif
#endif
}

float Vec3::Dot(Vec3Arg inV2) const
{
	return (*this * inV2).ReduceSum();
}

Vec3 Vec3::DotV(Vec3Arg inV2) const
{
	return Vec3::sReplicate(Dot(inV2));
}

Vec4 Vec3::DotV4(Vec3Arg inV2) const
{
	return Vec4::sReplicate(Dot(inV2));
}

float Vec3::LengthSq() const
{
	return Dot(*this);
}

float Vec3::Length() const
{
	return JPH::Sqrt(LengthSq());
}

Vec3 Vec3::Sqrt() const
{
#if defined(JPH_USE_SSE)
	return _mm_sqrt_ps(mValue);
#elif defined(JPH_USE_NEON)
	return vsqrtq_f32(mValue);
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t rvv_sqrt = __riscv_vfsqrt_v_f32m1(v, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_sqrt, 3);
	return res;
#else
	return Vec3(JPH::Sqrt(mF32[0]), JPH::Sqrt(mF32[1]), JPH::Sqrt(mF32[2]));
#endif
}

Vec3 Vec3::Normalized() const
{
	return *this / Length();
}

Vec3 Vec3::NormalizedOr(Vec3Arg inZeroValue) const
{
#if defined(JPH_USE_SSE4_1) && !defined(JPH_PLATFORM_WASM) // _mm_blendv_ps has problems on FireFox
	Type mul = _mm_mul_ps(mValue, mValue);
	Type shuf = _mm_movehdup_ps(mul);
	Type sums = _mm_add_ps(mul, shuf);
	shuf = _mm_movehl_ps(mul, mul);
	sums = _mm_add_ps(sums, shuf);
	Type len_sq = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 0, 0, 0));
	// clang with '-ffast-math' (which you should not use!) can generate _mm_rsqrt_ps
	// instructions which produce INFs/NaNs when they get a denormal float as input.
	// We therefore treat denormals as zero here.
	Type is_zero = _mm_cmple_ps(len_sq, _mm_set1_ps(FLT_MIN));
#ifdef JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
	if (_mm_movemask_ps(is_zero) == 0xf)
		return inZeroValue;
	else
		return _mm_div_ps(mValue, _mm_sqrt_ps(len_sq));
#else
	return _mm_blendv_ps(_mm_div_ps(mValue, _mm_sqrt_ps(len_sq)), inZeroValue.mValue, is_zero);
#endif // JPH_FLOATING_POINT_EXCEPTIONS_ENABLED
#elif defined(JPH_USE_NEON)
	float32x4_t mul = vmulq_f32(mValue, mValue);
	mul = vsetq_lane_f32(0, mul, 3);
	float32x4_t len_sq = vdupq_n_f32(vaddvq_f32(mul));
	uint32x4_t is_zero = vcleq_f32(len_sq, vdupq_n_f32(FLT_MIN));
	return vbslq_f32(is_zero, inZeroValue.mValue, vdivq_f32(mValue, vsqrtq_f32(len_sq)));
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t src = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t zeros = __riscv_vfmv_v_f_f32m1(0.0f, 3);
	const vfloat32m1_t mul = __riscv_vfmul_vv_f32m1(src, src, 3);
	const vfloat32m1_t sum = __riscv_vfredosum_vs_f32m1_f32m1(mul, zeros, 3);
	const float dot = __riscv_vfmv_f_s_f32m1_f32(sum);
	if (dot <= FLT_MIN)
		return inZeroValue;

	const vfloat32m1_t splat = __riscv_vrgather_vx_f32m1(sum, 0, 3);
	const vfloat32m1_t length = __riscv_vfsqrt_v_f32m1(splat, 3);

	Vec3 v;
	const vfloat32m1_t norm = __riscv_vfdiv_vv_f32m1(src, length, 3);
	__riscv_vse32_v_f32m1(v.mF32, norm, 3);
	return v;
#else
	float len_sq = LengthSq();
	if (len_sq <= FLT_MIN)
		return inZeroValue;
	else
		return *this / JPH::Sqrt(len_sq);
#endif
}

bool Vec3::IsNormalized(float inTolerance) const
{
	return abs(LengthSq() - 1.0f) <= inTolerance;
}

bool Vec3::IsNaN() const
{
#if defined(JPH_USE_AVX512)
	return (_mm_fpclass_ps_mask(mValue, 0b10000001) & 0x7) != 0;
#elif defined(JPH_USE_SSE)
	return (_mm_movemask_ps(_mm_cmpunord_ps(mValue, mValue)) & 0x7) != 0;
#elif defined(JPH_USE_NEON)
	uint32x4_t mask = JPH_NEON_UINT32x4(1, 1, 1, 0);
	uint32x4_t is_equal = vceqq_f32(mValue, mValue); // If a number is not equal to itself it's a NaN
	return vaddvq_u32(vandq_u32(is_equal, mask)) != 3;
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	const vbool32_t mask = __riscv_vmfeq_vv_f32m1_b32(v, v, 3);
	const uint32 eq = __riscv_vcpop_m_b32(mask, 3);
	return eq != 3;
#else
	return isnan(mF32[0]) || isnan(mF32[1]) || isnan(mF32[2]);
#endif
}

void Vec3::StoreFloat3(Float3 *outV) const
{
#if defined(JPH_USE_SSE)
	_mm_store_ss(&outV->x, mValue);
	Vec3 t = Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_UNUSED>();
	_mm_store_ss(&outV->y, t.mValue);
	t = t.Swizzle<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_UNUSED>();
	_mm_store_ss(&outV->z, t.mValue);
#elif defined(JPH_USE_NEON)
	float32x2_t xy = vget_low_f32(mValue);
	vst1_f32(&outV->x, xy);
	vst1q_lane_f32(&outV->z, mValue, 2);
#elif defined(JPH_USE_RVV)
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 3);
	__riscv_vse32_v_f32m1(&outV->x, v, 3);
#else
	outV->x = mF32[0];
	outV->y = mF32[1];
	outV->z = mF32[2];
#endif
}

UVec4 Vec3::ToInt() const
{
#if defined(JPH_USE_SSE)
	return _mm_cvttps_epi32(mValue);
#elif defined(JPH_USE_NEON)
	return vcvtq_u32_f32(mValue);
#elif defined(JPH_USE_RVV)
	UVec4 res;
	const vfloat32m1_t v = __riscv_vle32_v_f32m1(mF32, 4);
	const vuint32m1_t cast = __riscv_vfcvt_rtz_xu_f_v_u32m1(v, 4);
	__riscv_vse32_v_u32m1(res.mU32, cast, 4);
	return res;
#else
	return UVec4(uint32(mF32[0]), uint32(mF32[1]), uint32(mF32[2]), uint32(mF32[3]));
#endif
}

UVec4 Vec3::ReinterpretAsInt() const
{
#if defined(JPH_USE_SSE)
	return UVec4(_mm_castps_si128(mValue));
#elif defined(JPH_USE_NEON)
	return vreinterpretq_u32_f32(mValue);
#else
	return *reinterpret_cast<const UVec4 *>(this);
#endif
}

float Vec3::ReduceMin() const
{
	Vec3 v = sMin(mValue, Swizzle<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_Z>());
	v = sMin(v, v.Swizzle<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
	return v.GetX();
}

float Vec3::ReduceMax() const
{
	Vec3 v = sMax(mValue, Swizzle<SWIZZLE_Y, SWIZZLE_UNUSED, SWIZZLE_Z>());
	v = sMax(v, v.Swizzle<SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>());
	return v.GetX();
}

Vec3 Vec3::GetNormalizedPerpendicular() const
{
#if defined(JPH_USE_SSE)
	// Build both perpendicular candidates without explicit masking:
	// perp_x = [z, 0, -x, 0] (used when |x| > |y|)
	// perp_y = [0, z, -y, 0] (used when |x| <= |y|)
	__m128 zero = _mm_setzero_ps();
	__m128 neg = _mm_sub_ps(zero, mValue);
	__m128 perp_x = _mm_shuffle_ps(_mm_unpackhi_ps(mValue, zero), neg, _MM_SHUFFLE(0, 0, 1, 0));
	__m128 perp_y = _mm_shuffle_ps(_mm_unpackhi_ps(zero, mValue), neg, _MM_SHUFFLE(1, 1, 1, 0));

	// Compare squared components instead of absolute values (saves the abs computation).
	__m128 sq = _mm_mul_ps(mValue, mValue);
	__m128 xx = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 yy = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 zz = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 x_gt_y = _mm_cmpgt_ps(xx, yy);

	// Select perpendicular based on |x| > |y|.
#if defined(JPH_USE_SSE4_1) && !defined(JPH_PLATFORM_WASM) // _mm_blendv_ps has problems on FireFox
	__m128 result = _mm_blendv_ps(perp_y, perp_x, x_gt_y);
#else
	__m128 result = _mm_or_ps(_mm_and_ps(x_gt_y, perp_x), _mm_andnot_ps(x_gt_y, perp_y));
#endif

	// Normalize. Since the result has only two nonzero components; one of x^2 / y^2 plus z^2; the squared length is max(xx, yy) + zz. All lanes of the sqrt input are identical.
	__m128 len = _mm_sqrt_ps(_mm_add_ps(_mm_max_ps(xx, yy), zz));
	return _mm_div_ps(result, len);
#else
	float x = mF32[0], y = mF32[1], z = mF32[2];
	float xx = x * x, yy = y * y, zz = z * z;
#ifdef JPH_CROSS_PLATFORM_DETERMINISTIC
	Vec3 perp_x(z, 0.0f, 0.0f - x);
	Vec3 perp_y(0.0f, z, 0.0f - y);
#else
	Vec3 perp_x(z, 0.0f, -x);
	Vec3 perp_y(0.0f, z, -y);
#endif // JPH_CROSS_PLATFORM_DETERMINISTIC
	return (xx > yy ? perp_x : perp_y) / JPH::Sqrt(max(xx, yy) + zz);
#endif // JPH_USE_SSE
}

Vec3 Vec3::GetSign() const
{
#if defined(JPH_USE_AVX512)
	Type one = _mm_set1_ps(1.0f);
	return _mm_or_ps(_mm_fixupimm_ps(mValue, mValue, _mm_set1_epi32(0xA9A90100), 0), one);
#elif defined(JPH_USE_SSE)
	Type minus_one = _mm_set1_ps(-1.0f);
	Type one = _mm_set1_ps(1.0f);
	return _mm_or_ps(_mm_and_ps(mValue, minus_one), one);
#elif defined(JPH_USE_NEON)
	Type minus_one = vdupq_n_f32(-1.0f);
	Type one = vdupq_n_f32(1.0f);
	return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(mValue), vreinterpretq_u32_f32(minus_one)), vreinterpretq_u32_f32(one)));
#elif defined(JPH_USE_RVV)
	Vec3 res;
	const vfloat32m1_t rvv_in = __riscv_vle32_v_f32m1(mF32, 3);
	const vfloat32m1_t rvv_one = __riscv_vfmv_v_f_f32m1(1.0, 3);
	const vfloat32m1_t rvv_signs = __riscv_vfsgnj_vv_f32m1(rvv_one, rvv_in, 3);
	__riscv_vse32_v_f32m1(res.mF32, rvv_signs, 3);
	return res;
#else
	return Vec3(std::signbit(mF32[0])? -1.0f : 1.0f,
				std::signbit(mF32[1])? -1.0f : 1.0f,
				std::signbit(mF32[2])? -1.0f : 1.0f);
#endif
}

template <int X, int Y, int Z>
JPH_INLINE Vec3 Vec3::FlipSign() const
{
	static_assert(X == 1 || X == -1, "X must be 1 or -1");
	static_assert(Y == 1 || Y == -1, "Y must be 1 or -1");
	static_assert(Z == 1 || Z == -1, "Z must be 1 or -1");
	return Vec3::sXor(*this, Vec3(X > 0? 0.0f : -0.0f, Y > 0? 0.0f : -0.0f, Z > 0? 0.0f : -0.0f));
}

uint32 Vec3::CompressUnitVector() const
{
	constexpr float cOneOverSqrt2 = 0.70710678f;
	constexpr uint cNumBits = 14;
	constexpr uint cMask = (1 << cNumBits) - 1;
	constexpr uint cMaxValue = cMask - 1; // Need odd number of buckets to quantize to or else we can't encode 0
	constexpr float cScale = float(cMaxValue) / (2.0f * cOneOverSqrt2);

	// Store sign bit
	Vec3 v = *this;
	uint32 max_element = v.Abs().GetHighestComponentIndex();
	uint32 value = 0;
	if (v[max_element] < 0.0f)
	{
		value = 0x80000000u;
		v = -v;
	}

	// Store highest component
	value |= max_element << 29;

	// Store the other two components in a compressed format
	UVec4 compressed = Vec3::sClamp((v + Vec3::sReplicate(cOneOverSqrt2)) * cScale + Vec3::sReplicate(0.5f), Vec3::sZero(), Vec3::sReplicate(cMaxValue)).ToInt();
	switch (max_element)
	{
	case 0:
		compressed = compressed.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>();
		break;

	case 1:
		compressed = compressed.Swizzle<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_UNUSED, SWIZZLE_UNUSED>();
		break;
	}

	value |= compressed.GetX();
	value |= compressed.GetY() << cNumBits;
	return value;
}

Vec3 Vec3::sDecompressUnitVector(uint32 inValue)
{
	constexpr float cOneOverSqrt2 = 0.70710678f;
	constexpr uint cNumBits = 14;
	constexpr uint cMask = (1u << cNumBits) - 1;
	constexpr uint cMaxValue = cMask - 1; // Need odd number of buckets to quantize to or else we can't encode 0
	constexpr int cHalfMaxValue = int(cMaxValue >> 1);
	constexpr float cScale = 2.0f * cOneOverSqrt2 / float(cMaxValue);

	// Restore two components
	Vec3 v = Vec3(float(int(inValue & cMask) - cHalfMaxValue), float(int((inValue >> cNumBits) & cMask) - cHalfMaxValue), 0) * cScale;
	JPH_ASSERT(v.GetZ() == 0.0f);

	// Restore the highest component
	v.SetZ(JPH::Sqrt(max(1.0f - v.LengthSq(), 0.0f)));

	// Extract sign
	if ((inValue & 0x80000000u) != 0)
		v = -v;

	// Swizzle the components in place
	switch ((inValue >> 29) & 3)
	{
	case 0:
		v = v.Swizzle<SWIZZLE_Z, SWIZZLE_X, SWIZZLE_Y>();
		break;

	case 1:
		v = v.Swizzle<SWIZZLE_X, SWIZZLE_Z, SWIZZLE_Y>();
		break;
	}

	return v;
}

JPH_NAMESPACE_END
