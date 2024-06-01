// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Math/UVec8.h>

JPH_NAMESPACE_BEGIN

Vec8::Vec8(Vec4Arg inLo, Vec4Arg inHi) :
	mValue(_mm256_insertf128_ps(_mm256_castps128_ps256(inLo.mValue), inHi.mValue, 1))
{
}

Vec8 Vec8::sZero()
{
	return _mm256_setzero_ps();
}

Vec8 Vec8::sReplicate(float inV)
{
	return _mm256_set1_ps(inV);
}

Vec8 Vec8::sSplatX(Vec4Arg inV)
{
	return _mm256_set1_ps(inV.GetX());
}

Vec8 Vec8::sSplatY(Vec4Arg inV)
{
	return _mm256_set1_ps(inV.GetY());
}

Vec8 Vec8::sSplatZ(Vec4Arg inV)
{
	return _mm256_set1_ps(inV.GetZ());
}

Vec8 Vec8::sFusedMultiplyAdd(Vec8Arg inMul1, Vec8Arg inMul2, Vec8Arg inAdd)
{
#ifdef JPH_USE_FMADD
	return _mm256_fmadd_ps(inMul1.mValue, inMul2.mValue, inAdd.mValue);
#else
	return _mm256_add_ps(_mm256_mul_ps(inMul1.mValue, inMul2.mValue), inAdd.mValue);
#endif
}

Vec8 Vec8::sSelect(Vec8Arg inV1, Vec8Arg inV2, UVec8Arg inControl)
{
	return _mm256_blendv_ps(inV1.mValue, inV2.mValue, _mm256_castsi256_ps(inControl.mValue));
}

Vec8 Vec8::sMin(Vec8Arg inV1, Vec8Arg inV2)
{
	return _mm256_min_ps(inV1.mValue, inV2.mValue);
}

Vec8 Vec8::sMax(Vec8Arg inV1, Vec8Arg inV2)
{
	return _mm256_max_ps(inV1.mValue, inV2.mValue);
}

UVec8 Vec8::sLess(Vec8Arg inV1, Vec8Arg inV2)
{
	return _mm256_castps_si256(_mm256_cmp_ps(inV1.mValue, inV2.mValue, _CMP_LT_OQ));
}

UVec8 Vec8::sGreater(Vec8Arg inV1, Vec8Arg inV2)
{
	return _mm256_castps_si256(_mm256_cmp_ps(inV1.mValue, inV2.mValue, _CMP_GT_OQ));
}

Vec8 Vec8::sLoadFloat8(const float *inV)
{
	return _mm256_loadu_ps(inV);
}

Vec8 Vec8::sLoadFloat8Aligned(const float *inV)
{
	return _mm256_load_ps(inV);
}

Vec8 Vec8::operator * (Vec8Arg inV2) const
{
	return _mm256_mul_ps(mValue, inV2.mValue);
}

Vec8 Vec8::operator * (float inV2) const
{
	return _mm256_mul_ps(mValue, _mm256_set1_ps(inV2));
}

Vec8 Vec8::operator + (Vec8Arg inV2) const
{
	return _mm256_add_ps(mValue, inV2.mValue);
}

Vec8 Vec8::operator - (Vec8Arg inV2) const
{
	return _mm256_sub_ps(mValue, inV2.mValue);
}

Vec8 Vec8::operator / (Vec8Arg inV2) const
{
	return _mm256_div_ps(mValue, inV2.mValue);
}

Vec8 Vec8::Reciprocal() const
{
	return Vec8::sReplicate(1.0f) / mValue;
}

template<uint32 SwizzleX, uint32 SwizzleY, uint32 SwizzleZ, uint32 SwizzleW>
Vec8 Vec8::Swizzle() const
{
	static_assert(SwizzleX <= 3, "SwizzleX template parameter out of range");
	static_assert(SwizzleY <= 3, "SwizzleY template parameter out of range");
	static_assert(SwizzleZ <= 3, "SwizzleZ template parameter out of range");
	static_assert(SwizzleW <= 3, "SwizzleW template parameter out of range");

	return _mm256_shuffle_ps(mValue, mValue, _MM_SHUFFLE(SwizzleW, SwizzleZ, SwizzleY, SwizzleX));
}

Vec8 Vec8::Abs() const
{
#if defined(JPH_USE_AVX512)
	return _mm256_range_ps(mValue, mValue, 0b1000);
#else
	return _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), mValue), mValue);
#endif
}

Vec4 Vec8::LowerVec4() const
{
	return _mm256_castps256_ps128(mValue);
}

Vec4 Vec8::UpperVec4() const
{
	return _mm256_extractf128_ps(mValue, 1);
}

float Vec8::ReduceMin() const
{
	return Vec4::sMin(LowerVec4(), UpperVec4()).ReduceMin();
}

JPH_NAMESPACE_END
