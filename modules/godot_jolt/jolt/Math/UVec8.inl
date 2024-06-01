// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

JPH_NAMESPACE_BEGIN

UVec8::UVec8(UVec4Arg inLo, UVec4Arg inHi) :
	mValue(_mm256_insertf128_si256(_mm256_castsi128_si256(inLo.mValue), inHi.mValue, 1))
{
}

bool UVec8::operator == (UVec8Arg inV2) const
{
	return sEquals(*this, inV2).TestAllTrue();
}

UVec8 UVec8::sReplicate(uint32 inV)
{
	return _mm256_set1_epi32(int(inV));
}

UVec8 UVec8::sSplatX(UVec4Arg inV)
{
	return _mm256_set1_epi32(inV.GetX());
}

UVec8 UVec8::sSplatY(UVec4Arg inV)
{
	return _mm256_set1_epi32(inV.GetY());
}

UVec8 UVec8::sSplatZ(UVec4Arg inV)
{
	return _mm256_set1_epi32(inV.GetZ());
}

UVec8 UVec8::sEquals(UVec8Arg inV1, UVec8Arg inV2)
{
#ifdef JPH_USE_AVX2
	return _mm256_cmpeq_epi32(inV1.mValue, inV2.mValue);
#else
	return UVec8(UVec4::sEquals(inV1.LowerVec4(), inV2.LowerVec4()), UVec4::sEquals(inV1.UpperVec4(), inV2.UpperVec4()));
#endif
}

UVec8 UVec8::sSelect(UVec8Arg inV1, UVec8Arg inV2, UVec8Arg inControl)
{
	return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(inV1.mValue), _mm256_castsi256_ps(inV2.mValue), _mm256_castsi256_ps(inControl.mValue)));
}

UVec8 UVec8::sOr(UVec8Arg inV1, UVec8Arg inV2)
{
	return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(inV1.mValue), _mm256_castsi256_ps(inV2.mValue)));
}

UVec8 UVec8::sXor(UVec8Arg inV1, UVec8Arg inV2)
{
	return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(inV1.mValue), _mm256_castsi256_ps(inV2.mValue)));
}

UVec8 UVec8::sAnd(UVec8Arg inV1, UVec8Arg inV2)
{
	return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(inV1.mValue), _mm256_castsi256_ps(inV2.mValue)));
}

template<uint32 SwizzleX, uint32 SwizzleY, uint32 SwizzleZ, uint32 SwizzleW>
UVec8 UVec8::Swizzle() const
{
	static_assert(SwizzleX <= 3, "SwizzleX template parameter out of range");
	static_assert(SwizzleY <= 3, "SwizzleY template parameter out of range");
	static_assert(SwizzleZ <= 3, "SwizzleZ template parameter out of range");
	static_assert(SwizzleW <= 3, "SwizzleW template parameter out of range");

	return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(mValue), _mm256_castsi256_ps(mValue), _MM_SHUFFLE(SwizzleW, SwizzleZ, SwizzleY, SwizzleX)));
}

bool UVec8::TestAnyTrue() const
{
	return _mm256_movemask_ps(_mm256_castsi256_ps(mValue)) != 0;
}

bool UVec8::TestAllTrue() const
{
	return _mm256_movemask_ps(_mm256_castsi256_ps(mValue)) == 0xff;
}

UVec4 UVec8::LowerVec4() const
{
	return _mm256_castsi256_si128(mValue);
}

UVec4 UVec8::UpperVec4() const
{
	return _mm_castps_si128(_mm256_extractf128_ps(_mm256_castsi256_ps(mValue), 1));
}

Vec8 UVec8::ToFloat() const
{
	return _mm256_cvtepi32_ps(mValue);
}

template <const uint Count>
UVec8 UVec8::LogicalShiftLeft() const
{
	static_assert(Count <= 31, "Invalid shift");

#ifdef JPH_USE_AVX2
	return _mm256_slli_epi32(mValue, Count);
#else
	return UVec8(LowerVec4().LogicalShiftLeft<Count>(), UpperVec4().LogicalShiftLeft<Count>());
#endif
}

template <const uint Count>
UVec8 UVec8::LogicalShiftRight() const
{
	static_assert(Count <= 31, "Invalid shift");

#ifdef JPH_USE_AVX2
	return _mm256_srli_epi32(mValue, Count);
#else
	return UVec8(LowerVec4().LogicalShiftRight<Count>(), UpperVec4().LogicalShiftRight<Count>());
#endif
}

template <const uint Count>
UVec8 UVec8::ArithmeticShiftRight() const
{
	static_assert(Count <= 31, "Invalid shift");

#ifdef JPH_USE_AVX2
	return _mm256_srai_epi32(mValue, Count);
#else
	return UVec8(LowerVec4().ArithmeticShiftRight<Count>(), UpperVec4().ArithmeticShiftRight<Count>());
#endif
}

JPH_NAMESPACE_END
