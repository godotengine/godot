// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Math/MathTypes.h>

JPH_NAMESPACE_BEGIN

class [[nodiscard]] Vec8
{
public:
	JPH_OVERRIDE_NEW_DELETE

	/// Constructor
								Vec8() = default; ///< Intentionally not initialized for performance reasons
								Vec8(const Vec8 &inRHS) = default;
	JPH_INLINE					Vec8(__m256 inRHS) : mValue(inRHS)				{ }

	/// Set 256 bit vector from 2 128 bit vectors
	JPH_INLINE					Vec8(Vec4Arg inLo, Vec4Arg inHi);

	/// Vector with all zeros
	static JPH_INLINE Vec8		sZero();

	/// Replicate across all components
	static JPH_INLINE Vec8		sReplicate(float inV);

	/// Replicate the X component of inV to all components
	static JPH_INLINE Vec8		sSplatX(Vec4Arg inV);

	/// Replicate the Y component of inV to all components
	static JPH_INLINE Vec8		sSplatY(Vec4Arg inV);

	/// Replicate the Z component of inV to all components
	static JPH_INLINE Vec8		sSplatZ(Vec4Arg inV);

	/// Calculates inMul1 * inMul2 + inAdd
	static JPH_INLINE Vec8		sFusedMultiplyAdd(Vec8Arg inMul1, Vec8Arg inMul2, Vec8Arg inAdd);

	/// Component wise select, returns inV1 when highest bit of inControl = 0 and inV2 when highest bit of inControl = 1
	static JPH_INLINE Vec8		sSelect(Vec8Arg inV1, Vec8Arg inV2, UVec8Arg inControl);

	/// Component wise min
	static JPH_INLINE Vec8		sMin(Vec8Arg inV1, Vec8Arg inV2);

	/// Component wise max
	static JPH_INLINE Vec8		sMax(Vec8Arg inV1, Vec8Arg inV2);

	/// Less than
	static JPH_INLINE UVec8		sLess(Vec8Arg inV1, Vec8Arg inV2);

	/// Greater than
	static JPH_INLINE UVec8		sGreater(Vec8Arg inV1, Vec8Arg inV2);

	/// Load from memory
	static JPH_INLINE Vec8		sLoadFloat8(const float *inV);

	/// Load 8 floats from memory, 32 bytes aligned
	static JPH_INLINE Vec8		sLoadFloat8Aligned(const float *inV);

	/// Get float component by index
	JPH_INLINE float			operator [] (uint inCoordinate) const			{ JPH_ASSERT(inCoordinate < 8); return mF32[inCoordinate]; }
	JPH_INLINE float &			operator [] (uint inCoordinate)					{ JPH_ASSERT(inCoordinate < 8); return mF32[inCoordinate]; }

	/// Multiply two float vectors
	JPH_INLINE Vec8				operator * (Vec8Arg inV2) const;

	/// Multiply vector by float
	JPH_INLINE Vec8				operator * (float inV2) const;

	/// Add two float vectors
	JPH_INLINE Vec8				operator + (Vec8Arg inV2) const;

	/// Subtract two float vectors
	JPH_INLINE Vec8				operator - (Vec8Arg inV2) const;

	/// Divide
	JPH_INLINE Vec8				operator / (Vec8Arg inV2) const;

	/// Reciprocal vector
	JPH_INLINE Vec8				Reciprocal() const;

	/// 256 bit variant of Vec::Swizzle (no cross 128 bit lane swizzle)
	template<uint32 SwizzleX, uint32 SwizzleY, uint32 SwizzleZ, uint32 SwizzleW>
	JPH_INLINE Vec8				Swizzle() const;

	/// Get absolute value of all components
	JPH_INLINE Vec8				Abs() const;

	/// Fetch the lower 128 bit from a 256 bit variable
	JPH_INLINE Vec4				LowerVec4() const;

	/// Fetch the higher 128 bit from a 256 bit variable
	JPH_INLINE Vec4				UpperVec4() const;

	/// Get the minimum value of the 8 floats
	JPH_INLINE float			ReduceMin() const;

	union
	{
		__m256					mValue;
		float					mF32[8];
	};
};

static_assert(is_trivial<Vec8>(), "Is supposed to be a trivial type!");

JPH_NAMESPACE_END

#include "Vec8.inl"
