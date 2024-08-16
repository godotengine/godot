// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Math/Vec8.h>

JPH_NAMESPACE_BEGIN

class [[nodiscard]] UVec8
{
public:
	JPH_OVERRIDE_NEW_DELETE

								UVec8() = default; ///< Intentionally not initialized for performance reasons
								UVec8(const UVec8 &inRHS) = default;
	JPH_INLINE					UVec8(__m256i inRHS) : mValue(inRHS)				{ }

	/// Set 256 bit vector from 2 128 bit vectors
	JPH_INLINE					UVec8(UVec4Arg inLo, UVec4Arg inHi);

	/// Comparison
	JPH_INLINE bool				operator == (UVec8Arg inV2) const;
	JPH_INLINE bool				operator != (UVec8Arg inV2) const					{ return !(*this == inV2); }

	/// Replicate int across all components
	static JPH_INLINE UVec8		sReplicate(uint32 inV);

	/// Replicate the X component of inV to all components
	static JPH_INLINE UVec8		sSplatX(UVec4Arg inV);

	/// Replicate the Y component of inV to all components
	static JPH_INLINE UVec8		sSplatY(UVec4Arg inV);

	/// Replicate the Z component of inV to all components
	static JPH_INLINE UVec8		sSplatZ(UVec4Arg inV);

	/// Equals (component wise)
	static JPH_INLINE UVec8		sEquals(UVec8Arg inV1, UVec8Arg inV2);

	/// Component wise select, returns inV1 when highest bit of inControl = 0 and inV2 when highest bit of inControl = 1
	static JPH_INLINE UVec8		sSelect(UVec8Arg inV1, UVec8Arg inV2, UVec8Arg inControl);

	/// Logical or
	static JPH_INLINE UVec8		sOr(UVec8Arg inV1, UVec8Arg inV2);

	/// Logical xor
	static JPH_INLINE UVec8		sXor(UVec8Arg inV1, UVec8Arg inV2);

	/// Logical and
	static JPH_INLINE UVec8		sAnd(UVec8Arg inV1, UVec8Arg inV2);

	/// Get float component by index
	JPH_INLINE uint32			operator [] (uint inCoordinate) const				{ JPH_ASSERT(inCoordinate < 8); return mU32[inCoordinate]; }
	JPH_INLINE uint32 &			operator [] (uint inCoordinate)						{ JPH_ASSERT(inCoordinate < 8); return mU32[inCoordinate]; }

	/// 256 bit variant of Vec::Swizzle (no cross 128 bit lane swizzle)
	template<uint32 SwizzleX, uint32 SwizzleY, uint32 SwizzleZ, uint32 SwizzleW>
	JPH_INLINE UVec8			Swizzle() const;

	/// Test if any of the components are true (true is when highest bit of component is set)
	JPH_INLINE bool				TestAnyTrue() const;

	/// Test if all components are true (true is when highest bit of component is set)
	JPH_INLINE bool				TestAllTrue() const;

	/// Fetch the lower 128 bit from a 256 bit variable
	JPH_INLINE UVec4			LowerVec4() const;

	/// Fetch the higher 128 bit from a 256 bit variable
	JPH_INLINE UVec4			UpperVec4() const;

	/// Converts int to float
	JPH_INLINE Vec8				ToFloat() const;

	/// Shift all components by Count bits to the left (filling with zeros from the left)
	template <const uint Count>
	JPH_INLINE UVec8			LogicalShiftLeft() const;

	/// Shift all components by Count bits to the right (filling with zeros from the right)
	template <const uint Count>
	JPH_INLINE UVec8			LogicalShiftRight() const;

	/// Shift all components by Count bits to the right (shifting in the value of the highest bit)
	template <const uint Count>
	JPH_INLINE UVec8			ArithmeticShiftRight() const;

	union
	{
		__m256i					mValue;
		uint32					mU32[8];
	};
};

static_assert(is_trivial<UVec8>(), "Is supposed to be a trivial type!");

JPH_NAMESPACE_END

#include "UVec8.inl"
