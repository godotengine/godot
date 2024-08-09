/// @ref core
/// @file glm/integer.hpp
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
///
/// @defgroup core_func_integer Integer functions
/// @ingroup core
///
/// Provides GLSL functions on integer types
///
/// These all operate component-wise. The description is per component.
/// The notation [a, b] means the set of bits from bit-number a through bit-number
/// b, inclusive. The lowest-order bit is bit 0.
///
/// Include <glm/integer.hpp> to use these core features.

#pragma once

#include "detail/qualifier.hpp"
#include "common.hpp"
#include "vector_relational.hpp"

namespace glm
{
	/// @addtogroup core_func_integer
	/// @{

	/// Adds 32-bit unsigned integer x and y, returning the sum
	/// modulo pow(2, 32). The value carry is set to 0 if the sum was
	/// less than pow(2, 32), or to 1 otherwise.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/uaddCarry.xml">GLSL uaddCarry man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, qualifier Q>
	GLM_FUNC_DECL vec<L, uint, Q> uaddCarry(
		vec<L, uint, Q> const& x,
		vec<L, uint, Q> const& y,
		vec<L, uint, Q> & carry);

	/// Subtracts the 32-bit unsigned integer y from x, returning
	/// the difference if non-negative, or pow(2, 32) plus the difference
	/// otherwise. The value borrow is set to 0 if x >= y, or to 1 otherwise.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/usubBorrow.xml">GLSL usubBorrow man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, qualifier Q>
	GLM_FUNC_DECL vec<L, uint, Q> usubBorrow(
		vec<L, uint, Q> const& x,
		vec<L, uint, Q> const& y,
		vec<L, uint, Q> & borrow);

	/// Multiplies 32-bit integers x and y, producing a 64-bit
	/// result. The 32 least-significant bits are returned in lsb.
	/// The 32 most-significant bits are returned in msb.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/umulExtended.xml">GLSL umulExtended man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, qualifier Q>
	GLM_FUNC_DECL void umulExtended(
		vec<L, uint, Q> const& x,
		vec<L, uint, Q> const& y,
		vec<L, uint, Q> & msb,
		vec<L, uint, Q> & lsb);

	/// Multiplies 32-bit integers x and y, producing a 64-bit
	/// result. The 32 least-significant bits are returned in lsb.
	/// The 32 most-significant bits are returned in msb.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/imulExtended.xml">GLSL imulExtended man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, qualifier Q>
	GLM_FUNC_DECL void imulExtended(
		vec<L, int, Q> const& x,
		vec<L, int, Q> const& y,
		vec<L, int, Q> & msb,
		vec<L, int, Q> & lsb);

	/// Extracts bits [offset, offset + bits - 1] from value,
	/// returning them in the least significant bits of the result.
	/// For unsigned data types, the most significant bits of the
	/// result will be set to zero. For signed data types, the
	/// most significant bits will be set to the value of bit offset + base - 1.
	///
	/// If bits is zero, the result will be zero. The result will be
	/// undefined if offset or bits is negative, or if the sum of
	/// offset and bits is greater than the number of bits used
	/// to store the operand.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldExtract.xml">GLSL bitfieldExtract man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> bitfieldExtract(
		vec<L, T, Q> const& Value,
		int Offset,
		int Bits);

	/// Returns the insertion the bits least-significant bits of insert into base.
	///
	/// The result will have bits [offset, offset + bits - 1] taken
	/// from bits [0, bits - 1] of insert, and all other bits taken
	/// directly from the corresponding bits of base. If bits is
	/// zero, the result will simply be base. The result will be
	/// undefined if offset or bits is negative, or if the sum of
	/// offset and bits is greater than the number of bits used to
	/// store the operand.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldInsert.xml">GLSL bitfieldInsert man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> bitfieldInsert(
		vec<L, T, Q> const& Base,
		vec<L, T, Q> const& Insert,
		int Offset,
		int Bits);

	/// Returns the reversal of the bits of value.
	/// The bit numbered n of the result will be taken from bit (bits - 1) - n of value,
	/// where bits is the total number of bits used to represent value.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitfieldReverse.xml">GLSL bitfieldReverse man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> bitfieldReverse(vec<L, T, Q> const& v);

	/// Returns the number of bits set to 1 in the binary representation of value.
	///
	/// @tparam genType Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitCount.xml">GLSL bitCount man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<typename genType>
	GLM_FUNC_DECL int bitCount(genType v);

	/// Returns the number of bits set to 1 in the binary representation of value.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar or vector types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/bitCount.xml">GLSL bitCount man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, int, Q> bitCount(vec<L, T, Q> const& v);

	/// Returns the bit number of the least significant bit set to
	/// 1 in the binary representation of value.
	/// If value is zero, -1 will be returned.
	///
	/// @tparam genIUType Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findLSB.xml">GLSL findLSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<typename genIUType>
	GLM_FUNC_DECL int findLSB(genIUType x);

	/// Returns the bit number of the least significant bit set to
	/// 1 in the binary representation of value.
	/// If value is zero, -1 will be returned.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findLSB.xml">GLSL findLSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, int, Q> findLSB(vec<L, T, Q> const& v);

	/// Returns the bit number of the most significant bit in the binary representation of value.
	/// For positive integers, the result will be the bit number of the most significant bit set to 1.
	/// For negative integers, the result will be the bit number of the most significant
	/// bit set to 0. For a value of zero or negative one, -1 will be returned.
	///
	/// @tparam genIUType Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findMSB.xml">GLSL findMSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<typename genIUType>
	GLM_FUNC_DECL int findMSB(genIUType x);

	/// Returns the bit number of the most significant bit in the binary representation of value.
	/// For positive integers, the result will be the bit number of the most significant bit set to 1.
	/// For negative integers, the result will be the bit number of the most significant
	/// bit set to 0. For a value of zero or negative one, -1 will be returned.
	///
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Signed or unsigned integer scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/findMSB.xml">GLSL findMSB man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.8 Integer Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, int, Q> findMSB(vec<L, T, Q> const& v);

	/// @}
}//namespace glm

#include "detail/func_integer.inl"
