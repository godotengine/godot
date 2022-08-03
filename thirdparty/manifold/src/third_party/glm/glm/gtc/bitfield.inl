/// @ref gtc_bitfield

#include "../simd/integer.h"

namespace glm{
namespace detail
{
	template<typename PARAM, typename RET>
	GLM_FUNC_DECL RET bitfieldInterleave(PARAM x, PARAM y);

	template<typename PARAM, typename RET>
	GLM_FUNC_DECL RET bitfieldInterleave(PARAM x, PARAM y, PARAM z);

	template<typename PARAM, typename RET>
	GLM_FUNC_DECL RET bitfieldInterleave(PARAM x, PARAM y, PARAM z, PARAM w);

	template<>
	GLM_FUNC_QUALIFIER glm::uint16 bitfieldInterleave(glm::uint8 x, glm::uint8 y)
	{
		glm::uint16 REG1(x);
		glm::uint16 REG2(y);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint16>(0x0F0F);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint16>(0x0F0F);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint16>(0x3333);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint16>(0x3333);

		REG1 = ((REG1 <<  1) | REG1) & static_cast<glm::uint16>(0x5555);
		REG2 = ((REG2 <<  1) | REG2) & static_cast<glm::uint16>(0x5555);

		return REG1 | static_cast<glm::uint16>(REG2 << 1);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave(glm::uint16 x, glm::uint16 y)
	{
		glm::uint32 REG1(x);
		glm::uint32 REG2(y);

		REG1 = ((REG1 <<  8) | REG1) & static_cast<glm::uint32>(0x00FF00FF);
		REG2 = ((REG2 <<  8) | REG2) & static_cast<glm::uint32>(0x00FF00FF);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint32>(0x0F0F0F0F);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint32>(0x0F0F0F0F);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint32>(0x33333333);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint32>(0x33333333);

		REG1 = ((REG1 <<  1) | REG1) & static_cast<glm::uint32>(0x55555555);
		REG2 = ((REG2 <<  1) | REG2) & static_cast<glm::uint32>(0x55555555);

		return REG1 | (REG2 << 1);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint64 bitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		glm::uint64 REG1(x);
		glm::uint64 REG2(y);

		REG1 = ((REG1 << 16) | REG1) & static_cast<glm::uint64>(0x0000FFFF0000FFFFull);
		REG2 = ((REG2 << 16) | REG2) & static_cast<glm::uint64>(0x0000FFFF0000FFFFull);

		REG1 = ((REG1 <<  8) | REG1) & static_cast<glm::uint64>(0x00FF00FF00FF00FFull);
		REG2 = ((REG2 <<  8) | REG2) & static_cast<glm::uint64>(0x00FF00FF00FF00FFull);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint64>(0x0F0F0F0F0F0F0F0Full);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint64>(0x0F0F0F0F0F0F0F0Full);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint64>(0x3333333333333333ull);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint64>(0x3333333333333333ull);

		REG1 = ((REG1 <<  1) | REG1) & static_cast<glm::uint64>(0x5555555555555555ull);
		REG2 = ((REG2 <<  1) | REG2) & static_cast<glm::uint64>(0x5555555555555555ull);

		return REG1 | (REG2 << 1);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave(glm::uint8 x, glm::uint8 y, glm::uint8 z)
	{
		glm::uint32 REG1(x);
		glm::uint32 REG2(y);
		glm::uint32 REG3(z);

		REG1 = ((REG1 << 16) | REG1) & static_cast<glm::uint32>(0xFF0000FFu);
		REG2 = ((REG2 << 16) | REG2) & static_cast<glm::uint32>(0xFF0000FFu);
		REG3 = ((REG3 << 16) | REG3) & static_cast<glm::uint32>(0xFF0000FFu);

		REG1 = ((REG1 <<  8) | REG1) & static_cast<glm::uint32>(0x0F00F00Fu);
		REG2 = ((REG2 <<  8) | REG2) & static_cast<glm::uint32>(0x0F00F00Fu);
		REG3 = ((REG3 <<  8) | REG3) & static_cast<glm::uint32>(0x0F00F00Fu);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint32>(0xC30C30C3u);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint32>(0xC30C30C3u);
		REG3 = ((REG3 <<  4) | REG3) & static_cast<glm::uint32>(0xC30C30C3u);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint32>(0x49249249u);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint32>(0x49249249u);
		REG3 = ((REG3 <<  2) | REG3) & static_cast<glm::uint32>(0x49249249u);

		return REG1 | (REG2 << 1) | (REG3 << 2);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint64 bitfieldInterleave(glm::uint16 x, glm::uint16 y, glm::uint16 z)
	{
		glm::uint64 REG1(x);
		glm::uint64 REG2(y);
		glm::uint64 REG3(z);

		REG1 = ((REG1 << 32) | REG1) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);
		REG2 = ((REG2 << 32) | REG2) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);
		REG3 = ((REG3 << 32) | REG3) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);

		REG1 = ((REG1 << 16) | REG1) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);
		REG2 = ((REG2 << 16) | REG2) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);
		REG3 = ((REG3 << 16) | REG3) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);

		REG1 = ((REG1 <<  8) | REG1) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);
		REG2 = ((REG2 <<  8) | REG2) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);
		REG3 = ((REG3 <<  8) | REG3) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);
		REG3 = ((REG3 <<  4) | REG3) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint64>(0x9249249249249249ull);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint64>(0x9249249249249249ull);
		REG3 = ((REG3 <<  2) | REG3) & static_cast<glm::uint64>(0x9249249249249249ull);

		return REG1 | (REG2 << 1) | (REG3 << 2);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint64 bitfieldInterleave(glm::uint32 x, glm::uint32 y, glm::uint32 z)
	{
		glm::uint64 REG1(x);
		glm::uint64 REG2(y);
		glm::uint64 REG3(z);

		REG1 = ((REG1 << 32) | REG1) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);
		REG2 = ((REG2 << 32) | REG2) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);
		REG3 = ((REG3 << 32) | REG3) & static_cast<glm::uint64>(0xFFFF00000000FFFFull);

		REG1 = ((REG1 << 16) | REG1) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);
		REG2 = ((REG2 << 16) | REG2) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);
		REG3 = ((REG3 << 16) | REG3) & static_cast<glm::uint64>(0x00FF0000FF0000FFull);

		REG1 = ((REG1 <<  8) | REG1) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);
		REG2 = ((REG2 <<  8) | REG2) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);
		REG3 = ((REG3 <<  8) | REG3) & static_cast<glm::uint64>(0xF00F00F00F00F00Full);

		REG1 = ((REG1 <<  4) | REG1) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);
		REG2 = ((REG2 <<  4) | REG2) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);
		REG3 = ((REG3 <<  4) | REG3) & static_cast<glm::uint64>(0x30C30C30C30C30C3ull);

		REG1 = ((REG1 <<  2) | REG1) & static_cast<glm::uint64>(0x9249249249249249ull);
		REG2 = ((REG2 <<  2) | REG2) & static_cast<glm::uint64>(0x9249249249249249ull);
		REG3 = ((REG3 <<  2) | REG3) & static_cast<glm::uint64>(0x9249249249249249ull);

		return REG1 | (REG2 << 1) | (REG3 << 2);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave(glm::uint8 x, glm::uint8 y, glm::uint8 z, glm::uint8 w)
	{
		glm::uint32 REG1(x);
		glm::uint32 REG2(y);
		glm::uint32 REG3(z);
		glm::uint32 REG4(w);

		REG1 = ((REG1 << 12) | REG1) & static_cast<glm::uint32>(0x000F000Fu);
		REG2 = ((REG2 << 12) | REG2) & static_cast<glm::uint32>(0x000F000Fu);
		REG3 = ((REG3 << 12) | REG3) & static_cast<glm::uint32>(0x000F000Fu);
		REG4 = ((REG4 << 12) | REG4) & static_cast<glm::uint32>(0x000F000Fu);

		REG1 = ((REG1 <<  6) | REG1) & static_cast<glm::uint32>(0x03030303u);
		REG2 = ((REG2 <<  6) | REG2) & static_cast<glm::uint32>(0x03030303u);
		REG3 = ((REG3 <<  6) | REG3) & static_cast<glm::uint32>(0x03030303u);
		REG4 = ((REG4 <<  6) | REG4) & static_cast<glm::uint32>(0x03030303u);

		REG1 = ((REG1 <<  3) | REG1) & static_cast<glm::uint32>(0x11111111u);
		REG2 = ((REG2 <<  3) | REG2) & static_cast<glm::uint32>(0x11111111u);
		REG3 = ((REG3 <<  3) | REG3) & static_cast<glm::uint32>(0x11111111u);
		REG4 = ((REG4 <<  3) | REG4) & static_cast<glm::uint32>(0x11111111u);

		return REG1 | (REG2 << 1) | (REG3 << 2) | (REG4 << 3);
	}

	template<>
	GLM_FUNC_QUALIFIER glm::uint64 bitfieldInterleave(glm::uint16 x, glm::uint16 y, glm::uint16 z, glm::uint16 w)
	{
		glm::uint64 REG1(x);
		glm::uint64 REG2(y);
		glm::uint64 REG3(z);
		glm::uint64 REG4(w);

		REG1 = ((REG1 << 24) | REG1) & static_cast<glm::uint64>(0x000000FF000000FFull);
		REG2 = ((REG2 << 24) | REG2) & static_cast<glm::uint64>(0x000000FF000000FFull);
		REG3 = ((REG3 << 24) | REG3) & static_cast<glm::uint64>(0x000000FF000000FFull);
		REG4 = ((REG4 << 24) | REG4) & static_cast<glm::uint64>(0x000000FF000000FFull);

		REG1 = ((REG1 << 12) | REG1) & static_cast<glm::uint64>(0x000F000F000F000Full);
		REG2 = ((REG2 << 12) | REG2) & static_cast<glm::uint64>(0x000F000F000F000Full);
		REG3 = ((REG3 << 12) | REG3) & static_cast<glm::uint64>(0x000F000F000F000Full);
		REG4 = ((REG4 << 12) | REG4) & static_cast<glm::uint64>(0x000F000F000F000Full);

		REG1 = ((REG1 <<  6) | REG1) & static_cast<glm::uint64>(0x0303030303030303ull);
		REG2 = ((REG2 <<  6) | REG2) & static_cast<glm::uint64>(0x0303030303030303ull);
		REG3 = ((REG3 <<  6) | REG3) & static_cast<glm::uint64>(0x0303030303030303ull);
		REG4 = ((REG4 <<  6) | REG4) & static_cast<glm::uint64>(0x0303030303030303ull);

		REG1 = ((REG1 <<  3) | REG1) & static_cast<glm::uint64>(0x1111111111111111ull);
		REG2 = ((REG2 <<  3) | REG2) & static_cast<glm::uint64>(0x1111111111111111ull);
		REG3 = ((REG3 <<  3) | REG3) & static_cast<glm::uint64>(0x1111111111111111ull);
		REG4 = ((REG4 <<  3) | REG4) & static_cast<glm::uint64>(0x1111111111111111ull);

		return REG1 | (REG2 << 1) | (REG3 << 2) | (REG4 << 3);
	}
}//namespace detail

	template<typename genIUType>
	GLM_FUNC_QUALIFIER genIUType mask(genIUType Bits)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genIUType>::is_integer, "'mask' accepts only integer values");

		return Bits >= sizeof(genIUType) * 8 ? ~static_cast<genIUType>(0) : (static_cast<genIUType>(1) << Bits) - static_cast<genIUType>(1);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> mask(vec<L, T, Q> const& v)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_integer, "'mask' accepts only integer values");

		return detail::functor1<vec, L, T, T, Q>::call(mask, v);
	}

	template<typename genIType>
	GLM_FUNC_QUALIFIER genIType bitfieldRotateRight(genIType In, int Shift)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genIType>::is_integer, "'bitfieldRotateRight' accepts only integer values");

		int const BitSize = static_cast<genIType>(sizeof(genIType) * 8);
		return (In << static_cast<genIType>(Shift)) | (In >> static_cast<genIType>(BitSize - Shift));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> bitfieldRotateRight(vec<L, T, Q> const& In, int Shift)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_integer, "'bitfieldRotateRight' accepts only integer values");

		int const BitSize = static_cast<int>(sizeof(T) * 8);
		return (In << static_cast<T>(Shift)) | (In >> static_cast<T>(BitSize - Shift));
	}

	template<typename genIType>
	GLM_FUNC_QUALIFIER genIType bitfieldRotateLeft(genIType In, int Shift)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genIType>::is_integer, "'bitfieldRotateLeft' accepts only integer values");

		int const BitSize = static_cast<genIType>(sizeof(genIType) * 8);
		return (In >> static_cast<genIType>(Shift)) | (In << static_cast<genIType>(BitSize - Shift));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> bitfieldRotateLeft(vec<L, T, Q> const& In, int Shift)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<T>::is_integer, "'bitfieldRotateLeft' accepts only integer values");

		int const BitSize = static_cast<int>(sizeof(T) * 8);
		return (In >> static_cast<T>(Shift)) | (In << static_cast<T>(BitSize - Shift));
	}

	template<typename genIUType>
	GLM_FUNC_QUALIFIER genIUType bitfieldFillOne(genIUType Value, int FirstBit, int BitCount)
	{
		return Value | static_cast<genIUType>(mask(BitCount) << FirstBit);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> bitfieldFillOne(vec<L, T, Q> const& Value, int FirstBit, int BitCount)
	{
		return Value | static_cast<T>(mask(BitCount) << FirstBit);
	}

	template<typename genIUType>
	GLM_FUNC_QUALIFIER genIUType bitfieldFillZero(genIUType Value, int FirstBit, int BitCount)
	{
		return Value & static_cast<genIUType>(~(mask(BitCount) << FirstBit));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> bitfieldFillZero(vec<L, T, Q> const& Value, int FirstBit, int BitCount)
	{
		return Value & static_cast<T>(~(mask(BitCount) << FirstBit));
	}

	GLM_FUNC_QUALIFIER int16 bitfieldInterleave(int8 x, int8 y)
	{
		union sign8
		{
			int8 i;
			uint8 u;
		} sign_x, sign_y;

		union sign16
		{
			int16 i;
			uint16 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint16 bitfieldInterleave(uint8 x, uint8 y)
	{
		return detail::bitfieldInterleave<uint8, uint16>(x, y);
	}

	GLM_FUNC_QUALIFIER uint16 bitfieldInterleave(u8vec2 const& v)
	{
		return detail::bitfieldInterleave<uint8, uint16>(v.x, v.y);
	}

	GLM_FUNC_QUALIFIER u8vec2 bitfieldDeinterleave(glm::uint16 x)
	{
		uint16 REG1(x);
		uint16 REG2(x >>= 1);

		REG1 = REG1 & static_cast<uint16>(0x5555);
		REG2 = REG2 & static_cast<uint16>(0x5555);

		REG1 = ((REG1 >> 1) | REG1) & static_cast<uint16>(0x3333);
		REG2 = ((REG2 >> 1) | REG2) & static_cast<uint16>(0x3333);

		REG1 = ((REG1 >> 2) | REG1) & static_cast<uint16>(0x0F0F);
		REG2 = ((REG2 >> 2) | REG2) & static_cast<uint16>(0x0F0F);

		REG1 = ((REG1 >> 4) | REG1) & static_cast<uint16>(0x00FF);
		REG2 = ((REG2 >> 4) | REG2) & static_cast<uint16>(0x00FF);

		REG1 = ((REG1 >> 8) | REG1) & static_cast<uint16>(0xFFFF);
		REG2 = ((REG2 >> 8) | REG2) & static_cast<uint16>(0xFFFF);

		return glm::u8vec2(REG1, REG2);
	}

	GLM_FUNC_QUALIFIER int32 bitfieldInterleave(int16 x, int16 y)
	{
		union sign16
		{
			int16 i;
			uint16 u;
		} sign_x, sign_y;

		union sign32
		{
			int32 i;
			uint32 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint32 bitfieldInterleave(uint16 x, uint16 y)
	{
		return detail::bitfieldInterleave<uint16, uint32>(x, y);
	}

	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave(u16vec2 const& v)
	{
		return detail::bitfieldInterleave<uint16, uint32>(v.x, v.y);
	}

	GLM_FUNC_QUALIFIER glm::u16vec2 bitfieldDeinterleave(glm::uint32 x)
	{
		glm::uint32 REG1(x);
		glm::uint32 REG2(x >>= 1);

		REG1 = REG1 & static_cast<glm::uint32>(0x55555555);
		REG2 = REG2 & static_cast<glm::uint32>(0x55555555);

		REG1 = ((REG1 >> 1) | REG1) & static_cast<glm::uint32>(0x33333333);
		REG2 = ((REG2 >> 1) | REG2) & static_cast<glm::uint32>(0x33333333);

		REG1 = ((REG1 >> 2) | REG1) & static_cast<glm::uint32>(0x0F0F0F0F);
		REG2 = ((REG2 >> 2) | REG2) & static_cast<glm::uint32>(0x0F0F0F0F);

		REG1 = ((REG1 >> 4) | REG1) & static_cast<glm::uint32>(0x00FF00FF);
		REG2 = ((REG2 >> 4) | REG2) & static_cast<glm::uint32>(0x00FF00FF);

		REG1 = ((REG1 >> 8) | REG1) & static_cast<glm::uint32>(0x0000FFFF);
		REG2 = ((REG2 >> 8) | REG2) & static_cast<glm::uint32>(0x0000FFFF);

		return glm::u16vec2(REG1, REG2);
	}

	GLM_FUNC_QUALIFIER int64 bitfieldInterleave(int32 x, int32 y)
	{
		union sign32
		{
			int32 i;
			uint32 u;
		} sign_x, sign_y;

		union sign64
		{
			int64 i;
			uint64 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(uint32 x, uint32 y)
	{
		return detail::bitfieldInterleave<uint32, uint64>(x, y);
	}

	GLM_FUNC_QUALIFIER glm::uint64 bitfieldInterleave(u32vec2 const& v)
	{
		return detail::bitfieldInterleave<uint32, uint64>(v.x, v.y);
	}

	GLM_FUNC_QUALIFIER glm::u32vec2 bitfieldDeinterleave(glm::uint64 x)
	{
		glm::uint64 REG1(x);
		glm::uint64 REG2(x >>= 1);

		REG1 = REG1 & static_cast<glm::uint64>(0x5555555555555555ull);
		REG2 = REG2 & static_cast<glm::uint64>(0x5555555555555555ull);

		REG1 = ((REG1 >> 1) | REG1) & static_cast<glm::uint64>(0x3333333333333333ull);
		REG2 = ((REG2 >> 1) | REG2) & static_cast<glm::uint64>(0x3333333333333333ull);

		REG1 = ((REG1 >> 2) | REG1) & static_cast<glm::uint64>(0x0F0F0F0F0F0F0F0Full);
		REG2 = ((REG2 >> 2) | REG2) & static_cast<glm::uint64>(0x0F0F0F0F0F0F0F0Full);

		REG1 = ((REG1 >> 4) | REG1) & static_cast<glm::uint64>(0x00FF00FF00FF00FFull);
		REG2 = ((REG2 >> 4) | REG2) & static_cast<glm::uint64>(0x00FF00FF00FF00FFull);

		REG1 = ((REG1 >> 8) | REG1) & static_cast<glm::uint64>(0x0000FFFF0000FFFFull);
		REG2 = ((REG2 >> 8) | REG2) & static_cast<glm::uint64>(0x0000FFFF0000FFFFull);

		REG1 = ((REG1 >> 16) | REG1) & static_cast<glm::uint64>(0x00000000FFFFFFFFull);
		REG2 = ((REG2 >> 16) | REG2) & static_cast<glm::uint64>(0x00000000FFFFFFFFull);

		return glm::u32vec2(REG1, REG2);
	}

	GLM_FUNC_QUALIFIER int32 bitfieldInterleave(int8 x, int8 y, int8 z)
	{
		union sign8
		{
			int8 i;
			uint8 u;
		} sign_x, sign_y, sign_z;

		union sign32
		{
			int32 i;
			uint32 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		sign_z.i = z;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u, sign_z.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint32 bitfieldInterleave(uint8 x, uint8 y, uint8 z)
	{
		return detail::bitfieldInterleave<uint8, uint32>(x, y, z);
	}

	GLM_FUNC_QUALIFIER uint32 bitfieldInterleave(u8vec3 const& v)
	{
		return detail::bitfieldInterleave<uint8, uint32>(v.x, v.y, v.z);
	}

	GLM_FUNC_QUALIFIER int64 bitfieldInterleave(int16 x, int16 y, int16 z)
	{
		union sign16
		{
			int16 i;
			uint16 u;
		} sign_x, sign_y, sign_z;

		union sign64
		{
			int64 i;
			uint64 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		sign_z.i = z;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u, sign_z.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(uint16 x, uint16 y, uint16 z)
	{
		return detail::bitfieldInterleave<uint32, uint64>(x, y, z);
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(u16vec3 const& v)
	{
		return detail::bitfieldInterleave<uint32, uint64>(v.x, v.y, v.z);
	}

	GLM_FUNC_QUALIFIER int64 bitfieldInterleave(int32 x, int32 y, int32 z)
	{
		union sign16
		{
			int32 i;
			uint32 u;
		} sign_x, sign_y, sign_z;

		union sign64
		{
			int64 i;
			uint64 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		sign_z.i = z;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u, sign_z.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(uint32 x, uint32 y, uint32 z)
	{
		return detail::bitfieldInterleave<uint32, uint64>(x, y, z);
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(u32vec3 const& v)
	{
		return detail::bitfieldInterleave<uint32, uint64>(v.x, v.y, v.z);
	}

	GLM_FUNC_QUALIFIER int32 bitfieldInterleave(int8 x, int8 y, int8 z, int8 w)
	{
		union sign8
		{
			int8 i;
			uint8 u;
		} sign_x, sign_y, sign_z, sign_w;

		union sign32
		{
			int32 i;
			uint32 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		sign_z.i = z;
		sign_w.i = w;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u, sign_z.u, sign_w.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint32 bitfieldInterleave(uint8 x, uint8 y, uint8 z, uint8 w)
	{
		return detail::bitfieldInterleave<uint8, uint32>(x, y, z, w);
	}

	GLM_FUNC_QUALIFIER uint32 bitfieldInterleave(u8vec4 const& v)
	{
		return detail::bitfieldInterleave<uint8, uint32>(v.x, v.y, v.z, v.w);
	}

	GLM_FUNC_QUALIFIER int64 bitfieldInterleave(int16 x, int16 y, int16 z, int16 w)
	{
		union sign16
		{
			int16 i;
			uint16 u;
		} sign_x, sign_y, sign_z, sign_w;

		union sign64
		{
			int64 i;
			uint64 u;
		} result;

		sign_x.i = x;
		sign_y.i = y;
		sign_z.i = z;
		sign_w.i = w;
		result.u = bitfieldInterleave(sign_x.u, sign_y.u, sign_z.u, sign_w.u);

		return result.i;
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(uint16 x, uint16 y, uint16 z, uint16 w)
	{
		return detail::bitfieldInterleave<uint16, uint64>(x, y, z, w);
	}

	GLM_FUNC_QUALIFIER uint64 bitfieldInterleave(u16vec4 const& v)
	{
		return detail::bitfieldInterleave<uint16, uint64>(v.x, v.y, v.z, v.w);
	}
}//namespace glm
