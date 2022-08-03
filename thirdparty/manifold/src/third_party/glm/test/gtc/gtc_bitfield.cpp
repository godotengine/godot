#include <glm/gtc/bitfield.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/vector_relational.hpp>
#include <glm/integer.hpp>
#include <ctime>
#include <cstdio>
#include <vector>

namespace mask
{
	template<typename genType>
	struct type
	{
		genType		Value;
		genType		Return;
	};

	inline int mask_zero(int Bits)
	{
		return ~((~0) << Bits);
	}

	inline int mask_mix(int Bits)
	{
		return Bits >= sizeof(int) * 8 ? 0xffffffff : (static_cast<int>(1) << Bits) - static_cast<int>(1);
	}

	inline int mask_half(int Bits)
	{
		// We do the shift in two steps because 1 << 32 on an int is undefined.

		int const Half = Bits >> 1;
		int const Fill = ~0;
		int const ShiftHaft = (Fill << Half);
		int const Rest = Bits - Half;
		int const Reversed = ShiftHaft << Rest;

		return ~Reversed;
	}

	inline int mask_loop(int Bits)
	{
		int Mask = 0;
		for(int Bit = 0; Bit < Bits; ++Bit)
			Mask |= (static_cast<int>(1) << Bit);
		return Mask;
	}

	int perf()
	{
		int const Count = 100000000;

		std::clock_t Timestamp1 = std::clock();

		{
			std::vector<int> Mask;
			Mask.resize(Count);
			for(int i = 0; i < Count; ++i)
				Mask[i] = mask_mix(i % 32);
		}

		std::clock_t Timestamp2 = std::clock();

		{
			std::vector<int> Mask;
			Mask.resize(Count);
			for(int i = 0; i < Count; ++i)
				Mask[i] = mask_loop(i % 32);
		}

		std::clock_t Timestamp3 = std::clock();

		{
			std::vector<int> Mask;
			Mask.resize(Count);
			for(int i = 0; i < Count; ++i)
				Mask[i] = glm::mask(i % 32);
		}

		std::clock_t Timestamp4 = std::clock();

		{
			std::vector<int> Mask;
			Mask.resize(Count);
			for(int i = 0; i < Count; ++i)
				Mask[i] = mask_zero(i % 32);
		}

		std::clock_t Timestamp5 = std::clock();

		{
			std::vector<int> Mask;
			Mask.resize(Count);
			for(int i = 0; i < Count; ++i)
				Mask[i] = mask_half(i % 32);
		}

		std::clock_t Timestamp6 = std::clock();

		std::clock_t TimeMix = Timestamp2 - Timestamp1;
		std::clock_t TimeLoop = Timestamp3 - Timestamp2;
		std::clock_t TimeDefault = Timestamp4 - Timestamp3;
		std::clock_t TimeZero = Timestamp5 - Timestamp4;
		std::clock_t TimeHalf = Timestamp6 - Timestamp5;

		std::printf("mask[mix]: %d\n", static_cast<unsigned int>(TimeMix));
		std::printf("mask[loop]: %d\n", static_cast<unsigned int>(TimeLoop));
		std::printf("mask[default]: %d\n", static_cast<unsigned int>(TimeDefault));
		std::printf("mask[zero]: %d\n", static_cast<unsigned int>(TimeZero));
		std::printf("mask[half]: %d\n", static_cast<unsigned int>(TimeHalf));

		return TimeDefault < TimeLoop ? 0 : 1;
	}

	int test_uint()
	{
		type<glm::uint> const Data[] =
		{
			{ 0, 0x00000000},
			{ 1, 0x00000001},
			{ 2, 0x00000003},
			{ 3, 0x00000007},
			{31, 0x7fffffff},
			{32, 0xffffffff}
		};

		int Error = 0;
/* mask_zero is sadly not a correct code
		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<int>); ++i)
		{
			int Result = mask_zero(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}
*/
		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<int>); ++i)
		{
			int Result = mask_mix(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<int>); ++i)
		{
			int Result = mask_half(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<int>); ++i)
		{
			int Result = mask_loop(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<int>); ++i)
		{
			int Result = glm::mask(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test_uvec4()
	{
		type<glm::ivec4> const Data[] =
		{
			{glm::ivec4( 0), glm::ivec4(0x00000000)},
			{glm::ivec4( 1), glm::ivec4(0x00000001)},
			{glm::ivec4( 2), glm::ivec4(0x00000003)},
			{glm::ivec4( 3), glm::ivec4(0x00000007)},
			{glm::ivec4(31), glm::ivec4(0x7fffffff)},
			{glm::ivec4(32), glm::ivec4(0xffffffff)}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::ivec4>); i < n; ++i)
		{
			glm::ivec4 Result = glm::mask(Data[i].Value);
			Error += glm::all(glm::equal(Data[i].Return, Result)) ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_uint();
		Error += test_uvec4();

		return Error;
	}
}//namespace mask

namespace bitfieldInterleave3
{
	template<typename PARAM, typename RET>
	inline RET refBitfieldInterleave(PARAM x, PARAM y, PARAM z)
	{
		RET Result = 0; 
		for(RET i = 0; i < sizeof(PARAM) * 8; ++i)
		{
			Result |= ((RET(x) & (RET(1U) << i)) << ((i << 1) + 0));
			Result |= ((RET(y) & (RET(1U) << i)) << ((i << 1) + 1));
			Result |= ((RET(z) & (RET(1U) << i)) << ((i << 1) + 2));
		}
		return Result;
	}

	int test()
	{
		int Error(0);

		glm::uint16 x_max = 1 << 11;
		glm::uint16 y_max = 1 << 11;
		glm::uint16 z_max = 1 << 11;

		for(glm::uint16 z = 0; z < z_max; z += 27)
		for(glm::uint16 y = 0; y < y_max; y += 27)
		for(glm::uint16 x = 0; x < x_max; x += 27)
		{
			glm::uint64 ResultA = refBitfieldInterleave<glm::uint16, glm::uint64>(x, y, z);
			glm::uint64 ResultB = glm::bitfieldInterleave(x, y, z);
			Error += ResultA == ResultB ? 0 : 1;
		}

		return Error;
	}
}

namespace bitfieldInterleave4
{
	template<typename PARAM, typename RET>
	inline RET loopBitfieldInterleave(PARAM x, PARAM y, PARAM z, PARAM w)
	{
		RET const v[4] = {x, y, z, w};
		RET Result = 0; 
		for(RET i = 0; i < sizeof(PARAM) * 8; i++)
		{
			Result |= ((((v[0] >> i) & 1U)) << ((i << 2) + 0));
			Result |= ((((v[1] >> i) & 1U)) << ((i << 2) + 1));
			Result |= ((((v[2] >> i) & 1U)) << ((i << 2) + 2));
			Result |= ((((v[3] >> i) & 1U)) << ((i << 2) + 3));
		}
		return Result;
	}

	int test()
	{
		int Error(0);

		glm::uint16 x_max = 1 << 11;
		glm::uint16 y_max = 1 << 11;
		glm::uint16 z_max = 1 << 11;
		glm::uint16 w_max = 1 << 11;

		for(glm::uint16 w = 0; w < w_max; w += 27)
		for(glm::uint16 z = 0; z < z_max; z += 27)
		for(glm::uint16 y = 0; y < y_max; y += 27)
		for(glm::uint16 x = 0; x < x_max; x += 27)
		{
			glm::uint64 ResultA = loopBitfieldInterleave<glm::uint16, glm::uint64>(x, y, z, w);
			glm::uint64 ResultB = glm::bitfieldInterleave(x, y, z, w);
			Error += ResultA == ResultB ? 0 : 1;
		}

		return Error;
	}
}

namespace bitfieldInterleave
{
	inline glm::uint64 fastBitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		glm::uint64 REG1;
		glm::uint64 REG2;

		REG1 = x;
		REG1 = ((REG1 << 16) | REG1) & glm::uint64(0x0000FFFF0000FFFF);
		REG1 = ((REG1 <<  8) | REG1) & glm::uint64(0x00FF00FF00FF00FF);
		REG1 = ((REG1 <<  4) | REG1) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		REG1 = ((REG1 <<  2) | REG1) & glm::uint64(0x3333333333333333);
		REG1 = ((REG1 <<  1) | REG1) & glm::uint64(0x5555555555555555);

		REG2 = y;
		REG2 = ((REG2 << 16) | REG2) & glm::uint64(0x0000FFFF0000FFFF);
		REG2 = ((REG2 <<  8) | REG2) & glm::uint64(0x00FF00FF00FF00FF);
		REG2 = ((REG2 <<  4) | REG2) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		REG2 = ((REG2 <<  2) | REG2) & glm::uint64(0x3333333333333333);
		REG2 = ((REG2 <<  1) | REG2) & glm::uint64(0x5555555555555555);

		return REG1 | (REG2 << 1);
	}

	inline glm::uint64 interleaveBitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		glm::uint64 REG1;
		glm::uint64 REG2;

		REG1 = x;
		REG2 = y;

		REG1 = ((REG1 << 16) | REG1) & glm::uint64(0x0000FFFF0000FFFF);
		REG2 = ((REG2 << 16) | REG2) & glm::uint64(0x0000FFFF0000FFFF);

		REG1 = ((REG1 <<  8) | REG1) & glm::uint64(0x00FF00FF00FF00FF);
		REG2 = ((REG2 <<  8) | REG2) & glm::uint64(0x00FF00FF00FF00FF);

		REG1 = ((REG1 <<  4) | REG1) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		REG2 = ((REG2 <<  4) | REG2) & glm::uint64(0x0F0F0F0F0F0F0F0F);

		REG1 = ((REG1 <<  2) | REG1) & glm::uint64(0x3333333333333333);
		REG2 = ((REG2 <<  2) | REG2) & glm::uint64(0x3333333333333333);

		REG1 = ((REG1 <<  1) | REG1) & glm::uint64(0x5555555555555555);
		REG2 = ((REG2 <<  1) | REG2) & glm::uint64(0x5555555555555555);

		return REG1 | (REG2 << 1);
	}
/*
	inline glm::uint64 loopBitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		static glm::uint64 const Mask[5] = 
		{
			0x5555555555555555,
			0x3333333333333333,
			0x0F0F0F0F0F0F0F0F,
			0x00FF00FF00FF00FF,
			0x0000FFFF0000FFFF
		};

		glm::uint64 REG1 = x;
		glm::uint64 REG2 = y;
		for(int i = 4; i >= 0; --i)
		{
			REG1 = ((REG1 << (1 << i)) | REG1) & Mask[i];
			REG2 = ((REG2 << (1 << i)) | REG2) & Mask[i];
		}

		return REG1 | (REG2 << 1);
	}
*/
#if GLM_ARCH & GLM_ARCH_SSE2_BIT
	inline glm::uint64 sseBitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		__m128i const Array = _mm_set_epi32(0, y, 0, x);

		__m128i const Mask4 = _mm_set1_epi32(0x0000FFFF);
		__m128i const Mask3 = _mm_set1_epi32(0x00FF00FF);
		__m128i const Mask2 = _mm_set1_epi32(0x0F0F0F0F);
		__m128i const Mask1 = _mm_set1_epi32(0x33333333);
		__m128i const Mask0 = _mm_set1_epi32(0x55555555);

		__m128i Reg1;
		__m128i Reg2;

		// REG1 = x;
		// REG2 = y;
		Reg1 = _mm_load_si128(&Array);

		//REG1 = ((REG1 << 16) | REG1) & glm::uint64(0x0000FFFF0000FFFF);
		//REG2 = ((REG2 << 16) | REG2) & glm::uint64(0x0000FFFF0000FFFF);
		Reg2 = _mm_slli_si128(Reg1, 2);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask4);

		//REG1 = ((REG1 <<  8) | REG1) & glm::uint64(0x00FF00FF00FF00FF);
		//REG2 = ((REG2 <<  8) | REG2) & glm::uint64(0x00FF00FF00FF00FF);
		Reg2 = _mm_slli_si128(Reg1, 1);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask3);

		//REG1 = ((REG1 <<  4) | REG1) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		//REG2 = ((REG2 <<  4) | REG2) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		Reg2 = _mm_slli_epi32(Reg1, 4);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask2);

		//REG1 = ((REG1 <<  2) | REG1) & glm::uint64(0x3333333333333333);
		//REG2 = ((REG2 <<  2) | REG2) & glm::uint64(0x3333333333333333);
		Reg2 = _mm_slli_epi32(Reg1, 2);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask1);

		//REG1 = ((REG1 <<  1) | REG1) & glm::uint64(0x5555555555555555);
		//REG2 = ((REG2 <<  1) | REG2) & glm::uint64(0x5555555555555555);
		Reg2 = _mm_slli_epi32(Reg1, 1);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask0);

		//return REG1 | (REG2 << 1);
		Reg2 = _mm_slli_epi32(Reg1, 1);
		Reg2 = _mm_srli_si128(Reg2, 8);
		Reg1 = _mm_or_si128(Reg1, Reg2);
	
		__m128i Result;
		_mm_store_si128(&Result, Reg1);
		return *reinterpret_cast<glm::uint64*>(&Result);
	}

	inline glm::uint64 sseUnalignedBitfieldInterleave(glm::uint32 x, glm::uint32 y)
	{
		__m128i const Array = _mm_set_epi32(0, y, 0, x);

		__m128i const Mask4 = _mm_set1_epi32(0x0000FFFF);
		__m128i const Mask3 = _mm_set1_epi32(0x00FF00FF);
		__m128i const Mask2 = _mm_set1_epi32(0x0F0F0F0F);
		__m128i const Mask1 = _mm_set1_epi32(0x33333333);
		__m128i const Mask0 = _mm_set1_epi32(0x55555555);

		__m128i Reg1;
		__m128i Reg2;

		// REG1 = x;
		// REG2 = y;
		Reg1 = _mm_loadu_si128(&Array);

		//REG1 = ((REG1 << 16) | REG1) & glm::uint64(0x0000FFFF0000FFFF);
		//REG2 = ((REG2 << 16) | REG2) & glm::uint64(0x0000FFFF0000FFFF);
		Reg2 = _mm_slli_si128(Reg1, 2);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask4);

		//REG1 = ((REG1 <<  8) | REG1) & glm::uint64(0x00FF00FF00FF00FF);
		//REG2 = ((REG2 <<  8) | REG2) & glm::uint64(0x00FF00FF00FF00FF);
		Reg2 = _mm_slli_si128(Reg1, 1);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask3);

		//REG1 = ((REG1 <<  4) | REG1) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		//REG2 = ((REG2 <<  4) | REG2) & glm::uint64(0x0F0F0F0F0F0F0F0F);
		Reg2 = _mm_slli_epi32(Reg1, 4);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask2);

		//REG1 = ((REG1 <<  2) | REG1) & glm::uint64(0x3333333333333333);
		//REG2 = ((REG2 <<  2) | REG2) & glm::uint64(0x3333333333333333);
		Reg2 = _mm_slli_epi32(Reg1, 2);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask1);

		//REG1 = ((REG1 <<  1) | REG1) & glm::uint64(0x5555555555555555);
		//REG2 = ((REG2 <<  1) | REG2) & glm::uint64(0x5555555555555555);
		Reg2 = _mm_slli_epi32(Reg1, 1);
		Reg1 = _mm_or_si128(Reg2, Reg1);
		Reg1 = _mm_and_si128(Reg1, Mask0);

		//return REG1 | (REG2 << 1);
		Reg2 = _mm_slli_epi32(Reg1, 1);
		Reg2 = _mm_srli_si128(Reg2, 8);
		Reg1 = _mm_or_si128(Reg1, Reg2);

		__m128i Result;
		_mm_store_si128(&Result, Reg1);
		return *reinterpret_cast<glm::uint64*>(&Result);
	}
#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

	int test()
	{
		int Error = 0;

/*
		{
			for(glm::uint32 y = 0; y < (1 << 10); ++y)
			for(glm::uint32 x = 0; x < (1 << 10); ++x)
			{
				glm::uint64 A = glm::bitfieldInterleave(x, y);
				glm::uint64 B = fastBitfieldInterleave(x, y);
				//glm::uint64 C = loopBitfieldInterleave(x, y);
				glm::uint64 D = interleaveBitfieldInterleave(x, y);

				assert(A == B);
				//assert(A == C);
				assert(A == D);

#				if GLM_ARCH & GLM_ARCH_SSE2_BIT
					glm::uint64 E = sseBitfieldInterleave(x, y);
					glm::uint64 F = sseUnalignedBitfieldInterleave(x, y);
					assert(A == E);
					assert(A == F);

					__m128i G = glm_i128_interleave(_mm_set_epi32(0, y, 0, x));
					glm::uint64 Result[2];
					_mm_storeu_si128((__m128i*)Result, G);
					assert(A == Result[0]);
#				endif//GLM_ARCH & GLM_ARCH_SSE2_BIT
			}
		}
*/
		{
			for(glm::uint8 y = 0; y < 127; ++y)
			for(glm::uint8 x = 0; x < 127; ++x)
			{
				glm::uint64 A(glm::bitfieldInterleave(glm::u8vec2(x, y)));
				glm::uint64 B(glm::bitfieldInterleave(glm::u16vec2(x, y)));
				glm::uint64 C(glm::bitfieldInterleave(glm::u32vec2(x, y)));

				Error += A == B ? 0 : 1;
				Error += A == C ? 0 : 1;

				glm::u32vec2 const& D = glm::bitfieldDeinterleave(C);
				Error += D.x == x ? 0 : 1;
				Error += D.y == y ? 0 : 1;
			}
		}

		{
			for(glm::uint8 y = 0; y < 127; ++y)
			for(glm::uint8 x = 0; x < 127; ++x)
			{
				glm::int64 A(glm::bitfieldInterleave(glm::int8(x), glm::int8(y)));
				glm::int64 B(glm::bitfieldInterleave(glm::int16(x), glm::int16(y)));
				glm::int64 C(glm::bitfieldInterleave(glm::int32(x), glm::int32(y)));

				Error += A == B ? 0 : 1;
				Error += A == C ? 0 : 1;
			}
		}

		return Error;
	}

	int perf()
	{
		glm::uint32 x_max = 1 << 11;
		glm::uint32 y_max = 1 << 10;

		// ALU
		std::vector<glm::uint64> Data(x_max * y_max);
		std::vector<glm::u32vec2> Param(x_max * y_max);
		for(glm::uint32 i = 0; i < Param.size(); ++i)
			Param[i] = glm::u32vec2(i % x_max, i / y_max);

		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = glm::bitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("glm::bitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}

		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = fastBitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("fastBitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}
/*
		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = loopBitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("loopBitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}
*/
		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = interleaveBitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("interleaveBitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}

#		if GLM_ARCH & GLM_ARCH_SSE2_BIT
		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = sseBitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("sseBitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}

		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = sseUnalignedBitfieldInterleave(Param[i].x, Param[i].y);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("sseUnalignedBitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}
#		endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

		{
			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < Data.size(); ++i)
				Data[i] = glm::bitfieldInterleave(Param[i].x, Param[i].y, Param[i].x);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("glm::detail::bitfieldInterleave Time %d clocks\n", static_cast<int>(Time));
		}

#		if(GLM_ARCH & GLM_ARCH_SSE2_BIT && !(GLM_COMPILER & GLM_COMPILER_GCC))
		{
			// SIMD
			std::vector<__m128i> SimdData;
			SimdData.resize(static_cast<std::size_t>(x_max * y_max));
			std::vector<__m128i> SimdParam;
			SimdParam.resize(static_cast<std::size_t>(x_max * y_max));
			for(std::size_t i = 0; i < SimdParam.size(); ++i)
				SimdParam[i] = _mm_set_epi32(static_cast<int>(i % static_cast<std::size_t>(x_max)), 0, static_cast<int>(i / static_cast<std::size_t>(y_max)), 0);

			std::clock_t LastTime = std::clock();

			for(std::size_t i = 0; i < SimdData.size(); ++i)
				SimdData[i] = glm_i128_interleave(SimdParam[i]);

			std::clock_t Time = std::clock() - LastTime;

			std::printf("_mm_bit_interleave_si128 Time %d clocks\n", static_cast<int>(Time));
		}
#		endif//GLM_ARCH & GLM_ARCH_SSE2_BIT

		return 0;
	}
}//namespace bitfieldInterleave

namespace bitfieldInterleave5
{
	GLM_FUNC_QUALIFIER glm::uint16 bitfieldInterleave_u8vec2(glm::uint8 x, glm::uint8 y)
	{
		glm::uint32 Result = (glm::uint32(y) << 16) | glm::uint32(x);
		Result = ((Result <<  4) | Result) & 0x0F0F0F0F;
		Result = ((Result <<  2) | Result) & 0x33333333;
		Result = ((Result <<  1) | Result) & 0x55555555;
		return static_cast<glm::uint16>((Result & 0x0000FFFF) | (Result >> 15));
	}

	GLM_FUNC_QUALIFIER glm::u8vec2 bitfieldDeinterleave_u8vec2(glm::uint16 InterleavedBitfield)
	{
		glm::uint32 Result(InterleavedBitfield);
		Result = ((Result << 15) | Result) & 0x55555555;
		Result = ((Result >>  1) | Result) & 0x33333333;
		Result = ((Result >>  2) | Result) & 0x0F0F0F0F;
		Result = ((Result >>  4) | Result) & 0x00FF00FF;
		return glm::u8vec2(Result & 0x0000FFFF, Result >> 16);
	}

	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave_u8vec4(glm::uint8 x, glm::uint8 y, glm::uint8 z, glm::uint8 w)
	{
		glm::uint64 Result = (glm::uint64(w) << 48) | (glm::uint64(z) << 32) | (glm::uint64(y) << 16) | glm::uint64(x);
		Result = ((Result << 12) | Result) & 0x000F000F000F000Full;
		Result = ((Result <<  6) | Result) & 0x0303030303030303ull;
		Result = ((Result <<  3) | Result) & 0x1111111111111111ull;

		const glm::uint32 a = static_cast<glm::uint32>((Result & 0x000000000000FFFF) >> ( 0 - 0));
		const glm::uint32 b = static_cast<glm::uint32>((Result & 0x00000000FFFF0000) >> (16 - 3));
		const glm::uint32 c = static_cast<glm::uint32>((Result & 0x0000FFFF00000000) >> (32 - 6));
		const glm::uint32 d = static_cast<glm::uint32>((Result & 0xFFFF000000000000) >> (48 - 12));

		return a | b | c | d;
	}

	GLM_FUNC_QUALIFIER glm::u8vec4 bitfieldDeinterleave_u8vec4(glm::uint32 InterleavedBitfield)
	{
		glm::uint64 Result(InterleavedBitfield);
		Result = ((Result << 15) | Result) & 0x9249249249249249ull;
		Result = ((Result >>  1) | Result) & 0x30C30C30C30C30C3ull;
		Result = ((Result >>  2) | Result) & 0xF00F00F00F00F00Full;
		Result = ((Result >>  4) | Result) & 0x00FF0000FF0000FFull;
		return glm::u8vec4(
			(Result >> 0) & 0x000000000000FFFFull,
			(Result >> 16) & 0x00000000FFFF0000ull,
			(Result >> 32) & 0x0000FFFF00000000ull,
			(Result >> 48) & 0xFFFF000000000000ull);
	}

	GLM_FUNC_QUALIFIER glm::uint32 bitfieldInterleave_u16vec2(glm::uint16 x, glm::uint16 y)
	{
		glm::uint64 Result = (glm::uint64(y) << 32) | glm::uint64(x);
		Result = ((Result <<  8) | Result) & static_cast<glm::uint32>(0x00FF00FF00FF00FFull);
		Result = ((Result <<  4) | Result) & static_cast<glm::uint32>(0x0F0F0F0F0F0F0F0Full);
		Result = ((Result <<  2) | Result) & static_cast<glm::uint32>(0x3333333333333333ull);
		Result = ((Result <<  1) | Result) & static_cast<glm::uint32>(0x5555555555555555ull);
		return static_cast<glm::uint32>((Result & 0x00000000FFFFFFFFull) | (Result >> 31));
	}

	GLM_FUNC_QUALIFIER glm::u16vec2 bitfieldDeinterleave_u16vec2(glm::uint32 InterleavedBitfield)
	{
		glm::uint64 Result(InterleavedBitfield);
		Result = ((Result << 31) | Result) & 0x5555555555555555ull;
		Result = ((Result >>  1) | Result) & 0x3333333333333333ull;
		Result = ((Result >>  2) | Result) & 0x0F0F0F0F0F0F0F0Full;
		Result = ((Result >>  4) | Result) & 0x00FF00FF00FF00FFull;
		Result = ((Result >>  8) | Result) & 0x0000FFFF0000FFFFull;
		return glm::u16vec2(Result & 0x00000000FFFFFFFFull, Result >> 32);
	}

	int test()
	{
		int Error = 0;

		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
		{
			glm::uint16 A = bitfieldInterleave_u8vec2(glm::uint8(i), glm::uint8(j));
			glm::uint16 B = glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j));
			Error += A == B ? 0 : 1;

			glm::u8vec2 C = bitfieldDeinterleave_u8vec2(A);
			Error += C.x == glm::uint8(i) ? 0 : 1;
			Error += C.y == glm::uint8(j) ? 0 : 1;
		}

		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
		{
			glm::uint32 A = bitfieldInterleave_u8vec4(glm::uint8(i), glm::uint8(j), glm::uint8(i), glm::uint8(j));
			glm::uint32 B = glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j), glm::uint8(i), glm::uint8(j));
			Error += A == B ? 0 : 1;
/*
			glm::u8vec4 C = bitfieldDeinterleave_u8vec4(A);
			Error += C.x == glm::uint8(i) ? 0 : 1;
			Error += C.y == glm::uint8(j) ? 0 : 1;
			Error += C.z == glm::uint8(i) ? 0 : 1;
			Error += C.w == glm::uint8(j) ? 0 : 1;
*/
		}

		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
		{
			glm::uint32 A = bitfieldInterleave_u16vec2(glm::uint16(i), glm::uint16(j));
			glm::uint32 B = glm::bitfieldInterleave(glm::uint16(i), glm::uint16(j));
			Error += A == B ? 0 : 1;
		}

		return Error;
	}

	int perf_old_u8vec2(std::vector<glm::uint16>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();
		
		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("glm::bitfieldInterleave<u8vec2> Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf_new_u8vec2(std::vector<glm::uint16>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();

		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == bitfieldInterleave_u8vec2(glm::uint8(i), glm::uint8(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("bitfieldInterleave_u8vec2 Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf_old_u8vec4(std::vector<glm::uint32>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();

		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j), glm::uint8(i), glm::uint8(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("glm::bitfieldInterleave<u8vec4> Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf_new_u8vec4(std::vector<glm::uint32>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();

		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == bitfieldInterleave_u8vec4(glm::uint8(i), glm::uint8(j), glm::uint8(i), glm::uint8(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("bitfieldInterleave_u8vec4 Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf_old_u16vec2(std::vector<glm::uint32>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();

		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == glm::bitfieldInterleave(glm::uint16(i), glm::uint16(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("glm::bitfieldInterleave<u16vec2> Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf_new_u16vec2(std::vector<glm::uint32>& Result)
	{
		int Error = 0;

		const std::clock_t BeginTime = std::clock();

		for(glm::size_t k = 0; k < 10000; ++k)
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Error += Result[j * 256 + i] == bitfieldInterleave_u16vec2(glm::uint16(i), glm::uint16(j)) ? 0 : 1;

		const std::clock_t EndTime = std::clock();

		std::printf("bitfieldInterleave_u16vec2 Time %d clocks\n", static_cast<int>(EndTime - BeginTime));

		return Error;
	}

	int perf()
	{
		int Error = 0;

		std::printf("bitfieldInterleave perf: init\r");

		std::vector<glm::uint16> Result_u8vec2(256 * 256, 0);
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Result_u8vec2[j * 256 + i] = glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j));

		Error += perf_old_u8vec2(Result_u8vec2);
		Error += perf_new_u8vec2(Result_u8vec2);

		std::vector<glm::uint32> Result_u8vec4(256 * 256, 0);
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Result_u8vec4[j * 256 + i] = glm::bitfieldInterleave(glm::uint8(i), glm::uint8(j), glm::uint8(i), glm::uint8(j));
		
		Error += perf_old_u8vec4(Result_u8vec4);
		Error += perf_new_u8vec4(Result_u8vec4);

		std::vector<glm::uint32> Result_u16vec2(256 * 256, 0);
		for(glm::size_t j = 0; j < 256; ++j)
		for(glm::size_t i = 0; i < 256; ++i)
			Result_u16vec2[j * 256 + i] = glm::bitfieldInterleave(glm::uint16(i), glm::uint16(j));

		Error += perf_old_u16vec2(Result_u16vec2);
		Error += perf_new_u16vec2(Result_u16vec2);

		std::printf("bitfieldInterleave perf: %d Errors\n", Error);

		return Error;
	}

}//namespace bitfieldInterleave5

static int test_bitfieldRotateRight()
{
	glm::ivec4 const A = glm::bitfieldRotateRight(glm::ivec4(2), 1);
	glm::ivec4 const B = glm::ivec4(2) >> 1;

	return A == B;
}

static int test_bitfieldRotateLeft()
{
	glm::ivec4 const A = glm::bitfieldRotateLeft(glm::ivec4(2), 1);
	glm::ivec4 const B = glm::ivec4(2) << 1;

	return A == B;
}

int main()
{
	int Error = 0;

/* Tests for a faster and to reserve bitfieldInterleave
	Error += ::bitfieldInterleave5::test();
	Error += ::bitfieldInterleave5::perf();
*/
	Error += ::mask::test();
	Error += ::bitfieldInterleave3::test();
	Error += ::bitfieldInterleave4::test();
	Error += ::bitfieldInterleave::test();

	Error += test_bitfieldRotateRight();
	Error += test_bitfieldRotateLeft();

#	ifdef NDEBUG
		Error += ::mask::perf();
		Error += ::bitfieldInterleave::perf();
#	endif//NDEBUG

	return Error;
}
