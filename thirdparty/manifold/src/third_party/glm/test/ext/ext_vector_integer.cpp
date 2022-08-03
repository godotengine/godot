#include <glm/ext/vector_integer.hpp>
#include <glm/ext/scalar_int_sized.hpp>
#include <glm/ext/scalar_uint_sized.hpp>
#include <vector>
#include <ctime>
#include <cstdio>

namespace isPowerOfTwo
{
	template<typename genType>
	struct type
	{
		genType		Value;
		bool		Return;
	};

	template <glm::length_t L>
	int test_int16()
	{
		type<glm::int16> const Data[] =
		{
			{ 0x0001, true },
			{ 0x0002, true },
			{ 0x0004, true },
			{ 0x0080, true },
			{ 0x0000, true },
			{ 0x0003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::int16>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::int16>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	int test_uint16()
	{
		type<glm::uint16> const Data[] =
		{
			{ 0x0001, true },
			{ 0x0002, true },
			{ 0x0004, true },
			{ 0x0000, true },
			{ 0x0000, true },
			{ 0x0003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint16>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::uint16>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	int test_int32()
	{
		type<int> const Data[] =
		{
			{ 0x00000001, true },
			{ 0x00000002, true },
			{ 0x00000004, true },
			{ 0x0000000f, false },
			{ 0x00000000, true },
			{ 0x00000003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::int32>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	template <glm::length_t L>
	int test_uint32()
	{
		type<glm::uint> const Data[] =
		{
			{ 0x00000001, true },
			{ 0x00000002, true },
			{ 0x00000004, true },
			{ 0x80000000, true },
			{ 0x00000000, true },
			{ 0x00000003, false }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint>); i < n; ++i)
		{
			glm::vec<L, bool> const Result = glm::isPowerOfTwo(glm::vec<L, glm::uint32>(Data[i].Value));
			Error += glm::vec<L, bool>(Data[i].Return) == Result ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += test_int16<1>();
		Error += test_int16<2>();
		Error += test_int16<3>();
		Error += test_int16<4>();

		Error += test_uint16<1>();
		Error += test_uint16<2>();
		Error += test_uint16<3>();
		Error += test_uint16<4>();

		Error += test_int32<1>();
		Error += test_int32<2>();
		Error += test_int32<3>();
		Error += test_int32<4>();

		Error += test_uint32<1>();
		Error += test_uint32<2>();
		Error += test_uint32<3>();
		Error += test_uint32<4>();

		return Error;
	}
}//isPowerOfTwo

namespace prevPowerOfTwo
{
	template <glm::length_t L, typename T>
	int run()
	{
		int Error = 0;

		glm::vec<L, T> const A = glm::prevPowerOfTwo(glm::vec<L, T>(7));
		Error += A == glm::vec<L, T>(4) ? 0 : 1;

		glm::vec<L, T> const B = glm::prevPowerOfTwo(glm::vec<L, T>(15));
		Error += B == glm::vec<L, T>(8) ? 0 : 1;

		glm::vec<L, T> const C = glm::prevPowerOfTwo(glm::vec<L, T>(31));
		Error += C == glm::vec<L, T>(16) ? 0 : 1;

		glm::vec<L, T> const D = glm::prevPowerOfTwo(glm::vec<L, T>(32));
		Error += D == glm::vec<L, T>(32) ? 0 : 1;

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace prevPowerOfTwo

namespace nextPowerOfTwo
{
	template <glm::length_t L, typename T>
	int run()
	{
		int Error = 0;

		glm::vec<L, T> const A = glm::nextPowerOfTwo(glm::vec<L, T>(7));
		Error += A == glm::vec<L, T>(8) ? 0 : 1;

		glm::vec<L, T> const B = glm::nextPowerOfTwo(glm::vec<L, T>(15));
		Error += B == glm::vec<L, T>(16) ? 0 : 1;

		glm::vec<L, T> const C = glm::nextPowerOfTwo(glm::vec<L, T>(31));
		Error += C == glm::vec<L, T>(32) ? 0 : 1;

		glm::vec<L, T> const D = glm::nextPowerOfTwo(glm::vec<L, T>(32));
		Error += D == glm::vec<L, T>(32) ? 0 : 1;

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace nextPowerOfTwo

namespace prevMultiple
{
	template<typename genIUType>
	struct type
	{
		genIUType Source;
		genIUType Multiple;
		genIUType Return;
	};

	template <glm::length_t L, typename T>
	int run()
	{
		type<T> const Data[] =
		{
			{ 8, 3, 6 },
			{ 7, 7, 7 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, T> const Result0 = glm::prevMultiple(glm::vec<L, T>(Data[i].Source), Data[i].Multiple);
			Error += glm::vec<L, T>(Data[i].Return) == Result0 ? 0 : 1;

			glm::vec<L, T> const Result1 = glm::prevMultiple(glm::vec<L, T>(Data[i].Source), glm::vec<L, T>(Data[i].Multiple));
			Error += glm::vec<L, T>(Data[i].Return) == Result1 ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace prevMultiple

namespace nextMultiple
{
	template<typename genIUType>
	struct type
	{
		genIUType Source;
		genIUType Multiple;
		genIUType Return;
	};

	template <glm::length_t L, typename T>
	int run()
	{
		type<T> const Data[] =
		{
			{ 3, 4, 4 },
			{ 6, 3, 6 },
			{ 5, 3, 6 },
			{ 7, 7, 7 },
			{ 0, 1, 0 },
			{ 8, 3, 9 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, T> const Result0 = glm::nextMultiple(glm::vec<L, T>(Data[i].Source), glm::vec<L, T>(Data[i].Multiple));
			Error += glm::vec<L, T>(Data[i].Return) == Result0 ? 0 : 1;

			glm::vec<L, T> const Result1 = glm::nextMultiple(glm::vec<L, T>(Data[i].Source), Data[i].Multiple);
			Error += glm::vec<L, T>(Data[i].Return) == Result1 ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		return Error;
	}
}//namespace nextMultiple

namespace findNSB
{
	template<typename T>
	struct type
	{
		T Source;
		int SignificantBitCount;
		int Return;
	};

	template <glm::length_t L, typename T>
	int run()
	{
		type<T> const Data[] =
		{
			{ 0x00, 1,-1 },
			{ 0x01, 2,-1 },
			{ 0x02, 2,-1 },
			{ 0x06, 3,-1 },
			{ 0x01, 1, 0 },
			{ 0x03, 1, 0 },
			{ 0x03, 2, 1 },
			{ 0x07, 2, 1 },
			{ 0x05, 2, 2 },
			{ 0x0D, 2, 2 }
		};

		int Error = 0;

		for (std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			glm::vec<L, int> const Result0 = glm::findNSB<L, T, glm::defaultp>(glm::vec<L, T>(Data[i].Source), glm::vec<L, int>(Data[i].SignificantBitCount));
			Error += glm::vec<L, int>(Data[i].Return) == Result0 ? 0 : 1;
			assert(!Error);
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<1, glm::uint8>();
		Error += run<2, glm::uint8>();
		Error += run<3, glm::uint8>();
		Error += run<4, glm::uint8>();

		Error += run<1, glm::uint16>();
		Error += run<2, glm::uint16>();
		Error += run<3, glm::uint16>();
		Error += run<4, glm::uint16>();

		Error += run<1, glm::uint32>();
		Error += run<2, glm::uint32>();
		Error += run<3, glm::uint32>();
		Error += run<4, glm::uint32>();

		Error += run<1, glm::uint64>();
		Error += run<2, glm::uint64>();
		Error += run<3, glm::uint64>();
		Error += run<4, glm::uint64>();

		Error += run<1, glm::int8>();
		Error += run<2, glm::int8>();
		Error += run<3, glm::int8>();
		Error += run<4, glm::int8>();

		Error += run<1, glm::int16>();
		Error += run<2, glm::int16>();
		Error += run<3, glm::int16>();
		Error += run<4, glm::int16>();

		Error += run<1, glm::int32>();
		Error += run<2, glm::int32>();
		Error += run<3, glm::int32>();
		Error += run<4, glm::int32>();

		Error += run<1, glm::int64>();
		Error += run<2, glm::int64>();
		Error += run<3, glm::int64>();
		Error += run<4, glm::int64>();


		return Error;
	}
}//namespace findNSB

int main()
{
	int Error = 0;

	Error += isPowerOfTwo::test();
	Error += prevPowerOfTwo::test();
	Error += nextPowerOfTwo::test();
	Error += prevMultiple::test();
	Error += nextMultiple::test();
	Error += findNSB::test();

	return Error;
}
