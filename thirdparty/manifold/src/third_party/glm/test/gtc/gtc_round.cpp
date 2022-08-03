#include <glm/gtc/round.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/gtc/epsilon.hpp>
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

	int test_int16()
	{
		type<glm::int16> const Data[] =
		{
			{0x0001, true},
			{0x0002, true},
			{0x0004, true},
			{0x0080, true},
			{0x0000, true},
			{0x0003, false}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::int16>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test_uint16()
	{
		type<glm::uint16> const Data[] =
		{
			{0x0001, true},
			{0x0002, true},
			{0x0004, true},
			{0x0000, true},
			{0x0000, true},
			{0x0003, false}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint16>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test_int32()
	{
		type<int> const Data[] =
		{
			{0x00000001, true},
			{0x00000002, true},
			{0x00000004, true},
			{0x0000000f, false},
			{0x00000000, true},
			{0x00000003, false}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::bvec1 Result = glm::isPowerOfTwo(glm::ivec1(Data[i].Value));
			Error += glm::all(glm::equal(glm::bvec1(Data[i].Return), Result)) ? 0 : 1;
		}

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::bvec2 Result = glm::isPowerOfTwo(glm::ivec2(Data[i].Value));
			Error += glm::all(glm::equal(glm::bvec2(Data[i].Return), Result)) ? 0 : 1;
		}

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::bvec3 Result = glm::isPowerOfTwo(glm::ivec3(Data[i].Value));
			Error += glm::all(glm::equal(glm::bvec3(Data[i].Return), Result)) ? 0 : 1;
		}

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			glm::bvec4 Result = glm::isPowerOfTwo(glm::ivec4(Data[i].Value));
			Error += glm::all(glm::equal(glm::bvec4(Data[i].Return), Result)) ? 0 : 1;
		}

		return Error;
	}

	int test_uint32()
	{
		type<glm::uint> const Data[] =
		{
			{0x00000001, true},
			{0x00000002, true},
			{0x00000004, true},
			{0x80000000, true},
			{0x00000000, true},
			{0x00000003, false}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_int16();
		Error += test_uint16();
		Error += test_int32();
		Error += test_uint32();

		return Error;
	}
}//isPowerOfTwo

namespace ceilPowerOfTwo_advanced
{
	template<typename genIUType>
	GLM_FUNC_QUALIFIER genIUType highestBitValue(genIUType Value)
	{
		genIUType tmp = Value;
		genIUType result = genIUType(0);
		while(tmp)
		{
			result = (tmp & (~tmp + 1)); // grab lowest bit
			tmp &= ~result; // clear lowest bit
		}
		return result;
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER genType ceilPowerOfTwo_loop(genType value)
	{
		return glm::isPowerOfTwo(value) ? value : highestBitValue(value) << 1;
	}

	template<typename genType>
	struct type
	{
		genType		Value;
		genType		Return;
	};

	int test_int32()
	{
		type<glm::int32> const Data[] =
		{
			{0x0000ffff, 0x00010000},
			{-3, -4},
			{-8, -8},
			{0x00000001, 0x00000001},
			{0x00000002, 0x00000002},
			{0x00000004, 0x00000004},
			{0x00000007, 0x00000008},
			{0x0000fff0, 0x00010000},
			{0x0000f000, 0x00010000},
			{0x08000000, 0x08000000},
			{0x00000000, 0x00000000},
			{0x00000003, 0x00000004}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::int32>); i < n; ++i)
		{
			glm::int32 Result = glm::ceilPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test_uint32()
	{
		type<glm::uint32> const Data[] =
		{
			{0x00000001, 0x00000001},
			{0x00000002, 0x00000002},
			{0x00000004, 0x00000004},
			{0x00000007, 0x00000008},
			{0x0000ffff, 0x00010000},
			{0x0000fff0, 0x00010000},
			{0x0000f000, 0x00010000},
			{0x80000000, 0x80000000},
			{0x00000000, 0x00000000},
			{0x00000003, 0x00000004}
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint32>); i < n; ++i)
		{
			glm::uint32 Result = glm::ceilPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int perf()
	{
		int Error(0);

		std::vector<glm::uint> v;
		v.resize(100000000);

		std::clock_t Timestramp0 = std::clock();

		for(glm::uint32 i = 0, n = static_cast<glm::uint>(v.size()); i < n; ++i)
			v[i] = ceilPowerOfTwo_loop(i);

		std::clock_t Timestramp1 = std::clock();

		for(glm::uint32 i = 0, n = static_cast<glm::uint>(v.size()); i < n; ++i)
			v[i] = glm::ceilPowerOfTwo(i);

		std::clock_t Timestramp2 = std::clock();

		std::printf("ceilPowerOfTwo_loop: %d clocks\n", static_cast<int>(Timestramp1 - Timestramp0));
		std::printf("glm::ceilPowerOfTwo: %d clocks\n", static_cast<int>(Timestramp2 - Timestramp1));

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_int32();
		Error += test_uint32();

		return Error;
	}
}//namespace ceilPowerOfTwo_advanced

namespace roundPowerOfTwo
{
	int test()
	{
		int Error = 0;
		
		glm::uint32 const A = glm::roundPowerOfTwo(7u);
		Error += A == 8u ? 0 : 1;
		
		glm::uint32 const B = glm::roundPowerOfTwo(15u);
		Error += B == 16u ? 0 : 1;

		glm::uint32 const C = glm::roundPowerOfTwo(31u);
		Error += C == 32u ? 0 : 1;
		
		glm::uint32 const D = glm::roundPowerOfTwo(9u);
		Error += D == 8u ? 0 : 1;
		
		glm::uint32 const E = glm::roundPowerOfTwo(17u);
		Error += E == 16u ? 0 : 1;
		
		glm::uint32 const F = glm::roundPowerOfTwo(33u);
		Error += F == 32u ? 0 : 1;
		
		return Error;
	}
}//namespace roundPowerOfTwo

namespace floorPowerOfTwo
{
	int test()
	{
		int Error = 0;
		
		glm::uint32 const A = glm::floorPowerOfTwo(7u);
		Error += A == 4u ? 0 : 1;
		
		glm::uint32 const B = glm::floorPowerOfTwo(15u);
		Error += B == 8u ? 0 : 1;
		
		glm::uint32 const C = glm::floorPowerOfTwo(31u);
		Error += C == 16u ? 0 : 1;
		
		return Error;
	}
}//namespace floorPowerOfTwo

namespace ceilPowerOfTwo
{
	int test()
	{
		int Error = 0;
		
		glm::uint32 const A = glm::ceilPowerOfTwo(7u);
		Error += A == 8u ? 0 : 1;
		
		glm::uint32 const B = glm::ceilPowerOfTwo(15u);
		Error += B == 16u ? 0 : 1;
		
		glm::uint32 const C = glm::ceilPowerOfTwo(31u);
		Error += C == 32u ? 0 : 1;
		
		return Error;
	}
}//namespace ceilPowerOfTwo

namespace floorMultiple
{
	template<typename genType>
	struct type
	{
		genType		Source;
		genType		Multiple;
		genType		Return;
		genType		Epsilon;
	};

	int test_float()
	{
		type<glm::float64> const Data[] = 
		{
			{3.4, 0.3, 3.3, 0.0001},
			{-1.4, 0.3, -1.5, 0.0001},
		};

		int Error(0);
		
		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::float64>); i < n; ++i)
		{
			glm::float64 Result = glm::floorMultiple(Data[i].Source, Data[i].Multiple);
			Error += glm::epsilonEqual(Data[i].Return, Result, Data[i].Epsilon) ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_float();

		return Error;
	}
}//namespace floorMultiple

namespace ceilMultiple
{
	template<typename genType>
	struct type
	{
		genType		Source;
		genType		Multiple;
		genType		Return;
		genType		Epsilon;
	};

	int test_float()
	{
		type<glm::float64> const Data[] = 
		{
			{3.4, 0.3, 3.6, 0.0001},
			{-1.4, 0.3, -1.2, 0.0001},
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::float64>); i < n; ++i)
		{
			glm::float64 Result = glm::ceilMultiple(Data[i].Source, Data[i].Multiple);
			Error += glm::epsilonEqual(Data[i].Return, Result, Data[i].Epsilon) ? 0 : 1;
		}

		return Error;
	}

	int test_int()
	{
		type<int> const Data[] = 
		{
			{3, 4, 4, 0},
			{7, 4, 8, 0},
			{5, 4, 8, 0},
			{1, 4, 4, 0},
			{1, 3, 3, 0},
			{4, 3, 6, 0},
			{4, 1, 4, 0},
			{1, 1, 1, 0},
			{7, 1, 7, 0},
		};

		int Error(0);

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			int Result = glm::ceilMultiple(Data[i].Source, Data[i].Multiple);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_int();
		Error += test_float();

		return Error;
	}
}//namespace ceilMultiple

int main()
{
	int Error(0);

	Error += isPowerOfTwo::test();
	Error += floorPowerOfTwo::test();
	Error += roundPowerOfTwo::test();
	Error += ceilPowerOfTwo::test();
	Error += ceilPowerOfTwo_advanced::test();
	
#	ifdef NDEBUG
		Error += ceilPowerOfTwo_advanced::perf();
#	endif//NDEBUG

	Error += floorMultiple::test();
	Error += ceilMultiple::test();

	return Error;
}
