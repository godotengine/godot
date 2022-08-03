#include <glm/ext/scalar_integer.hpp>
#include <glm/ext/scalar_int_sized.hpp>
#include <glm/ext/scalar_uint_sized.hpp>
#include <vector>
#include <ctime>
#include <cstdio>

#if GLM_LANG & GLM_LANG_CXX11_FLAG
#include <chrono>

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

		int Error = 0;

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

		int Error = 0;

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

		int Error = 0;

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<int>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
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

		int Error = 0;

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<glm::uint>); i < n; ++i)
		{
			bool Result = glm::isPowerOfTwo(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += test_int16();
		Error += test_uint16();
		Error += test_int32();
		Error += test_uint32();

		return Error;
	}
}//isPowerOfTwo

namespace nextPowerOfTwo_advanced
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
	GLM_FUNC_QUALIFIER genType nextPowerOfTwo_loop(genType value)
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
			glm::int32 Result = glm::nextPowerOfTwo(Data[i].Value);
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
			glm::uint32 Result = glm::nextPowerOfTwo(Data[i].Value);
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
			v[i] = nextPowerOfTwo_loop(i);

		std::clock_t Timestramp1 = std::clock();

		for(glm::uint32 i = 0, n = static_cast<glm::uint>(v.size()); i < n; ++i)
			v[i] = glm::nextPowerOfTwo(i);

		std::clock_t Timestramp2 = std::clock();

		std::printf("nextPowerOfTwo_loop: %d clocks\n", static_cast<int>(Timestramp1 - Timestramp0));
		std::printf("glm::nextPowerOfTwo: %d clocks\n", static_cast<int>(Timestramp2 - Timestramp1));

		return Error;
	}

	int test()
	{
		int Error(0);

		Error += test_int32();
		Error += test_uint32();

		return Error;
	}
}//namespace nextPowerOfTwo_advanced

namespace prevPowerOfTwo
{
	template <typename T>
	int run()
	{
		int Error = 0;

		T const A = glm::prevPowerOfTwo(static_cast<T>(7));
		Error += A == static_cast<T>(4) ? 0 : 1;

		T const B = glm::prevPowerOfTwo(static_cast<T>(15));
		Error += B == static_cast<T>(8) ? 0 : 1;

		T const C = glm::prevPowerOfTwo(static_cast<T>(31));
		Error += C == static_cast<T>(16) ? 0 : 1;

		T const D = glm::prevPowerOfTwo(static_cast<T>(32));
		Error += D == static_cast<T>(32) ? 0 : 1;

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<glm::int8>();
		Error += run<glm::int16>();
		Error += run<glm::int32>();
		Error += run<glm::int64>();

		Error += run<glm::uint8>();
		Error += run<glm::uint16>();
		Error += run<glm::uint32>();
		Error += run<glm::uint64>();

		return Error;
	}
}//namespace prevPowerOfTwo

namespace nextPowerOfTwo
{
	template <typename T>
	int run()
	{
		int Error = 0;

		T const A = glm::nextPowerOfTwo(static_cast<T>(7));
		Error += A == static_cast<T>(8) ? 0 : 1;

		T const B = glm::nextPowerOfTwo(static_cast<T>(15));
		Error += B == static_cast<T>(16) ? 0 : 1;

		T const C = glm::nextPowerOfTwo(static_cast<T>(31));
		Error += C == static_cast<T>(32) ? 0 : 1;

		T const D = glm::nextPowerOfTwo(static_cast<T>(32));
		Error += D == static_cast<T>(32) ? 0 : 1;

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<glm::int8>();
		Error += run<glm::int16>();
		Error += run<glm::int32>();
		Error += run<glm::int64>();

		Error += run<glm::uint8>();
		Error += run<glm::uint16>();
		Error += run<glm::uint32>();
		Error += run<glm::uint64>();

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

	template <typename T>
	int run()
	{
		type<T> const Data[] =
		{
			{8, 3, 6},
			{7, 7, 7}
		};

		int Error = 0;
		
		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			T const Result = glm::prevMultiple(Data[i].Source, Data[i].Multiple);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<glm::int8>();
		Error += run<glm::int16>();
		Error += run<glm::int32>();
		Error += run<glm::int64>();

		Error += run<glm::uint8>();
		Error += run<glm::uint16>();
		Error += run<glm::uint32>();
		Error += run<glm::uint64>();

		return Error;
	}
}//namespace prevMultiple

namespace nextMultiple
{
	static glm::uint const Multiples = 128;

	int perf_nextMultiple(glm::uint Samples)
	{
		std::vector<glm::uint> Results(Samples * Multiples);

		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		for(glm::uint Source = 0; Source < Samples; ++Source)
		for(glm::uint Multiple = 0; Multiple < Multiples; ++Multiple)
		{
			Results[Source * Multiples + Multiple] = glm::nextMultiple(Source, Multiples);
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::printf("- glm::nextMultiple Time %d microseconds\n", static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

		glm::uint Result = 0;
		for(std::size_t i = 0, n = Results.size(); i < n; ++i)
			Result += Results[i];

		return Result > 0 ? 0 : 1;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T nextMultipleMod(T Source, T Multiple)
	{
		T const Tmp = Source - static_cast<T>(1);
		return Tmp + (Multiple - (Tmp % Multiple));
	}

	int perf_nextMultipleMod(glm::uint Samples)
	{
		std::vector<glm::uint> Results(Samples * Multiples);

		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		for(glm::uint Multiple = 0; Multiple < Multiples; ++Multiple)
			for (glm::uint Source = 0; Source < Samples; ++Source)
		{
			Results[Source * Multiples + Multiple] = nextMultipleMod(Source, Multiples);
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::printf("- nextMultipleMod Time %d microseconds\n", static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

		glm::uint Result = 0;
		for(std::size_t i = 0, n = Results.size(); i < n; ++i)
			Result += Results[i];

		return Result > 0 ? 0 : 1;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T nextMultipleNeg(T Source, T Multiple)
	{
		if(Source > static_cast<T>(0))
		{
			T const Tmp = Source - static_cast<T>(1);
			return Tmp + (Multiple - (Tmp % Multiple));
		}
		else
			return Source + (-Source % Multiple);
	}

	int perf_nextMultipleNeg(glm::uint Samples)
	{
		std::vector<glm::uint> Results(Samples * Multiples);

		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		for(glm::uint Source = 0; Source < Samples; ++Source)
		for(glm::uint Multiple = 0; Multiple < Multiples; ++Multiple)
		{
			Results[Source * Multiples + Multiple] = nextMultipleNeg(Source, Multiples);
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::printf("- nextMultipleNeg Time %d microseconds\n", static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

		glm::uint Result = 0;
		for (std::size_t i = 0, n = Results.size(); i < n; ++i)
			Result += Results[i];

		return Result > 0 ? 0 : 1;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T nextMultipleUFloat(T Source, T Multiple)
	{
		return Source + (Multiple - std::fmod(Source, Multiple));
	}

	int perf_nextMultipleUFloat(glm::uint Samples)
	{
		std::vector<float> Results(Samples * Multiples);

		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		for(glm::uint Source = 0; Source < Samples; ++Source)
		for(glm::uint Multiple = 0; Multiple < Multiples; ++Multiple)
		{
			Results[Source * Multiples + Multiple] = nextMultipleUFloat(static_cast<float>(Source), static_cast<float>(Multiples));
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::printf("- nextMultipleUFloat Time %d microseconds\n", static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

		float Result = 0;
		for (std::size_t i = 0, n = Results.size(); i < n; ++i)
			Result += Results[i];

		return Result > 0.0f ? 0 : 1;
	}

	template <typename T>
	GLM_FUNC_QUALIFIER T nextMultipleFloat(T Source, T Multiple)
	{
		if(Source > static_cast<float>(0))
			return Source + (Multiple - std::fmod(Source, Multiple));
		else
			return Source + std::fmod(-Source, Multiple);
	}

	int perf_nextMultipleFloat(glm::uint Samples)
	{
		std::vector<float> Results(Samples * Multiples);

		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

		for(glm::uint Source = 0; Source < Samples; ++Source)
		for(glm::uint Multiple = 0; Multiple < Multiples; ++Multiple)
		{
			Results[Source * Multiples + Multiple] = nextMultipleFloat(static_cast<float>(Source), static_cast<float>(Multiples));
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		std::printf("- nextMultipleFloat Time %d microseconds\n", static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));

		float Result = 0;
		for (std::size_t i = 0, n = Results.size(); i < n; ++i)
			Result += Results[i];

		return Result > 0.0f ? 0 : 1;
	}

	template<typename genIUType>
	struct type
	{
		genIUType Source;
		genIUType Multiple;
		genIUType Return;
	};

	template <typename T>
	int test_uint()
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

		for(std::size_t i = 0, n = sizeof(Data) / sizeof(type<T>); i < n; ++i)
		{
			T const Result0 = glm::nextMultiple(Data[i].Source, Data[i].Multiple);
			Error += Data[i].Return == Result0 ? 0 : 1;
			assert(!Error);

			T const Result1 = nextMultipleMod(Data[i].Source, Data[i].Multiple);
			Error += Data[i].Return == Result1 ? 0 : 1;
			assert(!Error);
		}

		return Error;
	}

	int perf()
	{
		int Error = 0;

		glm::uint const Samples = 10000;

		for(int i = 0; i < 4; ++i)
		{
			std::printf("Run %d :\n", i);
			Error += perf_nextMultiple(Samples);
			Error += perf_nextMultipleMod(Samples);
			Error += perf_nextMultipleNeg(Samples);
			Error += perf_nextMultipleUFloat(Samples);
			Error += perf_nextMultipleFloat(Samples);
			std::printf("\n");
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += test_uint<glm::int8>();
		Error += test_uint<glm::int16>();
		Error += test_uint<glm::int32>();
		Error += test_uint<glm::int64>();

		Error += test_uint<glm::uint8>();
		Error += test_uint<glm::uint16>();
		Error += test_uint<glm::uint32>();
		Error += test_uint<glm::uint64>();

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

	template <typename T>
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
			int const Result0 = glm::findNSB(Data[i].Source, Data[i].SignificantBitCount);
			Error += Data[i].Return == Result0 ? 0 : 1;
			assert(!Error);
		}

		return Error;
	}

	int test()
	{
		int Error = 0;

		Error += run<glm::uint8>();
		Error += run<glm::uint16>();
		Error += run<glm::uint32>();
		Error += run<glm::uint64>();

		Error += run<glm::int8>();
		Error += run<glm::int16>();
		Error += run<glm::int32>();
		Error += run<glm::int64>();

		return Error;
	}
}//namespace findNSB

int main()
{
	int Error = 0;

	Error += findNSB::test();

	Error += isPowerOfTwo::test();
	Error += prevPowerOfTwo::test();
	Error += nextPowerOfTwo::test();
	Error += nextPowerOfTwo_advanced::test();
	Error += prevMultiple::test();
	Error += nextMultiple::test();

#	ifdef NDEBUG
		Error += nextPowerOfTwo_advanced::perf();
		Error += nextMultiple::perf();
#	endif//NDEBUG

	return Error;
}

#else

int main()
{
	return 0;
}

#endif
