#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/gtc/constants.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/ext/vector_relational.hpp>
#include <glm/ext/vector_float1.hpp>
#include <glm/common.hpp>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <vector>
#include <cstdio>
#include <cmath>
#include <ctime>

// This file has divisions by zero to test isnan
#if GLM_COMPILER & GLM_COMPILER_VC
#	pragma warning(disable : 4723)
#endif

namespace floor_
{
	static int test()
	{
		int Error = 0;

		{
			float A = 1.1f;
			float B = glm::floor(A);
			Error += glm::equal(B, 1.f, 0.0001f) ? 0 : 1;
		}

		{
			double A = 1.1;
			double B = glm::floor(A);
			Error += glm::equal(B, 1.0, 0.0001) ? 0 : 1;
		}

		{
			glm::vec1 A(1.1f);
			glm::vec1 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::vec1(1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec1 A(1.1);
			glm::dvec1 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::dvec1(1.0), 0.0001)) ? 0 : 1;
		}

		{
			glm::vec2 A(1.1f);
			glm::vec2 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::vec2(1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec2 A(1.1);
			glm::dvec2 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::dvec2(1.0), 0.0001)) ? 0 : 1;
		}

		{
			glm::vec3 A(1.1f);
			glm::vec3 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::vec3(1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec3 A(1.1);
			glm::dvec3 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::dvec3(1.0), 0.0001)) ? 0 : 1;
		}

		{
			glm::vec4 A(1.1f);
			glm::vec4 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::vec4(1.0), 0.0001f)) ? 0 : 1;
		}

		{
			glm::dvec4 A(1.1);
			glm::dvec4 B = glm::floor(A);

			Error += glm::all(glm::equal(B, glm::dvec4(1.0), 0.0001)) ? 0 : 1;
		}

		return Error;
	}
}//namespace floor

namespace modf_
{
	static int test()
	{
		int Error(0);

		{
			float X(1.5f);
			float I(0.0f);
			float A = glm::modf(X, I);

			Error += glm::equal(I, 1.0f, 0.0001f) ? 0 : 1;
			Error += glm::equal(A, 0.5f, 0.0001f) ? 0 : 1;
		}

		{
			glm::vec4 X(1.1f, 1.2f, 1.5f, 1.7f);
			glm::vec4 I(0.0f);
			glm::vec4 A = glm::modf(X, I);

			Error += glm::ivec4(I) == glm::ivec4(1) ? 0 : 1;
			Error += glm::all(glm::equal(A, glm::vec4(0.1f, 0.2f, 0.5f, 0.7f), 0.00001f)) ? 0 : 1;
		}

		{
			glm::dvec4 X(1.1, 1.2, 1.5, 1.7);
			glm::dvec4 I(0.0);
			glm::dvec4 A = glm::modf(X, I);

			Error += glm::ivec4(I) == glm::ivec4(1) ? 0 : 1;
			Error += glm::all(glm::equal(A, glm::dvec4(0.1, 0.2, 0.5, 0.7), 0.000000001)) ? 0 : 1;
		}

		{
			double X(1.5);
			double I(0.0);
			double A = glm::modf(X, I);

			Error += glm::equal(I, 1.0, 0.0001) ? 0 : 1;
			Error += glm::equal(A, 0.5, 0.0001) ? 0 : 1;
		}

		return Error;
	}
}//namespace modf

namespace mod_
{
	static int test()
	{
		int Error(0);

		{
			float A(1.5f);
			float B(1.0f);
			float C = glm::mod(A, B);

			Error += glm::equal(C, 0.5f, 0.00001f) ? 0 : 1;
		}

		{
			float A(-0.2f);
			float B(1.0f);
			float C = glm::mod(A, B);

			Error += glm::equal(C, 0.8f, 0.00001f) ? 0 : 1;
		}

		{
			float A(3.0);
			float B(2.0f);
			float C = glm::mod(A, B);

			Error += glm::equal(C, 1.0f, 0.00001f) ? 0 : 1;
		}

		{
			glm::vec4 A(3.0);
			float B(2.0f);
			glm::vec4 C = glm::mod(A, B);

			Error += glm::all(glm::equal(C, glm::vec4(1.0f), 0.00001f)) ? 0 : 1;
		}

		{
			glm::vec4 A(3.0);
			glm::vec4 B(2.0f);
			glm::vec4 C = glm::mod(A, B);

			Error += glm::all(glm::equal(C, glm::vec4(1.0f), 0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace mod_

namespace floatBitsToInt
{
	static int test()
	{
		int Error = 0;
	
		{
			float A = 1.0f;
			int B = glm::floatBitsToInt(A);
			float C = glm::intBitsToFloat(B);
			Error += glm::equal(A, C, 0.0001f) ? 0 : 1;
		}

		{
			glm::vec2 A(1.0f, 2.0f);
			glm::ivec2 B = glm::floatBitsToInt(A);
			glm::vec2 C = glm::intBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}

		{
			glm::vec3 A(1.0f, 2.0f, 3.0f);
			glm::ivec3 B = glm::floatBitsToInt(A);
			glm::vec3 C = glm::intBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}
	
		{
			glm::vec4 A(1.0f, 2.0f, 3.0f, 4.0f);
			glm::ivec4 B = glm::floatBitsToInt(A);
			glm::vec4 C = glm::intBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}
	
		return Error;
	}
}//namespace floatBitsToInt

namespace floatBitsToUint
{
	static int test()
	{
		int Error = 0;
	
		{
			float A = 1.0f;
			glm::uint B = glm::floatBitsToUint(A);
			float C = glm::uintBitsToFloat(B);
			Error += glm::equal(A, C, 0.0001f) ? 0 : 1;
		}
	
		{
			glm::vec2 A(1.0f, 2.0f);
			glm::uvec2 B = glm::floatBitsToUint(A);
			glm::vec2 C = glm::uintBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}
	
		{
			glm::vec3 A(1.0f, 2.0f, 3.0f);
			glm::uvec3 B = glm::floatBitsToUint(A);
			glm::vec3 C = glm::uintBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}
	
		{
			glm::vec4 A(1.0f, 2.0f, 3.0f, 4.0f);
			glm::uvec4 B = glm::floatBitsToUint(A);
			glm::vec4 C = glm::uintBitsToFloat(B);
			Error += glm::all(glm::equal(A, C, 0.0001f)) ? 0 : 1;
		}
	
		return Error;
	}
}//namespace floatBitsToUint

namespace min_
{
	static int test()
	{
		int Error = 0;

		glm::vec1 A0 = glm::min(glm::vec1(1), glm::vec1(1));
		bool A1 = glm::all(glm::equal(A0, glm::vec1(1), glm::epsilon<float>()));
		Error += A1 ? 0 : 1;

		glm::vec2 B0 = glm::min(glm::vec2(1), glm::vec2(1));
		glm::vec2 B1 = glm::min(glm::vec2(1), 1.0f);
		bool B2 = glm::all(glm::equal(B0, B1, glm::epsilon<float>()));
		Error += B2 ? 0 : 1;

		glm::vec3 C0 = glm::min(glm::vec3(1), glm::vec3(1));
		glm::vec3 C1 = glm::min(glm::vec3(1), 1.0f);
		bool C2 = glm::all(glm::equal(C0, C1, glm::epsilon<float>()));
		Error += C2 ? 0 : 1;

		glm::vec4 D0 = glm::min(glm::vec4(1), glm::vec4(1));
		glm::vec4 D1 = glm::min(glm::vec4(1), 1.0f);
		bool D2 = glm::all(glm::equal(D0, D1, glm::epsilon<float>()));
		Error += D2 ? 0 : 1;

		return Error;
	}

	int min_tern(int a, int b)
	{
		return a < b ? a : b;
	}

	int min_int(int x, int y)
	{
		return y ^ ((x ^ y) & -(x < y)); 
	}

	static int perf(std::size_t Count)
	{
		std::vector<int> A(Count);
		std::vector<int> B(Count);

		std::size_t const InternalCount = 200000;

		for(std::size_t i = 0; i < Count; ++i)
		{
			A[i] = glm::linearRand(-1000, 1000);
			B[i] = glm::linearRand(-1000, 1000);
		}

		int Error = 0;

		glm::int32 SumA = 0;
		{
			std::clock_t Timestamp0 = std::clock();

			for (std::size_t j = 0; j < InternalCount; ++j)
			for (std::size_t i = 0; i < Count; ++i)
				SumA += min_tern(A[i], B[i]);

			std::clock_t Timestamp1 = std::clock();

			std::printf("min_tern Time %d clocks\n", static_cast<int>(Timestamp1 - Timestamp0));
		}

		glm::int32 SumB = 0;
		{
			std::clock_t Timestamp0 = std::clock();

			for (std::size_t j = 0; j < InternalCount; ++j)
			for (std::size_t i = 0; i < Count; ++i)
				SumB += min_int(A[i], B[i]);

			std::clock_t Timestamp1 = std::clock();

			std::printf("min_int Time %d clocks\n", static_cast<int>(Timestamp1 - Timestamp0));
		}

		Error += SumA == SumB ? 0 : 1;

		return Error;
	}
}//namespace min_

namespace max_
{
	static int test()
	{
		int Error = 0;

		glm::vec1 A0 = glm::max(glm::vec1(1), glm::vec1(1));
		bool A1 = glm::all(glm::equal(A0, glm::vec1(1), glm::epsilon<float>()));
		Error += A1 ? 0 : 1;


		glm::vec2 B0 = glm::max(glm::vec2(1), glm::vec2(1));
		glm::vec2 B1 = glm::max(glm::vec2(1), 1.0f);
		bool B2 = glm::all(glm::equal(B0, B1, glm::epsilon<float>()));
		Error += B2 ? 0 : 1;

		glm::vec3 C0 = glm::max(glm::vec3(1), glm::vec3(1));
		glm::vec3 C1 = glm::max(glm::vec3(1), 1.0f);
		bool C2 = glm::all(glm::equal(C0, C1, glm::epsilon<float>()));
		Error += C2 ? 0 : 1;

		glm::vec4 D0 = glm::max(glm::vec4(1), glm::vec4(1));
		glm::vec4 D1 = glm::max(glm::vec4(1), 1.0f);
		bool D2 = glm::all(glm::equal(D0, D1, glm::epsilon<float>()));
		Error += D2 ? 0 : 1;

		return Error;
	}
}//namespace max_

namespace clamp_
{
	static int test()
	{
		int Error = 0;

		return Error;
	}
}//namespace clamp_

namespace mix_
{
	template<typename T, typename B>
	struct entry
	{
		T x;
		T y;
		B a;
		T Result;
	};

	entry<float, bool> const TestBool[] =
	{
		{0.0f, 1.0f, false, 0.0f},
		{0.0f, 1.0f, true, 1.0f},
		{-1.0f, 1.0f, false, -1.0f},
		{-1.0f, 1.0f, true, 1.0f}
	};

	entry<float, float> const TestFloat[] =
	{
		{0.0f, 1.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 1.0f, 1.0f},
		{-1.0f, 1.0f, 0.0f, -1.0f},
		{-1.0f, 1.0f, 1.0f, 1.0f}
	};

	entry<glm::vec2, bool> const TestVec2Bool[] =
	{
		{glm::vec2(0.0f), glm::vec2(1.0f), false, glm::vec2(0.0f)},
		{glm::vec2(0.0f), glm::vec2(1.0f), true, glm::vec2(1.0f)},
		{glm::vec2(-1.0f), glm::vec2(1.0f), false, glm::vec2(-1.0f)},
		{glm::vec2(-1.0f), glm::vec2(1.0f), true, glm::vec2(1.0f)}
	};

	entry<glm::vec2, glm::bvec2> const TestBVec2[] =
	{
		{glm::vec2(0.0f), glm::vec2(1.0f), glm::bvec2(false), glm::vec2(0.0f)},
		{glm::vec2(0.0f), glm::vec2(1.0f), glm::bvec2(true), glm::vec2(1.0f)},
		{glm::vec2(-1.0f), glm::vec2(1.0f), glm::bvec2(false), glm::vec2(-1.0f)},
		{glm::vec2(-1.0f), glm::vec2(1.0f), glm::bvec2(true), glm::vec2(1.0f)},
		{glm::vec2(-1.0f), glm::vec2(1.0f), glm::bvec2(true, false), glm::vec2(1.0f, -1.0f)}
	};

	entry<glm::vec3, bool> const TestVec3Bool[] =
	{
		{glm::vec3(0.0f), glm::vec3(1.0f), false, glm::vec3(0.0f)},
		{glm::vec3(0.0f), glm::vec3(1.0f), true, glm::vec3(1.0f)},
		{glm::vec3(-1.0f), glm::vec3(1.0f), false, glm::vec3(-1.0f)},
		{glm::vec3(-1.0f), glm::vec3(1.0f), true, glm::vec3(1.0f)}
	};

	entry<glm::vec3, glm::bvec3> const TestBVec3[] =
	{
		{glm::vec3(0.0f), glm::vec3(1.0f), glm::bvec3(false), glm::vec3(0.0f)},
		{glm::vec3(0.0f), glm::vec3(1.0f), glm::bvec3(true), glm::vec3(1.0f)},
		{glm::vec3(-1.0f), glm::vec3(1.0f), glm::bvec3(false), glm::vec3(-1.0f)},
		{glm::vec3(-1.0f), glm::vec3(1.0f), glm::bvec3(true), glm::vec3(1.0f)},
		{glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(4.0f, 5.0f, 6.0f), glm::bvec3(true, false, true), glm::vec3(4.0f, 2.0f, 6.0f)}
	};

	entry<glm::vec4, bool> const TestVec4Bool[] = 
	{
		{glm::vec4(0.0f), glm::vec4(1.0f), false, glm::vec4(0.0f)},
		{glm::vec4(0.0f), glm::vec4(1.0f), true, glm::vec4(1.0f)},
		{glm::vec4(-1.0f), glm::vec4(1.0f), false, glm::vec4(-1.0f)},
		{glm::vec4(-1.0f), glm::vec4(1.0f), true, glm::vec4(1.0f)}
	};

	entry<glm::vec4, glm::bvec4> const TestBVec4[] = 
	{
		{glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(2.0f, 2.0f, 3.0f, 3.0f), glm::bvec4(false, true, false, true), glm::vec4(0.0f, 2.0f, 1.0f, 3.0f)},
		{glm::vec4(0.0f), glm::vec4(1.0f), glm::bvec4(true), glm::vec4(1.0f)},
		{glm::vec4(-1.0f), glm::vec4(1.0f), glm::bvec4(false), glm::vec4(-1.0f)},
		{glm::vec4(-1.0f), glm::vec4(1.0f), glm::bvec4(true), glm::vec4(1.0f)},
		{glm::vec4(1.0f, 2.0f, 3.0f, 4.0f), glm::vec4(5.0f, 6.0f, 7.0f, 8.0f), glm::bvec4(true, false, true, false), glm::vec4(5.0f, 2.0f, 7.0f, 4.0f)}
	};

	static int test()
	{
		int Error = 0;

		// Float with bool
		{
			for(std::size_t i = 0; i < sizeof(TestBool) / sizeof(entry<float, bool>); ++i)
			{
				float Result = glm::mix(TestBool[i].x, TestBool[i].y, TestBool[i].a);
				Error += glm::equal(Result, TestBool[i].Result, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// Float with float
		{
			for(std::size_t i = 0; i < sizeof(TestFloat) / sizeof(entry<float, float>); ++i)
			{
				float Result = glm::mix(TestFloat[i].x, TestFloat[i].y, TestFloat[i].a);
				Error += glm::equal(Result, TestFloat[i].Result, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec2 with bool
		{
			for(std::size_t i = 0; i < sizeof(TestVec2Bool) / sizeof(entry<glm::vec2, bool>); ++i)
			{
				glm::vec2 Result = glm::mix(TestVec2Bool[i].x, TestVec2Bool[i].y, TestVec2Bool[i].a);
				Error += glm::equal(Result.x, TestVec2Bool[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestVec2Bool[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec2 with bvec2
		{
			for(std::size_t i = 0; i < sizeof(TestBVec2) / sizeof(entry<glm::vec2, glm::bvec2>); ++i)
			{
				glm::vec2 Result = glm::mix(TestBVec2[i].x, TestBVec2[i].y, TestBVec2[i].a);
				Error += glm::equal(Result.x, TestBVec2[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestBVec2[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec3 with bool
		{
			for(std::size_t i = 0; i < sizeof(TestVec3Bool) / sizeof(entry<glm::vec3, bool>); ++i)
			{
				glm::vec3 Result = glm::mix(TestVec3Bool[i].x, TestVec3Bool[i].y, TestVec3Bool[i].a);
				Error += glm::equal(Result.x, TestVec3Bool[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestVec3Bool[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.z, TestVec3Bool[i].Result.z, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec3 with bvec3
		{
			for(std::size_t i = 0; i < sizeof(TestBVec3) / sizeof(entry<glm::vec3, glm::bvec3>); ++i)
			{
				glm::vec3 Result = glm::mix(TestBVec3[i].x, TestBVec3[i].y, TestBVec3[i].a);
				Error += glm::equal(Result.x, TestBVec3[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestBVec3[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.z, TestBVec3[i].Result.z, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec4 with bool
		{
			for(std::size_t i = 0; i < sizeof(TestVec4Bool) / sizeof(entry<glm::vec4, bool>); ++i)
			{
				glm::vec4 Result = glm::mix(TestVec4Bool[i].x, TestVec4Bool[i].y, TestVec4Bool[i].a);
				Error += glm::equal(Result.x, TestVec4Bool[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestVec4Bool[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.z, TestVec4Bool[i].Result.z, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.w, TestVec4Bool[i].Result.w, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		// vec4 with bvec4
		{
			for(std::size_t i = 0; i < sizeof(TestBVec4) / sizeof(entry<glm::vec4, glm::bvec4>); ++i)
			{
				glm::vec4 Result = glm::mix(TestBVec4[i].x, TestBVec4[i].y, TestBVec4[i].a);
				Error += glm::equal(Result.x, TestBVec4[i].Result.x, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.y, TestBVec4[i].Result.y, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.z, TestBVec4[i].Result.z, glm::epsilon<float>()) ? 0 : 1;
				Error += glm::equal(Result.w, TestBVec4[i].Result.w, glm::epsilon<float>()) ? 0 : 1;
			}
		}

		return Error;
	}
}//namespace mix_

namespace step_
{
	template<typename EDGE, typename VEC>
	struct entry
	{
		EDGE edge;
		VEC x;
		VEC result;
	};

	entry<float, glm::vec4> TestVec4Scalar [] =
	{
		{ 1.0f, glm::vec4(1.0f, 2.0f, 3.0f, 4.0f), glm::vec4(1.0f) },
		{ 0.0f, glm::vec4(1.0f, 2.0f, 3.0f, 4.0f), glm::vec4(1.0f) },
		{ 0.0f, glm::vec4(-1.0f, -2.0f, -3.0f, -4.0f), glm::vec4(0.0f) }
	};

	entry<glm::vec4, glm::vec4> TestVec4Vector [] =
	{
		{ glm::vec4(-1.0f, -2.0f, -3.0f, -4.0f), glm::vec4(-2.0f, -3.0f, -4.0f, -5.0f), glm::vec4(0.0f) },
		{ glm::vec4( 0.0f, 1.0f, 2.0f, 3.0f), glm::vec4( 1.0f, 2.0f, 3.0f, 4.0f), glm::vec4(1.0f) },
		{ glm::vec4( 2.0f, 3.0f, 4.0f, 5.0f), glm::vec4( 1.0f, 2.0f, 3.0f, 4.0f), glm::vec4(0.0f) },
		{ glm::vec4( 0.0f, 1.0f, 2.0f, 3.0f), glm::vec4(-1.0f,-2.0f,-3.0f,-4.0f), glm::vec4(0.0f) }
	};

	static int test()
	{
		int Error = 0;

		// scalar
		{
			float const Edge = 2.0f;

			float const A = glm::step(Edge, 1.0f);
			Error += glm::equal(A, 0.0f, glm::epsilon<float>()) ? 0 : 1;

			float const B = glm::step(Edge, 3.0f);
			Error += glm::equal(B, 1.0f, glm::epsilon<float>()) ? 0 : 1;

			float const C = glm::step(Edge, 2.0f);
			Error += glm::equal(C, 1.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		// vec4 and float
		{
			for (std::size_t i = 0; i < sizeof(TestVec4Scalar) / sizeof(entry<float, glm::vec4>); ++i)
			{
				glm::vec4 Result = glm::step(TestVec4Scalar[i].edge, TestVec4Scalar[i].x);
				Error += glm::all(glm::equal(Result, TestVec4Scalar[i].result, glm::epsilon<float>())) ? 0 : 1;
			}
		}

		// vec4 and vec4
		{
			for (std::size_t i = 0; i < sizeof(TestVec4Vector) / sizeof(entry<glm::vec4, glm::vec4>); ++i)
			{
				glm::vec4 Result = glm::step(TestVec4Vector[i].edge, TestVec4Vector[i].x);
				Error += glm::all(glm::equal(Result, TestVec4Vector[i].result, glm::epsilon<float>())) ? 0 : 1;
			}
		}

		return Error;
	}
}//namespace step_

namespace round_
{
	static int test()
	{
		int Error = 0;

		{
			float A = glm::round(0.0f);
			Error += glm::equal(A, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::round(0.5f);
			Error += glm::equal(B, 1.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::round(1.0f);
			Error += glm::equal(C, 1.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::round(0.1f);
			Error += glm::equal(D, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::round(0.9f);
			Error += glm::equal(E, 1.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::round(1.5f);
			Error += glm::equal(F, 2.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::round(1.9f);
			Error += glm::equal(G, 2.0f, glm::epsilon<float>()) ? 0 : 1;
		}
	
		{
			float A = glm::round(-0.0f);
			Error += glm::equal(A, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::round(-0.5f);
			Error += glm::equal(B, -1.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::round(-1.0f);
			Error += glm::equal(C, -1.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::round(-0.1f);
			Error += glm::equal(D, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::round(-0.9f);
			Error += glm::equal(E, -1.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::round(-1.5f);
			Error += glm::equal(F, -2.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::round(-1.9f);
			Error += glm::equal(G, -2.0f, glm::epsilon<float>()) ? 0 : 1;
		}
	
		return Error;
	}
}//namespace round_

namespace roundEven
{
	static int test()
	{
		int Error = 0;

		{
			float A1 = glm::roundEven(-1.5f);
			Error += glm::equal(A1, -2.0f, 0.0001f) ? 0 : 1;

			float A2 = glm::roundEven(1.5f);
			Error += glm::equal(A2, 2.0f, 0.0001f) ? 0 : 1;

			float A5 = glm::roundEven(-2.5f);
			Error += glm::equal(A5, -2.0f, 0.0001f) ? 0 : 1;

			float A6 = glm::roundEven(2.5f);
			Error += glm::equal(A6, 2.0f, 0.0001f) ? 0 : 1;

			float A3 = glm::roundEven(-3.5f);
			Error += glm::equal(A3, -4.0f, 0.0001f) ? 0 : 1;

			float A4 = glm::roundEven(3.5f);
			Error += glm::equal(A4, 4.0f, 0.0001f) ? 0 : 1;

			float C7 = glm::roundEven(-4.5f);
			Error += glm::equal(C7, -4.0f, 0.0001f) ? 0 : 1;

			float C8 = glm::roundEven(4.5f);
			Error += glm::equal(C8, 4.0f, 0.0001f) ? 0 : 1;

			float C1 = glm::roundEven(-5.5f);
			Error += glm::equal(C1, -6.0f, 0.0001f) ? 0 : 1;

			float C2 = glm::roundEven(5.5f);
			Error += glm::equal(C2, 6.0f, 0.0001f) ? 0 : 1;

			float C3 = glm::roundEven(-6.5f);
			Error += glm::equal(C3, -6.0f, 0.0001f) ? 0 : 1;

			float C4 = glm::roundEven(6.5f);
			Error += glm::equal(C4, 6.0f, 0.0001f) ? 0 : 1;

			float C5 = glm::roundEven(-7.5f);
			Error += glm::equal(C5, -8.0f, 0.0001f) ? 0 : 1;

			float C6 = glm::roundEven(7.5f);
			Error += glm::equal(C6, 8.0f, 0.0001f) ? 0 : 1;

			Error += 0;
		}

		{
			float A7 = glm::roundEven(-2.4f);
			Error += glm::equal(A7, -2.0f, 0.0001f) ? 0 : 1;

			float A8 = glm::roundEven(2.4f);
			Error += glm::equal(A8, 2.0f, 0.0001f) ? 0 : 1;

			float B1 = glm::roundEven(-2.6f);
			Error += glm::equal(B1, -3.0f, 0.0001f) ? 0 : 1;

			float B2 = glm::roundEven(2.6f);
			Error += glm::equal(B2, 3.0f, 0.0001f) ? 0 : 1;

			float B3 = glm::roundEven(-2.0f);
			Error += glm::equal(B3, -2.0f, 0.0001f) ? 0 : 1;

			float B4 = glm::roundEven(2.0f);
			Error += glm::equal(B4, 2.0f, 0.0001f) ? 0 : 1;

			Error += 0;
		}

		{
			float A = glm::roundEven(0.0f);
			Error += glm::equal(A, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::roundEven(0.5f);
			Error += glm::equal(B, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::roundEven(1.0f);
			Error += glm::equal(C, 1.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::roundEven(0.1f);
			Error += glm::equal(D, 0.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::roundEven(0.9f);
			Error += glm::equal(E, 1.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::roundEven(1.5f);
			Error += glm::equal(F, 2.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::roundEven(1.9f);
			Error += glm::equal(G, 2.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		{
			float A = glm::roundEven(-0.0f);
			Error += glm::equal(A,  0.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::roundEven(-0.5f);
			Error += glm::equal(B, -0.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::roundEven(-1.0f);
			Error += glm::equal(C, -1.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::roundEven(-0.1f);
			Error += glm::equal(D,  0.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::roundEven(-0.9f);
			Error += glm::equal(E, -1.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::roundEven(-1.5f);
			Error += glm::equal(F, -2.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::roundEven(-1.9f);
			Error += glm::equal(G, -2.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		{
			float A = glm::roundEven(1.5f);
			Error += glm::equal(A, 2.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::roundEven(2.5f);
			Error += glm::equal(B, 2.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::roundEven(3.5f);
			Error += glm::equal(C, 4.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::roundEven(4.5f);
			Error += glm::equal(D, 4.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::roundEven(5.5f);
			Error += glm::equal(E, 6.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::roundEven(6.5f);
			Error += glm::equal(F, 6.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::roundEven(7.5f);
			Error += glm::equal(G, 8.0f, glm::epsilon<float>()) ? 0 : 1;
		}
	
		{
			float A = glm::roundEven(-1.5f);
			Error += glm::equal(A, -2.0f, glm::epsilon<float>()) ? 0 : 1;
			float B = glm::roundEven(-2.5f);
			Error += glm::equal(B, -2.0f, glm::epsilon<float>()) ? 0 : 1;
			float C = glm::roundEven(-3.5f);
			Error += glm::equal(C, -4.0f, glm::epsilon<float>()) ? 0 : 1;
			float D = glm::roundEven(-4.5f);
			Error += glm::equal(D, -4.0f, glm::epsilon<float>()) ? 0 : 1;
			float E = glm::roundEven(-5.5f);
			Error += glm::equal(E, -6.0f, glm::epsilon<float>()) ? 0 : 1;
			float F = glm::roundEven(-6.5f);
			Error += glm::equal(F, -6.0f, glm::epsilon<float>()) ? 0 : 1;
			float G = glm::roundEven(-7.5f);
			Error += glm::equal(G, -8.0f, glm::epsilon<float>()) ? 0 : 1;
		}

		return Error;
	}
}//namespace roundEven

namespace isnan_
{
	static int test()
	{
		int Error = 0;

		float Zero_f = 0.0;
		double Zero_d = 0.0;

		{
			Error += true == glm::isnan(0.0/Zero_d) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::dvec2(0.0 / Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::dvec3(0.0 / Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::dvec4(0.0 / Zero_d))) ? 0 : 1;
		}

		{
			Error += true == glm::isnan(0.0f/Zero_f) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::vec2(0.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::vec3(0.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isnan(glm::vec4(0.0f/Zero_f))) ? 0 : 1;
		}

		return Error;
	}
}//namespace isnan_

namespace isinf_
{
	static int test()
	{
		int Error = 0;

		float Zero_f = 0.0;
		double Zero_d = 0.0;

		{
			Error += true == glm::isinf( 1.0/Zero_d) ? 0 : 1;
			Error += true == glm::isinf(-1.0/Zero_d) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec2( 1.0/Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec2(-1.0/Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec3( 1.0/Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec3(-1.0/Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec4( 1.0/Zero_d))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::dvec4(-1.0/Zero_d))) ? 0 : 1;
		}

		{
			Error += true == glm::isinf( 1.0f/Zero_f) ? 0 : 1;
			Error += true == glm::isinf(-1.0f/Zero_f) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec2( 1.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec2(-1.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec3( 1.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec3(-1.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec4( 1.0f/Zero_f))) ? 0 : 1;
			Error += true == glm::any(glm::isinf(glm::vec4(-1.0f/Zero_f))) ? 0 : 1;
		}

		return Error;
	}
}//namespace isinf_

namespace sign
{
	template<typename genFIType> 
	GLM_FUNC_QUALIFIER genFIType sign_if(genFIType x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<genFIType>::is_iec559 ||
			(std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer), "'sign' only accept signed inputs");

		genFIType result;
		if(x > genFIType(0))
			result = genFIType(1);
		else if(x < genFIType(0))
			result = genFIType(-1);
		else
			result = genFIType(0);
		return result;
	}

	template<typename genFIType> 
	GLM_FUNC_QUALIFIER genFIType sign_alu1(genFIType x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer, 
			"'sign' only accept integer inputs");

		return (x >> 31) | (static_cast<unsigned>(-x) >> 31);
	}

	GLM_FUNC_QUALIFIER int sign_alu2(int x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<int>::is_signed && std::numeric_limits<int>::is_integer, "'sign' only accept integer inputs");

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(push)
#			pragma warning(disable : 4146) //cast truncates constant value
#		endif

		return -(static_cast<unsigned>(x) >> 31) | (-static_cast<unsigned>(x) >> 31);

#		if GLM_COMPILER & GLM_COMPILER_VC
#			pragma warning(pop)
#		endif
	}

	template<typename genFIType> 
	GLM_FUNC_QUALIFIER genFIType sign_sub(genFIType x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer, 
			"'sign' only accept integer inputs");

		return (static_cast<unsigned>(-x) >> 31) - (static_cast<unsigned>(x) >> 31);
	}

	template<typename genFIType> 
	GLM_FUNC_QUALIFIER genFIType sign_cmp(genFIType x)
	{
		GLM_STATIC_ASSERT(
			std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer, 
			"'sign' only accept integer inputs");

		return (x > 0) - (x < 0);
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
			{ std::numeric_limits<glm::int32>::max(),  1},
			{ std::numeric_limits<glm::int32>::min(), -1},
			{ 0, 0},
			{ 1, 1},
			{ 2, 1},
			{ 3, 1},
			{-1,-1},
			{-2,-1},
			{-3,-1}
		};

		int Error = 0;

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::int32>); ++i)
		{
			glm::int32 Result = glm::sign(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::int32>); ++i)
		{
			glm::int32 Result = sign_cmp(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::int32>); ++i)
		{
			glm::int32 Result = sign_if(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::int32>); ++i)
		{
			glm::int32 Result = sign_alu1(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::int32>); ++i)
		{
			glm::int32 Result = sign_alu2(Data[i].Value);
			Error += Data[i].Return == Result ? 0 : 1;
		}

		return Error;
	}

	int test_i32vec4()
	{
		type<glm::ivec4> const Data[] =
		{
			{glm::ivec4( 1), glm::ivec4( 1)},
			{glm::ivec4( 0), glm::ivec4( 0)},
			{glm::ivec4( 2), glm::ivec4( 1)},
			{glm::ivec4( 3), glm::ivec4( 1)},
			{glm::ivec4(-1), glm::ivec4(-1)},
			{glm::ivec4(-2), glm::ivec4(-1)},
			{glm::ivec4(-3), glm::ivec4(-1)}
		};

		int Error = 0;

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::ivec4>); ++i)
		{
			glm::ivec4 Result = glm::sign(Data[i].Value);
			Error += glm::all(glm::equal(Data[i].Return, Result)) ? 0 : 1;
		}

		return Error;
	}

	int test_f32vec4()
	{
		type<glm::vec4> const Data[] =
		{
			{glm::vec4( 1), glm::vec4( 1)},
			{glm::vec4( 0), glm::vec4( 0)},
			{glm::vec4( 2), glm::vec4( 1)},
			{glm::vec4( 3), glm::vec4( 1)},
			{glm::vec4(-1), glm::vec4(-1)},
			{glm::vec4(-2), glm::vec4(-1)},
			{glm::vec4(-3), glm::vec4(-1)}
		};

		int Error = 0;

		for(std::size_t i = 0; i < sizeof(Data) / sizeof(type<glm::vec4>); ++i)
		{
			glm::vec4 Result = glm::sign(Data[i].Value);
			Error += glm::all(glm::equal(Data[i].Return, Result, glm::epsilon<float>())) ? 0 : 1;
		}

		return Error;
	}

	static int test()
	{
		int Error = 0;

		Error += test_int32();
		Error += test_i32vec4();
		Error += test_f32vec4();

		return Error;
	}

	int perf_rand(std::size_t Samples)
	{
		int Error = 0;

		std::size_t const Count = Samples;
		std::vector<glm::int32> Input, Output;
		Input.resize(Count);
		Output.resize(Count);
		for(std::size_t i = 0; i < Count; ++i)
			Input[i] = static_cast<glm::int32>(glm::linearRand(-65536.f, 65536.f));

		std::clock_t Timestamp0 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_cmp(Input[i]);

		std::clock_t Timestamp1 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_if(Input[i]);

		std::clock_t Timestamp2 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_alu1(Input[i]);

		std::clock_t Timestamp3 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_alu2(Input[i]);

		std::clock_t Timestamp4 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_sub(Input[i]);

		std::clock_t Timestamp5 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = glm::sign(Input[i]);

		std::clock_t Timestamp6 = std::clock();

		std::printf("sign_cmp(rand) Time %d clocks\n", static_cast<int>(Timestamp1 - Timestamp0));
		std::printf("sign_if(rand) Time %d clocks\n", static_cast<int>(Timestamp2 - Timestamp1));
		std::printf("sign_alu1(rand) Time %d clocks\n", static_cast<int>(Timestamp3 - Timestamp2));
		std::printf("sign_alu2(rand) Time %d clocks\n", static_cast<int>(Timestamp4 - Timestamp3));
		std::printf("sign_sub(rand) Time %d clocks\n", static_cast<int>(Timestamp5 - Timestamp4));
		std::printf("glm::sign(rand) Time %d clocks\n", static_cast<int>(Timestamp6 - Timestamp5));

		return Error;
	}

	int perf_linear(std::size_t Samples)
	{
		int Error = 0;

		std::size_t const Count = Samples;
		std::vector<glm::int32> Input, Output;
		Input.resize(Count);
		Output.resize(Count);
		for(std::size_t i = 0; i < Count; ++i)
			Input[i] = static_cast<glm::int32>(i);

		std::clock_t Timestamp0 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_cmp(Input[i]);

		std::clock_t Timestamp1 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_if(Input[i]);

		std::clock_t Timestamp2 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_alu1(Input[i]);

		std::clock_t Timestamp3 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_alu2(Input[i]);

		std::clock_t Timestamp4 = std::clock();

		for(std::size_t i = 0; i < Count; ++i)
			Output[i] = sign_sub(Input[i]);

		std::clock_t Timestamp5 = std::clock();

		std::printf("sign_cmp(linear) Time %d clocks\n", static_cast<int>(Timestamp1 - Timestamp0));
		std::printf("sign_if(linear) Time %d clocks\n", static_cast<int>(Timestamp2 - Timestamp1));
		std::printf("sign_alu1(linear) Time %d clocks\n", static_cast<int>(Timestamp3 - Timestamp2));
		std::printf("sign_alu2(linear) Time %d clocks\n", static_cast<int>(Timestamp4 - Timestamp3));
		std::printf("sign_sub(linear) Time %d clocks\n", static_cast<int>(Timestamp5 - Timestamp4));

		return Error;
	}

	int perf_linear_cal(std::size_t Samples)
	{
		int Error = 0;

		glm::int32 const Count = static_cast<glm::int32>(Samples);

		std::clock_t Timestamp0 = std::clock();
		glm::int32 Sum = 0;

		for(glm::int32 i = 1; i < Count; ++i)
			Sum += sign_cmp(i);

		std::clock_t Timestamp1 = std::clock();

		for(glm::int32 i = 1; i < Count; ++i)
			Sum += sign_if(i);

		std::clock_t Timestamp2 = std::clock();

		for(glm::int32 i = 1; i < Count; ++i)
			Sum += sign_alu1(i);

		std::clock_t Timestamp3 = std::clock();

		for(glm::int32 i = 1; i < Count; ++i)
			Sum += sign_alu2(i);

		std::clock_t Timestamp4 = std::clock();

		for(glm::int32 i = 1; i < Count; ++i)
			Sum += sign_sub(i);

		std::clock_t Timestamp5 = std::clock();

		std::printf("Sum %d\n", static_cast<int>(Sum));

		std::printf("sign_cmp(linear_cal) Time %d clocks\n", static_cast<int>(Timestamp1 - Timestamp0));
		std::printf("sign_if(linear_cal) Time %d clocks\n", static_cast<int>(Timestamp2 - Timestamp1));
		std::printf("sign_alu1(linear_cal) Time %d clocks\n", static_cast<int>(Timestamp3 - Timestamp2));
		std::printf("sign_alu2(linear_cal) Time %d clocks\n", static_cast<int>(Timestamp4 - Timestamp3));
		std::printf("sign_sub(linear_cal) Time %d clocks\n", static_cast<int>(Timestamp5 - Timestamp4));

		return Error;
	}

	static int perf(std::size_t Samples)
	{
		int Error(0);

		Error += perf_linear_cal(Samples);
		Error += perf_linear(Samples);
		Error += perf_rand(Samples);

		return Error;
	}
}//namespace sign

namespace frexp_
{
	static int test()
	{
		int Error = 0;

		{
			glm::vec1 const x(1024);
			glm::ivec1 exp;
			glm::vec1 A = glm::frexp(x, exp);
			Error += glm::all(glm::equal(A, glm::vec1(0.5), glm::epsilon<float>())) ? 0 : 1;
			Error += glm::all(glm::equal(exp, glm::ivec1(11))) ? 0 : 1;
		}

		{
			glm::vec2 const x(1024, 0.24);
			glm::ivec2 exp;
			glm::vec2 A = glm::frexp(x, exp);
			Error += glm::all(glm::equal(A, glm::vec2(0.5, 0.96), glm::epsilon<float>())) ? 0 : 1;
			Error += glm::all(glm::equal(exp, glm::ivec2(11, -2))) ? 0 : 1;
		}

		{
			glm::vec3 const x(1024, 0.24, 0);
			glm::ivec3 exp;
			glm::vec3 A = glm::frexp(x, exp);
			Error += glm::all(glm::equal(A, glm::vec3(0.5, 0.96, 0.0), glm::epsilon<float>())) ? 0 : 1;
			Error += glm::all(glm::equal(exp, glm::ivec3(11, -2, 0))) ? 0 : 1;
		}

		{
			glm::vec4 const x(1024, 0.24, 0, -1.33);
			glm::ivec4 exp;
			glm::vec4 A = glm::frexp(x, exp);
			Error += glm::all(glm::equal(A, glm::vec4(0.5, 0.96, 0.0, -0.665), glm::epsilon<float>())) ? 0 : 1;
			Error += glm::all(glm::equal(exp, glm::ivec4(11, -2, 0, 1))) ? 0 : 1;
		}

		return Error;
	}
}//namespace frexp_

namespace ldexp_
{
	static int test()
	{
		int Error(0);

		{
			glm::vec1 A = glm::vec1(0.5);
			glm::ivec1 exp = glm::ivec1(11);
			glm::vec1 x = glm::ldexp(A, exp);
			Error += glm::all(glm::equal(x, glm::vec1(1024),0.00001f)) ? 0 : 1;
		}

		{
			glm::vec2 A = glm::vec2(0.5, 0.96);
			glm::ivec2 exp = glm::ivec2(11, -2);
			glm::vec2 x = glm::ldexp(A, exp);
			Error += glm::all(glm::equal(x, glm::vec2(1024, .24),0.00001f)) ? 0 : 1;
		}

		{
			glm::vec3 A = glm::vec3(0.5, 0.96, 0.0);
			glm::ivec3 exp = glm::ivec3(11, -2, 0);
			glm::vec3 x = glm::ldexp(A, exp);
			Error += glm::all(glm::equal(x, glm::vec3(1024, .24, 0),0.00001f)) ? 0 : 1;
		}

		{
			glm::vec4 A = glm::vec4(0.5, 0.96, 0.0, -0.665);
			glm::ivec4 exp = glm::ivec4(11, -2, 0, 1);
			glm::vec4 x = glm::ldexp(A, exp);
			Error += glm::all(glm::equal(x, glm::vec4(1024, .24, 0, -1.33),0.00001f)) ? 0 : 1;
		}

		return Error;
	}
}//namespace ldexp_

static int test_constexpr()
{
#if GLM_HAS_CONSTEXPR
	static_assert(glm::abs(1.0f) > 0.0f, "GLM: Failed constexpr");
	constexpr glm::vec1 const A = glm::abs(glm::vec1(1.0f));
	constexpr glm::vec2 const B = glm::abs(glm::vec2(1.0f));
	constexpr glm::vec3 const C = glm::abs(glm::vec3(1.0f));
	constexpr glm::vec4 const D = glm::abs(glm::vec4(1.0f));
#endif // GLM_HAS_CONSTEXPR

	return 0;
}

int main()
{
	int Error = 0;

	Error += test_constexpr();
	Error += sign::test();
	Error += floor_::test();
	Error += mod_::test();
	Error += modf_::test();
	Error += floatBitsToInt::test();
	Error += floatBitsToUint::test();
	Error += mix_::test();
	Error += step_::test();
	Error += max_::test();
	Error += min_::test();
	Error += clamp_::test();
	Error += round_::test();
	Error += roundEven::test();
	Error += isnan_::test();
	Error += isinf_::test();
	Error += frexp_::test();
	Error += ldexp_::test();

#	ifdef NDEBUG
		std::size_t Samples = 1000;
#	else
		std::size_t Samples = 1;
#	endif
	Error += sign::perf(Samples);

	Error += min_::perf(Samples);

	return Error;
}

