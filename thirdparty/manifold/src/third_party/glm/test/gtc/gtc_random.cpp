#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/gtc/random.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_precision.hpp>
#if GLM_LANG & GLM_LANG_CXX0X_FLAG
#	include <array>
#endif

std::size_t const TestSamples = 10000;

int test_linearRand()
{
	int Error = 0;

	glm::int32 const Min = 16;
	glm::int32 const Max = 32;

	{
		glm::u8vec2 AMin(std::numeric_limits<glm::u8>::max());
		glm::u8vec2 AMax(std::numeric_limits<glm::u8>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::u8vec2 A = glm::linearRand(glm::u8vec2(Min), glm::u8vec2(Max));
				AMin = glm::min(AMin, A);
				AMax = glm::max(AMax, A);

				if(!glm::all(glm::lessThanEqual(A, glm::u8vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(A, glm::u8vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(AMin, glm::u8vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(AMax, glm::u8vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::u16vec2 BMin(std::numeric_limits<glm::u16>::max());
		glm::u16vec2 BMax(std::numeric_limits<glm::u16>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::u16vec2 B = glm::linearRand(glm::u16vec2(Min), glm::u16vec2(Max));
				BMin = glm::min(BMin, B);
				BMax = glm::max(BMax, B);

				if(!glm::all(glm::lessThanEqual(B, glm::u16vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(B, glm::u16vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(BMin, glm::u16vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(BMax, glm::u16vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::u32vec2 CMin(std::numeric_limits<glm::u32>::max());
		glm::u32vec2 CMax(std::numeric_limits<glm::u32>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::u32vec2 C = glm::linearRand(glm::u32vec2(Min), glm::u32vec2(Max));
				CMin = glm::min(CMin, C);
				CMax = glm::max(CMax, C);

				if(!glm::all(glm::lessThanEqual(C, glm::u32vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(C, glm::u32vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(CMin, glm::u32vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(CMax, glm::u32vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::u64vec2 DMin(std::numeric_limits<glm::u64>::max());
		glm::u64vec2 DMax(std::numeric_limits<glm::u64>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::u64vec2 D = glm::linearRand(glm::u64vec2(Min), glm::u64vec2(Max));
				DMin = glm::min(DMin, D);
				DMax = glm::max(DMax, D);

				if(!glm::all(glm::lessThanEqual(D, glm::u64vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(D, glm::u64vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(DMin, glm::u64vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(DMax, glm::u64vec2(Max))) ? 0 : 1;
			assert(!Error);
		}
	}

	{
		glm::i8vec2 AMin(std::numeric_limits<glm::i8>::max());
		glm::i8vec2 AMax(std::numeric_limits<glm::i8>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::i8vec2 A = glm::linearRand(glm::i8vec2(Min), glm::i8vec2(Max));
				AMin = glm::min(AMin, A);
				AMax = glm::max(AMax, A);

				if(!glm::all(glm::lessThanEqual(A, glm::i8vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(A, glm::i8vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(AMin, glm::i8vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(AMax, glm::i8vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::i16vec2 BMin(std::numeric_limits<glm::i16>::max());
		glm::i16vec2 BMax(std::numeric_limits<glm::i16>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::i16vec2 B = glm::linearRand(glm::i16vec2(Min), glm::i16vec2(Max));
				BMin = glm::min(BMin, B);
				BMax = glm::max(BMax, B);

				if(!glm::all(glm::lessThanEqual(B, glm::i16vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(B, glm::i16vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(BMin, glm::i16vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(BMax, glm::i16vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::i32vec2 CMin(std::numeric_limits<glm::i32>::max());
		glm::i32vec2 CMax(std::numeric_limits<glm::i32>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::i32vec2 C = glm::linearRand(glm::i32vec2(Min), glm::i32vec2(Max));
				CMin = glm::min(CMin, C);
				CMax = glm::max(CMax, C);

				if(!glm::all(glm::lessThanEqual(C, glm::i32vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(C, glm::i32vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(CMin, glm::i32vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(CMax, glm::i32vec2(Max))) ? 0 : 1;
			assert(!Error);
		}

		glm::i64vec2 DMin(std::numeric_limits<glm::i64>::max());
		glm::i64vec2 DMax(std::numeric_limits<glm::i64>::min());
		{
			for(std::size_t i = 0; i < TestSamples; ++i)
			{
				glm::i64vec2 D = glm::linearRand(glm::i64vec2(Min), glm::i64vec2(Max));
				DMin = glm::min(DMin, D);
				DMax = glm::max(DMax, D);

				if(!glm::all(glm::lessThanEqual(D, glm::i64vec2(Max))))
					++Error;
				if(!glm::all(glm::greaterThanEqual(D, glm::i64vec2(Min))))
					++Error;
				assert(!Error);
			}

			Error += glm::all(glm::equal(DMin, glm::i64vec2(Min))) ? 0 : 1;
			Error += glm::all(glm::equal(DMax, glm::i64vec2(Max))) ? 0 : 1;
			assert(!Error);
		}
	}

	for(std::size_t i = 0; i < TestSamples; ++i)
	{
		glm::f32vec2 const A(glm::linearRand(glm::f32vec2(static_cast<float>(Min)), glm::f32vec2(static_cast<float>(Max))));
		if(!glm::all(glm::lessThanEqual(A, glm::f32vec2(static_cast<float>(Max)))))
			++Error;
		if(!glm::all(glm::greaterThanEqual(A, glm::f32vec2(static_cast<float>(Min)))))
			++Error;

		glm::f64vec2 const B(glm::linearRand(glm::f64vec2(Min), glm::f64vec2(Max)));
		if(!glm::all(glm::lessThanEqual(B, glm::f64vec2(Max))))
			++Error;
		if(!glm::all(glm::greaterThanEqual(B, glm::f64vec2(Min))))
			++Error;
		assert(!Error);
	}

	{
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0;
		for(std::size_t i = 0; i < TestSamples; ++i)
		{
			ResultFloat += glm::linearRand(-1.0f, 1.0f);
			ResultDouble += glm::linearRand(-1.0, 1.0);
		}

		Error += glm::epsilonEqual(ResultFloat, 0.0f, 0.0001f);
		Error += glm::epsilonEqual(ResultDouble, 0.0, 0.0001);
		assert(!Error);
	}

	return Error;
}

int test_circularRand()
{
	int Error = 0;

	{
		std::size_t Max = TestSamples;
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0;
		double Radius = 2.0;

		for(std::size_t i = 0; i < Max; ++i)
		{
			ResultFloat += glm::length(glm::circularRand(1.0f));
			ResultDouble += glm::length(glm::circularRand(Radius));
		}

		Error += glm::epsilonEqual(ResultFloat, float(Max), 0.01f) ? 0 : 1;
		Error += glm::epsilonEqual(ResultDouble, double(Max) * double(Radius), 0.01) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_sphericalRand()
{
	int Error = 0;

	{
		std::size_t Max = TestSamples;
		float ResultFloatA = 0.0f;
		float ResultFloatB = 0.0f;
		float ResultFloatC = 0.0f;
		double ResultDoubleA = 0.0;
		double ResultDoubleB = 0.0;
		double ResultDoubleC = 0.0;

		for(std::size_t i = 0; i < Max; ++i)
		{
			ResultFloatA += glm::length(glm::sphericalRand(1.0f));
			ResultDoubleA += glm::length(glm::sphericalRand(1.0));
			ResultFloatB += glm::length(glm::sphericalRand(2.0f));
			ResultDoubleB += glm::length(glm::sphericalRand(2.0));
			ResultFloatC += glm::length(glm::sphericalRand(3.0f));
			ResultDoubleC += glm::length(glm::sphericalRand(3.0));
		}

		Error += glm::epsilonEqual(ResultFloatA, float(Max), 0.01f) ? 0 : 1;
		Error += glm::epsilonEqual(ResultDoubleA, double(Max), 0.0001) ? 0 : 1;
		Error += glm::epsilonEqual(ResultFloatB, float(Max * 2), 0.01f) ? 0 : 1;
		Error += glm::epsilonEqual(ResultDoubleB, double(Max * 2), 0.0001) ? 0 : 1;
		Error += glm::epsilonEqual(ResultFloatC, float(Max * 3), 0.01f) ? 0 : 1;
		Error += glm::epsilonEqual(ResultDoubleC, double(Max * 3), 0.01) ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_diskRand()
{
	int Error = 0;

	{
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0;

		for(std::size_t i = 0; i < TestSamples; ++i)
		{
			ResultFloat += glm::length(glm::diskRand(2.0f));
			ResultDouble += glm::length(glm::diskRand(2.0));
		}

		Error += ResultFloat < float(TestSamples) * 2.f ? 0 : 1;
		Error += ResultDouble < double(TestSamples) * 2.0 ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int test_ballRand()
{
	int Error = 0;

	{
		float ResultFloat = 0.0f;
		double ResultDouble = 0.0;

		for(std::size_t i = 0; i < TestSamples; ++i)
		{
			ResultFloat += glm::length(glm::ballRand(2.0f));
			ResultDouble += glm::length(glm::ballRand(2.0));
		}

		Error += ResultFloat < float(TestSamples) * 2.f ? 0 : 1;
		Error += ResultDouble < double(TestSamples) * 2.0 ? 0 : 1;
		assert(!Error);
	}

	return Error;
}
/*
#if(GLM_LANG & GLM_LANG_CXX0X_FLAG)
int test_grid()
{
	int Error = 0;

	typedef std::array<int, 8> colors;
	typedef std::array<int, 8 * 8> grid;

	grid Grid;
	colors Colors;

	grid GridBest;
	colors ColorsBest;

	while(true)
	{
		for(std::size_t i = 0; i < Grid.size(); ++i)
			Grid[i] = int(glm::linearRand(0.0, 8.0 * 8.0 * 8.0 - 1.0) / 64.0);

		for(std::size_t i = 0; i < Grid.size(); ++i)
			++Colors[Grid[i]];

		bool Exit = true;
		for(std::size_t i = 0; i < Colors.size(); ++i)
		{
			if(Colors[i] == 8)
				continue;

			Exit = false;
			break;
		}

		if(Exit == true)
			break;
	}

	return Error;
}
#endif
*/
int main()
{
	int Error = 0;

	Error += test_linearRand();
	Error += test_circularRand();
	Error += test_sphericalRand();
	Error += test_diskRand();
	Error += test_ballRand();
/*
#if(GLM_LANG & GLM_LANG_CXX0X_FLAG)
	Error += test_grid();
#endif
*/
	return Error;
}
