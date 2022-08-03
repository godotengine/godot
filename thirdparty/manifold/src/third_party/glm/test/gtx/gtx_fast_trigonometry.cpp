#include <glm/ext/scalar_ulp.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/type_precision.hpp>
#include <glm/gtx/fast_trigonometry.hpp>
#include <glm/gtx/integer.hpp>
#include <glm/gtx/common.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/trigonometric.hpp>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <vector>

namespace fastCos
{
	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;

		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastCos(i);

		const std::clock_t timestamp2 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::cos(i);

		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;
		std::printf("fastCos Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("cos Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastCos

namespace fastSin
{
	/*
	float sin(float x) {
	float temp;
	temp = (x + M_PI) / ((2 * M_PI) - M_PI);
	return limited_sin((x + M_PI) - ((2 * M_PI) - M_PI) * temp));
	}
	*/

	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;

		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastSin(i);

		const std::clock_t timestamp2 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::sin(i);

		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;
		std::printf("fastSin Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("sin Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastSin

namespace fastTan
{
	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;

		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastTan(i);

		const std::clock_t timestamp2 = std::clock();
		for (float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::tan(i);

		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;
		std::printf("fastTan Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("tan Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastTan

namespace fastAcos
{
	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;

		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastAcos(i);

		const std::clock_t timestamp2 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::acos(i);

		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;

		std::printf("fastAcos Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("acos Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastAcos

namespace fastAsin
{
	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;
		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastAsin(i);
		const std::clock_t timestamp2 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::asin(i);
		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;
		std::printf("fastAsin Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("asin Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastAsin

namespace fastAtan
{
	int perf(bool NextFloat)
	{
		const float begin = -glm::pi<float>();
		const float end = glm::pi<float>();
		float result = 0.f;
		const std::clock_t timestamp1 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::fastAtan(i);
		const std::clock_t timestamp2 = std::clock();
		for(float i = begin; i < end; i = NextFloat ? glm::nextFloat(i) : i += 0.1f)
			result = glm::atan(i);
		const std::clock_t timestamp3 = std::clock();
		const std::clock_t time_fast = timestamp2 - timestamp1;
		const std::clock_t time_default = timestamp3 - timestamp2;
		std::printf("fastAtan Time %d clocks\n", static_cast<int>(time_fast));
		std::printf("atan Time %d clocks\n", static_cast<int>(time_default));

		return time_fast <= time_default ? 0 : 1;
	}
}//namespace fastAtan

namespace taylorCos
{
	using glm::qualifier;
	using glm::length_t;
	
	glm::vec4 const AngleShift(0.0f, glm::half_pi<float>(), glm::pi<float>(), glm::three_over_two_pi<float>());

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> taylorSeriesNewCos(glm::vec<L, T, Q> const& x)
	{
		glm::vec<L, T, Q> const Powed2(x * x);
		glm::vec<L, T, Q> const Powed4(Powed2 * Powed2);
		glm::vec<L, T, Q> const Powed6(Powed4 * Powed2);
		glm::vec<L, T, Q> const Powed8(Powed4 * Powed4);

		return static_cast<T>(1)
			- Powed2 * static_cast<T>(0.5)
			+ Powed4 * static_cast<T>(0.04166666666666666666666666666667)
			- Powed6 * static_cast<T>(0.00138888888888888888888888888889)
			+ Powed8 * static_cast<T>(2.4801587301587301587301587301587e-5);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> taylorSeriesNewCos6(glm::vec<L, T, Q> const& x)
	{
		glm::vec<L, T, Q> const Powed2(x * x);
		glm::vec<L, T, Q> const Powed4(Powed2 * Powed2);
		glm::vec<L, T, Q> const Powed6(Powed4 * Powed2);

		return static_cast<T>(1)
			- Powed2 * static_cast<T>(0.5)
			+ Powed4 * static_cast<T>(0.04166666666666666666666666666667)
			- Powed6 * static_cast<T>(0.00138888888888888888888888888889);
	}

	template<glm::length_t L, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, float, Q> fastAbs(glm::vec<L, float, Q> x)
	{
		int* Pointer = reinterpret_cast<int*>(&x[0]);
		Pointer[0] &= 0x7fffffff;
		Pointer[1] &= 0x7fffffff;
		Pointer[2] &= 0x7fffffff;
		Pointer[3] &= 0x7fffffff;
		return x;
	}

	template<glm::length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> fastCosNew(glm::vec<L, T, Q> const& x)
	{
		glm::vec<L, T, Q> const Angle0_PI(fastAbs(fmod(x + glm::pi<T>(), glm::two_pi<T>()) - glm::pi<T>()));
		return taylorSeriesNewCos6(x);
/*
		vec<L, bool, Q> const FirstQuarterPi(lessThanEqual(Angle0_PI, vec<L, T, Q>(glm::half_pi<T>())));

		vec<L, T, Q> const RevertAngle(mix(vec<L, T, Q>(glm::pi<T>()), vec<L, T, Q>(0), FirstQuarterPi));
		vec<L, T, Q> const ReturnSign(mix(vec<L, T, Q>(-1), vec<L, T, Q>(1), FirstQuarterPi));
		vec<L, T, Q> const SectionAngle(RevertAngle - Angle0_PI);

		return ReturnSign * taylorSeriesNewCos(SectionAngle);
*/
	}

	int perf_fastCosNew(float Begin, float End, std::size_t Samples)
	{
		std::vector<glm::vec4> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = fastCosNew(AngleShift + glm::vec4(Begin + Steps * static_cast<float>(i)));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("fastCosNew %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i].x >= -1.0f && Results[i].x <= 1.0f ? 0 : 1;
		return Error;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> deterministic_fmod(glm::vec<L, T, Q> const& x, T y)
	{
		return x - y * trunc(x / y);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> fastCosDeterminisctic(glm::vec<L, T, Q> const& x)
	{
		glm::vec<L, T, Q> const Angle0_PI(abs(deterministic_fmod(x + glm::pi<T>(), glm::two_pi<T>()) - glm::pi<T>()));
		glm::vec<L, bool, Q> const FirstQuarterPi(lessThanEqual(Angle0_PI, glm::vec<L, T, Q>(glm::half_pi<T>())));

		glm::vec<L, T, Q> const RevertAngle(mix(glm::vec<L, T, Q>(glm::pi<T>()), glm::vec<L, T, Q>(0), FirstQuarterPi));
		glm::vec<L, T, Q> const ReturnSign(mix(glm::vec<L, T, Q>(-1), glm::vec<L, T, Q>(1), FirstQuarterPi));
		glm::vec<L, T, Q> const SectionAngle(RevertAngle - Angle0_PI);

		return ReturnSign * taylorSeriesNewCos(SectionAngle);
	}

	int perf_fastCosDeterminisctic(float Begin, float End, std::size_t Samples)
	{
		std::vector<glm::vec4> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = taylorCos::fastCosDeterminisctic(AngleShift + glm::vec4(Begin + Steps * static_cast<float>(i)));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("fastCosDeterminisctic %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i].x >= -1.0f && Results[i].x <= 1.0f ? 0 : 1;
		return Error;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> taylorSeriesRefCos(glm::vec<L, T, Q> const& x)
	{
		return static_cast<T>(1)
			- (x * x) / glm::factorial(static_cast<T>(2))
			+ (x * x * x * x) / glm::factorial(static_cast<T>(4))
			- (x * x * x * x * x * x) / glm::factorial(static_cast<T>(6))
			+ (x * x * x * x * x * x * x * x) / glm::factorial(static_cast<T>(8));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER glm::vec<L, T, Q> fastRefCos(glm::vec<L, T, Q> const& x)
	{
		glm::vec<L, T, Q> const Angle0_PI(glm::abs(fmod(x + glm::pi<T>(), glm::two_pi<T>()) - glm::pi<T>()));
//		return taylorSeriesRefCos(Angle0_PI);

		glm::vec<L, bool, Q> const FirstQuarterPi(lessThanEqual(Angle0_PI, glm::vec<L, T, Q>(glm::half_pi<T>())));

		glm::vec<L, T, Q> const RevertAngle(mix(glm::vec<L, T, Q>(glm::pi<T>()), glm::vec<L, T, Q>(0), FirstQuarterPi));
		glm::vec<L, T, Q> const ReturnSign(mix(glm::vec<L, T, Q>(-1), glm::vec<L, T, Q>(1), FirstQuarterPi));
		glm::vec<L, T, Q> const SectionAngle(RevertAngle - Angle0_PI);

		return ReturnSign * taylorSeriesRefCos(SectionAngle);
	}

	int perf_fastCosRef(float Begin, float End, std::size_t Samples)
	{
		std::vector<glm::vec4> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = taylorCos::fastRefCos(AngleShift + glm::vec4(Begin + Steps * static_cast<float>(i)));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("fastCosRef %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i].x >= -1.0f && Results[i].x <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf_fastCosOld(float Begin, float End, std::size_t Samples)
	{
		std::vector<glm::vec4> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = glm::fastCos(AngleShift + glm::vec4(Begin + Steps * static_cast<float>(i)));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("fastCosOld %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i].x >= -1.0f && Results[i].x <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf_cos(float Begin, float End, std::size_t Samples)
	{
		std::vector<glm::vec4> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = glm::cos(AngleShift + glm::vec4(Begin + Steps * static_cast<float>(i)));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("cos %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i].x >= -1.0f && Results[i].x <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf(std::size_t const Samples)
	{
		int Error = 0;

		float const Begin = -glm::pi<float>();
		float const End = glm::pi<float>();

		Error += perf_cos(Begin, End, Samples);
		Error += perf_fastCosOld(Begin, End, Samples);
		Error += perf_fastCosRef(Begin, End, Samples);
		//Error += perf_fastCosNew(Begin, End, Samples);
		Error += perf_fastCosDeterminisctic(Begin, End, Samples);

		return Error;
	}

	int test()
	{
		int Error = 0;

		//for(float Angle = -4.0f * glm::pi<float>(); Angle < 4.0f * glm::pi<float>(); Angle += 0.1f)
		//for(float Angle = -720.0f; Angle < 720.0f; Angle += 0.1f)
		for(float Angle = 0.0f; Angle < 180.0f; Angle += 0.1f)
		{
			float const modAngle = std::fmod(glm::abs(Angle), 360.f);
			assert(modAngle >= 0.0f && modAngle <= 360.f);
			float const radAngle = glm::radians(modAngle);
			float const Cos0 = std::cos(radAngle);

			float const Cos1 = taylorCos::fastRefCos(glm::fvec1(radAngle)).x;
			Error += glm::abs(Cos1 - Cos0) < 0.1f ? 0 : 1;

			//float const Cos2 = taylorCos::fastCosNew(glm::fvec1(radAngle)).x;
			//Error += glm::abs(Cos2 - Cos0) < 0.1f ? 0 : 1;

			assert(!Error);
		}

		return Error;
	}
}//namespace taylorCos

namespace taylor2
{
	glm::vec4 const AngleShift(0.0f, glm::pi<float>() * 0.5f, glm::pi<float>() * 1.0f, glm::pi<float>() * 1.5f);

	float taylorCosA(float x)
	{
		return 1.f
			- (x * x) * (1.f / 2.f)
			+ (x * x * x * x) * (1.f / 24.f)
			- (x * x * x * x * x * x) * (1.f / 720.f)
			+ (x * x * x * x * x * x * x * x) * (1.f / 40320.f);
	}

	float taylorCosB(float x)
	{
		return 1.f
			- (x * x) * (1.f / 2.f)
			+ (x * x * x * x) * (1.f / 24.f)
			- (x * x * x * x * x * x) * (1.f / 720.f)
			+ (x * x * x * x * x * x * x * x) * (1.f / 40320.f);
	}

	float taylorCosC(float x)
	{
		return 1.f
			- (x * x) * (1.f / 2.f)
			+ ((x * x) * (x * x)) * (1.f / 24.f)
			- (((x * x) * (x * x)) * (x * x)) * (1.f / 720.f)
			+ (((x * x) * (x * x)) * ((x * x) * (x * x))) * (1.f / 40320.f);
	}

	int perf_taylorCosA(float Begin, float End, std::size_t Samples)
	{
		std::vector<float> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = taylorCosA(AngleShift.x + Begin + Steps * static_cast<float>(i));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("taylorCosA %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i] >= -1.0f && Results[i] <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf_taylorCosB(float Begin, float End, std::size_t Samples)
	{
		std::vector<float> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = taylorCosB(AngleShift.x + Begin + Steps * static_cast<float>(i));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("taylorCosB %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i] >= -1.0f && Results[i] <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf_taylorCosC(float Begin, float End, std::size_t Samples)
	{
		std::vector<float> Results;
		Results.resize(Samples);

		float const Steps = (End - Begin) / static_cast<float>(Samples);

		std::clock_t const TimeStampBegin = std::clock();

		for(std::size_t i = 0; i < Samples; ++i)
			Results[i] = taylorCosC(AngleShift.x + Begin + Steps * static_cast<float>(i));

		std::clock_t const TimeStampEnd = std::clock();

		std::printf("taylorCosC %d clocks\n", static_cast<int>(TimeStampEnd - TimeStampBegin));

		int Error = 0;
		for(std::size_t i = 0; i < Samples; ++i)
			Error += Results[i] >= -1.0f && Results[i] <= 1.0f ? 0 : 1;
		return Error;
	}

	int perf(std::size_t Samples)
	{
		int Error = 0;

		float const Begin = -glm::pi<float>();
		float const End = glm::pi<float>();

		Error += perf_taylorCosA(Begin, End, Samples);
		Error += perf_taylorCosB(Begin, End, Samples);
		Error += perf_taylorCosC(Begin, End, Samples);

		return Error;
	}

}//namespace taylor2

int main()
{
	int Error(0);

	Error += ::taylor2::perf(1000);
	Error += ::taylorCos::test();
	Error += ::taylorCos::perf(1000);

#	ifdef NDEBUG
		::fastCos::perf(false);
		::fastSin::perf(false);
		::fastTan::perf(false);
		::fastAcos::perf(false);
		::fastAsin::perf(false);
		::fastAtan::perf(false);
#	endif//NDEBUG

	return Error;
}
