#include <glm/ext/scalar_common.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/scalar_relational.hpp>
#include <glm/common.hpp>

#if ((GLM_LANG & GLM_LANG_CXX11_FLAG) || (GLM_COMPILER & GLM_COMPILER_VC))
#	define GLM_NAN(T) NAN
#else
#	define GLM_NAN(T) (static_cast<T>(0.0f) / static_cast<T>(0.0f))
#endif

template <typename T>
static int test_min()
{
	int Error = 0;

	T const N = static_cast<T>(0);
	T const B = static_cast<T>(1);
	Error += glm::equal(glm::min(N, B), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(B, N), N, glm::epsilon<T>()) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::equal(glm::min(N, B, C), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(B, N, C), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(C, N, B), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(C, B, N), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(B, C, N), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(N, C, B), N, glm::epsilon<T>()) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += glm::equal(glm::min(D, N, B, C), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(B, D, N, C), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(C, N, D, B), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(C, B, D, N), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(B, C, N, D), N, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::min(N, C, B, D), N, glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_min_nan()
{
	int Error = 0;

	T const B = static_cast<T>(1);
	T const N = static_cast<T>(GLM_NAN(T));
	Error += glm::isnan(glm::min(N, B)) ? 0 : 1;
	Error += !glm::isnan(glm::min(B, N)) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::isnan(glm::min(N, B, C)) ? 0 : 1;
	Error += !glm::isnan(glm::min(B, N, C)) ? 0 : 1;
	Error += !glm::isnan(glm::min(C, N, B)) ? 0 : 1;
	Error += !glm::isnan(glm::min(C, B, N)) ? 0 : 1;
	Error += !glm::isnan(glm::min(B, C, N)) ? 0 : 1;
	Error += glm::isnan(glm::min(N, C, B)) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += !glm::isnan(glm::min(D, N, B, C)) ? 0 : 1;
	Error += !glm::isnan(glm::min(B, D, N, C)) ? 0 : 1;
	Error += !glm::isnan(glm::min(C, N, D, B)) ? 0 : 1;
	Error += !glm::isnan(glm::min(C, B, D, N)) ? 0 : 1;
	Error += !glm::isnan(glm::min(B, C, N, D)) ? 0 : 1;
	Error += glm::isnan(glm::min(N, C, B, D)) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_max()
{
	int Error = 0;

	T const N = static_cast<T>(0);
	T const B = static_cast<T>(1);
	Error += glm::equal(glm::max(N, B), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(B, N), B, glm::epsilon<T>()) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::equal(glm::max(N, B, C), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(B, N, C), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(C, N, B), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(C, B, N), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(B, C, N), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(N, C, B), C, glm::epsilon<T>()) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += glm::equal(glm::max(D, N, B, C), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(B, D, N, C), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(C, N, D, B), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(C, B, D, N), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(B, C, N, D), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::max(N, C, B, D), D, glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_max_nan()
{
	int Error = 0;

	T const B = static_cast<T>(1);
	T const N = static_cast<T>(GLM_NAN(T));
	Error += glm::isnan(glm::max(N, B)) ? 0 : 1;
	Error += !glm::isnan(glm::max(B, N)) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::isnan(glm::max(N, B, C)) ? 0 : 1;
	Error += !glm::isnan(glm::max(B, N, C)) ? 0 : 1;
	Error += !glm::isnan(glm::max(C, N, B)) ? 0 : 1;
	Error += !glm::isnan(glm::max(C, B, N)) ? 0 : 1;
	Error += !glm::isnan(glm::max(B, C, N)) ? 0 : 1;
	Error += glm::isnan(glm::max(N, C, B)) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += !glm::isnan(glm::max(D, N, B, C)) ? 0 : 1;
	Error += !glm::isnan(glm::max(B, D, N, C)) ? 0 : 1;
	Error += !glm::isnan(glm::max(C, N, D, B)) ? 0 : 1;
	Error += !glm::isnan(glm::max(C, B, D, N)) ? 0 : 1;
	Error += !glm::isnan(glm::max(B, C, N, D)) ? 0 : 1;
	Error += glm::isnan(glm::max(N, C, B, D)) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_fmin()
{
	int Error = 0;

	T const B = static_cast<T>(1);
	T const N = static_cast<T>(GLM_NAN(T));
	Error += glm::equal(glm::fmin(N, B), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(B, N), B, glm::epsilon<T>()) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::equal(glm::fmin(N, B, C), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(B, N, C), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(C, N, B), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(C, B, N), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(B, C, N), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(N, C, B), B, glm::epsilon<T>()) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += glm::equal(glm::fmin(D, N, B, C), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(B, D, N, C), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(C, N, D, B), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(C, B, D, N), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(B, C, N, D), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmin(N, C, B, D), B, glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

template <typename T>
static int test_fmax()
{
	int Error = 0;

	T const B = static_cast<T>(1);
	T const N = static_cast<T>(GLM_NAN(T));
	Error += glm::equal(glm::fmax(N, B), B, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(B, N), B, glm::epsilon<T>()) ? 0 : 1;

	T const C = static_cast<T>(2);
	Error += glm::equal(glm::fmax(N, B, C), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(B, N, C), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(C, N, B), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(C, B, N), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(B, C, N), C, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(N, C, B), C, glm::epsilon<T>()) ? 0 : 1;

	T const D = static_cast<T>(3);
	Error += glm::equal(glm::fmax(D, N, B, C), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(B, D, N, C), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(C, N, D, B), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(C, B, D, N), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(B, C, N, D), D, glm::epsilon<T>()) ? 0 : 1;
	Error += glm::equal(glm::fmax(N, C, B, D), D, glm::epsilon<T>()) ? 0 : 1;

	return Error;
}

static int test_clamp()
{
	int Error = 0;

	float A = glm::clamp(0.5f);
	Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

	float B = glm::clamp(0.0f);
	Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

	float C = glm::clamp(1.0f);
	Error += glm::equal(C, 1.0f, 0.00001f) ? 0 : 1;

	float D = glm::clamp(-0.5f);
	Error += glm::equal(D, 0.0f, 0.00001f) ? 0 : 1;

	float E = glm::clamp(1.5f);
	Error += glm::equal(E, 1.0f, 0.00001f) ? 0 : 1;

	return Error;
}

static int test_repeat()
{
	int Error = 0;

	float A = glm::repeat(0.5f);
	Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

	float B = glm::repeat(0.0f);
	Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

	float C = glm::repeat(1.0f);
	Error += glm::equal(C, 0.0f, 0.00001f) ? 0 : 1;

	float D = glm::repeat(-0.5f);
	Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

	float E = glm::repeat(1.5f);
	Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

	float F = glm::repeat(0.9f);
	Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

	return Error;
}

static int test_mirrorClamp()
{
	int Error = 0;

	float A = glm::mirrorClamp(0.5f);
	Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

	float B = glm::mirrorClamp(0.0f);
	Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

	float C = glm::mirrorClamp(1.1f);
	Error += glm::equal(C, 0.1f, 0.00001f) ? 0 : 1;

	float D = glm::mirrorClamp(-0.5f);
	Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

	float E = glm::mirrorClamp(1.5f);
	Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

	float F = glm::mirrorClamp(0.9f);
	Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

	float G = glm::mirrorClamp(3.1f);
	Error += glm::equal(G, 0.1f, 0.00001f) ? 0 : 1;

	float H = glm::mirrorClamp(-3.1f);
	Error += glm::equal(H, 0.1f, 0.00001f) ? 0 : 1;

	float I = glm::mirrorClamp(-0.9f);
	Error += glm::equal(I, 0.9f, 0.00001f) ? 0 : 1;

	return Error;
}

static int test_mirrorRepeat()
{
	int Error = 0;

	float A = glm::mirrorRepeat(0.5f);
	Error += glm::equal(A, 0.5f, 0.00001f) ? 0 : 1;

	float B = glm::mirrorRepeat(0.0f);
	Error += glm::equal(B, 0.0f, 0.00001f) ? 0 : 1;

	float C = glm::mirrorRepeat(1.0f);
	Error += glm::equal(C, 1.0f, 0.00001f) ? 0 : 1;

	float D = glm::mirrorRepeat(-0.5f);
	Error += glm::equal(D, 0.5f, 0.00001f) ? 0 : 1;

	float E = glm::mirrorRepeat(1.5f);
	Error += glm::equal(E, 0.5f, 0.00001f) ? 0 : 1;

	float F = glm::mirrorRepeat(0.9f);
	Error += glm::equal(F, 0.9f, 0.00001f) ? 0 : 1;

	float G = glm::mirrorRepeat(3.0f);
	Error += glm::equal(G, 1.0f, 0.00001f) ? 0 : 1;

	float H = glm::mirrorRepeat(-3.0f);
	Error += glm::equal(H, 1.0f, 0.00001f) ? 0 : 1;

	float I = glm::mirrorRepeat(-1.0f);
	Error += glm::equal(I, 1.0f, 0.00001f) ? 0 : 1;

	return Error;
}

static int test_iround()
{
	int Error = 0;

	for(float f = 0.0f; f < 3.1f; f += 0.05f)
	{
		int RoundFast = static_cast<int>(glm::iround(f));
		int RoundSTD = static_cast<int>(glm::round(f));
		Error += RoundFast == RoundSTD ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

static int test_uround()
{
	int Error = 0;

	for(float f = 0.0f; f < 3.1f; f += 0.05f)
	{
		int RoundFast = static_cast<int>(glm::uround(f));
		int RoundSTD = static_cast<int>(glm::round(f));
		Error += RoundFast == RoundSTD ? 0 : 1;
		assert(!Error);
	}

	return Error;
}

int main()
{
	int Error = 0;

	Error += test_min<float>();
	Error += test_min<double>();
	Error += test_min_nan<float>();
	Error += test_min_nan<double>();

	Error += test_max<float>();
	Error += test_max<double>();
	Error += test_max_nan<float>();
	Error += test_max_nan<double>();

	Error += test_fmin<float>();
	Error += test_fmin<double>();

	Error += test_fmax<float>();
	Error += test_fmax<double>();

	Error += test_clamp();
	Error += test_repeat();
	Error += test_mirrorClamp();
	Error += test_mirrorRepeat();

	Error += test_iround();
	Error += test_uround();

	return Error;
}
