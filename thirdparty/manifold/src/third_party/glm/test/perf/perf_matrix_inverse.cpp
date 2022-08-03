#define GLM_FORCE_INLINE
#include <glm/matrix.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/vector_float4.hpp>
#if GLM_CONFIG_SIMD == GLM_ENABLE
#include <glm/gtc/type_aligned.hpp>
#include <vector>
#include <chrono>
#include <cstdio>

template <typename matType>
static void test_mat_inverse(std::vector<matType> const& I, std::vector<matType>& O)
{
	for (std::size_t i = 0, n = I.size(); i < n; ++i)
		O[i] = glm::inverse(I[i]);
}

template <typename matType>
static int launch_mat_inverse(std::vector<matType>& O, matType const& Scale, std::size_t Samples)
{
	typedef typename matType::value_type T;

	std::vector<matType> I(Samples);
	O.resize(Samples);

	for(std::size_t i = 0; i < Samples; ++i)
		I[i] = Scale * static_cast<T>(i) + Scale;

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	test_mat_inverse<matType>(I, O);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	return static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
}

template <typename packedMatType, typename alignedMatType>
static int comp_mat2_inverse(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Scale(0.01, 0.02, 0.03, 0.05);

	std::vector<packedMatType> SISD;
	std::printf("- SISD: %d us\n", launch_mat_inverse<packedMatType>(SISD, Scale, Samples));

	std::vector<alignedMatType> SIMD;
	std::printf("- SIMD: %d us\n", launch_mat_inverse<alignedMatType>(SIMD, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedMatType const A = SISD[i];
		packedMatType const B = SIMD[i];
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

template <typename packedMatType, typename alignedMatType>
static int comp_mat3_inverse(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Scale(0.01, 0.02, 0.03, 0.05, 0.01, 0.02, 0.03, 0.05, 0.01);

	std::vector<packedMatType> SISD;
	std::printf("- SISD: %d us\n", launch_mat_inverse<packedMatType>(SISD, Scale, Samples));

	std::vector<alignedMatType> SIMD;
	std::printf("- SIMD: %d us\n", launch_mat_inverse<alignedMatType>(SIMD, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedMatType const A = SISD[i];
		packedMatType const B = SIMD[i];
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

template <typename packedMatType, typename alignedMatType>
static int comp_mat4_inverse(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Scale(0.01, 0.02, 0.05, 0.04, 0.02, 0.08, 0.05, 0.01, 0.08, 0.03, 0.05, 0.06, 0.02, 0.03, 0.07, 0.05);

	std::vector<packedMatType> SISD;
	std::printf("- SISD: %d us\n", launch_mat_inverse<packedMatType>(SISD, Scale, Samples));

	std::vector<alignedMatType> SIMD;
	std::printf("- SIMD: %d us\n", launch_mat_inverse<alignedMatType>(SIMD, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedMatType const A = SISD[i];
		packedMatType const B = SIMD[i];
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
		assert(!Error);
	}
	
	return Error;
}

int main()
{
	std::size_t const Samples = 100000;

	int Error = 0;

	std::printf("glm::inverse(mat2):\n");
	Error += comp_mat2_inverse<glm::mat2, glm::aligned_mat2>(Samples);
	
	std::printf("glm::inverse(dmat2):\n");
	Error += comp_mat2_inverse<glm::dmat2, glm::aligned_dmat2>(Samples);

	std::printf("glm::inverse(mat3):\n");
	Error += comp_mat3_inverse<glm::mat3, glm::aligned_mat3>(Samples);
	
	std::printf("glm::inverse(dmat3):\n");
	Error += comp_mat3_inverse<glm::dmat3, glm::aligned_dmat3>(Samples);

	std::printf("glm::inverse(mat4):\n");
	Error += comp_mat4_inverse<glm::mat4, glm::aligned_mat4>(Samples);
	
	std::printf("glm::inverse(dmat4):\n");
	Error += comp_mat4_inverse<glm::dmat4, glm::aligned_dmat4>(Samples);

	return Error;
}

#else

int main()
{
	return 0;
}

#endif
