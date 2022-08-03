#define GLM_FORCE_INLINE
#include <glm/ext/matrix_float2x2.hpp>
#include <glm/ext/matrix_double2x2.hpp>
#include <glm/ext/matrix_float3x3.hpp>
#include <glm/ext/matrix_double3x3.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_double4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_relational.hpp>
#include <glm/ext/vector_float4.hpp>
#if GLM_CONFIG_SIMD == GLM_ENABLE
#include <glm/gtc/type_aligned.hpp>
#include <vector>
#include <chrono>
#include <cstdio>

template <typename matType, typename vecType>
static void test_vec_mul_mat(matType const& M, std::vector<vecType> const& I, std::vector<vecType>& O)
{
	for (std::size_t i = 0, n = I.size(); i < n; ++i)
		O[i] = I[i] * M;
}

template <typename matType, typename vecType>
static int launch_vec_mul_mat(std::vector<vecType>& O, matType const& Transform, vecType const& Scale, std::size_t Samples)
{
	typedef typename matType::value_type T;

	std::vector<vecType> I(Samples);
	O.resize(Samples);

	for(std::size_t i = 0; i < Samples; ++i)
		I[i] = Scale * static_cast<T>(i);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	test_vec_mul_mat<matType, vecType>(Transform, I, O);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	return static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
}

template <typename packedMatType, typename packedVecType, typename alignedMatType, typename alignedVecType>
static int comp_vec2_mul_mat2(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Transform(1, 2, 3, 4);
	packedVecType const Scale(0.01, 0.02);

	std::vector<packedVecType> SISD;
	std::printf("- SISD: %d us\n", launch_vec_mul_mat<packedMatType, packedVecType>(SISD, Transform, Scale, Samples));

	std::vector<alignedVecType> SIMD;
	std::printf("- SIMD: %d us\n", launch_vec_mul_mat<alignedMatType, alignedVecType>(SIMD, Transform, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedVecType const A = SISD[i];
		packedVecType const B = packedVecType(SIMD[i]);
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
	}
	
	return Error;
}

template <typename packedMatType, typename packedVecType, typename alignedMatType, typename alignedVecType>
static int comp_vec3_mul_mat3(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Transform(1, 2, 3, 4, 5, 6, 7, 8, 9);
	packedVecType const Scale(0.01, 0.02, 0.05);

	std::vector<packedVecType> SISD;
	std::printf("- SISD: %d us\n", launch_vec_mul_mat<packedMatType, packedVecType>(SISD, Transform, Scale, Samples));

	std::vector<alignedVecType> SIMD;
	std::printf("- SIMD: %d us\n", launch_vec_mul_mat<alignedMatType, alignedVecType>(SIMD, Transform, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedVecType const A = SISD[i];
		packedVecType const B = SIMD[i];
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
	}
	
	return Error;
}

template <typename packedMatType, typename packedVecType, typename alignedMatType, typename alignedVecType>
static int comp_vec4_mul_mat4(std::size_t Samples)
{
	typedef typename packedMatType::value_type T;
	
	int Error = 0;

	packedMatType const Transform(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
	packedVecType const Scale(0.01, 0.02, 0.03, 0.05);

	std::vector<packedVecType> SISD;
	std::printf("- SISD: %d us\n", launch_vec_mul_mat<packedMatType, packedVecType>(SISD, Transform, Scale, Samples));

	std::vector<alignedVecType> SIMD;
	std::printf("- SIMD: %d us\n", launch_vec_mul_mat<alignedMatType, alignedVecType>(SIMD, Transform, Scale, Samples));

	for(std::size_t i = 0; i < Samples; ++i)
	{
		packedVecType const A = SISD[i];
		packedVecType const B = SIMD[i];
		Error += glm::all(glm::equal(A, B, static_cast<T>(0.001))) ? 0 : 1;
	}
	
	return Error;
}

int main()
{
	std::size_t const Samples = 100000;
	
	int Error = 0;

	std::printf("vec2 * mat2:\n");
	Error += comp_vec2_mul_mat2<glm::mat2, glm::vec2, glm::aligned_mat2, glm::aligned_vec2>(Samples);
	
	std::printf("dvec2 * dmat2:\n");
	Error += comp_vec2_mul_mat2<glm::dmat2, glm::dvec2,glm::aligned_dmat2, glm::aligned_dvec2>(Samples);

	std::printf("vec3 * mat3:\n");
	Error += comp_vec3_mul_mat3<glm::mat3, glm::vec3, glm::aligned_mat3, glm::aligned_vec3>(Samples);
	
	std::printf("dvec3 * dmat3:\n");
	Error += comp_vec3_mul_mat3<glm::dmat3, glm::dvec3, glm::aligned_dmat3, glm::aligned_dvec3>(Samples);

	std::printf("vec4 * mat4:\n");
	Error += comp_vec4_mul_mat4<glm::mat4, glm::vec4, glm::aligned_mat4, glm::aligned_vec4>(Samples);
	
	std::printf("dvec4 * dmat4:\n");
	Error += comp_vec4_mul_mat4<glm::dmat4, glm::dvec4, glm::aligned_dmat4, glm::aligned_dvec4>(Samples);

	return Error;
}

#else

int main()
{
	return 0;
}

#endif
