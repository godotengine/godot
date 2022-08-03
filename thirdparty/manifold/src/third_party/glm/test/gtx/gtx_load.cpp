#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
/*
#if GLM_CONFIG_SIMD == GLM_ENABLE

#include <glm/gtx/common.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_aligned.hpp>
#include <glm/ext/vector_relational.hpp>

namespace glm
{
	enum genTypeEnum
	{
		QUALIFIER_HIGHP,
		QUALIFIER_MEDIUMP,
		QUALIFIER_LOWP
	};

	template <typename genType>
	struct genTypeTrait
	{};

	template <length_t L, typename T>
	struct genTypeTrait<vec<L, T, aligned_highp> >
	{
		static const genTypeEnum GENTYPE = QUALIFIER_HIGHP;
	};

	template <length_t L, typename T>
	struct genTypeTrait<vec<L, T, aligned_mediump> >
	{
		static const genTypeEnum GENTYPE = QUALIFIER_MEDIUMP;
	};

	template <length_t L, typename T>
	struct genTypeTrait<vec<L, T, aligned_lowp> >
	{
		static const genTypeEnum GENTYPE = QUALIFIER_LOWP;
	};

	template<length_t L, typename T, qualifier Q, bool isAligned>
	struct load_gentype
	{
	
	};

#	if GLM_ARCH & GLM_ARCH_SSE_BIT
	template<qualifier Q>
	struct load_gentype<4, float, Q, true>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<4, float, Q> load(float const* Mem)
		{
			vec<4, float, Q> Result;
			Result.data = _mm_loadu_ps(Mem);
			return Result;
		}
	};
#	endif//GLM_ARCH & GLM_ARCH_SSE_BIT

	template<typename genType>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType example_identity()
	{
		return detail::init_gentype<genType, detail::genTypeTrait<genType>::GENTYPE>::identity();
	}

	template <typename genType, typename valType>
	genType load(valType const* Mem)
	{
		
	}

	aligned_vec4 loadu(float const* Mem)
	{
		aligned_vec4 Result;
#		if GLM_ARCH & GLM_ARCH_SSE_BIT
			Result.data = _mm_loadu_ps(Mem);
#		else
			Result[0] = *(Mem + 0);
			Result[1] = *(Mem + 1);
			Result[2] = *(Mem + 2);
			Result[3] = *(Mem + 3);
#		endif//GLM_ARCH & GLM_ARCH_SSE_BIT
		return Result;
	}

	aligned_vec4 loada(float const* Mem)
	{
		aligned_vec4 Result;
#		if GLM_ARCH & GLM_ARCH_SSE_BIT
			Result.data = _mm_load_ps(Mem);
#		else
			Result[0] = *(Mem + 0);
			Result[1] = *(Mem + 1);
			Result[2] = *(Mem + 2);
			Result[3] = *(Mem + 3);
#		endif//GLM_ARCH & GLM_ARCH_SSE_BIT
		return Result;
	}
}//namespace glm

int test_vec4_load()
{
	int Error = 0;

	float Data[] = {1.f, 2.f, 3.f, 4.f};
	glm::aligned_vec4 const V = glm::loadu(Data);
	Error += glm::all(glm::equal(V, glm::aligned_vec4(1.f, 2.f, 3.f, 4.f), glm::epsilon<float>())) ? 0 : 1;

	return Error;
}
#endif
*/
int main()
{
	int Error = 0;
/*
#	if GLM_CONFIG_SIMD == GLM_ENABLE
		Error += test_vec4_load();
#	endif
*/
	return Error;
}
