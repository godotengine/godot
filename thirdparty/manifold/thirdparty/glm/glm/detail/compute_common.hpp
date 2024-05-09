#pragma once

#include "setup.hpp"
#include <limits>

namespace glm{
namespace detail
{
	template<typename genFIType, bool /*signed*/>
	struct compute_abs
	{};

	template<typename genFIType>
	struct compute_abs<genFIType, true>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static genFIType call(genFIType x)
		{
			GLM_STATIC_ASSERT(
				std::numeric_limits<genFIType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT || std::numeric_limits<genFIType>::is_signed,
				"'abs' only accept floating-point and integer scalar or vector inputs");

			return x >= genFIType(0) ? x : -x;
			// TODO, perf comp with: *(((int *) &x) + 1) &= 0x7fffffff;
		}
	};

#if (GLM_COMPILER & GLM_COMPILER_CUDA) || (GLM_COMPILER & GLM_COMPILER_HIP)
	template<>
	struct compute_abs<float, true>
	{
		GLM_FUNC_QUALIFIER static float call(float x)
		{
			return fabsf(x);
		}
	};
#endif

	template<typename genFIType>
	struct compute_abs<genFIType, false>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static genFIType call(genFIType x)
		{
			GLM_STATIC_ASSERT(
				(!std::numeric_limits<genFIType>::is_signed && std::numeric_limits<genFIType>::is_integer),
				"'abs' only accept floating-point and integer scalar or vector inputs");
			return x;
		}
	};
}//namespace detail
}//namespace glm
