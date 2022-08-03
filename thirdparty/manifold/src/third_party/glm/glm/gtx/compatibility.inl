#include <limits>

namespace glm
{
	// isfinite
	template<typename genType>
	GLM_FUNC_QUALIFIER bool isfinite(
		genType const& x)
	{
#		if GLM_HAS_CXX11_STL
			return std::isfinite(x) != 0;
#		elif GLM_COMPILER & GLM_COMPILER_VC
			return _finite(x) != 0;
#		elif GLM_COMPILER & GLM_COMPILER_GCC && GLM_PLATFORM & GLM_PLATFORM_ANDROID
			return _isfinite(x) != 0;
#		else
			if (std::numeric_limits<genType>::is_integer || std::denorm_absent == std::numeric_limits<genType>::has_denorm)
				return std::numeric_limits<genType>::min() <= x && std::numeric_limits<genType>::max() >= x;
			else
				return -std::numeric_limits<genType>::max() <= x && std::numeric_limits<genType>::max() >= x;
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<1, bool, Q> isfinite(
		vec<1, T, Q> const& x)
	{
		return vec<1, bool, Q>(
			isfinite(x.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<2, bool, Q> isfinite(
		vec<2, T, Q> const& x)
	{
		return vec<2, bool, Q>(
			isfinite(x.x),
			isfinite(x.y));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, bool, Q> isfinite(
		vec<3, T, Q> const& x)
	{
		return vec<3, bool, Q>(
			isfinite(x.x),
			isfinite(x.y),
			isfinite(x.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, bool, Q> isfinite(
		vec<4, T, Q> const& x)
	{
		return vec<4, bool, Q>(
			isfinite(x.x),
			isfinite(x.y),
			isfinite(x.z),
			isfinite(x.w));
	}

}//namespace glm
