/// @ref ext_scalar_reciprocal

#include "../trigonometric.hpp"
#include <limits>

namespace glm
{
	// sec
	template<typename genType>
	GLM_FUNC_QUALIFIER genType sec(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'sec' only accept floating-point values");
		return genType(1) / glm::cos(angle);
	}

	// csc
	template<typename genType>
	GLM_FUNC_QUALIFIER genType csc(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'csc' only accept floating-point values");
		return genType(1) / glm::sin(angle);
	}

	// cot
	template<typename genType>
	GLM_FUNC_QUALIFIER genType cot(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'cot' only accept floating-point values");

		genType const pi_over_2 = genType(3.1415926535897932384626433832795 / 2.0);
		return glm::tan(pi_over_2 - angle);
	}

	// asec
	template<typename genType>
	GLM_FUNC_QUALIFIER genType asec(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'asec' only accept floating-point values");
		return acos(genType(1) / x);
	}

	// acsc
	template<typename genType>
	GLM_FUNC_QUALIFIER genType acsc(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'acsc' only accept floating-point values");
		return asin(genType(1) / x);
	}

	// acot
	template<typename genType>
	GLM_FUNC_QUALIFIER genType acot(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'acot' only accept floating-point values");

		genType const pi_over_2 = genType(3.1415926535897932384626433832795 / 2.0);
		return pi_over_2 - atan(x);
	}

	// sech
	template<typename genType>
	GLM_FUNC_QUALIFIER genType sech(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'sech' only accept floating-point values");
		return genType(1) / glm::cosh(angle);
	}

	// csch
	template<typename genType>
	GLM_FUNC_QUALIFIER genType csch(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'csch' only accept floating-point values");
		return genType(1) / glm::sinh(angle);
	}

	// coth
	template<typename genType>
	GLM_FUNC_QUALIFIER genType coth(genType angle)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'coth' only accept floating-point values");
		return glm::cosh(angle) / glm::sinh(angle);
	}

	// asech
	template<typename genType>
	GLM_FUNC_QUALIFIER genType asech(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'asech' only accept floating-point values");
		return acosh(genType(1) / x);
	}

	// acsch
	template<typename genType>
	GLM_FUNC_QUALIFIER genType acsch(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'acsch' only accept floating-point values");
		return asinh(genType(1) / x);
	}

	// acoth
	template<typename genType>
	GLM_FUNC_QUALIFIER genType acoth(genType x)
	{
		GLM_STATIC_ASSERT(std::numeric_limits<genType>::is_iec559 || GLM_CONFIG_UNRESTRICTED_FLOAT, "'acoth' only accept floating-point values");
		return atanh(genType(1) / x);
	}
}//namespace glm
