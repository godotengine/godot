/// @ref core
/// @file glm/exponential.hpp
///
/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
///
/// @defgroup core_func_exponential Exponential functions
/// @ingroup core
///
/// Provides GLSL exponential functions
///
/// These all operate component-wise. The description is per component.
///
/// Include <glm/exponential.hpp> to use these core features.

#pragma once

#include "detail/type_vec1.hpp"
#include "detail/type_vec2.hpp"
#include "detail/type_vec3.hpp"
#include "detail/type_vec4.hpp"
#include <cmath>

namespace glm
{
	/// @addtogroup core_func_exponential
	/// @{

	/// Returns 'base' raised to the power 'exponent'.
	///
	/// @param base Floating point value. pow function is defined for input values of 'base' defined in the range (inf-, inf+) in the limit of the type qualifier.
	/// @param exponent Floating point value representing the 'exponent'.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/pow.xml">GLSL pow man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> pow(vec<L, T, Q> const& base, vec<L, T, Q> const& exponent);

	/// Returns the natural exponentiation of x, i.e., e^x.
	///
	/// @param v exp function is defined for input values of v defined in the range (inf-, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/exp.xml">GLSL exp man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> exp(vec<L, T, Q> const& v);

	/// Returns the natural logarithm of v, i.e.,
	/// returns the value y which satisfies the equation x = e^y.
	/// Results are undefined if v <= 0.
	///
	/// @param v log function is defined for input values of v defined in the range (0, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/log.xml">GLSL log man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> log(vec<L, T, Q> const& v);

	/// Returns 2 raised to the v power.
	///
	/// @param v exp2 function is defined for input values of v defined in the range (inf-, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/exp2.xml">GLSL exp2 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> exp2(vec<L, T, Q> const& v);

	/// Returns the base 2 log of x, i.e., returns the value y,
	/// which satisfies the equation x = 2 ^ y.
	///
	/// @param v log2 function is defined for input values of v defined in the range (0, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/log2.xml">GLSL log2 man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> log2(vec<L, T, Q> const& v);

	/// Returns the positive square root of v.
	///
	/// @param v sqrt function is defined for input values of v defined in the range [0, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/sqrt.xml">GLSL sqrt man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> sqrt(vec<L, T, Q> const& v);

	/// Returns the reciprocal of the positive square root of v.
	///
	/// @param v inversesqrt function is defined for input values of v defined in the range [0, inf+) in the limit of the type qualifier.
	/// @tparam L An integer between 1 and 4 included that qualify the dimension of the vector.
	/// @tparam T Floating-point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/inversesqrt.xml">GLSL inversesqrt man page</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 8.2 Exponential Functions</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> inversesqrt(vec<L, T, Q> const& v);

	/// @}
}//namespace glm

#include "detail/func_exponential.inl"
