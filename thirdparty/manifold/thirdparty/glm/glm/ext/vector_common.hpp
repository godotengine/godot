/// @ref ext_vector_common
/// @file glm/ext/vector_common.hpp
///
/// @defgroup ext_vector_common GLM_EXT_vector_common
/// @ingroup ext
///
/// Exposes min and max functions for 3 to 4 vector parameters.
///
/// Include <glm/ext/vector_common.hpp> to use the features of this extension.
///
/// @see core_common
/// @see ext_scalar_common

#pragma once

// Dependency:
#include "../ext/scalar_common.hpp"
#include "../common.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_EXT_vector_common extension included")
#endif

namespace glm
{
	/// @addtogroup ext_vector_common
	/// @{

	/// Return the minimum component-wise values of 3 inputs
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> min(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c);

	/// Return the minimum component-wise values of 4 inputs
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> min(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c, vec<L, T, Q> const& d);

	/// Return the maximum component-wise values of 3 inputs
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> max(vec<L, T, Q> const& x, vec<L, T, Q> const& y, vec<L, T, Q> const& z);

	/// Return the maximum component-wise values of 4 inputs
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point or integer scalar types
	/// @tparam Q Value from qualifier enum
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR vec<L, T, Q> max( vec<L, T, Q> const& x, vec<L, T, Q> const& y, vec<L, T, Q> const& z, vec<L, T, Q> const& w);

	/// Returns y if y < x; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmin">std::fmin documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmin(vec<L, T, Q> const& x, T y);

	/// Returns y if y < x; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmin">std::fmin documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmin(vec<L, T, Q> const& x, vec<L, T, Q> const& y);

	/// Returns y if y < x; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmin">std::fmin documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmin(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c);

	/// Returns y if y < x; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmin">std::fmin documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmin(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c, vec<L, T, Q> const& d);

	/// Returns y if x < y; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmax">std::fmax documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmax(vec<L, T, Q> const& a, T b);

	/// Returns y if x < y; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmax">std::fmax documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmax(vec<L, T, Q> const& a, vec<L, T, Q> const& b);

	/// Returns y if x < y; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmax">std::fmax documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmax(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c);

	/// Returns y if x < y; otherwise, it returns x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see <a href="http://en.cppreference.com/w/cpp/numeric/math/fmax">std::fmax documentation</a>
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fmax(vec<L, T, Q> const& a, vec<L, T, Q> const& b, vec<L, T, Q> const& c, vec<L, T, Q> const& d);

	/// Returns min(max(x, minVal), maxVal) for each component in x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fclamp(vec<L, T, Q> const& x, T minVal, T maxVal);

	/// Returns min(max(x, minVal), maxVal) for each component in x. If one of the two arguments is NaN, the value of the other argument is returned.
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> fclamp(vec<L, T, Q> const& x, vec<L, T, Q> const& minVal, vec<L, T, Q> const& maxVal);

	/// Simulate GL_CLAMP OpenGL wrap mode
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> clamp(vec<L, T, Q> const& Texcoord);

	/// Simulate GL_REPEAT OpenGL wrap mode
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> repeat(vec<L, T, Q> const& Texcoord);

	/// Simulate GL_MIRRORED_REPEAT OpenGL wrap mode
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> mirrorClamp(vec<L, T, Q> const& Texcoord);

	/// Simulate GL_MIRROR_REPEAT OpenGL wrap mode
	///
	/// @tparam L Integer between 1 and 4 included that qualify the dimension of the vector
	/// @tparam T Floating-point scalar types
	/// @tparam Q Value from qualifier enum
	///
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, T, Q> mirrorRepeat(vec<L, T, Q> const& Texcoord);

	/// Returns a value equal to the nearest integer to x.
	/// The fraction 0.5 will round in a direction chosen by the
	/// implementation, presumably the direction that is fastest.
	///
	/// @param x The values of the argument must be greater or equal to zero.
	/// @tparam T floating point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/round.xml">GLSL round man page</a>
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, int, Q> iround(vec<L, T, Q> const& x);

	/// Returns a value equal to the nearest integer to x.
	/// The fraction 0.5 will round in a direction chosen by the
	/// implementation, presumably the direction that is fastest.
	///
	/// @param x The values of the argument must be greater or equal to zero.
	/// @tparam T floating point scalar types.
	///
	/// @see <a href="http://www.opengl.org/sdk/docs/manglsl/xhtml/round.xml">GLSL round man page</a>
	/// @see ext_vector_common extension.
	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_DECL vec<L, uint, Q> uround(vec<L, T, Q> const& x);

	/// @}
}//namespace glm

#include "vector_common.inl"
