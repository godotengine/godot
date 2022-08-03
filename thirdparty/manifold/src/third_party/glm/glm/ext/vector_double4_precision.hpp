/// @ref core
/// @file glm/ext/vector_double4_precision.hpp

#pragma once
#include "../detail/setup.hpp"
#include "../detail/type_vec4.hpp"

namespace glm
{
	/// @addtogroup core_vector_precision
	/// @{

	/// 4 components vector of high double-qualifier floating-point numbers.
	/// There is no guarantee on the actual qualifier.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.5 Vectors</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef vec<4, double, highp>		highp_dvec4;

	/// 4 components vector of medium double-qualifier floating-point numbers.
	/// There is no guarantee on the actual qualifier.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.5 Vectors</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef vec<4, double, mediump>		mediump_dvec4;

	/// 4 components vector of low double-qualifier floating-point numbers.
	/// There is no guarantee on the actual qualifier.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.5 Vectors</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef vec<4, double, lowp>		lowp_dvec4;

	/// @}
}//namespace glm
