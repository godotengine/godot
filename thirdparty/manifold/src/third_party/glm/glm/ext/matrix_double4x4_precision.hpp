/// @ref core
/// @file glm/ext/matrix_double4x4_precision.hpp

#pragma once
#include "../detail/type_mat4x4.hpp"

namespace glm
{
	/// @addtogroup core_matrix_precision
	/// @{

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, lowp>		lowp_dmat4;

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, mediump>	mediump_dmat4;

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, highp>	highp_dmat4;

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, lowp>		lowp_dmat4x4;

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, mediump>	mediump_dmat4x4;

	/// 4 columns of 4 components matrix of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	///
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.1.6 Matrices</a>
	/// @see <a href="http://www.opengl.org/registry/doc/GLSLangSpec.4.20.8.pdf">GLSL 4.20.8 specification, section 4.7.2 Precision Qualifier</a>
	typedef mat<4, 4, double, highp>	highp_dmat4x4;

	/// @}
}//namespace glm
