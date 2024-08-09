/// @ref gtc_type_aligned
/// @file glm/gtc/type_aligned.hpp
///
/// @see core (dependence)
///
/// @defgroup gtc_type_aligned GLM_GTC_type_aligned
/// @ingroup gtc
///
/// Include <glm/gtc/type_aligned.hpp> to use the features of this extension.
///
/// Aligned types allowing SIMD optimizations of vectors and matrices types

#pragma once

#if (GLM_CONFIG_ALIGNED_GENTYPES == GLM_DISABLE)
#	error "GLM: Aligned gentypes require to enable C++ language extensions. Define GLM_FORCE_ALIGNED_GENTYPES before including GLM headers to use aligned types."
#endif

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
# pragma message("GLM: GLM_GTC_type_aligned extension included")
#endif

#include "../mat4x4.hpp"
#include "../mat4x3.hpp"
#include "../mat4x2.hpp"
#include "../mat3x4.hpp"
#include "../mat3x3.hpp"
#include "../mat3x2.hpp"
#include "../mat2x4.hpp"
#include "../mat2x3.hpp"
#include "../mat2x2.hpp"
#include "../gtc/vec1.hpp"
#include "../vec2.hpp"
#include "../vec3.hpp"
#include "../vec4.hpp"

namespace glm
{
	/// @addtogroup gtc_type_aligned
	/// @{

	// -- *vec1 --

	/// 1 component vector aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, float, aligned_highp>	aligned_highp_vec1;

	/// 1 component vector aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, float, aligned_mediump>	aligned_mediump_vec1;

	/// 1 component vector aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, float, aligned_lowp>		aligned_lowp_vec1;

	/// 1 component vector aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, double, aligned_highp>	aligned_highp_dvec1;

	/// 1 component vector aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, double, aligned_mediump>	aligned_mediump_dvec1;

	/// 1 component vector aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, double, aligned_lowp>	aligned_lowp_dvec1;

	/// 1 component vector aligned in memory of signed integer numbers.
	typedef vec<1, int, aligned_highp>		aligned_highp_ivec1;

	/// 1 component vector aligned in memory of signed integer numbers.
	typedef vec<1, int, aligned_mediump>	aligned_mediump_ivec1;

	/// 1 component vector aligned in memory of signed integer numbers.
	typedef vec<1, int, aligned_lowp>		aligned_lowp_ivec1;

	/// 1 component vector aligned in memory of unsigned integer numbers.
	typedef vec<1, uint, aligned_highp>		aligned_highp_uvec1;

	/// 1 component vector aligned in memory of unsigned integer numbers.
	typedef vec<1, uint, aligned_mediump>	aligned_mediump_uvec1;

	/// 1 component vector aligned in memory of unsigned integer numbers.
	typedef vec<1, uint, aligned_lowp>		aligned_lowp_uvec1;

	/// 1 component vector aligned in memory of bool values.
	typedef vec<1, bool, aligned_highp>		aligned_highp_bvec1;

	/// 1 component vector aligned in memory of bool values.
	typedef vec<1, bool, aligned_mediump>	aligned_mediump_bvec1;

	/// 1 component vector aligned in memory of bool values.
	typedef vec<1, bool, aligned_lowp>		aligned_lowp_bvec1;

	/// 1 component vector tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, float, packed_highp>		packed_highp_vec1;

	/// 1 component vector tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, float, packed_mediump>	packed_mediump_vec1;

	/// 1 component vector tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, float, packed_lowp>		packed_lowp_vec1;

	/// 1 component vector tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<1, double, packed_highp>	packed_highp_dvec1;

	/// 1 component vector tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<1, double, packed_mediump>	packed_mediump_dvec1;

	/// 1 component vector tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<1, double, packed_lowp>		packed_lowp_dvec1;

	/// 1 component vector tightly packed in memory of signed integer numbers.
	typedef vec<1, int, packed_highp>		packed_highp_ivec1;

	/// 1 component vector tightly packed in memory of signed integer numbers.
	typedef vec<1, int, packed_mediump>		packed_mediump_ivec1;

	/// 1 component vector tightly packed in memory of signed integer numbers.
	typedef vec<1, int, packed_lowp>		packed_lowp_ivec1;

	/// 1 component vector tightly packed in memory of unsigned integer numbers.
	typedef vec<1, uint, packed_highp>		packed_highp_uvec1;

	/// 1 component vector tightly packed in memory of unsigned integer numbers.
	typedef vec<1, uint, packed_mediump>	packed_mediump_uvec1;

	/// 1 component vector tightly packed in memory of unsigned integer numbers.
	typedef vec<1, uint, packed_lowp>		packed_lowp_uvec1;

	/// 1 component vector tightly packed in memory of bool values.
	typedef vec<1, bool, packed_highp>		packed_highp_bvec1;

	/// 1 component vector tightly packed in memory of bool values.
	typedef vec<1, bool, packed_mediump>	packed_mediump_bvec1;

	/// 1 component vector tightly packed in memory of bool values.
	typedef vec<1, bool, packed_lowp>		packed_lowp_bvec1;

	// -- *vec2 --

	/// 2 components vector aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<2, float, aligned_highp>	aligned_highp_vec2;

	/// 2 components vector aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<2, float, aligned_mediump>	aligned_mediump_vec2;

	/// 2 components vector aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<2, float, aligned_lowp>		aligned_lowp_vec2;

	/// 2 components vector aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<2, double, aligned_highp>	aligned_highp_dvec2;

	/// 2 components vector aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<2, double, aligned_mediump>	aligned_mediump_dvec2;

	/// 2 components vector aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<2, double, aligned_lowp>	aligned_lowp_dvec2;

	/// 2 components vector aligned in memory of signed integer numbers.
	typedef vec<2, int, aligned_highp>		aligned_highp_ivec2;

	/// 2 components vector aligned in memory of signed integer numbers.
	typedef vec<2, int, aligned_mediump>	aligned_mediump_ivec2;

	/// 2 components vector aligned in memory of signed integer numbers.
	typedef vec<2, int, aligned_lowp>		aligned_lowp_ivec2;

	/// 2 components vector aligned in memory of unsigned integer numbers.
	typedef vec<2, uint, aligned_highp>		aligned_highp_uvec2;

	/// 2 components vector aligned in memory of unsigned integer numbers.
	typedef vec<2, uint, aligned_mediump>	aligned_mediump_uvec2;

	/// 2 components vector aligned in memory of unsigned integer numbers.
	typedef vec<2, uint, aligned_lowp>		aligned_lowp_uvec2;

	/// 2 components vector aligned in memory of bool values.
	typedef vec<2, bool, aligned_highp>		aligned_highp_bvec2;

	/// 2 components vector aligned in memory of bool values.
	typedef vec<2, bool, aligned_mediump>	aligned_mediump_bvec2;

	/// 2 components vector aligned in memory of bool values.
	typedef vec<2, bool, aligned_lowp>		aligned_lowp_bvec2;

	/// 2 components vector tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<2, float, packed_highp>		packed_highp_vec2;

	/// 2 components vector tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<2, float, packed_mediump>	packed_mediump_vec2;

	/// 2 components vector tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<2, float, packed_lowp>		packed_lowp_vec2;

	/// 2 components vector tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<2, double, packed_highp>	packed_highp_dvec2;

	/// 2 components vector tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<2, double, packed_mediump>	packed_mediump_dvec2;

	/// 2 components vector tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<2, double, packed_lowp>		packed_lowp_dvec2;

	/// 2 components vector tightly packed in memory of signed integer numbers.
	typedef vec<2, int, packed_highp>		packed_highp_ivec2;

	/// 2 components vector tightly packed in memory of signed integer numbers.
	typedef vec<2, int, packed_mediump>		packed_mediump_ivec2;

	/// 2 components vector tightly packed in memory of signed integer numbers.
	typedef vec<2, int, packed_lowp>		packed_lowp_ivec2;

	/// 2 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<2, uint, packed_highp>		packed_highp_uvec2;

	/// 2 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<2, uint, packed_mediump>	packed_mediump_uvec2;

	/// 2 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<2, uint, packed_lowp>		packed_lowp_uvec2;

	/// 2 components vector tightly packed in memory of bool values.
	typedef vec<2, bool, packed_highp>		packed_highp_bvec2;

	/// 2 components vector tightly packed in memory of bool values.
	typedef vec<2, bool, packed_mediump>	packed_mediump_bvec2;

	/// 2 components vector tightly packed in memory of bool values.
	typedef vec<2, bool, packed_lowp>		packed_lowp_bvec2;

	// -- *vec3 --

	/// 3 components vector aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<3, float, aligned_highp>	aligned_highp_vec3;

	/// 3 components vector aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<3, float, aligned_mediump>	aligned_mediump_vec3;

	/// 3 components vector aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<3, float, aligned_lowp>		aligned_lowp_vec3;

	/// 3 components vector aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<3, double, aligned_highp>	aligned_highp_dvec3;

	/// 3 components vector aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<3, double, aligned_mediump>	aligned_mediump_dvec3;

	/// 3 components vector aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<3, double, aligned_lowp>	aligned_lowp_dvec3;

	/// 3 components vector aligned in memory of signed integer numbers.
	typedef vec<3, int, aligned_highp>		aligned_highp_ivec3;

	/// 3 components vector aligned in memory of signed integer numbers.
	typedef vec<3, int, aligned_mediump>	aligned_mediump_ivec3;

	/// 3 components vector aligned in memory of signed integer numbers.
	typedef vec<3, int, aligned_lowp>		aligned_lowp_ivec3;

	/// 3 components vector aligned in memory of unsigned integer numbers.
	typedef vec<3, uint, aligned_highp>		aligned_highp_uvec3;

	/// 3 components vector aligned in memory of unsigned integer numbers.
	typedef vec<3, uint, aligned_mediump>	aligned_mediump_uvec3;

	/// 3 components vector aligned in memory of unsigned integer numbers.
	typedef vec<3, uint, aligned_lowp>		aligned_lowp_uvec3;

	/// 3 components vector aligned in memory of bool values.
	typedef vec<3, bool, aligned_highp>		aligned_highp_bvec3;

	/// 3 components vector aligned in memory of bool values.
	typedef vec<3, bool, aligned_mediump>	aligned_mediump_bvec3;

	/// 3 components vector aligned in memory of bool values.
	typedef vec<3, bool, aligned_lowp>		aligned_lowp_bvec3;

	/// 3 components vector tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<3, float, packed_highp>		packed_highp_vec3;

	/// 3 components vector tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<3, float, packed_mediump>	packed_mediump_vec3;

	/// 3 components vector tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<3, float, packed_lowp>		packed_lowp_vec3;

	/// 3 components vector tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<3, double, packed_highp>	packed_highp_dvec3;

	/// 3 components vector tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<3, double, packed_mediump>	packed_mediump_dvec3;

	/// 3 components vector tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<3, double, packed_lowp>		packed_lowp_dvec3;

	/// 3 components vector tightly packed in memory of signed integer numbers.
	typedef vec<3, int, packed_highp>		packed_highp_ivec3;

	/// 3 components vector tightly packed in memory of signed integer numbers.
	typedef vec<3, int, packed_mediump>		packed_mediump_ivec3;

	/// 3 components vector tightly packed in memory of signed integer numbers.
	typedef vec<3, int, packed_lowp>		packed_lowp_ivec3;

	/// 3 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<3, uint, packed_highp>		packed_highp_uvec3;

	/// 3 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<3, uint, packed_mediump>	packed_mediump_uvec3;

	/// 3 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<3, uint, packed_lowp>		packed_lowp_uvec3;

	/// 3 components vector tightly packed in memory of bool values.
	typedef vec<3, bool, packed_highp>		packed_highp_bvec3;

	/// 3 components vector tightly packed in memory of bool values.
	typedef vec<3, bool, packed_mediump>	packed_mediump_bvec3;

	/// 3 components vector tightly packed in memory of bool values.
	typedef vec<3, bool, packed_lowp>		packed_lowp_bvec3;

	// -- *vec4 --

	/// 4 components vector aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<4, float, aligned_highp>	aligned_highp_vec4;

	/// 4 components vector aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<4, float, aligned_mediump>	aligned_mediump_vec4;

	/// 4 components vector aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<4, float, aligned_lowp>		aligned_lowp_vec4;

	/// 4 components vector aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<4, double, aligned_highp>	aligned_highp_dvec4;

	/// 4 components vector aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<4, double, aligned_mediump>	aligned_mediump_dvec4;

	/// 4 components vector aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<4, double, aligned_lowp>	aligned_lowp_dvec4;

	/// 4 components vector aligned in memory of signed integer numbers.
	typedef vec<4, int, aligned_highp>		aligned_highp_ivec4;

	/// 4 components vector aligned in memory of signed integer numbers.
	typedef vec<4, int, aligned_mediump>	aligned_mediump_ivec4;

	/// 4 components vector aligned in memory of signed integer numbers.
	typedef vec<4, int, aligned_lowp>		aligned_lowp_ivec4;

	/// 4 components vector aligned in memory of unsigned integer numbers.
	typedef vec<4, uint, aligned_highp>		aligned_highp_uvec4;

	/// 4 components vector aligned in memory of unsigned integer numbers.
	typedef vec<4, uint, aligned_mediump>	aligned_mediump_uvec4;

	/// 4 components vector aligned in memory of unsigned integer numbers.
	typedef vec<4, uint, aligned_lowp>		aligned_lowp_uvec4;

	/// 4 components vector aligned in memory of bool values.
	typedef vec<4, bool, aligned_highp>		aligned_highp_bvec4;

	/// 4 components vector aligned in memory of bool values.
	typedef vec<4, bool, aligned_mediump>	aligned_mediump_bvec4;

	/// 4 components vector aligned in memory of bool values.
	typedef vec<4, bool, aligned_lowp>		aligned_lowp_bvec4;

	/// 4 components vector tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<4, float, packed_highp>		packed_highp_vec4;

	/// 4 components vector tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<4, float, packed_mediump>	packed_mediump_vec4;

	/// 4 components vector tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<4, float, packed_lowp>		packed_lowp_vec4;

	/// 4 components vector tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef vec<4, double, packed_highp>	packed_highp_dvec4;

	/// 4 components vector tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef vec<4, double, packed_mediump>	packed_mediump_dvec4;

	/// 4 components vector tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef vec<4, double, packed_lowp>		packed_lowp_dvec4;

	/// 4 components vector tightly packed in memory of signed integer numbers.
	typedef vec<4, int, packed_highp>		packed_highp_ivec4;

	/// 4 components vector tightly packed in memory of signed integer numbers.
	typedef vec<4, int, packed_mediump>		packed_mediump_ivec4;

	/// 4 components vector tightly packed in memory of signed integer numbers.
	typedef vec<4, int, packed_lowp>		packed_lowp_ivec4;

	/// 4 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<4, uint, packed_highp>		packed_highp_uvec4;

	/// 4 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<4, uint, packed_mediump>	packed_mediump_uvec4;

	/// 4 components vector tightly packed in memory of unsigned integer numbers.
	typedef vec<4, uint, packed_lowp>		packed_lowp_uvec4;

	/// 4 components vector tightly packed in memory of bool values.
	typedef vec<4, bool, packed_highp>		packed_highp_bvec4;

	/// 4 components vector tightly packed in memory of bool values.
	typedef vec<4, bool, packed_mediump>	packed_mediump_bvec4;

	/// 4 components vector tightly packed in memory of bool values.
	typedef vec<4, bool, packed_lowp>		packed_lowp_bvec4;

	// -- *mat2 --

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_highp>		aligned_highp_mat2;

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_mediump>	aligned_mediump_mat2;

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_lowp>		aligned_lowp_mat2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_highp>	aligned_highp_dmat2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_mediump>	aligned_mediump_dmat2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_lowp>		aligned_lowp_dmat2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_highp>		packed_highp_mat2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_mediump>	packed_mediump_mat2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_lowp>		packed_lowp_mat2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_highp>		packed_highp_dmat2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_mediump>	packed_mediump_dmat2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_lowp>		packed_lowp_dmat2;

	// -- *mat3 --

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_highp>		aligned_highp_mat3;

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_mediump>	aligned_mediump_mat3;

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_lowp>		aligned_lowp_mat3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_highp>	aligned_highp_dmat3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_mediump>	aligned_mediump_dmat3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_lowp>		aligned_lowp_dmat3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_highp>		packed_highp_mat3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_mediump>	packed_mediump_mat3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_lowp>		packed_lowp_mat3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_highp>		packed_highp_dmat3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_mediump>	packed_mediump_dmat3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_lowp>		packed_lowp_dmat3;

	// -- *mat4 --

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_highp>		aligned_highp_mat4;

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_mediump>	aligned_mediump_mat4;

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_lowp>		aligned_lowp_mat4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_highp>	aligned_highp_dmat4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_mediump>	aligned_mediump_dmat4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_lowp>		aligned_lowp_dmat4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_highp>		packed_highp_mat4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_mediump>	packed_mediump_mat4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_lowp>		packed_lowp_mat4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_highp>		packed_highp_dmat4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_mediump>	packed_mediump_dmat4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_lowp>		packed_lowp_dmat4;

	// -- *mat2x2 --

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_highp>		aligned_highp_mat2x2;

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_mediump>	aligned_mediump_mat2x2;

	/// 2 by 2 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, aligned_lowp>		aligned_lowp_mat2x2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_highp>	aligned_highp_dmat2x2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_mediump>	aligned_mediump_dmat2x2;

	/// 2 by 2 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, aligned_lowp>		aligned_lowp_dmat2x2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_highp>		packed_highp_mat2x2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_mediump>	packed_mediump_mat2x2;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, float, packed_lowp>		packed_lowp_mat2x2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_highp>		packed_highp_dmat2x2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_mediump>	packed_mediump_dmat2x2;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 2, double, packed_lowp>		packed_lowp_dmat2x2;

	// -- *mat2x3 --

	/// 2 by 3 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, aligned_highp>		aligned_highp_mat2x3;

	/// 2 by 3 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, aligned_mediump>	aligned_mediump_mat2x3;

	/// 2 by 3 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, aligned_lowp>		aligned_lowp_mat2x3;

	/// 2 by 3 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, aligned_highp>	aligned_highp_dmat2x3;

	/// 2 by 3 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, aligned_mediump>	aligned_mediump_dmat2x3;

	/// 2 by 3 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, aligned_lowp>		aligned_lowp_dmat2x3;

	/// 2 by 3 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, packed_highp>		packed_highp_mat2x3;

	/// 2 by 3 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, packed_mediump>	packed_mediump_mat2x3;

	/// 2 by 3 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 3, float, packed_lowp>		packed_lowp_mat2x3;

	/// 2 by 3 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, packed_highp>		packed_highp_dmat2x3;

	/// 2 by 3 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, packed_mediump>	packed_mediump_dmat2x3;

	/// 2 by 3 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 3, double, packed_lowp>		packed_lowp_dmat2x3;

	// -- *mat2x4 --

	/// 2 by 4 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, aligned_highp>		aligned_highp_mat2x4;

	/// 2 by 4 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, aligned_mediump>	aligned_mediump_mat2x4;

	/// 2 by 4 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, aligned_lowp>		aligned_lowp_mat2x4;

	/// 2 by 4 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, aligned_highp>	aligned_highp_dmat2x4;

	/// 2 by 4 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, aligned_mediump>	aligned_mediump_dmat2x4;

	/// 2 by 4 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, aligned_lowp>		aligned_lowp_dmat2x4;

	/// 2 by 4 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, packed_highp>		packed_highp_mat2x4;

	/// 2 by 4 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, packed_mediump>	packed_mediump_mat2x4;

	/// 2 by 4 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 4, float, packed_lowp>		packed_lowp_mat2x4;

	/// 2 by 4 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, packed_highp>		packed_highp_dmat2x4;

	/// 2 by 4 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, packed_mediump>	packed_mediump_dmat2x4;

	/// 2 by 4 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<2, 4, double, packed_lowp>		packed_lowp_dmat2x4;

	// -- *mat3x2 --

	/// 3 by 2 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, aligned_highp>		aligned_highp_mat3x2;

	/// 3 by 2 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, aligned_mediump>	aligned_mediump_mat3x2;

	/// 3 by 2 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, aligned_lowp>		aligned_lowp_mat3x2;

	/// 3 by 2 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, aligned_highp>	aligned_highp_dmat3x2;

	/// 3 by 2 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, aligned_mediump>	aligned_mediump_dmat3x2;

	/// 3 by 2 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, aligned_lowp>		aligned_lowp_dmat3x2;

	/// 3 by 2 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, packed_highp>		packed_highp_mat3x2;

	/// 3 by 2 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, packed_mediump>	packed_mediump_mat3x2;

	/// 3 by 2 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 2, float, packed_lowp>		packed_lowp_mat3x2;

	/// 3 by 2 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, packed_highp>		packed_highp_dmat3x2;

	/// 3 by 2 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, packed_mediump>	packed_mediump_dmat3x2;

	/// 3 by 2 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 2, double, packed_lowp>		packed_lowp_dmat3x2;

	// -- *mat3x3 --

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_highp>		aligned_highp_mat3x3;

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_mediump>	aligned_mediump_mat3x3;

	/// 3 by 3 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, aligned_lowp>		aligned_lowp_mat3x3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_highp>	aligned_highp_dmat3x3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_mediump>	aligned_mediump_dmat3x3;

	/// 3 by 3 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, aligned_lowp>		aligned_lowp_dmat3x3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_highp>		packed_highp_mat3x3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_mediump>	packed_mediump_mat3x3;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, float, packed_lowp>		packed_lowp_mat3x3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_highp>		packed_highp_dmat3x3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_mediump>	packed_mediump_dmat3x3;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 3, double, packed_lowp>		packed_lowp_dmat3x3;

	// -- *mat3x4 --

	/// 3 by 4 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, aligned_highp>		aligned_highp_mat3x4;

	/// 3 by 4 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, aligned_mediump>	aligned_mediump_mat3x4;

	/// 3 by 4 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, aligned_lowp>		aligned_lowp_mat3x4;

	/// 3 by 4 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, aligned_highp>	aligned_highp_dmat3x4;

	/// 3 by 4 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, aligned_mediump>	aligned_mediump_dmat3x4;

	/// 3 by 4 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, aligned_lowp>		aligned_lowp_dmat3x4;

	/// 3 by 4 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, packed_highp>		packed_highp_mat3x4;

	/// 3 by 4 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, packed_mediump>	packed_mediump_mat3x4;

	/// 3 by 4 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 4, float, packed_lowp>		packed_lowp_mat3x4;

	/// 3 by 4 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, packed_highp>		packed_highp_dmat3x4;

	/// 3 by 4 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, packed_mediump>	packed_mediump_dmat3x4;

	/// 3 by 4 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<3, 4, double, packed_lowp>		packed_lowp_dmat3x4;

	// -- *mat4x2 --

	/// 4 by 2 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, aligned_highp>		aligned_highp_mat4x2;

	/// 4 by 2 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, aligned_mediump>	aligned_mediump_mat4x2;

	/// 4 by 2 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, aligned_lowp>		aligned_lowp_mat4x2;

	/// 4 by 2 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, aligned_highp>	aligned_highp_dmat4x2;

	/// 4 by 2 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, aligned_mediump>	aligned_mediump_dmat4x2;

	/// 4 by 2 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, aligned_lowp>		aligned_lowp_dmat4x2;

	/// 4 by 2 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, packed_highp>		packed_highp_mat4x2;

	/// 4 by 2 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, packed_mediump>	packed_mediump_mat4x2;

	/// 4 by 2 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 2, float, packed_lowp>		packed_lowp_mat4x2;

	/// 4 by 2 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, packed_highp>		packed_highp_dmat4x2;

	/// 4 by 2 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, packed_mediump>	packed_mediump_dmat4x2;

	/// 4 by 2 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 2, double, packed_lowp>		packed_lowp_dmat4x2;

	// -- *mat4x3 --

	/// 4 by 3 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, aligned_highp>		aligned_highp_mat4x3;

	/// 4 by 3 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, aligned_mediump>	aligned_mediump_mat4x3;

	/// 4 by 3 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, aligned_lowp>		aligned_lowp_mat4x3;

	/// 4 by 3 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, aligned_highp>	aligned_highp_dmat4x3;

	/// 4 by 3 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, aligned_mediump>	aligned_mediump_dmat4x3;

	/// 4 by 3 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, aligned_lowp>		aligned_lowp_dmat4x3;

	/// 4 by 3 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, packed_highp>		packed_highp_mat4x3;

	/// 4 by 3 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, packed_mediump>	packed_mediump_mat4x3;

	/// 4 by 3 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 3, float, packed_lowp>		packed_lowp_mat4x3;

	/// 4 by 3 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, packed_highp>		packed_highp_dmat4x3;

	/// 4 by 3 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, packed_mediump>	packed_mediump_dmat4x3;

	/// 4 by 3 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 3, double, packed_lowp>		packed_lowp_dmat4x3;

	// -- *mat4x4 --

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_highp>		aligned_highp_mat4x4;

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_mediump>	aligned_mediump_mat4x4;

	/// 4 by 4 matrix aligned in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, aligned_lowp>		aligned_lowp_mat4x4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_highp>	aligned_highp_dmat4x4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_mediump>	aligned_mediump_dmat4x4;

	/// 4 by 4 matrix aligned in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, aligned_lowp>		aligned_lowp_dmat4x4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_highp>		packed_highp_mat4x4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_mediump>	packed_mediump_mat4x4;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, float, packed_lowp>		packed_lowp_mat4x4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using high precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_highp>		packed_highp_dmat4x4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using medium precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_mediump>	packed_mediump_dmat4x4;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers using low precision arithmetic in term of ULPs.
	typedef mat<4, 4, double, packed_lowp>		packed_lowp_dmat4x4;

	// -- default --

#if(defined(GLM_PRECISION_LOWP_FLOAT))
	typedef aligned_lowp_vec1			aligned_vec1;
	typedef aligned_lowp_vec2			aligned_vec2;
	typedef aligned_lowp_vec3			aligned_vec3;
	typedef aligned_lowp_vec4			aligned_vec4;
	typedef packed_lowp_vec1			packed_vec1;
	typedef packed_lowp_vec2			packed_vec2;
	typedef packed_lowp_vec3			packed_vec3;
	typedef packed_lowp_vec4			packed_vec4;

	typedef aligned_lowp_mat2			aligned_mat2;
	typedef aligned_lowp_mat3			aligned_mat3;
	typedef aligned_lowp_mat4			aligned_mat4;
	typedef packed_lowp_mat2			packed_mat2;
	typedef packed_lowp_mat3			packed_mat3;
	typedef packed_lowp_mat4			packed_mat4;

	typedef aligned_lowp_mat2x2			aligned_mat2x2;
	typedef aligned_lowp_mat2x3			aligned_mat2x3;
	typedef aligned_lowp_mat2x4			aligned_mat2x4;
	typedef aligned_lowp_mat3x2			aligned_mat3x2;
	typedef aligned_lowp_mat3x3			aligned_mat3x3;
	typedef aligned_lowp_mat3x4			aligned_mat3x4;
	typedef aligned_lowp_mat4x2			aligned_mat4x2;
	typedef aligned_lowp_mat4x3			aligned_mat4x3;
	typedef aligned_lowp_mat4x4			aligned_mat4x4;
	typedef packed_lowp_mat2x2			packed_mat2x2;
	typedef packed_lowp_mat2x3			packed_mat2x3;
	typedef packed_lowp_mat2x4			packed_mat2x4;
	typedef packed_lowp_mat3x2			packed_mat3x2;
	typedef packed_lowp_mat3x3			packed_mat3x3;
	typedef packed_lowp_mat3x4			packed_mat3x4;
	typedef packed_lowp_mat4x2			packed_mat4x2;
	typedef packed_lowp_mat4x3			packed_mat4x3;
	typedef packed_lowp_mat4x4			packed_mat4x4;
#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
	typedef aligned_mediump_vec1		aligned_vec1;
	typedef aligned_mediump_vec2		aligned_vec2;
	typedef aligned_mediump_vec3		aligned_vec3;
	typedef aligned_mediump_vec4		aligned_vec4;
	typedef packed_mediump_vec1			packed_vec1;
	typedef packed_mediump_vec2			packed_vec2;
	typedef packed_mediump_vec3			packed_vec3;
	typedef packed_mediump_vec4			packed_vec4;

	typedef aligned_mediump_mat2		aligned_mat2;
	typedef aligned_mediump_mat3		aligned_mat3;
	typedef aligned_mediump_mat4		aligned_mat4;
	typedef packed_mediump_mat2			packed_mat2;
	typedef packed_mediump_mat3			packed_mat3;
	typedef packed_mediump_mat4			packed_mat4;

	typedef aligned_mediump_mat2x2		aligned_mat2x2;
	typedef aligned_mediump_mat2x3		aligned_mat2x3;
	typedef aligned_mediump_mat2x4		aligned_mat2x4;
	typedef aligned_mediump_mat3x2		aligned_mat3x2;
	typedef aligned_mediump_mat3x3		aligned_mat3x3;
	typedef aligned_mediump_mat3x4		aligned_mat3x4;
	typedef aligned_mediump_mat4x2		aligned_mat4x2;
	typedef aligned_mediump_mat4x3		aligned_mat4x3;
	typedef aligned_mediump_mat4x4		aligned_mat4x4;
	typedef packed_mediump_mat2x2		packed_mat2x2;
	typedef packed_mediump_mat2x3		packed_mat2x3;
	typedef packed_mediump_mat2x4		packed_mat2x4;
	typedef packed_mediump_mat3x2		packed_mat3x2;
	typedef packed_mediump_mat3x3		packed_mat3x3;
	typedef packed_mediump_mat3x4		packed_mat3x4;
	typedef packed_mediump_mat4x2		packed_mat4x2;
	typedef packed_mediump_mat4x3		packed_mat4x3;
	typedef packed_mediump_mat4x4		packed_mat4x4;
#else //defined(GLM_PRECISION_HIGHP_FLOAT)
	/// 1 component vector aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_vec1			aligned_vec1;

	/// 2 components vector aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_vec2			aligned_vec2;

	/// 3 components vector aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_vec3			aligned_vec3;

	/// 4 components vector aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_vec4 			aligned_vec4;

	/// 1 component vector tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_vec1			packed_vec1;

	/// 2 components vector tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_vec2			packed_vec2;

	/// 3 components vector tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_vec3			packed_vec3;

	/// 4 components vector tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_vec4			packed_vec4;

	/// 2 by 2 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat2			aligned_mat2;

	/// 3 by 3 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat3			aligned_mat3;

	/// 4 by 4 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat4			aligned_mat4;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat2			packed_mat2;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat3			packed_mat3;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat4			packed_mat4;

	/// 2 by 2 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat2x2		aligned_mat2x2;

	/// 2 by 3 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat2x3		aligned_mat2x3;

	/// 2 by 4 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat2x4		aligned_mat2x4;

	/// 3 by 2 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat3x2		aligned_mat3x2;

	/// 3 by 3 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat3x3		aligned_mat3x3;

	/// 3 by 4 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat3x4		aligned_mat3x4;

	/// 4 by 2 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat4x2		aligned_mat4x2;

	/// 4 by 3 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat4x3		aligned_mat4x3;

	/// 4 by 4 matrix tightly aligned in memory of single-precision floating-point numbers.
	typedef aligned_highp_mat4x4		aligned_mat4x4;

	/// 2 by 2 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat2x2			packed_mat2x2;

	/// 2 by 3 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat2x3			packed_mat2x3;

	/// 2 by 4 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat2x4			packed_mat2x4;

	/// 3 by 2 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat3x2			packed_mat3x2;

	/// 3 by 3 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat3x3			packed_mat3x3;

	/// 3 by 4 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat3x4			packed_mat3x4;

	/// 4 by 2 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat4x2			packed_mat4x2;

	/// 4 by 3 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat4x3			packed_mat4x3;

	/// 4 by 4 matrix tightly packed in memory of single-precision floating-point numbers.
	typedef packed_highp_mat4x4			packed_mat4x4;
#endif//GLM_PRECISION

#if(defined(GLM_PRECISION_LOWP_DOUBLE))
	typedef aligned_lowp_dvec1			aligned_dvec1;
	typedef aligned_lowp_dvec2			aligned_dvec2;
	typedef aligned_lowp_dvec3			aligned_dvec3;
	typedef aligned_lowp_dvec4			aligned_dvec4;
	typedef packed_lowp_dvec1			packed_dvec1;
	typedef packed_lowp_dvec2			packed_dvec2;
	typedef packed_lowp_dvec3			packed_dvec3;
	typedef packed_lowp_dvec4			packed_dvec4;

	typedef aligned_lowp_dmat2			aligned_dmat2;
	typedef aligned_lowp_dmat3			aligned_dmat3;
	typedef aligned_lowp_dmat4			aligned_dmat4;
	typedef packed_lowp_dmat2			packed_dmat2;
	typedef packed_lowp_dmat3			packed_dmat3;
	typedef packed_lowp_dmat4			packed_dmat4;

	typedef aligned_lowp_dmat2x2		aligned_dmat2x2;
	typedef aligned_lowp_dmat2x3		aligned_dmat2x3;
	typedef aligned_lowp_dmat2x4		aligned_dmat2x4;
	typedef aligned_lowp_dmat3x2		aligned_dmat3x2;
	typedef aligned_lowp_dmat3x3		aligned_dmat3x3;
	typedef aligned_lowp_dmat3x4		aligned_dmat3x4;
	typedef aligned_lowp_dmat4x2		aligned_dmat4x2;
	typedef aligned_lowp_dmat4x3		aligned_dmat4x3;
	typedef aligned_lowp_dmat4x4		aligned_dmat4x4;
	typedef packed_lowp_dmat2x2			packed_dmat2x2;
	typedef packed_lowp_dmat2x3			packed_dmat2x3;
	typedef packed_lowp_dmat2x4			packed_dmat2x4;
	typedef packed_lowp_dmat3x2			packed_dmat3x2;
	typedef packed_lowp_dmat3x3			packed_dmat3x3;
	typedef packed_lowp_dmat3x4			packed_dmat3x4;
	typedef packed_lowp_dmat4x2			packed_dmat4x2;
	typedef packed_lowp_dmat4x3			packed_dmat4x3;
	typedef packed_lowp_dmat4x4			packed_dmat4x4;
#elif(defined(GLM_PRECISION_MEDIUMP_DOUBLE))
	typedef aligned_mediump_dvec1		aligned_dvec1;
	typedef aligned_mediump_dvec2		aligned_dvec2;
	typedef aligned_mediump_dvec3		aligned_dvec3;
	typedef aligned_mediump_dvec4		aligned_dvec4;
	typedef packed_mediump_dvec1		packed_dvec1;
	typedef packed_mediump_dvec2		packed_dvec2;
	typedef packed_mediump_dvec3		packed_dvec3;
	typedef packed_mediump_dvec4		packed_dvec4;

	typedef aligned_mediump_dmat2		aligned_dmat2;
	typedef aligned_mediump_dmat3		aligned_dmat3;
	typedef aligned_mediump_dmat4		aligned_dmat4;
	typedef packed_mediump_dmat2		packed_dmat2;
	typedef packed_mediump_dmat3		packed_dmat3;
	typedef packed_mediump_dmat4		packed_dmat4;

	typedef aligned_mediump_dmat2x2		aligned_dmat2x2;
	typedef aligned_mediump_dmat2x3		aligned_dmat2x3;
	typedef aligned_mediump_dmat2x4		aligned_dmat2x4;
	typedef aligned_mediump_dmat3x2		aligned_dmat3x2;
	typedef aligned_mediump_dmat3x3		aligned_dmat3x3;
	typedef aligned_mediump_dmat3x4		aligned_dmat3x4;
	typedef aligned_mediump_dmat4x2		aligned_dmat4x2;
	typedef aligned_mediump_dmat4x3		aligned_dmat4x3;
	typedef aligned_mediump_dmat4x4		aligned_dmat4x4;
	typedef packed_mediump_dmat2x2		packed_dmat2x2;
	typedef packed_mediump_dmat2x3		packed_dmat2x3;
	typedef packed_mediump_dmat2x4		packed_dmat2x4;
	typedef packed_mediump_dmat3x2		packed_dmat3x2;
	typedef packed_mediump_dmat3x3		packed_dmat3x3;
	typedef packed_mediump_dmat3x4		packed_dmat3x4;
	typedef packed_mediump_dmat4x2		packed_dmat4x2;
	typedef packed_mediump_dmat4x3		packed_dmat4x3;
	typedef packed_mediump_dmat4x4		packed_dmat4x4;
#else //defined(GLM_PRECISION_HIGHP_DOUBLE)
	/// 1 component vector aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dvec1			aligned_dvec1;

	/// 2 components vector aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dvec2			aligned_dvec2;

	/// 3 components vector aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dvec3			aligned_dvec3;

	/// 4 components vector aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dvec4			aligned_dvec4;

	/// 1 component vector tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dvec1			packed_dvec1;

	/// 2 components vector tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dvec2			packed_dvec2;

	/// 3 components vector tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dvec3			packed_dvec3;

	/// 4 components vector tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dvec4			packed_dvec4;

	/// 2 by 2 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat2			aligned_dmat2;

	/// 3 by 3 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat3			aligned_dmat3;

	/// 4 by 4 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat4			aligned_dmat4;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat2			packed_dmat2;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat3			packed_dmat3;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat4			packed_dmat4;

	/// 2 by 2 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat2x2		aligned_dmat2x2;

	/// 2 by 3 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat2x3		aligned_dmat2x3;

	/// 2 by 4 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat2x4		aligned_dmat2x4;

	/// 3 by 2 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat3x2		aligned_dmat3x2;

	/// 3 by 3 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat3x3		aligned_dmat3x3;

	/// 3 by 4 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat3x4		aligned_dmat3x4;

	/// 4 by 2 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat4x2		aligned_dmat4x2;

	/// 4 by 3 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat4x3		aligned_dmat4x3;

	/// 4 by 4 matrix tightly aligned in memory of double-precision floating-point numbers.
	typedef aligned_highp_dmat4x4		aligned_dmat4x4;

	/// 2 by 2 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat2x2		packed_dmat2x2;

	/// 2 by 3 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat2x3		packed_dmat2x3;

	/// 2 by 4 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat2x4		packed_dmat2x4;

	/// 3 by 2 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat3x2		packed_dmat3x2;

	/// 3 by 3 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat3x3		packed_dmat3x3;

	/// 3 by 4 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat3x4		packed_dmat3x4;

	/// 4 by 2 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat4x2		packed_dmat4x2;

	/// 4 by 3 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat4x3		packed_dmat4x3;

	/// 4 by 4 matrix tightly packed in memory of double-precision floating-point numbers.
	typedef packed_highp_dmat4x4		packed_dmat4x4;
#endif//GLM_PRECISION

#if(defined(GLM_PRECISION_LOWP_INT))
	typedef aligned_lowp_ivec1			aligned_ivec1;
	typedef aligned_lowp_ivec2			aligned_ivec2;
	typedef aligned_lowp_ivec3			aligned_ivec3;
	typedef aligned_lowp_ivec4			aligned_ivec4;
#elif(defined(GLM_PRECISION_MEDIUMP_INT))
	typedef aligned_mediump_ivec1		aligned_ivec1;
	typedef aligned_mediump_ivec2		aligned_ivec2;
	typedef aligned_mediump_ivec3		aligned_ivec3;
	typedef aligned_mediump_ivec4		aligned_ivec4;
#else //defined(GLM_PRECISION_HIGHP_INT)
	/// 1 component vector aligned in memory of signed integer numbers.
	typedef aligned_highp_ivec1			aligned_ivec1;

	/// 2 components vector aligned in memory of signed integer numbers.
	typedef aligned_highp_ivec2			aligned_ivec2;

	/// 3 components vector aligned in memory of signed integer numbers.
	typedef aligned_highp_ivec3			aligned_ivec3;

	/// 4 components vector aligned in memory of signed integer numbers.
	typedef aligned_highp_ivec4			aligned_ivec4;

	/// 1 component vector tightly packed in memory of signed integer numbers.
	typedef packed_highp_ivec1			packed_ivec1;

	/// 2 components vector tightly packed in memory of signed integer numbers.
	typedef packed_highp_ivec2			packed_ivec2;

	/// 3 components vector tightly packed in memory of signed integer numbers.
	typedef packed_highp_ivec3			packed_ivec3;

	/// 4 components vector tightly packed in memory of signed integer numbers.
	typedef packed_highp_ivec4			packed_ivec4;
#endif//GLM_PRECISION

	// -- Unsigned integer definition --

#if(defined(GLM_PRECISION_LOWP_UINT))
	typedef aligned_lowp_uvec1			aligned_uvec1;
	typedef aligned_lowp_uvec2			aligned_uvec2;
	typedef aligned_lowp_uvec3			aligned_uvec3;
	typedef aligned_lowp_uvec4			aligned_uvec4;
#elif(defined(GLM_PRECISION_MEDIUMP_UINT))
	typedef aligned_mediump_uvec1		aligned_uvec1;
	typedef aligned_mediump_uvec2		aligned_uvec2;
	typedef aligned_mediump_uvec3		aligned_uvec3;
	typedef aligned_mediump_uvec4		aligned_uvec4;
#else //defined(GLM_PRECISION_HIGHP_UINT)
	/// 1 component vector aligned in memory of unsigned integer numbers.
	typedef aligned_highp_uvec1			aligned_uvec1;

	/// 2 components vector aligned in memory of unsigned integer numbers.
	typedef aligned_highp_uvec2			aligned_uvec2;

	/// 3 components vector aligned in memory of unsigned integer numbers.
	typedef aligned_highp_uvec3			aligned_uvec3;

	/// 4 components vector aligned in memory of unsigned integer numbers.
	typedef aligned_highp_uvec4			aligned_uvec4;

	/// 1 component vector tightly packed in memory of unsigned integer numbers.
	typedef packed_highp_uvec1			packed_uvec1;

	/// 2 components vector tightly packed in memory of unsigned integer numbers.
	typedef packed_highp_uvec2			packed_uvec2;

	/// 3 components vector tightly packed in memory of unsigned integer numbers.
	typedef packed_highp_uvec3			packed_uvec3;

	/// 4 components vector tightly packed in memory of unsigned integer numbers.
	typedef packed_highp_uvec4			packed_uvec4;
#endif//GLM_PRECISION

#if(defined(GLM_PRECISION_LOWP_BOOL))
	typedef aligned_lowp_bvec1			aligned_bvec1;
	typedef aligned_lowp_bvec2			aligned_bvec2;
	typedef aligned_lowp_bvec3			aligned_bvec3;
	typedef aligned_lowp_bvec4			aligned_bvec4;
#elif(defined(GLM_PRECISION_MEDIUMP_BOOL))
	typedef aligned_mediump_bvec1		aligned_bvec1;
	typedef aligned_mediump_bvec2		aligned_bvec2;
	typedef aligned_mediump_bvec3		aligned_bvec3;
	typedef aligned_mediump_bvec4		aligned_bvec4;
#else //defined(GLM_PRECISION_HIGHP_BOOL)
	/// 1 component vector aligned in memory of bool values.
	typedef aligned_highp_bvec1			aligned_bvec1;

	/// 2 components vector aligned in memory of bool values.
	typedef aligned_highp_bvec2			aligned_bvec2;

	/// 3 components vector aligned in memory of bool values.
	typedef aligned_highp_bvec3			aligned_bvec3;

	/// 4 components vector aligned in memory of bool values.
	typedef aligned_highp_bvec4			aligned_bvec4;

	/// 1 components vector tightly packed in memory of bool values.
	typedef packed_highp_bvec1			packed_bvec1;

	/// 2 components vector tightly packed in memory of bool values.
	typedef packed_highp_bvec2			packed_bvec2;

	/// 3 components vector tightly packed in memory of bool values.
	typedef packed_highp_bvec3			packed_bvec3;

	/// 4 components vector tightly packed in memory of bool values.
	typedef packed_highp_bvec4			packed_bvec4;
#endif//GLM_PRECISION

	/// @}
}//namespace glm
