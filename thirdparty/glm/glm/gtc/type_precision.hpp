/// @ref gtc_type_precision
/// @file glm/gtc/type_precision.hpp
///
/// @see core (dependence)
/// @see gtc_quaternion (dependence)
///
/// @defgroup gtc_type_precision GLM_GTC_type_precision
/// @ingroup gtc
///
/// Include <glm/gtc/type_precision.hpp> to use the features of this extension.
///
/// Defines specific C++-based qualifier types.

#pragma once

// Dependency:
#include "../gtc/quaternion.hpp"
#include "../gtc/vec1.hpp"
#include "../ext/vector_int1_sized.hpp"
#include "../ext/vector_int2_sized.hpp"
#include "../ext/vector_int3_sized.hpp"
#include "../ext/vector_int4_sized.hpp"
#include "../ext/scalar_int_sized.hpp"
#include "../ext/vector_uint1_sized.hpp"
#include "../ext/vector_uint2_sized.hpp"
#include "../ext/vector_uint3_sized.hpp"
#include "../ext/vector_uint4_sized.hpp"
#include "../ext/scalar_uint_sized.hpp"
#include "../detail/type_vec2.hpp"
#include "../detail/type_vec3.hpp"
#include "../detail/type_vec4.hpp"
#include "../detail/type_mat2x2.hpp"
#include "../detail/type_mat2x3.hpp"
#include "../detail/type_mat2x4.hpp"
#include "../detail/type_mat3x2.hpp"
#include "../detail/type_mat3x3.hpp"
#include "../detail/type_mat3x4.hpp"
#include "../detail/type_mat4x2.hpp"
#include "../detail/type_mat4x3.hpp"
#include "../detail/type_mat4x4.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_type_precision extension included")
#endif

namespace glm
{
	///////////////////////////
	// Signed int vector types

	/// @addtogroup gtc_type_precision
	/// @{

	/// Low qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_int8;

	/// Low qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_int16;

	/// Low qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_int32;

	/// Low qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_int64;

	/// Low qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_int8_t;

	/// Low qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_int16_t;

	/// Low qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_int32_t;

	/// Low qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_int64_t;

	/// Low qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 lowp_i8;

	/// Low qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 lowp_i16;

	/// Low qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 lowp_i32;

	/// Low qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 lowp_i64;

	/// Medium qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_int8;

	/// Medium qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_int16;

	/// Medium qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_int32;

	/// Medium qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_int64;

	/// Medium qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_int8_t;

	/// Medium qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_int16_t;

	/// Medium qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_int32_t;

	/// Medium qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_int64_t;

	/// Medium qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 mediump_i8;

	/// Medium qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 mediump_i16;

	/// Medium qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 mediump_i32;

	/// Medium qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 mediump_i64;

	/// High qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_int8;

	/// High qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_int16;

	/// High qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_int32;

	/// High qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_int64;

	/// High qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_int8_t;

	/// High qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_int16_t;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_int32_t;

	/// High qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_int64_t;

	/// High qualifier 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 highp_i8;

	/// High qualifier 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 highp_i16;

	/// High qualifier 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 highp_i32;

	/// High qualifier 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 highp_i64;


#if GLM_HAS_EXTENDED_INTEGER_TYPE
	using std::int8_t;
	using std::int16_t;
	using std::int32_t;
	using std::int64_t;
#else
	/// 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 int8_t;

	/// 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 int16_t;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 int32_t;

	/// 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 int64_t;
#endif

	/// 8 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int8 i8;

	/// 16 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int16 i16;

	/// 32 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int32 i32;

	/// 64 bit signed integer type.
	/// @see gtc_type_precision
	typedef detail::int64 i64;

	/////////////////////////////
	// Unsigned int vector types

	/// Low qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_uint8;

	/// Low qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_uint16;

	/// Low qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_uint32;

	/// Low qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_uint64;

	/// Low qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_uint8_t;

	/// Low qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_uint16_t;

	/// Low qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_uint32_t;

	/// Low qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_uint64_t;

	/// Low qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 lowp_u8;

	/// Low qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 lowp_u16;

	/// Low qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 lowp_u32;

	/// Low qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 lowp_u64;

	/// Medium qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_uint8;

	/// Medium qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_uint16;

	/// Medium qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_uint32;

	/// Medium qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_uint64;

	/// Medium qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_uint8_t;

	/// Medium qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_uint16_t;

	/// Medium qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_uint32_t;

	/// Medium qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_uint64_t;

	/// Medium qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 mediump_u8;

	/// Medium qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 mediump_u16;

	/// Medium qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 mediump_u32;

	/// Medium qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 mediump_u64;

	/// High qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_uint8;

	/// High qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_uint16;

	/// High qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_uint32;

	/// High qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_uint64;

	/// High qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_uint8_t;

	/// High qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_uint16_t;

	/// High qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_uint32_t;

	/// High qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_uint64_t;

	/// High qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 highp_u8;

	/// High qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 highp_u16;

	/// High qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 highp_u32;

	/// High qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 highp_u64;

#if GLM_HAS_EXTENDED_INTEGER_TYPE
	using std::uint8_t;
	using std::uint16_t;
	using std::uint32_t;
	using std::uint64_t;
#else
	/// Default qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 uint8_t;

	/// Default qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 uint16_t;

	/// Default qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 uint32_t;

	/// Default qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 uint64_t;
#endif

	/// Default qualifier 8 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint8 u8;

	/// Default qualifier 16 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint16 u16;

	/// Default qualifier 32 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint32 u32;

	/// Default qualifier 64 bit unsigned integer type.
	/// @see gtc_type_precision
	typedef detail::uint64 u64;





	//////////////////////
	// Float vector types

	/// Single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float float32;

	/// Double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef double float64;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32_t;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64_t;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_f32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_f64;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32_t;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64_t;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_f32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_f64;


	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_float32_t;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_float64_t;

	/// Low 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 lowp_f32;

	/// Low 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 lowp_f64;


	/// Medium 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 mediump_float32;

	/// Medium 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 mediump_float64;

	/// Medium 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 mediump_float32_t;

	/// Medium 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 mediump_float64_t;

	/// Medium 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 mediump_f32;

	/// Medium 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 mediump_f64;


	/// High 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 highp_float32;

	/// High 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 highp_float64;

	/// High 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 highp_float32_t;

	/// High 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 highp_float64_t;

	/// High 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 highp_f32;

	/// High 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float64 highp_f64;


#if(defined(GLM_PRECISION_LOWP_FLOAT))
	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef lowp_float32_t float32_t;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef lowp_float64_t float64_t;

	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef lowp_f32 f32;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef lowp_f64 f64;

#elif(defined(GLM_PRECISION_MEDIUMP_FLOAT))
	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef mediump_float32 float32_t;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef mediump_float64 float64_t;

	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef mediump_float32 f32;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef mediump_float64 f64;

#else//(defined(GLM_PRECISION_HIGHP_FLOAT))

	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef highp_float32_t float32_t;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef highp_float64_t float64_t;

	/// Default 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef highp_float32_t f32;

	/// Default 64 bit double-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef highp_float64_t f64;
#endif


	/// Low single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, float, lowp> lowp_fvec1;

	/// Low single-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, float, lowp> lowp_fvec2;

	/// Low single-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, float, lowp> lowp_fvec3;

	/// Low single-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, float, lowp> lowp_fvec4;


	/// Medium single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, float, mediump> mediump_fvec1;

	/// Medium Single-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, float, mediump> mediump_fvec2;

	/// Medium Single-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, float, mediump> mediump_fvec3;

	/// Medium Single-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, float, mediump> mediump_fvec4;


	/// High single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, float, highp> highp_fvec1;

	/// High Single-qualifier floating-point vector of 2 components.
	/// @see core_precision
	typedef vec<2, float, highp> highp_fvec2;

	/// High Single-qualifier floating-point vector of 3 components.
	/// @see core_precision
	typedef vec<3, float, highp> highp_fvec3;

	/// High Single-qualifier floating-point vector of 4 components.
	/// @see core_precision
	typedef vec<4, float, highp> highp_fvec4;


	/// Low single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f32, lowp> lowp_f32vec1;

	/// Low single-qualifier floating-point vector of 2 components.
	/// @see core_precision
	typedef vec<2, f32, lowp> lowp_f32vec2;

	/// Low single-qualifier floating-point vector of 3 components.
	/// @see core_precision
	typedef vec<3, f32, lowp> lowp_f32vec3;

	/// Low single-qualifier floating-point vector of 4 components.
	/// @see core_precision
	typedef vec<4, f32, lowp> lowp_f32vec4;

	/// Medium single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f32, mediump> mediump_f32vec1;

	/// Medium single-qualifier floating-point vector of 2 components.
	/// @see core_precision
	typedef vec<2, f32, mediump> mediump_f32vec2;

	/// Medium single-qualifier floating-point vector of 3 components.
	/// @see core_precision
	typedef vec<3, f32, mediump> mediump_f32vec3;

	/// Medium single-qualifier floating-point vector of 4 components.
	/// @see core_precision
	typedef vec<4, f32, mediump> mediump_f32vec4;

	/// High single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f32, highp> highp_f32vec1;

	/// High single-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, f32, highp> highp_f32vec2;

	/// High single-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, f32, highp> highp_f32vec3;

	/// High single-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, f32, highp> highp_f32vec4;


	/// Low double-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f64, lowp> lowp_f64vec1;

	/// Low double-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, f64, lowp> lowp_f64vec2;

	/// Low double-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, f64, lowp> lowp_f64vec3;

	/// Low double-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, f64, lowp> lowp_f64vec4;

	/// Medium double-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f64, mediump> mediump_f64vec1;

	/// Medium double-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, f64, mediump> mediump_f64vec2;

	/// Medium double-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, f64, mediump> mediump_f64vec3;

	/// Medium double-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, f64, mediump> mediump_f64vec4;

	/// High double-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f64, highp> highp_f64vec1;

	/// High double-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, f64, highp> highp_f64vec2;

	/// High double-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, f64, highp> highp_f64vec3;

	/// High double-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, f64, highp> highp_f64vec4;



	//////////////////////
	// Float matrix types

	/// Low single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef lowp_f32 lowp_fmat1x1;

	/// Low single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, lowp> lowp_fmat2x2;

	/// Low single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, lowp> lowp_fmat2x3;

	/// Low single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, lowp> lowp_fmat2x4;

	/// Low single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, lowp> lowp_fmat3x2;

	/// Low single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, lowp> lowp_fmat3x3;

	/// Low single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, lowp> lowp_fmat3x4;

	/// Low single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, lowp> lowp_fmat4x2;

	/// Low single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, lowp> lowp_fmat4x3;

	/// Low single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, lowp> lowp_fmat4x4;

	/// Low single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef lowp_fmat1x1 lowp_fmat1;

	/// Low single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef lowp_fmat2x2 lowp_fmat2;

	/// Low single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef lowp_fmat3x3 lowp_fmat3;

	/// Low single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef lowp_fmat4x4 lowp_fmat4;


	/// Medium single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef mediump_f32 mediump_fmat1x1;

	/// Medium single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, mediump> mediump_fmat2x2;

	/// Medium single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, mediump> mediump_fmat2x3;

	/// Medium single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, mediump> mediump_fmat2x4;

	/// Medium single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, mediump> mediump_fmat3x2;

	/// Medium single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, mediump> mediump_fmat3x3;

	/// Medium single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, mediump> mediump_fmat3x4;

	/// Medium single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, mediump> mediump_fmat4x2;

	/// Medium single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, mediump> mediump_fmat4x3;

	/// Medium single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, mediump> mediump_fmat4x4;

	/// Medium single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef mediump_fmat1x1 mediump_fmat1;

	/// Medium single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mediump_fmat2x2 mediump_fmat2;

	/// Medium single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mediump_fmat3x3 mediump_fmat3;

	/// Medium single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mediump_fmat4x4 mediump_fmat4;


	/// High single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef highp_f32 highp_fmat1x1;

	/// High single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, highp> highp_fmat2x2;

	/// High single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, highp> highp_fmat2x3;

	/// High single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, highp> highp_fmat2x4;

	/// High single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, highp> highp_fmat3x2;

	/// High single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, highp> highp_fmat3x3;

	/// High single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, highp> highp_fmat3x4;

	/// High single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, highp> highp_fmat4x2;

	/// High single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, highp> highp_fmat4x3;

	/// High single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, highp> highp_fmat4x4;

	/// High single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef highp_fmat1x1 highp_fmat1;

	/// High single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef highp_fmat2x2 highp_fmat2;

	/// High single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef highp_fmat3x3 highp_fmat3;

	/// High single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef highp_fmat4x4 highp_fmat4;


	/// Low single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 lowp_f32mat1x1;

	/// Low single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, lowp> lowp_f32mat2x2;

	/// Low single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, lowp> lowp_f32mat2x3;

	/// Low single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, lowp> lowp_f32mat2x4;

	/// Low single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, lowp> lowp_f32mat3x2;

	/// Low single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, lowp> lowp_f32mat3x3;

	/// Low single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, lowp> lowp_f32mat3x4;

	/// Low single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, lowp> lowp_f32mat4x2;

	/// Low single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, lowp> lowp_f32mat4x3;

	/// Low single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, lowp> lowp_f32mat4x4;

	/// Low single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32, lowp> lowp_f32mat1;

	/// Low single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef lowp_f32mat2x2 lowp_f32mat2;

	/// Low single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef lowp_f32mat3x3 lowp_f32mat3;

	/// Low single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef lowp_f32mat4x4 lowp_f32mat4;


	/// High single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 mediump_f32mat1x1;

	/// Low single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, mediump> mediump_f32mat2x2;

	/// Medium single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, mediump> mediump_f32mat2x3;

	/// Medium single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, mediump> mediump_f32mat2x4;

	/// Medium single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, mediump> mediump_f32mat3x2;

	/// Medium single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, mediump> mediump_f32mat3x3;

	/// Medium single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, mediump> mediump_f32mat3x4;

	/// Medium single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, mediump> mediump_f32mat4x2;

	/// Medium single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, mediump> mediump_f32mat4x3;

	/// Medium single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, mediump> mediump_f32mat4x4;

	/// Medium single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32, mediump> f32mat1;

	/// Medium single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mediump_f32mat2x2 mediump_f32mat2;

	/// Medium single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mediump_f32mat3x3 mediump_f32mat3;

	/// Medium single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mediump_f32mat4x4 mediump_f32mat4;


	/// High single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 highp_f32mat1x1;

	/// High single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, highp> highp_f32mat2x2;

	/// High single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, highp> highp_f32mat2x3;

	/// High single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, highp> highp_f32mat2x4;

	/// High single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, highp> highp_f32mat3x2;

	/// High single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, highp> highp_f32mat3x3;

	/// High single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, highp> highp_f32mat3x4;

	/// High single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, highp> highp_f32mat4x2;

	/// High single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, highp> highp_f32mat4x3;

	/// High single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, highp> highp_f32mat4x4;

	/// High single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32, highp> f32mat1;

	/// High single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef highp_f32mat2x2 highp_f32mat2;

	/// High single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef highp_f32mat3x3 highp_f32mat3;

	/// High single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef highp_f32mat4x4 highp_f32mat4;


	/// Low double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f64 lowp_f64mat1x1;

	/// Low double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f64, lowp> lowp_f64mat2x2;

	/// Low double-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f64, lowp> lowp_f64mat2x3;

	/// Low double-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f64, lowp> lowp_f64mat2x4;

	/// Low double-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f64, lowp> lowp_f64mat3x2;

	/// Low double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f64, lowp> lowp_f64mat3x3;

	/// Low double-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f64, lowp> lowp_f64mat3x4;

	/// Low double-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f64, lowp> lowp_f64mat4x2;

	/// Low double-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f64, lowp> lowp_f64mat4x3;

	/// Low double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f64, lowp> lowp_f64mat4x4;

	/// Low double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef lowp_f64mat1x1 lowp_f64mat1;

	/// Low double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef lowp_f64mat2x2 lowp_f64mat2;

	/// Low double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef lowp_f64mat3x3 lowp_f64mat3;

	/// Low double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef lowp_f64mat4x4 lowp_f64mat4;


	/// Medium double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f64 Highp_f64mat1x1;

	/// Medium double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f64, mediump> mediump_f64mat2x2;

	/// Medium double-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f64, mediump> mediump_f64mat2x3;

	/// Medium double-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f64, mediump> mediump_f64mat2x4;

	/// Medium double-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f64, mediump> mediump_f64mat3x2;

	/// Medium double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f64, mediump> mediump_f64mat3x3;

	/// Medium double-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f64, mediump> mediump_f64mat3x4;

	/// Medium double-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f64, mediump> mediump_f64mat4x2;

	/// Medium double-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f64, mediump> mediump_f64mat4x3;

	/// Medium double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f64, mediump> mediump_f64mat4x4;

	/// Medium double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef mediump_f64mat1x1 mediump_f64mat1;

	/// Medium double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mediump_f64mat2x2 mediump_f64mat2;

	/// Medium double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mediump_f64mat3x3 mediump_f64mat3;

	/// Medium double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mediump_f64mat4x4 mediump_f64mat4;

	/// High double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f64 highp_f64mat1x1;

	/// High double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f64, highp> highp_f64mat2x2;

	/// High double-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f64, highp> highp_f64mat2x3;

	/// High double-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f64, highp> highp_f64mat2x4;

	/// High double-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f64, highp> highp_f64mat3x2;

	/// High double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f64, highp> highp_f64mat3x3;

	/// High double-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f64, highp> highp_f64mat3x4;

	/// High double-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f64, highp> highp_f64mat4x2;

	/// High double-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f64, highp> highp_f64mat4x3;

	/// High double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f64, highp> highp_f64mat4x4;

	/// High double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef highp_f64mat1x1 highp_f64mat1;

	/// High double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef highp_f64mat2x2 highp_f64mat2;

	/// High double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef highp_f64mat3x3 highp_f64mat3;

	/// High double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef highp_f64mat4x4 highp_f64mat4;


	/////////////////////////////
	// Signed int vector types

	/// Low qualifier signed integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, int, lowp>		lowp_ivec1;

	/// Low qualifier signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, int, lowp>		lowp_ivec2;

	/// Low qualifier signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, int, lowp>		lowp_ivec3;

	/// Low qualifier signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, int, lowp>		lowp_ivec4;


	/// Medium qualifier signed integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, int, mediump>	mediump_ivec1;

	/// Medium qualifier signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, int, mediump>	mediump_ivec2;

	/// Medium qualifier signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, int, mediump>	mediump_ivec3;

	/// Medium qualifier signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, int, mediump>	mediump_ivec4;


	/// High qualifier signed integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, int, highp>		highp_ivec1;

	/// High qualifier signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, int, highp>		highp_ivec2;

	/// High qualifier signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, int, highp>		highp_ivec3;

	/// High qualifier signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, int, highp>		highp_ivec4;


	/// Low qualifier 8 bit signed integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, i8, lowp>		lowp_i8vec1;

	/// Low qualifier 8 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i8, lowp>		lowp_i8vec2;

	/// Low qualifier 8 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i8, lowp>		lowp_i8vec3;

	/// Low qualifier 8 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i8, lowp>		lowp_i8vec4;


	/// Medium qualifier 8 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i8, mediump>		mediump_i8vec1;

	/// Medium qualifier 8 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i8, mediump>		mediump_i8vec2;

	/// Medium qualifier 8 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i8, mediump>		mediump_i8vec3;

	/// Medium qualifier 8 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i8, mediump>		mediump_i8vec4;


	/// High qualifier 8 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i8, highp>		highp_i8vec1;

	/// High qualifier 8 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i8, highp>		highp_i8vec2;

	/// High qualifier 8 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i8, highp>		highp_i8vec3;

	/// High qualifier 8 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i8, highp>		highp_i8vec4;


	/// Low qualifier 16 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i16, lowp>		lowp_i16vec1;

	/// Low qualifier 16 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i16, lowp>		lowp_i16vec2;

	/// Low qualifier 16 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i16, lowp>		lowp_i16vec3;

	/// Low qualifier 16 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i16, lowp>		lowp_i16vec4;


	/// Medium qualifier 16 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i16, mediump>	mediump_i16vec1;

	/// Medium qualifier 16 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i16, mediump>	mediump_i16vec2;

	/// Medium qualifier 16 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i16, mediump>	mediump_i16vec3;

	/// Medium qualifier 16 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i16, mediump>	mediump_i16vec4;


	/// High qualifier 16 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i16, highp>		highp_i16vec1;

	/// High qualifier 16 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i16, highp>		highp_i16vec2;

	/// High qualifier 16 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i16, highp>		highp_i16vec3;

	/// High qualifier 16 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i16, highp>		highp_i16vec4;


	/// Low qualifier 32 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i32, lowp>		lowp_i32vec1;

	/// Low qualifier 32 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i32, lowp>		lowp_i32vec2;

	/// Low qualifier 32 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i32, lowp>		lowp_i32vec3;

	/// Low qualifier 32 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i32, lowp>		lowp_i32vec4;


	/// Medium qualifier 32 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i32, mediump>	mediump_i32vec1;

	/// Medium qualifier 32 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i32, mediump>	mediump_i32vec2;

	/// Medium qualifier 32 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i32, mediump>	mediump_i32vec3;

	/// Medium qualifier 32 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i32, mediump>	mediump_i32vec4;


	/// High qualifier 32 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i32, highp>		highp_i32vec1;

	/// High qualifier 32 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i32, highp>		highp_i32vec2;

	/// High qualifier 32 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i32, highp>		highp_i32vec3;

	/// High qualifier 32 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i32, highp>		highp_i32vec4;


	/// Low qualifier 64 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i64, lowp>		lowp_i64vec1;

	/// Low qualifier 64 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i64, lowp>		lowp_i64vec2;

	/// Low qualifier 64 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i64, lowp>		lowp_i64vec3;

	/// Low qualifier 64 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i64, lowp>		lowp_i64vec4;


	/// Medium qualifier 64 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i64, mediump>	mediump_i64vec1;

	/// Medium qualifier 64 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i64, mediump>	mediump_i64vec2;

	/// Medium qualifier 64 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i64, mediump>	mediump_i64vec3;

	/// Medium qualifier 64 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i64, mediump>	mediump_i64vec4;


	/// High qualifier 64 bit signed integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, i64, highp>		highp_i64vec1;

	/// High qualifier 64 bit signed integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, i64, highp>		highp_i64vec2;

	/// High qualifier 64 bit signed integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, i64, highp>		highp_i64vec3;

	/// High qualifier 64 bit signed integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, i64, highp>		highp_i64vec4;


	/////////////////////////////
	// Unsigned int vector types

	/// Low qualifier unsigned integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, uint, lowp>		lowp_uvec1;

	/// Low qualifier unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, uint, lowp>		lowp_uvec2;

	/// Low qualifier unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, uint, lowp>		lowp_uvec3;

	/// Low qualifier unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, uint, lowp>		lowp_uvec4;


	/// Medium qualifier unsigned integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, uint, mediump>	mediump_uvec1;

	/// Medium qualifier unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, uint, mediump>	mediump_uvec2;

	/// Medium qualifier unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, uint, mediump>	mediump_uvec3;

	/// Medium qualifier unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, uint, mediump>	mediump_uvec4;


	/// High qualifier unsigned integer vector of 1 component type.
	/// @see gtc_type_precision
	typedef vec<1, uint, highp>		highp_uvec1;

	/// High qualifier unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, uint, highp>		highp_uvec2;

	/// High qualifier unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, uint, highp>		highp_uvec3;

	/// High qualifier unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, uint, highp>		highp_uvec4;


	/// Low qualifier 8 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u8, lowp>		lowp_u8vec1;

	/// Low qualifier 8 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u8, lowp>		lowp_u8vec2;

	/// Low qualifier 8 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u8, lowp>		lowp_u8vec3;

	/// Low qualifier 8 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u8, lowp>		lowp_u8vec4;


	/// Medium qualifier 8 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u8, mediump>		mediump_u8vec1;

	/// Medium qualifier 8 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u8, mediump>		mediump_u8vec2;

	/// Medium qualifier 8 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u8, mediump>		mediump_u8vec3;

	/// Medium qualifier 8 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u8, mediump>		mediump_u8vec4;


	/// High qualifier 8 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u8, highp>		highp_u8vec1;

	/// High qualifier 8 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u8, highp>		highp_u8vec2;

	/// High qualifier 8 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u8, highp>		highp_u8vec3;

	/// High qualifier 8 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u8, highp>		highp_u8vec4;


	/// Low qualifier 16 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u16, lowp>		lowp_u16vec1;

	/// Low qualifier 16 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u16, lowp>		lowp_u16vec2;

	/// Low qualifier 16 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u16, lowp>		lowp_u16vec3;

	/// Low qualifier 16 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u16, lowp>		lowp_u16vec4;


	/// Medium qualifier 16 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u16, mediump>	mediump_u16vec1;

	/// Medium qualifier 16 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u16, mediump>	mediump_u16vec2;

	/// Medium qualifier 16 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u16, mediump>	mediump_u16vec3;

	/// Medium qualifier 16 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u16, mediump>	mediump_u16vec4;


	/// High qualifier 16 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u16, highp>		highp_u16vec1;

	/// High qualifier 16 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u16, highp>		highp_u16vec2;

	/// High qualifier 16 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u16, highp>		highp_u16vec3;

	/// High qualifier 16 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u16, highp>		highp_u16vec4;


	/// Low qualifier 32 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u32, lowp>		lowp_u32vec1;

	/// Low qualifier 32 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u32, lowp>		lowp_u32vec2;

	/// Low qualifier 32 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u32, lowp>		lowp_u32vec3;

	/// Low qualifier 32 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u32, lowp>		lowp_u32vec4;


	/// Medium qualifier 32 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u32, mediump>	mediump_u32vec1;

	/// Medium qualifier 32 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u32, mediump>	mediump_u32vec2;

	/// Medium qualifier 32 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u32, mediump>	mediump_u32vec3;

	/// Medium qualifier 32 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u32, mediump>	mediump_u32vec4;


	/// High qualifier 32 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u32, highp>		highp_u32vec1;

	/// High qualifier 32 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u32, highp>		highp_u32vec2;

	/// High qualifier 32 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u32, highp>		highp_u32vec3;

	/// High qualifier 32 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u32, highp>		highp_u32vec4;


	/// Low qualifier 64 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u64, lowp>		lowp_u64vec1;

	/// Low qualifier 64 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u64, lowp>		lowp_u64vec2;

	/// Low qualifier 64 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u64, lowp>		lowp_u64vec3;

	/// Low qualifier 64 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u64, lowp>		lowp_u64vec4;


	/// Medium qualifier 64 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u64, mediump>	mediump_u64vec1;

	/// Medium qualifier 64 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u64, mediump>	mediump_u64vec2;

	/// Medium qualifier 64 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u64, mediump>	mediump_u64vec3;

	/// Medium qualifier 64 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u64, mediump>	mediump_u64vec4;


	/// High qualifier 64 bit unsigned integer scalar type.
	/// @see gtc_type_precision
	typedef vec<1, u64, highp>		highp_u64vec1;

	/// High qualifier 64 bit unsigned integer vector of 2 components type.
	/// @see gtc_type_precision
	typedef vec<2, u64, highp>		highp_u64vec2;

	/// High qualifier 64 bit unsigned integer vector of 3 components type.
	/// @see gtc_type_precision
	typedef vec<3, u64, highp>		highp_u64vec3;

	/// High qualifier 64 bit unsigned integer vector of 4 components type.
	/// @see gtc_type_precision
	typedef vec<4, u64, highp>		highp_u64vec4;


	//////////////////////
	// Float vector types

	/// 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 float32_t;

	/// 32 bit single-qualifier floating-point scalar.
	/// @see gtc_type_precision
	typedef float32 f32;

#	ifndef GLM_FORCE_SINGLE_ONLY

		/// 64 bit double-qualifier floating-point scalar.
		/// @see gtc_type_precision
		typedef float64 float64_t;

		/// 64 bit double-qualifier floating-point scalar.
		/// @see gtc_type_precision
		typedef float64 f64;
#	endif//GLM_FORCE_SINGLE_ONLY

	/// Single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, float, defaultp> fvec1;

	/// Single-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, float, defaultp> fvec2;

	/// Single-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, float, defaultp> fvec3;

	/// Single-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, float, defaultp> fvec4;


	/// Single-qualifier floating-point vector of 1 component.
	/// @see gtc_type_precision
	typedef vec<1, f32, defaultp> f32vec1;

	/// Single-qualifier floating-point vector of 2 components.
	/// @see gtc_type_precision
	typedef vec<2, f32, defaultp> f32vec2;

	/// Single-qualifier floating-point vector of 3 components.
	/// @see gtc_type_precision
	typedef vec<3, f32, defaultp> f32vec3;

	/// Single-qualifier floating-point vector of 4 components.
	/// @see gtc_type_precision
	typedef vec<4, f32, defaultp> f32vec4;

#	ifndef GLM_FORCE_SINGLE_ONLY
		/// Double-qualifier floating-point vector of 1 component.
		/// @see gtc_type_precision
		typedef vec<1, f64, defaultp> f64vec1;

		/// Double-qualifier floating-point vector of 2 components.
		/// @see gtc_type_precision
		typedef vec<2, f64, defaultp> f64vec2;

		/// Double-qualifier floating-point vector of 3 components.
		/// @see gtc_type_precision
		typedef vec<3, f64, defaultp> f64vec3;

		/// Double-qualifier floating-point vector of 4 components.
		/// @see gtc_type_precision
		typedef vec<4, f64, defaultp> f64vec4;
#	endif//GLM_FORCE_SINGLE_ONLY


	//////////////////////
	// Float matrix types

	/// Single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32> fmat1;

	/// Single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, defaultp> fmat2;

	/// Single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, defaultp> fmat3;

	/// Single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, defaultp> fmat4;


	/// Single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 fmat1x1;

	/// Single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, defaultp> fmat2x2;

	/// Single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, defaultp> fmat2x3;

	/// Single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, defaultp> fmat2x4;

	/// Single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, defaultp> fmat3x2;

	/// Single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, defaultp> fmat3x3;

	/// Single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, defaultp> fmat3x4;

	/// Single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, defaultp> fmat4x2;

	/// Single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, defaultp> fmat4x3;

	/// Single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, defaultp> fmat4x4;


	/// Single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f32, defaultp> f32mat1;

	/// Single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, defaultp> f32mat2;

	/// Single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, defaultp> f32mat3;

	/// Single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, defaultp> f32mat4;


	/// Single-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f32 f32mat1x1;

	/// Single-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f32, defaultp> f32mat2x2;

	/// Single-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f32, defaultp> f32mat2x3;

	/// Single-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f32, defaultp> f32mat2x4;

	/// Single-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f32, defaultp> f32mat3x2;

	/// Single-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f32, defaultp> f32mat3x3;

	/// Single-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f32, defaultp> f32mat3x4;

	/// Single-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f32, defaultp> f32mat4x2;

	/// Single-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f32, defaultp> f32mat4x3;

	/// Single-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f32, defaultp> f32mat4x4;


#	ifndef GLM_FORCE_SINGLE_ONLY

	/// Double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef detail::tmat1x1<f64, defaultp> f64mat1;

	/// Double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f64, defaultp> f64mat2;

	/// Double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f64, defaultp> f64mat3;

	/// Double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f64, defaultp> f64mat4;


	/// Double-qualifier floating-point 1x1 matrix.
	/// @see gtc_type_precision
	//typedef f64 f64mat1x1;

	/// Double-qualifier floating-point 2x2 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 2, f64, defaultp> f64mat2x2;

	/// Double-qualifier floating-point 2x3 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 3, f64, defaultp> f64mat2x3;

	/// Double-qualifier floating-point 2x4 matrix.
	/// @see gtc_type_precision
	typedef mat<2, 4, f64, defaultp> f64mat2x4;

	/// Double-qualifier floating-point 3x2 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 2, f64, defaultp> f64mat3x2;

	/// Double-qualifier floating-point 3x3 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 3, f64, defaultp> f64mat3x3;

	/// Double-qualifier floating-point 3x4 matrix.
	/// @see gtc_type_precision
	typedef mat<3, 4, f64, defaultp> f64mat3x4;

	/// Double-qualifier floating-point 4x2 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 2, f64, defaultp> f64mat4x2;

	/// Double-qualifier floating-point 4x3 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 3, f64, defaultp> f64mat4x3;

	/// Double-qualifier floating-point 4x4 matrix.
	/// @see gtc_type_precision
	typedef mat<4, 4, f64, defaultp> f64mat4x4;

#	endif//GLM_FORCE_SINGLE_ONLY

	//////////////////////////
	// Quaternion types

	/// Single-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f32, defaultp> f32quat;

	/// Low single-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f32, lowp> lowp_f32quat;

	/// Low double-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f64, lowp> lowp_f64quat;

	/// Medium single-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f32, mediump> mediump_f32quat;

#	ifndef GLM_FORCE_SINGLE_ONLY

	/// Medium double-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f64, mediump> mediump_f64quat;

	/// High single-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f32, highp> highp_f32quat;

	/// High double-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f64, highp> highp_f64quat;

	/// Double-qualifier floating-point quaternion.
	/// @see gtc_type_precision
	typedef qua<f64, defaultp> f64quat;

#	endif//GLM_FORCE_SINGLE_ONLY

	/// @}
}//namespace glm

#include "type_precision.inl"
