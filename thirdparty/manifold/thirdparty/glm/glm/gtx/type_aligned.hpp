/// @ref gtx_type_aligned
/// @file glm/gtx/type_aligned.hpp
///
/// @see core (dependence)
/// @see gtc_quaternion (dependence)
///
/// @defgroup gtx_type_aligned GLM_GTX_type_aligned
/// @ingroup gtx
///
/// Include <glm/gtx/type_aligned.hpp> to use the features of this extension.
///
/// Defines aligned types.

#pragma once

// Dependency:
#include "../gtc/type_precision.hpp"
#include "../gtc/quaternion.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_type_aligned is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_type_aligned extension included")
#	endif
#endif

namespace glm
{
	///////////////////////////
	// Signed int vector types

	/// @addtogroup gtx_type_aligned
	/// @{

	/// Low qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int8, aligned_lowp_int8, 1);

	/// Low qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int16, aligned_lowp_int16, 2);

	/// Low qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int32, aligned_lowp_int32, 4);

	/// Low qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int64, aligned_lowp_int64, 8);


	/// Low qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int8_t, aligned_lowp_int8_t, 1);

	/// Low qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int16_t, aligned_lowp_int16_t, 2);

	/// Low qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int32_t, aligned_lowp_int32_t, 4);

	/// Low qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_int64_t, aligned_lowp_int64_t, 8);


	/// Low qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_i8, aligned_lowp_i8, 1);

	/// Low qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_i16, aligned_lowp_i16, 2);

	/// Low qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_i32, aligned_lowp_i32, 4);

	/// Low qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_i64, aligned_lowp_i64, 8);


	/// Medium qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int8, aligned_mediump_int8, 1);

	/// Medium qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int16, aligned_mediump_int16, 2);

	/// Medium qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int32, aligned_mediump_int32, 4);

	/// Medium qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int64, aligned_mediump_int64, 8);


	/// Medium qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int8_t, aligned_mediump_int8_t, 1);

	/// Medium qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int16_t, aligned_mediump_int16_t, 2);

	/// Medium qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int32_t, aligned_mediump_int32_t, 4);

	/// Medium qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_int64_t, aligned_mediump_int64_t, 8);


	/// Medium qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_i8, aligned_mediump_i8, 1);

	/// Medium qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_i16, aligned_mediump_i16, 2);

	/// Medium qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_i32, aligned_mediump_i32, 4);

	/// Medium qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_i64, aligned_mediump_i64, 8);


	/// High qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int8, aligned_highp_int8, 1);

	/// High qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int16, aligned_highp_int16, 2);

	/// High qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int32, aligned_highp_int32, 4);

	/// High qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int64, aligned_highp_int64, 8);


	/// High qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int8_t, aligned_highp_int8_t, 1);

	/// High qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int16_t, aligned_highp_int16_t, 2);

	/// High qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int32_t, aligned_highp_int32_t, 4);

	/// High qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_int64_t, aligned_highp_int64_t, 8);


	/// High qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_i8, aligned_highp_i8, 1);

	/// High qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_i16, aligned_highp_i16, 2);

	/// High qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_i32, aligned_highp_i32, 4);

	/// High qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_i64, aligned_highp_i64, 8);


	/// Default qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int8, aligned_int8, 1);

	/// Default qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int16, aligned_int16, 2);

	/// Default qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int32, aligned_int32, 4);

	/// Default qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int64, aligned_int64, 8);


	/// Default qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int8_t, aligned_int8_t, 1);

	/// Default qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int16_t, aligned_int16_t, 2);

	/// Default qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int32_t, aligned_int32_t, 4);

	/// Default qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(int64_t, aligned_int64_t, 8);


	/// Default qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i8, aligned_i8, 1);

	/// Default qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i16, aligned_i16, 2);

	/// Default qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i32, aligned_i32, 4);

	/// Default qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i64, aligned_i64, 8);


	/// Default qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(ivec1, aligned_ivec1, 4);

	/// Default qualifier 32 bit signed integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(ivec2, aligned_ivec2, 8);

	/// Default qualifier 32 bit signed integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(ivec3, aligned_ivec3, 16);

	/// Default qualifier 32 bit signed integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(ivec4, aligned_ivec4, 16);


	/// Default qualifier 8 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i8vec1, aligned_i8vec1, 1);

	/// Default qualifier 8 bit signed integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i8vec2, aligned_i8vec2, 2);

	/// Default qualifier 8 bit signed integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i8vec3, aligned_i8vec3, 4);

	/// Default qualifier 8 bit signed integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i8vec4, aligned_i8vec4, 4);


	/// Default qualifier 16 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i16vec1, aligned_i16vec1, 2);

	/// Default qualifier 16 bit signed integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i16vec2, aligned_i16vec2, 4);

	/// Default qualifier 16 bit signed integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i16vec3, aligned_i16vec3, 8);

	/// Default qualifier 16 bit signed integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i16vec4, aligned_i16vec4, 8);


	/// Default qualifier 32 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i32vec1, aligned_i32vec1, 4);

	/// Default qualifier 32 bit signed integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i32vec2, aligned_i32vec2, 8);

	/// Default qualifier 32 bit signed integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i32vec3, aligned_i32vec3, 16);

	/// Default qualifier 32 bit signed integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i32vec4, aligned_i32vec4, 16);


	/// Default qualifier 64 bit signed integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i64vec1, aligned_i64vec1, 8);

	/// Default qualifier 64 bit signed integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i64vec2, aligned_i64vec2, 16);

	/// Default qualifier 64 bit signed integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i64vec3, aligned_i64vec3, 32);

	/// Default qualifier 64 bit signed integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(i64vec4, aligned_i64vec4, 32);


	/////////////////////////////
	// Unsigned int vector types

	/// Low qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint8, aligned_lowp_uint8, 1);

	/// Low qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint16, aligned_lowp_uint16, 2);

	/// Low qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint32, aligned_lowp_uint32, 4);

	/// Low qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint64, aligned_lowp_uint64, 8);


	/// Low qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint8_t, aligned_lowp_uint8_t, 1);

	/// Low qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint16_t, aligned_lowp_uint16_t, 2);

	/// Low qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint32_t, aligned_lowp_uint32_t, 4);

	/// Low qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_uint64_t, aligned_lowp_uint64_t, 8);


	/// Low qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_u8, aligned_lowp_u8, 1);

	/// Low qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_u16, aligned_lowp_u16, 2);

	/// Low qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_u32, aligned_lowp_u32, 4);

	/// Low qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(lowp_u64, aligned_lowp_u64, 8);


	/// Medium qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint8, aligned_mediump_uint8, 1);

	/// Medium qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint16, aligned_mediump_uint16, 2);

	/// Medium qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint32, aligned_mediump_uint32, 4);

	/// Medium qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint64, aligned_mediump_uint64, 8);


	/// Medium qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint8_t, aligned_mediump_uint8_t, 1);

	/// Medium qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint16_t, aligned_mediump_uint16_t, 2);

	/// Medium qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint32_t, aligned_mediump_uint32_t, 4);

	/// Medium qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_uint64_t, aligned_mediump_uint64_t, 8);


	/// Medium qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_u8, aligned_mediump_u8, 1);

	/// Medium qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_u16, aligned_mediump_u16, 2);

	/// Medium qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_u32, aligned_mediump_u32, 4);

	/// Medium qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mediump_u64, aligned_mediump_u64, 8);


	/// High qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint8, aligned_highp_uint8, 1);

	/// High qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint16, aligned_highp_uint16, 2);

	/// High qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint32, aligned_highp_uint32, 4);

	/// High qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint64, aligned_highp_uint64, 8);


	/// High qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint8_t, aligned_highp_uint8_t, 1);

	/// High qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint16_t, aligned_highp_uint16_t, 2);

	/// High qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint32_t, aligned_highp_uint32_t, 4);

	/// High qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_uint64_t, aligned_highp_uint64_t, 8);


	/// High qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_u8, aligned_highp_u8, 1);

	/// High qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_u16, aligned_highp_u16, 2);

	/// High qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_u32, aligned_highp_u32, 4);

	/// High qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(highp_u64, aligned_highp_u64, 8);


	/// Default qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint8, aligned_uint8, 1);

	/// Default qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint16, aligned_uint16, 2);

	/// Default qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint32, aligned_uint32, 4);

	/// Default qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint64, aligned_uint64, 8);


	/// Default qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint8_t, aligned_uint8_t, 1);

	/// Default qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint16_t, aligned_uint16_t, 2);

	/// Default qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint32_t, aligned_uint32_t, 4);

	/// Default qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uint64_t, aligned_uint64_t, 8);


	/// Default qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u8, aligned_u8, 1);

	/// Default qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u16, aligned_u16, 2);

	/// Default qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u32, aligned_u32, 4);

	/// Default qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u64, aligned_u64, 8);


	/// Default qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uvec1, aligned_uvec1, 4);

	/// Default qualifier 32 bit unsigned integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uvec2, aligned_uvec2, 8);

	/// Default qualifier 32 bit unsigned integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uvec3, aligned_uvec3, 16);

	/// Default qualifier 32 bit unsigned integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(uvec4, aligned_uvec4, 16);


	/// Default qualifier 8 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u8vec1, aligned_u8vec1, 1);

	/// Default qualifier 8 bit unsigned integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u8vec2, aligned_u8vec2, 2);

	/// Default qualifier 8 bit unsigned integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u8vec3, aligned_u8vec3, 4);

	/// Default qualifier 8 bit unsigned integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u8vec4, aligned_u8vec4, 4);


	/// Default qualifier 16 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u16vec1, aligned_u16vec1, 2);

	/// Default qualifier 16 bit unsigned integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u16vec2, aligned_u16vec2, 4);

	/// Default qualifier 16 bit unsigned integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u16vec3, aligned_u16vec3, 8);

	/// Default qualifier 16 bit unsigned integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u16vec4, aligned_u16vec4, 8);


	/// Default qualifier 32 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u32vec1, aligned_u32vec1, 4);

	/// Default qualifier 32 bit unsigned integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u32vec2, aligned_u32vec2, 8);

	/// Default qualifier 32 bit unsigned integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u32vec3, aligned_u32vec3, 16);

	/// Default qualifier 32 bit unsigned integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u32vec4, aligned_u32vec4, 16);


	/// Default qualifier 64 bit unsigned integer aligned scalar type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u64vec1, aligned_u64vec1, 8);

	/// Default qualifier 64 bit unsigned integer aligned vector of 2 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u64vec2, aligned_u64vec2, 16);

	/// Default qualifier 64 bit unsigned integer aligned vector of 3 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u64vec3, aligned_u64vec3, 32);

	/// Default qualifier 64 bit unsigned integer aligned vector of 4 components type.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(u64vec4, aligned_u64vec4, 32);


	//////////////////////
	// Float vector types

	/// 32 bit single-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float32, aligned_float32, 4);

	/// 32 bit single-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float32_t, aligned_float32_t, 4);

	/// 32 bit single-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float32, aligned_f32, 4);

#	ifndef GLM_FORCE_SINGLE_ONLY

	/// 64 bit double-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float64, aligned_float64, 8);

	/// 64 bit double-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float64_t, aligned_float64_t, 8);

	/// 64 bit double-qualifier floating-point aligned scalar.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(float64, aligned_f64, 8);

#	endif//GLM_FORCE_SINGLE_ONLY


	/// Single-qualifier floating-point aligned vector of 1 component.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(vec1, aligned_vec1, 4);

	/// Single-qualifier floating-point aligned vector of 2 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(vec2, aligned_vec2, 8);

	/// Single-qualifier floating-point aligned vector of 3 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(vec3, aligned_vec3, 16);

	/// Single-qualifier floating-point aligned vector of 4 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(vec4, aligned_vec4, 16);


	/// Single-qualifier floating-point aligned vector of 1 component.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fvec1, aligned_fvec1, 4);

	/// Single-qualifier floating-point aligned vector of 2 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fvec2, aligned_fvec2, 8);

	/// Single-qualifier floating-point aligned vector of 3 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fvec3, aligned_fvec3, 16);

	/// Single-qualifier floating-point aligned vector of 4 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fvec4, aligned_fvec4, 16);


	/// Single-qualifier floating-point aligned vector of 1 component.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32vec1, aligned_f32vec1, 4);

	/// Single-qualifier floating-point aligned vector of 2 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32vec2, aligned_f32vec2, 8);

	/// Single-qualifier floating-point aligned vector of 3 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32vec3, aligned_f32vec3, 16);

	/// Single-qualifier floating-point aligned vector of 4 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32vec4, aligned_f32vec4, 16);


	/// Double-qualifier floating-point aligned vector of 1 component.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(dvec1, aligned_dvec1, 8);

	/// Double-qualifier floating-point aligned vector of 2 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(dvec2, aligned_dvec2, 16);

	/// Double-qualifier floating-point aligned vector of 3 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(dvec3, aligned_dvec3, 32);

	/// Double-qualifier floating-point aligned vector of 4 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(dvec4, aligned_dvec4, 32);


#	ifndef GLM_FORCE_SINGLE_ONLY

	/// Double-qualifier floating-point aligned vector of 1 component.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64vec1, aligned_f64vec1, 8);

	/// Double-qualifier floating-point aligned vector of 2 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64vec2, aligned_f64vec2, 16);

	/// Double-qualifier floating-point aligned vector of 3 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64vec3, aligned_f64vec3, 32);

	/// Double-qualifier floating-point aligned vector of 4 components.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64vec4, aligned_f64vec4, 32);

#	endif//GLM_FORCE_SINGLE_ONLY

	//////////////////////
	// Float matrix types

	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef detail::tmat1<f32> mat1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat2, aligned_mat2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat3, aligned_mat3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat4, aligned_mat4, 16);


	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef detail::tmat1x1<f32> mat1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat2x2, aligned_mat2x2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat3x3, aligned_mat3x3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(mat4x4, aligned_mat4x4, 16);


	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef detail::tmat1x1<f32> fmat1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat2x2, aligned_fmat2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat3x3, aligned_fmat3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat4x4, aligned_fmat4, 16);


	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef f32 fmat1x1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat2x2, aligned_fmat2x2, 16);

	/// Single-qualifier floating-point aligned 2x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat2x3, aligned_fmat2x3, 16);

	/// Single-qualifier floating-point aligned 2x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat2x4, aligned_fmat2x4, 16);

	/// Single-qualifier floating-point aligned 3x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat3x2, aligned_fmat3x2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat3x3, aligned_fmat3x3, 16);

	/// Single-qualifier floating-point aligned 3x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat3x4, aligned_fmat3x4, 16);

	/// Single-qualifier floating-point aligned 4x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat4x2, aligned_fmat4x2, 16);

	/// Single-qualifier floating-point aligned 4x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat4x3, aligned_fmat4x3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(fmat4x4, aligned_fmat4x4, 16);


	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef detail::tmat1x1<f32, defaultp> f32mat1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat2x2, aligned_f32mat2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat3x3, aligned_f32mat3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat4x4, aligned_f32mat4, 16);


	/// Single-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef f32 f32mat1x1;

	/// Single-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat2x2, aligned_f32mat2x2, 16);

	/// Single-qualifier floating-point aligned 2x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat2x3, aligned_f32mat2x3, 16);

	/// Single-qualifier floating-point aligned 2x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat2x4, aligned_f32mat2x4, 16);

	/// Single-qualifier floating-point aligned 3x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat3x2, aligned_f32mat3x2, 16);

	/// Single-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat3x3, aligned_f32mat3x3, 16);

	/// Single-qualifier floating-point aligned 3x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat3x4, aligned_f32mat3x4, 16);

	/// Single-qualifier floating-point aligned 4x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat4x2, aligned_f32mat4x2, 16);

	/// Single-qualifier floating-point aligned 4x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat4x3, aligned_f32mat4x3, 16);

	/// Single-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32mat4x4, aligned_f32mat4x4, 16);


#	ifndef GLM_FORCE_SINGLE_ONLY

	/// Double-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef detail::tmat1x1<f64, defaultp> f64mat1;

	/// Double-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat2x2, aligned_f64mat2, 32);

	/// Double-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat3x3, aligned_f64mat3, 32);

	/// Double-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat4x4, aligned_f64mat4, 32);


	/// Double-qualifier floating-point aligned 1x1 matrix.
	/// @see gtx_type_aligned
	//typedef f64 f64mat1x1;

	/// Double-qualifier floating-point aligned 2x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat2x2, aligned_f64mat2x2, 32);

	/// Double-qualifier floating-point aligned 2x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat2x3, aligned_f64mat2x3, 32);

	/// Double-qualifier floating-point aligned 2x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat2x4, aligned_f64mat2x4, 32);

	/// Double-qualifier floating-point aligned 3x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat3x2, aligned_f64mat3x2, 32);

	/// Double-qualifier floating-point aligned 3x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat3x3, aligned_f64mat3x3, 32);

	/// Double-qualifier floating-point aligned 3x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat3x4, aligned_f64mat3x4, 32);

	/// Double-qualifier floating-point aligned 4x2 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat4x2, aligned_f64mat4x2, 32);

	/// Double-qualifier floating-point aligned 4x3 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat4x3, aligned_f64mat4x3, 32);

	/// Double-qualifier floating-point aligned 4x4 matrix.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64mat4x4, aligned_f64mat4x4, 32);

#	endif//GLM_FORCE_SINGLE_ONLY


	//////////////////////////
	// Quaternion types

	/// Single-qualifier floating-point aligned quaternion.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(quat, aligned_quat, 16);

	/// Single-qualifier floating-point aligned quaternion.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(quat, aligned_fquat, 16);

	/// Double-qualifier floating-point aligned quaternion.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(dquat, aligned_dquat, 32);

	/// Single-qualifier floating-point aligned quaternion.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f32quat, aligned_f32quat, 16);

#	ifndef GLM_FORCE_SINGLE_ONLY

	/// Double-qualifier floating-point aligned quaternion.
	/// @see gtx_type_aligned
	GLM_ALIGNED_TYPEDEF(f64quat, aligned_f64quat, 32);

#	endif//GLM_FORCE_SINGLE_ONLY

	/// @}
}//namespace glm

#include "type_aligned.inl"
