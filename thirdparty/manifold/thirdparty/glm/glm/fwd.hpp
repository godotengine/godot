#pragma once

#include "detail/qualifier.hpp"

namespace glm
{
#if GLM_HAS_EXTENDED_INTEGER_TYPE
	typedef std::int8_t				int8;
	typedef std::int16_t			int16;
	typedef std::int32_t			int32;
	typedef std::int64_t			int64;

	typedef std::uint8_t			uint8;
	typedef std::uint16_t			uint16;
	typedef std::uint32_t			uint32;
	typedef std::uint64_t			uint64;
#else
	typedef signed char				int8;
	typedef signed short			int16;
	typedef signed int				int32;
	typedef detail::int64			int64;

	typedef unsigned char			uint8;
	typedef unsigned short			uint16;
	typedef unsigned int			uint32;
	typedef detail::uint64			uint64;
#endif

	// Scalar int

	typedef int8					lowp_i8;
	typedef int8					mediump_i8;
	typedef int8					highp_i8;
	typedef int8					i8;

	typedef int8					lowp_int8;
	typedef int8					mediump_int8;
	typedef int8					highp_int8;

	typedef int8					lowp_int8_t;
	typedef int8					mediump_int8_t;
	typedef int8					highp_int8_t;
	typedef int8					int8_t;

	typedef int16					lowp_i16;
	typedef int16					mediump_i16;
	typedef int16					highp_i16;
	typedef int16					i16;

	typedef int16					lowp_int16;
	typedef int16					mediump_int16;
	typedef int16					highp_int16;

	typedef int16					lowp_int16_t;
	typedef int16					mediump_int16_t;
	typedef int16					highp_int16_t;
	typedef int16					int16_t;

	typedef int32					lowp_i32;
	typedef int32					mediump_i32;
	typedef int32					highp_i32;
	typedef int32					i32;

	typedef int32					lowp_int32;
	typedef int32					mediump_int32;
	typedef int32					highp_int32;

	typedef int32					lowp_int32_t;
	typedef int32					mediump_int32_t;
	typedef int32					highp_int32_t;
	typedef int32					int32_t;

	typedef int64					lowp_i64;
	typedef int64					mediump_i64;
	typedef int64					highp_i64;
	typedef int64					i64;

	typedef int64					lowp_int64;
	typedef int64					mediump_int64;
	typedef int64					highp_int64;

	typedef int64					lowp_int64_t;
	typedef int64					mediump_int64_t;
	typedef int64					highp_int64_t;
	typedef int64					int64_t;

	// Scalar uint

	typedef unsigned int			uint;

	typedef uint8					lowp_u8;
	typedef uint8					mediump_u8;
	typedef uint8					highp_u8;
	typedef uint8					u8;

	typedef uint8					lowp_uint8;
	typedef uint8					mediump_uint8;
	typedef uint8					highp_uint8;

	typedef uint8					lowp_uint8_t;
	typedef uint8					mediump_uint8_t;
	typedef uint8					highp_uint8_t;
	typedef uint8					uint8_t;

	typedef uint16					lowp_u16;
	typedef uint16					mediump_u16;
	typedef uint16					highp_u16;
	typedef uint16					u16;

	typedef uint16					lowp_uint16;
	typedef uint16					mediump_uint16;
	typedef uint16					highp_uint16;

	typedef uint16					lowp_uint16_t;
	typedef uint16					mediump_uint16_t;
	typedef uint16					highp_uint16_t;
	typedef uint16					uint16_t;

	typedef uint32					lowp_u32;
	typedef uint32					mediump_u32;
	typedef uint32					highp_u32;
	typedef uint32					u32;

	typedef uint32					lowp_uint32;
	typedef uint32					mediump_uint32;
	typedef uint32					highp_uint32;

	typedef uint32					lowp_uint32_t;
	typedef uint32					mediump_uint32_t;
	typedef uint32					highp_uint32_t;
	typedef uint32					uint32_t;

	typedef uint64					lowp_u64;
	typedef uint64					mediump_u64;
	typedef uint64					highp_u64;
	typedef uint64					u64;

	typedef uint64					lowp_uint64;
	typedef uint64					mediump_uint64;
	typedef uint64					highp_uint64;

	typedef uint64					lowp_uint64_t;
	typedef uint64					mediump_uint64_t;
	typedef uint64					highp_uint64_t;
	typedef uint64					uint64_t;

	// Scalar float

	typedef float					lowp_f32;
	typedef float					mediump_f32;
	typedef float					highp_f32;
	typedef float					f32;

	typedef float					lowp_float32;
	typedef float					mediump_float32;
	typedef float					highp_float32;
	typedef float					float32;

	typedef float					lowp_float32_t;
	typedef float					mediump_float32_t;
	typedef float					highp_float32_t;
	typedef float					float32_t;


	typedef double					lowp_f64;
	typedef double					mediump_f64;
	typedef double					highp_f64;
	typedef double					f64;

	typedef double					lowp_float64;
	typedef double					mediump_float64;
	typedef double					highp_float64;
	typedef double					float64;

	typedef double					lowp_float64_t;
	typedef double					mediump_float64_t;
	typedef double					highp_float64_t;
	typedef double					float64_t;

	// Vector bool

	typedef vec<1, bool, lowp>		lowp_bvec1;
	typedef vec<2, bool, lowp>		lowp_bvec2;
	typedef vec<3, bool, lowp>		lowp_bvec3;
	typedef vec<4, bool, lowp>		lowp_bvec4;

	typedef vec<1, bool, mediump>	mediump_bvec1;
	typedef vec<2, bool, mediump>	mediump_bvec2;
	typedef vec<3, bool, mediump>	mediump_bvec3;
	typedef vec<4, bool, mediump>	mediump_bvec4;

	typedef vec<1, bool, highp>		highp_bvec1;
	typedef vec<2, bool, highp>		highp_bvec2;
	typedef vec<3, bool, highp>		highp_bvec3;
	typedef vec<4, bool, highp>		highp_bvec4;

	typedef vec<1, bool, defaultp>	bvec1;
	typedef vec<2, bool, defaultp>	bvec2;
	typedef vec<3, bool, defaultp>	bvec3;
	typedef vec<4, bool, defaultp>	bvec4;

	// Vector int

	typedef vec<1, int, lowp>		lowp_ivec1;
	typedef vec<2, int, lowp>		lowp_ivec2;
	typedef vec<3, int, lowp>		lowp_ivec3;
	typedef vec<4, int, lowp>		lowp_ivec4;

	typedef vec<1, int, mediump>	mediump_ivec1;
	typedef vec<2, int, mediump>	mediump_ivec2;
	typedef vec<3, int, mediump>	mediump_ivec3;
	typedef vec<4, int, mediump>	mediump_ivec4;

	typedef vec<1, int, highp>		highp_ivec1;
	typedef vec<2, int, highp>		highp_ivec2;
	typedef vec<3, int, highp>		highp_ivec3;
	typedef vec<4, int, highp>		highp_ivec4;

	typedef vec<1, int, defaultp>	ivec1;
	typedef vec<2, int, defaultp>	ivec2;
	typedef vec<3, int, defaultp>	ivec3;
	typedef vec<4, int, defaultp>	ivec4;

	typedef vec<1, i8, lowp>		lowp_i8vec1;
	typedef vec<2, i8, lowp>		lowp_i8vec2;
	typedef vec<3, i8, lowp>		lowp_i8vec3;
	typedef vec<4, i8, lowp>		lowp_i8vec4;

	typedef vec<1, i8, mediump>		mediump_i8vec1;
	typedef vec<2, i8, mediump>		mediump_i8vec2;
	typedef vec<3, i8, mediump>		mediump_i8vec3;
	typedef vec<4, i8, mediump>		mediump_i8vec4;

	typedef vec<1, i8, highp>		highp_i8vec1;
	typedef vec<2, i8, highp>		highp_i8vec2;
	typedef vec<3, i8, highp>		highp_i8vec3;
	typedef vec<4, i8, highp>		highp_i8vec4;

	typedef vec<1, i8, defaultp>	i8vec1;
	typedef vec<2, i8, defaultp>	i8vec2;
	typedef vec<3, i8, defaultp>	i8vec3;
	typedef vec<4, i8, defaultp>	i8vec4;

	typedef vec<1, i16, lowp>		lowp_i16vec1;
	typedef vec<2, i16, lowp>		lowp_i16vec2;
	typedef vec<3, i16, lowp>		lowp_i16vec3;
	typedef vec<4, i16, lowp>		lowp_i16vec4;

	typedef vec<1, i16, mediump>	mediump_i16vec1;
	typedef vec<2, i16, mediump>	mediump_i16vec2;
	typedef vec<3, i16, mediump>	mediump_i16vec3;
	typedef vec<4, i16, mediump>	mediump_i16vec4;

	typedef vec<1, i16, highp>		highp_i16vec1;
	typedef vec<2, i16, highp>		highp_i16vec2;
	typedef vec<3, i16, highp>		highp_i16vec3;
	typedef vec<4, i16, highp>		highp_i16vec4;

	typedef vec<1, i16, defaultp>	i16vec1;
	typedef vec<2, i16, defaultp>	i16vec2;
	typedef vec<3, i16, defaultp>	i16vec3;
	typedef vec<4, i16, defaultp>	i16vec4;

	typedef vec<1, i32, lowp>		lowp_i32vec1;
	typedef vec<2, i32, lowp>		lowp_i32vec2;
	typedef vec<3, i32, lowp>		lowp_i32vec3;
	typedef vec<4, i32, lowp>		lowp_i32vec4;

	typedef vec<1, i32, mediump>	mediump_i32vec1;
	typedef vec<2, i32, mediump>	mediump_i32vec2;
	typedef vec<3, i32, mediump>	mediump_i32vec3;
	typedef vec<4, i32, mediump>	mediump_i32vec4;

	typedef vec<1, i32, highp>		highp_i32vec1;
	typedef vec<2, i32, highp>		highp_i32vec2;
	typedef vec<3, i32, highp>		highp_i32vec3;
	typedef vec<4, i32, highp>		highp_i32vec4;

	typedef vec<1, i32, defaultp>	i32vec1;
	typedef vec<2, i32, defaultp>	i32vec2;
	typedef vec<3, i32, defaultp>	i32vec3;
	typedef vec<4, i32, defaultp>	i32vec4;

	typedef vec<1, i64, lowp>		lowp_i64vec1;
	typedef vec<2, i64, lowp>		lowp_i64vec2;
	typedef vec<3, i64, lowp>		lowp_i64vec3;
	typedef vec<4, i64, lowp>		lowp_i64vec4;

	typedef vec<1, i64, mediump>	mediump_i64vec1;
	typedef vec<2, i64, mediump>	mediump_i64vec2;
	typedef vec<3, i64, mediump>	mediump_i64vec3;
	typedef vec<4, i64, mediump>	mediump_i64vec4;

	typedef vec<1, i64, highp>		highp_i64vec1;
	typedef vec<2, i64, highp>		highp_i64vec2;
	typedef vec<3, i64, highp>		highp_i64vec3;
	typedef vec<4, i64, highp>		highp_i64vec4;

	typedef vec<1, i64, defaultp>	i64vec1;
	typedef vec<2, i64, defaultp>	i64vec2;
	typedef vec<3, i64, defaultp>	i64vec3;
	typedef vec<4, i64, defaultp>	i64vec4;

	// Vector uint

	typedef vec<1, uint, lowp>		lowp_uvec1;
	typedef vec<2, uint, lowp>		lowp_uvec2;
	typedef vec<3, uint, lowp>		lowp_uvec3;
	typedef vec<4, uint, lowp>		lowp_uvec4;

	typedef vec<1, uint, mediump>	mediump_uvec1;
	typedef vec<2, uint, mediump>	mediump_uvec2;
	typedef vec<3, uint, mediump>	mediump_uvec3;
	typedef vec<4, uint, mediump>	mediump_uvec4;

	typedef vec<1, uint, highp>		highp_uvec1;
	typedef vec<2, uint, highp>		highp_uvec2;
	typedef vec<3, uint, highp>		highp_uvec3;
	typedef vec<4, uint, highp>		highp_uvec4;

	typedef vec<1, uint, defaultp>	uvec1;
	typedef vec<2, uint, defaultp>	uvec2;
	typedef vec<3, uint, defaultp>	uvec3;
	typedef vec<4, uint, defaultp>	uvec4;

	typedef vec<1, u8, lowp>		lowp_u8vec1;
	typedef vec<2, u8, lowp>		lowp_u8vec2;
	typedef vec<3, u8, lowp>		lowp_u8vec3;
	typedef vec<4, u8, lowp>		lowp_u8vec4;

	typedef vec<1, u8, mediump>		mediump_u8vec1;
	typedef vec<2, u8, mediump>		mediump_u8vec2;
	typedef vec<3, u8, mediump>		mediump_u8vec3;
	typedef vec<4, u8, mediump>		mediump_u8vec4;

	typedef vec<1, u8, highp>		highp_u8vec1;
	typedef vec<2, u8, highp>		highp_u8vec2;
	typedef vec<3, u8, highp>		highp_u8vec3;
	typedef vec<4, u8, highp>		highp_u8vec4;

	typedef vec<1, u8, defaultp>	u8vec1;
	typedef vec<2, u8, defaultp>	u8vec2;
	typedef vec<3, u8, defaultp>	u8vec3;
	typedef vec<4, u8, defaultp>	u8vec4;

	typedef vec<1, u16, lowp>		lowp_u16vec1;
	typedef vec<2, u16, lowp>		lowp_u16vec2;
	typedef vec<3, u16, lowp>		lowp_u16vec3;
	typedef vec<4, u16, lowp>		lowp_u16vec4;

	typedef vec<1, u16, mediump>	mediump_u16vec1;
	typedef vec<2, u16, mediump>	mediump_u16vec2;
	typedef vec<3, u16, mediump>	mediump_u16vec3;
	typedef vec<4, u16, mediump>	mediump_u16vec4;

	typedef vec<1, u16, highp>		highp_u16vec1;
	typedef vec<2, u16, highp>		highp_u16vec2;
	typedef vec<3, u16, highp>		highp_u16vec3;
	typedef vec<4, u16, highp>		highp_u16vec4;

	typedef vec<1, u16, defaultp>	u16vec1;
	typedef vec<2, u16, defaultp>	u16vec2;
	typedef vec<3, u16, defaultp>	u16vec3;
	typedef vec<4, u16, defaultp>	u16vec4;

	typedef vec<1, u32, lowp>		lowp_u32vec1;
	typedef vec<2, u32, lowp>		lowp_u32vec2;
	typedef vec<3, u32, lowp>		lowp_u32vec3;
	typedef vec<4, u32, lowp>		lowp_u32vec4;

	typedef vec<1, u32, mediump>	mediump_u32vec1;
	typedef vec<2, u32, mediump>	mediump_u32vec2;
	typedef vec<3, u32, mediump>	mediump_u32vec3;
	typedef vec<4, u32, mediump>	mediump_u32vec4;

	typedef vec<1, u32, highp>		highp_u32vec1;
	typedef vec<2, u32, highp>		highp_u32vec2;
	typedef vec<3, u32, highp>		highp_u32vec3;
	typedef vec<4, u32, highp>		highp_u32vec4;

	typedef vec<1, u32, defaultp>	u32vec1;
	typedef vec<2, u32, defaultp>	u32vec2;
	typedef vec<3, u32, defaultp>	u32vec3;
	typedef vec<4, u32, defaultp>	u32vec4;

	typedef vec<1, u64, lowp>		lowp_u64vec1;
	typedef vec<2, u64, lowp>		lowp_u64vec2;
	typedef vec<3, u64, lowp>		lowp_u64vec3;
	typedef vec<4, u64, lowp>		lowp_u64vec4;

	typedef vec<1, u64, mediump>	mediump_u64vec1;
	typedef vec<2, u64, mediump>	mediump_u64vec2;
	typedef vec<3, u64, mediump>	mediump_u64vec3;
	typedef vec<4, u64, mediump>	mediump_u64vec4;

	typedef vec<1, u64, highp>		highp_u64vec1;
	typedef vec<2, u64, highp>		highp_u64vec2;
	typedef vec<3, u64, highp>		highp_u64vec3;
	typedef vec<4, u64, highp>		highp_u64vec4;

	typedef vec<1, u64, defaultp>	u64vec1;
	typedef vec<2, u64, defaultp>	u64vec2;
	typedef vec<3, u64, defaultp>	u64vec3;
	typedef vec<4, u64, defaultp>	u64vec4;

	// Vector float

	typedef vec<1, float, lowp>			lowp_vec1;
	typedef vec<2, float, lowp>			lowp_vec2;
	typedef vec<3, float, lowp>			lowp_vec3;
	typedef vec<4, float, lowp>			lowp_vec4;

	typedef vec<1, float, mediump>		mediump_vec1;
	typedef vec<2, float, mediump>		mediump_vec2;
	typedef vec<3, float, mediump>		mediump_vec3;
	typedef vec<4, float, mediump>		mediump_vec4;

	typedef vec<1, float, highp>		highp_vec1;
	typedef vec<2, float, highp>		highp_vec2;
	typedef vec<3, float, highp>		highp_vec3;
	typedef vec<4, float, highp>		highp_vec4;

	typedef vec<1, float, defaultp>		vec1;
	typedef vec<2, float, defaultp>		vec2;
	typedef vec<3, float, defaultp>		vec3;
	typedef vec<4, float, defaultp>		vec4;

	typedef vec<1, float, lowp>			lowp_fvec1;
	typedef vec<2, float, lowp>			lowp_fvec2;
	typedef vec<3, float, lowp>			lowp_fvec3;
	typedef vec<4, float, lowp>			lowp_fvec4;

	typedef vec<1, float, mediump>		mediump_fvec1;
	typedef vec<2, float, mediump>		mediump_fvec2;
	typedef vec<3, float, mediump>		mediump_fvec3;
	typedef vec<4, float, mediump>		mediump_fvec4;

	typedef vec<1, float, highp>		highp_fvec1;
	typedef vec<2, float, highp>		highp_fvec2;
	typedef vec<3, float, highp>		highp_fvec3;
	typedef vec<4, float, highp>		highp_fvec4;

	typedef vec<1, f32, defaultp>		fvec1;
	typedef vec<2, f32, defaultp>		fvec2;
	typedef vec<3, f32, defaultp>		fvec3;
	typedef vec<4, f32, defaultp>		fvec4;

	typedef vec<1, f32, lowp>			lowp_f32vec1;
	typedef vec<2, f32, lowp>			lowp_f32vec2;
	typedef vec<3, f32, lowp>			lowp_f32vec3;
	typedef vec<4, f32, lowp>			lowp_f32vec4;

	typedef vec<1, f32, mediump>		mediump_f32vec1;
	typedef vec<2, f32, mediump>		mediump_f32vec2;
	typedef vec<3, f32, mediump>		mediump_f32vec3;
	typedef vec<4, f32, mediump>		mediump_f32vec4;

	typedef vec<1, f32, highp>			highp_f32vec1;
	typedef vec<2, f32, highp>			highp_f32vec2;
	typedef vec<3, f32, highp>			highp_f32vec3;
	typedef vec<4, f32, highp>			highp_f32vec4;

	typedef vec<1, f32, defaultp>		f32vec1;
	typedef vec<2, f32, defaultp>		f32vec2;
	typedef vec<3, f32, defaultp>		f32vec3;
	typedef vec<4, f32, defaultp>		f32vec4;

	typedef vec<1, f64, lowp>			lowp_dvec1;
	typedef vec<2, f64, lowp>			lowp_dvec2;
	typedef vec<3, f64, lowp>			lowp_dvec3;
	typedef vec<4, f64, lowp>			lowp_dvec4;

	typedef vec<1, f64, mediump>		mediump_dvec1;
	typedef vec<2, f64, mediump>		mediump_dvec2;
	typedef vec<3, f64, mediump>		mediump_dvec3;
	typedef vec<4, f64, mediump>		mediump_dvec4;

	typedef vec<1, f64, highp>			highp_dvec1;
	typedef vec<2, f64, highp>			highp_dvec2;
	typedef vec<3, f64, highp>			highp_dvec3;
	typedef vec<4, f64, highp>			highp_dvec4;

	typedef vec<1, f64, defaultp>		dvec1;
	typedef vec<2, f64, defaultp>		dvec2;
	typedef vec<3, f64, defaultp>		dvec3;
	typedef vec<4, f64, defaultp>		dvec4;

	typedef vec<1, f64, lowp>			lowp_f64vec1;
	typedef vec<2, f64, lowp>			lowp_f64vec2;
	typedef vec<3, f64, lowp>			lowp_f64vec3;
	typedef vec<4, f64, lowp>			lowp_f64vec4;

	typedef vec<1, f64, mediump>		mediump_f64vec1;
	typedef vec<2, f64, mediump>		mediump_f64vec2;
	typedef vec<3, f64, mediump>		mediump_f64vec3;
	typedef vec<4, f64, mediump>		mediump_f64vec4;

	typedef vec<1, f64, highp>			highp_f64vec1;
	typedef vec<2, f64, highp>			highp_f64vec2;
	typedef vec<3, f64, highp>			highp_f64vec3;
	typedef vec<4, f64, highp>			highp_f64vec4;

	typedef vec<1, f64, defaultp>		f64vec1;
	typedef vec<2, f64, defaultp>		f64vec2;
	typedef vec<3, f64, defaultp>		f64vec3;
	typedef vec<4, f64, defaultp>		f64vec4;

	// Matrix NxN

	typedef mat<2, 2, f32, lowp>		lowp_mat2;
	typedef mat<3, 3, f32, lowp>		lowp_mat3;
	typedef mat<4, 4, f32, lowp>		lowp_mat4;

	typedef mat<2, 2, f32, mediump>		mediump_mat2;
	typedef mat<3, 3, f32, mediump>		mediump_mat3;
	typedef mat<4, 4, f32, mediump>		mediump_mat4;

	typedef mat<2, 2, f32, highp>		highp_mat2;
	typedef mat<3, 3, f32, highp>		highp_mat3;
	typedef mat<4, 4, f32, highp>		highp_mat4;

	typedef mat<2, 2, f32, defaultp>	mat2;
	typedef mat<3, 3, f32, defaultp>	mat3;
	typedef mat<4, 4, f32, defaultp>	mat4;

	typedef mat<2, 2, f32, lowp>		lowp_fmat2;
	typedef mat<3, 3, f32, lowp>		lowp_fmat3;
	typedef mat<4, 4, f32, lowp>		lowp_fmat4;

	typedef mat<2, 2, f32, mediump>		mediump_fmat2;
	typedef mat<3, 3, f32, mediump>		mediump_fmat3;
	typedef mat<4, 4, f32, mediump>		mediump_fmat4;

	typedef mat<2, 2, f32, highp>		highp_fmat2;
	typedef mat<3, 3, f32, highp>		highp_fmat3;
	typedef mat<4, 4, f32, highp>		highp_fmat4;

	typedef mat<2, 2, f32, defaultp>	fmat2;
	typedef mat<3, 3, f32, defaultp>	fmat3;
	typedef mat<4, 4, f32, defaultp>	fmat4;

	typedef mat<2, 2, f32, lowp>		lowp_f32mat2;
	typedef mat<3, 3, f32, lowp>		lowp_f32mat3;
	typedef mat<4, 4, f32, lowp>		lowp_f32mat4;

	typedef mat<2, 2, f32, mediump>		mediump_f32mat2;
	typedef mat<3, 3, f32, mediump>		mediump_f32mat3;
	typedef mat<4, 4, f32, mediump>		mediump_f32mat4;

	typedef mat<2, 2, f32, highp>		highp_f32mat2;
	typedef mat<3, 3, f32, highp>		highp_f32mat3;
	typedef mat<4, 4, f32, highp>		highp_f32mat4;

	typedef mat<2, 2, f32, defaultp>	f32mat2;
	typedef mat<3, 3, f32, defaultp>	f32mat3;
	typedef mat<4, 4, f32, defaultp>	f32mat4;

	typedef mat<2, 2, f64, lowp>		lowp_dmat2;
	typedef mat<3, 3, f64, lowp>		lowp_dmat3;
	typedef mat<4, 4, f64, lowp>		lowp_dmat4;

	typedef mat<2, 2, f64, mediump>		mediump_dmat2;
	typedef mat<3, 3, f64, mediump>		mediump_dmat3;
	typedef mat<4, 4, f64, mediump>		mediump_dmat4;

	typedef mat<2, 2, f64, highp>		highp_dmat2;
	typedef mat<3, 3, f64, highp>		highp_dmat3;
	typedef mat<4, 4, f64, highp>		highp_dmat4;

	typedef mat<2, 2, f64, defaultp>	dmat2;
	typedef mat<3, 3, f64, defaultp>	dmat3;
	typedef mat<4, 4, f64, defaultp>	dmat4;

	typedef mat<2, 2, f64, lowp>		lowp_f64mat2;
	typedef mat<3, 3, f64, lowp>		lowp_f64mat3;
	typedef mat<4, 4, f64, lowp>		lowp_f64mat4;

	typedef mat<2, 2, f64, mediump>		mediump_f64mat2;
	typedef mat<3, 3, f64, mediump>		mediump_f64mat3;
	typedef mat<4, 4, f64, mediump>		mediump_f64mat4;

	typedef mat<2, 2, f64, highp>		highp_f64mat2;
	typedef mat<3, 3, f64, highp>		highp_f64mat3;
	typedef mat<4, 4, f64, highp>		highp_f64mat4;

	typedef mat<2, 2, f64, defaultp>	f64mat2;
	typedef mat<3, 3, f64, defaultp>	f64mat3;
	typedef mat<4, 4, f64, defaultp>	f64mat4;

	// Matrix MxN

	typedef mat<2, 2, f32, lowp>		lowp_mat2x2;
	typedef mat<2, 3, f32, lowp>		lowp_mat2x3;
	typedef mat<2, 4, f32, lowp>		lowp_mat2x4;
	typedef mat<3, 2, f32, lowp>		lowp_mat3x2;
	typedef mat<3, 3, f32, lowp>		lowp_mat3x3;
	typedef mat<3, 4, f32, lowp>		lowp_mat3x4;
	typedef mat<4, 2, f32, lowp>		lowp_mat4x2;
	typedef mat<4, 3, f32, lowp>		lowp_mat4x3;
	typedef mat<4, 4, f32, lowp>		lowp_mat4x4;

	typedef mat<2, 2, f32, mediump>		mediump_mat2x2;
	typedef mat<2, 3, f32, mediump>		mediump_mat2x3;
	typedef mat<2, 4, f32, mediump>		mediump_mat2x4;
	typedef mat<3, 2, f32, mediump>		mediump_mat3x2;
	typedef mat<3, 3, f32, mediump>		mediump_mat3x3;
	typedef mat<3, 4, f32, mediump>		mediump_mat3x4;
	typedef mat<4, 2, f32, mediump>		mediump_mat4x2;
	typedef mat<4, 3, f32, mediump>		mediump_mat4x3;
	typedef mat<4, 4, f32, mediump>		mediump_mat4x4;

	typedef mat<2, 2, f32, highp>		highp_mat2x2;
	typedef mat<2, 3, f32, highp>		highp_mat2x3;
	typedef mat<2, 4, f32, highp>		highp_mat2x4;
	typedef mat<3, 2, f32, highp>		highp_mat3x2;
	typedef mat<3, 3, f32, highp>		highp_mat3x3;
	typedef mat<3, 4, f32, highp>		highp_mat3x4;
	typedef mat<4, 2, f32, highp>		highp_mat4x2;
	typedef mat<4, 3, f32, highp>		highp_mat4x3;
	typedef mat<4, 4, f32, highp>		highp_mat4x4;

	typedef mat<2, 2, f32, defaultp>	mat2x2;
	typedef mat<2, 3, f32, defaultp>	mat2x3;
	typedef mat<2, 4, f32, defaultp>	mat2x4;
	typedef mat<3, 2, f32, defaultp>	mat3x2;
	typedef mat<3, 3, f32, defaultp>	mat3x3;
	typedef mat<3, 4, f32, defaultp>	mat3x4;
	typedef mat<4, 2, f32, defaultp>	mat4x2;
	typedef mat<4, 3, f32, defaultp>	mat4x3;
	typedef mat<4, 4, f32, defaultp>	mat4x4;

	typedef mat<2, 2, f32, lowp>		lowp_fmat2x2;
	typedef mat<2, 3, f32, lowp>		lowp_fmat2x3;
	typedef mat<2, 4, f32, lowp>		lowp_fmat2x4;
	typedef mat<3, 2, f32, lowp>		lowp_fmat3x2;
	typedef mat<3, 3, f32, lowp>		lowp_fmat3x3;
	typedef mat<3, 4, f32, lowp>		lowp_fmat3x4;
	typedef mat<4, 2, f32, lowp>		lowp_fmat4x2;
	typedef mat<4, 3, f32, lowp>		lowp_fmat4x3;
	typedef mat<4, 4, f32, lowp>		lowp_fmat4x4;

	typedef mat<2, 2, f32, mediump>		mediump_fmat2x2;
	typedef mat<2, 3, f32, mediump>		mediump_fmat2x3;
	typedef mat<2, 4, f32, mediump>		mediump_fmat2x4;
	typedef mat<3, 2, f32, mediump>		mediump_fmat3x2;
	typedef mat<3, 3, f32, mediump>		mediump_fmat3x3;
	typedef mat<3, 4, f32, mediump>		mediump_fmat3x4;
	typedef mat<4, 2, f32, mediump>		mediump_fmat4x2;
	typedef mat<4, 3, f32, mediump>		mediump_fmat4x3;
	typedef mat<4, 4, f32, mediump>		mediump_fmat4x4;

	typedef mat<2, 2, f32, highp>		highp_fmat2x2;
	typedef mat<2, 3, f32, highp>		highp_fmat2x3;
	typedef mat<2, 4, f32, highp>		highp_fmat2x4;
	typedef mat<3, 2, f32, highp>		highp_fmat3x2;
	typedef mat<3, 3, f32, highp>		highp_fmat3x3;
	typedef mat<3, 4, f32, highp>		highp_fmat3x4;
	typedef mat<4, 2, f32, highp>		highp_fmat4x2;
	typedef mat<4, 3, f32, highp>		highp_fmat4x3;
	typedef mat<4, 4, f32, highp>		highp_fmat4x4;

	typedef mat<2, 2, f32, defaultp>	fmat2x2;
	typedef mat<2, 3, f32, defaultp>	fmat2x3;
	typedef mat<2, 4, f32, defaultp>	fmat2x4;
	typedef mat<3, 2, f32, defaultp>	fmat3x2;
	typedef mat<3, 3, f32, defaultp>	fmat3x3;
	typedef mat<3, 4, f32, defaultp>	fmat3x4;
	typedef mat<4, 2, f32, defaultp>	fmat4x2;
	typedef mat<4, 3, f32, defaultp>	fmat4x3;
	typedef mat<4, 4, f32, defaultp>	fmat4x4;

	typedef mat<2, 2, f32, lowp>		lowp_f32mat2x2;
	typedef mat<2, 3, f32, lowp>		lowp_f32mat2x3;
	typedef mat<2, 4, f32, lowp>		lowp_f32mat2x4;
	typedef mat<3, 2, f32, lowp>		lowp_f32mat3x2;
	typedef mat<3, 3, f32, lowp>		lowp_f32mat3x3;
	typedef mat<3, 4, f32, lowp>		lowp_f32mat3x4;
	typedef mat<4, 2, f32, lowp>		lowp_f32mat4x2;
	typedef mat<4, 3, f32, lowp>		lowp_f32mat4x3;
	typedef mat<4, 4, f32, lowp>		lowp_f32mat4x4;
	
	typedef mat<2, 2, f32, mediump>		mediump_f32mat2x2;
	typedef mat<2, 3, f32, mediump>		mediump_f32mat2x3;
	typedef mat<2, 4, f32, mediump>		mediump_f32mat2x4;
	typedef mat<3, 2, f32, mediump>		mediump_f32mat3x2;
	typedef mat<3, 3, f32, mediump>		mediump_f32mat3x3;
	typedef mat<3, 4, f32, mediump>		mediump_f32mat3x4;
	typedef mat<4, 2, f32, mediump>		mediump_f32mat4x2;
	typedef mat<4, 3, f32, mediump>		mediump_f32mat4x3;
	typedef mat<4, 4, f32, mediump>		mediump_f32mat4x4;

	typedef mat<2, 2, f32, highp>		highp_f32mat2x2;
	typedef mat<2, 3, f32, highp>		highp_f32mat2x3;
	typedef mat<2, 4, f32, highp>		highp_f32mat2x4;
	typedef mat<3, 2, f32, highp>		highp_f32mat3x2;
	typedef mat<3, 3, f32, highp>		highp_f32mat3x3;
	typedef mat<3, 4, f32, highp>		highp_f32mat3x4;
	typedef mat<4, 2, f32, highp>		highp_f32mat4x2;
	typedef mat<4, 3, f32, highp>		highp_f32mat4x3;
	typedef mat<4, 4, f32, highp>		highp_f32mat4x4;

	typedef mat<2, 2, f32, defaultp>	f32mat2x2;
	typedef mat<2, 3, f32, defaultp>	f32mat2x3;
	typedef mat<2, 4, f32, defaultp>	f32mat2x4;
	typedef mat<3, 2, f32, defaultp>	f32mat3x2;
	typedef mat<3, 3, f32, defaultp>	f32mat3x3;
	typedef mat<3, 4, f32, defaultp>	f32mat3x4;
	typedef mat<4, 2, f32, defaultp>	f32mat4x2;
	typedef mat<4, 3, f32, defaultp>	f32mat4x3;
	typedef mat<4, 4, f32, defaultp>	f32mat4x4;

	typedef mat<2, 2, double, lowp>		lowp_dmat2x2;
	typedef mat<2, 3, double, lowp>		lowp_dmat2x3;
	typedef mat<2, 4, double, lowp>		lowp_dmat2x4;
	typedef mat<3, 2, double, lowp>		lowp_dmat3x2;
	typedef mat<3, 3, double, lowp>		lowp_dmat3x3;
	typedef mat<3, 4, double, lowp>		lowp_dmat3x4;
	typedef mat<4, 2, double, lowp>		lowp_dmat4x2;
	typedef mat<4, 3, double, lowp>		lowp_dmat4x3;
	typedef mat<4, 4, double, lowp>		lowp_dmat4x4;

	typedef mat<2, 2, double, mediump>	mediump_dmat2x2;
	typedef mat<2, 3, double, mediump>	mediump_dmat2x3;
	typedef mat<2, 4, double, mediump>	mediump_dmat2x4;
	typedef mat<3, 2, double, mediump>	mediump_dmat3x2;
	typedef mat<3, 3, double, mediump>	mediump_dmat3x3;
	typedef mat<3, 4, double, mediump>	mediump_dmat3x4;
	typedef mat<4, 2, double, mediump>	mediump_dmat4x2;
	typedef mat<4, 3, double, mediump>	mediump_dmat4x3;
	typedef mat<4, 4, double, mediump>	mediump_dmat4x4;

	typedef mat<2, 2, double, highp>	highp_dmat2x2;
	typedef mat<2, 3, double, highp>	highp_dmat2x3;
	typedef mat<2, 4, double, highp>	highp_dmat2x4;
	typedef mat<3, 2, double, highp>	highp_dmat3x2;
	typedef mat<3, 3, double, highp>	highp_dmat3x3;
	typedef mat<3, 4, double, highp>	highp_dmat3x4;
	typedef mat<4, 2, double, highp>	highp_dmat4x2;
	typedef mat<4, 3, double, highp>	highp_dmat4x3;
	typedef mat<4, 4, double, highp>	highp_dmat4x4;

	typedef mat<2, 2, double, defaultp>	dmat2x2;
	typedef mat<2, 3, double, defaultp>	dmat2x3;
	typedef mat<2, 4, double, defaultp>	dmat2x4;
	typedef mat<3, 2, double, defaultp>	dmat3x2;
	typedef mat<3, 3, double, defaultp>	dmat3x3;
	typedef mat<3, 4, double, defaultp>	dmat3x4;
	typedef mat<4, 2, double, defaultp>	dmat4x2;
	typedef mat<4, 3, double, defaultp>	dmat4x3;
	typedef mat<4, 4, double, defaultp>	dmat4x4;

	typedef mat<2, 2, f64, lowp>		lowp_f64mat2x2;
	typedef mat<2, 3, f64, lowp>		lowp_f64mat2x3;
	typedef mat<2, 4, f64, lowp>		lowp_f64mat2x4;
	typedef mat<3, 2, f64, lowp>		lowp_f64mat3x2;
	typedef mat<3, 3, f64, lowp>		lowp_f64mat3x3;
	typedef mat<3, 4, f64, lowp>		lowp_f64mat3x4;
	typedef mat<4, 2, f64, lowp>		lowp_f64mat4x2;
	typedef mat<4, 3, f64, lowp>		lowp_f64mat4x3;
	typedef mat<4, 4, f64, lowp>		lowp_f64mat4x4;

	typedef mat<2, 2, f64, mediump>		mediump_f64mat2x2;
	typedef mat<2, 3, f64, mediump>		mediump_f64mat2x3;
	typedef mat<2, 4, f64, mediump>		mediump_f64mat2x4;
	typedef mat<3, 2, f64, mediump>		mediump_f64mat3x2;
	typedef mat<3, 3, f64, mediump>		mediump_f64mat3x3;
	typedef mat<3, 4, f64, mediump>		mediump_f64mat3x4;
	typedef mat<4, 2, f64, mediump>		mediump_f64mat4x2;
	typedef mat<4, 3, f64, mediump>		mediump_f64mat4x3;
	typedef mat<4, 4, f64, mediump>		mediump_f64mat4x4;

	typedef mat<2, 2, f64, highp>		highp_f64mat2x2;
	typedef mat<2, 3, f64, highp>		highp_f64mat2x3;
	typedef mat<2, 4, f64, highp>		highp_f64mat2x4;
	typedef mat<3, 2, f64, highp>		highp_f64mat3x2;
	typedef mat<3, 3, f64, highp>		highp_f64mat3x3;
	typedef mat<3, 4, f64, highp>		highp_f64mat3x4;
	typedef mat<4, 2, f64, highp>		highp_f64mat4x2;
	typedef mat<4, 3, f64, highp>		highp_f64mat4x3;
	typedef mat<4, 4, f64, highp>		highp_f64mat4x4;

	typedef mat<2, 2, f64, defaultp>	f64mat2x2;
	typedef mat<2, 3, f64, defaultp>	f64mat2x3;
	typedef mat<2, 4, f64, defaultp>	f64mat2x4;
	typedef mat<3, 2, f64, defaultp>	f64mat3x2;
	typedef mat<3, 3, f64, defaultp>	f64mat3x3;
	typedef mat<3, 4, f64, defaultp>	f64mat3x4;
	typedef mat<4, 2, f64, defaultp>	f64mat4x2;
	typedef mat<4, 3, f64, defaultp>	f64mat4x3;
	typedef mat<4, 4, f64, defaultp>	f64mat4x4;

	// Signed integer matrix MxN

	typedef mat<2, 2, int, lowp>		lowp_imat2x2;
	typedef mat<2, 3, int, lowp>		lowp_imat2x3;
	typedef mat<2, 4, int, lowp>		lowp_imat2x4;
	typedef mat<3, 2, int, lowp>		lowp_imat3x2;
	typedef mat<3, 3, int, lowp>		lowp_imat3x3;
	typedef mat<3, 4, int, lowp>		lowp_imat3x4;
	typedef mat<4, 2, int, lowp>		lowp_imat4x2;
	typedef mat<4, 3, int, lowp>		lowp_imat4x3;
	typedef mat<4, 4, int, lowp>		lowp_imat4x4;

	typedef mat<2, 2, int, mediump>		mediump_imat2x2;
	typedef mat<2, 3, int, mediump>		mediump_imat2x3;
	typedef mat<2, 4, int, mediump>		mediump_imat2x4;
	typedef mat<3, 2, int, mediump>		mediump_imat3x2;
	typedef mat<3, 3, int, mediump>		mediump_imat3x3;
	typedef mat<3, 4, int, mediump>		mediump_imat3x4;
	typedef mat<4, 2, int, mediump>		mediump_imat4x2;
	typedef mat<4, 3, int, mediump>		mediump_imat4x3;
	typedef mat<4, 4, int, mediump>		mediump_imat4x4;

	typedef mat<2, 2, int, highp>		highp_imat2x2;
	typedef mat<2, 3, int, highp>		highp_imat2x3;
	typedef mat<2, 4, int, highp>		highp_imat2x4;
	typedef mat<3, 2, int, highp>		highp_imat3x2;
	typedef mat<3, 3, int, highp>		highp_imat3x3;
	typedef mat<3, 4, int, highp>		highp_imat3x4;
	typedef mat<4, 2, int, highp>		highp_imat4x2;
	typedef mat<4, 3, int, highp>		highp_imat4x3;
	typedef mat<4, 4, int, highp>		highp_imat4x4;

	typedef mat<2, 2, int, defaultp>	imat2x2;
	typedef mat<2, 3, int, defaultp>	imat2x3;
	typedef mat<2, 4, int, defaultp>	imat2x4;
	typedef mat<3, 2, int, defaultp>	imat3x2;
	typedef mat<3, 3, int, defaultp>	imat3x3;
	typedef mat<3, 4, int, defaultp>	imat3x4;
	typedef mat<4, 2, int, defaultp>	imat4x2;
	typedef mat<4, 3, int, defaultp>	imat4x3;
	typedef mat<4, 4, int, defaultp>	imat4x4;


	typedef mat<2, 2, int8, lowp>		lowp_i8mat2x2;
	typedef mat<2, 3, int8, lowp>		lowp_i8mat2x3;
	typedef mat<2, 4, int8, lowp>		lowp_i8mat2x4;
	typedef mat<3, 2, int8, lowp>		lowp_i8mat3x2;
	typedef mat<3, 3, int8, lowp>		lowp_i8mat3x3;
	typedef mat<3, 4, int8, lowp>		lowp_i8mat3x4;
	typedef mat<4, 2, int8, lowp>		lowp_i8mat4x2;
	typedef mat<4, 3, int8, lowp>		lowp_i8mat4x3;
	typedef mat<4, 4, int8, lowp>		lowp_i8mat4x4;

	typedef mat<2, 2, int8, mediump>	mediump_i8mat2x2;
	typedef mat<2, 3, int8, mediump>	mediump_i8mat2x3;
	typedef mat<2, 4, int8, mediump>	mediump_i8mat2x4;
	typedef mat<3, 2, int8, mediump>	mediump_i8mat3x2;
	typedef mat<3, 3, int8, mediump>	mediump_i8mat3x3;
	typedef mat<3, 4, int8, mediump>	mediump_i8mat3x4;
	typedef mat<4, 2, int8, mediump>	mediump_i8mat4x2;
	typedef mat<4, 3, int8, mediump>	mediump_i8mat4x3;
	typedef mat<4, 4, int8, mediump>	mediump_i8mat4x4;

	typedef mat<2, 2, int8, highp>		highp_i8mat2x2;
	typedef mat<2, 3, int8, highp>		highp_i8mat2x3;
	typedef mat<2, 4, int8, highp>		highp_i8mat2x4;
	typedef mat<3, 2, int8, highp>		highp_i8mat3x2;
	typedef mat<3, 3, int8, highp>		highp_i8mat3x3;
	typedef mat<3, 4, int8, highp>		highp_i8mat3x4;
	typedef mat<4, 2, int8, highp>		highp_i8mat4x2;
	typedef mat<4, 3, int8, highp>		highp_i8mat4x3;
	typedef mat<4, 4, int8, highp>		highp_i8mat4x4;

	typedef mat<2, 2, int8, defaultp>	i8mat2x2;
	typedef mat<2, 3, int8, defaultp>	i8mat2x3;
	typedef mat<2, 4, int8, defaultp>	i8mat2x4;
	typedef mat<3, 2, int8, defaultp>	i8mat3x2;
	typedef mat<3, 3, int8, defaultp>	i8mat3x3;
	typedef mat<3, 4, int8, defaultp>	i8mat3x4;
	typedef mat<4, 2, int8, defaultp>	i8mat4x2;
	typedef mat<4, 3, int8, defaultp>	i8mat4x3;
	typedef mat<4, 4, int8, defaultp>	i8mat4x4;


	typedef mat<2, 2, int16, lowp>		lowp_i16mat2x2;
	typedef mat<2, 3, int16, lowp>		lowp_i16mat2x3;
	typedef mat<2, 4, int16, lowp>		lowp_i16mat2x4;
	typedef mat<3, 2, int16, lowp>		lowp_i16mat3x2;
	typedef mat<3, 3, int16, lowp>		lowp_i16mat3x3;
	typedef mat<3, 4, int16, lowp>		lowp_i16mat3x4;
	typedef mat<4, 2, int16, lowp>		lowp_i16mat4x2;
	typedef mat<4, 3, int16, lowp>		lowp_i16mat4x3;
	typedef mat<4, 4, int16, lowp>		lowp_i16mat4x4;

	typedef mat<2, 2, int16, mediump>	mediump_i16mat2x2;
	typedef mat<2, 3, int16, mediump>	mediump_i16mat2x3;
	typedef mat<2, 4, int16, mediump>	mediump_i16mat2x4;
	typedef mat<3, 2, int16, mediump>	mediump_i16mat3x2;
	typedef mat<3, 3, int16, mediump>	mediump_i16mat3x3;
	typedef mat<3, 4, int16, mediump>	mediump_i16mat3x4;
	typedef mat<4, 2, int16, mediump>	mediump_i16mat4x2;
	typedef mat<4, 3, int16, mediump>	mediump_i16mat4x3;
	typedef mat<4, 4, int16, mediump>	mediump_i16mat4x4;

	typedef mat<2, 2, int16, highp>		highp_i16mat2x2;
	typedef mat<2, 3, int16, highp>		highp_i16mat2x3;
	typedef mat<2, 4, int16, highp>		highp_i16mat2x4;
	typedef mat<3, 2, int16, highp>		highp_i16mat3x2;
	typedef mat<3, 3, int16, highp>		highp_i16mat3x3;
	typedef mat<3, 4, int16, highp>		highp_i16mat3x4;
	typedef mat<4, 2, int16, highp>		highp_i16mat4x2;
	typedef mat<4, 3, int16, highp>		highp_i16mat4x3;
	typedef mat<4, 4, int16, highp>		highp_i16mat4x4;

	typedef mat<2, 2, int16, defaultp>	i16mat2x2;
	typedef mat<2, 3, int16, defaultp>	i16mat2x3;
	typedef mat<2, 4, int16, defaultp>	i16mat2x4;
	typedef mat<3, 2, int16, defaultp>	i16mat3x2;
	typedef mat<3, 3, int16, defaultp>	i16mat3x3;
	typedef mat<3, 4, int16, defaultp>	i16mat3x4;
	typedef mat<4, 2, int16, defaultp>	i16mat4x2;
	typedef mat<4, 3, int16, defaultp>	i16mat4x3;
	typedef mat<4, 4, int16, defaultp>	i16mat4x4;


	typedef mat<2, 2, int32, lowp>		lowp_i32mat2x2;
	typedef mat<2, 3, int32, lowp>		lowp_i32mat2x3;
	typedef mat<2, 4, int32, lowp>		lowp_i32mat2x4;
	typedef mat<3, 2, int32, lowp>		lowp_i32mat3x2;
	typedef mat<3, 3, int32, lowp>		lowp_i32mat3x3;
	typedef mat<3, 4, int32, lowp>		lowp_i32mat3x4;
	typedef mat<4, 2, int32, lowp>		lowp_i32mat4x2;
	typedef mat<4, 3, int32, lowp>		lowp_i32mat4x3;
	typedef mat<4, 4, int32, lowp>		lowp_i32mat4x4;

	typedef mat<2, 2, int32, mediump>	mediump_i32mat2x2;
	typedef mat<2, 3, int32, mediump>	mediump_i32mat2x3;
	typedef mat<2, 4, int32, mediump>	mediump_i32mat2x4;
	typedef mat<3, 2, int32, mediump>	mediump_i32mat3x2;
	typedef mat<3, 3, int32, mediump>	mediump_i32mat3x3;
	typedef mat<3, 4, int32, mediump>	mediump_i32mat3x4;
	typedef mat<4, 2, int32, mediump>	mediump_i32mat4x2;
	typedef mat<4, 3, int32, mediump>	mediump_i32mat4x3;
	typedef mat<4, 4, int32, mediump>	mediump_i32mat4x4;

	typedef mat<2, 2, int32, highp>		highp_i32mat2x2;
	typedef mat<2, 3, int32, highp>		highp_i32mat2x3;
	typedef mat<2, 4, int32, highp>		highp_i32mat2x4;
	typedef mat<3, 2, int32, highp>		highp_i32mat3x2;
	typedef mat<3, 3, int32, highp>		highp_i32mat3x3;
	typedef mat<3, 4, int32, highp>		highp_i32mat3x4;
	typedef mat<4, 2, int32, highp>		highp_i32mat4x2;
	typedef mat<4, 3, int32, highp>		highp_i32mat4x3;
	typedef mat<4, 4, int32, highp>		highp_i32mat4x4;

	typedef mat<2, 2, int32, defaultp>	i32mat2x2;
	typedef mat<2, 3, int32, defaultp>	i32mat2x3;
	typedef mat<2, 4, int32, defaultp>	i32mat2x4;
	typedef mat<3, 2, int32, defaultp>	i32mat3x2;
	typedef mat<3, 3, int32, defaultp>	i32mat3x3;
	typedef mat<3, 4, int32, defaultp>	i32mat3x4;
	typedef mat<4, 2, int32, defaultp>	i32mat4x2;
	typedef mat<4, 3, int32, defaultp>	i32mat4x3;
	typedef mat<4, 4, int32, defaultp>	i32mat4x4;


	typedef mat<2, 2, int64, lowp>		lowp_i64mat2x2;
	typedef mat<2, 3, int64, lowp>		lowp_i64mat2x3;
	typedef mat<2, 4, int64, lowp>		lowp_i64mat2x4;
	typedef mat<3, 2, int64, lowp>		lowp_i64mat3x2;
	typedef mat<3, 3, int64, lowp>		lowp_i64mat3x3;
	typedef mat<3, 4, int64, lowp>		lowp_i64mat3x4;
	typedef mat<4, 2, int64, lowp>		lowp_i64mat4x2;
	typedef mat<4, 3, int64, lowp>		lowp_i64mat4x3;
	typedef mat<4, 4, int64, lowp>		lowp_i64mat4x4;

	typedef mat<2, 2, int64, mediump>	mediump_i64mat2x2;
	typedef mat<2, 3, int64, mediump>	mediump_i64mat2x3;
	typedef mat<2, 4, int64, mediump>	mediump_i64mat2x4;
	typedef mat<3, 2, int64, mediump>	mediump_i64mat3x2;
	typedef mat<3, 3, int64, mediump>	mediump_i64mat3x3;
	typedef mat<3, 4, int64, mediump>	mediump_i64mat3x4;
	typedef mat<4, 2, int64, mediump>	mediump_i64mat4x2;
	typedef mat<4, 3, int64, mediump>	mediump_i64mat4x3;
	typedef mat<4, 4, int64, mediump>	mediump_i64mat4x4;

	typedef mat<2, 2, int64, highp>		highp_i64mat2x2;
	typedef mat<2, 3, int64, highp>		highp_i64mat2x3;
	typedef mat<2, 4, int64, highp>		highp_i64mat2x4;
	typedef mat<3, 2, int64, highp>		highp_i64mat3x2;
	typedef mat<3, 3, int64, highp>		highp_i64mat3x3;
	typedef mat<3, 4, int64, highp>		highp_i64mat3x4;
	typedef mat<4, 2, int64, highp>		highp_i64mat4x2;
	typedef mat<4, 3, int64, highp>		highp_i64mat4x3;
	typedef mat<4, 4, int64, highp>		highp_i64mat4x4;

	typedef mat<2, 2, int64, defaultp>	i64mat2x2;
	typedef mat<2, 3, int64, defaultp>	i64mat2x3;
	typedef mat<2, 4, int64, defaultp>	i64mat2x4;
	typedef mat<3, 2, int64, defaultp>	i64mat3x2;
	typedef mat<3, 3, int64, defaultp>	i64mat3x3;
	typedef mat<3, 4, int64, defaultp>	i64mat3x4;
	typedef mat<4, 2, int64, defaultp>	i64mat4x2;
	typedef mat<4, 3, int64, defaultp>	i64mat4x3;
	typedef mat<4, 4, int64, defaultp>	i64mat4x4;


	// Unsigned integer matrix MxN

	typedef mat<2, 2, uint, lowp>		lowp_umat2x2;
	typedef mat<2, 3, uint, lowp>		lowp_umat2x3;
	typedef mat<2, 4, uint, lowp>		lowp_umat2x4;
	typedef mat<3, 2, uint, lowp>		lowp_umat3x2;
	typedef mat<3, 3, uint, lowp>		lowp_umat3x3;
	typedef mat<3, 4, uint, lowp>		lowp_umat3x4;
	typedef mat<4, 2, uint, lowp>		lowp_umat4x2;
	typedef mat<4, 3, uint, lowp>		lowp_umat4x3;
	typedef mat<4, 4, uint, lowp>		lowp_umat4x4;

	typedef mat<2, 2, uint, mediump>	mediump_umat2x2;
	typedef mat<2, 3, uint, mediump>	mediump_umat2x3;
	typedef mat<2, 4, uint, mediump>	mediump_umat2x4;
	typedef mat<3, 2, uint, mediump>	mediump_umat3x2;
	typedef mat<3, 3, uint, mediump>	mediump_umat3x3;
	typedef mat<3, 4, uint, mediump>	mediump_umat3x4;
	typedef mat<4, 2, uint, mediump>	mediump_umat4x2;
	typedef mat<4, 3, uint, mediump>	mediump_umat4x3;
	typedef mat<4, 4, uint, mediump>	mediump_umat4x4;

	typedef mat<2, 2, uint, highp>		highp_umat2x2;
	typedef mat<2, 3, uint, highp>		highp_umat2x3;
	typedef mat<2, 4, uint, highp>		highp_umat2x4;
	typedef mat<3, 2, uint, highp>		highp_umat3x2;
	typedef mat<3, 3, uint, highp>		highp_umat3x3;
	typedef mat<3, 4, uint, highp>		highp_umat3x4;
	typedef mat<4, 2, uint, highp>		highp_umat4x2;
	typedef mat<4, 3, uint, highp>		highp_umat4x3;
	typedef mat<4, 4, uint, highp>		highp_umat4x4;

	typedef mat<2, 2, uint, defaultp>	umat2x2;
	typedef mat<2, 3, uint, defaultp>	umat2x3;
	typedef mat<2, 4, uint, defaultp>	umat2x4;
	typedef mat<3, 2, uint, defaultp>	umat3x2;
	typedef mat<3, 3, uint, defaultp>	umat3x3;
	typedef mat<3, 4, uint, defaultp>	umat3x4;
	typedef mat<4, 2, uint, defaultp>	umat4x2;
	typedef mat<4, 3, uint, defaultp>	umat4x3;
	typedef mat<4, 4, uint, defaultp>	umat4x4;


	typedef mat<2, 2, uint8, lowp>		lowp_u8mat2x2;
	typedef mat<2, 3, uint8, lowp>		lowp_u8mat2x3;
	typedef mat<2, 4, uint8, lowp>		lowp_u8mat2x4;
	typedef mat<3, 2, uint8, lowp>		lowp_u8mat3x2;
	typedef mat<3, 3, uint8, lowp>		lowp_u8mat3x3;
	typedef mat<3, 4, uint8, lowp>		lowp_u8mat3x4;
	typedef mat<4, 2, uint8, lowp>		lowp_u8mat4x2;
	typedef mat<4, 3, uint8, lowp>		lowp_u8mat4x3;
	typedef mat<4, 4, uint8, lowp>		lowp_u8mat4x4;

	typedef mat<2, 2, uint8, mediump>	mediump_u8mat2x2;
	typedef mat<2, 3, uint8, mediump>	mediump_u8mat2x3;
	typedef mat<2, 4, uint8, mediump>	mediump_u8mat2x4;
	typedef mat<3, 2, uint8, mediump>	mediump_u8mat3x2;
	typedef mat<3, 3, uint8, mediump>	mediump_u8mat3x3;
	typedef mat<3, 4, uint8, mediump>	mediump_u8mat3x4;
	typedef mat<4, 2, uint8, mediump>	mediump_u8mat4x2;
	typedef mat<4, 3, uint8, mediump>	mediump_u8mat4x3;
	typedef mat<4, 4, uint8, mediump>	mediump_u8mat4x4;

	typedef mat<2, 2, uint8, highp>		highp_u8mat2x2;
	typedef mat<2, 3, uint8, highp>		highp_u8mat2x3;
	typedef mat<2, 4, uint8, highp>		highp_u8mat2x4;
	typedef mat<3, 2, uint8, highp>		highp_u8mat3x2;
	typedef mat<3, 3, uint8, highp>		highp_u8mat3x3;
	typedef mat<3, 4, uint8, highp>		highp_u8mat3x4;
	typedef mat<4, 2, uint8, highp>		highp_u8mat4x2;
	typedef mat<4, 3, uint8, highp>		highp_u8mat4x3;
	typedef mat<4, 4, uint8, highp>		highp_u8mat4x4;

	typedef mat<2, 2, uint8, defaultp>	u8mat2x2;
	typedef mat<2, 3, uint8, defaultp>	u8mat2x3;
	typedef mat<2, 4, uint8, defaultp>	u8mat2x4;
	typedef mat<3, 2, uint8, defaultp>	u8mat3x2;
	typedef mat<3, 3, uint8, defaultp>	u8mat3x3;
	typedef mat<3, 4, uint8, defaultp>	u8mat3x4;
	typedef mat<4, 2, uint8, defaultp>	u8mat4x2;
	typedef mat<4, 3, uint8, defaultp>	u8mat4x3;
	typedef mat<4, 4, uint8, defaultp>	u8mat4x4;


	typedef mat<2, 2, uint16, lowp>		lowp_u16mat2x2;
	typedef mat<2, 3, uint16, lowp>		lowp_u16mat2x3;
	typedef mat<2, 4, uint16, lowp>		lowp_u16mat2x4;
	typedef mat<3, 2, uint16, lowp>		lowp_u16mat3x2;
	typedef mat<3, 3, uint16, lowp>		lowp_u16mat3x3;
	typedef mat<3, 4, uint16, lowp>		lowp_u16mat3x4;
	typedef mat<4, 2, uint16, lowp>		lowp_u16mat4x2;
	typedef mat<4, 3, uint16, lowp>		lowp_u16mat4x3;
	typedef mat<4, 4, uint16, lowp>		lowp_u16mat4x4;

	typedef mat<2, 2, uint16, mediump>	mediump_u16mat2x2;
	typedef mat<2, 3, uint16, mediump>	mediump_u16mat2x3;
	typedef mat<2, 4, uint16, mediump>	mediump_u16mat2x4;
	typedef mat<3, 2, uint16, mediump>	mediump_u16mat3x2;
	typedef mat<3, 3, uint16, mediump>	mediump_u16mat3x3;
	typedef mat<3, 4, uint16, mediump>	mediump_u16mat3x4;
	typedef mat<4, 2, uint16, mediump>	mediump_u16mat4x2;
	typedef mat<4, 3, uint16, mediump>	mediump_u16mat4x3;
	typedef mat<4, 4, uint16, mediump>	mediump_u16mat4x4;

	typedef mat<2, 2, uint16, highp>	highp_u16mat2x2;
	typedef mat<2, 3, uint16, highp>	highp_u16mat2x3;
	typedef mat<2, 4, uint16, highp>	highp_u16mat2x4;
	typedef mat<3, 2, uint16, highp>	highp_u16mat3x2;
	typedef mat<3, 3, uint16, highp>	highp_u16mat3x3;
	typedef mat<3, 4, uint16, highp>	highp_u16mat3x4;
	typedef mat<4, 2, uint16, highp>	highp_u16mat4x2;
	typedef mat<4, 3, uint16, highp>	highp_u16mat4x3;
	typedef mat<4, 4, uint16, highp>	highp_u16mat4x4;

	typedef mat<2, 2, uint16, defaultp>	u16mat2x2;
	typedef mat<2, 3, uint16, defaultp>	u16mat2x3;
	typedef mat<2, 4, uint16, defaultp>	u16mat2x4;
	typedef mat<3, 2, uint16, defaultp>	u16mat3x2;
	typedef mat<3, 3, uint16, defaultp>	u16mat3x3;
	typedef mat<3, 4, uint16, defaultp>	u16mat3x4;
	typedef mat<4, 2, uint16, defaultp>	u16mat4x2;
	typedef mat<4, 3, uint16, defaultp>	u16mat4x3;
	typedef mat<4, 4, uint16, defaultp>	u16mat4x4;


	typedef mat<2, 2, uint32, lowp>		lowp_u32mat2x2;
	typedef mat<2, 3, uint32, lowp>		lowp_u32mat2x3;
	typedef mat<2, 4, uint32, lowp>		lowp_u32mat2x4;
	typedef mat<3, 2, uint32, lowp>		lowp_u32mat3x2;
	typedef mat<3, 3, uint32, lowp>		lowp_u32mat3x3;
	typedef mat<3, 4, uint32, lowp>		lowp_u32mat3x4;
	typedef mat<4, 2, uint32, lowp>		lowp_u32mat4x2;
	typedef mat<4, 3, uint32, lowp>		lowp_u32mat4x3;
	typedef mat<4, 4, uint32, lowp>		lowp_u32mat4x4;

	typedef mat<2, 2, uint32, mediump>	mediump_u32mat2x2;
	typedef mat<2, 3, uint32, mediump>	mediump_u32mat2x3;
	typedef mat<2, 4, uint32, mediump>	mediump_u32mat2x4;
	typedef mat<3, 2, uint32, mediump>	mediump_u32mat3x2;
	typedef mat<3, 3, uint32, mediump>	mediump_u32mat3x3;
	typedef mat<3, 4, uint32, mediump>	mediump_u32mat3x4;
	typedef mat<4, 2, uint32, mediump>	mediump_u32mat4x2;
	typedef mat<4, 3, uint32, mediump>	mediump_u32mat4x3;
	typedef mat<4, 4, uint32, mediump>	mediump_u32mat4x4;

	typedef mat<2, 2, uint32, highp>	highp_u32mat2x2;
	typedef mat<2, 3, uint32, highp>	highp_u32mat2x3;
	typedef mat<2, 4, uint32, highp>	highp_u32mat2x4;
	typedef mat<3, 2, uint32, highp>	highp_u32mat3x2;
	typedef mat<3, 3, uint32, highp>	highp_u32mat3x3;
	typedef mat<3, 4, uint32, highp>	highp_u32mat3x4;
	typedef mat<4, 2, uint32, highp>	highp_u32mat4x2;
	typedef mat<4, 3, uint32, highp>	highp_u32mat4x3;
	typedef mat<4, 4, uint32, highp>	highp_u32mat4x4;

	typedef mat<2, 2, uint32, defaultp>	u32mat2x2;
	typedef mat<2, 3, uint32, defaultp>	u32mat2x3;
	typedef mat<2, 4, uint32, defaultp>	u32mat2x4;
	typedef mat<3, 2, uint32, defaultp>	u32mat3x2;
	typedef mat<3, 3, uint32, defaultp>	u32mat3x3;
	typedef mat<3, 4, uint32, defaultp>	u32mat3x4;
	typedef mat<4, 2, uint32, defaultp>	u32mat4x2;
	typedef mat<4, 3, uint32, defaultp>	u32mat4x3;
	typedef mat<4, 4, uint32, defaultp>	u32mat4x4;


	typedef mat<2, 2, uint64, lowp>		lowp_u64mat2x2;
	typedef mat<2, 3, uint64, lowp>		lowp_u64mat2x3;
	typedef mat<2, 4, uint64, lowp>		lowp_u64mat2x4;
	typedef mat<3, 2, uint64, lowp>		lowp_u64mat3x2;
	typedef mat<3, 3, uint64, lowp>		lowp_u64mat3x3;
	typedef mat<3, 4, uint64, lowp>		lowp_u64mat3x4;
	typedef mat<4, 2, uint64, lowp>		lowp_u64mat4x2;
	typedef mat<4, 3, uint64, lowp>		lowp_u64mat4x3;
	typedef mat<4, 4, uint64, lowp>		lowp_u64mat4x4;

	typedef mat<2, 2, uint64, mediump>	mediump_u64mat2x2;
	typedef mat<2, 3, uint64, mediump>	mediump_u64mat2x3;
	typedef mat<2, 4, uint64, mediump>	mediump_u64mat2x4;
	typedef mat<3, 2, uint64, mediump>	mediump_u64mat3x2;
	typedef mat<3, 3, uint64, mediump>	mediump_u64mat3x3;
	typedef mat<3, 4, uint64, mediump>	mediump_u64mat3x4;
	typedef mat<4, 2, uint64, mediump>	mediump_u64mat4x2;
	typedef mat<4, 3, uint64, mediump>	mediump_u64mat4x3;
	typedef mat<4, 4, uint64, mediump>	mediump_u64mat4x4;

	typedef mat<2, 2, uint64, highp>	highp_u64mat2x2;
	typedef mat<2, 3, uint64, highp>	highp_u64mat2x3;
	typedef mat<2, 4, uint64, highp>	highp_u64mat2x4;
	typedef mat<3, 2, uint64, highp>	highp_u64mat3x2;
	typedef mat<3, 3, uint64, highp>	highp_u64mat3x3;
	typedef mat<3, 4, uint64, highp>	highp_u64mat3x4;
	typedef mat<4, 2, uint64, highp>	highp_u64mat4x2;
	typedef mat<4, 3, uint64, highp>	highp_u64mat4x3;
	typedef mat<4, 4, uint64, highp>	highp_u64mat4x4;

	typedef mat<2, 2, uint64, defaultp>	u64mat2x2;
	typedef mat<2, 3, uint64, defaultp>	u64mat2x3;
	typedef mat<2, 4, uint64, defaultp>	u64mat2x4;
	typedef mat<3, 2, uint64, defaultp>	u64mat3x2;
	typedef mat<3, 3, uint64, defaultp>	u64mat3x3;
	typedef mat<3, 4, uint64, defaultp>	u64mat3x4;
	typedef mat<4, 2, uint64, defaultp>	u64mat4x2;
	typedef mat<4, 3, uint64, defaultp>	u64mat4x3;
	typedef mat<4, 4, uint64, defaultp>	u64mat4x4;

	// Quaternion

	typedef qua<float, lowp>			lowp_quat;
	typedef qua<float, mediump>			mediump_quat;
	typedef qua<float, highp>			highp_quat;
	typedef qua<float, defaultp>		quat;

	typedef qua<float, lowp>			lowp_fquat;
	typedef qua<float, mediump>			mediump_fquat;
	typedef qua<float, highp>			highp_fquat;
	typedef qua<float, defaultp>		fquat;

	typedef qua<f32, lowp>				lowp_f32quat;
	typedef qua<f32, mediump>			mediump_f32quat;
	typedef qua<f32, highp>				highp_f32quat;
	typedef qua<f32, defaultp>			f32quat;

	typedef qua<double, lowp>			lowp_dquat;
	typedef qua<double, mediump>		mediump_dquat;
	typedef qua<double, highp>			highp_dquat;
	typedef qua<double, defaultp>		dquat;

	typedef qua<f64, lowp>				lowp_f64quat;
	typedef qua<f64, mediump>			mediump_f64quat;
	typedef qua<f64, highp>				highp_f64quat;
	typedef qua<f64, defaultp>			f64quat;
}//namespace glm


