#include <glm/gtc/type_precision.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/ext/vector_relational.hpp>
#include <vector>
#if GLM_HAS_OPENMP
#	include <omp.h>
#endif

#if GLM_HAS_STATIC_ASSERT
static_assert(sizeof(glm::lowp_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec4) == 4, "int8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec4) == 4, "int8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec4) == 4, "int8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec4) == 8, "int16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec4) == 8, "int16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec4) == 8, "int16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec4) == 16, "int32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec4) == 16, "int32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec4) == 16, "int32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec4) == 32, "int64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec4) == 32, "int64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec4) == 32, "int64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec1) == 1, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec1) == 2, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec1) == 4, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec1) == 8, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec2) == 2, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec2) == 4, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec2) == 8, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec2) == 16, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec3) == 3, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec3) == 6, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec3) == 12, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec3) == 24, "uint64 size isn't 8 bytes on this platform");


static_assert(sizeof(glm::lowp_u8vec4) == 4, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::mediump_u8vec4) == 4, "uint8 size isn't 1 byte on this platform");
static_assert(sizeof(glm::highp_u8vec4) == 4, "uint8 size isn't 1 byte on this platform");

static_assert(sizeof(glm::lowp_u16vec4) == 8, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::mediump_u16vec4) == 8, "uint16 size isn't 2 bytes on this platform");
static_assert(sizeof(glm::highp_u16vec4) == 8, "uint16 size isn't 2 bytes on this platform");

static_assert(sizeof(glm::lowp_u32vec4) == 16, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::mediump_u32vec4) == 16, "uint32 size isn't 4 bytes on this platform");
static_assert(sizeof(glm::highp_u32vec4) == 16, "uint32 size isn't 4 bytes on this platform");

static_assert(sizeof(glm::lowp_u64vec4) == 32, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::mediump_u64vec4) == 32, "uint64 size isn't 8 bytes on this platform");
static_assert(sizeof(glm::highp_u64vec4) == 32, "uint64 size isn't 8 bytes on this platform");

#endif

static int test_scalar_size()
{
	int Error = 0;

	Error += sizeof(glm::int8) != 1;
	Error += sizeof(glm::int16) != 2;
	Error += sizeof(glm::int32) != 4;
	Error += sizeof(glm::int64) != 8;
	Error += sizeof(glm::uint8) != 1;
	Error += sizeof(glm::uint16) != 2;
	Error += sizeof(glm::uint32) != 4;
	Error += sizeof(glm::uint64) != 8;
	Error += sizeof(glm::float32) != 4;
	Error += sizeof(glm::float64) != 8;

	Error += sizeof(glm::lowp_int8) != 1;
	Error += sizeof(glm::lowp_int16) != 2;
	Error += sizeof(glm::lowp_int32) != 4;
	Error += sizeof(glm::lowp_int64) != 8;
	Error += sizeof(glm::lowp_uint8) != 1;
	Error += sizeof(glm::lowp_uint16) != 2;
	Error += sizeof(glm::lowp_uint32) != 4;
	Error += sizeof(glm::lowp_uint64) != 8;
	Error += sizeof(glm::lowp_float32) != 4;
	Error += sizeof(glm::lowp_float64) != 8;

	Error += sizeof(glm::mediump_int8) != 1;
	Error += sizeof(glm::mediump_int16) != 2;
	Error += sizeof(glm::mediump_int32) != 4;
	Error += sizeof(glm::mediump_int64) != 8;
	Error += sizeof(glm::mediump_uint8) != 1;
	Error += sizeof(glm::mediump_uint16) != 2;
	Error += sizeof(glm::mediump_uint32) != 4;
	Error += sizeof(glm::mediump_uint64) != 8;
	Error += sizeof(glm::mediump_float32) != 4;
	Error += sizeof(glm::mediump_float64) != 8;

	Error += sizeof(glm::highp_int8) != 1;
	Error += sizeof(glm::highp_int16) != 2;
	Error += sizeof(glm::highp_int32) != 4;
	Error += sizeof(glm::highp_int64) != 8;
	Error += sizeof(glm::highp_uint8) != 1;
	Error += sizeof(glm::highp_uint16) != 2;
	Error += sizeof(glm::highp_uint32) != 4;
	Error += sizeof(glm::highp_uint64) != 8;
	Error += sizeof(glm::highp_float32) != 4;
	Error += sizeof(glm::highp_float64) != 8;

	return Error;
}

static int test_fvec_size()
{
	int Error = 0;

	Error += sizeof(glm::f32vec2) != 8;
	Error += sizeof(glm::f32vec3) != 12;
	Error += sizeof(glm::f32vec4) != 16;
	Error += sizeof(glm::f64vec2) != 16;
	Error += sizeof(glm::f64vec3) != 24;
	Error += sizeof(glm::f64vec4) != 32;

	Error += sizeof(glm::lowp_f32vec2) != 8;
	Error += sizeof(glm::lowp_f32vec3) != 12;
	Error += sizeof(glm::lowp_f32vec4) != 16;
	Error += sizeof(glm::lowp_f64vec2) != 16;
	Error += sizeof(glm::lowp_f64vec3) != 24;
	Error += sizeof(glm::lowp_f64vec4) != 32;

	Error += sizeof(glm::mediump_f32vec2) != 8;
	Error += sizeof(glm::mediump_f32vec3) != 12;
	Error += sizeof(glm::mediump_f32vec4) != 16;
	Error += sizeof(glm::mediump_f64vec2) != 16;
	Error += sizeof(glm::mediump_f64vec3) != 24;
	Error += sizeof(glm::mediump_f64vec4) != 32;

	Error += sizeof(glm::highp_f32vec2) != 8;
	Error += sizeof(glm::highp_f32vec3) != 12;
	Error += sizeof(glm::highp_f32vec4) != 16;
	Error += sizeof(glm::highp_f64vec2) != 16;
	Error += sizeof(glm::highp_f64vec3) != 24;
	Error += sizeof(glm::highp_f64vec4) != 32;

	return Error;
}

static int test_fvec_precision()
{
	int Error = 0;

	{
		glm::f32vec2 v1(1.f);
		glm::lowp_f32vec2 v2(v1);
		glm::mediump_f32vec2 v3(v1);
		glm::highp_f32vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f32vec2(v2), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec2(v3), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec2(v4), glm::epsilon<float>())) ? 0 : 1;
	}

	{
		glm::f32vec3 v1(1.f);
		glm::lowp_f32vec3 v2(v1);
		glm::mediump_f32vec3 v3(v1);
		glm::highp_f32vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f32vec3(v2), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec3(v3), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec3(v4), glm::epsilon<float>())) ? 0 : 1;
	}
	
	{
		glm::f32vec4 v1(1.f);
		glm::lowp_f32vec4 v2(v1);
		glm::mediump_f32vec4 v3(v1);
		glm::highp_f32vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f32vec4(v2), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec4(v3), glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f32vec4(v4), glm::epsilon<float>())) ? 0 : 1;
	}
	
	return Error;
}

static int test_dvec_precision()
{
	int Error = 0;

	{
		glm::f64vec2 v1(1.0);
		glm::lowp_f64vec2 v2(v1);
		glm::mediump_f64vec2 v3(v1);
		glm::highp_f64vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f64vec2(v2), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec2(v3), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec2(v4), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::f64vec3 v1(1.0);
		glm::lowp_f64vec3 v2(v1);
		glm::mediump_f64vec3 v3(v1);
		glm::highp_f64vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f64vec3(v2), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec3(v3), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec3(v4), glm::epsilon<double>())) ? 0 : 1;
	}

	{
		glm::f64vec4 v1(1.0);
		glm::lowp_f64vec4 v2(v1);
		glm::mediump_f64vec4 v3(v1);
		glm::highp_f64vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::f64vec4(v2), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec4(v3), glm::epsilon<double>())) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::f64vec4(v4), glm::epsilon<double>())) ? 0 : 1;
	}

	return Error;
}

static int test_ivec_size()
{
	int Error = 0;

	Error += sizeof(glm::i8vec2) != 2;
	Error += sizeof(glm::i8vec3) != 3;
	Error += sizeof(glm::i8vec4) != 4;
	Error += sizeof(glm::i16vec2) != 4;
	Error += sizeof(glm::i16vec3) != 6;
	Error += sizeof(glm::i16vec4) != 8;
	Error += sizeof(glm::i32vec2) != 8;
	Error += sizeof(glm::i32vec3) != 12;
	Error += sizeof(glm::i32vec4) != 16;
	Error += sizeof(glm::i64vec2) != 16;
	Error += sizeof(glm::i64vec3) != 24;
	Error += sizeof(glm::i64vec4) != 32;

	Error += sizeof(glm::lowp_i8vec2) != 2;
	Error += sizeof(glm::lowp_i8vec3) != 3;
	Error += sizeof(glm::lowp_i8vec4) != 4;
	Error += sizeof(glm::lowp_i16vec2) != 4;
	Error += sizeof(glm::lowp_i16vec3) != 6;
	Error += sizeof(glm::lowp_i16vec4) != 8;
	Error += sizeof(glm::lowp_i32vec2) != 8;
	Error += sizeof(glm::lowp_i32vec3) != 12;
	Error += sizeof(glm::lowp_i32vec4) != 16;
	Error += sizeof(glm::lowp_i64vec2) != 16;
	Error += sizeof(glm::lowp_i64vec3) != 24;
	Error += sizeof(glm::lowp_i64vec4) != 32;

	Error += sizeof(glm::mediump_i8vec2) != 2;
	Error += sizeof(glm::mediump_i8vec3) != 3;
	Error += sizeof(glm::mediump_i8vec4) != 4;
	Error += sizeof(glm::mediump_i16vec2) != 4;
	Error += sizeof(glm::mediump_i16vec3) != 6;
	Error += sizeof(glm::mediump_i16vec4) != 8;
	Error += sizeof(glm::mediump_i32vec2) != 8;
	Error += sizeof(glm::mediump_i32vec3) != 12;
	Error += sizeof(glm::mediump_i32vec4) != 16;
	Error += sizeof(glm::mediump_i64vec2) != 16;
	Error += sizeof(glm::mediump_i64vec3) != 24;
	Error += sizeof(glm::mediump_i64vec4) != 32;

	Error += sizeof(glm::highp_i8vec2) != 2;
	Error += sizeof(glm::highp_i8vec3) != 3;
	Error += sizeof(glm::highp_i8vec4) != 4;
	Error += sizeof(glm::highp_i16vec2) != 4;
	Error += sizeof(glm::highp_i16vec3) != 6;
	Error += sizeof(glm::highp_i16vec4) != 8;
	Error += sizeof(glm::highp_i32vec2) != 8;
	Error += sizeof(glm::highp_i32vec3) != 12;
	Error += sizeof(glm::highp_i32vec4) != 16;
	Error += sizeof(glm::highp_i64vec2) != 16;
	Error += sizeof(glm::highp_i64vec3) != 24;
	Error += sizeof(glm::highp_i64vec4) != 32;

	return Error;
}

static int test_ivec_precision()
{
	int Error = 0;

	{
		glm::i8vec2 v1(0);
		glm::lowp_i8vec2 v2(v1);
		glm::mediump_i8vec2 v3(v1);
		glm::highp_i8vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i8vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec2(v4))) ? 0 : 1;
	}

	{
		glm::i8vec3 v1(0);
		glm::lowp_i8vec3 v2(v1);
		glm::mediump_i8vec3 v3(v1);
		glm::highp_i8vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i8vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec3(v4))) ? 0 : 1;
	}

	{
		glm::i8vec4 v1(0);
		glm::lowp_i8vec4 v2(v1);
		glm::mediump_i8vec4 v3(v1);
		glm::highp_i8vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i8vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i8vec4(v4))) ? 0 : 1;
	}

	{
		glm::i16vec2 v1(0);
		glm::lowp_i16vec2 v2(v1);
		glm::mediump_i16vec2 v3(v1);
		glm::highp_i16vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i16vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec2(v4))) ? 0 : 1;
	}

	{
		glm::i16vec3 v1(0);
		glm::lowp_i16vec3 v2(v1);
		glm::mediump_i16vec3 v3(v1);
		glm::highp_i16vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i16vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec3(v4))) ? 0 : 1;
	}

	{
		glm::i16vec4 v1(0);
		glm::lowp_i16vec4 v2(v1);
		glm::mediump_i16vec4 v3(v1);
		glm::highp_i16vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i16vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i16vec4(v4))) ? 0 : 1;
	}

	{
		glm::i32vec2 v1(0);
		glm::lowp_i32vec2 v2(v1);
		glm::mediump_i32vec2 v3(v1);
		glm::highp_i32vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i32vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec2(v4))) ? 0 : 1;
	}

	{
		glm::i32vec3 v1(0);
		glm::lowp_i32vec3 v2(v1);
		glm::mediump_i32vec3 v3(v1);
		glm::highp_i32vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i32vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec3(v4))) ? 0 : 1;
	}

	{
		glm::i32vec4 v1(0);
		glm::lowp_i32vec4 v2(v1);
		glm::mediump_i32vec4 v3(v1);
		glm::highp_i32vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i32vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i32vec4(v4))) ? 0 : 1;
	}

	{
		glm::i64vec2 v1(0);
		glm::lowp_i64vec2 v2(v1);
		glm::mediump_i64vec2 v3(v1);
		glm::highp_i64vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i64vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec2(v4))) ? 0 : 1;
	}

	{
		glm::i64vec3 v1(0);
		glm::lowp_i64vec3 v2(v1);
		glm::mediump_i64vec3 v3(v1);
		glm::highp_i64vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i64vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec3(v4))) ? 0 : 1;
	}

	{
		glm::i64vec4 v1(0);
		glm::lowp_i64vec4 v2(v1);
		glm::mediump_i64vec4 v3(v1);
		glm::highp_i64vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::i64vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::i64vec4(v4))) ? 0 : 1;
	}

	return Error;
}

static int test_uvec_size()
{
	int Error = 0;

	Error += sizeof(glm::u8vec2) != 2;
	Error += sizeof(glm::u8vec3) != 3;
	Error += sizeof(glm::u8vec4) != 4;
	Error += sizeof(glm::u16vec2) != 4;
	Error += sizeof(glm::u16vec3) != 6;
	Error += sizeof(glm::u16vec4) != 8;
	Error += sizeof(glm::u32vec2) != 8;
	Error += sizeof(glm::u32vec3) != 12;
	Error += sizeof(glm::u32vec4) != 16;
	Error += sizeof(glm::u64vec2) != 16;
	Error += sizeof(glm::u64vec3) != 24;
	Error += sizeof(glm::u64vec4) != 32;

	Error += sizeof(glm::lowp_u8vec2) != 2;
	Error += sizeof(glm::lowp_u8vec3) != 3;
	Error += sizeof(glm::lowp_u8vec4) != 4;
	Error += sizeof(glm::lowp_u16vec2) != 4;
	Error += sizeof(glm::lowp_u16vec3) != 6;
	Error += sizeof(glm::lowp_u16vec4) != 8;
	Error += sizeof(glm::lowp_u32vec2) != 8;
	Error += sizeof(glm::lowp_u32vec3) != 12;
	Error += sizeof(glm::lowp_u32vec4) != 16;
	Error += sizeof(glm::lowp_u64vec2) != 16;
	Error += sizeof(glm::lowp_u64vec3) != 24;
	Error += sizeof(glm::lowp_u64vec4) != 32;

	Error += sizeof(glm::mediump_u8vec2) != 2;
	Error += sizeof(glm::mediump_u8vec3) != 3;
	Error += sizeof(glm::mediump_u8vec4) != 4;
	Error += sizeof(glm::mediump_u16vec2) != 4;
	Error += sizeof(glm::mediump_u16vec3) != 6;
	Error += sizeof(glm::mediump_u16vec4) != 8;
	Error += sizeof(glm::mediump_u32vec2) != 8;
	Error += sizeof(glm::mediump_u32vec3) != 12;
	Error += sizeof(glm::mediump_u32vec4) != 16;
	Error += sizeof(glm::mediump_u64vec2) != 16;
	Error += sizeof(glm::mediump_u64vec3) != 24;
	Error += sizeof(glm::mediump_u64vec4) != 32;

	Error += sizeof(glm::highp_u8vec2) != 2;
	Error += sizeof(glm::highp_u8vec3) != 3;
	Error += sizeof(glm::highp_u8vec4) != 4;
	Error += sizeof(glm::highp_u16vec2) != 4;
	Error += sizeof(glm::highp_u16vec3) != 6;
	Error += sizeof(glm::highp_u16vec4) != 8;
	Error += sizeof(glm::highp_u32vec2) != 8;
	Error += sizeof(glm::highp_u32vec3) != 12;
	Error += sizeof(glm::highp_u32vec4) != 16;
	Error += sizeof(glm::highp_u64vec2) != 16;
	Error += sizeof(glm::highp_u64vec3) != 24;
	Error += sizeof(glm::highp_u64vec4) != 32;

	return Error;
}

static int test_uvec_precision()
{
	int Error = 0;

	{
		glm::u8vec2 v1(0);
		glm::lowp_u8vec2 v2(v1);
		glm::mediump_u8vec2 v3(v1);
		glm::highp_u8vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u8vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec2(v4))) ? 0 : 1;
	}

	{
		glm::u8vec3 v1(0);
		glm::lowp_u8vec3 v2(v1);
		glm::mediump_u8vec3 v3(v1);
		glm::highp_u8vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u8vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec3(v4))) ? 0 : 1;
	}
	
	{
		glm::u8vec4 v1(0);
		glm::lowp_u8vec4 v2(v1);
		glm::mediump_u8vec4 v3(v1);
		glm::highp_u8vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u8vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u8vec4(v4))) ? 0 : 1;
	}

	{
		glm::u16vec2 v1(0);
		glm::lowp_u16vec2 v2(v1);
		glm::mediump_u16vec2 v3(v1);
		glm::highp_u16vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u16vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec2(v4))) ? 0 : 1;
	}

	{
		glm::u16vec3 v1(0);
		glm::lowp_u16vec3 v2(v1);
		glm::mediump_u16vec3 v3(v1);
		glm::highp_u16vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u16vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec3(v4))) ? 0 : 1;
	}
	
	{
		glm::u16vec4 v1(0);
		glm::lowp_u16vec4 v2(v1);
		glm::mediump_u16vec4 v3(v1);
		glm::highp_u16vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u16vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u16vec4(v4))) ? 0 : 1;
	}

	{
		glm::u32vec2 v1(0);
		glm::lowp_u32vec2 v2(v1);
		glm::mediump_u32vec2 v3(v1);
		glm::highp_u32vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u32vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec2(v4))) ? 0 : 1;
	}

	{
		glm::u32vec3 v1(0);
		glm::lowp_u32vec3 v2(v1);
		glm::mediump_u32vec3 v3(v1);
		glm::highp_u32vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u32vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec3(v4))) ? 0 : 1;
	}
	
	{
		glm::u32vec4 v1(0);
		glm::lowp_u32vec4 v2(v1);
		glm::mediump_u32vec4 v3(v1);
		glm::highp_u32vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u32vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u32vec4(v4))) ? 0 : 1;
	}
	
	{
		glm::u64vec2 v1(0);
		glm::lowp_u64vec2 v2(v1);
		glm::mediump_u64vec2 v3(v1);
		glm::highp_u64vec2 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u64vec2(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec2(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec2(v4))) ? 0 : 1;
	}

	{
		glm::u64vec3 v1(0);
		glm::lowp_u64vec3 v2(v1);
		glm::mediump_u64vec3 v3(v1);
		glm::highp_u64vec3 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u64vec3(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec3(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec3(v4))) ? 0 : 1;
	}
	
	{
		glm::u64vec4 v1(0);
		glm::lowp_u64vec4 v2(v1);
		glm::mediump_u64vec4 v3(v1);
		glm::highp_u64vec4 v4(v1);

		Error += glm::all(glm::equal(v1, glm::u64vec4(v2))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec4(v3))) ? 0 : 1;
		Error += glm::all(glm::equal(v1, glm::u64vec4(v4))) ? 0 : 1;
	}
	
	return Error;
}

static int test_fmat_size()
{
	int Error = 0;

	Error += sizeof(glm::mat2) != 16;
	Error += sizeof(glm::mat3) != 36;
	Error += sizeof(glm::mat4) != 64;
	Error += sizeof(glm::mat2x2) != 16;
	Error += sizeof(glm::mat2x3) != 24;
	Error += sizeof(glm::mat2x4) != 32;
	Error += sizeof(glm::mat3x2) != 24;
	Error += sizeof(glm::mat3x3) != 36;
	Error += sizeof(glm::mat3x4) != 48;
	Error += sizeof(glm::mat4x2) != 32;
	Error += sizeof(glm::mat4x3) != 48;
	Error += sizeof(glm::mat4x4) != 64;

	Error += sizeof(glm::fmat2) != 16;
	Error += sizeof(glm::fmat3) != 36;
	Error += sizeof(glm::fmat4) != 64;
	Error += sizeof(glm::fmat2x2) != 16;
	Error += sizeof(glm::fmat2x3) != 24;
	Error += sizeof(glm::fmat2x4) != 32;
	Error += sizeof(glm::fmat3x2) != 24;
	Error += sizeof(glm::fmat3x3) != 36;
	Error += sizeof(glm::fmat3x4) != 48;
	Error += sizeof(glm::fmat4x2) != 32;
	Error += sizeof(glm::fmat4x3) != 48;
	Error += sizeof(glm::fmat4x4) != 64;

	Error += sizeof(glm::f32mat2) != 16;
	Error += sizeof(glm::f32mat3) != 36;
	Error += sizeof(glm::f32mat4) != 64;
	Error += sizeof(glm::f32mat2x2) != 16;
	Error += sizeof(glm::f32mat2x3) != 24;
	Error += sizeof(glm::f32mat2x4) != 32;
	Error += sizeof(glm::f32mat3x2) != 24;
	Error += sizeof(glm::f32mat3x3) != 36;
	Error += sizeof(glm::f32mat3x4) != 48;
	Error += sizeof(glm::f32mat4x2) != 32;
	Error += sizeof(glm::f32mat4x3) != 48;
	Error += sizeof(glm::f32mat4x4) != 64;


	Error += sizeof(glm::lowp_mat2) != 16;
	Error += sizeof(glm::lowp_mat3) != 36;
	Error += sizeof(glm::lowp_mat4) != 64;
	Error += sizeof(glm::lowp_mat2x2) != 16;
	Error += sizeof(glm::lowp_mat2x3) != 24;
	Error += sizeof(glm::lowp_mat2x4) != 32;
	Error += sizeof(glm::lowp_mat3x2) != 24;
	Error += sizeof(glm::lowp_mat3x3) != 36;
	Error += sizeof(glm::lowp_mat3x4) != 48;
	Error += sizeof(glm::lowp_mat4x2) != 32;
	Error += sizeof(glm::lowp_mat4x3) != 48;
	Error += sizeof(glm::lowp_mat4x4) != 64;

	Error += sizeof(glm::lowp_fmat2) != 16;
	Error += sizeof(glm::lowp_fmat3) != 36;
	Error += sizeof(glm::lowp_fmat4) != 64;
	Error += sizeof(glm::lowp_fmat2x2) != 16;
	Error += sizeof(glm::lowp_fmat2x3) != 24;
	Error += sizeof(glm::lowp_fmat2x4) != 32;
	Error += sizeof(glm::lowp_fmat3x2) != 24;
	Error += sizeof(glm::lowp_fmat3x3) != 36;
	Error += sizeof(glm::lowp_fmat3x4) != 48;
	Error += sizeof(glm::lowp_fmat4x2) != 32;
	Error += sizeof(glm::lowp_fmat4x3) != 48;
	Error += sizeof(glm::lowp_fmat4x4) != 64;

	Error += sizeof(glm::lowp_f32mat2) != 16;
	Error += sizeof(glm::lowp_f32mat3) != 36;
	Error += sizeof(glm::lowp_f32mat4) != 64;
	Error += sizeof(glm::lowp_f32mat2x2) != 16;
	Error += sizeof(glm::lowp_f32mat2x3) != 24;
	Error += sizeof(glm::lowp_f32mat2x4) != 32;
	Error += sizeof(glm::lowp_f32mat3x2) != 24;
	Error += sizeof(glm::lowp_f32mat3x3) != 36;
	Error += sizeof(glm::lowp_f32mat3x4) != 48;
	Error += sizeof(glm::lowp_f32mat4x2) != 32;
	Error += sizeof(glm::lowp_f32mat4x3) != 48;
	Error += sizeof(glm::lowp_f32mat4x4) != 64;

	Error += sizeof(glm::mediump_mat2) != 16;
	Error += sizeof(glm::mediump_mat3) != 36;
	Error += sizeof(glm::mediump_mat4) != 64;
	Error += sizeof(glm::mediump_mat2x2) != 16;
	Error += sizeof(glm::mediump_mat2x3) != 24;
	Error += sizeof(glm::mediump_mat2x4) != 32;
	Error += sizeof(glm::mediump_mat3x2) != 24;
	Error += sizeof(glm::mediump_mat3x3) != 36;
	Error += sizeof(glm::mediump_mat3x4) != 48;
	Error += sizeof(glm::mediump_mat4x2) != 32;
	Error += sizeof(glm::mediump_mat4x3) != 48;
	Error += sizeof(glm::mediump_mat4x4) != 64;

	Error += sizeof(glm::mediump_fmat2) != 16;
	Error += sizeof(glm::mediump_fmat3) != 36;
	Error += sizeof(glm::mediump_fmat4) != 64;
	Error += sizeof(glm::mediump_fmat2x2) != 16;
	Error += sizeof(glm::mediump_fmat2x3) != 24;
	Error += sizeof(glm::mediump_fmat2x4) != 32;
	Error += sizeof(glm::mediump_fmat3x2) != 24;
	Error += sizeof(glm::mediump_fmat3x3) != 36;
	Error += sizeof(glm::mediump_fmat3x4) != 48;
	Error += sizeof(glm::mediump_fmat4x2) != 32;
	Error += sizeof(glm::mediump_fmat4x3) != 48;
	Error += sizeof(glm::mediump_fmat4x4) != 64;

	Error += sizeof(glm::mediump_f32mat2) != 16;
	Error += sizeof(glm::mediump_f32mat3) != 36;
	Error += sizeof(glm::mediump_f32mat4) != 64;
	Error += sizeof(glm::mediump_f32mat2x2) != 16;
	Error += sizeof(glm::mediump_f32mat2x3) != 24;
	Error += sizeof(glm::mediump_f32mat2x4) != 32;
	Error += sizeof(glm::mediump_f32mat3x2) != 24;
	Error += sizeof(glm::mediump_f32mat3x3) != 36;
	Error += sizeof(glm::mediump_f32mat3x4) != 48;
	Error += sizeof(glm::mediump_f32mat4x2) != 32;
	Error += sizeof(glm::mediump_f32mat4x3) != 48;
	Error += sizeof(glm::mediump_f32mat4x4) != 64;

	Error += sizeof(glm::highp_mat2) != 16;
	Error += sizeof(glm::highp_mat3) != 36;
	Error += sizeof(glm::highp_mat4) != 64;
	Error += sizeof(glm::highp_mat2x2) != 16;
	Error += sizeof(glm::highp_mat2x3) != 24;
	Error += sizeof(glm::highp_mat2x4) != 32;
	Error += sizeof(glm::highp_mat3x2) != 24;
	Error += sizeof(glm::highp_mat3x3) != 36;
	Error += sizeof(glm::highp_mat3x4) != 48;
	Error += sizeof(glm::highp_mat4x2) != 32;
	Error += sizeof(glm::highp_mat4x3) != 48;
	Error += sizeof(glm::highp_mat4x4) != 64;

	Error += sizeof(glm::highp_fmat2) != 16;
	Error += sizeof(glm::highp_fmat3) != 36;
	Error += sizeof(glm::highp_fmat4) != 64;
	Error += sizeof(glm::highp_fmat2x2) != 16;
	Error += sizeof(glm::highp_fmat2x3) != 24;
	Error += sizeof(glm::highp_fmat2x4) != 32;
	Error += sizeof(glm::highp_fmat3x2) != 24;
	Error += sizeof(glm::highp_fmat3x3) != 36;
	Error += sizeof(glm::highp_fmat3x4) != 48;
	Error += sizeof(glm::highp_fmat4x2) != 32;
	Error += sizeof(glm::highp_fmat4x3) != 48;
	Error += sizeof(glm::highp_fmat4x4) != 64;

	Error += sizeof(glm::highp_f32mat2) != 16;
	Error += sizeof(glm::highp_f32mat3) != 36;
	Error += sizeof(glm::highp_f32mat4) != 64;
	Error += sizeof(glm::highp_f32mat2x2) != 16;
	Error += sizeof(glm::highp_f32mat2x3) != 24;
	Error += sizeof(glm::highp_f32mat2x4) != 32;
	Error += sizeof(glm::highp_f32mat3x2) != 24;
	Error += sizeof(glm::highp_f32mat3x3) != 36;
	Error += sizeof(glm::highp_f32mat3x4) != 48;
	Error += sizeof(glm::highp_f32mat4x2) != 32;
	Error += sizeof(glm::highp_f32mat4x3) != 48;
	Error += sizeof(glm::highp_f32mat4x4) != 64;

	return Error;
}

static int test_dmat_size()
{
	int Error = 0;

	Error += sizeof(glm::f64mat2) != 32;
	Error += sizeof(glm::f64mat3) != 72;
	Error += sizeof(glm::f64mat4) != 128;
	Error += sizeof(glm::f64mat2x2) != 32;
	Error += sizeof(glm::f64mat2x3) != 48;
	Error += sizeof(glm::f64mat2x4) != 64;
	Error += sizeof(glm::f64mat3x2) != 48;
	Error += sizeof(glm::f64mat3x3) != 72;
	Error += sizeof(glm::f64mat3x4) != 96;
	Error += sizeof(glm::f64mat4x2) != 64;
	Error += sizeof(glm::f64mat4x3) != 96;
	Error += sizeof(glm::f64mat4x4) != 128;

	Error += sizeof(glm::lowp_f64mat2) != 32;
	Error += sizeof(glm::lowp_f64mat3) != 72;
	Error += sizeof(glm::lowp_f64mat4) != 128;
	Error += sizeof(glm::lowp_f64mat2x2) != 32;
	Error += sizeof(glm::lowp_f64mat2x3) != 48;
	Error += sizeof(glm::lowp_f64mat2x4) != 64;
	Error += sizeof(glm::lowp_f64mat3x2) != 48;
	Error += sizeof(glm::lowp_f64mat3x3) != 72;
	Error += sizeof(glm::lowp_f64mat3x4) != 96;
	Error += sizeof(glm::lowp_f64mat4x2) != 64;
	Error += sizeof(glm::lowp_f64mat4x3) != 96;
	Error += sizeof(glm::lowp_f64mat4x4) != 128;

	Error += sizeof(glm::mediump_f64mat2) != 32;
	Error += sizeof(glm::mediump_f64mat3) != 72;
	Error += sizeof(glm::mediump_f64mat4) != 128;
	Error += sizeof(glm::mediump_f64mat2x2) != 32;
	Error += sizeof(glm::mediump_f64mat2x3) != 48;
	Error += sizeof(glm::mediump_f64mat2x4) != 64;
	Error += sizeof(glm::mediump_f64mat3x2) != 48;
	Error += sizeof(glm::mediump_f64mat3x3) != 72;
	Error += sizeof(glm::mediump_f64mat3x4) != 96;
	Error += sizeof(glm::mediump_f64mat4x2) != 64;
	Error += sizeof(glm::mediump_f64mat4x3) != 96;
	Error += sizeof(glm::mediump_f64mat4x4) != 128;

	Error += sizeof(glm::highp_f64mat2) != 32;
	Error += sizeof(glm::highp_f64mat3) != 72;
	Error += sizeof(glm::highp_f64mat4) != 128;
	Error += sizeof(glm::highp_f64mat2x2) != 32;
	Error += sizeof(glm::highp_f64mat2x3) != 48;
	Error += sizeof(glm::highp_f64mat2x4) != 64;
	Error += sizeof(glm::highp_f64mat3x2) != 48;
	Error += sizeof(glm::highp_f64mat3x3) != 72;
	Error += sizeof(glm::highp_f64mat3x4) != 96;
	Error += sizeof(glm::highp_f64mat4x2) != 64;
	Error += sizeof(glm::highp_f64mat4x3) != 96;
	Error += sizeof(glm::highp_f64mat4x4) != 128;

	return Error;
}

static int test_quat_size()
{
	int Error = 0;

	Error += sizeof(glm::f32quat) != 16;
	Error += sizeof(glm::f64quat) != 32;

	Error += sizeof(glm::lowp_f32quat) != 16;
	Error += sizeof(glm::lowp_f64quat) != 32;

	Error += sizeof(glm::mediump_f32quat) != 16;
	Error += sizeof(glm::mediump_f64quat) != 32;

	Error += sizeof(glm::highp_f32quat) != 16;
	Error += sizeof(glm::highp_f64quat) != 32;

	return Error;
}

static int test_quat_precision()
{
	int Error = 0;
	
	{
		glm::f32quat q1(0.f, glm::vec3(0.f, 0.f, 1.f));
		glm::lowp_f32quat qA(q1);
		glm::mediump_f32quat qB(q1);
		glm::highp_f32quat qC(q1);
		glm::f32quat q2(qA);
		glm::f32quat q3(qB);
		glm::f32quat q4(qC);
		
		Error += glm::all(glm::equal(q1, q2, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(q1, q3, glm::epsilon<float>())) ? 0 : 1;
		Error += glm::all(glm::equal(q1, q4, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

static int test_fvec_conversion()
{
	int Error(0);

	{
		glm::highp_vec4 a = glm::vec4(1, 2, 3, 4);
		glm::mediump_vec4 b = glm::vec4(1, 2, 3, 4);
		glm::lowp_vec4 c = b;
		glm::mediump_vec4 d = c;
		glm::lowp_ivec4 e = glm::ivec4(d);
		glm::lowp_ivec3 f = glm::ivec3(e);

		Error += glm::all(glm::equal(b, d, glm::epsilon<float>())) ? 0 : 1;
	}

	return Error;
}

#if GLM_HAS_OPENMP
static int test_openmp()
{
	std::vector<glm::u8vec3> VectorA(1000);
	std::vector<glm::u8vec3> VectorB(1000);
	std::vector<glm::u8vec3> VectorC(1000);

	for (std::size_t i = 0; i < VectorA.size(); ++i)
	{
		VectorA[i] = glm::u8vec3(1, 1, 1);
		VectorB[i] = glm::u8vec3(1, 1, 1);
	}

	#pragma omp parallel for default(none) shared(VectorA, VectorB, VectorC)
	for (int i = 0; i < static_cast<int>(VectorC.size()); ++i)
	{
		VectorC[i] = VectorA[i] + VectorB[i];
	}

	return 0;
}
#endif//GLM_HAS_OPENMP

int main()
{
	int Error = 0;

	Error += test_scalar_size();
	Error += test_fvec_size();

	Error += test_fvec_precision();
	Error += test_fvec_conversion();

	Error += test_dvec_precision();

	Error += test_uvec_size();
	Error += test_uvec_precision();
	Error += test_ivec_size();
	Error += test_ivec_precision();

	Error += test_fmat_size();
	Error += test_dmat_size();
	Error += test_quat_size();
	Error += test_quat_precision();

#	if GLM_HAS_OPENMP
		Error += test_openmp();
#	endif//

	return Error;
}
