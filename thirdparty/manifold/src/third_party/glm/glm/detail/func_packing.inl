/// @ref core
/// @file glm/detail/func_packing.inl

#include "../common.hpp"
#include "type_half.hpp"

namespace glm
{
	GLM_FUNC_QUALIFIER uint packUnorm2x16(vec2 const& v)
	{
		union
		{
			unsigned short in[2];
			uint out;
		} u;

		vec<2, unsigned short, defaultp> result(round(clamp(v, 0.0f, 1.0f) * 65535.0f));

		u.in[0] = result[0];
		u.in[1] = result[1];

		return u.out;
	}

	GLM_FUNC_QUALIFIER vec2 unpackUnorm2x16(uint p)
	{
		union
		{
			uint in;
			unsigned short out[2];
		} u;

		u.in = p;

		return vec2(u.out[0], u.out[1]) * 1.5259021896696421759365224689097e-5f;
	}

	GLM_FUNC_QUALIFIER uint packSnorm2x16(vec2 const& v)
	{
		union
		{
			signed short in[2];
			uint out;
		} u;
 
		vec<2, short, defaultp> result(round(clamp(v, -1.0f, 1.0f) * 32767.0f));

		u.in[0] = result[0];
		u.in[1] = result[1];

		return u.out;
	}

	GLM_FUNC_QUALIFIER vec2 unpackSnorm2x16(uint p)
	{
		union
		{
			uint in;
			signed short out[2];
		} u;

		u.in = p;

		return clamp(vec2(u.out[0], u.out[1]) * 3.0518509475997192297128208258309e-5f, -1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER uint packUnorm4x8(vec4 const& v)
	{
		union
		{
			unsigned char in[4];
			uint out;
		} u;

		vec<4, unsigned char, defaultp> result(round(clamp(v, 0.0f, 1.0f) * 255.0f));

		u.in[0] = result[0];
		u.in[1] = result[1];
		u.in[2] = result[2];
		u.in[3] = result[3];

		return u.out;
	}

	GLM_FUNC_QUALIFIER vec4 unpackUnorm4x8(uint p)
	{
		union
		{
			uint in;
			unsigned char out[4];
		} u;

		u.in = p;

		return vec4(u.out[0], u.out[1], u.out[2], u.out[3]) * 0.0039215686274509803921568627451f;
	}

	GLM_FUNC_QUALIFIER uint packSnorm4x8(vec4 const& v)
	{
		union
		{
			signed char in[4];
			uint out;
		} u;

		vec<4, signed char, defaultp> result(round(clamp(v, -1.0f, 1.0f) * 127.0f));

		u.in[0] = result[0];
		u.in[1] = result[1];
		u.in[2] = result[2];
		u.in[3] = result[3];

		return u.out;
	}

	GLM_FUNC_QUALIFIER glm::vec4 unpackSnorm4x8(uint p)
	{
		union
		{
			uint in;
			signed char out[4];
		} u;

		u.in = p;

		return clamp(vec4(u.out[0], u.out[1], u.out[2], u.out[3]) * 0.0078740157480315f, -1.0f, 1.0f);
	}

	GLM_FUNC_QUALIFIER double packDouble2x32(uvec2 const& v)
	{
		union
		{
			uint   in[2];
			double out;
		} u;

		u.in[0] = v[0];
		u.in[1] = v[1];

		return u.out;
	}

	GLM_FUNC_QUALIFIER uvec2 unpackDouble2x32(double v)
	{
		union
		{
			double in;
			uint   out[2];
		} u;

		u.in = v;

		return uvec2(u.out[0], u.out[1]);
	}

	GLM_FUNC_QUALIFIER uint packHalf2x16(vec2 const& v)
	{
		union
		{
			signed short in[2];
			uint out;
		} u;

		u.in[0] = detail::toFloat16(v.x);
		u.in[1] = detail::toFloat16(v.y);

		return u.out;
	}

	GLM_FUNC_QUALIFIER vec2 unpackHalf2x16(uint v)
	{
		union
		{
			uint in;
			signed short out[2];
		} u;

		u.in = v;

		return vec2(
			detail::toFloat32(u.out[0]),
			detail::toFloat32(u.out[1]));
	}
}//namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "func_packing_simd.inl"
#endif

