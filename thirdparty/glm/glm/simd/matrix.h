/// @ref simd
/// @file glm/simd/matrix.h

#pragma once

#include "geometric.h"

#if GLM_ARCH & GLM_ARCH_SSE2_BIT

GLM_FUNC_QUALIFIER void glm_mat4_matrixCompMult(glm_vec4 const in1[4], glm_vec4 const in2[4], glm_vec4 out[4])
{
	out[0] = _mm_mul_ps(in1[0], in2[0]);
	out[1] = _mm_mul_ps(in1[1], in2[1]);
	out[2] = _mm_mul_ps(in1[2], in2[2]);
	out[3] = _mm_mul_ps(in1[3], in2[3]);
}

GLM_FUNC_QUALIFIER void glm_mat4_add(glm_vec4 const in1[4], glm_vec4 const in2[4], glm_vec4 out[4])
{
	out[0] = _mm_add_ps(in1[0], in2[0]);
	out[1] = _mm_add_ps(in1[1], in2[1]);
	out[2] = _mm_add_ps(in1[2], in2[2]);
	out[3] = _mm_add_ps(in1[3], in2[3]);
}

GLM_FUNC_QUALIFIER void glm_mat4_sub(glm_vec4 const in1[4], glm_vec4 const in2[4], glm_vec4 out[4])
{
	out[0] = _mm_sub_ps(in1[0], in2[0]);
	out[1] = _mm_sub_ps(in1[1], in2[1]);
	out[2] = _mm_sub_ps(in1[2], in2[2]);
	out[3] = _mm_sub_ps(in1[3], in2[3]);
}

GLM_FUNC_QUALIFIER glm_vec4 glm_mat4_mul_vec4(glm_vec4 const m[4], glm_vec4 v)
{
	__m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 v1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 v2 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 v3 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 m0 = _mm_mul_ps(m[0], v0);
	__m128 m1 = _mm_mul_ps(m[1], v1);
	__m128 m2 = _mm_mul_ps(m[2], v2);
	__m128 m3 = _mm_mul_ps(m[3], v3);

	__m128 a0 = _mm_add_ps(m0, m1);
	__m128 a1 = _mm_add_ps(m2, m3);
	__m128 a2 = _mm_add_ps(a0, a1);

	return a2;
}

GLM_FUNC_QUALIFIER __m128 glm_vec4_mul_mat4(glm_vec4 v, glm_vec4 const m[4])
{
	__m128 i0 = m[0];
	__m128 i1 = m[1];
	__m128 i2 = m[2];
	__m128 i3 = m[3];

	__m128 m0 = _mm_mul_ps(v, i0);
	__m128 m1 = _mm_mul_ps(v, i1);
	__m128 m2 = _mm_mul_ps(v, i2);
	__m128 m3 = _mm_mul_ps(v, i3);

	__m128 u0 = _mm_unpacklo_ps(m0, m1);
	__m128 u1 = _mm_unpackhi_ps(m0, m1);
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(m2, m3);
	__m128 u3 = _mm_unpackhi_ps(m2, m3);
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);
	__m128 f2 = _mm_add_ps(f0, f1);

	return f2;
}

GLM_FUNC_QUALIFIER void glm_mat4_mul(glm_vec4 const in1[4], glm_vec4 const in2[4], glm_vec4 out[4])
{
	{
		__m128 e0 = _mm_shuffle_ps(in2[0], in2[0], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 e1 = _mm_shuffle_ps(in2[0], in2[0], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 e2 = _mm_shuffle_ps(in2[0], in2[0], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 e3 = _mm_shuffle_ps(in2[0], in2[0], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 m0 = _mm_mul_ps(in1[0], e0);
		__m128 m1 = _mm_mul_ps(in1[1], e1);
		__m128 m2 = _mm_mul_ps(in1[2], e2);
		__m128 m3 = _mm_mul_ps(in1[3], e3);

		__m128 a0 = _mm_add_ps(m0, m1);
		__m128 a1 = _mm_add_ps(m2, m3);
		__m128 a2 = _mm_add_ps(a0, a1);

		out[0] = a2;
	}

	{
		__m128 e0 = _mm_shuffle_ps(in2[1], in2[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 e1 = _mm_shuffle_ps(in2[1], in2[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 e2 = _mm_shuffle_ps(in2[1], in2[1], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 e3 = _mm_shuffle_ps(in2[1], in2[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 m0 = _mm_mul_ps(in1[0], e0);
		__m128 m1 = _mm_mul_ps(in1[1], e1);
		__m128 m2 = _mm_mul_ps(in1[2], e2);
		__m128 m3 = _mm_mul_ps(in1[3], e3);

		__m128 a0 = _mm_add_ps(m0, m1);
		__m128 a1 = _mm_add_ps(m2, m3);
		__m128 a2 = _mm_add_ps(a0, a1);

		out[1] = a2;
	}

	{
		__m128 e0 = _mm_shuffle_ps(in2[2], in2[2], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 e1 = _mm_shuffle_ps(in2[2], in2[2], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 e2 = _mm_shuffle_ps(in2[2], in2[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 e3 = _mm_shuffle_ps(in2[2], in2[2], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 m0 = _mm_mul_ps(in1[0], e0);
		__m128 m1 = _mm_mul_ps(in1[1], e1);
		__m128 m2 = _mm_mul_ps(in1[2], e2);
		__m128 m3 = _mm_mul_ps(in1[3], e3);

		__m128 a0 = _mm_add_ps(m0, m1);
		__m128 a1 = _mm_add_ps(m2, m3);
		__m128 a2 = _mm_add_ps(a0, a1);

		out[2] = a2;
	}

	{
		//(__m128&)_mm_shuffle_epi32(__m128i&)in2[0], _MM_SHUFFLE(3, 3, 3, 3))
		__m128 e0 = _mm_shuffle_ps(in2[3], in2[3], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 e1 = _mm_shuffle_ps(in2[3], in2[3], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 e2 = _mm_shuffle_ps(in2[3], in2[3], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 e3 = _mm_shuffle_ps(in2[3], in2[3], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 m0 = _mm_mul_ps(in1[0], e0);
		__m128 m1 = _mm_mul_ps(in1[1], e1);
		__m128 m2 = _mm_mul_ps(in1[2], e2);
		__m128 m3 = _mm_mul_ps(in1[3], e3);

		__m128 a0 = _mm_add_ps(m0, m1);
		__m128 a1 = _mm_add_ps(m2, m3);
		__m128 a2 = _mm_add_ps(a0, a1);

		out[3] = a2;
	}
}

GLM_FUNC_QUALIFIER void glm_mat4_transpose(glm_vec4 const in[4], glm_vec4 out[4])
{
	__m128 tmp0 = _mm_shuffle_ps(in[0], in[1], 0x44);
	__m128 tmp2 = _mm_shuffle_ps(in[0], in[1], 0xEE);
	__m128 tmp1 = _mm_shuffle_ps(in[2], in[3], 0x44);
	__m128 tmp3 = _mm_shuffle_ps(in[2], in[3], 0xEE);

	out[0] = _mm_shuffle_ps(tmp0, tmp1, 0x88);
	out[1] = _mm_shuffle_ps(tmp0, tmp1, 0xDD);
	out[2] = _mm_shuffle_ps(tmp2, tmp3, 0x88);
	out[3] = _mm_shuffle_ps(tmp2, tmp3, 0xDD);
}

GLM_FUNC_QUALIFIER glm_vec4 glm_mat4_determinant_highp(glm_vec4 const in[4])
{
	__m128 Fac0;
	{
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		//	valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac0 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac1;
	{
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		//	valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac1 = _mm_sub_ps(Mul00, Mul01);
	}


	__m128 Fac2;
	{
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		//	valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac2 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac3;
	{
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		//	valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac3 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac4;
	{
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		//	valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac4 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac5;
	{
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		//	valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac5 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 SignA = _mm_set_ps( 1.0f,-1.0f, 1.0f,-1.0f);
	__m128 SignB = _mm_set_ps(-1.0f, 1.0f,-1.0f, 1.0f);

	// m[1][0]
	// m[0][0]
	// m[0][0]
	// m[0][0]
	__m128 Temp0 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][1]
	// m[0][1]
	// m[0][1]
	// m[0][1]
	__m128 Temp1 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(1, 1, 1, 1));
	__m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][2]
	// m[0][2]
	// m[0][2]
	// m[0][2]
	__m128 Temp2 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(2, 2, 2, 2));
	__m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][3]
	// m[0][3]
	// m[0][3]
	// m[0][3]
	__m128 Temp3 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(3, 3, 3, 3));
	__m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

	// col0
	// + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
	// - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
	// + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
	// - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
	__m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
	__m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
	__m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
	__m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
	__m128 Add00 = _mm_add_ps(Sub00, Mul02);
	__m128 Inv0 = _mm_mul_ps(SignB, Add00);

	// col1
	// - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
	// + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
	// - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
	// + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
	__m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
	__m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
	__m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
	__m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
	__m128 Add01 = _mm_add_ps(Sub01, Mul05);
	__m128 Inv1 = _mm_mul_ps(SignA, Add01);

	// col2
	// + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
	// - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
	// + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
	// - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
	__m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
	__m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
	__m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
	__m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
	__m128 Add02 = _mm_add_ps(Sub02, Mul08);
	__m128 Inv2 = _mm_mul_ps(SignB, Add02);

	// col3
	// - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
	// + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
	// - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
	// + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
	__m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
	__m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
	__m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
	__m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
	__m128 Add03 = _mm_add_ps(Sub03, Mul11);
	__m128 Inv3 = _mm_mul_ps(SignA, Add03);

	__m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

	//	valType Determinant = m[0][0] * Inverse[0][0]
	//						+ m[0][1] * Inverse[1][0]
	//						+ m[0][2] * Inverse[2][0]
	//						+ m[0][3] * Inverse[3][0];
	__m128 Det0 = glm_vec4_dot(in[0], Row2);
	return Det0;
}

GLM_FUNC_QUALIFIER glm_vec4 glm_mat4_determinant_lowp(glm_vec4 const m[4])
{
	// _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(

	//T SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
	//T SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
	//T SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
	//T SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
	//T SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
	//T SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

	// First 2 columns
 	__m128 Swp2A = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[2]), _MM_SHUFFLE(0, 1, 1, 2)));
 	__m128 Swp3A = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[3]), _MM_SHUFFLE(3, 2, 3, 3)));
	__m128 MulA = _mm_mul_ps(Swp2A, Swp3A);

	// Second 2 columns
	__m128 Swp2B = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[2]), _MM_SHUFFLE(3, 2, 3, 3)));
	__m128 Swp3B = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[3]), _MM_SHUFFLE(0, 1, 1, 2)));
	__m128 MulB = _mm_mul_ps(Swp2B, Swp3B);

	// Columns subtraction
	__m128 SubE = _mm_sub_ps(MulA, MulB);

	// Last 2 rows
	__m128 Swp2C = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[2]), _MM_SHUFFLE(0, 0, 1, 2)));
	__m128 Swp3C = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[3]), _MM_SHUFFLE(1, 2, 0, 0)));
	__m128 MulC = _mm_mul_ps(Swp2C, Swp3C);
	__m128 SubF = _mm_sub_ps(_mm_movehl_ps(MulC, MulC), MulC);

	//vec<4, T, Q> DetCof(
	//	+ (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02),
	//	- (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04),
	//	+ (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05),
	//	- (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05));

	__m128 SubFacA = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(SubE), _MM_SHUFFLE(2, 1, 0, 0)));
	__m128 SwpFacA = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[1]), _MM_SHUFFLE(0, 0, 0, 1)));
	__m128 MulFacA = _mm_mul_ps(SwpFacA, SubFacA);

	__m128 SubTmpB = _mm_shuffle_ps(SubE, SubF, _MM_SHUFFLE(0, 0, 3, 1));
	__m128 SubFacB = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(SubTmpB), _MM_SHUFFLE(3, 1, 1, 0)));//SubF[0], SubE[3], SubE[3], SubE[1];
	__m128 SwpFacB = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[1]), _MM_SHUFFLE(1, 1, 2, 2)));
	__m128 MulFacB = _mm_mul_ps(SwpFacB, SubFacB);

	__m128 SubRes = _mm_sub_ps(MulFacA, MulFacB);

	__m128 SubTmpC = _mm_shuffle_ps(SubE, SubF, _MM_SHUFFLE(1, 0, 2, 2));
	__m128 SubFacC = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(SubTmpC), _MM_SHUFFLE(3, 3, 2, 0)));
	__m128 SwpFacC = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m[1]), _MM_SHUFFLE(2, 3, 3, 3)));
	__m128 MulFacC = _mm_mul_ps(SwpFacC, SubFacC);

	__m128 AddRes = _mm_add_ps(SubRes, MulFacC);
	__m128 DetCof = _mm_mul_ps(AddRes, _mm_setr_ps( 1.0f,-1.0f, 1.0f,-1.0f));

	//return m[0][0] * DetCof[0]
	//	 + m[0][1] * DetCof[1]
	//	 + m[0][2] * DetCof[2]
	//	 + m[0][3] * DetCof[3];

	return glm_vec4_dot(m[0], DetCof);
}

GLM_FUNC_QUALIFIER glm_vec4 glm_mat4_determinant(glm_vec4 const m[4])
{
	// _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(add)

	//T SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
	//T SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
	//T SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
	//T SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
	//T SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
	//T SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

	// First 2 columns
 	__m128 Swp2A = _mm_shuffle_ps(m[2], m[2], _MM_SHUFFLE(0, 1, 1, 2));
 	__m128 Swp3A = _mm_shuffle_ps(m[3], m[3], _MM_SHUFFLE(3, 2, 3, 3));
	__m128 MulA = _mm_mul_ps(Swp2A, Swp3A);

	// Second 2 columns
	__m128 Swp2B = _mm_shuffle_ps(m[2], m[2], _MM_SHUFFLE(3, 2, 3, 3));
	__m128 Swp3B = _mm_shuffle_ps(m[3], m[3], _MM_SHUFFLE(0, 1, 1, 2));
	__m128 MulB = _mm_mul_ps(Swp2B, Swp3B);

	// Columns subtraction
	__m128 SubE = _mm_sub_ps(MulA, MulB);

	// Last 2 rows
	__m128 Swp2C = _mm_shuffle_ps(m[2], m[2], _MM_SHUFFLE(0, 0, 1, 2));
	__m128 Swp3C = _mm_shuffle_ps(m[3], m[3], _MM_SHUFFLE(1, 2, 0, 0));
	__m128 MulC = _mm_mul_ps(Swp2C, Swp3C);
	__m128 SubF = _mm_sub_ps(_mm_movehl_ps(MulC, MulC), MulC);

	//vec<4, T, Q> DetCof(
	//	+ (m[1][1] * SubFactor00 - m[1][2] * SubFactor01 + m[1][3] * SubFactor02),
	//	- (m[1][0] * SubFactor00 - m[1][2] * SubFactor03 + m[1][3] * SubFactor04),
	//	+ (m[1][0] * SubFactor01 - m[1][1] * SubFactor03 + m[1][3] * SubFactor05),
	//	- (m[1][0] * SubFactor02 - m[1][1] * SubFactor04 + m[1][2] * SubFactor05));

	__m128 SubFacA = _mm_shuffle_ps(SubE, SubE, _MM_SHUFFLE(2, 1, 0, 0));
	__m128 SwpFacA = _mm_shuffle_ps(m[1], m[1], _MM_SHUFFLE(0, 0, 0, 1));
	__m128 MulFacA = _mm_mul_ps(SwpFacA, SubFacA);

	__m128 SubTmpB = _mm_shuffle_ps(SubE, SubF, _MM_SHUFFLE(0, 0, 3, 1));
	__m128 SubFacB = _mm_shuffle_ps(SubTmpB, SubTmpB, _MM_SHUFFLE(3, 1, 1, 0));//SubF[0], SubE[3], SubE[3], SubE[1];
	__m128 SwpFacB = _mm_shuffle_ps(m[1], m[1], _MM_SHUFFLE(1, 1, 2, 2));
	__m128 MulFacB = _mm_mul_ps(SwpFacB, SubFacB);

	__m128 SubRes = _mm_sub_ps(MulFacA, MulFacB);

	__m128 SubTmpC = _mm_shuffle_ps(SubE, SubF, _MM_SHUFFLE(1, 0, 2, 2));
	__m128 SubFacC = _mm_shuffle_ps(SubTmpC, SubTmpC, _MM_SHUFFLE(3, 3, 2, 0));
	__m128 SwpFacC = _mm_shuffle_ps(m[1], m[1], _MM_SHUFFLE(2, 3, 3, 3));
	__m128 MulFacC = _mm_mul_ps(SwpFacC, SubFacC);

	__m128 AddRes = _mm_add_ps(SubRes, MulFacC);
	__m128 DetCof = _mm_mul_ps(AddRes, _mm_setr_ps( 1.0f,-1.0f, 1.0f,-1.0f));

	//return m[0][0] * DetCof[0]
	//	 + m[0][1] * DetCof[1]
	//	 + m[0][2] * DetCof[2]
	//	 + m[0][3] * DetCof[3];

	return glm_vec4_dot(m[0], DetCof);
}

GLM_FUNC_QUALIFIER void glm_mat4_inverse(glm_vec4 const in[4], glm_vec4 out[4])
{
	__m128 Fac0;
	{
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		//	valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac0 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac1;
	{
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		//	valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac1 = _mm_sub_ps(Mul00, Mul01);
	}


	__m128 Fac2;
	{
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		//	valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac2 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac3;
	{
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		//	valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac3 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac4;
	{
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		//	valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac4 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac5;
	{
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		//	valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac5 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 SignA = _mm_set_ps( 1.0f,-1.0f, 1.0f,-1.0f);
	__m128 SignB = _mm_set_ps(-1.0f, 1.0f,-1.0f, 1.0f);

	// m[1][0]
	// m[0][0]
	// m[0][0]
	// m[0][0]
	__m128 Temp0 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][1]
	// m[0][1]
	// m[0][1]
	// m[0][1]
	__m128 Temp1 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(1, 1, 1, 1));
	__m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][2]
	// m[0][2]
	// m[0][2]
	// m[0][2]
	__m128 Temp2 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(2, 2, 2, 2));
	__m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][3]
	// m[0][3]
	// m[0][3]
	// m[0][3]
	__m128 Temp3 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(3, 3, 3, 3));
	__m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

	// col0
	// + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
	// - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
	// + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
	// - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
	__m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
	__m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
	__m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
	__m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
	__m128 Add00 = _mm_add_ps(Sub00, Mul02);
	__m128 Inv0 = _mm_mul_ps(SignB, Add00);

	// col1
	// - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
	// + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
	// - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
	// + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
	__m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
	__m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
	__m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
	__m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
	__m128 Add01 = _mm_add_ps(Sub01, Mul05);
	__m128 Inv1 = _mm_mul_ps(SignA, Add01);

	// col2
	// + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
	// - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
	// + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
	// - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
	__m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
	__m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
	__m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
	__m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
	__m128 Add02 = _mm_add_ps(Sub02, Mul08);
	__m128 Inv2 = _mm_mul_ps(SignB, Add02);

	// col3
	// - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
	// + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
	// - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
	// + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
	__m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
	__m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
	__m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
	__m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
	__m128 Add03 = _mm_add_ps(Sub03, Mul11);
	__m128 Inv3 = _mm_mul_ps(SignA, Add03);

	__m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

	//	valType Determinant = m[0][0] * Inverse[0][0]
	//						+ m[0][1] * Inverse[1][0]
	//						+ m[0][2] * Inverse[2][0]
	//						+ m[0][3] * Inverse[3][0];
	__m128 Det0 = glm_vec4_dot(in[0], Row2);
	__m128 Rcp0 = _mm_div_ps(_mm_set1_ps(1.0f), Det0);
	//__m128 Rcp0 = _mm_rcp_ps(Det0);

	//	Inverse /= Determinant;
	out[0] = _mm_mul_ps(Inv0, Rcp0);
	out[1] = _mm_mul_ps(Inv1, Rcp0);
	out[2] = _mm_mul_ps(Inv2, Rcp0);
	out[3] = _mm_mul_ps(Inv3, Rcp0);
}

GLM_FUNC_QUALIFIER void glm_mat4_inverse_lowp(glm_vec4 const in[4], glm_vec4 out[4])
{
	__m128 Fac0;
	{
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		//	valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
		//	valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac0 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac1;
	{
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		//	valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
		//	valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac1 = _mm_sub_ps(Mul00, Mul01);
	}


	__m128 Fac2;
	{
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		//	valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
		//	valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac2 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac3;
	{
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		//	valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
		//	valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac3 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac4;
	{
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		//	valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
		//	valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac4 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 Fac5;
	{
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
		//	valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
		//	valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

		__m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

		__m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
		__m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));

		__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
		__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
		Fac5 = _mm_sub_ps(Mul00, Mul01);
	}

	__m128 SignA = _mm_set_ps( 1.0f,-1.0f, 1.0f,-1.0f);
	__m128 SignB = _mm_set_ps(-1.0f, 1.0f,-1.0f, 1.0f);

	// m[1][0]
	// m[0][0]
	// m[0][0]
	// m[0][0]
	__m128 Temp0 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][1]
	// m[0][1]
	// m[0][1]
	// m[0][1]
	__m128 Temp1 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(1, 1, 1, 1));
	__m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][2]
	// m[0][2]
	// m[0][2]
	// m[0][2]
	__m128 Temp2 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(2, 2, 2, 2));
	__m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

	// m[1][3]
	// m[0][3]
	// m[0][3]
	// m[0][3]
	__m128 Temp3 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(3, 3, 3, 3));
	__m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

	// col0
	// + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
	// - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
	// + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
	// - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
	__m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
	__m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
	__m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
	__m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
	__m128 Add00 = _mm_add_ps(Sub00, Mul02);
	__m128 Inv0 = _mm_mul_ps(SignB, Add00);

	// col1
	// - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
	// + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
	// - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
	// + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
	__m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
	__m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
	__m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
	__m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
	__m128 Add01 = _mm_add_ps(Sub01, Mul05);
	__m128 Inv1 = _mm_mul_ps(SignA, Add01);

	// col2
	// + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
	// - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
	// + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
	// - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
	__m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
	__m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
	__m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
	__m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
	__m128 Add02 = _mm_add_ps(Sub02, Mul08);
	__m128 Inv2 = _mm_mul_ps(SignB, Add02);

	// col3
	// - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
	// + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
	// - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
	// + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
	__m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
	__m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
	__m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
	__m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
	__m128 Add03 = _mm_add_ps(Sub03, Mul11);
	__m128 Inv3 = _mm_mul_ps(SignA, Add03);

	__m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

	//	valType Determinant = m[0][0] * Inverse[0][0]
	//						+ m[0][1] * Inverse[1][0]
	//						+ m[0][2] * Inverse[2][0]
	//						+ m[0][3] * Inverse[3][0];
	__m128 Det0 = glm_vec4_dot(in[0], Row2);
	__m128 Rcp0 = _mm_rcp_ps(Det0);
	//__m128 Rcp0 = _mm_div_ps(one, Det0);
	//	Inverse /= Determinant;
	out[0] = _mm_mul_ps(Inv0, Rcp0);
	out[1] = _mm_mul_ps(Inv1, Rcp0);
	out[2] = _mm_mul_ps(Inv2, Rcp0);
	out[3] = _mm_mul_ps(Inv3, Rcp0);
}
/*
GLM_FUNC_QUALIFIER void glm_mat4_rotate(__m128 const in[4], float Angle, float const v[3], __m128 out[4])
{
	float a = glm::radians(Angle);
	float c = cos(a);
	float s = sin(a);

	glm::vec4 AxisA(v[0], v[1], v[2], float(0));
	__m128 AxisB = _mm_set_ps(AxisA.w, AxisA.z, AxisA.y, AxisA.x);
	__m128 AxisC = detail::sse_nrm_ps(AxisB);

	__m128 Cos0 = _mm_set_ss(c);
	__m128 CosA = _mm_shuffle_ps(Cos0, Cos0, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 Sin0 = _mm_set_ss(s);
	__m128 SinA = _mm_shuffle_ps(Sin0, Sin0, _MM_SHUFFLE(0, 0, 0, 0));

	// vec<3, T, Q> temp = (valType(1) - c) * axis;
	__m128 Temp0 = _mm_sub_ps(one, CosA);
	__m128 Temp1 = _mm_mul_ps(Temp0, AxisC);

	//Rotate[0][0] = c + temp[0] * axis[0];
	//Rotate[0][1] = 0 + temp[0] * axis[1] + s * axis[2];
	//Rotate[0][2] = 0 + temp[0] * axis[2] - s * axis[1];
	__m128 Axis0 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 TmpA0 = _mm_mul_ps(Axis0, AxisC);
	__m128 CosA0 = _mm_shuffle_ps(Cos0, Cos0, _MM_SHUFFLE(1, 1, 1, 0));
	__m128 TmpA1 = _mm_add_ps(CosA0, TmpA0);
	__m128 SinA0 = SinA;//_mm_set_ps(0.0f, s, -s, 0.0f);
	__m128 TmpA2 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(3, 1, 2, 3));
	__m128 TmpA3 = _mm_mul_ps(SinA0, TmpA2);
	__m128 TmpA4 = _mm_add_ps(TmpA1, TmpA3);

	//Rotate[1][0] = 0 + temp[1] * axis[0] - s * axis[2];
	//Rotate[1][1] = c + temp[1] * axis[1];
	//Rotate[1][2] = 0 + temp[1] * axis[2] + s * axis[0];
	__m128 Axis1 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 TmpB0 = _mm_mul_ps(Axis1, AxisC);
	__m128 CosA1 = _mm_shuffle_ps(Cos0, Cos0, _MM_SHUFFLE(1, 1, 0, 1));
	__m128 TmpB1 = _mm_add_ps(CosA1, TmpB0);
	__m128 SinB0 = SinA;//_mm_set_ps(-s, 0.0f, s, 0.0f);
	__m128 TmpB2 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(3, 0, 3, 2));
	__m128 TmpB3 = _mm_mul_ps(SinA0, TmpB2);
	__m128 TmpB4 = _mm_add_ps(TmpB1, TmpB3);

	//Rotate[2][0] = 0 + temp[2] * axis[0] + s * axis[1];
	//Rotate[2][1] = 0 + temp[2] * axis[1] - s * axis[0];
	//Rotate[2][2] = c + temp[2] * axis[2];
	__m128 Axis2 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 TmpC0 = _mm_mul_ps(Axis2, AxisC);
	__m128 CosA2 = _mm_shuffle_ps(Cos0, Cos0, _MM_SHUFFLE(1, 0, 1, 1));
	__m128 TmpC1 = _mm_add_ps(CosA2, TmpC0);
	__m128 SinC0 = SinA;//_mm_set_ps(s, -s, 0.0f, 0.0f);
	__m128 TmpC2 = _mm_shuffle_ps(AxisC, AxisC, _MM_SHUFFLE(3, 3, 0, 1));
	__m128 TmpC3 = _mm_mul_ps(SinA0, TmpC2);
	__m128 TmpC4 = _mm_add_ps(TmpC1, TmpC3);

	__m128 Result[4];
	Result[0] = TmpA4;
	Result[1] = TmpB4;
	Result[2] = TmpC4;
	Result[3] = _mm_set_ps(1, 0, 0, 0);

	//mat<4, 4, valType> Result;
	//Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
	//Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
	//Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
	//Result[3] = m[3];
	//return Result;
	sse_mul_ps(in, Result, out);
}
*/
GLM_FUNC_QUALIFIER void glm_mat4_outerProduct(__m128 const& c, __m128 const& r, __m128 out[4])
{
	out[0] = _mm_mul_ps(c, _mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 0)));
	out[1] = _mm_mul_ps(c, _mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1)));
	out[2] = _mm_mul_ps(c, _mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 2, 2)));
	out[3] = _mm_mul_ps(c, _mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 3, 3)));
}

#endif//GLM_ARCH & GLM_ARCH_SSE2_BIT
