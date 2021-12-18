
vec2 select2(vec2 a, vec2 b, bvec2 c) {
	vec2 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;

	return ret;
}

vec3 select3(vec3 a, vec3 b, bvec3 c) {
	vec3 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;

	return ret;
}

vec4 select4(vec4 a, vec4 b, bvec4 c) {
	vec4 ret;

	ret.x = c.x ? b.x : a.x;
	ret.y = c.y ? b.y : a.y;
	ret.z = c.z ? b.z : a.z;
	ret.w = c.w ? b.w : a.w;

	return ret;
}

highp vec4 texel2DFetch(highp sampler2D tex, ivec2 size, ivec2 coord) {
	float x_coord = float(2 * coord.x + 1) / float(size.x * 2);
	float y_coord = float(2 * coord.y + 1) / float(size.y * 2);

	return texture2DLod(tex, vec2(x_coord, y_coord), 0.0);
}

#if defined(SINH_USED)

highp float sinh(highp float x) {
	return 0.5 * (exp(x) - exp(-x));
}

highp vec2 sinh(highp vec2 x) {
	return 0.5 * vec2(exp(x.x) - exp(-x.x), exp(x.y) - exp(-x.y));
}

highp vec3 sinh(highp vec3 x) {
	return 0.5 * vec3(exp(x.x) - exp(-x.x), exp(x.y) - exp(-x.y), exp(x.z) - exp(-x.z));
}

highp vec4 sinh(highp vec4 x) {
	return 0.5 * vec4(exp(x.x) - exp(-x.x), exp(x.y) - exp(-x.y), exp(x.z) - exp(-x.z), exp(x.w) - exp(-x.w));
}

#endif

#if defined(COSH_USED)

highp float cosh(highp float x) {
	return 0.5 * (exp(x) + exp(-x));
}

highp vec2 cosh(highp vec2 x) {
	return 0.5 * vec2(exp(x.x) + exp(-x.x), exp(x.y) + exp(-x.y));
}

highp vec3 cosh(highp vec3 x) {
	return 0.5 * vec3(exp(x.x) + exp(-x.x), exp(x.y) + exp(-x.y), exp(x.z) + exp(-x.z));
}

highp vec4 cosh(highp vec4 x) {
	return 0.5 * vec4(exp(x.x) + exp(-x.x), exp(x.y) + exp(-x.y), exp(x.z) + exp(-x.z), exp(x.w) + exp(-x.w));
}

#endif

#if defined(TANH_USED)

highp float tanh(highp float x) {
	highp float exp2x = exp(2.0 * x);
	return (exp2x - 1.0) / (exp2x + 1.0);
}

highp vec2 tanh(highp vec2 x) {
	highp float exp2x = exp(2.0 * x.x);
	highp float exp2y = exp(2.0 * x.y);
	return vec2((exp2x - 1.0) / (exp2x + 1.0), (exp2y - 1.0) / (exp2y + 1.0));
}

highp vec3 tanh(highp vec3 x) {
	highp float exp2x = exp(2.0 * x.x);
	highp float exp2y = exp(2.0 * x.y);
	highp float exp2z = exp(2.0 * x.z);
	return vec3((exp2x - 1.0) / (exp2x + 1.0), (exp2y - 1.0) / (exp2y + 1.0), (exp2z - 1.0) / (exp2z + 1.0));
}

highp vec4 tanh(highp vec4 x) {
	highp float exp2x = exp(2.0 * x.x);
	highp float exp2y = exp(2.0 * x.y);
	highp float exp2z = exp(2.0 * x.z);
	highp float exp2w = exp(2.0 * x.w);
	return vec4((exp2x - 1.0) / (exp2x + 1.0), (exp2y - 1.0) / (exp2y + 1.0), (exp2z - 1.0) / (exp2z + 1.0), (exp2w - 1.0) / (exp2w + 1.0));
}

#endif

#if defined(ASINH_USED)

highp float asinh(highp float x) {
	return sign(x) * log(abs(x) + sqrt(1.0 + x * x));
}

highp vec2 asinh(highp vec2 x) {
	return vec2(sign(x.x) * log(abs(x.x) + sqrt(1.0 + x.x * x.x)), sign(x.y) * log(abs(x.y) + sqrt(1.0 + x.y * x.y)));
}

highp vec3 asinh(highp vec3 x) {
	return vec3(sign(x.x) * log(abs(x.x) + sqrt(1.0 + x.x * x.x)), sign(x.y) * log(abs(x.y) + sqrt(1.0 + x.y * x.y)), sign(x.z) * log(abs(x.z) + sqrt(1.0 + x.z * x.z)));
}

highp vec4 asinh(highp vec4 x) {
	return vec4(sign(x.x) * log(abs(x.x) + sqrt(1.0 + x.x * x.x)), sign(x.y) * log(abs(x.y) + sqrt(1.0 + x.y * x.y)), sign(x.z) * log(abs(x.z) + sqrt(1.0 + x.z * x.z)), sign(x.w) * log(abs(x.w) + sqrt(1.0 + x.w * x.w)));
}

#endif

#if defined(ACOSH_USED)

highp float acosh(highp float x) {
	return log(x + sqrt(x * x - 1.0));
}

highp vec2 acosh(highp vec2 x) {
	return vec2(log(x.x + sqrt(x.x * x.x - 1.0)), log(x.y + sqrt(x.y * x.y - 1.0)));
}

highp vec3 acosh(highp vec3 x) {
	return vec3(log(x.x + sqrt(x.x * x.x - 1.0)), log(x.y + sqrt(x.y * x.y - 1.0)), log(x.z + sqrt(x.z * x.z - 1.0)));
}

highp vec4 acosh(highp vec4 x) {
	return vec4(log(x.x + sqrt(x.x * x.x - 1.0)), log(x.y + sqrt(x.y * x.y - 1.0)), log(x.z + sqrt(x.z * x.z - 1.0)), log(x.w + sqrt(x.w * x.w - 1.0)));
}

#endif

#if defined(ATANH_USED)

highp float atanh(highp float x) {
	return 0.5 * log((1.0 + x) / (1.0 - x));
}

highp vec2 atanh(highp vec2 x) {
	return 0.5 * vec2(log((1.0 + x.x) / (1.0 - x.x)), log((1.0 + x.y) / (1.0 - x.y)));
}

highp vec3 atanh(highp vec3 x) {
	return 0.5 * vec3(log((1.0 + x.x) / (1.0 - x.x)), log((1.0 + x.y) / (1.0 - x.y)), log((1.0 + x.z) / (1.0 - x.z)));
}

highp vec4 atanh(highp vec4 x) {
	return 0.5 * vec4(log((1.0 + x.x) / (1.0 - x.x)), log((1.0 + x.y) / (1.0 - x.y)), log((1.0 + x.z) / (1.0 - x.z)), log((1.0 + x.w) / (1.0 - x.w)));
}

#endif

#if defined(ROUND_USED)

highp float round(highp float x) {
	return floor(x + 0.5);
}

highp vec2 round(highp vec2 x) {
	return floor(x + vec2(0.5));
}

highp vec3 round(highp vec3 x) {
	return floor(x + vec3(0.5));
}

highp vec4 round(highp vec4 x) {
	return floor(x + vec4(0.5));
}

#endif

#if defined(ROUND_EVEN_USED)

highp float roundEven(highp float x) {
	highp float t = x + 0.5;
	highp float f = floor(t);
	highp float r;
	if (t == f) {
		if (x > 0)
			r = f - mod(f, 2);
		else
			r = f + mod(f, 2);
	} else
		r = f;
	return r;
}

highp vec2 roundEven(highp vec2 x) {
	return vec2(roundEven(x.x), roundEven(x.y));
}

highp vec3 roundEven(highp vec3 x) {
	return vec3(roundEven(x.x), roundEven(x.y), roundEven(x.z));
}

highp vec4 roundEven(highp vec4 x) {
	return vec4(roundEven(x.x), roundEven(x.y), roundEven(x.z), roundEven(x.w));
}

#endif

#if defined(IS_INF_USED)

bool isinf(highp float x) {
	return (2 * x == x) && (x != 0);
}

bvec2 isinf(highp vec2 x) {
	return bvec2((2 * x.x == x.x) && (x.x != 0), (2 * x.y == x.y) && (x.y != 0));
}

bvec3 isinf(highp vec3 x) {
	return bvec3((2 * x.x == x.x) && (x.x != 0), (2 * x.y == x.y) && (x.y != 0), (2 * x.z == x.z) && (x.z != 0));
}

bvec4 isinf(highp vec4 x) {
	return bvec4((2 * x.x == x.x) && (x.x != 0), (2 * x.y == x.y) && (x.y != 0), (2 * x.z == x.z) && (x.z != 0), (2 * x.w == x.w) && (x.w != 0));
}

#endif

#if defined(IS_NAN_USED)

bool isnan(highp float x) {
	return x != x;
}

bvec2 isnan(highp vec2 x) {
	return bvec2(x.x != x.x, x.y != x.y);
}

bvec3 isnan(highp vec3 x) {
	return bvec3(x.x != x.x, x.y != x.y, x.z != x.z);
}

bvec4 isnan(highp vec4 x) {
	return bvec4(x.x != x.x, x.y != x.y, x.z != x.z, x.w != x.w);
}

#endif

#if defined(TRUNC_USED)

highp float trunc(highp float x) {
	return x < 0.0 ? -floor(-x) : floor(x);
}

highp vec2 trunc(highp vec2 x) {
	return vec2(x.x < 0.0 ? -floor(-x.x) : floor(x.x), x.y < 0.0 ? -floor(-x.y) : floor(x.y));
}

highp vec3 trunc(highp vec3 x) {
	return vec3(x.x < 0.0 ? -floor(-x.x) : floor(x.x), x.y < 0.0 ? -floor(-x.y) : floor(x.y), x.z < 0.0 ? -floor(-x.z) : floor(x.z));
}

highp vec4 trunc(highp vec4 x) {
	return vec4(x.x < 0.0 ? -floor(-x.x) : floor(x.x), x.y < 0.0 ? -floor(-x.y) : floor(x.y), x.z < 0.0 ? -floor(-x.z) : floor(x.z), x.w < 0.0 ? -floor(-x.w) : floor(x.w));
}

#endif

#if defined(DETERMINANT_USED)

highp float determinant(highp mat2 m) {
	return m[0].x * m[1].y - m[1].x * m[0].y;
}

highp float determinant(highp mat3 m) {
	return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}

highp float determinant(highp mat4 m) {
	highp float s00 = m[2].z * m[3].w - m[3].z * m[2].w;
	highp float s01 = m[2].y * m[3].w - m[3].y * m[2].w;
	highp float s02 = m[2].y * m[3].z - m[3].y * m[2].z;
	highp float s03 = m[2].x * m[3].w - m[3].x * m[2].w;
	highp float s04 = m[2].x * m[3].z - m[3].x * m[2].z;
	highp float s05 = m[2].x * m[3].y - m[3].x * m[2].y;
	highp vec4 c = vec4((m[1].y * s00 - m[1].z * s01 + m[1].w * s02), -(m[1].x * s00 - m[1].z * s03 + m[1].w * s04), (m[1].x * s01 - m[1].y * s03 + m[1].w * s05), -(m[1].x * s02 - m[1].y * s04 + m[1].z * s05));
	return m[0].x * c.x + m[0].y * c.y + m[0].z * c.z + m[0].w * c.w;
}

#endif

#if defined(INVERSE_USED)

highp mat2 inverse(highp mat2 m) {
	highp float d = 1.0 / (m[0].x * m[1].y - m[1].x * m[0].y);
	return mat2(
			vec2(m[1].y * d, -m[0].y * d),
			vec2(-m[1].x * d, m[0].x * d));
}

highp mat3 inverse(highp mat3 m) {
	highp float c01 = m[2].z * m[1].y - m[1].z * m[2].y;
	highp float c11 = -m[2].z * m[1].x + m[1].z * m[2].x;
	highp float c21 = m[2].y * m[1].x - m[1].y * m[2].x;
	highp float d = 1.0 / (m[0].x * c01 + m[0].y * c11 + m[0].z * c21);

	return mat3(c01, (-m[2].z * m[0].y + m[0].z * m[2].y), (m[1].z * m[0].y - m[0].z * m[1].y),
				   c11, (m[2].z * m[0].x - m[0].z * m[2].x), (-m[1].z * m[0].x + m[0].z * m[1].x),
				   c21, (-m[2].y * m[0].x + m[0].y * m[2].x), (m[1].y * m[0].x - m[0].y * m[1].x)) *
			d;
}

highp mat4 inverse(highp mat4 m) {
	highp float c00 = m[2].z * m[3].w - m[3].z * m[2].w;
	highp float c02 = m[1].z * m[3].w - m[3].z * m[1].w;
	highp float c03 = m[1].z * m[2].w - m[2].z * m[1].w;

	highp float c04 = m[2].y * m[3].w - m[3].y * m[2].w;
	highp float c06 = m[1].y * m[3].w - m[3].y * m[1].w;
	highp float c07 = m[1].y * m[2].w - m[2].y * m[1].w;

	highp float c08 = m[2].y * m[3].z - m[3].y * m[2].z;
	highp float c10 = m[1].y * m[3].z - m[3].y * m[1].z;
	highp float c11 = m[1].y * m[2].z - m[2].y * m[1].z;

	highp float c12 = m[2].x * m[3].w - m[3].x * m[2].w;
	highp float c14 = m[1].x * m[3].w - m[3].x * m[1].w;
	highp float c15 = m[1].x * m[2].w - m[2].x * m[1].w;

	highp float c16 = m[2].x * m[3].z - m[3].x * m[2].z;
	highp float c18 = m[1].x * m[3].z - m[3].x * m[1].z;
	highp float c19 = m[1].x * m[2].z - m[2].x * m[1].z;

	highp float c20 = m[2].x * m[3].y - m[3].x * m[2].y;
	highp float c22 = m[1].x * m[3].y - m[3].x * m[1].y;
	highp float c23 = m[1].x * m[2].y - m[2].x * m[1].y;

	vec4 f0 = vec4(c00, c00, c02, c03);
	vec4 f1 = vec4(c04, c04, c06, c07);
	vec4 f2 = vec4(c08, c08, c10, c11);
	vec4 f3 = vec4(c12, c12, c14, c15);
	vec4 f4 = vec4(c16, c16, c18, c19);
	vec4 f5 = vec4(c20, c20, c22, c23);

	vec4 v0 = vec4(m[1].x, m[0].x, m[0].x, m[0].x);
	vec4 v1 = vec4(m[1].y, m[0].y, m[0].y, m[0].y);
	vec4 v2 = vec4(m[1].z, m[0].z, m[0].z, m[0].z);
	vec4 v3 = vec4(m[1].w, m[0].w, m[0].w, m[0].w);

	vec4 inv0 = vec4(v1 * f0 - v2 * f1 + v3 * f2);
	vec4 inv1 = vec4(v0 * f0 - v2 * f3 + v3 * f4);
	vec4 inv2 = vec4(v0 * f1 - v1 * f3 + v3 * f5);
	vec4 inv3 = vec4(v0 * f2 - v1 * f4 + v2 * f5);

	vec4 sa = vec4(+1, -1, +1, -1);
	vec4 sb = vec4(-1, +1, -1, +1);

	mat4 inv = mat4(inv0 * sa, inv1 * sb, inv2 * sa, inv3 * sb);

	vec4 r0 = vec4(inv[0].x, inv[1].x, inv[2].x, inv[3].x);
	vec4 d0 = vec4(m[0] * r0);

	highp float d1 = (d0.x + d0.y) + (d0.z + d0.w);
	highp float d = 1.0 / d1;

	return inv * d;
}

#endif

#ifndef USE_GLES_OVER_GL

#if defined(TRANSPOSE_USED)

highp mat2 transpose(highp mat2 m) {
	return mat2(
			vec2(m[0].x, m[1].x),
			vec2(m[0].y, m[1].y));
}

highp mat3 transpose(highp mat3 m) {
	return mat3(
			vec3(m[0].x, m[1].x, m[2].x),
			vec3(m[0].y, m[1].y, m[2].y),
			vec3(m[0].z, m[1].z, m[2].z));
}

#endif

highp mat4 transpose(highp mat4 m) {
	return mat4(
			vec4(m[0].x, m[1].x, m[2].x, m[3].x),
			vec4(m[0].y, m[1].y, m[2].y, m[3].y),
			vec4(m[0].z, m[1].z, m[2].z, m[3].z),
			vec4(m[0].w, m[1].w, m[2].w, m[3].w));
}

#if defined(OUTER_PRODUCT_USED)

highp mat2 outerProduct(highp vec2 c, highp vec2 r) {
	return mat2(c * r.x, c * r.y);
}

highp mat3 outerProduct(highp vec3 c, highp vec3 r) {
	return mat3(c * r.x, c * r.y, c * r.z);
}

highp mat4 outerProduct(highp vec4 c, highp vec4 r) {
	return mat4(c * r.x, c * r.y, c * r.z, c * r.w);
}

#endif

#endif
