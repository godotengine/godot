// bc7e.ispc - Fast high quality SIMD BC7 encoder
// Copyright (C) 2018-2020 Binomial LLC, All rights reserved. Apache 2.0 license - see LICENSE.
// Typically compiled as: ispc -g -O2 "%(Filename).ispc" -o "$(TargetDir)%(Filename).obj" -h "$(ProjectDir)%(Filename)_ispc.h" --target=sse2,sse4,avx,avx2 --opt=fast-math --opt=disable-assertions
// --opt=fast-math is optional (doesn't make much if any measurable difference). 
// Thanks to ArasP for the determinism fix.

#define BC7E_NON_DETERMINISTIC (0)
#define BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX (34)
#define BC7E_BLOCK_SIZE (16)
#define BC7E_MAX_PARTITIONS0 (16)
#define BC7E_MAX_PARTITIONS1 (64)
#define BC7E_MAX_PARTITIONS2 (64)
#define BC7E_MAX_PARTITIONS3 (64)
#define BC7E_MAX_PARTITIONS7 (64)
#define BC7E_MAX_UBER_LEVEL (4)

typedef unsigned int8 uint8_t;
typedef unsigned int64 uint16_t;
typedef unsigned int32 uint32_t;
typedef unsigned int64 uint64_t;
typedef int8 int8_t;
typedef int int32_t;
typedef int64 int64_t;

#ifndef UINT16_MAX
#define UINT16_MAX (0xFFFF)
#endif

#ifndef UINT_MAX
#define UINT_MAX (0xFFFFFFFFU)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX (0xFFFFFFFFFFFFFFFFULL)
#endif

#ifndef INT64_MAX
#define INT64_MAX (0x7FFFFFFFFFFFFFFFULL)
#endif

struct bc7e_compress_block_params
{
	uint32_t m_max_partitions_mode[8];

	uint32_t m_weights[4];

	uint32_t m_uber_level;
	uint32_t m_refinement_passes;
	
	uint32_t m_mode4_rotation_mask;
	uint32_t m_mode4_index_mask;
	uint32_t m_mode5_rotation_mask;
	uint32_t m_uber1_mask;
	
	bool m_perceptual;
	bool m_pbit_search;
	bool m_mode6_only;
	bool m_unused0;
	
	struct
	{
		uint32_t m_max_mode13_partitions_to_try;
		uint32_t m_max_mode0_partitions_to_try;
		uint32_t m_max_mode2_partitions_to_try;
		bool m_use_mode[7];
		bool m_unused1;
	} m_opaque_settings;

	struct
	{
		uint32_t m_max_mode7_partitions_to_try;
		uint32_t m_mode67_error_weight_mul[4];
				
		bool m_use_mode4;
		bool m_use_mode5;
		bool m_use_mode6;
		bool m_use_mode7;

		bool m_use_mode4_rotation;
		bool m_use_mode5_rotation;
		bool m_unused2;
		bool m_unused3;
	} m_alpha_settings;

};

static inline uniform int32_t clampi(uniform int32_t value, uniform int32_t low, uniform int32_t high) { return clamp(value, low, high); }
static inline uniform uint32_t clampu(uniform uint32_t value, uniform uint32_t low, uniform uint32_t high) { return clamp(value, low, high); }
static inline uniform float clampf(uniform float value, uniform float low, uniform float high) { return clamp(value, low, high); }

static inline int32_t clampi(int32_t value, int32_t low, int32_t high) { return clamp(value, low, high); }
static inline uint32_t clampu(uint32_t value, uint32_t low, uint32_t high) { return clamp(value, low, high); }
static inline float clampf(float value, float low, float high) { return clamp(value, low, high); }

static inline uniform float saturate(uniform float value) { return clampf(value, 0, 1.0f); }
static inline uniform float saturate255(uniform float value) { return clampf(value, 0, 255.0f); }

static inline float saturate(float value) { return clampf(value, 0, 1.0f); }
static inline float saturate255(float value) { return clampf(value, 0, 255.0f); }

static inline uniform uint8_t minimumub(uniform uint8_t a, uniform uint8_t b) { return min(a, b); }
static inline uniform int32_t minimumi(uniform int32_t a, uniform int32_t b) { return min(a, b); }
static inline uniform uint32_t minimumu(uniform uint32_t a, uniform uint32_t b) { return min(a, b); }
static inline uniform uint64_t minimumu64(uniform uint64_t a, uniform uint64_t b) { return min(a, b); }
static inline uniform float minimumf(uniform float a, uniform float b) { return min(a, b); }
				
static inline uniform uint8_t maximumub(uniform uint8_t a, uniform uint8_t b) { return max(a, b); }
static inline uniform int32_t maximumi(uniform int32_t a, uniform int32_t b) { return max(a, b); }
static inline uniform uint32_t maximumu(uniform uint32_t a, uniform uint32_t b) { return max(a, b); }
static inline uniform float maximumf(uniform float a, uniform float b) { return max(a, b); }

static inline uint8_t minimumub(uint8_t a, uint8_t b) { return min(a, b); }
static inline int32_t minimumi(int32_t a, int32_t b) { return min(a, b); }
static inline uint32_t minimumu(uint32_t a, uint32_t b) { return min(a, b); }
static inline uint64_t minimumu64(uint64_t a, uint64_t b) { return min(a, b); }
static inline float minimumf(float a, float b) { return min(a, b); }
				
static inline uint8_t maximumub(uint8_t a, uint8_t b) { return max(a, b); }
static inline int32_t maximumi(int32_t a, int32_t b) { return max(a, b); }
static inline uint32_t maximumu(uint32_t a, uint32_t b) { return max(a, b); }
static inline float maximumf(float a, float b) { return max(a, b); }
				
static inline uniform int32_t iabs32(uniform int32_t v) { uniform uint32_t msk = v >> 31; return (v ^ msk) - msk; }
static inline int32_t iabs32(int32_t v) { uint32_t msk = v >> 31; return (v ^ msk) - msk; }

static inline void swapub(uniform uint8_t * uniform a, uniform uint8_t * uniform b) { uniform uint8_t t = *a; *a = *b; *b = t; }
static inline void swapu(uniform uint32_t * uniform a, uniform uint32_t * uniform b) { uniform uint32_t t = *a; *a = *b; *b = t; }
static inline void swapi(uniform int32_t * uniform a, uniform int32_t * uniform b) { uniform int32_t t = *a; *a = *b; *b = t; }
static inline void swapf(uniform float * uniform a, uniform float * uniform b) { uniform float t = *a; *a = *b; *b = t; }

static inline void swapub(varying uint8_t *uniform a, varying uint8_t *uniform b) { uint8_t t = *a; *a = *b; *b = t; }
static inline void swapi(varying int32_t *uniform a, varying int32_t *uniform b) { int32_t t = *a; *a = *b; *b = t; }
static inline void swapu(varying uint32_t *uniform a, varying uint32_t *uniform b) { uint32_t t = *a; *a = *b; *b = t; }
static inline void swapf(varying float *uniform a, varying float *uniform b) { float t = *a; *a = *b; *b = t; }

static inline float square(float s) { return s * s; }
static inline int square(int s) { return s * s; }

struct color_quad_u8
{
	uint8_t m_c[4];
};

struct color_quad_i
{
	int32_t m_c[4];
};

struct color_quad_f
{
	float m_c[4];
};

static inline color_quad_i component_min_rgb(const varying color_quad_i * uniform pA, const varying color_quad_i * uniform pB) 
{ 
	color_quad_i res; 
	res.m_c[0] = minimumi(pA->m_c[0], pB->m_c[0]);
	res.m_c[1] = minimumi(pA->m_c[1], pB->m_c[1]);
	res.m_c[2] = minimumi(pA->m_c[2], pB->m_c[2]);
	res.m_c[3] = 255;
	return res;
}

static inline color_quad_i component_max_rgb(const varying color_quad_i * uniform pA, const varying color_quad_i * uniform pB) 
{ 
	color_quad_i res;
	res.m_c[0] = maximumi(pA->m_c[0], pB->m_c[0]);
	res.m_c[1] = maximumi(pA->m_c[1], pB->m_c[1]);
	res.m_c[2] = maximumi(pA->m_c[2], pB->m_c[2]);
	res.m_c[3] = 255;
	return res;
}

static inline varying color_quad_i *color_quad_i_set_clamped(varying color_quad_i * uniform pRes, varying int32_t r, varying int32_t g, varying int32_t b, varying int32_t a)
{
	pRes->m_c[0] = clampi(r, 0, 255);
	pRes->m_c[1] = clampi(g, 0, 255);
	pRes->m_c[2] = clampi(b, 0, 255);
	pRes->m_c[3] = clampi(a, 0, 255);
	return pRes;
}

static inline varying color_quad_i *color_quad_i_set(varying color_quad_i * uniform pRes, varying int32_t r, varying int32_t g, varying int32_t b, varying int32_t a)
{
	pRes->m_c[0] = r;
	pRes->m_c[1] = g;
	pRes->m_c[2] = b;
	pRes->m_c[3] = a;
	return pRes;
}

static inline varying color_quad_i *color_quad_i_set(varying color_quad_i * uniform pRes, uniform int32_t r, uniform int32_t g, uniform int32_t b, uniform int32_t a)
{
	pRes->m_c[0] = r;
	pRes->m_c[1] = g;
	pRes->m_c[2] = b;
	pRes->m_c[3] = a;
	return pRes;
}

static inline bool color_quad_i_equals(const varying color_quad_i * uniform pLHS, const varying color_quad_i * uniform pRHS)
{
	return (pLHS->m_c[0] == pRHS->m_c[0]) && (pLHS->m_c[1] == pRHS->m_c[1]) && (pLHS->m_c[2] == pRHS->m_c[2]) && (pLHS->m_c[3] == pRHS->m_c[3]);
}

static inline bool color_quad_i_notequals(const varying color_quad_i * uniform pLHS, const varying color_quad_i * uniform pRHS)
{
	return !color_quad_i_equals(pLHS, pRHS);
}

struct vec4F
{
	float m_c[4];
};

static inline varying vec4F * uniform vec4F_set_scalar(varying vec4F * uniform pV, float x)
{
	pV->m_c[0] = x;
	pV->m_c[1] = x;
	pV->m_c[2] = x;
	pV->m_c[3] = x;
	return pV;
}

static inline varying vec4F * uniform vec4F_set(varying vec4F * uniform pV, float x, float y, float z, float w)
{
	pV->m_c[0] = x;
	pV->m_c[1] = y;
	pV->m_c[2] = z;
	pV->m_c[3] = w;
	return pV;
}

static inline varying vec4F * uniform vec4F_saturate_in_place(varying vec4F * uniform pV)
{
	pV->m_c[0] = saturate(pV->m_c[0]);
	pV->m_c[1] = saturate(pV->m_c[1]);
	pV->m_c[2] = saturate(pV->m_c[2]);
	pV->m_c[3] = saturate(pV->m_c[3]);
	return pV;
}

static inline vec4F vec4F_saturate(const varying vec4F * uniform pV)
{
	vec4F res;
	res.m_c[0] = saturate(pV->m_c[0]);
	res.m_c[1] = saturate(pV->m_c[1]);
	res.m_c[2] = saturate(pV->m_c[2]);
	res.m_c[3] = saturate(pV->m_c[3]);
	return res;
}

static inline vec4F vec4F_from_color(const varying color_quad_i * uniform pC)
{
	vec4F res;
	vec4F_set(&res, pC->m_c[0], pC->m_c[1], pC->m_c[2], pC->m_c[3]);
	return res;
}

static inline vec4F vec4F_add(const varying vec4F * uniform pLHS, const varying vec4F * uniform pRHS)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] + pRHS->m_c[0], pLHS->m_c[1] + pRHS->m_c[1], pLHS->m_c[2] + pRHS->m_c[2], pLHS->m_c[3] + pRHS->m_c[3]);
	return res;
}

static inline vec4F vec4F_sub(const varying vec4F * uniform pLHS, const varying vec4F *uniform pRHS)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] - pRHS->m_c[0], pLHS->m_c[1] - pRHS->m_c[1], pLHS->m_c[2] - pRHS->m_c[2], pLHS->m_c[3] - pRHS->m_c[3]);
	return res;
}

static inline float vec4F_dot(const varying vec4F * uniform pLHS, const varying vec4F * uniform pRHS)
{
	return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2] + pLHS->m_c[3] * pRHS->m_c[3];
}

static inline vec4F vec4F_mul(const varying vec4F * uniform pLHS, float s)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s, pLHS->m_c[3] * s);
	return res;
}

static inline varying vec4F *vec4F_normalize_in_place(varying vec4F * uniform pV)
{
	float s = pV->m_c[0] * pV->m_c[0] + pV->m_c[1] * pV->m_c[1] + pV->m_c[2] * pV->m_c[2] + pV->m_c[3] * pV->m_c[3];
				
	if (s != 0.0f)
	{
		s = 1.0f / sqrt(s);

		pV->m_c[0] *= s;
		pV->m_c[1] *= s;
		pV->m_c[2] *= s;
		pV->m_c[3] *= s;
	}

	return pV;
}

static const uniform uint32_t g_bc7_weights2[4] = { 0, 21, 43, 64 };
static const uniform uint32_t g_bc7_weights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uniform uint32_t g_bc7_weights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

// Precomputed weight constants used during least fit determination. For each entry in g_bc7_weights[]: w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w
static const uniform float g_bc7_weights2x[4 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.107666f, 0.220459f, 0.451416f, 0.328125f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const uniform float g_bc7_weights3x[8 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.079102f, 0.202148f, 0.516602f, 0.281250f, 0.177979f, 0.243896f, 0.334229f, 0.421875f, 0.334229f, 0.243896f, 0.177979f, 0.578125f, 0.516602f, 0.202148f,
	0.079102f, 0.718750f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const uniform float g_bc7_weights4x[16 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.003906f, 0.058594f, 0.878906f, 0.062500f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.041260f, 0.161865f, 0.635010f, 0.203125f, 0.070557f, 0.195068f, 0.539307f, 0.265625f, 0.107666f, 0.220459f,
	0.451416f, 0.328125f, 0.165039f, 0.241211f, 0.352539f, 0.406250f, 0.219727f, 0.249023f, 0.282227f, 0.468750f, 0.282227f, 0.249023f, 0.219727f, 0.531250f, 0.352539f, 0.241211f, 0.165039f, 0.593750f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 0.539307f, 0.195068f, 0.070557f, 0.734375f,
	0.635010f, 0.161865f, 0.041260f, 0.796875f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 0.878906f, 0.058594f, 0.003906f, 0.937500f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };

static const uniform int g_bc7_partition1[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

static const uniform int g_bc7_partition2[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,		0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,		0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,		0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,		0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,		0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
	0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,		0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,		0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,		0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,		0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,		0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,		0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,		0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,		0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,		0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,		0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,		0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,		0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,		0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,		0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,		0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
	0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,		0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,		0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,		0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,		0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,		0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,		0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,		0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
	0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,		0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,		0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,		0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,		0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,		0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
	0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,		0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,		0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,		0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,		0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,		0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,		0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};

static const uniform int g_bc7_table_anchor_index_second_subset[64] =
{
	15,15,15,15,15,15,15,15,		15,15,15,15,15,15,15,15,		15, 2, 8, 2, 2, 8, 8,15,		2, 8, 2, 2, 8, 8, 2, 2,		15,15, 6, 8, 2, 8,15,15,		2, 8, 2, 2, 2,15,15, 6,		6, 2, 6, 8,15,15, 2, 2,		15,15,15,15,15, 2, 2,15
};

static const uniform int g_bc7_partition3[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2,		0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1,		0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1,		0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2,		0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2,		0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1,		0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1,
	0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,		0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,		0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2,		0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2,		0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2,		0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0,
	0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2,		0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0,		0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2,		0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1,		0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2,		0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1,		0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2,		0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0,
	0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0,		0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2,		0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0,		0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1,		0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2,		0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2,		0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1,		0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1,
	0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2,		0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1,		0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2,		0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0,		0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,		0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,		0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0,		0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,
	0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1,		0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1,		0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2,		0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1,		0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1,		0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1,		0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1,
	0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2,		0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1,		0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2,		0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2,		0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2,		0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2,		0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2,
	0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2,		0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2,		0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2,		0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2,		0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1,		0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2,		0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2,		0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0,
};

static const uniform int g_bc7_table_anchor_index_third_subset_1[64] =
{
	3, 3,15,15, 8, 3,15,15,		8, 8, 6, 6, 6, 5, 3, 3,		3, 3, 8,15, 3, 3, 6,10,		5, 8, 8, 6, 8, 5,15,15,		8,15, 3, 5, 6,10, 8,15,		15, 3,15, 5,15,15,15,15,		3,15, 5, 5, 5, 8, 5,10,		5,10, 8,13,15,12, 3, 3
};

static const uniform int g_bc7_table_anchor_index_third_subset_2[64] =
{
	15, 8, 8, 3,15,15, 3, 8,		15,15,15,15,15,15,15, 8,		15, 8,15, 3,15, 8,15, 8,		3,15, 6,10,15,15,10, 8,		15, 3,15,10,10, 8, 9,10,		6,15, 8,15, 3, 6, 6, 8,		15, 3,15,15,15,15,15,15,		15,15,15,15, 3,15,15, 8
};

static const uniform int g_bc7_num_subsets[8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
static const uniform int g_bc7_partition_bits[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
static const uniform int g_bc7_rotation_bits[8] = { 0, 0, 0, 0, 2, 2, 0, 0 };
static const uniform int g_bc7_color_index_bitcount[8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
static int uniform get_bc7_color_index_size(uniform int mode, uniform int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
#pragma ignore warning(perf)
static int varying get_bc7_color_index_size(varying int mode, varying int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
static uniform int g_bc7_alpha_index_bitcount[8] = { 0, 0, 0, 0, 3, 2, 4, 2 };
static uniform int get_bc7_alpha_index_size(uniform int mode, uniform int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
#pragma ignore warning(perf)
static varying int get_bc7_alpha_index_size(varying int mode, varying int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
static const uniform int g_bc7_mode_has_p_bits[8] = { 1, 1, 0, 1, 0, 0, 1, 1 };
static const uniform int g_bc7_mode_has_shared_p_bits[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
static const uniform int g_bc7_color_precision_table[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const uniform int g_bc7_color_precision_plus_pbit_table[8] = { 5, 7, 5, 8, 5, 7, 8, 6 };
static const uniform int g_bc7_alpha_precision_table[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
static const uniform int g_bc7_alpha_precision_plus_pbit_table[8] = { 0, 0, 0, 0, 6, 8, 8, 6 };
static uniform bool get_bc7_mode_has_seperate_alpha_selectors(uniform int mode) { return (mode == 4) || (mode == 5); }
static varying bool get_bc7_mode_has_seperate_alpha_selectors(varying int mode) { return (mode == 4) || (mode == 5); }

struct endpoint_err
{
	uint16_t m_error;
	uint8_t m_lo;
	uint8_t m_hi;
};

static uniform endpoint_err g_bc7_mode_1_optimal_endpoints[256][2]; // [c][pbit]
const uniform uint32_t BC7E_MODE_1_OPTIMAL_INDEX = 2;

static uniform endpoint_err g_bc7_mode_7_optimal_endpoints[256][2][2]; // [c][pbit][hp][lp]
const uniform uint32_t BC7E_MODE_7_OPTIMAL_INDEX = 1;

static uniform endpoint_err g_bc7_mode_6_optimal_endpoints[256][2][2]; // [c][hp][lp]
const uniform uint32_t BC7E_MODE_6_OPTIMAL_INDEX = 5;

static uniform uint32_t g_bc7_mode_4_optimal_endpoints3[256]; // [c]
static uniform uint32_t g_bc7_mode_4_optimal_endpoints2[256]; // [c]
const uniform uint32_t BC7E_MODE_4_OPTIMAL_INDEX3 = 2;
const uniform uint32_t BC7E_MODE_4_OPTIMAL_INDEX2 = 1;

static uniform uint32_t g_bc7_mode_5_optimal_endpoints[256]; // [c]
const uniform uint32_t BC7E_MODE_5_OPTIMAL_INDEX = 1;

static uniform endpoint_err g_bc7_mode_0_optimal_endpoints[256][2][2]; // [c][hp][lp]
const uniform uint32_t BC7E_MODE_0_OPTIMAL_INDEX = 2;

static uniform bool g_codec_initialized;

export void bc7e_compress_block_init()
{
	if (g_codec_initialized)
		return;

	// Mode 0: 444.1
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)
			{
				uniform endpoint_err best;
				best.m_error = (uint16_t)UINT16_MAX;

				for (uniform uint32_t l = 0; l < 16; l++)
				{
					uniform uint32_t low = ((l << 1) | lp) << 3;
					low |= (low >> 5);

					for (uniform uint32_t h = 0; h < 16; h++)
					{
						uniform uint32_t high = ((h << 1) | hp) << 3;
						high |= (high >> 5);

						const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_0_optimal_endpoints[c][hp][lp] = best;
			} // lp
		} // hp
	} // c

	// Mode 1: 666.1
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t lp = 0; lp < 2; lp++)
		{
			uniform endpoint_err best;
			best.m_error = (uint16_t)UINT16_MAX;

			for (uniform uint32_t l = 0; l < 64; l++)
			{
				uniform uint32_t low = ((l << 1) | lp) << 1;
				low |= (low >> 7);

				for (uniform uint32_t h = 0; h < 64; h++)
				{
					uniform uint32_t high = ((h << 1) | lp) << 1;
					high |= (high >> 7);

					const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;

					const uniform int err = (k - c) * (k - c);
					if (err < best.m_error)
					{
						best.m_error = (uint16_t)err;
						best.m_lo = (uint8_t)l;
						best.m_hi = (uint8_t)h;
					}
				} // h
			} // l

			g_bc7_mode_1_optimal_endpoints[c][lp] = best;
		} // lp
	} // c

	// Mode 6: 777.1 4-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)
			{
				uniform endpoint_err best;
				best.m_error = (uint16_t)UINT16_MAX;

				for (uniform uint32_t l = 0; l < 128; l++)
				{
					uniform uint32_t low = (l << 1) | lp;
				
					for (uniform uint32_t h = 0; h < 128; h++)
					{
						uniform uint32_t high = (h << 1) | hp;
					
						const uniform int k = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_6_optimal_endpoints[c][hp][lp] = best;
			} // lp
		} // hp
	} // c

	// Mode 5: 777 2-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 128; l++)
		{
			uniform uint32_t low = l << 1;
			low |= (low >> 7);

			for (uniform uint32_t h = 0; h < 128; h++)
			{
				uniform uint32_t high = h << 1;
				high |= (high >> 7);

				const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_5_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_5_OPTIMAL_INDEX] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_5_optimal_endpoints[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c


	//Mode 4: 555 3-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 32; l++)
		{
			uniform uint32_t low = l << 3;
			low |= (low >> 5);

			for (uniform uint32_t h = 0; h < 32; h++)
			{
				uniform uint32_t high = h << 3;
				high |= (high >> 5);

				const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_4_optimal_endpoints3[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c
	
	// Mode 4: 555 2-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 32; l++)
		{
			uniform uint32_t low = l << 3;
			low |= (low >> 5);

			for (uniform uint32_t h = 0; h < 32; h++)
			{
				uniform uint32_t high = h << 3;
				high |= (high >> 5);

				const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_4_optimal_endpoints2[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c

	// Mode 7: 555.1 2-bit indices 
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)	
			{
				for (uniform uint32_t l = 0; l < 32; l++)
				{
					uniform uint32_t low = ((l << 1) | lp) << 2;
					low |= (low >> 6);

					for (uniform uint32_t h = 0; h < 32; h++)
					{
						uniform uint32_t high = ((h << 1) | hp) << 2;
						high |= (high >> 6);

						const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_7_optimal_endpoints[c][hp][lp] = best;
			
			} // hp

		} // lp

	} // c

	g_codec_initialized = true;
}

static void compute_least_squares_endpoints_rgba(uint32_t N, const varying int *uniform pSelectors, const uniform vec4F *uniform pSelector_weights, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const varying color_quad_i * uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];

#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
		q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
		q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
		q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;
	q10_a = t_a - q00_a;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
	pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
	pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
	pXl->m_c[3] = (float)(iz00 * q00_a + iz01 * q10_a); pXh->m_c[3] = (float)(iz10 * q00_a + iz11 * q10_a);
}

static void compute_least_squares_endpoints_rgb(uint32_t N, const varying int *uniform pSelectors, const uniform vec4F *uniform pSelector_weights, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const varying color_quad_i *uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];
#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
		q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
		q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
	pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
	pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
}

static void compute_least_squares_endpoints_a(uint32_t N, const varying int *uniform pSelectors, const vec4F *uniform pSelector_weights, varying float *uniform pXl, varying float *uniform pXh, const varying color_quad_i *uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];
#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
	}

	q10_a = t_a - q00_a;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	*pXl = (float)(iz00 * q00_a + iz01 * q10_a); *pXh = (float)(iz10 * q00_a + iz11 * q10_a);
}

struct color_cell_compressor_params
{
	uniform uint32_t m_num_selector_weights;
	const uint32_t *uniform m_pSelector_weights;
	const vec4F *uniform m_pSelector_weightsx;
	uniform uint32_t m_comp_bits;
	uniform uint32_t m_weights[4];
	uniform bool m_has_alpha;
	uniform bool m_has_pbits;
	uniform bool m_endpoints_share_pbit;
	uniform bool m_perceptual;
};

static inline void color_cell_compressor_params_clear(uniform color_cell_compressor_params *uniform p)
{
	p->m_num_selector_weights = 0;
	p->m_pSelector_weights = NULL;
	p->m_pSelector_weightsx = NULL;
	p->m_comp_bits = 0;
	p->m_perceptual = false;
	p->m_weights[0] = 1;
	p->m_weights[1] = 1;
	p->m_weights[2] = 1;
	p->m_weights[3] = 1;
	p->m_has_alpha = false;
	p->m_has_pbits = false;
	p->m_endpoints_share_pbit = false;
}

struct color_cell_compressor_results
{
	uint64_t m_best_overall_err;
	color_quad_i m_low_endpoint;
	color_quad_i m_high_endpoint;
	uint32_t m_pbits[2];
	varying int *uniform m_pSelectors;
	varying int *uniform m_pSelectors_temp;
};

static inline color_quad_i scale_color(const varying color_quad_i *uniform pC, const uniform color_cell_compressor_params *uniform pParams)
{
	color_quad_i results;

	const uint32_t n = pParams->m_comp_bits + (pParams->m_has_pbits ? 1 : 0);
	assert((n >= 4) && (n <= 8));

	for (uniform uint32_t i = 0; i < 4; i++)
	{
		uint32_t v = pC->m_c[i] << (8 - n);
#pragma ignore warning(perf)
		v |= (v >> n);
		assert(v <= 255);
		results.m_c[i] = v;
	}

	return results;
}

static const float pr_weight = (.5f / (1.0f - .2126f)) * (.5f / (1.0f - .2126f));
static const float pb_weight = (.5f / (1.0f - .0722f)) * (.5f / (1.0f - .0722f));

static inline uint64_t compute_color_distance_rgb(const varying color_quad_i * uniform pE1, const varying color_quad_i *uniform pE2, uniform bool perceptual, const uint32_t uniform weights[4])
{
	if (perceptual)
	{
		const float l1 = pE1->m_c[0] * .2126f + pE1->m_c[1] * .7152f + pE1->m_c[2] * .0722f;
		const float cr1 = pE1->m_c[0] - l1;
		const float cb1 = pE1->m_c[2] - l1;

		const float l2 = pE2->m_c[0] * .2126f + pE2->m_c[1] * .7152f + pE2->m_c[2] * .0722f;
		const float cr2 = pE2->m_c[0] - l2;
		const float cb2 = pE2->m_c[2] - l2;

		float dl = l1 - l2;
		float dcr = cr1 - cr2;
		float dcb = cb1 - cb2;

		return (int64_t)(weights[0] * (dl * dl) + weights[1] * pr_weight * (dcr * dcr) + weights[2] * pb_weight * (dcb * dcb));
	}
	else
	{
		float dr = (float)pE1->m_c[0] - (float)pE2->m_c[0];
		float dg = (float)pE1->m_c[1] - (float)pE2->m_c[1];
		float db = (float)pE1->m_c[2] - (float)pE2->m_c[2];
		
		return (int64_t)(weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db);
	}
}

static inline uint64_t compute_color_distance_rgba(const varying color_quad_i *uniform pE1, const varying color_quad_i *uniform pE2, uniform bool perceptual, const uint32_t uniform weights[4])
{
	float da = (float)pE1->m_c[3] - (float)pE2->m_c[3];
	float a_err = weights[3] * (da * da);

	if (perceptual)
	{
		const float l1 = pE1->m_c[0] * .2126f + pE1->m_c[1] * .7152f + pE1->m_c[2] * .0722f;
		const float cr1 = pE1->m_c[0] - l1;
		const float cb1 = pE1->m_c[2] - l1;

		const float l2 = pE2->m_c[0] * .2126f + pE2->m_c[1] * .7152f + pE2->m_c[2] * .0722f;
		const float cr2 = pE2->m_c[0] - l2;
		const float cb2 = pE2->m_c[2] - l2;

		float dl = l1 - l2;
		float dcr = cr1 - cr2;
		float dcb = cb1 - cb2;

		return (int64_t)(weights[0] * (dl * dl) + weights[1] * pr_weight * (dcr * dcr) + weights[2] * pb_weight * (dcb * dcb) + a_err);
	}
	else
	{
		float dr = (float)pE1->m_c[0] - (float)pE2->m_c[0];
		float dg = (float)pE1->m_c[1] - (float)pE2->m_c[1];
		float db = (float)pE1->m_c[2] - (float)pE2->m_c[2];
		
		return (int64_t)(weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db + a_err);
	}
}

static uint64_t pack_mode1_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 2; p++)
	{
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_1_optimal_endpoints[r][p].m_error + g_bc7_mode_1_optimal_endpoints[g][p].m_error + g_bc7_mode_1_optimal_endpoints[b][p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	const endpoint_err *pEr = &g_bc7_mode_1_optimal_endpoints[r][best_p];
	const endpoint_err *pEg = &g_bc7_mode_1_optimal_endpoints[g][best_p];
	const endpoint_err *pEb = &g_bc7_mode_1_optimal_endpoints[b][best_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);
#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);
	pResults->m_pbits[0] = best_p;
	pResults->m_pbits[1] = 0;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_1_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		low |= (low >> 7);

		uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		high |= (high >> 7);

		p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;
	}

	p.m_c[3] = 255;

	uint64_t total_err = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t pack_mode24_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t er, eg, eb;

	if (pParams->m_num_selector_weights == 8)
	{
		er = g_bc7_mode_4_optimal_endpoints3[r];
		eg = g_bc7_mode_4_optimal_endpoints3[g];
		eb = g_bc7_mode_4_optimal_endpoints3[b];
	}
	else
	{
		er = g_bc7_mode_4_optimal_endpoints2[r];
		eg = g_bc7_mode_4_optimal_endpoints2[g];
		eb = g_bc7_mode_4_optimal_endpoints2[b];
	}
	
	color_quad_i_set(&pResults->m_low_endpoint, er & 0xFF, eg & 0xFF, eb & 0xFF, 0);
	color_quad_i_set(&pResults->m_high_endpoint, er >> 8, eg >> 8, eb >> 8, 0);

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = (pParams->m_num_selector_weights == 8) ? BC7E_MODE_4_OPTIMAL_INDEX3 : BC7E_MODE_4_OPTIMAL_INDEX2;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = pResults->m_low_endpoint.m_c[i] << 3;
		low |= (low >> 5);

		uint32_t high = pResults->m_high_endpoint.m_c[i] << 3;
		high |= (high >> 5);

		if (pParams->m_num_selector_weights == 8)
			p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;
		else
			p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;
	}
	
	p.m_c[3] = 255;

	uint64_t total_err = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t pack_mode0_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_0_optimal_endpoints[r][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[g][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[b][p >> 1][p & 1].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	const endpoint_err *pEr = &g_bc7_mode_0_optimal_endpoints[r][best_p >> 1][best_p & 1];
	const endpoint_err *pEg = &g_bc7_mode_0_optimal_endpoints[g][best_p >> 1][best_p & 1];
	const endpoint_err *pEb = &g_bc7_mode_0_optimal_endpoints[b][best_p >> 1][best_p & 1];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);

	pResults->m_pbits[0] = best_p & 1;
	pResults->m_pbits[1] = best_p >> 1;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_0_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 3;
		low |= (low >> 5);

		uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1]) << 3;
		high |= (high >> 5);

		p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;
	}
	
	p.m_c[3] = 255;

	uint64_t total_err = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t pack_mode6_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
		uniform uint32_t hi_p = p >> 1;
		uniform uint32_t lo_p = p & 1;
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_6_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[a][hi_p][lo_p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	uint32_t best_hi_p = best_p >> 1;
	uint32_t best_lo_p = best_p & 1;

	const endpoint_err *pEr = &g_bc7_mode_6_optimal_endpoints[r][best_hi_p][best_lo_p];
	const endpoint_err *pEg = &g_bc7_mode_6_optimal_endpoints[g][best_hi_p][best_lo_p];
	const endpoint_err *pEb = &g_bc7_mode_6_optimal_endpoints[b][best_hi_p][best_lo_p];
	const endpoint_err *pEa = &g_bc7_mode_6_optimal_endpoints[a][best_hi_p][best_lo_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

	pResults->m_pbits[0] = best_lo_p;
	pResults->m_pbits[1] = best_hi_p;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_6_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 4; i++)
	{
		uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
		uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
		
		p.m_c[i] = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;
	}

	uint64_t total_err = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgba(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t pack_mode7_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
		uniform uint32_t hi_p = p >> 1;
		uniform uint32_t lo_p = p & 1;
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_7_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[a][hi_p][lo_p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	uint32_t best_hi_p = best_p >> 1;
	uint32_t best_lo_p = best_p & 1;

	const endpoint_err *pEr = &g_bc7_mode_7_optimal_endpoints[r][best_hi_p][best_lo_p];
	const endpoint_err *pEg = &g_bc7_mode_7_optimal_endpoints[g][best_hi_p][best_lo_p];
	const endpoint_err *pEb = &g_bc7_mode_7_optimal_endpoints[b][best_hi_p][best_lo_p];
	const endpoint_err *pEa = &g_bc7_mode_7_optimal_endpoints[a][best_hi_p][best_lo_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

	pResults->m_pbits[0] = best_lo_p;
	pResults->m_pbits[1] = best_hi_p;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_7_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 4; i++)
	{
		uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
		uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
		
		p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;
	}

	uint64_t total_err = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgba(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t evaluate_solution(const varying color_quad_i *uniform pLow, const varying color_quad_i *uniform pHigh, const varying uint32_t *uniform pbits, 
	const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	color_quad_i quantMinColor = *pLow;
	color_quad_i quantMaxColor = *pHigh;

	if (pParams->m_has_pbits)
	{
		uint32_t minPBit, maxPBit;

		if (pParams->m_endpoints_share_pbit)
			maxPBit = minPBit = pbits[0];
		else
		{
			minPBit = pbits[0];
			maxPBit = pbits[1];
		}

		quantMinColor.m_c[0] = (pLow->m_c[0] << 1) | minPBit;
		quantMinColor.m_c[1] = (pLow->m_c[1] << 1) | minPBit;
		quantMinColor.m_c[2] = (pLow->m_c[2] << 1) | minPBit;
		quantMinColor.m_c[3] = (pLow->m_c[3] << 1) | minPBit;

		quantMaxColor.m_c[0] = (pHigh->m_c[0] << 1) | maxPBit;
		quantMaxColor.m_c[1] = (pHigh->m_c[1] << 1) | maxPBit;
		quantMaxColor.m_c[2] = (pHigh->m_c[2] << 1) | maxPBit;
		quantMaxColor.m_c[3] = (pHigh->m_c[3] << 1) | maxPBit;
	}

	color_quad_i actualMinColor = scale_color(&quantMinColor, pParams);
	color_quad_i actualMaxColor = scale_color(&quantMaxColor, pParams);

	const uniform uint32_t N = pParams->m_num_selector_weights;
	const uniform uint32_t nc = pParams->m_has_alpha ? 4 : 3;		

	float total_errf = 0;

	float wr = pParams->m_weights[0];
	float wg = pParams->m_weights[1];
	float wb = pParams->m_weights[2];
	float wa = pParams->m_weights[3];

	color_quad_f weightedColors[16];
	weightedColors[0].m_c[0] = actualMinColor.m_c[0];
	weightedColors[0].m_c[1] = actualMinColor.m_c[1];
	weightedColors[0].m_c[2] = actualMinColor.m_c[2];
	weightedColors[0].m_c[3] = actualMinColor.m_c[3];

	weightedColors[N - 1].m_c[0] = actualMaxColor.m_c[0];
	weightedColors[N - 1].m_c[1] = actualMaxColor.m_c[1];
	weightedColors[N - 1].m_c[2] = actualMaxColor.m_c[2];
	weightedColors[N - 1].m_c[3] = actualMaxColor.m_c[3];
		
	for (uniform uint32_t i = 1; i < (N - 1); i++)
		for (uniform uint32_t j = 0; j < nc; j++)
			weightedColors[i].m_c[j] = floor((weightedColors[0].m_c[j] * (64.0f - pParams->m_pSelector_weights[i]) + weightedColors[N - 1].m_c[j] * pParams->m_pSelector_weights[i] + 32) * (1.0f / 64.0f));

	if (!pParams->m_perceptual)
	{
		if (!pParams->m_has_alpha)
		{
			if (N == 16)
			{
				float lr = actualMinColor.m_c[0];
				float lg = actualMinColor.m_c[1];
				float lb = actualMinColor.m_c[2];

				float dr = actualMaxColor.m_c[0] - lr;
				float dg = actualMaxColor.m_c[1] - lg;
				float db = actualMaxColor.m_c[2] - lb;
			
				const float f = N / (dr * dr + dg * dg + db * db);

				lr *= -dr;
				lg *= -dg;
				lb *= -db;

				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const varying color_quad_i *uniform pC = &pPixels[i];
					float r = pC->m_c[0];
					float g = pC->m_c[1];
					float b = pC->m_c[2];

					float best_sel = floor(((r * dr + lr) + (g * dg + lg) + (b * db + lb)) * f + .5f);
					best_sel = clamp(best_sel, (float)1, (float)(N - 1));

					float best_sel0 = best_sel - 1;

#pragma ignore warning(perf)
					float dr0 = weightedColors[(int)best_sel0].m_c[0] - r;

#pragma ignore warning(perf)
					float dg0 = weightedColors[(int)best_sel0].m_c[1] - g;

#pragma ignore warning(perf)
					float db0 = weightedColors[(int)best_sel0].m_c[2] - b;

					float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;

#pragma ignore warning(perf)
					float dr1 = weightedColors[(int)best_sel].m_c[0] - r;

#pragma ignore warning(perf)
					float dg1 = weightedColors[(int)best_sel].m_c[1] - g;

#pragma ignore warning(perf)
					float db1 = weightedColors[(int)best_sel].m_c[2] - b;

					float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;

					float min_err = min(err0, err1);
					total_errf += min_err;
					pResults->m_pSelectors_temp[i] = (int)select(min_err == err0, best_sel0, best_sel);
				}
			}
			else if (N == 8)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
				
					float best_err;
					int best_sel;

					{
						float dr0 = weightedColors[0].m_c[0] - pr;
						float dg0 = weightedColors[0].m_c[1] - pg;
						float db0 = weightedColors[0].m_c[2] - pb;
						float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;

						float dr1 = weightedColors[1].m_c[0] - pr;
						float dg1 = weightedColors[1].m_c[1] - pg;
						float db1 = weightedColors[1].m_c[2] - pb;
						float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;

						float dr2 = weightedColors[2].m_c[0] - pr;
						float dg2 = weightedColors[2].m_c[1] - pg;
						float db2 = weightedColors[2].m_c[2] - pb;
						float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;

						float dr3 = weightedColors[3].m_c[0] - pr;
						float dg3 = weightedColors[3].m_c[1] - pg;
						float db3 = weightedColors[3].m_c[2] - pb;
						float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;

						best_err = min(min(min(err0, err1), err2), err3);
									
						best_sel = select(best_err == err1, 1, 0);
						best_sel = select(best_err == err2, 2, best_sel);
						best_sel = select(best_err == err3, 3, best_sel);
					}

					{
						float dr0 = weightedColors[4].m_c[0] - pr;
						float dg0 = weightedColors[4].m_c[1] - pg;
						float db0 = weightedColors[4].m_c[2] - pb;
						float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;

						float dr1 = weightedColors[5].m_c[0] - pr;
						float dg1 = weightedColors[5].m_c[1] - pg;
						float db1 = weightedColors[5].m_c[2] - pb;
						float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;

						float dr2 = weightedColors[6].m_c[0] - pr;
						float dg2 = weightedColors[6].m_c[1] - pg;
						float db2 = weightedColors[6].m_c[2] - pb;
						float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;

						float dr3 = weightedColors[7].m_c[0] - pr;
						float dg3 = weightedColors[7].m_c[1] - pg;
						float db3 = weightedColors[7].m_c[2] - pb;
						float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;

						best_err = min(best_err, min(min(min(err0, err1), err2), err3));

						best_sel = select(best_err == err0, 4, best_sel);
						best_sel = select(best_err == err1, 5, best_sel);
						best_sel = select(best_err == err2, 6, best_sel);
						best_sel = select(best_err == err3, 7, best_sel);
					}
				
					total_errf += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
			else // if (N == 4)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
				
					float dr0 = weightedColors[0].m_c[0] - pr;
					float dg0 = weightedColors[0].m_c[1] - pg;
					float db0 = weightedColors[0].m_c[2] - pb;
					float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;

					float dr1 = weightedColors[1].m_c[0] - pr;
					float dg1 = weightedColors[1].m_c[1] - pg;
					float db1 = weightedColors[1].m_c[2] - pb;
					float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;

					float dr2 = weightedColors[2].m_c[0] - pr;
					float dg2 = weightedColors[2].m_c[1] - pg;
					float db2 = weightedColors[2].m_c[2] - pb;
					float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;

					float dr3 = weightedColors[3].m_c[0] - pr;
					float dg3 = weightedColors[3].m_c[1] - pg;
					float db3 = weightedColors[3].m_c[2] - pb;
					float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;

					float best_err = min(min(min(err0, err1), err2), err3);

					int best_sel = select(best_err == err1, 1, 0);
					best_sel = select(best_err == err2, 2, best_sel);
					best_sel = select(best_err == err3, 3, best_sel);
								
					total_errf += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
		}
		else
		{
			// alpha
			if (N == 16)
			{
				float lr = actualMinColor.m_c[0];
				float lg = actualMinColor.m_c[1];
				float lb = actualMinColor.m_c[2];
				float la = actualMinColor.m_c[3];

				float dr = actualMaxColor.m_c[0] - lr;
				float dg = actualMaxColor.m_c[1] - lg;
				float db = actualMaxColor.m_c[2] - lb;
				float da = actualMaxColor.m_c[3] - la;
			
				const float f = N / (dr * dr + dg * dg + db * db + da * da);

				lr *= -dr;
				lg *= -dg;
				lb *= -db;
				la *= -da;

				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const varying color_quad_i *uniform pC = &pPixels[i];
					float r = pC->m_c[0];
					float g = pC->m_c[1];
					float b = pC->m_c[2];
					float a = pC->m_c[3];

					float best_sel = floor(((r * dr + lr) + (g * dg + lg) + (b * db + lb) + (a * da + la)) * f + .5f);
					best_sel = clamp(best_sel, (float)1, (float)(N - 1));

					float best_sel0 = best_sel - 1;

#pragma ignore warning(perf)
					float dr0 = weightedColors[(int)best_sel0].m_c[0] - r;
#pragma ignore warning(perf)
					float dg0 = weightedColors[(int)best_sel0].m_c[1] - g;
#pragma ignore warning(perf)
					float db0 = weightedColors[(int)best_sel0].m_c[2] - b;
#pragma ignore warning(perf)
					float da0 = weightedColors[(int)best_sel0].m_c[3] - a;
					float err0 = (wr * dr0 * dr0) + (wg * dg0 * dg0) + (wb * db0 * db0) + (wa * da0 * da0);

#pragma ignore warning(perf)
					float dr1 = weightedColors[(int)best_sel].m_c[0] - r;
#pragma ignore warning(perf)
					float dg1 = weightedColors[(int)best_sel].m_c[1] - g;
#pragma ignore warning(perf)
					float db1 = weightedColors[(int)best_sel].m_c[2] - b;
#pragma ignore warning(perf)
					float da1 = weightedColors[(int)best_sel].m_c[3] - a;

					float err1 = (wr * dr1 * dr1) + (wg * dg1 * dg1) + (wb * db1 * db1) + (wa * da1 * da1);

					float min_err = min(err0, err1);
					total_errf += min_err;
					pResults->m_pSelectors_temp[i] = (int)select(min_err == err0, best_sel0, best_sel);
				}
			}
			else if (N == 8)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
					float pa = (float)pPixels[i].m_c[3];
				
					float best_err;
					int best_sel;

					{
						float dr0 = weightedColors[0].m_c[0] - pr;
						float dg0 = weightedColors[0].m_c[1] - pg;
						float db0 = weightedColors[0].m_c[2] - pb;
						float da0 = weightedColors[0].m_c[3] - pa;
						float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;

						float dr1 = weightedColors[1].m_c[0] - pr;
						float dg1 = weightedColors[1].m_c[1] - pg;
						float db1 = weightedColors[1].m_c[2] - pb;
						float da1 = weightedColors[1].m_c[3] - pa;
						float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;

						float dr2 = weightedColors[2].m_c[0] - pr;
						float dg2 = weightedColors[2].m_c[1] - pg;
						float db2 = weightedColors[2].m_c[2] - pb;
						float da2 = weightedColors[2].m_c[3] - pa;
						float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;

						float dr3 = weightedColors[3].m_c[0] - pr;
						float dg3 = weightedColors[3].m_c[1] - pg;
						float db3 = weightedColors[3].m_c[2] - pb;
						float da3 = weightedColors[3].m_c[3] - pa;
						float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;

						best_err = min(min(min(err0, err1), err2), err3);
									
						best_sel = select(best_err == err1, 1, 0);
						best_sel = select(best_err == err2, 2, best_sel);
						best_sel = select(best_err == err3, 3, best_sel);
					}

					{
						float dr0 = weightedColors[4].m_c[0] - pr;
						float dg0 = weightedColors[4].m_c[1] - pg;
						float db0 = weightedColors[4].m_c[2] - pb;
						float da0 = weightedColors[4].m_c[3] - pa;
						float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;

						float dr1 = weightedColors[5].m_c[0] - pr;
						float dg1 = weightedColors[5].m_c[1] - pg;
						float db1 = weightedColors[5].m_c[2] - pb;
						float da1 = weightedColors[5].m_c[3] - pa;
						float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;

						float dr2 = weightedColors[6].m_c[0] - pr;
						float dg2 = weightedColors[6].m_c[1] - pg;
						float db2 = weightedColors[6].m_c[2] - pb;
						float da2 = weightedColors[6].m_c[3] - pa;
						float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;

						float dr3 = weightedColors[7].m_c[0] - pr;
						float dg3 = weightedColors[7].m_c[1] - pg;
						float db3 = weightedColors[7].m_c[2] - pb;
						float da3 = weightedColors[7].m_c[3] - pa;
						float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;

						best_err = min(best_err, min(min(min(err0, err1), err2), err3));

						best_sel = select(best_err == err0, 4, best_sel);
						best_sel = select(best_err == err1, 5, best_sel);
						best_sel = select(best_err == err2, 6, best_sel);
						best_sel = select(best_err == err3, 7, best_sel);
					}
				
					total_errf += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
			else // if (N == 4)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
					float pa = (float)pPixels[i].m_c[3];
				
					float dr0 = weightedColors[0].m_c[0] - pr;
					float dg0 = weightedColors[0].m_c[1] - pg;
					float db0 = weightedColors[0].m_c[2] - pb;
					float da0 = weightedColors[0].m_c[3] - pa;
					float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;

					float dr1 = weightedColors[1].m_c[0] - pr;
					float dg1 = weightedColors[1].m_c[1] - pg;
					float db1 = weightedColors[1].m_c[2] - pb;
					float da1 = weightedColors[1].m_c[3] - pa;
					float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;

					float dr2 = weightedColors[2].m_c[0] - pr;
					float dg2 = weightedColors[2].m_c[1] - pg;
					float db2 = weightedColors[2].m_c[2] - pb;
					float da2 = weightedColors[2].m_c[3] - pa;
					float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;

					float dr3 = weightedColors[3].m_c[0] - pr;
					float dg3 = weightedColors[3].m_c[1] - pg;
					float db3 = weightedColors[3].m_c[2] - pb;
					float da3 = weightedColors[3].m_c[3] - pa;
					float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;

					float best_err = min(min(min(err0, err1), err2), err3);

					int best_sel = select(best_err == err1, 1, 0);
					best_sel = select(best_err == err2, 2, best_sel);
					best_sel = select(best_err == err3, 3, best_sel);
								
					total_errf += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
		}
	}
	else
	{
		wg *= pr_weight;
		wb *= pb_weight;

		float weightedColorsY[16], weightedColorsCr[16], weightedColorsCb[16];
		
		for (uniform uint32_t i = 0; i < N; i++)
		{
			float r = weightedColors[i].m_c[0];
			float g = weightedColors[i].m_c[1];
			float b = weightedColors[i].m_c[2];

			float y = r * .2126f + g * .7152f + b * .0722f;
									
			weightedColorsY[i] = y;
			weightedColorsCr[i] = r - y;
			weightedColorsCb[i] = b - y;
		}

		if (pParams->m_has_alpha)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				float r = pPixels[i].m_c[0];
				float g = pPixels[i].m_c[1];
				float b = pPixels[i].m_c[2];
				float a = pPixels[i].m_c[3];

				float y = r * .2126f + g * .7152f + b * .0722f;
				float cr = r - y;
				float cb = b - y;

				float best_err = 1e+10f;
				int32_t best_sel;
								
				for (uniform uint32_t j = 0; j < N; j++)
				{
					float dl = y - weightedColorsY[j];
					float dcr = cr - weightedColorsCr[j];
					float dcb = cb - weightedColorsCb[j];
					float da = a - weightedColors[j].m_c[3];

					float err = (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da);
					if (err < best_err)
					{
						best_err = err;
						best_sel = j;
					}
				}
				
				total_errf += best_err;

				pResults->m_pSelectors_temp[i] = best_sel;
			}
		}
		else
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				float r = pPixels[i].m_c[0];
				float g = pPixels[i].m_c[1];
				float b = pPixels[i].m_c[2];

				float y = r * .2126f + g * .7152f + b * .0722f;
				float cr = r - y;
				float cb = b - y;

				float best_err = 1e+10f;
				int32_t best_sel;
								
				for (uniform uint32_t j = 0; j < N; j++)
				{
					float dl = y - weightedColorsY[j];
					float dcr = cr - weightedColorsCr[j];
					float dcb = cb - weightedColorsCb[j];

					float err = (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb);
					if (err < best_err)
					{
						best_err = err;
						best_sel = j;
					}
				}
				
				total_errf += best_err;

				pResults->m_pSelectors_temp[i] = best_sel;
			}
		}
	}

	uint64_t total_err = (int64)total_errf;

	if (total_err < pResults->m_best_overall_err)
	{
		pResults->m_best_overall_err = total_err;

		pResults->m_low_endpoint = *pLow;
		pResults->m_high_endpoint = *pHigh;

		pResults->m_pbits[0] = pbits[0];
		pResults->m_pbits[1] = pbits[1];

		for (uniform uint32_t i = 0; i < num_pixels; i++)
			pResults->m_pSelectors[i] = pResults->m_pSelectors_temp[i];
	}
				
	return total_err;
}

static void fixDegenerateEndpoints(uniform uint32_t mode, varying color_quad_i *uniform pTrialMinColor, varying color_quad_i *uniform pTrialMaxColor, const varying vec4F *uniform pXl, const varying vec4F *uniform pXh, uniform uint32_t iscale)
{
	if ((mode == 1) || (mode == 4)) // also mode 2
	{
		// fix degenerate case where the input collapses to a single colorspace voxel, and we loose all freedom (test with grayscale ramps)
		for (uniform uint32_t i = 0; i < 3; i++)
		{
			if (pTrialMinColor->m_c[i] == pTrialMaxColor->m_c[i])
			{
				if (abs(pXl->m_c[i] - pXh->m_c[i]) > 0.0f)
				{
					if (pTrialMinColor->m_c[i] > (iscale >> 1))
					{
						if (pTrialMinColor->m_c[i] > 0)
							pTrialMinColor->m_c[i]--;
						else
							if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
					}
					else
					{
						if (pTrialMaxColor->m_c[i] < iscale)
							pTrialMaxColor->m_c[i]++;
						else if (pTrialMinColor->m_c[i] > 0)
							pTrialMinColor->m_c[i]--;
					}

					if (mode == 4)
					{
						if (pTrialMinColor->m_c[i] > (iscale >> 1))
						{
							if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
							else if (pTrialMinColor->m_c[i] > 0)
								pTrialMinColor->m_c[i]--;
						}
						else
						{
							if (pTrialMinColor->m_c[i] > 0)
								pTrialMinColor->m_c[i]--;
							else if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
						}
					}
				}
			}
		}
	}
}

static uint64_t find_optimal_solution(uniform uint32_t mode, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, 
	uniform bool pbit_search, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	vec4F xl = *pXl;
	vec4F xh = *pXh;

	vec4F_saturate_in_place(&xl);
	vec4F_saturate_in_place(&xh);
		
	if (pParams->m_has_pbits)
	{
		if (pbit_search)
		{
			// compensated rounding+pbit search
			const uniform int iscalep = (1 << (pParams->m_comp_bits + 1)) - 1;
			const uniform float scalep = (float)iscalep;

			const uniform int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

			if (!pParams->m_endpoints_share_pbit)
			{
				color_quad_i lo[2], hi[2];
								
				for (uniform int p = 0; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;

					// Notes: The pbit controls which quantization intervals are selected.
					// total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
					// pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
					// rearranging you get for pbit 0: b=floor(v*(total_levels-1)/2+.5)
					// rearranging you get for pbit 1: b=floor((v*(total_levels-1)-1)/2+.5)
					for (uniform uint32_t c = 0; c < 4; c++)
					{
						xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMinColor.m_c[c] = clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

						xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMaxColor.m_c[c] = clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
					}
																				
					lo[p] = xMinColor;
					hi[p] = xMaxColor;

					for (uniform int c = 0; c < 4; c++)
					{
						lo[p].m_c[c] >>= 1;
						hi[p].m_c[c] >>= 1;
					}
				}

				fixDegenerateEndpoints(mode, &lo[0], &hi[0], &xl, &xh, iscalep >> 1);
				fixDegenerateEndpoints(mode, &lo[1], &hi[1], &xl, &xh, iscalep >> 1);

				uint32_t pbits[2];
				
				pbits[0] = 0; pbits[1] = 0;
				evaluate_solution(&lo[0], &hi[0], pbits, pParams, pResults, num_pixels, pPixels);

				pbits[0] = 0; pbits[1] = 1;
				evaluate_solution(&lo[0], &hi[1], pbits, pParams, pResults, num_pixels, pPixels);

				pbits[0] = 1; pbits[1] = 0;
				evaluate_solution(&lo[1], &hi[0], pbits, pParams, pResults, num_pixels, pPixels);
				
				pbits[0] = 1; pbits[1] = 1;
				evaluate_solution(&lo[1], &hi[1], pbits, pParams, pResults, num_pixels, pPixels);
			}
			else
			{
				// Endpoints share pbits
				color_quad_i lo[2], hi[2];

				for (uniform int p = 0; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;
								
					for (uniform uint32_t c = 0; c < 4; c++)
					{
						xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMinColor.m_c[c] = clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

						xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMaxColor.m_c[c] = clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
					}
										
					lo[p] = xMinColor;
					hi[p] = xMaxColor;

					for (uniform int c = 0; c < 4; c++)
					{
						lo[p].m_c[c] >>= 1;
						hi[p].m_c[c] >>= 1;
					}
				}

				fixDegenerateEndpoints(mode, &lo[0], &hi[0], &xl, &xh, iscalep >> 1);
				fixDegenerateEndpoints(mode, &lo[1], &hi[1], &xl, &xh, iscalep >> 1);
				
				uint32_t pbits[2];
				
				pbits[0] = 0; pbits[1] = 0;
				evaluate_solution(&lo[0], &hi[0], pbits, pParams, pResults, num_pixels, pPixels);

				pbits[0] = 1; pbits[1] = 1;
				evaluate_solution(&lo[1], &hi[1], pbits, pParams, pResults, num_pixels, pPixels);
			}
		}
		else
		{
			// compensated rounding
			const uniform int iscalep = (1 << (pParams->m_comp_bits + 1)) - 1;
			const uniform float scalep = (float)iscalep;

			const uniform int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

			uint32_t best_pbits[2];
			color_quad_i bestMinColor, bestMaxColor;
						
			if (!pParams->m_endpoints_share_pbit)
			{
				float best_err0 = 1e+9;
				float best_err1 = 1e+9;
								
				for (uniform int p = 0; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;

					// Notes: The pbit controls which quantization intervals are selected.
					// total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
					// pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
					// rearranging you get for pbit 0: b=floor(v*(total_levels-1)/2+.5)
					// rearranging you get for pbit 1: b=floor((v*(total_levels-1)-1)/2+.5)
					for (uniform uint32_t c = 0; c < 4; c++)
					{
						xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMinColor.m_c[c] = clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

						xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMaxColor.m_c[c] = clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
					}
																				
					color_quad_i scaledLow = scale_color(&xMinColor, pParams);
					color_quad_i scaledHigh = scale_color(&xMaxColor, pParams);

					float err0 = 0;
					float err1 = 0;
					for (uniform int i = 0; i < totalComps; i++)
					{
						err0 += square(scaledLow.m_c[i] - xl.m_c[i]*255.0f);
						err1 += square(scaledHigh.m_c[i] - xh.m_c[i]*255.0f);
					}

					if (err0 < best_err0)
					{
						best_err0 = err0;
						best_pbits[0] = p;
						
						bestMinColor.m_c[0] = xMinColor.m_c[0] >> 1;
						bestMinColor.m_c[1] = xMinColor.m_c[1] >> 1;
						bestMinColor.m_c[2] = xMinColor.m_c[2] >> 1;
						bestMinColor.m_c[3] = xMinColor.m_c[3] >> 1;
					}

					if (err1 < best_err1)
					{
						best_err1 = err1;
						best_pbits[1] = p;

						bestMaxColor.m_c[0] = xMaxColor.m_c[0] >> 1;
						bestMaxColor.m_c[1] = xMaxColor.m_c[1] >> 1;
						bestMaxColor.m_c[2] = xMaxColor.m_c[2] >> 1;
						bestMaxColor.m_c[3] = xMaxColor.m_c[3] >> 1;
					}
				}
			}
			else
			{
				// Endpoints share pbits
				float best_err = 1e+9;

				for (uniform int p = 0; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;
								
					for (uniform uint32_t c = 0; c < 4; c++)
					{
						xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMinColor.m_c[c] = clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

						xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
						xMaxColor.m_c[c] = clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
					}
										
					color_quad_i scaledLow = scale_color(&xMinColor, pParams);
					color_quad_i scaledHigh = scale_color(&xMaxColor, pParams);

					float err = 0;
					for (uniform int i = 0; i < totalComps; i++)
						err += square((scaledLow.m_c[i]/255.0f) - xl.m_c[i]) + square((scaledHigh.m_c[i]/255.0f) - xh.m_c[i]);

					if (err < best_err)
					{
						best_err = err;
						best_pbits[0] = p;
						best_pbits[1] = p;
						
						bestMinColor.m_c[0] = xMinColor.m_c[0] >> 1;
						bestMinColor.m_c[1] = xMinColor.m_c[1] >> 1;
						bestMinColor.m_c[2] = xMinColor.m_c[2] >> 1;
						bestMinColor.m_c[3] = xMinColor.m_c[3] >> 1;

						bestMaxColor.m_c[0] = xMaxColor.m_c[0] >> 1;
						bestMaxColor.m_c[1] = xMaxColor.m_c[1] >> 1;
						bestMaxColor.m_c[2] = xMaxColor.m_c[2] >> 1;
						bestMaxColor.m_c[3] = xMaxColor.m_c[3] >> 1;
					}
				}
			}

			fixDegenerateEndpoints(mode, &bestMinColor, &bestMaxColor, &xl, &xh, iscalep >> 1);

			if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_i_notequals(&bestMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&bestMaxColor, &pResults->m_high_endpoint) || (best_pbits[0] != pResults->m_pbits[0]) || (best_pbits[1] != pResults->m_pbits[1]))
			{
				evaluate_solution(&bestMinColor, &bestMaxColor, best_pbits, pParams, pResults, num_pixels, pPixels);
			}
		}
	}
	else
	{
		const uniform int iscale = (1 << pParams->m_comp_bits) - 1;
		const uniform float scale = (float)iscale;

		color_quad_i trialMinColor, trialMaxColor;
		color_quad_i_set_clamped(&trialMinColor, (int)(xl.m_c[0] * scale + .5f), (int)(xl.m_c[1] * scale + .5f), (int)(xl.m_c[2] * scale + .5f), (int)(xl.m_c[3] * scale + .5f));
		color_quad_i_set_clamped(&trialMaxColor, (int)(xh.m_c[0] * scale + .5f), (int)(xh.m_c[1] * scale + .5f), (int)(xh.m_c[2] * scale + .5f), (int)(xh.m_c[3] * scale + .5f));

		fixDegenerateEndpoints(mode, &trialMinColor, &trialMaxColor, &xl, &xh, iscale);

		if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_i_notequals(&trialMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&trialMaxColor, &pResults->m_high_endpoint))
		{
			uint32_t pbits[2];
			pbits[0] = 0;
			pbits[1] = 0;

			evaluate_solution(&trialMinColor, &trialMaxColor, pbits, pParams, pResults, num_pixels, pPixels);
		}
	}

	return pResults->m_best_overall_err;
}

// Note: In mode 6, m_has_alpha will only be true for transparent blocks.
static uint64_t color_cell_compression(uniform uint32_t mode, const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, 
	const uniform bc7e_compress_block_params *uniform pComp_params, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, uniform bool refinement)
{
	pResults->m_best_overall_err = UINT64_MAX;

	if ((mode != 6) && (mode != 7))
	{
		assert(!pParams->m_has_alpha);
	}

	if ((mode <= 2) || (mode == 4) || (mode >= 6))
	{
		const uint32_t cr = pPixels[0].m_c[0];
		const uint32_t cg = pPixels[0].m_c[1];
		const uint32_t cb = pPixels[0].m_c[2];
		const uint32_t ca = pPixels[0].m_c[3];

		bool allSame = true;
		for (uniform uint32_t i = 1; i < num_pixels; i++)
		{
			if ((cr != pPixels[i].m_c[0]) || (cg != pPixels[i].m_c[1]) || (cb != pPixels[i].m_c[2]) || (ca != pPixels[i].m_c[3]))
			{
				allSame = false;
				break;
			}
		}

		cif (allSame)
		{
			if (mode == 0)
				return pack_mode0_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
			if (mode == 1)
				return pack_mode1_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
			else if (mode == 6)
				return pack_mode6_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
			else if (mode == 7)
				return pack_mode7_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
			else
				return pack_mode24_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
		}
	}

	vec4F meanColor, axis;
	vec4F_set_scalar(&meanColor, 0.0f);

	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		vec4F color = vec4F_from_color(&pPixels[i]);
		meanColor = vec4F_add(&meanColor, &color);
	}
				
	vec4F meanColorScaled = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels));

	meanColor = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels * 255.0f));
	vec4F_saturate_in_place(&meanColor);

	if (pParams->m_has_alpha)
	{
		vec4F v;
		vec4F_set_scalar(&v, 0.0f);
		cfor (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			vec4F color = vec4F_from_color(&pPixels[i]);
			color = vec4F_sub(&color, &meanColorScaled);

			vec4F a = vec4F_mul(&color, color.m_c[0]);
			vec4F b = vec4F_mul(&color, color.m_c[1]);
			vec4F c = vec4F_mul(&color, color.m_c[2]);
			vec4F d = vec4F_mul(&color, color.m_c[3]);

			vec4F n = i ? v : color;
			vec4F_normalize_in_place(&n);

			v.m_c[0] += vec4F_dot(&a, &n);
			v.m_c[1] += vec4F_dot(&b, &n);
			v.m_c[2] += vec4F_dot(&c, &n);
			v.m_c[3] += vec4F_dot(&d, &n);
		}
		axis = v;
		vec4F_normalize_in_place(&axis);
	}
	else
	{
		float cov[6];
		cov[0] = 0; cov[1] = 0; cov[2] = 0; 
		cov[3] = 0; cov[4] = 0;	cov[5] = 0;

		cfor (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *varying pV = &pPixels[i];

			float r = pV->m_c[0] - meanColorScaled.m_c[0];
			float g = pV->m_c[1] - meanColorScaled.m_c[1];
			float b = pV->m_c[2] - meanColorScaled.m_c[2];
				
			cov[0] += r*r;
			cov[1] += r*g;
			cov[2] += r*b;
			cov[3] += g*g;
			cov[4] += g*b;
			cov[5] += b*b;
		}

		float vfr, vfg, vfb;
		//vfr = hi[0] - lo[0];
		//vfg = hi[1] - lo[1];
		//vfb = hi[2] - lo[2];
		// This is more stable.
		vfr = .9f;
		vfg = 1.0f;
		vfb = .7f;

		for (uniform uint32_t iter = 0; iter < 3; iter++)
		{
			float r = vfr*cov[0] + vfg*cov[1] + vfb*cov[2];
			float g = vfr*cov[1] + vfg*cov[3] + vfb*cov[4];
			float b = vfr*cov[2] + vfg*cov[4] + vfb*cov[5];

			float m = maximumf(maximumf(abs(r), abs(g)), abs(b));
			if (m > 1e-10f)
			{
				m = 1.0f / m;
				r *= m;
				g *= m;
				b *= m;
			}

			//float delta = square(vfr - r) + square(vfg - g) + square(vfb - b);

			vfr = r;
			vfg = g;
			vfb = b;

			//if ((iter > 1) && (delta < 1e-8f))
			//	break;
		}

		float len = vfr*vfr + vfg*vfg + vfb*vfb;

		if (len < 1e-10f)
			vec4F_set_scalar(&axis, 0.0f);
		else
		{
			len = 1.0f / sqrt(len);
			vfr *= len;
			vfg *= len;
			vfb *= len;
			vec4F_set(&axis, vfr, vfg, vfb, 0);
		}
	}

	cif (vec4F_dot(&axis, &axis) < .5f)
	{
		if (pParams->m_perceptual)
			vec4F_set(&axis, .213f, .715f, .072f, pParams->m_has_alpha ? .715f : 0);
		else
			vec4F_set(&axis, 1.0f, 1.0f, 1.0f, pParams->m_has_alpha ? 1.0f : 0);
		vec4F_normalize_in_place(&axis);
	}

	float l = 1e+9f, h = -1e+9f;

	cfor (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		vec4F color = vec4F_from_color(&pPixels[i]);

		vec4F q = vec4F_sub(&color, &meanColorScaled);
		float d = vec4F_dot(&q, &axis);

		l = minimumf(l, d);
		h = maximumf(h, d);
	}

	l *= (1.0f / 255.0f);
	h *= (1.0f / 255.0f);

	vec4F b0 = vec4F_mul(&axis, l);
	vec4F b1 = vec4F_mul(&axis, h);
	vec4F c0 = vec4F_add(&meanColor, &b0);
	vec4F c1 = vec4F_add(&meanColor, &b1);
	vec4F minColor = vec4F_saturate(&c0);
	vec4F maxColor = vec4F_saturate(&c1);
				
	vec4F whiteVec;
	vec4F_set_scalar(&whiteVec, 1.0f);
	if (vec4F_dot(&minColor, &whiteVec) > vec4F_dot(&maxColor, &whiteVec))
	{
		vec4F temp = minColor;
		minColor = maxColor;
		maxColor = temp;
	}

	if (!find_optimal_solution(mode, &minColor, &maxColor, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels))
		return 0;
	
	if (!refinement)
		return pResults->m_best_overall_err;
	
	for (uniform uint32_t i = 0; i < pComp_params->m_refinement_passes; i++)
	{
		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);
		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
		else
		{
			compute_least_squares_endpoints_rgb(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);

			xl.m_c[3] = 255.0f;
			xh.m_c[3] = 255.0f;
		}

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels))
			return 0;
	}

	if (pComp_params->m_uber_level > 0)
	{
		int selectors_temp[16], selectors_temp1[16];
		for (uniform uint32_t i = 0; i < num_pixels; i++)
			selectors_temp[i] = pResults->m_pSelectors[i];

		const uniform int max_selector = pParams->m_num_selector_weights - 1;

		uint32_t min_sel = 16;
		uint32_t max_sel = 0;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			min_sel = minimumu(min_sel, sel);
			max_sel = maximumu(max_sel, sel);
		}

		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);

		if (pComp_params->m_uber1_mask & 1)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
					sel++;
				selectors_temp1[i] = sel;
			}
						
			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels))
				return 0;
		}

		if (pComp_params->m_uber1_mask & 2)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == max_sel) && (sel > 0))
					sel--;
				selectors_temp1[i] = sel;
			}

			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels))
				return 0;
		}

		if (pComp_params->m_uber1_mask & 4)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
					sel++;
				else if ((sel == max_sel) && (sel > 0))
					sel--;
				selectors_temp1[i] = sel;
			}

			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels))
				return 0;
		}

		const uint32_t uber_err_thresh = (num_pixels * 56) >> 4;
		if ((pComp_params->m_uber_level >= 2) && (pResults->m_best_overall_err > uber_err_thresh))
		{
			const uniform int Q = (pComp_params->m_uber_level >= 4) ? (pComp_params->m_uber_level - 2) : 1;
			for (uniform int ly = -Q; ly <= 1; ly++)
			{
				for (uniform int hy = max_selector - 1; hy <= (max_selector + Q); hy++)
				{
					if ((ly == 0) && (hy == max_selector))
						continue;

					for (uniform uint32_t i = 0; i < num_pixels; i++)
						selectors_temp1[i] = (int)clampf(floor((float)max_selector * ((float)(int)selectors_temp[i] - (float)ly) / ((float)hy - (float)ly) + .5f), 0, (float)max_selector);

					vec4F_set_scalar(&xl, 0.0f);
					vec4F_set_scalar(&xh, 0.0f);
					if (pParams->m_has_alpha)
						compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
					else
					{
						compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
						xl.m_c[3] = 255.0f;
						xh.m_c[3] = 255.0f;
					}

					xl = vec4F_mul(&xl, (1.0f / 255.0f));
					xh = vec4F_mul(&xh, (1.0f / 255.0f));

					if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search && (pComp_params->m_uber_level >= 2), num_pixels, pPixels))
						return 0;
				}
			}
		}
	}

	if ((mode <= 2) || (mode == 4) || (mode >= 6))
	{
		color_cell_compressor_results avg_results;
					
		avg_results.m_best_overall_err = pResults->m_best_overall_err;
		avg_results.m_pSelectors = pResults->m_pSelectors;
		avg_results.m_pSelectors_temp = pResults->m_pSelectors_temp;
								 
		const uint32_t r = (int)(.5f + meanColor.m_c[0] * 255.0f);
		const uint32_t g = (int)(.5f + meanColor.m_c[1] * 255.0f);
		const uint32_t b = (int)(.5f + meanColor.m_c[2] * 255.0f);
		const uint32_t a = (int)(.5f + meanColor.m_c[3] * 255.0f);

		uint64_t avg_err;
		if (mode == 0)
			avg_err = pack_mode0_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 1)
			avg_err = pack_mode1_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 6)
			avg_err = pack_mode6_to_one_color(pParams, &avg_results, r, g, b, a, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 7)
			avg_err = pack_mode7_to_one_color(pParams, &avg_results, r, g, b, a, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else
			avg_err = pack_mode24_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels);

		if (avg_err < pResults->m_best_overall_err)
		{
			pResults->m_best_overall_err = avg_err;
			pResults->m_low_endpoint = avg_results.m_low_endpoint;
			pResults->m_high_endpoint = avg_results.m_high_endpoint;
			pResults->m_pbits[0] = avg_results.m_pbits[0];
			pResults->m_pbits[1] = avg_results.m_pbits[1];

			for (uniform uint32_t i = 0; i < num_pixels; i++)
				pResults->m_pSelectors[i] = pResults->m_pSelectors_temp[i];
		}
	}
					
	return pResults->m_best_overall_err;
}

static uint64_t color_cell_compression_est(uniform uint32_t mode, const uniform color_cell_compressor_params *uniform pParams, uint64_t best_err_so_far, uniform uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	assert((pParams->m_num_selector_weights == 4) || (pParams->m_num_selector_weights == 8));

	float lr = 255, lg = 255, lb = 255;
	float hr = 0, hg = 0, hb = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		const varying color_quad_i *uniform pC = &pPixels[i];

		float r = pC->m_c[0];
		float g = pC->m_c[1];
		float b = pC->m_c[2];
		
		lr = min(lr, r);
		lg = min(lg, g);
		lb = min(lb, b);

		hr = max(hr, r);
		hg = max(hg, g);
		hb = max(hb, b);
	}
			
	const uniform uint32_t N = 1 << g_bc7_color_index_bitcount[mode];
						
	uint64_t total_err = 0;
	
	float sr = lr;
	float sg = lg;
	float sb = lb;

	float dir = hr - lr;
	float dig = hg - lg;
	float dib = hb - lb;	

	float far = dir;
	float fag = dig;
	float fab = dib;

	float low = far * sr + fag * sg + fab * sb;
	float high = far * hr + fag * hg + fab * hb;

	float scale = ((float)N - 1) / (float)(high - low);
	float inv_n = 1.0f / ((float)N - 1);

	float total_errf = 0;

	// We don't handle perceptual very well here, but the difference is very slight (<.05 dB avg Luma PSNR across a large corpus) and the perf lost was high (2x slower).
	if ((pParams->m_weights[0] != 1) || (pParams->m_weights[1] != 1) || (pParams->m_weights[2] != 1))
	{
		float wr = pParams->m_weights[0];
		float wg = pParams->m_weights[1];
		float wb = pParams->m_weights[2];

		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *uniform pC = &pPixels[i];

			float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2];

			float s = clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

			float itr = sr + dir * s;
			float itg = sg + dig * s;
			float itb = sb + dib * s;

			float dr = itr - (float)pC->m_c[0];
			float dg = itg - (float)pC->m_c[1];
			float db = itb - (float)pC->m_c[2];

			total_errf += wr * dr * dr + wg * dg * dg + wb * db * db;
		}
	}
	else
	{
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *uniform pC = &pPixels[i];

			float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2];

			float s = clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

			float itr = sr + dir * s;
			float itg = sg + dig * s;
			float itb = sb + dib * s;

			float dr = itr - (float)pC->m_c[0];
			float dg = itg - (float)pC->m_c[1];
			float db = itb - (float)pC->m_c[2];

			total_errf += dr * dr + dg * dg + db * db;
		}
	}

	total_err = (int64_t)total_errf;

	return total_err;
}

static uint64_t color_cell_compression_est_mode7(uniform uint32_t mode, const uniform color_cell_compressor_params *uniform pParams, uint64_t best_err_so_far, uniform uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	assert((mode == 7) && (pParams->m_num_selector_weights == 4));

	float lr = 255, lg = 255, lb = 255, la = 255;
	float hr = 0, hg = 0, hb = 0, ha = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		const varying color_quad_i *uniform pC = &pPixels[i];

		float r = pC->m_c[0];
		float g = pC->m_c[1];
		float b = pC->m_c[2];
		float a = pC->m_c[3];
		
		lr = min(lr, r);
		lg = min(lg, g);
		lb = min(lb, b);
		la = min(la, a);

		hr = max(hr, r);
		hg = max(hg, g);
		hb = max(hb, b);
		ha = max(ha, a);
	}
			
	const uniform uint32_t N = 4;
						
	uint64_t total_err = 0;
	
	float sr = lr;
	float sg = lg;
	float sb = lb;
	float sa = la;

	float dir = hr - lr;
	float dig = hg - lg;
	float dib = hb - lb;	
	float dia = ha - la;	

	float far = dir;
	float fag = dig;
	float fab = dib;
	float faa = dia;

	float low = far * sr + fag * sg + fab * sb + faa * sa;
	float high = far * hr + fag * hg + fab * hb + faa * ha;

	float scale = ((float)N - 1) / (float)(high - low);
	float inv_n = 1.0f / ((float)N - 1);

	float total_errf = 0;

	// We don't handle perceptual very well here, but the difference is very slight (<.05 dB avg Luma PSNR across a large corpus) and the perf lost was high (2x slower).
	if ( (!pParams->m_perceptual) && ((pParams->m_weights[0] != 1) || (pParams->m_weights[1] != 1) || (pParams->m_weights[2] != 1) || (pParams->m_weights[3] != 1)) )
	{
		float wr = pParams->m_weights[0];
		float wg = pParams->m_weights[1];
		float wb = pParams->m_weights[2];
		float wa = pParams->m_weights[3];

		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *uniform pC = &pPixels[i];

			float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2] + faa * (float)pC->m_c[3];

			float s = clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

			float itr = sr + dir * s;
			float itg = sg + dig * s;
			float itb = sb + dib * s;
			float ita = sa + dia * s;

			float dr = itr - (float)pC->m_c[0];
			float dg = itg - (float)pC->m_c[1];
			float db = itb - (float)pC->m_c[2];
			float da = ita - (float)pC->m_c[3];

			total_errf += wr * dr * dr + wg * dg * dg + wb * db * db + wa * da * da;
		}
	}
	else
	{
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *uniform pC = &pPixels[i];

			float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2] + faa * (float)pC->m_c[3];

			float s = clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

			float itr = sr + dir * s;
			float itg = sg + dig * s;
			float itb = sb + dib * s;
			float ita = sa + dia * s;

			float dr = itr - (float)pC->m_c[0];
			float dg = itg - (float)pC->m_c[1];
			float db = itb - (float)pC->m_c[2];
			float da = ita - (float)pC->m_c[3];

			total_errf += dr * dr + dg * dg + db * db + da * da;
		}
	}

	total_err = (int64_t)total_errf;

	return total_err;
}

static uint32_t estimate_partition(uniform uint32_t mode, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params)
{
	const uniform uint32_t total_subsets = g_bc7_num_subsets[mode];
	uniform uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode], 1U << g_bc7_partition_bits[mode]);

	if (total_partitions <= 1)
		return 0;

	uint64_t best_err = UINT64_MAX;
	uint32_t best_partition = 0;

	uniform color_cell_compressor_params params;
	color_cell_compressor_params_clear(&params);

	params.m_pSelector_weights = (g_bc7_color_index_bitcount[mode] == 2) ? g_bc7_weights2 : g_bc7_weights3;
	params.m_num_selector_weights = 1 << g_bc7_color_index_bitcount[mode];

	memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));
	
	if (mode >= 6)
	{
		params.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
	}

	params.m_perceptual = pComp_params->m_perceptual;

	for (uniform uint32_t partition = 0; partition < total_partitions; partition++)
	{
		const int *uniform pPartition = (total_subsets == 3) ? &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

		varying color_quad_i subset_colors[3][16];
		uniform uint32_t subset_total_colors[3];
		subset_total_colors[0] = 0;
		subset_total_colors[1] = 0;
		subset_total_colors[2] = 0;
		
		for (uniform uint32_t index = 0; index < 16; index++)
		{
			const uniform uint32_t p = pPartition[index];

			subset_colors[p][subset_total_colors[p]] = pPixels[index];
			subset_total_colors[p]++;
		}

		uint64_t total_subset_err = 0;

		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			uint64_t err;
			if (mode == 7)
				err = color_cell_compression_est_mode7(mode, &params, best_err, subset_total_colors[subset], &subset_colors[subset][0]);
			else
				err = color_cell_compression_est(mode, &params, best_err, subset_total_colors[subset], &subset_colors[subset][0]);

			total_subset_err += err;

		} // subset

		if (total_subset_err < best_err)
		{
			best_err = total_subset_err;
			best_partition = partition;
			if (!best_err)
				break;
		}

		if (total_subsets == 2)
		{
			if ((partition == BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX) && (best_partition != BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX))
				break;
		}

	} // partition

	return best_partition;
}

struct solution
{
	uint32_t m_index;
	uint64_t m_err;
};

static uniform uint32_t estimate_partition_list(uniform uint32_t mode, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, 
	varying solution *uniform pSolutions, uniform int32_t max_solutions)
{
	const uniform int32_t orig_max_solutions = max_solutions;

	const uniform uint32_t total_subsets = g_bc7_num_subsets[mode];
	uniform uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode], 1U << g_bc7_partition_bits[mode]);

	if (total_partitions <= 1)
	{
		pSolutions[0].m_index = 0;
		pSolutions[0].m_err = 0;
		return 1;
	}
	else if (max_solutions >= total_partitions)
	{
		for (uniform int i = 0; i < total_partitions; i++)
		{
			pSolutions[i].m_index = i;
			pSolutions[i].m_err = i;
		}
		return total_partitions;
	}

	const uniform int32_t HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD = 4;
	if (total_subsets == 2)
	{
		if (max_solutions < HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD)
			max_solutions = HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD;
	}
						
	uniform color_cell_compressor_params params;
	color_cell_compressor_params_clear(&params);

	params.m_pSelector_weights = (g_bc7_color_index_bitcount[mode] == 2) ? g_bc7_weights2 : g_bc7_weights3;
	params.m_num_selector_weights = 1 << g_bc7_color_index_bitcount[mode];

	memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));

	if (mode >= 6)
	{
		params.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
	}

	params.m_perceptual = pComp_params->m_perceptual;

	uniform int32_t num_solutions = 0;

	for (uniform uint32_t partition = 0; partition < total_partitions; partition++)
	{
		const int *uniform pPartition = (total_subsets == 3) ? &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

		varying color_quad_i subset_colors[3][16];
		uniform uint32_t subset_total_colors[3];
		subset_total_colors[0] = 0;
		subset_total_colors[1] = 0;
		subset_total_colors[2] = 0;

		for (uniform uint32_t index = 0; index < 16; index++)
		{
			const uniform uint32_t p = pPartition[index];

			subset_colors[p][subset_total_colors[p]] = pPixels[index];
			subset_total_colors[p]++;
		}
				
		uint64_t total_subset_err = 0;

		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			uint64_t err;
			if (mode == 7)
				err = color_cell_compression_est_mode7(mode, &params, UINT64_MAX, subset_total_colors[subset], &subset_colors[subset][0]);
			else
				err = color_cell_compression_est(mode, &params, UINT64_MAX, subset_total_colors[subset], &subset_colors[subset][0]);

			total_subset_err += err;

		} // subset

		int32_t i;
		for (i = 0; i < num_solutions; i++)
		{
#pragma ignore warning(perf)
			if (total_subset_err < pSolutions[i].m_err)
				break;
		}
						
		if (i < num_solutions)
		{
			int32_t solutions_to_move = (max_solutions - 1) - i;
			int32_t num_elements_at_i = num_solutions - i;
			if (solutions_to_move > num_elements_at_i)
				solutions_to_move = num_elements_at_i;
																
			assert(((i + 1) + solutions_to_move) <= max_solutions);
			assert((i + solutions_to_move) <= num_solutions);
			
			for (int32_t j = solutions_to_move - 1; j >= 0; --j)
			{
#pragma ignore warning(perf)
				pSolutions[i + j + 1] = pSolutions[i + j];
			}
		}

		if (num_solutions < max_solutions)
			num_solutions++;

		if (i < num_solutions)
		{
#pragma ignore warning(perf)
			pSolutions[i].m_err = total_subset_err;

#pragma ignore warning(perf)
			pSolutions[i].m_index = partition;
		}

#if BC7E_NON_DETERMINISTIC
		if ((total_subsets == 2) && (partition == BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX))
		{
			if (all(i >= HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD))
				break;
		}
#endif

	} // partition

#if 0
	for (uniform int i = 0; i < num_solutions; i++)
	{
		assert(pSolutions[i].m_index < total_partitions);
	}

	for (uniform int i = 0; i < (num_solutions - 1); i++)
	{
		assert(pSolutions[i].m_err <= pSolutions[i + 1].m_err);
	}
#endif

	return min(num_solutions, orig_max_solutions);
}

static inline void set_block_bits(uint8_t *pBytes, uint32_t val, uint32_t num_bits, varying uint32_t *uniform pCur_ofs)
{
	assert(num_bits < 32);
	uint32_t limit = 1U << num_bits;
	assert(val < limit);
		
	while (num_bits)
	{
		const uint32_t n = minimumu(8 - (*pCur_ofs & 7), num_bits);

#pragma ignore warning(perf)
		pBytes[*pCur_ofs >> 3] |= (uint8_t)(val << (*pCur_ofs & 7));

		val >>= n;
		num_bits -= n;
		*pCur_ofs += n;
	}

	assert(*pCur_ofs <= 128);
}

struct bc7_optimization_results
{
	uint32_t m_mode;
	uint32_t m_partition;
	int m_selectors[16];
	int m_alpha_selectors[16];
	color_quad_i m_low[3];
	color_quad_i m_high[3];
	uint32_t m_pbits[3][2];
	uint32_t m_rotation;
	uint32_t m_index_selector;
};

static void encode_bc7_block(void *pBlock, const varying bc7_optimization_results *uniform pResults)
{
	const uint32_t best_mode = pResults->m_mode;

#pragma ignore warning(perf)
	const uint32_t total_subsets = g_bc7_num_subsets[best_mode];

#pragma ignore warning(perf)
	const uint32_t total_partitions = 1 << g_bc7_partition_bits[best_mode];

	const int *pPartition;
	if (total_subsets == 1)
		pPartition = &g_bc7_partition1[0];
	else if (total_subsets == 2)
		pPartition = &g_bc7_partition2[pResults->m_partition * 16];
	else
		pPartition = &g_bc7_partition3[pResults->m_partition * 16];

	int color_selectors[16];
	for (uniform int i = 0; i < 16; i++)
		color_selectors[i] = pResults->m_selectors[i];

	int alpha_selectors[16];
	for (uniform int i = 0; i < 16; i++)
		alpha_selectors[i] = pResults->m_alpha_selectors[i];

	color_quad_i low[3], high[3];
	low[0] = pResults->m_low[0];
	low[1] = pResults->m_low[1];
	low[2] = pResults->m_low[2];

	high[0] = pResults->m_high[0];
	high[1] = pResults->m_high[1];
	high[2] = pResults->m_high[2];
	
	uint32_t pbits[3][2];
	for (uniform int i = 0; i < 3; i++)
	{
		pbits[i][0] = pResults->m_pbits[i][0];
		pbits[i][1] = pResults->m_pbits[i][1];
	}

	int anchor[3];
	anchor[0] = -1;
	anchor[1] = -1;
	anchor[2] = -1;

	for (uniform uint32_t k = 0; k < total_subsets; k++)
	{
		uint32_t anchor_index = 0;
		if (k)
		{
			if ((total_subsets == 3) && (k == 1))
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_third_subset_1[pResults->m_partition];
			}
			else if ((total_subsets == 3) && (k == 2))
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_third_subset_2[pResults->m_partition];
			}
			else
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_second_subset[pResults->m_partition];
			}
		}

		anchor[k] = anchor_index;

		const uint32_t color_index_bits = get_bc7_color_index_size(best_mode, pResults->m_index_selector);
		const uint32_t num_color_indices = 1 << color_index_bits;

#pragma ignore warning(perf)
		if (color_selectors[anchor_index] & (num_color_indices >> 1))
		{
			for (uniform uint32_t i = 0; i < 16; i++)
			{
#pragma ignore warning(perf)
				if (pPartition[i] == k)
					color_selectors[i] = (num_color_indices - 1) - color_selectors[i];
			}

			if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
			{
				for (uniform uint32_t q = 0; q < 3; q++)
				{
					int t = low[k].m_c[q];
					low[k].m_c[q] = high[k].m_c[q];
					high[k].m_c[q] = t;
				}
			}
			else
			{
				color_quad_i tmp = low[k];
				low[k] = high[k];
				high[k] = tmp;
			}

#pragma ignore warning(perf)
			if (!g_bc7_mode_has_shared_p_bits[best_mode])
			{
				uint32_t t = pbits[k][0];
				pbits[k][0] = pbits[k][1];
				pbits[k][1] = t;
			}
		}

		if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
		{
			const uint32_t alpha_index_bits = get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);
			const uint32_t num_alpha_indices = 1 << alpha_index_bits;

#pragma ignore warning(perf)
			if (alpha_selectors[anchor_index] & (num_alpha_indices >> 1))
			{
				for (uniform uint32_t i = 0; i < 16; i++)
				{
#pragma ignore warning(perf)
					if (pPartition[i] == k)
						alpha_selectors[i] = (num_alpha_indices - 1) - alpha_selectors[i];
				}

				int t = low[k].m_c[3];
				low[k].m_c[3] = high[k].m_c[3];
				high[k].m_c[3] = t;
			}
		}
	}

	uint8_t *pBlock_bytes = (uint8_t *)(pBlock);
	memset(pBlock_bytes, 0, BC7E_BLOCK_SIZE);

	uint32_t cur_bit_ofs = 0;
		
	set_block_bits(pBlock_bytes, 1 << best_mode, best_mode + 1, &cur_bit_ofs);

	if ((best_mode == 4) || (best_mode == 5))
		set_block_bits(pBlock_bytes, pResults->m_rotation, 2, &cur_bit_ofs);

	if (best_mode == 4)
		set_block_bits(pBlock_bytes, pResults->m_index_selector, 1, &cur_bit_ofs);

	if (total_partitions > 1)
		set_block_bits(pBlock_bytes, pResults->m_partition, (total_partitions == 64) ? 6 : 4, &cur_bit_ofs);

	const uint32_t total_comps = (best_mode >= 4) ? 4 : 3;
	for (uniform uint32_t comp = 0; comp < total_comps; comp++)
	{
		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, low[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
			set_block_bits(pBlock_bytes, high[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
		}
	}

#pragma ignore warning(perf)
	if (g_bc7_mode_has_p_bits[best_mode])
	{
		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, pbits[subset][0], 1, &cur_bit_ofs);
#pragma ignore warning(perf)
			if (!g_bc7_mode_has_shared_p_bits[best_mode])
				set_block_bits(pBlock_bytes, pbits[subset][1], 1, &cur_bit_ofs);
		}
	}

	for (uniform uint32_t y = 0; y < 4; y++)
	{
		for (uniform uint32_t x = 0; x < 4; x++)
		{
			uniform int idx = x + y * 4;

			uint32_t n = pResults->m_index_selector ? get_bc7_alpha_index_size(best_mode, pResults->m_index_selector) : get_bc7_color_index_size(best_mode, pResults->m_index_selector);

			if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
				n--;

			set_block_bits(pBlock_bytes, pResults->m_index_selector ? alpha_selectors[idx] : color_selectors[idx], n, &cur_bit_ofs);
		}
	}

	if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
	{
		for (uniform uint32_t y = 0; y < 4; y++)
		{
			for (uniform uint32_t x = 0; x < 4; x++)
			{
				int idx = x + y * 4;

				uint32_t n = pResults->m_index_selector ? get_bc7_color_index_size(best_mode, pResults->m_index_selector) : get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);

				if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
					n--;

				set_block_bits(pBlock_bytes, pResults->m_index_selector ? color_selectors[idx] : alpha_selectors[idx], n, &cur_bit_ofs);
			}
		}
	}

	assert(cur_bit_ofs == 128);
}

static inline void encode_bc7_block_mode6(void *pBlock, varying bc7_optimization_results *uniform pResults)
{
	color_quad_i low, high;
	uint32_t pbits[2];
		
	uint32_t invert_selectors = 0;
	if (pResults->m_selectors[0] & 8)
	{
		invert_selectors = 15;
							
		low = pResults->m_high[0];
		high = pResults->m_low[0];

		pbits[0] = pResults->m_pbits[0][1];
		pbits[1] = pResults->m_pbits[0][0];
	}
	else
	{
		low = pResults->m_low[0];
		high = pResults->m_high[0];

		pbits[0] = pResults->m_pbits[0][0];
		pbits[1] = pResults->m_pbits[0][1];
	}

	uint64_t l = 0, h = 0;

	l = 1 << 6;

	l |= (low.m_c[0] << 7);
	l |= (high.m_c[0] << 14);

	l |= (low.m_c[1] << 21);
	l |= ((uint64_t)high.m_c[1] << 28);

	l |= ((uint64_t)low.m_c[2] << 35);
	l |= ((uint64_t)high.m_c[2] << 42);

	l |= ((uint64_t)low.m_c[3] << 49);
	l |= ((uint64_t)high.m_c[3] << 56);

	l |= ((uint64_t)pbits[0] << 63);
		
	h = pbits[1];
		
	h |= ((invert_selectors ^ pResults->m_selectors[0]) << 1);

	// TODO: Just invert all these bits in one single operation, not as individual
	h |= ((invert_selectors ^ pResults->m_selectors[1]) << 4);
	h |= ((invert_selectors ^ pResults->m_selectors[2]) << 8);
	h |= ((invert_selectors ^ pResults->m_selectors[3]) << 12);
	h |= ((invert_selectors ^ pResults->m_selectors[4]) << 16);
	
	h |= ((invert_selectors ^ pResults->m_selectors[5]) << 20);
	h |= ((invert_selectors ^ pResults->m_selectors[6]) << 24);
	h |= ((invert_selectors ^ pResults->m_selectors[7]) << 28);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[8]) << 32);

	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[9]) << 36);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[10]) << 40);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[11]) << 44);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[12]) << 48);

	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[13]) << 52);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[14]) << 56);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[15]) << 60);

#pragma ignore warning(perf)	
	((uint64_t *)(pBlock))[0] = l;

#pragma ignore warning(perf)
	((uint64_t *)(pBlock))[1] = h;
}

static void handle_alpha_block_mode4(const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a, 
	varying bc7_optimization_results *uniform pOpt_results4, varying uint64_t *uniform pMode4_err)
{
	pParams->m_has_alpha = false;
	pParams->m_comp_bits = 5;
	pParams->m_has_pbits = false;
	pParams->m_endpoints_share_pbit = false;				
	pParams->m_perceptual = pComp_params->m_perceptual;

	for (uniform uint32_t index_selector = 0; index_selector < 2; index_selector++)
	{
		if ((pComp_params->m_mode4_index_mask & (1 << index_selector)) == 0)
			continue;

		if (index_selector)
		{
			pParams->m_pSelector_weights = g_bc7_weights3;
			pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights3x[0];
			pParams->m_num_selector_weights = 8;
		}
		else
		{
			pParams->m_pSelector_weights = g_bc7_weights2;
			pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights2x[0];
			pParams->m_num_selector_weights = 4;
		}
								
		color_cell_compressor_results results;
		
		int selectors[16];
		results.m_pSelectors = selectors;

		int selectors_temp[16];
		results.m_pSelectors_temp = selectors_temp;
				
		uint64_t trial_err = color_cell_compression(4, pParams, &results, pComp_params, 16, pPixels, true);
		assert(trial_err == results.m_best_overall_err);

		uint32_t la = minimumi((lo_a + 2) >> 2, 63);
		uint32_t ha = minimumi((hi_a + 2) >> 2, 63);

		if (la == ha)
		{
			if (lo_a != hi_a)
			{
				if (ha != 63)
					ha++;
				else if (la != 0)
					la--;
			}
		}

		uint64_t best_alpha_err = UINT64_MAX;
		uint32_t best_la = 0, best_ha = 0;
		int best_alpha_selectors[16];
						
		for (uniform int32_t pass = 0; pass < 2; pass++)
		{
			int32_t vals[8];

			if (index_selector == 0)
			{
				vals[0] = (la << 2) | (la >> 4);
				vals[7] = (ha << 2) | (ha >> 4);

				for (uniform uint32_t i = 1; i < 7; i++)
					vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
			}
			else
			{
				vals[0] = (la << 2) | (la >> 4);
				vals[3] = (ha << 2) | (ha >> 4);

				const uniform int32_t w_s1 = 21, w_s2 = 43;
				vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
				vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
			}

			uint64_t trial_alpha_err = 0;

			int trial_alpha_selectors[16];
			for (uniform uint32_t i = 0; i < 16; i++)
			{
				const int32_t a = pPixels[i].m_c[3];

				int s = 0;
				int32_t be = iabs32(a - vals[0]);

				int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
				e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
				e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

				if (index_selector == 0)
				{
					e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; }
					e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; }
					e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; }
					e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; }
				}

				trial_alpha_err += (be * be) * pParams->m_weights[3];

				trial_alpha_selectors[i] = s;
			}

			if (trial_alpha_err < best_alpha_err)
			{
				best_alpha_err = trial_alpha_err;
				best_la = la;
				best_ha = ha;
				for (uniform uint32_t i = 0; i < 16; i++)
					best_alpha_selectors[i] = trial_alpha_selectors[i];
			}

			if (pass == 0) 
			{
				float xl, xh;
				compute_least_squares_endpoints_a(16, trial_alpha_selectors, index_selector ? (const vec4F * uniform)&g_bc7_weights2x[0] : (const vec4F * uniform)&g_bc7_weights3x[0], &xl, &xh, pPixels);
				if (xl > xh)
					swapf(&xl, &xh);
				la = clampi((int)floor(xl * (63.0f / 255.0f) + .5f), 0, 63);
				ha = clampi((int)floor(xh * (63.0f / 255.0f) + .5f), 0, 63);
			}
						
		} // pass

		if (pComp_params->m_uber_level > 0)
		{
			const uniform int D = min((int)pComp_params->m_uber_level, 3);
			for (uniform int ld = -D; ld <= D; ld++)
			{
				for (uniform int hd = -D; hd <= D; hd++)
				{
					la = clamp((int)best_la + ld, 0, 63);
					ha = clamp((int)best_ha + hd, 0, 63);
					
					int32_t vals[8];

					if (index_selector == 0)
					{
						vals[0] = (la << 2) | (la >> 4);
						vals[7] = (ha << 2) | (ha >> 4);

						for (uniform uint32_t i = 1; i < 7; i++)
							vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
					}
					else
					{
						vals[0] = (la << 2) | (la >> 4);
						vals[3] = (ha << 2) | (ha >> 4);

						const uniform int32_t w_s1 = 21, w_s2 = 43;
						vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
						vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
					}

					uint64_t trial_alpha_err = 0;

					int trial_alpha_selectors[16];
					for (uniform uint32_t i = 0; i < 16; i++)
					{
						const int32_t a = pPixels[i].m_c[3];

						int s = 0;
						int32_t be = iabs32(a - vals[0]);

						int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
						e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
						e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

						if (index_selector == 0)
						{
							e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; }
							e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; }
							e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; }
							e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; }
						}

						trial_alpha_err += (be * be) * pParams->m_weights[3];

						trial_alpha_selectors[i] = s;
					}

					if (trial_alpha_err < best_alpha_err)
					{
						best_alpha_err = trial_alpha_err;
						best_la = la;
						best_ha = ha;
						for (uniform uint32_t i = 0; i < 16; i++)
							best_alpha_selectors[i] = trial_alpha_selectors[i];
					}
				
				} // hd

			} // ld
		}

		trial_err += best_alpha_err;

		if (trial_err < *pMode4_err)
		{
			*pMode4_err = trial_err;

			pOpt_results4->m_mode = 4;
			pOpt_results4->m_index_selector = index_selector;
			pOpt_results4->m_rotation = 0;
			pOpt_results4->m_partition = 0;

			pOpt_results4->m_low[0] = results.m_low_endpoint;
			pOpt_results4->m_high[0] = results.m_high_endpoint;
			pOpt_results4->m_low[0].m_c[3] = best_la;
			pOpt_results4->m_high[0].m_c[3] = best_ha;

			for (uniform uint32_t i = 0; i < 16; i++)
				pOpt_results4->m_selectors[i] = selectors[i];

			for (uniform uint32_t i = 0; i < 16; i++)
				pOpt_results4->m_alpha_selectors[i] = best_alpha_selectors[i];
		}

	} // index_selector
}

static void handle_alpha_block_mode5(const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a, 
	varying bc7_optimization_results *uniform pOpt_results5, varying uint64_t *uniform pMode5_err)
{
	pParams->m_pSelector_weights = g_bc7_weights2;
	pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights2x[0];
	pParams->m_num_selector_weights = 4;

	pParams->m_comp_bits = 7;
	pParams->m_has_alpha = false;
	pParams->m_has_pbits = false;
	pParams->m_endpoints_share_pbit = false;				
	
	pParams->m_perceptual = pComp_params->m_perceptual;
		
	color_cell_compressor_results results5;
	results5.m_pSelectors = pOpt_results5->m_selectors;

	int selectors_temp[16];
	results5.m_pSelectors_temp = selectors_temp;

	*pMode5_err = color_cell_compression(5, pParams, &results5, pComp_params, 16, pPixels, true);
	assert(*pMode5_err == results5.m_best_overall_err);

	pOpt_results5->m_low[0] = results5.m_low_endpoint;
	pOpt_results5->m_high[0] = results5.m_high_endpoint;

	cif (lo_a == hi_a)
	{
		pOpt_results5->m_low[0].m_c[3] = lo_a;
		pOpt_results5->m_high[0].m_c[3] = hi_a;
		for (uniform uint32_t i = 0; i < 16; i++)
			pOpt_results5->m_alpha_selectors[i] = 0;
	}
	else
	{
		uint64_t mode5_alpha_err = UINT64_MAX;

		for (uniform uint32_t pass = 0; pass < 2; pass++)
		{
			int32_t vals[4];
			vals[0] = lo_a;
			vals[3] = hi_a;

			const uniform int32_t w_s1 = 21, w_s2 = 43;
			vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
			vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

			int trial_alpha_selectors[16];

			uint64_t trial_alpha_err = 0;
			for (uniform uint32_t i = 0; i < 16; i++)
			{
				const int32_t a = pPixels[i].m_c[3];

				int s = 0;
				int32_t be = iabs32(a - vals[0]);
				int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
				e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
				e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

				trial_alpha_selectors[i] = s;
								
				trial_alpha_err += (be * be) * pParams->m_weights[3];
			}

			if (trial_alpha_err < mode5_alpha_err)
			{
				mode5_alpha_err = trial_alpha_err;
				pOpt_results5->m_low[0].m_c[3] = lo_a;
				pOpt_results5->m_high[0].m_c[3] = hi_a;
				for (uniform uint32_t i = 0; i < 16; i++)
					pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
			}

			if (!pass)
			{
				float xl, xh;
				compute_least_squares_endpoints_a(16, trial_alpha_selectors, (const vec4F * uniform)&g_bc7_weights2x[0], &xl, &xh, pPixels);

				uint32_t new_lo_a = clampi((int)floor(xl + .5f), 0, 255);
				uint32_t new_hi_a = clampi((int)floor(xh + .5f), 0, 255);
				if (new_lo_a > new_hi_a)
					swapu(&new_lo_a, &new_hi_a);

				if ((new_lo_a == lo_a) && (new_hi_a == hi_a))
					break;

				lo_a = new_lo_a;
				hi_a = new_hi_a;
			}
		}

		if (pComp_params->m_uber_level > 0)
		{
			const uniform int D = min((int)pComp_params->m_uber_level, 3);
			for (uniform int ld = -D; ld <= D; ld++)
			{
				for (uniform int hd = -D; hd <= D; hd++)
				{
					lo_a = clamp((int)pOpt_results5->m_low[0].m_c[3] + ld, 0, 255);
					hi_a = clamp((int)pOpt_results5->m_high[0].m_c[3] + hd, 0, 255);
					
					int32_t vals[4];
					vals[0] = lo_a;
					vals[3] = hi_a;

					const uniform int32_t w_s1 = 21, w_s2 = 43;
					vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
					vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

					int trial_alpha_selectors[16];

					uint64_t trial_alpha_err = 0;
					for (uniform uint32_t i = 0; i < 16; i++)
					{
						const int32_t a = pPixels[i].m_c[3];

						int s = 0;
						int32_t be = iabs32(a - vals[0]);
						int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
						e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
						e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

						trial_alpha_selectors[i] = s;
								
						trial_alpha_err += (be * be) * pParams->m_weights[3];
					}

					if (trial_alpha_err < mode5_alpha_err)
					{
						mode5_alpha_err = trial_alpha_err;
						pOpt_results5->m_low[0].m_c[3] = lo_a;
						pOpt_results5->m_high[0].m_c[3] = hi_a;
						for (uniform uint32_t i = 0; i < 16; i++)
							pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
					}
				
				} // hd

			} // ld
		}

		*pMode5_err += mode5_alpha_err;
	}

	pOpt_results5->m_mode = 5;
	pOpt_results5->m_index_selector = 0;
	pOpt_results5->m_rotation = 0;
	pOpt_results5->m_partition = 0;
}

static void handle_alpha_block(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a)
{
	pParams->m_perceptual = pComp_params->m_perceptual;

	bc7_optimization_results opt_results;
	
	uint64_t best_err = UINT64_MAX;
		
	// Mode 4
	if (pComp_params->m_alpha_settings.m_use_mode4)
	{
		uniform color_cell_compressor_params params4 = *pParams;

		const uniform int num_rotations = (pComp_params->m_perceptual || (!pComp_params->m_alpha_settings.m_use_mode4_rotation)) ? 1 : 4;
		for (uniform uint32_t rotation = 0; rotation < num_rotations; rotation++)
		{
			if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0)
				continue;

			memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
			if (rotation)
				swapu(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
							
			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results4;

			uint64_t trial_mode4_err = best_err;

			handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);

			if (trial_mode4_err < best_err)
			{
				best_err = trial_mode4_err;

				opt_results.m_mode = 4;
				opt_results.m_index_selector = trial_opt_results4.m_index_selector;
				opt_results.m_rotation = rotation;
				opt_results.m_partition = 0;

				opt_results.m_low[0] = trial_opt_results4.m_low[0];
				opt_results.m_high[0] = trial_opt_results4.m_high[0];

				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
				
				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
			}
		} // rotation
	}
	
	// Mode 6
	if (pComp_params->m_alpha_settings.m_use_mode6)
	{
		uniform color_cell_compressor_params params6 = *pParams;

		params6.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params6.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params6.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params6.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];

		color_cell_compressor_results results6;
		
		params6.m_pSelector_weights = g_bc7_weights4;
		params6.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
		params6.m_num_selector_weights = 16;

		params6.m_comp_bits = 7;
		params6.m_has_pbits = true;
		params6.m_endpoints_share_pbit = false;
		params6.m_has_alpha = true;
				
		int selectors[16];
		results6.m_pSelectors = selectors;

		int selectors_temp[16];
		results6.m_pSelectors_temp = selectors_temp;
				
		uint64_t mode6_err = color_cell_compression(6, &params6, &results6, pComp_params, 16, pPixels, true);
		assert(mode6_err == results6.m_best_overall_err);
		
		if (mode6_err < best_err)
		{
			best_err = mode6_err;

			opt_results.m_mode = 6;
			opt_results.m_index_selector = 0;
			opt_results.m_rotation = 0;
			opt_results.m_partition = 0;

			opt_results.m_low[0] = results6.m_low_endpoint;
			opt_results.m_high[0] = results6.m_high_endpoint;

			opt_results.m_pbits[0][0] = results6.m_pbits[0];
			opt_results.m_pbits[0][1] = results6.m_pbits[1];

			for (uniform int i = 0; i < 16; i++)
				opt_results.m_selectors[i] = selectors[i];
		}
	}

	// Mode 5
	if (pComp_params->m_alpha_settings.m_use_mode5)
	{
		uniform color_cell_compressor_params params5 = *pParams;

		const uniform int num_rotations = (pComp_params->m_perceptual || (!pComp_params->m_alpha_settings.m_use_mode5_rotation)) ? 1 : 4;
		for (uniform uint32_t rotation = 0; rotation < num_rotations; rotation++)
		{
			if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0)
				continue;

			memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
			if (rotation)
				swapu(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results5;

			uint64_t trial_mode5_err = 0;

			handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);

			if (trial_mode5_err < best_err)
			{
				best_err = trial_mode5_err;

				opt_results = trial_opt_results5;
				opt_results.m_rotation = rotation;
			}
		} // rotation
	}

	// Mode 7
	if (pComp_params->m_alpha_settings.m_use_mode7)
	{
		solution solutions[BC7E_MAX_PARTITIONS7];
		uniform uint32_t num_solutions = estimate_partition_list(7, pPixels, pComp_params, solutions, pComp_params->m_alpha_settings.m_max_mode7_partitions_to_try);

		uniform color_cell_compressor_params params7 = *pParams;
		
		params7.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params7.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params7.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params7.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
		
		params7.m_pSelector_weights = g_bc7_weights2;
		params7.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		params7.m_num_selector_weights = 4;

		params7.m_comp_bits = 5;
		params7.m_has_pbits = true;
		params7.m_endpoints_share_pbit = false;
				
		params7.m_has_alpha = true;

		int selectors_temp[16];

		const uniform bool disable_faster_part_selection = false;

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions; solution_index++)
		{
			const uint32_t trial_partition = solutions[solution_index].m_index;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors7[2];
			subset_total_colors7[0] = 0;
			subset_total_colors7[1] = 0;
			 
			int subset_pixel_index7[2][16];
			int subset_selectors7[2][16];
			color_cell_compressor_results subset_results7[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index7[p][subset_total_colors7[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors7[p]++;
			}

			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results7[subset];

				pResults->m_pSelectors = &subset_selectors7[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], (num_solutions <= 2) || disable_faster_part_selection);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;
										
				opt_results.m_mode = 7;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors7[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index7[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
					}

					opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
				}
			}

		} // solution_index

		if ((num_solutions > 2) && (opt_results.m_mode == 7) && (!disable_faster_part_selection))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors7[2];
			subset_total_colors7[0] = 0;
			subset_total_colors7[1] = 0;
			 
			int subset_pixel_index7[2][16];
			int subset_selectors7[2][16];
			color_cell_compressor_results subset_results7[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index7[p][subset_total_colors7[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors7[p]++;
			}

			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results7[subset];

				pResults->m_pSelectors = &subset_selectors7[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;
										
				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors7[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index7[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
					}

					opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
				}
			}
		}
	}

	encode_bc7_block(pBlock, &opt_results);
}

static void handle_opaque_block(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams)
{
	int selectors_temp[16];
		
	bc7_optimization_results opt_results;
		
	uint64_t best_err = UINT64_MAX;

	// Mode 6
	if (pComp_params->m_opaque_settings.m_use_mode[6])
	{
		pParams->m_pSelector_weights = g_bc7_weights4;
		pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights4x[0];
		pParams->m_num_selector_weights = 16;

		pParams->m_comp_bits = 7;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		pParams->m_perceptual = pComp_params->m_perceptual;
				
		color_cell_compressor_results results6;						
		results6.m_pSelectors = opt_results.m_selectors;
		results6.m_pSelectors_temp = selectors_temp;

		best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, true);
						
		opt_results.m_mode = 6;
		opt_results.m_index_selector = 0;
		opt_results.m_rotation = 0;
		opt_results.m_partition = 0;

		opt_results.m_low[0] = results6.m_low_endpoint;
		opt_results.m_high[0] = results6.m_high_endpoint;

		opt_results.m_pbits[0][0] = results6.m_pbits[0];
		opt_results.m_pbits[0][1] = results6.m_pbits[1];
	}

	solution solutions2[BC7E_MAX_PARTITIONS3];
	uniform uint32_t num_solutions2 = 0;
	if (pComp_params->m_opaque_settings.m_use_mode[1] || pComp_params->m_opaque_settings.m_use_mode[3])
	{
		if (pComp_params->m_opaque_settings.m_max_mode13_partitions_to_try == 1)
		{
			solutions2[0].m_index = estimate_partition(1, pPixels, pComp_params);
			num_solutions2 = 1;
		}
		else
		{
			num_solutions2 = estimate_partition_list(1, pPixels, pComp_params, solutions2, pComp_params->m_opaque_settings.m_max_mode13_partitions_to_try);
		}
	}
		
	const uniform bool disable_faster_part_selection = false;
								
	// Mode 1
	if (pComp_params->m_opaque_settings.m_use_mode[1])
	{
		pParams->m_pSelector_weights = g_bc7_weights3;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
		pParams->m_num_selector_weights = 8;

		pParams->m_comp_bits = 6;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = true;

		pParams->m_perceptual = pComp_params->m_perceptual;

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
		{
			const uint32_t trial_partition = solutions2[solution_index].m_index;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];
						
			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors1[2];
			subset_total_colors1[0] = 0;
			subset_total_colors1[1] = 0;
				
			int subset_pixel_index1[2][16];
			int subset_selectors1[2][16];
			color_cell_compressor_results subset_results1[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index1[p][subset_total_colors1[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors1[p]++;
			}
								
			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results1[subset];

				pResults->m_pSelectors = &subset_selectors1[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], (num_solutions2 <= 2) || disable_faster_part_selection);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
					
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;

				opt_results.m_mode = 1;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors1[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index1[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
					}

					opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
				}
			}
		}

		if ((num_solutions2 > 2) && (opt_results.m_mode == 1) && (!disable_faster_part_selection))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];
						
			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors1[2];
			subset_total_colors1[0] = 0;
			subset_total_colors1[1] = 0;
				
			int subset_pixel_index1[2][16];
			int subset_selectors1[2][16];
			color_cell_compressor_results subset_results1[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index1[p][subset_total_colors1[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors1[p]++;
			}
								
			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results1[subset];

				pResults->m_pSelectors = &subset_selectors1[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
					
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors1[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index1[subset][i];
#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
					}

					opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
				}
			}
		}
	}
		
	// Mode 0
	if (pComp_params->m_opaque_settings.m_use_mode[0])
	{
		solution solutions3[BC7E_MAX_PARTITIONS0];
		uniform uint32_t num_solutions3 = 0;
		if (pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try == 1)
		{
			solutions3[0].m_index = estimate_partition(0, pPixels, pComp_params);
			num_solutions3 = 1;
		}
		else
		{
			num_solutions3 = estimate_partition_list(0, pPixels, pComp_params, solutions3, pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try);
		}

		pParams->m_pSelector_weights = g_bc7_weights3;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
		pParams->m_num_selector_weights = 8;

		pParams->m_comp_bits = 4;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		pParams->m_perceptual = pComp_params->m_perceptual;
				
		for (uniform uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
		{
			const uint32_t best_partition0 = solutions3[solution_index].m_index;

			const int *pPartition = &g_bc7_partition3[best_partition0 * 16];

			color_quad_i subset_colors[3][16];
						
			uint32_t subset_total_colors0[3];
			subset_total_colors0[0] = 0;
			subset_total_colors0[1] = 0;
			subset_total_colors0[2] = 0;

			int subset_pixel_index0[3][16];
						
			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors0[p]] = pPixels[idx];
#pragma ignore warning(perf)				
				subset_pixel_index0[p][subset_total_colors0[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors0[p]++;
			}
									
			color_cell_compressor_results subset_results0[3];
			int subset_selectors0[3][16];			

			uint64_t mode0_err = 0;
			for (uniform uint32_t subset = 0; subset < 3; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results0[subset];

				pResults->m_pSelectors = &subset_selectors0[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(0, pParams, pResults, pComp_params, subset_total_colors0[subset], &subset_colors[subset][0], true);
				assert(err == pResults->m_best_overall_err);

				mode0_err += err;
				if (mode0_err > best_err)
					break;
			} // subset

			if (mode0_err < best_err)
			{
				best_err = mode0_err;

				opt_results.m_mode = 0;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = best_partition0;

				for (uniform uint32_t subset = 0; subset < 3; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors0[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index0[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors0[subset][i];
					}

					opt_results.m_low[subset] = subset_results0[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results0[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results0[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results0[subset].m_pbits[1];
				}
			}
		}
	}
		
	// Mode 3
	if (pComp_params->m_opaque_settings.m_use_mode[3])
	{
		pParams->m_pSelector_weights = g_bc7_weights2;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		pParams->m_num_selector_weights = 4;

		pParams->m_comp_bits = 7;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		pParams->m_perceptual = pComp_params->m_perceptual;

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
		{
			const uint32_t trial_partition = solutions2[solution_index].m_index;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors3[2];
			subset_total_colors3[0] = 0;
			subset_total_colors3[1] = 0;
			 
			int subset_pixel_index3[2][16];
			int subset_selectors3[2][16];
			color_cell_compressor_results subset_results3[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index3[p][subset_total_colors3[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors3[p]++;
			}

			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results3[subset];

				pResults->m_pSelectors = &subset_selectors3[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], (num_solutions2 <= 2) || disable_faster_part_selection);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;
										
				opt_results.m_mode = 3;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors3[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index3[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
					}

					opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
				}
			}

		} // solution_index

		if ((num_solutions2 > 2) && (opt_results.m_mode == 3) && (!disable_faster_part_selection))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors3[2];
			subset_total_colors3[0] = 0;
			subset_total_colors3[1] = 0;
			 
			int subset_pixel_index3[2][16];
			int subset_selectors3[2][16];
			color_cell_compressor_results subset_results3[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index3[p][subset_total_colors3[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors3[p]++;
			}

			uint64_t trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results3[subset];

				pResults->m_pSelectors = &subset_selectors3[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;
										
				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors3[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index3[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
					}

					opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
				}
			}
		}
	}

	// Mode 5
	if ((!pComp_params->m_perceptual) && (pComp_params->m_opaque_settings.m_use_mode[5]))
	{
		uniform color_cell_compressor_params params5 = *pParams;

		for (uniform uint32_t rotation = 0; rotation < 4; rotation++)
		{
			if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0)
				continue;

			memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
			if (rotation)
				swapu(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = 255, trial_hi_a = 255;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results5;

			uint64_t trial_mode5_err = 0;

			handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);

			if (trial_mode5_err < best_err)
			{
				best_err = trial_mode5_err;

				opt_results = trial_opt_results5;
				opt_results.m_rotation = rotation;
			}
		} // rotation
	}

	// Mode 2
	if (pComp_params->m_opaque_settings.m_use_mode[2])
	{
		solution solutions3[BC7E_MAX_PARTITIONS2];
		uniform uint32_t num_solutions3 = 0;
		if (pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try == 1)
		{
			solutions3[0].m_index = estimate_partition(2, pPixels, pComp_params);
			num_solutions3 = 1;
		}
		else
		{
			num_solutions3 = estimate_partition_list(2, pPixels, pComp_params, solutions3, pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try);
		}

		pParams->m_pSelector_weights = g_bc7_weights2;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		pParams->m_num_selector_weights = 4;

		pParams->m_comp_bits = 5;
		pParams->m_has_pbits = false;
		pParams->m_endpoints_share_pbit = false;

		pParams->m_perceptual = pComp_params->m_perceptual;

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
		{
			const int32_t best_partition2 = solutions3[solution_index].m_index;
						
			uint32_t subset_total_colors2[3];
			subset_total_colors2[0] = 0;
			subset_total_colors2[1] = 0;
			subset_total_colors2[2] = 0;

			int subset_pixel_index2[3][16];
							
			const int *pPartition = &g_bc7_partition3[best_partition2 * 16];

			color_quad_i subset_colors[3][16];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors2[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index2[p][subset_total_colors2[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors2[p]++;
			}
			
			int subset_selectors2[3][16];
			color_cell_compressor_results subset_results2[3];
						
			uint64_t mode2_err = 0;
			for (uniform uint32_t subset = 0; subset < 3; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results2[subset];

				pResults->m_pSelectors = &subset_selectors2[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				uint64_t err = color_cell_compression(2, pParams, pResults, pComp_params, subset_total_colors2[subset], &subset_colors[subset][0], true);
				assert(err == pResults->m_best_overall_err);

				mode2_err += err;
				if (mode2_err > best_err)
					break;
			} // subset

			if (mode2_err < best_err)
			{
				best_err = mode2_err;

				opt_results.m_mode = 2;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = best_partition2;

				for (uniform uint32_t subset = 0; subset < 3; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors2[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index2[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors2[subset][i];
					}

					opt_results.m_low[subset] = subset_results2[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results2[subset].m_high_endpoint;
				}
			}
		}
	}

	// Mode 4
	if ((!pComp_params->m_perceptual) && (pComp_params->m_opaque_settings.m_use_mode[4]))
	{
		uniform color_cell_compressor_params params4 = *pParams;

		for (uniform uint32_t rotation = 0; rotation < 4; rotation++)
		{
			if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0)
				continue;

			memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
			if (rotation)
				swapu(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
							
			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = 255, trial_hi_a = 255;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results4;

			uint64_t trial_mode4_err = best_err;

			handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);

			if (trial_mode4_err < best_err)
			{
				best_err = trial_mode4_err;

				opt_results.m_mode = 4;
				opt_results.m_index_selector = trial_opt_results4.m_index_selector;
				opt_results.m_rotation = rotation;
				opt_results.m_partition = 0;

				opt_results.m_low[0] = trial_opt_results4.m_low[0];
				opt_results.m_high[0] = trial_opt_results4.m_high[0];

				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
				
				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
			}
		} // rotation
	}
	
	encode_bc7_block(pBlock, &opt_results);
}

// all solid color blocks can be 100% perfectly encoded with just mode 5
static void handle_block_solid(void *varying pBlock, uint32_t cr, uint32_t cg, uint32_t cb, uint32_t ca)
{
	#pragma ignore warning(perf)
	uint32_t er = g_bc7_mode_5_optimal_endpoints[cr];
	#pragma ignore warning(perf)
	uint32_t eg = g_bc7_mode_5_optimal_endpoints[cg];
	#pragma ignore warning(perf)
	uint32_t eb = g_bc7_mode_5_optimal_endpoints[cb];
	color_quad_i lp, hp;
	color_quad_i_set(&lp, er & 0xFF, eg & 0xFF, eb & 0xFF, ca);
	color_quad_i_set(&hp, er >> 8, eg >> 8, eb >> 8, ca);

	bc7_optimization_results opt;
	opt.m_mode = 5;
	opt.m_low[0] = lp;
	opt.m_high[0] = hp;
	opt.m_pbits[0][0] = 0;
	opt.m_pbits[0][1] = 0;
	opt.m_index_selector = 0;
	opt.m_rotation = 0;
	opt.m_partition = 0;
	for (uniform int i = 0; i < 16; ++i)
		opt.m_selectors[i] = BC7E_MODE_5_OPTIMAL_INDEX;
	for (uniform int i = 0; i < 16; ++i)
		opt.m_alpha_selectors[i] = 0;
	encode_bc7_block(pBlock, &opt);
}

static void handle_opaque_block_mode6(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams)
{
	int selectors_temp[16];
		
	bc7_optimization_results opt_results;
		
	uint64_t best_err = UINT64_MAX;

	// Mode 6
	pParams->m_pSelector_weights = g_bc7_weights4;
	pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights4x[0];
	pParams->m_num_selector_weights = 16;

	pParams->m_comp_bits = 7;
	pParams->m_has_pbits = true;
	pParams->m_endpoints_share_pbit = false;

	pParams->m_perceptual = pComp_params->m_perceptual;
				
	color_cell_compressor_results results6;						
	results6.m_pSelectors = opt_results.m_selectors;
	results6.m_pSelectors_temp = selectors_temp;

	best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, true);
						
	opt_results.m_mode = 6;
	opt_results.m_index_selector = 0;
	opt_results.m_rotation = 0;
	opt_results.m_partition = 0;

	opt_results.m_low[0] = results6.m_low_endpoint;
	opt_results.m_high[0] = results6.m_high_endpoint;

	opt_results.m_pbits[0][0] = results6.m_pbits[0];
	opt_results.m_pbits[0][1] = results6.m_pbits[1];
		
	encode_bc7_block_mode6(pBlock, &opt_results);
}

export void bc7e_compress_blocks(uniform uint32_t num_blocks, uniform uint64_t * uniform pBlocks, const uniform uint32_t * uniform pPixelsRGBA, const uniform bc7e_compress_block_params * uniform pComp_params)
{
	if (!g_codec_initialized)
	{
		// Caller has forgotten to initialize the codec, or another thread is still working on that. We can't continue.
		// What do we do here?
		assert(0);

		memset(pBlocks, 0, num_blocks * 16);
		return;
	}

	uniform color_cell_compressor_params params;
	color_cell_compressor_params_clear(&params);

	memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));

	assert(pComp_params->m_mode4_rotation_mask != 0);
	assert(pComp_params->m_mode4_index_mask != 0);
	assert(pComp_params->m_mode5_rotation_mask != 0);
	assert(pComp_params->m_uber1_mask != 0);
	
	foreach (block_index = 0 ... num_blocks)
	{
		const uniform color_quad_u8 *varying pSrcPixels = &((const uniform color_quad_u8 *)(pPixelsRGBA))[block_index * 16];

		varying color_quad_i temp_pixels[16];

		int lo_r = 255, hi_r = 0;
		int lo_g = 255, hi_g = 0;
		int lo_b = 255, hi_b = 0;
		float lo_a = 255, hi_a = 0;
		
		for (uniform uint32_t i = 0; i < 16; i++)
		{
#pragma ignore warning(perf)
			color_quad_u8 c = pSrcPixels[i];

			int r = c.m_c[0];
			int g = c.m_c[1];
			int b = c.m_c[2];
			int a = c.m_c[3];
												
			temp_pixels[i].m_c[0] = r;
			temp_pixels[i].m_c[1] = g;
			temp_pixels[i].m_c[2] = b;
			temp_pixels[i].m_c[3] = a;

			lo_r = min(lo_r, r); hi_r = max(hi_r, r);
			lo_g = min(lo_g, g); hi_g = max(hi_g, g);
			lo_b = min(lo_b, b); hi_b = max(hi_b, b);

			float fa = a;

			lo_a = min(lo_a, fa);
			hi_a = max(hi_a, fa);
		}

		bool all_same = lo_r==hi_r && lo_g==hi_g && lo_b==hi_b && lo_a==hi_a;

		uniform uint64_t *varying pBlock = &pBlocks[block_index * 2];
			
		cif (all_same)
			handle_block_solid(pBlock, lo_r, lo_g, lo_b, lo_a);
		else
		{
			const bool has_alpha = (lo_a < 255);
			// TODO: alpha block mode 6 only
			cif (has_alpha)
				handle_alpha_block(pBlock, temp_pixels, pComp_params, &params, (int)lo_a, (int)hi_a);
			else
			{
				if (pComp_params->m_mode6_only)
					handle_opaque_block_mode6(pBlock, temp_pixels, pComp_params, &params);
				else
					handle_opaque_block(pBlock, temp_pixels, pComp_params, &params);
			}
		}
	}
}

export void bc7e_compress_block_params_init(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	p->m_max_partitions_mode[0] = BC7E_MAX_PARTITIONS0;
	p->m_max_partitions_mode[1] = BC7E_MAX_PARTITIONS1;
	p->m_max_partitions_mode[2] = BC7E_MAX_PARTITIONS2;
	p->m_max_partitions_mode[3] = BC7E_MAX_PARTITIONS3;
	p->m_max_partitions_mode[4] = 0;
	p->m_max_partitions_mode[5] = 0;
	p->m_max_partitions_mode[6] = 0;
	p->m_max_partitions_mode[7] = BC7E_MAX_PARTITIONS7;

	p->m_perceptual = perceptual;
	if (perceptual)
	{
		p->m_weights[0] = 128;
		p->m_weights[1] = 64;
		p->m_weights[2] = 16;
		p->m_weights[3] = 256;
	}
	else
	{
		p->m_weights[0] = 1;
		p->m_weights[1] = 1;
		p->m_weights[2] = 1;
		p->m_weights[3] = 1;
	}

	p->m_pbit_search = false;
	p->m_mode6_only = false;
	p->m_refinement_passes = 1;
	p->m_mode4_rotation_mask = 0xF;
	p->m_mode4_index_mask = 3;
	p->m_mode5_rotation_mask = 0xF;
	p->m_uber1_mask = 7;
	for (uniform uint32_t i = 0; i < 7; i++)
		p->m_opaque_settings.m_use_mode[i] = true;
	p->m_opaque_settings.m_max_mode13_partitions_to_try = 1;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 1;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 1;
	p->m_alpha_settings.m_use_mode4 = true;
	p->m_alpha_settings.m_use_mode5 = true;
	p->m_alpha_settings.m_use_mode6 = true;
	p->m_alpha_settings.m_use_mode7 = true;
	p->m_alpha_settings.m_use_mode4_rotation = true;
	p->m_alpha_settings.m_use_mode5_rotation = true;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 1;
	p->m_alpha_settings.m_mode67_error_weight_mul[0] = 1;
	p->m_alpha_settings.m_mode67_error_weight_mul[1] = 1;
	p->m_alpha_settings.m_mode67_error_weight_mul[2] = 1;
	p->m_alpha_settings.m_mode67_error_weight_mul[3] = 1;
	p->m_uber_level = 0;
}

export void bc7e_compress_block_params_init_slowest(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_opaque_settings.m_max_mode13_partitions_to_try = 4;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 4;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 4;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 4;

	p->m_pbit_search = true;
	p->m_uber_level = 4;
}

export void bc7e_compress_block_params_init_veryslow(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 2;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 2;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

	p->m_pbit_search = true;
	p->m_uber_level = 2;
}

export void bc7e_compress_block_params_init_slow(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

	p->m_pbit_search = true;
	p->m_uber_level = 0;
}

export void bc7e_compress_block_params_init_basic(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 1;
	}
	else
	{
		p->m_max_partitions_mode[1] = 32;
		p->m_max_partitions_mode[2] = 32;
		p->m_max_partitions_mode[3] = 32;
		p->m_max_partitions_mode[7] = 32;

		p->m_opaque_settings.m_use_mode[2] = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 1;
	}
}

export void bc7e_compress_block_params_init_fast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
	
		p->m_opaque_settings.m_max_mode13_partitions_to_try = 1;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}		
	else
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
	
		p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
}

export void bc7e_compress_block_params_init_veryfast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
	else
	{
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
}

export void bc7e_compress_block_params_init_ultrafast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_mode6_only = true;

	p->m_alpha_settings.m_use_mode4 = true;
	p->m_alpha_settings.m_use_mode5 = true;
	p->m_alpha_settings.m_use_mode7 = false;

	p->m_mode4_rotation_mask = 1+4;
	p->m_mode4_index_mask = 3;
	p->m_mode5_rotation_mask = 1;
				
	p->m_pbit_search = false;
	p->m_uber_level = 0;
}
