// File: bc7enc.c - Richard Geldreich, Jr. 3/31/2020 - MIT license or public domain (see end of file)
// Currently supports modes 1, 6 for RGB blocks, and modes 5, 6, 7 for RGBA blocks.
#include "bc7enc.h"
#include <math.h>
#include <memory.h>
#include <assert.h>
#include <limits.h>
#include <algorithm>

// Helpers
static inline int32_t clampi(int32_t value, int32_t low, int32_t high) { if (value < low) value = low; else if (value > high) value = high;	return value; }
static inline float clampf(float value, float low, float high) { if (value < low) value = low; else if (value > high) value = high;	return value; }
static inline float saturate(float value) { return clampf(value, 0, 1.0f); }
static inline uint8_t minimumub(uint8_t a, uint8_t b) { return (a < b) ? a : b; }
static inline int32_t minimumi(int32_t a, int32_t b) { return (a < b) ? a : b; }
static inline uint32_t minimumu(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline float minimumf(float a, float b) { return (a < b) ? a : b; }
static inline uint8_t maximumub(uint8_t a, uint8_t b) { return (a > b) ? a : b; }
static inline uint32_t maximumu(uint32_t a, uint32_t b) { return (a > b) ? a : b; }
static inline int32_t maximumi(int32_t a, int32_t b) { return (a > b) ? a : b; }
static inline float maximumf(float a, float b) { return (a > b) ? a : b; }
static inline int squarei(int i) { return i * i; }
static inline float squaref(float i) { return i * i; }
template <typename T0, typename T1> inline T0 lerp(T0 a, T0 b, T1 c) { return a + (b - a) * c; }

static inline int32_t iabs32(int32_t v) { uint32_t msk = v >> 31; return (v ^ msk) - msk; }
static inline void swapub(uint8_t* a, uint8_t* b) { uint8_t t = *a; *a = *b; *b = t; }
static inline void swapu(uint32_t* a, uint32_t* b) { uint32_t t = *a; *a = *b; *b = t; }
static inline void swapf(float* a, float* b) { float t = *a; *a = *b; *b = t; }

struct vec4F { float m_c[4]; };

static inline color_rgba *color_quad_u8_set_clamped(color_rgba *pRes, int32_t r, int32_t g, int32_t b, int32_t a) { pRes->m_c[0] = (uint8_t)clampi(r, 0, 255); pRes->m_c[1] = (uint8_t)clampi(g, 0, 255); pRes->m_c[2] = (uint8_t)clampi(b, 0, 255); pRes->m_c[3] = (uint8_t)clampi(a, 0, 255); return pRes; }
static inline color_rgba *color_quad_u8_set(color_rgba *pRes, int32_t r, int32_t g, int32_t b, int32_t a) { assert((uint32_t)(r | g | b | a) <= 255); pRes->m_c[0] = (uint8_t)r; pRes->m_c[1] = (uint8_t)g; pRes->m_c[2] = (uint8_t)b; pRes->m_c[3] = (uint8_t)a; return pRes; }
static inline bool color_quad_u8_notequals(const color_rgba *pLHS, const color_rgba *pRHS) { return (pLHS->m_c[0] != pRHS->m_c[0]) || (pLHS->m_c[1] != pRHS->m_c[1]) || (pLHS->m_c[2] != pRHS->m_c[2]) || (pLHS->m_c[3] != pRHS->m_c[3]); }
static inline vec4F *vec4F_set_scalar(vec4F *pV, float x) {	pV->m_c[0] = x; pV->m_c[1] = x; pV->m_c[2] = x;	pV->m_c[3] = x;	return pV; }
static inline vec4F *vec4F_set(vec4F *pV, float x, float y, float z, float w) {	pV->m_c[0] = x;	pV->m_c[1] = y;	pV->m_c[2] = z;	pV->m_c[3] = w;	return pV; }
static inline vec4F *vec4F_saturate_in_place(vec4F *pV) { pV->m_c[0] = saturate(pV->m_c[0]); pV->m_c[1] = saturate(pV->m_c[1]); pV->m_c[2] = saturate(pV->m_c[2]); pV->m_c[3] = saturate(pV->m_c[3]); return pV; }
static inline vec4F vec4F_saturate(const vec4F *pV) { vec4F res; res.m_c[0] = saturate(pV->m_c[0]); res.m_c[1] = saturate(pV->m_c[1]); res.m_c[2] = saturate(pV->m_c[2]); res.m_c[3] = saturate(pV->m_c[3]); return res; }
static inline vec4F vec4F_from_color(const color_rgba *pC) { vec4F res; vec4F_set(&res, pC->m_c[0], pC->m_c[1], pC->m_c[2], pC->m_c[3]); return res; }
static inline vec4F vec4F_add(const vec4F *pLHS, const vec4F *pRHS) { vec4F res; vec4F_set(&res, pLHS->m_c[0] + pRHS->m_c[0], pLHS->m_c[1] + pRHS->m_c[1], pLHS->m_c[2] + pRHS->m_c[2], pLHS->m_c[3] + pRHS->m_c[3]); return res; }
static inline vec4F vec4F_sub(const vec4F *pLHS, const vec4F *pRHS) { vec4F res; vec4F_set(&res, pLHS->m_c[0] - pRHS->m_c[0], pLHS->m_c[1] - pRHS->m_c[1], pLHS->m_c[2] - pRHS->m_c[2], pLHS->m_c[3] - pRHS->m_c[3]); return res; }
static inline float vec4F_dot(const vec4F *pLHS, const vec4F *pRHS) { return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2] + pLHS->m_c[3] * pRHS->m_c[3]; }
static inline vec4F vec4F_mul(const vec4F *pLHS, float s) { vec4F res; vec4F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s, pLHS->m_c[3] * s); return res; }
static inline vec4F *vec4F_normalize_in_place(vec4F *pV) { float s = pV->m_c[0] * pV->m_c[0] + pV->m_c[1] * pV->m_c[1] + pV->m_c[2] * pV->m_c[2] + pV->m_c[3] * pV->m_c[3]; if (s != 0.0f) { s = 1.0f / sqrtf(s); pV->m_c[0] *= s; pV->m_c[1] *= s; pV->m_c[2] *= s; pV->m_c[3] *= s; } return pV; }

// Various BC7 tables
static const uint32_t g_bc7_weights2[4] = { 0, 21, 43, 64 };
static const uint32_t g_bc7_weights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uint32_t g_bc7_weights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
// Precomputed weight constants used during least fit determination. For each entry in g_bc7_weights[]: w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w
static const float g_bc7_weights2x[4 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.107666f, 0.220459f, 0.451416f, 0.328125f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const float g_bc7_weights3x[8 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.079102f, 0.202148f, 0.516602f, 0.281250f, 0.177979f, 0.243896f, 0.334229f, 0.421875f, 0.334229f, 0.243896f, 0.177979f, 0.578125f, 0.516602f, 0.202148f,
	0.079102f, 0.718750f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const float g_bc7_weights4x[16 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.003906f, 0.058594f, 0.878906f, 0.062500f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.041260f, 0.161865f, 0.635010f, 0.203125f, 0.070557f, 0.195068f, 0.539307f, 0.265625f, 0.107666f, 0.220459f,
	0.451416f, 0.328125f, 0.165039f, 0.241211f, 0.352539f, 0.406250f, 0.219727f, 0.249023f, 0.282227f, 0.468750f, 0.282227f, 0.249023f, 0.219727f, 0.531250f, 0.352539f, 0.241211f, 0.165039f, 0.593750f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 0.539307f, 0.195068f, 0.070557f, 0.734375f,
	0.635010f, 0.161865f, 0.041260f, 0.796875f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 0.878906f, 0.058594f, 0.003906f, 0.937500f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };

static const uint8_t g_bc7_partition1[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
static const uint8_t g_bc7_partition2[64 * 16] =
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

static const uint8_t g_bc7_partition3[64 * 16] =
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

static const uint8_t g_bc7_table_anchor_index_third_subset_1[64] =
{
	3, 3,15,15, 8, 3,15,15,		8, 8, 6, 6, 6, 5, 3, 3,		3, 3, 8,15, 3, 3, 6,10,		5, 8, 8, 6, 8, 5,15,15,		8,15, 3, 5, 6,10, 8,15,		15, 3,15, 5,15,15,15,15,		3,15, 5, 5, 5, 8, 5,10,		5,10, 8,13,15,12, 3, 3
};

static const uint8_t g_bc7_table_anchor_index_third_subset_2[64] =
{
	15, 8, 8, 3,15,15, 3, 8,		15,15,15,15,15,15,15, 8,		15, 8,15, 3,15, 8,15, 8,		3,15, 6,10,15,15,10, 8,		15, 3,15,10,10, 8, 9,10,		6,15, 8,15, 3, 6, 6, 8,		15, 3,15,15,15,15,15,15,		15,15,15,15, 3,15,15, 8
};

static const uint8_t g_bc7_table_anchor_index_second_subset[64] = {	15,15,15,15,15,15,15,15,		15,15,15,15,15,15,15,15,		15, 2, 8, 2, 2, 8, 8,15,		2, 8, 2, 2, 8, 8, 2, 2,		15,15, 6, 8, 2, 8,15,15,		2, 8, 2, 2, 2,15,15, 6,		6, 2, 6, 8,15,15, 2, 2,		15,15,15,15,15, 2, 2,15 };
static const uint8_t g_bc7_num_subsets[8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
static const uint8_t g_bc7_partition_bits[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
static const uint8_t g_bc7_color_index_bitcount[8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
static int get_bc7_color_index_size(int mode, int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
static uint8_t g_bc7_alpha_index_bitcount[8] = { 0, 0, 0, 0, 3, 2, 4, 2 };
static int get_bc7_alpha_index_size(int mode, int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
static const uint8_t g_bc7_mode_has_p_bits[8] = { 1, 1, 0, 1, 0, 0, 1, 1 };
static const uint8_t g_bc7_mode_has_shared_p_bits[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
static const uint8_t g_bc7_color_precision_table[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const int8_t g_bc7_alpha_precision_table[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
static bool get_bc7_mode_has_seperate_alpha_selectors(int mode) { return (mode == 4) || (mode == 5); }

typedef struct { uint16_t m_error; uint8_t m_lo; uint8_t m_hi; } endpoint_err;

static endpoint_err g_bc7_mode_1_optimal_endpoints[256][2]; // [c][pbit]
static const uint32_t BC7ENC_MODE_1_OPTIMAL_INDEX = 2;

static endpoint_err g_bc7_mode_7_optimal_endpoints[256][2][2]; // [c][pbit][hp][lp]
const uint32_t BC7E_MODE_7_OPTIMAL_INDEX = 1;

static float g_mode1_rgba_midpoints[64][2];
static float g_mode5_rgba_midpoints[128];
static float g_mode7_rgba_midpoints[32][2];

static uint8_t g_mode6_reduced_quant[2048][2];

static bool g_initialized;

// Initialize the lookup table used for optimal single color compression in mode 1/7. Must be called before encoding.
void bc7enc_compress_block_init()
{
	if (g_initialized)
		return;

	// Mode 7 endpoint midpoints
	for (uint32_t p = 0; p < 2; p++)
	{
		for (uint32_t i = 0; i < 32; i++)
		{
			uint32_t vl = ((i << 1) | p) << 2;
			vl |= (vl >> 6);
			float lo = vl / 255.0f;

			uint32_t vh = ((minimumi(31, (i + 1)) << 1) | p) << 2;
			vh |= (vh >> 6);
			float hi = vh / 255.0f;

			//g_mode7_quant_values[i][p] = lo;
			if (i == 31)
				g_mode7_rgba_midpoints[i][p] = 1.0f;
			else
				g_mode7_rgba_midpoints[i][p] = (lo + hi) / 2.0f;
		}
	}

	// Mode 1 endpoint midpoints
	for (uint32_t p = 0; p < 2; p++)
	{
		for (uint32_t i = 0; i < 64; i++)
		{
			uint32_t vl = ((i << 1) | p) << 1;
			vl |= (vl >> 7);
			float lo = vl / 255.0f;

			uint32_t vh = ((minimumi(63, (i + 1)) << 1) | p) << 1;
			vh |= (vh >> 7);
			float hi = vh / 255.0f;

			//g_mode1_quant_values[i][p] = lo;
			if (i == 63)
				g_mode1_rgba_midpoints[i][p] = 1.0f;
			else
				g_mode1_rgba_midpoints[i][p] = (lo + hi) / 2.0f;
		}
	}

	// Mode 5 endpoint midpoints
	for (uint32_t i = 0; i < 128; i++)
	{
		uint32_t vl = (i << 1);
		vl |= (vl >> 7);
		float lo = vl / 255.0f;

		uint32_t vh = minimumi(127, i + 1) << 1;
		vh |= (vh >> 7);
		float hi = vh / 255.0f;

		if (i == 127)
			g_mode5_rgba_midpoints[i] = 1.0f;
		else
			g_mode5_rgba_midpoints[i] = (lo + hi) / 2.0f;
	}

	for (uint32_t p = 0; p < 2; p++)
	{
		for (uint32_t i = 0; i < 2048; i++)
		{
			float f = i / 2047.0f;

			float best_err = 1e+9f;
			int best_index = 0;
			for (int j = 0; j < 64; j++)
			{
				int ik = (j * 127 + 31) / 63;
				float k = ((ik << 1) + p) / 255.0f;

				float e = fabsf(k - f);
				if (e < best_err)
				{
					best_err = e;
					best_index = ik;
				}
			}

			g_mode6_reduced_quant[i][p] = (uint8_t)best_index;
		}
	} // p

	// Mode 1
	for (int c = 0; c < 256; c++)
	{
		for (uint32_t lp = 0; lp < 2; lp++)
		{
			endpoint_err best;
			best.m_error = (uint16_t)UINT16_MAX;
			for (uint32_t l = 0; l < 64; l++)
			{
				uint32_t low = ((l << 1) | lp) << 1;
				low |= (low >> 7);
				for (uint32_t h = 0; h < 64; h++)
				{
					uint32_t high = ((h << 1) | lp) << 1;
					high |= (high >> 7);
					const int k = (low * (64 - g_bc7_weights3[BC7ENC_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7ENC_MODE_1_OPTIMAL_INDEX] + 32) >> 6;
					const int err = (k - c) * (k - c);
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

	// Mode 7: 555.1 2-bit indices 
	for (int c = 0; c < 256; c++)
	{
		for (uint32_t hp = 0; hp < 2; hp++)
		{
			for (uint32_t lp = 0; lp < 2; lp++)
			{
				endpoint_err best;
				best.m_error = (uint16_t)UINT16_MAX;
				best.m_lo = 0;
				best.m_hi = 0;

				for (uint32_t l = 0; l < 32; l++)
				{
					uint32_t low = ((l << 1) | lp) << 2;
					low |= (low >> 6);

					for (uint32_t h = 0; h < 32; h++)
					{
						uint32_t high = ((h << 1) | hp) << 2;
						high |= (high >> 6);

						const int k = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;

						const int err = (k - c) * (k - c);
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

	g_initialized = true;
}

static void compute_least_squares_endpoints_rgba(uint32_t N, const uint8_t *pSelectors, const vec4F *pSelector_weights, vec4F *pXl, vec4F *pXh, const color_rgba *pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];
		z00 += pSelector_weights[sel].m_c[0];
		z10 += pSelector_weights[sel].m_c[1];
		z11 += pSelector_weights[sel].m_c[2];
		float w = pSelector_weights[sel].m_c[3];
		q00_r += w * pColors[i].m_c[0]; t_r += pColors[i].m_c[0];
		q00_g += w * pColors[i].m_c[1]; t_g += pColors[i].m_c[1];
		q00_b += w * pColors[i].m_c[2]; t_b += pColors[i].m_c[2];
		q00_a += w * pColors[i].m_c[3]; t_a += pColors[i].m_c[3];
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

	for (uint32_t c = 0; c < 4; c++)
	{
		if ((pXl->m_c[c] < 0.0f) || (pXh->m_c[c] > 255.0f))
		{
			uint32_t lo_v = UINT32_MAX, hi_v = 0;
			for (uint32_t i = 0; i < N; i++)
			{
				lo_v = minimumu(lo_v, pColors[i].m_c[c]);
				hi_v = maximumu(hi_v, pColors[i].m_c[c]);
			}

			if (lo_v == hi_v)
			{
				pXl->m_c[c] = (float)lo_v;
				pXh->m_c[c] = (float)hi_v;
			}
		}
	}
}

static void compute_least_squares_endpoints_rgb(uint32_t N, const uint8_t *pSelectors, const vec4F *pSelector_weights, vec4F *pXl, vec4F *pXh, const color_rgba*pColors)
{
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];
		z00 += pSelector_weights[sel].m_c[0];
		z10 += pSelector_weights[sel].m_c[1];
		z11 += pSelector_weights[sel].m_c[2];
		float w = pSelector_weights[sel].m_c[3];
		q00_r += w * pColors[i].m_c[0]; t_r += pColors[i].m_c[0];
		q00_g += w * pColors[i].m_c[1]; t_g += pColors[i].m_c[1];
		q00_b += w * pColors[i].m_c[2]; t_b += pColors[i].m_c[2];
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
	pXl->m_c[3] = 255.0f; pXh->m_c[3] = 255.0f;

	for (uint32_t c = 0; c < 3; c++)
	{
		if ((pXl->m_c[c] < 0.0f) || (pXh->m_c[c] > 255.0f))
		{
			uint32_t lo_v = UINT32_MAX, hi_v = 0;
			for (uint32_t i = 0; i < N; i++)
			{
				lo_v = minimumu(lo_v, pColors[i].m_c[c]);
				hi_v = maximumu(hi_v, pColors[i].m_c[c]);
			}

			if (lo_v == hi_v)
			{
				pXl->m_c[c] = (float)lo_v;
				pXh->m_c[c] = (float)hi_v;
			}
		}
	}
}

static void compute_least_squares_endpoints_a(uint32_t N, const uint8_t* pSelectors, const vec4F* pSelector_weights, float* pXl, float* pXh, const color_rgba *pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

		z00 += pSelector_weights[sel].m_c[0];
		z10 += pSelector_weights[sel].m_c[1];
		z11 += pSelector_weights[sel].m_c[2];

		float w = pSelector_weights[sel].m_c[3];

		q00_a += w * pColors[i].m_c[3]; t_a += pColors[i].m_c[3];
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

	if ((*pXl < 0.0f) || (*pXh > 255.0f))
	{
		uint32_t lo_v = UINT32_MAX, hi_v = 0;
		for (uint32_t i = 0; i < N; i++)
		{
			lo_v = minimumu(lo_v, pColors[i].m_c[3]);
			hi_v = maximumu(hi_v, pColors[i].m_c[3]);
		}

		if (lo_v == hi_v)
		{
			*pXl = (float)lo_v;
			*pXh = (float)hi_v;
		}
	}
}

struct color_cell_compressor_params
{
	uint32_t m_num_pixels;
	const color_rgba *m_pPixels;
	uint32_t m_num_selector_weights;
	const uint32_t *m_pSelector_weights;
	const vec4F *m_pSelector_weightsx;
	uint32_t m_comp_bits;
	uint32_t m_weights[4];
	bool m_has_alpha;
	bool m_has_pbits;
	bool m_endpoints_share_pbit;
	bool m_perceptual;
};

struct color_cell_compressor_results
{
	uint64_t m_best_overall_err;
	color_rgba m_low_endpoint;
	color_rgba m_high_endpoint;
	uint32_t m_pbits[2];
	uint8_t *m_pSelectors;
	uint8_t *m_pSelectors_temp;
};

static inline color_rgba scale_color(const color_rgba *pC, const color_cell_compressor_params *pParams)
{
	color_rgba results;

	const uint32_t n = pParams->m_comp_bits + (pParams->m_has_pbits ? 1 : 0);
	assert((n >= 4) && (n <= 8));

	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t v = pC->m_c[i] << (8 - n);
		v |= (v >> n);
		assert(v <= 255);
		results.m_c[i] = (uint8_t)(v);
	}

	return results;
}

static inline uint64_t compute_color_distance_rgb(const color_rgba *pE1, const color_rgba *pE2, bool perceptual, const uint32_t weights[4])
{
	int dr, dg, db;

	if (perceptual)
	{
		const int l1 = pE1->m_c[0] * 109 + pE1->m_c[1] * 366 + pE1->m_c[2] * 37;
		const int cr1 = ((int)pE1->m_c[0] << 9) - l1;
		const int cb1 = ((int)pE1->m_c[2] << 9) - l1;
		const int l2 = pE2->m_c[0] * 109 + pE2->m_c[1] * 366 + pE2->m_c[2] * 37;
		const int cr2 = ((int)pE2->m_c[0] << 9) - l2;
		const int cb2 = ((int)pE2->m_c[2] << 9) - l2;
		dr = (l1 - l2) >> 8;
		dg = (cr1 - cr2) >> 8;
		db = (cb1 - cb2) >> 8;
	}
	else
	{
		dr = (int)pE1->m_c[0] - (int)pE2->m_c[0];
		dg = (int)pE1->m_c[1] - (int)pE2->m_c[1];
		db = (int)pE1->m_c[2] - (int)pE2->m_c[2];
	}

	return weights[0] * (uint32_t)(dr * dr) + weights[1] * (uint32_t)(dg * dg) + weights[2] * (uint32_t)(db * db);
}

static inline uint64_t compute_color_distance_rgba(const color_rgba *pE1, const color_rgba *pE2, bool perceptual, const uint32_t weights[4])
{
	int da = (int)pE1->m_c[3] - (int)pE2->m_c[3];
	return compute_color_distance_rgb(pE1, pE2, perceptual, weights) + (weights[3] * (uint32_t)(da * da));
}

static uint64_t pack_mode1_to_one_color(const color_cell_compressor_params *pParams, color_cell_compressor_results *pResults, uint32_t r, uint32_t g, uint32_t b, uint8_t *pSelectors)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uint32_t p = 0; p < 2; p++)
	{
		uint32_t err = g_bc7_mode_1_optimal_endpoints[r][p].m_error + g_bc7_mode_1_optimal_endpoints[g][p].m_error + g_bc7_mode_1_optimal_endpoints[b][p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
			if (!best_err)
				break;
		}
	}

	const endpoint_err *pEr = &g_bc7_mode_1_optimal_endpoints[r][best_p];
	const endpoint_err *pEg = &g_bc7_mode_1_optimal_endpoints[g][best_p];
	const endpoint_err *pEb = &g_bc7_mode_1_optimal_endpoints[b][best_p];

	color_quad_u8_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);
	color_quad_u8_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);
	pResults->m_pbits[0] = best_p;
	pResults->m_pbits[1] = 0;

	memset(pSelectors, BC7ENC_MODE_1_OPTIMAL_INDEX, pParams->m_num_pixels);

	color_rgba p;
	for (uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		low |= (low >> 7);

		uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		high |= (high >> 7);

		p.m_c[i] = (uint8_t)((low * (64 - g_bc7_weights3[BC7ENC_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7ENC_MODE_1_OPTIMAL_INDEX] + 32) >> 6);
	}
	p.m_c[3] = 255;

	uint64_t total_err = 0;
	for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		total_err += compute_color_distance_rgb(&p, &pParams->m_pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t pack_mode7_to_one_color(const color_cell_compressor_params* pParams, color_cell_compressor_results* pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
	uint8_t* pSelectors, uint32_t num_pixels, const color_rgba *pPixels)
{
	uint32_t best_err = UINT_MAX;
	uint32_t best_p = 0;

	for (uint32_t p = 0; p < 4; p++)
	{
		uint32_t hi_p = p >> 1;
		uint32_t lo_p = p & 1;
		uint32_t err = g_bc7_mode_7_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[a][hi_p][lo_p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
			if (!best_err)
				break;
		}
	}

	uint32_t best_hi_p = best_p >> 1;
	uint32_t best_lo_p = best_p & 1;

	const endpoint_err* pEr = &g_bc7_mode_7_optimal_endpoints[r][best_hi_p][best_lo_p];
	const endpoint_err* pEg = &g_bc7_mode_7_optimal_endpoints[g][best_hi_p][best_lo_p];
	const endpoint_err* pEb = &g_bc7_mode_7_optimal_endpoints[b][best_hi_p][best_lo_p];
	const endpoint_err* pEa = &g_bc7_mode_7_optimal_endpoints[a][best_hi_p][best_lo_p];

	color_quad_u8_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);
	color_quad_u8_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);
	pResults->m_pbits[0] = best_lo_p;
	pResults->m_pbits[1] = best_hi_p;

	for (uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = (uint8_t)BC7E_MODE_7_OPTIMAL_INDEX;

	color_rgba p;

	for (uint32_t i = 0; i < 4; i++)
	{
		uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
		uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];

		low = (low << 2) | (low >> 6);
		high = (high << 2) | (high >> 6);

		p.m_c[i] = (uint8_t)((low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6);
	}

	uint64_t total_err = 0;
	for (uint32_t i = 0; i < num_pixels; i++)
		total_err += compute_color_distance_rgba(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static uint64_t evaluate_solution(const color_rgba *pLow, const color_rgba *pHigh, const uint32_t pbits[2], const color_cell_compressor_params *pParams, color_cell_compressor_results *pResults,
	const bc7enc_compress_block_params* pComp_params)
{
	color_rgba quantMinColor = *pLow;
	color_rgba quantMaxColor = *pHigh;

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

		quantMinColor.m_c[0] = (uint8_t)((pLow->m_c[0] << 1) | minPBit);
		quantMinColor.m_c[1] = (uint8_t)((pLow->m_c[1] << 1) | minPBit);
		quantMinColor.m_c[2] = (uint8_t)((pLow->m_c[2] << 1) | minPBit);
		quantMinColor.m_c[3] = (uint8_t)((pLow->m_c[3] << 1) | minPBit);

		quantMaxColor.m_c[0] = (uint8_t)((pHigh->m_c[0] << 1) | maxPBit);
		quantMaxColor.m_c[1] = (uint8_t)((pHigh->m_c[1] << 1) | maxPBit);
		quantMaxColor.m_c[2] = (uint8_t)((pHigh->m_c[2] << 1) | maxPBit);
		quantMaxColor.m_c[3] = (uint8_t)((pHigh->m_c[3] << 1) | maxPBit);
	}

	color_rgba actualMinColor = scale_color(&quantMinColor, pParams);
	color_rgba actualMaxColor = scale_color(&quantMaxColor, pParams);

	const uint32_t N = pParams->m_num_selector_weights;

	color_rgba weightedColors[16];
	weightedColors[0] = actualMinColor;
	weightedColors[N - 1] = actualMaxColor;

	const uint32_t nc = pParams->m_has_alpha ? 4 : 3;
	for (uint32_t i = 1; i < (N - 1); i++)
		for (uint32_t j = 0; j < nc; j++)
			weightedColors[i].m_c[j] = (uint8_t)((actualMinColor.m_c[j] * (64 - pParams->m_pSelector_weights[i]) + actualMaxColor.m_c[j] * pParams->m_pSelector_weights[i] + 32) >> 6);

	const int lr = actualMinColor.m_c[0];
	const int lg = actualMinColor.m_c[1];
	const int lb = actualMinColor.m_c[2];
	const int dr = actualMaxColor.m_c[0] - lr;
	const int dg = actualMaxColor.m_c[1] - lg;
	const int db = actualMaxColor.m_c[2] - lb;
	
	uint64_t total_err = 0;

	if (pComp_params->m_force_selectors)
	{
		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			const uint32_t best_sel = pComp_params->m_selectors[i];

			uint64_t best_err;
			if (pParams->m_has_alpha)
				best_err = compute_color_distance_rgba(&weightedColors[best_sel], &pParams->m_pPixels[i], pParams->m_perceptual, pParams->m_weights);
			else
				best_err = compute_color_distance_rgb(&weightedColors[best_sel], &pParams->m_pPixels[i], pParams->m_perceptual, pParams->m_weights);

			total_err += best_err;

			pResults->m_pSelectors_temp[i] = (uint8_t)best_sel;
		}
	}
	else if (!pParams->m_perceptual)
	{
		if (pParams->m_has_alpha)
		{
			const int la = actualMinColor.m_c[3];
			const int da = actualMaxColor.m_c[3] - la;

			const float f = N / (float)(squarei(dr) + squarei(dg) + squarei(db) + squarei(da) + .00000125f);

			for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
			{
				const color_rgba *pC = &pParams->m_pPixels[i];
				int r = pC->m_c[0];
				int g = pC->m_c[1];
				int b = pC->m_c[2];
				int a = pC->m_c[3];

				int best_sel = (int)((float)((r - lr) * dr + (g - lg) * dg + (b - lb) * db + (a - la) * da) * f + .5f);
				best_sel = clampi(best_sel, 1, N - 1);

				uint64_t err0 = compute_color_distance_rgba(&weightedColors[best_sel - 1], pC, false, pParams->m_weights);
				uint64_t err1 = compute_color_distance_rgba(&weightedColors[best_sel], pC, false, pParams->m_weights);

				if (err1 > err0)
				{
					err1 = err0;
					--best_sel;
				}
				total_err += err1;

				pResults->m_pSelectors_temp[i] = (uint8_t)best_sel;
			}
		}
		else
		{
			const float f = N / (float)(squarei(dr) + squarei(dg) + squarei(db) + .00000125f);

			for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
			{
				const color_rgba *pC = &pParams->m_pPixels[i];
				int r = pC->m_c[0];
				int g = pC->m_c[1];
				int b = pC->m_c[2];

				int sel = (int)((float)((r - lr) * dr + (g - lg) * dg + (b - lb) * db) * f + .5f);
				sel = clampi(sel, 1, N - 1);

				uint64_t err0 = compute_color_distance_rgb(&weightedColors[sel - 1], pC, false, pParams->m_weights);
				uint64_t err1 = compute_color_distance_rgb(&weightedColors[sel], pC, false, pParams->m_weights);

				int best_sel = sel;
				uint64_t best_err = err1;
				if (err0 < best_err)
				{
					best_err = err0;
					best_sel = sel - 1;
				}

				total_err += best_err;

				pResults->m_pSelectors_temp[i] = (uint8_t)best_sel;
			}
		}
	}
	else
	{
		// TODO: This could be improved.
		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			uint64_t best_err = UINT64_MAX;
			uint32_t best_sel = 0;

			if (pParams->m_has_alpha)
			{
				for (uint32_t j = 0; j < N; j++)
				{
					uint64_t err = compute_color_distance_rgba(&weightedColors[j], &pParams->m_pPixels[i], true, pParams->m_weights);
					if (err < best_err)
					{
						best_err = err;
						best_sel = j;
					}
				}
			}
			else
			{
				for (uint32_t j = 0; j < N; j++)
				{
					uint64_t err = compute_color_distance_rgb(&weightedColors[j], &pParams->m_pPixels[i], true, pParams->m_weights);
					if (err < best_err)
					{
						best_err = err;
						best_sel = j;
					}
				}
			}

			total_err += best_err;

			pResults->m_pSelectors_temp[i] = (uint8_t)best_sel;
		}
	}

	if (total_err < pResults->m_best_overall_err)
	{
		pResults->m_best_overall_err = total_err;

		pResults->m_low_endpoint = *pLow;
		pResults->m_high_endpoint = *pHigh;

		pResults->m_pbits[0] = pbits[0];
		pResults->m_pbits[1] = pbits[1];

		memcpy(pResults->m_pSelectors, pResults->m_pSelectors_temp, sizeof(pResults->m_pSelectors[0]) * pParams->m_num_pixels);
	}
				
	return total_err;
}

static void fixDegenerateEndpoints(uint32_t mode, color_rgba *pTrialMinColor, color_rgba *pTrialMaxColor, const vec4F *pXl, const vec4F *pXh, uint32_t iscale,
	const bc7enc_compress_block_params* pComp_params)
{
	//if ((mode == 1) || (mode == 7))
	//if (mode == 1)
	if ( (mode == 1) || ((mode == 6) && (pComp_params->m_quant_mode6_endpoints)) )
	{
		// fix degenerate case where the input collapses to a single colorspace voxel, and we loose all freedom (test with grayscale ramps)
		for (uint32_t i = 0; i < 3; i++)
		{
			if (pTrialMinColor->m_c[i] == pTrialMaxColor->m_c[i])
			{
				if (fabs(pXl->m_c[i] - pXh->m_c[i]) > 0.0f)
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
				}
			}
		}
	}
}

static uint64_t find_optimal_solution(uint32_t mode, vec4F xl, vec4F xh, const color_cell_compressor_params *pParams, color_cell_compressor_results *pResults,
	const bc7enc_compress_block_params* pComp_params)
{
	vec4F_saturate_in_place(&xl); vec4F_saturate_in_place(&xh);

	if (pParams->m_has_pbits)
	{
		const int iscalep = (1 << (pParams->m_comp_bits + 1)) - 1;
		const float scalep = (float)iscalep;

		const int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

		uint32_t best_pbits[2];
		color_rgba bestMinColor, bestMaxColor;

		if (!pParams->m_endpoints_share_pbit)
		{
			if ((pParams->m_comp_bits == 7) && (pComp_params->m_quant_mode6_endpoints))
			{
				best_pbits[0] = 0;
				bestMinColor.m_c[0] = g_mode6_reduced_quant[(int)((xl.m_c[0] * 2047.0f) + .5f)][0];
				bestMinColor.m_c[1] = g_mode6_reduced_quant[(int)((xl.m_c[1] * 2047.0f) + .5f)][0];
				bestMinColor.m_c[2] = g_mode6_reduced_quant[(int)((xl.m_c[2] * 2047.0f) + .5f)][0];
				bestMinColor.m_c[3] = g_mode6_reduced_quant[(int)((xl.m_c[3] * 2047.0f) + .5f)][0];

				best_pbits[1] = 1;
				bestMaxColor.m_c[0] = g_mode6_reduced_quant[(int)((xh.m_c[0] * 2047.0f) + .5f)][1];
				bestMaxColor.m_c[1] = g_mode6_reduced_quant[(int)((xh.m_c[1] * 2047.0f) + .5f)][1];
				bestMaxColor.m_c[2] = g_mode6_reduced_quant[(int)((xh.m_c[2] * 2047.0f) + .5f)][1];
				bestMaxColor.m_c[3] = g_mode6_reduced_quant[(int)((xh.m_c[3] * 2047.0f) + .5f)][1];
			}
			else
			{
				float best_err0 = 1e+9;
				float best_err1 = 1e+9;

				for (int p = 0; p < 2; p++)
				{
					color_rgba xMinColor, xMaxColor;

					// Notes: The pbit controls which quantization intervals are selected.
					// total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
					// pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
					// rearranging you get for pbit 0: b=floor(v*(total_levels-1)/2+.5)
					// rearranging you get for pbit 1: b=floor((v*(total_levels-1)-1)/2+.5)
					if (pParams->m_comp_bits == 5)
					{
						for (uint32_t c = 0; c < 4; c++)
						{
							int vl = (int)(xl.m_c[c] * 31.0f);
							vl += (xl.m_c[c] > g_mode7_rgba_midpoints[vl][p]);
							xMinColor.m_c[c] = (uint8_t)clampi(vl * 2 + p, p, 63 - 1 + p);

							int vh = (int)(xh.m_c[c] * 31.0f);
							vh += (xh.m_c[c] > g_mode7_rgba_midpoints[vh][p]);
							xMaxColor.m_c[c] = (uint8_t)clampi(vh * 2 + p, p, 63 - 1 + p);
						}
					}
					else
					{
						for (uint32_t c = 0; c < 4; c++)
						{
							xMinColor.m_c[c] = (uint8_t)(clampi(((int)((xl.m_c[c] * scalep - p) / 2.0f + .5f)) * 2 + p, p, iscalep - 1 + p));
							xMaxColor.m_c[c] = (uint8_t)(clampi(((int)((xh.m_c[c] * scalep - p) / 2.0f + .5f)) * 2 + p, p, iscalep - 1 + p));
						}
					}

					color_rgba scaledLow = scale_color(&xMinColor, pParams);
					color_rgba scaledHigh = scale_color(&xMaxColor, pParams);

					float err0 = 0, err1 = 0;
					for (int i = 0; i < totalComps; i++)
					{
						err0 += squaref(scaledLow.m_c[i] - xl.m_c[i] * 255.0f);
						err1 += squaref(scaledHigh.m_c[i] - xh.m_c[i] * 255.0f);
					}

					if (p == 1)
					{
						err0 *= pComp_params->m_pbit1_weight;
						err1 *= pComp_params->m_pbit1_weight;
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
		}
		else
		{
			if ((mode == 1) && (pComp_params->m_bias_mode1_pbits))
			{
				float x = 0.0f;
				for (uint32_t c = 0; c < 3; c++)
					x = std::max(std::max(x, xl.m_c[c]), xh.m_c[c]);
				
				int p = 0;
				if (x > (253.0f / 255.0f))
					p = 1;

				color_rgba xMinColor, xMaxColor;
				for (uint32_t c = 0; c < 4; c++)
				{
					int vl = (int)(xl.m_c[c] * 63.0f);
					vl += (xl.m_c[c] > g_mode1_rgba_midpoints[vl][p]);
					xMinColor.m_c[c] = (uint8_t)clampi(vl * 2 + p, p, 127 - 1 + p);

					int vh = (int)(xh.m_c[c] * 63.0f);
					vh += (xh.m_c[c] > g_mode1_rgba_midpoints[vh][p]);
					xMaxColor.m_c[c] = (uint8_t)clampi(vh * 2 + p, p, 127 - 1 + p);
				}

				best_pbits[0] = p;
				best_pbits[1] = p;
				for (uint32_t j = 0; j < 4; j++)
				{
					bestMinColor.m_c[j] = xMinColor.m_c[j] >> 1;
					bestMaxColor.m_c[j] = xMaxColor.m_c[j] >> 1;
				}
			}
			else
			{
				// Endpoints share pbits
				float best_err = 1e+9;

				for (int p = 0; p < 2; p++)
				{
					color_rgba xMinColor, xMaxColor;
					if (pParams->m_comp_bits == 6)
					{
						for (uint32_t c = 0; c < 4; c++)
						{
							int vl = (int)(xl.m_c[c] * 63.0f);
							vl += (xl.m_c[c] > g_mode1_rgba_midpoints[vl][p]);
							xMinColor.m_c[c] = (uint8_t)clampi(vl * 2 + p, p, 127 - 1 + p);

							int vh = (int)(xh.m_c[c] * 63.0f);
							vh += (xh.m_c[c] > g_mode1_rgba_midpoints[vh][p]);
							xMaxColor.m_c[c] = (uint8_t)clampi(vh * 2 + p, p, 127 - 1 + p);
						}
					}
					else
					{
						for (uint32_t c = 0; c < 4; c++)
						{
							xMinColor.m_c[c] = (uint8_t)(clampi(((int)((xl.m_c[c] * scalep - p) / 2.0f + .5f)) * 2 + p, p, iscalep - 1 + p));
							xMaxColor.m_c[c] = (uint8_t)(clampi(((int)((xh.m_c[c] * scalep - p) / 2.0f + .5f)) * 2 + p, p, iscalep - 1 + p));
						}
					}

					color_rgba scaledLow = scale_color(&xMinColor, pParams);
					color_rgba scaledHigh = scale_color(&xMaxColor, pParams);

					float err = 0;
					for (int i = 0; i < totalComps; i++)
						err += squaref((scaledLow.m_c[i] / 255.0f) - xl.m_c[i]) + squaref((scaledHigh.m_c[i] / 255.0f) - xh.m_c[i]);

					if (p == 1)
						err *= pComp_params->m_pbit1_weight;

					if (err < best_err)
					{
						best_err = err;
						best_pbits[0] = p;
						best_pbits[1] = p;
						for (uint32_t j = 0; j < 4; j++)
						{
							bestMinColor.m_c[j] = xMinColor.m_c[j] >> 1;
							bestMaxColor.m_c[j] = xMaxColor.m_c[j] >> 1;
						}
					}
				}
			}
		}
						
		fixDegenerateEndpoints(mode, &bestMinColor, &bestMaxColor, &xl, &xh, iscalep >> 1, pComp_params);

		if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_u8_notequals(&bestMinColor, &pResults->m_low_endpoint) || color_quad_u8_notequals(&bestMaxColor, &pResults->m_high_endpoint) || (best_pbits[0] != pResults->m_pbits[0]) || (best_pbits[1] != pResults->m_pbits[1]))
			evaluate_solution(&bestMinColor, &bestMaxColor, best_pbits, pParams, pResults, pComp_params);
	}
	else
	{
		const int iscale = (1 << pParams->m_comp_bits) - 1;
		const float scale = (float)iscale;

		color_rgba trialMinColor, trialMaxColor;
		if (pParams->m_comp_bits == 7)
		{
			for (uint32_t c = 0; c < 4; c++)
			{
				int vl = (int)(xl.m_c[c] * 127.0f);
				vl += (xl.m_c[c] > g_mode5_rgba_midpoints[vl]);
				trialMinColor.m_c[c] = (uint8_t)clampi(vl, 0, 127);

				int vh = (int)(xh.m_c[c] * 127.0f);
				vh += (xh.m_c[c] > g_mode5_rgba_midpoints[vh]);
				trialMaxColor.m_c[c] = (uint8_t)clampi(vh, 0, 127);
			}
		}
		else
		{
			color_quad_u8_set_clamped(&trialMinColor, (int)(xl.m_c[0] * scale + .5f), (int)(xl.m_c[1] * scale + .5f), (int)(xl.m_c[2] * scale + .5f), (int)(xl.m_c[3] * scale + .5f));
			color_quad_u8_set_clamped(&trialMaxColor, (int)(xh.m_c[0] * scale + .5f), (int)(xh.m_c[1] * scale + .5f), (int)(xh.m_c[2] * scale + .5f), (int)(xh.m_c[3] * scale + .5f));
		}

		fixDegenerateEndpoints(mode, &trialMinColor, &trialMaxColor, &xl, &xh, iscale, pComp_params);

		if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_u8_notequals(&trialMinColor, &pResults->m_low_endpoint) || color_quad_u8_notequals(&trialMaxColor, &pResults->m_high_endpoint))
			evaluate_solution(&trialMinColor, &trialMaxColor, pResults->m_pbits, pParams, pResults, pComp_params);
	}

	return pResults->m_best_overall_err;
}

static uint64_t color_cell_compression(uint32_t mode, const color_cell_compressor_params *pParams, color_cell_compressor_results *pResults, const bc7enc_compress_block_params *pComp_params)
{
	assert((mode == 6) || (mode == 7) || (!pParams->m_has_alpha));

	pResults->m_best_overall_err = UINT64_MAX;

	// If the partition's colors are all the same in mode 1, then just pack them as a single color.
	if (mode == 1)
	{
		const uint32_t cr = pParams->m_pPixels[0].m_c[0], cg = pParams->m_pPixels[0].m_c[1], cb = pParams->m_pPixels[0].m_c[2];

		bool allSame = true;
		for (uint32_t i = 1; i < pParams->m_num_pixels; i++)
		{
			if ((cr != pParams->m_pPixels[i].m_c[0]) || (cg != pParams->m_pPixels[i].m_c[1]) || (cb != pParams->m_pPixels[i].m_c[2]))
			{
				allSame = false;
				break;
			}
		}

		if (allSame)
			return pack_mode1_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors);
	}
	else if (mode == 7)
	{
		const uint32_t cr = pParams->m_pPixels[0].m_c[0], cg = pParams->m_pPixels[0].m_c[1], cb = pParams->m_pPixels[0].m_c[2], ca = pParams->m_pPixels[0].m_c[3];

		bool allSame = true;
		for (uint32_t i = 1; i < pParams->m_num_pixels; i++)
		{
			if ((cr != pParams->m_pPixels[i].m_c[0]) || (cg != pParams->m_pPixels[i].m_c[1]) || (cb != pParams->m_pPixels[i].m_c[2]) || (ca != pParams->m_pPixels[i].m_c[3]))
			{
				allSame = false;
				break;
			}
		}

		if (allSame)
			return pack_mode7_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, pParams->m_num_pixels, pParams->m_pPixels);
	}

	// Compute partition's mean color and principle axis.
	vec4F meanColor, axis;
	vec4F_set_scalar(&meanColor, 0.0f);

	for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
	{
		vec4F color = vec4F_from_color(&pParams->m_pPixels[i]);
		meanColor = vec4F_add(&meanColor, &color);
	}
				
	vec4F meanColorScaled = vec4F_mul(&meanColor, 1.0f / (float)(pParams->m_num_pixels));

	meanColor = vec4F_mul(&meanColor, 1.0f / (float)(pParams->m_num_pixels * 255.0f));
	vec4F_saturate_in_place(&meanColor);

	if (pParams->m_has_alpha)
	{
		// Use incremental PCA for RGBA PCA, because it's simple.
		vec4F_set_scalar(&axis, 0.0f);
		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			vec4F color = vec4F_from_color(&pParams->m_pPixels[i]);
			color = vec4F_sub(&color, &meanColorScaled);
			vec4F a = vec4F_mul(&color, color.m_c[0]);
			vec4F b = vec4F_mul(&color, color.m_c[1]);
			vec4F c = vec4F_mul(&color, color.m_c[2]);
			vec4F d = vec4F_mul(&color, color.m_c[3]);
			vec4F n = i ? axis : color;
			vec4F_normalize_in_place(&n);
			axis.m_c[0] += vec4F_dot(&a, &n);
			axis.m_c[1] += vec4F_dot(&b, &n);
			axis.m_c[2] += vec4F_dot(&c, &n);
			axis.m_c[3] += vec4F_dot(&d, &n);
		}
		vec4F_normalize_in_place(&axis);
	}
	else
	{
		// Use covar technique for RGB PCA, because it doesn't require per-pixel normalization.
		float cov[6] = { 0, 0, 0, 0, 0, 0 };

		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			const color_rgba *pV = &pParams->m_pPixels[i];
			float r = pV->m_c[0] - meanColorScaled.m_c[0];
			float g = pV->m_c[1] - meanColorScaled.m_c[1];
			float b = pV->m_c[2] - meanColorScaled.m_c[2];
			cov[0] += r*r; cov[1] += r*g; cov[2] += r*b; cov[3] += g*g; cov[4] += g*b; cov[5] += b*b;
		}

		float vfr = .9f, vfg = 1.0f, vfb = .7f;
		for (uint32_t iter = 0; iter < 3; iter++)
		{
			float r = vfr*cov[0] + vfg*cov[1] + vfb*cov[2];
			float g = vfr*cov[1] + vfg*cov[3] + vfb*cov[4];
			float b = vfr*cov[2] + vfg*cov[4] + vfb*cov[5];

			float m = maximumf(maximumf(fabsf(r), fabsf(g)), fabsf(b));
			if (m > 1e-10f)
			{
				m = 1.0f / m;
				r *= m; g *= m;	b *= m;
			}

			vfr = r; vfg = g; vfb = b;
		}

		float len = vfr*vfr + vfg*vfg + vfb*vfb;
		if (len < 1e-10f)
			vec4F_set_scalar(&axis, 0.0f);
		else
		{
			len = 1.0f / sqrtf(len);
			vfr *= len; vfg *= len; vfb *= len;
			vec4F_set(&axis, vfr, vfg, vfb, 0);
		}
	}

	// TODO: Try picking the 2 colors with the largest projection onto the axis, instead of computing new colors along the axis.
				
	if (vec4F_dot(&axis, &axis) < .5f)
	{
		if (pParams->m_perceptual)
			vec4F_set(&axis, .213f, .715f, .072f, pParams->m_has_alpha ? .715f : 0);
		else
			vec4F_set(&axis, 1.0f, 1.0f, 1.0f, pParams->m_has_alpha ? 1.0f : 0);
		vec4F_normalize_in_place(&axis);
	}

	float l = 1e+9f, h = -1e+9f;

	for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
	{
		vec4F color = vec4F_from_color(&pParams->m_pPixels[i]);

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
#if 0
		// Don't compile correctly with VC 2019 in release.
		vec4F temp = minColor;
		minColor = maxColor;
		maxColor = temp;
#else
		float a = minColor.m_c[0], b = minColor.m_c[1], c = minColor.m_c[2], d = minColor.m_c[3];
		minColor.m_c[0] = maxColor.m_c[0];
		minColor.m_c[1] = maxColor.m_c[1];
		minColor.m_c[2] = maxColor.m_c[2];
		minColor.m_c[3] = maxColor.m_c[3];
		maxColor.m_c[0] = a;
		maxColor.m_c[1] = b;
		maxColor.m_c[2] = c;
		maxColor.m_c[3] = d;
#endif
	}

	// First find a solution using the block's PCA.
	if (!find_optimal_solution(mode, minColor, maxColor, pParams, pResults, pComp_params))
		return 0;
	
	if (pComp_params->m_try_least_squares)
	{
		// Now try to refine the solution using least squares by computing the optimal endpoints from the current selectors.
		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);
		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(pParams->m_num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);
		else
			compute_least_squares_endpoints_rgb(pParams->m_num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (!find_optimal_solution(mode, xl, xh, pParams, pResults, pComp_params))
			return 0;
	}
	
	if (pComp_params->m_uber_level > 0)
	{
		// In uber level 1, try varying the selectors a little, somewhat like cluster fit would. First try incrementing the minimum selectors,
		// then try decrementing the selectrors, then try both.
		uint8_t selectors_temp[16], selectors_temp1[16];
		memcpy(selectors_temp, pResults->m_pSelectors, pParams->m_num_pixels);

		const int max_selector = pParams->m_num_selector_weights - 1;

		uint32_t min_sel = 16;
		uint32_t max_sel = 0;
		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			min_sel = minimumu(min_sel, sel);
			max_sel = maximumu(max_sel, sel);
		}

		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
				sel++;
			selectors_temp1[i] = (uint8_t)sel;
		}

		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);
		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);
		else
			compute_least_squares_endpoints_rgb(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (!find_optimal_solution(mode, xl, xh, pParams, pResults, pComp_params))
			return 0;

		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			if ((sel == max_sel) && (sel > 0))
				sel--;
			selectors_temp1[i] = (uint8_t)sel;
		}

		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);
		else
			compute_least_squares_endpoints_rgb(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (!find_optimal_solution(mode, xl, xh, pParams, pResults, pComp_params))
			return 0;

		for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
				sel++;
			else if ((sel == max_sel) && (sel > 0))
				sel--;
			selectors_temp1[i] = (uint8_t)sel;
		}

		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);
		else
			compute_least_squares_endpoints_rgb(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (!find_optimal_solution(mode, xl, xh, pParams, pResults, pComp_params))
			return 0;

		// In uber levels 2+, try taking more advantage of endpoint extrapolation by scaling the selectors in one direction or another.
		const uint32_t uber_err_thresh = (pParams->m_num_pixels * 56) >> 4;
		if ((pComp_params->m_uber_level >= 2) && (pResults->m_best_overall_err > uber_err_thresh))
		{
			const int Q = (pComp_params->m_uber_level >= 4) ? (pComp_params->m_uber_level - 2) : 1;
			for (int ly = -Q; ly <= 1; ly++)
			{
				for (int hy = max_selector - 1; hy <= (max_selector + Q); hy++)
				{
					if ((ly == 0) && (hy == max_selector))
						continue;

					for (uint32_t i = 0; i < pParams->m_num_pixels; i++)
						selectors_temp1[i] = (uint8_t)clampf(floorf((float)max_selector * ((float)selectors_temp[i] - (float)ly) / ((float)hy - (float)ly) + .5f), 0, (float)max_selector);

					//vec4F xl, xh;
					vec4F_set_scalar(&xl, 0.0f);
					vec4F_set_scalar(&xh, 0.0f);
					if (pParams->m_has_alpha)
						compute_least_squares_endpoints_rgba(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);
					else
						compute_least_squares_endpoints_rgb(pParams->m_num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pParams->m_pPixels);

					xl = vec4F_mul(&xl, (1.0f / 255.0f));
					xh = vec4F_mul(&xh, (1.0f / 255.0f));

					if (!find_optimal_solution(mode, xl, xh, pParams, pResults, pComp_params))
						return 0;
				}
			}
		}
	}

	if (mode == 1)
	{
		// Try encoding the partition as a single color by using the optimal singe colors tables to encode the block to its mean.
		color_cell_compressor_results avg_results = *pResults;
		const uint32_t r = (int)(.5f + meanColor.m_c[0] * 255.0f), g = (int)(.5f + meanColor.m_c[1] * 255.0f), b = (int)(.5f + meanColor.m_c[2] * 255.0f);
		uint64_t avg_err = pack_mode1_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp);
		if (avg_err < pResults->m_best_overall_err)
		{
			*pResults = avg_results;
			memcpy(pResults->m_pSelectors, pResults->m_pSelectors_temp, sizeof(pResults->m_pSelectors[0]) * pParams->m_num_pixels);
			pResults->m_best_overall_err = avg_err;
		}
	}
	else if (mode == 7)
	{
		// Try encoding the partition as a single color by using the optimal singe colors tables to encode the block to its mean.
		color_cell_compressor_results avg_results = *pResults;
		const uint32_t r = (int)(.5f + meanColor.m_c[0] * 255.0f), g = (int)(.5f + meanColor.m_c[1] * 255.0f), b = (int)(.5f + meanColor.m_c[2] * 255.0f), a = (int)(.5f + meanColor.m_c[3] * 255.0f);
		uint64_t avg_err = pack_mode7_to_one_color(pParams, &avg_results, r, g, b, a, pResults->m_pSelectors_temp, pParams->m_num_pixels, pParams->m_pPixels);
		if (avg_err < pResults->m_best_overall_err)
		{
			*pResults = avg_results;
			memcpy(pResults->m_pSelectors, pResults->m_pSelectors_temp, sizeof(pResults->m_pSelectors[0]) * pParams->m_num_pixels);
			pResults->m_best_overall_err = avg_err;
		}
	}
				
	return pResults->m_best_overall_err;
}

static uint64_t color_cell_compression_est_mode1(uint32_t num_pixels, const color_rgba *pPixels, bool perceptual, uint32_t pweights[4], uint64_t best_err_so_far)
{
	// Find RGB bounds as an approximation of the block's principle axis
	uint32_t lr = 255, lg = 255, lb = 255;
	uint32_t hr = 0, hg = 0, hb = 0;
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		const color_rgba *pC = &pPixels[i];
		if (pC->m_c[0] < lr) lr = pC->m_c[0];
		if (pC->m_c[1] < lg) lg = pC->m_c[1];
		if (pC->m_c[2] < lb) lb = pC->m_c[2];
		if (pC->m_c[0] > hr) hr = pC->m_c[0];
		if (pC->m_c[1] > hg) hg = pC->m_c[1];
		if (pC->m_c[2] > hb) hb = pC->m_c[2];
	}
		
	color_rgba lowColor; color_quad_u8_set(&lowColor, lr, lg, lb, 0);
	color_rgba highColor; color_quad_u8_set(&highColor, hr, hg, hb, 0);

	// Place endpoints at bbox diagonals and compute interpolated colors 
	const uint32_t N = 8;
	color_rgba weightedColors[8];

	weightedColors[0] = lowColor;
	weightedColors[N - 1] = highColor;
	for (uint32_t i = 1; i < (N - 1); i++)
	{
		weightedColors[i].m_c[0] = (uint8_t)((lowColor.m_c[0] * (64 - g_bc7_weights3[i]) + highColor.m_c[0] * g_bc7_weights3[i] + 32) >> 6);
		weightedColors[i].m_c[1] = (uint8_t)((lowColor.m_c[1] * (64 - g_bc7_weights3[i]) + highColor.m_c[1] * g_bc7_weights3[i] + 32) >> 6);
		weightedColors[i].m_c[2] = (uint8_t)((lowColor.m_c[2] * (64 - g_bc7_weights3[i]) + highColor.m_c[2] * g_bc7_weights3[i] + 32) >> 6);
	}

	// Compute dots and thresholds
	const int ar = highColor.m_c[0] - lowColor.m_c[0];
	const int ag = highColor.m_c[1] - lowColor.m_c[1];
	const int ab = highColor.m_c[2] - lowColor.m_c[2];

	int dots[8];
	for (uint32_t i = 0; i < N; i++)
		dots[i] = weightedColors[i].m_c[0] * ar + weightedColors[i].m_c[1] * ag + weightedColors[i].m_c[2] * ab;

	int thresh[8 - 1];
	for (uint32_t i = 0; i < (N - 1); i++)
		thresh[i] = (dots[i] + dots[i + 1] + 1) >> 1;

	uint64_t total_err = 0;
	if (perceptual)
	{
		// Transform block's interpolated colors to YCbCr
		int l1[8], cr1[8], cb1[8];
		for (int j = 0; j < 8; j++)
		{
			const color_rgba *pE1 = &weightedColors[j];
			l1[j] = pE1->m_c[0] * 109 + pE1->m_c[1] * 366 + pE1->m_c[2] * 37;
			cr1[j] = ((int)pE1->m_c[0] << 9) - l1[j];
			cb1[j] = ((int)pE1->m_c[2] << 9) - l1[j];
		}

		for (uint32_t i = 0; i < num_pixels; i++)
		{
			const color_rgba *pC = &pPixels[i];

			int d = ar * pC->m_c[0] + ag * pC->m_c[1] + ab * pC->m_c[2];

			// Find approximate selector
			uint32_t s = 0;
			if (d >= thresh[6])
				s = 7;
			else if (d >= thresh[5])
				s = 6;
			else if (d >= thresh[4])
				s = 5;
			else if (d >= thresh[3])
				s = 4;
			else if (d >= thresh[2])
				s = 3;
			else if (d >= thresh[1])
				s = 2;
			else if (d >= thresh[0])
				s = 1;

			// Compute error
			const int l2 = pC->m_c[0] * 109 + pC->m_c[1] * 366 + pC->m_c[2] * 37;
			const int cr2 = ((int)pC->m_c[0] << 9) - l2;
			const int cb2 = ((int)pC->m_c[2] << 9) - l2;

			const int dl = (l1[s] - l2) >> 8;
			const int dcr = (cr1[s] - cr2) >> 8;
			const int dcb = (cb1[s] - cb2) >> 8;

			int ie = (pweights[0] * dl * dl) + (pweights[1] * dcr * dcr) + (pweights[2] * dcb * dcb);

			total_err += ie;
			if (total_err > best_err_so_far)
				break;
		}
	}
	else
	{
		for (uint32_t i = 0; i < num_pixels; i++)
		{
			const color_rgba *pC = &pPixels[i];

			int d = ar * pC->m_c[0] + ag * pC->m_c[1] + ab * pC->m_c[2];

			// Find approximate selector
			uint32_t s = 0;
			if (d >= thresh[6])
				s = 7;
			else if (d >= thresh[5])
				s = 6;
			else if (d >= thresh[4])
				s = 5;
			else if (d >= thresh[3])
				s = 4;
			else if (d >= thresh[2])
				s = 3;
			else if (d >= thresh[1])
				s = 2;
			else if (d >= thresh[0])
				s = 1;

			// Compute error
			const color_rgba *pE1 = &weightedColors[s];

			int dr = (int)pE1->m_c[0] - (int)pC->m_c[0];
			int dg = (int)pE1->m_c[1] - (int)pC->m_c[1];
			int db = (int)pE1->m_c[2] - (int)pC->m_c[2];

			total_err += pweights[0] * (dr * dr) + pweights[1] * (dg * dg) + pweights[2] * (db * db);
			if (total_err > best_err_so_far)
				break;
		}
	}

	return total_err;
}

static uint64_t color_cell_compression_est_mode7(uint32_t num_pixels, const color_rgba * pPixels, bool perceptual, uint32_t pweights[4], uint64_t best_err_so_far)
{
	// Find RGB bounds as an approximation of the block's principle axis
	uint32_t lr = 255, lg = 255, lb = 255, la = 255;
	uint32_t hr = 0, hg = 0, hb = 0, ha = 0;
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		const color_rgba* pC = &pPixels[i];
		if (pC->m_c[0] < lr) lr = pC->m_c[0];
		if (pC->m_c[1] < lg) lg = pC->m_c[1];
		if (pC->m_c[2] < lb) lb = pC->m_c[2];
		if (pC->m_c[3] < la) la = pC->m_c[3];

		if (pC->m_c[0] > hr) hr = pC->m_c[0];
		if (pC->m_c[1] > hg) hg = pC->m_c[1];
		if (pC->m_c[2] > hb) hb = pC->m_c[2];
		if (pC->m_c[3] > ha) ha = pC->m_c[3];
	}

	color_rgba lowColor; color_quad_u8_set(&lowColor, lr, lg, lb, la);
	color_rgba highColor; color_quad_u8_set(&highColor, hr, hg, hb, ha);

	// Place endpoints at bbox diagonals and compute interpolated colors 
	const uint32_t N = 4;
	color_rgba weightedColors[4];

	weightedColors[0] = lowColor;
	weightedColors[N - 1] = highColor;
	for (uint32_t i = 1; i < (N - 1); i++)
	{
		weightedColors[i].m_c[0] = (uint8_t)((lowColor.m_c[0] * (64 - g_bc7_weights2[i]) + highColor.m_c[0] * g_bc7_weights2[i] + 32) >> 6);
		weightedColors[i].m_c[1] = (uint8_t)((lowColor.m_c[1] * (64 - g_bc7_weights2[i]) + highColor.m_c[1] * g_bc7_weights2[i] + 32) >> 6);
		weightedColors[i].m_c[2] = (uint8_t)((lowColor.m_c[2] * (64 - g_bc7_weights2[i]) + highColor.m_c[2] * g_bc7_weights2[i] + 32) >> 6);
		weightedColors[i].m_c[3] = (uint8_t)((lowColor.m_c[3] * (64 - g_bc7_weights2[i]) + highColor.m_c[3] * g_bc7_weights2[i] + 32) >> 6);
	}

	// Compute dots and thresholds
	const int ar = highColor.m_c[0] - lowColor.m_c[0];
	const int ag = highColor.m_c[1] - lowColor.m_c[1];
	const int ab = highColor.m_c[2] - lowColor.m_c[2];
	const int aa = highColor.m_c[3] - lowColor.m_c[3];

	int dots[4];
	for (uint32_t i = 0; i < N; i++)
		dots[i] = weightedColors[i].m_c[0] * ar + weightedColors[i].m_c[1] * ag + weightedColors[i].m_c[2] * ab + weightedColors[i].m_c[3] * aa;

	int thresh[4 - 1];
	for (uint32_t i = 0; i < (N - 1); i++)
		thresh[i] = (dots[i] + dots[i + 1] + 1) >> 1;

	uint64_t total_err = 0;
	if (perceptual)
	{
		// Transform block's interpolated colors to YCbCr
		int l1[4], cr1[4], cb1[4];
		for (int j = 0; j < 4; j++)
		{
			const color_rgba* pE1 = &weightedColors[j];
			l1[j] = pE1->m_c[0] * 109 + pE1->m_c[1] * 366 + pE1->m_c[2] * 37;
			cr1[j] = ((int)pE1->m_c[0] << 9) - l1[j];
			cb1[j] = ((int)pE1->m_c[2] << 9) - l1[j];
		}

		for (uint32_t i = 0; i < num_pixels; i++)
		{
			const color_rgba* pC = &pPixels[i];

			int d = ar * pC->m_c[0] + ag * pC->m_c[1] + ab * pC->m_c[2] + aa * pC->m_c[3];

			// Find approximate selector
			uint32_t s = 0;
			if (d >= thresh[2])
				s = 3;
			else if (d >= thresh[1])
				s = 2;
			else if (d >= thresh[0])
				s = 1;

			// Compute error
			const int l2 = pC->m_c[0] * 109 + pC->m_c[1] * 366 + pC->m_c[2] * 37;
			const int cr2 = ((int)pC->m_c[0] << 9) - l2;
			const int cb2 = ((int)pC->m_c[2] << 9) - l2;

			const int dl = (l1[s] - l2) >> 8;
			const int dcr = (cr1[s] - cr2) >> 8;
			const int dcb = (cb1[s] - cb2) >> 8;

			const int dca = (int)pC->m_c[3] - (int)weightedColors[s].m_c[3];

			int ie = (pweights[0] * dl * dl) + (pweights[1] * dcr * dcr) + (pweights[2] * dcb * dcb) + (pweights[3] * dca * dca);

			total_err += ie;
			if (total_err > best_err_so_far)
				break;
		}
	}
	else
	{
		for (uint32_t i = 0; i < num_pixels; i++)
		{
			const color_rgba* pC = &pPixels[i];

			int d = ar * pC->m_c[0] + ag * pC->m_c[1] + ab * pC->m_c[2] + aa * pC->m_c[3];

			// Find approximate selector
			uint32_t s = 0;
			if (d >= thresh[2])
				s = 3;
			else if (d >= thresh[1])
				s = 2;
			else if (d >= thresh[0])
				s = 1;

			// Compute error
			const color_rgba* pE1 = &weightedColors[s];

			int dr = (int)pE1->m_c[0] - (int)pC->m_c[0];
			int dg = (int)pE1->m_c[1] - (int)pC->m_c[1];
			int db = (int)pE1->m_c[2] - (int)pC->m_c[2];
			int da = (int)pE1->m_c[3] - (int)pC->m_c[3];

			total_err += pweights[0] * (dr * dr) + pweights[1] * (dg * dg) + pweights[2] * (db * db) + pweights[3] * (da * da);
			if (total_err > best_err_so_far)
				break;
		}
	}

	return total_err;
}

// This table contains bitmasks indicating which "key" partitions must be best ranked before this partition is worth evaluating.
// We first rank the best/most used 14 partitions (sorted by usefulness), record the best one found as the key partition, then use 
// that to control the other partitions to evaluate. The quality loss is ~.08 dB RGB PSNR, the perf gain is up to ~11% (at uber level 0).
static const uint32_t g_partition_predictors[35] =
{
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	(1 << 1) | (1 << 2) | (1 << 8),
	(1 << 1) | (1 << 3) | (1 << 7),
	UINT32_MAX,
	UINT32_MAX,
	(1 << 2) | (1 << 8) | (1 << 16),
	(1 << 7) | (1 << 3) | (1 << 15),
	UINT32_MAX,
	(1 << 8) | (1 << 14) | (1 << 16),
	(1 << 7) | (1 << 14) | (1 << 15),
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	(1 << 14) | (1 << 15),
	(1 << 16) | (1 << 22) | (1 << 14),
	(1 << 17) | (1 << 24) | (1 << 14),
	(1 << 2) | (1 << 14) | (1 << 15) | (1 << 1),
	UINT32_MAX,
	(1 << 1) | (1 << 3) | (1 << 14) | (1 << 16) | (1 << 22),
	UINT32_MAX,
	(1 << 1) | (1 << 2) | (1 << 15) | (1 << 17) | (1 << 24),
	(1 << 1) | (1 << 3) | (1 << 22),
	UINT32_MAX,
	UINT32_MAX,
	UINT32_MAX,
	(1 << 14) | (1 << 15) | (1 << 16) | (1 << 17),
	UINT32_MAX,
	UINT32_MAX,
	(1 << 1) | (1 << 2) | (1 << 3) | (1 << 27) | (1 << 4) | (1 << 24),
	(1 << 14) | (1 << 15) | (1 << 16) | (1 << 11) | (1 << 17) | (1 << 27)
};

// Estimate the partition used by modes 1/7. This scans through each partition and computes an approximate error for each.
static uint32_t estimate_partition(const color_rgba *pPixels, const bc7enc_compress_block_params *pComp_params, uint32_t pweights[4], uint32_t mode)
{
	const uint32_t total_partitions = minimumu(pComp_params->m_max_partitions, BC7ENC_MAX_PARTITIONS);
	if (total_partitions <= 1)
		return 0;

	uint64_t best_err = UINT64_MAX;
	uint32_t best_partition = 0;

	// Partition order sorted by usage frequency across a large test corpus. Pattern 34 (checkerboard) must appear in slot 34.
	// Using a sorted order allows the user to decrease the # of partitions to scan with minimal loss in quality.
	static const uint8_t s_sorted_partition_order[64] =
	{
		1 - 1, 14 - 1, 2 - 1, 3 - 1, 16 - 1, 15 - 1, 11 - 1, 17 - 1,
		4 - 1, 24 - 1, 27 - 1, 7 - 1, 8 - 1, 22 - 1, 20 - 1, 30 - 1,
		9 - 1, 5 - 1, 10 - 1, 21 - 1, 6 - 1, 32 - 1, 23 - 1, 18 - 1,
		19 - 1, 12 - 1, 13 - 1, 31 - 1, 25 - 1, 26 - 1, 29 - 1, 28 - 1,
		33 - 1, 34 - 1, 35 - 1, 46 - 1, 47 - 1, 52 - 1, 50 - 1, 51 - 1,
		49 - 1, 39 - 1, 40 - 1, 38 - 1, 54 - 1, 53 - 1, 55 - 1, 37 - 1,
		58 - 1, 59 - 1, 56 - 1, 42 - 1, 41 - 1, 43 - 1, 44 - 1, 60 - 1,
		45 - 1, 57 - 1, 48 - 1, 36 - 1, 61 - 1, 64 - 1, 63 - 1, 62 - 1
	};

	assert(s_sorted_partition_order[34] == 34);

	int best_key_partition = 0;

	for (uint32_t partition_iter = 0; (partition_iter < total_partitions) && (best_err > 0); partition_iter++)
	{
		const uint32_t partition = s_sorted_partition_order[partition_iter];

		// Check to see if we should bother evaluating this partition at all, depending on the best partition found from the first 14.
		if (pComp_params->m_mode17_partition_estimation_filterbank)
		{
			if ((partition_iter >= 14) && (partition_iter <= 34))
			{
				const uint32_t best_key_partition_bitmask = 1 << (best_key_partition + 1);
				if ((g_partition_predictors[partition] & best_key_partition_bitmask) == 0)
				{
					if (partition_iter == 34)
						break;

					continue;
				}
			}
		}

		const uint8_t *pPartition = &g_bc7_partition2[partition * 16];

		color_rgba subset_colors[2][16];
		uint32_t subset_total_colors[2] = { 0, 0 };
		for (uint32_t index = 0; index < 16; index++)
			subset_colors[pPartition[index]][subset_total_colors[pPartition[index]]++] = pPixels[index];
						
		uint64_t total_subset_err = 0;
		for (uint32_t subset = 0; (subset < 2) && (total_subset_err < best_err); subset++)
		{
			if (mode == 7)
				total_subset_err += color_cell_compression_est_mode7(subset_total_colors[subset], &subset_colors[subset][0], pComp_params->m_perceptual, pweights, best_err);
			else
				total_subset_err += color_cell_compression_est_mode1(subset_total_colors[subset], &subset_colors[subset][0], pComp_params->m_perceptual, pweights, best_err);
		}

		if (partition < 16)
		{
			total_subset_err = (uint64_t)((double)total_subset_err * pComp_params->m_low_frequency_partition_weight + .5f);
		}

		if (total_subset_err < best_err)
		{
			best_err = total_subset_err;
			best_partition = partition;
		}

		// If the checkerboard pattern doesn't get the highest ranking vs. the previous (lower frequency) patterns, then just stop now because statistically the subsequent patterns won't do well either.
		if ((partition == 34) && (best_partition != 34))
			break;

		if (partition_iter == 13)
			best_key_partition = best_partition;

	} // partition

	return best_partition;
}

static void set_block_bits(uint8_t *pBytes, uint32_t val, uint32_t num_bits, uint32_t *pCur_ofs)
{
	assert((num_bits <= 32) && (val < (1ULL << num_bits)));
	while (num_bits)
	{
		const uint32_t n = minimumu(8 - (*pCur_ofs & 7), num_bits);
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
	uint8_t m_selectors[16];
	uint8_t m_alpha_selectors[16];
	color_rgba m_low[3];
	color_rgba m_high[3];
	uint32_t m_pbits[3][2];
	uint32_t m_rotation;
	uint32_t m_index_selector;
};

void encode_bc7_block(void* pBlock, const bc7_optimization_results* pResults)
{
	assert(pResults->m_index_selector <= 1);
	assert(pResults->m_rotation <= 3);

	const uint32_t best_mode = pResults->m_mode;

	const uint32_t total_subsets = g_bc7_num_subsets[best_mode];
	const uint32_t total_partitions = 1 << g_bc7_partition_bits[best_mode];
	//const uint32_t num_rotations = 1 << g_bc7_rotation_bits[best_mode];
	//const uint32_t num_index_selectors = (best_mode == 4) ? 2 : 1;

	const uint8_t* pPartition;
	if (total_subsets == 1)
		pPartition = &g_bc7_partition1[0];
	else if (total_subsets == 2)
		pPartition = &g_bc7_partition2[pResults->m_partition * 16];
	else
		pPartition = &g_bc7_partition3[pResults->m_partition * 16];

	uint8_t color_selectors[16];
	memcpy(color_selectors, pResults->m_selectors, 16);

	uint8_t alpha_selectors[16];
	memcpy(alpha_selectors, pResults->m_alpha_selectors, 16);

	color_rgba low[3], high[3];
	memcpy(low, pResults->m_low, sizeof(low));
	memcpy(high, pResults->m_high, sizeof(high));

	uint32_t pbits[3][2];
	memcpy(pbits, pResults->m_pbits, sizeof(pbits));

	int anchor[3] = { -1, -1, -1 };

	for (uint32_t k = 0; k < total_subsets; k++)
	{
		uint32_t anchor_index = 0;
		if (k)
		{
			if ((total_subsets == 3) && (k == 1))
				anchor_index = g_bc7_table_anchor_index_third_subset_1[pResults->m_partition];
			else if ((total_subsets == 3) && (k == 2))
				anchor_index = g_bc7_table_anchor_index_third_subset_2[pResults->m_partition];
			else
				anchor_index = g_bc7_table_anchor_index_second_subset[pResults->m_partition];
		}

		anchor[k] = anchor_index;

		const uint32_t color_index_bits = get_bc7_color_index_size(best_mode, pResults->m_index_selector);
		const uint32_t num_color_indices = 1 << color_index_bits;

		if (color_selectors[anchor_index] & (num_color_indices >> 1))
		{
			for (uint32_t i = 0; i < 16; i++)
				if (pPartition[i] == k)
					color_selectors[i] = (uint8_t)((num_color_indices - 1) - color_selectors[i]);

			if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
			{
				for (uint32_t q = 0; q < 3; q++)
				{
					uint8_t t = low[k].m_c[q];
					low[k].m_c[q] = high[k].m_c[q];
					high[k].m_c[q] = t;
				}
			}
			else
			{
				color_rgba tmp = low[k];
				low[k] = high[k];
				high[k] = tmp;
			}

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

			if (alpha_selectors[anchor_index] & (num_alpha_indices >> 1))
			{
				for (uint32_t i = 0; i < 16; i++)
					if (pPartition[i] == k)
						alpha_selectors[i] = (uint8_t)((num_alpha_indices - 1) - alpha_selectors[i]);

				uint8_t t = low[k].m_c[3];
				low[k].m_c[3] = high[k].m_c[3];
				high[k].m_c[3] = t;
			}
		}
	}

	uint8_t* pBlock_bytes = (uint8_t*)(pBlock);
	memset(pBlock_bytes, 0, BC7ENC_BLOCK_SIZE);

	uint32_t cur_bit_ofs = 0;
	set_block_bits(pBlock_bytes, 1 << best_mode, best_mode + 1, &cur_bit_ofs);

	if ((best_mode == 4) || (best_mode == 5))
		set_block_bits(pBlock_bytes, pResults->m_rotation, 2, &cur_bit_ofs);

	if (best_mode == 4)
		set_block_bits(pBlock_bytes, pResults->m_index_selector, 1, &cur_bit_ofs);

	if (total_partitions > 1)
		set_block_bits(pBlock_bytes, pResults->m_partition, (total_partitions == 64) ? 6 : 4, &cur_bit_ofs);

	const uint32_t total_comps = (best_mode >= 4) ? 4 : 3;
	for (uint32_t comp = 0; comp < total_comps; comp++)
	{
		for (uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, low[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
			set_block_bits(pBlock_bytes, high[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
		}
	}

	if (g_bc7_mode_has_p_bits[best_mode])
	{
		for (uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, pbits[subset][0], 1, &cur_bit_ofs);
			if (!g_bc7_mode_has_shared_p_bits[best_mode])
				set_block_bits(pBlock_bytes, pbits[subset][1], 1, &cur_bit_ofs);
		}
	}

	for (uint32_t y = 0; y < 4; y++)
	{
		for (uint32_t x = 0; x < 4; x++)
		{
			int idx = x + y * 4;

			uint32_t n = pResults->m_index_selector ? get_bc7_alpha_index_size(best_mode, pResults->m_index_selector) : get_bc7_color_index_size(best_mode, pResults->m_index_selector);

			if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
				n--;

			set_block_bits(pBlock_bytes, pResults->m_index_selector ? alpha_selectors[idx] : color_selectors[idx], n, &cur_bit_ofs);
		}
	}

	if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
	{
		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
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

static void handle_alpha_block_mode5(const color_rgba* pPixels, const bc7enc_compress_block_params* pComp_params, color_cell_compressor_params* pParams, uint32_t lo_a, uint32_t hi_a, bc7_optimization_results* pOpt_results5, uint64_t* pMode5_err, uint64_t* pMode5_alpha_err)
{
	pParams->m_pSelector_weights = g_bc7_weights2;
	pParams->m_pSelector_weightsx = (const vec4F*)g_bc7_weights2x;
	pParams->m_num_selector_weights = 4;

	pParams->m_comp_bits = 7;
	pParams->m_has_pbits = false;
	pParams->m_endpoints_share_pbit = false;
	pParams->m_has_alpha = false;

	pParams->m_perceptual = pComp_params->m_perceptual;

	pParams->m_num_pixels = 16;
	pParams->m_pPixels = pPixels;

	color_cell_compressor_results results5;
	results5.m_pSelectors = pOpt_results5->m_selectors;

	uint8_t selectors_temp[16];
	results5.m_pSelectors_temp = selectors_temp;

	*pMode5_err = color_cell_compression(5, pParams, &results5, pComp_params);
	assert(*pMode5_err == results5.m_best_overall_err);

	pOpt_results5->m_low[0] = results5.m_low_endpoint;
	pOpt_results5->m_high[0] = results5.m_high_endpoint;

	if (lo_a == hi_a)
	{
		*pMode5_alpha_err = 0;
		pOpt_results5->m_low[0].m_c[3] = (uint8_t)lo_a;
		pOpt_results5->m_high[0].m_c[3] = (uint8_t)hi_a;
		memset(pOpt_results5->m_alpha_selectors, 0, sizeof(pOpt_results5->m_alpha_selectors));
	}
	else
	{
		*pMode5_alpha_err = UINT64_MAX;

		const uint32_t total_passes = (pComp_params->m_uber_level >= 1) ? 3 : 2;
		for (uint32_t pass = 0; pass < total_passes; pass++)
		{
			int32_t vals[4];
			vals[0] = lo_a;
			vals[3] = hi_a;

			const int32_t w_s1 = 21, w_s2 = 43;
			vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
			vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

			uint8_t trial_alpha_selectors[16];

			uint64_t trial_alpha_err = 0;
			for (uint32_t i = 0; i < 16; i++)
			{
				const int32_t a = pParams->m_pPixels[i].m_c[3];

				int s = 0;
				int32_t be = iabs32(a - vals[0]);
				int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
				e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
				e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

				trial_alpha_selectors[i] = (uint8_t)s;

				uint32_t a_err = (uint32_t)(be * be) * pParams->m_weights[3];

				trial_alpha_err += a_err;
			}

			if (trial_alpha_err < *pMode5_alpha_err)
			{
				*pMode5_alpha_err = trial_alpha_err;
				pOpt_results5->m_low[0].m_c[3] = (uint8_t)lo_a;
				pOpt_results5->m_high[0].m_c[3] = (uint8_t)hi_a;
				memcpy(pOpt_results5->m_alpha_selectors, trial_alpha_selectors, sizeof(pOpt_results5->m_alpha_selectors));
			}

			if (pass != (total_passes - 1U)) 
			{
				float xl, xh;
				compute_least_squares_endpoints_a(16, trial_alpha_selectors, (const vec4F*)g_bc7_weights2x, &xl, &xh, pParams->m_pPixels);

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

		*pMode5_err += *pMode5_alpha_err;
	}
}

static void handle_alpha_block(void *pBlock, const color_rgba *pPixels, const bc7enc_compress_block_params *pComp_params, color_cell_compressor_params *pParams)
{
	assert((pComp_params->m_mode_mask & (1 << 6)) || (pComp_params->m_mode_mask & (1 << 5)) || (pComp_params->m_mode_mask & (1 << 7)));

	pParams->m_pSelector_weights = g_bc7_weights4;
	pParams->m_pSelector_weightsx = (const vec4F *)g_bc7_weights4x;
	pParams->m_num_selector_weights = 16;
	pParams->m_comp_bits = 7;
	pParams->m_has_pbits = true;
	pParams->m_endpoints_share_pbit = false;
	pParams->m_has_alpha = true;
	pParams->m_perceptual = pComp_params->m_perceptual;
	pParams->m_num_pixels = 16;
	pParams->m_pPixels = pPixels;
		
	bc7_optimization_results opt_results6, opt_results5, opt_results7;
	color_cell_compressor_results results6;
	memset(&results6, 0, sizeof(results6));

	uint64_t best_err = UINT64_MAX;
	uint32_t best_mode = 0;
	uint8_t selectors_temp[16];

	if (pComp_params->m_mode_mask & (1 << 6))
	{
		results6.m_pSelectors = opt_results6.m_selectors;
		results6.m_pSelectors_temp = selectors_temp;

		best_err = (uint64_t)(color_cell_compression(6, pParams, &results6, pComp_params) * pComp_params->m_mode6_error_weight + .5f);
		best_mode = 6;
	}

	if ((best_err > 0) && (pComp_params->m_mode_mask & (1 << 5)))
	{
		uint32_t lo_a = 255, hi_a = 0;
		for (uint32_t i = 0; i < 16; i++)
		{
			uint32_t a = pPixels[i].m_c[3];
			lo_a = minimumu(lo_a, a);
			hi_a = maximumu(hi_a, a);
		}

		uint64_t mode5_err, mode5_alpha_err;
		handle_alpha_block_mode5(pPixels, pComp_params, pParams, lo_a, hi_a, &opt_results5, &mode5_err, &mode5_alpha_err);

		mode5_err = (uint64_t)(mode5_err * pComp_params->m_mode5_error_weight + .5f);

		if (mode5_err < best_err)
		{
			best_err = mode5_err;
			best_mode = 5;
		}
	}

	if ((best_err > 0) && (pComp_params->m_mode_mask & (1 << 7)))
	{
		const uint32_t trial_partition = estimate_partition(pPixels, pComp_params, pParams->m_weights, 7);

		pParams->m_pSelector_weights = g_bc7_weights2;
		pParams->m_pSelector_weightsx = (const vec4F*)g_bc7_weights2x;
		pParams->m_num_selector_weights = 4;
		pParams->m_comp_bits = 5;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;
		pParams->m_has_alpha = true;

		const uint8_t* pPartition = &g_bc7_partition2[trial_partition * 16];

		color_rgba subset_colors[2][16];

		uint32_t subset_total_colors7[2] = { 0, 0 };

		uint8_t subset_pixel_index7[2][16];
		uint8_t subset_selectors7[2][16];
		color_cell_compressor_results subset_results7[2];

		for (uint32_t idx = 0; idx < 16; idx++)
		{
			const uint32_t p = pPartition[idx];
			subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
			subset_pixel_index7[p][subset_total_colors7[p]] = (uint8_t)idx;
			subset_total_colors7[p]++;
		}

		uint64_t trial_err = 0;
		for (uint32_t subset = 0; subset < 2; subset++)
		{
			pParams->m_num_pixels = subset_total_colors7[subset];
			pParams->m_pPixels = &subset_colors[subset][0];

			color_cell_compressor_results* pResults = &subset_results7[subset];
			pResults->m_pSelectors = &subset_selectors7[subset][0];
			pResults->m_pSelectors_temp = selectors_temp;
			uint64_t err = color_cell_compression(7, pParams, pResults, pComp_params);
			trial_err += err;
			if ((uint64_t)(trial_err * pComp_params->m_mode7_error_weight + .5f) > best_err)
				break;

		} // subset

		const uint64_t mode7_trial_err = (uint64_t)(trial_err * pComp_params->m_mode7_error_weight + .5f);

		if (mode7_trial_err < best_err)
		{
			best_err = mode7_trial_err;
			best_mode = 7;
			opt_results7.m_mode = 7;
			opt_results7.m_partition = trial_partition;
			opt_results7.m_index_selector = 0;
			opt_results7.m_rotation = 0;
			for (uint32_t subset = 0; subset < 2; subset++)
			{
				for (uint32_t i = 0; i < subset_total_colors7[subset]; i++)
					opt_results7.m_selectors[subset_pixel_index7[subset][i]] = subset_selectors7[subset][i];
				opt_results7.m_low[subset] = subset_results7[subset].m_low_endpoint;
				opt_results7.m_high[subset] = subset_results7[subset].m_high_endpoint;
				opt_results7.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
				opt_results7.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
			}
		}
	}

	if (best_mode == 7)
	{
		encode_bc7_block(pBlock, &opt_results7);
	}
	else if (best_mode == 5)
	{
		opt_results5.m_mode = 5;
		opt_results5.m_partition = 0;
		opt_results5.m_rotation = 0;
		opt_results5.m_index_selector = 0;

		encode_bc7_block(pBlock, &opt_results5);
	}
	else if (best_mode == 6)
	{
		opt_results6.m_mode = 6;
		opt_results6.m_partition = 0;
		opt_results6.m_low[0] = results6.m_low_endpoint;
		opt_results6.m_high[0] = results6.m_high_endpoint;
		opt_results6.m_pbits[0][0] = results6.m_pbits[0];
		opt_results6.m_pbits[0][1] = results6.m_pbits[1];
		opt_results6.m_rotation = 0;
		opt_results6.m_index_selector = 0;

		encode_bc7_block(pBlock, &opt_results6);
	}
	else
	{
		assert(0);
	}
}

static void handle_opaque_block(void *pBlock, const color_rgba *pPixels, const bc7enc_compress_block_params *pComp_params, color_cell_compressor_params *pParams)
{
	assert((pComp_params->m_mode_mask & (1 << 6)) || (pComp_params->m_mode_mask & (1 << 1)));

	uint8_t selectors_temp[16];
		
	bc7_optimization_results opt_results;

	uint64_t best_err = UINT64_MAX;
		
	pParams->m_perceptual = pComp_params->m_perceptual;
	pParams->m_num_pixels = 16;
	pParams->m_pPixels = pPixels;
	pParams->m_has_alpha = false;

	opt_results.m_partition = 0;
	opt_results.m_index_selector = 0;
	opt_results.m_rotation = 0;

	// Mode 6
	if (pComp_params->m_mode_mask & (1 << 6))
	{
		pParams->m_pSelector_weights = g_bc7_weights4;
		pParams->m_pSelector_weightsx = (const vec4F*)g_bc7_weights4x;
		pParams->m_num_selector_weights = 16;
		pParams->m_comp_bits = 7;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		color_cell_compressor_results results6;
		results6.m_pSelectors = opt_results.m_selectors;
		results6.m_pSelectors_temp = selectors_temp;

		best_err = (uint64_t)(color_cell_compression(6, pParams, &results6, pComp_params) * pComp_params->m_mode6_error_weight + .5f);

		opt_results.m_mode = 6;
		opt_results.m_low[0] = results6.m_low_endpoint;
		opt_results.m_high[0] = results6.m_high_endpoint;
		opt_results.m_pbits[0][0] = results6.m_pbits[0];
		opt_results.m_pbits[0][1] = results6.m_pbits[1];
	}

	// Mode 1
	if ((best_err > 0) && (pComp_params->m_max_partitions > 0) && (pComp_params->m_mode_mask & (1 << 1)))
	{
		const uint32_t trial_partition = estimate_partition(pPixels, pComp_params, pParams->m_weights, 1);
		
		pParams->m_pSelector_weights = g_bc7_weights3;
		pParams->m_pSelector_weightsx = (const vec4F *)g_bc7_weights3x;
		pParams->m_num_selector_weights = 8;
		pParams->m_comp_bits = 6;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = true;

		const uint8_t *pPartition = &g_bc7_partition2[trial_partition * 16];

		color_rgba subset_colors[2][16];

		uint32_t subset_total_colors1[2] = { 0, 0 };

		uint8_t subset_pixel_index1[2][16];
		uint8_t subset_selectors1[2][16];
		color_cell_compressor_results subset_results1[2];

		for (uint32_t idx = 0; idx < 16; idx++)
		{
			const uint32_t p = pPartition[idx];
			subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
			subset_pixel_index1[p][subset_total_colors1[p]] = (uint8_t)idx;
			subset_total_colors1[p]++;
		}

		uint64_t trial_err = 0;
		for (uint32_t subset = 0; subset < 2; subset++)
		{
			pParams->m_num_pixels = subset_total_colors1[subset];
			pParams->m_pPixels = &subset_colors[subset][0];

			color_cell_compressor_results *pResults = &subset_results1[subset];
			pResults->m_pSelectors = &subset_selectors1[subset][0];
			pResults->m_pSelectors_temp = selectors_temp;
			uint64_t err = color_cell_compression(1, pParams, pResults, pComp_params);
			
			trial_err += err;
			if ((uint64_t)(trial_err * pComp_params->m_mode1_error_weight + .5f) > best_err)
				break;

		} // subset

		const uint64_t mode1_trial_err = (uint64_t)(trial_err * pComp_params->m_mode1_error_weight + .5f);
		if (mode1_trial_err < best_err)
		{
			best_err = mode1_trial_err;
			opt_results.m_mode = 1;
			opt_results.m_partition = trial_partition;
			for (uint32_t subset = 0; subset < 2; subset++)
			{
				for (uint32_t i = 0; i < subset_total_colors1[subset]; i++)
					opt_results.m_selectors[subset_pixel_index1[subset][i]] = subset_selectors1[subset][i];
				opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
				opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;
				opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
			}
		}
	}

	encode_bc7_block(pBlock, &opt_results);
}

bool bc7enc_compress_block(void *pBlock, const void *pPixelsRGBA, const bc7enc_compress_block_params *pComp_params)
{
	assert(g_bc7_mode_1_optimal_endpoints[255][0].m_hi != 0);

	const color_rgba *pPixels = (const color_rgba *)(pPixelsRGBA);

	color_cell_compressor_params params;
	if (pComp_params->m_perceptual)
	{
		// https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
		const float pr_weight = (.5f / (1.0f - .2126f)) * (.5f / (1.0f - .2126f));
		const float pb_weight = (.5f / (1.0f - .0722f)) * (.5f / (1.0f - .0722f));
		params.m_weights[0] = (int)(pComp_params->m_weights[0] * 4.0f);
		params.m_weights[1] = (int)(pComp_params->m_weights[1] * 4.0f * pr_weight);
		params.m_weights[2] = (int)(pComp_params->m_weights[2] * 4.0f * pb_weight);
		params.m_weights[3] = pComp_params->m_weights[3] * 4;
	}
	else
		memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));
	
	if (pComp_params->m_force_alpha)
	{
		handle_alpha_block(pBlock, pPixels, pComp_params, &params);
		return true;
	}

	for (uint32_t i = 0; i < 16; i++)
	{
		if (pPixels[i].m_c[3] < 255)
		{
			handle_alpha_block(pBlock, pPixels, pComp_params, &params);
			return true;
		}
	}
	handle_opaque_block(pBlock, pPixels, pComp_params, &params);
	return false;
}

static const uint8_t g_tdefl_small_dist_extra[512] =
{
	0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
	7, 7, 7, 7, 7, 7, 7, 7
};

static const uint8_t g_tdefl_large_dist_extra[128] =
{
	0, 0, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
	12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
	13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
};

static inline uint32_t compute_match_cost_estimate(uint32_t dist, uint32_t match_len_in_bytes)
{
	assert(match_len_in_bytes <= 258);

	uint32_t len_cost = 6;
	if (match_len_in_bytes >= 12)
		len_cost = 9;
	else if (match_len_in_bytes >= 8)
		len_cost = 8;
	else if (match_len_in_bytes >= 6)
		len_cost = 7;

	uint32_t dist_cost = 5;
	if (dist < 512)
		dist_cost += g_tdefl_small_dist_extra[dist & 511];
	else
	{
		dist_cost += g_tdefl_large_dist_extra[std::min<uint32_t>(dist, 32767) >> 8];
		while (dist >= 32768)
		{
			dist_cost++;
			dist >>= 1;
		}
	}
	return len_cost + dist_cost;
}

class tracked_stat
{
public:
	tracked_stat() { clear(); }

	void clear() { m_num = 0; m_total = 0; m_total2 = 0; }

	void update(uint32_t val) { m_num++; m_total += val; m_total2 += val * val; }

	tracked_stat& operator += (uint32_t val) { update(val); return *this; }

	uint32_t get_number_of_values() { return m_num; }
	uint64_t get_total() const { return m_total; }
	uint64_t get_total2() const { return m_total2; }

	float get_average() const { return m_num ? (float)m_total / m_num : 0.0f; };
	float get_std_dev() const { return m_num ? sqrtf((float)(m_num * m_total2 - m_total * m_total)) / m_num : 0.0f; }
	float get_variance() const { float s = get_std_dev(); return s * s; }

private:
	uint32_t m_num;
	uint64_t m_total;
	uint64_t m_total2;
};

static inline float compute_block_max_std_dev(const color_rgba* pPixels)
{
	tracked_stat r_stats, g_stats, b_stats, a_stats;

	for (uint32_t i = 0; i < 16; i++)
	{
		r_stats.update(pPixels[i].m_c[0]);
		g_stats.update(pPixels[i].m_c[1]);
		b_stats.update(pPixels[i].m_c[2]);
		a_stats.update(pPixels[i].m_c[3]);
	}

	return std::max<float>(std::max<float>(std::max(r_stats.get_std_dev(), g_stats.get_std_dev()), b_stats.get_std_dev()), a_stats.get_std_dev());
}

struct bc7_block
{
	uint8_t m_bytes[16];

	uint32_t get_mode() const
	{
		uint32_t bc7_mode = 0;
		while (((m_bytes[0] & (1 << bc7_mode)) == 0) && (bc7_mode < 8))
			bc7_mode++;
		return bc7_mode;
	}
};

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
If you use this software in a product, attribution / credits is requested but not required.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright(c) 2020-2021 Richard Geldreich, Jr.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions :
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain(www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non - commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain.We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors.We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/
