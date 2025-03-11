// File: basisu_astc_hdr_common.cpp
#include "basisu_enc.h"
#include "basisu_gpu_texture.h"
#include "../transcoder/basisu_astc_helpers.h"
#include "../transcoder/basisu_astc_hdr_core.h"
#include "basisu_astc_hdr_common.h"

using namespace basist;

#ifndef __EMSCRIPTEN__
	#define BASISU_MULTITHREADED_INIT (0)
#endif

namespace basisu
{

const uint8_t g_ise_weight_lerps[MAX_SUPPORTED_ISE_WEIGHT_INDEX + 1][33] =
{
	{ 2, 0, 64 }, // 0, note ise range=0 is invalid for 4x4 block sizes (<24 weight bits in the block)
	{ 3, 0, 32, 64 }, // 1
	{ 4, 0, 21, 43, 64 }, // 2
	{ 5, 0, 16, 32, 48, 64 }, // 3
	{ 6, 0, 64, 12, 52, 25, 39 }, // 4
	{ 8, 0, 9, 18, 27, 37, 46, 55, 64 }, // 5
	{ 10, 0, 64, 7, 57, 14, 50, 21, 43, 28, 36 }, // 6
	{ 12, 0, 64, 17, 47, 5, 59, 23, 41, 11, 53, 28, 36 }, // 7
	{ 16, 0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64 }, // 8
	{ 20, 0,64,16,48,3,61,19,45,6,58,23,41,9,55,26,38,13,51,29,35}, // 9
	{ 24, 0,64,8,56,16,48,24,40,2,62,11,53,19,45,27,37,5,59,13,51,22,42,30,34}, // 10
	{ 32, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64}, // 11
};

//--------------------------------------------------------------------------------------------------------------------------

const float DEF_R_ERROR_SCALE = 2.0f;
const float DEF_G_ERROR_SCALE = 3.0f;

void astc_hdr_codec_base_options::init()
{
	m_r_err_scale = DEF_R_ERROR_SCALE;
	m_g_err_scale = DEF_G_ERROR_SCALE;
	m_q_log_bias = Q_LOG_BIAS_4x4;

	m_ultra_quant = false;

	// Disabling by default to avoid transcoding outliers (try kodim26). The quality lost is very low. TODO: Could include the uber result in the output.
	m_allow_uber_mode = false;

	m_mode7_full_s_optimization = true;

	m_take_first_non_clamping_mode11_submode = false;
	m_take_first_non_clamping_mode7_submode = false;

	m_disable_weight_plane_optimization = true;
}

//--------------------------------------------------------------------------------------------------------------------------
// max usable qlog8 value is 247, 248=inf, >=249 is nan
// max usable qlog7 value is 123, 124=inf, >=125 is nan

//const uint32_t TOTAL_USABLE_QLOG8 = 248; // 0-247 are usable, 0=0, 247=60416.0, 246=55296.0

// nearest values given a positive half float value (only)
static uint16_t g_half_to_qlog7[32768], g_half_to_qlog8[32768];

const uint32_t HALF_TO_QLOG_TABS_MIN_BITS = 7;
const uint32_t HALF_TO_QLOG_TABS_MAX_BITS = 8;
static uint16_t* g_pHalf_to_qlog_tabs[2] =
{
	g_half_to_qlog7,
	g_half_to_qlog8,
};

#if 0
static inline uint32_t half_to_qlog7_8(half_float h, uint32_t bits)
{
	assert((bits >= HALF_TO_QLOG_TABS_MIN_BITS) && (bits <= HALF_TO_QLOG_TABS_MAX_BITS));
	assert(h < 32768);

	return g_pHalf_to_qlog_tabs[bits - HALF_TO_QLOG_TABS_MIN_BITS][h];
}
#endif

// TODO: Tune this
static inline uint32_t quant_qlog16(uint32_t q16, uint32_t desired_bits)
{
	assert((desired_bits >= 7) && (desired_bits <= 12));
	assert(q16 <= 65535);

	const uint32_t shift = 16 - desired_bits;
	uint32_t e = (q16 + (1U << (shift - 1U)) - 1U) >> shift;

	uint32_t max_val = (1U << desired_bits) - 1U;
	e = minimum<uint32_t>(e, max_val);

	return e;
}

static void compute_half_to_qlog_table(uint32_t bits, uint16_t* pTable, const basisu::vector<float>& qlog16_to_float)
{
	assert(bits >= 5 && bits <= 12);
	const uint32_t max_val = (1 << bits) - 1;

	const uint32_t FIRST_INVALID_QLOG16_INDEX = 63488; // first inf, rest are inf/nan's
	assert(std::isinf(qlog16_to_float[FIRST_INVALID_QLOG16_INDEX]));
	assert(std::isinf(qlog16_to_float[FIRST_INVALID_QLOG16_INDEX + 1]));
	assert(!std::isnan(qlog16_to_float[FIRST_INVALID_QLOG16_INDEX - 1]));
	assert(!std::isinf(qlog16_to_float[FIRST_INVALID_QLOG16_INDEX - 1]));

	// For all positive half-floats
	for (uint32_t h = 0; h < 32768; h++)
	{
		// Skip invalid values
		if (is_half_inf_or_nan((half_float)h))
			continue;
		const float desired_val = half_to_float((half_float)h);

		float best_err = BIG_FLOAT_VAL;
		uint32_t best_qlog = 0;
		
		double prev_err = BIG_FLOAT_VAL;

		// For all possible qlog's
		for (uint32_t i = 0; i <= max_val; i++)
		{
			// Skip invalid values
			uint32_t idx = i << (16 - bits);
			if (idx >= FIRST_INVALID_QLOG16_INDEX)
				break;

			float v = qlog16_to_float[idx];
			//assert(!std::isinf(v) && !std::isnan(v)); // too clostly in debug

			// Compute error
			float err = fabsf(v - desired_val);

			if (err > prev_err)
			{
				// Every remaining entry will have guaranteed higher error
				break;
			}

			prev_err = err;
						
			// Find best
			if (err < best_err)
			{
				best_err = err;
				best_qlog = i;
				
				if (best_err == 0.0f)
					break;
			}
		}

		pTable[h] = (uint16_t)best_qlog;
	}
}

static void init_qlog_tables()
{
	basisu::vector<float> qlog16_to_float(65536);

	// for all possible qlog16, compute the corresponding half float
	for (uint32_t i = 0; i <= 65535; i++)
	{
		half_float h = astc_helpers::qlog16_to_half(i);

		qlog16_to_float[i] = half_to_float(h);
	}

#if BASISU_MULTITHREADED_INIT
	job_pool jp(3);
	
	for (uint32_t bits = HALF_TO_QLOG_TABS_MIN_BITS; bits <= HALF_TO_QLOG_TABS_MAX_BITS; bits++)
	{
		jp.add_job( [bits, &qlog16_to_float]() { compute_half_to_qlog_table(bits, g_pHalf_to_qlog_tabs[bits - HALF_TO_QLOG_TABS_MIN_BITS], qlog16_to_float); });
	}

	jp.wait_for_all();
#else
	// for all possible half floats, find the nearest qlog5-12 float
	for (uint32_t bits = HALF_TO_QLOG_TABS_MIN_BITS; bits <= HALF_TO_QLOG_TABS_MAX_BITS; bits++)
	{
		compute_half_to_qlog_table(bits, g_pHalf_to_qlog_tabs[bits - HALF_TO_QLOG_TABS_MIN_BITS], qlog16_to_float);

#if 0
		std::vector<uint16_t> check_tab(32768);
		compute_half_to_qlog_table_orig(bits, check_tab.data(), qlog16_to_float);
		for (uint32_t i = 0; i < (1 << bits); i++)
		{
			assert(check_tab[i] == g_pHalf_to_qlog_tabs[bits - HALF_TO_QLOG_TABS_MIN_BITS][i]);
		}
#endif
	}
#endif // BASISU_MULTITHREADED_INIT
}

//--------------------------------------------------------------------------------------------------------------------------

static vec3F calc_mean(uint32_t num_pixels, const vec4F* pPixels)
{
	vec3F mean(0.0f);

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		const vec4F& p = pPixels[i];

		mean[0] += p[0];
		mean[1] += p[1];
		mean[2] += p[2];
	}

	return mean / static_cast<float>(num_pixels);
}

static vec3F calc_rgb_pca(uint32_t num_pixels, const vec4F* pPixels, const vec3F& mean_color)
{
	float cov[6] = { 0, 0, 0, 0, 0, 0 };

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		const vec4F& v = pPixels[i];

		float r = v[0] - mean_color[0];
		float g = v[1] - mean_color[1];
		float b = v[2] - mean_color[2];

		cov[0] += r * r;
		cov[1] += r * g;
		cov[2] += r * b;
		cov[3] += g * g;
		cov[4] += g * b;
		cov[5] += b * b;
	}

	float xr = .9f, xg = 1.0f, xb = .7f;
	for (uint32_t iter = 0; iter < 3; iter++)
	{
		float r = xr * cov[0] + xg * cov[1] + xb * cov[2];
		float g = xr * cov[1] + xg * cov[3] + xb * cov[4];
		float b = xr * cov[2] + xg * cov[4] + xb * cov[5];

		float m = maximumf(maximumf(fabsf(r), fabsf(g)), fabsf(b));

		if (m > 1e-10f)
		{
			m = 1.0f / m;

			r *= m;
			g *= m;
			b *= m;
		}

		xr = r;
		xg = g;
		xb = b;
	}

	float len = xr * xr + xg * xg + xb * xb;

	vec3F axis(0.5773502691f);

	if (len >= 1e-10f)
	{
		len = 1.0f / sqrtf(len);

		xr *= len;
		xg *= len;
		xb *= len;

		axis.set(xr, xg, xb);
	}

	return axis;
}

void encode_astc_block_stats::init(uint32_t num_pixels, const vec4F pBlock_pixels_q16[])
{
	m_num_pixels = num_pixels;
	m_mean_q16 = calc_mean(num_pixels, pBlock_pixels_q16);
	m_axis_q16 = calc_rgb_pca(num_pixels, pBlock_pixels_q16, m_mean_q16);
}

static vec3F interp_color(const vec3F& mean, const vec3F& dir, float df, const aabb3F& colorspace_box, const aabb3F& input_box, bool* pInside = nullptr)
{
#if 0
	assert(mean[0] >= input_box[0][0]);
	assert(mean[1] >= input_box[0][1]);
	assert(mean[2] >= input_box[0][2]);
	assert(mean[0] <= input_box[1][0]);
	assert(mean[1] <= input_box[1][1]);
	assert(mean[2] <= input_box[1][2]);
#endif

	if (pInside)
		*pInside = false;

	vec3F k(mean + dir * df);
	if (colorspace_box.contains(k))
	{
		if (pInside)
			*pInside = true;

		return k;
	}

	// starts inside
	vec3F s(mean);

	// ends outside
	vec3F e(mean + dir * df);

	// a ray guaranteed to go from the outside to inside
	ray3F r(e, (s - e).normalize_in_place());
	vec3F c;
	float t = 0.0f;

	intersection::result res = intersection::ray_aabb(c, t, r, input_box);
	if (res != intersection::cSuccess)
		c = k;

	return c;
}

// all in Q16 space, 0-65535
static bool compute_least_squares_endpoints_rgb(
	uint32_t N, const uint8_t* pSelectors, const vec4F* pSelector_weights,
	vec3F* pXl, vec3F* pXh, const vec4F* pColors, const aabb3F& input_box)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// https://web.archive.org/web/20150319232457/http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;

	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];
		
		z00 += pSelector_weights[sel][0];
		z10 += pSelector_weights[sel][1];
		z11 += pSelector_weights[sel][2];

		float w = pSelector_weights[sel][3];

		q00_r += w * pColors[i][0];
		t_r += pColors[i][0];

		q00_g += w * pColors[i][1];
		t_g += pColors[i][1];

		q00_b += w * pColors[i][2];
		t_b += pColors[i][2];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det == 0.0f)
		return false;

	det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;
		
	(*pXl)[0] = (float)(iz00 * q00_r + iz01 * q10_r);
	(*pXh)[0] = (float)(iz10 * q00_r + iz11 * q10_r);

	(*pXl)[1] = (float)(iz00 * q00_g + iz01 * q10_g);
	(*pXh)[1] = (float)(iz10 * q00_g + iz11 * q10_g);

	(*pXl)[2] = (float)(iz00 * q00_b + iz01 * q10_b);
	(*pXh)[2] = (float)(iz10 * q00_b + iz11 * q10_b);

	for (uint32_t c = 0; c < 3; c++)
	{
		float l = (*pXl)[c], h = (*pXh)[c];

		if (input_box.get_dim(c) < .0000125f)
		{
			l = input_box[0][c];
			h = input_box[1][c];
		}
		
		(*pXl)[c] = l;
		(*pXh)[c] = h;
	}

	vec3F mean((*pXl + *pXh) * .5f);
	vec3F dir(*pXh - *pXl);

	float ln = dir.length();
	if (ln)
	{
		dir /= ln;

		float ld = (*pXl - mean).dot(dir);
		float hd = (*pXh - mean).dot(dir);

		aabb3F colorspace_box(vec3F(0.0f), vec3F(MAX_QLOG16_VAL));

		bool was_inside1 = false;

		vec3F l = interp_color(mean, dir, ld, colorspace_box, input_box, &was_inside1);
		if (!was_inside1)
			*pXl = l;

		bool was_inside2 = false;
		vec3F h = interp_color(mean, dir, hd, colorspace_box, input_box, &was_inside2);
		if (!was_inside2)
			*pXh = h;
	}

	pXl->clamp(0.0f, MAX_QLOG16_VAL);
	pXh->clamp(0.0f, MAX_QLOG16_VAL);

	return true;
}

static bool compute_least_squares_endpoints_rgb_raw_weights(
	uint32_t N, const uint8_t* pRaw_weights, 
	vec3F* pXl, vec3F* pXh, const vec4F* pColors, const aabb3F& input_box)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// https://web.archive.org/web/20150319232457/http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
		
	for (uint32_t i = 0; i < N; i++)
	{
		const float wt = (float)pRaw_weights[i] * (1.0f / 64.0f);
		assert(wt <= 1.0f);

		const float w0 = wt * wt;
		const float w1 = (1.0f - wt) * wt;
		const float w2 = (1.0f - wt) * (1.0f - wt);
		const float w3 = wt;

		z00 += w0;
		z10 += w1;
		z11 += w2;

		float w = w3;
		q00_r += w * pColors[i][0];
		t_r += pColors[i][0];

		q00_g += w * pColors[i][1];
		t_g += pColors[i][1];

		q00_b += w * pColors[i][2];
		t_b += pColors[i][2];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det == 0.0f)
		return false;

	det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	(*pXl)[0] = (float)(iz00 * q00_r + iz01 * q10_r);
	(*pXh)[0] = (float)(iz10 * q00_r + iz11 * q10_r);

	(*pXl)[1] = (float)(iz00 * q00_g + iz01 * q10_g);
	(*pXh)[1] = (float)(iz10 * q00_g + iz11 * q10_g);

	(*pXl)[2] = (float)(iz00 * q00_b + iz01 * q10_b);
	(*pXh)[2] = (float)(iz10 * q00_b + iz11 * q10_b);

	for (uint32_t c = 0; c < 3; c++)
	{
		float l = (*pXl)[c], h = (*pXh)[c];

		if (input_box.get_dim(c) < .0000125f)
		{
			l = input_box[0][c];
			h = input_box[1][c];
		}

		(*pXl)[c] = l;
		(*pXh)[c] = h;
	}

	vec3F mean((*pXl + *pXh) * .5f);
	vec3F dir(*pXh - *pXl);

	float ln = dir.length();
	if (ln)
	{
		dir /= ln;

		float ld = (*pXl - mean).dot(dir);
		float hd = (*pXh - mean).dot(dir);

		aabb3F colorspace_box(vec3F(0.0f), vec3F(MAX_QLOG16_VAL));

		bool was_inside1 = false;

		vec3F l = interp_color(mean, dir, ld, colorspace_box, input_box, &was_inside1);
		if (!was_inside1)
			*pXl = l;

		bool was_inside2 = false;
		vec3F h = interp_color(mean, dir, hd, colorspace_box, input_box, &was_inside2);
		if (!was_inside2)
			*pXh = h;
	}

	pXl->clamp(0.0f, MAX_QLOG16_VAL);
	pXh->clamp(0.0f, MAX_QLOG16_VAL);

	return true;
}

static bool compute_least_squares_endpoints_2D(
	uint32_t N, const uint8_t* pSelectors, const vec4F* pSelector_weights,
	vec2F* pXl, vec2F* pXh, const vec2F* pColors, const aabb2F& input_box)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// https://web.archive.org/web/20150319232457/http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	
	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];
		z00 += pSelector_weights[sel][0];
		z10 += pSelector_weights[sel][1];
		z11 += pSelector_weights[sel][2];

		float w = pSelector_weights[sel][3];
		q00_r += w * pColors[i][0];
		t_r += pColors[i][0];

		q00_g += w * pColors[i][1];
		t_g += pColors[i][1];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det == 0.0f)
		return false;

	det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	(*pXl)[0] = (float)(iz00 * q00_r + iz01 * q10_r);
	(*pXh)[0] = (float)(iz10 * q00_r + iz11 * q10_r);

	(*pXl)[1] = (float)(iz00 * q00_g + iz01 * q10_g);
	(*pXh)[1] = (float)(iz10 * q00_g + iz11 * q10_g);

	for (uint32_t c = 0; c < 2; c++)
	{
		float l = (*pXl)[c], h = (*pXh)[c];

		if (input_box.get_dim(c) < .0000125f)
		{
			l = input_box[0][c];
			h = input_box[1][c];
		}

		(*pXl)[c] = l;
		(*pXh)[c] = h;
	}
		
	pXl->clamp(0.0f, MAX_QLOG16_VAL);
	pXh->clamp(0.0f, MAX_QLOG16_VAL);

	return true;
}

static bool compute_least_squares_endpoints_1D(
	uint32_t N, const uint8_t* pSelectors, const vec4F* pSelector_weights,
	vec1F* pXl, vec1F* pXh, const vec1F* pColors, const aabb1F& input_box)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// https://web.archive.org/web/20150319232457/http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;

	for (uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];
		z00 += pSelector_weights[sel][0];
		z10 += pSelector_weights[sel][1];
		z11 += pSelector_weights[sel][2];

		float w = pSelector_weights[sel][3];
		q00_r += w * pColors[i][0];
		t_r += pColors[i][0];
	}

	q10_r = t_r - q00_r;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det == 0.0f)
		return false;

	det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	(*pXl)[0] = (float)(iz00 * q00_r + iz01 * q10_r);
	(*pXh)[0] = (float)(iz10 * q00_r + iz11 * q10_r);

	for (uint32_t c = 0; c < 1; c++)
	{
		float l = (*pXl)[c], h = (*pXh)[c];

		if (input_box.get_dim(c) < .0000125f)
		{
			l = input_box[0][c];
			h = input_box[1][c];
		}

		(*pXl)[c] = l;
		(*pXh)[c] = h;
	}

	pXl->clamp(0.0f, MAX_QLOG16_VAL);
	pXh->clamp(0.0f, MAX_QLOG16_VAL);

	return true;
}

static bool compute_weighted_least_squares_endpoints_rgb(
	uint32_t N, 
	const uint8_t* pSelectors, const vec4F* pSelector_weights, const float* pRaw_weights, /* ti */
	const float* pEmphasis_weights /* wi */,
	vec3F* pXl, vec3F* pXh, 
	const vec4F* pColors, /* pi */
	const aabb3F& input_box)
{
	(void)input_box;

	assert(N);
	assert((pSelectors && pSelector_weights) || pRaw_weights);
	assert(pEmphasis_weights);

	// Pi = pixel colors
	// Ti = project weights, [0,1]
	// Wi = emphasis weights

	float total_wi = 0.0f;
	for (uint32_t i = 0; i < N; i++)
		total_wi += pEmphasis_weights[i];

	if (total_wi == 0.0f)
		return false;

	float weighted_mean_tw = 0.0f;
	float weighted_mean_pw[3] = { 0.0f };

	for (uint32_t i = 0; i < N; i++)
	{
		const float wi = pEmphasis_weights[i];
		const float ti = pSelectors ? pSelector_weights[pSelectors[i]][3] : pRaw_weights[i];
		const float pi_r = pColors[i][0], pi_g = pColors[i][1], pi_b = pColors[i][2];

		weighted_mean_tw += wi * ti;
		
		weighted_mean_pw[0] += wi * pi_r;
		weighted_mean_pw[1] += wi * pi_g;
		weighted_mean_pw[2] += wi * pi_b;
	}

	weighted_mean_tw /= total_wi;

	weighted_mean_pw[0] /= total_wi;
	weighted_mean_pw[1] /= total_wi;
	weighted_mean_pw[2] /= total_wi;

	float spt[3] = { 0.0f };
	float stt = 0.0f;

	for (uint32_t i = 0; i < N; i++)
	{
		const float wi = pEmphasis_weights[i];
		const float ti = pSelectors ? pSelector_weights[pSelectors[i]][3] : pRaw_weights[i];
		const float pi_r = pColors[i][0], pi_g = pColors[i][1], pi_b = pColors[i][2];
		
		spt[0] += wi * (pi_r - weighted_mean_pw[0]) * (ti - weighted_mean_tw);
		spt[1] += wi * (pi_g - weighted_mean_pw[1]) * (ti - weighted_mean_tw);
		spt[2] += wi * (pi_b - weighted_mean_pw[2]) * (ti - weighted_mean_tw);

		stt += wi * square(ti - weighted_mean_tw);
	}

	if (stt == 0.0f)
		return false;

	for (uint32_t i = 0; i < 3; i++)
	{
		float h = weighted_mean_pw[i] + (spt[i] / stt) * (1.0f - weighted_mean_tw);
		float l = weighted_mean_pw[i] - (spt[i] / stt) * weighted_mean_tw;
				
		(*pXh)[i] = h;
		(*pXl)[i] = l;
	}

	pXl->clamp(0.0f, MAX_QLOG16_VAL);
	pXh->clamp(0.0f, MAX_QLOG16_VAL);

	return true;
}

static vec4F g_astc_ls_weights_ise[MAX_SUPPORTED_ISE_WEIGHT_INDEX + 1][MAX_SUPPORTED_WEIGHT_LEVELS];

static uint8_t g_map_astc_to_linear_order[MAX_SUPPORTED_ISE_WEIGHT_INDEX + 1][MAX_SUPPORTED_WEIGHT_LEVELS]; // [ise_range][astc_index] -> linear index
static uint8_t g_map_linear_to_astc_order[MAX_SUPPORTED_ISE_WEIGHT_INDEX + 1][MAX_SUPPORTED_WEIGHT_LEVELS]; // [ise_range][linear_index] -> astc_index

static void encode_astc_hdr_init()
{
	// Precomputed weight constants used during least fit determination. For each entry: w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w
	for (uint32_t range = MIN_SUPPORTED_ISE_WEIGHT_INDEX; range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX; range++)
	{
		const uint32_t num_levels = g_ise_weight_lerps[range][0];
		assert(num_levels == astc_helpers::get_ise_levels(range));
		assert((num_levels >= MIN_SUPPORTED_WEIGHT_LEVELS) && (num_levels <= MAX_SUPPORTED_WEIGHT_LEVELS));

		for (uint32_t i = 0; i < num_levels; i++)
		{
			float w = g_ise_weight_lerps[range][1 + i] * (1.0f / 64.0f);

			g_astc_ls_weights_ise[range][i].set(w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w);
		}
	}

	for (uint32_t ise_range = MIN_SUPPORTED_ISE_WEIGHT_INDEX; ise_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX; ise_range++)
	{
		const uint32_t num_levels = g_ise_weight_lerps[ise_range][0];
		assert((num_levels >= MIN_SUPPORTED_WEIGHT_LEVELS) && (num_levels <= MAX_SUPPORTED_WEIGHT_LEVELS));

		uint32_t s[MAX_SUPPORTED_WEIGHT_LEVELS];
		for (uint32_t i = 0; i < num_levels; i++)
			s[i] = (g_ise_weight_lerps[ise_range][1 + i] << 8) + i;

		std::sort(s, s + num_levels);

		for (uint32_t i = 0; i < num_levels; i++)
			g_map_linear_to_astc_order[ise_range][i] = (uint8_t)(s[i] & 0xFF);

		for (uint32_t i = 0; i < num_levels; i++)
			g_map_astc_to_linear_order[ise_range][g_map_linear_to_astc_order[ise_range][i]] = (uint8_t)i;
	}

	//init_quantize_tables();
}

bool g_astc_hdr_enc_initialized;

void astc_hdr_enc_init()
{
	if (g_astc_hdr_enc_initialized)
		return;

	astc_hdr_core_init();

	astc_helpers::init_tables(true);

	init_qlog_tables();

	encode_astc_hdr_init();

	g_astc_hdr_enc_initialized = true;
}

void interpolate_qlog12_colors(
	const int e[2][3],
	half_float* pDecoded_half,
	vec3F* pDecoded_float,
	uint32_t n, uint32_t ise_weight_range)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));

	for (uint32_t i = 0; i < 2; i++)
	{
		for (uint32_t j = 0; j < 3; j++)
		{
			assert(in_range(e[i][j], 0, 0xFFF));
		}
	}

	for (uint32_t i = 0; i < n; i++)
	{
		const int c = g_ise_weight_lerps[ise_weight_range][1 + i];
		assert(c == (int)astc_helpers::dequant_bise_weight(i, ise_weight_range));

		half_float rf, gf, bf;

		{
			uint32_t r0 = e[0][0] << 4;
			uint32_t r1 = e[1][0] << 4;
			int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
			rf = astc_helpers::qlog16_to_half(ri);
		}

		{
			uint32_t g0 = e[0][1] << 4;
			uint32_t g1 = e[1][1] << 4;
			int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
			gf = astc_helpers::qlog16_to_half(gi);
		}

		{
			uint32_t b0 = e[0][2] << 4;
			uint32_t b1 = e[1][2] << 4;
			int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
			bf = astc_helpers::qlog16_to_half(bi);
		}

		if (pDecoded_half)
		{
			pDecoded_half[i * 3 + 0] = rf;
			pDecoded_half[i * 3 + 1] = gf;
			pDecoded_half[i * 3 + 2] = bf;
		}

		if (pDecoded_float)
		{
			pDecoded_float[i][0] = half_to_float(rf);
			pDecoded_float[i][1] = half_to_float(gf);
			pDecoded_float[i][2] = half_to_float(bf);
		}
	}
}

// decoded in ASTC order, not linear order
// return false if the ISE endpoint quantization leads to non-valid endpoints being decoded
bool get_astc_hdr_mode_11_block_colors(
	const uint8_t* pEndpoints,
	half_float* pDecoded_half,
	vec3F* pDecoded_float,
	uint32_t n, uint32_t ise_weight_range, uint32_t ise_endpoint_range)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));

	int e[2][3];
	if (!decode_mode11_to_qlog12(pEndpoints, e, ise_endpoint_range))
		return false;

	interpolate_qlog12_colors(e, pDecoded_half, pDecoded_float, n, ise_weight_range);

	return true;
}

// decoded in ASTC order, not linear order
// return false if the ISE endpoint quantization leads to non-valid endpoints being decoded
bool get_astc_hdr_mode_7_block_colors(
	const uint8_t* pEndpoints,
	half_float* pDecoded_half,
	vec3F* pDecoded_float,
	uint32_t n, uint32_t ise_weight_range, uint32_t ise_endpoint_range)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));

	int e[2][3];
	if (!decode_mode7_to_qlog12(pEndpoints, e, nullptr, ise_endpoint_range))
		return false;

	interpolate_qlog12_colors(e, pDecoded_half, pDecoded_float, n, ise_weight_range);

	return true;
}

double eval_selectors_f(
	uint32_t num_pixels,
	uint8_t* pWeights,
	const half_float* pBlock_pixels_half,
	uint32_t num_weight_levels,
	const half_float* pDecoded_half,
	const astc_hdr_codec_base_options& coptions,
	uint32_t usable_selector_bitmask)
{
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert(usable_selector_bitmask);

	const float R_WEIGHT = coptions.m_r_err_scale;
	const float G_WEIGHT = coptions.m_g_err_scale;

	double total_error = 0;

#ifdef _DEBUG
	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 0]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 1]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 2]));
	}
#endif

	double decoded_half_q[MAX_SUPPORTED_WEIGHT_LEVELS][3];

	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		const half_float* p = &pDecoded_half[i * 3];

		decoded_half_q[i][0] = q(p[0], coptions.m_q_log_bias);
		decoded_half_q[i][1] = q(p[1], coptions.m_q_log_bias);
		decoded_half_q[i][2] = q(p[2], coptions.m_q_log_bias);
	}

	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

		const double desired_half_r_q = q(pDesired_half[0], coptions.m_q_log_bias);
		const double desired_half_g_q = q(pDesired_half[1], coptions.m_q_log_bias);
		const double desired_half_b_q = q(pDesired_half[2], coptions.m_q_log_bias);

		double lowest_e = BIG_FLOAT_VAL;

		//double dists[MAX_SUPPORTED_WEIGHT_LEVELS];

		// this is an approximation of MSLE
		for (uint32_t i = 0; i < num_weight_levels; i++)
		{
			if (((1 << i) & usable_selector_bitmask) == 0)
				continue;

			// compute piecewise linear approximation of log2(a+eps)-log2(b+eps), for each component, then MSLE
			double rd = decoded_half_q[i][0] - desired_half_r_q;
			double gd = decoded_half_q[i][1] - desired_half_g_q;
			double bd = decoded_half_q[i][2] - desired_half_b_q;

			double e = R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;

			//dists[i] = e;

			if (e < lowest_e)
			{
				lowest_e = e;
				pWeights[p] = (uint8_t)i;
			}
		}

		total_error += lowest_e;

	} // p

	return total_error;
}

double eval_selectors(
	uint32_t num_pixels,
	uint8_t* pWeights,
	uint32_t ise_weight_range,
	const half_float* pBlock_pixels_half,
	uint32_t num_weight_levels,
	const half_float* pDecoded_half,
	const astc_hdr_codec_base_options& coptions,
	uint32_t usable_selector_bitmask)
{
	if ((coptions.m_r_err_scale != 2.0f) || (coptions.m_g_err_scale != 3.0f))
	{
		return eval_selectors_f(
			num_pixels,
			pWeights,
			pBlock_pixels_half,
			num_weight_levels,
			pDecoded_half,
			coptions,
			usable_selector_bitmask);
	}

	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert(usable_selector_bitmask);

	uint64_t total_error = 0;

#ifdef _DEBUG
	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 0]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 1]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 2]));
	}
#endif

	int64_t decoded_half_q[MAX_SUPPORTED_WEIGHT_LEVELS][3];

	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		const half_float* p = &pDecoded_half[i * 3];

		decoded_half_q[i][0] = q2(p[0], coptions.m_q_log_bias);
		decoded_half_q[i][1] = q2(p[1], coptions.m_q_log_bias);
		decoded_half_q[i][2] = q2(p[2], coptions.m_q_log_bias);
	}

	if (usable_selector_bitmask != UINT32_MAX)
	{
		for (uint32_t p = 0; p < num_pixels; p++)
		{
			const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

			const int64_t desired_half_r_q = q2(pDesired_half[0], coptions.m_q_log_bias);
			const int64_t desired_half_g_q = q2(pDesired_half[1], coptions.m_q_log_bias);
			const int64_t desired_half_b_q = q2(pDesired_half[2], coptions.m_q_log_bias);

			int64_t lowest_e = INT64_MAX;

			for (uint32_t i = 0; i < num_weight_levels; i++)
			{
				if (((1 << i) & usable_selector_bitmask) == 0)
					continue;

				int64_t rd = decoded_half_q[i][0] - desired_half_r_q;
				int64_t gd = decoded_half_q[i][1] - desired_half_g_q;
				int64_t bd = decoded_half_q[i][2] - desired_half_b_q;

				int64_t e = 2 * (rd * rd) + 3 * (gd * gd) + bd * bd;

				if (e < lowest_e)
				{
					lowest_e = e;
					pWeights[p] = (uint8_t)i;
				}
			}

			total_error += lowest_e;

		} // p
	}
	else
	{
		if ((num_weight_levels <= 4) || (coptions.m_disable_weight_plane_optimization))
		{
			for (uint32_t p = 0; p < num_pixels; p++)
			{
				const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

				const half_float desired_r = pDesired_half[0], desired_g = pDesired_half[1], desired_b = pDesired_half[2];

				const int64_t desired_half_r_q = q2(desired_r, coptions.m_q_log_bias);
				const int64_t desired_half_g_q = q2(desired_g, coptions.m_q_log_bias);
				const int64_t desired_half_b_q = q2(desired_b, coptions.m_q_log_bias);

				int64_t lowest_e = INT64_MAX;

				uint32_t i;
				for (i = 0; (i + 1) < num_weight_levels; i += 2)
				{
					int64_t e0, e1;

					{
						int64_t rd0 = decoded_half_q[i][0] - desired_half_r_q; // 27 bits maximum with half float inputs
						int64_t gd0 = decoded_half_q[i][1] - desired_half_g_q;
						int64_t bd0 = decoded_half_q[i][2] - desired_half_b_q;
						e0 = ((2 * (rd0 * rd0) + 3 * (gd0 * gd0) + bd0 * bd0) << 5) | i; // max 62 bits (27*2+3+5)
					}

					{
						int64_t rd1 = decoded_half_q[i + 1][0] - desired_half_r_q;
						int64_t gd1 = decoded_half_q[i + 1][1] - desired_half_g_q;
						int64_t bd1 = decoded_half_q[i + 1][2] - desired_half_b_q;
						e1 = ((2 * (rd1 * rd1) + 3 * (gd1 * gd1) + bd1 * bd1) << 5) | (i + 1);
					}

					lowest_e = minimum(lowest_e, e0, e1);
				}

				if (i != num_weight_levels)
				{
					int64_t rd0 = decoded_half_q[i][0] - desired_half_r_q;
					int64_t gd0 = decoded_half_q[i][1] - desired_half_g_q;
					int64_t bd0 = decoded_half_q[i][2] - desired_half_b_q;
					int64_t e0 = ((2 * (rd0 * rd0) + 3 * (gd0 * gd0) + bd0 * bd0) << 5) | i;

					lowest_e = minimum(lowest_e, e0);
				}

				pWeights[p] = (uint8_t)(lowest_e & 31);

				total_error += (lowest_e >> 5);

			} // p
		}
		else
		{
			const auto& weight_val_to_ise_tab = astc_helpers::g_dequant_tables.get_weight_tab(ise_weight_range).m_val_to_ise;
			const int lo_index = weight_val_to_ise_tab[0], hi_index = weight_val_to_ise_tab[64], mid_index = weight_val_to_ise_tab[32];

			const vec3F low_color((float)pDecoded_half[lo_index * 3 + 0], (float)pDecoded_half[lo_index * 3 + 1], (float)pDecoded_half[lo_index * 3 + 2]);
			const vec3F high_color((float)pDecoded_half[hi_index * 3 + 0], (float)pDecoded_half[hi_index * 3 + 1], (float)pDecoded_half[hi_index * 3 + 2]);
			const vec3F mid_color((float)pDecoded_half[mid_index * 3 + 0], (float)pDecoded_half[mid_index * 3 + 1], (float)pDecoded_half[mid_index * 3 + 2]);
						
			const vec3F block_dir(high_color - low_color);

			for (uint32_t p = 0; p < num_pixels; p++)
			{
				const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

				const half_float desired_r = pDesired_half[0], desired_g = pDesired_half[1], desired_b = pDesired_half[2];

				const int64_t desired_half_r_q = q2(desired_r, coptions.m_q_log_bias);
				const int64_t desired_half_g_q = q2(desired_g, coptions.m_q_log_bias);
				const int64_t desired_half_b_q = q2(desired_b, coptions.m_q_log_bias);
				
				// Determine which side of the middle plane the point is for a modest gain
				vec3F c((float)desired_r - mid_color[0], (float)desired_g - mid_color[1], (float)desired_b - mid_color[2]);
				float d = c.dot(block_dir);
								
				int i = 0, high_index = (num_weight_levels / 2) + 1;
				if (d >= 0.0f)
				{
					i = num_weight_levels / 2;
					high_index = num_weight_levels;
				}

				int64_t lowest_e = INT64_MAX;

				for (; (i + 1) < high_index; i += 2)
				{
					int64_t e0, e1;

					{
						int64_t rd0 = decoded_half_q[i][0] - desired_half_r_q; // 27 bits maximum with half float inputs
						int64_t gd0 = decoded_half_q[i][1] - desired_half_g_q;
						int64_t bd0 = decoded_half_q[i][2] - desired_half_b_q;
						e0 = ((2 * (rd0 * rd0) + 3 * (gd0 * gd0) + bd0 * bd0) << 5) | i; // max 62 bits (27*2+3+5)
					}

					{
						int64_t rd1 = decoded_half_q[i + 1][0] - desired_half_r_q;
						int64_t gd1 = decoded_half_q[i + 1][1] - desired_half_g_q;
						int64_t bd1 = decoded_half_q[i + 1][2] - desired_half_b_q;
						e1 = ((2 * (rd1 * rd1) + 3 * (gd1 * gd1) + bd1 * bd1) << 5) | (i + 1);
					}

					lowest_e = minimum(lowest_e, e0, e1);
				}

				if (i != high_index)
				{
					int64_t rd0 = decoded_half_q[i][0] - desired_half_r_q;
					int64_t gd0 = decoded_half_q[i][1] - desired_half_g_q;
					int64_t bd0 = decoded_half_q[i][2] - desired_half_b_q;
					int64_t e0 = ((2 * (rd0 * rd0) + 3 * (gd0 * gd0) + bd0 * bd0) << 5) | i;

					lowest_e = minimum(lowest_e, e0);
				}

				pWeights[p] = (uint8_t)(lowest_e & 31);

				total_error += (lowest_e >> 5);

			} // p
		}
	}

	return (double)total_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double eval_selectors_dual_plane(
	uint32_t channel_index,
	uint32_t num_pixels,
	uint8_t* pWeights0, uint8_t* pWeights1,
	const half_float* pBlock_pixels_half,
	uint32_t num_weight_levels,
	const half_float* pDecoded_half,
	const astc_hdr_codec_base_options& coptions,
	uint32_t usable_selector_bitmask)
{
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert(usable_selector_bitmask);

	const float R_WEIGHT = coptions.m_r_err_scale;
	const float G_WEIGHT = coptions.m_g_err_scale;

	double total_error = 0;

#ifdef _DEBUG
	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 0]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 1]));
		assert(!is_half_inf_or_nan(pDecoded_half[i * 3 + 2]));
	}
#endif

	double decoded_half_q[MAX_SUPPORTED_WEIGHT_LEVELS][3];

	for (uint32_t i = 0; i < num_weight_levels; i++)
	{
		const half_float* p = &pDecoded_half[i * 3];

		decoded_half_q[i][0] = q(p[0], coptions.m_q_log_bias);
		decoded_half_q[i][1] = q(p[1], coptions.m_q_log_bias);
		decoded_half_q[i][2] = q(p[2], coptions.m_q_log_bias);
	}

	const double channel_weights[3] = { R_WEIGHT, G_WEIGHT, 1.0f };

	const uint32_t first_channel = (channel_index + 1) % 3;
	const uint32_t second_channel = (channel_index + 2) % 3;
	
	// First plane
	const double first_channel_weight = channel_weights[first_channel];
	const double second_channel_weight = channel_weights[second_channel];
		
	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

		const double desired_half_x_q = q(pDesired_half[first_channel], coptions.m_q_log_bias);
		const double desired_half_y_q = q(pDesired_half[second_channel], coptions.m_q_log_bias);

		double lowest_e = BIG_FLOAT_VAL;

		// this is an approximation of MSLE
		for (uint32_t i = 0; i < num_weight_levels; i++)
		{
			if (((1 << i) & usable_selector_bitmask) == 0)
				continue;

			double xd = decoded_half_q[i][first_channel] - desired_half_x_q;
			double yd = decoded_half_q[i][second_channel] - desired_half_y_q;

			double e = first_channel_weight * (xd * xd) + second_channel_weight * (yd * yd);

			if (e < lowest_e)
			{
				lowest_e = e;
				pWeights0[p] = (uint8_t)i;
			}
		}

		total_error += lowest_e;

	} // p

	// Second plane
	const double alt_channel_weight = channel_weights[channel_index];

	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p * 3];

		const double desired_half_a_q = q(pDesired_half[channel_index], coptions.m_q_log_bias);
		
		double lowest_e = BIG_FLOAT_VAL;

		// this is an approximation of MSLE
		for (uint32_t i = 0; i < num_weight_levels; i++)
		{
			if (((1 << i) & usable_selector_bitmask) == 0)
				continue;

			double ad = decoded_half_q[i][channel_index] - desired_half_a_q;

			double e = alt_channel_weight * (ad * ad);

			if (e < lowest_e)
			{
				lowest_e = e;
				pWeights1[p] = (uint8_t)i;
			}
		}

		total_error += lowest_e;

	} // p

	return total_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double compute_block_error(uint32_t num_pixels, const half_float* pOrig_block, const half_float* pPacked_block, const astc_hdr_codec_base_options& coptions)
{
	const float R_WEIGHT = coptions.m_r_err_scale;
	const float G_WEIGHT = coptions.m_g_err_scale;

	double total_error = 0;

	for (uint32_t p = 0; p < num_pixels; p++)
	{
		double rd = q(pOrig_block[p * 3 + 0], coptions.m_q_log_bias) - q(pPacked_block[p * 3 + 0], coptions.m_q_log_bias);
		double gd = q(pOrig_block[p * 3 + 1], coptions.m_q_log_bias) - q(pPacked_block[p * 3 + 1], coptions.m_q_log_bias);
		double bd = q(pOrig_block[p * 3 + 2], coptions.m_q_log_bias) - q(pPacked_block[p * 3 + 2], coptions.m_q_log_bias);

		double e = R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;

		total_error += e;
	}

	return total_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double compute_block_error_from_raw_weights(
	uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3],
	const uint8_t* pRaw_weights,
	int endpoints_qlog12[2][3],
	const astc_hdr_codec_base_options& coptions)
{
	// qlog12->qlog16
	int trial_e[2][3];
	for (uint32_t i = 0; i < 3; i++)
	{
		assert(endpoints_qlog12[0][i] <= (int)basist::MAX_QLOG12);
		assert(endpoints_qlog12[1][i] <= (int)basist::MAX_QLOG12);

		trial_e[0][i] = endpoints_qlog12[0][i] << 4;
		trial_e[1][i] = endpoints_qlog12[1][i] << 4;
	}

	const float R_WEIGHT = coptions.m_r_err_scale, G_WEIGHT = coptions.m_g_err_scale;

	double trial_error = 0;
	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p][0];

		const double desired_half_r_q = q(pDesired_half[0], coptions.m_q_log_bias), desired_half_g_q = q(pDesired_half[1], coptions.m_q_log_bias), desired_half_b_q = q(pDesired_half[2], coptions.m_q_log_bias);

		const uint32_t c = pRaw_weights[p];
		assert(c <= 64);

		{
			half_float rf, gf, bf;
			{
				uint32_t r0 = trial_e[0][0], r1 = trial_e[1][0];
				int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
				rf = astc_helpers::qlog16_to_half(ri);
			}
			{
				uint32_t g0 = trial_e[0][1], g1 = trial_e[1][1];
				int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
				gf = astc_helpers::qlog16_to_half(gi);
			}
			{
				uint32_t b0 = trial_e[0][2], b1 = trial_e[1][2];
				int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
				bf = astc_helpers::qlog16_to_half(bi);
			}

			const double decoded_half_q0 = q(rf, coptions.m_q_log_bias), decoded_half_q1 = q(gf, coptions.m_q_log_bias), decoded_half_q2 = q(bf, coptions.m_q_log_bias);
			const double rd = decoded_half_q0 - desired_half_r_q, gd = decoded_half_q1 - desired_half_g_q, bd = decoded_half_q2 - desired_half_b_q;
			trial_error += R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;
		}
	}

	return trial_error;
}

//--------------------------------------------------------------------------------------------------------------------------

static inline int compute_clamped_val(int v, int l, int h, bool& did_clamp, int& max_clamp_mag)
{
	assert(l < h);

	if (v < l)
	{
		max_clamp_mag = basisu::maximum<int>(max_clamp_mag, l - v);

		v = l;
		did_clamp = true;
	}
	else if (v > h)
	{
		max_clamp_mag = basisu::maximum<int>(max_clamp_mag, v - h);

		v = h;
		did_clamp = true;
	}

	return v;
}

//--------------------------------------------------------------------------------------------------------------------------

const uint8_t s_b_bits[8] = { 7, 8, 6, 7,  8, 6, 7, 6 };
const uint8_t s_c_bits[8] = { 6, 6, 7, 7,  6, 7, 7, 7 };
const uint8_t s_d_bits[8] = { 7, 6, 7, 6,  5, 6, 5, 6 };

// val_q[] must be already packed to qlog9-qlog12.
bool pack_astc_mode11_submode(uint32_t submode, uint8_t* pEndpoints, int val_q[2][3], int& max_clamp_mag, bool early_out_if_clamped, int max_clamp_mag_accept_thresh)
{
	assert(submode <= 7);

	const uint32_t a_bits = 9 + (submode >> 1);
	const uint32_t b_bits = s_b_bits[submode];
	const uint32_t c_bits = s_c_bits[submode];
	const uint32_t d_bits = s_d_bits[submode];

	const int max_a_val = (1 << a_bits) - 1;
	const int max_b_val = (1 << b_bits) - 1;
	const int max_c_val = (1 << c_bits) - 1;

	// The maximum usable value before it turns to NaN/Inf
	const int max_a_qlog = get_max_qlog(a_bits);
	BASISU_NOTE_UNUSED(max_a_qlog);

	const int min_d_val = -(1 << (d_bits - 1));
	const int max_d_val = -min_d_val - 1;
	assert((max_d_val - min_d_val + 1) == (1 << d_bits));

	int highest_q = -1, highest_val = 0, highest_comp = 0;

	for (uint32_t c = 0; c < 3; c++)
	{
		assert(val_q[0][c] <= max_a_qlog);
		assert(val_q[1][c] <= max_a_qlog);
	}

	for (uint32_t v = 0; v < 2; v++)
	{
		for (uint32_t c = 0; c < 3; c++)
		{
			assert(val_q[v][c] >= 0 && val_q[v][c] <= max_a_val);

			if (val_q[v][c] > highest_q)
			{
				highest_q = val_q[v][c];
				highest_val = v;
				highest_comp = c;
			}
		}
	}

	const bool had_tie = (val_q[highest_val ^ 1][highest_comp] == highest_q);

	if (highest_val != 1)
	{
		for (uint32_t c = 0; c < 3; c++)
		{
			std::swap(val_q[0][c], val_q[1][c]);
		}
	}

	if (highest_comp)
	{
		std::swap(val_q[0][0], val_q[0][highest_comp]);
		std::swap(val_q[1][0], val_q[1][highest_comp]);
	}

	int orig_q[2][3];
	memcpy(orig_q, val_q, sizeof(int) * 6);

	// val[1][0] is now guaranteed to be highest
	int best_va = 0, best_vb0 = 0, best_vb1 = 0, best_vc = 0, best_vd0 = 0, best_vd1 = 0;
	int best_max_clamp_mag = 0;
	bool best_did_clamp = false;
	int best_q[2][3] = { { 0, 0, 0}, { 0, 0, 0 } };
	BASISU_NOTE_UNUSED(best_q);
	uint32_t best_dist = UINT_MAX;

	for (uint32_t pass = 0; pass < 2; pass++)
	{
		int trial_va = val_q[1][0];

		assert(trial_va <= max_a_val);
		assert(trial_va >= val_q[1][1]);
		assert(trial_va >= val_q[1][2]);

		assert(trial_va >= val_q[0][0]);
		assert(trial_va >= val_q[0][1]);
		assert(trial_va >= val_q[0][2]);

		bool did_clamp = false;
		int trial_max_clamp_mag = 0;

		int trial_vb0 = compute_clamped_val(trial_va - val_q[1][1], 0, max_b_val, did_clamp, trial_max_clamp_mag);
		int trial_vb1 = compute_clamped_val(trial_va - val_q[1][2], 0, max_b_val, did_clamp, trial_max_clamp_mag);
		int trial_vc = compute_clamped_val(trial_va - val_q[0][0], 0, max_c_val, did_clamp, trial_max_clamp_mag);
		int trial_vd0 = compute_clamped_val((trial_va - trial_vb0 - trial_vc) - val_q[0][1], min_d_val, max_d_val, did_clamp, trial_max_clamp_mag);
		int trial_vd1 = compute_clamped_val((trial_va - trial_vb1 - trial_vc) - val_q[0][2], min_d_val, max_d_val, did_clamp, trial_max_clamp_mag);

		if ((early_out_if_clamped) && (did_clamp) && (trial_max_clamp_mag > max_clamp_mag_accept_thresh))
		{
			if ((!had_tie) || (pass == 1))
			{
				max_clamp_mag = trial_max_clamp_mag;
				return true;
			}
		}

		if (!did_clamp)
		{
			// Make sure decoder gets the expected values
			assert(trial_va == val_q[1][0]);
			assert(trial_va - trial_vb0 == val_q[1][1]);
			assert(trial_va - trial_vb1 == val_q[1][2]);

			assert((trial_va - trial_vc) == val_q[0][0]);
			assert((trial_va - trial_vb0 - trial_vc - trial_vd0) == val_q[0][1]);
			assert((trial_va - trial_vb1 - trial_vc - trial_vd1) == val_q[0][2]);
		}

		const int r_e0 = clamp<int>(trial_va, 0, max_a_val);
		const int r_e1 = clamp<int>(trial_va - trial_vb0, 0, max_a_val);
		const int r_e2 = clamp<int>(trial_va - trial_vb1, 0, max_a_val);

		const int r_f0 = clamp<int>(trial_va - trial_vc, 0, max_a_val);
		const int r_f1 = clamp<int>(trial_va - trial_vb0 - trial_vc - trial_vd0, 0, max_a_val);
		const int r_f2 = clamp<int>(trial_va - trial_vb1 - trial_vc - trial_vd1, 0, max_a_val);

		assert(r_e0 <= max_a_qlog);
		assert(r_e1 <= max_a_qlog);
		assert(r_e2 <= max_a_qlog);

		assert(r_f0 <= max_a_qlog);
		assert(r_f1 <= max_a_qlog);
		assert(r_f2 <= max_a_qlog);

		if ((!did_clamp) || (!had_tie))
		{
			best_va = trial_va;
			best_vb0 = trial_vb0;
			best_vb1 = trial_vb1;
			best_vc = trial_vc;
			best_vd0 = trial_vd0;
			best_vd1 = trial_vd1;
			best_max_clamp_mag = trial_max_clamp_mag;
			best_did_clamp = did_clamp;

			best_q[1][0] = r_e0;
			best_q[1][1] = r_e1;
			best_q[1][2] = r_e2;
			best_q[0][0] = r_f0;
			best_q[0][1] = r_f1;
			best_q[0][2] = r_f2;
			break;
		}

		// we had a tie and it did clamp, try swapping L/H for a potential slight gain

		const uint32_t r_dist1 = basisu::square<int>(r_e0 - val_q[1][0]) + basisu::square<int>(r_e1 - val_q[1][1]) + basisu::square<int>(r_e2 - val_q[1][2]);
		const uint32_t r_dist0 = basisu::square<int>(r_f0 - val_q[0][0]) + basisu::square<int>(r_f1 - val_q[0][1]) + basisu::square<int>(r_f2 - val_q[0][2]);

		const uint32_t total_dist = r_dist1 + r_dist0;

		if (total_dist < best_dist)
		{
			best_dist = total_dist;

			best_va = trial_va;
			best_vb0 = trial_vb0;
			best_vb1 = trial_vb1;
			best_vc = trial_vc;
			best_vd0 = trial_vd0;
			best_vd1 = trial_vd1;
			best_did_clamp = did_clamp;

			best_q[1][0] = r_e0;
			best_q[1][1] = r_e1;
			best_q[1][2] = r_e2;
			best_q[0][0] = r_f0;
			best_q[0][1] = r_f1;
			best_q[0][2] = r_f2;
		}

		for (uint32_t c = 0; c < 3; c++)
			std::swap(val_q[0][c], val_q[1][c]);
	}

	// pack bits now
	int v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

	int x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
	switch (submode)
	{
	case 0:
		x0 = get_bit(best_vb0, 6); x1 = get_bit(best_vb1, 6); x2 = get_bit(best_vd0, 6); x3 = get_bit(best_vd1, 6); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	case 1:
		x0 = get_bit(best_vb0, 6); x1 = get_bit(best_vb1, 6); x2 = get_bit(best_vb0, 7); x3 = get_bit(best_vb1, 7); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	case 2:
		x0 = get_bit(best_va, 9); x1 = get_bit(best_vc, 6); x2 = get_bit(best_vd0, 6); x3 = get_bit(best_vd1, 6); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	case 3:
		x0 = get_bit(best_vb0, 6); x1 = get_bit(best_vb1, 6); x2 = get_bit(best_va, 9); x3 = get_bit(best_vc, 6); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	case 4:
		x0 = get_bit(best_vb0, 6); x1 = get_bit(best_vb1, 6); x2 = get_bit(best_vb0, 7); x3 = get_bit(best_vb1, 7); x4 = get_bit(best_va, 9); x5 = get_bit(best_va, 10);
		break;
	case 5:
		x0 = get_bit(best_va, 9); x1 = get_bit(best_va, 10); x2 = get_bit(best_vc, 7); x3 = get_bit(best_vc, 6); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	case 6:
		x0 = get_bit(best_vb0, 6); x1 = get_bit(best_vb1, 6); x2 = get_bit(best_va, 11); x3 = get_bit(best_vc, 6); x4 = get_bit(best_va, 9); x5 = get_bit(best_va, 10);
		break;
	case 7:
		x0 = get_bit(best_va, 9); x1 = get_bit(best_va, 10); x2 = get_bit(best_va, 11); x3 = get_bit(best_vc, 6); x4 = get_bit(best_vd0, 5); x5 = get_bit(best_vd1, 5);
		break;
	default:
		break;
	}

	// write mode
	pack_bit(v1, 7, submode, 0);
	pack_bit(v2, 7, submode, 1);
	pack_bit(v3, 7, submode, 2);

	// highest component
	pack_bit(v4, 7, highest_comp, 0);
	pack_bit(v5, 7, highest_comp, 1);

	// write bit 8 of va
	pack_bit(v1, 6, best_va, 8);

	// extra bits
	pack_bit(v2, 6, x0);
	pack_bit(v3, 6, x1);
	pack_bit(v4, 6, x2);
	pack_bit(v5, 6, x3);
	pack_bit(v4, 5, x4);
	pack_bit(v5, 5, x5);

	v0 = best_va & 0xFF;
	v1 |= (best_vc & 63);
	v2 |= (best_vb0 & 63);
	v3 |= (best_vb1 & 63);
	v4 |= (best_vd0 & 31);
	v5 |= (best_vd1 & 31);

	assert(in_range(v0, 0, 255) && in_range(v1, 0, 255) && in_range(v2, 0, 255) && in_range(v3, 0, 255) && in_range(v4, 0, 255) && in_range(v5, 0, 255));

	pEndpoints[0] = (uint8_t)v0;
	pEndpoints[1] = (uint8_t)v1;
	pEndpoints[2] = (uint8_t)v2;
	pEndpoints[3] = (uint8_t)v3;
	pEndpoints[4] = (uint8_t)v4;
	pEndpoints[5] = (uint8_t)v5;

#ifdef _DEBUG
	// Test for valid pack by unpacking
	{
		if (highest_comp)
		{
			std::swap(best_q[0][0], best_q[0][highest_comp]);
			std::swap(best_q[1][0], best_q[1][highest_comp]);

			std::swap(orig_q[0][0], orig_q[0][highest_comp]);
			std::swap(orig_q[1][0], orig_q[1][highest_comp]);
		}

		int test_e[2][3];
		decode_mode11_to_qlog12(pEndpoints, test_e, astc_helpers::BISE_256_LEVELS);
		for (uint32_t i = 0; i < 2; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				assert(best_q[i][j] == test_e[i][j] >> (12 - a_bits));

				if (!best_did_clamp)
				{
					assert((orig_q[i][j] == test_e[i][j] >> (12 - a_bits)) ||
						(orig_q[1 - i][j] == test_e[i][j] >> (12 - a_bits)));
				}
			}
		}
	}
#endif

	max_clamp_mag = best_max_clamp_mag;

	return best_did_clamp;
}

bool pack_astc_mode11_submode(uint32_t submode, uint8_t* pEndpoints, const vec3F& low_q16, const vec3F& high_q16, int& max_clamp_mag, bool early_out_if_clamped, int max_clamp_mag_accept_thresh)
{
	assert(submode <= 7);
		
	const uint32_t a_bits = 9 + (submode >> 1);
	const int max_a_val = (1 << a_bits) - 1;

	// The maximum usable value before it turns to NaN/Inf
	const int max_a_qlog = get_max_qlog(a_bits);

	int val_q[2][3];

	for (uint32_t c = 0; c < 3; c++)
	{
#if 0
		// This is very slightly better, but ~8% slower likely due to the table lookups.
		const half_float l = astc_helpers::qlog16_to_half((uint32_t)std::round(low_q16[c]));
		val_q[0][c] = half_to_qlog7_12(l, a_bits);

		const half_float h = astc_helpers::qlog16_to_half((uint32_t)std::round(high_q16[c]));
		val_q[1][c] = half_to_qlog7_12(h, a_bits);
#else
		// TODO: Tune quant_qlog16() for higher precision.
		val_q[0][c] = quant_qlog16((uint32_t)std::round(low_q16[c]), a_bits);
		val_q[1][c] = quant_qlog16((uint32_t)std::round(high_q16[c]), a_bits);
#endif

#if 1
		if (val_q[0][c] == val_q[1][c])
		{
#if 0
			if (l <= h)
#else
			if (low_q16[c] < high_q16[c])
#endif
			{
				if (val_q[0][c])
					val_q[0][c]--;

				if (val_q[1][c] != max_a_val)
					val_q[1][c]++;
			}
			else
			{
				if (val_q[0][c] != max_a_val)
					val_q[0][c]++;

				if (val_q[1][c])
					val_q[1][c]--;
			}
		}
#endif

		val_q[0][c] = minimum<uint32_t>(val_q[0][c], max_a_qlog);
		val_q[1][c] = minimum<uint32_t>(val_q[1][c], max_a_qlog);
	}

	return pack_astc_mode11_submode(submode, pEndpoints, val_q, max_clamp_mag, early_out_if_clamped, max_clamp_mag_accept_thresh);
}

//--------------------------------------------------------------------------------------------------------------------------

void pack_astc_mode11_direct(uint8_t* pEndpoints, vec3F l_q16, vec3F h_q16)
{
	float lg = l_q16.dot(vec3F(1.0f)), hg = h_q16.dot(vec3F(1.0f));
	if (lg > hg)
	{
		// Ensure low endpoint is generally less bright than high in direct mode.
		std::swap(l_q16, h_q16);
	}

	for (uint32_t i = 0; i < 3; i++)
	{
		// TODO: This goes from QLOG16->HALF->QLOG8/7
		half_float l_half = astc_helpers::qlog16_to_half(clamp((int)std::round(l_q16[i]), 0, 65535));
		half_float h_half = astc_helpers::qlog16_to_half(clamp((int)std::round(h_q16[i]), 0, 65535));

		int l_q, h_q;

		if (i == 2)
		{
			l_q = g_half_to_qlog7[bounds_check((uint32_t)l_half, 0U, 32768U)];
			h_q = g_half_to_qlog7[bounds_check((uint32_t)h_half, 0U, 32768U)];

			l_q = minimum<uint32_t>(l_q, MAX_QLOG7);
			h_q = minimum<uint32_t>(h_q, MAX_QLOG7);
		}
		else
		{
			l_q = g_half_to_qlog8[bounds_check((uint32_t)l_half, 0U, 32768U)];
			h_q = g_half_to_qlog8[bounds_check((uint32_t)h_half, 0U, 32768U)];

			// this quantizes R and G as 7 bits vs. 8, for grayscale.
			//l_q = g_half_to_qlog7[bounds_check((uint32_t)l_half, 0U, 32768U)] << 1;
			//h_q = g_half_to_qlog7[bounds_check((uint32_t)h_half, 0U, 32768U)] << 1;
						
			l_q = minimum<uint32_t>(l_q, MAX_QLOG8);
			h_q = minimum<uint32_t>(h_q, MAX_QLOG8);
		}

#if 1
		if (l_q == h_q)
		{
			const int m = (i == 2) ? MAX_QLOG7 : MAX_QLOG8;

			if (l_q16[i] <= h_q16[i])
			{
				if (l_q)
					l_q--;

				if (h_q != m)
					h_q++;
			}
			else
			{
				if (h_q)
					h_q--;

				if (l_q != m)
					l_q++;
			}
		}
#endif

		if (i == 2)
		{
			assert(l_q <= (int)MAX_QLOG7 && h_q <= (int)MAX_QLOG7);
			l_q |= 128;
			h_q |= 128;
		}
		else
		{
			assert(l_q <= (int)MAX_QLOG8 && h_q <= (int)MAX_QLOG8);
		}

		pEndpoints[2 * i + 0] = (uint8_t)l_q;
		pEndpoints[2 * i + 1] = (uint8_t)h_q;
	}
}

//--------------------------------------------------------------------------------------------------------------------------

bool pack_astc_mode7_submode(uint32_t submode, uint8_t* pEndpoints, const vec3F& rgb_q16, float s_q16, int& max_clamp_mag, uint32_t ise_weight_range, bool early_out_if_clamped, int max_clamp_mag_accept_thresh)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));

	assert(submode <= 5);
	max_clamp_mag = 0;

	static const uint8_t s_r_bits[6] = { 11, 11, 10, 9, 8, 7 };
	static const uint8_t s_g_b_bits[6] = { 5, 6, 5, 6, 7, 7 };
	static const uint8_t s_s_bits[6] = { 7, 5, 8, 7, 6, 7 };

	// The precision of the components
	const uint32_t prec_bits = s_r_bits[submode];

	int qlog[4], pack_bits[4];

	for (uint32_t i = 0; i < 4; i++)
	{
		const float f = (i == 3) ? s_q16 : rgb_q16[i];

		// The # of bits the component is packed into
		if (i == 0)
			pack_bits[i] = s_r_bits[submode];
		else if (i == 3)
			pack_bits[i] = s_s_bits[submode];
		else
			pack_bits[i] = s_g_b_bits[submode];

#if 0
		// this is slightly worse
		// TODO: going from qlog16 to half loses some precision. Then going from half to qlog 7-12 will have extra error.
		half_float h = qlog_to_half(clamp((int)std::round(f), 0, MAX_QLOG16), 16);
		qlog[i] = half_to_qlog7_12((half_float)bounds_check((uint32_t)h, 0U, 32768U), prec_bits);
#else
		qlog[i] = quant_qlog16(clamp<int>((int)std::round(f), 0, MAX_QLOG16), prec_bits);

		// Only bias if there are enough texel weights, 4=6 weights
		if (ise_weight_range >= 4)
		{
			// Explictly bias the high color, and the scale up, to better exploit the weights.
			// The quantized range also then encompases the complete input range.
			const uint32_t max_val = (1 << prec_bits) - 1;
			const uint32_t K = 3;
			if (i == 3)
			{
				qlog[i] = minimum<uint32_t>(qlog[i] + K * 2, max_val);
			}
			else
			{
				qlog[i] = minimum<uint32_t>(qlog[i] + K, max_val);
			}
		}
#endif

		if (i != 3)
			qlog[i] = minimum<uint32_t>(qlog[i], get_max_qlog(prec_bits));

		// If S=0, we lose freedom for the texel weights to add any value.
		if ((i == 3) && (qlog[i] == 0))
			qlog[i] = 1;
	}

	uint32_t maj_index = 0;

	bool did_clamp = false;

	if (submode != 5)
	{
		int largest_qlog = 0;
		for (uint32_t i = 0; i < 3; i++)
		{
			if (qlog[i] > largest_qlog)
			{
				largest_qlog = qlog[i];
				maj_index = i;
			}
		}

		if (maj_index)
		{
			std::swap(qlog[0], qlog[maj_index]);
		}

		assert(qlog[0] >= qlog[1]);
		assert(qlog[0] >= qlog[2]);

		qlog[1] = qlog[0] - qlog[1];
		qlog[2] = qlog[0] - qlog[2];

		for (uint32_t i = 1; i < 4; i++)
		{
			const int max_val = (1 << pack_bits[i]) - 1;

			if (qlog[i] > max_val)
			{
				max_clamp_mag = maximum<int>(max_clamp_mag, qlog[i] - max_val);
				qlog[i] = max_val;
				did_clamp = true;

				if ((early_out_if_clamped) && (max_clamp_mag > max_clamp_mag_accept_thresh))
					return true;
			}
		}
	}

	for (uint32_t i = 0; i < 4; i++)
	{
		const int max_val = (1 << pack_bits[i]) - 1; (void)max_val;

		assert(qlog[i] <= max_val);
	}

	int mode = 0;

	int r = qlog[0] & 63; // 6-bits
	int g = qlog[1] & 31; // 5-bits
	int b = qlog[2] & 31; // 5-bits
	int s = qlog[3] & 31; // 5-bits

	int x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0, x6 = 0;

	switch (submode)
	{
	case 0:
	{
		mode = (maj_index << 2) | 0;
		assert((mode & 0xC) != 0xC);

		x0 = get_bit(qlog[0], 9); // R9
		x1 = get_bit(qlog[0], 8); // R8
		x2 = get_bit(qlog[0], 7); // R7
		x3 = get_bit(qlog[0], 10); // R10
		x4 = get_bit(qlog[0], 6); // R6 
		x5 = get_bit(qlog[3], 6); // S6
		x6 = get_bit(qlog[3], 5); // S5
		break;
	}
	case 1:
	{
		mode = (maj_index << 2) | 1;
		assert((mode & 0xC) != 0xC);

		x0 = get_bit(qlog[0], 8); // R8
		x1 = get_bit(qlog[1], 5); // G5
		x2 = get_bit(qlog[0], 7); // R7
		x3 = get_bit(qlog[2], 5); // B5
		x4 = get_bit(qlog[0], 6); // R6 
		x5 = get_bit(qlog[0], 10); // R10
		x6 = get_bit(qlog[0], 9); // R9
		break;
	}
	case 2:
	{
		mode = (maj_index << 2) | 2;
		assert((mode & 0xC) != 0xC);

		x0 = get_bit(qlog[0], 9); // R9
		x1 = get_bit(qlog[0], 8); // R8
		x2 = get_bit(qlog[0], 7); // R7
		x3 = get_bit(qlog[0], 6); // R6
		x4 = get_bit(qlog[3], 7); // S7 
		x5 = get_bit(qlog[3], 6); // S6
		x6 = get_bit(qlog[3], 5); // S5
		break;
	}
	case 3:
	{
		mode = (maj_index << 2) | 3;
		assert((mode & 0xC) != 0xC);

		x0 = get_bit(qlog[0], 8); // R8
		x1 = get_bit(qlog[1], 5); // G5
		x2 = get_bit(qlog[0], 7); // R7
		x3 = get_bit(qlog[2], 5); // B5
		x4 = get_bit(qlog[0], 6); // R6 
		x5 = get_bit(qlog[3], 6); // S6
		x6 = get_bit(qlog[3], 5); // S5
		break;
	}
	case 4:
	{
		mode = maj_index | 0xC; // 0b1100
		assert((mode & 0xC) == 0xC);
		assert(mode != 0xF);

		x0 = get_bit(qlog[1], 6); // G6
		x1 = get_bit(qlog[1], 5); // G5
		x2 = get_bit(qlog[2], 6); // B6
		x3 = get_bit(qlog[2], 5); // B5
		x4 = get_bit(qlog[0], 6); // R6 
		x5 = get_bit(qlog[0], 7); // R7
		x6 = get_bit(qlog[3], 5); // S5
		break;
	}
	case 5:
	{
		mode = 0xF;

		x0 = get_bit(qlog[1], 6); // G6
		x1 = get_bit(qlog[1], 5); // G5
		x2 = get_bit(qlog[2], 6); // B6
		x3 = get_bit(qlog[2], 5); // B5
		x4 = get_bit(qlog[0], 6); // R6 
		x5 = get_bit(qlog[3], 6); // S6
		x6 = get_bit(qlog[3], 5); // S5
		break;
	}
	default:
	{
		assert(0);
		break;
	}
	}

	pEndpoints[0] = (uint8_t)((get_bit(mode, 1) << 7) | (get_bit(mode, 0) << 6) | r);
	pEndpoints[1] = (uint8_t)((get_bit(mode, 2) << 7) | (x0 << 6) | (x1 << 5) | g);
	pEndpoints[2] = (uint8_t)((get_bit(mode, 3) << 7) | (x2 << 6) | (x3 << 5) | b);
	pEndpoints[3] = (uint8_t)((x4 << 7) | (x5 << 6) | (x6 << 5) | s);

#ifdef _DEBUG
	// Test for valid pack by unpacking
	{
		const int inv_shift = 12 - prec_bits;

		int unpacked_e[2][3];
		if (submode != 5)
		{
			unpacked_e[1][0] = left_shift32(qlog[0], inv_shift);
			unpacked_e[1][1] = clamp(left_shift32((qlog[0] - qlog[1]), inv_shift), 0, 0xFFF);
			unpacked_e[1][2] = clamp(left_shift32((qlog[0] - qlog[2]), inv_shift), 0, 0xFFF);

			unpacked_e[0][0] = clamp(left_shift32((qlog[0] - qlog[3]), inv_shift), 0, 0xFFF);
			unpacked_e[0][1] = clamp(left_shift32(((qlog[0] - qlog[1]) - qlog[3]), inv_shift), 0, 0xFFF);
			unpacked_e[0][2] = clamp(left_shift32(((qlog[0] - qlog[2]) - qlog[3]), inv_shift), 0, 0xFFF);
		}
		else
		{
			unpacked_e[1][0] = left_shift32(qlog[0], inv_shift);
			unpacked_e[1][1] = left_shift32(qlog[1], inv_shift);
			unpacked_e[1][2] = left_shift32(qlog[2], inv_shift);

			unpacked_e[0][0] = clamp(left_shift32((qlog[0] - qlog[3]), inv_shift), 0, 0xFFF);
			unpacked_e[0][1] = clamp(left_shift32((qlog[1] - qlog[3]), inv_shift), 0, 0xFFF);
			unpacked_e[0][2] = clamp(left_shift32((qlog[2] - qlog[3]), inv_shift), 0, 0xFFF);
		}

		if (maj_index)
		{
			std::swap(unpacked_e[0][0], unpacked_e[0][maj_index]);
			std::swap(unpacked_e[1][0], unpacked_e[1][maj_index]);
		}

		int e[2][3];
		decode_mode7_to_qlog12_ise20(pEndpoints, e, nullptr);

		for (uint32_t i = 0; i < 3; i++)
		{
			assert(unpacked_e[0][i] == e[0][i]);
			assert(unpacked_e[1][i] == e[1][i]);
		}
	}
#endif

	return did_clamp;
}

//--------------------------------------------------------------------------------------------------------------------------

bool pack_mode11(mode11_log_desc& desc, uint8_t* pEndpoints)
{
	memset(pEndpoints, 0, NUM_MODE11_ENDPOINTS);

	if (desc.is_direct())
	{
		if ((desc.m_a < 0) || (desc.m_c < 0) || (desc.m_b0 < 0))
			return false;

		if (!((desc.m_a <= 255) && (desc.m_c <= 255) && (desc.m_b0 <= 127)))
			return false;

		pEndpoints[0] = (uint8_t)desc.m_a;
		pEndpoints[2] = (uint8_t)desc.m_c;
		pEndpoints[4] = (uint8_t)desc.m_b0 | 128;

		if ((desc.m_b1 < 0) || (desc.m_d0 < 0) || (desc.m_d1 < 0))
			return false;

		if (!((desc.m_b1 <= 255) && (desc.m_d0 <= 255) && (desc.m_d1 <= 127)))
			return false;

		pEndpoints[1] = (uint8_t)desc.m_b1;
		pEndpoints[3] = (uint8_t)desc.m_d0;
		pEndpoints[5] = (uint8_t)desc.m_d1 | 128;
		
		return true;
	}

	if (!((desc.m_a >= 0) && (desc.m_a <= desc.m_max_a_val)))
		return false;
	if (!(((desc.m_c >= 0) && (desc.m_c <= desc.m_max_c_val))))
		return false;
	if (!((desc.m_b0 >= 0) && (desc.m_b0 <= desc.m_max_b_val)))
		return false;
	if (!((desc.m_b1 >= 0) && (desc.m_b1 <= desc.m_max_b_val)))
		return false;
	if (!((desc.m_d0 >= desc.m_min_d_val) && (desc.m_d0 <= desc.m_max_d_val)))
		return false;
	if (!((desc.m_d1 >= desc.m_min_d_val) && (desc.m_d1 <= desc.m_max_d_val)))
		return false;

	const int va = desc.m_a, vb0 = desc.m_b0, vb1 = desc.m_b1, vc = desc.m_c, vd0 = desc.m_d0, vd1 = desc.m_d1;
	
	int v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
	
	int x0 = 0, x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0;
	switch (desc.m_submode)
	{
	case 0:
		x0 = get_bit(vb0, 6); x1 = get_bit(vb1, 6); x2 = get_bit(vd0, 6); x3 = get_bit(vd1, 6); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	case 1:
		x0 = get_bit(vb0, 6); x1 = get_bit(vb1, 6); x2 = get_bit(vb0, 7); x3 = get_bit(vb1, 7); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	case 2:
		x0 = get_bit(va, 9); x1 = get_bit(vc, 6); x2 = get_bit(vd0, 6); x3 = get_bit(vd1, 6); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	case 3:
		x0 = get_bit(vb0, 6); x1 = get_bit(vb1, 6); x2 = get_bit(va, 9); x3 = get_bit(vc, 6); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	case 4:
		x0 = get_bit(vb0, 6); x1 = get_bit(vb1, 6); x2 = get_bit(vb0, 7); x3 = get_bit(vb1, 7); x4 = get_bit(va, 9); x5 = get_bit(va, 10);
		break;
	case 5:
		x0 = get_bit(va, 9); x1 = get_bit(va, 10); x2 = get_bit(vc, 7); x3 = get_bit(vc, 6); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	case 6:
		x0 = get_bit(vb0, 6); x1 = get_bit(vb1, 6); x2 = get_bit(va, 11); x3 = get_bit(vc, 6); x4 = get_bit(va, 9); x5 = get_bit(va, 10);
		break;
	case 7:
		x0 = get_bit(va, 9); x1 = get_bit(va, 10); x2 = get_bit(va, 11); x3 = get_bit(vc, 6); x4 = get_bit(vd0, 5); x5 = get_bit(vd1, 5);
		break;
	default:
		break;
	}

	// write mode
	pack_bit(v1, 7, desc.m_submode, 0);
	pack_bit(v2, 7, desc.m_submode, 1);
	pack_bit(v3, 7, desc.m_submode, 2);

	// highest component
	pack_bit(v4, 7, desc.m_maj_comp, 0);
	pack_bit(v5, 7, desc.m_maj_comp, 1);

	// write bit 8 of va
	pack_bit(v1, 6, va, 8);

	// extra bits
	pack_bit(v2, 6, x0);
	pack_bit(v3, 6, x1);
	pack_bit(v4, 6, x2);
	pack_bit(v5, 6, x3);
	pack_bit(v4, 5, x4);
	pack_bit(v5, 5, x5);

	v0 = va & 0xFF;
	v1 |= (vc & 63);
	v2 |= (vb0 & 63);
	v3 |= (vb1 & 63);
	v4 |= (vd0 & 31);
	v5 |= (vd1 & 31);

	assert(in_range(v0, 0, 255) && in_range(v1, 0, 255) && in_range(v2, 0, 255) && in_range(v3, 0, 255) && in_range(v4, 0, 255) && in_range(v5, 0, 255));

	pEndpoints[0] = (uint8_t)v0;
	pEndpoints[1] = (uint8_t)v1;
	pEndpoints[2] = (uint8_t)v2;
	pEndpoints[3] = (uint8_t)v3;
	pEndpoints[4] = (uint8_t)v4;
	pEndpoints[5] = (uint8_t)v5;

	return true;
}

static inline int astc_hdr_sign_extend(int src, int num_src_bits)
{
	assert(basisu::in_range(num_src_bits, 2, 31));

	const bool negative = (src & (1 << (num_src_bits - 1))) != 0;
	if (negative)
		return src | ~((1 << num_src_bits) - 1);
	else
		return src & ((1 << num_src_bits) - 1);
}

void unpack_mode11(const uint8_t* pEndpoints, mode11_log_desc& desc)
{
	clear_obj(desc);

	pack_bit(desc.m_maj_comp, 0, pEndpoints[4], 7);
	pack_bit(desc.m_maj_comp, 1, pEndpoints[5], 7);

	if (desc.m_maj_comp == 3)
	{
		desc.m_a = pEndpoints[0];
		desc.m_c = pEndpoints[2];
		desc.m_b0 = pEndpoints[4] & 0x7F;

		desc.m_b1 = pEndpoints[1];
		desc.m_d0 = pEndpoints[3];
		desc.m_d1 = pEndpoints[5] & 0x7F;
		
		return;
	}

	pack_bit(desc.m_submode, 0, pEndpoints[1], 7);
	pack_bit(desc.m_submode, 1, pEndpoints[2], 7);
	pack_bit(desc.m_submode, 2, pEndpoints[3], 7);

	desc.m_a = pEndpoints[0];		// 8 bits
	pack_bit(desc.m_a, 8, pEndpoints[1], 6);

	desc.m_c = pEndpoints[1] & 63;	// 6 bits
	desc.m_b0 = pEndpoints[2] & 63; // 6 bits
	desc.m_b1 = pEndpoints[3] & 63; // 6 bits
	desc.m_d0 = pEndpoints[4] & 31; // 5 bits
	desc.m_d1 = pEndpoints[5] & 31; // 5 bits

	const int x0 = get_bit(pEndpoints[2], 6);
	const int x1 = get_bit(pEndpoints[3], 6);
	const int x2 = get_bit(pEndpoints[4], 6);
	const int x3 = get_bit(pEndpoints[5], 6);
	const int x4 = get_bit(pEndpoints[4], 5);
	const int x5 = get_bit(pEndpoints[5], 5);

	switch (desc.m_submode)
	{
	case 0:
		pack_bit(desc.m_b0, 6, x0, 0); pack_bit(desc.m_b1, 6, x1, 0); pack_bit(desc.m_d0, 6, x2, 0); pack_bit(desc.m_d1, 6, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	case 1:
		pack_bit(desc.m_b0, 6, x0, 0); pack_bit(desc.m_b1, 6, x1, 0); pack_bit(desc.m_b0, 7, x2, 0); pack_bit(desc.m_b1, 7, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	case 2:
		pack_bit(desc.m_a, 9, x0, 0); pack_bit(desc.m_c, 6, x1, 0); pack_bit(desc.m_d0, 6, x2, 0); pack_bit(desc.m_d1, 6, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	case 3:
		pack_bit(desc.m_b0, 6, x0, 0); pack_bit(desc.m_b1, 6, x1, 0); pack_bit(desc.m_a, 9, x2, 0); pack_bit(desc.m_c, 6, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	case 4:
		pack_bit(desc.m_b0, 6, x0, 0); pack_bit(desc.m_b1, 6, x1, 0); pack_bit(desc.m_b0, 7, x2, 0); pack_bit(desc.m_b1, 7, x3, 0); pack_bit(desc.m_a, 9, x4, 0); pack_bit(desc.m_a, 10, x5, 0);
		break;
	case 5:
		pack_bit(desc.m_a, 9, x0, 0); pack_bit(desc.m_a, 10, x1, 0); pack_bit(desc.m_c, 7, x2, 0); pack_bit(desc.m_c, 6, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	case 6:
		pack_bit(desc.m_b0, 6, x0, 0); pack_bit(desc.m_b1, 6, x1, 0); pack_bit(desc.m_a, 11, x2, 0); pack_bit(desc.m_c, 6, x3, 0); pack_bit(desc.m_a, 9, x4, 0); pack_bit(desc.m_a, 10, x5, 0);
		break;
	case 7:
	default:
		pack_bit(desc.m_a, 9, x0, 0); pack_bit(desc.m_a, 10, x1, 0); pack_bit(desc.m_a, 11, x2, 0); pack_bit(desc.m_c, 6, x3, 0); pack_bit(desc.m_d0, 5, x4, 0); pack_bit(desc.m_d1, 5, x5, 0);
		break;
	}

	desc.m_a_bits = 9 + (desc.m_submode >> 1);
	desc.m_b_bits = s_b_bits[desc.m_submode];
	desc.m_c_bits = s_c_bits[desc.m_submode];
	desc.m_d_bits = s_d_bits[desc.m_submode];

	desc.m_max_a_val = (1 << desc.m_a_bits) - 1;
	desc.m_max_b_val = (1 << desc.m_b_bits) - 1;
	desc.m_max_c_val = (1 << desc.m_c_bits) - 1;

	desc.m_min_d_val = -(1 << (desc.m_d_bits - 1));
	desc.m_max_d_val = -desc.m_min_d_val - 1;

	desc.m_d0 = astc_hdr_sign_extend(desc.m_d0, desc.m_d_bits);
	desc.m_d1 = astc_hdr_sign_extend(desc.m_d1, desc.m_d_bits);

	assert((desc.m_a >= 0) && (desc.m_a <= desc.m_max_a_val));
	assert((desc.m_c >= 0) && (desc.m_c <= desc.m_max_c_val));
	assert((desc.m_b0 >= 0) && (desc.m_b0 <= desc.m_max_b_val));
	assert((desc.m_b1 >= 0) && (desc.m_b1 <= desc.m_max_b_val));
	assert((desc.m_d0 >= desc.m_min_d_val) && (desc.m_d0 <= desc.m_max_d_val));
	assert((desc.m_d1 >= desc.m_min_d_val) && (desc.m_d1 <= desc.m_max_d_val));
}

//--------------------------------------------------------------------------------------------------------------------------

void decode_cem_11_config(const uint8_t* pEndpoints, int& submode_index, int& maj_index)
{
	submode_index = 0;
	maj_index = 0;

	pack_bit(submode_index, 0, pEndpoints[1], 7);
	pack_bit(submode_index, 1, pEndpoints[2], 7);
	pack_bit(submode_index, 2, pEndpoints[3], 7);

	pack_bit(maj_index, 0, pEndpoints[4], 7);
	pack_bit(maj_index, 1, pEndpoints[5], 7);
}

//--------------------------------------------------------------------------------------------------------------------------

void decode_cem_7_config(const uint8_t* pEndpoints, int& submode_index, int &maj_index)
{
	const int v0 = pEndpoints[0], v1 = pEndpoints[1], v2 = pEndpoints[2], v3 = pEndpoints[3];
	(void)v3;

	// Extract mode bits and unpack to major component and mode.
	const int modeval = ((v0 & 0xC0) >> 6) | ((v1 & 0x80) >> 5) | ((v2 & 0x80) >> 4);

	if ((modeval & 0xC) != 0xC)
	{
		maj_index = modeval >> 2;
		submode_index = modeval & 3;
	}
	else if (modeval != 0xF)
	{
		maj_index = modeval & 3;
		submode_index = 4;
	}
	else
	{
		maj_index = 0;
		submode_index = 5;
	}
}

//--------------------------------------------------------------------------------------------------------------------------
// TODO: Use pack_mode11() as a shared function.

bool pack_mode11(
	const vec3F& low_color_q16, const vec3F& high_color_q16,
	uint32_t ise_endpoint_range, uint8_t* pEndpoints, 
	const astc_hdr_codec_base_options& coptions,
	bool direct_only, int32_t first_submode, int32_t last_submode, bool ignore_clamping, uint32_t& submode_used)
{
	uint8_t orig_trial_endpoints[NUM_MODE11_ENDPOINTS];

	if (direct_only)
	{
		first_submode = -1;
		last_submode = -1;
	}

	assert(first_submode <= last_submode);
	assert((first_submode >= -1) && (first_submode <= 7));
	assert((last_submode >= -1) && (last_submode <= 7));

	memset(pEndpoints, 0, NUM_MODE11_ENDPOINTS);

	double best_trial_dist = BIG_FLOAT_VAL;
	int best_submode = 0;

	for (int submode = last_submode; submode >= first_submode; submode--)
	{
		bool did_clamp = false;
		int max_clamp_mag = 0;
		if (submode == -1)
		{
			// If it had to clamp with one of the submodes, try direct which can't clamp, but has low precision.
			pack_astc_mode11_direct(orig_trial_endpoints, low_color_q16, high_color_q16);
		}
		else
		{
			const int MAX_CLAMP_MAG_ACCEPT_THRESH = 32;
			did_clamp = pack_astc_mode11_submode(submode, orig_trial_endpoints, low_color_q16, high_color_q16, max_clamp_mag, !ignore_clamping, MAX_CLAMP_MAG_ACCEPT_THRESH);

			if (!ignore_clamping)
			{
				// If it had to clamp and the clamp was too high, it'll distort the endpoint colors too much, which could lead to noticeable artifacts.
				if ((did_clamp) && (max_clamp_mag > MAX_CLAMP_MAG_ACCEPT_THRESH))
					continue;
			}
		}

		uint8_t trial_endpoints[NUM_MODE11_ENDPOINTS];

		// This will distort the endpoints if the ISE endpoint range isn't 256 levels (20).
		// It could massively distort the endpoints, but still result in a valid encoding.
		basist::astc_6x6_hdr::requantize_ise_endpoints(11, astc_helpers::BISE_256_LEVELS, orig_trial_endpoints, ise_endpoint_range, trial_endpoints);

		int e[2][3];
		if (!decode_mode11_to_qlog12(trial_endpoints, e, ise_endpoint_range))
			continue;

		vec3F e0(
			(float)(e[0][0] << 4),
			(float)(e[0][1] << 4),
			(float)(e[0][2] << 4)
		);

		vec3F e1(
			(float)(e[1][0] << 4),
			(float)(e[1][1] << 4),
			(float)(e[1][2] << 4)
		);

		double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
		double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
		double dist = helpers::minimum(dist0, dist1);

		if (dist < best_trial_dist)
		{
			best_trial_dist = dist;
			best_submode = submode;
			memcpy(pEndpoints, trial_endpoints, NUM_MODE11_ENDPOINTS);
		}

		if (coptions.m_take_first_non_clamping_mode11_submode)
		{
			if (!did_clamp)
				break;
		}

	} // submode

	if ((coptions.m_ultra_quant) &&
		(ise_endpoint_range < astc_helpers::BISE_256_LEVELS) &&
		(best_trial_dist != BIG_FLOAT_VAL))
	{
		uint8_t orig_best_trial_endpoints[NUM_MODE11_ENDPOINTS];
		memcpy(orig_best_trial_endpoints, pEndpoints, NUM_MODE11_ENDPOINTS);

		for (uint32_t c = 0; c < NUM_MODE11_ENDPOINTS; c++)
		{
			for (int dt = 0; dt <= 1; dt++)
			{
				const int d = dt ? 1 : -1;

				uint8_t varied_endpoints[NUM_MODE11_ENDPOINTS];
				memcpy(varied_endpoints, orig_best_trial_endpoints, NUM_MODE11_ENDPOINTS);

				int ise = varied_endpoints[c];

				int rank = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_ISE_to_rank[ise];
				rank = clamp<int>(rank + d, 0, astc_helpers::get_ise_levels(ise_endpoint_range) - 1);

				ise = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_rank_to_ISE[rank];

				varied_endpoints[c] = (uint8_t)ise;

				int e[2][3];
				if (!decode_mode11_to_qlog12(varied_endpoints, e, ise_endpoint_range))
					continue;

				vec3F e0(
					(float)(e[0][0] << 4),
					(float)(e[0][1] << 4),
					(float)(e[0][2] << 4)
				);

				vec3F e1(
					(float)(e[1][0] << 4),
					(float)(e[1][1] << 4),
					(float)(e[1][2] << 4)
				);

				double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
				double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
				double dist = helpers::minimum(dist0, dist1);

				if (dist < best_trial_dist)
				{
					best_trial_dist = dist;
					memcpy(pEndpoints, varied_endpoints, NUM_MODE11_ENDPOINTS);
				}
			} // d
		} // c
	} // if (coptions.m_ultra_quant)
		
	submode_used = best_submode + 1;

	return (best_trial_dist != BIG_FLOAT_VAL);
}

bool try_mode11(uint32_t num_pixels,
	uint8_t* pEndpoints, uint8_t* pWeights, double& cur_block_error, uint32_t& submode_used,
	const vec3F& low_color_q16, const vec3F& high_color_q16,
	const basist::half_float block_pixels_half[][3],
	uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions, bool direct_only, uint32_t ise_endpoint_range,
	bool constrain_ise_weight_selectors,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping) // -1, 7
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((num_weight_levels >= MIN_SUPPORTED_WEIGHT_LEVELS) && (num_weight_levels <= MAX_SUPPORTED_WEIGHT_LEVELS));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert(num_weight_levels == astc_helpers::get_ise_levels(ise_weight_range));

	half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];
	uint8_t orig_trial_endpoints[NUM_MODE11_ENDPOINTS], trial_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

	if (direct_only)
	{
		first_submode = -1;
		last_submode = -1;
	}

	assert(first_submode <= last_submode);
	assert((first_submode >= -1) && (first_submode <= 7));
	assert((last_submode >= -1) && (last_submode <= 7));

	uint8_t best_trial_endpoints[NUM_MODE11_ENDPOINTS];
	clear_obj(best_trial_endpoints);
	double best_trial_dist = BIG_FLOAT_VAL;
	int best_submode = 0;

	for (int submode = last_submode; submode >= first_submode; submode--)
	{
		bool did_clamp = false;
		int max_clamp_mag = 0;
		if (submode == -1)
		{
			// If it had to clamp with one of the submodes, try direct which can't clamp, but has low precision.
			pack_astc_mode11_direct(orig_trial_endpoints, low_color_q16, high_color_q16);
		}
		else
		{
			const int MAX_CLAMP_MAG_ACCEPT_THRESH = 32;
			did_clamp = pack_astc_mode11_submode(submode, orig_trial_endpoints, low_color_q16, high_color_q16, max_clamp_mag, !ignore_clamping, MAX_CLAMP_MAG_ACCEPT_THRESH);

			if (!ignore_clamping)
			{
				// If it had to clamp and the clamp was too high, it'll distort the endpoint colors too much, which could lead to noticeable artifacts.
				if ((did_clamp) && (max_clamp_mag > MAX_CLAMP_MAG_ACCEPT_THRESH))
					continue;
			}
		}

		uint8_t trial_endpoints[NUM_MODE11_ENDPOINTS];

		// This will distort the endpoints if the ISE endpoint range isn't 256 levels (20).
		// It could massively distort the endpoints, but still result in a valid encoding.
		basist::astc_6x6_hdr::requantize_ise_endpoints(11, astc_helpers::BISE_256_LEVELS, orig_trial_endpoints, ise_endpoint_range, trial_endpoints);

		int e[2][3];
		if (!decode_mode11_to_qlog12(trial_endpoints, e, ise_endpoint_range))
			continue;

		vec3F e0(
			(float)(e[0][0] << 4),
			(float)(e[0][1] << 4),
			(float)(e[0][2] << 4)
		);

		vec3F e1(
			(float)(e[1][0] << 4),
			(float)(e[1][1] << 4),
			(float)(e[1][2] << 4)
		);

		double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
		double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
		double dist = helpers::minimum(dist0, dist1);

		if (dist < best_trial_dist)
		{
			best_trial_dist = dist;
			best_submode = submode;
			memcpy(best_trial_endpoints, trial_endpoints, sizeof(best_trial_endpoints));
		}

		if (coptions.m_take_first_non_clamping_mode11_submode)
		{
			if (!did_clamp)
				break;
		}

	} // submode

	if ((coptions.m_ultra_quant) &&
		(ise_endpoint_range < astc_helpers::BISE_256_LEVELS) &&
		(best_trial_dist != BIG_FLOAT_VAL))
	{
		uint8_t orig_best_trial_endpoints[NUM_MODE11_ENDPOINTS];
		memcpy(orig_best_trial_endpoints, best_trial_endpoints, NUM_MODE11_ENDPOINTS);

		for (uint32_t c = 0; c < NUM_MODE11_ENDPOINTS; c++)
		{
			for (int dt = 0; dt <= 1; dt++)
			{
				const int d = dt ? 1 : -1;

				uint8_t varied_endpoints[NUM_MODE11_ENDPOINTS];
				memcpy(varied_endpoints, orig_best_trial_endpoints, NUM_MODE11_ENDPOINTS);

				int ise = varied_endpoints[c];

				int rank = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_ISE_to_rank[ise];
				rank = clamp<int>(rank + d, 0, astc_helpers::get_ise_levels(ise_endpoint_range) - 1);

				ise = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_rank_to_ISE[rank];

				varied_endpoints[c] = (uint8_t)ise;

				int e[2][3];
				if (!decode_mode11_to_qlog12(varied_endpoints, e, ise_endpoint_range))
					continue;

				vec3F e0(
					(float)(e[0][0] << 4),
					(float)(e[0][1] << 4),
					(float)(e[0][2] << 4)
				);

				vec3F e1(
					(float)(e[1][0] << 4),
					(float)(e[1][1] << 4),
					(float)(e[1][2] << 4)
				);

				double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
				double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
				double dist = helpers::minimum(dist0, dist1);

				if (dist < best_trial_dist)
				{
					best_trial_dist = dist;
					memcpy(best_trial_endpoints, varied_endpoints, NUM_MODE11_ENDPOINTS);
				}
			} // d
		} // c
	} // if (coptions.m_ultra_quant)

	bool improved_flag = false;

	if (best_trial_dist != BIG_FLOAT_VAL)
	{
		if (get_astc_hdr_mode_11_block_colors(best_trial_endpoints, &decoded_half[0][0], nullptr, num_weight_levels, ise_weight_range, ise_endpoint_range))
		{
			uint32_t usable_selector_bitmask = UINT32_MAX;
			if ((constrain_ise_weight_selectors) && (ise_weight_range == astc_helpers::BISE_16_LEVELS))
				usable_selector_bitmask = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 10) | (1 << 11) | (1 << 14) | (1 << 15);
			else if ((constrain_ise_weight_selectors) && (ise_weight_range == astc_helpers::BISE_12_LEVELS))
				usable_selector_bitmask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);

			double trial_blk_error = eval_selectors(num_pixels, trial_weights, ise_weight_range, &block_pixels_half[0][0], num_weight_levels, &decoded_half[0][0], coptions, usable_selector_bitmask);
			if (trial_blk_error < cur_block_error)
			{
				cur_block_error = trial_blk_error;
				memcpy(pEndpoints, best_trial_endpoints, NUM_MODE11_ENDPOINTS);
				memcpy(pWeights, trial_weights, num_pixels);
				submode_used = best_submode + 1;
				improved_flag = true;
			}
		}
	}

	return improved_flag;
}

//--------------------------------------------------------------------------------------------------------------------------

bool try_mode11_dual_plane(uint32_t channel_index, uint32_t num_pixels,
	uint8_t* pEndpoints, uint8_t* pWeights0, uint8_t* pWeights1, double& cur_block_error, uint32_t& submode_used,
	const vec3F& low_color_q16, const vec3F& high_color_q16,
	const basist::half_float block_pixels_half[][3],
	uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions, bool direct_only, uint32_t ise_endpoint_range,
	bool constrain_ise_weight_selectors,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping) // -1, 7
{
	assert(channel_index <= 2);
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((num_weight_levels >= MIN_SUPPORTED_WEIGHT_LEVELS) && (num_weight_levels <= MAX_SUPPORTED_WEIGHT_LEVELS));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert(num_weight_levels == astc_helpers::get_ise_levels(ise_weight_range));

	half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];
	uint8_t orig_trial_endpoints[NUM_MODE11_ENDPOINTS], trial_weights0[MAX_ASTC_HDR_ENC_BLOCK_PIXELS], trial_weights1[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

	if (direct_only)
	{
		first_submode = -1;
		last_submode = -1;
	}

	assert(first_submode <= last_submode);
	assert((first_submode >= -1) && (first_submode <= 7));
	assert((last_submode >= -1) && (last_submode <= 7));

	uint8_t best_trial_endpoints[NUM_MODE11_ENDPOINTS];
	clear_obj(best_trial_endpoints);

	double best_trial_dist = BIG_FLOAT_VAL;
	int best_submode = 0;

	for (int submode = last_submode; submode >= first_submode; submode--)
	{
		bool did_clamp = false;
		int max_clamp_mag = 0;
		if (submode == -1)
		{
			// If it had to clamp with one of the submodes, try direct which can't clamp, but has low precision.
			pack_astc_mode11_direct(orig_trial_endpoints, low_color_q16, high_color_q16);
		}
		else
		{
			const int MAX_CLAMP_MAG_ACCEPT_THRESH = 32;
			did_clamp = pack_astc_mode11_submode(submode, orig_trial_endpoints, low_color_q16, high_color_q16, max_clamp_mag, !ignore_clamping, MAX_CLAMP_MAG_ACCEPT_THRESH);

			if (!ignore_clamping)
			{
				// If it had to clamp and the clamp was too high, it'll distort the endpoint colors too much, which could lead to noticeable artifacts.
				if ((did_clamp) && (max_clamp_mag > MAX_CLAMP_MAG_ACCEPT_THRESH))
					continue;
			}
		}

		uint8_t trial_endpoints[NUM_MODE11_ENDPOINTS];

		// This will distort the endpoints if the ISE endpoint range isn't 256 levels (20).
		// It could massively distort the endpoints, but still result in a valid encoding.
		basist::astc_6x6_hdr::requantize_ise_endpoints(11, astc_helpers::BISE_256_LEVELS, orig_trial_endpoints, ise_endpoint_range, trial_endpoints);

		int e[2][3];
		if (!decode_mode11_to_qlog12(trial_endpoints, e, ise_endpoint_range))
			continue;

		vec3F e0(
			(float)(e[0][0] << 4),
			(float)(e[0][1] << 4),
			(float)(e[0][2] << 4)
		);

		vec3F e1(
			(float)(e[1][0] << 4),
			(float)(e[1][1] << 4),
			(float)(e[1][2] << 4)
		);

		double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
		double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
		double dist = helpers::minimum(dist0, dist1);

		if (dist < best_trial_dist)
		{
			best_trial_dist = dist;
			best_submode = submode;
			memcpy(best_trial_endpoints, trial_endpoints, sizeof(best_trial_endpoints));
		}

		if (coptions.m_take_first_non_clamping_mode11_submode)
		{
			if (!did_clamp)
				break;
		}

	} // submode

	if ((coptions.m_ultra_quant) &&
		(ise_endpoint_range < astc_helpers::BISE_256_LEVELS) &&
		(best_trial_dist != BIG_FLOAT_VAL))
	{
		uint8_t orig_best_trial_endpoints[NUM_MODE11_ENDPOINTS];
		memcpy(orig_best_trial_endpoints, best_trial_endpoints, NUM_MODE11_ENDPOINTS);

		for (uint32_t c = 0; c < NUM_MODE11_ENDPOINTS; c++)
		{
			for (int dt = 0; dt <= 1; dt++)
			{
				const int d = dt ? 1 : -1;

				uint8_t varied_endpoints[NUM_MODE11_ENDPOINTS];
				memcpy(varied_endpoints, orig_best_trial_endpoints, NUM_MODE11_ENDPOINTS);

				int ise = varied_endpoints[c];

				int rank = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_ISE_to_rank[ise];
				rank = clamp<int>(rank + d, 0, astc_helpers::get_ise_levels(ise_endpoint_range) - 1);

				ise = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_rank_to_ISE[rank];

				varied_endpoints[c] = (uint8_t)ise;

				int e[2][3];
				if (!decode_mode11_to_qlog12(varied_endpoints, e, ise_endpoint_range))
					continue;

				vec3F e0(
					(float)(e[0][0] << 4),
					(float)(e[0][1] << 4),
					(float)(e[0][2] << 4)
				);

				vec3F e1(
					(float)(e[1][0] << 4),
					(float)(e[1][1] << 4),
					(float)(e[1][2] << 4)
				);

				double dist0 = e0.squared_distance_d(low_color_q16) + e1.squared_distance_d(high_color_q16);
				double dist1 = e1.squared_distance_d(low_color_q16) + e0.squared_distance_d(high_color_q16);
				double dist = helpers::minimum(dist0, dist1);

				if (dist < best_trial_dist)
				{
					best_trial_dist = dist;
					memcpy(best_trial_endpoints, varied_endpoints, NUM_MODE11_ENDPOINTS);
				}
			} // d
		} // c
	} // if (coptions.m_ultra_quant)

	bool improved_flag = false;

	if (best_trial_dist != BIG_FLOAT_VAL)
	{
		if (get_astc_hdr_mode_11_block_colors(best_trial_endpoints, &decoded_half[0][0], nullptr, num_weight_levels, ise_weight_range, ise_endpoint_range))
		{
			uint32_t usable_selector_bitmask = UINT32_MAX;
			if ((constrain_ise_weight_selectors) && (ise_weight_range == astc_helpers::BISE_16_LEVELS))
				usable_selector_bitmask = (1 << 0) | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 10) | (1 << 11) | (1 << 14) | (1 << 15);
			else if ((constrain_ise_weight_selectors) && (ise_weight_range == astc_helpers::BISE_12_LEVELS))
				usable_selector_bitmask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);

			double trial_blk_error = eval_selectors_dual_plane(channel_index, num_pixels, trial_weights0, trial_weights1, &block_pixels_half[0][0], num_weight_levels, &decoded_half[0][0], coptions, usable_selector_bitmask);
			if (trial_blk_error < cur_block_error)
			{
				cur_block_error = trial_blk_error;
				memcpy(pEndpoints, best_trial_endpoints, NUM_MODE11_ENDPOINTS);
				memcpy(pWeights0, trial_weights0, num_pixels);
				memcpy(pWeights1, trial_weights1, num_pixels);
				submode_used = best_submode + 1;
				improved_flag = true;
			}
		}
	}

	return improved_flag;
}

//--------------------------------------------------------------------------------------------------------------------------

bool pack_mode7(
	const vec3F& high_color_q16, const float s_q16,
	uint32_t ise_endpoint_range, uint8_t* pEndpoints,
	uint32_t ise_weight_range, // only used for determining biasing during packing
	const astc_hdr_codec_base_options& coptions,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping, uint32_t& submode_used)
{
	assert(first_submode <= last_submode);
	assert((first_submode >= 0) && (first_submode <= (int)MAX_MODE7_SUBMODE_INDEX));
	assert(last_submode <= (int)MAX_MODE7_SUBMODE_INDEX);

	uint8_t unquant_trial_endpoints[NUM_MODE7_ENDPOINTS];

	memset(pEndpoints, 0, NUM_MODE7_ENDPOINTS);

	double best_trial_dist = BIG_FLOAT_VAL;
	int best_trial_submode = 0;

	for (int submode = first_submode; submode <= last_submode; submode++)
	{
		const int MAX_CLAMP_MAG_ACCEPT_THRESH = 16;

		int max_clamp_mag = 0;
		const bool did_clamp = pack_astc_mode7_submode(submode, unquant_trial_endpoints, high_color_q16, s_q16, max_clamp_mag, ise_weight_range, !ignore_clamping, MAX_CLAMP_MAG_ACCEPT_THRESH);

		if (submode < 5)
		{
			if (!ignore_clamping)
			{
				if ((did_clamp) && (max_clamp_mag > MAX_CLAMP_MAG_ACCEPT_THRESH))
					continue;
			}
		}

		uint8_t trial_endpoints[NUM_MODE7_ENDPOINTS];

		// This will distort the endpoints if the ISE endpoint range isn't 256 levels (20).
		// It could massively distort the endpoints, but still result in a valid encoding.
		basist::astc_6x6_hdr::requantize_ise_endpoints(7, astc_helpers::BISE_256_LEVELS, unquant_trial_endpoints, ise_endpoint_range, trial_endpoints);

		int e[2][3];
		int decoded_s = 0;
		if (!decode_mode7_to_qlog12(trial_endpoints, e, &decoded_s, ise_endpoint_range))
			continue;

		// e1 is always the high color
		vec3F e1(
			(float)(e[1][0] << 4),
			(float)(e[1][1] << 4),
			(float)(e[1][2] << 4)
		);

		decoded_s <<= 4;

		double dist = e1.squared_distance_d(high_color_q16) + squared((double)decoded_s - s_q16) * 3;

		if (dist < best_trial_dist)
		{
			best_trial_dist = dist;
			best_trial_submode = submode;
			memcpy(pEndpoints, trial_endpoints, NUM_MODE7_ENDPOINTS);
		}

		if (coptions.m_take_first_non_clamping_mode7_submode)
		{
			if (!did_clamp)
				break;
		}

	} // submode

	if ((coptions.m_ultra_quant) &&
		(ise_endpoint_range < astc_helpers::BISE_256_LEVELS) &&
		(best_trial_dist != BIG_FLOAT_VAL))
	{
		uint8_t orig_best_trial_endpoints[NUM_MODE7_ENDPOINTS];
		memcpy(orig_best_trial_endpoints, pEndpoints, NUM_MODE7_ENDPOINTS);

		vec3F low_color_q16(high_color_q16 - vec3F(s_q16));
		low_color_q16.clamp(0.0f, 65535.0f);

		for (uint32_t c = 0; c < NUM_MODE7_ENDPOINTS; c++)
		{
			for (int dt = 0; dt <= 1; dt++)
			{
				const int d = dt ? 1 : -1;

				uint8_t varied_endpoints[NUM_MODE7_ENDPOINTS];
				memcpy(varied_endpoints, orig_best_trial_endpoints, NUM_MODE7_ENDPOINTS);

				int ise = varied_endpoints[c];

				int rank = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_ISE_to_rank[ise];
				rank = clamp<int>(rank + d, 0, astc_helpers::get_ise_levels(ise_endpoint_range) - 1);

				ise = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_rank_to_ISE[rank];

				varied_endpoints[c] = (uint8_t)ise;

				int e[2][3];
				int decoded_s = 0;
				if (!decode_mode7_to_qlog12(varied_endpoints, e, &decoded_s, ise_endpoint_range))
					continue;

				// e1 is always the high color
				vec3F e1(
					(float)(e[1][0] << 4),
					(float)(e[1][1] << 4),
					(float)(e[1][2] << 4)
				);

				decoded_s <<= 4;

				double dist = e1.squared_distance_d(high_color_q16) + squared((double)decoded_s - s_q16) * 3;

				if (dist < best_trial_dist)
				{
					best_trial_dist = dist;
					memcpy(pEndpoints, varied_endpoints, NUM_MODE7_ENDPOINTS);
				}

			} // d
		} // c
	}

	submode_used = best_trial_submode;

	return (best_trial_dist != BIG_FLOAT_VAL);
}

//--------------------------------------------------------------------------------------------------------------------------

bool try_mode7(
	uint32_t num_pixels,
	uint8_t* pEndpoints, uint8_t* pWeights, double& cur_block_error, uint32_t& submode_used,
	const vec3F& high_color_q16, const float s_q16,
	const half_float block_pixels_half[][3],
	uint32_t num_weight_levels, uint32_t ise_weight_range, const astc_hdr_codec_base_options& coptions,
	uint32_t ise_endpoint_range,
	int32_t first_submode, int32_t last_submode)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));

	assert(first_submode <= last_submode);
	assert((first_submode >= 0) && (first_submode <= (int)MAX_MODE7_SUBMODE_INDEX));
	assert(last_submode <= (int)MAX_MODE7_SUBMODE_INDEX);
	assert(num_weight_levels == astc_helpers::get_ise_levels(ise_weight_range));

	uint8_t unquant_trial_endpoints[NUM_MODE7_ENDPOINTS];

	uint8_t best_trial_endpoints[NUM_MODE7_ENDPOINTS];
	clear_obj(best_trial_endpoints);
	double best_trial_dist = BIG_FLOAT_VAL;
	int best_trial_submode = 0;
		
	for (int submode = first_submode; submode <= last_submode; submode++)
	{
		const int MAX_CLAMP_MAG_ACCEPT_THRESH = 16;

		int max_clamp_mag = 0;
		const bool did_clamp = pack_astc_mode7_submode(submode, unquant_trial_endpoints, high_color_q16, s_q16, max_clamp_mag, ise_weight_range, true, MAX_CLAMP_MAG_ACCEPT_THRESH);

		if (submode < 5)
		{
			if ((did_clamp) && (max_clamp_mag > MAX_CLAMP_MAG_ACCEPT_THRESH))
				continue;
		}

		uint8_t trial_endpoints[NUM_MODE7_ENDPOINTS];

		// This will distort the endpoints if the ISE endpoint range isn't 256 levels (20).
		// It could massively distort the endpoints, but still result in a valid encoding.
		basist::astc_6x6_hdr::requantize_ise_endpoints(7, astc_helpers::BISE_256_LEVELS, unquant_trial_endpoints, ise_endpoint_range, trial_endpoints);

		int e[2][3];
		int decoded_s = 0;
		if (!decode_mode7_to_qlog12(trial_endpoints, e, &decoded_s, ise_endpoint_range))
			continue;

		// e1 is always the high color
		vec3F e1(
			(float)(e[1][0] << 4),
			(float)(e[1][1] << 4),
			(float)(e[1][2] << 4)
		);

		decoded_s <<= 4;

		double dist = e1.squared_distance_d(high_color_q16) + squared((double)decoded_s - s_q16) * 3;

		if (dist < best_trial_dist)
		{
			best_trial_dist = dist;
			best_trial_submode = submode;
			memcpy(best_trial_endpoints, trial_endpoints, sizeof(best_trial_endpoints));
		}

		if (coptions.m_take_first_non_clamping_mode7_submode)
		{
			if (!did_clamp)
				break;
		}

	} // submode

	if ((coptions.m_ultra_quant) &&
		(ise_endpoint_range < astc_helpers::BISE_256_LEVELS) &&
		(best_trial_dist != BIG_FLOAT_VAL))
	{
		uint8_t orig_best_trial_endpoints[NUM_MODE7_ENDPOINTS];
		memcpy(orig_best_trial_endpoints, best_trial_endpoints, NUM_MODE7_ENDPOINTS);

		vec3F low_color_q16(high_color_q16 - vec3F(s_q16));
		low_color_q16.clamp(0.0f, 65535.0f);

		for (uint32_t c = 0; c < NUM_MODE7_ENDPOINTS; c++)
		{
			for (int dt = 0; dt <= 1; dt++)
			{
				const int d = dt ? 1 : -1;

				uint8_t varied_endpoints[NUM_MODE7_ENDPOINTS];
				memcpy(varied_endpoints, orig_best_trial_endpoints, NUM_MODE7_ENDPOINTS);

				int ise = varied_endpoints[c];

				int rank = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_ISE_to_rank[ise];
				rank = clamp<int>(rank + d, 0, astc_helpers::get_ise_levels(ise_endpoint_range) - 1);

				ise = astc_helpers::g_dequant_tables.get_endpoint_tab(ise_endpoint_range).m_rank_to_ISE[rank];

				varied_endpoints[c] = (uint8_t)ise;

				int e[2][3];
				int decoded_s = 0;
				if (!decode_mode7_to_qlog12(varied_endpoints, e, &decoded_s, ise_endpoint_range))
					continue;

				// e1 is always the high color
				vec3F e1(
					(float)(e[1][0] << 4),
					(float)(e[1][1] << 4),
					(float)(e[1][2] << 4)
				);

				decoded_s <<= 4;

				double dist = e1.squared_distance_d(high_color_q16) + squared((double)decoded_s - s_q16) * 3;

				if (dist < best_trial_dist)
				{
					best_trial_dist = dist;
					memcpy(best_trial_endpoints, varied_endpoints, NUM_MODE7_ENDPOINTS);
				}

			} // d
		} // c
	}

	bool improved_flag = false;

	if (best_trial_dist != BIG_FLOAT_VAL)
	{
		half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];
		uint8_t trial_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

		if (get_astc_hdr_mode_7_block_colors(best_trial_endpoints, &decoded_half[0][0], nullptr, num_weight_levels, ise_weight_range, ise_endpoint_range))
		{
			double trial_blk_error = eval_selectors(num_pixels, trial_weights, ise_weight_range, &block_pixels_half[0][0], num_weight_levels, &decoded_half[0][0], coptions);
			if (trial_blk_error < cur_block_error)
			{
				cur_block_error = trial_blk_error;
				memcpy(pEndpoints, best_trial_endpoints, NUM_MODE7_ENDPOINTS);
				memcpy(pWeights, trial_weights, num_pixels);
				submode_used = best_trial_submode;
				improved_flag = true;
			}
		}
	}

	return improved_flag;
}

//--------------------------------------------------------------------------------------------------------------------------
const float LOW_EMPHASIS_WEIGHT = 1.0f, MIDDLE_EMPHASIS_WEIGHT = 1.25f, HIGH_EMPHASIS_WEIGHT = 1.0f;
const float LOW_EMPHASIS_WEIGHT_HEAVY = 1.0f, MIDDLE_EMPHASIS_WEIGHT_HEAVY = 4.0f, HIGH_EMPHASIS_WEIGHT_HEAVY = 1.0f;

double encode_astc_hdr_block_mode_11(
	uint32_t num_pixels,
	const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	uint32_t ise_weight_range,
	uint32_t& best_submode,
	double cur_block_error,
	uint8_t* blk_endpoints, uint8_t* blk_weights,
	const astc_hdr_codec_base_options& coptions,
	bool direct_only,
	uint32_t ise_endpoint_range,
	bool uber_mode,
	bool constrain_ise_weight_selectors,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping, opt_mode_t opt_mode,
	const encode_astc_block_stats* pBlock_stats)
{
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((ise_endpoint_range >= astc_helpers::FIRST_VALID_ENDPOINT_ISE_RANGE) && (ise_endpoint_range <= astc_helpers::LAST_VALID_ENDPOINT_ISE_RANGE));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));

	assert((first_submode >= FIRST_MODE11_SUBMODE_INDEX) && (first_submode <= last_submode));
	assert(last_submode <= MAX_MODE11_SUBMODE_INDEX);

	best_submode = 0;

	const uint32_t num_weight_levels = astc_helpers::get_ise_levels(ise_weight_range);
	assert(num_weight_levels <= MAX_SUPPORTED_WEIGHT_LEVELS);

	vec3F block_mean_color_q16, block_axis_q16;
	if (!pBlock_stats)
	{
		block_mean_color_q16 = calc_mean(num_pixels, pBlock_pixels_q16);
		block_axis_q16 = calc_rgb_pca(num_pixels, pBlock_pixels_q16, block_mean_color_q16);
	}
	else
	{
		assert(num_pixels == pBlock_stats->m_num_pixels);
		block_mean_color_q16 = pBlock_stats->m_mean_q16;
		block_axis_q16 = pBlock_stats->m_axis_q16;
	}

	aabb3F color_box_q16(cInitExpand);

	float l = BIG_FLOAT_VAL, h = -BIG_FLOAT_VAL;
	vec3F low_color_q16, high_color_q16;

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		color_box_q16.expand(pBlock_pixels_q16[i]);

		vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
		float kd = k.dot(block_axis_q16);

		if (kd < l)
		{
			l = kd;
			low_color_q16 = pBlock_pixels_q16[i];
		}

		if (kd > h)
		{
			h = kd;
			high_color_q16 = pBlock_pixels_q16[i];
		}
	}
		
	vec3F old_low_color_q16(low_color_q16), old_high_color_q16(high_color_q16);
	
	for (uint32_t i = 0; i < 3; i++)
	{
		low_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 1.0f / 64.0f);
		high_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 63.0f / 64.0f);
	}

	uint8_t trial_blk_endpoints[NUM_MODE11_ENDPOINTS];
	uint8_t trial_blk_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	uint32_t trial_best_submode = 0;

	clear_obj(trial_blk_endpoints);
	clear_obj(trial_blk_weights);

	double trial_blk_error = BIG_FLOAT_VAL;
			
	bool did_improve = try_mode11(num_pixels, trial_blk_endpoints, trial_blk_weights, trial_blk_error, trial_best_submode,
		low_color_q16, high_color_q16,
		pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
		first_submode, last_submode, ignore_clamping);

	// If we couldn't find ANY usable solution due to endpoint quantization, just return. There's nothing we can do.
	if (!did_improve)
		return cur_block_error;

	// Did the solution improve?
	if (trial_blk_error < cur_block_error)
	{
		cur_block_error = trial_blk_error;
		memcpy(blk_endpoints, trial_blk_endpoints, NUM_MODE11_ENDPOINTS);
		memcpy(blk_weights, trial_blk_weights, num_pixels);
		best_submode = trial_best_submode;
	}

	if (opt_mode == cNoOpt)
		return cur_block_error;

	// least squares on the most promising trial weight indices found
	const uint32_t NUM_LS_PASSES = 3;

	float emphasis_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

	if (opt_mode == cWeightedAverage)
	{
		const uint32_t NUM_OPT_PASSES = 3;
		for (uint32_t pass = 0; pass < NUM_OPT_PASSES; pass++)
		{
			vec3F low_p(0.0f);
			float total_low = 0.0f;

			vec3F high_p(0.0f);
			float total_high = 0.0f;

			for (uint32_t i = 0; i < num_pixels; i++)
			{
				vec3F p(pBlock_pixels_q16[i]);
				float lerp = g_ise_weight_lerps[ise_weight_range][trial_blk_weights[i] + 1] * (1.0f / 64.0f);

				low_p += p * (1.0f - lerp);
				total_low += (1.0f - lerp);

				high_p += p * lerp;
				total_high += lerp;
			}

			if (total_low != 0.0f)
				low_p *= (1.0f / total_low);

			if (total_high != 0.0f)
				high_p *= (1.0f / total_high);

			vec3F low, high;

			bool was_improved = try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
				low_p, high_p,
				pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
				first_submode, last_submode, ignore_clamping);

			if (!was_improved)
				break;

			memcpy(trial_blk_weights, blk_weights, num_pixels);
		}
	}
	else if (opt_mode == cOrdinaryLeastSquares)
	{
		for (uint32_t pass = 0; pass < NUM_LS_PASSES; pass++)
		{
			vec3F l_q16, h_q16;

			if (!compute_least_squares_endpoints_rgb(num_pixels, trial_blk_weights, &g_astc_ls_weights_ise[ise_weight_range][0], &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16))
				break;
			
			bool was_improved = try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
				l_q16, h_q16,
				pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
				first_submode, last_submode, ignore_clamping);

			if (!was_improved)
				break;

			// It's improved, so let's take the new weight indices.
			memcpy(trial_blk_weights, blk_weights, num_pixels);

		} // pass
	}
	else
	{
		if (h == l)
		{
			for (uint32_t i = 0; i < num_pixels; i++)
				emphasis_weights[i] = 1.0f;
		}
		else
		{
			float mid = (0.0f - l) / (h - l);
			mid = clamp(mid, .01f, .99f);

			float lw = LOW_EMPHASIS_WEIGHT, mw = MIDDLE_EMPHASIS_WEIGHT, hw = HIGH_EMPHASIS_WEIGHT;
			if (opt_mode == cWeightedLeastSquaresHeavy)
				lw = LOW_EMPHASIS_WEIGHT_HEAVY, mw = MIDDLE_EMPHASIS_WEIGHT_HEAVY, hw = HIGH_EMPHASIS_WEIGHT_HEAVY;
						
			for (uint32_t i = 0; i < num_pixels; i++)
			{
				vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
				float kd = k.dot(block_axis_q16);

				assert((kd >= l) && (kd <= h));

				float v = (kd - l) / (h - l);
										
				if (v < mid)
					v = lerp(lw, mw, v / mid);
				else
					v = lerp(mw, hw, (v - mid) * (1.0f - mid));

				emphasis_weights[i] = v;
			}

#if 0
			if (num_pixels == 6 * 6)
			{
				const float EDGE_WEIGHT = .1f;
				for (uint32_t i = 0; i < 6; i++)
				{
					emphasis_weights[i] += EDGE_WEIGHT;
					emphasis_weights[i + 5 * 6] += EDGE_WEIGHT;
					emphasis_weights[i * 6] += EDGE_WEIGHT;
					emphasis_weights[5 + i * 6] += EDGE_WEIGHT;
				}
			}
#endif
		}

		for (uint32_t pass = 0; pass < NUM_LS_PASSES; pass++)
		{
			vec3F l_q16, h_q16;

			if (!compute_weighted_least_squares_endpoints_rgb(
				num_pixels,
				trial_blk_weights, &g_astc_ls_weights_ise[ise_weight_range][0], nullptr,
				emphasis_weights,
				&l_q16, &h_q16,
				pBlock_pixels_q16,
				color_box_q16))
				break;

			bool was_improved = try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
				l_q16, h_q16,
				pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
				first_submode, last_submode, ignore_clamping);

			if (!was_improved)
				break;

			// It's improved, so let's take the new weight indices.
			memcpy(trial_blk_weights, blk_weights, num_pixels);

		} // pass
	}

	if ( (uber_mode) && (ise_weight_range >= astc_helpers::BISE_3_LEVELS) &&
		((opt_mode == cOrdinaryLeastSquares) || (opt_mode == cWeightedLeastSquares) || (opt_mode == cWeightedLeastSquaresHeavy)) )
	{
		// Try varying the current best weight indices. This can be expanded/improved, but at potentially great cost.

		uint8_t temp_astc_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
		memcpy(temp_astc_weights, trial_blk_weights, num_pixels);

		uint32_t min_lin_sel = 256, max_lin_sel = 0;
		for (uint32_t i = 0; i < num_pixels; i++)
		{
			const uint32_t astc_sel = temp_astc_weights[i];

			const uint32_t lin_sel = g_map_astc_to_linear_order[ise_weight_range][astc_sel];
			assert(lin_sel < num_weight_levels);

			min_lin_sel = minimumu(min_lin_sel, lin_sel);
			max_lin_sel = maximumu(max_lin_sel, lin_sel);
		}

		bool was_improved = false;
		(void)was_improved;

		{
			bool weights_changed = false;
			uint8_t trial_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
			for (uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t astc_sel = temp_astc_weights[i];
				uint32_t lin_sel = g_map_astc_to_linear_order[ise_weight_range][astc_sel];

				if ((lin_sel == min_lin_sel) && (lin_sel < (num_weight_levels - 1)))
				{
					lin_sel++;
					weights_changed = true;
				}

				trial_weights[i] = g_map_linear_to_astc_order[ise_weight_range][lin_sel];
			}

			if (weights_changed)
			{
				vec3F l_q16, h_q16;

				bool succeeded;
				if (opt_mode == cOrdinaryLeastSquares)
					succeeded = compute_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);
				else
					succeeded = compute_weighted_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], nullptr, emphasis_weights, &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);

				if (succeeded)
				{
					if (try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
						l_q16, h_q16,
						pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
						first_submode, last_submode, ignore_clamping))
					{
						was_improved = true;
					}
				}
			}
		}

		{
			bool weights_changed = false;
			uint8_t trial_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

			for (uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t astc_sel = temp_astc_weights[i];
				uint32_t lin_sel = g_map_astc_to_linear_order[ise_weight_range][astc_sel];

				if ((lin_sel == max_lin_sel) && (lin_sel > 0))
				{
					lin_sel--;
					weights_changed = true;
				}

				trial_weights[i] = g_map_linear_to_astc_order[ise_weight_range][lin_sel];
			}

			if (weights_changed)
			{
				vec3F l_q16, h_q16;

				bool succeeded;
				if (opt_mode == cOrdinaryLeastSquares)
					succeeded = compute_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);
				else
					succeeded = compute_weighted_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], nullptr, emphasis_weights, &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);

				if (succeeded)
				{
					if (try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
						l_q16, h_q16,
						pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
						first_submode, last_submode, ignore_clamping))
					{
						was_improved = true;
					}
				}
			}
		}

		{
			bool weights_changed = false;
			uint8_t trial_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
			for (uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t astc_sel = temp_astc_weights[i];
				uint32_t lin_sel = g_map_astc_to_linear_order[ise_weight_range][astc_sel];

				if ((lin_sel == max_lin_sel) && (lin_sel > 0))
				{
					lin_sel--;
					weights_changed = true;
				}
				else if ((lin_sel == min_lin_sel) && (lin_sel < (num_weight_levels - 1)))
				{
					lin_sel++;
					weights_changed = true;
				}

				trial_weights[i] = g_map_linear_to_astc_order[ise_weight_range][lin_sel];
			}

			if (weights_changed)
			{
				vec3F l_q16, h_q16;
				bool succeeded;
				if (opt_mode == cOrdinaryLeastSquares)
					succeeded = compute_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);
				else
					succeeded = compute_weighted_least_squares_endpoints_rgb(num_pixels, trial_weights, &g_astc_ls_weights_ise[ise_weight_range][0], nullptr, emphasis_weights, &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16);

				if (succeeded)
				{
					if (try_mode11(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
						l_q16, h_q16,
						pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
						first_submode, last_submode, ignore_clamping))
					{
						was_improved = true;
					}
				}
			}
		}

	} // uber_mode

	return cur_block_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double encode_astc_hdr_block_downsampled_mode_11(
	uint32_t block_x, uint32_t block_y, uint32_t grid_x, uint32_t grid_y,
	uint32_t ise_weight_range, uint32_t ise_endpoint_range,
	uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	double cur_block_error,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping, opt_mode_t opt_mode,
	uint8_t* pBlk_endpoints, uint8_t* pBlk_weights, uint32_t& best_submode,
	const astc_hdr_codec_base_options& coptions,
	const encode_astc_block_stats* pBlock_stats)
{
	assert((block_x >= 4) && (block_y >= 4) && (block_x <= MAX_ASTC_HDR_BLOCK_W) && (block_y <= MAX_ASTC_HDR_BLOCK_H));
	assert((grid_x >= 2) && (grid_y >= 2) && (grid_x <= block_x) && (grid_y <= block_y));

	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((ise_endpoint_range >= astc_helpers::FIRST_VALID_ENDPOINT_ISE_RANGE) && (ise_endpoint_range <= astc_helpers::LAST_VALID_ENDPOINT_ISE_RANGE));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));

	assert((first_submode >= FIRST_MODE11_SUBMODE_INDEX) && (first_submode <= last_submode));
	assert(last_submode <= MAX_MODE11_SUBMODE_INDEX);

	best_submode = 0;

	assert(astc_helpers::get_ise_levels(ise_weight_range) <= MAX_SUPPORTED_WEIGHT_LEVELS);

	const uint32_t num_weights = grid_x * grid_y;

	vec3F block_mean_color_q16, block_axis_q16;
	if (!pBlock_stats)
	{
		block_mean_color_q16 = calc_mean(num_pixels, pBlock_pixels_q16);
		block_axis_q16 = calc_rgb_pca(num_pixels, pBlock_pixels_q16, block_mean_color_q16);
	}
	else
	{
		assert(num_pixels == pBlock_stats->m_num_pixels);
		block_mean_color_q16 = pBlock_stats->m_mean_q16;
		block_axis_q16 = pBlock_stats->m_axis_q16;
	}

	aabb3F color_box_q16(cInitExpand);

	float l = BIG_FLOAT_VAL, h = -BIG_FLOAT_VAL;
	vec3F low_color_q16, high_color_q16;

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		color_box_q16.expand(pBlock_pixels_q16[i]);

		vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
		float kd = k.dot(block_axis_q16);

		if (kd < l)
		{
			l = kd;
			low_color_q16 = pBlock_pixels_q16[i];
		}

		if (kd > h)
		{
			h = kd;
			high_color_q16 = pBlock_pixels_q16[i];
		}
	}

	vec3F old_low_color_q16(low_color_q16), old_high_color_q16(high_color_q16);

	for (uint32_t i = 0; i < 3; i++)
	{
		low_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 1.0f / 64.0f);
		high_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 63.0f / 64.0f);
	}

	const uint32_t NUM_PASSES = 3;
	for (uint32_t pass = 0; pass < NUM_PASSES; pass++)
	{
		uint8_t trial_blk_endpoints[NUM_MODE11_ENDPOINTS];
		uint8_t trial_blk_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // at block resolution, not grid res
		uint32_t trial_best_submode = 0;

		clear_obj(trial_blk_endpoints);
		clear_obj(trial_blk_weights);
				
		double trial_blk_error = BIG_FLOAT_VAL;

		bool could_pack = try_mode11(num_pixels, trial_blk_endpoints, trial_blk_weights, trial_blk_error, trial_best_submode,
			low_color_q16, high_color_q16,
			pBlock_pixels_half, 32, astc_helpers::BISE_32_LEVELS, coptions, false, ise_endpoint_range, false,
			first_submode, last_submode, ignore_clamping);

		if (!could_pack)
			break;

		uint8_t trial_downsampled_ise_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

		downsample_ise_weights(
			astc_helpers::BISE_32_LEVELS, ise_weight_range,
			block_x, block_y, grid_x, grid_y,
			trial_blk_weights, trial_downsampled_ise_weights);

		uint8_t trial_downsampled_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
		dequantize_astc_weights(num_weights, trial_downsampled_ise_weights, ise_weight_range, trial_downsampled_raw_weights);

		uint8_t trial_upsampled_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // raw weights, NOT ISE
		astc_helpers::upsample_weight_grid(block_x, block_y, grid_x, grid_y, trial_downsampled_raw_weights, trial_upsampled_raw_weights);

		//------

		int trial_e[2][3];
		if (!decode_mode11_to_qlog12(trial_blk_endpoints, trial_e, ise_endpoint_range))
			return cur_block_error;

		double trial_error = compute_block_error_from_raw_weights(num_pixels, pBlock_pixels_half, trial_upsampled_raw_weights, trial_e, coptions);

		if (trial_error < cur_block_error)
		{
			cur_block_error = trial_error;
			memcpy(pBlk_endpoints, trial_blk_endpoints, NUM_MODE11_ENDPOINTS);
			memcpy(pBlk_weights, trial_downsampled_ise_weights, num_weights);
			best_submode = trial_best_submode;
		}
		else if (pass)
			break;
						
		if ((opt_mode == cWeightedLeastSquares) || (opt_mode == cWeightedLeastSquaresHeavy))
		{
			float emphasis_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
			if (h == l)
			{
				for (uint32_t i = 0; i < num_pixels; i++)
					emphasis_weights[i] = 1.0f;
			}
			else
			{
				float mid = (0.0f - l) / (h - l);
				mid = clamp(mid, .01f, .99f);

				float lw = LOW_EMPHASIS_WEIGHT, mw = MIDDLE_EMPHASIS_WEIGHT, hw = HIGH_EMPHASIS_WEIGHT;
				if (opt_mode == cWeightedLeastSquaresHeavy)
					lw = LOW_EMPHASIS_WEIGHT_HEAVY, mw = MIDDLE_EMPHASIS_WEIGHT_HEAVY, hw = HIGH_EMPHASIS_WEIGHT_HEAVY;

				for (uint32_t i = 0; i < num_pixels; i++)
				{
					vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
					float kd = k.dot(block_axis_q16);

					assert((kd >= l) && (kd <= h));

					float v = (kd - l) / (h - l);

					if (v < mid)
						v = lerp(lw, mw, v / mid);
					else
						v = lerp(mw, hw, (v - mid) * (1.0f - mid));

					emphasis_weights[i] = v;
				}
			}

			float trial_upsampled_raw_weightsf[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
			for (uint32_t i = 0; i < num_pixels; i++)
				trial_upsampled_raw_weightsf[i] = (float)trial_upsampled_raw_weights[i] * (1.0f / 64.0f);

			if (!compute_weighted_least_squares_endpoints_rgb(num_pixels, nullptr, nullptr, trial_upsampled_raw_weightsf, emphasis_weights, &low_color_q16, &high_color_q16, pBlock_pixels_q16, color_box_q16))
				return false;
		}
		else
		{
			if (!compute_least_squares_endpoints_rgb_raw_weights(num_pixels, trial_upsampled_raw_weights, &low_color_q16, &high_color_q16, pBlock_pixels_q16, color_box_q16))
				break;
		}

		bool pack_succeeded = pack_mode11(low_color_q16, high_color_q16, ise_endpoint_range, trial_blk_endpoints, coptions, false, first_submode, last_submode, false, trial_best_submode);
		if (!pack_succeeded)
			break;

		if (!decode_mode11_to_qlog12(trial_blk_endpoints, trial_e, ise_endpoint_range))
			break;

		trial_error = compute_block_error_from_raw_weights(num_pixels, pBlock_pixels_half, trial_upsampled_raw_weights, trial_e, coptions);

		if (trial_error < cur_block_error)
		{
			cur_block_error = trial_error;
			memcpy(pBlk_endpoints, trial_blk_endpoints, NUM_MODE11_ENDPOINTS);
			memcpy(pBlk_weights, trial_downsampled_ise_weights, num_weights);
			best_submode = trial_best_submode;
		}
		else
		{
			break;
		}

    } // pass

	return cur_block_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double encode_astc_hdr_block_mode_11_dual_plane(
	uint32_t num_pixels,
	const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	uint32_t channel_index,		// 0-2
	uint32_t ise_weight_range,
	uint32_t& best_submode,
	double cur_block_error,
	uint8_t* blk_endpoints, uint8_t* blk_weights0, uint8_t* blk_weights1,
	const astc_hdr_codec_base_options& coptions,
	bool direct_only,
	uint32_t ise_endpoint_range,
	bool uber_mode,
	bool constrain_ise_weight_selectors,
	int32_t first_submode, int32_t last_submode, bool ignore_clamping)
{
	(void)uber_mode;

	assert(channel_index <= 2);
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((ise_endpoint_range >= astc_helpers::FIRST_VALID_ENDPOINT_ISE_RANGE) && (ise_endpoint_range <= astc_helpers::LAST_VALID_ENDPOINT_ISE_RANGE));
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));

	assert((first_submode >= FIRST_MODE11_SUBMODE_INDEX) && (first_submode <= last_submode));
	assert(last_submode <= MAX_MODE11_SUBMODE_INDEX);
	
	assert(num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS);

	best_submode = 0;

	const uint32_t num_weight_levels = astc_helpers::get_ise_levels(ise_weight_range);
	assert(num_weight_levels <= MAX_SUPPORTED_WEIGHT_LEVELS);

	vec4F temp_block_pixels_q16[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		temp_block_pixels_q16[i] = pBlock_pixels_q16[i];
		temp_block_pixels_q16[i][channel_index] = 0.0f;
	}

	vec3F block_mean_color_q16(calc_mean(num_pixels, temp_block_pixels_q16));
	vec3F block_axis_q16(calc_rgb_pca(num_pixels, temp_block_pixels_q16, block_mean_color_q16));

	float l = BIG_FLOAT_VAL, h = -BIG_FLOAT_VAL;
	vec3F low_color_q16, high_color_q16;

	aabb3F color_box_q16(cInitExpand);

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		color_box_q16.expand(pBlock_pixels_q16[i]);

		vec3F k(vec3F(temp_block_pixels_q16[i]) - block_mean_color_q16);
		float kd = k.dot(block_axis_q16);

		if (kd < l)
		{
			l = kd;
			low_color_q16 = pBlock_pixels_q16[i];
		}

		if (kd > h)
		{
			h = kd;
			high_color_q16 = pBlock_pixels_q16[i];
		}
	}

	low_color_q16[channel_index] = 0.0f;
	high_color_q16[channel_index] = 0.0f;

	float a = low_color_q16.dot(vec3F(1.0f)), b = high_color_q16.dot(vec3F(1.0f));
	if (a <= b)
	{
		low_color_q16[channel_index] = color_box_q16.get_low()[channel_index];
		high_color_q16[channel_index] = color_box_q16.get_high()[channel_index];
	}
	else
	{
		high_color_q16[channel_index] = color_box_q16.get_low()[channel_index];
		low_color_q16[channel_index] = color_box_q16.get_high()[channel_index];
	}

	vec3F old_low_color_q16(low_color_q16), old_high_color_q16(high_color_q16);
	for (uint32_t i = 0; i < 3; i++)
	{
		low_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 1.0f / 64.0f);
		high_color_q16[i] = lerp<float>(old_low_color_q16[i], old_high_color_q16[i], 63.0f / 64.0f);
	}

	uint8_t trial_blk_endpoints[NUM_MODE11_ENDPOINTS];
	uint8_t trial_blk_weights0[MAX_ASTC_HDR_ENC_BLOCK_PIXELS], trial_blk_weights1[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	uint32_t trial_best_submode = 0;

	clear_obj(trial_blk_endpoints);
	clear_obj(trial_blk_weights0);
	clear_obj(trial_blk_weights1);

	double trial_blk_error = BIG_FLOAT_VAL;

	bool did_improve = try_mode11_dual_plane(channel_index, num_pixels, trial_blk_endpoints, trial_blk_weights0, trial_blk_weights1, trial_blk_error, trial_best_submode,
		low_color_q16, high_color_q16, 
		pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
		first_submode, last_submode, ignore_clamping);

	// If we couldn't find ANY usable solution due to endpoint quantization, just return. There's nothing we can do.
	if (!did_improve)
		return cur_block_error;

	// Did the solution improve?
	if (trial_blk_error < cur_block_error)
	{
		cur_block_error = trial_blk_error;
		memcpy(blk_endpoints, trial_blk_endpoints, NUM_MODE11_ENDPOINTS);
		memcpy(blk_weights0, trial_blk_weights0, num_pixels);
		memcpy(blk_weights1, trial_blk_weights1, num_pixels);
		best_submode = trial_best_submode;
	}

	const uint32_t chan0 = (channel_index + 1) % 3, chan1 = (channel_index + 2) % 3;

	vec2F plane0_q16[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	aabb2F plane0_bounds;
	plane0_bounds[0].set(color_box_q16.get_low()[chan0], color_box_q16.get_low()[chan1]);
	plane0_bounds[1].set(color_box_q16.get_high()[chan0], color_box_q16.get_high()[chan1]);

	vec1F plane1_q16[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	aabb1F plane1_bounds;
	plane1_bounds[0].set(color_box_q16.get_low()[channel_index]);
	plane1_bounds[1].set(color_box_q16.get_high()[channel_index]);

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		plane0_q16[i][0] = pBlock_pixels_q16[i][chan0];
		plane0_q16[i][1] = pBlock_pixels_q16[i][chan1];

		plane1_q16[i][0] = pBlock_pixels_q16[i][channel_index];
	}

	const uint32_t NUM_LS_PASSES = 3;

	for (uint32_t pass = 0; pass < NUM_LS_PASSES; pass++)
	{
		vec2F l0_q16, h0_q16;
		if (!compute_least_squares_endpoints_2D(num_pixels, trial_blk_weights0, &g_astc_ls_weights_ise[ise_weight_range][0], &l0_q16, &h0_q16, plane0_q16, plane0_bounds))
			break;

		vec1F l1_q16, h1_q16;
		if (!compute_least_squares_endpoints_1D(num_pixels, trial_blk_weights1, &g_astc_ls_weights_ise[ise_weight_range][0], &l1_q16, &h1_q16, plane1_q16, plane1_bounds))
			break;

		vec3F l_q16, h_q16;

		l_q16[channel_index] = l1_q16[0];
		h_q16[channel_index] = h1_q16[0];

		l_q16[chan0] = l0_q16[0];
		h_q16[chan0] = h0_q16[0];

		l_q16[chan1] = l0_q16[1];
		h_q16[chan1] = h0_q16[1];

		bool was_improved = try_mode11_dual_plane(channel_index, num_pixels, blk_endpoints, blk_weights0, blk_weights1, cur_block_error, best_submode,
			l_q16, h_q16,
			pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, direct_only, ise_endpoint_range, constrain_ise_weight_selectors,
			first_submode, last_submode, ignore_clamping);

		if (!was_improved)
			break;

		// It's improved, so let's take the new weight indices.
		memcpy(trial_blk_weights0, blk_weights0, num_pixels);
		memcpy(trial_blk_weights1, blk_weights1, num_pixels);

	} // pass
	
	return cur_block_error;
}

//--------------------------------------------------------------------------------------------------------------------------

double encode_astc_hdr_block_mode_7(
	uint32_t num_pixels,
	const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	uint32_t ise_weight_range,
	uint32_t& best_submode,
	double cur_block_error,
	uint8_t* blk_endpoints,  //[4]
	uint8_t* blk_weights, // [num_pixels]
	const astc_hdr_codec_base_options& coptions,
	uint32_t ise_endpoint_range, 
	int first_submode, int last_submode,
	const encode_astc_block_stats* pBlock_stats)
{
	assert((num_pixels >= 1) && (num_pixels <= MAX_ASTC_HDR_ENC_BLOCK_PIXELS));
	assert((ise_weight_range >= MIN_SUPPORTED_ISE_WEIGHT_INDEX) && (ise_weight_range <= MAX_SUPPORTED_ISE_WEIGHT_INDEX));
	assert((ise_endpoint_range >= astc_helpers::FIRST_VALID_ENDPOINT_ISE_RANGE) && (ise_endpoint_range <= astc_helpers::LAST_VALID_ENDPOINT_ISE_RANGE));

	const uint32_t num_weight_levels = astc_helpers::get_ise_levels(ise_weight_range);
	assert(num_weight_levels <= MAX_SUPPORTED_WEIGHT_LEVELS);

	best_submode = 0;

	vec3F block_mean_color_q16;
	if (!pBlock_stats)
		block_mean_color_q16 = calc_mean(num_pixels, pBlock_pixels_q16);
	else
	{
		assert(num_pixels == pBlock_stats->m_num_pixels);
		block_mean_color_q16 = pBlock_stats->m_mean_q16;
	}

	vec3F block_axis_q16(0.577350259f);

	aabb3F color_box_q16(cInitExpand);

	float l = BIG_FLOAT_VAL, h = -BIG_FLOAT_VAL;
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		color_box_q16.expand(pBlock_pixels_q16[i]);

		vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
		float kd = k.dot(block_axis_q16);

		l = basisu::minimum<float>(l, kd);
		h = basisu::maximum<float>(h, kd);
	}

	vec3F low_color_q16(interp_color(block_mean_color_q16, block_axis_q16, l, color_box_q16, color_box_q16));
	vec3F high_color_q16(interp_color(block_mean_color_q16, block_axis_q16, h, color_box_q16, color_box_q16));

	low_color_q16.clamp(0.0f, MAX_QLOG16_VAL);
	high_color_q16.clamp(0.0f, MAX_QLOG16_VAL);

	vec3F diff(high_color_q16 - low_color_q16);

	// The mul here (* block_axis_q16[0]) is because the "S" or scale value is subtracted from the high color with a scale of 1.0, 
	// i.e. it's equivalent to a vector of (1,1,1) multiplied by scale before the sub. We want to actually move along the grayscale axis, or (0.577350259, 0.577350259, 0.577350259).
	float s_q16 = diff.dot(block_axis_q16) * block_axis_q16[0];

	uint8_t trial_blk_endpoints[NUM_MODE7_ENDPOINTS];
	uint8_t trial_blk_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	uint32_t trial_best_submode = 0;

	clear_obj(trial_blk_endpoints);
	clear_obj(trial_blk_weights);

	double trial_blk_error = BIG_FLOAT_VAL;

	bool did_improve = try_mode7(num_pixels, trial_blk_endpoints, trial_blk_weights, trial_blk_error, trial_best_submode,
		high_color_q16, ceilf(s_q16),
		pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode);

	// If we couldn't find ANY usable solution due to endpoint quantization, just return. There's nothing we can do.
	if (!did_improve)
	{
		return cur_block_error;
	}

	// Did the solution improve?
	if (trial_blk_error < cur_block_error)
	{
		cur_block_error = trial_blk_error;
		memcpy(blk_endpoints, trial_blk_endpoints, NUM_MODE7_ENDPOINTS);
		memcpy(blk_weights, trial_blk_weights, num_pixels);
		best_submode = trial_best_submode;
	}

#if 1
	{
		//const float TL = 8830.0f;// (float)half_to_qlog16(float_to_half(0.00061f));
		//const float TH = 41600.0f;// (float)half_to_qlog16(float_to_half(40.0f));
		//float zl = minimum<float>(color_box_q16[0][0], color_box_q16[0][1], color_box_q16[0][2]);
		//float zh = minimum<float>(color_box_q16[1][0], color_box_q16[1][1], color_box_q16[1][2]);

		//if ((zl <= TL) && (zh >= TH))
		{
			// Try a simpler technique for artifact reduction
			l = BIG_FLOAT_VAL;
			h = -BIG_FLOAT_VAL;

			vec3F alt_low_color_q16(0.0f), alt_high_color_q16(0.0f);
			for (uint32_t i = 0; i < num_pixels; i++)
			{
				color_box_q16.expand(pBlock_pixels_q16[i]);

				vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
				float kd = k.dot(block_axis_q16);

				if (kd < l)
				{
					alt_low_color_q16 = pBlock_pixels_q16[i];
					l = kd;
				}

				if (kd > h)
				{
					alt_high_color_q16 = pBlock_pixels_q16[i];
					h = kd;
				}
			}

			vec3F old_alt_low_color_q16(alt_low_color_q16);

			for (uint32_t i = 0; i < 3; i++)
				alt_low_color_q16[i] = lerp<float>(old_alt_low_color_q16[i], alt_high_color_q16[i], 1.0f / 64.0f);

			vec3F alt_diff(alt_high_color_q16 - alt_low_color_q16);

			// The mul here (* block_axis_q16[0]) is because the "S" or scale value is subtracted from the high color with a scale of 1.0, 
			// i.e. it's equivalent to a vector of (1,1,1) multiplied by scale before the sub. We want to actually move along the grayscale axis, or (0.577350259, 0.577350259, 0.577350259).
			float alt_s_q16 = alt_diff.dot(block_axis_q16) * block_axis_q16[0];

			try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
				alt_high_color_q16, ceilf(alt_s_q16),
				pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode);
		}
	}
#endif

	const float one_over_num_pixels = 1.0f / (float)num_pixels;

	const uint32_t NUM_TRIALS = 2;
	for (uint32_t trial = 0; trial < NUM_TRIALS; trial++)
	{
		// Given a set of selectors and S, try to compute a better high color
		vec3F new_high_color_q16(block_mean_color_q16);

		int e[2][3];
		int cur_s = 0;
		if (!decode_mode7_to_qlog12(trial_blk_endpoints, e, &cur_s, ise_endpoint_range))
			break;

		cur_s <<= 4;

		for (uint32_t i = 0; i < num_pixels; i++)
		{
			uint32_t astc_sel = trial_blk_weights[i];
			float lerp = g_ise_weight_lerps[ise_weight_range][astc_sel + 1] * (1.0f / 64.0f);

			float k = (float)cur_s * (1.0f - lerp) * one_over_num_pixels;
			new_high_color_q16[0] += k;
			new_high_color_q16[1] += k;
			new_high_color_q16[2] += k;
		}

		bool improved = try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
			new_high_color_q16, (float)cur_s,
			pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode);

		if (improved)
		{
			memcpy(trial_blk_endpoints, blk_endpoints, NUM_MODE7_ENDPOINTS);
			memcpy(trial_blk_weights, blk_weights, num_pixels);
		}

		// Given a set of selectors and a high color, try to compute a better S.
		float t = 0.0f;

		for (uint32_t i = 0; i < num_pixels; i++)
		{
			uint32_t astc_sel = trial_blk_weights[i];
			float lerp = g_ise_weight_lerps[ise_weight_range][astc_sel + 1] * (1.0f / 64.0f);

			t += (1.0f) - lerp;
		}

		t *= one_over_num_pixels;

		//int e[2][3];
		if (!decode_mode7_to_qlog12(trial_blk_endpoints, e, nullptr, ise_endpoint_range))
			break;

		vec3F cur_h_q16((float)(e[1][0] << 4), (float)(e[1][1] << 4), (float)(e[1][2] << 4));

		if (fabs(t) > .0000125f)
		{
			float s_r = (cur_h_q16[0] - block_mean_color_q16[0]) / t;
			float s_g = (cur_h_q16[1] - block_mean_color_q16[1]) / t;
			float s_b = (cur_h_q16[2] - block_mean_color_q16[2]) / t;

			// TODO: gather statistics on these
			if (try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
				cur_h_q16, ceilf(s_r),
				pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode))
			{
				improved = true;
			}

			if (coptions.m_mode7_full_s_optimization)
			{
				if (try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
					cur_h_q16, ceilf(s_g),
					pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode))
				{
					improved = true;
				}

				if (try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
					cur_h_q16, ceilf(s_b),
					pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode))
				{
					improved = true;
				}

				if (try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
					cur_h_q16, ceilf((s_r + s_g + s_b) / 3.0f),
					pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode))
				{
					improved = true;
				}

				// Added this - quite strong.
				if (try_mode7(num_pixels, blk_endpoints, blk_weights, cur_block_error, best_submode,
					cur_h_q16, minimum(maximum(s_r, s_g, s_b) * 1.1f, 65535.0f),
					pBlock_pixels_half, num_weight_levels, ise_weight_range, coptions, ise_endpoint_range, first_submode, last_submode))
				{
					improved = true;
				}
			} // if (coptions.m_mode7_full_s_optimization)

		} // if (fabs(t) > .0000125f)

		if (!improved)
			break;

		memcpy(trial_blk_endpoints, blk_endpoints, NUM_MODE7_ENDPOINTS);
		memcpy(trial_blk_weights, blk_weights, num_pixels);

	} // trial

	return cur_block_error;
}

//--------------------------------------------------------------------------------------------------------------------------

void dequantize_astc_weights(uint32_t n, const uint8_t* pSrc_ise_vals, uint32_t from_ise_range, uint8_t* pDst_raw_weights)
{
	const auto& dequant_tab = astc_helpers::g_dequant_tables.get_weight_tab(from_ise_range).m_ISE_to_val;

	for (uint32_t i = 0; i < n; i++)
		pDst_raw_weights[i] = dequant_tab[pSrc_ise_vals[i]];
}

//--------------------------------------------------------------------------------------------------------------------------

// For each output (2x2) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_2x2[4][36] = {
{0.165438f, 0.132609f, 0.092681f, 0.028953f, 0.000000f, 0.000000f, 0.133716f, 0.111240f, 0.065133f, 0.022236f, 0.000000f, 0.000000f, 0.092623f, 0.063898f, 0.039120f, 0.000000f, 0.000000f, 0.000000f, 0.028168f, 0.024184f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.027262f, 0.091051f, 0.132446f, 0.164791f, 0.000000f, 0.000000f, 0.026038f, 0.066511f, 0.111644f, 0.133197f, 0.000000f, 0.000000f, 0.000000f, 0.040053f, 0.064757f, 0.091196f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.024265f, 0.026789f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028282f, 0.024804f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.092871f, 0.066580f, 0.042024f, 0.000000f, 0.000000f, 0.000000f, 0.132115f, 0.107586f, 0.061943f, 0.025551f, 0.000000f, 0.000000f, 0.166111f, 0.132946f, 0.089043f, 0.030145f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.024535f, 0.028835f, 0.000000f, 0.000000f, 0.000000f, 0.044465f, 0.063652f, 0.093251f, 0.000000f, 0.000000f, 0.025961f, 0.063339f, 0.107329f, 0.132240f, 0.000000f, 0.000000f, 0.029844f, 0.089249f, 0.132200f, 0.165099f},
};

// For each output (3x2) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_3x2[6][36] = {
{0.257933f, 0.144768f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.213754f, 0.109376f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.140969f, 0.064128f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.041270f, 0.027803f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.046066f, 0.153691f, 0.153395f, 0.042845f, 0.000000f, 0.000000f, 0.038497f, 0.131674f, 0.126804f, 0.041513f, 0.000000f, 0.000000f, 0.028434f, 0.081152f, 0.075499f, 0.025372f, 0.000000f, 0.000000f, 0.000000f, 0.030067f, 0.024989f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.147088f, 0.258980f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.105549f, 0.211746f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.066714f, 0.144015f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.027755f, 0.038152f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.044268f, 0.030990f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.141642f, 0.069930f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.207393f, 0.105354f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.255911f, 0.144511f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.026658f, 0.032535f, 0.000000f, 0.000000f, 0.000000f, 0.024618f, 0.079487f, 0.080415f, 0.026311f, 0.000000f, 0.000000f, 0.038382f, 0.133569f, 0.133162f, 0.033451f, 0.000000f, 0.000000f, 0.043697f, 0.152483f, 0.154345f, 0.040885f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.026401f, 0.040228f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.066688f, 0.142350f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.108504f, 0.210286f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.149666f, 0.255876f},
};

// For each output (4x2) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_4x2[8][36] = {
{0.318857f, 0.081413f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.262816f, 0.064811f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.175211f, 0.046152f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.050740f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.163830f, 0.223661f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.128904f, 0.194332f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.080369f, 0.121162f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.041941f, 0.045801f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.230801f, 0.166220f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.193495f, 0.136548f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.113816f, 0.085890f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.043771f, 0.029459f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.087528f, 0.318213f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.059739f, 0.262039f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.046515f, 0.175973f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.049993f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.054078f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.173243f, 0.055145f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.254561f, 0.059695f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.319463f, 0.083816f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.038171f, 0.037447f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.076263f, 0.117360f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.134218f, 0.202503f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.163759f, 0.230278f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.044607f, 0.035170f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.114466f, 0.088407f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.201026f, 0.127983f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.224148f, 0.164194f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.052817f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.043531f, 0.174390f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.060164f, 0.262636f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.089340f, 0.317122f},
};

// For each output (5x2) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_5x2[10][36] = {
{0.393855f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.327491f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.216089f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062565f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.303101f, 0.078223f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.261199f, 0.068761f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.160056f, 0.054634f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.074026f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.202529f, 0.207447f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.151013f, 0.157673f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.100074f, 0.095239f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.043623f, 0.042402f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.083336f, 0.309647f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.061432f, 0.269582f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.046328f, 0.166035f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.063640f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.397684f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.326178f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.217856f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.058282f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.065541f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.215996f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.321124f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.397338f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.069030f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.159434f, 0.051902f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.266327f, 0.065732f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.305627f, 0.081948f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.038550f, 0.046259f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.092606f, 0.100038f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.162523f, 0.163345f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.199767f, 0.196912f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.066709f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.050841f, 0.169003f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.061591f, 0.265094f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.081426f, 0.305335f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.063517f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.210896f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.316133f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.027674f, 0.381781f},
};

// For each output (6x2) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_6x2[12][36] = {
{0.395563f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.328397f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.214936f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.061104f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.395041f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.323513f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.208086f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.073360f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.393200f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.317339f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.218679f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.070782f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.399071f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.321356f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.214689f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.064883f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.399159f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.326009f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.212426f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062406f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398973f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.326510f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.217446f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.057071f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.065386f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.215039f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.321113f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398462f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.072234f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.211515f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.319185f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.397066f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.053184f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.213286f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.332634f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.400895f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.063501f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.207210f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.334096f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.395193f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.074315f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.216723f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.320827f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.388135f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.063571f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.215814f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.325843f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.394772f},
};

// For each output (2x3) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_2x3[6][36] = {
{0.253933f, 0.211745f, 0.142964f, 0.043509f, 0.000000f, 0.000000f, 0.146094f, 0.108119f, 0.068727f, 0.024908f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.043336f, 0.140540f, 0.208745f, 0.253069f, 0.000000f, 0.000000f, 0.031333f, 0.069242f, 0.108596f, 0.145138f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.044780f, 0.036916f, 0.026808f, 0.000000f, 0.000000f, 0.000000f, 0.151455f, 0.129189f, 0.076266f, 0.030885f, 0.000000f, 0.000000f, 0.151915f, 0.131628f, 0.081598f, 0.031903f, 0.000000f, 0.000000f, 0.043838f, 0.032645f, 0.030173f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028998f, 0.038454f, 0.046460f, 0.000000f, 0.000000f, 0.033717f, 0.076274f, 0.130140f, 0.153377f, 0.000000f, 0.000000f, 0.025762f, 0.077843f, 0.130195f, 0.150217f, 0.000000f, 0.000000f, 0.000000f, 0.029422f, 0.034493f, 0.044648f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.145243f, 0.107655f, 0.062280f, 0.033041f, 0.000000f, 0.000000f, 0.257369f, 0.210260f, 0.139667f, 0.044485f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.037604f, 0.064104f, 0.105759f, 0.144848f, 0.000000f, 0.000000f, 0.042699f, 0.141511f, 0.207704f, 0.255772f},
};

// For each output (3x3) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_3x3[9][36] = {
{0.412913f, 0.237773f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.237370f, 0.111944f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.066531f, 0.251421f, 0.245639f, 0.065785f, 0.000000f, 0.000000f, 0.047059f, 0.143642f, 0.128760f, 0.051164f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.234587f, 0.419421f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.110765f, 0.235227f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.067391f, 0.044131f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.248992f, 0.133218f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.247568f, 0.139987f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.072238f, 0.046475f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.040674f, 0.048555f, 0.000000f, 0.000000f, 0.000000f, 0.049640f, 0.158199f, 0.158521f, 0.046044f, 0.000000f, 0.000000f, 0.043591f, 0.153956f, 0.155258f, 0.049378f, 0.000000f, 0.000000f, 0.000000f, 0.046674f, 0.049509f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.049528f, 0.063611f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.137662f, 0.252612f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.134924f, 0.246668f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.042655f, 0.072341f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.237403f, 0.114850f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.418506f, 0.229241f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.049009f, 0.142093f, 0.136891f, 0.036294f, 0.000000f, 0.000000f, 0.074433f, 0.244437f, 0.251631f, 0.065212f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.121166f, 0.231108f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.236230f, 0.411495f},
};

// For each output (4x3) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_4x3[12][36] = {
{0.508292f, 0.132529f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.285382f, 0.073798f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.266624f, 0.378457f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.144380f, 0.210539f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.380292f, 0.270590f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.200825f, 0.148293f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.130560f, 0.507542f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.071578f, 0.290320f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.094051f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.322294f, 0.082665f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.316365f, 0.092271f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.092353f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.046081f, 0.061377f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.158151f, 0.235006f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.152896f, 0.232594f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.052844f, 0.061053f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.061619f, 0.046867f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.227763f, 0.158202f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.222620f, 0.155545f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.073398f, 0.053986f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.082287f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.084098f, 0.330283f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.085224f, 0.323658f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.094451f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.286413f, 0.077046f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.512915f, 0.123625f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.140389f, 0.213324f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.267125f, 0.379163f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.208464f, 0.139969f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.382876f, 0.268691f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.080416f, 0.285653f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.131803f, 0.502128f},
};

// For each output (5x3) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_5x3[15][36] = {
{0.618662f, 0.032137f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.349200f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.497060f, 0.129255f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.281642f, 0.092043f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.333166f, 0.338337f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.164333f, 0.164165f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.129409f, 0.504176f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.085525f, 0.280890f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.636943f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.363057f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.113467f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.394204f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.386741f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.105588f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.086925f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.317750f, 0.095763f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.321008f, 0.086368f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.092185f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.057696f, 0.061462f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.184995f, 0.197656f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.186342f, 0.186715f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.059712f, 0.065422f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.091939f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.079906f, 0.328876f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.085955f, 0.320229f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.093096f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.099585f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398489f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.388782f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.113144f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.360655f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.639345f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.285578f, 0.088663f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.495946f, 0.129812f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.177513f, 0.166195f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.329950f, 0.326342f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.082692f, 0.279744f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.134353f, 0.503211f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.361178f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.638822f},
};

// For each output (6x3) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_6x3[18][36] = {
{0.640623f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359377f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.638697f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.361303f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.640672f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359328f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.637721f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.362279f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.647342f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.352658f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.638418f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.361582f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.111041f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.395972f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.387932f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.105054f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.101949f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.395728f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.401263f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.101060f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.098132f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.388180f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.402030f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.111659f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.096173f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.393865f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.386312f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.123650f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.104357f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398062f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.393265f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.104316f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.097666f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.400772f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.390396f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.111166f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359466f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.640534f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.360569f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.639431f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.355750f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.644250f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.353865f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.646135f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.357727f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.642273f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359539f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.640461f},
};

// For each output (2x4) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_2x4[8][36] = {
{0.312206f, 0.261492f, 0.177496f, 0.055798f, 0.000000f, 0.000000f, 0.081944f, 0.062361f, 0.048703f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.054679f, 0.172805f, 0.260561f, 0.314742f, 0.000000f, 0.000000f, 0.000000f, 0.049040f, 0.065652f, 0.082520f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.164115f, 0.129589f, 0.083879f, 0.029309f, 0.000000f, 0.000000f, 0.231202f, 0.198851f, 0.118719f, 0.044334f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.035855f, 0.083276f, 0.127764f, 0.166965f, 0.000000f, 0.000000f, 0.045347f, 0.116503f, 0.193645f, 0.230645f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.223790f, 0.194804f, 0.115855f, 0.047371f, 0.000000f, 0.000000f, 0.164616f, 0.125798f, 0.087268f, 0.040497f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.044738f, 0.118365f, 0.198854f, 0.230745f, 0.000000f, 0.000000f, 0.029646f, 0.078141f, 0.131405f, 0.168106f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.080206f, 0.060505f, 0.041197f, 0.000000f, 0.000000f, 0.000000f, 0.320486f, 0.265233f, 0.174992f, 0.057380f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.051057f, 0.058139f, 0.082120f, 0.000000f, 0.000000f, 0.056168f, 0.174118f, 0.260525f, 0.317873f},
};

// For each output (3x4) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_3x4[12][36] = {
{0.503381f, 0.288537f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.130806f, 0.077275f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.088808f, 0.319226f, 0.312498f, 0.086797f, 0.000000f, 0.000000f, 0.000000f, 0.092065f, 0.079421f, 0.021185f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.286250f, 0.514036f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.072999f, 0.126714f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.261935f, 0.133191f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.376226f, 0.207118f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.021529f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.059585f, 0.153016f, 0.152552f, 0.043373f, 0.000000f, 0.000000f, 0.063990f, 0.231504f, 0.235283f, 0.060696f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.146403f, 0.262394f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.208547f, 0.382656f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.374676f, 0.209306f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.270440f, 0.145577f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.059636f, 0.233975f, 0.235944f, 0.069029f, 0.000000f, 0.000000f, 0.048950f, 0.150198f, 0.154340f, 0.047929f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.200921f, 0.380881f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.146928f, 0.271271f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.128883f, 0.075468f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.509859f, 0.285791f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.095842f, 0.086878f, 0.000000f, 0.000000f, 0.000000f, 0.092942f, 0.314169f, 0.319263f, 0.090906f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.079652f, 0.124852f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.289868f, 0.505628f},
};

// For each output (4x4) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_4x4[16][36] = {
{0.665277f, 0.167914f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.166809f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.325854f, 0.449938f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.094690f, 0.129518f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.455174f, 0.326025f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.109174f, 0.109627f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.166733f, 0.664155f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.169112f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.320619f, 0.090788f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.462066f, 0.126527f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.165890f, 0.235855f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.233931f, 0.364324f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.239319f, 0.151533f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.363629f, 0.245519f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.106763f, 0.311932f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.119451f, 0.461853f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.451893f, 0.124086f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.326160f, 0.097861f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.239712f, 0.365585f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.164178f, 0.230525f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.360274f, 0.237862f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.246139f, 0.155726f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.121863f, 0.457051f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.097828f, 0.323258f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.163634f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.667648f, 0.168718f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.094870f, 0.132660f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.316878f, 0.455591f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.116917f, 0.098433f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.458816f, 0.325834f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.168403f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.172019f, 0.659578f},
};

// For each output (5x4) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_5x4[20][36] = {
{0.773702f, 0.033711f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.192588f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.633422f, 0.166577f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.170080f, 0.029921f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.388335f, 0.403694f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.100996f, 0.106975f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.161122f, 0.655288f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.183590f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.801705f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.198295f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.400989f, 0.025097f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.573915f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.309345f, 0.085396f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.478694f, 0.126565f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.194664f, 0.187267f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.292735f, 0.308960f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.016375f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.098049f, 0.295983f, 0.000000f, 0.000000f, 0.017892f, 0.000000f, 0.111938f, 0.476138f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.043545f, 0.386448f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.570007f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.566407f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.402307f, 0.031286f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.463145f, 0.120696f, 0.000000f, 0.019497f, 0.000000f, 0.000000f, 0.311721f, 0.084942f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.296730f, 0.300781f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.204639f, 0.197849f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.122117f, 0.469302f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.102545f, 0.306036f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.562064f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.041534f, 0.396403f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.190134f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.773971f, 0.035896f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.169927f, 0.035812f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.630284f, 0.163977f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.112667f, 0.106813f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.393502f, 0.387018f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.177024f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.170482f, 0.652494f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.192274f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033039f, 0.774687f},
};

// For each output (6x4) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_6x4[24][36] = {
{0.804254f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.195746f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.804177f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.195823f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.799585f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.200415f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.803604f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.196396f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.807256f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.192744f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.805135f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.194865f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.410532f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.589468f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.408690f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.591310f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.416225f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.583775f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.414279f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.585721f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.406723f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.593277f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.402510f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.597490f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.584784f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.415216f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.590427f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.409573f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.590073f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.409927f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.580348f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.419652f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.588321f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.411679f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.587022f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.412978f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.193281f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.806719f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.189163f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.810837f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.195108f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.804892f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.188290f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.811710f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.192914f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.807086f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.195292f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.804708f},
};

// For each output (2x5) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_2x5[10][36] = {
{0.387593f, 0.325123f, 0.221104f, 0.066180f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.065940f, 0.214659f, 0.326737f, 0.392664f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.309603f, 0.265953f, 0.168780f, 0.060600f, 0.000000f, 0.000000f, 0.084707f, 0.063017f, 0.047341f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062836f, 0.170767f, 0.261053f, 0.307978f, 0.000000f, 0.000000f, 0.000000f, 0.049286f, 0.064361f, 0.083719f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.195787f, 0.153943f, 0.095706f, 0.042417f, 0.000000f, 0.000000f, 0.190695f, 0.154435f, 0.097288f, 0.040258f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.029471f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.017536f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.039307f, 0.094677f, 0.158696f, 0.199136f, 0.000000f, 0.000000f, 0.040959f, 0.093353f, 0.155294f, 0.201042f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.079432f, 0.065739f, 0.044876f, 0.000000f, 0.000000f, 0.000000f, 0.309205f, 0.264700f, 0.167247f, 0.068801f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.052112f, 0.064829f, 0.081363f, 0.000000f, 0.000000f, 0.064024f, 0.161136f, 0.263743f, 0.312793f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.393277f, 0.324792f, 0.213188f, 0.068743f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.066964f, 0.215440f, 0.323005f, 0.394591f},
};

// For each output (3x5) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_3x5[15][36] = {
{0.620557f, 0.350797f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028646f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.110170f, 0.397489f, 0.386326f, 0.106015f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.357348f, 0.642652f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.503934f, 0.275289f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.128280f, 0.092497f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.102294f, 0.316223f, 0.313576f, 0.092518f, 0.000000f, 0.000000f, 0.000000f, 0.081158f, 0.094231f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.279079f, 0.502163f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.086083f, 0.132675f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.325483f, 0.157739f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.322567f, 0.172225f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.021986f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.063342f, 0.192228f, 0.186950f, 0.057021f, 0.000000f, 0.000000f, 0.054779f, 0.186114f, 0.185666f, 0.073901f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.172195f, 0.331802f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.148212f, 0.322038f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.025751f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.123726f, 0.081188f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.507339f, 0.287746f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.093924f, 0.094021f, 0.000000f, 0.000000f, 0.000000f, 0.097070f, 0.315697f, 0.314560f, 0.084728f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.082560f, 0.129771f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.277014f, 0.486817f, 0.023837f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.644191f, 0.355809f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.107771f, 0.387615f, 0.393454f, 0.111159f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.360886f, 0.639114f},
};

// For each output (4x5) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_4x5[20][36] = {
{0.778254f, 0.190730f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.031016f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.401147f, 0.570243f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028610f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.563768f, 0.394241f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.041992f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.196238f, 0.767548f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.036214f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.637514f, 0.166734f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.167634f, 0.028118f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.322778f, 0.473312f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.085399f, 0.118511f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.471429f, 0.308185f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.118025f, 0.102361f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.176592f, 0.643933f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.179475f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.391609f, 0.100882f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.390531f, 0.116978f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.017259f, 0.000000f, 0.201618f, 0.301555f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.197600f, 0.281968f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.016735f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.293309f, 0.192842f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.268674f, 0.208109f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.020330f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.118514f, 0.380746f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.097621f, 0.381305f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.021814f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.157977f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.657533f, 0.184490f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.097522f, 0.128585f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.309864f, 0.464029f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.128900f, 0.090864f, 0.000000f, 0.025393f, 0.000000f, 0.000000f, 0.464029f, 0.290814f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.024593f, 0.172268f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.173412f, 0.629727f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.029582f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.778816f, 0.191602f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.036297f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.394454f, 0.569249f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.039685f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.561207f, 0.399108f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.034683f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.193744f, 0.771574f},
};

// For each output (5x5) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_5x5[25][36] = {
{1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.794727f, 0.205273f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.465125f, 0.484079f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028881f, 0.000000f, 0.000000f, 0.021914f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.192446f, 0.772941f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.034613f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033123f, 0.930510f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.036367f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.800234f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.199766f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.629079f, 0.165939f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.166390f, 0.019675f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.018918f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.378734f, 0.373861f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.111597f, 0.135808f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.177492f, 0.641195f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.181313f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028722f, 0.761781f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.209497f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.475763f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.471882f, 0.029551f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.022804f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.382714f, 0.116167f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.383377f, 0.117742f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.254151f, 0.249987f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.241972f, 0.253891f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.017950f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.122722f, 0.376847f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.095099f, 0.369986f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.017396f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.029442f, 0.472507f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.471751f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.026300f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.190299f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.776924f, 0.032778f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.171498f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.666385f, 0.162117f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.125713f, 0.117624f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.387084f, 0.369579f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.028493f, 0.169318f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.173770f, 0.628419f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.198951f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.035634f, 0.765415f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.963102f, 0.036898f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.030322f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.771054f, 0.198624f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.021816f, 0.020944f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.481761f, 0.475479f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.032816f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.198418f, 0.768766f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033338f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966662f},
};

// For each output (6x5) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_6x5[30][36] = {
{0.966284f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033716f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.966287f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033713f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.966287f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033713f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.966290f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033710f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966125f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033875f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966273f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033727f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.800857f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.199143f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.773463f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.201165f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.025372f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.805735f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.194265f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.788791f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.211209f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.785975f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.214025f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.787286f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.212714f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.490845f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.487242f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.021913f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.490663f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.486878f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.022459f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.505452f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.494548f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.495383f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.482180f, 0.000000f, 0.022437f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.022727f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.496545f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.480728f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.486261f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.486387f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.027352f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.196272f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.803728f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.210059f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.789941f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.212947f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.787053f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.215261f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.784739f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.209116f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.790884f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.205881f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.794119f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033710f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966290f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033711f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966289f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033713f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966287f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033719f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966281f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033712f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966288f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033712f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966288f},
};

// For each output (2x6) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_2x6[12][36] = {
{0.388815f, 0.325435f, 0.220189f, 0.065562f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.064515f, 0.214042f, 0.327700f, 0.393742f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398821f, 0.326200f, 0.217851f, 0.057128f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062546f, 0.216408f, 0.322269f, 0.398777f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.396575f, 0.330631f, 0.212857f, 0.059936f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.070253f, 0.215326f, 0.317576f, 0.396845f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.398130f, 0.324745f, 0.213572f, 0.063553f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.062009f, 0.216253f, 0.324683f, 0.397055f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.397646f, 0.321346f, 0.212334f, 0.068675f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.067073f, 0.210768f, 0.318165f, 0.403993f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.395756f, 0.325048f, 0.211862f, 0.067334f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.065475f, 0.214113f, 0.324009f, 0.396403f},
};

// For each output (3x6) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_3x6[18][36] = {
{0.640136f, 0.359864f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.108112f, 0.399968f, 0.388087f, 0.103833f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.356122f, 0.643878f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.646308f, 0.353692f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.122937f, 0.390166f, 0.380558f, 0.106339f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.355015f, 0.644985f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.642874f, 0.357126f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.111570f, 0.398638f, 0.387639f, 0.102153f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359134f, 0.640866f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.640159f, 0.359841f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.098908f, 0.393303f, 0.400421f, 0.107369f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.357119f, 0.642881f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.640541f, 0.359459f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.116318f, 0.397635f, 0.395084f, 0.090964f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.361948f, 0.638052f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.645448f, 0.354552f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.106981f, 0.389214f, 0.395056f, 0.108749f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.359592f, 0.640408f},
};

// For each output (4x6) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_4x6[24][36] = {
{0.806928f, 0.193072f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.412216f, 0.587784f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.590075f, 0.409925f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.200682f, 0.799318f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.809822f, 0.190178f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.423474f, 0.576526f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.580816f, 0.419184f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.190240f, 0.809760f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.800320f, 0.199680f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.408625f, 0.591375f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.583392f, 0.416608f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.200372f, 0.799628f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.798914f, 0.201086f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.411243f, 0.588757f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.586520f, 0.413480f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.203588f, 0.796412f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.802040f, 0.197960f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.411175f, 0.588825f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.599873f, 0.400127f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.193060f, 0.806940f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.806073f, 0.193927f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.408705f, 0.591295f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.585711f, 0.414289f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.197672f, 0.802328f},
};

// For each output (5x6) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_5x6[30][36] = {
{0.966289f, 0.033711f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.794848f, 0.205152f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.473272f, 0.496525f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.030202f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.196955f, 0.803045f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033711f, 0.966289f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966284f, 0.033716f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.795787f, 0.204213f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.500928f, 0.499072f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.198603f, 0.801397f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033716f, 0.966284f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966283f, 0.033717f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.788424f, 0.211576f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.029276f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.484227f, 0.486497f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.201499f, 0.798501f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033724f, 0.966276f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966283f, 0.033717f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.791336f, 0.208664f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.490188f, 0.509812f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.204835f, 0.795165f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033703f, 0.966297f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.966276f, 0.033724f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.799276f, 0.200724f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.022501f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.494443f, 0.483055f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.205967f, 0.794033f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033726f, 0.966274f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.965971f, 0.034029f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.798640f, 0.201360f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.502577f, 0.497423f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.203927f, 0.796073f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.033706f, 0.966294f},
};

// For each output (6x6) sample, the weight of each input (6x6) sample.
static const float g_weight_downsample_6x6_to_6x6[36][36] = {
{1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f},
{0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f},
};

//--------------------------------------------------------------------------------------------------------------------------

const struct downsample_matrix_6x6
{
	uint32_t m_grid_width, m_grid_height;
	const float* m_p;
} g_downsample_matrices_6x6[] = {
	{ 2, 2, (const float*)g_weight_downsample_6x6_to_2x2 },
	{ 3, 2, (const float*)g_weight_downsample_6x6_to_3x2 },
	{ 4, 2, (const float*)g_weight_downsample_6x6_to_4x2 },
	{ 5, 2, (const float*)g_weight_downsample_6x6_to_5x2 },
	{ 6, 2, (const float*)g_weight_downsample_6x6_to_6x2 },
	{ 2, 3, (const float*)g_weight_downsample_6x6_to_2x3 },
	{ 3, 3, (const float*)g_weight_downsample_6x6_to_3x3 },
	{ 4, 3, (const float*)g_weight_downsample_6x6_to_4x3 },
	{ 5, 3, (const float*)g_weight_downsample_6x6_to_5x3 },
	{ 6, 3, (const float*)g_weight_downsample_6x6_to_6x3 },
	{ 2, 4, (const float*)g_weight_downsample_6x6_to_2x4 },
	{ 3, 4, (const float*)g_weight_downsample_6x6_to_3x4 },
	{ 4, 4, (const float*)g_weight_downsample_6x6_to_4x4 },
	{ 5, 4, (const float*)g_weight_downsample_6x6_to_5x4 },
	{ 6, 4, (const float*)g_weight_downsample_6x6_to_6x4 },
	{ 2, 5, (const float*)g_weight_downsample_6x6_to_2x5 },
	{ 3, 5, (const float*)g_weight_downsample_6x6_to_3x5 },
	{ 4, 5, (const float*)g_weight_downsample_6x6_to_4x5 },
	{ 5, 5, (const float*)g_weight_downsample_6x6_to_5x5 },
	{ 6, 5, (const float*)g_weight_downsample_6x6_to_6x5 },
	{ 2, 6, (const float*)g_weight_downsample_6x6_to_2x6 },
	{ 3, 6, (const float*)g_weight_downsample_6x6_to_3x6 },
	{ 4, 6, (const float*)g_weight_downsample_6x6_to_4x6 },
	{ 5, 6, (const float*)g_weight_downsample_6x6_to_5x6 },
	{ 6, 6, (const float*)g_weight_downsample_6x6_to_6x6 }
};
//const uint32_t NUM_DOWNSAMPLE_MATRICES_6x6 = sizeof(g_downsample_matrices_6x6) / sizeof(g_downsample_matrices_6x6[0]);

//--------------------------------------------------------------------------------------------------------------------------

const float* get_6x6_downsample_matrix(uint32_t grid_width, uint32_t grid_height)
{
	// TODO: Use hash or map lookup.
	for (const auto& m : g_downsample_matrices_6x6)
		if ((m.m_grid_width == grid_width) && (m.m_grid_height == grid_height))
			return m.m_p;

	assert(0);
	return nullptr;
}

void downsample_weight_grid(
	const float* pMatrix_weights,
	uint32_t bx, uint32_t by,		// source/from dimension (block size)
	uint32_t wx, uint32_t wy,		// dest/to dimension (grid size)
	const uint8_t* pSrc_weights,	// these are dequantized weights, NOT ISE symbols, [by][bx]
	uint8_t* pDst_weights)			// [wy][wx]
{
	const uint32_t total_block_samples = bx * by;

	for (uint32_t y = 0; y < wy; y++)
	{
		for (uint32_t x = 0; x < wx; x++)
		{
			float total = 0.5f;

			for (uint32_t i = 0; i < total_block_samples; i++)
				if (pMatrix_weights[i])
					total += pMatrix_weights[i] * (float)pSrc_weights[i];

			pDst_weights[x + y * wx] = (uint8_t)clamp((int)total, 0, 64);

			pMatrix_weights += total_block_samples;
		}
	}
}

//--------------------------------------------------------------------------------------------------------------------------

void downsample_ise_weights(
	uint32_t dequant_weight_ise_range, uint32_t quant_weight_ise_range,
	uint32_t block_w, uint32_t block_h,
	uint32_t grid_w, uint32_t grid_h,
	const uint8_t* pSrc_weights, uint8_t* pDst_weights)
{
	assert((block_w <= MAX_ASTC_HDR_BLOCK_W) && (block_h <= MAX_ASTC_HDR_BLOCK_H));
	assert((grid_w >= 2) && (grid_w <= MAX_ASTC_HDR_BLOCK_W));
	assert((grid_h >= 2) && (grid_h <= MAX_ASTC_HDR_BLOCK_H));
	
	assert(dequant_weight_ise_range >= astc_helpers::FIRST_VALID_WEIGHT_ISE_RANGE);
	assert(dequant_weight_ise_range <= astc_helpers::LAST_VALID_WEIGHT_ISE_RANGE);

	assert(quant_weight_ise_range >= astc_helpers::FIRST_VALID_WEIGHT_ISE_RANGE);
	assert(quant_weight_ise_range <= astc_helpers::LAST_VALID_WEIGHT_ISE_RANGE);

	if ((block_w == grid_w) && (block_h == grid_h))
	{
		if (dequant_weight_ise_range != quant_weight_ise_range)
		{
			basist::astc_6x6_hdr::requantize_astc_weights(block_w * block_h, pSrc_weights, dequant_weight_ise_range, pDst_weights, quant_weight_ise_range);
		}
		else
		{
			if (pDst_weights != pSrc_weights)
				memcpy(pDst_weights, pSrc_weights, block_w * block_h);
		}

		return;
	}

	uint8_t desired_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

	const auto& dequant_tab = astc_helpers::g_dequant_tables.get_weight_tab(dequant_weight_ise_range).m_ISE_to_val;

	for (uint32_t by = 0; by < block_h; by++)
		for (uint32_t bx = 0; bx < block_w; bx++)
			desired_weights[bx + by * block_w] = dequant_tab[pSrc_weights[bx + by * block_w]];

	uint8_t downsampled_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];

	const float* pDownsample_matrix = get_6x6_downsample_matrix(grid_w, grid_h);
	assert(pDownsample_matrix);

	downsample_weight_grid(
		pDownsample_matrix,
		block_w, block_h,		// source/from dimension (block size)
		grid_w, grid_h,			// dest/to dimension (grid size)
		desired_weights,		// these are dequantized weights, NOT ISE symbols, [by][bx]
		downsampled_weights);	// [wy][wx]

	const auto& weight_quant_tab = astc_helpers::g_dequant_tables.get_weight_tab(quant_weight_ise_range).m_val_to_ise;

	for (uint32_t gy = 0; gy < grid_h; gy++)
		for (uint32_t gx = 0; gx < grid_w; gx++)
			pDst_weights[gx + gy * grid_w] = weight_quant_tab[downsampled_weights[gx + gy * grid_w]];
}

void downsample_ise_weights_dual_plane(
	uint32_t dequant_weight_ise_range, uint32_t quant_weight_ise_range,
	uint32_t block_w, uint32_t block_h,
	uint32_t grid_w, uint32_t grid_h,
	const uint8_t* pSrc_weights0, const uint8_t* pSrc_weights1,
	uint8_t* pDst_weights)
{
	uint8_t downsampled_weights0[MAX_ASTC_HDR_BLOCK_W * MAX_ASTC_HDR_BLOCK_H], downsampled_weights1[MAX_ASTC_HDR_BLOCK_W * MAX_ASTC_HDR_BLOCK_H];

	downsample_ise_weights(
		dequant_weight_ise_range, quant_weight_ise_range,
		block_w, block_h,
		grid_w, grid_h,
		pSrc_weights0, downsampled_weights0);

	downsample_ise_weights(
		dequant_weight_ise_range, quant_weight_ise_range,
		block_w, block_h,
		grid_w, grid_h,
		pSrc_weights1, downsampled_weights1);

	const uint32_t num_grid_samples = grid_w * grid_h;
	for (uint32_t i = 0; i < num_grid_samples; i++)
	{
		pDst_weights[i * 2 + 0] = downsampled_weights0[i];
		pDst_weights[i * 2 + 1] = downsampled_weights1[i];
	}
}

static bool refine_endpoints_mode11(
	uint32_t endpoint_ise_range,
	uint8_t* pEndpoint_vals, // the endpoints to optimize
	uint32_t block_w, uint32_t block_h, // block dimensions
	uint32_t grid_w, uint32_t grid_h, const uint8_t* pWeights, uint32_t weight_ise_range, // weight grid
	uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	const uint8_t* pPixel_block_ofs, // maps this subset's pixels to block offsets
	astc_hdr_codec_base_options& coptions,
	bool direct_only, int first_submode, int last_submode,
	opt_mode_t opt_mode)
{
	if (opt_mode == cNoOpt)
		return false;

	const uint32_t num_block_pixels = block_w * block_h;

	uint8_t def_pixel_block_ofs[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	if (!pPixel_block_ofs)
	{
		for (uint32_t i = 0; i < num_block_pixels; i++)
			def_pixel_block_ofs[i] = (uint8_t)i;
		
		pPixel_block_ofs = def_pixel_block_ofs;
	}

	const uint32_t num_weights = grid_w * grid_h;

	uint8_t dequantized_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	for (uint32_t i = 0; i < num_weights; i++)
		dequantized_raw_weights[i] = astc_helpers::g_dequant_tables.get_weight_tab(weight_ise_range).m_ISE_to_val[pWeights[i]];

	uint8_t upsampled_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // raw weights, NOT ISE
	astc_helpers::upsample_weight_grid(block_w, block_h, grid_w, grid_h, dequantized_raw_weights, upsampled_weights);

	aabb3F color_box_q16(cInitExpand);

	uint8_t trial_blk_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // raw weights, NOT ISE
	float trial_blk_raw_weightsf[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		color_box_q16.expand(pBlock_pixels_q16[i]);

		assert(pPixel_block_ofs[i] < num_block_pixels);

		trial_blk_raw_weights[i] = upsampled_weights[pPixel_block_ofs[i]];
		trial_blk_raw_weightsf[i] = (float)trial_blk_raw_weights[i] * (1.0f / 64.0f);
	}
	
	vec3F l_q16, h_q16;
	if (opt_mode == cOrdinaryLeastSquares)
	{
		if (!compute_least_squares_endpoints_rgb_raw_weights(num_pixels, trial_blk_raw_weights, &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16))
			return false;
	}
	else if ((opt_mode == cWeightedLeastSquares) || (opt_mode == cWeightedLeastSquaresHeavy))
	{
		vec3F block_mean_color_q16(calc_mean(num_pixels, pBlock_pixels_q16));
		vec3F block_axis_q16(calc_rgb_pca(num_pixels, pBlock_pixels_q16, block_mean_color_q16));
		float l = BIG_FLOAT_VAL, h = -BIG_FLOAT_VAL;
		for (uint32_t i = 0; i < num_pixels; i++)
		{
			vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
			float kd = k.dot(block_axis_q16);
			if (kd < l)
				l = kd;
			if (kd > h)
				h = kd;
		}
		float emphasis_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
		if (h == l)
		{
			for (uint32_t i = 0; i < num_pixels; i++)
				emphasis_weights[i] = 1.0f;
		}
		else
		{
			float mid = (0.0f - l) / (h - l);
			mid = clamp(mid, .01f, .99f);
				
			float lw = LOW_EMPHASIS_WEIGHT, mw = MIDDLE_EMPHASIS_WEIGHT, hw = HIGH_EMPHASIS_WEIGHT;
			if (opt_mode == cWeightedLeastSquaresHeavy)
				lw = LOW_EMPHASIS_WEIGHT_HEAVY, mw = MIDDLE_EMPHASIS_WEIGHT_HEAVY, hw = HIGH_EMPHASIS_WEIGHT_HEAVY;

			for (uint32_t i = 0; i < num_pixels; i++)
			{
				vec3F k(vec3F(pBlock_pixels_q16[i]) - block_mean_color_q16);
				float kd = k.dot(block_axis_q16);

				assert((kd >= l) && (kd <= h));

				float v = (kd - l) / (h - l);

				if (v < mid)
					v = lerp(lw, mw, v / mid);
				else
					v = lerp(mw, hw, (v - mid) * (1.0f - mid));

				emphasis_weights[i] = v;
			}
		}

		if (!compute_weighted_least_squares_endpoints_rgb(num_pixels, nullptr, nullptr, trial_blk_raw_weightsf, emphasis_weights, &l_q16, &h_q16, pBlock_pixels_q16, color_box_q16))
			return false;
	}
	else
	{
		assert(opt_mode == cWeightedAverage);

		l_q16.set(0.0f);
		float total_low = 0.0f;

		h_q16.set(0.0f);
		float total_high = 0.0f;

		for (uint32_t i = 0; i < num_pixels; i++)
		{
			vec3F p(pBlock_pixels_q16[i]);
			float lerp = (float)trial_blk_raw_weights[i] * (1.0f / 64.0f);

			l_q16 += p * (1.0f - lerp);
			total_low += (1.0f - lerp);

			h_q16 += p * lerp;
			total_high += lerp;
		}

		if (total_low != 0.0f)
			l_q16 *= (1.0f / total_low);
		else
			return false;

		if (total_high != 0.0f)
			h_q16 *= (1.0f / total_high);
		else
			return false;
	}

	uint8_t trial_endpoints[NUM_MODE11_ENDPOINTS];
	
	uint32_t submode_used;

	bool pack_succeeded = pack_mode11(l_q16, h_q16, endpoint_ise_range, trial_endpoints, coptions, direct_only, first_submode, last_submode, false, submode_used);
	if (!pack_succeeded)
		return false;

	int cur_e[2][3];
	if (!decode_mode11_to_qlog12(pEndpoint_vals, cur_e, endpoint_ise_range))
		return false;

	int trial_e[2][3];
	if (!decode_mode11_to_qlog12(trial_endpoints, trial_e, endpoint_ise_range))
		return false;

	for (uint32_t i = 0; i < 3; i++)
	{
		cur_e[0][i] <<= 4;
		cur_e[1][i] <<= 4;

		trial_e[0][i] <<= 4;
		trial_e[1][i] <<= 4;
	}

	const float R_WEIGHT = coptions.m_r_err_scale, G_WEIGHT = coptions.m_g_err_scale;

	double cur_error = 0, trial_error = 0;
		
	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p][0];

		const double desired_half_r_q = q(pDesired_half[0], coptions.m_q_log_bias), desired_half_g_q = q(pDesired_half[1], coptions.m_q_log_bias), desired_half_b_q = q(pDesired_half[2], coptions.m_q_log_bias);

		const uint32_t c = trial_blk_raw_weights[p];
		assert(c <= 64);

		{
			half_float rf, gf, bf;

			{
				uint32_t r0 = cur_e[0][0], r1 = cur_e[1][0];
				int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
				rf = astc_helpers::qlog16_to_half(ri);
			}

			{
				uint32_t g0 = cur_e[0][1], g1 = cur_e[1][1];
				int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
				gf = astc_helpers::qlog16_to_half(gi);
			}

			{
				uint32_t b0 = cur_e[0][2], b1 = cur_e[1][2];
				int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
				bf = astc_helpers::qlog16_to_half(bi);
			}

			const double decoded_half_q0 = q(rf, coptions.m_q_log_bias), decoded_half_q1 = q(gf, coptions.m_q_log_bias), decoded_half_q2 = q(bf, coptions.m_q_log_bias);

			const double rd = decoded_half_q0 - desired_half_r_q, gd = decoded_half_q1 - desired_half_g_q, bd = decoded_half_q2 - desired_half_b_q;

			cur_error += R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;
		}

		{
			half_float rf, gf, bf;

			{
				uint32_t r0 = trial_e[0][0], r1 = trial_e[1][0];
				int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
				rf = astc_helpers::qlog16_to_half(ri);
			}

			{
				uint32_t g0 = trial_e[0][1], g1 = trial_e[1][1];
				int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
				gf = astc_helpers::qlog16_to_half(gi);
			}

			{
				uint32_t b0 = trial_e[0][2], b1 = trial_e[1][2];
				int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
				bf = astc_helpers::qlog16_to_half(bi);
			}

			const double decoded_half_q0 = q(rf, coptions.m_q_log_bias), decoded_half_q1 = q(gf, coptions.m_q_log_bias), decoded_half_q2 = q(bf, coptions.m_q_log_bias);

			const double rd = decoded_half_q0 - desired_half_r_q, gd = decoded_half_q1 - desired_half_g_q, bd = decoded_half_q2 - desired_half_b_q;

			trial_error += R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;
		}

	} // p

	if (trial_error < cur_error)
	{
		memcpy(pEndpoint_vals, trial_endpoints, NUM_MODE11_ENDPOINTS);
		return true;
	}

	return false;
}

static bool refine_endpoints_mode7(
	uint32_t endpoint_ise_range,
	uint8_t* pEndpoint_vals, // the endpoints to optimize
	uint32_t block_w, uint32_t block_h, // block dimensions
	uint32_t grid_w, uint32_t grid_h, const uint8_t* pWeights, uint32_t weight_ise_range, // weight grid
	uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	const uint8_t* pPixel_block_ofs, // maps this subset's pixels to block offsets
	astc_hdr_codec_base_options& coptions,
	int first_submode, int last_submode)
{
	const uint32_t num_block_pixels = block_w * block_h;

	uint8_t def_pixel_block_ofs[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	if (!pPixel_block_ofs)
	{
		for (uint32_t i = 0; i < num_block_pixels; i++)
			def_pixel_block_ofs[i] = (uint8_t)i;

		pPixel_block_ofs = def_pixel_block_ofs;
	}

	const uint32_t num_weights = grid_w * grid_h;

	uint8_t dequantized_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS];
	for (uint32_t i = 0; i < num_weights; i++)
		dequantized_raw_weights[i] = astc_helpers::g_dequant_tables.get_weight_tab(weight_ise_range).m_ISE_to_val[pWeights[i]];

	uint8_t upsampled_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // raw weights, NOT ISE
	astc_helpers::upsample_weight_grid(block_w, block_h, grid_w, grid_h, dequantized_raw_weights, upsampled_weights);

	uint8_t trial_blk_raw_weights[MAX_ASTC_HDR_ENC_BLOCK_PIXELS]; // raw weights, NOT ISE
	for (uint32_t i = 0; i < num_pixels; i++)
	{
		assert(pPixel_block_ofs[i] < num_block_pixels);

		trial_blk_raw_weights[i] = upsampled_weights[pPixel_block_ofs[i]];
	}

	//--

	int cur_e[2][3];
	int cur_s = 0;
	if (!decode_mode7_to_qlog12(pEndpoint_vals, cur_e, &cur_s, endpoint_ise_range))
		return false;

	cur_s <<= 4;

	vec3F block_mean_color_q16(calc_mean(num_pixels, pBlock_pixels_q16));

	vec3F new_high_color_q16(block_mean_color_q16);
		
	const float one_over_num_pixels = 1.0f / (float)num_pixels;

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		float lerp = trial_blk_raw_weights[i] * (1.0f / 64.0f);

		float k = (float)cur_s * (1.0f - lerp) * one_over_num_pixels;
		new_high_color_q16[0] += k;
		new_high_color_q16[1] += k;
		new_high_color_q16[2] += k;
	}
					
	// Given a set of selectors and a high color, try to compute a better S.
	float t = 0.0f;

	for (uint32_t i = 0; i < num_pixels; i++)
	{
		float lerp = trial_blk_raw_weights[i] * (1.0f / 64.0f);

		t += (1.0f) - lerp;
	}

	t *= one_over_num_pixels;
	
	if (fabs(t) < .0000125f)
		return false;

	uint8_t trial_endpoints[NUM_MODE7_ENDPOINTS];

	uint32_t submode_used;
	if (!pack_mode7(new_high_color_q16, (float)cur_s, endpoint_ise_range, trial_endpoints, weight_ise_range, coptions, first_submode, last_submode, false, submode_used))
		return false;

	int trial_e[2][3];
	if (!decode_mode7_to_qlog12(trial_endpoints, trial_e, nullptr, endpoint_ise_range))
		return false;

	vec3F cur_h_q16((float)(trial_e[1][0] << 4), (float)(trial_e[1][1] << 4), (float)(trial_e[1][2] << 4));

	float s_r = (cur_h_q16[0] - block_mean_color_q16[0]) / t;
	//float s_g = (cur_h_q16[1] - block_mean_color_q16[1]) / t;
	//float s_b = (cur_h_q16[2] - block_mean_color_q16[2]) / t;
	float new_s_q16 = ceilf(s_r);

	if (!pack_mode7(new_high_color_q16, new_s_q16, endpoint_ise_range, trial_endpoints, weight_ise_range, coptions, first_submode, last_submode, false, submode_used))
		return false;

	if (!decode_mode7_to_qlog12(trial_endpoints, trial_e, nullptr, endpoint_ise_range))
		return false;
	
	// --

	for (uint32_t i = 0; i < 3; i++)
	{
		cur_e[0][i] <<= 4;
		cur_e[1][i] <<= 4;

		trial_e[0][i] <<= 4;
		trial_e[1][i] <<= 4;
	}

	const float R_WEIGHT = coptions.m_r_err_scale, G_WEIGHT = coptions.m_g_err_scale;

	double cur_error = 0, trial_error = 0;

	for (uint32_t p = 0; p < num_pixels; p++)
	{
		const half_float* pDesired_half = &pBlock_pixels_half[p][0];

		const double desired_half_r_q = q(pDesired_half[0], coptions.m_q_log_bias), desired_half_g_q = q(pDesired_half[1], coptions.m_q_log_bias), desired_half_b_q = q(pDesired_half[2], coptions.m_q_log_bias);

		const uint32_t c = trial_blk_raw_weights[p];
		assert(c <= 64);

		{
			half_float rf, gf, bf;

			{
				uint32_t r0 = cur_e[0][0], r1 = cur_e[1][0];
				int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
				rf = astc_helpers::qlog16_to_half(ri);
			}

			{
				uint32_t g0 = cur_e[0][1], g1 = cur_e[1][1];
				int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
				gf = astc_helpers::qlog16_to_half(gi);
			}

			{
				uint32_t b0 = cur_e[0][2], b1 = cur_e[1][2];
				int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
				bf = astc_helpers::qlog16_to_half(bi);
			}

			const double decoded_half_q0 = q(rf, coptions.m_q_log_bias), decoded_half_q1 = q(gf, coptions.m_q_log_bias), decoded_half_q2 = q(bf, coptions.m_q_log_bias);

			const double rd = decoded_half_q0 - desired_half_r_q, gd = decoded_half_q1 - desired_half_g_q, bd = decoded_half_q2 - desired_half_b_q;

			cur_error += R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;
		}

		{
			half_float rf, gf, bf;

			{
				uint32_t r0 = trial_e[0][0], r1 = trial_e[1][0];
				int ri = (r0 * (64 - c) + r1 * c + 32) / 64;
				rf = astc_helpers::qlog16_to_half(ri);
			}

			{
				uint32_t g0 = trial_e[0][1], g1 = trial_e[1][1];
				int gi = (g0 * (64 - c) + g1 * c + 32) / 64;
				gf = astc_helpers::qlog16_to_half(gi);
			}

			{
				uint32_t b0 = trial_e[0][2], b1 = trial_e[1][2];
				int bi = (b0 * (64 - c) + b1 * c + 32) / 64;
				bf = astc_helpers::qlog16_to_half(bi);
			}

			const double decoded_half_q0 = q(rf, coptions.m_q_log_bias), decoded_half_q1 = q(gf, coptions.m_q_log_bias), decoded_half_q2 = q(bf, coptions.m_q_log_bias);

			const double rd = decoded_half_q0 - desired_half_r_q, gd = decoded_half_q1 - desired_half_g_q, bd = decoded_half_q2 - desired_half_b_q;

			trial_error += R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;
		}

	} // p

	if (trial_error < cur_error)
	{
		memcpy(pEndpoint_vals, trial_endpoints, NUM_MODE7_ENDPOINTS);
		return true;
	}

	return false;
}

bool refine_endpoints(
	uint32_t cem,
	uint32_t endpoint_ise_range,
	uint8_t* pEndpoint_vals, // the endpoints to optimize
	uint32_t block_w, uint32_t block_h, // block dimensions
	uint32_t grid_w, uint32_t grid_h, const uint8_t* pWeights, uint32_t weight_ise_range, // weight grid
	uint32_t num_pixels, const basist::half_float pBlock_pixels_half[][3], const vec4F pBlock_pixels_q16[],
	const uint8_t* pPixel_block_ofs, // maps this subset's pixels to block offsets
	astc_hdr_codec_base_options& coptions, opt_mode_t opt_mode)
{
	if (cem == 7)
	{
		return refine_endpoints_mode7(
			endpoint_ise_range,
			pEndpoint_vals,
			block_w, block_h,
			grid_w, grid_h, pWeights, weight_ise_range,
			num_pixels, pBlock_pixels_half, pBlock_pixels_q16,
			pPixel_block_ofs,
			coptions,
			FIRST_MODE7_SUBMODE_INDEX, MAX_MODE7_SUBMODE_INDEX);
	}
	else if (cem == 11)
	{
		return refine_endpoints_mode11(
			endpoint_ise_range,
			pEndpoint_vals,
			block_w, block_h,
			grid_w, grid_h, pWeights, weight_ise_range,
			num_pixels, pBlock_pixels_half, pBlock_pixels_q16,
			pPixel_block_ofs,
			coptions,
			false, FIRST_MODE11_SUBMODE_INDEX, MAX_MODE11_SUBMODE_INDEX, opt_mode);
	}

	return false;
}

} // namespace basisu

