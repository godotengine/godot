// File: basisu_astc_hdr_6x6_enc.cpp
#include "basisu_astc_hdr_6x6_enc.h"
#include "basisu_enc.h"
#include "basisu_astc_hdr_common.h"
#include "basisu_math.h"
#include "basisu_resampler.h"
#include "basisu_resampler_filters.h"

#define MINIZ_HEADER_FILE_ONLY
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include "basisu_miniz.h"

#include "3rdparty/android_astc_decomp.h"

#include <array>

using namespace basisu;
using namespace buminiz;
using namespace basist::astc_6x6_hdr;

namespace astc_6x6_hdr
{

static void atomic_max(std::atomic<uint32_t>& atomic_var, uint32_t new_value) 
{
	uint32_t current = atomic_var.load(std::memory_order_relaxed);
	for ( ; ; )
	{
		uint32_t new_max = std::max(current, new_value);
		if (atomic_var.compare_exchange_weak(current, new_max, std::memory_order_relaxed, std::memory_order_relaxed)) 
			break;
	}
}

void astc_hdr_6x6_global_config::set_user_level(int level)
{
	level = basisu::clamp<int>(level, 0, ASTC_HDR_6X6_MAX_USER_COMP_LEVEL);

	m_master_comp_level = 0;
	m_highest_comp_level = 0;
	m_num_reuse_xy_deltas = NUM_REUSE_XY_DELTAS;
	m_extra_patterns_flag = false;
	m_brute_force_partition_matching = false;

	switch (level)
	{
	case 0:
	{
		// Both reduce compression a lot when lambda>0
		m_favor_higher_compression = false;
		m_num_reuse_xy_deltas = NUM_REUSE_XY_DELTAS / 2;
		break;
	}
	case 1:
	{
		m_master_comp_level = 0;
		m_highest_comp_level = 0;
		break;
	}
	case 2:
	{
		m_master_comp_level = 0;
		m_highest_comp_level = 1;
		break;
	}
	case 3:
	{
		m_master_comp_level = 1;
		m_highest_comp_level = 1;
		break;
	}
	case 4:
	{
		m_master_comp_level = 1;
		m_highest_comp_level = 2;
		break;
	}
	case 5:
	{
		m_master_comp_level = 1;
		m_highest_comp_level = 3;
		break;
	}
	case 6:
	{
		m_master_comp_level = 1;
		m_highest_comp_level = 4;
		break;
	}
	case 7:
	{
		m_master_comp_level = 2;
		m_highest_comp_level = 2;
		break;
	}
	case 8:
	{
		m_master_comp_level = 2;
		m_highest_comp_level = 3;
		break;
	}
	case 9:
	{
		m_master_comp_level = 2;
		m_highest_comp_level = 4;
		break;
	}
	case 10:
	{
		m_master_comp_level = 3;
		m_highest_comp_level = 3;
		break;
	}
	case 11:
	{
		m_master_comp_level = 3;
		m_highest_comp_level = 4;
		break;
	}
	case 12:
	default:
	{
		m_master_comp_level = 4;
		m_highest_comp_level = 4;
		m_extra_patterns_flag = true;
		m_brute_force_partition_matching = true;
		break;
	}
	}
}

const float m1 = 0.1593017578125f;    // (2610 / 2^14) * (1/100)
const float m2 = 78.84375f;           // (2523 / 32) * (1/100)
const float c1 = 0.8359375f;          // 3424 / (2^12)
const float c2 = 18.8515625f;         // (2413 / 128)
const float c3 = 18.6875f;            // (2392 / 128)

static float forwardPQ(float Y)
{
	// 10,000 here is an absolute scale - it's in nits (cd per square meter)
	float L = Y * (1.0f / 10000.0f);

	float num = powf(L, m1);
	float N = powf((c1 + c2 * num) / (1 + c3 * num), m2);

	return N;
}

#if 0
static float inversePQ(float E)
{
	float N = powf(E, 1.0f / m2);

	float num = basisu::maximum<float>((N - c1), 0.0f) / (c2 - c3 * N);
	float L = powf(num, 1.0f / m1);

	return L * 10000.0f;
}
#endif

// PQ function approximation: convert input to bfloat16, look up in tables, bilinear interpolation between table entries.
// max_er: 0.000023007392883, max_rel_er: 0.000023472490284, avg_er: 0.000004330495689, 6-7x faster on x86
// Highest error is for values less than SMALLEST_PQ_VAL_IN.
//
// Approximation is round trip lossless for 10-12 bits at [0,10000] nits:
// for x [0,1024] (SCALE=1023) or for x [0,4095] (SCALE=4096): 
// round(forwardPQTab(inversePQ(x / SCALE)) * SCALE) == x
//
// bfloat16 has enough precision to handle 8-bit sRGB to linear conversions:
// round(linear_to_srgb(bfloat16_to_float(float_to_bfloat16(srgb_to_linear(isRGB/255.0f))))*255.0) is lossless

const int PQ_APPROX_MIN_EXP = -16, PQ_APPROX_MAX_EXP = 16;
const int PQ_APPROX_EXP_RANGE = (PQ_APPROX_MAX_EXP - PQ_APPROX_MIN_EXP + 1);

const float SMALLEST_PQ_VAL_IN = 0.000015258829080f;
const float SMALLEST_PQ_VAL = 0.000551903737f;		// forwardPQ(SMALLEST_PQ_VAL_IN)

const float LARGEST_PQ_VAL = 1.251312f; 

float g_pq_approx_tabs[PQ_APPROX_EXP_RANGE][128];

static void init_pq_tables()
{
	for (int exp = PQ_APPROX_MIN_EXP; exp <= PQ_APPROX_MAX_EXP; exp++)
	{
		for (int mant = 0; mant < 128; mant++)
		{
			bfloat16 b = bfloat16_init(1, exp, mant);
			float bf = bfloat16_to_float(b);

			float pq = forwardPQ(bf);

			g_pq_approx_tabs[exp - PQ_APPROX_MIN_EXP][mant] = pq;
		}
	}

	//fmt_printf("{.15} {.15}\n", g_pq_approx_tabs[0][0], inversePQ(g_pq_approx_tabs[0][0]));
	//fmt_printf("{.15}\n", forwardPQ(SMALLEST_PQ_VAL_IN));
}

static inline float forwardPQTab(float v)
{
	assert(g_pq_approx_tabs[0][0]);

	assert(v >= 0.0f);
	if (v == 0.0f)
		return 0.0f;

	bfloat16 bf = float_to_bfloat16(v, false);
	assert(v >= bfloat16_to_float(bf));

	int exp = bfloat16_get_exp(bf);

	if (exp < PQ_APPROX_MIN_EXP)
	{
		// not accurate but should be good enough for our uses
		return lerp(0.0f, SMALLEST_PQ_VAL, minimum(1.0f, v / SMALLEST_PQ_VAL_IN));
	}
	else if (exp > PQ_APPROX_MAX_EXP)
		return LARGEST_PQ_VAL;

	int mant = bfloat16_get_mantissa(bf);

	float a = g_pq_approx_tabs[exp - PQ_APPROX_MIN_EXP][mant];
	float bf_f32 = bfloat16_to_float(bf);

	int next_mant = mant + 1;
	int next_exp = exp;
	if (next_mant == 128)
	{
		next_mant = 0;
		next_exp++;
		if (next_exp > PQ_APPROX_MAX_EXP)
			return a;
	}

	float b = g_pq_approx_tabs[next_exp - PQ_APPROX_MIN_EXP][next_mant];

	bfloat16 next_bf = bfloat16_init(1, next_exp, next_mant);
	float next_bf_f32 = bfloat16_to_float(next_bf);
	assert(v <= next_bf_f32);

	float lerp_factor = (v - bf_f32) / (next_bf_f32 - bf_f32);
	assert((lerp_factor >= 0) && (lerp_factor <= 1.0f));

	return lerp(a, b, lerp_factor);
}

// 100 nits = ~.5 i
// This converts absolute linear RGB light in either REC 709 or REC2020/BT2100 color gamut to ICtCp, a coding space where Ct is scaled by 2. 
// To convert to perceptual ITP for error/distance calculations, multiply the result Ct by .5 (or set itp_flag to true).
// Assumes REC 709 input, or REC 2020/BT.2100 RGB input if rec2020_bt2100_color_gamut is true.
//
// ITP info:
// https://www.portrait.com/resource-center/ictcp-color-difference-metric/
// https://professional.dolby.com/siteassets/pdfs/measuringperceptualcolorvolume_v07.253.pdf (see scale to JND's)
// This also converts from a ICtCp coding space to threshold or perceptually uniform space ITP.
//
// Linear REC709 to REC2020/BT.2100 gamut conversion:
// rgb_2100[0] = rgb_in[0] * 0.6274f + rgb_in[1] * 0.3293f + rgb_in[2] * 0.0433f;
// rgb_2100[1] = rgb_in[0] * 0.0691f + rgb_in[1] * 0.9195f + rgb_in[2] * 0.0114f;
// rgb_2100[2] = rgb_in[0] * 0.0164f + rgb_in[1] * 0.0880f + rgb_in[2] * 0.8956f;
// const float S = 1.0f / 4096.0f;
// l = (1688.0f * S) * rgb_2100[0] + (2146.0f * S) * rgb_2100[1] + (262.0f * S) * rgb_2100[2];
// m = (683.0f * S) * rgb_2100[0] + (2951.0f * S) * rgb_2100[1] + (462.0f * S) * rgb_2100[2];
// s = (99.0f * S) * rgb_2100[0] + (309.0f * S) * rgb_2100[1] + (3688.0f * S) * rgb_2100[2];
static void linear_rgb_to_ictcp(const vec3F& rgb_in, vec3F& ictcp, bool itp_flag = false, bool rec2020_bt2100_color_gamut = false)
{
	vec3F rgb_2100(rgb_in);
	
	float l, m, s;
	if (!rec2020_bt2100_color_gamut)
	{
		// Assume REC 709 input color gamut
		// (REC2020_to_LMS * REC709_to_2020) * input_color
		l = rgb_2100[0] * 0.2958097f + rgb_2100[1] * 0.6230863f + rgb_2100[2] * 0.0811040f;
		m = rgb_2100[0] * 0.1562512f + rgb_2100[1] * 0.7272980f + rgb_2100[2] * 0.1164508f;
		s = rgb_2100[0] * 0.0351435f + rgb_2100[1] * 0.1565601f + rgb_2100[2] * 0.8082964f;
	}
	else
	{
		// Assumes REC2020/BT.2100 input color gamut (this is from the spec)
		l = 0.412109375f    * rgb_2100[0] + 0.52392578125f  * rgb_2100[1] + 0.06396484375f * rgb_2100[2];
		m = 0.166748046875f * rgb_2100[0] + 0.720458984375f * rgb_2100[1] + 0.11279296875f * rgb_2100[2];
		s = 0.024169921875f * rgb_2100[0] + 0.075439453125f * rgb_2100[1] + 0.900390625f   * rgb_2100[2];
	}

	float ld = forwardPQTab(l);
	float md = forwardPQTab(m);
	float sd = forwardPQTab(s);

	ictcp[0] = .5f * ld + .5f * md;

	// if ITP scale Ct by .5 (the ICtCp spec scaled Ct to better exploit the full scaled output, which is not perceptually linear)
	if (itp_flag)
		ictcp[1] = 0.806884765625f * ld + -1.6617431640625f * md + 0.8548583984375f * sd;
	else
		ictcp[1] = 1.61376953125f * ld + -3.323486328125f * md + 1.709716796875f * sd;

	ictcp[2] = 4.378173828125f * ld + -4.24560546875f * md + -0.132568359375f * sd;
}

static inline void linear_rgb_to_itp(const vec3F& rgb_in, vec3F& itp, const astc_hdr_6x6_global_config &cfg)
{
	linear_rgb_to_ictcp(rgb_in, itp, true, cfg.m_rec2020_bt2100_color_gamut);
}

#if 0
// Outputs rec2020/bt2100 color gamut (i.e. this doesn't convert back to REC709 gamut).
static void ictcp_to_linear_rgb(const vec3F& ictcp, vec3F& rgb, bool itp_flag = false)
{
	float ct = ictcp[1];

	if (itp_flag)
		ct *= 2.0f;

	float ld = ictcp[0] + ct * 0.008609037037932726f + ictcp[2] * 0.11102962500302596f;
	float md = ictcp[0] + ct * -0.008609037037932726f + ictcp[2] * -0.11102962500302596f;
	float sd = ictcp[0] + ct * 0.5600313357106792f + ictcp[2] * -0.32062717498731885f;

	float l = inversePQ(ld);
	float m = inversePQ(md);
	float s = inversePQ(sd);

	rgb[0] = l * 3.436606694333079f + m * -2.5064521186562705f + s * 0.06984542432319149f;
	rgb[1] = l * -0.7913295555989289f + m * 1.983600451792291f + s * -0.192270896193362f;
	rgb[2] = l * -0.025949899690592672f + m * -0.09891371471172646f + s * 1.1248636144023192f;
}
#endif

struct half_vec3
{
	basist::half_float m_vals[3];

	inline half_vec3() { }

	inline half_vec3(basist::half_float x, basist::half_float y, basist::half_float z)
	{
		m_vals[0] = x;
		m_vals[1] = y;
		m_vals[2] = z;
	}

	inline half_vec3(const half_vec3& other)
	{
		*this = other;
	}

	inline half_vec3& operator= (const half_vec3& rhs)
	{
		m_vals[0] = rhs.m_vals[0];
		m_vals[1] = rhs.m_vals[1];
		m_vals[2] = rhs.m_vals[2];
		return *this;
	}

	inline void clear()
	{
		clear_obj(m_vals);
	}

	inline half_vec3 &set(basist::half_float x, basist::half_float y, basist::half_float z)
	{
		m_vals[0] = x;
		m_vals[1] = y;
		m_vals[2] = z;
		return *this;
	}

	inline half_vec3& set(float x, float y, float z)
	{
		m_vals[0] = basist::float_to_half(x);
		m_vals[1] = basist::float_to_half(y);
		m_vals[2] = basist::float_to_half(z);
		return *this;
	}

	template<typename T>
	inline half_vec3& set_vec(const T& vec)
	{
		m_vals[0] = basist::float_to_half(vec[0]);
		m_vals[1] = basist::float_to_half(vec[1]);
		m_vals[2] = basist::float_to_half(vec[2]);
		return *this;
	}

	template<typename T>
	inline T get_vec() const
	{
		return T(basist::half_to_float(m_vals[0]), basist::half_to_float(m_vals[1]), basist::half_to_float(m_vals[2]));
	}

	inline basist::half_float operator[] (uint32_t c) const { assert(c < 3); return m_vals[c]; }
	inline basist::half_float& operator[] (uint32_t c) { assert(c < 3); return m_vals[c]; }

	float get_float_comp(uint32_t c) const
	{
		assert(c < 3);
		return basist::half_to_float(m_vals[c]);
	}

	half_vec3& set_float_comp(uint32_t c, float v)
	{
		assert(c < 3);
		m_vals[c] = basist::float_to_half(v);
		return *this;
	}
};

struct half_vec4
{
	basist::half_float m_vals[4];

	inline half_vec4() { }

	inline half_vec4(basist::half_float x, basist::half_float y, basist::half_float z, basist::half_float w)
	{
		m_vals[0] = x;
		m_vals[1] = y;
		m_vals[2] = z;
		m_vals[3] = w;
	}

	inline half_vec4(const half_vec4& other)
	{
		*this = other;
	}

	inline half_vec4& operator= (const half_vec4& rhs)
	{
		m_vals[0] = rhs.m_vals[0];
		m_vals[1] = rhs.m_vals[1];
		m_vals[2] = rhs.m_vals[2];
		m_vals[3] = rhs.m_vals[3];
		return *this;
	}

	inline void clear()
	{
		clear_obj(m_vals);
	}

	inline half_vec4& set(basist::half_float x, basist::half_float y, basist::half_float z, basist::half_float w)
	{
		m_vals[0] = x;
		m_vals[1] = y;
		m_vals[2] = z;
		m_vals[3] = w;
		return *this;
	}

	inline half_vec4& set(float x, float y, float z, float w)
	{
		m_vals[0] = basist::float_to_half(x);
		m_vals[1] = basist::float_to_half(y);
		m_vals[2] = basist::float_to_half(z);
		m_vals[3] = basist::float_to_half(w);
		return *this;
	}

	template<typename T>
	inline half_vec4& set_vec(const T& vec)
	{
		m_vals[0] = basist::float_to_half(vec[0]);
		m_vals[1] = basist::float_to_half(vec[1]);
		m_vals[2] = basist::float_to_half(vec[2]);
		m_vals[3] = basist::float_to_half(vec[3]);
		return *this;
	}

	template<typename T>
	inline T get_vec() const
	{
		return T(basist::half_to_float(m_vals[0]), basist::half_to_float(m_vals[1]), basist::half_to_float(m_vals[2]), basist::half_to_float(m_vals[3]));
	}

	inline basist::half_float operator[] (uint32_t c) const { assert(c < 4); return m_vals[c]; }
	inline basist::half_float &operator[] (uint32_t c) { assert(c < 4); return m_vals[c]; }

	float get_float_comp(uint32_t c) const
	{
		assert(c < 4);
		return basist::half_to_float(m_vals[c]);
	}

	half_vec4& set_float_comp(uint32_t c, float v)
	{
		assert(c < 4);
		m_vals[c] = basist::float_to_half(v);
		return *this;
	}
};

const uint32_t MAX_BLOCK_W = 6, MAX_BLOCK_H = 6;

struct trial_result
{
	astc_helpers::log_astc_block m_log_blk;
	double m_err;
	bool m_valid;
};

//----------------------------------------------------------

const uint32_t NUM_PART3_MAPPINGS = 6;
static uint8_t g_part3_mapping[NUM_PART3_MAPPINGS][3] =
{
	{ 0, 1, 2 },
	{ 1, 2, 0 },
	{ 2, 0, 1 },
	{ 0, 2, 1 },
	{ 1, 0, 2 },
	{ 2, 1, 0 }
};

struct partition_pattern_vec
{
	uint8_t m_parts[6 * 6];

	partition_pattern_vec()
	{
		clear();
	}

	partition_pattern_vec(const partition_pattern_vec& other)
	{
		*this = other;
	}

	void clear()
	{
		memset(m_parts, 0, sizeof(m_parts));
	}

	partition_pattern_vec& operator= (const partition_pattern_vec& rhs)
	{
		if (this == &rhs)
			return *this;
		memcpy(m_parts, rhs.m_parts, 36);
		return *this;
	}

	uint8_t operator[] (uint32_t i) const { assert(i < 36); return m_parts[i]; }
	uint8_t& operator[] (uint32_t i) { assert(i < 36); return m_parts[i]; }

	uint8_t operator() (uint32_t x, uint32_t y) const { assert((x < 6) && (y < 6)); return m_parts[x + y * 6]; }
	uint8_t& operator() (uint32_t x, uint32_t y) { assert((x < 6) && (y < 6)); return m_parts[x + y * 6]; }

	int get_squared_distance(const partition_pattern_vec& other) const
	{
		int total_dist = 0;
		for (uint32_t i = 0; i < 36; i++)
			total_dist += iabs((int)m_parts[i] - (int)other.m_parts[i]);
		return total_dist;
	}

	float get_distance(const partition_pattern_vec& other) const
	{
		return sqrtf((float)get_squared_distance(other));
	}

	partition_pattern_vec get_permuted2(uint32_t permute_index) const
	{
		assert(permute_index <= 1);

		partition_pattern_vec res;
		for (uint32_t i = 0; i < 36; i++)
		{
			assert(m_parts[i] <= 1);
			res.m_parts[i] = (uint8_t)(m_parts[i] ^ permute_index);
		}

		return res;
	}

	partition_pattern_vec get_permuted3(uint32_t permute_index) const
	{
		assert(permute_index <= 5);

		partition_pattern_vec res;
		for (uint32_t i = 0; i < 36; i++)
		{
			assert(m_parts[i] <= 2);
			res.m_parts[i] = g_part3_mapping[permute_index][m_parts[i]];
		}

		return res;
	}

	partition_pattern_vec get_canonicalized() const
	{
		partition_pattern_vec res;

		int new_labels[3] = { -1, -1, -1 };
		uint32_t next_index = 0;
		for (uint32_t i = 0; i < 36; i++)
		{
			uint32_t p = m_parts[i];
			if (new_labels[p] == -1)
				new_labels[p] = next_index++;

			res.m_parts[i] = (uint8_t)new_labels[p];
		}

		return res;
	}

	bool operator== (const partition_pattern_vec& rhs) const
	{
		return memcmp(m_parts, rhs.m_parts, sizeof(m_parts)) == 0;
	}

	operator size_t() const
	{
		return basisu::hash_hsieh(m_parts, sizeof(m_parts));
	}
};

struct vp_tree_node
{
	partition_pattern_vec m_vantage_point;
	uint32_t m_point_index;
	float m_dist;

	int m_inner_node, m_outer_node;
};

#define BRUTE_FORCE_PART_SEARCH (0)

class vp_tree
{
public:
	vp_tree()
	{
	}

	void clear()
	{
		m_nodes.clear();
	}

	// This requires no redundant patterns, i.e. all must be unique.
	bool init(uint32_t n, const partition_pattern_vec* pUnique_pats)
	{
		clear();

		uint_vec pat_indices(n);
		for (uint32_t i = 0; i < n; i++)
			pat_indices[i] = i;

		std::pair<int, float> root_idx = find_best_vantage_point(n, pUnique_pats, pat_indices);

		if (root_idx.first == -1)
			return false;

		m_nodes.resize(1);
		m_nodes[0].m_vantage_point = pUnique_pats[root_idx.first];
		m_nodes[0].m_point_index = root_idx.first;
		m_nodes[0].m_dist = root_idx.second;
		m_nodes[0].m_inner_node = -1;
		m_nodes[0].m_outer_node = -1;

		uint_vec inner_list, outer_list;
		
		inner_list.reserve(n / 2);
		outer_list.reserve(n / 2);

		for (uint32_t pat_index = 0; pat_index < n; pat_index++)
		{
			if ((int)pat_index == root_idx.first)
				continue;

			const float dist = m_nodes[0].m_vantage_point.get_distance(pUnique_pats[pat_index]);

			if (dist <= root_idx.second)
				inner_list.push_back(pat_index);
			else
				outer_list.push_back(pat_index);
		}

		if (inner_list.size())
		{
			m_nodes[0].m_inner_node = create_node(n, pUnique_pats, inner_list);
			if (m_nodes[0].m_inner_node < 0)
				return false;
		}

		if (outer_list.size())
		{
			m_nodes[0].m_outer_node = create_node(n, pUnique_pats, outer_list);
			if (m_nodes[0].m_outer_node < 0)
				return false;
		}

		return true;
	}

	struct result
	{
		uint32_t m_pat_index;
		uint32_t m_mapping_index;
		float m_dist;

		bool operator< (const result& rhs) const { return m_dist < rhs.m_dist; }
		bool operator> (const result& rhs) const { return m_dist > rhs.m_dist; }
	};

	class result_queue
	{
		enum { MaxSupportedSize = 256 + 1 };

	public:
		result_queue() : 
			m_cur_size(0) 
		{
		}

		size_t get_size() const
		{
			return m_cur_size;
		}

		bool empty() const
		{
			return !m_cur_size;
		}

		typedef std::array<result, MaxSupportedSize + 1> result_array_type;

		const result_array_type& get_elements() const { return m_elements; }
		result_array_type& get_elements() { return m_elements; }

		void clear()
		{
			m_cur_size = 0;
		}

		void reserve(uint32_t n)
		{
			BASISU_NOTE_UNUSED(n);
		}

		const result& top() const
		{
			assert(m_cur_size);
			return m_elements[1];
		}

		bool insert(const result& val, uint32_t max_size)
		{
			assert(max_size < MaxSupportedSize);

			if (m_cur_size >= MaxSupportedSize)
				return false;

			m_elements[++m_cur_size] = val;
			up_heap(m_cur_size);

			if (m_cur_size > max_size)
				pop();

			return true;
		}

		bool pop()
		{
			if (m_cur_size == 0) 
				return false;

			m_elements[1] = m_elements[m_cur_size--];
			down_heap(1);
			return true;
		}
								
		float get_highest_dist() const
		{
			if (!m_cur_size)
				return 0.0f;

			return top().m_dist;
		}
	
	private:
		result_array_type m_elements;
		size_t m_cur_size;

		void up_heap(size_t index)
		{
			while ((index > 1) && (m_elements[index] > m_elements[index >> 1]))
			{
				std::swap(m_elements[index], m_elements[index >> 1]);
				index >>= 1;
			}
		}

		void down_heap(size_t index)
		{
			for ( ; ; )
			{
				size_t largest = index, left_child = 2 * index, right_child = 2 * index + 1;

				if ((left_child <= m_cur_size) && (m_elements[left_child] > m_elements[largest]))
					largest = left_child;

				if ((right_child <= m_cur_size) && (m_elements[right_child] > m_elements[largest]))
					largest = right_child;

				if (largest == index)
					break;

				std::swap(m_elements[index], m_elements[largest]);
				index = largest;
			}
		}
	};
		
	void find_nearest(uint32_t num_subsets, const partition_pattern_vec& desired_pat, result_queue& results, uint32_t max_results)
	{
		assert((num_subsets >= 2) && (num_subsets <= 3));

		results.clear();

		if (!m_nodes.size())
			return;

		uint32_t num_desired_pats;
		partition_pattern_vec desired_pats[NUM_PART3_MAPPINGS];

		if (num_subsets == 2)
		{
			num_desired_pats = 2;
			for (uint32_t i = 0; i < 2; i++)
				desired_pats[i] = desired_pat.get_permuted2(i);
		}
		else
		{
			num_desired_pats = NUM_PART3_MAPPINGS;
			for (uint32_t i = 0; i < NUM_PART3_MAPPINGS; i++)
				desired_pats[i] = desired_pat.get_permuted3(i);
		}

#if 0
		find_nearest_at_node(0, num_desired_pats, desired_pats, results, max_results);
#else
		find_nearest_at_node_non_recursive(0, num_desired_pats, desired_pats, results, max_results);
#endif
	}

private:
	basisu::vector<vp_tree_node> m_nodes;

	void find_nearest_at_node(int node_index, uint32_t num_desired_pats, const partition_pattern_vec* pDesired_pats, result_queue& results, uint32_t max_results)
	{
		float best_dist_to_vantage = BIG_FLOAT_VAL;
		uint32_t best_mapping = 0;
		for (uint32_t i = 0; i < num_desired_pats; i++)
		{
			float dist = pDesired_pats[i].get_distance(m_nodes[node_index].m_vantage_point);
			if (dist < best_dist_to_vantage)
			{
				best_dist_to_vantage = dist;
				best_mapping = i;
			}
		}

		result r;
		r.m_dist = best_dist_to_vantage;
		r.m_mapping_index = best_mapping;
		r.m_pat_index = m_nodes[node_index].m_point_index;

		results.insert(r, max_results);

		if (best_dist_to_vantage <= m_nodes[node_index].m_dist)
		{
			// inner first
			if (m_nodes[node_index].m_inner_node >= 0)
				find_nearest_at_node(m_nodes[node_index].m_inner_node, num_desired_pats, pDesired_pats, results, max_results);

			if (m_nodes[node_index].m_outer_node >= 0)
			{
				if ( (results.get_size() < max_results) || 
					((m_nodes[node_index].m_dist - best_dist_to_vantage) <= results.get_highest_dist())
					)
				{
					find_nearest_at_node(m_nodes[node_index].m_outer_node, num_desired_pats, pDesired_pats, results, max_results);
				}
			}
		}
		else
		{
			// outer first
			if (m_nodes[node_index].m_outer_node >= 0)
				find_nearest_at_node(m_nodes[node_index].m_outer_node, num_desired_pats, pDesired_pats, results, max_results);

			if (m_nodes[node_index].m_inner_node >= 0)
			{
				if ( (results.get_size() < max_results) || 
					((best_dist_to_vantage - m_nodes[node_index].m_dist) <= results.get_highest_dist())
					)
				{
					find_nearest_at_node(m_nodes[node_index].m_inner_node, num_desired_pats, pDesired_pats, results, max_results);
				}
			}
		}
	}
		
	void find_nearest_at_node_non_recursive(int init_node_index, uint32_t num_desired_pats, const partition_pattern_vec* pDesired_pats, result_queue& results, uint32_t max_results)
	{
		uint_vec node_stack;
		node_stack.reserve(16);
		node_stack.push_back(init_node_index);
		
		do
		{
			const uint32_t node_index = node_stack.back();
			node_stack.pop_back();

			float best_dist_to_vantage = BIG_FLOAT_VAL;
			uint32_t best_mapping = 0;
			for (uint32_t i = 0; i < num_desired_pats; i++)
			{
				float dist = pDesired_pats[i].get_distance(m_nodes[node_index].m_vantage_point);
				if (dist < best_dist_to_vantage)
				{
					best_dist_to_vantage = dist;
					best_mapping = i;
				}
			}

			result r;
			r.m_dist = best_dist_to_vantage;
			r.m_mapping_index = best_mapping;
			r.m_pat_index = m_nodes[node_index].m_point_index;

			results.insert(r, max_results);

			if (best_dist_to_vantage <= m_nodes[node_index].m_dist)
			{
				if (m_nodes[node_index].m_outer_node >= 0)
				{
					if ((results.get_size() < max_results) ||
						((m_nodes[node_index].m_dist - best_dist_to_vantage) <= results.get_highest_dist())
						)
					{
						node_stack.push_back(m_nodes[node_index].m_outer_node);
					}
				}

				// inner first
				if (m_nodes[node_index].m_inner_node >= 0)
				{
					node_stack.push_back(m_nodes[node_index].m_inner_node);
				}
			}
			else
			{
				if (m_nodes[node_index].m_inner_node >= 0)
				{
					if ((results.get_size() < max_results) ||
						((best_dist_to_vantage - m_nodes[node_index].m_dist) <= results.get_highest_dist())
						)
					{
						node_stack.push_back(m_nodes[node_index].m_inner_node);
					}
				}

				// outer first
				if (m_nodes[node_index].m_outer_node >= 0)
				{
					node_stack.push_back(m_nodes[node_index].m_outer_node);
				}
			}

		} while (!node_stack.empty());
	}

	// returns the index of the new node, or -1 on error
	int create_node(uint32_t n, const partition_pattern_vec* pUnique_pats, const uint_vec& pat_indices)
	{
		std::pair<int, float> root_idx = find_best_vantage_point(n, pUnique_pats, pat_indices);

		if (root_idx.first < 0)
			return -1;

		m_nodes.resize(m_nodes.size() + 1);
		const uint32_t new_node_index = m_nodes.size_u32() - 1;
				
		m_nodes[new_node_index].m_vantage_point = pUnique_pats[root_idx.first];
		m_nodes[new_node_index].m_point_index = root_idx.first;
		m_nodes[new_node_index].m_dist = root_idx.second;
		m_nodes[new_node_index].m_inner_node = -1;
		m_nodes[new_node_index].m_outer_node = -1;

		uint_vec inner_list, outer_list;

		inner_list.reserve(pat_indices.size_u32() / 2);
		outer_list.reserve(pat_indices.size_u32() / 2);

		for (uint32_t pat_indices_iter = 0; pat_indices_iter < pat_indices.size(); pat_indices_iter++)
		{
			const uint32_t pat_index = pat_indices[pat_indices_iter];

			if ((int)pat_index == root_idx.first)
				continue;

			const float dist = m_nodes[new_node_index].m_vantage_point.get_distance(pUnique_pats[pat_index]);

			if (dist <= root_idx.second)
				inner_list.push_back(pat_index);
			else
				outer_list.push_back(pat_index);
		}

		if (inner_list.size())
			m_nodes[new_node_index].m_inner_node = create_node(n, pUnique_pats, inner_list);

		if (outer_list.size())
			m_nodes[new_node_index].m_outer_node = create_node(n, pUnique_pats, outer_list);

		return new_node_index;
	}

	// returns the pattern index of the vantage point (-1 on error), and the optimal split distance
	std::pair<int, float> find_best_vantage_point(uint32_t num_unique_pats, const partition_pattern_vec* pUnique_pats, const uint_vec &pat_indices)
	{
		BASISU_NOTE_UNUSED(num_unique_pats);

		const uint32_t n = pat_indices.size_u32();

		assert(n);
		if (n == 1)
			return std::pair(pat_indices[0], 0.0f);

		float best_split_metric = -1.0f;
		int best_split_pat = -1;
		float best_split_dist = 0.0f;
		float best_split_var = 0.0f;

		basisu::vector< std::pair<float, uint32_t> > dists;
		dists.reserve(n);
		
		float_vec float_dists;
		float_dists.reserve(n);
				
		for (uint32_t pat_indices_iter = 0; pat_indices_iter < n; pat_indices_iter++)
		{
			const uint32_t split_pat_index = pat_indices[pat_indices_iter];
			assert(split_pat_index < num_unique_pats);

			const partition_pattern_vec& trial_vantage = pUnique_pats[split_pat_index];
		
			dists.resize(0);
			float_dists.resize(0);

			for (uint32_t j = 0; j < n; j++)
			{
				const uint32_t pat_index = pat_indices[j];
				assert(pat_index < num_unique_pats);

				if (pat_index == split_pat_index)
					continue;
				
				float dist = trial_vantage.get_distance(pUnique_pats[pat_index]);
				dists.emplace_back(std::pair(dist, pat_index));

				float_dists.push_back(dist);
			}

			stats<double> s;
			s.calc(float_dists.size_u32(), float_dists.data());

			std::sort(dists.begin(), dists.end(), [](const auto &a, const auto &b) {
				return a.first < b.first;
				});

			const uint32_t num_dists = dists.size_u32();
			float split_dist = dists[num_dists / 2].first;
			if ((num_dists & 1) == 0)
				split_dist = (split_dist + dists[(num_dists / 2) - 1].first) * .5f;

			uint32_t total_inner = 0, total_outer = 0;
			
			for (uint32_t j = 0; j < n; j++)
			{
				const uint32_t pat_index = pat_indices[j];
				if (pat_index == split_pat_index)
					continue;
				
				float dist = trial_vantage.get_distance(pUnique_pats[pat_index]);

				if (dist <= split_dist)
					total_inner++;
				else
					total_outer++;
			}

			float split_metric = (float)minimum(total_inner, total_outer) / (float)maximum(total_inner, total_outer);
			
			if ( (split_metric > best_split_metric) ||
				 ((split_metric == best_split_metric) && (s.m_var > best_split_var)) )
			{
				best_split_metric = split_metric;
				best_split_dist = split_dist;
				best_split_pat = split_pat_index;
				best_split_var = (float)s.m_var;
			}
		}

		return std::pair(best_split_pat, best_split_dist);
	}
};

struct partition
{
	uint64_t m_p;

	inline partition() : 
		m_p(0)
	{
	}

	inline partition(uint64_t p) :
		m_p(p)
	{
		assert(p < (1ULL << 36));
	}

	inline partition& operator=(uint64_t p)
	{
		assert(p < (1ULL << 36));
		m_p = p;
		return *this;
	}

	inline bool operator< (const partition& p) const
	{
		return m_p < p.m_p;
	}

	inline bool operator== (const partition& p) const
	{
		return m_p == p.m_p;
	}

	inline operator size_t() const
	{
		return hash_hsieh((const uint8_t *)&m_p, sizeof(m_p));
	}
};

partition_pattern_vec g_partitions2[NUM_UNIQUE_PARTITIONS2];
int g_part2_seed_to_unique_index[1024];
vp_tree g_part2_vp_tree;

static inline vec3F vec3F_norm_approx(vec3F axis)
{
	float l = axis.norm();
	axis = (fabs(l) >= SMALL_FLOAT_VAL) ? (axis * bu_math::inv_sqrt(l)) : vec3F(0.577350269f);
	return axis;
}

static void init_partitions2_6x6()
{
#if 0
	// makes pattern bits to the 10-bit ASTC seed index
	typedef basisu::hash_map<uint64_t, uint32_t> partition2_hash_map;
	partition2_hash_map phash;
	phash.reserve(1024);

	for (uint32_t i = 0; i < 1024; i++)
	{
		uint64_t p_bits = 0;
		uint64_t p_bits_inv = 0;
				
		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				uint64_t p = astc_helpers::compute_texel_partition(i, x, y, 0, 2, false);
				assert(p < 2);
								
				p_bits |= (p << (x + y * 6));
				p_bits_inv |= ((1 - p) << (x + y * 6));
			}
		}
				
		if (!p_bits)
			continue;
		if (p_bits == ((1ULL << 36) - 1))
			continue;

		assert(p_bits < (1ULL << 36));
		assert(p_bits_inv < (1ULL << 36));

		if (phash.contains(p_bits))
		{
		}
		else if (phash.contains(p_bits_inv))
		{
		}
		else
		{
			auto res = phash.insert(p_bits, i);
			assert(res.second);
			BASISU_NOTE_UNUSED(res);
		}
	}
		
	uint32_t num_unique_partitions2 = 0;
		
	for (const auto& r : phash)
	{
		assert(r.second < 1024);
		
		const uint32_t unique_index = num_unique_partitions2;
		assert(unique_index < NUM_UNIQUE_PARTITIONS2);

		partition_pattern_vec pat_vec;
		for (uint32_t i = 0; i < 36; i++)
			pat_vec[i] = (uint8_t)((r.first >> i) & 1);

		g_partitions2[unique_index] = pat_vec;
		
		assert(g_part2_unique_index_to_seed[unique_index] == r.second);
		g_part2_seed_to_unique_index[r.second] = unique_index;

		num_unique_partitions2++;
	}
	assert(num_unique_partitions2 == NUM_UNIQUE_PARTITIONS2);
#else
	for (uint32_t unique_index = 0; unique_index < NUM_UNIQUE_PARTITIONS2; unique_index++)
	{
		const uint32_t seed_index = g_part2_unique_index_to_seed[unique_index];
		assert(seed_index < 1024);

		assert(g_part2_seed_to_unique_index[seed_index] == 0);
		g_part2_seed_to_unique_index[seed_index] = unique_index;

		partition_pattern_vec& pat_vec = g_partitions2[unique_index];

		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				uint8_t p = (uint8_t)astc_helpers::compute_texel_partition(seed_index, x, y, 0, 2, false);
				assert(p < 2);

				pat_vec[x + y * 6] = p;
			}
		}
	}
#endif

	g_part2_vp_tree.init(NUM_UNIQUE_PARTITIONS2, g_partitions2);
}

static bool estimate_partition2_6x6(
	const basist::half_float pBlock_pixels_half[][3],
	int* pBest_parts, uint32_t num_best_parts)
{
	const uint32_t BLOCK_W = 6, BLOCK_H = 6, BLOCK_T = BLOCK_W * BLOCK_H;
		
	vec3F training_vecs[BLOCK_T], mean(0.0f);

	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F& v = training_vecs[i];

		v[0] = (float)pBlock_pixels_half[i][0];
		v[1] = (float)pBlock_pixels_half[i][1];
		v[2] = (float)pBlock_pixels_half[i][2];

		mean += v;
	}
	mean *= (1.0f / (float)BLOCK_T);

	vec3F max_vals(-BIG_FLOAT_VAL);

	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F& v = training_vecs[i];
		max_vals = vec3F::component_max(max_vals, v);
	}

	// Initialize principle axis approximation
	vec3F axis(max_vals - mean);

	// Incremental approx. PCA - only viable if we have a reasonably fast approximation for 1.0/sqrt(x).
	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		axis = vec3F_norm_approx(axis);

		vec3F color(training_vecs[i] - mean);

		float d = color.dot(axis);

		axis += color * d;
	}

	if (axis.norm() < SMALL_FLOAT_VAL)
		axis.set(0.57735027f);
	else
		axis.normalize_in_place();

#if BRUTE_FORCE_PART_SEARCH
	int desired_parts[BLOCK_H][BLOCK_W]; // [y][x]
	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		float proj = (training_vecs[i] - mean).dot(axis);

		desired_parts[i / BLOCK_W][i % BLOCK_W] = proj < 0.0f;
	}
#else
	partition_pattern_vec desired_part;

	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		float proj = (training_vecs[i] - mean).dot(axis);

		desired_part.m_parts[i] = proj < 0.0f;
	}
#endif
	
	//interval_timer tm;
	//tm.start();
	
#if BRUTE_FORCE_PART_SEARCH
	uint32_t part_similarity[NUM_UNIQUE_PARTITIONS2];

	for (uint32_t part_index = 0; part_index < NUM_UNIQUE_PARTITIONS2; part_index++)
	{
		const partition_pattern_vec &pat_vec = g_partitions2[part_index];

		int total_sim_non_inv = 0;
		int total_sim_inv = 0;

		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				int part = pat_vec[x + y * 6];

				if (part == desired_parts[y][x])
					total_sim_non_inv++;

				if ((part ^ 1) == desired_parts[y][x])
					total_sim_inv++;
			}
		}

		int total_sim = maximum(total_sim_non_inv, total_sim_inv);

		part_similarity[part_index] = (total_sim << 16) | part_index;

	} // part_index;

	std::sort(part_similarity, part_similarity + NUM_UNIQUE_PARTITIONS2);

	for (uint32_t i = 0; i < num_best_parts; i++)
		pBest_parts[i] = part_similarity[(NUM_UNIQUE_PARTITIONS2 - 1) - i] & 0xFFFF;
#else
	vp_tree::result_queue results;
	results.reserve(num_best_parts);
	g_part2_vp_tree.find_nearest(2, desired_part, results, num_best_parts);

	assert(results.get_size() == num_best_parts);

	const auto& elements = results.get_elements();

	for (uint32_t i = 0; i < results.get_size(); i++)
		pBest_parts[i] = elements[1 + i].m_pat_index;
#endif

	//fmt_printf("{} ", tm.get_elapsed_ms());

	return true;
}

const uint32_t MIN_REFINE_LEVEL = 0;

static bool encode_block_2_subsets(
	trial_result res[2],
	uint32_t grid_w, uint32_t grid_h,
	uint32_t cem,
	uint32_t weights_ise_range, uint32_t endpoints_ise_range,
	const half_vec3* pBlock_pixels_half, const vec4F* pBlock_pixels_q16,
	astc_hdr_codec_base_options& coptions,
	bool uber_mode_flag,
	int unique_pat_index,
	uint32_t comp_level,
	opt_mode_t mode11_opt_mode,
	bool refine_endpoints_flag)
{
	const uint32_t num_endpoint_vals = (cem == 11) ? basist::NUM_MODE11_ENDPOINTS : basist::NUM_MODE7_ENDPOINTS;

	res[0].m_valid = false;
	res[1].m_valid = false;

	const uint32_t BLOCK_W = 6, BLOCK_H = 6;

	astc_helpers::log_astc_block best_log_blk;
	clear_obj(best_log_blk);

	best_log_blk.m_num_partitions = 2;
	best_log_blk.m_color_endpoint_modes[0] = (uint8_t)cem;
	best_log_blk.m_color_endpoint_modes[1] = (uint8_t)cem;
	best_log_blk.m_grid_width = (uint8_t)grid_w;
	best_log_blk.m_grid_height = (uint8_t)grid_h;

	best_log_blk.m_weight_ise_range = (uint8_t)weights_ise_range;
	best_log_blk.m_endpoint_ise_range = (uint8_t)endpoints_ise_range;

	partition_pattern_vec* pPat = &g_partitions2[unique_pat_index];
	const uint32_t p_seed = g_part2_unique_index_to_seed[unique_pat_index];

	vec4F part_pixels_q16[2][64];
	half_vec3 part_half_pixels[2][64];
	uint8_t part_pixel_index[2][64];
	uint32_t part_total_pixels[2] = { 0 };

	for (uint32_t y = 0; y < BLOCK_H; y++)
	{
		for (uint32_t x = 0; x < BLOCK_W; x++)
		{
			uint32_t part_index = (*pPat)[x + y * BLOCK_W];

			uint32_t l = part_total_pixels[part_index];

			part_pixels_q16[part_index][l] = pBlock_pixels_q16[x + y * BLOCK_W];
			part_half_pixels[part_index][l] = pBlock_pixels_half[x + y * BLOCK_W];
			part_pixel_index[part_index][l] = (uint8_t)(x + y * BLOCK_W);

			part_total_pixels[part_index] = l + 1;
		} // x 
	} // y

	uint8_t blk_endpoints[2][basist::NUM_MODE11_ENDPOINTS];
	uint8_t blk_weights[2][BLOCK_W * BLOCK_H];
	uint32_t best_submode[2];

	for (uint32_t part_iter = 0; part_iter < 2; part_iter++)
	{
		assert(part_total_pixels[part_iter]);

		double e;
		if (cem == 7)
		{
			e = encode_astc_hdr_block_mode_7(
				part_total_pixels[part_iter],
				(basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
				best_log_blk.m_weight_ise_range,
				best_submode[part_iter],
				BIG_FLOAT_VAL,
				blk_endpoints[part_iter],
				blk_weights[part_iter],
				coptions,
				best_log_blk.m_endpoint_ise_range);
		}
		else
		{
			assert(cem == 11);

			e = encode_astc_hdr_block_mode_11(
				part_total_pixels[part_iter],
				(basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
				best_log_blk.m_weight_ise_range,
				best_submode[part_iter],
				BIG_FLOAT_VAL,
				blk_endpoints[part_iter],
				blk_weights[part_iter],
				coptions,
				false,
				best_log_blk.m_endpoint_ise_range, uber_mode_flag, false, -1, 7, false,
				mode11_opt_mode);
		}

		if (e == BIG_FLOAT_VAL)
			return false;

	} // part_iter

	uint8_t ise_weights[BLOCK_W * BLOCK_H];

	uint32_t src_pixel_index[2] = { 0, 0 };
	for (uint32_t y = 0; y < BLOCK_H; y++)
	{
		for (uint32_t x = 0; x < BLOCK_W; x++)
		{
			uint32_t part_index = (*pPat)[x + y * BLOCK_W];
			ise_weights[x + y * BLOCK_W] = blk_weights[part_index][src_pixel_index[part_index]];
			src_pixel_index[part_index]++;
		} // x
	} // y

	if ((grid_w == BLOCK_W) && (grid_h == BLOCK_H))
	{
		best_log_blk.m_partition_id = (uint16_t)p_seed;

		memcpy(best_log_blk.m_endpoints, blk_endpoints[0], num_endpoint_vals);
		memcpy(best_log_blk.m_endpoints + num_endpoint_vals, blk_endpoints[1], num_endpoint_vals);
		memcpy(best_log_blk.m_weights, ise_weights, BLOCK_W * BLOCK_H);

		res[0].m_valid = true;
		res[0].m_log_blk = best_log_blk;
	}
	else
	{
		uint8_t desired_weights[BLOCK_H * BLOCK_W];

		const auto& dequant_tab = astc_helpers::g_dequant_tables.get_weight_tab(weights_ise_range).m_ISE_to_val;

		for (uint32_t by = 0; by < BLOCK_H; by++)
			for (uint32_t bx = 0; bx < BLOCK_W; bx++)
				desired_weights[bx + by * BLOCK_W] = dequant_tab[ise_weights[bx + by * BLOCK_W]];

		uint8_t downsampled_weights[BLOCK_H * BLOCK_W];

		const float* pDownsample_matrix = get_6x6_downsample_matrix(grid_w, grid_h);
		if (!pDownsample_matrix)
		{
			assert(0);
			return false;
		}

		downsample_weight_grid(
			pDownsample_matrix,
			BLOCK_W, BLOCK_H,		// source/from dimension (block size)
			grid_w, grid_h,			// dest/to dimension (grid size)
			desired_weights,		// these are dequantized weights, NOT ISE symbols, [by][bx]
			downsampled_weights);	// [wy][wx]
				
		best_log_blk.m_partition_id = (uint16_t)p_seed;
		memcpy(best_log_blk.m_endpoints, blk_endpoints[0], num_endpoint_vals);
		memcpy(best_log_blk.m_endpoints + num_endpoint_vals, blk_endpoints[1], num_endpoint_vals);

		const auto& weight_to_ise = astc_helpers::g_dequant_tables.get_weight_tab(weights_ise_range).m_val_to_ise;

		for (uint32_t gy = 0; gy < grid_h; gy++)
			for (uint32_t gx = 0; gx < grid_w; gx++)
				best_log_blk.m_weights[gx + gy * grid_w] = weight_to_ise[downsampled_weights[gx + gy * grid_w]];

		res[0].m_valid = true;
		res[0].m_log_blk = best_log_blk;

		if ((refine_endpoints_flag) && (comp_level >= MIN_REFINE_LEVEL) && ((grid_w < 6) || (grid_h < 6)))
		{
			bool any_refined = false;

			for (uint32_t part_iter = 0; part_iter < 2; part_iter++)
			{
				bool refine_status = refine_endpoints(
					cem,
					endpoints_ise_range,
					best_log_blk.m_endpoints + part_iter * num_endpoint_vals, // the endpoints to optimize
					BLOCK_W, BLOCK_H, // block dimensions
					grid_w, grid_h, best_log_blk.m_weights, weights_ise_range, // weight grid
					part_total_pixels[part_iter], (basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
					&part_pixel_index[part_iter][0], // maps this subset's pixels to block offsets
					coptions, mode11_opt_mode);

				if (refine_status)
					any_refined = true;
			}

			if (any_refined)
			{
				res[1].m_valid = true;
				res[1].m_log_blk = best_log_blk;
			}
		}
	}

	return true;
}

typedef basisu::hash_map<partition_pattern_vec, std::pair<uint32_t, uint32_t > > partition3_hash_map;

partition_pattern_vec g_partitions3[NUM_UNIQUE_PARTITIONS3];
int g_part3_seed_to_unique_index[1024];
vp_tree g_part3_vp_tree;

static void init_partitions3_6x6()
{
	uint32_t t = 0;

	for (uint32_t i = 0; i < 1024; i++)
		g_part3_seed_to_unique_index[i] = -1;

	partition3_hash_map part3_hash;
	part3_hash.reserve(512);
		
	for (uint32_t seed_index = 0; seed_index < 1024; seed_index++)
	{
		partition_pattern_vec p3;
		uint32_t part_hist[3] = { 0 };

		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				uint64_t p = astc_helpers::compute_texel_partition(seed_index, x, y, 0, 3, false);
				assert(p < 3);

				p3.m_parts[x + y * 6] = (uint8_t)p;
				part_hist[p]++;
			}
		}

		if (!part_hist[0] || !part_hist[1] || !part_hist[2])
			continue;

		uint32_t j;
		for (j = 0; j < NUM_PART3_MAPPINGS; j++)
		{
			partition_pattern_vec temp_part3(p3.get_permuted3(j));

			if (part3_hash.contains(temp_part3))
				break;
		}
		if (j < NUM_PART3_MAPPINGS)
			continue;

		part3_hash.insert(p3, std::make_pair(seed_index, t) );

		assert(g_part3_unique_index_to_seed[t] == seed_index);
		g_part3_seed_to_unique_index[seed_index] = t;
		g_partitions3[t] = p3;

		t++;
	}

	g_part3_vp_tree.init(NUM_UNIQUE_PARTITIONS3, g_partitions3);
}

static bool estimate_partition3_6x6(
	const basist::half_float pBlock_pixels_half[][3],
	int* pBest_parts, uint32_t num_best_parts)
{
	const uint32_t BLOCK_W = 6, BLOCK_H = 6, BLOCK_T = BLOCK_W * BLOCK_H, NUM_SUBSETS = 3;

	assert(num_best_parts && (num_best_parts <= NUM_UNIQUE_PARTITIONS3));

	vec3F training_vecs[BLOCK_T], mean(0.0f);

	float brightest_inten = 0.0f, darkest_inten = BIG_FLOAT_VAL;
	vec3F cluster_centroids[NUM_SUBSETS];

	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F& v = training_vecs[i];

		v.set((float)pBlock_pixels_half[i][0], (float)pBlock_pixels_half[i][1], (float)pBlock_pixels_half[i][2]);

		float inten = v.dot(vec3F(1.0f));
		if (inten < darkest_inten)
		{
			darkest_inten = inten;
			cluster_centroids[0] = v;
		}

		if (inten > brightest_inten)
		{
			brightest_inten = inten;
			cluster_centroids[1] = v;
		}
	}

	if (cluster_centroids[0] == cluster_centroids[1])
		return false;

	float furthest_dist2 = 0.0f;
	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F& v = training_vecs[i];

		float dist_a = v.squared_distance(cluster_centroids[0]);
		if (dist_a == 0.0f)
			continue;

		float dist_b = v.squared_distance(cluster_centroids[1]);
		if (dist_b == 0.0f)
			continue;

		float dist2 = dist_a + dist_b;
		if (dist2 > furthest_dist2)
		{
			furthest_dist2 = dist2;
			cluster_centroids[2] = v;
		}
	}

	if ((cluster_centroids[0] == cluster_centroids[2]) || (cluster_centroids[1] == cluster_centroids[2]))
		return false;
		
	uint32_t cluster_pixels[NUM_SUBSETS][BLOCK_T];
	uint32_t num_cluster_pixels[NUM_SUBSETS];
	vec3F new_cluster_means[NUM_SUBSETS];

	const uint32_t NUM_ITERS = 4;
	
	for (uint32_t s = 0; s < NUM_ITERS; s++)
	{
		memset(num_cluster_pixels, 0, sizeof(num_cluster_pixels));
		memset(new_cluster_means, 0, sizeof(new_cluster_means));

		for (uint32_t i = 0; i < BLOCK_T; i++)
		{
			float d[NUM_SUBSETS] = { 
				training_vecs[i].squared_distance(cluster_centroids[0]), 
				training_vecs[i].squared_distance(cluster_centroids[1]), 
				training_vecs[i].squared_distance(cluster_centroids[2]) };

			float min_d = d[0];
			uint32_t min_idx = 0;
			for (uint32_t j = 1; j < NUM_SUBSETS; j++)
			{
				if (d[j] < min_d)
				{
					min_d = d[j];
					min_idx = j;
				}
			}

			cluster_pixels[min_idx][num_cluster_pixels[min_idx]] = i;
			new_cluster_means[min_idx] += training_vecs[i];
			num_cluster_pixels[min_idx]++;
		} // i

		for (uint32_t j = 0; j < NUM_SUBSETS; j++)
		{
			if (!num_cluster_pixels[j])
				return false;

			cluster_centroids[j] = new_cluster_means[j] / (float)num_cluster_pixels[j];
		}
	} // s
		
	partition_pattern_vec desired_part;
	for (uint32_t p = 0; p < NUM_SUBSETS; p++)
	{
		for (uint32_t i = 0; i < num_cluster_pixels[p]; i++)
		{
			const uint32_t pix_index = cluster_pixels[p][i];
			desired_part[pix_index] = (uint8_t)p;
		}
	}

#if BRUTE_FORCE_PART_SEARCH
	partition_pattern_vec desired_parts[NUM_PART3_MAPPINGS];
	for (uint32_t j = 0; j < NUM_PART3_MAPPINGS; j++)
		desired_parts[j] = desired_part.get_permuted3(j);

	uint32_t part_similarity[NUM_UNIQUE_PARTITIONS3];

	for (uint32_t part_index = 0; part_index < NUM_UNIQUE_PARTITIONS3; part_index++)
	{
		const partition_pattern_vec& pat = g_partitions3[part_index];

		uint32_t lowest_pat_dist = UINT32_MAX;
		for (uint32_t p = 0; p < NUM_PART3_MAPPINGS; p++)
		{
			uint32_t dist = pat.get_squared_distance(desired_parts[p]);
			if (dist < lowest_pat_dist)
				lowest_pat_dist = dist;
		}

		part_similarity[part_index] = (lowest_pat_dist << 16) | part_index;

	} // part_index;

	std::sort(part_similarity, part_similarity + NUM_UNIQUE_PARTITIONS3);
		
	for (uint32_t i = 0; i < num_best_parts; i++)
		pBest_parts[i] = part_similarity[i] & 0xFFFF;
#else
	vp_tree::result_queue results;
	results.reserve(num_best_parts);
	g_part3_vp_tree.find_nearest(3, desired_part, results, num_best_parts);

	assert(results.get_size() == num_best_parts);

	const auto& elements = results.get_elements();

	for (uint32_t i = 0; i < results.get_size(); i++)
		pBest_parts[i] = elements[1 + i].m_pat_index;
#endif

	return true;
}

static bool encode_block_3_subsets(
	trial_result& res,
	uint32_t cem,
	uint32_t grid_w, uint32_t grid_h,
	uint32_t weights_ise_range, uint32_t endpoints_ise_range,
	const half_vec3* pBlock_pixels_half, const vec4F* pBlock_pixels_q16,
	astc_hdr_codec_base_options& coptions,
	bool uber_mode_flag,
	const int* pEst_patterns, int num_est_patterns,
	uint32_t comp_level, 
	opt_mode_t mode11_opt_mode)
{
	BASISU_NOTE_UNUSED(uber_mode_flag);
	const uint32_t BLOCK_W = 6, BLOCK_H = 6, NUM_SUBSETS = 3;
	const uint32_t num_endpoint_vals = astc_helpers::get_num_cem_values(cem);
		
	res.m_valid = false;
		
	double best_e = BIG_FLOAT_VAL;

	astc_helpers::log_astc_block best_log_blk;
	clear_obj(best_log_blk);

	best_log_blk.m_num_partitions = NUM_SUBSETS;
	best_log_blk.m_color_endpoint_modes[0] = (uint8_t)cem;
	best_log_blk.m_color_endpoint_modes[1] = (uint8_t)cem;
	best_log_blk.m_color_endpoint_modes[2] = (uint8_t)cem;
	best_log_blk.m_grid_width = (uint8_t)grid_w;
	best_log_blk.m_grid_height = (uint8_t)grid_h;

	best_log_blk.m_weight_ise_range = (uint8_t)weights_ise_range;
	best_log_blk.m_endpoint_ise_range = (uint8_t)endpoints_ise_range;

	const uint32_t n = num_est_patterns ? num_est_patterns : NUM_UNIQUE_PARTITIONS3;

	for (uint32_t unique_p_iter = 0; unique_p_iter < n; unique_p_iter++)
	{
		const uint32_t unique_part_index = num_est_patterns ? pEst_patterns[unique_p_iter] : unique_p_iter;
		assert(unique_part_index < NUM_UNIQUE_PARTITIONS3);
		const partition_pattern_vec*pPart = &g_partitions3[unique_part_index];

		vec4F part_pixels_q16[NUM_SUBSETS][64];
		half_vec3 part_half_pixels[NUM_SUBSETS][64];
		uint8_t part_pixel_index[NUM_SUBSETS][64];
		uint32_t part_total_pixels[NUM_SUBSETS] = { 0 };

		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				const uint32_t part_index = pPart->m_parts[x + y * BLOCK_W];

				uint32_t l = part_total_pixels[part_index];

				part_pixels_q16[part_index][l] = pBlock_pixels_q16[x + y * BLOCK_W];
				part_half_pixels[part_index][l] = pBlock_pixels_half[x + y * BLOCK_W];
				part_pixel_index[part_index][l] = (uint8_t)(x + y * BLOCK_W);

				part_total_pixels[part_index] = l + 1;
			} // x 
		} // y

		uint8_t blk_endpoints[NUM_SUBSETS][basist::NUM_MODE11_ENDPOINTS];
		uint8_t blk_weights[NUM_SUBSETS][BLOCK_W * BLOCK_H];
		uint32_t best_submode[NUM_SUBSETS];

		double e = 0.0f;
		for (uint32_t part_iter = 0; part_iter < NUM_SUBSETS; part_iter++)
		{
			assert(part_total_pixels[part_iter]);

			if (cem == 7)
			{
				e += encode_astc_hdr_block_mode_7(
					part_total_pixels[part_iter],
					(basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
					best_log_blk.m_weight_ise_range,
					best_submode[part_iter],
					BIG_FLOAT_VAL,
					blk_endpoints[part_iter],
					blk_weights[part_iter],
					coptions,
					best_log_blk.m_endpoint_ise_range);
			}
			else
			{
				assert(cem == 11);

				e += encode_astc_hdr_block_mode_11(
					part_total_pixels[part_iter],
					(basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
					best_log_blk.m_weight_ise_range,
					best_submode[part_iter],
					BIG_FLOAT_VAL,
					blk_endpoints[part_iter],
					blk_weights[part_iter],
					coptions,
					false, best_log_blk.m_endpoint_ise_range, uber_mode_flag, false, 
					FIRST_MODE11_SUBMODE_INDEX, MAX_MODE11_SUBMODE_INDEX, false, mode11_opt_mode);
			}

		} // part_iter

		uint8_t ise_weights[BLOCK_W * BLOCK_H];

		uint32_t src_pixel_index[NUM_SUBSETS] = { 0 };
		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				const uint32_t part_index = pPart->m_parts[x + y * BLOCK_W];

				ise_weights[x + y * BLOCK_W] = blk_weights[part_index][src_pixel_index[part_index]];
				src_pixel_index[part_index]++;
			} // x
		} // y

		if ((grid_w == BLOCK_W) && (grid_h == BLOCK_H))
		{
			if (e < best_e)
			{
				best_e = e;
				best_log_blk.m_partition_id = (uint16_t)g_part3_unique_index_to_seed[unique_part_index];

				for (uint32_t p = 0; p < NUM_SUBSETS; p++)
					memcpy(best_log_blk.m_endpoints + num_endpoint_vals * p, blk_endpoints[p], num_endpoint_vals);
				
				memcpy(best_log_blk.m_weights, ise_weights, BLOCK_W * BLOCK_H);
			}
		}
		else
		{
			uint8_t desired_weights[BLOCK_H * BLOCK_W];

			const auto& dequant_tab = astc_helpers::g_dequant_tables.get_weight_tab(weights_ise_range).m_ISE_to_val;

			for (uint32_t by = 0; by < BLOCK_H; by++)
				for (uint32_t bx = 0; bx < BLOCK_W; bx++)
					desired_weights[bx + by * BLOCK_W] = dequant_tab[ise_weights[bx + by * BLOCK_W]];

			uint8_t downsampled_weights[BLOCK_H * BLOCK_W];

			const float* pDownsample_matrix = get_6x6_downsample_matrix(grid_w, grid_h);
			if (!pDownsample_matrix)
			{
				assert(0);
				return false;
			}

			downsample_weight_grid(
				pDownsample_matrix,
				BLOCK_W, BLOCK_H,		// source/from dimension (block size)
				grid_w, grid_h,			// dest/to dimension (grid size)
				desired_weights,		// these are dequantized weights, NOT ISE symbols, [by][bx]
				downsampled_weights);	// [wy][wx]

			astc_helpers::log_astc_block trial_blk(best_log_blk);

			trial_blk.m_partition_id = (uint16_t)g_part3_unique_index_to_seed[unique_part_index];
			
			for (uint32_t p = 0; p < NUM_SUBSETS; p++)
				memcpy(trial_blk.m_endpoints + num_endpoint_vals * p, blk_endpoints[p], num_endpoint_vals);

			const auto& weight_to_ise = astc_helpers::g_dequant_tables.get_weight_tab(weights_ise_range).m_val_to_ise;

			for (uint32_t gy = 0; gy < grid_h; gy++)
				for (uint32_t gx = 0; gx < grid_w; gx++)
					trial_blk.m_weights[gx + gy * grid_w] = weight_to_ise[downsampled_weights[gx + gy * grid_w]];

			if ((comp_level >= MIN_REFINE_LEVEL) && ((grid_w < 6) || (grid_h < 6)))
			{
				for (uint32_t part_iter = 0; part_iter < NUM_SUBSETS; part_iter++)
				{
					bool refine_status = refine_endpoints(
						cem,
						endpoints_ise_range,
						trial_blk.m_endpoints + part_iter * num_endpoint_vals, // the endpoints to optimize
						BLOCK_W, BLOCK_H, // block dimensions
						grid_w, grid_h, trial_blk.m_weights, weights_ise_range, // weight grid
						part_total_pixels[part_iter], (basist::half_float(*)[3])part_half_pixels[part_iter], (vec4F*)part_pixels_q16[part_iter],
						&part_pixel_index[part_iter][0], // maps this subset's pixels to block offsets
						coptions, mode11_opt_mode);

					BASISU_NOTE_UNUSED(refine_status);
				}
			}

			half_vec4 decoded_pixels_half4[BLOCK_H][BLOCK_W]; // [y][x]
			bool status = astc_helpers::decode_block(trial_blk, decoded_pixels_half4, BLOCK_W, BLOCK_H, astc_helpers::cDecodeModeHDR16);
			assert(status);
			if (!status)
				return false;

			half_vec3 decoded_pixels_half3[BLOCK_H][BLOCK_W];
			for (uint32_t y = 0; y < BLOCK_H; y++)
				for (uint32_t x = 0; x < BLOCK_W; x++)
					decoded_pixels_half3[y][x].set(decoded_pixels_half4[y][x][0], decoded_pixels_half4[y][x][1], decoded_pixels_half4[y][x][2]);

			double trial_err = compute_block_error(BLOCK_W * BLOCK_H, (const basist::half_float*)pBlock_pixels_half, (const basist::half_float*)decoded_pixels_half3, coptions);
			if (trial_err < best_e)
			{
				best_e = trial_err;
				best_log_blk = trial_blk;
			}
		}

	} // unique_p_iter

	if (best_e < BIG_FLOAT_VAL)
	{
		res.m_log_blk = best_log_blk;
		res.m_valid = true;
		res.m_err = best_e;
	}
	else
	{
		res.m_valid = false;
	}

	return res.m_valid;
}

static uint32_t encode_values(bitwise_coder &coder, uint32_t total_values, const uint8_t *pVals, uint32_t endpoint_range)
{
	const uint32_t MAX_VALS = 64;
	uint32_t bit_values[MAX_VALS], tq_values[(MAX_VALS + 2) / 3];
	uint32_t total_tq_values = 0, tq_accum = 0, tq_mul = 1;

	assert((total_values) && (total_values <= MAX_VALS));
	
	const uint32_t ep_bits = astc_helpers::g_ise_range_table[endpoint_range][0];
	const uint32_t ep_trits = astc_helpers::g_ise_range_table[endpoint_range][1];
	const uint32_t ep_quints = astc_helpers::g_ise_range_table[endpoint_range][2];

	for (uint32_t i = 0; i < total_values; i++)
	{
		uint32_t val = pVals[i];

		uint32_t bits = val & ((1 << ep_bits) - 1);
		uint32_t tq = val >> ep_bits;

		bit_values[i] = bits;

		if (ep_trits)
		{
			assert(tq < 3);
			tq_accum += tq * tq_mul;
			tq_mul *= 3;
			if (tq_mul == 243)
			{
				assert(total_tq_values < BASISU_ARRAY_SIZE(tq_values));
				tq_values[total_tq_values++] = tq_accum;
				tq_accum = 0;
				tq_mul = 1;
			}
		}
		else if (ep_quints)
		{
			assert(tq < 5);
			tq_accum += tq * tq_mul;
			tq_mul *= 5;
			if (tq_mul == 125)
			{
				assert(total_tq_values < BASISU_ARRAY_SIZE(tq_values));
				tq_values[total_tq_values++] = tq_accum;
				tq_accum = 0;
				tq_mul = 1;
			}
		}
	}

	uint32_t total_bits_output = 0;
	
	for (uint32_t i = 0; i < total_tq_values; i++)
	{
		const uint32_t num_bits = ep_trits ? 8 : 7;
		coder.put_bits(tq_values[i], num_bits);
		total_bits_output += num_bits;
	}

	if (tq_mul > 1)
	{
		uint32_t num_bits;
		if (ep_trits)
		{
			if (tq_mul == 3)
				num_bits = 2;
			else if (tq_mul == 9)
				num_bits = 4;
			else if (tq_mul == 27)
				num_bits = 5;
			else //if (tq_mul == 81)
				num_bits = 7;
		}
		else
		{
			if (tq_mul == 5)
				num_bits = 3;
			else //if (tq_mul == 25)
				num_bits = 5;
		}
		coder.put_bits(tq_accum, num_bits);
		total_bits_output += num_bits;
	}

	for (uint32_t i = 0; i < total_values; i++)
	{
		coder.put_bits(bit_values[i], ep_bits);
		total_bits_output += ep_bits;
	}

	return total_bits_output;
}

static inline uint32_t get_num_endpoint_vals(uint32_t cem)
{
	assert((cem == 7) || (cem == 11));
	return (cem == 11) ? basist::NUM_MODE11_ENDPOINTS : basist::NUM_MODE7_ENDPOINTS;
}

static void code_block(bitwise_coder& coder,
	const astc_helpers::log_astc_block& log_blk,
	block_mode block_mode_index,
	endpoint_mode em, const uint8_t *pEP_deltas)
{
	coder.put_truncated_binary((uint32_t)block_mode_index, (uint32_t)block_mode::cBMTotalModes);
	coder.put_truncated_binary((uint32_t)em, (uint32_t)endpoint_mode::cTotal);

	const uint32_t num_endpoint_vals = get_num_endpoint_vals(log_blk.m_color_endpoint_modes[0]);

	if ((em == endpoint_mode::cUseLeftDelta) || (em == endpoint_mode::cUseUpperDelta))
	{
		assert(log_blk.m_num_partitions == 1);

		for (uint32_t i = 0; i < num_endpoint_vals; i++)
			coder.put_bits(pEP_deltas[i], NUM_ENDPOINT_DELTA_BITS);
	}
	else if (em == endpoint_mode::cRaw)
	{
		if (log_blk.m_num_partitions == 2)
		{
			const int unique_partition_index = g_part2_seed_to_unique_index[log_blk.m_partition_id];
			assert(unique_partition_index != -1);
			
			coder.put_truncated_binary(unique_partition_index, NUM_UNIQUE_PARTITIONS2);
		}
		else if (log_blk.m_num_partitions == 3)
		{
			const int unique_partition_index = g_part3_seed_to_unique_index[log_blk.m_partition_id];
			assert(unique_partition_index != -1);

			coder.put_truncated_binary(unique_partition_index, NUM_UNIQUE_PARTITIONS3);
		}
		
		encode_values(coder, num_endpoint_vals * log_blk.m_num_partitions, log_blk.m_endpoints, log_blk.m_endpoint_ise_range);
	}

	encode_values(coder, log_blk.m_grid_width * log_blk.m_grid_height * (log_blk.m_dual_plane ? 2 : 1), log_blk.m_weights, log_blk.m_weight_ise_range);
}

struct smooth_map_params
{
	bool m_no_mse_scaling;

	float m_max_smooth_std_dev;
	float m_smooth_max_mse_scale;

	float m_max_med_smooth_std_dev;
	float m_med_smooth_max_mse_scale;

	float m_max_ultra_smooth_std_dev;
	float m_ultra_smooth_max_mse_scale;

	bool m_debug_images;

	smooth_map_params()
	{
		clear();
	}

	void clear()
	{
		m_no_mse_scaling = false;

		// 3x3 region
		m_max_smooth_std_dev = 100.0f;
		m_smooth_max_mse_scale = 13000.0f;
				
		// 7x7 region
		m_max_med_smooth_std_dev = 9.0f;
		m_med_smooth_max_mse_scale = 15000.0f;

		// 11x11 region
		m_max_ultra_smooth_std_dev = 4.0f;
		//m_ultra_smooth_max_mse_scale = 4500.0f;
		//m_ultra_smooth_max_mse_scale = 10000.0f;
		//m_ultra_smooth_max_mse_scale = 50000.0f;
		//m_ultra_smooth_max_mse_scale = 100000.0f;
		//m_ultra_smooth_max_mse_scale = 400000.0f;
		//m_ultra_smooth_max_mse_scale = 800000.0f;
		m_ultra_smooth_max_mse_scale = 2000000.0f;

		m_debug_images = true;
	}
};

Resampler::Contrib_List* g_contrib_lists[7]; // 1-6

static void init_contrib_lists()
{
	for (uint32_t dst_width = 1; dst_width <= 6; dst_width++)
		//g_contrib_lists[dst_width] = Resampler::make_clist(6, 6, basisu::Resampler::BOUNDARY_CLAMP, gaussian_filter, BASISU_GAUSSIAN_FILTER_SUPPORT, 6.0f / (float)dst_width, 0.0f);
		g_contrib_lists[dst_width] = Resampler::make_clist(6, 6, basisu::Resampler::BOUNDARY_CLAMP, gaussian_filter, BASISU_BELL_FILTER_SUPPORT, 6.0f / (float)dst_width, 0.0f);
}

#if 0
static void filter_block(uint32_t grid_x, uint32_t grid_y, const vec3F* pSrc_block, half_vec3 *pDst_block_half3, vec4F *pDst_block_q16)
{
	vec3F temp_block[6][6]; // [y][x]

	// first filter rows to temp_block
	if (grid_x == 6)
	{
		memcpy(temp_block, pSrc_block, sizeof(vec3F) * 6 * 6);
	}
	else
	{
		Resampler::Contrib_List* pRow_lists = g_contrib_lists[grid_x];

		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				vec3F p(0.0f);

				for (uint32_t i = 0; i < pRow_lists[x].n; i++)
					p += pSrc_block[y * 6 + pRow_lists[x].p[i].pixel] * pRow_lists[x].p[i].weight;

				p.clamp(0.0f, basist::ASTC_HDR_MAX_VAL);

				temp_block[y][x] = p;
			} // x
		} // y
	}

	// filter columns
	if (grid_y == 6)
	{
		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				for (uint32_t c = 0; c < 3; c++)
				{
					const basist::half_float h = basist::float_to_half(temp_block[y][x][c]);
					
					pDst_block_half3[x + y * 6][c] = h;
					pDst_block_q16[x + y * 6][c] = (float)half_to_qlog16(h);
				}

				pDst_block_q16[x + y * 6][3] = 0.0f;
			} // x
		} // y
	}
	else
	{
		Resampler::Contrib_List* pCol_lists = g_contrib_lists[grid_y];

		for (uint32_t x = 0; x < 6; x++)
		{
			for (uint32_t y = 0; y < 6; y++)
			{
				vec3F p(0.0f);

				for (uint32_t i = 0; i < pCol_lists[y].n; i++)
					p += temp_block[pCol_lists[y].p[i].pixel][x] * pCol_lists[y].p[i].weight;
				
				p.clamp(0.0f, basist::ASTC_HDR_MAX_VAL);
				
				for (uint32_t c = 0; c < 3; c++)
				{
					const basist::half_float h = basist::float_to_half(p[c]);

					pDst_block_half3[x + y * 6][c] = h;
					pDst_block_q16[x + y * 6][c] = (float)half_to_qlog16(h);
				}

				pDst_block_q16[x + y * 6][3] = 0.0f;
				
			} // x
		} // y
	}
}
#endif

static void filter_block(uint32_t grid_x, uint32_t grid_y, const vec4F* pSrc_block, vec4F* pDst_block)
{
	vec4F temp_block[6][6]; // [y][x]

	// first filter rows to temp_block
	if (grid_x == 6)
	{
		memcpy(temp_block, pSrc_block, sizeof(vec4F) * 6 * 6);
	}
	else
	{
		Resampler::Contrib_List* pRow_lists = g_contrib_lists[grid_x];

		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				vec3F p(0.0f);

				for (uint32_t i = 0; i < pRow_lists[x].n; i++)
					p += vec3F(pSrc_block[y * 6 + pRow_lists[x].p[i].pixel]) * pRow_lists[x].p[i].weight;

				p.clamp(0.0f, basist::ASTC_HDR_MAX_VAL);

				temp_block[y][x] = p;
			} // x
		} // y
	}

	// filter columns
	if (grid_y == 6)
	{
		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				for (uint32_t c = 0; c < 3; c++)
					pDst_block[x + y * 6][c] = temp_block[y][x][c];
			} // x
		} // y
	}
	else
	{
		Resampler::Contrib_List* pCol_lists = g_contrib_lists[grid_y];

		for (uint32_t x = 0; x < 6; x++)
		{
			for (uint32_t y = 0; y < 6; y++)
			{
				vec3F p(0.0f);

				for (uint32_t i = 0; i < pCol_lists[y].n; i++)
					p += temp_block[pCol_lists[y].p[i].pixel][x] * pCol_lists[y].p[i].weight;

				p.clamp(0.0f, basist::ASTC_HDR_MAX_VAL);

				pDst_block[x + y * 6] = p;

			} // x
		} // y
	}
}

static void filter_block(uint32_t grid_x, uint32_t grid_y, const vec3F* pSrc_block, vec3F* pDst_block)
{
	vec3F temp_block[6][6]; // [y][x]

	// first filter rows to temp_block
	if (grid_x == 6)
	{
		memcpy(temp_block, pSrc_block, sizeof(vec3F) * 6 * 6);
	}
	else
	{
		Resampler::Contrib_List* pRow_lists = g_contrib_lists[grid_x];

		for (uint32_t y = 0; y < 6; y++)
		{
			for (uint32_t x = 0; x < 6; x++)
			{
				vec3F p(0.0f);

				for (uint32_t i = 0; i < pRow_lists[x].n; i++)
					p += vec3F(pSrc_block[y * 6 + pRow_lists[x].p[i].pixel]) * pRow_lists[x].p[i].weight;
								
				temp_block[y][x] = p;
			} // x
		} // y
	}

	// filter columns
	if (grid_y == 6)
	{
		memcpy((void *)pDst_block, temp_block, sizeof(vec3F) * 6 * 6);
	}
	else
	{
		Resampler::Contrib_List* pCol_lists = g_contrib_lists[grid_y];

		for (uint32_t x = 0; x < 6; x++)
		{
			for (uint32_t y = 0; y < 6; y++)
			{
				vec3F& p = pDst_block[x + y * 6];
				p.set(0.0f);

				for (uint32_t i = 0; i < pCol_lists[y].n; i++)
					p += temp_block[pCol_lists[y].p[i].pixel][x] * pCol_lists[y].p[i].weight;
			} // x
		} // y
	}
}

static float diff_blocks(const vec4F* pA, const vec4F* pB)
{
	const uint32_t BLOCK_T = 36;

	float diff = 0.0f;
	for (uint32_t i = 0; i < BLOCK_T; i++)
		diff += square(pA[i][0] - pB[i][0]) + square(pA[i][1] - pB[i][1]) + square(pA[i][2] - pB[i][2]);
	
	return diff * (1.0f / (float)BLOCK_T);
}

static float sub_and_compute_std_dev(const vec3F* pA, const vec3F* pB)
{
	const uint32_t BLOCK_T = 36;

	vec3F mean(0.0f);

	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F diff(pA[i] - pB[i]);
		mean += diff;
	}

	mean *= (1.0f / (float)BLOCK_T);

	vec3F diff_sum(0.0f);
	for (uint32_t i = 0; i < BLOCK_T; i++)
	{
		vec3F diff(pA[i] - pB[i]);
		diff -= mean;
		diff_sum += vec3F::component_mul(diff, diff);
	}

	vec3F var(diff_sum * (1.0f / (float)BLOCK_T));

	vec3F std_dev(sqrtf(var[0]), sqrtf(var[1]), sqrtf(var[2]));

	return maximum(std_dev[0], std_dev[1], std_dev[2]);
}

static void create_smooth_maps2(
	vector2D<float>& smooth_block_mse_scales,
	const image& orig_img,
	smooth_map_params& params, image* pUltra_smooth_img = nullptr)
{
	const uint32_t width = orig_img.get_width();
	const uint32_t height = orig_img.get_height();
	//const uint32_t total_pixels = orig_img.get_total_pixels();
	const uint32_t num_comps = 3;

	if (params.m_no_mse_scaling)
	{
		smooth_block_mse_scales.set_all(1.0f);
		return;
	}

	smooth_block_mse_scales.resize(width, height);

	image smooth_vis, med_smooth_vis, ultra_smooth_vis;

	if (params.m_debug_images)
	{
		smooth_vis.resize(width, height);
		med_smooth_vis.resize(width, height);
		ultra_smooth_vis.resize(width, height);
	}

	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			{
				tracked_stat_dbl comp_stats[4];
				for (int yd = -1; yd <= 1; yd++)
				{
					for (int xd = -1; xd <= 1; xd++)
					{
						const color_rgba& p = orig_img.get_clamped((int)x + xd, (int)y + yd);

						comp_stats[0].update((float)p[0]);
						comp_stats[1].update((float)p[1]);
						comp_stats[2].update((float)p[2]);
					}
				}

				float max_std_dev = 0.0f;
				for (uint32_t i = 0; i < num_comps; i++)
					max_std_dev = basisu::maximum(max_std_dev, (float)comp_stats[i].get_std_dev());

				float yl = clampf(max_std_dev / params.m_max_smooth_std_dev, 0.0f, 1.0f);
				//yl = powf(yl, 2.0f);
				yl = powf(yl, 1.0f / 2.0f); // substantially less bits

				smooth_block_mse_scales(x, y) = lerp(params.m_smooth_max_mse_scale, 1.0f, yl);

				if (params.m_debug_images)
				{
					//smooth_vis(x, y).set(clamp((int)((smooth_block_mse_scales(x, y) - 1.0f) / (params.m_smooth_max_mse_scale - 1.0f) * 255.0f + .5f), 0, 255));
					// white=high local activity (edges/detail)
					// black=low local activity (smooth - error is amplified)
					smooth_vis(x, y).set(clamp((int)((yl * 255.0f) + .5f), 0, 255));
				}
			}

			{
				tracked_stat_dbl comp_stats[4];

				const int S = 3;
				for (int yd = -S; yd < S; yd++)
				{
					for (int xd = -S; xd < S; xd++)
					{
						const color_rgba& p = orig_img.get_clamped((int)x + xd, (int)y + yd);

						comp_stats[0].update((float)p[0]);
						comp_stats[1].update((float)p[1]);
						comp_stats[2].update((float)p[2]);
					}
				}

				float max_std_dev = 0.0f;
				for (uint32_t i = 0; i < num_comps; i++)
					max_std_dev = basisu::maximum(max_std_dev, (float)comp_stats[i].get_std_dev());

				float yl = clampf(max_std_dev / params.m_max_med_smooth_std_dev, 0.0f, 1.0f);
				//yl = powf(yl, 2.0f);

				smooth_block_mse_scales(x, y) = lerp(params.m_med_smooth_max_mse_scale, smooth_block_mse_scales(x, y), yl);

				if (params.m_debug_images)
					med_smooth_vis(x, y).set((int)std::round(yl * 255.0f));
			}

			{
				tracked_stat_dbl comp_stats[4];

				const int S = 5;
				for (int yd = -S; yd < S; yd++)
				{
					for (int xd = -S; xd < S; xd++)
					{
						const color_rgba& p = orig_img.get_clamped((int)x + xd, (int)y + yd);

						comp_stats[0].update((float)p[0]);
						comp_stats[1].update((float)p[1]);
						comp_stats[2].update((float)p[2]);
					}
				}

				float max_std_dev = 0.0f;
				for (uint32_t i = 0; i < num_comps; i++)
					max_std_dev = basisu::maximum(max_std_dev, (float)comp_stats[i].get_std_dev());

				float yl = clampf(max_std_dev / params.m_max_ultra_smooth_std_dev, 0.0f, 1.0f);
				yl = powf(yl, 2.0f);
				
				smooth_block_mse_scales(x, y) = lerp(params.m_ultra_smooth_max_mse_scale, smooth_block_mse_scales(x, y), yl);

				if (params.m_debug_images)
					ultra_smooth_vis(x, y).set((int)std::round(yl * 255.0f));
			}

		}
	}

	if (params.m_debug_images)
	{
		save_png("dbg_smooth_vis.png", smooth_vis);
		save_png("dbg_med_smooth_vis.png", med_smooth_vis);
		save_png("dbg_ultra_smooth_vis.png", ultra_smooth_vis);

		image vis_img(width, height);

		float max_scale = 0.0f;
		for (uint32_t y = 0; y < height; y++)
			for (uint32_t x = 0; x < width; x++)
				max_scale = basisu::maximumf(max_scale, smooth_block_mse_scales(x, y));

		for (uint32_t y = 0; y < height; y++)
			for (uint32_t x = 0; x < width; x++)
				vis_img(x, y).set((int)std::round(smooth_block_mse_scales(x, y) * 255.0f / max_scale));

		save_png("scale_vis.png", vis_img);
	}

	if (pUltra_smooth_img)
		*pUltra_smooth_img = ultra_smooth_vis;
}

const float REALLY_DARK_I_THRESHOLD = 0.0625f;
const float REALLY_DARK_MSE_ERR_SCALE = 128.0f;
const float REALLY_DARK_DELTA_ITP_JND_SCALE = 5.0f;

static float compute_pixel_mse_itp(const vec3F& orig_pixel_itp, const vec3F& comp_pixel_itp, bool delta_itp_dark_adjustment)
{
	float delta_i = orig_pixel_itp[0] - comp_pixel_itp[0];
	float delta_t = orig_pixel_itp[1] - comp_pixel_itp[1];
	float delta_p = orig_pixel_itp[2] - comp_pixel_itp[2];
		
	float err = (delta_i * delta_i) + (delta_t * delta_t) + (delta_p * delta_p);

	if (delta_itp_dark_adjustment)
	{
		// We have to process a large range of inputs, including extremely dark inputs. 
		// Artifically amplify MSE on very dark pixels - otherwise they'll be overly compressed at higher lambdas.
		// This is to better handle very dark signals which could be explictly overexposed.
		float s = bu_math::smoothstep(0.0f, REALLY_DARK_I_THRESHOLD, orig_pixel_itp[0]);
		s = lerp(REALLY_DARK_MSE_ERR_SCALE, 1.0f, s);
		err *= s;
	}

	return err;
}

static float compute_block_mse_itp(uint32_t block_w, uint32_t block_h, const vec3F* pOrig_pixels_itp, const vec3F* pComp_pixels_itp, bool delta_itp_dark_adjustment)
{
	float total_mse = 0.0f;

	for (uint32_t y = 0; y < block_h; y++)
	{
		for (uint32_t x = 0; x < block_w; x++)
		{
			total_mse += compute_pixel_mse_itp(pOrig_pixels_itp[x + y * block_w], pComp_pixels_itp[x + y * block_w], delta_itp_dark_adjustment);
		} // x
	} // y

	return total_mse * (1.0f / (float)(block_w * block_h));
}

static float compute_block_ssim_itp(uint32_t block_w, uint32_t block_h, const vec3F* pOrig_pixels_itp, const vec3F* pComp_pixels_itp)
{
	const uint32_t n = block_w * block_h;
	assert(n <= 36);

	stats<float> x_stats[3], y_stats[3];
	comparative_stats<float> xy_cov[3];

	for (uint32_t c = 0; c < 3; c++)
	{
		x_stats[c].calc_simplified(n, &pOrig_pixels_itp[0][c], 3);
		y_stats[c].calc_simplified(n, &pComp_pixels_itp[0][c], 3);
	}

	for (uint32_t c = 0; c < 3; c++)
		xy_cov[c].calc_cov(n, &pOrig_pixels_itp[0][c], &pComp_pixels_itp[0][c], 3, 3, &x_stats[c], &y_stats[c]);

	float ssim[3];
	const double d = 1.0f, k1 = .01f, k2 = .03f;

	// weight mean error more highly to reduce blocking
	float ap = 1.5f, bp = 1.0f, cp = 1.0f;

	const double s_c1 = square(k1 * d), s_c2 = square(k2 * d);
	const double s_c3(s_c2 * .5f);

	for (uint32_t c = 0; c < 3; c++)
	{
		float lum = (float)((2.0f * x_stats[c].m_avg * y_stats[c].m_avg + s_c1) / (square(x_stats[c].m_avg) + square(y_stats[c].m_avg) + s_c1));
		lum = saturate(lum);

		float con = (float)((2.0f * x_stats[c].m_std_dev * y_stats[c].m_std_dev + s_c2) / (x_stats[c].m_var + y_stats[c].m_var + s_c2));
		con = saturate(con);

		float str = (float)((xy_cov[c].m_cov + s_c3) / (x_stats[c].m_std_dev * y_stats[c].m_std_dev + s_c3));
		str = saturate(str);

		ssim[c] = powf(lum, ap) * powf(con, bp) * powf(str, cp);
	}

#if 0
	float final_ssim = (ssim[0] * .4f + ssim[1] * .3f + ssim[2] * .3f);
#elif 1
	float final_ssim = ssim[0] * ssim[1] * ssim[2];
#else
	const float LP = .75f;
	float final_ssim = ssim[0] * powf((ssim[1] + ssim[2]) * .5f, LP);
#endif

	return final_ssim;
}

// delta ITP, 1.0 is JND (Rec. ITU-R BT.2124), modified for higher error at low light
static float compute_pixel_delta_itp(const vec3F& a, const vec3F& b, const vec3F& orig, bool delta_itp_dark_adjustment)
{
	float delta_i = a[0] - b[0];
	float delta_t = a[1] - b[1];
	float delta_p = a[2] - b[2];

	float err = 720.0f * sqrtf((delta_i * delta_i) + (delta_t * delta_t) + (delta_p * delta_p));

	float s = bu_math::smoothstep(0.0f, REALLY_DARK_I_THRESHOLD, orig[0]);
	
	if (delta_itp_dark_adjustment)
	{
		// This is to better handle very dark signals which could be explictly overexposed.
		s = lerp(REALLY_DARK_DELTA_ITP_JND_SCALE, 1.0f, s);
		err *= s;
	}

	return err;
}

struct candidate_encoding
{
	encoding_type m_encoding_type;
		
	basist::half_float m_solid_color[3];

	uint32_t m_run_len;

	vec3F m_comp_pixels[MAX_BLOCK_H][MAX_BLOCK_W]; // [y][x]
	vec3F m_comp_pixels_itp[MAX_BLOCK_H][MAX_BLOCK_W]; // [y][x]
		
	endpoint_mode m_endpoint_mode;
	block_mode m_block_mode;

	bitwise_coder m_coder;
		
	// The block to code, which may not be valid ASTC. This may have to be transcoded (by requantizing the weights/endpoints) before it's valid ASTC.
	// Note the endpoints may be coded endpoints OR transcoded endpoints, depending on the encoding type.
	astc_helpers::log_astc_block m_coded_log_blk; 

	// The block the decoder outputs.
	astc_helpers::log_astc_block m_decomp_log_blk;

	int m_reuse_delta_index;

	float m_t, m_d, m_bits;
					
	candidate_encoding()
	{
		clear();
	}

	candidate_encoding(const candidate_encoding &other)
	{
		*this = other;
	}

	candidate_encoding(candidate_encoding&& other)
	{
		*this = std::move(other);
	}

	candidate_encoding& operator=(const candidate_encoding& rhs)
	{
		if (this == &rhs)
			return *this;

		m_encoding_type = rhs.m_encoding_type;
		memcpy(m_solid_color, rhs.m_solid_color, sizeof(m_solid_color));
		m_run_len = rhs.m_run_len;
		memcpy(m_comp_pixels, rhs.m_comp_pixels, sizeof(m_comp_pixels));
		m_endpoint_mode = rhs.m_endpoint_mode;
		m_block_mode = rhs.m_block_mode;
		m_coder = rhs.m_coder;
		m_coded_log_blk = rhs.m_coded_log_blk;
		m_decomp_log_blk = rhs.m_decomp_log_blk;
		m_reuse_delta_index = rhs.m_reuse_delta_index;
		
		return *this;
	}

	candidate_encoding& operator=(candidate_encoding&& rhs)
	{
		if (this == &rhs)
			return *this;

		m_encoding_type = rhs.m_encoding_type;
		memcpy(m_solid_color, rhs.m_solid_color, sizeof(m_solid_color));
		m_run_len = rhs.m_run_len;
		memcpy(m_comp_pixels, rhs.m_comp_pixels, sizeof(m_comp_pixels));
		m_endpoint_mode = rhs.m_endpoint_mode;
		m_block_mode = rhs.m_block_mode;
		m_coder = std::move(rhs.m_coder);
		m_coded_log_blk = rhs.m_coded_log_blk;
		m_decomp_log_blk = rhs.m_decomp_log_blk;
		m_reuse_delta_index = rhs.m_reuse_delta_index;

		return *this;
	}

	void clear()
	{
		m_encoding_type = encoding_type::cInvalid;

		clear_obj(m_solid_color);

		m_run_len = 0;

		clear_obj(m_comp_pixels);
						
		m_endpoint_mode = endpoint_mode::cInvalid;
		m_block_mode = block_mode::cInvalid;

		m_coder.restart();
		
		m_coded_log_blk.clear();
		m_decomp_log_blk.clear();

		m_t = 0;
		m_d = 0;
		m_bits = 0;
		
		m_reuse_delta_index = 0;
	}
};

bool decode_astc_block(uint32_t block_w, uint32_t block_h, astc_helpers::log_astc_block &log_blk, vec3F *pPixels)
{
	assert((block_w <= 6) && (block_h <= 6));

	half_vec4 decoded_pixels_half4[6 * 6]; // [y][x]
	bool status = astc_helpers::decode_block(log_blk, decoded_pixels_half4, block_w, block_h, astc_helpers::cDecodeModeHDR16);
	assert(status);

	if (!status)
		return false;

	for (uint32_t y = 0; y < block_h; y++)
	{
		for (uint32_t x = 0; x < block_w; x++)
		{
			pPixels[x + y * block_w].set(
				basist::half_to_float(decoded_pixels_half4[x + y * block_w][0]),
				basist::half_to_float(decoded_pixels_half4[x + y * block_w][1]),
				basist::half_to_float(decoded_pixels_half4[x + y * block_w][2]));
		} // x 
	} //y

	return true;
}

static inline bool validate_log_blk(const astc_helpers::log_astc_block &decomp_blk)
{
	astc_helpers::astc_block phys_blk;
	return astc_helpers::pack_astc_block(phys_blk, decomp_blk);
}

#define SYNC_MARKERS (0)

static bool decode_file(const uint8_vec& comp_data, vector2D<astc_helpers::astc_block>& decoded_blocks, uint32_t &width, uint32_t &height)
{
	interval_timer tm;
	tm.start();

	const uint32_t BLOCK_W = 6, BLOCK_H = 6;

	width = 0;
	height = 0;

	if (comp_data.size() <= 2*3)
		return false;

	basist::bitwise_decoder decoder;
	if (!decoder.init(comp_data.data(), comp_data.size_u32()))
		return false;

	if (decoder.get_bits(16) != 0xABCD)
		return false;

	width = decoder.get_bits(16);
	height = decoder.get_bits(16);
		
	if (!width || !height || (width > MAX_ASTC_HDR_6X6_DIM) || (height > MAX_ASTC_HDR_6X6_DIM))
		return false;

	const uint32_t num_blocks_x = (width + BLOCK_W - 1) / BLOCK_W;
	const uint32_t num_blocks_y = (height + BLOCK_H - 1) / BLOCK_H;
	const uint32_t total_blocks = num_blocks_x * num_blocks_y;

	decoded_blocks.resize(num_blocks_x, num_blocks_y);
	//memset(decoded_blocks.get_ptr(), 0, decoded_blocks.size_in_bytes());

	vector2D<astc_helpers::log_astc_block> decoded_log_blocks(num_blocks_x, num_blocks_y);
	//memset(decoded_log_blocks.get_ptr(), 0, decoded_log_blocks.size_in_bytes());

	uint32_t cur_bx = 0, cur_by = 0;
	uint32_t step_counter = 0;
	BASISU_NOTE_UNUSED(step_counter);
		
	while (cur_by < num_blocks_y)
	{
		step_counter++;
		
		//if ((cur_bx == 9) && (cur_by == 13))
		//	printf("!");

#if SYNC_MARKERS
		uint32_t mk = decoder.get_bits(16);
		if (mk != 0xDEAD)
		{
			printf("!");
			assert(0);
			return false;
		}
#endif
		if (decoder.get_bits_remaining() < 1)
			return false;

		encoding_type et = encoding_type::cBlock;

		uint32_t b0 = decoder.get_bits(1);
		if (!b0)
		{
			uint32_t b1 = decoder.get_bits(1);
			if (b1)
				et = encoding_type::cReuse;
			else
			{
				uint32_t b2 = decoder.get_bits(1);
				if (b2)
					et = encoding_type::cSolid;
				else
					et = encoding_type::cRun;
			}
		}

		switch (et)
		{
		case encoding_type::cRun:
		{
			if (!cur_bx && !cur_by)
				return false;

			const uint32_t run_len = decoder.decode_vlc(5) + 1;
			
			uint32_t num_blocks_remaining = total_blocks - (cur_bx + cur_by * num_blocks_x);
			if (run_len > num_blocks_remaining)
				return false;
						
			uint32_t prev_bx = cur_bx, prev_by = cur_by;

			if (cur_bx)
				prev_bx--;
			else
			{
				prev_bx = num_blocks_x - 1;
				prev_by--;
			}

			const astc_helpers::log_astc_block& prev_log_blk = decoded_log_blocks(prev_bx, prev_by);
			const astc_helpers::astc_block& prev_phys_blk = decoded_blocks(prev_bx, prev_by);

			for (uint32_t i = 0; i < run_len; i++)
			{
				decoded_log_blocks(cur_bx, cur_by) = prev_log_blk;
				decoded_blocks(cur_bx, cur_by) = prev_phys_blk;

				cur_bx++;
				if (cur_bx == num_blocks_x)
				{
					cur_bx = 0;
					cur_by++;
				}
			}

			break;
		}
		case encoding_type::cSolid:
		{
			const basist::half_float rh = (basist::half_float)decoder.get_bits(15);
			const basist::half_float gh = (basist::half_float)decoder.get_bits(15);
			const basist::half_float bh = (basist::half_float)decoder.get_bits(15);

			astc_helpers::log_astc_block& log_blk = decoded_log_blocks(cur_bx, cur_by);

			log_blk.clear();
			log_blk.m_solid_color_flag_hdr = true;
			log_blk.m_solid_color[0] = rh;
			log_blk.m_solid_color[1] = gh;
			log_blk.m_solid_color[2] = bh;
			log_blk.m_solid_color[3] = basist::float_to_half(1.0f);

			bool status = astc_helpers::pack_astc_block(decoded_blocks(cur_bx, cur_by), log_blk);
			if (!status)
				return false;

			cur_bx++;
			if (cur_bx == num_blocks_x)
			{
				cur_bx = 0;
				cur_by++;
			}
			
			break;
		}
		case encoding_type::cReuse:
		{
			if (!cur_bx && !cur_by)
				return false;

			const uint32_t reuse_delta_index = decoder.get_bits(REUSE_XY_DELTA_BITS);

			const int reuse_delta_x = g_reuse_xy_deltas[reuse_delta_index].m_x;
			const int reuse_delta_y = g_reuse_xy_deltas[reuse_delta_index].m_y;

			const int prev_bx = cur_bx + reuse_delta_x, prev_by = cur_by + reuse_delta_y;
			if ((prev_bx < 0) || (prev_bx >= (int)num_blocks_x))
				return false;
			if (prev_by < 0)
				return false;
			
			const astc_helpers::log_astc_block& prev_log_blk = decoded_log_blocks(prev_bx, prev_by);
			const astc_helpers::astc_block& prev_phys_blk = decoded_blocks(prev_bx, prev_by);

			if (prev_log_blk.m_solid_color_flag_hdr)
				return false;

			astc_helpers::log_astc_block& log_blk = decoded_log_blocks(cur_bx, cur_by);
			astc_helpers::astc_block& phys_blk = decoded_blocks(cur_bx, cur_by);
			
			log_blk = prev_log_blk;

			const uint32_t total_grid_weights = log_blk.m_grid_width * log_blk.m_grid_height * (log_blk.m_dual_plane ? 2 : 1);

			bool status = basist::astc_6x6_hdr::decode_values(decoder, total_grid_weights, log_blk.m_weight_ise_range, log_blk.m_weights);
			if (!status)
				return false;

			astc_helpers::log_astc_block decomp_blk;
			status = astc_helpers::unpack_block(&prev_phys_blk, decomp_blk, BLOCK_W, BLOCK_H);
			if (!status)
				return false;
			
			uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];
			basist::astc_6x6_hdr::requantize_astc_weights(total_grid_weights, log_blk.m_weights, log_blk.m_weight_ise_range, transcode_weights, decomp_blk.m_weight_ise_range);

			copy_weight_grid(log_blk.m_dual_plane, log_blk.m_grid_width, log_blk.m_grid_height, transcode_weights, decomp_blk);

			status = astc_helpers::pack_astc_block(phys_blk, decomp_blk);
			if (!status)
				return false;

			cur_bx++;
			if (cur_bx == num_blocks_x)
			{
				cur_bx = 0;
				cur_by++;
			}

			break;
		}
		case encoding_type::cBlock:
		{
			const block_mode bm = (block_mode)decoder.decode_truncated_binary((uint32_t)block_mode::cBMTotalModes);
			const endpoint_mode em = (endpoint_mode)decoder.decode_truncated_binary((uint32_t)endpoint_mode::cTotal);

			switch (em)
			{
			case endpoint_mode::cUseLeft:
			case endpoint_mode::cUseUpper:
			{
				int neighbor_bx = cur_bx, neighbor_by = cur_by;
				
				if (em == endpoint_mode::cUseLeft)
					neighbor_bx--;
				else
					neighbor_by--;

				if ((neighbor_bx < 0) || (neighbor_by < 0))
					return false;

				const astc_helpers::log_astc_block& neighbor_blk = decoded_log_blocks(neighbor_bx, neighbor_by);
				if (!neighbor_blk.m_color_endpoint_modes[0])
					return false;

				const block_mode_desc& bmd = g_block_mode_descs[(uint32_t)bm];
				const uint32_t num_endpoint_values = get_num_endpoint_vals(bmd.m_cem);

				if (bmd.m_cem != neighbor_blk.m_color_endpoint_modes[0])
					return false;

				astc_helpers::log_astc_block& log_blk = decoded_log_blocks(cur_bx, cur_by);
				astc_helpers::astc_block& phys_blk = decoded_blocks(cur_bx, cur_by);

				log_blk.clear();
				log_blk.m_num_partitions = 1;
				log_blk.m_color_endpoint_modes[0] = (uint8_t)bmd.m_cem;
				log_blk.m_endpoint_ise_range = neighbor_blk.m_endpoint_ise_range;
				log_blk.m_weight_ise_range = (uint8_t)bmd.m_weight_ise_range;
				log_blk.m_grid_width = (uint8_t)bmd.m_grid_x;
				log_blk.m_grid_height = (uint8_t)bmd.m_grid_y;
				log_blk.m_dual_plane = (uint8_t)bmd.m_dp;
				log_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;

				memcpy(log_blk.m_endpoints, neighbor_blk.m_endpoints, num_endpoint_values);

				const uint32_t total_grid_weights = bmd.m_grid_x * bmd.m_grid_y * (bmd.m_dp ? 2 : 1);

				bool status = decode_values(decoder, total_grid_weights, bmd.m_weight_ise_range, log_blk.m_weights);
				if (!status)
					return false;

				astc_helpers::log_astc_block decomp_blk;
				decomp_blk.clear();

				decomp_blk.m_num_partitions = 1;
				decomp_blk.m_color_endpoint_modes[0] = (uint8_t)bmd.m_cem;
				decomp_blk.m_endpoint_ise_range = (uint8_t)bmd.m_transcode_endpoint_ise_range;
				decomp_blk.m_weight_ise_range = (uint8_t)bmd.m_transcode_weight_ise_range;
				decomp_blk.m_dual_plane = bmd.m_dp;
				decomp_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;

				basist::astc_6x6_hdr::requantize_ise_endpoints(bmd.m_cem, log_blk.m_endpoint_ise_range, log_blk.m_endpoints, bmd.m_transcode_endpoint_ise_range, decomp_blk.m_endpoints);

				uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];
				basist::astc_6x6_hdr::requantize_astc_weights(total_grid_weights, log_blk.m_weights, bmd.m_weight_ise_range, transcode_weights, bmd.m_transcode_weight_ise_range);

				copy_weight_grid(bmd.m_dp, bmd.m_grid_x, bmd.m_grid_y, transcode_weights, decomp_blk);

				status = astc_helpers::pack_astc_block(phys_blk, decomp_blk);
				if (!status)
					return false;

				cur_bx++;
				if (cur_bx == num_blocks_x)
				{
					cur_bx = 0;
					cur_by++;
				}

				break;
			}
			case endpoint_mode::cUseLeftDelta:
			case endpoint_mode::cUseUpperDelta:
			{
				int neighbor_bx = cur_bx, neighbor_by = cur_by;

				if (em == endpoint_mode::cUseLeftDelta)
					neighbor_bx--;
				else
					neighbor_by--;

				if ((neighbor_bx < 0) || (neighbor_by < 0))
					return false;

				const astc_helpers::log_astc_block& neighbor_blk = decoded_log_blocks(neighbor_bx, neighbor_by);
				if (!neighbor_blk.m_color_endpoint_modes[0])
					return false;

				const block_mode_desc& bmd = g_block_mode_descs[(uint32_t)bm];
				const uint32_t num_endpoint_values = get_num_endpoint_vals(bmd.m_cem);

				if (bmd.m_cem != neighbor_blk.m_color_endpoint_modes[0])
					return false;

				astc_helpers::log_astc_block& log_blk = decoded_log_blocks(cur_bx, cur_by);
				astc_helpers::astc_block& phys_blk = decoded_blocks(cur_bx, cur_by);

				log_blk.clear();
				log_blk.m_num_partitions = 1;
				log_blk.m_color_endpoint_modes[0] = (uint8_t)bmd.m_cem;
				log_blk.m_dual_plane = bmd.m_dp;
				log_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;
				
				log_blk.m_endpoint_ise_range = (uint8_t)bmd.m_endpoint_ise_range;
				basist::astc_6x6_hdr::requantize_ise_endpoints(bmd.m_cem, neighbor_blk.m_endpoint_ise_range, neighbor_blk.m_endpoints, bmd.m_endpoint_ise_range, log_blk.m_endpoints);

				const int total_endpoint_delta_vals = 1 << NUM_ENDPOINT_DELTA_BITS;
				const int low_delta_limit = -(total_endpoint_delta_vals / 2); // high_delta_limit = (total_endpoint_delta_vals / 2) - 1;

				const auto& ise_to_rank = astc_helpers::g_dequant_tables.get_endpoint_tab(log_blk.m_endpoint_ise_range).m_ISE_to_rank;
				const auto& rank_to_ise = astc_helpers::g_dequant_tables.get_endpoint_tab(log_blk.m_endpoint_ise_range).m_rank_to_ISE;
				const int total_endpoint_levels = astc_helpers::get_ise_levels(log_blk.m_endpoint_ise_range);

				for (uint32_t i = 0; i < num_endpoint_values; i++)
				{
					int cur_val = ise_to_rank[log_blk.m_endpoints[i]];
					
					int delta = (int)decoder.get_bits(NUM_ENDPOINT_DELTA_BITS) + low_delta_limit;

					cur_val += delta;
					if ((cur_val < 0) || (cur_val >= total_endpoint_levels))
						return false;

					log_blk.m_endpoints[i] = rank_to_ise[cur_val];
				}

				log_blk.m_weight_ise_range = (uint8_t)bmd.m_weight_ise_range;
				log_blk.m_grid_width = (uint8_t)bmd.m_grid_x;
				log_blk.m_grid_height = (uint8_t)bmd.m_grid_y;

				const uint32_t total_grid_weights = bmd.m_grid_x * bmd.m_grid_y * (bmd.m_dp ? 2 : 1);

				bool status = decode_values(decoder, total_grid_weights, bmd.m_weight_ise_range, log_blk.m_weights);
				if (!status)
					return false;

				astc_helpers::log_astc_block decomp_blk;
				decomp_blk.clear();

				decomp_blk.m_num_partitions = 1;
				decomp_blk.m_color_endpoint_modes[0] = (uint8_t)bmd.m_cem;
				decomp_blk.m_endpoint_ise_range = (uint8_t)bmd.m_transcode_endpoint_ise_range;
				decomp_blk.m_weight_ise_range = (uint8_t)bmd.m_transcode_weight_ise_range;
				decomp_blk.m_dual_plane = (uint8_t)bmd.m_dp;
				decomp_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;

				basist::astc_6x6_hdr::requantize_ise_endpoints(bmd.m_cem, log_blk.m_endpoint_ise_range, log_blk.m_endpoints, bmd.m_transcode_endpoint_ise_range, decomp_blk.m_endpoints);

				uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];
				basist::astc_6x6_hdr::requantize_astc_weights(total_grid_weights, log_blk.m_weights, bmd.m_weight_ise_range, transcode_weights, bmd.m_transcode_weight_ise_range);

				copy_weight_grid(bmd.m_dp, bmd.m_grid_x, bmd.m_grid_y, transcode_weights, decomp_blk);

				status = astc_helpers::pack_astc_block(phys_blk, decomp_blk);
				if (!status)
					return false;

				cur_bx++;
				if (cur_bx == num_blocks_x)
				{
					cur_bx = 0;
					cur_by++;
				}

				break;
			}
			case endpoint_mode::cRaw:
			{
				const block_mode_desc& bmd = g_block_mode_descs[(uint32_t)bm];

				const uint32_t num_endpoint_values = get_num_endpoint_vals(bmd.m_cem);

				astc_helpers::log_astc_block& log_blk = decoded_log_blocks(cur_bx, cur_by);
				astc_helpers::astc_block& phys_blk = decoded_blocks(cur_bx, cur_by);

				log_blk.clear();
				log_blk.m_num_partitions = (uint8_t)bmd.m_num_partitions;
				
				for (uint32_t p = 0; p < bmd.m_num_partitions; p++)
					log_blk.m_color_endpoint_modes[p] = (uint8_t)bmd.m_cem;

				log_blk.m_endpoint_ise_range = (uint8_t)bmd.m_endpoint_ise_range;
				log_blk.m_weight_ise_range = (uint8_t)bmd.m_weight_ise_range;

				log_blk.m_grid_width = (uint8_t)bmd.m_grid_x;
				log_blk.m_grid_height = (uint8_t)bmd.m_grid_y;
				log_blk.m_dual_plane = (uint8_t)bmd.m_dp;
				log_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;

				if (bmd.m_num_partitions == 2)
				{
					const uint32_t unique_partition_index = decoder.decode_truncated_binary(NUM_UNIQUE_PARTITIONS2);
					log_blk.m_partition_id = (uint16_t)g_part2_unique_index_to_seed[unique_partition_index];
				}
				else if (bmd.m_num_partitions == 3)
				{
					const uint32_t unique_partition_index = decoder.decode_truncated_binary(NUM_UNIQUE_PARTITIONS3);
					log_blk.m_partition_id = (uint16_t)g_part3_unique_index_to_seed[unique_partition_index];
				}
				
				bool status = decode_values(decoder, num_endpoint_values * bmd.m_num_partitions, bmd.m_endpoint_ise_range, log_blk.m_endpoints);
				if (!status)
					return false;

				const uint32_t total_grid_weights = bmd.m_grid_x * bmd.m_grid_y * (bmd.m_dp ? 2 : 1);

				status = decode_values(decoder, total_grid_weights, bmd.m_weight_ise_range, log_blk.m_weights);
				if (!status)
					return false;

				astc_helpers::log_astc_block decomp_blk;
				decomp_blk.clear();
				
				decomp_blk.m_dual_plane = bmd.m_dp;
				decomp_blk.m_color_component_selector = (uint8_t)bmd.m_dp_channel;
				decomp_blk.m_partition_id = log_blk.m_partition_id;
								
				decomp_blk.m_num_partitions = (uint8_t)bmd.m_num_partitions;
				
				for (uint32_t p = 0; p < bmd.m_num_partitions; p++)
					decomp_blk.m_color_endpoint_modes[p] = (uint8_t)bmd.m_cem;

				decomp_blk.m_endpoint_ise_range = (uint8_t)bmd.m_transcode_endpoint_ise_range;
				decomp_blk.m_weight_ise_range = (uint8_t)bmd.m_transcode_weight_ise_range;

				for (uint32_t p = 0; p < bmd.m_num_partitions; p++)
					basist::astc_6x6_hdr::requantize_ise_endpoints(bmd.m_cem, bmd.m_endpoint_ise_range, log_blk.m_endpoints + num_endpoint_values * p, bmd.m_transcode_endpoint_ise_range, decomp_blk.m_endpoints + num_endpoint_values * p);

				uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];
				basist::astc_6x6_hdr::requantize_astc_weights(total_grid_weights, log_blk.m_weights, bmd.m_weight_ise_range, transcode_weights, bmd.m_transcode_weight_ise_range);

				copy_weight_grid(bmd.m_dp, bmd.m_grid_x, bmd.m_grid_y, transcode_weights, decomp_blk);

				status = astc_helpers::pack_astc_block(phys_blk, decomp_blk);
				if (!status)
					return false;

				cur_bx++;
				if (cur_bx == num_blocks_x)
				{
					cur_bx = 0;
					cur_by++;
				}

				break;
			}
			default:
			{
				assert(0);
				return false;
			}
			}

			break;
		}
		default:
		{
			assert(0);
			return false;
		}
		}
	}

	if (decoder.get_bits(16) != 0xA742)
	{
		fmt_error_printf("End marker not found!\n");
		return false;
	}

	//fmt_printf("Total decode_file() time: {} secs\n", tm.get_elapsed_secs());

	return true;
}

static bool unpack_physical_astc_block(const void* pBlock, uint32_t block_width, uint32_t block_height, vec4F* pPixels)
{
	astc_helpers::log_astc_block log_blk;
	if (!astc_helpers::unpack_block(pBlock, log_blk, block_width, block_height))
		return false;
	
	basist::half_float half_block[MAX_BLOCK_W * MAX_BLOCK_H][4];
	if (!astc_helpers::decode_block(log_blk, half_block, block_width, block_height, astc_helpers::cDecodeModeHDR16))
		return false;

	const uint32_t total_block_pixels = block_width * block_height;
	for (uint32_t p = 0; p < total_block_pixels; p++)
	{
		pPixels[p][0] = basist::half_to_float(half_block[p][0]);
		pPixels[p][1] = basist::half_to_float(half_block[p][1]);
		pPixels[p][2] = basist::half_to_float(half_block[p][2]);
		pPixels[p][3] = basist::half_to_float(half_block[p][3]);
	}

	return true;
}

static bool unpack_physical_astc_block_google(const void* pBlock, uint32_t block_width, uint32_t block_height, vec4F* pPixels)
{
	return basisu_astc::astc::decompress_hdr((float *)pPixels, (uint8_t*)pBlock, block_width, block_height);
}

static bool pack_bc6h_image(const imagef &src_img, vector2D<basist::bc6h_block> &bc6h_blocks, imagef *pPacked_bc6h_img, const fast_bc6h_params &enc_params)
{
	const uint32_t width = src_img.get_width();
	const uint32_t height = src_img.get_height();
	
	if (pPacked_bc6h_img)
		pPacked_bc6h_img->resize(width, height);

	interval_timer tm;
	double total_enc_time = 0.0f;
	BASISU_NOTE_UNUSED(total_enc_time);

	const uint32_t num_blocks_x = src_img.get_block_width(4);
	const uint32_t num_blocks_y = src_img.get_block_height(4);

	bc6h_blocks.resize(num_blocks_x, num_blocks_y);
				
	for (uint32_t by = 0; by < num_blocks_y; by++)
	{
		for (uint32_t bx = 0; bx < num_blocks_x; bx++)
		{
			// Extract source image block
			vec4F block_pixels[4][4]; // [y][x]
			src_img.extract_block_clamped(&block_pixels[0][0], bx * 4, by * 4, 4, 4);

			basist::half_float half_pixels[16 * 3]; // [y][x]

			for (uint32_t y = 0; y < 4; y++)
			{
				for (uint32_t x = 0; x < 4; x++)
				{
					for (uint32_t c = 0; c < 3; c++)
					{
						float v = block_pixels[y][x][c];

						basist::half_float h = basist::float_to_half(v);

						half_pixels[(x + y * 4) * 3 + c] = h;

					} // c

				} // x
			} // y

			basist::bc6h_block& bc6h_blk = bc6h_blocks(bx, by);

			tm.start();

			basist::astc_6x6_hdr::fast_encode_bc6h(half_pixels, &bc6h_blk, enc_params);

			total_enc_time += tm.get_elapsed_secs();

			if (pPacked_bc6h_img)
			{
				basist::half_float unpacked_blk[16 * 3];
				bool status = unpack_bc6h(&bc6h_blk, unpacked_blk, false);
				assert(status);
				if (!status)
				{
					fmt_error_printf("unpack_bc6h() failed\n");
					return false;
				}
							
				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						vec4F p;

						for (uint32_t c = 0; c < 3; c++)
						{
							float v = basist::half_to_float(unpacked_blk[(x + y * 4) * 3 + c]);
							p[c] = v;

						} // c

						p[3] = 1.0f;

						pPacked_bc6h_img->set_clipped(bx * 4 + x, by * 4 + y, p);
					} // x
				} // y
			}

		} // bx
	} // by

	//fmt_printf("Total BC6H encode time: {}\n", total_enc_time);

	return true;
}

static float dist_to_line_squared(const vec3F& p, const vec3F &line_org, const vec3F &line_dir)
{
	vec3F q(p - line_org);
	vec3F v(q - q.dot(line_dir) * line_dir);
	return v.dot(v);
}

static void estimate_partitions_mode7_and_11(
	uint32_t num_parts, // 2 or 3 partitions
	uint32_t num_unique_pats, const partition_pattern_vec* pUnique_pats, // list of all unique, canonicalized patterns
	uint32_t num_pats_to_examine, const uint32_t* pUnique_pat_indices_to_examine, // indices of pats to examine
	const vec3F *pHalf_pixels_as_floats, // block's half pixel values casted to floats
	const astc_hdr_codec_base_options& coptions, // options
	uint32_t num_desired_pats, 
	int *pDesired_pat_indices_mode11, int *pDesired_pat_indices_mode7) // output indices
{
	BASISU_NOTE_UNUSED(coptions);
	BASISU_NOTE_UNUSED(num_unique_pats);

	const uint32_t BLOCK_W = 6, BLOCK_H = 6, MAX_PARTS = 3; // BLOCK_T = 6 * 6
	assert(num_parts <= MAX_PARTS);

	struct candidate_res
	{
		float m_total_sq_dist;
		uint32_t m_index;
		bool operator< (const candidate_res& rhs) const { return m_total_sq_dist < rhs.m_total_sq_dist; }
	};

	const uint32_t MAX_CANDIDATES = 1024;
	assert(num_desired_pats && (num_desired_pats <= MAX_CANDIDATES));

	candidate_res mode11_candidates[MAX_CANDIDATES];
	candidate_res mode7_candidates[MAX_CANDIDATES];

	const vec3F grayscale_axis(0.5773502691f);
	
	for (uint32_t examine_iter = 0; examine_iter < num_pats_to_examine; examine_iter++)
	{
		const uint32_t unique_part_index = pUnique_pat_indices_to_examine[examine_iter];
		assert(unique_part_index < num_unique_pats);

		const partition_pattern_vec* pPat = &pUnique_pats[unique_part_index];

		vec3F part_means[MAX_PARTS];
		uint32_t part_total_texels[MAX_PARTS] = { 0 };

		for (uint32_t i = 0; i < num_parts; i++)
			part_means[i].clear();

		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				const uint32_t part_index = (*pPat)(x, y);
				assert(part_index < num_parts);

				part_means[part_index] += pHalf_pixels_as_floats[x + y * BLOCK_W];
				part_total_texels[part_index]++;

 			} // x
		} // y
		
		for (uint32_t i = 0; i < num_parts; i++)
		{
			assert(part_total_texels[i]);
			part_means[i] /= (float)part_total_texels[i];
		}

		float part_cov[MAX_PARTS][6];
		memset(part_cov, 0, sizeof(part_cov));

		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				const uint32_t part_index = (*pPat)(x, y);
				assert(part_index < num_parts);

				const vec3F p(pHalf_pixels_as_floats[x + y * BLOCK_W] - part_means[part_index]);

				const float r = p[0], g = p[1], b = p[2];

				part_cov[part_index][0] += r * r;
				part_cov[part_index][1] += r * g;
				part_cov[part_index][2] += r * b;
				part_cov[part_index][3] += g * g;
				part_cov[part_index][4] += g * b;
				part_cov[part_index][5] += b * b;

			} // x
		} // y

		// For each partition compute the total variance of all channels.
		float total_variance[MAX_PARTS];
		for (uint32_t part_index = 0; part_index < num_parts; part_index++)
			total_variance[part_index] = part_cov[part_index][0] + part_cov[part_index][3] + part_cov[part_index][5];

		vec3F part_axis[MAX_PARTS];
		float mode11_eigenvalue_est[MAX_PARTS]; // For each partition, compute the variance along the principle axis
		float mode7_eigenvalue_est[MAX_PARTS]; // For each partition, compute the variance along the principle axis

		for (uint32_t part_index = 0; part_index < num_parts; part_index++)
		{
			float* pCov = &part_cov[part_index][0];

			float xr = .9f, xg = 1.0f, xb = .7f;
			
			const uint32_t NUM_POWER_ITERS = 4;
			for (uint32_t iter = 0; iter < NUM_POWER_ITERS; iter++)
			{
				float r = xr * pCov[0] + xg * pCov[1] + xb * pCov[2];
				float g = xr * pCov[1] + xg * pCov[3] + xb * pCov[4];
				float b = xr * pCov[2] + xg * pCov[4] + xb * pCov[5];

				float m = maximumf(maximumf(fabsf(r), fabsf(g)), fabsf(b));

				if (m >= 1e-10f)
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

			float len_sq = xr * xr + xg * xg + xb * xb;
						
			if (len_sq < 1e-10f)
			{
				xr = grayscale_axis[0];
				xg = grayscale_axis[0];
				xb = grayscale_axis[0];
			}
			else
			{
				len_sq = 1.0f / sqrtf(len_sq);

				xr *= len_sq;
				xg *= len_sq;
				xb *= len_sq;
			}
			
			{
				// Transform the principle axis by the covariance matrix, which will scale the vector by its eigenvalue (the variance of the dataset projected onto the principle axis).
				float r = xr * pCov[0] + xg * pCov[1] + xb * pCov[2];
				float g = xr * pCov[1] + xg * pCov[3] + xb * pCov[4];
				float b = xr * pCov[2] + xg * pCov[4] + xb * pCov[5];

				// Estimate the principle eigenvalue by computing the magnitude of the transformed vector.
				// The result is the variance along the principle axis.
				//float z1 = sqrtf(r * r + g * g + b * b); // this works with the principle axis
				//float z2 = r * xr + g * xg + b * xb; // compute length projected along xr,xg,xb
				
				mode11_eigenvalue_est[part_index] = r * xr + g * xg + b * xb;
			}

			{
				const float yrgb = grayscale_axis[0];
				
				// Transform the grayscale axis by the covariance matrix, which will scale the vector by the eigenvalue (which is the variance of the dataset projected onto this vector).
				float r = yrgb * pCov[0] + yrgb * pCov[1] + yrgb * pCov[2];
				float g = yrgb * pCov[1] + yrgb * pCov[3] + yrgb * pCov[4];
				float b = yrgb * pCov[2] + yrgb * pCov[4] + yrgb * pCov[5];

				mode7_eigenvalue_est[part_index] = r * yrgb + g * yrgb + b * yrgb;
			}

		} // part_index
				
		// Compute the total variance (squared error) of the other 2 axes by subtracting the total variance of all channels by the variance of the principle axis.
		// TODO: Could also compute the ratio of the principle axis's variance vs. the total variance.
		float mode11_total_sq_dist_to_line_alt = 0.0f;
		for (uint32_t part_index = 0; part_index < num_parts; part_index++)
		{
			float d = maximum(0.0f, total_variance[part_index] - mode11_eigenvalue_est[part_index]);
			mode11_total_sq_dist_to_line_alt += d;
		}

		{
#if 0
			// TODO: This total distance can be computed rapidly. First compute the total variance of each channel (sum the diag entries of the covar matrix),
			// then compute the principle eigenvalue, and subtract. The result is the variance of the projection distances.
			float total_sq_dist_to_line = 0.0f;
			for (uint32_t i = 0; i < BLOCK_T; i++)
			{
				const uint32_t part_index = (*pPat)[i];
				assert(part_index < num_parts);

				total_sq_dist_to_line += dist_to_line_squared(pHalf_pixels_as_floats[i], part_means[part_index], part_axis[part_index]);
			}

			mode11_candidates[examine_iter].m_total_sq_dist = total_sq_dist_to_line;
#else
			mode11_candidates[examine_iter].m_total_sq_dist = mode11_total_sq_dist_to_line_alt;
#endif
			mode11_candidates[examine_iter].m_index = unique_part_index;
		}

		{
			float mode7_total_sq_dist_to_line_alt = 0.0f;
			for (uint32_t part_index = 0; part_index < num_parts; part_index++)
			{
				float d = maximum(0.0f, total_variance[part_index] - mode7_eigenvalue_est[part_index]);
				mode7_total_sq_dist_to_line_alt += d;
			}

			mode7_candidates[examine_iter].m_total_sq_dist = mode7_total_sq_dist_to_line_alt;
			mode7_candidates[examine_iter].m_index = unique_part_index;
		}

	} // examine_iter

	std::sort(&mode11_candidates[0], &mode11_candidates[num_pats_to_examine]);
	std::sort(&mode7_candidates[0], &mode7_candidates[num_pats_to_examine]);

	for (uint32_t i = 0; i < num_desired_pats; i++)
		pDesired_pat_indices_mode11[i] = mode11_candidates[i].m_index;

	for (uint32_t i = 0; i < num_desired_pats; i++)
		pDesired_pat_indices_mode7[i] = mode7_candidates[i].m_index;
}

static void estimate_partitions_mode7(
	uint32_t num_parts, // 2 or 3 partitions
	uint32_t num_unique_pats, const partition_pattern_vec* pUnique_pats, // list of all unique, canonicalized patterns
	uint32_t num_pats_to_examine, const uint32_t* pUnique_pat_indices_to_examine, // indices of pats to examine
	const vec3F* pHalf_pixels_as_floats, // block's half pixel values casted to floats
	const astc_hdr_codec_base_options& coptions, // options
	uint32_t num_desired_pats, uint32_t* pDesired_pat_indices) // output indices
{
	BASISU_NOTE_UNUSED(coptions);
	BASISU_NOTE_UNUSED(num_unique_pats);

	const uint32_t BLOCK_W = 6, BLOCK_H = 6, BLOCK_T = 6 * 6, MAX_PARTS = 3;
	assert(num_parts <= MAX_PARTS);

	struct candidate_res
	{
		float m_total_sq_dist;
		uint32_t m_index;
		bool operator< (const candidate_res& rhs) const { return m_total_sq_dist < rhs.m_total_sq_dist; }
	};

	const uint32_t MAX_CANDIDATES = 1024;
	assert(num_desired_pats && (num_desired_pats <= MAX_CANDIDATES));

	candidate_res candidates[MAX_CANDIDATES];

	for (uint32_t examine_iter = 0; examine_iter < num_pats_to_examine; examine_iter++)
	{
		const uint32_t unique_part_index = pUnique_pat_indices_to_examine[examine_iter];
		assert(unique_part_index < num_unique_pats);

		const partition_pattern_vec* pPat = &pUnique_pats[unique_part_index];

		vec3F part_means[MAX_PARTS];
		uint32_t part_total_texels[MAX_PARTS] = { 0 };

		for (uint32_t i = 0; i < num_parts; i++)
			part_means[i].clear();

		for (uint32_t y = 0; y < BLOCK_H; y++)
		{
			for (uint32_t x = 0; x < BLOCK_W; x++)
			{
				const uint32_t part_index = (*pPat)(x, y);
				assert(part_index < num_parts);

				part_means[part_index] += pHalf_pixels_as_floats[x + y * BLOCK_W];
				part_total_texels[part_index]++;

			} // x
		} // y

		for (uint32_t i = 0; i < num_parts; i++)
		{
			assert(part_total_texels[i]);
			part_means[i] /= (float)part_total_texels[i];
		}

		vec3F part_axis(0.5773502691f);
		
		// TODO: This total distance can be computed rapidly. First compute the total variance of each channel (sum the diag entries of the covar matrix),
		// then compute the principle eigenvalue, and subtract. The result is the variance of the projection distances.
		float total_sq_dist_to_line = 0.0f;
		for (uint32_t i = 0; i < BLOCK_T; i++)
		{
			const uint32_t part_index = (*pPat)[i];
			assert(part_index < num_parts);

			total_sq_dist_to_line += dist_to_line_squared(pHalf_pixels_as_floats[i], part_means[part_index], part_axis);
		}

		candidates[examine_iter].m_total_sq_dist = total_sq_dist_to_line;

		candidates[examine_iter].m_index = unique_part_index;

	} // examine_iter

	std::sort(&candidates[0], &candidates[num_pats_to_examine]);

	for (uint32_t i = 0; i < num_desired_pats; i++)
		pDesired_pat_indices[i] = candidates[i].m_index;
}

static float calc_deblocking_penalty_itp(
	uint32_t bx, uint32_t by, uint32_t width, uint32_t height,
	const imagef& pass_src_img_itp, const candidate_encoding& candidate)
{
	float total_deblock_penalty = 0.0f;

	float total_orig_mse = 0.0f, total_comp_mse = 0.0f;
	uint32_t total_c = 0;

	for (uint32_t b = 0; b < 4; b++)
	{
		for (uint32_t i = 0; i < 6; i++)
		{
			int ox = 0, oy = 0, qx = 0, qy = 0;

			switch (b)
			{
			case 0:
				ox = bx * 6 + i; oy = (by - 1) * 6 + 5;
				qx = bx * 6 + i; qy = by * 6;
				break;
			case 1:
				ox = bx * 6 + i; oy = (by + 1) * 6;
				qx = bx * 6 + i; qy = by * 6 + 5;
				break;
			case 2:
				ox = (bx - 1) * 6 + 5; oy = by * 6 + i;
				qx = bx * 6; qy = by * 6 + i;
				break;
			case 3:
				ox = (bx + 1) * 6; oy = by * 6 + i;
				qx = bx * 6 + 5; qy = by * 6 + i;
				break;
			}

			if ((ox < 0) || (oy < 0) || (ox >= (int)width) || (oy >= (int)height))
				continue;

			const vec3F& o_pixel_itp = pass_src_img_itp(ox, oy);
			const vec3F& q_pixel_itp = pass_src_img_itp(qx, qy);

			const vec3F &d_pixel_itp = candidate.m_comp_pixels_itp[qy - by * 6][qx - bx * 6]; // compressed block
			
			vec3F orig_delta_v(o_pixel_itp - q_pixel_itp);
			total_orig_mse += square(orig_delta_v[0]) + square(orig_delta_v[1]) + square(orig_delta_v[2]);

			vec3F d_delta_v(o_pixel_itp - d_pixel_itp);
			total_comp_mse += square(d_delta_v[0]) + square(d_delta_v[1]) + square(d_delta_v[2]);

			total_c++;
		}
	}

	if (total_c)
	{
		total_orig_mse /= (float)total_c;
		total_comp_mse /= (float)total_c;

		if (total_orig_mse)
		{
			total_deblock_penalty = fabsf((total_comp_mse - total_orig_mse) / total_orig_mse);
		}
	}

	return total_deblock_penalty;
}

static bool calc_strip_size(
	float lambda,
	uint32_t num_blocks_y, uint32_t total_threads, bool force_one_strip,
	uint32_t& res_total_strips, uint32_t& res_rows_per_strip, astc_hdr_6x6_global_config &global_cfg)
{
	uint32_t total_strips = 1;

	if (lambda == 0.0f)
	{
		if (!force_one_strip)
		{
			total_strips = total_threads;
		}
	}
	else
	{
		const uint32_t MIN_DESIRED_STRIPS = 8;
		const uint32_t MAX_TARGET_STRIPS = 32;
		const uint32_t TARGET_ASTC_6X6_ROWS_PER_STRIP = 12;

		if (!force_one_strip)
		{
			total_strips = maximum<uint32_t>(1, num_blocks_y / TARGET_ASTC_6X6_ROWS_PER_STRIP);

			if (num_blocks_y >= MIN_DESIRED_STRIPS * 2)
				total_strips = maximum(total_strips, MIN_DESIRED_STRIPS);
		}

		total_strips = minimum(total_strips, MAX_TARGET_STRIPS);
	}

	uint32_t rows_per_strip = 0;
	if (total_strips <= 1)
	{
		rows_per_strip = num_blocks_y;
	}
	else
	{
		rows_per_strip = (num_blocks_y / total_strips) & ~1;
		
		if (rows_per_strip < 2)
			rows_per_strip = 2;// num_blocks_y;
	}
		
	assert((rows_per_strip == num_blocks_y) || ((rows_per_strip & 1) == 0));

	total_strips = (num_blocks_y + rows_per_strip - 1) / rows_per_strip;
	
	if (global_cfg.m_debug_output)
	{
		fmt_printf("num_blocks_y: {}, total_threads : {}, Total strips : {}\n", num_blocks_y, total_threads, total_strips);
		fmt_printf("ASTC 6x6 block rows per strip: {}\n", rows_per_strip);
		fmt_printf("ASTC 6x6 block rows on final strip: {}\n", num_blocks_y - (total_strips - 1) * rows_per_strip);
	}

	uint32_t total_rows = 0;
	for (uint32_t strip_index = 0; strip_index < total_strips; strip_index++)
	{
		uint32_t strip_first_by = strip_index * rows_per_strip;
		uint32_t strip_last_by = minimum<uint32_t>(strip_first_by + rows_per_strip - 1, num_blocks_y);

		if (strip_index == (total_strips - 1))
			strip_last_by = num_blocks_y - 1;

		uint32_t num_strip_block_rows = (strip_last_by - strip_first_by) + 1;
		total_rows += num_strip_block_rows;

		if (global_cfg.m_debug_output)
			fmt_printf("Strip row: {}, total block rows: {}\n", strip_index, num_strip_block_rows);
	}

	if (total_rows != num_blocks_y)
	{
		fmt_error_printf("Strip calc failed\n");
		return false;
	}

	res_total_strips = total_strips;
	res_rows_per_strip = rows_per_strip;

	return true;
}

static void convet_rgb_image_to_itp(const imagef &src_img, imagef &dst_img, const astc_hdr_6x6_global_config& cfg)
{
	const uint32_t width = src_img.get_width(), height = src_img.get_height();

	dst_img.resize(width, height);

	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			vec3F src_rgb(src_img(x, y));

			vec3F src_itp;
			linear_rgb_to_itp(src_rgb, src_itp, cfg);

			dst_img(x, y) = src_itp;
		}
	}
}

const uint32_t BLOCK_W = 6, BLOCK_H = 6;
const uint32_t NUM_BLOCK_PIXELS = BLOCK_W * BLOCK_H;

const float SOLID_PENALTY = 4.0f;
const float REUSE_PENALTY = 1.0f;
const float RUN_PENALTY = 10.0f;

const float MSE_WEIGHT = 300000.0f;
const float SSIM_WEIGHT = 200.0f;
const float TWO_LEVEL_PENALTY = 1.425f;
const float SWITCH_TO_GAUSSIAN_FILTERED_THRESH1_D_SSIM = .04f;
const float SWITCH_TO_GAUSSIAN_FILTERED_THRESH2_D_SSIM = .04f;
const float COMPLEX_BLOCK_WEIGHT_GRID_2X2_MSE_PENALTY = 1.5f;
const float COMPLEX_BLOCK_WEIGHT_GRID_3X3_MSE_PENALTY = 1.25f;
const float COMPLEX_BLOCK_WEIGHT_GRID_4X4_MSE_PENALTY = 1.15f;

struct uastc_hdr_6x6_debug_state
{
	uint32_t m_encoding_type_hist[(uint32_t)encoding_type::cTotal] = { 0 };
	uint32_t m_endpoint_mode_hist[(uint32_t)endpoint_mode::cTotal] = { 0 };
	uint32_t m_block_mode_hist[(uint32_t)block_mode::cBMTotalModes] = { 0 };
	uint64_t m_block_mode_total_bits[(uint32_t)block_mode::cBMTotalModes] = { 0 };

	basisu::vector< basisu::stats<float> > m_block_mode_comp_stats[(uint32_t)block_mode::cBMTotalModes][3];
	basisu::vector< basisu::comparative_stats<float> > m_block_mode_comparative_stats[(uint32_t)block_mode::cBMTotalModes][3];

	std::atomic<uint32_t> m_total_gaussian1_blocks;
	std::atomic<uint32_t> m_total_gaussian2_blocks;
	std::atomic<uint32_t> m_total_filter_horizontal;
	std::atomic<uint32_t> m_detail_stats[5];
	std::atomic<uint32_t> m_total_mode7_skips;

	std::atomic<uint32_t> m_total_blocks_compressed;

	std::atomic<uint32_t> m_total_candidates_considered;
	std::atomic<uint32_t> m_max_candidates_considered;

	std::atomic<uint32_t> m_total_part2_stats[4];
	std::atomic<uint32_t> m_dp_stats[5];

	std::atomic<uint32_t> m_reuse_num_parts[4];
	std::atomic<uint32_t> m_reuse_total_dp;

	imagef m_stat_vis;
	std::mutex m_stat_vis_mutex;

	image m_part_vis;
	image m_mode_vis;
	image m_mode_vis2;
	image m_grid_vis;
	image m_enc_vis;
	std::mutex m_vis_image_mutex;

	std::atomic<uint32_t> m_comp_level_hist[ASTC_HDR_6X6_MAX_COMP_LEVEL + 1];
		
	std::atomic<uint32_t> m_total_jnd_replacements;

	std::mutex m_stats_mutex;

	uastc_hdr_6x6_debug_state()
	{
		for (uint32_t i = 0; i < (uint32_t)block_mode::cBMTotalModes; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				m_block_mode_comp_stats[i][j].reserve(512);
				m_block_mode_comparative_stats[i][j].reserve(512);
			}
		}
	}
	
	void init(uint32_t width, uint32_t height)
	{
		m_stat_vis.resize(width, height);
		m_part_vis.resize(width, height);
		m_mode_vis.resize(width, height);
		m_mode_vis2.resize(width, height);
		m_grid_vis.resize(width, height);
		m_enc_vis.resize(width, height);

		basisu::clear_obj(m_encoding_type_hist);
		basisu::clear_obj(m_endpoint_mode_hist);
		basisu::clear_obj(m_block_mode_hist);
		basisu::clear_obj(m_block_mode_total_bits);
		
		for (uint32_t i = 0; i < (uint32_t)block_mode::cBMTotalModes; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				m_block_mode_comp_stats[i][j].clear();
				m_block_mode_comparative_stats[i][j].clear();
			}
		}

		m_total_gaussian1_blocks.store(0);
		m_total_gaussian2_blocks.store(0);
		m_total_filter_horizontal.store(0);
		for (uint32_t i = 0; i < std::size(m_detail_stats); i++)
			m_detail_stats[i].store(0);
		m_total_mode7_skips.store(0);

		for (uint32_t i = 0; i < std::size(m_comp_level_hist); i++)
			m_comp_level_hist[i].store(0);

		m_total_blocks_compressed.store(0);

		m_total_candidates_considered.store(0);
		m_max_candidates_considered.store(0);

		for (uint32_t i = 0; i < std::size(m_total_part2_stats); i++)
			m_total_part2_stats[i].store(0);
		
		for (uint32_t i = 0; i < std::size(m_dp_stats); i++)
			m_dp_stats[i].store(0);

		for (uint32_t i = 0; i < std::size(m_reuse_num_parts); i++)
			m_reuse_num_parts[i] .store(0);

		m_reuse_total_dp.store(0);

		m_total_jnd_replacements.store(0);
	}

	void print(uint32_t total_blocks) const
	{
		fmt_printf("Total blocks: {}\n", total_blocks);
		fmt_printf("Total JND replacements: {} {3.2}%\n", m_total_jnd_replacements, (float)m_total_jnd_replacements * 100.0f / (float)total_blocks);
		fmt_printf("Comp level histogram: {} {} {} {} {}\n", m_comp_level_hist[0], m_comp_level_hist[1], m_comp_level_hist[2], m_comp_level_hist[3], m_comp_level_hist[4]);
		fmt_printf("Total gaussian 1 blocks: {} {3.2}%\n", m_total_gaussian1_blocks, (float)m_total_gaussian1_blocks * 100.0f / (float)total_blocks);
		fmt_printf("Total gaussian 2 blocks: {} {3.2}%\n", m_total_gaussian2_blocks, (float)m_total_gaussian2_blocks * 100.0f / (float)total_blocks);
		fmt_printf("Total filter horizontal: {} {3.2}%\n", m_total_filter_horizontal, (float)m_total_filter_horizontal * 100.0f / (float)total_blocks);
		fmt_printf("Detail stats: Detailed block low grid skip: {}, Blurry block skip: {}, Very blurry block skip: {}, NH:{} H:{}\n", m_detail_stats[0], m_detail_stats[1], m_detail_stats[2], m_detail_stats[3], m_detail_stats[4]);
		fmt_printf("Total mode7 skips: {}\n", m_total_mode7_skips);

		fmt_printf("Total candidates: {}, {} avg per block\n", m_total_candidates_considered, (float)m_total_candidates_considered / (float)total_blocks);
		fmt_printf("Max ever candidates: {}\n", m_max_candidates_considered);

		fmt_printf("Part2/3 stats: {} {} {} {}\n", m_total_part2_stats[0], m_total_part2_stats[1], m_total_part2_stats[2], m_total_part2_stats[3]);
		fmt_printf("Dual plane stats: {} {} {} {} {}\n", m_dp_stats[0], m_dp_stats[1], m_dp_stats[2], m_dp_stats[3], m_dp_stats[4]);
		fmt_printf("Reuse total dual plane: {}\n", m_reuse_total_dp);
		fmt_printf("Reuse part stats: {} {} {}\n", m_reuse_num_parts[1], m_reuse_num_parts[2], m_reuse_num_parts[3]);

		fmt_printf("\nEncoding type histogram:\n");
		for (uint32_t i = 0; i < std::size(m_encoding_type_hist); i++)
			fmt_printf("{}: {}\n", i, m_encoding_type_hist[i]);

		fmt_printf("\nEndpoint mode histogram:\n");
		for (uint32_t i = 0; i < std::size(m_endpoint_mode_hist); i++)
			fmt_printf("{}: {}\n", i, m_endpoint_mode_hist[i]);

		fmt_printf("\nBlock mode histogram:\n");

		uint32_t total_dp = 0, total_sp = 0;
		uint32_t total_mode11 = 0, total_mode7 = 0;
		uint32_t part_hist[3] = { 0 };
		uint32_t part2_mode7_total = 0, part2_mode11_total = 0;
		uint32_t total_used_modes = 0;
		for (uint32_t i = 0; i < std::size(m_block_mode_hist); i++)
		{
			const auto& bm_desc = g_block_mode_descs[i];

			const uint32_t total_uses = m_block_mode_hist[i];

			if (bm_desc.m_dp)
				total_dp += total_uses;
			else
				total_sp += total_uses;

			if (bm_desc.m_cem == 7)
				total_mode7 += total_uses;
			else
				total_mode11 += total_uses;

			part_hist[bm_desc.m_num_partitions - 1] += total_uses;

			if (bm_desc.m_num_partitions == 2)
			{
				if (bm_desc.m_cem == 7)
					part2_mode7_total += total_uses;
				else
				{
					assert(bm_desc.m_cem == 11);
					part2_mode11_total += total_uses;
				}
			}

			float avg_std_dev = 0.0f;
			float avg_cross_correlations[3] = { 0 };

			if (m_block_mode_comp_stats[i][0].size())
			{
				const uint32_t num_uses = m_block_mode_comp_stats[i][0].size_u32();

				for (uint32_t j = 0; j < num_uses; j++)
					avg_std_dev += (float)maximum(m_block_mode_comp_stats[i][0][j].m_std_dev, m_block_mode_comp_stats[i][1][j].m_std_dev, m_block_mode_comp_stats[i][2][j].m_std_dev);
				avg_std_dev /= (float)num_uses;

				for (uint32_t j = 0; j < num_uses; j++)
				{
					avg_cross_correlations[0] += fabsf((float)m_block_mode_comparative_stats[i][0][j].m_pearson);
					avg_cross_correlations[1] += fabsf((float)m_block_mode_comparative_stats[i][1][j].m_pearson);
					avg_cross_correlations[2] += fabsf((float)m_block_mode_comparative_stats[i][2][j].m_pearson);
				}

				avg_cross_correlations[0] /= (float)num_uses;
				avg_cross_correlations[1] /= (float)num_uses;
				avg_cross_correlations[2] /= (float)num_uses;
			}

			fmt_printf("{ 2}: uses: { 6}, cem: {}, dp: {} chan: {}, parts: {}, grid: {}x{}, endpoint levels: {}, weight levels: {}, Avg bits: {}, Avg Max Std Dev: {}, RG: {} RB: {} GB: {}\n", i, total_uses,
				bm_desc.m_cem,
				bm_desc.m_dp, bm_desc.m_dp_channel,
				bm_desc.m_num_partitions,
				bm_desc.m_grid_x, bm_desc.m_grid_y,
				astc_helpers::get_ise_levels(bm_desc.m_endpoint_ise_range),
				astc_helpers::get_ise_levels(bm_desc.m_weight_ise_range),
				total_uses ? ((double)m_block_mode_total_bits[i] / total_uses) : 0.0f,
				avg_std_dev, avg_cross_correlations[0], avg_cross_correlations[1], avg_cross_correlations[2]);

			if (total_uses)
				total_used_modes++;
		}

		fmt_printf("Total used modes: {}\n", total_used_modes);

		fmt_printf("Total single plane: {}, total dual plane: {}\n", total_sp, total_dp);
		fmt_printf("Total mode 11: {}, mode 7: {}\n", total_mode11, total_mode7);
		fmt_printf("Partition histogram: {} {} {}\n", part_hist[0], part_hist[1], part_hist[2]);
		fmt_printf("2 subset mode 7 uses: {}, mode 11 uses: {}\n", part2_mode7_total, part2_mode11_total);
	}
};

struct uastc_hdr_6x6_encode_state
{
	astc_hdr_codec_base_options master_coptions;
		
	imagef src_img;
		
	imagef src_img_filtered1;
	imagef src_img_filtered2;

	imagef src_img_itp;
	imagef src_img_filtered1_itp;
	imagef src_img_filtered2_itp;

	vector2D<float> smooth_block_mse_scales;

	imagef packed_img;

	basisu::vector<bitwise_coder> strip_bits;

	basisu::vector2D<astc_helpers::astc_block> final_astc_blocks;

	vector2D<candidate_encoding> coded_blocks;
};

static bool compress_strip_task(
	uint32_t strip_index, uint32_t total_strips, uint32_t strip_first_by, uint32_t strip_last_by,
	uint32_t num_blocks_x, uint32_t num_blocks_y, uint32_t total_blocks, uint32_t width, uint32_t height,
	astc_hdr_6x6_global_config &global_cfg, uastc_hdr_6x6_debug_state &debug_state, uastc_hdr_6x6_encode_state &enc_state)
{
	BASISU_NOTE_UNUSED(num_blocks_y);
	BASISU_NOTE_UNUSED(total_strips);
	
	vec3F prev_comp_pixels[BLOCK_H][BLOCK_W]; // [y][x]
	basisu::clear_obj(prev_comp_pixels);

	uint32_t prev_run_len = 0;

	bitwise_coder prev_encoding;
	candidate_encoding prev_candidate_encoding; // the previous candidate written, which may have been a run extension
	candidate_encoding prev_non_run_candidate_encoding; // the previous *non-run* candidate written

	bitwise_coder& strip_coded_bits = enc_state.strip_bits[strip_index];

	const uint32_t CANDIDATES_TO_RESERVE = 1536;

	basisu::vector<candidate_encoding> candidates;
	candidates.reserve(CANDIDATES_TO_RESERVE);

	for (uint32_t by = strip_first_by; by <= strip_last_by; by++)
	{
		const bool has_upper_neighbor = by > strip_first_by;

		for (uint32_t bx = 0; bx < num_blocks_x; bx++)
		{
			//if ((bx == 1) && (by == 2))
			//	basisu::fmt_printf("!");

			for (uint32_t outer_pass = 0; outer_pass < 3; outer_pass++)
			{
				const bool has_left_neighbor = bx > 0;
				//const bool has_prev = has_left_neighbor || has_upper_neighbor;

				// Select either the original source image, or the Gaussian filtered version.
				// From here the encoder *must* use these 2 sources.
				const imagef& pass_src_img = (outer_pass == 2) ? enc_state.src_img_filtered2 :
					((outer_pass == 1) ? enc_state.src_img_filtered1 : enc_state.src_img);

				const imagef& pass_src_img_itp = (outer_pass == 2) ? enc_state.src_img_filtered2_itp :
					((outer_pass == 1) ? enc_state.src_img_filtered1_itp : enc_state.src_img_itp);

				// Extract source image block
				vec4F block_pixels[BLOCK_H][BLOCK_W]; // [y][x]
				pass_src_img.extract_block_clamped(&block_pixels[0][0], bx * BLOCK_W, by * BLOCK_H, BLOCK_W, BLOCK_H);

				vec4F block_pixels_itp[BLOCK_H][BLOCK_W]; // [y][x]
				pass_src_img_itp.extract_block_clamped(&block_pixels_itp[0][0], bx * BLOCK_W, by * BLOCK_H, BLOCK_W, BLOCK_H);

				half_vec3 half_pixels[BLOCK_H][BLOCK_W]; // [y][x] half-float values
				vec3F half_pixels_as_floats[BLOCK_H][BLOCK_W]; // [y][x] half float values, integer bits as floats
				vec4F block_pixels_q16[BLOCK_H][BLOCK_W]; // [y][x], q16 space for low-level ASTC encoding
				vec3F block_pixels_as_itp[BLOCK_H][BLOCK_W]; // [y][x] input converted to itp space, for faster error calculations

				bool is_grayscale = true;

				candidates.resize(0);

				float block_ly = BIG_FLOAT_VAL, block_hy = 0.0f, block_avg_y = 0.0f;

				for (uint32_t y = 0; y < BLOCK_H; y++)
				{
					for (uint32_t x = 0; x < BLOCK_W; x++)
					{
						vec3F rgb_input;

						for (uint32_t c = 0; c < 3; c++)
						{
							float v = block_pixels[y][x][c];

							rgb_input[c] = v;

							const basist::half_float h = basisu::fast_float_to_half_no_clamp_neg_nan_or_inf(v);
							assert(h == basist::float_to_half(v));

							half_pixels[y][x][c] = h;

							block_pixels_q16[y][x][c] = (float)half_to_qlog16(h);

							half_pixels_as_floats[y][x][c] = (float)h;

						} // c

						float py = rgb_input.dot(vec3F(REC_709_R, REC_709_G, REC_709_B));
						if (py < block_ly)
							block_ly = py;
						if (py > block_hy)
							block_hy = py;
						block_avg_y += py;

						//linear_rgb_to_itp(rgb_input, block_pixels_as_itp[y][x]);

						block_pixels_as_itp[y][x] = block_pixels_itp[y][x];

						block_pixels_q16[y][x][3] = 0.0f;

						if ((half_pixels[y][x][0] != half_pixels[y][x][1]) || (half_pixels[y][x][0] != half_pixels[y][x][2]))
							is_grayscale = false;

					} // x
				} // y

				block_avg_y *= (1.0f / (float)NUM_BLOCK_PIXELS);

				encode_astc_block_stats enc_block_stats;
				enc_block_stats.init(NUM_BLOCK_PIXELS, &block_pixels_q16[0][0]);

				vec4F x_filtered[6][6], y_filtered[6][6];

				filter_block(3, 6, (vec4F*)block_pixels, (vec4F*)x_filtered); // filter rows (horizontal)
				filter_block(6, 3, (vec4F*)block_pixels, (vec4F*)y_filtered); // filter cols (vertically)

				const float filtered_x_err = diff_blocks((vec4F*)block_pixels, (vec4F*)x_filtered);
				const float filtered_y_err = diff_blocks((vec4F*)block_pixels, (vec4F*)y_filtered);
				const bool filter_horizontally = filtered_x_err < filtered_y_err;

				//const float block_mag_gradient_mag = block_max_gradient_mag(bx, by);

				if (filter_horizontally)
					debug_state.m_total_filter_horizontal.fetch_add(1, std::memory_order_relaxed);

				vec3F lowpass_filtered[6][6];
				filter_block(3, 3, &half_pixels_as_floats[0][0], &lowpass_filtered[0][0]);
				float lowpass_std_dev = sub_and_compute_std_dev(&lowpass_filtered[0][0], &half_pixels_as_floats[0][0]);

				const bool very_detailed_block = lowpass_std_dev > 350.0f;
				const bool very_blurry_block = lowpass_std_dev < 30.0f;
				const bool super_blurry_block = lowpass_std_dev < 15.0f;

				basisu::stats<float> half_comp_stats[3];
				for (uint32_t c = 0; c < 3; c++)
					half_comp_stats[c].calc(NUM_BLOCK_PIXELS, &half_pixels_as_floats[0][0][c], 3);

				const float SINGLE_PART_HALF_THRESH = 256.0f;
				const float COMPLEX_HALF_THRESH = 1024.0f;
				// HACK HACK
				const float VERY_COMPLEX_HALF_THRESH = 1400.0f; // 1536.0f;

				const float max_std_dev = (float)maximum(half_comp_stats[0].m_std_dev, half_comp_stats[1].m_std_dev, half_comp_stats[2].m_std_dev);

				const bool very_simple_block = (max_std_dev < SINGLE_PART_HALF_THRESH);
				const bool complex_block = (max_std_dev > COMPLEX_HALF_THRESH);
				const bool very_complex_block = (max_std_dev > VERY_COMPLEX_HALF_THRESH);

				// Dynamically choose a comp_level for this block.
				astc_hdr_codec_base_options coptions(enc_state.master_coptions);
				uint32_t comp_level = global_cfg.m_master_comp_level;

				if (very_complex_block)
					comp_level = global_cfg.m_highest_comp_level;
				else if (complex_block)
					comp_level = (global_cfg.m_master_comp_level + global_cfg.m_highest_comp_level + 1) / 2;

				debug_state.m_comp_level_hist[comp_level].fetch_add(1, std::memory_order_relaxed);

				bool any_2subset_enabled = false, any_2subset_mode11_enabled = false, any_2subset_mode7_enabled = false, any_3subset_enabled = false;
				BASISU_NOTE_UNUSED(any_2subset_mode11_enabled);

				for (uint32_t i = 0; i < (uint32_t)block_mode::cBMTotalModes; i++)
				{
					if (comp_level == 0)
					{
						if ((g_block_mode_descs[i].m_flags & BASIST_HDR_6X6_LEVEL0) == 0)
							continue;
					}
					else if (comp_level == 1)
					{
						if ((g_block_mode_descs[i].m_flags & BASIST_HDR_6X6_LEVEL1) == 0)
							continue;
					}
					else if (comp_level == 2)
					{
						if ((g_block_mode_descs[i].m_flags & BASIST_HDR_6X6_LEVEL2) == 0)
							continue;
					}

					if (g_block_mode_descs[i].m_num_partitions == 2)
					{
						any_2subset_enabled = true;

						if (g_block_mode_descs[i].m_cem == 7)
						{
							any_2subset_mode7_enabled = true;
						}
						else
						{
							assert(g_block_mode_descs[i].m_cem == 11);
							any_2subset_mode11_enabled = true;
						}
					}
					else if (g_block_mode_descs[i].m_num_partitions == 3)
						any_3subset_enabled = true;
				}

				coptions.m_mode7_full_s_optimization = (comp_level >= 2);

				const bool uber_mode_flag = (comp_level >= 3);
				coptions.m_allow_uber_mode = uber_mode_flag;

				coptions.m_ultra_quant = (comp_level >= 4);

				coptions.m_take_first_non_clamping_mode11_submode = (comp_level <= 2);
				coptions.m_take_first_non_clamping_mode7_submode = (comp_level <= 2);

				coptions.m_disable_weight_plane_optimization = (comp_level >= 2);

				// -------------------

				uint32_t total_used_block_chans = 0;
				for (uint32_t i = 0; i < 3; i++)
					total_used_block_chans += (half_comp_stats[i].m_range > 0.0f);

				const bool is_solid_block = (total_used_block_chans == 0);

				basisu::comparative_stats<float> half_cross_chan_stats[3];

				// R vs. G
				half_cross_chan_stats[0].calc_pearson(NUM_BLOCK_PIXELS,
					&half_pixels_as_floats[0][0][0], &half_pixels_as_floats[0][0][1],
					3, 3,
					&half_comp_stats[0], &half_comp_stats[1]);

				// R vs. B
				half_cross_chan_stats[1].calc_pearson(NUM_BLOCK_PIXELS,
					&half_pixels_as_floats[0][0][0], &half_pixels_as_floats[0][0][2],
					3, 3,
					&half_comp_stats[0], &half_comp_stats[2]);

				// G vs. B
				half_cross_chan_stats[2].calc_pearson(NUM_BLOCK_PIXELS,
					&half_pixels_as_floats[0][0][1], &half_pixels_as_floats[0][0][2],
					3, 3,
					&half_comp_stats[1], &half_comp_stats[2]);

				const float rg_corr = fabsf((float)half_cross_chan_stats[0].m_pearson);
				const float rb_corr = fabsf((float)half_cross_chan_stats[1].m_pearson);
				const float gb_corr = fabsf((float)half_cross_chan_stats[2].m_pearson);

				float min_corr = BIG_FLOAT_VAL, max_corr = -BIG_FLOAT_VAL;
				for (uint32_t i = 0; i < 3; i++)
				{
#if 0
					// 9/5/2025, wrong metric, we're iterating channels pairs here, not individual channels. 
					// On 3 active channel blocks this causes no difference.
					if (half_comp_stats[i].m_range > 0.0f) 
#else
					static const uint8_t s_chan_pairs[3][2] = { {0, 1}, {0, 2}, {1, 2} };
					
					const uint32_t chanA = s_chan_pairs[i][0];
					const uint32_t chanB = s_chan_pairs[i][1];
					
					if ((half_comp_stats[chanA].m_range > 0.0f) && (half_comp_stats[chanB].m_range > 0.0f))
#endif
					{
						const float c = fabsf((float)half_cross_chan_stats[i].m_pearson);
						min_corr = minimum(min_corr, c);
						max_corr = maximum(max_corr, c);
					}
				}

				bool use_single_subset_mode7 = true;
				if (comp_level <= 1)
				{
					// TODO: could also compute angle between principle axis and the grayscale axis.
					// TODO: Transform grayscale axis by covar matrix, compute variance vs. total variance
					const float MODE7_MIN_CHAN_CORR = .5f;
					const float MODE7_PCA_ANGLE_THRESH = .9f;
					use_single_subset_mode7 = is_grayscale || is_solid_block || ((total_used_block_chans == 1) || (min_corr >= MODE7_MIN_CHAN_CORR));

					if (use_single_subset_mode7)
					{
						float cos_ang = fabsf(enc_block_stats.m_axis_q16.dot(vec3F(0.5773502691f)));
						if (cos_ang < MODE7_PCA_ANGLE_THRESH)
							use_single_subset_mode7 = false;
					}
				}

				const float STRONG_CORR_THRESH = (comp_level <= 1) ? .5f : ((comp_level <= 3) ? .75f : .9f);

				int desired_dp_chan = -1;
				if (total_used_block_chans <= 1)
				{
					// no need for dual plane (except possibly 2x2 weight grids for RDO)
				}
				else
				{
					if (min_corr >= STRONG_CORR_THRESH)
					{
						// all channel pairs strongly correlated, no need for dual plane
						debug_state.m_dp_stats[0].fetch_add(1, std::memory_order_relaxed);
					}
					else
					{
						if (total_used_block_chans == 2)
						{
							if (half_comp_stats[0].m_range == 0.0f)
							{
								// r unused, check for strong gb correlation
								if (gb_corr < STRONG_CORR_THRESH)
									desired_dp_chan = 1;
							}
							else if (half_comp_stats[1].m_range == 0.0f)
							{
								// g unused, check for strong rb correlation
								if (rb_corr < STRONG_CORR_THRESH)
									desired_dp_chan = 0;
							}
							else
							{
								// b unused, check for strong rg correlation
								if (rg_corr < STRONG_CORR_THRESH)
									desired_dp_chan = 0;
							}
						}
						else
						{
							assert(total_used_block_chans == 3);

							// see if rg/rb is weakly correlated vs. gb
							if ((rg_corr < gb_corr) && (rb_corr < gb_corr))
								desired_dp_chan = 0;
							// see if gr/gb is weakly correlated vs. rb
							else if ((rg_corr < rb_corr) && (gb_corr < rb_corr))
								desired_dp_chan = 1;
							// assume b is weakest
							else
								desired_dp_chan = 2;
						}

						if (desired_dp_chan == -1)
							debug_state.m_dp_stats[1].fetch_add(1, std::memory_order_relaxed);
						else
							debug_state.m_dp_stats[2 + desired_dp_chan].fetch_add(1, std::memory_order_relaxed);
					}
				}

				// 2x2 is special for RDO at higher lambdas - always pick a preferred channel.
				int desired_dp_chan_2x2 = 0;
				if (total_used_block_chans == 2)
				{
					if (half_comp_stats[0].m_range == 0.0f)
						desired_dp_chan_2x2 = 1;
				}
				else if (total_used_block_chans == 3)
				{
					// see if rg/rb is weakly correlated vs. gb
					if ((rg_corr < gb_corr) && (rb_corr < gb_corr))
						desired_dp_chan_2x2 = 0;
					// see if gr/gb is weakly correlated vs. rb
					else if ((rg_corr < rb_corr) && (gb_corr < rb_corr))
						desired_dp_chan_2x2 = 1;
					// assume b is weakest
					else
						desired_dp_chan_2x2 = 2;
				}

				// Gather all candidate encodings
				bool status = false;

				// ---- Run candidate
				if ((global_cfg.m_use_runs) && (has_left_neighbor || has_upper_neighbor))
				{
					candidate_encoding candidate;
					candidate.m_coder.reserve(24);

					candidate.m_encoding_type = encoding_type::cRun;

					candidate.m_decomp_log_blk = prev_non_run_candidate_encoding.m_decomp_log_blk;
					candidate.m_coded_log_blk = prev_non_run_candidate_encoding.m_coded_log_blk;

					memcpy(candidate.m_comp_pixels, prev_comp_pixels, sizeof(prev_comp_pixels));

					if (!prev_run_len)
					{
						candidate.m_coder.put_bits(RUN_CODE, RUN_CODE_LEN);
						candidate.m_coder.put_vlc(0, 5);
					}
					else
					{
						// extend current run - compute the # of new bits needed for the extension.

						uint32_t prev_run_bits = prev_encoding.get_total_bits_u32();
						assert(prev_run_bits > 0);

						// We're not actually going to code this, because the previously emitted run code will be extended.
						bitwise_coder temp_coder;
						temp_coder.put_bits(RUN_CODE, RUN_CODE_LEN);
						temp_coder.put_vlc((prev_run_len + 1) - 1, 5);

						uint32_t cur_run_bits = temp_coder.get_total_bits_u32();
						assert(cur_run_bits >= prev_run_bits);

						uint32_t total_new_bits = cur_run_bits - prev_run_bits;
						if (total_new_bits > 0)
							candidate.m_coder.put_bits(0, total_new_bits); // dummy bits
					}

					candidate.m_run_len = prev_run_len + 1;

					candidates.emplace_back(std::move(candidate));
				}

				// ---- Reuse candidate
				if ((!is_solid_block) && (global_cfg.m_lambda > 0.0f))
				{
					for (uint32_t reuse_delta_index = 0; reuse_delta_index < global_cfg.m_num_reuse_xy_deltas; reuse_delta_index++)
					{
						const int reuse_delta_x = g_reuse_xy_deltas[reuse_delta_index].m_x;
						const int reuse_delta_y = g_reuse_xy_deltas[reuse_delta_index].m_y;

						const int reuse_bx = bx + reuse_delta_x, reuse_by = by + reuse_delta_y;
						if ((reuse_bx < 0) || (reuse_bx >= (int)num_blocks_x))
							continue;
						if (reuse_by < (int)strip_first_by)
							break;

						const candidate_encoding& prev_candidate = enc_state.coded_blocks(reuse_bx, reuse_by);

						// TODO - support this.
						if (prev_candidate.m_encoding_type == encoding_type::cSolid)
							continue;
						assert((prev_candidate.m_encoding_type == encoding_type::cBlock) || (prev_candidate.m_encoding_type == encoding_type::cReuse));

						candidate_encoding candidate;
						candidate.m_coder.reserve(24);
						astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;
						astc_helpers::log_astc_block& decomp_log_blk = candidate.m_decomp_log_blk;

						const astc_helpers::log_astc_block& prev_coded_log_blk = prev_candidate.m_coded_log_blk;

						const uint32_t grid_x = prev_coded_log_blk.m_grid_width, grid_y = prev_coded_log_blk.m_grid_height;
						const bool dual_plane = prev_candidate.m_coded_log_blk.m_dual_plane;
						const uint32_t num_grid_samples = grid_x * grid_y;
						const uint32_t num_endpoint_vals = get_num_endpoint_vals(prev_coded_log_blk.m_color_endpoint_modes[0]);

						coded_log_blk = prev_candidate.m_coded_log_blk;
						decomp_log_blk = prev_candidate.m_decomp_log_blk;

						if (prev_coded_log_blk.m_num_partitions == 1)
						{
							// Now encode the block using the transcoded endpoints
							basist::half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];

							if (prev_coded_log_blk.m_color_endpoint_modes[0] == 7)
							{
								status = get_astc_hdr_mode_7_block_colors(coded_log_blk.m_endpoints, &decoded_half[0][0], nullptr,
									astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), coded_log_blk.m_weight_ise_range, coded_log_blk.m_endpoint_ise_range);
							}
							else
							{
								status = get_astc_hdr_mode_11_block_colors(coded_log_blk.m_endpoints, &decoded_half[0][0], nullptr,
									astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), coded_log_blk.m_weight_ise_range, coded_log_blk.m_endpoint_ise_range);
							}
							assert(status);

							uint8_t trial_weights0[BLOCK_W * BLOCK_H], trial_weights1[BLOCK_W * BLOCK_H];
							uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];

							if (dual_plane)
							{
								eval_selectors_dual_plane(prev_candidate.m_coded_log_blk.m_color_component_selector,
									BLOCK_W * BLOCK_H, trial_weights0, trial_weights1, (basist::half_float*)&half_pixels[0][0][0], astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

								downsample_ise_weights_dual_plane(
									coded_log_blk.m_weight_ise_range, coded_log_blk.m_weight_ise_range,
									BLOCK_W, BLOCK_H,
									grid_x, grid_y,
									trial_weights0, trial_weights1, coded_log_blk.m_weights);

								basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples * 2, coded_log_blk.m_weights, coded_log_blk.m_weight_ise_range, transcode_weights, decomp_log_blk.m_weight_ise_range);
							}
							else
							{
								eval_selectors(BLOCK_W * BLOCK_H, trial_weights0, coded_log_blk.m_weight_ise_range, (basist::half_float*)&half_pixels[0][0][0], astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

								downsample_ise_weights(
									coded_log_blk.m_weight_ise_range, coded_log_blk.m_weight_ise_range,
									BLOCK_W, BLOCK_H,
									grid_x, grid_y,
									trial_weights0, coded_log_blk.m_weights);

								basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples, coded_log_blk.m_weights, coded_log_blk.m_weight_ise_range, transcode_weights, decomp_log_blk.m_weight_ise_range);
							}

							// Create the block the decoder would transcode into.
							copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_log_blk);
						}
						else if (prev_coded_log_blk.m_num_partitions == 2)
						{
							assert(!dual_plane);

							const int unique_pat_index = g_part2_seed_to_unique_index[coded_log_blk.m_partition_id];
							assert((unique_pat_index >= 0) && (unique_pat_index < (int)NUM_UNIQUE_PARTITIONS2));

							const partition_pattern_vec& pat_vec = g_partitions2[unique_pat_index];

							vec4F part_pixels_q16[2][64];
							half_vec3 part_half_pixels[2][64];
							uint32_t part_total_pixels[2] = { 0 };

							for (uint32_t y = 0; y < BLOCK_H; y++)
							{
								for (uint32_t x = 0; x < BLOCK_W; x++)
								{
									const uint32_t part_index = pat_vec[x + y * 6];

									uint32_t l = part_total_pixels[part_index];

									part_pixels_q16[part_index][l] = block_pixels_q16[y][x];
									part_half_pixels[part_index][l] = half_pixels[y][x];

									part_total_pixels[part_index] = l + 1;
								} // x 
							} // y

							uint8_t blk_weights[2][BLOCK_W * BLOCK_H];

							for (uint32_t part_index = 0; part_index < 2; part_index++)
							{
								basist::half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];

								if (prev_coded_log_blk.m_color_endpoint_modes[0] == 7)
								{
									status = get_astc_hdr_mode_7_block_colors(coded_log_blk.m_endpoints + num_endpoint_vals * part_index, &decoded_half[0][0], nullptr,
										astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), coded_log_blk.m_weight_ise_range, coded_log_blk.m_endpoint_ise_range);
								}
								else
								{
									status = get_astc_hdr_mode_11_block_colors(coded_log_blk.m_endpoints + num_endpoint_vals * part_index, &decoded_half[0][0], nullptr,
										astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), coded_log_blk.m_weight_ise_range, coded_log_blk.m_endpoint_ise_range);
								}
								assert(status);

								eval_selectors(part_total_pixels[part_index], blk_weights[part_index], coded_log_blk.m_weight_ise_range,
									(basist::half_float*)&part_half_pixels[part_index][0][0], astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

							} // part_index

							uint8_t ise_weights[BLOCK_W * BLOCK_H];

							uint32_t src_pixel_index[2] = { 0, 0 };
							for (uint32_t y = 0; y < BLOCK_H; y++)
							{
								for (uint32_t x = 0; x < BLOCK_W; x++)
								{
									const uint32_t part_index = pat_vec[x + y * 6];

									ise_weights[x + y * BLOCK_W] = blk_weights[part_index][src_pixel_index[part_index]];
									src_pixel_index[part_index]++;
								} // x
							} // y

							downsample_ise_weights(
								coded_log_blk.m_weight_ise_range, coded_log_blk.m_weight_ise_range,
								BLOCK_W, BLOCK_H,
								grid_x, grid_y,
								ise_weights, coded_log_blk.m_weights);

							// Transcode these codable weights to ASTC weights.
							uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H];
							basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples, coded_log_blk.m_weights, coded_log_blk.m_weight_ise_range, transcode_weights, decomp_log_blk.m_weight_ise_range);

							// Create the block the decoder would transcode into.
							copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_log_blk);
						}
						else if (prev_coded_log_blk.m_num_partitions == 3)
						{
							assert(!dual_plane);

							const int unique_pat_index = g_part3_seed_to_unique_index[coded_log_blk.m_partition_id];
							assert((unique_pat_index >= 0) && (unique_pat_index < (int)NUM_UNIQUE_PARTITIONS3));

							const partition_pattern_vec& pat = g_partitions3[unique_pat_index];

							vec4F part_pixels_q16[3][64];
							half_vec3 part_half_pixels[3][64];
							uint32_t part_total_pixels[3] = { 0 };

							for (uint32_t y = 0; y < BLOCK_H; y++)
							{
								for (uint32_t x = 0; x < BLOCK_W; x++)
								{
									const uint32_t part_index = pat.m_parts[x + y * BLOCK_W];

									uint32_t l = part_total_pixels[part_index];

									part_pixels_q16[part_index][l] = block_pixels_q16[y][x];
									part_half_pixels[part_index][l] = half_pixels[y][x];

									part_total_pixels[part_index] = l + 1;
								} // x 
							} // y

							uint8_t blk_weights[3][BLOCK_W * BLOCK_H];

							for (uint32_t part_index = 0; part_index < 3; part_index++)
							{
								basist::half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];

								status = get_astc_hdr_mode_7_block_colors(coded_log_blk.m_endpoints + num_endpoint_vals * part_index, &decoded_half[0][0], nullptr,
									astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), coded_log_blk.m_weight_ise_range, coded_log_blk.m_endpoint_ise_range);
								assert(status);

								eval_selectors(part_total_pixels[part_index], blk_weights[part_index], coded_log_blk.m_weight_ise_range,
									(basist::half_float*)&part_half_pixels[part_index][0][0], astc_helpers::get_ise_levels(coded_log_blk.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

							} // part_index

							uint8_t ise_weights[BLOCK_W * BLOCK_H];

							uint32_t src_pixel_index[3] = { 0 };
							for (uint32_t y = 0; y < BLOCK_H; y++)
							{
								for (uint32_t x = 0; x < BLOCK_W; x++)
								{
									const uint32_t part_index = pat.m_parts[x + y * BLOCK_W];

									ise_weights[x + y * BLOCK_W] = blk_weights[part_index][src_pixel_index[part_index]];
									src_pixel_index[part_index]++;
								} // x
							} // y

							downsample_ise_weights(
								coded_log_blk.m_weight_ise_range, coded_log_blk.m_weight_ise_range,
								BLOCK_W, BLOCK_H,
								grid_x, grid_y,
								ise_weights, coded_log_blk.m_weights);

							// Transcode these codable weights to ASTC weights.
							uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H];
							basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples, coded_log_blk.m_weights, coded_log_blk.m_weight_ise_range, transcode_weights, decomp_log_blk.m_weight_ise_range);

							// Create the block the decoder would transcode into.
							copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_log_blk);
						}

						if (!validate_log_blk(decomp_log_blk))
						{
							fmt_error_printf("pack_astc_block() failed\n");
							return false;
						}

						status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_log_blk, &candidate.m_comp_pixels[0][0]);
						if (!status)
						{
							fmt_error_printf("decode_astc_block() failed\n");
							return false;
						}

						candidate.m_coder.put_bits(REUSE_CODE, REUSE_CODE_LEN);
						candidate.m_coder.put_bits(reuse_delta_index, REUSE_XY_DELTA_BITS);
						encode_values(candidate.m_coder, num_grid_samples * (dual_plane ? 2 : 1), coded_log_blk.m_weights, coded_log_blk.m_weight_ise_range);

						candidate.m_encoding_type = encoding_type::cReuse;
						candidate.m_block_mode = prev_candidate.m_block_mode;
						candidate.m_endpoint_mode = prev_candidate.m_endpoint_mode;
						candidate.m_reuse_delta_index = reuse_delta_index;

						candidates.emplace_back(std::move(candidate));

					} // reuse_delta_index
				}

				// ---- Solid candidate
				if (global_cfg.m_use_solid_blocks)
				{
					candidate_encoding candidate;
					candidate.m_coder.reserve(24);

					// solid
					candidate.m_encoding_type = encoding_type::cSolid;

					float r = 0.0f, g = 0.0f, b = 0.0f;
					const float LOG_BIAS = .125f;
					bool solid_block = true;
					for (uint32_t y = 0; y < BLOCK_H; y++)
					{
						for (uint32_t x = 0; x < BLOCK_W; x++)
						{
							if ((block_pixels[0][0][0] != block_pixels[y][x][0]) ||
								(block_pixels[0][0][1] != block_pixels[y][x][1]) ||
								(block_pixels[0][0][2] != block_pixels[y][x][2]))
							{
								solid_block = false;
							}

							r += log2f(block_pixels[y][x][0] + LOG_BIAS);
							g += log2f(block_pixels[y][x][1] + LOG_BIAS);
							b += log2f(block_pixels[y][x][2] + LOG_BIAS);
						}
					}

					if (solid_block)
					{
						r = block_pixels[0][0][0];
						g = block_pixels[0][0][1];
						b = block_pixels[0][0][2];
					}
					else
					{
						r = maximum<float>(0.0f, powf(2.0f, r * (1.0f / (float)NUM_BLOCK_PIXELS)) - LOG_BIAS);
						g = maximum<float>(0.0f, powf(2.0f, g * (1.0f / (float)NUM_BLOCK_PIXELS)) - LOG_BIAS);
						b = maximum<float>(0.0f, powf(2.0f, b * (1.0f / (float)NUM_BLOCK_PIXELS)) - LOG_BIAS);

						r = minimum<float>(r, basist::MAX_HALF_FLOAT);
						g = minimum<float>(g, basist::MAX_HALF_FLOAT);
						b = minimum<float>(b, basist::MAX_HALF_FLOAT);
					}

					basist::half_float rh = float_to_half_non_neg_no_nan_inf(r), gh = float_to_half_non_neg_no_nan_inf(g), bh = float_to_half_non_neg_no_nan_inf(b);

					candidate.m_solid_color[0] = rh;
					candidate.m_solid_color[1] = gh;
					candidate.m_solid_color[2] = bh;

					candidate.m_coder.put_bits(SOLID_CODE, SOLID_CODE_LEN);

					candidate.m_coder.put_bits(rh, 15);
					candidate.m_coder.put_bits(gh, 15);
					candidate.m_coder.put_bits(bh, 15);

					vec3F cp(basist::half_to_float(rh), basist::half_to_float(gh), basist::half_to_float(bh));

					for (uint32_t y = 0; y < BLOCK_H; y++)
						for (uint32_t x = 0; x < BLOCK_W; x++)
							candidate.m_comp_pixels[y][x] = cp;

					astc_helpers::log_astc_block& log_blk = candidate.m_coded_log_blk;

					log_blk.clear();
					log_blk.m_solid_color_flag_hdr = true;
					log_blk.m_solid_color[0] = rh;
					log_blk.m_solid_color[1] = gh;
					log_blk.m_solid_color[2] = bh;
					log_blk.m_solid_color[3] = basist::float_to_half(1.0f);

					candidate.m_decomp_log_blk = log_blk;

					candidates.emplace_back(std::move(candidate));
				}

				if ((!is_solid_block) || (!global_cfg.m_use_solid_blocks))
				{
					static uint8_t s_parts2_normal[5] = { 0, 2, 4, 6, 8 };
					static uint8_t s_parts3_normal[5] = { 0, 0, 4, 6, 8 };

					static uint8_t s_parts2_complex[5] = { 0, 4, 8, 10, 16 };
					static uint8_t s_parts3_complex[5] = { 0, 0, 8, 10, 16 };

					static uint8_t s_parts2_very_complex[5] = { 0, 8, 12, 14, 20 };
					static uint8_t s_parts3_very_complex[5] = { 0, 0, 12, 14, 20 };

					uint32_t total_parts2 = 0, total_parts3 = 0;

					assert(comp_level < 5);
					if ((very_simple_block) && (comp_level <= 3))
					{
						// Block's std dev is so low that 2-3 subsets are unlikely to help much
						total_parts2 = 0;
						total_parts3 = 0;

						debug_state.m_total_part2_stats[0].fetch_add(1, std::memory_order_relaxed);
					}
					else if (very_complex_block)
					{
						total_parts2 = s_parts2_very_complex[comp_level];
						total_parts3 = s_parts3_very_complex[comp_level];

						if (global_cfg.m_extra_patterns_flag)
						{
							total_parts2 += (comp_level == 4) ? 30 : 20;
							total_parts3 += (comp_level == 4) ? 30 : 20;
						}

						debug_state.m_total_part2_stats[2].fetch_add(1, std::memory_order_relaxed);
					}
					else if (complex_block)
					{
						total_parts2 = s_parts2_complex[comp_level];
						total_parts3 = s_parts3_complex[comp_level];

						if (global_cfg.m_extra_patterns_flag)
						{
							total_parts2 += (comp_level == 4) ? 15 : 10;
							total_parts3 += (comp_level == 4) ? 15 : 10;
						}

						debug_state.m_total_part2_stats[3].fetch_add(1, std::memory_order_relaxed);
					}
					else
					{
						// moderate complexity - use defaults
						total_parts2 = s_parts2_normal[comp_level];
						total_parts3 = s_parts3_normal[comp_level];

						if (global_cfg.m_extra_patterns_flag)
						{
							total_parts2 += 5;
							total_parts3 += 5;
						}

						debug_state.m_total_part2_stats[1].fetch_add(1, std::memory_order_relaxed);
					}

					if (!any_2subset_enabled)
						total_parts2 = 0;

					if (!any_3subset_enabled)
						total_parts3 = 0;

					int best_parts2_mode11[NUM_UNIQUE_PARTITIONS2], best_parts2_mode7[NUM_UNIQUE_PARTITIONS2];
					bool has_estimated_parts2 = false;

					if (total_parts2)
					{
						if (global_cfg.m_brute_force_partition_matching)
						{
							int candidate_pats2[NUM_UNIQUE_PARTITIONS2];
							for (uint32_t i = 0; i < NUM_UNIQUE_PARTITIONS2; i++)
								candidate_pats2[i] = i;

							if (any_2subset_enabled)
							{
								estimate_partitions_mode7_and_11(
									2,
									NUM_UNIQUE_PARTITIONS2, g_partitions2,
									NUM_UNIQUE_PARTITIONS2, (uint32_t*)candidate_pats2,
									&half_pixels_as_floats[0][0],
									coptions,
									total_parts2, best_parts2_mode11, best_parts2_mode7);
							}

							has_estimated_parts2 = true;
						}
						else
						{
							if (comp_level >= 1)
							{
								const uint32_t MAX_CANDIDATES2 = 48;
								int candidate_pats2[MAX_CANDIDATES2 * 2];

								uint32_t num_candidate_pats2 = maximum((total_parts2 * 3) / 2, very_complex_block ? MAX_CANDIDATES2 : (MAX_CANDIDATES2 / 2));
								num_candidate_pats2 = minimum<uint32_t>(num_candidate_pats2, (uint32_t)std::size(candidate_pats2));

								has_estimated_parts2 = estimate_partition2_6x6((basist::half_float(*)[3])half_pixels, candidate_pats2, num_candidate_pats2);

								if (has_estimated_parts2)
								{
									estimate_partitions_mode7_and_11(
										2,
										NUM_UNIQUE_PARTITIONS2, g_partitions2,
										num_candidate_pats2, (uint32_t*)candidate_pats2,
										&half_pixels_as_floats[0][0],
										coptions,
										total_parts2, best_parts2_mode11, best_parts2_mode7);
								}
							}
							else
							{
								has_estimated_parts2 = estimate_partition2_6x6((basist::half_float(*)[3])half_pixels, best_parts2_mode11, total_parts2);

								if ((has_estimated_parts2) && (any_2subset_mode7_enabled))
									memcpy(best_parts2_mode7, best_parts2_mode11, total_parts2 * sizeof(best_parts2_mode7[0]));
							}
						}
					}

					int best_parts3[NUM_UNIQUE_PARTITIONS3];
					bool has_estimated_parts3 = false;

					if (total_parts3)
					{
#if 0
						has_estimated_parts3 = estimate_partition3_6x6((basist::half_float(*)[3])half_pixels, best_parts3, total_parts3);
#elif 1
						if (global_cfg.m_brute_force_partition_matching)
						{
							int candidate_pats3[NUM_UNIQUE_PARTITIONS3];
							for (uint32_t i = 0; i < NUM_UNIQUE_PARTITIONS3; i++)
								candidate_pats3[i] = i;

							estimate_partitions_mode7(
								3,
								NUM_UNIQUE_PARTITIONS3, g_partitions3,
								NUM_UNIQUE_PARTITIONS3, (uint32_t*)candidate_pats3,
								&half_pixels_as_floats[0][0],
								coptions,
								total_parts3, (uint32_t*)best_parts3);

							has_estimated_parts3 = true;
						}
						else
						{
							const uint32_t MAX_CANDIDATES3 = 48;
							int candidate_pats3[MAX_CANDIDATES3 * 2];

							uint32_t num_candidate_pats3 = maximum((total_parts3 * 3) / 2, very_complex_block ? MAX_CANDIDATES3 : (MAX_CANDIDATES3 / 2));
							num_candidate_pats3 = minimum<uint32_t>(num_candidate_pats3, (uint32_t)std::size(candidate_pats3));

							has_estimated_parts3 = estimate_partition3_6x6((basist::half_float(*)[3])half_pixels, candidate_pats3, num_candidate_pats3);

							if (has_estimated_parts3)
							{
								estimate_partitions_mode7(
									3,
									NUM_UNIQUE_PARTITIONS3, g_partitions3,
									num_candidate_pats3, (uint32_t*)candidate_pats3,
									&half_pixels_as_floats[0][0],
									coptions,
									total_parts3, (uint32_t*)best_parts3);
							}
						}
#endif
					}

					const opt_mode_t mode11_opt_mode = complex_block ? cWeightedLeastSquares : cOrdinaryLeastSquares;

					// ---- Encoded block candidate
					for (uint32_t block_mode_iter = 0; block_mode_iter < (uint32_t)block_mode::cBMTotalModes; block_mode_iter++)
					{
						const block_mode bm = (block_mode)block_mode_iter;

						if (comp_level == 0)
						{
							if ((g_block_mode_descs[block_mode_iter].m_flags & BASIST_HDR_6X6_LEVEL0) == 0)
								continue;
						}
						else if (comp_level == 1)
						{
							if ((g_block_mode_descs[block_mode_iter].m_flags & BASIST_HDR_6X6_LEVEL1) == 0)
								continue;
						}
						else if (comp_level == 2)
						{
							if ((g_block_mode_descs[block_mode_iter].m_flags & BASIST_HDR_6X6_LEVEL2) == 0)
								continue;
						}

						if (global_cfg.m_block_stat_optimizations_flag)
						{
							if ((comp_level <= 3) && (g_block_mode_descs[block_mode_iter].m_dp))
							{
								if ((global_cfg.m_lambda > 0.0f) && (!complex_block) && (g_block_mode_descs[block_mode_iter].m_grid_x == 2) && (g_block_mode_descs[block_mode_iter].m_grid_y == 2))
								{
									if (g_block_mode_descs[block_mode_iter].m_dp_channel != desired_dp_chan_2x2)
										continue;
								}
								else
								{
									if (g_block_mode_descs[block_mode_iter].m_dp_channel != desired_dp_chan)
										continue;
								}
							}

							if (comp_level <= 3)
							{
								const uint32_t grid_x = g_block_mode_descs[block_mode_iter].m_grid_x;
								const uint32_t grid_y = g_block_mode_descs[block_mode_iter].m_grid_y;

								if (!g_block_mode_descs[block_mode_iter].m_dp)
								{
									// Minor gain (.5-1% less canidates)
									if (very_detailed_block)
									{
										if (grid_x * grid_y <= 12)
										{
											debug_state.m_detail_stats[0].fetch_add(1, std::memory_order_relaxed);
											continue;
										}
									}

									// Major gains (10-25% less candidates)
									if (very_blurry_block)
									{
										if ((grid_x > 4) || (grid_y > 4) || (g_block_mode_descs[block_mode_iter].m_num_partitions > 1))
										{
											debug_state.m_detail_stats[1].fetch_add(1, std::memory_order_relaxed);
											continue;
										}
									}
									if (super_blurry_block)
									{
										if ((grid_x > 3) || (grid_y > 3) || (g_block_mode_descs[block_mode_iter].m_num_partitions > 1))
										{
											debug_state.m_detail_stats[2].fetch_add(1, std::memory_order_relaxed);
											continue;
										}
									}
								}

								if (grid_x != grid_y)
								{
									if (grid_x < grid_y)
									{
										if (!filter_horizontally)
										{
											debug_state.m_detail_stats[3].fetch_add(1, std::memory_order_relaxed);
											continue;
										}
									}
									else
									{
										if (filter_horizontally)
										{
											debug_state.m_detail_stats[4].fetch_add(1, std::memory_order_relaxed);
											continue;
										}
									}
								}
							}

							if (global_cfg.m_lambda == 0.0f)
							{
								// Rarely useful if lambda=0
								if ((g_block_mode_descs[block_mode_iter].m_grid_x == 2) && (g_block_mode_descs[block_mode_iter].m_grid_y == 2))
									continue;
							}
						} // block_stat_optimizations_flag

						if ((!use_single_subset_mode7) &&
							(g_block_mode_descs[block_mode_iter].m_cem == 7) &&
							(g_block_mode_descs[block_mode_iter].m_num_partitions == 1))
						{
							debug_state.m_total_mode7_skips.fetch_add(1, std::memory_order_relaxed);
							continue;
						}

						for (uint32_t endpoint_mode_iter = 0; endpoint_mode_iter < (uint32_t)endpoint_mode::cTotal; endpoint_mode_iter++)
						{
							if (global_cfg.m_lambda == 0.0f)
							{
								// No use trying anything else
								if (endpoint_mode_iter != (uint32_t)endpoint_mode::cRaw)
									continue;
							}

							if (global_cfg.m_disable_delta_endpoint_usage)
							{
								if ((endpoint_mode_iter == (uint32_t)endpoint_mode::cUseUpperDelta) || (endpoint_mode_iter == (uint32_t)endpoint_mode::cUseLeftDelta))
									continue;
							}

							if (!global_cfg.m_favor_higher_compression)
							{
								if (comp_level == 0)
								{
									if (endpoint_mode_iter == (uint32_t)endpoint_mode::cUseUpperDelta)
										continue;
								}

								if (comp_level <= 1)
								{
									if ((endpoint_mode_iter == (uint32_t)endpoint_mode::cUseLeft) || (endpoint_mode_iter == (uint32_t)endpoint_mode::cUseUpper))
										continue;
								}
							}

							const endpoint_mode em = (endpoint_mode)endpoint_mode_iter;

							switch (em)
							{
							case endpoint_mode::cUseLeft:
							case endpoint_mode::cUseUpper:
							{
								const block_mode_desc& local_md = g_block_mode_descs[block_mode_iter];
								const uint32_t cem = local_md.m_cem;

								if (local_md.m_num_partitions > 1)
									break;

								if ((em == endpoint_mode::cUseLeft) && (!has_left_neighbor))
									break;
								else if ((em == endpoint_mode::cUseUpper) && (!has_upper_neighbor))
									break;

								candidate_encoding candidate;
								candidate.m_coder.reserve(24);
								astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;

								int nx = bx, ny = by;
								if (em == endpoint_mode::cUseLeft)
									nx--;
								else
									ny--;

								const candidate_encoding& neighbor_blk = enc_state.coded_blocks(nx, ny);
								if (neighbor_blk.m_encoding_type == encoding_type::cSolid)
									break;
								assert((neighbor_blk.m_encoding_type == encoding_type::cBlock) || (neighbor_blk.m_encoding_type == encoding_type::cReuse));

								const block_mode_desc& neighbor_md = g_block_mode_descs[(uint32_t)neighbor_blk.m_block_mode];

								if (neighbor_md.m_cem != cem)
									break;

								assert(neighbor_blk.m_coded_log_blk.m_color_endpoint_modes[0] == cem);

								const uint32_t grid_x = local_md.m_grid_x, grid_y = local_md.m_grid_y;
								const bool dual_plane = local_md.m_dp;
								const uint32_t num_grid_samples = grid_x * grid_y;
								const uint32_t num_endpoint_vals = get_num_endpoint_vals(local_md.m_cem);

								coded_log_blk.m_grid_width = (uint8_t)grid_x;
								coded_log_blk.m_grid_height = (uint8_t)grid_y;
								coded_log_blk.m_dual_plane = (uint8_t)dual_plane;
								coded_log_blk.m_color_component_selector = (uint8_t)local_md.m_dp_channel;
								coded_log_blk.m_num_partitions = 1;
								coded_log_blk.m_color_endpoint_modes[0] = (uint8_t)neighbor_md.m_cem;
								coded_log_blk.m_weight_ise_range = (uint8_t)local_md.m_weight_ise_range;

								// We're not explictly writing any endpoints, just reusing existing ones. So copy the neighbor's endpoints unchanged (so no loss).
								coded_log_blk.m_endpoint_ise_range = neighbor_blk.m_coded_log_blk.m_endpoint_ise_range;
								memcpy(coded_log_blk.m_endpoints, neighbor_blk.m_coded_log_blk.m_endpoints, num_endpoint_vals);

								uint8_t transcode_endpoints[basist::NUM_MODE11_ENDPOINTS];

								// Requantize the neighbor's endpoints to whatever we'll have to transcode into to make a valid ASTC encoding.
								basist::astc_6x6_hdr::requantize_ise_endpoints(neighbor_md.m_cem,
									neighbor_blk.m_coded_log_blk.m_endpoint_ise_range, neighbor_blk.m_coded_log_blk.m_endpoints,
									local_md.m_transcode_endpoint_ise_range, transcode_endpoints);

								// Now encode the block using the transcoded endpoints
								basist::half_float decoded_half[MAX_SUPPORTED_WEIGHT_LEVELS][3];

								if (cem == 7)
								{
									status = get_astc_hdr_mode_7_block_colors(transcode_endpoints, &decoded_half[0][0], nullptr,
										astc_helpers::get_ise_levels(local_md.m_weight_ise_range), local_md.m_weight_ise_range, local_md.m_transcode_endpoint_ise_range);
								}
								else
								{
									status = get_astc_hdr_mode_11_block_colors(transcode_endpoints, &decoded_half[0][0], nullptr,
										astc_helpers::get_ise_levels(local_md.m_weight_ise_range), local_md.m_weight_ise_range, local_md.m_transcode_endpoint_ise_range);
								}
								if (!status)
									break;

								uint8_t trial_weights0[BLOCK_W * BLOCK_H], trial_weights1[BLOCK_W * BLOCK_H];
								if (dual_plane)
								{
									eval_selectors_dual_plane(local_md.m_dp_channel, BLOCK_W * BLOCK_H, trial_weights0, trial_weights1, (basist::half_float*)&half_pixels[0][0][0], astc_helpers::get_ise_levels(local_md.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

									downsample_ise_weights_dual_plane(
										local_md.m_weight_ise_range, local_md.m_weight_ise_range,
										BLOCK_W, BLOCK_H,
										grid_x, grid_y,
										trial_weights0, trial_weights1, coded_log_blk.m_weights);
								}
								else
								{
									eval_selectors(BLOCK_W * BLOCK_H, trial_weights0, local_md.m_weight_ise_range, (basist::half_float*)&half_pixels[0][0][0], astc_helpers::get_ise_levels(local_md.m_weight_ise_range), &decoded_half[0][0], coptions, UINT32_MAX);

									downsample_ise_weights(
										local_md.m_weight_ise_range, local_md.m_weight_ise_range,
										BLOCK_W, BLOCK_H,
										grid_x, grid_y,
										trial_weights0, coded_log_blk.m_weights);
								}

								// Transcode these codable weights to ASTC weights.
								uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];
								basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples * (dual_plane ? 2 : 1), coded_log_blk.m_weights, local_md.m_weight_ise_range, transcode_weights, local_md.m_transcode_weight_ise_range);

								// Create the block the decoder would transcode into.
								astc_helpers::log_astc_block& decomp_blk = candidate.m_decomp_log_blk;
								decomp_blk.clear();

								decomp_blk.m_color_endpoint_modes[0] = (uint8_t)local_md.m_cem;
								decomp_blk.m_dual_plane = local_md.m_dp;
								decomp_blk.m_color_component_selector = (uint8_t)local_md.m_dp_channel;
								decomp_blk.m_num_partitions = 1;
								decomp_blk.m_endpoint_ise_range = (uint8_t)local_md.m_transcode_endpoint_ise_range;
								decomp_blk.m_weight_ise_range = (uint8_t)local_md.m_transcode_weight_ise_range;

								memcpy(decomp_blk.m_endpoints, transcode_endpoints, num_endpoint_vals);

								copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_blk);

								if (!validate_log_blk(decomp_blk))
								{
									fmt_error_printf("pack_astc_block() failed\n");
									return false;
								}

								status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_blk, &candidate.m_comp_pixels[0][0]);
								if (!status)
								{
									fmt_error_printf("decode_astc_block() failed\n");
									return false;
								}

								candidate.m_coder.put_bits(BLOCK_CODE, BLOCK_CODE_LEN);
								code_block(candidate.m_coder, candidate.m_coded_log_blk, (block_mode)block_mode_iter, em, nullptr);

								candidate.m_encoding_type = encoding_type::cBlock;
								candidate.m_endpoint_mode = em;
								candidate.m_block_mode = bm;

								candidates.emplace_back(std::move(candidate));

								break;
							}
							case endpoint_mode::cUseLeftDelta:
							case endpoint_mode::cUseUpperDelta:
							{
								const block_mode_desc& local_md = g_block_mode_descs[block_mode_iter];
								const uint32_t cem = local_md.m_cem;

								if (local_md.m_num_partitions > 1)
									break;

								if ((em == endpoint_mode::cUseLeftDelta) && (!has_left_neighbor))
									break;
								else if ((em == endpoint_mode::cUseUpperDelta) && (!has_upper_neighbor))
									break;

								candidate_encoding candidate;
								candidate.m_coder.reserve(24);
								astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;

								int nx = bx, ny = by;
								if (em == endpoint_mode::cUseLeftDelta)
									nx--;
								else
									ny--;

								const candidate_encoding& neighbor_blk = enc_state.coded_blocks(nx, ny);
								if (neighbor_blk.m_encoding_type == encoding_type::cSolid)
									break;
								assert((neighbor_blk.m_encoding_type == encoding_type::cBlock) || (neighbor_blk.m_encoding_type == encoding_type::cReuse));

								const block_mode_desc& neighbor_md = g_block_mode_descs[(uint32_t)neighbor_blk.m_block_mode];

								if (neighbor_md.m_cem != cem)
									break;

								assert(neighbor_md.m_cem == local_md.m_cem);

								const uint32_t grid_x = local_md.m_grid_x, grid_y = local_md.m_grid_y;
								const bool dual_plane = local_md.m_dp;
								const uint32_t num_grid_samples = grid_x * grid_y;
								const uint32_t num_endpoint_vals = get_num_endpoint_vals(local_md.m_cem);

								// Dequantize neighbor's endpoints to ISE 20
								uint8_t neighbor_endpoints_ise20[basist::NUM_MODE11_ENDPOINTS];
								basist::astc_6x6_hdr::requantize_ise_endpoints(neighbor_md.m_cem,
									neighbor_blk.m_coded_log_blk.m_endpoint_ise_range, neighbor_blk.m_coded_log_blk.m_endpoints,
									astc_helpers::BISE_256_LEVELS, neighbor_endpoints_ise20);

								// Requantize neighbor's endpoints to our local desired coding ISE range
								uint8_t neighbor_endpoints_coding_ise_local[basist::NUM_MODE11_ENDPOINTS];
								basist::astc_6x6_hdr::requantize_ise_endpoints(neighbor_md.m_cem, astc_helpers::BISE_256_LEVELS, neighbor_endpoints_ise20, local_md.m_endpoint_ise_range, neighbor_endpoints_coding_ise_local);

								uint8_t blk_endpoints[basist::NUM_MODE11_ENDPOINTS];
								uint8_t blk_weights0[NUM_BLOCK_PIXELS], blk_weights1[NUM_BLOCK_PIXELS];

								// Now try to encode the current block using the neighbor's endpoints submode.
								double err = 0.0f;
								uint32_t best_submode = 0;

								if (cem == 7)
								{
									int maj_index, submode_index;
									decode_cem_7_config(neighbor_endpoints_ise20, submode_index, maj_index);

									int first_submode = submode_index, last_submode = submode_index;

									err = encode_astc_hdr_block_mode_7(
										NUM_BLOCK_PIXELS,
										(basist::half_float(*)[3])half_pixels, (vec4F*)block_pixels_q16,
										local_md.m_weight_ise_range,
										best_submode,
										BIG_FLOAT_VAL,
										blk_endpoints, blk_weights0,
										coptions,
										local_md.m_endpoint_ise_range,
										first_submode, last_submode,
										&enc_block_stats);
								}
								else
								{
									int maj_index, submode_index;
									decode_cem_11_config(neighbor_endpoints_ise20, submode_index, maj_index);

									int first_submode = -1, last_submode = -1;
									if (maj_index == 3)
									{
										// direct
									}
									else
									{
										first_submode = submode_index;
										last_submode = submode_index;
									}

									if (dual_plane)
									{
										err = encode_astc_hdr_block_mode_11_dual_plane(
											NUM_BLOCK_PIXELS,
											(basist::half_float(*)[3])half_pixels, (vec4F*)block_pixels_q16,
											local_md.m_dp_channel,
											local_md.m_weight_ise_range,
											best_submode,
											BIG_FLOAT_VAL,
											blk_endpoints, blk_weights0, blk_weights1,
											coptions,
											false,
											local_md.m_endpoint_ise_range,
											false, //uber_mode_flag,
											false,
											first_submode, last_submode, true);
									}
									else
									{
										err = encode_astc_hdr_block_mode_11(
											NUM_BLOCK_PIXELS,
											(basist::half_float(*)[3])half_pixels, (vec4F*)block_pixels_q16,
											local_md.m_weight_ise_range,
											best_submode,
											BIG_FLOAT_VAL,
											blk_endpoints, blk_weights0,
											coptions,
											false,
											local_md.m_endpoint_ise_range,
											false, //uber_mode_flag,
											false,
											first_submode, last_submode, true,
											mode11_opt_mode,
											&enc_block_stats);
									}
								}

								if (err == BIG_FLOAT_VAL)
									break;

								uint8_t endpoint_deltas[basist::NUM_MODE11_ENDPOINTS];

								// TODO: For now, just try 5 bits for each endpoint. Can tune later.
								// This isn't right, it's computing the deltas in ISE space.
								//const uint32_t NUM_ENDPOINT_DELTA_BITS = 5;
								const int total_endpoint_delta_vals = 1 << NUM_ENDPOINT_DELTA_BITS;
								const int low_delta_limit = -(total_endpoint_delta_vals / 2), high_delta_limit = (total_endpoint_delta_vals / 2) - 1;

								const auto& ise_to_rank = astc_helpers::g_dequant_tables.get_endpoint_tab(local_md.m_endpoint_ise_range).m_ISE_to_rank;

								bool all_deltas_in_limits = true;
								for (uint32_t i = 0; i < num_endpoint_vals; i++)
								{
									int endpoint_delta = (int)ise_to_rank[blk_endpoints[i]] - (int)ise_to_rank[neighbor_endpoints_coding_ise_local[i]];

									if ((endpoint_delta < low_delta_limit) || (endpoint_delta > high_delta_limit))
										all_deltas_in_limits = false;

									endpoint_deltas[i] = (uint8_t)(endpoint_delta + -low_delta_limit);
								}

								if (all_deltas_in_limits)
								{
									coded_log_blk.m_grid_width = (uint8_t)grid_x;
									coded_log_blk.m_grid_height = (uint8_t)grid_y;
									coded_log_blk.m_dual_plane = (uint8_t)dual_plane;
									coded_log_blk.m_color_component_selector = (uint8_t)local_md.m_dp_channel;
									coded_log_blk.m_num_partitions = 1;
									coded_log_blk.m_color_endpoint_modes[0] = (uint8_t)local_md.m_cem;
									coded_log_blk.m_weight_ise_range = (uint8_t)local_md.m_weight_ise_range;
									coded_log_blk.m_endpoint_ise_range = (uint8_t)local_md.m_endpoint_ise_range;

									memcpy(coded_log_blk.m_endpoints, blk_endpoints, num_endpoint_vals);

									uint8_t transcode_endpoints[basist::NUM_MODE11_ENDPOINTS];
									uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];

									basist::astc_6x6_hdr::requantize_ise_endpoints(local_md.m_cem, local_md.m_endpoint_ise_range, blk_endpoints, local_md.m_transcode_endpoint_ise_range, transcode_endpoints);

									if (dual_plane)
									{
										downsample_ise_weights_dual_plane(
											local_md.m_weight_ise_range, local_md.m_weight_ise_range,
											BLOCK_W, BLOCK_H,
											grid_x, grid_y,
											blk_weights0, blk_weights1,
											coded_log_blk.m_weights);
									}
									else
									{
										downsample_ise_weights(
											local_md.m_weight_ise_range, local_md.m_weight_ise_range,
											BLOCK_W, BLOCK_H,
											grid_x, grid_y,
											blk_weights0, coded_log_blk.m_weights);
									}

									basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples * (dual_plane ? 2 : 1), coded_log_blk.m_weights, local_md.m_weight_ise_range, transcode_weights, local_md.m_transcode_weight_ise_range);

									// Create the block the decoder would transcode into.

									astc_helpers::log_astc_block& decomp_blk = candidate.m_decomp_log_blk;
									decomp_blk.clear();

									decomp_blk.m_color_endpoint_modes[0] = (uint8_t)local_md.m_cem;
									decomp_blk.m_dual_plane = local_md.m_dp;
									decomp_blk.m_color_component_selector = (uint8_t)local_md.m_dp_channel;
									decomp_blk.m_num_partitions = 1;
									decomp_blk.m_endpoint_ise_range = (uint8_t)local_md.m_transcode_endpoint_ise_range;
									decomp_blk.m_weight_ise_range = (uint8_t)local_md.m_transcode_weight_ise_range;

									memcpy(decomp_blk.m_endpoints, transcode_endpoints, num_endpoint_vals);

									copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_blk);

									if (!validate_log_blk(decomp_blk))
									{
										fmt_error_printf("pack_astc_block() failed\n");
										return false;
									}

									status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_blk, &candidate.m_comp_pixels[0][0]);
									if (!status)
									{
										fmt_error_printf("decode_astc_block() failed\n");
										return false;
									}

									candidate.m_coder.put_bits(BLOCK_CODE, BLOCK_CODE_LEN);
									code_block(candidate.m_coder, candidate.m_coded_log_blk, bm, em, endpoint_deltas);

									candidate.m_encoding_type = encoding_type::cBlock;
									candidate.m_endpoint_mode = em;
									candidate.m_block_mode = bm;

									candidates.emplace_back(std::move(candidate));
								}

								break;
							}
							case endpoint_mode::cRaw:
							{
								//if (candidates.size() == 339)
								//	fmt_printf("!");

								const auto& mode_desc = g_block_mode_descs[(uint32_t)bm];
								const uint32_t cem = mode_desc.m_cem;
								//const uint32_t num_endpoint_vals = get_num_endpoint_vals(cem);
								const bool dual_plane = mode_desc.m_dp;

								if ((global_cfg.m_disable_twothree_subsets) && (mode_desc.m_num_partitions >= 2))
									break;

								if (mode_desc.m_num_partitions == 3)
								{
									assert(!dual_plane);

									if (!has_estimated_parts3)
										break;

									assert(mode_desc.m_weight_ise_range == mode_desc.m_transcode_weight_ise_range);
									assert(mode_desc.m_endpoint_ise_range == mode_desc.m_transcode_endpoint_ise_range);

									trial_result res;

									status = encode_block_3_subsets(
										res,
										cem,
										mode_desc.m_grid_x, mode_desc.m_grid_y,
										mode_desc.m_weight_ise_range, mode_desc.m_endpoint_ise_range,
										&half_pixels[0][0], (vec4F*)block_pixels_q16,
										coptions,
										uber_mode_flag,
										best_parts3, total_parts3, comp_level, mode11_opt_mode);

									if (!status)
										break;

									assert(res.m_valid);

									candidate_encoding candidate;
									candidate.m_coder.reserve(24);
									astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;

									coded_log_blk = res.m_log_blk;

									astc_helpers::log_astc_block& decomp_blk = candidate.m_decomp_log_blk;
									decomp_blk = res.m_log_blk;

									if (!validate_log_blk(decomp_blk))
									{
										fmt_error_printf("pack_astc_block() failed\n");
										return false;
									}

									status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_blk, &candidate.m_comp_pixels[0][0]);
									if (!status)
									{
										fmt_error_printf("decode_astc_block() failed\n");
										return false;
									}

									candidate.m_coder.put_bits(BLOCK_CODE, BLOCK_CODE_LEN);
									code_block(candidate.m_coder, candidate.m_coded_log_blk, bm, em, nullptr);

									candidate.m_encoding_type = encoding_type::cBlock;
									candidate.m_endpoint_mode = em;
									candidate.m_block_mode = bm;

									candidates.emplace_back(std::move(candidate));
								}
								else if (mode_desc.m_num_partitions == 2)
								{
									assert(!dual_plane);

									if (!has_estimated_parts2)
										break;

									assert(mode_desc.m_weight_ise_range == mode_desc.m_transcode_weight_ise_range);
									assert(mode_desc.m_endpoint_ise_range == mode_desc.m_transcode_endpoint_ise_range);

									for (uint32_t est_part_iter = 0; est_part_iter < total_parts2; est_part_iter++)
									{
										trial_result results[2];

										assert(((cem == 11) && any_2subset_mode11_enabled) || ((cem == 7) && any_2subset_mode7_enabled));

										status = encode_block_2_subsets(
											results,
											mode_desc.m_grid_x, mode_desc.m_grid_y,
											mode_desc.m_cem,
											mode_desc.m_weight_ise_range, mode_desc.m_endpoint_ise_range,
											&half_pixels[0][0], (vec4F*)block_pixels_q16,
											coptions,
											uber_mode_flag,
											(cem == 11) ? best_parts2_mode11[est_part_iter] : best_parts2_mode7[est_part_iter],
											comp_level,
											mode11_opt_mode,
											true);

										if (!status)
											continue;

										for (uint32_t r_iter = 0; r_iter < 2; r_iter++)
										{
											const trial_result& res = results[r_iter];

											if (!res.m_valid)
												continue;

											candidate_encoding candidate;
											candidate.m_coder.reserve(24);
											astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;

											coded_log_blk = res.m_log_blk;

											astc_helpers::log_astc_block& decomp_blk = candidate.m_decomp_log_blk;
											decomp_blk = res.m_log_blk;

											if (!validate_log_blk(decomp_blk))
											{
												fmt_error_printf("pack_astc_block() failed\n");
												return false;
											}

											status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_blk, &candidate.m_comp_pixels[0][0]);
											if (!status)
											{
												fmt_error_printf("decode_astc_block() failed\n");
												return false;
											}

											candidate.m_coder.put_bits(BLOCK_CODE, BLOCK_CODE_LEN);
											code_block(candidate.m_coder, candidate.m_coded_log_blk, bm, em, nullptr);

											candidate.m_encoding_type = encoding_type::cBlock;
											candidate.m_endpoint_mode = em;
											candidate.m_block_mode = bm;

											candidates.emplace_back(std::move(candidate));

										} // r_iter
									}
								}
								else
								{
									// 1 subset
									uint8_t blk_weights0[BLOCK_W * BLOCK_H], blk_weights1[BLOCK_W * BLOCK_H];
									uint32_t best_submode = 0;

									candidate_encoding candidate;
									candidate.m_coder.reserve(24);
									astc_helpers::log_astc_block& coded_log_blk = candidate.m_coded_log_blk;

									const uint32_t grid_x = mode_desc.m_grid_x, grid_y = mode_desc.m_grid_y;
									const uint32_t num_grid_samples = grid_x * grid_y;

									const half_vec3* pBlock_pixels_half = &half_pixels[0][0];
									const vec4F* pBlock_pixels_q16 = &block_pixels_q16[0][0];

									const uint32_t num_grid_samples_dp = num_grid_samples * (dual_plane ? 2 : 1);

									uint8_t transcode_weights[MAX_BLOCK_W * MAX_BLOCK_H * 2];

									coded_log_blk.m_grid_width = (uint8_t)grid_x;
									coded_log_blk.m_grid_height = (uint8_t)grid_y;
									coded_log_blk.m_dual_plane = (uint8_t)dual_plane;
									coded_log_blk.m_color_component_selector = (uint8_t)mode_desc.m_dp_channel;
									coded_log_blk.m_num_partitions = 1;
									coded_log_blk.m_color_endpoint_modes[0] = (uint8_t)mode_desc.m_cem;
									coded_log_blk.m_weight_ise_range = (uint8_t)mode_desc.m_weight_ise_range;
									coded_log_blk.m_endpoint_ise_range = (uint8_t)mode_desc.m_endpoint_ise_range;

									if ((cem == 11) && (!dual_plane) && ((grid_x < BLOCK_W) || (grid_y < BLOCK_H)))
									{
										double e = encode_astc_hdr_block_downsampled_mode_11(
											BLOCK_W, BLOCK_H, grid_x, grid_y,
											mode_desc.m_weight_ise_range, mode_desc.m_endpoint_ise_range,
											NUM_BLOCK_PIXELS, (basist::half_float(*)[3])pBlock_pixels_half, pBlock_pixels_q16,
											BIG_FLOAT_VAL,
											FIRST_MODE11_SUBMODE_INDEX, MAX_MODE11_SUBMODE_INDEX, false, mode11_opt_mode,
											coded_log_blk.m_endpoints, coded_log_blk.m_weights, best_submode,
											coptions,
											&enc_block_stats);

										if (e == BIG_FLOAT_VAL)
											break;
									}
									else
									{
										if (cem == 7)
										{
											assert(!dual_plane);

											double e = encode_astc_hdr_block_mode_7(
												NUM_BLOCK_PIXELS,
												(basist::half_float(*)[3])pBlock_pixels_half, pBlock_pixels_q16,
												mode_desc.m_weight_ise_range,
												best_submode,
												BIG_FLOAT_VAL,
												coded_log_blk.m_endpoints,
												blk_weights0,
												coptions,
												mode_desc.m_endpoint_ise_range,
												0, MAX_MODE7_SUBMODE_INDEX,
												&enc_block_stats);
											BASISU_NOTE_UNUSED(e);
										}
										else
										{
											double e;

											if (dual_plane)
											{
												e = encode_astc_hdr_block_mode_11_dual_plane(
													NUM_BLOCK_PIXELS,
													(basist::half_float(*)[3])pBlock_pixels_half, pBlock_pixels_q16,
													mode_desc.m_dp_channel,
													mode_desc.m_weight_ise_range,
													best_submode,
													BIG_FLOAT_VAL,
													coded_log_blk.m_endpoints,
													blk_weights0, blk_weights1,
													coptions,
													false,
													mode_desc.m_endpoint_ise_range, uber_mode_flag, false, -1, 7, false);
											}
											else
											{
												e = encode_astc_hdr_block_mode_11(
													NUM_BLOCK_PIXELS,
													(basist::half_float(*)[3])pBlock_pixels_half, pBlock_pixels_q16,
													mode_desc.m_weight_ise_range,
													best_submode,
													BIG_FLOAT_VAL,
													coded_log_blk.m_endpoints,
													blk_weights0,
													coptions,
													false,
													mode_desc.m_endpoint_ise_range, uber_mode_flag, false, -1, 7, false,
													mode11_opt_mode,
													&enc_block_stats);
											}

											if (e == BIG_FLOAT_VAL)
												break;
										}

										if (dual_plane)
										{
											downsample_ise_weights_dual_plane(
												mode_desc.m_weight_ise_range, mode_desc.m_weight_ise_range,
												BLOCK_W, BLOCK_H,
												grid_x, grid_y,
												blk_weights0, blk_weights1,
												coded_log_blk.m_weights);
										}
										else
										{
											downsample_ise_weights(
												mode_desc.m_weight_ise_range, mode_desc.m_weight_ise_range,
												BLOCK_W, BLOCK_H,
												grid_x, grid_y,
												blk_weights0, coded_log_blk.m_weights);

											if ((comp_level >= MIN_REFINE_LEVEL) && ((grid_x < BLOCK_W) || (grid_y < BLOCK_H)))
											{
												bool refine_status = refine_endpoints(cem,
													mode_desc.m_endpoint_ise_range, coded_log_blk.m_endpoints,
													6, 6, mode_desc.m_grid_x, mode_desc.m_grid_y,
													coded_log_blk.m_weights, mode_desc.m_weight_ise_range,
													BLOCK_W * BLOCK_H,
													(basist::half_float(*)[3])pBlock_pixels_half, (vec4F*)pBlock_pixels_q16,
													nullptr,
													coptions, mode11_opt_mode);
												BASISU_NOTE_UNUSED(refine_status);
											}
										}
									}

									basist::astc_6x6_hdr::requantize_astc_weights(num_grid_samples_dp, coded_log_blk.m_weights, mode_desc.m_weight_ise_range, transcode_weights, mode_desc.m_transcode_weight_ise_range);

									// Create the block the decoder would transcode into.
									astc_helpers::log_astc_block& decomp_blk = candidate.m_decomp_log_blk;
									decomp_blk.clear();

									decomp_blk.m_color_endpoint_modes[0] = (uint8_t)mode_desc.m_cem;
									decomp_blk.m_dual_plane = mode_desc.m_dp;
									decomp_blk.m_color_component_selector = (uint8_t)mode_desc.m_dp_channel;
									decomp_blk.m_num_partitions = 1;
									decomp_blk.m_endpoint_ise_range = (uint8_t)mode_desc.m_transcode_endpoint_ise_range;
									decomp_blk.m_weight_ise_range = (uint8_t)mode_desc.m_transcode_weight_ise_range;

									basist::astc_6x6_hdr::requantize_ise_endpoints(mode_desc.m_cem, mode_desc.m_endpoint_ise_range, coded_log_blk.m_endpoints, mode_desc.m_transcode_endpoint_ise_range, decomp_blk.m_endpoints);

									copy_weight_grid(dual_plane, grid_x, grid_y, transcode_weights, decomp_blk);

									if (!validate_log_blk(decomp_blk))
									{
										fmt_error_printf("pack_astc_block() failed\n");
										return false;
									}

									status = decode_astc_block(BLOCK_W, BLOCK_H, decomp_blk, &candidate.m_comp_pixels[0][0]);
									if (!status)
									{
										fmt_error_printf("decode_astc_block() failed\n");
										return false;
									}

									candidate.m_coder.put_bits(BLOCK_CODE, BLOCK_CODE_LEN);
									code_block(candidate.m_coder, candidate.m_coded_log_blk, bm, em, nullptr);

									candidate.m_encoding_type = encoding_type::cBlock;
									candidate.m_endpoint_mode = em;
									candidate.m_block_mode = bm;

									candidates.emplace_back(std::move(candidate));
								}

								break;
							}
							default:
								assert(0);
								fmt_debug_printf("Invalid endpoint mode\n");
								return false;

							} // switch (em)

						} // endpoint_mode_iter

					} // block_mode_iter

				} // is_solid_block

				//------------------------------------------------

				debug_state.m_total_candidates_considered.fetch_add(candidates.size_u32(), std::memory_order_relaxed);
				atomic_max(debug_state.m_max_candidates_considered, candidates.size_u32());

				for (uint32_t candidate_iter = 0; candidate_iter < candidates.size_u32(); candidate_iter++)
				{
					auto& candidate = candidates[candidate_iter];

					for (uint32_t y = 0; y < BLOCK_H; y++)
						for (uint32_t x = 0; x < BLOCK_W; x++)
							linear_rgb_to_itp(candidate.m_comp_pixels[y][x], candidate.m_comp_pixels_itp[y][x], global_cfg);
				}

				// Find best overall candidate
				double best_t = BIG_FLOAT_VAL;
				int best_candidate_index = -1;

				float best_d_ssim = BIG_FLOAT_VAL;

				if (global_cfg.m_lambda == 0.0f)
				{
					for (uint32_t candidate_iter = 0; candidate_iter < candidates.size_u32(); candidate_iter++)
					{
						const auto& candidate = candidates[candidate_iter];

						float candidate_d_ssim = 1.0f - compute_block_ssim_itp(BLOCK_W, BLOCK_H, &block_pixels_as_itp[0][0], &candidate.m_comp_pixels_itp[0][0]);

						if (candidate_d_ssim < best_d_ssim)
							best_d_ssim = candidate_d_ssim;

						candidate_d_ssim *= SSIM_WEIGHT;

						float candidate_mse = MSE_WEIGHT * compute_block_mse_itp(BLOCK_W, BLOCK_H, &block_pixels_as_itp[0][0], &candidate.m_comp_pixels_itp[0][0], global_cfg.m_delta_itp_dark_adjustment);

						candidate_mse += candidate_d_ssim;

						float total_deblock_penalty = 0.0f;
						if (global_cfg.m_deblocking_flag)
						{
							total_deblock_penalty = calc_deblocking_penalty_itp(bx, by, width, height, pass_src_img_itp, candidate) * global_cfg.m_deblock_penalty_weight;
						}
						candidate_mse += total_deblock_penalty * SSIM_WEIGHT;

						if ((candidate.m_encoding_type == encoding_type::cBlock) || (candidate.m_encoding_type == encoding_type::cReuse))
						{
							// Bias the encoder away from 2 level blocks on complex blocks
							// TODO: Perhaps only do this on large or non-interpolated grids
							if (complex_block)
							{
								if (candidate.m_coded_log_blk.m_weight_ise_range == astc_helpers::BISE_2_LEVELS)
								{
									candidate_mse *= TWO_LEVEL_PENALTY;
								}
							}

							// Bias the encoder away from smaller weight grids if the block is very complex
							// TODO: Use the DCT to compute an approximation of the block energy/variance retained vs. lost by downsampling.
							if (complex_block)
							{
								if ((candidate.m_coded_log_blk.m_grid_width == 2) && (candidate.m_coded_log_blk.m_grid_height == 2))
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_2X2_MSE_PENALTY;
								else if (minimum(candidate.m_coded_log_blk.m_grid_width, candidate.m_coded_log_blk.m_grid_height) <= 3)
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_3X3_MSE_PENALTY;
								else if (minimum(candidate.m_coded_log_blk.m_grid_width, candidate.m_coded_log_blk.m_grid_height) <= 4)
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_4X4_MSE_PENALTY;
							}
						}

						float candidate_t = candidate_mse;

						if (candidate_t < best_t)
						{
							best_t = candidate_t;
							best_candidate_index = candidate_iter;
						}

					} // candidate_iter

					if (global_cfg.m_gaussian1_fallback && (outer_pass == 0) && (very_complex_block) && (best_d_ssim > SWITCH_TO_GAUSSIAN_FILTERED_THRESH1_D_SSIM))
					{
						debug_state.m_total_gaussian1_blocks.fetch_add(1, std::memory_order_relaxed);
						continue;
					}

					const float block_y_contrast_ratio = block_hy / (block_ly + .00000125f);

					if (global_cfg.m_gaussian2_fallback && (comp_level >= 1) && (outer_pass == 1) && (very_complex_block) && (best_d_ssim > SWITCH_TO_GAUSSIAN_FILTERED_THRESH2_D_SSIM) &&
						(block_hy >= 18.0f) && (block_y_contrast_ratio > 150.0f) &&
						(block_avg_y >= 1.5f))
					{
						debug_state.m_total_gaussian2_blocks.fetch_add(1, std::memory_order_relaxed);
						continue;
					}
				}
				else
				{
					assert(enc_state.smooth_block_mse_scales.get_width() > 0);

					// Compute block's perceptual weighting
					float perceptual_scale = 0.0f;
					for (uint32_t y = 0; y < BLOCK_H; y++)
						for (uint32_t x = 0; x < BLOCK_W; x++)
							perceptual_scale = basisu::maximumf(perceptual_scale, enc_state.smooth_block_mse_scales.at_clamped(bx * BLOCK_W + x, by * BLOCK_H + y));

					// Very roughly normalize the computed distortion vs. bits.
					perceptual_scale *= 10.0f;

					for (uint32_t candidate_iter = 0; candidate_iter < candidates.size_u32(); candidate_iter++)
					{
						auto& candidate = candidates[candidate_iter];

						float d_ssim = 1.0f - compute_block_ssim_itp(BLOCK_W, BLOCK_H, &block_pixels_as_itp[0][0], &candidate.m_comp_pixels_itp[0][0]);

						if (d_ssim < best_d_ssim)
							best_d_ssim = (float)d_ssim;

						d_ssim *= SSIM_WEIGHT;

						float candidate_mse = MSE_WEIGHT * compute_block_mse_itp(BLOCK_W, BLOCK_H, &block_pixels_as_itp[0][0], &candidate.m_comp_pixels_itp[0][0], global_cfg.m_delta_itp_dark_adjustment);

						candidate_mse += d_ssim;

						float total_deblock_penalty = 0.0f;
						if (global_cfg.m_deblocking_flag)
						{
							total_deblock_penalty = calc_deblocking_penalty_itp(bx, by, width, height, pass_src_img_itp, candidate) * global_cfg.m_deblock_penalty_weight;
						}
						candidate_mse += total_deblock_penalty * SSIM_WEIGHT;

						if ((candidate.m_encoding_type == encoding_type::cBlock) || (candidate.m_encoding_type == encoding_type::cReuse))
						{
							// Bias the encoder away from 2 level blocks on complex blocks
							if (complex_block)
							{
								if (candidate.m_coded_log_blk.m_weight_ise_range == astc_helpers::BISE_2_LEVELS)
								{
									candidate_mse *= TWO_LEVEL_PENALTY;
								}
							}

							// Bias the encoder away from smaller weight grids if the block is very complex
							if (complex_block)
							{
								if ((candidate.m_coded_log_blk.m_grid_width == 2) && (candidate.m_coded_log_blk.m_grid_height == 2))
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_2X2_MSE_PENALTY;
								else if (minimum(candidate.m_coded_log_blk.m_grid_width, candidate.m_coded_log_blk.m_grid_height) <= 3)
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_3X3_MSE_PENALTY;
								else if (minimum(candidate.m_coded_log_blk.m_grid_width, candidate.m_coded_log_blk.m_grid_height) <= 4)
									candidate_mse *= COMPLEX_BLOCK_WEIGHT_GRID_4X4_MSE_PENALTY;
							}
						}

						float mode_penalty = 1.0f;
						if (candidate.m_encoding_type == encoding_type::cSolid)
							mode_penalty *= SOLID_PENALTY;
						else if (candidate.m_encoding_type == encoding_type::cReuse)
							mode_penalty *= REUSE_PENALTY;
						else if (candidate.m_encoding_type == encoding_type::cRun)
							mode_penalty *= (complex_block ? RUN_PENALTY * 2.0f : RUN_PENALTY);

						float candidate_bits = (float)candidate.m_coder.get_total_bits();
						float candidate_d = candidate_mse * mode_penalty;

						const float D_POWER = 2.0f;
						float candidate_t = perceptual_scale * powf(candidate_d, D_POWER) + candidate_bits * (global_cfg.m_lambda * 1000.0f);

						candidate.m_t = candidate_t;
						candidate.m_d = candidate_d;
						candidate.m_bits = candidate_bits;

						if (candidate_t < best_t)
						{
							best_t = candidate_t;
							best_candidate_index = candidate_iter;
						}

					} // candidate_iter

					if (global_cfg.m_gaussian1_fallback && (outer_pass == 0) && (very_complex_block) && (best_d_ssim > SWITCH_TO_GAUSSIAN_FILTERED_THRESH1_D_SSIM))
					{
						debug_state.m_total_gaussian1_blocks.fetch_add(1, std::memory_order_relaxed);
						continue;
					}

					const float block_y_contrast_ratio = block_hy / (block_ly + .00000125f);

					if (global_cfg.m_gaussian2_fallback && (comp_level >= 1) && (outer_pass == 1) && (very_complex_block) && (best_d_ssim > SWITCH_TO_GAUSSIAN_FILTERED_THRESH2_D_SSIM) &&
						(block_hy >= 18.0f) && (block_y_contrast_ratio > 150.0f) &&
						(block_avg_y >= 1.5f))
					{
						debug_state.m_total_gaussian2_blocks.fetch_add(1, std::memory_order_relaxed);
						continue;
					}

					if (global_cfg.m_rdo_candidate_diversity_boost)
					{
						// candidate diversity boosting - consider candidates along/near the Pareto front
						const candidate_encoding& comp_candidate = candidates[best_candidate_index];

						float best_d = BIG_FLOAT_VAL;

						for (uint32_t candidate_iter = 0; candidate_iter < candidates.size_u32(); candidate_iter++)
						{
							const auto& candidate = candidates[candidate_iter];

							if (candidate.m_bits <= comp_candidate.m_bits * global_cfg.m_rdo_candidate_diversity_boost_bit_window_weight)
							{
								if (candidate.m_d < best_d)
								{
									best_d = candidate.m_d;
									best_candidate_index = candidate_iter;
								}
							}
						}
					}

					// candidate JND optimization - if there's a cheaper to code candidate that is nearly equivalent visually to the best candidate chose, choose that
					if (global_cfg.m_jnd_optimization)
					{
						const candidate_encoding& cur_comp_candidate = candidates[best_candidate_index];

						float new_best_candidate_bits = BIG_FLOAT_VAL;
						int new_best_candidate_index = -1;

						for (uint32_t candidate_iter = 0; candidate_iter < candidates.size_u32(); candidate_iter++)
						{
							if ((int)candidate_iter == best_candidate_index)
								continue;

							const auto& candidate = candidates[candidate_iter];

							if (candidate.m_bits >= cur_comp_candidate.m_bits)
								continue;

							float max_delta_itp = 0.0f;
							for (uint32_t y = 0; y < BLOCK_H; y++)
							{
								for (uint32_t x = 0; x < BLOCK_W; x++)
								{
									float delta_itp = compute_pixel_delta_itp(cur_comp_candidate.m_comp_pixels_itp[y][x], candidate.m_comp_pixels_itp[y][x], block_pixels_as_itp[y][x], global_cfg.m_delta_itp_dark_adjustment);
									max_delta_itp = maximum(max_delta_itp, delta_itp);

									if (max_delta_itp >= global_cfg.m_jnd_delta_itp_thresh)
										goto skip;
								}
							}

						skip:
							if (max_delta_itp >= global_cfg.m_jnd_delta_itp_thresh)
								continue;

							if (candidate.m_bits < new_best_candidate_bits)
							{
								new_best_candidate_bits = candidate.m_bits;
								new_best_candidate_index = candidate_iter;
							}
						}

						if (new_best_candidate_index != -1)
						{
							best_candidate_index = new_best_candidate_index;
							debug_state.m_total_jnd_replacements.fetch_add(1, std::memory_order_relaxed);
						}
					}

				} // if (lambda == 0.0f)

				if (global_cfg.m_debug_images)
				{
					std::lock_guard<std::mutex> lck(debug_state.m_stat_vis_mutex);
					debug_state.m_stat_vis.fill_box(bx * 6, by * 6, 6, 6, vec4F(best_d_ssim, max_std_dev, lowpass_std_dev, 1.0f));
				}

				if (best_candidate_index < 0)
				{
					assert(best_candidate_index >= 0);
					fmt_error_printf("No candidates!\n");
					return false;
				}

				const auto& best_candidate = candidates[best_candidate_index];

				assert(best_candidate.m_encoding_type != encoding_type::cInvalid);

				if (best_candidate.m_encoding_type == encoding_type::cRun)
				{
					if (!prev_run_len)
					{
						if (prev_encoding.get_total_bits())
						{
#if SYNC_MARKERS
							strip_coded_bits.put_bits(0xDEAD, 16);
#endif

							strip_coded_bits.append(prev_encoding);
						}

						assert(best_candidate.m_coder.get_total_bits());

						prev_encoding = best_candidate.m_coder;

						prev_run_len = 1;
					}
					else
					{
						prev_run_len++;

						const uint32_t prev_run_bits = prev_encoding.get_total_bits_u32();
						assert(prev_run_bits);
						BASISU_NOTE_UNUSED(prev_run_bits);

						const uint32_t num_dummy_bits = best_candidate.m_coder.get_total_bits_u32();
						BASISU_NOTE_UNUSED(num_dummy_bits);

						// Rewrite the previous encoding to extend the run length.
						prev_encoding.restart();
						prev_encoding.put_bits(RUN_CODE, RUN_CODE_LEN);
						prev_encoding.put_vlc(prev_run_len - 1, 5);

						assert(prev_encoding.get_total_bits() == prev_run_bits + num_dummy_bits);
					}
				}
				else
				{
					if (prev_encoding.get_total_bits())
					{
#if SYNC_MARKERS
						strip_coded_bits.put_bits(0xDEAD, 16);
#endif

						strip_coded_bits.append(prev_encoding);
					}

					prev_encoding = best_candidate.m_coder;
					prev_run_len = 0;
				}

				memcpy(prev_comp_pixels, best_candidate.m_comp_pixels, sizeof(vec3F) * BLOCK_W * BLOCK_H);

				prev_candidate_encoding = best_candidate;

				if (best_candidate.m_encoding_type != encoding_type::cRun)
					prev_non_run_candidate_encoding = best_candidate;

				{
					std::lock_guard<std::mutex> lck(debug_state.m_stats_mutex);

					debug_state.m_encoding_type_hist[(uint32_t)best_candidate.m_encoding_type]++;

					if (best_candidate.m_encoding_type == encoding_type::cBlock)
					{
						debug_state.m_endpoint_mode_hist[(uint32_t)best_candidate.m_endpoint_mode]++;
					}

					if ((best_candidate.m_encoding_type == encoding_type::cReuse) || (best_candidate.m_encoding_type == encoding_type::cBlock))
					{
						const uint32_t bm_index = (uint32_t)best_candidate.m_block_mode;
						assert(bm_index < (uint32_t)block_mode::cBMTotalModes);

						debug_state.m_block_mode_hist[bm_index]++;
						debug_state.m_block_mode_total_bits[bm_index] += best_candidate.m_coder.get_total_bits();

						for (uint32_t i = 0; i < 3; i++)
						{
							debug_state.m_block_mode_comp_stats[bm_index][i].push_back(half_comp_stats[i]);
							debug_state.m_block_mode_comparative_stats[bm_index][i].push_back(half_cross_chan_stats[i]);
						}
					}

					if (best_candidate.m_encoding_type == encoding_type::cReuse)
					{
						debug_state.m_reuse_num_parts[best_candidate.m_coded_log_blk.m_num_partitions].fetch_add(1, std::memory_order_relaxed);

						if (best_candidate.m_coded_log_blk.m_dual_plane)
							debug_state.m_reuse_total_dp.fetch_add(1, std::memory_order_relaxed);
					}
				}

				enc_state.coded_blocks(bx, by) = prev_non_run_candidate_encoding;

				// Update decoded image
				vec4F decoded_float_pixels[BLOCK_H][BLOCK_W];
				for (uint32_t y = 0; y < BLOCK_H; y++)
					for (uint32_t x = 0; x < BLOCK_W; x++)
						decoded_float_pixels[y][x] = best_candidate.m_comp_pixels[y][x];

				enc_state.packed_img.set_block_clipped((vec4F*)decoded_float_pixels, bx * BLOCK_W, by * BLOCK_H, BLOCK_W, BLOCK_H);

				status = astc_helpers::pack_astc_block(enc_state.final_astc_blocks(bx, by), best_candidate.m_decomp_log_blk, nullptr, nullptr);
				if (!status)
				{
					fmt_error_printf("Failed packing block\n");
					return false;
				}

				const uint32_t r = debug_state.m_total_blocks_compressed.fetch_add(1, std::memory_order_relaxed);
				if ((r & 2047) == 2047)
				{
					if (global_cfg.m_status_output)
					{
						basisu::fmt_printf("{} of {} total blocks compressed, {3.2}%\n", r, total_blocks, (r * 100.0f) / total_blocks);
					}
				}

				if ((global_cfg.m_debug_images) &&
					((best_candidate.m_encoding_type != encoding_type::cRun) && (best_candidate.m_encoding_type != encoding_type::cSolid)))
				{
					std::lock_guard<std::mutex> lck(debug_state.m_vis_image_mutex);

					if (best_candidate.m_decomp_log_blk.m_num_partitions == 2)
					{
						const int part2_unique_index = g_part2_seed_to_unique_index[best_candidate.m_decomp_log_blk.m_partition_id];
						assert((part2_unique_index >= 0) && (part2_unique_index < (int)NUM_UNIQUE_PARTITIONS2));

						const partition_pattern_vec& pat = g_partitions2[part2_unique_index];

						for (uint32_t y = 0; y < 6; y++)
						{
							for (uint32_t x = 0; x < 6; x++)
							{
								const uint32_t p = pat[x + y * 6];
								debug_state.m_part_vis.set_clipped(bx * 6 + x, by * 6 + y, color_rgba(p ? 100 : 0, 128, p ? 100 : 0, 255));
							} // x
						} // y 
					}
					else if (best_candidate.m_decomp_log_blk.m_num_partitions == 3)
					{
						//part_vis.fill_box(bx * 6, by * 6, 6, 6, color_rgba(0, 0, 255, 255));

						const int part3_unique_index = g_part3_seed_to_unique_index[best_candidate.m_decomp_log_blk.m_partition_id];
						assert((part3_unique_index >= 0) && (part3_unique_index < (int)NUM_UNIQUE_PARTITIONS3));

						const partition_pattern_vec& pat = g_partitions3[part3_unique_index];

						for (uint32_t y = 0; y < 6; y++)
						{
							for (uint32_t x = 0; x < 6; x++)
							{
								const uint32_t p = pat[x + y * 6];
								color_rgba c(0, 0, 150, 255);
								if (p == 1)
									c.set(100, 0, 150, 255);
								else if (p == 2)
									c.set(0, 100, 150, 255);
								debug_state.m_part_vis.set_clipped(bx * 6 + x, by * 6 + y, c);
							} // x
						} // y 
					}
					else if (best_candidate.m_decomp_log_blk.m_dual_plane)
					{
						debug_state.m_part_vis.fill_box(bx * 6, by * 6, 6, 6, color_rgba(255, 0, 255, 255));
					}
					else
					{
						debug_state.m_part_vis.fill_box(bx * 6, by * 6, 6, 6, color_rgba(255, 0, 0, 255));
					}

					color_rgba c;
					c.set((best_candidate.m_coded_log_blk.m_grid_width * best_candidate.m_coded_log_blk.m_grid_height * 255 + 18) / 36);
					debug_state.m_grid_vis.fill_box(bx * 6, by * 6, 6, 6, c);

					c.set(0, 0, 0, 255);
					if (complex_block)
						c[0] = 255;

					if (very_complex_block)
						c[1] = 255;

					if (outer_pass == 2)
						c[2] = 255;
					else if (outer_pass == 1)
						c[2] = 128;

					debug_state.m_mode_vis.fill_box(bx * 6, by * 6, 6, 6, c);

					c.set(0, 255, 0, 255);
					if (best_candidate.m_coded_log_blk.m_color_endpoint_modes[0] == 7)
						c.set(255, 0, 0, 255);
					debug_state.m_mode_vis2.fill_box(bx * 6, by * 6, 6, 6, c);

					switch (best_candidate.m_encoding_type)
					{
					case encoding_type::cRun:
						c.set(0, 0, 0, 255);
						break;
					case encoding_type::cSolid:
						c.set(128, 128, 128, 255); // dark grey
						break;
					case encoding_type::cReuse:
						c.set(255, 255, 0, 255); // yellow
						break;
					case encoding_type::cBlock:
					{
						switch (best_candidate.m_endpoint_mode)
						{
						case endpoint_mode::cRaw:
							c.set(255, 0, 0, 255); // red
							break;
						case endpoint_mode::cUseLeft:
							c.set(0, 0, 255, 255); // blue
							break;
						case endpoint_mode::cUseUpper:
							c.set(0, 0, 192, 255); // darker blue
							break;
						case endpoint_mode::cUseLeftDelta:
							c.set(0, 255, 0, 255); // green
							break;
						case endpoint_mode::cUseUpperDelta:
							c.set(0, 192, 0, 255); // darker green
							break;
						default:
							break;
						}

						break;
					}
					default:
						break;
					}

					if (filtered_x_err < filtered_y_err)
						c[3] = 0;
					else
						c[3] = 255;

					debug_state.m_enc_vis.fill_box(bx * 6, by * 6, 6, 6, c);
				}

				break;

			} // outer_pass

		} // bx

	} // by

	if (prev_encoding.get_total_bits())
	{
#if SYNC_MARKERS
		strip_coded_bits.put_bits(0xDEAD, 16);
#endif

		strip_coded_bits.append(prev_encoding);
	}

	return true;
}

bool g_initialized = false;

void global_init()
{
	if (g_initialized)
		return;

	interval_timer tm;
	tm.start();

	init_pq_tables();
		
	init_partitions2_6x6();
	init_partitions3_6x6();

	init_contrib_lists();

	g_initialized = true;

	//fmt_printf("astc_6x6_hdr::global_init() total time: {}\n", tm.get_elapsed_secs());
}

bool compress_photo(const basisu::imagef &orig_src_img, const astc_hdr_6x6_global_config &orig_global_cfg, job_pool *pJob_pool,
	basisu::uint8_vec& intermediate_tex_data, basisu::uint8_vec& astc_tex_data, result_metrics& metrics)
{
	assert(g_initialized);
	if (!g_initialized)
		return false;
	
	assert(pJob_pool);

	if (orig_global_cfg.m_debug_output)
	{
		fmt_debug_printf("------ astc_6x6_hdr::compress_photo:\n");
		fmt_debug_printf("Source image dimensions: {}x{}\n", orig_src_img.get_width(), orig_src_img.get_height());
		fmt_debug_printf("Job pool total threads: {}\n", (uint64_t)pJob_pool->get_total_threads());
		orig_global_cfg.print();
	}

	if (!orig_src_img.get_width() || !orig_src_img.get_height())
	{
		assert(false);
		fmt_error_printf("compress_photo: Invalid source image\n");
		return false;
	}

	astc_hdr_6x6_global_config global_cfg(orig_global_cfg);

	uastc_hdr_6x6_encode_state enc_state;
	enc_state.master_coptions.m_q_log_bias = Q_LOG_BIAS_6x6;
	enc_state.src_img = orig_src_img;

	//src_img.crop(256, 256);

	const uint32_t width = enc_state.src_img.get_width();
	const uint32_t height = enc_state.src_img.get_height();
	const uint32_t num_blocks_x = enc_state.src_img.get_block_width(BLOCK_W);
	const uint32_t num_blocks_y = enc_state.src_img.get_block_height(BLOCK_H);
	const uint32_t total_blocks = num_blocks_x * num_blocks_y;

	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			for (uint32_t c = 0; c < 3; c++)
			{
				float f = enc_state.src_img(x, y)[c];

				if (std::isinf(f) || std::isnan(f) || (f < 0.0f))
					f = 0;
				else if (f > basist::ASTC_HDR_MAX_VAL)
					f = basist::ASTC_HDR_MAX_VAL;

				enc_state.src_img(x, y)[c] = f;
								
			} // c
						
		} // x
	} // y
	
	if (global_cfg.m_debug_images)
	{
		write_exr((global_cfg.m_debug_image_prefix + "orig.exr").c_str(), enc_state.src_img, 3, 0);
	}
			
	image src_img_compressed;
	tonemap_image_compressive2(src_img_compressed, enc_state.src_img);

	if (global_cfg.m_debug_images)
	{
		save_png(global_cfg.m_debug_image_prefix + "compressive_tone_map.png", src_img_compressed);
	}

	smooth_map_params rp;
	rp.m_debug_images = global_cfg.m_debug_images;

	if (global_cfg.m_lambda != 0.0f)
	{
		if (global_cfg.m_status_output)
			fmt_printf("Creating RDO perceptual weighting maps\n");

		create_smooth_maps2(enc_state.smooth_block_mse_scales, src_img_compressed, rp);
	}

	if (global_cfg.m_status_output)
		fmt_printf("Blurring image\n");

	enc_state.src_img_filtered1.resize(width, height);
	image_resample(enc_state.src_img, enc_state.src_img_filtered1, "gaussian", global_cfg.m_gaussian1_strength); //1.45f);
	
	enc_state.src_img_filtered2.resize(width, height);
	image_resample(enc_state.src_img, enc_state.src_img_filtered2, "gaussian", global_cfg.m_gaussian2_strength); //1.83f);
		
	if (global_cfg.m_debug_images)
	{
		write_exr((global_cfg.m_debug_image_prefix + "blurred1.exr").c_str(), enc_state.src_img_filtered1, 3, 0);
		write_exr((global_cfg.m_debug_image_prefix + "blurred2.exr").c_str(), enc_state.src_img_filtered2, 3, 0);
	}

	if (global_cfg.m_status_output)
		fmt_printf("Transforming to ITP\n");

	enc_state.src_img_itp.resize(width, height);
	convet_rgb_image_to_itp(enc_state.src_img, enc_state.src_img_itp, global_cfg);
	
	enc_state.src_img_filtered1_itp.resize(width, height);
	convet_rgb_image_to_itp(enc_state.src_img_filtered1, enc_state.src_img_filtered1_itp, global_cfg);
	
	enc_state.src_img_filtered2_itp.resize(width, height);
	convet_rgb_image_to_itp(enc_state.src_img_filtered2, enc_state.src_img_filtered2_itp, global_cfg);

	if (global_cfg.m_lambda == 0.0f)
		global_cfg.m_favor_higher_compression = false;

	uint32_t total_strips = 0, rows_per_strip = 0;
	if (!calc_strip_size(global_cfg.m_lambda, num_blocks_y, (uint32_t)pJob_pool->get_total_threads(), global_cfg.m_force_one_strip, total_strips, rows_per_strip, global_cfg))
	{
		fmt_error_printf("compress_photo: Failed computing strip sizes\n");
		return false;
	}
		
	if (global_cfg.m_debug_output)
		fmt_printf("lambda: {}, comp_level: {}, highest_comp_level: {}, extra patterns: {}\n", global_cfg.m_lambda, global_cfg.m_master_comp_level, global_cfg.m_highest_comp_level, global_cfg.m_extra_patterns_flag);
					
	enc_state.coded_blocks.resize(num_blocks_x, num_blocks_y);
						
	bitwise_coder coded_bits;

	coded_bits.put_bits(0xABCD, 16);
	coded_bits.put_bits(width, 16);
	coded_bits.put_bits(height, 16);
					
	enc_state.packed_img.resize(width, height);
		
	enc_state.strip_bits.resize(total_strips);

	enc_state.final_astc_blocks.resize(num_blocks_x, num_blocks_y);

	uastc_hdr_6x6_debug_state debug_state;

	if (global_cfg.m_debug_images)
		debug_state.init(width, height);
	else
		debug_state.init(0, 0);
		
	interval_timer tm;
	tm.start();

	std::atomic_bool any_failed_flag;
	any_failed_flag.store(false);

	for (uint32_t strip_index = 0; strip_index < total_strips; strip_index++)
	{
		const uint32_t strip_first_by = strip_index * rows_per_strip;
		
		uint32_t strip_last_by = minimum<uint32_t>(strip_first_by + rows_per_strip - 1, num_blocks_y);
		if (strip_index == (total_strips - 1))
			strip_last_by = num_blocks_y - 1;

		pJob_pool->add_job([&any_failed_flag, &global_cfg, &debug_state, &enc_state,
			strip_index, total_strips, strip_first_by, strip_last_by,
			num_blocks_x, num_blocks_y, total_blocks, width, height]
		{
			if (!any_failed_flag)
			{
				bool status = compress_strip_task(
					strip_index, total_strips, strip_first_by, strip_last_by,
					num_blocks_x, num_blocks_y, total_blocks, width, height,
					global_cfg, debug_state, enc_state);

				if (!status)
				{
					fmt_error_printf("compress_photo: compress_strip_task() failed\n");
					any_failed_flag.store(true, std::memory_order_relaxed);
				}
			}
		} );

		if (any_failed_flag)
			break;
	
	} // strip_index

	pJob_pool->wait_for_all();

	if (any_failed_flag)
	{
		fmt_error_printf("One or more strips failed during compression\n");
		return false;
	}
				
	if (global_cfg.m_debug_output)
		fmt_printf("Encoding time: {} secs\n", tm.get_elapsed_secs());

	if (global_cfg.m_debug_output)
		debug_state.print(total_blocks);

	if (global_cfg.m_debug_images)
	{
		save_png(global_cfg.m_debug_image_prefix +  "part_vis.png", debug_state.m_part_vis);
		save_png(global_cfg.m_debug_image_prefix + "grid_vis.png", debug_state.m_grid_vis);
		save_png(global_cfg.m_debug_image_prefix + "mode_vis.png", debug_state.m_mode_vis);
		save_png(global_cfg.m_debug_image_prefix + "mode_vis2.png", debug_state.m_mode_vis2);
		save_png(global_cfg.m_debug_image_prefix + "enc_vis.png", debug_state.m_enc_vis);
		write_exr((global_cfg.m_debug_image_prefix + "stat_vis.exr").c_str(), debug_state.m_stat_vis, 3, 0);
	}

	for (uint32_t i = 0; i < total_strips; i++)
		coded_bits.append(enc_state.strip_bits[i]);
		
	coded_bits.put_bits(0xA742, 16);

	coded_bits.flush();

	if (global_cfg.m_output_images)
	{
		write_exr((global_cfg.m_output_image_prefix + "comp.exr").c_str(), enc_state.packed_img, 3, 0);
	}
	
	if (global_cfg.m_debug_output)
		fmt_printf("\nTotal intermediate output bits/pixel: {3.4}\n", (float)coded_bits.get_total_bits() / (float)(width * height));

	vector2D<astc_helpers::astc_block> decoded_blocks1;
	vector2D<astc_helpers::astc_block> decoded_blocks2;
	
	if (global_cfg.m_debug_output)
		fmt_printf("decode_file\n");

	uint32_t unpacked_width = 0, unpacked_height = 0;
	bool status = decode_file(coded_bits.get_bytes(), decoded_blocks1, unpacked_width, unpacked_height);
	if (!status)
	{
		fmt_error_printf("decode_file() failed\n");
		return false;
	}

	if (global_cfg.m_debug_output)
		fmt_printf("decode_6x6_hdr\n");

	status = decode_6x6_hdr(coded_bits.get_bytes().get_ptr(), coded_bits.get_bytes().size_in_bytes_u32(), decoded_blocks2, unpacked_width, unpacked_height);
	if (!status)
	{
		fmt_error_printf("decode_6x6_hdr_file() failed\n");
		return false;
	}

	if ((enc_state.final_astc_blocks.get_width() != decoded_blocks1.get_width()) ||
		(enc_state.final_astc_blocks.get_height() != decoded_blocks1.get_height()))
	{
		fmt_error_printf("Decode size mismatch with decode_file\n");
		return false;
	}

	if ((enc_state.final_astc_blocks.get_width() != decoded_blocks2.get_width()) ||
		(enc_state.final_astc_blocks.get_height() != decoded_blocks2.get_height()))
	{
		fmt_error_printf("Decode size mismatch with decode_6x6_hdr_file\n");
		return false;
	}

	if (memcmp(decoded_blocks1.get_ptr(), enc_state.final_astc_blocks.get_ptr(), decoded_blocks1.size_in_bytes()) != 0)
	{
		fmt_error_printf("Decoded ASTC blocks verification failed\n");
		return false;
	}

	if (memcmp(decoded_blocks2.get_ptr(), enc_state.final_astc_blocks.get_ptr(), decoded_blocks2.size_in_bytes()) != 0)
	{
		fmt_error_printf("Decoded ASTC blocks verification failed\n");
		return false;
	}

	if (global_cfg.m_debug_output)
		basisu::fmt_printf("Decoded ASTC verification checks succeeded\n");

	if (global_cfg.m_output_images)
	{
		if (write_astc_file((global_cfg.m_output_image_prefix + "decoded.astc").c_str(), decoded_blocks1.get_ptr(), BLOCK_W, BLOCK_H, width, height))
		{
			basisu::platform_sleep(20);

			uint8_vec astc_file_data;
			if (read_file_to_vec((global_cfg.m_output_image_prefix + "decoded.astc").c_str(), astc_file_data))
			{
				if (astc_file_data.size() > 16)
				{
					astc_file_data.erase(0, 16);

					size_t comp_size = 0;
					void* pComp_data = tdefl_compress_mem_to_heap(&astc_file_data[0], astc_file_data.size(), &comp_size, TDEFL_MAX_PROBES_MASK);
					mz_free(pComp_data);

					if (global_cfg.m_debug_output)
					{
						fmt_printf(".ASTC file size (less header): {}, bits/pixel: {}, Deflate bits/pixel: {}\n",
							(uint64_t)astc_file_data.size(),
							(float)astc_file_data.size() * 8.0f / (float)(width * height),
							(float)comp_size * 8.0f / (float)(width * height));
					}
				}
			}
		}
	}

	// Must decode all the blocks (even padded rows/cols) to match what the transcoder does.
	imagef unpacked_astc_img(num_blocks_x * 6, num_blocks_y * 6);
	imagef unpacked_astc_google_img(num_blocks_x * 6, num_blocks_y * 6);

	for (uint32_t y = 0; y < decoded_blocks1.get_height(); y++)
	{
		for (uint32_t x = 0; x < decoded_blocks1.get_width(); x++)
		{
			const auto& phys_blk = decoded_blocks1(x, y);

			vec4F pixels[MAX_BLOCK_W * MAX_BLOCK_H];
			status = unpack_physical_astc_block(&phys_blk, BLOCK_W, BLOCK_H, pixels);
			if (!status)
			{
				fmt_error_printf("unpack_physical_astc_block() failed\n");
				return false;
			}
			
			unpacked_astc_img.set_block_clipped(pixels, x * BLOCK_W, y * BLOCK_H, BLOCK_W, BLOCK_H);

			vec4F pixels_google[MAX_BLOCK_W * MAX_BLOCK_H];
			status = unpack_physical_astc_block_google(&phys_blk, BLOCK_W, BLOCK_H, pixels_google);
			if (!status)
			{
				fmt_error_printf("unpack_physical_astc_block_google() failed\n");
				return false;
			}

			unpacked_astc_google_img.set_block_clipped(pixels_google, x * BLOCK_W, y * BLOCK_H, BLOCK_W, BLOCK_H);

			for (uint32_t i = 0; i < 36; i++)
			{
				if (pixels[i] != pixels_google[i])
				{
					fmt_error_printf("pixel unpack mismatch\n");
					return false;
				}
			}
		}
	}
		
	if (global_cfg.m_debug_output)
		fmt_printf("\nUnpack succeeded\n");

	imagef unpacked_bc6h_img;

	{
		vector2D<basist::bc6h_block> bc6h_blocks;
		
		fast_bc6h_params enc_params;
						
		bool pack_status = pack_bc6h_image(unpacked_astc_img, bc6h_blocks, &unpacked_bc6h_img, enc_params);
		if (!pack_status)
		{
			fmt_error_printf("pack_bc6h_image() failed!");
			return false;
		}

		unpacked_bc6h_img.crop(width, height);
		
		if (global_cfg.m_output_images)
		{
			write_exr((global_cfg.m_output_image_prefix + "unpacked_bc6h.exr").c_str(), unpacked_bc6h_img, 3, 0);
		}
	}

	unpacked_astc_img.crop(width, height);
	unpacked_astc_google_img.crop(width, height);
	
	if (global_cfg.m_output_images)
	{
		write_exr((global_cfg.m_output_image_prefix + "unpacked_astc.exr").c_str(), unpacked_astc_img, 3, 0);
		write_exr((global_cfg.m_output_image_prefix + "unpacked_google_astc.exr").c_str(), unpacked_astc_google_img, 3, 0);
	}

	// ASTC metrics
	if (global_cfg.m_image_stats)
	{
		image_metrics im;

		if (global_cfg.m_debug_output)
			printf("\nASTC log2 float error metrics:\n");

		for (uint32_t i = 0; i < 3; i++)
		{
			im.calc(enc_state.src_img, unpacked_astc_img, i, 1, true, true);

			if (global_cfg.m_debug_output)
			{
				printf("%c:   ", "RGBA"[i]);
				im.print_hp();
			}
		}
		
		metrics.m_im_astc_log2.calc(enc_state.src_img, unpacked_astc_img, 0, 3, true, true);

		if (global_cfg.m_debug_output)
		{
			printf("RGB: ");
			metrics.m_im_astc_log2.print_hp();

			printf("\n");
		}
	}

	if (global_cfg.m_image_stats)
	{
		image_metrics im;

		if (global_cfg.m_debug_output)
			printf("ASTC half float space error metrics (a piecewise linear approximation of log2 error):\n");

		for (uint32_t i = 0; i < 3; i++)
		{
			im.calc_half(enc_state.src_img, unpacked_astc_img, i, 1, true);

			if (global_cfg.m_debug_output)
			{
				printf("%c:   ", "RGBA"[i]);
				im.print_hp();
			}
		}

		metrics.m_im_astc_half.calc_half(enc_state.src_img, unpacked_astc_img, 0, 3, true);

		if (global_cfg.m_debug_output)
		{
			printf("RGB: ");
			metrics.m_im_astc_half.print_hp();
		}
	}

	// BC6H metrics
	if (global_cfg.m_image_stats)
	{
		image_metrics im;

		if (global_cfg.m_debug_output)
			printf("\nBC6H log2 float error metrics:\n");

		for (uint32_t i = 0; i < 3; i++)
		{
			im.calc(enc_state.src_img, unpacked_bc6h_img, i, 1, true, true);
			
			if (global_cfg.m_debug_output)
			{
				printf("%c:   ", "RGBA"[i]);
				im.print_hp();
			}
		}

		metrics.m_im_bc6h_log2.calc(enc_state.src_img, unpacked_bc6h_img, 0, 3, true, true);

		if (global_cfg.m_debug_output)
		{
			printf("RGB: ");
			metrics.m_im_bc6h_log2.print_hp();

			printf("\n");
		}
	}

	if (global_cfg.m_image_stats)
	{
		image_metrics im;
		
		if (global_cfg.m_debug_output)
			printf("BC6H half float space error metrics (a piecewise linear approximation of log2 error):\n");

		for (uint32_t i = 0; i < 3; i++)
		{
			im.calc_half(enc_state.src_img, unpacked_bc6h_img, i, 1, true);
			
			if (global_cfg.m_debug_output)
			{
				printf("%c:   ", "RGBA"[i]);
				im.print_hp();
			}
		}

		metrics.m_im_bc6h_half.calc_half(enc_state.src_img, unpacked_bc6h_img, 0, 3, true);
		
		if (global_cfg.m_debug_output)
		{
			printf("RGB: ");
			metrics.m_im_bc6h_half.print_hp();

			printf("\n");
		}
	}

	intermediate_tex_data.swap(coded_bits.get_bytes());

	astc_tex_data.resize(decoded_blocks1.size_in_bytes());
	memcpy(astc_tex_data.data(), decoded_blocks1.get_ptr(), decoded_blocks1.size_in_bytes());

	return true;
}

} // namespace astc_6x6_hdr
