// basisu_uastc_hdr_4x4_enc.cpp
#include "basisu_uastc_hdr_4x4_enc.h"
#include "../transcoder/basisu_transcoder.h"

using namespace basist;

namespace basisu
{

const uint32_t UHDR_MODE11_FIRST_ISE_RANGE = astc_helpers::BISE_3_LEVELS, UHDR_MODE11_LAST_ISE_RANGE = astc_helpers::BISE_16_LEVELS;
const uint32_t UHDR_MODE7_PART1_FIRST_ISE_RANGE = astc_helpers::BISE_3_LEVELS, UHDR_MODE7_PART1_LAST_ISE_RANGE = astc_helpers::BISE_16_LEVELS;
const uint32_t UHDR_MODE7_PART2_FIRST_ISE_RANGE = astc_helpers::BISE_3_LEVELS, UHDR_MODE7_PART2_LAST_ISE_RANGE = astc_helpers::BISE_8_LEVELS;
const uint32_t UHDR_MODE11_PART2_FIRST_ISE_RANGE = astc_helpers::BISE_3_LEVELS, UHDR_MODE11_PART2_LAST_ISE_RANGE = astc_helpers::BISE_4_LEVELS;

uastc_hdr_4x4_codec_options::uastc_hdr_4x4_codec_options() :
	astc_hdr_codec_base_options()
{
	init();
}

void uastc_hdr_4x4_codec_options::init()
{
	astc_hdr_codec_base_options::init();
	
	// This was the log bias we used on the initial release. It's too low.
	//m_q_log_bias = Q_LOG_BIAS_4x4;
		
	m_q_log_bias = Q_LOG_BIAS_6x6;

	m_bc6h_err_weight = .85f;

#if 0
	// HACK HACK
	m_disable_weight_plane_optimization = true;
	m_take_first_non_clamping_mode11_submode = false;
	m_take_first_non_clamping_mode7_submode = false;
#endif
				
	// Must set the quality level at least once to reset this struct.
	set_quality_level(cDefaultLevel);
}

void uastc_hdr_4x4_codec_options::set_quality_best()
{
	// highest achievable quality
	m_mode11_direct_only = false;

	m_use_solid = true;

	m_use_mode11_part1 = true;
	m_mode11_uber_mode = true;
	m_first_mode11_weight_ise_range = UHDR_MODE11_FIRST_ISE_RANGE;
	m_last_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE;
	m_first_mode11_submode = -1;
	m_last_mode11_submode = 7;

	m_use_mode7_part1 = true;
	m_first_mode7_part1_weight_ise_range = UHDR_MODE7_PART1_FIRST_ISE_RANGE;
	m_last_mode7_part1_weight_ise_range = UHDR_MODE7_PART1_LAST_ISE_RANGE;
	m_mode7_full_s_optimization = true;

	m_use_mode7_part2 = true;
	m_mode7_part2_part_masks = UINT32_MAX;
	m_first_mode7_part2_weight_ise_range = UHDR_MODE7_PART2_FIRST_ISE_RANGE;
	m_last_mode7_part2_weight_ise_range = UHDR_MODE7_PART2_LAST_ISE_RANGE;

	m_use_mode11_part2 = true;
	m_mode11_part2_part_masks = UINT32_MAX;
	m_first_mode11_part2_weight_ise_range = UHDR_MODE11_PART2_FIRST_ISE_RANGE;
	m_last_mode11_part2_weight_ise_range = UHDR_MODE11_PART2_LAST_ISE_RANGE;

	m_refine_weights = true;

	m_use_estimated_partitions = false;
	m_max_estimated_partitions = 0;
}

void uastc_hdr_4x4_codec_options::set_quality_normal()
{
	m_use_solid = true;

	// We'll allow uber mode in normal if the user allows it.
	m_use_mode11_part1 = true;
	m_mode11_uber_mode = true;
	m_first_mode11_weight_ise_range = 6;
	m_last_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE;

	m_use_mode7_part1 = true;
	m_first_mode7_part1_weight_ise_range = UHDR_MODE7_PART1_LAST_ISE_RANGE;
	m_last_mode7_part1_weight_ise_range = UHDR_MODE7_PART1_LAST_ISE_RANGE;

	m_use_mode7_part2 = true;
	m_mode7_part2_part_masks = UINT32_MAX;
	m_first_mode7_part2_weight_ise_range = UHDR_MODE7_PART2_LAST_ISE_RANGE;
	m_last_mode7_part2_weight_ise_range = UHDR_MODE7_PART2_LAST_ISE_RANGE;

	m_use_mode11_part2 = true;
	m_mode11_part2_part_masks = UINT32_MAX;
	m_first_mode11_part2_weight_ise_range = UHDR_MODE11_PART2_LAST_ISE_RANGE;
	m_last_mode11_part2_weight_ise_range = UHDR_MODE11_PART2_LAST_ISE_RANGE;

	m_refine_weights = true;
}

void uastc_hdr_4x4_codec_options::set_quality_fastest()
{
	m_use_solid = true;

	m_use_mode11_part1 = true;
	m_mode11_uber_mode = false;
	m_first_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE;
	m_last_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE;

	m_use_mode7_part1 = false;
	m_mode7_full_s_optimization = false;

	m_use_mode7_part2 = false;
	m_use_mode11_part2 = false;

	m_refine_weights = false;
}

void uastc_hdr_4x4_codec_options::set_quality_level(int level)
{
	level = clamp(level, cMinLevel, cMaxLevel);

	m_level = level;

	// First ensure all options are set to best.
	set_quality_best();

	switch (level)
	{
	case 0:
	{
		set_quality_fastest();
		break;
	}
	case 1:
	{
		set_quality_normal();

		m_first_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE - 1;
		m_last_mode11_weight_ise_range = UHDR_MODE11_LAST_ISE_RANGE;

		m_use_mode7_part1 = false;
		m_mode7_full_s_optimization = false;
		m_use_mode7_part2 = false;

		m_use_estimated_partitions = true;
		m_max_estimated_partitions = 1;

		m_mode11_part2_part_masks = 1 | 2;
		m_mode7_part2_part_masks = 1 | 2;

		// TODO: Disabling this hurts BC6H quality, but significantly speeds up compression.
		//m_refine_weights = false;
		break;
	}
	case 2:
	{
		set_quality_normal();

		m_use_estimated_partitions = true;
		m_max_estimated_partitions = 2;

		m_mode11_part2_part_masks = 1 | 2;
		m_mode7_part2_part_masks = 1 | 2;

		break;
	}
	case 3:
	{
		m_use_estimated_partitions = true;
		m_max_estimated_partitions = 2;

		m_mode11_part2_part_masks = 1 | 2 | 4 | 8;
		m_mode7_part2_part_masks = 1 | 2 | 4 | 8;

		break;
	}
	default:
	{
		// best options already set
		break;
	}
	}
}

//--------------------------------------------------------------------------------------------------------------------------

static bool pack_solid(const vec4F* pBlock_linear_colors, basisu::vector<astc_hdr_4x4_pack_results>& all_results, const uastc_hdr_4x4_codec_options& coptions)
{
	float r = 0.0f, g = 0.0f, b = 0.0f;

	const float LOG_BIAS = .125f;

	bool solid_block = true;
	for (uint32_t i = 0; i < 16; i++)
	{
		if ((pBlock_linear_colors[0][0] != pBlock_linear_colors[i][0]) ||
			(pBlock_linear_colors[0][1] != pBlock_linear_colors[i][1]) ||
			(pBlock_linear_colors[0][2] != pBlock_linear_colors[i][2]))
		{
			solid_block = false;
		}

		r += log2f(pBlock_linear_colors[i][0] + LOG_BIAS);
		g += log2f(pBlock_linear_colors[i][1] + LOG_BIAS);
		b += log2f(pBlock_linear_colors[i][2] + LOG_BIAS);
	}

	if (solid_block)
	{
		r = pBlock_linear_colors[0][0];
		g = pBlock_linear_colors[0][1];
		b = pBlock_linear_colors[0][2];
	}
	else
	{
		r = maximum<float>(0.0f, powf(2.0f, r * (1.0f / 16.0f)) - LOG_BIAS);
		g = maximum<float>(0.0f, powf(2.0f, g * (1.0f / 16.0f)) - LOG_BIAS);
		b = maximum<float>(0.0f, powf(2.0f, b * (1.0f / 16.0f)) - LOG_BIAS);

		// for safety
		r = minimum<float>(r, MAX_HALF_FLOAT);
		g = minimum<float>(g, MAX_HALF_FLOAT);
		b = minimum<float>(b, MAX_HALF_FLOAT);
	}

	half_float rh = float_to_half_non_neg_no_nan_inf(r), gh = float_to_half_non_neg_no_nan_inf(g), bh = float_to_half_non_neg_no_nan_inf(b), ah = float_to_half_non_neg_no_nan_inf(1.0f);

	astc_hdr_4x4_pack_results results;
	results.clear();

	uint8_t* packed_blk = (uint8_t*)&results.m_solid_blk;
	results.m_is_solid = true;

	packed_blk[0] = 0b11111100;
	packed_blk[1] = 255;
	packed_blk[2] = 255;
	packed_blk[3] = 255;
	packed_blk[4] = 255;
	packed_blk[5] = 255;
	packed_blk[6] = 255;
	packed_blk[7] = 255;

	packed_blk[8] = (uint8_t)rh;
	packed_blk[9] = (uint8_t)(rh >> 8);
	packed_blk[10] = (uint8_t)gh;
	packed_blk[11] = (uint8_t)(gh >> 8);
	packed_blk[12] = (uint8_t)bh;
	packed_blk[13] = (uint8_t)(bh >> 8);
	packed_blk[14] = (uint8_t)ah;
	packed_blk[15] = (uint8_t)(ah >> 8);

	results.m_best_block_error = 0;

	if (!solid_block)
	{
		const float R_WEIGHT = coptions.m_r_err_scale;
		const float G_WEIGHT = coptions.m_g_err_scale;

		// This MUST match how errors are computed in eval_selectors().
		for (uint32_t i = 0; i < 16; i++)
		{
			half_float dr = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][0]), dg = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][1]), db = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][2]);
			double rd = q(rh, Q_LOG_BIAS_4x4) - q(dr, Q_LOG_BIAS_4x4);
			double gd = q(gh, Q_LOG_BIAS_4x4) - q(dg, Q_LOG_BIAS_4x4);
			double bd = q(bh, Q_LOG_BIAS_4x4) - q(db, Q_LOG_BIAS_4x4);

			double e = R_WEIGHT * (rd * rd) + G_WEIGHT * (gd * gd) + bd * bd;

			results.m_best_block_error += e;
		}
	}

	const half_float hc[3] = { rh, gh, bh };

	bc6h_enc_block_solid_color(&results.m_bc6h_block, hc);

	all_results.push_back(results);

	return solid_block;
}

//--------------------------------------------------------------------------------------------------------------------------

static void pack_mode11(
	const vec4F* pBlock_linear_colors, const half_float pBlock_pixels_half[16][3], const vec4F pBlock_pixels_q16[16],
	basisu::vector<astc_hdr_4x4_pack_results>& all_results,
	const uastc_hdr_4x4_codec_options& coptions,
	uint32_t first_weight_ise_range, uint32_t last_weight_ise_range, bool constrain_ise_weight_selectors)
{
	BASISU_NOTE_UNUSED(pBlock_linear_colors);
	assert(first_weight_ise_range <= last_weight_ise_range);

	uint8_t trial_endpoints[NUM_MODE11_ENDPOINTS], trial_weights[16];
	uint32_t trial_submode11 = 0;

	clear_obj(trial_endpoints);
	clear_obj(trial_weights);
		
	for (uint32_t weight_ise_range = first_weight_ise_range; weight_ise_range <= last_weight_ise_range; weight_ise_range++)
	{
		const bool direct_only = coptions.m_mode11_direct_only;
		
		uint32_t endpoint_ise_range = astc_helpers::BISE_256_LEVELS;
		if (weight_ise_range == astc_helpers::BISE_16_LEVELS)
			endpoint_ise_range = astc_helpers::BISE_192_LEVELS;
		else
		{
			assert(weight_ise_range < astc_helpers::BISE_16_LEVELS);
		}
				
		double trial_error = encode_astc_hdr_block_mode_11(16, pBlock_pixels_half, pBlock_pixels_q16, weight_ise_range, trial_submode11, BIG_FLOAT_VAL, trial_endpoints, trial_weights, coptions, direct_only,
			endpoint_ise_range, coptions.m_mode11_uber_mode && (weight_ise_range >= astc_helpers::BISE_4_LEVELS) && coptions.m_allow_uber_mode, constrain_ise_weight_selectors, coptions.m_first_mode11_submode, coptions.m_last_mode11_submode, false, cOrdinaryLeastSquares);

		if (trial_error < BIG_FLOAT_VAL)
		{
			astc_hdr_4x4_pack_results results;
			results.clear();

			results.m_best_block_error = trial_error;

			results.m_best_submodes[0] = trial_submode11;
			results.m_constrained_weights = constrain_ise_weight_selectors;
						
			results.m_best_blk.m_num_partitions = 1;
			results.m_best_blk.m_color_endpoint_modes[0] = 11;
			results.m_best_blk.m_weight_ise_range = (uint8_t)weight_ise_range;
			results.m_best_blk.m_endpoint_ise_range = (uint8_t)endpoint_ise_range;
			
			memcpy(results.m_best_blk.m_endpoints, trial_endpoints, NUM_MODE11_ENDPOINTS);
			memcpy(results.m_best_blk.m_weights, trial_weights, 16);

#ifdef _DEBUG
			// Sanity checking
			{
				half_float block_pixels_half[16][3];
								
				for (uint32_t i = 0; i < 16; i++)
				{
					block_pixels_half[i][0] = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][0]);
					block_pixels_half[i][1] = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][1]);
					block_pixels_half[i][2] = float_to_half_non_neg_no_nan_inf(pBlock_linear_colors[i][2]);
				}
				
				half_float unpacked_astc_blk_rgba[4][4][4];
				bool res = astc_helpers::decode_block(results.m_best_blk, unpacked_astc_blk_rgba, 4, 4, astc_helpers::cDecodeModeHDR16);
				assert(res);

				half_float unpacked_astc_blk_rgb[4][4][3];
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
						for (uint32_t c = 0; c < 3; c++)
							unpacked_astc_blk_rgb[y][x][c] = unpacked_astc_blk_rgba[y][x][c];

				double cmp_err = compute_block_error(16, &block_pixels_half[0][0], &unpacked_astc_blk_rgb[0][0][0], coptions);
				assert(results.m_best_block_error == cmp_err);
			}
#endif

			// transcode to BC6H
			assert(results.m_best_blk.m_color_endpoint_modes[0] == 11);
			
			// Get qlog12 endpoints
			int e[2][3];
			bool success = decode_mode11_to_qlog12(results.m_best_blk.m_endpoints, e, results.m_best_blk.m_endpoint_ise_range);
			assert(success);
			BASISU_NOTE_UNUSED(success);

			// Transform endpoints to half float
			half_float h_e[3][2] =
			{
				{ qlog_to_half(e[0][0], 12), qlog_to_half(e[1][0], 12) },
				{ qlog_to_half(e[0][1], 12), qlog_to_half(e[1][1], 12) },
				{ qlog_to_half(e[0][2], 12), qlog_to_half(e[1][2], 12) }
			};

			// Transcode to bc6h
			success = transcode_bc6h_1subset(h_e, results.m_best_blk, results.m_bc6h_block);
			assert(success);

			all_results.push_back(results);
		}
	}
}

//--------------------------------------------------------------------------------------------------------------------------

static void pack_mode7_single_part(
	const half_float pBlock_pixels_half[16][3], const vec4F pBlock_pixels_q16[16],
	basisu::vector<astc_hdr_4x4_pack_results>& all_results, const uastc_hdr_4x4_codec_options& coptions,
	uint32_t first_mode7_part1_weight_ise_range, uint32_t last_mode7_part1_weight_ise_range)
{
	assert(first_mode7_part1_weight_ise_range <= last_mode7_part1_weight_ise_range);

	uint8_t trial_endpoints[NUM_MODE7_ENDPOINTS], trial_weights[16];
	uint32_t trial_submode7 = 0;

	clear_obj(trial_endpoints);
	clear_obj(trial_weights);

	for (uint32_t weight_ise_range = first_mode7_part1_weight_ise_range; weight_ise_range <= last_mode7_part1_weight_ise_range; weight_ise_range++)
	{
		const uint32_t ise_endpoint_range = astc_helpers::BISE_256_LEVELS;

		double trial_error = encode_astc_hdr_block_mode_7(16, pBlock_pixels_half, pBlock_pixels_q16, weight_ise_range, trial_submode7, BIG_FLOAT_VAL, trial_endpoints, trial_weights, coptions, ise_endpoint_range);

		if (trial_error < BIG_FLOAT_VAL)
		{
			astc_hdr_4x4_pack_results results;
			results.clear();

			results.m_best_block_error = trial_error;

			results.m_best_submodes[0] = trial_submode7;
			
			results.m_best_blk.m_num_partitions = 1;
			results.m_best_blk.m_color_endpoint_modes[0] = 7;
			results.m_best_blk.m_weight_ise_range = (uint8_t)weight_ise_range;
			results.m_best_blk.m_endpoint_ise_range = (uint8_t)ise_endpoint_range;
			
			memcpy(results.m_best_blk.m_endpoints, trial_endpoints, NUM_MODE7_ENDPOINTS);
			memcpy(results.m_best_blk.m_weights, trial_weights, 16);

			// transcode to BC6H
			assert(results.m_best_blk.m_color_endpoint_modes[0] == 7);
			
			// Get qlog12 endpoints
			int e[2][3];
			if (!decode_mode7_to_qlog12(results.m_best_blk.m_endpoints, e, nullptr, results.m_best_blk.m_endpoint_ise_range))
				continue;

			// Transform endpoints to half float
			half_float h_e[3][2] =
			{
				{ qlog_to_half(e[0][0], 12), qlog_to_half(e[1][0], 12) },
				{ qlog_to_half(e[0][1], 12), qlog_to_half(e[1][1], 12) },
				{ qlog_to_half(e[0][2], 12), qlog_to_half(e[1][2], 12) }
			};

			// Transcode to bc6h
			bool status = transcode_bc6h_1subset(h_e, results.m_best_blk, results.m_bc6h_block);
			assert(status);
			(void)status;

			all_results.push_back(results);
		}
	}
}

//--------------------------------------------------------------------------------------------------------------------------

static bool estimate_partition(
	const half_float pBlock_pixels_half[16][3],
	int* pBest_parts, uint32_t num_best_parts)
{
	assert(num_best_parts <= basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2);

	vec3F training_vecs[16], mean(0.0f);

	for (uint32_t i = 0; i < 16; i++)
	{
		vec3F& v = training_vecs[i];

		v[0] = (float)pBlock_pixels_half[i][0];
		v[1] = (float)pBlock_pixels_half[i][1];
		v[2] = (float)pBlock_pixels_half[i][2];

		mean += v;
	}
	mean *= (1.0f / 16.0f);

	vec3F cluster_centroids[2] = { mean - vec3F(.1f), mean + vec3F(.1f) };

	uint32_t cluster_pixels[2][16];
	uint32_t num_cluster_pixels[2];
	vec3F new_cluster_means[2];

	for (uint32_t s = 0; s < 4; s++)
	{
		num_cluster_pixels[0] = 0;
		num_cluster_pixels[1] = 0;

		new_cluster_means[0].clear();
		new_cluster_means[1].clear();

		for (uint32_t i = 0; i < 16; i++)
		{
			float d0 = training_vecs[i].squared_distance(cluster_centroids[0]);
			float d1 = training_vecs[i].squared_distance(cluster_centroids[1]);

			if (d0 < d1)
			{
				cluster_pixels[0][num_cluster_pixels[0]] = i;
				new_cluster_means[0] += training_vecs[i];
				num_cluster_pixels[0]++;
			}
			else
			{
				cluster_pixels[1][num_cluster_pixels[1]] = i;
				new_cluster_means[1] += training_vecs[i];
				num_cluster_pixels[1]++;
			}
		}

		if (!num_cluster_pixels[0] || !num_cluster_pixels[1])
			return false;

		cluster_centroids[0] = new_cluster_means[0] / (float)num_cluster_pixels[0];
		cluster_centroids[1] = new_cluster_means[1] / (float)num_cluster_pixels[1];
	}

	int desired_parts[4][4]; // [y][x]
	for (uint32_t p = 0; p < 2; p++)
	{
		for (uint32_t i = 0; i < num_cluster_pixels[p]; i++)
		{
			const uint32_t pix_index = cluster_pixels[p][i];

			desired_parts[pix_index >> 2][pix_index & 3] = p;
		}
	}

	uint32_t part_similarity[basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2];

	for (uint32_t part_index = 0; part_index < basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2; part_index++)
	{
		const uint32_t bc7_pattern = basist::g_astc_bc7_common_partitions2[part_index].m_bc7;

		int total_sim_non_inv = 0;
		int total_sim_inv = 0;

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				int part = basist::g_bc7_partition2[16 * bc7_pattern + x + y * 4];

				if (part == desired_parts[y][x])
					total_sim_non_inv++;

				if ((part ^ 1) == desired_parts[y][x])
					total_sim_inv++;
			}
		}

		int total_sim = maximum(total_sim_non_inv, total_sim_inv);

		part_similarity[part_index] = (total_sim << 8) | part_index;

	} // part_index;

	std::sort(part_similarity, part_similarity + basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2);

	for (uint32_t i = 0; i < num_best_parts; i++)
		pBest_parts[i] = part_similarity[(basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2 - 1) - i] & 0xFF;

	return true;
}

//--------------------------------------------------------------------------------------------------------------------------

static void pack_mode7_2part(
	const half_float pBlock_pixels_half[16][3], const vec4F pBlock_pixels_q16[16],
	basisu::vector<astc_hdr_4x4_pack_results>& all_results, const uastc_hdr_4x4_codec_options& coptions,
	int num_estimated_partitions, const int *pEstimated_partitions,
	uint32_t first_weight_ise_range, uint32_t last_weight_ise_range)
{
	assert(coptions.m_mode7_part2_part_masks);

	astc_helpers::log_astc_block trial_blk;
	clear_obj(trial_blk);
	trial_blk.m_grid_width = 4;
	trial_blk.m_grid_height = 4;

	trial_blk.m_num_partitions = 2;
	trial_blk.m_color_endpoint_modes[0] = 7;
	trial_blk.m_color_endpoint_modes[1] = 7;

	uint32_t first_part_index = 0, last_part_index = basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2;
		
	if (num_estimated_partitions)
	{
		first_part_index = 0;
		last_part_index = num_estimated_partitions;
	}
	
	for (uint32_t part_index_iter = first_part_index; part_index_iter < last_part_index; ++part_index_iter)
	{
		uint32_t part_index;
		if (num_estimated_partitions)
		{
			part_index = pEstimated_partitions[part_index_iter];
			assert(part_index < basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2);
		}
		else
		{
			part_index = part_index_iter;
			if (((1U << part_index) & coptions.m_mode7_part2_part_masks) == 0)
				continue;
		}
								
		const uint32_t astc_pattern = basist::g_astc_bc7_common_partitions2[part_index].m_astc;
		const uint32_t bc7_pattern = basist::g_astc_bc7_common_partitions2[part_index].m_bc7;
		const bool invert_flag = basist::g_astc_bc7_common_partitions2[part_index].m_invert;
				
		half_float part_pixels_half[2][16][3];
		vec4F part_pixels_q16[2][16];

		uint32_t pixel_part_index[4][4]; // [y][x]
		uint32_t num_part_pixels[2] = { 0, 0 };

		// Extract each subset's texels for this partition pattern
		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				uint32_t part = basist::g_bc7_partition2[16 * bc7_pattern + x + y * 4];
				if (invert_flag)
					part = 1 - part;

				pixel_part_index[y][x] = part;
								
				const uint32_t n = num_part_pixels[part];

				part_pixels_half[part][n][0] = pBlock_pixels_half[x + y * 4][0];
				part_pixels_half[part][n][1] = pBlock_pixels_half[x + y * 4][1];
				part_pixels_half[part][n][2] = pBlock_pixels_half[x + y * 4][2];
				part_pixels_q16[part][n] = pBlock_pixels_q16[x + y * 4];

				num_part_pixels[part] = n + 1;
			}
		}

		trial_blk.m_partition_id = (uint16_t)astc_pattern;
				
		for (uint32_t weight_ise_range = first_weight_ise_range; weight_ise_range <= last_weight_ise_range; weight_ise_range++)
		{
			assert(weight_ise_range <= astc_helpers::BISE_8_LEVELS);

			uint32_t ise_endpoint_range = astc_helpers::BISE_256_LEVELS;
			if (weight_ise_range == astc_helpers::BISE_5_LEVELS)
				ise_endpoint_range = astc_helpers::BISE_192_LEVELS;
			else if (weight_ise_range == astc_helpers::BISE_6_LEVELS)
				ise_endpoint_range = astc_helpers::BISE_128_LEVELS;
			else if (weight_ise_range == astc_helpers::BISE_8_LEVELS)
				ise_endpoint_range = astc_helpers::BISE_80_LEVELS;

			uint8_t trial_endpoints[2][NUM_MODE7_ENDPOINTS], trial_weights[2][16];
			uint32_t trial_submode7[2];

			clear_obj(trial_endpoints);
			clear_obj(trial_weights);
			clear_obj(trial_submode7);

			double total_trial_err = 0;
			for (uint32_t pack_part_index = 0; pack_part_index < 2; pack_part_index++)
			{
				total_trial_err += encode_astc_hdr_block_mode_7(
					num_part_pixels[pack_part_index], part_pixels_half[pack_part_index], part_pixels_q16[pack_part_index],
					weight_ise_range, trial_submode7[pack_part_index], BIG_FLOAT_VAL,
					&trial_endpoints[pack_part_index][0], &trial_weights[pack_part_index][0], coptions, ise_endpoint_range);

			} // pack_part_index

			if (total_trial_err < BIG_FLOAT_VAL)
			{
				trial_blk.m_weight_ise_range = (uint8_t)weight_ise_range;
				trial_blk.m_endpoint_ise_range = (uint8_t)ise_endpoint_range;

				for (uint32_t pack_part_index = 0; pack_part_index < 2; pack_part_index++)
					memcpy(&trial_blk.m_endpoints[pack_part_index * NUM_MODE7_ENDPOINTS], &trial_endpoints[pack_part_index][0], NUM_MODE7_ENDPOINTS);

				uint32_t src_pixel_index[2] = { 0, 0 };
				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t p = pixel_part_index[y][x];
						trial_blk.m_weights[x + y * 4] = trial_weights[p][src_pixel_index[p]++];
					}
				}
								
				astc_hdr_4x4_pack_results results;
				results.clear();

				results.m_best_block_error = total_trial_err;
				results.m_best_submodes[0] = trial_submode7[0];
				results.m_best_submodes[1] = trial_submode7[1];
				results.m_best_pat_index = part_index;

				results.m_best_blk = trial_blk;

				bool status = transcode_bc6h_2subsets(part_index, results.m_best_blk, results.m_bc6h_block);
				assert(status);
				BASISU_NOTE_UNUSED(status);

				all_results.push_back(results);
			}

		} // weight_ise_range

	} // part_index
}

//--------------------------------------------------------------------------------------------------------------------------

static void pack_mode11_2part(
	const half_float pBlock_pixels_half[16][3], const vec4F pBlock_pixels_q16[16],
	basisu::vector<astc_hdr_4x4_pack_results>& all_results, const uastc_hdr_4x4_codec_options& coptions,
	int num_estimated_partitions, const int* pEstimated_partitions)
{
	assert(coptions.m_mode11_part2_part_masks);

	astc_helpers::log_astc_block trial_blk;
	clear_obj(trial_blk);
	trial_blk.m_grid_width = 4;
	trial_blk.m_grid_height = 4;

	trial_blk.m_num_partitions = 2;
	trial_blk.m_color_endpoint_modes[0] = 11;
	trial_blk.m_color_endpoint_modes[1] = 11;
			
	uint32_t first_part_index = 0, last_part_index = basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2;

	if (num_estimated_partitions)
	{
		first_part_index = 0;
		last_part_index = num_estimated_partitions;
	}

	for (uint32_t part_index_iter = first_part_index; part_index_iter < last_part_index; ++part_index_iter)
	{
		uint32_t part_index;
		if (num_estimated_partitions)
		{
			part_index = pEstimated_partitions[part_index_iter];
			assert(part_index < basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2);
		}
		else
		{
			part_index = part_index_iter;
			if (((1U << part_index) & coptions.m_mode11_part2_part_masks) == 0)
				continue;
		}

		const uint32_t astc_pattern = basist::g_astc_bc7_common_partitions2[part_index].m_astc;
		const uint32_t bc7_pattern = basist::g_astc_bc7_common_partitions2[part_index].m_bc7;
		const bool invert_flag = basist::g_astc_bc7_common_partitions2[part_index].m_invert;

		half_float part_pixels_half[2][16][3];
		vec4F part_pixels_q16[2][16];

		uint32_t pixel_part_index[4][4]; // [y][x]
		uint32_t num_part_pixels[2] = { 0, 0 };

		// Extract each subset's texels for this partition pattern
		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				uint32_t part = basist::g_bc7_partition2[16 * bc7_pattern + x + y * 4];
				if (invert_flag)
					part = 1 - part;

				pixel_part_index[y][x] = part;
				
				const uint32_t n = num_part_pixels[part];

				part_pixels_half[part][n][0] = pBlock_pixels_half[x + y * 4][0];
				part_pixels_half[part][n][1] = pBlock_pixels_half[x + y * 4][1];
				part_pixels_half[part][n][2] = pBlock_pixels_half[x + y * 4][2];
				part_pixels_q16[part][n] = pBlock_pixels_q16[x + y * 4];

				num_part_pixels[part] = n + 1;
			}
		}
				
		trial_blk.m_partition_id = (uint16_t)astc_pattern;
						
		for (uint32_t weight_ise_range = coptions.m_first_mode11_part2_weight_ise_range; weight_ise_range <= coptions.m_last_mode11_part2_weight_ise_range; weight_ise_range++)
		{
			bool direct_only = false;
			uint32_t ise_endpoint_range = astc_helpers::BISE_64_LEVELS;
			if (weight_ise_range == astc_helpers::BISE_4_LEVELS)
				ise_endpoint_range = astc_helpers::BISE_40_LEVELS;

			uint8_t trial_endpoints[2][NUM_MODE11_ENDPOINTS], trial_weights[2][16];
			uint32_t trial_submode11[2];

			clear_obj(trial_endpoints); 
			clear_obj(trial_weights);
			clear_obj(trial_submode11);

			double total_trial_err = 0;
			for (uint32_t pack_part_index = 0; pack_part_index < 2; pack_part_index++)
			{
				total_trial_err += encode_astc_hdr_block_mode_11(
					num_part_pixels[pack_part_index], part_pixels_half[pack_part_index], part_pixels_q16[pack_part_index],
					weight_ise_range, trial_submode11[pack_part_index], BIG_FLOAT_VAL,
					&trial_endpoints[pack_part_index][0], &trial_weights[pack_part_index][0], coptions,
					direct_only, ise_endpoint_range, coptions.m_mode11_uber_mode && (weight_ise_range >= astc_helpers::BISE_4_LEVELS) && coptions.m_allow_uber_mode, false,
					coptions.m_first_mode11_submode, coptions.m_last_mode11_submode, false, cOrdinaryLeastSquares);

			} // pack_part_index

			if (total_trial_err < BIG_FLOAT_VAL)
			{
				trial_blk.m_weight_ise_range = (uint8_t)weight_ise_range;
				trial_blk.m_endpoint_ise_range = (uint8_t)ise_endpoint_range;

				for (uint32_t pack_part_index = 0; pack_part_index < 2; pack_part_index++)
					memcpy(&trial_blk.m_endpoints[pack_part_index * NUM_MODE11_ENDPOINTS], &trial_endpoints[pack_part_index][0], NUM_MODE11_ENDPOINTS);

				uint32_t src_pixel_index[2] = { 0, 0 };
				for (uint32_t y = 0; y < 4; y++)
				{
					for (uint32_t x = 0; x < 4; x++)
					{
						uint32_t p = pixel_part_index[y][x];
						trial_blk.m_weights[x + y * 4] = trial_weights[p][src_pixel_index[p]++];
					}
				}
								
				astc_hdr_4x4_pack_results results;
				results.clear();

				results.m_best_block_error = total_trial_err;
				results.m_best_submodes[0] = trial_submode11[0];
				results.m_best_submodes[1] = trial_submode11[1];
				results.m_best_pat_index = part_index;

				results.m_best_blk = trial_blk;

				bool status = transcode_bc6h_2subsets(part_index, results.m_best_blk, results.m_bc6h_block);
				assert(status);
				BASISU_NOTE_UNUSED(status);

				all_results.push_back(results);
			}

		} // weight_ise_range

	} // part_index
}

bool astc_hdr_4x4_enc_block(
	const float* pRGBPixels, const basist::half_float *pRGBPixelsHalf,
	const uastc_hdr_4x4_codec_options& coptions,
	basisu::vector<astc_hdr_4x4_pack_results>& all_results)
{
	assert(g_astc_hdr_enc_initialized);
	if (!g_astc_hdr_enc_initialized)
	{
		// astc_hdr_enc_init() MUST be called first.
		assert(0);
		return false;
	}

	assert(coptions.m_use_solid || coptions.m_use_mode11_part1 || coptions.m_use_mode7_part2 || coptions.m_use_mode7_part1 || coptions.m_use_mode11_part2);

	all_results.resize(0);

	const half_float (*pBlock_pixels_half)[16][3] = reinterpret_cast<const half_float(*)[16][3]>(pRGBPixelsHalf);
				
	vec4F block_linear_colors[16];
	vec4F block_pixels_q16[16];

	bool is_greyscale = true;
		
	for (uint32_t i = 0; i < 16; i++)
	{
		const float fr = pRGBPixels[i * 3 + 0], fg = pRGBPixels[i * 3 + 1], fb = pRGBPixels[i * 3 + 2];

		// Sanity check the input block.
		assert((fr >= 0) && (fr <= MAX_HALF_FLOAT) && (!std::isinf(fr)) && (!std::isnan(fr)));
		assert((fg >= 0) && (fg <= MAX_HALF_FLOAT) && (!std::isinf(fg)) && (!std::isnan(fg)));
		assert((fb >= 0) && (fb <= MAX_HALF_FLOAT) && (!std::isinf(fb)) && (!std::isnan(fb)));

		block_linear_colors[i].set(fr, fg, fb, 1.0f);

		const half_float hr = (*pBlock_pixels_half)[i][0];
		assert(hr == basist::float_to_half(fr));
		block_pixels_q16[i][0] = (float)half_to_qlog16(hr);

		const half_float hg = (*pBlock_pixels_half)[i][1];
		assert(hg == basist::float_to_half(fg));
		block_pixels_q16[i][1] = (float)half_to_qlog16(hg);

		const half_float hb = (*pBlock_pixels_half)[i][2];
		assert(hb == basist::float_to_half(fb));
		block_pixels_q16[i][2] = (float)half_to_qlog16(hb);
		
		block_pixels_q16[i][3] = 0.0f;

		if ((hr != hg) || (hr != hb))
			is_greyscale = false;
	} // i
							
	bool is_solid = false;
	if (coptions.m_use_solid)
		is_solid = pack_solid(block_linear_colors, all_results, coptions);

	if (!is_solid)
	{
		if ((is_greyscale) && (coptions.m_level == 0))
		{
			// Special case if it's a pure grayscale block - just try mode 7.
			pack_mode7_single_part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, 1, 1);
			pack_mode7_single_part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, UHDR_MODE7_PART1_LAST_ISE_RANGE, UHDR_MODE7_PART1_LAST_ISE_RANGE);
		}
		else
		{
			if (coptions.m_use_mode11_part1)
			{
				const size_t cur_num_results = all_results.size();

				pack_mode11(block_linear_colors, *pBlock_pixels_half, block_pixels_q16, all_results, coptions, coptions.m_first_mode11_weight_ise_range, coptions.m_last_mode11_weight_ise_range, false);
								
				if (coptions.m_last_mode11_weight_ise_range >= astc_helpers::BISE_12_LEVELS)
				{
					// Try constrained weights if we're allowed to use 12/16 level ISE weight modes
					pack_mode11(block_linear_colors, *pBlock_pixels_half, block_pixels_q16, all_results, coptions, maximum<uint32_t>(coptions.m_first_mode11_weight_ise_range, astc_helpers::BISE_12_LEVELS), coptions.m_last_mode11_weight_ise_range, true);
				}

				// If we couldn't get any mode 11 results at all, and we were restricted to just trying weight ISE range 8 (which required endpoint quantization) then 
				// fall back to weight ISE range 7 (which doesn't need any endpoint quantization).
				// This is to guarantee we always get at least 1 non-solid result.
				if (all_results.size() == cur_num_results)
				{
					if (coptions.m_first_mode11_weight_ise_range == astc_helpers::BISE_16_LEVELS)
					{
						pack_mode11(block_linear_colors, *pBlock_pixels_half, block_pixels_q16, all_results, coptions, astc_helpers::BISE_12_LEVELS, astc_helpers::BISE_12_LEVELS, false);
					}
				}
			}

			if (coptions.m_use_mode7_part1)
			{
				// Mode 7 1-subset never requires endpoint quantization, so it cannot fail to find at least one usable solution.
				pack_mode7_single_part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, coptions.m_first_mode7_part1_weight_ise_range, coptions.m_last_mode7_part1_weight_ise_range);
			}
			else if (is_greyscale)
			{
				// Special case if it's a pure grayscale block and mode 7 was disabled - try it anyway, because mode 11 has worse B channel quantization.
				pack_mode7_single_part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, 1, 1);
				pack_mode7_single_part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, UHDR_MODE7_PART1_LAST_ISE_RANGE, UHDR_MODE7_PART1_LAST_ISE_RANGE);
			}
		}
						
		bool have_est = false;
		int best_parts[basist::TOTAL_ASTC_BC6H_COMMON_PARTITIONS2];

		if ((coptions.m_use_mode7_part2) || (coptions.m_use_mode11_part2))
		{
			if (coptions.m_use_estimated_partitions)
				have_est = estimate_partition(*pBlock_pixels_half, best_parts, coptions.m_max_estimated_partitions);
		}

		if (coptions.m_use_mode7_part2)
		{
			const size_t cur_num_results = all_results.size();

			pack_mode7_2part(*pBlock_pixels_half, block_pixels_q16,
				all_results, coptions, have_est ? coptions.m_max_estimated_partitions : 0, best_parts, 
				coptions.m_first_mode7_part2_weight_ise_range, coptions.m_last_mode7_part2_weight_ise_range);

			// If we couldn't find any packable 2-subset mode 7 results at weight levels >= 5 levels (which always requires endpoint quant), then try falling back to 
			// 5 levels which doesn't require endpoint quantization.
			if (all_results.size() == cur_num_results)
			{
				if (coptions.m_first_mode7_part2_weight_ise_range >= astc_helpers::BISE_5_LEVELS)
				{
					pack_mode7_2part(*pBlock_pixels_half, block_pixels_q16,
						all_results, coptions, have_est ? coptions.m_max_estimated_partitions : 0, best_parts, 
						astc_helpers::BISE_4_LEVELS, astc_helpers::BISE_4_LEVELS);
				}
			}
		}
		
		if (coptions.m_use_mode11_part2)
		{
			// This always requires endpoint quant, so it could fail to find any usable solutions.
			pack_mode11_2part(*pBlock_pixels_half, block_pixels_q16, all_results, coptions, have_est ? coptions.m_max_estimated_partitions : 0, best_parts);
		}

		if (coptions.m_refine_weights)
		{
			// TODO: This is quite slow.
			for (uint32_t i = 0; i < all_results.size(); i++)
			{
				bool status = astc_hdr_4x4_refine_weights(pRGBPixelsHalf, all_results[i], coptions, coptions.m_bc6h_err_weight, &all_results[i].m_improved_via_refinement_flag);
				assert(status);
				BASISU_NOTE_UNUSED(status);
			}
		}

	} // !is_solid

	return true;
}

bool astc_hdr_4x4_pack_results_to_block(astc_blk& dst_blk, const astc_hdr_4x4_pack_results& results)
{
	assert(g_astc_hdr_enc_initialized);
	if (!g_astc_hdr_enc_initialized)
		return false;

	if (results.m_is_solid)
	{
		memcpy(&dst_blk, &results.m_solid_blk, sizeof(results.m_solid_blk));
	}
	else
	{
		bool status = astc_helpers::pack_astc_block((astc_helpers::astc_block&)dst_blk, results.m_best_blk);
		if (!status)
		{
			assert(0);
			return false;
		}
	}

	return true;
}

// Refines a block's chosen weight indices, balancing BC6H and ASTC HDR error.
bool astc_hdr_4x4_refine_weights(const half_float *pSource_block, 
	astc_hdr_4x4_pack_results& cur_results, const uastc_hdr_4x4_codec_options& coptions, float bc6h_weight, bool *pImproved_flag)
{
	if (pImproved_flag)
		*pImproved_flag = false;

	if (cur_results.m_is_solid)
		return true;

	const uint32_t total_weights = astc_helpers::get_ise_levels(cur_results.m_best_blk.m_weight_ise_range);
	assert((total_weights >= MIN_SUPPORTED_WEIGHT_LEVELS) && (total_weights <= MAX_SUPPORTED_WEIGHT_LEVELS));

	double best_err[4][4];
	uint8_t best_weight[4][4];
	for (uint32_t y = 0; y < 4; y++)
	{
		for (uint32_t x = 0; x < 4; x++)
		{
			best_err[y][x] = BIG_FLOAT_VAL;
			best_weight[y][x] = 0;
		}
	}

	astc_hdr_4x4_pack_results temp_results;

	const float c_weights[3] = { coptions.m_r_err_scale, coptions.m_g_err_scale, 1.0f };

	for (uint32_t weight_index = 0; weight_index < total_weights; weight_index++)
	{
		temp_results = cur_results;
		for (uint32_t i = 0; i < 16; i++)
			temp_results.m_best_blk.m_weights[i] = (uint8_t)weight_index;
		
		half_float unpacked_astc_blk_rgba[4][4][4];
		bool res = astc_helpers::decode_block(temp_results.m_best_blk, unpacked_astc_blk_rgba, 4, 4, astc_helpers::cDecodeModeHDR16);
		assert(res);

		basist::bc6h_block trial_bc6h_blk;
		res = basist::astc_hdr_transcode_to_bc6h(temp_results.m_best_blk, trial_bc6h_blk);
		assert(res);
				
		half_float unpacked_bc6h_blk[4][4][3];
		res = unpack_bc6h(&trial_bc6h_blk, unpacked_bc6h_blk, false);
		assert(res);
		BASISU_NOTE_UNUSED(res);

		for (uint32_t y = 0; y < 4; y++)
		{
			for (uint32_t x = 0; x < 4; x++)
			{
				double total_err = 0.0f;

				for (uint32_t c = 0; c < 3; c++)
				{
					const half_float orig_c = pSource_block[(x + y * 4) * 3 + c];
					const double orig_c_q = q(orig_c, Q_LOG_BIAS_4x4);
					
					const half_float astc_c = unpacked_astc_blk_rgba[y][x][c];
					const double astc_c_q = q(astc_c, Q_LOG_BIAS_4x4);
					const double astc_e = square(astc_c_q - orig_c_q) * c_weights[c];
					
					const half_float bc6h_c = unpacked_bc6h_blk[y][x][c];
					const double bc6h_c_q = q(bc6h_c, Q_LOG_BIAS_4x4);
					const double bc6h_e = square(bc6h_c_q - orig_c_q) * c_weights[c];

					const double overall_err = astc_e * (1.0f - bc6h_weight) + bc6h_e * bc6h_weight;

					total_err += overall_err;

				} //  c

				if (total_err < best_err[y][x])
				{
					best_err[y][x] = total_err;
					best_weight[y][x] = (uint8_t)weight_index;
				}

			} // x
		} // y

	} // weight_index

	bool any_changed = false;
	for (uint32_t i = 0; i < 16; i++)
	{
		if (cur_results.m_best_blk.m_weights[i] != best_weight[i >> 2][i & 3])
		{
			any_changed = true;
			break;
		}
	}

	if (any_changed)
	{
		memcpy(cur_results.m_best_blk.m_weights, best_weight, 16);

		{
			bool res = basist::astc_hdr_transcode_to_bc6h(cur_results.m_best_blk, cur_results.m_bc6h_block);
			assert(res);
			BASISU_NOTE_UNUSED(res);

			half_float unpacked_astc_blk_rgba[4][4][4];
			res = astc_helpers::decode_block(cur_results.m_best_blk, unpacked_astc_blk_rgba, 4, 4, astc_helpers::cDecodeModeHDR16);
			assert(res);

			half_float unpacked_astc_blk_rgb[4][4][3];
			for (uint32_t y = 0; y < 4; y++)
				for (uint32_t x = 0; x < 4; x++)
					for (uint32_t c = 0; c < 3; c++)
						unpacked_astc_blk_rgb[y][x][c] = unpacked_astc_blk_rgba[y][x][c];

			cur_results.m_best_block_error = compute_block_error(16, pSource_block, &unpacked_astc_blk_rgb[0][0][0], coptions);
		}

		if (pImproved_flag)
			*pImproved_flag = true;
	}

	return true;
}

void astc_hdr_4x4_block_stats::update(const astc_hdr_4x4_pack_results& log_blk)
{
	std::lock_guard<std::mutex> lck(m_mutex);

	m_total_blocks++;

	if (log_blk.m_improved_via_refinement_flag)
		m_total_refined++;

	if (log_blk.m_is_solid)
	{
		m_total_solid++;
	}
	else
	{
		int best_weight_range = log_blk.m_best_blk.m_weight_ise_range;

		if (log_blk.m_best_blk.m_color_endpoint_modes[0] == 7)
		{
			m_mode7_submode_hist[bounds_check(log_blk.m_best_submodes[0], 0U, 6U)]++;

			if (log_blk.m_best_blk.m_num_partitions == 2)
			{
				m_total_mode7_2part++;

				m_mode7_submode_hist[bounds_check(log_blk.m_best_submodes[1], 0U, 6U)]++;
				m_total_2part++;

				m_weight_range_hist_7_2part[bounds_check(best_weight_range, 0, 11)]++;

				m_part_hist[bounds_check(log_blk.m_best_pat_index, 0U, 32U)]++;
			}
			else
			{
				m_total_mode7_1part++;

				m_weight_range_hist_7[bounds_check(best_weight_range, 0, 11)]++;
			}
		}
		else
		{
			m_mode11_submode_hist[bounds_check(log_blk.m_best_submodes[0], 0U, 9U)]++;
			if (log_blk.m_constrained_weights)
				m_total_mode11_1part_constrained_weights++;

			if (log_blk.m_best_blk.m_num_partitions == 2)
			{
				m_total_mode11_2part++;

				m_mode11_submode_hist[bounds_check(log_blk.m_best_submodes[1], 0U, 9U)]++;
				m_total_2part++;

				m_weight_range_hist_11_2part[bounds_check(best_weight_range, 0, 11)]++;

				m_part_hist[bounds_check(log_blk.m_best_pat_index, 0U, 32U)]++;
			}
			else
			{
				m_total_mode11_1part++;

				m_weight_range_hist_11[bounds_check(best_weight_range, 0, 11)]++;
			}
		}
	}
}

void astc_hdr_4x4_block_stats::print()
{
	std::lock_guard<std::mutex> lck(m_mutex);

	assert(m_total_blocks);
	if (!m_total_blocks)
		return;

	printf("\nLow-level ASTC Encoder Statistics:\n");
	printf("Total blocks: %u\n", m_total_blocks);
	printf("Total solid: %u %3.2f%%\n", m_total_solid, (m_total_solid * 100.0f) / m_total_blocks);
	printf("Total refined: %u %3.2f%%\n", m_total_refined, (m_total_refined * 100.0f) / m_total_blocks);

	printf("Total mode 11, 1 partition: %u %3.2f%%\n", m_total_mode11_1part, (m_total_mode11_1part * 100.0f) / m_total_blocks);
	printf("Total mode 11, 1 partition, constrained weights: %u %3.2f%%\n", m_total_mode11_1part_constrained_weights, (m_total_mode11_1part_constrained_weights * 100.0f) / m_total_blocks);
	printf("Total mode 11, 2 partition: %u %3.2f%%\n", m_total_mode11_2part, (m_total_mode11_2part * 100.0f) / m_total_blocks);

	printf("Total mode 7, 1 partition: %u %3.2f%%\n", m_total_mode7_1part, (m_total_mode7_1part * 100.0f) / m_total_blocks);
	printf("Total mode 7, 2 partition: %u %3.2f%%\n", m_total_mode7_2part, (m_total_mode7_2part * 100.0f) / m_total_blocks);

	printf("Total 2 partitions: %u %3.2f%%\n", m_total_2part, (m_total_2part * 100.0f) / m_total_blocks);
	printf("\n");

	printf("ISE texel weight range histogram mode 11:\n");
	for (uint32_t i = 1; i <= UHDR_MODE11_LAST_ISE_RANGE; i++)
		printf("%u %u\n", i, m_weight_range_hist_11[i]);
	printf("\n");

	printf("ISE texel weight range histogram mode 11, 2 partition:\n");
	for (uint32_t i = 1; i <= UHDR_MODE11_PART2_LAST_ISE_RANGE; i++)
		printf("%u %u\n", i, m_weight_range_hist_11_2part[i]);
	printf("\n");

	printf("ISE texel weight range histogram mode 7:\n");
	for (uint32_t i = 1; i <= UHDR_MODE7_PART1_LAST_ISE_RANGE; i++)
		printf("%u %u\n", i, m_weight_range_hist_7[i]);
	printf("\n");

	printf("ISE texel weight range histogram mode 7, 2 partition:\n");
	for (uint32_t i = 1; i <= UHDR_MODE7_PART2_LAST_ISE_RANGE; i++)
		printf("%u %u\n", i, m_weight_range_hist_7_2part[i]);
	printf("\n");

	printf("Mode 11 submode histogram:\n");
	for (uint32_t i = 0; i <= MODE11_TOTAL_SUBMODES; i++) // +1 because of the extra direct encoding
		printf("%u %u\n", i, m_mode11_submode_hist[i]);
	printf("\n");

	printf("Mode 7 submode histogram:\n");
	for (uint32_t i = 0; i < MODE7_TOTAL_SUBMODES; i++)
		printf("%u %u\n", i, m_mode7_submode_hist[i]);
	printf("\n");

	printf("Partition pattern table usage histogram:\n");
	for (uint32_t i = 0; i < basist::TOTAL_ASTC_BC7_COMMON_PARTITIONS2; i++)
		printf("%u:%u ", i, m_part_hist[i]);
	printf("\n\n");
}

} // namespace basisu

