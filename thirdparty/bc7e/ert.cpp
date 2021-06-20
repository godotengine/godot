#include "ert.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

#define ERT_FAVOR_CONT_AND_REP0_MATCHES (1)
#define ERT_FAVOR_REP0_MATCHES (0)

namespace ert
{
	const uint32_t MAX_BLOCK_PIXELS = 12 * 12;
	const uint32_t MAX_BLOCK_SIZE_IN_BYTES = 256;
	const uint32_t MIN_MATCH_LEN = 3;
	const float LITERAL_BITS = 13.0f;
	const float MATCH_CONTINUE_BITS = 1.0f;
	const float MATCH_REP0_BITS = 4.0f;

	static inline float clampf(float value, float low, float high) { if (value < low) value = low; else if (value > high) value = high;	return value; }
	template<typename F> inline F lerp(F a, F b, F s) { return a + (b - a) * s; }

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

	static inline float compute_block_max_std_dev(const color_rgba* pPixels, uint32_t block_width, uint32_t block_height, uint32_t num_comps)
	{
		tracked_stat comp_stats[4];

		for (uint32_t y = 0; y < block_height; y++)
		{
			for (uint32_t x = 0; x < block_width; x++)
			{
				const color_rgba* pPixel = pPixels + x + y * block_width;

				for (uint32_t c = 0; c < num_comps; c++)
					comp_stats[c].update(pPixel->m_c[c]);
			}
		}

		float max_std_dev = 0.0f;
		for (uint32_t i = 0; i < num_comps; i++)
			max_std_dev = std::max(max_std_dev, comp_stats[i].get_std_dev());
		return max_std_dev;
	}

	static inline float compute_block_mse(const color_rgba* pPixelsA, const color_rgba* pPixelsB, uint32_t block_width, uint32_t block_height, uint32_t total_block_pixels, uint32_t num_comps, const uint32_t weights[4], float one_over_total_color_weight)
	{
		uint64_t total_err = 0;

		if ((block_width == 4) && (block_height == 4) && (num_comps == 4))
		{
			if ((weights[0] == 1) && (weights[1] == 1) && (weights[2] == 1) && (weights[3] == 1))
			{
				for (uint32_t i = 0; i < 16; i++)
				{
					const color_rgba* pA = pPixelsA + i;
					const color_rgba* pB = pPixelsB + i;

					const int dr = pA->m_c[0] - pB->m_c[0];
					const int dg = pA->m_c[1] - pB->m_c[1];
					const int db = pA->m_c[2] - pB->m_c[2];
					const int da = pA->m_c[3] - pB->m_c[3];

					total_err += dr * dr + dg * dg + db * db + da * da;
				}
			}
			else
			{
				for (uint32_t i = 0; i < 16; i++)
				{
					const color_rgba* pA = pPixelsA + i;
					const color_rgba* pB = pPixelsB + i;

					const int dr = pA->m_c[0] - pB->m_c[0];
					const int dg = pA->m_c[1] - pB->m_c[1];
					const int db = pA->m_c[2] - pB->m_c[2];
					const int da = pA->m_c[3] - pB->m_c[3];

					total_err += weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db + weights[3] * da * da;
				}
			}
		}
		else if ((block_width == 4) && (block_height == 4) && (num_comps == 3))
		{
			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t y_ofs = y * 4;
				for (uint32_t x = 0; x < 4; x++)
				{
					const color_rgba* pA = pPixelsA + x + y_ofs;
					const color_rgba* pB = pPixelsB + x + y_ofs;

					const int dr = pA->m_c[0] - pB->m_c[0];
					const int dg = pA->m_c[1] - pB->m_c[1];
					const int db = pA->m_c[2] - pB->m_c[2];

					total_err += weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db;
				}
			}
		}
		else if ((block_width == 4) && (block_height == 4) && (num_comps == 2))
		{
			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t y_ofs = y * 4;
				for (uint32_t x = 0; x < 4; x++)
				{
					const color_rgba* pA = pPixelsA + x + y_ofs;
					const color_rgba* pB = pPixelsB + x + y_ofs;

					const int dr = pA->m_c[0] - pB->m_c[0];
					const int dg = pA->m_c[1] - pB->m_c[1];

					total_err += weights[0] * dr * dr + weights[1] * dg * dg;
				}
			}
		}
		else if ((block_width == 4) && (block_height == 4) && (num_comps == 1))
		{
			for (uint32_t y = 0; y < 4; y++)
			{
				const uint32_t y_ofs = y * 4;
				for (uint32_t x = 0; x < 4; x++)
				{
					const color_rgba* pA = pPixelsA + x + y_ofs;
					const color_rgba* pB = pPixelsB + x + y_ofs;

					const int dr = pA->m_c[0] - pB->m_c[0];

					total_err += weights[0] * dr * dr;
				}
			}
		}
		else
		{
			for (uint32_t y = 0; y < block_height; y++)
			{
				const uint32_t y_ofs = y * block_width;
				for (uint32_t x = 0; x < block_width; x++)
				{
					const color_rgba* pA = pPixelsA + x + y_ofs;
					const color_rgba* pB = pPixelsB + x + y_ofs;

					for (uint32_t c = 0; c < num_comps; c++)
					{
						const int d = pA->m_c[c] - pB->m_c[c];
						total_err += weights[c] * d * d;
					}
				}
			}
		}

		return total_err * (one_over_total_color_weight / total_block_pixels);
	}	

	uint32_t hash_hsieh(const uint8_t* pBuf, size_t len, uint32_t salt)
	{
		if (!pBuf || !len)
			return 0;

		uint32_t h = static_cast<uint32_t>(len + (salt << 16));

		const uint32_t bytes_left = len & 3;
		len >>= 2;

		while (len--)
		{
			const uint16_t* pWords = reinterpret_cast<const uint16_t*>(pBuf);

			h += pWords[0];

			const uint32_t t = (pWords[1] << 11) ^ h;
			h = (h << 16) ^ t;

			pBuf += sizeof(uint32_t);

			h += h >> 11;
		}

		switch (bytes_left)
		{
		case 1:
			h += *reinterpret_cast<const signed char*>(pBuf);
			h ^= h << 10;
			h += h >> 1;
			break;
		case 2:
			h += *reinterpret_cast<const uint16_t*>(pBuf);
			h ^= h << 11;
			h += h >> 17;
			break;
		case 3:
			h += *reinterpret_cast<const uint16_t*>(pBuf);
			h ^= h << 16;
			h ^= (static_cast<signed char>(pBuf[sizeof(uint16_t)])) << 18;
			h += h >> 11;
			break;
		default:
			break;
		}

		h ^= h << 3;
		h += h >> 5;
		h ^= h << 4;
		h += h >> 17;
		h ^= h << 25;
		h += h >> 6;

		return h;
	}

	// BC7 entropy reduction transform with Deflate/LZMA/LZHAM optimizations
	bool reduce_entropy(void* pBlocks, uint32_t num_blocks,
		uint32_t total_block_stride_in_bytes, uint32_t block_size_to_optimize_in_bytes, uint32_t block_width, uint32_t block_height, uint32_t num_comps,
		const color_rgba* pBlock_pixels, const reduce_entropy_params& params, uint32_t& total_modified,
		pUnpack_block_func pUnpack_block_func, void* pUnpack_block_func_user_data,
		std::vector<float>* pBlock_mse_scales)
	{
		assert(total_block_stride_in_bytes && block_size_to_optimize_in_bytes);
		assert(total_block_stride_in_bytes >= block_size_to_optimize_in_bytes);
		
		assert(num_comps >= 1 && num_comps <= 4);
		for (uint32_t i = num_comps; i < 4; i++)
		{
			assert(!params.m_color_weights[i]);
			if (params.m_color_weights[i])
				return false;
		}

		const uint32_t total_color_weight = params.m_color_weights[0] + params.m_color_weights[1] + params.m_color_weights[2] + params.m_color_weights[3];
		assert(total_color_weight);
		const float one_over_total_color_weight = 1.0f / total_color_weight;

		assert((block_size_to_optimize_in_bytes >= MIN_MATCH_LEN) && (block_size_to_optimize_in_bytes <= MAX_BLOCK_SIZE_IN_BYTES));
		if ((block_size_to_optimize_in_bytes < MIN_MATCH_LEN) || (block_size_to_optimize_in_bytes > MAX_BLOCK_SIZE_IN_BYTES))
			return false;

		uint8_t* pBlock_bytes = (uint8_t*)pBlocks;

		const uint32_t total_block_pixels = block_width * block_height;
		if (total_block_pixels > MAX_BLOCK_PIXELS)
			return false;

		const int total_blocks_to_check = std::max<uint32_t>(1U, params.m_lookback_window_size / total_block_stride_in_bytes);

		std::vector<uint32_t> len_hist(MAX_BLOCK_SIZE_IN_BYTES + 1);
		std::vector<uint32_t> second_len_hist(MAX_BLOCK_SIZE_IN_BYTES + 1);
		uint32_t total_second_matches = 0;

		int prev_match_window_ofs_to_favor_cont = -1, prev_match_dist_to_favor = -1;
				
		uint32_t total_smooth_blocks = 0;

		const uint32_t HASH_SIZE = 8192;
		uint32_t hash[HASH_SIZE];
				
		for (uint32_t block_index = 0; block_index < num_blocks; block_index++)
		{
			if ((block_index & 0xFF) == 0)
				memset(hash, 0, sizeof(hash));

			uint8_t* pOrig_block = &pBlock_bytes[block_index * total_block_stride_in_bytes];
			const color_rgba* pPixels = &pBlock_pixels[block_index * total_block_pixels];

			color_rgba decoded_block[MAX_BLOCK_PIXELS];
			if (!(*pUnpack_block_func)(pOrig_block, decoded_block, block_index, pUnpack_block_func_user_data))
				return false;

			float cur_mse = compute_block_mse(pPixels, decoded_block, block_width, block_height, total_block_pixels, num_comps, params.m_color_weights, one_over_total_color_weight);

			if ((params.m_skip_zero_mse_blocks) && (cur_mse == 0.0f))
				continue;

			const float max_std_dev = compute_block_max_std_dev(pPixels, block_width, block_height, num_comps);
			
			float yl = clampf(max_std_dev / params.m_max_smooth_block_std_dev, 0.0f, 1.0f);
			yl = yl * yl;
			float smooth_block_mse_scale = lerp(params.m_smooth_block_max_mse_scale, 1.0f, yl);

			if (pBlock_mse_scales)
			{
				if ((*pBlock_mse_scales)[block_index] > 0.0f)
				{
					smooth_block_mse_scale = (*pBlock_mse_scales)[block_index];
				}
			}
			
			if (smooth_block_mse_scale > 1.0f)
				total_smooth_blocks++;
						
			float cur_bits = (LITERAL_BITS * block_size_to_optimize_in_bytes);
			float cur_t = cur_mse * smooth_block_mse_scale + cur_bits * params.m_lambda;

			int first_block_to_check = std::max<int>(0, block_index - total_blocks_to_check);
			int last_block_to_check = block_index - 1;

			uint8_t best_block[MAX_BLOCK_SIZE_IN_BYTES];
			memcpy(best_block, pOrig_block, block_size_to_optimize_in_bytes);

			float best_t = cur_t;
			uint32_t best_match_len = 0, best_match_src_window_ofs = 0, best_match_dst_window_ofs = 0, best_match_src_block_ofs = 0, best_match_dst_block_ofs = 0;
			float best_match_bits = 0;

			// Don't let thresh_ms_err be 0 to let zero error blocks have slightly increased distortion
			const float thresh_ms_err = params.m_max_allowed_rms_increase_ratio * params.m_max_allowed_rms_increase_ratio * std::max(cur_mse, 1.0f);
			
			for (int prev_block_index = last_block_to_check; prev_block_index >= first_block_to_check; --prev_block_index)
			{
				const uint8_t* pPrev_blk = &pBlock_bytes[prev_block_index * total_block_stride_in_bytes];

				for (uint32_t len = block_size_to_optimize_in_bytes; len >= MIN_MATCH_LEN; len--)
				{
					if (params.m_allow_relative_movement)
					{
						for (uint32_t src_ofs = 0; src_ofs <= (block_size_to_optimize_in_bytes - len); src_ofs++)
						{
							assert(len + src_ofs <= block_size_to_optimize_in_bytes);
							
							const uint32_t src_match_window_ofs = prev_block_index * total_block_stride_in_bytes + src_ofs;

							for (uint32_t dst_ofs = 0; dst_ofs <= (block_size_to_optimize_in_bytes - len); dst_ofs++)
							{
								assert(len + dst_ofs <= block_size_to_optimize_in_bytes);
								
								const uint32_t dst_match_window_ofs = block_index * total_block_stride_in_bytes + dst_ofs;

								const uint32_t match_dist = dst_match_window_ofs - src_match_window_ofs;
																
								float trial_match_bits, trial_total_bits;

								uint32_t hs = hash_hsieh(pPrev_blk + src_ofs, len, dst_ofs);

#if ERT_FAVOR_CONT_AND_REP0_MATCHES
								// Continue a previous match (which would cross block boundaries)
								if (((int)src_match_window_ofs == prev_match_window_ofs_to_favor_cont) && (dst_ofs == 0))
								{
									trial_match_bits = MATCH_CONTINUE_BITS;
									trial_total_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + MATCH_CONTINUE_BITS;
								}
								// Exploit REP0 matches
								else if ((prev_match_dist_to_favor != -1) && (src_match_window_ofs == (dst_match_window_ofs - prev_match_dist_to_favor)))
								{
									trial_match_bits = MATCH_REP0_BITS;
									trial_total_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + MATCH_REP0_BITS;
								}
								else
								{
									trial_match_bits = (float)compute_match_cost_estimate(match_dist, len);
									trial_total_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + trial_match_bits;
										
									uint32_t hash_check = hash[hs & (HASH_SIZE - 1)];
									if ((hash_check & 0xFF) == (block_index & 0xFF))
									{
										if ((hash_check >> 8) == (hs >> 8))
											continue;
									}
								}
#else
								uint32_t hash_check = hash[hs & (HASH_SIZE - 1)];
								if ((hash_check & 0xFF) == (block_index & 0xFF))
								{
									if ((hash_check >> 8) == (hs >> 8))
										continue;
								}
#endif

								hash[hs & (HASH_SIZE - 1)] = (hs & 0xFFFFFF00) | (block_index & 0xFF);

								const float trial_total_bits_times_lambda = trial_total_bits * params.m_lambda;
								
								uint8_t trial_block[MAX_BLOCK_SIZE_IN_BYTES];
								memcpy(trial_block, pOrig_block, block_size_to_optimize_in_bytes);
								memcpy(trial_block + dst_ofs, pPrev_blk + src_ofs, len);

								color_rgba decoded_trial_block[MAX_BLOCK_PIXELS];
								if (!(*pUnpack_block_func)(trial_block, decoded_trial_block, block_index, pUnpack_block_func_user_data))
									continue;

								float trial_mse = compute_block_mse(pPixels, decoded_trial_block, block_width, block_height, total_block_pixels, num_comps, params.m_color_weights, one_over_total_color_weight);

								if (trial_mse < thresh_ms_err)
								{
									float t = trial_mse * smooth_block_mse_scale + trial_total_bits_times_lambda;

									if (t < best_t)
									{
										best_t = t;
										memcpy(best_block, trial_block, block_size_to_optimize_in_bytes);
										best_match_len = len;
										best_match_src_window_ofs = src_match_window_ofs;
										best_match_dst_window_ofs = dst_match_window_ofs;
										best_match_src_block_ofs = src_ofs;
										best_match_dst_block_ofs = dst_ofs;
										best_match_bits = trial_match_bits;
									}
								}

							} // dst_ofs
						} // src_ofs
					}
					else
					{
						const uint32_t match_dist = (block_index - prev_block_index) * total_block_stride_in_bytes;

						// Assume the block has 1 match and block_size_to_optimize_in_bytes-match_len literals.
						const float trial_match_bits = (float)compute_match_cost_estimate(match_dist, len);
						const float trial_total_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + trial_match_bits;
						const float trial_total_bits_times_lambda = trial_total_bits * params.m_lambda;

						for (uint32_t ofs = 0; ofs <= (block_size_to_optimize_in_bytes - len); ofs++)
						{
							assert(len + ofs <= block_size_to_optimize_in_bytes);
							
							const uint32_t dst_match_window_ofs = block_index * total_block_stride_in_bytes + ofs;
							const uint32_t src_match_window_ofs = prev_block_index * total_block_stride_in_bytes + ofs;

							float trial_match_bits_to_use = trial_match_bits;
							float trial_total_bits_times_lambda_to_use = trial_total_bits_times_lambda;
														
							uint32_t hs = hash_hsieh(pPrev_blk + ofs, len, ofs);

#if ERT_FAVOR_CONT_AND_REP0_MATCHES
							// Continue a previous match (which would cross block boundaries)
							if (((int)src_match_window_ofs == prev_match_window_ofs_to_favor_cont) && (ofs == 0))
							{
								float continue_match_trial_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + MATCH_CONTINUE_BITS;
								trial_match_bits_to_use = MATCH_CONTINUE_BITS;
								trial_total_bits_times_lambda_to_use = continue_match_trial_bits * params.m_lambda;
							}
							// Exploit REP0 matches
							else if ((prev_match_dist_to_favor != -1) && (src_match_window_ofs == (dst_match_window_ofs - prev_match_dist_to_favor)))
							{
								float continue_match_trial_bits = (block_size_to_optimize_in_bytes - len) * LITERAL_BITS + MATCH_REP0_BITS;
								trial_match_bits_to_use = MATCH_REP0_BITS;
								trial_total_bits_times_lambda_to_use = continue_match_trial_bits * params.m_lambda;
							}
							else
							{
								uint32_t hash_check = hash[hs & (HASH_SIZE - 1)];
								if ((hash_check & 0xFF) == (block_index & 0xFF))
								{
									if ((hash_check >> 8) == (hs >> 8))
										continue;
								}
							}
#else
							uint32_t hash_check = hash[hs & (HASH_SIZE - 1)];
							if ((hash_check & 0xFF) == (block_index & 0xFF))
							{
								if ((hash_check >> 8) == (hs >> 8))
									continue;
							}
#endif

							hash[hs & (HASH_SIZE - 1)] = (hs & 0xFFFFFF00) | (block_index & 0xFF);

							uint8_t trial_block[MAX_BLOCK_SIZE_IN_BYTES];
							memcpy(trial_block, pOrig_block, block_size_to_optimize_in_bytes);
							memcpy(trial_block + ofs, pPrev_blk + ofs, len);

							color_rgba decoded_trial_block[MAX_BLOCK_PIXELS];
							if (!(*pUnpack_block_func)(trial_block, decoded_trial_block, block_index, pUnpack_block_func_user_data))
								continue;

							float trial_mse = compute_block_mse(pPixels, decoded_trial_block, block_width, block_height, total_block_pixels, num_comps, params.m_color_weights, one_over_total_color_weight);

							if (trial_mse < thresh_ms_err)
							{
								float t = trial_mse * smooth_block_mse_scale + trial_total_bits_times_lambda_to_use;
								
								if (t < best_t)
								{
									best_t = t;
									memcpy(best_block, trial_block, block_size_to_optimize_in_bytes);
									best_match_len = len;
									best_match_src_window_ofs = src_match_window_ofs;
									best_match_dst_window_ofs = dst_match_window_ofs;
									best_match_src_block_ofs = ofs;
									best_match_dst_block_ofs = ofs;
									best_match_bits = trial_match_bits_to_use;
								}
							}
						} // ofs
					}

				} // len

			} // prev_block_index

			if (best_t < cur_t)
			{
				uint32_t best_second_match_len = 0, best_second_match_src_window_ofs = 0, best_second_match_dst_window_ofs = 0, best_second_match_src_block_ofs = 0, best_second_match_dst_block_ofs = 0;
								
				// Try injecting a second match, being sure it does't overlap with the first.
				if ((params.m_try_two_matches) && (best_match_len <= (block_size_to_optimize_in_bytes - 3)))
				{
					uint8_t matched_flags[MAX_BLOCK_SIZE_IN_BYTES];
					memset(matched_flags, 0, sizeof(matched_flags));
					memset(matched_flags + best_match_dst_block_ofs, 1, best_match_len);

					uint8_t orig_best_block[MAX_BLOCK_SIZE_IN_BYTES];
					memcpy(orig_best_block, best_block, block_size_to_optimize_in_bytes);
										
					for (int prev_block_index = last_block_to_check; prev_block_index >= first_block_to_check; --prev_block_index)
					{
						const uint8_t* pPrev_blk = &pBlock_bytes[prev_block_index * total_block_stride_in_bytes];

						const uint32_t match_dist = (block_index - prev_block_index) * total_block_stride_in_bytes;

						for (uint32_t len = 3; len <= (block_size_to_optimize_in_bytes - best_match_len); len++)
						{
							const float trial_total_bits = (block_size_to_optimize_in_bytes - len - best_match_len) * LITERAL_BITS + compute_match_cost_estimate(match_dist, len) + best_match_bits;

							const float trial_total_bits_times_lambda = trial_total_bits * params.m_lambda;

							for (uint32_t ofs = 0; ofs <= (block_size_to_optimize_in_bytes - len); ofs++)
							{
								int i;
								for (i = 0; i < (int)len; i++)
									if (matched_flags[ofs + i])
										break;
								if (i != (int)len)
									continue;

								assert(len + ofs <= block_size_to_optimize_in_bytes);

								const uint32_t dst_match_window_ofs = block_index * total_block_stride_in_bytes + ofs;
								const uint32_t src_match_window_ofs = prev_block_index * total_block_stride_in_bytes + ofs;

								uint8_t trial_block[MAX_BLOCK_SIZE_IN_BYTES];
								memcpy(trial_block, orig_best_block, block_size_to_optimize_in_bytes);
								memcpy(trial_block + ofs, pPrev_blk + ofs, len);

								color_rgba decoded_trial_block[MAX_BLOCK_PIXELS];
								if (!(*pUnpack_block_func)(trial_block, decoded_trial_block, block_index, pUnpack_block_func_user_data))
									continue;

								float trial_mse = compute_block_mse(pPixels, decoded_trial_block, block_width, block_height, total_block_pixels, num_comps, params.m_color_weights, one_over_total_color_weight);

								if (trial_mse < thresh_ms_err)
								{
									float t = trial_mse * smooth_block_mse_scale + trial_total_bits_times_lambda;

									if (t < best_t)
									{
										best_t = t;
										memcpy(best_block, trial_block, block_size_to_optimize_in_bytes);
										best_second_match_len = len;
										best_second_match_src_window_ofs = src_match_window_ofs;
										best_second_match_dst_window_ofs = dst_match_window_ofs;
										best_second_match_src_block_ofs = ofs;
										best_second_match_dst_block_ofs = ofs;
									}
								}
							}
						}
					}
				}

				memcpy(pOrig_block, best_block, block_size_to_optimize_in_bytes);
				total_modified++;

				if ((best_second_match_len == 0) || (best_match_dst_window_ofs > best_second_match_dst_window_ofs))
				{
					int best_match_dist = best_match_dst_window_ofs - best_match_src_window_ofs;
					assert(best_match_dist >= 1);
					(void)best_match_dist;

					if (block_size_to_optimize_in_bytes == total_block_stride_in_bytes)
					{
						// If the match goes all the way to the end of a block, we can try to continue it on the next encoded block.
						if ((best_match_dst_block_ofs + best_match_len) == total_block_stride_in_bytes)
							prev_match_window_ofs_to_favor_cont = best_match_src_window_ofs + best_match_len;
						else
							prev_match_window_ofs_to_favor_cont = -1;
					}

#if ERT_FAVOR_REP0_MATCHES
					// Compute the window offset where a cheaper REP0 match would be available
					prev_match_dist_to_favor = best_match_dist;
#endif
				}
				else
				{
					int best_match_dist = best_second_match_dst_window_ofs - best_second_match_src_window_ofs;
					assert(best_match_dist >= 1);
					(void)best_match_dist;

					if (block_size_to_optimize_in_bytes == total_block_stride_in_bytes)
					{
						// If the match goes all the way to the end of a block, we can try to continue it on the next encoded block.
						if ((best_second_match_dst_block_ofs + best_second_match_len) == total_block_stride_in_bytes)
							prev_match_window_ofs_to_favor_cont = best_second_match_src_window_ofs + best_second_match_len;
						else
							prev_match_window_ofs_to_favor_cont = -1;
					}

#if ERT_FAVOR_REP0_MATCHES
					// Compute the window offset where a cheaper REP0 match would be available
					prev_match_dist_to_favor = best_match_dist;
#endif
				}

				len_hist[best_match_len]++;

				if (best_second_match_len)
				{
					second_len_hist[best_second_match_len]++;
					total_second_matches++;
				}
			}
			else
			{
				prev_match_window_ofs_to_favor_cont = -1;
			}
						
		} // block_index
				
		if (params.m_debug_output)
		{
			printf("Total smooth blocks: %3.2f%%\n", total_smooth_blocks * 100.0f / num_blocks);

			printf("Match length histogram:\n");
			for (uint32_t i = MIN_MATCH_LEN; i <= block_size_to_optimize_in_bytes; i++)
				printf("%u%c", len_hist[i], (i < block_size_to_optimize_in_bytes) ? ',' : '\n');

			printf("Total second matches: %u %3.2f%%\n", total_second_matches, total_second_matches * 100.0f / num_blocks);
			printf("Secod match length histogram:\n");
			for (uint32_t i = MIN_MATCH_LEN; i <= block_size_to_optimize_in_bytes; i++)
				printf("%u%c", second_len_hist[i], (i < block_size_to_optimize_in_bytes) ? ',' : '\n');
		}
		
		return true;
	}

} // namespace ert