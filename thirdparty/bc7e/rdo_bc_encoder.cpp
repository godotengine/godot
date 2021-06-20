// rdo_bc_encoder.cpp
#include "rdo_bc_encoder.h"

#define RGBCX_IMPLEMENTATION
#include "rgbcx.h"

#define DECODE_BC4_TO_GRAYSCALE (0)

#ifdef _MSC_VER
#pragma warning (disable: 4127) // conditional expression is constant
#endif

using namespace utils;

namespace rdo_bc
{
	static const char* get_dxgi_format_string(DXGI_FORMAT fmt)
	{
		switch (fmt)
		{
		case DXGI_FORMAT_BC1_UNORM: return "BC1_UNORM";
		case DXGI_FORMAT_BC4_UNORM: return "BC4_UNORM";
		case DXGI_FORMAT_BC3_UNORM: return "BC3_UNORM";
		case DXGI_FORMAT_BC5_UNORM: return "BC5_UNORM";
		case DXGI_FORMAT_BC7_UNORM: return "BC7_UNORM";
		default: break;
		}
		return "?";
	}

	static std::vector<float> compute_block_mse_scales(const image_u8& source_image, uint32_t blocks_x, uint32_t blocks_y, uint32_t total_blocks, bool rdo_debug_output)
	{
		const float ULTRASMOOTH_BLOCK_STD_DEV_THRESHOLD = 2.9f;
		const float DARK_THRESHOLD = 13.0f;
		const float BRIGHT_THRESHOLD = 222.0f;
		const float ULTRAMOOTH_BLOCK_MSE_SCALE = 120.0f;
		const uint32_t ULTRASMOOTH_REGION_TOO_SMALL_THRESHOLD = 64;

		image_u8 ultrasmooth_blocks_vis(blocks_x, blocks_y);

		for (uint32_t by = 0; by < blocks_y; by++)
		{
			for (uint32_t bx = 0; bx < blocks_x; bx++)
			{
				color_quad_u8 block_pixels[16];
				source_image.get_block(bx, by, 4, 4, block_pixels);

				tracked_stat y_stats;
				for (uint32_t y = 0; y < 4; y++)
					for (uint32_t x = 0; x < 4; x++)
					{
						int l = block_pixels[x + y * 4].get_luma();
						y_stats.update(l);
					}

				float max_std_dev = compute_block_max_std_dev((color_quad_u8*)block_pixels, 4, 4, 3);

				float yl = max_std_dev / ULTRASMOOTH_BLOCK_STD_DEV_THRESHOLD;

				yl = clamp(yl, 0.0f, 1.0f);
				yl *= yl;

				float y_avg = y_stats.get_mean();

				if ((y_avg < DARK_THRESHOLD) || (y_avg >= BRIGHT_THRESHOLD))
					yl = 1.0f;

				int k = std::min<int>((int)(yl * 255.0f + .5f), 255);

				ultrasmooth_blocks_vis.fill_box(bx, by, 1, 1, color_quad_u8((uint8_t)k, 255));
			}
		}

		for (int pass = 0; pass < 1; pass++)
		{
			image_u8 next_vis(ultrasmooth_blocks_vis);

			for (int y = 0; y < (int)blocks_y; y++)
			{
				for (int x = 0; x < (int)blocks_x; x++)
				{
					int m = 0;

					for (int dy = -1; dy <= 1; dy++)
						for (int dx = -1; dx <= 1; dx++)
						{
							if (ultrasmooth_blocks_vis.get_clamped(x + dx, y + dy).r == 255)
								m = std::max<int>(m, ultrasmooth_blocks_vis.get_clamped(x + dx, y + dy).r);
						}

					next_vis(x, y).set((uint8_t)m, 255);
				}
			}

			ultrasmooth_blocks_vis.swap(next_vis);
		}

		for (uint32_t pass = 0; pass < 32; pass++)
		{
			image_u8 next_vis(ultrasmooth_blocks_vis);
			for (int y = 0; y < (int)blocks_y; y++)
			{
				for (int x = 0; x < (int)blocks_x; x++)
				{
					if (ultrasmooth_blocks_vis.get_clamped(x, y).r < 255)
					{
						int m = 0;

						for (int dy = -1; dy <= 1; dy++)
							for (int dx = -1; dx <= 1; dx++)
								if (ultrasmooth_blocks_vis.get_clamped(x + dx, y + dy).r == 255)
									m++;

						if (m >= 5)
							next_vis.set_pixel_clipped(x, y, color_quad_u8(255, 255, 255, 255));
					}
				}
			}
			ultrasmooth_blocks_vis.swap(next_vis);
		}

		image_u8 orig_ultrasmooth_blocks_vis(ultrasmooth_blocks_vis);

		if (rdo_debug_output)
		{
			save_png("ultrasmooth_block_mask_pre_filter.png", ultrasmooth_blocks_vis, false);
		}

		for (uint32_t by = 0; by < blocks_y; by++)
		{
			for (uint32_t bx = 0; bx < blocks_x; bx++)
			{
				const bool is_ultrasmooth = ultrasmooth_blocks_vis(bx, by).r == 0;
				if (!is_ultrasmooth)
					continue;

				std::vector<image_u8::pixel_coord> filled_pixels;
				filled_pixels.reserve(256);

				uint32_t total_set_pixels = ultrasmooth_blocks_vis.flood_fill(bx, by, color_quad_u8(255, 255, 255, 255), color_quad_u8(0, 0, 0, 255), &filled_pixels);

				if (total_set_pixels < ULTRASMOOTH_REGION_TOO_SMALL_THRESHOLD)
				{
					for (uint32_t i = 0; i < filled_pixels.size(); i++)
						orig_ultrasmooth_blocks_vis(filled_pixels[i].m_x, filled_pixels[i].m_y) = color_quad_u8(255, 255, 255, 255);
				}

			} // bx
		} // by

		ultrasmooth_blocks_vis = orig_ultrasmooth_blocks_vis;

		if (rdo_debug_output)
		{
			save_png("ultrasmooth_block_mask.png", ultrasmooth_blocks_vis, false);
		}

		std::vector<float> block_mse_scales(total_blocks);

		uint32_t total_ultrasmooth_blocks = 0;
		for (uint32_t by = 0; by < blocks_y; by++)
		{
			for (uint32_t bx = 0; bx < blocks_x; bx++)
			{
				const bool is_ultrasmooth = ultrasmooth_blocks_vis(bx, by).r == 0;

				block_mse_scales[bx + by * blocks_x] = is_ultrasmooth ? ULTRAMOOTH_BLOCK_MSE_SCALE : -1.0f;

				total_ultrasmooth_blocks += is_ultrasmooth;
			}
		}

		if (rdo_debug_output)
			printf("Total ultrasmooth blocks: %3.2f%%\n", total_ultrasmooth_blocks * 100.0f / total_blocks);

		return block_mse_scales;
	}

	rdo_bc_encoder::rdo_bc_encoder() :
		m_pOrig_source_image(nullptr),
		m_orig_width(0),
		m_orig_height(0),
		m_blocks_x(0),
		m_blocks_y(0),
		m_total_blocks(0),
		m_bytes_per_block(0),
		m_pixel_format_bpp(0),
		m_total_texels(0),
		m_has_alpha(false)
	{
	}

	void rdo_bc_encoder::clear()
	{
		m_pOrig_source_image = nullptr;

		m_source_image.clear();

		m_params.clear();

		m_orig_width = 0;
		m_orig_height = 0;
		m_blocks_x = 0;
		m_blocks_y = 0;
		m_total_blocks = 0;
		m_bytes_per_block = 0;
		m_pixel_format_bpp = 0;
		m_total_texels = 0;
		m_has_alpha = false;

		m_packed_image8.clear();
		m_packed_image16.clear();

		m_prerdo_packed_image8.clear();
		m_prerdo_packed_image16.clear();

		m_bc7enc_pack_params.clear();
#if SUPPORT_BC7E
		memset(&m_bc7e_pack_params, 0, sizeof(m_bc7e_pack_params));
#endif
	}

	bool rdo_bc_encoder::init(const utils::image_u8& src_image, rdo_bc_params& params)
	{
		clear();

		m_pOrig_source_image = &src_image;
		m_params = params;

		init_encoders();

		if (!init_source_image())
			return false;

		return true;
	}

	bool rdo_bc_encoder::encode()
	{
		if (!m_packed_image8.size() && !m_packed_image16.size())
			return false;

		if (!init_encoder_params())
			return false;

		if (!encode_texture())
			return false;

		if (!postprocess_rdo())
			return false;

		return true;
	}

	void rdo_bc_encoder::init_encoders()
	{
		rgbcx::init(m_params.m_bc1_mode);
		bc7enc_compress_block_init();
#if SUPPORT_BC7E
		ispc::bc7e_compress_block_init();
#endif
	}

	bool rdo_bc_encoder::init_encoder_params()
	{
		bc7enc_compress_block_params_init(&m_bc7enc_pack_params);
		if (!m_params.m_perceptual)
			bc7enc_compress_block_params_init_linear_weights(&m_bc7enc_pack_params);
		m_bc7enc_pack_params.m_max_partitions = m_params.m_bc7enc_max_partitions_to_scan;
		m_bc7enc_pack_params.m_uber_level = std::min(BC7ENC_MAX_UBER_LEVEL, m_params.m_bc7_uber_level);

		if (m_params.m_bc7enc_mode6_only)
			m_bc7enc_pack_params.m_mode_mask = 1 << 6;

		if ((m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM) && (m_params.m_rdo_lambda > 0.0f))
		{
			// Slam off perceptual in RDO mode - we don't support it (too slow).
			m_params.m_perceptual = false;
			m_bc7enc_pack_params.m_perceptual = false;
			bc7enc_compress_block_params_init_linear_weights(&m_bc7enc_pack_params);
		}

		if ((m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM) && (m_params.m_bc7enc_reduce_entropy))
		{
			// Configure the BC7 encoder with some decent parameters for later RDO post-processing.
			// Textures with alpha are harder for BC7 to handle, so we use more conservative defaults.

			m_bc7enc_pack_params.m_mode17_partition_estimation_filterbank = false;

			if (m_params.m_bc7enc_rdo_bc7_weight_modes)
			{
				// Weight modes 5 and especially 6 more highly than the other modes.
				if (m_has_alpha)
				{
					m_bc7enc_pack_params.m_mode5_error_weight = .7f;
					m_bc7enc_pack_params.m_mode6_error_weight = .6f;
				}
				else
				{
					m_bc7enc_pack_params.m_mode6_error_weight = .4f;
				}
			}

			if (m_params.m_bc7enc_rdo_bc7_weight_low_frequency_partitions)
			{
				// Slightly prefer the lower frequency partition patterns.
				m_bc7enc_pack_params.m_low_frequency_partition_weight = .9999f;
			}

			if (m_params.m_bc7enc_rdo_bc7_quant_mode6_endpoints)
			{
				// As a good default, don't quantize mode 6 endpoints if the texture has alpha. This isn't required, but helps mask textures.
				//if (!has_alpha)
				m_bc7enc_pack_params.m_quant_mode6_endpoints = true;
			}

			if (m_params.m_bc7enc_rdo_bc7_pbit1_weighting)
			{
				// Favor p-bit 0 vs. 1, to slightly lower the entropy of output blocks with p-bits
				m_bc7enc_pack_params.m_pbit1_weight = 1.3f;
			}
		}

#if SUPPORT_BC7E
		// Now initialize the BC7 compressor's parameters.

		memset(&m_bc7e_pack_params, 0, sizeof(m_bc7e_pack_params));
		switch (m_params.m_bc7_uber_level)
		{
		case 0:
			ispc::bc7e_compress_block_params_init_ultrafast(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 1:
			ispc::bc7e_compress_block_params_init_veryfast(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 2:
			ispc::bc7e_compress_block_params_init_fast(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 3:
			ispc::bc7e_compress_block_params_init_basic(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 4:
			ispc::bc7e_compress_block_params_init_slow(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 5:
			ispc::bc7e_compress_block_params_init_veryslow(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		case 6:
		default:
			ispc::bc7e_compress_block_params_init_slowest(&m_bc7e_pack_params, m_params.m_perceptual);
			break;
		}
#endif

		if (m_params.m_status_output)
		{
			if (m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM)
			{
				if ((SUPPORT_BC7E) && (m_params.m_use_bc7e))
					printf("bc7e.ispc uber level: %u, perceptual: %u\n", m_params.m_bc7_uber_level, m_params.m_perceptual);
				else
				{
					printf("\nbc7enc parameters:\n");
					m_bc7enc_pack_params.print();
				}
			}
			else
			{
				printf("BC1 level: %u, use 3-color mode: %u, use 3-color mode for black: %u, bc1_mode: %u\n",
					m_params.m_bc1_quality_level, m_params.m_use_bc1_3color_mode, m_params.m_use_bc1_3color_mode_for_black, (int)m_params.m_bc1_mode);
			}

			if ((m_params.m_dxgi_format == DXGI_FORMAT_BC3_UNORM) || (m_params.m_dxgi_format == DXGI_FORMAT_BC4_UNORM) || (m_params.m_dxgi_format == DXGI_FORMAT_BC5_UNORM))
			{
				printf("Use high quality BC4 block encoder: %u, BC4 block radius: %u, use 6 value mode: %u, use 8 value mode: %u\n",
					m_params.m_use_hq_bc345, m_params.m_bc345_search_rad, (m_params.m_bc345_mode_mask & 2) != 0, (m_params.m_bc345_mode_mask & 1) != 0);
			}

			printf("\nrdo_bc_params:\n");
			printf("  Perceptual: %u\n", m_params.m_perceptual);
			printf("  Y Flip: %u\n", m_params.m_y_flip);
			printf("  DXGI format: 0x%X %s\n", m_params.m_dxgi_format, get_dxgi_format_string(m_params.m_dxgi_format));

			printf("BC1-5 parameters:\n");
			printf("  BC45 channels: %u %u\n", m_params.m_bc45_channel0, m_params.m_bc45_channel1);
			printf("  BC1 approximation mode: %u\n", (int)m_params.m_bc1_mode);
			printf("  Use BC1 3-color mode: %u\n", m_params.m_use_bc1_3color_mode);
			printf("  Use BC1 3-color mode for black: %u\n", m_params.m_use_bc1_3color_mode_for_black);
			printf("  BC1 quality level: %u\n", m_params.m_bc1_quality_level);
			printf("  Use HQ BC345: %u\n", m_params.m_use_hq_bc345);
			printf("  BC345 search radius: %u\n", m_params.m_bc345_search_rad);
			printf("  BC345 mode mask: 0x%X\n", m_params.m_bc345_mode_mask);
			
			printf("BC7 parameters:\n");
			printf("  Use bc7e: %u\n", m_params.m_use_bc7e);
			printf("  BC7 uber level: %u\n", m_params.m_bc7_uber_level);

			printf("RDO parameters:\n");
			printf("  Lambda: %f\n", m_params.m_rdo_lambda);
			printf("  Lookback window size: %u\n", m_params.m_lookback_window_size);
			printf("  Custom lookback window size: %u\n", m_params.m_custom_lookback_window_size);
			printf("  Try 2 matches: %u\n", m_params.m_rdo_try_2_matches);
			printf("  Smooth block error scale: %f\n", m_params.m_rdo_smooth_block_error_scale);
			printf("  Custom RDO smooth block error scale: %u\n", m_params.m_custom_rdo_smooth_block_error_scale);
			printf("  Max smooth block std dev: %f\n", m_params.m_rdo_max_smooth_block_std_dev);
			printf("  Allow relative movement: %u\n", m_params.m_rdo_allow_relative_movement);
			printf("  Ultrasmooth block handling: %u\n", m_params.m_rdo_ultrasmooth_block_handling);
			printf("  Multithreading: %u, max threads: %u\n", m_params.m_rdo_multithreading, m_params.m_rdo_max_threads);
			
			printf("bc7enc parameters:\n");
			printf("  Mode 6 only: %u\n", m_params.m_bc7enc_mode6_only);
			printf("  Max partitions to scan: %u\n", m_params.m_bc7enc_max_partitions_to_scan);
			printf("  Quant mode 6 endpoints: %u\n", m_params.m_bc7enc_rdo_bc7_quant_mode6_endpoints);
			printf("  Weight modes: %u\n", m_params.m_bc7enc_rdo_bc7_weight_modes);
			printf("  Weight low freq partitions: %u\n", m_params.m_bc7enc_rdo_bc7_weight_low_frequency_partitions);
			printf("  P-bit1 weighting: %u\n", m_params.m_bc7enc_rdo_bc7_pbit1_weighting);
			printf("  Reduce entropy mode: %u\n", m_params.m_bc7enc_reduce_entropy);
			printf("\n");
		}

		return true;
	}

	bool rdo_bc_encoder::init_source_image()
	{
		switch (m_params.m_dxgi_format)
		{
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC4_UNORM:
			m_pixel_format_bpp = 4;
			break;
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC7_UNORM:
			m_pixel_format_bpp = 8;
			break;
		default:
			return false;
		}

		m_bytes_per_block = (16 * m_pixel_format_bpp) / 8;
		assert((m_bytes_per_block == 8) || (m_bytes_per_block == 16));

		m_source_image = *m_pOrig_source_image;

		m_orig_width = m_source_image.width();
		m_orig_height = m_source_image.height();

		if (m_params.m_y_flip)
		{
			utils::image_u8 temp;
			temp.init(m_orig_width, m_orig_height);

			for (uint32_t y = 0; y < m_orig_height; y++)
				for (uint32_t x = 0; x < m_orig_width; x++)
					temp(x, (m_orig_height - 1) - y) = m_source_image(x, y);

			temp.swap(m_source_image);
		}

		m_source_image.crop_dup_borders((m_source_image.width() + 3) & ~3, (m_source_image.height() + 3) & ~3);

		m_blocks_x = m_source_image.width() / 4;
		m_blocks_y = m_source_image.height() / 4;
		m_total_blocks = m_blocks_x * m_blocks_y;
		m_total_texels = m_total_blocks * 16;

		bool has_alpha = false;
		for (int by = 0; by < ((int)m_blocks_y) && !has_alpha; by++)
		{
			for (uint32_t bx = 0; bx < m_blocks_x; bx++)
			{
				color_quad_u8 pixels[16];
				m_source_image.get_block(bx, by, 4, 4, pixels);

				for (uint32_t i = 0; i < 16; i++)
				{
					if (pixels[i].m_c[3] < 255)
					{
						has_alpha = true;
						break;
					}
				}
			}
		}
				
		if (m_pixel_format_bpp == 8)
			m_packed_image16.resize(m_total_blocks);
		else
			m_packed_image8.resize(m_total_blocks);

		return true;
	}
		
	bool rdo_bc_encoder::encode_texture()
	{
		clock_t start_t = clock();

		uint32_t bc7_mode_hist[8];
		memset(bc7_mode_hist, 0, sizeof(bc7_mode_hist));

#if SUPPORT_BC7E
		if ((m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM) && (m_params.m_use_bc7e))
		{
			if (m_params.m_status_output)
				printf("Using bc7e: ");

#pragma omp parallel for
			for (int32_t by = 0; by < static_cast<int32_t>(m_blocks_y); by++)
			{
				// Process 64 blocks at a time, for efficient SIMD processing.
				// Ideally, N >= 8 (or more) and (N % 8) == 0.
				const int N = 64;

				for (uint32_t bx = 0; bx < m_blocks_x; bx += N)
				{
					const uint32_t num_blocks_to_process = std::min<uint32_t>(m_blocks_x - bx, N);

					color_quad_u8 pixels[16 * N];

					// Extract num_blocks_to_process 4x4 pixel blocks from the source image and put them into the pixels[] array.
					for (uint32_t b = 0; b < num_blocks_to_process; b++)
						m_source_image.get_block(bx + b, by, 4, 4, pixels + b * 16);

					// Compress the blocks to BC7.
					// Note: If you've used Intel's ispc_texcomp, the input pixels are different. BC7E requires a pointer to an array of 16 pixels for each block.
					block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];
					ispc::bc7e_compress_blocks(num_blocks_to_process, reinterpret_cast<uint64_t*>(pBlock), reinterpret_cast<const uint32_t*>(pixels), &m_bc7e_pack_params);
				}

				if (m_params.m_status_output)
				{
					if ((by & 63) == 0)
						printf(".");
				}
			}

			for (int by = 0; by < (int)m_blocks_y; by++)
			{
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];

					uint32_t mode = ((uint8_t*)pBlock)[0];
					for (uint32_t m = 0; m <= 7; m++)
					{
						if (mode & (1 << m))
						{
							bc7_mode_hist[m]++;
							break;
						}
					}
				}
			}
		}
		else
#endif
		{
#pragma omp parallel for
			for (int by = 0; by < (int)m_blocks_y; by++)
			{
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					color_quad_u8 pixels[16];

					m_source_image.get_block(bx, by, 4, 4, pixels);

					switch (m_params.m_dxgi_format)
					{
					case DXGI_FORMAT_BC1_UNORM:
					{
						block8* pBlock = &m_packed_image8[bx + by * m_blocks_x];

						rgbcx::encode_bc1(m_params.m_bc1_quality_level, pBlock, &pixels[0].m_c[0], m_params.m_use_bc1_3color_mode, m_params.m_use_bc1_3color_mode_for_black);
						break;
					}
					case DXGI_FORMAT_BC3_UNORM:
					{
						block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];

						if (m_params.m_use_hq_bc345)
							rgbcx::encode_bc3_hq(m_params.m_bc1_quality_level, pBlock, &pixels[0].m_c[0], m_params.m_bc345_search_rad, m_params.m_bc345_mode_mask);
						else
							rgbcx::encode_bc3(m_params.m_bc1_quality_level, pBlock, &pixels[0].m_c[0]);
						break;
					}
					case DXGI_FORMAT_BC4_UNORM:
					{
						block8* pBlock = &m_packed_image8[bx + by * m_blocks_x];

						if (m_params.m_use_hq_bc345)
							rgbcx::encode_bc4_hq(pBlock, &pixels[0].m_c[m_params.m_bc45_channel0], 4, m_params.m_bc345_search_rad, m_params.m_bc345_mode_mask);
						else
							rgbcx::encode_bc4(pBlock, &pixels[0].m_c[m_params.m_bc45_channel0], 4);
						break;
					}
					case DXGI_FORMAT_BC5_UNORM:
					{
						block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];

						if (m_params.m_use_hq_bc345)
							rgbcx::encode_bc5_hq(pBlock, &pixels[0].m_c[0], m_params.m_bc45_channel0, m_params.m_bc45_channel1, 4, m_params.m_bc345_search_rad, m_params.m_bc345_mode_mask);
						else
							rgbcx::encode_bc5(pBlock, &pixels[0].m_c[0], m_params.m_bc45_channel0, m_params.m_bc45_channel1, 4);
						break;
					}
					case DXGI_FORMAT_BC7_UNORM:
					{
						block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];

						bc7enc_compress_block(pBlock, pixels, &m_bc7enc_pack_params);

#pragma omp critical
						{
							uint32_t mode = ((uint8_t*)pBlock)[0];
							for (uint32_t m = 0; m <= 7; m++)
							{
								if (mode & (1 << m))
								{
									bc7_mode_hist[m]++;
									break;
								}
							}
						}

						break;
					}
					default:
					{
						assert(0);
						break;
					}
					}
				}

				if (m_params.m_status_output)
				{
					if ((by & 127) == 0)
						printf(".");
				}
			}
		}

		clock_t end_t = clock();

		if (m_params.m_status_output)
		{
			printf("\nTotal encoding time: %f secs\n", (double)(end_t - start_t) / CLOCKS_PER_SEC);

			if (m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM)
			{
				printf("BC7 mode histogram:\n");
				for (uint32_t i = 0; i < 8; i++)
					printf("%u: %u\n", i, bc7_mode_hist[i]);
			}
		}

		return true;
	}

	bool rdo_bc_encoder::postprocess_rdo()
	{
		m_prerdo_packed_image8 = m_packed_image8;
		m_prerdo_packed_image16 = m_packed_image16;

		// Post-process the data with Rate Distortion Optimization
		if (m_params.m_rdo_lambda <= 0.0f)
			return true;

		const uint32_t MIN_RDO_MULTITHREADING_BLOCKS = 4096;
		const int rdo_total_threads = (m_params.m_rdo_multithreading && (m_params.m_rdo_max_threads > 1) && (m_total_blocks >= MIN_RDO_MULTITHREADING_BLOCKS)) ? m_params.m_rdo_max_threads : 1;

		if (m_params.m_status_output)
			printf("rdo_total_threads: %u\n", rdo_total_threads);

		int blocks_remaining = m_total_blocks, cur_block_index = 0;
		std::vector<int> blocks_to_do(rdo_total_threads), first_block_index(rdo_total_threads);
		for (int p = 0; p < rdo_total_threads; p++)
		{
			const int num_blocks = (p == (rdo_total_threads - 1)) ? blocks_remaining : (m_total_blocks / rdo_total_threads);

			blocks_to_do[p] = num_blocks;
			first_block_index[p] = cur_block_index;

			cur_block_index += num_blocks;
			blocks_remaining -= num_blocks;
		}

		assert(!blocks_remaining && cur_block_index == (int)m_total_blocks);

		ert::reduce_entropy_params ert_p;

		ert_p.m_lambda = m_params.m_rdo_lambda;
		ert_p.m_lookback_window_size = m_params.m_lookback_window_size;
		ert_p.m_smooth_block_max_mse_scale = m_params.m_rdo_smooth_block_error_scale;
		ert_p.m_max_smooth_block_std_dev = m_params.m_rdo_max_smooth_block_std_dev;
		ert_p.m_debug_output = m_params.m_rdo_debug_output;
		ert_p.m_try_two_matches = m_params.m_rdo_try_2_matches;
		ert_p.m_allow_relative_movement = m_params.m_rdo_allow_relative_movement;
		ert_p.m_skip_zero_mse_blocks = false;
		
		std::vector<float> block_rgb_mse_scales(compute_block_mse_scales(m_source_image, m_blocks_x, m_blocks_y, m_total_blocks, m_params.m_rdo_debug_output));

		std::vector<rgbcx::color32> block_pixels(m_total_blocks * 16);

		for (uint32_t by = 0; by < m_blocks_y; by++)
			for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				m_source_image.get_block(bx, by, 4, 4, (color_quad_u8*)&block_pixels[(bx + by * m_blocks_x) * 16]);

		unpacker_funcs block_unpackers;
		block_unpackers.m_allow_3color_mode = m_params.m_use_bc1_3color_mode;
		block_unpackers.m_use_bc1_3color_mode_for_black = m_params.m_use_bc1_3color_mode_for_black;
		block_unpackers.m_mode = m_params.m_bc1_mode;

		if (m_params.m_dxgi_format == DXGI_FORMAT_BC7_UNORM)
		{
			ert_p.m_lookback_window_size = std::max(16U, m_params.m_lookback_window_size);

			// BC7 RDO
			const uint32_t NUM_COMPONENTS = 4;

			if (!m_params.m_custom_rdo_smooth_block_error_scale)
			{
				// Attempt to compute a decent conservative smooth block MSE max scaling factor.
				// No single smooth block scale setting can work for all textures (unless it's ridiuclously large, killing efficiency).
				ert_p.m_smooth_block_max_mse_scale = lerp(15.0f, 50.0f, std::min(1.0f, ert_p.m_lambda / 4.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f (use -zb# to override)\n", ert_p.m_smooth_block_max_mse_scale);
			}

			for (uint32_t by = 0; by < m_blocks_y; by++)
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					float& s = block_rgb_mse_scales[bx + by * m_blocks_x];
					if (s > 0.0f)
						s = std::max(ert_p.m_smooth_block_max_mse_scale, s * std::min(ert_p.m_lambda, 3.0f));
				}

			if (m_params.m_status_output)
			{
				printf("\nERT parameters:\n");
				ert_p.print();
				printf("\n");
			}

			uint32_t total_modified = 0;

			clock_t rdo_start_t = clock();

#pragma omp parallel for
			for (int p = 0; p < rdo_total_threads; p++)
			{
				const int first_block_to_encode = first_block_index[p];
				const int num_blocks_to_encode = blocks_to_do[p];
				if (!num_blocks_to_encode)
					continue;

				uint32_t total_modified_local = 0;

				std::vector<float> local_block_rgb_mse_scales(num_blocks_to_encode);
				for (int i = 0; i < num_blocks_to_encode; i++)
					local_block_rgb_mse_scales[i] = block_rgb_mse_scales[first_block_to_encode + i];

				ert::reduce_entropy(&m_packed_image16[first_block_to_encode], num_blocks_to_encode,
					16, 16, 4, 4, NUM_COMPONENTS,
					(ert::color_rgba*)&block_pixels[16 * first_block_to_encode], ert_p, total_modified_local,
					unpacker_funcs::unpack_bc7_block, &block_unpackers,
					m_params.m_rdo_ultrasmooth_block_handling ? &local_block_rgb_mse_scales : nullptr);

#pragma omp critical
				{
					total_modified += total_modified_local;
				}
			} // p

			clock_t rdo_end_t = clock();

			if (m_params.m_status_output)
			{
				printf("Total RDO time: %f secs\n", (double)(rdo_end_t - rdo_start_t) / CLOCKS_PER_SEC);

				printf("Total blocks modified: %u %3.2f%%\n", total_modified, total_modified * 100.0f / m_total_blocks);

				uint32_t bc7_mode_hist[8];
				memset(bc7_mode_hist, 0, sizeof(bc7_mode_hist));

				for (int by = 0; by < (int)m_blocks_y; by++)
				{
					for (uint32_t bx = 0; bx < m_blocks_x; bx++)
					{
						block16* pBlock = &m_packed_image16[bx + by * m_blocks_x];

						const uint32_t mode_byte = ((uint8_t*)pBlock)[0];

						uint32_t m;
						for (m = 0; m <= 7; m++)
						{
							if (mode_byte & (1 << m))
							{
								bc7_mode_hist[m]++;
								break;
							}
						}
						assert(m != 8);
					}
				}

				printf("BC7 mode histogram:\n");
				for (uint32_t i = 0; i < 8; i++)
					printf("%u: %u\n", i, bc7_mode_hist[i]);
			}
		}
		else if (m_params.m_dxgi_format == DXGI_FORMAT_BC5_UNORM)
		{
			// BC5 RDO - One BC4 block for R followed by one BC4 block for G

			ert_p.m_lookback_window_size = std::max(16U, m_params.m_lookback_window_size);

			std::vector<rgbcx::color32> block_pixels_r(m_total_blocks * 16), block_pixels_g(m_total_blocks * 16);

			for (uint32_t by = 0; by < m_blocks_y; by++)
			{
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					color_quad_u8 orig_block[16];
					m_source_image.get_block(bx, by, 4, 4, orig_block);

					color_quad_u8* pDst_block_r = (color_quad_u8*)&block_pixels_r[(bx + by * m_blocks_x) * 16];
					color_quad_u8* pDst_block_g = (color_quad_u8*)&block_pixels_g[(bx + by * m_blocks_x) * 16];

					for (uint32_t i = 0; i < 16; i++)
					{
						pDst_block_r[i].set(orig_block[i].r, 0, 0, 0);
						pDst_block_g[i].set(orig_block[i].g, 0, 0, 0);
					}
				}
			}

			const uint32_t NUM_COMPONENTS = 1;

			ert_p.m_color_weights[1] = 0;
			ert_p.m_color_weights[2] = 0;
			ert_p.m_color_weights[3] = 0;

			if (!m_params.m_custom_rdo_smooth_block_error_scale)
			{
				// Attempt to compute a decent conservative smooth block MSE max scaling factor.
				// No single smooth block scale setting can work for all textures (unless it's ridiuclously large, killing efficiency).
				ert_p.m_smooth_block_max_mse_scale = lerp(10.0f, 30.0f, std::min(1.0f, ert_p.m_lambda / 4.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f (use -zb# to override)\n", ert_p.m_smooth_block_max_mse_scale);
			}

			if (m_params.m_status_output)
			{
				printf("\nERT parameters:\n");
				ert_p.print();
				printf("\n");
			}

			uint32_t total_modified_r = 0, total_modified_g = 0;

			clock_t rdo_start_t = clock();

#pragma omp parallel for
			for (int p = 0; p < rdo_total_threads; p++)
			{
				const int first_block_to_encode = first_block_index[p];
				const int num_blocks_to_encode = blocks_to_do[p];
				if (!num_blocks_to_encode)
					continue;

				uint32_t total_modified_local_r = 0, total_modified_local_g = 0;

				ert::reduce_entropy(&m_packed_image16[first_block_to_encode], num_blocks_to_encode,
					2 * sizeof(rgbcx::bc4_block), sizeof(rgbcx::bc4_block), 4, 4, NUM_COMPONENTS,
					(ert::color_rgba*)&block_pixels_r[16 * first_block_to_encode], ert_p, total_modified_local_r,
					unpacker_funcs::unpack_bc4_block, &block_unpackers);

				ert::reduce_entropy((uint8_t*)&m_packed_image16[first_block_to_encode] + sizeof(rgbcx::bc4_block), num_blocks_to_encode,
					2 * sizeof(rgbcx::bc4_block), sizeof(rgbcx::bc4_block), 4, 4, NUM_COMPONENTS,
					(ert::color_rgba*)&block_pixels_g[16 * first_block_to_encode], ert_p, total_modified_local_g,
					unpacker_funcs::unpack_bc4_block, &block_unpackers);

#pragma omp critical
				{
					total_modified_r += total_modified_local_r;
					total_modified_g += total_modified_local_g;
				}
			} // p

			clock_t rdo_end_t = clock();

			if (m_params.m_status_output)
			{
				printf("Total RDO time: %f secs\n", (double)(rdo_end_t - rdo_start_t) / CLOCKS_PER_SEC);

				printf("Total blocks modified R: %u %3.2f%%\n", total_modified_r, total_modified_r * 100.0f / m_total_blocks);
				printf("Total blocks modified G: %u %3.2f%%\n", total_modified_g, total_modified_g * 100.0f / m_total_blocks);
			}
		}
		else if (m_params.m_dxgi_format == DXGI_FORMAT_BC4_UNORM)
		{
			// BC4 RDO - One BC4 block for R

			const uint32_t NUM_COMPONENTS = 1;

			ert_p.m_color_weights[1] = 0;
			ert_p.m_color_weights[2] = 0;
			ert_p.m_color_weights[3] = 0;

			if (!m_params.m_custom_rdo_smooth_block_error_scale)
			{
				// Attempt to compute a decent conservative smooth block MSE max scaling factor.
				// No single smooth block scale setting can work for all textures (unless it's ridiuclously large, killing efficiency).
				ert_p.m_smooth_block_max_mse_scale = lerp(10.0f, 30.0f, std::min(1.0f, ert_p.m_lambda / 4.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f (use -zb# to override)\n", ert_p.m_smooth_block_max_mse_scale);
			}

			if (m_params.m_status_output)
			{
				printf("\nERT parameters:\n");
				ert_p.print();
				printf("\n");
			}

			uint32_t total_modified = 0;

			clock_t rdo_start_t = clock();

#pragma omp parallel for
			for (int p = 0; p < rdo_total_threads; p++)
			{
				const int first_block_to_encode = first_block_index[p];
				const int num_blocks_to_encode = blocks_to_do[p];
				if (!num_blocks_to_encode)
					continue;

				uint32_t total_modified_local = 0;

				ert::reduce_entropy(&m_packed_image8[first_block_to_encode], num_blocks_to_encode,
					sizeof(rgbcx::bc4_block), sizeof(rgbcx::bc4_block), 4, 4, NUM_COMPONENTS,
					(ert::color_rgba*)&block_pixels[16 * first_block_to_encode], ert_p, total_modified_local,
					unpacker_funcs::unpack_bc4_block, &block_unpackers);

#pragma omp critical
				{
					total_modified += total_modified_local;
				}
			} // p

			clock_t rdo_end_t = clock();

			if (m_params.m_status_output)
			{
				printf("Total RDO time: %f secs\n", (double)(rdo_end_t - rdo_start_t) / CLOCKS_PER_SEC);

				printf("Total blocks modified: %u %3.2f%%\n", total_modified, total_modified * 100.0f / m_total_blocks);
			}
		}
		else if (m_params.m_dxgi_format == DXGI_FORMAT_BC1_UNORM)
		{
			// BC1 RDO - One BC1 block
			const uint32_t NUM_COMPONENTS = 3;

			ert_p.m_color_weights[3] = 0;

			if (!m_params.m_custom_rdo_smooth_block_error_scale)
			{
				// This is just a hack - no single setting can work for all textures.
				ert_p.m_smooth_block_max_mse_scale = lerp(15.0f, 50.0f, std::min(1.0f, ert_p.m_lambda / 8.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f (use -zb# to override)\n", ert_p.m_smooth_block_max_mse_scale);
			}

			for (uint32_t by = 0; by < m_blocks_y; by++)
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					float& s = block_rgb_mse_scales[bx + by * m_blocks_x];
					if (s > 0.0f)
						s = std::max(ert_p.m_smooth_block_max_mse_scale, s * std::min(ert_p.m_lambda, 3.0f));
				}

			printf("\nERT parameters:\n");
			ert_p.print();
			printf("\n");

			uint32_t total_modified = 0;

			clock_t rdo_start_t = clock();

#pragma omp parallel for
			for (int p = 0; p < rdo_total_threads; p++)
			{
				const int first_block_to_encode = first_block_index[p];
				const int num_blocks_to_encode = blocks_to_do[p];
				if (!num_blocks_to_encode)
					continue;

				uint32_t total_modified_local = 0;

				std::vector<float> local_block_rgb_mse_scales(num_blocks_to_encode);
				for (int i = 0; i < num_blocks_to_encode; i++)
					local_block_rgb_mse_scales[i] = block_rgb_mse_scales[first_block_to_encode + i];

				ert::reduce_entropy(&m_packed_image8[first_block_to_encode], num_blocks_to_encode,
					sizeof(rgbcx::bc1_block), sizeof(rgbcx::bc1_block), 4, 4, NUM_COMPONENTS,
					(ert::color_rgba*)&block_pixels[16 * first_block_to_encode], ert_p, total_modified_local,
					unpacker_funcs::unpack_bc1_block, &block_unpackers,
					m_params.m_rdo_ultrasmooth_block_handling ? &local_block_rgb_mse_scales : nullptr);

#pragma omp critical
				{
					total_modified += total_modified_local;
				}
			} // p

			clock_t rdo_end_t = clock();

			if (m_params.m_status_output)
			{
				printf("Total RDO time: %f secs\n", (double)(rdo_end_t - rdo_start_t) / CLOCKS_PER_SEC);

				printf("Total blocks modified: %u %3.2f%%\n",
					total_modified, total_modified * 100.0f / m_total_blocks);
			}
		}
		else if (m_params.m_dxgi_format == DXGI_FORMAT_BC3_UNORM)
		{
			// BC3 RDO - One BC4 block followed by one BC1 block

			ert_p.m_lookback_window_size = std::max(16U, m_params.m_lookback_window_size);

			std::vector<rgbcx::color32> block_pixels_a(m_total_blocks * 16);

			for (uint32_t by = 0; by < m_blocks_y; by++)
			{
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					color_quad_u8 orig_block[16];
					m_source_image.get_block(bx, by, 4, 4, orig_block);

					color_quad_u8* pDst_block_a = (color_quad_u8*)&block_pixels_a[(bx + by * m_blocks_x) * 16];
					for (uint32_t i = 0; i < 16; i++)
						pDst_block_a[i].set(orig_block[i].a, 0, 0, 0);
				}
			}

			ert_p.m_color_weights[3] = 0;

			ert::reduce_entropy_params ert_alpha_p(ert_p);
			ert_alpha_p.m_color_weights[1] = 0;
			ert_alpha_p.m_color_weights[2] = 0;
			ert_alpha_p.m_color_weights[3] = 0;

			if (!m_params.m_custom_rdo_smooth_block_error_scale)
			{
				// This is just a hack - no single setting can work for all textures.
				ert_p.m_smooth_block_max_mse_scale = lerp(15.0f, 50.0f, std::min(1.0f, ert_p.m_lambda / 8.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f (use -zb# to override) for RGB\n", ert_p.m_smooth_block_max_mse_scale);

				ert_alpha_p.m_smooth_block_max_mse_scale = lerp(10.0f, 30.0f, std::min(1.0f, ert_alpha_p.m_lambda / 4.0f));

				if (m_params.m_status_output)
					printf("Using an automatically computed smooth block error scale of %f for Alpha\n", ert_alpha_p.m_smooth_block_max_mse_scale);
			}

			for (uint32_t by = 0; by < m_blocks_y; by++)
				for (uint32_t bx = 0; bx < m_blocks_x; bx++)
				{
					float& s = block_rgb_mse_scales[bx + by * m_blocks_x];
					if (s > 0.0f)
						s = std::max(ert_p.m_smooth_block_max_mse_scale, s * std::min(ert_p.m_lambda, 3.0f));
				}

			if (m_params.m_status_output)
			{
				printf("\nERT RGB parameters:\n");
				ert_p.print();

				printf("\nERT Alpha parameters:\n");
				ert_alpha_p.print();
				printf("\n");
			}

			uint32_t total_modified_rgb = 0, total_modified_alpha = 0;

			block_unpackers.m_allow_3color_mode = false;
			block_unpackers.m_use_bc1_3color_mode_for_black = false;

			clock_t rdo_start_t = clock();

#pragma omp parallel for
			for (int p = 0; p < rdo_total_threads; p++)
			{
				const int first_block_to_encode = first_block_index[p];
				const int num_blocks_to_encode = blocks_to_do[p];
				if (!num_blocks_to_encode)
					continue;

				uint32_t total_modified_local_rgb = 0, total_modified_local_alpha = 0;

				ert::reduce_entropy((uint8_t*)&m_packed_image16[first_block_to_encode], num_blocks_to_encode,
					sizeof(rgbcx::bc1_block) * 2, sizeof(rgbcx::bc4_block), 4, 4, 1,
					(ert::color_rgba*)&block_pixels_a[16 * first_block_to_encode], ert_alpha_p, total_modified_local_alpha,
					unpacker_funcs::unpack_bc4_block, &block_unpackers);

				std::vector<float> local_block_rgb_mse_scales(num_blocks_to_encode);
				for (int i = 0; i < num_blocks_to_encode; i++)
					local_block_rgb_mse_scales[i] = block_rgb_mse_scales[first_block_to_encode + i];

				ert::reduce_entropy((uint8_t*)&m_packed_image16[first_block_to_encode] + sizeof(rgbcx::bc1_block), num_blocks_to_encode,
					sizeof(rgbcx::bc1_block) * 2, sizeof(rgbcx::bc1_block), 4, 4, 3,
					(ert::color_rgba*)&block_pixels[16 * first_block_to_encode], ert_p, total_modified_local_rgb,
					unpacker_funcs::unpack_bc1_block, &block_unpackers,
					m_params.m_rdo_ultrasmooth_block_handling ? &local_block_rgb_mse_scales : nullptr);

#pragma omp critical
				{
					total_modified_rgb += total_modified_local_rgb;
					total_modified_alpha += total_modified_local_alpha;
				}
			} // p

			clock_t rdo_end_t = clock();

			if (m_params.m_status_output)
			{
				printf("Total RDO time: %f secs\n", (double)(rdo_end_t - rdo_start_t) / CLOCKS_PER_SEC);

				printf("Total RGB blocks modified: %u %3.2f%%\n", total_modified_rgb, total_modified_rgb * 100.0f / m_total_blocks);
				printf("Total Alpha blocks modified: %u %3.2f%%\n", total_modified_alpha, total_modified_alpha * 100.0f / m_total_blocks);
			}
		}

		return true;
	}

	bool rdo_bc_encoder::unpack_blocks(image_u8& unpacked_image) const
	{
		unpacked_image.init(get_blocks_x() * 4, get_blocks_y() * 4);

		bool bc1_punchthrough_flag = false;
		bool used_bc1_transparent_texels_for_black = false;

		bool unpack_failed = false;
				
#pragma omp parallel for
		for (int by = 0; by < (int)get_blocks_y(); by++)
		{
			for (uint32_t bx = 0; bx < get_blocks_x(); bx++)
			{
				const void* pBlock = (const uint8_t*)get_blocks() + (bx + by * get_blocks_x()) * get_bytes_per_block();

				color_quad_u8 unpacked_pixels[16];
				for (uint32_t i = 0; i < 16; i++)
					unpacked_pixels[i].set(0, 0, 0, 255);

				switch (m_params.m_dxgi_format)
				{
				case DXGI_FORMAT_BC1_UNORM:
				{
					const bool used_punchthrough = rgbcx::unpack_bc1(pBlock, unpacked_pixels, true, m_params.m_bc1_mode);

					if (used_punchthrough)
					{
						bc1_punchthrough_flag = true;

						const rgbcx::bc1_block* pBC1_block = (const rgbcx::bc1_block*)pBlock;

						for (uint32_t y = 0; y < 4; y++)
							for (uint32_t x = 0; x < 4; x++)
								if (pBC1_block->get_selector(x, y) == 3)
									used_bc1_transparent_texels_for_black = true;
					}

					break;
				}
				case DXGI_FORMAT_BC3_UNORM:
				{
					if (!rgbcx::unpack_bc3(pBlock, unpacked_pixels, m_params.m_bc1_mode))
						bc1_punchthrough_flag = true;
					break;
				}
				case DXGI_FORMAT_BC4_UNORM:
				{
					rgbcx::unpack_bc4(pBlock, &unpacked_pixels[0][0], 4);

#if DECODE_BC4_TO_GRAYSCALE
					for (uint32_t i = 0; i < 16; i++)
					{
						unpacked_pixels[i][1] = unpacked_pixels[i][0];
						unpacked_pixels[i][2] = unpacked_pixels[i][0];
					}
#endif
					break;
				}
				case DXGI_FORMAT_BC5_UNORM:
				{
					rgbcx::unpack_bc5(pBlock, &unpacked_pixels[0][0], 0, 1, 4);
					break;
				}
				case DXGI_FORMAT_BC7_UNORM:
				{
					if (!bc7decomp::unpack_bc7((const uint8_t*)pBlock, (bc7decomp::color_rgba*)unpacked_pixels))
					{
						fprintf(stderr, "bc7decomp::unpack_bc7() failed!\n");
						unpack_failed = true;
					}

					// Now unpack the block using the non-SSE reference decoder, to make sure we get the same exact unpacked bits.
					color_quad_u8 unpacked_pixels_ref[16];
					if (!bc7decomp_ref::unpack_bc7((const uint8_t*)pBlock, (bc7decomp::color_rgba*)unpacked_pixels_ref))
					{
						fprintf(stderr, "bc7decomp::unpack_bc7_ref() failed!\n");
						unpack_failed = true;
					}

					if (memcmp(unpacked_pixels, unpacked_pixels_ref, sizeof(unpacked_pixels)) != 0)
					{
						fprintf(stderr, "BC7 unpack verification failed!\n");
						unpack_failed = true;
					}

					break;
				}
				default:
					assert(0);
					break;
				}

				unpacked_image.set_block(bx, by, 4, 4, unpacked_pixels);
			} // bx
		} // by

		if (unpack_failed)
			return false;

		// Sanity check the BC1/BC3 output
		if (m_params.m_dxgi_format == DXGI_FORMAT_BC3_UNORM)
		{
			if (bc1_punchthrough_flag)
				fprintf(stderr, "WARNING: BC3 mode selected, but rgbcx::unpack_bc3() returned one or more blocks using 3-color mode!\n");
		}
		else if (m_params.m_dxgi_format == DXGI_FORMAT_BC1_UNORM)
		{
			if ((bc1_punchthrough_flag) && (!m_params.m_use_bc1_3color_mode))
				fprintf(stderr, "WARNING: BC1 output used 3-color mode, when this was disabled!\n");

			if ((used_bc1_transparent_texels_for_black) && (!used_bc1_transparent_texels_for_black))
				fprintf(stderr, "WARNING: BC1 output used the transparent selector for black, when this was disabled!\n");
		}

		if (m_params.m_status_output)
		{
			if ((m_params.m_dxgi_format == DXGI_FORMAT_BC1_UNORM) || (m_params.m_dxgi_format == DXGI_FORMAT_BC3_UNORM))
				printf("Output used 3-color mode: %u, output used transparent texels for black: %u\n", bc1_punchthrough_flag, used_bc1_transparent_texels_for_black);
		}

		return true;
	}

} // namespace rdo_bc
