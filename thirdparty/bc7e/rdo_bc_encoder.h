// rdo_bc_encoder.h
#pragma once

#ifndef SUPPORT_BC7E
#define SUPPORT_BC7E 0
#endif

#include "utils.h"
#include "ert.h"

#include "bc7decomp.h"
#include "rgbcx.h"

#include "bc7enc.h"

#if SUPPORT_BC7E
#include "bc7e_ispc.h"
#endif

#include "dds_defs.h"

namespace rdo_bc
{

	struct rdo_bc_params
	{
		rdo_bc_params()
		{
			clear();
		}

		void clear()
		{
			m_bc7_uber_level = 6; // BC7ENC_MAX_UBER_LEVEL;
			m_bc7enc_max_partitions_to_scan = BC7ENC_MAX_PARTITIONS;
			m_perceptual = false;
			m_y_flip = false;
			m_bc45_channel0 = 0;
			m_bc45_channel1 = 1;

			m_bc1_mode = rgbcx::bc1_approx_mode::cBC1Ideal;
			m_use_bc1_3color_mode = true;

			// We're just turning this on by default now, like NVDXT.EXE used to do back in the old original Xbox days.
			m_use_bc1_3color_mode_for_black = true; // false; 

			m_bc1_quality_level = rgbcx::MAX_LEVEL;

			m_dxgi_format = DXGI_FORMAT_BC7_UNORM;

			m_rdo_lambda = 0.0f;
			m_rdo_debug_output = false;
			m_rdo_smooth_block_error_scale = 15.0f;
			m_custom_rdo_smooth_block_error_scale = false;
			m_lookback_window_size = 128;
			m_custom_lookback_window_size = false;
			m_bc7enc_rdo_bc7_quant_mode6_endpoints = true;
			m_bc7enc_rdo_bc7_weight_modes = true;
			m_bc7enc_rdo_bc7_weight_low_frequency_partitions = true;
			m_bc7enc_rdo_bc7_pbit1_weighting = true;
			m_rdo_max_smooth_block_std_dev = 18.0f;
			m_rdo_allow_relative_movement = false;
			m_rdo_try_2_matches = true;
			m_rdo_ultrasmooth_block_handling = true;

			m_use_hq_bc345 = true;
			m_bc345_search_rad = 5;
			m_bc345_mode_mask = rgbcx::BC4_USE_ALL_MODES;

			m_bc7enc_mode6_only = false;
			m_rdo_multithreading = true;

			m_bc7enc_reduce_entropy = false;

			m_use_bc7e = false;

#if SUPPORT_BC7E
			// By default, if they've compiled in BC7E.ispc, then use that. In a rate distortion sense it's better overall.
			// https://richg42.blogspot.com/2021/02/average-rate-distortion-curves-for.html
			m_use_bc7e = true;
#endif
						
			m_status_output = false;
			
			m_rdo_max_threads = 128;
		}

		int m_bc7_uber_level;
		int m_bc7enc_max_partitions_to_scan;
		bool m_perceptual;
		bool m_y_flip;
		uint32_t m_bc45_channel0;
		uint32_t m_bc45_channel1;

		rgbcx::bc1_approx_mode m_bc1_mode;
		bool m_use_bc1_3color_mode;

		bool m_use_bc1_3color_mode_for_black;

		int m_bc1_quality_level;

		DXGI_FORMAT m_dxgi_format;

		float m_rdo_lambda;
		bool m_rdo_debug_output;
		float m_rdo_smooth_block_error_scale;
		bool m_custom_rdo_smooth_block_error_scale;
		uint32_t m_lookback_window_size;
		bool m_custom_lookback_window_size;
		bool m_bc7enc_rdo_bc7_quant_mode6_endpoints;
		bool m_bc7enc_rdo_bc7_weight_modes;
		bool m_bc7enc_rdo_bc7_weight_low_frequency_partitions;
		bool m_bc7enc_rdo_bc7_pbit1_weighting;
		float m_rdo_max_smooth_block_std_dev;
		bool m_rdo_allow_relative_movement;
		bool m_rdo_try_2_matches;
		bool m_rdo_ultrasmooth_block_handling;

		bool m_use_hq_bc345;
		int m_bc345_search_rad;
		uint32_t m_bc345_mode_mask;

		bool m_bc7enc_mode6_only;
		bool m_rdo_multithreading;

		bool m_bc7enc_reduce_entropy;

		bool m_use_bc7e;
		bool m_status_output;
		
		uint32_t m_rdo_max_threads;
	};

	class rdo_bc_encoder
	{
	public:
		rdo_bc_encoder();

		void clear();

		bool init(const utils::image_u8& src_image, rdo_bc_params& params);
		bool encode();

		const rdo_bc_params &get_params() const { return m_params; }

		const utils::image_u8* get_orig_source_image() const { return m_pOrig_source_image; }
		const utils::image_u8& get_source_image() const { return m_source_image; }

		const void* get_prerdo_blocks() const { return m_prerdo_packed_image8.size() ? (void*)m_prerdo_packed_image8.data() : (void*)m_prerdo_packed_image16.data(); }
		const void* get_blocks() const { return m_packed_image8.size() ? (void*)m_packed_image8.data() : (void*)m_packed_image16.data(); }

		bool unpack_blocks(utils::image_u8& unpacked_image) const;

		DXGI_FORMAT get_pixel_format() const { return m_params.m_dxgi_format; }

		uint32_t get_orig_width() const { return m_orig_width; }
		uint32_t get_orig_height() const { return m_orig_height; }
		uint32_t get_blocks_x() const { return m_blocks_x; }
		uint32_t get_blocks_y() const { return m_blocks_y; }
		uint32_t get_total_blocks() const { return m_total_blocks; }
		uint32_t get_total_blocks_size_in_bytes() const { return m_total_blocks * m_bytes_per_block; }
		uint32_t get_bytes_per_block() const { return m_bytes_per_block; }
		uint32_t get_pixel_format_bpp() const { return m_pixel_format_bpp; }
		uint32_t get_total_texels() const { return m_total_texels; }
		bool get_has_alpha() const { return m_has_alpha; }
								
	private:
		const utils::image_u8* m_pOrig_source_image;
		utils::image_u8 m_source_image;
		rdo_bc_params m_params;

		uint32_t m_orig_width, m_orig_height;
		uint32_t m_blocks_x, m_blocks_y, m_total_blocks, m_bytes_per_block, m_pixel_format_bpp;
		uint32_t m_total_texels;
		bool m_has_alpha;

		utils::block8_vec m_packed_image8;
		utils::block16_vec m_packed_image16;

		utils::block8_vec m_prerdo_packed_image8;
		utils::block16_vec m_prerdo_packed_image16;

		bc7enc_compress_block_params m_bc7enc_pack_params;
#if SUPPORT_BC7E
		ispc::bc7e_compress_block_params m_bc7e_pack_params;
#endif

		void init_encoders();
		bool init_source_image();
		bool init_encoder_params();
		bool encode_texture();

		struct unpacker_funcs
		{
			rgbcx::bc1_approx_mode m_mode;
			bool m_allow_3color_mode;
			bool m_use_bc1_3color_mode_for_black;

			static bool unpack_bc1_block(const void* pBlock, ert::color_rgba* pPixels, uint32_t block_index, void* pUser_data)
			{
				(void)block_index;
				const unpacker_funcs* pState = (const unpacker_funcs*)pUser_data;

				bool used_3color_mode = rgbcx::unpack_bc1(pBlock, pPixels, true, pState->m_mode);

				if (used_3color_mode)
				{
					if (!pState->m_allow_3color_mode)
						return false;

					if (!pState->m_use_bc1_3color_mode_for_black)
					{
						rgbcx::bc1_block* pBC1_block = (rgbcx::bc1_block*)pBlock;

						for (uint32_t y = 0; y < 4; y++)
						{
							for (uint32_t x = 0; x < 4; x++)
							{
								if (pBC1_block->get_selector(x, y) == 3)
									return false;
							} // x
						} // y
					}
				}

				return true;
			}

			// TODO: Enforce 6/8 color constraints
			static bool unpack_bc4_block(const void* pBlock, ert::color_rgba* pPixels, uint32_t block_index, void* pUser_data)
			{
				(void)block_index;
				(void)pUser_data;
				memset(pPixels, 0, sizeof(ert::color_rgba) * 16);
				rgbcx::unpack_bc4(pBlock, (uint8_t*)pPixels, 4);
				return true;
			}

			static bool unpack_bc7_block(const void* pBlock, ert::color_rgba* pPixels, uint32_t block_index, void* pUser_data)
			{
				(void)block_index;
				(void)pUser_data;
				return bc7decomp::unpack_bc7(pBlock, (bc7decomp::color_rgba*)pPixels);
			}
		};

		bool postprocess_rdo();
	};
		
} // namespace rdo_bc
