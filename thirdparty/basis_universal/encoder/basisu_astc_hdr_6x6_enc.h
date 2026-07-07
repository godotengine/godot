// File: basisu_astc_hdr_6x6_enc.h
#pragma once
#include "basisu_enc.h"
#include "../transcoder/basisu_astc_hdr_core.h"

namespace astc_6x6_hdr
{
	const uint32_t ASTC_HDR_6X6_MAX_USER_COMP_LEVEL = 12;

	const uint32_t ASTC_HDR_6X6_MAX_COMP_LEVEL = 4;
	
	const float LDR_BLACK_BIAS = 0.0f;// .49f;
		
	// Note: This struct is copied several times, so do not place any heavyweight objects in here.
	struct astc_hdr_6x6_global_config
	{
		// Important: The Delta ITP colorspace error metric we use internally makes several assumptions about the nature of the HDR RGB inputs supplied to the encoder.
		// This encoder computes colorspace error in the ICtCp (or more accurately the delta ITP, where CT is scaled by .5 vs. ICtCp to become T) colorspace, so getting this correct is important.
		// By default the encoder assumes the input is in absolute luminance (in nits or candela per square meter, cd/m^2), specified as positive-only linear light RGB, using the REC 709 colorspace gamut (but NOT the sRGB transfer function, i.e. linear light).
		// If the m_rec2020_bt2100_color_gamut flag is true, the input colorspace is treated as REC 2020/BT.2100 (which is wider than 709).
		// For SDR/LDR->HDR upconversion, the REC 709 sRGB input should be converted to linear light (sRGB->linear) and the resulting normalized linear RGB values scaled by either 80 or 100 nits (the luminance of a typical SDR monitor). 
		// SDR upconversion to normalized [0,1] (i.e. non-absolute) luminances may work but is not supported because ITP errors will not be predicted correctly.
		bool m_rec2020_bt2100_color_gamut = false; 

		// levels 0-3 normal levels, 4=exhaustive
		uint32_t m_master_comp_level = 0;
		uint32_t m_highest_comp_level = 1;

		float m_lambda = 0.0f;

		bool m_extra_patterns_flag = false; // def to false, works in comp levels [1,4]
		bool m_brute_force_partition_matching = false; // def to false

		bool m_jnd_optimization = false; // defaults to false for HDR inputs, on SDR upconverted images this can default to enabled
		float m_jnd_delta_itp_thresh = .75f;

		bool m_force_one_strip = false;
				
		bool m_gaussian1_fallback = true; // def to true, if this is disabled m_gaussian2_fallback should be disabled too
		float m_gaussian1_strength = 1.45f;

		bool m_gaussian2_fallback = true; // def to true, hopefully rarely kicks in
		float m_gaussian2_strength = 1.83f;
				
		// m_disable_delta_endpoint_usage may give a slight increase in RDO ASTC encoding efficiency. It's also faster.
		bool m_disable_delta_endpoint_usage = false;

		// Scale up Delta ITP errors for very dark pixels, assuming they will be brightly exposed > 1.0x.
		// We don't know if the output will be exposed, or not. If heavily exposed, our JND calculations will not be conservative enough.
		bool m_delta_itp_dark_adjustment = true;

		bool m_debug_images = false;
		std::string m_debug_image_prefix = "dbg_astc_hdr_6x6_devel_";

		bool m_output_images = false;
		std::string m_output_image_prefix = "dbg_astc_hdr_6x6_output_";

		bool m_debug_output = false;
		bool m_image_stats = false;
		bool m_status_output = false;

		//-------------------------------------------------------------------------------------
		// Very low level/devel parameters - intended for development. Best not to change them.
		//-------------------------------------------------------------------------------------
		bool m_deblocking_flag = true;
		float m_deblock_penalty_weight = .03f;
		bool m_disable_twothree_subsets = false; // def to false
		bool m_use_solid_blocks = true; // def to true
		bool m_use_runs = true; // def to true
		bool m_block_stat_optimizations_flag = true; // def to true	

		bool m_rdo_candidate_diversity_boost = true; // def to true
		float m_rdo_candidate_diversity_boost_bit_window_weight = 1.2f;

		bool m_favor_higher_compression = true; // utilize all modes
		uint32_t m_num_reuse_xy_deltas = basist::astc_6x6_hdr::NUM_REUSE_XY_DELTAS;

		void print() const
		{
			basisu::fmt_debug_printf("m_master_comp_level: {}, m_highest_comp_level: {}\n", m_master_comp_level, m_highest_comp_level);
			basisu::fmt_debug_printf("m_lambda: {}\n", m_lambda);
			basisu::fmt_debug_printf("m_rec2020_bt2100_color_gamut: {}\n", m_rec2020_bt2100_color_gamut);
			basisu::fmt_debug_printf("m_extra_patterns_flag: {}, m_brute_force_partition_matching: {}\n", m_extra_patterns_flag, m_brute_force_partition_matching);
			basisu::fmt_debug_printf("m_jnd_optimization: {}, m_jnd_delta_itp_thresh: {}\n", m_jnd_optimization, m_jnd_delta_itp_thresh);
			basisu::fmt_debug_printf("m_force_one_strip: {}\n", m_force_one_strip);
			basisu::fmt_debug_printf("m_gaussian1_fallback: {}, m_gaussian1_strength: {}\n", m_gaussian1_fallback, m_gaussian1_strength);
			basisu::fmt_debug_printf("m_gaussian2_fallback: {}, m_gaussian2_strength: {}\n", m_gaussian2_fallback, m_gaussian2_strength);
			basisu::fmt_debug_printf("m_disable_delta_endpoint_usage: {}\n", m_disable_delta_endpoint_usage);
			basisu::fmt_debug_printf("m_delta_itp_dark_adjustment: {}\n", m_delta_itp_dark_adjustment);
			basisu::fmt_debug_printf("m_debug_images: {}, m_debug_image_prefix: {}\n", m_debug_images, m_debug_image_prefix);
			basisu::fmt_debug_printf("m_output_images: {}, m_output_image_prefix: {}\n", m_output_images, m_output_image_prefix);
			basisu::fmt_debug_printf("m_image_stats: {}, m_status_output: {}\n", m_image_stats, m_status_output);
			basisu::fmt_debug_printf("m_deblocking_flag: {}, m_deblock_penalty_weight: {}\n", m_deblocking_flag, m_deblock_penalty_weight);
			basisu::fmt_debug_printf("m_disable_twothree_subsets: {}, m_use_solid_blocks: {}\n", m_disable_twothree_subsets, m_use_solid_blocks);
			basisu::fmt_debug_printf("m_use_runs: {}, m_block_stat_optimizations_flag: {}\n", m_use_runs, m_block_stat_optimizations_flag);
			basisu::fmt_debug_printf("m_rdo_candidate_diversity_boost: {}, m_rdo_candidate_diversity_boost_bit_window_weight: {}\n", m_rdo_candidate_diversity_boost, m_rdo_candidate_diversity_boost_bit_window_weight);
			basisu::fmt_debug_printf("m_favor_higher_compression: {}, m_num_reuse_xy_deltas: {}\n", m_favor_higher_compression, m_num_reuse_xy_deltas);
		}
				
		astc_hdr_6x6_global_config()
		{
		}

		void clear()
		{
			astc_hdr_6x6_global_config def;
			std::swap(*this, def);
		}

		// Max level is ASTC_HDR_6X6_MAX_USER_COMP_LEVEL
		void set_user_level(int level);
	};

	void global_init();

	struct result_metrics
	{
		basisu::image_metrics m_im_astc_log2;
		basisu::image_metrics m_im_astc_half;

		basisu::image_metrics m_im_bc6h_log2;
		basisu::image_metrics m_im_bc6h_half;
	};
	
	// The input image should be unpadded to 6x6 boundaries, i.e. the original unexpanded image.
	bool compress_photo(const basisu::imagef& orig_src_img, const astc_hdr_6x6_global_config& global_cfg, basisu::job_pool* pJob_pool,
		basisu::uint8_vec& intermediate_tex_data, basisu::uint8_vec& astc_tex_data, result_metrics& metrics);

} // namespace uastc_6x6_hdr
