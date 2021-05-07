// basisu_comp.h
// Copyright (C) 2019-2021 Binomial LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "basisu_frontend.h"
#include "basisu_backend.h"
#include "basisu_basis_file.h"
#include "../transcoder/basisu_global_selector_palette.h"
#include "../transcoder/basisu_transcoder.h"
#include "basisu_uastc_enc.h"

#define BASISU_LIB_VERSION 115
#define BASISU_LIB_VERSION_STRING "1.15"

#ifndef BASISD_SUPPORT_KTX2
	#error BASISD_SUPPORT_KTX2 is undefined
#endif
#ifndef BASISD_SUPPORT_KTX2_ZSTD
	#error BASISD_SUPPORT_KTX2_ZSTD is undefined
#endif

#if !BASISD_SUPPORT_KTX2
	#error BASISD_SUPPORT_KTX2 must be enabled when building the encoder. To reduce code size if KTX2 support is not needed, set BASISD_SUPPORT_KTX2_ZSTD to 0
#endif

namespace basisu
{
	const uint32_t BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION = 16384;

	// Allow block's color distance to increase by 1.5 while searching for an alternative nearby endpoint.
	const float BASISU_DEFAULT_ENDPOINT_RDO_THRESH = 1.5f; 
	
	// Allow block's color distance to increase by 1.25 while searching the selector history buffer for a close enough match.
	const float BASISU_DEFAULT_SELECTOR_RDO_THRESH = 1.25f; 

	const int BASISU_DEFAULT_QUALITY = 128;
	const float BASISU_DEFAULT_HYBRID_SEL_CB_QUALITY_THRESH = 2.0f;

	const uint32_t BASISU_MAX_IMAGE_DIMENSION = 16384;
	const uint32_t BASISU_QUALITY_MIN = 1;
	const uint32_t BASISU_QUALITY_MAX = 255;
	const uint32_t BASISU_MAX_ENDPOINT_CLUSTERS = basisu_frontend::cMaxEndpointClusters;
	const uint32_t BASISU_MAX_SELECTOR_CLUSTERS = basisu_frontend::cMaxSelectorClusters;

	const uint32_t BASISU_MAX_SLICES = 0xFFFFFF;

	const int BASISU_RDO_UASTC_DICT_SIZE_DEFAULT = 4096; // 32768;
	const int BASISU_RDO_UASTC_DICT_SIZE_MIN = 64;
	const int BASISU_RDO_UASTC_DICT_SIZE_MAX = 65536;

	struct image_stats
	{
		image_stats()
		{
			clear();
		}

		void clear()
		{
			m_filename.clear();
			m_width = 0;
			m_height = 0;
						
			m_basis_rgb_avg_psnr = 0.0f;
			m_basis_rgba_avg_psnr = 0.0f;
			m_basis_a_avg_psnr = 0.0f;
			m_basis_luma_709_psnr = 0.0f;
			m_basis_luma_601_psnr = 0.0f;
			m_basis_luma_709_ssim = 0.0f;

			m_bc7_rgb_avg_psnr = 0.0f;
			m_bc7_rgba_avg_psnr = 0.0f;
			m_bc7_a_avg_psnr = 0.0f;
			m_bc7_luma_709_psnr = 0.0f;
			m_bc7_luma_601_psnr = 0.0f;
			m_bc7_luma_709_ssim = 0.0f;
						
			m_best_etc1s_rgb_avg_psnr = 0.0f;
			m_best_etc1s_luma_709_psnr = 0.0f;
			m_best_etc1s_luma_601_psnr = 0.0f;
			m_best_etc1s_luma_709_ssim = 0.0f;
		}

		std::string m_filename;
		uint32_t m_width;
		uint32_t m_height;

		// .basis compressed (ETC1S or UASTC statistics)
		float m_basis_rgb_avg_psnr;
		float m_basis_rgba_avg_psnr;
		float m_basis_a_avg_psnr;
		float m_basis_luma_709_psnr;
		float m_basis_luma_601_psnr;
		float m_basis_luma_709_ssim;

		// BC7 statistics
		float m_bc7_rgb_avg_psnr;
		float m_bc7_rgba_avg_psnr;
		float m_bc7_a_avg_psnr;
		float m_bc7_luma_709_psnr;
		float m_bc7_luma_601_psnr;
		float m_bc7_luma_709_ssim;
		
		// Highest achievable quality ETC1S statistics
		float m_best_etc1s_rgb_avg_psnr;
		float m_best_etc1s_luma_709_psnr;
		float m_best_etc1s_luma_601_psnr;
		float m_best_etc1s_luma_709_ssim;
	};

	template<bool def>
	struct bool_param
	{
		bool_param() :
			m_value(def),
			m_changed(false)
		{
		}

		void clear()
		{
			m_value = def;
			m_changed = false;
		}

		operator bool() const
		{
			return m_value;
		}

		bool operator= (bool v)
		{
			m_value = v;
			m_changed = true;
			return m_value;
		}

		bool was_changed() const { return m_changed; }
		void set_changed(bool flag) { m_changed = flag; }

		bool m_value;
		bool m_changed;
	};

	template<typename T>
	struct param
	{
		param(T def, T min_v, T max_v) :
			m_value(def),
			m_def(def),
			m_min(min_v),
			m_max(max_v),
			m_changed(false)
		{
		}

		void clear()
		{
			m_value = m_def;
			m_changed = false;
		}

		operator T() const
		{
			return m_value;
		}

		T operator= (T v)
		{
			m_value = clamp<T>(v, m_min, m_max);
			m_changed = true;
			return m_value;
		}

		T operator *= (T v)
		{
			m_value *= v;
			m_changed = true;
			return m_value;
		}

		bool was_changed() const { return m_changed; }
		void set_changed(bool flag) { m_changed = flag; }

		T m_value;
		T m_def;
		T m_min;
		T m_max;
		bool m_changed;
	};

	struct basis_compressor_params
	{
		basis_compressor_params() :
			m_pSel_codebook(NULL),
			m_compression_level((int)BASISU_DEFAULT_COMPRESSION_LEVEL, 0, (int)BASISU_MAX_COMPRESSION_LEVEL),
			m_selector_rdo_thresh(BASISU_DEFAULT_SELECTOR_RDO_THRESH, 0.0f, 1e+10f),
			m_endpoint_rdo_thresh(BASISU_DEFAULT_ENDPOINT_RDO_THRESH, 0.0f, 1e+10f),
			m_hybrid_sel_cb_quality_thresh(BASISU_DEFAULT_HYBRID_SEL_CB_QUALITY_THRESH, 0.0f, 1e+10f),
			m_global_pal_bits(8, 0, ETC1_GLOBAL_SELECTOR_CODEBOOK_MAX_PAL_BITS),
			m_global_mod_bits(8, 0, basist::etc1_global_palette_entry_modifier::cTotalBits),
			m_mip_scale(1.0f, .000125f, 4.0f),
			m_mip_smallest_dimension(1, 1, 16384),
			m_max_endpoint_clusters(512),
			m_max_selector_clusters(512),
			m_quality_level(-1),
			m_pack_uastc_flags(cPackUASTCLevelDefault),
			m_rdo_uastc_quality_scalar(1.0f, 0.001f, 50.0f),
			m_rdo_uastc_dict_size(BASISU_RDO_UASTC_DICT_SIZE_DEFAULT, BASISU_RDO_UASTC_DICT_SIZE_MIN, BASISU_RDO_UASTC_DICT_SIZE_MAX),
			m_rdo_uastc_max_smooth_block_error_scale(UASTC_RDO_DEFAULT_SMOOTH_BLOCK_MAX_ERROR_SCALE, 1.0f, 300.0f),
			m_rdo_uastc_smooth_block_max_std_dev(UASTC_RDO_DEFAULT_MAX_SMOOTH_BLOCK_STD_DEV, .01f, 65536.0f),
			m_rdo_uastc_max_allowed_rms_increase_ratio(UASTC_RDO_DEFAULT_MAX_ALLOWED_RMS_INCREASE_RATIO, .01f, 100.0f),
			m_rdo_uastc_skip_block_rms_thresh(UASTC_RDO_DEFAULT_SKIP_BLOCK_RMS_THRESH, .01f, 100.0f),
			m_resample_width(0, 1, 16384),
			m_resample_height(0, 1, 16384),
			m_resample_factor(0.0f, .00125f, 100.0f),
			m_ktx2_uastc_supercompression(basist::KTX2_SS_NONE),
			m_ktx2_zstd_supercompression_level(6, INT_MIN, INT_MAX),
			m_pJob_pool(nullptr)
		{
			clear();
		}

		void clear()
		{
			m_pSel_codebook = NULL;

			m_uastc.clear();
			m_status_output.clear();

			m_source_filenames.clear();
			m_source_alpha_filenames.clear();

			m_source_images.clear();
			m_source_mipmap_images.clear();

			m_out_filename.clear();

			m_y_flip.clear();
			m_debug.clear();
			m_validate.clear();
			m_debug_images.clear();
			m_global_sel_pal.clear();
			m_auto_global_sel_pal.clear();
			m_no_hybrid_sel_cb.clear();
			m_perceptual.clear();
			m_no_selector_rdo.clear();
			m_selector_rdo_thresh.clear();
			m_read_source_images.clear();
			m_write_output_basis_files.clear();
			m_compression_level.clear();
			m_compute_stats.clear();
			m_check_for_alpha.clear();
			m_force_alpha.clear();
			m_multithreading.clear();
			m_swizzle[0] = 0;
			m_swizzle[1] = 1;
			m_swizzle[2] = 2;
			m_swizzle[3] = 3;
			m_renormalize.clear();
			m_hybrid_sel_cb_quality_thresh.clear();
			m_global_pal_bits.clear();
			m_global_mod_bits.clear();
			m_disable_hierarchical_endpoint_codebooks.clear();

			m_no_endpoint_rdo.clear();
			m_endpoint_rdo_thresh.clear();
						
			m_mip_gen.clear();
			m_mip_scale.clear();
			m_mip_filter = "kaiser";
			m_mip_scale = 1.0f;
			m_mip_srgb.clear();
			m_mip_premultiplied.clear();
			m_mip_renormalize.clear();
			m_mip_wrapping.clear();
			m_mip_fast.clear();
			m_mip_smallest_dimension.clear();

			m_max_endpoint_clusters = 0;
			m_max_selector_clusters = 0;
			m_quality_level = -1;

			m_tex_type = basist::cBASISTexType2D;
			m_userdata0 = 0;
			m_userdata1 = 0;
			m_us_per_frame = 0;

			m_pack_uastc_flags = cPackUASTCLevelDefault;
			m_rdo_uastc.clear();
			m_rdo_uastc_quality_scalar.clear();
			m_rdo_uastc_max_smooth_block_error_scale.clear();
			m_rdo_uastc_smooth_block_max_std_dev.clear();
			m_rdo_uastc_max_allowed_rms_increase_ratio.clear();
			m_rdo_uastc_skip_block_rms_thresh.clear();
			m_rdo_uastc_favor_simpler_modes_in_rdo_mode.clear();
			m_rdo_uastc_multithreading.clear();

			m_resample_width.clear();
			m_resample_height.clear();
			m_resample_factor.clear();

			m_pGlobal_codebooks = nullptr;

			m_create_ktx2_file.clear();
			m_ktx2_uastc_supercompression = basist::KTX2_SS_NONE;
			m_ktx2_key_values.clear();
			m_ktx2_zstd_supercompression_level.clear();
			m_ktx2_srgb_transfer_func.clear();

			m_pJob_pool = nullptr;
		}
				
		// Pointer to the global selector codebook, or nullptr to not use a global selector codebook
		const basist::etc1_global_selector_codebook *m_pSel_codebook;

		// True to generate UASTC .basis file data, otherwise ETC1S.
		bool_param<false> m_uastc;

		// If m_read_source_images is true, m_source_filenames (and optionally m_source_alpha_filenames) contains the filenames of PNG images to read. 
		// Otherwise, the compressor processes the images in m_source_images.
		basisu::vector<std::string> m_source_filenames;
		basisu::vector<std::string> m_source_alpha_filenames;
		
		basisu::vector<image> m_source_images;
		
		// Stores mipmaps starting from level 1. Level 0 is still stored in m_source_images, as usual.
		// If m_source_mipmaps isn't empty, automatic mipmap generation isn't done. m_source_mipmaps.size() MUST equal m_source_images.size() or the compressor returns an error.
		// The compressor applies the user-provided swizzling (in m_swizzle) to these images.
		basisu::vector< basisu::vector<image> > m_source_mipmap_images;
						
		// Filename of the output basis file
		std::string m_out_filename;

		// The params are done this way so we can detect when the user has explictly changed them.

		// Flip images across Y axis
		bool_param<false> m_y_flip;

		// If true, the compressor will print basis status to stdout during compression.
		bool_param<true> m_status_output;
		
		// Output debug information during compression
		bool_param<false> m_debug;
		bool_param<false> m_validate;
		
		// m_debug_images is pretty slow
		bool_param<false> m_debug_images;

		// Compression level, from 0 to BASISU_MAX_COMPRESSION_LEVEL (higher is slower)
		param<int> m_compression_level;

		bool_param<false> m_global_sel_pal;
		bool_param<false> m_auto_global_sel_pal;

		// Frontend/backend codec parameters
		bool_param<false> m_no_hybrid_sel_cb;
		
		// Use perceptual sRGB colorspace metrics instead of linear
		bool_param<true> m_perceptual;

		// Disable selector RDO, for faster compression but larger files
		bool_param<false> m_no_selector_rdo;
		param<float> m_selector_rdo_thresh;

		bool_param<false> m_no_endpoint_rdo;
		param<float> m_endpoint_rdo_thresh;

		// Read source images from m_source_filenames/m_source_alpha_filenames
		bool_param<false> m_read_source_images;

		// Write the output basis file to disk using m_out_filename
		bool_param<false> m_write_output_basis_files;
								
		// Compute and display image metrics 
		bool_param<false> m_compute_stats;
		
		// Check to see if any input image has an alpha channel, if so then the output basis file will have alpha channels
		bool_param<true> m_check_for_alpha;
		
		// Always put alpha slices in the output basis file, even when the input doesn't have alpha
		bool_param<false> m_force_alpha; 
		bool_param<true> m_multithreading;
		
		// Split the R channel to RGB and the G channel to alpha, then write a basis file with alpha channels
		char m_swizzle[4];

		bool_param<false> m_renormalize;

		bool_param<false> m_disable_hierarchical_endpoint_codebooks;

		// Global/hybrid selector codebook parameters
		param<float> m_hybrid_sel_cb_quality_thresh;
		param<int> m_global_pal_bits;
		param<int> m_global_mod_bits;
		
		// mipmap generation parameters
		bool_param<false> m_mip_gen;
		param<float> m_mip_scale;
		std::string m_mip_filter;
		bool_param<false> m_mip_srgb;
		bool_param<true> m_mip_premultiplied; // not currently supported
		bool_param<false> m_mip_renormalize; 
		bool_param<true> m_mip_wrapping;
		bool_param<true> m_mip_fast;
		param<int> m_mip_smallest_dimension;
				
		// Codebook size (quality) control. 
		// If m_quality_level != -1, it controls the quality level. It ranges from [0,255] or [BASISU_QUALITY_MIN, BASISU_QUALITY_MAX].
		// Otherwise m_max_endpoint_clusters/m_max_selector_clusters controls the codebook sizes directly.
		uint32_t m_max_endpoint_clusters;
		uint32_t m_max_selector_clusters;
		int m_quality_level;
		
		// m_tex_type, m_userdata0, m_userdata1, m_framerate - These fields go directly into the Basis file header.
		basist::basis_texture_type m_tex_type;
		uint32_t m_userdata0;
		uint32_t m_userdata1;
		uint32_t m_us_per_frame;

		// cPackUASTCLevelDefault, etc.
		uint32_t m_pack_uastc_flags;
		bool_param<false> m_rdo_uastc;
		param<float> m_rdo_uastc_quality_scalar;
		param<int> m_rdo_uastc_dict_size;
		param<float> m_rdo_uastc_max_smooth_block_error_scale;
		param<float> m_rdo_uastc_smooth_block_max_std_dev;
		param<float> m_rdo_uastc_max_allowed_rms_increase_ratio;
		param<float> m_rdo_uastc_skip_block_rms_thresh;
		bool_param<true> m_rdo_uastc_favor_simpler_modes_in_rdo_mode;
		bool_param<true> m_rdo_uastc_multithreading;

		param<int> m_resample_width;
		param<int> m_resample_height;
		param<float> m_resample_factor;
		const basist::basisu_lowlevel_etc1s_transcoder *m_pGlobal_codebooks;

		// KTX2 specific parameters.
		// Internally, the compressor always creates a .basis file then it converts that lossless to KTX2.
		bool_param<false> m_create_ktx2_file;
		basist::ktx2_supercompression m_ktx2_uastc_supercompression;
		basist::ktx2_transcoder::key_value_vec m_ktx2_key_values;
		param<int> m_ktx2_zstd_supercompression_level;
		bool_param<false> m_ktx2_srgb_transfer_func;

		job_pool *m_pJob_pool;
	};
	
	class basis_compressor
	{
		BASISU_NO_EQUALS_OR_COPY_CONSTRUCT(basis_compressor);

	public:
		basis_compressor();

		bool init(const basis_compressor_params &params);
		
		enum error_code
		{
			cECSuccess = 0,
			cECFailedReadingSourceImages,
			cECFailedValidating,
			cECFailedEncodeUASTC,
			cECFailedFrontEnd,
			cECFailedFontendExtract,
			cECFailedBackend,
			cECFailedCreateBasisFile,
			cECFailedWritingOutput,
			cECFailedUASTCRDOPostProcess,
			cECFailedCreateKTX2File
		};

		error_code process();

		// The output .basis file will always be valid of process() succeeded.
		const uint8_vec &get_output_basis_file() const { return m_output_basis_file; }
		
		// The output .ktx2 file will only be valid if m_create_ktx2_file was true and process() succeeded.
		const uint8_vec& get_output_ktx2_file() const { return m_output_ktx2_file; }

		const basisu::vector<image_stats> &get_stats() const { return m_stats; }

		uint32_t get_basis_file_size() const { return m_basis_file_size; }
		double get_basis_bits_per_texel() const { return m_basis_bits_per_texel; }
		
		bool get_any_source_image_has_alpha() const { return m_any_source_image_has_alpha; }
								
	private:
		basis_compressor_params m_params;
		
		basisu::vector<image> m_slice_images;

		basisu::vector<image_stats> m_stats;

		uint32_t m_basis_file_size;
		double m_basis_bits_per_texel;
						
		basisu_backend_slice_desc_vec m_slice_descs;

		uint32_t m_total_blocks;
		bool m_auto_global_sel_pal;

		basisu_frontend m_frontend;
		pixel_block_vec m_source_blocks;

		basisu::vector<gpu_image> m_frontend_output_textures;

		basisu::vector<gpu_image> m_best_etc1s_images;
		basisu::vector<image> m_best_etc1s_images_unpacked;

		basisu_backend m_backend;

		basisu_file m_basis_file;

		basisu::vector<gpu_image> m_decoded_output_textures;
		basisu::vector<image> m_decoded_output_textures_unpacked;
		basisu::vector<gpu_image> m_decoded_output_textures_bc7;
		basisu::vector<image> m_decoded_output_textures_unpacked_bc7;

		uint8_vec m_output_basis_file;
		uint8_vec m_output_ktx2_file;
		
		basisu::vector<gpu_image> m_uastc_slice_textures;
		basisu_backend_output m_uastc_backend_output;

		bool m_any_source_image_has_alpha;

		bool read_source_images();
		bool extract_source_blocks();
		bool process_frontend();
		bool extract_frontend_texture_data();
		bool process_backend();
		bool create_basis_file_and_transcode();
		bool write_output_files_and_compute_stats();
		error_code encode_slices_to_uastc();
		bool generate_mipmaps(const image &img, basisu::vector<image> &mips, bool has_alpha);
		bool validate_texture_type_constraints();
		bool validate_ktx2_constraints();
		void get_dfd(uint8_vec& dfd, const basist::ktx2_header& hdr);
		bool create_ktx2_file();
	};

} // namespace basisu

