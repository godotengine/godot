// basisu_comp.h
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
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
#include "transcoder/basisu_global_selector_palette.h"
#include "transcoder/basisu_transcoder.h"

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

			m_basis_etc1s_rgb_avg_psnr = 0.0f;
			m_basis_etc1s_luma_709_psnr = 0.0f;
			m_basis_etc1s_luma_601_psnr = 0.0f;
			m_basis_etc1s_luma_709_ssim = 0.0f;

			m_basis_bc1_rgb_avg_psnr = 0.0f;
			m_basis_bc1_luma_709_psnr = 0.0f;
			m_basis_bc1_luma_601_psnr = 0.0f;
			m_basis_bc1_luma_709_ssim = 0.0f;

			m_best_rgb_avg_psnr = 0.0f;
			m_best_luma_709_psnr = 0.0f;
			m_best_luma_601_psnr = 0.0f;
			m_best_luma_709_ssim = 0.0f;
		}

		std::string m_filename;
		uint32_t m_width;
		uint32_t m_height;

		// .basis compressed
		float m_basis_etc1s_rgb_avg_psnr;
		float m_basis_etc1s_luma_709_psnr;
		float m_basis_etc1s_luma_601_psnr;
		float m_basis_etc1s_luma_709_ssim;
		
		float m_basis_bc1_rgb_avg_psnr;
		float m_basis_bc1_luma_709_psnr;
		float m_basis_bc1_luma_601_psnr;
		float m_basis_bc1_luma_709_ssim;

		// Normal (highest quality) compressed ETC1S
		float m_best_rgb_avg_psnr;
		float m_best_luma_709_psnr;
		float m_best_luma_601_psnr;
		float m_best_luma_709_ssim;
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
			m_hybrid_sel_cb_quality_thresh(BASISU_DEFAULT_HYBRID_SEL_CB_QUALITY_THRESH, 0.0f, 1e+10f),
			m_global_pal_bits(8, 0, ETC1_GLOBAL_SELECTOR_CODEBOOK_MAX_PAL_BITS),
			m_global_mod_bits(8, 0, basist::etc1_global_palette_entry_modifier::cTotalBits),
			m_endpoint_rdo_thresh(BASISU_DEFAULT_ENDPOINT_RDO_THRESH, 0.0f, 1e+10f),
			m_selector_rdo_thresh(BASISU_DEFAULT_SELECTOR_RDO_THRESH, 0.0f, 1e+10f),
			m_pSel_codebook(NULL),
			m_max_endpoint_clusters(512),
			m_max_selector_clusters(512),
			m_quality_level(-1),
			m_mip_scale(1.0f, .000125f, 4.0f),
			m_mip_smallest_dimension(1, 1, 16384),
			m_compression_level((int)BASISU_DEFAULT_COMPRESSION_LEVEL, 0, (int)BASISU_MAX_COMPRESSION_LEVEL),
			m_pJob_pool(nullptr)
		{
			clear();
		}

		void clear()
		{
			m_pSel_codebook = NULL;

			m_source_filenames.clear();
			m_source_alpha_filenames.clear();

			m_source_images.clear();

			m_out_filename.clear();

			m_y_flip.clear();
			m_debug.clear();
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
			m_seperate_rg_to_color_alpha.clear();
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
			m_mip_smallest_dimension.clear();

			m_max_endpoint_clusters = 0;
			m_max_selector_clusters = 0;
			m_quality_level = -1;

			m_tex_type = basist::cBASISTexType2D;
			m_userdata0 = 0;
			m_userdata1 = 0;
			m_us_per_frame = 0;

			m_pJob_pool = nullptr;
		}

		// Pointer to the global selector codebook, or nullptr to not use a global selector codebook
		const basist::etc1_global_selector_codebook *m_pSel_codebook;

		// If m_read_source_images is true, m_source_filenames (and optionally m_source_alpha_filenames) contains the filenames of PNG images to read. 
		// Otherwise, the compressor processes the images in m_source_images.
		std::vector<std::string> m_source_filenames;
		std::vector<std::string> m_source_alpha_filenames;
		
		std::vector<image> m_source_images;
		// TODO: Allow caller to supply their own mipmaps
						
		// Filename of the output basis file
		std::string m_out_filename;	

		// The params are done this way so we can detect when the user has explictly changed them.

		// Flip images across Y axis
		bool_param<false> m_y_flip;
		
		// Output debug information during compression
		bool_param<false> m_debug;
		
		// m_debug_images is pretty slow
		bool_param<false> m_debug_images;

		// Compression level, from 0 to BASISU_MAX_COMPRESSION_LEVEL (higher is slower)
		param<int> m_compression_level;

		bool_param<false> m_global_sel_pal;
		bool_param<false> m_auto_global_sel_pal;

		// Frontend/backend codec parameters
		bool_param<false> m_no_hybrid_sel_cb;
		
		// Use perceptual sRGB colorspace metrics (for normal maps, etc.)
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
		bool_param<false> m_seperate_rg_to_color_alpha;

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
		param<int> m_mip_smallest_dimension;
				
		// Codebook size (quality) control. 
		// If m_quality_level != -1, it controls the quality level. It ranges from [0,255].
		// Otherwise m_max_endpoint_clusters/m_max_selector_clusters controls the codebook sizes directly.
		uint32_t m_max_endpoint_clusters;
		uint32_t m_max_selector_clusters;
		int m_quality_level;
		
		// m_tex_type, m_userdata0, m_userdata1, m_framerate - These fields go directly into the Basis file header.
		basist::basis_texture_type m_tex_type;
		uint32_t m_userdata0;
		uint32_t m_userdata1;
		uint32_t m_us_per_frame;

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
			cECFailedFrontEnd,
			cECFailedFontendExtract,
			cECFailedBackend,
			cECFailedCreateBasisFile,
			cECFailedWritingOutput
		};

		error_code process();

		const uint8_vec &get_output_basis_file() const { return m_output_basis_file; }
		const etc_block_vec &get_output_blocks() const { return m_output_blocks; }

		const std::vector<image_stats> &get_stats() const { return m_stats; }

		uint32_t get_basis_file_size() const { return m_basis_file_size; }
		double get_basis_bits_per_texel() const { return m_basis_bits_per_texel; }

		bool get_any_source_image_has_alpha() const { return m_any_source_image_has_alpha; }

	private:
		basis_compressor_params m_params;
		
		std::vector<image> m_slice_images;

		std::vector<image_stats> m_stats;

		uint32_t m_basis_file_size;
		double m_basis_bits_per_texel;
		
		basisu_backend_slice_desc_vec m_slice_descs;

		uint32_t m_total_blocks;
		bool m_auto_global_sel_pal;

		basisu_frontend m_frontend;
		pixel_block_vec m_source_blocks;

		std::vector<gpu_image> m_frontend_output_textures;

		std::vector<gpu_image> m_best_etc1s_images;
		std::vector<image> m_best_etc1s_images_unpacked;

		basisu_backend m_backend;

		basisu_file m_basis_file;

		std::vector<gpu_image> m_decoded_output_textures;
		std::vector<image> m_decoded_output_textures_unpacked;
		std::vector<gpu_image> m_decoded_output_textures_bc1;
		std::vector<image> m_decoded_output_textures_unpacked_bc1;

		uint8_vec m_output_basis_file;
		etc_block_vec m_output_blocks;

		bool m_any_source_image_has_alpha;

		bool read_source_images();
		bool process_frontend();
		bool extract_frontend_texture_data();
		bool process_backend();
		bool create_basis_file_and_transcode();
		bool write_output_files_and_compute_stats();
		bool generate_mipmaps(const image &img, std::vector<image> &mips, bool has_alpha);
		bool validate_texture_type_constraints();
	};

} // namespace basisu

