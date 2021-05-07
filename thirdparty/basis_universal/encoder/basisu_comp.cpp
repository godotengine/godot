// basisu_comp.cpp
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
#include "basisu_comp.h"
#include "basisu_enc.h"
#include <unordered_set>
#include <atomic>

// basisu_transcoder.cpp is where basisu_miniz lives now, we just need the declarations here.
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include "basisu_miniz.h"

#if !BASISD_SUPPORT_KTX2
#error BASISD_SUPPORT_KTX2 must be enabled (set to 1).
#endif

#if BASISD_SUPPORT_KTX2_ZSTD
#include "../zstd/zstd.h"
#endif

// Set to 1 to disable the mipPadding alignment workaround (which only seems to be needed when no key-values are written at all)
#define BASISU_DISABLE_KTX2_ALIGNMENT_WORKAROUND (0)

// Set to 1 to disable writing all KTX2 key values, triggering the validator bug.
#define BASISU_DISABLE_KTX2_KEY_VALUES (0)

using namespace buminiz;

#define BASISU_USE_STB_IMAGE_RESIZE_FOR_MIPMAP_GEN 0
#define DEBUG_CROP_TEXTURE_TO_64x64 (0)
#define DEBUG_RESIZE_TEXTURE (0)
#define DEBUG_EXTRACT_SINGLE_BLOCK (0)

namespace basisu
{
   basis_compressor::basis_compressor() :
		m_basis_file_size(0),
		m_basis_bits_per_texel(0.0f),
		m_total_blocks(0),
		m_auto_global_sel_pal(false),
		m_any_source_image_has_alpha(false)
	{
		debug_printf("basis_compressor::basis_compressor\n");
	}

	bool basis_compressor::init(const basis_compressor_params &params)
	{
		debug_printf("basis_compressor::init\n");

		m_params = params;

		if (m_params.m_debug)
		{
			debug_printf("basis_compressor::init:\n");

#define PRINT_BOOL_VALUE(v) debug_printf("%s: %u %u\n", BASISU_STRINGIZE2(v), static_cast<int>(m_params.v), m_params.v.was_changed());
#define PRINT_INT_VALUE(v) debug_printf("%s: %i %u\n", BASISU_STRINGIZE2(v), static_cast<int>(m_params.v), m_params.v.was_changed());
#define PRINT_UINT_VALUE(v) debug_printf("%s: %u %u\n", BASISU_STRINGIZE2(v), static_cast<uint32_t>(m_params.v), m_params.v.was_changed());
#define PRINT_FLOAT_VALUE(v) debug_printf("%s: %f %u\n", BASISU_STRINGIZE2(v), static_cast<float>(m_params.v), m_params.v.was_changed());

			debug_printf("Has global selector codebook: %i\n", m_params.m_pSel_codebook != nullptr);

			debug_printf("Source images: %u, source filenames: %u, source alpha filenames: %i, Source mipmap images: %u\n",
				m_params.m_source_images.size(), m_params.m_source_filenames.size(), m_params.m_source_alpha_filenames.size(), m_params.m_source_mipmap_images.size());

			if (m_params.m_source_mipmap_images.size())
			{
				debug_printf("m_source_mipmap_images array sizes:\n");
				for (uint32_t i = 0; i < m_params.m_source_mipmap_images.size(); i++)
					debug_printf("%u ", m_params.m_source_mipmap_images[i].size());
				debug_printf("\n");
			}

			PRINT_BOOL_VALUE(m_uastc);
			PRINT_BOOL_VALUE(m_y_flip);
			PRINT_BOOL_VALUE(m_debug);
			PRINT_BOOL_VALUE(m_validate);
			PRINT_BOOL_VALUE(m_debug_images);
			PRINT_BOOL_VALUE(m_global_sel_pal);
			PRINT_BOOL_VALUE(m_auto_global_sel_pal);
			PRINT_INT_VALUE(m_compression_level);
			PRINT_BOOL_VALUE(m_no_hybrid_sel_cb);
			PRINT_BOOL_VALUE(m_perceptual);
			PRINT_BOOL_VALUE(m_no_endpoint_rdo);
			PRINT_BOOL_VALUE(m_no_selector_rdo);
			PRINT_BOOL_VALUE(m_read_source_images);
			PRINT_BOOL_VALUE(m_write_output_basis_files);
			PRINT_BOOL_VALUE(m_compute_stats);
			PRINT_BOOL_VALUE(m_check_for_alpha);
			PRINT_BOOL_VALUE(m_force_alpha);
			debug_printf("swizzle: %d,%d,%d,%d\n",
				m_params.m_swizzle[0],
				m_params.m_swizzle[1],
				m_params.m_swizzle[2],
				m_params.m_swizzle[3]);
			PRINT_BOOL_VALUE(m_renormalize);
			PRINT_BOOL_VALUE(m_multithreading);
			PRINT_BOOL_VALUE(m_disable_hierarchical_endpoint_codebooks);
			
			PRINT_FLOAT_VALUE(m_hybrid_sel_cb_quality_thresh);
			
			PRINT_INT_VALUE(m_global_pal_bits);
			PRINT_INT_VALUE(m_global_mod_bits);

			PRINT_FLOAT_VALUE(m_endpoint_rdo_thresh);
			PRINT_FLOAT_VALUE(m_selector_rdo_thresh);
			
			PRINT_BOOL_VALUE(m_mip_gen);
			PRINT_BOOL_VALUE(m_mip_renormalize);
			PRINT_BOOL_VALUE(m_mip_wrapping);
			PRINT_BOOL_VALUE(m_mip_fast);
			PRINT_BOOL_VALUE(m_mip_srgb);
			PRINT_FLOAT_VALUE(m_mip_premultiplied);
			PRINT_FLOAT_VALUE(m_mip_scale);
			PRINT_INT_VALUE(m_mip_smallest_dimension);
			debug_printf("m_mip_filter: %s\n", m_params.m_mip_filter.c_str());

			debug_printf("m_max_endpoint_clusters: %u\n", m_params.m_max_endpoint_clusters);
			debug_printf("m_max_selector_clusters: %u\n", m_params.m_max_selector_clusters);
			debug_printf("m_quality_level: %i\n", m_params.m_quality_level);

			debug_printf("m_tex_type: %u\n", m_params.m_tex_type);
			debug_printf("m_userdata0: 0x%X, m_userdata1: 0x%X\n", m_params.m_userdata0, m_params.m_userdata1);
			debug_printf("m_us_per_frame: %i (%f fps)\n", m_params.m_us_per_frame, m_params.m_us_per_frame ? 1.0f / (m_params.m_us_per_frame / 1000000.0f) : 0);
			debug_printf("m_pack_uastc_flags: 0x%X\n", m_params.m_pack_uastc_flags);
			
			PRINT_BOOL_VALUE(m_rdo_uastc);
			PRINT_FLOAT_VALUE(m_rdo_uastc_quality_scalar);
			PRINT_INT_VALUE(m_rdo_uastc_dict_size);
			PRINT_FLOAT_VALUE(m_rdo_uastc_max_allowed_rms_increase_ratio);
			PRINT_FLOAT_VALUE(m_rdo_uastc_skip_block_rms_thresh);
			PRINT_FLOAT_VALUE(m_rdo_uastc_max_smooth_block_error_scale);
			PRINT_FLOAT_VALUE(m_rdo_uastc_smooth_block_max_std_dev);
			PRINT_BOOL_VALUE(m_rdo_uastc_favor_simpler_modes_in_rdo_mode)
			PRINT_BOOL_VALUE(m_rdo_uastc_multithreading);

			PRINT_INT_VALUE(m_resample_width);
			PRINT_INT_VALUE(m_resample_height);
			PRINT_FLOAT_VALUE(m_resample_factor);
			debug_printf("Has global codebooks: %u\n", m_params.m_pGlobal_codebooks ? 1 : 0);
			if (m_params.m_pGlobal_codebooks)
			{
				debug_printf("Global codebook endpoints: %u selectors: %u\n", m_params.m_pGlobal_codebooks->get_endpoints().size(), m_params.m_pGlobal_codebooks->get_selectors().size());
			}

			PRINT_BOOL_VALUE(m_create_ktx2_file);

			debug_printf("KTX2 UASTC supercompression: %u\n", m_params.m_ktx2_uastc_supercompression);
			debug_printf("KTX2 Zstd supercompression level: %i\n", (int)m_params.m_ktx2_zstd_supercompression_level);
			debug_printf("KTX2 sRGB transfer func: %u\n", (int)m_params.m_ktx2_srgb_transfer_func);
			debug_printf("Total KTX2 key values: %u\n", m_params.m_ktx2_key_values.size());
			for (uint32_t i = 0; i < m_params.m_ktx2_key_values.size(); i++)
			{
				debug_printf("Key: \"%s\"\n", m_params.m_ktx2_key_values[i].m_key.data());
				debug_printf("Value size: %u\n", m_params.m_ktx2_key_values[i].m_value.size());
			}
						
#undef PRINT_BOOL_VALUE
#undef PRINT_INT_VALUE
#undef PRINT_UINT_VALUE
#undef PRINT_FLOAT_VALUE
		}

		if ((m_params.m_read_source_images) && (!m_params.m_source_filenames.size()))
		{
			assert(0);
			return false;
		}

		return true;
	}
		
	basis_compressor::error_code basis_compressor::process()
	{
		debug_printf("basis_compressor::process\n");

		if (!read_source_images())
			return cECFailedReadingSourceImages;

		if (!validate_texture_type_constraints())
			return cECFailedValidating;

		if (m_params.m_create_ktx2_file)
		{
			if (!validate_ktx2_constraints())
				return cECFailedValidating;
		}

		if (!extract_source_blocks())
			return cECFailedFrontEnd;

		if (m_params.m_uastc)
		{
			error_code ec = encode_slices_to_uastc();
			if (ec != cECSuccess)
				return ec;
		}
		else
		{
			if (!process_frontend())
				return cECFailedFrontEnd;

			if (!extract_frontend_texture_data())
				return cECFailedFontendExtract;

			if (!process_backend())
				return cECFailedBackend;
		}

		if (!create_basis_file_and_transcode())
			return cECFailedCreateBasisFile;
		
		if (m_params.m_create_ktx2_file)
		{
			if (!create_ktx2_file())
				return cECFailedCreateKTX2File;
		}

		if (!write_output_files_and_compute_stats())
			return cECFailedWritingOutput;

		return cECSuccess;
	}

	basis_compressor::error_code basis_compressor::encode_slices_to_uastc()
	{
		debug_printf("basis_compressor::encode_slices_to_uastc\n");

		m_uastc_slice_textures.resize(m_slice_descs.size());
		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
			m_uastc_slice_textures[slice_index].init(texture_format::cUASTC4x4, m_slice_descs[slice_index].m_orig_width, m_slice_descs[slice_index].m_orig_height);

		m_uastc_backend_output.m_tex_format = basist::basis_tex_format::cUASTC4x4;
		m_uastc_backend_output.m_etc1s = false;
		m_uastc_backend_output.m_slice_desc = m_slice_descs;
		m_uastc_backend_output.m_slice_image_data.resize(m_slice_descs.size());
		m_uastc_backend_output.m_slice_image_crcs.resize(m_slice_descs.size());
				
		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
		{
			gpu_image& tex = m_uastc_slice_textures[slice_index];
			basisu_backend_slice_desc& slice_desc = m_slice_descs[slice_index];
			(void)slice_desc;

			const uint32_t num_blocks_x = tex.get_blocks_x();
			const uint32_t num_blocks_y = tex.get_blocks_y();
			const uint32_t total_blocks = tex.get_total_blocks();
			const image& source_image = m_slice_images[slice_index];
			
			std::atomic<uint32_t> total_blocks_processed;
			total_blocks_processed = 0;

			const uint32_t N = 256;
			for (uint32_t block_index_iter = 0; block_index_iter < total_blocks; block_index_iter += N)
			{
				const uint32_t first_index = block_index_iter;
				const uint32_t last_index = minimum<uint32_t>(total_blocks, block_index_iter + N);

				// FIXME: This sucks, but we're having a stack size related problem with std::function with emscripten.
#ifndef __EMSCRIPTEN__
				m_params.m_pJob_pool->add_job([this, first_index, last_index, num_blocks_x, num_blocks_y, total_blocks, &source_image, &tex, &total_blocks_processed]
					{
#endif
						BASISU_NOTE_UNUSED(num_blocks_y);
						
						uint32_t uastc_flags = m_params.m_pack_uastc_flags;
						if ((m_params.m_rdo_uastc) && (m_params.m_rdo_uastc_favor_simpler_modes_in_rdo_mode))
							uastc_flags |= cPackUASTCFavorSimplerModes;

						for (uint32_t block_index = first_index; block_index < last_index; block_index++)
						{
							const uint32_t block_x = block_index % num_blocks_x;
							const uint32_t block_y = block_index / num_blocks_x;

							color_rgba block_pixels[4][4];

							source_image.extract_block_clamped((color_rgba*)block_pixels, block_x * 4, block_y * 4, 4, 4);

							basist::uastc_block& dest_block = *(basist::uastc_block*)tex.get_block_ptr(block_x, block_y);

							encode_uastc(&block_pixels[0][0].r, dest_block, uastc_flags);

							total_blocks_processed++;
							
							uint32_t val = total_blocks_processed;
							if ((val & 16383) == 16383)
							{
								debug_printf("basis_compressor::encode_slices_to_uastc: %3.1f%% done\n", static_cast<float>(val) * 100.0f / total_blocks);
							}

						}

#ifndef __EMSCRIPTEN__
					});
#endif

			} // block_index_iter

#ifndef __EMSCRIPTEN__
			m_params.m_pJob_pool->wait_for_all();
#endif

			if (m_params.m_rdo_uastc)
			{
				uastc_rdo_params rdo_params;
				rdo_params.m_lambda = m_params.m_rdo_uastc_quality_scalar;
				rdo_params.m_max_allowed_rms_increase_ratio = m_params.m_rdo_uastc_max_allowed_rms_increase_ratio;
				rdo_params.m_skip_block_rms_thresh = m_params.m_rdo_uastc_skip_block_rms_thresh;
				rdo_params.m_lz_dict_size = m_params.m_rdo_uastc_dict_size;
				rdo_params.m_smooth_block_max_error_scale = m_params.m_rdo_uastc_max_smooth_block_error_scale;
				rdo_params.m_max_smooth_block_std_dev = m_params.m_rdo_uastc_smooth_block_max_std_dev;
								
				bool status = uastc_rdo(tex.get_total_blocks(), (basist::uastc_block*)tex.get_ptr(),
					(const color_rgba *)m_source_blocks[slice_desc.m_first_block_index].m_pixels, rdo_params, m_params.m_pack_uastc_flags, m_params.m_rdo_uastc_multithreading ? m_params.m_pJob_pool : nullptr,
					(m_params.m_rdo_uastc_multithreading && m_params.m_pJob_pool) ? basisu::minimum<uint32_t>(4, (uint32_t)m_params.m_pJob_pool->get_total_threads()) : 0);
				if (!status)
				{
					return cECFailedUASTCRDOPostProcess;
				}
			}

			m_uastc_backend_output.m_slice_image_data[slice_index].resize(tex.get_size_in_bytes());
			memcpy(&m_uastc_backend_output.m_slice_image_data[slice_index][0], tex.get_ptr(), tex.get_size_in_bytes());
			
			m_uastc_backend_output.m_slice_image_crcs[slice_index] = basist::crc16(tex.get_ptr(), tex.get_size_in_bytes(), 0);
						
		} // slice_index
				
		return cECSuccess;
	}

	bool basis_compressor::generate_mipmaps(const image &img, basisu::vector<image> &mips, bool has_alpha)
	{
		debug_printf("basis_compressor::generate_mipmaps\n");

		interval_timer tm;
		tm.start();

		uint32_t total_levels = 1;
		uint32_t w = img.get_width(), h = img.get_height();
		while (maximum<uint32_t>(w, h) > (uint32_t)m_params.m_mip_smallest_dimension)
		{
			w = maximum(w >> 1U, 1U);
			h = maximum(h >> 1U, 1U);
			total_levels++;
		}

#if BASISU_USE_STB_IMAGE_RESIZE_FOR_MIPMAP_GEN
		// Requires stb_image_resize
		stbir_filter filter = STBIR_FILTER_DEFAULT;
		if (m_params.m_mip_filter == "box")
			filter = STBIR_FILTER_BOX;
		else if (m_params.m_mip_filter == "triangle")
			filter = STBIR_FILTER_TRIANGLE;
		else if (m_params.m_mip_filter == "cubic")
			filter = STBIR_FILTER_CUBICBSPLINE;
		else if (m_params.m_mip_filter == "catmull")
			filter = STBIR_FILTER_CATMULLROM;
		else if (m_params.m_mip_filter == "mitchell")
			filter = STBIR_FILTER_MITCHELL;

		for (uint32_t level = 1; level < total_levels; level++)
		{
			const uint32_t level_width = maximum<uint32_t>(1, img.get_width() >> level);
			const uint32_t level_height = maximum<uint32_t>(1, img.get_height() >> level);

			image &level_img = *enlarge_vector(mips, 1);
			level_img.resize(level_width, level_height);
						
			int result = stbir_resize_uint8_generic( 
				(const uint8_t *)img.get_ptr(), img.get_width(), img.get_height(), img.get_pitch() * sizeof(color_rgba),
            (uint8_t *)level_img.get_ptr(), level_img.get_width(), level_img.get_height(), level_img.get_pitch() * sizeof(color_rgba),
            has_alpha ? 4 : 3, has_alpha ? 3 : STBIR_ALPHA_CHANNEL_NONE, m_params.m_mip_premultiplied ? STBIR_FLAG_ALPHA_PREMULTIPLIED : 0,
            m_params.m_mip_wrapping ? STBIR_EDGE_WRAP : STBIR_EDGE_CLAMP, filter, m_params.m_mip_srgb ? STBIR_COLORSPACE_SRGB : STBIR_COLORSPACE_LINEAR, 
				nullptr);

			if (result == 0)
			{
				error_printf("basis_compressor::generate_mipmaps: stbir_resize_uint8_generic() failed!\n");
				return false;
			}
			
			if (m_params.m_mip_renormalize)
				level_img.renormalize_normal_map();
		}
#else
		for (uint32_t level = 1; level < total_levels; level++)
		{
			const uint32_t level_width = maximum<uint32_t>(1, img.get_width() >> level);
			const uint32_t level_height = maximum<uint32_t>(1, img.get_height() >> level);

			image& level_img = *enlarge_vector(mips, 1);
			level_img.resize(level_width, level_height);

			const image* pSource_image = &img;

			if (m_params.m_mip_fast)
			{
				if (level > 1)
					pSource_image = &mips[level - 1];
			}

			bool status = image_resample(*pSource_image, level_img, m_params.m_mip_srgb, m_params.m_mip_filter.c_str(), m_params.m_mip_scale, m_params.m_mip_wrapping, 0, has_alpha ? 4 : 3);
			if (!status)
			{
				error_printf("basis_compressor::generate_mipmaps: image_resample() failed!\n");
				return false;
			}

			if (m_params.m_mip_renormalize)
				level_img.renormalize_normal_map();
		}
#endif

		if (m_params.m_debug)
			debug_printf("Total mipmap generation time: %f secs\n", tm.get_elapsed_secs());

		return true;
	}

	bool basis_compressor::read_source_images()
	{
		debug_printf("basis_compressor::read_source_images\n");

		const uint32_t total_source_files = m_params.m_read_source_images ? (uint32_t)m_params.m_source_filenames.size() : (uint32_t)m_params.m_source_images.size();
		if (!total_source_files)
			return false;

		m_stats.resize(0);
		m_slice_descs.resize(0);
		m_slice_images.resize(0);

		m_total_blocks = 0;
		uint32_t total_macroblocks = 0;

		m_any_source_image_has_alpha = false;

		basisu::vector<image> source_images;
		basisu::vector<std::string> source_filenames;
		
		// First load all source images, and determine if any have an alpha channel.
		for (uint32_t source_file_index = 0; source_file_index < total_source_files; source_file_index++)
		{
			const char *pSource_filename = "";

			image file_image;
			
			if (m_params.m_read_source_images)
			{
				pSource_filename = m_params.m_source_filenames[source_file_index].c_str();

				// Load the source image
				if (!load_image(pSource_filename, file_image))
				{
					error_printf("Failed reading source image: %s\n", pSource_filename);
					return false;
				}

				printf("Read source image \"%s\", %ux%u\n", pSource_filename, file_image.get_width(), file_image.get_height());

				// Optionally load another image and put a grayscale version of it into the alpha channel.
				if ((source_file_index < m_params.m_source_alpha_filenames.size()) && (m_params.m_source_alpha_filenames[source_file_index].size()))
				{
					const char *pSource_alpha_image = m_params.m_source_alpha_filenames[source_file_index].c_str();

					image alpha_data;

					if (!load_image(pSource_alpha_image, alpha_data))
					{
						error_printf("Failed reading source image: %s\n", pSource_alpha_image);
						return false;
					}

					printf("Read source alpha image \"%s\", %ux%u\n", pSource_alpha_image, alpha_data.get_width(), alpha_data.get_height());

					alpha_data.crop(file_image.get_width(), file_image.get_height());

					for (uint32_t y = 0; y < file_image.get_height(); y++)
						for (uint32_t x = 0; x < file_image.get_width(); x++)
							file_image(x, y).a = (uint8_t)alpha_data(x, y).get_709_luma();
				}
			}
			else
			{
				file_image = m_params.m_source_images[source_file_index];
			}

			if (m_params.m_renormalize)
				file_image.renormalize_normal_map();

			bool alpha_swizzled = false;
			if (m_params.m_swizzle[0] != 0 ||
				m_params.m_swizzle[1] != 1 ||
				m_params.m_swizzle[2] != 2 ||
				m_params.m_swizzle[3] != 3)
			{
				// Used for XY normal maps in RG - puts X in color, Y in alpha
				for (uint32_t y = 0; y < file_image.get_height(); y++)
					for (uint32_t x = 0; x < file_image.get_width(); x++)
					{
						const color_rgba &c = file_image(x, y);
						file_image(x, y).set_noclamp_rgba(c[m_params.m_swizzle[0]], c[m_params.m_swizzle[1]], c[m_params.m_swizzle[2]], c[m_params.m_swizzle[3]]);
					}
				alpha_swizzled = m_params.m_swizzle[3] != 3;
			}
						
			bool has_alpha = false;
			if (m_params.m_force_alpha || alpha_swizzled)
				has_alpha = true;
			else if (!m_params.m_check_for_alpha)
				file_image.set_alpha(255);
			else if (file_image.has_alpha())
				has_alpha = true;

			if (has_alpha)
				m_any_source_image_has_alpha = true;

			debug_printf("Source image index %u filename %s %ux%u has alpha: %u\n", source_file_index, pSource_filename, file_image.get_width(), file_image.get_height(), has_alpha);
												
			if (m_params.m_y_flip)
				file_image.flip_y();

#if DEBUG_EXTRACT_SINGLE_BLOCK
			image block_image(4, 4);
			const uint32_t block_x = 0;
			const uint32_t block_y = 0;
			block_image.blit(block_x * 4, block_y * 4, 4, 4, 0, 0, file_image, 0);
			file_image = block_image;
#endif

#if DEBUG_CROP_TEXTURE_TO_64x64
			file_image.resize(64, 64);
#endif

			if (m_params.m_resample_width > 0 && m_params.m_resample_height > 0)
			{
				int new_width = basisu::minimum<int>(m_params.m_resample_width, BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION);
				int new_height = basisu::minimum<int>(m_params.m_resample_height, BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION);

				debug_printf("Resampling to %ix%i\n", new_width, new_height);

				// TODO: A box filter - kaiser looks too sharp on video. Let the caller control this.
				image temp_img(new_width, new_height);
				image_resample(file_image, temp_img, m_params.m_perceptual, "box"); // "kaiser");
				temp_img.swap(file_image);
			}
			else if (m_params.m_resample_factor > 0.0f)
			{
				int new_width = basisu::minimum<int>(basisu::maximum(1, (int)ceilf(file_image.get_width() * m_params.m_resample_factor)), BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION);
				int new_height = basisu::minimum<int>(basisu::maximum(1, (int)ceilf(file_image.get_height() * m_params.m_resample_factor)), BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION);

				debug_printf("Resampling to %ix%i\n", new_width, new_height);

				// TODO: A box filter - kaiser looks too sharp on video. Let the caller control this.
				image temp_img(new_width, new_height);
				image_resample(file_image, temp_img, m_params.m_perceptual, "box"); // "kaiser");
				temp_img.swap(file_image);
			}

			if ((!file_image.get_width()) || (!file_image.get_height()))
			{
				error_printf("basis_compressor::read_source_images: Source image has a zero width and/or height!\n");
				return false;
			}

			if ((file_image.get_width() > BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION) || (file_image.get_height() > BASISU_MAX_SUPPORTED_TEXTURE_DIMENSION))
			{
				error_printf("basis_compressor::read_source_images: Source image is too large!\n");
				return false;
			}

			source_images.push_back(file_image);
			source_filenames.push_back(pSource_filename);
		}

		// Check if the caller has generated their own mipmaps. 
		if (m_params.m_source_mipmap_images.size())
		{
			// Make sure they've passed us enough mipmap chains.
			if ((m_params.m_source_images.size() != m_params.m_source_mipmap_images.size()) || (total_source_files != m_params.m_source_images.size()))
			{
				error_printf("basis_compressor::read_source_images(): m_params.m_source_mipmap_images.size() must equal m_params.m_source_images.size()!\n");
				return false;
			}

			// Check if any of the user-supplied mipmap levels has alpha.
			// We're assuming the user has already preswizzled their mipmap source images.
			if (!m_any_source_image_has_alpha)
			{
				for (uint32_t source_file_index = 0; source_file_index < total_source_files; source_file_index++)
				{
					for (uint32_t mip_index = 0; mip_index < m_params.m_source_mipmap_images[source_file_index].size(); mip_index++)
					{
						const image& mip_img = m_params.m_source_mipmap_images[source_file_index][mip_index];

						if (mip_img.has_alpha())
						{
							m_any_source_image_has_alpha = true;
							break;
						}
					}

					if (m_any_source_image_has_alpha)
						break;
				}
			}
		}

		debug_printf("Any source image has alpha: %u\n", m_any_source_image_has_alpha);

		for (uint32_t source_file_index = 0; source_file_index < total_source_files; source_file_index++)
		{
			image &file_image = source_images[source_file_index];
			const std::string &source_filename = source_filenames[source_file_index];

			// Now, for each source image, create the slices corresponding to that image.
			basisu::vector<image> slices;
			
			slices.reserve(32);
			
			// The first (largest) mipmap level.
			slices.push_back(file_image);
			
			if (m_params.m_source_mipmap_images.size())
			{
				// User-provided mipmaps for each layer or image in the texture array.
				for (uint32_t mip_index = 0; mip_index < m_params.m_source_mipmap_images[source_file_index].size(); mip_index++)
				{
					image& mip_img = m_params.m_source_mipmap_images[source_file_index][mip_index];

					if (m_params.m_swizzle[0] != 0 ||
						m_params.m_swizzle[1] != 1 ||
						m_params.m_swizzle[2] != 2 ||
						m_params.m_swizzle[3] != 3)
					{
						// Used for XY normal maps in RG - puts X in color, Y in alpha
						for (uint32_t y = 0; y < mip_img.get_height(); y++)
							for (uint32_t x = 0; x < mip_img.get_width(); x++)
							{
								const color_rgba &c = mip_img(x, y);
								mip_img(x, y).set_noclamp_rgba(c[m_params.m_swizzle[0]], c[m_params.m_swizzle[1]], c[m_params.m_swizzle[2]], c[m_params.m_swizzle[3]]);
							}
					}

					slices.push_back(mip_img);
				}
			}
			else if (m_params.m_mip_gen)
			{
				// Automatically generate mipmaps.
				if (!generate_mipmaps(file_image, slices, m_any_source_image_has_alpha))
					return false;
			}

			uint_vec mip_indices(slices.size());
			for (uint32_t i = 0; i < slices.size(); i++)
				mip_indices[i] = i;
						
			if ((m_any_source_image_has_alpha) && (!m_params.m_uastc))
			{
				// For ETC1S, if source has alpha, then even mips will have RGB, and odd mips will have alpha in RGB. 
				basisu::vector<image> alpha_slices;
				uint_vec new_mip_indices;

				alpha_slices.reserve(slices.size() * 2);

				for (uint32_t i = 0; i < slices.size(); i++)
				{
					image lvl_rgb(slices[i]);
					image lvl_a(lvl_rgb);

					for (uint32_t y = 0; y < lvl_a.get_height(); y++)
					{
						for (uint32_t x = 0; x < lvl_a.get_width(); x++)
						{
							uint8_t a = lvl_a(x, y).a;
							lvl_a(x, y).set_noclamp_rgba(a, a, a, 255);
						}
					}
					
					lvl_rgb.set_alpha(255);

					alpha_slices.push_back(lvl_rgb);
					new_mip_indices.push_back(i);

					alpha_slices.push_back(lvl_a);
					new_mip_indices.push_back(i);
				}

				slices.swap(alpha_slices);
				mip_indices.swap(new_mip_indices);
			}

			assert(slices.size() == mip_indices.size());
						
			for (uint32_t slice_index = 0; slice_index < slices.size(); slice_index++)
			{
				image& slice_image = slices[slice_index];
				const uint32_t orig_width = slice_image.get_width();
				const uint32_t orig_height = slice_image.get_height();

				bool is_alpha_slice = false;
				if (m_any_source_image_has_alpha)
				{
					if (m_params.m_uastc)
					{
						is_alpha_slice = slice_image.has_alpha();
					}
					else
					{
						is_alpha_slice = (slice_index & 1) != 0;
					}
				}

				// Enlarge the source image to 4x4 block boundaries, duplicating edge pixels if necessary to avoid introducing extra colors into blocks.
				slice_image.crop_dup_borders(slice_image.get_block_width(4) * 4, slice_image.get_block_height(4) * 4);

				if (m_params.m_debug_images)
				{
					save_png(string_format("basis_debug_source_image_%u_slice_%u.png", source_file_index, slice_index).c_str(), slice_image);
				}

				enlarge_vector(m_stats, 1);
				enlarge_vector(m_slice_images, 1);
				enlarge_vector(m_slice_descs, 1);

				const uint32_t dest_image_index = (uint32_t)m_stats.size() - 1;

				m_stats[dest_image_index].m_filename = source_filename.c_str();
				m_stats[dest_image_index].m_width = orig_width;
				m_stats[dest_image_index].m_height = orig_height;

				m_slice_images[dest_image_index] = slice_image;

				debug_printf("****** Slice %u: mip %u, alpha_slice: %u, filename: \"%s\", original: %ux%u actual: %ux%u\n", m_slice_descs.size() - 1, mip_indices[slice_index], is_alpha_slice, source_filename.c_str(), orig_width, orig_height, slice_image.get_width(), slice_image.get_height());

				basisu_backend_slice_desc &slice_desc = m_slice_descs[dest_image_index];

				slice_desc.m_first_block_index = m_total_blocks;

				slice_desc.m_orig_width = orig_width;
				slice_desc.m_orig_height = orig_height;

				slice_desc.m_width = slice_image.get_width();
				slice_desc.m_height = slice_image.get_height();

				slice_desc.m_num_blocks_x = slice_image.get_block_width(4);
				slice_desc.m_num_blocks_y = slice_image.get_block_height(4);

				slice_desc.m_num_macroblocks_x = (slice_desc.m_num_blocks_x + 1) >> 1;
				slice_desc.m_num_macroblocks_y = (slice_desc.m_num_blocks_y + 1) >> 1;

				slice_desc.m_source_file_index = source_file_index;
				
				slice_desc.m_mip_index = mip_indices[slice_index];

				slice_desc.m_alpha = is_alpha_slice;
				slice_desc.m_iframe = false;
				if (m_params.m_tex_type == basist::cBASISTexTypeVideoFrames)
				{
					slice_desc.m_iframe = (source_file_index == 0);
				}

				m_total_blocks += slice_desc.m_num_blocks_x * slice_desc.m_num_blocks_y;
				total_macroblocks += slice_desc.m_num_macroblocks_x * slice_desc.m_num_macroblocks_y;
			
			} // slice_index

		} // source_file_index

		debug_printf("Total blocks: %u, Total macroblocks: %u\n", m_total_blocks, total_macroblocks);

		// Make sure we don't have too many slices
		if (m_slice_descs.size() > BASISU_MAX_SLICES)
		{
			error_printf("Too many slices!\n");
			return false;
		}
				
		// Basic sanity check on the slices
		for (uint32_t i = 1; i < m_slice_descs.size(); i++)
		{
			const basisu_backend_slice_desc &prev_slice_desc = m_slice_descs[i - 1];
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[i];

			// Make sure images are in order
			int image_delta = (int)slice_desc.m_source_file_index - (int)prev_slice_desc.m_source_file_index;
			if (image_delta > 1)
				return false;	

			// Make sure mipmap levels are in order
			if (!image_delta)
			{
				int level_delta = (int)slice_desc.m_mip_index - (int)prev_slice_desc.m_mip_index;
				if (level_delta > 1)
					return false;
			}
		}

		if (m_params.m_status_output)
		{
			printf("Total basis file slices: %u\n", (uint32_t)m_slice_descs.size());
		}

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[i];

			if (m_params.m_status_output)
			{
				printf("Slice: %u, alpha: %u, orig width/height: %ux%u, width/height: %ux%u, first_block: %u, image_index: %u, mip_level: %u, iframe: %u\n",
					i, slice_desc.m_alpha, slice_desc.m_orig_width, slice_desc.m_orig_height, slice_desc.m_width, slice_desc.m_height, slice_desc.m_first_block_index, slice_desc.m_source_file_index, slice_desc.m_mip_index, slice_desc.m_iframe);
			}

			if (m_any_source_image_has_alpha)
			{
				if (!m_params.m_uastc)
				{
					// For ETC1S, alpha slices must be at odd slice indices.
					if (slice_desc.m_alpha)
					{
						if ((i & 1) == 0)
							return false;

						const basisu_backend_slice_desc& prev_slice_desc = m_slice_descs[i - 1];

						// Make sure previous slice has this image's color data
						if (prev_slice_desc.m_source_file_index != slice_desc.m_source_file_index)
							return false;
						if (prev_slice_desc.m_alpha)
							return false;
						if (prev_slice_desc.m_mip_index != slice_desc.m_mip_index)
							return false;
						if (prev_slice_desc.m_num_blocks_x != slice_desc.m_num_blocks_x)
							return false;
						if (prev_slice_desc.m_num_blocks_y != slice_desc.m_num_blocks_y)
							return false;
					}
					else if (i & 1)
						return false;
				}
			}
			else if (slice_desc.m_alpha)
			{
				return false;
			}

			if ((slice_desc.m_orig_width > slice_desc.m_width) || (slice_desc.m_orig_height > slice_desc.m_height))
				return false;
			if ((slice_desc.m_source_file_index == 0) && (m_params.m_tex_type == basist::cBASISTexTypeVideoFrames))
			{
				if (!slice_desc.m_iframe)
					return false;
			}
		}

		return true;
	}

	// Do some basic validation for 2D arrays, cubemaps, video, and volumes.
	bool basis_compressor::validate_texture_type_constraints() 
	{
		debug_printf("basis_compressor::validate_texture_type_constraints\n");

		// In 2D mode anything goes (each image may have a different resolution and # of mipmap levels).
		if (m_params.m_tex_type == basist::cBASISTexType2D)
			return true;
				
		uint32_t total_basis_images = 0;

		for (uint32_t slice_index = 0; slice_index < m_slice_images.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];
				
			total_basis_images = maximum<uint32_t>(total_basis_images, slice_desc.m_source_file_index + 1);
		}

		if (m_params.m_tex_type == basist::cBASISTexTypeCubemapArray)
		{
			// For cubemaps, validate that the total # of Basis images is a multiple of 6.
			if ((total_basis_images % 6) != 0)
			{
				error_printf("basis_compressor::validate_texture_type_constraints: For cubemaps the total number of input images is not a multiple of 6!\n");
				return false;
			}
		}

		// Now validate that all the mip0's have the same dimensions, and that each image has the same # of mipmap levels.
		uint_vec image_mipmap_levels(total_basis_images);

		int width = -1, height = -1;
		for (uint32_t slice_index = 0; slice_index < m_slice_images.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];

			image_mipmap_levels[slice_desc.m_source_file_index] = maximum(image_mipmap_levels[slice_desc.m_source_file_index], slice_desc.m_mip_index + 1);
				
			if (slice_desc.m_mip_index != 0)
				continue;

			if (width < 0)
			{
				width = slice_desc.m_orig_width;
				height = slice_desc.m_orig_height;
			}
			else if ((width != (int)slice_desc.m_orig_width) || (height != (int)slice_desc.m_orig_height))
			{
				error_printf("basis_compressor::validate_texture_type_constraints: The source image resolutions are not all equal!\n");
				return false;
			}
		}

		for (size_t i = 1; i < image_mipmap_levels.size(); i++)
		{
			if (image_mipmap_levels[0] != image_mipmap_levels[i])
			{
				error_printf("basis_compressor::validate_texture_type_constraints: Each image must have the same number of mipmap levels!\n");
				return false;
			}
		}

		return true;
	}

	bool basis_compressor::extract_source_blocks()
	{
		debug_printf("basis_compressor::extract_source_blocks\n");

		m_source_blocks.resize(m_total_blocks);

		for (uint32_t slice_index = 0; slice_index < m_slice_images.size(); slice_index++)
		{
			const basisu_backend_slice_desc& slice_desc = m_slice_descs[slice_index];

			const uint32_t num_blocks_x = slice_desc.m_num_blocks_x;
			const uint32_t num_blocks_y = slice_desc.m_num_blocks_y;

			const image& source_image = m_slice_images[slice_index];

			for (uint32_t block_y = 0; block_y < num_blocks_y; block_y++)
				for (uint32_t block_x = 0; block_x < num_blocks_x; block_x++)
					source_image.extract_block_clamped(m_source_blocks[slice_desc.m_first_block_index + block_x + block_y * num_blocks_x].get_ptr(), block_x * 4, block_y * 4, 4, 4);
		}

		return true;
	}

	bool basis_compressor::process_frontend()
	{
		debug_printf("basis_compressor::process_frontend\n");
						
#if 0
		// TODO
		basis_etc1_pack_params pack_params;
		pack_params.m_quality = cETCQualityMedium;
		pack_params.m_perceptual = m_params.m_perceptual;
		pack_params.m_use_color4 = false;

		pack_etc1_block_context pack_context;

		std::unordered_set<uint64_t> endpoint_hash;
		std::unordered_set<uint32_t> selector_hash;

		for (uint32_t i = 0; i < m_source_blocks.size(); i++)
		{
			etc_block blk;
			pack_etc1_block(blk, m_source_blocks[i].get_ptr(), pack_params, pack_context);

			const color_rgba c0(blk.get_block_color(0, false));
			endpoint_hash.insert((c0.r | (c0.g << 5) | (c0.b << 10)) | (blk.get_inten_table(0) << 16));

			const color_rgba c1(blk.get_block_color(1, false));
			endpoint_hash.insert((c1.r | (c1.g << 5) | (c1.b << 10)) | (blk.get_inten_table(1) << 16));

			selector_hash.insert(blk.get_raw_selector_bits());
		}

		const uint32_t total_unique_endpoints = (uint32_t)endpoint_hash.size();
		const uint32_t total_unique_selectors = (uint32_t)selector_hash.size();

		if (m_params.m_debug)
		{
			debug_printf("Unique endpoints: %u, unique selectors: %u\n", total_unique_endpoints, total_unique_selectors);
		}
#endif

		const double total_texels = m_total_blocks * 16.0f;

		int endpoint_clusters = m_params.m_max_endpoint_clusters;
		int selector_clusters = m_params.m_max_selector_clusters;

		if (endpoint_clusters > basisu_frontend::cMaxEndpointClusters)
		{
			error_printf("Too many endpoint clusters! (%u but max is %u)\n", endpoint_clusters, basisu_frontend::cMaxEndpointClusters);
			return false;
		}
		if (selector_clusters > basisu_frontend::cMaxSelectorClusters)
		{
			error_printf("Too many selector clusters! (%u but max is %u)\n", selector_clusters, basisu_frontend::cMaxSelectorClusters);
			return false;
		}
		
		if (m_params.m_quality_level != -1)
		{
			const float quality = saturate(m_params.m_quality_level / 255.0f);
									
			const float bits_per_endpoint_cluster = 14.0f;
			const float max_desired_endpoint_cluster_bits_per_texel = 1.0f; // .15f
			int max_endpoints = static_cast<int>((max_desired_endpoint_cluster_bits_per_texel * total_texels) / bits_per_endpoint_cluster);
			
			const float mid = 128.0f / 255.0f;

			float color_endpoint_quality = quality;

			const float endpoint_split_point = 0.5f;
			
			// In v1.2 and in previous versions, the endpoint codebook size at quality 128 was 3072. This wasn't quite large enough.
			const int ENDPOINT_CODEBOOK_MID_QUALITY_CODEBOOK_SIZE = 4800;
			const int MAX_ENDPOINT_CODEBOOK_SIZE = 8192;

			if (color_endpoint_quality <= mid)
			{
				color_endpoint_quality = lerp(0.0f, endpoint_split_point, powf(color_endpoint_quality / mid, .65f));

				max_endpoints = clamp<int>(max_endpoints, 256, ENDPOINT_CODEBOOK_MID_QUALITY_CODEBOOK_SIZE);
				max_endpoints = minimum<uint32_t>(max_endpoints, m_total_blocks);
								
				if (max_endpoints < 64)
					max_endpoints = 64;
				endpoint_clusters = clamp<uint32_t>((uint32_t)(.5f + lerp<float>(32, static_cast<float>(max_endpoints), color_endpoint_quality)), 32, basisu_frontend::cMaxEndpointClusters);
			}
			else
			{
				color_endpoint_quality = powf((color_endpoint_quality - mid) / (1.0f - mid), 1.6f);

				max_endpoints = clamp<int>(max_endpoints, 256, MAX_ENDPOINT_CODEBOOK_SIZE);
				max_endpoints = minimum<uint32_t>(max_endpoints, m_total_blocks);
								
				if (max_endpoints < ENDPOINT_CODEBOOK_MID_QUALITY_CODEBOOK_SIZE)
					max_endpoints = ENDPOINT_CODEBOOK_MID_QUALITY_CODEBOOK_SIZE;
				endpoint_clusters = clamp<uint32_t>((uint32_t)(.5f + lerp<float>(ENDPOINT_CODEBOOK_MID_QUALITY_CODEBOOK_SIZE, static_cast<float>(max_endpoints), color_endpoint_quality)), 32, basisu_frontend::cMaxEndpointClusters);
			}
						
			float bits_per_selector_cluster = m_params.m_global_sel_pal ? 21.0f : 14.0f;

			const float max_desired_selector_cluster_bits_per_texel = 1.0f; // .15f
			int max_selectors = static_cast<int>((max_desired_selector_cluster_bits_per_texel * total_texels) / bits_per_selector_cluster);
			max_selectors = clamp<int>(max_selectors, 256, basisu_frontend::cMaxSelectorClusters);
			max_selectors = minimum<uint32_t>(max_selectors, m_total_blocks);

			float color_selector_quality = quality;
			//color_selector_quality = powf(color_selector_quality, 1.65f);
			color_selector_quality = powf(color_selector_quality, 2.62f);

			if (max_selectors < 96)
				max_selectors = 96;
			selector_clusters = clamp<uint32_t>((uint32_t)(.5f + lerp<float>(96, static_cast<float>(max_selectors), color_selector_quality)), 8, basisu_frontend::cMaxSelectorClusters);

			debug_printf("Max endpoints: %u, max selectors: %u\n", endpoint_clusters, selector_clusters);

			if (m_params.m_quality_level >= 223)
			{
				if (!m_params.m_selector_rdo_thresh.was_changed())
				{
					if (!m_params.m_endpoint_rdo_thresh.was_changed())
						m_params.m_endpoint_rdo_thresh *= .25f;
					
					if (!m_params.m_selector_rdo_thresh.was_changed())
						m_params.m_selector_rdo_thresh *= .25f;
				}
			}
			else if (m_params.m_quality_level >= 192)
			{
				if (!m_params.m_endpoint_rdo_thresh.was_changed())
					m_params.m_endpoint_rdo_thresh *= .5f;

				if (!m_params.m_selector_rdo_thresh.was_changed())
					m_params.m_selector_rdo_thresh *= .5f;
			}
			else if (m_params.m_quality_level >= 160)
			{
				if (!m_params.m_endpoint_rdo_thresh.was_changed())
					m_params.m_endpoint_rdo_thresh *= .75f;

				if (!m_params.m_selector_rdo_thresh.was_changed())
					m_params.m_selector_rdo_thresh *= .75f;
			}
			else if (m_params.m_quality_level >= 129)
			{
				float l = (quality - 129 / 255.0f) / ((160 - 129) / 255.0f);

				if (!m_params.m_endpoint_rdo_thresh.was_changed())
					m_params.m_endpoint_rdo_thresh *= lerp<float>(1.0f, .75f, l);
				
				if (!m_params.m_selector_rdo_thresh.was_changed())
					m_params.m_selector_rdo_thresh *= lerp<float>(1.0f, .75f, l);
			}
		}

		m_auto_global_sel_pal = false;
		if (!m_params.m_global_sel_pal && m_params.m_auto_global_sel_pal)
		{
			const float bits_per_selector_cluster = 31.0f;
			double selector_codebook_bpp_est = (bits_per_selector_cluster * selector_clusters) / total_texels;
			debug_printf("selector_codebook_bpp_est: %f\n", selector_codebook_bpp_est);
			const float force_global_sel_pal_bpp_threshold = .15f;
			if ((total_texels <= 128.0f*128.0f) && (selector_codebook_bpp_est > force_global_sel_pal_bpp_threshold))
			{
				m_auto_global_sel_pal = true;
				debug_printf("Auto global selector palette enabled\n");
			}
		}

		basisu_frontend::params p;
		p.m_num_source_blocks = m_total_blocks;
		p.m_pSource_blocks = &m_source_blocks[0];
		p.m_max_endpoint_clusters = endpoint_clusters;
		p.m_max_selector_clusters = selector_clusters;
		p.m_perceptual = m_params.m_perceptual;
		p.m_debug_stats = m_params.m_debug;
		p.m_debug_images = m_params.m_debug_images;
		p.m_compression_level = m_params.m_compression_level;
		p.m_tex_type = m_params.m_tex_type;
		p.m_multithreaded = m_params.m_multithreading;
		p.m_disable_hierarchical_endpoint_codebooks = m_params.m_disable_hierarchical_endpoint_codebooks;
		p.m_validate = m_params.m_validate;
		p.m_pJob_pool = m_params.m_pJob_pool;
		p.m_pGlobal_codebooks = m_params.m_pGlobal_codebooks;

		if ((m_params.m_global_sel_pal) || (m_auto_global_sel_pal))
		{
			p.m_pGlobal_sel_codebook = m_params.m_pSel_codebook;
			p.m_num_global_sel_codebook_pal_bits = m_params.m_global_pal_bits;
			p.m_num_global_sel_codebook_mod_bits = m_params.m_global_mod_bits;
			p.m_use_hybrid_selector_codebooks = !m_params.m_no_hybrid_sel_cb;
			p.m_hybrid_codebook_quality_thresh = m_params.m_hybrid_sel_cb_quality_thresh;
		}

		if (!m_frontend.init(p))
		{
			error_printf("basisu_frontend::init() failed!\n");
			return false;
		}

		m_frontend.compress();

		if (m_params.m_debug_images)
		{
			for (uint32_t i = 0; i < m_slice_descs.size(); i++)
			{
				char filename[1024];
#ifdef _WIN32				
				sprintf_s(filename, sizeof(filename), "rdo_frontend_output_output_blocks_%u.png", i);
#else
				snprintf(filename, sizeof(filename), "rdo_frontend_output_output_blocks_%u.png", i);
#endif				
				m_frontend.dump_debug_image(filename, m_slice_descs[i].m_first_block_index, m_slice_descs[i].m_num_blocks_x, m_slice_descs[i].m_num_blocks_y, true);

#ifdef _WIN32
				sprintf_s(filename, sizeof(filename), "rdo_frontend_output_api_%u.png", i);
#else
				snprintf(filename, sizeof(filename), "rdo_frontend_output_api_%u.png", i);
#endif				
				m_frontend.dump_debug_image(filename, m_slice_descs[i].m_first_block_index, m_slice_descs[i].m_num_blocks_x, m_slice_descs[i].m_num_blocks_y, false);
			}
		}

		return true;
	}

	bool basis_compressor::extract_frontend_texture_data()
	{
		debug_printf("basis_compressor::extract_frontend_texture_data\n");

		m_frontend_output_textures.resize(m_slice_descs.size());
		m_best_etc1s_images.resize(m_slice_descs.size());
		m_best_etc1s_images_unpacked.resize(m_slice_descs.size());

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[i];

			const uint32_t num_blocks_x = slice_desc.m_num_blocks_x;
			const uint32_t num_blocks_y = slice_desc.m_num_blocks_y;

			const uint32_t width = num_blocks_x * 4;
			const uint32_t height = num_blocks_y * 4;

			m_frontend_output_textures[i].init(texture_format::cETC1, width, height);

			for (uint32_t block_y = 0; block_y < num_blocks_y; block_y++)
				for (uint32_t block_x = 0; block_x < num_blocks_x; block_x++)
					memcpy(m_frontend_output_textures[i].get_block_ptr(block_x, block_y, 0), &m_frontend.get_output_block(slice_desc.m_first_block_index + block_x + block_y * num_blocks_x), sizeof(etc_block));

#if 0
			if (m_params.m_debug_images)
			{
				char filename[1024];
				sprintf_s(filename, sizeof(filename), "rdo_etc_frontend_%u_", i);
				write_etc1_vis_images(m_frontend_output_textures[i], filename);
			}
#endif

			m_best_etc1s_images[i].init(texture_format::cETC1, width, height);
			for (uint32_t block_y = 0; block_y < num_blocks_y; block_y++)
				for (uint32_t block_x = 0; block_x < num_blocks_x; block_x++)
					memcpy(m_best_etc1s_images[i].get_block_ptr(block_x, block_y, 0), &m_frontend.get_etc1s_block(slice_desc.m_first_block_index + block_x + block_y * num_blocks_x), sizeof(etc_block));

			m_best_etc1s_images[i].unpack(m_best_etc1s_images_unpacked[i]);
		}

		return true;
	}

	bool basis_compressor::process_backend()
	{
		debug_printf("basis_compressor::process_backend\n");

		basisu_backend_params backend_params;
		backend_params.m_debug = m_params.m_debug;
		backend_params.m_debug_images = m_params.m_debug_images;
		backend_params.m_etc1s = true;
		backend_params.m_compression_level = m_params.m_compression_level;
		
		if (!m_params.m_no_endpoint_rdo)
			backend_params.m_endpoint_rdo_quality_thresh = m_params.m_endpoint_rdo_thresh;

		if (!m_params.m_no_selector_rdo)
			backend_params.m_selector_rdo_quality_thresh = m_params.m_selector_rdo_thresh;
				
		backend_params.m_use_global_sel_codebook = (m_frontend.get_params().m_pGlobal_sel_codebook != NULL);
		backend_params.m_global_sel_codebook_pal_bits = m_frontend.get_params().m_num_global_sel_codebook_pal_bits;
		backend_params.m_global_sel_codebook_mod_bits = m_frontend.get_params().m_num_global_sel_codebook_mod_bits;
		backend_params.m_use_hybrid_sel_codebooks = m_frontend.get_params().m_use_hybrid_selector_codebooks;
		backend_params.m_used_global_codebooks = m_frontend.get_params().m_pGlobal_codebooks != nullptr;

		m_backend.init(&m_frontend, backend_params, m_slice_descs, m_params.m_pSel_codebook);
		uint32_t total_packed_bytes = m_backend.encode();

		if (!total_packed_bytes)
		{
			error_printf("basis_compressor::encode() failed!\n");
			return false;
		}

		debug_printf("Total packed bytes (estimated): %u\n", total_packed_bytes);

		return true;
	}

	bool basis_compressor::create_basis_file_and_transcode()
	{
		debug_printf("basis_compressor::create_basis_file_and_transcode\n");

		const basisu_backend_output& encoded_output = m_params.m_uastc ? m_uastc_backend_output : m_backend.get_output();

		if (!m_basis_file.init(encoded_output, m_params.m_tex_type, m_params.m_userdata0, m_params.m_userdata1, m_params.m_y_flip, m_params.m_us_per_frame))
		{
			error_printf("basis_compressor::create_basis_file_and_transcode: basisu_backend:init() failed!\n");
			return false;
		}
	
		const uint8_vec &comp_data = m_basis_file.get_compressed_data();

		m_output_basis_file = comp_data;

		interval_timer tm;
		tm.start();

		basist::basisu_transcoder_init();

		debug_printf("basist::basisu_transcoder_init: Took %f ms\n", tm.get_elapsed_ms());

		// Verify the compressed data by transcoding it to ASTC (or ETC1)/BC7 and validating the CRC's.
		basist::basisu_transcoder decoder(m_params.m_pSel_codebook);
		if (!decoder.validate_file_checksums(&comp_data[0], (uint32_t)comp_data.size(), true))
		{
			error_printf("decoder.validate_file_checksums() failed!\n");
			return false;
		}

		m_decoded_output_textures.resize(m_slice_descs.size());
		m_decoded_output_textures_unpacked.resize(m_slice_descs.size());

		m_decoded_output_textures_bc7.resize(m_slice_descs.size());
		m_decoded_output_textures_unpacked_bc7.resize(m_slice_descs.size());
								
		tm.start();
		if (m_params.m_pGlobal_codebooks)
		{
			decoder.set_global_codebooks(m_params.m_pGlobal_codebooks);
		}

		if (!decoder.start_transcoding(&comp_data[0], (uint32_t)comp_data.size()))
		{
			error_printf("decoder.start_transcoding() failed!\n");
			return false;
		}

		double start_transcoding_time = tm.get_elapsed_secs();

		debug_printf("basisu_compressor::start_transcoding() took %3.3fms\n", start_transcoding_time * 1000.0f);

		uint32_t total_orig_pixels = 0;
		uint32_t total_texels = 0;

		double total_time_etc1s_or_astc = 0;

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			gpu_image decoded_texture;
			decoded_texture.init(m_params.m_uastc ? texture_format::cASTC4x4 : texture_format::cETC1, m_slice_descs[i].m_width, m_slice_descs[i].m_height);
						
			tm.start();

			basist::block_format format = m_params.m_uastc ? basist::block_format::cASTC_4x4 : basist::block_format::cETC1;
			uint32_t bytes_per_block = m_params.m_uastc ? 16 : 8;
						
			if (!decoder.transcode_slice(&comp_data[0], (uint32_t)comp_data.size(), i,
				reinterpret_cast<etc_block *>(decoded_texture.get_ptr()), m_slice_descs[i].m_num_blocks_x * m_slice_descs[i].m_num_blocks_y, format, bytes_per_block))
			{
				error_printf("Transcoding failed on slice %u!\n", i);
				return false;
			}

			total_time_etc1s_or_astc += tm.get_elapsed_secs();

			if (encoded_output.m_tex_format == basist::basis_tex_format::cETC1S)
			{
				uint32_t image_crc16 = basist::crc16(decoded_texture.get_ptr(), decoded_texture.get_size_in_bytes(), 0);
				if (image_crc16 != encoded_output.m_slice_image_crcs[i])
				{
					error_printf("Decoded image data CRC check failed on slice %u!\n", i);
					return false;
				}
				debug_printf("Decoded image data CRC check succeeded on slice %i\n", i);
			}

			m_decoded_output_textures[i] = decoded_texture;

			total_orig_pixels += m_slice_descs[i].m_orig_width * m_slice_descs[i].m_orig_height;
			total_texels += m_slice_descs[i].m_width * m_slice_descs[i].m_height;
		}
												
		double total_time_bc7 = 0;

		if (basist::basis_is_format_supported(basist::transcoder_texture_format::cTFBC7_RGBA, basist::basis_tex_format::cUASTC4x4) &&
			basist::basis_is_format_supported(basist::transcoder_texture_format::cTFBC7_RGBA, basist::basis_tex_format::cETC1S))
		{
			for (uint32_t i = 0; i < m_slice_descs.size(); i++)
			{
				gpu_image decoded_texture;
				decoded_texture.init(texture_format::cBC7, m_slice_descs[i].m_width, m_slice_descs[i].m_height);

				tm.start();

				if (!decoder.transcode_slice(&comp_data[0], (uint32_t)comp_data.size(), i,
					reinterpret_cast<etc_block*>(decoded_texture.get_ptr()), m_slice_descs[i].m_num_blocks_x * m_slice_descs[i].m_num_blocks_y, basist::block_format::cBC7, 16))
				{
					error_printf("Transcoding failed to BC7 on slice %u!\n", i);
					return false;
				}

				total_time_bc7 += tm.get_elapsed_secs();

				m_decoded_output_textures_bc7[i] = decoded_texture;
			}
		}

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			m_decoded_output_textures[i].unpack(m_decoded_output_textures_unpacked[i]);

			if (m_decoded_output_textures_bc7[i].get_pixel_width())
				m_decoded_output_textures_bc7[i].unpack(m_decoded_output_textures_unpacked_bc7[i]);
		}

		debug_printf("Transcoded to %s in %3.3fms, %f texels/sec\n", m_params.m_uastc ? "ASTC" : "ETC1", total_time_etc1s_or_astc * 1000.0f, total_orig_pixels / total_time_etc1s_or_astc);

		if (total_time_bc7 != 0)
			debug_printf("Transcoded to BC7 in %3.3fms, %f texels/sec\n", total_time_bc7 * 1000.0f, total_orig_pixels / total_time_bc7);

		debug_printf("Total .basis output file size: %u, %3.3f bits/texel\n", comp_data.size(), comp_data.size() * 8.0f / total_orig_pixels);
				
		uint32_t total_orig_texels = 0;
		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];

			total_orig_texels += slice_desc.m_orig_width * slice_desc.m_orig_height;

			const uint32_t total_blocks = slice_desc.m_num_blocks_x * slice_desc.m_num_blocks_y;
			BASISU_NOTE_UNUSED(total_blocks);

			assert(m_decoded_output_textures[slice_index].get_total_blocks() == total_blocks);
		}

		m_basis_file_size = (uint32_t)comp_data.size();
		m_basis_bits_per_texel = (comp_data.size() * 8.0f) / total_orig_texels;

		return true;
	}

	bool basis_compressor::write_output_files_and_compute_stats()
	{
		debug_printf("basis_compressor::write_output_files_and_compute_stats\n");

		const uint8_vec& comp_data = m_params.m_create_ktx2_file ? m_output_ktx2_file : m_basis_file.get_compressed_data();
		if (m_params.m_write_output_basis_files)
		{
			const std::string& output_filename = m_params.m_out_filename;

			if (!write_vec_to_file(output_filename.c_str(), comp_data))
			{
				error_printf("Failed writing output data to file \"%s\"\n", output_filename.c_str());
				return false;
			}

			printf("Wrote output .basis/.ktx2 file \"%s\"\n", output_filename.c_str());
		}

		size_t comp_size = 0;
		if ((m_params.m_compute_stats) && (m_params.m_uastc) && (comp_data.size()))
		{
			void* pComp_data = tdefl_compress_mem_to_heap(&comp_data[0], comp_data.size(), &comp_size, TDEFL_MAX_PROBES_MASK);// TDEFL_DEFAULT_MAX_PROBES);
			size_t decomp_size = 0;
			void* pDecomp_data = tinfl_decompress_mem_to_heap(pComp_data, comp_size, &decomp_size, 0);
			if ((decomp_size != comp_data.size()) || (memcmp(pDecomp_data, &comp_data[0], decomp_size) != 0))
			{
				printf("basis_compressor::create_basis_file_and_transcode:: miniz compression or decompression failed!\n");
				return false;
			}

			mz_free(pComp_data);
			mz_free(pDecomp_data);

			uint32_t total_texels = 0;
			for (uint32_t i = 0; i < m_slice_descs.size(); i++)
				total_texels += (m_slice_descs[i].m_num_blocks_x * m_slice_descs[i].m_num_blocks_y) * 16;
			
			m_basis_bits_per_texel = comp_size * 8.0f / total_texels;

			debug_printf(".basis file size: %u, LZ compressed file size: %u, %3.2f bits/texel\n",
				(uint32_t)comp_data.size(),
				(uint32_t)comp_size,
				m_basis_bits_per_texel);
		}

		m_stats.resize(m_slice_descs.size());
		
		uint32_t total_orig_texels = 0;

		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];
						
			total_orig_texels += slice_desc.m_orig_width * slice_desc.m_orig_height;

			if (m_params.m_compute_stats)
			{
				printf("Slice: %u\n", slice_index);

				image_stats &s = m_stats[slice_index];

				// TODO: We used to output SSIM (during heavy encoder development), but this slowed down compression too much. We'll be adding it back.

				image_metrics em;
								
				// ---- .basis stats
				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 3);
				em.print(".basis RGB Avg:          ");
				s.m_basis_rgb_avg_psnr = em.m_psnr;

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 4);
				em.print(".basis RGBA Avg:         ");
				s.m_basis_rgba_avg_psnr = em.m_psnr;

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 1);
				em.print(".basis R   Avg:          ");
				
				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 1, 1);
				em.print(".basis G   Avg:          ");
				
				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 2, 1);
				em.print(".basis B   Avg:          ");

				if (m_params.m_uastc)
				{
					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 3, 1);
					em.print(".basis A   Avg:          ");

					s.m_basis_a_avg_psnr = em.m_psnr;
				}

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 0);
				em.print(".basis 709 Luma:         ");
				s.m_basis_luma_709_psnr = static_cast<float>(em.m_psnr);
				s.m_basis_luma_709_ssim = static_cast<float>(em.m_ssim);

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 0, true, true);
				em.print(".basis 601 Luma:         ");
				s.m_basis_luma_601_psnr = static_cast<float>(em.m_psnr);
								
				if (m_slice_descs.size() == 1)
				{
					const uint32_t output_size = comp_size ? (uint32_t)comp_size : (uint32_t)comp_data.size();
					debug_printf(".basis RGB PSNR per bit/texel*10000: %3.3f\n", 10000.0f * s.m_basis_rgb_avg_psnr / ((output_size * 8.0f) / (slice_desc.m_orig_width * slice_desc.m_orig_height)));
					debug_printf(".basis Luma 709 PSNR per bit/texel*10000: %3.3f\n", 10000.0f * s.m_basis_luma_709_psnr / ((output_size * 8.0f) / (slice_desc.m_orig_width * slice_desc.m_orig_height)));
				}

				if (m_decoded_output_textures_unpacked_bc7[slice_index].get_width())
				{
					// ---- BC7 stats
					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 0, 3);
					em.print("BC7 RGB Avg:             ");
					s.m_bc7_rgb_avg_psnr = em.m_psnr;

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 0, 4);
					em.print("BC7 RGBA Avg:            ");
					s.m_bc7_rgba_avg_psnr = em.m_psnr;

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 0, 1);
					em.print("BC7 R   Avg:             ");

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 1, 1);
					em.print("BC7 G   Avg:             ");

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 2, 1);
					em.print("BC7 B   Avg:             ");

					if (m_params.m_uastc)
					{
						em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 3, 1);
						em.print("BC7 A   Avg:             ");

						s.m_bc7_a_avg_psnr = em.m_psnr;
					}

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 0, 0);
					em.print("BC7 709 Luma:            ");
					s.m_bc7_luma_709_psnr = static_cast<float>(em.m_psnr);
					s.m_bc7_luma_709_ssim = static_cast<float>(em.m_ssim);

					em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc7[slice_index], 0, 0, true, true);
					em.print("BC7 601 Luma:            ");
					s.m_bc7_luma_601_psnr = static_cast<float>(em.m_psnr);
				}

				if (!m_params.m_uastc)
				{
					// ---- Nearly best possible ETC1S stats
					em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 0);
					em.print("Unquantized ETC1S 709 Luma:    ");

					s.m_best_etc1s_luma_709_psnr = static_cast<float>(em.m_psnr);
					s.m_best_etc1s_luma_709_ssim = static_cast<float>(em.m_ssim);

					em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 0, true, true);
					em.print("Unquantized ETC1S 601 Luma:    ");

					s.m_best_etc1s_luma_601_psnr = static_cast<float>(em.m_psnr);

					em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 3);
					em.print("Unquantized ETC1S RGB Avg:     ");

					s.m_best_etc1s_rgb_avg_psnr = static_cast<float>(em.m_psnr);
				}
			}
		
			std::string out_basename;
			if (m_params.m_out_filename.size())
				string_get_filename(m_params.m_out_filename.c_str(), out_basename);
			else if (m_params.m_source_filenames.size())
				string_get_filename(m_params.m_source_filenames[slice_desc.m_source_file_index].c_str(), out_basename);

			string_remove_extension(out_basename);
			out_basename = "basis_debug_" + out_basename + string_format("_slice_%u", slice_index);

			if ((!m_params.m_uastc) && (m_frontend.get_params().m_debug_images))
			{
				// Write "best" ETC1S debug images
				if (!m_params.m_uastc)
				{
					gpu_image best_etc1s_gpu_image(m_best_etc1s_images[slice_index]);
					best_etc1s_gpu_image.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_best_etc1s.ktx").c_str(), best_etc1s_gpu_image);

					image best_etc1s_unpacked;
					best_etc1s_gpu_image.unpack(best_etc1s_unpacked);
					save_png(out_basename + "_best_etc1s.png", best_etc1s_unpacked);
				}
			}

			if (m_params.m_debug_images)
			{
				// Write decoded ETC1S/ASTC debug images
				{
					gpu_image decoded_etc1s_or_astc(m_decoded_output_textures[slice_index]);
					decoded_etc1s_or_astc.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_transcoded_etc1s_or_astc.ktx").c_str(), decoded_etc1s_or_astc);

					image temp(m_decoded_output_textures_unpacked[slice_index]);
					temp.crop(slice_desc.m_orig_width, slice_desc.m_orig_height);
					save_png(out_basename + "_transcoded_etc1s_or_astc.png", temp);
				}

				// Write decoded BC7 debug images
				if (m_decoded_output_textures_bc7[slice_index].get_pixel_width())
				{
					gpu_image decoded_bc7(m_decoded_output_textures_bc7[slice_index]);
					decoded_bc7.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_transcoded_bc7.ktx").c_str(), decoded_bc7);

					image temp(m_decoded_output_textures_unpacked_bc7[slice_index]);
					temp.crop(slice_desc.m_orig_width, slice_desc.m_orig_height);
					save_png(out_basename + "_transcoded_bc7.png", temp);
				}
			}
		}
				
		return true;
	}
	
	// Make sure all the mip 0's have the same dimensions and number of mipmap levels, or we can't encode the KTX2 file.
	bool basis_compressor::validate_ktx2_constraints()
	{
		uint32_t base_width = 0, base_height = 0;
		uint32_t total_layers = 0;
		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			if (m_slice_descs[i].m_mip_index == 0)
			{
				if (!base_width)
				{
					base_width = m_slice_descs[i].m_orig_width;
					base_height = m_slice_descs[i].m_orig_height;
				}
				else
				{
					if ((m_slice_descs[i].m_orig_width != base_width) || (m_slice_descs[i].m_orig_height != base_height))
					{
						return false;
					}
				}

				total_layers = maximum<uint32_t>(total_layers, m_slice_descs[i].m_source_file_index + 1);
			}
		}

		basisu::vector<uint32_t> total_mips(total_layers);
		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
			total_mips[m_slice_descs[i].m_source_file_index] = maximum<uint32_t>(total_mips[m_slice_descs[i].m_source_file_index], m_slice_descs[i].m_mip_index + 1);

		for (uint32_t i = 1; i < total_layers; i++)
		{
			if (total_mips[0] != total_mips[i])
			{
				return false;
			}
		}

		return true;
	}

	static uint8_t g_ktx2_etc1s_nonalpha_dfd[44] = { 0x2C,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x28,0x0,0xA3,0x1,0x2,0x0,0x3,0x3,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x3F,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xFF,0xFF,0xFF,0xFF };
	static uint8_t g_ktx2_etc1s_alpha_dfd[60] = { 0x3C,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x38,0x0,0xA3,0x1,0x2,0x0,0x3,0x3,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x3F,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xFF,0xFF,0xFF,0xFF,0x40,0x0,0x3F,0xF,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xFF,0xFF,0xFF,0xFF };
	static uint8_t g_ktx2_uastc_nonalpha_dfd[44] = { 0x2C,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x28,0x0,0xA6,0x1,0x2,0x0,0x3,0x3,0x0,0x0,0x10,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x7F,0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xFF,0xFF,0xFF,0xFF };
	static uint8_t g_ktx2_uastc_alpha_dfd[44] = { 0x2C,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x0,0x28,0x0,0xA6,0x1,0x2,0x0,0x3,0x3,0x0,0x0,0x10,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x7F,0x3,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xFF,0xFF,0xFF,0xFF };
		
	void basis_compressor::get_dfd(uint8_vec &dfd, const basist::ktx2_header &header)
	{
		const uint8_t* pDFD;
		uint32_t dfd_len;

		if (m_params.m_uastc)
		{
			if (m_any_source_image_has_alpha)
			{
				pDFD = g_ktx2_uastc_alpha_dfd;
				dfd_len = sizeof(g_ktx2_uastc_alpha_dfd);
			}
			else
			{
				pDFD = g_ktx2_uastc_nonalpha_dfd;
				dfd_len = sizeof(g_ktx2_uastc_nonalpha_dfd);
			}
		}
		else
		{
			if (m_any_source_image_has_alpha)
			{
				pDFD = g_ktx2_etc1s_alpha_dfd;
				dfd_len = sizeof(g_ktx2_etc1s_alpha_dfd);
			}
			else
			{
				pDFD = g_ktx2_etc1s_nonalpha_dfd;
				dfd_len = sizeof(g_ktx2_etc1s_nonalpha_dfd);
			}
		}
				
		assert(dfd_len >= 44);

		dfd.resize(dfd_len);
		memcpy(dfd.data(), pDFD, dfd_len);

		uint32_t dfd_bits = basisu::read_le_dword(dfd.data() + 3 * sizeof(uint32_t));
		
		dfd_bits &= ~(0xFF << 16);

		if (m_params.m_ktx2_srgb_transfer_func)
			dfd_bits |= (basist::KTX2_KHR_DF_TRANSFER_SRGB << 16);
		else
			dfd_bits |= (basist::KTX2_KHR_DF_TRANSFER_LINEAR << 16);

		basisu::write_le_dword(dfd.data() + 3 * sizeof(uint32_t), dfd_bits);

		if (header.m_supercompression_scheme != basist::KTX2_SS_NONE)
		{
			uint32_t plane_bits = basisu::read_le_dword(dfd.data() + 5 * sizeof(uint32_t));

			plane_bits &= ~0xFF;

			basisu::write_le_dword(dfd.data() + 5 * sizeof(uint32_t), plane_bits);
		}

		// Fix up the DFD channel(s)
		uint32_t dfd_chan0 = basisu::read_le_dword(dfd.data() + 7 * sizeof(uint32_t));

		if (m_params.m_uastc)
		{
			dfd_chan0 &= ~(0xF << 24);
			
			// TODO: Allow the caller to override this
			if (m_any_source_image_has_alpha)
				dfd_chan0 |= (basist::KTX2_DF_CHANNEL_UASTC_RGBA << 24);
			else
				dfd_chan0 |= (basist::KTX2_DF_CHANNEL_UASTC_RGB << 24);
		}

		basisu::write_le_dword(dfd.data() + 7 * sizeof(uint32_t), dfd_chan0);
	}

	bool basis_compressor::create_ktx2_file()
	{
		if (m_params.m_uastc)
		{
			if ((m_params.m_ktx2_uastc_supercompression != basist::KTX2_SS_NONE) && (m_params.m_ktx2_uastc_supercompression != basist::KTX2_SS_ZSTANDARD))
				return false;
		}

		const basisu_backend_output& backend_output = m_backend.get_output();

		// Determine the width/height, number of array layers, mipmap levels, and the number of faces (1 for 2D, 6 for cubemap).
		// This does not support 1D or 3D.
		uint32_t base_width = 0, base_height = 0, total_layers = 0, total_levels = 0, total_faces = 1;
				
		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			if ((m_slice_descs[i].m_mip_index == 0) && (!base_width))
			{
				base_width = m_slice_descs[i].m_orig_width;
				base_height = m_slice_descs[i].m_orig_height;
			}

			total_layers = maximum<uint32_t>(total_layers, m_slice_descs[i].m_source_file_index + 1);

			if (!m_slice_descs[i].m_source_file_index)
				total_levels = maximum<uint32_t>(total_levels, m_slice_descs[i].m_mip_index + 1);
		}

		if (m_params.m_tex_type == basist::cBASISTexTypeCubemapArray)
		{
			assert((total_layers % 6) == 0);
			
			total_layers /= 6;
			assert(total_layers >= 1);

			total_faces = 6;
		}

		basist::ktx2_header header;
		memset(&header, 0, sizeof(header));

		memcpy(header.m_identifier, basist::g_ktx2_file_identifier, sizeof(basist::g_ktx2_file_identifier));
		header.m_pixel_width = base_width;
		header.m_pixel_height = base_height;
		header.m_face_count = total_faces;
		header.m_vk_format = basist::KTX2_VK_FORMAT_UNDEFINED;
		header.m_type_size = 1;
		header.m_level_count = total_levels;
		header.m_layer_count = (total_layers > 1) ? total_layers : 0;

		if (m_params.m_uastc)
		{
			switch (m_params.m_ktx2_uastc_supercompression)
			{
			case basist::KTX2_SS_NONE:
			{
				header.m_supercompression_scheme = basist::KTX2_SS_NONE;
				break;
			}
			case basist::KTX2_SS_ZSTANDARD:
			{
#if BASISD_SUPPORT_KTX2_ZSTD
				header.m_supercompression_scheme = basist::KTX2_SS_ZSTANDARD;
#else
				header.m_supercompression_scheme = basist::KTX2_SS_NONE;
#endif
				break;
			}
			default: assert(0); return false;
			}
		}

		basisu::vector<uint8_vec> level_data_bytes(total_levels);
		basisu::vector<uint8_vec> compressed_level_data_bytes(total_levels);
		uint_vec slice_level_offsets(m_slice_descs.size());

		// This will append the texture data in the correct order (for each level: layer, then face).
		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
		{
			const basisu_backend_slice_desc& slice_desc = m_slice_descs[slice_index];

			slice_level_offsets[slice_index] = level_data_bytes[slice_desc.m_mip_index].size();

			if (m_params.m_uastc)
				append_vector(level_data_bytes[slice_desc.m_mip_index], m_uastc_backend_output.m_slice_image_data[slice_index]);
			else
				append_vector(level_data_bytes[slice_desc.m_mip_index], backend_output.m_slice_image_data[slice_index]);
		}

		// UASTC supercompression
		if ((m_params.m_uastc) && (header.m_supercompression_scheme == basist::KTX2_SS_ZSTANDARD))
		{
#if BASISD_SUPPORT_KTX2_ZSTD
			for (uint32_t level_index = 0; level_index < total_levels; level_index++)
			{
				compressed_level_data_bytes[level_index].resize(ZSTD_compressBound(level_data_bytes[level_index].size()));

				size_t result = ZSTD_compress(compressed_level_data_bytes[level_index].data(), compressed_level_data_bytes[level_index].size(),
					level_data_bytes[level_index].data(), level_data_bytes[level_index].size(),
					m_params.m_ktx2_zstd_supercompression_level);

				if (ZSTD_isError(result))
					return false;

				compressed_level_data_bytes[level_index].resize(result);
			}
#else
			// Can't get here
			assert(0);
			return false;
#endif
		}
		else
		{
			// No supercompression
			compressed_level_data_bytes = level_data_bytes;
		}
				
		uint8_vec etc1s_global_data;

		// Create ETC1S global supercompressed data
		if (!m_params.m_uastc)
		{
			basist::ktx2_etc1s_global_data_header etc1s_global_data_header;
			clear_obj(etc1s_global_data_header);

			etc1s_global_data_header.m_endpoint_count = backend_output.m_num_endpoints;
			etc1s_global_data_header.m_selector_count = backend_output.m_num_selectors;
			etc1s_global_data_header.m_endpoints_byte_length = backend_output.m_endpoint_palette.size();
			etc1s_global_data_header.m_selectors_byte_length = backend_output.m_selector_palette.size();
			etc1s_global_data_header.m_tables_byte_length = backend_output.m_slice_image_tables.size();

			basisu::vector<basist::ktx2_etc1s_image_desc> etc1s_image_descs(total_levels * total_layers * total_faces);
			memset(etc1s_image_descs.data(), 0, etc1s_image_descs.size_in_bytes());

			for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
			{
				const basisu_backend_slice_desc& slice_desc = m_slice_descs[slice_index];

				const uint32_t level_index = slice_desc.m_mip_index;
				uint32_t layer_index = slice_desc.m_source_file_index;
				uint32_t face_index = 0;

				if (m_params.m_tex_type == basist::cBASISTexTypeCubemapArray)
				{
					face_index = layer_index % 6;
					layer_index /= 6;
				}

				const uint32_t etc1s_image_index = level_index * (total_layers * total_faces) + layer_index * total_faces + face_index;

				if (slice_desc.m_alpha)
				{
					etc1s_image_descs[etc1s_image_index].m_alpha_slice_byte_length = backend_output.m_slice_image_data[slice_index].size();
					etc1s_image_descs[etc1s_image_index].m_alpha_slice_byte_offset = slice_level_offsets[slice_index];
				}
				else
				{
					if (m_params.m_tex_type == basist::cBASISTexTypeVideoFrames)
						etc1s_image_descs[etc1s_image_index].m_image_flags = !slice_desc.m_iframe ? basist::KTX2_IMAGE_IS_P_FRAME : 0;

					etc1s_image_descs[etc1s_image_index].m_rgb_slice_byte_length = backend_output.m_slice_image_data[slice_index].size();
					etc1s_image_descs[etc1s_image_index].m_rgb_slice_byte_offset = slice_level_offsets[slice_index];
				}
			} // slice_index

			append_vector(etc1s_global_data, (const uint8_t*)&etc1s_global_data_header, sizeof(etc1s_global_data_header));
			append_vector(etc1s_global_data, (const uint8_t*)etc1s_image_descs.data(), etc1s_image_descs.size_in_bytes());
			append_vector(etc1s_global_data, backend_output.m_endpoint_palette);
			append_vector(etc1s_global_data, backend_output.m_selector_palette);
			append_vector(etc1s_global_data, backend_output.m_slice_image_tables);
			
			header.m_supercompression_scheme = basist::KTX2_SS_BASISLZ;
		}

		// Key values
		basist::ktx2_transcoder::key_value_vec key_values(m_params.m_ktx2_key_values);
		key_values.enlarge(1);
		
		const char* pKTXwriter = "KTXwriter";
		key_values.back().m_key.resize(strlen(pKTXwriter) + 1);
		memcpy(key_values.back().m_key.data(), pKTXwriter, strlen(pKTXwriter) + 1);

		char writer_id[128];
#ifdef _MSC_VER
		sprintf_s(writer_id, sizeof(writer_id), "Basis Universal %s", BASISU_LIB_VERSION_STRING);
#else
		snprintf(writer_id, sizeof(writer_id), "Basis Universal %s", BASISU_LIB_VERSION_STRING);
#endif
		key_values.back().m_value.resize(strlen(writer_id) + 1);
		memcpy(key_values.back().m_value.data(), writer_id, strlen(writer_id) + 1);

		key_values.sort();

#if BASISU_DISABLE_KTX2_KEY_VALUES
		// HACK HACK - Clear the key values array, which causes no key values to be written (triggering the ktx2check validator bug).
		key_values.clear();
#endif

		uint8_vec key_value_data;

		// DFD
		uint8_vec dfd;
		get_dfd(dfd, header);

		const uint32_t kvd_file_offset = sizeof(header) + sizeof(basist::ktx2_level_index) * total_levels + dfd.size();

		for (uint32_t pass = 0; pass < 2; pass++)
		{
			for (uint32_t i = 0; i < key_values.size(); i++)
			{
				if (key_values[i].m_key.size() < 2)
					return false;

				if (key_values[i].m_key.back() != 0)
					return false;

				const uint64_t total_len = (uint64_t)key_values[i].m_key.size() + (uint64_t)key_values[i].m_value.size();
				if (total_len >= UINT32_MAX)
					return false;

				packed_uint<4> le_len((uint32_t)total_len);
				append_vector(key_value_data, (const uint8_t*)&le_len, sizeof(le_len));

				append_vector(key_value_data, key_values[i].m_key);
				append_vector(key_value_data, key_values[i].m_value);

				const uint32_t ofs = key_value_data.size() & 3;
				const uint32_t padding = (4 - ofs) & 3;
				for (uint32_t p = 0; p < padding; p++)
					key_value_data.push_back(0);
			}

			if (header.m_supercompression_scheme != basist::KTX2_SS_NONE)
				break;

#if BASISU_DISABLE_KTX2_ALIGNMENT_WORKAROUND
			break;
#endif
			
			// Hack to ensure the KVD block ends on a 16 byte boundary, because we have no other official way of aligning the data.
			uint32_t kvd_end_file_offset = kvd_file_offset + key_value_data.size();
			uint32_t bytes_needed_to_pad = (16 - (kvd_end_file_offset & 15)) & 15;
			if (!bytes_needed_to_pad)
			{
				// We're good. No need to add a dummy key.
				break;
			}

			assert(!pass);
			if (pass)
				return false;

			if (bytes_needed_to_pad < 6)
				bytes_needed_to_pad += 16;

			printf("WARNING: Due to a KTX2 validator bug related to mipPadding, we must insert a dummy key into the KTX2 file of %u bytes\n", bytes_needed_to_pad);
			
			// We're not good - need to add a dummy key large enough to force file alignment so the mip level array gets aligned. 
			// We can't just add some bytes before the mip level array because ktx2check will see that as extra data in the file that shouldn't be there in ktxValidator::validateDataSize().
			key_values.enlarge(1);
			for (uint32_t i = 0; i < (bytes_needed_to_pad - 4 - 1 - 1); i++)
				key_values.back().m_key.push_back(127);
			
			key_values.back().m_key.push_back(0);

			key_values.back().m_value.push_back(0);

			key_values.sort();

			key_value_data.resize(0);
			
			// Try again
		}

		basisu::vector<basist::ktx2_level_index> level_index_array(total_levels);
		memset(level_index_array.data(), 0, level_index_array.size_in_bytes());
				
		m_output_ktx2_file.clear();
		m_output_ktx2_file.reserve(m_output_basis_file.size());

		// Dummy header
		m_output_ktx2_file.resize(sizeof(header));

		// Level index array
		append_vector(m_output_ktx2_file, (const uint8_t*)level_index_array.data(), level_index_array.size_in_bytes());
				
		// DFD
		const uint8_t* pDFD = dfd.data();
		uint32_t dfd_len = dfd.size();

		header.m_dfd_byte_offset = m_output_ktx2_file.size();
		header.m_dfd_byte_length = dfd_len;
		append_vector(m_output_ktx2_file, pDFD, dfd_len);

		// Key value data
		if (key_value_data.size())
		{
			assert(kvd_file_offset == m_output_ktx2_file.size());

			header.m_kvd_byte_offset = m_output_ktx2_file.size();
			header.m_kvd_byte_length = key_value_data.size();
			append_vector(m_output_ktx2_file, key_value_data);
		}

		// Global Supercompressed Data
		if (etc1s_global_data.size())
		{
			uint32_t ofs = m_output_ktx2_file.size() & 7;
			uint32_t padding = (8 - ofs) & 7;
			for (uint32_t i = 0; i < padding; i++)
				m_output_ktx2_file.push_back(0);

			header.m_sgd_byte_length = etc1s_global_data.size();
			header.m_sgd_byte_offset = m_output_ktx2_file.size();

			append_vector(m_output_ktx2_file, etc1s_global_data);
		}

		// mipPadding
		if (header.m_supercompression_scheme == basist::KTX2_SS_NONE)
		{
			// We currently can't do this or the validator will incorrectly give an error.
			uint32_t ofs = m_output_ktx2_file.size() & 15;
			uint32_t padding = (16 - ofs) & 15;

			// Make sure we're always aligned here (due to a validator bug).
			if (padding)
			{
				printf("Warning: KTX2 mip level data is not 16-byte aligned. This may trigger a ktx2check validation bug. Writing %u bytes of mipPadding.\n", padding);
			}

			for (uint32_t i = 0; i < padding; i++)
				m_output_ktx2_file.push_back(0);
		}

		// Level data - write the smallest mipmap first.
		for (int level = total_levels - 1; level >= 0; level--)
		{
			level_index_array[level].m_byte_length = compressed_level_data_bytes[level].size();
			if (m_params.m_uastc)
				level_index_array[level].m_uncompressed_byte_length = level_data_bytes[level].size();

			level_index_array[level].m_byte_offset = m_output_ktx2_file.size();
			append_vector(m_output_ktx2_file, compressed_level_data_bytes[level]);
		}
		
		// Write final header
		memcpy(m_output_ktx2_file.data(), &header, sizeof(header));

		// Write final level index array
		memcpy(m_output_ktx2_file.data() + sizeof(header), level_index_array.data(), level_index_array.size_in_bytes());

		debug_printf("Total .ktx2 output file size: %u\n", m_output_ktx2_file.size());

		return true;
	}

} // namespace basisu
