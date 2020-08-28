// basisu_comp.cpp
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
#include "basisu_comp.h"
#include "basisu_enc.h"
#include <unordered_set>

#define BASISU_USE_STB_IMAGE_RESIZE_FOR_MIPMAP_GEN 0
#define DEBUG_CROP_TEXTURE_TO_64x64 (0)
#define DEBUG_RESIZE_TEXTURE (0)
#define DEBUG_EXTRACT_SINGLE_BLOCK (0)

namespace basisu
{
   basis_compressor::basis_compressor() :
		m_total_blocks(0),
		m_auto_global_sel_pal(false),
		m_basis_file_size(0),
		m_basis_bits_per_texel(0),
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

			debug_printf("Source images: %u, source filenames: %u, source alpha filenames: %i\n", 
				(uint32_t)m_params.m_source_images.size(), (uint32_t)m_params.m_source_filenames.size(), (uint32_t)m_params.m_source_alpha_filenames.size());

			PRINT_BOOL_VALUE(m_y_flip);
			PRINT_BOOL_VALUE(m_debug);
			PRINT_BOOL_VALUE(m_debug_images);
			PRINT_BOOL_VALUE(m_global_sel_pal);
			PRINT_BOOL_VALUE(m_auto_global_sel_pal);
			PRINT_BOOL_VALUE(m_compression_level);
			PRINT_BOOL_VALUE(m_no_hybrid_sel_cb);
			PRINT_BOOL_VALUE(m_perceptual);
			PRINT_BOOL_VALUE(m_no_endpoint_rdo);
			PRINT_BOOL_VALUE(m_no_selector_rdo);
			PRINT_BOOL_VALUE(m_read_source_images);
			PRINT_BOOL_VALUE(m_write_output_basis_files);
			PRINT_BOOL_VALUE(m_compute_stats);
			PRINT_BOOL_VALUE(m_check_for_alpha)
			PRINT_BOOL_VALUE(m_force_alpha)
			PRINT_BOOL_VALUE(m_seperate_rg_to_color_alpha);
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

		if (!process_frontend())
			return cECFailedFrontEnd;

		if (!extract_frontend_texture_data())
			return cECFailedFontendExtract;

		if (!process_backend())
			return cECFailedBackend;

		if (!create_basis_file_and_transcode())
			return cECFailedCreateBasisFile;

		if (!write_output_files_and_compute_stats())
			return cECFailedWritingOutput;

		return cECSuccess;
	}

	bool basis_compressor::generate_mipmaps(const image &img, std::vector<image> &mips, bool has_alpha)
	{
		debug_printf("basis_compressor::generate_mipmaps\n");

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

			image &level_img = *enlarge_vector(mips, 1);
			level_img.resize(level_width, level_height);

			bool status = image_resample(img, level_img, m_params.m_mip_srgb, m_params.m_mip_filter.c_str(), m_params.m_mip_scale, m_params.m_mip_wrapping, 0, has_alpha ? 4 : 3);
			if (!status)
			{
				error_printf("basis_compressor::generate_mipmaps: image_resample() failed!\n");
				return false;
			}

			if (m_params.m_mip_renormalize)
				level_img.renormalize_normal_map();
		}
#endif

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

		std::vector<image> source_images;
		std::vector<std::string> source_filenames;
		
		// First load all source images, and determine if any have an alpha channel.
		for (uint32_t source_file_index = 0; source_file_index < total_source_files; source_file_index++)
		{
			const char *pSource_filename = "";

			image file_image;
			
			if (m_params.m_read_source_images)
			{
				pSource_filename = m_params.m_source_filenames[source_file_index].c_str();

				// Load the source image
				if (!load_png(pSource_filename, file_image))
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

					if (!load_png(pSource_alpha_image, alpha_data))
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

			if (m_params.m_seperate_rg_to_color_alpha)
			{
				// Used for XY normal maps in RG - puts X in color, Y in alpha
				for (uint32_t y = 0; y < file_image.get_height(); y++)
					for (uint32_t x = 0; x < file_image.get_width(); x++)
					{
						const color_rgba &c = file_image(x, y);
						file_image(x, y).set_noclamp_rgba(c.r, c.r, c.r, c.g);
					}
			}
						
			bool has_alpha = false;
			if ((m_params.m_force_alpha) || (m_params.m_seperate_rg_to_color_alpha))
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
#if DEBUG_RESIZE_TEXTURE
			image temp_img((file_image.get_width() + 1) / 2, (file_image.get_height() + 1) / 2);
			image_resample(file_image, temp_img, m_params.m_perceptual, "kaiser");
			temp_img.swap(file_image);
#endif

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

		debug_printf("Any source image has alpha: %u\n", m_any_source_image_has_alpha);

		for (uint32_t source_file_index = 0; source_file_index < total_source_files; source_file_index++)
		{
			image &file_image = source_images[source_file_index];
			const std::string &source_filename = source_filenames[source_file_index];

			std::vector<image> slices;
			
			slices.reserve(32);
			slices.push_back(file_image);
									
			if (m_params.m_mip_gen)
			{
				if (!generate_mipmaps(file_image, slices, m_any_source_image_has_alpha))
					return false;
			}

			uint_vec mip_indices(slices.size());
			for (uint32_t i = 0; i < slices.size(); i++)
				mip_indices[i] = i;
						
			if (m_any_source_image_has_alpha)
			{
				// If source has alpha, then even mips will have RGB, and odd mips will have alpha in RGB. 
				std::vector<image> alpha_slices;
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
				const bool is_alpha_slice = m_any_source_image_has_alpha && ((slice_index & 1) != 0);

				image &slice_image = slices[slice_index];
				const uint32_t orig_width = slice_image.get_width();
				const uint32_t orig_height = slice_image.get_height();

				// Enlarge the source image to 4x4 block boundaries, duplicating edge pixels if necessary to avoid introducing extra colors into blocks.
				slice_image.crop_dup_borders(slice_image.get_block_width(4) * 4, slice_image.get_block_height(4) * 4);

				if (m_params.m_debug_images)
				{
					save_png(string_format("basis_debug_source_image_%u_%u.png", source_file_index, slice_index).c_str(), slice_image);
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

		printf("Total basis file slices: %u\n", (uint32_t)m_slice_descs.size());

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[i];

			printf("Slice: %u, alpha: %u, orig width/height: %ux%u, width/height: %ux%u, first_block: %u, image_index: %u, mip_level: %u, iframe: %u\n", 
				i, slice_desc.m_alpha, slice_desc.m_orig_width, slice_desc.m_orig_height, slice_desc.m_width, slice_desc.m_height, slice_desc.m_first_block_index, slice_desc.m_source_file_index, slice_desc.m_mip_index, slice_desc.m_iframe);

			if (m_any_source_image_has_alpha)
			{
				// Alpha slices must be at odd slice indices
				if (slice_desc.m_alpha)
				{
					if ((i & 1) == 0)
						return false;
					
					const basisu_backend_slice_desc &prev_slice_desc = m_slice_descs[i - 1];

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

	bool basis_compressor::process_frontend()
	{
		debug_printf("basis_compressor::process_frontend\n");
				
		m_source_blocks.resize(m_total_blocks);
				
		for (uint32_t slice_index = 0; slice_index < m_slice_images.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];

			const uint32_t num_blocks_x = slice_desc.m_num_blocks_x;
			const uint32_t num_blocks_y = slice_desc.m_num_blocks_y;

			const image &source_image = m_slice_images[slice_index];

			for (uint32_t block_y = 0; block_y < num_blocks_y; block_y++)
				for (uint32_t block_x = 0; block_x < num_blocks_x; block_x++)
					source_image.extract_block_clamped(m_source_blocks[slice_desc.m_first_block_index + block_x + block_y * num_blocks_x].get_ptr(), block_x * 4, block_y * 4, 4, 4);
		}
				
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
			if (color_endpoint_quality <= mid)
			{
				color_endpoint_quality = lerp(0.0f, endpoint_split_point, powf(color_endpoint_quality / mid, .65f));

				max_endpoints = clamp<int>(max_endpoints, 256, 3072);
				max_endpoints = minimum<uint32_t>(max_endpoints, m_total_blocks);
								
				if (max_endpoints < 64)
					max_endpoints = 64;
				endpoint_clusters = clamp<uint32_t>((uint32_t)(.5f + lerp<float>(32, static_cast<float>(max_endpoints), color_endpoint_quality)), 32, basisu_frontend::cMaxEndpointClusters);
			}
			else
			{
				color_endpoint_quality = powf((color_endpoint_quality - mid) / (1.0f - mid), 1.6f);

				max_endpoints = clamp<int>(max_endpoints, 256, 8192);
				max_endpoints = minimum<uint32_t>(max_endpoints, m_total_blocks);
								
				if (max_endpoints < 3072)
					max_endpoints = 3072;
				endpoint_clusters = clamp<uint32_t>((uint32_t)(.5f + lerp<float>(3072, static_cast<float>(max_endpoints), color_endpoint_quality)), 32, basisu_frontend::cMaxEndpointClusters);
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
		p.m_pJob_pool = m_params.m_pJob_pool;

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

		const basisu_backend_output &encoded_output = m_backend.get_output();

		if (!m_basis_file.init(encoded_output, m_params.m_tex_type, m_params.m_userdata0, m_params.m_userdata1, m_params.m_y_flip, m_params.m_us_per_frame))
		{
			error_printf("basis_compressor::write_output_files_and_compute_stats: basisu_backend:init() failed!\n");
			return false;
		}

		const uint8_vec &comp_data = m_basis_file.get_compressed_data();

		m_output_basis_file = comp_data;

		// Verify the compressed data by transcoding it to ETC1/BC1 and validating the CRC's.
		basist::basisu_transcoder decoder(m_params.m_pSel_codebook);
		if (!decoder.validate_file_checksums(&comp_data[0], (uint32_t)comp_data.size(), true))
		{
			error_printf("decoder.validate_file_checksums() failed!\n");
			return false;
		}

		m_decoded_output_textures.resize(m_slice_descs.size());
		m_decoded_output_textures_unpacked.resize(m_slice_descs.size());

		m_decoded_output_textures_bc1.resize(m_slice_descs.size());
		m_decoded_output_textures_unpacked_bc1.resize(m_slice_descs.size());

		interval_timer tm;
		tm.start();

		if (!decoder.start_transcoding(&comp_data[0], (uint32_t)comp_data.size()))
		{
			error_printf("decoder.start_transcoding() failed!\n");
			return false;
		}

		debug_printf("basisu_comppressor::start_transcoding() took %3.3fms\n", tm.get_elapsed_ms());

		uint32_t total_orig_pixels = 0;
		uint32_t total_texels = 0;

		double total_time_etc1 = 0;

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			gpu_image decoded_texture;
			decoded_texture.init(texture_format::cETC1, m_slice_descs[i].m_width, m_slice_descs[i].m_height);
						
			tm.start();
						
			if (!decoder.transcode_slice(&comp_data[0], (uint32_t)comp_data.size(), i,
				reinterpret_cast<etc_block *>(decoded_texture.get_ptr()), m_slice_descs[i].m_num_blocks_x * m_slice_descs[i].m_num_blocks_y, basist::block_format::cETC1, 8))
			{
				error_printf("Transcoding failed to ETC1 on slice %u!\n", i);
				return false;
			}

			total_time_etc1 += tm.get_elapsed_secs();

			uint32_t image_crc16 = basist::crc16(decoded_texture.get_ptr(), decoded_texture.get_size_in_bytes(), 0);
			if (image_crc16 != m_backend.get_output().m_slice_image_crcs[i])
			{
				error_printf("Decoded image data CRC check failed on slice %u!\n", i);
				return false;
			}
			debug_printf("Decoded image data CRC check succeeded on slice %i\n", i);

			m_decoded_output_textures[i] = decoded_texture;

			total_orig_pixels += m_slice_descs[i].m_orig_width * m_slice_descs[i].m_orig_height;
			total_texels += m_slice_descs[i].m_width * m_slice_descs[i].m_height;
		}

		tm.start();
				
		basist::basisu_transcoder_init();

		debug_printf("basist::basisu_transcoder_init: Took %f ms\n", tm.get_elapsed_ms());
				
		double total_time_bc1 = 0;

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			gpu_image decoded_texture;
			decoded_texture.init(texture_format::cBC1, m_slice_descs[i].m_width, m_slice_descs[i].m_height);

			tm.start();

			if (!decoder.transcode_slice(&comp_data[0], (uint32_t)comp_data.size(), i,
				reinterpret_cast<etc_block *>(decoded_texture.get_ptr()), m_slice_descs[i].m_num_blocks_x * m_slice_descs[i].m_num_blocks_y, basist::block_format::cBC1, 8))
			{
				error_printf("Transcoding failed to BC1 on slice %u!\n", i);
				return false;
			}

			total_time_bc1 += tm.get_elapsed_secs();

			m_decoded_output_textures_bc1[i] = decoded_texture;
		}

		for (uint32_t i = 0; i < m_slice_descs.size(); i++)
		{
			m_decoded_output_textures[i].unpack(m_decoded_output_textures_unpacked[i]);
			m_decoded_output_textures_bc1[i].unpack(m_decoded_output_textures_unpacked_bc1[i]);
		}

		debug_printf("Transcoded to ETC1 in %3.3fms, %f texels/sec\n", total_time_etc1 * 1000.0f, total_orig_pixels / total_time_etc1);

		debug_printf("Transcoded to BC1 in %3.3fms, %f texels/sec\n", total_time_bc1 * 1000.0f, total_orig_pixels / total_time_bc1);

		debug_printf("Total .basis output file size: %u, %3.3f bits/texel\n", comp_data.size(), comp_data.size() * 8.0f / total_orig_pixels);

		m_output_blocks.resize(0);

		uint32_t total_orig_texels = 0;
		for (uint32_t slice_index = 0; slice_index < m_slice_descs.size(); slice_index++)
		{
			const basisu_backend_slice_desc &slice_desc = m_slice_descs[slice_index];

			total_orig_texels += slice_desc.m_orig_width * slice_desc.m_orig_height;

			const uint32_t total_blocks = slice_desc.m_num_blocks_x * slice_desc.m_num_blocks_y;

			assert(m_decoded_output_textures[slice_index].get_total_blocks() == total_blocks);

			memcpy(enlarge_vector(m_output_blocks, total_blocks), m_decoded_output_textures[slice_index].get_ptr(), sizeof(etc_block) * total_blocks);
		}

		m_basis_file_size = (uint32_t)comp_data.size();
		m_basis_bits_per_texel = (comp_data.size() * 8.0f) / total_orig_texels;

		return true;
	}

	bool basis_compressor::write_output_files_and_compute_stats()
	{
		debug_printf("basis_compressor::write_output_files_and_compute_stats\n");

		if (m_params.m_write_output_basis_files)
		{
			const uint8_vec &comp_data = m_basis_file.get_compressed_data();

			const std::string& basis_filename = m_params.m_out_filename;

			if (!write_vec_to_file(basis_filename.c_str(), comp_data))
			{
				error_printf("Failed writing output data to file \"%s\"\n", basis_filename.c_str());
				return false;
			}

			printf("Wrote output .basis file \"%s\"\n", basis_filename.c_str());
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
								
				// ---- .basis ETC1S stats
				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 0);
				em.print(".basis ETC1S 709 Luma:         ");
								
				s.m_basis_etc1s_luma_709_psnr = static_cast<float>(em.m_psnr);
				s.m_basis_etc1s_luma_709_ssim = static_cast<float>(em.m_ssim);

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 0, true, true);
				em.print(".basis ETC1S 601 Luma:         ");

				s.m_basis_etc1s_luma_601_psnr = static_cast<float>(em.m_psnr);

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked[slice_index], 0, 3);
				em.print(".basis ETC1S RGB Avg:          ");

				s.m_basis_etc1s_rgb_avg_psnr = em.m_psnr;

				if (m_slice_descs.size() == 1)
				{
					debug_printf(".basis Luma 709 PSNR per bit/texel*10000: %3.3f\n", 10000.0f * s.m_basis_etc1s_luma_709_psnr / ((m_backend.get_output().get_output_size_estimate() * 8.0f) / (slice_desc.m_orig_width * slice_desc.m_orig_height)));
				}

				// ---- .basis BC1 stats
				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc1[slice_index], 0, 0);
				em.print(".basis BC1 709 Luma:           ");
								
				s.m_basis_bc1_luma_709_psnr = static_cast<float>(em.m_psnr);
				s.m_basis_bc1_luma_709_ssim = static_cast<float>(em.m_ssim);

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc1[slice_index], 0, 0, true, true);
				em.print(".basis BC1 601 Luma:           ");

				s.m_basis_bc1_luma_601_psnr = static_cast<float>(em.m_psnr);

				em.calc(m_slice_images[slice_index], m_decoded_output_textures_unpacked_bc1[slice_index], 0, 3);
				em.print(".basis BC1 RGB Avg:            ");

				s.m_basis_bc1_rgb_avg_psnr = static_cast<float>(em.m_psnr);

				// ---- Nearly best possible ETC1S stats
				em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 0);
				em.print("Unquantized ETC1S 709 Luma:    ");

				s.m_best_luma_709_psnr = static_cast<float>(em.m_psnr);
				s.m_best_luma_709_ssim = static_cast<float>(em.m_ssim);

				em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 0, true, true);
				em.print("Unquantized ETC1S 601 Luma:    ");

				s.m_best_luma_601_psnr = static_cast<float>(em.m_psnr);
				
				em.calc(m_slice_images[slice_index], m_best_etc1s_images_unpacked[slice_index], 0, 3);
				em.print("Unquantized ETC1S RGB Avg:     ");

				s.m_best_rgb_avg_psnr = static_cast<float>(em.m_psnr);
			}
		
			if (m_frontend.get_params().m_debug_images)
			{
				std::string out_basename;
				if (m_params.m_out_filename.size())
					string_get_filename(m_params.m_out_filename.c_str(), out_basename);
				else if (m_params.m_source_filenames.size())
					string_get_filename(m_params.m_source_filenames[slice_desc.m_source_file_index].c_str(), out_basename);
								
				string_remove_extension(out_basename);
				out_basename = "basis_debug_" + out_basename + string_format("_slice_%u", slice_index);

				// Write "best" ETC1S debug images
				{
					gpu_image best_etc1s_gpu_image(m_best_etc1s_images[slice_index]);
					best_etc1s_gpu_image.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_best_etc1s.ktx").c_str(), best_etc1s_gpu_image);

					image best_etc1s_unpacked;
					best_etc1s_gpu_image.unpack(best_etc1s_unpacked);
					save_png(out_basename + "_best_etc1s.png", best_etc1s_unpacked);
				}

				// Write decoded ETC1S debug images
				{
					gpu_image decoded_etc1s(m_decoded_output_textures[slice_index]);
					decoded_etc1s.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_decoded_etc1s.ktx").c_str(), decoded_etc1s);

					image temp(m_decoded_output_textures_unpacked[slice_index]);
					temp.crop(slice_desc.m_orig_width, slice_desc.m_orig_height);
					save_png(out_basename + "_decoded_etc1s.png", temp);
				}
				
				// Write decoded BC1 debug images
				{
					gpu_image decoded_bc1(m_decoded_output_textures_bc1[slice_index]);
					decoded_bc1.override_dimensions(slice_desc.m_orig_width, slice_desc.m_orig_height);
					write_compressed_texture_file((out_basename + "_decoded_bc1.ktx").c_str(), decoded_bc1);

					image temp(m_decoded_output_textures_unpacked_bc1[slice_index]);
					temp.crop(slice_desc.m_orig_width, slice_desc.m_orig_height);
					save_png(out_basename + "_decoded_bc1.png", temp);
				}
			}
		}
				
		return true;
	}

} // namespace basisu
