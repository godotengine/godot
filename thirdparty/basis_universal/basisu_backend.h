// basisu_backend.h
// Copyright (C) 2019-2020 Binomial LLC. All Rights Reserved.
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

#include "transcoder/basisu.h"
#include "basisu_enc.h"
#include "transcoder/basisu_transcoder_internal.h"
#include "transcoder/basisu_global_selector_palette.h"
#include "basisu_frontend.h"

namespace basisu
{
	struct encoder_block
	{
		encoder_block()
		{
			clear();
		}
				
		uint32_t m_endpoint_predictor; 

		int m_endpoint_index;
		int m_selector_index;

		int m_selector_history_buf_index;

		bool m_is_cr_target;
		void clear()
		{
			m_endpoint_predictor = 0;
			
			m_endpoint_index = 0;
			m_selector_index = 0;
						
			m_selector_history_buf_index = 0;
			m_is_cr_target = false;
		}
	};

	typedef std::vector<encoder_block> encoder_block_vec;
	typedef vector2D<encoder_block> encoder_block_vec2D;

	struct etc1_endpoint_palette_entry
	{
		etc1_endpoint_palette_entry()
		{
			clear();
		}

		color_rgba m_color5;
		uint32_t m_inten5;
		bool m_color5_valid;
				
		void clear()
		{
			clear_obj(*this);
		}
	};

	typedef std::vector<etc1_endpoint_palette_entry> etc1_endpoint_palette_entry_vec;

	struct basisu_backend_params
	{
		bool m_etc1s;
		bool m_debug, m_debug_images;
		float m_endpoint_rdo_quality_thresh;
		float m_selector_rdo_quality_thresh;
		uint32_t m_compression_level;

		bool m_use_global_sel_codebook;
		uint32_t m_global_sel_codebook_pal_bits;
		uint32_t m_global_sel_codebook_mod_bits;
		bool m_use_hybrid_sel_codebooks;

		basisu_backend_params()
		{
			clear();
		}

		void clear()
		{
			m_etc1s = false;
			m_debug = false;
			m_debug_images = false;
			m_endpoint_rdo_quality_thresh = 0.0f;
			m_selector_rdo_quality_thresh = 0.0f;
			m_compression_level = 0;

			m_use_global_sel_codebook = false;
			m_global_sel_codebook_pal_bits = ETC1_GLOBAL_SELECTOR_CODEBOOK_MAX_PAL_BITS;
			m_global_sel_codebook_mod_bits = basist::etc1_global_palette_entry_modifier::cTotalBits;
			m_use_hybrid_sel_codebooks = false;
		}
	};

	struct basisu_backend_slice_desc
	{
		basisu_backend_slice_desc()
		{
			clear();
		}
		void clear()
		{
			clear_obj(*this);
		}
		uint32_t m_first_block_index;

		uint32_t m_orig_width;
		uint32_t m_orig_height;

		uint32_t m_width;
		uint32_t m_height;

		uint32_t m_num_blocks_x;
		uint32_t m_num_blocks_y;

		uint32_t m_num_macroblocks_x;
		uint32_t m_num_macroblocks_y;

		uint32_t m_source_file_index;		// also the basis image index
		uint32_t m_mip_index;
		bool m_alpha;
		bool m_iframe;
	};

	typedef std::vector<basisu_backend_slice_desc> basisu_backend_slice_desc_vec;

	struct basisu_backend_output
	{
		basist::basis_tex_format m_tex_format;

		bool m_etc1s;

		uint32_t m_num_endpoints;
		uint32_t m_num_selectors;

		uint8_vec m_endpoint_palette;
		uint8_vec m_selector_palette;

		basisu_backend_slice_desc_vec m_slice_desc;

		uint8_vec m_slice_image_tables;
		std::vector<uint8_vec> m_slice_image_data;
		uint16_vec m_slice_image_crcs;

		basisu_backend_output()
		{
			clear();
		}

		void clear()
		{
			m_tex_format = basist::basis_tex_format::cETC1S;
			m_etc1s = false;

			m_num_endpoints = 0;
			m_num_selectors = 0;

			m_endpoint_palette.clear();
			m_selector_palette.clear();
			m_slice_desc.clear();
			m_slice_image_tables.clear();
			m_slice_image_data.clear();
			m_slice_image_crcs.clear();
		}

		uint32_t get_output_size_estimate() const
		{
			uint32_t total_compressed_bytes = (uint32_t)(m_slice_image_tables.size() + m_endpoint_palette.size() + m_selector_palette.size());
			for (uint32_t i = 0; i < m_slice_image_data.size(); i++)
				total_compressed_bytes += (uint32_t)m_slice_image_data[i].size();

			return total_compressed_bytes;
		}
	};

	class basisu_backend
	{
		BASISU_NO_EQUALS_OR_COPY_CONSTRUCT(basisu_backend);

	public:

		basisu_backend();

		void clear();

		void init(basisu_frontend *pFront_end, basisu_backend_params &params, const basisu_backend_slice_desc_vec &slice_desc, const basist::etc1_global_selector_codebook *pGlobal_sel_codebook);

		uint32_t encode();

		const basisu_backend_output &get_output() const { return m_output; }

	private:
		basisu_frontend *m_pFront_end;
		basisu_backend_params m_params;
		basisu_backend_slice_desc_vec m_slices;
		basisu_backend_output m_output;
		const basist::etc1_global_selector_codebook *m_pGlobal_sel_codebook;

		etc1_endpoint_palette_entry_vec m_endpoint_palette;
		basist::etc1_selector_palette_entry_vec m_selector_palette;

		struct etc1_global_selector_cb_entry_desc
		{
			uint32_t m_pal_index;
			uint32_t m_mod_index;
			bool m_was_used;
		};

		typedef std::vector<etc1_global_selector_cb_entry_desc> etc1_global_selector_cb_entry_desc_vec;

		etc1_global_selector_cb_entry_desc_vec m_global_selector_palette_desc;

		std::vector<encoder_block_vec2D> m_slice_encoder_blocks;

		// Maps OLD to NEW endpoint/selector indices
		uint_vec m_endpoint_remap_table_old_to_new;
		uint_vec m_endpoint_remap_table_new_to_old;

		uint_vec m_selector_remap_table_old_to_new;

		// Maps NEW to OLD endpoint/selector indices
		uint_vec m_selector_remap_table_new_to_old;

		uint32_t get_total_slices() const
		{
			return (uint32_t)m_slices.size();
		}

		uint32_t get_total_slice_blocks() const
		{
			return m_pFront_end->get_total_output_blocks();
		}

		uint32_t get_block_index(uint32_t slice_index, uint32_t block_x, uint32_t block_y) const
		{
			const basisu_backend_slice_desc &slice = m_slices[slice_index];

			assert((block_x < slice.m_num_blocks_x) && (block_y < slice.m_num_blocks_y));

			return slice.m_first_block_index + block_y * slice.m_num_blocks_x + block_x;
		}
				
		uint32_t get_total_blocks(uint32_t slice_index) const
		{
			return m_slices[slice_index].m_num_blocks_x * m_slices[slice_index].m_num_blocks_y;
		}
								
		uint32_t get_total_blocks() const
		{
			uint32_t total_blocks = 0;
			for (uint32_t i = 0; i < m_slices.size(); i++)
				total_blocks += get_total_blocks(i);
			return total_blocks;
		}

		// Returns the total number of input texels, not counting padding up to blocks/macroblocks.
		uint32_t get_total_input_texels(uint32_t slice_index) const
		{
			return m_slices[slice_index].m_orig_width * m_slices[slice_index].m_orig_height;
		}

		uint32_t get_total_input_texels() const
		{
			uint32_t total_texels = 0;
			for (uint32_t i = 0; i < m_slices.size(); i++)
				total_texels += get_total_input_texels(i);
			return total_texels;
		}

		int find_slice(uint32_t block_index, uint32_t *pBlock_x, uint32_t *pBlock_y) const
		{
			for (uint32_t i = 0; i < m_slices.size(); i++)
			{
				if ((block_index >= m_slices[i].m_first_block_index) && (block_index < (m_slices[i].m_first_block_index + m_slices[i].m_num_blocks_x * m_slices[i].m_num_blocks_y)))
				{
					const uint32_t ofs = block_index - m_slices[i].m_first_block_index;
					const uint32_t x = ofs % m_slices[i].m_num_blocks_x;
					const uint32_t y = ofs / m_slices[i].m_num_blocks_x;

					if (pBlock_x) *pBlock_x = x;
					if (pBlock_y) *pBlock_y = y;

					return i;
				}
			}
			return -1;
		}

		void create_endpoint_palette();

		void create_selector_palette();

		// endpoint palette
		//   5:5:5 and predicted 4:4:4 colors, 1 or 2 3-bit intensity table indices
		// selector palette
		//   4x4 2-bit selectors

		// per-macroblock:
		//  4 diff bits
		//  4 flip bits
		//  Endpoint template index, 1-8 endpoint indices
		//      Alternately, if no template applies, we can send 4 ETC1S bits followed by 4-8 endpoint indices
		//  4 selector indices

		void reoptimize_and_sort_endpoints_codebook(uint32_t total_block_endpoints_remapped, uint_vec &all_endpoint_indices);
		void sort_selector_codebook();
		void create_encoder_blocks();
		void compute_slice_crcs();
		bool encode_image();
		bool encode_endpoint_palette();
		bool encode_selector_palette();
		int find_video_frame(int slice_index, int delta);
		void check_for_valid_cr_blocks();
	};

} // namespace basisu

