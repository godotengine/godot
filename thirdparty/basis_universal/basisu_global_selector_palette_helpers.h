// File: basisu_global_selector_palette_helpers.h
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

#include "transcoder/basisu.h"
#include "basisu_etc.h"
#include "transcoder/basisu_global_selector_palette.h"

namespace basisu
{
	const uint32_t cPixelBlockWidth = 4;
	const uint32_t cPixelBlockHeight = 4;
	const uint32_t cPixelBlockTotalPixels = cPixelBlockWidth * cPixelBlockHeight;

	struct pixel_block
	{
		color_rgba m_pixels[cPixelBlockHeight][cPixelBlockWidth]; // [y][x]

		const color_rgba &operator() (uint32_t x, uint32_t y) const { assert((x < cPixelBlockWidth) && (y < cPixelBlockHeight)); return m_pixels[y][x]; }
		color_rgba &operator() (uint32_t x, uint32_t y) { assert((x < cPixelBlockWidth) && (y < cPixelBlockHeight)); return m_pixels[y][x]; }

		const color_rgba *get_ptr() const { return &m_pixels[0][0]; }
		color_rgba *get_ptr() { return &m_pixels[0][0]; }

		void clear() { clear_obj(*this); }
	};
	typedef std::vector<pixel_block> pixel_block_vec;

	uint64_t etc1_global_selector_codebook_find_best_entry(const basist::etc1_global_selector_codebook &codebook,
		uint32_t num_src_pixel_blocks, const pixel_block *pSrc_pixel_blocks, const etc_block *pBlock_endpoints,
		uint32_t &palette_index, basist::etc1_global_palette_entry_modifier &palette_modifier,
		bool perceptual, uint32_t max_pal_entries, uint32_t max_modifiers);

} // namespace basisu
