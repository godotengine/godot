// basiu_global_selector_palette_helpers.cpp
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
#include "basisu_global_selector_palette_helpers.h"

namespace basisu
{
	uint64_t etc1_global_selector_codebook_find_best_entry(const basist::etc1_global_selector_codebook &codebook,
		uint32_t num_src_pixel_blocks, const pixel_block *pSrc_pixel_blocks, const etc_block *pBlock_endpoints,
		uint32_t &palette_index, basist::etc1_global_palette_entry_modifier &palette_modifier,
		bool perceptual, uint32_t max_pal_entries, uint32_t max_modifiers)
	{
		uint64_t best_err = UINT64_MAX;
		uint32_t best_pal_index = 0;
		basist::etc1_global_palette_entry_modifier best_pal_modifier;

		if (!max_pal_entries)
			max_pal_entries = codebook.size();

		if (!max_modifiers)
			max_modifiers = basist::etc1_global_palette_entry_modifier::cTotalValues;

		for (uint32_t pal_index = 0; pal_index < max_pal_entries; pal_index++)
		{
			for (uint32_t mod_index = 0; mod_index < max_modifiers; mod_index++)
			{
				const basist::etc1_global_palette_entry_modifier pal_modifier(mod_index);

				const basist::etc1_selector_palette_entry pal_entry(codebook.get_entry(pal_index, pal_modifier));

				uint64_t trial_err = 0;
				for (uint32_t block_index = 0; block_index < num_src_pixel_blocks; block_index++)
				{
					etc_block trial_block(pBlock_endpoints[block_index]);

					for (uint32_t y = 0; y < 4; y++)
						for (uint32_t x = 0; x < 4; x++)
							trial_block.set_selector(x, y, pal_entry(x, y));

					trial_err += trial_block.evaluate_etc1_error(reinterpret_cast<const basisu::color_rgba *>(pSrc_pixel_blocks[block_index].get_ptr()), perceptual);
					if (trial_err >= best_err)
						break;
				}

				if (trial_err < best_err)
				{
					best_err = trial_err;
					best_pal_index = pal_index;
					best_pal_modifier = pal_modifier;
				}
			} // mod_index
		} // pal_index

		palette_index = best_pal_index;
		palette_modifier = best_pal_modifier;

		return best_err;
	}

} // namespace basisu
