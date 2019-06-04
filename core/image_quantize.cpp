/*************************************************************************/
/*  image_quantize.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "image.h"
#include "print_string.h"
#include <stdio.h>
#ifdef TOOLS_ENABLED
#include "os/os.h"
#include "set.h"
#include "sort.h"

//#define QUANTIZE_SPEED_OVER_QUALITY

Image::MCBlock::MCBlock() {
}

Image::MCBlock::MCBlock(BColorPos *p_colors, int p_color_count) {

	colors = p_colors;
	color_count = p_color_count;
	min_color.color = BColor(255, 255, 255, 255);
	max_color.color = BColor(0, 0, 0, 0);
	shrink();
}

int Image::MCBlock::get_longest_axis_index() const {

	int max_dist = -1;
	int max_index = 0;

	for (int i = 0; i < 4; i++) {

		int d = max_color.color.col[i] - min_color.color.col[i];
		if (d > max_dist) {
			max_index = i;
			max_dist = d;
		}
	}

	return max_index;
}
int Image::MCBlock::get_longest_axis_length() const {

	int max_dist = -1;

	for (int i = 0; i < 4; i++) {

		int d = max_color.color.col[i] - min_color.color.col[i];
		if (d > max_dist) {
			max_dist = d;
		}
	}

	return max_dist;
}

bool Image::MCBlock::operator<(const MCBlock &p_block) const {

	int alen = get_longest_axis_length();
	int blen = p_block.get_longest_axis_length();
	if (alen == blen) {

		return colors < p_block.colors;
	} else
		return alen < blen;
}

void Image::MCBlock::shrink() {

	min_color = colors[0];
	max_color = colors[0];

	for (int i = 1; i < color_count; i++) {

		for (int j = 0; j < 4; j++) {

			min_color.color.col[j] = MIN(min_color.color.col[j], colors[i].color.col[j]);
			max_color.color.col[j] = MAX(max_color.color.col[j], colors[i].color.col[j]);
		}
	}
}

void Image::quantize() {

	bool has_alpha = detect_alpha() != ALPHA_NONE;

	bool quantize_fast = OS::get_singleton()->has_environment("QUANTIZE_FAST");

	convert(FORMAT_RGBA);

	ERR_FAIL_COND(format != FORMAT_RGBA);

	DVector<uint8_t> indexed_data;

	{
		int color_count = data.size() / 4;

		ERR_FAIL_COND(color_count == 0);

		Set<MCBlock> block_queue;

		DVector<BColorPos> data_colors;
		data_colors.resize(color_count);

		DVector<BColorPos>::Write dcw = data_colors.write();

		DVector<uint8_t>::Read dr = data.read();
		const BColor *drptr = (const BColor *)&dr[0];
		BColorPos *bcptr = &dcw[0];

		{
			for (int i = 0; i < color_count; i++) {

				//uint32_t data_ofs=i<<2;
				bcptr[i].color = drptr[i]; //BColor(drptr[data_ofs+0],drptr[data_ofs+1],drptr[data_ofs+2],drptr[data_ofs+3]);
				bcptr[i].index = i;
			}
		}

		//printf("color count: %i\n",color_count);
		/*
		for(int i=0;i<color_count;i++) {

			BColor bc = ((BColor*)&wb[0])[i];
			printf("%i - %i,%i,%i,%i\n",i,bc.r,bc.g,bc.b,bc.a);
		}*/

		MCBlock initial_block((BColorPos *)&dcw[0], color_count);

		block_queue.insert(initial_block);

		while (block_queue.size() < 256 && block_queue.back()->get().color_count > 1) {

			MCBlock longest = block_queue.back()->get();
			//printf("longest: %i (%i)\n",longest.get_longest_axis_index(),longest.get_longest_axis_length());

			block_queue.erase(block_queue.back());

			BColorPos *first = longest.colors;
			BColorPos *median = longest.colors + (longest.color_count + 1) / 2;
			BColorPos *end = longest.colors + longest.color_count;

#if 0
			int lai =longest.get_longest_axis_index();
			switch(lai) {
#if 0
				case 0: { SortArray<BColorPos,BColorPos::SortR> sort; sort.sort(first,end-first); } break;
				case 1: { SortArray<BColorPos,BColorPos::SortG> sort; sort.sort(first,end-first); } break;
				case 2: { SortArray<BColorPos,BColorPos::SortB> sort; sort.sort(first,end-first); } break;
				case 3: { SortArray<BColorPos,BColorPos::SortA> sort; sort.sort(first,end-first); } break;
#else
				case 0: { SortArray<BColorPos,BColorPos::SortR> sort; sort.nth_element(0,end-first,median-first,first); } break;
				case 1: { SortArray<BColorPos,BColorPos::SortG> sort; sort.nth_element(0,end-first,median-first,first); } break;
				case 2: { SortArray<BColorPos,BColorPos::SortB> sort; sort.nth_element(0,end-first,median-first,first); } break;
				case 3: { SortArray<BColorPos,BColorPos::SortA> sort; sort.nth_element(0,end-first,median-first,first); } break;
#endif

			}

			//avoid same color from being split in 2
			//search forward and flip
			BColorPos *median_end=median;
			BColorPos *p=median_end+1;

			while(p!=end) {
				if (median_end->color==p->color) {
					SWAP(*(median_end+1),*p);
					median_end++;
				}
				p++;
			}

			//search backward and flip
			BColorPos *median_begin=median;
			p=median_begin-1;

			while(p!=(first-1)) {
				if (median_begin->color==p->color) {
					SWAP(*(median_begin-1),*p);
					median_begin--;
				}
				p--;
			}


			if (first < median_begin) {
				median=median_begin;
			} else if (median_end < end-1) {
				median=median_end+1;
			} else {
				break; //shouldn't have arrived here, since it means all pixels are equal, but wathever
			}

			MCBlock left(first,median-first);
			MCBlock right(median,end-median);

			block_queue.insert(left);
			block_queue.insert(right);

#else
			switch (longest.get_longest_axis_index()) {
				case 0: {
					SortArray<BColorPos, BColorPos::SortR> sort;
					sort.nth_element(0, end - first, median - first, first);
				} break;
				case 1: {
					SortArray<BColorPos, BColorPos::SortG> sort;
					sort.nth_element(0, end - first, median - first, first);
				} break;
				case 2: {
					SortArray<BColorPos, BColorPos::SortB> sort;
					sort.nth_element(0, end - first, median - first, first);
				} break;
				case 3: {
					SortArray<BColorPos, BColorPos::SortA> sort;
					sort.nth_element(0, end - first, median - first, first);
				} break;
			}

			MCBlock left(first, median - first);
			MCBlock right(median, end - median);

			block_queue.insert(left);
			block_queue.insert(right);

#endif
		}

		while (block_queue.size() > 256) {

			block_queue.erase(block_queue.front()); // erase least significant
		}

		int res_colors = 0;

		int comp_size = (has_alpha ? 4 : 3);
		indexed_data.resize(color_count + 256 * comp_size);

		DVector<uint8_t>::Write iw = indexed_data.write();
		uint8_t *iwptr = &iw[0];
		BColor pallete[256];

		//	print_line("applying quantization - res colors "+itos(block_queue.size()));

		while (block_queue.size()) {

			const MCBlock &b = block_queue.back()->get();

			uint64_t sum[4] = { 0, 0, 0, 0 };

			for (int i = 0; i < b.color_count; i++) {

				sum[0] += b.colors[i].color.col[0];
				sum[1] += b.colors[i].color.col[1];
				sum[2] += b.colors[i].color.col[2];
				sum[3] += b.colors[i].color.col[3];
			}

			BColor c(sum[0] / b.color_count, sum[1] / b.color_count, sum[2] / b.color_count, sum[3] / b.color_count);

			//printf(" %i: %i,%i,%i,%i out of %i\n",res_colors,c.r,c.g,c.b,c.a,b.color_count);

			for (int i = 0; i < comp_size; i++) {
				iwptr[color_count + res_colors * comp_size + i] = c.col[i];
			}

			if (quantize_fast) {
				for (int i = 0; i < b.color_count; i++) {
					iwptr[b.colors[i].index] = res_colors;
				}
			} else {

				pallete[res_colors] = c;
			}

			res_colors++;

			block_queue.erase(block_queue.back());
		}

		if (!quantize_fast) {

			for (int i = 0; i < color_count; i++) {

				const BColor &c = drptr[i];
				uint8_t best_dist_idx = 0;
				uint32_t dist = 0xFFFFFFFF;

				for (int j = 0; j < res_colors; j++) {

					const BColor &pc = pallete[j];
					uint32_t d = 0;
					{
						int16_t v = (int16_t)c.r - (int16_t)pc.r;
						d += v * v;
					}
					{
						int16_t v = (int16_t)c.g - (int16_t)pc.g;
						d += v * v;
					}
					{
						int16_t v = (int16_t)c.b - (int16_t)pc.b;
						d += v * v;
					}
					{
						int16_t v = (int16_t)c.a - (int16_t)pc.a;
						d += v * v;
					}

					if (d <= dist) {
						best_dist_idx = j;
						dist = d;
					}
				}

				iwptr[i] = best_dist_idx;
			}
		}

		//iw = DVector<uint8_t>::Write();
		//dr = DVector<uint8_t>::Read();
		//wb = DVector<uint8_t>::Write();
	}

	print_line(itos(indexed_data.size()));
	data = indexed_data;
	format = has_alpha ? FORMAT_INDEXED_ALPHA : FORMAT_INDEXED;

} //do none

#else

void Image::quantize() {} //do none

#endif
