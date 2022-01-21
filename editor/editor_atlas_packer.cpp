/*************************************************************************/
/*  editor_atlas_packer.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_atlas_packer.h"

void EditorAtlasPacker::_plot_triangle(Ref<BitMap> p_bitmap, Vector2i *vertices) {
	int width = p_bitmap->get_size().width;
	int height = p_bitmap->get_size().height;
	int x[3];
	int y[3];

	for (int j = 0; j < 3; j++) {
		x[j] = vertices[j].x;
		y[j] = vertices[j].y;
	}

	// sort the points vertically
	if (y[1] > y[2]) {
		SWAP(x[1], x[2]);
		SWAP(y[1], y[2]);
	}
	if (y[0] > y[1]) {
		SWAP(x[0], x[1]);
		SWAP(y[0], y[1]);
	}
	if (y[1] > y[2]) {
		SWAP(x[1], x[2]);
		SWAP(y[1], y[2]);
	}

	double dx_far = double(x[2] - x[0]) / (y[2] - y[0] + 1);
	double dx_upper = double(x[1] - x[0]) / (y[1] - y[0] + 1);
	double dx_low = double(x[2] - x[1]) / (y[2] - y[1] + 1);
	double xf = x[0];
	double xt = x[0] + dx_upper; // if y[0] == y[1], special case
	for (int yi = y[0]; yi <= (y[2] > height - 1 ? height - 1 : y[2]); yi++) {
		if (yi >= 0) {
			for (int xi = (xf > 0 ? int(xf) : 0); xi <= (xt < width ? xt : width - 1); xi++) {
				//pixels[int(x + y * width)] = color;

				p_bitmap->set_bit(Point2(xi, yi), true);
			}

			for (int xi = (xf < width ? int(xf) : width - 1); xi >= (xt > 0 ? xt : 0); xi--) {
				p_bitmap->set_bit(Point2(xi, yi), true);
			}
		}
		xf += dx_far;
		if (yi < y[1]) {
			xt += dx_upper;
		} else {
			xt += dx_low;
		}
	}
}

void EditorAtlasPacker::chart_pack(Vector<Chart> &charts, int &r_width, int &r_height, int p_atlas_max_size, int p_cell_resolution) {
	int divide_by = MIN(64, p_cell_resolution);
	Vector<PlottedBitmap> bitmaps;

	int max_w = 0;

	for (int i = 0; i < charts.size(); i++) {
		const Chart &chart = charts[i];

		//generate aabb

		Rect2i aabb;
		int vertex_count = chart.vertices.size();
		const Vector2 *vertices = chart.vertices.ptr();

		for (int j = 0; j < vertex_count; j++) {
			if (j == 0) {
				aabb.position = vertices[j];
			} else {
				aabb.expand_to(vertices[j]);
			}
		}

		Ref<BitMap> src_bitmap;
		src_bitmap.instantiate();
		src_bitmap->create(aabb.size / divide_by);

		int w = src_bitmap->get_size().width;
		int h = src_bitmap->get_size().height;

		//plot triangles, using divisor

		for (int j = 0; j < chart.faces.size(); j++) {
			Vector2i v[3];
			for (int k = 0; k < 3; k++) {
				Vector2 vtx = chart.vertices[chart.faces[j].vertex[k]];
				vtx -= aabb.position;
				vtx /= divide_by;
				v[k] = vtx;
			}

			_plot_triangle(src_bitmap, v);
		}

		//src_bitmap->convert_to_image()->save_png("bitmap" + itos(i) + ".png");

		//grow by 1 for each side

		int bmw = src_bitmap->get_size().width + 2;
		int bmh = src_bitmap->get_size().height + 2;

		int heights_size = -1;
		bool transpose = false;
		if (chart.can_transpose && bmh > bmw) {
			heights_size = bmh;
			transpose = true;
		} else {
			heights_size = bmw;
		}

		max_w = MAX(max_w, heights_size);

		Vector<int> top_heights;
		Vector<int> bottom_heights;
		top_heights.resize(heights_size);
		bottom_heights.resize(heights_size);

		for (int x = 0; x < heights_size; x++) {
			top_heights.write[x] = -1;
			bottom_heights.write[x] = 0x7FFFFFFF;
		}

		for (int x = 0; x < bmw; x++) {
			for (int y = 0; y < bmh; y++) {
				bool found_pixel = false;
				for (int lx = x - 1; lx < x + 2 && !found_pixel; lx++) {
					for (int ly = y - 1; ly < y + 2 && !found_pixel; ly++) {
						int px = lx - 1;
						if (px < 0 || px >= w) {
							continue;
						}
						int py = ly - 1;
						if (py < 0 || py >= h) {
							continue;
						}

						if (src_bitmap->get_bit(Vector2(px, py))) {
							found_pixel = true;
						}
					}
				}
				if (found_pixel) {
					if (transpose) {
						if (x > top_heights[y]) {
							top_heights.write[y] = x;
						}
						if (x < bottom_heights[y]) {
							bottom_heights.write[y] = x;
						}
					} else {
						if (y > top_heights[x]) {
							top_heights.write[x] = y;
						}
						if (y < bottom_heights[x]) {
							bottom_heights.write[x] = y;
						}
					}
				}
			}
		}

		String row;
		for (int j = 0; j < top_heights.size(); j++) {
			row += "(" + itos(top_heights[j]) + "-" + itos(bottom_heights[j]) + "),";
		}

		PlottedBitmap plotted_bitmap;
		plotted_bitmap.offset = aabb.position;
		plotted_bitmap.top_heights = top_heights;
		plotted_bitmap.bottom_heights = bottom_heights;
		plotted_bitmap.chart_index = i;
		plotted_bitmap.transposed = transpose;
		plotted_bitmap.area = bmw * bmh;

		bitmaps.push_back(plotted_bitmap);
	}

	bitmaps.sort();

	int atlas_max_width = nearest_power_of_2_templated(p_atlas_max_size) / divide_by;
	int atlas_w = nearest_power_of_2_templated(max_w);
	int atlas_h;
	while (true) {
		atlas_h = 0;

		//do a tetris
		Vector<int> heights;
		heights.resize(atlas_w);
		for (int i = 0; i < atlas_w; i++) {
			heights.write[i] = 0;
		}

		int *atlas_ptr = heights.ptrw();

		for (int i = 0; i < bitmaps.size(); i++) {
			int best_height = 0x7FFFFFFF;
			int best_height_offset = -1;
			int w = bitmaps[i].top_heights.size();

			const int *top_heights = bitmaps[i].top_heights.ptr();
			const int *bottom_heights = bitmaps[i].bottom_heights.ptr();

			for (int j = 0; j <= atlas_w - w; j++) {
				int height = 0;

				for (int k = 0; k < w; k++) {
					int pixmap_h = bottom_heights[k];
					if (pixmap_h == 0x7FFFFFFF) {
						continue; //no pixel here, anything is fine
					}

					int h = MAX(0, atlas_ptr[j + k] - pixmap_h);
					if (h > height) {
						height = h;
					}
				}

				if (height < best_height) {
					best_height = height;
					best_height_offset = j;
				}
			}

			for (int j = 0; j < w; j++) { //add
				if (top_heights[j] == -1) { //unused
					continue;
				}
				int height = best_height + top_heights[j] + 1;
				atlas_ptr[j + best_height_offset] = height;
				atlas_h = MAX(atlas_h, height);
			}

			// set
			Vector2 offset = bitmaps[i].offset;
			if (bitmaps[i].transposed) {
				SWAP(offset.x, offset.y);
			}

			Vector2 final_pos = Vector2(best_height_offset * divide_by, best_height * divide_by) + Vector2(divide_by, divide_by) - offset;
			charts.write[bitmaps[i].chart_index].final_offset = final_pos;
			charts.write[bitmaps[i].chart_index].transposed = bitmaps[i].transposed;
		}

		if (atlas_h <= atlas_w * 2 || atlas_w >= atlas_max_width) {
			break; //ok this one is enough
		}

		//try again
		atlas_w *= 2;
	}

	r_width = atlas_w * divide_by;
	r_height = atlas_h * divide_by;
}
