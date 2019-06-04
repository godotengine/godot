/*************************************************************************/
/*  editor_atlas.cpp                                                     */
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
#include "editor_atlas.h"
#include "print_string.h"
#include <cfloat>

struct _EditorAtlasWorkRect {

	Size2i s;
	Point2i p;
	int idx;
	_FORCE_INLINE_ bool operator<(const _EditorAtlasWorkRect &p_r) const { return s.width > p_r.s.width; };
};

struct _EditorAtlasWorkRectResult {

	Vector<_EditorAtlasWorkRect> result;
	int max_w;
	int max_h;
};

void EditorAtlas::fit(const Vector<Size2i> &p_rects, Vector<Point2i> &r_result, Size2i &r_size) {

	ERR_FAIL_COND(p_rects.size() == 0);

	Vector<_EditorAtlasWorkRect> wrects;
	wrects.resize(p_rects.size());
	long total_area = 0;
	for (int i = 0; i < p_rects.size(); i++) {
		wrects[i].s = p_rects[i];
		wrects[i].idx = i;
		total_area += p_rects[i].width * p_rects[i].height;
	}
	wrects.sort();
	int widest = wrects[0].s.width;

	Vector<_EditorAtlasWorkRectResult> results;

	for (int i = 0; i <= 12; i++) {

		int w = 1 << i;
		int max_h = 0;
		int max_w = 0;
		if (w < widest)
			continue;

		Vector<int> wmax;
		wmax.resize(total_area / w);
		for (int j = 0; j < wmax.size(); j++)
			wmax[j] = 0;

		for (int j = 0; j < wrects.size(); j++) {

			int new_x = 0;
			int new_y = 0;

			int piece_w = wrects[j].s.width;
			int piece_h = wrects[j].s.height;

			bool found_place;

			do {
				found_place = true;
				new_x = 0;
				if (wmax.size() <= new_y + piece_h) {
					int prevS = wmax.size();
					wmax.resize(new_y + piece_h + 128);
					for (int k = prevS; k < wmax.size(); k++)
						wmax[k] = 0;
				}
				for (int k = 0; k < piece_h; k++) {
					if (new_x < wmax[new_y + k]) new_x = wmax[new_y + k];
					if (new_x + piece_w > w) {
						new_y += k + 1;
						found_place = false;
						break;
					}
				}
				if (found_place) {
					// one more check is calculating lost space of atlas
					long lost_area = 0;
					for (int k = 0; k < piece_h; k++) {
						lost_area += new_x - wmax[new_y + k];
					}
					if (lost_area >= piece_w * piece_h / 2) {
						found_place = false;
						new_y++;
					}
				}
			} while (!found_place);

			wrects[j].p.x = new_x;
			wrects[j].p.y = new_y;

			int end_h = new_y + piece_h;
			int end_w = new_x + piece_w;

			for (int k = 0; k < piece_h; k++) {
				wmax[new_y + k] = end_w;
			}

			if (end_h > max_h)
				max_h = end_h;

			if (end_w > max_w)
				max_w = end_w;
		}

		_EditorAtlasWorkRectResult result;
		result.result = wrects;
		result.max_h = max_h;
		result.max_w = max_w;
		results.push_back(result);
		float perimeter = (next_power_of_2(max_w) + max_h) * 2.f;
		print_line("Processing atlas: width " + itos(w) + " ,height " + itos(max_h) + " ,perimeter " + rtos(perimeter));
	}

	//find the result with the most efficiency

	int best = -1;
	float min_perimeter = FLT_MAX;

	for (int i = 0; i < results.size(); i++) {

		float h = results[i].max_h;
		float w = results[i].max_w;
		float perimeter = (next_power_of_2(w) + h) * 2.f;
		if (perimeter < min_perimeter) {
			best = i;
			min_perimeter = perimeter;
		}
	}

	if (best < 0) {
		ERR_PRINT("Atlas processing failed!");
		return;
	}

	r_result.resize(p_rects.size());

	for (int i = 0; i < p_rects.size(); i++) {

		r_result[results[best].result[i].idx] = results[best].result[i].p;
	}

	r_size = Size2(results[best].max_w, results[best].max_h);
}
