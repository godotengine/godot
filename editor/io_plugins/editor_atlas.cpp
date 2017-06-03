/*************************************************************************/
/*  editor_atlas.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

	//super simple, almost brute force scanline stacking fitter
	//it's pretty basic for now, but it tries to make sure that the aspect ratio of the
	//resulting atlas is somehow square. This is necessary because video cards have limits
	//on texture size (usually 2048 or 4096), so the more square a texture, the more chances
	//it will work in every hardware.
	// for example, it will prioritize a 1024x1024 atlas (works everywhere) instead of a
	// 256x8192 atlas (won't work anywhere).

	ERR_FAIL_COND(p_rects.size() == 0);

	Vector<_EditorAtlasWorkRect> wrects;
	wrects.resize(p_rects.size());
	for (int i = 0; i < p_rects.size(); i++) {
		wrects[i].s = p_rects[i];
		wrects[i].idx = i;
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

		Vector<int> hmax;
		hmax.resize(w);
		for (int j = 0; j < w; j++)
			hmax[j] = 0;

		//place them
		int ofs = 0;

		for (int j = 0; j < wrects.size(); j++) {

			if (ofs + wrects[j].s.width > w) {

				ofs = 0;
			}

			int from_y = 0;
			for (int k = 0; k < wrects[j].s.width; k++) {

				if (hmax[ofs + k] > from_y)
					from_y = hmax[ofs + k];
			}

			wrects[j].p.x = ofs;
			wrects[j].p.y = from_y;

			int end_h = from_y + wrects[j].s.height;
			int end_w = ofs + wrects[j].s.width;

			for (int k = 0; k < wrects[j].s.width; k++) {

				hmax[ofs + k] = end_h;
			}

			if (end_h > max_h)
				max_h = end_h;

			if (end_w > max_w)
				max_w = end_w;

			ofs += wrects[j].s.width;
		}

		_EditorAtlasWorkRectResult result;
		result.result = wrects;
		result.max_h = max_h;
		result.max_w = max_w;
		results.push_back(result);
	}

	//find the result with the best aspect ratio

	int best = -1;
	float best_aspect = 1e20;

	for (int i = 0; i < results.size(); i++) {

		float h = results[i].max_h;
		float w = results[i].max_w;
		float aspect = h > w ? h / w : w / h;
		if (aspect < best_aspect) {
			best = i;
			best_aspect = aspect;
		}
	}

	r_result.resize(p_rects.size());

	for (int i = 0; i < p_rects.size(); i++) {

		r_result[results[best].result[i].idx] = results[best].result[i].p;
	}

	r_size = Size2(results[best].max_w, results[best].max_h);
}
