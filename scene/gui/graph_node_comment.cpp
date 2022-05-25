/*************************************************************************/
/*  graph_node_comment.cpp                                               */
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

#include "graph_node_comment.h"

#include "core/string/translation.h"

struct _MinSizeCache {
	int min_size;
	bool will_stretch;
	int final_size;
};

GraphNodeComment::GraphNodeComment() {
}

void GraphNodeComment::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<StyleBox> sb_frame = get_theme_stylebox(selected ? SNAME("selected_frame") : SNAME("frame"));
			Ref<StyleBoxFlat> sb_frame_flat = sb_frame;
			Ref<StyleBoxTexture> sb_frame_texture = sb_frame;

			if (tint_color_enabled) {
				if (sb_frame_flat.is_valid()) {
					sb_frame_flat = sb_frame_flat->duplicate(); //TODO: @Geometror Cache this?
					sb_frame_flat->set_bg_color(tint_color);
					sb_frame_flat->set_border_color(tint_color.darkened(0.2));
					draw_style_box(sb_frame_flat, Rect2(Point2(), get_size()));
				} else if (sb_frame_texture.is_valid()) {
					sb_frame_texture = sb_frame_flat->duplicate(); //TODO: @Geometror Cache this?
					sb_frame_texture->set_modulate(tint_color);
					draw_style_box(sb_frame_texture, Rect2(Point2(), get_size()));
				}
			} else {
				draw_style_box(sb_frame_flat, Rect2(Point2(), get_size()));
			}

			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			Color resizer_color = get_theme_color(SNAME("resizer_color"));
			int title_offset = get_theme_constant(SNAME("title_v_offset"));
			int title_h_offset = get_theme_constant(SNAME("title_h_offset"));
			Color title_color = get_theme_color(SNAME("title_color"));

			title_buf->draw(get_canvas_item(), Point2(sb_frame->get_margin(SIDE_LEFT) + title_h_offset, -title_buf->get_size().y + title_offset), title_color);

			if (resizable) {
				draw_texture(resizer, get_size() - resizer->get_size(), resizer_color);
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;
	}
}

void GraphNodeComment::_resort() {
	// First pass, determine minimum size AND amount of stretchable elements.

	Size2i new_size = get_size();
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> sb_slot = get_theme_stylebox(SNAME("slot"));

	int separation = get_theme_constant(SNAME("separation"));

	bool first = true;
	int children_count = 0;
	int stretch_min = 0;
	int stretch_avail = 0;
	float stretch_ratio_total = 0;
	HashMap<Control *, _MinSizeCache> min_size_cache;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible_in_tree()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_combined_minimum_size();
		_MinSizeCache msc;

		stretch_min += size.height;
		msc.min_size = size.height;
		msc.will_stretch = c->get_v_size_flags() & SIZE_EXPAND;

		if (msc.will_stretch) {
			stretch_avail += msc.min_size;
			stretch_ratio_total += c->get_stretch_ratio();
		}
		msc.final_size = msc.min_size;
		min_size_cache[c] = msc;
		children_count++;
	}

	if (children_count == 0) {
		return;
	}

	int stretch_max = new_size.height - (children_count - 1) * separation;
	int stretch_diff = stretch_max - stretch_min;
	if (stretch_diff < 0) {
		// Avoid negative stretch space.
		stretch_diff = 0;
	}

	stretch_avail += stretch_diff - sb_frame->get_margin(SIDE_BOTTOM) - sb_frame->get_margin(SIDE_TOP); //available stretch space.
	// Second, pass successively to discard elements that can't be stretched, this will run
	// while stretchable elements exist.

	while (stretch_ratio_total > 0) {
		// First of all, don't even be here if no stretchable objects exist.
		bool refit_successful = true;

		for (int i = 0; i < get_child_count(false); i++) {
			Control *c = Object::cast_to<Control>(get_child(i, false));
			if (!c || !c->is_visible_in_tree()) {
				continue;
			}
			if (c->is_set_as_top_level()) {
				continue;
			}

			ERR_FAIL_COND(!min_size_cache.has(c));
			_MinSizeCache &msc = min_size_cache[c];

			if (msc.will_stretch) {
				int final_pixel_size = stretch_avail * c->get_stretch_ratio() / stretch_ratio_total;
				if (final_pixel_size < msc.min_size) {
					// If the available stretching area is too small for a Control,
					// then remove it from stretching area.
					msc.will_stretch = false;
					stretch_ratio_total -= c->get_stretch_ratio();
					refit_successful = false;
					stretch_avail -= msc.min_size;
					msc.final_size = msc.min_size;
					break;
				} else {
					msc.final_size = final_pixel_size;
				}
			}
		}

		if (refit_successful) {
			break;
		}
	}

	// Final pass, draw and stretch elements.

	int ofs = sb_frame->get_margin(SIDE_TOP);

	first = true;
	int idx = 0;
	cache_y.clear();
	int width = new_size.width - sb_frame->get_minimum_size().x;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible_in_tree()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		_MinSizeCache &msc = min_size_cache[c];

		if (first) {
			first = false;
		} else {
			ofs += separation;
		}

		int from = ofs;
		int to = ofs + msc.final_size;

		if (msc.will_stretch && idx == children_count - 1) {
			// Adjust so the last one always fits perfect.
			// Compensating for numerical imprecision.

			to = new_size.height - sb_frame->get_margin(SIDE_BOTTOM);
		}

		int size = to - from;

		float margin = sb_frame->get_margin(SIDE_LEFT);
		Rect2 rect(margin, from, width, size);

		fit_child_in_rect(c, rect);
		cache_y.push_back(from - sb_frame->get_margin(SIDE_TOP) + size * 0.5);

		ofs = to;
		idx++;
	}

	update();
}

void GraphNodeComment::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tint_color_enabled", "p_enable"), &GraphNodeComment::set_tint_color_enabled);
	ClassDB::bind_method(D_METHOD("is_tint_color_enabled"), &GraphNodeComment::is_tint_color_enabled);

	ClassDB::bind_method(D_METHOD("set_tint_color", "p_color"), &GraphNodeComment::set_tint_color);
	ClassDB::bind_method(D_METHOD("get_tint_color"), &GraphNodeComment::get_tint_color);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tint_color_enabled"), "set_tint_color_enabled", "is_tint_color_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "tint_color"), "set_tint_color", "get_tint_color");
}

void GraphNodeComment::set_tint_color_enabled(bool p_enable) {
	tint_color_enabled = p_enable;
	update();
}

bool GraphNodeComment::is_tint_color_enabled() const {
	return tint_color_enabled;
}

void GraphNodeComment::set_tint_color(const Color &p_color) {
	tint_color = p_color;
	update();
}

Color GraphNodeComment::get_tint_color() const {
	return tint_color;
}

bool GraphNodeComment::has_point(const Point2 &p_point) const {
	Ref<StyleBox> frame = get_theme_stylebox(SNAME("frame"));
	Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

	if (Rect2(get_size() - resizer->get_size(), resizer->get_size()).has_point(p_point)) {
		return true;
	}

	if (Rect2(0, 0, get_size().width, frame->get_margin(SIDE_TOP)).has_point(p_point)) {
		return true;
	}

	return false;
}
