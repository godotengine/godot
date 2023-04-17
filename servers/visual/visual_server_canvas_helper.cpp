/**************************************************************************/
/*  visual_server_canvas_helper.cpp                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "visual_server_canvas_helper.h"

#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

LocalVector<MultiRect> VisualServerCanvasHelper::_tilemap_multirects;
Mutex VisualServerCanvasHelper::_tilemap_mutex;

bool VisualServerCanvasHelper::_multirect_enabled = true;

MultiRect::MultiRect() {
	begin();
}
MultiRect::~MultiRect() {
	end();
}

void MultiRect::begin() {
	DEV_CHECK_ONCE(!rects.size());
	rects.clear();
	sources.clear();

	state.flags = 0;
	state_set = false;
}

void MultiRect::add_rect(RID p_canvas_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, RID p_normal_map, bool p_clip_uv) {
	bool new_common_data = true;

	Rect2 rect = p_rect;
	Rect2 source = p_src_rect;

	// To make the rendering code as efficient as possible,
	// a single MultiRect command should have identical flips and transpose etc.
	// If these change, it flushes the previous multirect and starts a new one.
	uint32_t flags = 0;

	if (p_rect.size.x < 0) {
		flags |= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		rect.size.x = -rect.size.x;
	}
	if (source.size.x < 0) {
		flags ^= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		source.size.x = -source.size.x;
	}
	if (p_rect.size.y < 0) {
		flags |= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		rect.size.y = -rect.size.y;
	}
	if (source.size.y < 0) {
		flags ^= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		source.size.y = -source.size.y;
	}

	if (p_transpose) {
		flags |= RasterizerCanvas::CANVAS_RECT_TRANSPOSE;
		SWAP(rect.size.x, rect.size.y);
	}

	if (p_clip_uv) {
		flags |= RasterizerCanvas::CANVAS_RECT_CLIP_UV;
	}

	VisualServerCanvasHelper::State s;
	s.item = p_canvas_item;
	s.texture = p_texture;
	s.modulate = p_modulate;
	s.normal_map = p_normal_map;
	s.flags = flags;

	if (!is_empty()) {
		if ((state != s) ||
				(rects.size() >= MAX_RECTS)) {
			end();
		} else {
			new_common_data = false;
		}
	}

	if (new_common_data) {
		state = s;
	}

	rects.push_back(rect);
	sources.push_back(source);
}

void MultiRect::begin(const VisualServerCanvasHelper::State &p_state) {
	DEV_CHECK_ONCE(!rects.size());
	rects.clear();
	sources.clear();

	state = p_state;
	state_set = true;
}

uint32_t MultiRect::flags_from_rects(Rect2 &r_rect, Rect2 &r_source) {
	uint32_t flags = 0;

	if (r_rect.size.x < 0) {
		flags |= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		r_rect.size.x = -r_rect.size.x;
	}
	if (r_rect.size.y < 0) {
		flags |= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		r_rect.size.y = -r_rect.size.y;
	}

	if (r_source.size.x < 0) {
		flags ^= RasterizerCanvas::CANVAS_RECT_FLIP_H;
		r_source.size.x = -r_source.size.x;
	}
	if (r_source.size.y < 0) {
		flags ^= RasterizerCanvas::CANVAS_RECT_FLIP_V;
		r_source.size.y = -r_source.size.y;
	}

	return flags;
}

bool MultiRect::add_pre_flipped(const Rect2 &p_rect, const Rect2 &p_src_rect) {
	if (rects.is_full()) {
		return false;
	}
	*rects.request() = p_rect;
	*sources.request() = p_src_rect;
	return true;
}

bool MultiRect::add(const Rect2 &p_rect, const Rect2 &p_src_rect, bool p_commit_on_flip_change) {
	if (rects.is_full()) {
		return false;
	}

	Rect2 rect = p_rect;
	Rect2 source = p_src_rect;

	uint32_t flags = flags_from_rects(rect, source);

	if (state_set) {
		// if we are changing these flips, we can no longer continue the same multirect
		if ((state.flags & (RasterizerCanvas::CANVAS_RECT_FLIP_H | RasterizerCanvas::CANVAS_RECT_FLIP_V)) != flags) {
			// different state requires a new multirect
			return false;
		}

	} else {
		state.flags |= flags;
		state_set = true;
	}

	*rects.request() = rect;
	*sources.request() = source;
	return true;
}

void MultiRect::end() {
	if (!is_empty()) {
		if (VisualServerCanvasHelper::_multirect_enabled) {
			VisualServer::get_singleton()->canvas_item_add_texture_multirect_region(state.item, rects, state.texture, sources, state.modulate, state.flags, state.normal_map);

		} else {
			// legacy path
			bool transpose = state.flags & RasterizerCanvas::CANVAS_RECT_TRANSPOSE;
			bool clip_uv = state.flags & RasterizerCanvas::CANVAS_RECT_CLIP_UV;

			for (uint32_t n = 0; n < rects.size(); n++) {
				VisualServer::get_singleton()->canvas_item_add_texture_rect_region(state.item, rects[n], state.texture, sources[n], state.modulate, transpose, state.normal_map, clip_uv);
			}
		}

		rects.clear();
		sources.clear();
	}
	state_set = false;
}

void VisualServerCanvasHelper::tilemap_begin() {
	if (_multirect_enabled) {
		_tilemap_mutex.lock();
	}
}

void VisualServerCanvasHelper::tilemap_add_rect(RID p_canvas_item, const Rect2 &p_rect, RID p_texture, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, RID p_normal_map, bool p_clip_uv) {
	if (!_multirect_enabled) {
		VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, p_texture, p_src_rect, p_modulate, p_transpose, p_normal_map, p_clip_uv);
		return;
	}

	Rect2 rect = p_rect;
	Rect2 source = p_src_rect;

	// To make the rendering code as efficient as possible,
	// a single MultiRect command should have identical flips and transpose etc.
	// If these change, it flushes the previous multirect and starts a new one.
	uint32_t flags = MultiRect::flags_from_rects(rect, source);

	if (p_transpose) {
		flags |= RasterizerCanvas::CANVAS_RECT_TRANSPOSE;
		SWAP(rect.size.x, rect.size.y);
	}

	if (p_clip_uv) {
		flags |= RasterizerCanvas::CANVAS_RECT_CLIP_UV;
	}

	State state;
	state.item = p_canvas_item;
	state.texture = p_texture;
	state.modulate = p_modulate;
	state.normal_map = p_normal_map;
	state.flags = flags;

	// attempt to add to existing multirect
	for (int n = _tilemap_multirects.size() - 1; n >= 0; n--) {
		MultiRect &mr = _tilemap_multirects[n];

		// matches state?
		if (mr.state == state) {
			// add .. this may fail if the multirect is full
			if (mr.add_pre_flipped(rect, source)) {
				return;
			}
		}

		// disallow if we overlap a multirect
		if (mr.overlaps(rect)) {
			break;
		}
	}

	// create new multirect
	_tilemap_multirects.resize(_tilemap_multirects.size() + 1);
	MultiRect &mr = _tilemap_multirects[_tilemap_multirects.size() - 1];
	mr.begin(state);
	mr.add_pre_flipped(rect, source);
}

void VisualServerCanvasHelper::tilemap_end() {
	if (!_multirect_enabled) {
		return;
	}

	for (uint32_t n = 0; n < _tilemap_multirects.size(); n++) {
		_tilemap_multirects[n].end();
	}

	_tilemap_multirects.clear();
	_tilemap_mutex.unlock();
}
