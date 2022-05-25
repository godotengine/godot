/*************************************************************************/
/*  graph_node.cpp                                                       */
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

#include "graph_node.h"

#include "core/string/translation.h"

struct _MinSizeCache {
	int min_size;
	bool will_stretch;
	int final_size;
};

bool GraphNode::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	String slot_property_name = str.get_slice("/", 2);

	Slot slot_info;
	if (slot_table.has(idx)) {
		slot_info = slot_table[idx];
	}

	if (slot_property_name == "left_enabled") {
		slot_info.enable_left = p_value;
	} else if (slot_property_name == "left_type") {
		slot_info.type_left = p_value;
	} else if (slot_property_name == "left_icon") {
		slot_info.custom_slot_left = p_value;
	} else if (slot_property_name == "left_color") {
		slot_info.color_left = p_value;
	} else if (slot_property_name == "right_enabled") {
		slot_info.enable_right = p_value;
	} else if (slot_property_name == "right_type") {
		slot_info.type_right = p_value;
	} else if (slot_property_name == "right_color") {
		slot_info.color_right = p_value;
	} else if (slot_property_name == "right_icon") {
		slot_info.custom_slot_right = p_value;
	} else if (slot_property_name == "draw_stylebox") {
		slot_info.draw_stylebox = p_value;
	} else {
		return false;
	}

	set_slot(idx,
			slot_info.enable_left,
			slot_info.type_left,
			slot_info.color_left,
			slot_info.enable_right,
			slot_info.type_right,
			slot_info.color_right,
			slot_info.custom_slot_left,
			slot_info.custom_slot_right,
			slot_info.draw_stylebox);
	update();
	return true;
}

bool GraphNode::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	StringName slot_property_name = str.get_slice("/", 2);

	Slot slot_info;
	if (slot_table.has(idx)) {
		slot_info = slot_table[idx];
	}

	if (slot_property_name == "left_enabled") {
		r_ret = slot_info.enable_left;
	} else if (slot_property_name == "left_type") {
		r_ret = slot_info.type_left;
	} else if (slot_property_name == "left_color") {
		r_ret = slot_info.color_left;
	} else if (slot_property_name == "left_icon") {
		r_ret = slot_info.custom_slot_left;
	} else if (slot_property_name == "right_enabled") {
		r_ret = slot_info.enable_right;
	} else if (slot_property_name == "right_type") {
		r_ret = slot_info.type_right;
	} else if (slot_property_name == "right_color") {
		r_ret = slot_info.color_right;
	} else if (slot_property_name == "right_icon") {
		r_ret = slot_info.custom_slot_right;
	} else if (slot_property_name == "draw_stylebox") {
		r_ret = slot_info.draw_stylebox;
	} else {
		return false;
	}

	return true;
}

void GraphNode::_get_property_list(List<PropertyInfo> *p_list) const {
	int idx = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || c->is_set_as_top_level()) {
			continue;
		}

		String base = "slot/" + itos(idx) + "/";

		p_list->push_back(PropertyInfo(Variant::BOOL, base + "left_enabled"));
		p_list->push_back(PropertyInfo(Variant::INT, base + "left_type"));
		p_list->push_back(PropertyInfo(Variant::COLOR, base + "left_color"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, base + "left_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		p_list->push_back(PropertyInfo(Variant::BOOL, base + "right_enabled"));
		p_list->push_back(PropertyInfo(Variant::INT, base + "right_type"));
		p_list->push_back(PropertyInfo(Variant::COLOR, base + "right_color"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, base + "right_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
		p_list->push_back(PropertyInfo(Variant::BOOL, base + "draw_stylebox"));
		idx++;
	}
}

void GraphNode::_resort() {
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

		Size2i size = c->get_combined_minimum_size() + (slot_table[i].draw_stylebox ? sb_slot->get_minimum_size() : Size2());
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

	// Available stretch space.
	stretch_avail += stretch_diff - sb_frame->get_margin(SIDE_BOTTOM) - sb_frame->get_margin(SIDE_TOP);
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

		float margin = sb_frame->get_margin(SIDE_LEFT) + (slot_table[i].draw_stylebox ? sb_slot->get_margin(SIDE_LEFT) : 0);
		float final_width = width - (slot_table[i].draw_stylebox ? sb_slot->get_minimum_size().x : 0);
		Rect2 rect(margin, from, final_width, size);

		fit_child_in_rect(c, rect);
		cache_y.push_back(from - sb_frame->get_margin(SIDE_TOP) + size * 0.5);

		ofs = to;
		idx++;
	}

	update();
	port_pos_dirty = true;
}

void GraphNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<StyleBox> sb_frame;

			sb_frame = get_theme_stylebox(selected ? SNAME("selected_frame") : SNAME("frame"));

			Ref<StyleBox> sb_slot = get_theme_stylebox(SNAME("slot"));

			Ref<Texture2D> port = get_theme_icon(SNAME("port"));
			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			Color resizer_color = get_theme_color(SNAME("resizer_color"));
			int title_offset = get_theme_constant(SNAME("title_v_offset"));
			int title_h_offset = get_theme_constant(SNAME("title_h_offset"));
			Color title_color = get_theme_color(SNAME("title_color"));
			Point2i icofs = -port->get_size() * 0.5;
			int edgeofs = get_theme_constant(SNAME("port_offset"));
			icofs.y += sb_frame->get_margin(SIDE_TOP);

			draw_style_box(sb_frame, Rect2(Point2(), get_size()));

			switch (overlay) {
				case OVERLAY_DISABLED: {
				} break;
				case OVERLAY_BREAKPOINT: {
					draw_style_box(get_theme_stylebox(SNAME("breakpoint")), Rect2(Point2(), get_size()));
				} break;
				case OVERLAY_POSITION: {
					draw_style_box(get_theme_stylebox(SNAME("position")), Rect2(Point2(), get_size()));

				} break;
			}

			int width = get_size().width - sb_frame->get_minimum_size().x;

			title_buf->draw(get_canvas_item(), Point2(sb_frame->get_margin(SIDE_LEFT) + title_h_offset, -title_buf->get_size().y + title_offset), title_color);

			for (const KeyValue<int, Slot> &E : slot_table) {
				if (E.key < 0 || E.key >= cache_y.size()) {
					continue;
				}
				if (!slot_table.has(E.key)) {
					continue;
				}
				const Slot &slot = slot_table[E.key];
				// Left port.
				if (slot.enable_left) {
					Ref<Texture2D> p = port;
					if (slot.custom_slot_left.is_valid()) {
						p = slot.custom_slot_left;
					}
					p->draw(get_canvas_item(), icofs + Point2(edgeofs, cache_y[E.key]), slot.color_left);
				}
				// Right port.
				if (slot.enable_right) {
					Ref<Texture2D> p = port;
					if (slot.custom_slot_right.is_valid()) {
						p = slot.custom_slot_right;
					}
					p->draw(get_canvas_item(), icofs + Point2(get_size().x - edgeofs, cache_y[E.key]), slot.color_right);
				}

				// Draw slot stylebox.
				if (slot.draw_stylebox) {
					Control *c = Object::cast_to<Control>(get_child(E.key, false));
					Rect2 c_rect = c->get_rect();
					c_rect.position.x = sb_frame->get_margin(SIDE_LEFT);
					c_rect.size.width = width;
					draw_style_box(sb_slot, c_rect);
				}
			}

			if (resizable) {
				draw_texture(resizer, get_size() - resizer->get_size(), resizer_color);
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;
	}
}

void GraphNode::_shape_title() {
	Ref<Font> font = get_theme_font(SNAME("title_font"));
	int font_size = get_theme_font_size(SNAME("title_font_size"));

	title_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		title_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		title_buf->set_direction((TextServer::Direction)text_direction);
	}
	title_buf->add_string(title, font, font_size, opentype_features, (!language.is_empty()) ? language : TranslationServer::get_singleton()->get_tool_locale());
}

void GraphNode::set_slot(int p_idx, bool p_enable_left, int p_type_left, const Color &p_color_left, bool p_enable_right, int p_type_right, const Color &p_color_right, const Ref<Texture2D> &p_custom_left, const Ref<Texture2D> &p_custom_right, bool p_draw_stylebox) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set slot with p_idx (%d) lesser than zero.", p_idx));

	if (!p_enable_left && p_type_left == 0 && p_color_left == Color(1, 1, 1, 1) &&
			!p_enable_right && p_type_right == 0 && p_color_right == Color(1, 1, 1, 1) &&
			!p_custom_left.is_valid() && !p_custom_right.is_valid()) {
		slot_table.erase(p_idx);
		return;
	}

	Slot slot;
	slot.enable_left = p_enable_left;
	slot.type_left = p_type_left;
	slot.color_left = p_color_left;
	slot.enable_right = p_enable_right;
	slot.type_right = p_type_right;
	slot.color_right = p_color_right;
	slot.custom_slot_left = p_custom_left;
	slot.custom_slot_right = p_custom_right;
	slot.draw_stylebox = p_draw_stylebox;
	slot_table[p_idx] = slot;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::clear_slot(int p_idx) {
	slot_table.erase(p_idx);
	update();
	port_pos_dirty = true;
}

void GraphNode::clear_all_slots() {
	slot_table.clear();
	update();
	port_pos_dirty = true;
}

bool GraphNode::is_slot_enabled_left(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return false;
	}
	return slot_table[p_idx].enable_left;
}

void GraphNode::set_slot_enabled_left(int p_idx, bool p_enable_left) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_left for the slot with p_idx (%d) lesser than zero.", p_idx));

	slot_table[p_idx].enable_left = p_enable_left;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_left(int p_idx, int p_type_left) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set type_left for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_table[p_idx].type_left = p_type_left;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_left(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return 0;
	}
	return slot_table[p_idx].type_left;
}

void GraphNode::set_slot_color_left(int p_idx, const Color &p_color_left) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set color_left for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_table[p_idx].color_left = p_color_left;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

Color GraphNode::get_slot_color_left(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return Color(1, 1, 1, 1);
	}
	return slot_table[p_idx].color_left;
}

bool GraphNode::is_slot_enabled_right(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return false;
	}
	return slot_table[p_idx].enable_right;
}

void GraphNode::set_slot_enabled_right(int p_idx, bool p_enable_right) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_right for the slot with p_idx (%d) lesser than zero.", p_idx));

	slot_table[p_idx].enable_right = p_enable_right;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_right(int p_idx, int p_type_right) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set type_right for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_table[p_idx].type_right = p_type_right;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_right(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return 0;
	}
	return slot_table[p_idx].type_right;
}

void GraphNode::set_slot_color_right(int p_idx, const Color &p_color_right) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set color_right for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_table[p_idx].color_right = p_color_right;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

Color GraphNode::get_slot_color_right(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return Color(1, 1, 1, 1);
	}
	return slot_table[p_idx].color_right;
}

bool GraphNode::is_slot_draw_stylebox(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return false;
	}
	return slot_table[p_idx].draw_stylebox;
}

void GraphNode::set_slot_draw_stylebox(int p_idx, bool p_enable) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set draw_stylebox for the slot with p_idx (%d) lesser than zero.", p_idx));

	slot_table[p_idx].draw_stylebox = p_enable;
	update();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

Size2 GraphNode::get_minimum_size() const {
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> sb_slot = get_theme_stylebox(SNAME("slot"));

	int separation = get_theme_constant(SNAME("separation"));
	int title_h_offset = get_theme_constant(SNAME("title_h_offset"));

	bool first = true;

	Size2 minsize;
	minsize.x = title_buf->get_size().x + title_h_offset;

	//TODO: @Geometror Change how this works.
	if (show_close) {
		int close_h_offset = get_theme_constant(SNAME("close_h_offset"));
		minsize.x += close_button->get_size().width + close_h_offset;
	}

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_combined_minimum_size();
		if (slot_table.has(i)) {
			size += slot_table[i].draw_stylebox ? sb_slot->get_minimum_size() : Size2();
		}

		minsize.y += size.y;
		minsize.x = MAX(minsize.x, size.x);

		if (first) {
			first = false;
		} else {
			minsize.y += separation;
		}
	}

	return minsize + sb_frame->get_minimum_size();
}

void GraphNode::_port_pos_update() {
	int edgeofs = get_theme_constant(SNAME("port_offset"));
	int separation = get_theme_constant(SNAME("separation"));

	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	input_port_cache.clear();
	output_port_cache.clear();
	int vertical_ofs = 0;

	int child_idx = 0;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_rect().size;

		int pos_y = sb_frame->get_margin(SIDE_TOP) + vertical_ofs;

		if (slot_table.has(child_idx)) {
			if (slot_table[child_idx].enable_left) {
				PortCache port_cache;
				port_cache.pos = Point2i(edgeofs, pos_y + size.height / 2);
				port_cache.type = slot_table[child_idx].type_left;
				port_cache.color = slot_table[child_idx].color_left;
				port_cache.height = size.height;
				input_port_cache.push_back(port_cache);
			}
			if (slot_table[child_idx].enable_right) {
				PortCache port_cache;
				port_cache.pos = Point2i(get_size().width - edgeofs, pos_y + size.height / 2);
				port_cache.type = slot_table[child_idx].type_right;
				port_cache.color = slot_table[child_idx].color_right;
				port_cache.height = size.height;
				output_port_cache.push_back(port_cache);
			}
		}

		vertical_ofs += separation;
		vertical_ofs += size.y;
		child_idx++;
	}

	port_pos_dirty = false;
}

int GraphNode::get_port_input_count() {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	return input_port_cache.size();
}

int GraphNode::get_port_output_height(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, output_port_cache.size(), 0);
	return output_port_cache[p_idx].height;
}

int GraphNode::get_port_input_height(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, input_port_cache.size(), 0);
	return input_port_cache[p_idx].height;
}

int GraphNode::get_port_output_count() {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	return output_port_cache.size();
}

Vector2 GraphNode::get_port_input_position(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, input_port_cache.size(), Vector2());
	Vector2 pos = input_port_cache[p_idx].pos;
	pos.x *= get_scale().x;
	pos.y *= get_scale().y;
	return pos;
}

int GraphNode::get_port_input_type(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, input_port_cache.size(), 0);
	return input_port_cache[p_idx].type;
}

Color GraphNode::get_port_input_color(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, input_port_cache.size(), Color());
	return input_port_cache[p_idx].color;
}

Vector2 GraphNode::get_port_output_position(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, output_port_cache.size(), Vector2());
	Vector2 pos = output_port_cache[p_idx].pos;
	pos.x *= get_scale().x;
	pos.y *= get_scale().y;
	return pos;
}

int GraphNode::get_port_output_type(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, output_port_cache.size(), 0);
	return output_port_cache[p_idx].type;
}

Color GraphNode::get_port_output_color(int p_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, output_port_cache.size(), Color());
	return output_port_cache[p_idx].color;
}

void GraphNode::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid()) {
		ERR_FAIL_COND_MSG(get_parent_control() == nullptr, "GraphNode must be the child of a GraphEdit node.");

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			Vector2 mpos = mb->get_position();
			if (close_rect.size != Size2() && close_rect.has_point(mpos)) {
				// Send focus to parent.
				get_parent_control()->grab_focus();
				emit_signal(SNAME("close_request"));
				accept_event();
				return;
			}

			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

			if (resizable && mpos.x > get_size().x - resizer->get_width() && mpos.y > get_size().y - resizer->get_height()) {
				resizing = true;
				resizing_from = mpos;
				resizing_from_size = get_size();
				accept_event();
				return;
			}

			emit_signal(SNAME("raise_request"));
		}

		if (!mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			resizing = false;
		}
	}

	Ref<InputEventMouseMotion> mm = p_ev;
	if (resizing && mm.is_valid()) {
		Vector2 mpos = mm->get_position();

		Vector2 diff = mpos - resizing_from;

		emit_signal(SNAME("resize_request"), resizing_from_size + diff);
	}
}

void GraphNode::set_overlay(Overlay p_overlay) {
	overlay = p_overlay;
	update();
}

GraphNode::Overlay GraphNode::get_overlay() const {
	return overlay;
}

Vector<int> GraphNode::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

Vector<int> GraphNode::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_EXPAND);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

void GraphNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_slot", "idx", "enable_left", "type_left", "color_left", "enable_right", "type_right", "color_right", "custom_left", "custom_right", "enable"), &GraphNode::set_slot, DEFVAL(Ref<Texture2D>()), DEFVAL(Ref<Texture2D>()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("clear_slot", "idx"), &GraphNode::clear_slot);
	ClassDB::bind_method(D_METHOD("clear_all_slots"), &GraphNode::clear_all_slots);

	ClassDB::bind_method(D_METHOD("is_slot_enabled_left", "idx"), &GraphNode::is_slot_enabled_left);
	ClassDB::bind_method(D_METHOD("set_slot_enabled_left", "idx", "enable_left"), &GraphNode::set_slot_enabled_left);

	ClassDB::bind_method(D_METHOD("set_slot_type_left", "idx", "type_left"), &GraphNode::set_slot_type_left);
	ClassDB::bind_method(D_METHOD("get_slot_type_left", "idx"), &GraphNode::get_slot_type_left);

	ClassDB::bind_method(D_METHOD("set_slot_color_left", "idx", "color_left"), &GraphNode::set_slot_color_left);
	ClassDB::bind_method(D_METHOD("get_slot_color_left", "idx"), &GraphNode::get_slot_color_left);

	ClassDB::bind_method(D_METHOD("is_slot_enabled_right", "idx"), &GraphNode::is_slot_enabled_right);
	ClassDB::bind_method(D_METHOD("set_slot_enabled_right", "idx", "enable_right"), &GraphNode::set_slot_enabled_right);

	ClassDB::bind_method(D_METHOD("set_slot_type_right", "idx", "type_right"), &GraphNode::set_slot_type_right);
	ClassDB::bind_method(D_METHOD("get_slot_type_right", "idx"), &GraphNode::get_slot_type_right);

	ClassDB::bind_method(D_METHOD("set_slot_color_right", "idx", "color_right"), &GraphNode::set_slot_color_right);
	ClassDB::bind_method(D_METHOD("get_slot_color_right", "idx"), &GraphNode::get_slot_color_right);

	ClassDB::bind_method(D_METHOD("is_slot_draw_stylebox", "idx"), &GraphNode::is_slot_draw_stylebox);
	ClassDB::bind_method(D_METHOD("set_slot_draw_stylebox", "idx", "draw_stylebox"), &GraphNode::set_slot_draw_stylebox);

	ClassDB::bind_method(D_METHOD("set_position_offset", "offset"), &GraphNode::set_position_offset);
	ClassDB::bind_method(D_METHOD("get_position_offset"), &GraphNode::get_position_offset);

	ClassDB::bind_method(D_METHOD("set_resizable", "resizable"), &GraphNode::set_resizable);
	ClassDB::bind_method(D_METHOD("is_resizable"), &GraphNode::is_resizable);

	ClassDB::bind_method(D_METHOD("set_selected", "selected"), &GraphNode::set_selected);
	ClassDB::bind_method(D_METHOD("is_selected"), &GraphNode::is_selected);

	ClassDB::bind_method(D_METHOD("get_port_output_count"), &GraphNode::get_port_output_count);
	ClassDB::bind_method(D_METHOD("get_port_input_count"), &GraphNode::get_port_input_count);

	ClassDB::bind_method(D_METHOD("get_port_output_position", "idx"), &GraphNode::get_port_output_position);
	ClassDB::bind_method(D_METHOD("get_port_output_type", "idx"), &GraphNode::get_port_output_type);
	ClassDB::bind_method(D_METHOD("get_port_output_color", "idx"), &GraphNode::get_port_output_color);
	ClassDB::bind_method(D_METHOD("get_port_input_position", "idx"), &GraphNode::get_port_input_position);
	ClassDB::bind_method(D_METHOD("get_port_input_type", "idx"), &GraphNode::get_port_input_type);
	ClassDB::bind_method(D_METHOD("get_port_input_color", "idx"), &GraphNode::get_port_input_color);

	ClassDB::bind_method(D_METHOD("set_overlay", "overlay"), &GraphNode::set_overlay);
	ClassDB::bind_method(D_METHOD("get_overlay"), &GraphNode::get_overlay);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "overlay", PROPERTY_HINT_ENUM, "Disabled,Breakpoint,Position"), "set_overlay", "get_overlay");

	ADD_SIGNAL(MethodInfo("slot_updated", PropertyInfo(Variant::INT, "idx")));

	BIND_ENUM_CONSTANT(OVERLAY_DISABLED);
	BIND_ENUM_CONSTANT(OVERLAY_BREAKPOINT);
	BIND_ENUM_CONSTANT(OVERLAY_POSITION);
}

GraphNode::GraphNode() {
	title_buf.instantiate();
	set_mouse_filter(MOUSE_FILTER_STOP);
}
