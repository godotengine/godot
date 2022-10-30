/**************************************************************************/
/*  graph_node.cpp                                                        */
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

#include "graph_node.h"

#include "core/string/translation.h"

bool GraphNode::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	String slot_property_name = str.get_slice("/", 2);

	Slot slot;
	if (slot_table.has(idx)) {
		slot = slot_table[idx];
	}

	if (slot_property_name == "left_enabled") {
		slot.enable_left = p_value;
	} else if (slot_property_name == "left_type") {
		slot.type_left = p_value;
	} else if (slot_property_name == "left_icon") {
		slot.custom_slot_left = p_value;
	} else if (slot_property_name == "left_color") {
		slot.color_left = p_value;
	} else if (slot_property_name == "right_enabled") {
		slot.enable_right = p_value;
	} else if (slot_property_name == "right_type") {
		slot.type_right = p_value;
	} else if (slot_property_name == "right_color") {
		slot.color_right = p_value;
	} else if (slot_property_name == "right_icon") {
		slot.custom_slot_right = p_value;
	} else if (slot_property_name == "draw_stylebox") {
		slot.draw_stylebox = p_value;
	} else {
		return false;
	}

	set_slot(idx,
			slot.enable_left,
			slot.type_left,
			slot.color_left,
			slot.enable_right,
			slot.type_right,
			slot.color_right,
			slot.custom_slot_left,
			slot.custom_slot_right,
			slot.draw_stylebox);

	queue_redraw();
	return true;
}

bool GraphNode::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	StringName slot_property_name = str.get_slice("/", 2);

	Slot slot;
	if (slot_table.has(idx)) {
		slot = slot_table[idx];
	}

	if (slot_property_name == "left_enabled") {
		r_ret = slot.enable_left;
	} else if (slot_property_name == "left_type") {
		r_ret = slot.type_left;
	} else if (slot_property_name == "left_color") {
		r_ret = slot.color_left;
	} else if (slot_property_name == "left_icon") {
		r_ret = slot.custom_slot_left;
	} else if (slot_property_name == "right_enabled") {
		r_ret = slot.enable_right;
	} else if (slot_property_name == "right_type") {
		r_ret = slot.type_right;
	} else if (slot_property_name == "right_color") {
		r_ret = slot.color_right;
	} else if (slot_property_name == "right_icon") {
		r_ret = slot.custom_slot_right;
	} else if (slot_property_name == "draw_stylebox") {
		r_ret = slot.draw_stylebox;
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
		msc.will_stretch = c->get_v_size_flags().has_flag(SIZE_EXPAND);

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

	queue_redraw();
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

			int width = get_size().width - sb_frame->get_minimum_size().x;

			title_buf->draw(get_canvas_item(), Point2(sb_frame->get_margin(SIDE_LEFT) + title_h_offset, -title_buf->get_size().y + title_offset), title_color);

			if (get_child_count() > 0) {
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
						if (!c || !c->is_visible_in_tree()) {
							continue;
						}
						Rect2 c_rect = c->get_rect();
						c_rect.position.x = sb_frame->get_margin(SIDE_LEFT);
						c_rect.size.width = width;
						draw_style_box(sb_slot, c_rect);
					}
				}
			}

			if (resizable) {
				draw_texture(resizer, get_size() - resizer->get_size(), resizer_color);
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape_title();

			update_minimum_size();
			queue_redraw();
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
	title_buf->add_string(title, font, font_size, language);
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
	queue_redraw();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::clear_slot(int p_idx) {
	slot_table.erase(p_idx);
	queue_redraw();
	port_pos_dirty = true;
}

void GraphNode::clear_all_slots() {
	slot_table.clear();
	queue_redraw();
	port_pos_dirty = true;
}

bool GraphNode::is_slot_enabled_left(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return false;
	}
	return slot_table[p_idx].enable_left;
}

void GraphNode::set_slot_enabled_left(int p_idx, bool p_enable) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_left for the slot with p_idx (%d) lesser than zero.", p_idx));

	if (slot_table[p_idx].enable_left == p_enable) {
		return;
	}

	slot_table[p_idx].enable_left = p_enable;
	queue_redraw();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_left(int p_idx, int p_type) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set type_left for the slot '%d' because it hasn't been enabled.", p_idx));

	if (slot_table[p_idx].type_left == p_type) {
		return;
	}

	slot_table[p_idx].type_left = p_type;
	queue_redraw();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_left(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return 0;
	}
	return slot_table[p_idx].type_left;
}

void GraphNode::set_slot_color_left(int p_idx, const Color &p_color) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set color_left for the slot '%d' because it hasn't been enabled.", p_idx));

	if (slot_table[p_idx].color_left == p_color) {
		return;
	}

	slot_table[p_idx].color_left = p_color;
	queue_redraw();
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

void GraphNode::set_slot_enabled_right(int p_idx, bool p_enable) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_right for the slot with p_idx (%d) lesser than zero.", p_idx));

	if (slot_table[p_idx].enable_right == p_enable) {
		return;
	}

	slot_table[p_idx].enable_right = p_enable;
	queue_redraw();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_right(int p_idx, int p_type) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set type_right for the slot '%d' because it hasn't been enabled.", p_idx));

	if (slot_table[p_idx].type_right == p_type) {
		return;
	}

	slot_table[p_idx].type_right = p_type;
	queue_redraw();
	port_pos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_right(int p_idx) const {
	if (!slot_table.has(p_idx)) {
		return 0;
	}
	return slot_table[p_idx].type_right;
}

void GraphNode::set_slot_color_right(int p_idx, const Color &p_color) {
	ERR_FAIL_COND_MSG(!slot_table.has(p_idx), vformat("Cannot set color_right for the slot '%d' because it hasn't been enabled.", p_idx));

	if (slot_table[p_idx].color_right == p_color) {
		return;
	}

	slot_table[p_idx].color_right = p_color;
	queue_redraw();
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
	queue_redraw();
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

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible()) {
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
				// port_cache.height = size.height;
				port_cache.slot_idx = child_idx;
				input_port_cache.push_back(port_cache);
			}
			if (slot_table[child_idx].enable_right) {
				PortCache port_cache;
				port_cache.pos = Point2i(get_size().width - edgeofs, pos_y + size.height / 2);
				port_cache.type = slot_table[child_idx].type_right;
				port_cache.color = slot_table[child_idx].color_right;
				// port_cache.height = size.height;
				port_cache.slot_idx = child_idx;
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

int GraphNode::get_port_output_count() {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	return output_port_cache.size();
}

Vector2 GraphNode::get_port_input_position(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, input_port_cache.size(), Vector2());
	Vector2 pos = input_port_cache[p_port_idx].pos;
	return pos;
}

int GraphNode::get_port_input_type(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, input_port_cache.size(), 0);
	return input_port_cache[p_port_idx].type;
}

Color GraphNode::get_port_input_color(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, input_port_cache.size(), Color());
	return input_port_cache[p_port_idx].color;
}

int GraphNode::get_port_input_slot(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, input_port_cache.size(), -1);
	return input_port_cache[p_port_idx].slot_idx;
}

Vector2 GraphNode::get_port_output_position(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, output_port_cache.size(), Vector2());
	Vector2 pos = output_port_cache[p_port_idx].pos;
	return pos;
}

int GraphNode::get_port_output_type(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, output_port_cache.size(), 0);
	return output_port_cache[p_port_idx].type;
}

Color GraphNode::get_port_output_color(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, output_port_cache.size(), Color());
	return output_port_cache[p_port_idx].color;
}

int GraphNode::get_port_output_slot(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, output_port_cache.size(), -1);
	return output_port_cache[p_port_idx].slot_idx;
}

void GraphNode::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid()) {
		ERR_FAIL_COND_MSG(get_parent_control() == nullptr, "GraphNode must be the child of a GraphEdit node.");

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			Vector2 mpos = mb->get_position();

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

void GraphNode::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	_shape_title();

	queue_redraw();
	update_minimum_size();
}

String GraphNode::get_title() const {
	return title;
}

void GraphNode::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape_title();
		queue_redraw();
	}
}

Control::TextDirection GraphNode::get_text_direction() const {
	return text_direction;
}

void GraphNode::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape_title();
		queue_redraw();
	}
}

String GraphNode::get_language() const {
	return language;
}

Control::CursorShape GraphNode::get_cursor_shape(const Point2 &p_pos) const {
	if (resizable) {
		Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

		if (resizing || (p_pos.x > get_size().x - resizer->get_width() && p_pos.y > get_size().y - resizer->get_height())) {
			return CURSOR_FDIAGSIZE;
		}
	}

	return Control::get_cursor_shape(p_pos);
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
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphNode::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphNode::get_title);

	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &GraphNode::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &GraphNode::get_text_direction);

	ClassDB::bind_method(D_METHOD("set_language", "language"), &GraphNode::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &GraphNode::get_language);

	ClassDB::bind_method(D_METHOD("set_slot", "slot_idx", "enable_left_port", "type_left", "color_left", "enable_right_port", "type_right", "color_right", "custom_icon_left", "custom_icon_right", "draw_stylebox"), &GraphNode::set_slot, DEFVAL(Ref<Texture2D>()), DEFVAL(Ref<Texture2D>()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("clear_slot", "slot_idx"), &GraphNode::clear_slot);
	ClassDB::bind_method(D_METHOD("clear_all_slots"), &GraphNode::clear_all_slots);

	ClassDB::bind_method(D_METHOD("is_slot_enabled_left", "slot_idx"), &GraphNode::is_slot_enabled_left);
	ClassDB::bind_method(D_METHOD("set_slot_enabled_left", "slot_idx", "enable"), &GraphNode::set_slot_enabled_left);

	ClassDB::bind_method(D_METHOD("set_slot_type_left", "slot_idx", "type"), &GraphNode::set_slot_type_left);
	ClassDB::bind_method(D_METHOD("get_slot_type_left", "slot_idx"), &GraphNode::get_slot_type_left);

	ClassDB::bind_method(D_METHOD("set_slot_color_left", "slot_idx", "color"), &GraphNode::set_slot_color_left);
	ClassDB::bind_method(D_METHOD("get_slot_color_left", "slot_idx"), &GraphNode::get_slot_color_left);

	ClassDB::bind_method(D_METHOD("is_slot_enabled_right", "slot_idx"), &GraphNode::is_slot_enabled_right);
	ClassDB::bind_method(D_METHOD("set_slot_enabled_right", "slot_idx", "enable"), &GraphNode::set_slot_enabled_right);

	ClassDB::bind_method(D_METHOD("set_slot_type_right", "slot_idx", "type"), &GraphNode::set_slot_type_right);
	ClassDB::bind_method(D_METHOD("get_slot_type_right", "slot_idx"), &GraphNode::get_slot_type_right);

	ClassDB::bind_method(D_METHOD("set_slot_color_right", "slot_idx", "color"), &GraphNode::set_slot_color_right);
	ClassDB::bind_method(D_METHOD("get_slot_color_right", "slot_idx"), &GraphNode::get_slot_color_right);

	ClassDB::bind_method(D_METHOD("is_slot_draw_stylebox", "slot_idx"), &GraphNode::is_slot_draw_stylebox);
	ClassDB::bind_method(D_METHOD("set_slot_draw_stylebox", "slot_idx", "draw_stylebox"), &GraphNode::set_slot_draw_stylebox);

	ClassDB::bind_method(D_METHOD("get_port_input_count"), &GraphNode::get_port_input_count);
	ClassDB::bind_method(D_METHOD("get_port_input_position", "port_idx"), &GraphNode::get_port_input_position);
	ClassDB::bind_method(D_METHOD("get_port_input_type", "port_idx"), &GraphNode::get_port_input_type);
	ClassDB::bind_method(D_METHOD("get_port_input_color", "port_idx"), &GraphNode::get_port_input_color);
	ClassDB::bind_method(D_METHOD("get_port_input_slot", "port_idx"), &GraphNode::get_port_input_slot);

	ClassDB::bind_method(D_METHOD("get_port_output_count"), &GraphNode::get_port_output_count);
	ClassDB::bind_method(D_METHOD("get_port_output_position", "port_idx"), &GraphNode::get_port_output_position);
	ClassDB::bind_method(D_METHOD("get_port_output_type", "port_idx"), &GraphNode::get_port_output_type);
	ClassDB::bind_method(D_METHOD("get_port_output_color", "port_idx"), &GraphNode::get_port_output_color);
	ClassDB::bind_method(D_METHOD("get_port_output_slot", "port_idx"), &GraphNode::get_port_output_slot);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_GROUP("", "");
	ADD_SIGNAL(MethodInfo("slot_updated", PropertyInfo(Variant::INT, "slot_idx")));
}

GraphNode::GraphNode() {
	title_buf.instantiate();
	set_mouse_filter(MOUSE_FILTER_STOP);
}
