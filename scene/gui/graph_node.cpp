/*************************************************************************/
/*  graph_node.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#ifdef TOOLS_ENABLED
#include "graph_edit.h"
#endif

struct _MinSizeCache {
	int min_size;
	bool will_stretch;
	int final_size;
};

bool GraphNode::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		double value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				_shape();
				update();
			}
		} else {
			if ((double)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				_shape();
				update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	String what = str.get_slice("/", 2);

	Slot si;
	if (slot_info.has(idx)) {
		si = slot_info[idx];
	}

	if (what == "left_enabled") {
		si.enable_left = p_value;
	} else if (what == "left_type") {
		si.type_left = p_value;
	} else if (what == "left_icon") {
		si.custom_slot_left = p_value;
	} else if (what == "left_color") {
		si.color_left = p_value;
	} else if (what == "right_enabled") {
		si.enable_right = p_value;
	} else if (what == "right_type") {
		si.type_right = p_value;
	} else if (what == "right_color") {
		si.color_right = p_value;
	} else if (what == "right_icon") {
		si.custom_slot_right = p_value;
	} else {
		return false;
	}

	set_slot(idx, si.enable_left, si.type_left, si.color_left, si.enable_right, si.type_right, si.color_right, si.custom_slot_left, si.custom_slot_right);
	update();
	return true;
}

bool GraphNode::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}

	if (!str.begins_with("slot/")) {
		return false;
	}

	int idx = str.get_slice("/", 1).to_int();
	String what = str.get_slice("/", 2);

	Slot si;
	if (slot_info.has(idx)) {
		si = slot_info[idx];
	}

	if (what == "left_enabled") {
		r_ret = si.enable_left;
	} else if (what == "left_type") {
		r_ret = si.type_left;
	} else if (what == "left_color") {
		r_ret = si.color_left;
	} else if (what == "left_icon") {
		r_ret = si.custom_slot_left;
	} else if (what == "right_enabled") {
		r_ret = si.enable_right;
	} else if (what == "right_type") {
		r_ret = si.type_right;
	} else if (what == "right_color") {
		r_ret = si.color_right;
	} else if (what == "right_icon") {
		r_ret = si.custom_slot_right;
	} else {
		return false;
	}

	return true;
}

void GraphNode::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::FLOAT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));

	int idx = 0;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
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

		idx++;
	}
}

void GraphNode::_resort() {
	/** First pass, determine minimum size AND amount of stretchable elements */

	Size2i new_size = get_size();
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("frame"));

	int sep = get_theme_constant(SNAME("separation"));

	bool first = true;
	int children_count = 0;
	int stretch_min = 0;
	int stretch_avail = 0;
	float stretch_ratio_total = 0;
	Map<Control *, _MinSizeCache> min_size_cache;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
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

	int stretch_max = new_size.height - (children_count - 1) * sep;
	int stretch_diff = stretch_max - stretch_min;
	if (stretch_diff < 0) {
		//avoid negative stretch space
		stretch_diff = 0;
	}

	stretch_avail += stretch_diff - sb->get_margin(SIDE_BOTTOM) - sb->get_margin(SIDE_TOP); //available stretch space.
	/** Second, pass successively to discard elements that can't be stretched, this will run while stretchable
		elements exist */

	while (stretch_ratio_total > 0) { // first of all, don't even be here if no stretchable objects exist
		bool refit_successful = true; //assume refit-test will go well

		for (int i = 0; i < get_child_count(); i++) {
			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c || !c->is_visible_in_tree()) {
				continue;
			}
			if (c->is_set_as_top_level()) {
				continue;
			}

			ERR_FAIL_COND(!min_size_cache.has(c));
			_MinSizeCache &msc = min_size_cache[c];

			if (msc.will_stretch) { //wants to stretch
				//let's see if it can really stretch

				int final_pixel_size = stretch_avail * c->get_stretch_ratio() / stretch_ratio_total;
				if (final_pixel_size < msc.min_size) {
					//if available stretching area is too small for widget,
					//then remove it from stretching area
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

		if (refit_successful) { //uf refit went well, break
			break;
		}
	}

	/** Final pass, draw and stretch elements **/

	int ofs = sb->get_margin(SIDE_TOP);

	first = true;
	int idx = 0;
	cache_y.clear();
	int w = new_size.width - sb->get_minimum_size().x;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
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
			ofs += sep;
		}

		int from = ofs;
		int to = ofs + msc.final_size;

		if (msc.will_stretch && idx == children_count - 1) {
			//adjust so the last one always fits perfect
			//compensating for numerical imprecision

			to = new_size.height - sb->get_margin(SIDE_BOTTOM);
		}

		int size = to - from;

		Rect2 rect(sb->get_margin(SIDE_LEFT), from, w, size);

		fit_child_in_rect(c, rect);
		cache_y.push_back(from - sb->get_margin(SIDE_TOP) + size * 0.5);

		ofs = to;
		idx++;
	}

	update();
	connpos_dirty = true;
}

bool GraphNode::has_point(const Point2 &p_point) const {
	if (comment) {
		Ref<StyleBox> comment = get_theme_stylebox(SNAME("comment"));
		Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

		if (Rect2(get_size() - resizer->get_size(), resizer->get_size()).has_point(p_point)) {
			return true;
		}

		if (Rect2(0, 0, get_size().width, comment->get_margin(SIDE_TOP)).has_point(p_point)) {
			return true;
		}

		return false;

	} else {
		return Control::has_point(p_point);
	}
}

void GraphNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<StyleBox> sb;

			if (comment) {
				sb = get_theme_stylebox(selected ? "commentfocus" : "comment");

			} else {
				sb = get_theme_stylebox(selected ? "selectedframe" : "frame");
			}

			//sb=sb->duplicate();
			//sb->call("set_modulate",modulate);
			Ref<Texture2D> port = get_theme_icon(SNAME("port"));
			Ref<Texture2D> close = get_theme_icon(SNAME("close"));
			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			int close_offset = get_theme_constant(SNAME("close_offset"));
			int close_h_offset = get_theme_constant(SNAME("close_h_offset"));
			Color close_color = get_theme_color(SNAME("close_color"));
			Color resizer_color = get_theme_color(SNAME("resizer_color"));
			int title_offset = get_theme_constant(SNAME("title_offset"));
			int title_h_offset = get_theme_constant(SNAME("title_h_offset"));
			Color title_color = get_theme_color(SNAME("title_color"));
			Point2i icofs = -port->get_size() * 0.5;
			int edgeofs = get_theme_constant(SNAME("port_offset"));
			icofs.y += sb->get_margin(SIDE_TOP);

			draw_style_box(sb, Rect2(Point2(), get_size()));

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

			int w = get_size().width - sb->get_minimum_size().x;

			if (show_close) {
				w -= close->get_width();
			}

			title_buf->set_width(w);
			title_buf->draw(get_canvas_item(), Point2(sb->get_margin(SIDE_LEFT) + title_h_offset, -title_buf->get_size().y + title_offset), title_color);
			if (show_close) {
				Vector2 cpos = Point2(w + sb->get_margin(SIDE_LEFT) + close_h_offset, -close->get_height() + close_offset);
				draw_texture(close, cpos, close_color);
				close_rect.position = cpos;
				close_rect.size = close->get_size();
			} else {
				close_rect = Rect2();
			}

			for (const KeyValue<int, Slot> &E : slot_info) {
				if (E.key < 0 || E.key >= cache_y.size()) {
					continue;
				}
				if (!slot_info.has(E.key)) {
					continue;
				}
				const Slot &s = slot_info[E.key];
				//left
				if (s.enable_left) {
					Ref<Texture2D> p = port;
					if (s.custom_slot_left.is_valid()) {
						p = s.custom_slot_left;
					}
					p->draw(get_canvas_item(), icofs + Point2(edgeofs, cache_y[E.key]), s.color_left);
				}
				if (s.enable_right) {
					Ref<Texture2D> p = port;
					if (s.custom_slot_right.is_valid()) {
						p = s.custom_slot_right;
					}
					p->draw(get_canvas_item(), icofs + Point2(get_size().x - edgeofs, cache_y[E.key]), s.color_right);
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
			_shape();

			minimum_size_changed();
			update();
		} break;
	}
}

void GraphNode::_shape() {
	Ref<Font> font = get_theme_font(SNAME("title_font"));
	int font_size = get_theme_font_size(SNAME("title_font_size"));

	title_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		title_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		title_buf->set_direction((TextServer::Direction)text_direction);
	}
	title_buf->add_string(title, font, font_size, opentype_features, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
}

#ifdef TOOLS_ENABLED
void GraphNode::_edit_set_position(const Point2 &p_position) {
	GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
	if (graph) {
		Point2 offset = (p_position + graph->get_scroll_ofs()) * graph->get_zoom();
		set_position_offset(offset);
	}
	set_position(p_position);
}

void GraphNode::_validate_property(PropertyInfo &property) const {
	Control::_validate_property(property);
	GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
	if (graph) {
		if (property.name == "rect_position") {
			property.usage |= PROPERTY_USAGE_READ_ONLY;
		}
	}
}
#endif

void GraphNode::set_slot(int p_idx, bool p_enable_left, int p_type_left, const Color &p_color_left, bool p_enable_right, int p_type_right, const Color &p_color_right, const Ref<Texture2D> &p_custom_left, const Ref<Texture2D> &p_custom_right) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set slot with p_idx (%d) lesser than zero.", p_idx));

	if (!p_enable_left && p_type_left == 0 && p_color_left == Color(1, 1, 1, 1) &&
			!p_enable_right && p_type_right == 0 && p_color_right == Color(1, 1, 1, 1) &&
			!p_custom_left.is_valid() && !p_custom_right.is_valid()) {
		slot_info.erase(p_idx);
		return;
	}

	Slot s;
	s.enable_left = p_enable_left;
	s.type_left = p_type_left;
	s.color_left = p_color_left;
	s.enable_right = p_enable_right;
	s.type_right = p_type_right;
	s.color_right = p_color_right;
	s.custom_slot_left = p_custom_left;
	s.custom_slot_right = p_custom_right;
	slot_info[p_idx] = s;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::clear_slot(int p_idx) {
	slot_info.erase(p_idx);
	update();
	connpos_dirty = true;
}

void GraphNode::clear_all_slots() {
	slot_info.clear();
	update();
	connpos_dirty = true;
}

bool GraphNode::is_slot_enabled_left(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return false;
	}
	return slot_info[p_idx].enable_left;
}

void GraphNode::set_slot_enabled_left(int p_idx, bool p_enable_left) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_left for the slot with p_idx (%d) lesser than zero.", p_idx));

	slot_info[p_idx].enable_left = p_enable_left;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_left(int p_idx, int p_type_left) {
	ERR_FAIL_COND_MSG(!slot_info.has(p_idx), vformat("Cannot set type_left for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_info[p_idx].type_left = p_type_left;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_left(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return 0;
	}
	return slot_info[p_idx].type_left;
}

void GraphNode::set_slot_color_left(int p_idx, const Color &p_color_left) {
	ERR_FAIL_COND_MSG(!slot_info.has(p_idx), vformat("Cannot set color_left for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_info[p_idx].color_left = p_color_left;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

Color GraphNode::get_slot_color_left(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return Color(1, 1, 1, 1);
	}
	return slot_info[p_idx].color_left;
}

bool GraphNode::is_slot_enabled_right(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return false;
	}
	return slot_info[p_idx].enable_right;
}

void GraphNode::set_slot_enabled_right(int p_idx, bool p_enable_right) {
	ERR_FAIL_COND_MSG(p_idx < 0, vformat("Cannot set enable_right for the slot with p_idx (%d) lesser than zero.", p_idx));

	slot_info[p_idx].enable_right = p_enable_right;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

void GraphNode::set_slot_type_right(int p_idx, int p_type_right) {
	ERR_FAIL_COND_MSG(!slot_info.has(p_idx), vformat("Cannot set type_right for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_info[p_idx].type_right = p_type_right;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

int GraphNode::get_slot_type_right(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return 0;
	}
	return slot_info[p_idx].type_right;
}

void GraphNode::set_slot_color_right(int p_idx, const Color &p_color_right) {
	ERR_FAIL_COND_MSG(!slot_info.has(p_idx), vformat("Cannot set color_right for the slot '%d' because it hasn't been enabled.", p_idx));

	slot_info[p_idx].color_right = p_color_right;
	update();
	connpos_dirty = true;

	emit_signal(SNAME("slot_updated"), p_idx);
}

Color GraphNode::get_slot_color_right(int p_idx) const {
	if (!slot_info.has(p_idx)) {
		return Color(1, 1, 1, 1);
	}
	return slot_info[p_idx].color_right;
}

Size2 GraphNode::get_minimum_size() const {
	int sep = get_theme_constant(SNAME("separation"));
	Ref<StyleBox> sb = get_theme_stylebox(SNAME("frame"));
	bool first = true;

	Size2 minsize;
	minsize.x = title_buf->get_size().x;
	if (show_close) {
		Ref<Texture2D> close = get_theme_icon(SNAME("close"));
		minsize.x += sep + close->get_width();
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_combined_minimum_size();

		minsize.y += size.y;
		minsize.x = MAX(minsize.x, size.x);

		if (first) {
			first = false;
		} else {
			minsize.y += sep;
		}
	}

	return minsize + sb->get_minimum_size();
}

void GraphNode::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	_shape();

	update();
	minimum_size_changed();
}

String GraphNode::get_title() const {
	return title;
}

void GraphNode::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape();
		update();
	}
}

Control::TextDirection GraphNode::get_text_direction() const {
	return text_direction;
}

void GraphNode::clear_opentype_features() {
	opentype_features.clear();
	_shape();
	update();
}

void GraphNode::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		_shape();
		update();
	}
}

int GraphNode::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void GraphNode::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		update();
	}
}

String GraphNode::get_language() const {
	return language;
}

void GraphNode::set_position_offset(const Vector2 &p_offset) {
	position_offset = p_offset;
	emit_signal(SNAME("position_offset_changed"));
	update();
}

Vector2 GraphNode::get_position_offset() const {
	return position_offset;
}

void GraphNode::set_selected(bool p_selected) {
	selected = p_selected;
	update();
}

bool GraphNode::is_selected() {
	return selected;
}

void GraphNode::set_drag(bool p_drag) {
	if (p_drag) {
		drag_from = get_position_offset();
	} else {
		emit_signal(SNAME("dragged"), drag_from, get_position_offset()); //useful for undo/redo
	}
}

Vector2 GraphNode::get_drag_from() {
	return drag_from;
}

void GraphNode::set_show_close_button(bool p_enable) {
	show_close = p_enable;
	update();
}

bool GraphNode::is_close_button_visible() const {
	return show_close;
}

void GraphNode::_connpos_update() {
	int edgeofs = get_theme_constant(SNAME("port_offset"));
	int sep = get_theme_constant(SNAME("separation"));

	Ref<StyleBox> sb = get_theme_stylebox(SNAME("frame"));
	conn_input_cache.clear();
	conn_output_cache.clear();
	int vofs = 0;

	int idx = 0;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_rect().size;

		int y = sb->get_margin(SIDE_TOP) + vofs;
		int h = size.y;

		if (slot_info.has(idx)) {
			if (slot_info[idx].enable_left) {
				ConnCache cc;
				cc.pos = Point2i(edgeofs, y + h / 2);
				cc.type = slot_info[idx].type_left;
				cc.color = slot_info[idx].color_left;
				conn_input_cache.push_back(cc);
			}
			if (slot_info[idx].enable_right) {
				ConnCache cc;
				cc.pos = Point2i(get_size().width - edgeofs, y + h / 2);
				cc.type = slot_info[idx].type_right;
				cc.color = slot_info[idx].color_right;
				conn_output_cache.push_back(cc);
			}
		}

		vofs += sep;
		vofs += size.y;
		idx++;
	}

	connpos_dirty = false;
}

int GraphNode::get_connection_input_count() {
	if (connpos_dirty) {
		_connpos_update();
	}

	return conn_input_cache.size();
}

int GraphNode::get_connection_output_count() {
	if (connpos_dirty) {
		_connpos_update();
	}

	return conn_output_cache.size();
}

Vector2 GraphNode::get_connection_input_position(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_input_cache.size(), Vector2());
	Vector2 pos = conn_input_cache[p_idx].pos;
	pos.x *= get_scale().x;
	pos.y *= get_scale().y;
	return pos;
}

int GraphNode::get_connection_input_type(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_input_cache.size(), 0);
	return conn_input_cache[p_idx].type;
}

Color GraphNode::get_connection_input_color(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_input_cache.size(), Color());
	return conn_input_cache[p_idx].color;
}

Vector2 GraphNode::get_connection_output_position(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_output_cache.size(), Vector2());
	Vector2 pos = conn_output_cache[p_idx].pos;
	pos.x *= get_scale().x;
	pos.y *= get_scale().y;
	return pos;
}

int GraphNode::get_connection_output_type(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_output_cache.size(), 0);
	return conn_output_cache[p_idx].type;
}

Color GraphNode::get_connection_output_color(int p_idx) {
	if (connpos_dirty) {
		_connpos_update();
	}

	ERR_FAIL_INDEX_V(p_idx, conn_output_cache.size(), Color());
	return conn_output_cache[p_idx].color;
}

void GraphNode::gui_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventMouseButton> mb = p_ev;
	if (mb.is_valid()) {
		ERR_FAIL_COND_MSG(get_parent_control() == nullptr, "GraphNode must be the child of a GraphEdit node.");

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			Vector2 mpos = mb->get_position();
			if (close_rect.size != Size2() && close_rect.has_point(mpos)) {
				//send focus to parent
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

void GraphNode::set_comment(bool p_enable) {
	comment = p_enable;
	update();
}

bool GraphNode::is_comment() const {
	return comment;
}

void GraphNode::set_resizable(bool p_enable) {
	resizable = p_enable;
	update();
}

bool GraphNode::is_resizable() const {
	return resizable;
}

void GraphNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphNode::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphNode::get_title);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &GraphNode::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &GraphNode::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &GraphNode::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &GraphNode::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &GraphNode::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &GraphNode::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &GraphNode::get_language);

	ClassDB::bind_method(D_METHOD("set_slot", "idx", "enable_left", "type_left", "color_left", "enable_right", "type_right", "color_right", "custom_left", "custom_right"), &GraphNode::set_slot, DEFVAL(Ref<Texture2D>()), DEFVAL(Ref<Texture2D>()));
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

	ClassDB::bind_method(D_METHOD("set_position_offset", "offset"), &GraphNode::set_position_offset);
	ClassDB::bind_method(D_METHOD("get_position_offset"), &GraphNode::get_position_offset);

	ClassDB::bind_method(D_METHOD("set_comment", "comment"), &GraphNode::set_comment);
	ClassDB::bind_method(D_METHOD("is_comment"), &GraphNode::is_comment);

	ClassDB::bind_method(D_METHOD("set_resizable", "resizable"), &GraphNode::set_resizable);
	ClassDB::bind_method(D_METHOD("is_resizable"), &GraphNode::is_resizable);

	ClassDB::bind_method(D_METHOD("set_selected", "selected"), &GraphNode::set_selected);
	ClassDB::bind_method(D_METHOD("is_selected"), &GraphNode::is_selected);

	ClassDB::bind_method(D_METHOD("get_connection_output_count"), &GraphNode::get_connection_output_count);
	ClassDB::bind_method(D_METHOD("get_connection_input_count"), &GraphNode::get_connection_input_count);

	ClassDB::bind_method(D_METHOD("get_connection_output_position", "idx"), &GraphNode::get_connection_output_position);
	ClassDB::bind_method(D_METHOD("get_connection_output_type", "idx"), &GraphNode::get_connection_output_type);
	ClassDB::bind_method(D_METHOD("get_connection_output_color", "idx"), &GraphNode::get_connection_output_color);
	ClassDB::bind_method(D_METHOD("get_connection_input_position", "idx"), &GraphNode::get_connection_input_position);
	ClassDB::bind_method(D_METHOD("get_connection_input_type", "idx"), &GraphNode::get_connection_input_type);
	ClassDB::bind_method(D_METHOD("get_connection_input_color", "idx"), &GraphNode::get_connection_input_color);

	ClassDB::bind_method(D_METHOD("set_show_close_button", "show"), &GraphNode::set_show_close_button);
	ClassDB::bind_method(D_METHOD("is_close_button_visible"), &GraphNode::is_close_button_visible);

	ClassDB::bind_method(D_METHOD("set_overlay", "overlay"), &GraphNode::set_overlay);
	ClassDB::bind_method(D_METHOD("get_overlay"), &GraphNode::get_overlay);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position_offset"), "set_position_offset", "get_position_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_close"), "set_show_close_button", "is_close_button_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resizable"), "set_resizable", "is_resizable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selected"), "set_selected", "is_selected");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "comment"), "set_comment", "is_comment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "overlay", PROPERTY_HINT_ENUM, "Disabled,Breakpoint,Position"), "set_overlay", "get_overlay");

	ADD_SIGNAL(MethodInfo("position_offset_changed"));
	ADD_SIGNAL(MethodInfo("slot_updated", PropertyInfo(Variant::INT, "idx")));
	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::VECTOR2, "from"), PropertyInfo(Variant::VECTOR2, "to")));
	ADD_SIGNAL(MethodInfo("raise_request"));
	ADD_SIGNAL(MethodInfo("close_request"));
	ADD_SIGNAL(MethodInfo("resize_request", PropertyInfo(Variant::VECTOR2, "new_minsize")));

	BIND_ENUM_CONSTANT(OVERLAY_DISABLED);
	BIND_ENUM_CONSTANT(OVERLAY_BREAKPOINT);
	BIND_ENUM_CONSTANT(OVERLAY_POSITION);
}

GraphNode::GraphNode() {
	title_buf.instantiate();
	set_mouse_filter(MOUSE_FILTER_STOP);
}
