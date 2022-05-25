/*************************************************************************/
/*  base_graph_node.cpp                                                  */
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

#ifdef TOOLS_ENABLED
#include "graph_edit.h"
#endif

bool BaseGraphNode::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		int value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				_shape_title();
				update();
			}
		} else {
			if (!opentype_features.has(tag) || (int)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				_shape_title();
				update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	update();
	return true;
}

bool BaseGraphNode::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String opentype_feature_name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(opentype_feature_name);
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

	return true;
}

void BaseGraphNode::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::INT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));

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

void BaseGraphNode::set_titlebar_control(Control *p_control) {
	if (p_control) {
		titlebar_control = p_control;
		update();
	}
}

Control *BaseGraphNode::get_titlebar_control() const {
	return titlebar_control;
}

void BaseGraphNode::_close_requested() {
	// Send focus to parent.
	get_parent_control()->grab_focus();
	emit_signal(SNAME("close_request"));
}

Size2 BaseGraphNode::get_minimum_size() const {
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> sb_slot = get_theme_stylebox(SNAME("slot"));

	int separation = get_theme_constant(SNAME("separation"));
	int title_h_offset = get_theme_constant(SNAME("title_h_offset"));

	bool first = true;

	Size2 minsize;
	minsize.width = title_buf->get_size().width + title_h_offset;
	if (show_close && close_button) {
		minsize.width += close_button->get_minimum_size().width;
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

//TODO: @Geometror Finish custom titlebar implementation. Rename to _update_titlebar?
void BaseGraphNode::_resort_titlebar() {
	Size2i new_size = get_size();
	Ref<StyleBox> sb_title = get_theme_stylebox(SNAME("title_bar"));

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		//TODO: @Geometror Rename close_rect to titlebar_control_area?
		if (c == close_button) {
			Rect2 close_rect;
			Size2 close_btn_minsize = close_button->get_minimum_size();

			int close_offset = get_theme_constant(SNAME("close_v_offset"));
			int close_h_offset = get_theme_constant(SNAME("close_h_offset"));

			Vector2 cpos = Point2(new_size.width + sb_title->get_margin(SIDE_LEFT) + close_h_offset - close_btn_minsize.width, -close_btn_minsize.height + close_offset);
			//draw_texture(close, cpos, close_color);
			close_rect.position = cpos;
			close_rect.size = close_btn_minsize;

			fit_child_in_rect(close_button, close_rect);
		}
	}
}

void BaseGraphNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			_resort_titlebar();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape_title();
			if (close_button) {
				Ref<Texture2D> close_icon = get_theme_icon(SNAME("close"), SNAME("GraphNodeCloseButton"));
				close_button->set_icon(close_icon);
			}
			update_minimum_size();
			update();
		} break;
	}
}

void BaseGraphNode::_shape_title() {
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

#ifdef TOOLS_ENABLED
void BaseGraphNode::_edit_set_position(const Point2 &p_position) {
	GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
	if (graph) {
		Point2 offset = (p_position + graph->get_scroll_ofs()) * graph->get_zoom();
		set_position_offset(offset);
	}
	set_position(p_position);
}

void BaseGraphNode::_validate_property(PropertyInfo &property) const {
	Control::_validate_property(property);
	GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
	if (graph) {
		if (property.name == "rect_position") {
			property.usage |= PROPERTY_USAGE_READ_ONLY;
		}
	}
}
#endif

void BaseGraphNode::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	_shape_title();

	update();
	update_minimum_size();
}

String BaseGraphNode::get_title() const {
	return title;
}

void BaseGraphNode::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape_title();
		update();
	}
}

Control::TextDirection BaseGraphNode::get_text_direction() const {
	return text_direction;
}

void BaseGraphNode::clear_opentype_features() {
	opentype_features.clear();
	_shape_title();
	update();
}

void BaseGraphNode::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		_shape_title();
		update();
	}
}

int BaseGraphNode::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void BaseGraphNode::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape_title();
		update();
	}
}

String BaseGraphNode::get_language() const {
	return language;
}

void BaseGraphNode::set_position_offset(const Vector2 &p_offset) {
	position_offset = p_offset;
	emit_signal(SNAME("position_offset_changed"));
	update();
}

Vector2 BaseGraphNode::get_position_offset() const {
	return position_offset;
}

void BaseGraphNode::set_selected(bool p_selected) {
	selected = p_selected;
	update();
}

bool BaseGraphNode::is_selected() {
	return selected;
}

void BaseGraphNode::set_drag(bool p_drag) {
	if (p_drag) {
		drag_from = get_position_offset();
	} else {
		emit_signal(SNAME("dragged"), drag_from, get_position_offset()); // Required for undo/redo.
	}
}

Vector2 BaseGraphNode::get_drag_from() {
	return drag_from;
}

void BaseGraphNode::gui_input(const Ref<InputEvent> &p_ev) {
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

void BaseGraphNode::set_resizable(bool p_enable) {
	resizable = p_enable;
	update();
}

bool BaseGraphNode::is_resizable() const {
	return resizable;
}

void BaseGraphNode::set_show_close_button(bool p_enable) {
	show_close = p_enable;
	// Create the button only when requested.
	if (p_enable && !close_button) {
		close_button = memnew(Button);
		add_child(close_button, false, INTERNAL_MODE_FRONT);
		close_button->set_theme_type_variation("GraphNodeCloseButton");
		Ref<Texture2D> close_icon = get_theme_icon(SNAME("close"), SNAME("GraphNodeCloseButton"));
		close_button->set_icon(close_icon);
		close_button->connect("pressed", callable_mp(this, &BaseGraphNode::_close_requested));
		//close_button->set_expand_icon(true);
	} else if (!p_enable && close_button) {
		close_button->hide();
	} else {
		close_button->show();
	}
	update();
}

bool BaseGraphNode::is_close_button_visible() const {
	return show_close;
}

Vector<int> BaseGraphNode::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

Vector<int> BaseGraphNode::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_EXPAND);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

void BaseGraphNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &BaseGraphNode::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &BaseGraphNode::get_title);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &BaseGraphNode::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &BaseGraphNode::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &BaseGraphNode::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &BaseGraphNode::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &BaseGraphNode::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &BaseGraphNode::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &BaseGraphNode::get_language);

	ClassDB::bind_method(D_METHOD("set_resizable", "resizable"), &BaseGraphNode::set_resizable);
	ClassDB::bind_method(D_METHOD("is_resizable"), &BaseGraphNode::is_resizable);

	ClassDB::bind_method(D_METHOD("set_selected", "selected"), &BaseGraphNode::set_selected);
	ClassDB::bind_method(D_METHOD("is_selected"), &BaseGraphNode::is_selected);

	ClassDB::bind_method(D_METHOD("set_show_close_button", "show"), &BaseGraphNode::set_show_close_button);
	ClassDB::bind_method(D_METHOD("is_close_button_visible"), &BaseGraphNode::is_close_button_visible);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position_offset"), "set_position_offset", "get_position_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_close"), "set_show_close_button", "is_close_button_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resizable"), "set_resizable", "is_resizable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selected"), "set_selected", "is_selected");

	ADD_SIGNAL(MethodInfo("position_offset_changed"));
	ADD_SIGNAL(MethodInfo("dragged", PropertyInfo(Variant::VECTOR2, "from"), PropertyInfo(Variant::VECTOR2, "to")));
	ADD_SIGNAL(MethodInfo("raise_request"));
	ADD_SIGNAL(MethodInfo("close_request"));
	ADD_SIGNAL(MethodInfo("resize_request", PropertyInfo(Variant::VECTOR2, "new_minsize")));
}

BaseGraphNode::BaseGraphNode() {
	title_buf.instantiate();
	set_mouse_filter(MOUSE_FILTER_STOP);
}
