/*************************************************************************/
/*  control.cpp                                                          */
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

#include "control.h"

#include "container.h"
#include "core/config/project_settings.h"
#include "core/math/geometry_2d.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"
#include "servers/rendering_server.h"
#include "servers/text_server.h"

#ifdef TOOLS_ENABLED
#include "editor/plugins/canvas_item_editor_plugin.h"
#endif

#ifdef TOOLS_ENABLED
Dictionary Control::_edit_get_state() const {
	Dictionary s;
	s["rotation"] = get_rotation();
	s["scale"] = get_scale();
	s["pivot"] = get_pivot_offset();
	Array anchors;
	anchors.push_back(get_anchor(SIDE_LEFT));
	anchors.push_back(get_anchor(SIDE_TOP));
	anchors.push_back(get_anchor(SIDE_RIGHT));
	anchors.push_back(get_anchor(SIDE_BOTTOM));
	s["anchors"] = anchors;
	Array offsets;
	offsets.push_back(get_offset(SIDE_LEFT));
	offsets.push_back(get_offset(SIDE_TOP));
	offsets.push_back(get_offset(SIDE_RIGHT));
	offsets.push_back(get_offset(SIDE_BOTTOM));
	s["offsets"] = offsets;
	return s;
}

void Control::_edit_set_state(const Dictionary &p_state) {
	ERR_FAIL_COND((p_state.size() <= 0) ||
				  !p_state.has("rotation") || !p_state.has("scale") ||
				  !p_state.has("pivot") || !p_state.has("anchors") || !p_state.has("offsets"));
	Dictionary state = p_state;

	set_rotation(state["rotation"]);
	set_scale(state["scale"]);
	set_pivot_offset(state["pivot"]);
	Array anchors = state["anchors"];
	data.anchor[SIDE_LEFT] = anchors[0];
	data.anchor[SIDE_TOP] = anchors[1];
	data.anchor[SIDE_RIGHT] = anchors[2];
	data.anchor[SIDE_BOTTOM] = anchors[3];
	Array offsets = state["offsets"];
	data.offset[SIDE_LEFT] = offsets[0];
	data.offset[SIDE_TOP] = offsets[1];
	data.offset[SIDE_RIGHT] = offsets[2];
	data.offset[SIDE_BOTTOM] = offsets[3];
	_size_changed();
}

void Control::_edit_set_position(const Point2 &p_position) {
#ifdef TOOLS_ENABLED
	ERR_FAIL_COND_MSG(!Engine::get_singleton()->is_editor_hint(), "This function can only be used from editor plugins.");
	set_position(p_position, CanvasItemEditor::get_singleton()->is_anchors_mode_enabled() && Object::cast_to<Control>(data.parent));
#else
	// Unlikely to happen. TODO: enclose all _edit_ functions into TOOLS_ENABLED
	set_position(p_position);
#endif
};

Point2 Control::_edit_get_position() const {
	return get_position();
};

void Control::_edit_set_scale(const Size2 &p_scale) {
	set_scale(p_scale);
}

Size2 Control::_edit_get_scale() const {
	return data.scale;
}

void Control::_edit_set_rect(const Rect2 &p_edit_rect) {
#ifdef TOOLS_ENABLED
	ERR_FAIL_COND_MSG(!Engine::get_singleton()->is_editor_hint(), "This function can only be used from editor plugins.");
	set_position((get_position() + get_transform().basis_xform(p_edit_rect.position)).snapped(Vector2(1, 1)), CanvasItemEditor::get_singleton()->is_anchors_mode_enabled());
	set_size(p_edit_rect.size.snapped(Vector2(1, 1)), CanvasItemEditor::get_singleton()->is_anchors_mode_enabled());
#else
	// Unlikely to happen. TODO: enclose all _edit_ functions into TOOLS_ENABLED
	set_position((get_position() + get_transform().basis_xform(p_edit_rect.position)).snapped(Vector2(1, 1)));
	set_size(p_edit_rect.size.snapped(Vector2(1, 1)));
#endif
}

Rect2 Control::_edit_get_rect() const {
	return Rect2(Point2(), get_size());
}

bool Control::_edit_use_rect() const {
	return true;
}

void Control::_edit_set_rotation(real_t p_rotation) {
	set_rotation(p_rotation);
}

real_t Control::_edit_get_rotation() const {
	return get_rotation();
}

bool Control::_edit_use_rotation() const {
	return true;
}

void Control::_edit_set_pivot(const Point2 &p_pivot) {
	Vector2 delta_pivot = p_pivot - get_pivot_offset();
	Vector2 move = Vector2((cos(data.rotation) - 1.0) * delta_pivot.x - sin(data.rotation) * delta_pivot.y, sin(data.rotation) * delta_pivot.x + (cos(data.rotation) - 1.0) * delta_pivot.y);
	set_position(get_position() + move);
	set_pivot_offset(p_pivot);
}

Point2 Control::_edit_get_pivot() const {
	return get_pivot_offset();
}

bool Control::_edit_use_pivot() const {
	return true;
}

Size2 Control::_edit_get_minimum_size() const {
	return get_combined_minimum_size();
}
#endif

String Control::properties_managed_by_container[] = {
	"offset_left",
	"offset_top",
	"offset_right",
	"offset_bottom",
	"anchor_left",
	"anchor_top",
	"anchor_right",
	"anchor_bottom",
	"rect_position",
	"rect_scale",
	"rect_size"
};

void Control::accept_event() {
	if (is_inside_tree()) {
		get_viewport()->_gui_accept_event();
	}
}

void Control::set_custom_minimum_size(const Size2 &p_custom) {
	if (p_custom == data.custom_minimum_size) {
		return;
	}
	data.custom_minimum_size = p_custom;
	minimum_size_changed();
}

Size2 Control::get_custom_minimum_size() const {
	return data.custom_minimum_size;
}

void Control::_update_minimum_size_cache() {
	Size2 minsize = get_minimum_size();
	minsize.x = MAX(minsize.x, data.custom_minimum_size.x);
	minsize.y = MAX(minsize.y, data.custom_minimum_size.y);

	bool size_changed = false;
	if (data.minimum_size_cache != minsize) {
		size_changed = true;
	}

	data.minimum_size_cache = minsize;
	data.minimum_size_valid = true;

	if (size_changed) {
		minimum_size_changed();
	}
}

Size2 Control::get_combined_minimum_size() const {
	if (!data.minimum_size_valid) {
		const_cast<Control *>(this)->_update_minimum_size_cache();
	}
	return data.minimum_size_cache;
}

Transform2D Control::_get_internal_transform() const {
	Transform2D rot_scale;
	rot_scale.set_rotation_and_scale(data.rotation, data.scale);
	Transform2D offset;
	offset.set_origin(-data.pivot_offset);

	return offset.affine_inverse() * (rot_scale * offset);
}

bool Control::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	// Prefixes "custom_*" are supported for compatibility with 3.x.
	if (!name.begins_with("theme_override") && !name.begins_with("custom")) {
		return false;
	}

	if (p_value.get_type() == Variant::NIL) {
		if (name.begins_with("theme_override_icons/") || name.begins_with("custom_icons/")) {
			String dname = name.get_slicec('/', 1);
			if (data.icon_override.has(dname)) {
				data.icon_override[dname]->disconnect("changed", callable_mp(this, &Control::_override_changed));
			}
			data.icon_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("theme_override_styles/") || name.begins_with("custom_styles/")) {
			String dname = name.get_slicec('/', 1);
			if (data.style_override.has(dname)) {
				data.style_override[dname]->disconnect("changed", callable_mp(this, &Control::_override_changed));
			}
			data.style_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("theme_override_fonts/") || name.begins_with("custom_fonts/")) {
			String dname = name.get_slicec('/', 1);
			if (data.font_override.has(dname)) {
				data.font_override[dname]->disconnect("changed", callable_mp(this, &Control::_override_changed));
			}
			data.font_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("theme_override_font_sizes/") || name.begins_with("custom_font_sizes/")) {
			String dname = name.get_slicec('/', 1);
			data.font_size_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("theme_override_colors/") || name.begins_with("custom_colors/")) {
			String dname = name.get_slicec('/', 1);
			data.color_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("theme_override_constants/") || name.begins_with("custom_constants/")) {
			String dname = name.get_slicec('/', 1);
			data.constant_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
		} else {
			return false;
		}

	} else {
		if (name.begins_with("theme_override_icons/") || name.begins_with("custom_icons/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_icon_override(dname, p_value);
		} else if (name.begins_with("theme_override_styles/") || name.begins_with("custom_styles/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_style_override(dname, p_value);
		} else if (name.begins_with("theme_override_fonts/") || name.begins_with("custom_fonts/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_font_override(dname, p_value);
		} else if (name.begins_with("theme_override_font_sizes/") || name.begins_with("custom_font_sizes/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_font_size_override(dname, p_value);
		} else if (name.begins_with("theme_override_colors/") || name.begins_with("custom_colors/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_color_override(dname, p_value);
		} else if (name.begins_with("theme_override_constants/") || name.begins_with("custom_constants/")) {
			String dname = name.get_slicec('/', 1);
			add_theme_constant_override(dname, p_value);
		} else {
			return false;
		}
	}
	return true;
}

void Control::_update_minimum_size() {
	if (!is_inside_tree()) {
		return;
	}

	Size2 minsize = get_combined_minimum_size();
	data.updating_last_minimum_size = false;

	if (minsize != data.last_minimum_size) {
		data.last_minimum_size = minsize;
		_size_changed();
		emit_signal(SceneStringNames::get_singleton()->minimum_size_changed);
	}
}

bool Control::_get(const StringName &p_name, Variant &r_ret) const {
	String sname = p_name;
	if (!sname.begins_with("theme_override")) {
		return false;
	}

	if (sname.begins_with("theme_override_icons/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.icon_override.has(name) ? Variant(data.icon_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_styles/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.style_override.has(name) ? Variant(data.style_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_fonts/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.font_override.has(name) ? Variant(data.font_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_font_sizes/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.font_size_override.has(name) ? Variant(data.font_size_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_colors/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.color_override.has(name) ? Variant(data.color_override[name]) : Variant();
	} else if (sname.begins_with("theme_override_constants/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.constant_override.has(name) ? Variant(data.constant_override[name]) : Variant();
	} else {
		return false;
	}

	return true;
}

void Control::_get_property_list(List<PropertyInfo> *p_list) const {
	Ref<Theme> theme = Theme::get_default();

	p_list->push_back(PropertyInfo(Variant::NIL, "Theme Overrides", PROPERTY_HINT_NONE, "theme_override_", PROPERTY_USAGE_GROUP));

	{
		List<StringName> names;
		theme->get_color_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.color_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::COLOR, "theme_override_colors/" + E, PROPERTY_HINT_NONE, "", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_constant_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.constant_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::INT, "theme_override_constants/" + E, PROPERTY_HINT_RANGE, "-16384,16384", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_font_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.font_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_fonts/" + E, PROPERTY_HINT_RESOURCE_TYPE, "Font", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_font_size_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.font_size_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::INT, "theme_override_font_sizes/" + E, PROPERTY_HINT_NONE, "", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_icon_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.icon_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_icons/" + E, PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", usage));
		}
	}
	{
		List<StringName> names;
		theme->get_stylebox_list(get_class_name(), &names);
		for (const StringName &E : names) {
			uint32_t usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.style_override.has(E)) {
				usage |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;
			}

			p_list->push_back(PropertyInfo(Variant::OBJECT, "theme_override_styles/" + E, PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", usage));
		}
	}
}

void Control::_validate_property(PropertyInfo &property) const {
	if (property.name == "theme_type_variation") {
		List<StringName> names;

		// Only the default theme and the project theme are used for the list of options.
		// This is an imposed limitation to simplify the logic needed to leverage those options.
		Theme::get_default()->get_type_variation_list(get_class_name(), &names);
		if (Theme::get_project_default().is_valid()) {
			Theme::get_project_default()->get_type_variation_list(get_class_name(), &names);
		}
		names.sort_custom<StringName::AlphCompare>();

		Vector<StringName> unique_names;
		String hint_string;
		for (const StringName &E : names) {
			// Skip duplicate values.
			if (unique_names.has(E)) {
				continue;
			}

			hint_string += String(E) + ",";
			unique_names.append(E);
		}

		property.hint_string = hint_string;
	}
	if (!Object::cast_to<Container>(get_parent())) {
		return;
	}
	// Disable the property if it's managed by the parent container.
	bool property_is_managed_by_container = false;
	for (unsigned i = 0; i < properties_managed_by_container_count; i++) {
		property_is_managed_by_container = properties_managed_by_container[i] == property.name;
		if (property_is_managed_by_container) {
			break;
		}
	}
	if (property_is_managed_by_container) {
		property.usage |= PROPERTY_USAGE_READ_ONLY;
	}
}

Control *Control::get_parent_control() const {
	return data.parent;
}

Window *Control::get_parent_window() const {
	return data.parent_window;
}

void Control::set_layout_direction(Control::LayoutDirection p_direction) {
	ERR_FAIL_INDEX((int)p_direction, 4);

	data.layout_dir = p_direction;
	data.is_rtl_dirty = true;

	propagate_notification(NOTIFICATION_LAYOUT_DIRECTION_CHANGED);
}

Control::LayoutDirection Control::get_layout_direction() const {
	return data.layout_dir;
}

bool Control::is_layout_rtl() const {
	if (data.is_rtl_dirty) {
		const_cast<Control *>(this)->data.is_rtl_dirty = false;
		if (data.layout_dir == LAYOUT_DIRECTION_INHERITED) {
			Window *parent_window = get_parent_window();
			Control *parent_control = get_parent_control();
			if (parent_control) {
				const_cast<Control *>(this)->data.is_rtl = parent_control->is_layout_rtl();
			} else if (parent_window) {
				const_cast<Control *>(this)->data.is_rtl = parent_window->is_layout_rtl();
			} else {
				if (GLOBAL_GET(SNAME("internationalization/rendering/force_right_to_left_layout_direction"))) {
					const_cast<Control *>(this)->data.is_rtl = true;
				} else {
					String locale = TranslationServer::get_singleton()->get_tool_locale();
					const_cast<Control *>(this)->data.is_rtl = TS->is_locale_right_to_left(locale);
				}
			}
		} else if (data.layout_dir == LAYOUT_DIRECTION_LOCALE) {
			if (GLOBAL_GET(SNAME("internationalization/rendering/force_right_to_left_layout_direction"))) {
				const_cast<Control *>(this)->data.is_rtl = true;
			} else {
				String locale = TranslationServer::get_singleton()->get_tool_locale();
				const_cast<Control *>(this)->data.is_rtl = TS->is_locale_right_to_left(locale);
			}
		} else {
			const_cast<Control *>(this)->data.is_rtl = (data.layout_dir == LAYOUT_DIRECTION_RTL);
		}
	}
	return data.is_rtl;
}

void Control::set_auto_translate(bool p_enable) {
	if (p_enable == data.auto_translate) {
		return;
	}

	data.auto_translate = p_enable;

	notification(MainLoop::NOTIFICATION_TRANSLATION_CHANGED);
}

bool Control::is_auto_translating() const {
	return data.auto_translate;
}

void Control::_clear_size_warning() {
	data.size_warning = false;
}

//moved theme configuration here, so controls can set up even if still not inside active scene

void Control::add_child_notify(Node *p_child) {
	Control *child_c = Object::cast_to<Control>(p_child);

	if (child_c && child_c->data.theme.is_null() && (data.theme_owner || data.theme_owner_window)) {
		_propagate_theme_changed(child_c, data.theme_owner, data.theme_owner_window); //need to propagate here, since many controls may require setting up stuff
	}

	Window *child_w = Object::cast_to<Window>(p_child);

	if (child_w && child_w->theme.is_null() && (data.theme_owner || data.theme_owner_window)) {
		_propagate_theme_changed(child_w, data.theme_owner, data.theme_owner_window); //need to propagate here, since many controls may require setting up stuff
	}
}

void Control::remove_child_notify(Node *p_child) {
	Control *child_c = Object::cast_to<Control>(p_child);

	if (child_c && (child_c->data.theme_owner || child_c->data.theme_owner_window) && child_c->data.theme.is_null()) {
		_propagate_theme_changed(child_c, nullptr, nullptr);
	}

	Window *child_w = Object::cast_to<Window>(p_child);

	if (child_w && (child_w->theme_owner || child_w->theme_owner_window) && child_w->theme.is_null()) {
		_propagate_theme_changed(child_w, nullptr, nullptr);
	}
}

void Control::_update_canvas_item_transform() {
	Transform2D xform = _get_internal_transform();
	xform[2] += get_position();

	RenderingServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), xform);
}

void Control::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			data.minimum_size_valid = false;
			data.is_rtl_dirty = true;
			_size_changed();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			get_viewport()->_gui_remove_control(this);
		} break;
		case NOTIFICATION_READY: {
#ifdef DEBUG_ENABLED
			connect("ready", callable_mp(this, &Control::_clear_size_warning), varray(), CONNECT_DEFERRED | CONNECT_ONESHOT);
#endif
		} break;

		case NOTIFICATION_ENTER_CANVAS: {
			data.parent = Object::cast_to<Control>(get_parent());
			data.parent_window = Object::cast_to<Window>(get_parent());
			data.is_rtl_dirty = true;

			Node *parent = this; //meh
			Control *parent_control = nullptr;
			bool subwindow = false;

			while (parent) {
				parent = parent->get_parent();

				if (!parent) {
					break;
				}

				CanvasItem *ci = Object::cast_to<CanvasItem>(parent);
				if (ci && ci->is_set_as_top_level()) {
					subwindow = true;
					break;
				}

				parent_control = Object::cast_to<Control>(parent);

				if (parent_control) {
					break;
				} else if (ci) {
				} else {
					break;
				}
			}

			if (parent_control && !subwindow) {
				//do nothing, has a parent control and not top_level
				if (data.theme.is_null() && parent_control->data.theme_owner) {
					data.theme_owner = parent_control->data.theme_owner;
					notification(NOTIFICATION_THEME_CHANGED);
				}
			} else {
				//is a regular root control or top_level
				data.RI = get_viewport()->_gui_add_root_control(this);
			}

			data.parent_canvas_item = get_parent_item();

			if (data.parent_canvas_item) {
				data.parent_canvas_item->connect("item_rect_changed", callable_mp(this, &Control::_size_changed));
			} else {
				//connect viewport
				get_viewport()->connect("size_changed", callable_mp(this, &Control::_size_changed));
			}
		} break;
		case NOTIFICATION_EXIT_CANVAS: {
			if (data.parent_canvas_item) {
				data.parent_canvas_item->disconnect("item_rect_changed", callable_mp(this, &Control::_size_changed));
				data.parent_canvas_item = nullptr;
			} else if (!is_set_as_top_level()) {
				//disconnect viewport
				get_viewport()->disconnect("size_changed", callable_mp(this, &Control::_size_changed));
			}

			if (data.RI) {
				get_viewport()->_gui_remove_root_control(data.RI);
				data.RI = nullptr;
			}

			data.parent = nullptr;
			data.parent_canvas_item = nullptr;
			data.parent_window = nullptr;
			data.is_rtl_dirty = true;

		} break;
		case NOTIFICATION_MOVED_IN_PARENT: {
			// some parents need to know the order of the children to draw (like TabContainer)
			// update if necessary
			if (data.parent) {
				data.parent->update();
			}
			update();

			if (data.RI) {
				get_viewport()->_gui_set_root_order_dirty();
			}

		} break;
		case NOTIFICATION_RESIZED: {
			emit_signal(SceneStringNames::get_singleton()->resized);
		} break;
		case NOTIFICATION_DRAW: {
			_update_canvas_item_transform();
			RenderingServer::get_singleton()->canvas_item_set_custom_rect(get_canvas_item(), !data.disable_visibility_clip, Rect2(Point2(), get_size()));
			RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), data.clip_contents);
			//emit_signal(SceneStringNames::get_singleton()->draw);

		} break;
		case NOTIFICATION_MOUSE_ENTER: {
			emit_signal(SceneStringNames::get_singleton()->mouse_entered);
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			emit_signal(SceneStringNames::get_singleton()->mouse_exited);
		} break;
		case NOTIFICATION_FOCUS_ENTER: {
			emit_signal(SceneStringNames::get_singleton()->focus_entered);
			update();
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			emit_signal(SceneStringNames::get_singleton()->focus_exited);
			update();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			minimum_size_changed();
			update();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				if (get_viewport() != nullptr) {
					get_viewport()->_gui_hide_control(this);
				}

				//remove key focus

			} else {
				data.minimum_size_valid = false;
				_size_changed();
			}

		} break;
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			data.is_rtl_dirty = true;
			_size_changed();
		} break;
	}
}

bool Control::has_point(const Point2 &p_point) const {
	bool ret;
	if (GDVIRTUAL_CALL(_has_point, p_point, ret)) {
		return ret;
	}
	return Rect2(Point2(), get_size()).has_point(p_point);
}

void Control::set_drag_forwarding(Object *p_target) {
	if (p_target) {
		data.drag_owner = p_target->get_instance_id();
	} else {
		data.drag_owner = ObjectID();
	}
}

Variant Control::get_drag_data(const Point2 &p_point) {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			return obj->call("_get_drag_data_fw", p_point, this);
		}
	}

	Variant dd;
	if (GDVIRTUAL_CALL(_get_drag_data, p_point, dd)) {
		return dd;
	}

	return Variant();
}

bool Control::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			return obj->call("_can_drop_data_fw", p_point, p_data, this);
		}
	}

	bool ret;
	if (GDVIRTUAL_CALL(_can_drop_data, p_point, p_data, ret)) {
		return ret;
	}
	return false;
}

void Control::drop_data(const Point2 &p_point, const Variant &p_data) {
	if (data.drag_owner.is_valid()) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			obj->call("_drop_data_fw", p_point, p_data, this);
			return;
		}
	}

	GDVIRTUAL_CALL(_drop_data, p_point, p_data);
}

void Control::force_drag(const Variant &p_data, Control *p_control) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(p_data.get_type() == Variant::NIL);

	get_viewport()->_gui_force_drag(this, p_data, p_control);
}

void Control::set_drag_preview(Control *p_control) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!get_viewport()->gui_is_dragging());
	get_viewport()->_gui_set_drag_preview(this, p_control);
}

void Control::_call_gui_input(const Ref<InputEvent> &p_event) {
	emit_signal(SceneStringNames::get_singleton()->gui_input, p_event); //signal should be first, so it's possible to override an event (and then accept it)
	if (!is_inside_tree() || get_viewport()->is_input_handled()) {
		return; //input was handled, abort
	}
	GDVIRTUAL_CALL(_gui_input, p_event);
	if (!is_inside_tree() || get_viewport()->is_input_handled()) {
		return; //input was handled, abort
	}
	gui_input(p_event);
}
void Control::gui_input(const Ref<InputEvent> &p_event) {
}

Size2 Control::get_minimum_size() const {
	Vector2 ms;
	if (GDVIRTUAL_CALL(_get_minimum_size, ms)) {
		return ms;
	}
	return Vector2();
}

template <class T>
T Control::get_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types) {
	ERR_FAIL_COND_V_MSG(p_theme_types.size() == 0, T(), "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			if (theme_owner && theme_owner->data.theme->has_theme_item(p_data_type, p_name, E)) {
				return theme_owner->data.theme->get_theme_item(p_data_type, p_name, E);
			}

			if (theme_owner_window && theme_owner_window->theme->has_theme_item(p_data_type, p_name, E)) {
				return theme_owner_window->theme->get_theme_item(p_data_type, p_name, E);
			}
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		for (const StringName &E : p_theme_types) {
			if (Theme::get_project_default()->has_theme_item(p_data_type, p_name, E)) {
				return Theme::get_project_default()->get_theme_item(p_data_type, p_name, E);
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	for (const StringName &E : p_theme_types) {
		if (Theme::get_default()->has_theme_item(p_data_type, p_name, E)) {
			return Theme::get_default()->get_theme_item(p_data_type, p_name, E);
		}
	}
	// If they don't exist, use any type to return the default/empty value.
	return Theme::get_default()->get_theme_item(p_data_type, p_name, p_theme_types[0]);
}

bool Control::has_theme_item_in_types(Control *p_theme_owner, Window *p_theme_owner_window, Theme::DataType p_data_type, const StringName &p_name, List<StringName> p_theme_types) {
	ERR_FAIL_COND_V_MSG(p_theme_types.size() == 0, false, "At least one theme type must be specified.");

	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		// For each theme resource check the theme types provided and see if p_name exists with any of them.
		for (const StringName &E : p_theme_types) {
			if (theme_owner && theme_owner->data.theme->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}

			if (theme_owner_window && theme_owner_window->theme->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		for (const StringName &E : p_theme_types) {
			if (Theme::get_project_default()->has_theme_item(p_data_type, p_name, E)) {
				return true;
			}
		}
	}

	// Lastly, fall back on the items defined in the default Theme, if they exist.
	for (const StringName &E : p_theme_types) {
		if (Theme::get_default()->has_theme_item(p_data_type, p_name, E)) {
			return true;
		}
	}
	return false;
}

void Control::_get_theme_type_dependencies(const StringName &p_theme_type, List<StringName> *p_list) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (Theme::get_project_default().is_valid() && Theme::get_project_default()->get_type_variation_base(data.theme_type_variation) != StringName()) {
			Theme::get_project_default()->get_type_dependencies(get_class_name(), data.theme_type_variation, p_list);
		} else {
			Theme::get_default()->get_type_dependencies(get_class_name(), data.theme_type_variation, p_list);
		}
	} else {
		Theme::get_default()->get_type_dependencies(p_theme_type, StringName(), p_list);
	}
}

Ref<Texture2D> Control::get_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<Texture2D> *tex = data.icon_override.getptr(p_name);
		if (tex) {
			return *tex;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<Ref<Texture2D>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
}

Ref<StyleBox> Control::get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<StyleBox> *style = data.style_override.getptr(p_name);
		if (style) {
			return *style;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<Ref<StyleBox>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
}

Ref<Font> Control::get_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Ref<Font> *font = data.font_override.getptr(p_name);
		if (font) {
			return *font;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<Ref<Font>>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
}

int Control::get_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const int *font_size = data.font_size_override.getptr(p_name);
		if (font_size) {
			return *font_size;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<int>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
}

Color Control::get_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const Color *color = data.color_override.getptr(p_name);
		if (color) {
			return *color;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<Color>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
}

int Control::get_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		const int *constant = data.constant_override.getptr(p_name);
		if (constant) {
			return *constant;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return get_theme_item_in_types<int>(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
}

bool Control::has_theme_icon_override(const StringName &p_name) const {
	const Ref<Texture2D> *tex = data.icon_override.getptr(p_name);
	return tex != nullptr;
}

bool Control::has_theme_stylebox_override(const StringName &p_name) const {
	const Ref<StyleBox> *style = data.style_override.getptr(p_name);
	return style != nullptr;
}

bool Control::has_theme_font_override(const StringName &p_name) const {
	const Ref<Font> *font = data.font_override.getptr(p_name);
	return font != nullptr;
}

bool Control::has_theme_font_size_override(const StringName &p_name) const {
	const int *font_size = data.font_size_override.getptr(p_name);
	return font_size != nullptr;
}

bool Control::has_theme_color_override(const StringName &p_name) const {
	const Color *color = data.color_override.getptr(p_name);
	return color != nullptr;
}

bool Control::has_theme_constant_override(const StringName &p_name) const {
	const int *constant = data.constant_override.getptr(p_name);
	return constant != nullptr;
}

bool Control::has_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_icon_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
}

bool Control::has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_stylebox_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
}

bool Control::has_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_font_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
}

bool Control::has_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_font_size_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
}

bool Control::has_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_color_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
}

bool Control::has_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == data.theme_type_variation) {
		if (has_theme_constant_override(p_name)) {
			return true;
		}
	}

	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return has_theme_item_in_types(data.theme_owner, data.theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
}

float Control::fetch_theme_default_base_scale(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_theme_base_scale()) {
			return theme_owner->data.theme->get_default_theme_base_scale();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_theme_base_scale()) {
			return theme_owner_window->theme->get_default_theme_base_scale();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		if (Theme::get_project_default()->has_default_theme_base_scale()) {
			return Theme::get_project_default()->get_default_theme_base_scale();
		}
	}

	// Lastly, fall back on the default Theme.
	return Theme::get_default()->get_default_theme_base_scale();
}

float Control::get_theme_default_base_scale() const {
	return fetch_theme_default_base_scale(data.theme_owner, data.theme_owner_window);
}

Ref<Font> Control::fetch_theme_default_font(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_theme_font()) {
			return theme_owner->data.theme->get_default_theme_font();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_theme_font()) {
			return theme_owner_window->theme->get_default_theme_font();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		if (Theme::get_project_default()->has_default_theme_font()) {
			return Theme::get_project_default()->get_default_theme_font();
		}
	}

	// Lastly, fall back on the default Theme.
	return Theme::get_default()->get_default_theme_font();
}

Ref<Font> Control::get_theme_default_font() const {
	return fetch_theme_default_font(data.theme_owner, data.theme_owner_window);
}

int Control::fetch_theme_default_font_size(Control *p_theme_owner, Window *p_theme_owner_window) {
	// First, look through each control or window node in the branch, until no valid parent can be found.
	// Only nodes with a theme resource attached are considered.
	// For each theme resource see if their assigned theme has the default value defined and valid.
	Control *theme_owner = p_theme_owner;
	Window *theme_owner_window = p_theme_owner_window;

	while (theme_owner || theme_owner_window) {
		if (theme_owner && theme_owner->data.theme->has_default_theme_font_size()) {
			return theme_owner->data.theme->get_default_theme_font_size();
		}

		if (theme_owner_window && theme_owner_window->theme->has_default_theme_font_size()) {
			return theme_owner_window->theme->get_default_theme_font_size();
		}

		Node *parent = theme_owner ? theme_owner->get_parent() : theme_owner_window->get_parent();
		Control *parent_c = Object::cast_to<Control>(parent);
		if (parent_c) {
			theme_owner = parent_c->data.theme_owner;
			theme_owner_window = parent_c->data.theme_owner_window;
		} else {
			Window *parent_w = Object::cast_to<Window>(parent);
			if (parent_w) {
				theme_owner = parent_w->theme_owner;
				theme_owner_window = parent_w->theme_owner_window;
			} else {
				theme_owner = nullptr;
				theme_owner_window = nullptr;
			}
		}
	}

	// Secondly, check the project-defined Theme resource.
	if (Theme::get_project_default().is_valid()) {
		if (Theme::get_project_default()->has_default_theme_font_size()) {
			return Theme::get_project_default()->get_default_theme_font_size();
		}
	}

	// Lastly, fall back on the default Theme.
	return Theme::get_default()->get_default_theme_font_size();
}

int Control::get_theme_default_font_size() const {
	return fetch_theme_default_font_size(data.theme_owner, data.theme_owner_window);
}

Rect2 Control::get_parent_anchorable_rect() const {
	if (!is_inside_tree()) {
		return Rect2();
	}

	Rect2 parent_rect;
	if (data.parent_canvas_item) {
		parent_rect = data.parent_canvas_item->get_anchorable_rect();
	} else {
#ifdef TOOLS_ENABLED
		Node *edited_root = get_tree()->get_edited_scene_root();
		if (edited_root && (this == edited_root || edited_root->is_ancestor_of(this))) {
			parent_rect.size = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
		} else {
			parent_rect = get_viewport()->get_visible_rect();
		}

#else
		parent_rect = get_viewport()->get_visible_rect();
#endif
	}

	return parent_rect;
}

Size2 Control::get_parent_area_size() const {
	return get_parent_anchorable_rect().size;
}

void Control::_size_changed() {
	Rect2 parent_rect = get_parent_anchorable_rect();

	real_t edge_pos[4];

	for (int i = 0; i < 4; i++) {
		real_t area = parent_rect.size[i & 1];
		edge_pos[i] = data.offset[i] + (data.anchor[i] * area);
	}

	Point2 new_pos_cache = Point2(edge_pos[0], edge_pos[1]);
	Size2 new_size_cache = Point2(edge_pos[2], edge_pos[3]) - new_pos_cache;

	Size2 minimum_size = get_combined_minimum_size();

	if (minimum_size.width > new_size_cache.width) {
		if (data.h_grow == GROW_DIRECTION_BEGIN) {
			new_pos_cache.x += new_size_cache.width - minimum_size.width;
		} else if (data.h_grow == GROW_DIRECTION_BOTH) {
			new_pos_cache.x += 0.5 * (new_size_cache.width - minimum_size.width);
		}

		new_size_cache.width = minimum_size.width;
	}

	if (is_layout_rtl()) {
		new_pos_cache.x = parent_rect.size.x - new_pos_cache.x - new_size_cache.x;
	}

	if (minimum_size.height > new_size_cache.height) {
		if (data.v_grow == GROW_DIRECTION_BEGIN) {
			new_pos_cache.y += new_size_cache.height - minimum_size.height;
		} else if (data.v_grow == GROW_DIRECTION_BOTH) {
			new_pos_cache.y += 0.5 * (new_size_cache.height - minimum_size.height);
		}

		new_size_cache.height = minimum_size.height;
	}

	bool pos_changed = new_pos_cache != data.pos_cache;
	bool size_changed = new_size_cache != data.size_cache;

	data.pos_cache = new_pos_cache;
	data.size_cache = new_size_cache;

	if (is_inside_tree()) {
		if (size_changed) {
			notification(NOTIFICATION_RESIZED);
		}
		if (pos_changed || size_changed) {
			item_rect_changed(size_changed);
			_notify_transform();
		}

		if (pos_changed && !size_changed) {
			_update_canvas_item_transform(); //move because it won't be updated
		}
	}
}

void Control::set_anchor(Side p_side, real_t p_anchor, bool p_keep_offset, bool p_push_opposite_anchor) {
	ERR_FAIL_INDEX((int)p_side, 4);

	Rect2 parent_rect = get_parent_anchorable_rect();
	real_t parent_range = (p_side == SIDE_LEFT || p_side == SIDE_RIGHT) ? parent_rect.size.x : parent_rect.size.y;
	real_t previous_pos = data.offset[p_side] + data.anchor[p_side] * parent_range;
	real_t previous_opposite_pos = data.offset[(p_side + 2) % 4] + data.anchor[(p_side + 2) % 4] * parent_range;

	data.anchor[p_side] = p_anchor;

	if (((p_side == SIDE_LEFT || p_side == SIDE_TOP) && data.anchor[p_side] > data.anchor[(p_side + 2) % 4]) ||
			((p_side == SIDE_RIGHT || p_side == SIDE_BOTTOM) && data.anchor[p_side] < data.anchor[(p_side + 2) % 4])) {
		if (p_push_opposite_anchor) {
			data.anchor[(p_side + 2) % 4] = data.anchor[p_side];
		} else {
			data.anchor[p_side] = data.anchor[(p_side + 2) % 4];
		}
	}

	if (!p_keep_offset) {
		data.offset[p_side] = previous_pos - data.anchor[p_side] * parent_range;
		if (p_push_opposite_anchor) {
			data.offset[(p_side + 2) % 4] = previous_opposite_pos - data.anchor[(p_side + 2) % 4] * parent_range;
		}
	}
	if (is_inside_tree()) {
		_size_changed();
	}

	update();
}

void Control::_set_anchor(Side p_side, real_t p_anchor) {
	set_anchor(p_side, p_anchor);
}

void Control::set_anchor_and_offset(Side p_side, real_t p_anchor, real_t p_pos, bool p_push_opposite_anchor) {
	set_anchor(p_side, p_anchor, false, p_push_opposite_anchor);
	set_offset(p_side, p_pos);
}

void Control::set_anchors_preset(LayoutPreset p_preset, bool p_keep_offsets) {
	ERR_FAIL_INDEX((int)p_preset, 16);

	//Left
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_TOP_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_LEFT_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(SIDE_LEFT, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(SIDE_LEFT, 0.5, p_keep_offsets);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			set_anchor(SIDE_LEFT, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Top
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_TOP_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(SIDE_TOP, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(SIDE_TOP, 0.5, p_keep_offsets);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			set_anchor(SIDE_TOP, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			set_anchor(SIDE_RIGHT, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(SIDE_RIGHT, 0.5, p_keep_offsets);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(SIDE_RIGHT, ANCHOR_END, p_keep_offsets);
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			set_anchor(SIDE_BOTTOM, ANCHOR_BEGIN, p_keep_offsets);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(SIDE_BOTTOM, 0.5, p_keep_offsets);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(SIDE_BOTTOM, ANCHOR_END, p_keep_offsets);
			break;
	}
}

void Control::set_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	ERR_FAIL_INDEX((int)p_preset, 16);
	ERR_FAIL_INDEX((int)p_resize_mode, 4);

	// Calculate the size if the node is not resized
	Size2 min_size = get_minimum_size();
	Size2 new_size = get_size();
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_HEIGHT) {
		new_size.x = min_size.x;
	}
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_WIDTH) {
		new_size.y = min_size.y;
	}

	Rect2 parent_rect = get_parent_anchorable_rect();

	real_t x = parent_rect.size.x;
	if (is_layout_rtl()) {
		x = parent_rect.size.x - x - new_size.x;
	}
	//Left
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_TOP_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_LEFT_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			data.offset[0] = x * (0.0 - data.anchor[0]) + p_margin + parent_rect.position.x;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.offset[0] = x * (0.5 - data.anchor[0]) - new_size.x / 2 + parent_rect.position.x;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			data.offset[0] = x * (1.0 - data.anchor[0]) - new_size.x - p_margin + parent_rect.position.x;
			break;
	}

	// Top
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_TOP_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			data.offset[1] = parent_rect.size.y * (0.0 - data.anchor[1]) + p_margin + parent_rect.position.y;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.offset[1] = parent_rect.size.y * (0.5 - data.anchor[1]) - new_size.y / 2 + parent_rect.position.y;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			data.offset[1] = parent_rect.size.y * (1.0 - data.anchor[1]) - new_size.y - p_margin + parent_rect.position.y;
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			data.offset[2] = x * (0.0 - data.anchor[2]) + new_size.x + p_margin + parent_rect.position.x;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.offset[2] = x * (0.5 - data.anchor[2]) + new_size.x / 2 + parent_rect.position.x;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			data.offset[2] = x * (1.0 - data.anchor[2]) - p_margin + parent_rect.position.x;
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			data.offset[3] = parent_rect.size.y * (0.0 - data.anchor[3]) + new_size.y + p_margin + parent_rect.position.y;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.offset[3] = parent_rect.size.y * (0.5 - data.anchor[3]) + new_size.y / 2 + parent_rect.position.y;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			data.offset[3] = parent_rect.size.y * (1.0 - data.anchor[3]) - p_margin + parent_rect.position.y;
			break;
	}

	_size_changed();
}

void Control::set_anchors_and_offsets_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	set_anchors_preset(p_preset);
	set_offsets_preset(p_preset, p_resize_mode, p_margin);
}

real_t Control::get_anchor(Side p_side) const {
	ERR_FAIL_INDEX_V(int(p_side), 4, 0.0);

	return data.anchor[p_side];
}

void Control::set_offset(Side p_side, real_t p_value) {
	ERR_FAIL_INDEX((int)p_side, 4);

	data.offset[p_side] = p_value;
	_size_changed();
}

void Control::set_begin(const Size2 &p_point) {
	data.offset[0] = p_point.x;
	data.offset[1] = p_point.y;
	_size_changed();
}

void Control::set_end(const Size2 &p_point) {
	data.offset[2] = p_point.x;
	data.offset[3] = p_point.y;
	_size_changed();
}

real_t Control::get_offset(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, 0);

	return data.offset[p_side];
}

Size2 Control::get_begin() const {
	return Size2(data.offset[0], data.offset[1]);
}

Size2 Control::get_end() const {
	return Size2(data.offset[2], data.offset[3]);
}

Point2 Control::get_global_position() const {
	return get_global_transform().get_origin();
}

Point2 Control::get_screen_position() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Point2());
	Point2 global_pos = get_viewport()->get_canvas_transform().xform(get_global_position());
	Window *w = Object::cast_to<Window>(get_viewport());
	if (w && !w->is_embedding_subwindows()) {
		global_pos += w->get_position();
	}

	return global_pos;
}

void Control::_set_global_position(const Point2 &p_point) {
	set_global_position(p_point);
}

void Control::set_global_position(const Point2 &p_point, bool p_keep_offsets) {
	Transform2D inv;

	if (data.parent_canvas_item) {
		inv = data.parent_canvas_item->get_global_transform().affine_inverse();
	}

	set_position(inv.xform(p_point), p_keep_offsets);
}

void Control::_compute_anchors(Rect2 p_rect, const real_t p_offsets[4], real_t (&r_anchors)[4]) {
	Size2 parent_rect_size = get_parent_anchorable_rect().size;
	ERR_FAIL_COND(parent_rect_size.x == 0.0);
	ERR_FAIL_COND(parent_rect_size.y == 0.0);

	real_t x = p_rect.position.x;
	if (is_layout_rtl()) {
		x = parent_rect_size.x - x - p_rect.size.x;
	}
	r_anchors[0] = (x - p_offsets[0]) / parent_rect_size.x;
	r_anchors[1] = (p_rect.position.y - p_offsets[1]) / parent_rect_size.y;
	r_anchors[2] = (x + p_rect.size.x - p_offsets[2]) / parent_rect_size.x;
	r_anchors[3] = (p_rect.position.y + p_rect.size.y - p_offsets[3]) / parent_rect_size.y;
}

void Control::_compute_offsets(Rect2 p_rect, const real_t p_anchors[4], real_t (&r_offsets)[4]) {
	Size2 parent_rect_size = get_parent_anchorable_rect().size;

	real_t x = p_rect.position.x;
	if (is_layout_rtl()) {
		x = parent_rect_size.x - x - p_rect.size.x;
	}
	r_offsets[0] = x - (p_anchors[0] * parent_rect_size.x);
	r_offsets[1] = p_rect.position.y - (p_anchors[1] * parent_rect_size.y);
	r_offsets[2] = x + p_rect.size.x - (p_anchors[2] * parent_rect_size.x);
	r_offsets[3] = p_rect.position.y + p_rect.size.y - (p_anchors[3] * parent_rect_size.y);
}

void Control::_set_position(const Size2 &p_point) {
	set_position(p_point);
}

void Control::set_position(const Size2 &p_point, bool p_keep_offsets) {
	if (p_keep_offsets) {
		_compute_anchors(Rect2(p_point, data.size_cache), data.offset, data.anchor);
	} else {
		_compute_offsets(Rect2(p_point, data.size_cache), data.anchor, data.offset);
	}
	_size_changed();
}

void Control::set_rect(const Rect2 &p_rect) {
	for (int i = 0; i < 4; i++) {
		data.anchor[i] = ANCHOR_BEGIN;
	}

	_compute_offsets(p_rect, data.anchor, data.offset);
	if (is_inside_tree()) {
		_size_changed();
	}
}

void Control::_set_size(const Size2 &p_size) {
#ifdef DEBUG_ENABLED
	if (data.size_warning && (data.anchor[SIDE_LEFT] != data.anchor[SIDE_RIGHT] || data.anchor[SIDE_TOP] != data.anchor[SIDE_BOTTOM])) {
		WARN_PRINT("Nodes with non-equal opposite anchors will have their size overridden after _ready(). \nIf you want to set size, change the anchors or consider using set_deferred().");
	}
#endif
	set_size(p_size);
}

void Control::set_size(const Size2 &p_size, bool p_keep_offsets) {
	Size2 new_size = p_size;
	Size2 min = get_combined_minimum_size();
	if (new_size.x < min.x) {
		new_size.x = min.x;
	}
	if (new_size.y < min.y) {
		new_size.y = min.y;
	}

	if (p_keep_offsets) {
		_compute_anchors(Rect2(data.pos_cache, new_size), data.offset, data.anchor);
	} else {
		_compute_offsets(Rect2(data.pos_cache, new_size), data.anchor, data.offset);
	}
	_size_changed();
}

Size2 Control::get_position() const {
	return data.pos_cache;
}

Size2 Control::get_size() const {
	return data.size_cache;
}

Rect2 Control::get_global_rect() const {
	return Rect2(get_global_position(), get_size());
}

Rect2 Control::get_screen_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());

	Rect2 r(get_global_position(), get_size());

	Window *w = Object::cast_to<Window>(get_viewport());
	if (w && !w->is_embedding_subwindows()) {
		r.position += w->get_position();
	}

	return r;
}

Rect2 Control::get_window_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2 gr = get_global_rect();
	gr.position += get_viewport()->get_visible_rect().position;
	return gr;
}

Rect2 Control::get_rect() const {
	return Rect2(get_position(), get_size());
}

Rect2 Control::get_anchorable_rect() const {
	return Rect2(Point2(), get_size());
}

void Control::begin_bulk_theme_override() {
	data.bulk_theme_override = true;
}

void Control::end_bulk_theme_override() {
	ERR_FAIL_COND(!data.bulk_theme_override);

	data.bulk_theme_override = false;
	_notify_theme_changed();
}

void Control::add_theme_icon_override(const StringName &p_name, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_COND(!p_icon.is_valid());

	if (data.icon_override.has(p_name)) {
		data.icon_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.icon_override[p_name] = p_icon;
	data.icon_override[p_name]->connect("changed", callable_mp(this, &Control::_override_changed), Vector<Variant>(), CONNECT_REFERENCE_COUNTED);
	_notify_theme_changed();
}

void Control::add_theme_style_override(const StringName &p_name, const Ref<StyleBox> &p_style) {
	ERR_FAIL_COND(!p_style.is_valid());

	if (data.style_override.has(p_name)) {
		data.style_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.style_override[p_name] = p_style;
	data.style_override[p_name]->connect("changed", callable_mp(this, &Control::_override_changed), Vector<Variant>(), CONNECT_REFERENCE_COUNTED);
	_notify_theme_changed();
}

void Control::add_theme_font_override(const StringName &p_name, const Ref<Font> &p_font) {
	ERR_FAIL_COND(!p_font.is_valid());

	if (data.font_override.has(p_name)) {
		data.font_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.font_override[p_name] = p_font;
	data.font_override[p_name]->connect("changed", callable_mp(this, &Control::_override_changed), Vector<Variant>(), CONNECT_REFERENCE_COUNTED);
	_notify_theme_changed();
}

void Control::add_theme_font_size_override(const StringName &p_name, int p_font_size) {
	data.font_size_override[p_name] = p_font_size;
	_notify_theme_changed();
}

void Control::add_theme_color_override(const StringName &p_name, const Color &p_color) {
	data.color_override[p_name] = p_color;
	_notify_theme_changed();
}

void Control::add_theme_constant_override(const StringName &p_name, int p_constant) {
	data.constant_override[p_name] = p_constant;
	_notify_theme_changed();
}

void Control::remove_theme_icon_override(const StringName &p_name) {
	if (data.icon_override.has(p_name)) {
		data.icon_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.icon_override.erase(p_name);
	_notify_theme_changed();
}

void Control::remove_theme_style_override(const StringName &p_name) {
	if (data.style_override.has(p_name)) {
		data.style_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.style_override.erase(p_name);
	_notify_theme_changed();
}

void Control::remove_theme_font_override(const StringName &p_name) {
	if (data.font_override.has(p_name)) {
		data.font_override[p_name]->disconnect("changed", callable_mp(this, &Control::_override_changed));
	}

	data.font_override.erase(p_name);
	_notify_theme_changed();
}

void Control::remove_theme_font_size_override(const StringName &p_name) {
	data.font_size_override.erase(p_name);
	_notify_theme_changed();
}

void Control::remove_theme_color_override(const StringName &p_name) {
	data.color_override.erase(p_name);
	_notify_theme_changed();
}

void Control::remove_theme_constant_override(const StringName &p_name) {
	data.constant_override.erase(p_name);
	_notify_theme_changed();
}

void Control::set_focus_mode(FocusMode p_focus_mode) {
	ERR_FAIL_INDEX((int)p_focus_mode, 3);

	if (is_inside_tree() && p_focus_mode == FOCUS_NONE && data.focus_mode != FOCUS_NONE && has_focus()) {
		release_focus();
	}

	data.focus_mode = p_focus_mode;
}

static Control *_next_control(Control *p_from) {
	if (p_from->is_set_as_top_level()) {
		return nullptr; // can't go above
	}

	Control *parent = Object::cast_to<Control>(p_from->get_parent());

	if (!parent) {
		return nullptr;
	}

	int next = p_from->get_index();
	ERR_FAIL_INDEX_V(next, parent->get_child_count(), nullptr);
	for (int i = (next + 1); i < parent->get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(parent->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
			continue;
		}

		return c;
	}

	//no next in parent, try the same in parent
	return _next_control(parent);
}

Control *Control::find_next_valid_focus() const {
	Control *from = const_cast<Control *>(this);

	while (true) {
		// If the focus property is manually overwritten, attempt to use it.

		if (!data.focus_next.is_empty()) {
			Node *n = get_node(data.focus_next);
			Control *c;
			if (n) {
				c = Object::cast_to<Control>(n);
				ERR_FAIL_COND_V_MSG(!c, nullptr, "Next focus node is not a control: " + n->get_name() + ".");
			} else {
				return nullptr;
			}
			if (c->is_visible() && c->get_focus_mode() != FOCUS_NONE) {
				return c;
			}
		}

		// find next child

		Control *next_child = nullptr;

		for (int i = 0; i < from->get_child_count(); i++) {
			Control *c = Object::cast_to<Control>(from->get_child(i));
			if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
				continue;
			}

			next_child = c;
			break;
		}

		if (!next_child) {
			next_child = _next_control(from);
			if (!next_child) { //nothing else.. go up and find either window or subwindow
				next_child = const_cast<Control *>(this);
				while (next_child && !next_child->is_set_as_top_level()) {
					next_child = cast_to<Control>(next_child->get_parent());
				}

				if (!next_child) {
					next_child = const_cast<Control *>(this);
					while (next_child) {
						if (next_child->data.RI) {
							break;
						}
						next_child = next_child->get_parent_control();
					}
				}
			}
		}

		if (next_child == this) { // no next control->
			return (get_focus_mode() == FOCUS_ALL) ? next_child : nullptr;
		}
		if (next_child) {
			if (next_child->get_focus_mode() == FOCUS_ALL) {
				return next_child;
			}
			from = next_child;
		} else {
			break;
		}
	}

	return nullptr;
}

static Control *_prev_control(Control *p_from) {
	Control *child = nullptr;
	for (int i = p_from->get_child_count() - 1; i >= 0; i--) {
		Control *c = Object::cast_to<Control>(p_from->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
			continue;
		}

		child = c;
		break;
	}

	if (!child) {
		return p_from;
	}

	//no prev in parent, try the same in parent
	return _prev_control(child);
}

Control *Control::find_prev_valid_focus() const {
	Control *from = const_cast<Control *>(this);

	while (true) {
		// If the focus property is manually overwritten, attempt to use it.

		if (!data.focus_prev.is_empty()) {
			Node *n = get_node(data.focus_prev);
			Control *c;
			if (n) {
				c = Object::cast_to<Control>(n);
				ERR_FAIL_COND_V_MSG(!c, nullptr, "Previous focus node is not a control: " + n->get_name() + ".");
			} else {
				return nullptr;
			}
			if (c->is_visible() && c->get_focus_mode() != FOCUS_NONE) {
				return c;
			}
		}

		// find prev child

		Control *prev_child = nullptr;

		if (from->is_set_as_top_level() || !Object::cast_to<Control>(from->get_parent())) {
			//find last of the children

			prev_child = _prev_control(from);

		} else {
			for (int i = (from->get_index() - 1); i >= 0; i--) {
				Control *c = Object::cast_to<Control>(from->get_parent()->get_child(i));

				if (!c || !c->is_visible_in_tree() || c->is_set_as_top_level()) {
					continue;
				}

				prev_child = c;
				break;
			}

			if (!prev_child) {
				prev_child = Object::cast_to<Control>(from->get_parent());
			} else {
				prev_child = _prev_control(prev_child);
			}
		}

		if (prev_child == this) { // no prev control->
			return (get_focus_mode() == FOCUS_ALL) ? prev_child : nullptr;
		}

		if (prev_child->get_focus_mode() == FOCUS_ALL) {
			return prev_child;
		}

		from = prev_child;
	}

	return nullptr;
}

Control::FocusMode Control::get_focus_mode() const {
	return data.focus_mode;
}

bool Control::has_focus() const {
	return is_inside_tree() && get_viewport()->_gui_control_has_focus(this);
}

void Control::grab_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	if (data.focus_mode == FOCUS_NONE) {
		WARN_PRINT("This control can't grab focus. Use set_focus_mode() to allow a control to get focus.");
		return;
	}

	get_viewport()->_gui_control_grab_focus(this);
}

void Control::release_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	if (!has_focus()) {
		return;
	}

	get_viewport()->_gui_remove_focus();
	update();
}

bool Control::is_top_level_control() const {
	return is_inside_tree() && (!data.parent_canvas_item && !data.RI && is_set_as_top_level());
}

void Control::_propagate_theme_changed(Node *p_at, Control *p_owner, Window *p_owner_window, bool p_assign) {
	Control *c = Object::cast_to<Control>(p_at);

	if (c && c != p_owner && c->data.theme.is_valid()) { // has a theme, this can't be propagated
		return;
	}

	Window *w = c == nullptr ? Object::cast_to<Window>(p_at) : nullptr;

	if (w && w != p_owner_window && w->theme.is_valid()) { // has a theme, this can't be propagated
		return;
	}

	for (int i = 0; i < p_at->get_child_count(); i++) {
		CanvasItem *child = Object::cast_to<CanvasItem>(p_at->get_child(i));
		if (child) {
			_propagate_theme_changed(child, p_owner, p_owner_window, p_assign);
		} else {
			Window *window = Object::cast_to<Window>(p_at->get_child(i));
			if (window) {
				_propagate_theme_changed(window, p_owner, p_owner_window, p_assign);
			}
		}
	}

	if (c) {
		if (p_assign) {
			c->data.theme_owner = p_owner;
			c->data.theme_owner_window = p_owner_window;
		}
		c->notification(Control::NOTIFICATION_THEME_CHANGED);
		c->emit_signal(SceneStringNames::get_singleton()->theme_changed);
	}

	if (w) {
		if (p_assign) {
			w->theme_owner = p_owner;
			w->theme_owner_window = p_owner_window;
		}
		w->notification(Window::NOTIFICATION_THEME_CHANGED);
		w->emit_signal(SceneStringNames::get_singleton()->theme_changed);
	}
}

void Control::_theme_changed() {
	_propagate_theme_changed(this, this, nullptr, false);
}

void Control::_notify_theme_changed() {
	if (!data.bulk_theme_override) {
		notification(NOTIFICATION_THEME_CHANGED);
	}
}

void Control::set_theme(const Ref<Theme> &p_theme) {
	if (data.theme == p_theme) {
		return;
	}

	if (data.theme.is_valid()) {
		data.theme->disconnect("changed", callable_mp(this, &Control::_theme_changed));
	}

	data.theme = p_theme;
	if (!p_theme.is_null()) {
		data.theme_owner = this;
		data.theme_owner_window = nullptr;
		_propagate_theme_changed(this, this, nullptr);
	} else {
		Control *parent_c = Object::cast_to<Control>(get_parent());

		if (parent_c && (parent_c->data.theme_owner || parent_c->data.theme_owner_window)) {
			Control::_propagate_theme_changed(this, parent_c->data.theme_owner, parent_c->data.theme_owner_window);
		} else {
			Window *parent_w = cast_to<Window>(get_parent());
			if (parent_w && (parent_w->theme_owner || parent_w->theme_owner_window)) {
				Control::_propagate_theme_changed(this, parent_w->theme_owner, parent_w->theme_owner_window);
			} else {
				Control::_propagate_theme_changed(this, nullptr, nullptr);
			}
		}
	}

	if (data.theme.is_valid()) {
		data.theme->connect("changed", callable_mp(this, &Control::_theme_changed), varray(), CONNECT_DEFERRED);
	}
}

Ref<Theme> Control::get_theme() const {
	return data.theme;
}

void Control::set_theme_type_variation(const StringName &p_theme_type) {
	data.theme_type_variation = p_theme_type;
	_propagate_theme_changed(this, data.theme_owner, data.theme_owner_window);
}

StringName Control::get_theme_type_variation() const {
	return data.theme_type_variation;
}

void Control::set_tooltip(const String &p_tooltip) {
	data.tooltip = p_tooltip;
	update_configuration_warnings();
}

String Control::get_tooltip(const Point2 &p_pos) const {
	return data.tooltip;
}

Control *Control::make_custom_tooltip(const String &p_text) const {
	Object *ret = nullptr;
	if (GDVIRTUAL_CALL(_make_custom_tooltip, p_text, ret)) {
		return Object::cast_to<Control>(ret);
	}
	return nullptr;
}

void Control::set_default_cursor_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(int(p_shape), CURSOR_MAX);

	data.default_cursor = p_shape;
}

Control::CursorShape Control::get_default_cursor_shape() const {
	return data.default_cursor;
}

Control::CursorShape Control::get_cursor_shape(const Point2 &p_pos) const {
	return data.default_cursor;
}

Transform2D Control::get_transform() const {
	Transform2D xform = _get_internal_transform();
	xform[2] += get_position();
	return xform;
}

String Control::_get_tooltip() const {
	return data.tooltip;
}

void Control::set_focus_neighbor(Side p_side, const NodePath &p_neighbor) {
	ERR_FAIL_INDEX((int)p_side, 4);
	data.focus_neighbor[p_side] = p_neighbor;
}

NodePath Control::get_focus_neighbor(Side p_side) const {
	ERR_FAIL_INDEX_V((int)p_side, 4, NodePath());
	return data.focus_neighbor[p_side];
}

void Control::set_focus_next(const NodePath &p_next) {
	data.focus_next = p_next;
}

NodePath Control::get_focus_next() const {
	return data.focus_next;
}

void Control::set_focus_previous(const NodePath &p_prev) {
	data.focus_prev = p_prev;
}

NodePath Control::get_focus_previous() const {
	return data.focus_prev;
}

#define MAX_NEIGHBOR_SEARCH_COUNT 512

Control *Control::_get_focus_neighbor(Side p_side, int p_count) {
	ERR_FAIL_INDEX_V((int)p_side, 4, nullptr);

	if (p_count >= MAX_NEIGHBOR_SEARCH_COUNT) {
		return nullptr;
	}
	if (!data.focus_neighbor[p_side].is_empty()) {
		Control *c = nullptr;
		Node *n = get_node(data.focus_neighbor[p_side]);
		if (n) {
			c = Object::cast_to<Control>(n);
			ERR_FAIL_COND_V_MSG(!c, nullptr, "Neighbor focus node is not a control: " + n->get_name() + ".");
		} else {
			return nullptr;
		}
		bool valid = true;
		if (!c->is_visible()) {
			valid = false;
		}
		if (c->get_focus_mode() == FOCUS_NONE) {
			valid = false;
		}
		if (valid) {
			return c;
		}

		c = c->_get_focus_neighbor(p_side, p_count + 1);
		return c;
	}

	real_t dist = 1e7;
	Control *result = nullptr;

	Point2 points[4];

	Transform2D xform = get_global_transform();

	points[0] = xform.xform(Point2());
	points[1] = xform.xform(Point2(get_size().x, 0));
	points[2] = xform.xform(get_size());
	points[3] = xform.xform(Point2(0, get_size().y));

	const Vector2 dir[4] = {
		Vector2(-1, 0),
		Vector2(0, -1),
		Vector2(1, 0),
		Vector2(0, 1)
	};

	Vector2 vdir = dir[p_side];

	real_t maxd = -1e7;

	for (int i = 0; i < 4; i++) {
		real_t d = vdir.dot(points[i]);
		if (d > maxd) {
			maxd = d;
		}
	}

	Node *base = this;

	while (base) {
		Control *c = Object::cast_to<Control>(base);
		if (c) {
			if (c->data.RI) {
				break;
			}
		}
		base = base->get_parent();
	}

	if (!base) {
		return nullptr;
	}

	_window_find_focus_neighbor(vdir, base, points, maxd, dist, &result);

	return result;
}

void Control::_window_find_focus_neighbor(const Vector2 &p_dir, Node *p_at, const Point2 *p_points, real_t p_min, real_t &r_closest_dist, Control **r_closest) {
	if (Object::cast_to<Viewport>(p_at)) {
		return; //bye
	}

	Control *c = Object::cast_to<Control>(p_at);

	if (c && c != this && c->get_focus_mode() == FOCUS_ALL && c->is_visible_in_tree()) {
		Point2 points[4];

		Transform2D xform = c->get_global_transform();

		points[0] = xform.xform(Point2());
		points[1] = xform.xform(Point2(c->get_size().x, 0));
		points[2] = xform.xform(c->get_size());
		points[3] = xform.xform(Point2(0, c->get_size().y));

		real_t min = 1e7;

		for (int i = 0; i < 4; i++) {
			real_t d = p_dir.dot(points[i]);
			if (d < min) {
				min = d;
			}
		}

		if (min > (p_min - CMP_EPSILON)) {
			for (int i = 0; i < 4; i++) {
				Vector2 la = p_points[i];
				Vector2 lb = p_points[(i + 1) % 4];

				for (int j = 0; j < 4; j++) {
					Vector2 fa = points[j];
					Vector2 fb = points[(j + 1) % 4];

					Vector2 pa, pb;
					real_t d = Geometry2D::get_closest_points_between_segments(la, lb, fa, fb, pa, pb);
					//real_t d = Geometry2D::get_closest_distance_between_segments(Vector3(la.x,la.y,0),Vector3(lb.x,lb.y,0),Vector3(fa.x,fa.y,0),Vector3(fb.x,fb.y,0));
					if (d < r_closest_dist) {
						r_closest_dist = d;
						*r_closest = c;
					}
				}
			}
		}
	}

	for (int i = 0; i < p_at->get_child_count(); i++) {
		Node *child = p_at->get_child(i);
		Control *childc = Object::cast_to<Control>(child);
		if (childc && childc->data.RI) {
			continue; //subwindow, ignore
		}
		_window_find_focus_neighbor(p_dir, p_at->get_child(i), p_points, p_min, r_closest_dist, r_closest);
	}
}

void Control::set_h_size_flags(int p_flags) {
	if (data.h_size_flags == p_flags) {
		return;
	}
	data.h_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

int Control::get_h_size_flags() const {
	return data.h_size_flags;
}

void Control::set_v_size_flags(int p_flags) {
	if (data.v_size_flags == p_flags) {
		return;
	}
	data.v_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

void Control::set_stretch_ratio(real_t p_ratio) {
	if (data.expand == p_ratio) {
		return;
	}

	data.expand = p_ratio;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

real_t Control::get_stretch_ratio() const {
	return data.expand;
}

void Control::grab_click_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	get_viewport()->_gui_grab_click_focus(this);
}

void Control::minimum_size_changed() {
	if (!is_inside_tree() || data.block_minimum_size_adjust) {
		return;
	}

	Control *invalidate = this;

	//invalidate cache upwards
	while (invalidate && invalidate->data.minimum_size_valid) {
		invalidate->data.minimum_size_valid = false;
		if (invalidate->is_set_as_top_level()) {
			break; // do not go further up
		}
		if (!invalidate->data.parent && get_parent()) {
			Window *parent_window = Object::cast_to<Window>(get_parent());
			if (parent_window && parent_window->is_wrapping_controls()) {
				parent_window->child_controls_changed();
			}
		}
		invalidate = invalidate->data.parent;
	}

	if (!is_visible_in_tree()) {
		return;
	}

	if (data.updating_last_minimum_size) {
		return;
	}

	data.updating_last_minimum_size = true;

	MessageQueue::get_singleton()->push_call(this, "_update_minimum_size");
}

int Control::get_v_size_flags() const {
	return data.v_size_flags;
}

void Control::set_mouse_filter(MouseFilter p_filter) {
	ERR_FAIL_INDEX(p_filter, 3);
	data.mouse_filter = p_filter;
	update_configuration_warnings();
}

Control::MouseFilter Control::get_mouse_filter() const {
	return data.mouse_filter;
}

Control *Control::get_focus_owner() const {
	ERR_FAIL_COND_V(!is_inside_tree(), nullptr);
	return get_viewport()->_gui_get_focus_owner();
}

void Control::warp_mouse(const Point2 &p_to_pos) {
	ERR_FAIL_COND(!is_inside_tree());
	get_viewport()->warp_mouse(get_global_transform().xform(p_to_pos));
}

bool Control::is_text_field() const {
	/*
    if (get_script_instance()) {
        Variant v=p_point;
        const Variant *p[2]={&v,&p_data};
        Callable::CallError ce;
        Variant ret = get_script_instance()->call("is_text_field",p,2,ce);
        if (ce.error==Callable::CallError::CALL_OK)
            return ret;
    }
  */
	return false;
}

Array Control::structured_text_parser(StructuredTextParser p_theme_type, const Array &p_args, const String p_text) const {
	Array ret;
	switch (p_theme_type) {
		case STRUCTURED_TEXT_URI: {
			int prev = 0;
			for (int i = 0; i < p_text.length(); i++) {
				if ((p_text[i] == '\\') || (p_text[i] == '/') || (p_text[i] == '.') || (p_text[i] == ':') || (p_text[i] == '&') || (p_text[i] == '=') || (p_text[i] == '@') || (p_text[i] == '?') || (p_text[i] == '#')) {
					if (prev != i) {
						ret.push_back(Vector2i(prev, i));
					}
					ret.push_back(Vector2i(i, i + 1));
					prev = i + 1;
				}
			}
			if (prev != p_text.length()) {
				ret.push_back(Vector2i(prev, p_text.length()));
			}
		} break;
		case STRUCTURED_TEXT_FILE: {
			int prev = 0;
			for (int i = 0; i < p_text.length(); i++) {
				if ((p_text[i] == '\\') || (p_text[i] == '/') || (p_text[i] == ':')) {
					if (prev != i) {
						ret.push_back(Vector2i(prev, i));
					}
					ret.push_back(Vector2i(i, i + 1));
					prev = i + 1;
				}
			}
			if (prev != p_text.length()) {
				ret.push_back(Vector2i(prev, p_text.length()));
			}
		} break;
		case STRUCTURED_TEXT_EMAIL: {
			bool local = true;
			int prev = 0;
			for (int i = 0; i < p_text.length(); i++) {
				if ((p_text[i] == '@') && local) { // Add full "local" as single context.
					local = false;
					ret.push_back(Vector2i(prev, i));
					ret.push_back(Vector2i(i, i + 1));
					prev = i + 1;
				} else if (!local & (p_text[i] == '.')) { // Add each dot separated "domain" part as context.
					if (prev != i) {
						ret.push_back(Vector2i(prev, i));
					}
					ret.push_back(Vector2i(i, i + 1));
					prev = i + 1;
				}
			}
			if (prev != p_text.length()) {
				ret.push_back(Vector2i(prev, p_text.length()));
			}
		} break;
		case STRUCTURED_TEXT_LIST: {
			if (p_args.size() == 1 && p_args[0].get_type() == Variant::STRING) {
				Vector<String> tags = p_text.split(String(p_args[0]));
				int prev = 0;
				for (int i = 0; i < tags.size(); i++) {
					if (prev != i) {
						ret.push_back(Vector2i(prev, prev + tags[i].length()));
					}
					ret.push_back(Vector2i(prev + tags[i].length(), prev + tags[i].length() + 1));
					prev = prev + tags[i].length() + 1;
				}
			}
		} break;
		case STRUCTURED_TEXT_CUSTOM: {
			Array r;
			if (GDVIRTUAL_CALL(_structured_text_parser, p_args, p_text, r)) {
				for (int i = 0; i < r.size(); i++) {
					if (r[i].get_type() == Variant::VECTOR2I) {
						ret.push_back(Vector2i(r[i]));
					}
				}
			}
		} break;
		case STRUCTURED_TEXT_NONE:
		case STRUCTURED_TEXT_DEFAULT:
		default: {
			ret.push_back(Vector2i(0, p_text.length()));
		}
	}
	return ret;
}

void Control::set_rotation(real_t p_radians) {
	data.rotation = p_radians;
	update();
	_notify_transform();
}

real_t Control::get_rotation() const {
	return data.rotation;
}

void Control::_override_changed() {
	notification(NOTIFICATION_THEME_CHANGED);
	emit_signal(SceneStringNames::get_singleton()->theme_changed);
	minimum_size_changed(); // overrides are likely to affect minimum size
}

void Control::set_pivot_offset(const Vector2 &p_pivot) {
	data.pivot_offset = p_pivot;
	update();
	_notify_transform();
}

Vector2 Control::get_pivot_offset() const {
	return data.pivot_offset;
}

void Control::set_scale(const Vector2 &p_scale) {
	data.scale = p_scale;
	// Avoid having 0 scale values, can lead to errors in physics and rendering.
	if (data.scale.x == 0) {
		data.scale.x = CMP_EPSILON;
	}
	if (data.scale.y == 0) {
		data.scale.y = CMP_EPSILON;
	}
	update();
	_notify_transform();
}

Vector2 Control::get_scale() const {
	return data.scale;
}

Control *Control::get_root_parent_control() const {
	const CanvasItem *ci = this;
	const Control *root = this;

	while (ci) {
		const Control *c = Object::cast_to<Control>(ci);
		if (c) {
			root = c;

			if (c->data.RI || c->is_top_level_control()) {
				break;
			}
		}

		ci = ci->get_parent_item();
	}

	return const_cast<Control *>(root);
}

void Control::set_block_minimum_size_adjust(bool p_block) {
	data.block_minimum_size_adjust = p_block;
}

bool Control::is_minimum_size_adjust_blocked() const {
	return data.block_minimum_size_adjust;
}

void Control::set_disable_visibility_clip(bool p_ignore) {
	data.disable_visibility_clip = p_ignore;
	update();
}

bool Control::is_visibility_clip_disabled() const {
	return data.disable_visibility_clip;
}

void Control::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	Node::get_argument_options(p_function, p_idx, r_options);

	if (p_idx == 0) {
		List<StringName> sn;
		String pf = p_function;
		if (pf == "add_theme_color_override" || pf == "has_theme_color" || pf == "has_theme_color_override" || pf == "get_theme_color") {
			Theme::get_default()->get_color_list(get_class(), &sn);
		} else if (pf == "add_theme_style_override" || pf == "has_theme_style" || pf == "has_theme_style_override" || pf == "get_theme_style") {
			Theme::get_default()->get_stylebox_list(get_class(), &sn);
		} else if (pf == "add_theme_font_override" || pf == "has_theme_font" || pf == "has_theme_font_override" || pf == "get_theme_font") {
			Theme::get_default()->get_font_list(get_class(), &sn);
		} else if (pf == "add_theme_font_size_override" || pf == "has_theme_font_size" || pf == "has_theme_font_size_override" || pf == "get_theme_font_size") {
			Theme::get_default()->get_font_size_list(get_class(), &sn);
		} else if (pf == "add_theme_constant_override" || pf == "has_theme_constant" || pf == "has_theme_constant_override" || pf == "get_theme_constant") {
			Theme::get_default()->get_constant_list(get_class(), &sn);
		}

		sn.sort_custom<StringName::AlphCompare>();
		for (const StringName &name : sn) {
			r_options->push_back(String(name).quote());
		}
	}
}

TypedArray<String> Control::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (data.mouse_filter == MOUSE_FILTER_IGNORE && data.tooltip != "") {
		warnings.push_back(TTR("The Hint Tooltip won't be displayed as the control's Mouse Filter is set to \"Ignore\". To solve this, set the Mouse Filter to \"Stop\" or \"Pass\"."));
	}

	return warnings;
}

void Control::set_clip_contents(bool p_clip) {
	data.clip_contents = p_clip;
	update();
}

bool Control::is_clipping_contents() {
	return data.clip_contents;
}

void Control::set_h_grow_direction(GrowDirection p_direction) {
	ERR_FAIL_INDEX((int)p_direction, 3);

	data.h_grow = p_direction;
	_size_changed();
}

Control::GrowDirection Control::get_h_grow_direction() const {
	return data.h_grow;
}

void Control::set_v_grow_direction(GrowDirection p_direction) {
	ERR_FAIL_INDEX((int)p_direction, 3);

	data.v_grow = p_direction;
	_size_changed();
}

Control::GrowDirection Control::get_v_grow_direction() const {
	return data.v_grow;
}

void Control::_bind_methods() {
	//ClassDB::bind_method(D_METHOD("_window_resize_event"),&Control::_window_resize_event);
	ClassDB::bind_method(D_METHOD("_update_minimum_size"), &Control::_update_minimum_size);

	ClassDB::bind_method(D_METHOD("accept_event"), &Control::accept_event);
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &Control::get_minimum_size);
	ClassDB::bind_method(D_METHOD("get_combined_minimum_size"), &Control::get_combined_minimum_size);
	ClassDB::bind_method(D_METHOD("set_anchors_preset", "preset", "keep_offsets"), &Control::set_anchors_preset, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_offsets_preset", "preset", "resize_mode", "margin"), &Control::set_offsets_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_anchors_and_offsets_preset", "preset", "resize_mode", "margin"), &Control::set_anchors_and_offsets_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("_set_anchor", "side", "anchor"), &Control::_set_anchor);
	ClassDB::bind_method(D_METHOD("set_anchor", "side", "anchor", "keep_offset", "push_opposite_anchor"), &Control::set_anchor, DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_anchor", "side"), &Control::get_anchor);
	ClassDB::bind_method(D_METHOD("set_offset", "side", "offset"), &Control::set_offset);
	ClassDB::bind_method(D_METHOD("set_anchor_and_offset", "side", "anchor", "offset", "push_opposite_anchor"), &Control::set_anchor_and_offset, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_begin", "position"), &Control::set_begin);
	ClassDB::bind_method(D_METHOD("set_end", "position"), &Control::set_end);
	ClassDB::bind_method(D_METHOD("set_position", "position", "keep_offsets"), &Control::set_position, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_position", "position"), &Control::_set_position);
	ClassDB::bind_method(D_METHOD("set_size", "size", "keep_offsets"), &Control::set_size, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_size", "size"), &Control::_set_size);
	ClassDB::bind_method(D_METHOD("set_custom_minimum_size", "size"), &Control::set_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("set_global_position", "position", "keep_offsets"), &Control::set_global_position, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_global_position", "position"), &Control::_set_global_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "radians"), &Control::set_rotation);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Control::set_scale);
	ClassDB::bind_method(D_METHOD("set_pivot_offset", "pivot_offset"), &Control::set_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_offset", "offset"), &Control::get_offset);
	ClassDB::bind_method(D_METHOD("get_begin"), &Control::get_begin);
	ClassDB::bind_method(D_METHOD("get_end"), &Control::get_end);
	ClassDB::bind_method(D_METHOD("get_position"), &Control::get_position);
	ClassDB::bind_method(D_METHOD("get_size"), &Control::get_size);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Control::get_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &Control::get_scale);
	ClassDB::bind_method(D_METHOD("get_pivot_offset"), &Control::get_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_custom_minimum_size"), &Control::get_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("get_parent_area_size"), &Control::get_parent_area_size);
	ClassDB::bind_method(D_METHOD("get_global_position"), &Control::get_global_position);
	ClassDB::bind_method(D_METHOD("get_rect"), &Control::get_rect);
	ClassDB::bind_method(D_METHOD("get_global_rect"), &Control::get_global_rect);
	ClassDB::bind_method(D_METHOD("set_focus_mode", "mode"), &Control::set_focus_mode);
	ClassDB::bind_method(D_METHOD("get_focus_mode"), &Control::get_focus_mode);
	ClassDB::bind_method(D_METHOD("has_focus"), &Control::has_focus);
	ClassDB::bind_method(D_METHOD("grab_focus"), &Control::grab_focus);
	ClassDB::bind_method(D_METHOD("release_focus"), &Control::release_focus);
	ClassDB::bind_method(D_METHOD("find_prev_valid_focus"), &Control::find_prev_valid_focus);
	ClassDB::bind_method(D_METHOD("find_next_valid_focus"), &Control::find_next_valid_focus);
	ClassDB::bind_method(D_METHOD("get_focus_owner"), &Control::get_focus_owner);

	ClassDB::bind_method(D_METHOD("set_h_size_flags", "flags"), &Control::set_h_size_flags);
	ClassDB::bind_method(D_METHOD("get_h_size_flags"), &Control::get_h_size_flags);

	ClassDB::bind_method(D_METHOD("set_stretch_ratio", "ratio"), &Control::set_stretch_ratio);
	ClassDB::bind_method(D_METHOD("get_stretch_ratio"), &Control::get_stretch_ratio);

	ClassDB::bind_method(D_METHOD("set_v_size_flags", "flags"), &Control::set_v_size_flags);
	ClassDB::bind_method(D_METHOD("get_v_size_flags"), &Control::get_v_size_flags);

	ClassDB::bind_method(D_METHOD("set_theme", "theme"), &Control::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme"), &Control::get_theme);

	ClassDB::bind_method(D_METHOD("set_theme_type_variation", "theme_type"), &Control::set_theme_type_variation);
	ClassDB::bind_method(D_METHOD("get_theme_type_variation"), &Control::get_theme_type_variation);

	ClassDB::bind_method(D_METHOD("begin_bulk_theme_override"), &Control::begin_bulk_theme_override);
	ClassDB::bind_method(D_METHOD("end_bulk_theme_override"), &Control::end_bulk_theme_override);

	ClassDB::bind_method(D_METHOD("add_theme_icon_override", "name", "texture"), &Control::add_theme_icon_override);
	ClassDB::bind_method(D_METHOD("add_theme_stylebox_override", "name", "stylebox"), &Control::add_theme_style_override);
	ClassDB::bind_method(D_METHOD("add_theme_font_override", "name", "font"), &Control::add_theme_font_override);
	ClassDB::bind_method(D_METHOD("add_theme_font_size_override", "name", "font_size"), &Control::add_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("add_theme_color_override", "name", "color"), &Control::add_theme_color_override);
	ClassDB::bind_method(D_METHOD("add_theme_constant_override", "name", "constant"), &Control::add_theme_constant_override);

	ClassDB::bind_method(D_METHOD("remove_theme_icon_override", "name"), &Control::remove_theme_icon_override);
	ClassDB::bind_method(D_METHOD("remove_theme_stylebox_override", "name"), &Control::remove_theme_style_override);
	ClassDB::bind_method(D_METHOD("remove_theme_font_override", "name"), &Control::remove_theme_font_override);
	ClassDB::bind_method(D_METHOD("remove_theme_font_size_override", "name"), &Control::remove_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("remove_theme_color_override", "name"), &Control::remove_theme_color_override);
	ClassDB::bind_method(D_METHOD("remove_theme_constant_override", "name"), &Control::remove_theme_constant_override);

	ClassDB::bind_method(D_METHOD("get_theme_icon", "name", "theme_type"), &Control::get_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_stylebox", "name", "theme_type"), &Control::get_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font", "name", "theme_type"), &Control::get_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font_size", "name", "theme_type"), &Control::get_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_color", "name", "theme_type"), &Control::get_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_constant", "name", "theme_type"), &Control::get_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("has_theme_icon_override", "name"), &Control::has_theme_icon_override);
	ClassDB::bind_method(D_METHOD("has_theme_stylebox_override", "name"), &Control::has_theme_stylebox_override);
	ClassDB::bind_method(D_METHOD("has_theme_font_override", "name"), &Control::has_theme_font_override);
	ClassDB::bind_method(D_METHOD("has_theme_font_size_override", "name"), &Control::has_theme_font_size_override);
	ClassDB::bind_method(D_METHOD("has_theme_color_override", "name"), &Control::has_theme_color_override);
	ClassDB::bind_method(D_METHOD("has_theme_constant_override", "name"), &Control::has_theme_constant_override);

	ClassDB::bind_method(D_METHOD("has_theme_icon", "name", "theme_type"), &Control::has_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_stylebox", "name", "theme_type"), &Control::has_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font", "name", "theme_type"), &Control::has_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font_size", "name", "theme_type"), &Control::has_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_color", "name", "theme_type"), &Control::has_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_constant", "name", "theme_type"), &Control::has_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("get_theme_default_base_scale"), &Control::get_theme_default_base_scale);
	ClassDB::bind_method(D_METHOD("get_theme_default_font"), &Control::get_theme_default_font);
	ClassDB::bind_method(D_METHOD("get_theme_default_font_size"), &Control::get_theme_default_font_size);

	ClassDB::bind_method(D_METHOD("get_parent_control"), &Control::get_parent_control);

	ClassDB::bind_method(D_METHOD("set_h_grow_direction", "direction"), &Control::set_h_grow_direction);
	ClassDB::bind_method(D_METHOD("get_h_grow_direction"), &Control::get_h_grow_direction);

	ClassDB::bind_method(D_METHOD("set_v_grow_direction", "direction"), &Control::set_v_grow_direction);
	ClassDB::bind_method(D_METHOD("get_v_grow_direction"), &Control::get_v_grow_direction);

	ClassDB::bind_method(D_METHOD("set_tooltip", "tooltip"), &Control::set_tooltip);
	ClassDB::bind_method(D_METHOD("get_tooltip", "at_position"), &Control::get_tooltip, DEFVAL(Point2()));
	ClassDB::bind_method(D_METHOD("_get_tooltip"), &Control::_get_tooltip);

	ClassDB::bind_method(D_METHOD("set_default_cursor_shape", "shape"), &Control::set_default_cursor_shape);
	ClassDB::bind_method(D_METHOD("get_default_cursor_shape"), &Control::get_default_cursor_shape);
	ClassDB::bind_method(D_METHOD("get_cursor_shape", "position"), &Control::get_cursor_shape, DEFVAL(Point2()));

	ClassDB::bind_method(D_METHOD("set_focus_neighbor", "side", "neighbor"), &Control::set_focus_neighbor);
	ClassDB::bind_method(D_METHOD("get_focus_neighbor", "side"), &Control::get_focus_neighbor);

	ClassDB::bind_method(D_METHOD("set_focus_next", "next"), &Control::set_focus_next);
	ClassDB::bind_method(D_METHOD("get_focus_next"), &Control::get_focus_next);

	ClassDB::bind_method(D_METHOD("set_focus_previous", "previous"), &Control::set_focus_previous);
	ClassDB::bind_method(D_METHOD("get_focus_previous"), &Control::get_focus_previous);

	ClassDB::bind_method(D_METHOD("force_drag", "data", "preview"), &Control::force_drag);

	ClassDB::bind_method(D_METHOD("set_mouse_filter", "filter"), &Control::set_mouse_filter);
	ClassDB::bind_method(D_METHOD("get_mouse_filter"), &Control::get_mouse_filter);

	ClassDB::bind_method(D_METHOD("set_clip_contents", "enable"), &Control::set_clip_contents);
	ClassDB::bind_method(D_METHOD("is_clipping_contents"), &Control::is_clipping_contents);

	ClassDB::bind_method(D_METHOD("grab_click_focus"), &Control::grab_click_focus);

	ClassDB::bind_method(D_METHOD("set_drag_forwarding", "target"), &Control::set_drag_forwarding);
	ClassDB::bind_method(D_METHOD("set_drag_preview", "control"), &Control::set_drag_preview);

	ClassDB::bind_method(D_METHOD("warp_mouse", "to_position"), &Control::warp_mouse);

	ClassDB::bind_method(D_METHOD("minimum_size_changed"), &Control::minimum_size_changed);

	ClassDB::bind_method(D_METHOD("set_layout_direction", "direction"), &Control::set_layout_direction);
	ClassDB::bind_method(D_METHOD("get_layout_direction"), &Control::get_layout_direction);
	ClassDB::bind_method(D_METHOD("is_layout_rtl"), &Control::is_layout_rtl);

	ClassDB::bind_method(D_METHOD("set_auto_translate", "enable"), &Control::set_auto_translate);
	ClassDB::bind_method(D_METHOD("is_auto_translating"), &Control::is_auto_translating);

	ADD_GROUP("Anchor", "anchor_");
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_left", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_top", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_right", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "anchor_bottom", PROPERTY_HINT_RANGE, "0,1,0.001,or_lesser,or_greater"), "_set_anchor", "get_anchor", SIDE_BOTTOM);

	ADD_GROUP("Offset", "offset_");
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_left", PROPERTY_HINT_RANGE, "-4096,4096"), "set_offset", "get_offset", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_top", PROPERTY_HINT_RANGE, "-4096,4096"), "set_offset", "get_offset", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_right", PROPERTY_HINT_RANGE, "-4096,4096"), "set_offset", "get_offset", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::INT, "offset_bottom", PROPERTY_HINT_RANGE, "-4096,4096"), "set_offset", "get_offset", SIDE_BOTTOM);

	ADD_GROUP("Grow Direction", "grow_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "grow_horizontal", PROPERTY_HINT_ENUM, "Begin,End,Both"), "set_h_grow_direction", "get_h_grow_direction");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "grow_vertical", PROPERTY_HINT_ENUM, "Begin,End,Both"), "set_v_grow_direction", "get_v_grow_direction");

	ADD_GROUP("Layout Direction", "layout_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "layout_direction", PROPERTY_HINT_ENUM, "Inherited,Locale,Left-to-Right,Right-to-Left"), "set_layout_direction", "get_layout_direction");

	ADD_GROUP("Auto Translate", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_translate"), "set_auto_translate", "is_auto_translating");

	ADD_GROUP("Rect", "rect_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_global_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "_set_global_position", "get_global_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_min_size"), "set_custom_minimum_size", "get_custom_minimum_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rect_rotation", PROPERTY_HINT_RANGE, "-360,360,0.1,or_lesser,or_greater,radians"), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_scale"), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "rect_pivot_offset"), "set_pivot_offset", "get_pivot_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rect_clip_content"), "set_clip_contents", "is_clipping_contents");

	ADD_GROUP("Hint", "hint_");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "hint_tooltip", PROPERTY_HINT_MULTILINE_TEXT), "set_tooltip", "_get_tooltip");

	ADD_GROUP("Focus", "focus_");
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_left", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_top", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_right", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::NODE_PATH, "focus_neighbor_bottom", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_neighbor", "get_focus_neighbor", SIDE_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "focus_next", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_next", "get_focus_next");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "focus_previous", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Control"), "set_focus_previous", "get_focus_previous");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "focus_mode", PROPERTY_HINT_ENUM, "None,Click,All"), "set_focus_mode", "get_focus_mode");

	ADD_GROUP("Mouse", "mouse_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_filter", PROPERTY_HINT_ENUM, "Stop,Pass,Ignore"), "set_mouse_filter", "get_mouse_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_default_cursor_shape", PROPERTY_HINT_ENUM, "Arrow,I-Beam,Pointing Hand,Cross,Wait,Busy,Drag,Can Drop,Forbidden,Vertical Resize,Horizontal Resize,Secondary Diagonal Resize,Main Diagonal Resize,Move,Vertical Split,Horizontal Split,Help"), "set_default_cursor_shape", "get_default_cursor_shape");

	ADD_GROUP("Size Flags", "size_flags_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_horizontal", PROPERTY_HINT_FLAGS, "Fill,Expand,Shrink Center,Shrink End"), "set_h_size_flags", "get_h_size_flags");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_vertical", PROPERTY_HINT_FLAGS, "Fill,Expand,Shrink Center,Shrink End"), "set_v_size_flags", "get_v_size_flags");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "size_flags_stretch_ratio", PROPERTY_HINT_RANGE, "0,20,0.01,or_greater"), "set_stretch_ratio", "get_stretch_ratio");

	ADD_GROUP("Theme", "theme_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "Theme"), "set_theme", "get_theme");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "theme_type_variation", PROPERTY_HINT_ENUM_SUGGESTION), "set_theme_type_variation", "get_theme_type_variation");

	BIND_ENUM_CONSTANT(FOCUS_NONE);
	BIND_ENUM_CONSTANT(FOCUS_CLICK);
	BIND_ENUM_CONSTANT(FOCUS_ALL);

	BIND_CONSTANT(NOTIFICATION_RESIZED);
	BIND_CONSTANT(NOTIFICATION_MOUSE_ENTER);
	BIND_CONSTANT(NOTIFICATION_MOUSE_EXIT);
	BIND_CONSTANT(NOTIFICATION_FOCUS_ENTER);
	BIND_CONSTANT(NOTIFICATION_FOCUS_EXIT);
	BIND_CONSTANT(NOTIFICATION_THEME_CHANGED);
	BIND_CONSTANT(NOTIFICATION_SCROLL_BEGIN);
	BIND_CONSTANT(NOTIFICATION_SCROLL_END);
	BIND_CONSTANT(NOTIFICATION_LAYOUT_DIRECTION_CHANGED);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);

	BIND_ENUM_CONSTANT(PRESET_TOP_LEFT);
	BIND_ENUM_CONSTANT(PRESET_TOP_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_LEFT);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_LEFT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_TOP);
	BIND_ENUM_CONSTANT(PRESET_CENTER_RIGHT);
	BIND_ENUM_CONSTANT(PRESET_CENTER_BOTTOM);
	BIND_ENUM_CONSTANT(PRESET_CENTER);
	BIND_ENUM_CONSTANT(PRESET_LEFT_WIDE);
	BIND_ENUM_CONSTANT(PRESET_TOP_WIDE);
	BIND_ENUM_CONSTANT(PRESET_RIGHT_WIDE);
	BIND_ENUM_CONSTANT(PRESET_BOTTOM_WIDE);
	BIND_ENUM_CONSTANT(PRESET_VCENTER_WIDE);
	BIND_ENUM_CONSTANT(PRESET_HCENTER_WIDE);
	BIND_ENUM_CONSTANT(PRESET_WIDE);

	BIND_ENUM_CONSTANT(PRESET_MODE_MINSIZE);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_WIDTH);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_HEIGHT);
	BIND_ENUM_CONSTANT(PRESET_MODE_KEEP_SIZE);

	BIND_ENUM_CONSTANT(SIZE_FILL);
	BIND_ENUM_CONSTANT(SIZE_EXPAND);
	BIND_ENUM_CONSTANT(SIZE_EXPAND_FILL);
	BIND_ENUM_CONSTANT(SIZE_SHRINK_CENTER);
	BIND_ENUM_CONSTANT(SIZE_SHRINK_END);

	BIND_ENUM_CONSTANT(MOUSE_FILTER_STOP);
	BIND_ENUM_CONSTANT(MOUSE_FILTER_PASS);
	BIND_ENUM_CONSTANT(MOUSE_FILTER_IGNORE);

	BIND_ENUM_CONSTANT(GROW_DIRECTION_BEGIN);
	BIND_ENUM_CONSTANT(GROW_DIRECTION_END);
	BIND_ENUM_CONSTANT(GROW_DIRECTION_BOTH);

	BIND_ENUM_CONSTANT(ANCHOR_BEGIN);
	BIND_ENUM_CONSTANT(ANCHOR_END);

	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_INHERITED);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LOCALE);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LTR);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_RTL);

	BIND_ENUM_CONSTANT(TEXT_DIRECTION_INHERITED);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_AUTO);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_LTR);
	BIND_ENUM_CONSTANT(TEXT_DIRECTION_RTL);

	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_DEFAULT);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_URI);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_FILE);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_EMAIL);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_LIST);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_NONE);
	BIND_ENUM_CONSTANT(STRUCTURED_TEXT_CUSTOM);

	ADD_SIGNAL(MethodInfo("resized"));
	ADD_SIGNAL(MethodInfo("gui_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));
	ADD_SIGNAL(MethodInfo("focus_entered"));
	ADD_SIGNAL(MethodInfo("focus_exited"));
	ADD_SIGNAL(MethodInfo("size_flags_changed"));
	ADD_SIGNAL(MethodInfo("minimum_size_changed"));
	ADD_SIGNAL(MethodInfo("theme_changed"));

	GDVIRTUAL_BIND(_has_point, "position");
	GDVIRTUAL_BIND(_structured_text_parser, "args", "text");
	GDVIRTUAL_BIND(_get_minimum_size);

	GDVIRTUAL_BIND(_get_drag_data, "at_position");
	GDVIRTUAL_BIND(_can_drop_data, "at_position", "data");
	GDVIRTUAL_BIND(_drop_data, "at_position", "data");
	GDVIRTUAL_BIND(_make_custom_tooltip, "for_text");

	GDVIRTUAL_BIND(_gui_input, "event");
}
