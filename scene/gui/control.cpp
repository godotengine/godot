/*************************************************************************/
/*  control.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "control.h"
#include "project_settings.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/viewport.h"
#include "servers/visual_server.h"

#include "message_queue.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/scene_string_names.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif
#include <stdio.h>

Dictionary Control::_edit_get_state() const {

	Dictionary s;
	s["rect"] = get_rect();
	s["rotation"] = get_rotation();
	s["scale"] = get_scale();
	Array anchors;
	anchors.push_back(get_anchor(MARGIN_LEFT));
	anchors.push_back(get_anchor(MARGIN_TOP));
	anchors.push_back(get_anchor(MARGIN_RIGHT));
	anchors.push_back(get_anchor(MARGIN_BOTTOM));
	s["anchors"] = anchors;
	return s;
}
void Control::_edit_set_state(const Dictionary &p_state) {

	Dictionary state = p_state;

	Rect2 rect = state["rect"];
	set_position(rect.position);
	set_size(rect.size);
	set_rotation(state["rotation"]);
	set_scale(state["scale"]);
	Array anchors = state["anchors"];
	set_anchor(MARGIN_LEFT, anchors[0]);
	set_anchor(MARGIN_TOP, anchors[1]);
	set_anchor(MARGIN_RIGHT, anchors[2]);
	set_anchor(MARGIN_BOTTOM, anchors[3]);
}

void Control::_edit_set_position(const Point2 &p_position) {
	set_position(p_position);
};

Point2 Control::_edit_get_position() const {
	return get_position();
};

void Control::_edit_set_rect(const Rect2 &p_edit_rect) {

	Transform2D xform = _get_internal_transform();

	Vector2 new_pos = xform.basis_xform(p_edit_rect.position);

	Vector2 pos = get_position() + new_pos;

	Rect2 new_rect = get_rect();
	new_rect.position = pos.snapped(Vector2(1, 1));
	new_rect.size = p_edit_rect.size.snapped(Vector2(1, 1));

	set_position(new_rect.position);
	set_size(new_rect.size);
}

Rect2 Control::_edit_get_rect() const {
	return Rect2(Point2(), get_size());
}

bool Control::_edit_use_rect() const {
	return true;
}

void Control::_edit_set_rotation(float p_rotation) {
	set_rotation(p_rotation);
}

float Control::_edit_get_rotation() const {
	return get_rotation();
}

bool Control::_edit_use_rotation() const {
	return true;
}

void Control::_edit_set_pivot(const Point2 &p_pivot) {
	set_pivot_offset(p_pivot);
}

Point2 Control::_edit_get_pivot() const {
	return get_pivot_offset();
}

bool Control::_edit_use_pivot() const {
	return true;
}

void Control::set_custom_minimum_size(const Size2 &p_custom) {

	if (p_custom == data.custom_minimum_size)
		return;
	data.custom_minimum_size = p_custom;
	minimum_size_changed();
}

Size2 Control::get_custom_minimum_size() const {

	return data.custom_minimum_size;
}

Size2 Control::get_combined_minimum_size() const {

	Size2 minsize = get_minimum_size();
	minsize.x = MAX(minsize.x, data.custom_minimum_size.x);
	minsize.y = MAX(minsize.y, data.custom_minimum_size.y);
	return minsize;
}

Size2 Control::_edit_get_minimum_size() const {

	return get_combined_minimum_size();
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
	if (!name.begins_with("custom")) {
		return false;
	}

	if (p_value.get_type() == Variant::NIL) {

		if (name.begins_with("custom_icons/")) {
			String dname = name.get_slicec('/', 1);
			data.icon_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else if (name.begins_with("custom_shaders/")) {
			String dname = name.get_slicec('/', 1);
			data.shader_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else if (name.begins_with("custom_styles/")) {
			String dname = name.get_slicec('/', 1);
			data.style_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else if (name.begins_with("custom_fonts/")) {
			String dname = name.get_slicec('/', 1);
			if (data.font_override.has(dname)) {
				_unref_font(data.font_override[dname]);
			}
			data.font_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else if (name.begins_with("custom_colors/")) {
			String dname = name.get_slicec('/', 1);
			data.color_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else if (name.begins_with("custom_constants/")) {
			String dname = name.get_slicec('/', 1);
			data.constant_override.erase(dname);
			notification(NOTIFICATION_THEME_CHANGED);
			update();
		} else
			return false;

	} else {
		if (name.begins_with("custom_icons/")) {
			String dname = name.get_slicec('/', 1);
			notification(NOTIFICATION_THEME_CHANGED);
			add_icon_override(dname, p_value);
		} else if (name.begins_with("custom_shaders/")) {
			String dname = name.get_slicec('/', 1);
			add_shader_override(dname, p_value);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("custom_styles/")) {
			String dname = name.get_slicec('/', 1);
			add_style_override(dname, p_value);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("custom_fonts/")) {
			String dname = name.get_slicec('/', 1);
			add_font_override(dname, p_value);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("custom_colors/")) {
			String dname = name.get_slicec('/', 1);
			add_color_override(dname, p_value);
			notification(NOTIFICATION_THEME_CHANGED);
		} else if (name.begins_with("custom_constants/")) {
			String dname = name.get_slicec('/', 1);
			add_constant_override(dname, p_value);
			notification(NOTIFICATION_THEME_CHANGED);
		} else
			return false;
	}
	return true;
}

void Control::_update_minimum_size() {

	if (!is_inside_tree())
		return;

	data.pending_min_size_update = false;
	Size2 minsize = get_combined_minimum_size();
	if (minsize.x > data.size_cache.x ||
			minsize.y > data.size_cache.y) {
		_size_changed();
	}

	emit_signal(SceneStringNames::get_singleton()->minimum_size_changed);
}

bool Control::_get(const StringName &p_name, Variant &r_ret) const {

	String sname = p_name;

	if (!sname.begins_with("custom")) {
		return false;
	}

	if (sname.begins_with("custom_icons/")) {
		String name = sname.get_slicec('/', 1);

		r_ret = data.icon_override.has(name) ? Variant(data.icon_override[name]) : Variant();
	} else if (sname.begins_with("custom_shaders/")) {
		String name = sname.get_slicec('/', 1);

		r_ret = data.shader_override.has(name) ? Variant(data.shader_override[name]) : Variant();
	} else if (sname.begins_with("custom_styles/")) {
		String name = sname.get_slicec('/', 1);

		r_ret = data.style_override.has(name) ? Variant(data.style_override[name]) : Variant();
	} else if (sname.begins_with("custom_fonts/")) {
		String name = sname.get_slicec('/', 1);

		r_ret = data.font_override.has(name) ? Variant(data.font_override[name]) : Variant();
	} else if (sname.begins_with("custom_colors/")) {
		String name = sname.get_slicec('/', 1);
		r_ret = data.color_override.has(name) ? Variant(data.color_override[name]) : Variant();
	} else if (sname.begins_with("custom_constants/")) {
		String name = sname.get_slicec('/', 1);

		r_ret = data.constant_override.has(name) ? Variant(data.constant_override[name]) : Variant();
	} else
		return false;

	return true;
}
void Control::_get_property_list(List<PropertyInfo> *p_list) const {

	Ref<Theme> theme;
	if (data.theme.is_valid()) {

		theme = data.theme;
	} else {
		theme = Theme::get_default();
	}

	{
		List<StringName> names;
		theme->get_icon_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.icon_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::OBJECT, "custom_icons/" + E->get(), PROPERTY_HINT_RESOURCE_TYPE, "Texture", hint));
		}
	}
	{
		List<StringName> names;
		theme->get_shader_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.shader_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::OBJECT, "custom_shaders/" + E->get(), PROPERTY_HINT_RESOURCE_TYPE, "CanvasItemShader,CanvasItemShaderGraph", hint));
		}
	}
	{
		List<StringName> names;
		theme->get_stylebox_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.style_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::OBJECT, "custom_styles/" + E->get(), PROPERTY_HINT_RESOURCE_TYPE, "StyleBox", hint));
		}
	}
	{
		List<StringName> names;
		theme->get_font_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.font_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::OBJECT, "custom_fonts/" + E->get(), PROPERTY_HINT_RESOURCE_TYPE, "Font", hint));
		}
	}
	{
		List<StringName> names;
		theme->get_color_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.color_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::COLOR, "custom_colors/" + E->get(), PROPERTY_HINT_NONE, "", hint));
		}
	}
	{
		List<StringName> names;
		theme->get_constant_list(get_class_name(), &names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {

			uint32_t hint = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CHECKABLE;
			if (data.constant_override.has(E->get()))
				hint |= PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_CHECKED;

			p_list->push_back(PropertyInfo(Variant::INT, "custom_constants/" + E->get(), PROPERTY_HINT_RANGE, "-16384,16384", hint));
		}
	}
}

Control *Control::get_parent_control() const {

	return data.parent;
}

void Control::_resize(const Size2 &p_size) {

	_size_changed();
}

//moved theme configuration here, so controls can set up even if still not inside active scene

void Control::add_child_notify(Node *p_child) {

	Control *child_c = Object::cast_to<Control>(p_child);
	if (!child_c)
		return;

	if (child_c->data.theme.is_null() && data.theme_owner) {
		_propagate_theme_changed(child_c, data.theme_owner); //need to propagate here, since many controls may require setting up stuff
	}
}

void Control::remove_child_notify(Node *p_child) {

	Control *child_c = Object::cast_to<Control>(p_child);
	if (!child_c)
		return;

	if (child_c->data.theme_owner && child_c->data.theme.is_null()) {
		_propagate_theme_changed(child_c, NULL);
	}
}

void Control::_update_canvas_item_transform() {

	Transform2D xform = _get_internal_transform();
	xform[2] += get_position();

	VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), xform);
}

void Control::_notification(int p_notification) {

	switch (p_notification) {

		case NOTIFICATION_ENTER_TREE: {

			_size_changed();

		} break;
		case NOTIFICATION_EXIT_TREE: {

			get_viewport()->_gui_remove_control(this);

		} break;

		case NOTIFICATION_ENTER_CANVAS: {

			data.parent = Object::cast_to<Control>(get_parent());

			if (is_set_as_toplevel()) {
				data.SI = get_viewport()->_gui_add_subwindow_control(this);

				if (data.theme.is_null() && data.parent && data.parent->data.theme_owner) {
					data.theme_owner = data.parent->data.theme_owner;
					notification(NOTIFICATION_THEME_CHANGED);
				}

			} else {

				Node *parent = this; //meh
				Control *parent_control = NULL;
				bool subwindow = false;

				while (parent) {

					parent = parent->get_parent();

					if (!parent)
						break;

					CanvasItem *ci = Object::cast_to<CanvasItem>(parent);
					if (ci && ci->is_set_as_toplevel()) {
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

				if (parent_control) {
					//do nothing, has a parent control
					if (data.theme.is_null() && parent_control->data.theme_owner) {
						data.theme_owner = parent_control->data.theme_owner;
						notification(NOTIFICATION_THEME_CHANGED);
					}
				} else if (subwindow) {
					//is a subwindow (process input before other controls for that canvas)
					data.SI = get_viewport()->_gui_add_subwindow_control(this);
				} else {
					//is a regular root control
					data.RI = get_viewport()->_gui_add_root_control(this);
				}

				data.parent_canvas_item = get_parent_item();

				if (data.parent_canvas_item) {

					data.parent_canvas_item->connect("item_rect_changed", this, "_size_changed");
				} else {
					//connect viewport
					get_viewport()->connect("size_changed", this, "_size_changed");
				}
			}

			/*
			if (data.theme.is_null() && data.parent && data.parent->data.theme_owner) {
				data.theme_owner=data.parent->data.theme_owner;
				notification(NOTIFICATION_THEME_CHANGED);
			}
			*/

		} break;
		case NOTIFICATION_EXIT_CANVAS: {

			if (data.parent_canvas_item) {

				data.parent_canvas_item->disconnect("item_rect_changed", this, "_size_changed");
				data.parent_canvas_item = NULL;
			} else if (!is_set_as_toplevel()) {
				//disconnect viewport
				get_viewport()->disconnect("size_changed", this, "_size_changed");
			}

			if (data.MI) {
				get_viewport()->_gui_remove_modal_control(data.MI);
				data.MI = NULL;
			}

			if (data.SI) {
				get_viewport()->_gui_remove_subwindow_control(data.SI);
				data.SI = NULL;
			}

			if (data.RI) {
				get_viewport()->_gui_remove_root_control(data.RI);
				data.RI = NULL;
			}

			data.parent = NULL;
			data.parent_canvas_item = NULL;
			/*
			if (data.theme_owner && data.theme.is_null()) {
				data.theme_owner=NULL;
				notification(NOTIFICATION_THEME_CHANGED);
			}
			*/

		} break;
		case NOTIFICATION_MOVED_IN_PARENT: {
			// some parents need to know the order of the childrens to draw (like TabContainer)
			// update if necessary
			if (data.parent)
				data.parent->update();
			update();

			if (data.SI) {
				get_viewport()->_gui_set_subwindow_order_dirty();
			}
			if (data.RI) {
				get_viewport()->_gui_set_root_order_dirty();
			}

		} break;
		case NOTIFICATION_RESIZED: {

			emit_signal(SceneStringNames::get_singleton()->resized);
		} break;
		case NOTIFICATION_DRAW: {

			_update_canvas_item_transform();
			VisualServer::get_singleton()->canvas_item_set_custom_rect(get_canvas_item(), !data.disable_visibility_clip, Rect2(Point2(), get_size()));
			VisualServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), data.clip_contents);
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

			update();
		} break;
		case NOTIFICATION_MODAL_CLOSE: {

			emit_signal("modal_closed");
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			if (!is_visible_in_tree()) {

				if (get_viewport() != NULL)
					get_viewport()->_gui_hid_control(this);

				if (is_inside_tree()) {
					_modal_stack_remove();
					minimum_size_changed();
				}

				//remove key focus
				//remove modalness
			} else {

				_size_changed();
			}

		} break;
		case SceneTree::NOTIFICATION_WM_UNFOCUS_REQUEST: {

			get_viewport()->_gui_unfocus_control(this);

		} break;
	}
}

bool Control::clips_input() const {

	return false;
}
bool Control::has_point(const Point2 &p_point) const {

	if (get_script_instance()) {
		Variant v = p_point;
		const Variant *p = &v;
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->has_point, &p, 1, ce);
		if (ce.error == Variant::CallError::CALL_OK) {
			return ret;
		}
	}
	/*if (has_stylebox("mask")) {
		Ref<StyleBox> mask = get_stylebox("mask");
		return mask->test_mask(p_point,Rect2(Point2(),get_size()));
	}*/
	return Rect2(Point2(), get_size()).has_point(p_point);
}

void Control::set_drag_forwarding(Control *p_target) {

	if (p_target)
		data.drag_owner = p_target->get_instance_id();
	else
		data.drag_owner = 0;
}

Variant Control::get_drag_data(const Point2 &p_point) {

	if (data.drag_owner) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			Control *c = Object::cast_to<Control>(obj);
			return c->call("get_drag_data_fw", p_point, this);
		}
	}

	if (get_script_instance()) {
		Variant v = p_point;
		const Variant *p = &v;
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->get_drag_data, &p, 1, ce);
		if (ce.error == Variant::CallError::CALL_OK)
			return ret;
	}

	return Variant();
}

bool Control::can_drop_data(const Point2 &p_point, const Variant &p_data) const {

	if (data.drag_owner) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			Control *c = Object::cast_to<Control>(obj);
			return c->call("can_drop_data_fw", p_point, p_data, this);
		}
	}

	if (get_script_instance()) {
		Variant v = p_point;
		const Variant *p[2] = { &v, &p_data };
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->can_drop_data, p, 2, ce);
		if (ce.error == Variant::CallError::CALL_OK)
			return ret;
	}

	return Variant();
}
void Control::drop_data(const Point2 &p_point, const Variant &p_data) {

	if (data.drag_owner) {
		Object *obj = ObjectDB::get_instance(data.drag_owner);
		if (obj) {
			Control *c = Object::cast_to<Control>(obj);
			c->call("drop_data_fw", p_point, p_data, this);
			return;
		}
	}

	if (get_script_instance()) {
		Variant v = p_point;
		const Variant *p[2] = { &v, &p_data };
		Variant::CallError ce;
		Variant ret = get_script_instance()->call(SceneStringNames::get_singleton()->drop_data, p, 2, ce);
		if (ce.error == Variant::CallError::CALL_OK)
			return;
	}
}

void Control::force_drag(const Variant &p_data, Control *p_control) {

	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(p_data.get_type() == Variant::NIL);

	get_viewport()->_gui_force_drag(this, p_data, p_control);
}

void Control::set_drag_preview(Control *p_control) {

	ERR_FAIL_COND(!is_inside_tree());
	get_viewport()->_gui_set_drag_preview(this, p_control);
}

bool Control::is_window_modal_on_top() const {

	if (!is_inside_tree())
		return false;

	return get_viewport()->_gui_is_modal_on_top(this);
}

uint64_t Control::get_modal_frame() const {

	return data.modal_frame;
}

Size2 Control::get_minimum_size() const {

	ScriptInstance *si = const_cast<Control *>(this)->get_script_instance();
	if (si) {

		Variant::CallError ce;
		Variant s = si->call(SceneStringNames::get_singleton()->_get_minimum_size, NULL, 0, ce);
		if (ce.error == Variant::CallError::CALL_OK)
			return s;
	}
	return Size2();
}

Ref<Texture> Control::get_icon(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {

		const Ref<Texture> *tex = data.icon_override.getptr(p_name);
		if (tex)
			return *tex;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_icon(p_name, class_name)) {
				return theme_owner->data.theme->get_icon(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->get_icon(p_name, type);
}

Ref<Shader> Control::get_shader(const StringName &p_name, const StringName &p_type) const {
	if (p_type == StringName() || p_type == "") {

		const Ref<Shader> *sdr = data.shader_override.getptr(p_name);
		if (sdr)
			return *sdr;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_shader(p_name, class_name)) {
				return theme_owner->data.theme->get_shader(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->get_shader(p_name, type);
}

Ref<StyleBox> Control::get_stylebox(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		const Ref<StyleBox> *style = data.style_override.getptr(p_name);
		if (style)
			return *style;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	StringName class_name = type;

	while (theme_owner) {

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_stylebox(p_name, class_name)) {
				return theme_owner->data.theme->get_stylebox(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	class_name = type;

	while (class_name != StringName()) {
		if (Theme::get_default()->has_stylebox(p_name, class_name))
			return Theme::get_default()->get_stylebox(p_name, class_name);

		class_name = ClassDB::get_parent_class_nocheck(class_name);
	}
	return Theme::get_default()->get_stylebox(p_name, type);
}
Ref<Font> Control::get_font(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		const Ref<Font> *font = data.font_override.getptr(p_name);
		if (font)
			return *font;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_font(p_name, class_name)) {
				return theme_owner->data.theme->get_font(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		if (theme_owner->data.theme->get_default_theme_font().is_valid())
			return theme_owner->data.theme->get_default_theme_font();
		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->get_font(p_name, type);
}
Color Control::get_color(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		const Color *color = data.color_override.getptr(p_name);
		if (color)
			return *color;
	}

	StringName type = p_type ? p_type : get_class_name();
	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_color(p_name, class_name)) {
				return theme_owner->data.theme->get_color(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->get_color(p_name, type);
}

int Control::get_constant(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		const int *constant = data.constant_override.getptr(p_name);
		if (constant)
			return *constant;
	}

	StringName type = p_type ? p_type : get_class_name();
	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_constant(p_name, class_name)) {
				return theme_owner->data.theme->get_constant(p_name, class_name);
			}

			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->get_constant(p_name, type);
}

bool Control::has_icon_override(const StringName &p_name) const {

	const Ref<Texture> *tex = data.icon_override.getptr(p_name);
	if (tex)
		return true;
	else
		return false;
}

bool Control::has_shader_override(const StringName &p_name) const {

	const Ref<Shader> *sdr = data.shader_override.getptr(p_name);
	if (sdr)
		return true;
	else
		return false;
}

bool Control::has_stylebox_override(const StringName &p_name) const {

	const Ref<StyleBox> *style = data.style_override.getptr(p_name);
	if (style)
		return true;
	else
		return false;
}

bool Control::has_font_override(const StringName &p_name) const {

	const Ref<Font> *font = data.font_override.getptr(p_name);
	if (font)
		return true;
	else
		return false;
}

bool Control::has_color_override(const StringName &p_name) const {

	const Color *color = data.color_override.getptr(p_name);
	if (color)
		return true;
	else
		return false;
}

bool Control::has_constant_override(const StringName &p_name) const {

	const int *constant = data.constant_override.getptr(p_name);
	if (constant)
		return true;
	else
		return false;
}

bool Control::has_icon(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_icon_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_icon(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_icon(p_name, type);
}

bool Control::has_shader(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_shader_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_shader(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_shader(p_name, type);
}
bool Control::has_stylebox(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_stylebox_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_stylebox(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_stylebox(p_name, type);
}
bool Control::has_font(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_font_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_font(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_font(p_name, type);
}

bool Control::has_color(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_color_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_color(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_color(p_name, type);
}

bool Control::has_constant(const StringName &p_name, const StringName &p_type) const {

	if (p_type == StringName() || p_type == "") {
		if (has_constant_override(p_name) == true)
			return true;
	}

	StringName type = p_type ? p_type : get_class_name();

	// try with custom themes
	Control *theme_owner = data.theme_owner;

	while (theme_owner) {

		StringName class_name = type;

		while (class_name != StringName()) {
			if (theme_owner->data.theme->has_constant(p_name, class_name)) {
				return true;
			}
			class_name = ClassDB::get_parent_class_nocheck(class_name);
		}

		Control *parent = Object::cast_to<Control>(theme_owner->get_parent());

		if (parent)
			theme_owner = parent->data.theme_owner;
		else
			theme_owner = NULL;
	}

	return Theme::get_default()->has_constant(p_name, type);
}

Size2 Control::get_parent_area_size() const {

	ERR_FAIL_COND_V(!is_inside_tree(), Size2());

	Size2 parent_size;

	if (data.parent_canvas_item) {

		parent_size = data.parent_canvas_item->_edit_get_rect().size;
	} else {

		parent_size = get_viewport()->get_visible_rect().size;
	}

	return parent_size;
}

void Control::_size_changed() {

	if (!is_inside_tree())
		return;

	Size2 parent_size = get_parent_area_size();

	float margin_pos[4];

	for (int i = 0; i < 4; i++) {

		float area = parent_size[i & 1];
		margin_pos[i] = data.margin[i] + (data.anchor[i] * area);
	}

	Point2 new_pos_cache = Point2(margin_pos[0], margin_pos[1]);
	Size2 new_size_cache = Point2(margin_pos[2], margin_pos[3]) - new_pos_cache;

	Size2 minimum_size = get_combined_minimum_size();

	if (data.h_grow == GROW_DIRECTION_BEGIN) {
		if (minimum_size.width > new_size_cache.width) {
			new_pos_cache.x = new_pos_cache.x + new_size_cache.width - minimum_size.width;
			new_size_cache.width = minimum_size.width;
		}
	} else {
		new_size_cache.width = MAX(minimum_size.width, new_size_cache.width);
	}

	if (data.v_grow == GROW_DIRECTION_BEGIN) {
		if (minimum_size.height > new_size_cache.height) {
			new_pos_cache.y = new_pos_cache.y + new_size_cache.height - minimum_size.height;
			new_size_cache.height = minimum_size.height;
		}
	} else {
		new_size_cache.height = MAX(minimum_size.height, new_size_cache.height);
	}

	if (get_viewport()->is_snap_controls_to_pixels_enabled()) {
		new_size_cache = new_size_cache.floor();
		new_pos_cache = new_pos_cache.floor();
	}
	bool pos_changed = new_pos_cache != data.pos_cache;
	bool size_changed = new_size_cache != data.size_cache;

	data.pos_cache = new_pos_cache;
	data.size_cache = new_size_cache;

	if (size_changed) {
		notification(NOTIFICATION_RESIZED);
	}
	if (pos_changed || size_changed) {
		item_rect_changed(size_changed);
		_change_notify_margins();
		_notify_transform();
	}

	if (pos_changed && !size_changed) {
		_update_canvas_item_transform(); //move because it won't be updated
	}
}

float Control::_get_parent_range(int p_idx) const {

	if (!is_inside_tree()) {

		return 1.0;
	}
	if (data.parent_canvas_item) {

		return data.parent_canvas_item->_edit_get_rect().size[p_idx & 1];
	} else {
		return get_viewport()->get_visible_rect().size[p_idx & 1];
	}

	return 1.0;
}

float Control::_get_range(int p_idx) const {

	p_idx &= 1;

	float parent_range = _get_parent_range(p_idx);
	float from = _a2s(data.margin[p_idx], data.anchor[p_idx], parent_range);
	float to = _a2s(data.margin[p_idx + 2], data.anchor[p_idx + 2], parent_range);

	return to - from;
}

float Control::_s2a(float p_val, float p_anchor, float p_range) const {
	return p_val - (p_anchor * p_range);
}

float Control::_a2s(float p_val, float p_anchor, float p_range) const {
	return Math::floor(p_val + (p_anchor * p_range));
}

void Control::set_anchor(Margin p_margin, float p_anchor, bool p_keep_margin, bool p_push_opposite_anchor) {
	bool pushed = false;
	data.anchor[p_margin] = CLAMP(p_anchor, 0.0, 1.0);

	if (((p_margin == MARGIN_LEFT || p_margin == MARGIN_TOP) && data.anchor[p_margin] > data.anchor[(p_margin + 2) % 4]) ||
			((p_margin == MARGIN_RIGHT || p_margin == MARGIN_BOTTOM) && data.anchor[p_margin] < data.anchor[(p_margin + 2) % 4])) {
		if (p_push_opposite_anchor) {
			data.anchor[(p_margin + 2) % 4] = data.anchor[p_margin];
			pushed = true;
		} else {
			data.anchor[p_margin] = data.anchor[(p_margin + 2) % 4];
		}
	}

	if (is_inside_tree()) {
		if (p_keep_margin) {
			_size_changed();
		}
	}
	update();
	_change_notify();
}

void Control::_set_anchor(Margin p_margin, float p_anchor) {
#ifdef TOOLS_ENABLED
	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint()) {
		set_anchor(p_margin, p_anchor, EDITOR_DEF("editors/2d/keep_margins_when_changing_anchors", false));
	} else {
		set_anchor(p_margin, p_anchor, false);
	}
#else
	set_anchor(p_margin, p_anchor, false);
#endif
}

void Control::set_anchor_and_margin(Margin p_margin, float p_anchor, float p_pos, bool p_push_opposite_anchor) {

	set_anchor(p_margin, p_anchor, false, p_push_opposite_anchor);
	set_margin(p_margin, p_pos);
}

void Control::set_anchors_preset(LayoutPreset p_preset, bool p_keep_margin) {
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
			set_anchor(MARGIN_LEFT, ANCHOR_BEGIN, p_keep_margin);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(MARGIN_LEFT, 0.5, p_keep_margin);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			set_anchor(MARGIN_LEFT, ANCHOR_END, p_keep_margin);
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
			set_anchor(MARGIN_TOP, ANCHOR_BEGIN, p_keep_margin);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(MARGIN_TOP, 0.5, p_keep_margin);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			set_anchor(MARGIN_TOP, ANCHOR_END, p_keep_margin);
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			set_anchor(MARGIN_RIGHT, ANCHOR_BEGIN, p_keep_margin);
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			set_anchor(MARGIN_RIGHT, 0.5, p_keep_margin);
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(MARGIN_RIGHT, ANCHOR_END, p_keep_margin);
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			set_anchor(MARGIN_BOTTOM, ANCHOR_BEGIN, p_keep_margin);
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			set_anchor(MARGIN_BOTTOM, 0.5, p_keep_margin);
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			set_anchor(MARGIN_BOTTOM, ANCHOR_END, p_keep_margin);
			break;
	}
}

void Control::set_margins_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	// Calculate the size if the node is not resized
	Size2 min_size = get_minimum_size();
	Size2 new_size = get_size();
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_HEIGHT) {
		new_size.x = min_size.x;
	}
	if (p_resize_mode == PRESET_MODE_MINSIZE || p_resize_mode == PRESET_MODE_KEEP_WIDTH) {
		new_size.y = min_size.y;
	}

	float pw = _get_parent_range(0);
	float ph = _get_parent_range(1);

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
			data.margin[0] = pw * (0.0 - data.anchor[0]) + p_margin;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.margin[0] = pw * (0.5 - data.anchor[0]) - new_size.x / 2;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_RIGHT_WIDE:
			data.margin[0] = pw * (1.0 - data.anchor[0]) - new_size.x - p_margin;
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
			data.margin[1] = ph * (0.0 - data.anchor[1]) + p_margin;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.margin[1] = ph * (0.5 - data.anchor[1]) - new_size.y / 2;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_BOTTOM_WIDE:
			data.margin[1] = ph * (1.0 - data.anchor[1]) - new_size.y - p_margin;
			break;
	}

	// Right
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_BOTTOM_LEFT:
		case PRESET_CENTER_LEFT:
		case PRESET_LEFT_WIDE:
			data.margin[2] = pw * (0.0 - data.anchor[2]) + new_size.x + p_margin;
			break;

		case PRESET_CENTER_TOP:
		case PRESET_CENTER_BOTTOM:
		case PRESET_CENTER:
		case PRESET_VCENTER_WIDE:
			data.margin[2] = pw * (0.5 - data.anchor[2]) + new_size.x / 2;
			break;

		case PRESET_TOP_RIGHT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_RIGHT:
		case PRESET_TOP_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_HCENTER_WIDE:
		case PRESET_WIDE:
			data.margin[2] = pw * (1.0 - data.anchor[2]) - p_margin;
			break;
	}

	// Bottom
	switch (p_preset) {
		case PRESET_TOP_LEFT:
		case PRESET_TOP_RIGHT:
		case PRESET_CENTER_TOP:
		case PRESET_TOP_WIDE:
			data.margin[3] = ph * (0.0 - data.anchor[3]) + new_size.y + p_margin;
			break;

		case PRESET_CENTER_LEFT:
		case PRESET_CENTER_RIGHT:
		case PRESET_CENTER:
		case PRESET_HCENTER_WIDE:
			data.margin[3] = ph * (0.5 - data.anchor[3]) + new_size.y / 2;
			break;

		case PRESET_BOTTOM_LEFT:
		case PRESET_BOTTOM_RIGHT:
		case PRESET_CENTER_BOTTOM:
		case PRESET_LEFT_WIDE:
		case PRESET_RIGHT_WIDE:
		case PRESET_BOTTOM_WIDE:
		case PRESET_VCENTER_WIDE:
		case PRESET_WIDE:
			data.margin[3] = ph * (1.0 - data.anchor[3]) - p_margin;
			break;
	}

	_size_changed();
}

void Control::set_anchors_and_margins_preset(LayoutPreset p_preset, LayoutPresetMode p_resize_mode, int p_margin) {
	set_anchors_preset(p_preset);
	set_margins_preset(p_preset, p_resize_mode, p_margin);
}

float Control::get_anchor(Margin p_margin) const {

	return data.anchor[p_margin];
}

void Control::_change_notify_margins() {

	// this avoids sending the whole object data again on a change
	_change_notify("margin_left");
	_change_notify("margin_top");
	_change_notify("margin_right");
	_change_notify("margin_bottom");
	_change_notify("rect_position");
	_change_notify("rect_size");
}

void Control::set_margin(Margin p_margin, float p_value) {

	data.margin[p_margin] = p_value;
	_size_changed();
}

void Control::set_begin(const Size2 &p_point) {

	data.margin[0] = p_point.x;
	data.margin[1] = p_point.y;
	_size_changed();
}

void Control::set_end(const Size2 &p_point) {

	data.margin[2] = p_point.x;
	data.margin[3] = p_point.y;
	_size_changed();
}

float Control::get_margin(Margin p_margin) const {

	return data.margin[p_margin];
}

Size2 Control::get_begin() const {

	return Size2(data.margin[0], data.margin[1]);
}
Size2 Control::get_end() const {

	return Size2(data.margin[2], data.margin[3]);
}

Point2 Control::get_global_position() const {

	return get_global_transform().get_origin();
}

void Control::set_global_position(const Point2 &p_point) {

	Transform2D inv;

	if (data.parent_canvas_item) {

		inv = data.parent_canvas_item->get_global_transform().affine_inverse();
	}

	set_position(inv.xform(p_point));
}

void Control::set_position(const Size2 &p_point) {

	float pw = _get_parent_range(0);
	float ph = _get_parent_range(1);

	float x = _a2s(data.margin[0], data.anchor[0], pw);
	float y = _a2s(data.margin[1], data.anchor[1], ph);
	float x2 = _a2s(data.margin[2], data.anchor[2], pw);
	float y2 = _a2s(data.margin[3], data.anchor[3], ph);

	Size2 ret = Size2(x2 - x, y2 - y);
	Size2 min = get_combined_minimum_size();

	Size2 size = Size2(MAX(min.width, ret.width), MAX(min.height, ret.height));
	float w = size.x;
	float h = size.y;

	x = p_point.x;
	y = p_point.y;

	data.margin[0] = _s2a(x, data.anchor[0], pw);
	data.margin[1] = _s2a(y, data.anchor[1], ph);
	data.margin[2] = _s2a(x + w, data.anchor[2], pw);
	data.margin[3] = _s2a(y + h, data.anchor[3], ph);

	_size_changed();
}

void Control::set_size(const Size2 &p_size) {

	Size2 new_size = p_size;
	Size2 min = get_combined_minimum_size();
	if (new_size.x < min.x)
		new_size.x = min.x;
	if (new_size.y < min.y)
		new_size.y = min.y;

	float pw = _get_parent_range(0);
	float ph = _get_parent_range(1);

	float x = _a2s(data.margin[0], data.anchor[0], pw);
	float y = _a2s(data.margin[1], data.anchor[1], ph);

	float w = new_size.width;
	float h = new_size.height;

	data.margin[2] = _s2a(x + w, data.anchor[2], pw);
	data.margin[3] = _s2a(y + h, data.anchor[3], ph);

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

Rect2 Control::get_window_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2 gr = get_global_rect();
	gr.position += get_viewport()->get_visible_rect().position;
	return gr;
}

Rect2 Control::get_rect() const {

	return Rect2(get_position(), get_size());
}

void Control::add_icon_override(const StringName &p_name, const Ref<Texture> &p_icon) {

	ERR_FAIL_COND(p_icon.is_null());
	data.icon_override[p_name] = p_icon;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}

void Control::add_shader_override(const StringName &p_name, const Ref<Shader> &p_shader) {
	ERR_FAIL_COND(p_shader.is_null());
	data.shader_override[p_name] = p_shader;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}
void Control::add_style_override(const StringName &p_name, const Ref<StyleBox> &p_style) {

	ERR_FAIL_COND(p_style.is_null());
	data.style_override[p_name] = p_style;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}

void Control::add_font_override(const StringName &p_name, const Ref<Font> &p_font) {

	ERR_FAIL_COND(p_font.is_null());
	if (data.font_override.has(p_name)) {
		_unref_font(data.font_override[p_name]);
	}
	data.font_override[p_name] = p_font;

	if (p_font.is_valid()) {
		_ref_font(p_font);
	}

	notification(NOTIFICATION_THEME_CHANGED);
	update();
}
void Control::add_color_override(const StringName &p_name, const Color &p_color) {

	data.color_override[p_name] = p_color;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}
void Control::add_constant_override(const StringName &p_name, int p_constant) {

	data.constant_override[p_name] = p_constant;
	notification(NOTIFICATION_THEME_CHANGED);
	update();
}

void Control::set_focus_mode(FocusMode p_focus_mode) {

	if (is_inside_tree() && p_focus_mode == FOCUS_NONE && data.focus_mode != FOCUS_NONE && has_focus())
		release_focus();

	data.focus_mode = p_focus_mode;
}

static Control *_next_control(Control *p_from) {

	if (p_from->is_set_as_toplevel())
		return NULL; // can't go above

	Control *parent = Object::cast_to<Control>(p_from->get_parent());

	if (!parent) {

		return NULL;
	}

	int next = p_from->get_position_in_parent();
	ERR_FAIL_INDEX_V(next, parent->get_child_count(), NULL);
	for (int i = (next + 1); i < parent->get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(parent->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_toplevel())
			continue;

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
			if (n) {
				from = Object::cast_to<Control>(n);

				if (!from) {

					ERR_EXPLAIN("Next focus node is not a control: " + n->get_name());
					ERR_FAIL_V(NULL);
				}
			} else {
				return NULL;
			}
			if (from->is_visible() && from->get_focus_mode() != FOCUS_NONE)
				return from;
		}

		// find next child

		Control *next_child = NULL;

		for (int i = 0; i < from->get_child_count(); i++) {

			Control *c = Object::cast_to<Control>(from->get_child(i));
			if (!c || !c->is_visible_in_tree() || c->is_set_as_toplevel()) {
				continue;
			}

			next_child = c;
			break;
		}

		if (next_child) {

			from = next_child;
		} else {

			next_child = _next_control(from);
			if (!next_child) { //nothing else.. go up and find either window or subwindow
				next_child = const_cast<Control *>(this);
				while (next_child && !next_child->is_set_as_toplevel()) {

					next_child = cast_to<Control>(next_child->get_parent());
				}

				if (!next_child) {

					next_child = const_cast<Control *>(this);
					while (next_child) {

						if (next_child->data.SI || next_child->data.RI)
							break;
						next_child = next_child->get_parent_control();
					}
				}
			}
		}

		if (next_child == this) // no next control->
			return (get_focus_mode() == FOCUS_ALL) ? next_child : NULL;
		if (next_child) {
			if (next_child->get_focus_mode() == FOCUS_ALL)
				return next_child;
			from = next_child;
		} else
			break;
	}

	return NULL;
}

static Control *_prev_control(Control *p_from) {

	Control *child = NULL;
	for (int i = p_from->get_child_count() - 1; i >= 0; i--) {

		Control *c = Object::cast_to<Control>(p_from->get_child(i));
		if (!c || !c->is_visible_in_tree() || c->is_set_as_toplevel())
			continue;

		child = c;
		break;
	}

	if (!child)
		return p_from;

	//no prev in parent, try the same in parent
	return _prev_control(child);
}

Control *Control::find_prev_valid_focus() const {
	Control *from = const_cast<Control *>(this);

	while (true) {

		// If the focus property is manually overwritten, attempt to use it.

		if (!data.focus_prev.is_empty()) {
			Node *n = get_node(data.focus_prev);
			if (n) {
				from = Object::cast_to<Control>(n);

				if (!from) {

					ERR_EXPLAIN("Prev focus node is not a control: " + n->get_name());
					ERR_FAIL_V(NULL);
				}
			} else {
				return NULL;
			}
			if (from->is_visible() && from->get_focus_mode() != FOCUS_NONE)
				return from;
		}

		// find prev child

		Control *prev_child = NULL;

		if (from->is_set_as_toplevel() || !Object::cast_to<Control>(from->get_parent())) {

			//find last of the childs

			prev_child = _prev_control(from);

		} else {

			for (int i = (from->get_position_in_parent() - 1); i >= 0; i--) {

				Control *c = Object::cast_to<Control>(from->get_parent()->get_child(i));

				if (!c || !c->is_visible_in_tree() || c->is_set_as_toplevel()) {
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

		if (prev_child == this) // no prev control->
			return (get_focus_mode() == FOCUS_ALL) ? prev_child : NULL;

		if (prev_child->get_focus_mode() == FOCUS_ALL)
			return prev_child;

		from = prev_child;
	}

	return NULL;
}

Control::FocusMode Control::get_focus_mode() const {

	return data.focus_mode;
}
bool Control::has_focus() const {

	return is_inside_tree() && get_viewport()->_gui_control_has_focus(this);
}

void Control::grab_focus() {

	if (!is_inside_tree()) {
		ERR_FAIL_COND(!is_inside_tree());
	}
	if (data.focus_mode == FOCUS_NONE)
		return;

	get_viewport()->_gui_control_grab_focus(this);
}

void Control::release_focus() {

	ERR_FAIL_COND(!is_inside_tree());

	if (!has_focus())
		return;

	get_viewport()->_gui_remove_focus();
	update();
}

bool Control::is_toplevel_control() const {

	return is_inside_tree() && (!data.parent_canvas_item && !data.RI && is_set_as_toplevel());
}

void Control::show_modal(bool p_exclusive) {

	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND(!data.SI);

	if (is_visible_in_tree())
		hide();

	ERR_FAIL_COND(data.MI != NULL);
	show();
	raise();
	data.modal_exclusive = p_exclusive;
	data.MI = get_viewport()->_gui_show_modal(this);
	data.modal_frame = Engine::get_singleton()->get_frames_drawn();
}

void Control::_modal_set_prev_focus_owner(ObjectID p_prev) {
	data.modal_prev_focus_owner = p_prev;
}

void Control::_modal_stack_remove() {

	ERR_FAIL_COND(!is_inside_tree());

	if (!data.MI)
		return;

	get_viewport()->_gui_remove_from_modal_stack(data.MI, data.modal_prev_focus_owner);

	data.MI = NULL;
	data.modal_prev_focus_owner = 0;
}

void Control::_propagate_theme_changed(CanvasItem *p_at, Control *p_owner, bool p_assign) {

	Control *c = Object::cast_to<Control>(p_at);

	if (c && c != p_owner && c->data.theme.is_valid()) // has a theme, this can't be propagated
		return;

	for (int i = 0; i < p_at->get_child_count(); i++) {

		CanvasItem *child = Object::cast_to<CanvasItem>(p_at->get_child(i));
		if (child) {
			_propagate_theme_changed(child, p_owner, p_assign);
		}
	}

	if (c) {

		if (p_assign) {
			c->data.theme_owner = p_owner;
		}
		c->notification(NOTIFICATION_THEME_CHANGED);
		c->update();
	}
}

void Control::_theme_changed() {

	_propagate_theme_changed(this, this, false);
}

void Control::set_theme(const Ref<Theme> &p_theme) {

	if (data.theme == p_theme)
		return;

	if (data.theme.is_valid()) {
		data.theme->disconnect("changed", this, "_theme_changed");
	}

	data.theme = p_theme;
	if (!p_theme.is_null()) {

		_propagate_theme_changed(this, this);
	} else {

		Control *parent = cast_to<Control>(get_parent());
		if (parent && parent->data.theme_owner) {
			_propagate_theme_changed(this, parent->data.theme_owner);
		} else {

			_propagate_theme_changed(this, NULL);
		}
	}

	if (data.theme.is_valid()) {
		data.theme->connect("changed", this, "_theme_changed");
	}
}

void Control::accept_event() {

	if (is_inside_tree())
		get_viewport()->_gui_accept_event();
}

Ref<Theme> Control::get_theme() const {

	return data.theme;
}

void Control::set_tooltip(const String &p_tooltip) {

	data.tooltip = p_tooltip;
}
String Control::get_tooltip(const Point2 &p_pos) const {

	return data.tooltip;
}

void Control::set_default_cursor_shape(CursorShape p_shape) {

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

void Control::set_focus_neighbour(Margin p_margin, const NodePath &p_neighbour) {

	ERR_FAIL_INDEX(p_margin, 4);
	data.focus_neighbour[p_margin] = p_neighbour;
}

NodePath Control::get_focus_neighbour(Margin p_margin) const {

	ERR_FAIL_INDEX_V(p_margin, 4, NodePath());
	return data.focus_neighbour[p_margin];
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

#define MAX_NEIGHBOUR_SEARCH_COUNT 512

Control *Control::_get_focus_neighbour(Margin p_margin, int p_count) {

	if (p_count >= MAX_NEIGHBOUR_SEARCH_COUNT)
		return NULL;
	if (!data.focus_neighbour[p_margin].is_empty()) {

		Control *c = NULL;
		Node *n = get_node(data.focus_neighbour[p_margin]);
		if (n) {
			c = Object::cast_to<Control>(n);

			if (!c) {

				ERR_EXPLAIN("Neighbour focus node is not a control: " + n->get_name());
				ERR_FAIL_V(NULL);
			}
		} else {
			return NULL;
		}
		bool valid = true;
		if (!c->is_visible())
			valid = false;
		if (c->get_focus_mode() == FOCUS_NONE)
			valid = false;
		if (valid)
			return c;

		c = c->_get_focus_neighbour(p_margin, p_count + 1);
		return c;
	}

	float dist = 1e7;
	Control *result = NULL;

	Point2 points[4];

	Transform2D xform = get_global_transform();
	Rect2 rect = _edit_get_rect();

	points[0] = xform.xform(rect.position);
	points[1] = xform.xform(rect.position + Point2(rect.size.x, 0));
	points[2] = xform.xform(rect.position + rect.size);
	points[3] = xform.xform(rect.position + Point2(0, rect.size.y));

	const Vector2 dir[4] = {
		Vector2(-1, 0),
		Vector2(0, -1),
		Vector2(1, 0),
		Vector2(0, 1)
	};

	Vector2 vdir = dir[p_margin];

	float maxd = -1e7;

	for (int i = 0; i < 4; i++) {

		float d = vdir.dot(points[i]);
		if (d > maxd)
			maxd = d;
	}

	Node *base = this;

	while (base) {

		Control *c = Object::cast_to<Control>(base);
		if (c) {
			if (c->data.SI)
				break;
			if (c->data.RI)
				break;
		}
		base = base->get_parent();
	}

	if (!base)
		return NULL;

	_window_find_focus_neighbour(vdir, base, points, maxd, dist, &result);

	return result;
}

void Control::_window_find_focus_neighbour(const Vector2 &p_dir, Node *p_at, const Point2 *p_points, float p_min, float &r_closest_dist, Control **r_closest) {

	if (Object::cast_to<Viewport>(p_at))
		return; //bye

	Control *c = Object::cast_to<Control>(p_at);

	if (c && c != this && c->get_focus_mode() == FOCUS_ALL && c->is_visible_in_tree()) {

		Point2 points[4];

		Transform2D xform = c->get_global_transform();
		Rect2 rect = c->_edit_get_rect();

		points[0] = xform.xform(rect.position);
		points[1] = xform.xform(rect.position + Point2(rect.size.x, 0));
		points[2] = xform.xform(rect.position + rect.size);
		points[3] = xform.xform(rect.position + Point2(0, rect.size.y));

		float min = 1e7;

		for (int i = 0; i < 4; i++) {

			float d = p_dir.dot(points[i]);
			if (d < min)
				min = d;
		}

		if (min > (p_min - CMP_EPSILON)) {

			for (int i = 0; i < 4; i++) {

				Vector2 la = p_points[i];
				Vector2 lb = p_points[(i + 1) % 4];

				for (int j = 0; j < 4; j++) {

					Vector2 fa = points[j];
					Vector2 fb = points[(j + 1) % 4];

					Vector2 pa, pb;
					float d = Geometry::get_closest_points_between_segments(la, lb, fa, fb, pa, pb);
					//float d = Geometry::get_closest_distance_between_segments(Vector3(la.x,la.y,0),Vector3(lb.x,lb.y,0),Vector3(fa.x,fa.y,0),Vector3(fb.x,fb.y,0));
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
		if (childc && childc->data.SI)
			continue; //subwindow, ignore
		_window_find_focus_neighbour(p_dir, p_at->get_child(i), p_points, p_min, r_closest_dist, r_closest);
	}
}

void Control::set_h_size_flags(int p_flags) {

	if (data.h_size_flags == p_flags)
		return;
	data.h_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

int Control::get_h_size_flags() const {
	return data.h_size_flags;
}
void Control::set_v_size_flags(int p_flags) {

	if (data.v_size_flags == p_flags)
		return;
	data.v_size_flags = p_flags;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

void Control::set_stretch_ratio(float p_ratio) {

	if (data.expand == p_ratio)
		return;

	data.expand = p_ratio;
	emit_signal(SceneStringNames::get_singleton()->size_flags_changed);
}

float Control::get_stretch_ratio() const {

	return data.expand;
}

void Control::grab_click_focus() {

	ERR_FAIL_COND(!is_inside_tree());

	get_viewport()->_gui_grab_click_focus(this);
}

void Control::minimum_size_changed() {

	if (!is_inside_tree() || data.block_minimum_size_adjust)
		return;

	if (data.pending_min_size_update)
		return;

	data.pending_min_size_update = true;
	MessageQueue::get_singleton()->push_call(this, "_update_minimum_size");

	if (!is_toplevel_control()) {
		Control *pc = get_parent_control();
		if (pc)
			pc->minimum_size_changed();
	}
}

int Control::get_v_size_flags() const {
	return data.v_size_flags;
}

void Control::set_mouse_filter(MouseFilter p_filter) {

	ERR_FAIL_INDEX(p_filter, 3);
	data.mouse_filter = p_filter;
}

Control::MouseFilter Control::get_mouse_filter() const {

	return data.mouse_filter;
}

void Control::set_pass_on_modal_close_click(bool p_pass_on) {

	data.pass_on_modal_close_click = p_pass_on;
}

bool Control::pass_on_modal_close_click() const {

	return data.pass_on_modal_close_click;
}

Control *Control::get_focus_owner() const {

	ERR_FAIL_COND_V(!is_inside_tree(), NULL);
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
        Variant::CallError ce;
        Variant ret = get_script_instance()->call("is_text_field",p,2,ce);
        if (ce.error==Variant::CallError::CALL_OK)
            return ret;
    }
  */
	return false;
}

void Control::set_rotation(float p_radians) {

	data.rotation = p_radians;
	update();
	_notify_transform();
	_change_notify("rect_rotation");
}

float Control::get_rotation() const {

	return data.rotation;
}

void Control::set_rotation_degrees(float p_degrees) {
	set_rotation(Math::deg2rad(p_degrees));
}

float Control::get_rotation_degrees() const {
	return Math::rad2deg(get_rotation());
}

//needed to update the control if the font changes..
void Control::_ref_font(Ref<Font> p_sc) {

	if (!data.font_refcount.has(p_sc)) {
		data.font_refcount[p_sc] = 1;
		p_sc->connect("changed", this, "_font_changed");
	} else {
		data.font_refcount[p_sc] += 1;
	}
}

void Control::_unref_font(Ref<Font> p_sc) {

	ERR_FAIL_COND(!data.font_refcount.has(p_sc));
	data.font_refcount[p_sc]--;
	if (data.font_refcount[p_sc] == 0) {
		p_sc->disconnect("changed", this, "_font_changed");
		data.font_refcount.erase(p_sc);
	}
}

void Control::_font_changed() {

	update();
	notification(NOTIFICATION_THEME_CHANGED);
	minimum_size_changed(); //fonts affect minimum size pretty much almost always
}

void Control::set_pivot_offset(const Vector2 &p_pivot) {

	data.pivot_offset = p_pivot;
	update();
	_notify_transform();
	_change_notify("rect_pivot_offset");
}

Vector2 Control::get_pivot_offset() const {

	return data.pivot_offset;
}

void Control::set_scale(const Vector2 &p_scale) {

	data.scale = p_scale;
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

			if (c->data.RI || c->data.MI || c->is_toplevel_control())
				break;
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
		if (pf == "add_color_override" || pf == "has_color" || pf == "has_color_override" || pf == "get_color") {
			Theme::get_default()->get_color_list(get_class(), &sn);
		} else if (pf == "add_style_override" || pf == "has_style" || pf == "has_style_override" || pf == "get_style") {
			Theme::get_default()->get_stylebox_list(get_class(), &sn);
		} else if (pf == "add_font_override" || pf == "has_font" || pf == "has_font_override" || pf == "get_font") {
			Theme::get_default()->get_font_list(get_class(), &sn);
		} else if (pf == "add_constant_override" || pf == "has_constant" || pf == "has_constant_override" || pf == "get_constant") {
			Theme::get_default()->get_constant_list(get_class(), &sn);
		}

		sn.sort_custom<StringName::AlphCompare>();
		for (List<StringName>::Element *E = sn.front(); E; E = E->next()) {
			r_options->push_back("\"" + E->get() + "\"");
		}
	}
}
void Control::set_clip_contents(bool p_clip) {

	data.clip_contents = p_clip;
	update();
}

bool Control::is_clipping_contents() {

	return data.clip_contents;
}

void Control::set_h_grow_direction(GrowDirection p_direction) {

	data.h_grow = p_direction;
	_size_changed();
}

Control::GrowDirection Control::get_h_grow_direction() const {

	return data.h_grow;
}

void Control::set_v_grow_direction(GrowDirection p_direction) {

	data.v_grow = p_direction;
	_size_changed();
}
Control::GrowDirection Control::get_v_grow_direction() const {

	return data.v_grow;
}

void Control::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("_window_resize_event"),&Control::_window_resize_event);
	ClassDB::bind_method(D_METHOD("_size_changed"), &Control::_size_changed);
	ClassDB::bind_method(D_METHOD("_update_minimum_size"), &Control::_update_minimum_size);

	ClassDB::bind_method(D_METHOD("accept_event"), &Control::accept_event);
	ClassDB::bind_method(D_METHOD("get_minimum_size"), &Control::get_minimum_size);
	ClassDB::bind_method(D_METHOD("get_combined_minimum_size"), &Control::get_combined_minimum_size);
	ClassDB::bind_method(D_METHOD("set_anchors_preset", "preset", "keep_margin"), &Control::set_anchors_preset, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_margins_preset", "preset", "resize_mode", "margin"), &Control::set_margins_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_anchors_and_margins_preset", "preset", "resize_mode", "margin"), &Control::set_anchors_and_margins_preset, DEFVAL(PRESET_MODE_MINSIZE), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_anchor", "margin", "anchor", "keep_margin", "push_opposite_anchor"), &Control::set_anchor, DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("_set_anchor", "margin", "anchor"), &Control::_set_anchor);
	ClassDB::bind_method(D_METHOD("get_anchor", "margin"), &Control::get_anchor);
	ClassDB::bind_method(D_METHOD("set_margin", "margin", "offset"), &Control::set_margin);
	ClassDB::bind_method(D_METHOD("set_anchor_and_margin", "margin", "anchor", "offset", "push_opposite_anchor"), &Control::set_anchor_and_margin, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_begin", "position"), &Control::set_begin);
	ClassDB::bind_method(D_METHOD("set_end", "position"), &Control::set_end);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &Control::set_position);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &Control::set_size);
	ClassDB::bind_method(D_METHOD("set_custom_minimum_size", "size"), &Control::set_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("set_global_position", "position"), &Control::set_global_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "radians"), &Control::set_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "degrees"), &Control::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Control::set_scale);
	ClassDB::bind_method(D_METHOD("set_pivot_offset", "pivot_offset"), &Control::set_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_margin", "margin"), &Control::get_margin);
	ClassDB::bind_method(D_METHOD("get_begin"), &Control::get_begin);
	ClassDB::bind_method(D_METHOD("get_end"), &Control::get_end);
	ClassDB::bind_method(D_METHOD("get_position"), &Control::get_position);
	ClassDB::bind_method(D_METHOD("get_size"), &Control::get_size);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Control::get_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &Control::get_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_scale"), &Control::get_scale);
	ClassDB::bind_method(D_METHOD("get_pivot_offset"), &Control::get_pivot_offset);
	ClassDB::bind_method(D_METHOD("get_custom_minimum_size"), &Control::get_custom_minimum_size);
	ClassDB::bind_method(D_METHOD("get_parent_area_size"), &Control::get_size);
	ClassDB::bind_method(D_METHOD("get_global_position"), &Control::get_global_position);
	ClassDB::bind_method(D_METHOD("get_rect"), &Control::get_rect);
	ClassDB::bind_method(D_METHOD("get_global_rect"), &Control::get_global_rect);
	ClassDB::bind_method(D_METHOD("show_modal", "exclusive"), &Control::show_modal, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_focus_mode", "mode"), &Control::set_focus_mode);
	ClassDB::bind_method(D_METHOD("get_focus_mode"), &Control::get_focus_mode);
	ClassDB::bind_method(D_METHOD("has_focus"), &Control::has_focus);
	ClassDB::bind_method(D_METHOD("grab_focus"), &Control::grab_focus);
	ClassDB::bind_method(D_METHOD("release_focus"), &Control::release_focus);
	ClassDB::bind_method(D_METHOD("get_focus_owner"), &Control::get_focus_owner);

	ClassDB::bind_method(D_METHOD("set_h_size_flags", "flags"), &Control::set_h_size_flags);
	ClassDB::bind_method(D_METHOD("get_h_size_flags"), &Control::get_h_size_flags);

	ClassDB::bind_method(D_METHOD("set_stretch_ratio", "ratio"), &Control::set_stretch_ratio);
	ClassDB::bind_method(D_METHOD("get_stretch_ratio"), &Control::get_stretch_ratio);

	ClassDB::bind_method(D_METHOD("set_v_size_flags", "flags"), &Control::set_v_size_flags);
	ClassDB::bind_method(D_METHOD("get_v_size_flags"), &Control::get_v_size_flags);

	ClassDB::bind_method(D_METHOD("set_theme", "theme"), &Control::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme"), &Control::get_theme);

	ClassDB::bind_method(D_METHOD("add_icon_override", "name", "texture"), &Control::add_icon_override);
	ClassDB::bind_method(D_METHOD("add_shader_override", "name", "shader"), &Control::add_shader_override);
	ClassDB::bind_method(D_METHOD("add_style_override", "name", "stylebox"), &Control::add_style_override);
	ClassDB::bind_method(D_METHOD("add_font_override", "name", "font"), &Control::add_font_override);
	ClassDB::bind_method(D_METHOD("add_color_override", "name", "color"), &Control::add_color_override);
	ClassDB::bind_method(D_METHOD("add_constant_override", "name", "constant"), &Control::add_constant_override);

	ClassDB::bind_method(D_METHOD("get_icon", "name", "type"), &Control::get_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_stylebox", "name", "type"), &Control::get_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_font", "name", "type"), &Control::get_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_color", "name", "type"), &Control::get_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_constant", "name", "type"), &Control::get_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("has_icon_override", "name"), &Control::has_icon_override);
	ClassDB::bind_method(D_METHOD("has_stylebox_override", "name"), &Control::has_stylebox_override);
	ClassDB::bind_method(D_METHOD("has_font_override", "name"), &Control::has_font_override);
	ClassDB::bind_method(D_METHOD("has_color_override", "name"), &Control::has_color_override);
	ClassDB::bind_method(D_METHOD("has_constant_override", "name"), &Control::has_constant_override);

	ClassDB::bind_method(D_METHOD("has_icon", "name", "type"), &Control::has_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_stylebox", "name", "type"), &Control::has_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_font", "name", "type"), &Control::has_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_color", "name", "type"), &Control::has_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_constant", "name", "type"), &Control::has_constant, DEFVAL(""));

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

	ClassDB::bind_method(D_METHOD("set_focus_neighbour", "margin", "neighbour"), &Control::set_focus_neighbour);
	ClassDB::bind_method(D_METHOD("get_focus_neighbour", "margin"), &Control::get_focus_neighbour);

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

	ClassDB::bind_method(D_METHOD("_theme_changed"), &Control::_theme_changed);

	ClassDB::bind_method(D_METHOD("_font_changed"), &Control::_font_changed);

	BIND_VMETHOD(MethodInfo("_gui_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	BIND_VMETHOD(MethodInfo(Variant::VECTOR2, "_get_minimum_size"));
	BIND_VMETHOD(MethodInfo(Variant::OBJECT, "get_drag_data", PropertyInfo(Variant::VECTOR2, "position")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "can_drop_data", PropertyInfo(Variant::VECTOR2, "position"), PropertyInfo(Variant::NIL, "data")));
	BIND_VMETHOD(MethodInfo("drop_data", PropertyInfo(Variant::VECTOR2, "position"), PropertyInfo(Variant::NIL, "data")));

	ADD_GROUP("Anchor", "anchor_");
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anchor_left", PROPERTY_HINT_RANGE, "0,1,0.01"), "_set_anchor", "get_anchor", MARGIN_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anchor_top", PROPERTY_HINT_RANGE, "0,1,0.01"), "_set_anchor", "get_anchor", MARGIN_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anchor_right", PROPERTY_HINT_RANGE, "0,1,0.01"), "_set_anchor", "get_anchor", MARGIN_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::REAL, "anchor_bottom", PROPERTY_HINT_RANGE, "0,1,0.01"), "_set_anchor", "get_anchor", MARGIN_BOTTOM);

	ADD_GROUP("Margin", "margin_");
	ADD_PROPERTYINZ(PropertyInfo(Variant::INT, "margin_left", PROPERTY_HINT_RANGE, "-4096,4096"), "set_margin", "get_margin", MARGIN_LEFT);
	ADD_PROPERTYINZ(PropertyInfo(Variant::INT, "margin_top", PROPERTY_HINT_RANGE, "-4096,4096"), "set_margin", "get_margin", MARGIN_TOP);
	ADD_PROPERTYINZ(PropertyInfo(Variant::INT, "margin_right", PROPERTY_HINT_RANGE, "-4096,4096"), "set_margin", "get_margin", MARGIN_RIGHT);
	ADD_PROPERTYINZ(PropertyInfo(Variant::INT, "margin_bottom", PROPERTY_HINT_RANGE, "-4096,4096"), "set_margin", "get_margin", MARGIN_BOTTOM);

	ADD_GROUP("Grow Direction", "grow_");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "grow_horizontal", PROPERTY_HINT_ENUM, "Begin,End"), "set_h_grow_direction", "get_h_grow_direction");
	ADD_PROPERTYNO(PropertyInfo(Variant::INT, "grow_vertical", PROPERTY_HINT_ENUM, "Begin,End"), "set_v_grow_direction", "get_v_grow_direction");

	ADD_GROUP("Rect", "rect_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "rect_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_position", "get_position");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "rect_size", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_size", "get_size");
	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "rect_min_size"), "set_custom_minimum_size", "get_custom_minimum_size");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "rect_rotation", PROPERTY_HINT_RANGE, "-1080,1080,0.01"), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTYNO(PropertyInfo(Variant::VECTOR2, "rect_scale"), "set_scale", "get_scale");
	ADD_PROPERTYNO(PropertyInfo(Variant::VECTOR2, "rect_pivot_offset"), "set_pivot_offset", "get_pivot_offset");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "rect_clip_content"), "set_clip_contents", "is_clipping_contents");

	ADD_GROUP("Hint", "hint_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "hint_tooltip", PROPERTY_HINT_MULTILINE_TEXT), "set_tooltip", "_get_tooltip");

	ADD_GROUP("Focus", "focus_");
	ADD_PROPERTYINZ(PropertyInfo(Variant::NODE_PATH, "focus_neighbour_left"), "set_focus_neighbour", "get_focus_neighbour", MARGIN_LEFT);
	ADD_PROPERTYINZ(PropertyInfo(Variant::NODE_PATH, "focus_neighbour_top"), "set_focus_neighbour", "get_focus_neighbour", MARGIN_TOP);
	ADD_PROPERTYINZ(PropertyInfo(Variant::NODE_PATH, "focus_neighbour_right"), "set_focus_neighbour", "get_focus_neighbour", MARGIN_RIGHT);
	ADD_PROPERTYINZ(PropertyInfo(Variant::NODE_PATH, "focus_neighbour_bottom"), "set_focus_neighbour", "get_focus_neighbour", MARGIN_BOTTOM);
	ADD_PROPERTYNZ(PropertyInfo(Variant::NODE_PATH, "focus_next"), "set_focus_next", "get_focus_next");
	ADD_PROPERTYNZ(PropertyInfo(Variant::NODE_PATH, "focus_previous"), "set_focus_previous", "get_focus_previous");

	ADD_GROUP("Mouse", "mouse_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mouse_filter", PROPERTY_HINT_ENUM, "Stop,Pass,Ignore"), "set_mouse_filter", "get_mouse_filter");

	ADD_GROUP("Size Flags", "size_flags_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_horizontal", PROPERTY_HINT_FLAGS, "Fill,Expand,Shrink Center,Shrink End"), "set_h_size_flags", "get_h_size_flags");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "size_flags_vertical", PROPERTY_HINT_FLAGS, "Fill,Expand,Shrink Center,Shrink End"), "set_v_size_flags", "get_v_size_flags");
	ADD_PROPERTYNO(PropertyInfo(Variant::REAL, "size_flags_stretch_ratio", PROPERTY_HINT_RANGE, "0,128,0.01"), "set_stretch_ratio", "get_stretch_ratio");
	ADD_GROUP("Theme", "");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "Theme"), "set_theme", "get_theme");
	ADD_GROUP("", "");

	BIND_ENUM_CONSTANT(FOCUS_NONE);
	BIND_ENUM_CONSTANT(FOCUS_CLICK);
	BIND_ENUM_CONSTANT(FOCUS_ALL);

	BIND_CONSTANT(NOTIFICATION_RESIZED);
	BIND_CONSTANT(NOTIFICATION_MOUSE_ENTER);
	BIND_CONSTANT(NOTIFICATION_MOUSE_EXIT);
	BIND_CONSTANT(NOTIFICATION_FOCUS_ENTER);
	BIND_CONSTANT(NOTIFICATION_FOCUS_EXIT);
	BIND_CONSTANT(NOTIFICATION_THEME_CHANGED);
	BIND_CONSTANT(NOTIFICATION_MODAL_CLOSE);

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

	BIND_ENUM_CONSTANT(ANCHOR_BEGIN);
	BIND_ENUM_CONSTANT(ANCHOR_END);

	ADD_SIGNAL(MethodInfo("resized"));
	ADD_SIGNAL(MethodInfo("gui_input", PropertyInfo(Variant::OBJECT, "ev", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));
	ADD_SIGNAL(MethodInfo("focus_entered"));
	ADD_SIGNAL(MethodInfo("focus_exited"));
	ADD_SIGNAL(MethodInfo("size_flags_changed"));
	ADD_SIGNAL(MethodInfo("minimum_size_changed"));
	ADD_SIGNAL(MethodInfo("modal_closed"));

	BIND_VMETHOD(MethodInfo(Variant::BOOL, "has_point", PropertyInfo(Variant::VECTOR2, "point")));
}
Control::Control() {

	data.parent = NULL;

	data.mouse_filter = MOUSE_FILTER_STOP;
	data.pass_on_modal_close_click = true;

	data.SI = NULL;
	data.MI = NULL;
	data.RI = NULL;
	data.modal = false;
	data.theme_owner = NULL;
	data.modal_exclusive = false;
	data.default_cursor = CURSOR_ARROW;
	data.h_size_flags = SIZE_FILL;
	data.v_size_flags = SIZE_FILL;
	data.expand = 1;
	data.pending_min_size_update = false;
	data.rotation = 0;
	data.parent_canvas_item = NULL;
	data.scale = Vector2(1, 1);
	data.drag_owner = 0;
	data.modal_frame = 0;
	data.block_minimum_size_adjust = false;
	data.disable_visibility_clip = false;
	data.h_grow = GROW_DIRECTION_END;
	data.v_grow = GROW_DIRECTION_END;

	data.clip_contents = false;
	for (int i = 0; i < 4; i++) {
		data.anchor[i] = ANCHOR_BEGIN;
		data.margin[i] = 0;
	}
	data.focus_mode = FOCUS_NONE;
	data.modal_prev_focus_owner = 0;
}

Control::~Control() {
}
