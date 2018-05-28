/*************************************************************************/
/*  editor_inspector.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor_inspector.h"
#include "array_property_edit.h"
#include "dictionary_property_edit.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "multi_node_edit.h"
#include "scene/resources/packed_scene.h"

// TODO:
// arrays and dictionary
// replace property editor in sectionedpropertyeditor

Size2 EditorProperty::get_minimum_size() const {

	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (!c->is_visible())
			continue;
		if (c == bottom_editor)
			continue;

		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	if (keying) {
		Ref<Texture> key = get_icon("Key", "EditorIcons");
		ms.width += key->get_width() + get_constant("hseparator", "Tree");
	}

	if (checkable) {
		Ref<Texture> check = get_icon("checked", "CheckBox");
		ms.width += check->get_width() + get_constant("hseparator", "Tree");
	}

	if (bottom_editor != NULL) {
		Ref<Font> font = get_font("font", "Tree");
		ms.height += font->get_height();
		ms.height += get_constant("vseparation", "Tree");
		Size2 bems = bottom_editor->get_combined_minimum_size();
		bems.width += get_constant("item_margin", "Tree");
		ms.height += bems.height;
		ms.width = MAX(ms.width, bems.width);
	}

	return ms;
}

void EditorProperty::_notification(int p_what) {

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		Size2 size = get_size();
		Rect2 rect;
		Rect2 bottom_rect;

		{
			int child_room = size.width / 2;
			Ref<Font> font = get_font("font", "Tree");
			int height = font->get_height();

			//compute room needed
			for (int i = 0; i < get_child_count(); i++) {

				Control *c = Object::cast_to<Control>(get_child(i));
				if (!c)
					continue;
				if (c->is_set_as_toplevel())
					continue;
				if (c == bottom_editor)
					continue;

				Size2 minsize = c->get_combined_minimum_size();
				child_room = MAX(child_room, minsize.width);
				height = MAX(height, minsize.height);
			}

			text_size = MAX(0, size.width - child_room + 4 * EDSCALE);

			rect = Rect2(text_size, 0, size.width - text_size, height);

			if (bottom_editor) {

				int m = get_constant("item_margin", "Tree");
				bottom_rect = Rect2(m, rect.size.height + get_constant("vseparation", "Tree"), size.width - m, bottom_editor->get_combined_minimum_size().height);
			}
		}

		if (keying) {
			Ref<Texture> key;

			if (use_keying_next()) {
				key = get_icon("KeyNext", "EditorIcons");
			} else {
				key = get_icon("Key", "EditorIcons");
			}

			rect.size.x -= key->get_width() + get_constant("hseparator", "Tree");
		}

		//set children
		for (int i = 0; i < get_child_count(); i++) {

			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;
			if (c == bottom_editor)
				continue;

			fit_child_in_rect(c, rect);
		}

		if (bottom_editor) {
			fit_child_in_rect(bottom_editor, bottom_rect);
		}

		update(); //need to redraw text
	}

	if (p_what == NOTIFICATION_DRAW) {
		Ref<Font> font = get_font("font", "Tree");

		Size2 size = get_size();
		if (bottom_editor) {
			size.height = bottom_editor->get_margin(MARGIN_TOP);
		} else if (label_reference) {
			size.height = label_reference->get_size().height;
		}

		if (selected) {
			Ref<StyleBox> sb = get_stylebox("selected", "Tree");
			draw_style_box(sb, Rect2(Vector2(), size));
		}

		Color color;
		if (draw_red) {
			color = get_color("error_color", "Editor");
		} else {
			color = get_color("font_color", "Tree");
		}
		if (label.find(".") != -1) {
			color.a = 0.5; //this should be un-hacked honestly, as it's used for editor overrides
		}

		int ofs = 0;
		if (checkable) {
			Ref<Texture> checkbox;
			if (checked)
				checkbox = get_icon("checked", "CheckBox");
			else
				checkbox = get_icon("unchecked", "CheckBox");

			Color color(1, 1, 1);
			if (check_hover) {
				color.r *= 1.2;
				color.g *= 1.2;
				color.b *= 1.2;
			}
			check_rect = Rect2(ofs, ((size.height - checkbox->get_height()) / 2), checkbox->get_width(), checkbox->get_height());
			draw_texture(checkbox, check_rect.position, color);
			ofs += get_constant("hseparator", "Tree");
			ofs += checkbox->get_width();
		} else {
			check_rect = Rect2();
		}

		int text_limit = text_size;

		if (can_revert) {
			Ref<Texture> reload_icon = get_icon("ReloadSmall", "EditorIcons");
			text_limit -= reload_icon->get_width() + get_constant("hseparator", "Tree") * 2;
			revert_rect = Rect2(text_limit + get_constant("hseparator", "Tree"), (size.height - reload_icon->get_height()) / 2, reload_icon->get_width(), reload_icon->get_height());

			Color color(1, 1, 1);
			if (revert_hover) {
				color.r *= 1.2;
				color.g *= 1.2;
				color.b *= 1.2;
			}

			draw_texture(reload_icon, revert_rect.position, color);
		} else {
			revert_rect = Rect2();
		}

		int v_ofs = (size.height - font->get_height()) / 2;
		draw_string(font, Point2(ofs, v_ofs + font->get_ascent()), label, color, text_limit);

		if (keying) {
			Ref<Texture> key;

			if (use_keying_next()) {
				key = get_icon("KeyNext", "EditorIcons");
			} else {
				key = get_icon("Key", "EditorIcons");
			}

			ofs = size.width - key->get_width() - get_constant("hseparator", "Tree");

			Color color(1, 1, 1);
			if (keying_hover) {
				color.r *= 1.2;
				color.g *= 1.2;
				color.b *= 1.2;
			}
			keying_rect = Rect2(ofs, ((size.height - key->get_height()) / 2), key->get_width(), key->get_height());
			draw_texture(key, keying_rect.position, color);
		} else {
			keying_rect = Rect2();
		}

		//int vs = get_constant("vseparation", "Tree");
		Color guide_color = get_color("guide_color", "Tree");
		int vs_height = get_size().height; // vs / 2;
		draw_line(Point2(0, vs_height), Point2(get_size().width, vs_height), guide_color);
	}
}

void EditorProperty::set_label(const String &p_label) {
	label = p_label;
	update();
}

String EditorProperty::get_label() const {
	return label;
}

Object *EditorProperty::get_edited_object() {
	return object;
}

StringName EditorProperty::get_edited_property() {
	return property;
}

void EditorProperty::update_property() {
	if (get_script_instance())
		get_script_instance()->call("update_property");
}

void EditorProperty::set_read_only(bool p_read_only) {
	read_only = p_read_only;
}

bool EditorProperty::is_read_only() const {
	return read_only;
}

bool EditorProperty::_might_be_in_instance() {

	if (!object)
		return false;

	Node *node = Object::cast_to<Node>(object);

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();

	bool might_be = false;

	while (node) {

		if (node->get_scene_instance_state().is_valid()) {
			might_be = true;
			break;
		}
		if (node == edited_scene) {
			if (node->get_scene_inherited_state().is_valid()) {
				might_be = true;
				break;
			}
			might_be = false;
			break;
		}
		node = node->get_owner();
	}

	return might_be; // or might not be
}

bool EditorProperty::_get_instanced_node_original_property(const StringName &p_prop, Variant &value) {

	Node *node = Object::cast_to<Node>(object);

	if (!node)
		return false;

	Node *orig = node;

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();

	bool found = false;

	while (node) {

		Ref<SceneState> ss;

		if (node == edited_scene) {
			ss = node->get_scene_inherited_state();

		} else {
			ss = node->get_scene_instance_state();
		}

		if (ss.is_valid()) {

			NodePath np = node->get_path_to(orig);
			int node_idx = ss->find_node_by_path(np);
			if (node_idx >= 0) {
				bool lfound = false;
				Variant lvar;
				lvar = ss->get_property_value(node_idx, p_prop, lfound);
				if (lfound) {

					found = true;
					value = lvar;
				}
			}
		}
		if (node == edited_scene) {
			//just in case
			break;
		}
		node = node->get_owner();
	}

	return found;
}

bool EditorProperty::_is_property_different(const Variant &p_current, const Variant &p_orig, int p_usage) {

	// this is a pretty difficult function, because a property may not be saved but may have
	// the flag to not save if one or if zero

	{
		Node *node = Object::cast_to<Node>(object);
		if (!node)
			return false;

		Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
		bool found_state = false;

		while (node) {

			Ref<SceneState> ss;

			if (node == edited_scene) {
				ss = node->get_scene_inherited_state();

			} else {
				ss = node->get_scene_instance_state();
			}

			if (ss.is_valid()) {
				found_state = true;
			}
			if (node == edited_scene) {
				//just in case
				break;
			}
			node = node->get_owner();
		}

		if (!found_state)
			return false; //pointless to check if we are not comparing against anything.
	}

	if (p_orig.get_type() == Variant::NIL) {
		// not found (was not saved)
		// check if it was not saved due to being zero or one
		if (p_current.is_zero() && property_usage & PROPERTY_USAGE_STORE_IF_NONZERO)
			return false;
		if (p_current.is_one() && property_usage & PROPERTY_USAGE_STORE_IF_NONONE)
			return false;
	}

	if (p_current.get_type() == Variant::REAL && p_orig.get_type() == Variant::REAL) {
		float a = p_current;
		float b = p_orig;

		return Math::abs(a - b) > CMP_EPSILON; //this must be done because, as some scenes save as text, there might be a tiny difference in floats due to numerical error
	}

	return bool(Variant::evaluate(Variant::OP_NOT_EQUAL, p_current, p_orig));
}

bool EditorProperty::_is_instanced_node_with_original_property_different() {

	bool mbi = _might_be_in_instance();
	if (mbi) {
		Variant vorig;
		int usage = property_usage & (PROPERTY_USAGE_STORE_IF_NONONE | PROPERTY_USAGE_STORE_IF_NONZERO);
		if (_get_instanced_node_original_property(property, vorig) || usage) {
			Variant v = object->get(property);

			if (_is_property_different(v, vorig, usage)) {
				return true;
			}
		}
	}
	return false;
}

void EditorProperty::update_reload_status() {

	if (property == StringName())
		return; //no property, so nothing to do

	bool has_reload = false;

	if (_is_instanced_node_with_original_property_different()) {
		has_reload = true;
	}

	if (object->call("property_can_revert", property).operator bool()) {

		has_reload = true;
	}

	if (!has_reload && !object->get_script().is_null()) {
		Ref<Script> scr = object->get_script();
		Variant orig_value;
		if (scr->get_property_default_value(property, orig_value)) {
			if (orig_value != object->get(property)) {
				has_reload = true;
			}
		}
	}

	if (has_reload != can_revert) {
		can_revert = has_reload;
		update();
	}
}

bool EditorProperty::use_keying_next() const {
	return false;
}
void EditorProperty::set_checkable(bool p_checkable) {

	checkable = p_checkable;
	update();
	queue_sort();
}

bool EditorProperty::is_checkable() const {

	return checkable;
}

void EditorProperty::set_checked(bool p_checked) {

	checked = p_checked;
	update();
}

bool EditorProperty::is_checked() const {

	return checked;
}

void EditorProperty::set_draw_red(bool p_draw_red) {

	draw_red = p_draw_red;
	update();
}

void EditorProperty::set_keying(bool p_keying) {
	keying = p_keying;
	update();
	queue_sort();
}

bool EditorProperty::is_keying() const {
	return keying;
}

bool EditorProperty::is_draw_red() const {

	return draw_red;
}

void EditorProperty::_focusable_focused(int p_index) {

	if (!selectable)
		return;
	bool already_selected = selected;
	selected = true;
	selected_focusable = p_index;
	update();
	if (!already_selected && selected) {
		emit_signal("selected", property, selected_focusable);
	}
}

void EditorProperty::add_focusable(Control *p_control) {

	p_control->connect("focus_entered", this, "_focusable_focused", varray(focusables.size()));
	focusables.push_back(p_control);
}

void EditorProperty::select(int p_focusable) {

	bool already_selected = selected;

	if (p_focusable >= 0) {
		ERR_FAIL_INDEX(p_focusable, focusables.size());
		focusables[p_focusable]->grab_focus();
	} else {
		selected = true;
		update();
	}

	if (!already_selected && selected) {
		emit_signal("selected", property, selected_focusable);
	}
}

void EditorProperty::deselect() {
	selected = false;
	selected_focusable = -1;
	update();
}

bool EditorProperty::is_selected() const {
	return selected;
}

void EditorProperty::_gui_input(const Ref<InputEvent> &p_event) {

	if (property == StringName())
		return;

	Ref<InputEventMouse> me = p_event;

	if (me.is_valid()) {

		bool button_left = me->get_button_mask() & BUTTON_MASK_LEFT;

		bool new_keying_hover = keying_rect.has_point(me->get_position()) && !button_left;
		if (new_keying_hover != keying_hover) {
			keying_hover = new_keying_hover;
			update();
		}

		bool new_revert_hover = revert_rect.has_point(me->get_position()) && !button_left;
		if (new_revert_hover != revert_hover) {
			revert_hover = new_revert_hover;
			update();
		}

		bool new_check_hover = check_rect.has_point(me->get_position()) && !button_left;
		if (new_check_hover != check_hover) {
			check_hover = new_check_hover;
			update();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

		if (!selected && selectable) {
			selected = true;
			emit_signal("selected", property, -1);
			update();
		}

		if (keying_rect.has_point(mb->get_position())) {
			emit_signal("property_keyed", property);
		}

		if (revert_rect.has_point(mb->get_position())) {

			Variant vorig;

			if (_might_be_in_instance() && _get_instanced_node_original_property(property, vorig)) {

				emit_signal("property_changed", property, vorig.duplicate(true));
				update_property();
				return;
			}

			if (object->call("property_can_revert", property).operator bool()) {
				Variant rev = object->call("property_get_revert", property);
				emit_signal("property_changed", property, rev);
				update_property();
			}

			if (!object->get_script().is_null()) {
				Ref<Script> scr = object->get_script();
				Variant orig_value;
				if (scr->get_property_default_value(property, orig_value)) {
					emit_signal("property_changed", property, orig_value);
					update_property();
				}
			}
		}
		if (check_rect.has_point(mb->get_position())) {
			checked = !checked;
			update();
			emit_signal("property_checked", property, checked);
		}
	}
}

void EditorProperty::set_label_reference(Control *p_control) {

	label_reference = p_control;
}
void EditorProperty::set_bottom_editor(Control *p_control) {

	bottom_editor = p_control;
}
Variant EditorProperty::get_drag_data(const Point2 &p_point) {

	if (property == StringName())
		return Variant();

	Dictionary dp;
	dp["type"] = "obj_property";
	dp["object"] = object;
	dp["property"] = property;
	dp["value"] = object->get(property);

	Label *label = memnew(Label);
	label->set_text(property);
	set_drag_preview(label);
	return dp;
}

void EditorProperty::set_use_folding(bool p_use_folding) {

	use_folding = p_use_folding;
}

bool EditorProperty::is_using_folding() const {

	return use_folding;
}

void EditorProperty::expand_all_folding() {
}

void EditorProperty::collapse_all_folding() {
}

void EditorProperty::set_selectable(bool p_selectable) {
	selectable = p_selectable;
}

bool EditorProperty::is_selectable() const {
	return selectable;
}

void EditorProperty::set_object_and_property(Object *p_object, const StringName &p_property) {
	object = p_object;
	property = p_property;
}

void EditorProperty::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_label", "text"), &EditorProperty::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &EditorProperty::get_label);

	ClassDB::bind_method(D_METHOD("set_read_only", "read_only"), &EditorProperty::set_read_only);
	ClassDB::bind_method(D_METHOD("is_read_only"), &EditorProperty::is_read_only);

	ClassDB::bind_method(D_METHOD("set_checkable", "checkable"), &EditorProperty::set_checkable);
	ClassDB::bind_method(D_METHOD("is_checkable"), &EditorProperty::is_checkable);

	ClassDB::bind_method(D_METHOD("set_checked", "checked"), &EditorProperty::set_checked);
	ClassDB::bind_method(D_METHOD("is_checked"), &EditorProperty::is_checked);

	ClassDB::bind_method(D_METHOD("set_draw_red", "draw_red"), &EditorProperty::set_draw_red);
	ClassDB::bind_method(D_METHOD("is_draw_red"), &EditorProperty::is_draw_red);

	ClassDB::bind_method(D_METHOD("set_keying", "keying"), &EditorProperty::set_keying);
	ClassDB::bind_method(D_METHOD("is_keying"), &EditorProperty::is_keying);

	ClassDB::bind_method(D_METHOD("get_edited_property"), &EditorProperty::get_edited_property);
	ClassDB::bind_method(D_METHOD("get_edited_object"), &EditorProperty::get_edited_object);

	ClassDB::bind_method(D_METHOD("_gui_input"), &EditorProperty::_gui_input);
	ClassDB::bind_method(D_METHOD("_focusable_focused"), &EditorProperty::_focusable_focused);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "is_read_only");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checkable"), "set_checkable", "is_checkable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checked"), "set_checked", "is_checked");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_red"), "set_draw_red", "is_draw_red");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "keying"), "set_keying", "is_keying");
	ADD_SIGNAL(MethodInfo("property_changed", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("multiple_properties_changed", PropertyInfo(Variant::POOL_STRING_ARRAY, "properties"), PropertyInfo(Variant::ARRAY, "value")));
	ADD_SIGNAL(MethodInfo("property_keyed", PropertyInfo(Variant::STRING, "property")));
	ADD_SIGNAL(MethodInfo("property_keyed_with_value", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("property_checked", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::STRING, "bool")));
	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
	ADD_SIGNAL(MethodInfo("object_id_selected", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "focusable_idx")));

	MethodInfo vm;
	vm.name = "update_property";
	BIND_VMETHOD(vm);
}

EditorProperty::EditorProperty() {

	selectable = true;
	text_size = 0;
	read_only = false;
	checkable = false;
	checked = false;
	draw_red = false;
	keying = false;
	keying_hover = false;
	revert_hover = false;
	check_hover = false;
	can_revert = false;
	use_folding = false;
	property_usage = 0;
	selected = false;
	selected_focusable = -1;
	label_reference = NULL;
	bottom_editor = NULL;
}
////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorPlugin::add_custom_control(Control *control) {

	AddedEditor ae;
	ae.property_editor = control;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor(const String &p_for_property, Control *p_prop) {

	ERR_FAIL_COND(Object::cast_to<EditorProperty>(p_prop) == NULL);

	AddedEditor ae;
	ae.properties.push_back(p_for_property);
	ae.property_editor = p_prop;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop) {

	AddedEditor ae;
	ae.properties = p_properties;
	ae.property_editor = p_prop;
	ae.label = p_label;
	added_editors.push_back(ae);
}

bool EditorInspectorPlugin::can_handle(Object *p_object) {

	if (get_script_instance()) {
		return get_script_instance()->call("can_handle", p_object);
	}
	return false;
}
void EditorInspectorPlugin::parse_begin(Object *p_object) {

	if (get_script_instance()) {
		get_script_instance()->call("parse_begin", p_object);
	}
}

void EditorInspectorPlugin::parse_category(Object *p_object, const String &p_parse_category) {

	if (get_script_instance()) {
		get_script_instance()->call("parse_category", p_object, p_parse_category);
	}
}

bool EditorInspectorPlugin::parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage) {

	if (get_script_instance()) {
		Variant arg[6] = {
			p_object, p_type, p_path, p_hint, p_hint_text, p_usage
		};
		const Variant *argptr[6] = {
			&arg[0], &arg[1], &arg[2], &arg[3], &arg[4], &arg[5]
		};

		Variant::CallError err;
		return get_script_instance()->call("parse_property", (const Variant **)&argptr, 6, err);
	}
	return false;
}
void EditorInspectorPlugin::parse_end() {

	if (get_script_instance()) {
		get_script_instance()->call("parse_end");
	}
}

void EditorInspectorPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_custom_control", "control"), &EditorInspectorPlugin::add_custom_control);
	ClassDB::bind_method(D_METHOD("add_property_editor", "property", "editor"), &EditorInspectorPlugin::add_property_editor);
	ClassDB::bind_method(D_METHOD("add_property_editor_for_multiple_properties", "label", "properties", "editor"), &EditorInspectorPlugin::add_property_editor_for_multiple_properties);

	MethodInfo vm;
	vm.name = "can_handle";
	vm.return_val.type = Variant::BOOL;
	vm.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
	BIND_VMETHOD(vm);
	vm.name = "parse_begin";
	vm.return_val.type = Variant::NIL;
	BIND_VMETHOD(vm);
	vm.name = "parse_category";
	vm.arguments.push_back(PropertyInfo(Variant::STRING, "category"));
	BIND_VMETHOD(vm);
	vm.arguments.pop_back();
	vm.name = "parse_property";
	vm.return_val.type = Variant::BOOL;
	vm.arguments.push_back(PropertyInfo(Variant::INT, "type"));
	vm.arguments.push_back(PropertyInfo(Variant::STRING, "path"));
	vm.arguments.push_back(PropertyInfo(Variant::INT, "hint"));
	vm.arguments.push_back(PropertyInfo(Variant::STRING, "hint_text"));
	vm.arguments.push_back(PropertyInfo(Variant::INT, "usage"));
	BIND_VMETHOD(vm);
	vm.arguments.clear();
	vm.name = "parse_end";
	vm.return_val.type = Variant::NIL;
	BIND_VMETHOD(vm);
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorCategory::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		draw_rect(Rect2(Vector2(), get_size()), bg_color);
		Ref<Font> font = get_font("font", "Tree");

		int hs = get_constant("hseparation", "Tree");

		int w = font->get_string_size(label).width;
		if (icon.is_valid()) {
			w += hs + icon->get_width();
		}

		int ofs = (get_size().width - w) / 2;

		if (icon.is_valid()) {
			draw_texture(icon, Point2(ofs, (get_size().height - icon->get_height()) / 2).floor());
			ofs += hs + icon->get_width();
		}

		Color color = get_color("font_color", "Tree");
		draw_string(font, Point2(ofs, font->get_ascent() + (get_size().height - font->get_height()) / 2).floor(), label, color, get_size().width);
	}
}

Size2 EditorInspectorCategory::get_minimum_size() const {

	Ref<Font> font = get_font("font", "Tree");

	Size2 ms;
	ms.width = 1;
	ms.height = font->get_height();
	if (icon.is_valid()) {
		ms.height = MAX(icon->get_height(), ms.height);
	}
	ms.height += get_constant("vseparation", "Tree");

	return ms;
}

EditorInspectorCategory::EditorInspectorCategory() {
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorSection::_notification(int p_what) {

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		Ref<Font> font = get_font("font", "Tree");
		Ref<Texture> arrow;

#ifdef TOOLS_ENABLED
		if (foldable) {
			if (object->editor_is_section_unfolded(section)) {
				arrow = get_icon("arrow", "Tree");
			} else {
				arrow = get_icon("arrow_collapsed", "Tree");
			}
		}
#endif

		Size2 size = get_size();
		Point2 offset;
		offset.y = font->get_height();
		if (arrow.is_valid()) {
			offset.y = MAX(offset.y, arrow->get_height());
		}

		offset.y += get_constant("vseparation", "Tree");
		offset.x += get_constant("item_margin", "Tree");

		Rect2 rect(offset, size - offset);

		//set children
		for (int i = 0; i < get_child_count(); i++) {

			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;
			if (!c->is_visible_in_tree())
				continue;

			fit_child_in_rect(c, rect);
		}

		update(); //need to redraw text
	}

	if (p_what == NOTIFICATION_DRAW) {

		Ref<Texture> arrow;

#ifdef TOOLS_ENABLED
		if (foldable) {
			if (object->editor_is_section_unfolded(section)) {
				arrow = get_icon("arrow", "Tree");
			} else {
				arrow = get_icon("arrow_collapsed", "Tree");
			}
		}
#endif

		Ref<Font> font = get_font("font", "Tree");

		int h = font->get_height();
		if (arrow.is_valid()) {
			h = MAX(h, arrow->get_height());
		}
		h += get_constant("vseparation", "Tree");

		draw_rect(Rect2(Vector2(), Vector2(get_size().width, h)), bg_color);

		int hs = get_constant("hseparation", "Tree");

		int ofs = 0;
		if (arrow.is_valid()) {
			draw_texture(arrow, Point2(ofs, (h - arrow->get_height()) / 2).floor());
			ofs += hs + arrow->get_width();
		}

		Color color = get_color("font_color", "Tree");
		draw_string(font, Point2(ofs, font->get_ascent() + (h - font->get_height()) / 2).floor(), label, color, get_size().width);
	}
}

Size2 EditorInspectorSection::get_minimum_size() const {

	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (!c->is_visible())
			continue;
		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	Ref<Font> font = get_font("font", "Tree");
	ms.height += font->get_ascent() + get_constant("vseparation", "Tree");
	ms.width += get_constant("item_margin", "Tree");

	return ms;
}

void EditorInspectorSection::setup(const String &p_section, const String &p_label, Object *p_object, const Color &p_bg_color, bool p_foldable) {

	section = p_section;
	label = p_label;
	object = p_object;
	bg_color = p_bg_color;
	foldable = p_foldable;

#ifdef TOOLS_ENABLED
	if (foldable) {
		if (object->editor_is_section_unfolded(section)) {
			vbox->show();
		} else {
			vbox->hide();
		}
	}
		//	void editor_set_section_unfold(const String &p_section, bool p_unfolded);

#endif
}

void EditorInspectorSection::_gui_input(const Ref<InputEvent> &p_event) {

	if (!foldable)
		return;

#ifdef TOOLS_ENABLED

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		bool unfold = !object->editor_is_section_unfolded(section);
		object->editor_set_section_unfold(section, unfold);
		if (unfold) {
			vbox->show();
		} else {
			vbox->hide();
		}
	}
#endif
}

VBoxContainer *EditorInspectorSection::get_vbox() {
	return vbox;
}

void EditorInspectorSection::unfold() {

	if (!foldable)
		return;
#ifdef TOOLS_ENABLED

	object->editor_set_section_unfold(section, true);
	vbox->show();
	update();
#endif
}

void EditorInspectorSection::fold() {
	if (!foldable)
		return;

#ifdef TOOLS_ENABLED

	object->editor_set_section_unfold(section, false);
	vbox->hide();
	update();
#endif
}

void EditorInspectorSection::_bind_methods() {

	ClassDB::bind_method(D_METHOD("setup", "section", "label", "object", "bg_color", "foldable"), &EditorInspectorSection::setup);
	ClassDB::bind_method(D_METHOD("get_vbox"), &EditorInspectorSection::get_vbox);
	ClassDB::bind_method(D_METHOD("unfold"), &EditorInspectorSection::unfold);
	ClassDB::bind_method(D_METHOD("fold"), &EditorInspectorSection::fold);
	ClassDB::bind_method(D_METHOD("_gui_input"), &EditorInspectorSection::_gui_input);
}

EditorInspectorSection::EditorInspectorSection() {
	object = NULL;
	foldable = false;
	vbox = memnew(VBoxContainer);
	add_child(vbox);
}

////////////////////////////////////////////////
////////////////////////////////////////////////

Ref<EditorInspectorPlugin> EditorInspector::inspector_plugins[MAX_PLUGINS];
int EditorInspector::inspector_plugin_count = 0;

void EditorInspector::add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {

	ERR_FAIL_COND(inspector_plugin_count == MAX_PLUGINS);

	for (int i = 0; i < inspector_plugin_count; i++) {
		if (inspector_plugins[i] == p_plugin)
			return; //already exists
	}
	inspector_plugins[inspector_plugin_count++] = p_plugin;
}

void EditorInspector::remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {

	ERR_FAIL_COND(inspector_plugin_count == MAX_PLUGINS);

	int idx = -1;
	for (int i = 0; i < inspector_plugin_count; i++) {
		if (inspector_plugins[i] == p_plugin) {
			idx = i;
			break;
		}
	}

	for (int i = idx; i < inspector_plugin_count - 1; i++) {
		inspector_plugins[i] = inspector_plugins[i + 1];
	}
	inspector_plugin_count--;
}

void EditorInspector::cleanup_plugins() {
	for (int i = 0; i < inspector_plugin_count; i++) {
		inspector_plugins[i].unref();
	}
	inspector_plugin_count = 0;
}

void EditorInspector::set_undo_redo(UndoRedo *p_undo_redo) {
	undo_redo = p_undo_redo;
}

String EditorInspector::get_selected_path() const {

	return property_selected;
}

void EditorInspector::_parse_added_editors(VBoxContainer *current_vbox, Ref<EditorInspectorPlugin> ped) {

	for (List<EditorInspectorPlugin::AddedEditor>::Element *F = ped->added_editors.front(); F; F = F->next()) {

		EditorProperty *ep = Object::cast_to<EditorProperty>(F->get().property_editor);
		current_vbox->add_child(F->get().property_editor);

		if (ep) {

			ep->object = object;
			ep->connect("property_changed", this, "_property_changed");
			ep->connect("property_keyed", this, "_property_keyed");
			ep->connect("property_keyed_with_value", this, "_property_keyed_with_value");
			ep->connect("property_checked", this, "_property_checked");
			ep->connect("selected", this, "_property_selected");
			ep->connect("multiple_properties_changed", this, "_multiple_properties_changed");
			ep->connect("resource_selected", this, "_resource_selected", varray(), CONNECT_DEFERRED);
			ep->connect("object_id_selected", this, "_object_id_selected", varray(), CONNECT_DEFERRED);

			if (F->get().properties.size()) {

				if (F->get().properties.size() == 1) {
					//since it's one, associate:
					ep->property = F->get().properties[0];
					ep->property_usage = 0;
				}

				if (F->get().label != String()) {
					ep->set_label(F->get().label);
				}

				for (int i = 0; i < F->get().properties.size(); i++) {
					String prop = F->get().properties[i];

					if (!editor_property_map.has(prop)) {
						editor_property_map[prop] = List<EditorProperty *>();
					}
					editor_property_map[prop].push_back(ep);
				}
			}

			ep->set_read_only(read_only);
			ep->update_property();
			ep->update_reload_status();
		}
	}
	ped->added_editors.clear();
}

void EditorInspector::update_tree() {

	//to update properly if all is refreshed
	StringName current_selected = property_selected;
	int current_focusable = property_focusable;

	_clear();

	if (!object)
		return;

	List<Ref<EditorInspectorPlugin> > valid_plugins;

	for (int i = inspector_plugin_count - 1; i >= 0; i--) { //start by last, so lastly added can override newly added
		if (!inspector_plugins[i]->can_handle(object))
			continue;
		valid_plugins.push_back(inspector_plugins[i]);
	}

	bool draw_red = false;

	{
		Node *nod = Object::cast_to<Node>(object);
		Node *es = EditorNode::get_singleton()->get_edited_scene();
		if (nod && es != nod && nod->get_owner() != es) {
			draw_red = true;
		}
	}

	//	TreeItem *current_category = NULL;

	String filter = search_box ? search_box->get_text() : "";
	String group;
	String group_base;

	List<PropertyInfo> plist;
	object->get_property_list(&plist, true);

	HashMap<String, VBoxContainer *> item_path;
	item_path[""] = main_vbox;

	Color sscolor = get_color("prop_subsection", "Editor");

	for (List<Ref<EditorInspectorPlugin> >::Element *E = valid_plugins.front(); E; E = E->next()) {
		Ref<EditorInspectorPlugin> ped = E->get();
		ped->parse_begin(object);
		_parse_added_editors(main_vbox, ped);
	}

	for (List<PropertyInfo>::Element *I = plist.front(); I; I = I->next()) {

		PropertyInfo &p = I->get();

		//make sure the property can be edited

		if (p.usage & PROPERTY_USAGE_GROUP) {

			group = p.name;
			group_base = p.hint_string;

			continue;

		} else if (p.usage & PROPERTY_USAGE_CATEGORY) {

			group = "";
			group_base = "";

			if (!show_categories)
				continue;

			List<PropertyInfo>::Element *N = I->next();
			bool valid = true;
			//if no properties in category, skip
			while (N) {
				if (N->get().usage & PROPERTY_USAGE_EDITOR)
					break;
				if (N->get().usage & PROPERTY_USAGE_CATEGORY) {
					valid = false;
					break;
				}
				N = N->next();
			}
			if (!valid)
				continue; //empty, ignore

			EditorInspectorCategory *category = memnew(EditorInspectorCategory);
			main_vbox->add_child(category);

			String type = p.name;
			if (has_icon(type, "EditorIcons"))
				category->icon = get_icon(type, "EditorIcons");
			else
				category->icon = get_icon("Object", "EditorIcons");
			category->label = type;

			category->bg_color = get_color("prop_category", "Editor");
			if (use_doc_hints) {
				StringName type = p.name;
				if (!class_descr_cache.has(type)) {

					String descr;
					DocData *dd = EditorHelp::get_doc_data();
					Map<String, DocData::ClassDoc>::Element *E = dd->class_list.find(type);
					if (E) {
						descr = E->get().brief_description;
					}
					class_descr_cache[type] = descr.word_wrap(80);
				}

				category->set_tooltip(TTR("Class:") + " " + p.name + (class_descr_cache[type] == "" ? "" : "\n\n" + class_descr_cache[type]));
			}

			for (List<Ref<EditorInspectorPlugin> >::Element *E = valid_plugins.front(); E; E = E->next()) {
				Ref<EditorInspectorPlugin> ped = E->get();
				ped->parse_category(object, p.name);
				_parse_added_editors(main_vbox, ped);
			}

			continue;

		} else if (!(p.usage & PROPERTY_USAGE_EDITOR))
			continue;

		if (hide_script && p.name == "script")
			continue;

		String basename = p.name;
		if (group != "") {
			if (group_base != "") {
				if (basename.begins_with(group_base)) {
					basename = basename.replace_first(group_base, "");
				} else if (group_base.begins_with(basename)) {
					//keep it, this is used pretty often
				} else {
					group = ""; //no longer using group base, clear
				}
			}
		}

		if (group != "") {
			basename = group + "/" + basename;
		}

		String name = (basename.find("/") != -1) ? basename.right(basename.find_last("/") + 1) : basename;

		if (capitalize_paths) {
			int dot = name.find(".");
			if (dot != -1) {
				String ov = name.right(dot);
				name = name.substr(0, dot);
				name = name.camelcase_to_underscore().capitalize();
				name += ov;

			} else {
				name = name.camelcase_to_underscore().capitalize();
			}
		}

		String path = basename.left(basename.find_last("/"));

		if (use_filter && filter != "") {

			String cat = path;

			if (capitalize_paths)
				cat = cat.capitalize();

			if (!filter.is_subsequence_ofi(cat) && !filter.is_subsequence_ofi(name))
				continue;
		}

		VBoxContainer *current_vbox = main_vbox;

		{

			String acc_path = "";
			int level = 1;
			for (int i = 0; i < path.get_slice_count("/"); i++) {
				String path_name = path.get_slice("/", i);
				if (i > 0)
					acc_path += "/";
				acc_path += path_name;
				if (!item_path.has(acc_path)) {
					EditorInspectorSection *section = memnew(EditorInspectorSection);
					current_vbox->add_child(section);
					sections.push_back(section);

					if (capitalize_paths)
						path_name = path_name.capitalize();
					Color c = sscolor;
					c.a /= level;
					section->setup(path_name, acc_path, object, c, use_folding);

					item_path[acc_path] = section->get_vbox();
				}
				current_vbox = item_path[acc_path];
				level = (MIN(level + 1, 4));
			}
		}

		bool checkable = false;
		bool checked = false;
		if (p.usage & PROPERTY_USAGE_CHECKABLE) {
			checkable = true;
			checked = p.usage & PROPERTY_USAGE_CHECKED;
		}

		String doc_hint;

		if (use_doc_hints) {

			StringName classname = object->get_class_name();
			StringName propname = p.name;
			String descr;
			bool found = false;

			Map<StringName, Map<StringName, String> >::Element *E = descr_cache.find(classname);
			if (E) {
				Map<StringName, String>::Element *F = E->get().find(propname);
				if (F) {
					found = true;
					descr = F->get();
				}
			}

			if (!found) {
				DocData *dd = EditorHelp::get_doc_data();
				Map<String, DocData::ClassDoc>::Element *E = dd->class_list.find(classname);
				while (E && descr == String()) {
					for (int i = 0; i < E->get().properties.size(); i++) {
						if (E->get().properties[i].name == propname.operator String()) {
							descr = E->get().properties[i].description.strip_edges().word_wrap(80);
							break;
						}
					}
					if (!E->get().inherits.empty()) {
						E = dd->class_list.find(E->get().inherits);
					} else {
						break;
					}
				}
				descr_cache[classname][propname] = descr;
			}

			doc_hint = descr;
		}

#if 0
		if (p.name == selected_property) {

			item->select(1);
		}
#endif
		for (List<Ref<EditorInspectorPlugin> >::Element *E = valid_plugins.front(); E; E = E->next()) {
			Ref<EditorInspectorPlugin> ped = E->get();
			ped->parse_property(object, p.type, p.name, p.hint, p.hint_string, p.usage);
			List<EditorInspectorPlugin::AddedEditor> editors = ped->added_editors; //make a copy, since plugins may be used again in a sub-inspector
			ped->added_editors.clear();

			for (List<EditorInspectorPlugin::AddedEditor>::Element *F = editors.front(); F; F = F->next()) {

				EditorProperty *ep = Object::cast_to<EditorProperty>(F->get().property_editor);
				current_vbox->add_child(F->get().property_editor);

				if (ep) {

					ep->object = object;
					ep->connect("property_changed", this, "_property_changed");
					ep->connect("property_keyed", this, "_property_keyed");
					ep->connect("property_keyed_with_value", this, "_property_keyed_with_value");
					ep->connect("property_checked", this, "_property_checked");
					ep->connect("selected", this, "_property_selected");
					ep->connect("multiple_properties_changed", this, "_multiple_properties_changed");
					ep->connect("resource_selected", this, "_resource_selected", varray(), CONNECT_DEFERRED);
					ep->connect("object_id_selected", this, "_object_id_selected", varray(), CONNECT_DEFERRED);
					if (doc_hint != String()) {
						ep->set_tooltip(TTR("Property: ") + p.name + "\n\n" + doc_hint);
					} else {
						ep->set_tooltip(TTR("Property: ") + p.name);
					}
					ep->set_draw_red(draw_red);
					ep->set_use_folding(use_folding);
					ep->set_checkable(checkable);
					ep->set_checked(checked);
					ep->set_keying(keying);

					if (F->get().properties.size()) {

						if (F->get().properties.size() == 1) {
							//since it's one, associate:
							ep->property = F->get().properties[0];
							ep->property_usage = p.usage;
							//and set label?
						}

						if (F->get().label != String()) {
							ep->set_label(F->get().label);
						} else {
							//use existin one
							ep->set_label(name);
						}
						for (int i = 0; i < F->get().properties.size(); i++) {
							String prop = F->get().properties[i];

							if (!editor_property_map.has(prop)) {
								editor_property_map[prop] = List<EditorProperty *>();
							}
							editor_property_map[prop].push_back(ep);
						}
					}

					ep->set_read_only(read_only);
					ep->update_property();
					ep->update_reload_status();

					if (current_selected && ep->property == current_selected) {
						ep->select(current_focusable);
					}
				}
			}
		}
	}

	for (List<Ref<EditorInspectorPlugin> >::Element *E = valid_plugins.front(); E; E = E->next()) {
		Ref<EditorInspectorPlugin> ped = E->get();
		ped->parse_end();
		_parse_added_editors(main_vbox, ped);
	}

	//see if this property exists and should be kept
}
void EditorInspector::update_property(const String &p_prop) {
	if (!editor_property_map.has(p_prop))
		return;

	for (List<EditorProperty *>::Element *E = editor_property_map[p_prop].front(); E; E = E->next()) {
		E->get()->update_property();
		E->get()->update_reload_status();
	}
}

void EditorInspector::_clear() {

	while (main_vbox->get_child_count()) {
		memdelete(main_vbox->get_child(0));
	}
	property_selected = StringName();
	property_focusable = -1;
	editor_property_map.clear();
	sections.clear();
	pending.clear();
}

void EditorInspector::refresh() {

	if (refresh_countdown > 0)
		return;
	refresh_countdown = EditorSettings::get_singleton()->get("docks/property_editor/auto_refresh_interval");
}

Object *EditorInspector::get_edited_object() {
	return object;
}

void EditorInspector::edit(Object *p_object) {
	if (object == p_object)
		return;
	if (object) {

		_clear();
		object->remove_change_receptor(this);
	}

	object = p_object;

	if (object) {
		object->add_change_receptor(this);
		update_tree();
	}
}

void EditorInspector::set_keying(bool p_active) {
	if (keying == p_active)
		return;
	keying = p_active;
	update_tree();
}
void EditorInspector::set_read_only(bool p_read_only) {
	read_only = p_read_only;
	update_tree();
}

bool EditorInspector::is_capitalize_paths_enabled() const {

	return capitalize_paths;
}
void EditorInspector::set_enable_capitalize_paths(bool p_capitalize) {
	capitalize_paths = p_capitalize;
	update_tree();
}

void EditorInspector::set_autoclear(bool p_enable) {
	autoclear = p_enable;
}

void EditorInspector::set_show_categories(bool p_show) {
	show_categories = p_show;
	update_tree();
}

void EditorInspector::set_use_doc_hints(bool p_enable) {
	use_doc_hints = p_enable;
	update_tree();
}
void EditorInspector::set_hide_script(bool p_hide) {
	hide_script = p_hide;
	update_tree();
}
void EditorInspector::set_use_filter(bool p_use) {
	use_filter = p_use;
	update_tree();
}
void EditorInspector::register_text_enter(Node *p_line_edit) {
	search_box = Object::cast_to<LineEdit>(p_line_edit);
	if (search_box)
		search_box->connect("text_changed", this, "_filter_changed");
}

void EditorInspector::_filter_changed(const String &p_text) {

	update_tree();
}

void EditorInspector::set_subsection_selectable(bool p_selectable) {
}

void EditorInspector::set_property_selectable(bool p_selectable) {
}

void EditorInspector::set_use_folding(bool p_enable) {
	use_folding = p_enable;
	update_tree();
}

void EditorInspector::collapse_all_folding() {

	for (List<EditorInspectorSection *>::Element *E = sections.front(); E; E = E->next()) {
		E->get()->fold();
	}

	for (Map<StringName, List<EditorProperty *> >::Element *F = editor_property_map.front(); F; F = F->next()) {
		for (List<EditorProperty *>::Element *E = F->get().front(); E; E = E->next()) {
			E->get()->collapse_all_folding();
		}
	}
}

void EditorInspector::expand_all_folding() {
	for (List<EditorInspectorSection *>::Element *E = sections.front(); E; E = E->next()) {
		E->get()->unfold();
	}
	for (Map<StringName, List<EditorProperty *> >::Element *F = editor_property_map.front(); F; F = F->next()) {
		for (List<EditorProperty *>::Element *E = F->get().front(); E; E = E->next()) {
			E->get()->expand_all_folding();
		}
	}
}

void EditorInspector::set_scroll_offset(int p_offset) {
	set_v_scroll(p_offset);
}

int EditorInspector::get_scroll_offset() const {
	return get_v_scroll();
}

void EditorInspector::_edit_request_change(Object *p_object, const String &p_property) {

	if (object != p_object) //may be undoing/redoing for a non edited object, so ignore
		return;

	if (changing)
		return;

	if (p_property == String())
		update_tree_pending = true;
	else {
		pending.insert(p_property);
	}
}

void EditorInspector::_edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all, const String &p_changed_field) {

	if (autoclear && editor_property_map.has(p_name)) {
		for (List<EditorProperty *>::Element *E = editor_property_map[p_name].front(); E; E = E->next()) {
			if (E->get()->is_checkable()) {
				E->get()->set_checked(true);
			}
		}
	}

	if (!undo_redo || Object::cast_to<ArrayPropertyEdit>(object) || Object::cast_to<DictionaryPropertyEdit>(object)) { //kind of hacky

		object->set(p_name, p_value);
		if (p_refresh_all)
			_edit_request_change(object, "");
		else
			_edit_request_change(object, p_name);

		emit_signal(_prop_edited, p_name);

	} else if (Object::cast_to<MultiNodeEdit>(object)) {

		Object::cast_to<MultiNodeEdit>(object)->set_property_field(p_name, p_value, p_changed_field);
		_edit_request_change(object, p_name);
		emit_signal(_prop_edited, p_name);
	} else {

		undo_redo->create_action(TTR("Set") + " " + p_name, UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(object, p_name, p_value);
		undo_redo->add_undo_property(object, p_name, object->get(p_name));

		if (p_refresh_all) {
			undo_redo->add_do_method(this, "_edit_request_change", object, "");
			undo_redo->add_undo_method(this, "_edit_request_change", object, "");
		} else {

			undo_redo->add_do_method(this, "_edit_request_change", object, p_name);
			undo_redo->add_undo_method(this, "_edit_request_change", object, p_name);
		}

		Resource *r = Object::cast_to<Resource>(object);
		if (r) {
			if (!r->is_edited() && String(p_name) != "resource/edited") {
				undo_redo->add_do_method(r, "set_edited", true);
				undo_redo->add_undo_method(r, "set_edited", false);
			}

			if (String(p_name) == "resource_local_to_scene") {
				bool prev = object->get(p_name);
				bool next = p_value;
				if (next) {
					undo_redo->add_do_method(r, "setup_local_to_scene");
				}
				if (prev) {
					undo_redo->add_undo_method(r, "setup_local_to_scene");
				}
			}
		}
		undo_redo->add_do_method(this, "emit_signal", _prop_edited, p_name);
		undo_redo->add_undo_method(this, "emit_signal", _prop_edited, p_name);
		changing++;
		undo_redo->commit_action();
		changing--;
	}

	if (editor_property_map.has(p_name)) {
		for (List<EditorProperty *>::Element *E = editor_property_map[p_name].front(); E; E = E->next()) {
			E->get()->update_reload_status();
		}
	}
}

void EditorInspector::_property_changed(const String &p_path, const Variant &p_value) {

	_edit_set(p_path, p_value, false, "");
}

void EditorInspector::_multiple_properties_changed(Vector<String> p_paths, Array p_values) {

	ERR_FAIL_COND(p_paths.size() == 0 || p_values.size() == 0);
	ERR_FAIL_COND(p_paths.size() != p_values.size());
	String names;
	for (int i = 0; i < p_paths.size(); i++) {
		if (i > 0)
			names += ",";
		names += p_paths[i];
	}
	undo_redo->create_action(TTR("Set Multiple:") + " " + names, UndoRedo::MERGE_ENDS);
	for (int i = 0; i < p_paths.size(); i++) {
		_edit_set(p_paths[i], p_values[i], false, "");
	}
	changing++;
	undo_redo->commit_action();
	changing--;
}

void EditorInspector::_property_keyed(const String &p_path) {

	if (!object)
		return;

	emit_signal("property_keyed", p_path, object->get(p_path), false); //second param is deprecated
}

void EditorInspector::_property_keyed_with_value(const String &p_path, const Variant &p_value) {

	if (!object)
		return;

	emit_signal("property_keyed", p_path, p_value, false); //second param is deprecated
}

void EditorInspector::_property_checked(const String &p_path, bool p_checked) {

	if (!object)
		return;

	//property checked
	if (autoclear) {

		if (!p_checked) {
			object->set(p_path, Variant());
		} else {

			Variant to_create;
			List<PropertyInfo> pinfo;
			object->get_property_list(&pinfo);
			for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
				if (E->get().name == p_path) {
					Variant::CallError ce;
					to_create = Variant::construct(E->get().type, NULL, 0, ce);
					break;
				}
			}
			object->set(p_path, to_create);
		}

		if (editor_property_map.has(p_path)) {
			for (List<EditorProperty *>::Element *E = editor_property_map[p_path].front(); E; E = E->next()) {
				E->get()->update_property();
				E->get()->update_reload_status();
			}
		}

	} else {
		emit_signal("property_toggled", p_path, p_checked);
	}
}

void EditorInspector::_property_selected(const String &p_path, int p_focusable) {

	property_selected = p_path;
	property_focusable = p_focusable;
	//deselect the others
	for (Map<StringName, List<EditorProperty *> >::Element *F = editor_property_map.front(); F; F = F->next()) {
		if (F->key() == property_selected)
			continue;
		for (List<EditorProperty *>::Element *E = F->get().front(); E; E = E->next()) {
			if (E->get()->is_selected())
				E->get()->deselect();
		}
	}
}

void EditorInspector::_object_id_selected(const String &p_path, ObjectID p_id) {

	emit_signal("object_id_selected", p_id);
}

void EditorInspector::_resource_selected(const String &p_path, RES p_resource) {
	emit_signal("resource_selected", p_resource, p_path);
}

void EditorInspector::_node_removed(Node *p_node) {

	if (p_node == object) {
		edit(NULL);
	}
}

void EditorInspector::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		get_tree()->connect("node_removed", this, "_node_removed");
		add_style_override("bg", get_stylebox("bg", "Tree"));
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {

		get_tree()->disconnect("node_removed", this, "_node_removed");
		edit(NULL);
	}

	if (p_what == NOTIFICATION_PROCESS) {

		if (refresh_countdown > 0) {
			refresh_countdown -= get_process_delta_time();
			if (refresh_countdown <= 0) {
				for (Map<StringName, List<EditorProperty *> >::Element *F = editor_property_map.front(); F; F = F->next()) {
					for (List<EditorProperty *>::Element *E = F->get().front(); E; E = E->next()) {
						E->get()->update_property();
						E->get()->update_reload_status();
					}
				}
			}
		}

		changing++;

		if (update_tree_pending) {

			update_tree();
			update_tree_pending = false;
			pending.clear();

		} else {

			while (pending.size()) {
				StringName prop = pending.front()->get();
				if (editor_property_map.has(prop)) {
					for (List<EditorProperty *>::Element *E = editor_property_map[prop].front(); E; E = E->next()) {
						E->get()->update_property();
						E->get()->update_reload_status();
					}
				}
				pending.erase(pending.front());
			}
		}

		changing--;
	}

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		update_tree();
	}
}

void EditorInspector::_changed_callback(Object *p_changed, const char *p_prop) {
	//this is called when property change is notified via _change_notify()
	_edit_request_change(p_changed, p_prop);
}

void EditorInspector::_bind_methods() {

	ClassDB::bind_method("_multiple_properties_changed", &EditorInspector::_multiple_properties_changed);
	ClassDB::bind_method("_property_changed", &EditorInspector::_property_changed);
	ClassDB::bind_method("_edit_request_change", &EditorInspector::_edit_request_change);
	ClassDB::bind_method("_node_removed", &EditorInspector::_node_removed);
	ClassDB::bind_method("_filter_changed", &EditorInspector::_filter_changed);
	ClassDB::bind_method("_property_keyed", &EditorInspector::_property_keyed);
	ClassDB::bind_method("_property_keyed_with_value", &EditorInspector::_property_keyed_with_value);
	ClassDB::bind_method("_property_checked", &EditorInspector::_property_checked);
	ClassDB::bind_method("_property_selected", &EditorInspector::_property_selected);
	ClassDB::bind_method("_resource_selected", &EditorInspector::_resource_selected);
	ClassDB::bind_method("_object_id_selected", &EditorInspector::_object_id_selected);

	ADD_SIGNAL(MethodInfo("property_keyed", PropertyInfo(Variant::STRING, "property")));
	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::OBJECT, "res"), PropertyInfo(Variant::STRING, "prop")));
	ADD_SIGNAL(MethodInfo("object_id_selected", PropertyInfo(Variant::INT, "id")));
}

EditorInspector::EditorInspector() {
	object = NULL;
	undo_redo = NULL;
	main_vbox = memnew(VBoxContainer);
	main_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(main_vbox);
	set_enable_h_scroll(false);
	set_enable_v_scroll(true);

	show_categories = false;
	hide_script = true;
	use_doc_hints = false;
	capitalize_paths = false;
	use_filter = false;
	autoclear = false;
	changing = 0;
	use_folding = false;
	update_all_pending = false;
	update_tree_pending = false;
	refresh_countdown = 0;
	read_only = false;
	search_box = NULL;
	keying = false;
	_prop_edited = "property_edited";
	set_process(true);
	property_focusable = -1;
}
