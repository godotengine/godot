/**************************************************************************/
/*  editor_path.cpp                                                       */
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

#include "editor_path.h"

#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/multi_node_edit.h"

String EditorPath::_get_human_readable_name(Object *p_obj, String p_prop_name) const {
	String obj_text;
	if (p_obj->has_method("_get_editor_name")) {
		obj_text = p_obj->call("_get_editor_name");
	} else if (Object::cast_to<Resource>(p_obj)) {
		Resource *r = Object::cast_to<Resource>(p_obj);
		if (r->get_path().is_resource_file()) {
			obj_text = r->get_path().get_file();
		} else if (!r->get_name().is_empty()) {
			obj_text = r->get_name();
		} else {
			obj_text = r->get_class();
		}
	} else if (Object::cast_to<Node>(p_obj)) {
		obj_text = Object::cast_to<Node>(p_obj)->get_name();
	} else if (p_obj->is_class("EditorDebuggerRemoteObject")) {
		obj_text = p_obj->call("get_title");
	} else {
		obj_text = p_obj->get_class();
	}

	if (p_prop_name.is_empty()) {
		return obj_text;
	}

	String obj_name;
	Vector<String> name_parts = p_prop_name.split("/");
	for (int i = 0; i < name_parts.size(); i++) {
		if (i > 0) {
			obj_name += " > ";
		}
		obj_name += name_parts[i].capitalize();
	}

	return vformat("%s (%s)", obj_name, obj_text);
}

void EditorPath::_add_children_to_popup() {
	ObjectID selected_obj_id = history->get_path_object(history->get_path_size() - 1);

	for (int i = 0; i < objects.size(); i++) {
		const Property &prop = objects[i];

		ObjectID obj_id = prop.value;
		Object *obj = ObjectDB::get_instance(obj_id);

		String obj_name;
		Ref<Texture2D> obj_icon;
		if (obj) {
			obj_name = _get_human_readable_name(obj, prop.name);

			if (Object::cast_to<MultiNodeEdit>(obj)) {
				obj_icon = EditorNode::get_singleton()->get_class_icon(Object::cast_to<MultiNodeEdit>(obj)->get_edited_class_name());
			} else {
				obj_icon = EditorNode::get_singleton()->get_object_icon(obj);
			}
		}

		int index = sub_objects_menu->get_item_count();
		sub_objects_menu->add_icon_item(obj_icon, obj_name, index - 1);
		sub_objects_menu->set_item_indent(index, prop.depth);
		sub_objects_menu->set_item_disabled(index, selected_obj_id == obj_id);

		if (i == 0) {
			sub_objects_menu->add_separator();
		}
	}
}

void EditorPath::_add_object_properties(Object *p_obj, int p_depth) {
	if (p_depth > 8) {
		return;
	}

	List<PropertyInfo> pinfo;
	p_obj->get_property_list(&pinfo);
	for (const PropertyInfo &E : pinfo) {
		if (!(E.usage & PROPERTY_USAGE_EDITOR)) {
			continue;
		}
		if (E.hint != PROPERTY_HINT_RESOURCE_TYPE) {
			continue;
		}

		Variant value = p_obj->get(E.name);
		if (value.get_type() != Variant::OBJECT) {
			continue;
		}
		Object *obj = value;
		if (!obj) {
			continue;
		}

		Property prop;
		prop.name = E.name;
		prop.value = obj->get_instance_id();
		prop.depth = p_depth;

		objects.push_back(prop);
		_add_object_properties(obj, p_depth + 1);
	}
}

void EditorPath::_toggle_popup() {
	if (sub_objects_menu->is_visible()) {
		sub_objects_menu->hide();
		return;
	}

	sub_objects_menu->clear();

	Size2 size = get_size();
	Point2 gp = get_screen_position();
	gp.y += size.y;

	sub_objects_menu->set_position(gp);
	sub_objects_menu->set_size(Size2(size.width, 1));
	sub_objects_menu->set_parent_rect(Rect2(Point2(gp - sub_objects_menu->get_position()), size));

	sub_objects_menu->take_mouse_focus();
	sub_objects_menu->popup();
}

void EditorPath::_about_to_show() {
	ObjectID base_object_id = history->get_path_object(0);
	Object *base_obj = ObjectDB::get_instance(base_object_id);
	if (!base_obj) {
		return;
	}

	Property base_object_prop;
	base_object_prop.name = "";
	base_object_prop.value = base_object_id;
	base_object_prop.depth = 0;

	objects.clear();
	objects.push_back(base_object_prop);
	_add_object_properties(base_obj);

	_add_children_to_popup();

	if (objects.size() == 1) {
		sub_objects_menu->add_item(TTR("No sub-resources found."));
		sub_objects_menu->set_item_disabled(sub_objects_menu->get_item_count() - 1, true);
	}
}

void EditorPath::update_path() {
	Object *obj = ObjectDB::get_instance(history->get_path_object(history->get_path_size() - 1));
	if (!obj) {
		return;
	}

	Ref<Texture2D> obj_icon;
	if (Object::cast_to<MultiNodeEdit>(obj)) {
		obj_icon = EditorNode::get_singleton()->get_class_icon(Object::cast_to<MultiNodeEdit>(obj)->get_edited_class_name());
	} else {
		obj_icon = EditorNode::get_singleton()->get_object_icon(obj);
	}

	if (obj_icon.is_valid()) {
		current_object_icon->set_texture(obj_icon);
	}

	String name = _get_human_readable_name(obj);

	current_object_label->set_text(name);
	set_tooltip_text(obj->get_class());
}

void EditorPath::clear_path() {
	set_disabled(true);
	set_tooltip_text("");

	current_object_label->set_text("");
	current_object_icon->set_texture(nullptr);
	sub_objects_icon->hide();
}

void EditorPath::enable_path() {
	set_disabled(false);
	sub_objects_icon->show();
}

void EditorPath::_id_pressed(int p_id) {
	ERR_FAIL_INDEX(p_id, objects.size());

	Object *obj = ObjectDB::get_instance(objects[p_id].value);
	if (!obj) {
		return;
	}

	EditorNode::get_singleton()->push_item(obj, objects[p_id].name);
}

void EditorPath::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			update_path();

			sub_objects_icon->set_texture(get_theme_icon(SNAME("arrow"), SNAME("OptionButton")));
			current_object_label->add_theme_font_override("font", get_theme_font(SNAME("main"), SNAME("EditorFonts")));
		} break;

		case NOTIFICATION_READY: {
			connect("pressed", callable_mp(this, &EditorPath::_toggle_popup));
		} break;
	}
}

void EditorPath::_bind_methods() {
}

EditorPath::EditorPath(EditorSelectionHistory *p_history) {
	history = p_history;

	MarginContainer *main_mc = memnew(MarginContainer);
	main_mc->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	main_mc->add_theme_constant_override("margin_left", 4 * EDSCALE);
	main_mc->add_theme_constant_override("margin_right", 6 * EDSCALE);
	add_child(main_mc);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_mc->add_child(main_hb);

	current_object_icon = memnew(TextureRect);
	current_object_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	main_hb->add_child(current_object_icon);

	current_object_label = memnew(Label);
	current_object_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	current_object_label->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hb->add_child(current_object_label);

	sub_objects_icon = memnew(TextureRect);
	sub_objects_icon->hide();
	sub_objects_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	main_hb->add_child(sub_objects_icon);

	sub_objects_menu = memnew(PopupMenu);
	add_child(sub_objects_menu);
	sub_objects_menu->connect("about_to_popup", callable_mp(this, &EditorPath::_about_to_show));
	sub_objects_menu->connect("id_pressed", callable_mp(this, &EditorPath::_id_pressed));

	set_tooltip_text(TTR("Open a list of sub-resources."));
}
