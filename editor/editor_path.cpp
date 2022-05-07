/*************************************************************************/
/*  editor_path.cpp                                                      */
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

#include "editor_path.h"

#include "editor_node.h"
#include "editor_scale.h"

void EditorPath::_add_children_to_popup(Object *p_obj, int p_depth) {
	if (p_depth > 8) {
		return;
	}

	List<PropertyInfo> pinfo;
	p_obj->get_property_list(&pinfo);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		if (!(E->get().usage & PROPERTY_USAGE_EDITOR)) {
			continue;
		}
		if (E->get().hint != PROPERTY_HINT_RESOURCE_TYPE) {
			continue;
		}

		Variant value = p_obj->get(E->get().name);
		if (value.get_type() != Variant::OBJECT) {
			continue;
		}
		Object *obj = value;
		if (!obj) {
			continue;
		}

		Ref<Texture> icon = EditorNode::get_singleton()->get_object_icon(obj);

		String proper_name = "";
		Vector<String> name_parts = E->get().name.split("/");

		for (int i = 0; i < name_parts.size(); i++) {
			if (i > 0) {
				proper_name += " > ";
			}
			proper_name += name_parts[i].capitalize();
		}

		int index = sub_objects_menu->get_item_count();
		sub_objects_menu->add_icon_item(icon, proper_name, objects.size());
		sub_objects_menu->set_item_h_offset(index, p_depth * 10 * EDSCALE);
		objects.push_back(obj->get_instance_id());

		_add_children_to_popup(obj, p_depth + 1);
	}
}

void EditorPath::_show_popup() {
	sub_objects_menu->clear();

	Size2 size = get_size();
	Point2 gp = get_global_position();
	gp.y += size.y;

	sub_objects_menu->set_position(gp);
	sub_objects_menu->set_size(Size2(size.width, 1));
	sub_objects_menu->set_parent_rect(Rect2(Point2(gp - sub_objects_menu->get_position()), size));

	sub_objects_menu->popup();
}

void EditorPath::_about_to_show() {
	Object *obj = ObjectDB::get_instance(history->get_path_object(history->get_path_size() - 1));
	if (!obj) {
		return;
	}

	objects.clear();

	_add_children_to_popup(obj);
	if (sub_objects_menu->get_item_count() == 0) {
		sub_objects_menu->add_item(TTR("No sub-resources found."));
		sub_objects_menu->set_item_disabled(0, true);
	}
}

void EditorPath::update_path() {
	for (int i = 0; i < history->get_path_size(); i++) {
		Object *obj = ObjectDB::get_instance(history->get_path_object(i));
		if (!obj) {
			continue;
		}

		Ref<Texture> icon = EditorNode::get_singleton()->get_object_icon(obj);
		if (icon.is_valid()) {
			current_object_icon->set_texture(icon);
		}

		if (i == history->get_path_size() - 1) {
			String name;
			if (Object::cast_to<Resource>(obj)) {
				Resource *r = Object::cast_to<Resource>(obj);
				if (r->get_path().is_resource_file()) {
					name = r->get_path().get_file();
				} else {
					name = r->get_name();
				}

				if (name == "") {
					name = r->get_class();
				}
			} else if (obj->is_class("ScriptEditorDebuggerInspectedObject")) {
				name = obj->call("get_title");
			} else if (Object::cast_to<Node>(obj)) {
				name = Object::cast_to<Node>(obj)->get_name();
			} else if (Object::cast_to<Resource>(obj) && Object::cast_to<Resource>(obj)->get_name() != "") {
				name = Object::cast_to<Resource>(obj)->get_name();
			} else {
				name = obj->get_class();
			}

			current_object_label->set_text(" " + name); // An extra space so the text is not too close of the icon.
			set_tooltip(obj->get_class());
		}
	}
}

void EditorPath::clear_path() {
	set_disabled(true);
	set_tooltip("");

	current_object_label->set_text("");
	current_object_icon->set_texture(nullptr);
	sub_objects_icon->set_visible(false);
}

void EditorPath::enable_path() {
	set_disabled(false);
	sub_objects_icon->set_visible(true);
}

void EditorPath::_id_pressed(int p_idx) {
	ERR_FAIL_INDEX(p_idx, objects.size());

	Object *obj = ObjectDB::get_instance(objects[p_idx]);
	if (!obj) {
		return;
	}

	EditorNode::get_singleton()->push_item(obj);
}

void EditorPath::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			update_path();

			// Button overrides Control's method, so we have to improvise.
			sub_objects_icon->set_texture(sub_objects_icon->get_icon("select_arrow", "Tree"));
			current_object_label->add_font_override("font", get_font("main", "EditorFonts"));
		} break;

		case NOTIFICATION_READY: {
			connect("pressed", this, "_show_popup");
		} break;
	}
}

void EditorPath::_bind_methods() {
	ClassDB::bind_method("_show_popup", &EditorPath::_show_popup);
	ClassDB::bind_method("_about_to_show", &EditorPath::_about_to_show);
	ClassDB::bind_method("_id_pressed", &EditorPath::_id_pressed);
}

EditorPath::EditorPath(EditorHistory *p_history) {
	history = p_history;

	MarginContainer *main_mc = memnew(MarginContainer);
	main_mc->set_anchors_and_margins_preset(PRESET_WIDE);
	main_mc->add_constant_override("margin_left", 4 * EDSCALE);
	main_mc->add_constant_override("margin_right", 6 * EDSCALE);
	main_mc->set_mouse_filter(MOUSE_FILTER_PASS);
	add_child(main_mc);

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_mc->add_child(main_hb);

	current_object_icon = memnew(TextureRect);
	current_object_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	main_hb->add_child(current_object_icon);

	current_object_label = memnew(Label);
	current_object_label->set_clip_text(true);
	current_object_label->set_align(Label::ALIGN_LEFT);
	current_object_label->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hb->add_child(current_object_label);

	sub_objects_icon = memnew(TextureRect);
	sub_objects_icon->set_visible(false);
	sub_objects_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	main_hb->add_child(sub_objects_icon);

	sub_objects_menu = memnew(PopupMenu);
	add_child(sub_objects_menu);
	sub_objects_menu->connect("about_to_show", this, "_about_to_show");
	sub_objects_menu->connect("id_pressed", this, "_id_pressed");

	set_tooltip(TTR("Open a list of sub-resources."));
}
