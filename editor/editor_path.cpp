/*************************************************************************/
/*  editor_path.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

		Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(obj);

		int index = get_popup()->get_item_count();
		get_popup()->add_icon_item(icon, E->get().name.capitalize(), objects.size());
		get_popup()->set_item_h_offset(index, p_depth * 10 * EDSCALE);
		objects.push_back(obj->get_instance_id());

		_add_children_to_popup(obj, p_depth + 1);
	}
}

void EditorPath::_about_to_show() {
	Object *obj = ObjectDB::get_instance(history->get_path_object(history->get_path_size() - 1));
	if (!obj) {
		return;
	}

	objects.clear();
	get_popup()->clear();
	get_popup()->set_size(Size2(get_size().width, 1));

	_add_children_to_popup(obj);
	if (get_popup()->get_item_count() == 0) {
		get_popup()->add_item(TTR("No sub-resources found."));
		get_popup()->set_item_disabled(0, true);
	}
}

void EditorPath::update_path() {
	for (int i = 0; i < history->get_path_size(); i++) {
		Object *obj = ObjectDB::get_instance(history->get_path_object(i));
		if (!obj) {
			continue;
		}

		Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(obj);
		if (icon.is_valid()) {
			set_icon(icon);
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
			} else if (obj->is_class("EditorDebuggerRemoteObject")) {
				name = obj->call("get_title");
			} else if (Object::cast_to<Node>(obj)) {
				name = Object::cast_to<Node>(obj)->get_name();
			} else if (Object::cast_to<Resource>(obj) && Object::cast_to<Resource>(obj)->get_name() != "") {
				name = Object::cast_to<Resource>(obj)->get_name();
			} else {
				name = obj->get_class();
			}

			set_text(" " + name); // An extra space so the text is not too close of the icon.
			set_tooltip(obj->get_class());
		}
	}
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
		case NOTIFICATION_THEME_CHANGED: {
			update_path();
		} break;
	}
}

void EditorPath::_bind_methods() {
}

EditorPath::EditorPath(EditorHistory *p_history) {
	history = p_history;
	set_clip_text(true);
	set_text_align(ALIGN_LEFT);
	get_popup()->connect("about_to_popup", callable_mp(this, &EditorPath::_about_to_show));
	get_popup()->connect("id_pressed", callable_mp(this, &EditorPath::_id_pressed));
}
