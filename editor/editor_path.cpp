/*************************************************************************/
/*  editor_path.cpp                                                      */
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
#include "editor_path.h"

#include "editor_node.h"
#include "editor_scale.h"

void EditorPath::_add_children_to_popup(Object *p_obj, int p_depth) {

	if (p_depth > 8)
		return;

	List<PropertyInfo> pinfo;
	p_obj->get_property_list(&pinfo);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_EDITOR))
			continue;
		if (E->get().hint != PROPERTY_HINT_RESOURCE_TYPE)
			continue;

		Variant value = p_obj->get(E->get().name);
		if (value.get_type() != Variant::OBJECT)
			continue;
		Object *obj = value;
		if (!obj)
			continue;

		Ref<Texture> icon;

		if (has_icon(obj->get_class(), "EditorIcons"))
			icon = get_icon(obj->get_class(), "EditorIcons");
		else
			icon = get_icon("Object", "EditorIcons");

		int index = popup->get_item_count();
		popup->add_icon_item(icon, E->get().name.capitalize(), objects.size());
		popup->set_item_h_offset(index, p_depth * 10 * EDSCALE);
		objects.push_back(obj->get_instance_id());

		_add_children_to_popup(obj, p_depth + 1);
	}
}

void EditorPath::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT && mb->is_pressed()) {

		Object *obj = ObjectDB::get_instance(history->get_path_object(history->get_path_size() - 1));
		if (!obj)
			return;

		objects.clear();
		popup->clear();
		_add_children_to_popup(obj);
		popup->set_position(get_global_position() + Vector2(0, get_size().height));
		popup->set_size(Size2(get_size().width, 1));
		popup->popup();
	}
}

void EditorPath::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_MOUSE_ENTER: {
			mouse_over = true;
			update();
		} break;
		case NOTIFICATION_MOUSE_EXIT: {
			mouse_over = false;
			update();
		} break;
		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();
			Ref<Font> label_font = get_font("font", "Label");
			Size2i size = get_size();
			Ref<Texture> sn = get_icon("SmallNext", "EditorIcons");
			Ref<StyleBox> sb = get_stylebox("pressed", "Button");

			int ofs = sb->get_margin(MARGIN_LEFT);

			if (mouse_over) {
				draw_style_box(sb, Rect2(Point2(), get_size()));
			}

			for (int i = 0; i < history->get_path_size(); i++) {

				Object *obj = ObjectDB::get_instance(history->get_path_object(i));
				if (!obj)
					continue;

				String type = obj->get_class();

				Ref<Texture> icon;

				if (has_icon(obj->get_class(), "EditorIcons"))
					icon = get_icon(obj->get_class(), "EditorIcons");
				else
					icon = get_icon("Object", "EditorIcons");

				icon->draw(ci, Point2i(ofs, (size.height - icon->get_height()) / 2));

				ofs += icon->get_width();

				if (i == history->get_path_size() - 1) {
					//add name
					ofs += 4;
					int left = size.width - ofs;
					if (left < 0)
						continue;
					String name;
					if (Object::cast_to<Resource>(obj)) {

						Resource *r = Object::cast_to<Resource>(obj);
						if (r->get_path().is_resource_file())
							name = r->get_path().get_file();
						else
							name = r->get_name();

						if (name == "")
							name = r->get_class();
					} else if (obj->is_class("ScriptEditorDebuggerInspectedObject"))
						name = obj->call("get_title");
					else if (Object::cast_to<Node>(obj))
						name = Object::cast_to<Node>(obj)->get_name();
					else if (Object::cast_to<Resource>(obj) && Object::cast_to<Resource>(obj)->get_name() != "")
						name = Object::cast_to<Resource>(obj)->get_name();
					else
						name = obj->get_class();

					set_tooltip(obj->get_class());

					label_font->draw(ci, Point2i(ofs, (size.height - label_font->get_height()) / 2 + label_font->get_ascent()), name, get_color("font_color", "Label"), left);
				} else {
					//add arrow

					//sn->draw(ci,Point2i(ofs,(size.height-sn->get_height())/2));
					//ofs+=sn->get_width();
					ofs += 5; //just looks better! somehow
				}
			}

		} break;
	}
}

void EditorPath::update_path() {

	update();
}

void EditorPath::_popup_select(int p_idx) {

	ERR_FAIL_INDEX(p_idx, objects.size());

	Object *obj = ObjectDB::get_instance(objects[p_idx]);
	if (!obj)
		return;

	EditorNode::get_singleton()->push_item(obj);
}

void EditorPath::_bind_methods() {

	ClassDB::bind_method("_gui_input", &EditorPath::_gui_input);
	ClassDB::bind_method("_popup_select", &EditorPath::_popup_select);
}

EditorPath::EditorPath(EditorHistory *p_history) {

	history = p_history;
	mouse_over = false;
	popup = memnew(PopupMenu);
	popup->connect("id_pressed", this, "_popup_select");
	add_child(popup);
}
