/*************************************************************************/
/*  editor_path.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

void EditorPath::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_DRAW: {

			RID ci = get_canvas_item();
			Ref<Font> label_font = get_font("font", "Label");
			Size2i size = get_size();
			Ref<Texture> sn = get_icon("SmallNext", "EditorIcons");

			int ofs = 5;
			for (int i = 0; i < history->get_path_size(); i++) {

				Object *obj = ObjectDB::get_instance(history->get_path_object(i));
				if (!obj)
					continue;

				String type = obj->get_type();

				Ref<Texture> icon;

				if (has_icon(obj->get_type(), "EditorIcons"))
					icon = get_icon(obj->get_type(), "EditorIcons");
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
					if (obj->cast_to<Resource>()) {

						Resource *r = obj->cast_to<Resource>();
						if (r->get_path().is_resource_file())
							name = r->get_path().get_file();
						else
							name = r->get_name();

						if (name == "")
							name = r->get_type();
					} else if (obj->cast_to<Node>()) {

						name = obj->cast_to<Node>()->get_name();
					} else if (obj->cast_to<Resource>() && obj->cast_to<Resource>()->get_name() != "") {
						name = obj->cast_to<Resource>()->get_name();
					} else {
						name = obj->get_type();
					}

					set_tooltip(obj->get_type());

					label_font->draw(ci, Point2i(ofs, (size.height - label_font->get_height()) / 2 + label_font->get_ascent()), name, Color(1, 1, 1), left);
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

EditorPath::EditorPath(EditorHistory *p_history) {

	history = p_history;
}
