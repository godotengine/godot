/*************************************************************************/
/*  resource_preview.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "resource_preview.h"

void ResourcePreview::toggle_preview(bool hidden) {
	int child_count = get_child_count();

	if (!hidden) {
		for (int i = 1; i < child_count; i++) {
			Control *control = Object::cast_to<Control>(get_child(i));

			control->show();
		}
		collapse_button->set_normal_texture(get_icon("PanelCollapse", "EditorIcons"));
	} else {
		for (int i = 1; i < child_count; i++) {
			Control *control = Object::cast_to<Control>(get_child(i));
			control->hide();
		}
		collapse_button->set_normal_texture(get_icon("PanelExpand", "EditorIcons"));
	}
	collapse_button->set_modulate(Color(1, 1, 1, 0.5));
}

void ResourcePreview::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		collapse_button->set_normal_texture(get_icon("PanelCollapse", "EditorIcons"));
	}
}

ResourcePreview::ResourcePreview() {
	this->set_area_as_parent_rect();
	collapse_button = memnew(TextureButton);
	collapse_button->set_custom_minimum_size(Vector2(36, 12));
	collapse_button->set_anchors_preset(PRESET_TOP_WIDE, true);
	collapse_button->set_expand(true);
	collapse_button->set_stretch_mode(TextureButton::STRETCH_KEEP_CENTERED);
	collapse_button->set_margin(MARGIN_RIGHT, 0);
	collapse_button->set_modulate(Color(1, 1, 1, 0.5));
	collapse_button->set_toggle_mode(true);

	collapse_button->connect("toggled", this, "toggle_preview");
	collapse_button->connect("mouse_entered", collapse_button, "set_modulate", varray(Color(1, 1, 1, 1)));
	collapse_button->connect("mouse_exited", collapse_button, "set_modulate", varray(Color(1, 1, 1, 0.5)));

	add_child(collapse_button);
}

void ResourcePreview::_bind_methods() {
	ClassDB::bind_method("toggle_preview", &ResourcePreview::toggle_preview);
}

ResourcePreview::~ResourcePreview() {
}
