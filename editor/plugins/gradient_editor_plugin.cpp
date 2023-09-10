/**************************************************************************/
/*  gradient_editor_plugin.cpp                                            */
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

#include "gradient_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "node_3d_editor_plugin.h"

void GradientReverseButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<Texture2D> icon = get_editor_theme_icon(SNAME("ReverseGradient"));
			if (is_pressed()) {
				draw_texture_rect(icon, Rect2(margin, margin, icon->get_width(), icon->get_height()), false, get_theme_color(SNAME("icon_pressed_color"), SNAME("Button")));
			} else {
				draw_texture_rect(icon, Rect2(margin, margin, icon->get_width(), icon->get_height()));
			}
		} break;
	}
}

Size2 GradientReverseButton::get_minimum_size() const {
	return (get_editor_theme_icon(SNAME("ReverseGradient"))->get_size() + Size2(margin * 2, margin * 2));
}

///////////////////////

bool EditorInspectorPluginGradient::can_handle(Object *p_object) {
	return Object::cast_to<Gradient>(p_object) != nullptr;
}

void EditorInspectorPluginGradient::parse_begin(Object *p_object) {
	Gradient *gradient = Object::cast_to<Gradient>(p_object);
	Ref<Gradient> g(gradient);

	editor = memnew(GradientEditor);
	editor->set_gradient(g);
	add_custom_control(editor);

	int picker_shape = EDITOR_GET("interface/inspector/default_color_picker_shape");
	editor->get_picker()->set_picker_shape((ColorPicker::PickerShapeType)picker_shape);

	reverse_btn = memnew(GradientReverseButton);

	gradient_tools_hbox = memnew(HBoxContainer);
	gradient_tools_hbox->add_child(reverse_btn);

	add_custom_control(gradient_tools_hbox);

	reverse_btn->connect("pressed", callable_mp(this, &EditorInspectorPluginGradient::_reverse_button_pressed));
	reverse_btn->set_tooltip_text(TTR("Reverse/mirror gradient."));
}

void EditorInspectorPluginGradient::_reverse_button_pressed() {
	editor->reverse_gradient();
}

GradientEditorPlugin::GradientEditorPlugin() {
	Ref<EditorInspectorPluginGradient> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
