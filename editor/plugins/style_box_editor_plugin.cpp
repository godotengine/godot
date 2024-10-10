/**************************************************************************/
/*  style_box_editor_plugin.cpp                                           */
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

#include "style_box_editor_plugin.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/resources/style_box_texture.h"

bool StyleBoxPreview::grid_preview_enabled = true;

void StyleBoxPreview::_grid_preview_toggled(bool p_active) {
	grid_preview_enabled = p_active;
	queue_redraw();
}

void StyleBoxPreview::edit(const Ref<StyleBox> &p_stylebox) {
	if (stylebox.is_valid()) {
		stylebox->disconnect_changed(callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
	}
	stylebox = p_stylebox;
	if (stylebox.is_valid()) {
		stylebox->connect_changed(callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
	}
	Ref<StyleBoxTexture> sbt = stylebox;
	grid_preview->set_visible(sbt.is_valid());
	queue_redraw();
}

void StyleBoxPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			set_texture(get_editor_theme_icon(SNAME("Checkerboard")));
			grid_preview->set_button_icon(get_editor_theme_icon(SNAME("StyleBoxGrid")));
		} break;
		case NOTIFICATION_DRAW: {
			_redraw();
		} break;
	}
}

void StyleBoxPreview::_redraw() {
	if (stylebox.is_valid()) {
		float grid_button_width = get_editor_theme_icon(SNAME("StyleBoxGrid"))->get_size().x;
		Rect2 preview_rect = get_rect();
		preview_rect = preview_rect.grow(-grid_button_width);

		// Re-adjust preview panel to fit all drawn content.
		Rect2 drawing_rect = stylebox->get_draw_rect(preview_rect);
		preview_rect.size -= drawing_rect.size - preview_rect.size;
		preview_rect.position -= drawing_rect.position - preview_rect.position;

		draw_style_box(stylebox, preview_rect);

		Ref<StyleBoxTexture> sbt = stylebox;
		// Draw the "grid". Use white lines, as well as subtle black lines to ensure contrast.
		if (sbt.is_valid() && grid_preview->is_pressed()) {
			const Color dark_color = Color(0, 0, 0, 0.4);
			const Color bright_color = Color(1, 1, 1, 0.8);
			int x_left = drawing_rect.position.x + sbt->get_margin(SIDE_LEFT);
			int x_right = drawing_rect.position.x + drawing_rect.size.width - sbt->get_margin(SIDE_RIGHT);
			int y_top = drawing_rect.position.y + sbt->get_margin(SIDE_TOP);
			int y_bottom = drawing_rect.position.y + drawing_rect.size.height - sbt->get_margin(SIDE_BOTTOM);

			draw_line(Point2(x_left + 2, 0), Point2(x_left + 2, get_size().height), dark_color);
			draw_line(Point2(x_right + 1, 0), Point2(x_right + 1, get_size().height), dark_color);
			draw_line(Point2(0, y_top + 2), Point2(get_size().width, y_top + 2), dark_color);
			draw_line(Point2(0, y_bottom + 1), Point2(get_size().width, y_bottom + 1), dark_color);

			draw_line(Point2(x_left + 1, 0), Point2(x_left + 1, get_size().height), bright_color);
			draw_line(Point2(x_right, 0), Point2(x_right, get_size().height), bright_color);
			draw_line(Point2(0, y_top + 1), Point2(get_size().width, y_top + 1), bright_color);
			draw_line(Point2(0, y_bottom), Point2(get_size().width, y_bottom), bright_color);
		}
	}
}

StyleBoxPreview::StyleBoxPreview() {
	set_clip_contents(true);
	set_custom_minimum_size(Size2(0, 150) * EDSCALE);
	set_stretch_mode(TextureRect::STRETCH_TILE);
	set_texture_repeat(CanvasItem::TEXTURE_REPEAT_ENABLED);
	set_anchors_and_offsets_preset(PRESET_FULL_RECT);

	grid_preview = memnew(Button);
	// This theme variation works better than the normal theme because there's no focus highlight.
	grid_preview->set_theme_type_variation("PreviewLightButton");
	grid_preview->set_toggle_mode(true);
	grid_preview->connect(SceneStringName(toggled), callable_mp(this, &StyleBoxPreview::_grid_preview_toggled));
	grid_preview->set_pressed(grid_preview_enabled);
	add_child(grid_preview);
}

bool EditorInspectorPluginStyleBox::can_handle(Object *p_object) {
	return Object::cast_to<StyleBox>(p_object) != nullptr;
}

void EditorInspectorPluginStyleBox::parse_begin(Object *p_object) {
	Ref<StyleBox> sb = Ref<StyleBox>(Object::cast_to<StyleBox>(p_object));

	StyleBoxPreview *preview = memnew(StyleBoxPreview);
	preview->edit(sb);
	add_custom_control(preview);
}

StyleBoxEditorPlugin::StyleBoxEditorPlugin() {
	Ref<EditorInspectorPluginStyleBox> inspector_plugin;
	inspector_plugin.instantiate();
	add_inspector_plugin(inspector_plugin);
}
