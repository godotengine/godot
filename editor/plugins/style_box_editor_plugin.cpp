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

#include "editor/editor_scale.h"
#include "scene/gui/texture_button.h"

bool StyleBoxPreview::grid_preview_enabled = true;

void StyleBoxPreview::_grid_preview_toggled(bool p_active) {
	grid_preview_enabled = p_active;
	preview->queue_redraw();
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

void StyleBoxPreview::edit(const Ref<StyleBox> &p_stylebox) {
	if (stylebox.is_valid()) {
		stylebox->disconnect("changed", callable_mp(this, &StyleBoxPreview::_sb_changed));
	}
	stylebox = p_stylebox;
	if (p_stylebox.is_valid()) {
		preview->add_theme_style_override("panel", stylebox);
		stylebox->connect("changed", callable_mp(this, &StyleBoxPreview::_sb_changed));
	}
	Ref<StyleBoxTexture> sbt = p_stylebox;
	grid_preview->set_visible(sbt.is_valid());
	_sb_changed();
}

void StyleBoxPreview::_sb_changed() {
	preview->queue_redraw();
}

void StyleBoxPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (!is_inside_tree()) {
				// TODO: This is a workaround because `NOTIFICATION_THEME_CHANGED`
				// is getting called for some reason when the `TexturePreview` is
				// getting destroyed, which causes `get_theme_font()` to return `nullptr`.
				// See https://github.com/godotengine/godot/issues/50743.
				break;
			}
			grid_preview->set_texture_normal(get_theme_icon(SNAME("StyleBoxGridInvisible"), SNAME("EditorIcons")));
			grid_preview->set_texture_pressed(get_theme_icon(SNAME("StyleBoxGridVisible"), SNAME("EditorIcons")));
			grid_preview->set_texture_hover(get_theme_icon(SNAME("StyleBoxGridVisible"), SNAME("EditorIcons")));
			checkerboard->set_texture(get_theme_icon(SNAME("Checkerboard"), SNAME("EditorIcons")));
		} break;
	}
}

void StyleBoxPreview::_redraw() {
	if (stylebox.is_valid()) {
		Ref<Texture2D> grid_texture_disabled = get_theme_icon(SNAME("StyleBoxGridInvisible"), SNAME("EditorIcons"));
		Rect2 preview_rect = preview->get_rect();
		preview_rect.position += grid_texture_disabled->get_size();
		preview_rect.size -= grid_texture_disabled->get_size() * 2;

		// Re-adjust preview panel to fit all drawn content
		Rect2 draw_rect = stylebox->get_draw_rect(preview_rect);
		preview_rect.size -= draw_rect.size - preview_rect.size;
		preview_rect.position -= draw_rect.position - preview_rect.position;

		preview->draw_style_box(stylebox, preview_rect);

		Ref<StyleBoxTexture> sbt = stylebox;
		if (sbt.is_valid() && grid_preview->is_pressed()) {
			for (int i = 0; i < 2; i++) {
				Color c = i == 1 ? Color(1, 1, 1, 0.8) : Color(0, 0, 0, 0.4);
				int x = draw_rect.position.x + sbt->get_margin(SIDE_LEFT) + (1 - i);
				preview->draw_line(Point2(x, 0), Point2(x, preview->get_size().height), c);
				int x2 = draw_rect.position.x + draw_rect.size.width - sbt->get_margin(SIDE_RIGHT) + (1 - i);
				preview->draw_line(Point2(x2, 0), Point2(x2, preview->get_size().height), c);
				int y = draw_rect.position.y + sbt->get_margin(SIDE_TOP) + (1 - i);
				preview->draw_line(Point2(0, y), Point2(preview->get_size().width, y), c);
				int y2 = draw_rect.position.y + draw_rect.size.height - sbt->get_margin(SIDE_BOTTOM) + (1 - i);
				preview->draw_line(Point2(0, y2), Point2(preview->get_size().width, y2), c);
			}
		}
	}
}

void StyleBoxPreview::_bind_methods() {
}

StyleBoxPreview::StyleBoxPreview() {
	checkerboard = memnew(TextureRect);
	checkerboard->set_stretch_mode(TextureRect::STRETCH_TILE);
	checkerboard->set_texture_repeat(CanvasItem::TEXTURE_REPEAT_ENABLED);
	checkerboard->set_custom_minimum_size(Size2(0.0, 150.0) * EDSCALE);

	preview = memnew(Control);
	preview->set_clip_contents(true);
	preview->connect("draw", callable_mp(this, &StyleBoxPreview::_redraw));
	checkerboard->add_child(preview);
	preview->set_anchors_and_offsets_preset(PRESET_FULL_RECT);

	add_margin_child(TTR("Preview:"), checkerboard);
	grid_preview = memnew(TextureButton);
	preview->add_child(grid_preview);
	grid_preview->set_toggle_mode(true);
	grid_preview->connect("toggled", callable_mp(this, &StyleBoxPreview::_grid_preview_toggled));
	grid_preview->set_pressed(grid_preview_enabled);
}

StyleBoxEditorPlugin::StyleBoxEditorPlugin() {
	Ref<EditorInspectorPluginStyleBox> inspector_plugin;
	inspector_plugin.instantiate();
	add_inspector_plugin(inspector_plugin);
}
