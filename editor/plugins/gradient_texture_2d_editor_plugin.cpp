/**************************************************************************/
/*  gradient_texture_2d_editor_plugin.cpp                                 */
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

#include "gradient_texture_2d_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/gui/box_container.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/separator.h"

Point2 GradientTexture2DEditorRect::_get_handle_position(const Handle p_handle) {
	// Get the handle's mouse position in pixels relative to offset.
	return (p_handle == HANDLE_FILL_FROM ? texture->get_fill_from() : texture->get_fill_to()).clamp(Vector2(), Vector2(1, 1)) * size;
}

void GradientTexture2DEditorRect::_update_fill_position() {
	if (handle == HANDLE_NONE) {
		return;
	}

	// Update the texture's fill_from/fill_to property based on mouse input.
	Vector2 percent = ((get_local_mouse_position() - offset) / size).clamp(Vector2(), Vector2(1, 1));
	if (snap_enabled) {
		percent = (percent - Vector2(0.5, 0.5)).snapped(Vector2(snap_size, snap_size)) + Vector2(0.5, 0.5);
	}

	String property_name = handle == HANDLE_FILL_FROM ? "fill_from" : "fill_to";

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(vformat(TTR("Set %s"), property_name), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_property(texture.ptr(), property_name, percent);
	undo_redo->add_undo_property(texture.ptr(), property_name, handle == HANDLE_FILL_FROM ? texture->get_fill_from() : texture->get_fill_to());
	undo_redo->commit_action();
}

void GradientTexture2DEditorRect::gui_input(const Ref<InputEvent> &p_event) {
	// Grab/release handle.
	const Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			Point2 mouse_position = mb->get_position() - offset;
			if (Rect2(_get_handle_position(HANDLE_FILL_FROM).round() - handle_size / 2, handle_size).has_point(mouse_position)) {
				handle = HANDLE_FILL_FROM;
			} else if (Rect2(_get_handle_position(HANDLE_FILL_TO).round() - handle_size / 2, handle_size).has_point(mouse_position)) {
				handle = HANDLE_FILL_TO;
			} else {
				handle = HANDLE_NONE;
			}
		} else {
			_update_fill_position();
			handle = HANDLE_NONE;
		}
	}

	// Move handle.
	const Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		_update_fill_position();
	}
}

void GradientTexture2DEditorRect::set_texture(Ref<GradientTexture2D> &p_texture) {
	texture = p_texture;
	texture->connect("changed", callable_mp((CanvasItem *)this, &CanvasItem::queue_redraw));
}

void GradientTexture2DEditorRect::set_snap_enabled(bool p_snap_enabled) {
	snap_enabled = p_snap_enabled;
	queue_redraw();
}

void GradientTexture2DEditorRect::set_snap_size(float p_snap_size) {
	snap_size = p_snap_size;
	queue_redraw();
}

void GradientTexture2DEditorRect::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			checkerboard->set_texture(get_theme_icon(SNAME("GuiMiniCheckerboard"), SNAME("EditorIcons")));
		} break;

		case NOTIFICATION_DRAW: {
			if (texture.is_null()) {
				return;
			}

			const Ref<Texture2D> fill_from_icon = get_theme_icon(SNAME("EditorPathSmoothHandle"), SNAME("EditorIcons"));
			const Ref<Texture2D> fill_to_icon = get_theme_icon(SNAME("EditorPathSharpHandle"), SNAME("EditorIcons"));
			handle_size = fill_from_icon->get_size();

			Size2 rect_size = get_size();

			// Get the size and position to draw the texture and handles at.
			// Subtract handle sizes so they stay inside the preview, but keep the texture's aspect ratio.
			Size2 available_size = rect_size - handle_size;
			Size2 ratio = available_size / texture->get_size();
			size = MIN(ratio.x, ratio.y) * texture->get_size();
			offset = ((rect_size - size) / 2).round();

			checkerboard->set_rect(Rect2(offset, size));

			draw_set_transform(offset);
			draw_texture_rect(texture, Rect2(Point2(), size));

			// Draw grid snap lines.
			if (snap_enabled) {
				const Color primary_line_color = Color(0.5, 0.5, 0.5, 0.9);
				const Color line_color = Color(0.5, 0.5, 0.5, 0.5);

				// Draw border and centered axis lines.
				draw_rect(Rect2(Point2(), size), primary_line_color, false);
				draw_line(Point2(size.width / 2, 0), Point2(size.width / 2, size.height), primary_line_color);
				draw_line(Point2(0, size.height / 2), Point2(size.width, size.height / 2), primary_line_color);

				// Draw vertical lines.
				int prev_idx = 0;
				for (int x = 0; x < size.width; x++) {
					int idx = int((x / size.width - 0.5) / snap_size);

					if (x > 0 && prev_idx != idx) {
						draw_line(Point2(x, 0), Point2(x, size.height), line_color);
					}

					prev_idx = idx;
				}

				// Draw horizontal lines.
				prev_idx = 0;
				for (int y = 0; y < size.height; y++) {
					int idx = int((y / size.height - 0.5) / snap_size);

					if (y > 0 && prev_idx != idx) {
						draw_line(Point2(0, y), Point2(size.width, y), line_color);
					}

					prev_idx = idx;
				}
			}

			// Draw handles.
			draw_texture(fill_from_icon, (_get_handle_position(HANDLE_FILL_FROM) - handle_size / 2).round());
			draw_texture(fill_to_icon, (_get_handle_position(HANDLE_FILL_TO) - handle_size / 2).round());
		} break;
	}
}

GradientTexture2DEditorRect::GradientTexture2DEditorRect() {
	checkerboard = memnew(TextureRect);
	checkerboard->set_stretch_mode(TextureRect::STRETCH_TILE);
	checkerboard->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	checkerboard->set_draw_behind_parent(true);
	add_child(checkerboard, false, INTERNAL_MODE_FRONT);

	set_custom_minimum_size(Size2(0, 250 * EDSCALE));
}

///////////////////////

void GradientTexture2DEditor::_reverse_button_pressed() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Swap GradientTexture2D Fill Points"));
	undo_redo->add_do_property(texture.ptr(), "fill_from", texture->get_fill_to());
	undo_redo->add_do_property(texture.ptr(), "fill_to", texture->get_fill_from());
	undo_redo->add_undo_property(texture.ptr(), "fill_from", texture->get_fill_from());
	undo_redo->add_undo_property(texture.ptr(), "fill_to", texture->get_fill_to());
	undo_redo->commit_action();
}

void GradientTexture2DEditor::_set_snap_enabled(bool p_enabled) {
	texture_editor_rect->set_snap_enabled(p_enabled);

	snap_size_edit->set_visible(p_enabled);
}

void GradientTexture2DEditor::_set_snap_size(float p_snap_size) {
	texture_editor_rect->set_snap_size(MAX(p_snap_size, 0.01));
}

void GradientTexture2DEditor::set_texture(Ref<GradientTexture2D> &p_texture) {
	texture = p_texture;
	texture_editor_rect->set_texture(p_texture);
}

void GradientTexture2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			reverse_button->set_icon(get_theme_icon(SNAME("ReverseGradient"), SNAME("EditorIcons")));
			snap_button->set_icon(get_theme_icon(SNAME("SnapGrid"), SNAME("EditorIcons")));
		} break;
	}
}

GradientTexture2DEditor::GradientTexture2DEditor() {
	HFlowContainer *toolbar = memnew(HFlowContainer);
	add_child(toolbar);

	reverse_button = memnew(Button);
	reverse_button->set_tooltip_text(TTR("Swap Gradient Fill Points"));
	toolbar->add_child(reverse_button);
	reverse_button->connect("pressed", callable_mp(this, &GradientTexture2DEditor::_reverse_button_pressed));

	toolbar->add_child(memnew(VSeparator));

	snap_button = memnew(Button);
	snap_button->set_tooltip_text(TTR("Toggle Grid Snap"));
	snap_button->set_toggle_mode(true);
	toolbar->add_child(snap_button);
	snap_button->connect("toggled", callable_mp(this, &GradientTexture2DEditor::_set_snap_enabled));

	snap_size_edit = memnew(EditorSpinSlider);
	snap_size_edit->set_min(0.01);
	snap_size_edit->set_max(0.5);
	snap_size_edit->set_step(0.01);
	snap_size_edit->set_value(0.1);
	snap_size_edit->set_custom_minimum_size(Size2(65 * EDSCALE, 0));
	toolbar->add_child(snap_size_edit);
	snap_size_edit->connect("value_changed", callable_mp(this, &GradientTexture2DEditor::_set_snap_size));

	texture_editor_rect = memnew(GradientTexture2DEditorRect);
	add_child(texture_editor_rect);

	set_mouse_filter(MOUSE_FILTER_STOP);
	_set_snap_enabled(snap_button->is_pressed());
	_set_snap_size(snap_size_edit->get_value());
}

///////////////////////

bool EditorInspectorPluginGradientTexture2D::can_handle(Object *p_object) {
	return Object::cast_to<GradientTexture2D>(p_object) != nullptr;
}

void EditorInspectorPluginGradientTexture2D::parse_begin(Object *p_object) {
	GradientTexture2D *texture = Object::cast_to<GradientTexture2D>(p_object);
	if (!texture) {
		return;
	}
	Ref<GradientTexture2D> t(texture);

	GradientTexture2DEditor *editor = memnew(GradientTexture2DEditor);
	editor->set_texture(t);
	add_custom_control(editor);
}

///////////////////////

GradientTexture2DEditorPlugin::GradientTexture2DEditorPlugin() {
	Ref<EditorInspectorPluginGradientTexture2D> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
