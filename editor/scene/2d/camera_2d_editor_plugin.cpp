/**************************************************************************/
/*  camera_2d_editor_plugin.cpp                                           */
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

#include "camera_2d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/camera_2d.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"

void Camera2DEditor::edit(Camera2D *p_camera) {
	if (p_camera == selected_camera) {
		return;
	}
	const Callable update_overlays = callable_mp(plugin, &EditorPlugin::update_overlays);

	if (selected_camera) {
		selected_camera->disconnect(SceneStringName(draw), update_overlays);
		if (drag_type != Drag::NONE) {
			selected_camera->set_limit_rect(drag_revert);
		}
		drag_type = Drag::NONE;
		hover_type = Drag::NONE;
		CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_ARROW);
	}
	selected_camera = p_camera;

	if (selected_camera) {
		selected_camera->connect(SceneStringName(draw), update_overlays);
	}
	plugin->update_overlays();
}

bool Camera2DEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!selected_camera || !selected_camera->is_limit_enabled()) {
		return false;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				if (hover_type != Drag::NONE) {
					Vector2 pos = CanvasItemEditor::get_singleton()->get_canvas_transform().affine_inverse().xform(mb->get_position());
					const Rect2 limit_rect = selected_camera->get_limit_rect();

					drag_type = hover_type;
					drag_revert = selected_camera->get_limit_rect();
					center_drag_point = pos - limit_rect.position;
					return true;
				}
			} else if (drag_type != Drag::NONE) {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Edit Camera2D Limits"));
				ur->add_do_method(selected_camera, "_set_limit_rect", selected_camera->get_limit_rect());
				ur->add_do_method(this, "_update_overlays_if_needed", selected_camera);
				ur->add_undo_method(selected_camera, "_set_limit_rect", drag_revert);
				ur->add_undo_method(this, "_update_overlays_if_needed", selected_camera);
				ur->commit_action(false);

				drag_type = Drag::NONE;
				return true;
			}
		} else if (drag_type != Drag::NONE && mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			selected_camera->set_limit_rect(drag_revert);
			drag_type = Drag::NONE;
			plugin->update_overlays();
			_update_hover(mb->get_position());
			return true;
		}
		return false;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Vector2 pos = mm->get_position();
		if (drag_type == Drag::NONE) {
			_update_hover(pos);
			return false;
		}

		pos = CanvasItemEditor::get_singleton()->get_canvas_transform().affine_inverse().xform(pos);
		pos = CanvasItemEditor::get_singleton()->snap_point(pos);

		switch (drag_type) {
			case Drag::LEFT: {
				selected_camera->set_limit(SIDE_LEFT, MIN(selected_camera->get_limit(SIDE_RIGHT), pos.x));
				plugin->update_overlays();
			} break;

			case Drag::RIGHT: {
				selected_camera->set_limit(SIDE_RIGHT, MAX(selected_camera->get_limit(SIDE_LEFT), pos.x));
				plugin->update_overlays();
			} break;

			case Drag::TOP: {
				selected_camera->set_limit(SIDE_TOP, MIN(selected_camera->get_limit(SIDE_BOTTOM), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::BOTTOM: {
				selected_camera->set_limit(SIDE_BOTTOM, MAX(selected_camera->get_limit(SIDE_TOP), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::TOP_LEFT: {
				selected_camera->set_limit(SIDE_LEFT, MIN(selected_camera->get_limit(SIDE_RIGHT), pos.x));
				selected_camera->set_limit(SIDE_TOP, MIN(selected_camera->get_limit(SIDE_BOTTOM), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::TOP_RIGHT: {
				selected_camera->set_limit(SIDE_RIGHT, MAX(selected_camera->get_limit(SIDE_LEFT), pos.x));
				selected_camera->set_limit(SIDE_TOP, MIN(selected_camera->get_limit(SIDE_BOTTOM), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::BOTTOM_LEFT: {
				selected_camera->set_limit(SIDE_LEFT, MIN(selected_camera->get_limit(SIDE_RIGHT), pos.x));
				selected_camera->set_limit(SIDE_BOTTOM, MAX(selected_camera->get_limit(SIDE_TOP), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::BOTTOM_RIGHT: {
				selected_camera->set_limit(SIDE_RIGHT, MAX(selected_camera->get_limit(SIDE_LEFT), pos.x));
				selected_camera->set_limit(SIDE_BOTTOM, MAX(selected_camera->get_limit(SIDE_TOP), pos.y));
				plugin->update_overlays();
			} break;

			case Drag::CENTER: {
				Rect2 target_rect = selected_camera->get_limit_rect();
				target_rect.position = pos - center_drag_point;
				selected_camera->set_limit_rect(target_rect);
				plugin->update_overlays();
			} break;

			case Drag::NONE: {
			} break;
		}
		return true;
	}

	return false;
}

void Camera2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!selected_camera || !selected_camera->is_limit_enabled()) {
		return;
	}
	Rect2 limit_rect = selected_camera->get_limit_rect();
	limit_rect = CanvasItemEditor::get_singleton()->get_canvas_transform().xform(limit_rect);
	p_overlay->draw_rect(limit_rect, Color(1, 1, 0.25, 0.63), false, 3);
}

void Camera2DEditor::_menu_option(int p_option) {
	switch (p_option) {
		case MENU_SNAP_LIMITS_TO_VIEWPORT: {
			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(TTR("Snap Camera2D Limits to the Viewport"), UndoRedo::MERGE_DISABLE, selected_camera);
			ur->add_do_method(this, "_snap_limits_to_viewport", selected_camera);
			ur->add_undo_method(selected_camera, "_set_limit_rect", selected_camera->get_limit_rect());
			ur->add_undo_method(this, "_update_overlays_if_needed", selected_camera);
			ur->commit_action();
		} break;
	}
}

void Camera2DEditor::_snap_limits_to_viewport(Camera2D *p_camera) {
	p_camera->set_limit(SIDE_LEFT, 0);
	p_camera->set_limit(SIDE_TOP, 0);
	p_camera->set_limit(SIDE_RIGHT, GLOBAL_GET("display/window/size/viewport_width"));
	p_camera->set_limit(SIDE_BOTTOM, GLOBAL_GET("display/window/size/viewport_height"));
	_update_overlays_if_needed(p_camera);
}

void Camera2DEditor::_update_overlays_if_needed(Camera2D *p_camera) {
	if (p_camera == selected_camera) {
		plugin->update_overlays();
	}
}

void Camera2DEditor::_update_hover(const Vector2 &p_mouse_pos) {
	if (CanvasItemEditor::get_singleton()->get_current_tool() != CanvasItemEditor::TOOL_SELECT) {
		hover_type = Drag::NONE;
		CanvasItemEditor::get_singleton()->set_cursor_shape_override();
		return;
	}

	const Rect2 limit_rect = CanvasItemEditor::get_singleton()->get_canvas_transform().xform(selected_camera->get_limit_rect());
	const float drag_tolerance = 8.0;
	const Vector2 tolerance_vector = Vector2(1, 1) * drag_tolerance;

	hover_type = Drag::NONE;
	if (Rect2(limit_rect.position - tolerance_vector, tolerance_vector * 2).has_point(p_mouse_pos)) {
		hover_type = Drag::TOP_LEFT;
	} else if (Rect2(Vector2(limit_rect.get_end().x, limit_rect.position.y) - tolerance_vector, tolerance_vector * 2).has_point(p_mouse_pos)) {
		hover_type = Drag::TOP_RIGHT;
	} else if (Rect2(Vector2(limit_rect.position.x, limit_rect.get_end().y) - tolerance_vector, tolerance_vector * 2).has_point(p_mouse_pos)) {
		hover_type = Drag::BOTTOM_LEFT;
	} else if (Rect2(limit_rect.get_end() - tolerance_vector, tolerance_vector * 2).has_point(p_mouse_pos)) {
		hover_type = Drag::BOTTOM_RIGHT;
	} else if (p_mouse_pos.y > limit_rect.position.y && p_mouse_pos.y < limit_rect.get_end().y) {
		if (Math::abs(p_mouse_pos.x - limit_rect.position.x) < drag_tolerance) {
			hover_type = Drag::LEFT;
		} else if (Math::abs(p_mouse_pos.x - limit_rect.get_end().x) < drag_tolerance) {
			hover_type = Drag::RIGHT;
		}
	} else if (p_mouse_pos.x > limit_rect.position.x && p_mouse_pos.x < limit_rect.get_end().x) {
		if (Math::abs(p_mouse_pos.y - limit_rect.position.y) < drag_tolerance) {
			hover_type = Drag::TOP;
		} else if (Math::abs(p_mouse_pos.y - limit_rect.get_end().y) < drag_tolerance) {
			hover_type = Drag::BOTTOM;
		}
	}

	if (hover_type == Drag::NONE && limit_rect.has_point(p_mouse_pos)) {
		const Rect2 editor_rect = Rect2(Vector2(), CanvasItemEditor::get_singleton()->get_viewport_control()->get_size());
		const Rect2 transformed_rect = selected_camera->get_viewport()->get_canvas_transform().xform_inv(limit_rect);

		// Only allow center drag if any limit edge is visible on screen.
		bool edge_visible = false;
		edge_visible = edge_visible || (transformed_rect.get_end().y > editor_rect.position.y && transformed_rect.position.y < editor_rect.get_end().y && transformed_rect.position.x > editor_rect.position.x && transformed_rect.position.x < editor_rect.get_end().x);
		edge_visible = edge_visible || (transformed_rect.get_end().y > editor_rect.position.y && transformed_rect.position.y < editor_rect.get_end().y && transformed_rect.get_end().x > editor_rect.position.x && transformed_rect.get_end().x < editor_rect.get_end().x);
		edge_visible = edge_visible || (transformed_rect.get_end().x > editor_rect.position.x && transformed_rect.position.x < editor_rect.get_end().x && transformed_rect.position.y > editor_rect.position.y && transformed_rect.position.y < editor_rect.get_end().y);
		edge_visible = edge_visible || (transformed_rect.get_end().x > editor_rect.position.x && transformed_rect.position.x < editor_rect.get_end().x && transformed_rect.get_end().y > editor_rect.position.y && transformed_rect.get_end().y < editor_rect.get_end().y);

		if (edge_visible) {
			hover_type = Drag::CENTER;
		}
	}

	switch (hover_type) {
		case Drag::NONE: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override();
		} break;
		case Drag::LEFT:
		case Drag::RIGHT: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_HSIZE);
		} break;
		case Drag::TOP:
		case Drag::BOTTOM: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_VSIZE);
		} break;
		case Drag::TOP_LEFT:
		case Drag::BOTTOM_RIGHT: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_FDIAGSIZE);
		} break;
		case Drag::TOP_RIGHT:
		case Drag::BOTTOM_LEFT: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_BDIAGSIZE);
		} break;
		case Drag::CENTER: {
			CanvasItemEditor::get_singleton()->set_cursor_shape_override(CURSOR_MOVE);
		} break;
	}
}

void Camera2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			options->set_button_icon(get_editor_theme_icon(SNAME("Camera2D")));
		} break;
	}
}

void Camera2DEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_snap_limits_to_viewport", "camera"), &Camera2DEditor::_snap_limits_to_viewport);
	ClassDB::bind_method(D_METHOD("_update_overlays_if_needed", "camera"), &Camera2DEditor::_update_overlays_if_needed);
}

Camera2DEditor::Camera2DEditor(EditorPlugin *p_plugin) {
	plugin = p_plugin;

	options = memnew(MenuButton);
	options->set_text(TTRC("Camera2D"));
	options->get_popup()->add_item(TTRC("Snap the Limits to the Viewport"), MENU_SNAP_LIMITS_TO_VIEWPORT);
	options->set_switch_on_hover(true);
	options->hide();
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(options);
	options->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &Camera2DEditor::_menu_option));
}

void Camera2DEditorPlugin::edit(Object *p_object) {
	camera_2d_editor->edit(Object::cast_to<Camera2D>(p_object));
}

bool Camera2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Camera2D");
}

void Camera2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		camera_2d_editor->options->show();
	} else {
		camera_2d_editor->options->hide();
	}
}

Camera2DEditorPlugin::Camera2DEditorPlugin() {
	camera_2d_editor = memnew(Camera2DEditor(this));
	EditorNode::get_singleton()->get_gui_base()->add_child(camera_2d_editor);
}
