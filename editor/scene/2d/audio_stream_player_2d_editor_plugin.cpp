/**************************************************************************/
/*  audio_stream_player_2d_editor_plugin.cpp                              */
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

#include "audio_stream_player_2d_editor_plugin.h"

#include "core/input/input_event.h"
#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/main/scene_tree.h"

void AudioStreamPlayer2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", callable_mp(this, &AudioStreamPlayer2DEditor::_node_removed));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", callable_mp(this, &AudioStreamPlayer2DEditor::_node_removed));
		} break;
	}
}

void AudioStreamPlayer2DEditor::_node_removed(Node *p_node) {
	if (p_node == selected_player) {
		selected_player->disconnect(SceneStringName(draw), callable_mp(plugin, &EditorPlugin::update_overlays));
		selected_player = nullptr;
		dragging = false;
	}
}

Vector2 AudioStreamPlayer2DEditor::_get_handle_screen_position() const {
	const Transform2D canvas_xform = CanvasItemEditor::get_singleton()->get_canvas_transform();
	const Vector2 center = canvas_xform.xform(selected_player->get_global_position());
	const real_t radius = selected_player->get_max_distance() * canvas_xform.get_scale().x;
	// Place the handle on the right of the circle.
	return center + Vector2(radius, 0);
}

void AudioStreamPlayer2DEditor::edit(AudioStreamPlayer2D *p_player) {
	if (p_player == selected_player) {
		return;
	}
	// Follow the player's draw signal, which fires when Max Distance changes in the editor.
	const Callable update_overlays = callable_mp(plugin, &EditorPlugin::update_overlays);

	if (selected_player) {
		selected_player->disconnect(SceneStringName(draw), update_overlays);
	}
	selected_player = p_player;

	if (selected_player) {
		selected_player->connect(SceneStringName(draw), update_overlays);
	}
	plugin->update_overlays();
}

bool AudioStreamPlayer2DEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!selected_player || !selected_player->is_inside_tree()) {
		return false;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) {
			const real_t grab_threshold = EDITOR_GET("editors/polygon_editor/point_grab_radius");
			if (_get_handle_screen_position().distance_to(mb->get_position()) < grab_threshold) {
				dragging = true;
				drag_from_max_distance = selected_player->get_max_distance();
				return true;
			}
		} else if (dragging) {
			dragging = false;
			if (selected_player->get_max_distance() != drag_from_max_distance) {
				EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
				ur->create_action(TTR("Change AudioStreamPlayer2D Max Distance"));
				ur->add_do_property(selected_player, "max_distance", selected_player->get_max_distance());
				ur->add_undo_property(selected_player, "max_distance", drag_from_max_distance);
				ur->commit_action(false);
			}
			return true;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && dragging) {
		const Transform2D canvas_xform = CanvasItemEditor::get_singleton()->get_canvas_transform();
		const Vector2 mouse_pos = CanvasItemEditor::get_singleton()->snap_point(canvas_xform.affine_inverse().xform(mm->get_position()));
		// Max Distance is a world-space radius, so it maps to the node-to-cursor distance (kept above the minimum).
		const real_t new_distance = MAX((real_t)1.0, selected_player->get_global_position().distance_to(mouse_pos));
		selected_player->set_max_distance(new_distance);
		CanvasItemEditor::get_singleton()->update_viewport();
		return true;
	}

	return false;
}

void AudioStreamPlayer2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!selected_player || !selected_player->is_inside_tree()) {
		return;
	}

	const real_t max_distance = selected_player->get_max_distance();
	if (max_distance <= 0.0) {
		return;
	}

	const Transform2D canvas_xform = CanvasItemEditor::get_singleton()->get_canvas_transform();
	const Vector2 center = canvas_xform.xform(selected_player->get_global_position());
	// Max Distance is a world-space radius, so only the (uniform) editor zoom scales it.
	const real_t radius = max_distance * canvas_xform.get_scale().x;
	if (radius < 1.0) {
		return;
	}

	// Circle marking the Max Distance boundary.
	p_overlay->draw_arc(center, radius, 0, Math::TAU, 64, Color(1, 0.5, 0.2), 2.0, true);

	// Draggable handle for editing Max Distance directly in the viewport.
	const Ref<Texture2D> handle = get_editor_theme_icon(SNAME("EditorHandle"));
	p_overlay->draw_texture(handle, center + Vector2(radius, 0) - handle->get_size() / 2);
}

AudioStreamPlayer2DEditor::AudioStreamPlayer2DEditor(EditorPlugin *p_plugin) {
	plugin = p_plugin;
}

void AudioStreamPlayer2DEditorPlugin::edit(Object *p_object) {
	audio_stream_player_2d_editor->edit(Object::cast_to<AudioStreamPlayer2D>(p_object));
}

bool AudioStreamPlayer2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AudioStreamPlayer2D");
}

void AudioStreamPlayer2DEditorPlugin::make_visible(bool p_visible) {
	if (!p_visible) {
		audio_stream_player_2d_editor->edit(nullptr);
	}
}

AudioStreamPlayer2DEditorPlugin::AudioStreamPlayer2DEditorPlugin() {
	audio_stream_player_2d_editor = memnew(AudioStreamPlayer2DEditor(this));
	EditorNode::get_singleton()->get_gui_base()->add_child(audio_stream_player_2d_editor);
}
