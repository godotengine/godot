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

#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/settings/editor_settings.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "servers/audio/audio_server.h"
#include "servers/audio/audio_server_debug.h"
#include "servers/rendering/rendering_server.h"

AudioStreamPlayer2DEditor::AudioStreamPlayer2DEditor() {
	grab_threshold = EDITOR_GET("editors/polygon_editor/point_grab_radius");
	canvas_item_editor = CanvasItemEditor::get_singleton();
	AudioServer::get_singleton()->connect("_debug_audio_2d_visualization_changed", callable_mp((CanvasItem *)canvas_item_editor->get_viewport_control(), &CanvasItem::queue_redraw));
}

void AudioStreamPlayer2DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		edit(nullptr);
	}
}

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

void AudioStreamPlayer2DEditor::edit(AudioStreamPlayer2D *p_node) {
	if (node && node->is_connected(SceneStringName(draw), callable_mp((CanvasItem *)canvas_item_editor->get_viewport_control(), &CanvasItem::queue_redraw))) {
		node->disconnect(SceneStringName(draw), callable_mp((CanvasItem *)canvas_item_editor->get_viewport_control(), &CanvasItem::queue_redraw));
	}

	if (p_node) {
		node = p_node;
		node->connect(SceneStringName(draw), callable_mp((CanvasItem *)canvas_item_editor->get_viewport_control(), &CanvasItem::queue_redraw));
	} else {
		node = nullptr;
	}

	canvas_item_editor->get_viewport_control()->queue_redraw();
}

bool AudioStreamPlayer2DEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!node || !node->is_visible_in_tree()) {
		return false;
	}

	Viewport *vp = node->get_viewport();
	if (vp && !vp->is_visible_subviewport()) {
		return false;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				Vector2 handle_pos = Vector2(node->get_max_distance(), 0);

				Transform2D xform = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
				if (xform.xform(handle_pos).distance_to(mb->get_position()) < grab_threshold) {
					original_max_distance = node->get_max_distance();
					pressed = true;
					return true;
				}
			} else if (pressed) {
				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				undo_redo->create_action(TTR("Set AudioStreamPlayer2D Max Distance"));

				undo_redo->add_do_method(node, "set_max_distance", node->get_max_distance());
				undo_redo->add_undo_method(node, "set_max_distance", original_max_distance);
				undo_redo->commit_action();

				pressed = false;
				return true;
			}
		} else if (mb->get_button_index() == MouseButton::RIGHT && pressed) {
			node->set_max_distance(original_max_distance);
			canvas_item_editor->update_viewport();
			pressed = false;
			return true;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && pressed) {
		Vector2 cpoint = canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(mm->get_position()));
		cpoint = Transform2D(0, node->get_global_position()).affine_inverse().xform(cpoint);
		float new_distance = MAX(1.0, cpoint.length());
		node->set_max_distance(new_distance);
		return true;
	}

	return false;
}

void AudioStreamPlayer2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!node || !node->is_visible_in_tree()) {
		return;
	}

	Viewport *vp = node->get_viewport();
	if (vp && !vp->is_visible_subviewport()) {
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	AudioServerDebug *audio_server_debug = AudioServerDebug::get_singleton();
	float max_distance = node->get_max_distance();
	float attenuation = node->get_attenuation();

	if (audio_server_debug->get_debug_audio_2d_visualization_mode() == 0) {
		Color debug_color = audio_server_debug->get_debug_audio_2d_visualization_color();

		// center
		debug_color.a = 0.4;
		Transform2D trans_center = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
		trans_center.scale_basis(Vector2(max_distance, max_distance));
		rs->canvas_item_add_mesh(p_overlay->get_canvas_item(), audio_server_debug->get_debug_audio_2d_visualization_circle_mesh_rid(), trans_center, debug_color);

		// outline
		debug_color.a = 0.9;
		Transform2D transf_outline = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
		transf_outline.scale_basis(Vector2(max_distance, max_distance));
		rs->canvas_item_add_mesh(p_overlay->get_canvas_item(), audio_server_debug->get_debug_audio_2d_visualization_outline_mesh_rid(), transf_outline, debug_color);
	} else if (audio_server_debug->get_debug_audio_2d_visualization_mode() == 1) {
		int ring_count = audio_server_debug->get_debug_audio_2d_visualization_ring_count();
		float scale_factor = max_distance / (float(ring_count) + 0.5f);
		Color debug_color = audio_server_debug->get_debug_audio_2d_visualization_color();

		// center
		debug_color.a = Math::pow(1.0f - float(-1.0 + 1.0) / float(ring_count + 1.0), attenuation) * 0.9;
		Transform2D trans_center = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
		trans_center.scale_basis(Vector2(scale_factor / 2.0, scale_factor / 2.0));
		rs->canvas_item_add_mesh(p_overlay->get_canvas_item(), audio_server_debug->get_debug_audio_2d_visualization_circle_mesh_rid(), trans_center, debug_color);

		// rings
		Transform2D trans_rings = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
		trans_rings.scale_basis(Vector2(scale_factor, scale_factor));
		const Vector<RID> &ring_meshes = audio_server_debug->get_debug_audio_2d_visualization_rings_mesh_rids();
		for (int i = 0; i < ring_meshes.size(); i++) {
			debug_color.a = Math::pow(1.0f - float(i + 1.0) / float(ring_count + 1.0), attenuation) * 0.9;
			rs->canvas_item_add_mesh(p_overlay->get_canvas_item(), ring_meshes[i], trans_rings, debug_color);
		}

		// outline
		debug_color.a = 0.9;
		Transform2D transf_outline = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());
		transf_outline.scale_basis(Vector2(max_distance, max_distance));
		rs->canvas_item_add_mesh(p_overlay->get_canvas_item(), audio_server_debug->get_debug_audio_2d_visualization_outline_mesh_rid(), transf_outline, debug_color);
	}

	Transform2D gt = canvas_item_editor->get_canvas_transform() * Transform2D(0, node->get_global_position());

	Ref<Texture2D> handle = get_editor_theme_icon(SNAME("EditorHandle"));
	Vector2 size = handle->get_size() * 0.5;
	Vector2 handle_pos = Vector2(node->get_max_distance(), 0);

	p_overlay->draw_texture(handle, gt.xform(handle_pos) - size);
}

void AudioStreamPlayer2DEditorPlugin::edit(Object *p_object) {
	AudioStreamPlayer2D *audio = Object::cast_to<AudioStreamPlayer2D>(p_object);
	audio_editor->edit(audio);
}

bool AudioStreamPlayer2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AudioStreamPlayer2D");
}

void AudioStreamPlayer2DEditorPlugin::make_visible(bool p_visible) {
	if (!p_visible) {
		audio_editor->edit(nullptr);
	}
}

AudioStreamPlayer2DEditorPlugin::AudioStreamPlayer2DEditorPlugin() {
	audio_editor = memnew(AudioStreamPlayer2DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(audio_editor);
}
