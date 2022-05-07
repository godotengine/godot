/*************************************************************************/
/*  ray_cast_2d_editor_plugin.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "ray_cast_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_node.h"

void RayCast2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", this, "_node_removed");
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", this, "_node_removed");
		} break;
	}
}

void RayCast2DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
	}
}

bool RayCast2DEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!node || !node->is_visible_in_tree()) {
		return false;
	}

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {
		if (mb->is_pressed()) {
			if (xform.xform(node->get_cast_to()).distance_to(mb->get_position()) < 8) {
				pressed = true;
				original_cast_to = node->get_cast_to();

				return true;
			} else {
				pressed = false;

				return false;
			}
		} else if (pressed) {
			undo_redo->create_action(TTR("Set cast_to"));
			undo_redo->add_do_method(node, "set_cast_to", node->get_cast_to());
			undo_redo->add_do_method(canvas_item_editor, "update_viewport");
			undo_redo->add_undo_method(node, "set_cast_to", original_cast_to);
			undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			undo_redo->commit_action();

			pressed = false;

			return true;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && pressed) {
		Vector2 point = canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(mm->get_position()));
		point = node->get_global_transform().affine_inverse().xform(point);

		node->set_cast_to(point);
		canvas_item_editor->update_viewport();
		node->_change_notify();

		return true;
	}

	return false;
}

void RayCast2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!node || !node->is_visible_in_tree()) {
		return;
	}

	Transform2D gt = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

	const Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	p_overlay->draw_texture(handle, gt.xform(node->get_cast_to()) - handle->get_size() / 2);
}

void RayCast2DEditor::edit(Node *p_node) {
	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (p_node) {
		node = Object::cast_to<RayCast2D>(p_node);
	} else {
		node = nullptr;
	}

	canvas_item_editor->update_viewport();
}

void RayCast2DEditor::_bind_methods() {
	ClassDB::bind_method("_node_removed", &RayCast2DEditor::_node_removed);
}

RayCast2DEditor::RayCast2DEditor() {
	undo_redo = EditorNode::get_singleton()->get_undo_redo();
}

///////////////////////

void RayCast2DEditorPlugin::edit(Object *p_object) {
	ray_cast_2d_editor->edit(Object::cast_to<RayCast2D>(p_object));
}

bool RayCast2DEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<RayCast2D>(p_object) != nullptr;
}

void RayCast2DEditorPlugin::make_visible(bool p_visible) {
	if (!p_visible) {
		edit(nullptr);
	}
}

RayCast2DEditorPlugin::RayCast2DEditorPlugin(EditorNode *p_editor) {
	ray_cast_2d_editor = memnew(RayCast2DEditor);
	p_editor->get_gui_base()->add_child(ray_cast_2d_editor);
}
