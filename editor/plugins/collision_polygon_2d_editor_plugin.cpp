/*************************************************************************/
/*  collision_polygon_2d_editor_plugin.cpp                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "collision_polygon_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"

void CollisionPolygon2DEditor::_enter_edit_mode() {

	mode = MODE_EDIT;
	button_edit->set_pressed(true);
	button_create->set_pressed(false);
}

bool CollisionPolygon2DEditor::_is_in_create_mode() const {

	return mode == MODE_CREATE;
}

bool CollisionPolygon2DEditor::_is_in_edit_mode() const {

	return mode == MODE_EDIT;
}

Node2D *CollisionPolygon2DEditor::_get_node() const {

	return node;
}

void CollisionPolygon2DEditor::_set_node(Node *p_node) {

	node = Object::cast_to<CollisionPolygon2D>(p_node);
}

int CollisionPolygon2DEditor::_get_polygon_count() const {

	return 1;
}

Vector<Vector2> CollisionPolygon2DEditor::_get_polygon(int p_polygon) const {

	return Variant(node->get_polygon());
}

void CollisionPolygon2DEditor::_set_polygon(int p_polygon, const Vector<Vector2> &p_points) const {

	node->set_polygon(Variant(p_points));
}

Vector2 CollisionPolygon2DEditor::_get_offset() const {

	return Vector2(0, 0);
}

void CollisionPolygon2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			button_create->set_icon(get_icon("Edit", "EditorIcons"));
			button_edit->set_icon(get_icon("MovePoint", "EditorIcons"));
			button_edit->set_pressed(true);

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}

void CollisionPolygon2DEditor::_menu_option(int p_option) {

	switch (p_option) {

		case MODE_CREATE: {

			mode = MODE_CREATE;
			button_create->set_pressed(true);
			button_edit->set_pressed(false);
		} break;
		case MODE_EDIT: {

			mode = MODE_EDIT;
			button_create->set_pressed(false);
			button_edit->set_pressed(true);
		} break;
	}
}

void CollisionPolygon2DEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_menu_option"), &CollisionPolygon2DEditor::_menu_option);
}

CollisionPolygon2DEditor::CollisionPolygon2DEditor(EditorNode *p_editor) : AbstractPolygon2DEditor(p_editor) {

	node = NULL;

	add_child(memnew(VSeparator));
	button_create = memnew(ToolButton);
	add_child(button_create);
	button_create->connect("pressed", this, "_menu_option", varray(MODE_CREATE));
	button_create->set_toggle_mode(true);
	button_create->set_tooltip(TTR("Create a new polygon from scratch."));

	button_edit = memnew(ToolButton);
	add_child(button_edit);
	button_edit->connect("pressed", this, "_menu_option", varray(MODE_EDIT));
	button_edit->set_toggle_mode(true);
	button_edit->set_tooltip(TTR("Edit existing polygon:\nLMB: Move Point.\nCtrl+LMB: Split Segment.\nRMB: Erase Point."));

	mode = MODE_EDIT;
}

void CollisionPolygon2DEditorPlugin::edit(Object *p_object) {

	collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool CollisionPolygon2DEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("CollisionPolygon2D");
}

void CollisionPolygon2DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		collision_polygon_editor->show();
	} else {

		collision_polygon_editor->hide();
		collision_polygon_editor->edit(NULL);
	}
}

CollisionPolygon2DEditorPlugin::CollisionPolygon2DEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	collision_polygon_editor = memnew(CollisionPolygon2DEditor(p_node));
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(collision_polygon_editor);

	collision_polygon_editor->hide();
}

CollisionPolygon2DEditorPlugin::~CollisionPolygon2DEditorPlugin() {
}
