/*************************************************************************/
/*  camera_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "camera_editor_plugin.h"

#include "spatial_editor_plugin.h"

void CameraEditor::_notification(int p_what) {

	switch (p_what) {

		/*		case NOTIFICATION_PROCESS: {

			if (preview->is_pressed() && node)
				node->call("make_current");

		} break;*/
	}
}
void CameraEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		SpatialEditor::get_singleton()->set_custom_camera(NULL);
		hide();
	}
}

void CameraEditor::_pressed() {

	Node *sn = (node && preview->is_pressed()) ? node : NULL;
	SpatialEditor::get_singleton()->set_custom_camera(sn);
}

void CameraEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_pressed"), &CameraEditor::_pressed);
}

void CameraEditor::edit(Node *p_camera) {

	node = p_camera;

	if (!node) {
		preview->set_pressed(false);
		SpatialEditor::get_singleton()->set_custom_camera(NULL);
	} else {

		if (preview->is_pressed())
			SpatialEditor::get_singleton()->set_custom_camera(p_camera);
		else
			SpatialEditor::get_singleton()->set_custom_camera(NULL);
	}
}

CameraEditor::CameraEditor() {

	preview = memnew(Button);
	add_child(preview);

	preview->set_text(TTR("Preview"));
	preview->set_toggle_mode(true);
	preview->set_anchor(MARGIN_LEFT, Control::ANCHOR_END);
	preview->set_anchor(MARGIN_RIGHT, Control::ANCHOR_END);
	preview->set_margin(MARGIN_LEFT, 60);
	preview->set_margin(MARGIN_RIGHT, 0);
	preview->set_margin(MARGIN_TOP, 0);
	preview->set_margin(MARGIN_BOTTOM, 10);
	preview->connect("pressed", this, "_pressed");
}

void CameraEditorPlugin::edit(Object *p_object) {

	SpatialEditor::get_singleton()->set_can_preview(p_object->cast_to<Camera>());
	//camera_editor->edit(p_object->cast_to<Node>());
}

bool CameraEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Camera");
}

void CameraEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		//SpatialEditor::get_singleton()->set_can_preview(p_object->cast_to<Camera>());
	} else {
		SpatialEditor::get_singleton()->set_can_preview(NULL);
	}
}

CameraEditorPlugin::CameraEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	/*	camera_editor = memnew( CameraEditor );
	editor->get_viewport()->add_child(camera_editor);

	camera_editor->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
	camera_editor->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);
	camera_editor->set_margin(MARGIN_LEFT,60);
	camera_editor->set_margin(MARGIN_RIGHT,0);
	camera_editor->set_margin(MARGIN_TOP,0);
	camera_editor->set_margin(MARGIN_BOTTOM,10);


	camera_editor->hide();
*/
}

CameraEditorPlugin::~CameraEditorPlugin() {
}
