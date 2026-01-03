/**************************************************************************/
/*  camera_3d_editor_plugin.cpp                                           */
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

#include "camera_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "node_3d_editor_plugin.h"
#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/foldable_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/viewport.h"

void Camera3DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		Node3DEditor::get_singleton()->set_custom_camera(nullptr);
		hide();
	}
}

void Camera3DEditor::_pressed() {
	Node *sn = (node && preview->is_pressed()) ? node : nullptr;
	Node3DEditor::get_singleton()->set_custom_camera(sn);
}

void Camera3DEditor::edit(Node *p_camera) {
	node = p_camera;

	if (!node) {
		preview->set_pressed(false);
		Node3DEditor::get_singleton()->set_custom_camera(nullptr);
	} else {
		if (preview->is_pressed()) {
			Node3DEditor::get_singleton()->set_custom_camera(p_camera);
		} else {
			Node3DEditor::get_singleton()->set_custom_camera(nullptr);
		}
	}
}

Camera3DEditor::Camera3DEditor() {
	preview = memnew(Button);
	add_child(preview);

	preview->set_text(TTR("Preview"));
	preview->set_toggle_mode(true);
	preview->set_anchor(SIDE_LEFT, Control::ANCHOR_END);
	preview->set_anchor(SIDE_RIGHT, Control::ANCHOR_END);
	preview->set_offset(SIDE_LEFT, -60);
	preview->set_offset(SIDE_RIGHT, 0);
	preview->set_offset(SIDE_TOP, 0);
	preview->set_offset(SIDE_BOTTOM, 10);
	preview->connect(SceneStringName(pressed), callable_mp(this, &Camera3DEditor::_pressed));
}

bool Camera3DPreview::camera_preview_folded = false;

void Camera3DPreview::_update_sub_viewport_size() {
	if (sub_viewport != nullptr) {
		const Size2i camera_size = Node3DEditor::get_camera_viewport_size(camera);
		centering_container->set_ratio(camera_size.aspect());
	}
}

void Camera3DPreview::_camera_exiting() {
	if (sub_viewport != nullptr) {
		const Size2i camera_size = Node3DEditor::get_camera_viewport_size(camera);
		centering_container->set_ratio(camera_size.aspect());
	}
}

void Camera3DPreview::_toggle_folding(bool p_folded) {
	camera_preview_folded = p_folded;
}

Camera3DPreview::Camera3DPreview(Camera3D *p_camera) {
	camera = p_camera;
	camera->connect(SceneStringName(tree_exiting), callable_mp(this, &Camera3DPreview::_camera_exiting));

	FoldableContainer *folder = memnew(FoldableContainer);
	folder->set_title(TTRC("Camera Preview"));
	folder->set_title_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	folder->set_folded(camera_preview_folded);
	folder->connect("folding_changed", callable_mp(this, &Camera3DPreview::_toggle_folding));
	add_child(folder);

	centering_container = memnew(AspectRatioContainer);
	centering_container->set_custom_minimum_size(Size2(0.0, 256.0) * EDSCALE);
	folder->add_child(centering_container);

	SubViewportContainer *sub_viewport_container = memnew(SubViewportContainer);
	sub_viewport_container->set_stretch(true);
	sub_viewport_container->set_texture_filter(TEXTURE_FILTER_NEAREST_WITH_MIPMAPS);
	centering_container->add_child(sub_viewport_container);

	sub_viewport = memnew(SubViewport);
	sub_viewport_container->add_child(sub_viewport);

	RenderingServer::get_singleton()->viewport_attach_camera(sub_viewport->get_viewport_rid(), camera->get_camera());

	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &Camera3DPreview::_update_sub_viewport_size));
	_update_sub_viewport_size();
}

bool EditorInspectorPluginCamera3DPreview::can_handle(Object *p_object) {
	return Object::cast_to<Camera3D>(p_object) != nullptr;
}

void EditorInspectorPluginCamera3DPreview::parse_begin(Object *p_object) {
	Camera3D *camera = Object::cast_to<Camera3D>(p_object);
	Camera3DPreview *preview = memnew(Camera3DPreview(camera));
	add_custom_control(preview);
}

void Camera3DEditorPlugin::edit(Object *p_object) {
	Node3DEditor::get_singleton()->set_can_preview(Object::cast_to<Camera3D>(p_object));
}

bool Camera3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Camera3D");
}

void Camera3DEditorPlugin::make_visible(bool p_visible) {
	if (!p_visible) {
		Node3DEditor::get_singleton()->set_can_preview(nullptr);
	}
}

Camera3DEditorPlugin::Camera3DEditorPlugin() {
	Ref<EditorInspectorPluginCamera3DPreview> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
