/**************************************************************************/
/*  packed_scene_editor_plugin.cpp                                        */
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

#include "packed_scene_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/plugins/scene_preview.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/button.h"
#include "scene/resources/packed_scene.h"
#include "scene/scene_string_names.h"

void PackedSceneEditor::_on_open_scene_pressed() {
	// Using deferred call because changing scene updates the Inspector and thus destroys this plugin.
	callable_mp(EditorNode::get_singleton(), &EditorNode::open_request).call_deferred(packed_scene->get_path());
}

void PackedSceneEditor::_on_open_preview_pressed() {
	if (preview == nullptr) {
		Node *root = packed_scene->instantiate();
		if (Object::cast_to<Node3D>(root) != nullptr) {
			Scene3DPreview *preview_3d = memnew(Scene3DPreview());
			preview_3d->edit(Object::cast_to<Node3D>(root));
			preview = preview_3d;
		} else if (Object::cast_to<Node2D>(root) != nullptr) {
			Scene2DPreview *preview_2d = memnew(Scene2DPreview());
			preview_2d->edit(Object::cast_to<Node2D>(root));
			preview = preview_2d;
		} else if (Object::cast_to<Control>(root) != nullptr) {
			SceneControlPreview *preview_control = memnew(SceneControlPreview());
			preview_control->edit(Object::cast_to<Control>(root));
			preview = preview_control;
		} else {
			Label *hint = memnew(Label);
			hint->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
			hint->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
			hint->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
			hint->set_anchors_and_offsets_preset(LayoutPreset::PRESET_FULL_RECT, LayoutPresetMode::PRESET_MODE_MINSIZE, 10);
			hint->set_text(vformat(TTR("No preview avaible for node type \"%s\"."), root->get_class()));
			root->queue_free();
			SceneControlPreview *preview_control = memnew(SceneControlPreview());
			preview_control->edit(hint);
			preview = preview_control;
		}
		add_child(preview);
		move_child(preview, 0);
	} else {
		preview->queue_free();
		preview = nullptr;
	}
}

PackedSceneEditor::PackedSceneEditor(Ref<PackedScene> &p_packed_scene) {
	packed_scene = p_packed_scene;

	HBoxContainer *buttons = memnew(HBoxContainer);
	add_child(buttons);
	buttons->set_alignment(ALIGNMENT_CENTER);

	if (packed_scene->get_path().get_file().is_valid_filename()) {
		Button *open_editor_button = EditorInspector::create_inspector_action_button(TTR("Open"));
		open_editor_button->connect(SNAME("pressed"), callable_mp(this, &PackedSceneEditor::_on_open_scene_pressed));
		buttons->add_child(open_editor_button);
	}

	Button *open_scene_button = EditorInspector::create_inspector_action_button(TTR("Preview"));
	open_scene_button->connect(SNAME("pressed"), callable_mp(this, &PackedSceneEditor::_on_open_preview_pressed));
	if (packed_scene->get_state()->get_node_count() <= 25) {
		_on_open_preview_pressed();
	}
	buttons->add_child(open_scene_button);

	add_child(memnew(Control)); // Add padding before the regular properties.
}

///////////////////////

bool EditorInspectorPluginPackedScene::can_handle(Object *p_object) {
	return Object::cast_to<PackedScene>(p_object) != nullptr;
}

void EditorInspectorPluginPackedScene::parse_begin(Object *p_object) {
	Ref<PackedScene> packed_scene(p_object);
	PackedSceneEditor *editor = memnew(PackedSceneEditor(packed_scene));
	add_custom_control(editor);
}

///////////////////////

PackedSceneEditorPlugin::PackedSceneEditorPlugin() {
	Ref<EditorInspectorPluginPackedScene> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
