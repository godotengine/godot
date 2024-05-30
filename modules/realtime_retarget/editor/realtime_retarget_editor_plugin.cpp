/*************************************************************************/
/*  realtime_retarget_editor_plugin.cpp                                  */
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

#include "realtime_retarget_editor_plugin.h"

#include "editor/plugins/animation_player_editor_plugin.h"
#include "../src/retarget_utility.h"
#include "post_import_plugin_realtime_retarget.h"
#include "scene/scene_string_names.h"

void RetargetAnimationPlayerEditor::_update_editor(String p_animation_name) {
	if (!rap) {
		return;
	}
	String an = rap->get_assigned_animation();
	if (rap->has_animation(an)) {
		show();
		animation_name->set_text(an);
		Ref<Animation> assigned_animation = rap->get_animation(an);
		Dictionary meta_dict = assigned_animation->get_meta(REALTIME_RETARGET_META, Dictionary());
		Array meta_dict_keys = meta_dict.keys();
		Dictionary retarget_states;
		for (int i = 0; i < meta_dict_keys.size(); i++) {
			retarget_states[meta_dict_keys[i]] = state_names[meta_dict[meta_dict_keys[i]]];
		}
		set_meta("_retarget_states", retarget_states); // Hack!!! Use metadata as a place to store properties for display temporary values.
		meta_inspector->update_property();
	} else {
		hide();
	}
}

void RetargetAnimationPlayerEditor::_warning_pressed() {
	warning_dialog->popup_centered();
}

void RetargetAnimationPlayerEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_editor();
			AnimationPlayerEditor *ape = AnimationPlayerEditor::get_singleton();
			if (ape && !ape->is_connected("animation_selected", callable_mp(this, &RetargetAnimationPlayerEditor::_update_editor))) {
				ape->connect("animation_selected", callable_mp(this, &RetargetAnimationPlayerEditor::_update_editor));
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			warning->set_icon(get_theme_icon(SNAME("NodeWarning"), SNAME("EditorIcons")));
			warning->add_theme_color_override(SNAME("font_color"), get_theme_color(SNAME("warning_color"), SNAME("Editor")));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			AnimationPlayerEditor *ape = AnimationPlayerEditor::get_singleton();
			if (ape && ape->is_connected("animation_selected", callable_mp(this, &RetargetAnimationPlayerEditor::_update_editor))) {
				ape->disconnect("animation_selected", callable_mp(this, &RetargetAnimationPlayerEditor::_update_editor));
				ape = nullptr;
			}
		} break;
	}
}

RetargetAnimationPlayerEditor::RetargetAnimationPlayerEditor(RetargetAnimationPlayer *p_retarget_animation_player) {
	rap = p_retarget_animation_player;

	state_names.resize(3);
	state_names.write[0] = "Absolute";
	state_names.write[1] = "Local";
	state_names.write[2] = "Global";

	warning = memnew(Button);
	warning->set_text(TTR("Editing Animation for retarget directly isn't recommended!"));
	warning->set_clip_text(true);
	warning->connect("pressed", callable_mp(this, &RetargetAnimationPlayerEditor::_warning_pressed));
	add_child(warning);

	warning_dialog = memnew(AcceptDialog);
	warning_dialog->set_text(TTR("The Global/Local transform track values are applied to the model via conversion depending on the model's rest value.\nIt means that the key values and the model's pose values are very different, so it must be converted properly in the importer currently.\nBy the way, the tracks not specified in Retarget States will implicitly be assigned Absolute transform."));
	add_child(warning_dialog);

	animation_name = memnew(Label);
	add_child(animation_name);

	meta_inspector = memnew(EditorPropertyDictionary);
	meta_inspector->set_object_and_property(this, "metadata/_retarget_states");
	meta_inspector->set_label("Retarget States");
	meta_inspector->set_selectable(false);
	meta_inspector->set_read_only(true);
	add_child(meta_inspector);
}

RetargetAnimationPlayerEditor::~RetargetAnimationPlayerEditor() {
}

bool EditorInspectorPluginRetargetAnimationPlayer::can_handle(Object *p_object) {
	return Object::cast_to<RetargetAnimationPlayer>(p_object) != nullptr;
}

void EditorInspectorPluginRetargetAnimationPlayer::parse_begin(Object *p_object) {
	RetargetAnimationPlayer *rap = Object::cast_to<RetargetAnimationPlayer>(p_object);
	if (!rap) {
		return;
	}
	editor = memnew(RetargetAnimationPlayerEditor(rap));
	add_custom_control(editor);
}

bool RetargetAnimationPlayerEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("RetargetAnimationPlayer");
}

RetargetAnimationPlayerEditorPlugin::RetargetAnimationPlayerEditorPlugin() {
	rap_plugin = memnew(EditorInspectorPluginRetargetAnimationPlayer);
	EditorInspector::add_inspector_plugin(rap_plugin);
}

RealtimeRetargetEditorPlugin::RealtimeRetargetEditorPlugin() {
	Ref<PostImportPluginRealtimeRetarget> post_import_plugin_realtime_retarget;
	post_import_plugin_realtime_retarget.instantiate();
	add_scene_post_import_plugin(post_import_plugin_realtime_retarget);
}
