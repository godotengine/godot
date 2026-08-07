/**************************************************************************/
/*  shader_editor_plugin.cpp                                              */
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

#include "shader_editor_plugin.h"

#include "core/variant/variant.h"
#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/script/script_editor_base.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/shader/editor_shader_language_plugin.h"
#include "editor/shader/text_shader_language_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/shader.h"

void ShaderEditorPlugin::shortcut_input(const Ref<InputEvent> &p_event) {
	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	if (make_floating_shortcut.is_valid() && make_floating_shortcut->matches_event(p_event)) {
		shader_dock->make_floating();
	}
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		shader_dock->open();
	}
}

void ShaderEditorPlugin::set_current() {
	shader_dock->make_visible();
	TextEditorBase *text_shader_editor = Object::cast_to<TextEditorBase>(script_editor->get_current_editor());
	if (text_shader_editor) {
		text_shader_editor->ensure_focus();
	}
}

void ShaderEditorPlugin::edit(Object *p_object) {
	if (!p_object) {
		return;
	}

	if (Object::cast_to<Shader>(p_object) || Object::cast_to<ShaderInclude>(p_object)) {
		script_editor->edit(Object::cast_to<Resource>(p_object), false);
	}
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Shader>(p_object) || Object::cast_to<ShaderInclude>(p_object);
}

void ShaderEditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {
#ifndef DISABLE_DEPRECATED
	for (int i = 0; i < old_layout_keys.size(); i++) {
		if (new_layout_keys[i] != "" && p_layout->has_section_key(config_section, old_layout_keys[i])) {
			p_layout->set_value(config_section, new_layout_keys[i], p_layout->get_value(config_section, old_layout_keys[i]));
		}
	}
#endif

	ScriptEditorBase *current_editor = script_editor->get_current_editor();
	if (bool(EDITOR_GET("editors/shader_editor/behavior/files/restore_shaders_on_load"))) {
		script_editor->set_window_layout(p_layout);
	}
	if (current_editor) {
		script_editor->edit(current_editor->get_edited_resource());
	}
}

void ShaderEditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {
#ifndef DISABLE_DEPRECATED
	for (int i = 0; i < old_layout_keys.size(); i++) {
		if (p_layout->has_section_key(config_section, old_layout_keys[i])) {
			p_layout->erase_section_key(config_section, old_layout_keys[i]);
		}
	}
#endif

	script_editor->get_window_layout(p_layout);
}

String ShaderEditorPlugin::get_unsaved_status(const String &p_for_scene) const {
	const PackedStringArray unsaved_scripts = script_editor->get_unsaved_scripts();
	if (unsaved_scripts.is_empty()) {
		return String();
	}

	PackedStringArray message;
	if (!p_for_scene.is_empty()) {
		PackedStringArray unsaved_built_in_scripts;

		const String scene_file = p_for_scene.get_file();
		for (const String &E : unsaved_scripts) {
			if (!E.is_resource_file() && E.contains(scene_file)) {
				unsaved_built_in_scripts.append(E);
			}
		}

		if (unsaved_built_in_scripts.is_empty()) {
			return String();
		} else {
			message.push_back(TTR("There are unsaved changes in the following built-in resource(s)"));
			message.append_array(unsaved_built_in_scripts);
			return String("\n").join(message);
		}
	}

	message.push_back(TTR("Save changes to the following file(s) before quitting?"));
	message.append_array(unsaved_scripts);
	return String("\n").join(message);
}

void ShaderEditorPlugin::save_external_data() {
	if (!EditorNode::get_singleton()->is_exiting()) {
		script_editor->save_all_scripts();
	}
}

ShaderEditorPlugin::ShaderEditorPlugin() {
	shader_editor_plugin = this;

	Ref<TextShaderLanguagePlugin> text_shader_lang;
	text_shader_lang.instantiate();
	EditorShaderLanguagePlugin::register_shader_language(text_shader_lang);

	shader_dock = memnew(EditorDock);
	shader_dock->set_name(TTRC("Shader Editor"));
	shader_dock->set_icon_name("ShaderDock");
	shader_dock->set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_shader_editor_bottom_panel", TTRC("Toggle Shader Editor Dock"), KeyModifierMask::ALT | Key::S));
	shader_dock->set_default_slot(EditorDock::DOCK_SLOT_BOTTOM);
	shader_dock->set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);
	shader_dock->set_custom_minimum_size(Size2(460, 300) * EDSCALE);
	EditorDockManager::get_singleton()->add_dock(shader_dock);

	set_process_shortcut_input(true);

	make_floating_shortcut = ED_SHORTCUT_AND_COMMAND("shader_editor/make_floating", TTRC("Make Floating"));

	script_editor = memnew(ScriptEditor(config_section, "shader_editor_cache.cfg", nullptr, shader_dock));
	script_editor->set_handled_resource_types({ "Shader", "VisualShader", "ShaderInclude" });
	shader_dock->add_child(script_editor);
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
	EditorShaderLanguagePlugin::clear_registered_shader_languages();
}
