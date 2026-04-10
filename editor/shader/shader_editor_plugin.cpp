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

#include "editor/docks/editor_dock.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/shader.h"

void ShaderEditorPlugin::shortcut_input(const Ref<InputEvent> &p_event) {
	if (p_event.is_null() || !p_event->is_pressed() || p_event->is_echo()) {
		return;
	}

	if (make_floating_shortcut.is_valid() && make_floating_shortcut->matches_event(p_event)) {
		EditorDockManager::get_singleton()->make_dock_floating(shader_dock);
	}
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		shader_dock->make_visible();
	}
}

void ShaderEditorPlugin::edit(Object *p_object) {
	if (!p_object) {
		return;
	}

	if (Object::cast_to<Shader>(p_object) || Object::cast_to<ShaderInclude>(p_object)) {
		script_editor->edit(Object::cast_to<Resource>(p_object));
	}
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Shader>(p_object) || Object::cast_to<ShaderInclude>(p_object);
}

void ShaderEditorPlugin::set_window_layout(Ref<ConfigFile> p_layout) {
	if (bool(EDITOR_GET("editors/shader_editor/behavior/files/restore_shaders_on_load"))) {
		script_editor->set_window_layout(p_layout);
	}
}

void ShaderEditorPlugin::get_window_layout(Ref<ConfigFile> p_layout) {
	script_editor->get_window_layout(p_layout);
}

ShaderEditorPlugin::ShaderEditorPlugin() {
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

	script_editor = memnew(ScriptEditor(nullptr, shader_dock));
	script_editor->set_config_section("ShaderEditor");
	script_editor->set_handled_file_types({ "Shader", "ShaderInclude" });
	shader_dock->add_child(script_editor);
}
