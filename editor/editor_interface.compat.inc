/**************************************************************************/
/*  editor_interface.compat.inc                                           */
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

#ifndef DISABLE_DEPRECATED

#include "editor_interface.h"

#include "editor/editor_data.h"
#include "editor/editor_paths.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "scene/main/node.h"

EditorFileSystem *EditorInterface::get_resource_file_system_compat_76176() {
	return get_resource_file_system();
}

EditorPaths *EditorInterface::get_editor_paths_compat_76176() {
	return get_editor_paths();
}

EditorResourcePreview *EditorInterface::get_resource_previewer_compat_76176() {
	return get_resource_previewer();
}

EditorSelection *EditorInterface::get_selection_compat_76176() {
	return get_selection();
}

Ref<EditorSettings> EditorInterface::get_editor_settings_compat_76176() {
	return get_editor_settings();
}

Control *EditorInterface::get_base_control_compat_76176() {
	return get_base_control();
}

VBoxContainer *EditorInterface::get_editor_main_screen_compat_76176() {
	return get_editor_main_screen();
}

ScriptEditor *EditorInterface::get_script_editor_compat_76176() {
	return get_script_editor();
}

FileSystemDock *EditorInterface::get_file_system_dock_compat_76176() {
	return get_file_system_dock();
}

Node *EditorInterface::get_edited_scene_root_compat_76176() {
	return get_edited_scene_root();
}

void EditorInterface::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_resource_filesystem"), &EditorInterface::get_resource_file_system_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_editor_paths"), &EditorInterface::get_editor_paths_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_resource_previewer"), &EditorInterface::get_resource_previewer_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_selection"), &EditorInterface::get_selection_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_editor_settings"), &EditorInterface::get_editor_settings_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_base_control"), &EditorInterface::get_base_control_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_editor_main_screen"), &EditorInterface::get_editor_main_screen_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_script_editor"), &EditorInterface::get_script_editor_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_file_system_dock"), &EditorInterface::get_file_system_dock_compat_76176);
	ClassDB::bind_compatibility_method(D_METHOD("get_edited_scene_root"), &EditorInterface::get_edited_scene_root_compat_76176);
}

#endif
