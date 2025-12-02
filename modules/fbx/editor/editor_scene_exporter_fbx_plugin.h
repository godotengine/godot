/**************************************************************************/
/*  editor_scene_exporter_fbx_plugin.h                                    */
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

#pragma once

#include "../fbx_document.h"
#include "editor_scene_exporter_fbx_settings.h"

#include "editor/plugins/editor_plugin.h"

class ConfirmationDialog;
class EditorFileDialog;
class EditorInspector;

class SceneExporterFBXPlugin : public EditorPlugin {
	GDCLASS(SceneExporterFBXPlugin, EditorPlugin);

	Ref<FBXDocument> _fbx_document;
	Ref<EditorSceneExporterFBXSettings> _export_settings;
	String export_path;

	EditorInspector *_settings_inspector = nullptr;
	ConfirmationDialog *_config_dialog = nullptr;
	EditorFileDialog *_file_dialog = nullptr;

	void _popup_fbx_settings_dialog(const String &p_selected_path);
	void _popup_fbx_export_dialog();
	void _export_scene_as_fbx();

protected:
	void _notification(int p_what);

public:
	virtual String get_plugin_name() const override;
	bool has_main_screen() const override;
	SceneExporterFBXPlugin();
};
