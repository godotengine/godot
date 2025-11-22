/**************************************************************************/
/*  editor_scene_exporter_fbx_settings.cpp                                */
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

#include "editor_scene_exporter_fbx_settings.h"

String EditorSceneExporterFBXSettings::get_copyright() const {
	return _copyright;
}

void EditorSceneExporterFBXSettings::set_copyright(const String &p_copyright) {
	_copyright = p_copyright;
}

double EditorSceneExporterFBXSettings::get_bake_fps() const {
	return _bake_fps;
}

void EditorSceneExporterFBXSettings::set_bake_fps(const double p_bake_fps) {
	_bake_fps = p_bake_fps;
}

int EditorSceneExporterFBXSettings::get_naming_version() const {
	return _naming_version;
}

void EditorSceneExporterFBXSettings::set_naming_version(const int p_naming_version) {
	_naming_version = p_naming_version;
}

int EditorSceneExporterFBXSettings::get_export_format() const {
	return _export_format;
}

void EditorSceneExporterFBXSettings::set_export_format(const int p_export_format) {
	_export_format = p_export_format;
}

void EditorSceneExporterFBXSettings::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_copyright"), &EditorSceneExporterFBXSettings::get_copyright);
	ClassDB::bind_method(D_METHOD("set_copyright", "copyright"), &EditorSceneExporterFBXSettings::set_copyright);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "copyright", PROPERTY_HINT_PLACEHOLDER_TEXT, "Example: 2014 Godette"), "set_copyright", "get_copyright");

	ClassDB::bind_method(D_METHOD("get_bake_fps"), &EditorSceneExporterFBXSettings::get_bake_fps);
	ClassDB::bind_method(D_METHOD("set_bake_fps", "bake_fps"), &EditorSceneExporterFBXSettings::set_bake_fps);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bake_fps", PROPERTY_HINT_RANGE, "0.001,120,0.0001,or_greater"), "set_bake_fps", "get_bake_fps");

	ClassDB::bind_method(D_METHOD("get_naming_version"), &EditorSceneExporterFBXSettings::get_naming_version);
	ClassDB::bind_method(D_METHOD("set_naming_version", "naming_version"), &EditorSceneExporterFBXSettings::set_naming_version);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "naming_version", PROPERTY_HINT_ENUM, "Version 0,Version 2"), "set_naming_version", "get_naming_version");

	ClassDB::bind_method(D_METHOD("get_export_format"), &EditorSceneExporterFBXSettings::get_export_format);
	ClassDB::bind_method(D_METHOD("set_export_format", "export_format"), &EditorSceneExporterFBXSettings::set_export_format);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "export_format", PROPERTY_HINT_ENUM, "Binary,ASCII"), "set_export_format", "get_export_format");
}
