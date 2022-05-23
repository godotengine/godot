/*************************************************************************/
/*  gradle_export_util.h                                                 */
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

#ifndef GODOT_GRADLE_EXPORT_UTIL_H
#define GODOT_GRADLE_EXPORT_UTIL_H

#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "editor/editor_export.h"

const String godot_project_name_xml_string = R"(<?xml version="1.0" encoding="utf-8"?>
<!--WARNING: THIS FILE WILL BE OVERWRITTEN AT BUILD TIME-->
<resources>
	<string name="godot_project_name_string">%s</string>
</resources>
)";

// Supported XR modes.
// This should match the entries in 'platform/android/java/lib/src/org/godotengine/godot/xr/XRMode.java'
static const int XR_MODE_REGULAR = 0;
static const int XR_MODE_OVR = 1;
static const int XR_MODE_OPENXR = 2;

// Supported XR hand tracking modes.
static const int XR_HAND_TRACKING_NONE = 0;
static const int XR_HAND_TRACKING_OPTIONAL = 1;
static const int XR_HAND_TRACKING_REQUIRED = 2;

// Supported XR hand tracking frequencies.
static const int XR_HAND_TRACKING_FREQUENCY_LOW = 0;
static const int XR_HAND_TRACKING_FREQUENCY_HIGH = 1;

// Supported XR passthrough modes.
static const int XR_PASSTHROUGH_NONE = 0;
static const int XR_PASSTHROUGH_OPTIONAL = 1;
static const int XR_PASSTHROUGH_REQUIRED = 2;

struct CustomExportData {
	String assets_directory;
	bool debug;
	Vector<String> libs;
};

int _get_android_orientation_value(OS::ScreenOrientation screen_orientation);

String _get_android_orientation_label(OS::ScreenOrientation screen_orientation);

// Utility method used to create a directory.
Error create_directory(const String &p_dir);

// Writes p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_file_at_path(const String &p_path, const Vector<uint8_t> &p_data);

// Writes string p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_string_at_path(const String &p_path, const String &p_data);

// Implementation of EditorExportSaveFunction.
// This method will only be called as an input to export_project_files.
// It is used by the export_project_files method to save all the asset files into the gradle project.
// It's functionality mirrors that of the method save_apk_file.
// This method will be called ONLY when custom build is enabled.
Error rename_and_store_file_in_gradle_project(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

// Creates strings.xml files inside the gradle project for different locales.
Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &project_name);

String bool_to_string(bool v);

String _get_gles_tag();

String _get_screen_sizes_tag(const Ref<EditorExportPreset> &p_preset);

String _get_xr_features_tag(const Ref<EditorExportPreset> &p_preset);

String _get_activity_tag(const Ref<EditorExportPreset> &p_preset);

String _get_application_tag(const Ref<EditorExportPreset> &p_preset, bool p_has_storage_permission);

#endif //GODOT_GRADLE_EXPORT_UTIL_H
