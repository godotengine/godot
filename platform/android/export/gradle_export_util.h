/*************************************************************************/
/*  gradle_export_util.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

// Utility method used to create a directory.
Error create_directory(const String &p_dir) {
	if (!DirAccess::exists(p_dir)) {
		DirAccess *filesystem_da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		ERR_FAIL_COND_V_MSG(!filesystem_da, ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
		Error err = filesystem_da->make_dir_recursive(p_dir);
		ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
		memdelete(filesystem_da);
	}
	return OK;
}

// Implementation of EditorExportSaveSharedObject.
// This method will only be called as an input to export_project_files.
// This method lets the .so files for all ABIs to be copied
// into the gradle project from the .AAR file
Error ignore_so_file(void *p_userdata, const SharedObject &p_so) {
	return OK;
}

// Writes p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_file_at_path(const String &p_path, const Vector<uint8_t> &p_data) {
	String dir = p_path.get_base_dir();
	Error err = create_directory(dir);
	if (err != OK) {
		return err;
	}
	FileAccess *fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!fa, ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_buffer(p_data.ptr(), p_data.size());
	memdelete(fa);
	return OK;
}

// Writes string p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_string_at_path(const String &p_path, const String &p_data) {
	String dir = p_path.get_base_dir();
	Error err = create_directory(dir);
	if (err != OK) {
		return err;
	}
	FileAccess *fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(!fa, ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_string(p_data);
	memdelete(fa);
	return OK;
}

// Implementation of EditorExportSaveFunction.
// This method will only be called as an input to export_project_files.
// It is used by the export_project_files method to save all the asset files into the gradle project.
// It's functionality mirrors that of the method save_apk_file.
// This method will be called ONLY when custom build is enabled.
Error rename_and_store_file_in_gradle_project(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
	String dst_path = p_path.replace_first("res://", "res://android/build/assets/");
	Error err = store_file_at_path(dst_path, p_data);
	return err;
}

// Creates strings.xml files inside the gradle project for different locales.
Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &project_name) {
	// Stores the string into the default values directory.
	String processed_default_xml_string = vformat(godot_project_name_xml_string, project_name.xml_escape(true));
	store_string_at_path("res://android/build/res/values/godot_project_name_string.xml", processed_default_xml_string);

	// Searches the Gradle project res/ directory to find all supported locales
	DirAccessRef da = DirAccess::open("res://android/build/res");
	if (!da) {
		return ERR_CANT_OPEN;
	}
	da->list_dir_begin();
	while (true) {
		String file = da->get_next();
		if (file == "") {
			break;
		}
		if (!file.begins_with("values-")) {
			// NOTE: This assumes all directories that start with "values-" are for localization.
			continue;
		}
		String locale = file.replace("values-", "").replace("-r", "_");
		String property_name = "application/config/name_" + locale;
		String locale_directory = "res://android/build/res/" + file + "/godot_project_name_string.xml";
		if (ProjectSettings::get_singleton()->has_setting(property_name)) {
			String locale_project_name = ProjectSettings::get_singleton()->get(property_name);
			String processed_xml_string = vformat(godot_project_name_xml_string, locale_project_name.xml_escape(true));
			store_string_at_path(locale_directory, processed_xml_string);
		} else {
			// TODO: Once the legacy build system is deprecated we don't need to have xml files for this else branch
			store_string_at_path(locale_directory, processed_default_xml_string);
		}
	}
	da->list_dir_end();
	return OK;
}

String bool_to_string(bool v) {
	return v ? "true" : "false";
}

String _get_gles_tag() {
	bool min_gles3 = ProjectSettings::get_singleton()->get("rendering/quality/driver/driver_name") == "GLES3" &&
					 !ProjectSettings::get_singleton()->get("rendering/quality/driver/fallback_to_gles2");
	return min_gles3 ? "    <uses-feature android:glEsVersion=\"0x00030000\" android:required=\"true\" />\n" : "";
}

String _get_screen_sizes_tag(const Ref<EditorExportPreset> &p_preset) {
	String manifest_screen_sizes = "    <supports-screens \n        tools:node=\"replace\"";
	String sizes[] = { "small", "normal", "large", "xlarge" };
	size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
	for (size_t i = 0; i < num_sizes; i++) {
		String feature_name = vformat("screen/support_%s", sizes[i]);
		String feature_support = bool_to_string(p_preset->get(feature_name));
		String xml_entry = vformat("\n        android:%sScreens=\"%s\"", sizes[i], feature_support);
		manifest_screen_sizes += xml_entry;
	}
	manifest_screen_sizes += " />\n";
	return manifest_screen_sizes;
}

String _get_xr_features_tag(const Ref<EditorExportPreset> &p_preset) {
	String manifest_xr_features;
	bool uses_xr = (int)(p_preset->get("xr_features/xr_mode")) == 1;
	if (uses_xr) {
		int dof_index = p_preset->get("xr_features/degrees_of_freedom"); // 0: none, 1: 3dof and 6dof, 2: 6dof
		if (dof_index == 1) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"android.hardware.vr.headtracking\" android:required=\"false\" android:version=\"1\" />\n";
		} else if (dof_index == 2) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"android.hardware.vr.headtracking\" android:required=\"true\" android:version=\"1\" />\n";
		}
		int hand_tracking_index = p_preset->get("xr_features/hand_tracking"); // 0: none, 1: optional, 2: required
		if (hand_tracking_index == 1) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"oculus.software.handtracking\" android:required=\"false\" />\n";
		} else if (hand_tracking_index == 2) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"oculus.software.handtracking\" android:required=\"true\" />\n";
		}
	}
	return manifest_xr_features;
}

String _get_instrumentation_tag(const Ref<EditorExportPreset> &p_preset) {
	String package_name = p_preset->get("package/unique_name");
	String manifest_instrumentation_text = vformat(
			"    <instrumentation\n"
			"        tools:node=\"replace\"\n"
			"        android:name=\".GodotInstrumentation\"\n"
			"        android:icon=\"@mipmap/icon\"\n"
			"        android:label=\"@string/godot_project_name_string\"\n"
			"        android:targetPackage=\"%s\" />\n",
			package_name);
	return manifest_instrumentation_text;
}

String _get_plugins_tag(const String &plugins_names) {
	if (!plugins_names.empty()) {
		return vformat("    <meta-data tools:node=\"replace\" android:name=\"plugins\" android:value=\"%s\" />\n", plugins_names);
	} else {
		return "    <meta-data tools:node=\"remove\" android:name=\"plugins\" />\n";
	}
}

String _get_activity_tag(const Ref<EditorExportPreset> &p_preset) {
	bool uses_xr = (int)(p_preset->get("xr_features/xr_mode")) == 1;
	String orientation = (int)(p_preset->get("screen/orientation")) == 1 ? "portrait" : "landscape";
	String manifest_activity_text = vformat(
			"        <activity android:name=\"com.godot.game.GodotApp\" "
			"tools:replace=\"android:screenOrientation\" "
			"android:screenOrientation=\"%s\">\n",
			orientation);
	if (uses_xr) {
		String focus_awareness = bool_to_string(p_preset->get("xr_features/focus_awareness"));
		manifest_activity_text += vformat("            <meta-data tools:node=\"replace\" android:name=\"com.oculus.vr.focusaware\" android:value=\"%s\" />\n", focus_awareness);
	} else {
		manifest_activity_text += "            <meta-data tools:node=\"remove\" android:name=\"com.oculus.vr.focusaware\" />\n";
	}
	manifest_activity_text += "        </activity>\n";
	return manifest_activity_text;
}

String _get_application_tag(const Ref<EditorExportPreset> &p_preset, const String &plugins_names) {
	bool uses_xr = (int)(p_preset->get("xr_features/xr_mode")) == 1;
	String manifest_application_text =
			"    <application android:label=\"@string/godot_project_name_string\"\n"
			"        android:allowBackup=\"false\" tools:ignore=\"GoogleAppIndexingWarning\"\n"
			"        android:icon=\"@mipmap/icon\">)\n\n"
			"        <meta-data tools:node=\"remove\" android:name=\"xr_mode_metadata_name\" />\n";

	manifest_application_text += _get_plugins_tag(plugins_names);
	if (uses_xr) {
		manifest_application_text += "        <meta-data tools:node=\"replace\" android:name=\"com.samsung.android.vr.application.mode\" android:value=\"vr_only\" />\n";
	}
	manifest_application_text += _get_activity_tag(p_preset);
	manifest_application_text += "    </application>\n";
	return manifest_application_text;
}

#endif //GODOT_GRADLE_EXPORT_UTIL_H
