/*************************************************************************/
/*  gradle_export_util.cpp                                               */
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

#include "gradle_export_util.h"

int _get_android_orientation_value(OS::ScreenOrientation screen_orientation) {
	switch (screen_orientation) {
		case OS::SCREEN_PORTRAIT:
			return 1;
		case OS::SCREEN_REVERSE_LANDSCAPE:
			return 8;
		case OS::SCREEN_REVERSE_PORTRAIT:
			return 9;
		case OS::SCREEN_SENSOR_LANDSCAPE:
			return 11;
		case OS::SCREEN_SENSOR_PORTRAIT:
			return 12;
		case OS::SCREEN_SENSOR:
			return 13;
		case OS::SCREEN_LANDSCAPE:
		default:
			return 0;
	}
}

String _get_android_orientation_label(OS::ScreenOrientation screen_orientation) {
	switch (screen_orientation) {
		case OS::SCREEN_PORTRAIT:
			return "portrait";
		case OS::SCREEN_REVERSE_LANDSCAPE:
			return "reverseLandscape";
		case OS::SCREEN_REVERSE_PORTRAIT:
			return "reversePortrait";
		case OS::SCREEN_SENSOR_LANDSCAPE:
			return "userLandscape";
		case OS::SCREEN_SENSOR_PORTRAIT:
			return "userPortrait";
		case OS::SCREEN_SENSOR:
			return "fullUser";
		case OS::SCREEN_LANDSCAPE:
		default:
			return "landscape";
	}
}

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
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Unable to write data into " + p_path);
		}
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
Error rename_and_store_file_in_gradle_project(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {
	CustomExportData *export_data = (CustomExportData *)p_userdata;
	String dst_path = p_path.replace_first("res://", export_data->assets_directory + "/");
	print_verbose("Saving project files from " + p_path + " into " + dst_path);
	Error err = store_file_at_path(dst_path, p_data);
	return err;
}

String _android_xml_escape(const String &p_string) {
	// Android XML requires strings to be both valid XML (`xml_escape()`) but also
	// to escape characters which are valid XML but have special meaning in Android XML.
	// https://developer.android.com/guide/topics/resources/string-resource.html#FormattingAndStyling
	// Note: Didn't handle U+XXXX unicode chars, could be done if needed.
	return p_string
			.replace("@", "\\@")
			.replace("?", "\\?")
			.replace("'", "\\'")
			.replace("\"", "\\\"")
			.replace("\n", "\\n")
			.replace("\t", "\\t")
			.xml_escape(false);
}

// Creates strings.xml files inside the gradle project for different locales.
Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &project_name) {
	print_verbose("Creating strings resources for supported locales for project " + project_name);
	// Stores the string into the default values directory.
	String processed_default_xml_string = vformat(godot_project_name_xml_string, _android_xml_escape(project_name));
	store_string_at_path("res://android/build/res/values/godot_project_name_string.xml", processed_default_xml_string);

	// Searches the Gradle project res/ directory to find all supported locales
	DirAccessRef da = DirAccess::open("res://android/build/res");
	if (!da) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Unable to open Android resources directory.");
		}
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
			String processed_xml_string = vformat(godot_project_name_xml_string, _android_xml_escape(locale_project_name));
			print_verbose("Storing project name for locale " + locale + " under " + locale_directory);
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
	int xr_mode_index = (int)(p_preset->get("xr_features/xr_mode"));
	bool uses_xr = xr_mode_index == XR_MODE_OVR || xr_mode_index == XR_MODE_OPENXR;
	if (uses_xr) {
		manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"android.hardware.vr.headtracking\" android:required=\"true\" android:version=\"1\" />\n";

		int hand_tracking_index = p_preset->get("xr_features/hand_tracking"); // 0: none, 1: optional, 2: required
		if (hand_tracking_index == XR_HAND_TRACKING_OPTIONAL) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"oculus.software.handtracking\" android:required=\"false\" />\n";
		} else if (hand_tracking_index == XR_HAND_TRACKING_REQUIRED) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"oculus.software.handtracking\" android:required=\"true\" />\n";
		}

		int passthrough_mode = p_preset->get("xr_features/passthrough");
		if (passthrough_mode == XR_PASSTHROUGH_OPTIONAL) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"com.oculus.feature.PASSTHROUGH\" android:required=\"false\" />\n";
		} else if (passthrough_mode == XR_PASSTHROUGH_REQUIRED) {
			manifest_xr_features += "    <uses-feature tools:node=\"replace\" android:name=\"com.oculus.feature.PASSTHROUGH\" android:required=\"true\" />\n";
		}
	}
	return manifest_xr_features;
}

String _get_activity_tag(const Ref<EditorExportPreset> &p_preset) {
	int xr_mode_index = (int)(p_preset->get("xr_features/xr_mode"));
	bool uses_xr = xr_mode_index == XR_MODE_OVR || xr_mode_index == XR_MODE_OPENXR;
	String orientation = _get_android_orientation_label(
			OS::get_singleton()->get_screen_orientation_from_string(GLOBAL_GET("display/window/handheld/orientation")));
	String manifest_activity_text = vformat(
			"        <activity android:name=\"com.godot.game.GodotApp\" "
			"tools:replace=\"android:screenOrientation,android:excludeFromRecents,android:resizeableActivity\" "
			"android:excludeFromRecents=\"%s\" "
			"android:screenOrientation=\"%s\" "
			"android:resizeableActivity=\"%s\">\n",
			bool_to_string(p_preset->get("package/exclude_from_recents")),
			orientation,
			bool_to_string(bool(GLOBAL_GET("display/window/size/resizable"))));
	if (uses_xr) {
		manifest_activity_text += "            <meta-data tools:node=\"replace\" android:name=\"com.oculus.vr.focusaware\" android:value=\"true\" />\n";
	} else {
		manifest_activity_text += "            <meta-data tools:node=\"remove\" android:name=\"com.oculus.vr.focusaware\" />\n";
	}
	manifest_activity_text += "        </activity>\n";
	return manifest_activity_text;
}

String _get_application_tag(const Ref<EditorExportPreset> &p_preset, bool p_has_read_write_storage_permission) {
	int xr_mode_index = (int)(p_preset->get("xr_features/xr_mode"));
	bool uses_xr = xr_mode_index == XR_MODE_OVR || xr_mode_index == XR_MODE_OPENXR;
	String manifest_application_text = vformat(
			"    <application android:label=\"@string/godot_project_name_string\"\n"
			"        android:allowBackup=\"%s\"\n"
			"        android:isGame=\"%s\"\n"
			"        android:hasFragileUserData=\"%s\"\n"
			"        android:requestLegacyExternalStorage=\"%s\"\n"
			"        tools:replace=\"android:allowBackup,android:isGame,android:hasFragileUserData,android:requestLegacyExternalStorage\"\n"
			"        tools:ignore=\"GoogleAppIndexingWarning\"\n"
			"        android:icon=\"@mipmap/icon\" >\n\n"
			"        <meta-data tools:node=\"remove\" android:name=\"xr_mode_metadata_name\" />\n"
			"        <meta-data tools:node=\"remove\" android:name=\"xr_hand_tracking_version_name\" />\n"
			"        <meta-data tools:node=\"remove\" android:name=\"xr_hand_tracking_metadata_name\" />\n",
			bool_to_string(p_preset->get("user_data_backup/allow")),
			bool_to_string(p_preset->get("package/classify_as_game")),
			bool_to_string(p_preset->get("package/retain_data_on_uninstall")),
			bool_to_string(p_has_read_write_storage_permission));

	if (uses_xr) {
		if (xr_mode_index == XR_MODE_OVR) {
			manifest_application_text += "        <meta-data tools:node=\"replace\" android:name=\"com.samsung.android.vr.application.mode\" android:value=\"vr_only\" />\n";
		}

		bool hand_tracking_enabled = (int)(p_preset->get("xr_features/hand_tracking")) > XR_HAND_TRACKING_NONE;
		if (hand_tracking_enabled) {
			int hand_tracking_frequency_index = p_preset->get("xr_features/hand_tracking_frequency");
			String hand_tracking_frequency = hand_tracking_frequency_index == XR_HAND_TRACKING_FREQUENCY_LOW ? "LOW" : "HIGH";
			manifest_application_text += vformat(
					"        <meta-data tools:node=\"replace\" android:name=\"com.oculus.handtracking.frequency\" android:value=\"%s\" />\n",
					hand_tracking_frequency);
			manifest_application_text += "        <meta-data tools:node=\"replace\" android:name=\"com.oculus.handtracking.version\" android:value=\"V2.0\" />\n";
		}
	} else {
		manifest_application_text += "        <meta-data tools:node=\"remove\" android:name=\"com.oculus.supportedDevices\" />\n";
	}
	manifest_application_text += _get_activity_tag(p_preset);
	manifest_application_text += "    </application>\n";
	return manifest_application_text;
}
