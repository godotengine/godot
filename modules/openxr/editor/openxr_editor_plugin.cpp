/**************************************************************************/
/*  openxr_editor_plugin.cpp                                              */
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

#include "openxr_editor_plugin.h"

#include "../action_map/openxr_action_map.h"

#include "editor/editor_node.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/settings/editor_command_palette.h"
#include "platform/android/export/export_plugin.h"

#include <openxr/openxr.h>

////////////////////////////////////////////////////////////////////////////
// OpenXRExportPlugin

bool OpenXRExportPlugin::supports_platform(const Ref<EditorExportPlatform> &p_export_platform) const {
	return p_export_platform->is_class(EditorExportPlatformAndroid::get_class_static());
}

bool OpenXRExportPlugin::is_openxr_mode() const {
	int xr_mode_index = get_option("xr_features/xr_mode");

	return xr_mode_index == XR_MODE_OPENXR;
}

void OpenXRExportPlugin::_get_export_options(const Ref<EditorExportPlatform> &p_export_platform, List<EditorExportPlatform::ExportOption> *r_options) const {
	if (!supports_platform(p_export_platform)) {
		return;
	}

	PropertyInfo openxr_version(Variant::STRING, "xr_features/custom_openxr_loader", PROPERTY_HINT_PLACEHOLDER_TEXT, "Custom OpenXR Loader [default if blank]");
	Variant default_value = "";
	bool update_visibility = true;
	r_options->push_back(EditorExportPlatform::ExportOption(openxr_version, default_value, update_visibility));
}

Dictionary OpenXRExportPlugin::_get_export_options_overrides(const Ref<EditorExportPlatform> &p_export_platform) const {
	if (!supports_platform(p_export_platform)) {
		return Dictionary();
	}

	Dictionary overrides;

	if (!is_openxr_mode()) {
		// Creating an override will hide this property.
		overrides["xr_features/openxr_version"] = "disabled";
	}

	return overrides;
}

String OpenXRExportPlugin::_get_export_option_warning(const Ref<EditorExportPlatform> &p_export_platform, const String &p_option_name) const {
	if (!supports_platform(p_export_platform)) {
		return String();
	}

	bool gradle_build = get_option("gradle_build/use_gradle_build");
	if (is_openxr_mode() && !gradle_build) {
		return "\"Use Gradle Build\" must be enabled when xr_mode is set to \"OpenXR\".";
	}

	return String();
}

PackedStringArray OpenXRExportPlugin::get_android_dependencies(const Ref<EditorExportPlatform> &p_export_platform, bool p_debug) const {
	PackedStringArray ret;

	if (!supports_platform(p_export_platform)) {
		return ret;
	}

	if (is_openxr_mode()) {
		String openxr_loader = get_option("xr_features/custom_openxr_loader");

		// If no loader is specified, match the openxr loader in our thirdparty folder.
		if (openxr_loader.is_empty()) {
			// Loader is always identified by the full API version even if we're initializing for OpenXR 1.0.
			int major = XR_VERSION_MAJOR(XR_CURRENT_API_VERSION);
			int minor = XR_VERSION_MINOR(XR_CURRENT_API_VERSION);
			int patch = XR_VERSION_PATCH(XR_CURRENT_API_VERSION);
			openxr_loader = "org.khronos.openxr:openxr_loader_for_android:" + String::num_int64(major) + "." + String::num_int64(minor) + "." + String::num_int64(patch);
		}

		// We allow plugins to mark this as disabled as well if they introduce their own loader logic.
		if (openxr_loader != "disabled") {
			ret.push_back(openxr_loader);
		}
	}

	return ret;
}

PackedStringArray OpenXRExportPlugin::_get_export_features(const Ref<EditorExportPlatform> &p_export_platform, bool p_debug) const {
	PackedStringArray features;

	if (!supports_platform(p_export_platform) || !is_openxr_mode()) {
		return features;
	}

	// Placeholder for now

	return features;
}

String OpenXRExportPlugin::get_android_manifest_element_contents(const Ref<EditorExportPlatform> &p_export_platform, bool p_debug) const {
	String contents;

	if (!supports_platform(p_export_platform) || !is_openxr_mode()) {
		return contents;
	}

	// Add in permissions.
	contents += "    <uses-permission android:name=\"org.khronos.openxr.permission.OPENXR\" />\n";
	contents += "    <uses-permission android:name=\"org.khronos.openxr.permission.OPENXR_SYSTEM\" />\n";

	// Add in queries.
	contents += "    <queries>\n";
	// To talk to the broker.
	contents += "        <provider android:authorities=\"org.khronos.openxr.runtime_broker;org.khronos.openxr.system_runtime_broker\" />\n";

	// So client-side code of runtime/layers can talk to their service sides
	contents += "        <intent>\n";
	contents += "            <action android:name=\"org.khronos.openxr.OpenXRRuntimeService\" />\n";
	contents += "        </intent>\n";
	contents += "        <intent>\n";
	contents += "            <action android:name=\"org.khronos.openxr.OpenXRApiLayerService\" />\n";
	contents += "        </intent>\n";

	contents += "    </queries>\n";

	// Add in features.

	// Indicates we support 6DOF tracking and as this is not required, also support 3DOF tracking.
	contents += "    <uses-feature android:name=\"android.hardware.vr.headtracking\" android:required=\"false\" android:version=\"1\"/>\n";

	return contents;
}

String OpenXRExportPlugin::get_android_manifest_activity_element_contents(const Ref<EditorExportPlatform> &p_export_platform, bool p_debug) const {
	String contents;

	if (!supports_platform(p_export_platform) || !is_openxr_mode()) {
		return contents;
	}

	contents += "            <intent-filter>\n";
	contents += "                <action android:name=\"android.intent.action.MAIN\" />\n";

	// OpenXR category tag to indicate the activity starts in an immersive OpenXR mode.
	// See https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#android-runtime-category.
	contents += "                <category android:name=\"org.khronos.openxr.intent.category.IMMERSIVE_HMD\" />\n";

	contents += "            </intent-filter>\n";

	return contents;
}

////////////////////////////////////////////////////////////////////////////
// OpenXREditorPlugin

void OpenXREditorPlugin::edit(Object *p_node) {
	if (Object::cast_to<OpenXRActionMap>(p_node)) {
		String path = Object::cast_to<OpenXRActionMap>(p_node)->get_path();
		if (path.is_resource_file()) {
			action_map_editor->open_action_map(path);
		}
	}
}

bool OpenXREditorPlugin::handles(Object *p_node) const {
	return (Object::cast_to<OpenXRActionMap>(p_node) != nullptr);
}

void OpenXREditorPlugin::make_visible(bool p_visible) {
}

OpenXREditorPlugin::OpenXREditorPlugin() {
	action_map_editor = memnew(OpenXRActionMapEditor);
	EditorNode::get_bottom_panel()->add_item(TTRC("OpenXR Action Map"), action_map_editor, ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_openxr_action_map_bottom_panel", TTRC("Toggle OpenXR Action Map Bottom Panel")));

	binding_modifier_inspector_plugin = Ref<EditorInspectorPluginBindingModifier>(memnew(EditorInspectorPluginBindingModifier));
	EditorInspector::add_inspector_plugin(binding_modifier_inspector_plugin);

#ifndef ANDROID_ENABLED
	select_runtime = memnew(OpenXRSelectRuntime);
	add_control_to_container(CONTAINER_TOOLBAR, select_runtime);
#endif
}

void OpenXREditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Initialize our export plugin
			openxr_export_plugin.instantiate();
			add_export_plugin(openxr_export_plugin);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// Clean up our export plugin
			remove_export_plugin(openxr_export_plugin);

			openxr_export_plugin.unref();
		}
	}
}
