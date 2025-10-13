/**************************************************************************/
/*  export_plugin.cpp                                                     */
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

#include "export_plugin.h"

#include "logo_svg.gen.h"
#include "run_icon_svg.gen.h"

#include "editor/editor_node.h"

Vector<String> EditorExportPlatformVisionOS::device_types({ "realityDevice" });

void EditorExportPlatformVisionOS::initialize() {
	if (EditorNode::get_singleton()) {
		EditorExportPlatformAppleEmbedded::_initialize(_visionos_logo_svg, _visionos_run_icon_svg);
#ifdef MACOS_ENABLED
		_start_remote_device_poller_thread();
#endif
	}
}

EditorExportPlatformVisionOS::~EditorExportPlatformVisionOS() {
#ifdef MACOS_ENABLED
	_stop_remote_device_poller_thread();
#endif
}

void EditorExportPlatformVisionOS::get_export_options(List<ExportOption> *r_options) const {
	EditorExportPlatformAppleEmbedded::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/min_visionos_version"), get_minimum_deployment_target()));
}

Vector<EditorExportPlatformAppleEmbedded::IconInfo> EditorExportPlatformVisionOS::get_icon_infos() const {
	return Vector<EditorExportPlatformAppleEmbedded::IconInfo>();
}

String EditorExportPlatformVisionOS::_process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing) {
	// Do visionOS specific processing first, and call super implementation if there are no matches

	String strnew;

	// Supported Destinations
	if (p_line.contains("$targeted_device_family")) {
		strnew += p_line.replace("$targeted_device_family", "7") + "\n";

		// MoltenVK Framework not used on visionOS
	} else if (p_line.contains("$moltenvk_buildfile")) {
		strnew += p_line.replace("$moltenvk_buildfile", "") + "\n";
	} else if (p_line.contains("$moltenvk_fileref")) {
		strnew += p_line.replace("$moltenvk_fileref", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildphase")) {
		strnew += p_line.replace("$moltenvk_buildphase", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildgrp")) {
		strnew += p_line.replace("$moltenvk_buildgrp", "") + "\n";

		// Launch Storyboard
	} else if (p_line.contains("$plist_launch_screen_name")) {
		strnew += p_line.replace("$plist_launch_screen_name", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_file_reference")) {
		strnew += p_line.replace("$pbx_launch_screen_file_reference", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_copy_files")) {
		strnew += p_line.replace("$pbx_launch_screen_copy_files", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_phase")) {
		strnew += p_line.replace("$pbx_launch_screen_build_phase", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_reference")) {
		strnew += p_line.replace("$pbx_launch_screen_build_reference", "") + "\n";

		// OS Deployment Target
	} else if (p_line.contains("$os_deployment_target")) {
		String min_version = p_preset->get("application/min_" + get_platform_name() + "_version");
		String value = "XROS_DEPLOYMENT_TARGET = " + min_version + ";";
		strnew += p_line.replace("$os_deployment_target", value) + "\n";

		// Valid Archs
	} else if (p_line.contains("$valid_archs")) {
		strnew += p_line.replace("$valid_archs", "arm64") + "\n";

		// Apple Embedded common
	} else {
		strnew += EditorExportPlatformAppleEmbedded::_process_config_file_line(p_preset, p_line, p_config, p_debug, p_code_signing);
	}
	return strnew;
}
