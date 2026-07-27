/**************************************************************************/
/*  export_plugin.h                                                       */
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

#include "editor/export/editor_export_platform_apple_embedded.h"

class EditorExportPlatformVisionOS : public EditorExportPlatformAppleEmbedded {
	GDCLASS(EditorExportPlatformVisionOS, EditorExportPlatformAppleEmbedded);

	static Vector<String> device_types;

	virtual String get_platform_name() const override { return "visionos"; }
	virtual String get_sdk_name() const override { return "xros"; }
	virtual const Vector<String> get_device_types() const override { return device_types; }

	virtual String get_minimum_deployment_target() const override { return "26.0"; }

	virtual Vector<EditorExportPlatformAppleEmbedded::IconInfo> get_icon_infos() const override;

	virtual void get_export_options(List<ExportOption> *r_options) const override;

	virtual String _process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing) override;

public:
	virtual String get_name() const override { return "visionOS"; }
	virtual String get_os_name() const override { return "visionOS"; }

	virtual void get_platform_features(List<String> *r_features) const override {
		EditorExportPlatformAppleEmbedded::get_platform_features(r_features);
		r_features->push_back("visionos");
	}

	virtual void initialize() override;
	~EditorExportPlatformVisionOS();
};
