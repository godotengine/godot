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

#include "core/config/project_settings.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/export/editor_export_platform.h"
#include "editor/import/resource_importer_texture_settings.h"
#include "main/splash.gen.h"
#include "scene/resources/image_texture.h"

#include <string.h>

class EditorExportPlatformOpenHarmony : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformOpenHarmony, EditorExportPlatform);

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;

	String get_tool_path() const;
	String get_java_sdk_path() const;
	String get_sdk_path() const;
	String get_hvigor_path() const;
	String get_hvigor_path_ide() const;
	String get_hdc_path() const;

	Vector<String> devices;
	SafeFlag devices_changed;
	Mutex device_lock;
	Thread check_for_changes_thread;
	SafeFlag quit_request;
	SafeFlag has_runnable_preset;
	static void _check_for_changes_poll_thread(void *p_ud);
	void _update_preset_status();
	void _remove_dir_recursive(const String &p_dir);

protected:
	void _notification(int p_what);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const override;

	virtual void get_export_options(List<ExportOption> *r_options) const override;
	virtual bool get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const override;
	virtual String get_export_option_warning(const EditorExportPreset *p_preset, const StringName &p_name) const override;

	virtual String get_name() const override;

	virtual String get_os_name() const override;

	virtual Ref<Texture2D> get_logo() const override;

	virtual Ref<Texture2D> get_run_icon() const override;

	virtual bool poll_export() override;
	virtual int get_options_count() const override;
	virtual String get_options_tooltip() const override;
	virtual String get_option_label(int p_device) const override;
	virtual String get_option_tooltip(int p_device) const override;
	virtual String get_device_architecture(int p_device) const override;

	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug = false) const override;

	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const override;

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override;

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags = 0) override;

	Error export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, bool should_sign, bool export_project_only, BitField<EditorExportPlatform::DebugFlags> p_flags);

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) override;

	virtual void get_platform_features(List<String> *r_features) const override;

	EditorExportPlatformOpenHarmony();
	~EditorExportPlatformOpenHarmony();
};
