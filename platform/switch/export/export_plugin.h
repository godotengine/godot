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

#ifndef SWITCH_EXPORT_PLUGIN_H
#define SWITCH_EXPORT_PLUGIN_H

#include "core/io/file_access.h"
#include "editor/editor_settings.h"
#include "editor/export/editor_export_platform_pc.h"
#include "scene/resources/texture.h"

class EditorExportPlatformSwitch : public EditorExportPlatformPC {
	GDCLASS(EditorExportPlatformSwitch, EditorExportPlatformPC);

	Ref<ImageTexture> run_icon;
	Ref<ImageTexture> stop_icon;

	int menu_options = 0;

public:
	virtual void get_export_options(List<ExportOption> *r_options) const override;
	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override;
	virtual bool get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const override;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override;
	virtual String get_template_file_name(const String &p_target, const String &p_arch) const override;
	String get_export_option_warning(const EditorExportPreset *p_preset, const StringName &p_name) const override;
	virtual Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) override;

	virtual Ref<Texture2D> get_run_icon() const override;
	virtual bool poll_export() override;
	virtual Ref<ImageTexture> get_option_icon(int p_index) const override;
	virtual int get_options_count() const override;
	virtual String get_option_label(int p_index) const override;
	virtual String get_option_tooltip(int p_index) const override;

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) override;

	EditorExportPlatformSwitch();
};

#endif // SWITCH_EXPORT_PLUGIN_H
