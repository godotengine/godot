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

#ifndef WEB_EXPORT_PLUGIN_H
#define WEB_EXPORT_PLUGIN_H

#include "editor_http_server.h"

#include "core/config/project_settings.h"
#include "core/io/image_loader.h"
#include "core/io/stream_peer_tls.h"
#include "core/io/tcp_server.h"
#include "core/io/zip_io.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/export/editor_export_platform.h"
#include "main/splash.gen.h"

class EditorExportPlatformWeb : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformWeb, EditorExportPlatform);

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;
	Ref<ImageTexture> stop_icon;
	int menu_options = 0;

	Ref<EditorHTTPServer> server;

	String _get_template_name(bool p_extension, bool p_thread_support, bool p_debug) const {
		String name = "web";
		if (p_extension) {
			name += "_dlink";
		}
		if (!p_thread_support) {
			name += "_nothreads";
		}
		if (p_debug) {
			name += "_debug.zip";
		} else {
			name += "_release.zip";
		}
		return name;
	}

	Ref<Image> _get_project_icon() const {
		Ref<Image> icon;
		icon.instantiate();
		const String icon_path = String(GLOBAL_GET("application/config/icon")).strip_edges();
		if (icon_path.is_empty() || ImageLoader::load_image(icon_path, icon) != OK) {
			return EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("DefaultProjectIcon"), EditorStringName(EditorIcons))->get_image();
		}
		return icon;
	}

	Ref<Image> _get_project_splash() const {
		Ref<Image> splash;
		splash.instantiate();
		const String splash_path = String(GLOBAL_GET("application/boot_splash/image")).strip_edges();
		if (splash_path.is_empty() || ImageLoader::load_image(splash_path, splash) != OK) {
			return Ref<Image>(memnew(Image(boot_splash_png)));
		}
		return splash;
	}

	Error _extract_template(const String &p_template, const String &p_dir, const String &p_name, bool pwa);
	void _replace_strings(const HashMap<String, String> &p_replaces, Vector<uint8_t> &r_template);
	void _fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug, int p_flags, const Vector<SharedObject> p_shared_objects, const Dictionary &p_file_sizes);
	Error _add_manifest_icon(const String &p_path, const String &p_icon, int p_size, Array &r_arr);
	Error _build_pwa(const Ref<EditorExportPreset> &p_preset, const String p_path, const Vector<SharedObject> &p_shared_objects);
	Error _write_or_error(const uint8_t *p_content, int p_len, String p_path);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const override;

	virtual void get_export_options(List<ExportOption> *r_options) const override;

	virtual String get_name() const override;
	virtual String get_os_name() const override;
	virtual Ref<Texture2D> get_logo() const override;

	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug = false) const override;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const override;
	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override;

	virtual bool poll_export() override;
	virtual int get_options_count() const override;
	virtual String get_option_label(int p_index) const override { return p_index ? TTR("Stop HTTP Server") : TTR("Run in Browser"); }
	virtual String get_option_tooltip(int p_index) const override { return p_index ? TTR("Stop HTTP Server") : TTR("Run exported HTML in the system's default browser."); }
	virtual Ref<ImageTexture> get_option_icon(int p_index) const override;
	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_option, int p_debug_flags) override;
	virtual Ref<Texture2D> get_run_icon() const override;

	virtual void get_platform_features(List<String> *r_features) const override {
		r_features->push_back("web");
		r_features->push_back(get_os_name().to_lower());
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, HashSet<String> &p_features) override {
	}

	String get_debug_protocol() const override { return "ws://"; }

	EditorExportPlatformWeb();
	~EditorExportPlatformWeb();
};

#endif // WEB_EXPORT_PLUGIN_H
