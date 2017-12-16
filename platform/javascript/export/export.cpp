/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor/editor_node.h"
#include "editor_export.h"
#include "io/zip_io.h"
#include "main/splash.gen.h"
#include "platform/javascript/logo.gen.h"
#include "platform/javascript/run_icon.gen.h"

#define EXPORT_TEMPLATE_WEBASSEMBLY_RELEASE "webassembly_release.zip"
#define EXPORT_TEMPLATE_WEBASSEMBLY_DEBUG "webassembly_debug.zip"

class EditorExportPlatformJavaScript : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformJavaScript, EditorExportPlatform)

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;
	bool runnable_when_last_polled;

	void _fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;
	virtual String get_os_name() const;
	virtual Ref<Texture> get_logo() const;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual String get_binary_extension(const Ref<EditorExportPreset> &p_preset) const;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool poll_devices();
	virtual int get_device_count() const;
	virtual String get_device_name(int p_device) const { return TTR("Run in Browser"); }
	virtual String get_device_info(int p_device) const { return TTR("Run exported HTML in the system's default browser."); }
	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags);
	virtual Ref<Texture> get_run_icon() const;

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("web");
		r_features->push_back("JavaScript");
	}

	EditorExportPlatformJavaScript();
};

void EditorExportPlatformJavaScript::_fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug) {

	String str_template = String::utf8(reinterpret_cast<const char *>(p_html.ptr()), p_html.size());
	String str_export;
	Vector<String> lines = str_template.split("\n");

	for (int i = 0; i < lines.size(); i++) {

		String current_line = lines[i];
		current_line = current_line.replace("$GODOT_BASENAME", p_name);
		current_line = current_line.replace("$GODOT_HEAD_INCLUDE", p_preset->get("html/head_include"));
		current_line = current_line.replace("$GODOT_DEBUG_ENABLED", p_debug ? "true" : "false");
		str_export += current_line + "\n";
	}

	CharString cs = str_export.utf8();
	p_html.resize(cs.length());
	for (int i = 0; i < cs.length(); i++) {
		p_html[i] = cs[i];
	}
}

void EditorExportPlatformJavaScript::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}
}

void EditorExportPlatformJavaScript::get_export_options(List<ExportOption> *r_options) {

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/custom_html_shell", PROPERTY_HINT_GLOBAL_FILE, "html"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/head_include", PROPERTY_HINT_MULTILINE_TEXT), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));
}

String EditorExportPlatformJavaScript::get_name() const {

	return "HTML5";
}

String EditorExportPlatformJavaScript::get_os_name() const {

	return "JavaScript";
}

Ref<Texture> EditorExportPlatformJavaScript::get_logo() const {

	return logo;
}

bool EditorExportPlatformJavaScript::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	r_missing_templates = false;

	if (find_export_template(EXPORT_TEMPLATE_WEBASSEMBLY_RELEASE) == String())
		r_missing_templates = true;
	else if (find_export_template(EXPORT_TEMPLATE_WEBASSEMBLY_DEBUG) == String())
		r_missing_templates = true;

	return !r_missing_templates;
}

String EditorExportPlatformJavaScript::get_binary_extension(const Ref<EditorExportPreset> &p_preset) const {

	return "html";
}

Error EditorExportPlatformJavaScript::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String custom_debug = p_preset->get("custom_template/debug");
	String custom_release = p_preset->get("custom_template/release");
	String custom_html = p_preset->get("html/custom_html_shell");

	String template_path = p_debug ? custom_debug : custom_release;

	template_path = template_path.strip_edges();

	if (template_path == String()) {

		if (p_debug)
			template_path = find_export_template(EXPORT_TEMPLATE_WEBASSEMBLY_DEBUG);
		else
			template_path = find_export_template(EXPORT_TEMPLATE_WEBASSEMBLY_RELEASE);
	}

	if (template_path != String() && !FileAccess::exists(template_path)) {
		EditorNode::get_singleton()->show_warning(TTR("Template file not found:\n") + template_path);
		return ERR_FILE_NOT_FOUND;
	}

	String pck_path = p_path.get_basename() + ".pck";
	Error error = save_pack(p_preset, pck_path);
	if (error != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Could not write file:\n") + pck_path);
		return error;
	}

	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);
	unzFile pkg = unzOpen2(template_path.utf8().get_data(), &io);

	if (!pkg) {

		EditorNode::get_singleton()->show_warning(TTR("Could not open template for export:\n") + template_path);
		return ERR_FILE_NOT_FOUND;
	}

	if (unzGoToFirstFile(pkg) != UNZ_OK) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid export template:\n") + template_path);
		unzClose(pkg);
		return ERR_FILE_CORRUPT;
	}

	do {
		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptrw(), data.size());
		unzCloseCurrentFile(pkg);

		//write

		if (file == "godot.html") {

			if (!custom_html.empty()) {
				continue;
			}
			_fix_html(data, p_preset, p_path.get_file().get_basename(), p_debug);
			file = p_path.get_file();

		} else if (file == "godot.js") {

			file = p_path.get_file().get_basename() + ".js";
		} else if (file == "godot.wasm") {

			file = p_path.get_file().get_basename() + ".wasm";
		}

		String dst = p_path.get_base_dir().plus_file(file);
		FileAccess *f = FileAccess::open(dst, FileAccess::WRITE);
		if (!f) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:\n") + dst);
			unzClose(pkg);
			return ERR_FILE_CANT_WRITE;
		}
		f->store_buffer(data.ptr(), data.size());
		memdelete(f);

	} while (unzGoToNextFile(pkg) == UNZ_OK);
	unzClose(pkg);

	if (!custom_html.empty()) {

		FileAccess *f = FileAccess::open(custom_html, FileAccess::READ);
		if (!f) {
			EditorNode::get_singleton()->show_warning(TTR("Could not read custom HTML shell:\n") + custom_html);
			return ERR_FILE_CANT_READ;
		}
		Vector<uint8_t> buf;
		buf.resize(f->get_len());
		f->get_buffer(buf.ptrw(), buf.size());
		memdelete(f);
		_fix_html(buf, p_preset, p_path.get_file().get_basename(), p_debug);

		f = FileAccess::open(p_path, FileAccess::WRITE);
		if (!f) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:\n") + p_path);
			return ERR_FILE_CANT_WRITE;
		}
		f->store_buffer(buf.ptr(), buf.size());
		memdelete(f);
	}

	Ref<Image> splash;
	String splash_path = GLOBAL_GET("application/boot_splash/image");
	splash_path = splash_path.strip_edges();
	if (!splash_path.empty()) {
		splash.instance();
		Error err = splash->load(splash_path);
		if (err) {
			EditorNode::get_singleton()->show_warning(TTR("Could not read boot splash image file:\n") + splash_path + "\nUsing default boot splash image");
			splash.unref();
		}
	}
	if (splash.is_null()) {
		splash = Ref<Image>(memnew(Image(boot_splash_png)));
	}
	String png_path = p_path.get_base_dir().plus_file(p_path.get_file().get_basename() + ".png");
	if (splash->save_png(png_path) != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Could not write file:\n") + png_path);
		return ERR_FILE_CANT_WRITE;
	}
	return OK;
}

bool EditorExportPlatformJavaScript::poll_devices() {

	Ref<EditorExportPreset> preset;

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {

		Ref<EditorExportPreset> ep = EditorExport::get_singleton()->get_export_preset(i);
		if (ep->is_runnable() && ep->get_platform() == this) {
			preset = ep;
			break;
		}
	}

	bool prev = runnable_when_last_polled;
	runnable_when_last_polled = preset.is_valid();
	return runnable_when_last_polled != prev;
}

int EditorExportPlatformJavaScript::get_device_count() const {

	return runnable_when_last_polled;
}

Error EditorExportPlatformJavaScript::run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) {

	String path = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmp_export.html");
	Error err = export_project(p_preset, true, path, p_debug_flags);
	if (err) {
		return err;
	}
	OS::get_singleton()->shell_open(path);
	return OK;
}

Ref<Texture> EditorExportPlatformJavaScript::get_run_icon() const {

	return run_icon;
}

EditorExportPlatformJavaScript::EditorExportPlatformJavaScript() {

	Ref<Image> img = memnew(Image(_javascript_logo));
	logo.instance();
	logo->create_from_image(img);

	img = Ref<Image>(memnew(Image(_javascript_run_icon)));
	run_icon.instance();
	run_icon->create_from_image(img);

	runnable_when_last_polled = false;
}

void register_javascript_exporter() {

	Ref<EditorExportPlatformJavaScript> platform;
	platform.instance();
	EditorExport::get_singleton()->add_export_platform(platform);
}
