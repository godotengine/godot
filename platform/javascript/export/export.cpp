/*************************************************************************/
/*  export.cpp                                                           */
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

#include "core/io/tcp_server.h"
#include "core/io/zip_io.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "main/splash.gen.h"
#include "platform/javascript/logo.gen.h"
#include "platform/javascript/run_icon.gen.h"

#define EXPORT_TEMPLATE_WEBASSEMBLY_RELEASE "webassembly_release.zip"
#define EXPORT_TEMPLATE_WEBASSEMBLY_DEBUG "webassembly_debug.zip"

class EditorHTTPServer : public Reference {

private:
	Ref<TCP_Server> server;
	Ref<StreamPeerTCP> connection;
	uint64_t time;
	uint8_t req_buf[4096];
	int req_pos;

	void _clear_client() {
		connection = Ref<StreamPeerTCP>();
		memset(req_buf, 0, sizeof(req_buf));
		time = 0;
		req_pos = 0;
	}

public:
	EditorHTTPServer() {
		server.instance();
		stop();
	}

	void stop() {
		server->stop();
		_clear_client();
	}

	Error listen(int p_port, IP_Address p_address) {
		return server->listen(p_port, p_address);
	}

	bool is_listening() const {
		return server->is_listening();
	}

	void _send_response() {
		Vector<String> psa = String((char *)req_buf).split("\r\n");
		int len = psa.size();
		ERR_FAIL_COND_MSG(len < 4, "Not enough response headers, got: " + itos(len) + ", expected >= 4.");

		Vector<String> req = psa[0].split(" ", false);
		ERR_FAIL_COND_MSG(req.size() < 2, "Invalid protocol or status code.");

		// Wrong protocol
		ERR_FAIL_COND_MSG(req[0] != "GET" || req[2] != "HTTP/1.1", "Invalid method or HTTP version.");

		const String cache_path = EditorSettings::get_singleton()->get_cache_dir();
		const String basereq = "/tmp_js_export";
		String filepath;
		String ctype;
		if (req[1] == basereq + ".html") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "text/html";
		} else if (req[1] == basereq + ".js") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/javascript";
		} else if (req[1] == basereq + ".audio.worklet.js") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/javascript";
		} else if (req[1] == basereq + ".worker.js") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/javascript";
		} else if (req[1] == basereq + ".pck") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/octet-stream";
		} else if (req[1] == basereq + ".png" || req[1] == "/favicon.png") {
			// Also allow serving the generated favicon for a smoother loading experience.
			if (req[1] == "/favicon.png") {
				filepath = EditorSettings::get_singleton()->get_cache_dir().plus_file("favicon.png");
			} else {
				filepath = basereq + ".png";
			}
			ctype = "image/png";
		} else if (req[1] == basereq + ".side.wasm") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/wasm";
		} else if (req[1] == basereq + ".wasm") {
			filepath = cache_path.plus_file(req[1].get_file());
			ctype = "application/wasm";
		} else if (req[1].ends_with(".wasm")) {
			filepath = cache_path.plus_file(req[1].get_file()); // TODO dangerous?
			ctype = "application/wasm";
		}
		if (filepath.empty() || !FileAccess::exists(filepath)) {
			String s = "HTTP/1.1 404 Not Found\r\n";
			s += "Connection: Close\r\n";
			s += "\r\n";
			CharString cs = s.utf8();
			connection->put_data((const uint8_t *)cs.get_data(), cs.size() - 1);
			return;
		}
		FileAccess *f = FileAccess::open(filepath, FileAccess::READ);
		ERR_FAIL_COND(!f);
		String s = "HTTP/1.1 200 OK\r\n";
		s += "Connection: Close\r\n";
		s += "Content-Type: " + ctype + "\r\n";
		s += "Access-Control-Allow-Origin: *\r\n";
		s += "Cross-Origin-Opener-Policy: same-origin\r\n";
		s += "Cross-Origin-Embedder-Policy: require-corp\r\n";
		s += "\r\n";
		CharString cs = s.utf8();
		Error err = connection->put_data((const uint8_t *)cs.get_data(), cs.size() - 1);
		if (err != OK) {
			memdelete(f);
			ERR_FAIL();
		}

		while (true) {
			uint8_t bytes[4096];
			int read = f->get_buffer(bytes, 4096);
			if (read < 1) {
				break;
			}
			err = connection->put_data(bytes, read);
			if (err != OK) {
				memdelete(f);
				ERR_FAIL();
			}
		}
		memdelete(f);
	}

	void poll() {
		if (!server->is_listening())
			return;
		if (connection.is_null()) {
			if (!server->is_connection_available())
				return;
			connection = server->take_connection();
			time = OS::get_singleton()->get_ticks_usec();
		}
		if (OS::get_singleton()->get_ticks_usec() - time > 1000000) {
			_clear_client();
			return;
		}
		if (connection->get_status() != StreamPeerTCP::STATUS_CONNECTED)
			return;

		while (true) {

			char *r = (char *)req_buf;
			int l = req_pos - 1;
			if (l > 3 && r[l] == '\n' && r[l - 1] == '\r' && r[l - 2] == '\n' && r[l - 3] == '\r') {
				_send_response();
				_clear_client();
				return;
			}

			int read = 0;
			ERR_FAIL_COND(req_pos >= 4096);
			Error err = connection->get_partial_data(&req_buf[req_pos], 1, read);
			if (err != OK) {
				// Got an error
				_clear_client();
				return;
			} else if (read != 1) {
				// Busy, wait next poll
				return;
			}
			req_pos += read;
		}
	}
};

class EditorExportPlatformJavaScript : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformJavaScript, EditorExportPlatform);

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;
	Ref<ImageTexture> stop_icon;
	int menu_options;

	enum ExportMode {
		EXPORT_MODE_NORMAL = 0,
		EXPORT_MODE_THREADS = 1,
		EXPORT_MODE_GDNATIVE = 2,
	};

	String _get_template_name(ExportMode p_mode, bool p_debug) const {
		String name = "webassembly";
		switch (p_mode) {
			case EXPORT_MODE_THREADS:
				name += "_threads";
				break;
			case EXPORT_MODE_GDNATIVE:
				name += "_gdnative";
				break;
			default:
				break;
		}
		if (p_debug) {
			name += "_debug.zip";
		} else {
			name += "_release.zip";
		}
		return name;
	}

	void _fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug, const Vector<SharedObject> p_shared_objects);

private:
	Ref<EditorHTTPServer> server;
	bool server_quit;
	Mutex *server_lock;
	Thread *server_thread;

	static void _server_thread_poll(void *data);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;
	virtual String get_os_name() const;
	virtual Ref<Texture> get_logo() const;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool poll_export();
	virtual int get_options_count() const;
	virtual String get_option_label(int p_index) const { return p_index ? TTR("Stop HTTP Server") : TTR("Run in Browser"); }
	virtual String get_option_tooltip(int p_index) const { return p_index ? TTR("Stop HTTP Server") : TTR("Run exported HTML in the system's default browser."); }
	virtual Ref<ImageTexture> get_option_icon(int p_index) const;
	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_option, int p_debug_flags);
	virtual Ref<Texture> get_run_icon() const;

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("web");
		r_features->push_back(get_os_name());
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) {
	}

	EditorExportPlatformJavaScript();
	~EditorExportPlatformJavaScript();
};

void EditorExportPlatformJavaScript::_fix_html(Vector<uint8_t> &p_html, const Ref<EditorExportPreset> &p_preset, const String &p_name, bool p_debug, const Vector<SharedObject> p_shared_objects) {

	String str_template = String::utf8(reinterpret_cast<const char *>(p_html.ptr()), p_html.size());
	String str_export;
	Vector<String> lines = str_template.split("\n");
	String libs;
	for (int i = 0; i < p_shared_objects.size(); i++) {
		libs += "\"" + p_shared_objects[i].path.get_file() + "\",";
	}

	for (int i = 0; i < lines.size(); i++) {

		String current_line = lines[i];
		current_line = current_line.replace("$GODOT_BASENAME", p_name);
		current_line = current_line.replace("$GODOT_PROJECT_NAME", ProjectSettings::get_singleton()->get_setting("application/config/name"));
		current_line = current_line.replace("$GODOT_HEAD_INCLUDE", p_preset->get("html/head_include"));
		current_line = current_line.replace("$GODOT_FULL_WINDOW", p_preset->get("html/full_window_size") ? "true" : "false");
		current_line = current_line.replace("$GODOT_GDNATIVE_LIBS", libs);
		current_line = current_line.replace("$GODOT_DEBUG_ENABLED", p_debug ? "true" : "false");
		str_export += current_line + "\n";
	}

	CharString cs = str_export.utf8();
	p_html.resize(cs.length());
	for (int i = 0; i < cs.length(); i++) {
		p_html.write[i] = cs[i];
	}
}

void EditorExportPlatformJavaScript::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	if (p_preset->get("vram_texture_compression/for_desktop")) {
		r_features->push_back("s3tc");
	}

	if (p_preset->get("vram_texture_compression/for_mobile")) {
		String driver = ProjectSettings::get_singleton()->get("rendering/quality/driver/driver_name");
		if (driver == "GLES2") {
			r_features->push_back("etc");
		} else if (driver == "GLES3") {
			r_features->push_back("etc2");
			if (ProjectSettings::get_singleton()->get("rendering/quality/driver/fallback_to_gles2")) {
				r_features->push_back("etc");
			}
		}
	}
	ExportMode mode = (ExportMode)(int)p_preset->get("variant/export_type");
	if (mode == EXPORT_MODE_THREADS) {
		r_features->push_back("threads");
	} else if (mode == EXPORT_MODE_GDNATIVE) {
		r_features->push_back("wasm32");
	}
}

void EditorExportPlatformJavaScript::get_export_options(List<ExportOption> *r_options) {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "variant/export_type", PROPERTY_HINT_ENUM, "Regular,Threads,GDNative"), 0)); // Export type.
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "vram_texture_compression/for_desktop"), true)); // S3TC
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "vram_texture_compression/for_mobile"), false)); // ETC or ETC2, depending on renderer
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/custom_html_shell", PROPERTY_HINT_FILE, "*.html"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "html/head_include", PROPERTY_HINT_MULTILINE_TEXT), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "html/full_window_size"), true));
}

String EditorExportPlatformJavaScript::get_name() const {

	return "HTML5";
}

String EditorExportPlatformJavaScript::get_os_name() const {

	return "HTML5";
}

Ref<Texture> EditorExportPlatformJavaScript::get_logo() const {

	return logo;
}

bool EditorExportPlatformJavaScript::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	String err;
	bool valid = false;
	ExportMode mode = (ExportMode)(int)p_preset->get("variant/export_type");

	// Look for export templates (first official, and if defined custom templates).
	bool dvalid = exists_export_template(_get_template_name(mode, true), &err);
	bool rvalid = exists_export_template(_get_template_name(mode, false), &err);

	if (p_preset->get("custom_template/debug") != "") {
		dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
		if (!dvalid) {
			err += TTR("Custom debug template not found.") + "\n";
		}
	}
	if (p_preset->get("custom_template/release") != "") {
		rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
		if (!rvalid) {
			err += TTR("Custom release template not found.") + "\n";
		}
	}

	valid = dvalid || rvalid;
	r_missing_templates = !valid;

	// Validate the rest of the configuration.

	if (p_preset->get("vram_texture_compression/for_mobile")) {
		String etc_error = test_etc2();
		if (etc_error != String()) {
			valid = false;
			err += etc_error;
		}
	}

	if (!err.empty())
		r_error = err;

	return valid;
}

List<String> EditorExportPlatformJavaScript::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {

	List<String> list;
	list.push_back("html");
	return list;
}

Error EditorExportPlatformJavaScript::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String custom_debug = p_preset->get("custom_template/debug");
	String custom_release = p_preset->get("custom_template/release");
	String custom_html = p_preset->get("html/custom_html_shell");

	String template_path = p_debug ? custom_debug : custom_release;

	template_path = template_path.strip_edges();

	if (template_path == String()) {

		ExportMode mode = (ExportMode)(int)p_preset->get("variant/export_type");
		template_path = find_export_template(_get_template_name(mode, p_debug));
	}

	if (!DirAccess::exists(p_path.get_base_dir())) {
		return ERR_FILE_BAD_PATH;
	}

	if (template_path != String() && !FileAccess::exists(template_path)) {
		EditorNode::get_singleton()->show_warning(TTR("Template file not found:") + "\n" + template_path);
		return ERR_FILE_NOT_FOUND;
	}

	Vector<SharedObject> shared_objects;
	String pck_path = p_path.get_basename() + ".pck";
	Error error = save_pack(p_preset, pck_path, &shared_objects);
	if (error != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + pck_path);
		return error;
	}
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	for (int i = 0; i < shared_objects.size(); i++) {
		String dst = p_path.get_base_dir().plus_file(shared_objects[i].path.get_file());
		error = da->copy(shared_objects[i].path, dst);
		if (error != OK) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + shared_objects[i].path.get_file());
			memdelete(da);
			return error;
		}
	}
	memdelete(da);

	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);
	unzFile pkg = unzOpen2(template_path.utf8().get_data(), &io);

	if (!pkg) {

		EditorNode::get_singleton()->show_warning(TTR("Could not open template for export:") + "\n" + template_path);
		return ERR_FILE_NOT_FOUND;
	}

	if (unzGoToFirstFile(pkg) != UNZ_OK) {
		EditorNode::get_singleton()->show_warning(TTR("Invalid export template:") + "\n" + template_path);
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
			_fix_html(data, p_preset, p_path.get_file().get_basename(), p_debug, shared_objects);
			file = p_path.get_file();

		} else if (file == "godot.js") {

			file = p_path.get_file().get_basename() + ".js";
		} else if (file == "godot.worker.js") {

			file = p_path.get_file().get_basename() + ".worker.js";

		} else if (file == "godot.side.wasm") {

			file = p_path.get_file().get_basename() + ".side.wasm";

		} else if (file == "godot.audio.worklet.js") {

			file = p_path.get_file().get_basename() + ".audio.worklet.js";

		} else if (file == "godot.wasm") {

			file = p_path.get_file().get_basename() + ".wasm";
		}

		String dst = p_path.get_base_dir().plus_file(file);
		FileAccess *f = FileAccess::open(dst, FileAccess::WRITE);
		if (!f) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + dst);
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
			EditorNode::get_singleton()->show_warning(TTR("Could not read custom HTML shell:") + "\n" + custom_html);
			return ERR_FILE_CANT_READ;
		}
		Vector<uint8_t> buf;
		buf.resize(f->get_len());
		f->get_buffer(buf.ptrw(), buf.size());
		memdelete(f);
		_fix_html(buf, p_preset, p_path.get_file().get_basename(), p_debug, shared_objects);

		f = FileAccess::open(p_path, FileAccess::WRITE);
		if (!f) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + p_path);
			return ERR_FILE_CANT_WRITE;
		}
		f->store_buffer(buf.ptr(), buf.size());
		memdelete(f);
	}

	Ref<Image> splash;
	const String splash_path = String(GLOBAL_GET("application/boot_splash/image")).strip_edges();
	if (!splash_path.empty()) {
		splash.instance();
		const Error err = splash->load(splash_path);
		if (err) {
			EditorNode::get_singleton()->show_warning(TTR("Could not read boot splash image file:") + "\n" + splash_path + "\n" + TTR("Using default boot splash image."));
			splash.unref();
		}
	}
	if (splash.is_null()) {
		splash = Ref<Image>(memnew(Image(boot_splash_png)));
	}
	const String splash_png_path = p_path.get_base_dir().plus_file(p_path.get_file().get_basename() + ".png");
	if (splash->save_png(splash_png_path) != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + splash_png_path);
		return ERR_FILE_CANT_WRITE;
	}

	// Save a favicon that can be accessed without waiting for the project to finish loading.
	// This way, the favicon can be displayed immediately when loading the page.
	Ref<Image> favicon;
	const String favicon_path = String(GLOBAL_GET("application/config/icon")).strip_edges();
	if (!favicon_path.empty()) {
		favicon.instance();
		const Error err = favicon->load(favicon_path);
		if (err) {
			favicon.unref();
		}
	}

	if (favicon.is_valid()) {
		const String favicon_png_path = p_path.get_base_dir().plus_file("favicon.png");
		if (favicon->save_png(favicon_png_path) != OK) {
			EditorNode::get_singleton()->show_warning(TTR("Could not write file:") + "\n" + favicon_png_path);
			return ERR_FILE_CANT_WRITE;
		}
	}

	return OK;
}

bool EditorExportPlatformJavaScript::poll_export() {

	Ref<EditorExportPreset> preset;

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {

		Ref<EditorExportPreset> ep = EditorExport::get_singleton()->get_export_preset(i);
		if (ep->is_runnable() && ep->get_platform() == this) {
			preset = ep;
			break;
		}
	}

	int prev = menu_options;
	menu_options = preset.is_valid();
	if (server->is_listening()) {
		if (menu_options == 0) {
			server_lock->lock();
			server->stop();
			server_lock->unlock();
		} else {
			menu_options += 1;
		}
	}
	return menu_options != prev;
}

Ref<ImageTexture> EditorExportPlatformJavaScript::get_option_icon(int p_index) const {
	return p_index == 1 ? stop_icon : EditorExportPlatform::get_option_icon(p_index);
}

int EditorExportPlatformJavaScript::get_options_count() const {

	return menu_options;
}

Error EditorExportPlatformJavaScript::run(const Ref<EditorExportPreset> &p_preset, int p_option, int p_debug_flags) {

	if (p_option == 1) {
		server_lock->lock();
		server->stop();
		server_lock->unlock();
		return OK;
	}

	const String basepath = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmp_js_export");
	Error err = export_project(p_preset, true, basepath + ".html", p_debug_flags);
	if (err != OK) {
		// Export generates several files, clean them up on failure.
		DirAccess::remove_file_or_error(basepath + ".html");
		DirAccess::remove_file_or_error(basepath + ".js");
		DirAccess::remove_file_or_error(basepath + ".worker.js");
		DirAccess::remove_file_or_error(basepath + ".audio.worklet.js");
		DirAccess::remove_file_or_error(basepath + ".pck");
		DirAccess::remove_file_or_error(basepath + ".png");
		DirAccess::remove_file_or_error(basepath + ".side.wasm");
		DirAccess::remove_file_or_error(basepath + ".wasm");
		DirAccess::remove_file_or_error(EditorSettings::get_singleton()->get_cache_dir().plus_file("favicon.png"));
		return err;
	}

	const uint16_t bind_port = EDITOR_GET("export/web/http_port");
	// Resolve host if needed.
	const String bind_host = EDITOR_GET("export/web/http_host");
	IP_Address bind_ip;
	if (bind_host.is_valid_ip_address()) {
		bind_ip = bind_host;
	} else {
		bind_ip = IP::get_singleton()->resolve_hostname(bind_host);
	}
	ERR_FAIL_COND_V_MSG(!bind_ip.is_valid(), ERR_INVALID_PARAMETER, "Invalid editor setting 'export/web/http_host': '" + bind_host + "'. Try using '127.0.0.1'.");

	// Restart server.
	server_lock->lock();
	server->stop();
	err = server->listen(bind_port, bind_ip);
	server_lock->unlock();
	ERR_FAIL_COND_V_MSG(err != OK, err, "Unable to start HTTP server.");

	OS::get_singleton()->shell_open(String("http://" + bind_host + ":" + itos(bind_port) + "/tmp_js_export.html"));
	// FIXME: Find out how to clean up export files after running the successfully
	// exported game. Might not be trivial.
	return OK;
}

Ref<Texture> EditorExportPlatformJavaScript::get_run_icon() const {

	return run_icon;
}

void EditorExportPlatformJavaScript::_server_thread_poll(void *data) {
	EditorExportPlatformJavaScript *ej = (EditorExportPlatformJavaScript *)data;
	while (!ej->server_quit) {
		OS::get_singleton()->delay_usec(1000);
		ej->server_lock->lock();
		ej->server->poll();
		ej->server_lock->unlock();
	}
}

EditorExportPlatformJavaScript::EditorExportPlatformJavaScript() {

	server.instance();
	server_quit = false;
	server_lock = Mutex::create();
	server_thread = Thread::create(_server_thread_poll, this);

	Ref<Image> img = memnew(Image(_javascript_logo));
	logo.instance();
	logo->create_from_image(img);

	img = Ref<Image>(memnew(Image(_javascript_run_icon)));
	run_icon.instance();
	run_icon->create_from_image(img);

	Ref<Theme> theme = EditorNode::get_singleton()->get_editor_theme();
	if (theme.is_valid())
		stop_icon = theme->get_icon("Stop", "EditorIcons");
	else
		stop_icon.instance();

	menu_options = 0;
}

EditorExportPlatformJavaScript::~EditorExportPlatformJavaScript() {
	server->stop();
	server_quit = true;
	Thread::wait_to_finish(server_thread);
	memdelete(server_lock);
	memdelete(server_thread);
}

void register_javascript_exporter() {

	EDITOR_DEF("export/web/http_host", "localhost");
	EDITOR_DEF("export/web/http_port", 8060);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "export/web/http_port", PROPERTY_HINT_RANGE, "1,65535,1"));

	Ref<EditorExportPlatformJavaScript> platform;
	platform.instance();
	EditorExport::get_singleton()->add_export_platform(platform);
}
