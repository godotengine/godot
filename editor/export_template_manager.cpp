/*************************************************************************/
/*  export_template_manager.cpp                                          */
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

#include "export_template_manager.h"

#include "core/input/input.h"
#include "core/io/json.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/keyboard.h"
#include "core/version.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "progress_dialog.h"
#include "scene/gui/link_button.h"

void ExportTemplateManager::_update_template_list() {
	while (current_hb->get_child_count()) {
		memdelete(current_hb->get_child(0));
	}

	while (installed_vb->get_child_count()) {
		memdelete(installed_vb->get_child(0));
	}

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(EditorSettings::get_singleton()->get_templates_dir());

	Set<String> templates;
	d->list_dir_begin();
	if (err == OK) {
		String c = d->get_next();
		while (c != String()) {
			if (d->current_is_dir() && !c.begins_with(".")) {
				templates.insert(c);
			}
			c = d->get_next();
		}
	}
	d->list_dir_end();

	memdelete(d);

	String current_version = VERSION_FULL_CONFIG;
	// Downloadable export templates are only available for stable and official alpha/beta/RC builds
	// (which always have a number following their status, e.g. "alpha1").
	// Therefore, don't display download-related features when using a development version
	// (whose builds aren't numbered).
	const bool downloads_available =
			String(VERSION_STATUS) != String("dev") &&
			String(VERSION_STATUS) != String("alpha") &&
			String(VERSION_STATUS) != String("beta") &&
			String(VERSION_STATUS) != String("rc");

	Label *current = memnew(Label);
	current->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	current_hb->add_child(current);

	if (templates.has(current_version)) {
		current->add_theme_color_override("font_color", current->get_theme_color("success_color", "Editor"));

		// Only display a redownload button if it can be downloaded in the first place
		if (downloads_available) {
			Button *redownload = memnew(Button);
			redownload->set_text(TTR("Redownload"));
			current_hb->add_child(redownload);
			redownload->connect("pressed", callable_mp(this, &ExportTemplateManager::_download_template), varray(current_version));
		}

		Button *uninstall = memnew(Button);
		uninstall->set_text(TTR("Uninstall"));
		current_hb->add_child(uninstall);
		current->set_text(current_version + " " + TTR("(Installed)"));
		uninstall->connect("pressed", callable_mp(this, &ExportTemplateManager::_uninstall_template), varray(current_version));

	} else {
		current->add_theme_color_override("font_color", current->get_theme_color("error_color", "Editor"));
		Button *redownload = memnew(Button);
		redownload->set_text(TTR("Download"));

		if (!downloads_available) {
			redownload->set_disabled(true);
			redownload->set_tooltip(TTR("Official export templates aren't available for development builds."));
		}

		redownload->connect("pressed", callable_mp(this, &ExportTemplateManager::_download_template), varray(current_version));
		current_hb->add_child(redownload);
		current->set_text(current_version + " " + TTR("(Missing)"));
	}

	for (Set<String>::Element *E = templates.back(); E; E = E->prev()) {
		HBoxContainer *hbc = memnew(HBoxContainer);
		Label *version = memnew(Label);
		version->set_modulate(current->get_theme_color("disabled_font_color", "Editor"));
		String text = E->get();
		if (text == current_version) {
			text += " " + TTR("(Current)");
		}
		version->set_text(text);
		version->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		hbc->add_child(version);

		Button *uninstall = memnew(Button);

		uninstall->set_text(TTR("Uninstall"));
		hbc->add_child(uninstall);
		uninstall->connect("pressed", callable_mp(this, &ExportTemplateManager::_uninstall_template), varray(E->get()));

		installed_vb->add_child(hbc);
	}
}

void ExportTemplateManager::_download_template(const String &p_version) {
	while (template_list->get_child_count()) {
		memdelete(template_list->get_child(0));
	}
	template_downloader->popup_centered();
	template_list_state->set_text(TTR("Retrieving mirrors, please wait..."));
	template_download_progress->set_max(100);
	template_download_progress->set_value(0);
	request_mirror->request("https://godotengine.org/mirrorlist/" + p_version + ".json");
	template_list_state->show();
	template_download_progress->show();
}

void ExportTemplateManager::_uninstall_template(const String &p_version) {
	remove_confirm->set_text(vformat(TTR("Remove template version '%s'?"), p_version));
	remove_confirm->popup_centered();
	to_remove = p_version;
}

void ExportTemplateManager::_uninstall_template_confirm() {
	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	const String &templates_dir = EditorSettings::get_singleton()->get_templates_dir();
	Error err = da->change_dir(templates_dir);
	ERR_FAIL_COND_MSG(err != OK, "Could not access templates directory at '" + templates_dir + "'.");
	err = da->change_dir(to_remove);
	ERR_FAIL_COND_MSG(err != OK, "Could not access templates directory at '" + templates_dir.plus_file(to_remove) + "'.");

	err = da->erase_contents_recursive();
	ERR_FAIL_COND_MSG(err != OK, "Could not remove all templates in '" + templates_dir.plus_file(to_remove) + "'.");

	da->change_dir("..");
	err = da->remove(to_remove);
	ERR_FAIL_COND_MSG(err != OK, "Could not remove templates directory at '" + templates_dir.plus_file(to_remove) + "'.");

	_update_template_list();
}

bool ExportTemplateManager::_install_from_file(const String &p_file, bool p_use_progress) {
	// unzClose() will take care of closing the file stored in the unzFile,
	// so we don't need to `memdelete(fa)` in this method.
	FileAccess *fa = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&fa);

	unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
	if (!pkg) {
		EditorNode::get_singleton()->show_warning(TTR("Can't open export templates zip."));
		return false;
	}
	int ret = unzGoToFirstFile(pkg);

	int fc = 0; //count them and find version
	String version;
	String contents_dir;

	while (ret == UNZ_OK) {
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String file = fname;

		if (file.ends_with("version.txt")) {
			Vector<uint8_t> data;
			data.resize(info.uncompressed_size);

			//read
			unzOpenCurrentFile(pkg);
			ret = unzReadCurrentFile(pkg, data.ptrw(), data.size());
			unzCloseCurrentFile(pkg);

			String data_str;
			data_str.parse_utf8((const char *)data.ptr(), data.size());
			data_str = data_str.strip_edges();

			// Version number should be of the form major.minor[.patch].status[.module_config]
			// so it can in theory have 3 or more slices.
			if (data_str.get_slice_count(".") < 3) {
				EditorNode::get_singleton()->show_warning(vformat(TTR("Invalid version.txt format inside templates: %s."), data_str));
				unzClose(pkg);
				return false;
			}

			version = data_str;
			contents_dir = file.get_base_dir().trim_suffix("/").trim_suffix("\\");
		}

		if (file.get_file().size() != 0) {
			fc++;
		}

		ret = unzGoToNextFile(pkg);
	}

	if (version == String()) {
		EditorNode::get_singleton()->show_warning(TTR("No version.txt found inside templates."));
		unzClose(pkg);
		return false;
	}

	String template_path = EditorSettings::get_singleton()->get_templates_dir().plus_file(version);

	DirAccessRef d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->make_dir_recursive(template_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error creating path for templates:") + "\n" + template_path);
		unzClose(pkg);
		return false;
	}

	ret = unzGoToFirstFile(pkg);

	EditorProgress *p = nullptr;
	if (p_use_progress) {
		p = memnew(EditorProgress("ltask", TTR("Extracting Export Templates"), fc));
	}

	fc = 0;

	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String file_path(String(fname).simplify_path());

		String file = file_path.get_file();

		if (file.size() == 0) {
			ret = unzGoToNextFile(pkg);
			continue;
		}

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptrw(), data.size());
		unzCloseCurrentFile(pkg);

		String base_dir = file_path.get_base_dir().trim_suffix("/");

		if (base_dir != contents_dir && base_dir.begins_with(contents_dir)) {
			base_dir = base_dir.substr(contents_dir.length(), file_path.length()).trim_prefix("/");
			file = base_dir.plus_file(file);

			DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			ERR_CONTINUE(!da);

			String output_dir = template_path.plus_file(base_dir);

			if (!DirAccess::exists(output_dir)) {
				Error mkdir_err = da->make_dir_recursive(output_dir);
				ERR_CONTINUE(mkdir_err != OK);
			}
		}

		if (p) {
			p->step(TTR("Importing:") + " " + file, fc);
		}

		String to_write = template_path.plus_file(file);
		FileAccessRef f = FileAccess::open(to_write, FileAccess::WRITE);

		if (!f) {
			ret = unzGoToNextFile(pkg);
			fc++;
			ERR_CONTINUE_MSG(true, "Can't open file from path '" + String(to_write) + "'.");
		}

		f->store_buffer(data.ptr(), data.size());

#ifndef WINDOWS_ENABLED
		FileAccess::set_unix_permissions(to_write, (info.external_fa >> 16) & 0x01FF);
#endif

		ret = unzGoToNextFile(pkg);
		fc++;
	}

	if (p) {
		memdelete(p);
	}

	unzClose(pkg);

	_update_template_list();
	return true;
}

void ExportTemplateManager::popup_manager() {
	_update_template_list();
	popup_centered(Size2(400, 400) * EDSCALE);
}

void ExportTemplateManager::ok_pressed() {
	template_open->popup_file_dialog();
}

void ExportTemplateManager::_http_download_mirror_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data) {
	if (p_status != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		EditorNode::get_singleton()->show_warning(TTR("Error getting the list of mirrors."));
		return;
	}

	String mirror_str;
	{
		const uint8_t *r = p_data.ptr();
		mirror_str.parse_utf8((const char *)r, p_data.size());
	}

	template_list_state->hide();
	template_download_progress->hide();

	Variant r;
	String errs;
	int errline;
	Error err = JSON::parse(mirror_str, r, errs, errline);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error parsing JSON of mirror list. Please report this issue!"));
		return;
	}

	bool mirrors_found = false;

	Dictionary d = r;
	if (d.has("mirrors")) {
		Array mirrors = d["mirrors"];
		for (int i = 0; i < mirrors.size(); i++) {
			Dictionary m = mirrors[i];
			ERR_CONTINUE(!m.has("url") || !m.has("name"));
			LinkButton *lb = memnew(LinkButton);
			lb->set_text(m["name"]);
			lb->connect("pressed", callable_mp(this, &ExportTemplateManager::_begin_template_download), varray(m["url"]));
			template_list->add_child(lb);
			mirrors_found = true;
		}
	}

	if (!mirrors_found) {
		EditorNode::get_singleton()->show_warning(TTR("No download links found for this version. Direct download is only available for official releases."));
		return;
	}
}

void ExportTemplateManager::_http_download_templates_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data) {
	switch (p_status) {
		case HTTPRequest::RESULT_CANT_RESOLVE: {
			template_list_state->set_text(TTR("Can't resolve."));
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH:
		case HTTPRequest::RESULT_SSL_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			template_list_state->set_text(TTR("Can't connect."));
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			template_list_state->set_text(TTR("No response."));
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			template_list_state->set_text(TTR("Request Failed."));
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			template_list_state->set_text(TTR("Redirect Loop."));
		} break;
		default: {
			if (p_code != 200) {
				template_list_state->set_text(TTR("Failed:") + " " + itos(p_code));
			} else {
				String path = download_templates->get_download_file();
				template_list_state->set_text(TTR("Download Complete."));
				template_downloader->hide();
				bool ret = _install_from_file(path, false);
				if (ret) {
					// Clean up downloaded file.
					DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
					Error err = da->remove(path);
					if (err != OK) {
						EditorNode::get_singleton()->add_io_error(TTR("Cannot remove temporary file:") + "\n" + path + "\n");
					}
				} else {
					EditorNode::get_singleton()->add_io_error(vformat(TTR("Templates installation failed.\nThe problematic templates archives can be found at '%s'."), path));
				}
			}
		} break;
	}

	set_process(false);
}

void ExportTemplateManager::_begin_template_download(const String &p_url) {
	if (Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
		OS::get_singleton()->shell_open(p_url);
		return;
	}

	for (int i = 0; i < template_list->get_child_count(); i++) {
		BaseButton *b = Object::cast_to<BaseButton>(template_list->get_child(0));
		if (b) {
			b->set_disabled(true);
		}
	}

	download_data.clear();
	download_templates->set_download_file(EditorSettings::get_singleton()->get_cache_dir().plus_file("tmp_templates.tpz"));
	download_templates->set_use_threads(true);

	Error err = download_templates->request(p_url);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error requesting URL:") + " " + p_url);
		return;
	}

	set_process(true);

	template_list_state->show();
	template_download_progress->set_max(100);
	template_download_progress->set_value(0);
	template_download_progress->show();
	template_list_state->set_text(TTR("Connecting to Mirror..."));
}

void ExportTemplateManager::_window_template_downloader_closed() {
	download_templates->cancel_request();
}

void ExportTemplateManager::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible()) {
			set_process(false);
		}
	}
	if (p_what == NOTIFICATION_PROCESS) {
		update_countdown -= get_process_delta_time();

		if (update_countdown > 0) {
			return;
		}
		update_countdown = 0.5;
		String status;
		bool errored = false;

		switch (download_templates->get_http_client_status()) {
			case HTTPClient::STATUS_DISCONNECTED:
				status = TTR("Disconnected");
				errored = true;
				break;
			case HTTPClient::STATUS_RESOLVING:
				status = TTR("Resolving");
				break;
			case HTTPClient::STATUS_CANT_RESOLVE:
				status = TTR("Can't Resolve");
				errored = true;
				break;
			case HTTPClient::STATUS_CONNECTING:
				status = TTR("Connecting...");
				break;
			case HTTPClient::STATUS_CANT_CONNECT:
				status = TTR("Can't Connect");
				errored = true;
				break;
			case HTTPClient::STATUS_CONNECTED:
				status = TTR("Connected");
				break;
			case HTTPClient::STATUS_REQUESTING:
				status = TTR("Requesting...");
				break;
			case HTTPClient::STATUS_BODY:
				status = TTR("Downloading");
				if (download_templates->get_body_size() > 0) {
					status += " " + String::humanize_size(download_templates->get_downloaded_bytes()) + "/" + String::humanize_size(download_templates->get_body_size());
					template_download_progress->set_max(download_templates->get_body_size());
					template_download_progress->set_value(download_templates->get_downloaded_bytes());
				} else {
					status += " " + String::humanize_size(download_templates->get_downloaded_bytes());
				}
				break;
			case HTTPClient::STATUS_CONNECTION_ERROR:
				status = TTR("Connection Error");
				errored = true;
				break;
			case HTTPClient::STATUS_SSL_HANDSHAKE_ERROR:
				status = TTR("SSL Handshake Error");
				errored = true;
				break;
		}

		template_list_state->set_text(status);
		if (errored) {
			set_process(false);
		}
	}
}

bool ExportTemplateManager::can_install_android_template() {
	const String templates_dir = EditorSettings::get_singleton()->get_templates_dir().plus_file(VERSION_FULL_CONFIG);
	return FileAccess::exists(templates_dir.plus_file("android_source.zip"));
}

Error ExportTemplateManager::install_android_template() {
	// To support custom Android builds, we install the Java source code and buildsystem
	// from android_source.zip to the project's res://android folder.

	DirAccessRef da = DirAccess::open("res://");
	ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);

	// Make res://android dir (if it does not exist).
	da->make_dir("android");
	{
		// Add version, to ensure building won't work if template and Godot version don't match.
		FileAccessRef f = FileAccess::open("res://android/.build_version", FileAccess::WRITE);
		ERR_FAIL_COND_V(!f, ERR_CANT_CREATE);
		f->store_line(VERSION_FULL_CONFIG);
		f->close();
	}

	// Create the android plugins directory.
	Error err = da->make_dir_recursive("android/plugins");
	ERR_FAIL_COND_V(err != OK, err);

	err = da->make_dir_recursive("android/build");
	ERR_FAIL_COND_V(err != OK, err);
	{
		// Add an empty .gdignore file to avoid scan.
		FileAccessRef f = FileAccess::open("res://android/build/.gdignore", FileAccess::WRITE);
		ERR_FAIL_COND_V(!f, ERR_CANT_CREATE);
		f->store_line("");
		f->close();
	}

	// Uncompress source template.

	const String &templates_path = EditorSettings::get_singleton()->get_templates_dir().plus_file(VERSION_FULL_CONFIG);
	const String &source_zip = templates_path.plus_file("android_source.zip");
	ERR_FAIL_COND_V(!FileAccess::exists(source_zip), ERR_CANT_OPEN);

	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	unzFile pkg = unzOpen2(source_zip.utf8().get_data(), &io);
	ERR_FAIL_COND_V_MSG(!pkg, ERR_CANT_OPEN, "Android sources not in ZIP format.");

	int ret = unzGoToFirstFile(pkg);
	int total_files = 0;
	// Count files to unzip.
	while (ret == UNZ_OK) {
		total_files++;
		ret = unzGoToNextFile(pkg);
	}
	ret = unzGoToFirstFile(pkg);

	ProgressDialog::get_singleton()->add_task("uncompress_src", TTR("Uncompressing Android Build Sources"), total_files);

	Set<String> dirs_tested;
	int idx = 0;
	while (ret == UNZ_OK) {
		// Get file path.
		unz_file_info info;
		char fpath[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fpath, 16384, nullptr, 0, nullptr, 0);

		String path = fpath;
		String base_dir = path.get_base_dir();

		if (!path.ends_with("/")) {
			Vector<uint8_t> data;
			data.resize(info.uncompressed_size);

			// Read.
			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, data.ptrw(), data.size());
			unzCloseCurrentFile(pkg);

			if (!dirs_tested.has(base_dir)) {
				da->make_dir_recursive(String("android/build").plus_file(base_dir));
				dirs_tested.insert(base_dir);
			}

			String to_write = String("res://android/build").plus_file(path);
			FileAccess *f = FileAccess::open(to_write, FileAccess::WRITE);
			if (f) {
				f->store_buffer(data.ptr(), data.size());
				memdelete(f);
#ifndef WINDOWS_ENABLED
				FileAccess::set_unix_permissions(to_write, (info.external_fa >> 16) & 0x01FF);
#endif
			} else {
				ERR_PRINT("Can't uncompress file: " + to_write);
			}
		}

		ProgressDialog::get_singleton()->task_step("uncompress_src", path, idx);

		idx++;
		ret = unzGoToNextFile(pkg);
	}

	ProgressDialog::get_singleton()->end_task("uncompress_src");
	unzClose(pkg);

	return OK;
}

void ExportTemplateManager::_bind_methods() {
}

ExportTemplateManager::ExportTemplateManager() {
	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	current_hb = memnew(HBoxContainer);
	main_vb->add_margin_child(TTR("Current Version:"), current_hb, false);

	installed_scroll = memnew(ScrollContainer);
	main_vb->add_margin_child(TTR("Installed Versions:"), installed_scroll, true);

	installed_vb = memnew(VBoxContainer);
	installed_scroll->add_child(installed_vb);
	installed_scroll->set_enable_v_scroll(true);
	installed_scroll->set_enable_h_scroll(false);
	installed_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	get_cancel()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Install From File"));

	remove_confirm = memnew(ConfirmationDialog);
	remove_confirm->set_title(TTR("Remove Template"));
	add_child(remove_confirm);
	remove_confirm->connect("confirmed", callable_mp(this, &ExportTemplateManager::_uninstall_template_confirm));

	template_open = memnew(FileDialog);
	template_open->set_title(TTR("Select Template File"));
	template_open->add_filter("*.tpz ; " + TTR("Godot Export Templates"));
	template_open->set_access(FileDialog::ACCESS_FILESYSTEM);
	template_open->set_file_mode(FileDialog::FILE_MODE_OPEN_FILE);
	template_open->connect("file_selected", callable_mp(this, &ExportTemplateManager::_install_from_file), varray(true));
	add_child(template_open);

	set_title(TTR("Export Template Manager"));
	set_hide_on_ok(false);

	request_mirror = memnew(HTTPRequest);
	add_child(request_mirror);
	request_mirror->connect("request_completed", callable_mp(this, &ExportTemplateManager::_http_download_mirror_completed));

	download_templates = memnew(HTTPRequest);
	add_child(download_templates);
	download_templates->connect("request_completed", callable_mp(this, &ExportTemplateManager::_http_download_templates_completed));

	template_downloader = memnew(AcceptDialog);
	template_downloader->set_title(TTR("Download Templates"));
	template_downloader->get_ok()->set_text(TTR("Close"));
	template_downloader->set_exclusive(true);
	add_child(template_downloader);
	template_downloader->connect("cancelled", callable_mp(this, &ExportTemplateManager::_window_template_downloader_closed));

	VBoxContainer *vbc = memnew(VBoxContainer);
	template_downloader->add_child(vbc);
	ScrollContainer *sc = memnew(ScrollContainer);
	sc->set_custom_minimum_size(Size2(400, 200) * EDSCALE);
	vbc->add_margin_child(TTR("Select mirror from list: (Shift+Click: Open in Browser)"), sc);
	template_list = memnew(VBoxContainer);
	sc->add_child(template_list);
	sc->set_enable_v_scroll(true);
	sc->set_enable_h_scroll(false);
	template_list_state = memnew(Label);
	vbc->add_child(template_list_state);
	template_download_progress = memnew(ProgressBar);
	vbc->add_child(template_download_progress);

	update_countdown = 0;
}
