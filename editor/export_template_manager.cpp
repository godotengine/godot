/*************************************************************************/
/*  export_template_manager.cpp                                          */
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
#include "export_template_manager.h"

#include "editor_node.h"
#include "editor_scale.h"
#include "io/json.h"
#include "io/zip_io.h"
#include "os/dir_access.h"
#include "version.h"

void ExportTemplateManager::_update_template_list() {

	while (current_hb->get_child_count()) {
		memdelete(current_hb->get_child(0));
	}

	while (installed_vb->get_child_count()) {
		memdelete(installed_vb->get_child(0));
	}

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(EditorSettings::get_singleton()->get_templates_dir());

	d->list_dir_begin();
	Set<String> templates;

	if (err == OK) {

		bool isdir;
		String c = d->get_next(&isdir);
		while (c != String()) {
			if (isdir && !c.begins_with(".")) {
				templates.insert(c);
			}
			c = d->get_next(&isdir);
		}
	}
	d->list_dir_end();

	memdelete(d);

	String current_version = itos(VERSION_MAJOR) + "." + itos(VERSION_MINOR) + "-" + VERSION_STATUS + VERSION_MODULE_CONFIG;

	Label *current = memnew(Label);
	current->set_h_size_flags(SIZE_EXPAND_FILL);
	current_hb->add_child(current);

	if (templates.has(current_version)) {
		current->add_color_override("font_color", get_color("success_color", "Editor"));
		Button *redownload = memnew(Button);
		redownload->set_text(TTR("Re-Download"));
		current_hb->add_child(redownload);
		redownload->connect("pressed", this, "_download_template", varray(current_version));

		Button *uninstall = memnew(Button);
		uninstall->set_text(TTR("Uninstall"));
		current_hb->add_child(uninstall);
		current->set_text(current_version + " " + TTR("(Installed)"));
		uninstall->connect("pressed", this, "_uninstall_template", varray(current_version));

	} else {
		current->add_color_override("font_color", get_color("error_color", "Editor"));
		Button *redownload = memnew(Button);
		redownload->set_text(TTR("Download"));
		redownload->connect("pressed", this, "_download_template", varray(current_version));
		current_hb->add_child(redownload);
		current->set_text(current_version + " " + TTR("(Missing)"));
	}

	for (Set<String>::Element *E = templates.back(); E; E = E->prev()) {

		HBoxContainer *hbc = memnew(HBoxContainer);
		Label *version = memnew(Label);
		version->set_modulate(get_color("disabled_font_color", "Editor"));
		String text = E->get();
		if (text == current_version) {
			text += " " + TTR("(Current)");
		}
		version->set_text(text);
		version->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(version);

		Button *uninstall = memnew(Button);

		uninstall->set_text(TTR("Uninstall"));
		hbc->add_child(uninstall);
		uninstall->connect("pressed", this, "_uninstall_template", varray(E->get()));

		installed_vb->add_child(hbc);
	}
}

void ExportTemplateManager::_download_template(const String &p_version) {

	print_line("download " + p_version);
	while (template_list->get_child_count()) {
		memdelete(template_list->get_child(0));
	}
	template_downloader->popup_centered_minsize();
	template_list_state->set_text(TTR("Retrieving mirrors, please wait.."));
	template_download_progress->set_max(100);
	template_download_progress->set_value(0);
	request_mirror->request("https://godotengine.org/mirrorlist/" + p_version + ".json");
	template_list_state->show();
	template_download_progress->show();
}

void ExportTemplateManager::_uninstall_template(const String &p_version) {

	remove_confirm->set_text(vformat(TTR("Remove template version '%s'?"), p_version));
	remove_confirm->popup_centered_minsize();
	to_remove = p_version;
}

void ExportTemplateManager::_uninstall_template_confirm() {

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(EditorSettings::get_singleton()->get_templates_dir());

	ERR_FAIL_COND(err != OK);

	err = d->change_dir(to_remove);

	ERR_FAIL_COND(err != OK);

	Vector<String> files;

	d->list_dir_begin();

	bool isdir;
	String c = d->get_next(&isdir);
	while (c != String()) {
		if (!isdir) {
			files.push_back(c);
		}
		c = d->get_next(&isdir);
	}

	d->list_dir_end();

	for (int i = 0; i < files.size(); i++) {
		d->remove(files[i]);
	}

	d->change_dir("..");
	d->remove(to_remove);

	_update_template_list();
}

void ExportTemplateManager::_install_from_file(const String &p_file, bool p_use_progress) {

	FileAccess *fa = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&fa);

	unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::get_singleton()->show_warning(TTR("Can't open export templates zip."));
		return;
	}
	int ret = unzGoToFirstFile(pkg);

	int fc = 0; //count them and find version
	String version;

	while (ret == UNZ_OK) {

		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

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

			if (data_str.get_slice_count("-") != 2 || data_str.get_slice_count(".") != 2) {
				EditorNode::get_singleton()->show_warning(TTR("Invalid version.txt format inside templates."));
				unzClose(pkg);
				return;
			}

			String ver = data_str.get_slice("-", 0);

			int major = ver.get_slice(".", 0).to_int();
			int minor = ver.get_slice(".", 1).to_int();
			String rev = data_str.get_slice("-", 1);

			if (!rev.is_valid_identifier()) {
				EditorNode::get_singleton()->show_warning(TTR("Invalid version.txt format inside templates. Revision is not a valid identifier."));
				unzClose(pkg);
				return;
			}

			version = itos(major) + "." + itos(minor) + "-" + rev;
		}

		fc++;
		ret = unzGoToNextFile(pkg);
	}

	if (version == String()) {
		EditorNode::get_singleton()->show_warning(TTR("No version.txt found inside templates."));
		unzClose(pkg);
		return;
	}

	String template_path = EditorSettings::get_singleton()->get_templates_dir().plus_file(version);

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->make_dir_recursive(template_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error creating path for templates:\n") + template_path);
		unzClose(pkg);
		return;
	}

	memdelete(d);

	ret = unzGoToFirstFile(pkg);

	EditorProgress *p = NULL;
	if (p_use_progress) {
		p = memnew(EditorProgress("ltask", TTR("Extracting Export Templates"), fc));
	}

	fc = 0;

	while (ret == UNZ_OK) {

		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = String(fname).get_file();

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptrw(), data.size());
		unzCloseCurrentFile(pkg);

		if (p) {
			p->step(TTR("Importing:") + " " + file, fc);
		}

		FileAccess *f = FileAccess::open(template_path.plus_file(file), FileAccess::WRITE);

		if (!f) {
			ret = unzGoToNextFile(pkg);
			fc++;
			ERR_CONTINUE(!f);
		}

		f->store_buffer(data.ptr(), data.size());

		memdelete(f);

		ret = unzGoToNextFile(pkg);
		fc++;
	}

	if (p) {
		memdelete(p);
	}

	unzClose(pkg);

	_update_template_list();
}

void ExportTemplateManager::popup_manager() {

	_update_template_list();
	popup_centered_minsize(Size2(400, 400) * EDSCALE);
}

void ExportTemplateManager::ok_pressed() {

	template_open->popup_centered_ratio();
}

void ExportTemplateManager::_http_download_mirror_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data) {

	if (p_status != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		EditorNode::get_singleton()->show_warning("Error getting the list of mirrors.");
		return;
	}

	String mirror_str;
	{
		PoolByteArray::Read r = p_data.read();
		mirror_str.parse_utf8((const char *)r.ptr(), p_data.size());
	}

	template_list_state->hide();
	template_download_progress->hide();

	Variant r;
	String errs;
	int errline;
	Error err = JSON::parse(mirror_str, r, errs, errline);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning("Error parsing JSON of mirror list. Please report this issue!");
		return;
	}

	bool mirrors_found = false;

	Dictionary d = r;
	print_line(r);
	if (d.has("mirrors")) {
		Array mirrors = d["mirrors"];
		for (int i = 0; i < mirrors.size(); i++) {
			Dictionary m = mirrors[i];
			ERR_CONTINUE(!m.has("url") || !m.has("name"));
			LinkButton *lb = memnew(LinkButton);
			lb->set_text(m["name"]);
			lb->connect("pressed", this, "_begin_template_download", varray(m["url"]));
			template_list->add_child(lb);
			mirrors_found = true;
		}
	}

	if (!mirrors_found) {
		EditorNode::get_singleton()->show_warning(TTR("No download links found for this version. Direct download is only available for official releases."));
		return;
	}
}
void ExportTemplateManager::_http_download_templates_completed(int p_status, int p_code, const PoolStringArray &headers, const PoolByteArray &p_data) {

	switch (p_status) {

		case HTTPRequest::RESULT_CANT_RESOLVE: {
			template_list_state->set_text(TTR("Can't resolve."));
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH: {
			template_list_state->set_text(TTR("Can't connect."));
		} break;
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
				String path = EditorSettings::get_singleton()->get_cache_dir().plus_file("tmp_templates.tpz");
				FileAccess *f = FileAccess::open(path, FileAccess::WRITE);
				if (!f) {
					template_list_state->set_text(TTR("Can't write file."));
				} else {
					int size = p_data.size();
					PoolVector<uint8_t>::Read r = p_data.read();
					f->store_buffer(r.ptr(), size);
					memdelete(f);
					template_list_state->set_text(TTR("Download Complete."));
					template_downloader->hide();
					_install_from_file(path, false);
				}
			}
		} break;
	}

	set_process(false);
}

void ExportTemplateManager::_begin_template_download(const String &p_url) {

	for (int i = 0; i < template_list->get_child_count(); i++) {
		BaseButton *b = Object::cast_to<BaseButton>(template_list->get_child(0));
		if (b) {
			b->set_disabled(true);
		}
	}

	download_data.clear();

	Error err = download_templates->request(p_url);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error requesting url: ") + p_url);
		return;
	}

	set_process(true);

	template_list_state->show();
	template_download_progress->set_max(100);
	template_download_progress->set_value(0);
	template_download_progress->show();
	template_list_state->set_text(TTR("Connecting to Mirror.."));
}

void ExportTemplateManager::_notification(int p_what) {

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
			case HTTPClient::STATUS_RESOLVING: status = TTR("Resolving"); break;
			case HTTPClient::STATUS_CANT_RESOLVE:
				status = TTR("Can't Resolve");
				errored = true;
				break;
			case HTTPClient::STATUS_CONNECTING: status = TTR("Connecting.."); break;
			case HTTPClient::STATUS_CANT_CONNECT:
				status = TTR("Can't Connect");
				errored = true;
				break;
			case HTTPClient::STATUS_CONNECTED: status = TTR("Connected"); break;
			case HTTPClient::STATUS_REQUESTING: status = TTR("Requesting.."); break;
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
			;
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (!is_visible_in_tree()) {
			print_line("closed");
			download_templates->cancel_request();
			set_process(false);
		}
	}
}

void ExportTemplateManager::_bind_methods() {

	ClassDB::bind_method("_download_template", &ExportTemplateManager::_download_template);
	ClassDB::bind_method("_uninstall_template", &ExportTemplateManager::_uninstall_template);
	ClassDB::bind_method("_uninstall_template_confirm", &ExportTemplateManager::_uninstall_template_confirm);
	ClassDB::bind_method("_install_from_file", &ExportTemplateManager::_install_from_file);
	ClassDB::bind_method("_http_download_mirror_completed", &ExportTemplateManager::_http_download_mirror_completed);
	ClassDB::bind_method("_http_download_templates_completed", &ExportTemplateManager::_http_download_templates_completed);
	ClassDB::bind_method("_begin_template_download", &ExportTemplateManager::_begin_template_download);
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
	installed_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	get_cancel()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Install From File"));

	remove_confirm = memnew(ConfirmationDialog);
	remove_confirm->set_title(TTR("Remove Template"));
	add_child(remove_confirm);
	remove_confirm->connect("confirmed", this, "_uninstall_template_confirm");

	template_open = memnew(FileDialog);
	template_open->set_title(TTR("Select template file"));
	template_open->add_filter("*.tpz ; Godot Export Templates");
	template_open->set_access(FileDialog::ACCESS_FILESYSTEM);
	template_open->set_mode(FileDialog::MODE_OPEN_FILE);
	template_open->connect("file_selected", this, "_install_from_file", varray(true));
	add_child(template_open);

	set_title(TTR("Export Template Manager"));
	set_hide_on_ok(false);

	request_mirror = memnew(HTTPRequest);
	add_child(request_mirror);
	request_mirror->connect("request_completed", this, "_http_download_mirror_completed");

	download_templates = memnew(HTTPRequest);
	add_child(download_templates);
	download_templates->connect("request_completed", this, "_http_download_templates_completed");

	template_downloader = memnew(AcceptDialog);
	template_downloader->set_title(TTR("Download Templates"));
	template_downloader->get_ok()->set_text(TTR("Close"));
	add_child(template_downloader);

	VBoxContainer *vbc = memnew(VBoxContainer);
	template_downloader->add_child(vbc);
	ScrollContainer *sc = memnew(ScrollContainer);
	sc->set_custom_minimum_size(Size2(400, 200) * EDSCALE);
	vbc->add_margin_child(TTR("Select mirror from list: "), sc);
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
