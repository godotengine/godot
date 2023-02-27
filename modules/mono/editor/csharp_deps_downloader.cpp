/**************************************************************************/
/*  csharp_deps_downloader.cpp                                            */
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

#include "csharp_deps_downloader.h"

#include "editor/editor_paths.h"
#include "editor/editor_settings.h"

#include "modules/mono/csharp_script.h"

void CSharpDepsDownloader::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			update_countdown -= get_process_delta_time();
			if (update_countdown > 0) {
				return;
			}
			update_countdown = 0.5;

			String status;
			int downloaded_bytes;
			int total_bytes;
			bool success = _humanize_http_status(http_request, &status, &downloaded_bytes, &total_bytes);

			if (downloaded_bytes >= 0) {
				if (total_bytes > 0) {
					_set_current_progress_value(float(downloaded_bytes) / total_bytes, status);
				} else {
					_set_current_progress_value(0, status);
				}
			} else {
				_set_current_progress_status(status);
			}

			if (!success) {
				set_process(false);
			}
		} break;
	}
}

void CSharpDepsDownloader::_set_current_progress_status(const String &p_status, bool p_error) {
	ERR_FAIL_COND(progress == nullptr);

	current_status = p_status;

	if (progress->step(current_status, current_progress)) {
		_cancel_download();
	}
}

void CSharpDepsDownloader::_set_current_progress_value(float p_value, const String &p_status) {
	ERR_FAIL_COND(progress == nullptr);

	current_progress = p_value * 100;
	current_status = p_status;

	if (progress->step(current_status, current_progress)) {
		_cancel_download();
	}
}

void CSharpDepsDownloader::_show_download_error_alert() {
	OS::get_singleton()->alert(vformat(TTR("Failed to download C# dependencies: %s"), current_status),
			TTR("Failed to download C# dependencies"));
}

void CSharpDepsDownloader::_cancel_download() {
	if (progress != nullptr) {
		memdelete(progress);
		progress = nullptr;
	}

	if (is_downloading) {
		http_request->cancel_request();
		is_downloading = false;
	}
}

void CSharpDepsDownloader::_start_download(const String &p_url) {
	if (is_downloading) {
		return;
	}

	is_downloading = true;
	current_progress = 0;

	progress = memnew(EditorProgress("csharp_deps_download", TTR("Downloading C# Dependencies"), 100, /* p_can_cancel: */ true));

	_set_current_progress_status(TTR("Starting the download..."));

	http_request->set_download_file(EditorPaths::get_singleton()->get_cache_dir().path_join("tmp_csharp_deps.zip"));
	http_request->set_use_threads(true);

	const String proxy_host = EDITOR_GET("network/http_proxy/host");
	const int proxy_port = EDITOR_GET("network/http_proxy/port");
	http_request->set_http_proxy(proxy_host, proxy_port);
	http_request->set_https_proxy(proxy_host, proxy_port);

	Error err = http_request->request(p_url);
	if (err != OK) {
		_set_current_progress_status(TTR("Error requesting URL:") + " " + p_url, true);

		memdelete(progress);
		progress = nullptr;

		_show_download_error_alert();

		return;
	}

	set_process(true);
	_set_current_progress_status(TTR("Connecting to the mirror..."));
}

bool CSharpDepsDownloader::_humanize_http_status(HTTPRequest *p_request, String *r_status, int *r_downloaded_bytes, int *r_total_bytes) {
	*r_status = "";
	*r_downloaded_bytes = -1;
	*r_total_bytes = -1;
	bool success = true;

	switch (p_request->get_http_client_status()) {
		case HTTPClient::STATUS_DISCONNECTED:
			*r_status = TTR("Disconnected");
			success = false;
			break;
		case HTTPClient::STATUS_RESOLVING:
			*r_status = TTR("Resolving");
			break;
		case HTTPClient::STATUS_CANT_RESOLVE:
			*r_status = TTR("Can't Resolve");
			success = false;
			break;
		case HTTPClient::STATUS_CONNECTING:
			*r_status = TTR("Connecting...");
			break;
		case HTTPClient::STATUS_CANT_CONNECT:
			*r_status = TTR("Can't Connect");
			success = false;
			break;
		case HTTPClient::STATUS_CONNECTED:
			*r_status = TTR("Connected");
			break;
		case HTTPClient::STATUS_REQUESTING:
			*r_status = TTR("Requesting...");
			break;
		case HTTPClient::STATUS_BODY:
			*r_status = TTR("Downloading");
			*r_downloaded_bytes = p_request->get_downloaded_bytes();
			*r_total_bytes = p_request->get_body_size();

			if (p_request->get_body_size() > 0) {
				*r_status += " " + String::humanize_size(p_request->get_downloaded_bytes()) + "/" + String::humanize_size(p_request->get_body_size());
			} else {
				*r_status += " " + String::humanize_size(p_request->get_downloaded_bytes());
			}
			break;
		case HTTPClient::STATUS_CONNECTION_ERROR:
			*r_status = TTR("Connection Error");
			success = false;
			break;
		case HTTPClient::STATUS_TLS_HANDSHAKE_ERROR:
			*r_status = TTR("TLS Handshake Error");
			success = false;
			break;
	}

	return success;
}

void CSharpDepsDownloader::_download_completed(int p_status, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_data) {
	bool success = false;

	switch (p_status) {
		case HTTPRequest::RESULT_CANT_RESOLVE: {
			_set_current_progress_status(TTR("Can't resolve the requested address."), true);
		} break;
		case HTTPRequest::RESULT_BODY_SIZE_LIMIT_EXCEEDED:
		case HTTPRequest::RESULT_CONNECTION_ERROR:
		case HTTPRequest::RESULT_CHUNKED_BODY_SIZE_MISMATCH:
		case HTTPRequest::RESULT_TLS_HANDSHAKE_ERROR:
		case HTTPRequest::RESULT_CANT_CONNECT: {
			_set_current_progress_status(TTR("Can't connect to the mirror."), true);
		} break;
		case HTTPRequest::RESULT_NO_RESPONSE: {
			_set_current_progress_status(TTR("No response from the mirror."), true);
		} break;
		case HTTPRequest::RESULT_REQUEST_FAILED: {
			_set_current_progress_status(TTR("Request failed."), true);
		} break;
		case HTTPRequest::RESULT_REDIRECT_LIMIT_REACHED: {
			_set_current_progress_status(TTR("Request ended up in a redirect loop."), true);
		} break;
		default: {
			if (p_code != 200) {
				_set_current_progress_status(TTR("Request failed:") + " " + itos(p_code), true);
			} else {
				success = true;

				_set_current_progress_status(TTR("Download complete; extracting templates..."));
				String path = http_request->get_download_file();

				is_downloading = false;
				bool ret = _install_file(path, true);
				if (ret) {
					// Clean up downloaded file.
					Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
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

	if (progress != nullptr) {
		memdelete(progress);
		progress = nullptr;
	}

	if (!success) {
		_show_download_error_alert();
	}

	set_process(false);
}

bool CSharpDepsDownloader::_install_file(const String &p_file, bool p_skip_progress) {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

	unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
	if (!pkg) {
		EditorNode::get_singleton()->show_warning(TTR("Can't open the C# dependencies file."));
		return false;
	}
	int ret = unzGoToFirstFile(pkg);

	// Count them and find version.
	int fc = 0;
	String contents_dir = "GodotSharp";

	while (ret == UNZ_OK) {
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
		if (ret != UNZ_OK) {
			break;
		}

		String file = String::utf8(fname);

		if (file.get_file().size() != 0) {
			fc++;
		}

		ret = unzGoToNextFile(pkg);
	}

	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String extract_dir = OS::get_singleton()->get_executable_path().get_base_dir().path_join("GodotSharp");
	Error err = d->make_dir_recursive(extract_dir);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error creating path for extracting templates:") + "\n" + extract_dir);
		unzClose(pkg);
		return false;
	}

	EditorProgress *p = nullptr;
	if (!p_skip_progress) {
		p = memnew(EditorProgress("csharp_deps_extract", TTR("Extracting C# Dependencies"), fc));
	}

	fc = 0;
	ret = unzGoToFirstFile(pkg);
	while (ret == UNZ_OK) {
		// Get filename.
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);
		if (ret != UNZ_OK) {
			break;
		}

		String file_path(String::utf8(fname));
		bool is_dir = file_path.ends_with("/");
		file_path = file_path.simplify_path();

		String file = file_path.get_file();

		if (file.size() == 0) {
			ret = unzGoToNextFile(pkg);
			continue;
		}

		Vector<uint8_t> uncomp_data;
		uncomp_data.resize(info.uncompressed_size);

		// Read
		unzOpenCurrentFile(pkg);
		ret = unzReadCurrentFile(pkg, uncomp_data.ptrw(), uncomp_data.size());
		ERR_BREAK_MSG(ret < 0, vformat("An error occurred while attempting to read from file: %s. This file will not be used.", file));
		unzCloseCurrentFile(pkg);

		if (file_path.trim_suffix("/") == contents_dir) {
			ret = unzGoToNextFile(pkg);
			continue;
		}

		String base_dir = file_path.get_base_dir().trim_suffix("/");

		if (base_dir != contents_dir && base_dir.begins_with(contents_dir)) {
			base_dir = base_dir.substr(contents_dir.length(), file_path.length()).trim_prefix("/");
			file = base_dir.path_join(file);

			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			ERR_CONTINUE(da.is_null());

			String output_dir = extract_dir.path_join(base_dir);

			if (!DirAccess::exists(output_dir)) {
				Error mkdir_err = da->make_dir_recursive(output_dir);
				ERR_CONTINUE(mkdir_err != OK);
			}
		}

		if (p) {
			p->step(TTR("Importing:") + " " + file, fc);
		}

		String to_write = extract_dir.path_join(file);

		if (is_dir) {
			Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			ERR_CONTINUE(da.is_null());

			if (!DirAccess::exists(to_write)) {
				Error mkdir_err = da->make_dir_recursive(to_write);
				ERR_CONTINUE(mkdir_err != OK);
			}
		} else {
			Ref<FileAccess> f = FileAccess::open(to_write, FileAccess::WRITE);

			if (f.is_null()) {
				ret = unzGoToNextFile(pkg);
				fc++;
				ERR_CONTINUE_MSG(true, "Can't open file from path '" + String(to_write) + "'.");
			}

			f->store_buffer(uncomp_data.ptr(), uncomp_data.size());
			f.unref(); // close file.
#ifndef WINDOWS_ENABLED
			FileAccess::set_unix_permissions(to_write, (info.external_fa >> 16) & 0x01FF);
#endif
		}

		ret = unzGoToNextFile(pkg);
		fc++;
	}

	if (p) {
		memdelete(p);
	}
	unzClose(pkg);

	_install_completed();

	return true;
}

void CSharpDepsDownloader::_install_completed() {
	callable_mp(CSharpLanguage::get_singleton(), &CSharpLanguage::ensure_dotnet_initialized)
			.call_deferred();
}

void CSharpDepsDownloader::download_deps(const String &p_url) {
	_start_download(p_url);
}

CSharpDepsDownloader::CSharpDepsDownloader() {
	http_request = memnew(HTTPRequest);
	add_child(http_request);
	http_request->connect("request_completed", callable_mp(this, &CSharpDepsDownloader::_download_completed), CONNECT_DEFERRED);
}
