/**************************************************************************/
/*  android_sdk_manager.cpp                                               */
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

#include "android_sdk_manager.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/label.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/rich_text_label.h"

#include "modules/zip/zip_reader.h"

static const char *ANDROID_CLI_URL_LINUX = "https://dl.google.com/android/cli/latest/linux_x86_64/android";
static const char *ANDROID_CLI_URL_MAC_X86 = "https://dl.google.com/android/cli/latest/darwin_x86_64/android";
static const char *ANDROID_CLI_URL_MAC_ARM = "https://dl.google.com/android/cli/latest/darwin_arm64/android";
static const char *ANDROID_CLI_URL_WIN = "https://dl.google.com/android/cli/latest/windows_x86_64/android.exe";

// Android SDK packages.
static const char *ANDROID_SDK_PACKAGES[] = {
	"platform-tools",
	"build-tools/36.1.0", // Should match the value in 'platform/android/java/app/config.gradle#buildTools'.
	"platforms/android-36",
	"ndk/29.0.14206865", // Should match the value in 'platform/android/java/app/config.gradle#ndkVersion'.
	"cmake/3.22.1",
	"cmdline-tools/latest",
	nullptr
};

static const uint64_t INSTALL_ANDROID_SDK_POLL_TIME = 3000000;

void AndroidSDKManager::_bind_methods() {
	ADD_SIGNAL(MethodInfo("java_sdk_installed"));
	ADD_SIGNAL(MethodInfo("android_sdk_installed"));
}

void AndroidSDKManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			connect(SNAME("canceled"), callable_mp(this, &AndroidSDKManager::_cancel_setup));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			disconnect("canceled", callable_mp(this, &AndroidSDKManager::_cancel_setup));
		} break;

		case NOTIFICATION_PROCESS: {
			switch (current_setup_status) {
				case DOWNLOADING_ANDROID_SDK:
				case DOWNLOADING_JAVA_SDK: {
					int downloaded_bytes = downloader->get_downloaded_bytes();
					if (downloaded_bytes > 0) {
						setup_progress_bar->set_indeterminate(false);
						setup_progress_bar->set_max(downloader->get_body_size());
						setup_progress_bar->set_value(downloaded_bytes);
					}
				} break;

				case INSTALLING_ANDROID_SDK: {
					if (!setup_process_data.is_empty()) {
						int setup_android_pid = setup_process_data["pid"];
						if (OS::get_singleton()->is_process_running(setup_android_pid)) {
							Ref<FileAccess> fa_out = setup_process_data["stdio"];
							Ref<FileAccess> fa_err = setup_process_data["stderr"];
							if (fa_out->is_open() && fa_err->is_open() && OS::get_singleton()->get_ticks_usec() - install_android_sdk_time > INSTALL_ANDROID_SDK_POLL_TIME) {
								install_android_sdk_time = OS::get_singleton()->get_ticks_usec();
								PackedByteArray buf;

								String output;
								buf.resize(fa_out->get_length());
								uint64_t buf_size = fa_out->get_buffer(buf.ptrw(), buf.size());
								output.append_utf8((const char *)buf.ptr(), buf_size);

								String err_output;
								buf.resize(fa_err->get_length());
								buf_size = fa_err->get_buffer(buf.ptrw(), buf.size());
								err_output.append_utf8((const char *)buf.ptr(), buf_size);

								if (!output.is_empty()) {
									print_verbose(output);
									execute_outputs->add_text(output);
								}

								if (!err_output.is_empty()) {
									print_verbose(err_output);
									execute_outputs->add_text(err_output);
								}
							}
						} else {
							int exit_code = OS::get_singleton()->get_process_exit_code(setup_android_pid);
							print_verbose("Exit code: " + itos(exit_code));
							execute_outputs->add_text("\nExit code: " + itos(exit_code));
							_android_sdk_installed(exit_code);
						}
					}
				} break;

				default: {
				} break;
			}
		} break;
	}
}

void AndroidSDKManager::_cancel_setup() {
	switch (current_setup_status) {
		case IDLE: {
		} break;

		case PROMPTING_ANDROID_ONLINE_ACCESS: {
			print_verbose("Canceling online mode request prompt for Android SDK setup.");
		} break;

		case PROMPTING_ANDROID_SDK_SETUP: {
			print_verbose("Canceling Android SDK setup prompt.");
		} break;

		case PROMPTING_JAVA_ONLINE_ACCESS: {
			print_verbose("Canceling online mode request prompt for Java SDK setup.");
		} break;

		case PROMPTING_JAVA_SDK_SETUP: {
			print_verbose("Canceling Java SDK setup prompt.");
		} break;

		case DOWNLOADING_ANDROID_SDK: {
			print_verbose("Canceling Android SDK download.");
			downloader->cancel_request();
		} break;

		case DOWNLOADING_JAVA_SDK: {
			print_verbose("Canceling Java SDK download.");
			downloader->cancel_request();
		} break;

		case INSTALLING_ANDROID_SDK: {
			print_verbose("Canceling Android SDK installation.");
			int setup_android_pid = setup_process_data["pid"];
			if (setup_android_pid > 0 && OS::get_singleton()->is_process_running(setup_android_pid)) {
				OS::get_singleton()->kill(setup_android_pid);
			}
		} break;

		case INSTALLING_JAVA_SDK: {
			print_verbose("Canceling Java SDK installation.");
		} break;
	}

	_hide_setup_dialog(current_setup_status);

	if (on_setup_cancelled.is_valid()) {
		on_setup_cancelled.call_deferred();
	}

	on_setup_completed = Callable();
	on_setup_cancelled = Callable();
}

void AndroidSDKManager::_show_setup_dialog(SetupStatus p_status) {
	if (current_setup_status != IDLE) {
		_hide_setup_dialog(current_setup_status);
	}
	current_setup_status = p_status;

	String label;
	bool outputs_visible = false;
	bool ok_button_visible = false;
	bool show_progress_bar = false;
	switch (p_status) {
		case IDLE: {
			label = "";
		} break;
		case PROMPTING_ANDROID_ONLINE_ACCESS: {
			label = TTRC("Enable online mode for Android SDK setup?");
			ok_button_visible = true;
			connect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_force_online_mode));
		} break;
		case PROMPTING_ANDROID_SDK_SETUP: {
			label = TTRC("Setup Android SDK?");
			ok_button_visible = true;
			connect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_setup_android_sdk));
		} break;
		case PROMPTING_JAVA_ONLINE_ACCESS: {
			label = TTRC("Enable online mode for Java SDK setup?");
			ok_button_visible = true;
			connect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_force_online_mode));
		} break;
		case PROMPTING_JAVA_SDK_SETUP: {
			label = TTRC("Setup Java SDK?");
			ok_button_visible = true;
			connect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_setup_java_sdk));
		} break;
		case DOWNLOADING_ANDROID_SDK: {
			label = TTRC("Downloading Android SDK...");
			show_progress_bar = true;
		} break;
		case DOWNLOADING_JAVA_SDK: {
			label = TTRC("Downloading Java SDK...");
			show_progress_bar = true;
		} break;
		case INSTALLING_ANDROID_SDK: {
			label = TTRC("Installing Android SDK...");
			outputs_visible = true;
			show_progress_bar = true;
		} break;
		case INSTALLING_JAVA_SDK: {
			label = TTRC("Installing Java SDK...");
			show_progress_bar = true;
		} break;
	}

	get_ok_button()->set_visible(ok_button_visible);
	execute_outputs->clear();
	execute_outputs->set_visible(outputs_visible);
	setup_label->set_text(label);
	setup_progress_bar->set_indeterminate(true);
	setup_progress_bar->set_visible(show_progress_bar);
	reset_size();
	popup_centered();

	set_process(true);
}

void AndroidSDKManager::_hide_setup_dialog(SetupStatus p_status) {
	if (p_status != current_setup_status) {
		return;
	}

	switch (p_status) {
		case PROMPTING_ANDROID_ONLINE_ACCESS: {
			disconnect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_force_online_mode));
		} break;
		case PROMPTING_ANDROID_SDK_SETUP: {
			disconnect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_setup_android_sdk));
		} break;
		case PROMPTING_JAVA_ONLINE_ACCESS: {
			disconnect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_force_online_mode));
		} break;
		case PROMPTING_JAVA_SDK_SETUP: {
			disconnect(SNAME("confirmed"), callable_mp(this, &AndroidSDKManager::_setup_java_sdk));
		} break;
		default: {
		} break;
	}

	current_setup_status = IDLE;
	set_process(false);

	hide();

	execute_outputs->clear();
	execute_outputs->set_visible(false);
	setup_label->set_text("");
	setup_progress_bar->set_visible(false);
	setup_progress_bar->set_indeterminate(true);
	get_ok_button()->hide();
}

void AndroidSDKManager::run_setup(const Callable &p_on_setup_completed, const Callable &p_on_setup_cancelled) {
	on_setup_cancelled = p_on_setup_cancelled;
	if (!is_java_sdk_setup()) {
		on_setup_completed = callable_mp(this, &AndroidSDKManager::run_setup).bind(p_on_setup_completed, p_on_setup_cancelled);
		_show_setup_dialog(PROMPTING_JAVA_SDK_SETUP);
	} else if (!is_android_sdk_setup()) {
		on_setup_completed = callable_mp(this, &AndroidSDKManager::run_setup).bind(p_on_setup_completed, p_on_setup_cancelled);
		_show_setup_dialog(PROMPTING_ANDROID_SDK_SETUP);
	} else {
		on_setup_completed = Callable();
		on_setup_cancelled = Callable();
		if (p_on_setup_completed.is_valid()) {
			p_on_setup_completed.call_deferred();
		}
	}
}

void AndroidSDKManager::_install_android_sdk_packages() {
	String cli_bin = get_android_cli_path();
	if (!FileAccess::exists(cli_bin)) {
		ERR_PRINT("Unable to find android-cli binary");
		return;
	}

	String default_android_sdk_path = EditorPaths::get_singleton()->get_default_android_sdk_path();

	List<String> cli_args;
	cli_args.push_back("--sdk=" + default_android_sdk_path);
	cli_args.push_back("sdk");
	cli_args.push_back("install");

	const char **packages = ANDROID_SDK_PACKAGES;
	while (*packages) {
		String package = String(*packages);
		print_verbose("Requesting Android SDK package " + package);
		cli_args.push_back(package);
		packages++;
	}

	print_verbose("Installing Android SDK packages to " + default_android_sdk_path);
	setup_process_data = OS::get_singleton()->execute_with_pipe(cli_bin, cli_args, false);
	if (!setup_process_data.has("pid") || setup_process_data["pid"].operator int() <= 0) {
		ERR_PRINT("Installation of Android SDK packages failed.");
		_cancel_setup();
	} else {
		install_android_sdk_time = OS::get_singleton()->get_ticks_usec();
		_show_setup_dialog(INSTALLING_ANDROID_SDK);
	}
}

void AndroidSDKManager::_android_sdk_installed(int p_exit_code) {
	if (p_exit_code != 0) {
		ERR_PRINT("Installation of Android SDK packages failed. Check output for the error.");
		execute_outputs->add_text(TTR("Installation of Android SDK packages failed. Check output for the error."));
	} else {
		// Update the editor settings.
		EditorSettings::get_singleton()->set_setting("export/android/android_sdk_path", EditorPaths::get_singleton()->get_default_android_sdk_path());
		EditorSettings::get_singleton()->save();

		_hide_setup_dialog(current_setup_status);
		emit_signal(SNAME("android_sdk_installed"));

		// Run the next setup step..
		if (on_setup_completed.is_valid()) {
			on_setup_completed.call_deferred();
		}
	}
}

void AndroidSDKManager::_setup_android_sdk() {
	if (is_android_sdk_setup()) {
		print_verbose("Android SDK is already setup!");
		return;
	}

	if (_is_android_sdk_setup(EditorPaths::get_singleton()->get_default_android_sdk_path())) {
		// Check if the Android SDK is already set up in the default path. If it's, we just update the editor settings
		// to point to the default Android SDK path.
		_android_sdk_installed(0);
	} else {
		if (_is_online()) {
			String cli_bin = get_android_cli_path();
			if (!FileAccess::exists(cli_bin)) {
				_download_android_cli();
			} else {
				_install_android_sdk_packages();
			}
		} else {
			_show_setup_dialog(PROMPTING_ANDROID_ONLINE_ACCESS);
		}
	}
}

void AndroidSDKManager::_setup_java_sdk() {
	if (is_java_sdk_setup()) {
		print_verbose("Java SDK is already setup!");
		return;
	}

	if (_is_java_sdk_setup(EditorPaths::get_singleton()->get_default_java_sdk_path())) {
		// Check if the Java SDK is already set up in the default path. If it's, we just update the editor settings
		// to point to the default Java SDK path.
		_java_sdk_installed();
	} else {
		if (_is_online()) {
			_download_java_sdk();
		} else {
			_show_setup_dialog(PROMPTING_JAVA_ONLINE_ACCESS);
		}
	}
}

void AndroidSDKManager::_download_java_sdk() {
	String os_str;
	if (OS::get_singleton()->has_feature("windows")) {
		os_str = "windows";
	} else if (OS::get_singleton()->has_feature("macos")) {
		os_str = "mac";
	} else if (OS::get_singleton()->has_feature("linux")) {
		os_str = "linux";
	}

	if (os_str.is_empty()) {
		ERR_PRINT("Unsupported OS for Java SDK download.");
		return;
	}

	String arch_str = "x64";
	if (Engine::get_singleton()->get_architecture_name() == "arm64") {
		arch_str = "aarch64";
	}

	String url = vformat("https://api.adoptium.net/v3/binary/latest/%d/ga/%s/%s/jdk/hotspot/normal/eclipse", DEFAULT_JAVA_VERSION, os_str, arch_str);

	Error err = downloader->request(url);
	if (err != OK) {
		ERR_PRINT("Failed to start Java SDK download.");
		_cancel_setup();
	} else {
		print_verbose("Downloading Java SDK from " + url);
		_show_setup_dialog(DOWNLOADING_JAVA_SDK);
	}
}

void AndroidSDKManager::_on_download_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != HTTPClient::RESPONSE_OK) {
		ERR_PRINT(vformat("Download failed. Result: %d\nResponse code: %d", p_result, p_code));
		_cancel_setup();
	} else {
		if (current_setup_status == DOWNLOADING_JAVA_SDK) {
			_java_sdk_downloaded(p_body);
		} else if (current_setup_status == DOWNLOADING_ANDROID_SDK) {
			_android_cli_downloaded(p_body);
		}
	}
}

void AndroidSDKManager::_java_sdk_downloaded(const PackedByteArray &p_body) {
	_show_setup_dialog(INSTALLING_JAVA_SDK);

	String cache_dir = EditorPaths::get_singleton()->get_cache_dir();
	String archive_ext = ".tar.gz";
#ifdef WINDOWS_ENABLED
	archive_ext = ".zip";
#endif
	String archive_path = cache_dir.path_join("java_sdk_temp" + archive_ext);

	Ref<FileAccess> f = FileAccess::open(archive_path, FileAccess::WRITE);
	if (f.is_null()) {
		ERR_PRINT("Failed to save Java SDK archive.");
		return;
	}
	print_verbose("Storing Java SDK archive to " + archive_path);
	f->store_buffer(p_body.ptr(), p_body.size());
	f.unref();

	String default_java_sdk_path = EditorPaths::get_singleton()->get_default_java_sdk_path();
	Error err = _extract_java_sdk(archive_path, default_java_sdk_path);
	if (err != OK) {
		ERR_PRINT("Unable to extract Java SDK.");
		_cancel_setup();
		return;
	}

	// Remove temporary archive.
	print_verbose("Deleting Java SDK archive...");
	DirAccess::remove_file_or_error(archive_path);

	_java_sdk_installed();
}

void AndroidSDKManager::_java_sdk_installed() {
	// Update the editor settings.
	EditorSettings::get_singleton()->set_setting("export/android/java_sdk_path", EditorPaths::get_singleton()->get_default_java_sdk_path());
	EditorSettings::get_singleton()->save();

	_hide_setup_dialog(current_setup_status);
	emit_signal(SNAME("java_sdk_installed"));

	// Run the next setup step..
	if (on_setup_completed.is_valid()) {
		on_setup_completed.call_deferred();
	}
}

Error AndroidSDKManager::_extract_java_sdk(const String &p_file, const String &p_target_path) {
	print_verbose(vformat("Extracting Java SDK from %s to %s", p_file, p_target_path));
	if (OS::get_singleton()->has_feature("windows")) {
		Ref<ZIPReader> reader;
		reader.instantiate();
		Error err = reader->open(p_file);
		if (err != OK) {
			ERR_PRINT(vformat("Can't open the Java SDK zip file %s: %s", p_file, error_names[err]));
			return err;
		}

		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		PackedStringArray reader_files = reader->get_files();
		for (const String &file : reader_files) {
			String file_path = file;
			if (file_path.ends_with("/")) {
				continue;
			}

			// Temurin archives usually have a root directory like `jdk-17.0.x+y`.
			// We want to extract its contents directly into target_path.
			int first_slash = file_path.find("/");
			if (first_slash != -1) {
				file_path = file_path.substr(first_slash + 1);
			}

			if (file_path.is_empty()) {
				continue;
			}

			String output_file = p_target_path.path_join(file_path);
			err = da->make_dir_recursive(output_file.get_base_dir());
			if (err != OK) {
				ERR_PRINT(vformat("Failed to create directory %s.", output_file.get_base_dir()));
				return err;
			}

			PackedByteArray buffer = reader->read_file(file, true);
			Ref<FileAccess> f = FileAccess::open(output_file, FileAccess::WRITE);
			if (f.is_valid()) {
				f->store_buffer(buffer);
				f.unref();
			}
		}
		return OK;
	} else {
		// Use `tar` for extraction on Linux and macOS.
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (!da->dir_exists(p_target_path)) {
			da->make_dir_recursive(p_target_path);
		}

		// We strip the root folder.
		uint64_t strip_components = 1;
		if (OS::get_singleton()->has_feature("macos")) {
			// The actual content is 3 components deep for macOS archives (e.g. 'root/Contents/Home/').
			strip_components = 3;
		}

		List<String> args;
		args.push_back("-xzf");
		args.push_back(p_file);
		args.push_back("-C");
		args.push_back(p_target_path);
		args.push_back("--strip-components=" + String::num_uint64(strip_components));

		int retval;
		Error err = OS::get_singleton()->execute("tar", args, nullptr, &retval, false);
		if (err != OK || retval != 0) {
			ERR_PRINT("Failed to extract Java SDK using tar.");
			return FAILED;
		}
		return err;
	}
}

void AndroidSDKManager::_download_android_cli() {
	String url;
	if (OS::get_singleton()->has_feature("windows")) {
		url = ANDROID_CLI_URL_WIN;
	} else if (OS::get_singleton()->has_feature("macos")) {
		if (Engine::get_singleton()->get_architecture_name() == "arm64") {
			url = ANDROID_CLI_URL_MAC_ARM;
		} else {
			url = ANDROID_CLI_URL_MAC_X86;
		}
	} else if (OS::get_singleton()->has_feature("linux")) {
		url = ANDROID_CLI_URL_LINUX;
	}

	if (url.is_empty()) {
		ERR_PRINT("Unsupported OS for android-cli download.");
		return;
	}

	Error err = downloader->request(url);
	if (err != OK) {
		ERR_PRINT("Failed to start android-cli download.");
		_cancel_setup();
	} else {
		print_verbose("Downloading android-cli from " + url);
		_show_setup_dialog(DOWNLOADING_ANDROID_SDK);
	}
}

void AndroidSDKManager::_android_cli_downloaded(const PackedByteArray &p_body) {
	String cli_path = EditorPaths::get_singleton()->get_cache_dir().path_join("android-cli");
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!da->dir_exists(cli_path)) {
		da->make_dir_recursive(cli_path);
	}

	String exe_ext;
#ifdef WINDOWS_ENABLED
	exe_ext = ".exe";
#endif
	String cli_bin = cli_path.path_join("android" + exe_ext);

	Ref<FileAccess> f = FileAccess::open(cli_bin, FileAccess::WRITE);
	if (f.is_null()) {
		ERR_PRINT("Failed to save android-cli binary.");
	} else {
		print_verbose("Storing android-cli to " + cli_bin);
		f->store_buffer(p_body.ptr(), p_body.size());
		f.unref();

#ifndef WINDOWS_ENABLED
		FileAccess::set_unix_permissions(cli_bin, BitField<FileAccess::UnixPermissionFlags>(0755));
#endif

		_install_android_sdk_packages();
	}
}

bool AndroidSDKManager::_is_online() {
	return (int)EDITOR_GET("network/connection/network_mode") == EditorSettings::NETWORK_ONLINE;
}

void AndroidSDKManager::_force_online_mode() {
	EditorSettings::get_singleton()->set_setting("network/connection/network_mode", EditorSettings::NETWORK_ONLINE);
	EditorSettings::get_singleton()->notify_changes();
	EditorSettings::get_singleton()->save();

	if (current_setup_status == PROMPTING_ANDROID_ONLINE_ACCESS) {
		_setup_android_sdk();
	} else if (current_setup_status == PROMPTING_JAVA_ONLINE_ACCESS) {
		_setup_java_sdk();
	}
}

bool AndroidSDKManager::is_java_sdk_setup(String *r_error) {
	return _is_java_sdk_setup(EDITOR_GET("export/android/java_sdk_path"), r_error);
}

bool AndroidSDKManager::_is_java_sdk_setup(const String &p_java_sdk_path, String *r_error) {
	// Check the Java SDK is set up.
	if (p_java_sdk_path.is_empty()) {
		if (r_error) {
			*r_error += TTR("Invalid Java SDK path in Editor Settings.") + "\n";
		}
		return false;
	} else {
		// Validate the given path by checking that `java` is present under the `bin` directory.
		Error errn;
		// Check for the bin directory.
		Ref<DirAccess> da = DirAccess::open(p_java_sdk_path.path_join("bin"), &errn);
		if (errn != OK) {
			if (r_error) {
				*r_error += TTR("Invalid Java SDK setup.") + " ";
				*r_error += "\n";
			}
			return false;
		} else {
			// Check for the `java` command.
			String java_path = _get_java_path(p_java_sdk_path);
			if (!FileAccess::exists(java_path)) {
				if (r_error) {
					*r_error += TTR("Unable to find 'java' command using the Java SDK path. Please check the Java SDK directory specified in Editor Settings.") + "\n";
				}
				return false;
			}
		}
	}
	return true;
}

bool AndroidSDKManager::is_android_sdk_setup(String *r_error) {
	return _is_android_sdk_setup(EDITOR_GET("export/android/android_sdk_path"), r_error);
}

bool AndroidSDKManager::_is_android_sdk_setup(const String &p_android_sdk_path, String *r_error) {
	// Check the Android SDK is set up.
	if (p_android_sdk_path.is_empty()) {
		if (r_error) {
			*r_error += TTR("Invalid Android SDK path in Editor Settings.") + "\n";
		}
		return false;
	} else {
		Error errn;

		// Check the Android SDK packages are installed.
		const char **packages = ANDROID_SDK_PACKAGES;
		while (*packages) {
			String package = String(*packages);
			Ref<DirAccess> da = DirAccess::open(p_android_sdk_path.path_join(package), &errn);
			if (errn != OK) {
				if (r_error) {
					*r_error += TTR("Unable to find Android SDK package '") + package + "'.\n";
				}
				return false;
			}

			packages++;
		}

		// Validate that adb is available.
		String adb_path = _get_adb_path(p_android_sdk_path);
		if (!FileAccess::exists(adb_path)) {
			if (r_error) {
				*r_error += TTR("Unable to find Android SDK platform-tools' adb command. Please check in the Android SDK directory specified in Editor Settings.") + "\n";
			}
			return false;
		}
	}

	return true;
}

String AndroidSDKManager::get_android_cli_path() {
	String cli_path = EditorPaths::get_singleton()->get_cache_dir().path_join("android-cli");
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".exe";
	}
	return cli_path.path_join("android" + exe_ext);
}

String AndroidSDKManager::_get_java_path(const String &p_java_sdk_path) {
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".exe";
	}
	return p_java_sdk_path.path_join("bin/java" + exe_ext);
}

String AndroidSDKManager::get_java_path() {
	return _get_java_path(EDITOR_GET("export/android/java_sdk_path"));
}

String AndroidSDKManager::_get_adb_path(const String &p_android_sdk_path) {
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".exe";
	}
	return p_android_sdk_path.path_join("platform-tools/adb" + exe_ext);
}

String AndroidSDKManager::get_adb_path() {
	return _get_adb_path(EDITOR_GET("export/android/android_sdk_path"));
}

String AndroidSDKManager::get_apksigner_path(int p_target_sdk, bool p_check_executes) {
	if (p_target_sdk == -1) {
		p_target_sdk = DEFAULT_TARGET_SDK_VERSION;
	}
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".bat";
	}
	String apksigner_command_name = "apksigner" + exe_ext;
	String sdk_path = EDITOR_GET("export/android/android_sdk_path");
	String apksigner_path;

	Error errn;
	String build_tools_dir = sdk_path.path_join("build-tools");
	Ref<DirAccess> da = DirAccess::open(build_tools_dir, &errn);
	if (errn != OK) {
		print_error("Unable to open Android 'build-tools' directory.");
		return apksigner_path;
	}

	// There are additional versions directories we need to go through.
	Vector<String> dir_list = da->get_directories();

	// We need to use the version of build_tools that matches the Target SDK
	// If somehow we can't find that, we see if a version between 28 and the default target SDK exists.
	// We need to avoid versions <= 27 because they fail on Java versions >9
	// If we can't find that, we just use the first valid version.
	Vector<String> ideal_versions;
	Vector<String> other_versions;
	Vector<String> versions;
	bool found_target_sdk = false;
	// We only allow for versions <= 27 if specifically set
	int min_version = p_target_sdk <= 27 ? p_target_sdk : 28;
	for (String sub_dir : dir_list) {
		if (!sub_dir.begins_with(".")) {
			Vector<String> ver_numbers = sub_dir.split(".");
			// Dir not a version number, will use as last resort
			if (!ver_numbers.size() || !ver_numbers[0].is_valid_int()) {
				other_versions.push_back(sub_dir);
				continue;
			}
			int ver_number = ver_numbers[0].to_int();
			if (ver_number == p_target_sdk) {
				found_target_sdk = true;
				//ensure this is in front of the ones we check
				versions.push_back(sub_dir);
			} else {
				if (ver_number >= min_version && ver_number <= DEFAULT_TARGET_SDK_VERSION) {
					ideal_versions.push_back(sub_dir);
				} else {
					other_versions.push_back(sub_dir);
				}
			}
		}
	}
	// we will check ideal versions first, then other versions.
	versions.append_array(ideal_versions);
	versions.append_array(other_versions);

	if (!versions.size()) {
		print_error("Unable to find the 'apksigner' tool.");
		return apksigner_path;
	}

	int i;
	bool failed = false;
	String version_to_use;

	String java_sdk_path = EDITOR_GET("export/android/java_sdk_path");
	if (!java_sdk_path.is_empty()) {
		OS::get_singleton()->set_environment("JAVA_HOME", java_sdk_path);

#ifdef UNIX_ENABLED
		String env_path = OS::get_singleton()->get_environment("PATH");
		if (!env_path.contains(java_sdk_path)) {
			OS::get_singleton()->set_environment("PATH", java_sdk_path + "/bin:" + env_path);
		}
#endif
	}

	List<String> args;
	args.push_back("--version");
	String output;
	int retval;
	Error err;
	for (i = 0; i < versions.size(); i++) {
		// Check if the tool is here.
		apksigner_path = build_tools_dir.path_join(versions[i]).path_join(apksigner_command_name);
		if (FileAccess::exists(apksigner_path)) {
			version_to_use = versions[i];
			// If we aren't exporting, just break here.
			if (!p_check_executes) {
				break;
			}
			// we only check to see if it executes on export because it is slow to load
			err = OS::get_singleton()->execute(apksigner_path, args, &output, &retval, false);
			if (err || retval) {
				failed = true;
			} else {
				break;
			}
		}
	}
	if (i == versions.size()) {
		if (failed) {
			print_error("All located 'apksigner' tools in " + build_tools_dir + " failed to execute");
			return "<FAILED>";
		} else {
			print_error("Unable to find the 'apksigner' tool.");
			return "";
		}
	}
	if (!found_target_sdk) {
		print_line("Could not find version of build tools that matches Target SDK, using " + version_to_use);
	} else if (failed && found_target_sdk) {
		print_line("Version of build tools that matches Target SDK failed to execute, using " + version_to_use);
	}

	return apksigner_path;
}

AndroidSDKManager::AndroidSDKManager() {
	set_title(TTRC("Android SDK Manager"));
	add_cancel_button();
	get_ok_button()->hide();

	VBoxContainer *container = memnew(VBoxContainer);
	add_child(container);

	setup_label = memnew(Label);
	container->add_child(setup_label);

	setup_progress_bar = memnew(ProgressBar);
	setup_progress_bar->set_indeterminate(true);
	container->add_child(setup_progress_bar);

	execute_outputs = memnew(RichTextLabel);
	execute_outputs->set_selection_enabled(true);
	execute_outputs->set_context_menu_enabled(true);
	execute_outputs->set_scroll_follow(true);
	execute_outputs->set_custom_minimum_size(Size2i(300, 200) * EDSCALE);
	execute_outputs->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	container->add_child(execute_outputs);

	downloader = memnew(HTTPRequest);
	add_child(downloader);
	downloader->connect(SNAME("request_completed"), callable_mp(this, &AndroidSDKManager::_on_download_completed));
}
