/**************************************************************************/
/*  android_sdk_manager.h                                                 */
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

#include "scene/gui/dialogs.h"
#include "scene/main/http_request.h"

class Label;
class ProgressBar;
class RichTextLabel;

class AndroidSDKManager : public AcceptDialog {
	GDCLASS(AndroidSDKManager, AcceptDialog)

	enum SetupStatus {
		IDLE,
		PROMPTING_ANDROID_ONLINE_ACCESS,
		PROMPTING_ANDROID_SDK_SETUP,
		PROMPTING_JAVA_ONLINE_ACCESS,
		PROMPTING_JAVA_SDK_SETUP,
		DOWNLOADING_ANDROID_SDK,
		DOWNLOADING_JAVA_SDK,
		INSTALLING_ANDROID_SDK,
		INSTALLING_JAVA_SDK,
	};

	void _cancel_setup();
	void _on_download_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);

	void _setup_android_sdk();
	void _download_android_cli();
	void _android_cli_downloaded(const PackedByteArray &p_body);
	void _install_android_sdk_packages();
	void _android_sdk_installed(int p_exit_code);

	void _setup_java_sdk();
	void _download_java_sdk();
	void _java_sdk_downloaded(const PackedByteArray &p_body);
	Error _extract_java_sdk(const String &p_file, const String &p_target_path);
	void _java_sdk_installed();

	void _show_setup_dialog(SetupStatus p_status);
	void _hide_setup_dialog(SetupStatus p_status);

	bool _is_online();
	void _force_online_mode();

	static bool _is_android_sdk_setup(const String &p_android_sdk_path, String *r_error = nullptr);
	static bool _is_java_sdk_setup(const String &p_java_sdk_path, String *r_error = nullptr);

	static String _get_java_path(const String &p_java_sdk_path);
	static String _get_adb_path(const String &p_android_sdk_path);

	static String get_android_cli_path();

	Callable on_setup_completed;
	Callable on_setup_cancelled;
	Dictionary setup_process_data;
	uint64_t install_android_sdk_time;
	Label *setup_label = nullptr;
	RichTextLabel *execute_outputs = nullptr;
	ProgressBar *setup_progress_bar = nullptr;
	HTTPRequest *downloader = nullptr;
	SetupStatus current_setup_status = IDLE;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	constexpr static const int DEFAULT_MIN_SDK_VERSION = 24; // Should match the value in 'platform/android/java/app/config.gradle#minSdk'.
	constexpr static const int VULKAN_MIN_SDK_VERSION = 29; // Minimum recommended sdk version for Vulkan 1.1 support. See https://developer.android.com/games/develop/vulkan/native-engine-support#recommendations.
	constexpr static const int DEFAULT_TARGET_SDK_VERSION = 36; // Should match the value in 'platform/android/java/app/config.gradle#targetSdk'.
	constexpr static const int DEFAULT_JAVA_VERSION = 17; // Should match the value in 'platform/android/java/app/config.gradle#javaVersion'.

	void run_setup(const Callable &p_on_setup_completed = Callable(), const Callable &p_on_setup_cancelled = Callable());

	/// Returns true if the Android SDK is set up.
	static bool is_android_sdk_setup(String *r_error = nullptr);
	/// Returns true if the Java SDK is set up.
	static bool is_java_sdk_setup(String *r_error = nullptr);

	static String get_java_path();
	static String get_adb_path();
	static String get_apksigner_path(int p_target_sdk = -1, bool p_check_executes = false);

	AndroidSDKManager();
};
