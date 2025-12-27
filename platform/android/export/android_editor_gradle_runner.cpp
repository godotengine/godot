/**************************************************************************/
/*  android_editor_gradle_runner.cpp                                      */
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

#ifdef ANDROID_ENABLED
#include "android_editor_gradle_runner.h"

#include "editor/editor_interface.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/rich_text_label.h"

#include "../java_godot_wrapper.h"
#include "../os_android.h"

void AndroidEditorGradleRunner::run_gradle(const String &p_project_path, const String &p_build_path, const String &p_output_path, const String &p_export_format, const List<String> &p_gradle_build_args, const List<String> &p_gradle_copy_args) {
	project_path = p_project_path;
	build_path = p_build_path;
	output_path = p_output_path;
	export_format = p_export_format;
	gradle_build_args = p_gradle_build_args;
	gradle_copy_args = p_gradle_copy_args;

	if (output_dialog == nullptr) {
		output_label = memnew(RichTextLabel);
		output_label->set_selection_enabled(true);
		output_label->set_context_menu_enabled(true);
		output_label->set_scroll_follow(true);

		output_dialog = memnew(ConfirmationDialog);
		output_dialog->set_unparent_when_invisible(true);
		output_dialog->set_title(TTR("Building Android Project (gradle)"));
		output_dialog->add_child(output_label);

		output_dialog->connect("canceled", callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_cancel));
	}

	output_label->clear();
	output_dialog->get_ok_button()->set_disabled(true);

	EditorInterface::get_singleton()->popup_dialog_centered_ratio(output_dialog);

	state = STATE_BUILDING;
	_android_gradle_build_connect();
}

void AndroidEditorGradleRunner::_android_gradle_build_connect() {
	_android_gradle_build_output(0, TTR("> Connecting to Gradle Build Environment..."));

	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	if (!godot_java->build_env_connect(callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_build))) {
		_android_gradle_build_failed(TTR("Unable to connect to Gradle Build Environment service"));
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_disconnect() {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	godot_java->build_env_disconnect();
}

void AndroidEditorGradleRunner::_android_gradle_build_output(int p_type, const String &p_line) {
	if (p_type == 0) {
		print_line(p_line);
		output_label->append_text("[color=green]" + p_line + "[/color]\n");
	} else if (p_type == 1) {
		print_line(p_line);
		output_label->add_text(p_line + "\n");
	} else {
		print_error(p_line);
		output_label->append_text("[color=red]" + p_line + "[/color]\n");
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_build() {
	_android_gradle_build_output(0, TTR("> Starting Gradle build..."));

	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	job_id = godot_java->build_env_execute(
			"gradle",
			gradle_build_args,
			project_path,
			build_path,
			callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_output),
			callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_build_callback));
	if (job_id < 0) {
		_android_gradle_build_failed(TTR("Failed to execute Gradle command"));
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_build_callback(int p_exit_code) {
	job_id = -1;
	if (p_exit_code != 0) {
		_android_gradle_build_failed();
		return;
	}

	_android_gradle_build_copy();
}

void AndroidEditorGradleRunner::_android_gradle_build_copy() {
	_android_gradle_build_output(0, TTR("> Copying Gradle artifacts..."));

	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	job_id = godot_java->build_env_execute(
			"gradle",
			gradle_copy_args,
			project_path,
			build_path,
			callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_output),
			callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_copy_callback));
	if (job_id < 0) {
		_android_gradle_build_failed(TTR("Failed to execute Gradle command"));
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_copy_callback(int p_exit_code) {
	job_id = -1;
	if (p_exit_code != 0) {
		_android_gradle_build_failed();
	} else {
		_android_gradle_build_clean_project(true);
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_clean_project(bool p_was_successful) {
	if (state != STATE_CLEANING) {
		state = STATE_CLEANING;

		if (p_was_successful) {
			output_dialog->hide();

			bool prompt_apk_install = EDITOR_GET("export/android/install_exported_apk");
			if (prompt_apk_install && export_format == "apk") {
				OS_Android::get_singleton()->shell_open(output_path);
			}
		} else {
			output_dialog->get_ok_button()->set_disabled(false);
		}

		GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
		godot_java->build_env_clean_project(
				project_path,
				build_path,
				callable_mp(this, &AndroidEditorGradleRunner::_android_gradle_build_clean_project_callback));
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_clean_project_callback() {
	// Ensure we haven't switched back to STATE_BUILDING in the meantime.
	if (state == STATE_CLEANING) {
		_android_gradle_build_disconnect();
		state = STATE_IDLE;
	}
}

void AndroidEditorGradleRunner::_android_gradle_build_failed(const String &p_msg) {
	job_id = -1;

	if (p_msg != "") {
		_android_gradle_build_output(1, p_msg);
	}

	_android_gradle_build_clean_project(false);
}

void AndroidEditorGradleRunner::_android_gradle_build_cancel() {
	if (job_id > 0) {
		GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
		godot_java->build_env_cancel(job_id);
		_android_gradle_build_clean_project(false);
	}
}

#endif // ANDROID_ENABLED
