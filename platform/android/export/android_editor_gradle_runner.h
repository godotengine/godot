/**************************************************************************/
/*  android_editor_gradle_runner.h                                        */
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

#ifdef ANDROID_ENABLED

#include "core/object/object.h"

class ConfirmationDialog;
class RichTextLabel;

class AndroidEditorGradleRunner : public Object {
	GDCLASS(AndroidEditorGradleRunner, Object);

	RichTextLabel *output_label = nullptr;
	ConfirmationDialog *output_dialog = nullptr;

	enum State {
		STATE_IDLE,
		STATE_BUILDING,
		STATE_CLEANING,
	};
	State state = STATE_IDLE;

	String project_path;
	String build_path;
	String output_path;
	String export_format;
	List<String> gradle_build_args;
	List<String> gradle_copy_args;
	int64_t job_id;

	void _android_gradle_build_connect();
	void _android_gradle_build_disconnect();
	void _android_gradle_build_output(int p_type, const String &p_line);
	void _android_gradle_build_build();
	void _android_gradle_build_build_callback(int p_exit_code);
	void _android_gradle_build_copy();
	void _android_gradle_build_copy_callback(int p_exit_code);
	void _android_gradle_build_clean_project(bool p_was_successful);
	void _android_gradle_build_clean_project_callback();

	void _android_gradle_build_failed(const String &p_msg = String());
	void _android_gradle_build_cancel();

public:
	void run_gradle(const String &p_project_path, const String &p_build_path, const String &p_output_path, const String &p_export_format, const List<String> &p_gradle_build_args, const List<String> &p_gradle_copy_args);
};

#endif // ANDROID_ENABLED
