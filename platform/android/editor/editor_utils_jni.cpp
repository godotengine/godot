/**************************************************************************/
/*  editor_utils_jni.cpp                                                  */
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

#include "editor_utils_jni.h"

#include "jni_utils.h"

#ifdef TOOLS_ENABLED
#include "editor/debugger/editor_debugger_node.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/run/editor_run_bar.h"
#include "main/main.h"
#endif

extern "C" {
JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_EditorUtils_runScene(JNIEnv *p_env, jclass, jstring p_scene, jobjectArray p_scene_args) {
#ifdef TOOLS_ENABLED
	Vector<String> scene_args;
	jint length = p_env->GetArrayLength(p_scene_args);
	for (jint i = 0; i < length; ++i) {
		jstring j_arg = (jstring)p_env->GetObjectArrayElement(p_scene_args, i);
		String arg = jstring_to_string(j_arg, p_env);
		scene_args.push_back(arg);
		p_env->DeleteLocalRef(j_arg);
	}

	String scene = jstring_to_string(p_scene, p_env);

	EditorRunBar *editor_run_bar = EditorRunBar::get_singleton();
	if (editor_run_bar != nullptr) {
		editor_run_bar->stop_playing();
		// Ensure that all ScriptEditorDebugger instances are explicitly stopped.
		// If not, a closing instance from the previous run session will trigger `_stop_and_notify()`, in turn causing
		// the closure of the ScriptEditorDebugger instances of the run session we're about to launch.
		EditorDebuggerNode *dbg_node = EditorDebuggerNode::get_singleton();
		if (dbg_node != nullptr) {
			for (int i = 0; ScriptEditorDebugger *dbg = dbg_node->get_debugger(i); i++) {
				dbg->stop();
			}
		}

		if (scene.is_empty()) {
			editor_run_bar->play_main_scene(false);
		} else {
			editor_run_bar->play_custom_scene(scene, scene_args);
		}
	} else {
		List<String> args;

		for (const String &a : Main::get_forwardable_cli_arguments(Main::CLI_SCOPE_PROJECT)) {
			args.push_back(a);
		}

		for (const String &arg : scene_args) {
			args.push_back(arg);
		}

		if (!scene.is_empty()) {
			args.push_back("--scene");
			args.push_back(scene);
		}

		Error err = OS::get_singleton()->create_instance(args);
		ERR_FAIL_COND(err);
	}
#endif
}
}
