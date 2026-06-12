/**************************************************************************/
/*  game_menu_utils_jni.cpp                                               */
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

#include "game_menu_utils_jni.h"

#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/run/game_view_plugin.h"

_FORCE_INLINE_ static GameViewPlugin *_get_game_view_plugin() {
	ERR_FAIL_NULL_V(EditorNode::get_singleton(), nullptr);
	ERR_FAIL_NULL_V(EditorNode::get_singleton()->get_editor_main_screen(), nullptr);
	return Object::cast_to<GameViewPlugin>(EditorNode::get_singleton()->get_editor_main_screen()->get_plugin_by_name("Game"));
}

extern "C" {

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setSuspend(JNIEnv *env, jclass clazz, jboolean enabled) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_suspend(enabled);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_nextFrame(JNIEnv *env, jclass clazz) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->next_frame();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setNodeType(JNIEnv *env, jclass clazz, jint type) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_node_type(type);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setSelectMode(JNIEnv *env, jclass clazz, jint mode) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_select_mode(mode);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setSelectionVisible(JNIEnv *env, jclass clazz, jboolean visible) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_selection_visible(visible);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setCameraOverride(JNIEnv *env, jclass clazz, jboolean enabled) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_camera_override(enabled);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setCameraManipulateMode(JNIEnv *env, jclass clazz, jint mode) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_camera_manipulate_mode(static_cast<EditorDebuggerNode::CameraOverride>(mode));
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_resetCamera2DPosition(JNIEnv *env, jclass clazz) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->reset_camera_2d_position();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_resetCamera3DPosition(JNIEnv *env, jclass clazz) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->reset_camera_3d_position();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_playMainScene(JNIEnv *env, jclass clazz) {
	if (EditorInterface::get_singleton()) {
		EditorInterface::get_singleton()->play_main_scene();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setDebugMuteAudio(JNIEnv *env, jclass clazz, jboolean enabled) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_debug_mute_audio(enabled);
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_resetTimeScale(JNIEnv *env, jclass clazz) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->reset_time_scale();
	}
}

JNIEXPORT void JNICALL Java_org_godotengine_godot_editor_utils_GameMenuUtils_setTimeScale(JNIEnv *env, jclass clazz, jdouble scale) {
	GameViewPlugin *game_view_plugin = _get_game_view_plugin();
	if (game_view_plugin != nullptr && game_view_plugin->get_debugger().is_valid()) {
		game_view_plugin->get_debugger()->set_time_scale(scale);
	}
}
}
