/**************************************************************************/
/*  scene_debugger.cpp                                                    */
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

#include "scene_debugger.h"

#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/math/math_fieldwise.h"
#include "core/os/time.h"
#include "core/templates/local_vector.h"
#include "scene/2d/camera_2d.h"
#include "scene/debugger/scene_debugger_object.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"
#include "scene/theme/theme_db.h"
#include "servers/audio/audio_server.h"

#ifndef _3D_DISABLED
#include "scene/3d/camera_3d.h"
#endif

#ifdef DEBUG_ENABLED
#include "scene/debugger/runtime_node_select.h"
#endif

SceneDebugger::SceneDebugger() {
	singleton = this;

#ifdef DEBUG_ENABLED
	LiveEditor::singleton = memnew(LiveEditor);
	RuntimeNodeSelect::singleton = memnew(RuntimeNodeSelect);

	EngineDebugger::register_message_capture("scene", EngineDebugger::Capture(nullptr, SceneDebugger::parse_message));
#endif
}

SceneDebugger::~SceneDebugger() {
#ifdef DEBUG_ENABLED
	if (LiveEditor::singleton) {
		EngineDebugger::unregister_message_capture("scene");
		memdelete(LiveEditor::singleton);
		LiveEditor::singleton = nullptr;
	}

	if (RuntimeNodeSelect::singleton) {
		memdelete(RuntimeNodeSelect::singleton);
		RuntimeNodeSelect::singleton = nullptr;
	}
#endif // DEBUG_ENABLED

	singleton = nullptr;
}

void SceneDebugger::initialize() {
	if (EngineDebugger::is_active()) {
#ifdef DEBUG_ENABLED
		_init_message_handlers();
#endif
		memnew(SceneDebugger);
	}
}

void SceneDebugger::deinitialize() {
	if (singleton) {
		memdelete(singleton);
	}
}

#ifdef DEBUG_ENABLED

void SceneDebugger::_handle_input(const Ref<InputEvent> &p_event, const Ref<Shortcut> &p_shortcut) {
	Ref<InputEventKey> k = p_event;
	if (p_shortcut.is_valid() && k.is_valid() && k->is_pressed() && !k->is_echo() && p_shortcut->matches_event(k)) {
		EngineDebugger::get_singleton()->send_message("request_quit", Array());
	}
}

void SceneDebugger::_handle_embed_input(const Ref<InputEvent> &p_event, const Dictionary &p_settings) {
	Ref<InputEventKey> k = p_event;
	if (k.is_null() || !k->is_pressed()) {
		return;
	}

	Ref<Shortcut> p_shortcut = p_settings.get("editor/next_frame_embedded_project", Ref<Shortcut>());
	if (p_shortcut.is_valid() && p_shortcut->matches_event(k)) {
		EngineDebugger::get_singleton()->send_message("request_embed_next_frame", Array());
		return;
	}

	if (k->is_echo()) {
		return;
	} // Shortcuts that doesn't need is_echo goes below here

	p_shortcut = p_settings.get("editor/suspend_resume_embedded_project", Ref<Shortcut>());
	if (p_shortcut.is_valid() && p_shortcut->matches_event(k)) {
		EngineDebugger::get_singleton()->send_message("request_embed_suspend_toggle", Array());
		return;
	}
}

Error SceneDebugger::_msg_setup_scene(const Array &p_args) {
	SceneTree::get_singleton()->get_root()->connect(SceneStringName(window_input), callable_mp_static(SceneDebugger::_handle_input).bind(DebuggerMarshalls::deserialize_key_shortcut(p_args)));
	return OK;
}

Error SceneDebugger::_msg_request_scene_tree(const Array &p_args) {
	LiveEditor::get_singleton()->_send_tree();
	return OK;
}

Error SceneDebugger::_msg_save_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	_save_node(p_args[0], p_args[1]);
	Array arr;
	arr.append(p_args[1]);
	EngineDebugger::get_singleton()->send_message("filesystem:update_file", { arr });
	return OK;
}

Error SceneDebugger::_msg_inspect_objects(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	Vector<ObjectID> ids;
	for (const Variant &id : (Array)p_args[0]) {
		ids.append(ObjectID(id.operator uint64_t()));
	}
	_send_object_ids(ids, p_args[1]);
	return OK;
}

#ifndef DISABLE_DEPRECATED
Error SceneDebugger::_msg_inspect_object(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	// Legacy compatibility: convert single object ID to new format, then send single object response.
	Vector<ObjectID> ids;
	ids.append(ObjectID(p_args[0].operator uint64_t()));

	SceneDebuggerObject obj(ids[0]);
	if (obj.id.is_null()) {
		EngineDebugger::get_singleton()->send_message("scene:inspect_object", Array());
		return OK;
	}

	Array arr;
	obj.serialize(arr);
	EngineDebugger::get_singleton()->send_message("scene:inspect_object", arr);
	return OK;
}
#endif // DISABLE_DEPRECATED

Error SceneDebugger::_msg_clear_selection(const Array &p_args) {
	RuntimeNodeSelect::get_singleton()->_clear_selection();
	return OK;
}

Error SceneDebugger::_msg_suspend_changed(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool suspended = p_args[0];
	SceneTree::get_singleton()->set_suspend(suspended);
	RuntimeNodeSelect::get_singleton()->_update_input_state();
	return OK;
}

Error SceneDebugger::_msg_next_frame(const Array &p_args) {
	_next_frame();
	return OK;
}

Error SceneDebugger::_msg_speed_changed(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	double time_scale_user = p_args[0];
	Engine::get_singleton()->set_user_time_scale(time_scale_user);
	return OK;
}

Error SceneDebugger::_msg_debug_mute_audio(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool do_mute = p_args[0];
	AudioServer::get_singleton()->set_debug_mute(do_mute);
	return OK;
}

Error SceneDebugger::_msg_override_cameras(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool enable = p_args[0];
	bool from_editor = p_args[1];
	SceneTree::get_singleton()->get_root()->enable_camera_2d_override(enable);
#ifndef _3D_DISABLED
	SceneTree::get_singleton()->get_root()->enable_camera_3d_override(enable);
#endif // _3D_DISABLED
	RuntimeNodeSelect::get_singleton()->_set_camera_override_enabled(enable && !from_editor);
	return OK;
}

Error SceneDebugger::_msg_set_object_property(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	_set_object_property(p_args[0], p_args[1], p_args[2]);
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}

Error SceneDebugger::_msg_set_object_property_field(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 4, ERR_INVALID_DATA);
	_set_object_property(p_args[0], p_args[1], p_args[2], p_args[3]);
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}

Error SceneDebugger::_msg_reload_cached_files(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	PackedStringArray files = p_args[0];
	reload_cached_files(files);
	return OK;
}

Error SceneDebugger::_msg_setup_embedded_shortcuts(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty() || p_args[0].get_type() != Variant::DICTIONARY, ERR_INVALID_DATA);
	Dictionary dict = p_args[0];
	LocalVector<Variant> keys = dict.get_key_list();

	for (const Variant &key : keys) {
		dict[key] = DebuggerMarshalls::deserialize_key_shortcut(dict[key]);
	}

	SceneTree::get_singleton()->get_root()->connect(SceneStringName(window_input), callable_mp_static(SceneDebugger::_handle_embed_input).bind(dict));
	return OK;
}

// region Live editing.

Error SceneDebugger::_msg_live_set_root(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_root_func(p_args[0], p_args[1]);
	return OK;
}

Error SceneDebugger::_msg_live_node_path(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_node_path_func(p_args[0], p_args[1]);
	return OK;
}

Error SceneDebugger::_msg_live_res_path(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_res_path_func(p_args[0], p_args[1]);
	return OK;
}

Error SceneDebugger::_msg_live_node_prop_res(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_node_set_res_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_node_prop(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_node_set_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_res_prop_res(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_res_set_res_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_res_prop(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_res_set_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_node_call(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LocalVector<Variant> args;
	LocalVector<Variant *> argptrs;
	args.resize(p_args.size() - 2);
	argptrs.resize(args.size());
	for (uint32_t i = 0; i < args.size(); i++) {
		args[i] = p_args[i + 2];
		argptrs[i] = &args[i];
	}
	LiveEditor::get_singleton()->_node_call_func(p_args[0], p_args[1], argptrs.size() ? (const Variant **)argptrs.ptr() : nullptr, argptrs.size());
	return OK;
}

Error SceneDebugger::_msg_live_res_call(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LocalVector<Variant> args;
	LocalVector<Variant *> argptrs;
	args.resize(p_args.size() - 2);
	argptrs.resize(args.size());
	for (uint32_t i = 0; i < args.size(); i++) {
		args[i] = p_args[i + 2];
		argptrs[i] = &args[i];
	}
	LiveEditor::get_singleton()->_res_call_func(p_args[0], p_args[1], argptrs.size() ? (const Variant **)argptrs.ptr() : nullptr, argptrs.size());
	return OK;
}

Error SceneDebugger::_msg_live_create_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_create_node_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_instantiate_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_instance_node_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_remove_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_remove_node_func(p_args[0]);
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}

Error SceneDebugger::_msg_live_remove_and_keep_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_remove_and_keep_node_func(p_args[0], p_args[1]);
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}

Error SceneDebugger::_msg_live_restore_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_restore_node_func(p_args[0], p_args[1], p_args[2]);
	return OK;
}

Error SceneDebugger::_msg_live_duplicate_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_duplicate_node_func(p_args[0], p_args[1]);
	return OK;
}

Error SceneDebugger::_msg_live_reparent_node(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 4, ERR_INVALID_DATA);
	LiveEditor::get_singleton()->_reparent_node_func(p_args[0], p_args[1], p_args[2], p_args[3]);
	return OK;
}

// endregion

// region Runtime Node Selection.

Error SceneDebugger::_msg_runtime_node_select_setup(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty() || p_args[0].get_type() != Variant::DICTIONARY, ERR_INVALID_DATA);
	RuntimeNodeSelect::get_singleton()->_setup(p_args[0]);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_set_type(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	RuntimeNodeSelect::NodeType type = (RuntimeNodeSelect::NodeType)(int)p_args[0];
	RuntimeNodeSelect::get_singleton()->_node_set_type(type);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_set_mode(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	RuntimeNodeSelect::SelectMode mode = (RuntimeNodeSelect::SelectMode)(int)p_args[0];
	RuntimeNodeSelect::get_singleton()->_select_set_mode(mode);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_set_visible(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool visible = p_args[0];
	RuntimeNodeSelect::get_singleton()->_set_selection_visible(visible);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_set_avoid_locked(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool avoid_locked = p_args[0];
	RuntimeNodeSelect::get_singleton()->_set_avoid_locked(avoid_locked);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_set_prefer_group(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	bool prefer_group = p_args[0];
	RuntimeNodeSelect::get_singleton()->_set_prefer_group(prefer_group);
	return OK;
}

Error SceneDebugger::_msg_runtime_node_select_reset_camera_2d(const Array &p_args) {
	RuntimeNodeSelect::get_singleton()->_reset_camera_2d();
	return OK;
}

Error SceneDebugger::_msg_transform_camera_2d(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!SceneTree::get_singleton()->get_root()->is_camera_2d_override_enabled(), ERR_BUG);
	Transform2D transform = p_args[0];
	Camera2D *override_camera = SceneTree::get_singleton()->get_root()->get_override_camera_2d();
	override_camera->set_offset(transform.affine_inverse().get_origin());
	override_camera->set_zoom(transform.get_scale());
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}

#ifndef _3D_DISABLED
Error SceneDebugger::_msg_runtime_node_select_reset_camera_3d(const Array &p_args) {
	RuntimeNodeSelect::get_singleton()->_reset_camera_3d();
	return OK;
}

Error SceneDebugger::_msg_transform_camera_3d(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.size() < 5, ERR_INVALID_DATA);
	ERR_FAIL_COND_V(!SceneTree::get_singleton()->get_root()->is_camera_3d_override_enabled(), ERR_BUG);
	Transform3D transform = p_args[0];
	bool is_perspective = p_args[1];
	float size_or_fov = p_args[2];
	float depth_near = p_args[3];
	float depth_far = p_args[4];

	Camera3D *override_camera = SceneTree::get_singleton()->get_root()->get_override_camera_3d();
	if (is_perspective) {
		override_camera->set_perspective(size_or_fov, depth_near, depth_far);
	} else {
		override_camera->set_orthogonal(size_or_fov, depth_near, depth_far);
	}
	override_camera->set_transform(transform);
	RuntimeNodeSelect::get_singleton()->_queue_selection_update();
	return OK;
}
#endif // _3D_DISABLED

// endregion

// region Embedded process screenshot.

Error SceneDebugger::_msg_rq_screenshot(const Array &p_args) {
	ERR_FAIL_COND_V(p_args.is_empty(), ERR_INVALID_DATA);

	Viewport *viewport = SceneTree::get_singleton()->get_root();
	ERR_FAIL_NULL_V_MSG(viewport, ERR_UNCONFIGURED, "Cannot get a viewport from the main screen.");
	Ref<ViewportTexture> texture = viewport->get_texture();
	ERR_FAIL_COND_V_MSG(texture.is_null(), ERR_UNCONFIGURED, "Cannot get a viewport texture from the main screen.");
	Ref<Image> img = texture->get_image();
	ERR_FAIL_COND_V_MSG(img.is_null(), ERR_UNCONFIGURED, "Cannot get an image from a viewport texture of the main screen.");
	img->clear_mipmaps();

	const String TEMP_DIR = OS::get_singleton()->get_temp_path();
	uint32_t suffix_i = 0;
	String path;
	while (true) {
		String datetime = Time::get_singleton()->get_datetime_string_from_system().remove_chars("-T:");
		datetime += itos(Time::get_singleton()->get_ticks_usec());
		String suffix = datetime + (suffix_i > 0 ? itos(suffix_i) : "");
		path = TEMP_DIR.path_join("scr-" + suffix + ".png");
		if (!DirAccess::exists(path)) {
			break;
		}
		suffix_i += 1;
	}
	img->save_png(path);

	Array arr;
	arr.append(p_args[0]);
	arr.append(img->get_width());
	arr.append(img->get_height());
	arr.append(path);
	EngineDebugger::get_singleton()->send_message("game_view:get_screenshot", arr);

	return OK;
}

// endregion

HashMap<String, SceneDebugger::ParseMessageFunc> SceneDebugger::message_handlers;

Error SceneDebugger::parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	ERR_FAIL_NULL_V(SceneTree::get_singleton(), ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(LiveEditor::get_singleton(), ERR_UNCONFIGURED);
	ERR_FAIL_NULL_V(RuntimeNodeSelect::get_singleton(), ERR_UNCONFIGURED);

	ParseMessageFunc *fn_ptr = message_handlers.getptr(p_msg);
	if (fn_ptr) {
		r_captured = true;
		return (*fn_ptr)(p_args);
	}

	if (p_msg.begins_with("live_") || p_msg.begins_with("runtime_node_select_")) {
		// Messages with these prefixes are reserved and should be handled by the LiveEditor or RuntimeNodeSelect classes,
		// so return ERR_SKIP.
		r_captured = true;
		return ERR_SKIP;
	}

	r_captured = false;

	return OK;
}

void SceneDebugger::_init_message_handlers() {
	message_handlers["setup_scene"] = _msg_setup_scene;
	message_handlers["setup_embedded_shortcuts"] = _msg_setup_embedded_shortcuts;
	message_handlers["request_scene_tree"] = _msg_request_scene_tree;
	message_handlers["save_node"] = _msg_save_node;
	message_handlers["inspect_objects"] = _msg_inspect_objects;
#ifndef DISABLE_DEPRECATED
	message_handlers["inspect_object"] = _msg_inspect_object;
#endif // DISABLE_DEPRECATED
	message_handlers["clear_selection"] = _msg_clear_selection;
	message_handlers["suspend_changed"] = _msg_suspend_changed;
	message_handlers["next_frame"] = _msg_next_frame;
	message_handlers["speed_changed"] = _msg_speed_changed;
	message_handlers["debug_mute_audio"] = _msg_debug_mute_audio;
	message_handlers["override_cameras"] = _msg_override_cameras;
	message_handlers["transform_camera_2d"] = _msg_transform_camera_2d;
#ifndef _3D_DISABLED
	message_handlers["transform_camera_3d"] = _msg_transform_camera_3d;
#endif // _3D_DISABLED
	message_handlers["set_object_property"] = _msg_set_object_property;
	message_handlers["set_object_property_field"] = _msg_set_object_property_field;
	message_handlers["reload_cached_files"] = _msg_reload_cached_files;
	message_handlers["live_set_root"] = _msg_live_set_root;
	message_handlers["live_node_path"] = _msg_live_node_path;
	message_handlers["live_res_path"] = _msg_live_res_path;
	message_handlers["live_node_prop_res"] = _msg_live_node_prop_res;
	message_handlers["live_node_prop"] = _msg_live_node_prop;
	message_handlers["live_res_prop_res"] = _msg_live_res_prop_res;
	message_handlers["live_res_prop"] = _msg_live_res_prop;
	message_handlers["live_node_call"] = _msg_live_node_call;
	message_handlers["live_res_call"] = _msg_live_res_call;
	message_handlers["live_create_node"] = _msg_live_create_node;
	message_handlers["live_instantiate_node"] = _msg_live_instantiate_node;
	message_handlers["live_remove_node"] = _msg_live_remove_node;
	message_handlers["live_remove_and_keep_node"] = _msg_live_remove_and_keep_node;
	message_handlers["live_restore_node"] = _msg_live_restore_node;
	message_handlers["live_duplicate_node"] = _msg_live_duplicate_node;
	message_handlers["live_reparent_node"] = _msg_live_reparent_node;
	message_handlers["runtime_node_select_setup"] = _msg_runtime_node_select_setup;
	message_handlers["runtime_node_select_set_type"] = _msg_runtime_node_select_set_type;
	message_handlers["runtime_node_select_set_mode"] = _msg_runtime_node_select_set_mode;
	message_handlers["runtime_node_select_set_visible"] = _msg_runtime_node_select_set_visible;
	message_handlers["runtime_node_select_set_avoid_locked"] = _msg_runtime_node_select_set_avoid_locked;
	message_handlers["runtime_node_select_set_prefer_group"] = _msg_runtime_node_select_set_prefer_group;
	message_handlers["runtime_node_select_reset_camera_2d"] = _msg_runtime_node_select_reset_camera_2d;
#ifndef _3D_DISABLED
	message_handlers["runtime_node_select_reset_camera_3d"] = _msg_runtime_node_select_reset_camera_3d;
#endif
	message_handlers["rq_screenshot"] = _msg_rq_screenshot;
}

void SceneDebugger::_save_node(ObjectID id, const String &p_path) {
	Node *node = ObjectDB::get_instance<Node>(id);
	ERR_FAIL_NULL(node);

#ifdef TOOLS_ENABLED
	HashMap<const Node *, Node *> duplimap;
	Node *copy = node->duplicate_from_editor(duplimap);
#else
	Node *copy = node->duplicate();
#endif // TOOLS_ENABLED

	// Handle Unique Nodes.
	for (int i = 0; i < copy->get_child_count(false); i++) {
		_set_node_owner_recursive(copy->get_child(i, false), copy);
	}
	// Root node cannot ever be unique name in its own Scene!
	copy->set_unique_name_in_owner(false);

	Ref<PackedScene> ps = memnew(PackedScene);
	ps->pack(copy);
	ResourceSaver::save(ps, p_path);

	memdelete(copy);
}

void SceneDebugger::_set_node_owner_recursive(Node *p_node, Node *p_owner) {
	if (!p_node->get_owner()) {
		p_node->set_owner(p_owner);
	}

	for (int i = 0; i < p_node->get_child_count(false); i++) {
		_set_node_owner_recursive(p_node->get_child(i, false), p_owner);
	}
}

void SceneDebugger::_send_object_ids(const Vector<ObjectID> &p_ids, bool p_update_selection) {
	Vector<ObjectID> ids = p_ids;
	if (ids.size() > RuntimeNodeSelect::get_singleton()->max_selection) {
		ids.resize(RuntimeNodeSelect::get_singleton()->max_selection);
		EngineDebugger::get_singleton()->send_message("show_selection_limit_warning", Array());
	}

	LocalVector<Node *> nodes;
	Array objs;
	bool objs_missing = false;
	for (const ObjectID &id : ids) {
		SceneDebuggerObject obj(id);
		if (obj.id.is_null()) {
			objs_missing = true;
			continue;
		}

		if (p_update_selection) {
			if (Node *node = ObjectDB::get_instance<Node>(id)) {
				nodes.push_back(node);
			}
		}

		Array arr;
		obj.serialize(arr);
		objs.append(arr);
	}

	if (p_update_selection) {
		RuntimeNodeSelect::get_singleton()->_set_selected_nodes(Vector<Node *>(nodes));
	}

	if (objs_missing) {
		Array invalid_selection;
		for (const ObjectID &id : ids) {
			invalid_selection.append(id);
		}

		Array arr;
		arr.append(invalid_selection);
		EngineDebugger::get_singleton()->send_message("remote_selection_invalidated", arr);

		EngineDebugger::get_singleton()->send_message(objs.is_empty() ? "remote_nothing_selected" : "remote_objects_selected", objs);
	} else {
		EngineDebugger::get_singleton()->send_message(p_update_selection ? "remote_objects_selected" : "scene:inspect_objects", objs);
	}
}

void SceneDebugger::_set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value, const String &p_field) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	String prop_name;
	if (p_property.begins_with("Members/")) {
		prop_name = p_property.get_slicec('/', p_property.get_slice_count("/") - 1);
	} else {
		prop_name = p_property;
	}

	Variant value = p_value;
	if (p_value.is_string() && (obj->get_static_property_type(prop_name) == Variant::OBJECT || p_property == "script")) {
		value = ResourceLoader::load(p_value);
	}

	if (!p_field.is_empty()) {
		// Only one specific field.
		value = fieldwise_assign(obj->get(prop_name), value, p_field);
	}

	obj->set(prop_name, value);
}

void SceneDebugger::_next_frame() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree->is_suspended()) {
		return;
	}

	scene_tree->set_suspend(false);
	RenderingServer::get_singleton()->connect("frame_post_draw", callable_mp(scene_tree, &SceneTree::set_suspend).bind(true), Object::CONNECT_ONE_SHOT);
}

void SceneDebugger::add_to_cache(const String &p_filename, Node *p_node) {
	LiveEditor *debugger = LiveEditor::get_singleton();
	if (!debugger) {
		return;
	}

	if (EngineDebugger::get_script_debugger() && !p_filename.is_empty()) {
		debugger->live_scene_edit_cache[p_filename].insert(p_node);
	}
}

void SceneDebugger::remove_from_cache(const String &p_filename, Node *p_node) {
	LiveEditor *debugger = LiveEditor::get_singleton();
	if (!debugger) {
		return;
	}

	HashMap<String, HashSet<Node *>> &edit_cache = debugger->live_scene_edit_cache;
	HashMap<String, HashSet<Node *>>::Iterator E = edit_cache.find(p_filename);
	if (E) {
		E->value.erase(p_node);
		if (E->value.is_empty()) {
			edit_cache.remove(E);
		}
	}

	HashMap<Node *, HashMap<ObjectID, Node *>> &remove_list = debugger->live_edit_remove_list;
	HashMap<Node *, HashMap<ObjectID, Node *>>::Iterator F = remove_list.find(p_node);
	if (F) {
		for (const KeyValue<ObjectID, Node *> &G : F->value) {
			memdelete(G.value);
		}
		remove_list.remove(F);
	}
}

void SceneDebugger::reload_cached_files(const PackedStringArray &p_files) {
	for (const String &file : p_files) {
		Ref<Resource> res = ResourceCache::get_ref(file);
		if (res.is_valid()) {
			res->reload_from_file();
		}
	}
}

SceneDebuggerObject::SceneDebuggerObject(ObjectID p_id) :
		SceneDebuggerObject(ObjectDB::get_instance(p_id)) {
}

/// LiveEditor
LiveEditor *LiveEditor::get_singleton() {
	return singleton;
}

void LiveEditor::_send_tree() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Array arr;
	// Encoded as a flat list depth first.
	SceneDebuggerTree tree(scene_tree->root);
	tree.serialize(arr);
	EngineDebugger::get_singleton()->send_message("scene:scene_tree", arr);
}

void LiveEditor::_node_path_func(const NodePath &p_path, int p_id) {
	live_edit_node_path_cache[p_id] = p_path;
}

void LiveEditor::_res_path_func(const String &p_path, int p_id) {
	live_edit_resource_cache[p_id] = p_path;
}

void LiveEditor::_node_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	if (!live_edit_node_path_cache.has(p_id)) {
		return;
	}

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(np)) {
			continue;
		}
		Node *n2 = n->get_node(np);

		// Do not change transform of edited scene root, unless it's the scene being played.
		// See GH-86659 for additional context.
		bool keep_transform = (n2 == n) && (n2->get_parent() != scene_tree->root);
		Variant orig_tf;

		if (keep_transform) {
			if (n2->is_class("Node3D")) {
				orig_tf = n2->call("get_transform");
			} else if (n2->is_class("CanvasItem")) {
				orig_tf = n2->call("_edit_get_state");
			}
		}

		n2->set(p_prop, p_value);

		if (keep_transform) {
			if (n2->is_class("Node3D")) {
				Variant new_tf = n2->call("get_transform");
				if (new_tf != orig_tf) {
					n2->call("set_transform", orig_tf);
				}
			} else if (n2->is_class("CanvasItem")) {
				Variant new_tf = n2->call("_edit_get_state");
				if (new_tf != orig_tf) {
					n2->call("_edit_set_state", orig_tf);
				}
			}
		}
	}
}

void LiveEditor::_node_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {
	Ref<Resource> r = ResourceLoader::load(p_value);
	if (r.is_null()) {
		return;
	}
	_node_set_func(p_id, p_prop, r);
}

void LiveEditor::_node_call_func(int p_id, const StringName &p_method, const Variant **p_args, int p_argcount) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}
	if (!live_edit_node_path_cache.has(p_id)) {
		return;
	}

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(np)) {
			continue;
		}
		Node *n2 = n->get_node(np);

		// Do not change transform of edited scene root, unless it's the scene being played.
		// See GH-86659 for additional context.
		bool keep_transform = (n2 == n) && (n2->get_parent() != scene_tree->root);
		Variant orig_tf;

		if (keep_transform) {
			if (n2->is_class("Node3D")) {
				orig_tf = n2->call("get_transform");
			} else if (n2->is_class("CanvasItem")) {
				orig_tf = n2->call("_edit_get_state");
			}
		}

		Callable::CallError ce;
		n2->callp(p_method, p_args, p_argcount, ce);

		if (keep_transform) {
			if (n2->is_class("Node3D")) {
				Variant new_tf = n2->call("get_transform");
				if (new_tf != orig_tf) {
					n2->call("set_transform", orig_tf);
				}
			} else if (n2->is_class("CanvasItem")) {
				Variant new_tf = n2->call("_edit_get_state");
				if (new_tf != orig_tf) {
					n2->call("_edit_set_state", orig_tf);
				}
			}
		}
	}
}

void LiveEditor::_res_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {
	if (!live_edit_resource_cache.has(p_id)) {
		return;
	}

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp)) {
		return;
	}

	Ref<Resource> r = ResourceCache::get_ref(resp);
	if (r.is_null()) {
		return;
	}

	r->set(p_prop, p_value);
}

void LiveEditor::_res_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {
	Ref<Resource> r = ResourceLoader::load(p_value);
	if (r.is_null()) {
		return;
	}
	_res_set_func(p_id, p_prop, r);
}

void LiveEditor::_res_call_func(int p_id, const StringName &p_method, const Variant **p_args, int p_argcount) {
	if (!live_edit_resource_cache.has(p_id)) {
		return;
	}

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp)) {
		return;
	}

	Ref<Resource> r = ResourceCache::get_ref(resp);
	if (r.is_null()) {
		return;
	}

	Callable::CallError ce;
	r->callp(p_method, p_args, p_argcount, ce);
}

void LiveEditor::_root_func(const NodePath &p_scene_path, const String &p_scene_from) {
	live_edit_root = p_scene_path;
	live_edit_scene = p_scene_from;
}

void LiveEditor::_create_node_func(const NodePath &p_parent, const String &p_type, const String &p_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_parent)) {
			continue;
		}
		Node *n2 = n->get_node(p_parent);

		Node *no = Object::cast_to<Node>(ClassDB::instantiate(p_type));
		if (!no) {
			continue;
		}

		no->set_name(p_name);
		n2->add_child(no);
	}
}

void LiveEditor::_instance_node_func(const NodePath &p_parent, const String &p_path, const String &p_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Ref<PackedScene> ps = ResourceLoader::load(p_path);

	if (ps.is_null()) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_parent)) {
			continue;
		}
		Node *n2 = n->get_node(p_parent);

		Node *no = ps->instantiate();
		if (!no) {
			continue;
		}

		no->set_name(p_name);
		n2->add_child(no);
	}
}

void LiveEditor::_remove_node_func(const NodePath &p_at) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	LocalVector<Node *> to_delete;

	for (const Node *n : E->value) {
		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		to_delete.push_back(n2);
	}

	for (Node *node : to_delete) {
		memdelete(node);
	}
}

void LiveEditor::_remove_and_keep_node_func(const NodePath &p_at, ObjectID p_keep_id) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	LocalVector<Node *> to_remove;
	for (Node *n : E->value) {
		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}

		to_remove.push_back(n);
	}

	for (Node *n : to_remove) {
		Node *n2 = n->get_node(p_at);
		n2->get_parent()->remove_child(n2);
		live_edit_remove_list[n][p_keep_id] = n2;
	}
}

void LiveEditor::_restore_node_func(ObjectID p_id, const NodePath &p_at, int p_at_pos) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (HashSet<Node *>::Iterator F = E->value.begin(); F;) {
		HashSet<Node *>::Iterator N = F;
		++N;

		Node *n = *F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		HashMap<Node *, HashMap<ObjectID, Node *>>::Iterator EN = live_edit_remove_list.find(n);

		if (!EN) {
			continue;
		}

		HashMap<ObjectID, Node *>::Iterator FN = EN->value.find(p_id);

		if (!FN) {
			continue;
		}
		n2->add_child(FN->value);

		EN->value.remove(FN);

		if (EN->value.is_empty()) {
			live_edit_remove_list.remove(EN);
		}

		F = N;
	}
}

void LiveEditor::_duplicate_node_func(const NodePath &p_at, const String &p_new_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		Node *dup = n2->duplicate(Node::DUPLICATE_SIGNALS | Node::DUPLICATE_GROUPS | Node::DUPLICATE_SCRIPTS);

		if (!dup) {
			continue;
		}

		dup->set_name(p_new_name);
		n2->get_parent()->add_child(dup);
	}
}

void LiveEditor::_reparent_node_func(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	HashMap<String, HashSet<Node *>>::Iterator E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Node *F : E->value) {
		Node *n = F;

		if (base && !base->is_ancestor_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *nfrom = n->get_node(p_at);

		if (!n->has_node(p_new_place)) {
			continue;
		}
		Node *nto = n->get_node(p_new_place);

		nfrom->get_parent()->remove_child(nfrom);
		nfrom->set_name(p_new_name);

		nto->add_child(nfrom);
		if (p_at_pos >= 0) {
			nto->move_child(nfrom, p_at_pos);
		}
	}
}
#endif // DEBUG_ENABLED
