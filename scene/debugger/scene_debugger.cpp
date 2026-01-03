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

#include "core/config/project_settings.h"
#include "core/debugger/debugger_marshalls.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/dir_access.h"
#include "core/io/marshalls.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/math/math_fieldwise.h"
#include "core/object/script_language.h"
#include "core/os/time.h"
#include "core/templates/local_vector.h"
#include "scene/2d/camera_2d.h"
#include "scene/gui/popup_menu.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"
#include "scene/theme/theme_db.h"
#include "servers/audio/audio_server.h"

#ifndef PHYSICS_2D_DISABLED
#include "scene/2d/physics/collision_object_2d.h"
#include "scene/2d/physics/collision_polygon_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#endif // PHYSICS_2D_DISABLED

#ifndef _3D_DISABLED
#include "scene/3d/camera_3d.h"
#ifndef PHYSICS_3D_DISABLED
#include "scene/3d/physics/collision_object_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#endif // PHYSICS_3D_DISABLED
#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/surface_tool.h"
#endif // _3D_DISABLED

SceneDebugger::SceneDebugger() {
	singleton = this;

#ifdef DEBUG_ENABLED
	LiveEditor::singleton = memnew(LiveEditor);
	RuntimeNodeSelect::singleton = memnew(RuntimeNodeSelect);

	EngineDebugger::register_message_capture("scene", EngineDebugger::Capture(nullptr, SceneDebugger::parse_message));
#endif // DEBUG_ENABLED
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
#ifndef _2D_DISABLED
	SceneTree::get_singleton()->get_root()->enable_camera_2d_override(enable);
#endif // _2D_DISABLED
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

#ifndef _2D_DISABLED
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
#endif // _2D_DISABLED

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
#ifndef _2D_DISABLED
	message_handlers["transform_camera_2d"] = _msg_transform_camera_2d;
#endif // _2D_DISABLED
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
#ifndef _2D_DISABLED
	message_handlers["runtime_node_select_reset_camera_2d"] = _msg_runtime_node_select_reset_camera_2d;
#endif // _2D_DISABLED
#ifndef _3D_DISABLED
	message_handlers["runtime_node_select_reset_camera_3d"] = _msg_runtime_node_select_reset_camera_3d;
#endif // _3D_DISABLED
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

/// SceneDebuggerObject
SceneDebuggerObject::SceneDebuggerObject(Object *p_obj) {
	if (!p_obj) {
		return;
	}

	id = p_obj->get_instance_id();
	class_name = p_obj->get_class();

	if (ScriptInstance *si = p_obj->get_script_instance()) {
		// Read script instance constants and variables.
		if (!si->get_script().is_null()) {
			Script *s = si->get_script().ptr();
			_parse_script_properties(s, si);
		}
	}

	if (Node *node = Object::cast_to<Node>(p_obj)) {
		// For debugging multiplayer.
		{
			PropertyInfo pi(Variant::INT, String("Node/multiplayer_authority"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_READ_ONLY);
			properties.push_back(SceneDebuggerProperty(pi, node->get_multiplayer_authority()));
		}

		// Add specialized NodePath info (if inside tree).
		if (node->is_inside_tree()) {
			PropertyInfo pi(Variant::NODE_PATH, String("Node/path"));
			properties.push_back(SceneDebuggerProperty(pi, node->get_path()));
		} else { // Can't ask for path if a node is not in tree.
			PropertyInfo pi(Variant::STRING, String("Node/path"));
			properties.push_back(SceneDebuggerProperty(pi, "[Orphan]"));
		}
	} else if (Script *s = Object::cast_to<Script>(p_obj)) {
		// Add script constants (no instance).
		_parse_script_properties(s, nullptr);
	}

	// Add base object properties.
	List<PropertyInfo> pinfo;
	p_obj->get_property_list(&pinfo, true);
	for (const PropertyInfo &E : pinfo) {
		if (E.usage & (PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CATEGORY)) {
			properties.push_back(SceneDebuggerProperty(E, p_obj->get(E.name)));
		}
	}
}

void SceneDebuggerObject::_parse_script_properties(Script *p_script, ScriptInstance *p_instance) {
	typedef HashMap<const Script *, HashSet<StringName>> ScriptMemberMap;
	typedef HashMap<const Script *, HashMap<StringName, Variant>> ScriptConstantsMap;

	ScriptMemberMap members;
	if (p_instance) {
		members[p_script] = HashSet<StringName>();
		p_script->get_members(&(members[p_script]));
	}

	ScriptConstantsMap constants;
	constants[p_script] = HashMap<StringName, Variant>();
	p_script->get_constants(&(constants[p_script]));

	Ref<Script> base = p_script->get_base_script();
	while (base.is_valid()) {
		if (p_instance) {
			members[base.ptr()] = HashSet<StringName>();
			base->get_members(&(members[base.ptr()]));
		}

		constants[base.ptr()] = HashMap<StringName, Variant>();
		base->get_constants(&(constants[base.ptr()]));

		base = base->get_base_script();
	}

	HashSet<String> exported_members;

	if (p_instance) {
		List<PropertyInfo> pinfo;
		p_instance->get_property_list(&pinfo);
		for (const PropertyInfo &E : pinfo) {
			if (E.usage & (PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CATEGORY)) {
				exported_members.insert(E.name);
			}
		}
	}

	// Members
	for (KeyValue<const Script *, HashSet<StringName>> sm : members) {
		for (const StringName &E : sm.value) {
			if (exported_members.has(E)) {
				continue; // Exported variables already show up in the inspector.
			}
			if (String(E).begins_with("@")) {
				continue; // Skip groups.
			}

			Variant m;
			if (p_instance->get(E, m)) {
				String script_path = sm.key == p_script ? "" : sm.key->get_path().get_file() + "/";
				PropertyInfo pi(m.get_type(), "Members/" + script_path + E);
				properties.push_back(SceneDebuggerProperty(pi, m));
			}
		}
	}
	// Constants
	for (KeyValue<const Script *, HashMap<StringName, Variant>> &sc : constants) {
		for (const KeyValue<StringName, Variant> &E : sc.value) {
			String script_path = sc.key == p_script ? "" : sc.key->get_path().get_file() + "/";
			if (E.value.get_type() == Variant::OBJECT) {
				Variant inst_id = ((Object *)E.value)->get_instance_id();
				PropertyInfo pi(inst_id.get_type(), "Constants/" + E.key, PROPERTY_HINT_OBJECT_ID, "Object");
				properties.push_back(SceneDebuggerProperty(pi, inst_id));
			} else {
				PropertyInfo pi(E.value.get_type(), "Constants/" + script_path + E.key);
				properties.push_back(SceneDebuggerProperty(pi, E.value));
			}
		}
	}
}

void SceneDebuggerObject::serialize(Array &r_arr, int p_max_size) {
	Array send_props;
	for (SceneDebuggerProperty &property : properties) {
		const PropertyInfo &pi = property.first;
		Variant &var = property.second;

		Ref<Resource> res = var;

		Array prop = { pi.name, pi.type };
		PropertyHint hint = pi.hint;
		String hint_string = pi.hint_string;
		if (res.is_valid() && !res->get_path().is_empty()) {
			var = res->get_path();
		} else { //only send information that can be sent..
			int len = 0; //test how big is this to encode
			encode_variant(var, nullptr, len);
			if (len > p_max_size) { //limit to max size
				hint = PROPERTY_HINT_OBJECT_TOO_BIG;
				hint_string = "";
				var = Variant();
			}
		}
		prop.push_back(hint);
		prop.push_back(hint_string);
		prop.push_back(pi.usage);
		prop.push_back(var);
		send_props.push_back(prop);
	}
	r_arr.push_back(uint64_t(id));
	r_arr.push_back(class_name);
	r_arr.push_back(send_props);
}

void SceneDebuggerObject::deserialize(const Array &p_arr) {
#define CHECK_TYPE(p_what, p_type) ERR_FAIL_COND(p_what.get_type() != Variant::p_type);
	ERR_FAIL_COND(p_arr.size() < 3);
	CHECK_TYPE(p_arr[0], INT);
	CHECK_TYPE(p_arr[1], STRING);
	CHECK_TYPE(p_arr[2], ARRAY);

	deserialize(uint64_t(p_arr[0]), p_arr[1], p_arr[2]);
}

void SceneDebuggerObject::deserialize(uint64_t p_id, const String &p_class_name, const Array &p_props) {
	id = p_id;
	class_name = p_class_name;

	for (int i = 0; i < p_props.size(); i++) {
		CHECK_TYPE(p_props[i], ARRAY);
		Array prop = p_props[i];

		ERR_FAIL_COND(prop.size() != 6);
		CHECK_TYPE(prop[0], STRING);
		CHECK_TYPE(prop[1], INT);
		CHECK_TYPE(prop[2], INT);
		CHECK_TYPE(prop[3], STRING);
		CHECK_TYPE(prop[4], INT);

		PropertyInfo pinfo;
		pinfo.name = prop[0];
		pinfo.type = Variant::Type(int(prop[1]));
		pinfo.hint = PropertyHint(int(prop[2]));
		pinfo.hint_string = prop[3];
		pinfo.usage = PropertyUsageFlags(int(prop[4]));
		Variant var = prop[5];

		if (pinfo.type == Variant::OBJECT) {
			if (var.is_zero()) {
				var = Ref<Resource>();
			} else if (var.get_type() == Variant::OBJECT) {
				if (((Object *)var)->is_class("EncodedObjectAsID")) {
					var = Object::cast_to<EncodedObjectAsID>(var)->get_object_id();
					pinfo.type = var.get_type();
					pinfo.hint = PROPERTY_HINT_OBJECT_ID;
					pinfo.hint_string = "Object";
				}
			}
		}
		properties.push_back(SceneDebuggerProperty(pinfo, var));
	}
}

/// SceneDebuggerTree
SceneDebuggerTree::SceneDebuggerTree(Node *p_root) {
	// Flatten tree into list, depth first, use stack to avoid recursion.
	List<Node *> stack;
	stack.push_back(p_root);
	bool is_root = true;
	const StringName &is_visible_sn = SNAME("is_visible");
	const StringName &is_visible_in_tree_sn = SNAME("is_visible_in_tree");
	while (stack.size()) {
		Node *n = stack.front()->get();
		stack.pop_front();

		int count = n->get_child_count();
		for (int i = 0; i < count; i++) {
			stack.push_front(n->get_child(count - i - 1));
		}

		int view_flags = 0;
		if (is_root) {
			// Prevent root window visibility from being changed.
			is_root = false;
		} else if (n->has_method(is_visible_sn)) {
			const Variant visible = n->call(is_visible_sn);
			if (visible.get_type() == Variant::BOOL) {
				view_flags = RemoteNode::VIEW_HAS_VISIBLE_METHOD;
				view_flags |= uint8_t(visible) * RemoteNode::VIEW_VISIBLE;
			}
			if (n->has_method(is_visible_in_tree_sn)) {
				const Variant visible_in_tree = n->call(is_visible_in_tree_sn);
				if (visible_in_tree.get_type() == Variant::BOOL) {
					view_flags |= uint8_t(visible_in_tree) * RemoteNode::VIEW_VISIBLE_IN_TREE;
				}
			}
		}

		String class_name;
		ScriptInstance *script_instance = n->get_script_instance();
		if (script_instance) {
			Ref<Script> script = script_instance->get_script();
			if (script.is_valid()) {
				class_name = script->get_global_name();

				if (class_name.is_empty()) {
					// If there is no class_name in this script we just take the script path.
					class_name = script->get_path();
				}
			}
		}
		nodes.push_back(RemoteNode(count, n->get_name(), class_name.is_empty() ? n->get_class() : class_name, n->get_instance_id(), n->get_scene_file_path(), view_flags));
	}
}

void SceneDebuggerTree::serialize(Array &p_arr) {
	for (const RemoteNode &n : nodes) {
		p_arr.push_back(n.child_count);
		p_arr.push_back(n.name);
		p_arr.push_back(n.type_name);
		p_arr.push_back(n.id);
		p_arr.push_back(n.scene_file_path);
		p_arr.push_back(n.view_flags);
	}
}

void SceneDebuggerTree::deserialize(const Array &p_arr) {
	int idx = 0;
	while (p_arr.size() > idx) {
		ERR_FAIL_COND(p_arr.size() < 6);
		CHECK_TYPE(p_arr[idx], INT); // child_count.
		CHECK_TYPE(p_arr[idx + 1], STRING); // name.
		CHECK_TYPE(p_arr[idx + 2], STRING); // type_name.
		CHECK_TYPE(p_arr[idx + 3], INT); // id.
		CHECK_TYPE(p_arr[idx + 4], STRING); // scene_file_path.
		CHECK_TYPE(p_arr[idx + 5], INT); // view_flags.
		nodes.push_back(RemoteNode(p_arr[idx], p_arr[idx + 1], p_arr[idx + 2], p_arr[idx + 3], p_arr[idx + 4], p_arr[idx + 5]));
		idx += 6;
	}
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

/// RuntimeNodeSelect
RuntimeNodeSelect *RuntimeNodeSelect::get_singleton() {
	return singleton;
}

RuntimeNodeSelect::~RuntimeNodeSelect() {
	if (selection_list && !selection_list->is_visible()) {
		memdelete(selection_list);
	}

	if (draw_canvas.is_valid()) {
		RS::get_singleton()->free_rid(sel_drag_ci);
		RS::get_singleton()->free_rid(sbox_2d_ci);
		RS::get_singleton()->free_rid(draw_canvas);
	}
}

void RuntimeNodeSelect::_setup(const Dictionary &p_settings) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(root->is_connected(SceneStringName(window_input), callable_mp(this, &RuntimeNodeSelect::_root_window_input)));

	root->connect(SceneStringName(window_input), callable_mp(this, &RuntimeNodeSelect::_root_window_input));
	root->connect("size_changed", callable_mp(this, &RuntimeNodeSelect::_queue_selection_update), CONNECT_DEFERRED);

	max_selection = p_settings.get("debugger/max_node_selection", 1);

	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &RuntimeNodeSelect::_pan_callback), callable_mp(this, &RuntimeNodeSelect::_zoom_callback));

	ViewPanner::ControlScheme panning_scheme = (ViewPanner::ControlScheme)p_settings.get("editors/panning/2d_editor_panning_scheme", 0).operator int();
	bool simple_panning = p_settings.get("editors/panning/simple_panning", false);
	int pan_speed = p_settings.get("editors/panning/2d_editor_pan_speed", 20);
	Array keys = p_settings.get("canvas_item_editor/pan_view", Array()).operator Array();
	panner->setup(panning_scheme, DebuggerMarshalls::deserialize_key_shortcut(keys), simple_panning);
	panner->setup_warped_panning(root, p_settings.get("editors/panning/warped_mouse_panning", true));
	panner->set_scroll_speed(pan_speed);

	sel_2d_grab_dist = p_settings.get("editors/polygon_editor/point_grab_radius", 0);
	sel_2d_scale = MAX(1, Math::ceil(2.0 / (float)GLOBAL_GET("display/window/stretch/scale")));

	selection_area_fill = p_settings.get("box_selection_fill_color", Color());
	selection_area_outline = p_settings.get("box_selection_stroke_color", Color());

	draw_canvas = RS::get_singleton()->canvas_create();
	sel_drag_ci = RS::get_singleton()->canvas_item_create();

	/// 2D Selection Box Generation

	sbox_2d_ci = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->viewport_attach_canvas(root->get_viewport_rid(), draw_canvas);
	RS::get_singleton()->canvas_item_set_parent(sel_drag_ci, draw_canvas);
	RS::get_singleton()->canvas_item_set_parent(sbox_2d_ci, draw_canvas);

#ifndef _3D_DISABLED
	cursor = Cursor();

	camera_fov = p_settings.get("editors/3d/default_fov", 70);
	camera_znear = p_settings.get("editors/3d/default_z_near", 0.05);
	camera_zfar = p_settings.get("editors/3d/default_z_far", 4'000);

	invert_x_axis = p_settings.get("editors/3d/navigation/invert_x_axis", false);
	invert_y_axis = p_settings.get("editors/3d/navigation/invert_y_axis", false);
	warped_mouse_panning_3d = p_settings.get("editors/3d/navigation/warped_mouse_panning", true);

	freelook_base_speed = p_settings.get("editors/3d/freelook/freelook_base_speed", 5);
	freelook_sensitivity = Math::deg_to_rad((real_t)p_settings.get("editors/3d/freelook/freelook_sensitivity", 0.25));
	orbit_sensitivity = Math::deg_to_rad((real_t)p_settings.get("editors/3d/navigation_feel/orbit_sensitivity", 0.004));
	translation_sensitivity = p_settings.get("editors/3d/navigation_feel/translation_sensitivity", 1);

	/// 3D Selection Box Generation
	// Copied from the Node3DEditor implementation.

	sbox_3d_color = p_settings.get("editors/3d/selection_box_color", Color());

	// Use two AABBs to create the illusion of a slightly thicker line.
	AABB aabb(Vector3(), Vector3(1, 1, 1));

	// Create a x-ray (visible through solid surfaces) and standard version of the selection box.
	// Both will be drawn at the same position, but with different opacity.
	// This lets the user see where the selection is while still having a sense of depth.
	Ref<SurfaceTool> st = memnew(SurfaceTool);
	Ref<SurfaceTool> st_xray = memnew(SurfaceTool);

	st->begin(Mesh::PRIMITIVE_LINES);
	st_xray->begin(Mesh::PRIMITIVE_LINES);
	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);

		st->add_vertex(a);
		st->add_vertex(b);
		st_xray->add_vertex(a);
		st_xray->add_vertex(b);
	}

	Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	mat->set_albedo(sbox_3d_color);
	mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st->set_material(mat);
	sbox_3d_mesh = st->commit();

	Ref<StandardMaterial3D> mat_xray = memnew(StandardMaterial3D);
	mat_xray->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	mat_xray->set_albedo(sbox_3d_color * Color(1, 1, 1, 0.15));
	mat_xray->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st_xray->set_material(mat_xray);
	sbox_3d_mesh_xray = st_xray->commit();
#endif // _3D_DISABLED

	SceneTree::get_singleton()->connect("process_frame", callable_mp(this, &RuntimeNodeSelect::_process_frame));
	SceneTree::get_singleton()->connect("physics_frame", callable_mp(this, &RuntimeNodeSelect::_physics_frame));

	// This function will be called before the root enters the tree at first when the Game view is passing its settings to
	// the debugger, so queue the update for after it enters.
	root->connect(SceneStringName(tree_entered), callable_mp(this, &RuntimeNodeSelect::_update_input_state), Object::CONNECT_ONE_SHOT);
}

void RuntimeNodeSelect::_node_set_type(NodeType p_type) {
	node_select_type = p_type;
	_update_input_state();
}

void RuntimeNodeSelect::_select_set_mode(SelectMode p_mode) {
	node_select_mode = p_mode;
}

void RuntimeNodeSelect::_set_camera_override_enabled(bool p_enabled) {
	camera_override = p_enabled;

	if (camera_first_override) {
#ifndef _2D_DISABLED
		_reset_camera_2d();
#endif // _2D_DISABLED
#ifndef _3D_DISABLED
		_reset_camera_3d();
#endif // _3D_DISABLED

		camera_first_override = false;
	} else if (p_enabled) {
#ifndef _2D_DISABLED
		_update_view_2d();
#endif // _2D_DISABLED

#ifndef _3D_DISABLED
		Window *root = SceneTree::get_singleton()->get_root();
		ERR_FAIL_COND(!root->is_camera_3d_override_enabled());
		Camera3D *override_camera = root->get_override_camera_3d();
		override_camera->set_transform(_get_cursor_transform());
		override_camera->set_perspective(camera_fov * cursor.fov_scale, camera_znear, camera_zfar);
#endif // _3D_DISABLED
	}
}

void RuntimeNodeSelect::_root_window_input(const Ref<InputEvent> &p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	if (node_select_type == NODE_TYPE_NONE || (selection_list && selection_list->is_visible())) {
		// Workaround for platforms that don't allow subwindows.
		if (selection_list && selection_list->is_visible() && selection_list->is_embedded()) {
			root->set_disable_input_override(false);
			selection_list->push_input(p_event);
			callable_mp(root->get_viewport(), &Viewport::set_disable_input_override).call_deferred(true);
		}

		return;
	}

	bool is_dragging_camera = false;
	if (camera_override) {
		if (node_select_type == NODE_TYPE_2D) {
			is_dragging_camera = panner->gui_input(p_event, Rect2(Vector2(), root->get_visible_rect().get_size()));
#ifndef _3D_DISABLED
		} else if (node_select_type == NODE_TYPE_3D && selection_drag_state == SELECTION_DRAG_NONE) {
			if (_handle_3d_input(p_event)) {
				return;
			}
#endif // _3D_DISABLED
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (selection_drag_state == SELECTION_DRAG_MOVE) {
		Ref<InputEventMouseMotion> m = p_event;
		if (m.is_valid()) {
			_update_selection_drag(root->get_screen_transform().affine_inverse().xform(m->get_position()));
			return;
		} else if (b.is_valid()) {
			// Account for actions like zooming.
			_update_selection_drag(root->get_screen_transform().affine_inverse().xform(b->get_position()));
		}
	}

	if (b.is_null()) {
		return;
	}

	// Ignore mouse wheel inputs.
	if (b->get_button_index() != MouseButton::LEFT && b->get_button_index() != MouseButton::RIGHT) {
		return;
	}

	if (selection_drag_state == SELECTION_DRAG_MOVE && !b->is_pressed() && b->get_button_index() == MouseButton::LEFT) {
		selection_drag_state = SELECTION_DRAG_END;
		selection_drag_area = selection_drag_area.abs();
		_update_selection_drag();

		// Trigger a selection in the position on release.
		if (multi_shortcut_pressed) {
			selection_position = root->get_screen_transform().affine_inverse().xform(b->get_position());
		}
	}

	if (!is_dragging_camera && b->is_pressed()) {
		multi_shortcut_pressed = b->is_shift_pressed();
		list_shortcut_pressed = node_select_mode == SELECT_MODE_SINGLE && b->get_button_index() == MouseButton::RIGHT && b->is_alt_pressed();
		if (list_shortcut_pressed || b->get_button_index() == MouseButton::LEFT) {
			selection_position = root->get_screen_transform().affine_inverse().xform(b->get_position());
		}
	}
}

void RuntimeNodeSelect::_items_popup_index_pressed(int p_index, PopupMenu *p_popup) {
	Object *obj = p_popup->get_item_metadata(p_index).get_validated_object();
	if (obj) {
		Vector<Node *> node;
		node.append(Object::cast_to<Node>(obj));
		_send_ids(node);
	}
}

void RuntimeNodeSelect::_update_input_state() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	// This function can be called at the very beginning, when the root hasn't entered the tree yet.
	// So check first to avoid a crash.
	if (!scene_tree->get_root()->is_inside_tree()) {
		return;
	}

	bool disable_input = scene_tree->is_suspended() || node_select_type != RuntimeNodeSelect::NODE_TYPE_NONE;
	Input::get_singleton()->set_disable_input(disable_input);
	Input::get_singleton()->set_mouse_mode_override_enabled(disable_input);
	scene_tree->get_root()->set_disable_input_override(disable_input);
}

void RuntimeNodeSelect::_process_frame() {
#ifndef _3D_DISABLED
	if (camera_freelook) {
		Transform3D transform = _get_cursor_transform();
		Vector3 forward = transform.basis.xform(Vector3(0, 0, -1));
		const Vector3 right = transform.basis.xform(Vector3(1, 0, 0));
		Vector3 up = transform.basis.xform(Vector3(0, 1, 0));

		Vector3 direction;

		Input *input = Input::get_singleton();
		bool was_input_disabled = input->is_input_disabled();
		if (was_input_disabled) {
			input->set_disable_input(false);
		}

		if (input->is_physical_key_pressed(Key::A)) {
			direction -= right;
		}
		if (input->is_physical_key_pressed(Key::D)) {
			direction += right;
		}
		if (input->is_physical_key_pressed(Key::W)) {
			direction += forward;
		}
		if (input->is_physical_key_pressed(Key::S)) {
			direction -= forward;
		}
		if (input->is_physical_key_pressed(Key::E)) {
			direction += up;
		}
		if (input->is_physical_key_pressed(Key::Q)) {
			direction -= up;
		}

		real_t speed = freelook_base_speed;
		if (input->is_physical_key_pressed(Key::SHIFT)) {
			speed *= 3.0;
		}
		if (input->is_physical_key_pressed(Key::ALT)) {
			speed *= 0.333333;
		}

		if (was_input_disabled) {
			input->set_disable_input(true);
		}

		if (direction != Vector3()) {
			Window *root = SceneTree::get_singleton()->get_root();
			ERR_FAIL_COND(!root->is_camera_3d_override_enabled());

			// Calculate the process time manually, as the time scale is frozen.
			const double process_time = (1.0 / Engine::get_singleton()->get_frames_per_second()) * Engine::get_singleton()->get_unfrozen_time_scale();
			const Vector3 motion = direction * speed * process_time;
			cursor.pos += motion;
			cursor.eye_pos += motion;

			root->get_override_camera_3d()->set_transform(_get_cursor_transform());
		}
	}
#endif // _3D_DISABLED

	if (selection_update_queued || !SceneTree::get_singleton()->is_suspended()) {
		selection_update_queued = false;
		if (has_selection) {
			_update_selection();
		}
	}
}

void RuntimeNodeSelect::_physics_frame() {
	if (selection_drag_state != SELECTION_DRAG_END && (selection_drag_state == SELECTION_DRAG_MOVE || Math::is_inf(selection_position.x))) {
		return;
	}

	Window *root = SceneTree::get_singleton()->get_root();
	bool selection_drag_valid = selection_drag_state == SELECTION_DRAG_END && selection_drag_area.get_area() > SELECTION_MIN_AREA;
	Vector<SelectResult> items;

	if (node_select_type == NODE_TYPE_2D) {
		if (selection_drag_valid) {
			for (int i = 0; i < root->get_child_count(); i++) {
				_find_canvas_items_at_rect(selection_drag_area, root->get_child(i), items);
			}
		} else if (!Math::is_inf(selection_position.x)) {
			for (int i = 0; i < root->get_child_count(); i++) {
				_find_canvas_items_at_pos(selection_position, root->get_child(i), items);
			}
		}

#ifndef _3D_DISABLED
	} else if (node_select_type == NODE_TYPE_3D) {
		if (selection_drag_valid) {
			_find_3d_items_at_rect(selection_drag_area, items);
		} else {
			_find_3d_items_at_pos(selection_position, items);
		}
#endif // _3D_DISABLED
	}

	if ((prefer_group_selection || avoid_locked_nodes) && !list_shortcut_pressed && node_select_mode == SELECT_MODE_SINGLE) {
		for (int i = 0; i < items.size(); i++) {
			Node *node = items[i].item;
			Node *final_node = node;
			real_t order = items[i].order;

			// Replace the node by the group if grouped.
			if (prefer_group_selection) {
				while (node && node != root) {
					if (node->has_meta("_edit_group_")) {
						final_node = node;

						if (Object::cast_to<CanvasItem>(final_node)) {
							CanvasItem *ci_tmp = Object::cast_to<CanvasItem>(final_node);
							order = ci_tmp->get_effective_z_index() + ci_tmp->get_canvas_layer();
#ifndef _3D_DISABLED
						} else if (Object::cast_to<Node3D>(final_node)) {
							Node3D *node3d_tmp = Object::cast_to<Node3D>(final_node);
							Camera3D *camera = root->get_camera_3d();
							Vector3 pos = camera->project_ray_origin(selection_position);
							order = -pos.distance_to(node3d_tmp->get_global_transform().origin);
#endif // _3D_DISABLED
						}
					}
					node = node->get_parent();
				}
			}

			// Filter out locked nodes.
			if (avoid_locked_nodes && final_node->get_meta("_edit_lock_", false)) {
				items.remove_at(i);
				i--;
				continue;
			}

			items.write[i].item = final_node;
			items.write[i].order = order;
		}
	}

	// Remove possible duplicates.
	for (int i = 0; i < items.size(); i++) {
		Node *item = items[i].item;
		for (int j = 0; j < i; j++) {
			if (items[j].item == item) {
				items.remove_at(i);
				i--;

				break;
			}
		}
	}

	items.sort();

	switch (selection_drag_state) {
		case SELECTION_DRAG_END: {
			selection_position = Point2(Math::INF, Math::INF);
			selection_drag_state = SELECTION_DRAG_NONE;

			if (selection_drag_area.get_area() > SELECTION_MIN_AREA) {
				if (!items.is_empty()) {
					Vector<Node *> nodes;
					for (const SelectResult item : items) {
						nodes.append(item.item);
					}
					_send_ids(nodes, false);
				}

				_update_selection_drag();
				return;
			}

			_update_selection_drag();
		} break;

		case SELECTION_DRAG_NONE: {
			if (node_select_mode == SELECT_MODE_LIST) {
				break;
			}

			if (multi_shortcut_pressed) {
				// Allow forcing box selection when an item was clicked.
				selection_drag_state = SELECTION_DRAG_MOVE;
			} else if (items.is_empty()) {
#ifdef _3D_DISABLED
				if (!selected_ci_nodes.is_empty()) {
#else
				if (!selected_ci_nodes.is_empty() || !selected_3d_nodes.is_empty()) {
#endif // _3D_DISABLED
					EngineDebugger::get_singleton()->send_message("remote_nothing_selected", Array());
					_clear_selection();
				}

				selection_drag_state = SELECTION_DRAG_MOVE;
			} else {
				break;
			}

			[[fallthrough]];
		}

		case SELECTION_DRAG_MOVE: {
			selection_drag_area.position = selection_position;

			// Stop selection on click, so it can happen on release if the selection area doesn't pass the threshold.
			if (multi_shortcut_pressed) {
				return;
			}
		}
	}

	if (items.is_empty()) {
		selection_position = Point2(Math::INF, Math::INF);
		return;
	}
	if ((!list_shortcut_pressed && node_select_mode == SELECT_MODE_SINGLE) || items.size() == 1) {
		selection_position = Point2(Math::INF, Math::INF);

		Vector<Node *> node;
		node.append(items[0].item);
		_send_ids(node);

		return;
	}

	if (!selection_list && (list_shortcut_pressed || node_select_mode == SELECT_MODE_LIST)) {
		_open_selection_list(items, selection_position);
	}

	selection_position = Point2(Math::INF, Math::INF);
}

void RuntimeNodeSelect::_send_ids(const Vector<Node *> &p_picked_nodes, bool p_invert_new_selections) {
	ERR_FAIL_COND(p_picked_nodes.is_empty());

	Vector<Node *> picked_nodes = p_picked_nodes;
	Array message;

	if (!multi_shortcut_pressed) {
		if (picked_nodes.size() > max_selection) {
			picked_nodes.resize(max_selection);
			EngineDebugger::get_singleton()->send_message("show_selection_limit_warning", Array());
		}

		for (const Node *node : picked_nodes) {
			SceneDebuggerObject obj(node->get_instance_id());
			Array arr;
			obj.serialize(arr);
			message.append(arr);
		}

		EngineDebugger::get_singleton()->send_message("remote_objects_selected", message);
		_set_selected_nodes(picked_nodes);

		return;
	}

	LocalVector<Node *> nodes;
	LocalVector<ObjectID> ids;
	for (Node *node : picked_nodes) {
		ObjectID id = node->get_instance_id();
		if (CanvasItem *ci = Object::cast_to<CanvasItem>(node)) {
			if (selected_ci_nodes.has(id)) {
				if (p_invert_new_selections) {
					selected_ci_nodes.erase(id);
				}
			} else {
				ids.push_back(id);
				nodes.push_back(ci);
			}
		} else {
#ifndef _3D_DISABLED
			if (Node3D *node3d = Object::cast_to<Node3D>(node)) {
				if (selected_3d_nodes.has(id)) {
					if (p_invert_new_selections) {
						selected_3d_nodes.erase(id);
					}
				} else {
					ids.push_back(id);
					nodes.push_back(node3d);
				}
			}
#endif // _3D_DISABLED
		}
	}

	uint32_t limit = max_selection - selected_ci_nodes.size();
#ifndef _3D_DISABLED
	limit -= selected_3d_nodes.size();
#endif // _3D_DISABLED
	if (ids.size() > limit) {
		ids.resize(limit);
		nodes.resize(limit);
		EngineDebugger::get_singleton()->send_message("show_selection_limit_warning", Array());
	}

	for (ObjectID id : selected_ci_nodes) {
		ids.push_back(id);
		nodes.push_back(ObjectDB::get_instance<Node>(id));
	}
#ifndef _3D_DISABLED
	for (const KeyValue<ObjectID, Ref<SelectionBox3D>> &KV : selected_3d_nodes) {
		ids.push_back(KV.key);
		nodes.push_back(ObjectDB::get_instance<Node>(KV.key));
	}
#endif // _3D_DISABLED

	if (ids.is_empty()) {
		EngineDebugger::get_singleton()->send_message("remote_nothing_selected", message);
	} else {
		for (const ObjectID &id : ids) {
			SceneDebuggerObject obj(id);
			Array arr;
			obj.serialize(arr);
			message.append(arr);
		}

		EngineDebugger::get_singleton()->send_message("remote_objects_selected", message);
	}

	_set_selected_nodes(Vector<Node *>(nodes));
}

void RuntimeNodeSelect::_set_selected_nodes(const Vector<Node *> &p_nodes) {
	if (p_nodes.is_empty()) {
		_clear_selection();
		return;
	}

	bool changed = false;
	LocalVector<ObjectID> nodes_ci;
#ifndef _3D_DISABLED
	HashMap<ObjectID, Ref<SelectionBox3D>> nodes_3d;
#endif // _3D_DISABLED

	for (Node *node : p_nodes) {
		ObjectID id = node->get_instance_id();
		if (Object::cast_to<CanvasItem>(node)) {
			if (!changed || !selected_ci_nodes.has(id)) {
				changed = true;
			}

			nodes_ci.push_back(id);
		} else {
#ifndef _3D_DISABLED
			Node3D *node_3d = Object::cast_to<Node3D>(node);
			if (!node_3d || !node_3d->is_inside_world()) {
				continue;
			}

			if (!changed || !selected_3d_nodes.has(id)) {
				changed = true;
			}

			if (selected_3d_nodes.has(id)) {
				// Assign an already available visual instance.
				nodes_3d[id] = selected_3d_nodes.get(id);
				continue;
			}

			if (sbox_3d_mesh.is_null() || sbox_3d_mesh_xray.is_null()) {
				continue;
			}

			Ref<SelectionBox3D> sb;
			sb.instantiate();
			nodes_3d[id] = sb;

			RID scenario = node_3d->get_world_3d()->get_scenario();

			sb->instance = RS::get_singleton()->instance_create2(sbox_3d_mesh->get_rid(), scenario);
			sb->instance_ofs = RS::get_singleton()->instance_create2(sbox_3d_mesh->get_rid(), scenario);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance, RS::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_ofs, RS::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_ofs, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_ofs, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

			sb->instance_xray = RS::get_singleton()->instance_create2(sbox_3d_mesh_xray->get_rid(), scenario);
			sb->instance_xray_ofs = RS::get_singleton()->instance_create2(sbox_3d_mesh_xray->get_rid(), scenario);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_xray, RS::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_cast_shadows_setting(sb->instance_xray_ofs, RS::SHADOW_CASTING_SETTING_OFF);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray_ofs, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
			RS::get_singleton()->instance_geometry_set_flag(sb->instance_xray_ofs, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
#endif // _3D_DISABLED
		}
	}

#ifdef _3D_DISABLED
	if (!changed && nodes_ci.size() == selected_ci_nodes.size()) {
		return;
	}
#else
	if (!changed && nodes_ci.size() == selected_ci_nodes.size() && nodes_3d.size() == selected_3d_nodes.size()) {
		return;
	}
#endif // _3D_DISABLED

	_clear_selection();
	selected_ci_nodes = nodes_ci;
	has_selection = !nodes_ci.is_empty();

#ifndef _3D_DISABLED
	if (!nodes_3d.is_empty()) {
		selected_3d_nodes = nodes_3d;
		has_selection = true;
	}
#endif // _3D_DISABLED

	_queue_selection_update();
}

void RuntimeNodeSelect::_queue_selection_update() {
	if (has_selection && selection_visible) {
		if (SceneTree::get_singleton()->is_suspended()) {
			_update_selection();
		} else {
			selection_update_queued = true;
		}
	}
}

void RuntimeNodeSelect::_update_selection() {
	RS::get_singleton()->canvas_item_clear(sbox_2d_ci);
	RS::get_singleton()->canvas_item_set_visible(sbox_2d_ci, selection_visible);

	for (LocalVector<ObjectID>::Iterator E = selected_ci_nodes.begin(); E != selected_ci_nodes.end(); ++E) {
		ObjectID id = *E;
		CanvasItem *ci = ObjectDB::get_instance<CanvasItem>(id);
		if (!ci) {
			selected_ci_nodes.erase(id);
			--E;
			continue;
		}

		if (!ci->is_inside_tree()) {
			continue;
		}

		Transform2D xform = ci->get_global_transform_with_canvas();

		// Fallback.
		Rect2 rect = Rect2(Vector2(), Vector2(10, 10));

		if (ci->_edit_use_rect()) {
			rect = ci->_edit_get_rect();
		} else {
#ifndef PHYSICS_2D_DISABLED
			CollisionShape2D *collision_shape = Object::cast_to<CollisionShape2D>(ci);
			if (collision_shape) {
				Ref<Shape2D> shape = collision_shape->get_shape();
				if (shape.is_valid()) {
					rect = shape->get_rect();
				}
			}
#endif // PHYSICS_2D_DISABLED
		}

		const Vector2 endpoints[4] = {
			xform.xform(rect.position),
			xform.xform(rect.position + Point2(rect.size.x, 0)),
			xform.xform(rect.position + rect.size),
			xform.xform(rect.position + Point2(0, rect.size.y))
		};

		const Color selection_color_2d = Color(1, 0.6, 0.4, 0.7);
		for (int i = 0; i < 4; i++) {
			RS::get_singleton()->canvas_item_add_line(sbox_2d_ci, endpoints[i], endpoints[(i + 1) % 4], selection_color_2d, sel_2d_scale);
		}
	}

#ifndef _3D_DISABLED
	for (HashMap<ObjectID, Ref<SelectionBox3D>>::ConstIterator KV = selected_3d_nodes.begin(); KV != selected_3d_nodes.end(); ++KV) {
		ObjectID id = KV->key;
		Node3D *node_3d = ObjectDB::get_instance<Node3D>(id);
		if (!node_3d) {
			selected_3d_nodes.erase(id);
			--KV;
			continue;
		}

		if (!node_3d->is_inside_tree()) {
			continue;
		}

		// Fallback.
		AABB bounds(Vector3(-0.5, -0.5, -0.5), Vector3(1, 1, 1));

		VisualInstance3D *visual_instance = Object::cast_to<VisualInstance3D>(node_3d);
		if (visual_instance) {
			bounds = visual_instance->get_aabb();
		} else {
#ifndef PHYSICS_3D_DISABLED
			CollisionShape3D *collision_shape = Object::cast_to<CollisionShape3D>(node_3d);
			if (collision_shape) {
				Ref<Shape3D> shape = collision_shape->get_shape();
				if (shape.is_valid()) {
					bounds = shape->get_debug_mesh()->get_aabb();
				}
			}
#endif // PHYSICS_3D_DISABLED
		}

		Transform3D xform_to_top_level_parent_space = node_3d->get_global_transform().affine_inverse() * node_3d->get_global_transform();
		bounds = xform_to_top_level_parent_space.xform(bounds);
		Transform3D t = node_3d->get_global_transform();

		Ref<SelectionBox3D> sb = KV->value;
		if (t == sb->transform && bounds == sb->bounds) {
			continue; // Nothing changed.
		}
		sb->transform = t;
		sb->bounds = bounds;

		Transform3D t_offset = t;

		// Apply AABB scaling before item's global transform.
		{
			const Vector3 offset(0.005, 0.005, 0.005);
			Basis aabb_s;
			aabb_s.scale(bounds.size + offset);
			t.translate_local(bounds.position - offset / 2);
			t.basis = t.basis * aabb_s;
		}
		{
			const Vector3 offset(0.01, 0.01, 0.01);
			Basis aabb_s;
			aabb_s.scale(bounds.size + offset);
			t_offset.translate_local(bounds.position - offset / 2);
			t_offset.basis = t_offset.basis * aabb_s;
		}

		RS::get_singleton()->instance_set_visible(sb->instance, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_ofs, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_xray, selection_visible);
		RS::get_singleton()->instance_set_visible(sb->instance_xray_ofs, selection_visible);

		RS::get_singleton()->instance_set_transform(sb->instance, t);
		RS::get_singleton()->instance_set_transform(sb->instance_ofs, t_offset);
		RS::get_singleton()->instance_set_transform(sb->instance_xray, t);
		RS::get_singleton()->instance_set_transform(sb->instance_xray_ofs, t_offset);
	}
#endif // _3D_DISABLED
}

void RuntimeNodeSelect::_clear_selection() {
	selected_ci_nodes.clear();
	if (draw_canvas.is_valid()) {
		RS::get_singleton()->canvas_item_clear(sbox_2d_ci);
	}

#ifndef _3D_DISABLED
	selected_3d_nodes.clear();
#endif // _3D_DISABLED

	has_selection = false;
}

void RuntimeNodeSelect::_update_selection_drag(const Point2 &p_end_pos) {
	RS::get_singleton()->canvas_item_clear(sel_drag_ci);

	if (selection_drag_state != SELECTION_DRAG_MOVE) {
		return;
	}

	selection_drag_area.size = p_end_pos - selection_drag_area.position;

	if (selection_drag_state == SELECTION_DRAG_END) {
		return;
	}

	Rect2 selection_drawing = selection_drag_area.abs();
	int thickness = 1;

	const Vector2 endpoints[4] = {
		selection_drawing.position,
		selection_drawing.position + Point2(selection_drawing.size.x, 0),
		selection_drawing.position + selection_drawing.size,
		selection_drawing.position + Point2(0, selection_drawing.size.y)
	};

	// Draw fill.
	RS::get_singleton()->canvas_item_add_rect(sel_drag_ci, selection_drawing, selection_area_fill);
	// Draw outline.
	for (int i = 0; i < 4; i++) {
		RS::get_singleton()->canvas_item_add_line(sel_drag_ci, endpoints[i], endpoints[(i + 1) % 4], selection_area_outline, thickness);
	}
}

void RuntimeNodeSelect::_open_selection_list(const Vector<SelectResult> &p_items, const Point2 &p_pos) {
	Window *root = SceneTree::get_singleton()->get_root();

	selection_list = memnew(PopupMenu);
	selection_list->set_theme(ThemeDB::get_singleton()->get_default_theme());
	selection_list->set_auto_translate_mode(Node::AUTO_TRANSLATE_MODE_DISABLED);
	selection_list->set_force_native(true);
	selection_list->connect("index_pressed", callable_mp(this, &RuntimeNodeSelect::_items_popup_index_pressed).bind(selection_list));
	selection_list->connect("popup_hide", callable_mp(this, &RuntimeNodeSelect::_close_selection_list));

	root->add_child(selection_list);

	for (const SelectResult &I : p_items) {
		int locked = 0;
		if (I.item->get_meta("_edit_lock_", false)) {
			locked = 1;
		} else {
			Node *scene = SceneTree::get_singleton()->get_root();
			Node *node = I.item;

			while (node && node != scene->get_parent()) {
				if (node->has_meta("_edit_group_")) {
					locked = 2;
				}
				node = node->get_parent();
			}
		}

		String suffix;
		if (locked == 1) {
			suffix = " (" + RTR("Locked") + ")";
		} else if (locked == 2) {
			suffix = " (" + RTR("Grouped") + ")";
		}

		selection_list->add_item((String)I.item->get_name() + suffix);
		selection_list->set_item_metadata(-1, I.item);
	}

	selection_list->set_position(selection_list->is_embedded() ? p_pos : (Input::get_singleton()->get_mouse_position() + root->get_position()));
	selection_list->reset_size();
	selection_list->popup();

	selection_list->set_content_scale_factor(1);
	selection_list->set_min_size(selection_list->get_contents_minimum_size());
	selection_list->reset_size();

	// FIXME: Ugly hack that stops the popup from hiding when the button is released.
	selection_list->call_deferred(SNAME("set_position"), selection_list->get_position() + Point2(1, 0));
}

void RuntimeNodeSelect::_close_selection_list() {
	selection_list->queue_free();
	selection_list = nullptr;
}

void RuntimeNodeSelect::_set_selection_visible(bool p_visible) {
	selection_visible = p_visible;

	if (has_selection) {
		_update_selection();
	}
}

void RuntimeNodeSelect::_set_avoid_locked(bool p_enabled) {
	avoid_locked_nodes = p_enabled;
}

void RuntimeNodeSelect::_set_prefer_group(bool p_enabled) {
	prefer_group_selection = p_enabled;
}

// Copied and trimmed from the CanvasItemEditor implementation.
void RuntimeNodeSelect::_find_canvas_items_at_pos(const Point2 &p_pos, Node *p_node, Vector<SelectResult> &r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	if (!p_node || Object::cast_to<Viewport>(p_node)) {
		return;
	}

	CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);
	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		if (ci) {
			if (!ci->is_set_as_top_level()) {
				_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, p_parent_xform * ci->get_transform(), p_canvas_xform);
			} else {
				_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, ci->get_transform(), p_canvas_xform);
			}
		} else {
			CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
			_find_canvas_items_at_pos(p_pos, p_node->get_child(i), r_items, Transform2D(), cl ? cl->get_transform() : p_canvas_xform);
		}
	}

	if (!ci || !ci->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = p_canvas_xform;
	if (!ci->is_set_as_top_level()) {
		xform *= p_parent_xform;
	}

	Window *root = SceneTree::get_singleton()->get_root();
	Point2 pos;

	// Cameras don't affect `CanvasLayer`s.
	if (!ci->get_canvas_layer_node() || ci->get_canvas_layer_node()->is_following_viewport()) {
		pos = root->get_canvas_transform().affine_inverse().xform(p_pos);
	} else {
		pos = p_pos;
	}

	xform = (xform * ci->get_transform()).affine_inverse();
	const real_t local_grab_distance = xform.basis_xform(Vector2(sel_2d_grab_dist, 0)).length() / view_2d_zoom;
	if (ci->_edit_is_selected_on_click(xform.xform(pos), local_grab_distance)) {
		SelectResult res;
		res.item = ci;
		res.order = ci->get_effective_z_index() + ci->get_canvas_layer();
		r_items.push_back(res);

#ifndef PHYSICS_2D_DISABLED
		// If it's a shape, get the collision object it's from.
		// FIXME: If the collision object has multiple shapes, only the topmost will be above it in the list.
		if (Object::cast_to<CollisionShape2D>(ci) || Object::cast_to<CollisionPolygon2D>(ci)) {
			CollisionObject2D *collision_object = Object::cast_to<CollisionObject2D>(ci->get_parent());
			if (collision_object) {
				SelectResult res_col;
				res_col.item = ci->get_parent();
				res_col.order = collision_object->get_z_index() + ci->get_canvas_layer();
				r_items.push_back(res_col);
			}
		}
#endif // PHYSICS_2D_DISABLED
	}
}

// Copied and trimmed from the CanvasItemEditor implementation.
void RuntimeNodeSelect::_find_canvas_items_at_rect(const Rect2 &p_rect, Node *p_node, Vector<SelectResult> &r_items, const Transform2D &p_parent_xform, const Transform2D &p_canvas_xform) {
	if (!p_node || Object::cast_to<Viewport>(p_node)) {
		return;
	}

	CanvasItem *ci = Object::cast_to<CanvasItem>(p_node);
	for (int i = p_node->get_child_count() - 1; i >= 0; i--) {
		if (ci) {
			if (!ci->is_set_as_top_level()) {
				_find_canvas_items_at_rect(p_rect, p_node->get_child(i), r_items, p_parent_xform * ci->get_transform(), p_canvas_xform);
			} else {
				_find_canvas_items_at_rect(p_rect, p_node->get_child(i), r_items, ci->get_transform(), p_canvas_xform);
			}
		} else {
			CanvasLayer *cl = Object::cast_to<CanvasLayer>(p_node);
			_find_canvas_items_at_rect(p_rect, p_node->get_child(i), r_items, Transform2D(), cl ? cl->get_transform() : p_canvas_xform);
		}
	}

	if (!ci || !ci->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = p_canvas_xform;
	if (!ci->is_set_as_top_level()) {
		xform *= p_parent_xform;
	}

	Window *root = SceneTree::get_singleton()->get_root();
	Rect2 rect;
	// Cameras don't affect `CanvasLayer`s.
	if (!ci->get_canvas_layer_node() || ci->get_canvas_layer_node()->is_following_viewport()) {
		rect = root->get_canvas_transform().affine_inverse().xform(p_rect);
	} else {
		rect = p_rect;
	}
	rect = (xform * ci->get_transform()).affine_inverse().xform(rect);

	bool selected = false;
	if (ci->_edit_use_rect()) {
		Rect2 ci_rect = ci->_edit_get_rect();
		if (rect.has_point(ci_rect.position) &&
				rect.has_point(ci_rect.position + Vector2(ci_rect.size.x, 0)) &&
				rect.has_point(ci_rect.position + Vector2(ci_rect.size.x, ci_rect.size.y)) &&
				rect.has_point(ci_rect.position + Vector2(0, ci_rect.size.y))) {
			selected = true;
		}
	} else if (rect.has_point(Point2())) {
		selected = true;
	}

	if (selected) {
		SelectResult res;
		res.item = ci;
		res.order = ci->get_effective_z_index() + ci->get_canvas_layer();
		r_items.push_back(res);
	}
}

void RuntimeNodeSelect::_pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event) {
	Vector2 scroll = SceneTree::get_singleton()->get_root()->get_screen_transform().affine_inverse().xform(p_scroll_vec);
	view_2d_offset.x -= scroll.x / view_2d_zoom;
	view_2d_offset.y -= scroll.y / view_2d_zoom;

#ifndef _2D_DISABLED
	_update_view_2d();
#endif // _2D_DISABLED
}

// A very shallow copy of the same function inside CanvasItemEditor.
void RuntimeNodeSelect::_zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event) {
	real_t prev_zoom = view_2d_zoom;
	view_2d_zoom = CLAMP(view_2d_zoom * p_zoom_factor, VIEW_2D_MIN_ZOOM, VIEW_2D_MAX_ZOOM);

	Vector2 pos = SceneTree::get_singleton()->get_root()->get_screen_transform().affine_inverse().xform(p_origin);
	view_2d_offset += pos / prev_zoom - pos / view_2d_zoom;

	// We want to align in-scene pixels to screen pixels, this prevents blurry rendering
	// of small details (texts, lines).
	// This correction adds a jitter movement when zooming, so we correct only when the
	// zoom factor is an integer. (in the other cases, all pixels won't be aligned anyway)
	const real_t closest_zoom_factor = Math::round(view_2d_zoom);
	if (Math::is_zero_approx(view_2d_zoom - closest_zoom_factor)) {
		// Make sure scene pixel at view_offset is aligned on a screen pixel.
		Vector2 view_offset_int = view_2d_offset.floor();
		Vector2 view_offset_frac = view_2d_offset - view_offset_int;
		view_2d_offset = view_offset_int + (view_offset_frac * closest_zoom_factor).round() / closest_zoom_factor;
	}

#ifndef _2D_DISABLED
	_update_view_2d();
#endif // _2D_DISABLED
}

#ifndef _2D_DISABLED
void RuntimeNodeSelect::_reset_camera_2d() {
	camera_first_override = true;
	Window *root = SceneTree::get_singleton()->get_root();
	Camera2D *game_camera = root->is_camera_2d_override_enabled() ? root->get_overridden_camera_2d() : root->get_camera_2d();
	if (game_camera) {
		// Ideally we should be using Camera2D::get_camera_transform() but it's not so this hack will have to do for now.
		view_2d_offset = game_camera->get_camera_screen_center() - (0.5 * root->get_visible_rect().size);
	} else {
		view_2d_offset = Vector2();
	}

	view_2d_zoom = 1;

	if (root->is_camera_2d_override_enabled()) {
		_update_view_2d();
	}
}

void RuntimeNodeSelect::_update_view_2d() {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(!root->is_camera_2d_override_enabled());

	Camera2D *override_camera = root->get_override_camera_2d();
	override_camera->set_anchor_mode(Camera2D::ANCHOR_MODE_FIXED_TOP_LEFT);
	override_camera->set_zoom(Vector2(view_2d_zoom, view_2d_zoom));
	override_camera->set_offset(view_2d_offset);

	_queue_selection_update();
}
#endif // _2D_DISABLED

#ifndef _3D_DISABLED
void RuntimeNodeSelect::_find_3d_items_at_pos(const Point2 &p_pos, Vector<SelectResult> &r_items) {
	Window *root = SceneTree::get_singleton()->get_root();

	Vector3 ray, pos, to;
	Camera3D *camera = root->get_camera_3d();
	if (!camera) {
		return;
	}

	ray = camera->project_ray_normal(p_pos);
	pos = camera->project_ray_origin(p_pos);
	to = pos + ray * camera->get_far();

#ifndef PHYSICS_3D_DISABLED
	// Start with physical objects.
	PhysicsDirectSpaceState3D *ss = root->get_world_3d()->get_direct_space_state();
	PhysicsDirectSpaceState3D::RayResult result;
	HashSet<RID> excluded;
	PhysicsDirectSpaceState3D::RayParameters ray_params;
	ray_params.from = pos;
	ray_params.to = to;
	ray_params.collide_with_areas = true;
	while (true) {
		ray_params.exclude = excluded;
		if (ss->intersect_ray(ray_params, result)) {
			SelectResult res;
			res.item = Object::cast_to<Node>(result.collider);
			res.order = -pos.distance_to(result.position);

			// Fetch collision shapes.
			CollisionObject3D *collision = Object::cast_to<CollisionObject3D>(result.collider);
			if (collision) {
				List<uint32_t> owners;
				collision->get_shape_owners(&owners);
				for (uint32_t &I : owners) {
					SelectResult res_shape;
					res_shape.item = Object::cast_to<Node>(collision->shape_owner_get_owner(I));
					res_shape.order = res.order;
					r_items.push_back(res_shape);
				}
			}

			r_items.push_back(res);

			excluded.insert(result.rid);
		} else {
			break;
		}
	}
#endif // PHYSICS_3D_DISABLED

	// Then go for the meshes.
	Vector<ObjectID> items = RS::get_singleton()->instances_cull_ray(pos, to, root->get_world_3d()->get_scenario());
	for (int i = 0; i < items.size(); i++) {
		Object *obj = ObjectDB::get_instance(items[i]);

		GeometryInstance3D *geo_instance = Object::cast_to<GeometryInstance3D>(obj);
		if (geo_instance) {
			Ref<TriangleMesh> mesh_collision = geo_instance->generate_triangle_mesh();

			if (mesh_collision.is_valid()) {
				Transform3D gt = geo_instance->get_global_transform();
				Transform3D ai = gt.affine_inverse();
				Vector3 point, normal;
				if (mesh_collision->intersect_ray(ai.xform(pos), ai.basis.xform(ray).normalized(), point, normal)) {
					SelectResult res;
					res.item = Object::cast_to<Node>(obj);
					res.order = -pos.distance_to(gt.xform(point));
					r_items.push_back(res);

					continue;
				}
			}
		}

		items.remove_at(i);
		i--;
	}
}

void RuntimeNodeSelect::_find_3d_items_at_rect(const Rect2 &p_rect, Vector<SelectResult> &r_items) {
	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *camera = root->get_camera_3d();
	if (!camera) {
		return;
	}

	Vector3 cam_pos = camera->get_global_position();
	Vector3 dist_pos = camera->project_ray_origin(p_rect.position + p_rect.size / 2);

	real_t znear = camera->get_near();
	real_t zfar = camera->get_far();
	real_t zofs = MAX(0.0, 5.0 - znear);

	const Point2 pos_end = p_rect.position + p_rect.size;
	Vector3 box[4] = {
		Vector3(
				MIN(p_rect.position.x, pos_end.x),
				MIN(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MAX(p_rect.position.x, pos_end.x),
				MIN(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MAX(p_rect.position.x, pos_end.x),
				MAX(p_rect.position.y, pos_end.y),
				zofs),
		Vector3(
				MIN(p_rect.position.x, pos_end.x),
				MAX(p_rect.position.y, pos_end.y),
				zofs)
	};

	Vector<Plane> frustum;
	for (int i = 0; i < 4; i++) {
		Vector3 a = _get_screen_to_space(box[i]);
		Vector3 b = _get_screen_to_space(box[(i + 1) % 4]);
		frustum.push_back(Plane(a, b, cam_pos));
	}

	// Get the camera normal.
	Plane near_plane = Plane(camera->get_global_transform().basis.get_column(2), cam_pos);

	near_plane.d -= znear;
	frustum.push_back(near_plane);

	Plane far_plane = -near_plane;
	far_plane.d += zfar;
	frustum.push_back(far_plane);

	// Keep track of the currently listed nodes, so repeats can be ignored.
	HashSet<Node *> node_list;

#ifndef PHYSICS_3D_DISABLED
	Vector<Vector3> points = Geometry3D::compute_convex_mesh_points(&frustum[0], frustum.size());
	Ref<ConvexPolygonShape3D> shape;
	shape.instantiate();
	shape->set_points(points);

	// Start with physical objects.
	PhysicsDirectSpaceState3D *ss = root->get_world_3d()->get_direct_space_state();
	PhysicsDirectSpaceState3D::ShapeResult results[32];
	PhysicsDirectSpaceState3D::ShapeParameters shape_params;
	shape_params.shape_rid = shape->get_rid();
	shape_params.collide_with_areas = true;
	const int num_hits = ss->intersect_shape(shape_params, results, 32);
	for (int i = 0; i < num_hits; i++) {
		const PhysicsDirectSpaceState3D::ShapeResult &result = results[i];
		SelectResult res;
		res.item = Object::cast_to<Node>(result.collider);
		res.order = -dist_pos.distance_to(Object::cast_to<Node3D>(res.item)->get_global_transform().origin);

		// Fetch collision shapes.
		CollisionObject3D *collision = Object::cast_to<CollisionObject3D>(result.collider);
		if (collision) {
			List<uint32_t> owners;
			collision->get_shape_owners(&owners);
			for (uint32_t &I : owners) {
				SelectResult res_shape;
				res_shape.item = Object::cast_to<Node>(collision->shape_owner_get_owner(I));
				if (!node_list.has(res_shape.item)) {
					node_list.insert(res_shape.item);
					res_shape.order = res.order;
					r_items.push_back(res_shape);
				}
			}
		}

		if (!node_list.has(res.item)) {
			node_list.insert(res.item);
			r_items.push_back(res);
		}
	}
#endif // PHYSICS_3D_DISABLED

	// Then go for the meshes.
	Vector<ObjectID> items = RS::get_singleton()->instances_cull_convex(frustum, root->get_world_3d()->get_scenario());
	for (int i = 0; i < items.size(); i++) {
		Object *obj = ObjectDB::get_instance(items[i]);
		GeometryInstance3D *geo_instance = Object::cast_to<GeometryInstance3D>(obj);
		if (geo_instance) {
			Ref<TriangleMesh> mesh_collision = geo_instance->generate_triangle_mesh();

			if (mesh_collision.is_valid()) {
				Transform3D gt = geo_instance->get_global_transform();
				Vector3 mesh_scale = gt.get_basis().get_scale();
				gt.orthonormalize();

				Transform3D it = gt.affine_inverse();

				Vector<Plane> transformed_frustum;
				int plane_count = frustum.size();
				transformed_frustum.resize(plane_count);

				for (int j = 0; j < plane_count; j++) {
					transformed_frustum.write[j] = it.xform(frustum[j]);
				}
				Vector<Vector3> convex_points = Geometry3D::compute_convex_mesh_points(transformed_frustum.ptr(), plane_count);
				if (mesh_collision->inside_convex_shape(transformed_frustum.ptr(), transformed_frustum.size(), convex_points.ptr(), convex_points.size(), mesh_scale)) {
					SelectResult res;
					res.item = Object::cast_to<Node>(obj);
					if (!node_list.has(res.item)) {
						node_list.insert(res.item);
						res.order = -dist_pos.distance_to(gt.origin);
						r_items.push_back(res);
					}

					continue;
				}
			}
		}

		items.remove_at(i);
		i--;
	}
}

Vector3 RuntimeNodeSelect::_get_screen_to_space(const Vector3 &p_vector3) {
	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *camera = root->get_camera_3d();

	Transform3D camera_transform = camera->get_camera_transform();
	Size2 size = root->get_size();
	real_t znear = camera->get_near();
	Projection cm = Projection::create_perspective(camera->get_fov(), size.aspect(), znear + p_vector3.z, camera->get_far());
	Vector2 screen_he = cm.get_viewport_half_extents();
	return camera_transform.xform(Vector3(((p_vector3.x / size.width) * 2.0 - 1.0) * screen_he.x, ((1.0 - (p_vector3.y / size.height)) * 2.0 - 1.0) * screen_he.y, -(znear + p_vector3.z)));
}

bool RuntimeNodeSelect::_handle_3d_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		const real_t zoom_factor = 1.08 * b->get_factor();
		switch (b->get_button_index()) {
			case MouseButton::WHEEL_UP: {
				if (!camera_freelook) {
					_cursor_scale_distance(1.0 / zoom_factor);
				} else {
					_scale_freelook_speed(zoom_factor);
				}

				return true;
			} break;
			case MouseButton::WHEEL_DOWN: {
				if (!camera_freelook) {
					_cursor_scale_distance(zoom_factor);
				} else {
					_scale_freelook_speed(1.0 / zoom_factor);
				}

				return true;
			} break;
			case MouseButton::RIGHT: {
				_set_camera_freelook_enabled(b->is_pressed());
				return true;
			} break;
			default: {
			}
		}
	}

	Ref<InputEventMouseMotion> m = p_event;
	if (m.is_valid()) {
		if (camera_freelook) {
			_cursor_look(m);
		} else if (m->get_button_mask().has_flag(MouseButtonMask::MIDDLE)) {
			if (m->is_shift_pressed()) {
				_cursor_pan(m);
			} else {
				_cursor_orbit(m);
			}
		}

		return true;
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {
		if (k->get_physical_keycode() == Key::ESCAPE) {
			_set_camera_freelook_enabled(false);
			return true;
		} else if (k->is_ctrl_pressed()) {
			switch (k->get_physical_keycode()) {
				case Key::EQUAL: {
					ERR_FAIL_COND_V(!SceneTree::get_singleton()->get_root()->is_camera_3d_override_enabled(), false);
					cursor.fov_scale = CLAMP(cursor.fov_scale - 0.05, CAMERA_MIN_FOV_SCALE, CAMERA_MAX_FOV_SCALE);
					SceneTree::get_singleton()->get_root()->get_override_camera_3d()->set_perspective(camera_fov * cursor.fov_scale, camera_znear, camera_zfar);

					return true;
				} break;
				case Key::MINUS: {
					ERR_FAIL_COND_V(!SceneTree::get_singleton()->get_root()->is_camera_3d_override_enabled(), false);
					cursor.fov_scale = CLAMP(cursor.fov_scale + 0.05, CAMERA_MIN_FOV_SCALE, CAMERA_MAX_FOV_SCALE);
					SceneTree::get_singleton()->get_root()->get_override_camera_3d()->set_perspective(camera_fov * cursor.fov_scale, camera_znear, camera_zfar);

					return true;
				} break;
				case Key::KEY_0: {
					ERR_FAIL_COND_V(!SceneTree::get_singleton()->get_root()->is_camera_3d_override_enabled(), false);
					cursor.fov_scale = 1;
					SceneTree::get_singleton()->get_root()->get_override_camera_3d()->set_perspective(camera_fov, camera_znear, camera_zfar);

					return true;
				} break;
				default: {
				}
			}
		}
	}

	// TODO: Handle magnify and pan input gestures.

	return false;
}

void RuntimeNodeSelect::_set_camera_freelook_enabled(bool p_enabled) {
	camera_freelook = p_enabled;

	if (p_enabled) {
		// Make sure eye_pos is synced, because freelook referential is eye pos rather than orbit pos
		Vector3 forward = _get_cursor_transform().basis.xform(Vector3(0, 0, -1));
		cursor.eye_pos = cursor.pos - cursor.distance * forward;

		previous_mouse_position = SceneTree::get_singleton()->get_root()->get_mouse_position();

		// Hide mouse like in an FPS (warping doesn't work).
		Input::get_singleton()->set_mouse_mode_override(Input::MOUSE_MODE_CAPTURED);

	} else {
		// Restore mouse.
		Input::get_singleton()->set_mouse_mode_override(Input::MOUSE_MODE_VISIBLE);

		// Restore the previous mouse position when leaving freelook mode.
		// This is done because leaving `Input.MOUSE_MODE_CAPTURED` will center the cursor
		// due to OS limitations.
		Input::get_singleton()->warp_mouse(previous_mouse_position);
	}
}

void RuntimeNodeSelect::_cursor_scale_distance(real_t p_scale) {
	ERR_FAIL_COND(!SceneTree::get_singleton()->get_root()->is_camera_3d_override_enabled());
	real_t min_distance = MAX(camera_znear * 4, VIEW_3D_MIN_ZOOM);
	real_t max_distance = MIN(camera_zfar / 4, VIEW_3D_MAX_ZOOM);
	cursor.distance = CLAMP(cursor.distance * p_scale, min_distance, max_distance);

	SceneTree::get_singleton()->get_root()->get_override_camera_3d()->set_transform(_get_cursor_transform());
}

void RuntimeNodeSelect::_scale_freelook_speed(real_t p_scale) {
	real_t min_speed = MAX(camera_znear * 4, VIEW_3D_MIN_ZOOM);
	real_t max_speed = MIN(camera_zfar / 4, VIEW_3D_MAX_ZOOM);
	if (unlikely(min_speed > max_speed)) {
		freelook_base_speed = (min_speed + max_speed) / 2;
	} else {
		freelook_base_speed = CLAMP(freelook_base_speed * p_scale, min_speed, max_speed);
	}
}

void RuntimeNodeSelect::_cursor_look(Ref<InputEventWithModifiers> p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(!root->is_camera_3d_override_enabled());

	const Vector2 relative = _get_warped_mouse_motion(p_event, Rect2(Vector2(), root->get_size()));
	const Transform3D prev_camera_transform = _get_cursor_transform();

	if (invert_y_axis) {
		cursor.x_rot -= relative.y * freelook_sensitivity;
	} else {
		cursor.x_rot += relative.y * freelook_sensitivity;
	}
	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.x_rot = CLAMP(cursor.x_rot, -1.57, 1.57);

	cursor.y_rot += relative.x * freelook_sensitivity;

	// Look is like the opposite of Orbit: the focus point rotates around the camera.
	Transform3D camera_transform = _get_cursor_transform();
	Vector3 pos = camera_transform.xform(Vector3(0, 0, 0));
	Vector3 prev_pos = prev_camera_transform.xform(Vector3(0, 0, 0));
	Vector3 diff = prev_pos - pos;
	cursor.pos += diff;

	root->get_override_camera_3d()->set_transform(_get_cursor_transform());
}

void RuntimeNodeSelect::_cursor_pan(Ref<InputEventWithModifiers> p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(!root->is_camera_3d_override_enabled());

	// Reduce all sides of the area by 1, so warping works when windows are maximized/fullscreen.
	const Vector2 relative = _get_warped_mouse_motion(p_event, Rect2(Vector2(1, 1), root->get_size() - Vector2(2, 2)));
	const real_t pan_speed = translation_sensitivity / 150.0;

	Transform3D camera_transform;
	camera_transform.translate_local(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);

	Vector3 translation(1 * -relative.x * pan_speed, relative.y * pan_speed, 0);
	translation *= cursor.distance / 4;
	camera_transform.translate_local(translation);
	cursor.pos = camera_transform.origin;

	root->get_override_camera_3d()->set_transform(_get_cursor_transform());
}

void RuntimeNodeSelect::_cursor_orbit(Ref<InputEventWithModifiers> p_event) {
	Window *root = SceneTree::get_singleton()->get_root();
	ERR_FAIL_COND(!root->is_camera_3d_override_enabled());

	// Reduce all sides of the area by 1, so warping works when windows are maximized/fullscreen.
	const Vector2 relative = _get_warped_mouse_motion(p_event, Rect2(Vector2(1, 1), root->get_size() - Vector2(2, 2)));

	if (invert_y_axis) {
		cursor.x_rot -= relative.y * orbit_sensitivity;
	} else {
		cursor.x_rot += relative.y * orbit_sensitivity;
	}
	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.x_rot = CLAMP(cursor.x_rot, -1.57, 1.57);

	if (invert_x_axis) {
		cursor.y_rot -= relative.x * orbit_sensitivity;
	} else {
		cursor.y_rot += relative.x * orbit_sensitivity;
	}

	root->get_override_camera_3d()->set_transform(_get_cursor_transform());
}

Point2 RuntimeNodeSelect::_get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_event, Rect2 p_area) const {
	ERR_FAIL_COND_V(p_event.is_null(), Point2());

	if (warped_mouse_panning_3d) {
		return Input::get_singleton()->warp_mouse_motion(p_event, p_area);
	}

	return p_event->get_relative();
}

Transform3D RuntimeNodeSelect::_get_cursor_transform() {
	Transform3D camera_transform;
	camera_transform.translate_local(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
	camera_transform.translate_local(0, 0, cursor.distance);

	return camera_transform;
}

void RuntimeNodeSelect::_reset_camera_3d() {
	camera_first_override = true;

	cursor = Cursor();
	Window *root = SceneTree::get_singleton()->get_root();
	Camera3D *game_camera = root->is_camera_3d_override_enabled() ? root->get_overridden_camera_3d() : root->get_camera_3d();
	if (game_camera) {
		Transform3D transform = game_camera->get_camera_transform();
		transform.translate_local(0, 0, -cursor.distance);
		cursor.pos = transform.origin;

		cursor.x_rot = -game_camera->get_global_rotation().x;
		cursor.y_rot = -game_camera->get_global_rotation().y;

		cursor.fov_scale = CLAMP(game_camera->get_fov() / camera_fov, CAMERA_MIN_FOV_SCALE, CAMERA_MAX_FOV_SCALE);
	} else {
		cursor.fov_scale = 1.0;
	}

	if (root->is_camera_3d_override_enabled()) {
		Camera3D *override_camera = root->get_override_camera_3d();
		override_camera->set_transform(_get_cursor_transform());
		override_camera->set_perspective(camera_fov * cursor.fov_scale, camera_znear, camera_zfar);
	}
}
#endif // _3D_DISABLED

#endif // DEBUG_ENABLED
