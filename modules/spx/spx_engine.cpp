/**************************************************************************/
/*  spx_engine.cpp                                                        */
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

#include "spx_engine.h"
#include "core/extension/gdextension.h"
#include "core/os/memory.h"
#include "core/os/thread.h"
#include "gdextension_spx_ext.h"
#include "scene/main/window.h"
#include "scene/main/scene_tree.h"
#include "spx_input_mgr.h"
#include "spx_audio_mgr.h"
#include "spx_physic_mgr.h"
#include "spx_sprite_mgr.h"
#include "spx_ui_mgr.h"
#include "spx_camera_mgr.h"
#include "spx_scene_mgr.h"
#include "spx_platform_mgr.h"
#include "spx_res_mgr.h"
#include "spx_ext_mgr.h"
SpxEngine *SpxEngine::singleton = nullptr;

void SpxEngine::register_runtime_panic_callbacks(GDExtensionSpxGlobalRuntimePanicCallback callback) {
	singleton->on_runtime_panic = callback;
}

void SpxEngine::register_runtime_exit_callbacks(GDExtensionSpxGlobalRuntimeExitCallback callback) {
	singleton->on_runtime_exit = callback;
}

static SpxCallbackInfo get_default_spx_callbacks() {
	SpxCallbackInfo callbacks;
	callbacks.func_on_engine_start = [](){};
	callbacks.func_on_engine_fixed_update = [](GdFloat delta){};
	callbacks.func_on_engine_update = [](GdFloat delta){};
	callbacks.func_on_engine_destroy = [](){};
	callbacks.func_on_engine_pause = [](GdBool is_paused){};
	callbacks.func_on_scene_sprite_instantiated = [](GdObj obj,GdString type_name){};
	callbacks.func_on_sprite_ready = [](GdObj obj){};
	callbacks.func_on_sprite_updated = [](GdFloat delta){};
	callbacks.func_on_sprite_fixed_updated = [](GdFloat delta){};
	callbacks.func_on_sprite_destroyed = [](GdObj obj){};
	callbacks.func_on_sprite_frames_set_changed = [](GdObj obj){};
	callbacks.func_on_sprite_animation_changed = [](GdObj obj){};
	callbacks.func_on_sprite_frame_changed = [](GdObj obj){};
	callbacks.func_on_sprite_animation_looped = [](GdObj obj){};
	callbacks.func_on_sprite_animation_finished = [](GdObj obj){};
	callbacks.func_on_sprite_vfx_finished = [](GdObj obj){};
	callbacks.func_on_sprite_screen_exited = [](GdObj obj){};
	callbacks.func_on_sprite_screen_entered = [](GdObj obj){};
	callbacks.func_on_mouse_pressed = [](GdInt keyid){};
	callbacks.func_on_mouse_released = [](GdInt keyid){};
	callbacks.func_on_key_pressed = [](GdInt keyid){};
	callbacks.func_on_key_released = [](GdInt keyid){};
	callbacks.func_on_action_pressed = [](GdString action_name){};
	callbacks.func_on_action_just_pressed = [](GdString action_name){};
	callbacks.func_on_action_just_released = [](GdString action_name){};
	callbacks.func_on_axis_changed = [](GdString action_name, GdFloat value){};
	callbacks.func_on_collision_enter = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_collision_stay = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_collision_exit = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_trigger_enter = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_trigger_stay = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_trigger_exit = [](GdInt self_id, GdInt other_id){};
	callbacks.func_on_ui_ready = [](GdObj obj){};
	callbacks.func_on_ui_updated = [](GdObj obj){};
	callbacks.func_on_ui_destroyed = [](GdObj obj){};
	callbacks.func_on_ui_pressed = [](GdObj obj){};
	callbacks.func_on_ui_released = [](GdObj obj){};
	callbacks.func_on_ui_hovered = [](GdObj obj){};
	callbacks.func_on_ui_clicked = [](GdObj obj){};
	callbacks.func_on_ui_toggle = [](GdObj obj, GdBool is_on){};
	callbacks.func_on_ui_text_changed = [](GdObj obj, GdString text){};
	return callbacks;
}


void SpxEngine::register_callbacks(GDExtensionSpxCallbackInfoPtr callback_ptr) {
	if (singleton != nullptr) {
		print_error("SpxEngine::register_callbacks failed, already initialed! ");
		return;
	}
	singleton = new SpxEngine();
	singleton->mgrs.clear();
	singleton->input = memnew(SpxInputMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->input);
	singleton->audio = memnew(SpxAudioMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->audio);
	singleton->physic = memnew(SpxPhysicMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->physic);
	singleton->sprite = memnew(SpxSpriteMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->sprite);
	singleton->ui = memnew(SpxUiMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->ui);
	singleton->scene = memnew(SpxSceneMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->scene);
	singleton->camera = memnew(SpxCameraMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->camera);
	singleton->platform = memnew(SpxPlatformMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->platform);
	singleton->res = memnew(SpxResMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->res);
	singleton->ext = memnew(SpxExtMgr);
	singleton->mgrs.append((SpxBaseMgr *)singleton->ext);
	
	singleton->callbacks = *(SpxCallbackInfo *)callback_ptr;
	singleton->global_id = 1;
	singleton->is_spx_paused = false;
}

SpxCallbackInfo *SpxEngine::get_callbacks() {
	return &callbacks;
}

GdInt SpxEngine::get_unique_id() {
	return global_id++;
}

Node *SpxEngine::get_spx_root() {
	return spx_root;
}

SceneTree * SpxEngine::get_tree() {
	return tree;
}

Window *SpxEngine::get_root() {
	return tree->get_root();
}

void SpxEngine::set_root_node(SceneTree *p_tree, Node *p_node) {
	this->tree = p_tree;
	spx_root = p_node;
}

void SpxEngine::on_awake() {
	if (has_exit) {
		return;
	}
	for (auto mgr : mgrs) {
		mgr->on_awake();
	}
	for (auto mgr : mgrs) {
		mgr->on_start();
	}
	if (callbacks.func_on_engine_start != nullptr) {
		callbacks.func_on_engine_start();
	}
}

void SpxEngine::on_fixed_update(float delta) {
	if (has_exit || is_spx_paused) {
		return;
	}
	for (auto mgr : mgrs) {
		mgr->on_fixed_update(delta);
	}
	if (callbacks.func_on_engine_fixed_update != nullptr) {
		callbacks.func_on_engine_fixed_update(delta);
	}
}

void SpxEngine::on_update(float delta) {
	if (has_exit || is_spx_paused) {
		return;
	}
	if(is_defer_call_pause){
		_on_godot_pause_changed(defer_pause_value);
		is_defer_call_pause = false;
	}

	for (auto mgr : mgrs) {
		mgr->on_update(delta);
	}
	if (callbacks.func_on_engine_update != nullptr) {
		callbacks.func_on_engine_update(delta);
	}
}

void SpxEngine::on_exit(int exit_code) {
	if (has_exit) {
		return;
	}
	has_exit = true;
	for (auto mgr : mgrs) {
		mgr->on_exit(exit_code);
	}
	// remove all runtime callbacks
	callbacks = get_default_spx_callbacks();
}

void SpxEngine::on_destroy() {
	for (auto mgr : mgrs) {
		mgr->on_destroy();
	}	
	if (!has_exit) {
		if (callbacks.func_on_engine_destroy != nullptr) {
			callbacks.func_on_engine_destroy();
		}
	}
	callbacks = get_default_spx_callbacks();
	
	// Destroy svg global manager
	svgMgr->destroy();
	
	memdelete(input);
	memdelete(audio);
	memdelete(physic);
	memdelete(sprite);
	memdelete(ui);
	memdelete(scene);
	memdelete(camera);
	memdelete(platform);
	memdelete(res);
	memdelete(ext);
	mgrs.clear();
	singleton = nullptr;
}

// SPX Pause functionality implementation with thread safety
void SpxEngine::pause() {
	if (tree != nullptr) {
		if (Thread::is_main_thread()) {
			// Direct call on main thread
			tree->set_pause(true);
			// Directly notify about pause state change
			_on_godot_pause_changed(true);
		} else {
			// Use SceneTree to defer call to main thread
			tree->call_deferred("set_pause", true);
			is_defer_call_pause = true;
			defer_pause_value = true;
			// Defer the pause notification as well
			//callable_mp(this, &SpxEngine::_on_godot_pause_changed).call_deferred(true);
		}
	}
}

void SpxEngine::resume() {
	if (tree != nullptr) {
		if (Thread::is_main_thread()) {
			// Direct call on main thread
			tree->set_pause(false);
			// Directly notify about pause state change
			_on_godot_pause_changed(false);
		} else {
			// Use SceneTree to defer call to main thread
			tree->call_deferred("set_pause", false);
			is_defer_call_pause = true;
			defer_pause_value = false;
		}
	}
}

bool SpxEngine::is_paused() const {
	return is_spx_paused;
}

// Internal method for Godot pause synchronization
void SpxEngine::_on_godot_pause_changed(bool is_godot_paused) {
	if (is_godot_paused != is_spx_paused) {
		is_spx_paused = is_godot_paused;
		
		// Notify all managers about pause/resume
		for (auto mgr : mgrs) {
			if (is_spx_paused) {
				mgr->on_pause();
			} else {
				mgr->on_resume();
			}
		}
		
		// Call the pause callback to notify SPX users
		if (callbacks.func_on_engine_pause != nullptr) {
			callbacks.func_on_engine_pause(is_spx_paused);
		}
	}
}
