
#include "os_web.h"
#include "core/extension/gdextension_interface.h"
#include "modules/spx/spx_engine.h"
#include "godot_js.h"
#include <emscripten.h>

#include "api/javascript_bridge_singleton.h"
#include "display_server_web.h"


static void _godot_js_spx_on_runtime_panic(GdString msg){
	godot_js_spx_on_runtime_panic(msg);
}
static void _godot_js_spx_on_runtime_exit(GdInt code){
	godot_js_spx_on_runtime_exit((int)code);
}

static void _godot_js_spx_on_engine_start(){
	godot_js_spx_on_engine_start();
}
static void _godot_js_spx_on_engine_update(GdFloat delta){
	godot_js_spx_on_engine_update(delta);
}
static void _godot_js_spx_on_engine_fixed_update(GdFloat delta){
	godot_js_spx_on_engine_fixed_update(delta);
}
static void _godot_js_spx_on_engine_destroy(){
	godot_js_spx_on_engine_destroy();
}
static void _godot_js_spx_on_engine_pause(GdBool is_paused){
	godot_js_spx_on_engine_pause(is_paused);
}

static void _godot_js_spx_on_scene_sprite_instantiated(GdObj obj,GdString type_name){
	godot_js_spx_on_scene_sprite_instantiated(&obj, type_name);
}

static void _godot_js_spx_on_sprite_ready(GdObj obj){
	godot_js_spx_on_sprite_ready(&obj);
}
static void _godot_js_spx_on_sprite_updated(GdFloat delta){
	godot_js_spx_on_sprite_updated(delta);
}
static void _godot_js_spx_on_sprite_fixed_updated(GdFloat delta){
	godot_js_spx_on_sprite_fixed_updated(delta);
}
static void _godot_js_spx_on_sprite_destroyed(GdObj obj){
	godot_js_spx_on_sprite_destroyed(&obj);
}

static void _godot_js_spx_on_sprite_frames_set_changed(GdObj obj){
	godot_js_spx_on_sprite_frames_set_changed(&obj);
}
static void _godot_js_spx_on_sprite_animation_changed(GdObj obj){
	godot_js_spx_on_sprite_animation_changed(&obj);
}
static void _godot_js_spx_on_sprite_frame_changed(GdObj obj){
	godot_js_spx_on_sprite_frame_changed(&obj);
}
static void _godot_js_spx_on_sprite_animation_looped(GdObj obj){
	godot_js_spx_on_sprite_animation_looped(&obj);
}
static void _godot_js_spx_on_sprite_animation_finished(GdObj obj){
	godot_js_spx_on_sprite_animation_finished(&obj);
}

static void _godot_js_spx_on_sprite_vfx_finished(GdObj obj){
	godot_js_spx_on_sprite_vfx_finished(&obj);
}

static void _godot_js_spx_on_sprite_screen_exited(GdObj obj){
	godot_js_spx_on_sprite_screen_exited(&obj);
}
static void _godot_js_spx_on_sprite_screen_entered(GdObj obj){
	godot_js_spx_on_sprite_screen_entered(&obj);
}

static void _godot_js_spx_on_mouse_pressed(GdInt keyid){
	godot_js_spx_on_mouse_pressed(&keyid);
}
static void _godot_js_spx_on_mouse_released(GdInt keyid){
	godot_js_spx_on_mouse_released(&keyid);
}
static void _godot_js_spx_on_key_pressed(GdInt keyid){
	godot_js_spx_on_key_pressed(&keyid);
}
static void _godot_js_spx_on_key_released(GdInt keyid){
	godot_js_spx_on_key_released(&keyid);
}
static void _godot_js_spx_on_action_pressed(GdString action_name){
	godot_js_spx_on_action_pressed(action_name);
}
static void _godot_js_spx_on_action_just_pressed(GdString action_name){
	godot_js_spx_on_action_just_pressed(action_name);
}
static void _godot_js_spx_on_action_just_released(GdString action_name){
	godot_js_spx_on_action_just_released(action_name);
}
static void _godot_js_spx_on_axis_changed(GdString action_name, GdFloat value){
	godot_js_spx_on_axis_changed(action_name, value);
}

static void _godot_js_spx_on_collision_enter(GdInt self_id, GdInt other_id){
	godot_js_spx_on_collision_enter(&self_id, &other_id);
}
static void _godot_js_spx_on_collision_stay(GdInt self_id, GdInt other_id){
	godot_js_spx_on_collision_stay(&self_id, &other_id);
}
static void _godot_js_spx_on_collision_exit(GdInt self_id, GdInt other_id){
	godot_js_spx_on_collision_exit(&self_id, &other_id);
}
static void _godot_js_spx_on_trigger_enter(GdInt self_id, GdInt other_id){
	godot_js_spx_on_trigger_enter(&self_id, &other_id);
}
static void _godot_js_spx_on_trigger_stay(GdInt self_id, GdInt other_id){
	godot_js_spx_on_trigger_stay(&self_id, &other_id);
}
static void _godot_js_spx_on_trigger_exit(GdInt self_id, GdInt other_id){
	godot_js_spx_on_trigger_exit(&self_id, &other_id);
}

static void _godot_js_spx_on_ui_ready(GdObj obj){
	godot_js_spx_on_ui_ready(&obj);
}
static void _godot_js_spx_on_ui_updated(GdObj obj){
	godot_js_spx_on_ui_updated(&obj);
}
static void _godot_js_spx_on_ui_destroyed(GdObj obj){
	godot_js_spx_on_ui_destroyed(&obj);
}

static void _godot_js_spx_on_ui_pressed(GdObj obj){
	godot_js_spx_on_ui_pressed(&obj);
}
static void _godot_js_spx_on_ui_released(GdObj obj){
	godot_js_spx_on_ui_released(&obj);
}
static void _godot_js_spx_on_ui_hovered(GdObj obj){
	godot_js_spx_on_ui_hovered(&obj);
}
static void _godot_js_spx_on_ui_clicked(GdObj obj){
	godot_js_spx_on_ui_clicked(&obj);
}
static void _godot_js_spx_on_ui_toggle(GdObj obj, GdBool is_on){
	godot_js_spx_on_ui_toggle(&obj, is_on);
}
static void _godot_js_spx_on_ui_text_changed(GdObj obj, GdString text){
	godot_js_spx_on_ui_text_changed(&obj, text);
}


void OS_Web::register_spx_callbacks() {
 	SpxCallbackInfo* callback_infos = memnew(SpxCallbackInfo);
	// gdspx register callbacks
	callback_infos->func_on_engine_start = &_godot_js_spx_on_engine_start;
	callback_infos->func_on_engine_update = &_godot_js_spx_on_engine_update;
	callback_infos->func_on_engine_fixed_update = &_godot_js_spx_on_engine_fixed_update;
	callback_infos->func_on_engine_destroy = &_godot_js_spx_on_engine_destroy;
	callback_infos->func_on_engine_pause = &_godot_js_spx_on_engine_pause;
	callback_infos->func_on_scene_sprite_instantiated = &_godot_js_spx_on_scene_sprite_instantiated;
	callback_infos->func_on_sprite_ready = &_godot_js_spx_on_sprite_ready;
	callback_infos->func_on_sprite_updated = &_godot_js_spx_on_sprite_updated;
	callback_infos->func_on_sprite_fixed_updated = &_godot_js_spx_on_sprite_fixed_updated;
	callback_infos->func_on_sprite_destroyed = &_godot_js_spx_on_sprite_destroyed;
	callback_infos->func_on_sprite_frames_set_changed = &_godot_js_spx_on_sprite_frames_set_changed;
	callback_infos->func_on_sprite_animation_changed = &_godot_js_spx_on_sprite_animation_changed;
	callback_infos->func_on_sprite_frame_changed = &_godot_js_spx_on_sprite_frame_changed;
	callback_infos->func_on_sprite_animation_looped = &_godot_js_spx_on_sprite_animation_looped;
	callback_infos->func_on_sprite_animation_finished = &_godot_js_spx_on_sprite_animation_finished;
	callback_infos->func_on_sprite_vfx_finished = &_godot_js_spx_on_sprite_vfx_finished;
	callback_infos->func_on_sprite_screen_exited = &_godot_js_spx_on_sprite_screen_exited;
	callback_infos->func_on_sprite_screen_entered = &_godot_js_spx_on_sprite_screen_entered;
	callback_infos->func_on_mouse_pressed = &_godot_js_spx_on_mouse_pressed;
	callback_infos->func_on_mouse_released = &_godot_js_spx_on_mouse_released;
	callback_infos->func_on_key_pressed = &_godot_js_spx_on_key_pressed;
	callback_infos->func_on_key_released = &_godot_js_spx_on_key_released;
	callback_infos->func_on_action_pressed = &_godot_js_spx_on_action_pressed;
	callback_infos->func_on_action_just_pressed = &_godot_js_spx_on_action_just_pressed;
	callback_infos->func_on_action_just_released = &_godot_js_spx_on_action_just_released;
	callback_infos->func_on_axis_changed = &_godot_js_spx_on_axis_changed;
	callback_infos->func_on_collision_enter = &_godot_js_spx_on_collision_enter;
	callback_infos->func_on_collision_stay = &_godot_js_spx_on_collision_stay;
	callback_infos->func_on_collision_exit = &_godot_js_spx_on_collision_exit;
	callback_infos->func_on_trigger_enter = &_godot_js_spx_on_trigger_enter;
	callback_infos->func_on_trigger_stay = &_godot_js_spx_on_trigger_stay;
	callback_infos->func_on_trigger_exit = &_godot_js_spx_on_trigger_exit;
	callback_infos->func_on_ui_ready = &_godot_js_spx_on_ui_ready;
	callback_infos->func_on_ui_updated = &_godot_js_spx_on_ui_updated;
	callback_infos->func_on_ui_destroyed = &_godot_js_spx_on_ui_destroyed;
	callback_infos->func_on_ui_pressed = &_godot_js_spx_on_ui_pressed;
	callback_infos->func_on_ui_released = &_godot_js_spx_on_ui_released;
	callback_infos->func_on_ui_hovered = &_godot_js_spx_on_ui_hovered;
	callback_infos->func_on_ui_clicked = &_godot_js_spx_on_ui_clicked;
	callback_infos->func_on_ui_toggle = &_godot_js_spx_on_ui_toggle;
	callback_infos->func_on_ui_text_changed = &_godot_js_spx_on_ui_text_changed;
	
	SpxEngine::register_callbacks(callback_infos);
	SpxEngine::register_runtime_panic_callbacks(_godot_js_spx_on_runtime_panic);
	SpxEngine::register_runtime_exit_callbacks(_godot_js_spx_on_runtime_exit);
}