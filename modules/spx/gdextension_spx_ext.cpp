/**************************************************************************/
/*  gdextension_spx_ext.cpp                                               */
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

#include "gdextension_spx_ext.h"
#include "core/extension/gdextension.h"
#include "core/extension/gdextension_special_compat_hashes.h"
#include "core/variant/variant.h"
#include "gdextension_interface.h"
#include "scene/main/window.h"
#include "spx_engine.h"
#include "spx_audio_mgr.h"
#include "spx_camera_mgr.h"
#include "spx_ext_mgr.h"
#include "spx_input_mgr.h"
#include "spx_physic_mgr.h"
#include "spx_platform_mgr.h"
#include "spx_res_mgr.h"
#include "spx_scene_mgr.h"
#include "spx_sprite_mgr.h"
#include "spx_ui_mgr.h"

#define audioMgr SpxEngine::get_singleton()->get_audio()
#define cameraMgr SpxEngine::get_singleton()->get_camera()
#define extMgr SpxEngine::get_singleton()->get_ext()
#define inputMgr SpxEngine::get_singleton()->get_input()
#define physicMgr SpxEngine::get_singleton()->get_physic()
#define platformMgr SpxEngine::get_singleton()->get_platform()
#define resMgr SpxEngine::get_singleton()->get_res()
#define sceneMgr SpxEngine::get_singleton()->get_scene()
#define spriteMgr SpxEngine::get_singleton()->get_sprite()
#define uiMgr SpxEngine::get_singleton()->get_ui()

#define REGISTER_SPX_INTERFACE_FUNC(m_name) GDExtension::register_interface_function( #m_name, (GDExtensionInterfaceFunctionPtr)&gdextension_##m_name)
static void gdextension_spx_global_register_callbacks(GDExtensionSpxCallbackInfoPtr callback_ptr) {
	SpxEngine::register_callbacks(callback_ptr);
}
static void gdextension_spx_audio_stop_all() {
	 audioMgr->stop_all();
}
static void gdextension_spx_audio_create_audio(GdObj* ret_val) {
	*ret_val = audioMgr->create_audio();
}
static void gdextension_spx_audio_destroy_audio(GdObj obj) {
	 audioMgr->destroy_audio(obj);
}
static void gdextension_spx_audio_set_pitch(GdObj obj,GdFloat pitch) {
	 audioMgr->set_pitch(obj, pitch);
}
static void gdextension_spx_audio_get_pitch(GdObj obj,GdFloat* ret_val) {
	*ret_val = audioMgr->get_pitch(obj);
}
static void gdextension_spx_audio_set_pan(GdObj obj,GdFloat pan) {
	 audioMgr->set_pan(obj, pan);
}
static void gdextension_spx_audio_get_pan(GdObj obj,GdFloat* ret_val) {
	*ret_val = audioMgr->get_pan(obj);
}
static void gdextension_spx_audio_set_volume(GdObj obj,GdFloat volume) {
	 audioMgr->set_volume(obj, volume);
}
static void gdextension_spx_audio_get_volume(GdObj obj,GdFloat* ret_val) {
	*ret_val = audioMgr->get_volume(obj);
}
static void gdextension_spx_audio_play(GdObj obj,GdString path,GdInt* ret_val) {
	*ret_val = audioMgr->play(obj, path);
}
static void gdextension_spx_audio_pause(GdInt aid) {
	 audioMgr->pause(aid);
}
static void gdextension_spx_audio_resume(GdInt aid) {
	 audioMgr->resume(aid);
}
static void gdextension_spx_audio_stop(GdInt aid) {
	 audioMgr->stop(aid);
}
static void gdextension_spx_audio_set_loop(GdInt aid,GdBool loop) {
	 audioMgr->set_loop(aid, loop);
}
static void gdextension_spx_audio_get_loop(GdInt aid,GdBool* ret_val) {
	*ret_val = audioMgr->get_loop(aid);
}
static void gdextension_spx_audio_get_timer(GdInt aid,GdFloat* ret_val) {
	*ret_val = audioMgr->get_timer(aid);
}
static void gdextension_spx_audio_set_timer(GdInt aid,GdFloat time) {
	 audioMgr->set_timer(aid, time);
}
static void gdextension_spx_audio_is_playing(GdInt aid,GdBool* ret_val) {
	*ret_val = audioMgr->is_playing(aid);
}
static void gdextension_spx_camera_get_camera_position(GdVec2* ret_val) {
	*ret_val = cameraMgr->get_camera_position();
}
static void gdextension_spx_camera_set_camera_position(GdVec2 position) {
	 cameraMgr->set_camera_position(position);
}
static void gdextension_spx_camera_get_camera_zoom(GdVec2* ret_val) {
	*ret_val = cameraMgr->get_camera_zoom();
}
static void gdextension_spx_camera_set_camera_zoom(GdVec2 size) {
	 cameraMgr->set_camera_zoom(size);
}
static void gdextension_spx_camera_get_viewport_rect(GdRect2* ret_val) {
	*ret_val = cameraMgr->get_viewport_rect();
}
static void gdextension_spx_ext_request_exit(GdInt exit_code) {
	 extMgr->request_exit(exit_code);
}
static void gdextension_spx_ext_on_runtime_panic(GdString msg) {
	 extMgr->on_runtime_panic(msg);
}
static void gdextension_spx_ext_destroy_all_pens() {
	 extMgr->destroy_all_pens();
}
static void gdextension_spx_ext_create_pen(GdObj* ret_val) {
	*ret_val = extMgr->create_pen();
}
static void gdextension_spx_ext_destroy_pen(GdObj obj) {
	 extMgr->destroy_pen(obj);
}
static void gdextension_spx_ext_pen_stamp(GdObj obj) {
	 extMgr->pen_stamp(obj);
}
static void gdextension_spx_ext_move_pen_to(GdObj obj,GdVec2 position) {
	 extMgr->move_pen_to(obj, position);
}
static void gdextension_spx_ext_pen_down(GdObj obj,GdBool move_by_mouse) {
	 extMgr->pen_down(obj, move_by_mouse);
}
static void gdextension_spx_ext_pen_up(GdObj obj) {
	 extMgr->pen_up(obj);
}
static void gdextension_spx_ext_set_pen_color_to(GdObj obj,GdColor color) {
	 extMgr->set_pen_color_to(obj, color);
}
static void gdextension_spx_ext_change_pen_by(GdObj obj,GdInt property,GdFloat amount) {
	 extMgr->change_pen_by(obj, property, amount);
}
static void gdextension_spx_ext_set_pen_to(GdObj obj,GdInt property,GdFloat value) {
	 extMgr->set_pen_to(obj, property, value);
}
static void gdextension_spx_ext_change_pen_size_by(GdObj obj,GdFloat amount) {
	 extMgr->change_pen_size_by(obj, amount);
}
static void gdextension_spx_ext_set_pen_size_to(GdObj obj,GdFloat size) {
	 extMgr->set_pen_size_to(obj, size);
}
static void gdextension_spx_ext_set_pen_stamp_texture(GdObj obj,GdString texture_path) {
	 extMgr->set_pen_stamp_texture(obj, texture_path);
}
static void gdextension_spx_input_get_mouse_pos(GdVec2* ret_val) {
	*ret_val = inputMgr->get_mouse_pos();
}
static void gdextension_spx_input_get_key(GdInt key,GdBool* ret_val) {
	*ret_val = inputMgr->get_key(key);
}
static void gdextension_spx_input_get_mouse_state(GdInt mouse_id,GdBool* ret_val) {
	*ret_val = inputMgr->get_mouse_state(mouse_id);
}
static void gdextension_spx_input_get_key_state(GdInt key,GdInt* ret_val) {
	*ret_val = inputMgr->get_key_state(key);
}
static void gdextension_spx_input_get_axis(GdString neg_action,GdString pos_action,GdFloat* ret_val) {
	*ret_val = inputMgr->get_axis(neg_action, pos_action);
}
static void gdextension_spx_input_is_action_pressed(GdString action,GdBool* ret_val) {
	*ret_val = inputMgr->is_action_pressed(action);
}
static void gdextension_spx_input_is_action_just_pressed(GdString action,GdBool* ret_val) {
	*ret_val = inputMgr->is_action_just_pressed(action);
}
static void gdextension_spx_input_is_action_just_released(GdString action,GdBool* ret_val) {
	*ret_val = inputMgr->is_action_just_released(action);
}
static void gdextension_spx_physic_raycast(GdVec2 from,GdVec2 to,GdInt collision_mask,GdObj* ret_val) {
	*ret_val = physicMgr->raycast(from, to, collision_mask);
}
static void gdextension_spx_physic_check_collision(GdVec2 from,GdVec2 to,GdInt collision_mask,GdBool collide_with_areas,GdBool collide_with_bodies,GdBool* ret_val) {
	*ret_val = physicMgr->check_collision(from, to, collision_mask, collide_with_areas, collide_with_bodies);
}
static void gdextension_spx_physic_check_touched_camera_boundaries(GdObj obj,GdInt* ret_val) {
	*ret_val = physicMgr->check_touched_camera_boundaries(obj);
}
static void gdextension_spx_physic_check_touched_camera_boundary(GdObj obj,GdInt board_type,GdBool* ret_val) {
	*ret_val = physicMgr->check_touched_camera_boundary(obj, board_type);
}
static void gdextension_spx_physic_set_collision_system_type(GdBool is_collision_by_alpha) {
	 physicMgr->set_collision_system_type(is_collision_by_alpha);
}
static void gdextension_spx_platform_set_window_position(GdVec2 pos) {
	 platformMgr->set_window_position(pos);
}
static void gdextension_spx_platform_get_window_position(GdVec2* ret_val) {
	*ret_val = platformMgr->get_window_position();
}
static void gdextension_spx_platform_set_window_size(GdInt width,GdInt height) {
	 platformMgr->set_window_size(width, height);
}
static void gdextension_spx_platform_get_window_size(GdVec2* ret_val) {
	*ret_val = platformMgr->get_window_size();
}
static void gdextension_spx_platform_set_window_title(GdString title) {
	 platformMgr->set_window_title(title);
}
static void gdextension_spx_platform_get_window_title(GdString* ret_val) {
	*ret_val = platformMgr->get_window_title();
}
static void gdextension_spx_platform_set_window_fullscreen(GdBool enable) {
	 platformMgr->set_window_fullscreen(enable);
}
static void gdextension_spx_platform_is_window_fullscreen(GdBool* ret_val) {
	*ret_val = platformMgr->is_window_fullscreen();
}
static void gdextension_spx_platform_set_debug_mode(GdBool enable) {
	 platformMgr->set_debug_mode(enable);
}
static void gdextension_spx_platform_is_debug_mode(GdBool* ret_val) {
	*ret_val = platformMgr->is_debug_mode();
}
static void gdextension_spx_platform_get_time_scale(GdFloat* ret_val) {
	*ret_val = platformMgr->get_time_scale();
}
static void gdextension_spx_platform_set_time_scale(GdFloat time_scale) {
	 platformMgr->set_time_scale(time_scale);
}
static void gdextension_spx_platform_get_persistant_data_dir(GdString* ret_val) {
	*ret_val = platformMgr->get_persistant_data_dir();
}
static void gdextension_spx_platform_set_persistant_data_dir(GdString path) {
	 platformMgr->set_persistant_data_dir(path);
}
static void gdextension_spx_platform_is_in_persistant_data_dir(GdString path,GdBool* ret_val) {
	*ret_val = platformMgr->is_in_persistant_data_dir(path);
}
static void gdextension_spx_res_create_animation(GdString sprite_type_name,GdString anim_name,GdString context,GdInt fps,GdBool is_altas) {
	 resMgr->create_animation(sprite_type_name, anim_name, context, fps, is_altas);
}
static void gdextension_spx_res_set_load_mode(GdBool is_direct_mode) {
	 resMgr->set_load_mode(is_direct_mode);
}
static void gdextension_spx_res_get_load_mode(GdBool* ret_val) {
	*ret_val = resMgr->get_load_mode();
}
static void gdextension_spx_res_get_bound_from_alpha(GdString p_path,GdRect2* ret_val) {
	*ret_val = resMgr->get_bound_from_alpha(p_path);
}
static void gdextension_spx_res_get_image_size(GdString p_path,GdVec2* ret_val) {
	*ret_val = resMgr->get_image_size(p_path);
}
static void gdextension_spx_res_read_all_text(GdString p_path,GdString* ret_val) {
	*ret_val = resMgr->read_all_text(p_path);
}
static void gdextension_spx_res_has_file(GdString p_path,GdBool* ret_val) {
	*ret_val = resMgr->has_file(p_path);
}
static void gdextension_spx_res_reload_texture(GdString path) {
	 resMgr->reload_texture(path);
}
static void gdextension_spx_res_free_str(GdString str) {
	 resMgr->free_str(str);
}
static void gdextension_spx_res_set_default_font(GdString font_path) {
	 resMgr->set_default_font(font_path);
}
static void gdextension_spx_scene_change_scene_to_file(GdString path) {
	 sceneMgr->change_scene_to_file(path);
}
static void gdextension_spx_scene_destroy_all_sprites() {
	 sceneMgr->destroy_all_sprites();
}
static void gdextension_spx_scene_reload_current_scene(GdInt* ret_val) {
	*ret_val = sceneMgr->reload_current_scene();
}
static void gdextension_spx_scene_unload_current_scene() {
	 sceneMgr->unload_current_scene();
}
static void gdextension_spx_sprite_set_dont_destroy_on_load(GdObj obj) {
	 spriteMgr->set_dont_destroy_on_load(obj);
}
static void gdextension_spx_sprite_set_process(GdObj obj,GdBool is_on) {
	 spriteMgr->set_process(obj, is_on);
}
static void gdextension_spx_sprite_set_physic_process(GdObj obj,GdBool is_on) {
	 spriteMgr->set_physic_process(obj, is_on);
}
static void gdextension_spx_sprite_set_type_name(GdObj obj,GdString type_name) {
	 spriteMgr->set_type_name(obj, type_name);
}
static void gdextension_spx_sprite_set_child_position(GdObj obj,GdString path,GdVec2 pos) {
	 spriteMgr->set_child_position(obj, path, pos);
}
static void gdextension_spx_sprite_get_child_position(GdObj obj,GdString path,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_child_position(obj, path);
}
static void gdextension_spx_sprite_set_child_rotation(GdObj obj,GdString path,GdFloat rot) {
	 spriteMgr->set_child_rotation(obj, path, rot);
}
static void gdextension_spx_sprite_get_child_rotation(GdObj obj,GdString path,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_child_rotation(obj, path);
}
static void gdextension_spx_sprite_set_child_scale(GdObj obj,GdString path,GdVec2 scale) {
	 spriteMgr->set_child_scale(obj, path, scale);
}
static void gdextension_spx_sprite_get_child_scale(GdObj obj,GdString path,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_child_scale(obj, path);
}
static void gdextension_spx_sprite_check_collision(GdObj obj,GdObj target,GdBool is_src_trigger,GdBool is_dst_trigger,GdBool* ret_val) {
	*ret_val = spriteMgr->check_collision(obj, target, is_src_trigger, is_dst_trigger);
}
static void gdextension_spx_sprite_check_collision_with_point(GdObj obj,GdVec2 point,GdBool is_trigger,GdBool* ret_val) {
	*ret_val = spriteMgr->check_collision_with_point(obj, point, is_trigger);
}
static void gdextension_spx_sprite_create_backdrop(GdString path,GdObj* ret_val) {
	*ret_val = spriteMgr->create_backdrop(path);
}
static void gdextension_spx_sprite_create_sprite(GdString path,GdObj* ret_val) {
	*ret_val = spriteMgr->create_sprite(path);
}
static void gdextension_spx_sprite_clone_sprite(GdObj obj,GdObj* ret_val) {
	*ret_val = spriteMgr->clone_sprite(obj);
}
static void gdextension_spx_sprite_destroy_sprite(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->destroy_sprite(obj);
}
static void gdextension_spx_sprite_is_sprite_alive(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_sprite_alive(obj);
}
static void gdextension_spx_sprite_set_position(GdObj obj,GdVec2 pos) {
	 spriteMgr->set_position(obj, pos);
}
static void gdextension_spx_sprite_get_position(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_position(obj);
}
static void gdextension_spx_sprite_set_rotation(GdObj obj,GdFloat rot) {
	 spriteMgr->set_rotation(obj, rot);
}
static void gdextension_spx_sprite_get_rotation(GdObj obj,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_rotation(obj);
}
static void gdextension_spx_sprite_set_scale(GdObj obj,GdVec2 scale) {
	 spriteMgr->set_scale(obj, scale);
}
static void gdextension_spx_sprite_get_scale(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_scale(obj);
}
static void gdextension_spx_sprite_set_render_scale(GdObj obj,GdVec2 scale) {
	 spriteMgr->set_render_scale(obj, scale);
}
static void gdextension_spx_sprite_get_render_scale(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_render_scale(obj);
}
static void gdextension_spx_sprite_set_color(GdObj obj,GdColor color) {
	 spriteMgr->set_color(obj, color);
}
static void gdextension_spx_sprite_get_color(GdObj obj,GdColor* ret_val) {
	*ret_val = spriteMgr->get_color(obj);
}
static void gdextension_spx_sprite_set_material_shader(GdObj obj,GdString path) {
	 spriteMgr->set_material_shader(obj, path);
}
static void gdextension_spx_sprite_get_material_shader(GdObj obj,GdString* ret_val) {
	*ret_val = spriteMgr->get_material_shader(obj);
}
static void gdextension_spx_sprite_set_material_params(GdObj obj,GdString effect,GdFloat amount) {
	 spriteMgr->set_material_params(obj, effect, amount);
}
static void gdextension_spx_sprite_get_material_params(GdObj obj,GdString effect,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_material_params(obj, effect);
}
static void gdextension_spx_sprite_set_material_params_vec(GdObj obj,GdString effect,GdFloat x,GdFloat y,GdFloat z,GdFloat w) {
	 spriteMgr->set_material_params_vec(obj, effect, x, y, z, w);
}
static void gdextension_spx_sprite_set_material_params_vec4(GdObj obj,GdString effect,GdVec4 vec4) {
	 spriteMgr->set_material_params_vec4(obj, effect, vec4);
}
static void gdextension_spx_sprite_get_material_params_vec4(GdObj obj,GdString effect,GdVec4* ret_val) {
	*ret_val = spriteMgr->get_material_params_vec4(obj, effect);
}
static void gdextension_spx_sprite_set_material_params_color(GdObj obj,GdString effect,GdColor color) {
	 spriteMgr->set_material_params_color(obj, effect, color);
}
static void gdextension_spx_sprite_get_material_params_color(GdObj obj,GdString effect,GdColor* ret_val) {
	*ret_val = spriteMgr->get_material_params_color(obj, effect);
}
static void gdextension_spx_sprite_set_texture_altas(GdObj obj,GdString path,GdRect2 rect2) {
	 spriteMgr->set_texture_altas(obj, path, rect2);
}
static void gdextension_spx_sprite_set_texture(GdObj obj,GdString path) {
	 spriteMgr->set_texture(obj, path);
}
static void gdextension_spx_sprite_set_texture_altas_direct(GdObj obj,GdString path,GdRect2 rect2) {
	 spriteMgr->set_texture_altas_direct(obj, path, rect2);
}
static void gdextension_spx_sprite_set_texture_direct(GdObj obj,GdString path) {
	 spriteMgr->set_texture_direct(obj, path);
}
static void gdextension_spx_sprite_get_texture(GdObj obj,GdString* ret_val) {
	*ret_val = spriteMgr->get_texture(obj);
}
static void gdextension_spx_sprite_set_visible(GdObj obj,GdBool visible) {
	 spriteMgr->set_visible(obj, visible);
}
static void gdextension_spx_sprite_get_visible(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->get_visible(obj);
}
static void gdextension_spx_sprite_get_z_index(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_z_index(obj);
}
static void gdextension_spx_sprite_set_z_index(GdObj obj,GdInt z) {
	 spriteMgr->set_z_index(obj, z);
}
static void gdextension_spx_sprite_play_anim(GdObj obj,GdString p_name,GdFloat p_speed,GdBool isLoop,GdBool p_revert) {
	 spriteMgr->play_anim(obj, p_name, p_speed, isLoop, p_revert);
}
static void gdextension_spx_sprite_play_backwards_anim(GdObj obj,GdString p_name) {
	 spriteMgr->play_backwards_anim(obj, p_name);
}
static void gdextension_spx_sprite_pause_anim(GdObj obj) {
	 spriteMgr->pause_anim(obj);
}
static void gdextension_spx_sprite_stop_anim(GdObj obj) {
	 spriteMgr->stop_anim(obj);
}
static void gdextension_spx_sprite_is_playing_anim(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_playing_anim(obj);
}
static void gdextension_spx_sprite_set_anim(GdObj obj,GdString p_name) {
	 spriteMgr->set_anim(obj, p_name);
}
static void gdextension_spx_sprite_get_anim(GdObj obj,GdString* ret_val) {
	*ret_val = spriteMgr->get_anim(obj);
}
static void gdextension_spx_sprite_set_anim_frame(GdObj obj,GdInt p_frame) {
	 spriteMgr->set_anim_frame(obj, p_frame);
}
static void gdextension_spx_sprite_get_anim_frame(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_anim_frame(obj);
}
static void gdextension_spx_sprite_set_anim_speed_scale(GdObj obj,GdFloat p_speed_scale) {
	 spriteMgr->set_anim_speed_scale(obj, p_speed_scale);
}
static void gdextension_spx_sprite_get_anim_speed_scale(GdObj obj,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_anim_speed_scale(obj);
}
static void gdextension_spx_sprite_get_anim_playing_speed(GdObj obj,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_anim_playing_speed(obj);
}
static void gdextension_spx_sprite_set_anim_centered(GdObj obj,GdBool p_center) {
	 spriteMgr->set_anim_centered(obj, p_center);
}
static void gdextension_spx_sprite_is_anim_centered(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_anim_centered(obj);
}
static void gdextension_spx_sprite_set_anim_offset(GdObj obj,GdVec2 p_offset) {
	 spriteMgr->set_anim_offset(obj, p_offset);
}
static void gdextension_spx_sprite_get_anim_offset(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_anim_offset(obj);
}
static void gdextension_spx_sprite_set_anim_flip_h(GdObj obj,GdBool p_flip) {
	 spriteMgr->set_anim_flip_h(obj, p_flip);
}
static void gdextension_spx_sprite_is_anim_flipped_h(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_anim_flipped_h(obj);
}
static void gdextension_spx_sprite_set_anim_flip_v(GdObj obj,GdBool p_flip) {
	 spriteMgr->set_anim_flip_v(obj, p_flip);
}
static void gdextension_spx_sprite_is_anim_flipped_v(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_anim_flipped_v(obj);
}
static void gdextension_spx_sprite_set_velocity(GdObj obj,GdVec2 velocity) {
	 spriteMgr->set_velocity(obj, velocity);
}
static void gdextension_spx_sprite_get_velocity(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_velocity(obj);
}
static void gdextension_spx_sprite_is_on_floor(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_floor(obj);
}
static void gdextension_spx_sprite_is_on_floor_only(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_floor_only(obj);
}
static void gdextension_spx_sprite_is_on_wall(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_wall(obj);
}
static void gdextension_spx_sprite_is_on_wall_only(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_wall_only(obj);
}
static void gdextension_spx_sprite_is_on_ceiling(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_ceiling(obj);
}
static void gdextension_spx_sprite_is_on_ceiling_only(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_on_ceiling_only(obj);
}
static void gdextension_spx_sprite_get_last_motion(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_last_motion(obj);
}
static void gdextension_spx_sprite_get_position_delta(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_position_delta(obj);
}
static void gdextension_spx_sprite_get_floor_normal(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_floor_normal(obj);
}
static void gdextension_spx_sprite_get_wall_normal(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_wall_normal(obj);
}
static void gdextension_spx_sprite_get_real_velocity(GdObj obj,GdVec2* ret_val) {
	*ret_val = spriteMgr->get_real_velocity(obj);
}
static void gdextension_spx_sprite_move_and_slide(GdObj obj) {
	 spriteMgr->move_and_slide(obj);
}
static void gdextension_spx_sprite_set_gravity(GdObj obj,GdFloat gravity) {
	 spriteMgr->set_gravity(obj, gravity);
}
static void gdextension_spx_sprite_get_gravity(GdObj obj,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_gravity(obj);
}
static void gdextension_spx_sprite_set_mass(GdObj obj,GdFloat mass) {
	 spriteMgr->set_mass(obj, mass);
}
static void gdextension_spx_sprite_get_mass(GdObj obj,GdFloat* ret_val) {
	*ret_val = spriteMgr->get_mass(obj);
}
static void gdextension_spx_sprite_add_force(GdObj obj,GdVec2 force) {
	 spriteMgr->add_force(obj, force);
}
static void gdextension_spx_sprite_add_impulse(GdObj obj,GdVec2 impulse) {
	 spriteMgr->add_impulse(obj, impulse);
}
static void gdextension_spx_sprite_set_collision_layer(GdObj obj,GdInt layer) {
	 spriteMgr->set_collision_layer(obj, layer);
}
static void gdextension_spx_sprite_get_collision_layer(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_collision_layer(obj);
}
static void gdextension_spx_sprite_set_collision_mask(GdObj obj,GdInt mask) {
	 spriteMgr->set_collision_mask(obj, mask);
}
static void gdextension_spx_sprite_get_collision_mask(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_collision_mask(obj);
}
static void gdextension_spx_sprite_set_trigger_layer(GdObj obj,GdInt layer) {
	 spriteMgr->set_trigger_layer(obj, layer);
}
static void gdextension_spx_sprite_get_trigger_layer(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_trigger_layer(obj);
}
static void gdextension_spx_sprite_set_trigger_mask(GdObj obj,GdInt mask) {
	 spriteMgr->set_trigger_mask(obj, mask);
}
static void gdextension_spx_sprite_get_trigger_mask(GdObj obj,GdInt* ret_val) {
	*ret_val = spriteMgr->get_trigger_mask(obj);
}
static void gdextension_spx_sprite_set_collider_rect(GdObj obj,GdVec2 center,GdVec2 size) {
	 spriteMgr->set_collider_rect(obj, center, size);
}
static void gdextension_spx_sprite_set_collider_circle(GdObj obj,GdVec2 center,GdFloat radius) {
	 spriteMgr->set_collider_circle(obj, center, radius);
}
static void gdextension_spx_sprite_set_collider_capsule(GdObj obj,GdVec2 center,GdVec2 size) {
	 spriteMgr->set_collider_capsule(obj, center, size);
}
static void gdextension_spx_sprite_set_collision_enabled(GdObj obj,GdBool enabled) {
	 spriteMgr->set_collision_enabled(obj, enabled);
}
static void gdextension_spx_sprite_is_collision_enabled(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_collision_enabled(obj);
}
static void gdextension_spx_sprite_set_trigger_rect(GdObj obj,GdVec2 center,GdVec2 size) {
	 spriteMgr->set_trigger_rect(obj, center, size);
}
static void gdextension_spx_sprite_set_trigger_circle(GdObj obj,GdVec2 center,GdFloat radius) {
	 spriteMgr->set_trigger_circle(obj, center, radius);
}
static void gdextension_spx_sprite_set_trigger_capsule(GdObj obj,GdVec2 center,GdVec2 size) {
	 spriteMgr->set_trigger_capsule(obj, center, size);
}
static void gdextension_spx_sprite_set_trigger_enabled(GdObj obj,GdBool trigger) {
	 spriteMgr->set_trigger_enabled(obj, trigger);
}
static void gdextension_spx_sprite_is_trigger_enabled(GdObj obj,GdBool* ret_val) {
	*ret_val = spriteMgr->is_trigger_enabled(obj);
}
static void gdextension_spx_sprite_check_collision_by_color(GdObj obj,GdColor color,GdFloat color_threshold,GdFloat alpha_threshold,GdBool* ret_val) {
	*ret_val = spriteMgr->check_collision_by_color(obj, color, color_threshold, alpha_threshold);
}
static void gdextension_spx_sprite_check_collision_by_alpha(GdObj obj,GdFloat alpha_threshold,GdBool* ret_val) {
	*ret_val = spriteMgr->check_collision_by_alpha(obj, alpha_threshold);
}
static void gdextension_spx_sprite_check_collision_with_sprite_by_alpha(GdObj obj,GdObj obj_b,GdFloat alpha_threshold,GdBool* ret_val) {
	*ret_val = spriteMgr->check_collision_with_sprite_by_alpha(obj, obj_b, alpha_threshold);
}
static void gdextension_spx_ui_bind_node(GdObj obj,GdString rel_path,GdObj* ret_val) {
	*ret_val = uiMgr->bind_node(obj, rel_path);
}
static void gdextension_spx_ui_create_node(GdString path,GdObj* ret_val) {
	*ret_val = uiMgr->create_node(path);
}
static void gdextension_spx_ui_create_button(GdString path,GdString text,GdObj* ret_val) {
	*ret_val = uiMgr->create_button(path, text);
}
static void gdextension_spx_ui_create_label(GdString path,GdString text,GdObj* ret_val) {
	*ret_val = uiMgr->create_label(path, text);
}
static void gdextension_spx_ui_create_image(GdString path,GdObj* ret_val) {
	*ret_val = uiMgr->create_image(path);
}
static void gdextension_spx_ui_create_toggle(GdString path,GdBool value,GdObj* ret_val) {
	*ret_val = uiMgr->create_toggle(path, value);
}
static void gdextension_spx_ui_create_slider(GdString path,GdFloat value,GdObj* ret_val) {
	*ret_val = uiMgr->create_slider(path, value);
}
static void gdextension_spx_ui_create_input(GdString path,GdString text,GdObj* ret_val) {
	*ret_val = uiMgr->create_input(path, text);
}
static void gdextension_spx_ui_destroy_node(GdObj obj,GdBool* ret_val) {
	*ret_val = uiMgr->destroy_node(obj);
}
static void gdextension_spx_ui_get_type(GdObj obj,GdInt* ret_val) {
	*ret_val = uiMgr->get_type(obj);
}
static void gdextension_spx_ui_set_text(GdObj obj,GdString text) {
	 uiMgr->set_text(obj, text);
}
static void gdextension_spx_ui_get_text(GdObj obj,GdString* ret_val) {
	*ret_val = uiMgr->get_text(obj);
}
static void gdextension_spx_ui_set_texture(GdObj obj,GdString path) {
	 uiMgr->set_texture(obj, path);
}
static void gdextension_spx_ui_get_texture(GdObj obj,GdString* ret_val) {
	*ret_val = uiMgr->get_texture(obj);
}
static void gdextension_spx_ui_set_color(GdObj obj,GdColor color) {
	 uiMgr->set_color(obj, color);
}
static void gdextension_spx_ui_get_color(GdObj obj,GdColor* ret_val) {
	*ret_val = uiMgr->get_color(obj);
}
static void gdextension_spx_ui_set_font_size(GdObj obj,GdInt size) {
	 uiMgr->set_font_size(obj, size);
}
static void gdextension_spx_ui_get_font_size(GdObj obj,GdInt* ret_val) {
	*ret_val = uiMgr->get_font_size(obj);
}
static void gdextension_spx_ui_set_visible(GdObj obj,GdBool visible) {
	 uiMgr->set_visible(obj, visible);
}
static void gdextension_spx_ui_get_visible(GdObj obj,GdBool* ret_val) {
	*ret_val = uiMgr->get_visible(obj);
}
static void gdextension_spx_ui_set_interactable(GdObj obj,GdBool interactable) {
	 uiMgr->set_interactable(obj, interactable);
}
static void gdextension_spx_ui_get_interactable(GdObj obj,GdBool* ret_val) {
	*ret_val = uiMgr->get_interactable(obj);
}
static void gdextension_spx_ui_set_rect(GdObj obj,GdRect2 rect) {
	 uiMgr->set_rect(obj, rect);
}
static void gdextension_spx_ui_get_rect(GdObj obj,GdRect2* ret_val) {
	*ret_val = uiMgr->get_rect(obj);
}
static void gdextension_spx_ui_get_layout_direction(GdObj obj,GdInt* ret_val) {
	*ret_val = uiMgr->get_layout_direction(obj);
}
static void gdextension_spx_ui_set_layout_direction(GdObj obj,GdInt value) {
	 uiMgr->set_layout_direction(obj, value);
}
static void gdextension_spx_ui_get_layout_mode(GdObj obj,GdInt* ret_val) {
	*ret_val = uiMgr->get_layout_mode(obj);
}
static void gdextension_spx_ui_set_layout_mode(GdObj obj,GdInt value) {
	 uiMgr->set_layout_mode(obj, value);
}
static void gdextension_spx_ui_get_anchors_preset(GdObj obj,GdInt* ret_val) {
	*ret_val = uiMgr->get_anchors_preset(obj);
}
static void gdextension_spx_ui_set_anchors_preset(GdObj obj,GdInt value) {
	 uiMgr->set_anchors_preset(obj, value);
}
static void gdextension_spx_ui_get_scale(GdObj obj,GdVec2* ret_val) {
	*ret_val = uiMgr->get_scale(obj);
}
static void gdextension_spx_ui_set_scale(GdObj obj,GdVec2 value) {
	 uiMgr->set_scale(obj, value);
}
static void gdextension_spx_ui_get_position(GdObj obj,GdVec2* ret_val) {
	*ret_val = uiMgr->get_position(obj);
}
static void gdextension_spx_ui_set_position(GdObj obj,GdVec2 value) {
	 uiMgr->set_position(obj, value);
}
static void gdextension_spx_ui_get_size(GdObj obj,GdVec2* ret_val) {
	*ret_val = uiMgr->get_size(obj);
}
static void gdextension_spx_ui_set_size(GdObj obj,GdVec2 value) {
	 uiMgr->set_size(obj, value);
}
static void gdextension_spx_ui_get_global_position(GdObj obj,GdVec2* ret_val) {
	*ret_val = uiMgr->get_global_position(obj);
}
static void gdextension_spx_ui_set_global_position(GdObj obj,GdVec2 value) {
	 uiMgr->set_global_position(obj, value);
}
static void gdextension_spx_ui_get_rotation(GdObj obj,GdFloat* ret_val) {
	*ret_val = uiMgr->get_rotation(obj);
}
static void gdextension_spx_ui_set_rotation(GdObj obj,GdFloat value) {
	 uiMgr->set_rotation(obj, value);
}
static void gdextension_spx_ui_get_flip(GdObj obj,GdBool horizontal,GdBool* ret_val) {
	*ret_val = uiMgr->get_flip(obj, horizontal);
}
static void gdextension_spx_ui_set_flip(GdObj obj,GdBool horizontal,GdBool is_flip) {
	 uiMgr->set_flip(obj, horizontal, is_flip);
}



void gdextension_spx_setup_interface() {
	REGISTER_SPX_INTERFACE_FUNC(spx_global_register_callbacks);

	REGISTER_SPX_INTERFACE_FUNC(spx_audio_stop_all);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_create_audio);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_destroy_audio);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_set_pitch);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_get_pitch);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_set_pan);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_get_pan);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_set_volume);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_get_volume);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_play);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_pause);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_resume);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_stop);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_set_loop);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_get_loop);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_get_timer);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_set_timer);
	REGISTER_SPX_INTERFACE_FUNC(spx_audio_is_playing);
	REGISTER_SPX_INTERFACE_FUNC(spx_camera_get_camera_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_camera_set_camera_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_camera_get_camera_zoom);
	REGISTER_SPX_INTERFACE_FUNC(spx_camera_set_camera_zoom);
	REGISTER_SPX_INTERFACE_FUNC(spx_camera_get_viewport_rect);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_request_exit);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_on_runtime_panic);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_destroy_all_pens);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_create_pen);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_destroy_pen);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_pen_stamp);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_move_pen_to);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_pen_down);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_pen_up);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_set_pen_color_to);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_change_pen_by);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_set_pen_to);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_change_pen_size_by);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_set_pen_size_to);
	REGISTER_SPX_INTERFACE_FUNC(spx_ext_set_pen_stamp_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_get_mouse_pos);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_get_key);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_get_mouse_state);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_get_key_state);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_get_axis);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_is_action_pressed);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_is_action_just_pressed);
	REGISTER_SPX_INTERFACE_FUNC(spx_input_is_action_just_released);
	REGISTER_SPX_INTERFACE_FUNC(spx_physic_raycast);
	REGISTER_SPX_INTERFACE_FUNC(spx_physic_check_collision);
	REGISTER_SPX_INTERFACE_FUNC(spx_physic_check_touched_camera_boundaries);
	REGISTER_SPX_INTERFACE_FUNC(spx_physic_check_touched_camera_boundary);
	REGISTER_SPX_INTERFACE_FUNC(spx_physic_set_collision_system_type);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_window_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_get_window_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_window_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_get_window_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_window_title);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_get_window_title);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_window_fullscreen);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_is_window_fullscreen);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_debug_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_is_debug_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_get_time_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_time_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_get_persistant_data_dir);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_set_persistant_data_dir);
	REGISTER_SPX_INTERFACE_FUNC(spx_platform_is_in_persistant_data_dir);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_create_animation);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_set_load_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_get_load_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_get_bound_from_alpha);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_get_image_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_read_all_text);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_has_file);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_reload_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_free_str);
	REGISTER_SPX_INTERFACE_FUNC(spx_res_set_default_font);
	REGISTER_SPX_INTERFACE_FUNC(spx_scene_change_scene_to_file);
	REGISTER_SPX_INTERFACE_FUNC(spx_scene_destroy_all_sprites);
	REGISTER_SPX_INTERFACE_FUNC(spx_scene_reload_current_scene);
	REGISTER_SPX_INTERFACE_FUNC(spx_scene_unload_current_scene);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_dont_destroy_on_load);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_process);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_physic_process);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_type_name);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_child_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_child_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_child_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_child_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_child_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_child_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_check_collision);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_check_collision_with_point);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_create_backdrop);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_create_sprite);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_clone_sprite);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_destroy_sprite);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_sprite_alive);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_render_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_render_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_material_shader);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_material_shader);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_material_params);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_material_params);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_material_params_vec);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_material_params_vec4);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_material_params_vec4);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_material_params_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_material_params_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_texture_altas);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_texture_altas_direct);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_texture_direct);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_visible);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_visible);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_z_index);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_z_index);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_play_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_play_backwards_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_pause_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_stop_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_playing_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_anim);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_frame);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_anim_frame);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_speed_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_anim_speed_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_anim_playing_speed);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_centered);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_anim_centered);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_offset);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_anim_offset);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_flip_h);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_anim_flipped_h);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_anim_flip_v);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_anim_flipped_v);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_velocity);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_velocity);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_floor);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_floor_only);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_wall);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_wall_only);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_ceiling);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_on_ceiling_only);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_last_motion);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_position_delta);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_floor_normal);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_wall_normal);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_real_velocity);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_move_and_slide);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_gravity);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_gravity);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_mass);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_mass);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_add_force);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_add_impulse);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collision_layer);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_collision_layer);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collision_mask);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_collision_mask);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_layer);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_trigger_layer);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_mask);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_get_trigger_mask);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collider_rect);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collider_circle);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collider_capsule);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_collision_enabled);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_collision_enabled);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_rect);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_circle);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_capsule);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_set_trigger_enabled);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_is_trigger_enabled);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_check_collision_by_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_check_collision_by_alpha);
	REGISTER_SPX_INTERFACE_FUNC(spx_sprite_check_collision_with_sprite_by_alpha);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_bind_node);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_node);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_button);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_label);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_image);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_toggle);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_slider);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_create_input);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_destroy_node);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_type);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_text);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_text);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_texture);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_color);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_font_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_font_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_visible);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_visible);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_interactable);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_interactable);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_rect);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_rect);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_layout_direction);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_layout_direction);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_layout_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_layout_mode);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_anchors_preset);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_anchors_preset);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_scale);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_size);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_global_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_global_position);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_rotation);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_get_flip);
	REGISTER_SPX_INTERFACE_FUNC(spx_ui_set_flip);
	
}
