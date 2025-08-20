#ifndef GODOT_JS_SPX_H
#define GODOT_JS_SPX_H

#include <stddef.h>
#include <stdint.h>

typedef float GdFloat ;
typedef int64_t GdObj ;
typedef const void* GdString ;
typedef int64_t GdInt ;
typedef uint8_t GdBool ;

// GDExtention
extern void godot_js_spx_on_runtime_panic(GdString msg);
extern void godot_js_spx_on_runtime_exit(int code);
// Gdspx 
extern void godot_js_spx_on_engine_start();
extern void godot_js_spx_on_engine_update(GdFloat delta);
extern void godot_js_spx_on_engine_fixed_update(GdFloat delta);
extern void godot_js_spx_on_engine_destroy();
extern void godot_js_spx_on_engine_pause(GdBool is_paused);

extern void godot_js_spx_on_scene_sprite_instantiated(GdObj* obj,GdString type_name);

extern void godot_js_spx_on_sprite_ready(GdObj* obj);
extern void godot_js_spx_on_sprite_updated(GdFloat delta);
extern void godot_js_spx_on_sprite_fixed_updated(GdFloat delta);
extern void godot_js_spx_on_sprite_destroyed(GdObj* obj);

extern void godot_js_spx_on_sprite_frames_set_changed(GdObj* obj);
extern void godot_js_spx_on_sprite_animation_changed(GdObj* obj);
extern void godot_js_spx_on_sprite_frame_changed(GdObj* obj);
extern void godot_js_spx_on_sprite_animation_looped(GdObj* obj);
extern void godot_js_spx_on_sprite_animation_finished(GdObj* obj);

extern void godot_js_spx_on_sprite_vfx_finished(GdObj* obj);

extern void godot_js_spx_on_sprite_screen_exited(GdObj* obj);
extern void godot_js_spx_on_sprite_screen_entered(GdObj* obj);

extern void godot_js_spx_on_mouse_pressed(GdInt* keyid);
extern void godot_js_spx_on_mouse_released(GdInt* keyid);
extern void godot_js_spx_on_key_pressed(GdInt* keyid);
extern void godot_js_spx_on_key_released(GdInt* keyid);
extern void godot_js_spx_on_action_pressed(GdString action_name);
extern void godot_js_spx_on_action_just_pressed(GdString action_name);
extern void godot_js_spx_on_action_just_released(GdString action_name);
extern void godot_js_spx_on_axis_changed(GdString action_name, GdFloat value);

extern void godot_js_spx_on_collision_enter(GdInt* self_id, GdInt* other_id);
extern void godot_js_spx_on_collision_stay(GdInt* self_id, GdInt* other_id);
extern void godot_js_spx_on_collision_exit(GdInt* self_id, GdInt* other_id);
extern void godot_js_spx_on_trigger_enter(GdInt* self_id, GdInt* other_id);
extern void godot_js_spx_on_trigger_stay(GdInt* self_id, GdInt* other_id);
extern void godot_js_spx_on_trigger_exit(GdInt* self_id, GdInt* other_id);

extern void godot_js_spx_on_ui_ready(GdObj* obj);
extern void godot_js_spx_on_ui_updated(GdObj* obj);
extern void godot_js_spx_on_ui_destroyed(GdObj* obj);

extern void godot_js_spx_on_ui_pressed(GdObj* obj);
extern void godot_js_spx_on_ui_released(GdObj* obj);
extern void godot_js_spx_on_ui_hovered(GdObj* obj);
extern void godot_js_spx_on_ui_clicked(GdObj* obj);
extern void godot_js_spx_on_ui_toggle(GdObj* obj, GdBool is_on);
extern void godot_js_spx_on_ui_text_changed(GdObj* obj, GdString text);




#endif // GODOT_JS_SPX_H