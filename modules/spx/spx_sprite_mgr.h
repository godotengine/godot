/**************************************************************************/
/*  spx_sprite_mgr.h                                                      */
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

#ifndef SPX_SPRITE_MGR_H
#define SPX_SPRITE_MGR_H

#include "gdextension_spx_ext.h"
#include "scene/2d/animated_sprite_2d.h"
#include "spx_base_mgr.h"
#include "core/templates/hash_map.h"
#include <functional>
#include <unordered_set>

class SpxSprite;

typedef std::function<bool(GdColor, GdColor)> ColorCheckFunc;
#include <functional>
#include <unordered_set>

class TriggerPair {
public:
	GdObj id1;
	GdObj id2;

public:
	TriggerPair() :
			id1(0), id2(0) {}

	TriggerPair(GdObj id1, GdObj id2) {
		this->id1 = id1;
		this->id2 = id2;
		if (id1 > id2) {
			this->id1 = id2;
			this->id2 = id1;
		}
	}
	bool operator<(const TriggerPair &p_other) const {
		return id1 < p_other.id1 || (id1 == p_other.id1 && id2 < p_other.id2);
	}

	bool operator==(const TriggerPair &p_other) const {
		return id1 == p_other.id1 && id2 == p_other.id2;
	}

	bool operator!=(const TriggerPair &p_other) const {
		return !(*this == p_other);
	}

	size_t std_hash() const {
		return static_cast<size_t>(id1) ^ static_cast<size_t>(id2);
	}
};

namespace std {
template <>
struct hash<TriggerPair> {
	size_t operator()(const TriggerPair &pair) const {
		return pair.std_hash();
	}
};
}


class SpxSpriteMgr : SpxBaseMgr {
	SPXCLASS(SpxSpriteMgr, SpxBaseMgr)
public:
	virtual ~SpxSpriteMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

private:
	RBMap<GdObj, SpxSprite *> id_objects;
	std::unordered_set<TriggerPair> bounding_collision_pairs;
	std::unordered_set<TriggerPair> pixel_collision_pairs;

	Node* dont_destroy_root;
	Node* sprite_root;

	Ref<Image> _get_current_frame_image(AnimatedSprite2D *sprite);
	Rect2 _get_sprite_aabb(AnimatedSprite2D *anim2d);
	Vector2 _to_image_coord(const Transform2D &trans, Vector2 image_size, Vector2 pos);
	GdBool _check_collision(GdObj obj, ColorCheckFunc check_func);
	void _check_pixel_collision_events();

public:
	static StringName default_texture_anim;
	void on_awake() override;
	void on_start() override;
	void on_destroy() override;
	void on_update(float delta) override;

	SpxSprite *get_sprite(GdObj obj);
	void on_sprite_destroy(SpxSprite *sprite);
	void on_trigger_enter(GdInt self_id, GdInt other_id);
	void on_trigger_exit(GdInt self_id, GdInt other_id);
	GdObj _create_sprite(GdString path, GdBool is_backdrop);
	void destroy_all_sprites();
public:
	void set_dont_destroy_on_load(GdObj obj);
	// process
	void set_process(GdObj obj, GdBool is_on);
	void set_physic_process(GdObj obj, GdBool is_on);

	void set_type_name(GdObj obj,GdString type_name);

	// children
	void set_child_position(GdObj obj, GdString path, GdVec2 pos);
	GdVec2 get_child_position(GdObj obj, GdString path);
	void set_child_rotation(GdObj obj, GdString path, GdFloat rot);
	GdFloat get_child_rotation(GdObj obj, GdString path);
	void set_child_scale(GdObj obj, GdString path, GdVec2 scale);
	GdVec2 get_child_scale(GdObj obj, GdString path);

	GdBool check_collision(GdObj obj,GdObj target, GdBool is_src_trigger,GdBool is_dst_trigger);
	GdBool check_collision_with_point(GdObj obj,GdVec2 point, GdBool is_trigger);
	//
	GdObj create_backdrop(GdString path);
	GdObj create_sprite(GdString path);
	GdObj clone_sprite(GdObj obj);
	GdBool destroy_sprite(GdObj obj);
	GdBool is_sprite_alive(GdObj obj);
	void set_position(GdObj obj, GdVec2 pos);
	GdVec2 get_position(GdObj obj);
	void set_rotation(GdObj obj, GdFloat rot);
	GdFloat get_rotation(GdObj obj);
	void set_scale(GdObj obj, GdVec2 scale);
	GdVec2 get_scale(GdObj obj);
	void set_render_scale(GdObj obj, GdVec2 scale);
	GdVec2 get_render_scale(GdObj obj);
	void set_color(GdObj obj, GdColor color);
	GdColor get_color(GdObj obj);

	void set_material_shader(GdObj obj, GdString path);
	GdString get_material_shader(GdObj obj);
	void set_material_params(GdObj obj, GdString effect, GdFloat amount);
	GdFloat get_material_params(GdObj obj, GdString effect);

	void set_material_params_vec(GdObj obj, GdString effect, GdFloat x, GdFloat y, GdFloat z, GdFloat w);

	void set_material_params_vec4(GdObj obj, GdString effect, GdVec4 vec4);
	GdVec4 get_material_params_vec4(GdObj obj, GdString effect);

	void set_material_params_color(GdObj obj, GdString effect, GdColor color);
	GdColor get_material_params_color(GdObj obj, GdString effect);

	void set_texture_altas(GdObj obj, GdString path, GdRect2 rect2);
	void set_texture(GdObj obj, GdString path);
	void set_texture_altas_direct(GdObj obj, GdString path, GdRect2 rect2);
	void set_texture_direct(GdObj obj, GdString path);

	GdString get_texture(GdObj obj);
	void set_visible(GdObj obj, GdBool visible);
	GdBool get_visible(GdObj obj);
	GdInt get_z_index(GdObj obj);
	void set_z_index(GdObj obj, GdInt z);

	// animation
	void play_anim(GdObj obj, GdString p_name , GdFloat p_speed, GdBool isLoop, GdBool p_revert );
	void play_backwards_anim(GdObj obj,  GdString p_name );
	void pause_anim(GdObj obj);
	void stop_anim(GdObj obj);
	GdBool is_playing_anim(GdObj obj);
	void set_anim(GdObj obj, GdString p_name);
	GdString get_anim(GdObj obj);
	void set_anim_frame(GdObj obj, GdInt p_frame);
	GdInt get_anim_frame(GdObj obj);
	void set_anim_speed_scale(GdObj obj, GdFloat p_speed_scale);
	GdFloat get_anim_speed_scale(GdObj obj);
	GdFloat get_anim_playing_speed(GdObj obj);
	void set_anim_centered(GdObj obj, GdBool p_center);
	GdBool is_anim_centered(GdObj obj);
	void set_anim_offset(GdObj obj, GdVec2 p_offset);
	GdVec2 get_anim_offset(GdObj obj);
	void set_anim_flip_h(GdObj obj, GdBool p_flip);
	GdBool is_anim_flipped_h(GdObj obj);
	void set_anim_flip_v(GdObj obj, GdBool p_flip);
	GdBool is_anim_flipped_v(GdObj obj);
	GdString get_current_anim_name(GdObj obj);
	
	// physics
	void set_velocity(GdObj obj, GdVec2 velocity);
	GdVec2 get_velocity(GdObj obj);
	GdBool is_on_floor(GdObj obj);
	GdBool is_on_floor_only(GdObj obj);
	GdBool is_on_wall(GdObj obj);
	GdBool is_on_wall_only(GdObj obj);
	GdBool is_on_ceiling(GdObj obj);
	GdBool is_on_ceiling_only(GdObj obj);
	GdVec2 get_last_motion(GdObj obj);
	GdVec2 get_position_delta(GdObj obj);
	GdVec2 get_floor_normal(GdObj obj);
	GdVec2 get_wall_normal(GdObj obj);
	GdVec2 get_real_velocity(GdObj obj);
	void move_and_slide(GdObj obj);

	void set_gravity(GdObj obj, GdFloat gravity);
	GdFloat get_gravity(GdObj obj);
	void set_mass(GdObj obj, GdFloat mass);
	GdFloat get_mass(GdObj obj);
	void add_force(GdObj obj, GdVec2 force);
	void add_impulse(GdObj obj, GdVec2 impulse);

	void set_physics_mode(GdObj obj, GdInt mode);
	GdInt get_physics_mode(GdObj obj);
	void set_use_gravity(GdObj obj, GdBool enabled);
	GdBool is_use_gravity(GdObj obj);
	void set_gravity_scale(GdObj obj, GdFloat scale);
	GdFloat get_gravity_scale(GdObj obj);
	void set_drag(GdObj obj, GdFloat drag);
	GdFloat get_drag(GdObj obj);
	void set_friction(GdObj obj, GdFloat friction);
	GdFloat get_friction(GdObj obj);

	void set_collision_layer(GdObj obj, GdInt layer);
	GdInt get_collision_layer(GdObj obj);
	void set_collision_mask(GdObj obj, GdInt mask);
	GdInt get_collision_mask(GdObj obj);

	void set_trigger_layer(GdObj obj, GdInt layer);
	GdInt get_trigger_layer(GdObj obj);
	void set_trigger_mask(GdObj obj, GdInt mask);
	GdInt get_trigger_mask(GdObj obj);

	void set_collider_rect(GdObj obj, GdVec2 center, GdVec2 size);
	void set_collider_circle(GdObj obj, GdVec2 center, GdFloat radius);
	void set_collider_capsule(GdObj obj, GdVec2 center, GdVec2 size);
	void set_collision_enabled(GdObj obj, GdBool enabled);
	GdBool is_collision_enabled(GdObj obj);

	void set_trigger_rect(GdObj obj, GdVec2 center, GdVec2 size);
	void set_trigger_circle(GdObj obj, GdVec2 center, GdFloat radius);
	void set_trigger_capsule(GdObj obj, GdVec2 center, GdVec2 size);
	void set_trigger_enabled(GdObj obj, GdBool trigger);
	GdBool is_trigger_enabled(GdObj obj);

	// misc
	GdBool check_collision_by_color(GdObj obj, GdColor color,GdFloat color_threshold, GdFloat alpha_threshold);
	GdBool check_collision_by_alpha(GdObj obj, GdFloat alpha_threshold);
	GdBool check_collision_with_sprite_by_alpha(GdObj obj, GdObj obj_b, GdFloat alpha_threshold);

};

#endif // SPX_SPRITE_MGR_H
