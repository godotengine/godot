/**************************************************************************/
/*  spx_sprite.cpp                                                        */
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

#include "spx_sprite.h"

#include "core/config/project_settings.h"
#include "spx_base_mgr.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/2d/physics/area_2d.h"
#include "scene/2d/physics/collision_shape_2d.h"
#include "scene/2d/visible_on_screen_notifier_2d.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/2d/capsule_shape_2d.h"
#include "scene/resources/2d/circle_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"
#include "spx.h"
#include "spx_engine.h"
#include "spx_physic_mgr.h"
#include "spx_res_mgr.h"
#include "spx_sprite_mgr.h"
#include "spx_camera_mgr.h"
#include "svg_mgr.h"
#define SPX_CALLBACK SpxEngine::get_singleton()->get_callbacks()
#define spriteMgr SpxEngine::get_singleton()->get_sprite()

Node *SpxSprite::get_component(Node *node, StringName name, GdBool recursive) {
	for (int i = 0; i < node->get_child_count(); ++i) {
		Node *child = node->get_child(i);

		if (child->get_name() == name) {
			return child;
		}

		if (recursive) {
			Node *found_node = get_component(child, name, true);
			if (found_node != nullptr) {
				return found_node;
			}
		}
	}
	return nullptr;
}
void SpxSprite::set_use_default_frames(bool is_on) {
	use_default_frames = is_on;
}
bool SpxSprite::get_use_default_frames() {
	return use_default_frames;
}

void SpxSprite::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gid", "gid"), &SpxSprite::set_gid);
	ClassDB::bind_method(D_METHOD("get_gid"), &SpxSprite::get_gid);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "gid"), "set_gid", "get_gid");

	ClassDB::bind_method(D_METHOD("set_use_default_frames", "use_default_frames"), &SpxSprite::set_use_default_frames);
	ClassDB::bind_method(D_METHOD("get_use_default_frames"), &SpxSprite::get_use_default_frames);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "use_default_frames"), "set_use_default_frames", "get_use_default_frames");

	ClassDB::bind_method(D_METHOD("set_spx_type_name", "spx_type_name"), &SpxSprite::set_spx_type_name);
	ClassDB::bind_method(D_METHOD("get_spx_type_name"), &SpxSprite::get_spx_type_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "spx_type_name"), "set_spx_type_name", "get_spx_type_name");

	ClassDB::bind_method(D_METHOD("on_destroy_call"), &SpxSprite::on_destroy_call);

	ClassDB::bind_method(D_METHOD("on_area_entered", "area"), &SpxSprite::on_area_entered);
	ClassDB::bind_method(D_METHOD("on_area_exited", "area"), &SpxSprite::on_area_exited);

	ClassDB::bind_method(D_METHOD("on_sprite_frames_set_changed"), &SpxSprite::on_sprite_frames_set_changed);
	ClassDB::bind_method(D_METHOD("on_sprite_animation_changed"), &SpxSprite::on_sprite_animation_changed);
	ClassDB::bind_method(D_METHOD("on_sprite_frame_changed"), &SpxSprite::on_sprite_frame_changed);
	ClassDB::bind_method(D_METHOD("on_sprite_animation_looped"), &SpxSprite::on_sprite_animation_looped);
	ClassDB::bind_method(D_METHOD("on_sprite_animation_finished"), &SpxSprite::on_sprite_animation_finished);
	ClassDB::bind_method(D_METHOD("on_sprite_vfx_finished"), &SpxSprite::on_sprite_vfx_finished);
	ClassDB::bind_method(D_METHOD("on_sprite_screen_exited"), &SpxSprite::on_sprite_screen_exited);
	ClassDB::bind_method(D_METHOD("on_sprite_screen_entered"), &SpxSprite::on_sprite_screen_entered);

	// Physics mode method binding (using int type to avoid enum binding issues)
	ClassDB::bind_method(D_METHOD("set_physics_mode", "mode"), &SpxSprite::set_physics_mode);
	ClassDB::bind_method(D_METHOD("get_physics_mode"), &SpxSprite::get_physics_mode);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "physics_mode", PROPERTY_HINT_ENUM, "NoPhysics,Kinematic,Dynamic,Static"), "set_physics_mode", "get_physics_mode");
	
	// Gravity control
	ClassDB::bind_method(D_METHOD("set_use_gravity", "enabled"), &SpxSprite::set_use_gravity);
	ClassDB::bind_method(D_METHOD("is_use_gravity"), &SpxSprite::is_use_gravity);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gravity"), "set_use_gravity", "is_use_gravity");
	
	ClassDB::bind_method(D_METHOD("set_gravity_scale", "scale"), &SpxSprite::set_gravity_scale);
	ClassDB::bind_method(D_METHOD("get_gravity_scale"), &SpxSprite::get_gravity_scale);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "gravity_scale"), "set_gravity_scale", "get_gravity_scale");
	
	ClassDB::bind_method(D_METHOD("set_drag", "drag"), &SpxSprite::set_drag);
	ClassDB::bind_method(D_METHOD("get_drag"), &SpxSprite::get_drag);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "drag"), "set_drag", "get_drag");
	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &SpxSprite::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &SpxSprite::get_friction);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "friction"), "set_friction", "get_friction");
}

void SpxSprite::on_destroy_call() {
	if (!Spx::initialed)
		return;
	spriteMgr->on_sprite_destroy(this);
}

SpxSprite::SpxSprite() {
	// Get project gravity settings
	_gravity = ProjectSettings::get_singleton()->get_setting("physics/2d/default_gravity", 980.0);
	
	// Default to NO_PHYSICS mode for backward compatibility
	physics_mode = NO_PHYSICS;
}

SpxSprite::~SpxSprite() {
}

void SpxSprite::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PHYSICS_PROCESS: {
			_physics_process(get_physics_process_delta_time());
			break;
		}
		case NOTIFICATION_PREDELETE: {
			on_destroy_call();
			break;
		}
		case NOTIFICATION_DRAW: {
			_draw();
			break;
		}
		default:
			break;
	}
}

void SpxSprite::_draw() {
	if (!Spx::debug_mode) {
		return;
	}
	if (trigger2d != nullptr) {
		trigger2d->set_spx_debug_color(Color(1, 0, 0, 0.2));
	}
	if (collider2d != nullptr) {
		collider2d->set_spx_debug_color(Color(0, 0, 1, 0.2));
	}
}

void SpxSprite::on_start() {
	collider2d = (get_component<CollisionShape2D>());
	anim2d = (get_component<AnimatedSprite2D>());
	default_sprite_frames = anim2d->get_sprite_frames();
	if(default_sprite_frames.is_null() || resMgr->is_dynamic_anim_mode()) {
		default_sprite_frames.instantiate();
		anim2d->set_sprite_frames(default_sprite_frames);
	}

	visible_notifier = (get_component<VisibleOnScreenNotifier2D>());
	if (visible_notifier == nullptr) {
		visible_notifier = memnew(VisibleOnScreenNotifier2D);
		add_child(visible_notifier);
	}
	//anim2d->get_sprite_frames()->add_animation(SpxSpriteMgr::default_texture_anim);
	area2d = (get_component<Area2D>());
	if (area2d != nullptr) {
		trigger2d = get_component<CollisionShape2D>(area2d);
	}

	if (area2d != nullptr) {
		area2d->connect("area_entered", Callable(this, "on_area_entered"));
		area2d->connect("area_exited", Callable(this, "on_area_exited"));
	}
	if (anim2d != nullptr) {
		anim2d->connect("sprite_frames_changed", Callable(this, "on_sprite_frames_set_changed"));
		anim2d->connect("animation_changed", Callable(this, "on_sprite_animation_changed"));
		anim2d->connect("frame_changed", Callable(this, "on_sprite_frame_changed"));
		anim2d->connect("animation_looped", Callable(this, "on_sprite_animation_looped"));
		anim2d->connect("animation_finished", Callable(this, "on_sprite_animation_finished"));
	}
	if (visible_notifier != nullptr) {
		visible_notifier->connect("screen_exited", Callable(this, "on_sprite_screen_exited"));
		visible_notifier->connect("screen_entered", Callable(this, "on_sprite_screen_entered"));
	}
	_update_physics_mode();
}

void SpxSprite::set_gid(GdObj id) {
	this->gid = id;
}

GdObj SpxSprite::get_gid() {
	return gid;
}
void SpxSprite::set_type_name(GdString type_name) {
	auto name = SpxStr(type_name);
	spx_type_name = name;
	this->set_name(name);
}

void SpxSprite::set_spx_type_name(String type_name) {
	spx_type_name = type_name;
}

String SpxSprite::get_spx_type_name() {
	return spx_type_name;
}

void SpxSprite::on_area_entered(Node *node) {
	if (!Spx::initialed) {
		return;
	}
	// backdrop would not collision with other
	if(is_backdrop) {
		return ;
	}
	Node *parent_node = node->get_parent();
	const SpxSprite *other = Object::cast_to<SpxSprite>(parent_node);
	if (other != nullptr) {
		spriteMgr->on_trigger_enter(this->gid, other->gid);
	}
}

void SpxSprite::on_area_exited(Node *node) {
	if (!Spx::initialed) {
		return;
	}
	// backdrop would not collision with other
	if(is_backdrop) {
		return ;
	}
	Node *parent_node = node->get_parent();
	const SpxSprite *other = Object::cast_to<SpxSprite>(parent_node);
	if (other != nullptr) {
		spriteMgr->on_trigger_exit(this->gid, other->gid);
	}
}

void SpxSprite::on_sprite_frames_set_changed() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_frames_set_changed(this->gid);
}

void SpxSprite::on_sprite_animation_changed() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_animation_changed(this->gid);
}

void SpxSprite::on_sprite_frame_changed() {
	if (!Spx::initialed) {
		return;
	}
	
	// Handle dynamic frame offset
	_on_frame_changed();
	
	SPX_CALLBACK->func_on_sprite_frame_changed(this->gid);

	// update effect shader's atlas's uv rect
	if (anim2d != nullptr) {
		auto uv_rect = anim2d->get_uv_rect();
		if (default_material.is_null()) {
			return;
		}
		default_material->set_shader_parameter("atlas_uv_rect2", uv_rect);
	}
}

void SpxSprite::on_sprite_animation_looped() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_animation_looped(this->gid);
}

void SpxSprite::on_sprite_animation_finished() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_animation_finished(this->gid);
}

void SpxSprite::on_sprite_vfx_finished() {
	if (!Spx::initialed) {
		return;
	}
}

void SpxSprite::on_sprite_screen_exited() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_screen_exited(this->gid);
}

void SpxSprite::on_sprite_screen_entered() {
	if (!Spx::initialed) {
		return;
	}
	SPX_CALLBACK->func_on_sprite_screen_entered(this->gid);
}

void SpxSprite::set_color(GdColor color) {
	default_material->set_shader_parameter("color", color);
}

GdColor SpxSprite::get_color() {
	return anim2d->get_self_modulate();
}

void SpxSprite::set_material_shader(GdString path) {
	Ref<Shader> shader = ResourceLoader::load(SpxStr(path));
	if (shader.is_null()) {
		print_line("load spx_sprite_shader failed !",SpxStr(path));
		return;
	}

	default_material = anim2d->get_material();
	if (default_material.is_null())
	{
		default_material.instantiate();
		anim2d->set_material(default_material);
	}

	default_material.ptr()->set_shader(shader);
	// uv_effect dependon texture repeat
	anim2d->set_texture_repeat(TEXTURE_REPEAT_ENABLED);
}

GdString SpxSprite::get_material_shader() {
	default_material = anim2d->get_material();
	if (default_material.is_null())
	{
		return nullptr;
	}
	auto path = default_material.ptr()->get_shader()->get_path();
	return SpxReturnStr(path);
}

void SpxSprite::set_material_params(GdString effect, GdFloat amount) {
	if (default_material.is_null())
	{
		print_line("set_material_params failed, the material and shader have not been set, please initialize the shader first!");
		return;
	}

	default_material->set_shader_parameter(SpxStr(effect), amount);
}

GdFloat SpxSprite::get_material_params(GdString effect) {
	if (default_material.is_null())
	{
		print_line("get_material_params failed, the material and shader have not been set, please initialize the shader first!");
		return 0;
	}
	return default_material->get_shader_parameter(SpxStr(effect));
}

void SpxSprite::set_material_params_vec4(GdString effect, GdVec4 vec4) {
	if (default_material.is_null())
	{
		print_line("set_material_params_vec4 failed, the material and shader have not been set, please initialize the shader first!");
		return;
	}
	default_material->set_shader_parameter(SpxStr(effect), vec4);
}

GdVec4 SpxSprite::get_material_params_vec4(GdString effect) {
	if (default_material.is_null())
	{
		print_line("get_material_params_vec4 failed, the material and shader have not been set, please initialize the shader first!");
		return GdVec4();
	}
	return default_material->get_shader_parameter(SpxStr(effect));
}

void SpxSprite::set_material_params_color(GdString effect, GdColor color) {
	if (default_material.is_null())
	{
		print_line("set_material_params_color failed, the material and shader have not been set, please initialize the shader first!");
		return;
	}
	default_material->set_shader_parameter(SpxStr(effect), color);
}

GdColor SpxSprite::get_material_params_color(GdString effect) {
	if (default_material.is_null())
	{
		print_line("get_material_params_color failed, the material and shader have not been set, please initialize the shader first!");
		return GdColor();
	}
	return default_material->get_shader_parameter(SpxStr(effect));
}

void SpxSprite::set_texture_altas_direct(GdString path, GdRect2 rect2, GdBool direct) {
	auto path_str = SpxStr(path);
	current_anim_name = "";
	is_svg_mode = false;// svg don't support atlas
	Ref<Texture2D> texture = resMgr->load_texture(path_str, direct);

	Ref<AtlasTexture> atlas_texture_frame = memnew(AtlasTexture);
	atlas_texture_frame->set_atlas(texture);
	atlas_texture_frame->set_region(rect2);

	_play_single_image_animation(atlas_texture_frame);
}


void SpxSprite::set_texture_direct(GdString path, GdBool direct) {
	auto path_str = SpxStr(path);
	current_anim_name = "";

	Ref<Texture2D> texture = nullptr;
	is_svg_mode = svgMgr->is_svg_file(path_str);
	if (is_svg_mode){
		int target_scale = _get_actual_match_render_scale();
		current_svg_scale = target_scale;
		current_svg_path = path_str;
		texture = svgMgr->get_svg_image(path_str, target_scale);
	}else{
		texture = resMgr->load_texture(path_str, direct);
	}
	_play_single_image_animation(texture);
}

void SpxSprite::_play_single_image_animation(Ref<Texture2D> texture){
	if (texture.is_valid()) {
		is_single_image_mode = true;
		anim2d->set_sprite_frames(default_sprite_frames);
		auto frames = anim2d->get_sprite_frames();
		if (frames->get_frame_count(SpxSpriteMgr::default_texture_anim) == 0) {
			frames->add_frame(SpxSpriteMgr::default_texture_anim, texture);
		} else {
			frames->set_frame(SpxSpriteMgr::default_texture_anim, 0, texture);
		}
		anim2d->set_animation(SpxSpriteMgr::default_texture_anim);
	} else {
		print_error("can not set single image animation, texture is null");
	}
}

void SpxSprite::set_texture_altas(GdString path, GdRect2 rect2) {
	return set_texture_altas_direct(path, rect2, false);
}
void SpxSprite::set_texture(GdString path) {
	return set_texture_direct(path, false);
}
GdString SpxSprite::get_texture() {
	auto tex = anim2d->get_sprite_frames()->get_frame_texture(SpxSpriteMgr::default_texture_anim, 0);
	if (tex == nullptr)
		return nullptr;
	return SpxReturnStr(tex->get_name());
}

GdString SpxSprite::get_current_anim_name() {
	return SpxReturnStr(current_anim_name);
}

void SpxSprite::play_anim(GdString p_name, GdFloat p_speed, GdBool isLoop, GdBool p_from_end) {
	String anim_name = SpxStr(p_name);
	// Enhanced: Check if we need to use a scaled version of the animation
	String final_anim_key;
	is_svg_mode = false;
	is_single_image_mode = false;
	current_anim_name = anim_name;

	if (resMgr->is_dynamic_anim_mode()) {
		String sprite_type = get_spx_type_name();
		String base_anim_key = resMgr->get_anim_key_name(sprite_type, anim_name);
		current_svg_anim_key = base_anim_key;
		Ref<SpriteFrames> frames;
		is_svg_mode = svgMgr->is_svg_animation(base_anim_key);
		// Check if this is an SVG animation that supports scaling
		if (is_svg_mode) {
			// Calculate required scale based on current render scale
			int target_scale = _get_actual_match_render_scale();
			frames = svgMgr->get_svg_animation(base_anim_key, target_scale);
			final_anim_key = base_anim_key;
		} else {
			frames = resMgr->get_anim_frames(final_anim_key);
			// Use base animation for non-SVG animations
			final_anim_key = base_anim_key;
		}
		anim2d->set_sprite_frames(frames);
		frames->set_animation_loop(final_anim_key, isLoop);
	} else {
		final_anim_key = anim_name;
	}
	
	anim2d->play(final_anim_key, p_speed, p_from_end);
	
}

void SpxSprite::play_backwards_anim(GdString p_name) {
	auto anim_name = SpxStr(p_name);
	if (resMgr->is_dynamic_anim_mode()) {
		anim_name = resMgr->get_anim_key_name(get_spx_type_name(), anim_name);
		anim2d->set_sprite_frames(resMgr->get_anim_frames(anim_name));
	}
	anim2d->play_backwards(anim_name);
}

void SpxSprite::pause_anim() {
	anim2d->pause();
}

void SpxSprite::stop_anim() {
	anim2d->stop();
}

GdBool SpxSprite::is_playing_anim() const {
	return anim2d->is_playing();
}

void SpxSprite::set_anim(GdString p_name) {
	auto anim_name = SpxStr(p_name);
	anim2d->set_animation(StringName(anim_name));
}

GdString SpxSprite::get_anim() const {
	auto name = anim2d->get_animation();
	return SpxReturnStr(String(name));
}

void SpxSprite::set_anim_frame(GdInt p_frame) {
	anim2d->set_frame(p_frame);
}

GdInt SpxSprite::get_anim_frame() const {
	return anim2d->get_frame();
}

void SpxSprite::set_anim_speed_scale(GdFloat p_speed_scale) {
	anim2d->set_speed_scale(p_speed_scale);
}

GdFloat SpxSprite::get_anim_speed_scale() const {
	return anim2d->get_speed_scale();
}

GdFloat SpxSprite::get_anim_playing_speed() const {
	return anim2d->get_playing_speed();
}

void SpxSprite::set_anim_centered(GdBool p_center) {
	anim2d->set_centered(p_center);
}

GdBool SpxSprite::is_anim_centered() const {
	return anim2d->is_centered();
}

void SpxSprite::set_anim_offset(GdVec2 p_offset) {
	base_offset = p_offset;  // Save base offset
	anim2d->set_offset(p_offset);
}

GdVec2 SpxSprite::get_anim_offset() const {
	return anim2d->get_offset();
}

void SpxSprite::set_anim_flip_h(GdBool p_flip) {
	anim2d->set_flip_h(p_flip);
}

GdBool SpxSprite::is_anim_flipped_h() const {
	return anim2d->is_flipped_h();
}

void SpxSprite::set_anim_flip_v(GdBool p_flip) {
	anim2d->set_flip_v(p_flip);
}

GdBool SpxSprite::is_anim_flipped_v() const {
	return anim2d->is_flipped_v();
}

void SpxSprite::set_gravity(GdFloat gravity) {
	_gravity = gravity;
}

GdFloat SpxSprite::get_gravity() {
	return _gravity;
}

void SpxSprite::set_mass(GdFloat mass) {
	mass_value = mass;
}

GdFloat SpxSprite::get_mass() {
	return mass_value;
}

void SpxSprite::add_force(GdVec2 force) {
	if (physics_mode == NO_PHYSICS || physics_mode == STATIC) {
		return;  // These modes are not affected by forces
	}
	if (physics_mode == KINEMATIC) {
		return;  // Kinematic mode is not affected by forces
	}
	
	external_forces += Vector2(force.x, force.y);
}

void SpxSprite::add_impulse(GdVec2 impulse) {
	if (physics_mode == NO_PHYSICS || physics_mode == STATIC) {
		return;  // These modes are not affected by forces
	}
	if (physics_mode == KINEMATIC) {
		return;  // Kinematic mode is not affected by forces
	}
	
	applied_forces += Vector2(impulse.x, impulse.y);
}

void SpxSprite::set_trigger_layer(GdInt layer) {
	area2d->set_collision_layer((uint32_t)layer);
}

GdInt SpxSprite::get_trigger_layer() {
	return area2d->get_collision_layer();
}

void SpxSprite::set_trigger_mask(GdInt mask) {
	area2d->set_collision_mask((uint32_t)mask);
}

GdInt SpxSprite::get_trigger_mask() {
	return area2d->get_collision_mask();
}

void SpxSprite::set_collider_rect(GdVec2 center, GdVec2 size) {
	Ref<RectangleShape2D> rect = memnew(RectangleShape2D);
	rect->set_size(size);
	collider2d->set_shape(rect);
	collider2d->set_position(center);
}

void SpxSprite::set_collider_circle(GdVec2 center, GdFloat radius) {
	Ref<CircleShape2D> circle = memnew(CircleShape2D);
	circle->set_radius(radius);
	collider2d->set_shape(circle);
	collider2d->set_position(center);
}

void SpxSprite::set_collider_capsule(GdVec2 center, GdVec2 size) {
	Ref<CapsuleShape2D> capsule = memnew(CapsuleShape2D);
	capsule->set_radius(size.x / 2);
	capsule->set_height(size.y);
	collider2d->set_shape(capsule);
	collider2d->set_position(center);
}

void SpxSprite::set_collision_enabled(GdBool enabled) {
	collider2d->set_visible(enabled);
}

GdBool SpxSprite::is_collision_enabled() {
	return collider2d->is_visible();
}

void SpxSprite::set_trigger_capsule(GdVec2 center, GdVec2 size) {
	Ref<CapsuleShape2D> capsule = memnew(CapsuleShape2D);
	capsule->set_radius(size.x / 2);
	capsule->set_height(size.y);
	trigger2d->set_shape(capsule);
	trigger2d->set_position(center);
}

void SpxSprite::set_trigger_rect(GdVec2 center, GdVec2 size) {
	Ref<RectangleShape2D> rect = memnew(RectangleShape2D);
	rect->set_size(size);
	trigger2d->set_shape(rect);
	trigger2d->set_position(center);
}

void SpxSprite::set_trigger_circle(GdVec2 center, GdFloat radius) {
	Ref<CircleShape2D> circle = memnew(CircleShape2D);
	circle->set_radius(radius);
	trigger2d->set_shape(circle);
	trigger2d->set_position(center);
}

void SpxSprite::set_trigger_enabled(GdBool trigger) {
	area2d->set_visible(trigger);
}

GdBool SpxSprite::is_trigger_enabled() {
	return area2d->is_visible();
}

CollisionShape2D *SpxSprite::get_collider(bool is_trigger) {
	return is_trigger ? trigger2d : collider2d;
}

GdBool SpxSprite::check_collision(SpxSprite *other, GdBool is_src_trigger, GdBool is_dst_trigger) {
	if (other == nullptr)
		return false;
	auto this_shape = is_src_trigger ? this->trigger2d : this->collider2d;
	auto other_shape = is_dst_trigger ? other->trigger2d : other->collider2d;
	if (!this_shape->get_shape().is_valid()) {
		return false;
	}
	if (!other_shape->get_shape().is_valid()) {
		return false;
	}
	return this_shape->get_shape()->collide(this_shape->get_global_transform(), other_shape->get_shape(), other_shape->get_global_transform());
}

GdBool SpxSprite::check_collision_with_point(GdVec2 point, GdBool is_trigger) {
	auto this_shape = is_trigger ? this->trigger2d : this->collider2d;
	if (!this_shape->get_shape().is_valid()) {
		return false;
	}

	Ref<CircleShape2D> point_shape;
	point_shape.instantiate();
	point_shape->set_radius(3);

	Transform2D point_transform(0, point);
	Transform2D sprite_transform = get_global_transform();
	bool is_colliding = this_shape->get_shape()->collide(sprite_transform, point_shape, point_transform);
	return is_colliding;
}

void SpxSprite::set_render_scale(GdVec2 new_scale) {
	_render_scale = new_scale;
	update_anim_scale();
}

void SpxSprite::update_anim_scale(){
	GdVec2 finalScale = _render_scale;
	auto target_scale = _get_actual_match_render_scale();
	if(target_scale != current_svg_scale){
		current_svg_scale = target_scale;
		if(is_svg_mode){
			if(is_single_image_mode){
				Ref<Texture2D> texture = svgMgr->get_svg_image(current_svg_path, target_scale);
				_play_single_image_animation(texture);
			}else{ 
				// Save current animation state
				bool was_playing = anim2d->is_playing();
				int current_frame = anim2d->get_frame();
				float frame_progress = anim2d->get_frame_progress();
				float custom_speed_scale = anim2d->get_playing_speed() / anim2d->get_speed_scale();
				
				auto loop = false;
				auto animation = anim2d->get_animation();
				auto old_frames = anim2d->get_sprite_frames();
				if(old_frames.is_valid() && old_frames->has_animation(animation)){
					loop = old_frames->get_animation_loop(animation);
				}
				auto base_anim_key = current_svg_anim_key;
				auto frames = svgMgr->get_svg_animation(base_anim_key, target_scale);
				if (frames.is_valid()) {
					anim2d->set_sprite_frames(frames);
					frames->set_animation_loop(base_anim_key, loop);
					
					// Set animation without resetting state
					anim2d->set_animation(base_anim_key);
					
					// Restore animation state
					int max_frame = frames->get_frame_count(base_anim_key) - 1;
					if (current_frame > max_frame) {
						current_frame = max_frame;
					}
					anim2d->set_frame_and_progress(current_frame, frame_progress);
					
					// Resume playing if it was playing before
					if (was_playing) {
						anim2d->play(base_anim_key, custom_speed_scale);
					}
				}
			}
		}
	}
	if(is_svg_mode){
		finalScale.x = finalScale.x / current_svg_scale;
		finalScale.y = finalScale.y / current_svg_scale;
	}
	anim2d->set_scale(finalScale);
}

GdVec2 SpxSprite::get_render_scale() {
	return _render_scale;
}

int SpxSprite::_get_actual_match_render_scale() {
	Vector2 current_scale = _get_actual_render_scale();
	int optimal_scale = svgMgr->calculate_svg_scale(current_scale);
	return optimal_scale;
}

Vector2 SpxSprite::_get_actual_render_scale() {
	if (!anim2d) {
		return Vector2(1.0f, 1.0f);
	}
	
	// Get the global transform scale
	Vector2 global_scale = get_global_transform().get_scale() * _render_scale;
	
	// Consider camera zoom if available
	auto camera_mgr = SpxEngine::get_singleton()->get_camera();
	if (camera_mgr) {
		Vector2 camera_zoom = camera_mgr->get_camera_zoom();
		global_scale *= camera_zoom;
	}
	
	return global_scale;
}

void SpxSprite::_on_frame_changed() {
	if (anim2d == nullptr) {
		return;
	}
	
	// Handle dynamic frame offset
	if (enable_dynamic_frame_offset) {
		String current_anim = String(anim2d->get_animation());
		int current_frame = anim2d->get_frame();
		
		Vector2 frame_offset = resMgr->get_animation_frame_offset(
			current_anim, 
			current_frame
		);
		
		Vector2 final_offset = base_offset + frame_offset;
		anim2d->set_offset(final_offset);
	}
}

void SpxSprite::set_dynamic_frame_offset_enabled(GdBool enabled) {
	enable_dynamic_frame_offset = enabled;
	
	// if enable dynamic frame offset, update current frame offset immediately
	if (enabled) {
		_on_frame_changed();
	} else {
		// if disable, restore to base offset
		anim2d->set_offset(base_offset);
	}
}

GdBool SpxSprite::is_dynamic_frame_offset_enabled() const {
	return enable_dynamic_frame_offset;
}

// ============================================================================
// Physics mode implementation
// ============================================================================

void SpxSprite::_physics_process(double delta) {
	switch (physics_mode) {
		case DYNAMIC:
			_handle_dynamic_physics(delta);
			move_and_slide();
			break;
		case KINEMATIC:
			_handle_kinematic_physics(delta);
			move_and_slide();
			break;
		case STATIC:
			_handle_static_physics(delta);
			// Don't call move_and_slide(), keep completely static
			break;
		case NO_PHYSICS:
			_handle_no_physics(delta);
			// Direct transform movement, no physics
			break;
	}
}

void SpxSprite::_handle_dynamic_physics(double delta) {
	Vector2 vel = get_velocity();
	
	// Apply gravity
	if (use_gravity && !is_on_floor()) {
		vel.y += _gravity  * delta * gravity_scale * SpxPhysicDefine::get_global_gravity();
	}
	
	// Apply forces
	vel += applied_forces * delta / mass_value;
	vel += external_forces * delta;
	
	// Apply drag
	if (drag_value > 0) {
		vel = vel.move_toward(Vector2(), drag_value * vel.length() * delta * SpxPhysicDefine::get_global_air_drag());
	}
	
	// Ground friction
	if (is_on_floor() && ABS(vel.x) > 0) {
		vel.x = Math::move_toward(double(vel.x), 0.0,  delta*friction_value *  SpxPhysicDefine::get_global_friction());
	}
	
	set_velocity(vel);
	
	// Clear single-frame forces
	applied_forces = Vector2();
}

void SpxSprite::_handle_kinematic_physics(double delta) {
	// Kinematic mode: only execute user-set movement
	// velocity is controlled by user code, no modifications here
}

void SpxSprite::_handle_static_physics(double delta) {
	// Static mode: force to remain stationary
	set_velocity(Vector2());
	external_forces = Vector2();
	applied_forces = Vector2();
}

void SpxSprite::_handle_no_physics(double delta) {
	// NoPhysics mode: direct transform movement
	Vector2 vel = get_velocity();
	if (vel != Vector2()) {
		set_global_position(get_global_position() + vel * delta);
	}
	
	// Clear all physics-related forces
	external_forces = Vector2();
	applied_forces = Vector2();
}

// Physics mode management methods
void SpxSprite::set_physics_mode(GdInt mode) {
	physics_mode = static_cast<PhysicsMode>(mode);
	_update_physics_mode();
}

GdInt SpxSprite::get_physics_mode() const {
	return static_cast<GdInt>(physics_mode);
}

void SpxSprite::set_use_gravity(GdBool enabled) {
	use_gravity = enabled;
}

GdBool SpxSprite::is_use_gravity() const {
	return use_gravity;
}

void SpxSprite::set_gravity_scale(GdFloat scale) {
	gravity_scale = scale;
}

GdFloat SpxSprite::get_gravity_scale() const {
	return gravity_scale;
}

void SpxSprite::set_drag(GdFloat drag) {
	drag_value = drag;
}

GdFloat SpxSprite::get_drag() const {
	return drag_value;
}

void SpxSprite::set_friction(GdFloat friction) {
	friction_value = friction;
}

GdFloat SpxSprite::get_friction() const {
	return friction_value;
}

void SpxSprite::_update_physics_mode() {
	switch (physics_mode) {
		case DYNAMIC:
		case KINEMATIC:
			_enable_collision();
			set_physics_process(true);
			break;
		case STATIC:
			_enable_collision();
			set_physics_process(true);  // Need processing to clear any movement
			break;
		case NO_PHYSICS:
			_disable_collision();
			set_physics_process(true);  // Still need to process transform movement
			break;
	}
}

void SpxSprite::_enable_collision() {
	collision_enabled = true;
	if (collider2d != nullptr) {
		collider2d->set_disabled(false);
	}
}

void SpxSprite::_disable_collision() {
	collision_enabled = false;
	set_velocity(Vector2());  // Clear velocity to avoid move_and_slide effects
	if (collider2d != nullptr) {
		collider2d->set_disabled(true);
	}
}

