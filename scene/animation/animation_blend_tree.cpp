/*************************************************************************/
/*  animation_blend_tree.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "animation_blend_tree.h"

#include "scene/resources/animation.h"
#include "scene/scene_string_names.h"

void AnimationNodeAnimation::set_animation(const StringName &p_name) {
	animation = p_name;
}

StringName AnimationNodeAnimation::get_animation() const {
	return animation;
}

Vector<String> (*AnimationNodeAnimation::get_editable_animation_list)() = nullptr;

void AnimationNodeAnimation::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, time, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
}

void AnimationNodeAnimation::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "animation" && get_editable_animation_list) {
		Vector<String> names = get_editable_animation_list();
		String anims;
		for (int i = 0; i < names.size(); i++) {
			if (i > 0) {
				anims += ",";
			}
			anims += String(names[i]);
		}
		if (!anims.is_empty()) {
			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = anims;
		}
	}
}

double AnimationNodeAnimation::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	AnimationPlayer *ap = state->player;
	ERR_FAIL_COND_V(!ap, 0);

	double cur_time = get_parameter(time);

	if (!ap->has_animation(animation)) {
		AnimationNodeBlendTree *tree = Object::cast_to<AnimationNodeBlendTree>(parent);
		if (tree) {
			String node_name = tree->get_node_name(Ref<AnimationNodeAnimation>(this));
			make_invalid(vformat(RTR("On BlendTree node '%s', animation not found: '%s'"), node_name, animation));

		} else {
			make_invalid(vformat(RTR("Animation not found: '%s'"), animation));
		}

		return 0;
	}

	Ref<Animation> anim = ap->get_animation(animation);
	double anim_size = (double)anim->get_length();
	double step = 0.0;
	double prev_time = cur_time;
	Animation::LoopedFlag looped_flag = Animation::LOOPED_FLAG_NONE;
	bool node_backward = play_mode == PLAY_MODE_BACKWARD;

	if (p_seek) {
		step = p_time - cur_time;
		cur_time = p_time;
	} else {
		p_time *= backward ? -1.0 : 1.0;
		cur_time = cur_time + p_time;
		step = p_time;
	}

	if (anim->get_loop_mode() == Animation::LOOP_PINGPONG) {
		if (!Math::is_zero_approx(anim_size)) {
			if (prev_time >= 0 && cur_time < 0) {
				backward = !backward;
				looped_flag = node_backward ? Animation::LOOPED_FLAG_END : Animation::LOOPED_FLAG_START;
			}
			if (prev_time <= anim_size && cur_time > anim_size) {
				backward = !backward;
				looped_flag = node_backward ? Animation::LOOPED_FLAG_START : Animation::LOOPED_FLAG_END;
			}
			cur_time = Math::pingpong(cur_time, anim_size);
		}
	} else if (anim->get_loop_mode() == Animation::LOOP_LINEAR) {
		if (!Math::is_zero_approx(anim_size)) {
			if (prev_time >= 0 && cur_time < 0) {
				looped_flag = node_backward ? Animation::LOOPED_FLAG_END : Animation::LOOPED_FLAG_START;
			}
			if (prev_time <= anim_size && cur_time > anim_size) {
				looped_flag = node_backward ? Animation::LOOPED_FLAG_START : Animation::LOOPED_FLAG_END;
			}
			cur_time = Math::fposmod(cur_time, anim_size);
		}
		backward = false;
	} else {
		if (cur_time < 0) {
			step += cur_time;
			cur_time = 0;
		} else if (cur_time > anim_size) {
			step += anim_size - cur_time;
			cur_time = anim_size;
		}
		backward = false;

		// If ended, don't progress animation. So set delta to 0.
		if (p_time > 0) {
			if (play_mode == PLAY_MODE_FORWARD) {
				if (prev_time >= anim_size) {
					step = 0;
				}
			} else {
				if (prev_time <= 0) {
					step = 0;
				}
			}
		}
	}

	if (play_mode == PLAY_MODE_FORWARD) {
		blend_animation(animation, cur_time, step, p_seek, p_is_external_seeking, 1.0, looped_flag);
	} else {
		blend_animation(animation, anim_size - cur_time, -step, p_seek, p_is_external_seeking, 1.0, looped_flag);
	}
	set_parameter(time, cur_time);

	return anim_size - cur_time;
}

String AnimationNodeAnimation::get_caption() const {
	return "Animation";
}

void AnimationNodeAnimation::set_play_mode(PlayMode p_play_mode) {
	play_mode = p_play_mode;
}

AnimationNodeAnimation::PlayMode AnimationNodeAnimation::get_play_mode() const {
	return play_mode;
}

void AnimationNodeAnimation::set_backward(bool p_backward) {
	backward = p_backward;
}

bool AnimationNodeAnimation::is_backward() const {
	return backward;
}

void AnimationNodeAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_animation", "name"), &AnimationNodeAnimation::set_animation);
	ClassDB::bind_method(D_METHOD("get_animation"), &AnimationNodeAnimation::get_animation);

	ClassDB::bind_method(D_METHOD("set_play_mode", "mode"), &AnimationNodeAnimation::set_play_mode);
	ClassDB::bind_method(D_METHOD("get_play_mode"), &AnimationNodeAnimation::get_play_mode);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation"), "set_animation", "get_animation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "play_mode", PROPERTY_HINT_ENUM, "Forward,Backward"), "set_play_mode", "get_play_mode");

	BIND_ENUM_CONSTANT(PLAY_MODE_FORWARD);
	BIND_ENUM_CONSTANT(PLAY_MODE_BACKWARD);
}

AnimationNodeAnimation::AnimationNodeAnimation() {
}

////////////////////////////////////////////////////////

void AnimationNodeSync::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeSync::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeSync::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync"), "set_use_sync", "is_using_sync");
}

void AnimationNodeSync::set_use_sync(bool p_sync) {
	sync = p_sync;
}

bool AnimationNodeSync::is_using_sync() const {
	return sync;
}

AnimationNodeSync::AnimationNodeSync() {
}

////////////////////////////////////////////////////////

void AnimationNodeOneShot::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::BOOL, active));
	r_list->push_back(PropertyInfo(Variant::BOOL, prev_active, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::FLOAT, time, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::FLOAT, remaining, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::FLOAT, time_to_restart, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
}

Variant AnimationNodeOneShot::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == active || p_parameter == prev_active) {
		return false;
	} else if (p_parameter == time_to_restart) {
		return -1;
	} else {
		return 0.0;
	}
}

void AnimationNodeOneShot::set_fadein_time(double p_time) {
	fade_in = p_time;
}

void AnimationNodeOneShot::set_fadeout_time(double p_time) {
	fade_out = p_time;
}

double AnimationNodeOneShot::get_fadein_time() const {
	return fade_in;
}

double AnimationNodeOneShot::get_fadeout_time() const {
	return fade_out;
}

void AnimationNodeOneShot::set_autorestart(bool p_active) {
	autorestart = p_active;
}

void AnimationNodeOneShot::set_autorestart_delay(double p_time) {
	autorestart_delay = p_time;
}

void AnimationNodeOneShot::set_autorestart_random_delay(double p_time) {
	autorestart_random_delay = p_time;
}

bool AnimationNodeOneShot::has_autorestart() const {
	return autorestart;
}

double AnimationNodeOneShot::get_autorestart_delay() const {
	return autorestart_delay;
}

double AnimationNodeOneShot::get_autorestart_random_delay() const {
	return autorestart_random_delay;
}

void AnimationNodeOneShot::set_mix_mode(MixMode p_mix) {
	mix = p_mix;
}

AnimationNodeOneShot::MixMode AnimationNodeOneShot::get_mix_mode() const {
	return mix;
}

String AnimationNodeOneShot::get_caption() const {
	return "OneShot";
}

bool AnimationNodeOneShot::has_filter() const {
	return true;
}

double AnimationNodeOneShot::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	bool cur_active = get_parameter(active);
	bool cur_prev_active = get_parameter(prev_active);
	double cur_time = get_parameter(time);
	double cur_remaining = get_parameter(remaining);
	double cur_time_to_restart = get_parameter(time_to_restart);

	if (!cur_active) {
		//make it as if this node doesn't exist, pass input 0 by.
		if (cur_prev_active) {
			set_parameter(prev_active, false);
		}
		if (cur_time_to_restart >= 0.0 && !p_seek) {
			cur_time_to_restart -= p_time;
			if (cur_time_to_restart < 0) {
				//restart
				set_parameter(active, true);
				cur_active = true;
			}
			set_parameter(time_to_restart, cur_time_to_restart);
		}

		return blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, sync);
	}

	bool os_seek = p_seek;

	if (p_seek) {
		cur_time = p_time;
	}
	bool do_start = !cur_prev_active;

	if (do_start) {
		cur_time = 0;
		os_seek = true;
		set_parameter(prev_active, true);
	}

	real_t blend;

	if (cur_time < fade_in) {
		if (fade_in > 0) {
			blend = cur_time / fade_in;
		} else {
			blend = 0;
		}
	} else if (!do_start && cur_remaining <= fade_out) {
		if (fade_out > 0) {
			blend = (cur_remaining / fade_out);
		} else {
			blend = 0;
		}
	} else {
		blend = 1.0;
	}

	double main_rem;
	if (mix == MIX_MODE_ADD) {
		main_rem = blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, sync);
	} else {
		main_rem = blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0 - blend, FILTER_BLEND, sync); // Unlike below, processing this edge is a corner case.
	}
	double os_rem = blend_input(1, os_seek ? cur_time : p_time, os_seek, p_is_external_seeking, Math::is_zero_approx(blend) ? CMP_EPSILON : blend, FILTER_PASS, true); // Blend values must be more than CMP_EPSILON to process discrete keys in edge.

	if (do_start) {
		cur_remaining = os_rem;
	}

	if (!p_seek) {
		cur_time += p_time;
		cur_remaining = os_rem;
		if (cur_remaining <= 0) {
			set_parameter(active, false);
			set_parameter(prev_active, false);
			if (autorestart) {
				double restart_sec = autorestart_delay + Math::randd() * autorestart_random_delay;
				set_parameter(time_to_restart, restart_sec);
			}
		}
	}

	set_parameter(time, cur_time);
	set_parameter(remaining, cur_remaining);

	return MAX(main_rem, cur_remaining);
}

void AnimationNodeOneShot::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_fadein_time", "time"), &AnimationNodeOneShot::set_fadein_time);
	ClassDB::bind_method(D_METHOD("get_fadein_time"), &AnimationNodeOneShot::get_fadein_time);

	ClassDB::bind_method(D_METHOD("set_fadeout_time", "time"), &AnimationNodeOneShot::set_fadeout_time);
	ClassDB::bind_method(D_METHOD("get_fadeout_time"), &AnimationNodeOneShot::get_fadeout_time);

	ClassDB::bind_method(D_METHOD("set_autorestart", "enable"), &AnimationNodeOneShot::set_autorestart);
	ClassDB::bind_method(D_METHOD("has_autorestart"), &AnimationNodeOneShot::has_autorestart);

	ClassDB::bind_method(D_METHOD("set_autorestart_delay", "enable"), &AnimationNodeOneShot::set_autorestart_delay);
	ClassDB::bind_method(D_METHOD("get_autorestart_delay"), &AnimationNodeOneShot::get_autorestart_delay);

	ClassDB::bind_method(D_METHOD("set_autorestart_random_delay", "enable"), &AnimationNodeOneShot::set_autorestart_random_delay);
	ClassDB::bind_method(D_METHOD("get_autorestart_random_delay"), &AnimationNodeOneShot::get_autorestart_random_delay);

	ClassDB::bind_method(D_METHOD("set_mix_mode", "mode"), &AnimationNodeOneShot::set_mix_mode);
	ClassDB::bind_method(D_METHOD("get_mix_mode"), &AnimationNodeOneShot::get_mix_mode);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_mode", PROPERTY_HINT_ENUM, "Blend,Add"), "set_mix_mode", "get_mix_mode");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fadein_time", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater,suffix:s"), "set_fadein_time", "get_fadein_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fadeout_time", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater,suffix:s"), "set_fadeout_time", "get_fadeout_time");

	ADD_GROUP("Auto Restart", "autorestart_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autorestart"), "set_autorestart", "has_autorestart");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "autorestart_delay", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater,suffix:s"), "set_autorestart_delay", "get_autorestart_delay");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "autorestart_random_delay", PROPERTY_HINT_RANGE, "0,60,0.01,or_greater,suffix:s"), "set_autorestart_random_delay", "get_autorestart_random_delay");

	BIND_ENUM_CONSTANT(MIX_MODE_BLEND);
	BIND_ENUM_CONSTANT(MIX_MODE_ADD);
}

AnimationNodeOneShot::AnimationNodeOneShot() {
	add_input("in");
	add_input("shot");
}

////////////////////////////////////////////////

void AnimationNodeAdd2::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, add_amount, PROPERTY_HINT_RANGE, "0,1,0.01"));
}

Variant AnimationNodeAdd2::get_parameter_default_value(const StringName &p_parameter) const {
	return 0;
}

String AnimationNodeAdd2::get_caption() const {
	return "Add2";
}

bool AnimationNodeAdd2::has_filter() const {
	return true;
}

double AnimationNodeAdd2::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double amount = get_parameter(add_amount);
	double rem0 = blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, sync);
	blend_input(1, p_time, p_seek, p_is_external_seeking, amount, FILTER_PASS, sync);

	return rem0;
}

void AnimationNodeAdd2::_bind_methods() {
}

AnimationNodeAdd2::AnimationNodeAdd2() {
	add_input("in");
	add_input("add");
}

////////////////////////////////////////////////

void AnimationNodeAdd3::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, add_amount, PROPERTY_HINT_RANGE, "-1,1,0.01"));
}

Variant AnimationNodeAdd3::get_parameter_default_value(const StringName &p_parameter) const {
	return 0;
}

String AnimationNodeAdd3::get_caption() const {
	return "Add3";
}

bool AnimationNodeAdd3::has_filter() const {
	return true;
}

double AnimationNodeAdd3::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double amount = get_parameter(add_amount);
	blend_input(0, p_time, p_seek, p_is_external_seeking, MAX(0, -amount), FILTER_PASS, sync);
	double rem0 = blend_input(1, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, sync);
	blend_input(2, p_time, p_seek, p_is_external_seeking, MAX(0, amount), FILTER_PASS, sync);

	return rem0;
}

void AnimationNodeAdd3::_bind_methods() {
}

AnimationNodeAdd3::AnimationNodeAdd3() {
	add_input("-add");
	add_input("in");
	add_input("+add");
}

/////////////////////////////////////////////

void AnimationNodeBlend2::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, blend_amount, PROPERTY_HINT_RANGE, "0,1,0.01"));
}

Variant AnimationNodeBlend2::get_parameter_default_value(const StringName &p_parameter) const {
	return 0; //for blend amount
}

String AnimationNodeBlend2::get_caption() const {
	return "Blend2";
}

double AnimationNodeBlend2::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double amount = get_parameter(blend_amount);

	double rem0 = blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0 - amount, FILTER_BLEND, sync);
	double rem1 = blend_input(1, p_time, p_seek, p_is_external_seeking, amount, FILTER_PASS, sync);

	return amount > 0.5 ? rem1 : rem0; //hacky but good enough
}

bool AnimationNodeBlend2::has_filter() const {
	return true;
}

void AnimationNodeBlend2::_bind_methods() {
}

AnimationNodeBlend2::AnimationNodeBlend2() {
	add_input("in");
	add_input("blend");
}

//////////////////////////////////////

void AnimationNodeBlend3::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, blend_amount, PROPERTY_HINT_RANGE, "-1,1,0.01"));
}

Variant AnimationNodeBlend3::get_parameter_default_value(const StringName &p_parameter) const {
	return 0; //for blend amount
}

String AnimationNodeBlend3::get_caption() const {
	return "Blend3";
}

double AnimationNodeBlend3::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double amount = get_parameter(blend_amount);
	double rem0 = blend_input(0, p_time, p_seek, p_is_external_seeking, MAX(0, -amount), FILTER_IGNORE, sync);
	double rem1 = blend_input(1, p_time, p_seek, p_is_external_seeking, 1.0 - ABS(amount), FILTER_IGNORE, sync);
	double rem2 = blend_input(2, p_time, p_seek, p_is_external_seeking, MAX(0, amount), FILTER_IGNORE, sync);

	return amount > 0.5 ? rem2 : (amount < -0.5 ? rem0 : rem1); //hacky but good enough
}

void AnimationNodeBlend3::_bind_methods() {
}

AnimationNodeBlend3::AnimationNodeBlend3() {
	add_input("-blend");
	add_input("in");
	add_input("+blend");
}

/////////////////////////////////

void AnimationNodeTimeScale::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, scale, PROPERTY_HINT_RANGE, "-32,32,0.01,or_less,or_greater"));
}

Variant AnimationNodeTimeScale::get_parameter_default_value(const StringName &p_parameter) const {
	return 1.0; //initial timescale
}

String AnimationNodeTimeScale::get_caption() const {
	return "TimeScale";
}

double AnimationNodeTimeScale::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double cur_scale = get_parameter(scale);
	if (p_seek) {
		return blend_input(0, p_time, true, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
	} else {
		return blend_input(0, p_time * cur_scale, false, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
	}
}

void AnimationNodeTimeScale::_bind_methods() {
}

AnimationNodeTimeScale::AnimationNodeTimeScale() {
	add_input("in");
}

////////////////////////////////////

void AnimationNodeTimeSeek::get_parameter_list(List<PropertyInfo> *r_list) const {
	r_list->push_back(PropertyInfo(Variant::FLOAT, seek_pos, PROPERTY_HINT_RANGE, "-1,3600,0.01,or_greater"));
}

Variant AnimationNodeTimeSeek::get_parameter_default_value(const StringName &p_parameter) const {
	return 1.0; //initial timescale
}

String AnimationNodeTimeSeek::get_caption() const {
	return "Seek";
}

double AnimationNodeTimeSeek::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	double cur_seek_pos = get_parameter(seek_pos);
	if (p_seek) {
		return blend_input(0, p_time, true, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
	} else if (cur_seek_pos >= 0) {
		double ret = blend_input(0, cur_seek_pos, true, true, 1.0, FILTER_IGNORE, true);
		set_parameter(seek_pos, -1.0); //reset
		return ret;
	} else {
		return blend_input(0, p_time, false, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
	}
}

void AnimationNodeTimeSeek::_bind_methods() {
}

AnimationNodeTimeSeek::AnimationNodeTimeSeek() {
	add_input("in");
}

/////////////////////////////////////////////////

void AnimationNodeTransition::get_parameter_list(List<PropertyInfo> *r_list) const {
	String anims;
	for (int i = 0; i < enabled_inputs; i++) {
		if (i > 0) {
			anims += ",";
		}
		anims += inputs[i].name;
	}

	r_list->push_back(PropertyInfo(Variant::INT, current, PROPERTY_HINT_ENUM, anims));
	r_list->push_back(PropertyInfo(Variant::INT, prev_current, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::INT, prev, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::FLOAT, time, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
	r_list->push_back(PropertyInfo(Variant::FLOAT, prev_xfading, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
}

Variant AnimationNodeTransition::get_parameter_default_value(const StringName &p_parameter) const {
	if (p_parameter == time || p_parameter == prev_xfading) {
		return 0.0;
	} else if (p_parameter == prev || p_parameter == prev_current) {
		return -1;
	} else {
		return 0;
	}
}

String AnimationNodeTransition::get_caption() const {
	return "Transition";
}

void AnimationNodeTransition::_update_inputs() {
	while (get_input_count() < enabled_inputs) {
		add_input(inputs[get_input_count()].name);
	}

	while (get_input_count() > enabled_inputs) {
		remove_input(get_input_count() - 1);
	}
}

void AnimationNodeTransition::set_enabled_inputs(int p_inputs) {
	ERR_FAIL_INDEX(p_inputs, MAX_INPUTS);
	enabled_inputs = p_inputs;
	_update_inputs();
}

int AnimationNodeTransition::get_enabled_inputs() {
	return enabled_inputs;
}

void AnimationNodeTransition::set_input_as_auto_advance(int p_input, bool p_enable) {
	ERR_FAIL_INDEX(p_input, MAX_INPUTS);
	inputs[p_input].auto_advance = p_enable;
}

bool AnimationNodeTransition::is_input_set_as_auto_advance(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, MAX_INPUTS, false);
	return inputs[p_input].auto_advance;
}

void AnimationNodeTransition::set_input_caption(int p_input, const String &p_name) {
	ERR_FAIL_INDEX(p_input, MAX_INPUTS);
	inputs[p_input].name = p_name;
	set_input_name(p_input, p_name);
}

String AnimationNodeTransition::get_input_caption(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, MAX_INPUTS, String());
	return inputs[p_input].name;
}

void AnimationNodeTransition::set_xfade_time(double p_fade) {
	xfade_time = p_fade;
}

double AnimationNodeTransition::get_xfade_time() const {
	return xfade_time;
}

void AnimationNodeTransition::set_xfade_curve(const Ref<Curve> &p_curve) {
	xfade_curve = p_curve;
}

Ref<Curve> AnimationNodeTransition::get_xfade_curve() const {
	return xfade_curve;
}

void AnimationNodeTransition::set_from_start(bool p_from_start) {
	from_start = p_from_start;
}

bool AnimationNodeTransition::is_from_start() const {
	return from_start;
}

double AnimationNodeTransition::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	int cur_current = get_parameter(current);
	int cur_prev = get_parameter(prev);
	int cur_prev_current = get_parameter(prev_current);

	double cur_time = get_parameter(time);
	double cur_prev_xfading = get_parameter(prev_xfading);

	bool switched = cur_current != cur_prev_current;

	if (switched) {
		set_parameter(prev_current, cur_current);
		set_parameter(prev, cur_prev_current);

		cur_prev = cur_prev_current;
		cur_prev_xfading = xfade_time;
		cur_time = 0;
		switched = true;
	}

	if (cur_current < 0 || cur_current >= enabled_inputs || cur_prev >= enabled_inputs) {
		return 0;
	}

	double rem = 0.0;

	if (sync) {
		for (int i = 0; i < enabled_inputs; i++) {
			if (i != cur_current && i != cur_prev) {
				blend_input(i, p_time, p_seek, p_is_external_seeking, 0, FILTER_IGNORE, true);
			}
		}
	}

	if (cur_prev < 0) { // process current animation, check for transition

		rem = blend_input(cur_current, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, true);

		if (p_seek) {
			cur_time = p_time;
		} else {
			cur_time += p_time;
		}

		if (inputs[cur_current].auto_advance && rem <= xfade_time) {
			set_parameter(current, (cur_current + 1) % enabled_inputs);
		}

	} else { // cross-fading from prev to current

		real_t blend = xfade_time == 0 ? 0 : (cur_prev_xfading / xfade_time);
		if (xfade_curve.is_valid()) {
			blend = xfade_curve->sample(blend);
		}

		// Blend values must be more than CMP_EPSILON to process discrete keys in edge.
		real_t blend_inv = 1.0 - blend;
		if (from_start && !p_seek && switched) { //just switched, seek to start of current
			rem = blend_input(cur_current, 0, true, p_is_external_seeking, Math::is_zero_approx(blend_inv) ? CMP_EPSILON : blend_inv, FILTER_IGNORE, true);
		} else {
			rem = blend_input(cur_current, p_time, p_seek, p_is_external_seeking, Math::is_zero_approx(blend_inv) ? CMP_EPSILON : blend_inv, FILTER_IGNORE, true);
		}

		if (p_seek) {
			blend_input(cur_prev, p_time, true, p_is_external_seeking, Math::is_zero_approx(blend) ? CMP_EPSILON : blend, FILTER_IGNORE, true);
			cur_time = p_time;
		} else {
			blend_input(cur_prev, p_time, false, p_is_external_seeking, Math::is_zero_approx(blend) ? CMP_EPSILON : blend, FILTER_IGNORE, true);
			cur_time += p_time;
			cur_prev_xfading -= p_time;
			if (cur_prev_xfading < 0) {
				set_parameter(prev, -1);
			}
		}
	}

	set_parameter(time, cur_time);
	set_parameter(prev_xfading, cur_prev_xfading);

	return rem;
}

void AnimationNodeTransition::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name.begins_with("input_")) {
		String n = p_property.name.get_slicec('/', 0).get_slicec('_', 1);
		if (n != "count") {
			int idx = n.to_int();
			if (idx >= enabled_inputs) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
}

void AnimationNodeTransition::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled_inputs", "amount"), &AnimationNodeTransition::set_enabled_inputs);
	ClassDB::bind_method(D_METHOD("get_enabled_inputs"), &AnimationNodeTransition::get_enabled_inputs);

	ClassDB::bind_method(D_METHOD("set_input_as_auto_advance", "input", "enable"), &AnimationNodeTransition::set_input_as_auto_advance);
	ClassDB::bind_method(D_METHOD("is_input_set_as_auto_advance", "input"), &AnimationNodeTransition::is_input_set_as_auto_advance);

	ClassDB::bind_method(D_METHOD("set_input_caption", "input", "caption"), &AnimationNodeTransition::set_input_caption);
	ClassDB::bind_method(D_METHOD("get_input_caption", "input"), &AnimationNodeTransition::get_input_caption);

	ClassDB::bind_method(D_METHOD("set_xfade_time", "time"), &AnimationNodeTransition::set_xfade_time);
	ClassDB::bind_method(D_METHOD("get_xfade_time"), &AnimationNodeTransition::get_xfade_time);

	ClassDB::bind_method(D_METHOD("set_xfade_curve", "curve"), &AnimationNodeTransition::set_xfade_curve);
	ClassDB::bind_method(D_METHOD("get_xfade_curve"), &AnimationNodeTransition::get_xfade_curve);

	ClassDB::bind_method(D_METHOD("set_from_start", "from_start"), &AnimationNodeTransition::set_from_start);
	ClassDB::bind_method(D_METHOD("is_from_start"), &AnimationNodeTransition::is_from_start);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "enabled_inputs", PROPERTY_HINT_RANGE, "0,64,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_enabled_inputs", "get_enabled_inputs");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "xfade_time", PROPERTY_HINT_RANGE, "0,120,0.01,suffix:s"), "set_xfade_time", "get_xfade_time");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "xfade_curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_xfade_curve", "get_xfade_curve");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "from_start"), "set_from_start", "is_from_start");

	for (int i = 0; i < MAX_INPUTS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "input_" + itos(i) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_input_caption", "get_input_caption", i);
		ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "input_" + itos(i) + "/auto_advance", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_input_as_auto_advance", "is_input_set_as_auto_advance", i);
	}
}

AnimationNodeTransition::AnimationNodeTransition() {
	for (int i = 0; i < MAX_INPUTS; i++) {
		inputs[i].name = "state " + itos(i);
	}
}

/////////////////////

String AnimationNodeOutput::get_caption() const {
	return "Output";
}

double AnimationNodeOutput::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	return blend_input(0, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
}

AnimationNodeOutput::AnimationNodeOutput() {
	add_input("output");
}

///////////////////////////////////////////////////////
void AnimationNodeBlendTree::add_node(const StringName &p_name, Ref<AnimationNode> p_node, const Vector2 &p_position) {
	ERR_FAIL_COND(nodes.has(p_name));
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(String(p_name).contains("/"));

	Node n;
	n.node = p_node;
	n.position = p_position;
	n.connections.resize(n.node->get_input_count());
	nodes[p_name] = n;

	emit_changed();
	emit_signal(SNAME("tree_changed"));

	p_node->connect("tree_changed", callable_mp(this, &AnimationNodeBlendTree::_tree_changed), CONNECT_REFERENCE_COUNTED);
	p_node->connect("changed", callable_mp(this, &AnimationNodeBlendTree::_node_changed).bind(p_name), CONNECT_REFERENCE_COUNTED);
}

Ref<AnimationNode> AnimationNodeBlendTree::get_node(const StringName &p_name) const {
	ERR_FAIL_COND_V(!nodes.has(p_name), Ref<AnimationNode>());

	return nodes[p_name].node;
}

StringName AnimationNodeBlendTree::get_node_name(const Ref<AnimationNode> &p_node) const {
	for (const KeyValue<StringName, Node> &E : nodes) {
		if (E.value.node == p_node) {
			return E.key;
		}
	}

	ERR_FAIL_V(StringName());
}

void AnimationNodeBlendTree::set_node_position(const StringName &p_node, const Vector2 &p_position) {
	ERR_FAIL_COND(!nodes.has(p_node));
	nodes[p_node].position = p_position;
}

Vector2 AnimationNodeBlendTree::get_node_position(const StringName &p_node) const {
	ERR_FAIL_COND_V(!nodes.has(p_node), Vector2());
	return nodes[p_node].position;
}

void AnimationNodeBlendTree::get_child_nodes(List<ChildNode> *r_child_nodes) {
	Vector<StringName> ns;

	for (const KeyValue<StringName, Node> &E : nodes) {
		ns.push_back(E.key);
	}

	ns.sort_custom<StringName::AlphCompare>();

	for (int i = 0; i < ns.size(); i++) {
		ChildNode cn;
		cn.name = ns[i];
		cn.node = nodes[cn.name].node;
		r_child_nodes->push_back(cn);
	}
}

bool AnimationNodeBlendTree::has_node(const StringName &p_name) const {
	return nodes.has(p_name);
}

Vector<StringName> AnimationNodeBlendTree::get_node_connection_array(const StringName &p_name) const {
	ERR_FAIL_COND_V(!nodes.has(p_name), Vector<StringName>());
	return nodes[p_name].connections;
}

void AnimationNodeBlendTree::remove_node(const StringName &p_name) {
	ERR_FAIL_COND(!nodes.has(p_name));
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output); //can't delete output

	{
		Ref<AnimationNode> node = nodes[p_name].node;
		node->disconnect("tree_changed", callable_mp(this, &AnimationNodeBlendTree::_tree_changed));
		node->disconnect("changed", callable_mp(this, &AnimationNodeBlendTree::_node_changed));
	}

	nodes.erase(p_name);

	//erase connections to name
	for (KeyValue<StringName, Node> &E : nodes) {
		for (int i = 0; i < E.value.connections.size(); i++) {
			if (E.value.connections[i] == p_name) {
				E.value.connections.write[i] = StringName();
			}
		}
	}

	emit_changed();
	emit_signal(SNAME("tree_changed"));
}

void AnimationNodeBlendTree::rename_node(const StringName &p_name, const StringName &p_new_name) {
	ERR_FAIL_COND(!nodes.has(p_name));
	ERR_FAIL_COND(nodes.has(p_new_name));
	ERR_FAIL_COND(p_name == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(p_new_name == SceneStringNames::get_singleton()->output);

	nodes[p_name].node->disconnect("changed", callable_mp(this, &AnimationNodeBlendTree::_node_changed));

	nodes[p_new_name] = nodes[p_name];
	nodes.erase(p_name);

	//rename connections
	for (KeyValue<StringName, Node> &E : nodes) {
		for (int i = 0; i < E.value.connections.size(); i++) {
			if (E.value.connections[i] == p_name) {
				E.value.connections.write[i] = p_new_name;
			}
		}
	}
	//connection must be done with new name
	nodes[p_new_name].node->connect("changed", callable_mp(this, &AnimationNodeBlendTree::_node_changed).bind(p_new_name), CONNECT_REFERENCE_COUNTED);

	emit_signal(SNAME("tree_changed"));
}

void AnimationNodeBlendTree::connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node) {
	ERR_FAIL_COND(!nodes.has(p_output_node));
	ERR_FAIL_COND(!nodes.has(p_input_node));
	ERR_FAIL_COND(p_output_node == SceneStringNames::get_singleton()->output);
	ERR_FAIL_COND(p_input_node == p_output_node);

	Ref<AnimationNode> input = nodes[p_input_node].node;
	ERR_FAIL_INDEX(p_input_index, nodes[p_input_node].connections.size());

	for (KeyValue<StringName, Node> &E : nodes) {
		for (int i = 0; i < E.value.connections.size(); i++) {
			StringName output = E.value.connections[i];
			ERR_FAIL_COND(output == p_output_node);
		}
	}

	nodes[p_input_node].connections.write[p_input_index] = p_output_node;

	emit_changed();
}

void AnimationNodeBlendTree::disconnect_node(const StringName &p_node, int p_input_index) {
	ERR_FAIL_COND(!nodes.has(p_node));

	Ref<AnimationNode> input = nodes[p_node].node;
	ERR_FAIL_INDEX(p_input_index, nodes[p_node].connections.size());

	nodes[p_node].connections.write[p_input_index] = StringName();
}

AnimationNodeBlendTree::ConnectionError AnimationNodeBlendTree::can_connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node) const {
	if (!nodes.has(p_output_node) || p_output_node == SceneStringNames::get_singleton()->output) {
		return CONNECTION_ERROR_NO_OUTPUT;
	}

	if (!nodes.has(p_input_node)) {
		return CONNECTION_ERROR_NO_INPUT;
	}

	if (p_input_node == p_output_node) {
		return CONNECTION_ERROR_SAME_NODE;
	}

	Ref<AnimationNode> input = nodes[p_input_node].node;

	if (p_input_index < 0 || p_input_index >= nodes[p_input_node].connections.size()) {
		return CONNECTION_ERROR_NO_INPUT_INDEX;
	}

	if (nodes[p_input_node].connections[p_input_index] != StringName()) {
		return CONNECTION_ERROR_CONNECTION_EXISTS;
	}

	for (const KeyValue<StringName, Node> &E : nodes) {
		for (int i = 0; i < E.value.connections.size(); i++) {
			const StringName output = E.value.connections[i];
			if (output == p_output_node) {
				return CONNECTION_ERROR_CONNECTION_EXISTS;
			}
		}
	}
	return CONNECTION_OK;
}

void AnimationNodeBlendTree::get_node_connections(List<NodeConnection> *r_connections) const {
	for (const KeyValue<StringName, Node> &E : nodes) {
		for (int i = 0; i < E.value.connections.size(); i++) {
			const StringName output = E.value.connections[i];
			if (output != StringName()) {
				NodeConnection nc;
				nc.input_node = E.key;
				nc.input_index = i;
				nc.output_node = output;
				r_connections->push_back(nc);
			}
		}
	}
}

String AnimationNodeBlendTree::get_caption() const {
	return "BlendTree";
}

double AnimationNodeBlendTree::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	Ref<AnimationNodeOutput> output = nodes[SceneStringNames::get_singleton()->output].node;
	return _blend_node("output", nodes[SceneStringNames::get_singleton()->output].connections, this, output, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
}

void AnimationNodeBlendTree::get_node_list(List<StringName> *r_list) {
	for (const KeyValue<StringName, Node> &E : nodes) {
		r_list->push_back(E.key);
	}
}

void AnimationNodeBlendTree::set_graph_offset(const Vector2 &p_graph_offset) {
	graph_offset = p_graph_offset;
}

Vector2 AnimationNodeBlendTree::get_graph_offset() const {
	return graph_offset;
}

Ref<AnimationNode> AnimationNodeBlendTree::get_child_by_name(const StringName &p_name) {
	return get_node(p_name);
}

bool AnimationNodeBlendTree::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;
	if (prop_name.begins_with("nodes/")) {
		String node_name = prop_name.get_slicec('/', 1);
		String what = prop_name.get_slicec('/', 2);

		if (what == "node") {
			Ref<AnimationNode> anode = p_value;
			if (anode.is_valid()) {
				add_node(node_name, p_value);
			}
			return true;
		}

		if (what == "position") {
			if (nodes.has(node_name)) {
				nodes[node_name].position = p_value;
			}
			return true;
		}
	} else if (prop_name == "node_connections") {
		Array conns = p_value;
		ERR_FAIL_COND_V(conns.size() % 3 != 0, false);

		for (int i = 0; i < conns.size(); i += 3) {
			connect_node(conns[i], conns[i + 1], conns[i + 2]);
		}
		return true;
	}

	return false;
}

bool AnimationNodeBlendTree::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;
	if (prop_name.begins_with("nodes/")) {
		String node_name = prop_name.get_slicec('/', 1);
		String what = prop_name.get_slicec('/', 2);

		if (what == "node") {
			if (nodes.has(node_name)) {
				r_ret = nodes[node_name].node;
				return true;
			}
		}

		if (what == "position") {
			if (nodes.has(node_name)) {
				r_ret = nodes[node_name].position;
				return true;
			}
		}
	} else if (prop_name == "node_connections") {
		List<NodeConnection> nc;
		get_node_connections(&nc);
		Array conns;
		conns.resize(nc.size() * 3);

		int idx = 0;
		for (const NodeConnection &E : nc) {
			conns[idx * 3 + 0] = E.input_node;
			conns[idx * 3 + 1] = E.input_index;
			conns[idx * 3 + 2] = E.output_node;
			idx++;
		}

		r_ret = conns;
		return true;
	}

	return false;
}

void AnimationNodeBlendTree::_get_property_list(List<PropertyInfo> *p_list) const {
	List<StringName> names;
	for (const KeyValue<StringName, Node> &E : nodes) {
		names.push_back(E.key);
	}
	names.sort_custom<StringName::AlphCompare>();

	for (const StringName &E : names) {
		String prop_name = E;
		if (prop_name != "output") {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "nodes/" + prop_name + "/node", PROPERTY_HINT_RESOURCE_TYPE, "AnimationNode", PROPERTY_USAGE_NO_EDITOR));
		}
		p_list->push_back(PropertyInfo(Variant::VECTOR2, "nodes/" + prop_name + "/position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "node_connections", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
}

void AnimationNodeBlendTree::reset_state() {
	graph_offset = Vector2();
	nodes.clear();
	_initialize_node_tree();
	emit_changed();
	emit_signal(SNAME("tree_changed"));
}

void AnimationNodeBlendTree::_tree_changed() {
	emit_signal(SNAME("tree_changed"));
}

void AnimationNodeBlendTree::_node_changed(const StringName &p_node) {
	ERR_FAIL_COND(!nodes.has(p_node));
	nodes[p_node].connections.resize(nodes[p_node].node->get_input_count());
	emit_signal(SNAME("node_changed"), p_node);
}

void AnimationNodeBlendTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_node", "name", "node", "position"), &AnimationNodeBlendTree::add_node, DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("get_node", "name"), &AnimationNodeBlendTree::get_node);
	ClassDB::bind_method(D_METHOD("remove_node", "name"), &AnimationNodeBlendTree::remove_node);
	ClassDB::bind_method(D_METHOD("rename_node", "name", "new_name"), &AnimationNodeBlendTree::rename_node);
	ClassDB::bind_method(D_METHOD("has_node", "name"), &AnimationNodeBlendTree::has_node);
	ClassDB::bind_method(D_METHOD("connect_node", "input_node", "input_index", "output_node"), &AnimationNodeBlendTree::connect_node);
	ClassDB::bind_method(D_METHOD("disconnect_node", "input_node", "input_index"), &AnimationNodeBlendTree::disconnect_node);

	ClassDB::bind_method(D_METHOD("set_node_position", "name", "position"), &AnimationNodeBlendTree::set_node_position);
	ClassDB::bind_method(D_METHOD("get_node_position", "name"), &AnimationNodeBlendTree::get_node_position);

	ClassDB::bind_method(D_METHOD("set_graph_offset", "offset"), &AnimationNodeBlendTree::set_graph_offset);
	ClassDB::bind_method(D_METHOD("get_graph_offset"), &AnimationNodeBlendTree::get_graph_offset);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "graph_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_graph_offset", "get_graph_offset");

	BIND_CONSTANT(CONNECTION_OK);
	BIND_CONSTANT(CONNECTION_ERROR_NO_INPUT);
	BIND_CONSTANT(CONNECTION_ERROR_NO_INPUT_INDEX);
	BIND_CONSTANT(CONNECTION_ERROR_NO_OUTPUT);
	BIND_CONSTANT(CONNECTION_ERROR_SAME_NODE);
	BIND_CONSTANT(CONNECTION_ERROR_CONNECTION_EXISTS);

	ADD_SIGNAL(MethodInfo("node_changed", PropertyInfo(Variant::STRING_NAME, "node_name")));
}

void AnimationNodeBlendTree::_initialize_node_tree() {
	Ref<AnimationNodeOutput> output;
	output.instantiate();
	Node n;
	n.node = output;
	n.position = Vector2(300, 150);
	n.connections.resize(1);
	nodes["output"] = n;
}

AnimationNodeBlendTree::AnimationNodeBlendTree() {
	_initialize_node_tree();
}

AnimationNodeBlendTree::~AnimationNodeBlendTree() {
}
