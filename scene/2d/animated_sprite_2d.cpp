/*************************************************************************/
/*  animated_sprite_2d.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "animated_sprite_2d.h"

#include "scene/main/viewport.h"
#include "scene/scene_string_names.h"

#ifdef TOOLS_ENABLED
Dictionary AnimatedSprite2D::_edit_get_state() const {
	Dictionary state = Node2D::_edit_get_state();
	state["offset"] = offset;
	return state;
}

void AnimatedSprite2D::_edit_set_state(const Dictionary &p_state) {
	Node2D::_edit_set_state(p_state);
	set_offset(p_state["offset"]);
}

void AnimatedSprite2D::_edit_set_pivot(const Point2 &p_pivot) {
	set_offset(get_offset() - p_pivot);
	set_position(get_transform().xform(p_pivot));
}

Point2 AnimatedSprite2D::_edit_get_pivot() const {
	return Vector2();
}

bool AnimatedSprite2D::_edit_use_pivot() const {
	return true;
}

Rect2 AnimatedSprite2D::_edit_get_rect() const {
	return _get_rect();
}

bool AnimatedSprite2D::_edit_use_rect() const {
	if (!frames.is_valid() || !frames->has_animation(animation) || frame < 0 || frame >= frames->get_frame_count(animation)) {
		return false;
	}
	Ref<Texture2D> t;
	if (animation) {
		t = frames->get_frame(animation, frame);
	}
	return t.is_valid();
}
#endif

Rect2 AnimatedSprite2D::get_anchorable_rect() const {
	return _get_rect();
}

Rect2 AnimatedSprite2D::_get_rect() const {
	if (!frames.is_valid() || !frames->has_animation(animation) || frame < 0 || frame >= frames->get_frame_count(animation)) {
		return Rect2();
	}

	Ref<Texture2D> t;
	if (animation) {
		t = frames->get_frame(animation, frame);
	}
	if (t.is_null()) {
		return Rect2();
	}
	Size2 s = t->get_size();

	Point2 ofs = offset;
	if (centered) {
		ofs -= s / 2;
	}

	if (s == Size2(0, 0)) {
		s = Size2(1, 1);
	}

	return Rect2(ofs, s);
}

void AnimatedSprite2D::_validate_property(PropertyInfo &property) const {
	if (!frames.is_valid()) {
		return;
	}
	if (property.name == "animation") {
		property.hint = PROPERTY_HINT_ENUM;
		List<StringName> names;
		frames->get_animation_list(&names);
		names.sort_custom<StringName::AlphCompare>();

		bool current_found = false;

		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			if (E->prev()) {
				property.hint_string += ",";
			}

			property.hint_string += String(E->get());
			if (animation == E->get()) {
				current_found = true;
			}
		}

		if (!current_found) {
			if (property.hint_string.is_empty()) {
				property.hint_string = String(animation);
			} else {
				property.hint_string = String(animation) + "," + property.hint_string;
			}
		}
	}

	if (property.name == "frame") {
		property.hint = PROPERTY_HINT_RANGE;
		if (frames->has_animation(animation) && frames->get_frame_count(animation) > 1) {
			property.hint_string = "0," + itos(frames->get_frame_count(animation) - 1) + ",1";
		}
		property.usage |= PROPERTY_USAGE_KEYING_INCREMENTS;
	}
}

void AnimatedSprite2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (frames.is_null()) {
				return;
			}
			if (!frames->has_animation(animation)) {
				return;
			}
			if (frame < 0) {
				return;
			}

			double speed = frames->get_animation_speed(animation) * speed_scale;
			if (speed == 0) {
				return; //do nothing
			}

			double remaining = get_process_delta_time();

			while (remaining) {
				if (timeout <= 0) {
					timeout = _get_frame_duration();

					int fc = frames->get_frame_count(animation);
					if ((!backwards && frame >= fc - 1) || (backwards && frame <= 0)) {
						if (frames->get_animation_loop(animation)) {
							if (backwards) {
								frame = fc - 1;
							} else {
								frame = 0;
							}

							emit_signal(SceneStringNames::get_singleton()->animation_finished);
						} else {
							if (backwards) {
								frame = 0;
							} else {
								frame = fc - 1;
							}

							if (!is_over) {
								is_over = true;
								emit_signal(SceneStringNames::get_singleton()->animation_finished);
							}
						}
					} else {
						if (backwards) {
							frame--;
						} else {
							frame++;
						}
					}

					update();

					emit_signal(SceneStringNames::get_singleton()->frame_changed);
				}

				double to_process = MIN(timeout, remaining);
				remaining -= to_process;
				timeout -= to_process;
			}
		} break;

		case NOTIFICATION_DRAW: {
			if (frames.is_null()) {
				return;
			}
			if (frame < 0) {
				return;
			}
			if (!frames->has_animation(animation)) {
				return;
			}

			Ref<Texture2D> texture = frames->get_frame(animation, frame);
			if (texture.is_null()) {
				return;
			}

			RID ci = get_canvas_item();

			Size2 s = texture->get_size();
			Point2 ofs = offset;
			if (centered) {
				ofs -= s / 2;
			}

			if (get_viewport() && get_viewport()->is_snap_2d_transforms_to_pixel_enabled()) {
				ofs = ofs.floor();
			}
			Rect2 dst_rect(ofs, s);

			if (hflip) {
				dst_rect.size.x = -dst_rect.size.x;
			}
			if (vflip) {
				dst_rect.size.y = -dst_rect.size.y;
			}

			texture->draw_rect_region(ci, dst_rect, Rect2(Vector2(), texture->get_size()), Color(1, 1, 1), false);

		} break;
	}
}

void AnimatedSprite2D::set_sprite_frames(const Ref<SpriteFrames> &p_frames) {
	if (frames.is_valid()) {
		frames->disconnect("changed", callable_mp(this, &AnimatedSprite2D::_res_changed));
	}
	frames = p_frames;
	if (frames.is_valid()) {
		frames->connect("changed", callable_mp(this, &AnimatedSprite2D::_res_changed));
	}

	if (!frames.is_valid()) {
		frame = 0;
	} else {
		set_frame(frame);
	}

	notify_property_list_changed();
	_reset_timeout();
	update();
	update_configuration_warnings();
}

Ref<SpriteFrames> AnimatedSprite2D::get_sprite_frames() const {
	return frames;
}

void AnimatedSprite2D::set_frame(int p_frame) {
	if (!frames.is_valid()) {
		return;
	}

	if (frames->has_animation(animation)) {
		int limit = frames->get_frame_count(animation);
		if (p_frame >= limit) {
			p_frame = limit - 1;
		}
	}

	if (p_frame < 0) {
		p_frame = 0;
	}

	if (frame == p_frame) {
		return;
	}

	frame = p_frame;
	_reset_timeout();
	update();

	emit_signal(SceneStringNames::get_singleton()->frame_changed);
}

int AnimatedSprite2D::get_frame() const {
	return frame;
}

void AnimatedSprite2D::set_speed_scale(double p_speed_scale) {
	double elapsed = _get_frame_duration() - timeout;

	speed_scale = MAX(p_speed_scale, 0.0f);

	// We adapt the timeout so that the animation speed adapts as soon as the speed scale is changed
	_reset_timeout();
	timeout -= elapsed;
}

double AnimatedSprite2D::get_speed_scale() const {
	return speed_scale;
}

void AnimatedSprite2D::set_centered(bool p_center) {
	centered = p_center;
	update();
	item_rect_changed();
}

bool AnimatedSprite2D::is_centered() const {
	return centered;
}

void AnimatedSprite2D::set_offset(const Point2 &p_offset) {
	offset = p_offset;
	update();
	item_rect_changed();
}

Point2 AnimatedSprite2D::get_offset() const {
	return offset;
}

void AnimatedSprite2D::set_flip_h(bool p_flip) {
	hflip = p_flip;
	update();
}

bool AnimatedSprite2D::is_flipped_h() const {
	return hflip;
}

void AnimatedSprite2D::set_flip_v(bool p_flip) {
	vflip = p_flip;
	update();
}

bool AnimatedSprite2D::is_flipped_v() const {
	return vflip;
}

void AnimatedSprite2D::_res_changed() {
	set_frame(frame);

	update();
}

void AnimatedSprite2D::_set_playing(bool p_playing) {
	if (playing == p_playing) {
		return;
	}
	playing = p_playing;
	_reset_timeout();
	set_process_internal(playing);
}

bool AnimatedSprite2D::_is_playing() const {
	return playing;
}

void AnimatedSprite2D::play(const StringName &p_animation, const bool p_backwards) {
	backwards = p_backwards;

	if (p_animation) {
		set_animation(p_animation);
		if (frames.is_valid() && backwards && get_frame() == 0) {
			set_frame(frames->get_frame_count(p_animation) - 1);
		}
	}

	_set_playing(true);
}

void AnimatedSprite2D::stop() {
	_set_playing(false);
}

bool AnimatedSprite2D::is_playing() const {
	return playing;
}

double AnimatedSprite2D::_get_frame_duration() {
	if (frames.is_valid() && frames->has_animation(animation)) {
		double speed = frames->get_animation_speed(animation) * speed_scale;
		if (speed > 0) {
			return 1.0 / speed;
		}
	}
	return 0.0;
}

void AnimatedSprite2D::_reset_timeout() {
	if (!playing) {
		return;
	}

	timeout = _get_frame_duration();
	is_over = false;
}

void AnimatedSprite2D::set_animation(const StringName &p_animation) {
	ERR_FAIL_COND_MSG(frames == nullptr, vformat("There is no animation with name '%s'.", p_animation));
	ERR_FAIL_COND_MSG(frames->get_animation_names().find(p_animation) == -1, vformat("There is no animation with name '%s'.", p_animation));

	if (animation == p_animation) {
		return;
	}

	animation = p_animation;
	_reset_timeout();
	set_frame(0);
	notify_property_list_changed();
	update();
}

StringName AnimatedSprite2D::get_animation() const {
	return animation;
}

TypedArray<String> AnimatedSprite2D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (frames.is_null()) {
		warnings.push_back(TTR("A SpriteFrames resource must be created or set in the \"Frames\" property in order for AnimatedSprite to display frames."));
	}

	return warnings;
}

void AnimatedSprite2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sprite_frames", "sprite_frames"), &AnimatedSprite2D::set_sprite_frames);
	ClassDB::bind_method(D_METHOD("get_sprite_frames"), &AnimatedSprite2D::get_sprite_frames);

	ClassDB::bind_method(D_METHOD("set_animation", "animation"), &AnimatedSprite2D::set_animation);
	ClassDB::bind_method(D_METHOD("get_animation"), &AnimatedSprite2D::get_animation);

	ClassDB::bind_method(D_METHOD("_set_playing", "playing"), &AnimatedSprite2D::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_playing"), &AnimatedSprite2D::_is_playing);

	ClassDB::bind_method(D_METHOD("play", "anim", "backwards"), &AnimatedSprite2D::play, DEFVAL(StringName()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("stop"), &AnimatedSprite2D::stop);
	ClassDB::bind_method(D_METHOD("is_playing"), &AnimatedSprite2D::is_playing);

	ClassDB::bind_method(D_METHOD("set_centered", "centered"), &AnimatedSprite2D::set_centered);
	ClassDB::bind_method(D_METHOD("is_centered"), &AnimatedSprite2D::is_centered);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &AnimatedSprite2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &AnimatedSprite2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_flip_h", "flip_h"), &AnimatedSprite2D::set_flip_h);
	ClassDB::bind_method(D_METHOD("is_flipped_h"), &AnimatedSprite2D::is_flipped_h);

	ClassDB::bind_method(D_METHOD("set_flip_v", "flip_v"), &AnimatedSprite2D::set_flip_v);
	ClassDB::bind_method(D_METHOD("is_flipped_v"), &AnimatedSprite2D::is_flipped_v);

	ClassDB::bind_method(D_METHOD("set_frame", "frame"), &AnimatedSprite2D::set_frame);
	ClassDB::bind_method(D_METHOD("get_frame"), &AnimatedSprite2D::get_frame);

	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed_scale"), &AnimatedSprite2D::set_speed_scale);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &AnimatedSprite2D::get_speed_scale);

	ADD_SIGNAL(MethodInfo("frame_changed"));
	ADD_SIGNAL(MethodInfo("animation_finished"));

	ADD_GROUP("Animation", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "frames", PROPERTY_HINT_RESOURCE_TYPE, "SpriteFrames"), "set_sprite_frames", "get_sprite_frames");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "animation"), "set_animation", "get_animation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "frame"), "set_frame", "get_frame");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing"), "_set_playing", "_is_playing");
	ADD_GROUP("Offset", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "centered"), "set_centered", "is_centered");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_h"), "set_flip_h", "is_flipped_h");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_v"), "set_flip_v", "is_flipped_v");
}

AnimatedSprite2D::AnimatedSprite2D() {
}
