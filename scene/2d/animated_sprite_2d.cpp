/**************************************************************************/
/*  animated_sprite_2d.cpp                                                */
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
	if (frames.is_null() || !frames->has_animation(animation)) {
		return false;
	}
	if (frame < 0 || frame >= frames->get_frame_count(animation)) {
		return false;
	}

	Ref<Texture2D> t;
	if (animation) {
		t = frames->get_frame_texture(animation, frame);
	}
	return t.is_valid();
}
#endif

Rect2 AnimatedSprite2D::get_anchorable_rect() const {
	return _get_rect();
}

Rect2 AnimatedSprite2D::_get_rect() const {
	if (frames.is_null() || !frames->has_animation(animation)) {
		return Rect2();
	}
	if (frame < 0 || frame >= frames->get_frame_count(animation)) {
		return Rect2();
	}

	Ref<Texture2D> t;
	if (animation) {
		t = frames->get_frame_texture(animation, frame);
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

void AnimatedSprite2D::_validate_property(PropertyInfo &p_property) const {
	if (!frames.is_valid()) {
		return;
	}

	if (p_property.name == "animation") {
		p_property.hint = PROPERTY_HINT_ENUM;
		List<StringName> names;
		frames->get_animation_list(&names);
		names.sort_custom<StringName::AlphCompare>();

		bool current_found = false;
		bool is_first_element = true;

		for (const StringName &E : names) {
			if (!is_first_element) {
				p_property.hint_string += ",";
			} else {
				is_first_element = false;
			}

			p_property.hint_string += String(E);
			if (animation == E) {
				current_found = true;
			}
		}

		if (!current_found) {
			if (p_property.hint_string.is_empty()) {
				p_property.hint_string = String(animation);
			} else {
				p_property.hint_string = String(animation) + "," + p_property.hint_string;
			}
		}
		return;
	}

	if (p_property.name == "frame") {
		if (playing) {
			p_property.usage = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY;
			return;
		}

		p_property.hint = PROPERTY_HINT_RANGE;
		if (frames->has_animation(animation) && frames->get_frame_count(animation) > 0) {
			p_property.hint_string = "0," + itos(frames->get_frame_count(animation) - 1) + ",1";
		} else {
			// Avoid an error, `hint_string` is required for `PROPERTY_HINT_RANGE`.
			p_property.hint_string = "0,0,1";
		}
		p_property.usage |= PROPERTY_USAGE_KEYING_INCREMENTS;
	}
}

void AnimatedSprite2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (frames.is_null() || !frames->has_animation(animation)) {
				return;
			}

			double remaining = get_process_delta_time();
			int i = 0;
			while (remaining) {
				// Animation speed may be changed by animation_finished or frame_changed signals.
				double speed = frames->get_animation_speed(animation) * Math::abs(speed_scale);

				if (speed == 0) {
					return; // Do nothing.
				}

				// Frame count may be changed by animation_finished or frame_changed signals.
				int fc = frames->get_frame_count(animation);

				if (timeout <= 0) {
					int last_frame = fc - 1;
					if (!playing_backwards) {
						// Forward.
						if (frame >= last_frame) {
							if (frames->get_animation_loop(animation)) {
								frame = 0;
								emit_signal(SceneStringNames::get_singleton()->animation_finished);
							} else {
								frame = last_frame;
								if (!is_over) {
									is_over = true;
									emit_signal(SceneStringNames::get_singleton()->animation_finished);
								}
							}
						} else {
							frame++;
						}
					} else {
						// Reversed.
						if (frame <= 0) {
							if (frames->get_animation_loop(animation)) {
								frame = last_frame;
								emit_signal(SceneStringNames::get_singleton()->animation_finished);
							} else {
								frame = 0;
								if (!is_over) {
									is_over = true;
									emit_signal(SceneStringNames::get_singleton()->animation_finished);
								}
							}
						} else {
							frame--;
						}
					}

					timeout = _get_frame_duration();

					queue_redraw();

					emit_signal(SceneStringNames::get_singleton()->frame_changed);
				}

				double to_process = MIN(timeout / speed, remaining);
				timeout -= to_process * speed;
				remaining -= to_process;

				i++;
				if (i > fc) {
					return; // Prevents freezing if to_process is each time much less than remaining.
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
			if (frames.is_null() || !frames->has_animation(animation)) {
				return;
			}

			Ref<Texture2D> texture = frames->get_frame_texture(animation, frame);
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
		frames->disconnect(SceneStringNames::get_singleton()->changed, callable_mp(this, &AnimatedSprite2D::_res_changed));
	}

	frames = p_frames;
	if (frames.is_valid()) {
		frames->connect(SceneStringNames::get_singleton()->changed, callable_mp(this, &AnimatedSprite2D::_res_changed));
	}

	if (frames.is_null()) {
		frame = 0;
	} else {
		set_frame(frame);
	}

	notify_property_list_changed();
	_reset_timeout();
	queue_redraw();
	update_configuration_warnings();
}

Ref<SpriteFrames> AnimatedSprite2D::get_sprite_frames() const {
	return frames;
}

void AnimatedSprite2D::set_frame(int p_frame) {
	if (frames.is_null()) {
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
	queue_redraw();
	emit_signal(SceneStringNames::get_singleton()->frame_changed);
}

int AnimatedSprite2D::get_frame() const {
	return frame;
}

void AnimatedSprite2D::set_speed_scale(float p_speed_scale) {
	speed_scale = p_speed_scale;
	playing_backwards = signbit(speed_scale) != backwards;
}

float AnimatedSprite2D::get_speed_scale() const {
	return speed_scale;
}

void AnimatedSprite2D::set_centered(bool p_center) {
	centered = p_center;
	queue_redraw();
	item_rect_changed();
}

bool AnimatedSprite2D::is_centered() const {
	return centered;
}

void AnimatedSprite2D::set_offset(const Point2 &p_offset) {
	offset = p_offset;
	queue_redraw();
	item_rect_changed();
}

Point2 AnimatedSprite2D::get_offset() const {
	return offset;
}

void AnimatedSprite2D::set_flip_h(bool p_flip) {
	hflip = p_flip;
	queue_redraw();
}

bool AnimatedSprite2D::is_flipped_h() const {
	return hflip;
}

void AnimatedSprite2D::set_flip_v(bool p_flip) {
	vflip = p_flip;
	queue_redraw();
}

bool AnimatedSprite2D::is_flipped_v() const {
	return vflip;
}

void AnimatedSprite2D::_res_changed() {
	set_frame(frame);
	queue_redraw();
	notify_property_list_changed();
}

void AnimatedSprite2D::set_playing(bool p_playing) {
	if (playing == p_playing) {
		return;
	}
	playing = p_playing;
	playing_backwards = signbit(speed_scale) != backwards;
	set_process_internal(playing);
	notify_property_list_changed();
}

bool AnimatedSprite2D::is_playing() const {
	return playing;
}

void AnimatedSprite2D::play(const StringName &p_animation, bool p_backwards) {
	backwards = p_backwards;
	playing_backwards = signbit(speed_scale) != backwards;

	if (p_animation) {
		set_animation(p_animation);
		if (frames.is_valid() && playing_backwards && get_frame() == 0) {
			set_frame(frames->get_frame_count(p_animation) - 1);
		}
	}

	is_over = false;
	set_playing(true);
}

void AnimatedSprite2D::stop() {
	set_playing(false);
	backwards = false;
	_reset_timeout();
}

double AnimatedSprite2D::_get_frame_duration() {
	if (frames.is_valid() && frames->has_animation(animation)) {
		return frames->get_frame_duration(animation, frame);
	}
	return 0.0;
}

void AnimatedSprite2D::_reset_timeout() {
	timeout = _get_frame_duration();
	is_over = false;
}

void AnimatedSprite2D::set_animation(const StringName &p_animation) {
	ERR_FAIL_COND_MSG(frames == nullptr, vformat("There is no animation with name '%s'.", p_animation));
	ERR_FAIL_COND_MSG(!frames->get_animation_names().has(p_animation), vformat("There is no animation with name '%s'.", p_animation));

	if (animation == p_animation) {
		return;
	}

	animation = p_animation;
	set_frame(0);
	_reset_timeout();
	notify_property_list_changed();
	queue_redraw();
}

StringName AnimatedSprite2D::get_animation() const {
	return animation;
}

PackedStringArray AnimatedSprite2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node2D::get_configuration_warnings();
	if (frames.is_null()) {
		warnings.push_back(RTR("A SpriteFrames resource must be created or set in the \"Frames\" property in order for AnimatedSprite2D to display frames."));
	}
	return warnings;
}

void AnimatedSprite2D::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	if (p_idx == 0 && p_function == "play" && frames.is_valid()) {
		List<StringName> al;
		frames->get_animation_list(&al);
		for (const StringName &name : al) {
			r_options->push_back(String(name).quote());
		}
	}
	Node::get_argument_options(p_function, p_idx, r_options);
}

void AnimatedSprite2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sprite_frames", "sprite_frames"), &AnimatedSprite2D::set_sprite_frames);
	ClassDB::bind_method(D_METHOD("get_sprite_frames"), &AnimatedSprite2D::get_sprite_frames);

	ClassDB::bind_method(D_METHOD("set_animation", "animation"), &AnimatedSprite2D::set_animation);
	ClassDB::bind_method(D_METHOD("get_animation"), &AnimatedSprite2D::get_animation);

	ClassDB::bind_method(D_METHOD("set_playing", "playing"), &AnimatedSprite2D::set_playing);
	ClassDB::bind_method(D_METHOD("is_playing"), &AnimatedSprite2D::is_playing);

	ClassDB::bind_method(D_METHOD("play", "anim", "backwards"), &AnimatedSprite2D::play, DEFVAL(StringName()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("stop"), &AnimatedSprite2D::stop);

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
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing"), "set_playing", "is_playing");
	ADD_GROUP("Offset", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "centered"), "set_centered", "is_centered");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset", PROPERTY_HINT_NONE, "suffix:px"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_h"), "set_flip_h", "is_flipped_h");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flip_v"), "set_flip_v", "is_flipped_v");
}

AnimatedSprite2D::AnimatedSprite2D() {
}
