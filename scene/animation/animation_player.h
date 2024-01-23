/**************************************************************************/
/*  animation_player.h                                                    */
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

#ifndef ANIMATION_PLAYER_H
#define ANIMATION_PLAYER_H

#include "animation_mixer.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/animation.h"

class AnimationPlayer : public AnimationMixer {
	GDCLASS(AnimationPlayer, AnimationMixer);

#ifndef DISABLE_DEPRECATED
public:
	enum AnimationProcessCallback {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
		ANIMATION_PROCESS_MANUAL,
	};
	enum AnimationMethodCallMode {
		ANIMATION_METHOD_CALL_DEFERRED,
		ANIMATION_METHOD_CALL_IMMEDIATE,
	};
#endif // DISABLE_DEPRECATED

private:
	HashMap<StringName, StringName> animation_next_set; // For auto advance.

	float speed_scale = 1.0;
	double default_blend_time = 0.0;
	bool is_stopping = false;

	struct PlaybackData {
		AnimationData *from = nullptr;
		double pos = 0.0;
		float speed_scale = 1.0;
	};

	struct Blend {
		PlaybackData data;
		double blend_time = 0.0;
		double blend_left = 0.0;
	};

	struct Playback {
		PlaybackData current;
		StringName assigned;
		bool seeked = false;
		bool started = false;
		List<Blend> blend;
	} playback;

	struct BlendKey {
		StringName from;
		StringName to;
		static uint32_t hash(const BlendKey &p_key) {
			return hash_one_uint64((uint64_t(p_key.from.hash()) << 32) | uint32_t(p_key.to.hash()));
		}
		bool operator==(const BlendKey &bk) const {
			return from == bk.from && to == bk.to;
		}
		bool operator<(const BlendKey &bk) const {
			if (from == bk.from) {
				return to < bk.to;
			} else {
				return from < bk.from;
			}
		}
	};

	HashMap<BlendKey, double, BlendKey> blend_times;

	List<StringName> playback_queue;
	ObjectID tmp_from;
	bool end_reached = false;
	bool end_notify = false;

	StringName autoplay;

	bool reset_on_save = true;
	bool movie_quit_on_finish = false;

	void _process_playback_data(PlaybackData &cd, double p_delta, float p_blend, bool p_seeked, bool p_started, bool p_is_current = false);
	void _blend_playback_data(double p_delta, bool p_started);
	void _stop_internal(bool p_reset, bool p_keep_state);
	void _check_immediately_after_start();

	float get_current_blend_amount();

	bool playing = false;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	virtual void _validate_property(PropertyInfo &p_property) const override;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

	// Make animation instances.
	virtual bool _blend_pre_process(double p_delta, int p_track_count, const HashMap<NodePath, int> &p_track_map) override;
	virtual void _blend_post_process() override;

	virtual void _animation_removed(const StringName &p_name, const StringName &p_library) override;
	virtual void _rename_animation(const StringName &p_from_name, const StringName &p_to_name) override;

#ifndef DISABLE_DEPRECATED
	void _set_process_callback_bind_compat_80813(AnimationProcessCallback p_mode);
	AnimationProcessCallback _get_process_callback_bind_compat_80813() const;
	void _set_method_call_mode_bind_compat_80813(AnimationMethodCallMode p_mode);
	AnimationMethodCallMode _get_method_call_mode_bind_compat_80813() const;
	void _set_root_bind_compat_80813(const NodePath &p_root);
	NodePath _get_root_bind_compat_80813() const;
	void _seek_bind_compat_80813(double p_time, bool p_update = false);

	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	void animation_set_next(const StringName &p_animation, const StringName &p_next);
	StringName animation_get_next(const StringName &p_animation) const;

	void set_blend_time(const StringName &p_animation1, const StringName &p_animation2, double p_time);
	double get_blend_time(const StringName &p_animation1, const StringName &p_animation2) const;

	void set_default_blend_time(double p_default);
	double get_default_blend_time() const;

	void play(const StringName &p_name = StringName(), double p_custom_blend = -1, float p_custom_scale = 1.0, bool p_from_end = false);
	void play_backwards(const StringName &p_name = StringName(), double p_custom_blend = -1);
	void queue(const StringName &p_name);
	Vector<String> get_queue();
	void clear_queue();
	void pause();
	void stop(bool p_keep_state = false);
	bool is_playing() const;
	String get_current_animation() const;
	void set_current_animation(const String &p_animation);
	String get_assigned_animation() const;
	void set_assigned_animation(const String &p_animation);
	bool is_valid() const;

	void set_speed_scale(float p_speed);
	float get_speed_scale() const;
	float get_playing_speed() const;

	void set_autoplay(const String &p_name);
	String get_autoplay() const;

	void set_movie_quit_on_finish_enabled(bool p_enabled);
	bool is_movie_quit_on_finish_enabled() const;

	void seek(double p_time, bool p_update = false, bool p_update_only = false);

	double get_current_animation_position() const;
	double get_current_animation_length() const;

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;

	virtual void advance(double p_time) override;

	AnimationPlayer();
	~AnimationPlayer();
};

#ifndef DISABLE_DEPRECATED
VARIANT_ENUM_CAST(AnimationPlayer::AnimationProcessCallback);
VARIANT_ENUM_CAST(AnimationPlayer::AnimationMethodCallMode);
#endif // DISABLE_DEPRECATED

#endif // ANIMATION_PLAYER_H
