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

#include "scene/2d/node_2d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/animation.h"
#include "scene/resources/animation_library.h"
#include "scene/resources/audio_stream_polyphonic.h"

#ifdef TOOLS_ENABLED
class AnimatedValuesBackup : public RefCounted {
	GDCLASS(AnimatedValuesBackup, RefCounted);

	struct Entry {
		Object *object = nullptr;
		Vector<StringName> subpath; // Unused if bone
		int bone_idx = -1; // -1 if not a bone
		Variant value;
	};
	Vector<Entry> entries;

	friend class AnimationPlayer;

protected:
	static void _bind_methods();

public:
	void update_skeletons();
	void restore() const;
};
#endif

class AnimationPlayer : public Node {
	GDCLASS(AnimationPlayer, Node);

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

private:
	enum {
		NODE_CACHE_UPDATE_MAX = 1024,
		BLEND_FROM_MAX = 3
	};

	enum SpecialProperty {
		SP_NONE,
		SP_NODE2D_POS,
		SP_NODE2D_ROT,
		SP_NODE2D_SCALE,
	};

	uint32_t setup_pass = 1;

	struct TrackNodeCache {
		NodePath path;
		uint32_t id = 0;
		Ref<Resource> resource;
		Node *node = nullptr;
		Node2D *node_2d = nullptr;
#ifndef _3D_DISABLED
		Node3D *node_3d = nullptr;
		Skeleton3D *skeleton = nullptr;
		MeshInstance3D *node_blend_shape = nullptr;
		int blend_shape_idx = -1;
#endif // _3D_DISABLED
		int bone_idx = -1;
		// accumulated transforms

		bool loc_used = false;
		bool rot_used = false;
		bool scale_used = false;
		Vector3 init_loc = Vector3(0, 0, 0);
		Quaternion init_rot = Quaternion(0, 0, 0, 1);
		Vector3 init_scale = Vector3(1, 1, 1);

		Vector3 loc_accum;
		Quaternion rot_accum;
		Vector3 scale_accum;
		float blend_shape_accum = 0;
		uint64_t accum_pass = 0;

		bool audio_playing = false;
		float audio_start = 0.0;
		float audio_len = 0.0;

		bool animation_playing = false;

		struct PropertyAnim {
			TrackNodeCache *owner = nullptr;
			SpecialProperty special = SP_NONE; //small optimization
			Vector<StringName> subpath;
			Object *object = nullptr;
			Variant value_accum;
			uint64_t accum_pass = 0;
			Variant capture;
		};

		HashMap<StringName, PropertyAnim> property_anim;

		struct BezierAnim {
			Vector<StringName> bezier_property;
			TrackNodeCache *owner = nullptr;
			float bezier_accum = 0.0;
			Object *object = nullptr;
			uint64_t accum_pass = 0;
		};

		HashMap<StringName, BezierAnim> bezier_anim;

		struct PlayingAudioStreamInfo {
			AudioStreamPlaybackPolyphonic::ID index = -1;
			double start = 0.0;
			double len = 0.0;
		};

		struct AudioAnim {
			Ref<AudioStreamPolyphonic> audio_stream;
			Ref<AudioStreamPlaybackPolyphonic> audio_stream_playback;
			HashMap<int, PlayingAudioStreamInfo> playing_streams;
			Object *object = nullptr;
			uint64_t accum_pass = 0;
			double length = 0.0;
			double time = 0.0;
			bool loop = false;
			bool backward = false;
		};

		HashMap<StringName, AudioAnim> audio_anim;

		uint32_t last_setup_pass = 0;
		TrackNodeCache() {}
	};

	struct TrackNodeCacheKey {
		ObjectID id;
		int bone_idx = -1;
		int blend_shape_idx = -1;

		static uint32_t hash(const TrackNodeCacheKey &p_key) {
			uint32_t h = hash_one_uint64(p_key.id);
			h = hash_murmur3_one_32(p_key.bone_idx, h);
			return hash_fmix32(hash_murmur3_one_32(p_key.blend_shape_idx, h));
		}

		inline bool operator==(const TrackNodeCacheKey &p_right) const {
			return id == p_right.id && bone_idx == p_right.bone_idx && blend_shape_idx == p_right.blend_shape_idx;
		}

		inline bool operator<(const TrackNodeCacheKey &p_right) const {
			if (id == p_right.id) {
				if (blend_shape_idx == p_right.blend_shape_idx) {
					return bone_idx < p_right.bone_idx;
				} else {
					return blend_shape_idx < p_right.blend_shape_idx;
				}
			} else {
				return id < p_right.id;
			}
		}
	};

	HashMap<TrackNodeCacheKey, TrackNodeCache, TrackNodeCacheKey> node_cache_map;

	TrackNodeCache *cache_update[NODE_CACHE_UPDATE_MAX];
	int cache_update_size = 0;
	TrackNodeCache::PropertyAnim *cache_update_prop[NODE_CACHE_UPDATE_MAX];
	int cache_update_prop_size = 0;
	TrackNodeCache::BezierAnim *cache_update_bezier[NODE_CACHE_UPDATE_MAX];
	int cache_update_bezier_size = 0;
	TrackNodeCache::AudioAnim *cache_update_audio[NODE_CACHE_UPDATE_MAX];
	int cache_update_audio_size = 0;
	HashSet<TrackNodeCache *> playing_caches;
	Vector<Node *> playing_audio_stream_players;

	uint64_t accum_pass = 1;
	float speed_scale = 1.0;
	double default_blend_time = 0.0;
	bool is_stopping = false;

	struct AnimationData {
		String name;
		StringName next;
		Vector<TrackNodeCache *> node_cache;
		Ref<Animation> animation;
		StringName animation_library;
		uint64_t last_update = 0;
	};

	HashMap<StringName, AnimationData> animation_set;

	struct AnimationLibraryData {
		StringName name;
		Ref<AnimationLibrary> library;
		bool operator<(const AnimationLibraryData &p_data) const { return name.operator String() < p_data.name.operator String(); }
	};

	LocalVector<AnimationLibraryData> animation_libraries;

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
		List<Blend> blend;
		PlaybackData current;
		StringName assigned;
		bool seeked = false;
		bool started = false;
	} playback;

	List<StringName> queued;

	bool end_reached = false;
	bool end_notify = false;

	String autoplay;
	bool reset_on_save = true;
	AnimationProcessCallback process_callback = ANIMATION_PROCESS_IDLE;
	AnimationMethodCallMode method_call_mode = ANIMATION_METHOD_CALL_DEFERRED;
	int audio_max_polyphony = 32;
	bool movie_quit_on_finish = false;
	bool processing = false;
	bool active = true;

	NodePath root;

	void _animation_process_animation(AnimationData *p_anim, double p_prev_time, double p_time, double p_delta, float p_interp, bool p_is_current = true, bool p_seeked = false, bool p_started = false, Animation::LoopedFlag p_looped_flag = Animation::LOOPED_FLAG_NONE);

	void _ensure_node_caches(AnimationData *p_anim, Node *p_root_override = nullptr);
	void _animation_process_data(PlaybackData &cd, double p_delta, float p_blend, bool p_seeked, bool p_started);
	void _animation_process2(double p_delta, bool p_started);
	void _animation_update_transforms();
	void _animation_process(double p_delta);

	void _node_removed(Node *p_node);
	void _clear_audio_streams();
	void _stop_playing_caches(bool p_reset);

	// bind helpers
	Vector<String> _get_animation_list() const {
		List<StringName> animations;
		get_animation_list(&animations);
		Vector<String> ret;
		while (animations.size()) {
			ret.push_back(animations.front()->get());
			animations.pop_front();
		}
		return ret;
	}

	void _animation_changed(const StringName &p_name);

	void _set_process(bool p_process, bool p_force = false);
	void _stop_internal(bool p_reset, bool p_keep_state);

	bool playing = false;

	uint64_t animation_set_update_pass = 1;
	void _animation_set_cache_update();
	void _animation_added(const StringName &p_name, const StringName &p_library);
	void _animation_removed(const StringName &p_name, const StringName &p_library);
	void _animation_renamed(const StringName &p_name, const StringName &p_to_name, const StringName &p_library);
	void _rename_animation(const StringName &p_from_name, const StringName &p_to_name);

	TypedArray<StringName> _get_animation_library_list() const;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _validate_property(PropertyInfo &p_property) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

	GDVIRTUAL5RC(Variant, _post_process_key_value, Ref<Animation>, int, Variant, Object *, int);
	Variant post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx = -1);
	virtual Variant _post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx = -1);

public:
	StringName find_animation(const Ref<Animation> &p_animation) const;
	StringName find_animation_library(const Ref<Animation> &p_animation) const;

	Error add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_animation_library);
	void remove_animation_library(const StringName &p_name);
	void rename_animation_library(const StringName &p_name, const StringName &p_new_name);
	Ref<AnimationLibrary> get_animation_library(const StringName &p_name) const;
	void get_animation_library_list(List<StringName> *p_animations) const;
	bool has_animation_library(const StringName &p_name) const;

	Ref<Animation> get_animation(const StringName &p_name) const;
	void get_animation_list(List<StringName> *p_animations) const;
	bool has_animation(const StringName &p_name) const;

	void set_blend_time(const StringName &p_animation1, const StringName &p_animation2, double p_time);
	double get_blend_time(const StringName &p_animation1, const StringName &p_animation2) const;

	void animation_set_next(const StringName &p_animation, const StringName &p_next);
	StringName animation_get_next(const StringName &p_animation) const;

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
	void set_current_animation(const String &p_anim);
	String get_assigned_animation() const;
	void set_assigned_animation(const String &p_anim);
	void set_active(bool p_active);
	bool is_active() const;
	bool is_valid() const;

	void set_speed_scale(float p_speed);
	float get_speed_scale() const;
	float get_playing_speed() const;

	void set_autoplay(const String &p_name);
	String get_autoplay() const;

	void set_reset_on_save_enabled(bool p_enabled);
	bool is_reset_on_save_enabled() const;

	void set_process_callback(AnimationProcessCallback p_mode);
	AnimationProcessCallback get_process_callback() const;

	void set_method_call_mode(AnimationMethodCallMode p_mode);
	AnimationMethodCallMode get_method_call_mode() const;

	void set_audio_max_polyphony(int p_audio_max_polyphony);
	int get_audio_max_polyphony() const;

	void set_movie_quit_on_finish_enabled(bool p_enabled);
	bool is_movie_quit_on_finish_enabled() const;

	void seek(double p_time, bool p_update = false);
	void seek_delta(double p_time, double p_delta);
	double get_current_animation_position() const;
	double get_current_animation_length() const;

	void advance(double p_time);

	void set_root(const NodePath &p_root);
	NodePath get_root() const;

	void clear_caches(); ///< must be called by hand if an animation was modified after added

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;

#ifdef TOOLS_ENABLED
	Ref<AnimatedValuesBackup> backup_animated_values(Node *p_root_override = nullptr);
	Ref<AnimatedValuesBackup> apply_reset(bool p_user_initiated = false);
	bool can_apply_reset() const;
#endif

	AnimationPlayer();
	~AnimationPlayer();
};

VARIANT_ENUM_CAST(AnimationPlayer::AnimationProcessCallback);
VARIANT_ENUM_CAST(AnimationPlayer::AnimationMethodCallMode);

#endif // ANIMATION_PLAYER_H
