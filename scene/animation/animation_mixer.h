/**************************************************************************/
/*  animation_mixer.h                                                     */
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

#ifndef ANIMATION_MIXER_H
#define ANIMATION_MIXER_H

#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/animation.h"
#include "scene/resources/animation_library.h"
#include "scene/resources/audio_stream_polyphonic.h"

class AnimatedValuesBackup;

class AnimationMixer : public Node {
	GDCLASS(AnimationMixer, Node);
	friend AnimatedValuesBackup;
#ifdef TOOLS_ENABLED
	bool editing = false;
	bool dummy = false;
#endif // TOOLS_ENABLED

	bool reset_on_save = true;

public:
	enum AnimationCallbackModeProcess {
		ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS,
		ANIMATION_CALLBACK_MODE_PROCESS_IDLE,
		ANIMATION_CALLBACK_MODE_PROCESS_MANUAL,
	};

	enum AnimationCallbackModeMethod {
		ANIMATION_CALLBACK_MODE_METHOD_DEFERRED,
		ANIMATION_CALLBACK_MODE_METHOD_IMMEDIATE,
	};

	/* ---- Data ---- */
	struct AnimationLibraryData {
		StringName name;
		Ref<AnimationLibrary> library;
		bool operator<(const AnimationLibraryData &p_data) const { return name.operator String() < p_data.name.operator String(); }
	};

	struct AnimationData {
		String name;
		Ref<Animation> animation;
		StringName animation_library;
		uint64_t last_update = 0;
	};

	struct PlaybackInfo {
		double time = 0.0;
		double delta = 0.0;
		bool seeked = false;
		bool is_external_seeking = false;
		Animation::LoopedFlag looped_flag = Animation::LOOPED_FLAG_NONE;
		real_t weight = 0.0;
		Vector<real_t> track_weights;
	};

	struct AnimationInstance {
		AnimationData animation_data;
		PlaybackInfo playback_info;
	};

protected:
	/* ---- Data lists ---- */
	LocalVector<AnimationLibraryData> animation_libraries;
	HashMap<StringName, AnimationData> animation_set; // HashMap<Library name + Animation name, AnimationData>

	TypedArray<StringName> _get_animation_library_list() const;
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

	// For caches.
	uint64_t animation_set_update_pass = 1;
	void _animation_set_cache_update();

	// Signals.
	virtual void _animation_added(const StringName &p_name, const StringName &p_library);
	virtual void _animation_removed(const StringName &p_name, const StringName &p_library);
	virtual void _animation_renamed(const StringName &p_name, const StringName &p_to_name, const StringName &p_library);
	virtual void _animation_changed(const StringName &p_name);

	/* ---- General settings for animation ---- */
	AnimationCallbackModeProcess callback_mode_process = ANIMATION_CALLBACK_MODE_PROCESS_IDLE;
	AnimationCallbackModeMethod callback_mode_method = ANIMATION_CALLBACK_MODE_METHOD_DEFERRED;
	int audio_max_polyphony = 32;
	NodePath root_node;

	bool processing = false;
	bool active = true;

	void _set_process(bool p_process, bool p_force = false);

	/* ---- Caches for blending ---- */
	bool cache_valid = false;
	uint64_t setup_pass = 1;
	uint64_t process_pass = 1;

	struct TrackCache {
		bool root_motion = false;
		uint64_t setup_pass = 0;
		Animation::TrackType type = Animation::TrackType::TYPE_ANIMATION;
		Object *object = nullptr;
		ObjectID object_id;
		real_t total_weight = 0.0;

		TrackCache() = default;
		TrackCache(const TrackCache &p_other) :
				root_motion(p_other.root_motion),
				setup_pass(p_other.setup_pass),
				type(p_other.type),
				object(p_other.object),
				object_id(p_other.object_id),
				total_weight(p_other.total_weight) {}

		virtual ~TrackCache() {}
	};

	struct TrackCacheTransform : public TrackCache {
#ifndef _3D_DISABLED
		Node3D *node_3d = nullptr;
		Skeleton3D *skeleton = nullptr;
#endif // _3D_DISABLED
		int bone_idx = -1;
		bool loc_used = false;
		bool rot_used = false;
		bool scale_used = false;
		Vector3 init_loc = Vector3(0, 0, 0);
		Quaternion init_rot = Quaternion(0, 0, 0, 1);
		Vector3 init_scale = Vector3(1, 1, 1);
		Vector3 loc;
		Quaternion rot;
		Vector3 scale;

		TrackCacheTransform(const TrackCacheTransform &p_other) :
				TrackCache(p_other),
#ifndef _3D_DISABLED
				node_3d(p_other.node_3d),
				skeleton(p_other.skeleton),
#endif
				bone_idx(p_other.bone_idx),
				loc_used(p_other.loc_used),
				rot_used(p_other.rot_used),
				scale_used(p_other.scale_used),
				init_loc(p_other.init_loc),
				init_rot(p_other.init_rot),
				init_scale(p_other.init_scale),
				loc(p_other.loc),
				rot(p_other.rot),
				scale(p_other.scale) {
		}

		TrackCacheTransform() {
			type = Animation::TYPE_POSITION_3D;
		}
		~TrackCacheTransform() {}
	};

	struct RootMotionCache {
		Vector3 loc = Vector3(0, 0, 0);
		Quaternion rot = Quaternion(0, 0, 0, 1);
		Vector3 scale = Vector3(1, 1, 1);
	};

	struct TrackCacheBlendShape : public TrackCache {
		MeshInstance3D *mesh_3d = nullptr;
		float init_value = 0;
		float value = 0;
		int shape_index = -1;

		TrackCacheBlendShape(const TrackCacheBlendShape &p_other) :
				TrackCache(p_other),
				mesh_3d(p_other.mesh_3d),
				init_value(p_other.init_value),
				value(p_other.value),
				shape_index(p_other.shape_index) {}

		TrackCacheBlendShape() { type = Animation::TYPE_BLEND_SHAPE; }
		~TrackCacheBlendShape() {}
	};

	struct TrackCacheValue : public TrackCache {
		Variant init_value;
		Variant value;
		Vector<StringName> subpath;
		bool is_continuous = false;
		bool is_using_angle = false;
		Variant element_size;

		TrackCacheValue(const TrackCacheValue &p_other) :
				TrackCache(p_other),
				init_value(p_other.init_value),
				value(p_other.value),
				subpath(p_other.subpath),
				is_continuous(p_other.is_continuous),
				is_using_angle(p_other.is_using_angle),
				element_size(p_other.element_size) {}

		TrackCacheValue() { type = Animation::TYPE_VALUE; }
		~TrackCacheValue() {
			// Clear ref to avoid leaking.
			init_value = Variant();
			value = Variant();
		}
	};

	struct TrackCacheMethod : public TrackCache {
		TrackCacheMethod() { type = Animation::TYPE_METHOD; }
		~TrackCacheMethod() {}
	};

	struct TrackCacheBezier : public TrackCache {
		real_t init_value = 0.0;
		real_t value = 0.0;
		Vector<StringName> subpath;

		TrackCacheBezier(const TrackCacheBezier &p_other) :
				TrackCache(p_other),
				init_value(p_other.init_value),
				value(p_other.value),
				subpath(p_other.subpath) {}

		TrackCacheBezier() {
			type = Animation::TYPE_BEZIER;
		}
		~TrackCacheBezier() {}
	};

	// Audio stream information for each audio stream placed on the track.
	struct PlayingAudioStreamInfo {
		AudioStreamPlaybackPolyphonic::ID index = -1; // ID retrieved from AudioStreamPlaybackPolyphonic.
		double start = 0.0;
		double len = 0.0;
	};

	// Audio track information for mixng and ending.
	struct PlayingAudioTrackInfo {
		HashMap<int, PlayingAudioStreamInfo> stream_info;
		double length = 0.0;
		double time = 0.0;
		real_t volume = 0.0;
		bool loop = false;
		bool backward = false;
		bool use_blend = false;
	};

	struct TrackCacheAudio : public TrackCache {
		Ref<AudioStreamPolyphonic> audio_stream;
		Ref<AudioStreamPlaybackPolyphonic> audio_stream_playback;
		HashMap<ObjectID, PlayingAudioTrackInfo> playing_streams; // Key is Animation resource ObjectID.

		TrackCacheAudio(const TrackCacheAudio &p_other) :
				TrackCache(p_other),
				audio_stream(p_other.audio_stream),
				audio_stream_playback(p_other.audio_stream_playback),
				playing_streams(p_other.playing_streams) {}

		TrackCacheAudio() {
			type = Animation::TYPE_AUDIO;
		}
		~TrackCacheAudio() {}
	};

	struct TrackCacheAnimation : public TrackCache {
		bool playing = false;

		TrackCacheAnimation() {
			type = Animation::TYPE_ANIMATION;
		}
		~TrackCacheAnimation() {}
	};

	RootMotionCache root_motion_cache;
	HashMap<NodePath, TrackCache *> track_cache;
	HashSet<TrackCache *> playing_caches;
	Vector<Node *> playing_audio_stream_players;

	// Helpers.
	void _clear_caches();
	void _clear_audio_streams();
	void _clear_playing_caches();
	void _init_root_motion_cache();
	bool _update_caches();

	/* ---- Blending processor ---- */
	LocalVector<AnimationInstance> animation_instances;
	HashMap<NodePath, int> track_map;
	int track_count = 0;
	bool deterministic = false;

	/* ---- Root motion accumulator for Skeleton3D ---- */
	NodePath root_motion_track;
	Vector3 root_motion_position = Vector3(0, 0, 0);
	Quaternion root_motion_rotation = Quaternion(0, 0, 0, 1);
	Vector3 root_motion_scale = Vector3(0, 0, 0);
	Vector3 root_motion_position_accumulator = Vector3(0, 0, 0);
	Quaternion root_motion_rotation_accumulator = Quaternion(0, 0, 0, 1);
	Vector3 root_motion_scale_accumulator = Vector3(1, 1, 1);

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &p_property) const;

	static void _bind_methods();
	void _node_removed(Node *p_node);

	// Helper for extended class.
	virtual void _set_active(bool p_active);
	virtual void _remove_animation(const StringName &p_name);
	virtual void _rename_animation(const StringName &p_from_name, const StringName &p_to_name);

	/* ---- Blending processor ---- */
	virtual void _process_animation(double p_delta, bool p_update_only = false);
	virtual Variant _post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx = -1);
	Variant post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx = -1);
	GDVIRTUAL5RC(Variant, _post_process_key_value, Ref<Animation>, int, Variant, Object *, int);

	void _blend_init();
	virtual bool _blend_pre_process(double p_delta, int p_track_count, const HashMap<NodePath, int> &p_track_map);
	void _blend_calc_total_weight(); // For undeterministic blending.
	void _blend_process(double p_delta, bool p_update_only = false);
	void _blend_apply();
	virtual void _blend_post_process();
	void _call_object(Object *p_object, const StringName &p_method, const Vector<Variant> &p_params, bool p_deferred);

public:
	/* ---- Data lists ---- */
	Dictionary *get_animation_libraries();

	void get_animation_library_list(List<StringName> *p_animations) const;
	Ref<AnimationLibrary> get_animation_library(const StringName &p_name) const;
	bool has_animation_library(const StringName &p_name) const;
	StringName find_animation_library(const Ref<Animation> &p_animation) const;
	Error add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_animation_library);
	void remove_animation_library(const StringName &p_name);
	void rename_animation_library(const StringName &p_name, const StringName &p_new_name);

	void get_animation_list(List<StringName> *p_animations) const;
	Ref<Animation> get_animation(const StringName &p_name) const;
	bool has_animation(const StringName &p_name) const;
	StringName find_animation(const Ref<Animation> &p_animation) const;

	/* ---- General settings for animation ---- */
	void set_active(bool p_active);
	bool is_active() const;

	void set_deterministic(bool p_deterministic);
	bool is_deterministic() const;

	void set_root_node(const NodePath &p_path);
	NodePath get_root_node() const;

	void set_callback_mode_process(AnimationCallbackModeProcess p_mode);
	AnimationCallbackModeProcess get_callback_mode_process() const;

	void set_callback_mode_method(AnimationCallbackModeMethod p_mode);
	AnimationCallbackModeMethod get_callback_mode_method() const;

	void set_audio_max_polyphony(int p_audio_max_polyphony);
	int get_audio_max_polyphony() const;

	/* ---- Root motion accumulator for Skeleton3D ---- */
	void set_root_motion_track(const NodePath &p_track);
	NodePath get_root_motion_track() const;

	Vector3 get_root_motion_position() const;
	Quaternion get_root_motion_rotation() const;
	Vector3 get_root_motion_scale() const;

	Vector3 get_root_motion_position_accumulator() const;
	Quaternion get_root_motion_rotation_accumulator() const;
	Vector3 get_root_motion_scale_accumulator() const;

	/* ---- Blending processor ---- */
	void make_animation_instance(const StringName &p_name, const PlaybackInfo p_playback_info);
	void clear_animation_instances();
	virtual void advance(double p_time);
	virtual void clear_caches(); ///< must be called by hand if an animation was modified after added

	void set_reset_on_save_enabled(bool p_enabled);
	bool is_reset_on_save_enabled() const;

	bool can_apply_reset() const;
	void _build_backup_track_cache();
	Ref<AnimatedValuesBackup> make_backup();
	void restore(const Ref<AnimatedValuesBackup> &p_backup);
	void reset();

#ifdef TOOLS_ENABLED
	Ref<AnimatedValuesBackup> apply_reset(bool p_user_initiated = false);

	void set_editing(bool p_editing);
	bool is_editing() const;

	void set_dummy(bool p_dummy);
	bool is_dummy() const;
#endif // TOOLS_ENABLED

	AnimationMixer();
	~AnimationMixer();
};

class AnimatedValuesBackup : public RefCounted {
	GDCLASS(AnimatedValuesBackup, RefCounted);

	HashMap<NodePath, AnimationMixer::TrackCache *> data;

public:
	void set_data(const HashMap<NodePath, AnimationMixer::TrackCache *> p_data);
	HashMap<NodePath, AnimationMixer::TrackCache *> get_data() const;
	void clear_data();

	AnimationMixer::TrackCache *get_cache_copy(AnimationMixer::TrackCache *p_cache) const;

	~AnimatedValuesBackup() { clear_data(); }
};

VARIANT_ENUM_CAST(AnimationMixer::AnimationCallbackModeProcess);
VARIANT_ENUM_CAST(AnimationMixer::AnimationCallbackModeMethod);

#endif // ANIMATION_MIXER_H
