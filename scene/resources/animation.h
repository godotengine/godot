/*************************************************************************/
/*  animation.h                                                          */
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

#ifndef ANIMATION_H
#define ANIMATION_H

#include "core/io/resource.h"
#include "core/templates/local_vector.h"

#define ANIM_MIN_LENGTH 0.001

class Animation : public Resource {
	GDCLASS(Animation, Resource);
	RES_BASE_EXTENSION("anim");

public:
	enum TrackType {
		TYPE_VALUE, ///< Set a value in a property, can be interpolated.
		TYPE_POSITION_3D, ///< Position 3D track
		TYPE_ROTATION_3D, ///< Rotation 3D track
		TYPE_SCALE_3D, ///< Scale 3D track
		TYPE_BLEND_SHAPE, ///< Blend Shape track
		TYPE_METHOD, ///< Call any method on a specific node.
		TYPE_BEZIER, ///< Bezier curve
		TYPE_AUDIO,
		TYPE_ANIMATION,
	};

	enum InterpolationType {
		INTERPOLATION_NEAREST,
		INTERPOLATION_LINEAR,
		INTERPOLATION_CUBIC
	};

	enum UpdateMode {
		UPDATE_CONTINUOUS,
		UPDATE_DISCRETE,
		UPDATE_TRIGGER,
		UPDATE_CAPTURE,
	};

	enum LoopMode {
		LOOP_NONE,
		LOOP_LINEAR,
		LOOP_PINGPONG,
	};

	enum HandleMode {
		HANDLE_MODE_FREE,
		HANDLE_MODE_BALANCED,
	};

private:
	struct Key {
		real_t transition = 1.0;
		double time = 0.0; // time in secs
	};

	class BaseTrack {
	public:
		virtual ~BaseTrack() = default;
		TrackType type = TrackType::TYPE_ANIMATION;
		InterpolationType interpolation = INTERPOLATION_LINEAR;
		bool loop_wrap = true;
		NodePath path; // path to something
		bool imported = false;
		bool enabled = true;
		int32_t compressed_track = -1;
		virtual Key *write_key(int index) = 0;
		virtual int get_values_size() = 0;
		virtual void remove_key_at(int index) = 0;
		virtual Key get_key(int index) = 0;
		virtual int find_value(double p_time) const = 0;
		virtual void find_value_in_interval(double from_time, double to_time, List<int> *p_indices_out) = 0;
		virtual void clear() = 0;
		virtual int insert_key(double p_time, Key *p_value) = 0;
		virtual void set_key_time(int p_key_idx, double p_time) = 0;
	};

	template <class T>
	struct TKey : public Key {
		T value;
	};

	// Templated Track
	template <class T>
	class Track : public BaseTrack {
	public:
		Vector<TKey<T>> values;
		Key *write_key(int index) override {
			return static_cast<Key *>(&values.write[index]);
		}
		Key get_key(int index) override {
			return static_cast<Key>(values.write[index]);
		}
		int get_values_size() override {
			return values.size();
		}
		void remove_key_at(int index) override {
			values.remove_at(index);
		}
		int insert_key(double p_time, Key *p_value) override {
			int idx = values.size();
			TKey<T> value = *static_cast<TKey<T> *>(p_value);
			while (true) {
				// Condition for replacement.
				if (idx > 0 && Math::is_equal_approx((double)values[idx - 1].time, p_time)) {
					float transition = values[idx - 1].transition;
					values.write[idx - 1] = value;
					values.write[idx - 1].transition = transition;
					return idx - 1;

					// Condition for insert.
				} else if (idx == 0 || values[idx - 1].time < p_time) {
					values.insert(idx, value);
					return idx;
				}

				idx--;
			}

			return -1;
		};
		void set_key_time(int p_key_idx, double p_time) override {
			ERR_FAIL_COND(compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, values.size());
			TKey<T> key = values[p_key_idx];
			key.time = p_time;
			values.remove_at(p_key_idx);
			insert_key(p_time, &key);
		}
		int find_value(double p_time) const override {
			int len = values.size();
			if (len == 0) {
				return -2;
			}

			int low = 0;
			int high = len - 1;
			int middle = 0;

#ifdef DEBUG_ENABLED
			if (low > high) {
				ERR_PRINT("low > high, this may be a bug");
			}
#endif

			const TKey<T> *keys = &values[0];

			while (low <= high) {
				middle = (low + high) / 2;

				if (Math::is_equal_approx(p_time, (double)keys[middle].time)) { //match
					return middle;
				} else if (p_time < keys[middle].time) {
					high = middle - 1; //search low end of array
				} else {
					low = middle + 1; //search high end of array
				}
			}
			if (keys[middle].time > p_time) {
				middle--;
			}

			return middle;
		};
		void find_value_in_interval(double from_time, double to_time, List<int> *p_indices_out) override {
			int to = find_value(to_time);

			// can't really send the events == time, will be sent in the next frame.
			// if event>=len then it will probably never be requested by the anim player.

			if (to >= 0 && get_key(to).time >= to_time) {
				to--;
			}

			if (to < 0) {
				return; // not bother
			}

			int from = find_value(from_time);

			// position in the right first event.+
			if (from < 0 || get_key(to).time < from_time) {
				from++;
			}

			int max = get_values_size();

			for (int i = from; i <= to; i++) {
				ERR_CONTINUE(i < 0 || i >= max); // shouldn't happen
				p_indices_out->push_back(i);
			}
		};
		void clear() override {
			values.clear();
		};
	};
	const int32_t POSITION_TRACK_SIZE = 5;
	const int32_t ROTATION_TRACK_SIZE = 6;
	const int32_t SCALE_TRACK_SIZE = 5;
	const int32_t BLEND_SHAPE_TRACK_SIZE = 3;

	/* POSITION TRACK */

	class PositionTrack : public Track<Vector3> {
	public:
		PositionTrack() { type = TYPE_POSITION_3D; }
	};

	/* ROTATION TRACK */

	class RotationTrack : public Track<Quaternion> {
	public:
		RotationTrack() { type = TYPE_ROTATION_3D; }
	};

	/* SCALE TRACK */

	class ScaleTrack : public Track<Vector3> {
	public:
		ScaleTrack() { type = TYPE_SCALE_3D; }
	};

	/* BLEND SHAPE TRACK */

	class BlendShapeTrack : public Track<float> {
	public:
		BlendShapeTrack() { type = TYPE_BLEND_SHAPE; }
	};

	/* PROPERTY VALUE TRACK */

	class ValueTrack : public Track<Variant> {
	public:
		UpdateMode update_mode = UPDATE_CONTINUOUS;
		bool update_on_seek = false;
		ValueTrack() { type = TYPE_VALUE; }
	};

	/* METHOD TRACK */

	struct MethodHandle {
		StringName method;
		Vector<Variant> params;
	};

	class MethodTrack : public Track<MethodHandle> {
	public:
		MethodTrack() { type = TYPE_METHOD; }
	};

	/* BEZIER TRACK */
	struct BezierHandle {
		Vector2 in_handle; //relative (x always <0)
		Vector2 out_handle; //relative (x always >0)
		HandleMode handle_mode = HANDLE_MODE_BALANCED;
		real_t value = 0.0;
	};

	class BezierTrack : public Track<BezierHandle> {
	public:
		BezierTrack() { type = TYPE_BEZIER; }
	};

	/* AUDIO TRACK */

	struct AudioHandle {
		Ref<Resource> stream;
		real_t start_offset = 0.0; //offset from start
		real_t end_offset = 0.0; //offset from end, if 0 then full length or infinite
		AudioHandle() {}
	};

	class AudioTrack : public Track<AudioHandle> {
	public:
		AudioTrack() { type = TYPE_AUDIO; }
	};

	/* AUDIO TRACK */

	class AnimationTrack : public Track<StringName> {
	public:
		AnimationTrack() { type = TYPE_ANIMATION; }
	};

	Vector<BaseTrack *> tracks;

	/*
	template<class T>
	int _insert_pos(double p_time, T& p_keys);*/

	template <class T>
	void _clear(T &p_keys);

	template <class K>

	inline int _find(const Vector<K> &p_keys, double p_time, bool p_backward = false) const;

	_FORCE_INLINE_ Vector3 _interpolate(const Vector3 &p_a, const Vector3 &p_b, real_t p_c) const;
	_FORCE_INLINE_ Quaternion _interpolate(const Quaternion &p_a, const Quaternion &p_b, real_t p_c) const;
	_FORCE_INLINE_ Variant _interpolate(const Variant &p_a, const Variant &p_b, real_t p_c) const;
	_FORCE_INLINE_ real_t _interpolate(const real_t &p_a, const real_t &p_b, real_t p_c) const;

	_FORCE_INLINE_ Vector3 _cubic_interpolate(const Vector3 &p_pre_a, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ Quaternion _cubic_interpolate(const Quaternion &p_pre_a, const Quaternion &p_a, const Quaternion &p_b, const Quaternion &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ Variant _cubic_interpolate(const Variant &p_pre_a, const Variant &p_a, const Variant &p_b, const Variant &p_post_b, real_t p_c) const;
	_FORCE_INLINE_ real_t _cubic_interpolate(const real_t &p_pre_a, const real_t &p_a, const real_t &p_b, const real_t &p_post_b, real_t p_c) const;

	template <class T>
	_FORCE_INLINE_ T _interpolate(const Vector<TKey<T>> &p_keys, double p_time, InterpolationType p_interp, bool p_loop_wrap, bool *p_ok, bool p_backward = false) const;

	double length = 1.0;
	real_t step = 0.1;
	LoopMode loop_mode = LOOP_NONE;
	int pingponged = 0;

	/* Animation compression page format (version 1):
	 *
	 * Animation uses bitwidth based compression separated into small pages. The intention is that pages fit easily in the cache, so decoding is cache efficient.
	 * The page-based nature also makes future animation streaming from disk possible.
	 *
	 * Actual format:
	 *
	 * num_compressed_tracks = bounds.size()
	 * header : (x num_compressed_tracks)
	 * -------
	 * timeline_keys_offset : uint32_t - offset to time keys
	 * timeline_size : uint32_t - amount of time keys
	 * data_keys_offset : uint32_t offset to key data
	 *
	 * time key (uint32_t):
	 * ------------------
	 * frame : bits 0-15 - time offset of key, computed as: page.time_offset + frame * (1.0/fps)
	 * data_key_offset : bits 16-27 - offset to key data, computed as: data_keys_offset * 4 + data_key_offset
	 * data_key_count : bits 28-31 - amount of data keys pointed to, computed as: data_key_count+1 (max 16)
	 *
	 * data key:
	 * ---------
	 * X / Blend Shape : uint16_t - X coordinate of XYZ vector key, or Blend Shape value. If Blend shape, Y and Z are not present and can be ignored.
	 * Y : uint16_t
	 * Z : uint16_t
	 * If data_key_count+1 > 1 (if more than 1 key is stored):
	 * data_bitwidth : uint16_t - This is only present if data_key_count > 1. Contains delta bitwidth information.
	 *    X / Blend Shape delta bitwidth: bits 0-3 -
	 * if 0, nothing is present for X (use the first key-value for subsequent keys),
	 * else assume the number of bits present for each element (+ 1 for sign). Assumed always 16 bits, delta max signed 15 bits, with underflow and overflow supported.
	 *    Y delta bitwidth : bits 4-7
	 *    Z delta bitwidth : bits 8-11
	 *    FRAME delta bitwidth : 12-15 bits - always present (obviously), actual bitwidth is FRAME+1
	 * Data key is 4 bytes long for Blend Shapes, 8 bytes long for pos/rot/scale.
	 *
	 * delta keys:
	 * -----------
	 * Compressed format is packed in the following format after the data key, containing delta keys one after the next in a tightly bit packed fashion.
	 * FRAME bits -> X / Blend Shape Bits (if bitwidth > 0) -> Y Bits (if not Blend Shape and Y Bitwidth > 0) -> Z Bits (if not Blend Shape and Z Bitwidth > 0)
	 *
	 * data key format:
	 * ----------------
	 * Decoding keys means starting from the base key and going key by key applying deltas until the proper position is reached needed for interpolation.
	 * Resulting values are uint32_t
	 * data for X / Blend Shape, Y and Z must be normalized first: unorm = float(data) / 65535.0
	 * **Blend Shape**: (unorm * 2.0 - 1.0) * Compression::BLEND_SHAPE_RANGE
	 * **Pos/Scale**: unorm_vec3 * bounds[track].size + bounds[track].position
	 * **Rotation**: Quaternion(Vector3::octahedron_decode(unorm_vec3.xy),unorm_vec3.z * Math_PI * 2.0)
	 * **Frame**: page.time_offset + frame * (1.0/fps)
	 */

	struct Compression {
		enum {
			MAX_DATA_TRACK_SIZE = 16384,
			BLEND_SHAPE_RANGE = 8, // - 8.0 to 8.0
			FORMAT_VERSION = 1
		};
		struct Page {
			Vector<uint8_t> data;
			double time_offset;
		};

		uint32_t fps = 120;
		LocalVector<Page> pages;
		LocalVector<AABB> bounds; //used by position and scale tracks (which contain index to track and index to bounds).
		bool enabled = false;
	} compression;

	Vector3i _compress_key(uint32_t p_track, const AABB &p_bounds, int32_t p_key = -1, float p_time = 0.0);
	bool _rotation_interpolate_compressed(uint32_t p_compressed_track, double p_time, Quaternion &r_ret) const;
	bool _pos_scale_interpolate_compressed(uint32_t p_compressed_track, double p_time, Vector3 &r_ret) const;
	bool _blend_shape_interpolate_compressed(uint32_t p_compressed_track, double p_time, float &r_ret) const;
	template <uint32_t COMPONENTS>
	bool _fetch_compressed(uint32_t p_compressed_track, double p_time, Vector3i &r_current_value, double &r_current_time, Vector3i &r_next_value, double &r_next_time, uint32_t *key_index = nullptr) const;
	template <uint32_t COMPONENTS>
	bool _fetch_compressed_by_index(uint32_t p_compressed_track, int p_index, Vector3i &r_value, double &r_time) const;
	int _get_compressed_key_count(uint32_t p_compressed_track) const;
	template <uint32_t COMPONENTS>
	void _get_compressed_key_indices_in_range(uint32_t p_compressed_track, double p_time, double p_delta, List<int> *r_indices) const;
	_FORCE_INLINE_ Quaternion _uncompress_quaternion(const Vector3i &p_value) const;
	_FORCE_INLINE_ Vector3 _uncompress_pos_scale(uint32_t p_compressed_track, const Vector3i &p_value) const;
	_FORCE_INLINE_ float _uncompress_blend_shape(const Vector3i &p_value) const;

	// bind helpers
private:
	bool _position_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_alowed_linear_err, real_t p_allowed_angular_error, const Vector3 &p_norm);
	bool _rotation_track_optimize_key(const TKey<Quaternion> &t0, const TKey<Quaternion> &t1, const TKey<Quaternion> &t2, real_t p_allowed_angular_error, float p_max_optimizable_angle);
	bool _scale_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_allowed_linear_error);
	bool _blend_shape_track_optimize_key(const TKey<float> &t0, const TKey<float> &t1, const TKey<float> &t2, real_t p_allowed_unit_error);

	void _position_track_optimize(int p_idx, real_t p_allowed_linear_err, real_t p_allowed_angular_err);
	void _rotation_track_optimize(int p_idx, real_t p_allowed_angular_err, real_t p_max_optimizable_angle);
	void _scale_track_optimize(int p_idx, real_t p_allowed_linear_err);
	void _blend_shape_track_optimize(int p_idx, real_t p_allowed_unit_error);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual void reset_state() override;

	static void _bind_methods();

public:
	int add_track(TrackType p_type, int p_at_pos = -1);
	void remove_track(int p_track);

	int get_track_count() const;
	TrackType track_get_type(int p_track) const;

	void track_set_path(int p_track, const NodePath &p_path);
	NodePath track_get_path(int p_track) const;
	int find_track(const NodePath &p_path, const TrackType p_type) const;

	void track_move_up(int p_track);
	void track_move_down(int p_track);
	void track_move_to(int p_track, int p_to_index);
	void track_swap(int p_track, int p_with_track);

	void track_set_imported(int p_track, bool p_imported);
	bool track_is_imported(int p_track) const;

	void track_set_enabled(int p_track, bool p_enabled);
	bool track_is_enabled(int p_track) const;

	void track_insert_key(int p_track, double p_time, const Variant &p_key, real_t p_transition = 1);
	void track_set_key_transition(int p_track, int p_key_idx, real_t p_transition);
	void track_set_key_value(int p_track, int p_key_idx, const Variant &p_value);
	void track_set_key_time(int p_track, int p_key_idx, double p_time);
	int track_find_key(int p_track, double p_time, bool p_exact = false) const;
	void track_remove_key(int p_track, int p_idx);
	void track_remove_key_at_time(int p_track, double p_time);
	int track_get_key_count(int p_track) const;
	Variant track_get_key_value(int p_track, int p_key_idx) const;
	double track_get_key_time(int p_track, int p_key_idx) const;
	real_t track_get_key_transition(int p_track, int p_key_idx) const;
	bool track_is_compressed(int p_track) const;

	int position_track_insert_key(int p_track, double p_time, const Vector3 &p_position);
	Error position_track_get_key(int p_track, int p_key, Vector3 *r_position) const;
	Error position_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const;

	int rotation_track_insert_key(int p_track, double p_time, const Quaternion &p_rotation);
	Error rotation_track_get_key(int p_track, int p_key, Quaternion *r_rotation) const;
	Error rotation_track_interpolate(int p_track, double p_time, Quaternion *r_interpolation) const;

	int scale_track_insert_key(int p_track, double p_time, const Vector3 &p_scale);
	Error scale_track_get_key(int p_track, int p_key, Vector3 *r_scale) const;
	Error scale_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const;

	int blend_shape_track_insert_key(int p_track, double p_time, float p_blend);
	Error blend_shape_track_get_key(int p_track, int p_key, float *r_blend) const;
	Error blend_shape_track_interpolate(int p_track, double p_time, float *r_blend) const;

	void track_set_interpolation_type(int p_track, InterpolationType p_interp);
	InterpolationType track_get_interpolation_type(int p_track) const;

	int bezier_track_insert_key(int p_track, double p_time, real_t p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle, const HandleMode p_handle_mode = HandleMode::HANDLE_MODE_BALANCED);
	void bezier_track_set_key_handle_mode(int p_track, int p_index, HandleMode p_mode, double p_balanced_value_time_ratio = 1.0);
	void bezier_track_set_key_value(int p_track, int p_index, real_t p_value);
	void bezier_track_set_key_in_handle(int p_track, int p_index, const Vector2 &p_handle, double p_balanced_value_time_ratio = 1.0);
	void bezier_track_set_key_out_handle(int p_track, int p_index, const Vector2 &p_handle, double p_balanced_value_time_ratio = 1.0);
	real_t bezier_track_get_key_value(int p_track, int p_index) const;
	int bezier_track_get_key_handle_mode(int p_track, int p_index) const;
	Vector2 bezier_track_get_key_in_handle(int p_track, int p_index) const;
	Vector2 bezier_track_get_key_out_handle(int p_track, int p_index) const;

	real_t bezier_track_interpolate(int p_track, double p_time) const;

	int audio_track_insert_key(int p_track, double p_time, const Ref<Resource> &p_stream, real_t p_start_offset = 0, real_t p_end_offset = 0);
	void audio_track_set_key_stream(int p_track, int p_key, const Ref<Resource> &p_stream);
	void audio_track_set_key_start_offset(int p_track, int p_key, real_t p_offset);
	void audio_track_set_key_end_offset(int p_track, int p_key, real_t p_offset);
	Ref<Resource> audio_track_get_key_stream(int p_track, int p_key) const;
	real_t audio_track_get_key_start_offset(int p_track, int p_key) const;
	real_t audio_track_get_key_end_offset(int p_track, int p_key) const;

	int animation_track_insert_key(int p_track, double p_time, const StringName &p_animation);
	void animation_track_set_key_animation(int p_track, int p_key, const StringName &p_animation);
	StringName animation_track_get_key_animation(int p_track, int p_key) const;

	void track_set_interpolation_loop_wrap(int p_track, bool p_enable);
	bool track_get_interpolation_loop_wrap(int p_track) const;

	Variant value_track_interpolate(int p_track, double p_time) const;
	void value_track_set_update_mode(int p_track, UpdateMode p_mode);
	UpdateMode value_track_get_update_mode(int p_track) const;

	Vector<Variant> method_track_get_params(int p_track, int p_key_idx) const;
	StringName method_track_get_name(int p_track, int p_key_idx) const;

	void copy_track(int p_track, Ref<Animation> p_to_animation);

	void track_get_key_indices_in_range(int p_track, double p_time, double p_delta, List<int> *p_indices, int p_pingponged = 0) const;

	void set_length(real_t p_length);
	real_t get_length() const;

	void set_loop_mode(LoopMode p_loop_mode);
	LoopMode get_loop_mode() const;

	void set_step(real_t p_step);
	real_t get_step() const;

	void clear();

	void optimize(real_t p_allowed_linear_err = 0.05, real_t p_allowed_angular_err = 0.01, real_t p_max_optimizable_angle = Math_PI * 0.125);
	void compress(uint32_t p_page_size = 8192, uint32_t p_fps = 120, float p_split_tolerance = 4.0); // 4.0 seems to be the split tolerance sweet spot from many tests

	Animation();
	~Animation();
};

VARIANT_ENUM_CAST(Animation::TrackType);
VARIANT_ENUM_CAST(Animation::InterpolationType);
VARIANT_ENUM_CAST(Animation::UpdateMode);
VARIANT_ENUM_CAST(Animation::HandleMode);
VARIANT_ENUM_CAST(Animation::LoopMode);

#endif // ANIMATION_H
