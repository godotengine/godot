/*************************************************************************/
/*  animation.cpp                                                        */
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

#include "animation.h"

#include "core/math/geometry_3d.h"
#include "scene/scene_string_names.h"

bool Animation::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name.begins_with("tracks/")) {
		int track = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);

		if (tracks.size() == track && what == "type") {
			String type = p_value;

			if (type == "transform" || type == "transform3d") {
				add_track(TYPE_TRANSFORM3D);
			} else if (type == "value") {
				add_track(TYPE_VALUE);
			} else if (type == "method") {
				add_track(TYPE_METHOD);
			} else if (type == "bezier") {
				add_track(TYPE_BEZIER);
			} else if (type == "audio") {
				add_track(TYPE_AUDIO);
			} else if (type == "animation") {
				add_track(TYPE_ANIMATION);
			} else {
				return false;
			}

			return true;
		}

		ERR_FAIL_INDEX_V(track, tracks.size(), false);

		if (what == "path") {
			track_set_path(track, p_value);
		} else if (what == "interp") {
			track_set_interpolation_type(track, InterpolationType(p_value.operator int()));
		} else if (what == "loop_wrap") {
			track_set_interpolation_loop_wrap(track, p_value);
		} else if (what == "imported") {
			track_set_imported(track, p_value);
		} else if (what == "enabled") {
			track_set_enabled(track, p_value);
		} else if (what == "keys" || what == "key_values") {
			if (track_get_type(track) == TYPE_TRANSFORM3D) {
				TransformTrack *tt = static_cast<TransformTrack *>(tracks[track]);
				Vector<real_t> values = p_value;
				int vcount = values.size();
				ERR_FAIL_COND_V(vcount % TRANSFORM_TRACK_SIZE, false);

				const real_t *r = values.ptr();

				int64_t count = vcount / TRANSFORM_TRACK_SIZE;
				tt->transforms.resize(count);

				for (int i = 0; i < count; i++) {
					TKey<TransformKey> &tk = tt->transforms.write[i];
					const real_t *ofs = &r[i * TRANSFORM_TRACK_SIZE];
					tk.time = ofs[0];
					tk.transition = ofs[1];

					tk.value.loc.x = ofs[2];
					tk.value.loc.y = ofs[3];
					tk.value.loc.z = ofs[4];

					tk.value.rot.x = ofs[5];
					tk.value.rot.y = ofs[6];
					tk.value.rot.z = ofs[7];
					tk.value.rot.w = ofs[8];

					tk.value.scale.x = ofs[9];
					tk.value.scale.y = ofs[10];
					tk.value.scale.z = ofs[11];
				}

			} else if (track_get_type(track) == TYPE_VALUE) {
				ValueTrack *vt = static_cast<ValueTrack *>(tracks[track]);
				Dictionary d = p_value;
				ERR_FAIL_COND_V(!d.has("times"), false);
				ERR_FAIL_COND_V(!d.has("values"), false);
				if (d.has("cont")) {
					bool v = d["cont"];
					vt->update_mode = v ? UPDATE_CONTINUOUS : UPDATE_DISCRETE;
				}

				if (d.has("update")) {
					int um = d["update"];
					if (um < 0) {
						um = 0;
					} else if (um > 3) {
						um = 3;
					}
					vt->update_mode = UpdateMode(um);
				}

				Vector<real_t> times = d["times"];
				Array values = d["values"];

				ERR_FAIL_COND_V(times.size() != values.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();

					vt->values.resize(valcount);

					for (int i = 0; i < valcount; i++) {
						vt->values.write[i].time = rt[i];
						vt->values.write[i].value = values[i];
					}

					if (d.has("transitions")) {
						Vector<real_t> transitions = d["transitions"];
						ERR_FAIL_COND_V(transitions.size() != valcount, false);

						const real_t *rtr = transitions.ptr();

						for (int i = 0; i < valcount; i++) {
							vt->values.write[i].transition = rtr[i];
						}
					}
				}

				return true;

			} else if (track_get_type(track) == TYPE_METHOD) {
				while (track_get_key_count(track)) {
					track_remove_key(track, 0); //well shouldn't be set anyway
				}

				Dictionary d = p_value;
				ERR_FAIL_COND_V(!d.has("times"), false);
				ERR_FAIL_COND_V(!d.has("values"), false);

				Vector<real_t> times = d["times"];
				Array values = d["values"];

				ERR_FAIL_COND_V(times.size() != values.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();

					for (int i = 0; i < valcount; i++) {
						track_insert_key(track, rt[i], values[i]);
					}

					if (d.has("transitions")) {
						Vector<real_t> transitions = d["transitions"];
						ERR_FAIL_COND_V(transitions.size() != valcount, false);

						const real_t *rtr = transitions.ptr();

						for (int i = 0; i < valcount; i++) {
							track_set_key_transition(track, i, rtr[i]);
						}
					}
				}
			} else if (track_get_type(track) == TYPE_BEZIER) {
				BezierTrack *bt = static_cast<BezierTrack *>(tracks[track]);
				Dictionary d = p_value;
				ERR_FAIL_COND_V(!d.has("times"), false);
				ERR_FAIL_COND_V(!d.has("points"), false);

				Vector<real_t> times = d["times"];
				Vector<real_t> values = d["points"];

				ERR_FAIL_COND_V(times.size() * 5 != values.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();
					const real_t *rv = values.ptr();

					bt->values.resize(valcount);

					for (int i = 0; i < valcount; i++) {
						bt->values.write[i].time = rt[i];
						bt->values.write[i].transition = 0; //unused in bezier
						bt->values.write[i].value.value = rv[i * 5 + 0];
						bt->values.write[i].value.in_handle.x = rv[i * 5 + 1];
						bt->values.write[i].value.in_handle.y = rv[i * 5 + 2];
						bt->values.write[i].value.out_handle.x = rv[i * 5 + 3];
						bt->values.write[i].value.out_handle.y = rv[i * 5 + 4];
					}
				}

				return true;
			} else if (track_get_type(track) == TYPE_AUDIO) {
				AudioTrack *ad = static_cast<AudioTrack *>(tracks[track]);
				Dictionary d = p_value;
				ERR_FAIL_COND_V(!d.has("times"), false);
				ERR_FAIL_COND_V(!d.has("clips"), false);

				Vector<real_t> times = d["times"];
				Array clips = d["clips"];

				ERR_FAIL_COND_V(clips.size() != times.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();

					ad->values.clear();

					for (int i = 0; i < valcount; i++) {
						Dictionary d2 = clips[i];
						if (!d2.has("start_offset")) {
							continue;
						}
						if (!d2.has("end_offset")) {
							continue;
						}
						if (!d2.has("stream")) {
							continue;
						}

						TKey<AudioKey> ak;
						ak.time = rt[i];
						ak.value.start_offset = d2["start_offset"];
						ak.value.end_offset = d2["end_offset"];
						ak.value.stream = d2["stream"];

						ad->values.push_back(ak);
					}
				}

				return true;
			} else if (track_get_type(track) == TYPE_ANIMATION) {
				AnimationTrack *an = static_cast<AnimationTrack *>(tracks[track]);
				Dictionary d = p_value;
				ERR_FAIL_COND_V(!d.has("times"), false);
				ERR_FAIL_COND_V(!d.has("clips"), false);

				Vector<real_t> times = d["times"];
				Vector<String> clips = d["clips"];

				ERR_FAIL_COND_V(clips.size() != times.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();
					const String *rc = clips.ptr();

					an->values.resize(valcount);

					for (int i = 0; i < valcount; i++) {
						TKey<StringName> ak;
						ak.time = rt[i];
						ak.value = rc[i];
						an->values.write[i] = ak;
					}
				}

				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	} else {
		return false;
	}

	return true;
}

bool Animation::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name == "length") {
		r_ret = length;
	} else if (name == "loop_mode") {
		r_ret = loop_mode;
	} else if (name == "step") {
		r_ret = step;
	} else if (name.begins_with("tracks/")) {
		int track = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(track, tracks.size(), false);
		if (what == "type") {
			switch (track_get_type(track)) {
				case TYPE_TRANSFORM3D:
					r_ret = "transform";
					break;
				case TYPE_VALUE:
					r_ret = "value";
					break;
				case TYPE_METHOD:
					r_ret = "method";
					break;
				case TYPE_BEZIER:
					r_ret = "bezier";
					break;
				case TYPE_AUDIO:
					r_ret = "audio";
					break;
				case TYPE_ANIMATION:
					r_ret = "animation";
					break;
			}

			return true;

		} else if (what == "path") {
			r_ret = track_get_path(track);
		} else if (what == "interp") {
			r_ret = track_get_interpolation_type(track);
		} else if (what == "loop_wrap") {
			r_ret = track_get_interpolation_loop_wrap(track);
		} else if (what == "imported") {
			r_ret = track_is_imported(track);
		} else if (what == "enabled") {
			r_ret = track_is_enabled(track);
		} else if (what == "keys") {
			if (track_get_type(track) == TYPE_TRANSFORM3D) {
				Vector<real_t> keys;
				int kk = track_get_key_count(track);
				keys.resize(kk * TRANSFORM_TRACK_SIZE);

				real_t *w = keys.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					Vector3 loc;
					Quaternion rot;
					Vector3 scale;
					transform_track_get_key(track, i, &loc, &rot, &scale);

					w[idx++] = track_get_key_time(track, i);
					w[idx++] = track_get_key_transition(track, i);
					w[idx++] = loc.x;
					w[idx++] = loc.y;
					w[idx++] = loc.z;

					w[idx++] = rot.x;
					w[idx++] = rot.y;
					w[idx++] = rot.z;
					w[idx++] = rot.w;

					w[idx++] = scale.x;
					w[idx++] = scale.y;
					w[idx++] = scale.z;
				}

				r_ret = keys;
				return true;

			} else if (track_get_type(track) == TYPE_VALUE) {
				const ValueTrack *vt = static_cast<const ValueTrack *>(tracks[track]);

				Dictionary d;

				Vector<real_t> key_times;
				Vector<real_t> key_transitions;
				Array key_values;

				int kk = vt->values.size();

				key_times.resize(kk);
				key_transitions.resize(kk);
				key_values.resize(kk);

				real_t *wti = key_times.ptrw();
				real_t *wtr = key_transitions.ptrw();

				int idx = 0;

				const TKey<Variant> *vls = vt->values.ptr();

				for (int i = 0; i < kk; i++) {
					wti[idx] = vls[i].time;
					wtr[idx] = vls[i].transition;
					key_values[idx] = vls[i].value;
					idx++;
				}

				d["times"] = key_times;
				d["transitions"] = key_transitions;
				d["values"] = key_values;
				if (track_get_type(track) == TYPE_VALUE) {
					d["update"] = value_track_get_update_mode(track);
				}

				r_ret = d;

				return true;

			} else if (track_get_type(track) == TYPE_METHOD) {
				Dictionary d;

				Vector<real_t> key_times;
				Vector<real_t> key_transitions;
				Array key_values;

				int kk = track_get_key_count(track);

				key_times.resize(kk);
				key_transitions.resize(kk);
				key_values.resize(kk);

				real_t *wti = key_times.ptrw();
				real_t *wtr = key_transitions.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					wti[idx] = track_get_key_time(track, i);
					wtr[idx] = track_get_key_transition(track, i);
					key_values[idx] = track_get_key_value(track, i);
					idx++;
				}

				d["times"] = key_times;
				d["transitions"] = key_transitions;
				d["values"] = key_values;
				if (track_get_type(track) == TYPE_VALUE) {
					d["update"] = value_track_get_update_mode(track);
				}

				r_ret = d;

				return true;
			} else if (track_get_type(track) == TYPE_BEZIER) {
				const BezierTrack *bt = static_cast<const BezierTrack *>(tracks[track]);

				Dictionary d;

				Vector<real_t> key_times;
				Vector<real_t> key_points;

				int kk = bt->values.size();

				key_times.resize(kk);
				key_points.resize(kk * 5);

				real_t *wti = key_times.ptrw();
				real_t *wpo = key_points.ptrw();

				int idx = 0;

				const TKey<BezierKey> *vls = bt->values.ptr();

				for (int i = 0; i < kk; i++) {
					wti[idx] = vls[i].time;
					wpo[idx * 5 + 0] = vls[i].value.value;
					wpo[idx * 5 + 1] = vls[i].value.in_handle.x;
					wpo[idx * 5 + 2] = vls[i].value.in_handle.y;
					wpo[idx * 5 + 3] = vls[i].value.out_handle.x;
					wpo[idx * 5 + 4] = vls[i].value.out_handle.y;
					idx++;
				}

				d["times"] = key_times;
				d["points"] = key_points;

				r_ret = d;

				return true;
			} else if (track_get_type(track) == TYPE_AUDIO) {
				const AudioTrack *ad = static_cast<const AudioTrack *>(tracks[track]);

				Dictionary d;

				Vector<real_t> key_times;
				Array clips;

				int kk = ad->values.size();

				key_times.resize(kk);

				real_t *wti = key_times.ptrw();

				int idx = 0;

				const TKey<AudioKey> *vls = ad->values.ptr();

				for (int i = 0; i < kk; i++) {
					wti[idx] = vls[i].time;
					Dictionary clip;
					clip["start_offset"] = vls[i].value.start_offset;
					clip["end_offset"] = vls[i].value.end_offset;
					clip["stream"] = vls[i].value.stream;
					clips.push_back(clip);
					idx++;
				}

				d["times"] = key_times;
				d["clips"] = clips;

				r_ret = d;

				return true;
			} else if (track_get_type(track) == TYPE_ANIMATION) {
				const AnimationTrack *an = static_cast<const AnimationTrack *>(tracks[track]);

				Dictionary d;

				Vector<real_t> key_times;
				Vector<String> clips;

				int kk = an->values.size();

				key_times.resize(kk);
				clips.resize(kk);

				real_t *wti = key_times.ptrw();
				String *wcl = clips.ptrw();

				const TKey<StringName> *vls = an->values.ptr();

				for (int i = 0; i < kk; i++) {
					wti[i] = vls[i].time;
					wcl[i] = vls[i].value;
				}

				d["times"] = key_times;
				d["clips"] = clips;

				r_ret = d;

				return true;
			}
		} else {
			return false;
		}
	} else {
		return false;
	}

	return true;
}

void Animation::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < tracks.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, "tracks/" + itos(i) + "/type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "tracks/" + itos(i) + "/path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::INT, "tracks/" + itos(i) + "/interp", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/loop_wrap", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/imported", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::ARRAY, "tracks/" + itos(i) + "/keys", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL));
	}
}

void Animation::reset_state() {
	clear();
}

int Animation::add_track(TrackType p_type, int p_at_pos) {
	if (p_at_pos < 0 || p_at_pos >= tracks.size()) {
		p_at_pos = tracks.size();
	}

	switch (p_type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = memnew(TransformTrack);
			tracks.insert(p_at_pos, tt);
		} break;
		case TYPE_VALUE: {
			tracks.insert(p_at_pos, memnew(ValueTrack));

		} break;
		case TYPE_METHOD: {
			tracks.insert(p_at_pos, memnew(MethodTrack));

		} break;
		case TYPE_BEZIER: {
			tracks.insert(p_at_pos, memnew(BezierTrack));

		} break;
		case TYPE_AUDIO: {
			tracks.insert(p_at_pos, memnew(AudioTrack));

		} break;
		case TYPE_ANIMATION: {
			tracks.insert(p_at_pos, memnew(AnimationTrack));

		} break;
		default: {
			ERR_PRINT("Unknown track type");
		}
	}
	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
	return p_at_pos;
}

void Animation::remove_track(int p_track) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			_clear(tt->transforms);

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			_clear(vt->values);

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			_clear(mt->methods);

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bz = static_cast<BezierTrack *>(t);
			_clear(bz->values);

		} break;
		case TYPE_AUDIO: {
			AudioTrack *ad = static_cast<AudioTrack *>(t);
			_clear(ad->values);

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *an = static_cast<AnimationTrack *>(t);
			_clear(an->values);

		} break;
	}

	memdelete(t);
	tracks.remove(p_track);
	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

int Animation::get_track_count() const {
	return tracks.size();
}

Animation::TrackType Animation::track_get_type(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), TYPE_TRANSFORM3D);
	return tracks[p_track]->type;
}

void Animation::track_set_path(int p_track, const NodePath &p_path) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	tracks[p_track]->path = p_path;
	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

NodePath Animation::track_get_path(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), NodePath());
	return tracks[p_track]->path;
}

int Animation::find_track(const NodePath &p_path) const {
	for (int i = 0; i < tracks.size(); i++) {
		if (tracks[i]->path == p_path) {
			return i;
		}
	};
	return -1;
};

void Animation::track_set_interpolation_type(int p_track, InterpolationType p_interp) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	ERR_FAIL_INDEX(p_interp, 3);
	tracks[p_track]->interpolation = p_interp;
	emit_changed();
}

Animation::InterpolationType Animation::track_get_interpolation_type(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), INTERPOLATION_NEAREST);
	return tracks[p_track]->interpolation;
}

void Animation::track_set_interpolation_loop_wrap(int p_track, bool p_enable) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	tracks[p_track]->loop_wrap = p_enable;
	emit_changed();
}

bool Animation::track_get_interpolation_loop_wrap(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), INTERPOLATION_NEAREST);
	return tracks[p_track]->loop_wrap;
}

// transform
/*
template<class T>
int Animation::_insert_pos(double p_time, T& p_keys) {
	// simple, linear time inset that should be fast enough in reality.

	int idx=p_keys.size();

	while(true) {


		if (idx==0 || p_keys[idx-1].time < p_time) {
			//condition for insertion.
			p_keys.insert(idx,T());
			return idx;
		} else if (p_keys[idx-1].time == p_time) {
			// condition for replacing.
			return idx-1;
		}

		idx--;
	}
}

*/
template <class T, class V>
int Animation::_insert(double p_time, T &p_keys, const V &p_value) {
	int idx = p_keys.size();

	while (true) {
		// Condition for replacement.
		if (idx > 0 && Math::is_equal_approx((double)p_keys[idx - 1].time, p_time)) {
			float transition = p_keys[idx - 1].transition;
			p_keys.write[idx - 1] = p_value;
			p_keys.write[idx - 1].transition = transition;
			return idx - 1;

			// Condition for insert.
		} else if (idx == 0 || p_keys[idx - 1].time < p_time) {
			p_keys.insert(idx, p_value);
			return idx;
		}

		idx--;
	}

	return -1;
}

template <class T>
void Animation::_clear(T &p_keys) {
	p_keys.clear();
}

Error Animation::transform_track_get_key(int p_track, int p_key, Vector3 *r_loc, Quaternion *r_rot, Vector3 *r_scale) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];

	TransformTrack *tt = static_cast<TransformTrack *>(t);
	ERR_FAIL_COND_V(t->type != TYPE_TRANSFORM3D, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_key, tt->transforms.size(), ERR_INVALID_PARAMETER);

	if (r_loc) {
		*r_loc = tt->transforms[p_key].value.loc;
	}
	if (r_rot) {
		*r_rot = tt->transforms[p_key].value.rot;
	}
	if (r_scale) {
		*r_scale = tt->transforms[p_key].value.scale;
	}

	return OK;
}

int Animation::transform_track_insert_key(int p_track, double p_time, const Vector3 &p_loc, const Quaternion &p_rot, const Vector3 &p_scale) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_TRANSFORM3D, -1);

	TransformTrack *tt = static_cast<TransformTrack *>(t);

	TKey<TransformKey> tkey;
	tkey.time = p_time;
	tkey.value.loc = p_loc;
	tkey.value.rot = p_rot;
	tkey.value.scale = p_scale;

	int ret = _insert(p_time, tt->transforms, tkey);
	emit_changed();
	return ret;
}

void Animation::track_remove_key_at_time(int p_track, double p_time) {
	int idx = track_find_key(p_track, p_time, true);
	ERR_FAIL_COND(idx < 0);
	track_remove_key(p_track, idx);
}

void Animation::track_remove_key(int p_track, int p_idx) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX(p_idx, tt->transforms.size());
			tt->transforms.remove(p_idx);

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_idx, vt->values.size());
			vt->values.remove(p_idx);

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_idx, mt->methods.size());
			mt->methods.remove(p_idx);

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bz = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX(p_idx, bz->values.size());
			bz->values.remove(p_idx);

		} break;
		case TYPE_AUDIO: {
			AudioTrack *ad = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX(p_idx, ad->values.size());
			ad->values.remove(p_idx);

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *an = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX(p_idx, an->values.size());
			an->values.remove(p_idx);

		} break;
	}

	emit_changed();
}

int Animation::track_find_key(int p_track, double p_time, bool p_exact) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			int k = _find(tt->transforms, p_time);
			if (k < 0 || k >= tt->transforms.size()) {
				return -1;
			}
			if (tt->transforms[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			int k = _find(vt->values, p_time);
			if (k < 0 || k >= vt->values.size()) {
				return -1;
			}
			if (vt->values[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			int k = _find(mt->methods, p_time);
			if (k < 0 || k >= mt->methods.size()) {
				return -1;
			}
			if (mt->methods[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			int k = _find(bt->values, p_time);
			if (k < 0 || k >= bt->values.size()) {
				return -1;
			}
			if (bt->values[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			int k = _find(at->values, p_time);
			if (k < 0 || k >= at->values.size()) {
				return -1;
			}
			if (at->values[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			int k = _find(at->values, p_time);
			if (k < 0 || k >= at->values.size()) {
				return -1;
			}
			if (at->values[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
	}

	return -1;
}

void Animation::track_insert_key(int p_track, double p_time, const Variant &p_key, real_t p_transition) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			Dictionary d = p_key;
			Vector3 loc;
			if (d.has("location")) {
				loc = d["location"];
			}

			Quaternion rot;
			if (d.has("rotation")) {
				rot = d["rotation"];
			}

			Vector3 scale;
			if (d.has("scale")) {
				scale = d["scale"];
			}

			int idx = transform_track_insert_key(p_track, p_time, loc, rot, scale);
			track_set_key_transition(p_track, idx, p_transition);

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);

			TKey<Variant> k;
			k.time = p_time;
			k.transition = p_transition;
			k.value = p_key;
			_insert(p_time, vt->values, k);

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);

			ERR_FAIL_COND(p_key.get_type() != Variant::DICTIONARY);

			Dictionary d = p_key;
			ERR_FAIL_COND(!d.has("method") || (d["method"].get_type() != Variant::STRING_NAME && d["method"].get_type() != Variant::STRING));
			ERR_FAIL_COND(!d.has("args") || !d["args"].is_array());

			MethodKey k;

			k.time = p_time;
			k.transition = p_transition;
			k.method = d["method"];
			k.params = d["args"];

			_insert(p_time, mt->methods, k);

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);

			Array arr = p_key;
			ERR_FAIL_COND(arr.size() != 5);

			TKey<BezierKey> k;
			k.time = p_time;
			k.value.value = arr[0];
			k.value.in_handle.x = arr[1];
			k.value.in_handle.y = arr[2];
			k.value.out_handle.x = arr[3];
			k.value.out_handle.y = arr[4];
			_insert(p_time, bt->values, k);

		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);

			Dictionary k = p_key;
			ERR_FAIL_COND(!k.has("start_offset"));
			ERR_FAIL_COND(!k.has("end_offset"));
			ERR_FAIL_COND(!k.has("stream"));

			TKey<AudioKey> ak;
			ak.time = p_time;
			ak.value.start_offset = k["start_offset"];
			ak.value.end_offset = k["end_offset"];
			ak.value.stream = k["stream"];
			_insert(p_time, at->values, ak);

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);

			TKey<StringName> ak;
			ak.time = p_time;
			ak.value = p_key;

			_insert(p_time, at->values, ak);

		} break;
	}

	emit_changed();
}

int Animation::track_get_key_count(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			return tt->transforms.size();
		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			return vt->values.size();

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			return mt->methods.size();
		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			return bt->values.size();
		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			return at->values.size();
		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			return at->values.size();
		} break;
	}

	ERR_FAIL_V(-1);
}

Variant Animation::track_get_key_value(int p_track, int p_key_idx) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), Variant());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->transforms.size(), Variant());

			Dictionary d;
			d["location"] = tt->transforms[p_key_idx].value.loc;
			d["rotation"] = tt->transforms[p_key_idx].value.rot;
			d["scale"] = tt->transforms[p_key_idx].value.scale;

			return d;
		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, vt->values.size(), Variant());
			return vt->values[p_key_idx].value;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, mt->methods.size(), Variant());
			Dictionary d;
			d["method"] = mt->methods[p_key_idx].method;
			d["args"] = mt->methods[p_key_idx].params;
			return d;

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, bt->values.size(), Variant());

			Array arr;
			arr.resize(5);
			arr[0] = bt->values[p_key_idx].value.value;
			arr[1] = bt->values[p_key_idx].value.in_handle.x;
			arr[2] = bt->values[p_key_idx].value.in_handle.y;
			arr[3] = bt->values[p_key_idx].value.out_handle.x;
			arr[4] = bt->values[p_key_idx].value.out_handle.y;
			return arr;

		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, at->values.size(), Variant());

			Dictionary k;
			k["start_offset"] = at->values[p_key_idx].value.start_offset;
			k["end_offset"] = at->values[p_key_idx].value.end_offset;
			k["stream"] = at->values[p_key_idx].value.stream;
			return k;

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, at->values.size(), Variant());

			return at->values[p_key_idx].value;

		} break;
	}

	ERR_FAIL_V(Variant());
}

double Animation::track_get_key_time(int p_track, int p_key_idx) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->transforms.size(), -1);
			return tt->transforms[p_key_idx].time;
		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, vt->values.size(), -1);
			return vt->values[p_key_idx].time;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, mt->methods.size(), -1);
			return mt->methods[p_key_idx].time;

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, bt->values.size(), -1);
			return bt->values[p_key_idx].time;

		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, at->values.size(), -1);
			return at->values[p_key_idx].time;

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, at->values.size(), -1);
			return at->values[p_key_idx].time;

		} break;
	}

	ERR_FAIL_V(-1);
}

void Animation::track_set_key_time(int p_track, int p_key_idx, double p_time) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->transforms.size());
			TKey<TransformKey> key = tt->transforms[p_key_idx];
			key.time = p_time;
			tt->transforms.remove(p_key_idx);
			_insert(p_time, tt->transforms, key);
			return;
		}
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, vt->values.size());
			TKey<Variant> key = vt->values[p_key_idx];
			key.time = p_time;
			vt->values.remove(p_key_idx);
			_insert(p_time, vt->values, key);
			return;
		}
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, mt->methods.size());
			MethodKey key = mt->methods[p_key_idx];
			key.time = p_time;
			mt->methods.remove(p_key_idx);
			_insert(p_time, mt->methods, key);
			return;
		}
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, bt->values.size());
			TKey<BezierKey> key = bt->values[p_key_idx];
			key.time = p_time;
			bt->values.remove(p_key_idx);
			_insert(p_time, bt->values, key);
			return;
		}
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());
			TKey<AudioKey> key = at->values[p_key_idx];
			key.time = p_time;
			at->values.remove(p_key_idx);
			_insert(p_time, at->values, key);
			return;
		}
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());
			TKey<StringName> key = at->values[p_key_idx];
			key.time = p_time;
			at->values.remove(p_key_idx);
			_insert(p_time, at->values, key);
			return;
		}
	}

	ERR_FAIL();
}

real_t Animation::track_get_key_transition(int p_track, int p_key_idx) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->transforms.size(), -1);
			return tt->transforms[p_key_idx].transition;
		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, vt->values.size(), -1);
			return vt->values[p_key_idx].transition;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, mt->methods.size(), -1);
			return mt->methods[p_key_idx].transition;

		} break;
		case TYPE_BEZIER: {
			return 1; //bezier does not really use transitions
		} break;
		case TYPE_AUDIO: {
			return 1; //audio does not really use transitions
		} break;
		case TYPE_ANIMATION: {
			return 1; //animation does not really use transitions
		} break;
	}

	ERR_FAIL_V(0);
}

void Animation::track_set_key_value(int p_track, int p_key_idx, const Variant &p_value) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->transforms.size());

			Dictionary d = p_value;

			if (d.has("location")) {
				tt->transforms.write[p_key_idx].value.loc = d["location"];
			}
			if (d.has("rotation")) {
				tt->transforms.write[p_key_idx].value.rot = d["rotation"];
			}
			if (d.has("scale")) {
				tt->transforms.write[p_key_idx].value.scale = d["scale"];
			}

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, vt->values.size());

			vt->values.write[p_key_idx].value = p_value;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, mt->methods.size());

			Dictionary d = p_value;

			if (d.has("method")) {
				mt->methods.write[p_key_idx].method = d["method"];
			}
			if (d.has("args")) {
				mt->methods.write[p_key_idx].params = d["args"];
			}

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, bt->values.size());

			Array arr = p_value;
			ERR_FAIL_COND(arr.size() != 5);

			bt->values.write[p_key_idx].value.value = arr[0];
			bt->values.write[p_key_idx].value.in_handle.x = arr[1];
			bt->values.write[p_key_idx].value.in_handle.y = arr[2];
			bt->values.write[p_key_idx].value.out_handle.x = arr[3];
			bt->values.write[p_key_idx].value.out_handle.y = arr[4];

		} break;
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());

			Dictionary k = p_value;
			ERR_FAIL_COND(!k.has("start_offset"));
			ERR_FAIL_COND(!k.has("end_offset"));
			ERR_FAIL_COND(!k.has("stream"));

			at->values.write[p_key_idx].value.start_offset = k["start_offset"];
			at->values.write[p_key_idx].value.end_offset = k["end_offset"];
			at->values.write[p_key_idx].value.stream = k["stream"];

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());

			at->values.write[p_key_idx].value = p_value;

		} break;
	}

	emit_changed();
}

void Animation::track_set_key_transition(int p_track, int p_key_idx, real_t p_transition) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			TransformTrack *tt = static_cast<TransformTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->transforms.size());
			tt->transforms.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, vt->values.size());
			vt->values.write[p_key_idx].transition = p_transition;

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, mt->methods.size());
			mt->methods.write[p_key_idx].transition = p_transition;

		} break;
		case TYPE_BEZIER:
		case TYPE_AUDIO:
		case TYPE_ANIMATION: {
			// they don't use transition
		} break;
	}

	emit_changed();
}

template <class K>
int Animation::_find(const Vector<K> &p_keys, double p_time, bool p_backward) const {
	int len = p_keys.size();
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

	const K *keys = &p_keys[0];

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

	if (!p_backward) {
		if (keys[middle].time > p_time) {
			middle--;
		}
	} else {
		if (keys[middle].time < p_time) {
			middle++;
		}
	}

	return middle;
}

Animation::TransformKey Animation::_interpolate(const Animation::TransformKey &p_a, const Animation::TransformKey &p_b, real_t p_c) const {
	TransformKey ret;
	ret.loc = _interpolate(p_a.loc, p_b.loc, p_c);
	ret.rot = _interpolate(p_a.rot, p_b.rot, p_c);
	ret.scale = _interpolate(p_a.scale, p_b.scale, p_c);

	return ret;
}

Vector3 Animation::_interpolate(const Vector3 &p_a, const Vector3 &p_b, real_t p_c) const {
	return p_a.lerp(p_b, p_c);
}

Quaternion Animation::_interpolate(const Quaternion &p_a, const Quaternion &p_b, real_t p_c) const {
	return p_a.slerp(p_b, p_c);
}

Variant Animation::_interpolate(const Variant &p_a, const Variant &p_b, real_t p_c) const {
	Variant dst;
	Variant::interpolate(p_a, p_b, p_c, dst);
	return dst;
}

real_t Animation::_interpolate(const real_t &p_a, const real_t &p_b, real_t p_c) const {
	return p_a * (1.0 - p_c) + p_b * p_c;
}

Animation::TransformKey Animation::_cubic_interpolate(const Animation::TransformKey &p_pre_a, const Animation::TransformKey &p_a, const Animation::TransformKey &p_b, const Animation::TransformKey &p_post_b, real_t p_c) const {
	Animation::TransformKey tk;

	tk.loc = p_a.loc.cubic_interpolate(p_b.loc, p_pre_a.loc, p_post_b.loc, p_c);
	tk.scale = p_a.scale.cubic_interpolate(p_b.scale, p_pre_a.scale, p_post_b.scale, p_c);
	tk.rot = p_a.rot.cubic_slerp(p_b.rot, p_pre_a.rot, p_post_b.rot, p_c);

	return tk;
}

Vector3 Animation::_cubic_interpolate(const Vector3 &p_pre_a, const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_post_b, real_t p_c) const {
	return p_a.cubic_interpolate(p_b, p_pre_a, p_post_b, p_c);
}

Quaternion Animation::_cubic_interpolate(const Quaternion &p_pre_a, const Quaternion &p_a, const Quaternion &p_b, const Quaternion &p_post_b, real_t p_c) const {
	return p_a.cubic_slerp(p_b, p_pre_a, p_post_b, p_c);
}

Variant Animation::_cubic_interpolate(const Variant &p_pre_a, const Variant &p_a, const Variant &p_b, const Variant &p_post_b, real_t p_c) const {
	Variant::Type type_a = p_a.get_type();
	Variant::Type type_b = p_b.get_type();
	Variant::Type type_pa = p_pre_a.get_type();
	Variant::Type type_pb = p_post_b.get_type();

	//make int and real play along

	uint32_t vformat = 1 << type_a;
	vformat |= 1 << type_b;
	vformat |= 1 << type_pa;
	vformat |= 1 << type_pb;

	if (vformat == ((1 << Variant::INT) | (1 << Variant::FLOAT)) || vformat == (1 << Variant::FLOAT)) {
		//mix of real and int

		real_t p0 = p_pre_a;
		real_t p1 = p_a;
		real_t p2 = p_b;
		real_t p3 = p_post_b;

		real_t t = p_c;
		real_t t2 = t * t;
		real_t t3 = t2 * t;

		return 0.5f * ((p1 * 2.0f) +
							  (-p0 + p2) * t +
							  (2.0f * p0 - 5.0f * p1 + 4 * p2 - p3) * t2 +
							  (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);

	} else if ((vformat & (vformat - 1))) {
		return p_a; //can't interpolate, mix of types
	}

	switch (type_a) {
		case Variant::VECTOR2: {
			Vector2 a = p_a;
			Vector2 b = p_b;
			Vector2 pa = p_pre_a;
			Vector2 pb = p_post_b;

			return a.cubic_interpolate(b, pa, pb, p_c);
		}
		case Variant::RECT2: {
			Rect2 a = p_a;
			Rect2 b = p_b;
			Rect2 pa = p_pre_a;
			Rect2 pb = p_post_b;

			return Rect2(
					a.position.cubic_interpolate(b.position, pa.position, pb.position, p_c),
					a.size.cubic_interpolate(b.size, pa.size, pb.size, p_c));
		}
		case Variant::VECTOR3: {
			Vector3 a = p_a;
			Vector3 b = p_b;
			Vector3 pa = p_pre_a;
			Vector3 pb = p_post_b;

			return a.cubic_interpolate(b, pa, pb, p_c);
		}
		case Variant::QUATERNION: {
			Quaternion a = p_a;
			Quaternion b = p_b;
			Quaternion pa = p_pre_a;
			Quaternion pb = p_post_b;

			return a.cubic_slerp(b, pa, pb, p_c);
		}
		case Variant::AABB: {
			AABB a = p_a;
			AABB b = p_b;
			AABB pa = p_pre_a;
			AABB pb = p_post_b;

			return AABB(
					a.position.cubic_interpolate(b.position, pa.position, pb.position, p_c),
					a.size.cubic_interpolate(b.size, pa.size, pb.size, p_c));
		}
		default: {
			return _interpolate(p_a, p_b, p_c);
		}
	}
}

real_t Animation::_cubic_interpolate(const real_t &p_pre_a, const real_t &p_a, const real_t &p_b, const real_t &p_post_b, real_t p_c) const {
	return _interpolate(p_a, p_b, p_c);
}

template <class T>
T Animation::_interpolate(const Vector<TKey<T>> &p_keys, double p_time, InterpolationType p_interp, bool p_loop_wrap, bool *p_ok, bool p_backward) const {
	int len = _find(p_keys, length) + 1; // try to find last key (there may be more past the end)

	if (len <= 0) {
		// (-1 or -2 returned originally) (plus one above)
		// meaning no keys, or only key time is larger than length
		if (p_ok) {
			*p_ok = false;
		}
		return T();
	} else if (len == 1) { // one key found (0+1), return it

		if (p_ok) {
			*p_ok = true;
		}
		return p_keys[0].value;
	}

	int idx = _find(p_keys, p_time, p_backward);

	ERR_FAIL_COND_V(idx == -2, T());

	bool result = true;
	int next = 0;
	real_t c = 0.0;
	// prepare for all cases of interpolation

	if ((loop_mode == LOOP_LINEAR || loop_mode == LOOP_PINGPONG) && p_loop_wrap) {
		// loop
		if (!p_backward) {
			// no backward
			if (idx >= 0) {
				if (idx < len - 1) {
					next = idx + 1;
					real_t delta = p_keys[next].time - p_keys[idx].time;
					real_t from = p_time - p_keys[idx].time;

					if (Math::is_zero_approx(delta)) {
						c = 0;
					} else {
						c = from / delta;
					}
				} else {
					next = 0;
					real_t delta = (length - p_keys[idx].time) + p_keys[next].time;
					real_t from = p_time - p_keys[idx].time;

					if (Math::is_zero_approx(delta)) {
						c = 0;
					} else {
						c = from / delta;
					}
				}
			} else {
				// on loop, behind first key
				idx = len - 1;
				next = 0;
				real_t endtime = (length - p_keys[idx].time);
				if (endtime < 0) { // may be keys past the end
					endtime = 0;
				}
				real_t delta = endtime + p_keys[next].time;
				real_t from = endtime + p_time;

				if (Math::is_zero_approx(delta)) {
					c = 0;
				} else {
					c = from / delta;
				}
			}
		} else {
			// backward
			if (idx <= len - 1) {
				if (idx > 0) {
					next = idx - 1;
					real_t delta = (length - p_keys[next].time) - (length - p_keys[idx].time);
					real_t from = (length - p_time) - (length - p_keys[idx].time);

					if (Math::is_zero_approx(delta))
						c = 0;
					else
						c = from / delta;
				} else {
					next = len - 1;
					real_t delta = p_keys[idx].time + (length - p_keys[next].time);
					real_t from = (length - p_time) - (length - p_keys[idx].time);

					if (Math::is_zero_approx(delta))
						c = 0;
					else
						c = from / delta;
				}
			} else {
				// on loop, in front of last key
				idx = 0;
				next = len - 1;
				real_t endtime = p_keys[idx].time;
				if (endtime > length) // may be keys past the end
					endtime = length;
				real_t delta = p_keys[next].time - endtime;
				real_t from = p_time - endtime;

				if (Math::is_zero_approx(delta))
					c = 0;
				else
					c = from / delta;
			}
		}
	} else { // no loop
		if (!p_backward) {
			if (idx >= 0) {
				if (idx < len - 1) {
					next = idx + 1;
					real_t delta = p_keys[next].time - p_keys[idx].time;
					real_t from = p_time - p_keys[idx].time;

					if (Math::is_zero_approx(delta)) {
						c = 0;
					} else {
						c = from / delta;
					}
				} else {
					next = idx;
				}
			} else {
				idx = next = 0;
			}
		} else {
			if (idx <= len - 1) {
				if (idx > 0) {
					next = idx - 1;
					real_t delta = (length - p_keys[next].time) - (length - p_keys[idx].time);
					real_t from = (length - p_time) - (length - p_keys[idx].time);

					if (Math::is_zero_approx(delta)) {
						c = 0;
					} else {
						c = from / delta;
					}

				} else {
					next = idx;
				}
			} else {
				idx = next = len - 1;
			}
		}
	}

	if (p_ok) {
		*p_ok = result;
	}
	if (!result) {
		return T();
	}

	real_t tr = p_keys[idx].transition;

	if (tr == 0 || idx == next) {
		// don't interpolate if not needed
		return p_keys[idx].value;
	}

	if (tr != 1.0) {
		c = Math::ease(c, tr);
	}

	switch (p_interp) {
		case INTERPOLATION_NEAREST: {
			return p_keys[idx].value;
		} break;
		case INTERPOLATION_LINEAR: {
			return _interpolate(p_keys[idx].value, p_keys[next].value, c);
		} break;
		case INTERPOLATION_CUBIC: {
			int pre = idx - 1;
			if (pre < 0) {
				pre = 0;
			}
			int post = next + 1;
			if (post >= len) {
				post = next;
			}

			return _cubic_interpolate(p_keys[pre].value, p_keys[idx].value, p_keys[next].value, p_keys[post].value, c);

		} break;
		default:
			return p_keys[idx].value;
	}

	// do a barrel roll
}

Error Animation::transform_track_interpolate(int p_track, double p_time, Vector3 *r_loc, Quaternion *r_rot, Vector3 *r_scale, bool p_backward) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_TRANSFORM3D, ERR_INVALID_PARAMETER);

	TransformTrack *tt = static_cast<TransformTrack *>(t);

	bool ok = false;

	TransformKey tk = _interpolate(tt->transforms, p_time, tt->interpolation, tt->loop_wrap, &ok, p_backward);

	if (!ok) {
		return ERR_UNAVAILABLE;
	}

	if (r_loc) {
		*r_loc = tk.loc;
	}

	if (r_rot) {
		*r_rot = tk.rot;
	}

	if (r_scale) {
		*r_scale = tk.scale;
	}

	return OK;
}

Variant Animation::value_track_interpolate(int p_track, double p_time) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_VALUE, Variant());
	ValueTrack *vt = static_cast<ValueTrack *>(t);

	bool ok = false;

	Variant res = _interpolate(vt->values, p_time, (vt->update_mode == UPDATE_CONTINUOUS || vt->update_mode == UPDATE_CAPTURE) ? vt->interpolation : INTERPOLATION_NEAREST, vt->loop_wrap, &ok);

	if (ok) {
		return res;
	}

	return Variant();
}

void Animation::_value_track_get_key_indices_in_range(const ValueTrack *vt, double from_time, double to_time, List<int> *p_indices) const {
	if (from_time != length && to_time == length) {
		to_time = length * 1.001; //include a little more if at the end
	}
	int to = _find(vt->values, to_time);

	if (to >= 0 && from_time == to_time && vt->values[to].time == from_time) {
		//find exact (0 delta), return if found
		p_indices->push_back(to);
		return;
	}
	// can't really send the events == time, will be sent in the next frame.
	// if event>=len then it will probably never be requested by the anim player.

	if (to >= 0 && vt->values[to].time >= to_time) {
		to--;
	}

	if (to < 0) {
		return; // not bother
	}

	int from = _find(vt->values, from_time);

	// position in the right first event.+
	if (from < 0 || vt->values[from].time < from_time) {
		from++;
	}

	int max = vt->values.size();

	for (int i = from; i <= to; i++) {
		ERR_CONTINUE(i < 0 || i >= max); // shouldn't happen
		p_indices->push_back(i);
	}
}

void Animation::value_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices, int p_pingponged) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_VALUE);

	ValueTrack *vt = static_cast<ValueTrack *>(t);

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	switch (loop_mode) {
		case LOOP_NONE: {
			if (from_time < 0) {
				from_time = 0;
			}
			if (from_time > length) {
				from_time = length;
			}

			if (to_time < 0) {
				to_time = 0;
			}
			if (to_time > length) {
				to_time = length;
			}
		} break;
		case LOOP_LINEAR: {
			from_time = Math::fposmod(from_time, length);
			to_time = Math::fposmod(to_time, length);

			if (from_time > to_time) {
				// handle loop by splitting
				_value_track_get_key_indices_in_range(vt, from_time, length, p_indices);
				_value_track_get_key_indices_in_range(vt, 0, to_time, p_indices);
				return;
			}
		} break;
		case LOOP_PINGPONG: {
			from_time = Math::pingpong(from_time, length);
			to_time = Math::pingpong(to_time, length);

			if (p_pingponged == -1) {
				// handle loop by splitting
				_value_track_get_key_indices_in_range(vt, 0, from_time, p_indices);
				_value_track_get_key_indices_in_range(vt, 0, to_time, p_indices);
				return;
			}
			if (p_pingponged == 1) {
				// handle loop by splitting
				_value_track_get_key_indices_in_range(vt, from_time, length, p_indices);
				_value_track_get_key_indices_in_range(vt, to_time, length, p_indices);
				return;
			}
		} break;
	}

	_value_track_get_key_indices_in_range(vt, from_time, to_time, p_indices);
}

void Animation::value_track_set_update_mode(int p_track, UpdateMode p_mode) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_VALUE);
	ERR_FAIL_INDEX((int)p_mode, 4);

	ValueTrack *vt = static_cast<ValueTrack *>(t);
	vt->update_mode = p_mode;
}

Animation::UpdateMode Animation::value_track_get_update_mode(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), UPDATE_CONTINUOUS);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_VALUE, UPDATE_CONTINUOUS);

	ValueTrack *vt = static_cast<ValueTrack *>(t);
	return vt->update_mode;
}

template <class T>
void Animation::_track_get_key_indices_in_range(const Vector<T> &p_array, double from_time, double to_time, List<int> *p_indices) const {
	if (from_time != length && to_time == length) {
		to_time = length * 1.01; //include a little more if at the end
	}

	int to = _find(p_array, to_time);

	// can't really send the events == time, will be sent in the next frame.
	// if event>=len then it will probably never be requested by the anim player.

	if (to >= 0 && p_array[to].time >= to_time) {
		to--;
	}

	if (to < 0) {
		return; // not bother
	}

	int from = _find(p_array, from_time);

	// position in the right first event.+
	if (from < 0 || p_array[from].time < from_time) {
		from++;
	}

	int max = p_array.size();

	for (int i = from; i <= to; i++) {
		ERR_CONTINUE(i < 0 || i >= max); // shouldn't happen
		p_indices->push_back(i);
	}
}

void Animation::track_get_key_indices_in_range(int p_track, double p_time, double p_delta, List<int> *p_indices, int p_pingponged) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	const Track *t = tracks[p_track];

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	switch (loop_mode) {
		case LOOP_NONE: {
			if (from_time < 0) {
				from_time = 0;
			}
			if (from_time > length) {
				from_time = length;
			}

			if (to_time < 0) {
				to_time = 0;
			}
			if (to_time > length) {
				to_time = length;
			}
		} break;
		case LOOP_LINEAR: {
			if (from_time > length || from_time < 0) {
				from_time = Math::fposmod(from_time, length);
			}
			if (to_time > length || to_time < 0) {
				to_time = Math::fposmod(to_time, length);
			}

			if (from_time > to_time) {
				// handle loop by splitting
				switch (t->type) {
					case TYPE_TRANSFORM3D: {
						const TransformTrack *tt = static_cast<const TransformTrack *>(t);
						_track_get_key_indices_in_range(tt->transforms, from_time, length, p_indices);
						_track_get_key_indices_in_range(tt->transforms, 0, to_time, p_indices);
					} break;
					case TYPE_VALUE: {
						const ValueTrack *vt = static_cast<const ValueTrack *>(t);
						_track_get_key_indices_in_range(vt->values, from_time, length, p_indices);
						_track_get_key_indices_in_range(vt->values, 0, to_time, p_indices);
					} break;
					case TYPE_METHOD: {
						const MethodTrack *mt = static_cast<const MethodTrack *>(t);
						_track_get_key_indices_in_range(mt->methods, from_time, length, p_indices);
						_track_get_key_indices_in_range(mt->methods, 0, to_time, p_indices);
					} break;
					case TYPE_BEZIER: {
						const BezierTrack *bz = static_cast<const BezierTrack *>(t);
						_track_get_key_indices_in_range(bz->values, from_time, length, p_indices);
						_track_get_key_indices_in_range(bz->values, 0, to_time, p_indices);
					} break;
					case TYPE_AUDIO: {
						const AudioTrack *ad = static_cast<const AudioTrack *>(t);
						_track_get_key_indices_in_range(ad->values, from_time, length, p_indices);
						_track_get_key_indices_in_range(ad->values, 0, to_time, p_indices);
					} break;
					case TYPE_ANIMATION: {
						const AnimationTrack *an = static_cast<const AnimationTrack *>(t);
						_track_get_key_indices_in_range(an->values, from_time, length, p_indices);
						_track_get_key_indices_in_range(an->values, 0, to_time, p_indices);
					} break;
				}
				return;
			}
		} break;
		case LOOP_PINGPONG: {
			if (from_time > length || from_time < 0) {
				from_time = Math::pingpong(from_time, length);
			}
			if (to_time > length || to_time < 0) {
				to_time = Math::pingpong(to_time, length);
			}

			if ((int)Math::floor(abs(p_delta) / length) % 2 == 0) {
				if (p_pingponged == -1) {
					// handle loop by splitting
					switch (t->type) {
						case TYPE_TRANSFORM3D: {
							const TransformTrack *tt = static_cast<const TransformTrack *>(t);
							_track_get_key_indices_in_range(tt->transforms, 0, from_time, p_indices);
							_track_get_key_indices_in_range(tt->transforms, 0, to_time, p_indices);
						} break;
						case TYPE_VALUE: {
							const ValueTrack *vt = static_cast<const ValueTrack *>(t);
							_track_get_key_indices_in_range(vt->values, 0, from_time, p_indices);
							_track_get_key_indices_in_range(vt->values, 0, to_time, p_indices);
						} break;
						case TYPE_METHOD: {
							const MethodTrack *mt = static_cast<const MethodTrack *>(t);
							_track_get_key_indices_in_range(mt->methods, 0, from_time, p_indices);
							_track_get_key_indices_in_range(mt->methods, 0, to_time, p_indices);
						} break;
						case TYPE_BEZIER: {
							const BezierTrack *bz = static_cast<const BezierTrack *>(t);
							_track_get_key_indices_in_range(bz->values, 0, from_time, p_indices);
							_track_get_key_indices_in_range(bz->values, 0, to_time, p_indices);
						} break;
						case TYPE_AUDIO: {
							const AudioTrack *ad = static_cast<const AudioTrack *>(t);
							_track_get_key_indices_in_range(ad->values, 0, from_time, p_indices);
							_track_get_key_indices_in_range(ad->values, 0, to_time, p_indices);
						} break;
						case TYPE_ANIMATION: {
							const AnimationTrack *an = static_cast<const AnimationTrack *>(t);
							_track_get_key_indices_in_range(an->values, 0, from_time, p_indices);
							_track_get_key_indices_in_range(an->values, 0, to_time, p_indices);
						} break;
					}
					return;
				}
				if (p_pingponged == 1) {
					// handle loop by splitting
					switch (t->type) {
						case TYPE_TRANSFORM3D: {
							const TransformTrack *tt = static_cast<const TransformTrack *>(t);
							_track_get_key_indices_in_range(tt->transforms, from_time, length, p_indices);
							_track_get_key_indices_in_range(tt->transforms, to_time, length, p_indices);
						} break;
						case TYPE_VALUE: {
							const ValueTrack *vt = static_cast<const ValueTrack *>(t);
							_track_get_key_indices_in_range(vt->values, from_time, length, p_indices);
							_track_get_key_indices_in_range(vt->values, to_time, length, p_indices);
						} break;
						case TYPE_METHOD: {
							const MethodTrack *mt = static_cast<const MethodTrack *>(t);
							_track_get_key_indices_in_range(mt->methods, from_time, length, p_indices);
							_track_get_key_indices_in_range(mt->methods, to_time, length, p_indices);
						} break;
						case TYPE_BEZIER: {
							const BezierTrack *bz = static_cast<const BezierTrack *>(t);
							_track_get_key_indices_in_range(bz->values, from_time, length, p_indices);
							_track_get_key_indices_in_range(bz->values, to_time, length, p_indices);
						} break;
						case TYPE_AUDIO: {
							const AudioTrack *ad = static_cast<const AudioTrack *>(t);
							_track_get_key_indices_in_range(ad->values, from_time, length, p_indices);
							_track_get_key_indices_in_range(ad->values, to_time, length, p_indices);
						} break;
						case TYPE_ANIMATION: {
							const AnimationTrack *an = static_cast<const AnimationTrack *>(t);
							_track_get_key_indices_in_range(an->values, from_time, length, p_indices);
							_track_get_key_indices_in_range(an->values, to_time, length, p_indices);
						} break;
					}
					return;
				}
			}
		} break;
	}

	switch (t->type) {
		case TYPE_TRANSFORM3D: {
			const TransformTrack *tt = static_cast<const TransformTrack *>(t);
			_track_get_key_indices_in_range(tt->transforms, from_time, to_time, p_indices);
		} break;
		case TYPE_VALUE: {
			const ValueTrack *vt = static_cast<const ValueTrack *>(t);
			_track_get_key_indices_in_range(vt->values, from_time, to_time, p_indices);
		} break;
		case TYPE_METHOD: {
			const MethodTrack *mt = static_cast<const MethodTrack *>(t);
			_track_get_key_indices_in_range(mt->methods, from_time, to_time, p_indices);
		} break;
		case TYPE_BEZIER: {
			const BezierTrack *bz = static_cast<const BezierTrack *>(t);
			_track_get_key_indices_in_range(bz->values, from_time, to_time, p_indices);
		} break;
		case TYPE_AUDIO: {
			const AudioTrack *ad = static_cast<const AudioTrack *>(t);
			_track_get_key_indices_in_range(ad->values, from_time, to_time, p_indices);
		} break;
		case TYPE_ANIMATION: {
			const AnimationTrack *an = static_cast<const AnimationTrack *>(t);
			_track_get_key_indices_in_range(an->values, from_time, to_time, p_indices);
		} break;
	}
}

void Animation::_method_track_get_key_indices_in_range(const MethodTrack *mt, double from_time, double to_time, List<int> *p_indices) const {
	if (from_time != length && to_time == length) {
		to_time = length * 1.01; //include a little more if at the end
	}

	int to = _find(mt->methods, to_time);

	// can't really send the events == time, will be sent in the next frame.
	// if event>=len then it will probably never be requested by the anim player.

	if (to >= 0 && mt->methods[to].time >= to_time) {
		to--;
	}

	if (to < 0) {
		return; // not bother
	}

	int from = _find(mt->methods, from_time);

	// position in the right first event.+
	if (from < 0 || mt->methods[from].time < from_time) {
		from++;
	}

	int max = mt->methods.size();

	for (int i = from; i <= to; i++) {
		ERR_CONTINUE(i < 0 || i >= max); // shouldn't happen
		p_indices->push_back(i);
	}
}

void Animation::method_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices, int p_pingponged) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_METHOD);

	MethodTrack *mt = static_cast<MethodTrack *>(t);

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	switch (loop_mode) {
		case LOOP_NONE: {
			if (from_time < 0) {
				from_time = 0;
			}
			if (from_time > length) {
				from_time = length;
			}

			if (to_time < 0) {
				to_time = 0;
			}
			if (to_time > length) {
				to_time = length;
			}
		} break;
		case LOOP_LINEAR: {
			if (from_time > length || from_time < 0) {
				from_time = Math::fposmod(from_time, length);
			}
			if (to_time > length || to_time < 0) {
				to_time = Math::fposmod(to_time, length);
			}

			if (from_time > to_time) {
				// handle loop by splitting
				_method_track_get_key_indices_in_range(mt, from_time, length, p_indices);
				_method_track_get_key_indices_in_range(mt, 0, to_time, p_indices);
				return;
			}
		} break;
		case LOOP_PINGPONG: {
			if (from_time > length || from_time < 0) {
				from_time = Math::pingpong(from_time, length);
			}
			if (to_time > length || to_time < 0) {
				to_time = Math::pingpong(to_time, length);
			}

			if (p_pingponged == -1) {
				_method_track_get_key_indices_in_range(mt, 0, from_time, p_indices);
				_method_track_get_key_indices_in_range(mt, 0, to_time, p_indices);
				return;
			}
			if (p_pingponged == 1) {
				_method_track_get_key_indices_in_range(mt, from_time, length, p_indices);
				_method_track_get_key_indices_in_range(mt, to_time, length, p_indices);
				return;
			}
		} break;
		default:
			break;
	}

	_method_track_get_key_indices_in_range(mt, from_time, to_time, p_indices);
}

Vector<Variant> Animation::method_track_get_params(int p_track, int p_key_idx) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), Vector<Variant>());
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_METHOD, Vector<Variant>());

	MethodTrack *pm = static_cast<MethodTrack *>(t);

	ERR_FAIL_INDEX_V(p_key_idx, pm->methods.size(), Vector<Variant>());

	const MethodKey &mk = pm->methods[p_key_idx];

	return mk.params;
}

StringName Animation::method_track_get_name(int p_track, int p_key_idx) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), StringName());
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_METHOD, StringName());

	MethodTrack *pm = static_cast<MethodTrack *>(t);

	ERR_FAIL_INDEX_V(p_key_idx, pm->methods.size(), StringName());

	return pm->methods[p_key_idx].method;
}

int Animation::bezier_track_insert_key(int p_track, double p_time, real_t p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BEZIER, -1);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	TKey<BezierKey> k;
	k.time = p_time;
	k.value.value = p_value;
	k.value.in_handle = p_in_handle;
	if (k.value.in_handle.x > 0) {
		k.value.in_handle.x = 0;
	}
	k.value.out_handle = p_out_handle;
	if (k.value.out_handle.x < 0) {
		k.value.out_handle.x = 0;
	}

	int key = _insert(p_time, bt->values, k);

	emit_changed();

	return key;
}

void Animation::bezier_track_set_key_value(int p_track, int p_index, real_t p_value) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	bt->values.write[p_index].value.value = p_value;
	emit_changed();
}

void Animation::bezier_track_set_key_in_handle(int p_track, int p_index, const Vector2 &p_handle) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	bt->values.write[p_index].value.in_handle = p_handle;
	if (bt->values[p_index].value.in_handle.x > 0) {
		bt->values.write[p_index].value.in_handle.x = 0;
	}
	emit_changed();
}

void Animation::bezier_track_set_key_out_handle(int p_track, int p_index, const Vector2 &p_handle) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	bt->values.write[p_index].value.out_handle = p_handle;
	if (bt->values[p_index].value.out_handle.x < 0) {
		bt->values.write[p_index].value.out_handle.x = 0;
	}
	emit_changed();
}

real_t Animation::bezier_track_get_key_value(int p_track, int p_index) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BEZIER, 0);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX_V(p_index, bt->values.size(), 0);

	return bt->values[p_index].value.value;
}

Vector2 Animation::bezier_track_get_key_in_handle(int p_track, int p_index) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), Vector2());
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BEZIER, Vector2());

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX_V(p_index, bt->values.size(), Vector2());

	return bt->values[p_index].value.in_handle;
}

Vector2 Animation::bezier_track_get_key_out_handle(int p_track, int p_index) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), Vector2());
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BEZIER, Vector2());

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX_V(p_index, bt->values.size(), Vector2());

	return bt->values[p_index].value.out_handle;
}

static _FORCE_INLINE_ Vector2 _bezier_interp(real_t t, const Vector2 &start, const Vector2 &control_1, const Vector2 &control_2, const Vector2 &end) {
	/* Formula from Wikipedia article on Bezier curves. */
	real_t omt = (1.0 - t);
	real_t omt2 = omt * omt;
	real_t omt3 = omt2 * omt;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
}

real_t Animation::bezier_track_interpolate(int p_track, double p_time) const {
	//this uses a different interpolation scheme
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	Track *track = tracks[p_track];
	ERR_FAIL_COND_V(track->type != TYPE_BEZIER, 0);

	BezierTrack *bt = static_cast<BezierTrack *>(track);

	int len = _find(bt->values, length) + 1; // try to find last key (there may be more past the end)

	if (len <= 0) {
		// (-1 or -2 returned originally) (plus one above)
		return 0;
	} else if (len == 1) { // one key found (0+1), return it
		return bt->values[0].value.value;
	}

	int idx = _find(bt->values, p_time);

	ERR_FAIL_COND_V(idx == -2, 0);

	//there really is no looping interpolation on bezier

	if (idx < 0) {
		return bt->values[0].value.value;
	}

	if (idx >= bt->values.size() - 1) {
		return bt->values[bt->values.size() - 1].value.value;
	}

	double t = p_time - bt->values[idx].time;

	int iterations = 10;

	real_t duration = bt->values[idx + 1].time - bt->values[idx].time; // time duration between our two keyframes
	real_t low = 0.0; // 0% of the current animation segment
	real_t high = 1.0; // 100% of the current animation segment
	real_t middle;

	Vector2 start(0, bt->values[idx].value.value);
	Vector2 start_out = start + bt->values[idx].value.out_handle;
	Vector2 end(duration, bt->values[idx + 1].value.value);
	Vector2 end_in = end + bt->values[idx + 1].value.in_handle;

	//narrow high and low as much as possible
	for (int i = 0; i < iterations; i++) {
		middle = (low + high) / 2;

		Vector2 interp = _bezier_interp(middle, start, start_out, end_in, end);

		if (interp.x < t) {
			low = middle;
		} else {
			high = middle;
		}
	}

	//interpolate the result:
	Vector2 low_pos = _bezier_interp(low, start, start_out, end_in, end);
	Vector2 high_pos = _bezier_interp(high, start, start_out, end_in, end);
	real_t c = (t - low_pos.x) / (high_pos.x - low_pos.x);

	return low_pos.lerp(high_pos, c).y;
}

int Animation::audio_track_insert_key(int p_track, double p_time, const RES &p_stream, real_t p_start_offset, real_t p_end_offset) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_AUDIO, -1);

	AudioTrack *at = static_cast<AudioTrack *>(t);

	TKey<AudioKey> k;
	k.time = p_time;
	k.value.stream = p_stream;
	k.value.start_offset = p_start_offset;
	if (k.value.start_offset < 0) {
		k.value.start_offset = 0;
	}
	k.value.end_offset = p_end_offset;
	if (k.value.end_offset < 0) {
		k.value.end_offset = 0;
	}

	int key = _insert(p_time, at->values, k);

	emit_changed();

	return key;
}

void Animation::audio_track_set_key_stream(int p_track, int p_key, const RES &p_stream) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_AUDIO);

	AudioTrack *at = static_cast<AudioTrack *>(t);

	ERR_FAIL_INDEX(p_key, at->values.size());

	at->values.write[p_key].value.stream = p_stream;

	emit_changed();
}

void Animation::audio_track_set_key_start_offset(int p_track, int p_key, real_t p_offset) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_AUDIO);

	AudioTrack *at = static_cast<AudioTrack *>(t);

	ERR_FAIL_INDEX(p_key, at->values.size());

	if (p_offset < 0) {
		p_offset = 0;
	}

	at->values.write[p_key].value.start_offset = p_offset;

	emit_changed();
}

void Animation::audio_track_set_key_end_offset(int p_track, int p_key, real_t p_offset) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_AUDIO);

	AudioTrack *at = static_cast<AudioTrack *>(t);

	ERR_FAIL_INDEX(p_key, at->values.size());

	if (p_offset < 0) {
		p_offset = 0;
	}

	at->values.write[p_key].value.end_offset = p_offset;

	emit_changed();
}

RES Animation::audio_track_get_key_stream(int p_track, int p_key) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), RES());
	const Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_AUDIO, RES());

	const AudioTrack *at = static_cast<const AudioTrack *>(t);

	ERR_FAIL_INDEX_V(p_key, at->values.size(), RES());

	return at->values[p_key].value.stream;
}

real_t Animation::audio_track_get_key_start_offset(int p_track, int p_key) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	const Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_AUDIO, 0);

	const AudioTrack *at = static_cast<const AudioTrack *>(t);

	ERR_FAIL_INDEX_V(p_key, at->values.size(), 0);

	return at->values[p_key].value.start_offset;
}

real_t Animation::audio_track_get_key_end_offset(int p_track, int p_key) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	const Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_AUDIO, 0);

	const AudioTrack *at = static_cast<const AudioTrack *>(t);

	ERR_FAIL_INDEX_V(p_key, at->values.size(), 0);

	return at->values[p_key].value.end_offset;
}

//

int Animation::animation_track_insert_key(int p_track, double p_time, const StringName &p_animation) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_ANIMATION, -1);

	AnimationTrack *at = static_cast<AnimationTrack *>(t);

	TKey<StringName> k;
	k.time = p_time;
	k.value = p_animation;

	int key = _insert(p_time, at->values, k);

	emit_changed();

	return key;
}

void Animation::animation_track_set_key_animation(int p_track, int p_key, const StringName &p_animation) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_ANIMATION);

	AnimationTrack *at = static_cast<AnimationTrack *>(t);

	ERR_FAIL_INDEX(p_key, at->values.size());

	at->values.write[p_key].value = p_animation;

	emit_changed();
}

StringName Animation::animation_track_get_key_animation(int p_track, int p_key) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), StringName());
	const Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_ANIMATION, StringName());

	const AnimationTrack *at = static_cast<const AnimationTrack *>(t);

	ERR_FAIL_INDEX_V(p_key, at->values.size(), StringName());

	return at->values[p_key].value;
}

void Animation::set_length(real_t p_length) {
	if (p_length < ANIM_MIN_LENGTH) {
		p_length = ANIM_MIN_LENGTH;
	}
	length = p_length;
	emit_changed();
}

real_t Animation::get_length() const {
	return length;
}

void Animation::set_loop_mode(Animation::LoopMode p_loop_mode) {
	loop_mode = p_loop_mode;
	emit_changed();
}

Animation::LoopMode Animation::get_loop_mode() const {
	return loop_mode;
}

void Animation::track_set_imported(int p_track, bool p_imported) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	tracks[p_track]->imported = p_imported;
}

bool Animation::track_is_imported(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), false);
	return tracks[p_track]->imported;
}

void Animation::track_set_enabled(int p_track, bool p_enabled) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	tracks[p_track]->enabled = p_enabled;
	emit_changed();
}

bool Animation::track_is_enabled(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), false);
	return tracks[p_track]->enabled;
}

void Animation::track_move_up(int p_track) {
	if (p_track >= 0 && p_track < (tracks.size() - 1)) {
		SWAP(tracks.write[p_track], tracks.write[p_track + 1]);
	}

	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

void Animation::track_move_down(int p_track) {
	if (p_track > 0 && p_track < tracks.size()) {
		SWAP(tracks.write[p_track], tracks.write[p_track - 1]);
	}

	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

void Animation::track_move_to(int p_track, int p_to_index) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	ERR_FAIL_INDEX(p_to_index, tracks.size() + 1);
	if (p_track == p_to_index || p_track == p_to_index - 1) {
		return;
	}

	Track *track = tracks.get(p_track);
	tracks.remove(p_track);
	// Take into account that the position of the tracks that come after the one removed will change.
	tracks.insert(p_to_index > p_track ? p_to_index - 1 : p_to_index, track);

	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

void Animation::track_swap(int p_track, int p_with_track) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	ERR_FAIL_INDEX(p_with_track, tracks.size());
	if (p_track == p_with_track) {
		return;
	}
	SWAP(tracks.write[p_track], tracks.write[p_with_track]);

	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

void Animation::set_step(real_t p_step) {
	step = p_step;
	emit_changed();
}

real_t Animation::get_step() const {
	return step;
}

void Animation::copy_track(int p_track, Ref<Animation> p_to_animation) {
	ERR_FAIL_COND(p_to_animation.is_null());
	ERR_FAIL_INDEX(p_track, get_track_count());
	int dst_track = p_to_animation->get_track_count();
	p_to_animation->add_track(track_get_type(p_track));

	p_to_animation->track_set_path(dst_track, track_get_path(p_track));
	p_to_animation->track_set_imported(dst_track, track_is_imported(p_track));
	p_to_animation->track_set_enabled(dst_track, track_is_enabled(p_track));
	p_to_animation->track_set_interpolation_type(dst_track, track_get_interpolation_type(p_track));
	p_to_animation->track_set_interpolation_loop_wrap(dst_track, track_get_interpolation_loop_wrap(p_track));
	if (track_get_type(p_track) == TYPE_VALUE) {
		p_to_animation->value_track_set_update_mode(dst_track, value_track_get_update_mode(p_track));
	}

	for (int i = 0; i < track_get_key_count(p_track); i++) {
		p_to_animation->track_insert_key(dst_track, track_get_key_time(p_track, i), track_get_key_value(p_track, i), track_get_key_transition(p_track, i));
	}
}

void Animation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_track", "type", "at_position"), &Animation::add_track, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_track", "track_idx"), &Animation::remove_track);
	ClassDB::bind_method(D_METHOD("get_track_count"), &Animation::get_track_count);
	ClassDB::bind_method(D_METHOD("track_get_type", "track_idx"), &Animation::track_get_type);
	ClassDB::bind_method(D_METHOD("track_get_path", "track_idx"), &Animation::track_get_path);
	ClassDB::bind_method(D_METHOD("track_set_path", "track_idx", "path"), &Animation::track_set_path);
	ClassDB::bind_method(D_METHOD("find_track", "path"), &Animation::find_track);

	ClassDB::bind_method(D_METHOD("track_move_up", "track_idx"), &Animation::track_move_up);
	ClassDB::bind_method(D_METHOD("track_move_down", "track_idx"), &Animation::track_move_down);
	ClassDB::bind_method(D_METHOD("track_move_to", "track_idx", "to_idx"), &Animation::track_move_to);
	ClassDB::bind_method(D_METHOD("track_swap", "track_idx", "with_idx"), &Animation::track_swap);

	ClassDB::bind_method(D_METHOD("track_set_imported", "track_idx", "imported"), &Animation::track_set_imported);
	ClassDB::bind_method(D_METHOD("track_is_imported", "track_idx"), &Animation::track_is_imported);

	ClassDB::bind_method(D_METHOD("track_set_enabled", "track_idx", "enabled"), &Animation::track_set_enabled);
	ClassDB::bind_method(D_METHOD("track_is_enabled", "track_idx"), &Animation::track_is_enabled);

	ClassDB::bind_method(D_METHOD("transform_track_insert_key", "track_idx", "time", "location", "rotation", "scale"), &Animation::transform_track_insert_key);
	ClassDB::bind_method(D_METHOD("track_insert_key", "track_idx", "time", "key", "transition"), &Animation::track_insert_key, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("track_remove_key", "track_idx", "key_idx"), &Animation::track_remove_key);
	ClassDB::bind_method(D_METHOD("track_remove_key_at_time", "track_idx", "time"), &Animation::track_remove_key_at_time);
	ClassDB::bind_method(D_METHOD("track_set_key_value", "track_idx", "key", "value"), &Animation::track_set_key_value);
	ClassDB::bind_method(D_METHOD("track_set_key_transition", "track_idx", "key_idx", "transition"), &Animation::track_set_key_transition);
	ClassDB::bind_method(D_METHOD("track_set_key_time", "track_idx", "key_idx", "time"), &Animation::track_set_key_time);
	ClassDB::bind_method(D_METHOD("track_get_key_transition", "track_idx", "key_idx"), &Animation::track_get_key_transition);

	ClassDB::bind_method(D_METHOD("track_get_key_count", "track_idx"), &Animation::track_get_key_count);
	ClassDB::bind_method(D_METHOD("track_get_key_value", "track_idx", "key_idx"), &Animation::track_get_key_value);
	ClassDB::bind_method(D_METHOD("track_get_key_time", "track_idx", "key_idx"), &Animation::track_get_key_time);
	ClassDB::bind_method(D_METHOD("track_find_key", "track_idx", "time", "exact"), &Animation::track_find_key, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("track_set_interpolation_type", "track_idx", "interpolation"), &Animation::track_set_interpolation_type);
	ClassDB::bind_method(D_METHOD("track_get_interpolation_type", "track_idx"), &Animation::track_get_interpolation_type);

	ClassDB::bind_method(D_METHOD("track_set_interpolation_loop_wrap", "track_idx", "interpolation"), &Animation::track_set_interpolation_loop_wrap);
	ClassDB::bind_method(D_METHOD("track_get_interpolation_loop_wrap", "track_idx"), &Animation::track_get_interpolation_loop_wrap);

	ClassDB::bind_method(D_METHOD("transform_track_interpolate", "track_idx", "time_sec", "is_backward"), &Animation::_transform_track_interpolate, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("value_track_set_update_mode", "track_idx", "mode"), &Animation::value_track_set_update_mode);
	ClassDB::bind_method(D_METHOD("value_track_get_update_mode", "track_idx"), &Animation::value_track_get_update_mode);

	ClassDB::bind_method(D_METHOD("value_track_get_key_indices", "track_idx", "time_sec", "delta"), &Animation::_value_track_get_key_indices);
	ClassDB::bind_method(D_METHOD("value_track_interpolate", "track_idx", "time_sec"), &Animation::value_track_interpolate);

	ClassDB::bind_method(D_METHOD("method_track_get_key_indices", "track_idx", "time_sec", "delta"), &Animation::_method_track_get_key_indices);
	ClassDB::bind_method(D_METHOD("method_track_get_name", "track_idx", "key_idx"), &Animation::method_track_get_name);
	ClassDB::bind_method(D_METHOD("method_track_get_params", "track_idx", "key_idx"), &Animation::method_track_get_params);

	ClassDB::bind_method(D_METHOD("bezier_track_insert_key", "track_idx", "time", "value", "in_handle", "out_handle"), &Animation::bezier_track_insert_key, DEFVAL(Vector2()), DEFVAL(Vector2()));

	ClassDB::bind_method(D_METHOD("bezier_track_set_key_value", "track_idx", "key_idx", "value"), &Animation::bezier_track_set_key_value);
	ClassDB::bind_method(D_METHOD("bezier_track_set_key_in_handle", "track_idx", "key_idx", "in_handle"), &Animation::bezier_track_set_key_in_handle);
	ClassDB::bind_method(D_METHOD("bezier_track_set_key_out_handle", "track_idx", "key_idx", "out_handle"), &Animation::bezier_track_set_key_out_handle);

	ClassDB::bind_method(D_METHOD("bezier_track_get_key_value", "track_idx", "key_idx"), &Animation::bezier_track_get_key_value);
	ClassDB::bind_method(D_METHOD("bezier_track_get_key_in_handle", "track_idx", "key_idx"), &Animation::bezier_track_get_key_in_handle);
	ClassDB::bind_method(D_METHOD("bezier_track_get_key_out_handle", "track_idx", "key_idx"), &Animation::bezier_track_get_key_out_handle);

	ClassDB::bind_method(D_METHOD("bezier_track_interpolate", "track_idx", "time"), &Animation::bezier_track_interpolate);

	ClassDB::bind_method(D_METHOD("audio_track_insert_key", "track_idx", "time", "stream", "start_offset", "end_offset"), &Animation::audio_track_insert_key, DEFVAL(0), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("audio_track_set_key_stream", "track_idx", "key_idx", "stream"), &Animation::audio_track_set_key_stream);
	ClassDB::bind_method(D_METHOD("audio_track_set_key_start_offset", "track_idx", "key_idx", "offset"), &Animation::audio_track_set_key_start_offset);
	ClassDB::bind_method(D_METHOD("audio_track_set_key_end_offset", "track_idx", "key_idx", "offset"), &Animation::audio_track_set_key_end_offset);
	ClassDB::bind_method(D_METHOD("audio_track_get_key_stream", "track_idx", "key_idx"), &Animation::audio_track_get_key_stream);
	ClassDB::bind_method(D_METHOD("audio_track_get_key_start_offset", "track_idx", "key_idx"), &Animation::audio_track_get_key_start_offset);
	ClassDB::bind_method(D_METHOD("audio_track_get_key_end_offset", "track_idx", "key_idx"), &Animation::audio_track_get_key_end_offset);

	ClassDB::bind_method(D_METHOD("animation_track_insert_key", "track_idx", "time", "animation"), &Animation::animation_track_insert_key);
	ClassDB::bind_method(D_METHOD("animation_track_set_key_animation", "track_idx", "key_idx", "animation"), &Animation::animation_track_set_key_animation);
	ClassDB::bind_method(D_METHOD("animation_track_get_key_animation", "track_idx", "key_idx"), &Animation::animation_track_get_key_animation);

	ClassDB::bind_method(D_METHOD("set_length", "time_sec"), &Animation::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &Animation::get_length);

	ClassDB::bind_method(D_METHOD("set_loop_mode", "loop_mode"), &Animation::set_loop_mode);
	ClassDB::bind_method(D_METHOD("get_loop_mode"), &Animation::get_loop_mode);

	ClassDB::bind_method(D_METHOD("set_step", "size_sec"), &Animation::set_step);
	ClassDB::bind_method(D_METHOD("get_step"), &Animation::get_step);

	ClassDB::bind_method(D_METHOD("clear"), &Animation::clear);
	ClassDB::bind_method(D_METHOD("copy_track", "track_idx", "to_animation"), &Animation::copy_track);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0.001,99999,0.001"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step", PROPERTY_HINT_RANGE, "0,4096,0.001"), "set_step", "get_step");

	ADD_SIGNAL(MethodInfo("tracks_changed"));

	BIND_ENUM_CONSTANT(TYPE_VALUE);
	BIND_ENUM_CONSTANT(TYPE_TRANSFORM3D);
	BIND_ENUM_CONSTANT(TYPE_METHOD);
	BIND_ENUM_CONSTANT(TYPE_BEZIER);
	BIND_ENUM_CONSTANT(TYPE_AUDIO);
	BIND_ENUM_CONSTANT(TYPE_ANIMATION);

	BIND_ENUM_CONSTANT(INTERPOLATION_NEAREST);
	BIND_ENUM_CONSTANT(INTERPOLATION_LINEAR);
	BIND_ENUM_CONSTANT(INTERPOLATION_CUBIC);

	BIND_ENUM_CONSTANT(UPDATE_CONTINUOUS);
	BIND_ENUM_CONSTANT(UPDATE_DISCRETE);
	BIND_ENUM_CONSTANT(UPDATE_TRIGGER);
	BIND_ENUM_CONSTANT(UPDATE_CAPTURE);

	BIND_ENUM_CONSTANT(LOOP_NONE);
	BIND_ENUM_CONSTANT(LOOP_LINEAR);
	BIND_ENUM_CONSTANT(LOOP_PINGPONG);
}

void Animation::clear() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
	tracks.clear();
	loop_mode = LOOP_NONE;
	length = 1;
	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

bool Animation::_transform_track_optimize_key(const TKey<TransformKey> &t0, const TKey<TransformKey> &t1, const TKey<TransformKey> &t2, real_t p_alowed_linear_err, real_t p_alowed_angular_err, real_t p_max_optimizable_angle, const Vector3 &p_norm) {
	real_t c = (t1.time - t0.time) / (t2.time - t0.time);
	real_t t[3] = { -1, -1, -1 };

	{ //translation

		const Vector3 &v0 = t0.value.loc;
		const Vector3 &v1 = t1.value.loc;
		const Vector3 &v2 = t2.value.loc;

		if (v0.is_equal_approx(v2)) {
			//0 and 2 are close, let's see if 1 is close
			if (!v0.is_equal_approx(v1)) {
				//not close, not optimizable
				return false;
			}

		} else {
			Vector3 pd = (v2 - v0);
			real_t d0 = pd.dot(v0);
			real_t d1 = pd.dot(v1);
			real_t d2 = pd.dot(v2);
			if (d1 < d0 || d1 > d2) {
				return false;
			}

			Vector3 s[2] = { v0, v2 };
			real_t d = Geometry3D::get_closest_point_to_segment(v1, s).distance_to(v1);

			if (d > pd.length() * p_alowed_linear_err) {
				return false; //beyond allowed error for collinearity
			}

			if (p_norm != Vector3() && Math::acos(pd.normalized().dot(p_norm)) > p_alowed_angular_err) {
				return false;
			}

			t[0] = (d1 - d0) / (d2 - d0);
		}
	}

	{ //rotation

		const Quaternion &q0 = t0.value.rot;
		const Quaternion &q1 = t1.value.rot;
		const Quaternion &q2 = t2.value.rot;

		//localize both to rotation from q0

		if (q0.is_equal_approx(q2)) {
			if (!q0.is_equal_approx(q1)) {
				return false;
			}

		} else {
			Quaternion r02 = (q0.inverse() * q2).normalized();
			Quaternion r01 = (q0.inverse() * q1).normalized();

			Vector3 v02, v01;
			real_t a02, a01;

			r02.get_axis_angle(v02, a02);
			r01.get_axis_angle(v01, a01);

			if (Math::abs(a02) > p_max_optimizable_angle) {
				return false;
			}

			if (v01.dot(v02) < 0) {
				//make sure both rotations go the same way to compare
				v02 = -v02;
				a02 = -a02;
			}

			real_t err_01 = Math::acos(v01.normalized().dot(v02.normalized())) / Math_PI;
			if (err_01 > p_alowed_angular_err) {
				//not rotating in the same axis
				return false;
			}

			if (a01 * a02 < 0) {
				//not rotating in the same direction
				return false;
			}

			real_t tr = a01 / a02;
			if (tr < 0 || tr > 1) {
				return false; //rotating too much or too less
			}

			t[1] = tr;
		}
	}

	{ //scale

		const Vector3 &v0 = t0.value.scale;
		const Vector3 &v1 = t1.value.scale;
		const Vector3 &v2 = t2.value.scale;

		if (v0.is_equal_approx(v2)) {
			//0 and 2 are close, let's see if 1 is close
			if (!v0.is_equal_approx(v1)) {
				//not close, not optimizable
				return false;
			}

		} else {
			Vector3 pd = (v2 - v0);
			real_t d0 = pd.dot(v0);
			real_t d1 = pd.dot(v1);
			real_t d2 = pd.dot(v2);
			if (d1 < d0 || d1 > d2) {
				return false; //beyond segment range
			}

			Vector3 s[2] = { v0, v2 };
			real_t d = Geometry3D::get_closest_point_to_segment(v1, s).distance_to(v1);

			if (d > pd.length() * p_alowed_linear_err) {
				return false; //beyond allowed error for collinearity
			}

			t[2] = (d1 - d0) / (d2 - d0);
		}
	}

	bool erase = false;
	if (t[0] == -1 && t[1] == -1 && t[2] == -1) {
		erase = true;
	} else {
		erase = true;
		real_t lt = -1.0;
		for (int j = 0; j < 3; j++) {
			//search for t on first, one must be it
			if (t[j] != -1) {
				lt = t[j]; //official t
				//validate rest
				for (int k = j + 1; k < 3; k++) {
					if (t[k] == -1) {
						continue;
					}

					if (Math::abs(lt - t[k]) > p_alowed_linear_err) {
						erase = false;
						break;
					}
				}
				break;
			}
		}

		ERR_FAIL_COND_V(lt == -1, false);

		if (erase) {
			if (Math::abs(lt - c) > p_alowed_linear_err) {
				//todo, evaluate changing the transition if this fails?
				//this could be done as a second pass and would be
				//able to optimize more
				erase = false;
			}
		}
	}

	return erase;
}

void Animation::_transform_track_optimize(int p_idx, real_t p_allowed_linear_err, real_t p_allowed_angular_err, real_t p_max_optimizable_angle) {
	ERR_FAIL_INDEX(p_idx, tracks.size());
	ERR_FAIL_COND(tracks[p_idx]->type != TYPE_TRANSFORM3D);
	TransformTrack *tt = static_cast<TransformTrack *>(tracks[p_idx]);
	bool prev_erased = false;
	TKey<TransformKey> first_erased;

	Vector3 norm;

	for (int i = 1; i < tt->transforms.size() - 1; i++) {
		TKey<TransformKey> &t0 = tt->transforms.write[i - 1];
		TKey<TransformKey> &t1 = tt->transforms.write[i];
		TKey<TransformKey> &t2 = tt->transforms.write[i + 1];

		bool erase = _transform_track_optimize_key(t0, t1, t2, p_allowed_linear_err, p_allowed_angular_err, p_max_optimizable_angle, norm);
		if (erase && !prev_erased) {
			norm = (t2.value.loc - t1.value.loc).normalized();
		}

		if (prev_erased && !_transform_track_optimize_key(t0, first_erased, t2, p_allowed_linear_err, p_allowed_angular_err, p_max_optimizable_angle, norm)) {
			//avoid error to go beyond first erased key
			erase = false;
		}

		if (erase) {
			if (!prev_erased) {
				first_erased = t1;
				prev_erased = true;
			}

			tt->transforms.remove(i);
			i--;

		} else {
			prev_erased = false;
			norm = Vector3();
		}
	}
}

void Animation::optimize(real_t p_allowed_linear_err, real_t p_allowed_angular_err, real_t p_max_optimizable_angle) {
	for (int i = 0; i < tracks.size(); i++) {
		if (tracks[i]->type == TYPE_TRANSFORM3D) {
			_transform_track_optimize(i, p_allowed_linear_err, p_allowed_angular_err, p_max_optimizable_angle);
		}
	}
}

Animation::Animation() {}

Animation::~Animation() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
}
