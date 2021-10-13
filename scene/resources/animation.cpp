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

			if (type == "position_3d") {
				add_track(TYPE_POSITION_3D);
			} else if (type == "rotation_3d") {
				add_track(TYPE_ROTATION_3D);
			} else if (type == "scale_3d") {
				add_track(TYPE_SCALE_3D);
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
			if (track_get_type(track) == TYPE_POSITION_3D) {
				PositionTrack *tt = static_cast<PositionTrack *>(tracks[track]);
				Vector<real_t> values = p_value;
				int vcount = values.size();
				ERR_FAIL_COND_V(vcount % POSITION_TRACK_SIZE, false);

				const real_t *r = values.ptr();

				int64_t count = vcount / POSITION_TRACK_SIZE;
				tt->positions.resize(count);

				TKey<Vector3> *tw = tt->positions.ptrw();
				for (int i = 0; i < count; i++) {
					TKey<Vector3> &tk = tw[i];
					const real_t *ofs = &r[i * POSITION_TRACK_SIZE];
					tk.time = ofs[0];
					tk.transition = ofs[1];

					tk.value.x = ofs[2];
					tk.value.y = ofs[3];
					tk.value.z = ofs[4];
				}
			} else if (track_get_type(track) == TYPE_ROTATION_3D) {
				RotationTrack *rt = static_cast<RotationTrack *>(tracks[track]);
				Vector<real_t> values = p_value;
				int vcount = values.size();
				ERR_FAIL_COND_V(vcount % ROTATION_TRACK_SIZE, false);

				const real_t *r = values.ptr();

				int64_t count = vcount / ROTATION_TRACK_SIZE;
				rt->rotations.resize(count);

				TKey<Quaternion> *rw = rt->rotations.ptrw();
				for (int i = 0; i < count; i++) {
					TKey<Quaternion> &rk = rw[i];
					const real_t *ofs = &r[i * ROTATION_TRACK_SIZE];
					rk.time = ofs[0];
					rk.transition = ofs[1];

					rk.value.x = ofs[2];
					rk.value.y = ofs[3];
					rk.value.z = ofs[4];
					rk.value.w = ofs[5];
				}
			} else if (track_get_type(track) == TYPE_SCALE_3D) {
				ScaleTrack *st = static_cast<ScaleTrack *>(tracks[track]);
				Vector<real_t> values = p_value;
				int vcount = values.size();
				ERR_FAIL_COND_V(vcount % SCALE_TRACK_SIZE, false);

				const real_t *r = values.ptr();

				int64_t count = vcount / SCALE_TRACK_SIZE;
				st->scales.resize(count);

				TKey<Vector3> *sw = st->scales.ptrw();
				for (int i = 0; i < count; i++) {
					TKey<Vector3> &sk = sw[i];
					const real_t *ofs = &r[i * SCALE_TRACK_SIZE];
					sk.time = ofs[0];
					sk.transition = ofs[1];

					sk.value.x = ofs[2];
					sk.value.y = ofs[3];
					sk.value.z = ofs[4];
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
	} else if (name == "loop") {
		r_ret = loop;
	} else if (name == "step") {
		r_ret = step;
	} else if (name.begins_with("tracks/")) {
		int track = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(track, tracks.size(), false);
		if (what == "type") {
			switch (track_get_type(track)) {
				case TYPE_POSITION_3D:
					r_ret = "position_3d";
					break;
				case TYPE_ROTATION_3D:
					r_ret = "rotation_3d";
					break;
				case TYPE_SCALE_3D:
					r_ret = "scale_3d";
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
			if (track_get_type(track) == TYPE_POSITION_3D) {
				Vector<real_t> keys;
				int kk = track_get_key_count(track);
				keys.resize(kk * POSITION_TRACK_SIZE);

				real_t *w = keys.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					Vector3 loc;
					position_track_get_key(track, i, &loc);

					w[idx++] = track_get_key_time(track, i);
					w[idx++] = track_get_key_transition(track, i);
					w[idx++] = loc.x;
					w[idx++] = loc.y;
					w[idx++] = loc.z;
				}

				r_ret = keys;
				return true;
			} else if (track_get_type(track) == TYPE_ROTATION_3D) {
				Vector<real_t> keys;
				int kk = track_get_key_count(track);
				keys.resize(kk * ROTATION_TRACK_SIZE);

				real_t *w = keys.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					Quaternion rot;
					rotation_track_get_key(track, i, &rot);

					w[idx++] = track_get_key_time(track, i);
					w[idx++] = track_get_key_transition(track, i);
					w[idx++] = rot.x;
					w[idx++] = rot.y;
					w[idx++] = rot.z;
					w[idx++] = rot.w;
				}

				r_ret = keys;
				return true;

			} else if (track_get_type(track) == TYPE_SCALE_3D) {
				Vector<real_t> keys;
				int kk = track_get_key_count(track);
				keys.resize(kk * SCALE_TRACK_SIZE);

				real_t *w = keys.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					Vector3 scale;
					scale_track_get_key(track, i, &scale);

					w[idx++] = track_get_key_time(track, i);
					w[idx++] = track_get_key_transition(track, i);
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = memnew(PositionTrack);
			tracks.insert(p_at_pos, tt);
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = memnew(RotationTrack);
			tracks.insert(p_at_pos, rt);
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = memnew(ScaleTrack);
			tracks.insert(p_at_pos, st);
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			_clear(tt->positions);

		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			_clear(rt->rotations);

		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			_clear(st->scales);

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
	ERR_FAIL_INDEX_V(p_track, tracks.size(), TYPE_VALUE);
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

////

int Animation::position_track_insert_key(int p_track, double p_time, const Vector3 &p_position) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_POSITION_3D, -1);

	PositionTrack *tt = static_cast<PositionTrack *>(t);

	TKey<Vector3> tkey;
	tkey.time = p_time;
	tkey.value = p_position;

	int ret = _insert(p_time, tt->positions, tkey);
	emit_changed();
	return ret;
}

Error Animation::position_track_get_key(int p_track, int p_key, Vector3 *r_position) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];

	PositionTrack *tt = static_cast<PositionTrack *>(t);
	ERR_FAIL_COND_V(t->type != TYPE_POSITION_3D, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_key, tt->positions.size(), ERR_INVALID_PARAMETER);

	*r_position = tt->positions[p_key].value;

	return OK;
}

Error Animation::position_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_POSITION_3D, ERR_INVALID_PARAMETER);

	PositionTrack *tt = static_cast<PositionTrack *>(t);

	bool ok = false;

	Vector3 tk = _interpolate(tt->positions, p_time, tt->interpolation, tt->loop_wrap, &ok);

	if (!ok) {
		return ERR_UNAVAILABLE;
	}
	*r_interpolation = tk;
	return OK;
}

////

int Animation::rotation_track_insert_key(int p_track, double p_time, const Quaternion &p_rotation) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_ROTATION_3D, -1);

	RotationTrack *rt = static_cast<RotationTrack *>(t);

	TKey<Quaternion> tkey;
	tkey.time = p_time;
	tkey.value = p_rotation;

	int ret = _insert(p_time, rt->rotations, tkey);
	emit_changed();
	return ret;
}

Error Animation::rotation_track_get_key(int p_track, int p_key, Quaternion *r_rotation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];

	RotationTrack *rt = static_cast<RotationTrack *>(t);
	ERR_FAIL_COND_V(t->type != TYPE_ROTATION_3D, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_key, rt->rotations.size(), ERR_INVALID_PARAMETER);

	*r_rotation = rt->rotations[p_key].value;

	return OK;
}

Error Animation::rotation_track_interpolate(int p_track, double p_time, Quaternion *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_ROTATION_3D, ERR_INVALID_PARAMETER);

	RotationTrack *rt = static_cast<RotationTrack *>(t);

	bool ok = false;

	Quaternion tk = _interpolate(rt->rotations, p_time, rt->interpolation, rt->loop_wrap, &ok);

	if (!ok) {
		return ERR_UNAVAILABLE;
	}
	*r_interpolation = tk;
	return OK;
}

////

int Animation::scale_track_insert_key(int p_track, double p_time, const Vector3 &p_scale) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_SCALE_3D, -1);

	ScaleTrack *st = static_cast<ScaleTrack *>(t);

	TKey<Vector3> tkey;
	tkey.time = p_time;
	tkey.value = p_scale;

	int ret = _insert(p_time, st->scales, tkey);
	emit_changed();
	return ret;
}

Error Animation::scale_track_get_key(int p_track, int p_key, Vector3 *r_scale) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];

	ScaleTrack *st = static_cast<ScaleTrack *>(t);
	ERR_FAIL_COND_V(t->type != TYPE_SCALE_3D, ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_key, st->scales.size(), ERR_INVALID_PARAMETER);

	*r_scale = st->scales[p_key].value;

	return OK;
}

Error Animation::scale_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_SCALE_3D, ERR_INVALID_PARAMETER);

	ScaleTrack *st = static_cast<ScaleTrack *>(t);

	bool ok = false;

	Vector3 tk = _interpolate(st->scales, p_time, st->interpolation, st->loop_wrap, &ok);

	if (!ok) {
		return ERR_UNAVAILABLE;
	}
	*r_interpolation = tk;
	return OK;
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX(p_idx, tt->positions.size());
			tt->positions.remove(p_idx);

		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX(p_idx, rt->rotations.size());
			rt->rotations.remove(p_idx);

		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX(p_idx, st->scales.size());
			st->scales.remove(p_idx);

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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			int k = _find(tt->positions, p_time);
			if (k < 0 || k >= tt->positions.size()) {
				return -1;
			}
			if (tt->positions[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			int k = _find(rt->rotations, p_time);
			if (k < 0 || k >= rt->rotations.size()) {
				return -1;
			}
			if (rt->rotations[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *rt = static_cast<ScaleTrack *>(t);
			int k = _find(rt->scales, p_time);
			if (k < 0 || k >= rt->scales.size()) {
				return -1;
			}
			if (rt->scales[k].time != p_time && p_exact) {
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
		case TYPE_POSITION_3D: {
			ERR_FAIL_COND((p_key.get_type() != Variant::VECTOR3) && (p_key.get_type() != Variant::VECTOR3I));
			int idx = position_track_insert_key(p_track, p_time, p_key);
			track_set_key_transition(p_track, idx, p_transition);

		} break;
		case TYPE_ROTATION_3D: {
			ERR_FAIL_COND((p_key.get_type() != Variant::QUATERNION) && (p_key.get_type() != Variant::BASIS));
			int idx = rotation_track_insert_key(p_track, p_time, p_key);
			track_set_key_transition(p_track, idx, p_transition);

		} break;
		case TYPE_SCALE_3D: {
			ERR_FAIL_COND((p_key.get_type() != Variant::VECTOR3) && (p_key.get_type() != Variant::VECTOR3I));
			int idx = scale_track_insert_key(p_track, p_time, p_key);
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			return tt->positions.size();
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			return rt->rotations.size();
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			return st->scales.size();
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->positions.size(), Variant());

			return tt->positions[p_key_idx].value;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, rt->rotations.size(), Variant());

			return rt->rotations[p_key_idx].value;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, st->scales.size(), Variant());

			return st->scales[p_key_idx].value;
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->positions.size(), -1);
			return tt->positions[p_key_idx].time;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, rt->rotations.size(), -1);
			return rt->rotations[p_key_idx].time;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, st->scales.size(), -1);
			return st->scales[p_key_idx].time;
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());
			TKey<Vector3> key = tt->positions[p_key_idx];
			key.time = p_time;
			tt->positions.remove(p_key_idx);
			_insert(p_time, tt->positions, key);
			return;
		}
		case TYPE_ROTATION_3D: {
			RotationTrack *tt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->rotations.size());
			TKey<Quaternion> key = tt->rotations[p_key_idx];
			key.time = p_time;
			tt->rotations.remove(p_key_idx);
			_insert(p_time, tt->rotations, key);
			return;
		}
		case TYPE_SCALE_3D: {
			ScaleTrack *tt = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->scales.size());
			TKey<Vector3> key = tt->scales[p_key_idx];
			key.time = p_time;
			tt->scales.remove(p_key_idx);
			_insert(p_time, tt->scales, key);
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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, tt->positions.size(), -1);
			return tt->positions[p_key_idx].transition;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, rt->rotations.size(), -1);
			return rt->rotations[p_key_idx].transition;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX_V(p_key_idx, st->scales.size(), -1);
			return st->scales[p_key_idx].transition;
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
		case TYPE_POSITION_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::VECTOR3) && (p_value.get_type() != Variant::VECTOR3I));
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());

			tt->positions.write[p_key_idx].value = p_value;

		} break;
		case TYPE_ROTATION_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::QUATERNION) && (p_value.get_type() != Variant::BASIS));
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, rt->rotations.size());

			rt->rotations.write[p_key_idx].value = p_value;

		} break;
		case TYPE_SCALE_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::VECTOR3) && (p_value.get_type() != Variant::VECTOR3I));
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, st->scales.size());

			st->scales.write[p_key_idx].value = p_value;

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
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());
			tt->positions.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, rt->rotations.size());
			rt->rotations.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, st->scales.size());
			st->scales.write[p_key_idx].transition = p_transition;
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
int Animation::_find(const Vector<K> &p_keys, double p_time) const {
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

	if (keys[middle].time > p_time) {
		middle--;
	}

	return middle;
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
T Animation::_interpolate(const Vector<TKey<T>> &p_keys, double p_time, InterpolationType p_interp, bool p_loop_wrap, bool *p_ok) const {
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

	int idx = _find(p_keys, p_time);

	ERR_FAIL_COND_V(idx == -2, T());

	bool result = true;
	int next = 0;
	real_t c = 0.0;
	// prepare for all cases of interpolation

	if (loop && p_loop_wrap) {
		// loop
		if (idx >= 0) {
			if ((idx + 1) < len) {
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

	} else { // no loop

		if (idx >= 0) {
			if ((idx + 1) < len) {
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
			// only allow extending first key to anim start if looping
			if (loop) {
				idx = next = 0;
			} else {
				result = false;
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

void Animation::value_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_VALUE);

	ValueTrack *vt = static_cast<ValueTrack *>(t);

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	if (loop) {
		from_time = Math::fposmod(from_time, length);
		to_time = Math::fposmod(to_time, length);

		if (from_time > to_time) {
			// handle loop by splitting
			_value_track_get_key_indices_in_range(vt, from_time, length, p_indices);
			_value_track_get_key_indices_in_range(vt, 0, to_time, p_indices);
			return;
		}
	} else {
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

void Animation::track_get_key_indices_in_range(int p_track, double p_time, double p_delta, List<int> *p_indices) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	const Track *t = tracks[p_track];

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	if (loop) {
		if (from_time > length || from_time < 0) {
			from_time = Math::fposmod(from_time, length);
		}

		if (to_time > length || to_time < 0) {
			to_time = Math::fposmod(to_time, length);
		}

		if (from_time > to_time) {
			// handle loop by splitting

			switch (t->type) {
				case TYPE_POSITION_3D: {
					const PositionTrack *tt = static_cast<const PositionTrack *>(t);
					_track_get_key_indices_in_range(tt->positions, from_time, length, p_indices);
					_track_get_key_indices_in_range(tt->positions, 0, to_time, p_indices);

				} break;
				case TYPE_ROTATION_3D: {
					const RotationTrack *rt = static_cast<const RotationTrack *>(t);
					_track_get_key_indices_in_range(rt->rotations, from_time, length, p_indices);
					_track_get_key_indices_in_range(rt->rotations, 0, to_time, p_indices);

				} break;
				case TYPE_SCALE_3D: {
					const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
					_track_get_key_indices_in_range(st->scales, from_time, length, p_indices);
					_track_get_key_indices_in_range(st->scales, 0, to_time, p_indices);

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
	} else {
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
	}

	switch (t->type) {
		case TYPE_POSITION_3D: {
			const PositionTrack *tt = static_cast<const PositionTrack *>(t);
			_track_get_key_indices_in_range(tt->positions, from_time, to_time, p_indices);

		} break;
		case TYPE_ROTATION_3D: {
			const RotationTrack *rt = static_cast<const RotationTrack *>(t);
			_track_get_key_indices_in_range(rt->rotations, from_time, to_time, p_indices);

		} break;
		case TYPE_SCALE_3D: {
			const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
			_track_get_key_indices_in_range(st->scales, from_time, to_time, p_indices);

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

void Animation::method_track_get_key_indices(int p_track, double p_time, double p_delta, List<int> *p_indices) const {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_METHOD);

	MethodTrack *mt = static_cast<MethodTrack *>(t);

	double from_time = p_time - p_delta;
	double to_time = p_time;

	if (from_time > to_time) {
		SWAP(from_time, to_time);
	}

	if (loop) {
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
	} else {
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

void Animation::set_loop(bool p_enabled) {
	loop = p_enabled;
	emit_changed();
}

bool Animation::has_loop() const {
	return loop;
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

	ClassDB::bind_method(D_METHOD("position_track_insert_key", "track_idx", "time", "position"), &Animation::position_track_insert_key);
	ClassDB::bind_method(D_METHOD("rotation_track_insert_key", "track_idx", "time", "rotation"), &Animation::rotation_track_insert_key);
	ClassDB::bind_method(D_METHOD("scale_track_insert_key", "track_idx", "time", "scale"), &Animation::scale_track_insert_key);

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

	ClassDB::bind_method(D_METHOD("set_loop", "enabled"), &Animation::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &Animation::has_loop);

	ClassDB::bind_method(D_METHOD("set_step", "size_sec"), &Animation::set_step);
	ClassDB::bind_method(D_METHOD("get_step"), &Animation::get_step);

	ClassDB::bind_method(D_METHOD("clear"), &Animation::clear);
	ClassDB::bind_method(D_METHOD("copy_track", "track_idx", "to_animation"), &Animation::copy_track);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0.001,99999,0.001"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step", PROPERTY_HINT_RANGE, "0,4096,0.001"), "set_step", "get_step");

	ADD_SIGNAL(MethodInfo("tracks_changed"));

	BIND_ENUM_CONSTANT(TYPE_VALUE);
	BIND_ENUM_CONSTANT(TYPE_POSITION_3D);
	BIND_ENUM_CONSTANT(TYPE_ROTATION_3D);
	BIND_ENUM_CONSTANT(TYPE_SCALE_3D);
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
}

void Animation::clear() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
	tracks.clear();
	loop = false;
	length = 1;
	emit_changed();
	emit_signal(SceneStringNames::get_singleton()->tracks_changed);
}

bool Animation::_position_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_allowed_linear_err, real_t p_allowed_angular_error, const Vector3 &p_norm) {
	const Vector3 &v0 = t0.value;
	const Vector3 &v1 = t1.value;
	const Vector3 &v2 = t2.value;

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

		if (d > pd.length() * p_allowed_linear_err) {
			return false; //beyond allowed error for collinearity
		}

		if (p_norm != Vector3() && Math::acos(pd.normalized().dot(p_norm)) > p_allowed_angular_error) {
			return false;
		}
	}

	return true;
}

bool Animation::_rotation_track_optimize_key(const TKey<Quaternion> &t0, const TKey<Quaternion> &t1, const TKey<Quaternion> &t2, real_t p_allowed_angular_error, float p_max_optimizable_angle) {
	const Quaternion &q0 = t0.value;
	const Quaternion &q1 = t1.value;
	const Quaternion &q2 = t2.value;

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
		if (err_01 > p_allowed_angular_error) {
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
	}

	return true;
}

bool Animation::_scale_track_optimize_key(const TKey<Vector3> &t0, const TKey<Vector3> &t1, const TKey<Vector3> &t2, real_t p_allowed_linear_error) {
	const Vector3 &v0 = t0.value;
	const Vector3 &v1 = t1.value;
	const Vector3 &v2 = t2.value;

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

		if (d > pd.length() * p_allowed_linear_error) {
			return false; //beyond allowed error for colinearity
		}
	}

	return true;
}

void Animation::_position_track_optimize(int p_idx, real_t p_allowed_linear_err, real_t p_allowed_angular_err) {
	ERR_FAIL_INDEX(p_idx, tracks.size());
	ERR_FAIL_COND(tracks[p_idx]->type != TYPE_POSITION_3D);
	PositionTrack *tt = static_cast<PositionTrack *>(tracks[p_idx]);
	bool prev_erased = false;
	TKey<Vector3> first_erased;

	Vector3 norm;

	for (int i = 1; i < tt->positions.size() - 1; i++) {
		TKey<Vector3> &t0 = tt->positions.write[i - 1];
		TKey<Vector3> &t1 = tt->positions.write[i];
		TKey<Vector3> &t2 = tt->positions.write[i + 1];

		bool erase = _position_track_optimize_key(t0, t1, t2, p_allowed_linear_err, p_allowed_angular_err, norm);
		if (erase && !prev_erased) {
			norm = (t2.value - t1.value).normalized();
		}

		if (prev_erased && !_position_track_optimize_key(t0, first_erased, t2, p_allowed_linear_err, p_allowed_angular_err, norm)) {
			//avoid error to go beyond first erased key
			erase = false;
		}

		if (erase) {
			if (!prev_erased) {
				first_erased = t1;
				prev_erased = true;
			}

			tt->positions.remove(i);
			i--;

		} else {
			prev_erased = false;
			norm = Vector3();
		}
	}
}

void Animation::_rotation_track_optimize(int p_idx, real_t p_allowed_angular_err, real_t p_max_optimizable_angle) {
	ERR_FAIL_INDEX(p_idx, tracks.size());
	ERR_FAIL_COND(tracks[p_idx]->type != TYPE_ROTATION_3D);
	RotationTrack *tt = static_cast<RotationTrack *>(tracks[p_idx]);
	bool prev_erased = false;
	TKey<Quaternion> first_erased;

	for (int i = 1; i < tt->rotations.size() - 1; i++) {
		TKey<Quaternion> &t0 = tt->rotations.write[i - 1];
		TKey<Quaternion> &t1 = tt->rotations.write[i];
		TKey<Quaternion> &t2 = tt->rotations.write[i + 1];

		bool erase = _rotation_track_optimize_key(t0, t1, t2, p_allowed_angular_err, p_max_optimizable_angle);

		if (prev_erased && !_rotation_track_optimize_key(t0, first_erased, t2, p_allowed_angular_err, p_max_optimizable_angle)) {
			//avoid error to go beyond first erased key
			erase = false;
		}

		if (erase) {
			if (!prev_erased) {
				first_erased = t1;
				prev_erased = true;
			}

			tt->rotations.remove(i);
			i--;

		} else {
			prev_erased = false;
		}
	}
}

void Animation::_scale_track_optimize(int p_idx, real_t p_allowed_linear_err) {
	ERR_FAIL_INDEX(p_idx, tracks.size());
	ERR_FAIL_COND(tracks[p_idx]->type != TYPE_SCALE_3D);
	ScaleTrack *tt = static_cast<ScaleTrack *>(tracks[p_idx]);
	bool prev_erased = false;
	TKey<Vector3> first_erased;

	for (int i = 1; i < tt->scales.size() - 1; i++) {
		TKey<Vector3> &t0 = tt->scales.write[i - 1];
		TKey<Vector3> &t1 = tt->scales.write[i];
		TKey<Vector3> &t2 = tt->scales.write[i + 1];

		bool erase = _scale_track_optimize_key(t0, t1, t2, p_allowed_linear_err);

		if (prev_erased && !_scale_track_optimize_key(t0, first_erased, t2, p_allowed_linear_err)) {
			//avoid error to go beyond first erased key
			erase = false;
		}

		if (erase) {
			if (!prev_erased) {
				first_erased = t1;
				prev_erased = true;
			}

			tt->scales.remove(i);
			i--;

		} else {
			prev_erased = false;
		}
	}
}

void Animation::optimize(real_t p_allowed_linear_err, real_t p_allowed_angular_err, real_t p_max_optimizable_angle) {
	for (int i = 0; i < tracks.size(); i++) {
		if (tracks[i]->type == TYPE_POSITION_3D) {
			_position_track_optimize(i, p_allowed_linear_err, p_allowed_angular_err);
		} else if (tracks[i]->type == TYPE_ROTATION_3D) {
			_rotation_track_optimize(i, p_allowed_angular_err, p_max_optimizable_angle);
		} else if (tracks[i]->type == TYPE_SCALE_3D) {
			_scale_track_optimize(i, p_allowed_linear_err);
		}
	}
}

Animation::Animation() {}

Animation::~Animation() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
}
