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

#include "core/io/marshalls.h"
#include "core/math/geometry_3d.h"
#include "scene/scene_string_names.h"

bool Animation::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (p_name == SNAME("_compression")) {
		ERR_FAIL_COND_V(tracks.size() > 0, false); //can only set compression if no tracks exist
		Dictionary comp = p_value;
		ERR_FAIL_COND_V(!comp.has("fps"), false);
		ERR_FAIL_COND_V(!comp.has("bounds"), false);
		ERR_FAIL_COND_V(!comp.has("pages"), false);
		ERR_FAIL_COND_V(!comp.has("format_version"), false);
		uint32_t format_version = comp["format_version"];
		ERR_FAIL_COND_V(format_version > Compression::FORMAT_VERSION, false); // version does not match this supported version
		compression.fps = comp["fps"];
		Array bounds = comp["bounds"];
		compression.bounds.resize(bounds.size());
		for (int i = 0; i < bounds.size(); i++) {
			compression.bounds[i] = bounds[i];
		}
		Array pages = comp["pages"];
		compression.pages.resize(pages.size());
		for (int i = 0; i < pages.size(); i++) {
			Dictionary page = pages[i];
			ERR_FAIL_COND_V(!page.has("data"), false);
			ERR_FAIL_COND_V(!page.has("time_offset"), false);
			compression.pages[i].data = page["data"];
			compression.pages[i].time_offset = page["time_offset"];
		}
		compression.enabled = true;
		return true;
	} else if (name.begins_with("tracks/")) {
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
			} else if (type == "blend_shape") {
				add_track(TYPE_BLEND_SHAPE);
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
		} else if (what == "compressed_track") {
			int index = p_value;
			ERR_FAIL_COND_V(!compression.enabled, false);
			ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)index, compression.bounds.size(), false);
			Track *t = tracks[track];
			t->interpolation = INTERPOLATION_LINEAR; //only linear supported
			switch (t->type) {
				case TYPE_POSITION_3D: {
					PositionTrack *tt = static_cast<PositionTrack *>(t);
					tt->compressed_track = index;
				} break;
				case TYPE_ROTATION_3D: {
					RotationTrack *rt = static_cast<RotationTrack *>(t);
					rt->compressed_track = index;
				} break;
				case TYPE_SCALE_3D: {
					ScaleTrack *st = static_cast<ScaleTrack *>(t);
					st->compressed_track = index;
				} break;
				case TYPE_BLEND_SHAPE: {
					BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
					bst->compressed_track = index;
				} break;
				default: {
					return false;
				}
			}
			return true;
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
			} else if (track_get_type(track) == TYPE_BLEND_SHAPE) {
				BlendShapeTrack *st = static_cast<BlendShapeTrack *>(tracks[track]);
				Vector<real_t> values = p_value;
				int vcount = values.size();
				ERR_FAIL_COND_V(vcount % BLEND_SHAPE_TRACK_SIZE, false);

				const real_t *r = values.ptr();

				int64_t count = vcount / BLEND_SHAPE_TRACK_SIZE;
				st->blend_shapes.resize(count);

				TKey<float> *sw = st->blend_shapes.ptrw();
				for (int i = 0; i < count; i++) {
					TKey<float> &sk = sw[i];
					const real_t *ofs = &r[i * BLEND_SHAPE_TRACK_SIZE];
					sk.time = ofs[0];
					sk.transition = ofs[1];
					sk.value = ofs[2];
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

				ERR_FAIL_COND_V(times.size() * 6 != values.size(), false);

				if (times.size()) {
					int valcount = times.size();

					const real_t *rt = times.ptr();
					const real_t *rv = values.ptr();

					bt->values.resize(valcount);

					for (int i = 0; i < valcount; i++) {
						bt->values.write[i].time = rt[i];
						bt->values.write[i].transition = 0; //unused in bezier
						bt->values.write[i].value.value = rv[i * 6 + 0];
						bt->values.write[i].value.in_handle.x = rv[i * 6 + 1];
						bt->values.write[i].value.in_handle.y = rv[i * 6 + 2];
						bt->values.write[i].value.out_handle.x = rv[i * 6 + 3];
						bt->values.write[i].value.out_handle.y = rv[i * 6 + 4];
						bt->values.write[i].value.handle_mode = static_cast<HandleMode>((int)rv[i * 6 + 5]);
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

	if (p_name == SNAME("_compression")) {
		ERR_FAIL_COND_V(!compression.enabled, false);
		Dictionary comp;
		comp["fps"] = compression.fps;
		Array bounds;
		bounds.resize(compression.bounds.size());
		for (uint32_t i = 0; i < compression.bounds.size(); i++) {
			bounds[i] = compression.bounds[i];
		}
		comp["bounds"] = bounds;
		Array pages;
		pages.resize(compression.pages.size());
		for (uint32_t i = 0; i < compression.pages.size(); i++) {
			Dictionary page;
			page["data"] = compression.pages[i].data;
			page["time_offset"] = compression.pages[i].time_offset;
			pages[i] = page;
		}
		comp["pages"] = pages;
		comp["format_version"] = Compression::FORMAT_VERSION;

		r_ret = comp;
		return true;
	} else if (name == "length") {
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
				case TYPE_POSITION_3D:
					r_ret = "position_3d";
					break;
				case TYPE_ROTATION_3D:
					r_ret = "rotation_3d";
					break;
				case TYPE_SCALE_3D:
					r_ret = "scale_3d";
					break;
				case TYPE_BLEND_SHAPE:
					r_ret = "blend_shape";
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
		} else if (what == "compressed_track") {
			ERR_FAIL_COND_V(!compression.enabled, false);
			Track *t = tracks[track];
			switch (t->type) {
				case TYPE_POSITION_3D: {
					PositionTrack *tt = static_cast<PositionTrack *>(t);
					r_ret = tt->compressed_track;
				} break;
				case TYPE_ROTATION_3D: {
					RotationTrack *rt = static_cast<RotationTrack *>(t);
					r_ret = rt->compressed_track;
				} break;
				case TYPE_SCALE_3D: {
					ScaleTrack *st = static_cast<ScaleTrack *>(t);
					r_ret = st->compressed_track;
				} break;
				case TYPE_BLEND_SHAPE: {
					BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
					r_ret = bst->compressed_track;
				} break;
				default: {
					r_ret = Variant();
					ERR_FAIL_V(false);
				}
			}

			return true;

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
			} else if (track_get_type(track) == TYPE_BLEND_SHAPE) {
				Vector<real_t> keys;
				int kk = track_get_key_count(track);
				keys.resize(kk * BLEND_SHAPE_TRACK_SIZE);

				real_t *w = keys.ptrw();

				int idx = 0;
				for (int i = 0; i < track_get_key_count(track); i++) {
					float bs;
					blend_shape_track_get_key(track, i, &bs);

					w[idx++] = track_get_key_time(track, i);
					w[idx++] = track_get_key_transition(track, i);
					w[idx++] = bs;
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
				key_points.resize(kk * 6);

				real_t *wti = key_times.ptrw();
				real_t *wpo = key_points.ptrw();

				int idx = 0;

				const TKey<BezierKey> *vls = bt->values.ptr();

				for (int i = 0; i < kk; i++) {
					wti[idx] = vls[i].time;
					wpo[idx * 6 + 0] = vls[i].value.value;
					wpo[idx * 6 + 1] = vls[i].value.in_handle.x;
					wpo[idx * 6 + 2] = vls[i].value.in_handle.y;
					wpo[idx * 6 + 3] = vls[i].value.out_handle.x;
					wpo[idx * 6 + 4] = vls[i].value.out_handle.y;
					wpo[idx * 6 + 5] = (double)vls[i].value.handle_mode;
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
	if (compression.enabled) {
		p_list->push_back(PropertyInfo(Variant::DICTIONARY, "_compression", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
	for (int i = 0; i < tracks.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, "tracks/" + itos(i) + "/type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/imported", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "tracks/" + itos(i) + "/path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		if (track_is_compressed(i)) {
			p_list->push_back(PropertyInfo(Variant::INT, "tracks/" + itos(i) + "/compressed_track", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		} else {
			p_list->push_back(PropertyInfo(Variant::INT, "tracks/" + itos(i) + "/interp", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::BOOL, "tracks/" + itos(i) + "/loop_wrap", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::ARRAY, "tracks/" + itos(i) + "/keys", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		}
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
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = memnew(BlendShapeTrack);
			tracks.insert(p_at_pos, bst);
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
			ERR_FAIL_COND_MSG(tt->compressed_track >= 0, "Compressed tracks can't be manually removed. Call clear() to get rid of compression first.");
			_clear(tt->positions);

		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_COND_MSG(rt->compressed_track >= 0, "Compressed tracks can't be manually removed. Call clear() to get rid of compression first.");
			_clear(rt->rotations);

		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_COND_MSG(st->compressed_track >= 0, "Compressed tracks can't be manually removed. Call clear() to get rid of compression first.");
			_clear(st->scales);

		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			ERR_FAIL_COND_MSG(bst->compressed_track >= 0, "Compressed tracks can't be manually removed. Call clear() to get rid of compression first.");
			_clear(bst->blend_shapes);

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
	tracks.remove_at(p_track);
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

int Animation::find_track(const NodePath &p_path, const TrackType p_type) const {
	for (int i = 0; i < tracks.size(); i++) {
		if (tracks[i]->path == p_path && tracks[i]->type == p_type) {
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

	ERR_FAIL_COND_V(tt->compressed_track >= 0, -1);

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

	if (tt->compressed_track >= 0) {
		Vector3i key;
		double time;
		bool fetch_success = _fetch_compressed_by_index<3>(tt->compressed_track, p_key, key, time);
		if (!fetch_success) {
			return ERR_INVALID_PARAMETER;
		}

		*r_position = _uncompress_pos_scale(tt->compressed_track, key);
		return OK;
	}

	ERR_FAIL_INDEX_V(p_key, tt->positions.size(), ERR_INVALID_PARAMETER);

	*r_position = tt->positions[p_key].value;

	return OK;
}

Error Animation::position_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_POSITION_3D, ERR_INVALID_PARAMETER);

	PositionTrack *tt = static_cast<PositionTrack *>(t);

	if (tt->compressed_track >= 0) {
		if (_pos_scale_interpolate_compressed(tt->compressed_track, p_time, *r_interpolation)) {
			return OK;
		} else {
			return ERR_UNAVAILABLE;
		}
	}

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

	ERR_FAIL_COND_V(rt->compressed_track >= 0, -1);

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

	if (rt->compressed_track >= 0) {
		Vector3i key;
		double time;
		bool fetch_success = _fetch_compressed_by_index<3>(rt->compressed_track, p_key, key, time);
		if (!fetch_success) {
			return ERR_INVALID_PARAMETER;
		}

		*r_rotation = _uncompress_quaternion(key);
		return OK;
	}

	ERR_FAIL_INDEX_V(p_key, rt->rotations.size(), ERR_INVALID_PARAMETER);

	*r_rotation = rt->rotations[p_key].value;

	return OK;
}

Error Animation::rotation_track_interpolate(int p_track, double p_time, Quaternion *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_ROTATION_3D, ERR_INVALID_PARAMETER);

	RotationTrack *rt = static_cast<RotationTrack *>(t);

	if (rt->compressed_track >= 0) {
		if (_rotation_interpolate_compressed(rt->compressed_track, p_time, *r_interpolation)) {
			return OK;
		} else {
			return ERR_UNAVAILABLE;
		}
	}

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

	ERR_FAIL_COND_V(st->compressed_track >= 0, -1);

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

	if (st->compressed_track >= 0) {
		Vector3i key;
		double time;
		bool fetch_success = _fetch_compressed_by_index<3>(st->compressed_track, p_key, key, time);
		if (!fetch_success) {
			return ERR_INVALID_PARAMETER;
		}

		*r_scale = _uncompress_pos_scale(st->compressed_track, key);
		return OK;
	}

	ERR_FAIL_INDEX_V(p_key, st->scales.size(), ERR_INVALID_PARAMETER);

	*r_scale = st->scales[p_key].value;

	return OK;
}

Error Animation::scale_track_interpolate(int p_track, double p_time, Vector3 *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_SCALE_3D, ERR_INVALID_PARAMETER);

	ScaleTrack *st = static_cast<ScaleTrack *>(t);

	if (st->compressed_track >= 0) {
		if (_pos_scale_interpolate_compressed(st->compressed_track, p_time, *r_interpolation)) {
			return OK;
		} else {
			return ERR_UNAVAILABLE;
		}
	}

	bool ok = false;

	Vector3 tk = _interpolate(st->scales, p_time, st->interpolation, st->loop_wrap, &ok);

	if (!ok) {
		return ERR_UNAVAILABLE;
	}
	*r_interpolation = tk;
	return OK;
}

int Animation::blend_shape_track_insert_key(int p_track, double p_time, float p_blend_shape) {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), -1);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BLEND_SHAPE, -1);

	BlendShapeTrack *st = static_cast<BlendShapeTrack *>(t);

	ERR_FAIL_COND_V(st->compressed_track >= 0, -1);

	TKey<float> tkey;
	tkey.time = p_time;
	tkey.value = p_blend_shape;

	int ret = _insert(p_time, st->blend_shapes, tkey);
	emit_changed();
	return ret;
}

Error Animation::blend_shape_track_get_key(int p_track, int p_key, float *r_blend_shape) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];

	BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
	ERR_FAIL_COND_V(t->type != TYPE_BLEND_SHAPE, ERR_INVALID_PARAMETER);

	if (bst->compressed_track >= 0) {
		Vector3i key;
		double time;
		bool fetch_success = _fetch_compressed_by_index<1>(bst->compressed_track, p_key, key, time);
		if (!fetch_success) {
			return ERR_INVALID_PARAMETER;
		}

		*r_blend_shape = _uncompress_blend_shape(key);
		return OK;
	}

	ERR_FAIL_INDEX_V(p_key, bst->blend_shapes.size(), ERR_INVALID_PARAMETER);

	*r_blend_shape = bst->blend_shapes[p_key].value;

	return OK;
}

Error Animation::blend_shape_track_interpolate(int p_track, double p_time, float *r_interpolation) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), ERR_INVALID_PARAMETER);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BLEND_SHAPE, ERR_INVALID_PARAMETER);

	BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);

	if (bst->compressed_track >= 0) {
		if (_blend_shape_interpolate_compressed(bst->compressed_track, p_time, *r_interpolation)) {
			return OK;
		} else {
			return ERR_UNAVAILABLE;
		}
	}

	bool ok = false;

	float tk = _interpolate(bst->blend_shapes, p_time, bst->interpolation, bst->loop_wrap, &ok);

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

			ERR_FAIL_COND(tt->compressed_track >= 0);

			ERR_FAIL_INDEX(p_idx, tt->positions.size());
			tt->positions.remove_at(p_idx);

		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);

			ERR_FAIL_COND(rt->compressed_track >= 0);

			ERR_FAIL_INDEX(p_idx, rt->rotations.size());
			rt->rotations.remove_at(p_idx);

		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);

			ERR_FAIL_COND(st->compressed_track >= 0);

			ERR_FAIL_INDEX(p_idx, st->scales.size());
			st->scales.remove_at(p_idx);

		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);

			ERR_FAIL_COND(bst->compressed_track >= 0);

			ERR_FAIL_INDEX(p_idx, bst->blend_shapes.size());
			bst->blend_shapes.remove_at(p_idx);

		} break;
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_idx, vt->values.size());
			vt->values.remove_at(p_idx);

		} break;
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_idx, mt->methods.size());
			mt->methods.remove_at(p_idx);

		} break;
		case TYPE_BEZIER: {
			BezierTrack *bz = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX(p_idx, bz->values.size());
			bz->values.remove_at(p_idx);

		} break;
		case TYPE_AUDIO: {
			AudioTrack *ad = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX(p_idx, ad->values.size());
			ad->values.remove_at(p_idx);

		} break;
		case TYPE_ANIMATION: {
			AnimationTrack *an = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX(p_idx, an->values.size());
			an->values.remove_at(p_idx);

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

			if (tt->compressed_track >= 0) {
				double time;
				double time_next;
				Vector3i key;
				Vector3i key_next;
				uint32_t key_index;
				bool fetch_compressed_success = _fetch_compressed<3>(tt->compressed_track, p_time, key, time, key_next, time_next, &key_index);
				ERR_FAIL_COND_V(!fetch_compressed_success, -1);
				if (p_exact && time != p_time) {
					return -1;
				}
				return key_index;
			}

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

			if (rt->compressed_track >= 0) {
				double time;
				double time_next;
				Vector3i key;
				Vector3i key_next;
				uint32_t key_index;
				bool fetch_compressed_success = _fetch_compressed<3>(rt->compressed_track, p_time, key, time, key_next, time_next, &key_index);
				ERR_FAIL_COND_V(!fetch_compressed_success, -1);
				if (p_exact && time != p_time) {
					return -1;
				}
				return key_index;
			}

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
			ScaleTrack *st = static_cast<ScaleTrack *>(t);

			if (st->compressed_track >= 0) {
				double time;
				double time_next;
				Vector3i key;
				Vector3i key_next;
				uint32_t key_index;
				bool fetch_compressed_success = _fetch_compressed<3>(st->compressed_track, p_time, key, time, key_next, time_next, &key_index);
				ERR_FAIL_COND_V(!fetch_compressed_success, -1);
				if (p_exact && time != p_time) {
					return -1;
				}
				return key_index;
			}

			int k = _find(st->scales, p_time);
			if (k < 0 || k >= st->scales.size()) {
				return -1;
			}
			if (st->scales[k].time != p_time && p_exact) {
				return -1;
			}
			return k;

		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);

			if (bst->compressed_track >= 0) {
				double time;
				double time_next;
				Vector3i key;
				Vector3i key_next;
				uint32_t key_index;
				bool fetch_compressed_success = _fetch_compressed<1>(bst->compressed_track, p_time, key, time, key_next, time_next, &key_index);
				ERR_FAIL_COND_V(!fetch_compressed_success, -1);
				if (p_exact && time != p_time) {
					return -1;
				}
				return key_index;
			}

			int k = _find(bst->blend_shapes, p_time);
			if (k < 0 || k >= bst->blend_shapes.size()) {
				return -1;
			}
			if (bst->blend_shapes[k].time != p_time && p_exact) {
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
		case TYPE_BLEND_SHAPE: {
			ERR_FAIL_COND((p_key.get_type() != Variant::FLOAT) && (p_key.get_type() != Variant::INT));
			int idx = blend_shape_track_insert_key(p_track, p_time, p_key);
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
			ERR_FAIL_COND(arr.size() != 6);

			TKey<BezierKey> k;
			k.time = p_time;
			k.value.value = arr[0];
			k.value.in_handle.x = arr[1];
			k.value.in_handle.y = arr[2];
			k.value.out_handle.x = arr[3];
			k.value.out_handle.y = arr[4];
			k.value.handle_mode = static_cast<HandleMode>((int)arr[5]);
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
			if (tt->compressed_track >= 0) {
				return _get_compressed_key_count(tt->compressed_track);
			}
			return tt->positions.size();
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			if (rt->compressed_track >= 0) {
				return _get_compressed_key_count(rt->compressed_track);
			}
			return rt->rotations.size();
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			if (st->compressed_track >= 0) {
				return _get_compressed_key_count(st->compressed_track);
			}
			return st->scales.size();
		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			if (bst->compressed_track >= 0) {
				return _get_compressed_key_count(bst->compressed_track);
			}
			return bst->blend_shapes.size();
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
			Vector3 value;
			position_track_get_key(p_track, p_key_idx, &value);
			return value;
		} break;
		case TYPE_ROTATION_3D: {
			Quaternion value;
			rotation_track_get_key(p_track, p_key_idx, &value);
			return value;
		} break;
		case TYPE_SCALE_3D: {
			Vector3 value;
			scale_track_get_key(p_track, p_key_idx, &value);
			return value;
		} break;
		case TYPE_BLEND_SHAPE: {
			float value;
			blend_shape_track_get_key(p_track, p_key_idx, &value);
			return value;
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
			arr.resize(6);
			arr[0] = bt->values[p_key_idx].value.value;
			arr[1] = bt->values[p_key_idx].value.in_handle.x;
			arr[2] = bt->values[p_key_idx].value.in_handle.y;
			arr[3] = bt->values[p_key_idx].value.out_handle.x;
			arr[4] = bt->values[p_key_idx].value.out_handle.y;
			arr[5] = (double)bt->values[p_key_idx].value.handle_mode;
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
			if (tt->compressed_track >= 0) {
				Vector3i value;
				double time;
				bool fetch_compressed_success = _fetch_compressed_by_index<3>(tt->compressed_track, p_key_idx, value, time);
				ERR_FAIL_COND_V(!fetch_compressed_success, false);
				return time;
			}
			ERR_FAIL_INDEX_V(p_key_idx, tt->positions.size(), -1);
			return tt->positions[p_key_idx].time;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			if (rt->compressed_track >= 0) {
				Vector3i value;
				double time;
				bool fetch_compressed_success = _fetch_compressed_by_index<3>(rt->compressed_track, p_key_idx, value, time);
				ERR_FAIL_COND_V(!fetch_compressed_success, false);
				return time;
			}
			ERR_FAIL_INDEX_V(p_key_idx, rt->rotations.size(), -1);
			return rt->rotations[p_key_idx].time;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			if (st->compressed_track >= 0) {
				Vector3i value;
				double time;
				bool fetch_compressed_success = _fetch_compressed_by_index<3>(st->compressed_track, p_key_idx, value, time);
				ERR_FAIL_COND_V(!fetch_compressed_success, false);
				return time;
			}
			ERR_FAIL_INDEX_V(p_key_idx, st->scales.size(), -1);
			return st->scales[p_key_idx].time;
		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			if (bst->compressed_track >= 0) {
				Vector3i value;
				double time;
				bool fetch_compressed_success = _fetch_compressed_by_index<1>(bst->compressed_track, p_key_idx, value, time);
				ERR_FAIL_COND_V(!fetch_compressed_success, false);
				return time;
			}
			ERR_FAIL_INDEX_V(p_key_idx, bst->blend_shapes.size(), -1);
			return bst->blend_shapes[p_key_idx].time;
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
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());
			TKey<Vector3> key = tt->positions[p_key_idx];
			key.time = p_time;
			tt->positions.remove_at(p_key_idx);
			_insert(p_time, tt->positions, key);
			return;
		}
		case TYPE_ROTATION_3D: {
			RotationTrack *tt = static_cast<RotationTrack *>(t);
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->rotations.size());
			TKey<Quaternion> key = tt->rotations[p_key_idx];
			key.time = p_time;
			tt->rotations.remove_at(p_key_idx);
			_insert(p_time, tt->rotations, key);
			return;
		}
		case TYPE_SCALE_3D: {
			ScaleTrack *tt = static_cast<ScaleTrack *>(t);
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->scales.size());
			TKey<Vector3> key = tt->scales[p_key_idx];
			key.time = p_time;
			tt->scales.remove_at(p_key_idx);
			_insert(p_time, tt->scales, key);
			return;
		}
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *tt = static_cast<BlendShapeTrack *>(t);
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->blend_shapes.size());
			TKey<float> key = tt->blend_shapes[p_key_idx];
			key.time = p_time;
			tt->blend_shapes.remove_at(p_key_idx);
			_insert(p_time, tt->blend_shapes, key);
			return;
		}
		case TYPE_VALUE: {
			ValueTrack *vt = static_cast<ValueTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, vt->values.size());
			TKey<Variant> key = vt->values[p_key_idx];
			key.time = p_time;
			vt->values.remove_at(p_key_idx);
			_insert(p_time, vt->values, key);
			return;
		}
		case TYPE_METHOD: {
			MethodTrack *mt = static_cast<MethodTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, mt->methods.size());
			MethodKey key = mt->methods[p_key_idx];
			key.time = p_time;
			mt->methods.remove_at(p_key_idx);
			_insert(p_time, mt->methods, key);
			return;
		}
		case TYPE_BEZIER: {
			BezierTrack *bt = static_cast<BezierTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, bt->values.size());
			TKey<BezierKey> key = bt->values[p_key_idx];
			key.time = p_time;
			bt->values.remove_at(p_key_idx);
			_insert(p_time, bt->values, key);
			return;
		}
		case TYPE_AUDIO: {
			AudioTrack *at = static_cast<AudioTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());
			TKey<AudioKey> key = at->values[p_key_idx];
			key.time = p_time;
			at->values.remove_at(p_key_idx);
			_insert(p_time, at->values, key);
			return;
		}
		case TYPE_ANIMATION: {
			AnimationTrack *at = static_cast<AnimationTrack *>(t);
			ERR_FAIL_INDEX(p_key_idx, at->values.size());
			TKey<StringName> key = at->values[p_key_idx];
			key.time = p_time;
			at->values.remove_at(p_key_idx);
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
			if (tt->compressed_track >= 0) {
				return 1.0;
			}
			ERR_FAIL_INDEX_V(p_key_idx, tt->positions.size(), -1);
			return tt->positions[p_key_idx].transition;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			if (rt->compressed_track >= 0) {
				return 1.0;
			}
			ERR_FAIL_INDEX_V(p_key_idx, rt->rotations.size(), -1);
			return rt->rotations[p_key_idx].transition;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			if (st->compressed_track >= 0) {
				return 1.0;
			}
			ERR_FAIL_INDEX_V(p_key_idx, st->scales.size(), -1);
			return st->scales[p_key_idx].transition;
		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			if (bst->compressed_track >= 0) {
				return 1.0;
			}
			ERR_FAIL_INDEX_V(p_key_idx, bst->blend_shapes.size(), -1);
			return bst->blend_shapes[p_key_idx].transition;
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

bool Animation::track_is_compressed(int p_track) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), false);
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_POSITION_3D: {
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			return tt->compressed_track >= 0;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			return rt->compressed_track >= 0;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			return st->compressed_track >= 0;
		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			return bst->compressed_track >= 0;
		} break;
		default: {
			return false; //animation does not really use transitions
		} break;
	}

	ERR_FAIL_V(false);
}

void Animation::track_set_key_value(int p_track, int p_key_idx, const Variant &p_value) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];

	switch (t->type) {
		case TYPE_POSITION_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::VECTOR3) && (p_value.get_type() != Variant::VECTOR3I));
			PositionTrack *tt = static_cast<PositionTrack *>(t);
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());

			tt->positions.write[p_key_idx].value = p_value;

		} break;
		case TYPE_ROTATION_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::QUATERNION) && (p_value.get_type() != Variant::BASIS));
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_COND(rt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, rt->rotations.size());

			rt->rotations.write[p_key_idx].value = p_value;

		} break;
		case TYPE_SCALE_3D: {
			ERR_FAIL_COND((p_value.get_type() != Variant::VECTOR3) && (p_value.get_type() != Variant::VECTOR3I));
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_COND(st->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, st->scales.size());

			st->scales.write[p_key_idx].value = p_value;

		} break;
		case TYPE_BLEND_SHAPE: {
			ERR_FAIL_COND((p_value.get_type() != Variant::FLOAT) && (p_value.get_type() != Variant::INT));
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			ERR_FAIL_COND(bst->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, bst->blend_shapes.size());

			bst->blend_shapes.write[p_key_idx].value = p_value;

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
			ERR_FAIL_COND(arr.size() != 6);

			bt->values.write[p_key_idx].value.value = arr[0];
			bt->values.write[p_key_idx].value.in_handle.x = arr[1];
			bt->values.write[p_key_idx].value.in_handle.y = arr[2];
			bt->values.write[p_key_idx].value.out_handle.x = arr[3];
			bt->values.write[p_key_idx].value.out_handle.y = arr[4];
			bt->values.write[p_key_idx].value.handle_mode = static_cast<HandleMode>((int)arr[5]);

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
			ERR_FAIL_COND(tt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, tt->positions.size());
			tt->positions.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_ROTATION_3D: {
			RotationTrack *rt = static_cast<RotationTrack *>(t);
			ERR_FAIL_COND(rt->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, rt->rotations.size());
			rt->rotations.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_SCALE_3D: {
			ScaleTrack *st = static_cast<ScaleTrack *>(t);
			ERR_FAIL_COND(st->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, st->scales.size());
			st->scales.write[p_key_idx].transition = p_transition;
		} break;
		case TYPE_BLEND_SHAPE: {
			BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
			ERR_FAIL_COND(bst->compressed_track >= 0);
			ERR_FAIL_INDEX(p_key_idx, bst->blend_shapes.size());
			bst->blend_shapes.write[p_key_idx].transition = p_transition;
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

		return 0.5f *
				((p1 * 2.0f) +
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
	emit_changed();
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
					case TYPE_POSITION_3D: {
						const PositionTrack *tt = static_cast<const PositionTrack *>(t);
						if (tt->compressed_track >= 0) {
							_get_compressed_key_indices_in_range<3>(tt->compressed_track, from_time, length, p_indices);
							_get_compressed_key_indices_in_range<3>(tt->compressed_track, 0, to_time, p_indices);
						} else {
							_track_get_key_indices_in_range(tt->positions, from_time, length, p_indices);
							_track_get_key_indices_in_range(tt->positions, 0, to_time, p_indices);
						}
					} break;
					case TYPE_ROTATION_3D: {
						const RotationTrack *rt = static_cast<const RotationTrack *>(t);
						if (rt->compressed_track >= 0) {
							_get_compressed_key_indices_in_range<3>(rt->compressed_track, from_time, length, p_indices);
							_get_compressed_key_indices_in_range<3>(rt->compressed_track, 0, to_time, p_indices);
						} else {
							_track_get_key_indices_in_range(rt->rotations, from_time, length, p_indices);
							_track_get_key_indices_in_range(rt->rotations, 0, to_time, p_indices);
						}
					} break;
					case TYPE_SCALE_3D: {
						const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
						if (st->compressed_track >= 0) {
							_get_compressed_key_indices_in_range<3>(st->compressed_track, from_time, length, p_indices);
							_get_compressed_key_indices_in_range<3>(st->compressed_track, 0, to_time, p_indices);
						} else {
							_track_get_key_indices_in_range(st->scales, from_time, length, p_indices);
							_track_get_key_indices_in_range(st->scales, 0, to_time, p_indices);
						}
					} break;
					case TYPE_BLEND_SHAPE: {
						const BlendShapeTrack *bst = static_cast<const BlendShapeTrack *>(t);
						if (bst->compressed_track >= 0) {
							_get_compressed_key_indices_in_range<1>(bst->compressed_track, from_time, length, p_indices);
							_get_compressed_key_indices_in_range<1>(bst->compressed_track, 0, to_time, p_indices);
						} else {
							_track_get_key_indices_in_range(bst->blend_shapes, from_time, length, p_indices);
							_track_get_key_indices_in_range(bst->blend_shapes, 0, to_time, p_indices);
						}
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
						case TYPE_POSITION_3D: {
							const PositionTrack *tt = static_cast<const PositionTrack *>(t);
							if (tt->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(tt->compressed_track, 0, from_time, p_indices);
								_get_compressed_key_indices_in_range<3>(tt->compressed_track, 0, to_time, p_indices);
							} else {
								_track_get_key_indices_in_range(tt->positions, 0, from_time, p_indices);
								_track_get_key_indices_in_range(tt->positions, 0, to_time, p_indices);
							}
						} break;
						case TYPE_ROTATION_3D: {
							const RotationTrack *rt = static_cast<const RotationTrack *>(t);
							if (rt->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(rt->compressed_track, 0, from_time, p_indices);
								_get_compressed_key_indices_in_range<3>(rt->compressed_track, 0, to_time, p_indices);
							} else {
								_track_get_key_indices_in_range(rt->rotations, 0, from_time, p_indices);
								_track_get_key_indices_in_range(rt->rotations, 0, to_time, p_indices);
							}
						} break;
						case TYPE_SCALE_3D: {
							const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
							if (st->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(st->compressed_track, 0, from_time, p_indices);
								_get_compressed_key_indices_in_range<3>(st->compressed_track, 0, to_time, p_indices);
							} else {
								_track_get_key_indices_in_range(st->scales, 0, from_time, p_indices);
								_track_get_key_indices_in_range(st->scales, 0, to_time, p_indices);
							}
						} break;
						case TYPE_BLEND_SHAPE: {
							const BlendShapeTrack *bst = static_cast<const BlendShapeTrack *>(t);
							if (bst->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<1>(bst->compressed_track, 0, from_time, p_indices);
								_get_compressed_key_indices_in_range<1>(bst->compressed_track, 0, to_time, p_indices);
							} else {
								_track_get_key_indices_in_range(bst->blend_shapes, 0, from_time, p_indices);
								_track_get_key_indices_in_range(bst->blend_shapes, 0, to_time, p_indices);
							}
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
						case TYPE_POSITION_3D: {
							const PositionTrack *tt = static_cast<const PositionTrack *>(t);
							if (tt->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(tt->compressed_track, from_time, length, p_indices);
								_get_compressed_key_indices_in_range<3>(tt->compressed_track, to_time, length, p_indices);
							} else {
								_track_get_key_indices_in_range(tt->positions, from_time, length, p_indices);
								_track_get_key_indices_in_range(tt->positions, to_time, length, p_indices);
							}
						} break;
						case TYPE_ROTATION_3D: {
							const RotationTrack *rt = static_cast<const RotationTrack *>(t);
							if (rt->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(rt->compressed_track, from_time, length, p_indices);
								_get_compressed_key_indices_in_range<3>(rt->compressed_track, to_time, length, p_indices);
							} else {
								_track_get_key_indices_in_range(rt->rotations, from_time, length, p_indices);
								_track_get_key_indices_in_range(rt->rotations, to_time, length, p_indices);
							}
						} break;
						case TYPE_SCALE_3D: {
							const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
							if (st->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<3>(st->compressed_track, from_time, length, p_indices);
								_get_compressed_key_indices_in_range<3>(st->compressed_track, to_time, length, p_indices);
							} else {
								_track_get_key_indices_in_range(st->scales, from_time, length, p_indices);
								_track_get_key_indices_in_range(st->scales, to_time, length, p_indices);
							}
						} break;
						case TYPE_BLEND_SHAPE: {
							const BlendShapeTrack *bst = static_cast<const BlendShapeTrack *>(t);
							if (bst->compressed_track >= 0) {
								_get_compressed_key_indices_in_range<1>(bst->compressed_track, from_time, length, p_indices);
								_get_compressed_key_indices_in_range<1>(bst->compressed_track, to_time, length, p_indices);
							} else {
								_track_get_key_indices_in_range(bst->blend_shapes, from_time, length, p_indices);
								_track_get_key_indices_in_range(bst->blend_shapes, to_time, length, p_indices);
							}
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
		case TYPE_POSITION_3D: {
			const PositionTrack *tt = static_cast<const PositionTrack *>(t);
			if (tt->compressed_track >= 0) {
				_get_compressed_key_indices_in_range<3>(tt->compressed_track, from_time, to_time - from_time, p_indices);
			} else {
				_track_get_key_indices_in_range(tt->positions, from_time, to_time, p_indices);
			}
		} break;
		case TYPE_ROTATION_3D: {
			const RotationTrack *rt = static_cast<const RotationTrack *>(t);
			if (rt->compressed_track >= 0) {
				_get_compressed_key_indices_in_range<3>(rt->compressed_track, from_time, to_time - from_time, p_indices);
			} else {
				_track_get_key_indices_in_range(rt->rotations, from_time, to_time, p_indices);
			}
		} break;
		case TYPE_SCALE_3D: {
			const ScaleTrack *st = static_cast<const ScaleTrack *>(t);
			if (st->compressed_track >= 0) {
				_get_compressed_key_indices_in_range<3>(st->compressed_track, from_time, to_time - from_time, p_indices);
			} else {
				_track_get_key_indices_in_range(st->scales, from_time, to_time, p_indices);
			}
		} break;
		case TYPE_BLEND_SHAPE: {
			const BlendShapeTrack *bst = static_cast<const BlendShapeTrack *>(t);
			if (bst->compressed_track >= 0) {
				_get_compressed_key_indices_in_range<1>(bst->compressed_track, from_time, to_time - from_time, p_indices);
			} else {
				_track_get_key_indices_in_range(bst->blend_shapes, from_time, to_time, p_indices);
			}
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

int Animation::bezier_track_insert_key(int p_track, double p_time, real_t p_value, const Vector2 &p_in_handle, const Vector2 &p_out_handle, const HandleMode p_handle_mode) {
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
	k.value.handle_mode = p_handle_mode;

	int key = _insert(p_time, bt->values, k);

	emit_changed();

	return key;
}

void Animation::bezier_track_set_key_handle_mode(int p_track, int p_index, HandleMode p_mode, double p_balanced_value_time_ratio) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	bt->values.write[p_index].value.handle_mode = p_mode;

	if (p_mode == HANDLE_MODE_BALANCED) {
		Transform2D xform;
		xform.set_scale(Vector2(1.0, 1.0 / p_balanced_value_time_ratio));

		Vector2 vec_in = xform.xform(bt->values[p_index].value.in_handle);
		Vector2 vec_out = xform.xform(bt->values[p_index].value.out_handle);

		bt->values.write[p_index].value.in_handle = xform.affine_inverse().xform(-vec_out.normalized() * vec_in.length());
	}

	emit_changed();
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

void Animation::bezier_track_set_key_in_handle(int p_track, int p_index, const Vector2 &p_handle, double p_balanced_value_time_ratio) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	Vector2 in_handle = p_handle;
	if (in_handle.x > 0) {
		in_handle.x = 0;
	}
	bt->values.write[p_index].value.in_handle = in_handle;

	if (bt->values[p_index].value.handle_mode == HANDLE_MODE_BALANCED) {
		Transform2D xform;
		xform.set_scale(Vector2(1.0, 1.0 / p_balanced_value_time_ratio));

		Vector2 vec_out = xform.xform(bt->values[p_index].value.out_handle);
		Vector2 vec_in = xform.xform(in_handle);

		bt->values.write[p_index].value.out_handle = xform.affine_inverse().xform(-vec_in.normalized() * vec_out.length());
	}

	emit_changed();
}

void Animation::bezier_track_set_key_out_handle(int p_track, int p_index, const Vector2 &p_handle, double p_balanced_value_time_ratio) {
	ERR_FAIL_INDEX(p_track, tracks.size());
	Track *t = tracks[p_track];
	ERR_FAIL_COND(t->type != TYPE_BEZIER);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX(p_index, bt->values.size());

	Vector2 out_handle = p_handle;
	if (out_handle.x < 0) {
		out_handle.x = 0;
	}
	bt->values.write[p_index].value.out_handle = out_handle;

	if (bt->values[p_index].value.handle_mode == HANDLE_MODE_BALANCED) {
		Transform2D xform;
		xform.set_scale(Vector2(1.0, 1.0 / p_balanced_value_time_ratio));

		Vector2 vec_in = xform.xform(bt->values[p_index].value.in_handle);
		Vector2 vec_out = xform.xform(out_handle);

		bt->values.write[p_index].value.in_handle = xform.affine_inverse().xform(-vec_out.normalized() * vec_in.length());
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

int Animation::bezier_track_get_key_handle_mode(int p_track, int p_index) const {
	ERR_FAIL_INDEX_V(p_track, tracks.size(), 0);
	Track *t = tracks[p_track];
	ERR_FAIL_COND_V(t->type != TYPE_BEZIER, 0);

	BezierTrack *bt = static_cast<BezierTrack *>(t);

	ERR_FAIL_INDEX_V(p_index, bt->values.size(), 0);

	return bt->values[p_index].value.handle_mode;
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
	tracks.remove_at(p_track);
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
	ClassDB::bind_method(D_METHOD("find_track", "path", "type"), &Animation::find_track);

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
	ClassDB::bind_method(D_METHOD("blend_shape_track_insert_key", "track_idx", "time", "amount"), &Animation::blend_shape_track_insert_key);

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

	ClassDB::bind_method(D_METHOD("track_is_compressed", "track_idx"), &Animation::track_is_compressed);

	ClassDB::bind_method(D_METHOD("value_track_set_update_mode", "track_idx", "mode"), &Animation::value_track_set_update_mode);
	ClassDB::bind_method(D_METHOD("value_track_get_update_mode", "track_idx"), &Animation::value_track_get_update_mode);

	ClassDB::bind_method(D_METHOD("value_track_get_key_indices", "track_idx", "time_sec", "delta"), &Animation::_value_track_get_key_indices);
	ClassDB::bind_method(D_METHOD("value_track_interpolate", "track_idx", "time_sec"), &Animation::value_track_interpolate);

	ClassDB::bind_method(D_METHOD("method_track_get_key_indices", "track_idx", "time_sec", "delta"), &Animation::_method_track_get_key_indices);
	ClassDB::bind_method(D_METHOD("method_track_get_name", "track_idx", "key_idx"), &Animation::method_track_get_name);
	ClassDB::bind_method(D_METHOD("method_track_get_params", "track_idx", "key_idx"), &Animation::method_track_get_params);

	ClassDB::bind_method(D_METHOD("bezier_track_insert_key", "track_idx", "time", "value", "in_handle", "out_handle", "handle_mode"), &Animation::bezier_track_insert_key, DEFVAL(Vector2()), DEFVAL(Vector2()), DEFVAL(Animation::HandleMode::HANDLE_MODE_BALANCED));

	ClassDB::bind_method(D_METHOD("bezier_track_set_key_value", "track_idx", "key_idx", "value"), &Animation::bezier_track_set_key_value);
	ClassDB::bind_method(D_METHOD("bezier_track_set_key_in_handle", "track_idx", "key_idx", "in_handle", "balanced_value_time_ratio"), &Animation::bezier_track_set_key_in_handle, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("bezier_track_set_key_out_handle", "track_idx", "key_idx", "out_handle", "balanced_value_time_ratio"), &Animation::bezier_track_set_key_out_handle, DEFVAL(1.0));

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

	ClassDB::bind_method(D_METHOD("bezier_track_set_key_handle_mode", "track_idx", "key_idx", "key_handle_mode", "balanced_value_time_ratio"), &Animation::bezier_track_set_key_handle_mode, DEFVAL(1.0));
	ClassDB::bind_method(D_METHOD("bezier_track_get_key_handle_mode", "track_idx", "key_idx"), &Animation::bezier_track_get_key_handle_mode);

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

	ClassDB::bind_method(D_METHOD("compress", "page_size", "fps", "split_tolerance"), &Animation::compress, DEFVAL(8192), DEFVAL(120), DEFVAL(4.0));

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0.001,99999,0.001"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step", PROPERTY_HINT_RANGE, "0,4096,0.001"), "set_step", "get_step");

	ADD_SIGNAL(MethodInfo("tracks_changed"));

	BIND_ENUM_CONSTANT(TYPE_VALUE);
	BIND_ENUM_CONSTANT(TYPE_POSITION_3D);
	BIND_ENUM_CONSTANT(TYPE_ROTATION_3D);
	BIND_ENUM_CONSTANT(TYPE_SCALE_3D);
	BIND_ENUM_CONSTANT(TYPE_BLEND_SHAPE);
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

	BIND_ENUM_CONSTANT(HANDLE_MODE_FREE);
	BIND_ENUM_CONSTANT(HANDLE_MODE_BALANCED);
}

void Animation::clear() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
	tracks.clear();
	loop_mode = LOOP_NONE;
	length = 1;
	compression.enabled = false;
	compression.bounds.clear();
	compression.pages.clear();
	compression.fps = 120;
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

bool Animation::_blend_shape_track_optimize_key(const TKey<float> &t0, const TKey<float> &t1, const TKey<float> &t2, real_t p_allowed_unit_error) {
	float v0 = t0.value;
	float v1 = t1.value;
	float v2 = t2.value;

	if (Math::is_equal_approx(v1, v2, (float)p_allowed_unit_error)) {
		//0 and 2 are close, let's see if 1 is close
		if (!Math::is_equal_approx(v0, v1, (float)p_allowed_unit_error)) {
			//not close, not optimizable
			return false;
		}
	} else {
		/*
		TODO eventually discuss a way to optimize these better.
		float pd = (v2 - v0);
		real_t d0 = pd.dot(v0);
		real_t d1 = pd.dot(v1);
		real_t d2 = pd.dot(v2);
		if (d1 < d0 || d1 > d2) {
			return false; //beyond segment range
		}

		float s[2] = { v0, v2 };
		real_t d = Geometry3D::get_closest_point_to_segment(v1, s).distance_to(v1);

		if (d > pd.length() * p_allowed_linear_error) {
			return false; //beyond allowed error for colinearity
		}
*/
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

			tt->positions.remove_at(i);
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

			tt->rotations.remove_at(i);
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

			tt->scales.remove_at(i);
			i--;

		} else {
			prev_erased = false;
		}
	}
}

void Animation::_blend_shape_track_optimize(int p_idx, real_t p_allowed_linear_err) {
	ERR_FAIL_INDEX(p_idx, tracks.size());
	ERR_FAIL_COND(tracks[p_idx]->type != TYPE_BLEND_SHAPE);
	BlendShapeTrack *tt = static_cast<BlendShapeTrack *>(tracks[p_idx]);
	bool prev_erased = false;
	TKey<float> first_erased;
	first_erased.value = 0.0;

	for (int i = 1; i < tt->blend_shapes.size() - 1; i++) {
		TKey<float> &t0 = tt->blend_shapes.write[i - 1];
		TKey<float> &t1 = tt->blend_shapes.write[i];
		TKey<float> &t2 = tt->blend_shapes.write[i + 1];

		bool erase = _blend_shape_track_optimize_key(t0, t1, t2, p_allowed_linear_err);

		if (prev_erased && !_blend_shape_track_optimize_key(t0, first_erased, t2, p_allowed_linear_err)) {
			//avoid error to go beyond first erased key
			erase = false;
		}

		if (erase) {
			if (!prev_erased) {
				first_erased = t1;
				prev_erased = true;
			}

			tt->blend_shapes.remove_at(i);
			i--;

		} else {
			prev_erased = false;
		}
	}
}

void Animation::optimize(real_t p_allowed_linear_err, real_t p_allowed_angular_err, real_t p_max_optimizable_angle) {
	for (int i = 0; i < tracks.size(); i++) {
		if (track_is_compressed(i)) {
			continue; //not possible to optimize compressed track
		}
		if (tracks[i]->type == TYPE_POSITION_3D) {
			_position_track_optimize(i, p_allowed_linear_err, p_allowed_angular_err);
		} else if (tracks[i]->type == TYPE_ROTATION_3D) {
			_rotation_track_optimize(i, p_allowed_angular_err, p_max_optimizable_angle);
		} else if (tracks[i]->type == TYPE_SCALE_3D) {
			_scale_track_optimize(i, p_allowed_linear_err);
		} else if (tracks[i]->type == TYPE_BLEND_SHAPE) {
			_blend_shape_track_optimize(i, p_allowed_linear_err);
		}
	}
}

#define print_animc(m_str)
//#define print_animc(m_str) print_line(m_str);

struct AnimationCompressionDataState {
	enum {
		MIN_OPTIMIZE_PACKETS = 5,
		MAX_PACKETS = 16
	};

	uint32_t components = 3;
	LocalVector<uint8_t> data; //commited packets
	struct PacketData {
		int32_t data[3] = { 0, 0, 0 };
		uint32_t frame = 0;
	};

	float split_tolerance = 1.5;

	LocalVector<PacketData> temp_packets;

	//used for rollback if the new frame does not fit
	int32_t validated_packet_count = -1;

	static int32_t _compute_delta16_signed(int32_t p_from, int32_t p_to) {
		int32_t delta = p_to - p_from;
		if (delta > 32767) {
			return delta - 65536; // use wrap around
		} else if (delta < -32768) {
			return 65536 + delta; // use wrap around
		}
		return delta;
	}

	static uint32_t _compute_shift_bits_signed(int32_t p_delta) {
		if (p_delta == 0) {
			return 0;
		} else if (p_delta < 0) {
			p_delta = ABS(p_delta) - 1;
			if (p_delta == 0) {
				return 1;
			}
		}
		return nearest_shift(p_delta);
	}

	void _compute_max_shifts(uint32_t p_from, uint32_t p_to, uint32_t *max_shifts, uint32_t &max_frame_delta_shift) const {
		for (uint32_t j = 0; j < components; j++) {
			max_shifts[j] = 0;
		}
		max_frame_delta_shift = 0;

		for (uint32_t i = p_from + 1; i <= p_to; i++) {
			int32_t frame_delta = temp_packets[i].frame - temp_packets[i - 1].frame;
			max_frame_delta_shift = MAX(max_frame_delta_shift, nearest_shift(frame_delta));
			for (uint32_t j = 0; j < components; j++) {
				int32_t diff = _compute_delta16_signed(temp_packets[i - 1].data[j], temp_packets[i].data[j]);
				uint32_t shift = _compute_shift_bits_signed(diff);
				max_shifts[j] = MAX(shift, max_shifts[j]);
			}
		}
	}

	bool insert_key(uint32_t p_frame, const Vector3i &p_key) {
		if (temp_packets.size() == MAX_PACKETS) {
			commit_temp_packets();
		}
		PacketData packet;
		packet.frame = p_frame;
		for (int i = 0; i < 3; i++) {
			ERR_FAIL_COND_V(p_key[i] > 65535, false); // Sanity check
			packet.data[i] = p_key[i];
		}

		temp_packets.push_back(packet);

		if (temp_packets.size() >= MIN_OPTIMIZE_PACKETS) {
			uint32_t max_shifts[3] = { 0, 0, 0 }; // Base sizes, 16 bit
			uint32_t max_frame_delta_shift = 0;
			// Compute the average shift before the packet was added
			_compute_max_shifts(0, temp_packets.size() - 2, max_shifts, max_frame_delta_shift);

			float prev_packet_size_avg = 0;
			prev_packet_size_avg = float(1 << max_frame_delta_shift);
			for (uint32_t i = 0; i < components; i++) {
				prev_packet_size_avg += float(1 << max_shifts[i]);
			}
			prev_packet_size_avg /= float(1 + components);

			_compute_max_shifts(temp_packets.size() - 2, temp_packets.size() - 1, max_shifts, max_frame_delta_shift);

			float new_packet_size_avg = 0;
			new_packet_size_avg = float(1 << max_frame_delta_shift);
			for (uint32_t i = 0; i < components; i++) {
				new_packet_size_avg += float(1 << max_shifts[i]);
			}
			new_packet_size_avg /= float(1 + components);

			print_animc("packet count: " + rtos(temp_packets.size() - 1) + " size avg " + rtos(prev_packet_size_avg) + " new avg " + rtos(new_packet_size_avg));
			float ratio = (prev_packet_size_avg < new_packet_size_avg) ? (new_packet_size_avg / prev_packet_size_avg) : (prev_packet_size_avg / new_packet_size_avg);

			if (ratio > split_tolerance) {
				print_animc("split!");
				temp_packets.resize(temp_packets.size() - 1);
				commit_temp_packets();
				temp_packets.push_back(packet);
			}
		}

		return temp_packets.size() == 1; // First key
	}

	uint32_t get_temp_packet_size() const {
		if (temp_packets.size() == 0) {
			return 0;
		} else if (temp_packets.size() == 1) {
			return components == 1 ? 4 : 8; // 1 component packet is 16 bits and 16 bits unused. 3 component packets is 48 bits and 16 bits unused
		}
		uint32_t max_shifts[3] = { 0, 0, 0 }; //base sizes, 16 bit
		uint32_t max_frame_delta_shift = 0;

		_compute_max_shifts(0, temp_packets.size() - 1, max_shifts, max_frame_delta_shift);

		uint32_t size_bits = 16; //base value (all 4 bits of shift sizes for x,y,z,time)
		size_bits += max_frame_delta_shift * (temp_packets.size() - 1); //times
		for (uint32_t j = 0; j < components; j++) {
			size_bits += 16; //base value
			uint32_t shift = max_shifts[j];
			if (shift > 0) {
				shift += 1; //if not zero, add sign bit
			}
			size_bits += shift * (temp_packets.size() - 1);
		}
		if (size_bits % 8 != 0) { //wrap to 8 bits
			size_bits += 8 - (size_bits % 8);
		}
		uint32_t size_bytes = size_bits / 8; //wrap to words
		if (size_bytes % 4 != 0) {
			size_bytes += 4 - (size_bytes % 4);
		}
		return size_bytes;
	}

	static void _push_bits(LocalVector<uint8_t> &data, uint32_t &r_buffer, uint32_t &r_bits_used, uint32_t p_value, uint32_t p_bits) {
		r_buffer |= p_value << r_bits_used;
		r_bits_used += p_bits;
		while (r_bits_used >= 8) {
			uint8_t byte = r_buffer & 0xFF;
			data.push_back(byte);
			r_buffer >>= 8;
			r_bits_used -= 8;
		}
	}

	void commit_temp_packets() {
		if (temp_packets.size() == 0) {
			return; //nohing to do
		}
#define DEBUG_PACKET_PUSH
#ifdef DEBUG_PACKET_PUSH
#ifndef _MSC_VER
#warning Debugging packet push, disable this code in production to gain a bit more import performance.
#endif
		uint32_t debug_packet_push = get_temp_packet_size();
		uint32_t debug_data_size = data.size();
#endif
		// Store header

		uint8_t header[8];
		uint32_t header_bytes = 0;
		for (uint32_t i = 0; i < components; i++) {
			encode_uint16(temp_packets[0].data[i], &header[header_bytes]);
			header_bytes += 2;
		}

		uint32_t max_shifts[3] = { 0, 0, 0 }; //base sizes, 16 bit
		uint32_t max_frame_delta_shift = 0;

		if (temp_packets.size() > 1) {
			_compute_max_shifts(0, temp_packets.size() - 1, max_shifts, max_frame_delta_shift);
			uint16_t shift_header = (max_frame_delta_shift - 1) << 12;
			for (uint32_t i = 0; i < components; i++) {
				shift_header |= max_shifts[i] << (4 * i);
			}

			encode_uint16(shift_header, &header[header_bytes]);
			header_bytes += 2;
		}

		while (header_bytes % 4 != 0) {
			header[header_bytes++] = 0;
		}

		for (uint32_t i = 0; i < header_bytes; i++) {
			data.push_back(header[i]);
		}

		if (temp_packets.size() == 1) {
			temp_packets.clear();
			validated_packet_count = 0;
			return; //only header stored, nothing else to do
		}

		uint32_t bit_buffer = 0;
		uint32_t bits_used = 0;

		for (uint32_t i = 1; i < temp_packets.size(); i++) {
			uint32_t frame_delta = temp_packets[i].frame - temp_packets[i - 1].frame;
			_push_bits(data, bit_buffer, bits_used, frame_delta, max_frame_delta_shift);

			for (uint32_t j = 0; j < components; j++) {
				if (max_shifts[j] == 0) {
					continue; // Zero delta, do not store
				}
				int32_t delta = _compute_delta16_signed(temp_packets[i - 1].data[j], temp_packets[i].data[j]);

				ERR_FAIL_COND(delta < -32768 || delta > 32767); //sanity check

				uint16_t deltau;
				if (delta < 0) {
					deltau = (ABS(delta) - 1) | (1 << max_shifts[j]);
				} else {
					deltau = delta;
				}
				_push_bits(data, bit_buffer, bits_used, deltau, max_shifts[j] + 1); // Include sign bit
			}
		}
		if (bits_used != 0) {
			ERR_FAIL_COND(bit_buffer > 0xFF); // Sanity check
			data.push_back(bit_buffer);
		}

		while (data.size() % 4 != 0) {
			data.push_back(0); //pad to align with 4
		}

		temp_packets.clear();
		validated_packet_count = 0;

#ifdef DEBUG_PACKET_PUSH
		ERR_FAIL_COND((data.size() - debug_data_size) != debug_packet_push);
#endif
	}
};

struct AnimationCompressionTimeState {
	struct Packet {
		uint32_t frame;
		uint32_t offset;
		uint32_t count;
	};

	LocalVector<Packet> packets;
	//used for rollback
	int32_t key_index = 0;
	int32_t validated_packet_count = 0;
	int32_t validated_key_index = -1;
	bool needs_start_frame = false;
};

Vector3i Animation::_compress_key(uint32_t p_track, const AABB &p_bounds, int32_t p_key, float p_time) {
	Vector3i values;
	TrackType tt = track_get_type(p_track);
	switch (tt) {
		case TYPE_POSITION_3D: {
			Vector3 pos;
			if (p_key >= 0) {
				position_track_get_key(p_track, p_key, &pos);
			} else {
				position_track_interpolate(p_track, p_time, &pos);
			}
			pos = (pos - p_bounds.position) / p_bounds.size;
			for (int j = 0; j < 3; j++) {
				values[j] = CLAMP(int32_t(pos[j] * 65535.0), 0, 65535);
			}
		} break;
		case TYPE_ROTATION_3D: {
			Quaternion rot;
			if (p_key >= 0) {
				rotation_track_get_key(p_track, p_key, &rot);
			} else {
				rotation_track_interpolate(p_track, p_time, &rot);
			}
			Vector3 axis = rot.get_axis();
			float angle = rot.get_angle();
			angle = Math::fposmod(double(angle), double(Math_PI * 2.0));
			Vector2 oct = axis.octahedron_encode();
			Vector3 rot_norm(oct.x, oct.y, angle / (Math_PI * 2.0)); // high resolution rotation in 0-1 angle.

			for (int j = 0; j < 3; j++) {
				values[j] = CLAMP(int32_t(rot_norm[j] * 65535.0), 0, 65535);
			}
		} break;
		case TYPE_SCALE_3D: {
			Vector3 scale;
			if (p_key >= 0) {
				scale_track_get_key(p_track, p_key, &scale);
			} else {
				scale_track_interpolate(p_track, p_time, &scale);
			}
			scale = (scale - p_bounds.position) / p_bounds.size;
			for (int j = 0; j < 3; j++) {
				values[j] = CLAMP(int32_t(scale[j] * 65535.0), 0, 65535);
			}
		} break;
		case TYPE_BLEND_SHAPE: {
			float blend;
			if (p_key >= 0) {
				blend_shape_track_get_key(p_track, p_key, &blend);
			} else {
				blend_shape_track_interpolate(p_track, p_time, &blend);
			}

			blend = (blend / float(Compression::BLEND_SHAPE_RANGE)) * 0.5 + 0.5;
			values[0] = CLAMP(int32_t(blend * 65535.0), 0, 65535);
		} break;
		default: {
			ERR_FAIL_V(Vector3i()); //sanity check
		} break;
	}

	return values;
}

struct AnimationCompressionBufferBitsRead {
	uint32_t buffer = 0;
	uint32_t used = 0;
	const uint8_t *src_data = nullptr;

	_FORCE_INLINE_ uint32_t read(uint32_t p_bits) {
		uint32_t output = 0;
		uint32_t written = 0;
		while (p_bits > 0) {
			if (used == 0) {
				used = 8;
				buffer = *src_data;
				src_data++;
			}
			uint32_t to_write = MIN(used, p_bits);
			output |= (buffer & ((1 << to_write) - 1)) << written;
			buffer >>= to_write;
			used -= to_write;
			p_bits -= to_write;
			written += to_write;
		}
		return output;
	}
};

void Animation::compress(uint32_t p_page_size, uint32_t p_fps, float p_split_tolerance) {
	ERR_FAIL_COND_MSG(compression.enabled, "This animation is already compressed");

	p_split_tolerance = CLAMP(p_split_tolerance, 1.1, 8.0);
	compression.pages.clear();

	uint32_t base_page_size = 0; // Before compressing pages, compute how large the "end page" datablock is.
	LocalVector<uint32_t> tracks_to_compress;
	LocalVector<AABB> track_bounds;
	const uint32_t time_packet_size = 4;

	const uint32_t track_header_size = 4 + 4 + 4; // pointer to time (4 bytes), amount of time keys (4 bytes) pointer to track data (4 bytes)

	for (int i = 0; i < get_track_count(); i++) {
		TrackType type = track_get_type(i);
		if (type != TYPE_POSITION_3D && type != TYPE_ROTATION_3D && type != TYPE_SCALE_3D && type != TYPE_BLEND_SHAPE) {
			continue;
		}
		if (track_get_key_count(i) == 0) {
			continue; //do not compress, no keys
		}
		base_page_size += track_header_size; //pointer to beginning of each track timeline and amount of time keys
		base_page_size += time_packet_size; //for end of track time marker
		base_page_size += (type == TYPE_BLEND_SHAPE) ? 4 : 8; // at least the end of track packet (at much 8 bytes). This could be less, but have to be pessimistic.
		tracks_to_compress.push_back(i);

		AABB bounds;

		if (type == TYPE_POSITION_3D) {
			AABB aabb;
			int kcount = track_get_key_count(i);
			for (int j = 0; j < kcount; j++) {
				Vector3 pos;
				position_track_get_key(i, j, &pos);
				if (j == 0) {
					aabb.position = pos;
				} else {
					aabb.expand_to(pos);
				}
			}
			for (int j = 0; j < 3; j++) {
				//cant have zero
				if (aabb.size[j] < CMP_EPSILON) {
					aabb.size[j] = CMP_EPSILON;
				}
			}
			bounds = aabb;
		}
		if (type == TYPE_SCALE_3D) {
			AABB aabb;
			int kcount = track_get_key_count(i);
			for (int j = 0; j < kcount; j++) {
				Vector3 scale;
				scale_track_get_key(i, j, &scale);
				if (j == 0) {
					aabb.position = scale;
				} else {
					aabb.expand_to(scale);
				}
			}
			for (int j = 0; j < 3; j++) {
				//cant have zero
				if (aabb.size[j] < CMP_EPSILON) {
					aabb.size[j] = CMP_EPSILON;
				}
			}
			bounds = aabb;
		}

		track_bounds.push_back(bounds);
	}

	if (tracks_to_compress.size() == 0) {
		return; //nothing to compress
	}

	print_animc("Anim Compression:");
	print_animc("-----------------");
	print_animc("Tracks to compress: " + itos(tracks_to_compress.size()));

	uint32_t current_frame = 0;
	uint32_t base_page_frame = 0;
	double frame_len = 1.0 / double(p_fps);
	const uint32_t max_frames_per_page = 65536;

	print_animc("Frame Len: " + rtos(frame_len));

	LocalVector<AnimationCompressionDataState> data_tracks;
	LocalVector<AnimationCompressionTimeState> time_tracks;

	data_tracks.resize(tracks_to_compress.size());
	time_tracks.resize(tracks_to_compress.size());

	for (uint32_t i = 0; i < data_tracks.size(); i++) {
		data_tracks[i].split_tolerance = p_split_tolerance;
		if (track_get_type(tracks_to_compress[i]) == TYPE_BLEND_SHAPE) {
			data_tracks[i].components = 1;
		} else {
			data_tracks[i].components = 3;
		}
	}

	while (true) {
		// Begin by finding the keyframe in all tracks with the time closest to the current time
		const uint32_t FRAME_MAX = 0xFFFFFFFF;
		const int32_t NO_TRACK_FOUND = -1;
		uint32_t best_frame = FRAME_MAX;
		uint32_t best_invalid_frame = FRAME_MAX;
		int32_t best_frame_track = NO_TRACK_FOUND; // Default is -1, which means all keyframes for this page are exhausted.
		bool start_frame = false;

		for (uint32_t i = 0; i < tracks_to_compress.size(); i++) {
			uint32_t uncomp_track = tracks_to_compress[i];

			if (time_tracks[i].key_index == track_get_key_count(uncomp_track)) {
				if (time_tracks[i].needs_start_frame) {
					start_frame = true;
					best_frame = base_page_frame;
					best_frame_track = i;
					time_tracks[i].needs_start_frame = false;
					break;
				} else {
					continue; // This track is exhausted (all keys were added already), don't consider.
				}
			}

			uint32_t key_frame = double(track_get_key_time(uncomp_track, time_tracks[i].key_index)) / frame_len;

			if (time_tracks[i].needs_start_frame && key_frame > base_page_frame) {
				start_frame = true;
				best_frame = base_page_frame;
				best_frame_track = i;
				time_tracks[i].needs_start_frame = false;
				break;
			}

			ERR_FAIL_COND(key_frame < base_page_frame); // Sanity check, should never happen

			if (key_frame - base_page_frame >= max_frames_per_page) {
				// Invalid because beyond the max frames allowed per page
				best_invalid_frame = MIN(best_invalid_frame, key_frame);
			} else if (key_frame < best_frame) {
				best_frame = key_frame;
				best_frame_track = i;
			}
		}

		print_animc("*KEY*: Current Frame: " + itos(current_frame) + " Best Frame: " + rtos(best_frame) + " Best Track: " + itos(best_frame_track) + " Start: " + String(start_frame ? "true" : "false"));

		if (!start_frame && best_frame > current_frame) {
			// Any case where the current frame advanced, either because nothing was found or because something was found greater than the current one.
			print_animc("\tAdvance Condition.");
			bool rollback = false;

			// The frame has advanced, time to validate the previous frame
			uint32_t current_page_size = base_page_size;
			for (uint32_t i = 0; i < data_tracks.size(); i++) {
				uint32_t track_size = data_tracks[i].data.size(); // track size
				track_size += data_tracks[i].get_temp_packet_size(); // Add the temporary data
				if (track_size > Compression::MAX_DATA_TRACK_SIZE) {
					rollback = true; //track to large, time track can't point to keys any longer, because key offset is 12 bits
					break;
				}
				current_page_size += track_size;
			}
			for (uint32_t i = 0; i < time_tracks.size(); i++) {
				current_page_size += time_tracks[i].packets.size() * 4; // time packet is 32 bits
			}

			if (!rollback && current_page_size > p_page_size) {
				rollback = true;
			}

			print_animc("\tCurrent Page Size: " + itos(current_page_size) + "/" + itos(p_page_size) + " Rollback? " + String(rollback ? "YES!" : "no"));

			if (rollback) {
				// Not valid any longer, so rollback and commit page

				for (uint32_t i = 0; i < data_tracks.size(); i++) {
					data_tracks[i].temp_packets.resize(data_tracks[i].validated_packet_count);
				}
				for (uint32_t i = 0; i < time_tracks.size(); i++) {
					time_tracks[i].key_index = time_tracks[i].validated_key_index; //rollback key
					time_tracks[i].packets.resize(time_tracks[i].validated_packet_count);
				}

			} else {
				// All valid, so save rollback information
				for (uint32_t i = 0; i < data_tracks.size(); i++) {
					data_tracks[i].validated_packet_count = data_tracks[i].temp_packets.size();
				}
				for (uint32_t i = 0; i < time_tracks.size(); i++) {
					time_tracks[i].validated_key_index = time_tracks[i].key_index;
					time_tracks[i].validated_packet_count = time_tracks[i].packets.size();
				}

				// Accept this frame as the frame being processed (as long as it exists)
				if (best_frame != FRAME_MAX) {
					current_frame = best_frame;
					print_animc("\tValidated, New Current Frame: " + itos(current_frame));
				}
			}

			if (rollback || best_frame == FRAME_MAX) {
				// Commit the page if had to rollback or if no track was found
				print_animc("\tCommiting page..");

				// The end frame for the page depends entirely on whether its valid or
				// no more keys were found.
				// If not valid, then the end frame is the current frame (as this means the current frame is being rolled back
				// If valid, then the end frame is the next invalid one (in case more frames exist), or the current frame in case no more frames exist.
				uint32_t page_end_frame = (rollback || best_frame == FRAME_MAX) ? current_frame : best_invalid_frame;

				print_animc("\tEnd Frame: " + itos(page_end_frame) + ", " + rtos(page_end_frame * frame_len) + "s");

				// Add finalizer frames and commit pending tracks
				uint32_t finalizer_local_frame = page_end_frame - base_page_frame;

				uint32_t total_page_size = 0;

				for (uint32_t i = 0; i < data_tracks.size(); i++) {
					if (data_tracks[i].temp_packets.size() == 0 || (data_tracks[i].temp_packets[data_tracks[i].temp_packets.size() - 1].frame) < finalizer_local_frame) {
						// Add finalizer frame if it makes sense
						Vector3i values = _compress_key(tracks_to_compress[i], track_bounds[i], -1, page_end_frame * frame_len);

						bool first_key = data_tracks[i].insert_key(finalizer_local_frame, values);
						if (first_key) {
							AnimationCompressionTimeState::Packet p;
							p.count = 1;
							p.frame = finalizer_local_frame;
							p.offset = data_tracks[i].data.size();
							time_tracks[i].packets.push_back(p);
						} else {
							ERR_FAIL_COND(time_tracks[i].packets.size() == 0);
							time_tracks[i].packets[time_tracks[i].packets.size() - 1].count++;
						}
					}

					data_tracks[i].commit_temp_packets();
					total_page_size += data_tracks[i].data.size();
					total_page_size += time_tracks[i].packets.size() * 4;
					total_page_size += track_header_size;

					print_animc("\tTrack " + itos(i) + " time packets: " + itos(time_tracks[i].packets.size()) + " Packet data: " + itos(data_tracks[i].data.size()));
				}

				print_animc("\tTotal page Size: " + itos(total_page_size) + "/" + itos(p_page_size));

				// Create Page
				Vector<uint8_t> page_data;
				page_data.resize(total_page_size);
				{
					uint8_t *page_ptr = page_data.ptrw();
					uint32_t base_offset = data_tracks.size() * track_header_size;

					for (uint32_t i = 0; i < data_tracks.size(); i++) {
						encode_uint32(base_offset, page_ptr + (track_header_size * i + 0));
						uint16_t *key_time_ptr = (uint16_t *)(page_ptr + base_offset);
						for (uint32_t j = 0; j < time_tracks[i].packets.size(); j++) {
							key_time_ptr[j * 2 + 0] = uint16_t(time_tracks[i].packets[j].frame);
							uint16_t ptr = time_tracks[i].packets[j].offset / 4;
							ptr |= (time_tracks[i].packets[j].count - 1) << 12;
							key_time_ptr[j * 2 + 1] = ptr;
							base_offset += 4;
						}
						encode_uint32(time_tracks[i].packets.size(), page_ptr + (track_header_size * i + 4));
						encode_uint32(base_offset, page_ptr + (track_header_size * i + 8));
						memcpy(page_ptr + base_offset, data_tracks[i].data.ptr(), data_tracks[i].data.size());
						base_offset += data_tracks[i].data.size();

						//reset track
						data_tracks[i].data.clear();
						data_tracks[i].temp_packets.clear();
						data_tracks[i].validated_packet_count = -1;

						time_tracks[i].needs_start_frame = true; //Not required the first time, but from now on it is.
						time_tracks[i].packets.clear();
						time_tracks[i].validated_key_index = -1;
						time_tracks[i].validated_packet_count = 0;
					}
				}

				Compression::Page page;
				page.data = page_data;
				page.time_offset = base_page_frame * frame_len;
				compression.pages.push_back(page);

				if (!rollback && best_invalid_frame == FRAME_MAX) {
					break; // No more pages to add.
				}

				current_frame = page_end_frame;
				base_page_frame = page_end_frame;

				continue; // Start over
			}
		}

		// A key was found for the current frame and all is ok

		uint32_t comp_track = best_frame_track;
		Vector3i values;

		if (start_frame) {
			// Interpolate
			values = _compress_key(tracks_to_compress[comp_track], track_bounds[comp_track], -1, base_page_frame * frame_len);
		} else {
			uint32_t key = time_tracks[comp_track].key_index;
			values = _compress_key(tracks_to_compress[comp_track], track_bounds[comp_track], key);
			time_tracks[comp_track].key_index++; //goto next key (but could be rolled back if beyond page size).
		}

		bool first_key = data_tracks[comp_track].insert_key(best_frame - base_page_frame, values);
		if (first_key) {
			AnimationCompressionTimeState::Packet p;
			p.count = 1;
			p.frame = best_frame - base_page_frame;
			p.offset = data_tracks[comp_track].data.size();
			time_tracks[comp_track].packets.push_back(p);
		} else {
			ERR_CONTINUE(time_tracks[comp_track].packets.size() == 0);
			time_tracks[comp_track].packets[time_tracks[comp_track].packets.size() - 1].count++;
		}
	}

	compression.bounds = track_bounds;
	compression.fps = p_fps;
	compression.enabled = true;

	for (uint32_t i = 0; i < tracks_to_compress.size(); i++) {
		Track *t = tracks[tracks_to_compress[i]];
		t->interpolation = INTERPOLATION_LINEAR; //only linear supported
		switch (t->type) {
			case TYPE_POSITION_3D: {
				PositionTrack *tt = static_cast<PositionTrack *>(t);
				tt->positions.clear();
				tt->compressed_track = i;
			} break;
			case TYPE_ROTATION_3D: {
				RotationTrack *rt = static_cast<RotationTrack *>(t);
				rt->rotations.clear();
				rt->compressed_track = i;
			} break;
			case TYPE_SCALE_3D: {
				ScaleTrack *st = static_cast<ScaleTrack *>(t);
				st->scales.clear();
				st->compressed_track = i;
				print_line("Scale Bounds " + itos(i) + ": " + track_bounds[i]);
			} break;
			case TYPE_BLEND_SHAPE: {
				BlendShapeTrack *bst = static_cast<BlendShapeTrack *>(t);
				bst->blend_shapes.clear();
				bst->compressed_track = i;
			} break;
			default: {
			}
		}
	}
#if 1
	uint32_t orig_size = 0;
	for (int i = 0; i < get_track_count(); i++) {
		switch (track_get_type(i)) {
			case TYPE_SCALE_3D:
			case TYPE_POSITION_3D: {
				orig_size += sizeof(TKey<Vector3>) * track_get_key_count(i);
			} break;
			case TYPE_ROTATION_3D: {
				orig_size += sizeof(TKey<Quaternion>) * track_get_key_count(i);
			} break;
			case TYPE_BLEND_SHAPE: {
				orig_size += sizeof(TKey<float>) * track_get_key_count(i);
			} break;
			default: {
			}
		}
	}

	uint32_t new_size = 0;
	for (uint32_t i = 0; i < compression.pages.size(); i++) {
		new_size += compression.pages[i].data.size();
	}

	print_line("Original size: " + itos(orig_size) + " - Compressed size: " + itos(new_size) + " " + String::num(float(new_size) / float(orig_size) * 100, 2) + "% pages: " + itos(compression.pages.size()));
#endif
}

bool Animation::_rotation_interpolate_compressed(uint32_t p_compressed_track, double p_time, Quaternion &r_ret) const {
	Vector3i current;
	Vector3i next;
	double time_current;
	double time_next;

	if (!_fetch_compressed<3>(p_compressed_track, p_time, current, time_current, next, time_next)) {
		return false; //some sort of problem
	}

	if (time_current >= p_time || time_current == time_next) {
		r_ret = _uncompress_quaternion(current);
	} else if (p_time >= time_next) {
		r_ret = _uncompress_quaternion(next);
	} else {
		double c = (p_time - time_current) / (time_next - time_current);
		Quaternion from = _uncompress_quaternion(current);
		Quaternion to = _uncompress_quaternion(next);
		r_ret = from.slerp(to, c);
	}

	return true;
}

bool Animation::_pos_scale_interpolate_compressed(uint32_t p_compressed_track, double p_time, Vector3 &r_ret) const {
	Vector3i current;
	Vector3i next;
	double time_current;
	double time_next;

	if (!_fetch_compressed<3>(p_compressed_track, p_time, current, time_current, next, time_next)) {
		return false; //some sort of problem
	}

	if (time_current >= p_time || time_current == time_next) {
		r_ret = _uncompress_pos_scale(p_compressed_track, current);
	} else if (p_time >= time_next) {
		r_ret = _uncompress_pos_scale(p_compressed_track, next);
	} else {
		double c = (p_time - time_current) / (time_next - time_current);
		Vector3 from = _uncompress_pos_scale(p_compressed_track, current);
		Vector3 to = _uncompress_pos_scale(p_compressed_track, next);
		r_ret = from.lerp(to, c);
	}

	return true;
}
bool Animation::_blend_shape_interpolate_compressed(uint32_t p_compressed_track, double p_time, float &r_ret) const {
	Vector3i current;
	Vector3i next;
	double time_current;
	double time_next;

	if (!_fetch_compressed<1>(p_compressed_track, p_time, current, time_current, next, time_next)) {
		return false; //some sort of problem
	}

	if (time_current >= p_time || time_current == time_next) {
		r_ret = _uncompress_blend_shape(current);
	} else if (p_time >= time_next) {
		r_ret = _uncompress_blend_shape(next);
	} else {
		float c = (p_time - time_current) / (time_next - time_current);
		float from = _uncompress_blend_shape(current);
		float to = _uncompress_blend_shape(next);
		r_ret = Math::lerp(from, to, c);
	}

	return true;
}

template <uint32_t COMPONENTS>
bool Animation::_fetch_compressed(uint32_t p_compressed_track, double p_time, Vector3i &r_current_value, double &r_current_time, Vector3i &r_next_value, double &r_next_time, uint32_t *key_index) const {
	ERR_FAIL_COND_V(!compression.enabled, false);
	ERR_FAIL_UNSIGNED_INDEX_V(p_compressed_track, compression.bounds.size(), false);
	p_time = CLAMP(p_time, 0, length);
	if (key_index) {
		*key_index = 0;
	}

	double frame_to_sec = 1.0 / double(compression.fps);

	int32_t page_index = -1;
	for (uint32_t i = 0; i < compression.pages.size(); i++) {
		if (compression.pages[i].time_offset > p_time) {
			break;
		}
		page_index = i;
	}

	ERR_FAIL_COND_V(page_index == -1, false); //should not happen

	double page_base_time = compression.pages[page_index].time_offset;
	const uint8_t *page_data = compression.pages[page_index].data.ptr();
#ifndef _MSC_VER
#warning Little endian assumed. No major big endian hardware exists any longer, but in case it does it will need to be supported
#endif
	const uint32_t *indices = (const uint32_t *)page_data;
	const uint16_t *time_keys = (const uint16_t *)&page_data[indices[p_compressed_track * 3 + 0]];
	uint32_t time_key_count = indices[p_compressed_track * 3 + 1];

	int32_t packet_idx = 0;
	double packet_time = double(time_keys[0]) * frame_to_sec + page_base_time;
	uint32_t base_frame = time_keys[0];

	for (uint32_t i = 1; i < time_key_count; i++) {
		uint32_t f = time_keys[i * 2 + 0];
		double frame_time = double(f) * frame_to_sec + page_base_time;

		if (frame_time > p_time) {
			break;
		}

		if (key_index) {
			(*key_index) += (time_keys[(i - 1) * 2 + 1] >> 12) + 1;
		}

		packet_idx = i;
		packet_time = frame_time;
		base_frame = f;
	}

	const uint8_t *data_keys_base = (const uint8_t *)&page_data[indices[p_compressed_track * 3 + 2]];

	uint16_t time_key_data = time_keys[packet_idx * 2 + 1];
	uint32_t data_offset = (time_key_data & 0xFFF) * 4; // lower 12 bits
	uint32_t data_count = (time_key_data >> 12) + 1;

	const uint16_t *data_key = (const uint16_t *)(data_keys_base + data_offset);

	uint16_t decode[COMPONENTS];
	uint16_t decode_next[COMPONENTS];

	for (uint32_t i = 0; i < COMPONENTS; i++) {
		decode[i] = data_key[i];
		decode_next[i] = data_key[i];
	}

	double next_time = packet_time;

	if (p_time > packet_time) { // If its equal or less, then don't bother
		if (data_count > 1) {
			//decode forward
			uint32_t bit_width[COMPONENTS];
			for (uint32_t i = 0; i < COMPONENTS; i++) {
				bit_width[i] = (data_key[COMPONENTS] >> (i * 4)) & 0xF;
			}

			uint32_t frame_bit_width = (data_key[COMPONENTS] >> 12) + 1;

			AnimationCompressionBufferBitsRead buffer;

			buffer.src_data = (const uint8_t *)&data_key[COMPONENTS + 1];

			for (uint32_t i = 1; i < data_count; i++) {
				uint32_t frame_delta = buffer.read(frame_bit_width);
				base_frame += frame_delta;

				for (uint32_t j = 0; j < COMPONENTS; j++) {
					if (bit_width[j] == 0) {
						continue; // do none
					}
					uint32_t valueu = buffer.read(bit_width[j] + 1);
					bool sign = valueu & (1 << bit_width[j]);
					int16_t value = valueu & ((1 << bit_width[j]) - 1);
					if (sign) {
						value = -value - 1;
					}

					decode_next[j] += value;
				}

				next_time = double(base_frame) * frame_to_sec + page_base_time;
				if (p_time < next_time) {
					break;
				}

				packet_time = next_time;

				for (uint32_t j = 0; j < COMPONENTS; j++) {
					decode[j] = decode_next[j];
				}

				if (key_index) {
					(*key_index)++;
				}
			}
		}

		if (p_time > next_time) { // > instead of >= because if its equal, then it will be properly interpolated anyway
			// So, the last frame found still has a time that is less than the required frame,
			// will have to interpolate with the first frame of the next timekey.

			if ((uint32_t)packet_idx < time_key_count - 1) { // Sanity check but should not matter much, otherwise current next packet is last packet

				uint16_t time_key_data_next = time_keys[(packet_idx + 1) * 2 + 1];
				uint32_t data_offset_next = (time_key_data_next & 0xFFF) * 4; // Lower 12 bits

				const uint16_t *data_key_next = (const uint16_t *)(data_keys_base + data_offset_next);
				base_frame = time_keys[(packet_idx + 1) * 2 + 0];
				next_time = double(base_frame) * frame_to_sec + page_base_time;
				for (uint32_t i = 0; i < COMPONENTS; i++) {
					decode_next[i] = data_key_next[i];
				}
			}
		}
	}

	r_current_time = packet_time;
	r_next_time = next_time;

	for (uint32_t i = 0; i < COMPONENTS; i++) {
		r_current_value[i] = decode[i];
		r_next_value[i] = decode_next[i];
	}

	return true;
}

template <uint32_t COMPONENTS>
void Animation::_get_compressed_key_indices_in_range(uint32_t p_compressed_track, double p_time, double p_delta, List<int> *r_indices) const {
	ERR_FAIL_COND(!compression.enabled);
	ERR_FAIL_UNSIGNED_INDEX(p_compressed_track, compression.bounds.size());

	double frame_to_sec = 1.0 / double(compression.fps);
	uint32_t key_index = 0;

	for (uint32_t p = 0; p < compression.pages.size(); p++) {
		if (compression.pages[p].time_offset >= p_time + p_delta) {
			// Page beyond range
			return;
		}

		// Page within range

		uint32_t page_index = p;

		double page_base_time = compression.pages[page_index].time_offset;
		const uint8_t *page_data = compression.pages[page_index].data.ptr();
#ifndef _MSC_VER
#warning Little endian assumed. No major big endian hardware exists any longer, but in case it does it will need to be supported
#endif
		const uint32_t *indices = (const uint32_t *)page_data;
		const uint16_t *time_keys = (const uint16_t *)&page_data[indices[p_compressed_track * 3 + 0]];
		uint32_t time_key_count = indices[p_compressed_track * 3 + 1];

		for (uint32_t i = 0; i < time_key_count; i++) {
			uint32_t f = time_keys[i * 2 + 0];
			double frame_time = f * frame_to_sec + page_base_time;
			if (frame_time >= p_time + p_delta) {
				return;
			} else if (frame_time >= p_time) {
				r_indices->push_back(key_index);
			}

			key_index++;

			const uint8_t *data_keys_base = (const uint8_t *)&page_data[indices[p_compressed_track * 3 + 2]];

			uint16_t time_key_data = time_keys[i * 2 + 1];
			uint32_t data_offset = (time_key_data & 0xFFF) * 4; // lower 12 bits
			uint32_t data_count = (time_key_data >> 12) + 1;

			const uint16_t *data_key = (const uint16_t *)(data_keys_base + data_offset);

			if (data_count > 1) {
				//decode forward
				uint32_t bit_width[COMPONENTS];
				for (uint32_t j = 0; j < COMPONENTS; j++) {
					bit_width[j] = (data_key[COMPONENTS] >> (j * 4)) & 0xF;
				}

				uint32_t frame_bit_width = (data_key[COMPONENTS] >> 12) + 1;

				AnimationCompressionBufferBitsRead buffer;

				buffer.src_data = (const uint8_t *)&data_key[COMPONENTS + 1];

				for (uint32_t j = 1; j < data_count; j++) {
					uint32_t frame_delta = buffer.read(frame_bit_width);
					f += frame_delta;

					frame_time = f * frame_to_sec + page_base_time;
					if (frame_time >= p_time + p_delta) {
						return;
					} else if (frame_time >= p_time) {
						r_indices->push_back(key_index);
					}

					for (uint32_t k = 0; k < COMPONENTS; k++) {
						if (bit_width[k] == 0) {
							continue; // do none
						}
						buffer.read(bit_width[k] + 1); // skip
					}

					key_index++;
				}
			}
		}
	}
}

int Animation::_get_compressed_key_count(uint32_t p_compressed_track) const {
	ERR_FAIL_COND_V(!compression.enabled, -1);
	ERR_FAIL_UNSIGNED_INDEX_V(p_compressed_track, compression.bounds.size(), -1);

	int key_count = 0;

	for (uint32_t i = 0; i < compression.pages.size(); i++) {
		const uint8_t *page_data = compression.pages[i].data.ptr();
#ifndef _MSC_VER
#warning Little endian assumed. No major big endian hardware exists any longer, but in case it does it will need to be supported
#endif
		const uint32_t *indices = (const uint32_t *)page_data;
		const uint16_t *time_keys = (const uint16_t *)&page_data[indices[p_compressed_track * 3 + 0]];
		uint32_t time_key_count = indices[p_compressed_track * 3 + 1];

		for (uint32_t j = 0; j < time_key_count; j++) {
			key_count += (time_keys[j * 2 + 1] >> 12) + 1;
		}
	}

	return key_count;
}

Quaternion Animation::_uncompress_quaternion(const Vector3i &p_value) const {
	Vector3 axis = Vector3::octahedron_decode(Vector2(float(p_value.x) / 65535.0, float(p_value.y) / 65535.0));
	float angle = (float(p_value.z) / 65535.0) * 2.0 * Math_PI;
	return Quaternion(axis, angle);
}
Vector3 Animation::_uncompress_pos_scale(uint32_t p_compressed_track, const Vector3i &p_value) const {
	Vector3 pos_norm(float(p_value.x) / 65535.0, float(p_value.y) / 65535.0, float(p_value.z) / 65535.0);
	return compression.bounds[p_compressed_track].position + pos_norm * compression.bounds[p_compressed_track].size;
}
float Animation::_uncompress_blend_shape(const Vector3i &p_value) const {
	float bsn = float(p_value.x) / 65535.0;
	return (bsn * 2.0 - 1.0) * float(Compression::BLEND_SHAPE_RANGE);
}

template <uint32_t COMPONENTS>
bool Animation::_fetch_compressed_by_index(uint32_t p_compressed_track, int p_index, Vector3i &r_value, double &r_time) const {
	ERR_FAIL_COND_V(!compression.enabled, false);
	ERR_FAIL_UNSIGNED_INDEX_V(p_compressed_track, compression.bounds.size(), false);

	for (uint32_t i = 0; i < compression.pages.size(); i++) {
		const uint8_t *page_data = compression.pages[i].data.ptr();
#ifndef _MSC_VER
#warning Little endian assumed. No major big endian hardware exists any longer, but in case it does it will need to be supported
#endif
		const uint32_t *indices = (const uint32_t *)page_data;
		const uint16_t *time_keys = (const uint16_t *)&page_data[indices[p_compressed_track * 3 + 0]];
		uint32_t time_key_count = indices[p_compressed_track * 3 + 1];
		const uint8_t *data_keys_base = (const uint8_t *)&page_data[indices[p_compressed_track * 3 + 2]];

		for (uint32_t j = 0; j < time_key_count; j++) {
			uint32_t subkeys = (time_keys[j * 2 + 1] >> 12) + 1;
			if ((uint32_t)p_index < subkeys) {
				uint16_t data_offset = (time_keys[j * 2 + 1] & 0xFFF) * 4;

				const uint16_t *data_key = (const uint16_t *)(data_keys_base + data_offset);

				uint16_t frame = time_keys[j * 2 + 0];
				uint16_t decode[COMPONENTS];

				for (uint32_t k = 0; k < COMPONENTS; k++) {
					decode[k] = data_key[k];
				}

				if (p_index > 0) {
					uint32_t bit_width[COMPONENTS];
					for (uint32_t k = 0; k < COMPONENTS; k++) {
						bit_width[k] = (data_key[COMPONENTS] >> (k * 4)) & 0xF;
					}
					uint32_t frame_bit_width = (data_key[COMPONENTS] >> 12) + 1;

					AnimationCompressionBufferBitsRead buffer;
					buffer.src_data = (const uint8_t *)&data_key[COMPONENTS + 1];

					for (int k = 0; k < p_index; k++) {
						uint32_t frame_delta = buffer.read(frame_bit_width);
						frame += frame_delta;
						for (uint32_t l = 0; l < COMPONENTS; l++) {
							if (bit_width[l] == 0) {
								continue; // do none
							}
							uint32_t valueu = buffer.read(bit_width[l] + 1);
							bool sign = valueu & (1 << bit_width[l]);
							int16_t value = valueu & ((1 << bit_width[l]) - 1);
							if (sign) {
								value = -value - 1;
							}

							decode[l] += value;
						}
					}
				}

				r_time = compression.pages[i].time_offset + double(frame) / double(compression.fps);
				for (uint32_t l = 0; l < COMPONENTS; l++) {
					r_value[l] = decode[l];
				}

				return true;

			} else {
				p_index -= subkeys;
			}
		}
	}

	return false;
}

Animation::Animation() {}

Animation::~Animation() {
	for (int i = 0; i < tracks.size(); i++) {
		memdelete(tracks[i]);
	}
}
