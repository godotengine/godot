/**************************************************************************/
/*  animation_track_filter.cpp                                            */
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

#include "animation_track_filter.h"

Dictionary AnimationTrackFilter::get_tracks_bind() const {
	tracks.sort();
	Dictionary ret;
	for (const TrackFilterInfo &E : tracks) {
		ret[E.track] = E.amount;
	}
	return ret;
}

void AnimationTrackFilter::set_tracks_bind(const Dictionary &p_tracks) {
	tracks.clear();
	Array track_paths = p_tracks.keys();
	Array track_amounts = p_tracks.values();
	for (int i = 0; i < track_paths.size(); ++i) {
		//	Validation.
		ERR_CONTINUE(!Variant::can_convert(track_paths[i].get_type(), Variant::NODE_PATH));
		ERR_CONTINUE(!Variant::can_convert(track_amounts[i].get_type(), Variant::FLOAT));

		tracks.push_back({ track_paths[i], CLAMP(track_amounts[i].operator float(), 0.0f, 1.0f) });
	}
	emit_changed();
}

void AnimationTrackFilter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tracks", "tracks"), &AnimationTrackFilter::set_tracks_bind);
	ClassDB::bind_method(D_METHOD("get_tracks"), &AnimationTrackFilter::get_tracks_bind);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tracks"), "set_tracks", "get_tracks");

	ClassDB::bind_method(D_METHOD("has_trck", "track_path"), &AnimationTrackFilter::has_track);
	ClassDB::bind_method(D_METHOD("remove_track", "track_path"), &AnimationTrackFilter::remove_track);
	ClassDB::bind_method(D_METHOD("get_track_amount", "track_path"), &AnimationTrackFilter::get_track_amount);
	ClassDB::bind_method(D_METHOD("set_track", "track_path", "amount"), &AnimationTrackFilter::set_track, DEFVAL(1.0f));
	ClassDB::bind_method(D_METHOD("clear_tracks"), &AnimationTrackFilter::clear_tracks);
}

bool AnimationTrackFilter::has_track(const NodePath &p_track_path) const {
	for (const TrackFilterInfo &E : tracks) {
		if (E.track == p_track_path) {
			return true;
		}
	}
	return false;
}

void AnimationTrackFilter::remove_track(const NodePath &p_track_path) {
	for (size_t i = 0; i < tracks.size(); ++i) {
		if (tracks[i].track == p_track_path) {
			SWAP(tracks[i], tracks[tracks.size() - 1]);
			tracks.resize(tracks.size() - 1);
			emit_changed();
			return;
		}
	}
}

float AnimationTrackFilter::get_track_amount(const NodePath &p_track_path) const {
	for (const TrackFilterInfo &E : tracks) {
		if (E.track == p_track_path) {
			return E.amount;
		}
	}
	return 0.0f;
}

void AnimationTrackFilter::set_track(const NodePath &p_track_path, float p_amount) {
	p_amount = CLAMP(p_amount, 0.0f, 1.0f);

	if (Math::is_zero_approx(p_amount)) {
		remove_track(p_track_path);
		return;
	}

	for (TrackFilterInfo &E : tracks) {
		if (E.track == p_track_path) {
			E.amount = p_amount;
			emit_changed();
			return;
		}
	}
	tracks.push_back({ p_track_path, p_amount });
	emit_changed();
}

void AnimationTrackFilter::clear_tracks() {
	tracks.clear();
}
