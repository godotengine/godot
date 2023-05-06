/**************************************************************************/
/*  animation_track_filter.h                                              */
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

#ifndef ANIMATION_TRACK_FILTER_H
#define ANIMATION_TRACK_FILTER_H

#include "core/io/resource.h"
#include "core/variant/typed_array.h"

class AnimationTrackFilter : public Resource {
	GDCLASS(AnimationTrackFilter, Resource);

public:
	struct TrackFilterInfo {
		NodePath track;
		float amount;
		_FORCE_INLINE_ bool operator<(const TrackFilterInfo &p_other) const { return String(track) < String(p_other.track); }
	};

private:
	mutable LocalVector<TrackFilterInfo> tracks;

#ifdef TOOLS_ENABLED
	friend class AnimationNodeBlendTreeEditor;
#endif

protected:
	static void _bind_methods();

public:
	const LocalVector<TrackFilterInfo> &get_tracks() const { return tracks; }
	void set_tracks(const LocalVector<TrackFilterInfo> &p_tracks) {
		tracks = p_tracks;
		emit_changed();
	}

	// Todo: Use Typed Dictionary or TypedArray<Struct> instead of Dictionary.
	Dictionary get_tracks_bind() const;
	void set_tracks_bind(const Dictionary &p_tracks);

	float get_track_amount(const NodePath &p_track_path) const;
	bool has_track(const NodePath &p_track_path) const;
	void remove_track(const NodePath &p_track_path);
	void set_track(const NodePath &p_track_path, float p_amount);

	void clear_tracks();
};

#endif // ANIMATION_TRACK_FILTER_H
