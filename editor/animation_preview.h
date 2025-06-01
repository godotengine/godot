/**************************************************************************/
/*  animation_preview.h                                                   */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/os/thread.h"
#include "core/templates/safe_refcount.h"
#include "scene/main/node.h"
#include "scene/resources/animation.h"

struct TrackKeyTime {
	int track_index;
	float time;
	bool operator<(const TrackKeyTime &p_other) const {
		return time < p_other.time;
	}
};

class AnimationPreview : public RefCounted {
	GDCLASS(AnimationPreview, RefCounted);
	friend class Preview;

private:
	Vector<float> key_times;
	Vector<TrackKeyTime> track_key_times;
	float length;
	int track_count;

	friend class AnimationPreviewGenerator;
	uint64_t version = 1;

public:
	uint64_t get_version() const { return version; }
	float get_length() const { return length; }
	Vector<float> get_key_times() const { return key_times; }
	Vector<TrackKeyTime> get_key_times_with_tracks() const { return track_key_times; }

	int get_track_count() const { return track_count; }

	void create_key_region(Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs);

	AnimationPreview();
};

class AnimationPreviewGenerator : public Node {
	GDCLASS(AnimationPreviewGenerator, Node);

	static AnimationPreviewGenerator *singleton;

public:
	struct Preview {
		Ref<Animation> base_anim;
		Ref<AnimationPreview> preview;
		SafeFlag generating;
		ObjectID id;
		Thread *thread = nullptr;

		void operator=(const Preview &p_rhs) {
			base_anim = p_rhs.base_anim;
			preview = p_rhs.preview;
			generating.set_to(p_rhs.generating.is_set());
			id = p_rhs.id;
			thread = p_rhs.thread;
		}
		Preview(const Preview &p_rhs) {
			base_anim = p_rhs.base_anim;
			preview = p_rhs.preview;
			generating.set_to(p_rhs.generating.is_set());
			id = p_rhs.id;
			thread = p_rhs.thread;
		}
		Preview() {}
	};

private:
	HashMap<ObjectID, Preview> previews;

	static void _preview_thread(void *p_preview);

	void _update_emit(ObjectID p_id);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AnimationPreviewGenerator *get_singleton() { return singleton; }

	Ref<AnimationPreview> generate_preview(const Ref<Animation> &p_animation);

	void clear_cache();
	void invalidate_cache(const Ref<Animation> &p_animation);

	AnimationPreviewGenerator();
};
