/**************************************************************************/
/*  animation_preview.cpp                                                 */
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

#include "animation_preview.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/os/thread.h"

AnimationPreview::AnimationPreview() {
	length = 0;
}

////

void AnimationPreviewGenerator::_update_emit(ObjectID p_id) {
	emit_signal(SNAME("preview_updated"), p_id);
}

void AnimationPreviewGenerator::_preview_thread(void *p_preview) {
	Thread::set_name("AnimationPreviewGenerator");
	Preview *preview = static_cast<Preview *>(p_preview);
	if (!preview->base_anim.is_valid()) {
		preview->generating.clear();
		return;
	}

	Ref<AnimationPreview> anim_preview;
	anim_preview.instantiate();

	// set length
	anim_preview->length = preview->base_anim->get_length();
	anim_preview->track_count = preview->base_anim->get_track_count();

	// collect key frames
	Vector<float> key_times_result;
	Vector<TrackKeyTime> track_key_times_result;

	for (int i = 0; i < preview->base_anim->get_track_count(); i++) {
		for (int j = 0; j < preview->base_anim->track_get_key_count(i); j++) {
			float time = preview->base_anim->track_get_key_time(i, j);
			if (!Math::is_finite(time)) {
				continue;
			}
			key_times_result.push_back(time);
			track_key_times_result.push_back({ i, time });
			if (j % 100 == 0) { // Progressives Update
				callable_mp(singleton, &AnimationPreviewGenerator::_update_emit).call_deferred(preview->id);
			}
		}
	}

	// sort keyframes
	key_times_result.sort();
	track_key_times_result.sort();

	// remove duplicated
	if (!key_times_result.is_empty()) {
		Vector<float> unique_times;
		unique_times.push_back(key_times_result[0]);
		for (int i = 1; i < key_times_result.size(); i++) {
			if (Math::is_equal_approx(key_times_result[i], unique_times[unique_times.size() - 1])) {
				continue;
			}
			unique_times.push_back(key_times_result[i]);
		}
		key_times_result = unique_times;
	}

	// update preview
	anim_preview->key_times = key_times_result;
	anim_preview->track_key_times = track_key_times_result;
	preview->preview = anim_preview;
	anim_preview->version++;

	// fire finished
	callable_mp(singleton, &AnimationPreviewGenerator::_update_emit).call_deferred(preview->id);

	preview->generating.clear();
}

Ref<AnimationPreview> AnimationPreviewGenerator::generate_preview(const Ref<Animation> &p_anim) {
	ERR_FAIL_COND_V(p_anim.is_null(), Ref<AnimationPreview>());

	if (previews.has(p_anim->get_instance_id())) {
		return previews[p_anim->get_instance_id()].preview;
	}

	previews[p_anim->get_instance_id()] = Preview();

	Preview *preview = &previews[p_anim->get_instance_id()];
	preview->base_anim = p_anim;
	preview->generating.set();
	preview->id = p_anim->get_instance_id();

	float len_s = preview->base_anim->get_length();

	preview->preview.instantiate();
	preview->preview->length = len_s;

	preview->thread = memnew(Thread);
	preview->thread->start(_preview_thread, preview);

	return preview->preview;
}

void AnimationPreview::create_key_region(Vector<Vector2> &points, const Rect2 &rect, const float p_pixels_sec, float start_ofs) {
	Vector<TrackKeyTime> key_times_result = get_key_times_with_tracks();
	int curr_track_count = get_track_count();
	float track_h = curr_track_count > 0 ? (rect.size.height - 2) / curr_track_count : rect.size.height;

	float len = get_length();
	float pixel_begin = rect.position.x;
	float from_x = rect.position.x;
	float to_x = from_x + rect.size.x;

	for (const TrackKeyTime &kt : key_times_result) {
		float ofs = kt.time - start_ofs;
		if (ofs < 0 || ofs > len) {
			continue;
		}
		int x = pixel_begin + ofs * p_pixels_sec;

		if (x < from_x || x >= to_x) {
			continue;
		}

		int y = rect.position.y + 2 + track_h * kt.track_index + track_h / 2;
		points.push_back(Point2(x, y));
		points.push_back(Point2(x + 1, y));
	}
}

void AnimationPreviewGenerator::clear_cache() {
	print_line("Clearing AnimationPreview cache, size was: ", previews.size());
	for (KeyValue<ObjectID, Preview> &E : previews) {
		if (E.value.thread) {
			E.value.thread->wait_to_finish();
			memdelete(E.value.thread);
			E.value.thread = nullptr;
		}
	}
	previews.clear();
}

void AnimationPreviewGenerator::invalidate_cache(const Ref<Animation> &p_anim) {
	if (p_anim.is_valid()) {
		ObjectID id = p_anim->get_instance_id();
		if (previews.has(id)) {
			print_line("Invalidating cache for animation: ", p_anim->get_name());
			if (previews[id].thread) {
				previews[id].thread->wait_to_finish();
				memdelete(previews[id].thread);
				previews[id].thread = nullptr;
			}
			previews.erase(id);
		}
	}
}

void AnimationPreviewGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate_preview", "animation"), &AnimationPreviewGenerator::generate_preview);

	ADD_SIGNAL(MethodInfo("preview_updated", PropertyInfo(Variant::INT, "obj_id")));
}

AnimationPreviewGenerator *AnimationPreviewGenerator::singleton = nullptr;

void AnimationPreviewGenerator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			List<ObjectID> to_erase;
			for (KeyValue<ObjectID, Preview> &E : previews) {
				if (!E.value.generating.is_set()) {
					if (E.value.thread) {
						E.value.thread->wait_to_finish();
						memdelete(E.value.thread);
						E.value.thread = nullptr;
					}
					if (!ObjectDB::get_instance(E.key)) {
						to_erase.push_back(E.key);
					}
				}
			}
			while (to_erase.front()) {
				previews.erase(to_erase.front()->get());
				to_erase.pop_front();
			}
		} break;
	}
}

AnimationPreviewGenerator::AnimationPreviewGenerator() {
	singleton = this;
	set_process(true);
}
