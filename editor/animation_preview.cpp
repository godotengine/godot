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

	// Setze Länge
	anim_preview->length = preview->base_anim->get_length();
	if (anim_preview->length <= 0) {
		anim_preview->length = 0.0001;
	}

	// Sammle Keyframe-Zeiten
	Vector<float> key_times;
	Vector<TrackKeyTime> track_key_times;

	for (int i = 0; i < preview->base_anim->get_track_count(); i++) {
		for (int j = 0; j < preview->base_anim->track_get_key_count(i); j++) {
			float time = preview->base_anim->track_get_key_time(i, j);
			if (!Math::is_finite(time)) {
				continue;
			}
			key_times.push_back(time);
			track_key_times.push_back({ i, time });
			if (j % 100 == 0) { // Progressives Update
				callable_mp(singleton, &AnimationPreviewGenerator::_update_emit).call_deferred(preview->id);
			}
		}
	}

	// Sortiere Keyframe-Zeiten
	key_times.sort();
	track_key_times.sort();

	// Entferne Duplikate (optional)
	if (!key_times.is_empty()) {
		Vector<float> unique_times;
		unique_times.push_back(key_times[0]);
		for (int i = 1; i < key_times.size(); i++) {
			if (Math::is_equal_approx(key_times[i], unique_times[unique_times.size() - 1])) {
				continue;
			}
			unique_times.push_back(key_times[i]);
		}
		key_times = unique_times;
	}

	// Aktualisiere Vorschau
	anim_preview->key_times = key_times;
	anim_preview->track_key_times = track_key_times;
	preview->preview = anim_preview;
	anim_preview->version++;

	// Signalisiere Abschluss
	callable_mp(singleton, &AnimationPreviewGenerator::_update_emit).call_deferred(preview->id);

	preview->generating.clear();
}

Ref<AnimationPreview> AnimationPreviewGenerator::generate_preview(const Ref<Animation> &p_anim) {
	ERR_FAIL_COND_V(p_anim.is_null(), Ref<AnimationPreview>());


	if (previews.has(p_anim->get_instance_id())) {
		return previews[p_anim->get_instance_id()].preview;
	}

	//no preview exists

	previews[p_anim->get_instance_id()] = Preview();

	Preview *preview = &previews[p_anim->get_instance_id()];
	preview->base_anim = p_anim;
	preview->generating.set();
	preview->id = p_anim->get_instance_id();

	float len_s = preview->base_anim->get_length();
	if (len_s == 0) {
		len_s = 0.0001;
	}

	preview->preview.instantiate();
	preview->preview->length = len_s;
	
	preview->thread = memnew(Thread);
	preview->thread->start(_preview_thread, preview);

	return preview->preview;
}

void AnimationPreviewGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate_preview", "animation"), &AnimationPreviewGenerator::generate_preview);

	ADD_SIGNAL(MethodInfo("preview_updated", PropertyInfo(Variant::INT, "obj_id")));
}

/*
void AnimationPreview::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_length"), &AnimationPreview::get_length);
	ClassDB::bind_method(D_METHOD("get_key_times"), &AnimationPreview::get_key_times);
	ClassDB::bind_method(D_METHOD("get_key_times_with_tracks"), &AnimationPreview::get_key_times_with_tracks);
	ClassDB::bind_method(D_METHOD("get_version"), &AnimationPreview::get_version);
}

void AnimationPreviewGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate_preview", "animation"), &AnimationPreviewGenerator::generate_preview);
	ClassDB::bind_method(D_METHOD("clear_cache"), &AnimationPreviewGenerator::clear_cache);
	ClassDB::bind_method(D_METHOD("invalidate_cache", "animation"), &AnimationPreviewGenerator::invalidate_cache);
	ADD_SIGNAL(MethodInfo("preview_updated", PropertyInfo(Variant::INT, "obj_id")));
}
*/

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
