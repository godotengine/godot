/*************************************************************************/
/*  audio_stream_interactive.cpp                                         */
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

#include "audio_stream_interactive.h"

#include "core/math/math_funcs.h"
#include "core/string/print_string.h"

#include <iostream>

AudioStreamInteractive::AudioStreamInteractive() {
}

Ref<AudioStreamPlayback> AudioStreamInteractive::instantiate_playback() {
	Ref<AudioStreamPlaybackInteractive> playback_transitioner;
	playback_transitioner.instantiate();
	playback_transitioner->stream = Ref<AudioStreamInteractive>(this);
	return playback_transitioner;
}

String AudioStreamInteractive::get_stream_name() const {
	return "Transitioner";
}

void AudioStreamInteractive::set_clip_count(int p_count) {
	ERR_FAIL_COND(p_count < 0 || p_count > MAX_CLIPS);

	AudioServer::get_singleton()->lock();

	if (p_count < clip_count) {
		// Removing should stop players
		version++;
	}

#ifdef TOOLS_ENABLED
	stream_name_cache = "";
	if (p_count < clip_count) {
		for (int i = 0; i < clip_count; i++) {
			if (clips[i].auto_advance_next_clip >= p_count) {
				clips[i].auto_advance_next_clip = 0;
				clips[i].auto_advance = AUTO_ADVANCE_DISABLED;
			}
		}
		for (int i = 0; i < transition_count; i++) {
			if (transitions[i].filler_clip >= p_count) {
				transitions[i].filler_clip = 0;
				transitions[i].use_filler_clip = false;
			}
		}
		if (initial_clip >= p_count) {
			initial_clip = 0;
		}
	}
#endif
	clip_count = p_count;
	AudioServer::get_singleton()->unlock();

	notify_property_list_changed();
}

void AudioStreamInteractive::set_initial_clip(int p_clip) {
	ERR_FAIL_COND(p_clip < 0 || p_clip >= clip_count);
	initial_clip = p_clip;
}

int AudioStreamInteractive::get_initial_clip() const {
	return initial_clip;
}

int AudioStreamInteractive::get_clip_count() const {
	return clip_count;
}

void AudioStreamInteractive::set_clip_name(int p_clip, const StringName &p_name) {
	ERR_FAIL_COND(p_clip < 0 || p_clip >= MAX_CLIPS);
	clips[p_clip].name = p_name;
}

StringName AudioStreamInteractive::get_clip_name(int p_clip) const {
	ERR_FAIL_COND_V(p_clip < 0 || p_clip >= MAX_CLIPS, StringName());
	return clips[p_clip].name;
}

void AudioStreamInteractive::set_clip_stream(int p_clip, const Ref<AudioStream> &p_stream) {
	ERR_FAIL_COND(p_clip < 0 || p_clip >= MAX_CLIPS);
	AudioServer::get_singleton()->lock();
	if (clips[p_clip].stream.is_valid()) {
		version++;
	}
	clips[p_clip].stream = p_stream;
	AudioServer::get_singleton()->unlock();
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (clips[p_clip].name == StringName() && p_stream.is_valid()) {
			String name;
			if (!clips[p_clip].stream->get_name().is_empty()) {
				name = clips[p_clip].stream->get_name().replace(",", " ");
			} else if (clips[p_clip].stream->get_path().is_resource_file()) {
				name = clips[p_clip].stream->get_path().get_file().get_basename().replace(",", " ");
				name = name.capitalize();
			}

			if (name != "") {
				clips[p_clip].name = name;
			}
		}
	}
#endif

#ifdef TOOLS_ENABLED
	stream_name_cache = "";
	notify_property_list_changed(); // Hints change if stream changes
#endif
}

Ref<AudioStream> AudioStreamInteractive::get_clip_stream(int p_clip) const {
	ERR_FAIL_COND_V(p_clip < 0 || p_clip >= MAX_CLIPS, Ref<AudioStream>());
	return clips[p_clip].stream;
}

void AudioStreamInteractive::set_clip_auto_advance(int p_clip, AutoAdvanceMode p_mode) {
	ERR_FAIL_COND(p_clip < 0 || p_clip >= MAX_CLIPS);
	ERR_FAIL_INDEX(p_mode, 3);
	clips[p_clip].auto_advance = p_mode;
	notify_property_list_changed();
}

AudioStreamInteractive::AutoAdvanceMode AudioStreamInteractive::get_clip_auto_advance(int p_clip) const {
	ERR_FAIL_COND_V(p_clip < 0 || p_clip >= MAX_CLIPS, AUTO_ADVANCE_DISABLED);
	return clips[p_clip].auto_advance;
}

void AudioStreamInteractive::set_clip_auto_advance_next_clip(int p_clip, int p_index) {
	ERR_FAIL_COND(p_clip < 0 || p_clip >= MAX_CLIPS);
	clips[p_clip].auto_advance_next_clip = p_index;
}
int AudioStreamInteractive::get_clip_auto_advance_next_clip(int p_clip) const {
	ERR_FAIL_COND_V(p_clip < 0 || p_clip >= MAX_CLIPS, -1);
	return clips[p_clip].auto_advance_next_clip;
}

// TRANSITIONS

void AudioStreamInteractive::set_transition_count(int p_count) {
	ERR_FAIL_COND(p_count < 0 || p_count > MAX_TRANSITIONS);
	AudioServer::get_singleton()->lock();
	if (p_count < transition_count) {
		version++;
	}
	transition_count = p_count;
	AudioServer::get_singleton()->unlock();

	notify_property_list_changed();
}

int AudioStreamInteractive::get_transition_count() {
	return transition_count;
}

#ifdef TOOLS_ENABLED
void AudioStreamInteractive::_set_clip_preview(int p_clip) {
	clip_preview_set = p_clip;
	clip_preview_changed = true;
}

int AudioStreamInteractive::_get_clip_preview() const {
	return clip_preview_set;
}
#endif

void AudioStreamInteractive::set_transition_from_time(int p_transition, TransitionFromTime p_type) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	ERR_FAIL_INDEX(p_type, 4);
	transitions[p_transition].from_time = p_type;
	notify_property_list_changed();
}

AudioStreamInteractive::TransitionFromTime AudioStreamInteractive::get_transition_from_time(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, TRANSITION_FROM_TIME_END);
	return transitions[p_transition].from_time;
}

void AudioStreamInteractive::set_transition_to_time(int p_transition, TransitionToTime p_type) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	ERR_FAIL_INDEX(p_type, 3);
	transitions[p_transition].to_time = p_type;
}

AudioStreamInteractive::TransitionToTime AudioStreamInteractive::get_transition_to_time(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, TRANSITION_TO_TIME_START);
	return transitions[p_transition].to_time;
}

void AudioStreamInteractive::set_transition_fade_mode(int p_transition, FadeMode p_mode) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].fade_mode = p_mode;
	notify_property_list_changed();
}

AudioStreamInteractive::FadeMode AudioStreamInteractive::get_transition_fade_mode(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, FADE_DISABLED);
	return transitions[p_transition].fade_mode;
}

void AudioStreamInteractive::set_transition_fade_beats(int p_transition, float p_beats) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].fade_beats = p_beats;
}

float AudioStreamInteractive::get_transition_fade_beats(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, -1);
	return transitions[p_transition].fade_beats;
}

void AudioStreamInteractive::set_transition_source(int p_transition, TransitionClip p_source) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].source = p_source;
	notify_property_list_changed();
}

AudioStreamInteractive::TransitionClip AudioStreamInteractive::get_transition_source(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, TRANSITION_CLIP_ANY);
	return transitions[p_transition].source;
}

void AudioStreamInteractive::set_transition_source_clip(int p_transition, int p_index) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	ERR_FAIL_INDEX(p_index, clip_count);
	transitions[p_transition].source_clip = p_index;
}

int AudioStreamInteractive::get_transition_source_clip(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, -1);
	return transitions[p_transition].source_clip;
}

void AudioStreamInteractive::set_transition_source_mask(int p_transition, uint64_t p_mask) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].source_mask = p_mask;
}

uint64_t AudioStreamInteractive::get_transition_source_mask(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, 0);
	return transitions[p_transition].source_mask;
}

void AudioStreamInteractive::set_transition_dest(int p_transition, TransitionClip p_dest) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].dest = p_dest;
	notify_property_list_changed();
}

AudioStreamInteractive::TransitionClip AudioStreamInteractive::get_transition_dest(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, TRANSITION_CLIP_ANY);
	return transitions[p_transition].dest;
}

void AudioStreamInteractive::set_transition_dest_clip(int p_transition, int p_index) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	ERR_FAIL_INDEX(p_index, clip_count);
	transitions[p_transition].dest_clip = p_index;
}

int AudioStreamInteractive::get_transition_dest_clip(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, -1);
	return transitions[p_transition].dest_clip;
}

void AudioStreamInteractive::set_transition_dest_mask(int p_transition, uint64_t p_mask) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].dest_mask = p_mask;
}

uint64_t AudioStreamInteractive::get_transition_dest_mask(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, 0);
	return transitions[p_transition].dest_mask;
}

void AudioStreamInteractive::set_transition_use_filler_clip(int p_transition, bool p_enable) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].use_filler_clip = p_enable;
	notify_property_list_changed();
}

bool AudioStreamInteractive::is_transition_using_filler_clip(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, false);
	return transitions[p_transition].use_filler_clip;
}

void AudioStreamInteractive::set_transition_filler_clip(int p_transition, int p_clip_index) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].filler_clip = p_clip_index;
}

int AudioStreamInteractive::get_transition_filler_clip(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, -1);
	return transitions[p_transition].filler_clip;
}

void AudioStreamInteractive::set_transition_holds_previous(int p_transition, bool p_hold) {
	ERR_FAIL_COND(p_transition < 0 || p_transition >= MAX_TRANSITIONS);
	transitions[p_transition].hold_previous = p_hold;
}

bool AudioStreamInteractive::is_transition_holding_previous(int p_transition) const {
	ERR_FAIL_COND_V(p_transition < 0 || p_transition >= MAX_TRANSITIONS, false);
	return transitions[p_transition].hold_previous;
}

#ifdef TOOLS_ENABLED

PackedStringArray AudioStreamInteractive::_get_linked_undo_properties(const String &p_property, const Variant &p_new_value) const {
	PackedStringArray ret;

	if (p_property.begins_with("clip_") && p_property.ends_with("/stream")) {
		int clip = p_property.get_slicec('_', 1).to_int();
		if (clip < clip_count) {
			ret.push_back("clip_" + itos(clip) + "/name");
		}
	}

	if (p_property == "clip_count") {
		int new_clip_count = p_new_value;

		if (new_clip_count < clip_count) {
			for (int i = 0; i < clip_count; i++) {
				if (clips[i].auto_advance_next_clip >= new_clip_count) {
					ret.push_back("clip_" + itos(i) + "/auto_advance");
					ret.push_back("clip_" + itos(i) + "/next_clip");
				}
			}
			for (int i = 0; i < transition_count; i++) {
				if (transitions[i].filler_clip >= new_clip_count) {
					ret.push_back("transition_" + itos(i) + "/filler/enable");
					ret.push_back("transition_" + itos(i) + "/filler/clip");
				}
			}
			if (initial_clip >= new_clip_count) {
				ret.push_back("initial_clip");
			}
		}
	}
	return ret;
}

template <class T>
static void _test_and_swap(T &elem, uint32_t a, uint32_t b) {
	if ((uint32_t)elem == a) {
		elem = b;
	} else if (uint32_t(elem) == b) {
		elem = a;
	}
}
void AudioStreamInteractive::_inspector_array_swap_clip(uint32_t p_item_a, uint32_t p_item_b) {
	ERR_FAIL_INDEX(p_item_a, (uint32_t)clip_count);
	ERR_FAIL_INDEX(p_item_b, (uint32_t)clip_count);

	for (int i = 0; i < clip_count; i++) {
		_test_and_swap(clips[i].auto_advance_next_clip, p_item_a, p_item_b);
	}
	for (int i = 0; i < transition_count; i++) {
		uint64_t masks[2] = { transitions[i].source_mask, transitions[i].dest_mask };
		for (int j = 0; j < 2; j++) {
			bool flag_a = bool(masks[j] & (uint64_t(1) << p_item_a));
			bool flag_b = bool(masks[j] & (uint64_t(1) << p_item_b));
			masks[j] &= ~((uint64_t(1) << p_item_a) | (uint64_t(1) << p_item_b));
			SWAP(flag_a, flag_b);
			if (flag_a) {
				masks[j] |= (uint64_t(1) << p_item_b);
			}
			if (flag_b) {
				masks[j] |= (uint64_t(1) << p_item_a);
			}
		}
		transitions[i].source_mask = masks[0];
		transitions[i].dest_mask = masks[1];
		_test_and_swap(transitions[i].source_clip, p_item_a, p_item_b);
		_test_and_swap(transitions[i].dest_clip, p_item_a, p_item_b);

		_test_and_swap(transitions[i].filler_clip, p_item_a, p_item_b);
	}

	SWAP(clips[p_item_a], clips[p_item_b]);

	stream_name_cache = "";

	notify_property_list_changed();
}

String AudioStreamInteractive::_get_streams_hint() const {
	if (!stream_name_cache.is_empty()) {
		return stream_name_cache;
	}

	for (int i = 0; i < clip_count; i++) {
		if (i > 0) {
			stream_name_cache += ",";
		}
		String name = String(clips[i].name).replace(",", " ");

		if (name == "" && clips[i].stream.is_valid()) {
			if (!clips[i].stream->get_name().is_empty()) {
				name = clips[i].stream->get_name().replace(",", " ");
			} else if (clips[i].stream->get_path().is_resource_file()) {
				name = clips[i].stream->get_path().get_file().replace(",", " ");
			}
		}

		if (name == "") {
			name = "Clip " + itos(i);
		}

		stream_name_cache += name;
	}

	return stream_name_cache;
}

void AudioStreamInteractive::_set_transition_from_index(int p_transition, int p_index) {
	ERR_FAIL_INDEX(p_transition, transition_count);
	if (p_index >= TRANSITION_CLIP_SINGLE) {
		set_transition_source(p_transition, TRANSITION_CLIP_SINGLE);
		set_transition_source_clip(p_transition, p_index - TRANSITION_CLIP_SINGLE);
	} else {
		set_transition_source(p_transition, TransitionClip(p_index));
	}
}

int AudioStreamInteractive::_get_transition_from_index(int p_transition) const {
	ERR_FAIL_INDEX_V(p_transition, transition_count, -1);
	if (transitions[p_transition].source < TRANSITION_CLIP_SINGLE) {
		return transitions[p_transition].source;
	} else {
		return transitions[p_transition].source_clip + TRANSITION_CLIP_SINGLE;
	}
}

void AudioStreamInteractive::_set_transition_to_index(int p_transition, int p_index) {
	ERR_FAIL_INDEX(p_transition, transition_count);
	if (p_index >= TRANSITION_CLIP_SINGLE) {
		set_transition_dest(p_transition, TRANSITION_CLIP_SINGLE);
		set_transition_dest_clip(p_transition, p_index - TRANSITION_CLIP_SINGLE);
	} else {
		set_transition_dest(p_transition, TransitionClip(p_index));
	}
}

int AudioStreamInteractive::_get_transition_to_index(int p_transition) const {
	ERR_FAIL_INDEX_V(p_transition, transition_count, -1);
	if (transitions[p_transition].dest < TRANSITION_CLIP_SINGLE) {
		return transitions[p_transition].dest;
	} else {
		return transitions[p_transition].dest_clip + TRANSITION_CLIP_SINGLE;
	}
}

#endif
void AudioStreamInteractive::_validate_property(PropertyInfo &property) const {
	String prop = property.name;

#ifdef TOOLS_ENABLED
	if (prop == "switch_to") {
		property.hint_string = _get_streams_hint();
		return;
	}
#endif

	if (prop == "initial_clip") {
#ifdef TOOLS_ENABLED
		property.hint_string = _get_streams_hint();
#endif
	} else if (prop.begins_with("clip_") && prop != "clip_count") {
		int clip = prop.get_slicec('_', 1).to_int();
		if (clip >= clip_count) {
			property.usage = 0;
		} else if (prop == "clip_" + itos(clip) + "/next_clip") {
			if (clips[clip].auto_advance != AUTO_ADVANCE_ENABLED) {
				property.usage = 0;
			} else {
#ifdef TOOLS_ENABLED
				property.hint_string = _get_streams_hint();
#endif
			}
		}
	} else if (prop.begins_with("transition_") && prop != "transition_count") {
#ifdef TOOLS_ENABLED
		int transition = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (transition >= transition_count) {
			property.usage = 0;
		} else if (prop.ends_with("/from") || prop.ends_with("/to")) {
			property.hint_string = RTR("Any Clip,Multiple Clips");
			String hint = _get_streams_hint();
			if (hint != String()) {
				property.hint_string += "," + hint;
			}
		} else if (prop.ends_with("/from_mask")) {
			if (transitions[transition].source == TRANSITION_CLIP_MULTIPLE) {
				property.hint_string = _get_streams_hint();
			} else {
				property.usage = 0;
			}
		} else if (prop.ends_with("from_clip")) {
			if (transitions[transition].source == TRANSITION_CLIP_SINGLE) {
				property.hint_string = _get_streams_hint();
			} else {
				property.usage = 0;
			}
		} else if (prop.ends_with("to_mask")) {
			if (transitions[transition].dest == TRANSITION_CLIP_MULTIPLE) {
				property.hint_string = _get_streams_hint();
			} else {
				property.usage = 0;
			}
		} else if (prop.ends_with("to_clip")) {
			if (transitions[transition].dest == TRANSITION_CLIP_SINGLE) {
				property.hint_string = _get_streams_hint();
			} else {
				property.usage = 0;
			}
		} else if (prop.ends_with("/fade_beats") && transitions[transition].fade_mode == FADE_DISABLED) {
			property.usage = 0;
		} else if (prop.ends_with("filler/clip")) {
			if (!transitions[transition].use_filler_clip) {
				property.usage = 0;
			} else {
				property.hint_string = _get_streams_hint();
			}
		}
#endif
	}
}

void AudioStreamInteractive::_bind_methods() {
#ifdef TOOLS_ENABLED
	ClassDB::bind_method(D_METHOD("_get_linked_undo_properties", "for_property", "for_value"), &AudioStreamInteractive::_get_linked_undo_properties);
	ClassDB::bind_method(D_METHOD("_inspector_array_swap_clip", "a", "b"), &AudioStreamInteractive::_inspector_array_swap_clip);
#endif

	// CLIPS

	ClassDB::bind_method(D_METHOD("set_clip_count", "clip_count"), &AudioStreamInteractive::set_clip_count);
	ClassDB::bind_method(D_METHOD("get_clip_count"), &AudioStreamInteractive::get_clip_count);

	ClassDB::bind_method(D_METHOD("set_initial_clip", "clip_index"), &AudioStreamInteractive::set_initial_clip);
	ClassDB::bind_method(D_METHOD("get_initial_clip"), &AudioStreamInteractive::get_initial_clip);

	ClassDB::bind_method(D_METHOD("set_clip_name", "clip_index", "name"), &AudioStreamInteractive::set_clip_name);
	ClassDB::bind_method(D_METHOD("get_clip_name", "clip_index"), &AudioStreamInteractive::get_clip_name);

	ClassDB::bind_method(D_METHOD("set_clip_stream", "clip_index", "stream"), &AudioStreamInteractive::set_clip_stream);
	ClassDB::bind_method(D_METHOD("get_clip_stream", "clip_index"), &AudioStreamInteractive::get_clip_stream);

	ClassDB::bind_method(D_METHOD("set_clip_auto_advance", "clip_index", "mode"), &AudioStreamInteractive::set_clip_auto_advance);
	ClassDB::bind_method(D_METHOD("get_clip_auto_advance", "clip_index"), &AudioStreamInteractive::get_clip_auto_advance);

	ClassDB::bind_method(D_METHOD("set_clip_auto_advance_next_clip", "clip_index", "auto_advance_next_clip"), &AudioStreamInteractive::set_clip_auto_advance_next_clip);
	ClassDB::bind_method(D_METHOD("get_clip_auto_advance_next_clip", "clip_index"), &AudioStreamInteractive::get_clip_auto_advance_next_clip);

#ifdef TOOLS_ENABLED
	ClassDB::bind_method(D_METHOD("_set_clip_preview", "clip"), &AudioStreamInteractive::_set_clip_preview);
	ClassDB::bind_method(D_METHOD("_get_clip_preview"), &AudioStreamInteractive::_get_clip_preview);
#endif

	ADD_PROPERTY(PropertyInfo(Variant::INT, "initial_clip", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_DEFAULT), "set_initial_clip", "get_initial_clip");

#ifdef TOOLS_ENABLED
	ADD_PROPERTY(PropertyInfo(Variant::INT, "switch_to", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_clip_preview", "_get_clip_preview");
#endif

	ADD_PROPERTY(PropertyInfo(Variant::INT, "clip_count", PROPERTY_HINT_RANGE, "1," + itos(MAX_CLIPS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Clips,clip_,page_size=999,unfoldable,numbered,swap_method=_inspector_array_swap_clip,add_button_text=" + String(RTR("Add Clip"))), "set_clip_count", "get_clip_count");
	for (int i = 0; i < MAX_CLIPS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::STRING_NAME, "clip_" + itos(i) + "/name"), "set_clip_name", "get_clip_name", i);
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "clip_" + itos(i) + "/stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream", PROPERTY_USAGE_DEFAULT), "set_clip_stream", "get_clip_stream", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "clip_" + itos(i) + "/auto_advance", PROPERTY_HINT_ENUM, "Disabled,Enabled,ReturnToHold"), "set_clip_auto_advance", "get_clip_auto_advance", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "clip_" + itos(i) + "/next_clip", PROPERTY_HINT_ENUM, ""), "set_clip_auto_advance_next_clip", "get_clip_auto_advance_next_clip", i);
	}

	// TRANSITIONS

	ClassDB::bind_method(D_METHOD("set_transition_count", "transition_count"), &AudioStreamInteractive::set_transition_count);
	ClassDB::bind_method(D_METHOD("get_transition_count"), &AudioStreamInteractive::get_transition_count);

	ClassDB::bind_method(D_METHOD("set_transition_from_time", "transition_index", "from"), &AudioStreamInteractive::set_transition_from_time);
	ClassDB::bind_method(D_METHOD("get_transition_from_time", "transition_index"), &AudioStreamInteractive::get_transition_from_time);

	ClassDB::bind_method(D_METHOD("set_transition_to_time", "transition_index", "to"), &AudioStreamInteractive::set_transition_to_time);
	ClassDB::bind_method(D_METHOD("get_transition_to_time", "transition_index"), &AudioStreamInteractive::get_transition_to_time);

	ClassDB::bind_method(D_METHOD("set_transition_fade_mode", "transition_index", "mode"), &AudioStreamInteractive::set_transition_fade_mode);
	ClassDB::bind_method(D_METHOD("get_transition_fade_mode", "transition_index"), &AudioStreamInteractive::get_transition_fade_mode);

	ClassDB::bind_method(D_METHOD("set_transition_fade_beats", "transition_index", "beats"), &AudioStreamInteractive::set_transition_fade_beats);
	ClassDB::bind_method(D_METHOD("get_transition_fade_beats", "transition_index"), &AudioStreamInteractive::get_transition_fade_beats);

	ClassDB::bind_method(D_METHOD("set_transition_source", "transition_index", "source"), &AudioStreamInteractive::set_transition_source);
	ClassDB::bind_method(D_METHOD("get_transition_source", "transition_index"), &AudioStreamInteractive::get_transition_source);

	ClassDB::bind_method(D_METHOD("set_transition_source_clip", "transition_index", "clip"), &AudioStreamInteractive::set_transition_source_clip);
	ClassDB::bind_method(D_METHOD("get_transition_source_clip", "transition_index"), &AudioStreamInteractive::get_transition_source_clip);

	ClassDB::bind_method(D_METHOD("set_transition_source_mask", "transition_index", "mask"), &AudioStreamInteractive::set_transition_source_mask);
	ClassDB::bind_method(D_METHOD("get_transition_source_mask", "transition_index"), &AudioStreamInteractive::get_transition_source_mask);

	ClassDB::bind_method(D_METHOD("set_transition_dest", "transition_index", "dest"), &AudioStreamInteractive::set_transition_dest);
	ClassDB::bind_method(D_METHOD("get_transition_dest", "transition_index"), &AudioStreamInteractive::get_transition_dest);

	ClassDB::bind_method(D_METHOD("set_transition_dest_clip", "transition_index", "clip"), &AudioStreamInteractive::set_transition_dest_clip);
	ClassDB::bind_method(D_METHOD("get_transition_dest_clip", "transition_index"), &AudioStreamInteractive::get_transition_dest_clip);

	ClassDB::bind_method(D_METHOD("set_transition_dest_mask", "transition_index", "mask"), &AudioStreamInteractive::set_transition_dest_mask);
	ClassDB::bind_method(D_METHOD("get_transition_dest_mask", "transition_index"), &AudioStreamInteractive::get_transition_dest_mask);

	ClassDB::bind_method(D_METHOD("set_transition_use_filler_clip", "transition_index", "enable"), &AudioStreamInteractive::set_transition_use_filler_clip);
	ClassDB::bind_method(D_METHOD("is_transition_using_filler_clip", "transition_index"), &AudioStreamInteractive::is_transition_using_filler_clip);

	ClassDB::bind_method(D_METHOD("set_transition_filler_clip", "transition_index", "clip_index"), &AudioStreamInteractive::set_transition_filler_clip);
	ClassDB::bind_method(D_METHOD("get_transition_filler_clip", "transition_index"), &AudioStreamInteractive::get_transition_filler_clip);

	ClassDB::bind_method(D_METHOD("set_transition_holds_previous", "transition_index", "enable"), &AudioStreamInteractive::set_transition_holds_previous);
	ClassDB::bind_method(D_METHOD("is_transition_holding_previous", "transition_index"), &AudioStreamInteractive::is_transition_holding_previous);

#ifdef TOOLS_ENABLED

	ClassDB::bind_method(D_METHOD("_set_transition_from_index", "transition_index", "index"), &AudioStreamInteractive::_set_transition_from_index);
	ClassDB::bind_method(D_METHOD("_get_transition_from_index", "transition_index"), &AudioStreamInteractive::_get_transition_from_index);

	ClassDB::bind_method(D_METHOD("_set_transition_to_index", "transition_index", "index"), &AudioStreamInteractive::_set_transition_to_index);
	ClassDB::bind_method(D_METHOD("_get_transition_to_index", "transition_index"), &AudioStreamInteractive::_get_transition_to_index);
#endif

	ADD_PROPERTY(PropertyInfo(Variant::INT, "transition_count", PROPERTY_HINT_RANGE, "0," + itos(MAX_TRANSITIONS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Transitions,transition_,unfoldable,page_size=999,numbered,add_button_text=" + String(RTR("Add Transition"))), "set_transition_count", "get_transition_count");
	for (int i = 0; i < MAX_CLIPS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/from_source", PROPERTY_HINT_ENUM, "Any Clip,Multiple Clips,Single Clip", PROPERTY_USAGE_NO_EDITOR), "set_transition_source", "get_transition_source", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/from_clip", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_NO_EDITOR), "set_transition_source_clip", "get_transition_source_clip", i);
#ifdef TOOLS_ENABLED
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/from", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_EDITOR), "_set_transition_from_index", "_get_transition_from_index", i);
#endif
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/from_mask", PROPERTY_HINT_FLAGS, ""), "set_transition_source_mask", "get_transition_source_mask", i);

		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/to_source", PROPERTY_HINT_ENUM, "Any Clip,Multiple Clips,Single Clip", PROPERTY_USAGE_NO_EDITOR), "set_transition_dest", "get_transition_dest", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/to_clip", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_NO_EDITOR), "set_transition_dest_clip", "get_transition_dest_clip", i);
#ifdef TOOLS_ENABLED
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/to", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_EDITOR), "_set_transition_to_index", "_get_transition_to_index", i);
#endif

		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/to_mask", PROPERTY_HINT_FLAGS, ""), "set_transition_dest_mask", "get_transition_dest_mask", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/from_time", PROPERTY_HINT_ENUM, "Now,Next Beat,Next Bar,End", PROPERTY_USAGE_DEFAULT), "set_transition_from_time", "get_transition_from_time", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/to_time", PROPERTY_HINT_ENUM, "Same Position,Start,Previous Position", PROPERTY_USAGE_DEFAULT), "set_transition_to_time", "get_transition_to_time", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/fade_mode", PROPERTY_HINT_ENUM, "Disabled,In,Out,Cross"), "set_transition_fade_mode", "get_transition_fade_mode", i);
		ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "transition_" + itos(i) + "/fade_beats", PROPERTY_HINT_RANGE, "0.01,32,0.01"), "set_transition_fade_beats", "get_transition_fade_beats", i);
		ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "transition_" + itos(i) + "/hold_previous", PROPERTY_HINT_RANGE, "1,32,1"), "set_transition_holds_previous", "is_transition_holding_previous", i);
		ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "transition_" + itos(i) + "/filler/enable"), "set_transition_use_filler_clip", "is_transition_using_filler_clip", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/filler/clip", PROPERTY_HINT_ENUM, ""), "set_transition_filler_clip", "get_transition_filler_clip", i);
	}

	BIND_ENUM_CONSTANT(TRANSITION_FROM_TIME_NOW);
	BIND_ENUM_CONSTANT(TRANSITION_FROM_TIME_NEXT_BEAT);
	BIND_ENUM_CONSTANT(TRANSITION_FROM_TIME_NEXT_BAR);
	BIND_ENUM_CONSTANT(TRANSITION_FROM_TIME_END);

	BIND_ENUM_CONSTANT(TRANSITION_TO_TIME_SAME_POSITION);
	BIND_ENUM_CONSTANT(TRANSITION_TO_TIME_START);

	BIND_ENUM_CONSTANT(AUTO_ADVANCE_DISABLED);
	BIND_ENUM_CONSTANT(AUTO_ADVANCE_ENABLED);
	BIND_ENUM_CONSTANT(AUTO_ADVANCE_RETURN_TO_HOLD);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
AudioStreamPlaybackInteractive::AudioStreamPlaybackInteractive() {
}

AudioStreamPlaybackInteractive::~AudioStreamPlaybackInteractive() {
}

void AudioStreamPlaybackInteractive::stop() {
	if (!active) {
		return;
	}

	active = false;

	for (int i = 0; i < AudioStreamInteractive::MAX_CLIPS; i++) {
		if (states[i].playback.is_valid()) {
			states[i].playback->stop();
		}
		states[i].fade_speed = 0.0;
		states[i].fade_volume = 0.0;
		states[i].fade_wait = 0.0;
		states[i].reset_fade();
		states[i].active = false;
		states[i].auto_advance = -1;
		states[i].first_mix = true;
	}
}

void AudioStreamPlaybackInteractive::start(float p_from_pos) {
	if (active) {
		stop();
	}

	if (version != stream->version) {
		for (int i = 0; i < AudioStreamInteractive::MAX_CLIPS; i++) {
			Ref<AudioStream> src_stream;
			if (i < stream->clip_count) {
				src_stream = stream->clips[i].stream;
			}
			if (states[i].stream != src_stream) {
				states[i].stream.unref();
				states[i].playback.unref();

				states[i].stream = src_stream;
				states[i].playback = src_stream->instantiate_playback();
			}
		}

		version = stream->version;
	}

	int current = stream->initial_clip;
	if (current < 0 || current >= stream->clip_count) {
		return; //no playback possible
	}
	if (!states[current].playback.is_valid()) {
		return; //no playback possible
	}
	active = true;

#ifdef TOOLS_ENABLED
	stream->clip_preview_set = current;
	stream->clip_preview_changed = false;
#endif

	_queue(current, false);
}

void AudioStreamPlaybackInteractive::_queue(int p_to_clip_index, bool p_is_auto_advance) {
	ERR_FAIL_INDEX(p_to_clip_index, stream->clip_count);
	ERR_FAIL_COND(states[p_to_clip_index].playback.is_null());

	if (playback_current == -1) {
		// Nothing to do, start.
		int current = p_to_clip_index;
		State &state = states[current];
		state.active = true;
		state.fade_wait = 0;
		state.fade_volume = 1.0;
		state.fade_speed = 0;
		state.first_mix = true;

		state.playback->start(0);

		playback_current = current;

		if (stream->clips[current].auto_advance == AudioStreamInteractive::AUTO_ADVANCE_ENABLED && stream->clips[current].auto_advance_next_clip >= 0 && stream->clips[current].auto_advance_next_clip < stream->clip_count && stream->clips[current].auto_advance_next_clip != current) {
			//prepare auto advance
			state.auto_advance = stream->clips[current].auto_advance_next_clip;
		}
		return;
	}

	for (int i = 0; i < stream->clip_count; i++) {
		if (i == playback_current || i == p_to_clip_index) {
			continue;
		}
		if (states[i].active && states[i].fade_wait > 0) { // Waiting to kick in, terminate because change of plans.
			states[i].playback->stop();
			states[i].reset_fade();
			states[i].active = false;
		}
	}

	State &from_state = states[playback_current];
	State &to_state = states[p_to_clip_index];

	AudioStreamInteractive::Transition transition; // Use an empty transition by default

	for (int i = 0; i < stream->transition_count; i++) {
		bool source_match = false;
		switch (stream->transitions[i].source) {
			case AudioStreamInteractive::TRANSITION_CLIP_ANY: {
				source_match = true;
			} break;
			case AudioStreamInteractive::TRANSITION_CLIP_SINGLE: {
				source_match = stream->transitions[i].source_clip == playback_current;
			} break;
			case AudioStreamInteractive::TRANSITION_CLIP_MULTIPLE: {
				source_match = stream->transitions[i].source_mask & (1 << playback_current);
			} break;
		}
		if (!source_match) {
			continue;
		}
		bool dest_match = false;
		switch (stream->transitions[i].dest) {
			case AudioStreamInteractive::TRANSITION_CLIP_ANY: {
				dest_match = true;
			} break;
			case AudioStreamInteractive::TRANSITION_CLIP_SINGLE: {
				dest_match = stream->transitions[i].dest_clip == p_to_clip_index;
			} break;
			case AudioStreamInteractive::TRANSITION_CLIP_MULTIPLE: {
				dest_match = stream->transitions[i].dest_mask & (1 << p_to_clip_index);
			} break;
		}
		if (!dest_match) {
			continue;
		}
		transition = stream->transitions[i];
		break;
	}

	if (p_is_auto_advance) {
		transition.from_time = AudioStreamInteractive::TRANSITION_FROM_TIME_END;
		if (transition.to_time == AudioStreamInteractive::TRANSITION_TO_TIME_SAME_POSITION) {
			transition.to_time = AudioStreamInteractive::TRANSITION_TO_TIME_START;
		}
	}

	// Prepare the fadeout
	float current_pos = from_state.playback->get_playback_position();

	float src_fade_wait = 0;
	float dst_seek_to = 0;
	float fade_speed = 0;
	bool src_no_loop = false;

	if (from_state.stream->get_bpm()) {
		// Check if source speed has BPM, if so, transition syncs to BPM
		float beat_sec = 60 / float(from_state.stream->get_bpm());
		switch (transition.from_time) {
			case AudioStreamInteractive::TRANSITION_FROM_TIME_NOW: {
				src_fade_wait = 0;
			} break;
			case AudioStreamInteractive::TRANSITION_FROM_TIME_NEXT_BEAT: {
				float remainder = Math::fmod(current_pos, beat_sec);
				src_fade_wait = beat_sec - remainder;
			} break;
			case AudioStreamInteractive::TRANSITION_FROM_TIME_NEXT_BAR: {
				float bar_sec = beat_sec * from_state.stream->get_bar_beats();
				float remainder = Math::fmod(current_pos, bar_sec);
				src_fade_wait = bar_sec - remainder;
			} break;
			case AudioStreamInteractive::TRANSITION_FROM_TIME_END: {
				float end = from_state.stream->get_beat_count() > 0 ? float(from_state.stream->get_beat_count() * beat_sec) : from_state.stream->get_length();
				if (end == 0) {
					// Stream does not have a length.
					src_fade_wait = 0;
				} else {
					src_fade_wait = end - current_pos;
				}

				if (!from_state.stream->has_loop()) {
					src_no_loop = true;
				}

			} break;
		}
		// Fade speed also aligned to BPM
		fade_speed = 1.0 / (transition.fade_beats * beat_sec);
	} else {
		// Source has no BPM, so just simple transition.
		if (transition.from_time == AudioStreamInteractive::TRANSITION_FROM_TIME_END && from_state.stream->get_length() > 0) {
			float end = from_state.stream->get_length();
			src_fade_wait = end - current_pos;
			if (!from_state.stream->has_loop()) {
				src_no_loop = true;
			}
		} else {
			src_fade_wait = 0;
		}
		fade_speed = 1.0 / transition.fade_beats;
	}

	if (transition.to_time == AudioStreamInteractive::TRANSITION_TO_TIME_PREVIOUS_POSITION && to_state.stream->get_length() > 0.0) {
		dst_seek_to = to_state.previous_position;
	} else if (transition.to_time == AudioStreamInteractive::TRANSITION_TO_TIME_SAME_POSITION && transition.from_time != AudioStreamInteractive::TRANSITION_FROM_TIME_END && to_state.stream->get_length() > 0.0) {
		// Seeking to basically same position as when we start fading.
		dst_seek_to = current_pos + src_fade_wait;
		float end;
		if (to_state.stream->get_bpm() > 0 && to_state.stream->get_beat_count()) {
			float beat_sec = 60 / float(to_state.stream->get_bpm());
			end = to_state.stream->get_beat_count() * beat_sec;
		} else {
			end = to_state.stream->get_length();
		}

		if (dst_seek_to > end) {
			// Seeking too far away.
			dst_seek_to = 0; //past end, loop to beginning.
		}

	} else {
		// Seek to Start
		dst_seek_to = 0.0;
	}

	if (transition.fade_mode == AudioStreamInteractive::FADE_DISABLED || transition.fade_mode == AudioStreamInteractive::FADE_IN) {
		if (src_no_loop) {
			// If there is no fade in the source stream, then let it continue until it ends.
			from_state.fade_wait = 0;
			from_state.fade_speed = 0;
		} else {
			// Otherwise force a very quick fade to avoid clicks
			from_state.fade_wait = src_fade_wait;
			from_state.fade_speed = 1.0 / -0.001;
		}
	} else {
		// Regular fade.
		from_state.fade_wait = src_fade_wait;
		from_state.fade_speed = -fade_speed;
	}
	// keep volume, since it may have been fading in from something else.

	to_state.playback->start(dst_seek_to);
	to_state.active = true;
	to_state.fade_volume = 0.0;
	to_state.first_mix = true;

	int auto_advance_to = -1;

	if (stream->clips[p_to_clip_index].auto_advance == AudioStreamInteractive::AUTO_ADVANCE_ENABLED) {
		int next_clip = stream->clips[p_to_clip_index].auto_advance_next_clip;
		if (next_clip >= 0 && next_clip < (int)stream->clip_count && states[next_clip].playback.is_valid() && next_clip != p_to_clip_index && next_clip != playback_current && (!transition.use_filler_clip || next_clip != transition.filler_clip)) {
			auto_advance_to = next_clip;
		}
	}

	if (return_memory != -1 && stream->clips[p_to_clip_index].auto_advance == AudioStreamInteractive::AUTO_ADVANCE_RETURN_TO_HOLD) {
		auto_advance_to = return_memory;
		return_memory = -1;
	}

	if (transition.hold_previous) {
		return_memory = playback_current;
	}

	if (transition.use_filler_clip && transition.filler_clip >= 0 && transition.filler_clip < (int)stream->clip_count && states[transition.filler_clip].playback.is_valid() && playback_current != transition.filler_clip && p_to_clip_index != transition.filler_clip) {
		State &filler_state = states[transition.filler_clip];

		filler_state.playback->start(0);
		filler_state.active = true;

		// Filler state does not fade (bake fade in the audio clip if you want fading.
		filler_state.fade_volume = 1.0;
		filler_state.fade_speed = 0.0;

		filler_state.fade_wait = src_fade_wait;
		filler_state.first_mix = true;

		float filler_end;
		if (filler_state.stream->get_bpm() > 0 && filler_state.stream->get_beat_count() > 0) {
			float filler_beat_sec = 60 / float(filler_state.stream->get_bpm());
			filler_end = filler_beat_sec * filler_state.stream->get_beat_count();
		} else {
			filler_end = filler_state.stream->get_length();
		}

		if (!filler_state.stream->has_loop()) {
			src_no_loop = true;
		}

		if (transition.fade_mode == AudioStreamInteractive::FADE_DISABLED || transition.fade_mode == AudioStreamInteractive::FADE_OUT) {
			// No fading, immediately start at full volume.
			to_state.fade_volume = 0.0;
			to_state.fade_speed = 1.0; //start at full volume, as filler is meant as a transition.
		} else {
			// Fade enable, prepare fade.
			to_state.fade_volume = 0.0;
			to_state.fade_speed = fade_speed;
		}

		to_state.fade_wait = src_fade_wait + filler_end;

	} else {
		to_state.fade_wait = src_fade_wait;

		if (transition.fade_mode == AudioStreamInteractive::FADE_DISABLED || transition.fade_mode == AudioStreamInteractive::FADE_OUT) {
			to_state.fade_volume = 1.0;
			to_state.fade_speed = 0.0;
		} else {
			to_state.fade_volume = 0.0;
			to_state.fade_speed = fade_speed;
		}

		to_state.auto_advance = auto_advance_to;
	}
}

void AudioStreamPlaybackInteractive::seek(float p_time) {
	// Seek not supported
}

int AudioStreamPlaybackInteractive::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (active && version != stream->version) {
		stop();
	}

#ifdef TOOLS_ENABLED
	if (stream->clip_preview_changed) {
		_queue(stream->clip_preview_set, false);
		stream->clip_preview_changed = false;
		switch_request = -1;
	}
#endif

	if (switch_request != -1) {
		if (!active) {
			start();
		}
		_queue(switch_request, false);
		switch_request = -1;
	}

	if (!active) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0, 0.0);
		}
		return p_frames;
	}

	int todo = p_frames;

	while (todo) {
		int to_mix = MIN(todo, BUFFER_SIZE);
		_mix_internal(to_mix);
		for (int i = 0; i < to_mix; i++) {
			p_buffer[i] = mix_buffer[i];
		}
		p_buffer += to_mix;
		todo -= to_mix;
	}

	return p_frames;
}

void AudioStreamPlaybackInteractive::_mix_internal(int p_frames) {
	for (int i = 0; i < p_frames; i++) {
		mix_buffer[i] = AudioFrame(0, 0);
	}

	for (int i = 0; i < stream->clip_count; i++) {
		if (!states[i].active) {
			continue;
		}

		_mix_internal_state(i, p_frames);
	}
}

void AudioStreamPlaybackInteractive::_mix_internal_state(int p_state_idx, int p_frames) {
	State &state = states[p_state_idx];
	double mix_rate = double(AudioServer::get_singleton()->get_mix_rate());
	double frame_inc = 1.0 / mix_rate;

	int from_frame = 0;
	int queue_next = -1;

	if (state.first_mix) {
		// Did not start mixing yet, wait.
		double mix_time = p_frames * frame_inc;
		if (state.fade_wait < mix_time) {
			// time to start!
			from_frame = state.fade_wait * mix_rate;
			state.fade_wait = 0;
			queue_next = state.auto_advance;
			playback_current = p_state_idx;
			state.first_mix = false;
#ifdef TOOLS_ENABLED
			stream->clip_preview_set = playback_current;
#endif

		} else {
			// This is for fade in of new stream.
			state.fade_wait -= mix_time;
			return; // Nothing to do
		}
	}

	state.previous_position = state.playback->get_playback_position();
	state.playback->mix(temp_buffer + from_frame, 1.0, p_frames - from_frame);

	double frame_fade_inc = state.fade_speed * frame_inc;

	for (int i = from_frame; i < p_frames; i++) {
		if (state.fade_wait) {
			// This is for fade out of existing stream;
			state.fade_wait -= frame_inc;
			if (state.fade_wait < 0.0) {
				state.fade_wait = 0.0;
			}
		} else if (frame_fade_inc > 0) {
			state.fade_volume += frame_fade_inc;
			if (state.fade_volume >= 1.0) {
				state.fade_speed = 0.0;
				frame_fade_inc = 0.0;
				state.fade_volume = 1.0;
			}
		} else if (frame_fade_inc < 0.0) {
			state.fade_volume += frame_fade_inc;
			if (state.fade_volume <= 0.0) {
				state.fade_speed = 0.0;
				frame_fade_inc = 0.0;
				state.fade_volume = 0.0;
				state.playback->stop(); // Stop playback and break, no point to continue mixing
				break;
			}
		}

		mix_buffer[i] += temp_buffer[i] * state.fade_volume;
		state.previous_position += frame_inc;
	}

	if (!state.playback->is_playing()) {
		// It finished because it either reached end or faded out, so deactivate and continue.
		state.active = false;
	}
	if (queue_next != -1) {
		_queue(queue_next, true);
	}
}

void AudioStreamPlaybackInteractive::tag_used_streams() {
	for (int i = 0; i < stream->clip_count; i++) {
		if (states[i].active && !states[i].first_mix && states[i].playback->is_playing()) {
			states[i].stream->tag_used(states[i].playback->get_playback_position());
		}
	}
	stream->tag_used(0);
}

void AudioStreamPlaybackInteractive::switch_to_clip(int p_index) {
	switch_request = p_index;
}

int AudioStreamPlaybackInteractive::get_loop_count() const {
	return 0; // Looping not supported
}

float AudioStreamPlaybackInteractive::get_playback_position() const {
	return 0.0;
}

bool AudioStreamPlaybackInteractive::is_playing() const {
	return active;
}

void AudioStreamPlaybackInteractive::_bind_methods() {
	ClassDB::bind_method(D_METHOD("switch_to_clip", "clip_index"), &AudioStreamPlaybackInteractive::switch_to_clip);
}
