/**************************************************************************/
/*  audio_stream_player_2d.cpp                                            */
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

#include "audio_stream_player_2d.h"

#include "core/config/project_settings.h"
#include "scene/2d/area_2d.h"
#include "scene/2d/audio_listener_2d.h"
#include "scene/main/window.h"
#include "scene/resources/world_2d.h"

void AudioStreamPlayer2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			AudioServer::get_singleton()->add_listener_changed_callback(_listener_changed_cb, this);
			if (autoplay && !Engine::get_singleton()->is_editor_hint()) {
				play();
			}
			set_stream_paused(!can_process());
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_stream_paused(true);
			AudioServer::get_singleton()->remove_listener_changed_callback(_listener_changed_cb, this);
		} break;

		case NOTIFICATION_PREDELETE: {
			stop();
		} break;

		case NOTIFICATION_PAUSED: {
			if (!can_process()) {
				// Node can't process so we start fading out to silence.
				set_stream_paused(true);
			}
		} break;

		case NOTIFICATION_UNPAUSED: {
			set_stream_paused(false);
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// Update anything related to position first, if possible of course.
			if (setplay.get() > 0 || (active.is_set() && last_mix_count != AudioServer::get_singleton()->get_mix_count()) || force_update_panning) {
				force_update_panning = false;
				_update_panning();
			}

			if (setplayback.is_valid() && setplay.get() >= 0) {
				active.set();
				AudioServer::get_singleton()->start_playback_stream(setplayback, _get_actual_bus(), volume_vector, setplay.get(), pitch_scale);
				setplayback.unref();
				setplay.set(-1);
			}

			if (!stream_playbacks.is_empty() && active.is_set()) {
				// Stop playing if no longer active.
				Vector<Ref<AudioStreamPlayback>> playbacks_to_remove;
				for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
					if (playback.is_valid() && !AudioServer::get_singleton()->is_playback_active(playback) && !AudioServer::get_singleton()->is_playback_paused(playback)) {
						playbacks_to_remove.push_back(playback);
					}
				}
				// Now go through and remove playbacks that have finished. Removing elements from a Vector in a range based for is asking for trouble.
				for (Ref<AudioStreamPlayback> &playback : playbacks_to_remove) {
					stream_playbacks.erase(playback);
				}
				if (!playbacks_to_remove.is_empty() && stream_playbacks.is_empty()) {
					// This node is no longer actively playing audio.
					active.clear();
					set_physics_process_internal(false);
				}
				if (!playbacks_to_remove.is_empty()) {
					emit_signal(SNAME("finished"));
				}
			}

			while (stream_playbacks.size() > max_polyphony) {
				AudioServer::get_singleton()->stop_playback_stream(stream_playbacks[0]);
				stream_playbacks.remove_at(0);
			}
		} break;
	}
}

// Interacts with PhysicsServer2D, so can only be called during _physics_process
StringName AudioStreamPlayer2D::_get_actual_bus() {
	Vector2 global_pos = get_global_position();

	//check if any area is diverting sound into a bus
	Ref<World2D> world_2d = get_world_2d();
	ERR_FAIL_COND_V(world_2d.is_null(), SceneStringNames::get_singleton()->Master);

	PhysicsDirectSpaceState2D *space_state = PhysicsServer2D::get_singleton()->space_get_direct_state(world_2d->get_space());
	ERR_FAIL_NULL_V(space_state, SceneStringNames::get_singleton()->Master);
	PhysicsDirectSpaceState2D::ShapeResult sr[MAX_INTERSECT_AREAS];

	PhysicsDirectSpaceState2D::PointParameters point_params;
	point_params.position = global_pos;
	point_params.collision_mask = area_mask;
	point_params.collide_with_bodies = false;
	point_params.collide_with_areas = true;

	int areas = space_state->intersect_point(point_params, sr, MAX_INTERSECT_AREAS);

	for (int i = 0; i < areas; i++) {
		Area2D *area2d = Object::cast_to<Area2D>(sr[i].collider);
		if (!area2d) {
			continue;
		}

		if (!area2d->is_overriding_audio_bus()) {
			continue;
		}

		return area2d->get_audio_bus_name();
	}
	return default_bus;
}

// Interacts with PhysicsServer2D, so can only be called during _physics_process
void AudioStreamPlayer2D::_update_panning() {
	if (!active.is_set() || stream.is_null()) {
		return;
	}

	Ref<World2D> world_2d = get_world_2d();
	ERR_FAIL_COND(world_2d.is_null());

	Vector2 global_pos = get_global_position();

	HashSet<Viewport *> viewports = world_2d->get_viewports();

	volume_vector.resize(4);
	volume_vector.write[0] = AudioFrame(0, 0);
	volume_vector.write[1] = AudioFrame(0, 0);
	volume_vector.write[2] = AudioFrame(0, 0);
	volume_vector.write[3] = AudioFrame(0, 0);

	for (Viewport *vp : viewports) {
		if (!vp->is_audio_listener_2d()) {
			continue;
		}
		//compute matrix to convert to screen
		Vector2 screen_size = vp->get_visible_rect().size;
		Vector2 listener_in_global;
		Vector2 relative_to_listener;

		//screen in global is used for attenuation
		AudioListener2D *listener = vp->get_audio_listener_2d();
		Transform2D full_canvas_transform = vp->get_global_canvas_transform() * vp->get_canvas_transform();
		if (listener) {
			listener_in_global = listener->get_global_position();
			relative_to_listener = (global_pos - listener_in_global).rotated(-listener->get_global_rotation());
			relative_to_listener *= full_canvas_transform.get_scale(); // Default listener scales with canvas size, do the same here.
		} else {
			listener_in_global = full_canvas_transform.affine_inverse().xform(screen_size * 0.5);
			relative_to_listener = full_canvas_transform.xform(global_pos) - screen_size * 0.5;
		}

		float dist = global_pos.distance_to(listener_in_global); // Distance to listener, or screen if none.

		if (dist > max_distance) {
			continue; // Can't hear this sound in this viewport.
		}

		float multiplier = Math::pow(1.0f - dist / max_distance, attenuation);
		multiplier *= Math::db_to_linear(volume_db); // Also apply player volume!

		float pan = relative_to_listener.x / screen_size.x;
		// Don't let the panning effect extend (too far) beyond the screen.
		pan = CLAMP(pan, -1, 1);

		// Bake in a constant factor here to allow the project setting defaults for 2d and 3d to be normalized to 1.0.
		pan *= panning_strength * cached_global_panning_strength * 0.5f;

		pan = CLAMP(pan + 0.5, 0.0, 1.0);

		float l = 1.0 - pan;
		float r = pan;

		const AudioFrame &prev_sample = volume_vector[0];
		AudioFrame new_sample = AudioFrame(l, r) * multiplier;
		volume_vector.write[0] = AudioFrame(MAX(prev_sample[0], new_sample[0]), MAX(prev_sample[1], new_sample[1]));
	}

	for (const Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_bus_exclusive(playback, _get_actual_bus(), volume_vector);
	}

	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_pitch_scale(playback, pitch_scale);
	}

	last_mix_count = AudioServer::get_singleton()->get_mix_count();
}

void AudioStreamPlayer2D::set_stream(Ref<AudioStream> p_stream) {
	stop();
	stream = p_stream;
}

Ref<AudioStream> AudioStreamPlayer2D::get_stream() const {
	return stream;
}

void AudioStreamPlayer2D::set_volume_db(float p_volume) {
	volume_db = p_volume;
}

float AudioStreamPlayer2D::get_volume_db() const {
	return volume_db;
}

void AudioStreamPlayer2D::set_pitch_scale(float p_pitch_scale) {
	ERR_FAIL_COND(!(p_pitch_scale > 0.0));
	pitch_scale = p_pitch_scale;
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_pitch_scale(playback, p_pitch_scale);
	}
}

float AudioStreamPlayer2D::get_pitch_scale() const {
	return pitch_scale;
}

void AudioStreamPlayer2D::play(float p_from_pos) {
	if (stream.is_null()) {
		return;
	}
	ERR_FAIL_COND_MSG(!is_inside_tree(), "Playback can only happen when a node is inside the scene tree");
	if (stream->is_monophonic() && is_playing()) {
		stop();
	}
	Ref<AudioStreamPlayback> stream_playback = stream->instantiate_playback();
	ERR_FAIL_COND_MSG(stream_playback.is_null(), "Failed to instantiate playback.");

	stream_playbacks.push_back(stream_playback);
	setplayback = stream_playback;
	setplay.set(p_from_pos);
	active.set();
	set_physics_process_internal(true);
}

void AudioStreamPlayer2D::seek(float p_seconds) {
	if (is_playing()) {
		stop();
		play(p_seconds);
	}
}

void AudioStreamPlayer2D::stop() {
	setplay.set(-1);
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->stop_playback_stream(playback);
	}
	stream_playbacks.clear();
	active.clear();
	set_physics_process_internal(false);
}

bool AudioStreamPlayer2D::is_playing() const {
	for (const Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		if (AudioServer::get_singleton()->is_playback_active(playback)) {
			return true;
		}
	}
	if (setplay.get() >= 0) {
		return true; // play() has been called this frame, but no playback exists just yet.
	}
	return false;
}

float AudioStreamPlayer2D::get_playback_position() {
	// Return the playback position of the most recently started playback stream.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->get_playback_position(stream_playbacks[stream_playbacks.size() - 1]);
	}
	return 0;
}

void AudioStreamPlayer2D::set_bus(const StringName &p_bus) {
	default_bus = p_bus; // This will be pushed to the audio server during the next physics timestep, which is fast enough.
}

StringName AudioStreamPlayer2D::get_bus() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == default_bus) {
			return default_bus;
		}
	}
	return SceneStringNames::get_singleton()->Master;
}

void AudioStreamPlayer2D::set_autoplay(bool p_enable) {
	autoplay = p_enable;
}

bool AudioStreamPlayer2D::is_autoplay_enabled() {
	return autoplay;
}

void AudioStreamPlayer2D::_set_playing(bool p_enable) {
	if (p_enable) {
		play();
	} else {
		stop();
	}
}

bool AudioStreamPlayer2D::_is_active() const {
	return active.is_set();
}

void AudioStreamPlayer2D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "bus") {
		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0) {
				options += ",";
			}
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		p_property.hint_string = options;
	}
}

void AudioStreamPlayer2D::set_max_distance(float p_pixels) {
	ERR_FAIL_COND(p_pixels <= 0.0);
	max_distance = p_pixels;
}

float AudioStreamPlayer2D::get_max_distance() const {
	return max_distance;
}

void AudioStreamPlayer2D::set_attenuation(float p_curve) {
	attenuation = p_curve;
}

float AudioStreamPlayer2D::get_attenuation() const {
	return attenuation;
}

void AudioStreamPlayer2D::set_area_mask(uint32_t p_mask) {
	area_mask = p_mask;
}

uint32_t AudioStreamPlayer2D::get_area_mask() const {
	return area_mask;
}

void AudioStreamPlayer2D::set_stream_paused(bool p_pause) {
	// TODO this does not have perfect recall, fix that maybe? If there are zero playbacks registered with the AudioServer, this bool isn't persisted.
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_paused(playback, p_pause);
	}
}

bool AudioStreamPlayer2D::get_stream_paused() const {
	// There's currently no way to pause some playback streams but not others. Check the first and don't bother looking at the rest.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->is_playback_paused(stream_playbacks[0]);
	}
	return false;
}

bool AudioStreamPlayer2D::has_stream_playback() {
	return !stream_playbacks.is_empty();
}

Ref<AudioStreamPlayback> AudioStreamPlayer2D::get_stream_playback() {
	ERR_FAIL_COND_V_MSG(stream_playbacks.is_empty(), Ref<AudioStreamPlayback>(), "Player is inactive. Call play() before requesting get_stream_playback().");
	return stream_playbacks[stream_playbacks.size() - 1];
}

void AudioStreamPlayer2D::set_max_polyphony(int p_max_polyphony) {
	if (p_max_polyphony > 0) {
		max_polyphony = p_max_polyphony;
	}
}

int AudioStreamPlayer2D::get_max_polyphony() const {
	return max_polyphony;
}

void AudioStreamPlayer2D::set_panning_strength(float p_panning_strength) {
	ERR_FAIL_COND_MSG(p_panning_strength < 0, "Panning strength must be a positive number.");
	panning_strength = p_panning_strength;
}

float AudioStreamPlayer2D::get_panning_strength() const {
	return panning_strength;
}

void AudioStreamPlayer2D::_on_bus_layout_changed() {
	notify_property_list_changed();
}

void AudioStreamPlayer2D::_on_bus_renamed(int p_bus_index, const StringName &p_old_name, const StringName &p_new_name) {
	notify_property_list_changed();
}

void AudioStreamPlayer2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamPlayer2D::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamPlayer2D::get_stream);

	ClassDB::bind_method(D_METHOD("set_volume_db", "volume_db"), &AudioStreamPlayer2D::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &AudioStreamPlayer2D::get_volume_db);

	ClassDB::bind_method(D_METHOD("set_pitch_scale", "pitch_scale"), &AudioStreamPlayer2D::set_pitch_scale);
	ClassDB::bind_method(D_METHOD("get_pitch_scale"), &AudioStreamPlayer2D::get_pitch_scale);

	ClassDB::bind_method(D_METHOD("play", "from_position"), &AudioStreamPlayer2D::play, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "to_position"), &AudioStreamPlayer2D::seek);
	ClassDB::bind_method(D_METHOD("stop"), &AudioStreamPlayer2D::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &AudioStreamPlayer2D::is_playing);
	ClassDB::bind_method(D_METHOD("get_playback_position"), &AudioStreamPlayer2D::get_playback_position);

	ClassDB::bind_method(D_METHOD("set_bus", "bus"), &AudioStreamPlayer2D::set_bus);
	ClassDB::bind_method(D_METHOD("get_bus"), &AudioStreamPlayer2D::get_bus);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enable"), &AudioStreamPlayer2D::set_autoplay);
	ClassDB::bind_method(D_METHOD("is_autoplay_enabled"), &AudioStreamPlayer2D::is_autoplay_enabled);

	ClassDB::bind_method(D_METHOD("_set_playing", "enable"), &AudioStreamPlayer2D::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_active"), &AudioStreamPlayer2D::_is_active);

	ClassDB::bind_method(D_METHOD("set_max_distance", "pixels"), &AudioStreamPlayer2D::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &AudioStreamPlayer2D::get_max_distance);

	ClassDB::bind_method(D_METHOD("set_attenuation", "curve"), &AudioStreamPlayer2D::set_attenuation);
	ClassDB::bind_method(D_METHOD("get_attenuation"), &AudioStreamPlayer2D::get_attenuation);

	ClassDB::bind_method(D_METHOD("set_area_mask", "mask"), &AudioStreamPlayer2D::set_area_mask);
	ClassDB::bind_method(D_METHOD("get_area_mask"), &AudioStreamPlayer2D::get_area_mask);

	ClassDB::bind_method(D_METHOD("set_stream_paused", "pause"), &AudioStreamPlayer2D::set_stream_paused);
	ClassDB::bind_method(D_METHOD("get_stream_paused"), &AudioStreamPlayer2D::get_stream_paused);

	ClassDB::bind_method(D_METHOD("set_max_polyphony", "max_polyphony"), &AudioStreamPlayer2D::set_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_max_polyphony"), &AudioStreamPlayer2D::get_max_polyphony);

	ClassDB::bind_method(D_METHOD("set_panning_strength", "panning_strength"), &AudioStreamPlayer2D::set_panning_strength);
	ClassDB::bind_method(D_METHOD("get_panning_strength"), &AudioStreamPlayer2D::get_panning_strength);

	ClassDB::bind_method(D_METHOD("has_stream_playback"), &AudioStreamPlayer2D::has_stream_playback);
	ClassDB::bind_method(D_METHOD("get_stream_playback"), &AudioStreamPlayer2D::get_stream_playback);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "volume_db", PROPERTY_HINT_RANGE, "-80,24,suffix:dB"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pitch_scale", PROPERTY_HINT_RANGE, "0.01,4,0.01,or_greater"), "set_pitch_scale", "get_pitch_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream_paused", PROPERTY_HINT_NONE, ""), "set_stream_paused", "get_stream_paused");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_distance", PROPERTY_HINT_RANGE, "1,4096,1,or_greater,exp,suffix:px"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_attenuation", "get_attenuation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_polyphony", PROPERTY_HINT_NONE, ""), "set_max_polyphony", "get_max_polyphony");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "panning_strength", PROPERTY_HINT_RANGE, "0,3,0.01,or_greater"), "set_panning_strength", "get_panning_strength");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "area_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_area_mask", "get_area_mask");

	ADD_SIGNAL(MethodInfo("finished"));
}

AudioStreamPlayer2D::AudioStreamPlayer2D() {
	AudioServer::get_singleton()->connect("bus_layout_changed", callable_mp(this, &AudioStreamPlayer2D::_on_bus_layout_changed));
	AudioServer::get_singleton()->connect("bus_renamed", callable_mp(this, &AudioStreamPlayer2D::_on_bus_renamed));
	cached_global_panning_strength = GLOBAL_GET("audio/general/2d_panning_strength");
	set_hide_clip_children(true);
}

AudioStreamPlayer2D::~AudioStreamPlayer2D() {
}
