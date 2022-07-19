/*************************************************************************/
/*  audio_stream_player_2d.cpp                                           */
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

#include "audio_stream_player_2d.h"

#include "core/engine.h"
#include "scene/2d/area_2d.h"
#include "scene/main/viewport.h"

void AudioStreamPlayer2D::_mix_audio() {

	if (!stream_playback.is_valid() || !active.is_set() ||
			(stream_paused && !stream_paused_fade_out)) {
		return;
	}

	if (setseek.get() >= 0.0) {
		stream_playback->start(setseek.get());
		setseek.set(-1.0); //reset seek
	}

	//get data
	AudioFrame *buffer = mix_buffer.ptrw();
	int buffer_size = mix_buffer.size();

	if (stream_paused_fade_out) {
		// Short fadeout ramp
		buffer_size = MIN(buffer_size, 128);
	}

	stream_playback->mix(buffer, pitch_scale, buffer_size);

	//write all outputs
	int oc = output_count.get();
	for (int i = 0; i < oc; i++) {

		Output current = outputs[i];

		//see if current output exists, to keep volume ramp
		bool found = false;
		for (int j = i; j < prev_output_count; j++) {
			if (prev_outputs[j].viewport == current.viewport) {
				if (j != i) {
					SWAP(prev_outputs[j], prev_outputs[i]);
				}
				found = true;
				break;
			}
		}

		if (!found) {
			//create new if was not used before
			if (prev_output_count < MAX_OUTPUTS) {
				prev_outputs[prev_output_count] = prev_outputs[i]; //may be owned by another viewport
				prev_output_count++;
			}
			prev_outputs[i] = current;
		}

		//mix!
		AudioFrame target_volume = stream_paused_fade_out ? AudioFrame(0.f, 0.f) : current.vol;
		AudioFrame vol_prev = stream_paused_fade_in ? AudioFrame(0.f, 0.f) : prev_outputs[i].vol;
		AudioFrame vol_inc = (target_volume - vol_prev) / float(buffer_size);
		AudioFrame vol = vol_prev;

		int cc = AudioServer::get_singleton()->get_channel_count();

		if (cc == 1) {
			if (!AudioServer::get_singleton()->thread_has_channel_mix_buffer(current.bus_index, 0))
				continue; //may have been removed

			AudioFrame *target = AudioServer::get_singleton()->thread_get_channel_mix_buffer(current.bus_index, 0);

			for (int j = 0; j < buffer_size; j++) {

				target[j] += buffer[j] * vol;
				vol += vol_inc;
			}

		} else {
			AudioFrame *targets[4];
			bool valid = true;

			for (int k = 0; k < cc; k++) {
				if (!AudioServer::get_singleton()->thread_has_channel_mix_buffer(current.bus_index, k)) {
					valid = false; //may have been removed
					break;
				}

				targets[k] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(current.bus_index, k);
			}

			if (!valid)
				continue;

			for (int j = 0; j < buffer_size; j++) {

				AudioFrame frame = buffer[j] * vol;
				for (int k = 0; k < cc; k++) {
					targets[k][j] += frame;
				}
				vol += vol_inc;
			}
		}

		prev_outputs[i] = current;
	}

	prev_output_count = oc;

	//stream is no longer active, disable this.
	if (!stream_playback->is_playing()) {
		active.clear();
	}

	output_ready.clear();
	stream_paused_fade_in = false;
	stream_paused_fade_out = false;
}

void AudioStreamPlayer2D::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		AudioServer::get_singleton()->add_callback(_mix_audios, this);
		if (autoplay && !Engine::get_singleton()->is_editor_hint()) {
			play();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		AudioServer::get_singleton()->remove_callback(_mix_audios, this);
	}

	if (p_what == NOTIFICATION_PAUSED) {
		if (!can_process()) {
			// Node can't process so we start fading out to silence
			set_stream_paused(true);
		}
	}

	if (p_what == NOTIFICATION_UNPAUSED) {
		set_stream_paused(false);
	}

	if (p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {

		//update anything related to position first, if possible of course

		if (!output_ready.is_set()) {
			List<Viewport *> viewports;
			Ref<World2D> world_2d = get_world_2d();
			ERR_FAIL_COND(world_2d.is_null());

			int new_output_count = 0;

			Vector2 global_pos = get_global_position();

			int bus_index = AudioServer::get_singleton()->thread_find_bus_index(bus);

			//check if any area is diverting sound into a bus

			Physics2DDirectSpaceState *space_state = Physics2DServer::get_singleton()->space_get_direct_state(world_2d->get_space());

			Physics2DDirectSpaceState::ShapeResult sr[MAX_INTERSECT_AREAS];

			int areas = space_state->intersect_point(global_pos, sr, MAX_INTERSECT_AREAS, Set<RID>(), area_mask, false, true);

			for (int i = 0; i < areas; i++) {

				Area2D *area2d = Object::cast_to<Area2D>(sr[i].collider);
				if (!area2d)
					continue;

				if (!area2d->is_overriding_audio_bus())
					continue;

				StringName bus_name = area2d->get_audio_bus_name();
				bus_index = AudioServer::get_singleton()->thread_find_bus_index(bus_name);
				break;
			}

			world_2d->get_viewport_list(&viewports);
			for (List<Viewport *>::Element *E = viewports.front(); E; E = E->next()) {

				Viewport *vp = E->get();
				if (vp->is_audio_listener_2d()) {

					//compute matrix to convert to screen
					Transform2D to_screen = vp->get_global_canvas_transform() * vp->get_canvas_transform();
					Vector2 screen_size = vp->get_visible_rect().size;

					//screen in global is used for attenuation
					Vector2 screen_in_global = to_screen.affine_inverse().xform(screen_size * 0.5);

					float dist = global_pos.distance_to(screen_in_global); //distance to screen center

					if (dist > max_distance)
						continue; //can't hear this sound in this viewport

					float multiplier = Math::pow(1.0f - dist / max_distance, attenuation);
					multiplier *= Math::db2linear(volume_db); //also apply player volume!

					//point in screen is used for panning
					Vector2 point_in_screen = to_screen.xform(global_pos);

					float pan = CLAMP(point_in_screen.x / screen_size.width, 0.0, 1.0);

					float l = 1.0 - pan;
					float r = pan;

					outputs[new_output_count].vol = AudioFrame(l, r) * multiplier;
					outputs[new_output_count].bus_index = bus_index;
					outputs[new_output_count].viewport = vp; //keep pointer only for reference
					new_output_count++;
					if (new_output_count == MAX_OUTPUTS)
						break;
				}
			}

			output_count.set(new_output_count);
			output_ready.set();
		}

		//start playing if requested
		if (setplay.get() >= 0.0) {
			setseek.set(setplay.get());
			active.set();
			setplay.set(-1);
			//do not update, this makes it easier to animate (will shut off otherwise)
			//_change_notify("playing"); //update property in editor
		}

		//stop playing if no longer active
		if (!active.is_set()) {
			set_physics_process_internal(false);
			//do not update, this makes it easier to animate (will shut off otherwise)
			//_change_notify("playing"); //update property in editor
			emit_signal("finished");
		}
	}
}

void AudioStreamPlayer2D::set_stream(Ref<AudioStream> p_stream) {

	AudioServer::get_singleton()->lock();

	mix_buffer.resize(AudioServer::get_singleton()->thread_get_mix_buffer_size());

	if (stream_playback.is_valid()) {
		stream_playback.unref();
		stream.unref();
		active.clear();
		setseek.set(-1);
	}

	if (p_stream.is_valid()) {
		stream = p_stream;
		stream_playback = p_stream->instance_playback();
	}

	AudioServer::get_singleton()->unlock();

	if (p_stream.is_valid() && stream_playback.is_null()) {
		stream.unref();
	}
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
	ERR_FAIL_COND(p_pitch_scale <= 0.0);
	pitch_scale = p_pitch_scale;
}
float AudioStreamPlayer2D::get_pitch_scale() const {
	return pitch_scale;
}

void AudioStreamPlayer2D::play(float p_from_pos) {

	if (!is_playing()) {
		// Reset the prev_output_count if the stream is stopped
		prev_output_count = 0;
	}

	if (stream_playback.is_valid()) {
		setplay.set(p_from_pos);
		output_ready.clear();
		set_physics_process_internal(true);
	}
}

void AudioStreamPlayer2D::seek(float p_seconds) {

	if (stream_playback.is_valid()) {
		setseek.set(p_seconds);
	}
}

void AudioStreamPlayer2D::stop() {

	if (stream_playback.is_valid()) {
		active.clear();
		set_physics_process_internal(false);
		setplay.set(-1);
	}
}

bool AudioStreamPlayer2D::is_playing() const {

	if (stream_playback.is_valid()) {
		return active.is_set() || setplay.get() >= 0;
	}

	return false;
}

float AudioStreamPlayer2D::get_playback_position() {

	if (stream_playback.is_valid()) {
		float ss = setseek.get();
		if (ss >= 0.0) {
			return ss;
		}
		return stream_playback->get_playback_position();
	}

	return 0;
}

void AudioStreamPlayer2D::set_bus(const StringName &p_bus) {

	//if audio is active, must lock this
	AudioServer::get_singleton()->lock();
	bus = p_bus;
	AudioServer::get_singleton()->unlock();
}
StringName AudioStreamPlayer2D::get_bus() const {

	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == bus) {
			return bus;
		}
	}
	return "Master";
}

void AudioStreamPlayer2D::set_autoplay(bool p_enable) {

	autoplay = p_enable;
}
bool AudioStreamPlayer2D::is_autoplay_enabled() {

	return autoplay;
}

void AudioStreamPlayer2D::_set_playing(bool p_enable) {

	if (p_enable)
		play();
	else
		stop();
}
bool AudioStreamPlayer2D::_is_active() const {

	return active.is_set();
}

void AudioStreamPlayer2D::_validate_property(PropertyInfo &property) const {

	if (property.name == "bus") {

		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0)
				options += ",";
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		property.hint_string = options;
	}
}

void AudioStreamPlayer2D::_bus_layout_changed() {

	_change_notify();
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

	if (p_pause != stream_paused) {
		stream_paused = p_pause;
		stream_paused_fade_in = !p_pause;
		stream_paused_fade_out = p_pause;
	}
}

bool AudioStreamPlayer2D::get_stream_paused() const {

	return stream_paused;
}

Ref<AudioStreamPlayback> AudioStreamPlayer2D::get_stream_playback() {
	return stream_playback;
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

	ClassDB::bind_method(D_METHOD("get_stream_playback"), &AudioStreamPlayer2D::get_stream_playback);

	ClassDB::bind_method(D_METHOD("_bus_layout_changed"), &AudioStreamPlayer2D::_bus_layout_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE, "-80,24"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "pitch_scale", PROPERTY_HINT_RANGE, "0.01,4,0.01,or_greater"), "set_pitch_scale", "get_pitch_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream_paused", PROPERTY_HINT_NONE, ""), "set_stream_paused", "get_stream_paused");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_distance", PROPERTY_HINT_EXP_RANGE, "1,4096,1,or_greater"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "attenuation", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_attenuation", "get_attenuation");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "area_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_area_mask", "get_area_mask");

	ADD_SIGNAL(MethodInfo("finished"));
}

AudioStreamPlayer2D::AudioStreamPlayer2D() {

	volume_db = 0;
	pitch_scale = 1.0;
	autoplay = false;
	setseek.set(-1);
	prev_output_count = 0;
	max_distance = 2000;
	attenuation = 1;
	setplay.set(-1);
	area_mask = 1;
	stream_paused = false;
	stream_paused_fade_in = false;
	stream_paused_fade_out = false;
	AudioServer::get_singleton()->connect("bus_layout_changed", this, "_bus_layout_changed");
}

AudioStreamPlayer2D::~AudioStreamPlayer2D() {
}
