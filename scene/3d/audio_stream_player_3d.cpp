/*************************************************************************/
/*  audio_stream_player_3d.cpp                                           */
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

#include "audio_stream_player_3d.h"

#include "scene/3d/area_3d.h"
#include "scene/3d/audio_listener_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/viewport.h"

// Based on "A Novel Multichannel Panning Method for Standard and Arbitrary Loudspeaker Configurations" by Ramy Sadek and Chris Kyriakakis (2004)
// Speaker-Placement Correction Amplitude Panning (SPCAP)
class Spcap {
private:
	struct Speaker {
		Vector3 direction;
		real_t effective_number_of_speakers = 0; // precalculated
		mutable real_t squared_gain = 0; // temporary
	};

	Vector<Speaker> speakers;

public:
	Spcap(unsigned int speaker_count, const Vector3 *speaker_directions) {
		this->speakers.resize(speaker_count);
		Speaker *w = this->speakers.ptrw();
		for (unsigned int speaker_num = 0; speaker_num < speaker_count; speaker_num++) {
			w[speaker_num].direction = speaker_directions[speaker_num];
			w[speaker_num].squared_gain = 0.0;
			w[speaker_num].effective_number_of_speakers = 0.0;
			for (unsigned int other_speaker_num = 0; other_speaker_num < speaker_count; other_speaker_num++) {
				w[speaker_num].effective_number_of_speakers += 0.5 * (1.0 + w[speaker_num].direction.dot(w[other_speaker_num].direction));
			}
		}
	}

	unsigned int get_speaker_count() const {
		return (unsigned int)this->speakers.size();
	}

	Vector3 get_speaker_direction(unsigned int index) const {
		return this->speakers.ptr()[index].direction;
	}

	void calculate(const Vector3 &source_direction, real_t tightness, unsigned int volume_count, real_t *volumes) const {
		const Speaker *r = this->speakers.ptr();
		real_t sum_squared_gains = 0.0;
		for (unsigned int speaker_num = 0; speaker_num < (unsigned int)this->speakers.size(); speaker_num++) {
			real_t initial_gain = 0.5 * powf(1.0 + r[speaker_num].direction.dot(source_direction), tightness) / r[speaker_num].effective_number_of_speakers;
			r[speaker_num].squared_gain = initial_gain * initial_gain;
			sum_squared_gains += r[speaker_num].squared_gain;
		}

		for (unsigned int speaker_num = 0; speaker_num < MIN(volume_count, (unsigned int)this->speakers.size()); speaker_num++) {
			volumes[speaker_num] = sqrtf(r[speaker_num].squared_gain / sum_squared_gains);
		}
	}
};

//TODO: hardcoded main speaker directions for 2, 3.1, 5.1 and 7.1 setups - these are simplified and could also be made configurable
static const Vector3 speaker_directions[7] = {
	Vector3(-1.0, 0.0, -1.0).normalized(), // front-left
	Vector3(1.0, 0.0, -1.0).normalized(), // front-right
	Vector3(0.0, 0.0, -1.0).normalized(), // center
	Vector3(-1.0, 0.0, 1.0).normalized(), // rear-left
	Vector3(1.0, 0.0, 1.0).normalized(), // rear-right
	Vector3(-1.0, 0.0, 0.0).normalized(), // side-left
	Vector3(1.0, 0.0, 0.0).normalized(), // side-right
};

void AudioStreamPlayer3D::_calc_output_vol(const Vector3 &source_dir, real_t tightness, Vector<AudioFrame> &output) {
	unsigned int speaker_count = 0; // only main speakers (no LFE)
	switch (AudioServer::get_singleton()->get_speaker_mode()) {
		case AudioServer::SPEAKER_MODE_STEREO:
			speaker_count = 2;
			break;
		case AudioServer::SPEAKER_SURROUND_31:
			speaker_count = 3;
			break;
		case AudioServer::SPEAKER_SURROUND_51:
			speaker_count = 5;
			break;
		case AudioServer::SPEAKER_SURROUND_71:
			speaker_count = 7;
			break;
	}

	Spcap spcap(speaker_count, speaker_directions); //TODO: should only be created/recreated once the speaker mode / speaker positions changes
	real_t volumes[7];
	spcap.calculate(source_dir, tightness, speaker_count, volumes);

	switch (AudioServer::get_singleton()->get_speaker_mode()) {
		case AudioServer::SPEAKER_SURROUND_71:
			output.write[3].l = volumes[5]; // side-left
			output.write[3].r = volumes[6]; // side-right
			[[fallthrough]];
		case AudioServer::SPEAKER_SURROUND_51:
			output.write[2].l = volumes[3]; // rear-left
			output.write[2].r = volumes[4]; // rear-right
			[[fallthrough]];
		case AudioServer::SPEAKER_SURROUND_31:
			output.write[1].r = 1.0; // LFE - always full power
			output.write[1].l = volumes[2]; // center
			[[fallthrough]];
		case AudioServer::SPEAKER_MODE_STEREO:
			output.write[0].r = volumes[1]; // front-right
			output.write[0].l = volumes[0]; // front-left
			break;
	}
}

void AudioStreamPlayer3D::_calc_reverb_vol(Area3D *area, Vector3 listener_area_pos, Vector<AudioFrame> direct_path_vol, Vector<AudioFrame> &reverb_vol) {
	reverb_vol.resize(4);
	reverb_vol.write[0] = AudioFrame(0, 0);
	reverb_vol.write[1] = AudioFrame(0, 0);
	reverb_vol.write[2] = AudioFrame(0, 0);
	reverb_vol.write[3] = AudioFrame(0, 0);

	float uniformity = area->get_reverb_uniformity();
	float area_send = area->get_reverb_amount();

	if (uniformity > 0.0) {
		float distance = listener_area_pos.length();
		float attenuation = Math::db2linear(_get_attenuation_db(distance));

		// Determine the fraction of sound that would come from each speaker if they were all driven uniformly.
		float center_val[3] = { 0.5f, 0.25f, 0.16666f };
		int channel_count = AudioServer::get_singleton()->get_channel_count();
		AudioFrame center_frame(center_val[channel_count - 1], center_val[channel_count - 1]);

		if (attenuation < 1.0) {
			//pan the uniform sound
			Vector3 rev_pos = listener_area_pos;
			rev_pos.y = 0;
			rev_pos.normalize();

			if (channel_count >= 1) {
				// Stereo pair
				float c = rev_pos.x * 0.5 + 0.5;
				reverb_vol.write[0].l = 1.0 - c;
				reverb_vol.write[0].r = c;
			}

			if (channel_count >= 3) {
				// Center pair + Side pair
				float xl = Vector3(-1, 0, -1).normalized().dot(rev_pos) * 0.5 + 0.5;
				float xr = Vector3(1, 0, -1).normalized().dot(rev_pos) * 0.5 + 0.5;

				reverb_vol.write[1].l = xl;
				reverb_vol.write[1].r = xr;
				reverb_vol.write[2].l = 1.0 - xr;
				reverb_vol.write[2].r = 1.0 - xl;
			}

			if (channel_count >= 4) {
				// Rear pair
				// FIXME: Not sure what math should be done here
				float c = rev_pos.x * 0.5 + 0.5;
				reverb_vol.write[3].l = 1.0 - c;
				reverb_vol.write[3].r = c;
			}

			for (int i = 0; i < channel_count; i++) {
				reverb_vol.write[i] = reverb_vol[i].lerp(center_frame, attenuation);
			}
		} else {
			for (int i = 0; i < channel_count; i++) {
				reverb_vol.write[i] = center_frame;
			}
		}

		for (int i = 0; i < channel_count; i++) {
			reverb_vol.write[i] = direct_path_vol[i].lerp(reverb_vol[i] * attenuation, uniformity);
			reverb_vol.write[i] *= area_send;
		}

	} else {
		for (int i = 0; i < 4; i++) {
			reverb_vol.write[i] = direct_path_vol[i] * area_send;
		}
	}
}

float AudioStreamPlayer3D::_get_attenuation_db(float p_distance) const {
	float att = 0;
	switch (attenuation_model) {
		case ATTENUATION_INVERSE_DISTANCE: {
			att = Math::linear2db(1.0 / ((p_distance / unit_size) + CMP_EPSILON));
		} break;
		case ATTENUATION_INVERSE_SQUARE_DISTANCE: {
			float d = (p_distance / unit_size);
			d *= d;
			att = Math::linear2db(1.0 / (d + CMP_EPSILON));
		} break;
		case ATTENUATION_LOGARITHMIC: {
			att = -20 * Math::log(p_distance / unit_size + CMP_EPSILON);
		} break;
		case ATTENUATION_DISABLED:
			break;
		default: {
			ERR_PRINT("Unknown attenuation type");
			break;
		}
	}

	att += unit_db;
	if (att > max_db) {
		att = max_db;
	}

	return att;
}

void AudioStreamPlayer3D::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		velocity_tracker->reset(get_global_transform().origin);
		AudioServer::get_singleton()->add_listener_changed_callback(_listener_changed_cb, this);
		if (autoplay && !Engine::get_singleton()->is_editor_hint()) {
			play();
		}
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		stop();
		AudioServer::get_singleton()->remove_listener_changed_callback(_listener_changed_cb, this);
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

	if (p_what == NOTIFICATION_TRANSFORM_CHANGED) {
		if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
			velocity_tracker->update_position(get_global_transform().origin);
		}
	}

	if (p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {
		//update anything related to position first, if possible of course
		Vector<AudioFrame> volume_vector;
		if (setplay.get() > 0 || (active.is_set() && last_mix_count != AudioServer::get_singleton()->get_mix_count())) {
			volume_vector = _update_panning();
		}

		if (setplay.get() >= 0 && stream.is_valid()) {
			active.set();
			Ref<AudioStreamPlayback> new_playback = stream->instance_playback();
			ERR_FAIL_COND_MSG(new_playback.is_null(), "Failed to instantiate playback.");
			Map<StringName, Vector<AudioFrame>> bus_map;
			bus_map[_get_actual_bus()] = volume_vector;
			AudioServer::get_singleton()->start_playback_stream(new_playback, bus_map, setplay.get(), actual_pitch_scale, linear_attenuation, attenuation_filter_cutoff_hz);
			stream_playbacks.push_back(new_playback);
			setplay.set(-1);
		}

		if (!stream_playbacks.is_empty() && active.is_set()) {
			// Stop playing if no longer active.
			Vector<Ref<AudioStreamPlayback>> playbacks_to_remove;
			for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
				if (playback.is_valid() && !AudioServer::get_singleton()->is_playback_active(playback) && !AudioServer::get_singleton()->is_playback_paused(playback)) {
					emit_signal(SNAME("finished"));
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
		}

		while (stream_playbacks.size() > max_polyphony) {
			AudioServer::get_singleton()->stop_playback_stream(stream_playbacks[0]);
			stream_playbacks.remove(0);
		}
	}
}

Area3D *AudioStreamPlayer3D::_get_overriding_area() {
	//check if any area is diverting sound into a bus
	Ref<World3D> world_3d = get_world_3d();
	ERR_FAIL_COND_V(world_3d.is_null(), nullptr);

	Vector3 global_pos = get_global_transform().origin;

	PhysicsDirectSpaceState3D *space_state = PhysicsServer3D::get_singleton()->space_get_direct_state(world_3d->get_space());

	PhysicsDirectSpaceState3D::ShapeResult sr[MAX_INTERSECT_AREAS];

	int areas = space_state->intersect_point(global_pos, sr, MAX_INTERSECT_AREAS, Set<RID>(), area_mask, false, true);

	for (int i = 0; i < areas; i++) {
		if (!sr[i].collider) {
			continue;
		}

		Area3D *tarea = Object::cast_to<Area3D>(sr[i].collider);
		if (!tarea) {
			continue;
		}

		if (!tarea->is_overriding_audio_bus() && !tarea->is_using_reverb_bus()) {
			continue;
		}

		return tarea;
	}
	return nullptr;
}

StringName AudioStreamPlayer3D::_get_actual_bus() {
	Area3D *overriding_area = _get_overriding_area();
	if (overriding_area && overriding_area->is_overriding_audio_bus() && !overriding_area->is_using_reverb_bus()) {
		return overriding_area->get_audio_bus_name();
	}
	return bus;
}

Vector<AudioFrame> AudioStreamPlayer3D::_update_panning() {
	Vector<AudioFrame> output_volume_vector;
	output_volume_vector.resize(4);
	for (AudioFrame &frame : output_volume_vector) {
		frame = AudioFrame(0, 0);
	}

	if (!active.is_set() || stream.is_null()) {
		return output_volume_vector;
	}

	Vector3 linear_velocity;

	//compute linear velocity for doppler
	if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
		linear_velocity = velocity_tracker->get_tracked_linear_velocity();
	}

	Vector3 global_pos = get_global_transform().origin;

	Ref<World3D> world_3d = get_world_3d();
	ERR_FAIL_COND_V(world_3d.is_null(), output_volume_vector);

	Set<Camera3D *> cameras = world_3d->get_cameras();
	cameras.insert(get_viewport()->get_camera_3d());

	PhysicsDirectSpaceState3D *space_state = PhysicsServer3D::get_singleton()->space_get_direct_state(world_3d->get_space());

	for (Camera3D *camera : cameras) {
		if (!camera) {
			continue;
		}
		Viewport *vp = camera->get_viewport();
		if (!vp) {
			continue;
		}
		if (!vp->is_audio_listener_3d()) {
			continue;
		}

		bool listener_is_camera = true;
		Node3D *listener_node = camera;

		AudioListener3D *listener = vp->get_audio_listener_3d();
		if (listener) {
			listener_node = listener;
			listener_is_camera = false;
		}

		Vector3 local_pos = listener_node->get_global_transform().orthonormalized().affine_inverse().xform(global_pos);

		float dist = local_pos.length();

		Vector3 area_sound_pos;
		Vector3 listener_area_pos;

		Area3D *area = _get_overriding_area();

		if (area && area->is_using_reverb_bus() && area->get_reverb_uniformity() > 0) {
			area_sound_pos = space_state->get_closest_point_to_object_volume(area->get_rid(), listener_node->get_global_transform().origin);
			listener_area_pos = listener_node->get_global_transform().affine_inverse().xform(area_sound_pos);
		}

		if (max_distance > 0) {
			float total_max = max_distance;

			if (area && area->is_using_reverb_bus() && area->get_reverb_uniformity() > 0) {
				total_max = MAX(total_max, listener_area_pos.length());
			}
			if (total_max > max_distance) {
				continue; //can't hear this sound in this listener
			}
		}

		float multiplier = Math::db2linear(_get_attenuation_db(dist));
		if (max_distance > 0) {
			multiplier *= MAX(0, 1.0 - (dist / max_distance));
		}

		float db_att = (1.0 - MIN(1.0, multiplier)) * attenuation_filter_db;

		if (emission_angle_enabled) {
			Vector3 listenertopos = global_pos - listener_node->get_global_transform().origin;
			float c = listenertopos.normalized().dot(get_global_transform().basis.get_axis(2).normalized()); //it's z negative
			float angle = Math::rad2deg(Math::acos(c));
			if (angle > emission_angle) {
				db_att -= -emission_angle_filter_attenuation_db;
			}
		}

		linear_attenuation = Math::db2linear(db_att);
		for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
			AudioServer::get_singleton()->set_playback_highshelf_params(playback, linear_attenuation, attenuation_filter_cutoff_hz);
		}
		//TODO: The lower the second parameter (tightness) the more the sound will "enclose" the listener (more undirected / playing from
		//      speakers not facing the source) - this could be made distance dependent.
		_calc_output_vol(local_pos.normalized(), 4.0, output_volume_vector);

		for (unsigned int k = 0; k < 4; k++) {
			output_volume_vector.write[k] = multiplier * output_volume_vector[k];
		}

		Map<StringName, Vector<AudioFrame>> bus_volumes;
		if (area) {
			if (area->is_overriding_audio_bus()) {
				//override audio bus
				bus_volumes[area->get_audio_bus_name()] = output_volume_vector;
			}

			if (area->is_using_reverb_bus()) {
				StringName reverb_bus_name = area->get_reverb_bus();
				Vector<AudioFrame> reverb_vol;
				_calc_reverb_vol(area, listener_area_pos, output_volume_vector, reverb_vol);
				bus_volumes[reverb_bus_name] = reverb_vol;
			}
		} else {
			bus_volumes[bus] = output_volume_vector;
		}

		for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
			AudioServer::get_singleton()->set_playback_bus_volumes_linear(playback, bus_volumes);
		}

		if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
			Vector3 listener_velocity;

			if (listener_is_camera) {
				listener_velocity = camera->get_doppler_tracked_velocity();
			}

			Vector3 local_velocity = listener_node->get_global_transform().orthonormalized().basis.xform_inv(linear_velocity - listener_velocity);

			if (local_velocity != Vector3()) {
				float approaching = local_pos.normalized().dot(local_velocity.normalized());
				float velocity = local_velocity.length();
				float speed_of_sound = 343.0;

				float doppler_pitch_scale = pitch_scale * speed_of_sound / (speed_of_sound + velocity * approaching);
				doppler_pitch_scale = CLAMP(doppler_pitch_scale, (1 / 8.0), 8.0); //avoid crazy stuff

				actual_pitch_scale = doppler_pitch_scale;
			} else {
				actual_pitch_scale = pitch_scale;
			}
		} else {
			actual_pitch_scale = pitch_scale;
		}
		for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
			AudioServer::get_singleton()->set_playback_pitch_scale(playback, actual_pitch_scale);
		}
	}
	return output_volume_vector;
}

void AudioStreamPlayer3D::set_stream(Ref<AudioStream> p_stream) {
	stop();
	stream = p_stream;
}

Ref<AudioStream> AudioStreamPlayer3D::get_stream() const {
	return stream;
}

void AudioStreamPlayer3D::set_unit_db(float p_volume) {
	unit_db = p_volume;
}

float AudioStreamPlayer3D::get_unit_db() const {
	return unit_db;
}

void AudioStreamPlayer3D::set_unit_size(float p_volume) {
	unit_size = p_volume;
}

float AudioStreamPlayer3D::get_unit_size() const {
	return unit_size;
}

void AudioStreamPlayer3D::set_max_db(float p_boost) {
	max_db = p_boost;
}

float AudioStreamPlayer3D::get_max_db() const {
	return max_db;
}

void AudioStreamPlayer3D::set_pitch_scale(float p_pitch_scale) {
	ERR_FAIL_COND(p_pitch_scale <= 0.0);
	pitch_scale = p_pitch_scale;
}

float AudioStreamPlayer3D::get_pitch_scale() const {
	return pitch_scale;
}

void AudioStreamPlayer3D::play(float p_from_pos) {
	if (stream.is_null()) {
		return;
	}
	ERR_FAIL_COND_MSG(!is_inside_tree(), "Playback can only happen when a node is inside the scene tree");
	if (stream->is_monophonic() && is_playing()) {
		stop();
	}
	setplay.set(p_from_pos);
	active.set();
	set_physics_process_internal(true);
}

void AudioStreamPlayer3D::seek(float p_seconds) {
	stop();
	play(p_seconds);
}

void AudioStreamPlayer3D::stop() {
	setplay.set(-1);
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->stop_playback_stream(playback);
	}
	stream_playbacks.clear();
	active.clear();
	set_physics_process_internal(false);
}

bool AudioStreamPlayer3D::is_playing() const {
	for (const Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		if (AudioServer::get_singleton()->is_playback_active(playback)) {
			return true;
		}
	}
	return false;
}

float AudioStreamPlayer3D::get_playback_position() {
	// Return the playback position of the most recently started playback stream.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->get_playback_position(stream_playbacks[stream_playbacks.size() - 1]);
	}
	return 0;
}

void AudioStreamPlayer3D::set_bus(const StringName &p_bus) {
	//if audio is active, must lock this
	AudioServer::get_singleton()->lock();
	bus = p_bus;
	AudioServer::get_singleton()->unlock();
}

StringName AudioStreamPlayer3D::get_bus() const {
	for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
		if (AudioServer::get_singleton()->get_bus_name(i) == bus) {
			return bus;
		}
	}
	return SNAME("Master");
}

void AudioStreamPlayer3D::set_autoplay(bool p_enable) {
	autoplay = p_enable;
}

bool AudioStreamPlayer3D::is_autoplay_enabled() {
	return autoplay;
}

void AudioStreamPlayer3D::_set_playing(bool p_enable) {
	if (p_enable) {
		play();
	} else {
		stop();
	}
}

bool AudioStreamPlayer3D::_is_active() const {
	return active.is_set();
}

void AudioStreamPlayer3D::_validate_property(PropertyInfo &property) const {
	if (property.name == "bus") {
		String options;
		for (int i = 0; i < AudioServer::get_singleton()->get_bus_count(); i++) {
			if (i > 0) {
				options += ",";
			}
			String name = AudioServer::get_singleton()->get_bus_name(i);
			options += name;
		}

		property.hint_string = options;
	}
}

void AudioStreamPlayer3D::_bus_layout_changed() {
	notify_property_list_changed();
}

void AudioStreamPlayer3D::set_max_distance(float p_metres) {
	ERR_FAIL_COND(p_metres < 0.0);
	max_distance = p_metres;
}

float AudioStreamPlayer3D::get_max_distance() const {
	return max_distance;
}

void AudioStreamPlayer3D::set_area_mask(uint32_t p_mask) {
	area_mask = p_mask;
}

uint32_t AudioStreamPlayer3D::get_area_mask() const {
	return area_mask;
}

void AudioStreamPlayer3D::set_emission_angle_enabled(bool p_enable) {
	emission_angle_enabled = p_enable;
	update_gizmos();
}

bool AudioStreamPlayer3D::is_emission_angle_enabled() const {
	return emission_angle_enabled;
}

void AudioStreamPlayer3D::set_emission_angle(float p_angle) {
	ERR_FAIL_COND(p_angle < 0 || p_angle > 90);
	emission_angle = p_angle;
	update_gizmos();
}

float AudioStreamPlayer3D::get_emission_angle() const {
	return emission_angle;
}

void AudioStreamPlayer3D::set_emission_angle_filter_attenuation_db(float p_angle_attenuation_db) {
	emission_angle_filter_attenuation_db = p_angle_attenuation_db;
}

float AudioStreamPlayer3D::get_emission_angle_filter_attenuation_db() const {
	return emission_angle_filter_attenuation_db;
}

void AudioStreamPlayer3D::set_attenuation_filter_cutoff_hz(float p_hz) {
	attenuation_filter_cutoff_hz = p_hz;
}

float AudioStreamPlayer3D::get_attenuation_filter_cutoff_hz() const {
	return attenuation_filter_cutoff_hz;
}

void AudioStreamPlayer3D::set_attenuation_filter_db(float p_db) {
	attenuation_filter_db = p_db;
}

float AudioStreamPlayer3D::get_attenuation_filter_db() const {
	return attenuation_filter_db;
}

void AudioStreamPlayer3D::set_attenuation_model(AttenuationModel p_model) {
	ERR_FAIL_INDEX((int)p_model, 4);
	attenuation_model = p_model;
}

AudioStreamPlayer3D::AttenuationModel AudioStreamPlayer3D::get_attenuation_model() const {
	return attenuation_model;
}

void AudioStreamPlayer3D::set_doppler_tracking(DopplerTracking p_tracking) {
	if (doppler_tracking == p_tracking) {
		return;
	}

	doppler_tracking = p_tracking;

	if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
		set_notify_transform(true);
		velocity_tracker->set_track_physics_step(doppler_tracking == DOPPLER_TRACKING_PHYSICS_STEP);
		if (is_inside_tree()) {
			velocity_tracker->reset(get_global_transform().origin);
		}
	} else {
		set_notify_transform(false);
	}
}

AudioStreamPlayer3D::DopplerTracking AudioStreamPlayer3D::get_doppler_tracking() const {
	return doppler_tracking;
}

void AudioStreamPlayer3D::set_stream_paused(bool p_pause) {
	// TODO this does not have perfect recall, fix that maybe? If there are zero playbacks registered with the AudioServer, this bool isn't persisted.
	for (Ref<AudioStreamPlayback> &playback : stream_playbacks) {
		AudioServer::get_singleton()->set_playback_paused(playback, p_pause);
	}
}

bool AudioStreamPlayer3D::get_stream_paused() const {
	// There's currently no way to pause some playback streams but not others. Check the first and don't bother looking at the rest.
	if (!stream_playbacks.is_empty()) {
		return AudioServer::get_singleton()->is_playback_paused(stream_playbacks[0]);
	}
	return false;
}

Ref<AudioStreamPlayback> AudioStreamPlayer3D::get_stream_playback() {
	if (!stream_playbacks.is_empty()) {
		return stream_playbacks[stream_playbacks.size() - 1];
	}
	return nullptr;
}

void AudioStreamPlayer3D::set_max_polyphony(int p_max_polyphony) {
	if (p_max_polyphony > 0) {
		max_polyphony = p_max_polyphony;
	}
}

int AudioStreamPlayer3D::get_max_polyphony() const {
	return max_polyphony;
}

void AudioStreamPlayer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamPlayer3D::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamPlayer3D::get_stream);

	ClassDB::bind_method(D_METHOD("set_unit_db", "unit_db"), &AudioStreamPlayer3D::set_unit_db);
	ClassDB::bind_method(D_METHOD("get_unit_db"), &AudioStreamPlayer3D::get_unit_db);

	ClassDB::bind_method(D_METHOD("set_unit_size", "unit_size"), &AudioStreamPlayer3D::set_unit_size);
	ClassDB::bind_method(D_METHOD("get_unit_size"), &AudioStreamPlayer3D::get_unit_size);

	ClassDB::bind_method(D_METHOD("set_max_db", "max_db"), &AudioStreamPlayer3D::set_max_db);
	ClassDB::bind_method(D_METHOD("get_max_db"), &AudioStreamPlayer3D::get_max_db);

	ClassDB::bind_method(D_METHOD("set_pitch_scale", "pitch_scale"), &AudioStreamPlayer3D::set_pitch_scale);
	ClassDB::bind_method(D_METHOD("get_pitch_scale"), &AudioStreamPlayer3D::get_pitch_scale);

	ClassDB::bind_method(D_METHOD("play", "from_position"), &AudioStreamPlayer3D::play, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "to_position"), &AudioStreamPlayer3D::seek);
	ClassDB::bind_method(D_METHOD("stop"), &AudioStreamPlayer3D::stop);

	ClassDB::bind_method(D_METHOD("is_playing"), &AudioStreamPlayer3D::is_playing);
	ClassDB::bind_method(D_METHOD("get_playback_position"), &AudioStreamPlayer3D::get_playback_position);

	ClassDB::bind_method(D_METHOD("set_bus", "bus"), &AudioStreamPlayer3D::set_bus);
	ClassDB::bind_method(D_METHOD("get_bus"), &AudioStreamPlayer3D::get_bus);

	ClassDB::bind_method(D_METHOD("set_autoplay", "enable"), &AudioStreamPlayer3D::set_autoplay);
	ClassDB::bind_method(D_METHOD("is_autoplay_enabled"), &AudioStreamPlayer3D::is_autoplay_enabled);

	ClassDB::bind_method(D_METHOD("_set_playing", "enable"), &AudioStreamPlayer3D::_set_playing);
	ClassDB::bind_method(D_METHOD("_is_active"), &AudioStreamPlayer3D::_is_active);

	ClassDB::bind_method(D_METHOD("set_max_distance", "metres"), &AudioStreamPlayer3D::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &AudioStreamPlayer3D::get_max_distance);

	ClassDB::bind_method(D_METHOD("set_area_mask", "mask"), &AudioStreamPlayer3D::set_area_mask);
	ClassDB::bind_method(D_METHOD("get_area_mask"), &AudioStreamPlayer3D::get_area_mask);

	ClassDB::bind_method(D_METHOD("set_emission_angle", "degrees"), &AudioStreamPlayer3D::set_emission_angle);
	ClassDB::bind_method(D_METHOD("get_emission_angle"), &AudioStreamPlayer3D::get_emission_angle);

	ClassDB::bind_method(D_METHOD("set_emission_angle_enabled", "enabled"), &AudioStreamPlayer3D::set_emission_angle_enabled);
	ClassDB::bind_method(D_METHOD("is_emission_angle_enabled"), &AudioStreamPlayer3D::is_emission_angle_enabled);

	ClassDB::bind_method(D_METHOD("set_emission_angle_filter_attenuation_db", "db"), &AudioStreamPlayer3D::set_emission_angle_filter_attenuation_db);
	ClassDB::bind_method(D_METHOD("get_emission_angle_filter_attenuation_db"), &AudioStreamPlayer3D::get_emission_angle_filter_attenuation_db);

	ClassDB::bind_method(D_METHOD("set_attenuation_filter_cutoff_hz", "degrees"), &AudioStreamPlayer3D::set_attenuation_filter_cutoff_hz);
	ClassDB::bind_method(D_METHOD("get_attenuation_filter_cutoff_hz"), &AudioStreamPlayer3D::get_attenuation_filter_cutoff_hz);

	ClassDB::bind_method(D_METHOD("set_attenuation_filter_db", "db"), &AudioStreamPlayer3D::set_attenuation_filter_db);
	ClassDB::bind_method(D_METHOD("get_attenuation_filter_db"), &AudioStreamPlayer3D::get_attenuation_filter_db);

	ClassDB::bind_method(D_METHOD("set_attenuation_model", "model"), &AudioStreamPlayer3D::set_attenuation_model);
	ClassDB::bind_method(D_METHOD("get_attenuation_model"), &AudioStreamPlayer3D::get_attenuation_model);

	ClassDB::bind_method(D_METHOD("set_doppler_tracking", "mode"), &AudioStreamPlayer3D::set_doppler_tracking);
	ClassDB::bind_method(D_METHOD("get_doppler_tracking"), &AudioStreamPlayer3D::get_doppler_tracking);

	ClassDB::bind_method(D_METHOD("set_stream_paused", "pause"), &AudioStreamPlayer3D::set_stream_paused);
	ClassDB::bind_method(D_METHOD("get_stream_paused"), &AudioStreamPlayer3D::get_stream_paused);

	ClassDB::bind_method(D_METHOD("set_max_polyphony", "max_polyphony"), &AudioStreamPlayer3D::set_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_max_polyphony"), &AudioStreamPlayer3D::get_max_polyphony);

	ClassDB::bind_method(D_METHOD("get_stream_playback"), &AudioStreamPlayer3D::get_stream_playback);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "attenuation_model", PROPERTY_HINT_ENUM, "Inverse,Inverse Square,Log,Disabled"), "set_attenuation_model", "get_attenuation_model");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "unit_db", PROPERTY_HINT_RANGE, "-80,80"), "set_unit_db", "get_unit_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "unit_size", PROPERTY_HINT_RANGE, "0.1,100,0.01,or_greater"), "set_unit_size", "get_unit_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_db", PROPERTY_HINT_RANGE, "-24,6"), "set_max_db", "get_max_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pitch_scale", PROPERTY_HINT_RANGE, "0.01,4,0.01,or_greater"), "set_pitch_scale", "get_pitch_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stream_paused", PROPERTY_HINT_NONE, ""), "set_stream_paused", "get_stream_paused");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_distance", PROPERTY_HINT_RANGE, "0,4096,1,or_greater,exp"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_polyphony", PROPERTY_HINT_NONE, ""), "set_max_polyphony", "get_max_polyphony");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "area_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_area_mask", "get_area_mask");
	ADD_GROUP("Emission Angle", "emission_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "emission_angle_enabled"), "set_emission_angle_enabled", "is_emission_angle_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "emission_angle_degrees", PROPERTY_HINT_RANGE, "0.1,90,0.1,degrees"), "set_emission_angle", "get_emission_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "emission_angle_filter_attenuation_db", PROPERTY_HINT_RANGE, "-80,0,0.1"), "set_emission_angle_filter_attenuation_db", "get_emission_angle_filter_attenuation_db");
	ADD_GROUP("Attenuation Filter", "attenuation_filter_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attenuation_filter_cutoff_hz", PROPERTY_HINT_RANGE, "1,20500,1"), "set_attenuation_filter_cutoff_hz", "get_attenuation_filter_cutoff_hz");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attenuation_filter_db", PROPERTY_HINT_RANGE, "-80,0,0.1"), "set_attenuation_filter_db", "get_attenuation_filter_db");
	ADD_GROUP("Doppler", "doppler_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doppler_tracking", PROPERTY_HINT_ENUM, "Disabled,Idle,Physics"), "set_doppler_tracking", "get_doppler_tracking");

	BIND_ENUM_CONSTANT(ATTENUATION_INVERSE_DISTANCE);
	BIND_ENUM_CONSTANT(ATTENUATION_INVERSE_SQUARE_DISTANCE);
	BIND_ENUM_CONSTANT(ATTENUATION_LOGARITHMIC);
	BIND_ENUM_CONSTANT(ATTENUATION_DISABLED);

	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_DISABLED);
	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_IDLE_STEP);
	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_PHYSICS_STEP);

	ADD_SIGNAL(MethodInfo("finished"));
}

AudioStreamPlayer3D::AudioStreamPlayer3D() {
	velocity_tracker.instantiate();
	AudioServer::get_singleton()->connect("bus_layout_changed", callable_mp(this, &AudioStreamPlayer3D::_bus_layout_changed));
	set_disable_scale(true);
}

AudioStreamPlayer3D::~AudioStreamPlayer3D() {
}
