
#include "audio_stream_player_2d.h"

#include "engine.h"
#include "scene/2d/area_2d.h"
#include "scene/main/viewport.h"

void AudioStreamPlayer2D::_mix_audio() {

	if (!stream_playback.is_valid()) {
		return;
	}

	if (!active) {
		return;
	}

	if (setseek >= 0.0) {
		stream_playback->start(setseek);
		setseek = -1.0; //reset seek
	}

	//get data
	AudioFrame *buffer = mix_buffer.ptrw();
	int buffer_size = mix_buffer.size();

	//mix
	stream_playback->mix(buffer, 1.0, buffer_size);

	//write all outputs
	for (int i = 0; i < output_count; i++) {

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
		AudioFrame vol_inc = (current.vol - prev_outputs[i].vol) / float(buffer_size);
		AudioFrame vol = current.vol;

		int cc = AudioServer::get_singleton()->get_channel_count();

		if (cc == 1) {
			AudioFrame *target = AudioServer::get_singleton()->thread_get_channel_mix_buffer(current.bus_index, 0);

			for (int j = 0; j < buffer_size; j++) {

				target[j] += buffer[j] * vol;
				vol += vol_inc;
			}

		} else {
			AudioFrame *targets[4];

			for (int k = 0; k < cc; k++) {
				targets[k] = AudioServer::get_singleton()->thread_get_channel_mix_buffer(current.bus_index, k);
			}

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

	prev_output_count = output_count;

	//stream is no longer active, disable this.
	if (!stream_playback->is_playing()) {
		active = false;
	}

	output_ready = false;
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

	if (p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {

		//update anything related to position first, if possible of course

		if (!output_ready) {
			List<Viewport *> viewports;
			Ref<World2D> world_2d = get_world_2d();
			ERR_FAIL_COND(world_2d.is_null());

			int new_output_count = 0;

			Vector2 global_pos = get_global_position();

			int bus_index = AudioServer::get_singleton()->thread_find_bus_index(bus);

			//check if any area is diverting sound into a bus

			Physics2DDirectSpaceState *space_state = Physics2DServer::get_singleton()->space_get_direct_state(world_2d->get_space());

			Physics2DDirectSpaceState::ShapeResult sr[MAX_INTERSECT_AREAS];

			int areas = space_state->intersect_point(global_pos, sr, MAX_INTERSECT_AREAS, Set<RID>(), area_mask);

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
						continue; //cant hear this sound in this viewport

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

			output_count = new_output_count;
			output_ready = true;
		}

		//start playing if requested
		if (setplay >= 0.0) {
			setseek = setplay;
			active = true;
			setplay = -1;
			//do not update, this makes it easier to animate (will shut off otherise)
			//_change_notify("playing"); //update property in editor
		}

		//stop playing if no longer active
		if (!active) {
			set_physics_process_internal(false);
			//do not update, this makes it easier to animate (will shut off otherise)
			//_change_notify("playing"); //update property in editor
			emit_signal("finished");
		}
	}
}

void AudioStreamPlayer2D::set_stream(Ref<AudioStream> p_stream) {

	ERR_FAIL_COND(!p_stream.is_valid());
	AudioServer::get_singleton()->lock();

	mix_buffer.resize(AudioServer::get_singleton()->thread_get_mix_buffer_size());

	if (stream_playback.is_valid()) {
		stream_playback.unref();
		stream.unref();
		active = false;
		setseek = -1;
	}

	stream = p_stream;
	stream_playback = p_stream->instance_playback();

	AudioServer::get_singleton()->unlock();

	if (stream_playback.is_null()) {
		stream.unref();
		ERR_FAIL_COND(stream_playback.is_null());
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

void AudioStreamPlayer2D::play(float p_from_pos) {

	if (stream_playback.is_valid()) {
		setplay = p_from_pos;
		output_ready = false;
		set_physics_process_internal(true);
	}
}

void AudioStreamPlayer2D::seek(float p_seconds) {

	if (stream_playback.is_valid()) {
		setseek = p_seconds;
	}
}

void AudioStreamPlayer2D::stop() {

	if (stream_playback.is_valid()) {
		active = false;
		set_physics_process_internal(false);
		setplay = -1;
	}
}

bool AudioStreamPlayer2D::is_playing() const {

	if (stream_playback.is_valid()) {
		return active; // && stream_playback->is_playing();
	}

	return false;
}

float AudioStreamPlayer2D::get_playback_position() {

	if (stream_playback.is_valid()) {
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

	return active;
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

void AudioStreamPlayer2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamPlayer2D::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamPlayer2D::get_stream);

	ClassDB::bind_method(D_METHOD("set_volume_db", "volume_db"), &AudioStreamPlayer2D::set_volume_db);
	ClassDB::bind_method(D_METHOD("get_volume_db"), &AudioStreamPlayer2D::get_volume_db);

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

	ClassDB::bind_method(D_METHOD("_bus_layout_changed"), &AudioStreamPlayer2D::_bus_layout_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_stream", "get_stream");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_db", PROPERTY_HINT_RANGE, "-80,24"), "set_volume_db", "get_volume_db");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playing", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "_set_playing", "is_playing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autoplay"), "set_autoplay", "is_autoplay_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_distance", PROPERTY_HINT_RANGE, "1,65536,1"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "attenuation", PROPERTY_HINT_EXP_EASING), "set_attenuation", "get_attenuation");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bus", PROPERTY_HINT_ENUM, ""), "set_bus", "get_bus");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "area_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_area_mask", "get_area_mask");

	ADD_SIGNAL(MethodInfo("finished"));
}

AudioStreamPlayer2D::AudioStreamPlayer2D() {

	volume_db = 0;
	autoplay = false;
	setseek = -1;
	active = false;
	output_count = 0;
	prev_output_count = 0;
	max_distance = 2000;
	attenuation = 1;
	setplay = -1;
	output_ready = false;
	area_mask = 1;
	AudioServer::get_singleton()->connect("bus_layout_changed", this, "_bus_layout_changed");
}

AudioStreamPlayer2D::~AudioStreamPlayer2D() {
}
