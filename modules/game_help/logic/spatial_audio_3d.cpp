
#include "spatial_audio_3d.hpp"
#include "core/math/math_funcs.h"
#include "core/math/math_defs.h"
#include "servers/audio/effects/audio_effect_delay.h"
#include "servers/audio/effects/audio_effect_reverb.h"
#include "servers/audio/effects/audio_effect_filter.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"	


void SpatialAudio3D::_ready()
{
	player_camera = get_viewport()->get_camera_3d();
	raycasts_coords.clear();
	raycasts_coords.push_back(Vector3(0, 0, max_raycast_distance)); // N
	raycasts_coords.push_back(Vector3(max_raycast_distance, 0, max_raycast_distance)); // NW
	raycasts_coords.push_back(Vector3(max_raycast_distance, 0, 0)); // W
	raycasts_coords.push_back(Vector3(max_raycast_distance, 0,  - max_raycast_distance)); // SW
	raycasts_coords.push_back(Vector3(0, 0,  - max_raycast_distance)); // S
	raycasts_coords.push_back(Vector3( - max_raycast_distance, 0,  - max_raycast_distance)); // SE
	raycasts_coords.push_back(Vector3( - max_raycast_distance, 0, 0)); // E
	raycasts_coords.push_back(Vector3( - max_raycast_distance, 0, max_raycast_distance)); // NE
	raycasts_coords.push_back(Vector3(0, max_raycast_distance, 0));
	// add soundsource
	soundsource = memnew(Soundsource);// @export variables are overridden on _init, so we pass them right after to have the actual values

	// @export vars
	soundsource->loop = loop;
	soundsource->shut_up = shut_up;
	soundsource->debug = debug;
	soundsource->audiophysics_ticks = audiophysics_ticks;
	soundsource->speed_of_sound = speed_of_sound;
	soundsource->max_raycast_distance = max_raycast_distance;
	soundsource->roomsize_multiplicator = roomsize_multiplicator;
	soundsource->collision_mask = collision_mask;
	soundsource->reverb_enabled = reverb_enabled;
	soundsource->reverb_volume_db = reverb_volume_db;
	soundsource->bass_proximity = bass_proximity;
	soundsource->reverb_fadeintime = reverb_fadeintime;
	soundsource->reverb_fadeouttime = reverb_fadeouttime;
	soundsource->occlusion_lp_cutoff = occlusion_lp_cutoff;
	soundsource->occlusion_fadetime = occlusion_fadetime;

	// module vars
	soundsource->set_name(get_name());
	soundsource->set_stream(get_stream());
	soundsource->set_volume_db(get_volume_db());
	soundsource->set_max_polyphony(get_max_polyphony());
	soundsource->set_doppler_tracking(get_doppler_tracking());
	soundsource->soundsource = soundsource;

	add_child(soundsource);// this calls _ready(), so vars needs to be set before add_child()
	// but position can only be set when node is in tree, e.g. after add_child()

	// calculate how often (physics-frames) we run
	tick_interval = MAX(0, (int)(Engine::get_singleton()->get_physics_ticks_per_second() / MAX(1, audiophysics_ticks)));

	// start playing on autoplay
	if(is_autoplay_enabled())
	{
		stop();
		/* await get_tree()->create_timer(0->1)->timeout; */ // no equivalent to await in c++ !// reverbers aren't ready.
		soundsource->do_play();
	}
}

void SpatialAudio3D::physics_process(double _delta)
{
	// keep rotation locked
	set_global_rotation(Vector3(0, 0, 0));

	if(_tick_counter >= tick_interval)
	{
		soundsource->update_run();
		_tick_counter = 0;
	}

	// update tick
	_tick_counter += 1;
}

void SpatialAudio3D::do_play()
{
	soundsource->do_play();
}

void SpatialAudio3D::do_stop()
{
	soundsource->do_stop();
}

void SpatialAudio3D::do_set_stream(Ref<AudioStream> sound)
{
	soundsource->do_set_stream(sound);
}

RayCast3D* SpatialAudio3D::create_raycast(StringName name_p, const Vector3& target_position)
{
	RayCast3D* r = memnew(RayCast3D);
	r->set_name(name_p);
	r->set_target_position(target_position);
	r->set_collision_mask(collision_mask);
	r->set_enabled(false);
	return r;
}

LocalVector<RayCast3D*> SpatialAudio3D::create_raycast_sector(int start_angle, double width_factor, int bearing_raycount, int heading_count)
{
	LocalVector<RayCast3D*> rays;
	int i = 0;
	for(double heading=0; heading< heading_count; heading+=1)
	{// 0: ground, 7: above
		heading = heading / Math_PI;
		for(double bearing=(bearing_raycount / 2.0 *  - 1); bearing< (bearing_raycount / 2.0); bearing+=1)
		{// -7: right, 7: left
			bearing = bearing / bearing_raycount * width_factor;
			RayCast3D* mr  = memnew(RayCast3D);
			mr->set_name( "mray" + itos(i));
			add_child(mr);
			mr->set_target_position(Vector3(max_raycast_distance, 0, 0));
			mr->set_collision_mask(collision_mask);
			mr->set_debug_shape_custom_color(Color("#ff0"));
			mr->set_debug_shape_thickness(1);
			mr->set_rotation(Vector3(0, bearing, heading) + Vector3(0, Math::deg_to_rad( - (float)start_angle), 0));
			mr->set_enabled(false);

			if(debug)
			{
				Debugray* dray = memnew(Debugray);
				dray->set_visibility_range_end(max_raycast_distance);
				dray->set_visibility_range_end_margin(max_raycast_distance / 10.0);
				dray->set_visibility_range_fade_mode( GeometryInstance3D::VisibilityRangeFadeMode::VISIBILITY_RANGE_FADE_SELF);
				mr->add_child(dray);
			}

			rays.push_back(mr);
			i += 1;
		}
	}
	return rays;
}

Soundplayer* SpatialAudio3D::create_soundplayer(String name_p, bool with_reverb_fx)
{
	Soundplayer* soundplayer = memnew(Soundplayer);

	// @export vars
	soundplayer->loop = loop;
	soundplayer->shut_up = shut_up;
	soundplayer->debug = debug;
	soundplayer->audiophysics_ticks = audiophysics_ticks;
	soundplayer->speed_of_sound = speed_of_sound;
	soundplayer->max_raycast_distance = max_raycast_distance;
	soundplayer->roomsize_multiplicator = roomsize_multiplicator;
	soundplayer->reverb_enabled = reverb_enabled;
	soundplayer->reverb_volume_db = reverb_volume_db;
	soundplayer->bass_proximity = bass_proximity;
	soundplayer->reverb_fadeintime = reverb_fadeintime;
	soundplayer->reverb_fadeouttime = reverb_fadeouttime;
	soundplayer->occlusion_lp_cutoff = occlusion_lp_cutoff;
	soundplayer->occlusion_fadetime = occlusion_fadetime;

	// module vars
	soundplayer->set_name(name_p);
	soundplayer->set_stream(get_stream());
	soundplayer->set_volume_db(get_volume_db());
	soundplayer->set_max_polyphony(get_max_polyphony()); // max_polyphony;

	// BUG: Doppler-tracking interferes with set_delay.
	// The delay only grows bigger and bigger when doppler-tracking is used, so we won't use it for now.
	soundplayer->set_doppler_tracking(AudioStreamPlayer3D::DopplerTracking::DOPPLER_TRACKING_DISABLED);

	soundplayer->with_reverb_fx = with_reverb_fx;
	soundplayer->soundsource = soundsource;

	return soundplayer;
}

void SpatialAudio3D::create_audiobus(StringName bus_name, int vol_db)
{
	int a = AudioServer::get_singleton()->get_bus_count();
	AudioServer::get_singleton()->add_bus(a);
	AudioServer::get_singleton()->set_bus_name(a, bus_name);
	AudioServer::get_singleton()->set_bus_volume_db(a, vol_db);
	AudioServer::get_singleton()->set_bus_send(a, "Master");
}

void SpatialAudio3D::remove_audiobus(String bus_name)
{
	//print("removing bus ", bus_name)
	int a = AudioServer::get_singleton()->get_bus_index(bus_name);
	AudioServer::get_singleton()->remove_bus(a);
}

void SpatialAudio3D::set_audiobus_volume(StringName bus_name, float vol_db, double fadetime, Ref<Tween> tweenvar)
{
	//if debug: print("set_audiobus_volume(%s, %f, %f)" % [bus_name, vol_db, fadetime])
	int a = AudioServer::get_singleton()->get_bus_index(bus_name);
	if(fadetime == 0)
	{
		AudioServer::get_singleton()->set_bus_volume_db(a, vol_db);
	}
	else
	{
		double current_volume = AudioServer::get_singleton()->get_bus_volume_db(a);
		if(tweenvar.is_valid())
		{
			tweenvar->play();
			
			tweenvar->tween_method(callable_mp(this,&SpatialAudio3D::tweensetvol).bind(a), current_volume, vol_db, fadetime);
		}
		else
		{
			if(fade_tween.is_valid())
			{
				fade_tween->kill();
			}
			fade_tween = create_tween();
			fade_tween->tween_method(callable_mp(this,&SpatialAudio3D::tweensetvol).bind(a), current_volume, vol_db, fadetime);
		}
	}
}

void SpatialAudio3D::tweensetvol(float vol, int bus_index)
{
	AudioServer::get_singleton()->set_bus_volume_db(bus_index, vol);
}

void SpatialAudio3D::add_audioeffect(StringName bus_name, fx effect_type)
{
	int a = AudioServer::get_singleton()->get_bus_index(bus_name);

	if(effect_type == fx::delay)
	{
		Ref<AudioEffectDelay> delay ;
		delay.instantiate();
		delay->set_dry(0);
		delay->set_tap1_delay_ms(0);
		delay->set_tap1_level_db(0);
		delay->set_tap1_pan(0.0);
		delay->set_tap2_active(false);
		AudioServer::get_singleton()->add_bus_effect(a, delay, 0);
	}

	if(effect_type == fx::reverb)
	{
		Ref<AudioEffectReverb> reverb;
		reverb.instantiate();
		reverb->set_dry(0);
		reverb->set_spread(0);
		reverb->set_hpf(0.2);
		reverb->set_dry(0);
		reverb->set_wet(1);
		reverb->set_predelay_feedback(0);
		AudioServer::get_singleton()->add_bus_effect(a, reverb, 1);
	}

	if(effect_type == fx::lowpass)
	{
		Ref<AudioEffectLowPassFilter> lowpass;
		lowpass.instantiate();
		lowpass->set_cutoff(20500);
		AudioServer::get_singleton()->add_bus_effect(a, lowpass, 2);
	}
}

void SpatialAudio3D::set_audioeffect(StringName bus_name, fx effect_type, Dictionary params)
{
	//if debug: print("set_audioeffect(%s, %s, %s)" % [bus_name, effect_type, JSON.stringify(params)])
	int a = AudioServer::get_singleton()->get_bus_index(bus_name);

	if(effect_type == fx::delay)
	{
		Ref<AudioEffectDelay> delay_fx = AudioServer::get_singleton()->get_bus_effect(a, 0);
		delay_fx->set_tap1_delay_ms(params["delay"]);
	}

	if(effect_type == fx::reverb)
	{
		Ref<AudioEffectReverb> reverb_fx = AudioServer::get_singleton()->get_bus_effect(a, 1);
		reverb_fx->set_room_size(params["room_size"]);
		reverb_fx->set_wet(params["wetness"]);
		reverb_fx->set_dry(1.0 - (float)params["wetness"]);
	}

	if(effect_type == fx::reverb_hipass)
	{
		Ref<AudioEffectReverb> reverb_hipass_fx = AudioServer::get_singleton()->get_bus_effect(a, 1);
		reverb_hipass_fx->set_hpf(params["hipass"]);
	}

	if(effect_type == fx::lowpass)
	{
		Ref<AudioEffectLowPassFilter> lowpass_fx = AudioServer::get_singleton()->get_bus_effect(a, 2);
		float fadetime = params["fadetime"];
		float lowpass = params["lowpass"];
		if(fadetime== 0.0)
		{
			lowpass_fx->set_cutoff(lowpass);
		}
		else
		{
			if(fadetime > 0.0)
			{
				if(lowpass_tween.is_valid())
				{
					lowpass_tween->kill();
				}
				lowpass_tween = create_tween();
				// fading in higher frequencies (20'000 - 6'000) is less noticeable than fading in the lower frequencies.
				if(lowpass < lowpass_fx->get_cutoff())
				{// fade fast through the higher frequencies
					lowpass_tween->set_ease(Tween::EaseType::EASE_OUT);
				}
				else
				{
					lowpass_tween->set_ease(Tween::EaseType::EASE_IN);// we are low, start slowly
				}
				lowpass_tween->set_trans(Tween::TransitionType::TRANS_QUINT);
				lowpass_tween->tween_property(lowpass_fx.ptr(), NodePath("cutoff_hz"), lowpass, fadetime)->from_current();
			}
			else
			{
				//UtilityFunctions::push_error("ERROR: fadetime not set in SpatialAudioStreamPlayer3D --> set_audioeffect --> fx.lowpass!");
			}
		}
	}
}

void SpatialAudio3D::toggle_audioeffect(StringName bus_name, fx effect_type, bool enabled)
{
	int a = AudioServer::get_singleton()->get_bus_index(bus_name);

	if(effect_type == fx::delay)
	{
		AudioServer::get_singleton()->set_bus_effect_enabled(a, 0, enabled);
	}

	if(effect_type == fx::reverb)
	{
		AudioServer::get_singleton()->set_bus_effect_enabled(a, 1, enabled);
	}

	if(effect_type == fx::lowpass)
	{
		AudioServer::get_singleton()->set_bus_effect_enabled(a, 2, enabled);
	}
}

float SpatialAudio3D::calculate_delay(double distance)
{
	return Math::round(distance / speed_of_sound * 1000);
}

void Soundsource::set_delay_ms(int v)
{
	if(delay_ms != v)
	{
		delay_ms = v;
		Soundplayer* _soundplayer_active = soundplayer_active;
		Soundplayer* _soundplayer_standby = soundplayer_standby;

		if(soundplayer_active->state == Soundplayer::active)
		{
			soundplayer_standby->set_delay_ms(v);
			soundplayer_active->set_inactive(xfadetime);
			soundplayer_standby->set_active(xfadetime);

			soundplayer_active = _soundplayer_standby;
			soundplayer_standby = _soundplayer_active;
		}
	}
}

int Soundsource::get_delay_ms() {
	return delay_ms;
}

void Soundsource::_ready()
{
	String name = get_name();
	if(debug)
	{
		//print("spawning soundsource: ", name)
		Debugsphere* ds = memnew(Debugsphere);
		ds->max_raycast_distance = max_raycast_distance;
		ds->size = 0.0;
		ds->label_offset = Vector3(0, 3, 0);
		ds->set_name("Debugsphere " + name);
		add_child(ds);
		debugsphere = ds;
		ds->line1 = "☼ " + name;
	}

	// create soundplayers for delay-crossfading
	soundplayer_active = create_soundplayer(name + "-A", false);
	soundplayer_standby = create_soundplayer(name + "-B", false);
	add_child(soundplayer_active);
	add_child(soundplayer_standby);

	// set one active
	soundplayer_active->set_active();
	soundplayer_standby->set_inactive();

	// Only create raycasts, reverbers and measurement rays when reverb is enabled
	if(reverb_enabled)
	{

		// create raycasts
		int raycast_index = 0;
		for(Variant c : raycasts_coords)
		{
			RayCast3D* rc = create_raycast("ray " + name + "#" + String::num_int64(raycast_index), c);

			Debugray* dray = memnew(Debugray);
			dray->visibility_range_end = max_raycast_distance * 1.3;
			dray->visibility_range_end_margin = max_raycast_distance / 10.0;
			dray->visibility_range_fade_mode = GeometryInstance3D::VisibilityRangeFadeMode::VISIBILITY_RANGE_FADE_SELF;
			rc->add_child(dray);

			Debugray* dray_normal = memnew(Debugray);
			rc->add_child(dray_normal);

			add_child(rc);
			raycasts.push_back(rc);
		}

		// create reverbers
		raycast_index = 0;
		for(RayCast3D* rc : raycasts)
		{
			Reverber* reverber = create_reverber(name, raycast_index);
			get_tree()->get_root()->add_child(reverber);
			reverbers.push_back(reverber);
			raycast_index += 1;
		}

		// create measurement rays
		measurement_rays = create_raycast_sector(0, 2 * Math_PI, 12, 3);
		for(RayCast3D* mr : measurement_rays)
		{
			distances.push_back( - 1);
		}
	}
}

void Soundsource::physics_process(double _delta)
{
	if(debug)
	{
		dump_debug();
		debugsphere->update_label();
	}
}

void Soundsource::_exit_tree()
{
	for(Reverber* r : reverbers)
	{
		r->queue_free();
	}
	TypedArray<Node> children = get_children();
	for(int i = 0; i < children.size(); i++)
	{
		Node* c = Object::cast_to<Node>(children[i]);
		remove_child(c);
	}
}

Reverber* Soundsource::create_reverber(StringName name_p, int raycast_index)
{
	Reverber* reverber = memnew(Reverber);

	// @export vars
	reverber->loop = loop;
	reverber->shut_up = shut_up;
	reverber->debug = debug;
	reverber->audiophysics_ticks = audiophysics_ticks;
	reverber->speed_of_sound = speed_of_sound;
	reverber->max_raycast_distance = max_raycast_distance;
	reverber->roomsize_multiplicator = roomsize_multiplicator;
	reverber->reverb_enabled = reverb_enabled;
	reverber->reverb_volume_db = reverb_volume_db;
	reverber->bass_proximity = bass_proximity;
	reverber->reverb_fadeintime = reverb_fadeintime;
	reverber->reverb_fadeouttime = reverb_fadeouttime;
	reverber->occlusion_lp_cutoff = occlusion_lp_cutoff;
	reverber->occlusion_fadetime = occlusion_fadetime;

	// module vars
	reverber->set_name(name_p);
	reverber->set_stream(get_stream());
	reverber->set_volume_db(get_volume_db());
	reverber->set_max_polyphony(get_max_polyphony());
	reverber->soundsource = soundsource;

	// reverber vars
	reverber->raycast_index = raycast_index;

	return reverber;
}

void Soundsource::do_set_stream(Ref<AudioStream> s)
{
	set_stream(s);

	soundplayer_active->set_stream(s);
	soundplayer_standby->set_stream(s);

	if(reverb_enabled)
	{
		for(Reverber* r : reverbers)
		{
			if(r != nullptr)
			{
				r->soundplayer_active->set_stream(s);
				r->soundplayer_standby->set_stream(s);
			}
		}
	}
}

void Soundsource::do_play()
{
	if(shut_up)
	{
		return ;
	}

	// when playing finishes, signals from all soundplayers are being emitted at the same time.
	// we just want to trigger ONE do_play() when looping is enabled.
	if(OS::get_singleton()->get_ticks_msec() - _playing_since < 100)
	{
		return ;
	}
	_playing_since = OS::get_singleton()->get_ticks_msec();

	if(reverb_enabled)
	{
		for(Reverber* r : reverbers)
		{
			r->do_play();
		}
	}

	// start play on both
	soundplayer_active->do_play();
	soundplayer_standby->do_play();
}

void Soundsource::do_stop()
{
	if(reverb_enabled)
	{
		for(Reverber* r : reverbers)
		{
			r->do_stop();
		}
	}

	soundplayer_active->do_stop();
	soundplayer_standby->do_stop();
}

void Soundsource::update_run()
{
	distance_to_player = get_global_position().distance_to(player_camera->get_global_position());

	// only update delay when moved >5m since last update
	if(Math::abs(distance_to_player - distance_to_player_since_last_delay_update) > 5)
	{
		delay_ms = calculate_delay(distance_to_player);
		distance_to_player_since_last_delay_update = distance_to_player;
	}

	// calculate my occlusion
	soundplayer_active->update_effect_params();

	if(reverb_enabled)
	{

		// update distances for all reverbers
		calculate_all_distances();

		// calculate room_size and wetness for reverbers
		calculate_reverb();
	}
}

void Soundsource::calculate_all_distances()
{
	int raycast_index = 0;
	for(Reverber* reverber : reverbers)
	{
		RayCast3D* rc = raycasts[raycast_index];
		Vector3 target_position;
		bool colliding = false;
		Debugray* dray = Object::cast_to<Debugray>(rc->get_child(0));
		Debugray* dray_normal = Object::cast_to<Debugray>(rc->get_child(1));

		// if raycast is colliding and the angle between raycast and normal is "over 2.5 radians" (3.14 == exact 180° reflection; 2.5 == somewhat 30° tolerance)
		rc->force_raycast_update();
		if(rc->is_colliding() && (rc->get_collision_normal().angle_to(rc->get_target_position()) > 2.5))
		{
			// position reverber 10cm away from the collision/wall, otherwise occlusion detection doesn't work
			target_position = rc->get_collision_point() + rc->get_collision_normal() * 0.1;
			colliding = true;
			if(debug)
			{dray->draw(get_position(), to_local(target_position), Color("#f00"));
			}
			if(debug)
			{dray_normal->draw(to_local(rc->get_collision_point() + rc->get_collision_normal() * 0.1), to_local(rc->get_collision_point() + rc->get_collision_normal()), Color("#0f0"));
			}
		}

		else
		{
			target_position = rc->get_target_position() * 100;// move the inactive point far far away
			if(debug)
			{dray->clear();
			}
			if(debug)
			{dray_normal->clear();
			}
		}

		reverber->update_position(target_position, colliding);
		raycast_index += 1;
	}

	// measure room using measurement rays
	int ri = 0;
	for(RayCast3D* mr : measurement_rays)
	{
		mr->force_raycast_update();
		if(mr->is_colliding())
		{
			distances[ri] = get_global_position().distance_to(mr->get_collision_point());
			//if debug:
			//@warning_ignore("unsafe_method_access")
			//mr.get_child(0).draw(position, to_local(mr.target_position) + global_position, "#00f3")
		}
		else
		{
			distances[ri] =  - 1;
			//if debug:
			//@warning_ignore("unsafe_method_access")
			//mr.get_child(0).draw(position, to_local(mr.target_position) + global_position, "#ff01")
		}
		ri += 1;
	}
}

void Soundsource::calculate_reverb()
{
	// Find the reverb params
	double _room_size = 0.0;
	double _wetness = 1.0;

	Variant total_rays = distances.size();
	for(double distance : distances)
	{
		if(distance >= 0)
		{
			// find the average room size based on the raycast distances that are valid
			_room_size += (distance / (max_raycast_distance / roomsize_multiplicator)) / float(total_rays);
			_room_size = Math::snapped(_room_size, 0.001);
			_room_size = MIN(_room_size, 1.0);
		}
		else
		{
			// if a raycast did not hit anything we will reduce the reverb effect, almost no raycasts should hit when outdoors nowhere near buildings
			_wetness -= 1.0 / float(distances.size());
			_wetness = Math::snapped(_wetness, 0.001);
			_wetness = MAX(_wetness, 0.0);
		}
	}

	room_size = _room_size;
	wetness = _wetness;

	if(debug)
	{
		debugsphere->line2 =  "Room size: " + String::num(room_size);
		debugsphere->line3 = "Wetness: " + String::num( wetness);
	}
}

void Soundsource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_reverber", "name_p", "raycast_index"), &Soundsource::create_reverber);
	ClassDB::bind_method(D_METHOD("do_set_stream", "s"), &Soundsource::do_set_stream);
	ClassDB::bind_method(D_METHOD("do_play"), &Soundsource::do_play);
	ClassDB::bind_method(D_METHOD("do_stop"), &Soundsource::do_stop);
	ClassDB::bind_method(D_METHOD("update_run"), &Soundsource::update_run);
	ClassDB::bind_method(D_METHOD("set_delay_ms", "v"), &Soundsource::set_delay_ms);
	ClassDB::bind_method(D_METHOD("get_delay_ms"), &Soundsource::get_delay_ms);
	ClassDB::bind_method(D_METHOD("calculate_all_distances"), &Soundsource::calculate_all_distances);
	ClassDB::bind_method(D_METHOD("calculate_reverb"), &Soundsource::calculate_reverb);

}

void Reverber::_ready()
{
	set_name( get_name().str() + "#" + String::num_int64(raycast_index));

	// create reverb-soundplayers for AB-reverb
	soundplayer_active = create_soundplayer(get_name().str() + "-A");
	soundplayer_standby = create_soundplayer(get_name().str() + "-B");
	get_tree()->get_root()->add_child(soundplayer_active);
	get_tree()->get_root()->add_child(soundplayer_standby);
}

void Reverber::_exit_tree()
{
	soundplayer_active->queue_free();
	soundplayer_standby->queue_free();
}

void Reverber::physics_process(double _delta)
{

}

void Reverber::do_play()
{
	soundplayer_active->do_play();
	soundplayer_standby->do_play();
}

void Reverber::do_stop()
{
	soundplayer_active->do_stop();
	soundplayer_standby->do_stop();
}

void Reverber::update_position(Vector3 target_position, bool colliding)
{
	Soundplayer* _soundplayer_active = soundplayer_active;
	Soundplayer* _soundplayer_standby = soundplayer_standby;

	double fadeintime = ( reverb_fadeintime < 0 ? 1 * soundsource->wetness : reverb_fadeintime );
	double fadeouttime = ( reverb_fadeouttime < 0 ? 3 * soundsource->wetness : reverb_fadeouttime );

	// make sure inactive soundplayers stay silent
	soundplayer_standby->set_inactive();

	if(get_global_position().distance_to(target_position) > 10)
	{
		set_global_position(target_position);

		soundplayer_active->set_inactive(fadeouttime);
		soundplayer_standby->set_target_position(target_position);
		if(colliding)
		{
			soundplayer_standby->set_active(fadeintime);
			soundplayer_active = _soundplayer_standby;
			soundplayer_standby = _soundplayer_active;
		}
	}

	else if(get_global_position().distance_to(target_position) > 0.3)
	{
		set_global_position(target_position);
		soundplayer_active->update_effect_params();
		soundplayer_active->target_position = target_position;
	}

	// update effects on colliding (active) soundplayers
	if(colliding)
	{
		soundplayer_active->update_effect_params();
	}
}

void Reverber::_bind_methods() {
	ClassDB::bind_method(D_METHOD("do_play"), &Reverber::do_play);
	ClassDB::bind_method(D_METHOD("do_stop"), &Reverber::do_stop);
	ClassDB::bind_method(D_METHOD("update_position", "target_position", "colliding"), &Reverber::update_position);

}

void Soundplayer::set_state(Soundplayer::ss v)
{
	if(state != v)
	{
		state = v;
		//if debug: debugsphere.line4 = ds[v]
		//if debug: print("%s: %s --> %s" % [name, ds[state], ds[v]])
	}
}

Soundplayer::ss Soundplayer::get_state() {
	return state;
}

void Soundplayer::set_target_position(Vector3 v)
{
	if(target_position != v)
	{
		if(state != ss::fading_to_inactive)
		{
			target_position = v;
			set_global_position(v);
		}
	}
}

Vector3 Soundplayer::get_target_position() {
	return target_position;
}

void Soundplayer::set_delay_ms(int v)
{
	if(Math::abs(delay_ms - v) > 10 && (OS::get_singleton()->get_ticks_msec() - delay_updated_at) > 1000)
	{
		delay_ms = v;
		delay_updated_at = OS::get_singleton()->get_ticks_msec();
		if(debug)
		{
			debugsphere->line2 = String("%s ms") + String::num_int64(delay_ms); 
		}
		if(state != fading_to_inactive)
		{
			set_audioeffect(audiobus_name, delay, Dictionary {/* initializer lists are unsupported */ {"delay", delay_ms}, });
		}
	}
}

int Soundplayer::get_delay_ms() {
	return delay_ms;
}

void Soundplayer::set_room_size(double v)
{
	if(room_size != v)
	{
		room_size = v;
		if(state != fading_to_inactive)
		{
			set_audioeffect(audiobus_name, reverb, Dictionary {/* initializer lists are unsupported */ {"room_size", room_size},{"wetness", wetness}, });
		}
	}
}

double Soundplayer::get_room_size() {
	return room_size;
}

void Soundplayer::set_wetness(double v)
{
	if(wetness != v)
	{
		wetness = v;
		// audioeffect is already set together with room_size
	}
}

double Soundplayer::get_wetness() {
	return wetness;
}

void Soundplayer::set_proximity_volume(double v)
{
	if(Math::abs(proximity_volume - v) > 0.1)
	{
		proximity_volume = v;
		if(debug)
		{debugsphere->line3 = String("%d dB") + String::num_int64(v);
		}
		if(state == active)
		{
			set_audiobus_volume(audiobus_name, proximity_volume);
		}
	}
}

double Soundplayer::get_proximity_volume() {
	return proximity_volume;
}

void Soundplayer::set_proximity_bass(double v)
{
	if(proximity_bass != v)
	{
		proximity_bass = v;
		if(state != fading_to_inactive)
		{
			set_audioeffect(audiobus_name, reverb_hipass, Dictionary {/* initializer lists are unsupported */ {"hipass", proximity_bass}, });
		}
	}
}

double Soundplayer::get_proximity_bass() {
	return proximity_bass;
}

void Soundplayer::set_lp_cutoff(int v)
{
	if(Math::abs(lp_cutoff - v) > 20)
	{
		lp_cutoff = v;
		if(debug)
		{debugsphere->line4 = ( v != 20500 ? "occluded" : "" );
		}
		set_audioeffect(audiobus_name, lowpass, Dictionary {/* initializer lists are unsupported */ {"lowpass", lp_cutoff},{"fadetime", occlusion_fadetime}, });
	}
}

int Soundplayer::get_lp_cutoff() {
	return lp_cutoff;
}

void Soundplayer::_ready()
{

	if(debug)
	{
		//print("spawning soundplayer: ", name)
		debugsphere = memnew(Debugsphere);
		debugsphere->max_raycast_distance = max_raycast_distance;
		debugsphere->set_name("Debugsphere " + get_name());
		debugsphere->line1 = "♬";
		add_child(debugsphere);
		debugsphere->set_visible(false);
	}

	// ensure unique name for the mixer
	audiobus_name =  get_name().str() + "_" + String::num_int64( get_instance_id());;

	// create bus and add effects to it
	create_audiobus(audiobus_name);
	add_audioeffect(audiobus_name, delay);
	add_audioeffect(audiobus_name, reverb);
	if(with_reverb_fx == false)
	{
		toggle_audioeffect(audiobus_name, reverb, false);
	}
	add_audioeffect(audiobus_name, lowpass);

	// set initial state to inactive and mute
	state = ss::inactive;
	set_audiobus_volume(audiobus_name,  - 80);

	// set volume according to AudioStreamPlayer3D param
	proximity_volume = get_volume_db();

	// set my bus to this newly created bus.
	set_bus(audiobus_name);

	// create raycast for occlusion test
	occlusion_raycast = create_raycast("occray for " + get_name().str(), get_position());
	add_child(occlusion_raycast);

	// connect signal (used to restart playing if loop is enabled)
	connect("finished", callable_mp(this, &Soundplayer::_on_finished));
}

void Soundplayer::_on_finished()
{
	if(loop)
	{
		soundsource->do_play();
	}
}

void Soundplayer::physics_process(double _delta)
{
	if(debug)
	{
		dump_debug();
		debugsphere->update_label();
	}
}

void Soundplayer::_exit_tree()
{
	TypedArray<Node> children = get_children();
	for(int i = 0; i < children.size(); i++)
	{
		Node* c = Object::cast_to<Node>(children[i]);
		remove_child(c);
	}
	remove_audiobus(audiobus_name);
}

void Soundplayer::do_play()
{
	//if debug: printerr(str(Time.get_ticks_msec()) + ": start playing on " + name + ", stream: " + str(stream))
	play();
}

void Soundplayer::do_stop()
{
	//if debug: printerr(str(Time.get_ticks_msec()) + ": stop playing on " + name + ", stream: " + str(stream))
	stop();
}

void Soundplayer::set_active(double fadetime, Tween::EaseType easing, Tween::TransitionType transition)
{
	//if debug: print("SET ACTIVE: %s (state: %s, fadetime: %s)" % [name, ds[state], fadetime])
	if(state == ss::inactive)
	{
		if(debug)
		{debugsphere->set_visible(true);
		}
		update_effect_params();

		if(fadetime > 0)
		{
			state = ss::fading_to_active;

			if(fade_tween.is_valid())
			{
				fade_tween->kill();
			}
			fade_tween = create_tween();
			fade_tween->stop();
			fade_tween->set_ease(easing);
			fade_tween->set_trans(transition);

			set_audiobus_volume(audiobus_name, proximity_volume, fadetime, fade_tween);
			/* await get_tree().create_timer(fadetime + 0.01)->timeout; */ // no equivalent to await in c++ !
		}

		else
		{
			set_audiobus_volume(audiobus_name, proximity_volume);
		}

		state = ss::active;
	}
}

void Soundplayer::set_inactive(double fadetime, Tween::EaseType easing, Tween::TransitionType transition)
{
	//if debug: print("SET INACTIVE: %s (state: %s, fadetime: %s)" % [name, ds[state], fadetime])
	if(state == active || state == fading_to_active)
	{

		if(fadetime > 0)
		{
			state = fading_to_inactive;

			if(fade_tween.is_valid())
			{
				fade_tween->kill();
			}
			fade_tween = create_tween();
			fade_tween->stop();
			fade_tween->set_ease(easing);
			fade_tween->set_trans(transition);

			set_audiobus_volume(audiobus_name,  - 80, fadetime, fade_tween);
			/* await get_tree().create_timer(fadetime + 0.01)->timeout; */ // no equivalent to await in c++ !
		}

		else
		{
			set_audiobus_volume(audiobus_name,  - 80);
		}

		state = inactive;
		if(debug)
		{debugsphere->set_visible(false);
		}
	}
}

void Soundplayer::update_effect_params()
{
	// update distance vars
	distance_to_soundsource = get_global_position().distance_to(soundsource->get_global_position());
	distance_to_player = get_global_position().distance_to(player_camera->get_global_position());

	// occlusion
	lp_cutoff = calculate_occlusion_lowpass();

	// only calculate reverb effects on reverbers, not soundsource soundplayers.
	// soundsource will set delay on soundsource-soundplayers.
	if(with_reverb_fx)
	{
		// calculate and set delay
		delay_ms = calculate_delay(distance_to_soundsource) + calculate_delay(distance_to_player);

		// roomsize & wetness
		room_size = soundsource->room_size;
		wetness = soundsource->wetness;

		// set volume. further away = louder.
		// (without tuning, in small rooms, reverb is overwhelming, but you can't hear echo far away)
		proximity_volume = calculate_proximity_volume();

		// set proximity bass
		// more bass if closer to the wall, effect begins at 50m to a wall
		// hipass = 0.2
		proximity_bass = calculate_proximity_bass();
	}
}

int Soundplayer::calculate_proximity_volume()
{
	int proximity_reduction = 24;
	double max_volume = reverb_volume_db + get_volume_db();

	// calculate reduction based on the ratio player-to-max_raycast_distance and player-to-soundsource.
	// the closer you are to a wall, the less reverb should be heard. the further away you are, the more it should be audible.
	// the further away the soundsource is, the less reverb should be heard.
	double ratio_player_rc = MIN(1, distance_to_player / max_raycast_distance);
	double ratio_player_ss = MIN(1, distance_to_player / distance_to_soundsource);
	int prox_volume = (proximity_reduction * ratio_player_rc + proximity_reduction * ratio_player_ss) / 2 - proximity_reduction;

	// add soundsource max_volume
	prox_volume = prox_volume + max_volume;

	// limit maximum volume to max_volume
	prox_volume = MIN(prox_volume, max_volume);

	return prox_volume;
}

double Soundplayer::calculate_proximity_bass()
{
	return Math::snapped(MIN(0.2 * distance_to_player / MAX(bass_proximity, 0.001), 0.2), 0.001);
}

int Soundplayer::calculate_occlusion_lowpass()
{
	Variant limited_distance_to_player = CLAMP(distance_to_player, 0, max_raycast_distance);
	occlusion_raycast->set_target_position(get_global_position().direction_to(player_camera->get_global_position()) * max_raycast_distance * 10);

	int _cutoff = 20500;
	occlusion_raycast->force_raycast_update();
	if(occlusion_raycast->is_colliding())
	{
		Vector3 collision_point = occlusion_raycast->get_collision_point();
		double ray_distance = collision_point.distance_to(get_global_position());
		double wall_to_player_ratio = ray_distance / MAX(distance_to_player, 0.001);
		if(ray_distance < distance_to_player)
		{
			_cutoff = Math::snapped(occlusion_lp_cutoff * wall_to_player_ratio, 0.001);
		}
	}

	return _cutoff;
}

void Soundplayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_state", "v"), &Soundplayer::set_state);
	ClassDB::bind_method(D_METHOD("get_state"), &Soundplayer::get_state);
	ClassDB::bind_method(D_METHOD("set_target_position", "v"), &Soundplayer::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &Soundplayer::get_target_position);
	ClassDB::bind_method(D_METHOD("set_delay_ms", "v"), &Soundplayer::set_delay_ms);
	ClassDB::bind_method(D_METHOD("get_delay_ms"), &Soundplayer::get_delay_ms);
	ClassDB::bind_method(D_METHOD("set_room_size", "v"), &Soundplayer::set_room_size);
	ClassDB::bind_method(D_METHOD("get_room_size"), &Soundplayer::get_room_size);
	ClassDB::bind_method(D_METHOD("set_wetness", "v"), &Soundplayer::set_wetness);
	ClassDB::bind_method(D_METHOD("get_wetness"), &Soundplayer::get_wetness);
	ClassDB::bind_method(D_METHOD("set_proximity_volume", "v"), &Soundplayer::set_proximity_volume);
	ClassDB::bind_method(D_METHOD("get_proximity_volume"), &Soundplayer::get_proximity_volume);
	ClassDB::bind_method(D_METHOD("set_proximity_bass", "v"), &Soundplayer::set_proximity_bass);
	ClassDB::bind_method(D_METHOD("get_proximity_bass"), &Soundplayer::get_proximity_bass);
	ClassDB::bind_method(D_METHOD("set_lp_cutoff", "v"), &Soundplayer::set_lp_cutoff);
	ClassDB::bind_method(D_METHOD("get_lp_cutoff"), &Soundplayer::get_lp_cutoff);
	ClassDB::bind_method(D_METHOD("do_play"), &Soundplayer::do_play);
	ClassDB::bind_method(D_METHOD("do_stop"), &Soundplayer::do_stop);
	ClassDB::bind_method(D_METHOD("set_active", "fadetime", "easing", "transition"), &Soundplayer::set_active);
	ClassDB::bind_method(D_METHOD("set_inactive", "fadetime", "easing", "transition"), &Soundplayer::set_inactive);
	ClassDB::bind_method(D_METHOD("update_effect_params"), &Soundplayer::update_effect_params);
	ClassDB::bind_method(D_METHOD("calculate_proximity_volume"), &Soundplayer::calculate_proximity_volume);
	ClassDB::bind_method(D_METHOD("calculate_proximity_bass"), &Soundplayer::calculate_proximity_bass);
	ClassDB::bind_method(D_METHOD("calculate_occlusion_lowpass"), &Soundplayer::calculate_occlusion_lowpass);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(active, "active"), "active", active);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(inactive, "inactive"), "inactive", inactive);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(fading_to_active, "fading_to_active"), "fading_to_active", fading_to_active);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(fading_to_inactive, "fading_to_inactive"), "fading_to_inactive", fading_to_inactive);
}

void Debugsphere::_ready()
{
	MeshInstance3D* meshinstance = memnew(MeshInstance3D);
	SphereMesh* spheremesh = memnew(SphereMesh);
	StandardMaterial3D* mat = memnew(StandardMaterial3D);
	mat->set_albedo(color);
	//mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	//mat.fixed_size = true
	spheremesh->set_material(mat);
	//meshinstance.cast_shadow = false
	spheremesh->set_radius(size / 2);
	spheremesh->set_height(size);
	meshinstance->set_mesh(spheremesh);
	meshinstance->set_visibility_range_end( max_raycast_distance * 1.3);
	meshinstance->set_visibility_range_end_margin(max_raycast_distance / 10.0);
	meshinstance->set_visibility_range_fade_mode( GeometryInstance3D::VisibilityRangeFadeMode::VISIBILITY_RANGE_FADE_SELF);
	add_child(meshinstance);

	// create second sphere without depth-test, displayed when occluded
	MeshInstance3D* occluded_meshinstance = memnew(MeshInstance3D);
	SphereMesh* occluded_spheremesh = memnew(SphereMesh);
	StandardMaterial3D* occluded_mat = memnew(StandardMaterial3D);
	occluded_mat->set_albedo(Color(color, 0.2));
	//occluded_mat.fixed_size = true
	occluded_mat->set_flag(BaseMaterial3D::FLAG_DISABLE_DEPTH_TEST,true);
	occluded_mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
	occluded_spheremesh->set_material(occluded_mat);
	occluded_meshinstance->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
	occluded_spheremesh->set_radius(size / 2);
	occluded_spheremesh->set_height(size);
	occluded_meshinstance->set_mesh(occluded_spheremesh);
	occluded_meshinstance->set_visibility_range_end(max_raycast_distance * 1.3);
	occluded_meshinstance->set_visibility_range_end_margin(max_raycast_distance / 10.0);
	occluded_meshinstance->set_visibility_range_fade_mode(GeometryInstance3D::VISIBILITY_RANGE_FADE_SELF);
	add_child(occluded_meshinstance);

	// create label3d
	label = memnew(Label3D);
	label->set_position(label_offset);
	label->set_billboard_mode(BaseMaterial3D::BILLBOARD_ENABLED);
	label->set_draw_flag(Label3D::FLAG_FIXED_SIZE,true);
	label->set_draw_flag(Label3D::FLAG_DISABLE_DEPTH_TEST,true);
	label->set_pixel_size(0.0005);
	label->set_font_size(50);
	label->set_text("");
	label->set_visibility_range_end(max_raycast_distance * 1.3);
	label->set_visibility_range_end_margin(max_raycast_distance / 10.0);
	label->set_visibility_range_fade_mode(GeometryInstance3D::VISIBILITY_RANGE_FADE_SELF);
	add_child(label);
}

void Debugsphere::update_label()
{
	label->set_text(line1 + "\n" + line2 + "\n" + line3 + "\n" + line4);
}

void Debugsphere::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_label"), &Debugsphere::update_label);

}

void Debugray::_ready()
{
	immediate_mesh.instantiate();
	material.instantiate();
	material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);

	mesh = immediate_mesh;
	set_cast_shadows_setting( GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
}

void Debugray::draw(Vector3 pos1, Vector3 pos2, Color color)
{
	clear();
	immediate_mesh->surface_begin(Mesh::PRIMITIVE_LINES, material);
	immediate_mesh->surface_add_vertex(pos1);
	immediate_mesh->surface_add_vertex(pos2);
	immediate_mesh->surface_end();
	material->set_albedo(color);
	material->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
}

void Debugray::clear()
{
	immediate_mesh->clear_surfaces();
}

void Debugray::_bind_methods() {
	ClassDB::bind_method(D_METHOD("draw", "pos1", "pos2", "color"), &Debugray::draw);
	ClassDB::bind_method(D_METHOD("clear"), &Debugray::clear);

}

void SpatialAudio3D::dump_debug()
{
	// String _s = "";
	// for(Dictionary key : _debug)
	// {
	// 	_s += "%s: %s\n" % Array {/* initializer lists are unsupported */ key, JSON->stringify(_debug[key], "    "),  };
	// }

	// if(_s != "")
	// {
	// 	Node* ad = get_tree()->get_root().find_child("AudioDebug");
	// 	if(ad)
	// 	{
	// 		ad->text = _s;
	// 	}
	// }
}

void SpatialAudio3D::print_r(Variant obj)
{
	// if((bool)dynamic_cast<Object*>(&obj))
	// {
	// 	obj = inst_to_dict(obj);
	// }
	//UtilityFunctions::print(JSON->stringify(obj, "    "));
}

void SpatialAudio3D::set_reverb_enabled(bool value) {
	reverb_enabled = value;
}

bool SpatialAudio3D::get_reverb_enabled() {
	return reverb_enabled;
}

void SpatialAudio3D::set_reverb_volume_db(double value) {
	reverb_volume_db = value;
}

double SpatialAudio3D::get_reverb_volume_db() {
	return reverb_volume_db;
}

void SpatialAudio3D::set_reverb_fadeintime(double value) {
	reverb_fadeintime = value;
}

double SpatialAudio3D::get_reverb_fadeintime() {
	return reverb_fadeintime;
}

void SpatialAudio3D::set_reverb_fadeouttime(double value) {
	reverb_fadeouttime = value;
}

double SpatialAudio3D::get_reverb_fadeouttime() {
	return reverb_fadeouttime;
}

void SpatialAudio3D::set_occlusion_lp_cutoff(int value) {
	occlusion_lp_cutoff = value;
}

int SpatialAudio3D::get_occlusion_lp_cutoff() {
	return occlusion_lp_cutoff;
}

void SpatialAudio3D::set_occlusion_fadetime(double value) {
	occlusion_fadetime = value;
}

double SpatialAudio3D::get_occlusion_fadetime() {
	return occlusion_fadetime;
}

void SpatialAudio3D::set_bass_proximity(int value) {
	bass_proximity = value;
}

int SpatialAudio3D::get_bass_proximity() {
	return bass_proximity;
}

void SpatialAudio3D::set_max_raycast_distance(int value) {
	max_raycast_distance = value;
}

int SpatialAudio3D::get_max_raycast_distance() {
	return max_raycast_distance;
}

void SpatialAudio3D::set_collision_mask(int value) {
	collision_mask = value;
}

int SpatialAudio3D::get_collision_mask() {
	return collision_mask;
}

void SpatialAudio3D::set_roomsize_multiplicator(double value) {
	roomsize_multiplicator = value;
}

double SpatialAudio3D::get_roomsize_multiplicator() {
	return roomsize_multiplicator;
}

void SpatialAudio3D::set_speed_of_sound(int value) {
	speed_of_sound = value;
}

int SpatialAudio3D::get_speed_of_sound() {
	return speed_of_sound;
}

void SpatialAudio3D::set_audiophysics_ticks(int value) {
	audiophysics_ticks = value;
}

int SpatialAudio3D::get_audiophysics_ticks() {
	return audiophysics_ticks;
}

void SpatialAudio3D::set_loop(bool value) {
	loop = value;
}

bool SpatialAudio3D::get_loop() {
	return loop;
}

void SpatialAudio3D::set_shut_up(bool value) {
	shut_up = value;
}

bool SpatialAudio3D::get_shut_up() {
	return shut_up;
}

void SpatialAudio3D::set_debug(bool value) {
	debug = value;
}

bool SpatialAudio3D::get_debug() {
	return debug;
}

void SpatialAudio3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("do_play"), &SpatialAudio3D::do_play);
	ClassDB::bind_method(D_METHOD("do_stop"), &SpatialAudio3D::do_stop);
	ClassDB::bind_method(D_METHOD("do_set_stream", "sound"), &SpatialAudio3D::do_set_stream);
	ClassDB::bind_method(D_METHOD("create_raycast", "name_p", "target_position"), &SpatialAudio3D::create_raycast);
	//ClassDB::bind_method(D_METHOD("create_raycast_sector", "start_angle", "width_factor", "bearing_raycount", "heading_count"), &SpatialAudio3D::create_raycast_sector);
	ClassDB::bind_method(D_METHOD("create_soundplayer", "name_p", "with_reverb_fx"), &SpatialAudio3D::create_soundplayer);
	ClassDB::bind_method(D_METHOD("create_audiobus", "bus_name", "vol_db"), &SpatialAudio3D::create_audiobus);
	ClassDB::bind_method(D_METHOD("remove_audiobus", "bus_name"), &SpatialAudio3D::remove_audiobus);
	ClassDB::bind_method(D_METHOD("set_audiobus_volume", "bus_name", "vol_db", "fadetime", "tweenvar"), &SpatialAudio3D::set_audiobus_volume);
	ClassDB::bind_method(D_METHOD("tweensetvol", "vol", "bus_index"), &SpatialAudio3D::tweensetvol);
	ClassDB::bind_method(D_METHOD("add_audioeffect", "bus_name", "effect_type"), &SpatialAudio3D::add_audioeffect);
	ClassDB::bind_method(D_METHOD("set_audioeffect", "bus_name", "effect_type", "params"), &SpatialAudio3D::set_audioeffect);
	ClassDB::bind_method(D_METHOD("toggle_audioeffect", "bus_name", "effect_type", "enabled"), &SpatialAudio3D::toggle_audioeffect);
	ClassDB::bind_method(D_METHOD("calculate_delay", "distance"), &SpatialAudio3D::calculate_delay);
	ClassDB::bind_method(D_METHOD("dump_debug"), &SpatialAudio3D::dump_debug);
	ClassDB::bind_method(D_METHOD("print_r", "obj"), &SpatialAudio3D::print_r);
	ClassDB::bind_method(D_METHOD("set_reverb_enabled", "value"), &SpatialAudio3D::set_reverb_enabled);
	ClassDB::bind_method(D_METHOD("get_reverb_enabled"), &SpatialAudio3D::get_reverb_enabled);
	ClassDB::bind_method(D_METHOD("set_reverb_volume_db", "value"), &SpatialAudio3D::set_reverb_volume_db);
	ClassDB::bind_method(D_METHOD("get_reverb_volume_db"), &SpatialAudio3D::get_reverb_volume_db);
	ClassDB::bind_method(D_METHOD("set_reverb_fadeintime", "value"), &SpatialAudio3D::set_reverb_fadeintime);
	ClassDB::bind_method(D_METHOD("get_reverb_fadeintime"), &SpatialAudio3D::get_reverb_fadeintime);
	ClassDB::bind_method(D_METHOD("set_reverb_fadeouttime", "value"), &SpatialAudio3D::set_reverb_fadeouttime);
	ClassDB::bind_method(D_METHOD("get_reverb_fadeouttime"), &SpatialAudio3D::get_reverb_fadeouttime);
	ClassDB::bind_method(D_METHOD("set_occlusion_lp_cutoff", "value"), &SpatialAudio3D::set_occlusion_lp_cutoff);
	ClassDB::bind_method(D_METHOD("get_occlusion_lp_cutoff"), &SpatialAudio3D::get_occlusion_lp_cutoff);
	ClassDB::bind_method(D_METHOD("set_occlusion_fadetime", "value"), &SpatialAudio3D::set_occlusion_fadetime);
	ClassDB::bind_method(D_METHOD("get_occlusion_fadetime"), &SpatialAudio3D::get_occlusion_fadetime);
	ClassDB::bind_method(D_METHOD("set_bass_proximity", "value"), &SpatialAudio3D::set_bass_proximity);
	ClassDB::bind_method(D_METHOD("get_bass_proximity"), &SpatialAudio3D::get_bass_proximity);
	ClassDB::bind_method(D_METHOD("set_max_raycast_distance", "value"), &SpatialAudio3D::set_max_raycast_distance);
	ClassDB::bind_method(D_METHOD("get_max_raycast_distance"), &SpatialAudio3D::get_max_raycast_distance);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "value"), &SpatialAudio3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &SpatialAudio3D::get_collision_mask);
	ClassDB::bind_method(D_METHOD("set_roomsize_multiplicator", "value"), &SpatialAudio3D::set_roomsize_multiplicator);
	ClassDB::bind_method(D_METHOD("get_roomsize_multiplicator"), &SpatialAudio3D::get_roomsize_multiplicator);
	ClassDB::bind_method(D_METHOD("set_speed_of_sound", "value"), &SpatialAudio3D::set_speed_of_sound);
	ClassDB::bind_method(D_METHOD("get_speed_of_sound"), &SpatialAudio3D::get_speed_of_sound);
	ClassDB::bind_method(D_METHOD("set_audiophysics_ticks", "value"), &SpatialAudio3D::set_audiophysics_ticks);
	ClassDB::bind_method(D_METHOD("get_audiophysics_ticks"), &SpatialAudio3D::get_audiophysics_ticks);
	ClassDB::bind_method(D_METHOD("set_loop", "value"), &SpatialAudio3D::set_loop);
	ClassDB::bind_method(D_METHOD("get_loop"), &SpatialAudio3D::get_loop);
	ClassDB::bind_method(D_METHOD("set_shut_up", "value"), &SpatialAudio3D::set_shut_up);
	ClassDB::bind_method(D_METHOD("get_shut_up"), &SpatialAudio3D::get_shut_up);
	ClassDB::bind_method(D_METHOD("set_debug", "value"), &SpatialAudio3D::set_debug);
	ClassDB::bind_method(D_METHOD("get_debug"), &SpatialAudio3D::get_debug);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(delay, "delay"), "delay", delay);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(reverb, "reverb"), "reverb", reverb);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(reverb_hipass, "reverb_hipass"), "reverb_hipass", reverb_hipass);
	// ClassDB::bind_integer_constant(get_class_static(), _gde_constant_get_enum_name(lowpass, "lowpass"), "lowpass", lowpass);
	// ClassDB::add_property_category(get_class_static(), "Sound Properties","");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::BOOL, "reverb_enabled"), "set_reverb_enabled", "get_reverb_enabled");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::FLOAT, "reverb_volume_db", PROPERTY_HINT_RANGE, "-81,+80,3.0,suffix:dB"), "set_reverb_volume_db", "get_reverb_volume_db");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::FLOAT, "reverb_fadeintime", PROPERTY_HINT_RANGE, "-1,5,0.05,or_greater,suffix:s"), "set_reverb_fadeintime", "get_reverb_fadeintime");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::FLOAT, "reverb_fadeouttime", PROPERTY_HINT_RANGE, "-1,8,0.05,or_greater,suffix:s"), "set_reverb_fadeouttime", "get_reverb_fadeouttime");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "occlusion_lp_cutoff", PROPERTY_HINT_RANGE, "1,20_000,0.1,suffix:Hz"), "set_occlusion_lp_cutoff", "get_occlusion_lp_cutoff");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::FLOAT, "occlusion_fadetime", PROPERTY_HINT_RANGE, "0,5,0.05,or_greater,suffix:s"), "set_occlusion_fadetime", "get_occlusion_fadetime");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "bass_proximity", PROPERTY_HINT_RANGE, "0,100,0.1,or_greater,suffix:m"), "set_bass_proximity", "get_bass_proximity");
	// ClassDB::add_property_category(get_class_static(), "Physics","");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "max_raycast_distance", PROPERTY_HINT_RANGE, "1,100,1,or_greater,suffix:m"), "set_max_raycast_distance", "get_max_raycast_distance");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "collision_mask", EXPORT_FLAGS_3D_PHYSICS, ""), "set_collision_mask", "get_collision_mask");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::FLOAT, "roomsize_multiplicator"), "set_roomsize_multiplicator", "get_roomsize_multiplicator");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "speed_of_sound", PROPERTY_HINT_RANGE, "1,340,1,or_greater,suffix:m/s"), "set_speed_of_sound", "get_speed_of_sound");
	// ClassDB::add_property_category(get_class_static(), "System","");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::INT, "audiophysics_ticks", PROPERTY_HINT_RANGE, "1,60,0.1,or_greater,suffix:per second"), "set_audiophysics_ticks", "get_audiophysics_ticks");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::BOOL, "loop"), "set_loop", "get_loop");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::BOOL, "shut_up"), "set_shut_up", "get_shut_up");
	// ClassDB::add_property(get_class_static(), PropertyInfo(Variant::BOOL, "debug"), "set_debug", "get_debug");
}

