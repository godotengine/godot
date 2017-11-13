/*************************************************************************/
/*  register_server_types.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "register_server_types.h"
#include "engine.h"
#include "project_settings.h"

#include "arvr/arvr_interface.h"
#include "arvr/arvr_positional_tracker.h"
#include "arvr_server.h"
#include "audio/audio_effect.h"
#include "audio/audio_stream.h"
#include "audio/effects/audio_effect_amplify.h"
#include "audio/effects/audio_effect_chorus.h"
#include "audio/effects/audio_effect_compressor.h"
#include "audio/effects/audio_effect_delay.h"
#include "audio/effects/audio_effect_distortion.h"
#include "audio/effects/audio_effect_eq.h"
#include "audio/effects/audio_effect_filter.h"
#include "audio/effects/audio_effect_limiter.h"
#include "audio/effects/audio_effect_panner.h"
#include "audio/effects/audio_effect_phaser.h"
#include "audio/effects/audio_effect_pitch_shift.h"
#include "audio/effects/audio_effect_reverb.h"
#include "audio/effects/audio_effect_stereo_enhance.h"
#include "audio_server.h"
#include "physics/physics_server_sw.h"
#include "physics_2d/physics_2d_server_sw.h"
#include "physics_2d/physics_2d_server_wrap_mt.h"
#include "physics_2d_server.h"
#include "physics_server.h"
#include "script_debugger_remote.h"
#include "visual/shader_types.h"
#include "visual_server.h"

static void _debugger_get_resource_usage(List<ScriptDebuggerRemote::ResourceUsage> *r_usage) {

	List<VS::TextureInfo> tinfo;
	VS::get_singleton()->texture_debug_usage(&tinfo);

	for (List<VS::TextureInfo>::Element *E = tinfo.front(); E; E = E->next()) {

		ScriptDebuggerRemote::ResourceUsage usage;
		usage.path = E->get().path;
		usage.vram = E->get().bytes;
		usage.id = E->get().texture;
		usage.type = "Texture";
		usage.format = itos(E->get().size.width) + "x" + itos(E->get().size.height) + " " + Image::get_format_name(E->get().format);
		r_usage->push_back(usage);
	}
}

ShaderTypes *shader_types = NULL;

PhysicsServer *_createGodotPhysicsCallback() {
	return memnew(PhysicsServerSW);
}

Physics2DServer *_createGodotPhysics2DCallback() {
	return Physics2DServerWrapMT::init_server<Physics2DServerSW>();
}

void register_server_types() {

	ClassDB::register_virtual_class<VisualServer>();
	ClassDB::register_class<AudioServer>();
	ClassDB::register_virtual_class<PhysicsServer>();
	ClassDB::register_virtual_class<Physics2DServer>();
	ClassDB::register_class<ARVRServer>();

	shader_types = memnew(ShaderTypes);

	ClassDB::register_virtual_class<ARVRInterface>();
	ClassDB::register_class<ARVRPositionalTracker>();

	ClassDB::register_virtual_class<AudioStream>();
	ClassDB::register_virtual_class<AudioStreamPlayback>();
	ClassDB::register_class<AudioStreamRandomPitch>();
	ClassDB::register_virtual_class<AudioEffect>();
	ClassDB::register_class<AudioEffectEQ>();
	ClassDB::register_class<AudioEffectFilter>();
	ClassDB::register_class<AudioBusLayout>();

	{
		//audio effects
		ClassDB::register_class<AudioEffectAmplify>();

		ClassDB::register_class<AudioEffectReverb>();

		ClassDB::register_class<AudioEffectLowPassFilter>();
		ClassDB::register_class<AudioEffectHighPassFilter>();
		ClassDB::register_class<AudioEffectBandPassFilter>();
		ClassDB::register_class<AudioEffectNotchFilter>();
		ClassDB::register_class<AudioEffectBandLimitFilter>();
		ClassDB::register_class<AudioEffectLowShelfFilter>();
		ClassDB::register_class<AudioEffectHighShelfFilter>();

		ClassDB::register_class<AudioEffectEQ6>();
		ClassDB::register_class<AudioEffectEQ10>();
		ClassDB::register_class<AudioEffectEQ21>();

		ClassDB::register_class<AudioEffectDistortion>();

		ClassDB::register_class<AudioEffectStereoEnhance>();

		ClassDB::register_class<AudioEffectPanner>();
		ClassDB::register_class<AudioEffectChorus>();
		ClassDB::register_class<AudioEffectDelay>();
		ClassDB::register_class<AudioEffectCompressor>();
		ClassDB::register_class<AudioEffectLimiter>();
		ClassDB::register_class<AudioEffectPitchShift>();
		ClassDB::register_class<AudioEffectPhaser>();
	}

	ClassDB::register_virtual_class<Physics2DDirectBodyState>();
	ClassDB::register_virtual_class<Physics2DDirectSpaceState>();
	ClassDB::register_virtual_class<Physics2DShapeQueryResult>();
	ClassDB::register_class<Physics2DTestMotionResult>();
	ClassDB::register_class<Physics2DShapeQueryParameters>();

	ClassDB::register_class<PhysicsShapeQueryParameters>();
	ClassDB::register_virtual_class<PhysicsDirectBodyState>();
	ClassDB::register_virtual_class<PhysicsDirectSpaceState>();
	ClassDB::register_virtual_class<PhysicsShapeQueryResult>();

	ScriptDebuggerRemote::resource_usage_func = _debugger_get_resource_usage;

	// Physics 2D
	GLOBAL_DEF(Physics2DServerManager::setting_property_name, "DEFAULT");
	ProjectSettings::get_singleton()->set_custom_property_info(Physics2DServerManager::setting_property_name, PropertyInfo(Variant::STRING, Physics2DServerManager::setting_property_name, PROPERTY_HINT_ENUM, "DEFAULT"));

	Physics2DServerManager::register_server("GodotPhysics", &_createGodotPhysics2DCallback);
	Physics2DServerManager::set_default_server("GodotPhysics");

	// Physics 3D
	GLOBAL_DEF(PhysicsServerManager::setting_property_name, "DEFAULT");
	ProjectSettings::get_singleton()->set_custom_property_info(PhysicsServerManager::setting_property_name, PropertyInfo(Variant::STRING, PhysicsServerManager::setting_property_name, PROPERTY_HINT_ENUM, "DEFAULT"));

	PhysicsServerManager::register_server("GodotPhysics", &_createGodotPhysicsCallback);
	PhysicsServerManager::set_default_server("GodotPhysics");
}

void unregister_server_types() {

	memdelete(shader_types);
}

void register_server_singletons() {
	Engine::get_singleton()->add_singleton(Engine::Singleton("VisualServer", VisualServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("AudioServer", AudioServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("PhysicsServer", PhysicsServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("Physics2DServer", Physics2DServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("ARVRServer", ARVRServer::get_singleton()));
}
