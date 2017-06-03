/*************************************************************************/
/*  register_server_types.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "global_config.h"

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

void register_server_types() {

	GLOBAL_DEF("memory/multithread/thread_rid_pool_prealloc", 20);

	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("VisualServer", VisualServer::get_singleton()));
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("AudioServer", AudioServer::get_singleton()));
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("PhysicsServer", PhysicsServer::get_singleton()));
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("Physics2DServer", Physics2DServer::get_singleton()));

	shader_types = memnew(ShaderTypes);

	ClassDB::register_virtual_class<AudioStream>();
	ClassDB::register_virtual_class<AudioStreamPlayback>();
	ClassDB::register_virtual_class<AudioEffect>();
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
}

void unregister_server_types() {

	memdelete(shader_types);
}
