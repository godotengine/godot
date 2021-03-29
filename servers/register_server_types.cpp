/*************************************************************************/
/*  register_server_types.cpp                                            */
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

#include "register_server_types.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"

#include "audio/audio_effect.h"
#include "audio/audio_stream.h"
#include "audio/effects/audio_effect_amplify.h"
#include "audio/effects/audio_effect_capture.h"
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
#include "audio/effects/audio_effect_record.h"
#include "audio/effects/audio_effect_reverb.h"
#include "audio/effects/audio_effect_spectrum_analyzer.h"
#include "audio/effects/audio_effect_stereo_enhance.h"
#include "audio/effects/audio_stream_generator.h"
#include "audio_server.h"
#include "camera/camera_feed.h"
#include "camera_server.h"
#include "display_server.h"
#include "navigation_server_2d.h"
#include "navigation_server_3d.h"
#include "physics_2d/physics_server_2d_sw.h"
#include "physics_2d/physics_server_2d_wrap_mt.h"
#include "physics_3d/physics_server_3d_sw.h"
#include "physics_3d/physics_server_3d_wrap_mt.h"
#include "physics_server_2d.h"
#include "physics_server_3d.h"
#include "rendering/renderer_compositor.h"
#include "rendering/rendering_device.h"
#include "rendering/rendering_device_binds.h"
#include "rendering_server.h"
#include "servers/rendering/shader_types.h"
#include "text_server.h"
#include "xr/xr_interface.h"
#include "xr/xr_positional_tracker.h"
#include "xr_server.h"

ShaderTypes *shader_types = nullptr;

PhysicsServer3D *_createGodotPhysics3DCallback() {
	bool using_threads = GLOBAL_GET("physics/3d/run_on_thread");

	PhysicsServer3D *physics_server = memnew(PhysicsServer3DSW(using_threads));

	return memnew(PhysicsServer3DWrapMT(physics_server, using_threads));
}

PhysicsServer2D *_createGodotPhysics2DCallback() {
	bool using_threads = GLOBAL_GET("physics/2d/run_on_thread");

	PhysicsServer2D *physics_server = memnew(PhysicsServer2DSW(using_threads));

	return memnew(PhysicsServer2DWrapMT(physics_server, using_threads));
}

static bool has_server_feature_callback(const String &p_feature) {
	if (RenderingServer::get_singleton()) {
		if (RenderingServer::get_singleton()->has_os_feature(p_feature)) {
			return true;
		}
	}

	return false;
}

void preregister_server_types() {
	shader_types = memnew(ShaderTypes);

	GLOBAL_DEF("internationalization/rendering/text_driver", "");
	String text_driver_options;
	for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
		if (i > 0) {
			text_driver_options += ",";
		}
		text_driver_options += TextServerManager::get_interface_name(i);
	}
	ProjectSettings::get_singleton()->set_custom_property_info("internationalization/rendering/text_driver", PropertyInfo(Variant::STRING, "internationalization/rendering/text_driver", PROPERTY_HINT_ENUM, text_driver_options));
}

void register_server_types() {
	OS::get_singleton()->set_has_server_feature_callback(has_server_feature_callback);

	ClassDB::register_virtual_class<DisplayServer>();
	ClassDB::register_virtual_class<RenderingServer>();
	ClassDB::register_class<AudioServer>();

	ClassDB::register_class<TextServerManager>();
	ClassDB::register_virtual_class<TextServer>();
	TextServer::initialize_hex_code_box_fonts();

	ClassDB::register_virtual_class<PhysicsServer2D>();
	ClassDB::register_virtual_class<PhysicsServer3D>();
	ClassDB::register_virtual_class<NavigationServer2D>();
	ClassDB::register_virtual_class<NavigationServer3D>();
	ClassDB::register_class<XRServer>();
	ClassDB::register_class<CameraServer>();

	ClassDB::register_virtual_class<RenderingDevice>();

	ClassDB::register_virtual_class<XRInterface>();
	ClassDB::register_class<XRPositionalTracker>();

	ClassDB::register_virtual_class<AudioStream>();
	ClassDB::register_virtual_class<AudioStreamPlayback>();
	ClassDB::register_virtual_class<AudioStreamPlaybackResampled>();
	ClassDB::register_class<AudioStreamMicrophone>();
	ClassDB::register_class<AudioStreamRandomPitch>();
	ClassDB::register_virtual_class<AudioEffect>();
	ClassDB::register_virtual_class<AudioEffectInstance>();
	ClassDB::register_class<AudioEffectEQ>();
	ClassDB::register_class<AudioEffectFilter>();
	ClassDB::register_class<AudioBusLayout>();

	ClassDB::register_class<AudioStreamGenerator>();
	ClassDB::register_virtual_class<AudioStreamGeneratorPlayback>();

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

		ClassDB::register_class<AudioEffectRecord>();
		ClassDB::register_class<AudioEffectSpectrumAnalyzer>();
		ClassDB::register_virtual_class<AudioEffectSpectrumAnalyzerInstance>();

		ClassDB::register_class<AudioEffectCapture>();
	}

	ClassDB::register_virtual_class<RenderingDevice>();
	ClassDB::register_class<RDTextureFormat>();
	ClassDB::register_class<RDTextureView>();
	ClassDB::register_class<RDAttachmentFormat>();
	ClassDB::register_class<RDSamplerState>();
	ClassDB::register_class<RDVertexAttribute>();
	ClassDB::register_class<RDUniform>();
	ClassDB::register_class<RDPipelineRasterizationState>();
	ClassDB::register_class<RDPipelineMultisampleState>();
	ClassDB::register_class<RDPipelineDepthStencilState>();
	ClassDB::register_class<RDPipelineColorBlendStateAttachment>();
	ClassDB::register_class<RDPipelineColorBlendState>();
	ClassDB::register_class<RDShaderSource>();
	ClassDB::register_class<RDShaderBytecode>();
	ClassDB::register_class<RDShaderFile>();

	ClassDB::register_class<CameraFeed>();

	ClassDB::register_virtual_class<PhysicsDirectBodyState2D>();
	ClassDB::register_virtual_class<PhysicsDirectSpaceState2D>();
	ClassDB::register_virtual_class<PhysicsShapeQueryResult2D>();
	ClassDB::register_class<PhysicsTestMotionResult2D>();
	ClassDB::register_class<PhysicsShapeQueryParameters2D>();

	ClassDB::register_class<PhysicsShapeQueryParameters3D>();
	ClassDB::register_virtual_class<PhysicsDirectBodyState3D>();
	ClassDB::register_virtual_class<PhysicsDirectSpaceState3D>();
	ClassDB::register_virtual_class<PhysicsShapeQueryResult3D>();

	// Physics 2D
	GLOBAL_DEF(PhysicsServer2DManager::setting_property_name, "DEFAULT");
	ProjectSettings::get_singleton()->set_custom_property_info(PhysicsServer2DManager::setting_property_name, PropertyInfo(Variant::STRING, PhysicsServer2DManager::setting_property_name, PROPERTY_HINT_ENUM, "DEFAULT"));

	PhysicsServer2DManager::register_server("GodotPhysics2D", &_createGodotPhysics2DCallback);
	PhysicsServer2DManager::set_default_server("GodotPhysics2D");

	// Physics 3D
	GLOBAL_DEF(PhysicsServer3DManager::setting_property_name, "DEFAULT");
	ProjectSettings::get_singleton()->set_custom_property_info(PhysicsServer3DManager::setting_property_name, PropertyInfo(Variant::STRING, PhysicsServer3DManager::setting_property_name, PROPERTY_HINT_ENUM, "DEFAULT"));

	PhysicsServer3DManager::register_server("GodotPhysics3D", &_createGodotPhysics3DCallback);
	PhysicsServer3DManager::set_default_server("GodotPhysics3D");
}

void unregister_server_types() {
	memdelete(shader_types);
	TextServer::finish_hex_code_box_fonts();
}

void register_server_singletons() {
	Engine::get_singleton()->add_singleton(Engine::Singleton("DisplayServer", DisplayServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("RenderingServer", RenderingServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("AudioServer", AudioServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("PhysicsServer2D", PhysicsServer2D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("PhysicsServer3D", PhysicsServer3D::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("NavigationServer2D", NavigationServer2D::get_singleton_mut()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("NavigationServer3D", NavigationServer3D::get_singleton_mut()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("TextServerManager", TextServerManager::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("XRServer", XRServer::get_singleton()));
	Engine::get_singleton()->add_singleton(Engine::Singleton("CameraServer", CameraServer::get_singleton()));
}
