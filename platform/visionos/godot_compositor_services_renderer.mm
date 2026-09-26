/**************************************************************************/
/*  godot_compositor_services_renderer.mm                                 */
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

#import "godot_compositor_services_renderer.h"

#import "godot_swift_module-Swift.gen.h"

#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/vector.h"
#import "drivers/apple_embedded/os_apple_embedded.h"

#include "modules/modules_enabled.gen.h"
#if defined(MODULE_VISIONOS_XR_ENABLED)
#include "modules/visionos_xr/visionos_xr_interface.h"
#endif

#import <CompositorServices/CompositorServices.h>

extern void apple_embedded_finish();

#if defined(MODULE_VISIONOS_XR_ENABLED)

namespace {

inline Vector3 convert(simd_double3 p_v) {
	return Vector3(p_v.x, p_v.y, p_v.z);
}

inline Vector3 xyz(simd_double4 p_v) {
	return Vector3(p_v.x, p_v.y, p_v.z);
}

inline Transform3D convert(simd_double4x4 p_v) {
	return Transform3D(xyz(p_v.columns[0]), xyz(p_v.columns[1]), xyz(p_v.columns[2]), xyz(p_v.columns[3]));
}

} // namespace

@implementation GDTCompositorServicesRenderer {
	cp_layer_renderer_t _layer_renderer;
	cp_layer_renderer_capabilities_t _layer_renderer_capabilities;
}

- (instancetype)initWithLayerRenderer:(cp_layer_renderer_t)layer_renderer
						 capabilities:(cp_layer_renderer_capabilities_t)capabilities {
	self = [super init];
	if (self) {
		_layer_renderer = layer_renderer;
		_layer_renderer_capabilities = capabilities;
	}
	return self;
}

// On visionOS Compositor Services mode (used by the visionOS XR module),
// there's no way to easily show the boot logo on a simple quad.
- (void)setUpProjectDataShowingBootLogo:(BOOL)p_show_boot_logo {
	[super setUpProjectDataShowingBootLogo:NO];
}

- (void)updateXRInterface {
	Ref<VisionOSXRInterface> visionos_xr_interface = VisionOSXRInterface::find_interface();
	if (visionos_xr_interface.is_valid()) {
		visionos_xr_interface->update_layer_renderer(_layer_renderer, _layer_renderer_capabilities);
	}
}

- (void)startRenderLoop {
	Ref<VisionOSXRInterface> visionos_xr_interface = VisionOSXRInterface::find_interface();
	cp_layer_renderer_state previous_state = cp_layer_renderer_state_running;
	if (visionos_xr_interface.is_valid()) {
		visionos_xr_interface->emit_signal_enum(VisionOSXRInterface::VISIONOS_XR_SIGNAL_SESSION_STARTED);
	}
	while (true) {
		cp_layer_renderer_state state = cp_layer_renderer_get_state(_layer_renderer);
		if (state == cp_layer_renderer_state_invalidated) {
			if (visionos_xr_interface.is_valid()) {
				visionos_xr_interface->emit_signal_enum(VisionOSXRInterface::VISIONOS_XR_SIGNAL_SESSION_INVALIDATED);
			}
			// Exit render loop and wait for a new layer renderer
			return;
		} else if (state == cp_layer_renderer_state_paused) {
			if (previous_state == cp_layer_renderer_state_running && visionos_xr_interface.is_valid()) {
				visionos_xr_interface->emit_signal_enum(VisionOSXRInterface::VISIONOS_XR_SIGNAL_SESSION_PAUSED);
			}
			previous_state = state;
			cp_layer_renderer_wait_until_running(_layer_renderer);
			continue;
		} else {
			@autoreleasepool {
				if (previous_state == cp_layer_renderer_state_paused && visionos_xr_interface.is_valid()) {
					visionos_xr_interface->emit_signal_enum(VisionOSXRInterface::VISIONOS_XR_SIGNAL_SESSION_RESUMED);
				}
				[self renderFrame];
			}
		}
	}
}

- (void)renderFrame {
	safeDispatchSyncToMain(^{
		if (!OS_AppleEmbedded::get_singleton()) {
			return;
		}
		// Check state again after possible thread hop
		cp_layer_renderer_state state = cp_layer_renderer_get_state(_layer_renderer);
		if (state != cp_layer_renderer_state_running) {
			return;
		}
		OS_AppleEmbedded::get_singleton()->iterate();
	});
}

- (void)worldRecentered {
	Ref<VisionOSXRInterface> visionos_xr_interface = VisionOSXRInterface::find_interface();
	if (visionos_xr_interface.is_valid()) {
		visionos_xr_interface->emit_signal_enum(VisionOSXRInterface::VISIONOS_XR_SIGNAL_POSE_RECENTERED);
	}
}

- (void)onSpatialEvent:(SpatialEventObjC *)p_event {
	Ref<VisionOSXRInterface> visionos_xr_interface = VisionOSXRInterface::find_interface();
	if (visionos_xr_interface.is_valid()) {
		VisionOSSpatialEvent event;

		// Convert from the ObjC/Swift type to the C++ type.
		{
			// Ray
			event.has_ray = p_event.hasRay;
			if (p_event.hasRay) {
				event.ray.set_look_at(convert(p_event.rayOrigin), convert(p_event.rayOrigin + p_event.rayDirection));
			}
			// Hand
			static_assert((int)VisionOSSpatialEvent::Chirality::right == (int)ChiralityRight);
			event.chirality = (VisionOSSpatialEvent::Chirality)p_event.chirality;
			event.hand_pose = convert(p_event.handPose);
			// Phase
			static_assert((int)VisionOSSpatialEvent::Phase::ended == (int)PhaseEnded);
			event.phase = (VisionOSSpatialEvent::Phase)p_event.phase;
		}

		// Send the event to the VisionOSXRInterface.
		visionos_xr_interface->on_spatial_event(event);
	}
}

@end

#else

@implementation GDTCompositorServicesRenderer

- (instancetype)initWithLayerRenderer:(cp_layer_renderer_t)layer_renderer
						 capabilities:(cp_layer_renderer_capabilities_t)capabilities {
	self = [super init];
	return self;
}

- (void)updateXRInterface {
}

- (void)startRenderLoop {
}

- (void)renderFrame {
}

- (void)worldRecentered {
}

@end

#endif
