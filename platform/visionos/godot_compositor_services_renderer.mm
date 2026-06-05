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

#include "display_server_visionos.h"
#include "input_event_spatial.h"

#include "drivers/apple_embedded/os_apple_embedded.h"
#include "servers/rendering/rendering_server_default.h"

#include "modules/visionos_xr/visionos_xr_interface.h"

#import <CompositorServices/CompositorServices.h>

extern void apple_embedded_finish();

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

// Skip the boot splash in CompositorServices (immersive XR) mode.
// There is no visible window to display it on.
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

- (void)spatialEventWithIndex:(NSInteger)index
					 isActive:(BOOL)isActive
			 hasBeenCancelled:(BOOL)hasBeenCancelled
				 hasChirality:(BOOL)hasChirality
				   isLeftHand:(BOOL)isLeftHand
				 selectionRay:(SPRay3D)selectionRay
			  inputDevicePose:(SPPose3D)inputDevicePose {
	Vector3 selectionRayOrigin(selectionRay.origin.x, selectionRay.origin.y, selectionRay.origin.z);
	Vector3 selectionRayDirection(selectionRay.direction.x, selectionRay.direction.y, selectionRay.direction.z);
	InputEventSpatial::Chirality chirality = isLeftHand ? InputEventSpatial::CHIRALITY_LEFT : InputEventSpatial::CHIRALITY_RIGHT;
	InputEventSpatial::Phase phase = isActive ? InputEventSpatial::PHASE_ACTIVE : (hasBeenCancelled ? InputEventSpatial::PHASE_CANCELLED : InputEventSpatial::PHASE_ENDED);
	simd_double4 deviceInputRotationQuat = inputDevicePose.rotation.quaternion.vector;
	Vector3 inputDevicePosition(inputDevicePose.position.x, inputDevicePose.position.y, inputDevicePose.position.z);
	Quaternion inputDeviceRotation(deviceInputRotationQuat[0], deviceInputRotationQuat[1], deviceInputRotationQuat[2], deviceInputRotationQuat[3]);
	DisplayServerVisionOS::get_singleton()->spatial_event(index, phase, hasChirality, chirality, selectionRayOrigin, selectionRayDirection, inputDevicePosition, inputDeviceRotation);
}

@end
