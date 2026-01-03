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

#import "drivers/apple_embedded/os_apple_embedded.h"
#include "modules/visionos_xr/visionos_xr_interface.h"
#import <CompositorServices/CompositorServices.h>

extern void apple_embedded_finish();

@implementation GDTCompositorServicesRenderer {
	cp_layer_renderer_t _layer_renderer;
}

- (instancetype)initWithLayerRenderer:(cp_layer_renderer_t)layer_renderer {
	self = [super init];
	if (self) {
		_layer_renderer = layer_renderer;
	}
	return self;
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
			// There's no way to recover from the layer renderer being invalidated without opening a new immersive space
			// so exit the app. Unfortunately, pressing the Crown on Apple Vision Pro currently invalidates the layer renderer,
			// so we cannot currently background a Godot immersive game, and come back to it again later.
			NSLog(@"Compositor Services layer invalidated, exiting");
			safeDispatchSyncToMain(^{
				apple_embedded_finish();
				exit(0);
			});
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

@end
