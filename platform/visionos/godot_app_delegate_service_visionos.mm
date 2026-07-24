/**************************************************************************/
/*  godot_app_delegate_visionos.mm                                        */
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

#import "godot_app_delegate_service_visionos.h"

static GDTRenderMode _renderMode = GDTRenderModeWindowed;
static __weak cp_layer_renderer_t _layerRenderer = nil;
static __strong cp_layer_renderer_capabilities_t _layerRendererCapabilities = nil;

@implementation GDTAppDelegateServiceVisionOS

+ (GDTRenderMode)renderMode {
	return _renderMode;
}

+ (void)setRenderMode:(GDTRenderMode)renderMode {
	_renderMode = renderMode;
}

+ (cp_layer_renderer_t)layerRenderer {
	if (_renderMode != GDTRenderModeCompositorServices) {
		NSLog(@"GDTAppDelegate error, layerRenderer only supported in Compositor Services mode");
		return nil;
	}
	return _layerRenderer;
}

+ (void)setLayerRenderer:(cp_layer_renderer_t)layerRenderer {
	if (_renderMode != GDTRenderModeCompositorServices) {
		NSLog(@"GDTAppDelegate error, layerRenderer only supported in Compositor Services mode");
		return;
	}
	_layerRenderer = layerRenderer;
}

+ (cp_layer_renderer_capabilities_t)layerRendererCapabilities {
	if (_renderMode != GDTRenderModeCompositorServices) {
		NSLog(@"GDTAppDelegate error, layerRenderer only supported in Compositor Services mode");
		return nil;
	}
	return _layerRendererCapabilities;
}

+ (void)setLayerRendererCapabilities:(cp_layer_renderer_capabilities_t)layerRendererCapabilities {
	if (_renderMode != GDTRenderModeCompositorServices) {
		NSLog(@"GDTAppDelegate error, layerRenderer only supported in Compositor Services mode");
		return;
	}
	_layerRendererCapabilities = layerRendererCapabilities;
}

@end
