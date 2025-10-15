/**************************************************************************/
/*  godot_view.mm                                                         */
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

#include "godot_view_visionos.h"

#include "display_layer_visionos.h"

#include "core/error/error_macros.h"

#import <GameController/GameController.h>

@interface GDTViewVisionOS ()

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wobjc-property-synthesis")
@property(strong, nonatomic) CALayer<GDTDisplayLayer> *renderingLayer;
GODOT_CLANG_WARNING_POP

@end

@implementation GDTViewVisionOS

- (void)godot_commonInit {
	[super godot_commonInit];

	// Enable GamePad handler
	GCEventInteraction *gamepadInteraction = [[GCEventInteraction alloc] init];
	gamepadInteraction.handledEventTypes = GCUIEventTypeGamepad;
	[self addInteraction:gamepadInteraction];
}

- (CALayer<GDTDisplayLayer> *)initializeRenderingForDriver:(NSString *)driverName {
	if (self.renderingLayer) {
		return self.renderingLayer;
	}

	CALayer<GDTDisplayLayer> *layer = [GDTMetalLayer layer];

	layer.frame = self.bounds;
	layer.contentsScale = self.contentScaleFactor;

	[self.layer addSublayer:layer];
	self.renderingLayer = layer;

	[layer initializeDisplayLayer];

	return self.renderingLayer;
}

@end

GDTView *GDTViewCreate() {
	GDTViewVisionOS *view = [GDTViewVisionOS new];
	view.preferredFrameRate = 90;
	return view;
}
