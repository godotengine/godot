/*************************************************************************/
/*  godot_view.mm                                                        */
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

#import "godot_view.h"

#include "core/os/keyboard.h"
#include "core/string/ustring.h"
#import "display_layer.h"
#include "display_server_appletv.h"
#import "godot_view_renderer.h"
#include "os_appletv.h"

@interface GodotView ()

@property(assign, nonatomic) BOOL isActive;

// CADisplayLink available on 3.1+ synchronizes the animation timer & drawing with the refresh rate of the display, only supports animation intervals of 1/60 1/30 & 1/15
@property(strong, nonatomic) CADisplayLink *displayLink;

// An animation timer that, when animation is started, will periodically call -drawView at the given rate.
// Only used if CADisplayLink is not
@property(strong, nonatomic) NSTimer *animationTimer;

@property(strong, nonatomic) CALayer<DisplayLayer> *renderingLayer;

@end

@implementation GodotView

- (CALayer<DisplayLayer> *)initializeRenderingForDriver:(NSString *)driverName {
	if (self.renderingLayer) {
		return self.renderingLayer;
	}

	CALayer<DisplayLayer> *layer;

	if ([driverName isEqualToString:@"vulkan"]) {
		layer = [GodotMetalLayer layer];
	} else if ([driverName isEqualToString:@"opengl_es"]) {
		if (@available(iOS 13, *)) {
			NSLog(@"OpenGL ES is deprecated on iOS 13");
		}
#if defined(TARGET_OS_SIMULATOR) && TARGET_OS_SIMULATOR
		return nil;
#else
		layer = [GodotOpenGLLayer layer];
#endif
	} else {
		return nil;
	}

	layer.frame = self.bounds;
	layer.contentsScale = self.contentScaleFactor;

	[self.layer addSublayer:layer];
	self.renderingLayer = layer;

	[layer initializeDisplayLayer];

	return self.renderingLayer;
}

- (instancetype)initWithCoder:(NSCoder *)coder {
	self = [super initWithCoder:coder];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (instancetype)initWithFrame:(CGRect)frame {
	self = [super initWithFrame:frame];

	if (self) {
		[self godot_commonInit];
	}

	return self;
}

- (void)dealloc {
	[self stopRendering];

	self.renderer = nil;
	self.delegate = nil;

	if (self.renderingLayer) {
		[self.renderingLayer removeFromSuperlayer];
		self.renderingLayer = nil;
	}

	if (self.displayLink) {
		[self.displayLink invalidate];
		self.displayLink = nil;
	}

	if (self.animationTimer) {
		[self.animationTimer invalidate];
		self.animationTimer = nil;
	}
}

- (void)godot_commonInit {
	self.contentScaleFactor = [UIScreen mainScreen].nativeScale;
}

- (void)stopRendering {
	if (!self.isActive) {
		return;
	}

	self.isActive = NO;

	printf("******** stop animation!\n");

	if (self.useCADisplayLink) {
		[self.displayLink invalidate];
		self.displayLink = nil;
	} else {
		[self.animationTimer invalidate];
		self.animationTimer = nil;
	}
}

- (void)startRendering {
	if (self.isActive) {
		return;
	}

	self.isActive = YES;

	printf("start animation!\n");

	if (self.useCADisplayLink) {
		self.displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(drawView)];

		// Approximate frame rate
		// assumes device refreshes at 60 fps
		int displayFPS = (NSInteger)(1.0 / self.renderingInterval);

		self.displayLink.preferredFramesPerSecond = displayFPS;

		// Setup DisplayLink in main thread
		[self.displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
	} else {
		self.animationTimer = [NSTimer scheduledTimerWithTimeInterval:self.renderingInterval target:self selector:@selector(drawView) userInfo:nil repeats:YES];
	}
}

- (void)drawView {
	if (!self.isActive) {
		printf("draw view not active!\n");
		return;
	}

	if (self.useCADisplayLink) {
		// Pause the CADisplayLink to avoid recursion
		[self.displayLink setPaused:YES];

		// Process all input events
		while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, TRUE) == kCFRunLoopRunHandledSource)
			;

		// We are good to go, resume the CADisplayLink
		[self.displayLink setPaused:NO];
	}

	[self.renderingLayer renderDisplayLayer];

	if (!self.renderer) {
		return;
	}

	if ([self.renderer setupView:self]) {
		return;
	}

	if (self.delegate) {
		BOOL delegateFinishedSetup = [self.delegate godotViewFinishedSetup:self];

		if (!delegateFinishedSetup) {
			return;
		}
	}

	[self.renderer renderOnView:self];
}

- (BOOL)canRender {
	if (self.useCADisplayLink) {
		return self.displayLink != nil;
	} else {
		return self.animationTimer != nil;
	}
}

- (void)setRenderingInterval:(NSTimeInterval)renderingInterval {
	_renderingInterval = renderingInterval;

	if (self.canRender) {
		[self stopRendering];
		[self startRendering];
	}
}

- (void)layoutSubviews {
	if (self.renderingLayer) {
		self.renderingLayer.frame = self.bounds;
		[self.renderingLayer layoutDisplayLayer];

		if (DisplayServerAppleTV::get_singleton()) {
			DisplayServerAppleTV::get_singleton()->resize_window(self.bounds.size);
		}
	}

	[super layoutSubviews];
}

// MARK: - Input

// MARK: Menu Button

- (void)pressesBegan:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	NSArray *tlist = [event.allPresses allObjects];

	for (unsigned int i = 0; i < [tlist count]; i++) {
		UIPress *press = [tlist objectAtIndex:i];
		switch (press.type) {
			case UIPressTypeMenu: {
				int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");

				Input::get_singleton()->joy_button(joy_id, JOY_BUTTON_START, true);
			} break;
			default:
				[super pressesBegan:presses withEvent:event];
				break;
		}
	}
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	NSArray *tlist = [event.allPresses allObjects];

	for (unsigned int i = 0; i < [tlist count]; i++) {
		UIPress *press = [tlist objectAtIndex:i];
		switch (press.type) {
			case UIPressTypeMenu: {
				int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");

				Input::get_singleton()->joy_button(joy_id, JOY_BUTTON_START, false);
			} break;
			default:
				[super pressesEnded:presses withEvent:event];
				break;
		}
	}
}

- (void)pressesCancelled:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	NSArray *tlist = [event.allPresses allObjects];

	for (unsigned int i = 0; i < [tlist count]; i++) {
		UIPress *press = [tlist objectAtIndex:i];
		switch (press.type) {
			case UIPressTypeMenu: {
				int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");

				Input::get_singleton()->joy_button(joy_id, JOY_BUTTON_START, false);
			} break;
			default:
				[super pressesCancelled:presses withEvent:event];
				break;
		}
	}
}

@end
