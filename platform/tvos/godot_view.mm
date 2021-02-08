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

#include "core/project_settings.h"
#include "os_appletv.h"
#include "servers/audio_server.h"

#import <OpenGLES/EAGLDrawable.h>
#import <QuartzCore/QuartzCore.h>

#import "display_layer.h"
#import "godot_view_renderer.h"

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

// Implement this to override the default layer class (which is [CALayer class]).
// We do this so that our view will be backed by a layer that is capable of OpenGL ES rendering.
- (CALayer<DisplayLayer> *)initializeRendering {
	if (self.renderingLayer) {
		return self.renderingLayer;
	}

	CALayer<DisplayLayer> *layer = [GodotOpenGLLayer layer];

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

// Stop animating and release resources when they are no longer needed.
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

// Updates the OpenGL view when the timer fires
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

	[self.renderingLayer startRenderDisplayLayer];

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

	[self.renderingLayer stopRenderDisplayLayer];
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

// If our view is resized, we'll be asked to layout subviews.
// This is the perfect opportunity to also update the framebuffer so that it is
// the same size as our display area.

- (void)layoutSubviews {
	if (self.renderingLayer) {
		self.renderingLayer.frame = self.bounds;
		[self.renderingLayer layoutDisplayLayer];
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

				OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, true);
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

				OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, false);
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

				OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, false);
			} break;
			default:
				[super pressesCancelled:presses withEvent:event];
				break;
		}
	}
}

@end
