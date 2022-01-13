/*************************************************************************/
/*  godot_view.mm                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "os_iphone.h"
#include "servers/audio_server.h"

#import <OpenGLES/EAGLDrawable.h>
#import <QuartzCore/QuartzCore.h>

#import "display_layer.h"
#import "godot_view_gesture_recognizer.h"
#import "godot_view_renderer.h"

#import <CoreMotion/CoreMotion.h>

static const int max_touches = 8;

@interface GodotView () {
	UITouch *godot_touches[max_touches];
}

@property(assign, nonatomic) BOOL isActive;

// CADisplayLink available on 3.1+ synchronizes the animation timer & drawing with the refresh rate of the display, only supports animation intervals of 1/60 1/30 & 1/15
@property(strong, nonatomic) CADisplayLink *displayLink;

// An animation timer that, when animation is started, will periodically call -drawView at the given rate.
// Only used if CADisplayLink is not
@property(strong, nonatomic) NSTimer *animationTimer;

@property(strong, nonatomic) CALayer<DisplayLayer> *renderingLayer;

@property(strong, nonatomic) CMMotionManager *motionManager;

@property(strong, nonatomic) GodotViewGestureRecognizer *delayGestureRecognizer;

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

	if (self.motionManager) {
		[self.motionManager stopDeviceMotionUpdates];
		self.motionManager = nil;
	}

	if (self.displayLink) {
		[self.displayLink invalidate];
		self.displayLink = nil;
	}

	if (self.animationTimer) {
		[self.animationTimer invalidate];
		self.animationTimer = nil;
	}

	if (self.delayGestureRecognizer) {
		self.delayGestureRecognizer = nil;
	}
}

- (void)godot_commonInit {
	self.contentScaleFactor = [UIScreen mainScreen].nativeScale;

	[self initTouches];

	// Configure and start accelerometer
	if (!self.motionManager) {
		self.motionManager = [[CMMotionManager alloc] init];
		if (self.motionManager.deviceMotionAvailable) {
			self.motionManager.deviceMotionUpdateInterval = 1.0 / 70.0;
			[self.motionManager startDeviceMotionUpdatesUsingReferenceFrame:CMAttitudeReferenceFrameXMagneticNorthZVertical];
		} else {
			self.motionManager = nil;
		}
	}

	// Initialize delay gesture recognizer
	GodotViewGestureRecognizer *gestureRecognizer = [[GodotViewGestureRecognizer alloc] init];
	self.delayGestureRecognizer = gestureRecognizer;
	[self addGestureRecognizer:self.delayGestureRecognizer];
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

	[self clearTouches];
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

	[self handleMotion];
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

// MARK: Touches

- (void)initTouches {
	for (int i = 0; i < max_touches; i++) {
		godot_touches[i] = NULL;
	}
}

- (int)getTouchIDForTouch:(UITouch *)p_touch {
	int first = -1;
	for (int i = 0; i < max_touches; i++) {
		if (first == -1 && godot_touches[i] == NULL) {
			first = i;
			continue;
		}
		if (godot_touches[i] == p_touch) {
			return i;
		}
	}

	if (first != -1) {
		godot_touches[first] = p_touch;
		return first;
	}

	return -1;
}

- (int)removeTouch:(UITouch *)p_touch {
	int remaining = 0;
	for (int i = 0; i < max_touches; i++) {
		if (godot_touches[i] == NULL) {
			continue;
		}
		if (godot_touches[i] == p_touch) {
			godot_touches[i] = NULL;
		} else {
			++remaining;
		}
	}
	return remaining;
}

- (void)clearTouches {
	for (int i = 0; i < max_touches; i++) {
		godot_touches[i] = NULL;
	}
}

- (void)touchesBegan:(NSSet *)touchesSet withEvent:(UIEvent *)event {
	NSArray *tlist = [event.allTouches allObjects];
	for (unsigned int i = 0; i < [tlist count]; i++) {
		if ([touchesSet containsObject:[tlist objectAtIndex:i]]) {
			UITouch *touch = [tlist objectAtIndex:i];
			int tid = [self getTouchIDForTouch:touch];
			ERR_FAIL_COND(tid == -1);
			CGPoint touchPoint = [touch locationInView:self];

			if (touch.type == UITouchTypeStylus) {
				OSIPhone::get_singleton()->pencil_press(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, true, touch.tapCount > 1);
			} else {
				OSIPhone::get_singleton()->touch_press(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, true, touch.tapCount > 1);
			}
		}
	}
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
	NSArray *tlist = [event.allTouches allObjects];
	for (unsigned int i = 0; i < [tlist count]; i++) {
		if ([touches containsObject:[tlist objectAtIndex:i]]) {
			UITouch *touch = [tlist objectAtIndex:i];
			int tid = [self getTouchIDForTouch:touch];
			ERR_FAIL_COND(tid == -1);
			CGPoint touchPoint = [touch locationInView:self];
			CGPoint prev_point = [touch previousLocationInView:self];
			CGFloat force = touch.force;
			// Vector2 tilt = touch.azimuthUnitVector;

			if (touch.type == UITouchTypeStylus) {
				OSIPhone::get_singleton()->pencil_drag(tid, prev_point.x * self.contentScaleFactor, prev_point.y * self.contentScaleFactor, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, force);
			} else {
				OSIPhone::get_singleton()->touch_drag(tid, prev_point.x * self.contentScaleFactor, prev_point.y * self.contentScaleFactor, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor);
			}
		}
	}
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
	NSArray *tlist = [event.allTouches allObjects];
	for (unsigned int i = 0; i < [tlist count]; i++) {
		if ([touches containsObject:[tlist objectAtIndex:i]]) {
			UITouch *touch = [tlist objectAtIndex:i];
			int tid = [self getTouchIDForTouch:touch];
			ERR_FAIL_COND(tid == -1);
			[self removeTouch:touch];
			CGPoint touchPoint = [touch locationInView:self];
			if (touch.type == UITouchTypeStylus) {
				OSIPhone::get_singleton()->pencil_press(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, false, false);
			} else {
				OSIPhone::get_singleton()->touch_press(tid, touchPoint.x * self.contentScaleFactor, touchPoint.y * self.contentScaleFactor, false, false);
			}
		}
	}
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {
	NSArray *tlist = [event.allTouches allObjects];
	for (unsigned int i = 0; i < [tlist count]; i++) {
		if ([touches containsObject:[tlist objectAtIndex:i]]) {
			UITouch *touch = [tlist objectAtIndex:i];
			int tid = [self getTouchIDForTouch:touch];
			ERR_FAIL_COND(tid == -1);
			if (touch.type == UITouchTypeStylus) {
				OSIPhone::get_singleton()->pencil_cancelled(tid);
			} else {
				OSIPhone::get_singleton()->touches_cancelled(tid);
			}
		}
	}
	[self clearTouches];
}

// MARK: Motion

- (void)handleMotion {
	if (!self.motionManager) {
		return;
	}

	// Just using polling approach for now, we can set this up so it sends
	// data to us in intervals, might be better. See Apple reference pages
	// for more details:
	// https://developer.apple.com/reference/coremotion/cmmotionmanager?language=objc

	// Apple splits our accelerometer date into a gravity and user movement
	// component. We add them back together
	CMAcceleration gravity = self.motionManager.deviceMotion.gravity;
	CMAcceleration acceleration = self.motionManager.deviceMotion.userAcceleration;

	///@TODO We don't seem to be getting data here, is my device broken or
	/// is this code incorrect?
	CMMagneticField magnetic = self.motionManager.deviceMotion.magneticField.field;

	///@TODO we can access rotationRate as a CMRotationRate variable
	///(processed date) or CMGyroData (raw data), have to see what works
	/// best
	CMRotationRate rotation = self.motionManager.deviceMotion.rotationRate;

	// Adjust for screen orientation.
	// [[UIDevice currentDevice] orientation] changes even if we've fixed
	// our orientation which is not a good thing when you're trying to get
	// your user to move the screen in all directions and want consistent
	// output

	///@TODO Using [[UIApplication sharedApplication] statusBarOrientation]
	/// is a bit of a hack. Godot obviously knows the orientation so maybe
	/// we
	// can use that instead? (note that left and right seem swapped)

	UIInterfaceOrientation interfaceOrientation = UIInterfaceOrientationUnknown;

	if (@available(iOS 13, *)) {
		interfaceOrientation = [UIApplication sharedApplication].delegate.window.windowScene.interfaceOrientation;
	} else {
		interfaceOrientation = [[UIApplication sharedApplication] statusBarOrientation];
	}

	switch (interfaceOrientation) {
		case UIInterfaceOrientationLandscapeLeft: {
			OSIPhone::get_singleton()->update_gravity(-gravity.y, gravity.x, gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(-(acceleration.y + gravity.y), (acceleration.x + gravity.x), acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(-magnetic.y, magnetic.x, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(-rotation.y, rotation.x, rotation.z);
		} break;
		case UIInterfaceOrientationLandscapeRight: {
			OSIPhone::get_singleton()->update_gravity(gravity.y, -gravity.x, gravity.z);
			OSIPhone::get_singleton()->update_accelerometer((acceleration.y + gravity.y), -(acceleration.x + gravity.x), acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(magnetic.y, -magnetic.x, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(rotation.y, -rotation.x, rotation.z);
		} break;
		case UIInterfaceOrientationPortraitUpsideDown: {
			OSIPhone::get_singleton()->update_gravity(-gravity.x, gravity.y, gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(-(acceleration.x + gravity.x), (acceleration.y + gravity.y), acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(-magnetic.x, magnetic.y, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(-rotation.x, rotation.y, rotation.z);
		} break;
		default: { // assume portrait
			OSIPhone::get_singleton()->update_gravity(gravity.x, gravity.y, gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(acceleration.x + gravity.x, acceleration.y + gravity.y, acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(magnetic.x, magnetic.y, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(rotation.x, rotation.y, rotation.z);
		} break;
	}
}

@end
