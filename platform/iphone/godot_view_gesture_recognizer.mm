/*************************************************************************/
/*  godot_view_gesture_recognizer.mm                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#import "godot_view_gesture_recognizer.h"

#include "core/project_settings.h"

// Minimum distance for touches to move to fire
// a delay timer before scheduled time.
// Should be the low enough to not cause issues with dragging
// but big enough to allow click to work.
const CGFloat kGLGestureMovementDistance = 0.5;

@interface GodotViewGestureRecognizer ()

@property(nonatomic, readwrite, assign) NSTimeInterval delayTimeInterval;

@end

@interface GodotViewGestureRecognizer ()

// Timer used to delay begin touch message.
// Should work as simple emulation of UIDelayedAction
@property(strong, nonatomic) NSTimer *delayTimer;

// Delayed touch parameters
@property(strong, nonatomic) NSSet *delayedTouches;
@property(strong, nonatomic) UIEvent *delayedEvent;

@end

@implementation GodotViewGestureRecognizer

- (instancetype)init {
	self = [super init];

	self.cancelsTouchesInView = YES;
	self.delaysTouchesBegan = YES;
	self.delaysTouchesEnded = YES;

	self.delayTimeInterval = GLOBAL_GET("input_devices/pointing/ios/touch_delay");

	return self;
}

- (void)dealloc {
	if (self.delayTimer) {
		[self.delayTimer invalidate];
		self.delayTimer = nil;
	}

	if (self.delayedTouches) {
		self.delayedTouches = nil;
	}

	if (self.delayedEvent) {
		self.delayedEvent = nil;
	}
}

- (void)delayTouches:(NSSet *)touches andEvent:(UIEvent *)event {
	[self.delayTimer fire];

	self.delayedTouches = touches;
	self.delayedEvent = event;

	self.delayTimer = [NSTimer
			scheduledTimerWithTimeInterval:self.delayTimeInterval
									target:self
								  selector:@selector(fireDelayedTouches:)
								  userInfo:nil
								   repeats:NO];
}

- (void)fireDelayedTouches:(id)timer {
	[self.delayTimer invalidate];
	self.delayTimer = nil;

	if (self.delayedTouches) {
		[self.view touchesBegan:self.delayedTouches withEvent:self.delayedEvent];
	}

	self.delayedTouches = nil;
	self.delayedEvent = nil;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseBegan];
	[self delayTouches:cleared andEvent:event];
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseMoved];

	if (self.delayTimer) {
		// We should check if movement was significant enough to fire an event
		// for dragging to work correctly.
		for (UITouch *touch in cleared) {
			CGPoint from = [touch locationInView:self.view];
			CGPoint to = [touch previousLocationInView:self.view];
			CGFloat xDistance = from.x - to.x;
			CGFloat yDistance = from.y - to.y;

			CGFloat distance = sqrt(xDistance * xDistance + yDistance * yDistance);

			// Early exit, since one of touches has moved enough to fire a drag event.
			if (distance > kGLGestureMovementDistance) {
				[self.delayTimer fire];
				[self.view touchesMoved:cleared withEvent:event];
				return;
			}
		}

		return;
	}

	[self.view touchesMoved:cleared withEvent:event];
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
	[self.delayTimer fire];

	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseEnded];
	[self.view touchesEnded:cleared withEvent:event];
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {
	[self.delayTimer fire];
	[self.view touchesCancelled:touches withEvent:event];
};

- (NSSet *)copyClearedTouches:(NSSet *)touches phase:(UITouchPhase)phaseToSave {
	NSMutableSet *cleared = [touches mutableCopy];

	for (UITouch *touch in touches) {
		if (touch.phase != phaseToSave) {
			[cleared removeObject:touch];
		}
	}

	return cleared;
}

@end
