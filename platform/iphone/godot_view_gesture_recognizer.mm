/*************************************************************************/
/*  godot_view_gesture_recognizer.mm                                     */
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

#import "godot_view_gesture_recognizer.h"
#import "godot_view.h"

#include "core/project_settings.h"

// Minimum distance for touches to move to fire
// a delay timer before scheduled time.
// Should be the low enough to not cause issues with dragging
// but big enough to allow click to work.
const CGFloat kGLGestureMovementDistance = 0.5;

@interface GodotViewGestureRecognizer ()

@property(nonatomic, readwrite, assign) NSTimeInterval delayTimeInterval;

@end

@implementation GodotViewGestureRecognizer

- (GodotView *)godotView {
	return (GodotView *)self.view;
}

- (instancetype)init {
	self = [super init];

	self.cancelsTouchesInView = YES;
	self.delaysTouchesBegan = YES;
	self.delaysTouchesEnded = YES;

	self.delayTimeInterval = GLOBAL_GET("input_devices/pointing/ios/touch_delay");

	return self;
}

- (void)delayTouches:(NSSet *)touches andEvent:(UIEvent *)event {
	[delayTimer fire];

	delayedTouches = touches;
	delayedEvent = event;

	delayTimer = [NSTimer scheduledTimerWithTimeInterval:self.delayTimeInterval target:self selector:@selector(fireDelayedTouches:) userInfo:nil repeats:NO];
}

- (void)fireDelayedTouches:(id)timer {
	[delayTimer invalidate];
	delayTimer = nil;

	if (delayedTouches) {
		[self.godotView godotTouchesBegan:delayedTouches withEvent:delayedEvent];
	}

	delayedTouches = nil;
	delayedEvent = nil;
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseBegan];
	[self delayTouches:cleared andEvent:event];

	[super touchesBegan:touches withEvent:event];
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseMoved];

	if (delayTimer) {
		// We should check if movement was significant enough to fire an event
		// for dragging to work correctly.
		for (UITouch *touch in cleared) {
			CGPoint from = [touch locationInView:self.godotView];
			CGPoint to = [touch previousLocationInView:self.godotView];
			CGFloat xDistance = from.x - to.x;
			CGFloat yDistance = from.y - to.y;

			CGFloat distance = sqrt(xDistance * xDistance + yDistance * yDistance);

			// Early exit, since one of touches has moved enough to fire a drag event.
			if (distance > kGLGestureMovementDistance) {
				[delayTimer fire];
				[self.godotView godotTouchesMoved:cleared withEvent:event];
				return;
			}
		}
		return;
	}

	[self.godotView godotTouchesMoved:cleared withEvent:event];

	[super touchesMoved:touches withEvent:event];
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
	[delayTimer fire];

	NSSet *cleared = [self copyClearedTouches:touches phase:UITouchPhaseEnded];
	[self.godotView godotTouchesEnded:cleared withEvent:event];

	[super touchesEnded:touches withEvent:event];
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {
	[delayTimer fire];
	[self.godotView godotTouchesCancelled:touches withEvent:event];

	[super touchesCancelled:touches withEvent:event];
}

- (NSSet *)copyClearedTouches:(NSSet *)touches phase:(UITouchPhase)phaseToSave {
	NSMutableSet *cleared = [touches mutableCopy];

	for (UITouch *touch in touches) {
		if (touch.view != self.view || touch.phase != phaseToSave) {
			[cleared removeObject:touch];
		}
	}

	return cleared;
}

@end
