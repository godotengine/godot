/*************************************************************************/
/*  godot_view_gesture_recognizer.mm                                     */
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

#import "godot_view_gesture_recognizer.h"

#include "core/config/project_settings.h"
#include "os_appletv.h"

@interface GodotViewGestureRecognizer ()

// Timer used to delay end press message.
@property(nonatomic, readwrite, strong) NSTimer *delayTimer;

// Delayed touch parameters
@property(nonatomic, readwrite, copy) NSSet *delayedPresses;
@property(nonatomic, readwrite, strong) UIPressesEvent *delayedEvent;

@property(nonatomic, readwrite, assign) NSTimeInterval delayTimeInterval;

@end

@implementation GodotViewGestureRecognizer

- (instancetype)init {
	self = [super init];

	self.delayTimeInterval = GLOBAL_GET("input_devices/pointing/tvos/press_end_delay");
	self.allowedPressTypes = @[ @(UIPressTypeMenu) ];

	return self;
}

- (void)delayPresses:(NSSet *)presses andEvent:(UIPressesEvent *)event {
	[self.delayTimer fire];

	self.delayedPresses = presses;
	self.delayedEvent = event;

	self.delayTimer = [NSTimer scheduledTimerWithTimeInterval:self.delayTimeInterval target:self selector:@selector(fireDelayedPress:) userInfo:nil repeats:NO];
}

- (void)fireDelayedPress:(id)timer {
	[self.delayTimer invalidate];
	self.delayTimer = nil;

	if (self.delayedPresses) {
		[self.view pressesEnded:self.delayedPresses withEvent:self.delayedEvent];
	}

	self.delayedPresses = nil;
	self.delayedEvent = nil;
}

- (BOOL)overridesRemoteButtons {
	return OSAppleTV::get_singleton()->get_overrides_menu_button();
}

- (BOOL)shouldReceiveEvent:(UIEvent *)event {
	return self.overridesRemoteButtons;
}

- (void)pressesBegan:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	[self.delayTimer fire];
	[self.view pressesBegan:presses withEvent:event];
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	NSSet *cleared = [self copyClearedPresses:presses type:UIPressTypeMenu phase:UIPressPhaseEnded];
	[self delayPresses:cleared andEvent:event];
}

- (void)pressesCancelled:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	[self cancelDelayTimer];
	[self.view pressesCancelled:presses withEvent:event];
};

- (void)cancelDelayTimer {
	[self.delayTimer invalidate];
	self.delayTimer = nil;
	self.delayedPresses = nil;
	self.delayedEvent = nil;
}

- (NSSet *)copyClearedPresses:(NSSet *)presses type:(UIPressType)phaseToSave phase:(UIPressPhase)phase {
	NSMutableSet *cleared = [NSMutableSet new];

	for (UIPress *press in presses) {
		if (press.type == phaseToSave && press.phase == phase) {
			[cleared addObject:press];
		}
	}

	return cleared;
}

@end
