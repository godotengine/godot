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
#include "os_tvos.h"
#include "servers/audio_server.h"

#import <OpenGLES/EAGLDrawable.h>
#import <QuartzCore/QuartzCore.h>

#import "godot_view_gesture_recognizer.h"

@interface GodotView ()

@property(strong, nonatomic) GodotViewGestureRecognizer *delayGestureRecognizer;

@end

@implementation GodotView

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
	if (self.delayGestureRecognizer) {
		self.delayGestureRecognizer = nil;
	}
}

- (void)godot_commonInit {
	// Initialize delay gesture recognizer
	GodotViewGestureRecognizer *gestureRecognizer = [[GodotViewGestureRecognizer alloc] init];
	self.delayGestureRecognizer = gestureRecognizer;
	[self addGestureRecognizer:self.delayGestureRecognizer];
}

// MARK: - Input

// MARK: Menu Button

- (void)pressesBegan:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	if (!self.delayGestureRecognizer.overridesRemoteButtons) {
		return [super pressesEnded:presses withEvent:event];
	}

	NSArray *tlist = [event.allPresses allObjects];

	for (UIPress *press in tlist) {
		if ([presses containsObject:press] && press.type == UIPressTypeMenu) {
			int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");
			OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, true);
		} else {
			[super pressesBegan:presses withEvent:event];
		}
	}
}

- (void)pressesEnded:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	if (!self.delayGestureRecognizer.overridesRemoteButtons) {
		return [super pressesEnded:presses withEvent:event];
	}

	NSArray *tlist = [presses allObjects];

	for (UIPress *press in tlist) {
		if ([presses containsObject:press] && press.type == UIPressTypeMenu) {
			int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");
			OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, false);
		} else {
			[super pressesEnded:presses withEvent:event];
		}
	}
}

- (void)pressesCancelled:(NSSet<UIPress *> *)presses withEvent:(UIPressesEvent *)event {
	if (!self.delayGestureRecognizer.overridesRemoteButtons) {
		return [super pressesEnded:presses withEvent:event];
	}

	NSArray *tlist = [event.allPresses allObjects];

	for (UIPress *press in tlist) {
		if ([presses containsObject:press] && press.type == UIPressTypeMenu) {
			int joy_id = OSAppleTV::get_singleton()->joy_id_for_name("Remote");
			OSAppleTV::get_singleton()->joy_button(joy_id, JOY_START, false);
		} else {
			[super pressesCancelled:presses withEvent:event];
		}
	}
}

@end
