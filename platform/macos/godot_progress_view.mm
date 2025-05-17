/**************************************************************************/
/*  godot_progress_view.mm                                                */
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

#import "godot_progress_view.h"

@implementation GodotProgressView

- (id)init {
	self = [super init];
	pr_state = DisplayServer::PROGRESS_STATE_NOPROGRESS;
	pr_value = 0.f;
	pr_offset = 0.f;
	return self;
}

- (void)setValue:(float)value {
	pr_value = value;
}

- (void)setState:(DisplayServer::ProgressState)state {
	pr_state = state;
}

- (void)drawRect:(NSRect)dirtyRect {
	// Icon draw.
	[[NSGraphicsContext currentContext] setImageInterpolation:NSImageInterpolationHigh];
	[[NSApp applicationIconImage] drawInRect:self.bounds];

	if (pr_state == DisplayServer::PROGRESS_STATE_NOPROGRESS) {
		return;
	}

	// Border draw.
	NSRect rect = NSMakeRect(1.f, 1.f, self.bounds.size.width - 2.f, 16.f);
	NSBezierPath *bezier_path = [NSBezierPath bezierPathWithRoundedRect:rect xRadius:8.f yRadius:8.f];
	[bezier_path setLineWidth:2.0];
	[[NSColor grayColor] set];
	[bezier_path stroke];

	// Fill clip path.
	rect = NSMakeRect(2.f, 2.f, self.bounds.size.width - 4.f, 14.f);
	bezier_path = [NSBezierPath bezierPathWithRoundedRect:rect xRadius:7.f yRadius:7.f];
	[bezier_path setLineWidth:1.0];
	[bezier_path addClip];

	// Fill draw.
	if (pr_state == DisplayServer::PROGRESS_STATE_INDETERMINATE) {
		rect.size.width /= 5.0;
		pr_offset += rect.size.width / 10.0;
		if (pr_offset > self.bounds.size.width - rect.size.width) {
			pr_offset = 0.f;
		}
		rect.origin.x += pr_offset;
	} else {
		rect.size.width = Math::floor(rect.size.width * pr_value);
	}
	if (pr_state == DisplayServer::PROGRESS_STATE_ERROR) {
		[[NSColor colorWithSRGBRed:1.0 green:0.2 blue:0.2 alpha:1.0] set];
	} else if (pr_state == DisplayServer::PROGRESS_STATE_PAUSED) {
		[[NSColor colorWithSRGBRed:1.0 green:1.0 blue:0.2 alpha:1.0] set];
	} else {
		[[NSColor colorWithSRGBRed:0.2 green:0.6 blue:1.0 alpha:1.0] set];
	}
	NSRectFill(rect);
}

@end
