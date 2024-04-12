/**************************************************************************/
/*  godot_status_item.mm                                                  */
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

#include "godot_status_item.h"

#include "display_server_macos.h"

@implementation GodotStatusItemView

- (id)init {
	self = [super init];
	image = nullptr;
	return self;
}

- (void)setImage:(NSImage *)newImage {
	image = newImage;
	[self setNeedsDisplayInRect:self.frame];
}

- (void)setCallback:(const Callable &)callback {
	cb = callback;
}

- (void)drawRect:(NSRect)rect {
	if (image) {
		[image drawInRect:rect];
	}
}

- (void)processMouseEvent:(NSEvent *)event index:(MouseButton)index {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (!ds) {
		return;
	}

	if (cb.is_valid()) {
		Variant v_button = index;
		Variant v_pos = ds->mouse_get_position();
		Variant *v_args[2] = { &v_button, &v_pos };
		Variant ret;
		Callable::CallError ce;
		cb.callp((const Variant **)&v_args, 2, ret, ce);
	}
}

- (void)mouseDown:(NSEvent *)event {
	[super mouseDown:event];
	if (([event modifierFlags] & NSEventModifierFlagControl)) {
		[self processMouseEvent:event index:MouseButton::RIGHT];
	} else {
		[self processMouseEvent:event index:MouseButton::LEFT];
	}
}

- (void)rightMouseDown:(NSEvent *)event {
	[super rightMouseDown:event];

	[self processMouseEvent:event index:MouseButton::RIGHT];
}

- (void)otherMouseDown:(NSEvent *)event {
	[super otherMouseDown:event];

	if ((int)[event buttonNumber] == 2) {
		[self processMouseEvent:event index:MouseButton::MIDDLE];
	} else if ((int)[event buttonNumber] == 3) {
		[self processMouseEvent:event index:MouseButton::MB_XBUTTON1];
	} else if ((int)[event buttonNumber] == 4) {
		[self processMouseEvent:event index:MouseButton::MB_XBUTTON2];
	}
}

@end
