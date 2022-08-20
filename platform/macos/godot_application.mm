/*************************************************************************/
/*  godot_application.mm                                                 */
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

#include "godot_application.h"

#include "display_server_macos.h"

@implementation GodotApplication

- (void)sendEvent:(NSEvent *)event {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		if ([event type] == NSEventTypeLeftMouseDown || [event type] == NSEventTypeRightMouseDown || [event type] == NSEventTypeOtherMouseDown) {
			if (ds->mouse_process_popups()) {
				return;
			}
		}
		ds->send_event(event);
	}

	// From http://cocoadev.com/index.pl?GameKeyboardHandlingAlmost
	// This works around an AppKit bug, where key up events while holding
	// down the command key don't get sent to the key window.
	if ([event type] == NSEventTypeKeyUp && ([event modifierFlags] & NSEventModifierFlagCommand)) {
		[[self keyWindow] sendEvent:event];
	} else {
		[super sendEvent:event];
	}
}

@end
