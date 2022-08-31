/*************************************************************************/
/*  godot_menu_delegate.mm                                               */
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

#include "godot_menu_delegate.h"

#include "display_server_macos.h"
#include "godot_menu_item.h"
#include "key_mapping_macos.h"

@implementation GodotMenuDelegate

- (void)doNothing:(id)sender {
}

- (BOOL)menuHasKeyEquivalent:(NSMenu *)menu forEvent:(NSEvent *)event target:(id *)target action:(SEL *)action {
	NSString *ev_key = [[event charactersIgnoringModifiers] lowercaseString];
	NSUInteger ev_modifiers = [event modifierFlags] & NSDeviceIndependentModifierFlagsMask;
	for (int i = 0; i < [menu numberOfItems]; i++) {
		const NSMenuItem *menu_item = [menu itemAtIndex:i];
		if ([menu_item isEnabled] && [[menu_item keyEquivalent] compare:ev_key] == NSOrderedSame) {
			NSUInteger item_modifiers = [menu_item keyEquivalentModifierMask];

			if (ev_modifiers == item_modifiers) {
				GodotMenuItem *value = [menu_item representedObject];
				if (value->key_callback != Callable()) {
					// If custom callback is set, use it.
					Variant tag = value->meta;
					Variant *tagp = &tag;
					Variant ret;
					Callable::CallError ce;
					value->key_callback.callp((const Variant **)&tagp, 1, ret, ce);
				} else {
					// Otherwise redirect event to the engine.
					if (DisplayServer::get_singleton()) {
						DisplayServerMacOS::KeyEvent ke;

						ke.window_id = DisplayServer::MAIN_WINDOW_ID;
						ke.macos_state = [event modifierFlags];
						ke.pressed = true;
						ke.echo = [event isARepeat];
						ke.keycode = KeyMappingMacOS::remap_key([event keyCode], [event modifierFlags]);
						ke.physical_keycode = KeyMappingMacOS::translate_key([event keyCode]);
						ke.raw = false;
						ke.unicode = 0;

						reinterpret_cast<DisplayServerMacOS *>(DisplayServer::get_singleton())->push_to_key_event_buffer(ke);
					}
				}

				// Suppress default menu action.
				*target = self;
				*action = @selector(doNothing:);
				return YES;
			}
		}
	}
	return NO;
}

@end
