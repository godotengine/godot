/**************************************************************************/
/*  godot_menu_item.mm                                                    */
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

#import "godot_menu_item.h"

#include "core/string/fuzzy_search.h"

@implementation GodotMenuItem

- (id)init {
	self = [super init];

	self->callback = Callable();
	self->key_callback = Callable();
	self->checkable_type = GlobalMenuCheckType::CHECKABLE_TYPE_NONE;
	self->enabled = true;
	self->checked = false;
	self->max_states = 0;
	self->state = 0;
	self->accel = Key::NONE;

	return self;
}

@end

@implementation GodotSearchField

- (id)init {
	self = [super init];

	self->host_menu = nullptr;

	return self;
}

- (void)filterItems:(NSMenu *)p_menu filterQuery:(const String &)p_query {
	PackedStringArray search_names;

	for (NSInteger i = 1; i < [p_menu numberOfItems]; i++) {
		const NSMenuItem *menu_item = [p_menu itemAtIndex:i];
		search_names.append(String::utf8([[menu_item title] UTF8String]));
	}

	Vector<FuzzySearchResult> results;
	FuzzySearch fuzzy;
	fuzzy.set_query(p_query);
	fuzzy.search_all(search_names, results);

	for (NSInteger i = 1; i < [p_menu numberOfItems]; i++) {
		const NSMenuItem *menu_item = [p_menu itemAtIndex:i];

		bool submenu_visible = false;
		NSMenu *sub_menu = [menu_item submenu];
		if (sub_menu) {
			[self filterItems:sub_menu filterQuery:p_query];
			for (NSInteger j = 1; j < [sub_menu numberOfItems]; j++) {
				if (![sub_menu itemAtIndex:j].hidden) {
					submenu_visible = true;
					break;
				}
			}
		}

		menu_item.hidden = !(p_query.length() == 0 || submenu_visible);
	}

	for (const FuzzySearchResult &res : results) {
		const NSMenuItem *menu_item = [p_menu itemAtIndex:res.original_index + 1];
		menu_item.hidden = res.score <= 0;
	}
}

- (void)textDidChange:(NSNotification *)notification {
	if (host_menu) {
		String query = String::utf8([[self stringValue] UTF8String]);
		[self filterItems:host_menu filterQuery:query];
	}
}

@end
