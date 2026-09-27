/**************************************************************************/
/*  SDL_filesystem_apple_embedded.mm                                      */
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

// Minimal Apple-embedded fallback for the SDL filesystem API.
//
// Godot's SDL snapshot does not ship a sys filesystem driver for Apple
// platforms, but SDL_IOFromFile() (used for gamepad mapping files) calls
// SDL_GetPrefPath() on them. Provide it, rooted at the app's Documents
// directory like Godot's own user-data path. Only compiled for the Apple
// embedded platforms; every other platform either compiles SDL's own
// filesystem driver or never references this symbol.

#include "SDL3/SDL_filesystem.h"
#include "SDL3/SDL_stdinc.h"

#import <Foundation/Foundation.h>
#include <string.h>

extern "C" char *SDL_GetPrefPath(const char *org, const char *app) {
	@autoreleasepool {
		NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
		if ([paths count] == 0) {
			return nullptr;
		}

		NSString *base = [paths objectAtIndex:0];
		if (org && org[0] != '\0') {
			base = [base stringByAppendingPathComponent:[NSString stringWithUTF8String:org]];
		}
		if (app && app[0] != '\0') {
			base = [base stringByAppendingPathComponent:[NSString stringWithUTF8String:app]];
		}

		[[NSFileManager defaultManager] createDirectoryAtPath:base withIntermediateDirectories:YES attributes:nil error:nil];

		const char *utf8 = [[base stringByAppendingString:@"/"] UTF8String];
		if (!utf8) {
			return nullptr;
		}
		char *ret = (char *)SDL_malloc(strlen(utf8) + 1);
		if (ret) {
			strcpy(ret, utf8);
		}
		return ret;
	}
}
