/*************************************************************************/
/*  dir_access_osx.mm                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "dir_access_osx.h"

#if defined(UNIX_ENABLED) || defined(LIBC_FILEIO_ENABLED)

#include <errno.h>

#include <AppKit/NSWorkspace.h>
#include <Foundation/Foundation.h>

String DirAccessOSX::fix_unicode_name(const char *p_name) const {

	String fname;
	NSString *nsstr = [[NSString stringWithUTF8String:p_name] precomposedStringWithCanonicalMapping];

	fname.parse_utf8([nsstr UTF8String]);

	return fname;
}

int DirAccessOSX::get_drive_count() {
	NSArray *vols = [[NSWorkspace sharedWorkspace] mountedLocalVolumePaths];
	return [vols count];
}

String DirAccessOSX::get_drive(int p_drive) {
	NSArray *vols = [[NSWorkspace sharedWorkspace] mountedLocalVolumePaths];
	int count = [vols count];

	ERR_FAIL_INDEX_V(p_drive, count, "");

	NSString *path = vols[p_drive];
	return String([path UTF8String]);
}

#endif //posix_enabled
