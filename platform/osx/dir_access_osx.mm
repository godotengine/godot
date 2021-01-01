/*************************************************************************/
/*  dir_access_osx.mm                                                    */
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
	NSArray *res_keys = [NSArray arrayWithObjects:NSURLVolumeURLKey, NSURLIsSystemImmutableKey, nil];
	NSArray *vols = [[NSFileManager defaultManager] mountedVolumeURLsIncludingResourceValuesForKeys:res_keys options:NSVolumeEnumerationSkipHiddenVolumes];

	return [vols count];
}

String DirAccessOSX::get_drive(int p_drive) {
	NSArray *res_keys = [NSArray arrayWithObjects:NSURLVolumeURLKey, NSURLIsSystemImmutableKey, nil];
	NSArray *vols = [[NSFileManager defaultManager] mountedVolumeURLsIncludingResourceValuesForKeys:res_keys options:NSVolumeEnumerationSkipHiddenVolumes];
	int count = [vols count];

	ERR_FAIL_INDEX_V(p_drive, count, "");

	String volname;
	NSString *path = [vols[p_drive] path];

	volname.parse_utf8([path UTF8String]);

	return volname;
}

bool DirAccessOSX::is_hidden(const String &p_name) {
	String f = get_current_dir().plus_file(p_name);
	NSURL *url = [NSURL fileURLWithPath:@(f.utf8().get_data())];
	NSNumber *hidden = nil;
	if (![url getResourceValue:&hidden forKey:NSURLIsHiddenKey error:nil]) {
		return DirAccessUnix::is_hidden(p_name);
	}
	return [hidden boolValue];
}

#endif //posix_enabled
