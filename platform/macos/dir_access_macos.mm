/**************************************************************************/
/*  dir_access_macos.mm                                                   */
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

#import "dir_access_macos.h"

#if defined(UNIX_ENABLED)

#include "core/config/project_settings.h"

#import <AppKit/NSWorkspace.h>
#import <Foundation/Foundation.h>
#include <sys/attr.h>
#include <sys/mount.h>
#include <unistd.h>

#include <cerrno>

String DirAccessMacOS::get_filesystem_type() const {
	struct statfs fs;
	if (statfs(current_dir.utf8().get_data(), &fs) != 0) {
		return "";
	}
	return String::utf8(fs.f_fstypename).to_upper();
}

String DirAccessMacOS::fix_unicode_name(const char *p_name) const {
	String fname;
	if (p_name != nullptr) {
		NSString *nsstr = [[NSString stringWithUTF8String:p_name] precomposedStringWithCanonicalMapping];
		fname.append_utf8([nsstr UTF8String]);
	}

	return fname;
}

int DirAccessMacOS::get_drive_count() {
	NSArray *res_keys = [NSArray arrayWithObjects:NSURLVolumeURLKey, NSURLIsSystemImmutableKey, nil];
	NSArray *vols = [[NSFileManager defaultManager] mountedVolumeURLsIncludingResourceValuesForKeys:res_keys options:NSVolumeEnumerationSkipHiddenVolumes];

	return [vols count];
}

String DirAccessMacOS::get_drive(int p_drive) {
	NSArray *res_keys = [NSArray arrayWithObjects:NSURLVolumeURLKey, NSURLIsSystemImmutableKey, nil];
	NSArray *vols = [[NSFileManager defaultManager] mountedVolumeURLsIncludingResourceValuesForKeys:res_keys options:NSVolumeEnumerationSkipHiddenVolumes];
	int count = [vols count];

	ERR_FAIL_INDEX_V(p_drive, count, "");

	String volname;
	NSString *path = [vols[p_drive] path];

	volname.append_utf8([path UTF8String]);

	return volname;
}

bool DirAccessMacOS::is_hidden(const String &p_name) {
	// Only called from get_next(), so p_name is relative to the open dir_stream.
	ERR_FAIL_NULL_V(dir_stream, false);

	// Same rule as NSURLIsHiddenKey: the real name starts with a dot, or the UF_HIDDEN flag is set.
	// Using the real name makes "." and ".." follow the directory they refer to.
	struct attrlist attrs = {};
	attrs.bitmapcount = ATTR_BIT_MAP_COUNT;
	attrs.commonattr = ATTR_CMN_NAME | ATTR_CMN_FLAGS;

	struct __attribute__((packed)) {
		uint32_t length;
		attrreference_t name_ref;
		uint32_t flags;
		char name[NAME_MAX * 3 + 1];
	} buf;

	if (getattrlistat(dirfd(dir_stream), p_name.utf8().get_data(), &attrs, &buf, sizeof(buf), FSOPT_NOFOLLOW) != 0) {
		return DirAccessUnix::is_hidden(p_name);
	}

	const char *real_name = (const char *)&buf.name_ref + buf.name_ref.attr_dataoffset;
	return real_name[0] == '.' || (buf.flags & UF_HIDDEN) != 0;
}

bool DirAccessMacOS::is_case_sensitive(const String &p_path) const {
	String f = p_path;
	if (!f.is_absolute_path()) {
		f = get_current_dir().path_join(f);
	}
	f = fix_path(f);

	NSURL *url = [NSURL fileURLWithPath:@(f.utf8().get_data())];
	NSNumber *cs = nil;
	if (![url getResourceValue:&cs forKey:NSURLVolumeSupportsCaseSensitiveNamesKey error:nil]) {
		return false;
	}
	return [cs boolValue];
}

bool DirAccessMacOS::is_bundle(const String &p_file) const {
	String f = p_file;
	if (!f.is_absolute_path()) {
		f = get_current_dir().path_join(f);
	}
	f = fix_path(f);

	return [[NSWorkspace sharedWorkspace] isFilePackageAtPath:[NSString stringWithUTF8String:f.utf8().get_data()]];
}

#endif // UNIX_ENABLED
