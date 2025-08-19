/**************************************************************************/
/*  foundation_helpers.mm                                                 */
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

#import "foundation_helpers.h"

#import "core/string/ustring.h"

#import <CoreFoundation/CFString.h>

namespace conv {

NSString *to_nsstring(const String &p_str) {
	return [[NSString alloc] initWithBytes:(const void *)p_str.ptr()
									length:p_str.length() * sizeof(char32_t)
								  encoding:NSUTF32LittleEndianStringEncoding];
}

NSString *to_nsstring(const CharString &p_str) {
	return [[NSString alloc] initWithBytes:(const void *)p_str.ptr()
									length:p_str.length()
								  encoding:NSUTF8StringEncoding];
}

String to_string(NSString *p_str) {
	CFStringRef str = (__bridge CFStringRef)p_str;
	CFStringEncoding fastest = CFStringGetFastestEncoding(str);
	// Sometimes, CFString will return a pointer to it's encoded data,
	// so we can create the string without allocating intermediate buffers.
	const char *p = CFStringGetCStringPtr(str, fastest);
	if (p) {
		switch (fastest) {
			case kCFStringEncodingASCII:
				return String::ascii(Span(p, CFStringGetLength(str)));
			case kCFStringEncodingUTF8:
				return String::utf8(p);
			case kCFStringEncodingUTF32LE:
				return String::utf32(Span((char32_t *)p, CFStringGetLength(str)));
			default:
				break;
		}
	}

	CFRange range = CFRangeMake(0, CFStringGetLength(str));
	CFIndex byte_len = 0;
	// Try to losslessly convert the string directly into a String's buffer to avoid intermediate allocations.
	CFIndex n = CFStringGetBytes(str, range, kCFStringEncodingUTF32LE, 0, NO, nil, 0, &byte_len);
	if (n == range.length) {
		String res;
		res.resize_uninitialized((byte_len / sizeof(char32_t)) + 1);
		res[n] = 0;
		n = CFStringGetBytes(str, range, kCFStringEncodingUTF32LE, 0, NO, (UInt8 *)res.ptrw(), res.length() * sizeof(char32_t), nil);
		return res;
	}

	return String::utf8(p_str.UTF8String);
}

} //namespace conv
