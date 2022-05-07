/*************************************************************************/
/*  osx_utils.cpp                                                        */
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

#include "osx_utils.h"

#ifdef OSX_ENABLED

#include "core/print_string.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>

bool osx_is_app_bundle_installed(const String &p_bundle_id) {
	CFURLRef app_url = NULL;
	CFStringRef bundle_id = CFStringCreateWithCString(NULL, p_bundle_id.utf8(), kCFStringEncodingUTF8);
	OSStatus result = LSFindApplicationForInfo(kLSUnknownCreator, bundle_id, NULL, NULL, &app_url);
	CFRelease(bundle_id);

	if (app_url)
		CFRelease(app_url);

	switch (result) {
		case noErr:
			return true;
		case kLSApplicationNotFoundErr:
			break;
		default:
			break;
	}

	return false;
}

#endif
