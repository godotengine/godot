/**************************************************************************/
/*  openxr_local_floor_extension.cpp                                      */
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

#include "openxr_local_floor_extension.h"

#include "core/string/print_string.h"

OpenXRLocalFloorExtension *OpenXRLocalFloorExtension::singleton = nullptr;

OpenXRLocalFloorExtension *OpenXRLocalFloorExtension::get_singleton() {
	return singleton;
}

OpenXRLocalFloorExtension::OpenXRLocalFloorExtension() {
	singleton = this;
}

OpenXRLocalFloorExtension::~OpenXRLocalFloorExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRLocalFloorExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_LOCAL_FLOOR_EXTENSION_NAME] = &available;

	return request_extensions;
}

bool OpenXRLocalFloorExtension::is_available() {
	return available;
}
