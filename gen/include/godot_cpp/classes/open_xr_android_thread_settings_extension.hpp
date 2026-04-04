/**************************************************************************/
/*  open_xr_android_thread_settings_extension.hpp                         */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRAndroidThreadSettingsExtension : public OpenXRExtensionWrapper {
	GDEXTENSION_CLASS(OpenXRAndroidThreadSettingsExtension, OpenXRExtensionWrapper)

public:
	enum ThreadType {
		THREAD_TYPE_APPLICATION_MAIN = 0,
		THREAD_TYPE_APPLICATION_WORKER = 1,
		THREAD_TYPE_RENDERER_MAIN = 2,
		THREAD_TYPE_RENDERER_WORKER = 3,
	};

	bool set_application_thread_type(OpenXRAndroidThreadSettingsExtension::ThreadType p_thread_type, uint32_t p_thread_id = 0);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRExtensionWrapper::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRAndroidThreadSettingsExtension::ThreadType);

