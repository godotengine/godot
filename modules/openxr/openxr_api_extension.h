/**************************************************************************/
/*  openxr_api_extension.h                                                */
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

#ifndef OPENXR_API_EXTENSION_H
#define OPENXR_API_EXTENSION_H

#include "openxr_api.h"

#include "core/object/ref_counted.h"
#include "core/os/os.h"
#include "core/os/thread_safe.h"
#include "core/variant/native_ptr.h"

class OpenXRAPIExtension : public RefCounted {
	GDCLASS(OpenXRAPIExtension, RefCounted);

protected:
	_THREAD_SAFE_CLASS_

	static void _bind_methods();

public:
	uint64_t get_instance();
	uint64_t get_system_id();
	uint64_t get_session();

	// Helper method to convert an XrPosef to a Transform3D.
	Transform3D transform_from_pose(GDExtensionConstPtr<const void> p_pose);

	bool xr_result(uint64_t result, String format, Array args = Array());

	static bool openxr_is_enabled(bool p_check_run_in_editor = true);

	//TODO workaround as GDExtensionPtr<void> return type results in build error in godot-cpp
	uint64_t get_instance_proc_addr(String p_name);
	String get_error_string(uint64_t result);
	String get_swapchain_format_name(int64_t p_swapchain_format);

	bool is_initialized();
	bool is_running();

	uint64_t get_play_space();
	int64_t get_next_frame_time();
	bool can_render();

	OpenXRAPIExtension();
};

#endif // OPENXR_API_EXTENSION_H
