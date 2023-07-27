/**************************************************************************/
/*  openxr_api_extension.cpp                                              */
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

#include "openxr_api_extension.h"

void OpenXRAPIExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_instance"), &OpenXRAPIExtension::get_instance);
	ClassDB::bind_method(D_METHOD("get_system_id"), &OpenXRAPIExtension::get_system_id);
	ClassDB::bind_method(D_METHOD("get_session"), &OpenXRAPIExtension::get_session);

	ClassDB::bind_method(D_METHOD("transform_from_pose", "pose"), &OpenXRAPIExtension::transform_from_pose);
	ClassDB::bind_method(D_METHOD("xr_result", "result", "format", "args"), &OpenXRAPIExtension::xr_result);
	ClassDB::bind_static_method("OpenXRAPIExtension", D_METHOD("openxr_is_enabled", "check_run_in_editor"), &OpenXRAPIExtension::openxr_is_enabled);
	ClassDB::bind_method(D_METHOD("get_instance_proc_addr", "name"), &OpenXRAPIExtension::get_instance_proc_addr);
	ClassDB::bind_method(D_METHOD("get_error_string", "result"), &OpenXRAPIExtension::get_error_string);
	ClassDB::bind_method(D_METHOD("get_swapchain_format_name", "swapchain_format"), &OpenXRAPIExtension::get_swapchain_format_name);

	ClassDB::bind_method(D_METHOD("is_initialized"), &OpenXRAPIExtension::is_initialized);
	ClassDB::bind_method(D_METHOD("is_running"), &OpenXRAPIExtension::is_running);

	ClassDB::bind_method(D_METHOD("get_play_space"), &OpenXRAPIExtension::get_play_space);
	ClassDB::bind_method(D_METHOD("get_next_frame_time"), &OpenXRAPIExtension::get_next_frame_time);
	ClassDB::bind_method(D_METHOD("can_render"), &OpenXRAPIExtension::can_render);
}

uint64_t OpenXRAPIExtension::get_instance() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	return (uint64_t)OpenXRAPI::get_singleton()->get_instance();
}

uint64_t OpenXRAPIExtension::get_system_id() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	return (uint64_t)OpenXRAPI::get_singleton()->get_system_id();
}

uint64_t OpenXRAPIExtension::get_session() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	return (uint64_t)OpenXRAPI::get_singleton()->get_session();
}

Transform3D OpenXRAPIExtension::transform_from_pose(GDExtensionConstPtr<const void> p_pose) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), Transform3D());
	return OpenXRAPI::get_singleton()->transform_from_pose(*(XrPosef *)p_pose.data);
}

bool OpenXRAPIExtension::xr_result(uint64_t result, String format, Array args) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);
	return OpenXRAPI::get_singleton()->xr_result((XrResult)result, format.utf8().get_data(), args);
}

bool OpenXRAPIExtension::openxr_is_enabled(bool p_check_run_in_editor) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);
	return OpenXRAPI::openxr_is_enabled(p_check_run_in_editor);
}

uint64_t OpenXRAPIExtension::get_instance_proc_addr(String p_name) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	CharString str = p_name.utf8();
	PFN_xrVoidFunction addr = nullptr;
	XrResult result = OpenXRAPI::get_singleton()->get_instance_proc_addr(str.get_data(), &addr);
	if (result != XR_SUCCESS) {
		return 0;
	}
	return reinterpret_cast<uint64_t>(addr);
}

String OpenXRAPIExtension::get_error_string(uint64_t result) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), String());
	return OpenXRAPI::get_singleton()->get_error_string((XrResult)result);
}

String OpenXRAPIExtension::get_swapchain_format_name(int64_t p_swapchain_format) {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), String());
	return OpenXRAPI::get_singleton()->get_swapchain_format_name(p_swapchain_format);
}

bool OpenXRAPIExtension::is_initialized() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);
	return OpenXRAPI::get_singleton()->is_initialized();
}

bool OpenXRAPIExtension::is_running() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);
	return OpenXRAPI::get_singleton()->is_running();
}

uint64_t OpenXRAPIExtension::get_play_space() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	return (uint64_t)OpenXRAPI::get_singleton()->get_play_space();
}

int64_t OpenXRAPIExtension::get_next_frame_time() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), 0);
	return (XrTime)OpenXRAPI::get_singleton()->get_next_frame_time();
}

bool OpenXRAPIExtension::can_render() {
	ERR_FAIL_NULL_V(OpenXRAPI::get_singleton(), false);
	return OpenXRAPI::get_singleton()->can_render();
}

OpenXRAPIExtension::OpenXRAPIExtension() {
}
