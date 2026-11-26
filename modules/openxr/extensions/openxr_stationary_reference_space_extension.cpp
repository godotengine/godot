/**************************************************************************/
/*  openxr_stationary_reference_space_extension.cpp                       */
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

#include "openxr_stationary_reference_space_extension.h"

#include "../openxr_api.h"
#include "../openxr_util.h"

#include "core/object/class_db.h"

OpenXRStationaryReferenceSpaceExtension *OpenXRStationaryReferenceSpaceExtension::singleton = nullptr;

OpenXRStationaryReferenceSpaceExtension *OpenXRStationaryReferenceSpaceExtension::get_singleton() {
	return singleton;
}

OpenXRStationaryReferenceSpaceExtension::OpenXRStationaryReferenceSpaceExtension() {
	singleton = this;
}

OpenXRStationaryReferenceSpaceExtension::~OpenXRStationaryReferenceSpaceExtension() {
	singleton = nullptr;
}

void OpenXRStationaryReferenceSpaceExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_available"), &OpenXRStationaryReferenceSpaceExtension::is_available);
	ClassDB::bind_method(D_METHOD("get_generation_id"), &OpenXRStationaryReferenceSpaceExtension::get_generation_id);
}

HashMap<String, bool *> OpenXRStationaryReferenceSpaceExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_EXT_STATIONARY_REFERENCE_SPACE_EXTENSION_NAME] = &available;

	return request_extensions;
}

void OpenXRStationaryReferenceSpaceExtension::on_instance_created(const XrInstance p_instance) {
	if (available) {
		EXT_INIT_XR_FUNC(xrGetStationaryReferenceSpaceGenerationIdEXT);
	}
}

void OpenXRStationaryReferenceSpaceExtension::on_instance_destroyed() {
	available = false;
}

bool OpenXRStationaryReferenceSpaceExtension::is_available() const {
	return available;
}

String OpenXRStationaryReferenceSpaceExtension::get_generation_id() const {
	ERR_FAIL_COND_V(!available, String());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, String());

	XrSession session = openxr_api->get_session();
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, String());

	XrStationaryReferenceSpaceGenerationIdGetInfoEXT get_info = {
		XR_TYPE_STATIONARY_REFERENCE_SPACE_GENERATION_ID_GET_INFO_EXT, // type
		nullptr, // next
	};

	XrStationaryReferenceSpaceGenerationIdResultEXT generation_id_result = {
		XR_TYPE_STATIONARY_REFERENCE_SPACE_GENERATION_ID_RESULT_EXT, // type
		nullptr, // next
		{}, // generationId
	};

	XrResult result = xrGetStationaryReferenceSpaceGenerationIdEXT(session, &get_info, &generation_id_result);
	if (XR_FAILED(result)) {
		print_line(vformat("OpenXR: Failed to get stationary reference space generation ID [%s]", openxr_api->get_error_string(result)));
		return String();
	}

	return OpenXRUtil::string_from_xruuid(generation_id_result.generationId);
}
