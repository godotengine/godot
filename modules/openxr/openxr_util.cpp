/**************************************************************************/
/*  openxr_util.cpp                                                       */
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

#include "openxr_util.h"

#include <openxr/openxr_reflection.h>

#define XR_ENUM_CASE_STR(name, val) \
	case name:                      \
		return #name;
#define XR_ENUM_SWITCH(enumType, var)                                                                                           \
	switch (var) {                                                                                                              \
		XR_LIST_ENUM_##enumType(XR_ENUM_CASE_STR) default : return "Unknown " #enumType ": " + String::num_int64(int64_t(var)); \
	}

String OpenXRUtil::get_view_configuration_name(XrViewConfigurationType p_view_configuration){
	XR_ENUM_SWITCH(XrViewConfigurationType, p_view_configuration)
}

String OpenXRUtil::get_reference_space_name(XrReferenceSpaceType p_reference_space){
	XR_ENUM_SWITCH(XrReferenceSpaceType, p_reference_space)
}

String OpenXRUtil::get_structure_type_name(XrStructureType p_structure_type){
	XR_ENUM_SWITCH(XrStructureType, p_structure_type)
}

String OpenXRUtil::get_session_state_name(XrSessionState p_session_state){
	XR_ENUM_SWITCH(XrSessionState, p_session_state)
}

String OpenXRUtil::get_action_type_name(XrActionType p_action_type){
	XR_ENUM_SWITCH(XrActionType, p_action_type)
}

String OpenXRUtil::get_environment_blend_mode_name(XrEnvironmentBlendMode p_blend_mode) {
	XR_ENUM_SWITCH(XrEnvironmentBlendMode, p_blend_mode);
}

String OpenXRUtil::make_xr_version_string(XrVersion p_version) {
	String version;

	version += String::num_int64(XR_VERSION_MAJOR(p_version));
	version += String(".");
	version += String::num_int64(XR_VERSION_MINOR(p_version));
	version += String(".");
	version += String::num_int64(XR_VERSION_PATCH(p_version));

	return version;
}
