/**************************************************************************/
/*  openxr_api.cpp                                                        */
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

#include "openxr_api.h"

#include "extensions/openxr_extension_wrapper_extension.h"
#include "openxr_interface.h"
#include "openxr_util.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/os/memory.h"
#include "core/version.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

#include "openxr_platform_inc.h"

#ifdef VULKAN_ENABLED
#include "extensions/openxr_vulkan_extension.h"
#endif

#if defined(GLES3_ENABLED) && !defined(MACOS_ENABLED)
#include "extensions/openxr_opengl_extension.h"
#endif

#include "extensions/openxr_composition_layer_depth_extension.h"
#include "extensions/openxr_fb_display_refresh_rate_extension.h"
#include "extensions/openxr_fb_foveation_extension.h"
#include "extensions/openxr_fb_passthrough_extension_wrapper.h"
#include "extensions/openxr_fb_update_swapchain_extension.h"

#ifdef ANDROID_ENABLED
#define OPENXR_LOADER_NAME "libopenxr_loader.so"
#endif

OpenXRAPI *OpenXRAPI::singleton = nullptr;
Vector<OpenXRExtensionWrapper *> OpenXRAPI::registered_extension_wrappers;

bool OpenXRAPI::openxr_is_enabled(bool p_check_run_in_editor) {
	// @TODO we need an overrule switch so we can force enable openxr, i.e run "godot --openxr_enabled"

	if (Engine::get_singleton()->is_editor_hint() && p_check_run_in_editor) {
		// Disabled for now, using XR inside of the editor we'll be working on during the coming months.
		return false;
	} else {
		if (XRServer::get_xr_mode() == XRServer::XRMODE_DEFAULT) {
			return GLOBAL_GET("xr/openxr/enabled");
		} else {
			return XRServer::get_xr_mode() == XRServer::XRMODE_ON;
		}
	}
}

String OpenXRAPI::get_default_action_map_resource_name() {
	String name = GLOBAL_GET("xr/openxr/default_action_map");

	return name;
}

String OpenXRAPI::get_error_string(XrResult result) {
	if (XR_SUCCEEDED(result)) {
		return String("Succeeded");
	}

	if (instance == XR_NULL_HANDLE) {
		Array args;
		args.push_back(Variant(result));
		return String("Error code {0}").format(args);
	}

	char resultString[XR_MAX_RESULT_STRING_SIZE];
	xrResultToString(instance, result, resultString);

	return String(resultString);
}

String OpenXRAPI::get_swapchain_format_name(int64_t p_swapchain_format) const {
	// This is rendering engine dependent...
	if (graphics_extension) {
		return graphics_extension->get_swapchain_format_name(p_swapchain_format);
	}

	return String("Swapchain format ") + String::num_int64(int64_t(p_swapchain_format));
}

bool OpenXRAPI::load_layer_properties() {
	// This queries additional layers that are available and can be initialized when we create our OpenXR instance
	if (layer_properties != nullptr) {
		// already retrieved this
		return true;
	}

	// Note, instance is not yet setup so we can't use get_error_string to retrieve our error
	XrResult result = xrEnumerateApiLayerProperties(0, &num_layer_properties, nullptr);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate number of api layer properties");

	layer_properties = (XrApiLayerProperties *)memalloc(sizeof(XrApiLayerProperties) * num_layer_properties);
	ERR_FAIL_NULL_V(layer_properties, false);
	for (uint32_t i = 0; i < num_layer_properties; i++) {
		layer_properties[i].type = XR_TYPE_API_LAYER_PROPERTIES;
		layer_properties[i].next = nullptr;
	}

	result = xrEnumerateApiLayerProperties(num_layer_properties, &num_layer_properties, layer_properties);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate api layer properties");

	for (uint32_t i = 0; i < num_layer_properties; i++) {
		print_verbose(String("OpenXR: Found OpenXR layer ") + layer_properties[i].layerName);
	}

	return true;
}

bool OpenXRAPI::load_supported_extensions() {
	// This queries supported extensions that are available and can be initialized when we create our OpenXR instance

	if (supported_extensions != nullptr) {
		// already retrieved this
		return true;
	}

	// Note, instance is not yet setup so we can't use get_error_string to retrieve our error
	XrResult result = xrEnumerateInstanceExtensionProperties(nullptr, 0, &num_supported_extensions, nullptr);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate number of extension properties");

	supported_extensions = (XrExtensionProperties *)memalloc(sizeof(XrExtensionProperties) * num_supported_extensions);
	ERR_FAIL_NULL_V(supported_extensions, false);

	// set our types
	for (uint32_t i = 0; i < num_supported_extensions; i++) {
		supported_extensions[i].type = XR_TYPE_EXTENSION_PROPERTIES;
		supported_extensions[i].next = nullptr;
	}
	result = xrEnumerateInstanceExtensionProperties(nullptr, num_supported_extensions, &num_supported_extensions, supported_extensions);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate extension properties");

	for (uint32_t i = 0; i < num_supported_extensions; i++) {
		print_verbose(String("OpenXR: Found OpenXR extension ") + supported_extensions[i].extensionName);
	}

	return true;
}

bool OpenXRAPI::is_extension_supported(const String &p_extension) const {
	for (uint32_t i = 0; i < num_supported_extensions; i++) {
		if (supported_extensions[i].extensionName == p_extension) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::is_extension_enabled(const String &p_extension) const {
	CharString extension = p_extension.ascii();

	for (int i = 0; i < enabled_extensions.size(); i++) {
		if (strcmp(enabled_extensions[i].ptr(), extension.ptr()) == 0) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::is_top_level_path_supported(const String &p_toplevel_path) {
	String required_extension = OpenXRInteractionProfileMetadata::get_singleton()->get_top_level_extension(p_toplevel_path);

	// If unsupported is returned we likely have a misspelled interaction profile path in our action map. Always output that as an error.
	ERR_FAIL_COND_V_MSG(required_extension == XR_PATH_UNSUPPORTED_NAME, false, "OpenXR: Unsupported toplevel path " + p_toplevel_path);

	if (required_extension == "") {
		// no extension needed, core top level are always "supported", they just won't be used if not really supported
		return true;
	}

	if (!is_extension_enabled(required_extension)) {
		// It is very likely we have top level paths for which the extension is not available so don't flood the logs with unnecessary spam.
		print_verbose("OpenXR: Top level path " + p_toplevel_path + " requires extension " + required_extension);
		return false;
	}

	return true;
}

bool OpenXRAPI::is_interaction_profile_supported(const String &p_ip_path) {
	String required_extension = OpenXRInteractionProfileMetadata::get_singleton()->get_interaction_profile_extension(p_ip_path);

	// If unsupported is returned we likely have a misspelled interaction profile path in our action map. Always output that as an error.
	ERR_FAIL_COND_V_MSG(required_extension == XR_PATH_UNSUPPORTED_NAME, false, "OpenXR: Unsupported interaction profile " + p_ip_path);

	if (required_extension == "") {
		// no extension needed, core interaction profiles are always "supported", they just won't be used if not really supported
		return true;
	}

	if (!is_extension_enabled(required_extension)) {
		// It is very likely we have interaction profiles for which the extension is not available so don't flood the logs with unnecessary spam.
		print_verbose("OpenXR: Interaction profile " + p_ip_path + " requires extension " + required_extension);
		return false;
	}

	return true;
}

bool OpenXRAPI::interaction_profile_supports_io_path(const String &p_ip_path, const String &p_io_path) {
	if (!is_interaction_profile_supported(p_ip_path)) {
		return false;
	}

	const OpenXRInteractionProfileMetadata::IOPath *io_path = OpenXRInteractionProfileMetadata::get_singleton()->get_io_path(p_ip_path, p_io_path);

	// If the io_path is not part of our metadata we've likely got a misspelled name or a bad action map, report
	ERR_FAIL_NULL_V_MSG(io_path, false, "OpenXR: Unsupported io path " + String(p_ip_path) + String(p_io_path));

	if (io_path->openxr_extension_name == "") {
		// no extension needed, core io paths are always "supported", they just won't be used if not really supported
		return true;
	}

	if (!is_extension_enabled(io_path->openxr_extension_name)) {
		// It is very likely we have io paths for which the extension is not available so don't flood the logs with unnecessary spam.
		print_verbose("OpenXR: IO path " + String(p_ip_path) + String(p_io_path) + " requires extension " + io_path->openxr_extension_name);
		return false;
	}

	return true;
}

void OpenXRAPI::copy_string_to_char_buffer(const String p_string, char *p_buffer, int p_buffer_len) {
	CharString char_string = p_string.utf8();
	int len = char_string.length();
	if (len < p_buffer_len - 1) {
		// was having weird CI issues with strcpy so....
		memcpy(p_buffer, char_string.get_data(), len);
		p_buffer[len] = '\0';
	} else {
		memcpy(p_buffer, char_string.get_data(), p_buffer_len - 1);
		p_buffer[p_buffer_len - 1] = '\0';
	}
}

bool OpenXRAPI::create_instance() {
	// Create our OpenXR instance, this will query any registered extension wrappers for extensions we need to enable.

	// Append the extensions requested by the registered extension wrappers.
	HashMap<String, bool *> requested_extensions;
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		const HashMap<String, bool *> &wrapper_request_extensions = wrapper->get_requested_extensions();

		for (const KeyValue<String, bool *> &requested_extension : wrapper_request_extensions) {
			requested_extensions[requested_extension.key] = requested_extension.value;
		}
	}

	// Check which extensions are supported.
	enabled_extensions.clear();

	for (KeyValue<String, bool *> &requested_extension : requested_extensions) {
		if (!is_extension_supported(requested_extension.key)) {
			if (requested_extension.value == nullptr) {
				// Null means this is a mandatory extension so we fail.
				ERR_FAIL_V_MSG(false, String("OpenXR: OpenXR Runtime does not support ") + requested_extension.key + String(" extension!"));
			} else {
				// Set this extension as not supported.
				*requested_extension.value = false;
			}
		} else if (requested_extension.value != nullptr) {
			// Set this extension as supported.
			*requested_extension.value = true;

			// And record that we want to enable it.
			enabled_extensions.push_back(requested_extension.key.ascii());
		} else {
			// Record that we want to enable this.
			enabled_extensions.push_back(requested_extension.key.ascii());
		}
	}

	Vector<const char *> extension_ptrs;
	for (int i = 0; i < enabled_extensions.size(); i++) {
		print_verbose(String("OpenXR: Enabling extension ") + String(enabled_extensions[i]));
		extension_ptrs.push_back(enabled_extensions[i].get_data());
	}

	// Get our project name
	String project_name = GLOBAL_GET("application/config/name");

	// Create our OpenXR instance
	XrApplicationInfo application_info{
		"", // applicationName, we'll set this down below
		1, // applicationVersion, we don't currently have this
		"Godot Game Engine", // engineName
		VERSION_MAJOR * 10000 + VERSION_MINOR * 100 + VERSION_PATCH, // engineVersion 4.0 -> 40000, 4.0.1 -> 40001, 4.1 -> 40100, etc.
		XR_CURRENT_API_VERSION // apiVersion
	};

	void *next_pointer = nullptr;
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		void *np = wrapper->set_instance_create_info_and_get_next_pointer(next_pointer);
		if (np != nullptr) {
			next_pointer = np;
		}
	}

	XrInstanceCreateInfo instance_create_info = {
		XR_TYPE_INSTANCE_CREATE_INFO, // type
		next_pointer, // next
		0, // createFlags
		application_info, // applicationInfo
		0, // enabledApiLayerCount, need to find out if we need support for this?
		nullptr, // enabledApiLayerNames
		uint32_t(extension_ptrs.size()), // enabledExtensionCount
		extension_ptrs.ptr() // enabledExtensionNames
	};

	copy_string_to_char_buffer(project_name, instance_create_info.applicationInfo.applicationName, XR_MAX_APPLICATION_NAME_SIZE);

	XrResult result = xrCreateInstance(&instance_create_info, &instance);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "Failed to create XR instance.");

	// from this point on we can use get_error_string to get more info about our errors...

	XrInstanceProperties instanceProps = {
		XR_TYPE_INSTANCE_PROPERTIES, // type;
		nullptr, // next
		0, // runtimeVersion, from here will be set by our get call
		"" // runtimeName
	};

	OPENXR_API_INIT_XR_FUNC_V(xrGetInstanceProperties);

	result = xrGetInstanceProperties(instance, &instanceProps);
	if (XR_FAILED(result)) {
		// not fatal probably
		print_line("OpenXR: Failed to get XR instance properties [", get_error_string(result), "]");

		runtime_name = "";
		runtime_version = "";
	} else {
		runtime_name = instanceProps.runtimeName;
		runtime_version = OpenXRUtil::make_xr_version_string(instanceProps.runtimeVersion);
		print_line("OpenXR: Running on OpenXR runtime: ", runtime_name, " ", runtime_version);
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_instance_created(instance);
	}

	return true;
}

bool OpenXRAPI::get_system_info() {
	// Retrieve basic OpenXR system info based on the form factor we desire

	// Retrieve the system for our form factor, fails if form factor is not available
	XrSystemGetInfo system_get_info = {
		XR_TYPE_SYSTEM_GET_INFO, // type;
		nullptr, // next
		form_factor // formFactor
	};

	XrResult result = xrGetSystem(instance, &system_get_info, &system_id);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get system for our form factor [", get_error_string(result), "]");
		return false;
	}

	// obtain info about our system, writing this out completely to make CI on Linux happy..
	void *next_pointer = nullptr;
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		void *np = wrapper->set_system_properties_and_get_next_pointer(next_pointer);
		if (np != nullptr) {
			next_pointer = np;
		}
	}

	XrSystemProperties system_properties = {
		XR_TYPE_SYSTEM_PROPERTIES, // type
		next_pointer, // next
		0, // systemId, from here will be set by our get call
		0, // vendorId
		"", // systemName
		{
				0, // maxSwapchainImageHeight
				0, // maxSwapchainImageWidth
				0, // maxLayerCount
		}, // graphicsProperties
		{
				false, // orientationTracking
				false // positionTracking
		} // trackingProperties
	};

	result = xrGetSystemProperties(instance, system_id, &system_properties);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get System properties [", get_error_string(result), "]");
		return false;
	}

	// remember this state, we'll use it later
	system_name = String(system_properties.systemName);
	vendor_id = system_properties.vendorId;
	graphics_properties = system_properties.graphicsProperties;
	tracking_properties = system_properties.trackingProperties;

	return true;
}

bool OpenXRAPI::load_supported_view_configuration_types() {
	// This queries the supported configuration types, likely there will only be one choosing between Mono (phone AR) and Stereo (HMDs)

	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	if (supported_view_configuration_types != nullptr) {
		// free previous results
		memfree(supported_view_configuration_types);
		supported_view_configuration_types = nullptr;
	}

	XrResult result = xrEnumerateViewConfigurations(instance, system_id, 0, &num_view_configuration_types, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get view configuration count [", get_error_string(result), "]");
		return false;
	}

	supported_view_configuration_types = (XrViewConfigurationType *)memalloc(sizeof(XrViewConfigurationType) * num_view_configuration_types);
	ERR_FAIL_NULL_V(supported_view_configuration_types, false);

	result = xrEnumerateViewConfigurations(instance, system_id, num_view_configuration_types, &num_view_configuration_types, supported_view_configuration_types);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerateview configurations");
	ERR_FAIL_COND_V_MSG(num_view_configuration_types == 0, false, "OpenXR: Failed to enumerateview configurations"); // JIC there should be at least 1!

	for (uint32_t i = 0; i < num_view_configuration_types; i++) {
		print_verbose(String("OpenXR: Found supported view configuration ") + OpenXRUtil::get_view_configuration_name(supported_view_configuration_types[i]));
	}

	// Check value we loaded at startup...
	if (!is_view_configuration_supported(view_configuration)) {
		print_verbose(String("OpenXR: ") + OpenXRUtil::get_view_configuration_name(view_configuration) + String(" isn't supported, defaulting to ") + OpenXRUtil::get_view_configuration_name(supported_view_configuration_types[0]));

		view_configuration = supported_view_configuration_types[0];
	}

	return true;
}

bool OpenXRAPI::load_supported_environmental_blend_modes() {
	// This queries the supported environmental blend modes.

	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	if (supported_environment_blend_modes != nullptr) {
		// free previous results
		memfree(supported_environment_blend_modes);
		supported_environment_blend_modes = nullptr;
		num_supported_environment_blend_modes = 0;
	}

	XrResult result = xrEnumerateEnvironmentBlendModes(instance, system_id, view_configuration, 0, &num_supported_environment_blend_modes, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get supported environmental blend mode count [", get_error_string(result), "]");
		return false;
	}

	supported_environment_blend_modes = (XrEnvironmentBlendMode *)memalloc(sizeof(XrEnvironmentBlendMode) * num_supported_environment_blend_modes);
	ERR_FAIL_NULL_V(supported_environment_blend_modes, false);

	result = xrEnumerateEnvironmentBlendModes(instance, system_id, view_configuration, num_supported_environment_blend_modes, &num_supported_environment_blend_modes, supported_environment_blend_modes);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate environmental blend modes");
	ERR_FAIL_COND_V_MSG(num_supported_environment_blend_modes == 0, false, "OpenXR: Failed to enumerate environmental blend modes"); // JIC there should be at least 1!

	for (uint32_t i = 0; i < num_supported_environment_blend_modes; i++) {
		print_verbose(String("OpenXR: Found environmental blend mode ") + OpenXRUtil::get_environment_blend_mode_name(supported_environment_blend_modes[i]));
	}

	// Check value we loaded at startup...
	if (!is_environment_blend_mode_supported(environment_blend_mode)) {
		print_verbose(String("OpenXR: ") + OpenXRUtil::get_environment_blend_mode_name(environment_blend_mode) + String(" isn't supported, defaulting to ") + OpenXRUtil::get_environment_blend_mode_name(supported_environment_blend_modes[0]));

		environment_blend_mode = supported_environment_blend_modes[0];
	}

	return true;
}

bool OpenXRAPI::is_view_configuration_supported(XrViewConfigurationType p_configuration_type) const {
	ERR_FAIL_NULL_V(supported_view_configuration_types, false);

	for (uint32_t i = 0; i < num_view_configuration_types; i++) {
		if (supported_view_configuration_types[i] == p_configuration_type) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::load_supported_view_configuration_views(XrViewConfigurationType p_configuration_type) {
	// This loads our view configuration for each view so for a stereo HMD, we'll get two entries (that are likely identical)
	// The returned data supplies us with the recommended render target size

	if (!is_view_configuration_supported(p_configuration_type)) {
		print_line("OpenXR: View configuration ", OpenXRUtil::get_view_configuration_name(view_configuration), " is not supported.");
		return false;
	}

	if (view_configuration_views != nullptr) {
		// free previous results
		memfree(view_configuration_views);
		view_configuration_views = nullptr;
	}

	XrResult result = xrEnumerateViewConfigurationViews(instance, system_id, p_configuration_type, 0, &view_count, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get view configuration count [", get_error_string(result), "]");
		return false;
	}

	view_configuration_views = (XrViewConfigurationView *)memalloc(sizeof(XrViewConfigurationView) * view_count);
	ERR_FAIL_NULL_V(view_configuration_views, false);

	for (uint32_t i = 0; i < view_count; i++) {
		view_configuration_views[i].type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
		view_configuration_views[i].next = nullptr;
	}

	result = xrEnumerateViewConfigurationViews(instance, system_id, p_configuration_type, view_count, &view_count, view_configuration_views);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate view configurations");

	for (uint32_t i = 0; i < view_count; i++) {
		print_verbose("OpenXR: Found supported view configuration view");
		print_verbose(String(" - width: ") + itos(view_configuration_views[i].maxImageRectWidth));
		print_verbose(String(" - height: ") + itos(view_configuration_views[i].maxImageRectHeight));
		print_verbose(String(" - sample count: ") + itos(view_configuration_views[i].maxSwapchainSampleCount));
		print_verbose(String(" - recommended render width: ") + itos(view_configuration_views[i].recommendedImageRectWidth));
		print_verbose(String(" - recommended render height: ") + itos(view_configuration_views[i].recommendedImageRectHeight));
		print_verbose(String(" - recommended render sample count: ") + itos(view_configuration_views[i].recommendedSwapchainSampleCount));
	}

	return true;
}

void OpenXRAPI::destroy_instance() {
	if (view_configuration_views != nullptr) {
		memfree(view_configuration_views);
		view_configuration_views = nullptr;
	}

	if (supported_view_configuration_types != nullptr) {
		memfree(supported_view_configuration_types);
		supported_view_configuration_types = nullptr;
	}

	if (supported_environment_blend_modes != nullptr) {
		memfree(supported_environment_blend_modes);
		supported_environment_blend_modes = nullptr;
		num_supported_environment_blend_modes = 0;
	}

	if (instance != XR_NULL_HANDLE) {
		for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
			wrapper->on_instance_destroyed();
		}

		xrDestroyInstance(instance);
		instance = XR_NULL_HANDLE;
	}
	enabled_extensions.clear();

	if (graphics_extension != nullptr) {
		unregister_extension_wrapper(graphics_extension);
		memdelete(graphics_extension);
		graphics_extension = nullptr;
	}
}

bool OpenXRAPI::create_session() {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);
	ERR_FAIL_COND_V(session != XR_NULL_HANDLE, false);

	void *next_pointer = nullptr;
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		void *np = wrapper->set_session_create_and_get_next_pointer(next_pointer);
		if (np != nullptr) {
			next_pointer = np;
		}
	}

	XrSessionCreateInfo session_create_info = {
		XR_TYPE_SESSION_CREATE_INFO, // type
		next_pointer, // next
		0, // createFlags
		system_id // systemId
	};

	XrResult result = xrCreateSession(instance, &session_create_info, &session);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to create session [", get_error_string(result), "]");
		return false;
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_session_created(session);
	}

	return true;
}

bool OpenXRAPI::load_supported_reference_spaces() {
	// loads the supported reference spaces for our OpenXR session

	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	if (supported_reference_spaces != nullptr) {
		// free previous results
		memfree(supported_reference_spaces);
		supported_reference_spaces = nullptr;
	}

	XrResult result = xrEnumerateReferenceSpaces(session, 0, &num_reference_spaces, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get reference space count [", get_error_string(result), "]");
		return false;
	}

	supported_reference_spaces = (XrReferenceSpaceType *)memalloc(sizeof(XrReferenceSpaceType) * num_reference_spaces);
	ERR_FAIL_NULL_V(supported_reference_spaces, false);

	result = xrEnumerateReferenceSpaces(session, num_reference_spaces, &num_reference_spaces, supported_reference_spaces);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate reference spaces");
	ERR_FAIL_COND_V_MSG(num_reference_spaces == 0, false, "OpenXR: Failed to enumerate reference spaces");

	for (uint32_t i = 0; i < num_reference_spaces; i++) {
		print_verbose(String("OpenXR: Found supported reference space ") + OpenXRUtil::get_reference_space_name(supported_reference_spaces[i]));
	}

	// Check value we loaded at startup...
	if (!is_reference_space_supported(reference_space)) {
		print_verbose(String("OpenXR: ") + OpenXRUtil::get_reference_space_name(reference_space) + String(" isn't supported, defaulting to ") + OpenXRUtil::get_reference_space_name(supported_reference_spaces[0]));

		reference_space = supported_reference_spaces[0];
	}

	return true;
}

bool OpenXRAPI::is_reference_space_supported(XrReferenceSpaceType p_reference_space) {
	ERR_FAIL_NULL_V(supported_reference_spaces, false);

	for (uint32_t i = 0; i < num_reference_spaces; i++) {
		if (supported_reference_spaces[i] == p_reference_space) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::setup_spaces() {
	XrResult result;

	XrPosef identityPose = {
		{ 0.0, 0.0, 0.0, 1.0 },
		{ 0.0, 0.0, 0.0 }
	};

	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	// create play space
	{
		if (!is_reference_space_supported(reference_space)) {
			print_line("OpenXR: reference space ", OpenXRUtil::get_reference_space_name(reference_space), " is not supported.");
			return false;
		}

		XrReferenceSpaceCreateInfo play_space_create_info = {
			XR_TYPE_REFERENCE_SPACE_CREATE_INFO, // type
			nullptr, // next
			reference_space, // referenceSpaceType
			identityPose // poseInReferenceSpace
		};

		result = xrCreateReferenceSpace(session, &play_space_create_info, &play_space);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to create play space [", get_error_string(result), "]");
			return false;
		}
	}

	// create view space
	{
		if (!is_reference_space_supported(XR_REFERENCE_SPACE_TYPE_VIEW)) {
			print_line("OpenXR: reference space XR_REFERENCE_SPACE_TYPE_VIEW is not supported.");
			return false;
		}

		XrReferenceSpaceCreateInfo view_space_create_info = {
			XR_TYPE_REFERENCE_SPACE_CREATE_INFO, // type
			nullptr, // next
			XR_REFERENCE_SPACE_TYPE_VIEW, // referenceSpaceType
			identityPose // poseInReferenceSpace
		};

		result = xrCreateReferenceSpace(session, &view_space_create_info, &view_space);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to create view space [", get_error_string(result), "]");
			return false;
		}
	}

	return true;
}

bool OpenXRAPI::load_supported_swapchain_formats() {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	if (supported_swapchain_formats != nullptr) {
		// free previous results
		memfree(supported_swapchain_formats);
		supported_swapchain_formats = nullptr;
	}

	XrResult result = xrEnumerateSwapchainFormats(session, 0, &num_swapchain_formats, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get swapchain format count [", get_error_string(result), "]");
		return false;
	}

	supported_swapchain_formats = (int64_t *)memalloc(sizeof(int64_t) * num_swapchain_formats);
	ERR_FAIL_NULL_V(supported_swapchain_formats, false);

	result = xrEnumerateSwapchainFormats(session, num_swapchain_formats, &num_swapchain_formats, supported_swapchain_formats);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), false, "OpenXR: Failed to enumerate swapchain formats");

	for (uint32_t i = 0; i < num_swapchain_formats; i++) {
		print_verbose(String("OpenXR: Found supported swapchain format ") + get_swapchain_format_name(supported_swapchain_formats[i]));
	}

	return true;
}

bool OpenXRAPI::is_swapchain_format_supported(int64_t p_swapchain_format) {
	ERR_FAIL_NULL_V(supported_swapchain_formats, false);

	for (uint32_t i = 0; i < num_swapchain_formats; i++) {
		if (supported_swapchain_formats[i] == p_swapchain_format) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::create_swapchains() {
	ERR_FAIL_NULL_V(graphics_extension, false);
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	/*
		TODO: We need to improve on this, for now we're taking our old approach of creating our main swapchains and substituting
		those for the ones Godot normally creates.
		This however means we can only use swapchains for our main XR view.

		It would have been nicer if we could override the swapchain creation in Godot with ours but we have a timing issue here.
		We can't create XR swapchains until after our XR session is fully instantiated, yet Godot creates its swapchain much earlier.

		Also Godot only creates a swapchain for the main output.
		OpenXR will require us to create swapchains as the render target for additional viewports if we want to use the layer system
		to optimize text rendering and background rendering as OpenXR may choose to reuse the results for reprojection while we're
		already rendering the next frame.

		Finally an area we need to expand upon is that Foveated rendering is only enabled for the swap chain we create,
		as we render 3D content into internal buffers that are copied into the swapchain, we do now have (basic) VRS support
	*/

	Size2 recommended_size = get_recommended_target_size();
	uint32_t sample_count = 1;

	// We start with our color swapchain...
	{
		// Build a vector with swapchain formats we want to use, from best fit to worst
		Vector<int64_t> usable_swapchain_formats;
		int64_t swapchain_format_to_use = 0;

		graphics_extension->get_usable_swapchain_formats(usable_swapchain_formats);

		// now find out which one is supported
		for (int i = 0; i < usable_swapchain_formats.size() && swapchain_format_to_use == 0; i++) {
			if (is_swapchain_format_supported(usable_swapchain_formats[i])) {
				swapchain_format_to_use = usable_swapchain_formats[i];
			}
		}

		if (swapchain_format_to_use == 0) {
			swapchain_format_to_use = usable_swapchain_formats[0]; // just use the first one and hope for the best...
			print_line("Couldn't find usable color swap chain format, using", get_swapchain_format_name(swapchain_format_to_use), "instead.");
		} else {
			print_verbose(String("Using color swap chain format:") + get_swapchain_format_name(swapchain_format_to_use));
		}

		if (!create_swapchain(XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, swapchain_format_to_use, recommended_size.width, recommended_size.height, sample_count, view_count, swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain, &swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain_graphics_data)) {
			return false;
		}
	}

	views = (XrView *)memalloc(sizeof(XrView) * view_count);
	ERR_FAIL_NULL_V_MSG(views, false, "OpenXR Couldn't allocate memory for views");

	projection_views = (XrCompositionLayerProjectionView *)memalloc(sizeof(XrCompositionLayerProjectionView) * view_count);
	ERR_FAIL_NULL_V_MSG(projection_views, false, "OpenXR Couldn't allocate memory for projection views");

	// We create our depth swapchain if:
	// - we've enabled submitting depth buffer
	// - we support our depth layer extension
	// - we have our spacewarp extension (not yet implemented)
	if (submit_depth_buffer && OpenXRCompositionLayerDepthExtension::get_singleton()->is_available()) {
		// Build a vector with swapchain formats we want to use, from best fit to worst
		Vector<int64_t> usable_swapchain_formats;
		int64_t swapchain_format_to_use = 0;

		graphics_extension->get_usable_depth_formats(usable_swapchain_formats);

		// now find out which one is supported
		for (int i = 0; i < usable_swapchain_formats.size() && swapchain_format_to_use == 0; i++) {
			if (is_swapchain_format_supported(usable_swapchain_formats[i])) {
				swapchain_format_to_use = usable_swapchain_formats[i];
			}
		}

		if (swapchain_format_to_use == 0) {
			print_line("Couldn't find usable depth swap chain format, depth buffer will not be submitted.");
		} else {
			print_verbose(String("Using depth swap chain format:") + get_swapchain_format_name(swapchain_format_to_use));

			// Note, if VK_FORMAT_D32_SFLOAT is used here but we're using the forward+ renderer, we should probably output a warning.

			if (!create_swapchain(XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, swapchain_format_to_use, recommended_size.width, recommended_size.height, sample_count, view_count, swapchains[OPENXR_SWAPCHAIN_DEPTH].swapchain, &swapchains[OPENXR_SWAPCHAIN_DEPTH].swapchain_graphics_data)) {
				return false;
			}

			depth_views = (XrCompositionLayerDepthInfoKHR *)memalloc(sizeof(XrCompositionLayerDepthInfoKHR) * view_count);
			ERR_FAIL_NULL_V_MSG(depth_views, false, "OpenXR Couldn't allocate memory for depth views");
		}
	}

	// We create our velocity swapchain if:
	// - we have our spacewarp extension (not yet implemented)
	{
		// TBD
	}

	for (uint32_t i = 0; i < view_count; i++) {
		views[i].type = XR_TYPE_VIEW;
		views[i].next = nullptr;

		projection_views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
		projection_views[i].next = nullptr;
		projection_views[i].subImage.swapchain = swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain;
		projection_views[i].subImage.imageArrayIndex = i;
		projection_views[i].subImage.imageRect.offset.x = 0;
		projection_views[i].subImage.imageRect.offset.y = 0;
		projection_views[i].subImage.imageRect.extent.width = recommended_size.width;
		projection_views[i].subImage.imageRect.extent.height = recommended_size.height;

		if (submit_depth_buffer && OpenXRCompositionLayerDepthExtension::get_singleton()->is_available() && depth_views) {
			projection_views[i].next = &depth_views[i];

			depth_views[i].type = XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR;
			depth_views[i].next = nullptr;
			depth_views[i].subImage.swapchain = swapchains[OPENXR_SWAPCHAIN_DEPTH].swapchain;
			depth_views[i].subImage.imageArrayIndex = i;
			depth_views[i].subImage.imageRect.offset.x = 0;
			depth_views[i].subImage.imageRect.offset.y = 0;
			depth_views[i].subImage.imageRect.extent.width = recommended_size.width;
			depth_views[i].subImage.imageRect.extent.height = recommended_size.height;
			depth_views[i].minDepth = 0.0;
			depth_views[i].maxDepth = 1.0;
			depth_views[i].nearZ = 0.01; // Near and far Z will be set to the correct values in fill_projection_matrix
			depth_views[i].farZ = 100.0;
		}
	};

	return true;
};

void OpenXRAPI::destroy_session() {
	if (running && session != XR_NULL_HANDLE) {
		xrEndSession(session);
	}

	if (graphics_extension) {
		graphics_extension->cleanup_swapchain_graphics_data(&swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain_graphics_data);
	}

	if (views != nullptr) {
		memfree(views);
		views = nullptr;
	}

	if (projection_views != nullptr) {
		memfree(projection_views);
		projection_views = nullptr;
	}

	if (depth_views != nullptr) {
		memfree(depth_views);
		depth_views = nullptr;
	}

	for (int i = 0; i < OPENXR_SWAPCHAIN_MAX; i++) {
		if (swapchains[i].swapchain != XR_NULL_HANDLE) {
			xrDestroySwapchain(swapchains[i].swapchain);
			swapchains[i].swapchain = XR_NULL_HANDLE;
		}
	}

	if (supported_swapchain_formats != nullptr) {
		memfree(supported_swapchain_formats);
		supported_swapchain_formats = nullptr;
	}

	// destroy our spaces
	if (play_space != XR_NULL_HANDLE) {
		xrDestroySpace(play_space);
		play_space = XR_NULL_HANDLE;
	}
	if (view_space != XR_NULL_HANDLE) {
		xrDestroySpace(view_space);
		view_space = XR_NULL_HANDLE;
	}

	if (supported_reference_spaces != nullptr) {
		// free previous results
		memfree(supported_reference_spaces);
		supported_reference_spaces = nullptr;
	}

	if (session != XR_NULL_HANDLE) {
		for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
			wrapper->on_session_destroyed();
		}

		xrDestroySession(session);
		session = XR_NULL_HANDLE;
	}
}

bool OpenXRAPI::create_swapchain(XrSwapchainUsageFlags p_usage_flags, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size, XrSwapchain &r_swapchain, void **r_swapchain_graphics_data) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);
	ERR_FAIL_NULL_V(graphics_extension, false);

	XrResult result;

	void *next_pointer = nullptr;
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		void *np = wrapper->set_swapchain_create_info_and_get_next_pointer(next_pointer);
		if (np != nullptr) {
			next_pointer = np;
		}
	}

	XrSwapchainCreateInfo swapchain_create_info = {
		XR_TYPE_SWAPCHAIN_CREATE_INFO, // type
		next_pointer, // next
		0, // createFlags
		p_usage_flags, // usageFlags
		p_swapchain_format, // format
		p_sample_count, // sampleCount
		p_width, // width
		p_height, // height
		1, // faceCount
		p_array_size, // arraySize
		1 // mipCount
	};

	XrSwapchain new_swapchain;
	result = xrCreateSwapchain(session, &swapchain_create_info, &new_swapchain);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get swapchain [", get_error_string(result), "]");
		return false;
	}

	if (!graphics_extension->get_swapchain_image_data(new_swapchain, p_swapchain_format, p_width, p_height, p_sample_count, p_array_size, r_swapchain_graphics_data)) {
		xrDestroySwapchain(new_swapchain);
		return false;
	}

	r_swapchain = new_swapchain;

	return true;
}

bool OpenXRAPI::on_state_idle() {
	print_verbose("On state idle");

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_idle();
	}

	return true;
}

bool OpenXRAPI::on_state_ready() {
	print_verbose("On state ready");

	// begin session
	XrSessionBeginInfo session_begin_info = {
		XR_TYPE_SESSION_BEGIN_INFO, // type
		nullptr, // next
		view_configuration // primaryViewConfigurationType
	};

	XrResult result = xrBeginSession(session, &session_begin_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to begin session [", get_error_string(result), "]");
		return false;
	}

	// This is when we create our swapchain, this can be a "long" time after Godot finishes, we can deal with this for now
	// but once we want to provide Viewports for additional layers where OpenXR requires us to create further swapchains,
	// we'll be creating those viewport WAY before we reach this point.
	// We may need to implement a wait in our init in main.cpp polling our events until the session is ready.
	// That will be very very ugly
	// The other possibility is to create a separate OpenXRViewport type specifically for this goal as part of our OpenXR module

	if (!create_swapchains()) {
		return false;
	}

	// we're running
	running = true;

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_ready();
	}

	if (xr_interface) {
		xr_interface->on_state_ready();
	}

	// TODO Tell android

	return true;
}

bool OpenXRAPI::on_state_synchronized() {
	print_verbose("On state synchronized");

	// Just in case, see if we already have active trackers...
	List<RID> trackers;
	tracker_owner.get_owned_list(&trackers);
	for (int i = 0; i < trackers.size(); i++) {
		tracker_check_profile(trackers[i]);
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_synchronized();
	}

	return true;
}

bool OpenXRAPI::on_state_visible() {
	print_verbose("On state visible");

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_visible();
	}

	if (xr_interface) {
		xr_interface->on_state_visible();
	}

	return true;
}

bool OpenXRAPI::on_state_focused() {
	print_verbose("On state focused");

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_focused();
	}

	if (xr_interface) {
		xr_interface->on_state_focused();
	}

	return true;
}

bool OpenXRAPI::on_state_stopping() {
	print_verbose("On state stopping");

	if (xr_interface) {
		xr_interface->on_state_stopping();
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_stopping();
	}

	if (running) {
		XrResult result = xrEndSession(session);
		if (XR_FAILED(result)) {
			// we only report this..
			print_line("OpenXR: Failed to end session [", get_error_string(result), "]");
		}

		running = false;
	}

	// TODO further cleanup

	return true;
}

bool OpenXRAPI::on_state_loss_pending() {
	print_verbose("On state loss pending");

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_loss_pending();
	}

	// TODO need to look into the correct action here, read up on the spec but we may need to signal Godot to exit (if it's not already exiting)

	return true;
}

bool OpenXRAPI::on_state_exiting() {
	print_verbose("On state existing");

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_state_exiting();
	}

	// TODO need to look into the correct action here, read up on the spec but we may need to signal Godot to exit (if it's not already exiting)

	return true;
}

void OpenXRAPI::set_form_factor(XrFormFactor p_form_factor) {
	ERR_FAIL_COND(is_initialized());

	form_factor = p_form_factor;
}

void OpenXRAPI::set_view_configuration(XrViewConfigurationType p_view_configuration) {
	ERR_FAIL_COND(is_initialized());

	view_configuration = p_view_configuration;
}

void OpenXRAPI::set_reference_space(XrReferenceSpaceType p_reference_space) {
	ERR_FAIL_COND(is_initialized());

	reference_space = p_reference_space;
}

void OpenXRAPI::set_submit_depth_buffer(bool p_submit_depth_buffer) {
	ERR_FAIL_COND(is_initialized());

	submit_depth_buffer = p_submit_depth_buffer;
}

bool OpenXRAPI::is_initialized() {
	return (instance != XR_NULL_HANDLE);
}

bool OpenXRAPI::is_running() {
	if (instance == XR_NULL_HANDLE) {
		return false;
	}
	if (session == XR_NULL_HANDLE) {
		return false;
	}

	return running;
}

bool OpenXRAPI::openxr_loader_init() {
#ifdef ANDROID_ENABLED
	ERR_FAIL_COND_V_MSG(openxr_loader_library_handle != nullptr, false, "OpenXR Loader library is already loaded.");

	{
		Error error_code = OS::get_singleton()->open_dynamic_library(OPENXR_LOADER_NAME, openxr_loader_library_handle);
		ERR_FAIL_COND_V_MSG(error_code != OK, false, "OpenXR loader not found.");
	}

	{
		Error error_code = OS::get_singleton()->get_dynamic_library_symbol_handle(openxr_loader_library_handle, "xrGetInstanceProcAddr", (void *&)xrGetInstanceProcAddr);
		ERR_FAIL_COND_V_MSG(error_code != OK, false, "Symbol xrGetInstanceProcAddr not found in OpenXR Loader library.");
	}
#endif

	// Resolve the symbols that don't require an instance
	OPENXR_API_INIT_XR_FUNC_V(xrCreateInstance);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateApiLayerProperties);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateInstanceExtensionProperties);

	return true;
}

bool OpenXRAPI::resolve_instance_openxr_symbols() {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	OPENXR_API_INIT_XR_FUNC_V(xrAcquireSwapchainImage);
	OPENXR_API_INIT_XR_FUNC_V(xrApplyHapticFeedback);
	OPENXR_API_INIT_XR_FUNC_V(xrAttachSessionActionSets);
	OPENXR_API_INIT_XR_FUNC_V(xrBeginFrame);
	OPENXR_API_INIT_XR_FUNC_V(xrBeginSession);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateAction);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateActionSet);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateActionSpace);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateReferenceSpace);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateSession);
	OPENXR_API_INIT_XR_FUNC_V(xrCreateSwapchain);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroyAction);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroyActionSet);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroyInstance);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroySession);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroySpace);
	OPENXR_API_INIT_XR_FUNC_V(xrDestroySwapchain);
	OPENXR_API_INIT_XR_FUNC_V(xrEndFrame);
	OPENXR_API_INIT_XR_FUNC_V(xrEndSession);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateEnvironmentBlendModes);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateReferenceSpaces);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateSwapchainFormats);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateViewConfigurations);
	OPENXR_API_INIT_XR_FUNC_V(xrEnumerateViewConfigurationViews);
	OPENXR_API_INIT_XR_FUNC_V(xrGetActionStateBoolean);
	OPENXR_API_INIT_XR_FUNC_V(xrGetActionStateFloat);
	OPENXR_API_INIT_XR_FUNC_V(xrGetActionStateVector2f);
	OPENXR_API_INIT_XR_FUNC_V(xrGetCurrentInteractionProfile);
	OPENXR_API_INIT_XR_FUNC_V(xrGetSystem);
	OPENXR_API_INIT_XR_FUNC_V(xrGetSystemProperties);
	OPENXR_API_INIT_XR_FUNC_V(xrLocateViews);
	OPENXR_API_INIT_XR_FUNC_V(xrLocateSpace);
	OPENXR_API_INIT_XR_FUNC_V(xrPathToString);
	OPENXR_API_INIT_XR_FUNC_V(xrPollEvent);
	OPENXR_API_INIT_XR_FUNC_V(xrReleaseSwapchainImage);
	OPENXR_API_INIT_XR_FUNC_V(xrResultToString);
	OPENXR_API_INIT_XR_FUNC_V(xrStringToPath);
	OPENXR_API_INIT_XR_FUNC_V(xrSuggestInteractionProfileBindings);
	OPENXR_API_INIT_XR_FUNC_V(xrSyncActions);
	OPENXR_API_INIT_XR_FUNC_V(xrWaitFrame);
	OPENXR_API_INIT_XR_FUNC_V(xrWaitSwapchainImage);

	return true;
}

XrResult OpenXRAPI::try_get_instance_proc_addr(const char *p_name, PFN_xrVoidFunction *p_addr) {
	return xrGetInstanceProcAddr(instance, p_name, p_addr);
}

XrResult OpenXRAPI::get_instance_proc_addr(const char *p_name, PFN_xrVoidFunction *p_addr) {
	XrResult result = try_get_instance_proc_addr(p_name, p_addr);

	if (result != XR_SUCCESS) {
		String error_message = String("Symbol ") + p_name + " not found in OpenXR instance.";
		ERR_FAIL_V_MSG(result, error_message.utf8().get_data());
	}

	return result;
}

bool OpenXRAPI::initialize(const String &p_rendering_driver) {
	ERR_FAIL_COND_V_MSG(instance != XR_NULL_HANDLE, false, "OpenXR instance was already created");

	if (!openxr_loader_init()) {
		return false;
	}

	if (p_rendering_driver == "vulkan") {
#ifdef VULKAN_ENABLED
		graphics_extension = memnew(OpenXRVulkanExtension);
		register_extension_wrapper(graphics_extension);
#else
		// shouldn't be possible...
		ERR_FAIL_V(false);
#endif
	} else if (p_rendering_driver == "opengl3") {
#if defined(GLES3_ENABLED) && !defined(MACOS_ENABLED)
		graphics_extension = memnew(OpenXROpenGLExtension);
		register_extension_wrapper(graphics_extension);
#else
		// shouldn't be possible...
		ERR_FAIL_V(false);
#endif
	} else {
		ERR_FAIL_V_MSG(false, "OpenXR: Unsupported rendering device.");
	}

	// Also register our rendering extensions
	register_extension_wrapper(memnew(OpenXRFBUpdateSwapchainExtension(p_rendering_driver)));
	register_extension_wrapper(memnew(OpenXRFBFoveationExtension(p_rendering_driver)));

	// initialize
	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_before_instance_created();
	}

	if (!load_layer_properties()) {
		destroy_instance();
		return false;
	}

	if (!load_supported_extensions()) {
		destroy_instance();
		return false;
	}

	if (!create_instance()) {
		destroy_instance();
		return false;
	}

	if (!resolve_instance_openxr_symbols()) {
		destroy_instance();
		return false;
	}

	if (!get_system_info()) {
		destroy_instance();
		return false;
	}

	if (!load_supported_view_configuration_types()) {
		destroy_instance();
		return false;
	}

	if (!load_supported_view_configuration_views(view_configuration)) {
		destroy_instance();
		return false;
	}

	if (!load_supported_environmental_blend_modes()) {
		destroy_instance();
		return false;
	}

	return true;
}

bool OpenXRAPI::initialize_session() {
	if (!create_session()) {
		destroy_session();
		return false;
	}

	if (!load_supported_reference_spaces()) {
		destroy_session();
		return false;
	}

	if (!setup_spaces()) {
		destroy_session();
		return false;
	}

	if (!load_supported_swapchain_formats()) {
		destroy_session();
		return false;
	}

	return true;
}

void OpenXRAPI::finish() {
	destroy_session();

	destroy_instance();
}

void OpenXRAPI::set_xr_interface(OpenXRInterface *p_xr_interface) {
	xr_interface = p_xr_interface;
}

void OpenXRAPI::register_extension_wrapper(OpenXRExtensionWrapper *p_extension_wrapper) {
	registered_extension_wrappers.push_back(p_extension_wrapper);
}

void OpenXRAPI::unregister_extension_wrapper(OpenXRExtensionWrapper *p_extension_wrapper) {
	registered_extension_wrappers.erase(p_extension_wrapper);
}

void OpenXRAPI::register_extension_metadata() {
	for (OpenXRExtensionWrapper *extension_wrapper : registered_extension_wrappers) {
		extension_wrapper->on_register_metadata();
	}
}

void OpenXRAPI::cleanup_extension_wrappers() {
	for (OpenXRExtensionWrapper *extension_wrapper : registered_extension_wrappers) {
		// Fix crash when the extension wrapper comes from GDExtension.
		OpenXRExtensionWrapperExtension *gdextension_extension_wrapper = dynamic_cast<OpenXRExtensionWrapperExtension *>(extension_wrapper);
		if (gdextension_extension_wrapper) {
			memdelete(gdextension_extension_wrapper);
		} else {
			memdelete(extension_wrapper);
		}
	}
	registered_extension_wrappers.clear();
}

Size2 OpenXRAPI::get_recommended_target_size() {
	ERR_FAIL_NULL_V(view_configuration_views, Size2());

	Size2 target_size;

	target_size.width = view_configuration_views[0].recommendedImageRectWidth * render_target_size_multiplier;
	target_size.height = view_configuration_views[0].recommendedImageRectHeight * render_target_size_multiplier;

	return target_size;
}

XRPose::TrackingConfidence OpenXRAPI::get_head_center(Transform3D &r_transform, Vector3 &r_linear_velocity, Vector3 &r_angular_velocity) {
	XrResult result;

	if (!running) {
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	// xrWaitFrame not run yet
	if (frame_state.predictedDisplayTime == 0) {
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	// Get timing for the next frame, as that is the current frame we're processing
	XrTime display_time = get_next_frame_time();

	XrSpaceVelocity velocity = {
		XR_TYPE_SPACE_VELOCITY, // type
		nullptr, // next
		0, // velocityFlags
		{ 0.0, 0.0, 0.0 }, // linearVelocity
		{ 0.0, 0.0, 0.0 } // angularVelocity
	};

	XrSpaceLocation location = {
		XR_TYPE_SPACE_LOCATION, // type
		&velocity, // next
		0, // locationFlags
		{
				{ 0.0, 0.0, 0.0, 0.0 }, // orientation
				{ 0.0, 0.0, 0.0 } // position
		} // pose
	};

	result = xrLocateSpace(view_space, play_space, display_time, &location);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to locate view space in play space [", get_error_string(result), "]");
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	XRPose::TrackingConfidence confidence = transform_from_location(location, r_transform);
	parse_velocities(velocity, r_linear_velocity, r_angular_velocity);

	if (head_pose_confidence != confidence) {
		// prevent error spam
		head_pose_confidence = confidence;
		if (head_pose_confidence == XRPose::XR_TRACKING_CONFIDENCE_NONE) {
			print_line("OpenXR head space location not valid (check tracking?)");
		} else if (head_pose_confidence == XRPose::XR_TRACKING_CONFIDENCE_LOW) {
			print_verbose("OpenVR Head pose now tracking with low confidence");
		} else {
			print_verbose("OpenVR Head pose now tracking with high confidence");
		}
	}

	return confidence;
}

bool OpenXRAPI::get_view_transform(uint32_t p_view, Transform3D &r_transform) {
	if (!running) {
		return false;
	}

	// xrWaitFrame not run yet
	if (frame_state.predictedDisplayTime == 0) {
		return false;
	}

	// we don't have valid view info
	if (views == nullptr || !view_pose_valid) {
		return false;
	}

	// Note, the timing of this is set right before rendering, which is what we need here.
	r_transform = transform_from_pose(views[p_view].pose);

	return true;
}

bool OpenXRAPI::get_view_projection(uint32_t p_view, double p_z_near, double p_z_far, Projection &p_camera_matrix) {
	ERR_FAIL_NULL_V(graphics_extension, false);

	if (!running) {
		return false;
	}

	// xrWaitFrame not run yet
	if (frame_state.predictedDisplayTime == 0) {
		return false;
	}

	// we don't have valid view info
	if (views == nullptr || !view_pose_valid) {
		return false;
	}

	// if we're using depth views, make sure we update our near and far there...
	if (depth_views != nullptr) {
		for (uint32_t i = 0; i < view_count; i++) {
			depth_views[i].nearZ = p_z_near;
			depth_views[i].farZ = p_z_far;
		}
	}

	// now update our projection
	return graphics_extension->create_projection_fov(views[p_view].fov, p_z_near, p_z_far, p_camera_matrix);
}

bool OpenXRAPI::poll_events() {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	XrEventDataBuffer runtimeEvent;
	runtimeEvent.type = XR_TYPE_EVENT_DATA_BUFFER;
	runtimeEvent.next = nullptr;
	// runtimeEvent.varying = ...

	XrResult pollResult = xrPollEvent(instance, &runtimeEvent);
	while (pollResult == XR_SUCCESS) {
		bool handled = false;
		for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
			handled |= wrapper->on_event_polled(runtimeEvent);
		}
		switch (runtimeEvent.type) {
			case XR_TYPE_EVENT_DATA_EVENTS_LOST: {
				XrEventDataEventsLost *event = (XrEventDataEventsLost *)&runtimeEvent;

				// We probably didn't poll fast enough, just output warning
				WARN_PRINT("OpenXR EVENT: " + itos(event->lostEventCount) + " event data lost!");
			} break;
			case XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR: {
				// XrEventDataVisibilityMaskChangedKHR *event = (XrEventDataVisibilityMaskChangedKHR *)&runtimeEvent;

				// TODO implement this in the future, we should call xrGetVisibilityMaskKHR to obtain a mask,
				// this will allow us to prevent rendering the part of our view which is never displayed giving us
				// a decent performance improvement.

				print_verbose("OpenXR EVENT: STUB: visibility mask changed");
			} break;
			case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
				XrEventDataInstanceLossPending *event = (XrEventDataInstanceLossPending *)&runtimeEvent;

				// TODO We get this event if we're about to loose our OpenXR instance.
				// We should queue exiting Godot at this point.

				print_verbose(String("OpenXR EVENT: instance loss pending at ") + itos(event->lossTime));
				return false;
			} break;
			case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
				XrEventDataSessionStateChanged *event = (XrEventDataSessionStateChanged *)&runtimeEvent;

				session_state = event->state;
				if (session_state >= XR_SESSION_STATE_MAX_ENUM) {
					print_verbose(String("OpenXR EVENT: session state changed to UNKNOWN - ") + itos(session_state));
				} else {
					print_verbose(String("OpenXR EVENT: session state changed to ") + OpenXRUtil::get_session_state_name(session_state));

					switch (session_state) {
						case XR_SESSION_STATE_IDLE:
							on_state_idle();
							break;
						case XR_SESSION_STATE_READY:
							on_state_ready();
							break;
						case XR_SESSION_STATE_SYNCHRONIZED:
							on_state_synchronized();
							break;
						case XR_SESSION_STATE_VISIBLE:
							on_state_visible();
							break;
						case XR_SESSION_STATE_FOCUSED:
							on_state_focused();
							break;
						case XR_SESSION_STATE_STOPPING:
							on_state_stopping();
							break;
						case XR_SESSION_STATE_LOSS_PENDING:
							on_state_loss_pending();
							break;
						case XR_SESSION_STATE_EXITING:
							on_state_exiting();
							break;
						default:
							break;
					}
				}
			} break;
			case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING: {
				XrEventDataReferenceSpaceChangePending *event = (XrEventDataReferenceSpaceChangePending *)&runtimeEvent;

				print_verbose(String("OpenXR EVENT: reference space type ") + OpenXRUtil::get_reference_space_name(event->referenceSpaceType) + " change pending!");
				if (event->poseValid && xr_interface) {
					xr_interface->on_pose_recentered();
				}
			} break;
			case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED: {
				print_verbose("OpenXR EVENT: interaction profile changed!");

				XrEventDataInteractionProfileChanged *event = (XrEventDataInteractionProfileChanged *)&runtimeEvent;

				List<RID> trackers;
				tracker_owner.get_owned_list(&trackers);
				for (int i = 0; i < trackers.size(); i++) {
					tracker_check_profile(trackers[i], event->session);
				}

			} break;
			default:
				if (!handled) {
					print_verbose(String("OpenXR Unhandled event type ") + OpenXRUtil::get_structure_type_name(runtimeEvent.type));
				}
				break;
		}

		runtimeEvent.type = XR_TYPE_EVENT_DATA_BUFFER;
		pollResult = xrPollEvent(instance, &runtimeEvent);
	}

	if (pollResult == XR_EVENT_UNAVAILABLE) {
		// processed all events in the queue
		return true;
	} else {
		ERR_FAIL_V_MSG(false, "OpenXR: Failed to poll events!");
	}
}

bool OpenXRAPI::process() {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	if (!poll_events()) {
		return false;
	}

	if (!running) {
		return false;
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_process();
	}

	return true;
}

bool OpenXRAPI::acquire_image(OpenXRSwapChainInfo &p_swapchain) {
	ERR_FAIL_COND_V(p_swapchain.image_acquired, true); // This was not released when it should be, error out and reuse...

	XrResult result;

	if (!p_swapchain.skip_acquire_swapchain) {
		XrSwapchainImageAcquireInfo swapchain_image_acquire_info = {
			XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO, // type
			nullptr // next
		};

		result = xrAcquireSwapchainImage(p_swapchain.swapchain, &swapchain_image_acquire_info, &p_swapchain.image_index);
		if (!XR_UNQUALIFIED_SUCCESS(result)) {
			// Make sure end_frame knows we need to submit an empty frame
			frame_state.shouldRender = false;

			if (XR_FAILED(result)) {
				// Unexpected failure, log this!
				print_line("OpenXR: failed to acquire swapchain image [", get_error_string(result), "]");
				return false;
			} else {
				// In this scenario we silently fail, the XR runtime is simply not ready yet to acquire the swapchain.
				return false;
			}
		}
	}

	XrSwapchainImageWaitInfo swapchain_image_wait_info = {
		XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, // type
		nullptr, // next
		17000000 // timeout in nanoseconds
	};

	result = xrWaitSwapchainImage(p_swapchain.swapchain, &swapchain_image_wait_info);
	if (!XR_UNQUALIFIED_SUCCESS(result)) {
		// Make sure end_frame knows we need to submit an empty frame
		frame_state.shouldRender = false;

		if (XR_FAILED(result)) {
			// Unexpected failure, log this!
			print_line("OpenXR: failed to wait for swapchain image [", get_error_string(result), "]");
			return false;
		} else {
			// Make sure to skip trying to acquire the swapchain image in the next frame
			p_swapchain.skip_acquire_swapchain = true;
			return false;
		}
	} else {
		p_swapchain.skip_acquire_swapchain = false;
	}

	return true;
}

bool OpenXRAPI::release_image(OpenXRSwapChainInfo &p_swapchain) {
	XrSwapchainImageReleaseInfo swapchain_image_release_info = {
		XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO, // type
		nullptr // next
	};
	XrResult result = xrReleaseSwapchainImage(p_swapchain.swapchain, &swapchain_image_release_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to release swapchain image! [", get_error_string(result), "]");
		return false;
	}

	return true;
}

void OpenXRAPI::pre_render() {
	ERR_FAIL_COND(instance == XR_NULL_HANDLE);

	if (!running) {
		return;
	}

	// Waitframe does 2 important things in our process:
	// 1) It provides us with predictive timing, telling us when OpenXR expects to display the frame we're about to commit
	// 2) It will use the previous timing to pause our thread so that rendering starts as close to displaying as possible
	// This must thus be called as close to when we start rendering as possible
	XrFrameWaitInfo frame_wait_info = { XR_TYPE_FRAME_WAIT_INFO, nullptr };
	frame_state.predictedDisplayTime = 0;
	frame_state.predictedDisplayPeriod = 0;
	frame_state.shouldRender = false;

	XrResult result = xrWaitFrame(session, &frame_wait_info, &frame_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: xrWaitFrame() was not successful [", get_error_string(result), "]");

		// reset just in case
		frame_state.predictedDisplayTime = 0;
		frame_state.predictedDisplayPeriod = 0;
		frame_state.shouldRender = false;

		return;
	}

	if (frame_state.predictedDisplayPeriod > 500000000) {
		// display period more then 0.5 seconds? must be wrong data
		print_verbose(String("OpenXR resetting invalid display period ") + rtos(frame_state.predictedDisplayPeriod));
		frame_state.predictedDisplayPeriod = 0;
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_pre_render();
	}

	// Get our view info for the frame we're about to render, note from the OpenXR manual:
	// "Repeatedly calling xrLocateViews with the same time may not necessarily return the same result. Instead the prediction gets increasingly accurate as the function is called closer to the given time for which a prediction is made"

	// We're calling this "relatively" early, the positioning we're obtaining here will be used to do our frustum culling,
	// occlusion culling, etc. There is however a technique that we can investigate in the future where after our entire
	// Vulkan command buffer is build, but right before vkSubmitQueue is called, we call xrLocateViews one more time and
	// update the view and projection matrix once more with a slightly more accurate predication and then submit the
	// command queues.

	// That is not possible yet but worth investigating in the future.

	XrViewLocateInfo view_locate_info = {
		XR_TYPE_VIEW_LOCATE_INFO, // type
		nullptr, // next
		view_configuration, // viewConfigurationType
		frame_state.predictedDisplayTime, // displayTime
		play_space // space
	};
	XrViewState view_state = {
		XR_TYPE_VIEW_STATE, // type
		nullptr, // next
		0 // viewStateFlags
	};
	uint32_t view_count_output;
	result = xrLocateViews(session, &view_locate_info, &view_state, view_count, &view_count_output, views);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Couldn't locate views [", get_error_string(result), "]");
		return;
	}

	bool pose_valid = true;
	for (uint64_t i = 0; i < view_count_output; i++) {
		if ((view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0 ||
				(view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0) {
			pose_valid = false;
		}
	}
	if (view_pose_valid != pose_valid) {
		view_pose_valid = pose_valid;
		if (!view_pose_valid) {
			print_verbose("OpenXR View pose became invalid");
		} else {
			print_verbose("OpenXR View pose became valid");
		}
	}

	// let's start our frame..
	XrFrameBeginInfo frame_begin_info = {
		XR_TYPE_FRAME_BEGIN_INFO, // type
		nullptr // next
	};
	result = xrBeginFrame(session, &frame_begin_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to being frame [", get_error_string(result), "]");
		return;
	}
}

bool OpenXRAPI::pre_draw_viewport(RID p_render_target) {
	if (!can_render()) {
		return false;
	}

	// TODO: at some point in time we may support multiple viewports in which case we need to handle that...

	// Acquire our images
	for (int i = 0; i < OPENXR_SWAPCHAIN_MAX; i++) {
		if (!swapchains[i].image_acquired && swapchains[i].swapchain != XR_NULL_HANDLE) {
			if (!acquire_image(swapchains[i])) {
				return false;
			}
			swapchains[i].image_acquired = true;
		}
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_pre_draw_viewport(p_render_target);
	}

	return true;
}

XrSwapchain OpenXRAPI::get_color_swapchain() {
	return swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain;
}

RID OpenXRAPI::get_color_texture() {
	if (swapchains[OPENXR_SWAPCHAIN_COLOR].image_acquired) {
		return graphics_extension->get_texture(swapchains[OPENXR_SWAPCHAIN_COLOR].swapchain_graphics_data, swapchains[OPENXR_SWAPCHAIN_COLOR].image_index);
	} else {
		return RID();
	}
}

RID OpenXRAPI::get_depth_texture() {
	// Note, image will not be acquired if we didn't have a suitable swap chain format.
	if (submit_depth_buffer && swapchains[OPENXR_SWAPCHAIN_DEPTH].image_acquired) {
		return graphics_extension->get_texture(swapchains[OPENXR_SWAPCHAIN_DEPTH].swapchain_graphics_data, swapchains[OPENXR_SWAPCHAIN_DEPTH].image_index);
	} else {
		return RID();
	}
}

void OpenXRAPI::post_draw_viewport(RID p_render_target) {
	if (!can_render()) {
		return;
	}

	for (OpenXRExtensionWrapper *wrapper : registered_extension_wrappers) {
		wrapper->on_post_draw_viewport(p_render_target);
	}
};

void OpenXRAPI::end_frame() {
	XrResult result;

	ERR_FAIL_COND(instance == XR_NULL_HANDLE);

	if (!running) {
		return;
	}

	if (frame_state.shouldRender && view_pose_valid && !swapchains[OPENXR_SWAPCHAIN_COLOR].image_acquired) {
		print_line("OpenXR: No viewport was marked with use_xr, there is no rendered output!");
	}

	// must have:
	// - shouldRender set to true
	// - a valid view pose for projection_views[eye].pose to submit layer
	// - an image to render
	if (!frame_state.shouldRender || !view_pose_valid || !swapchains[OPENXR_SWAPCHAIN_COLOR].image_acquired) {
		// submit 0 layers when we shouldn't render
		XrFrameEndInfo frame_end_info = {
			XR_TYPE_FRAME_END_INFO, // type
			nullptr, // next
			frame_state.predictedDisplayTime, // displayTime
			environment_blend_mode, // environmentBlendMode
			0, // layerCount
			nullptr // layers
		};
		result = xrEndFrame(session, &frame_end_info);
		if (XR_FAILED(result)) {
			print_line("OpenXR: failed to end frame! [", get_error_string(result), "]");
			return;
		}

		// neither eye is rendered
		return;
	}

	// release our swapchain image if we acquired it
	for (int i = 0; i < OPENXR_SWAPCHAIN_MAX; i++) {
		if (swapchains[i].image_acquired) {
			swapchains[i].image_acquired = false; // whether we succeed or not, consider this released.

			release_image(swapchains[i]);
		}
	}

	for (uint32_t eye = 0; eye < view_count; eye++) {
		projection_views[eye].fov = views[eye].fov;
		projection_views[eye].pose = views[eye].pose;
	}

	Vector<const XrCompositionLayerBaseHeader *> layers_list;

	// Add composition layers from providers
	for (OpenXRCompositionLayerProvider *provider : composition_layer_providers) {
		XrCompositionLayerBaseHeader *layer = provider->get_composition_layer();
		if (layer) {
			layers_list.push_back(layer);
		}
	}

	XrCompositionLayerFlags layer_flags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
	if (layers_list.size() > 0 || environment_blend_mode != XR_ENVIRONMENT_BLEND_MODE_OPAQUE) {
		layer_flags |= XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
	}

	XrCompositionLayerProjection projection_layer = {
		XR_TYPE_COMPOSITION_LAYER_PROJECTION, // type
		nullptr, // next
		layer_flags, // layerFlags
		play_space, // space
		view_count, // viewCount
		projection_views, // views
	};
	layers_list.push_back((const XrCompositionLayerBaseHeader *)&projection_layer);

	XrFrameEndInfo frame_end_info = {
		XR_TYPE_FRAME_END_INFO, // type
		nullptr, // next
		frame_state.predictedDisplayTime, // displayTime
		environment_blend_mode, // environmentBlendMode
		static_cast<uint32_t>(layers_list.size()), // layerCount
		layers_list.ptr() // layers
	};
	result = xrEndFrame(session, &frame_end_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to end frame! [", get_error_string(result), "]");
		return;
	}
}

float OpenXRAPI::get_display_refresh_rate() const {
	OpenXRDisplayRefreshRateExtension *drrext = OpenXRDisplayRefreshRateExtension::get_singleton();
	if (drrext) {
		return drrext->get_refresh_rate();
	}

	return 0.0;
}

void OpenXRAPI::set_display_refresh_rate(float p_refresh_rate) {
	OpenXRDisplayRefreshRateExtension *drrext = OpenXRDisplayRefreshRateExtension::get_singleton();
	if (drrext != nullptr) {
		drrext->set_refresh_rate(p_refresh_rate);
	}
}

Array OpenXRAPI::get_available_display_refresh_rates() const {
	OpenXRDisplayRefreshRateExtension *drrext = OpenXRDisplayRefreshRateExtension::get_singleton();
	if (drrext != nullptr) {
		return drrext->get_available_refresh_rates();
	}

	return Array();
}

double OpenXRAPI::get_render_target_size_multiplier() const {
	return render_target_size_multiplier;
}

void OpenXRAPI::set_render_target_size_multiplier(double multiplier) {
	render_target_size_multiplier = multiplier;
}

bool OpenXRAPI::is_foveation_supported() const {
	OpenXRFBFoveationExtension *fov_ext = OpenXRFBFoveationExtension::get_singleton();
	return fov_ext != nullptr && fov_ext->is_enabled();
}

int OpenXRAPI::get_foveation_level() const {
	OpenXRFBFoveationExtension *fov_ext = OpenXRFBFoveationExtension::get_singleton();
	if (fov_ext != nullptr && fov_ext->is_enabled()) {
		switch (fov_ext->get_foveation_level()) {
			case XR_FOVEATION_LEVEL_NONE_FB:
				return 0;
			case XR_FOVEATION_LEVEL_LOW_FB:
				return 1;
			case XR_FOVEATION_LEVEL_MEDIUM_FB:
				return 2;
			case XR_FOVEATION_LEVEL_HIGH_FB:
				return 3;
			default:
				return 0;
		}
	}

	return 0;
}

void OpenXRAPI::set_foveation_level(int p_foveation_level) {
	ERR_FAIL_UNSIGNED_INDEX(p_foveation_level, 4);
	OpenXRFBFoveationExtension *fov_ext = OpenXRFBFoveationExtension::get_singleton();
	if (fov_ext != nullptr && fov_ext->is_enabled()) {
		XrFoveationLevelFB levels[] = { XR_FOVEATION_LEVEL_NONE_FB, XR_FOVEATION_LEVEL_LOW_FB, XR_FOVEATION_LEVEL_MEDIUM_FB, XR_FOVEATION_LEVEL_HIGH_FB };
		fov_ext->set_foveation_level(levels[p_foveation_level]);
	}
}

bool OpenXRAPI::get_foveation_dynamic() const {
	OpenXRFBFoveationExtension *fov_ext = OpenXRFBFoveationExtension::get_singleton();
	if (fov_ext != nullptr && fov_ext->is_enabled()) {
		return fov_ext->get_foveation_dynamic() == XR_FOVEATION_DYNAMIC_LEVEL_ENABLED_FB;
	}
	return false;
}

void OpenXRAPI::set_foveation_dynamic(bool p_foveation_dynamic) {
	OpenXRFBFoveationExtension *fov_ext = OpenXRFBFoveationExtension::get_singleton();
	if (fov_ext != nullptr && fov_ext->is_enabled()) {
		fov_ext->set_foveation_dynamic(p_foveation_dynamic ? XR_FOVEATION_DYNAMIC_LEVEL_ENABLED_FB : XR_FOVEATION_DYNAMIC_DISABLED_FB);
	}
}

OpenXRAPI::OpenXRAPI() {
	// OpenXRAPI is only constructed if OpenXR is enabled.
	singleton = this;

	if (Engine::get_singleton()->is_editor_hint()) {
		// Enabled OpenXR in the editor? Adjust our settings for the editor

	} else {
		// Load settings from project settings
		int form_factor_setting = GLOBAL_GET("xr/openxr/form_factor");
		switch (form_factor_setting) {
			case 0: {
				form_factor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
			} break;
			case 1: {
				form_factor = XR_FORM_FACTOR_HANDHELD_DISPLAY;
			} break;
			default:
				break;
		}

		int view_configuration_setting = GLOBAL_GET("xr/openxr/view_configuration");
		switch (view_configuration_setting) {
			case 0: {
				view_configuration = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO;
			} break;
			case 1: {
				view_configuration = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
			} break;
			/* we don't support quad and observer configurations (yet)
			case 2: {
				view_configuration = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO;
			} break;
			case 3: {
				view_configuration = XR_VIEW_CONFIGURATION_TYPE_SECONDARY_MONO_FIRST_PERSON_OBSERVER_MSFT;
			} break;
			*/
			default:
				break;
		}

		int reference_space_setting = GLOBAL_GET("xr/openxr/reference_space");
		switch (reference_space_setting) {
			case 0: {
				reference_space = XR_REFERENCE_SPACE_TYPE_LOCAL;
			} break;
			case 1: {
				reference_space = XR_REFERENCE_SPACE_TYPE_STAGE;
			} break;
			default:
				break;
		}

		int environment_blend_mode_setting = GLOBAL_GET("xr/openxr/environment_blend_mode");
		switch (environment_blend_mode_setting) {
			case 0: {
				environment_blend_mode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
			} break;
			case 1: {
				environment_blend_mode = XR_ENVIRONMENT_BLEND_MODE_ADDITIVE;
			} break;
			case 2: {
				environment_blend_mode = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
			} break;
			default:
				break;
		}

		submit_depth_buffer = GLOBAL_GET("xr/openxr/submit_depth_buffer");
	}

	// reset a few things that can't be done in our class definition
	frame_state.predictedDisplayTime = 0;
	frame_state.predictedDisplayPeriod = 0;
}

OpenXRAPI::~OpenXRAPI() {
	// cleanup our composition layer providers
	for (OpenXRCompositionLayerProvider *provider : composition_layer_providers) {
		memdelete(provider);
	}
	composition_layer_providers.clear();

	if (supported_extensions != nullptr) {
		memfree(supported_extensions);
		supported_extensions = nullptr;
	}

	if (layer_properties != nullptr) {
		memfree(layer_properties);
		layer_properties = nullptr;
	}

#ifdef ANDROID_ENABLED
	if (openxr_loader_library_handle) {
		OS::get_singleton()->close_dynamic_library(openxr_loader_library_handle);
		openxr_loader_library_handle = nullptr;
	}
#endif

	singleton = nullptr;
}

Transform3D OpenXRAPI::transform_from_pose(const XrPosef &p_pose) {
	Quaternion q(p_pose.orientation.x, p_pose.orientation.y, p_pose.orientation.z, p_pose.orientation.w);
	Basis basis(q);
	Vector3 origin(p_pose.position.x, p_pose.position.y, p_pose.position.z);

	return Transform3D(basis, origin);
}

template <typename T>
XRPose::TrackingConfidence _transform_from_location(const T &p_location, Transform3D &r_transform) {
	Basis basis;
	Vector3 origin;
	XRPose::TrackingConfidence confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;
	const XrPosef &pose = p_location.pose;

	// Check orientation
	if (p_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) {
		Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
		r_transform.basis = Basis(q);

		if (p_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT) {
			// Fully valid orientation, so either 3DOF or 6DOF tracking with high confidence so default to HIGH_TRACKING
			confidence = XRPose::XR_TRACKING_CONFIDENCE_HIGH;
		} else {
			// Orientation is being tracked but we're using old/predicted data, so low tracking confidence
			confidence = XRPose::XR_TRACKING_CONFIDENCE_LOW;
		}
	} else {
		r_transform.basis = Basis();
	}

	// Check location
	if (p_location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) {
		r_transform.origin = Vector3(pose.position.x, pose.position.y, pose.position.z);

		if (!(p_location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
			// Location is being tracked but we're using old/predicted data, so low tracking confidence
			confidence = XRPose::XR_TRACKING_CONFIDENCE_LOW;
		} else if (confidence == XRPose::XR_TRACKING_CONFIDENCE_NONE) {
			// Position tracking without orientation tracking?
			confidence = XRPose::XR_TRACKING_CONFIDENCE_HIGH;
		}
	} else {
		// No tracking or 3DOF I guess..
		r_transform.origin = Vector3();
	}

	return confidence;
}

XRPose::TrackingConfidence OpenXRAPI::transform_from_location(const XrSpaceLocation &p_location, Transform3D &r_transform) {
	return _transform_from_location(p_location, r_transform);
}

XRPose::TrackingConfidence OpenXRAPI::transform_from_location(const XrHandJointLocationEXT &p_location, Transform3D &r_transform) {
	return _transform_from_location(p_location, r_transform);
}

void OpenXRAPI::parse_velocities(const XrSpaceVelocity &p_velocity, Vector3 &r_linear_velocity, Vector3 &r_angular_velocity) {
	if (p_velocity.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT) {
		XrVector3f linear_velocity = p_velocity.linearVelocity;
		r_linear_velocity = Vector3(linear_velocity.x, linear_velocity.y, linear_velocity.z);
	} else {
		r_linear_velocity = Vector3();
	}
	if (p_velocity.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT) {
		XrVector3f angular_velocity = p_velocity.angularVelocity;
		r_angular_velocity = Vector3(angular_velocity.x, angular_velocity.y, angular_velocity.z);
	} else {
		r_angular_velocity = Vector3();
	}
}

bool OpenXRAPI::xr_result(XrResult result, const char *format, Array args) const {
	if (XR_SUCCEEDED(result))
		return true;

	char resultString[XR_MAX_RESULT_STRING_SIZE];
	xrResultToString(instance, result, resultString);

	print_error(String("OpenXR ") + String(format).format(args) + String(" [") + String(resultString) + String("]"));

	return false;
}

RID OpenXRAPI::get_tracker_rid(XrPath p_path) {
	List<RID> current;
	tracker_owner.get_owned_list(&current);
	for (int i = 0; i < current.size(); i++) {
		Tracker *tracker = tracker_owner.get_or_null(current[i]);
		if (tracker && tracker->toplevel_path == p_path) {
			return current[i];
		}
	}

	return RID();
}

RID OpenXRAPI::tracker_create(const String p_name) {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, RID());

	Tracker new_tracker;
	new_tracker.name = p_name;
	new_tracker.toplevel_path = XR_NULL_PATH;
	new_tracker.active_profile_rid = RID();

	XrResult result = xrStringToPath(instance, p_name.utf8().get_data(), &new_tracker.toplevel_path);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to get path for ", p_name, "! [", get_error_string(result), "]");
		return RID();
	}

	return tracker_owner.make_rid(new_tracker);
}

String OpenXRAPI::tracker_get_name(RID p_tracker) {
	if (p_tracker.is_null()) {
		return String("None");
	}

	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, String());

	return tracker->name;
}

void OpenXRAPI::tracker_check_profile(RID p_tracker, XrSession p_session) {
	if (p_session == XR_NULL_HANDLE) {
		p_session = session;
	}

	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL(tracker);

	if (tracker->toplevel_path == XR_NULL_PATH) {
		// no path, how was this even created?
		return;
	}

	XrInteractionProfileState profile_state = {
		XR_TYPE_INTERACTION_PROFILE_STATE, // type
		nullptr, // next
		XR_NULL_PATH // interactionProfile
	};

	XrResult result = xrGetCurrentInteractionProfile(p_session, tracker->toplevel_path, &profile_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get interaction profile for", itos(tracker->toplevel_path), "[", get_error_string(result), "]");
		return;
	}

	XrPath new_profile = profile_state.interactionProfile;
	XrPath was_profile = get_interaction_profile_path(tracker->active_profile_rid);
	if (was_profile != new_profile) {
		tracker->active_profile_rid = get_interaction_profile_rid(new_profile);

		if (xr_interface) {
			xr_interface->tracker_profile_changed(p_tracker, tracker->active_profile_rid);
		}
	}
}

void OpenXRAPI::tracker_free(RID p_tracker) {
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL(tracker);

	// there is nothing to free here

	tracker_owner.free(p_tracker);
}

RID OpenXRAPI::action_set_create(const String p_name, const String p_localized_name, const int p_priority) {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, RID());
	ActionSet action_set;

	action_set.name = p_name;
	action_set.is_attached = false;

	// create our action set...
	XrActionSetCreateInfo action_set_info = {
		XR_TYPE_ACTION_SET_CREATE_INFO, // type
		nullptr, // next
		"", // actionSetName
		"", // localizedActionSetName
		uint32_t(p_priority) // priority
	};

	copy_string_to_char_buffer(p_name, action_set_info.actionSetName, XR_MAX_ACTION_SET_NAME_SIZE);
	copy_string_to_char_buffer(p_localized_name, action_set_info.localizedActionSetName, XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);

	XrResult result = xrCreateActionSet(instance, &action_set_info, &action_set.handle);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to create action set ", p_name, "! [", get_error_string(result), "]");
		return RID();
	}

	return action_set_owner.make_rid(action_set);
}

String OpenXRAPI::action_set_get_name(RID p_action_set) {
	if (p_action_set.is_null()) {
		return String("None");
	}

	ActionSet *action_set = action_set_owner.get_or_null(p_action_set);
	ERR_FAIL_NULL_V(action_set, String());

	return action_set->name;
}

bool OpenXRAPI::attach_action_sets(const Vector<RID> &p_action_sets) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	Vector<XrActionSet> action_handles;
	action_handles.resize(p_action_sets.size());
	for (int i = 0; i < p_action_sets.size(); i++) {
		ActionSet *action_set = action_set_owner.get_or_null(p_action_sets[i]);
		ERR_FAIL_NULL_V(action_set, false);

		if (action_set->is_attached) {
			return false;
		}

		action_handles.set(i, action_set->handle);
	}

	// So according to the docs, once we attach our action set to our session it becomes read only..
	// https://www.khronos.org/registry/OpenXR/specs/1.0/man/html/xrAttachSessionActionSets.html
	XrSessionActionSetsAttachInfo attach_info = {
		XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO, // type
		nullptr, // next
		(uint32_t)p_action_sets.size(), // countActionSets,
		action_handles.ptr() // actionSets
	};

	XrResult result = xrAttachSessionActionSets(session, &attach_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to attach action sets! [", get_error_string(result), "]");
		return false;
	}

	for (int i = 0; i < p_action_sets.size(); i++) {
		ActionSet *action_set = action_set_owner.get_or_null(p_action_sets[i]);
		ERR_FAIL_NULL_V(action_set, false);
		action_set->is_attached = true;
	}

	/* For debugging:
	print_verbose("Attached set " + action_set->name);
	List<RID> action_rids;
	action_owner.get_owned_list(&action_rids);
	for (int i = 0; i < action_rids.size(); i++) {
		Action * action = action_owner.get_or_null(action_rids[i]);
		if (action && action->action_set_rid == p_action_set) {
			print_verbose(" - Action " + action->name + ": " + OpenXRUtil::get_action_type_name(action->action_type));
			for (int j = 0; j < action->trackers.size(); j++) {
				Tracker * tracker = tracker_owner.get_or_null(action->trackers[j].tracker_rid);
				if (tracker) {
					print_verbose("    - " + tracker->name);
				}
			}
		}
	}
	*/

	return true;
}

void OpenXRAPI::action_set_free(RID p_action_set) {
	ActionSet *action_set = action_set_owner.get_or_null(p_action_set);
	ERR_FAIL_NULL(action_set);

	if (action_set->handle != XR_NULL_HANDLE) {
		xrDestroyActionSet(action_set->handle);
	}

	action_set_owner.free(p_action_set);
}

RID OpenXRAPI::get_action_rid(XrAction p_action) {
	List<RID> current;
	action_owner.get_owned_list(&current);
	for (int i = 0; i < current.size(); i++) {
		Action *action = action_owner.get_or_null(current[i]);
		if (action && action->handle == p_action) {
			return current[i];
		}
	}

	return RID();
}

RID OpenXRAPI::action_create(RID p_action_set, const String p_name, const String p_localized_name, OpenXRAction::ActionType p_action_type, const Vector<RID> &p_trackers) {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, RID());

	Action action;
	action.name = p_name;

	ActionSet *action_set = action_set_owner.get_or_null(p_action_set);
	ERR_FAIL_NULL_V(action_set, RID());
	ERR_FAIL_COND_V(action_set->handle == XR_NULL_HANDLE, RID());
	action.action_set_rid = p_action_set;

	switch (p_action_type) {
		case OpenXRAction::OPENXR_ACTION_BOOL:
			action.action_type = XR_ACTION_TYPE_BOOLEAN_INPUT;
			break;
		case OpenXRAction::OPENXR_ACTION_FLOAT:
			action.action_type = XR_ACTION_TYPE_FLOAT_INPUT;
			break;
		case OpenXRAction::OPENXR_ACTION_VECTOR2:
			action.action_type = XR_ACTION_TYPE_VECTOR2F_INPUT;
			break;
		case OpenXRAction::OPENXR_ACTION_POSE:
			action.action_type = XR_ACTION_TYPE_POSE_INPUT;
			break;
		case OpenXRAction::OPENXR_ACTION_HAPTIC:
			action.action_type = XR_ACTION_TYPE_VIBRATION_OUTPUT;
			break;
		default:
			ERR_FAIL_V(RID());
			break;
	}

	Vector<XrPath> toplevel_paths;
	for (int i = 0; i < p_trackers.size(); i++) {
		Tracker *tracker = tracker_owner.get_or_null(p_trackers[i]);
		if (tracker != nullptr && tracker->toplevel_path != XR_NULL_PATH) {
			ActionTracker action_tracker = {
				p_trackers[i], // tracker
				XR_NULL_HANDLE, // space
				false // was_location_valid
			};
			action.trackers.push_back(action_tracker);

			toplevel_paths.push_back(tracker->toplevel_path);
		}
	}

	XrActionCreateInfo action_info = {
		XR_TYPE_ACTION_CREATE_INFO, // type
		nullptr, // next
		"", // actionName
		action.action_type, // actionType
		uint32_t(toplevel_paths.size()), // countSubactionPaths
		toplevel_paths.ptr(), // subactionPaths
		"" // localizedActionName
	};

	copy_string_to_char_buffer(p_name, action_info.actionName, XR_MAX_ACTION_NAME_SIZE);
	copy_string_to_char_buffer(p_localized_name, action_info.localizedActionName, XR_MAX_LOCALIZED_ACTION_NAME_SIZE);

	XrResult result = xrCreateAction(action_set->handle, &action_info, &action.handle);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to create action ", p_name, "! [", get_error_string(result), "]");
		return RID();
	}

	return action_owner.make_rid(action);
}

String OpenXRAPI::action_get_name(RID p_action) {
	if (p_action.is_null()) {
		return String("None");
	}

	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, String());

	return action->name;
}

void OpenXRAPI::action_free(RID p_action) {
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL(action);

	if (action->handle != XR_NULL_HANDLE) {
		xrDestroyAction(action->handle);
	}

	action_owner.free(p_action);
}

RID OpenXRAPI::get_interaction_profile_rid(XrPath p_path) {
	List<RID> current;
	interaction_profile_owner.get_owned_list(&current);
	for (int i = 0; i < current.size(); i++) {
		InteractionProfile *ip = interaction_profile_owner.get_or_null(current[i]);
		if (ip && ip->path == p_path) {
			return current[i];
		}
	}

	return RID();
}

XrPath OpenXRAPI::get_interaction_profile_path(RID p_interaction_profile) {
	if (p_interaction_profile.is_null()) {
		return XR_NULL_PATH;
	}

	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL_V(ip, XR_NULL_PATH);

	return ip->path;
}

RID OpenXRAPI::interaction_profile_create(const String p_name) {
	if (!is_interaction_profile_supported(p_name)) {
		// The extension enabling this path must not be active, we will silently skip this interaction profile
		return RID();
	}

	InteractionProfile new_interaction_profile;

	XrResult result = xrStringToPath(instance, p_name.utf8().get_data(), &new_interaction_profile.path);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to get path for ", p_name, "! [", get_error_string(result), "]");
		return RID();
	}

	RID existing_ip = get_interaction_profile_rid(new_interaction_profile.path);
	if (existing_ip.is_valid()) {
		return existing_ip;
	}

	new_interaction_profile.name = p_name;
	return interaction_profile_owner.make_rid(new_interaction_profile);
}

String OpenXRAPI::interaction_profile_get_name(RID p_interaction_profile) {
	if (p_interaction_profile.is_null()) {
		return String("None");
	}

	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL_V(ip, String());

	return ip->name;
}

void OpenXRAPI::interaction_profile_clear_bindings(RID p_interaction_profile) {
	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL(ip);

	ip->bindings.clear();
}

bool OpenXRAPI::interaction_profile_add_binding(RID p_interaction_profile, RID p_action, const String p_path) {
	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL_V(ip, false);

	if (!interaction_profile_supports_io_path(ip->name, p_path)) {
		return false;
	}

	XrActionSuggestedBinding binding;

	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_COND_V(action == nullptr || action->handle == XR_NULL_HANDLE, false);

	binding.action = action->handle;

	XrResult result = xrStringToPath(instance, p_path.utf8().get_data(), &binding.binding);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to get path for ", p_path, "! [", get_error_string(result), "]");
		return false;
	}

	ip->bindings.push_back(binding);

	return true;
}

bool OpenXRAPI::interaction_profile_suggest_bindings(RID p_interaction_profile) {
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, false);

	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL_V(ip, false);

	const XrInteractionProfileSuggestedBinding suggested_bindings = {
		XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING, // type
		nullptr, // next
		ip->path, // interactionProfile
		uint32_t(ip->bindings.size()), // countSuggestedBindings
		ip->bindings.ptr() // suggestedBindings
	};

	XrResult result = xrSuggestInteractionProfileBindings(instance, &suggested_bindings);
	if (result == XR_ERROR_PATH_UNSUPPORTED) {
		// this is fine, not all runtimes support all devices.
		print_verbose("OpenXR Interaction profile " + ip->name + " is not supported on this runtime");
	} else if (XR_FAILED(result)) {
		print_line("OpenXR: failed to suggest bindings for ", ip->name, "! [", get_error_string(result), "]");
		// reporting is enough...
	}

	/* For debugging:
	print_verbose("Suggested bindings for " + ip->name);
	for (int i = 0; i < ip->bindings.size(); i++) {
		uint32_t strlen;
		char path[XR_MAX_PATH_LENGTH];

		String action_name = action_get_name(get_action_rid(ip->bindings[i].action));

		XrResult result = xrPathToString(instance, ip->bindings[i].binding, XR_MAX_PATH_LENGTH, &strlen, path);
		if (XR_FAILED(result)) {
			print_line("OpenXR: failed to retrieve bindings for ", action_name, "! [", get_error_string(result), "]");
		}
		print_verbose(" - " + action_name + " => " + String(path));
	}
	*/

	return true;
}

void OpenXRAPI::interaction_profile_free(RID p_interaction_profile) {
	InteractionProfile *ip = interaction_profile_owner.get_or_null(p_interaction_profile);
	ERR_FAIL_NULL(ip);

	ip->bindings.clear();

	interaction_profile_owner.free(p_interaction_profile);
}

bool OpenXRAPI::sync_action_sets(const Vector<RID> p_active_sets) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	if (!running) {
		return false;
	}

	Vector<XrActiveActionSet> active_sets;
	for (int i = 0; i < p_active_sets.size(); i++) {
		ActionSet *action_set = action_set_owner.get_or_null(p_active_sets[i]);
		if (action_set && action_set->handle != XR_NULL_HANDLE) {
			XrActiveActionSet aset;
			aset.actionSet = action_set->handle;
			aset.subactionPath = XR_NULL_PATH;
			active_sets.push_back(aset);
		}
	}

	ERR_FAIL_COND_V(active_sets.size() == 0, false);

	XrActionsSyncInfo sync_info = {
		XR_TYPE_ACTIONS_SYNC_INFO, // type
		nullptr, // next
		uint32_t(active_sets.size()), // countActiveActionSets
		active_sets.ptr() // activeActionSets
	};

	XrResult result = xrSyncActions(session, &sync_info);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to sync active action sets! [", get_error_string(result), "]");
		return false;
	}

	return true;
}

bool OpenXRAPI::get_action_bool(RID p_action, RID p_tracker) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, false);
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, false);

	if (!running) {
		return false;
	}

	ERR_FAIL_COND_V(action->action_type != XR_ACTION_TYPE_BOOLEAN_INPUT, false);

	XrActionStateGetInfo get_info = {
		XR_TYPE_ACTION_STATE_GET_INFO, // type
		nullptr, // next
		action->handle, // action
		tracker->toplevel_path // subactionPath
	};

	XrActionStateBoolean result_state;
	result_state.type = XR_TYPE_ACTION_STATE_BOOLEAN,
	result_state.next = nullptr;
	XrResult result = xrGetActionStateBoolean(session, &get_info, &result_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: couldn't get action boolean! [", get_error_string(result), "]");
		return false;
	}

	return result_state.isActive && result_state.currentState;
}

float OpenXRAPI::get_action_float(RID p_action, RID p_tracker) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, 0.0);
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, 0.0);
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, 0.0);

	if (!running) {
		return 0.0;
	}

	ERR_FAIL_COND_V(action->action_type != XR_ACTION_TYPE_FLOAT_INPUT, 0.0);

	XrActionStateGetInfo get_info = {
		XR_TYPE_ACTION_STATE_GET_INFO, // type
		nullptr, // next
		action->handle, // action
		tracker->toplevel_path // subactionPath
	};

	XrActionStateFloat result_state;
	result_state.type = XR_TYPE_ACTION_STATE_FLOAT,
	result_state.next = nullptr;
	XrResult result = xrGetActionStateFloat(session, &get_info, &result_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: couldn't get action float! [", get_error_string(result), "]");
		return 0.0;
	}

	return result_state.isActive ? result_state.currentState : 0.0;
}

Vector2 OpenXRAPI::get_action_vector2(RID p_action, RID p_tracker) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, Vector2());
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, Vector2());
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, Vector2());

	if (!running) {
		return Vector2();
	}

	ERR_FAIL_COND_V(action->action_type != XR_ACTION_TYPE_VECTOR2F_INPUT, Vector2());

	XrActionStateGetInfo get_info = {
		XR_TYPE_ACTION_STATE_GET_INFO, // type
		nullptr, // next
		action->handle, // action
		tracker->toplevel_path // subactionPath
	};

	XrActionStateVector2f result_state;
	result_state.type = XR_TYPE_ACTION_STATE_VECTOR2F,
	result_state.next = nullptr;
	XrResult result = xrGetActionStateVector2f(session, &get_info, &result_state);
	if (XR_FAILED(result)) {
		print_line("OpenXR: couldn't get action vector2! [", get_error_string(result), "]");
		return Vector2();
	}

	return result_state.isActive ? Vector2(result_state.currentState.x, result_state.currentState.y) : Vector2();
}

XRPose::TrackingConfidence OpenXRAPI::get_action_pose(RID p_action, RID p_tracker, Transform3D &r_transform, Vector3 &r_linear_velocity, Vector3 &r_angular_velocity) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, XRPose::XR_TRACKING_CONFIDENCE_NONE);
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, XRPose::XR_TRACKING_CONFIDENCE_NONE);
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, XRPose::XR_TRACKING_CONFIDENCE_NONE);

	if (!running) {
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	ERR_FAIL_COND_V(action->action_type != XR_ACTION_TYPE_POSE_INPUT, XRPose::XR_TRACKING_CONFIDENCE_NONE);

	// print_verbose("Checking " + action->name + " => " + tracker->name + " (" + itos(tracker->toplevel_path) + ")");

	uint64_t index = 0xFFFFFFFF;
	uint64_t size = uint64_t(action->trackers.size());
	for (uint64_t i = 0; i < size && index == 0xFFFFFFFF; i++) {
		if (action->trackers[i].tracker_rid == p_tracker) {
			index = i;
		}
	}

	if (index == 0xFFFFFFFF) {
		// couldn't find it?
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	XrTime display_time = get_next_frame_time();
	if (display_time == 0) {
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	if (action->trackers[index].space == XR_NULL_HANDLE) {
		// if this is a pose we need to define spaces

		XrActionSpaceCreateInfo action_space_info = {
			XR_TYPE_ACTION_SPACE_CREATE_INFO, // type
			nullptr, // next
			action->handle, // action
			tracker->toplevel_path, // subactionPath
			{
					{ 0.0, 0.0, 0.0, 1.0 }, // orientation
					{ 0.0, 0.0, 0.0 } // position
			} // poseInActionSpace
		};

		XrSpace space;
		XrResult result = xrCreateActionSpace(session, &action_space_info, &space);
		if (XR_FAILED(result)) {
			print_line("OpenXR: couldn't create action space! [", get_error_string(result), "]");
			return XRPose::XR_TRACKING_CONFIDENCE_NONE;
		}

		action->trackers.ptrw()[index].space = space;
	}

	XrSpaceVelocity velocity = {
		XR_TYPE_SPACE_VELOCITY, // type
		nullptr, // next
		0, // velocityFlags
		{ 0.0, 0.0, 0.0 }, // linearVelocity
		{ 0.0, 0.0, 0.0 } // angularVelocity
	};

	XrSpaceLocation location = {
		XR_TYPE_SPACE_LOCATION, // type
		&velocity, // next
		0, // locationFlags
		{
				{ 0.0, 0.0, 0.0, 0.0 }, // orientation
				{ 0.0, 0.0, 0.0 } // position
		} // pose
	};

	XrResult result = xrLocateSpace(action->trackers[index].space, play_space, display_time, &location);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to locate space! [", get_error_string(result), "]");
		return XRPose::XR_TRACKING_CONFIDENCE_NONE;
	}

	XRPose::TrackingConfidence confidence = transform_from_location(location, r_transform);
	parse_velocities(velocity, r_linear_velocity, r_angular_velocity);

	return confidence;
}

bool OpenXRAPI::trigger_haptic_pulse(RID p_action, RID p_tracker, float p_frequency, float p_amplitude, XrDuration p_duration_ns) {
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);
	Action *action = action_owner.get_or_null(p_action);
	ERR_FAIL_NULL_V(action, false);
	Tracker *tracker = tracker_owner.get_or_null(p_tracker);
	ERR_FAIL_NULL_V(tracker, false);

	if (!running) {
		return false;
	}

	ERR_FAIL_COND_V(action->action_type != XR_ACTION_TYPE_VIBRATION_OUTPUT, false);

	XrHapticActionInfo action_info = {
		XR_TYPE_HAPTIC_ACTION_INFO, // type
		nullptr, // next
		action->handle, // action
		tracker->toplevel_path // subactionPath
	};

	XrHapticVibration vibration = {
		XR_TYPE_HAPTIC_VIBRATION, // type
		nullptr, // next
		p_duration_ns, // duration
		p_frequency, // frequency
		p_amplitude, // amplitude
	};

	XrResult result = xrApplyHapticFeedback(session, &action_info, (const XrHapticBaseHeader *)&vibration);
	if (XR_FAILED(result)) {
		print_line("OpenXR: failed to apply haptic feedback! [", get_error_string(result), "]");
		return false;
	}

	return true;
}

void OpenXRAPI::register_composition_layer_provider(OpenXRCompositionLayerProvider *provider) {
	composition_layer_providers.append(provider);
}

void OpenXRAPI::unregister_composition_layer_provider(OpenXRCompositionLayerProvider *provider) {
	composition_layer_providers.erase(provider);
}

const XrEnvironmentBlendMode *OpenXRAPI::get_supported_environment_blend_modes(uint32_t &count) {
	count = num_supported_environment_blend_modes;
	return supported_environment_blend_modes;
}

bool OpenXRAPI::is_environment_blend_mode_supported(XrEnvironmentBlendMode p_blend_mode) const {
	ERR_FAIL_NULL_V(supported_environment_blend_modes, false);

	for (uint32_t i = 0; i < num_supported_environment_blend_modes; i++) {
		if (supported_environment_blend_modes[i] == p_blend_mode) {
			return true;
		}
	}

	return false;
}

bool OpenXRAPI::set_environment_blend_mode(XrEnvironmentBlendMode p_blend_mode) {
	// We allow setting this when not initialized and will check if it is supported when initializing.
	// After OpenXR is initialized we verify we're setting a supported blend mode.
	if (!is_initialized() || is_environment_blend_mode_supported(p_blend_mode)) {
		environment_blend_mode = p_blend_mode;
		return true;
	}
	return false;
}
