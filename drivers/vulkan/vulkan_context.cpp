/**************************************************************************/
/*  vulkan_context.cpp                                                    */
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

#include "vulkan_context.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/version.h"
#include "servers/rendering/rendering_device.h"

#include "vk_enum_string_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
#define APP_SHORT_NAME "GodotEngine"

VulkanHooks *VulkanContext::vulkan_hooks = nullptr;

Vector<VkAttachmentReference> VulkanContext::_convert_VkAttachmentReference2(uint32_t p_count, const VkAttachmentReference2 *p_refs) {
	Vector<VkAttachmentReference> att_refs;

	if (p_refs != nullptr) {
		for (uint32_t i = 0; i < p_count; i++) {
			// We lose aspectMask in this conversion but we don't use it currently.

			VkAttachmentReference ref = {
				p_refs[i].attachment, /* attachment */
				p_refs[i].layout /* layout */
			};

			att_refs.push_back(ref);
		}
	}

	return att_refs;
}

VkResult VulkanContext::vkCreateRenderPass2KHR(VkDevice p_device, const VkRenderPassCreateInfo2 *p_create_info, const VkAllocationCallbacks *p_allocator, VkRenderPass *p_render_pass) {
	if (is_device_extension_enabled(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME)) {
		if (fpCreateRenderPass2KHR == nullptr) {
			fpCreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)vkGetDeviceProcAddr(p_device, "vkCreateRenderPass2KHR");
		}

		if (fpCreateRenderPass2KHR == nullptr) {
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		} else {
			return (fpCreateRenderPass2KHR)(p_device, p_create_info, p_allocator, p_render_pass);
		}
	} else {
		// need to fall back on vkCreateRenderPass

		const void *next = p_create_info->pNext; // ATM we only support multiview which should work if supported.

		Vector<VkAttachmentDescription> attachments;
		for (uint32_t i = 0; i < p_create_info->attachmentCount; i++) {
			// Basically the old layout just misses type and next.
			VkAttachmentDescription att = {
				p_create_info->pAttachments[i].flags, /* flags */
				p_create_info->pAttachments[i].format, /* format */
				p_create_info->pAttachments[i].samples, /* samples */
				p_create_info->pAttachments[i].loadOp, /* loadOp */
				p_create_info->pAttachments[i].storeOp, /* storeOp */
				p_create_info->pAttachments[i].stencilLoadOp, /* stencilLoadOp */
				p_create_info->pAttachments[i].stencilStoreOp, /* stencilStoreOp */
				p_create_info->pAttachments[i].initialLayout, /* initialLayout */
				p_create_info->pAttachments[i].finalLayout /* finalLayout */
			};

			attachments.push_back(att);
		}

		Vector<Vector<VkAttachmentReference>> attachment_references;
		Vector<VkSubpassDescription> subpasses;
		for (uint32_t i = 0; i < p_create_info->subpassCount; i++) {
			// Here we need to do more, again it's just stripping out type and next
			// but we have VkAttachmentReference2 to convert to VkAttachmentReference.
			// Also viewmask is not supported but we don't use it outside of multiview.

			Vector<VkAttachmentReference> input_attachments = _convert_VkAttachmentReference2(p_create_info->pSubpasses[i].inputAttachmentCount, p_create_info->pSubpasses[i].pInputAttachments);
			Vector<VkAttachmentReference> color_attachments = _convert_VkAttachmentReference2(p_create_info->pSubpasses[i].colorAttachmentCount, p_create_info->pSubpasses[i].pColorAttachments);
			Vector<VkAttachmentReference> resolve_attachments = _convert_VkAttachmentReference2(p_create_info->pSubpasses[i].colorAttachmentCount, p_create_info->pSubpasses[i].pResolveAttachments);
			Vector<VkAttachmentReference> depth_attachments = _convert_VkAttachmentReference2(p_create_info->pSubpasses[i].colorAttachmentCount, p_create_info->pSubpasses[i].pDepthStencilAttachment);

			VkSubpassDescription subpass = {
				p_create_info->pSubpasses[i].flags, /* flags */
				p_create_info->pSubpasses[i].pipelineBindPoint, /* pipelineBindPoint */
				p_create_info->pSubpasses[i].inputAttachmentCount, /* inputAttachmentCount */
				input_attachments.size() == 0 ? nullptr : input_attachments.ptr(), /* pInputAttachments */
				p_create_info->pSubpasses[i].colorAttachmentCount, /* colorAttachmentCount */
				color_attachments.size() == 0 ? nullptr : color_attachments.ptr(), /* pColorAttachments */
				resolve_attachments.size() == 0 ? nullptr : resolve_attachments.ptr(), /* pResolveAttachments */
				depth_attachments.size() == 0 ? nullptr : depth_attachments.ptr(), /* pDepthStencilAttachment */
				p_create_info->pSubpasses[i].preserveAttachmentCount, /* preserveAttachmentCount */
				p_create_info->pSubpasses[i].pPreserveAttachments /* pPreserveAttachments */
			};
			attachment_references.push_back(input_attachments);
			attachment_references.push_back(color_attachments);
			attachment_references.push_back(resolve_attachments);
			attachment_references.push_back(depth_attachments);

			subpasses.push_back(subpass);
		}

		Vector<VkSubpassDependency> dependencies;
		for (uint32_t i = 0; i < p_create_info->dependencyCount; i++) {
			// We lose viewOffset here but again I don't believe we use this anywhere.
			VkSubpassDependency dep = {
				p_create_info->pDependencies[i].srcSubpass, /* srcSubpass */
				p_create_info->pDependencies[i].dstSubpass, /* dstSubpass */
				p_create_info->pDependencies[i].srcStageMask, /* srcStageMask */
				p_create_info->pDependencies[i].dstStageMask, /* dstStageMask */
				p_create_info->pDependencies[i].srcAccessMask, /* srcAccessMask */
				p_create_info->pDependencies[i].dstAccessMask, /* dstAccessMask */
				p_create_info->pDependencies[i].dependencyFlags, /* dependencyFlags */
			};

			dependencies.push_back(dep);
		}

		// CorrelatedViewMask is not supported in vkCreateRenderPass but we
		// currently only use this for multiview.
		// We'll need to look into this.

		VkRenderPassCreateInfo create_info = {
			VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, /* sType */
			next, /* pNext*/
			p_create_info->flags, /* flags */
			(uint32_t)attachments.size(), /* attachmentCount */
			attachments.ptr(), /* pAttachments */
			(uint32_t)subpasses.size(), /* subpassCount */
			subpasses.ptr(), /* pSubpasses */
			(uint32_t)dependencies.size(), /* */
			dependencies.ptr(), /* */
		};

		return vkCreateRenderPass(device, &create_info, p_allocator, p_render_pass);
	}
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::_debug_messenger_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
		void *pUserData) {
	// This error needs to be ignored because the AMD allocator will mix up memory types on IGP processors.
	if (strstr(pCallbackData->pMessage, "Mapping an image with layout") != nullptr &&
			strstr(pCallbackData->pMessage, "can result in undefined behavior if this memory is used by the device") != nullptr) {
		return VK_FALSE;
	}
	// This needs to be ignored because Validator is wrong here.
	if (strstr(pCallbackData->pMessage, "Invalid SPIR-V binary version 1.3") != nullptr) {
		return VK_FALSE;
	}
	// This needs to be ignored because Validator is wrong here.
	if (strstr(pCallbackData->pMessage, "Shader requires flag") != nullptr) {
		return VK_FALSE;
	}

	// This needs to be ignored because Validator is wrong here.
	if (strstr(pCallbackData->pMessage, "SPIR-V module not valid: Pointer operand") != nullptr &&
			strstr(pCallbackData->pMessage, "must be a memory object") != nullptr) {
		return VK_FALSE;
	}

	if (pCallbackData->pMessageIdName && strstr(pCallbackData->pMessageIdName, "UNASSIGNED-CoreValidation-DrawState-ClearCmdBeforeDraw") != nullptr) {
		return VK_FALSE;
	}

	String type_string;
	switch (messageType) {
		case (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT):
			type_string = "GENERAL";
			break;
		case (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT):
			type_string = "VALIDATION";
			break;
		case (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT):
			type_string = "PERFORMANCE";
			break;
		case (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT):
			type_string = "VALIDATION|PERFORMANCE";
			break;
	}

	String objects_string;
	if (pCallbackData->objectCount > 0) {
		objects_string = "\n\tObjects - " + String::num_int64(pCallbackData->objectCount);
		for (uint32_t object = 0; object < pCallbackData->objectCount; ++object) {
			objects_string +=
					"\n\t\tObject[" + String::num_int64(object) + "]" +
					" - " + string_VkObjectType(pCallbackData->pObjects[object].objectType) +
					", Handle " + String::num_int64(pCallbackData->pObjects[object].objectHandle);
			if (nullptr != pCallbackData->pObjects[object].pObjectName && strlen(pCallbackData->pObjects[object].pObjectName) > 0) {
				objects_string += ", Name \"" + String(pCallbackData->pObjects[object].pObjectName) + "\"";
			}
		}
	}

	String labels_string;
	if (pCallbackData->cmdBufLabelCount > 0) {
		labels_string = "\n\tCommand Buffer Labels - " + String::num_int64(pCallbackData->cmdBufLabelCount);
		for (uint32_t cmd_buf_label = 0; cmd_buf_label < pCallbackData->cmdBufLabelCount; ++cmd_buf_label) {
			labels_string +=
					"\n\t\tLabel[" + String::num_int64(cmd_buf_label) + "]" +
					" - " + pCallbackData->pCmdBufLabels[cmd_buf_label].pLabelName +
					"{ ";
			for (int color_idx = 0; color_idx < 4; ++color_idx) {
				labels_string += String::num(pCallbackData->pCmdBufLabels[cmd_buf_label].color[color_idx]);
				if (color_idx < 3) {
					labels_string += ", ";
				}
			}
			labels_string += " }";
		}
	}

	String error_message(type_string +
			" - Message Id Number: " + String::num_int64(pCallbackData->messageIdNumber) +
			" | Message Id Name: " + pCallbackData->pMessageIdName +
			"\n\t" + pCallbackData->pMessage +
			objects_string + labels_string);

	// Convert VK severity to our own log macros.
	switch (messageSeverity) {
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
			print_verbose(error_message);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
			print_line(error_message);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
			WARN_PRINT(error_message);
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
			ERR_PRINT(error_message);
			CRASH_COND_MSG(Engine::get_singleton()->is_abort_on_gpu_errors_enabled(),
					"Crashing, because abort on GPU errors is enabled.");
			break;
		case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
			break; // Shouldn't happen, only handling to make compilers happy.
	}

	return VK_FALSE;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::_debug_report_callback(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT objectType,
		uint64_t object,
		size_t location,
		int32_t messageCode,
		const char *pLayerPrefix,
		const char *pMessage,
		void *pUserData) {
	String debugMessage = String("Vulkan Debug Report: object - ") +
			String::num_int64(object) + "\n" + pMessage;

	switch (flags) {
		case VK_DEBUG_REPORT_DEBUG_BIT_EXT:
		case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
			print_line(debugMessage);
			break;
		case VK_DEBUG_REPORT_WARNING_BIT_EXT:
		case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
			WARN_PRINT(debugMessage);
			break;
		case VK_DEBUG_REPORT_ERROR_BIT_EXT:
			ERR_PRINT(debugMessage);
			break;
	}

	return VK_FALSE;
}

VkBool32 VulkanContext::_check_layers(uint32_t check_count, const char *const *check_names, uint32_t layer_count, VkLayerProperties *layers) {
	for (uint32_t i = 0; i < check_count; i++) {
		VkBool32 found = 0;
		for (uint32_t j = 0; j < layer_count; j++) {
			if (!strcmp(check_names[i], layers[j].layerName)) {
				found = 1;
				break;
			}
		}
		if (!found) {
			WARN_PRINT("Can't find layer: " + String(check_names[i]));
			return 0;
		}
	}
	return 1;
}

Error VulkanContext::_get_preferred_validation_layers(uint32_t *count, const char *const **names) {
	static const LocalVector<LocalVector<const char *>> instance_validation_layers_alt{
		// Preferred set of validation layers.
		{ "VK_LAYER_KHRONOS_validation" },

		// Alternative (deprecated, removed in SDK 1.1.126.0) set of validation layers.
		{ "VK_LAYER_LUNARG_standard_validation" },

		// Alternative (deprecated, removed in SDK 1.1.121.1) set of validation layers.
		{ "VK_LAYER_GOOGLE_threading", "VK_LAYER_LUNARG_parameter_validation", "VK_LAYER_LUNARG_object_tracker", "VK_LAYER_LUNARG_core_validation", "VK_LAYER_GOOGLE_unique_objects" }
	};

	// Clear out-arguments.
	*count = 0;
	if (names != nullptr) {
		*names = nullptr;
	}

	VkResult err;
	uint32_t instance_layer_count;

	err = vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
	if (err) {
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	if (instance_layer_count < 1) {
		return OK;
	}

	VkLayerProperties *instance_layers = (VkLayerProperties *)malloc(sizeof(VkLayerProperties) * instance_layer_count);
	err = vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layers);
	if (err) {
		free(instance_layers);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	for (const LocalVector<const char *> &layer : instance_validation_layers_alt) {
		if (_check_layers(layer.size(), layer.ptr(), instance_layer_count, instance_layers)) {
			*count = layer.size();
			if (names != nullptr) {
				*names = layer.ptr();
			}
			break;
		}
	}

	free(instance_layers);

	return OK;
}

typedef VkResult(VKAPI_PTR *_vkEnumerateInstanceVersion)(uint32_t *);

Error VulkanContext::_obtain_vulkan_version() {
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkApplicationInfo.html#_description
	// For Vulkan 1.0 vkEnumerateInstanceVersion is not available, including not in the loader we compile against on Android.
	_vkEnumerateInstanceVersion func = (_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
	if (func != nullptr) {
		uint32_t api_version;
		VkResult res = func(&api_version);
		if (res == VK_SUCCESS) {
			instance_api_version = api_version;
		} else {
			// According to the documentation this shouldn't fail with anything except a memory allocation error
			// in which case we're in deep trouble anyway.
			ERR_FAIL_V(ERR_CANT_CREATE);
		}
	} else {
		print_line("vkEnumerateInstanceVersion not available, assuming Vulkan 1.0.");
		instance_api_version = VK_API_VERSION_1_0;
	}

	return OK;
}

bool VulkanContext::instance_extensions_initialized = false;
HashMap<CharString, bool> VulkanContext::requested_instance_extensions;

void VulkanContext::register_requested_instance_extension(const CharString &extension_name, bool p_required) {
	ERR_FAIL_COND_MSG(instance_extensions_initialized, "You can only registered extensions before the Vulkan instance is created");
	ERR_FAIL_COND(requested_instance_extensions.has(extension_name));

	requested_instance_extensions[extension_name] = p_required;
}

Error VulkanContext::_initialize_instance_extensions() {
	enabled_instance_extension_names.clear();

	// Make sure our core extensions are here
	register_requested_instance_extension(VK_KHR_SURFACE_EXTENSION_NAME, true);
	register_requested_instance_extension(_get_platform_surface_extension(), true);

	if (_use_validation_layers()) {
		register_requested_instance_extension(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, false);
	}

	// This extension allows us to use the properties2 features to query additional device capabilities
	register_requested_instance_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, false);

	// Only enable debug utils in verbose mode or DEV_ENABLED.
	// End users would get spammed with messages of varying verbosity due to the
	// mess that thirdparty layers/extensions and drivers seem to leave in their
	// wake, making the Windows registry a bottomless pit of broken layer JSON.
#ifdef DEV_ENABLED
	bool want_debug_utils = true;
#else
	bool want_debug_utils = OS::get_singleton()->is_stdout_verbose();
#endif
	if (want_debug_utils) {
		register_requested_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, false);
	}

	// Load instance extensions that are available...
	uint32_t instance_extension_count = 0;
	VkResult err = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
	ERR_FAIL_COND_V(err != VK_SUCCESS && err != VK_INCOMPLETE, ERR_CANT_CREATE);
	ERR_FAIL_COND_V_MSG(instance_extension_count == 0, ERR_CANT_CREATE, "No instance extensions found, is a driver installed?");

	VkExtensionProperties *instance_extensions = (VkExtensionProperties *)malloc(sizeof(VkExtensionProperties) * instance_extension_count);
	err = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, instance_extensions);
	if (err != VK_SUCCESS && err != VK_INCOMPLETE) {
		free(instance_extensions);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}
#ifdef DEV_ENABLED
	for (uint32_t i = 0; i < instance_extension_count; i++) {
		print_verbose(String("VULKAN: Found instance extension ") + String::utf8(instance_extensions[i].extensionName));
	}
#endif

	// Enable all extensions that are supported and requested
	for (uint32_t i = 0; i < instance_extension_count; i++) {
		CharString extension_name(instance_extensions[i].extensionName);
		if (requested_instance_extensions.has(extension_name)) {
			enabled_instance_extension_names.insert(extension_name);
		}
	}

	// Now check our requested extensions
	for (KeyValue<CharString, bool> &requested_extension : requested_instance_extensions) {
		if (!enabled_instance_extension_names.has(requested_extension.key)) {
			if (requested_extension.value) {
				free(instance_extensions);
				ERR_FAIL_V_MSG(ERR_BUG, String("Required extension ") + String::utf8(requested_extension.key) + String(" not found, is a driver installed?"));
			} else {
				print_verbose(String("Optional extension ") + String::utf8(requested_extension.key) + String(" not found"));
			}
		}
	}

	free(instance_extensions);

	instance_extensions_initialized = true;
	return OK;
}

bool VulkanContext::device_extensions_initialized = false;
HashMap<CharString, bool> VulkanContext::requested_device_extensions;

void VulkanContext::register_requested_device_extension(const CharString &extension_name, bool p_required) {
	ERR_FAIL_COND_MSG(device_extensions_initialized, "You can only registered extensions before the Vulkan instance is created");
	ERR_FAIL_COND(requested_device_extensions.has(extension_name));

	requested_device_extensions[extension_name] = p_required;
}

Error VulkanContext::_initialize_device_extensions() {
	// Look for device extensions.
	enabled_device_extension_names.clear();

	// Make sure our core extensions are here
	register_requested_device_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, true);

	register_requested_device_extension(VK_KHR_MULTIVIEW_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_16BIT_STORAGE_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME, false);
	register_requested_device_extension(VK_KHR_MAINTENANCE_2_EXTENSION_NAME, false);
	register_requested_device_extension(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME, false);
	register_requested_device_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, false);

	if (Engine::get_singleton()->is_generate_spirv_debug_info_enabled()) {
		register_requested_device_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, true);
	}

	// TODO consider the following extensions:
	// - VK_KHR_spirv_1_4
	// - VK_KHR_swapchain_mutable_format
	// - VK_EXT_full_screen_exclusive
	// - VK_EXT_hdr_metadata
	// - VK_KHR_depth_stencil_resolve

	// Even though the user "enabled" the extension via the command
	// line, we must make sure that it's enumerated for use with the
	// device.  Therefore, disable it here, and re-enable it again if
	// enumerated.
	if (VK_KHR_incremental_present_enabled) {
		register_requested_device_extension(VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME, false);
	}
	if (VK_GOOGLE_display_timing_enabled) {
		register_requested_device_extension(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME, false);
	}

	// obtain available device extensions
	uint32_t device_extension_count = 0;
	VkResult err = vkEnumerateDeviceExtensionProperties(gpu, nullptr, &device_extension_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	ERR_FAIL_COND_V_MSG(device_extension_count == 0, ERR_CANT_CREATE,
			"vkEnumerateDeviceExtensionProperties failed to find any extensions\n\n"
			"Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
			"vkCreateInstance Failure");

	VkExtensionProperties *device_extensions = (VkExtensionProperties *)malloc(sizeof(VkExtensionProperties) * device_extension_count);
	err = vkEnumerateDeviceExtensionProperties(gpu, nullptr, &device_extension_count, device_extensions);
	if (err) {
		free(device_extensions);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

#ifdef DEV_ENABLED
	for (uint32_t i = 0; i < device_extension_count; i++) {
		print_verbose(String("VULKAN: Found device extension ") + String::utf8(device_extensions[i].extensionName));
	}
#endif

	// Enable all extensions that are supported and requested
	for (uint32_t i = 0; i < device_extension_count; i++) {
		CharString extension_name(device_extensions[i].extensionName);
		if (requested_device_extensions.has(extension_name)) {
			enabled_device_extension_names.insert(extension_name);
		}
	}

	// Now check our requested extensions
	for (KeyValue<CharString, bool> &requested_extension : requested_device_extensions) {
		if (!enabled_device_extension_names.has(requested_extension.key)) {
			if (requested_extension.value) {
				free(device_extensions);
				ERR_FAIL_V_MSG(ERR_BUG,
						String("vkEnumerateDeviceExtensionProperties failed to find the ") + String::utf8(requested_extension.key) + String(" extension.\n\nDo you have a compatible Vulkan installable client driver (ICD) installed?\nvkCreateInstance Failure"));
			} else {
				print_verbose(String("Optional extension ") + String::utf8(requested_extension.key) + String(" not found"));
			}
		}
	}

	free(device_extensions);

	device_extensions_initialized = true;
	return OK;
}

uint32_t VulkanContext::SubgroupCapabilities::supported_stages_flags_rd() const {
	uint32_t flags = 0;

	if (supportedStages & VK_SHADER_STAGE_VERTEX_BIT) {
		flags += RenderingDevice::ShaderStage::SHADER_STAGE_VERTEX_BIT;
	}
	if (supportedStages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
		flags += RenderingDevice::ShaderStage::SHADER_STAGE_TESSELATION_CONTROL_BIT;
	}
	if (supportedStages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
		flags += RenderingDevice::ShaderStage::SHADER_STAGE_TESSELATION_EVALUATION_BIT;
	}
	// if (supportedStages & VK_SHADER_STAGE_GEOMETRY_BIT) {
	// 	flags += RenderingDevice::ShaderStage::SHADER_STAGE_GEOMETRY_BIT;
	// }
	if (supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT) {
		flags += RenderingDevice::ShaderStage::SHADER_STAGE_FRAGMENT_BIT;
	}
	if (supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) {
		flags += RenderingDevice::ShaderStage::SHADER_STAGE_COMPUTE_BIT;
	}

	return flags;
}

String VulkanContext::SubgroupCapabilities::supported_stages_desc() const {
	String res;

	if (supportedStages & VK_SHADER_STAGE_VERTEX_BIT) {
		res += ", STAGE_VERTEX";
	}
	if (supportedStages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
		res += ", STAGE_TESSELLATION_CONTROL";
	}
	if (supportedStages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
		res += ", STAGE_TESSELLATION_EVALUATION";
	}
	if (supportedStages & VK_SHADER_STAGE_GEOMETRY_BIT) {
		res += ", STAGE_GEOMETRY";
	}
	if (supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT) {
		res += ", STAGE_FRAGMENT";
	}
	if (supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) {
		res += ", STAGE_COMPUTE";
	}

	// These are not defined on Android GRMBL.
	if (supportedStages & 0x00000100 /* VK_SHADER_STAGE_RAYGEN_BIT_KHR */) {
		res += ", STAGE_RAYGEN_KHR";
	}
	if (supportedStages & 0x00000200 /* VK_SHADER_STAGE_ANY_HIT_BIT_KHR */) {
		res += ", STAGE_ANY_HIT_KHR";
	}
	if (supportedStages & 0x00000400 /* VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR */) {
		res += ", STAGE_CLOSEST_HIT_KHR";
	}
	if (supportedStages & 0x00000800 /* VK_SHADER_STAGE_MISS_BIT_KHR */) {
		res += ", STAGE_MISS_KHR";
	}
	if (supportedStages & 0x00001000 /* VK_SHADER_STAGE_INTERSECTION_BIT_KHR */) {
		res += ", STAGE_INTERSECTION_KHR";
	}
	if (supportedStages & 0x00002000 /* VK_SHADER_STAGE_CALLABLE_BIT_KHR */) {
		res += ", STAGE_CALLABLE_KHR";
	}
	if (supportedStages & 0x00000040 /* VK_SHADER_STAGE_TASK_BIT_NV */) {
		res += ", STAGE_TASK_NV";
	}
	if (supportedStages & 0x00000080 /* VK_SHADER_STAGE_MESH_BIT_NV */) {
		res += ", STAGE_MESH_NV";
	}

	return res.substr(2); // Remove first ", ".
}

uint32_t VulkanContext::SubgroupCapabilities::supported_operations_flags_rd() const {
	uint32_t flags = 0;

	if (supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_BASIC_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_VOTE_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_ARITHMETIC_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_BALLOT_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_SHUFFLE_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_SHUFFLE_RELATIVE_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_CLUSTERED_BIT;
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) {
		flags += RenderingDevice::SubgroupOperations::SUBGROUP_QUAD_BIT;
	}

	return flags;
}

String VulkanContext::SubgroupCapabilities::supported_operations_desc() const {
	String res;

	if (supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) {
		res += ", FEATURE_BASIC";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) {
		res += ", FEATURE_VOTE";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
		res += ", FEATURE_ARITHMETIC";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) {
		res += ", FEATURE_BALLOT";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) {
		res += ", FEATURE_SHUFFLE";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) {
		res += ", FEATURE_SHUFFLE_RELATIVE";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) {
		res += ", FEATURE_CLUSTERED";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) {
		res += ", FEATURE_QUAD";
	}
	if (supportedOperations & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV) {
		res += ", FEATURE_PARTITIONED_NV";
	}

	return res.substr(2); // Remove first ", ".
}

Error VulkanContext::_check_capabilities() {
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_multiview.html
	// https://www.khronos.org/blog/vulkan-subgroup-tutorial

	// For Vulkan 1.0 vkGetPhysicalDeviceProperties2 is not available, including not in the loader we compile against on Android.

	// So we check if the functions are accessible by getting their function pointers and skipping if not
	// (note that the desktop loader does a better job here but the android loader doesn't.)

	// Assume not supported until proven otherwise.
	vrs_capabilities.pipeline_vrs_supported = false;
	vrs_capabilities.primitive_vrs_supported = false;
	vrs_capabilities.attachment_vrs_supported = false;
	vrs_capabilities.min_texel_size = Size2i();
	vrs_capabilities.max_texel_size = Size2i();
	vrs_capabilities.texel_size = Size2i();
	multiview_capabilities.is_supported = false;
	multiview_capabilities.geometry_shader_is_supported = false;
	multiview_capabilities.tessellation_shader_is_supported = false;
	multiview_capabilities.max_view_count = 0;
	multiview_capabilities.max_instance_count = 0;
	subgroup_capabilities.size = 0;
	subgroup_capabilities.min_size = 0;
	subgroup_capabilities.max_size = 0;
	subgroup_capabilities.supportedStages = 0;
	subgroup_capabilities.supportedOperations = 0;
	subgroup_capabilities.quadOperationsInAllStages = false;
	subgroup_capabilities.size_control_is_supported = false;
	shader_capabilities.shader_float16_is_supported = false;
	shader_capabilities.shader_int8_is_supported = false;
	storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported = false;
	storage_buffer_capabilities.uniform_and_storage_buffer_16_bit_access_is_supported = false;
	storage_buffer_capabilities.storage_push_constant_16_is_supported = false;
	storage_buffer_capabilities.storage_input_output_16 = false;

	if (is_instance_extension_enabled(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
		// Check for extended features.
		PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2_func = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2");
		if (vkGetPhysicalDeviceFeatures2_func == nullptr) {
			// In Vulkan 1.0 might be accessible under its original extension name.
			vkGetPhysicalDeviceFeatures2_func = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2KHR");
		}
		if (vkGetPhysicalDeviceFeatures2_func != nullptr) {
			// Check our extended features.
			void *next = nullptr;

			// We must check that the relative extension is present before assuming a
			// feature as enabled.
			// See also: https://github.com/godotengine/godot/issues/65409

			VkPhysicalDeviceVulkan12Features device_features_vk12 = {};
			VkPhysicalDeviceShaderFloat16Int8FeaturesKHR shader_features = {};
			VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrs_features = {};
			VkPhysicalDevice16BitStorageFeaturesKHR storage_feature = {};
			VkPhysicalDeviceMultiviewFeatures multiview_features = {};
			VkPhysicalDevicePipelineCreationCacheControlFeatures pipeline_cache_control_features = {};

			if (device_api_version >= VK_API_VERSION_1_2) {
				device_features_vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
				device_features_vk12.pNext = next;
				next = &device_features_vk12;
			} else {
				if (is_device_extension_enabled(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
					shader_features = {
						/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
						/*pNext*/ next,
						/*shaderFloat16*/ false,
						/*shaderInt8*/ false,
					};
					next = &shader_features;
				}
			}

			if (is_device_extension_enabled(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
				vrs_features = {
					/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
					/*pNext*/ next,
					/*pipelineFragmentShadingRate*/ false,
					/*primitiveFragmentShadingRate*/ false,
					/*attachmentFragmentShadingRate*/ false,
				};
				next = &vrs_features;
			}

			if (is_device_extension_enabled(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
				storage_feature = {
					/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
					/*pNext*/ next,
					/*storageBuffer16BitAccess*/ false,
					/*uniformAndStorageBuffer16BitAccess*/ false,
					/*storagePushConstant16*/ false,
					/*storageInputOutput16*/ false,
				};
				next = &storage_feature;
			}

			if (is_device_extension_enabled(VK_KHR_MULTIVIEW_EXTENSION_NAME)) {
				multiview_features = {
					/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
					/*pNext*/ next,
					/*multiview*/ false,
					/*multiviewGeometryShader*/ false,
					/*multiviewTessellationShader*/ false,
				};
				next = &multiview_features;
			}

			if (is_device_extension_enabled(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME)) {
				pipeline_cache_control_features = {
					/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES,
					/*pNext*/ next,
					/*pipelineCreationCacheControl*/ false,
				};
				next = &pipeline_cache_control_features;
			}

			VkPhysicalDeviceFeatures2 device_features;
			device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			device_features.pNext = next;

			vkGetPhysicalDeviceFeatures2_func(gpu, &device_features);

			if (device_api_version >= VK_API_VERSION_1_2) {
#ifdef MACOS_ENABLED
				ERR_FAIL_COND_V_MSG(!device_features_vk12.shaderSampledImageArrayNonUniformIndexing, ERR_CANT_CREATE, "Your GPU doesn't support shaderSampledImageArrayNonUniformIndexing which is required to use the Vulkan-based renderers in Godot.");
#endif

				if (is_device_extension_enabled(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
					shader_capabilities.shader_float16_is_supported = device_features_vk12.shaderFloat16;
					shader_capabilities.shader_int8_is_supported = device_features_vk12.shaderInt8;
				}
			} else {
				if (is_device_extension_enabled(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
					shader_capabilities.shader_float16_is_supported = shader_features.shaderFloat16;
					shader_capabilities.shader_int8_is_supported = shader_features.shaderInt8;
				}
			}

			if (is_device_extension_enabled(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
				vrs_capabilities.pipeline_vrs_supported = vrs_features.pipelineFragmentShadingRate;
				vrs_capabilities.primitive_vrs_supported = vrs_features.primitiveFragmentShadingRate;
				vrs_capabilities.attachment_vrs_supported = vrs_features.attachmentFragmentShadingRate;
			}

			if (is_device_extension_enabled(VK_KHR_MULTIVIEW_EXTENSION_NAME)) {
				multiview_capabilities.is_supported = multiview_features.multiview;
				multiview_capabilities.geometry_shader_is_supported = multiview_features.multiviewGeometryShader;
				multiview_capabilities.tessellation_shader_is_supported = multiview_features.multiviewTessellationShader;
			}

			if (is_device_extension_enabled(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
				storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported = storage_feature.storageBuffer16BitAccess;
				storage_buffer_capabilities.uniform_and_storage_buffer_16_bit_access_is_supported = storage_feature.uniformAndStorageBuffer16BitAccess;
				storage_buffer_capabilities.storage_push_constant_16_is_supported = storage_feature.storagePushConstant16;
				storage_buffer_capabilities.storage_input_output_16 = storage_feature.storageInputOutput16;
			}

			if (is_device_extension_enabled(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME)) {
				pipeline_cache_control_support = pipeline_cache_control_features.pipelineCreationCacheControl;
			}
		}

		// Check extended properties.
		PFN_vkGetPhysicalDeviceProperties2 device_properties_func = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2");
		if (device_properties_func == nullptr) {
			// In Vulkan 1.0 might be accessible under its original extension name.
			device_properties_func = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2KHR");
		}
		if (device_properties_func != nullptr) {
			VkPhysicalDeviceFragmentShadingRatePropertiesKHR vrsProperties{};
			VkPhysicalDeviceMultiviewProperties multiviewProperties{};
			VkPhysicalDeviceSubgroupProperties subgroupProperties{};
			VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties = {};
			VkPhysicalDeviceProperties2 physicalDeviceProperties{};
			void *nextptr = nullptr;

			if (device_api_version >= VK_API_VERSION_1_1) { // Vulkan 1.1 or higher
				subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
				subgroupProperties.pNext = nextptr;

				nextptr = &subgroupProperties;

				subgroup_capabilities.size_control_is_supported = is_device_extension_enabled(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

				if (subgroup_capabilities.size_control_is_supported) {
					subgroupSizeControlProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
					subgroupSizeControlProperties.pNext = nextptr;

					nextptr = &subgroupSizeControlProperties;
				}
			}

			if (multiview_capabilities.is_supported) {
				multiviewProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES;
				multiviewProperties.pNext = nextptr;

				nextptr = &multiviewProperties;
			}

			if (vrs_capabilities.attachment_vrs_supported) {
				vrsProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
				vrsProperties.pNext = nextptr;

				nextptr = &vrsProperties;
			}

			physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			physicalDeviceProperties.pNext = nextptr;

			device_properties_func(gpu, &physicalDeviceProperties);

			subgroup_capabilities.size = subgroupProperties.subgroupSize;
			subgroup_capabilities.min_size = subgroupProperties.subgroupSize;
			subgroup_capabilities.max_size = subgroupProperties.subgroupSize;
			subgroup_capabilities.supportedStages = subgroupProperties.supportedStages;
			subgroup_capabilities.supportedOperations = subgroupProperties.supportedOperations;
			// Note: quadOperationsInAllStages will be true if:
			// - supportedStages has VK_SHADER_STAGE_ALL_GRAPHICS + VK_SHADER_STAGE_COMPUTE_BIT.
			// - supportedOperations has VK_SUBGROUP_FEATURE_QUAD_BIT.
			subgroup_capabilities.quadOperationsInAllStages = subgroupProperties.quadOperationsInAllStages;

			if (subgroup_capabilities.size_control_is_supported && (subgroupSizeControlProperties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
				subgroup_capabilities.min_size = subgroupSizeControlProperties.minSubgroupSize;
				subgroup_capabilities.max_size = subgroupSizeControlProperties.maxSubgroupSize;
			}

			if (vrs_capabilities.pipeline_vrs_supported || vrs_capabilities.primitive_vrs_supported || vrs_capabilities.attachment_vrs_supported) {
				print_verbose("- Vulkan Variable Rate Shading supported:");
				if (vrs_capabilities.pipeline_vrs_supported) {
					print_verbose("  Pipeline fragment shading rate");
				}
				if (vrs_capabilities.primitive_vrs_supported) {
					print_verbose("  Primitive fragment shading rate");
				}
				if (vrs_capabilities.attachment_vrs_supported) {
					// TODO expose these somehow to the end user.
					vrs_capabilities.min_texel_size.x = vrsProperties.minFragmentShadingRateAttachmentTexelSize.width;
					vrs_capabilities.min_texel_size.y = vrsProperties.minFragmentShadingRateAttachmentTexelSize.height;
					vrs_capabilities.max_texel_size.x = vrsProperties.maxFragmentShadingRateAttachmentTexelSize.width;
					vrs_capabilities.max_texel_size.y = vrsProperties.maxFragmentShadingRateAttachmentTexelSize.height;

					// We'll attempt to default to a texel size of 16x16
					vrs_capabilities.texel_size.x = CLAMP(16, vrs_capabilities.min_texel_size.x, vrs_capabilities.max_texel_size.x);
					vrs_capabilities.texel_size.y = CLAMP(16, vrs_capabilities.min_texel_size.y, vrs_capabilities.max_texel_size.y);

					print_verbose(String("  Attachment fragment shading rate") + String(", min texel size: (") + itos(vrs_capabilities.min_texel_size.x) + String(", ") + itos(vrs_capabilities.min_texel_size.y) + String(")") + String(", max texel size: (") + itos(vrs_capabilities.max_texel_size.x) + String(", ") + itos(vrs_capabilities.max_texel_size.y) + String(")"));
				}

			} else {
				print_verbose("- Vulkan Variable Rate Shading not supported");
			}

			if (multiview_capabilities.is_supported) {
				multiview_capabilities.max_view_count = multiviewProperties.maxMultiviewViewCount;
				multiview_capabilities.max_instance_count = multiviewProperties.maxMultiviewInstanceIndex;

				print_verbose("- Vulkan multiview supported:");
				print_verbose("  max view count: " + itos(multiview_capabilities.max_view_count));
				print_verbose("  max instances: " + itos(multiview_capabilities.max_instance_count));
			} else {
				print_verbose("- Vulkan multiview not supported");
			}

			print_verbose("- Vulkan subgroup:");
			print_verbose("  size: " + itos(subgroup_capabilities.size));
			print_verbose("  min size: " + itos(subgroup_capabilities.min_size));
			print_verbose("  max size: " + itos(subgroup_capabilities.max_size));
			print_verbose("  stages: " + subgroup_capabilities.supported_stages_desc());
			print_verbose("  supported ops: " + subgroup_capabilities.supported_operations_desc());
			if (subgroup_capabilities.quadOperationsInAllStages) {
				print_verbose("  quad operations in all stages");
			}
		} else {
			print_verbose("- Couldn't call vkGetPhysicalDeviceProperties2");
		}
	}

	return OK;
}

Error VulkanContext::_create_instance() {
	// Obtain Vulkan version.
	_obtain_vulkan_version();

	// Initialize extensions.
	{
		Error err = _initialize_instance_extensions();
		if (err != OK) {
			return err;
		}
	}

	int enabled_extension_count = 0;
	const char *enabled_extension_names[MAX_EXTENSIONS];
	ERR_FAIL_COND_V(enabled_instance_extension_names.size() > MAX_EXTENSIONS, ERR_CANT_CREATE);
	for (const CharString &extension_name : enabled_instance_extension_names) {
		enabled_extension_names[enabled_extension_count++] = extension_name.ptr();
	}

	// We'll set application version to the Vulkan version we're developing against, even if our instance is based on
	// an older Vulkan version, devices can still support newer versions of Vulkan.
	// The exception is when we're on Vulkan 1.0, we should not set this to anything but 1.0.
	// Note that this value is only used by validation layers to warn us about version issues.
	uint32_t application_api_version = instance_api_version == VK_API_VERSION_1_0 ? VK_API_VERSION_1_0 : VK_API_VERSION_1_2;

	CharString cs = GLOBAL_GET("application/config/name").operator String().utf8();
	const VkApplicationInfo app = {
		/*sType*/ VK_STRUCTURE_TYPE_APPLICATION_INFO,
		/*pNext*/ nullptr,
		/*pApplicationName*/ cs.get_data(),
		/*applicationVersion*/ 0, // It would be really nice if we store a version number in project settings, say "application/config/version"
		/*pEngineName*/ VERSION_NAME,
		/*engineVersion*/ VK_MAKE_VERSION(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH),
		/*apiVersion*/ application_api_version
	};
	VkInstanceCreateInfo inst_info{};
	inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	inst_info.pApplicationInfo = &app;
	inst_info.enabledExtensionCount = enabled_extension_count;
	inst_info.ppEnabledExtensionNames = (const char *const *)enabled_extension_names;
	if (_use_validation_layers()) {
		_get_preferred_validation_layers(&inst_info.enabledLayerCount, &inst_info.ppEnabledLayerNames);
	}

	/*
	 * This is info for a temp callback to use during CreateInstance.
	 * After the instance is created, we use the instance-based
	 * function to register the final callback.
	 */
	VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info = {};
	VkDebugReportCallbackCreateInfoEXT dbg_report_callback_create_info = {};
	if (is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		// VK_EXT_debug_utils style.
		dbg_messenger_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		dbg_messenger_create_info.pNext = nullptr;
		dbg_messenger_create_info.flags = 0;
		dbg_messenger_create_info.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		dbg_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		dbg_messenger_create_info.pfnUserCallback = _debug_messenger_callback;
		dbg_messenger_create_info.pUserData = this;
		inst_info.pNext = &dbg_messenger_create_info;
	} else if (is_instance_extension_enabled(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
		dbg_report_callback_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		dbg_report_callback_create_info.flags = VK_DEBUG_REPORT_INFORMATION_BIT_EXT |
				VK_DEBUG_REPORT_WARNING_BIT_EXT |
				VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
				VK_DEBUG_REPORT_ERROR_BIT_EXT |
				VK_DEBUG_REPORT_DEBUG_BIT_EXT;
		dbg_report_callback_create_info.pfnCallback = _debug_report_callback;
		dbg_report_callback_create_info.pUserData = this;
		inst_info.pNext = &dbg_report_callback_create_info;
	}

	VkResult err;

	if (vulkan_hooks) {
		if (!vulkan_hooks->create_vulkan_instance(&inst_info, &inst)) {
			return ERR_CANT_CREATE;
		}
	} else {
		err = vkCreateInstance(&inst_info, nullptr, &inst);
		ERR_FAIL_COND_V_MSG(err == VK_ERROR_INCOMPATIBLE_DRIVER, ERR_CANT_CREATE,
				"Cannot find a compatible Vulkan installable client driver (ICD).\n\n"
				"vkCreateInstance Failure");
		ERR_FAIL_COND_V_MSG(err == VK_ERROR_EXTENSION_NOT_PRESENT, ERR_CANT_CREATE,
				"Cannot find a specified extension library.\n"
				"Make sure your layers path is set appropriately.\n"
				"vkCreateInstance Failure");
		ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE,
				"vkCreateInstance failed.\n\n"
				"Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
				"Please look at the Getting Started guide for additional information.\n"
				"vkCreateInstance Failure");
	}

	inst_initialized = true;

#ifdef USE_VOLK
	volkLoadInstance(inst);
#endif

	if (is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		// Setup VK_EXT_debug_utils function pointers always (we use them for debug labels and names).
		CreateDebugUtilsMessengerEXT =
				(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
		DestroyDebugUtilsMessengerEXT =
				(PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
		SubmitDebugUtilsMessageEXT =
				(PFN_vkSubmitDebugUtilsMessageEXT)vkGetInstanceProcAddr(inst, "vkSubmitDebugUtilsMessageEXT");
		CmdBeginDebugUtilsLabelEXT =
				(PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(inst, "vkCmdBeginDebugUtilsLabelEXT");
		CmdEndDebugUtilsLabelEXT =
				(PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(inst, "vkCmdEndDebugUtilsLabelEXT");
		CmdInsertDebugUtilsLabelEXT =
				(PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(inst, "vkCmdInsertDebugUtilsLabelEXT");
		SetDebugUtilsObjectNameEXT =
				(PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(inst, "vkSetDebugUtilsObjectNameEXT");
		if (nullptr == CreateDebugUtilsMessengerEXT || nullptr == DestroyDebugUtilsMessengerEXT ||
				nullptr == SubmitDebugUtilsMessageEXT || nullptr == CmdBeginDebugUtilsLabelEXT ||
				nullptr == CmdEndDebugUtilsLabelEXT || nullptr == CmdInsertDebugUtilsLabelEXT ||
				nullptr == SetDebugUtilsObjectNameEXT) {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE,
					"GetProcAddr: Failed to init VK_EXT_debug_utils\n"
					"GetProcAddr: Failure");
		}

		err = CreateDebugUtilsMessengerEXT(inst, &dbg_messenger_create_info, nullptr, &dbg_messenger);
		switch (err) {
			case VK_SUCCESS:
				break;
			case VK_ERROR_OUT_OF_HOST_MEMORY:
				ERR_FAIL_V_MSG(ERR_CANT_CREATE,
						"CreateDebugUtilsMessengerEXT: out of host memory\n"
						"CreateDebugUtilsMessengerEXT Failure");
				break;
			default:
				ERR_FAIL_V_MSG(ERR_CANT_CREATE,
						"CreateDebugUtilsMessengerEXT: unknown failure\n"
						"CreateDebugUtilsMessengerEXT Failure");
				ERR_FAIL_V(ERR_CANT_CREATE);
				break;
		}
	} else if (is_instance_extension_enabled(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)) {
		CreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugReportCallbackEXT");
		DebugReportMessageEXT = (PFN_vkDebugReportMessageEXT)vkGetInstanceProcAddr(inst, "vkDebugReportMessageEXT");
		DestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(inst, "vkDestroyDebugReportCallbackEXT");

		if (nullptr == CreateDebugReportCallbackEXT || nullptr == DebugReportMessageEXT || nullptr == DestroyDebugReportCallbackEXT) {
			ERR_FAIL_V_MSG(ERR_CANT_CREATE,
					"GetProcAddr: Failed to init VK_EXT_debug_report\n"
					"GetProcAddr: Failure");
		}

		err = CreateDebugReportCallbackEXT(inst, &dbg_report_callback_create_info, nullptr, &dbg_debug_report);
		switch (err) {
			case VK_SUCCESS:
				break;
			case VK_ERROR_OUT_OF_HOST_MEMORY:
				ERR_FAIL_V_MSG(ERR_CANT_CREATE,
						"CreateDebugReportCallbackEXT: out of host memory\n"
						"CreateDebugReportCallbackEXT Failure");
				break;
			default:
				ERR_FAIL_V_MSG(ERR_CANT_CREATE,
						"CreateDebugReportCallbackEXT: unknown failure\n"
						"CreateDebugReportCallbackEXT Failure");
				ERR_FAIL_V(ERR_CANT_CREATE);
				break;
		}
	}

	return OK;
}

Error VulkanContext::_create_physical_device(VkSurfaceKHR p_surface) {
	// Make initial call to query gpu_count, then second call for gpu info.
	uint32_t gpu_count = 0;
	VkResult err = vkEnumeratePhysicalDevices(inst, &gpu_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	ERR_FAIL_COND_V_MSG(gpu_count == 0, ERR_CANT_CREATE,
			"vkEnumeratePhysicalDevices reported zero accessible devices.\n\n"
			"Do you have a compatible Vulkan installable client driver (ICD) installed?\n"
			"vkEnumeratePhysicalDevices Failure");

	VkPhysicalDevice *physical_devices = (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * gpu_count);
	err = vkEnumeratePhysicalDevices(inst, &gpu_count, physical_devices);
	if (err) {
		free(physical_devices);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	static const struct {
		uint32_t id;
		const char *name;
	} vendor_names[] = {
		{ 0x1002, "AMD" },
		{ 0x1010, "ImgTec" },
		{ 0x106B, "Apple" },
		{ 0x10DE, "NVIDIA" },
		{ 0x13B5, "ARM" },
		{ 0x5143, "Qualcomm" },
		{ 0x8086, "Intel" },
		{ 0, nullptr },
	};

	int32_t device_index = -1;
	if (vulkan_hooks) {
		if (!vulkan_hooks->get_physical_device(&gpu)) {
			return ERR_CANT_CREATE;
		}

		// Not really needed but nice to print the correct entry.
		for (uint32_t i = 0; i < gpu_count; ++i) {
			if (physical_devices[i] == gpu) {
				device_index = i;
				break;
			}
		}
	} else {
		// TODO: At least on Linux Laptops integrated GPUs fail with Vulkan in many instances.
		// The device should really be a preference, but for now choosing a discrete GPU over the
		// integrated one is better than the default.

		int type_selected = -1;
		print_verbose("Vulkan devices:");
		for (uint32_t i = 0; i < gpu_count; ++i) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(physical_devices[i], &props);

			bool present_supported = false;

			uint32_t device_queue_family_count = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &device_queue_family_count, nullptr);
			VkQueueFamilyProperties *device_queue_props = (VkQueueFamilyProperties *)malloc(device_queue_family_count * sizeof(VkQueueFamilyProperties));
			vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &device_queue_family_count, device_queue_props);
			for (uint32_t j = 0; j < device_queue_family_count; j++) {
				if ((device_queue_props[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
					VkBool32 supports;
					err = vkGetPhysicalDeviceSurfaceSupportKHR(
							physical_devices[i], j, p_surface, &supports);
					if (err == VK_SUCCESS && supports) {
						present_supported = true;
					} else {
						continue;
					}
				}
			}
			String name = String::utf8(props.deviceName);
			String vendor = "Unknown";
			String dev_type;
			switch (props.deviceType) {
				case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: {
					dev_type = "Discrete";
				} break;
				case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: {
					dev_type = "Integrated";
				} break;
				case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: {
					dev_type = "Virtual";
				} break;
				case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU: {
					dev_type = "CPU";
				} break;
				default: {
					dev_type = "Other";
				} break;
			}
			uint32_t vendor_idx = 0;
			while (vendor_names[vendor_idx].name != nullptr) {
				if (props.vendorID == vendor_names[vendor_idx].id) {
					vendor = vendor_names[vendor_idx].name;
					break;
				}
				vendor_idx++;
			}
			free(device_queue_props);
			print_verbose("  #" + itos(i) + ": " + vendor + " " + name + " - " + (present_supported ? "Supported" : "Unsupported") + ", " + dev_type);

			if (present_supported) { // Select first supported device of preferred type: Discrete > Integrated > Virtual > CPU > Other.
				switch (props.deviceType) {
					case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: {
						if (type_selected < 4) {
							type_selected = 4;
							device_index = i;
						}
					} break;
					case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: {
						if (type_selected < 3) {
							type_selected = 3;
							device_index = i;
						}
					} break;
					case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: {
						if (type_selected < 2) {
							type_selected = 2;
							device_index = i;
						}
					} break;
					case VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU: {
						if (type_selected < 1) {
							type_selected = 1;
							device_index = i;
						}
					} break;
					default: {
						if (type_selected < 0) {
							type_selected = 0;
							device_index = i;
						}
					} break;
				}
			}
		}

		int32_t user_device_index = Engine::get_singleton()->get_gpu_index(); // Force user selected GPU.
		if (user_device_index >= 0 && user_device_index < (int32_t)gpu_count) {
			device_index = user_device_index;
		}

		ERR_FAIL_COND_V_MSG(device_index == -1, ERR_CANT_CREATE, "None of Vulkan devices supports both graphics and present queues.");

		gpu = physical_devices[device_index];
	}

	free(physical_devices);

	// Get identifier properties.
	vkGetPhysicalDeviceProperties(gpu, &gpu_props);

	device_name = String::utf8(gpu_props.deviceName);
	device_type = gpu_props.deviceType;
	pipeline_cache_id = String::hex_encode_buffer(gpu_props.pipelineCacheUUID, VK_UUID_SIZE);
	pipeline_cache_id += "-driver-" + itos(gpu_props.driverVersion);
	{
		device_vendor = "Unknown";
		uint32_t vendor_idx = 0;
		while (vendor_names[vendor_idx].name != nullptr) {
			if (gpu_props.vendorID == vendor_names[vendor_idx].id) {
				device_vendor = vendor_names[vendor_idx].name;
				break;
			}
			vendor_idx++;
		}
	}

	// Get device version
	device_api_version = gpu_props.apiVersion;

	String rendering_method;
	if (OS::get_singleton()->get_current_rendering_method() == "mobile") {
		rendering_method = "Forward Mobile";
	} else {
		rendering_method = "Forward+";
	}

	// Output our device version
	print_line(vformat("Vulkan API %s - %s - Using Vulkan Device #%d: %s - %s", get_device_api_version(), rendering_method, device_index, device_vendor, device_name));

	{
		Error _err = _initialize_device_extensions();
		if (_err != OK) {
			return _err;
		}
	}

	// Call with nullptr data to get count.
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, nullptr);
	ERR_FAIL_COND_V(queue_family_count == 0, ERR_CANT_CREATE);

	queue_props = (VkQueueFamilyProperties *)malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, queue_props);
	// Query fine-grained feature support for this device.
	//  If app has specific feature requirements it should check supported
	//  features based on this query
	VkPhysicalDeviceFeatures features = {};
	vkGetPhysicalDeviceFeatures(gpu, &features);

	// Check required features and abort if any of them is missing.
	if (!features.imageCubeArray || !features.independentBlend) {
		String error_string = vformat("Your GPU (%s) does not support the following features which are required to use Vulkan-based renderers in Godot:\n\n", device_name);
		if (!features.imageCubeArray) {
			error_string += "- No support for image cube arrays.\n";
		}
		if (!features.independentBlend) {
			error_string += "- No support for independentBlend.\n";
		}
		error_string += "\nThis is usually a hardware limitation, so updating graphics drivers won't help in most cases.";

#if defined(ANDROID_ENABLED) || defined(IOS_ENABLED)
		// Android/iOS platform ports currently don't exit themselves when this method returns `ERR_CANT_CREATE`.
		OS::get_singleton()->alert(error_string + "\nClick OK to exit (black screen will be visible).");
#else
		OS::get_singleton()->alert(error_string + "\nClick OK to exit.");
#endif

		return ERR_CANT_CREATE;
	}

	memset(&physical_device_features, 0, sizeof(physical_device_features));
#define VK_DEVICEFEATURE_ENABLE_IF(x)            \
	if (features.x) {                            \
		physical_device_features.x = features.x; \
	} else                                       \
		((void)0)

	//
	// Opt-in to the features we actually need/use. These can be changed in the future.
	// We do this for multiple reasons:
	//
	//	1. Certain features (like sparse* stuff) cause unnecessary internal driver allocations.
	//	2. Others like shaderStorageImageMultisample are a huge red flag
	//	   (MSAA + Storage is rarely needed).
	//	3. Most features when turned off aren't actually off (we just promise the driver not to use them)
	//	   and it is validation what will complain. This allows us to target a minimum baseline.
	//
	// TODO: Allow the user to override these settings (i.e. turn off more stuff) using profiles
	// so they can target a broad range of HW. For example Mali HW does not have
	// shaderClipDistance/shaderCullDistance; thus validation would complain if such feature is used;
	// allowing them to fix the problem without even owning Mali HW to test on.
	//

	// Turn off robust buffer access, which can hamper performance on some hardware.
	//VK_DEVICEFEATURE_ENABLE_IF(robustBufferAccess);
	VK_DEVICEFEATURE_ENABLE_IF(fullDrawIndexUint32);
	VK_DEVICEFEATURE_ENABLE_IF(imageCubeArray);
	VK_DEVICEFEATURE_ENABLE_IF(independentBlend);
	VK_DEVICEFEATURE_ENABLE_IF(geometryShader);
	VK_DEVICEFEATURE_ENABLE_IF(tessellationShader);
	VK_DEVICEFEATURE_ENABLE_IF(sampleRateShading);
	VK_DEVICEFEATURE_ENABLE_IF(dualSrcBlend);
	VK_DEVICEFEATURE_ENABLE_IF(logicOp);
	VK_DEVICEFEATURE_ENABLE_IF(multiDrawIndirect);
	VK_DEVICEFEATURE_ENABLE_IF(drawIndirectFirstInstance);
	VK_DEVICEFEATURE_ENABLE_IF(depthClamp);
	VK_DEVICEFEATURE_ENABLE_IF(depthBiasClamp);
	VK_DEVICEFEATURE_ENABLE_IF(fillModeNonSolid);
	VK_DEVICEFEATURE_ENABLE_IF(depthBounds);
	VK_DEVICEFEATURE_ENABLE_IF(wideLines);
	VK_DEVICEFEATURE_ENABLE_IF(largePoints);
	VK_DEVICEFEATURE_ENABLE_IF(alphaToOne);
	VK_DEVICEFEATURE_ENABLE_IF(multiViewport);
	VK_DEVICEFEATURE_ENABLE_IF(samplerAnisotropy);
	VK_DEVICEFEATURE_ENABLE_IF(textureCompressionETC2);
	VK_DEVICEFEATURE_ENABLE_IF(textureCompressionASTC_LDR);
	VK_DEVICEFEATURE_ENABLE_IF(textureCompressionBC);
	//VK_DEVICEFEATURE_ENABLE_IF(occlusionQueryPrecise);
	//VK_DEVICEFEATURE_ENABLE_IF(pipelineStatisticsQuery);
	VK_DEVICEFEATURE_ENABLE_IF(vertexPipelineStoresAndAtomics);
	VK_DEVICEFEATURE_ENABLE_IF(fragmentStoresAndAtomics);
	VK_DEVICEFEATURE_ENABLE_IF(shaderTessellationAndGeometryPointSize);
	VK_DEVICEFEATURE_ENABLE_IF(shaderImageGatherExtended);
	VK_DEVICEFEATURE_ENABLE_IF(shaderStorageImageExtendedFormats);
	// Intel Arc doesn't support shaderStorageImageMultisample (yet? could be a driver thing), so it's
	// better for Validation to scream at us if we use it. Furthermore MSAA Storage is a huge red flag
	// for performance.
	//VK_DEVICEFEATURE_ENABLE_IF(shaderStorageImageMultisample);
	VK_DEVICEFEATURE_ENABLE_IF(shaderStorageImageReadWithoutFormat);
	VK_DEVICEFEATURE_ENABLE_IF(shaderStorageImageWriteWithoutFormat);
	VK_DEVICEFEATURE_ENABLE_IF(shaderUniformBufferArrayDynamicIndexing);
	VK_DEVICEFEATURE_ENABLE_IF(shaderSampledImageArrayDynamicIndexing);
	VK_DEVICEFEATURE_ENABLE_IF(shaderStorageBufferArrayDynamicIndexing);
	VK_DEVICEFEATURE_ENABLE_IF(shaderStorageImageArrayDynamicIndexing);
	VK_DEVICEFEATURE_ENABLE_IF(shaderClipDistance);
	VK_DEVICEFEATURE_ENABLE_IF(shaderCullDistance);
	VK_DEVICEFEATURE_ENABLE_IF(shaderFloat64);
	VK_DEVICEFEATURE_ENABLE_IF(shaderInt64);
	VK_DEVICEFEATURE_ENABLE_IF(shaderInt16);
	//VK_DEVICEFEATURE_ENABLE_IF(shaderResourceResidency);
	VK_DEVICEFEATURE_ENABLE_IF(shaderResourceMinLod);
	// We don't use sparse features and enabling them cause extra internal
	// allocations inside the Vulkan driver we don't need.
	//VK_DEVICEFEATURE_ENABLE_IF(sparseBinding);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidencyBuffer);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidencyImage2D);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidencyImage3D);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidency2Samples);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidency4Samples);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidency8Samples);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidency16Samples);
	//VK_DEVICEFEATURE_ENABLE_IF(sparseResidencyAliased);
	VK_DEVICEFEATURE_ENABLE_IF(variableMultisampleRate);
	//VK_DEVICEFEATURE_ENABLE_IF(inheritedQueries);

#define GET_INSTANCE_PROC_ADDR(inst, entrypoint)                                            \
	{                                                                                       \
		fp##entrypoint = (PFN_vk##entrypoint)vkGetInstanceProcAddr(inst, "vk" #entrypoint); \
		ERR_FAIL_NULL_V_MSG(fp##entrypoint, ERR_CANT_CREATE,                                \
				"vkGetInstanceProcAddr failed to find vk" #entrypoint);                     \
	}

	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceSupportKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceCapabilitiesKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceFormatsKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfacePresentModesKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetSwapchainImagesKHR);

	// Gets capability info for current Vulkan driver.
	{
		Error res = _check_capabilities();
		if (res != OK) {
			return res;
		}
	}

	device_initialized = true;
	return OK;
}

Error VulkanContext::_create_device() {
	VkResult err;
	float queue_priorities[1] = { 0.0 };
	VkDeviceQueueCreateInfo queues[2];
	queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queues[0].pNext = nullptr;
	queues[0].queueFamilyIndex = graphics_queue_family_index;
	queues[0].queueCount = 1;
	queues[0].pQueuePriorities = queue_priorities;
	queues[0].flags = 0;

	// Before we retrieved what is supported, here we tell Vulkan we want to enable these features using the same structs.
	void *nextptr = nullptr;

	VkPhysicalDeviceShaderFloat16Int8FeaturesKHR shader_features = {
		/*sType*/ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
		/*pNext*/ nextptr,
		/*shaderFloat16*/ shader_capabilities.shader_float16_is_supported,
		/*shaderInt8*/ shader_capabilities.shader_int8_is_supported,
	};
	nextptr = &shader_features;

	VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrs_features = {};
	if (vrs_capabilities.pipeline_vrs_supported || vrs_capabilities.primitive_vrs_supported || vrs_capabilities.attachment_vrs_supported) {
		// Insert into our chain to enable these features if they are available.
		vrs_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
		vrs_features.pNext = nextptr;
		vrs_features.pipelineFragmentShadingRate = vrs_capabilities.pipeline_vrs_supported;
		vrs_features.primitiveFragmentShadingRate = vrs_capabilities.primitive_vrs_supported;
		vrs_features.attachmentFragmentShadingRate = vrs_capabilities.attachment_vrs_supported;

		nextptr = &vrs_features;
	}

	VkPhysicalDevicePipelineCreationCacheControlFeatures pipeline_cache_control_features = {};
	if (pipeline_cache_control_support) {
		pipeline_cache_control_features.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES;
		pipeline_cache_control_features.pNext = nextptr;
		pipeline_cache_control_features.pipelineCreationCacheControl = pipeline_cache_control_support;

		nextptr = &pipeline_cache_control_features;
	}

	VkPhysicalDeviceVulkan11Features vulkan11features = {};
	VkPhysicalDevice16BitStorageFeaturesKHR storage_feature = {};
	VkPhysicalDeviceMultiviewFeatures multiview_features = {};
	if (device_api_version >= VK_API_VERSION_1_2) {
		// In Vulkan 1.2 and newer we use a newer struct to enable various features.

		vulkan11features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		vulkan11features.pNext = nextptr;
		vulkan11features.storageBuffer16BitAccess = storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported;
		vulkan11features.uniformAndStorageBuffer16BitAccess = storage_buffer_capabilities.uniform_and_storage_buffer_16_bit_access_is_supported;
		vulkan11features.storagePushConstant16 = storage_buffer_capabilities.storage_push_constant_16_is_supported;
		vulkan11features.storageInputOutput16 = storage_buffer_capabilities.storage_input_output_16;
		vulkan11features.multiview = multiview_capabilities.is_supported;
		vulkan11features.multiviewGeometryShader = multiview_capabilities.geometry_shader_is_supported;
		vulkan11features.multiviewTessellationShader = multiview_capabilities.tessellation_shader_is_supported;
		vulkan11features.variablePointersStorageBuffer = 0;
		vulkan11features.variablePointers = 0;
		vulkan11features.protectedMemory = 0;
		vulkan11features.samplerYcbcrConversion = 0;
		vulkan11features.shaderDrawParameters = 0;
		nextptr = &vulkan11features;
	} else {
		// On Vulkan 1.0 and 1.1 we use our older structs to initialize these features.
		storage_feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
		storage_feature.pNext = nextptr;
		storage_feature.storageBuffer16BitAccess = storage_buffer_capabilities.storage_buffer_16_bit_access_is_supported;
		storage_feature.uniformAndStorageBuffer16BitAccess = storage_buffer_capabilities.uniform_and_storage_buffer_16_bit_access_is_supported;
		storage_feature.storagePushConstant16 = storage_buffer_capabilities.storage_push_constant_16_is_supported;
		storage_feature.storageInputOutput16 = storage_buffer_capabilities.storage_input_output_16;
		nextptr = &storage_feature;

		if (device_api_version >= VK_API_VERSION_1_1) { // any Vulkan 1.1.x version
			multiview_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
			multiview_features.pNext = nextptr;
			multiview_features.multiview = multiview_capabilities.is_supported;
			multiview_features.multiviewGeometryShader = multiview_capabilities.geometry_shader_is_supported;
			multiview_features.multiviewTessellationShader = multiview_capabilities.tessellation_shader_is_supported;
			nextptr = &multiview_features;
		}
	}

	uint32_t enabled_extension_count = 0;
	const char *enabled_extension_names[MAX_EXTENSIONS];
	ERR_FAIL_COND_V(enabled_device_extension_names.size() > MAX_EXTENSIONS, ERR_CANT_CREATE);
	for (const CharString &extension_name : enabled_device_extension_names) {
		enabled_extension_names[enabled_extension_count++] = extension_name.ptr();
	}

	VkDeviceCreateInfo sdevice = {
		/*sType*/ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		/*pNext*/ nextptr,
		/*flags*/ 0,
		/*queueCreateInfoCount*/ 1,
		/*pQueueCreateInfos*/ queues,
		/*enabledLayerCount*/ 0,
		/*ppEnabledLayerNames*/ nullptr,
		/*enabledExtensionCount*/ enabled_extension_count,
		/*ppEnabledExtensionNames*/ (const char *const *)enabled_extension_names,
		/*pEnabledFeatures*/ &physical_device_features, // If specific features are required, pass them in here.
	};
	if (separate_present_queue) {
		queues[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queues[1].pNext = nullptr;
		queues[1].queueFamilyIndex = present_queue_family_index;
		queues[1].queueCount = 1;
		queues[1].pQueuePriorities = queue_priorities;
		queues[1].flags = 0;
		sdevice.queueCreateInfoCount = 2;
	}

	if (vulkan_hooks) {
		if (!vulkan_hooks->create_vulkan_device(&sdevice, &device)) {
			return ERR_CANT_CREATE;
		}
	} else {
		err = vkCreateDevice(gpu, &sdevice, nullptr, &device);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	}

	return OK;
}

Error VulkanContext::_initialize_queues(VkSurfaceKHR p_surface) {
	// Iterate over each queue to learn whether it supports presenting:
	VkBool32 *supportsPresent = (VkBool32 *)malloc(queue_family_count * sizeof(VkBool32));
	for (uint32_t i = 0; i < queue_family_count; i++) {
		fpGetPhysicalDeviceSurfaceSupportKHR(gpu, i, p_surface, &supportsPresent[i]);
	}

	// Search for a graphics and a present queue in the array of queue
	// families, try to find one that supports both.
	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	uint32_t presentQueueFamilyIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queue_family_count; i++) {
		if ((queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
			if (graphicsQueueFamilyIndex == UINT32_MAX) {
				graphicsQueueFamilyIndex = i;
			}

			if (supportsPresent[i] == VK_TRUE) {
				graphicsQueueFamilyIndex = i;
				presentQueueFamilyIndex = i;
				break;
			}
		}
	}

	if (presentQueueFamilyIndex == UINT32_MAX) {
		// If didn't find a queue that supports both graphics and present, then
		// find a separate present queue.
		for (uint32_t i = 0; i < queue_family_count; ++i) {
			if (supportsPresent[i] == VK_TRUE) {
				presentQueueFamilyIndex = i;
				break;
			}
		}
	}

	free(supportsPresent);

	// Generate error if could not find both a graphics and a present queue.
	ERR_FAIL_COND_V_MSG(graphicsQueueFamilyIndex == UINT32_MAX || presentQueueFamilyIndex == UINT32_MAX, ERR_CANT_CREATE,
			"Could not find both graphics and present queues\n");

	graphics_queue_family_index = graphicsQueueFamilyIndex;
	present_queue_family_index = presentQueueFamilyIndex;
	separate_present_queue = (graphics_queue_family_index != present_queue_family_index);

	_create_device();

	static PFN_vkGetDeviceProcAddr g_gdpa = nullptr;
#define GET_DEVICE_PROC_ADDR(dev, entrypoint)                                                     \
	{                                                                                             \
		if (!g_gdpa)                                                                              \
			g_gdpa = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr"); \
		fp##entrypoint = (PFN_vk##entrypoint)g_gdpa(dev, "vk" #entrypoint);                       \
		ERR_FAIL_NULL_V_MSG(fp##entrypoint, ERR_CANT_CREATE,                                      \
				"vkGetDeviceProcAddr failed to find vk" #entrypoint);                             \
	}

	GET_DEVICE_PROC_ADDR(device, CreateSwapchainKHR);
	GET_DEVICE_PROC_ADDR(device, DestroySwapchainKHR);
	GET_DEVICE_PROC_ADDR(device, GetSwapchainImagesKHR);
	GET_DEVICE_PROC_ADDR(device, AcquireNextImageKHR);
	GET_DEVICE_PROC_ADDR(device, QueuePresentKHR);
	if (is_device_extension_enabled(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME)) {
		GET_DEVICE_PROC_ADDR(device, GetRefreshCycleDurationGOOGLE);
		GET_DEVICE_PROC_ADDR(device, GetPastPresentationTimingGOOGLE);
	}

	vkGetDeviceQueue(device, graphics_queue_family_index, 0, &graphics_queue);

	if (!separate_present_queue) {
		present_queue = graphics_queue;
	} else {
		vkGetDeviceQueue(device, present_queue_family_index, 0, &present_queue);
	}

	// Get the list of VkFormat's that are supported:
	uint32_t formatCount;
	VkResult err = fpGetPhysicalDeviceSurfaceFormatsKHR(gpu, p_surface, &formatCount, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	VkSurfaceFormatKHR *surfFormats = (VkSurfaceFormatKHR *)malloc(formatCount * sizeof(VkSurfaceFormatKHR));
	err = fpGetPhysicalDeviceSurfaceFormatsKHR(gpu, p_surface, &formatCount, surfFormats);
	if (err) {
		free(surfFormats);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}
	// If the format list includes just one entry of VK_FORMAT_UNDEFINED,
	// the surface has no preferred format.  Otherwise, at least one
	// supported format will be returned.
	if (formatCount == 1 && surfFormats[0].format == VK_FORMAT_UNDEFINED) {
		format = VK_FORMAT_B8G8R8A8_UNORM;
		color_space = surfFormats[0].colorSpace;
	} else {
		// These should be ordered with the ones we want to use on top and fallback modes further down
		// we want a 32bit RGBA unsigned normalized buffer or similar.
		const VkFormat allowed_formats[] = {
			VK_FORMAT_B8G8R8A8_UNORM,
			VK_FORMAT_R8G8B8A8_UNORM
		};
		uint32_t allowed_formats_count = sizeof(allowed_formats) / sizeof(VkFormat);

		if (formatCount < 1) {
			free(surfFormats);
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "formatCount less than 1");
		}

		// Find the first format that we support.
		format = VK_FORMAT_UNDEFINED;
		for (uint32_t af = 0; af < allowed_formats_count && format == VK_FORMAT_UNDEFINED; af++) {
			for (uint32_t sf = 0; sf < formatCount && format == VK_FORMAT_UNDEFINED; sf++) {
				if (surfFormats[sf].format == allowed_formats[af]) {
					format = surfFormats[sf].format;
					color_space = surfFormats[sf].colorSpace;
				}
			}
		}

		if (format == VK_FORMAT_UNDEFINED) {
			free(surfFormats);
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "No usable surface format found.");
		}
	}

	free(surfFormats);

	Error serr = _create_semaphores();
	if (serr) {
		return serr;
	}

	queues_initialized = true;
	return OK;
}

Error VulkanContext::_create_semaphores() {
	VkResult err;

	// Create semaphores to synchronize acquiring presentable buffers before
	// rendering and waiting for drawing to be complete before presenting.
	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		/*sType*/ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ 0,
	};

	// Create fences that we can use to throttle if we get too far
	// ahead of the image presents.
	VkFenceCreateInfo fence_ci = {
		/*sType*/ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ VK_FENCE_CREATE_SIGNALED_BIT
	};
	for (uint32_t i = 0; i < FRAME_LAG; i++) {
		err = vkCreateFence(device, &fence_ci, nullptr, &fences[i]);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &draw_complete_semaphores[i]);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		if (separate_present_queue) {
			err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &image_ownership_semaphores[i]);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
		}
	}
	frame_index = 0;

	// Get Memory information and properties.
	vkGetPhysicalDeviceMemoryProperties(gpu, &memory_properties);

	return OK;
}

bool VulkanContext::_use_validation_layers() {
	return Engine::get_singleton()->is_validation_layers_enabled();
}

VkExtent2D VulkanContext::_compute_swapchain_extent(const VkSurfaceCapabilitiesKHR &p_surf_capabilities, int *p_window_width, int *p_window_height) const {
	// Width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
	if (p_surf_capabilities.currentExtent.width == 0xFFFFFFFF) {
		// If the surface size is undefined, the size is set to the size
		// of the images requested, which must fit within the minimum and
		// maximum values.
		VkExtent2D extent = {};
		extent.width = CLAMP((uint32_t)(*p_window_width), p_surf_capabilities.minImageExtent.width, p_surf_capabilities.maxImageExtent.width);
		extent.height = CLAMP((uint32_t)(*p_window_height), p_surf_capabilities.minImageExtent.height, p_surf_capabilities.maxImageExtent.height);
		return extent;
	} else {
		// If the surface size is defined, the swap chain size must match.
		*p_window_width = p_surf_capabilities.currentExtent.width;
		*p_window_height = p_surf_capabilities.currentExtent.height;
		return p_surf_capabilities.currentExtent;
	}
}

Error VulkanContext::_window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, VkSurfaceKHR p_surface, int p_width, int p_height) {
	ERR_FAIL_COND_V(windows.has(p_window_id), ERR_INVALID_PARAMETER);

	if (!device_initialized) {
		Error err = _create_physical_device(p_surface);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	if (!queues_initialized) {
		// We use a single GPU, but we need a surface to initialize the
		// queues, so this process must be deferred until a surface
		// is created.
		Error err = _initialize_queues(p_surface);
		ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);
	}

	Window window;
	window.surface = p_surface;
	window.width = p_width;
	window.height = p_height;
	window.vsync_mode = p_vsync_mode;
	Error err = _update_swap_chain(&window);
	ERR_FAIL_COND_V(err != OK, ERR_CANT_CREATE);

	windows[p_window_id] = window;
	return OK;
}

void VulkanContext::window_resize(DisplayServer::WindowID p_window, int p_width, int p_height) {
	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].width = p_width;
	windows[p_window].height = p_height;
	_update_swap_chain(&windows[p_window]);
}

int VulkanContext::window_get_width(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	return windows[p_window].width;
}

int VulkanContext::window_get_height(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	return windows[p_window].height;
}

bool VulkanContext::window_is_valid_swapchain(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), false);
	Window *w = &windows[p_window];
	return w->swapchain_image_resources != VK_NULL_HANDLE;
}

VkRenderPass VulkanContext::window_get_render_pass(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), VK_NULL_HANDLE);
	Window *w = &windows[p_window];
	// Vulkan use of currentbuffer.
	return w->render_pass;
}

VkFramebuffer VulkanContext::window_get_framebuffer(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), VK_NULL_HANDLE);
	ERR_FAIL_COND_V(!buffers_prepared, VK_NULL_HANDLE);
	Window *w = &windows[p_window];
	// Vulkan use of currentbuffer.
	if (w->swapchain_image_resources != VK_NULL_HANDLE) {
		return w->swapchain_image_resources[w->current_buffer].framebuffer;
	} else {
		return VK_NULL_HANDLE;
	}
}

void VulkanContext::window_destroy(DisplayServer::WindowID p_window_id) {
	ERR_FAIL_COND(!windows.has(p_window_id));
	_clean_up_swap_chain(&windows[p_window_id]);

	vkDestroySurfaceKHR(inst, windows[p_window_id].surface, nullptr);
	windows.erase(p_window_id);
}

Error VulkanContext::_clean_up_swap_chain(Window *window) {
	if (!window->swapchain) {
		return OK;
	}
	vkDeviceWaitIdle(device);

	// This destroys images associated it seems.
	fpDestroySwapchainKHR(device, window->swapchain, nullptr);
	window->swapchain = VK_NULL_HANDLE;
	vkDestroyRenderPass(device, window->render_pass, nullptr);
	window->render_pass = VK_NULL_HANDLE;
	if (window->swapchain_image_resources) {
		for (uint32_t i = 0; i < swapchainImageCount; i++) {
			vkDestroyImageView(device, window->swapchain_image_resources[i].view, nullptr);
			vkDestroyFramebuffer(device, window->swapchain_image_resources[i].framebuffer, nullptr);
		}

		free(window->swapchain_image_resources);
		window->swapchain_image_resources = nullptr;
		swapchainImageCount = 0;
	}
	if (separate_present_queue) {
		vkDestroyCommandPool(device, window->present_cmd_pool, nullptr);
	}

	for (uint32_t i = 0; i < FRAME_LAG; i++) {
		// Destroy the semaphores now (we'll re-create it later if we have to).
		// We must do this because the semaphore cannot be reused if it's in a signaled state
		// (which happens if vkAcquireNextImageKHR returned VK_ERROR_OUT_OF_DATE_KHR or VK_SUBOPTIMAL_KHR)
		// The only way to reset it would be to present the swapchain... the one we just destroyed.
		// And the API has no way to "unsignal" the semaphore.
		vkDestroySemaphore(device, window->image_acquired_semaphores[i], nullptr);
		window->image_acquired_semaphores[i] = 0;
	}

	return OK;
}

Error VulkanContext::_update_swap_chain(Window *window) {
	VkResult err;

	if (window->swapchain) {
		_clean_up_swap_chain(window);
	}

	// Check the surface capabilities and formats.
	VkSurfaceCapabilitiesKHR surfCapabilities;
	err = fpGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, window->surface, &surfCapabilities);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	{
		VkBool32 supports = VK_FALSE;
		err = vkGetPhysicalDeviceSurfaceSupportKHR(
				gpu, present_queue_family_index, window->surface, &supports);
		ERR_FAIL_COND_V_MSG(err != VK_SUCCESS || supports == false, ERR_CANT_CREATE,
				"Window's surface is not supported by device. Did the GPU go offline? Was the window "
				"created on another monitor? Check previous errors & try launching with "
				"--gpu-validation.");
	}

	uint32_t presentModeCount;
	err = fpGetPhysicalDeviceSurfacePresentModesKHR(gpu, window->surface, &presentModeCount, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	VkPresentModeKHR *presentModes = (VkPresentModeKHR *)malloc(presentModeCount * sizeof(VkPresentModeKHR));
	ERR_FAIL_NULL_V(presentModes, ERR_CANT_CREATE);
	err = fpGetPhysicalDeviceSurfacePresentModesKHR(gpu, window->surface, &presentModeCount, presentModes);
	if (err) {
		free(presentModes);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	VkExtent2D swapchainExtent = _compute_swapchain_extent(surfCapabilities, &window->width, &window->height);

	if (window->width == 0 || window->height == 0) {
		free(presentModes);
		// Likely window minimized, no swapchain created.
		return ERR_SKIP;
	}
	// The FIFO present mode is guaranteed by the spec to be supported
	// and to have no tearing.  It's a great default present mode to use.

	// There are times when you may wish to use another present mode.  The
	// following code shows how to select them, and the comments provide some
	// reasons you may wish to use them.
	//
	// It should be noted that Vulkan 1.0 doesn't provide a method for
	// synchronizing rendering with the presentation engine's display. There
	// is a method provided for throttling rendering with the display, but
	// there are some presentation engines for which this method will not work.
	// If an application doesn't throttle its rendering, and if it renders much
	// faster than the refresh rate of the display, this can waste power on
	// mobile devices. That is because power is being spent rendering images
	// that may never be seen.

	// VK_PRESENT_MODE_IMMEDIATE_KHR is for applications that don't care about
	// tearing, or have some way of synchronizing their rendering with the
	// display.
	// VK_PRESENT_MODE_MAILBOX_KHR may be useful for applications that
	// generally render a new presentable image every refresh cycle, but are
	// occasionally early.  In this case, the application wants the new image
	// to be displayed instead of the previously-queued-for-presentation image
	// that has not yet been displayed.
	// VK_PRESENT_MODE_FIFO_RELAXED_KHR is for applications that generally
	// render a new presentable image every refresh cycle, but are occasionally
	// late.  In this case (perhaps because of stuttering/latency concerns),
	// the application wants the late image to be immediately displayed, even
	// though that may mean some tearing.

	VkPresentModeKHR requested_present_mode = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;
	switch (window->vsync_mode) {
		case DisplayServer::VSYNC_MAILBOX:
			requested_present_mode = VkPresentModeKHR::VK_PRESENT_MODE_MAILBOX_KHR;
			break;
		case DisplayServer::VSYNC_ADAPTIVE:
			requested_present_mode = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_RELAXED_KHR;
			break;
		case DisplayServer::VSYNC_ENABLED:
			requested_present_mode = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR;
			break;
		case DisplayServer::VSYNC_DISABLED:
			requested_present_mode = VkPresentModeKHR::VK_PRESENT_MODE_IMMEDIATE_KHR;
			break;
	}

	// Check if the requested mode is available.
	bool present_mode_available = false;
	for (uint32_t i = 0; i < presentModeCount; i++) {
		if (presentModes[i] == requested_present_mode) {
			present_mode_available = true;
		}
	}

	// Set the windows present mode if it is available, otherwise FIFO is used (guaranteed supported).
	if (present_mode_available) {
		if (window->presentMode != requested_present_mode) {
			window->presentMode = requested_present_mode;
			print_verbose("Using present mode: " + String(string_VkPresentModeKHR(window->presentMode)));
		}
	} else {
		String present_mode_string;
		switch (window->vsync_mode) {
			case DisplayServer::VSYNC_MAILBOX:
				present_mode_string = "Mailbox";
				break;
			case DisplayServer::VSYNC_ADAPTIVE:
				present_mode_string = "Adaptive";
				break;
			case DisplayServer::VSYNC_ENABLED:
				present_mode_string = "Enabled";
				break;
			case DisplayServer::VSYNC_DISABLED:
				present_mode_string = "Disabled";
				break;
		}
		WARN_PRINT(vformat("The requested V-Sync mode %s is not available. Falling back to V-Sync mode Enabled.", present_mode_string));
		window->vsync_mode = DisplayServer::VSYNC_ENABLED; // Set to default.
	}

	free(presentModes);

	// Determine the number of VkImages to use in the swap chain.
	// Application desires to acquire 3 images at a time for triple
	// buffering.
	uint32_t desiredNumOfSwapchainImages = 3;
	if (desiredNumOfSwapchainImages < surfCapabilities.minImageCount) {
		desiredNumOfSwapchainImages = surfCapabilities.minImageCount;
	}
	// If maxImageCount is 0, we can ask for as many images as we want;
	// otherwise we're limited to maxImageCount.
	if ((surfCapabilities.maxImageCount > 0) && (desiredNumOfSwapchainImages > surfCapabilities.maxImageCount)) {
		// Application must settle for fewer images than desired.
		desiredNumOfSwapchainImages = surfCapabilities.maxImageCount;
	}

	VkSurfaceTransformFlagsKHR preTransform;
	if (surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	} else {
		preTransform = surfCapabilities.currentTransform;
	}

	VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

	if (OS::get_singleton()->is_layered_allowed() || !(surfCapabilities.supportedCompositeAlpha & compositeAlpha)) {
		// Find a supported composite alpha mode - one of these is guaranteed to be set.
		VkCompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
			VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
			VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		};

		for (uint32_t i = 0; i < ARRAY_SIZE(compositeAlphaFlags); i++) {
			if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
				compositeAlpha = compositeAlphaFlags[i];
				break;
			}
		}
	}

	VkSwapchainCreateInfoKHR swapchain_ci = {
		/*sType*/ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		/*pNext*/ nullptr,
		/*flags*/ 0,
		/*surface*/ window->surface,
		/*minImageCount*/ desiredNumOfSwapchainImages,
		/*imageFormat*/ format,
		/*imageColorSpace*/ color_space,
		/*imageExtent*/ {
				/*width*/ swapchainExtent.width,
				/*height*/ swapchainExtent.height,
		},
		/*imageArrayLayers*/ 1,
		/*imageUsage*/ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		/*imageSharingMode*/ VK_SHARING_MODE_EXCLUSIVE,
		/*queueFamilyIndexCount*/ 0,
		/*pQueueFamilyIndices*/ nullptr,
		/*preTransform*/ (VkSurfaceTransformFlagBitsKHR)preTransform,
		/*compositeAlpha*/ compositeAlpha,
		/*presentMode*/ window->presentMode,
		/*clipped*/ true,
		/*oldSwapchain*/ VK_NULL_HANDLE,
	};

	err = fpCreateSwapchainKHR(device, &swapchain_ci, nullptr, &window->swapchain);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	uint32_t sp_image_count;
	err = fpGetSwapchainImagesKHR(device, window->swapchain, &sp_image_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	if (swapchainImageCount == 0) {
		// Assign here for the first time.
		swapchainImageCount = sp_image_count;
	} else {
		ERR_FAIL_COND_V(swapchainImageCount != sp_image_count, ERR_BUG);
	}

	VkImage *swapchainImages = (VkImage *)malloc(swapchainImageCount * sizeof(VkImage));
	ERR_FAIL_NULL_V(swapchainImages, ERR_CANT_CREATE);
	err = fpGetSwapchainImagesKHR(device, window->swapchain, &swapchainImageCount, swapchainImages);
	if (err) {
		free(swapchainImages);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	window->swapchain_image_resources =
			(SwapchainImageResources *)malloc(sizeof(SwapchainImageResources) * swapchainImageCount);
	if (!window->swapchain_image_resources) {
		free(swapchainImages);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	for (uint32_t i = 0; i < swapchainImageCount; i++) {
		VkImageViewCreateInfo color_image_view = {
			/*sType*/ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*image*/ swapchainImages[i],
			/*viewType*/ VK_IMAGE_VIEW_TYPE_2D,
			/*format*/ format,
			/*components*/ {
					/*r*/ VK_COMPONENT_SWIZZLE_R,
					/*g*/ VK_COMPONENT_SWIZZLE_G,
					/*b*/ VK_COMPONENT_SWIZZLE_B,
					/*a*/ VK_COMPONENT_SWIZZLE_A,
			},
			/*subresourceRange*/ { /*aspectMask*/ VK_IMAGE_ASPECT_COLOR_BIT,
					/*baseMipLevel*/ 0,
					/*levelCount*/ 1,
					/*baseArrayLayer*/ 0,
					/*layerCount*/ 1 },
		};

		window->swapchain_image_resources[i].image = swapchainImages[i];

		color_image_view.image = window->swapchain_image_resources[i].image;

		err = vkCreateImageView(device, &color_image_view, nullptr, &window->swapchain_image_resources[i].view);
		if (err) {
			free(swapchainImages);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}
	}

	free(swapchainImages);

	/******** FRAMEBUFFER ************/

	{
		const VkAttachmentDescription2KHR attachment = {
			/*sType*/ VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*format*/ format,
			/*samples*/ VK_SAMPLE_COUNT_1_BIT,
			/*loadOp*/ VK_ATTACHMENT_LOAD_OP_CLEAR,
			/*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
			/*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			/*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
			/*initialLayout*/ VK_IMAGE_LAYOUT_UNDEFINED,
			/*finalLayout*/ VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,

		};
		const VkAttachmentReference2KHR color_reference = {
			/*sType*/ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR,
			/*pNext*/ nullptr,
			/*attachment*/ 0,
			/*layout*/ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			/*aspectMask*/ 0,
		};

		const VkSubpassDescription2KHR subpass = {
			/*sType*/ VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*pipelineBindPoint*/ VK_PIPELINE_BIND_POINT_GRAPHICS,
			/*viewMask*/ 0,
			/*inputAttachmentCount*/ 0,
			/*pInputAttachments*/ nullptr,
			/*colorAttachmentCount*/ 1,
			/*pColorAttachments*/ &color_reference,
			/*pResolveAttachments*/ nullptr,
			/*pDepthStencilAttachment*/ nullptr,
			/*preserveAttachmentCount*/ 0,
			/*pPreserveAttachments*/ nullptr,
		};

		const VkRenderPassCreateInfo2KHR rp_info = {
			/*sType*/ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*attachmentCount*/ 1,
			/*pAttachments*/ &attachment,
			/*subpassCount*/ 1,
			/*pSubpasses*/ &subpass,
			/*dependencyCount*/ 0,
			/*pDependencies*/ nullptr,
			/*correlatedViewMaskCount*/ 0,
			/*pCorrelatedViewMasks*/ nullptr,
		};

		err = vkCreateRenderPass2KHR(device, &rp_info, nullptr, &window->render_pass);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		for (uint32_t i = 0; i < swapchainImageCount; i++) {
			const VkFramebufferCreateInfo fb_info = {
				/*sType*/ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				/*pNext*/ nullptr,
				/*flags*/ 0,
				/*renderPass*/ window->render_pass,
				/*attachmentCount*/ 1,
				/*pAttachments*/ &window->swapchain_image_resources[i].view,
				/*width*/ (uint32_t)window->width,
				/*height*/ (uint32_t)window->height,
				/*layers*/ 1,
			};

			err = vkCreateFramebuffer(device, &fb_info, nullptr, &window->swapchain_image_resources[i].framebuffer);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
		}
	}

	/******** SEPARATE PRESENT QUEUE ************/

	if (separate_present_queue) {
		const VkCommandPoolCreateInfo present_cmd_pool_info = {
			/*sType*/ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*queueFamilyIndex*/ present_queue_family_index,
		};
		err = vkCreateCommandPool(device, &present_cmd_pool_info, nullptr, &window->present_cmd_pool);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
		const VkCommandBufferAllocateInfo present_cmd_info = {
			/*sType*/ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			/*pNext*/ nullptr,
			/*commandPool*/ window->present_cmd_pool,
			/*level*/ VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			/*commandBufferCount*/ 1,
		};
		for (uint32_t i = 0; i < swapchainImageCount; i++) {
			err = vkAllocateCommandBuffers(device, &present_cmd_info,
					&window->swapchain_image_resources[i].graphics_to_present_cmd);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

			const VkCommandBufferBeginInfo cmd_buf_info = {
				/*sType*/ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				/*pNext*/ nullptr,
				/*flags*/ VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				/*pInheritanceInfo*/ nullptr,
			};
			err = vkBeginCommandBuffer(window->swapchain_image_resources[i].graphics_to_present_cmd, &cmd_buf_info);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

			VkImageMemoryBarrier image_ownership_barrier = {
				/*sType*/ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				/*pNext*/ nullptr,
				/*srcAccessMask*/ 0,
				/*dstAccessMask*/ VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				/*oldLayout*/ VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				/*newLayout*/ VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				/*srcQueueFamilyIndex*/ graphics_queue_family_index,
				/*dstQueueFamilyIndex*/ present_queue_family_index,
				/*image*/ window->swapchain_image_resources[i].image,
				/*subresourceRange*/ { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
			};

			vkCmdPipelineBarrier(window->swapchain_image_resources[i].graphics_to_present_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_ownership_barrier);
			err = vkEndCommandBuffer(window->swapchain_image_resources[i].graphics_to_present_cmd);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
		}
	}

	// Reset current buffer.
	window->current_buffer = 0;

	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		/*sType*/ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ 0,
	};

	for (uint32_t i = 0; i < FRAME_LAG; i++) {
		VkResult vkerr = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &window->image_acquired_semaphores[i]);
		ERR_FAIL_COND_V(vkerr, ERR_CANT_CREATE);
	}

	return OK;
}

Error VulkanContext::initialize() {
#ifdef USE_VOLK
	if (volkInitialize() != VK_SUCCESS) {
		return FAILED;
	}
#endif

	Error err = _create_instance();
	if (err != OK) {
		return err;
	}

	return OK;
}

void VulkanContext::set_setup_buffer(VkCommandBuffer p_command_buffer) {
	command_buffer_queue.write[0] = p_command_buffer;
}

void VulkanContext::append_command_buffer(VkCommandBuffer p_command_buffer) {
	if (command_buffer_queue.size() <= command_buffer_count) {
		command_buffer_queue.resize(command_buffer_count + 1);
	}

	command_buffer_queue.write[command_buffer_count] = p_command_buffer;
	command_buffer_count++;
}

void VulkanContext::flush(bool p_flush_setup, bool p_flush_pending) {
	// Ensure everything else pending is executed.
	vkDeviceWaitIdle(device);

	// Flush the pending setup buffer.

	bool setup_flushable = p_flush_setup && command_buffer_queue[0];
	bool pending_flushable = p_flush_pending && command_buffer_count > 1;

	if (setup_flushable) {
		// Use a fence to wait for everything done.
		VkSubmitInfo submit_info;
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		submit_info.pWaitDstStageMask = nullptr;
		submit_info.waitSemaphoreCount = 0;
		submit_info.pWaitSemaphores = nullptr;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = command_buffer_queue.ptr();
		submit_info.signalSemaphoreCount = pending_flushable ? 1 : 0;
		submit_info.pSignalSemaphores = pending_flushable ? &draw_complete_semaphores[frame_index] : nullptr;
		VkResult err = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		command_buffer_queue.write[0] = nullptr;
		ERR_FAIL_COND(err);
	}

	if (pending_flushable) {
		// Use a fence to wait for everything to finish.

		VkSubmitInfo submit_info;
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		VkPipelineStageFlags wait_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		submit_info.pWaitDstStageMask = setup_flushable ? &wait_stage_mask : nullptr;
		submit_info.waitSemaphoreCount = setup_flushable ? 1 : 0;
		submit_info.pWaitSemaphores = setup_flushable ? &draw_complete_semaphores[frame_index] : nullptr;
		submit_info.commandBufferCount = command_buffer_count - 1;
		submit_info.pCommandBuffers = command_buffer_queue.ptr() + 1;
		submit_info.signalSemaphoreCount = 0;
		submit_info.pSignalSemaphores = nullptr;
		VkResult err = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		command_buffer_count = 1;
		ERR_FAIL_COND(err);
	}

	vkDeviceWaitIdle(device);
}

Error VulkanContext::prepare_buffers() {
	if (!queues_initialized) {
		return OK;
	}

	VkResult err;

	// Ensure no more than FRAME_LAG renderings are outstanding.
	vkWaitForFences(device, 1, &fences[frame_index], VK_TRUE, UINT64_MAX);
	vkResetFences(device, 1, &fences[frame_index]);

	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;

		w->semaphore_acquired = false;

		if (w->swapchain == VK_NULL_HANDLE) {
			continue;
		}

		do {
			// Get the index of the next available swapchain image.
			err =
					fpAcquireNextImageKHR(device, w->swapchain, UINT64_MAX,
							w->image_acquired_semaphores[frame_index], VK_NULL_HANDLE, &w->current_buffer);

			if (err == VK_ERROR_OUT_OF_DATE_KHR) {
				// Swapchain is out of date (e.g. the window was resized) and
				// must be recreated.
				print_verbose("Vulkan: Early out of date swapchain, recreating.");
				// resize_notify();
				_update_swap_chain(w);
			} else if (err == VK_SUBOPTIMAL_KHR) {
				// Swapchain is not as optimal as it could be, but the platform's
				// presentation engine will still present the image correctly.
				print_verbose("Vulkan: Early suboptimal swapchain, recreating.");
				Error swap_chain_err = _update_swap_chain(w);
				if (swap_chain_err == ERR_SKIP) {
					break;
				}
			} else if (err != VK_SUCCESS) {
				ERR_BREAK_MSG(err != VK_SUCCESS, "Vulkan: Did not create swapchain successfully. Error code: " + String(string_VkResult(err)));
			} else {
				w->semaphore_acquired = true;
			}
		} while (err != VK_SUCCESS);
	}

	buffers_prepared = true;

	return OK;
}

Error VulkanContext::swap_buffers() {
	if (!queues_initialized) {
		return OK;
	}

	//	print_line("swapbuffers?");
	VkResult err;

#if 0
	if (is_device_extension_enabled(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME)) {
		// Look at what happened to previous presents, and make appropriate
		// adjustments in timing.
		DemoUpdateTargetIPD(demo);

		// Note: a real application would position its geometry to that it's in
		// the correct location for when the next image is presented.  It might
		// also wait, so that there's less latency between any input and when
		// the next image is rendered/presented.  This demo program is so
		// simple that it doesn't do either of those.
	}
#endif
	// Wait for the image acquired semaphore to be signaled to ensure
	// that the image won't be rendered to until the presentation
	// engine has fully released ownership to the application, and it is
	// okay to render to the image.

	const VkCommandBuffer *commands_ptr = nullptr;
	uint32_t commands_to_submit = 0;

	if (command_buffer_queue[0] == nullptr) {
		// No setup command, but commands to submit, submit from the first and skip command.
		if (command_buffer_count > 1) {
			commands_ptr = command_buffer_queue.ptr() + 1;
			commands_to_submit = command_buffer_count - 1;
		}
	} else {
		commands_ptr = command_buffer_queue.ptr();
		commands_to_submit = command_buffer_count;
	}

	VkSemaphore *semaphores_to_acquire = (VkSemaphore *)alloca(windows.size() * sizeof(VkSemaphore));
	VkPipelineStageFlags *pipe_stage_flags = (VkPipelineStageFlags *)alloca(windows.size() * sizeof(VkPipelineStageFlags));
	uint32_t semaphores_to_acquire_count = 0;

	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;

		if (w->semaphore_acquired) {
			semaphores_to_acquire[semaphores_to_acquire_count] = w->image_acquired_semaphores[frame_index];
			pipe_stage_flags[semaphores_to_acquire_count] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			semaphores_to_acquire_count++;
		}
	}

	VkSubmitInfo submit_info;
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.pNext = nullptr;
	submit_info.waitSemaphoreCount = semaphores_to_acquire_count;
	submit_info.pWaitSemaphores = semaphores_to_acquire;
	submit_info.pWaitDstStageMask = pipe_stage_flags;
	submit_info.commandBufferCount = commands_to_submit;
	submit_info.pCommandBuffers = commands_ptr;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &draw_complete_semaphores[frame_index];
	err = vkQueueSubmit(graphics_queue, 1, &submit_info, fences[frame_index]);
	ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Vulkan: Cannot submit graphics queue. Error code: " + String(string_VkResult(err)));

	command_buffer_queue.write[0] = nullptr;
	command_buffer_count = 1;

	if (separate_present_queue) {
		// If we are using separate queues, change image ownership to the
		// present queue before presenting, waiting for the draw complete
		// semaphore and signaling the ownership released semaphore when finished.
		VkFence nullFence = VK_NULL_HANDLE;
		pipe_stage_flags[0] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = &draw_complete_semaphores[frame_index];
		submit_info.commandBufferCount = 0;

		VkCommandBuffer *cmdbufptr = (VkCommandBuffer *)alloca(sizeof(VkCommandBuffer *) * windows.size());
		submit_info.pCommandBuffers = cmdbufptr;

		for (KeyValue<int, Window> &E : windows) {
			Window *w = &E.value;

			if (w->swapchain == VK_NULL_HANDLE) {
				continue;
			}
			cmdbufptr[submit_info.commandBufferCount] = w->swapchain_image_resources[w->current_buffer].graphics_to_present_cmd;
			submit_info.commandBufferCount++;
		}

		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &image_ownership_semaphores[frame_index];
		err = vkQueueSubmit(present_queue, 1, &submit_info, nullFence);
		ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Vulkan: Cannot submit present queue. Error code: " + String(string_VkResult(err)));
	}

	// If we are using separate queues, we have to wait for image ownership,
	// otherwise wait for draw complete.
	VkPresentInfoKHR present = {
		/*sType*/ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		/*pNext*/ nullptr,
		/*waitSemaphoreCount*/ 1,
		/*pWaitSemaphores*/ (separate_present_queue) ? &image_ownership_semaphores[frame_index] : &draw_complete_semaphores[frame_index],
		/*swapchainCount*/ 0,
		/*pSwapchain*/ nullptr,
		/*pImageIndices*/ nullptr,
		/*pResults*/ nullptr,
	};

	VkSwapchainKHR *pSwapchains = (VkSwapchainKHR *)alloca(sizeof(VkSwapchainKHR *) * windows.size());
	uint32_t *pImageIndices = (uint32_t *)alloca(sizeof(uint32_t *) * windows.size());

	present.pSwapchains = pSwapchains;
	present.pImageIndices = pImageIndices;

	for (KeyValue<int, Window> &E : windows) {
		Window *w = &E.value;

		if (w->swapchain == VK_NULL_HANDLE) {
			continue;
		}
		pSwapchains[present.swapchainCount] = w->swapchain;
		pImageIndices[present.swapchainCount] = w->current_buffer;
		present.swapchainCount++;
	}

#if 0
	if (is_device_extension_enabled(VK_KHR_incremental_present_enabled)) {
		// If using VK_KHR_incremental_present, we provide a hint of the region
		// that contains changed content relative to the previously-presented
		// image.  The implementation can use this hint in order to save
		// work/power (by only copying the region in the hint).  The
		// implementation is free to ignore the hint though, and so we must
		// ensure that the entire image has the correctly-drawn content.
		uint32_t eighthOfWidth = width / 8;
		uint32_t eighthOfHeight = height / 8;
		VkRectLayerKHR rect = {
			/*offset.x*/ eighthOfWidth,
			/*offset.y*/ eighthOfHeight,
			/*extent.width*/ eighthOfWidth * 6,
			/*extent.height*/ eighthOfHeight * 6,
			/*layer*/ 0,
		};
		VkPresentRegionKHR region = {
			/*rectangleCount*/ 1,
			/*pRectangles*/ &rect,
		};
		VkPresentRegionsKHR regions = {
			/*sType*/ VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
			/*pNext*/ present.pNext,
			/*swapchainCount*/ present.swapchainCount,
			/*pRegions*/ &region,
		};
		present.pNext = &regions;
	}
#endif

#if 0
	if (is_device_extension_enabled(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME)) {
		VkPresentTimeGOOGLE ptime;
		if (prev_desired_present_time == 0) {
			// This must be the first present for this swapchain.
			//
			// We don't know where we are relative to the presentation engine's
			// display's refresh cycle.  We also don't know how long rendering
			// takes.  Let's make a grossly-simplified assumption that the
			// desiredPresentTime should be half way between now and
			// now+target_IPD.  We will adjust over time.
			uint64_t curtime = getTimeInNanoseconds();
			if (curtime == 0) {
				// Since we didn't find out the current time, don't give a
				// desiredPresentTime.
				ptime.desiredPresentTime = 0;
			} else {
				ptime.desiredPresentTime = curtime + (target_IPD >> 1);
			}
		} else {
			ptime.desiredPresentTime = (prev_desired_present_time + target_IPD);
		}
		ptime.presentID = next_present_id++;
		prev_desired_present_time = ptime.desiredPresentTime;

		VkPresentTimesInfoGOOGLE present_time = {
			/*sType*/ VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE,
			/*pNext*/ present.pNext,
			/*swapchainCount*/ present.swapchainCount,
			/*pTimes*/ &ptime,
		};
		if (is_device_extension_enabled(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME)) {
			present.pNext = &present_time;
		}
	}
#endif
	//	print_line("current buffer:  " + itos(current_buffer));
	err = fpQueuePresentKHR(present_queue, &present);

	frame_index += 1;
	frame_index %= FRAME_LAG;

	if (err == VK_ERROR_OUT_OF_DATE_KHR) {
		// Swapchain is out of date (e.g. the window was resized) and
		// must be recreated.
		print_verbose("Vulkan queue submit: Swapchain is out of date, recreating.");
		resize_notify();
	} else if (err == VK_SUBOPTIMAL_KHR) {
		// Swapchain is not as optimal as it could be, but the platform's
		// presentation engine will still present the image correctly.
		print_verbose("Vulkan queue submit: Swapchain is suboptimal.");
	} else {
		ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Error code: " + String(string_VkResult(err)));
	}

	buffers_prepared = false;
	return OK;
}

void VulkanContext::resize_notify() {
}

VkDevice VulkanContext::get_device() {
	return device;
}

VkPhysicalDevice VulkanContext::get_physical_device() {
	return gpu;
}

int VulkanContext::get_swapchain_image_count() const {
	return swapchainImageCount;
}

VkQueue VulkanContext::get_graphics_queue() const {
	return graphics_queue;
}

uint32_t VulkanContext::get_graphics_queue_family_index() const {
	return graphics_queue_family_index;
}

VkFormat VulkanContext::get_screen_format() const {
	return format;
}

VkPhysicalDeviceLimits VulkanContext::get_device_limits() const {
	return gpu_props.limits;
}

RID VulkanContext::local_device_create() {
	LocalDevice ld;

	{ // Create device.
		VkResult err;
		float queue_priorities[1] = { 0.0 };
		VkDeviceQueueCreateInfo queues[2];
		queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queues[0].pNext = nullptr;
		queues[0].queueFamilyIndex = graphics_queue_family_index;
		queues[0].queueCount = 1;
		queues[0].pQueuePriorities = queue_priorities;
		queues[0].flags = 0;

		uint32_t enabled_extension_count = 0;
		const char *enabled_extension_names[MAX_EXTENSIONS];
		ERR_FAIL_COND_V(enabled_device_extension_names.size() > MAX_EXTENSIONS, RID());
		for (const CharString &extension_name : enabled_device_extension_names) {
			enabled_extension_names[enabled_extension_count++] = extension_name.ptr();
		}

		VkDeviceCreateInfo sdevice = {
			/*sType =*/VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			/*pNext */ nullptr,
			/*flags */ 0,
			/*queueCreateInfoCount */ 1,
			/*pQueueCreateInfos */ queues,
			/*enabledLayerCount */ 0,
			/*ppEnabledLayerNames */ nullptr,
			/*enabledExtensionCount */ enabled_extension_count,
			/*ppEnabledExtensionNames */ (const char *const *)enabled_extension_names,
			/*pEnabledFeatures */ &physical_device_features, // If specific features are required, pass them in here.
		};
		err = vkCreateDevice(gpu, &sdevice, nullptr, &ld.device);
		ERR_FAIL_COND_V(err, RID());
	}

	{ // Create graphics queue.

		vkGetDeviceQueue(ld.device, graphics_queue_family_index, 0, &ld.queue);
	}

	return local_device_owner.make_rid(ld);
}

VkDevice VulkanContext::local_device_get_vk_device(RID p_local_device) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	return ld->device;
}

void VulkanContext::local_device_push_command_buffers(RID p_local_device, const VkCommandBuffer *p_buffers, int p_count) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	ERR_FAIL_COND(ld->waiting);

	VkSubmitInfo submit_info;
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.pNext = nullptr;
	submit_info.pWaitDstStageMask = nullptr;
	submit_info.waitSemaphoreCount = 0;
	submit_info.pWaitSemaphores = nullptr;
	submit_info.commandBufferCount = p_count;
	submit_info.pCommandBuffers = p_buffers;
	submit_info.signalSemaphoreCount = 0;
	submit_info.pSignalSemaphores = nullptr;

	VkResult err = vkQueueSubmit(ld->queue, 1, &submit_info, VK_NULL_HANDLE);
	if (err == VK_ERROR_OUT_OF_HOST_MEMORY) {
		print_line("Vulkan: Out of host memory!");
	}
	if (err == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
		print_line("Vulkan: Out of device memory!");
	}
	if (err == VK_ERROR_DEVICE_LOST) {
		print_line("Vulkan: Device lost!");
	}
	ERR_FAIL_COND(err);

	ld->waiting = true;
}

void VulkanContext::local_device_sync(RID p_local_device) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	ERR_FAIL_COND(!ld->waiting);

	vkDeviceWaitIdle(ld->device);
	ld->waiting = false;
}

void VulkanContext::local_device_free(RID p_local_device) {
	LocalDevice *ld = local_device_owner.get_or_null(p_local_device);
	vkDestroyDevice(ld->device, nullptr);
	local_device_owner.free(p_local_device);
}

void VulkanContext::command_begin_label(VkCommandBuffer p_command_buffer, String p_label_name, const Color p_color) {
	if (!is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		return;
	}

	CharString cs = p_label_name.utf8();
	VkDebugUtilsLabelEXT label;
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pNext = nullptr;
	label.pLabelName = cs.get_data();
	label.color[0] = p_color[0];
	label.color[1] = p_color[1];
	label.color[2] = p_color[2];
	label.color[3] = p_color[3];
	CmdBeginDebugUtilsLabelEXT(p_command_buffer, &label);
}

void VulkanContext::command_insert_label(VkCommandBuffer p_command_buffer, String p_label_name, const Color p_color) {
	if (!is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		return;
	}
	CharString cs = p_label_name.utf8();
	VkDebugUtilsLabelEXT label;
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pNext = nullptr;
	label.pLabelName = cs.get_data();
	label.color[0] = p_color[0];
	label.color[1] = p_color[1];
	label.color[2] = p_color[2];
	label.color[3] = p_color[3];
	CmdInsertDebugUtilsLabelEXT(p_command_buffer, &label);
}

void VulkanContext::command_end_label(VkCommandBuffer p_command_buffer) {
	if (!is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		return;
	}
	CmdEndDebugUtilsLabelEXT(p_command_buffer);
}

void VulkanContext::set_object_name(VkObjectType p_object_type, uint64_t p_object_handle, String p_object_name) {
	if (!is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		return;
	}
	CharString obj_data = p_object_name.utf8();
	VkDebugUtilsObjectNameInfoEXT name_info;
	name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	name_info.pNext = nullptr;
	name_info.objectType = p_object_type;
	name_info.objectHandle = p_object_handle;
	name_info.pObjectName = obj_data.get_data();
	SetDebugUtilsObjectNameEXT(device, &name_info);
}

String VulkanContext::get_device_vendor_name() const {
	return device_vendor;
}

String VulkanContext::get_device_name() const {
	return device_name;
}

RenderingDevice::DeviceType VulkanContext::get_device_type() const {
	return RenderingDevice::DeviceType(device_type);
}

String VulkanContext::get_device_api_version() const {
	return vformat("%d.%d.%d", VK_API_VERSION_MAJOR(device_api_version), VK_API_VERSION_MINOR(device_api_version), VK_API_VERSION_PATCH(device_api_version));
}

String VulkanContext::get_device_pipeline_cache_uuid() const {
	return pipeline_cache_id;
}

DisplayServer::VSyncMode VulkanContext::get_vsync_mode(DisplayServer::WindowID p_window) const {
	ERR_FAIL_COND_V_MSG(!windows.has(p_window), DisplayServer::VSYNC_ENABLED, "Could not get V-Sync mode for window with WindowID " + itos(p_window) + " because it does not exist.");
	return windows[p_window].vsync_mode;
}

void VulkanContext::set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_mode) {
	ERR_FAIL_COND_MSG(!windows.has(p_window), "Could not set V-Sync mode for window with WindowID " + itos(p_window) + " because it does not exist.");
	windows[p_window].vsync_mode = p_mode;
	_update_swap_chain(&windows[p_window]);
}

VulkanContext::VulkanContext() {
	command_buffer_queue.resize(1); // First one is always the setup command.
	command_buffer_queue.write[0] = nullptr;
}

VulkanContext::~VulkanContext() {
	if (queue_props) {
		free(queue_props);
	}
	if (device_initialized) {
		for (uint32_t i = 0; i < FRAME_LAG; i++) {
			vkDestroyFence(device, fences[i], nullptr);
			vkDestroySemaphore(device, draw_complete_semaphores[i], nullptr);
			if (separate_present_queue) {
				vkDestroySemaphore(device, image_ownership_semaphores[i], nullptr);
			}
		}
		if (inst_initialized && is_instance_extension_enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
			DestroyDebugUtilsMessengerEXT(inst, dbg_messenger, nullptr);
		}
		if (inst_initialized && dbg_debug_report != VK_NULL_HANDLE) {
			DestroyDebugReportCallbackEXT(inst, dbg_debug_report, nullptr);
		}
		vkDestroyDevice(device, nullptr);
	}
	if (inst_initialized) {
		vkDestroyInstance(inst, nullptr);
	}
}
