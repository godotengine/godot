/*************************************************************************/
/*  vulkan_context.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "vulkan_context.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/string/ustring.h"
#include "core/version.h"
#include "servers/rendering/rendering_device.h"

#include "vk_enum_string_helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
#define APP_SHORT_NAME "GodotEngine"

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
	/*
	// This is a valid warning because its illegal in Vulkan, but in practice it should work according to VK_KHR_maintenance2
	if (strstr(pCallbackData->pMessage, "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 with tiling VK_IMAGE_TILING_OPTIMAL does not support usage that includes VK_IMAGE_USAGE_STORAGE_BIT") != nullptr) {
		return VK_FALSE;
	}

	if (strstr(pCallbackData->pMessage, "VK_FORMAT_R4G4B4A4_UNORM_PACK16 with tiling VK_IMAGE_TILING_OPTIMAL does not support usage that includes VK_IMAGE_USAGE_STORAGE_BIT") != nullptr) {
		return VK_FALSE;
	}
*/
	// Workaround for Vulkan-Loader usability bug: https://github.com/KhronosGroup/Vulkan-Loader/issues/262.
	if (strstr(pCallbackData->pMessage, "wrong ELF class: ELFCLASS32") != nullptr) {
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

VkBool32 VulkanContext::_check_layers(uint32_t check_count, const char **check_names, uint32_t layer_count, VkLayerProperties *layers) {
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

Error VulkanContext::_create_validation_layers() {
	VkResult err;
	const char *instance_validation_layers_alt1[] = { "VK_LAYER_KHRONOS_validation" };
	const char *instance_validation_layers_alt2[] = { "VK_LAYER_LUNARG_standard_validation" };
	const char *instance_validation_layers_alt3[] = { "VK_LAYER_GOOGLE_threading", "VK_LAYER_LUNARG_parameter_validation", "VK_LAYER_LUNARG_object_tracker", "VK_LAYER_LUNARG_core_validation", "VK_LAYER_GOOGLE_unique_objects" };

	uint32_t instance_layer_count = 0;
	err = vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	VkBool32 validation_found = 0;
	uint32_t validation_layer_count = 0;
	const char **instance_validation_layers = nullptr;
	if (instance_layer_count > 0) {
		VkLayerProperties *instance_layers = (VkLayerProperties *)malloc(sizeof(VkLayerProperties) * instance_layer_count);
		err = vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layers);
		if (err) {
			free(instance_layers);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}

		validation_layer_count = ARRAY_SIZE(instance_validation_layers_alt1);
		instance_validation_layers = instance_validation_layers_alt1;
		validation_found = _check_layers(validation_layer_count, instance_validation_layers, instance_layer_count, instance_layers);

		// use alternative (deprecated, removed in SDK 1.1.126.0) set of validation layers
		if (!validation_found) {
			validation_layer_count = ARRAY_SIZE(instance_validation_layers_alt2);
			instance_validation_layers = instance_validation_layers_alt2;
			validation_found = _check_layers(validation_layer_count, instance_validation_layers, instance_layer_count, instance_layers);
		}

		// use alternative (deprecated, removed in SDK 1.1.121.1) set of validation layers
		if (!validation_found) {
			validation_layer_count = ARRAY_SIZE(instance_validation_layers_alt3);
			instance_validation_layers = instance_validation_layers_alt3;
			validation_found = _check_layers(validation_layer_count, instance_validation_layers, instance_layer_count, instance_layers);
		}

		free(instance_layers);
	}

	if (validation_found) {
		enabled_layer_count = validation_layer_count;
		for (uint32_t i = 0; i < validation_layer_count; i++) {
			enabled_layers[i] = instance_validation_layers[i];
		}
	} else {
		return ERR_CANT_CREATE;
	}

	return OK;
}

typedef VkResult(VKAPI_PTR *_vkEnumerateInstanceVersion)(uint32_t *);

Error VulkanContext::_obtain_vulkan_version() {
	// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkApplicationInfo.html#_description
	// for Vulkan 1.0 vkEnumerateInstanceVersion is not available, including not in the loader we compile against on Android.
	_vkEnumerateInstanceVersion func = (_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
	if (func != nullptr) {
		uint32_t api_version;
		VkResult res = func(&api_version);
		if (res == VK_SUCCESS) {
			vulkan_major = VK_VERSION_MAJOR(api_version);
			vulkan_minor = VK_VERSION_MINOR(api_version);
			uint32_t vulkan_patch = VK_VERSION_PATCH(api_version);

			print_line("Vulkan API " + itos(vulkan_major) + "." + itos(vulkan_minor) + "." + itos(vulkan_patch));
		} else {
			// according to the documentation this shouldn't fail with anything except a memory allocation error
			// in which case we're in deep trouble anyway
			ERR_FAIL_V(ERR_CANT_CREATE);
		}
	} else {
		print_line("vkEnumerateInstanceVersion not available, assuming Vulkan 1.0");
	}

	// we don't go above 1.2
	if ((vulkan_major > 1) || (vulkan_major == 1 && vulkan_minor > 2)) {
		vulkan_major = 1;
		vulkan_minor = 2;
	}

	return OK;
}

Error VulkanContext::_initialize_extensions() {
	uint32_t instance_extension_count = 0;

	enabled_extension_count = 0;
	enabled_layer_count = 0;
	enabled_debug_utils = false;
	enabled_debug_report = false;
	/* Look for instance extensions */
	VkBool32 surfaceExtFound = 0;
	VkBool32 platformSurfaceExtFound = 0;
	memset(extension_names, 0, sizeof(extension_names));

	VkResult err = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
	ERR_FAIL_COND_V(err != VK_SUCCESS && err != VK_INCOMPLETE, ERR_CANT_CREATE);

	if (instance_extension_count > 0) {
		VkExtensionProperties *instance_extensions = (VkExtensionProperties *)malloc(sizeof(VkExtensionProperties) * instance_extension_count);
		err = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, instance_extensions);
		if (err != VK_SUCCESS && err != VK_INCOMPLETE) {
			free(instance_extensions);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}
		for (uint32_t i = 0; i < instance_extension_count; i++) {
			if (!strcmp(VK_KHR_SURFACE_EXTENSION_NAME, instance_extensions[i].extensionName)) {
				surfaceExtFound = 1;
				extension_names[enabled_extension_count++] = VK_KHR_SURFACE_EXTENSION_NAME;
			}

			if (!strcmp(_get_platform_surface_extension(), instance_extensions[i].extensionName)) {
				platformSurfaceExtFound = 1;
				extension_names[enabled_extension_count++] = _get_platform_surface_extension();
			}
			if (!strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, instance_extensions[i].extensionName)) {
				if (use_validation_layers) {
					extension_names[enabled_extension_count++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
					enabled_debug_report = true;
				}
			}
			if (!strcmp(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, instance_extensions[i].extensionName)) {
				extension_names[enabled_extension_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
				enabled_debug_utils = true;
			}
			if (enabled_extension_count >= MAX_EXTENSIONS) {
				free(instance_extensions);
				ERR_FAIL_V_MSG(ERR_BUG, "Enabled extension count reaches MAX_EXTENSIONS, BUG");
			}
		}

		free(instance_extensions);
	}

	ERR_FAIL_COND_V_MSG(!surfaceExtFound, ERR_CANT_CREATE, "No surface extension found, is a driver installed?");
	ERR_FAIL_COND_V_MSG(!platformSurfaceExtFound, ERR_CANT_CREATE, "No platform surface extension found, is a driver installed?");

	return OK;
}

typedef void(VKAPI_PTR *_vkGetPhysicalDeviceProperties2)(VkPhysicalDevice, VkPhysicalDeviceProperties2 *);

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

	/* these are not defined on Android GRMBL */
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

	return res.substr(2); // remove first ", "
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

	return res.substr(2); // remove first ", "
}

Error VulkanContext::_check_capabilities() {
	// check subgroups
	// https://www.khronos.org/blog/vulkan-subgroup-tutorial
	// for Vulkan 1.0 vkGetPhysicalDeviceProperties2 is not available, including not in the loader we compile against on Android.
	_vkGetPhysicalDeviceProperties2 func = (_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2");
	if (func != nullptr) {
		VkPhysicalDeviceSubgroupProperties subgroupProperties;
		subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
		subgroupProperties.pNext = NULL;

		VkPhysicalDeviceProperties2 physicalDeviceProperties;
		physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physicalDeviceProperties.pNext = &subgroupProperties;

		func(gpu, &physicalDeviceProperties);

		subgroup_capabilities.size = subgroupProperties.subgroupSize;
		subgroup_capabilities.supportedStages = subgroupProperties.supportedStages;
		subgroup_capabilities.supportedOperations = subgroupProperties.supportedOperations;
		// Note: quadOperationsInAllStages will be true if:
		// - supportedStages has VK_SHADER_STAGE_ALL_GRAPHICS + VK_SHADER_STAGE_COMPUTE_BIT
		// - supportedOperations has VK_SUBGROUP_FEATURE_QUAD_BIT
		subgroup_capabilities.quadOperationsInAllStages = subgroupProperties.quadOperationsInAllStages;

		// only output this when debugging?
		print_line("- Vulkan subgroup size " + itos(subgroup_capabilities.size));
		print_line("- Vulkan subgroup stages " + subgroup_capabilities.supported_stages_desc());
		print_line("- Vulkan subgroup supported ops " + subgroup_capabilities.supported_operations_desc());
		if (subgroup_capabilities.quadOperationsInAllStages) {
			print_line("- Vulkan subgroup quad operations in all stages");
		}
	} else {
		subgroup_capabilities.size = 0;
		subgroup_capabilities.supportedStages = 0;
		subgroup_capabilities.supportedOperations = 0;
		subgroup_capabilities.quadOperationsInAllStages = false;
	}

	return OK;
}

Error VulkanContext::_create_physical_device() {
	/* obtain version */
	_obtain_vulkan_version();

	/* Look for validation layers */
	if (use_validation_layers) {
		_create_validation_layers();
	}

	/* initialise extensions */
	{
		Error err = _initialize_extensions();
		if (err != OK) {
			return err;
		}
	}

	CharString cs = ProjectSettings::get_singleton()->get("application/config/name").operator String().utf8();
	String name = "GodotEngine " + String(VERSION_FULL_NAME);
	CharString namecs = name.utf8();
	const VkApplicationInfo app = {
		/*sType*/ VK_STRUCTURE_TYPE_APPLICATION_INFO,
		/*pNext*/ nullptr,
		/*pApplicationName*/ cs.get_data(),
		/*applicationVersion*/ 0,
		/*pEngineName*/ namecs.get_data(),
		/*engineVersion*/ 0,
		/*apiVersion*/ VK_MAKE_VERSION(vulkan_major, vulkan_minor, 0)
	};
	VkInstanceCreateInfo inst_info = {
		/*sType*/ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ 0,
		/*pApplicationInfo*/ &app,
		/*enabledLayerCount*/ enabled_layer_count,
		/*ppEnabledLayerNames*/ (const char *const *)enabled_layers,
		/*enabledExtensionCount*/ enabled_extension_count,
		/*ppEnabledExtensionNames*/ (const char *const *)extension_names,
	};

	/*
	   * This is info for a temp callback to use during CreateInstance.
	   * After the instance is created, we use the instance-based
	   * function to register the final callback.
	   */
	VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info;
	VkDebugReportCallbackCreateInfoEXT dbg_report_callback_create_info{};
	if (enabled_debug_utils) {
		// VK_EXT_debug_utils style
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
	} else if (enabled_debug_report) {
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

	uint32_t gpu_count;

	VkResult err = vkCreateInstance(&inst_info, nullptr, &inst);
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

	inst_initialized = true;

	/* Make initial call to query gpu_count, then second call for gpu info*/
	err = vkEnumeratePhysicalDevices(inst, &gpu_count, nullptr);
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
	/* for now, just grab the first physical device */
	uint32_t device_index = 0;
	gpu = physical_devices[device_index];
	free(physical_devices);

	/* Look for device extensions */
	uint32_t device_extension_count = 0;
	VkBool32 swapchainExtFound = 0;
	enabled_extension_count = 0;
	memset(extension_names, 0, sizeof(extension_names));

	/* Get identifier properties */
	vkGetPhysicalDeviceProperties(gpu, &gpu_props);

	static const struct {
		uint32_t id;
		const char *name;
	} vendor_names[] = {
		{ 0x1002, "AMD" },
		{ 0x1010, "ImgTec" },
		{ 0x10DE, "NVIDIA" },
		{ 0x13B5, "ARM" },
		{ 0x5143, "Qualcomm" },
		{ 0x8086, "INTEL" },
		{ 0, nullptr },
	};
	device_name = gpu_props.deviceName;
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
#ifdef DEBUG_ENABLED
	print_line("Using Vulkan Device #" + itos(device_index) + ": " + device_vendor + " - " + device_name);
#endif
	device_api_version = gpu_props.apiVersion;

	err = vkEnumerateDeviceExtensionProperties(gpu, nullptr, &device_extension_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	if (device_extension_count > 0) {
		VkExtensionProperties *device_extensions = (VkExtensionProperties *)malloc(sizeof(VkExtensionProperties) * device_extension_count);
		err = vkEnumerateDeviceExtensionProperties(gpu, nullptr, &device_extension_count, device_extensions);
		if (err) {
			free(device_extensions);
			ERR_FAIL_V(ERR_CANT_CREATE);
		}

		for (uint32_t i = 0; i < device_extension_count; i++) {
			if (!strcmp(VK_KHR_SWAPCHAIN_EXTENSION_NAME, device_extensions[i].extensionName)) {
				swapchainExtFound = 1;
				extension_names[enabled_extension_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
			}
			if (enabled_extension_count >= MAX_EXTENSIONS) {
				free(device_extensions);
				ERR_FAIL_V_MSG(ERR_BUG, "Enabled extension count reaches MAX_EXTENSIONS, BUG");
			}
		}

		if (VK_KHR_incremental_present_enabled) {
			// Even though the user "enabled" the extension via the command
			// line, we must make sure that it's enumerated for use with the
			// device.  Therefore, disable it here, and re-enable it again if
			// enumerated.
			VK_KHR_incremental_present_enabled = false;
			for (uint32_t i = 0; i < device_extension_count; i++) {
				if (!strcmp(VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME, device_extensions[i].extensionName)) {
					extension_names[enabled_extension_count++] = VK_KHR_INCREMENTAL_PRESENT_EXTENSION_NAME;
					VK_KHR_incremental_present_enabled = true;
				}
				if (enabled_extension_count >= MAX_EXTENSIONS) {
					free(device_extensions);
					ERR_FAIL_V_MSG(ERR_BUG, "Enabled extension count reaches MAX_EXTENSIONS, BUG");
				}
			}
		}

		if (VK_GOOGLE_display_timing_enabled) {
			// Even though the user "enabled" the extension via the command
			// line, we must make sure that it's enumerated for use with the
			// device.  Therefore, disable it here, and re-enable it again if
			// enumerated.
			VK_GOOGLE_display_timing_enabled = false;
			for (uint32_t i = 0; i < device_extension_count; i++) {
				if (!strcmp(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME, device_extensions[i].extensionName)) {
					extension_names[enabled_extension_count++] = VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME;
					VK_GOOGLE_display_timing_enabled = true;
				}
				if (enabled_extension_count >= MAX_EXTENSIONS) {
					free(device_extensions);
					ERR_FAIL_V_MSG(ERR_BUG, "Enabled extension count reaches MAX_EXTENSIONS, BUG");
				}
			}
		}

		free(device_extensions);
	}

	ERR_FAIL_COND_V_MSG(!swapchainExtFound, ERR_CANT_CREATE,
			"vkEnumerateDeviceExtensionProperties failed to find the " VK_KHR_SWAPCHAIN_EXTENSION_NAME
			" extension.\n\nDo you have a compatible Vulkan installable client driver (ICD) installed?\n"
			"vkCreateInstance Failure");

	if (enabled_debug_utils) {
		// Setup VK_EXT_debug_utils function pointers always (we use them for
		// debug labels and names).
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
	} else if (enabled_debug_report) {
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

	/* Call with NULL data to get count */
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, nullptr);
	ERR_FAIL_COND_V(queue_family_count == 0, ERR_CANT_CREATE);

	queue_props = (VkQueueFamilyProperties *)malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, queue_props);

	// Query fine-grained feature support for this device.
	//  If app has specific feature requirements it should check supported
	//  features based on this query
	vkGetPhysicalDeviceFeatures(gpu, &physical_device_features);

	physical_device_features.robustBufferAccess = false; //turn off robust buffer access, which can hamper performance on some hardware

#define GET_INSTANCE_PROC_ADDR(inst, entrypoint)                                            \
	{                                                                                       \
		fp##entrypoint = (PFN_vk##entrypoint)vkGetInstanceProcAddr(inst, "vk" #entrypoint); \
		ERR_FAIL_COND_V_MSG(fp##entrypoint == nullptr, ERR_CANT_CREATE,                     \
				"vkGetInstanceProcAddr failed to find vk" #entrypoint);                     \
	}

	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceSupportKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceCapabilitiesKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfaceFormatsKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetPhysicalDeviceSurfacePresentModesKHR);
	GET_INSTANCE_PROC_ADDR(inst, GetSwapchainImagesKHR);

	// get info about what our vulkan driver is capable off
	{
		Error res = _check_capabilities();
		if (res != OK) {
			return res;
		}
	}

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

	VkDeviceCreateInfo sdevice = {
		/*sType*/ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ 0,
		/*queueCreateInfoCount*/ 1,
		/*pQueueCreateInfos*/ queues,
		/*enabledLayerCount*/ 0,
		/*ppEnabledLayerNames*/ nullptr,
		/*enabledExtensionCount*/ enabled_extension_count,
		/*ppEnabledExtensionNames*/ (const char *const *)extension_names,
		/*pEnabledFeatures*/ &physical_device_features, // If specific features are required, pass them in here

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
	err = vkCreateDevice(gpu, &sdevice, nullptr, &device);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	return OK;
}

Error VulkanContext::_initialize_queues(VkSurfaceKHR surface) {
	// Iterate over each queue to learn whether it supports presenting:
	VkBool32 *supportsPresent = (VkBool32 *)malloc(queue_family_count * sizeof(VkBool32));
	for (uint32_t i = 0; i < queue_family_count; i++) {
		fpGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &supportsPresent[i]);
	}

	// Search for a graphics and a present queue in the array of queue
	// families, try to find one that supports both
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

	// Generate error if could not find both a graphics and a present queue
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
		ERR_FAIL_COND_V_MSG(fp##entrypoint == nullptr, ERR_CANT_CREATE,                           \
				"vkGetDeviceProcAddr failed to find vk" #entrypoint);                             \
	}

	GET_DEVICE_PROC_ADDR(device, CreateSwapchainKHR);
	GET_DEVICE_PROC_ADDR(device, DestroySwapchainKHR);
	GET_DEVICE_PROC_ADDR(device, GetSwapchainImagesKHR);
	GET_DEVICE_PROC_ADDR(device, AcquireNextImageKHR);
	GET_DEVICE_PROC_ADDR(device, QueuePresentKHR);
	if (VK_GOOGLE_display_timing_enabled) {
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
	VkResult err = fpGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &formatCount, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	VkSurfaceFormatKHR *surfFormats = (VkSurfaceFormatKHR *)malloc(formatCount * sizeof(VkSurfaceFormatKHR));
	err = fpGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &formatCount, surfFormats);
	if (err) {
		free(surfFormats);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}
	// If the format list includes just one entry of VK_FORMAT_UNDEFINED,
	// the surface has no preferred format.  Otherwise, at least one
	// supported format will be returned.
	if (formatCount == 1 && surfFormats[0].format == VK_FORMAT_UNDEFINED) {
		format = VK_FORMAT_B8G8R8A8_UNORM;
	} else {
		if (formatCount < 1) {
			free(surfFormats);
			ERR_FAIL_V_MSG(ERR_CANT_CREATE, "formatCount less than 1");
		}
		format = surfFormats[0].format;
	}
	color_space = surfFormats[0].colorSpace;

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
	// rendering and waiting for drawing to be complete before presenting
	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		/*sType*/ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ 0,
	};

	// Create fences that we can use to throttle if we get too far
	// ahead of the image presents
	VkFenceCreateInfo fence_ci = {
		/*sType*/ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		/*pNext*/ nullptr,
		/*flags*/ VK_FENCE_CREATE_SIGNALED_BIT
	};
	for (uint32_t i = 0; i < FRAME_LAG; i++) {
		err = vkCreateFence(device, &fence_ci, nullptr, &fences[i]);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &image_acquired_semaphores[i]);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &draw_complete_semaphores[i]);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

		if (separate_present_queue) {
			err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &image_ownership_semaphores[i]);
			ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
		}
	}
	frame_index = 0;

	// Get Memory information and properties
	vkGetPhysicalDeviceMemoryProperties(gpu, &memory_properties);

	return OK;
}

Error VulkanContext::_window_create(DisplayServer::WindowID p_window_id, VkSurfaceKHR p_surface, int p_width, int p_height) {
	ERR_FAIL_COND_V(windows.has(p_window_id), ERR_INVALID_PARAMETER);

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

VkRenderPass VulkanContext::window_get_render_pass(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), VK_NULL_HANDLE);
	Window *w = &windows[p_window];
	//vulkan use of currentbuffer
	return w->render_pass;
}

VkFramebuffer VulkanContext::window_get_framebuffer(DisplayServer::WindowID p_window) {
	ERR_FAIL_COND_V(!windows.has(p_window), VK_NULL_HANDLE);
	ERR_FAIL_COND_V(!buffers_prepared, VK_NULL_HANDLE);
	Window *w = &windows[p_window];
	//vulkan use of currentbuffer
	return w->swapchain_image_resources[w->current_buffer].framebuffer;
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

	//this destroys images associated it seems
	fpDestroySwapchainKHR(device, window->swapchain, nullptr);
	window->swapchain = VK_NULL_HANDLE;
	vkDestroyRenderPass(device, window->render_pass, nullptr);
	if (window->swapchain_image_resources) {
		for (uint32_t i = 0; i < swapchainImageCount; i++) {
			vkDestroyImageView(device, window->swapchain_image_resources[i].view, nullptr);
			vkDestroyFramebuffer(device, window->swapchain_image_resources[i].framebuffer, nullptr);
		}

		free(window->swapchain_image_resources);
		window->swapchain_image_resources = nullptr;
	}
	if (separate_present_queue) {
		vkDestroyCommandPool(device, window->present_cmd_pool, nullptr);
	}
	return OK;
}

Error VulkanContext::_update_swap_chain(Window *window) {
	VkResult err;

	if (window->swapchain) {
		_clean_up_swap_chain(window);
	}

	// Check the surface capabilities and formats
	VkSurfaceCapabilitiesKHR surfCapabilities;
	err = fpGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, window->surface, &surfCapabilities);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	uint32_t presentModeCount;
	err = fpGetPhysicalDeviceSurfacePresentModesKHR(gpu, window->surface, &presentModeCount, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	VkPresentModeKHR *presentModes = (VkPresentModeKHR *)malloc(presentModeCount * sizeof(VkPresentModeKHR));
	ERR_FAIL_COND_V(!presentModes, ERR_CANT_CREATE);
	err = fpGetPhysicalDeviceSurfacePresentModesKHR(gpu, window->surface, &presentModeCount, presentModes);
	if (err) {
		free(presentModes);
		ERR_FAIL_V(ERR_CANT_CREATE);
	}

	VkExtent2D swapchainExtent;
	// width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
	if (surfCapabilities.currentExtent.width == 0xFFFFFFFF) {
		// If the surface size is undefined, the size is set to the size
		// of the images requested, which must fit within the minimum and
		// maximum values.
		swapchainExtent.width = window->width;
		swapchainExtent.height = window->height;

		if (swapchainExtent.width < surfCapabilities.minImageExtent.width) {
			swapchainExtent.width = surfCapabilities.minImageExtent.width;
		} else if (swapchainExtent.width > surfCapabilities.maxImageExtent.width) {
			swapchainExtent.width = surfCapabilities.maxImageExtent.width;
		}

		if (swapchainExtent.height < surfCapabilities.minImageExtent.height) {
			swapchainExtent.height = surfCapabilities.minImageExtent.height;
		} else if (swapchainExtent.height > surfCapabilities.maxImageExtent.height) {
			swapchainExtent.height = surfCapabilities.maxImageExtent.height;
		}
	} else {
		// If the surface size is defined, the swap chain size must match
		swapchainExtent = surfCapabilities.currentExtent;
		window->width = surfCapabilities.currentExtent.width;
		window->height = surfCapabilities.currentExtent.height;
	}

	if (window->width == 0 || window->height == 0) {
		free(presentModes);
		//likely window minimized, no swapchain created
		return OK;
	}
	// The FIFO present mode is guaranteed by the spec to be supported
	// and to have no tearing.  It's a great default present mode to use.
	VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;

	//  There are times when you may wish to use another present mode.  The
	//  following code shows how to select them, and the comments provide some
	//  reasons you may wish to use them.
	//
	// It should be noted that Vulkan 1.0 doesn't provide a method for
	// synchronizing rendering with the presentation engine's display.  There
	// is a method provided for throttling rendering with the display, but
	// there are some presentation engines for which this method will not work.
	// If an application doesn't throttle its rendering, and if it renders much
	// faster than the refresh rate of the display, this can waste power on
	// mobile devices.  That is because power is being spent rendering images
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

	if (window->presentMode != swapchainPresentMode) {
		for (size_t i = 0; i < presentModeCount; ++i) {
			if (presentModes[i] == window->presentMode) {
				swapchainPresentMode = window->presentMode;
				break;
			}
		}
	}
	free(presentModes);
	ERR_FAIL_COND_V_MSG(swapchainPresentMode != window->presentMode, ERR_CANT_CREATE, "Present mode specified is not supported\n");

	// Determine the number of VkImages to use in the swap chain.
	// Application desires to acquire 3 images at a time for triple
	// buffering
	uint32_t desiredNumOfSwapchainImages = 3;
	if (desiredNumOfSwapchainImages < surfCapabilities.minImageCount) {
		desiredNumOfSwapchainImages = surfCapabilities.minImageCount;
	}
	// If maxImageCount is 0, we can ask for as many images as we want;
	// otherwise we're limited to maxImageCount
	if ((surfCapabilities.maxImageCount > 0) && (desiredNumOfSwapchainImages > surfCapabilities.maxImageCount)) {
		// Application must settle for fewer images than desired:
		desiredNumOfSwapchainImages = surfCapabilities.maxImageCount;
	}

	VkSurfaceTransformFlagsKHR preTransform;
	if (surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
		preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	} else {
		preTransform = surfCapabilities.currentTransform;
	}

	// Find a supported composite alpha mode - one of these is guaranteed to be set
	VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	VkCompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
		VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
		VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
	};
	for (uint32_t i = 0; i < ARRAY_SIZE(compositeAlphaFlags); i++) {
		if (surfCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
			compositeAlpha = compositeAlphaFlags[i];
			break;
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
		/*presentMode*/ swapchainPresentMode,
		/*clipped*/ true,
		/*oldSwapchain*/ VK_NULL_HANDLE,
	};

	err = fpCreateSwapchainKHR(device, &swapchain_ci, nullptr, &window->swapchain);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	uint32_t sp_image_count;
	err = fpGetSwapchainImagesKHR(device, window->swapchain, &sp_image_count, nullptr);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	if (swapchainImageCount == 0) {
		//assign here for the first time.
		swapchainImageCount = sp_image_count;
	} else {
		ERR_FAIL_COND_V(swapchainImageCount != sp_image_count, ERR_BUG);
	}

	VkImage *swapchainImages = (VkImage *)malloc(swapchainImageCount * sizeof(VkImage));
	ERR_FAIL_COND_V(!swapchainImages, ERR_CANT_CREATE);
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
		const VkAttachmentDescription attachment = {
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
		const VkAttachmentReference color_reference = {
			/*attachment*/ 0,
			/*layout*/ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		const VkSubpassDescription subpass = {
			/*flags*/ 0,
			/*pipelineBindPoint*/ VK_PIPELINE_BIND_POINT_GRAPHICS,
			/*inputAttachmentCount*/ 0,
			/*pInputAttachments*/ nullptr,
			/*colorAttachmentCount*/ 1,
			/*pColorAttachments*/ &color_reference,
			/*pResolveAttachments*/ nullptr,
			/*pDepthStencilAttachment*/ nullptr,
			/*preserveAttachmentCount*/ 0,
			/*pPreserveAttachments*/ nullptr,
		};
		const VkRenderPassCreateInfo rp_info = {
			/*sTyp*/ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			/*pNext*/ nullptr,
			/*flags*/ 0,
			/*attachmentCount*/ 1,
			/*pAttachments*/ &attachment,
			/*subpassCount*/ 1,
			/*pSubpasses*/ &subpass,
			/*dependencyCount*/ 0,
			/*pDependencies*/ nullptr,
		};

		err = vkCreateRenderPass(device, &rp_info, nullptr, &window->render_pass);
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

	//reset current buffer
	window->current_buffer = 0;

	return OK;
}

Error VulkanContext::initialize() {
	Error err = _create_physical_device();
	if (err) {
		return err;
	}

	device_initialized = true;
	return OK;
}

void VulkanContext::set_setup_buffer(const VkCommandBuffer &pCommandBuffer) {
	command_buffer_queue.write[0] = pCommandBuffer;
}

void VulkanContext::append_command_buffer(const VkCommandBuffer &pCommandBuffer) {
	if (command_buffer_queue.size() <= command_buffer_count) {
		command_buffer_queue.resize(command_buffer_count + 1);
	}

	command_buffer_queue.write[command_buffer_count] = pCommandBuffer;
	command_buffer_count++;
}

void VulkanContext::flush(bool p_flush_setup, bool p_flush_pending) {
	// ensure everything else pending is executed
	vkDeviceWaitIdle(device);

	//flush the pending setup buffer

	if (p_flush_setup && command_buffer_queue[0]) {
		//use a fence to wait for everything done
		VkSubmitInfo submit_info;
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		submit_info.pWaitDstStageMask = nullptr;
		submit_info.waitSemaphoreCount = 0;
		submit_info.pWaitSemaphores = nullptr;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = command_buffer_queue.ptr();
		submit_info.signalSemaphoreCount = 0;
		submit_info.pSignalSemaphores = nullptr;
		VkResult err = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		command_buffer_queue.write[0] = nullptr;
		ERR_FAIL_COND(err);
		vkDeviceWaitIdle(device);
	}

	if (p_flush_pending && command_buffer_count > 1) {
		//use a fence to wait for everything done

		VkSubmitInfo submit_info;
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.pNext = nullptr;
		submit_info.pWaitDstStageMask = nullptr;
		submit_info.waitSemaphoreCount = 0;
		submit_info.pWaitSemaphores = nullptr;
		submit_info.commandBufferCount = command_buffer_count - 1;
		submit_info.pCommandBuffers = command_buffer_queue.ptr() + 1;
		submit_info.signalSemaphoreCount = 0;
		submit_info.pSignalSemaphores = nullptr;
		VkResult err = vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		ERR_FAIL_COND(err);
		vkDeviceWaitIdle(device);

		command_buffer_count = 1;
	}
}

Error VulkanContext::prepare_buffers() {
	if (!queues_initialized) {
		return OK;
	}

	VkResult err;

	// Ensure no more than FRAME_LAG renderings are outstanding
	vkWaitForFences(device, 1, &fences[frame_index], VK_TRUE, UINT64_MAX);
	vkResetFences(device, 1, &fences[frame_index]);

	for (Map<int, Window>::Element *E = windows.front(); E; E = E->next()) {
		Window *w = &E->get();

		if (w->swapchain == VK_NULL_HANDLE) {
			continue;
		}

		do {
			// Get the index of the next available swapchain image:
			err =
					fpAcquireNextImageKHR(device, w->swapchain, UINT64_MAX,
							image_acquired_semaphores[frame_index], VK_NULL_HANDLE, &w->current_buffer);

			if (err == VK_ERROR_OUT_OF_DATE_KHR) {
				// swapchain is out of date (e.g. the window was resized) and
				// must be recreated:
				print_line("early out of data");
				//resize_notify();
				_update_swap_chain(w);
			} else if (err == VK_SUBOPTIMAL_KHR) {
				print_line("early suboptimal");
				// swapchain is not as optimal as it could be, but the platform's
				// presentation engine will still present the image correctly.
				break;
			} else {
				ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
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
	if (VK_GOOGLE_display_timing_enabled) {
		// Look at what happened to previous presents, and make appropriate
		// adjustments in timing:
		DemoUpdateTargetIPD(demo);

		// Note: a real application would position its geometry to that it's in
		// the correct location for when the next image is presented.  It might
		// also wait, so that there's less latency between any input and when
		// the next image is rendered/presented.  This demo program is so
		// simple that it doesn't do either of those.
	}
#endif
	// Wait for the image acquired semaphore to be signalled to ensure
	// that the image won't be rendered to until the presentation
	// engine has fully released ownership to the application, and it is
	// okay to render to the image.

	const VkCommandBuffer *commands_ptr = nullptr;
	uint32_t commands_to_submit = 0;

	if (command_buffer_queue[0] == nullptr) {
		//no setup command, but commands to submit, submit from the first and skip command
		if (command_buffer_count > 1) {
			commands_ptr = command_buffer_queue.ptr() + 1;
			commands_to_submit = command_buffer_count - 1;
		}
	} else {
		commands_ptr = command_buffer_queue.ptr();
		commands_to_submit = command_buffer_count;
	}

	VkPipelineStageFlags pipe_stage_flags;
	VkSubmitInfo submit_info;
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.pNext = nullptr;
	submit_info.pWaitDstStageMask = &pipe_stage_flags;
	pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &image_acquired_semaphores[frame_index];
	submit_info.commandBufferCount = commands_to_submit;
	submit_info.pCommandBuffers = commands_ptr;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &draw_complete_semaphores[frame_index];
	err = vkQueueSubmit(graphics_queue, 1, &submit_info, fences[frame_index]);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	command_buffer_queue.write[0] = nullptr;
	command_buffer_count = 1;

	if (separate_present_queue) {
		// If we are using separate queues, change image ownership to the
		// present queue before presenting, waiting for the draw complete
		// semaphore and signalling the ownership released semaphore when finished
		VkFence nullFence = VK_NULL_HANDLE;
		pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = &draw_complete_semaphores[frame_index];
		submit_info.commandBufferCount = 0;

		VkCommandBuffer *cmdbufptr = (VkCommandBuffer *)alloca(sizeof(VkCommandBuffer *) * windows.size());
		submit_info.pCommandBuffers = cmdbufptr;

		for (Map<int, Window>::Element *E = windows.front(); E; E = E->next()) {
			Window *w = &E->get();

			if (w->swapchain == VK_NULL_HANDLE) {
				continue;
			}
			cmdbufptr[submit_info.commandBufferCount] = w->swapchain_image_resources[w->current_buffer].graphics_to_present_cmd;
			submit_info.commandBufferCount++;
		}

		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &image_ownership_semaphores[frame_index];
		err = vkQueueSubmit(present_queue, 1, &submit_info, nullFence);
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	}

	// If we are using separate queues, we have to wait for image ownership,
	// otherwise wait for draw complete
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

	for (Map<int, Window>::Element *E = windows.front(); E; E = E->next()) {
		Window *w = &E->get();

		if (w->swapchain == VK_NULL_HANDLE) {
			continue;
		}
		pSwapchains[present.swapchainCount] = w->swapchain;
		pImageIndices[present.swapchainCount] = w->current_buffer;
		present.swapchainCount++;
	}

#if 0
	if (VK_KHR_incremental_present_enabled) {
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
	if (VK_GOOGLE_display_timing_enabled) {
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
				// desiredPresentTime:
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
		if (VK_GOOGLE_display_timing_enabled) {
			present.pNext = &present_time;
		}
	}
#endif
	static int total_frames = 0;
	total_frames++;
	//	print_line("current buffer:  " + itos(current_buffer));
	err = fpQueuePresentKHR(present_queue, &present);

	frame_index += 1;
	frame_index %= FRAME_LAG;

	if (err == VK_ERROR_OUT_OF_DATE_KHR) {
		// swapchain is out of date (e.g. the window was resized) and
		// must be recreated:
		print_line("out of date");
		resize_notify();
	} else if (err == VK_SUBOPTIMAL_KHR) {
		// swapchain is not as optimal as it could be, but the platform's
		// presentation engine will still present the image correctly.
		print_line("suboptimal");
	} else {
		ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
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

uint32_t VulkanContext::get_graphics_queue() const {
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

	{ //create device
		VkResult err;
		float queue_priorities[1] = { 0.0 };
		VkDeviceQueueCreateInfo queues[2];
		queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queues[0].pNext = nullptr;
		queues[0].queueFamilyIndex = graphics_queue_family_index;
		queues[0].queueCount = 1;
		queues[0].pQueuePriorities = queue_priorities;
		queues[0].flags = 0;

		VkDeviceCreateInfo sdevice = {
			/*sType =*/VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			/*pNext */ nullptr,
			/*flags */ 0,
			/*queueCreateInfoCount */ 1,
			/*pQueueCreateInfos */ queues,
			/*enabledLayerCount */ 0,
			/*ppEnabledLayerNames */ nullptr,
			/*enabledExtensionCount */ enabled_extension_count,
			/*ppEnabledExtensionNames */ (const char *const *)extension_names,
			/*pEnabledFeatures */ &physical_device_features, // If specific features are required, pass them in here
		};
		err = vkCreateDevice(gpu, &sdevice, nullptr, &ld.device);
		ERR_FAIL_COND_V(err, RID());
	}

	{ //create graphics queue

		vkGetDeviceQueue(ld.device, graphics_queue_family_index, 0, &ld.queue);
	}

	return local_device_owner.make_rid(ld);
}

VkDevice VulkanContext::local_device_get_vk_device(RID p_local_device) {
	LocalDevice *ld = local_device_owner.getornull(p_local_device);
	return ld->device;
}

void VulkanContext::local_device_push_command_buffers(RID p_local_device, const VkCommandBuffer *p_buffers, int p_count) {
	LocalDevice *ld = local_device_owner.getornull(p_local_device);
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
		print_line("out of host memory");
	}
	if (err == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
		print_line("out of device memory");
	}
	if (err == VK_ERROR_DEVICE_LOST) {
		print_line("device lost");
	}
	ERR_FAIL_COND(err);

	ld->waiting = true;
}

void VulkanContext::local_device_sync(RID p_local_device) {
	LocalDevice *ld = local_device_owner.getornull(p_local_device);
	ERR_FAIL_COND(!ld->waiting);

	vkDeviceWaitIdle(ld->device);
	ld->waiting = false;
}

void VulkanContext::local_device_free(RID p_local_device) {
	LocalDevice *ld = local_device_owner.getornull(p_local_device);
	vkDestroyDevice(ld->device, nullptr);
	local_device_owner.free(p_local_device);
}

void VulkanContext::command_begin_label(VkCommandBuffer p_command_buffer, String p_label_name, const Color p_color) {
	if (!enabled_debug_utils) {
		return;
	}

	CharString cs = p_label_name.utf8().get_data();
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
	if (!enabled_debug_utils) {
		return;
	}
	CharString cs = p_label_name.utf8().get_data();
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
	if (!enabled_debug_utils) {
		return;
	}
	CmdEndDebugUtilsLabelEXT(p_command_buffer);
}

void VulkanContext::set_object_name(VkObjectType p_object_type, uint64_t p_object_handle, String p_object_name) {
	if (!enabled_debug_utils) {
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
String VulkanContext::get_device_pipeline_cache_uuid() const {
	return pipeline_cache_id;
}

VulkanContext::VulkanContext() {
	use_validation_layers = Engine::get_singleton()->is_validation_layers_enabled();

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
			vkDestroySemaphore(device, image_acquired_semaphores[i], nullptr);
			vkDestroySemaphore(device, draw_complete_semaphores[i], nullptr);
			if (separate_present_queue) {
				vkDestroySemaphore(device, image_ownership_semaphores[i], nullptr);
			}
		}
		if (inst_initialized && enabled_debug_utils) {
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
