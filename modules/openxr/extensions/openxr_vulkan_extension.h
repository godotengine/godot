/*************************************************************************/
/*  openxr_vulkan_extension.h                                            */
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

#ifndef OPENXR_VULKAN_EXTENSION_H
#define OPENXR_VULKAN_EXTENSION_H

#include "core/templates/vector.h"
#include "openxr_extension_wrapper.h"

#include "drivers/vulkan/vulkan_context.h"

// Forward declare these so we don't need OpenXR headers where-ever this is included
// Including OpenXR at this point gives loads and loads of compile issues especially
// on Windows because windows.h is EVIL and really shouldn't be included outside of platform
// but we really don't have a choice in the matter

struct XrGraphicsRequirementsVulkanKHR;
struct XrVulkanInstanceCreateInfoKHR;
struct XrVulkanGraphicsDeviceGetInfoKHR;
struct XrVulkanDeviceCreateInfoKHR;
struct XrGraphicsBindingVulkanKHR;

class OpenXRVulkanExtension : public OpenXRGraphicsExtensionWrapper, VulkanHooks {
public:
	OpenXRVulkanExtension(OpenXRAPI *p_openxr_api);
	virtual ~OpenXRVulkanExtension() override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void *set_session_create_and_get_next_pointer(void *p_next_pointer) override;

	virtual bool create_vulkan_instance(const VkInstanceCreateInfo *p_vulkan_create_info, VkInstance *r_instance) override;
	virtual bool get_physical_device(VkPhysicalDevice *r_device) override;
	virtual bool create_vulkan_device(const VkDeviceCreateInfo *p_device_create_info, VkDevice *r_device) override;

	virtual void get_usable_swapchain_formats(Vector<int64_t> &p_usable_swap_chains) override;
	virtual String get_swapchain_format_name(int64_t p_swapchain_format) const override;
	virtual bool get_swapchain_image_data(XrSwapchain p_swapchain, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size, void **r_swapchain_graphics_data) override;
	virtual void cleanup_swapchain_graphics_data(void **p_swapchain_graphics_data) override;
	virtual bool create_projection_fov(const XrFovf p_fov, double p_z_near, double p_z_far, CameraMatrix &r_camera_matrix) override;
	virtual bool copy_render_target_to_image(RID p_from_render_target, void *p_swapchain_graphics_data, int p_image_index) override;

private:
	static OpenXRVulkanExtension *singleton;
	static XrGraphicsBindingVulkanKHR graphics_binding_vulkan; // declaring this as static so we don't need to know its size and we only need it once when creating our session

	struct SwapchainGraphicsData {
		bool is_multiview;
		Vector<RID> image_rids;
		Vector<RID> framebuffers;
	};

	bool check_graphics_api_support(XrVersion p_desired_version);

	VkInstance vulkan_instance = nullptr;
	VkPhysicalDevice vulkan_physical_device = nullptr;
	VkDevice vulkan_device = nullptr;
	uint32_t vulkan_queue_family_index = 0;
	uint32_t vulkan_queue_index = 0;

	XrResult xrGetVulkanGraphicsRequirements2KHR(XrInstance p_instance, XrSystemId p_system_id, XrGraphicsRequirementsVulkanKHR *p_graphics_requirements);
	XrResult xrCreateVulkanInstanceKHR(XrInstance p_instance, const XrVulkanInstanceCreateInfoKHR *p_create_info, VkInstance *r_vulkan_instance, VkResult *r_vulkan_result);
	XrResult xrGetVulkanGraphicsDevice2KHR(XrInstance p_instance, const XrVulkanGraphicsDeviceGetInfoKHR *p_get_info, VkPhysicalDevice *r_vulkan_physical_device);
	XrResult xrCreateVulkanDeviceKHR(XrInstance p_instance, const XrVulkanDeviceCreateInfoKHR *p_create_info, VkDevice *r_device, VkResult *r_result);
};

#endif // !OPENXR_VULKAN_EXTENSION_H
