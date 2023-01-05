/*************************************************************************/
/*  openxr_extension_wrapper.h                                           */
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

#ifndef OPENXR_EXTENSION_WRAPPER_H
#define OPENXR_EXTENSION_WRAPPER_H

#include "core/error/error_macros.h"
#include "core/math/projection.h"
#include "core/templates/hash_map.h"
#include "core/templates/rid.h"

#include "thirdparty/openxr/src/common/xr_linear.h"
#include <openxr/openxr.h>

class OpenXRAPI;
class OpenXRActionMap;

class OpenXRExtensionWrapper {
protected:
	OpenXRAPI *openxr_api = nullptr;

	// Store extension we require.
	// If bool pointer is a nullptr this means this extension is mandatory and initialisation will fail if it is not available
	// If bool pointer is set, value will be set to true or false depending on whether extension is available
	HashMap<String, bool *> request_extensions;

public:
	virtual HashMap<String, bool *> get_request_extensions() {
		return request_extensions;
	}

	// These functions allow an extension to add entries to a struct chain.
	// `p_next_pointer` points to the last struct that was created for this chain
	// and should be used as the value for the `pNext` pointer in the first struct you add.
	// You should return the pointer to the last struct you define as your result.
	// If you are not adding any structs, just return `p_next_pointer`.
	// See existing extensions for examples of this implementation.
	virtual void *set_system_properties_and_get_next_pointer(void *p_next_pointer) { return p_next_pointer; }
	virtual void *set_session_create_and_get_next_pointer(void *p_next_pointer) { return p_next_pointer; }
	virtual void *set_swapchain_create_info_and_get_next_pointer(void *p_next_pointer) { return p_next_pointer; }
	virtual void *set_instance_create_info_and_get_next_pointer(void *p_next_pointer) { return p_next_pointer; }

	virtual void on_before_instance_created() {}
	virtual void on_instance_created(const XrInstance p_instance) {}
	virtual void on_instance_destroyed() {}
	virtual void on_session_created(const XrSession p_instance) {}
	virtual void on_process() {}
	virtual void on_pre_render() {}
	virtual void on_session_destroyed() {}

	virtual void on_state_idle() {}
	virtual void on_state_ready() {}
	virtual void on_state_synchronized() {}
	virtual void on_state_visible() {}
	virtual void on_state_focused() {}
	virtual void on_state_stopping() {}
	virtual void on_state_loss_pending() {}
	virtual void on_state_exiting() {}

	// Returns true if the event was handled, false otherwise.
	virtual bool on_event_polled(const XrEventDataBuffer &event) {
		return false;
	}

	OpenXRExtensionWrapper(OpenXRAPI *p_openxr_api) { openxr_api = p_openxr_api; };
	virtual ~OpenXRExtensionWrapper() = default;
};

class OpenXRGraphicsExtensionWrapper : public OpenXRExtensionWrapper {
public:
	virtual void get_usable_swapchain_formats(Vector<int64_t> &p_usable_swap_chains) = 0;
	virtual void get_usable_depth_formats(Vector<int64_t> &p_usable_swap_chains) = 0;
	virtual String get_swapchain_format_name(int64_t p_swapchain_format) const = 0;
	virtual bool get_swapchain_image_data(XrSwapchain p_swapchain, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size, void **r_swapchain_graphics_data) = 0;
	virtual void cleanup_swapchain_graphics_data(void **p_swapchain_graphics_data) = 0;
	virtual bool create_projection_fov(const XrFovf p_fov, double p_z_near, double p_z_far, Projection &r_camera_matrix) = 0;
	virtual RID get_texture(void *p_swapchain_graphics_data, int p_image_index) = 0;

	OpenXRGraphicsExtensionWrapper(OpenXRAPI *p_openxr_api) :
			OpenXRExtensionWrapper(p_openxr_api){};
};

#endif // OPENXR_EXTENSION_WRAPPER_H
