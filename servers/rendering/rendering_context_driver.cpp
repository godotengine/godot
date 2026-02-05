/**************************************************************************/
/*  rendering_context_driver.cpp                                          */
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

#include "rendering_context_driver.h"

#include "core/config/project_settings.h"

RenderingContextDriver::~RenderingContextDriver() {
}

RenderingContextDriver::SurfaceID RenderingContextDriver::surface_get_from_window(DisplayServer::WindowID p_window) const {
	HashMap<DisplayServer::WindowID, SurfaceID>::ConstIterator it = window_surface_map.find(p_window);
	if (it != window_surface_map.end()) {
		return it->value;
	} else {
		return SurfaceID();
	}
}

Error RenderingContextDriver::window_create(DisplayServer::WindowID p_window, const void *p_platform_data) {
	SurfaceID surface = surface_create(p_platform_data);
	if (surface != 0) {
		window_surface_map[p_window] = surface;
		return OK;
	} else {
		return ERR_CANT_CREATE;
	}
}

void RenderingContextDriver::window_set_size(DisplayServer::WindowID p_window, uint32_t p_width, uint32_t p_height) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_set_size(surface, p_width, p_height);
	}
}

void RenderingContextDriver::window_set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_vsync_mode) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_set_vsync_mode(surface, p_vsync_mode);
	}
}

DisplayServer::VSyncMode RenderingContextDriver::window_get_vsync_mode(DisplayServer::WindowID p_window) const {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		return surface_get_vsync_mode(surface);
	} else {
		return DisplayServer::VSYNC_DISABLED;
	}
}

void RenderingContextDriver::window_destroy(DisplayServer::WindowID p_window) {
	SurfaceID surface = surface_get_from_window(p_window);
	if (surface) {
		surface_destroy(surface);
	}

	window_surface_map.erase(p_window);
}

int32_t RenderingContextDriver::pick_device(SurfaceID p_surface, bool p_print_verbose) {
	if (p_print_verbose) {
		print_verbose("Devices:");
	}

	int32_t device_index = Engine::get_singleton()->get_gpu_index();
	const uint32_t device_count = device_get_count();
	const bool detect_device = (device_index < 0) || (device_index >= int32_t(device_count));
	uint32_t device_type_score = 0;
	for (uint32_t i = 0; i < device_count; i++) {
		RenderingContextDriver::Device device_option = device_get(i);
		String name = device_option.name;
		String vendor = get_device_vendor_name(device_option);
		String type = get_device_type_name(device_option);
		bool present_supported = p_surface != 0 ? device_supports_present(i, p_surface) : false;
		if (p_print_verbose) {
			print_verbose("  #" + itos(i) + ": " + vendor + " " + name + " - " + (present_supported ? "Supported" : "Unsupported") + ", " + type);
		}

		if (detect_device && (present_supported || p_surface == 0)) {
			// If a window was specified, present must be supported by the device to be available as an option.
			// Assign a score for each type of device and prefer the device with the higher score.
			uint32_t option_score = get_device_type_score(device_option);
			if (option_score > device_type_score) {
				device_index = i;
				device_type_score = option_score;
			}
		}
	}

	return device_index;
}

Error RenderingContextDriver::_check_excluded_devices() {
	// Check if the picked device from the context is excluded from using RenderingDevice. The names must be an exact string match.
	String device_list_string = GLOBAL_GET("rendering/rendering_device/excluded_device_list");
	PackedStringArray device_list = device_list_string.split(",");
	int32_t device_index = pick_device(0, false);
	if (device_index >= 0 && device_index < (int32_t)(device_get_count())) {
		const RenderingContextDriver::Device &device = device_get(device_index);
		return device_list.has(device.name) ? ERR_CANT_CREATE : OK;
	} else {
		// No valid device was found, so just fail on this step instead.
		return ERR_CANT_CREATE;
	}
}

String RenderingContextDriver::get_driver_and_device_memory_report() const {
	String report;

	const uint32_t num_tracked_obj_types = static_cast<uint32_t>(get_tracked_object_type_count());

	report += "=== Driver Memory Report ===";

	report += "\nLaunch with --extra-gpu-memory-tracking and build with "
			  "DEBUG_ENABLED for this functionality to work.";
	report += "\nDevice memory may be unavailable if the API does not support it"
			  "(e.g. VK_EXT_device_memory_report is unsupported).";
	report += "\n";

	report += "\nTotal Driver Memory:";
	report += String::num_real(double(get_driver_total_memory()) / (1024.0 * 1024.0));
	report += " MB";
	report += "\nTotal Driver Num Allocations: ";
	report += String::num_uint64(get_driver_allocation_count());

	report += "\nTotal Device Memory:";
	report += String::num_real(double(get_device_total_memory()) / (1024.0 * 1024.0));
	report += " MB";
	report += "\nTotal Device Num Allocations: ";
	report += String::num_uint64(get_device_allocation_count());

	report += "\n\nMemory use by object type (CSV format):";
	report += "\n\nCategory; Driver memory in MB; Driver Allocation Count; "
			  "Device memory in MB; Device Allocation Count";

	for (uint32_t i = 0u; i < num_tracked_obj_types; ++i) {
		report += "\n";
		report += get_tracked_object_name(i);
		report += ";";
		report += String::num_real(double(get_driver_memory_by_object_type(i)) / (1024.0 * 1024.0));
		report += ";";
		report += String::num_uint64(get_driver_allocs_by_object_type(i));
		report += ";";
		report += String::num_real(double(get_device_memory_by_object_type(i)) / (1024.0 * 1024.0));
		report += ";";
		report += String::num_uint64(get_device_allocs_by_object_type(i));
	}

	return report;
}

const char *RenderingContextDriver::get_tracked_object_name(uint32_t p_type_index) const {
	return "Tracking Unsupported by API";
}

uint64_t RenderingContextDriver::get_tracked_object_type_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_total_memory() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_allocation_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_memory_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_driver_allocs_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_total_memory() const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_allocation_count() const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_memory_by_object_type(uint32_t) const {
	return 0;
}

uint64_t RenderingContextDriver::get_device_allocs_by_object_type(uint32_t) const {
	return 0;
}

String RenderingContextDriver::get_device_vendor_name(const Device &p_device) {
	switch (p_device.vendor) {
		case RenderingContextDriver::Vendor::VENDOR_AMD:
			return "AMD";
		case RenderingContextDriver::Vendor::VENDOR_IMGTEC:
			return "ImgTec";
		case RenderingContextDriver::Vendor::VENDOR_APPLE:
			return "Apple";
		case RenderingContextDriver::Vendor::VENDOR_NVIDIA:
			return "NVIDIA";
		case RenderingContextDriver::Vendor::VENDOR_ARM:
			return "ARM";
		case RenderingContextDriver::Vendor::VENDOR_MICROSOFT:
			return "Microsoft";
		case RenderingContextDriver::Vendor::VENDOR_QUALCOMM:
			return "Qualcomm";
		case RenderingContextDriver::Vendor::VENDOR_INTEL:
			return "Intel";
		default:
			return "Unknown";
	}
}

String RenderingContextDriver::get_device_type_name(const Device &p_device) {
	switch (p_device.type) {
		case RenderingContextDriver::DEVICE_TYPE_INTEGRATED_GPU:
			return "Integrated";
		case RenderingContextDriver::DEVICE_TYPE_DISCRETE_GPU:
			return "Discrete";
		case RenderingContextDriver::DEVICE_TYPE_VIRTUAL_GPU:
			return "Virtual";
		case RenderingContextDriver::DEVICE_TYPE_CPU:
			return "CPU";
		case RenderingContextDriver::DEVICE_TYPE_OTHER:
		default:
			return "Other";
	}
}

uint32_t RenderingContextDriver::get_device_type_score(const Device &p_device) {
	static const bool prefer_integrated = OS::get_singleton()->get_user_prefers_integrated_gpu();
	switch (p_device.type) {
		case RenderingContextDriver::DEVICE_TYPE_INTEGRATED_GPU:
			return prefer_integrated ? 5 : 4;
		case RenderingContextDriver::DEVICE_TYPE_DISCRETE_GPU:
			return prefer_integrated ? 4 : 5;
		case RenderingContextDriver::DEVICE_TYPE_VIRTUAL_GPU:
			return 3;
		case RenderingContextDriver::DEVICE_TYPE_CPU:
			return 2;
		case RenderingContextDriver::DEVICE_TYPE_OTHER:
		default:
			return 1;
	}
}
