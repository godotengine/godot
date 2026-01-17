/**************************************************************************/
/*  openxr_d3d12_extension.cpp                                            */
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

#include "openxr_d3d12_extension.h"

#ifdef D3D12_ENABLED

#include "../../openxr_util.h"

#include "servers/rendering/rendering_server.h"
#include "servers/rendering/rendering_server_globals.h"

HashMap<String, bool *> OpenXRD3D12Extension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_KHR_D3D12_ENABLE_EXTENSION_NAME] = nullptr;

	return request_extensions;
}

void OpenXRD3D12Extension::on_instance_created(const XrInstance p_instance) {
	// Obtain pointers to functions we're accessing here.
	ERR_FAIL_NULL(OpenXRAPI::get_singleton());

	EXT_INIT_XR_FUNC(xrGetD3D12GraphicsRequirementsKHR);
	EXT_INIT_XR_FUNC(xrEnumerateSwapchainImages);
}

D3D_FEATURE_LEVEL OpenXRD3D12Extension::get_feature_level() const {
	XrGraphicsRequirementsD3D12KHR d3d12_requirements = {
		XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR, // type
		nullptr, // next
		{ 0, 0 }, // adapterLuid
		(D3D_FEATURE_LEVEL)0 // minFeatureLevel
	};

	XrResult result = xrGetD3D12GraphicsRequirementsKHR(OpenXRAPI::get_singleton()->get_instance(), OpenXRAPI::get_singleton()->get_system_id(), &d3d12_requirements);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get Direct3D 12 graphics requirements [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		return D3D_FEATURE_LEVEL_11_0; // Good default.
	}

	// #ifdef DEBUG
	print_line("OpenXR: xrGetD3D12GraphicsRequirementsKHR:");
	print_line(" - minFeatureLevel: ", (uint32_t)d3d12_requirements.minFeatureLevel);
	// #endif

	return d3d12_requirements.minFeatureLevel;
}

LUID OpenXRD3D12Extension::get_adapter_luid() const {
	XrGraphicsRequirementsD3D12KHR d3d12_requirements = {
		XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR, // type
		nullptr, // next
		{ 0, 0 }, // adapterLuid
		(D3D_FEATURE_LEVEL)0 // minFeatureLevel
	};

	XrResult result = xrGetD3D12GraphicsRequirementsKHR(OpenXRAPI::get_singleton()->get_instance(), OpenXRAPI::get_singleton()->get_system_id(), &d3d12_requirements);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get Direct3D 12 graphics requirements [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		return {};
	}

	return d3d12_requirements.adapterLuid;
}

void OpenXRD3D12Extension::set_device(ID3D12Device *p_device) {
	graphics_device = p_device;
}

void OpenXRD3D12Extension::set_command_queue(ID3D12CommandQueue *p_queue) {
	command_queue = p_queue;
}

void OpenXRD3D12Extension::cleanup_device() {
	command_queue.Reset();
	graphics_device.Reset();
}

XrGraphicsBindingD3D12KHR OpenXRD3D12Extension::graphics_binding_d3d12;

void *OpenXRD3D12Extension::set_session_create_and_get_next_pointer(void *p_next_pointer) {
	DEV_ASSERT(graphics_device && "Graphics Device was not specified yet.");
	DEV_ASSERT(command_queue && "Command queue was not specified yet.");

	graphics_binding_d3d12.type = XR_TYPE_GRAPHICS_BINDING_D3D12_KHR,
	graphics_binding_d3d12.next = p_next_pointer;
	graphics_binding_d3d12.device = graphics_device.Get();
	graphics_binding_d3d12.queue = command_queue.Get();

	return &graphics_binding_d3d12;
}

void OpenXRD3D12Extension::get_usable_swapchain_formats(Vector<int64_t> &p_usable_swap_chains) {
	p_usable_swap_chains.push_back(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
	p_usable_swap_chains.push_back(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
	p_usable_swap_chains.push_back(DXGI_FORMAT_R8G8B8A8_UNORM);
	p_usable_swap_chains.push_back(DXGI_FORMAT_B8G8R8A8_UNORM);
}

void OpenXRD3D12Extension::get_usable_depth_formats(Vector<int64_t> &p_usable_depth_formats) {
	p_usable_depth_formats.push_back(DXGI_FORMAT_D32_FLOAT);
	p_usable_depth_formats.push_back(DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
	p_usable_depth_formats.push_back(DXGI_FORMAT_D24_UNORM_S8_UINT);
	p_usable_depth_formats.push_back(DXGI_FORMAT_D16_UNORM);
}

bool OpenXRD3D12Extension::get_swapchain_image_data(XrSwapchain p_swapchain, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size, void **r_swapchain_graphics_data) {
	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL_V(rendering_server, false);
	RenderingDevice *rendering_device = rendering_server->get_rendering_device();
	ERR_FAIL_NULL_V(rendering_device, false);

	uint32_t swapchain_length;
	XrResult result = xrEnumerateSwapchainImages(p_swapchain, 0, &swapchain_length, nullptr);
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get swapchain image count [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		return false;
	}

	LocalVector<XrSwapchainImageD3D12KHR> images;
	images.resize(swapchain_length);

	for (XrSwapchainImageD3D12KHR &image : images) {
		image.type = XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR;
		image.next = nullptr;
		image.texture = nullptr;
	}

	result = xrEnumerateSwapchainImages(p_swapchain, swapchain_length, &swapchain_length, (XrSwapchainImageBaseHeader *)images.ptr());
	if (XR_FAILED(result)) {
		print_line("OpenXR: Failed to get swapchain images [", OpenXRAPI::get_singleton()->get_error_string(result), "]");
		return false;
	}

	SwapchainGraphicsData *data = memnew(SwapchainGraphicsData);
	if (data == nullptr) {
		print_line("OpenXR: Failed to allocate memory for swapchain data");
		return false;
	}
	*r_swapchain_graphics_data = data;
	data->is_multiview = (p_array_size > 1);

	RenderingDevice::DataFormat format = RenderingDevice::DATA_FORMAT_R8G8B8A8_SRGB;
	RenderingDevice::TextureSamples samples = RenderingDevice::TEXTURE_SAMPLES_1;
	uint64_t usage_flags = RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT;

	switch (p_swapchain_format) {
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
			// Even though this is an sRGB framebuffer format we're using UNORM here.
			// The reason here is because Godot does a linear to sRGB conversion while
			// with the sRGB format, this conversion would be doubled by the hardware.
			// This also means we're reading the values as is for our preview on screen.
			// The OpenXR runtime however is still treating this as an sRGB format and
			// will thus do an sRGB -> Linear conversion as expected.
			//format = RenderingDevice::DATA_FORMAT_R8G8B8A8_SRGB;
			format = RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
			//format = RenderingDevice::DATA_FORMAT_B8G8R8A8_SRGB;
			format = RenderingDevice::DATA_FORMAT_B8G8R8A8_UNORM;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_R8G8B8A8_UNORM:
			format = RenderingDevice::DATA_FORMAT_R8G8B8A8_UINT;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_B8G8R8A8_UNORM:
			format = RenderingDevice::DATA_FORMAT_B8G8R8A8_UINT;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_D32_FLOAT:
			format = RenderingDevice::DATA_FORMAT_D32_SFLOAT;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
			format = RenderingDevice::DATA_FORMAT_D32_SFLOAT_S8_UINT;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
			format = RenderingDevice::DATA_FORMAT_D24_UNORM_S8_UINT;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			break;
		case DXGI_FORMAT_D16_UNORM:
			format = RenderingDevice::DATA_FORMAT_D16_UNORM;
			usage_flags |= RenderingDevice::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			break;
		default:
			// continue with our default value
			print_line("OpenXR: Unsupported swapchain format", p_swapchain_format);
			break;
	}

	switch (p_sample_count) {
		case 1:
			samples = RenderingDevice::TEXTURE_SAMPLES_1;
			break;
		case 2:
			samples = RenderingDevice::TEXTURE_SAMPLES_2;
			break;
		case 4:
			samples = RenderingDevice::TEXTURE_SAMPLES_4;
			break;
		case 8:
			samples = RenderingDevice::TEXTURE_SAMPLES_8;
			break;
		case 16:
			samples = RenderingDevice::TEXTURE_SAMPLES_16;
			break;
		case 32:
			samples = RenderingDevice::TEXTURE_SAMPLES_32;
			break;
		case 64:
			samples = RenderingDevice::TEXTURE_SAMPLES_64;
			break;
		default:
			// continue with our default value
			print_line("OpenXR: Unsupported sample count", p_sample_count);
			break;
	}

	Vector<RID> texture_rids;

	for (const XrSwapchainImageD3D12KHR &swapchain_image : images) {
		RID texture_rid = rendering_device->texture_create_from_extension(
				p_array_size == 1 ? RenderingDevice::TEXTURE_TYPE_2D : RenderingDevice::TEXTURE_TYPE_2D_ARRAY,
				format,
				samples,
				usage_flags,
				(uint64_t)swapchain_image.texture,
				p_width,
				p_height,
				1,
				p_array_size,
				1);

		texture_rids.push_back(texture_rid);
	}

	data->texture_rids = texture_rids;

	return true;
}

bool OpenXRD3D12Extension::create_projection_fov(const XrFovf p_fov, double p_z_near, double p_z_far, Projection &r_camera_matrix) {
	OpenXRUtil::XrMatrix4x4f matrix;
	OpenXRUtil::XrMatrix4x4f_CreateProjectionFov(&matrix, p_fov, (float)p_z_near, (float)p_z_far);

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			r_camera_matrix.columns[j][i] = matrix.m[j * 4 + i];
		}
	}

	return true;
}

RID OpenXRD3D12Extension::get_texture(void *p_swapchain_graphics_data, int p_image_index) {
	SwapchainGraphicsData *data = (SwapchainGraphicsData *)p_swapchain_graphics_data;
	ERR_FAIL_NULL_V(data, RID());

	ERR_FAIL_INDEX_V(p_image_index, data->texture_rids.size(), RID());
	return data->texture_rids[p_image_index];
}

void OpenXRD3D12Extension::cleanup_swapchain_graphics_data(void **p_swapchain_graphics_data) {
	if (*p_swapchain_graphics_data == nullptr) {
		return;
	}

	RenderingServer *rendering_server = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rendering_server);
	RenderingDevice *rendering_device = rendering_server->get_rendering_device();
	ERR_FAIL_NULL(rendering_device);

	SwapchainGraphicsData *data = (SwapchainGraphicsData *)*p_swapchain_graphics_data;

	for (const RID &texture_rid : data->texture_rids) {
		// This should clean up our RIDs and associated texture objects but shouldn't destroy the images, they are owned by our XrSwapchain.
		rendering_device->free_rid(texture_rid);
	}
	data->texture_rids.clear();

	memdelete(data);
	*p_swapchain_graphics_data = nullptr;
}

#define ENUM_TO_STRING_CASE(e) \
	case e: {                  \
		return String(#e);     \
	} break;

String OpenXRD3D12Extension::get_swapchain_format_name(int64_t p_swapchain_format) const {
	// These are somewhat different per platform, will need to weed some stuff out...
	switch (p_swapchain_format) {
		ENUM_TO_STRING_CASE(DXGI_FORMAT_UNKNOWN)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32A32_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32A32_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32A32_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32A32_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32B32_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16B16A16_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G32_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32G8X24_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R10G10B10A2_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R10G10B10A2_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R10G10B10A2_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R11G11B10_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8B8A8_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16G16_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_D32_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R32_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R24G8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_D24_UNORM_S8_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_X24_TYPELESS_G8_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_FLOAT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_D16_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R16_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8_UINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8_SINT)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_A8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R1_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R9G9B9E5_SHAREDEXP)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R8G8_B8G8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_G8R8_G8B8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC1_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC1_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC1_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC2_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC2_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC2_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC3_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC3_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC3_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC4_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC4_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC4_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC5_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC5_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC5_SNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B5G6R5_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B5G5R5A1_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8A8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8X8_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8A8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8X8_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B8G8R8X8_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC6H_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC6H_UF16)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC6H_SF16)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC7_TYPELESS)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC7_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_BC7_UNORM_SRGB)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_AYUV)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_Y410)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_Y416)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_NV12)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_P010)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_P016)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_420_OPAQUE)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_YUY2)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_Y210)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_Y216)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_NV11)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_AI44)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_IA44)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_P8)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_A8P8)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_B4G4R4A4_UNORM)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_P208)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_V208)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_V408)
		ENUM_TO_STRING_CASE(DXGI_FORMAT_A4B4G4R4_UNORM)
		default: {
			return String("Swapchain format ") + String::num_int64(int64_t(p_swapchain_format));
		} break;
	}
}

#endif // D3D12_ENABLED
