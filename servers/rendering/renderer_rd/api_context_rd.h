/**************************************************************************/
/*  api_context_rd.h                                                      */
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

#ifndef API_CONTEXT_RD_H
#define API_CONTEXT_RD_H

#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_device_driver.h"

class ApiContextRD {
public:
	virtual const char *get_api_name() const = 0;
	virtual RenderingDevice::Capabilities get_device_capabilities() const = 0;
	virtual const RDD::MultiviewCapabilities &get_multiview_capabilities() const = 0;

	virtual int get_swapchain_image_count() const = 0;

	virtual Error window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, int p_width, int p_height, const void *p_platform_data) = 0;
	virtual void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) = 0;
	virtual int window_get_width(DisplayServer::WindowID p_window = 0) = 0;
	virtual int window_get_height(DisplayServer::WindowID p_window = 0) = 0;
	virtual bool window_is_valid_swapchain(DisplayServer::WindowID p_window = 0) = 0;
	virtual void window_destroy(DisplayServer::WindowID p_window_id) = 0;
	virtual RDD::RenderPassID window_get_render_pass(DisplayServer::WindowID p_window = 0) = 0;
	virtual RDD::FramebufferID window_get_framebuffer(DisplayServer::WindowID p_window = 0) = 0;

	virtual RID local_device_create() = 0;
	virtual void local_device_push_command_buffers(RID p_local_device, const RDD::CommandBufferID *p_buffers, int p_count) = 0;
	virtual void local_device_sync(RID p_local_device) = 0;
	virtual void local_device_free(RID p_local_device) = 0;

	virtual void set_setup_buffer(RDD::CommandBufferID p_command_buffer) = 0;
	virtual void append_command_buffer(RDD::CommandBufferID p_command_buffer) = 0;
	virtual void flush(bool p_flush_setup = false, bool p_flush_pending = false) = 0;
	virtual Error prepare_buffers(RDD::CommandBufferID p_command_buffer) = 0;
	virtual void postpare_buffers(RDD::CommandBufferID p_command_buffer) = 0;
	virtual Error swap_buffers() = 0;
	virtual Error initialize() = 0;

	virtual void command_begin_label(RDD::CommandBufferID p_command_buffer, String p_label_name, const Color &p_color) = 0;
	virtual void command_insert_label(RDD::CommandBufferID p_command_buffer, String p_label_name, const Color &p_color) = 0;
	virtual void command_end_label(RDD::CommandBufferID p_command_buffer) = 0;

	virtual String get_device_vendor_name() const = 0;
	virtual String get_device_name() const = 0;
	virtual RDD::DeviceType get_device_type() const = 0;
	virtual String get_device_api_version() const = 0;
	virtual String get_device_pipeline_cache_uuid() const = 0;

	virtual void set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_mode) = 0;
	virtual DisplayServer::VSyncMode get_vsync_mode(DisplayServer::WindowID p_window = 0) const = 0;

	virtual RenderingDeviceDriver *get_driver(RID p_local_device = RID()) = 0;

	virtual ~ApiContextRD();
};

#endif // API_CONTEXT_RD_H
