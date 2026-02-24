/**************************************************************************/
/*  rendering_context_driver_metal.cpp                                    */
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

#include "rendering_context_driver_metal.h"

#include "metal3_objects.h"
#include "metal_objects_shared.h"
#include "rendering_device_driver_metal3.h"

#include "core/templates/sort_array.h"

#include <os/log.h>
#include <os/signpost.h>

#include <objc/message.h>

#pragma mark - Logging

os_log_t LOG_DRIVER;
// Used for dynamic tracing.
os_log_t LOG_INTERVALS;

__attribute__((constructor)) static void InitializeLogging(void) {
	LOG_DRIVER = os_log_create("org.godotengine.godot.metal", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
	LOG_INTERVALS = os_log_create("org.godotengine.godot.metal", "events");
}

RenderingContextDriverMetal::RenderingContextDriverMetal() {
}

RenderingContextDriverMetal::~RenderingContextDriverMetal() {
}

Error RenderingContextDriverMetal::initialize() {
	if (OS::get_singleton()->get_environment("MTL_CAPTURE_ENABLED") == "1" || OS::get_singleton()->get_environment("MTLCAPTURE_DESTINATION_DEVELOPER_TOOLS_ENABLE") == "1") {
		capture_available = true;
	}

	metal_device = MTL::CreateSystemDefaultDevice();
#if TARGET_OS_OSX
	if (__builtin_available(macOS 13.3, *)) {
		metal_device->setShouldMaximizeConcurrentCompilation(true);
	}
#endif
	device.type = DEVICE_TYPE_INTEGRATED_GPU;
	device.vendor = Vendor::VENDOR_APPLE;
	device.workarounds = Workarounds();

	MetalDeviceProperties props(metal_device);
	int version = (int)props.features.highestFamily - (int)MTL::GPUFamilyApple1 + 1;
	device.name = vformat("%s (Apple%d)", metal_device->name()->utf8String(), version);

	return OK;
}

const RenderingContextDriver::Device &RenderingContextDriverMetal::device_get(uint32_t p_device_index) const {
	DEV_ASSERT(p_device_index < 1);
	return device;
}

uint32_t RenderingContextDriverMetal::device_get_count() const {
	return 1;
}

RenderingDeviceDriver *RenderingContextDriverMetal::driver_create() {
	return memnew(MTL3::RenderingDeviceDriverMetal(this));
}

void RenderingContextDriverMetal::driver_free(RenderingDeviceDriver *p_driver) {
	memdelete(p_driver);
}

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) SurfaceLayer : public RenderingContextDriverMetal::Surface {
	CA::MetalLayer *layer = nullptr;
	LocalVector<MDFrameBuffer> frame_buffers;
	LocalVector<MTL::Drawable *> drawables;
	uint32_t rear = -1;
	uint32_t front = 0;
	uint32_t count = 0;

public:
	SurfaceLayer(CA::MetalLayer *p_layer, MTL::Device *p_device) :
			Surface(p_device), layer(p_layer) {
		layer->setAllowsNextDrawableTimeout(true);
		layer->setFramebufferOnly(true);
		layer->setOpaque(!OS::get_singleton()->is_layered_allowed());
		layer->setPixelFormat(get_pixel_format());
		layer->setDevice(p_device);
	}

	~SurfaceLayer() override {
		layer = nullptr;
	}

	Error resize(uint32_t p_desired_framebuffer_count) override final {
		if (width == 0 || height == 0) {
			// Very likely the window is minimized, don't create a swap chain.
			return ERR_SKIP;
		}

		CGSize drawableSize = CGSizeMake(width, height);
		CGSize current = layer->drawableSize();
		if (!CGSizeEqualToSize(current, drawableSize)) {
			layer->setDrawableSize(drawableSize);
		}

		// Metal supports a maximum of 3 drawables.
		p_desired_framebuffer_count = MIN(3U, p_desired_framebuffer_count);
		layer->setMaximumDrawableCount(p_desired_framebuffer_count);

#if TARGET_OS_OSX
		// Display sync is only supported on macOS.
		switch (vsync_mode) {
			case DisplayServer::VSYNC_MAILBOX:
			case DisplayServer::VSYNC_ADAPTIVE:
			case DisplayServer::VSYNC_ENABLED:
				layer->setDisplaySyncEnabled(true);
				break;
			case DisplayServer::VSYNC_DISABLED:
				layer->setDisplaySyncEnabled(false);
				break;
		}
#endif
		drawables.resize(p_desired_framebuffer_count);
		frame_buffers.resize(p_desired_framebuffer_count);
		for (uint32_t i = 0; i < p_desired_framebuffer_count; i++) {
			// Reserve space for the drawable texture.
			frame_buffers[i].set_texture_count(1);
		}

		return OK;
	}

	RDD::FramebufferID acquire_next_frame_buffer() override final {
		if (count == frame_buffers.size()) {
			return RDD::FramebufferID();
		}

		rear = (rear + 1) % frame_buffers.size();
		count++;

		MDFrameBuffer &frame_buffer = frame_buffers[rear];
		frame_buffer.size = Size2i(width, height);

		CA::MetalDrawable *drawable = layer->nextDrawable();
		ERR_FAIL_NULL_V_MSG(drawable, RDD::FramebufferID(), "no drawable available");
		drawables[rear] = drawable;
		frame_buffer.set_texture(0, drawable->texture());

		return RDD::FramebufferID(&frame_buffer);
	}

	void present(MTL3::MDCommandBuffer *p_cmd_buffer) override final {
		if (count == 0) {
			return;
		}

		// Release texture and drawable.
		frame_buffers[front].unset_texture(0);
		MTL::Drawable *drawable = drawables[front];
		drawables[front] = nullptr;

		count--;
		front = (front + 1) % frame_buffers.size();

		if (vsync_mode != DisplayServer::VSYNC_DISABLED) {
			p_cmd_buffer->get_command_buffer()->presentDrawableAfterMinimumDuration(drawable, present_minimum_duration);
		} else {
			p_cmd_buffer->get_command_buffer()->presentDrawable(drawable);
		}
	}

	MTL::Drawable *next_drawable() override final {
		if (count == 0) {
			return nullptr;
		}

		// Release texture and drawable.
		frame_buffers[front].unset_texture(0);
		MTL::Drawable *drawable = drawables[front];
		drawables[front] = nullptr;

		count--;
		front = (front + 1) % frame_buffers.size();

		return drawable;
	}

	API_AVAILABLE(macos(26.0), ios(26.0))
	MTL::ResidencySet *get_residency_set() const override final {
		return layer->residencySet();
	}
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) SurfaceOffscreen : public RenderingContextDriverMetal::Surface {
	int frame_buffer_size = 3;
	MDFrameBuffer *frame_buffers;
	LocalVector<MTL::Texture *> textures;
	LocalVector<MTL::Drawable *> drawables;

	int32_t rear = -1;
	std::atomic_int count;
	uint64_t target_time = 0;
	CA::MetalLayer *layer;

public:
	SurfaceOffscreen(CA::MetalLayer *p_layer, MTL::Device *p_device) :
			Surface(p_device), layer(p_layer) {
		layer->setAllowsNextDrawableTimeout(true);
		layer->setFramebufferOnly(true);
		layer->setOpaque(!OS::get_singleton()->is_layered_allowed());
		layer->setPixelFormat(get_pixel_format());
		layer->setDevice(p_device);
#if TARGET_OS_OSX
		layer->setDisplaySyncEnabled(false);
#endif
		target_time = OS::get_singleton()->get_ticks_usec();

		textures.resize(frame_buffer_size);
		drawables.resize(frame_buffer_size);

		frame_buffers = memnew_arr(MDFrameBuffer, frame_buffer_size);
		for (int i = 0; i < frame_buffer_size; i++) {
			frame_buffers[i].set_texture_count(1);
		}
	}

	~SurfaceOffscreen() override {
		memdelete_arr(frame_buffers);
	}

	Error resize(uint32_t p_desired_framebuffer_count) override final {
		if (width == 0 || height == 0) {
			// Very likely the window is minimized, don't create a swap chain.
			return ERR_SKIP;
		}

		CGSize drawableSize = CGSizeMake(width, height);
		CGSize current = layer->drawableSize();
		if (!CGSizeEqualToSize(current, drawableSize)) {
			layer->setDrawableSize(drawableSize);
		}

		return OK;
	}

	RDD::FramebufferID acquire_next_frame_buffer() override final {
		if (count.load(std::memory_order_relaxed) == 3) {
			// Wait for a frame to be presented.
			return RDD::FramebufferID();
		}

		rear = (rear + 1) % 3;
		count.fetch_add(1, std::memory_order_relaxed);

		MDFrameBuffer &frame_buffer = frame_buffers[rear];

		if (textures[rear] == nullptr || textures[rear]->width() != width || textures[rear]->height() != height) {
			MTL::TextureDescriptor *texture_descriptor = MTL::TextureDescriptor::texture2DDescriptor(get_pixel_format(), width, height, false);
			texture_descriptor->setUsage(MTL::TextureUsageRenderTarget);
			texture_descriptor->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
			texture_descriptor->setStorageMode(MTL::StorageModePrivate);
			textures[rear] = device->newTexture(texture_descriptor);
		}

		frame_buffer.size = Size2i(width, height);
		uint64_t now = OS::get_singleton()->get_ticks_usec();
		if (now >= target_time) {
			target_time = now + 1'000'000; // 1 second into the future.
			CA::MetalDrawable *drawable = layer->nextDrawable();
			ERR_FAIL_NULL_V_MSG(drawable, RDD::FramebufferID(), "no drawable available");
			drawables[rear] = drawable;
			frame_buffer.set_texture(0, drawable->texture());
		} else {
			frame_buffer.set_texture(0, textures[rear]);
		}

		return RDD::FramebufferID(&frame_buffers[rear]);
	}

	void present(MTL3::MDCommandBuffer *p_cmd_buffer) override final {
		MDFrameBuffer *frame_buffer = &frame_buffers[rear];

		if (drawables[rear] != nullptr) {
			p_cmd_buffer->get_command_buffer()->presentDrawable(drawables[rear]);
			drawables[rear] = nullptr;
		}

		p_cmd_buffer->get_command_buffer()->addScheduledHandler([frame_buffer, this](MTL::CommandBuffer *) {
			frame_buffer->unset_texture(0);
			count.fetch_add(-1, std::memory_order_relaxed);
		});
	}

	MTL::Drawable *next_drawable() override final {
		if (count == 0) {
			return nullptr;
		}

		MDFrameBuffer *frame_buffer = &frame_buffers[rear];

		MTL::Drawable *next = drawables[rear];
		drawables[rear] = nullptr;

		frame_buffer->unset_texture(0);
		count--;

		return next;
	}

	API_AVAILABLE(macos(26.0), ios(26.0))
	MTL::ResidencySet *get_residency_set() const override final {
		return layer->residencySet();
	}
};

RenderingContextDriver::SurfaceID RenderingContextDriverMetal::surface_create(const void *p_platform_data) {
	const WindowPlatformData *wpd = (const WindowPlatformData *)(p_platform_data);
	Surface *surface;
	if (String v = OS::get_singleton()->get_environment("GODOT_MTL_OFF_SCREEN"); v == U"1") {
		surface = memnew(SurfaceOffscreen(wpd->layer, metal_device));
	} else {
		surface = memnew(SurfaceLayer(wpd->layer, metal_device));
	}

	return SurfaceID(surface);
}

void RenderingContextDriverMetal::surface_set_size(SurfaceID p_surface, uint32_t p_width, uint32_t p_height) {
	Surface *surface = (Surface *)(p_surface);
	if (surface->width == p_width && surface->height == p_height) {
		return;
	}
	surface->width = p_width;
	surface->height = p_height;
	surface->needs_resize = true;
}

void RenderingContextDriverMetal::surface_set_vsync_mode(SurfaceID p_surface, DisplayServer::VSyncMode p_vsync_mode) {
	Surface *surface = (Surface *)(p_surface);
	if (surface->vsync_mode == p_vsync_mode) {
		return;
	}
	surface->vsync_mode = p_vsync_mode;
	surface->needs_resize = true;
}

DisplayServer::VSyncMode RenderingContextDriverMetal::surface_get_vsync_mode(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->vsync_mode;
}

void RenderingContextDriverMetal::surface_set_hdr_output_enabled(SurfaceID p_surface, bool p_enabled) {
	Surface *surface = (Surface *)(p_surface);
	surface->hdr_output = p_enabled;
	surface->needs_resize = true;
}

bool RenderingContextDriverMetal::surface_get_hdr_output_enabled(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->hdr_output;
}

void RenderingContextDriverMetal::surface_set_hdr_output_reference_luminance(SurfaceID p_surface, float p_reference_luminance) {
	Surface *surface = (Surface *)(p_surface);
	surface->hdr_reference_luminance = p_reference_luminance;
}

float RenderingContextDriverMetal::surface_get_hdr_output_reference_luminance(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->hdr_reference_luminance;
}

void RenderingContextDriverMetal::surface_set_hdr_output_max_luminance(SurfaceID p_surface, float p_max_luminance) {
	Surface *surface = (Surface *)(p_surface);
	surface->hdr_max_luminance = p_max_luminance;
}

float RenderingContextDriverMetal::surface_get_hdr_output_max_luminance(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->hdr_max_luminance;
}

void RenderingContextDriverMetal::surface_set_hdr_output_linear_luminance_scale(SurfaceID p_surface, float p_linear_luminance_scale) {
	Surface *surface = (Surface *)(p_surface);
	surface->hdr_linear_luminance_scale = p_linear_luminance_scale;
}

float RenderingContextDriverMetal::surface_get_hdr_output_linear_luminance_scale(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->hdr_linear_luminance_scale;
}

float RenderingContextDriverMetal::surface_get_hdr_output_max_value(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return MAX(surface->hdr_max_luminance / MAX(surface->hdr_reference_luminance, 1.0f), 1.0f);
}

uint32_t RenderingContextDriverMetal::surface_get_width(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->width;
}

uint32_t RenderingContextDriverMetal::surface_get_height(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->height;
}

void RenderingContextDriverMetal::surface_set_needs_resize(SurfaceID p_surface, bool p_needs_resize) {
	Surface *surface = (Surface *)(p_surface);
	surface->needs_resize = p_needs_resize;
}

bool RenderingContextDriverMetal::surface_get_needs_resize(SurfaceID p_surface) const {
	Surface *surface = (Surface *)(p_surface);
	return surface->needs_resize;
}

void RenderingContextDriverMetal::surface_destroy(SurfaceID p_surface) {
	Surface *surface = (Surface *)(p_surface);
	memdelete(surface);
}
