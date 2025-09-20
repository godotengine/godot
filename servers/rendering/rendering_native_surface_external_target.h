/**************************************************************************/
/*  rendering_native_surface_external_target.h                            */
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

#pragma once

#include "servers/rendering/rendering_native_surface.h"

#include "core/variant/native_ptr.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#include "servers/display_server.h"
#include "servers/rendering/gl_manager.h"

#ifdef EXTERNAL_TARGET_ENABLED

class GLManagerExternal;

class RenderingNativeSurfaceExternalTarget : public RenderingNativeSurface { // Defines both Vulkan and OpenGL behavior.
	GDCLASS(RenderingNativeSurfaceExternalTarget, RenderingNativeSurface);

	static void _bind_methods();

private:
	// Common members:
	String rendering_driver;

	uint32_t width = 0;
	uint32_t height = 0;

	// Vulkan specific members:
	Callable post_images_created_callback;
	Callable pre_images_released_callback;
	RenderingContextDriver::SurfaceID surface;

#ifdef GLES3_ENABLED
	// OpenGL specific members:
	Callable make_current;
	Callable done_current;
	uint64_t get_proc_address = 0u;
#endif

public:
	// VULKAN SPECIFIC OPERATIONS:
	uint32_t get_width() const;
	uint32_t get_height() const;
	DisplayServer::WindowID get_window();
	void set_surface(RenderingContextDriver::SurfaceID p_surface);
	RenderingContextDriver::SurfaceID get_surface_id();

	static Ref<RenderingNativeSurfaceExternalTarget> create_api(String p_rendering_driver, Size2i p_initial_size);

#ifdef VULKAN_ENABLED
	static Ref<RenderingNativeSurfaceExternalTarget> create(String p_rendering_driver, Size2i p_initial_size);
#endif

	virtual RenderingContextDriver *create_rendering_context(const String &p_driver_name) override final;

	virtual void setup_external_swapchain_callbacks() override final;

	// Called by host (registered in GDExtension):

	void set_external_swapchain_callbacks(Callable p_images_created, Callable p_images_released);

	void resize(Size2i p_new_size);

	// Call in Qt to get the next image that is already ready by Godot
	// Wrap it in QSGTexture and set it as render target
	int acquire_next_image();

	// When the rendering is done for an image, call this in Qt to release it, so Godot knows that it can use it as a render target again.
	void release_image(int p_index);

	// OPENGL SPECIFIC OPERATIONS:
#ifdef GLES3_ENABLED
	void set_opengl_callbacks(Callable p_make_current, Callable p_done_current, uint64_t p_get_proc_address);
#endif
	virtual GLManager *create_gl_manager(const String &p_driver_name) override;
	uint32_t get_frame_texture(DisplayServer::WindowID p_window_id) const;

	virtual void *get_native_id() const override { return nullptr; }

	RenderingNativeSurfaceExternalTarget() {}
	RenderingNativeSurfaceExternalTarget(String p_rendering_driver, int p_width, int p_height);
	~RenderingNativeSurfaceExternalTarget() {}
};

#endif
