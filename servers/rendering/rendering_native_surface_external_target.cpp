/**************************************************************************/
/*  rendering_native_surface_external_target.cpp                          */
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

#include "servers/rendering/rendering_native_surface_external_target.h"

#include "servers/display_server_embedded.h"
#include "servers/rendering/rendering_device.h"

#ifdef EXTERNAL_TARGET_ENABLED

#ifdef VULKAN_ENABLED
#include "drivers/vulkan/rendering_device_driver_vulkan.h"
#endif

#if defined(GLES3_ENABLED)
#include "platform_gl.h"
#endif

struct WindowData {
#if defined(GLES3_ENABLED)
	uint32_t backingWidth;
	uint32_t backingHeight;
	GLuint viewFramebuffer;
	GLuint colorTexture;
	GLuint depthTexture;
#endif
};

class GLManagerExternal : public GLManager {
public:
	void set_surface(Ref<RenderingNativeSurfaceExternalTarget> p_surface);
#ifdef GLES3_ENABLED
	void set_opengl_callbacks(Callable p_make_current, Callable p_done_current, uint64_t p_get_proc_address);
#endif
	virtual Error initialize(void *p_native_display = nullptr) override;
	virtual Error open_display(void *p_native_display = nullptr) override { return OK; }
	Error window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) override;
	virtual void window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) override;
	virtual void window_make_current(DisplayServer::WindowID p_id) override;
	virtual void release_current() override {}
	virtual void swap_buffers() override;
	virtual void window_destroy(DisplayServer::WindowID p_id) override;
	virtual Size2i window_get_size(DisplayServer::WindowID p_id) override;
	void deinitialize();
	virtual int window_get_render_target(DisplayServer::WindowID p_id) const override;
	virtual int window_get_color_texture(DisplayServer::WindowID p_id) const override;

	virtual void set_use_vsync(bool p_use) override {}
	virtual bool is_using_vsync() const override { return false; }

	GLManagerExternal() {}
	~GLManagerExternal() {
		deinitialize();
	}

private:
	Ref<RenderingNativeSurfaceExternalTarget> surface;
	HashMap<DisplayServer::WindowID, WindowData> windows;
	HashMap<DisplayServer::WindowID, Ref<RenderingNativeSurface>> window_surface_map;
	HashMap<Ref<RenderingNativeSurface>, DisplayServer::WindowID> surface_window_map;
	DisplayServer::WindowID current_window = -1;
#if defined(GLES3_ENABLED)
	Callable make_current;
	Callable done_current;
#ifdef GLAD_ENABLED
	GLADloadfunc get_proc_address = nullptr;
#endif
#endif
};

void GLManagerExternal::set_surface(Ref<RenderingNativeSurfaceExternalTarget> p_surface) {
	surface = p_surface;
}

#ifdef GLES3_ENABLED
void GLManagerExternal::set_opengl_callbacks(Callable p_make_current, Callable p_done_current, uint64_t p_get_proc_address) {
	make_current = p_make_current;
	done_current = p_done_current;
#ifdef GLAD_ENABLED
	get_proc_address = (GLADloadfunc)p_get_proc_address;
#endif
}
#endif

Error GLManagerExternal::initialize(void *p_native_display) {
#ifdef GLES3_ENABLED
#ifdef GLAD_ENABLED
	RasterizerGLES3::preloadGL(get_proc_address);
#endif

	make_current.call();
#endif
	return OK;
}

Error GLManagerExternal::window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) {
	Ref<RenderingNativeSurfaceExternalTarget> external_surface = Object::cast_to<RenderingNativeSurfaceExternalTarget>(*p_native_surface);
	if (!external_surface.is_valid()) {
		ERR_PRINT("Given surface is not RenderingNativeSurfaceExternalTarget.");
		return FAILED;
	}

#if defined(GLES3_ENABLED)
	WindowData &gles_data = windows[p_id];
	gles_data.backingWidth = surface->get_width();
	gles_data.backingHeight = surface->get_height();

	make_current.call();

	// Generate Framebuffer.
	glGenFramebuffers(1, &gles_data.viewFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gles_data.viewFramebuffer);

	// Bind color texture.
	glGenTextures(1, &gles_data.colorTexture);
	glBindTexture(GL_TEXTURE_2D, gles_data.colorTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gles_data.backingWidth, gles_data.backingHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gles_data.colorTexture, 0);

	// Bind depth texture.
	glGenTextures(1, &gles_data.depthTexture);
	glBindTexture(GL_TEXTURE_2D, gles_data.depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, gles_data.backingWidth, gles_data.backingHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gles_data.depthTexture, 0);

	// Check if framebuffer has been created successfully.
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		ERR_PRINT(vformat("failed to make complete framebuffer object: code %d", glCheckFramebufferStatus(GL_FRAMEBUFFER)));
		return FAILED;
	}

	surface_window_map[external_surface] = p_id;
	window_surface_map[p_id] = external_surface;
#endif
	return OK;
}

void GLManagerExternal::window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	make_current.call();
	window_destroy(p_id);
	surface->resize(Size2i(p_width, p_height));
	window_create(p_id, surface, p_width, p_height);
#endif
}

void GLManagerExternal::window_make_current(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	make_current.call();
	glBindFramebuffer(GL_FRAMEBUFFER, gles_data.viewFramebuffer);
	current_window = p_id;
#endif
}

void GLManagerExternal::swap_buffers() {
	ERR_FAIL_COND(!windows.has(current_window));
#if defined(GLES3_ENABLED)
	make_current.call();
#ifdef DEBUG_ENABLED
	GLenum err = glGetError();
	if (err) {
		ERR_PRINT(vformat("DrawView: %d error", err));
	}
#endif
#endif
}

void GLManagerExternal::window_destroy(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	make_current.call();

	glDeleteFramebuffers(1, &gles_data.viewFramebuffer);
	gles_data.viewFramebuffer = 0;

	glDeleteTextures(1, &gles_data.colorTexture);
	gles_data.colorTexture = 0;

	if (gles_data.depthTexture) {
		glDeleteTextures(1, &gles_data.depthTexture);
		gles_data.depthTexture = 0;
	}

	Ref<RenderingNativeSurfaceExternalTarget> external_surface = window_surface_map[p_id];
	surface_window_map.erase(external_surface);
	window_surface_map.erase(p_id);
#endif
	windows.erase(p_id);
}

Size2i GLManagerExternal::window_get_size(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND_V(!windows.has(p_id), Size2i());
	const WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	return Size2i(surface->get_width(), surface->get_height());
#else
	return Size2i();
#endif
}

void GLManagerExternal::deinitialize() {
#if defined(GLES3_ENABLED)
	done_current.call();
#endif
}

int GLManagerExternal::window_get_render_target(DisplayServer::WindowID p_id) const {
	ERR_FAIL_COND_V(!windows.has(p_id), 0);
	const WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	return gles_data.viewFramebuffer;
#else
	return 0;
#endif
}

int GLManagerExternal::window_get_color_texture(DisplayServer::WindowID p_id) const {
	ERR_FAIL_COND_V(!windows.has(p_id), -1);
	const WindowData &gles_data = windows[p_id];
#if defined(GLES3_ENABLED)
	return gles_data.colorTexture;
#else
	return -1;
#endif
}

void RenderingNativeSurfaceExternalTarget::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_width"), &RenderingNativeSurfaceExternalTarget::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &RenderingNativeSurfaceExternalTarget::get_height);
	ClassDB::bind_method(D_METHOD("get_window"), &RenderingNativeSurfaceExternalTarget::get_window);

	ClassDB::bind_static_method("RenderingNativeSurfaceExternalTarget", D_METHOD("create", "rendering_driver", "initial_size"), &RenderingNativeSurfaceExternalTarget::create_api);

	ClassDB::bind_method(D_METHOD("set_external_swapchain_callbacks", "images_created", "images_released"), &RenderingNativeSurfaceExternalTarget::set_external_swapchain_callbacks);
	ClassDB::bind_method(D_METHOD("resize", "new_size"), &RenderingNativeSurfaceExternalTarget::resize);
	ClassDB::bind_method(D_METHOD("acquire_next_image"), &RenderingNativeSurfaceExternalTarget::acquire_next_image);
	ClassDB::bind_method(D_METHOD("release_image", "index"), &RenderingNativeSurfaceExternalTarget::release_image);

#ifdef GLES3_ENABLED
	ClassDB::bind_method(D_METHOD("set_opengl_callbacks", "make_current", "done_current", "get_proc_address"), &RenderingNativeSurfaceExternalTarget::set_opengl_callbacks);
#endif
	ClassDB::bind_method(D_METHOD("get_frame_texture", "window_id"), &RenderingNativeSurfaceExternalTarget::get_frame_texture);
}

#ifdef VULKAN_ENABLED
RenderingDeviceDriverVulkan *get_rendering_device_driver() {
	RenderingDeviceDriverVulkan *driver = (RenderingDeviceDriverVulkan *)RenderingDevice::get_singleton()->get_driver();

	return driver;
}

RDD::SwapChainID get_swapchain(RenderingContextDriver::SurfaceID p_surface) {
	RenderingDevice *rendering_device = RenderingDevice::get_singleton();
	RenderingContextDriverVulkan *context = (RenderingContextDriverVulkan *)rendering_device->get_context();
	DisplayServer::WindowID window = context->window_get_from_surface(p_surface);

	ERR_FAIL_COND_V_MSG(window == DisplayServer::INVALID_WINDOW_ID, RDD::SwapChainID(), "Window not set for this surface.");

	RDD::SwapChainID swapchain = rendering_device->screen_get_swapchain(window);

	return swapchain;
}
#endif

uint32_t RenderingNativeSurfaceExternalTarget::get_width() const {
	return width;
}

uint32_t RenderingNativeSurfaceExternalTarget::get_height() const {
	return height;
}

DisplayServer::WindowID RenderingNativeSurfaceExternalTarget::get_window() {
	DisplayServer::WindowID window = DisplayServer::INVALID_WINDOW_ID;

#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		RenderingDevice *rendering_device = RenderingDevice::get_singleton();
		RenderingContextDriverVulkan *context = (RenderingContextDriverVulkan *)rendering_device->get_context();
		window = context->window_get_from_surface(surface);
	}
#endif
#ifdef GLES3_ENABLED
	if (rendering_driver == "opengl3") {
		window = DisplayServerEmbedded::get_singleton()->get_native_surface_window_id(Ref<RenderingNativeSurface>(this));
	}
#endif

	return window;
}

void RenderingNativeSurfaceExternalTarget::set_surface(RenderingContextDriver::SurfaceID p_surface) {
	ERR_FAIL_COND_MSG(p_surface == -1, "Invalid surface given.");

	surface = p_surface;
}

RenderingContextDriver::SurfaceID RenderingNativeSurfaceExternalTarget::get_surface_id() {
	return surface;
}

Ref<RenderingNativeSurfaceExternalTarget> RenderingNativeSurfaceExternalTarget::create_api(String p_rendering_driver, Size2i p_initial_size) {
	Ref<RenderingNativeSurfaceExternalTarget> result = nullptr;
#ifdef VULKAN_ENABLED
	result = create(p_rendering_driver, p_initial_size);
#endif
	return result;
}

#ifdef VULKAN_ENABLED
Ref<RenderingNativeSurfaceExternalTarget> RenderingNativeSurfaceExternalTarget::create(String p_rendering_driver, Size2i p_initial_size) {
	Ref<RenderingNativeSurfaceExternalTarget> result(memnew(RenderingNativeSurfaceExternalTarget(p_rendering_driver, p_initial_size.width, p_initial_size.height)));
	return result;
}
#endif

RenderingContextDriver *RenderingNativeSurfaceExternalTarget::create_rendering_context(const String &p_driver_name) {
#if defined(VULKAN_ENABLED)
	return memnew(RenderingContextDriverVulkan);
#else
	return nullptr;
#endif
}

void RenderingNativeSurfaceExternalTarget::setup_external_swapchain_callbacks() {
#ifdef VULKAN_ENABLED
	RenderingDeviceDriverVulkan *driver = get_rendering_device_driver();
	RDD::SwapChainID swapchain = get_swapchain(surface);

	ERR_FAIL_COND_MSG(swapchain == RDD::SwapChainID(), "Swapchain not set for this surface.");

	driver->external_swap_chain_set_callbacks(swapchain, post_images_created_callback, pre_images_released_callback);
#endif
}

void RenderingNativeSurfaceExternalTarget::set_external_swapchain_callbacks(Callable p_images_created, Callable p_images_released) {
	// NOTE: p_images_created: host wraps godot's swapchain images into qsgvulkantextures usable by the host
	// NOTE: p_images_released: host frees all it's qsgvulkantextures

	post_images_created_callback = p_images_created;
	pre_images_released_callback = p_images_released;
}

void RenderingNativeSurfaceExternalTarget::resize(Size2i p_new_size) {
	width = p_new_size.x;
	height = p_new_size.y;

#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		RenderingDevice *rendering_device = RenderingDevice::get_singleton();
		RenderingContextDriverVulkan *context = (RenderingContextDriverVulkan *)rendering_device->get_context();
		context->surface_set_size(surface, p_new_size.x, p_new_size.y);
	}
#endif
}

int RenderingNativeSurfaceExternalTarget::acquire_next_image() {
#ifdef VULKAN_ENABLED
	RenderingDeviceDriverVulkan *driver = get_rendering_device_driver();
	RDD::SwapChainID swapchain = get_swapchain(surface);

	ERR_FAIL_COND_V_MSG(swapchain == RDD::SwapChainID(), -1, "Swapchain not set for this surface.");

	int acquired_buffer_index = driver->external_swap_chain_grab_image(swapchain);
	return acquired_buffer_index;
#endif
}

void RenderingNativeSurfaceExternalTarget::release_image(int p_index) {
#ifdef VULKAN_ENABLED
	RenderingDeviceDriverVulkan *driver = get_rendering_device_driver();
	RDD::SwapChainID swapchain = get_swapchain(surface);

	ERR_FAIL_COND_MSG(swapchain == RDD::SwapChainID(), "Swapchain not set for this surface.");

	driver->external_swap_chain_release_image(swapchain, p_index);
#endif
}

#ifdef GLES3_ENABLED
void RenderingNativeSurfaceExternalTarget::set_opengl_callbacks(Callable p_make_current, Callable p_done_current, uint64_t p_get_proc_address) {
	make_current = p_make_current;
	done_current = p_done_current;
	get_proc_address = p_get_proc_address;
}
#endif

GLManager *RenderingNativeSurfaceExternalTarget::create_gl_manager(const String &p_driver_name) {
#if defined(GLES3_ENABLED)
	GLManagerExternal *gl_manager = memnew(GLManagerExternal);
	gl_manager->set_surface(Ref<RenderingNativeSurfaceExternalTarget>(this));
	gl_manager->set_opengl_callbacks(make_current, done_current, get_proc_address);
	return (GLManager *)gl_manager;
#else
	return nullptr;
#endif
}

uint32_t RenderingNativeSurfaceExternalTarget::get_frame_texture(DisplayServer::WindowID p_window_id) const {
	return DisplayServer::get_singleton()->window_get_native_handle(DisplayServer::WINDOW_HANDLE, p_window_id);
}

RenderingNativeSurfaceExternalTarget::RenderingNativeSurfaceExternalTarget(String p_rendering_driver, int p_width, int p_height) {
	rendering_driver = p_rendering_driver;
	width = p_width;
	height = p_height;
}
#endif
