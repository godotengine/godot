/**************************************************************************/
/*  rendering_native_surface_apple.mm                                     */
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

#include "rendering_native_surface_apple.h"
#include "drivers/gles3/storage/texture_storage.h"
#include "drivers/metal/rendering_context_driver_metal.h"
#include "servers/rendering/gl_manager.h"

#import "rendering_context_driver_vulkan_apple.h"

#import <QuartzCore/CAMetalLayer.h>

#if defined(GLES3_ENABLED)
#import <QuartzCore/QuartzCore.h>

#if defined(IOS_ENABLED)
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#endif

struct WindowData {
	GLint backingWidth;
	GLint backingHeight;
	GLuint viewRenderbuffer, viewFramebuffer;
	GLuint depthRenderbuffer;
#if defined(IOS_ENABLED)
	CAEAGLLayer *layer;
#endif
#if defined(MACOS_ENABLED)
	CAOpenGLLayer *layer;
#endif
};

#define GL_ERR(expr)                                             \
	{                                                            \
		expr;                                                    \
		GLenum err = glGetError();                               \
		if (err) {                                               \
			NSLog(@"%s:%s: %x error", __FUNCTION__, #expr, err); \
		}                                                        \
	}

class GLManagerApple : public GLManager {
	DisplayServer::WindowID current_window = -1;

public:
	virtual Error initialize(void *p_native_display = nullptr) override;
	virtual Error open_display(void *p_native_display = nullptr) override { return OK; }
	virtual Error window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) override;
	virtual void window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) override;
	virtual void window_make_current(DisplayServer::WindowID p_id) override;
	virtual void release_current() override {}
	virtual void swap_buffers() override;
	virtual void window_destroy(DisplayServer::WindowID p_id) override;
	virtual Size2i window_get_size(DisplayServer::WindowID p_id) override;
	void deinitialize();

	virtual void set_use_vsync(bool p_use) override {}
	virtual bool is_using_vsync() const override { return false; }

	virtual int window_get_render_target(DisplayServer::WindowID p_window_id) const override;
	virtual int window_get_color_texture(DisplayServer::WindowID p_id) const override { return 0; }

	virtual ~GLManagerApple() {
		deinitialize();
	}

protected:
	Error create_framebuffer(DisplayServer::WindowID p_id, void *p_layer);

private:
	HashMap<DisplayServer::WindowID, WindowData> windows;
#if defined(IOS_ENABLED)
	EAGLContext *context = nullptr;
#endif
};

Error GLManagerApple::initialize(void *p_native_display) {
#if defined(IOS_ENABLED)
	// Create GL ES 3 context
	if (OS::get_singleton()->get_current_rendering_method() == "gl_compatibility" && context == nullptr) {
		context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
		ERR_FAIL_COND_V_MSG(!context, FAILED, "Failed to create OpenGL ES 3.0 context!");
	}

	if (![EAGLContext setCurrentContext:context]) {
		ERR_FAIL_V_MSG(FAILED, "Unable to set current EAGLContext");
	}
#endif
	return OK;
}

Size2i GLManagerApple::window_get_size(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND_V(!windows.has(p_id), Size2i());
	WindowData &gles_data = windows[p_id];
	return Size2i(gles_data.layer.bounds.size.width, gles_data.layer.bounds.size.height);
}

void GLManagerApple::window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(IOS_ENABLED)
	GL_ERR([EAGLContext setCurrentContext:context]);
	CAEAGLLayer *layer = gles_data.layer;
	window_destroy(p_id);
	create_framebuffer(p_id, (__bridge void *)layer);
#endif
}

void GLManagerApple::window_make_current(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(IOS_ENABLED)
	GL_ERR([EAGLContext setCurrentContext:context]);
	GL_ERR(glBindFramebufferOES(GL_FRAMEBUFFER_OES, gles_data.viewFramebuffer));
	current_window = p_id;
#endif
}

void GLManagerApple::swap_buffers() {
	ERR_FAIL_COND(!windows.has(current_window));
	WindowData &gles_data = windows[current_window];
#if defined(IOS_ENABLED)
	GL_ERR([EAGLContext setCurrentContext:context]);
	GL_ERR(glBindRenderbufferOES(GL_RENDERBUFFER_OES, gles_data.viewRenderbuffer));
	GL_ERR([context presentRenderbuffer:GL_RENDERBUFFER_OES]);
#endif
}

void GLManagerApple::deinitialize() {
#if defined(IOS_ENABLED)
	if ([EAGLContext currentContext] == context) {
		[EAGLContext setCurrentContext:nil];
	}

	if (context) {
		context = nil;
	}
#endif
}

Error GLManagerApple::window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) {
#if defined(IOS_ENABLED)
	NSLog(@"GLESContextApple::create_framebuffer surface");
	CAEAGLLayer *layer = nullptr;
	Ref<RenderingNativeSurfaceApple> apple_surface = Object::cast_to<RenderingNativeSurfaceApple>(*p_native_surface);
	if (apple_surface.is_valid()) {
		layer = (__bridge CAEAGLLayer *)(void *)apple_surface->get_layer();
	}
	ERR_FAIL_COND_V_MSG(layer == nullptr, ERR_CANT_CREATE, "Unable to create GL window");

	return create_framebuffer(p_id, (__bridge void *)layer);
#else
	return FAILED;
#endif
}

Error GLManagerApple::create_framebuffer(DisplayServer::WindowID p_id, void *p_layer) {
	WindowData &gles_data = windows[p_id];
#if defined(IOS_ENABLED)
	NSLog(@"GLESContextApple::create_framebuffer layer");
	GL_ERR([EAGLContext setCurrentContext:context]);
	gles_data.layer = (__bridge CAEAGLLayer *)p_layer;

	GL_ERR(glGenFramebuffersOES(1, &gles_data.viewFramebuffer));
	GL_ERR(glGenRenderbuffersOES(1, &gles_data.viewRenderbuffer));

	GL_ERR(glBindFramebufferOES(GL_FRAMEBUFFER_OES, gles_data.viewFramebuffer));
	GL_ERR(glBindRenderbufferOES(GL_RENDERBUFFER_OES, gles_data.viewRenderbuffer));
	// This call associates the storage for the current render buffer with the EAGLDrawable (our CAself)
	// allowing us to draw into a buffer that will later be rendered to screen wherever the layer is (which corresponds with our view).
	[CATransaction flush];
	GL_ERR([context renderbufferStorage:GL_RENDERBUFFER_OES fromDrawable:gles_data.layer]);
	GL_ERR(glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_COLOR_ATTACHMENT0_OES, GL_RENDERBUFFER_OES, gles_data.viewRenderbuffer));

	GL_ERR(glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_WIDTH_OES, &gles_data.backingWidth));
	GL_ERR(glGetRenderbufferParameterivOES(GL_RENDERBUFFER_OES, GL_RENDERBUFFER_HEIGHT_OES, &gles_data.backingHeight));

	// For this sample, we also need a depth buffer, so we'll create and attach one via another renderbuffer.
	GL_ERR(glGenRenderbuffersOES(1, &gles_data.depthRenderbuffer));
	GL_ERR(glBindRenderbufferOES(GL_RENDERBUFFER_OES, gles_data.depthRenderbuffer));
	GL_ERR(glRenderbufferStorageOES(GL_RENDERBUFFER_OES, GL_DEPTH_COMPONENT16_OES, gles_data.backingWidth, gles_data.backingHeight));
	GL_ERR(glFramebufferRenderbufferOES(GL_FRAMEBUFFER_OES, GL_DEPTH_ATTACHMENT_OES, GL_RENDERBUFFER_OES, gles_data.depthRenderbuffer));

	if (glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES) != GL_FRAMEBUFFER_COMPLETE_OES) {
		NSLog(@"failed to make complete framebuffer object %x", glCheckFramebufferStatusOES(GL_FRAMEBUFFER_OES));
		return FAILED;
	}

	return OK;
#else
	return FAILED;
#endif
}

// Clean up any buffers we have allocated.
void GLManagerApple::window_destroy(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
#if defined(IOS_ENABLED)
	GL_ERR([EAGLContext setCurrentContext:context]);
	GL_ERR(glDeleteFramebuffersOES(1, &gles_data.viewFramebuffer));
	gles_data.viewFramebuffer = 0;
	GL_ERR(glDeleteRenderbuffersOES(1, &gles_data.viewRenderbuffer));
	gles_data.viewRenderbuffer = 0;

	if (gles_data.depthRenderbuffer) {
		GL_ERR(glDeleteRenderbuffersOES(1, &gles_data.depthRenderbuffer));
		gles_data.depthRenderbuffer = 0;
	}
#endif
	windows.erase(p_id);
}

int GLManagerApple::window_get_render_target(DisplayServer::WindowID p_id) const {
	ERR_FAIL_COND_V(!windows.has(p_id), 0);
	const WindowData &gles_data = windows[p_id];
	return gles_data.viewFramebuffer;
}

#endif // GLES3_ENABLED

void RenderingNativeSurfaceApple::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceApple", D_METHOD("create", "layer"), &RenderingNativeSurfaceApple::create_api);
	ClassDB::bind_method(D_METHOD("get_layer"), &RenderingNativeSurfaceApple::get_layer);
}

Ref<RenderingNativeSurfaceApple> RenderingNativeSurfaceApple::create_api(/* GDExtensionConstPtr<const void> */ uint64_t p_layer) {
	return RenderingNativeSurfaceApple::create((void *)p_layer /* .operator const void *() */);
}

Ref<RenderingNativeSurfaceApple> RenderingNativeSurfaceApple::create(void *p_layer) {
	Ref<RenderingNativeSurfaceApple> result;
	if (!p_layer) {
		String rendering_driver = ::OS::get_singleton()->get_current_rendering_driver_name();
		CALayer *__block myLayer = nil;
		dispatch_sync(dispatch_get_main_queue(), ^{
#if defined(GLES3_ENABLED)
			if (rendering_driver == "opengl3") {
#if defined(IOS_ENABLED)
				myLayer = [[CAEAGLLayer alloc] init];
#elif defined(MACOS_ENABLED)
				myLayer = [[CAOpenGLLayer alloc] init];
#endif
			}
#endif
			if (rendering_driver == "vulkan") {
				myLayer = [[CAMetalLayer alloc] init];
			}
		});
		if (!myLayer) {
			return result;
		}
		p_layer = (void *)CFBridgingRetain(myLayer);
	} else {
		p_layer = (void *)CFBridgingRetain((__bridge CALayer *)p_layer);
	}

	result.instantiate();
	result->layer = p_layer;
	return result;
}

uint64_t RenderingNativeSurfaceApple::get_layer() {
	return (uint64_t)layer;
}

void *RenderingNativeSurfaceApple::get_native_id() const {
	return (void *)layer;
}

RenderingContextDriver *RenderingNativeSurfaceApple::create_rendering_context(const String &p_rendering_driver) {
#if defined(VULKAN_ENABLED)
	if (p_rendering_driver == "vulkan") {
		return memnew(RenderingContextDriverVulkanApple);
	}
#endif
#if defined(METAL_ENABLED)
	if (p_rendering_driver == "metal") {
		if (@available(ios 14.0, *)) {
			return memnew(RenderingContextDriverMetal);
		}
	}
#endif
	return nullptr;
}

GLManager *RenderingNativeSurfaceApple::create_gl_manager(const String &p_driver_name) {
#if defined(GLES3_ENABLED)
	if (p_driver_name == "opengl3") {
		return memnew(GLManagerApple);
	}
#endif
	return nullptr;
}

RenderingNativeSurfaceApple::RenderingNativeSurfaceApple() {
}

RenderingNativeSurfaceApple::~RenderingNativeSurfaceApple() {
	if (layer) {
		CFBridgingRelease(layer);
	}
}
