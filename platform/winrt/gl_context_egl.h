#ifndef CONTEXT_EGL_H
#define CONTEXT_EGL_H

#include <wrl.h>

#include "os/os.h"
#include "EGL/egl.h"
#include "error_list.h"
#include "drivers/gl_context/context_gl.h"

using namespace Windows::UI::Core;

class ContextEGL : public ContextGL {

	CoreWindow^ window;

	EGLDisplay mEglDisplay;
	EGLContext mEglContext;
	EGLSurface mEglSurface;

	EGLint width;
	EGLint height;

public:

	virtual void release_current();

	virtual void make_current();

	virtual int get_window_width();
	virtual int get_window_height();
	virtual void swap_buffers();

	virtual Error initialize();
	void reset();

	void cleanup();

	ContextEGL(CoreWindow^ p_window);
	~ContextEGL();

};

#endif

