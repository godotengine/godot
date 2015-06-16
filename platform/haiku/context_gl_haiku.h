#ifndef CONTEXT_GL_HAIKU_H
#define CONTEXT_GL_HAIKU_H

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)

#include "drivers/gl_context/context_gl.h"

#include "haiku_direct_window.h"
#include "haiku_gl_view.h"

class ContextGL_Haiku : public ContextGL {
private:
	HaikuGLView* view;
	HaikuDirectWindow* window;

public:
	ContextGL_Haiku(HaikuDirectWindow* p_window);
	~ContextGL_Haiku();

	virtual Error initialize();
	virtual void release_current();	
	virtual void make_current();	
	virtual void swap_buffers();
	virtual int get_window_width();
	virtual int get_window_height();
};

#endif
#endif
