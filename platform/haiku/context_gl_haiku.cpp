#include "context_gl_haiku.h"

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)

ContextGL_Haiku::ContextGL_Haiku(HaikuDirectWindow** p_window, OS::VideoMode& p_default_video_mode) {
	video_mode = p_default_video_mode;
	
	uint32 type = BGL_RGB|BGL_DOUBLE|BGL_DEPTH;

	BRect windowRect;
	windowRect.Set(50, 50, 800, 600);

	window = new HaikuDirectWindow(windowRect);
	view = new HaikuGLView(window->Bounds(), type);

	*p_window = window;
}

ContextGL_Haiku::~ContextGL_Haiku() {
	delete view;
}

Error ContextGL_Haiku::initialize() {
	window->AddChild(view);
	view->LockGL();
	window->SetHaikuGLView(view);
	window->InitMessageRunner();
	window->Show();

	return OK;
}

void ContextGL_Haiku::release_current() {
	ERR_PRINT("release_current() NOT IMPLEMENTED");
}

void ContextGL_Haiku::make_current() {
	ERR_PRINT("make_current() NOT IMPLEMENTED");
}

void ContextGL_Haiku::swap_buffers() {
	view->SwapBuffers();
}

int ContextGL_Haiku::get_window_width() {
	// TODO: implement
	return 800;
}

int ContextGL_Haiku::get_window_height() {
	// TODO: implement
	return 600;
} 

#endif
