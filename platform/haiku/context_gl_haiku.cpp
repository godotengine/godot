#include "context_gl_haiku.h"

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)

ContextGL_Haiku::ContextGL_Haiku(HaikuDirectWindow* p_window) {
	window = p_window;

	uint32 type = BGL_RGB | BGL_DOUBLE | BGL_DEPTH;
	view = new HaikuGLView(window->Bounds(), type);
}

ContextGL_Haiku::~ContextGL_Haiku() {
	delete view;
}

Error ContextGL_Haiku::initialize() {
	window->AddChild(view);
	window->SetHaikuGLView(view);

	return OK;
}

void ContextGL_Haiku::release_current() {
	view->UnlockGL();
}

void ContextGL_Haiku::make_current() {
	view->LockGL();
}

void ContextGL_Haiku::swap_buffers() {
	view->SwapBuffers();
}

int ContextGL_Haiku::get_window_width() {
	return window->Bounds().IntegerWidth();
}

int ContextGL_Haiku::get_window_height() {
	return window->Bounds().IntegerHeight();
} 

#endif
