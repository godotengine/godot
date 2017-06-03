/*************************************************************************/
/*  context_gl_haiku.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "context_gl_haiku.h"

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)

ContextGL_Haiku::ContextGL_Haiku(HaikuDirectWindow *p_window) {
	window = p_window;

	uint32 type = BGL_RGB | BGL_DOUBLE | BGL_DEPTH;
	view = new HaikuGLView(window->Bounds(), type);

	use_vsync = false;
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
	view->SwapBuffers(use_vsync);
}

int ContextGL_Haiku::get_window_width() {
	return window->Bounds().IntegerWidth();
}

int ContextGL_Haiku::get_window_height() {
	return window->Bounds().IntegerHeight();
}

void ContextGL_Haiku::set_use_vsync(bool p_use) {
	use_vsync = p_use;
}

bool ContextGL_Haiku::is_using_vsync() const {
	return use_vsync;
}

#endif
