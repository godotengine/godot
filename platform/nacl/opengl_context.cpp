/*************************************************************************/
/*  opengl_context.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "opengl_context.h"

#include <pthread.h>
#include "ppapi/gles2/gl2ext_ppapi.h"
#include "os_nacl.h"
#include "ppapi/cpp/instance.h"
#include "ppapi/cpp/module.h"
#include "ppapi/cpp/completion_callback.h"
#include "ppapi/utility/completion_callback_factory.h"


namespace {
// This is called by the brower when the 3D context has been flushed to the
// browser window.
void FlushCallback(void* data, int32_t result) {
  static_cast<OpenGLContext*>(data)->set_flush_pending(false);
  static_cast<OpenGLContext*>(data)->FlushContext();
}
}  // namespace

OpenGLContext::OpenGLContext(pp::Instance* p_instance)
	: pp::Graphics3DClient(p_instance),
      flush_pending_(false) {

  instance = p_instance;
  pp::Module* module = pp::Module::Get();
  assert(module);
  gles2_interface_ = static_cast<const struct PPB_OpenGLES2_Dev*>(
	  module->GetBrowserInterface(PPB_OPENGLES2_INTERFACE));
  assert(gles2_interface_);
}

OpenGLContext::~OpenGLContext() {
  glSetCurrentContextPPAPI(0);
}

bool OpenGLContext::MakeContextCurrent(pp::Instance* instance) {

  if (instance == NULL) {
    glSetCurrentContextPPAPI(0);
    return false;
  }
  // Lazily create the Pepper context.
  if (context_.is_null()) {
	int32_t attribs[] = {
		PP_GRAPHICS3DATTRIB_ALPHA_SIZE, 8,
		PP_GRAPHICS3DATTRIB_DEPTH_SIZE, 24,
		PP_GRAPHICS3DATTRIB_STENCIL_SIZE, 8,
		PP_GRAPHICS3DATTRIB_SAMPLES, 0,
		PP_GRAPHICS3DATTRIB_SAMPLE_BUFFERS, 0,
		PP_GRAPHICS3DATTRIB_WIDTH, width,
		PP_GRAPHICS3DATTRIB_HEIGHT, height,
		PP_GRAPHICS3DATTRIB_NONE
	};

	context_ = pp::Graphics3D(instance, pp::Graphics3D(), attribs);
	if (context_.is_null()) {
      glSetCurrentContextPPAPI(0);
      return false;
    }
	instance->BindGraphics(context_);
  }
  glSetCurrentContextPPAPI(context_.pp_resource());
  return true;
}

void OpenGLContext::ResizeContext(const pp::Size& size) {

	width = size.width();
	height = size.height();

	if (!context_.is_null()) {
		context_.ResizeBuffers(size.width(), size.height());
	}
}


void OpenGLContext::InvalidateContext(pp::Instance* instance) {
  glSetCurrentContextPPAPI(0);
}

void OpenGLContext::FlushContext() {
  if (flush_pending()) {
    // A flush is pending so do nothing; just drop this flush on the floor.
    return;
  }
  set_flush_pending(true);

	OSNacl* os = (OSNacl*)OS::get_singleton();
	MakeContextCurrent(instance);
	os->iterate();

  context_.SwapBuffers(pp::CompletionCallback(&FlushCallback, this));
}

