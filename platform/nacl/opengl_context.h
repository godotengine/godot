/*************************************************************************/
/*  opengl_context.h                                                     */
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
#ifndef EXAMPLES_TUMBLER_OPENGL_CONTEXT_H_
#define EXAMPLES_TUMBLER_OPENGL_CONTEXT_H_

///
/// @file
/// OpenGLContext manages the OpenGL context in the browser that is associated
/// with a @a pp::Instance instance.
///

#include <pthread.h>

#include <algorithm>
#include <string>

#include "ppapi/c/ppb_opengles2.h"
//#include "ppapi/cpp/dev/context_3d_dev.h"
#include "ppapi/cpp/graphics_3d_client.h"
#include "ppapi/cpp/graphics_3d.h"
//#include "ppapi/cpp/dev/surface_3d_dev.h"
#include "ppapi/cpp/instance.h"
#include "ppapi/cpp/size.h"

// A convenience wrapper for a shared OpenGLContext pointer type.  As other
// smart pointer types are needed, add them here.

#include <tr1/memory>

class OpenGLContext;

typedef std::tr1::shared_ptr<OpenGLContext> SharedOpenGLContext;

/// OpenGLContext manages an OpenGL rendering context in the browser.
///
class OpenGLContext : public pp::Graphics3DClient {
 public:
  explicit OpenGLContext(pp::Instance* instance);

  /// Release all the in-browser resources used by this context, and make this
  /// context invalid.
  virtual ~OpenGLContext();

  /// The Graphics3DClient interfcace.
  virtual void Graphics3DContextLost() {
    assert(!"Unexpectedly lost graphics context");
  }

  /// Make @a this the current 3D context in @a instance.
  /// @param instance The instance of the NaCl module that will receive the
  ///                 the current 3D context.
  /// @return success.
  bool MakeContextCurrent(pp::Instance* instance);

  /// Flush the contents of this context to the browser's 3D device.
  void FlushContext();

  /// Make the underlying 3D device invalid, so that any subsequent rendering
  /// commands will have no effect.  The next call to MakeContextCurrent() will
  /// cause the underlying 3D device to get rebound and start receiving
  /// receiving rendering commands again.  Use InvalidateContext(), for
  /// example, when resizing the context's viewing area.
  void InvalidateContext(pp::Instance* instance);

  void ResizeContext(const pp::Size& size);

  /// The OpenGL ES 2.0 interface.
  const struct PPB_OpenGLES2_Dev* gles2() const {
    return gles2_interface_;
  }

  /// The PP_Resource needed to make GLES2 calls through the Pepper interface.
  const PP_Resource gl_context() const {
    return context_.pp_resource();
  }

  /// Indicate whether a flush is pending.  This can only be called from the
  /// main thread; it is not thread safe.
  bool flush_pending() const {
    return flush_pending_;
  }
  void set_flush_pending(bool flag) {
    flush_pending_ = flag;
  }

 private:
  pp::Graphics3D context_;
  bool flush_pending_;

  int width, height;

  pp::Instance* instance;

  const struct PPB_OpenGLES2_Dev* gles2_interface_;
};

#endif  // EXAMPLES_TUMBLER_OPENGL_CONTEXT_H_

