/*************************************************************************/
/*  godot_profiler_opengl.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_PROFILER_OPENGL_H
#define GODOT_PROFILER_OPENGL_H

#include "godot_profiler.h"

#ifdef ENABLE_PERFETTO

// Begin an OpenGL slice which gets automatically closed when going out of scope,
// which - if enabled - also makes sure that only and all OpenGL commands within the slice's scope are executed.
#ifdef PERFETTO_SYNC_OPENGL

#define PERFETTO_INTERNAL_SCOPED_TRACK_EVENT_OPENGL(category, name, ...)      \
  struct PERFETTO_UID(ScopedEvent) {                                          \
    struct EventFinalizer {                                                   \
      /* The parameter is an implementation detail. It allows the          */ \
      /* anonymous struct to use aggregate initialization to invoke the    */ \
      /* lambda (which emits the BEGIN event and returns an integer)       */ \
      /* with the proper reference capture for any                         */ \
      /* TrackEventArgumentFunction in |__VA_ARGS__|. This is required so  */ \
      /* that the scoped event is exactly ONE line and can't escape the    */ \
      /* scope if used in a single line if statement.                      */ \
      EventFinalizer(...) {}                                                  \
      ~EventFinalizer() { glFinish(); TRACE_EVENT_END(category); }            \
    } finalizer;                                                              \
  } PERFETTO_UID(scoped_event) {                                              \
    [&]() {                                                                   \
      glFinish();                                                             \
      TRACE_EVENT_BEGIN(category, name, ##__VA_ARGS__);                       \
      return 0;                                                               \
    }()                                                                       \
  }

#define TRACE_EVENT_OPENGL(category, name, ...) \
  PERFETTO_INTERNAL_SCOPED_TRACK_EVENT_OPENGL(category, name, "OpenGL synchronized", true, ##__VA_ARGS__)

#else

#define TRACE_EVENT_OPENGL(category, name, ...) \
  PERFETTO_INTERNAL_SCOPED_TRACK_EVENT(category, name, "OpenGL synchronized", false, ##__VA_ARGS__)

#endif

#else

#define TRACE_EVENT_OPENGL(...) ((void)0)

#endif

#endif
