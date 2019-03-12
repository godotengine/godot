/*************************************************************************/
/*  debug_gl.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef DEBUGGL_H
#define DEBUGGL_H

#include "debug_gl.h"

#include "core/ustring.h"

#include <glad/glad.h>

class DebugGLRegions;
class DebugGLMarkers;
class DebugGLLabels;

// Abstraction for the various OpenGL debug extensions, which allow:
// - Capturing OpenGL driver debug output
// - Setting human readable names for various OpenGL resources
// - Inserting debug text markers in the command stream
// - Labeling regions in the OpenGL command stream
//
// Extension Specs:
//  (1) https://www.khronos.org/registry/OpenGL/extensions/KHR/KHR_debug.txt
//  (2) https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_debug_output.txt
//  (3) https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_debug_marker.txt
//  (4) https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_debug_label.txt
//
// Feature support:
//   |    Extension      |  Debug Output  |  Object Labels  |  Markers  |  Regions |
//   |-------------------|----------------|-----------------|-----------|----------|
//   | KHR_debug         |  X             |  X              |  X        |  X       |
//   | ARB_debug_output  |  X             |                 |           |          |
//   | EXT_debug_marker  |                |                 |  X        |  X       |
//   | EXT_debug_label   |                |  X              |           |          |
//
// If KHR_debug is supported, none of the other extensions will be used.
// ARB_debug_output is for really old drivers, while EXT_debug_marker / EXT_debug_label
// seem to be exclusively supported by Apple drivers (i.e. on iOS/OSX).
class DebugGL {
	friend struct DebugScopeGL;

public:
	static void initialize();

	/**
	 * Inserts a labeled marker into the OpenGL command stream. The given message
	 * might be truncated if it exceeds the length supported by the OpenGL driver.
	 */
	static inline void insert_marker(const char *message);

	static void insert_marker(const String &message) {
		insert_marker(message.utf8().get_data());
	}

	static inline void push_region(const char *message);
	static void push_region(const String &message) {
		push_region(message.utf8().get_data());
	}
	static inline void pop_region();

	static void label_buffer(GLuint p_id, const String &p_label) {
		label_buffer(p_id, p_label.utf8().get_data());
	}
	static inline void label_buffer(GLuint p_id, const char *p_label);

	void label_shader(GLuint p_id, const String &p_label) {
		label_shader(p_id, p_label.utf8().get_data());
	}
	static inline void label_shader(GLuint p_id, const char *p_label);

	void label_program(GLuint p_id, const String &p_label) {
		label_program(p_id, p_label.utf8().get_data());
	}
	static inline void label_program(GLuint p_id, const char *p_label);

	void label_vertex_array(GLuint p_id, const String &p_label) {
		label_vertex_array(p_id, p_label.utf8().get_data());
	}
	static inline void label_vertex_array(GLuint p_id, const char *p_label);

	void label_transform_feedback(GLuint p_id, const String &p_label) {
		label_transform_feedback(p_id, p_label.utf8().get_data());
	}
	static inline void label_transform_feedback(GLuint p_id, const char *p_label);

	void label_sampler(GLuint p_id, const String &p_label) {
		label_sampler(p_id, p_label.utf8().get_data());
	}
	static inline void label_sampler(GLuint p_id, const char *p_label);

	void label_texture(GLuint p_id, const String &p_label) {
		label_texture(p_id, p_label.utf8().get_data());
	}
	static inline void label_texture(GLuint p_id, const char *p_label);

	void label_renderbuffer(GLuint p_id, const String &p_label) {
		label_renderbuffer(p_id, p_label.utf8().get_data());
	}
	static inline void label_renderbuffer(GLuint p_id, const char *p_label);

	void label_framebuffer(GLuint p_id, const String &p_label) {
		label_framebuffer(p_id, p_label.utf8().get_data());
	}
	static inline void label_framebuffer(GLuint p_id, const char *p_label);

private:
	static DebugGLRegions *regions;

	static DebugGLLabels *labels;

	static DebugGLMarkers *markers;

	static void initialize_logging();
	static void initialize_markers_and_regions();
	static void initialize_labels();

	// This will forward to the method below using user_param as "this".
	static void GLAPIENTRY gl_debug_print(GLenum source,
			GLenum type,
			GLuint id,
			GLenum severity,
			GLsizei length,
			const GLchar *message,
			const GLvoid *user_param);
};

/**
 * Regions are used to mark an area of the OpenGL command stream with a human-readable
 * name, which eases debugging. These regions are visible in a flamegraph-like manner
 * when using frame debuggers such as NVidia Nsight or Renderdoc.
 */
class DebugGLRegions {
public:
	virtual ~DebugGLRegions() = default;

	virtual void push(const char *msg) = 0;
	virtual void pop() = 0;
};

/**
 * Markers can be inserted to represent events during the OpenGL command stream.
 * They should be visible in frame debuggers and make debugging easier.
 */
class DebugGLMarkers {
public:
	virtual ~DebugGLMarkers() = default;

	virtual void insert(const char *message) = 0;
};

/**
 * Object labeling can be used to give human-readable names to resources on the
 * OpenGL driver side. This makes it easier to discern which resources are currently 
 * bound to the OpenGL state machine.
 */
class DebugGLLabels {
public:
	virtual ~DebugGLLabels() = default;

	virtual void label_buffer(GLuint p_id, const char *p_label) = 0;
	virtual void label_shader(GLuint p_id, const char *p_label) = 0;
	virtual void label_program(GLuint p_id, const char *p_label) = 0;
	virtual void label_vertex_array(GLuint p_id, const char *p_label) = 0;
	virtual void label_transform_feedback(GLuint p_id, const char *p_label) = 0;
	virtual void label_sampler(GLuint p_id, const char *p_label) = 0;
	virtual void label_texture(GLuint p_id, const char *p_label) = 0;
	virtual void label_renderbuffer(GLuint p_id, const char *p_label) = 0;
	virtual void label_framebuffer(GLuint p_id, const char *p_label) = 0;
};

/**
 * Pushes a debug region and pops it using RAII.
 */
struct DebugGLRegionScope {
	DebugGLRegionScope(const char *p_msg) {
		DebugGL::push_region(p_msg);
	}

	DebugGLRegionScope(const String &p_msg) {
		DebugGL::push_region(p_msg);
	}

	~DebugGLRegionScope() {
		DebugGL::pop_region();
	}
};

inline void DebugGL::insert_marker(const char *message) {
	if (unlikely(markers)) {
		markers->insert(message);
	}
}

inline void DebugGL::push_region(const char *message) {
	if (unlikely(regions)) {
		regions->push(message);
	}
}

inline void DebugGL::pop_region() {
	if (unlikely(regions)) {
		regions->pop();
	}
}

inline void DebugGL::label_buffer(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_buffer(p_id, p_label);
	}
}

inline void DebugGL::label_shader(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_shader(p_id, p_label);
	}
}

inline void DebugGL::label_program(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_program(p_id, p_label);
	}
}

inline void DebugGL::label_vertex_array(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_vertex_array(p_id, p_label);
	}
}

inline void DebugGL::label_transform_feedback(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_transform_feedback(p_id, p_label);
	}
}

inline void DebugGL::label_sampler(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_sampler(p_id, p_label);
	}
}

inline void DebugGL::label_texture(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_texture(p_id, p_label);
	}
}

inline void DebugGL::label_renderbuffer(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_renderbuffer(p_id, p_label);
	}
}

inline void DebugGL::label_framebuffer(GLuint p_id, const char *p_label) {
	if (unlikely(labels)) {
		labels->label_framebuffer(p_id, p_label);
	}
}

// This macro magic is necessary to generate somewhat unique variable names
#define _DEBUG_GL_CONCAT2(a, b) a##b
#define _DEBUG_GL_CONCAT(a, b) _DEBUG_GL_CONCAT2(a, b)

// Assign a name to the current C++ scope in the OpenGL command stream and
// automatically remove it when the C++ scope ends
#define DEBUG_GL_REGION(m_name) DebugGLRegionScope _DEBUG_GL_CONCAT(_debug_scope_, __LINE__)(m_name);

#endif // DEBUGGL_H
