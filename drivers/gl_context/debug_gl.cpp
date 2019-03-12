/*************************************************************************/
/*  debug_gl.cpp														 */
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

#include "debug_gl.h"
#include "core/print_string.h"

DebugGLRegions *DebugGL::regions = NULL;

DebugGLMarkers *DebugGL::markers = NULL;

DebugGLLabels *DebugGL::labels = NULL;

void DebugGL::initialize() {

	initialize_markers_and_regions();
	initialize_labels();
	initialize_logging();
}

void DebugGL::initialize_logging() {

	// Luckily, the three different types of debug output extensions
	// all use the same prototypes
	PFNGLDEBUGMESSAGECALLBACKPROC install_callback;
	PFNGLDEBUGMESSAGECONTROLPROC set_log_level;

	if (GLAD_GL_ES_VERSION_3_2) {
		install_callback = glDebugMessageCallback;
		set_log_level = glDebugMessageControl;
	} else if (GLAD_GL_KHR_debug) {
		install_callback = glDebugMessageCallbackKHR;
		set_log_level = glDebugMessageControlKHR;
	} else if (GLAD_GL_ARB_debug_output) {
		install_callback = glDebugMessageCallbackARB;
		set_log_level = glDebugMessageControlARB;
	} else {
		print_verbose("Cannot install OpenGL debug output callback since no supported extension is available.");
		return;
	}

	install_callback((GLDEBUGPROC)gl_debug_print, NULL);

	// Disable everything first
	set_log_level(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_FALSE);

	// Then enable a selection of events
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PORTABILITY, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	set_log_level(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);

	print_line("godot: ENABLING GL DEBUG");
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glEnable(GL_DEBUG_OUTPUT);
}

/**
 * Implement GL debug regions using the official Khronos extension.
 */
class DebugGLRegionsKHR : public DebugGLRegions {
public:
	DebugGLRegionsKHR();

	void push(const char *msg) override;
	void pop() override;

private:
	int max_message_length = 0;
	int max_stack_depth = 0;
	int current_stack_depth = 0;

	// This is glDebugMessageInsert or the equivalent extension functions
	PFNGLPUSHDEBUGGROUPPROC push_debug_group;
	PFNGLPOPDEBUGGROUPPROC pop_debug_group;
};

DebugGLRegionsKHR::DebugGLRegionsKHR() {
	if (GLAD_GL_ES_VERSION_3_2) {
		push_debug_group = glPushDebugGroup;
		pop_debug_group = glPopDebugGroup;
		glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH, &max_message_length);
		glGetIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH, &max_stack_depth);

	} else if (GLAD_GL_KHR_debug) {
		push_debug_group = glPushDebugGroupKHR;
		pop_debug_group = glPopDebugGroupKHR;
		glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH_KHR, &max_message_length);
		glGetIntegerv(GL_MAX_DEBUG_GROUP_STACK_DEPTH_KHR, &max_stack_depth);

	} else {

		CRASH_NOW(); // Reaching this indicates a bug
	}
}

void DebugGLRegionsKHR::push(const char *msg) {
	if (++current_stack_depth <= max_stack_depth) {
		int msg_len = (int)strlen(msg);
		if (msg_len + 1 > max_message_length) {
			msg_len = max_message_length - 1;
		}
		push_debug_group(
				GL_DEBUG_SOURCE_APPLICATION,
				0,
				msg_len,
				msg);
	}
}

void DebugGLRegionsKHR::pop() {
	if (current_stack_depth-- > max_stack_depth) {
		return; // We previously exceeded the max stack depth on the driver side
	}
	pop_debug_group();
}

/**
 * Implement GL markers using the official Khronos extension.
 */
class DebugGLMarkersKHR : public DebugGLMarkers {
public:
	DebugGLMarkersKHR();

	void insert(const char *message) override;

private:
	// Maximum length of a message supported by the driver, including null byte
	int max_message_length = 0;

	// This is glDebugMessageInsert or the equivalent extension functions
	PFNGLDEBUGMESSAGEINSERTPROC insert_message;
};

DebugGLMarkersKHR::DebugGLMarkersKHR() {
	if (GLAD_GL_ES_VERSION_3_2) {
		insert_message = glDebugMessageInsert;
		glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH, &max_message_length);

	} else if (GLAD_GL_KHR_debug) {
		insert_message = glDebugMessageInsertKHR;
		glGetIntegerv(GL_MAX_DEBUG_MESSAGE_LENGTH_KHR, &max_message_length);

	} else {

		CRASH_NOW(); // Reaching this indicates a bug
	}
}

void DebugGLMarkersKHR::insert(const char *message) {

	int len = strlen(message);
	if (len + 1 > max_message_length) {
		len = max_message_length - 1;
	}

	if (len > 0) {
		insert_message(
				GL_DEBUG_SOURCE_APPLICATION,
				GL_DEBUG_TYPE_MARKER,
				1,
				GL_DEBUG_SEVERITY_NOTIFICATION,
				len,
				message);
	}
}

/**
 * Implement GL markers using the Apple specific extension.
 */
class DebugGLMarkersApple : public DebugGLMarkers {
public:
	void insert(const char *message) override;
};

void DebugGLMarkersApple::insert(const char *message) {
	glInsertEventMarkerEXT(0, message);
}

/**
 * Implement GL regions using the Apple specific extension.
 */
class DebugGLRegionsApple : public DebugGLRegions {
public:
	void push(const char *msg) override;
	void pop() override;
};

void DebugGLRegionsApple::push(const char *msg) {
	glPushGroupMarkerEXT(0, msg);
}

void DebugGLRegionsApple::pop() {
	glPopGroupMarkerEXT();
}

void DebugGL::initialize_markers_and_regions() {

	if (markers) {
		memdelete(markers);
		markers = NULL;
	}

	if (regions) {
		memdelete(regions);
		regions = NULL;
	}

	if (GLAD_GL_ES_VERSION_3_2 || GLAD_GL_KHR_debug) {
		markers = memnew(DebugGLMarkersKHR);
		regions = memnew(DebugGLRegionsKHR);
	} else if (GLAD_GL_EXT_debug_marker) {
		markers = memnew(DebugGLMarkersApple);
		regions = memnew(DebugGLRegionsApple);
	}
}

/**
 * Implement object labeling using the official Khronos extension.
 */
class DebugGLLabelsKHR : public DebugGLLabels {
public:
	DebugGLLabelsKHR();

	void label_buffer(GLuint p_id, const char *p_label) override {
		label_object(GL_BUFFER, p_id, get_label_length(p_label), p_label);
	}
	void label_shader(GLuint p_id, const char *p_label) override {
		label_object(GL_SHADER, p_id, get_label_length(p_label), p_label);
	}
	void label_program(GLuint p_id, const char *p_label) override {
		label_object(GL_PROGRAM, p_id, get_label_length(p_label), p_label);
	}
	void label_vertex_array(GLuint p_id, const char *p_label) override {
		label_object(GL_VERTEX_ARRAY, p_id, get_label_length(p_label), p_label);
	}
	void label_transform_feedback(GLuint p_id, const char *p_label) override {
		label_object(GL_TRANSFORM_FEEDBACK, p_id, get_label_length(p_label), p_label);
	}
	void label_sampler(GLuint p_id, const char *p_label) override {
		label_object(GL_SAMPLER, p_id, get_label_length(p_label), p_label);
	}
	void label_texture(GLuint p_id, const char *p_label) override {
		label_object(GL_TEXTURE, p_id, get_label_length(p_label), p_label);
	}
	void label_renderbuffer(GLuint p_id, const char *p_label) override {
		label_object(GL_RENDERBUFFER, p_id, get_label_length(p_label), p_label);
	}
	void label_framebuffer(GLuint p_id, const char *p_label) override {
		label_object(GL_FRAMEBUFFER, p_id, get_label_length(p_label), p_label);
	}

private:
	// Maximum length of a label supported by the driver, including null byte
	int max_label_length = 0;

	// This is glObjectLabel or the equivalent extension functions
	PFNGLOBJECTLABELPROC label_object;

	int get_label_length(const char *p_label) const {
		int len = strlen(p_label);
		if (len < max_label_length) {
			return len;
		} else {
			return max_label_length - 1;
		}
	}
};

DebugGLLabelsKHR::DebugGLLabelsKHR() {
	if (GLAD_GL_ES_VERSION_3_2) {
		label_object = glObjectLabel;
		glGetIntegerv(GL_MAX_LABEL_LENGTH, &max_label_length);

	} else if (GLAD_GL_KHR_debug) {
		label_object = glObjectLabelKHR;
		glGetIntegerv(GL_MAX_LABEL_LENGTH_KHR, &max_label_length);

	} else {

		CRASH_NOW(); // Reaching this indicates a bug
	}
}

/**
 * Implement object labeling using the Apple extension.
 */
class DebugGLLabelsApple : public DebugGLLabels {
public:
	void label_buffer(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_BUFFER_OBJECT_EXT, p_id, 0, p_label);
	}
	void label_shader(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_SHADER_OBJECT_EXT, p_id, 0, p_label);
	}
	void label_program(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_PROGRAM_OBJECT_EXT, p_id, 0, p_label);
	}
	void label_vertex_array(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_VERTEX_ARRAY_OBJECT_EXT, p_id, 0, p_label);
	}
	void label_transform_feedback(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_TRANSFORM_FEEDBACK, p_id, 0, p_label);
	}
	void label_sampler(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_SAMPLER, p_id, 0, p_label);
	}
	void label_texture(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_TEXTURE, p_id, 0, p_label);
	}
	void label_renderbuffer(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_RENDERBUFFER, p_id, 0, p_label);
	}
	void label_framebuffer(GLuint p_id, const char *p_label) override {
		glLabelObjectEXT(GL_FRAMEBUFFER, p_id, 0, p_label);
	}
};

void DebugGL::initialize_labels() {

	if (labels) {
		memdelete(labels);
		labels = NULL;
	}

	if (GLAD_GL_ES_VERSION_3_2 || GLAD_GL_KHR_debug) {
		labels = memnew(DebugGLLabelsKHR);
	} else if (GLAD_GL_EXT_debug_label) {
		labels = memnew(DebugGLLabelsApple);
	}
}

static const char *get_source_string(GLenum source) {
	switch (source) {
		case GL_DEBUG_SOURCE_API:
			return "OpenGL";
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			return "Window System";
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			return "Shader Compiler";
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			return "Third Party";
		case GL_DEBUG_SOURCE_APPLICATION:
			return "Application";
		case GL_DEBUG_SOURCE_OTHER:
			return "Other";
		default:
			return "Unknown";
	}
}

static const char *get_type_string(GLenum type) {
	switch (type) {
		case GL_DEBUG_TYPE_ERROR:
			return "Error";
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			return "Deprecated behavior";
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			return "Undefined behavior";
		case GL_DEBUG_TYPE_PORTABILITY:
			return "Portability";
		case GL_DEBUG_TYPE_PERFORMANCE:
			return "Performance";
		case GL_DEBUG_TYPE_OTHER:
			return "Other";
		case GL_DEBUG_TYPE_MARKER:
			return "Marker";
		case GL_DEBUG_TYPE_PUSH_GROUP:
			return "PushGroup";
		case GL_DEBUG_TYPE_POP_GROUP:
			return "PopGroup";
		default:
			return "Unknown";
	}
}

static const char *get_severity_string(GLenum type) {
	switch (type) {
		case GL_DEBUG_SEVERITY_HIGH:
			return "High";
		case GL_DEBUG_SEVERITY_MEDIUM:
			return "Medium";
		case GL_DEBUG_SEVERITY_LOW:
			return "Low";
		case GL_DEBUG_SEVERITY_NOTIFICATION:
			return "Notification";
		default:
			return "Unknown";
	}
}

void DebugGL::gl_debug_print(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const GLvoid *user_param) {
	const char *source_str = get_source_string(source);
	const char *type_str = get_type_string(type);
	const char *severity_str = get_severity_string(severity);

	String output = String() + "GL ERROR: Source: " + source_str + "\tType: " + type_str + "\tID: " + itos(id) + "\tSeverity: " + severity_str + "\tMessage: " + message;
	if (severity == GL_DEBUG_SEVERITY_HIGH) {
		ERR_PRINTS(output);
	} else {
		print_verbose(output);
	}
}
