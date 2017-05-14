/*************************************************************************/
/*  gl_error_handler.cpp                                                 */
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

#include <stdio.h>

#include "platform_config.h"

#ifndef GLES3_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include GLES3_INCLUDE_H
#endif

#include "gl_error_handler.h"

const char *get_gl_error_desc(GLenum p_status) {
	switch (p_status) {
		case GL_NO_ERROR:
			return "GL_NO_ERROR";
		case GL_INVALID_ENUM:
			return "GL_INVALID_ENUM";
		case GL_INVALID_VALUE:
			return "GL_INVALID_VALUE";
		case GL_INVALID_OPERATION:
			return "GL_INVALID_OPERATION";
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			return "GL_INVALID_FRAMEBUFFER_OPERATION";
		case GL_OUT_OF_MEMORY:
			return "GL_OUT_OF_MEMORY";
#ifdef GL_STACK_UNDERFLOW
		case GL_STACK_UNDERFLOW:
			return "GL_STACK_UNDERFLOW";
#endif
#ifdef GL_STACK_OVERFLOW
		case GL_STACK_OVERFLOW:
			return "GL_STACK_OVERFLOW";
#endif
		default:
			static char unknownstr[250];
			sprintf(unknownstr, "%i", p_status);
			return unknownstr;
	};
};

bool _err_print_gl_errors(const char *p_function, const char *p_file, int p_line, const char *p_command) {
	GLenum glerr = glGetError();
	if (glerr != GL_NO_ERROR) {
		char error_text[1024];

		sprintf(error_text, "GLError %s in %s \n", get_gl_error_desc(glerr), p_command);
		_err_print_error(p_function, p_file, p_line, error_text);

    return true;
  } else {
    return false;
	};
};

const char *get_gl_fb_error_desc(GLenum p_status) {
	switch (p_status) {
		case GL_FRAMEBUFFER_COMPLETE:
			return "GL_FRAMEBUFFER_COMPLETE";
		case GL_FRAMEBUFFER_UNDEFINED:
			return "GL_FRAMEBUFFER_UNDEFINED";
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
#ifdef GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER";
#endif
#ifdef GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER";
#endif
		case GL_FRAMEBUFFER_UNSUPPORTED:
			return "GL_FRAMEBUFFER_UNSUPPORTED";
		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE";
#ifdef GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS";
#endif
		default:
			static char unknownstr[250];
			sprintf(unknownstr, "%i", p_status);
			return unknownstr;
	};
};

void _err_print_gl_fb_error(const char *p_function, const char *p_file, int p_line, GLenum p_status, const char *p_framebuffer_desc) {
	char error_text[1024];
	sprintf(error_text, "Error creating %s: %s\n", p_framebuffer_desc, get_gl_fb_error_desc(p_status));
	_err_print_error(p_function, p_file, p_line, error_text);
};
