/*************************************************************************/
/*  ctxgl_procaddr.cpp                                                   */
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
#ifdef OPENGL_ENABLED
#include "ctxgl_procaddr.h"
#include <GL/gl.h>
#include <stdio.h>

static PROC _gl_procs[] = {
	(PROC)glCullFace,
	(PROC)glFrontFace,
	(PROC)glHint,
	(PROC)glLineWidth,
	(PROC)glPointSize,
	(PROC)glPolygonMode,
	(PROC)glScissor,
	(PROC)glTexParameterf,
	(PROC)glTexParameterfv,
	(PROC)glTexParameteri,
	(PROC)glTexParameteriv,
	(PROC)glTexImage1D,
	(PROC)glTexImage2D,
	(PROC)glDrawBuffer,
	(PROC)glClear,
	(PROC)glClearColor,
	(PROC)glClearStencil,
	(PROC)glClearDepth,
	(PROC)glStencilMask,
	(PROC)glColorMask,
	(PROC)glDepthMask,
	(PROC)glDisable,
	(PROC)glEnable,
	(PROC)glFinish,
	(PROC)glFlush,
	(PROC)glBlendFunc,
	(PROC)glLogicOp,
	(PROC)glStencilFunc,
	(PROC)glStencilOp,
	(PROC)glDepthFunc,
	(PROC)glPixelStoref,
	(PROC)glPixelStorei,
	(PROC)glReadBuffer,
	(PROC)glReadPixels,
	(PROC)glGetBooleanv,
	(PROC)glGetDoublev,
	(PROC)glGetError,
	(PROC)glGetFloatv,
	(PROC)glGetIntegerv,
	(PROC)glGetString,
	(PROC)glGetTexImage,
	(PROC)glGetTexParameterfv,
	(PROC)glGetTexParameteriv,
	(PROC)glGetTexLevelParameterfv,
	(PROC)glGetTexLevelParameteriv,
	(PROC)glIsEnabled,
	(PROC)glDepthRange,
	(PROC)glViewport,
	/* not detected in ATI */
	(PROC)glDrawArrays,
	(PROC)glDrawElements,
	(PROC)glGetPointerv,
	(PROC)glPolygonOffset,
	(PROC)glCopyTexImage1D,
	(PROC)glCopyTexImage2D,
	(PROC)glCopyTexSubImage1D,
	(PROC)glCopyTexSubImage2D,
	(PROC)glTexSubImage1D,
	(PROC)glTexSubImage2D,
	(PROC)glBindTexture,
	(PROC)glDeleteTextures,
	(PROC)glGenTextures,
	(PROC)glIsTexture,

	0
};

static const char *_gl_proc_names[] = {
	"glCullFace",
	"glFrontFace",
	"glHint",
	"glLineWidth",
	"glPointSize",
	"glPolygonMode",
	"glScissor",
	"glTexParameterf",
	"glTexParameterfv",
	"glTexParameteri",
	"glTexParameteriv",
	"glTexImage1D",
	"glTexImage2D",
	"glDrawBuffer",
	"glClear",
	"glClearColor",
	"glClearStencil",
	"glClearDepth",
	"glStencilMask",
	"glColorMask",
	"glDepthMask",
	"glDisable",
	"glEnable",
	"glFinish",
	"glFlush",
	"glBlendFunc",
	"glLogicOp",
	"glStencilFunc",
	"glStencilOp",
	"glDepthFunc",
	"glPixelStoref",
	"glPixelStorei",
	"glReadBuffer",
	"glReadPixels",
	"glGetBooleanv",
	"glGetDoublev",
	"glGetError",
	"glGetFloatv",
	"glGetIntegerv",
	"glGetString",
	"glGetTexImage",
	"glGetTexParameterfv",
	"glGetTexParameteriv",
	"glGetTexLevelParameterfv",
	"glGetTexLevelParameteriv",
	"glIsEnabled",
	"glDepthRange",
	"glViewport",
	/* not detected in ati */
	"glDrawArrays",
	"glDrawElements",
	"glGetPointerv",
	"glPolygonOffset",
	"glCopyTexImage1D",
	"glCopyTexImage2D",
	"glCopyTexSubImage1D",
	"glCopyTexSubImage2D",
	"glTexSubImage1D",
	"glTexSubImage2D",
	"glBindTexture",
	"glDeleteTextures",
	"glGenTextures",
	"glIsTexture",

	0
};

PROC get_gl_proc_address(const char *p_address) {

	PROC proc = wglGetProcAddress((const CHAR *)p_address);
	if (!proc) {

		int i = 0;
		while (_gl_procs[i]) {

			if (strcmp(p_address, _gl_proc_names[i]) == 0) {
				return _gl_procs[i];
			}
			i++;
		}
	}
	return proc;
}
#endif
