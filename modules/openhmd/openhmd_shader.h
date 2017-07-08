/*************************************************************************/
/*  openhmd_shaders.h                                                    */
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

/**
  I'm just parking this code here for now, this compiles and uses the shader
  that is needed for the lens distortion.

  There is a big question whether, and how much, of this code should be in
  OpenHMD itself or how much of it we should piggy back onto Godots shader
  logic. 
**/

#ifndef OPENHMD_SHADER_H
#define OPENHMD_SHADER_H

// include this for now...
#include "platform_config.h"
#ifndef GLES3_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include GLES3_INCLUDE_H
#endif

#include <openhmd.h>
#include <stdio.h>

class OpenHMDShader {
private:
	static float vertice_data[12];

	GLuint shader_program;
	GLuint vao;
	GLuint vbo;

	GLint mvp_id;
	GLint lens_center_id;
	float lens_center[2][2];

	void compile_shader_src(GLuint shader, const char *src);
	void link_shader();

public:
	/* set shader properties */
	void set_device_parameters(ohmd_device *p_device);

	/* render one of our eye textures to screen */
	void render_eye(GLuint p_texture_id, int p_left_or_right_eye);

	OpenHMDShader();
	~OpenHMDShader();
};

#endif /* OPENHMD_SHADER_H */