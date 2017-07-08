/*************************************************************************/
/*  openhmd_shaders.cpp                                                  */
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

#include "openhmd_shader.h"

const char *const distortion_vert_330 =
		"#version 330\n"
		"\n"
		"layout (location=0) in vec2 coords;"
		"uniform mat4 mvp;"
		"out vec2 T;"
		"\n"
		"void main(void)\n"
		"{\n"
		"T = coords;\n"
		"gl_Position = mvp * vec4(coords, 0.0, 1.0);\n"
		"}";

const char *const distortion_frag_330 =
		"#version 330\n"
		"\n"
		"//per eye texture to warp for lens distortion\n"
		"uniform sampler2D warpTexture;\n"
		"\n"
		"//Position of lens center in m (usually eye_w/2, eye_h/2)\n"
		"uniform vec2 LensCenter;\n"
		"//Scale from texture co-ords to m (usually eye_w, eye_h)\n"
		"uniform vec2 ViewportScale;\n"
		"//Distortion overall scale in m (usually ~eye_w/2)\n"
		"uniform float WarpScale;\n"
		"//Distoriton coefficients (PanoTools model) [a,b,c,d]\n"
		"uniform vec4 HmdWarpParam;\n"
		"\n"
		"//chromatic distortion post scaling\n"
		"uniform vec3 aberr;\n"
		"\n"
		"in vec2 T;\n"
		"out vec4 color;\n"
		"\n"
		"void main()\n"
		"{\n"
		"//output_loc is the fragment location on screen from [0,1]x[0,1]\n"
		"vec2 output_loc = vec2(T.s, T.t);\n"
		"//Compute fragment location in lens-centered co-ordinates at world scale\n"
		"vec2 r = output_loc * ViewportScale - LensCenter;\n"
		"//scale for distortion model\n"
		"//distortion model has r=1 being the largest circle inscribed (e.g. eye_w/2)\n"
		"r /= WarpScale;\n"
		"\n"
		"//|r|**2\n"
		"float r_mag = length(r);\n"
		"//offset for which fragment is sourced\n"
		"vec2 r_displaced = r * (HmdWarpParam.w + HmdWarpParam.z * r_mag +\n"
		"HmdWarpParam.y * r_mag * r_mag +\n"
		"HmdWarpParam.x * r_mag * r_mag * r_mag);\n"
		"//back to world scale\n"
		"r_displaced *= WarpScale;\n"
		"//back to viewport co-ord\n"
		"vec2 tc_r = (LensCenter + aberr.r * r_displaced) / ViewportScale;\n"
		"vec2 tc_g = (LensCenter + aberr.g * r_displaced) / ViewportScale;\n"
		"vec2 tc_b = (LensCenter + aberr.b * r_displaced) / ViewportScale;\n"
		"\n"
		"float red = texture(warpTexture, tc_r).r;\n"
		"float green = texture(warpTexture, tc_g).g;\n"
		"float blue = texture(warpTexture, tc_b).b;\n"
		"//Black edges off the texture\n"
		"color = ((tc_g.x < 0.0) || (tc_g.x > 1.0) || (tc_g.y < 0.0) || (tc_g.y > 1.0)) ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(red, green, blue, 1.0);\n"
		"}";

float OpenHMDShader::vertice_data[12] = {
	0.0, 1.0,
	1.0, 0.0,
	0.0, 0.0,
	0.0, 1.0,
	1.0, 1.0,
	1.0, 0.0
};

void OpenHMDShader::compile_shader_src(GLuint shader, const char *src) {
	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);

	GLint status;
	GLint length;
	char log[4096] = { 0 };

	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	glGetShaderInfoLog(shader, 4096, &length, log);
	if (status == GL_FALSE) {
		printf("Compile failed %s\n", log);
	};
};

void OpenHMDShader::link_shader() {
	printf("Compiling OpenHMD shaders\n");

	// Create the handels
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	shader_program = glCreateProgram();

	// Attach the shaders to a program handel.
	glAttachShader(shader_program, vertexShader);
	glAttachShader(shader_program, fragmentShader);

	// Load and compile the Vertex Shader
	compile_shader_src(vertexShader, distortion_vert_330);

	// Load and compile the Fragment Shader
	compile_shader_src(fragmentShader, distortion_frag_330);

	// The shader objects are not needed any more,
	// the shader_program is the complete shader to be used.
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	printf("Linking OpenHMD shaders\n");
	glLinkProgram(shader_program);

	GLint status;
	GLint length;
	char log[4096] = { 0 };

	glGetProgramiv(shader_program, GL_LINK_STATUS, &status);
	glGetProgramInfoLog(shader_program, 4096, &length, log);
	if (status == GL_FALSE) {
		printf("Link failed %s\n", log);

		glDeleteProgram(shader_program);
		shader_program = 0;
		return;
	};

	// Get our uniforms, these don't change, more effective to load them once
	glUseProgram(shader_program);
	mvp_id = glGetUniformLocation(shader_program, "mvp");
	lens_center_id = glGetUniformLocation(shader_program, "LensCenter");

	// and set some properties that never change
	glUniform1i(glGetUniformLocation(shader_program, "warpTexture"), 0);
	glUseProgram(0);
};

void OpenHMDShader::set_device_parameters(ohmd_device *p_device) {
	if (shader_program != 0) {
		float viewport_scale[2];
		float distortion_coeffs[4];
		float aberr_scale[3];
		float sep;

		// get scale
		ohmd_device_getf(p_device, OHMD_SCREEN_HORIZONTAL_SIZE, &(viewport_scale[0]));
		viewport_scale[0] /= 2.0f;
		ohmd_device_getf(p_device, OHMD_SCREEN_VERTICAL_SIZE, &(viewport_scale[1]));

		// get distortion coefficients
		ohmd_device_getf(p_device, OHMD_UNIVERSAL_DISTORTION_K, &(distortion_coeffs[0]));
		ohmd_device_getf(p_device, OHMD_UNIVERSAL_ABERRATION_K, &(aberr_scale[0]));

		//calculate lens centers (assuming the eye separation is the distance betweenteh lense centers)
		ohmd_device_getf(p_device, OHMD_LENS_HORIZONTAL_SEPARATION, &sep);
		ohmd_device_getf(p_device, OHMD_LENS_VERTICAL_POSITION, &(lens_center[0][1]));
		ohmd_device_getf(p_device, OHMD_LENS_VERTICAL_POSITION, &(lens_center[1][1]));
		lens_center[0][0] = viewport_scale[0] - sep / 2.0f;
		lens_center[1][0] = sep / 2.0f;

		// assume calibration was for lens view to which ever edge of screen is further away from lens center
		float warp_scale = (lens_center[0][0] > lens_center[1][0]) ? lens_center[0][0] : lens_center[1][0];
		float warp_adj = 1.0f;

		// load these just once into our shader program, they do not change
		glUseProgram(shader_program);
		glUniform2f(glGetUniformLocation(shader_program, "ViewportScale"), viewport_scale[0], viewport_scale[1]);
		glUniform4f(glGetUniformLocation(shader_program, "HmdWarpParam"), distortion_coeffs[0], distortion_coeffs[1], distortion_coeffs[2], distortion_coeffs[3]);
		glUniform3f(glGetUniformLocation(shader_program, "aberr"), aberr_scale[0], aberr_scale[1], aberr_scale[2]);
		glUniform1f(glGetUniformLocation(shader_program, "WarpScale"), warp_scale * warp_adj);
		glUseProgram(0);
	};
};

void OpenHMDShader::render_eye(GLuint p_texture_id, int p_left_or_right_eye) {
	GLuint was_program;
	glGetIntegerv(GL_CURRENT_PROGRAM, (GLint *)&was_program);

	if (shader_program != 0) {
		float m[4][4];

		// calculate an MVP that covers the correct side of the window
		m[0][0] = 1.0;
		m[0][1] = 0.0;
		m[0][2] = 0.0;
		m[0][3] = 0.0;
		m[1][0] = 0.0;
		m[1][1] = 2.0;
		m[1][2] = 0.0;
		m[1][3] = 0.0;
		m[2][0] = 0.0;
		m[2][1] = 0.0;
		m[2][2] = 0.0;
		m[2][3] = 0.0;
		if (p_left_or_right_eye == 0) {
			m[3][0] = -1.0;
			m[3][1] = -1.0;
			m[3][2] = 0.0;
			m[3][3] = 1.0;
		} else {
			m[3][0] = 0.0;
			m[3][1] = -1.0;
			m[3][2] = 0.0;
			m[3][3] = 1.0;
		};

		// set our shader up
		glUseProgram(shader_program);
		glUniformMatrix4fv(mvp_id, 1, false, (const float *)m);
		glUniform2fv(lens_center_id, 1, lens_center[p_left_or_right_eye]);

		// set our texture
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, p_texture_id);

		// bind our vao to restore our state
		glBindVertexArray(vao);

		// render our rectangle
		glDrawArrays(GL_TRIANGLES, 0, 3 * 2);

		// and unbind
		glBindVertexArray(0);
		glUseProgram(was_program);
	}
};

OpenHMDShader::OpenHMDShader() {
	// Create our shader program
	link_shader();

	// Need a Vertex Array Object
	glGenVertexArrays(1, &vao);

	// Bind our VAO, all relevant state changes are bound to our VAO, will be unset when we unbind, and reset when we bind...
	glBindVertexArray(vao);

	// Need a Vertex Buffer Object
	glGenBuffers(1, &vbo);

	// Now bind our Vertex Buffer Object and load up some data!
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, vertice_data, GL_STATIC_DRAW);

	// And setup our attributes
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (GLvoid *)0);

	// and unbind our vao to return back to our old state
	glBindVertexArray(0);
};

OpenHMDShader::~OpenHMDShader() {
	if (shader_program != 0) {
		glDeleteProgram(shader_program);
		shader_program = 0;
	};

	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
};
