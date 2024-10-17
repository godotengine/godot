/**************************************************************************/
/*  feed_effects.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef GLES3_ENABLED

#include "feed_effects.h"

#ifdef ANDROID_ENABLED
#include <GLES3/gl3ext.h>
#endif

#define GL_PROGRAM_POINT_SIZE 0x8642

using namespace GLES3;

FeedEffects *FeedEffects::singleton = nullptr;

FeedEffects *FeedEffects::get_singleton() {
	return singleton;
}

FeedEffects::FeedEffects() {
	singleton = this;

	feed.shader.initialize();
	feed.shader_version = feed.shader.version_create();
	feed.shader.version_bind_shader(feed.shader_version, FeedShaderGLES3::MODE_DEFAULT);

	{ // Screen Triangle.
		glGenBuffers(1, &screen_triangle);
		glBindBuffer(GL_ARRAY_BUFFER, screen_triangle);

		const float qv[6] = {
			-1.0f,
			-1.0f,
			3.0f,
			-1.0f,
			-1.0f,
			3.0f,
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, qv, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &screen_triangle_array);
		glBindVertexArray(screen_triangle_array);
		glBindBuffer(GL_ARRAY_BUFFER, screen_triangle);
		glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
		glEnableVertexAttribArray(RS::ARRAY_VERTEX);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}
}

FeedEffects::~FeedEffects() {
	singleton = nullptr;
	glDeleteBuffers(1, &screen_triangle);
	glDeleteVertexArrays(1, &screen_triangle_array);
	feed.shader.version_free(feed.shader_version);
}

Transform3D transform3D_from_mat4(const float *p_mat4) {
	Transform3D res;

	res.basis.rows[0][0] = p_mat4[0];
	res.basis.rows[1][0] = p_mat4[1];
	res.basis.rows[2][0] = p_mat4[2];
	// p_mat4[3] = 0;
	res.basis.rows[0][1] = p_mat4[4];
	res.basis.rows[1][1] = p_mat4[5];
	res.basis.rows[2][1] = p_mat4[6];
	// p_mat4[7] = 0;
	res.basis.rows[0][2] = p_mat4[8];
	res.basis.rows[1][2] = p_mat4[9];
	res.basis.rows[2][2] = p_mat4[10];
	// p_mat4[11] = 0;
	res.origin.x = p_mat4[12];
	res.origin.y = p_mat4[13];
	res.origin.z = p_mat4[14];
	// p_mat4[15] = 1;

	return res;
}

void FeedEffects::draw() {
	bool success = feed.shader.version_bind_shader(feed.shader_version, FeedShaderGLES3::MODE_DEFAULT, FeedShaderGLES3::USE_EXTERNAL_SAMPLER);
	if (!success) {
		OS::get_singleton()->print("Godot : FeedShaderGLES3 Could not bind version_bind_shader USE_EXTERNAL_SAMPLER");
		return;
	}

	draw_screen_triangle();
}

void FeedEffects::draw_screen_triangle() {
	glBindVertexArray(screen_triangle_array);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
}

#endif // GLES3_ENABLED
