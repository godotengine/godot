/*************************************************************************/
/*  copy_effects.cpp                                                     */
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

#ifdef GLES3_ENABLED

#include "copy_effects.h"

using namespace GLES3;

CopyEffects *CopyEffects::singleton = nullptr;

CopyEffects *CopyEffects::get_singleton() {
	return singleton;
}

CopyEffects::CopyEffects() {
	singleton = this;

	copy.shader.initialize();
	copy.shader_version = copy.shader.version_create();
	copy.shader.version_bind_shader(copy.shader_version, CopyShaderGLES3::MODE_DEFAULT);

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

	{ // Screen Quad

		glGenBuffers(1, &quad);
		glBindBuffer(GL_ARRAY_BUFFER, quad);

		const float qv[12] = {
			-1.0f,
			-1.0f,
			1.0f,
			-1.0f,
			1.0f,
			1.0f,
			-1.0f,
			-1.0f,
			1.0f,
			1.0f,
			-1.0f,
			1.0f,
		};

		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, qv, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		glGenVertexArrays(1, &quad_array);
		glBindVertexArray(quad_array);
		glBindBuffer(GL_ARRAY_BUFFER, quad);
		glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
		glEnableVertexAttribArray(RS::ARRAY_VERTEX);
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	}
}

CopyEffects::~CopyEffects() {
	singleton = nullptr;
	glDeleteBuffers(1, &screen_triangle);
	glDeleteVertexArrays(1, &screen_triangle_array);
	glDeleteBuffers(1, &quad);
	glDeleteVertexArrays(1, &quad_array);
	copy.shader.version_free(copy.shader_version);
}

void CopyEffects::copy_to_rect(const Rect2 &p_rect) {
	bool success = copy.shader.version_bind_shader(copy.shader_version, CopyShaderGLES3::MODE_COPY_SECTION);
	if (!success) {
		return;
	}

	copy.shader.version_set_uniform(CopyShaderGLES3::COPY_SECTION, p_rect.position.x, p_rect.position.y, p_rect.size.x, p_rect.size.y, copy.shader_version, CopyShaderGLES3::MODE_COPY_SECTION);
	draw_screen_quad();
}

void CopyEffects::copy_screen() {
	bool success = copy.shader.version_bind_shader(copy.shader_version, CopyShaderGLES3::MODE_DEFAULT);
	if (!success) {
		return;
	}

	draw_screen_triangle();
}

void CopyEffects::bilinear_blur(GLuint p_source_texture, int p_mipmap_count, const Rect2i &p_region) {
	GLuint framebuffers[2];
	glGenFramebuffers(2, framebuffers);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[0]);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_source_texture, 0);

	Rect2i source_region = p_region;
	Rect2i dest_region = p_region;
	for (int i = 1; i < p_mipmap_count; i++) {
		dest_region.position.x >>= 1;
		dest_region.position.y >>= 1;
		dest_region.size.x = MAX(1, dest_region.size.x >> 1);
		dest_region.size.y = MAX(1, dest_region.size.y >> 1);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers[i % 2]);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_source_texture, i);
		glBlitFramebuffer(source_region.position.x, source_region.position.y, source_region.position.x + source_region.size.x, source_region.position.y + source_region.size.y,
				dest_region.position.x, dest_region.position.y, dest_region.position.x + dest_region.size.x, dest_region.position.y + dest_region.size.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffers[i % 2]);
		source_region = dest_region;
	}
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glDeleteFramebuffers(2, framebuffers);
}

void CopyEffects::set_color(const Color &p_color, const Rect2i &p_region) {
	bool success = copy.shader.version_bind_shader(copy.shader_version, CopyShaderGLES3::MODE_SIMPLE_COLOR);
	if (!success) {
		return;
	}

	copy.shader.version_set_uniform(CopyShaderGLES3::COPY_SECTION, p_region.position.x, p_region.position.y, p_region.size.x, p_region.size.y, copy.shader_version, CopyShaderGLES3::MODE_SIMPLE_COLOR);
	copy.shader.version_set_uniform(CopyShaderGLES3::COLOR_IN, p_color, copy.shader_version, CopyShaderGLES3::MODE_SIMPLE_COLOR);
	draw_screen_quad();
}

void CopyEffects::draw_screen_triangle() {
	glBindVertexArray(screen_triangle_array);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
}

void CopyEffects::draw_screen_quad() {
	glBindVertexArray(quad_array);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
}

#endif // GLES3_ENABLED
