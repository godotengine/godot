/**************************************************************************/
/*  glow.cpp                                                              */
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

#include "glow.h"
#include "../storage/texture_storage.h"

using namespace GLES3;

Glow *Glow::singleton = nullptr;

Glow *Glow::get_singleton() {
	return singleton;
}

Glow::Glow() {
	singleton = this;

	glow.shader.initialize();
	glow.shader_version = glow.shader.version_create();

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

Glow::~Glow() {
	glDeleteBuffers(1, &screen_triangle);
	glDeleteVertexArrays(1, &screen_triangle_array);

	glow.shader.version_free(glow.shader_version);

	singleton = nullptr;
}

void Glow::_draw_screen_triangle() {
	glBindVertexArray(screen_triangle_array);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
}

void Glow::process_glow(GLuint p_source_color, Size2i p_size, const Glow::GLOWLEVEL *p_glow_buffers, uint32_t p_view, bool p_use_multiview) {
	ERR_FAIL_COND(p_source_color == 0);
	ERR_FAIL_COND(p_glow_buffers[3].color == 0);

	// Reset some OpenGL state...
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	// Start with our filter pass
	{
		glBindFramebuffer(GL_FRAMEBUFFER, p_glow_buffers[0].fbo);
		glViewport(0, 0, p_glow_buffers[0].size.x, p_glow_buffers[0].size.y);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(p_use_multiview ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D, p_source_color);

		uint64_t specialization = p_use_multiview ? GlowShaderGLES3::USE_MULTIVIEW : 0;
		bool success = glow.shader.version_bind_shader(glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		if (!success) {
			return;
		}

		glow.shader.version_set_uniform(GlowShaderGLES3::PIXEL_SIZE, 1.0 / p_glow_buffers[0].size.x, 1.0 / p_glow_buffers[0].size.y, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::VIEW, float(p_view), glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::LUMINANCE_MULTIPLIER, luminance_multiplier, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::GLOW_BLOOM, glow_bloom, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::GLOW_HDR_THRESHOLD, glow_hdr_bleed_threshold, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::GLOW_HDR_SCALE, glow_hdr_bleed_scale, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);
		glow.shader.version_set_uniform(GlowShaderGLES3::GLOW_LUMINANCE_CAP, glow_hdr_luminance_cap, glow.shader_version, GlowShaderGLES3::MODE_FILTER, specialization);

		_draw_screen_triangle();
	}

	// Continue with downsampling
	{
		bool success = glow.shader.version_bind_shader(glow.shader_version, GlowShaderGLES3::MODE_DOWNSAMPLE, 0);
		if (!success) {
			return;
		}

		for (int i = 1; i < 4; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, p_glow_buffers[i].fbo);
			glViewport(0, 0, p_glow_buffers[i].size.x, p_glow_buffers[i].size.y);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, p_glow_buffers[i - 1].color);

			glow.shader.version_set_uniform(GlowShaderGLES3::PIXEL_SIZE, 1.0 / p_glow_buffers[i].size.x, 1.0 / p_glow_buffers[i].size.y, glow.shader_version, GlowShaderGLES3::MODE_DOWNSAMPLE);

			_draw_screen_triangle();
		}
	}

	// Now upsample
	{
		bool success = glow.shader.version_bind_shader(glow.shader_version, GlowShaderGLES3::MODE_UPSAMPLE, 0);
		if (!success) {
			return;
		}

		for (int i = 2; i >= 0; i--) {
			glBindFramebuffer(GL_FRAMEBUFFER, p_glow_buffers[i].fbo);
			glViewport(0, 0, p_glow_buffers[i].size.x, p_glow_buffers[i].size.y);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, p_glow_buffers[i + 1].color);

			glow.shader.version_set_uniform(GlowShaderGLES3::PIXEL_SIZE, 1.0 / p_glow_buffers[i].size.x, 1.0 / p_glow_buffers[i].size.y, glow.shader_version, GlowShaderGLES3::MODE_UPSAMPLE);

			_draw_screen_triangle();
		}
	}

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
}

#endif // GLES3_ENABLED
