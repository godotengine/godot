/**************************************************************************/
/*  post_effects.cpp                                                      */
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

#include "post_effects.h"
#include "../storage/texture_storage.h"

using namespace GLES3;

PostEffects *PostEffects::singleton = nullptr;

PostEffects *PostEffects::get_singleton() {
	return singleton;
}

PostEffects::PostEffects() {
	singleton = this;

	post.shader.initialize();
	post.shader_version = post.shader.version_create();
	post.shader.version_bind_shader(post.shader_version, PostShaderGLES3::MODE_DEFAULT);

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

PostEffects::~PostEffects() {
	singleton = nullptr;
	glDeleteBuffers(1, &screen_triangle);
	glDeleteVertexArrays(1, &screen_triangle_array);
	post.shader.version_free(post.shader_version);
}

void PostEffects::_draw_screen_triangle() {
	glBindVertexArray(screen_triangle_array);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
}

void PostEffects::post_copy(
		GLuint p_dest_framebuffer, Size2i p_dest_size, GLuint p_source_color,
		GLuint p_source_depth, bool p_ssao_enabled, int p_ssao_quality_level, float p_ssao_strength, float p_ssao_radius,
		Size2i p_source_size, float p_luminance_multiplier, const Glow::GLOWLEVEL *p_glow_buffers, float p_glow_intensity,
		float p_srgb_white, uint32_t p_view, bool p_use_multiview, uint64_t p_spec_constants, bool p_filter) {
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	glDisable(GL_BLEND);

	glBindFramebuffer(GL_FRAMEBUFFER, p_dest_framebuffer);
	glViewport(0, 0, p_dest_size.x, p_dest_size.y);

	PostShaderGLES3::ShaderVariant mode = PostShaderGLES3::MODE_DEFAULT;
	uint64_t flags = p_spec_constants;
	if (p_use_multiview) {
		flags |= PostShaderGLES3::USE_MULTIVIEW;
	}
	if (p_glow_buffers != nullptr) {
		flags |= PostShaderGLES3::USE_GLOW;
	}
	if (p_ssao_enabled) {
		if (p_ssao_quality_level == RS::ENV_SSAO_QUALITY_VERY_LOW) {
			flags |= PostShaderGLES3::USE_SSAO_ABYSS;
		} else if (p_ssao_quality_level == RS::ENV_SSAO_QUALITY_LOW) {
			flags |= PostShaderGLES3::USE_SSAO_LOW;
		} else if (p_ssao_quality_level == RS::ENV_SSAO_QUALITY_HIGH) {
			flags |= PostShaderGLES3::USE_SSAO_HIGH;
		} else if (p_ssao_quality_level == RS::ENV_SSAO_QUALITY_ULTRA) {
			flags |= PostShaderGLES3::USE_SSAO_MEGA;
		} else {
			flags |= PostShaderGLES3::USE_SSAO_MED;
		}
	}
	if (p_luminance_multiplier != 1.0) {
		flags |= PostShaderGLES3::USE_LUMINANCE_MULTIPLIER;
	}

	bool success = post.shader.version_bind_shader(post.shader_version, mode, flags);
	if (!success) {
		return;
	}

	GLenum texture_target = p_use_multiview ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture_target, p_source_color);

	glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, p_filter ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, p_filter ? GL_LINEAR : GL_NEAREST);

	if (p_ssao_enabled) {
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(texture_target, p_source_depth);
		glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // Thanks to mrjustaguy!
		glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		post.shader.version_set_uniform(PostShaderGLES3::SSAO_INTENSITY, p_ssao_strength, post.shader_version, mode, flags);
		post.shader.version_set_uniform(PostShaderGLES3::SSAO_RADIUS_FRAC, p_ssao_radius, post.shader_version, mode, flags);
		post.shader.version_set_uniform(PostShaderGLES3::SSAO_PRN_UV, // This converts the UV coordinate into a pseudo-random number.
				p_source_size.x * 1.087f * ((1.0f + sqrt(5.0f)) / 2.0f),
				p_source_size.y * 1.087f * ((9.0f + sqrt(221.0f)) / 10.0f),
				post.shader_version, mode, flags);
	}

	if (p_glow_buffers != nullptr) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, p_glow_buffers[0].color);

		post.shader.version_set_uniform(PostShaderGLES3::PIXEL_SIZE, 1.0 / p_source_size.x, 1.0 / p_source_size.y, post.shader_version, mode, flags);
		post.shader.version_set_uniform(PostShaderGLES3::GLOW_INTENSITY, p_glow_intensity, post.shader_version, mode, flags);
		post.shader.version_set_uniform(PostShaderGLES3::SRGB_WHITE, p_srgb_white, post.shader_version, mode, flags);
	}

	post.shader.version_set_uniform(PostShaderGLES3::VIEW, float(p_view), post.shader_version, mode, flags);
	post.shader.version_set_uniform(PostShaderGLES3::LUMINANCE_MULTIPLIER, p_luminance_multiplier, post.shader_version, mode, flags);

	_draw_screen_triangle();

	// Reset state
	if (p_glow_buffers != nullptr) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	if (p_ssao_enabled) {
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(texture_target, 0);
	}

	// Return back to nearest
	glActiveTexture(GL_TEXTURE0);
	glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(texture_target, 0);

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
}

#endif // GLES3_ENABLED
