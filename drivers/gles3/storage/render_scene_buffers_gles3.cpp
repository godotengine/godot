/**************************************************************************/
/*  render_scene_buffers_gles3.cpp                                        */
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

#include "render_scene_buffers_gles3.h"
#include "config.h"
#include "texture_storage.h"
#include "utilities.h"

#ifdef ANDROID_ENABLED
#define glFramebufferTextureMultiviewOVR GLES3::Config::get_singleton()->eglFramebufferTextureMultiviewOVR
#define glTexStorage3DMultisample GLES3::Config::get_singleton()->eglTexStorage3DMultisample
#define glFramebufferTexture2DMultisampleEXT GLES3::Config::get_singleton()->eglFramebufferTexture2DMultisampleEXT
#define glFramebufferTextureMultisampleMultiviewOVR GLES3::Config::get_singleton()->eglFramebufferTextureMultisampleMultiviewOVR
#endif // ANDROID_ENABLED

// Will only be defined if GLES 3.2 headers are included
#ifndef GL_TEXTURE_2D_MULTISAMPLE_ARRAY
#define GL_TEXTURE_2D_MULTISAMPLE_ARRAY 0x9102
#endif

RenderSceneBuffersGLES3::~RenderSceneBuffersGLES3() {
	free_render_buffer_data();
}

void RenderSceneBuffersGLES3::_rt_attach_textures(GLuint p_color, GLuint p_depth, GLsizei p_samples, uint32_t p_view_count) {
	if (p_view_count > 1) {
		if (p_samples > 1) {
#if defined(ANDROID_ENABLED) || defined(WEB_ENABLED)
			glFramebufferTextureMultisampleMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, p_color, 0, p_samples, 0, p_view_count);
			glFramebufferTextureMultisampleMultiviewOVR(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, p_depth, 0, p_samples, 0, p_view_count);
#else
			ERR_PRINT_ONCE("Multiview MSAA isn't supported on this platform.");
#endif
		} else {
#ifndef IOS_ENABLED
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, p_color, 0, 0, p_view_count);
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, p_depth, 0, 0, p_view_count);
#else
			ERR_PRINT_ONCE("Multiview isn't supported on this platform.");
#endif
		}
	} else {
		if (p_samples > 1) {
#ifdef ANDROID_ENABLED
			glFramebufferTexture2DMultisampleEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_color, 0, p_samples);
			glFramebufferTexture2DMultisampleEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, p_depth, 0, p_samples);
#else
			ERR_PRINT_ONCE("MSAA via EXT_multisampled_render_to_texture isn't supported on this platform.");
#endif
		} else {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_color, 0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, p_depth, 0);
		}
	}
}

GLuint RenderSceneBuffersGLES3::_rt_get_cached_fbo(GLuint p_color, GLuint p_depth, GLsizei p_samples, uint32_t p_view_count) {
	FBDEF new_fbo;

#if defined(ANDROID_ENABLED) || defined(WEB_ENABLED)
	// There shouldn't be more then 3 entries in this...
	for (const FBDEF &cached_fbo : msaa3d.cached_fbos) {
		if (cached_fbo.color == p_color && cached_fbo.depth == p_depth) {
			return cached_fbo.fbo;
		}
	}

	new_fbo.color = p_color;
	new_fbo.depth = p_depth;

	glGenFramebuffers(1, &new_fbo.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, new_fbo.fbo);

	_rt_attach_textures(p_color, p_depth, p_samples, p_view_count);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		WARN_PRINT("Could not create 3D MSAA framebuffer, status: " + GLES3::TextureStorage::get_singleton()->get_framebuffer_error(status));

		glDeleteFramebuffers(1, &new_fbo.fbo);

		new_fbo.fbo = 0;
	} else {
		// cache it!
		msaa3d.cached_fbos.push_back(new_fbo);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif

	return new_fbo.fbo;
}

void RenderSceneBuffersGLES3::configure(const RenderSceneBuffersConfiguration *p_config) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLES3::Config *config = GLES3::Config::get_singleton();

	free_render_buffer_data();

	internal_size = p_config->get_internal_size();
	target_size = p_config->get_target_size();
	scaling_3d_mode = p_config->get_scaling_3d_mode();
	//fsr_sharpness = p_config->get_fsr_sharpness();
	//texture_mipmap_bias = p_config->get_texture_mipmap_bias();
	render_target = p_config->get_render_target();
	msaa3d.mode = p_config->get_msaa_3d();
	//screen_space_aa = p_config->get_screen_space_aa();
	//use_debanding = p_config->get_use_debanding();
	view_count = config->multiview_supported ? p_config->get_view_count() : 1;

	ERR_FAIL_COND(view_count == 0);
	bool use_multiview = view_count > 1;

	// Check our scaling mode
	if (scaling_3d_mode != RS::VIEWPORT_SCALING_3D_MODE_OFF && internal_size.x == 0 && internal_size.y == 0) {
		// Disable, no size set.
		scaling_3d_mode = RS::VIEWPORT_SCALING_3D_MODE_OFF;
	} else if (scaling_3d_mode != RS::VIEWPORT_SCALING_3D_MODE_OFF && internal_size == target_size) {
		// If size matches, we won't use scaling.
		scaling_3d_mode = RS::VIEWPORT_SCALING_3D_MODE_OFF;
	} else if (scaling_3d_mode != RS::VIEWPORT_SCALING_3D_MODE_OFF && scaling_3d_mode != RS::VIEWPORT_SCALING_3D_MODE_BILINEAR) {
		// We only support bilinear scaling atm.
		WARN_PRINT_ONCE("GLES only supports bilinear scaling.");
		scaling_3d_mode = RS::VIEWPORT_SCALING_3D_MODE_BILINEAR;
	}

	bool use_internal_buffer = scaling_3d_mode != RS::VIEWPORT_SCALING_3D_MODE_OFF; // TODO also need this if doing post processing like glow
	if (use_internal_buffer) {
		// Setup our internal buffer.
		bool is_transparent = texture_storage->render_target_get_transparent(render_target);
		GLuint color_internal_format = is_transparent ? GL_RGBA8 : GL_RGB10_A2;
		GLuint color_format = GL_RGBA;
		GLuint color_type = is_transparent ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT_2_10_10_10_REV;

		GLenum texture_target = use_multiview ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;

		// Create our color buffer.
		glGenTextures(1, &internal3d.color);
		glBindTexture(texture_target, internal3d.color);

		if (use_multiview) {
			glTexImage3D(texture_target, 0, color_internal_format, internal_size.x, internal_size.y, view_count, 0, color_format, color_type, nullptr);
		} else {
			glTexImage2D(texture_target, 0, color_internal_format, internal_size.x, internal_size.y, 0, color_format, color_type, nullptr);
		}

		glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLES3::Utilities::get_singleton()->texture_allocated_data(internal3d.color, internal_size.x * internal_size.y * view_count * 4, "3D color texture");

		// Create our depth buffer.
		glGenTextures(1, &internal3d.depth);
		glBindTexture(texture_target, internal3d.depth);

		if (use_multiview) {
			glTexImage3D(texture_target, 0, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, view_count, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
		} else {
			glTexImage2D(texture_target, 0, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
		}

		glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLES3::Utilities::get_singleton()->texture_allocated_data(internal3d.depth, internal_size.x * internal_size.y * view_count * 3, "3D depth texture");

		// Create our internal 3D FBO.
		// Note that if MSAA is used and our rt_msaa_* extensions are available, this is only used for blitting and effects.
		glGenFramebuffers(1, &internal3d.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, internal3d.fbo);

#ifndef IOS_ENABLED
		if (use_multiview) {
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, internal3d.color, 0, 0, view_count);
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, internal3d.depth, 0, 0, view_count);
		} else {
#else
		{
#endif
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_target, internal3d.color, 0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, texture_target, internal3d.depth, 0);
		}

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_clear_intermediate_buffers();
			WARN_PRINT("Could not create 3D buffers, status: " + texture_storage->get_framebuffer_error(status));
		}

		glBindTexture(texture_target, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	// Check if we support MSAA.
	if (msaa3d.mode != RS::VIEWPORT_MSAA_DISABLED && internal_size.x == 0 && internal_size.y == 0) {
		// Disable, no size set.
		msaa3d.mode = RS::VIEWPORT_MSAA_DISABLED;
	} else if (!use_multiview && msaa3d.mode != RS::VIEWPORT_MSAA_DISABLED && !config->msaa_supported && !config->rt_msaa_supported) {
		WARN_PRINT_ONCE("MSAA is not supported on this device.");
		msaa3d.mode = RS::VIEWPORT_MSAA_DISABLED;
	} else if (use_multiview && msaa3d.mode != RS::VIEWPORT_MSAA_DISABLED && !config->msaa_multiview_supported && !config->rt_msaa_multiview_supported) {
		WARN_PRINT_ONCE("Multiview MSAA is not supported on this device.");
		msaa3d.mode = RS::VIEWPORT_MSAA_DISABLED;
	}

	if (msaa3d.mode != RS::VIEWPORT_MSAA_DISABLED) {
		// Setup MSAA.
		const GLsizei samples[] = { 1, 2, 4, 8 };
		msaa3d.samples = samples[msaa3d.mode];

		// Constrain by limits of OpenGL driver.
		if (msaa3d.samples > config->msaa_max_samples) {
			msaa3d.samples = config->msaa_max_samples;
		}

		if (!use_multiview && !config->rt_msaa_supported) {
			// Render to texture extensions not supported? fall back to MSAA framebuffer through GL_EXT_framebuffer_multisample.
			// Note, if 2D MSAA matches 3D MSAA and we're not scaling, it would be ideal if we reuse our 2D MSAA buffer here.
			// We can't however because we don't trigger a change in configuration if 2D MSAA changes.
			// We'll accept the overhead in this situation.

			msaa3d.needs_resolve = true;
			msaa3d.check_fbo_cache = false;

			bool is_transparent = texture_storage->render_target_get_transparent(render_target);
			GLuint color_internal_format = is_transparent ? GL_RGBA8 : GL_RGB10_A2;

			// Create our color buffer.
			glGenRenderbuffers(1, &msaa3d.color);
			glBindRenderbuffer(GL_RENDERBUFFER, msaa3d.color);

			glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa3d.samples, color_internal_format, internal_size.x, internal_size.y);
			GLES3::Utilities::get_singleton()->render_buffer_allocated_data(msaa3d.color, internal_size.x * internal_size.y * view_count * 4 * msaa3d.samples, "MSAA 3D color render buffer");

			// Create our depth buffer.
			glGenRenderbuffers(1, &msaa3d.depth);
			glBindRenderbuffer(GL_RENDERBUFFER, msaa3d.depth);

			glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa3d.samples, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y);
			GLES3::Utilities::get_singleton()->render_buffer_allocated_data(msaa3d.depth, internal_size.x * internal_size.y * view_count * 3 * msaa3d.samples, "MSAA 3D depth render buffer");

			// Create our MSAA 3D FBO.
			glGenFramebuffers(1, &msaa3d.fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, msaa3d.fbo);

			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaa3d.color);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msaa3d.depth);

			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (status != GL_FRAMEBUFFER_COMPLETE) {
				_clear_msaa3d_buffers();
				WARN_PRINT("Could not create 3D MSAA buffers, status: " + texture_storage->get_framebuffer_error(status));
			}

			glBindRenderbuffer(GL_RENDERBUFFER, 0);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
#if !defined(IOS_ENABLED) && !defined(WEB_ENABLED)
		} else if (use_multiview && !config->rt_msaa_multiview_supported) {
			// Render to texture extensions not supported? fall back to MSAA textures through GL_EXT_multiview_texture_multisample.
			msaa3d.needs_resolve = true;
			msaa3d.check_fbo_cache = false;

			bool is_transparent = texture_storage->render_target_get_transparent(render_target);
			GLuint color_internal_format = is_transparent ? GL_RGBA8 : GL_RGB10_A2;

			// Create our color buffer.
			glGenTextures(1, &msaa3d.color);
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.color);

#ifdef ANDROID_ENABLED
			glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.samples, color_internal_format, internal_size.x, internal_size.y, view_count, GL_TRUE);
#else
			glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.samples, color_internal_format, internal_size.x, internal_size.y, view_count, GL_TRUE);
#endif

			GLES3::Utilities::get_singleton()->texture_allocated_data(msaa3d.color, internal_size.x * internal_size.y * view_count * 4 * msaa3d.samples, "MSAA 3D color texture");

			// Create our depth buffer.
			glGenTextures(1, &msaa3d.depth);
			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.depth);

#ifdef ANDROID_ENABLED
			glTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.samples, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, view_count, GL_TRUE);
#else
			glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, msaa3d.samples, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, view_count, GL_TRUE);
#endif

			GLES3::Utilities::get_singleton()->texture_allocated_data(msaa3d.depth, internal_size.x * internal_size.y * view_count * msaa3d.samples, "MSAA 3D depth texture");

			// Create our MSAA 3D FBO.
			glGenFramebuffers(1, &msaa3d.fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, msaa3d.fbo);

			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msaa3d.color, 0, 0, view_count);
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, msaa3d.depth, 0, 0, view_count);

			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (status != GL_FRAMEBUFFER_COMPLETE) {
				_clear_msaa3d_buffers();
				WARN_PRINT("Could not create 3D MSAA buffers, status: " + texture_storage->get_framebuffer_error(status));
			}

			glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
#if defined(ANDROID_ENABLED) || defined(WEB_ENABLED) // Only supported on OpenGLES!
		} else if (!use_internal_buffer) {
			// We are going to render directly into our render target textures,
			// these can change from frame to frame as we cycle through swapchains,
			// hence we'll use our FBO cache here.
			msaa3d.needs_resolve = false;
			msaa3d.check_fbo_cache = true;
		} else if (use_internal_buffer) {
			// We can combine MSAA and scaling/effects.
			msaa3d.needs_resolve = false;
			msaa3d.check_fbo_cache = false;

			// We render to our internal textures, MSAA is only done in tile memory only.
			// On mobile this means MSAA never leaves tile memory = efficiency!
			glGenFramebuffers(1, &msaa3d.fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, msaa3d.fbo);

			_rt_attach_textures(internal3d.color, internal3d.depth, msaa3d.samples, view_count);

			GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (status != GL_FRAMEBUFFER_COMPLETE) {
				_clear_msaa3d_buffers();
				WARN_PRINT("Could not create 3D MSAA framebuffer, status: " + texture_storage->get_framebuffer_error(status));
			}

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif
		} else {
			// HUH? how did we get here?
			WARN_PRINT_ONCE("MSAA is not supported on this device.");
			msaa3d.mode = RS::VIEWPORT_MSAA_DISABLED;
			msaa3d.samples = 1;
			msaa3d.check_fbo_cache = false;
		}
	} else {
		msaa3d.samples = 1;
		msaa3d.check_fbo_cache = false;
	}
}

void RenderSceneBuffersGLES3::_clear_msaa3d_buffers() {
	for (const FBDEF &cached_fbo : msaa3d.cached_fbos) {
		GLuint fbo = cached_fbo.fbo;
		glDeleteFramebuffers(1, &fbo);
	}
	msaa3d.cached_fbos.clear();

	if (msaa3d.fbo) {
		glDeleteFramebuffers(1, &msaa3d.fbo);
		msaa3d.fbo = 0;
	}

	if (msaa3d.color != 0) {
		if (view_count == 1) {
			GLES3::Utilities::get_singleton()->render_buffer_free_data(msaa3d.color);
		} else {
			GLES3::Utilities::get_singleton()->texture_free_data(msaa3d.color);
		}
		msaa3d.color = 0;
	}

	if (msaa3d.depth != 0) {
		if (view_count == 1) {
			GLES3::Utilities::get_singleton()->render_buffer_free_data(msaa3d.depth);
		} else {
			GLES3::Utilities::get_singleton()->texture_free_data(msaa3d.depth);
		}
		msaa3d.depth = 0;
	}
}

void RenderSceneBuffersGLES3::_clear_intermediate_buffers() {
	if (internal3d.fbo) {
		glDeleteFramebuffers(1, &internal3d.fbo);
		internal3d.fbo = 0;
	}

	if (internal3d.color != 0) {
		GLES3::Utilities::get_singleton()->texture_free_data(internal3d.color);
		internal3d.color = 0;
	}

	if (internal3d.depth != 0) {
		GLES3::Utilities::get_singleton()->texture_free_data(internal3d.depth);
		internal3d.depth = 0;
	}
}

void RenderSceneBuffersGLES3::check_backbuffer(bool p_need_color, bool p_need_depth) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();

	// Setup our back buffer

	if (backbuffer3d.fbo == 0) {
		glGenFramebuffers(1, &backbuffer3d.fbo);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, backbuffer3d.fbo);

	bool is_transparent = texture_storage->render_target_get_transparent(render_target);
	GLuint color_internal_format = is_transparent ? GL_RGBA8 : GL_RGB10_A2;
	GLuint color_format = GL_RGBA;
	GLuint color_type = is_transparent ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT_2_10_10_10_REV;

	bool use_multiview = view_count > 1 && GLES3::Config::get_singleton()->multiview_supported;
	GLenum texture_target = use_multiview ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;

	if (backbuffer3d.color == 0 && p_need_color) {
		glGenTextures(1, &backbuffer3d.color);
		glBindTexture(texture_target, backbuffer3d.color);

		if (use_multiview) {
			glTexImage3D(texture_target, 0, color_internal_format, internal_size.x, internal_size.y, view_count, 0, color_format, color_type, nullptr);
		} else {
			glTexImage2D(texture_target, 0, color_internal_format, internal_size.x, internal_size.y, 0, color_format, color_type, nullptr);
		}

		glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLES3::Utilities::get_singleton()->texture_allocated_data(backbuffer3d.color, internal_size.x * internal_size.y * view_count * 4, "3D Back buffer color texture");

#ifndef IOS_ENABLED
		if (use_multiview) {
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, backbuffer3d.color, 0, 0, view_count);
		} else {
#else
		{
#endif
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_target, backbuffer3d.color, 0);
		}
	}

	if (backbuffer3d.depth == 0 && p_need_depth) {
		glGenTextures(1, &backbuffer3d.depth);
		glBindTexture(texture_target, backbuffer3d.depth);

		if (use_multiview) {
			glTexImage3D(texture_target, 0, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, view_count, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
		} else {
			glTexImage2D(texture_target, 0, GL_DEPTH_COMPONENT24, internal_size.x, internal_size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
		}

		glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLES3::Utilities::get_singleton()->texture_allocated_data(backbuffer3d.depth, internal_size.x * internal_size.y * view_count * 3, "3D back buffer depth texture");

#ifndef IOS_ENABLED
		if (use_multiview) {
			glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, backbuffer3d.depth, 0, 0, view_count);
		} else {
#else
		{
#endif
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, texture_target, backbuffer3d.depth, 0);
		}
	}

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) {
		_clear_back_buffers();
		WARN_PRINT("Could not create 3D back buffers, status: " + texture_storage->get_framebuffer_error(status));
	}

	glBindTexture(texture_target, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderSceneBuffersGLES3::_clear_back_buffers() {
	if (backbuffer3d.fbo) {
		glDeleteFramebuffers(1, &backbuffer3d.fbo);
		backbuffer3d.fbo = 0;
	}

	if (backbuffer3d.color != 0) {
		GLES3::Utilities::get_singleton()->texture_free_data(backbuffer3d.color);
		backbuffer3d.color = 0;
	}

	if (backbuffer3d.depth != 0) {
		GLES3::Utilities::get_singleton()->texture_free_data(backbuffer3d.depth);
		backbuffer3d.depth = 0;
	}
}

void RenderSceneBuffersGLES3::free_render_buffer_data() {
	_clear_msaa3d_buffers();
	_clear_intermediate_buffers();
	_clear_back_buffers();
}

GLuint RenderSceneBuffersGLES3::get_render_fbo() {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	GLuint rt_fbo = 0;

	if (msaa3d.check_fbo_cache) {
		GLuint color = texture_storage->render_target_get_color(render_target);
		GLuint depth = texture_storage->render_target_get_depth(render_target);

		rt_fbo = _rt_get_cached_fbo(color, depth, msaa3d.samples, view_count);
	} else if (msaa3d.fbo != 0) {
		// We have an MSAA fbo, render to our MSAA buffer
		return msaa3d.fbo;
	} else if (internal3d.fbo != 0) {
		// We have an internal buffer, render to our internal buffer!
		return internal3d.fbo;
	} else {
		rt_fbo = texture_storage->render_target_get_fbo(render_target);
	}

	if (texture_storage->render_target_is_reattach_textures(render_target)) {
		GLuint color = texture_storage->render_target_get_color(render_target);
		GLuint depth = texture_storage->render_target_get_depth(render_target);

		glBindFramebuffer(GL_FRAMEBUFFER, rt_fbo);
		_rt_attach_textures(color, depth, msaa3d.samples, view_count);
		glBindFramebuffer(GL_FRAMEBUFFER, texture_storage->system_fbo);
	}

	return rt_fbo;
}

#endif // GLES3_ENABLED
