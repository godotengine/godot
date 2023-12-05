/**************************************************************************/
/*  config.cpp                                                            */
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

#include "config.h"

#include "../rasterizer_gles3.h"
#include "texture_storage.h"

using namespace GLES3;

#define _GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

Config *Config::singleton = nullptr;

Config::Config() {
	singleton = this;

	{
		int64_t max_extensions = 0;
		glGetInteger64v(GL_NUM_EXTENSIONS, &max_extensions);
		for (int64_t i = 0; i < max_extensions; i++) {
			const GLubyte *s = glGetStringi(GL_EXTENSIONS, i);
			if (!s) {
				break;
			}
			extensions.insert((const char *)s);
		}
	}

	bptc_supported = extensions.has("GL_ARB_texture_compression_bptc") || extensions.has("EXT_texture_compression_bptc");
	astc_supported = extensions.has("GL_KHR_texture_compression_astc") || extensions.has("GL_OES_texture_compression_astc") || extensions.has("GL_KHR_texture_compression_astc_ldr") || extensions.has("GL_KHR_texture_compression_astc_hdr");
	astc_hdr_supported = extensions.has("GL_KHR_texture_compression_astc_ldr");
	astc_layered_supported = extensions.has("GL_KHR_texture_compression_astc_sliced_3d");

	if (RasterizerGLES3::is_gles_over_gl()) {
		float_texture_supported = true;
		etc2_supported = false;
		s3tc_supported = true;
		rgtc_supported = true; //RGTC - core since OpenGL version 3.0
	} else {
		float_texture_supported = extensions.has("GL_EXT_color_buffer_float");
		etc2_supported = true;
#if defined(ANDROID_ENABLED) || defined(IOS_ENABLED)
		// Some Android devices report support for S3TC but we don't expect that and don't export the textures.
		// This could be fixed but so few devices support it that it doesn't seem useful (and makes bigger APKs).
		// For good measure we do the same hack for iOS, just in case.
		s3tc_supported = false;
#else
		s3tc_supported = extensions.has("GL_EXT_texture_compression_dxt1") || extensions.has("GL_EXT_texture_compression_s3tc") || extensions.has("WEBGL_compressed_texture_s3tc");
#endif
		rgtc_supported = extensions.has("GL_EXT_texture_compression_rgtc") || extensions.has("GL_ARB_texture_compression_rgtc") || extensions.has("EXT_texture_compression_rgtc");
	}

	glGetInteger64v(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &max_vertex_texture_image_units);
	glGetInteger64v(GL_MAX_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
	glGetInteger64v(GL_MAX_TEXTURE_SIZE, &max_texture_size);
	glGetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, &max_uniform_buffer_size);
	glGetInteger64v(GL_MAX_VIEWPORT_DIMS, max_viewport_size);

	support_anisotropic_filter = extensions.has("GL_EXT_texture_filter_anisotropic");
	if (support_anisotropic_filter) {
		glGetFloatv(_GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropic_level);
		anisotropic_level = MIN(float(1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"))), anisotropic_level);
	}

	multiview_supported = extensions.has("GL_OVR_multiview2") || extensions.has("GL_OVR_multiview");
#ifdef ANDROID_ENABLED
	if (multiview_supported) {
		eglFramebufferTextureMultiviewOVR = (PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC)eglGetProcAddress("glFramebufferTextureMultiviewOVR");
		if (eglFramebufferTextureMultiviewOVR == nullptr) {
			multiview_supported = false;
		}
	}
#endif

	force_vertex_shading = false; //GLOBAL_GET("rendering/quality/shading/force_vertex_shading");
	use_nearest_mip_filter = GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter");

	use_depth_prepass = bool(GLOBAL_GET("rendering/driver/depth_prepass/enable"));
	if (use_depth_prepass) {
		String vendors = GLOBAL_GET("rendering/driver/depth_prepass/disable_for_vendors");
		Vector<String> vendor_match = vendors.split(",");
		const String &renderer = String::utf8((const char *)glGetString(GL_RENDERER));
		for (int i = 0; i < vendor_match.size(); i++) {
			String v = vendor_match[i].strip_edges();
			if (v == String()) {
				continue;
			}

			if (renderer.findn(v) != -1) {
				use_depth_prepass = false;
			}
		}
	}

	max_renderable_elements = GLOBAL_GET("rendering/limits/opengl/max_renderable_elements");
	max_renderable_lights = GLOBAL_GET("rendering/limits/opengl/max_renderable_lights");
	max_lights_per_object = GLOBAL_GET("rendering/limits/opengl/max_lights_per_object");
}

Config::~Config() {
	singleton = nullptr;
}

#endif // GLES3_ENABLED
