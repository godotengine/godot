/*************************************************************************/
/*  config.cpp                                                           */
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

#include "config.h"
#include "core/config/project_settings.h"
#include "core/templates/vector.h"

using namespace GLES3;

#define _GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF

Config *Config::singleton = nullptr;

Config::Config() {
	singleton = this;

	{
		int max_extensions = 0;
		glGetIntegerv(GL_NUM_EXTENSIONS, &max_extensions);
		for (int i = 0; i < max_extensions; i++) {
			const GLubyte *s = glGetStringi(GL_EXTENSIONS, i);
			if (!s) {
				break;
			}
			extensions.insert((const char *)s);
		}
	}

	keep_original_textures = true; // false
	depth_internalformat = GL_DEPTH_COMPONENT;
	depth_type = GL_UNSIGNED_INT;

	srgb_decode_supported = extensions.has("GL_EXT_texture_sRGB_decode");
	etc2_supported = true;
#ifdef GLES_OVER_GL
	float_texture_supported = true;
	s3tc_supported = true;
	etc_supported = false; // extensions.has("GL_OES_compressed_ETC1_RGB8_texture");
	bptc_supported = extensions.has("GL_ARB_texture_compression_bptc") || extensions.has("EXT_texture_compression_bptc");
	rgtc_supported = extensions.has("GL_EXT_texture_compression_rgtc") || extensions.has("GL_ARB_texture_compression_rgtc") || extensions.has("EXT_texture_compression_rgtc");
	support_npot_repeat_mipmap = true;
	depth_buffer_internalformat = GL_DEPTH_COMPONENT24;
#else
	float_texture_supported = extensions.has("GL_ARB_texture_float") || extensions.has("GL_OES_texture_float");
	s3tc_supported = extensions.has("GL_EXT_texture_compression_s3tc") || extensions.has("WEBGL_compressed_texture_s3tc");
	etc_supported = extensions.has("GL_OES_compressed_ETC1_RGB8_texture") || extensions.has("WEBGL_compressed_texture_etc1");
	bptc_supported = false;
	rgtc_supported = false;
	support_npot_repeat_mipmap = extensions.has("GL_OES_texture_npot");

#ifdef JAVASCRIPT_ENABLED
	// RenderBuffer internal format must be 16 bits in WebGL,
	// but depth_texture should default to 32 always
	// if the implementation doesn't support 32, it should just quietly use 16 instead
	// https://www.khronos.org/registry/webgl/extensions/WEBGL_depth_texture/
	depth_buffer_internalformat = GL_DEPTH_COMPONENT16;
	depth_type = GL_UNSIGNED_INT;
#else
	// on mobile check for 24 bit depth support for RenderBufferStorage
	if (extensions.has("GL_OES_depth24")) {
		depth_buffer_internalformat = _DEPTH_COMPONENT24_OES;
		depth_type = GL_UNSIGNED_INT;
	} else {
		depth_buffer_internalformat = GL_DEPTH_COMPONENT16;
		depth_type = GL_UNSIGNED_SHORT;
	}
#endif
#endif

#ifdef GLES_OVER_GL
	use_rgba_2d_shadows = false;
	use_rgba_3d_shadows = false;
	support_depth_cubemaps = true;
#else
	use_rgba_2d_shadows = !(float_texture_supported && extensions.has("GL_EXT_texture_rg"));
	use_rgba_3d_shadows = false;
	support_depth_cubemaps = extensions.has("GL_OES_depth_texture_cube_map");
#endif

#ifdef GLES_OVER_GL
	support_32_bits_indices = true;
#else
	support_32_bits_indices = extensions.has("GL_OES_element_index_uint");
#endif

#ifdef GLES_OVER_GL
	support_write_depth = true;
#elif defined(JAVASCRIPT_ENABLED)
	support_write_depth = false;
#else
	support_write_depth = extensions.has("GL_EXT_frag_depth");
#endif

	//picky requirements for these
	support_shadow_cubemaps = support_write_depth && support_depth_cubemaps;
	// the use skeleton software path should be used if either float texture is not supported,
	// OR max_vertex_texture_image_units is zero
	use_skeleton_software = (float_texture_supported == false) || (max_vertex_texture_image_units == 0);

	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &max_vertex_texture_image_units);
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
	glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &max_uniform_buffer_size);

	support_anisotropic_filter = extensions.has("GL_EXT_texture_filter_anisotropic");
	if (support_anisotropic_filter) {
		glGetFloatv(_GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &anisotropic_level);
		anisotropic_level = MIN(float(1 << int(ProjectSettings::get_singleton()->get("rendering/textures/default_filters/anisotropic_filtering_level"))), anisotropic_level);
	}

	force_vertex_shading = false; //GLOBAL_GET("rendering/quality/shading/force_vertex_shading");
	use_nearest_mip_filter = GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter");

	use_depth_prepass = bool(GLOBAL_GET("rendering/driver/depth_prepass/enable"));
	if (use_depth_prepass) {
		String vendors = GLOBAL_GET("rendering/driver/depth_prepass/disable_for_vendors");
		Vector<String> vendor_match = vendors.split(",");
		String renderer = (const char *)glGetString(GL_RENDERER);
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
