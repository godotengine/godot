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
#include "core/templates/vector.h"

using namespace GLES3;

Config *Config::singleton = nullptr;

Config::Config() {
	singleton = this;

	{
		const GLubyte *extension_string = glGetString(GL_EXTENSIONS);

		Vector<String> exts = String((const char *)extension_string).split(" ");

		for (int i = 0; i < exts.size(); i++) {
			extensions.insert(exts[i]);
		}
	}

	keep_original_textures = true; // false
	shrink_textures_x2 = false;
	depth_internalformat = GL_DEPTH_COMPONENT;
	depth_type = GL_UNSIGNED_INT;

#ifdef GLES_OVER_GL
	float_texture_supported = true;
	s3tc_supported = true;
	etc_supported = false;
	support_npot_repeat_mipmap = true;
	depth_buffer_internalformat = GL_DEPTH_COMPONENT24;
#else
	float_texture_supported = extensions.has("GL_ARB_texture_float") || extensions.has("GL_OES_texture_float");
	s3tc_supported = extensions.has("GL_EXT_texture_compression_s3tc") || extensions.has("WEBGL_compressed_texture_s3tc");
	etc_supported = extensions.has("GL_OES_compressed_ETC1_RGB8_texture") || extensions.has("WEBGL_compressed_texture_etc1");
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
	//TODO: causes huge problems with desktop video drivers. Making false for now, needs to be true to render SCREEN_TEXTURE mipmaps
	render_to_mipmap_supported = false;
#else
	//check if mipmaps can be used for SCREEN_TEXTURE and Glow on Mobile and web platforms
	render_to_mipmap_supported = extensions.has("GL_OES_fbo_render_mipmap") && extensions.has("GL_EXT_texture_lod");
#endif

#ifdef GLES_OVER_GL
	use_rgba_2d_shadows = false;
	support_depth_texture = true;
	use_rgba_3d_shadows = false;
	support_depth_cubemaps = true;
#else
	use_rgba_2d_shadows = !(float_texture_supported && extensions.has("GL_EXT_texture_rg"));
	support_depth_texture = extensions.has("GL_OES_depth_texture") || extensions.has("WEBGL_depth_texture");
	use_rgba_3d_shadows = !support_depth_texture;
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

	support_half_float_vertices = true;
//every platform should support this except web, iOS has issues with their support, so add option to disable
#ifdef JAVASCRIPT_ENABLED
	support_half_float_vertices = false;
#endif
	bool disable_half_float = false; //GLOBAL_GET("rendering/opengl/compatibility/disable_half_float");
	if (disable_half_float) {
		support_half_float_vertices = false;
	}

	etc_supported = extensions.has("GL_OES_compressed_ETC1_RGB8_texture");
	latc_supported = extensions.has("GL_EXT_texture_compression_latc");
	bptc_supported = extensions.has("GL_ARB_texture_compression_bptc");
	rgtc_supported = extensions.has("GL_EXT_texture_compression_rgtc") || extensions.has("GL_ARB_texture_compression_rgtc") || extensions.has("EXT_texture_compression_rgtc");
	bptc_supported = extensions.has("GL_ARB_texture_compression_bptc") || extensions.has("EXT_texture_compression_bptc");
	srgb_decode_supported = extensions.has("GL_EXT_texture_sRGB_decode");

	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &max_vertex_texture_image_units);
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_texture_image_units);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);

	force_vertex_shading = false; //GLOBAL_GET("rendering/quality/shading/force_vertex_shading");
	use_fast_texture_filter = false; //GLOBAL_GET("rendering/quality/filters/use_nearest_mipmap_filter");
	// should_orphan = GLOBAL_GET("rendering/options/api_usage_legacy/orphan_buffers");
}

Config::~Config() {
	singleton = nullptr;
}

#endif // GLES3_ENABLED
