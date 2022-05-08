/*************************************************************************/
/*  config.h                                                             */
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

#ifndef CONFIG_GLES3_H
#define CONFIG_GLES3_H

#ifdef GLES3_ENABLED

#include "core/string/ustring.h"
#include "core/templates/set.h"

// This must come first to avoid windows.h mess
#include "platform_config.h"
#ifndef OPENGL_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include OPENGL_INCLUDE_H
#endif

namespace GLES3 {

class Config {
private:
	static Config *singleton;

public:
	bool use_nearest_mip_filter = false;
	bool use_skeleton_software = false;

	int max_vertex_texture_image_units = 0;
	int max_texture_image_units = 0;
	int max_texture_size = 0;
	int max_uniform_buffer_size = 0;

	// TODO implement wireframe in OpenGL
	// bool generate_wireframes;

	Set<String> extensions;

	bool float_texture_supported = false;
	bool s3tc_supported = false;
	bool latc_supported = false;
	bool rgtc_supported = false;
	bool bptc_supported = false;
	bool etc_supported = false;
	bool etc2_supported = false;
	bool srgb_decode_supported = false;

	bool keep_original_textures = false;

	bool force_vertex_shading = false;

	bool use_rgba_2d_shadows = false;
	bool use_rgba_3d_shadows = false;

	bool support_32_bits_indices = false;
	bool support_write_depth = false;
	bool support_half_float_vertices = false;
	bool support_npot_repeat_mipmap = false;
	bool support_depth_cubemaps = false;
	bool support_shadow_cubemaps = false;
	bool support_anisotropic_filter = false;
	float anisotropic_level = 0.0f;

	GLuint depth_internalformat = 0;
	GLuint depth_type = 0;
	GLuint depth_buffer_internalformat = 0;

	// in some cases the legacy render didn't orphan. We will mark these
	// so the user can switch orphaning off for them.
	bool should_orphan = true;

	static Config *get_singleton() { return singleton; };

	Config();
	~Config();
};

} // namespace GLES3

#endif // GLES3_ENABLED

#endif // !CONFIG_GLES3_H
