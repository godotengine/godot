/*************************************************************************/
/*  rasterizer_scene_opengl.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RASTERIZER_SCENE_OPENGL_H
#define RASTERIZER_SCENE_OPENGL_H

#include "drivers/opengl/rasterizer_platforms.h"
#ifdef OPENGL_BACKEND_ENABLED

#include "core/math/camera_matrix.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "drivers/opengl/rasterizer_common_stubs.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering_server.h"
#include "shaders/scene.glsl.gen.h"

class RasterizerSceneOpenGL : public StubsScene {
public:
	struct State {
		SceneShaderOpenGL scene_shader;

	} state;

public:
	RasterizerSceneOpenGL() {}
	~RasterizerSceneOpenGL() {}
};

#endif // OPENGL_BACKEND_ENABLED

#endif // RASTERIZER_SCENE_OPENGL_H
