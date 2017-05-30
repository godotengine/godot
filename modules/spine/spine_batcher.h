/*************************************************************************/
/*  spine_batcher.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifdef MODULE_SPINE_ENABLED

#ifndef SPINE_BATCHER_H
#define SPINE_BATCHER_H

#include "scene/2d/node_2d.h"

class SpineBatcher {

	Node2D *owner;

	enum {
		CMD_DRAW_ELEMENT,
		CMD_SET_BLEND_MODE,
	};

	struct Command {
		Command() {}
		virtual ~Command() {}

		int cmd;
		virtual void draw(RID ci) {}
	};

	struct SetBlendMode : Command {
		int mode;

		SetBlendMode(int p_mode);
		void draw(RID ci);
	};

	struct Elements : Command {
		Ref<Texture> texture;
		int vertices_count;
		int indies_count;
		Vector2 *vertices;
		Color *colors;
		Vector2 *uvs;
		int* indies;

		Elements();
		~Elements();
		void draw(RID ci);
	};

	Elements *elements;

	List<Command *> element_list;
	List<Command *> drawed_list;

	void push_elements();

public:

	void reset();

	void add(Ref<Texture> p_texture,
		const float* p_vertices, const float* p_uvs, int p_vertices_count,
		const int* p_indies, int p_indies_count,
		Color *p_color, bool flip_x, bool flip_y);

	void add_set_blender_mode(bool p_mode);

	void flush();

	SpineBatcher(Node2D *owner);
	~SpineBatcher();
};

#endif // SPINE_BATCHER_H

#endif // MODULE_SPINE_ENABLED
