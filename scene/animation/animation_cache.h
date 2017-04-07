/*************************************************************************/
/*  animation_cache.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef ANIMATION_CACHE_H
#define ANIMATION_CACHE_H

#include "scene/3d/skeleton.h"
#include "scene/resources/animation.h"

class AnimationCache : public Object {

	GDCLASS(AnimationCache, Object);

	struct Path {

		RES resource;
		Object *object;
		Skeleton *skeleton; // haxor
		Node *node;
		Spatial *spatial;

		int bone_idx;
		StringName property;
		bool valid;
		Path() {
			object = NULL;
			skeleton = NULL;
			node = NULL;
			bone_idx = -1;
			valid = false;
			spatial = NULL;
		}
	};

	Set<Node *> connected_nodes;
	Vector<Path> path_cache;

	Node *root;
	Ref<Animation> animation;
	bool cache_dirty;
	bool cache_valid;

	void _node_exit_tree(Node *p_node);

	void _clear_cache();
	void _update_cache();
	void _animation_changed();

protected:
	static void _bind_methods();

public:
	void set_track_transform(int p_idx, const Transform &p_transform);
	void set_track_value(int p_idx, const Variant &p_value);
	void call_track(int p_idx, const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	void set_all(float p_time, float p_delta = 0);

	void set_animation(const Ref<Animation> &p_animation);
	void set_root(Node *p_root);

	AnimationCache();
};

#endif // ANIMATION_CACHE_H
