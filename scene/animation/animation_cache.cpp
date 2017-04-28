/*************************************************************************/
/*  animation_cache.cpp                                                  */
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
#include "animation_cache.h"

void AnimationCache::_node_exit_tree(Node *p_node) {

	//it is one shot, so it disconnects upon arrival

	ERR_FAIL_COND(!connected_nodes.has(p_node));

	connected_nodes.erase(p_node);

	for (int i = 0; i < path_cache.size(); i++) {

		if (path_cache[i].node != p_node)
			continue;

		path_cache[i].valid = false; //invalidate path cache
	}
}

void AnimationCache::_animation_changed() {

	_clear_cache();
}

void AnimationCache::_clear_cache() {

	while (connected_nodes.size()) {

		connected_nodes.front()->get()->disconnect("tree_exited", this, "_node_exit_tree");
		connected_nodes.erase(connected_nodes.front());
	}
	path_cache.clear();
	cache_valid = false;
	cache_dirty = true;
}

void AnimationCache::_update_cache() {

	cache_valid = false;

	ERR_FAIL_COND(!root);
	ERR_FAIL_COND(!root->is_inside_tree());
	ERR_FAIL_COND(animation.is_null());

	for (int i = 0; i < animation->get_track_count(); i++) {

		NodePath np = animation->track_get_path(i);

		Node *node = root->get_node(np);
		if (!node) {

			path_cache.push_back(Path());
			ERR_EXPLAIN("Invalid Track Path in Animation: " + np);
			ERR_CONTINUE(!node);
		}

		Path path;

		Ref<Resource> res;

		if (np.get_subname_count()) {

			if (animation->track_get_type(i) == Animation::TYPE_TRANSFORM) {

				path_cache.push_back(Path());
				ERR_EXPLAIN("Transform tracks can't have a subpath: " + np);
				ERR_CONTINUE(animation->track_get_type(i) == Animation::TYPE_TRANSFORM);
			}

			RES res;

			for (int j = 0; j < np.get_subname_count(); j++) {
				res = j == 0 ? node->get(np.get_subname(j)) : res->get(np.get_subname(j));
				if (res.is_null())
					break;
			}

			if (res.is_null()) {

				path_cache.push_back(Path());
				ERR_EXPLAIN("Invalid Track SubPath in Animation: " + np);
				ERR_CONTINUE(res.is_null());
			}

			path.resource = res;
			path.object = res.ptr();

		} else {

			if (animation->track_get_type(i) == Animation::TYPE_TRANSFORM) {
				StringName property = np.get_property();
				String ps = property;

				Spatial *sp = node->cast_to<Spatial>();

				if (!sp) {

					path_cache.push_back(Path());
					ERR_EXPLAIN("Transform track not of type Spatial: " + np);
					ERR_CONTINUE(!sp);
				}

				if (ps != "") {

					Skeleton *sk = node->cast_to<Skeleton>();
					if (!sk) {

						path_cache.push_back(Path());
						ERR_EXPLAIN("Property defined in Transform track, but not a Skeleton!: " + np);
						ERR_CONTINUE(!sk);
					}

					int idx = sk->find_bone(ps);
					if (idx == -1) {

						path_cache.push_back(Path());
						ERR_EXPLAIN("Property defined in Transform track, but not a Skeleton Bone!: " + np);
						ERR_CONTINUE(idx == -1);
					}

					path.bone_idx = idx;
					path.skeleton = sk;
				}

				path.spatial = sp;
			}

			path.node = node;
			path.object = node;
		}

		if (animation->track_get_type(i) == Animation::TYPE_VALUE) {

			if (np.get_property().operator String() == "") {

				path_cache.push_back(Path());
				ERR_EXPLAIN("Value Track lacks property: " + np);
				ERR_CONTINUE(np.get_property().operator String() == "");
			}

			path.property = np.get_property();

		} else if (animation->track_get_type(i) == Animation::TYPE_METHOD) {

			if (np.get_property().operator String() != "") {

				path_cache.push_back(Path());
				ERR_EXPLAIN("Method Track has property: " + np);
				ERR_CONTINUE(np.get_property().operator String() != "");
			}
		}

		path.valid = true;

		path_cache.push_back(path);

		if (!connected_nodes.has(path.node)) {
			connected_nodes.insert(path.node);
			path.node->connect("tree_exited", this, "_node_exit_tree", Node::make_binds(path.node), CONNECT_ONESHOT);
		}
	}

	cache_dirty = false;
	cache_valid = true;
}

void AnimationCache::set_track_transform(int p_idx, const Transform &p_transform) {

	if (cache_dirty)
		_update_cache();

	ERR_FAIL_COND(!cache_valid);
	ERR_FAIL_INDEX(p_idx, path_cache.size());
	Path &p = path_cache[p_idx];
	if (!p.valid)
		return;

	ERR_FAIL_COND(!p.node);
	ERR_FAIL_COND(!p.spatial);

	if (p.skeleton) {
		p.skeleton->set_bone_pose(p.bone_idx, p_transform);
	} else {
		p.spatial->set_transform(p_transform);
	}
}

void AnimationCache::set_track_value(int p_idx, const Variant &p_value) {

	if (cache_dirty)
		_update_cache();

	ERR_FAIL_COND(!cache_valid);
	ERR_FAIL_INDEX(p_idx, path_cache.size());
	Path &p = path_cache[p_idx];
	if (!p.valid)
		return;

	ERR_FAIL_COND(!p.object);
	p.object->set(p.property, p_value);
}

void AnimationCache::call_track(int p_idx, const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (cache_dirty)
		_update_cache();

	ERR_FAIL_COND(!cache_valid);
	ERR_FAIL_INDEX(p_idx, path_cache.size());
	Path &p = path_cache[p_idx];
	if (!p.valid)
		return;

	ERR_FAIL_COND(!p.object);
	p.object->call(p_method, p_args, p_argcount, r_error);
}

void AnimationCache::set_all(float p_time, float p_delta) {

	if (cache_dirty)
		_update_cache();

	ERR_FAIL_COND(!cache_valid);

	int tc = animation->get_track_count();
	for (int i = 0; i < tc; i++) {

		switch (animation->track_get_type(i)) {

			case Animation::TYPE_TRANSFORM: {

				Vector3 loc, scale;
				Quat rot;
				animation->transform_track_interpolate(i, p_time, &loc, &rot, &scale);
				Transform tr(Basis(rot), loc);
				tr.basis.scale(scale);

				set_track_transform(i, tr);

			} break;
			case Animation::TYPE_VALUE: {

				if (animation->value_track_get_update_mode(i) == Animation::UPDATE_CONTINUOUS || (animation->value_track_get_update_mode(i) == Animation::UPDATE_DISCRETE && p_delta == 0)) {
					Variant v = animation->value_track_interpolate(i, p_time);
					set_track_value(i, v);
				} else {

					List<int> indices;
					animation->value_track_get_key_indices(i, p_time, p_delta, &indices);

					for (List<int>::Element *E = indices.front(); E; E = E->next()) {

						Variant v = animation->track_get_key_value(i, E->get());
						set_track_value(i, v);
					}
				}

			} break;
			case Animation::TYPE_METHOD: {

				List<int> indices;
				animation->method_track_get_key_indices(i, p_time, p_delta, &indices);

				for (List<int>::Element *E = indices.front(); E; E = E->next()) {

					Vector<Variant> args = animation->method_track_get_params(i, E->get());
					StringName name = animation->method_track_get_name(i, E->get());
					Variant::CallError err;

					if (!args.size()) {

						call_track(i, name, NULL, 0, err);
					} else {

						Vector<Variant *> argptrs;
						argptrs.resize(args.size());
						for (int j = 0; j < args.size(); j++) {

							argptrs[j] = &args[j];
						}

						call_track(i, name, (const Variant **)&argptrs[0], args.size(), err);
					}
				}

			} break;
			default: {}
		}
	}
}

void AnimationCache::set_animation(const Ref<Animation> &p_animation) {

	_clear_cache();

	if (animation.is_valid())
		animation->disconnect("changed", this, "_animation_changed");

	animation = p_animation;

	if (animation.is_valid())
		animation->connect("changed", this, "_animation_changed");
}

void AnimationCache::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_node_exit_tree"), &AnimationCache::_node_exit_tree);
	ClassDB::bind_method(D_METHOD("_animation_changed"), &AnimationCache::_animation_changed);
}

void AnimationCache::set_root(Node *p_root) {

	_clear_cache();
	root = p_root;
}

AnimationCache::AnimationCache() {

	root = NULL;
	cache_dirty = true;
	cache_valid = false;
}
