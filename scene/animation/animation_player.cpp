/**************************************************************************/
/*  animation_player.cpp                                                  */
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

#include "animation_player.h"

#include "core/config/engine.h"
#include "core/object/message_queue.h"
#include "scene/scene_string_names.h"
#include "servers/audio/audio_stream.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/2d/skeleton_2d.h"

void AnimatedValuesBackup::update_skeletons() {
	for (int i = 0; i < entries.size(); i++) {
		if (entries[i].bone_idx != -1) {
			// 3D bone
			Object::cast_to<Skeleton3D>(entries[i].object)->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		} else {
			Bone2D *bone = Object::cast_to<Bone2D>(entries[i].object);
			if (bone && bone->skeleton) {
				// 2D bone
				bone->skeleton->_update_transform();
			}
		}
	}
}

void AnimatedValuesBackup::restore() const {
	for (int i = 0; i < entries.size(); i++) {
		const AnimatedValuesBackup::Entry *entry = &entries[i];
		if (entry->bone_idx == -1) {
			entry->object->set_indexed(entry->subpath, entry->value);
		} else {
			Array arr = entry->value;
			if (arr.size() == 3) {
				Object::cast_to<Skeleton3D>(entry->object)->set_bone_pose_position(entry->bone_idx, arr[0]);
				Object::cast_to<Skeleton3D>(entry->object)->set_bone_pose_rotation(entry->bone_idx, arr[1]);
				Object::cast_to<Skeleton3D>(entry->object)->set_bone_pose_scale(entry->bone_idx, arr[2]);
			}
		}
	}
}

void AnimatedValuesBackup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("restore"), &AnimatedValuesBackup::restore);
}
#endif

bool AnimationPlayer::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name.begins_with("playback/play")) { // bw compatibility

		set_current_animation(p_value);

	} else if (name.begins_with("anims/")) {
		// Backwards compatibility with 3.x, add them to "default" library.
		String which = name.get_slicec('/', 1);

		Ref<Animation> anim = p_value;
		Ref<AnimationLibrary> al;
		if (!has_animation_library(StringName())) {
			al.instantiate();
			add_animation_library(StringName(), al);
		} else {
			al = get_animation_library(StringName());
		}
		al->add_animation(which, anim);

	} else if (name.begins_with("libraries")) {
		Dictionary d = p_value;
		while (animation_libraries.size()) {
			remove_animation_library(animation_libraries[0].name);
		}
		List<Variant> keys;
		d.get_key_list(&keys);
		for (const Variant &K : keys) {
			StringName lib_name = K;
			Ref<AnimationLibrary> lib = d[lib_name];
			add_animation_library(lib_name, lib);
		}
		emit_signal("animation_libraries_updated");
	} else if (name.begins_with("next/")) {
		String which = name.get_slicec('/', 1);
		animation_set_next(which, p_value);

	} else if (p_name == SceneStringNames::get_singleton()->blend_times) {
		Array array = p_value;
		int len = array.size();
		ERR_FAIL_COND_V(len % 3, false);

		for (int i = 0; i < len / 3; i++) {
			StringName from = array[i * 3 + 0];
			StringName to = array[i * 3 + 1];
			float time = array[i * 3 + 2];

			set_blend_time(from, to, time);
		}

	} else {
		return false;
	}

	return true;
}

bool AnimationPlayer::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name == "playback/play") { // bw compatibility

		r_ret = get_current_animation();

	} else if (name.begins_with("libraries")) {
		Dictionary d;
		for (const AnimationLibraryData &lib : animation_libraries) {
			d[lib.name] = lib.library;
		}

		r_ret = d;

	} else if (name.begins_with("next/")) {
		String which = name.get_slicec('/', 1);

		r_ret = animation_get_next(which);

	} else if (name == "blend_times") {
		Vector<BlendKey> keys;
		for (const KeyValue<BlendKey, double> &E : blend_times) {
			keys.ordered_insert(E.key);
		}

		Array array;
		for (int i = 0; i < keys.size(); i++) {
			array.push_back(keys[i].from);
			array.push_back(keys[i].to);
			array.push_back(blend_times.get(keys[i]));
		}

		r_ret = array;
	} else {
		return false;
	}

	return true;
}

void AnimationPlayer::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "current_animation") {
		List<String> names;

		for (const KeyValue<StringName, AnimationData> &E : animation_set) {
			names.push_back(E.key);
		}
		names.sort();
		names.push_front("[stop]");
		String hint;
		for (List<String>::Element *E = names.front(); E; E = E->next()) {
			if (E != names.front()) {
				hint += ",";
			}
			hint += E->get();
		}

		p_property.hint_string = hint;
	}
}

void AnimationPlayer::_get_property_list(List<PropertyInfo> *p_list) const {
	List<PropertyInfo> anim_names;

	anim_names.push_back(PropertyInfo(Variant::DICTIONARY, PNAME("libraries")));

	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.next != StringName()) {
			anim_names.push_back(PropertyInfo(Variant::STRING, "next/" + String(E.key), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		}
	}

	anim_names.sort();

	for (const PropertyInfo &E : anim_names) {
		p_list->push_back(E);
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "blend_times", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
}

void AnimationPlayer::advance(double p_time) {
	_animation_process(p_time);
}

void AnimationPlayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!processing) {
				//make sure that a previous process state was not saved
				//only process if "processing" is set
				set_physics_process_internal(false);
				set_process_internal(false);
			}
			clear_caches();
		} break;

		case NOTIFICATION_READY: {
			if (!Engine::get_singleton()->is_editor_hint() && animation_set.has(autoplay)) {
				play(autoplay);
				_animation_process(0);
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (process_callback == ANIMATION_PROCESS_PHYSICS) {
				break;
			}

			if (processing) {
				_animation_process(get_process_delta_time());
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (process_callback == ANIMATION_PROCESS_IDLE) {
				break;
			}

			if (processing) {
				_animation_process(get_physics_process_delta_time());
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			clear_caches();
		} break;
	}
}

void AnimationPlayer::_ensure_node_caches(AnimationData *p_anim, Node *p_root_override) {
	// Already cached?
	if (p_anim->node_cache.size() == p_anim->animation->get_track_count()) {
		return;
	}

	Node *parent = p_root_override ? p_root_override : get_node_or_null(root);

	ERR_FAIL_NULL(parent);

	Animation *a = p_anim->animation.operator->();

	p_anim->node_cache.resize(a->get_track_count());

	setup_pass++;

	for (int i = 0; i < a->get_track_count(); i++) {
		p_anim->node_cache.write[i] = nullptr;

		if (!a->track_is_enabled(i)) {
			continue;
		}

		Ref<Resource> resource;
		Vector<StringName> leftover_path;
		Node *child = parent->get_node_and_resource(a->track_get_path(i), resource, leftover_path);
		ERR_CONTINUE_MSG(!child, "On Animation: '" + p_anim->name + "', couldn't resolve track:  '" + String(a->track_get_path(i)) + "'."); // couldn't find the child node
		ObjectID id = resource.is_valid() ? resource->get_instance_id() : child->get_instance_id();
		int bone_idx = -1;
		int blend_shape_idx = -1;

#ifndef _3D_DISABLED
		if (a->track_get_path(i).get_subname_count() == 1 && Object::cast_to<Skeleton3D>(child)) {
			Skeleton3D *sk = Object::cast_to<Skeleton3D>(child);
			bone_idx = sk->find_bone(a->track_get_path(i).get_subname(0));
			if (bone_idx == -1) {
				continue;
			}
		}

		if (a->track_get_type(i) == Animation::TYPE_BLEND_SHAPE) {
			MeshInstance3D *mi_3d = Object::cast_to<MeshInstance3D>(child);
			if (!mi_3d) {
				continue;
			}
			if (a->track_get_path(i).get_subname_count() != 1) {
				continue;
			}

			blend_shape_idx = mi_3d->find_blend_shape_by_name(a->track_get_path(i).get_subname(0));
			if (blend_shape_idx == -1) {
				continue;
			}
		}

#endif // _3D_DISABLED

		if (!child->is_connected("tree_exiting", callable_mp(this, &AnimationPlayer::_node_removed))) {
			child->connect("tree_exiting", callable_mp(this, &AnimationPlayer::_node_removed).bind(child), CONNECT_ONE_SHOT);
		}

		TrackNodeCacheKey key;
		key.id = id;
		key.bone_idx = bone_idx;
		key.blend_shape_idx = blend_shape_idx;

		if (!node_cache_map.has(key)) {
			node_cache_map[key] = TrackNodeCache();
		}

		TrackNodeCache *node_cache = &node_cache_map[key];
		p_anim->node_cache.write[i] = node_cache;

		node_cache->path = a->track_get_path(i);
		node_cache->node = child;
		node_cache->resource = resource;
		node_cache->node_2d = Object::cast_to<Node2D>(child);
#ifndef _3D_DISABLED
		if (a->track_get_type(i) == Animation::TYPE_POSITION_3D || a->track_get_type(i) == Animation::TYPE_ROTATION_3D || a->track_get_type(i) == Animation::TYPE_SCALE_3D) {
			// special cases and caches for transform tracks

			if (node_cache->last_setup_pass != setup_pass) {
				node_cache->loc_used = false;
				node_cache->rot_used = false;
				node_cache->scale_used = false;
			}

			// cache node_3d
			node_cache->node_3d = Object::cast_to<Node3D>(child);
			// cache skeleton
			node_cache->skeleton = Object::cast_to<Skeleton3D>(child);
			if (node_cache->skeleton) {
				if (a->track_get_path(i).get_subname_count() == 1) {
					StringName bone_name = a->track_get_path(i).get_subname(0);

					node_cache->bone_idx = node_cache->skeleton->find_bone(bone_name);
					if (node_cache->bone_idx < 0) {
						// broken track (nonexistent bone)
						node_cache->skeleton = nullptr;
						node_cache->node_3d = nullptr;
						ERR_CONTINUE(node_cache->bone_idx < 0);
					}
					Transform3D rest = node_cache->skeleton->get_bone_rest(bone_idx);
					node_cache->init_loc = rest.origin;
					node_cache->init_rot = rest.basis.get_rotation_quaternion();
					node_cache->init_scale = rest.basis.get_scale();
				} else {
					// Not a skeleton, the node can be accessed with the node_3d member.
					node_cache->skeleton = nullptr;
				}
			}

			switch (a->track_get_type(i)) {
				case Animation::TYPE_POSITION_3D: {
					node_cache->loc_used = true;
				} break;
				case Animation::TYPE_ROTATION_3D: {
					node_cache->rot_used = true;
				} break;
				case Animation::TYPE_SCALE_3D: {
					node_cache->scale_used = true;
				} break;
				default: {
				}
			}
		}

		if (a->track_get_type(i) == Animation::TYPE_BLEND_SHAPE) {
			// special cases and caches for transform tracks
			node_cache->node_blend_shape = Object::cast_to<MeshInstance3D>(child);
			node_cache->blend_shape_idx = blend_shape_idx;
		}

#endif // _3D_DISABLED

		if (a->track_get_type(i) == Animation::TYPE_VALUE) {
			if (!node_cache->property_anim.has(a->track_get_path(i).get_concatenated_subnames())) {
				TrackNodeCache::PropertyAnim pa;
				pa.subpath = leftover_path;
				pa.object = resource.is_valid() ? (Object *)resource.ptr() : (Object *)child;
				pa.special = SP_NONE;
				pa.owner = p_anim->node_cache[i];
				if (false && node_cache->node_2d) {
					if (leftover_path.size() == 1 && leftover_path[0] == SceneStringNames::get_singleton()->transform_pos) {
						pa.special = SP_NODE2D_POS;
					} else if (leftover_path.size() == 1 && leftover_path[0] == SceneStringNames::get_singleton()->transform_rot) {
						pa.special = SP_NODE2D_ROT;
					} else if (leftover_path.size() == 1 && leftover_path[0] == SceneStringNames::get_singleton()->transform_scale) {
						pa.special = SP_NODE2D_SCALE;
					}
				}
				node_cache->property_anim[a->track_get_path(i).get_concatenated_subnames()] = pa;
			}
		}

		if (a->track_get_type(i) == Animation::TYPE_BEZIER && leftover_path.size()) {
			if (!node_cache->bezier_anim.has(a->track_get_path(i).get_concatenated_subnames())) {
				TrackNodeCache::BezierAnim ba;
				ba.bezier_property = leftover_path;
				ba.object = resource.is_valid() ? (Object *)resource.ptr() : (Object *)child;
				ba.owner = p_anim->node_cache[i];

				node_cache->bezier_anim[a->track_get_path(i).get_concatenated_subnames()] = ba;
			}
		}

		if (a->track_get_type(i) == Animation::TYPE_AUDIO) {
			if (!node_cache->audio_anim.has(a->track_get_path(i).get_concatenated_names())) {
				TrackNodeCache::AudioAnim aa;
				aa.object = (Object *)child;
				aa.audio_stream.instantiate();
				aa.audio_stream->set_polyphony(audio_max_polyphony);

				node_cache->audio_anim[a->track_get_path(i).get_concatenated_names()] = aa;
			}
		}

		node_cache->last_setup_pass = setup_pass;
	}
}

static void _call_object(Object *p_object, const StringName &p_method, const Vector<Variant> &p_params, bool p_deferred) {
	// Separate function to use alloca() more efficiently
	const Variant **argptrs = (const Variant **)alloca(sizeof(const Variant **) * p_params.size());
	const Variant *args = p_params.ptr();
	uint32_t argcount = p_params.size();
	for (uint32_t i = 0; i < argcount; i++) {
		argptrs[i] = &args[i];
	}
	if (p_deferred) {
		MessageQueue::get_singleton()->push_callp(p_object, p_method, argptrs, argcount);
	} else {
		Callable::CallError ce;
		p_object->callp(p_method, argptrs, argcount, ce);
	}
}

Variant AnimationPlayer::post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
	Variant res;
	if (GDVIRTUAL_CALL(_post_process_key_value, p_anim, p_track, p_value, const_cast<Object *>(p_object), p_object_idx, res)) {
		return res;
	}

	return _post_process_key_value(p_anim, p_track, p_value, p_object, p_object_idx);
}

Variant AnimationPlayer::_post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
	switch (p_anim->track_get_type(p_track)) {
#ifndef _3D_DISABLED
		case Animation::TYPE_POSITION_3D: {
			if (p_object_idx >= 0) {
				const Skeleton3D *skel = Object::cast_to<Skeleton3D>(p_object);
				return Vector3(p_value) * skel->get_motion_scale();
			}
			return p_value;
		} break;
#endif // _3D_DISABLED
		default: {
		} break;
	}
	return p_value;
}

void AnimationPlayer::_animation_process_animation(AnimationData *p_anim, double p_prev_time, double p_time, double p_delta, float p_interp, bool p_is_current, bool p_seeked, bool p_started, Animation::LoopedFlag p_looped_flag) {
	_ensure_node_caches(p_anim);
	ERR_FAIL_COND(p_anim->node_cache.size() != p_anim->animation->get_track_count());

	Animation *a = p_anim->animation.operator->();
#ifdef TOOLS_ENABLED
	bool can_call = is_inside_tree() && !Engine::get_singleton()->is_editor_hint();
#endif // TOOLS_ENABLED
	bool backward = signbit(p_delta);

	for (int i = 0; i < a->get_track_count(); i++) {
		// If an animation changes this animation (or it animates itself)
		// we need to recreate our animation cache
		if (p_anim->node_cache.size() != a->get_track_count()) {
			_ensure_node_caches(p_anim);
		}

		TrackNodeCache *nc = p_anim->node_cache[i];

		if (!nc) {
			continue; // no node cache for this track, skip it
		}

		if (!a->track_is_enabled(i)) {
			continue; // do nothing if the track is disabled
		}

		if (a->track_get_key_count(i) == 0) {
			continue; // do nothing if track is empty
		}

		switch (a->track_get_type(i)) {
			case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
				if (!nc->node_3d) {
					continue;
				}

				Vector3 loc;

				Error err = a->try_position_track_interpolate(i, p_time, &loc);
				//ERR_CONTINUE(err!=OK); //used for testing, should be removed

				if (err != OK) {
					continue;
				}
				loc = post_process_key_value(a, i, loc, nc->node_3d, nc->bone_idx);

				if (nc->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_size >= NODE_CACHE_UPDATE_MAX);
					cache_update[cache_update_size++] = nc;
					nc->accum_pass = accum_pass;
					nc->loc_accum = loc;
					nc->rot_accum = nc->init_rot;
					nc->scale_accum = nc->init_scale;
				} else {
					nc->loc_accum = nc->loc_accum.lerp(loc, p_interp);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_ROTATION_3D: {
#ifndef _3D_DISABLED
				if (!nc->node_3d) {
					continue;
				}

				Quaternion rot;

				Error err = a->try_rotation_track_interpolate(i, p_time, &rot);
				//ERR_CONTINUE(err!=OK); //used for testing, should be removed

				if (err != OK) {
					continue;
				}
				rot = post_process_key_value(a, i, rot, nc->node_3d, nc->bone_idx);

				if (nc->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_size >= NODE_CACHE_UPDATE_MAX);
					cache_update[cache_update_size++] = nc;
					nc->accum_pass = accum_pass;
					nc->loc_accum = nc->init_loc;
					nc->rot_accum = rot;
					nc->scale_accum = nc->init_scale;
				} else {
					nc->rot_accum = nc->rot_accum.slerp(rot, p_interp);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_SCALE_3D: {
#ifndef _3D_DISABLED
				if (!nc->node_3d) {
					continue;
				}

				Vector3 scale;

				Error err = a->try_scale_track_interpolate(i, p_time, &scale);
				//ERR_CONTINUE(err!=OK); //used for testing, should be removed

				if (err != OK) {
					continue;
				}
				scale = post_process_key_value(a, i, scale, nc->node_3d, nc->bone_idx);

				if (nc->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_size >= NODE_CACHE_UPDATE_MAX);
					cache_update[cache_update_size++] = nc;
					nc->accum_pass = accum_pass;
					nc->loc_accum = nc->init_loc;
					nc->rot_accum = nc->init_rot;
					nc->scale_accum = scale;
				} else {
					nc->scale_accum = nc->scale_accum.lerp(scale, p_interp);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
				if (!nc->node_blend_shape) {
					continue;
				}

				float blend;

				Error err = a->try_blend_shape_track_interpolate(i, p_time, &blend);
				//ERR_CONTINUE(err!=OK); //used for testing, should be removed

				if (err != OK) {
					continue;
				}
				blend = post_process_key_value(a, i, blend, nc->node_blend_shape, nc->blend_shape_idx);

				if (nc->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_size >= NODE_CACHE_UPDATE_MAX);
					nc->accum_pass = accum_pass;
					cache_update[cache_update_size++] = nc;
					nc->blend_shape_accum = blend;
				} else {
					nc->blend_shape_accum = Math::lerp(nc->blend_shape_accum, blend, p_interp);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_VALUE: {
				if (!nc->node) {
					continue;
				}

				//StringName property=a->track_get_path(i).get_property();

				HashMap<StringName, TrackNodeCache::PropertyAnim>::Iterator E = nc->property_anim.find(a->track_get_path(i).get_concatenated_subnames());
				ERR_CONTINUE(!E); //should it continue, or create a new one?

				TrackNodeCache::PropertyAnim *pa = &E->value;

				Animation::UpdateMode update_mode = a->value_track_get_update_mode(i);

				if (update_mode == Animation::UPDATE_CAPTURE) {
					if (p_started || pa->capture == Variant()) {
						pa->capture = pa->object->get_indexed(pa->subpath);
					}

					int key_count = a->track_get_key_count(i);
					if (key_count == 0) {
						continue; //eeh not worth it
					}

					double first_key_time = a->track_get_key_time(i, 0);
					double transition = 1.0;
					int first_key = 0;

					if (first_key_time == 0.0) {
						//ignore, use for transition
						if (key_count == 1) {
							continue; //with one key we can't do anything
						}
						transition = (double)a->track_get_key_transition(i, 0);
						first_key_time = a->track_get_key_time(i, 1);
						first_key = 1;
					}

					if (p_time < first_key_time) {
						double c = Math::ease(p_time / first_key_time, transition);
						Variant first_value = a->track_get_key_value(i, first_key);
						first_value = post_process_key_value(a, i, first_value, nc->node);
						Variant interp_value = Animation::interpolate_variant(pa->capture, first_value, c);
						if (pa->accum_pass != accum_pass) {
							ERR_CONTINUE(cache_update_prop_size >= NODE_CACHE_UPDATE_MAX);
							cache_update_prop[cache_update_prop_size++] = pa;
							pa->value_accum = interp_value;
							pa->accum_pass = accum_pass;
						} else {
							pa->value_accum = Animation::interpolate_variant(pa->value_accum, interp_value, p_interp);
						}

						continue; //handled
					}
				}

				if (update_mode == Animation::UPDATE_CONTINUOUS || update_mode == Animation::UPDATE_CAPTURE) {
					Variant value = a->value_track_interpolate(i, p_time);

					if (value == Variant()) {
						continue;
					}
					value = post_process_key_value(a, i, value, nc->node);

					if (pa->accum_pass != accum_pass) {
						ERR_CONTINUE(cache_update_prop_size >= NODE_CACHE_UPDATE_MAX);
						cache_update_prop[cache_update_prop_size++] = pa;
						pa->value_accum = value;
						pa->accum_pass = accum_pass;
					} else {
						pa->value_accum = Animation::interpolate_variant(pa->value_accum, value, p_interp);
					}

				} else {
					List<int> indices;

					if (p_seeked) {
						int found_key = a->track_find_key(i, p_time);
						if (found_key >= 0) {
							indices.push_back(found_key);
						}
					} else {
						if (p_started) {
							int first_key = a->track_find_key(i, p_prev_time, Animation::FIND_MODE_EXACT);
							if (first_key >= 0) {
								indices.push_back(first_key);
							}
						}
						a->track_get_key_indices_in_range(i, p_time, p_delta, &indices, p_looped_flag);
					}

					for (int &F : indices) {
						Variant value = a->track_get_key_value(i, F);
						value = post_process_key_value(a, i, value, nc->node);
						switch (pa->special) {
							case SP_NONE: {
								bool valid;
								pa->object->set_indexed(pa->subpath, value, &valid); //you are not speshul
#ifdef DEBUG_ENABLED
								if (!valid) {
									ERR_PRINT("Failed setting track value '" + String(pa->owner->path) + "'. Check if the property exists or the type of key is valid. Animation '" + a->get_name() + "' at node '" + get_path() + "'.");
								}
#endif

							} break;
							case SP_NODE2D_POS: {
#ifdef DEBUG_ENABLED
								if (value.get_type() != Variant::VECTOR2) {
									ERR_PRINT("Position key at time " + rtos(p_time) + " in Animation Track '" + String(pa->owner->path) + "' not of type Vector2(). Animation '" + a->get_name() + "' at node '" + get_path() + "'.");
								}
#endif
								static_cast<Node2D *>(pa->object)->set_position(value);
							} break;
							case SP_NODE2D_ROT: {
#ifdef DEBUG_ENABLED
								if (value.is_num()) {
									ERR_PRINT("Rotation key at time " + rtos(p_time) + " in Animation Track '" + String(pa->owner->path) + "' not numerical. Animation '" + a->get_name() + "' at node '" + get_path() + "'.");
								}
#endif

								static_cast<Node2D *>(pa->object)->set_rotation((double)value);
							} break;
							case SP_NODE2D_SCALE: {
#ifdef DEBUG_ENABLED
								if (value.get_type() != Variant::VECTOR2) {
									ERR_PRINT("Scale key at time " + rtos(p_time) + " in Animation Track '" + String(pa->owner->path) + "' not of type Vector2()." + a->get_name() + "' at node '" + get_path() + "'.");
								}
#endif

								static_cast<Node2D *>(pa->object)->set_scale(value);
							} break;
						}
					}
				}

			} break;
			case Animation::TYPE_METHOD: {
#ifdef TOOLS_ENABLED
				if (!can_call) {
					continue;
				}
#endif // TOOLS_ENABLED
				if (!p_is_current || !nc->node || is_stopping) {
					continue;
				}

				List<int> indices;

				if (p_seeked) {
					int found_key = a->track_find_key(i, p_time, Animation::FIND_MODE_EXACT);
					if (found_key >= 0) {
						indices.push_back(found_key);
					}
				} else {
					if (p_started) {
						int first_key = a->track_find_key(i, p_prev_time, Animation::FIND_MODE_EXACT);
						if (first_key >= 0) {
							indices.push_back(first_key);
						}
					}
					a->track_get_key_indices_in_range(i, p_time, p_delta, &indices, p_looped_flag);
				}

				for (int &E : indices) {
					StringName method = a->method_track_get_name(i, E);
					Vector<Variant> params = a->method_track_get_params(i, E);
#ifdef DEBUG_ENABLED
					if (!nc->node->has_method(method)) {
						ERR_PRINT("Invalid method call '" + method + "'. '" + a->get_name() + "' at node '" + get_path() + "'.");
					}
#endif
					_call_object(nc->node, method, params, method_call_mode == ANIMATION_METHOD_CALL_DEFERRED);
				}

			} break;
			case Animation::TYPE_BEZIER: {
				if (!nc->node) {
					continue;
				}

				HashMap<StringName, TrackNodeCache::BezierAnim>::Iterator E = nc->bezier_anim.find(a->track_get_path(i).get_concatenated_subnames());
				ERR_CONTINUE(!E); //should it continue, or create a new one?

				TrackNodeCache::BezierAnim *ba = &E->value;

				real_t bezier = a->bezier_track_interpolate(i, p_time);
				bezier = post_process_key_value(a, i, bezier, nc->node);
				if (ba->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_bezier_size >= NODE_CACHE_UPDATE_MAX);
					cache_update_bezier[cache_update_bezier_size++] = ba;
					ba->bezier_accum = bezier;
					ba->accum_pass = accum_pass;
				} else {
					ba->bezier_accum = Math::lerp(ba->bezier_accum, (float)bezier, p_interp);
				}

			} break;
			case Animation::TYPE_AUDIO: {
				if (!nc->node || is_stopping) {
					continue;
				}
#ifdef TOOLS_ENABLED
				if (p_seeked && !can_call) {
					continue; // To avoid spamming the preview in editor.
				}
#endif // TOOLS_ENABLED
				HashMap<StringName, TrackNodeCache::AudioAnim>::Iterator E = nc->audio_anim.find(a->track_get_path(i).get_concatenated_names());
				ERR_CONTINUE(!E); //should it continue, or create a new one?

				TrackNodeCache::AudioAnim *aa = &E->value;
				Node *asp = Object::cast_to<Node>(aa->object);
				if (!asp) {
					continue;
				}
				aa->length = a->get_length();
				aa->time = p_time;
				aa->loop = a->get_loop_mode() != Animation::LOOP_NONE;
				aa->backward = backward;
				if (aa->accum_pass != accum_pass) {
					ERR_CONTINUE(cache_update_audio_size >= NODE_CACHE_UPDATE_MAX);
					cache_update_audio[cache_update_audio_size++] = aa;
					aa->accum_pass = accum_pass;
				}

				HashMap<int, TrackNodeCache::PlayingAudioStreamInfo> &map = aa->playing_streams;
				// Find stream.
				int idx = -1;
				if (p_seeked || p_started) {
					idx = a->track_find_key(i, p_time);
					// Discard previous stream when seeking.
					if (map.has(idx)) {
						aa->audio_stream_playback->stop_stream(map[idx].index);
						map.erase(idx);
					}
				} else {
					List<int> to_play;

					a->track_get_key_indices_in_range(i, p_time, p_delta, &to_play, p_looped_flag);
					if (to_play.size()) {
						idx = to_play.back()->get();
					}
				}
				if (idx < 0) {
					continue;
				}

				// Play stream.
				Ref<AudioStream> stream = a->audio_track_get_key_stream(i, idx);
				if (stream.is_valid()) {
					double start_ofs = a->audio_track_get_key_start_offset(i, idx);
					double end_ofs = a->audio_track_get_key_end_offset(i, idx);
					double len = stream->get_length();

					if (p_seeked || p_started) {
						start_ofs += p_time - a->track_get_key_time(i, idx);
					}

					if (aa->object->call(SNAME("get_stream")) != aa->audio_stream) {
						aa->object->call(SNAME("set_stream"), aa->audio_stream);
						aa->audio_stream_playback.unref();
						if (!playing_audio_stream_players.has(asp)) {
							playing_audio_stream_players.push_back(asp);
						}
					}
					if (!aa->object->call(SNAME("is_playing"))) {
						aa->object->call(SNAME("play"));
					}
					if (!aa->object->call(SNAME("has_stream_playback"))) {
						aa->audio_stream_playback.unref();
						continue;
					}
					if (aa->audio_stream_playback.is_null()) {
						aa->audio_stream_playback = aa->object->call(SNAME("get_stream_playback"));
					}

					TrackNodeCache::PlayingAudioStreamInfo pasi;
					pasi.index = aa->audio_stream_playback->play_stream(stream, start_ofs);
					pasi.start = p_time;
					if (len && end_ofs > 0) { // Force an end at a time.
						pasi.len = len - start_ofs - end_ofs;
					} else {
						pasi.len = 0;
					}
					map[idx] = pasi;
				}

			} break;
			case Animation::TYPE_ANIMATION: {
				if (is_stopping) {
					continue;
				}

				AnimationPlayer *player = Object::cast_to<AnimationPlayer>(nc->node);
				if (!player) {
					continue;
				}

				if (p_seeked) {
					//seek
					int idx = a->track_find_key(i, p_time);
					if (idx < 0) {
						continue;
					}

					double pos = a->track_get_key_time(i, idx);

					StringName anim_name = a->animation_track_get_key_animation(i, idx);
					if (String(anim_name) == "[stop]" || !player->has_animation(anim_name)) {
						continue;
					}

					Ref<Animation> anim = player->get_animation(anim_name);

					double at_anim_pos = 0.0;

					switch (anim->get_loop_mode()) {
						case Animation::LOOP_NONE: {
							at_anim_pos = MIN((double)anim->get_length(), p_time - pos); //seek to end
						} break;

						case Animation::LOOP_LINEAR: {
							at_anim_pos = Math::fposmod(p_time - pos, (double)anim->get_length()); //seek to loop
						} break;

						case Animation::LOOP_PINGPONG: {
							at_anim_pos = Math::pingpong(p_time - pos, (double)anim->get_length());
						} break;

						default:
							break;
					}

					if (player->is_playing()) {
						player->seek(at_anim_pos);
						player->play(anim_name);
						nc->animation_playing = true;
						playing_caches.insert(nc);
					} else {
						player->set_assigned_animation(anim_name);
						player->seek(at_anim_pos, true);
					}
				} else {
					//find stuff to play
					List<int> to_play;
					if (p_started) {
						int first_key = a->track_find_key(i, p_prev_time, Animation::FIND_MODE_EXACT);
						if (first_key >= 0) {
							to_play.push_back(first_key);
						}
					}
					a->track_get_key_indices_in_range(i, p_time, p_delta, &to_play, p_looped_flag);
					if (to_play.size()) {
						int idx = to_play.back()->get();

						StringName anim_name = a->animation_track_get_key_animation(i, idx);
						if (String(anim_name) == "[stop]" || !player->has_animation(anim_name)) {
							if (playing_caches.has(nc)) {
								playing_caches.erase(nc);
								player->stop();
								nc->animation_playing = false;
							}
						} else {
							player->seek(0.0);
							player->play(anim_name);
							nc->animation_playing = true;
							playing_caches.insert(nc);
						}
					}
				}

			} break;
		}
	}
}

void AnimationPlayer::_animation_process_data(PlaybackData &cd, double p_delta, float p_blend, bool p_seeked, bool p_started) {
	double delta = p_delta * speed_scale * cd.speed_scale;
	double next_pos = cd.pos + delta;
	bool backwards = signbit(delta); // Negative zero means playing backwards too.

	real_t len = cd.from->animation->get_length();
	Animation::LoopedFlag looped_flag = Animation::LOOPED_FLAG_NONE;

	switch (cd.from->animation->get_loop_mode()) {
		case Animation::LOOP_NONE: {
			if (next_pos < 0) {
				next_pos = 0;
			} else if (next_pos > len) {
				next_pos = len;
			}
			delta = next_pos - cd.pos; // Fix delta (after determination of backwards because negative zero is lost here).
		} break;

		case Animation::LOOP_LINEAR: {
			if (next_pos < 0 && cd.pos >= 0) {
				looped_flag = Animation::LOOPED_FLAG_START;
			}
			if (next_pos > len && cd.pos <= len) {
				looped_flag = Animation::LOOPED_FLAG_END;
			}
			next_pos = Math::fposmod(next_pos, (double)len);
		} break;

		case Animation::LOOP_PINGPONG: {
			if (next_pos < 0 && cd.pos >= 0) {
				cd.speed_scale *= -1.0;
				looped_flag = Animation::LOOPED_FLAG_START;
			}
			if (next_pos > len && cd.pos <= len) {
				cd.speed_scale *= -1.0;
				looped_flag = Animation::LOOPED_FLAG_END;
			}
			next_pos = Math::pingpong(next_pos, (double)len);
		} break;

		default:
			break;
	}

	double prev_pos = cd.pos; // The animation may be changed during process, so it is safer that the state is changed before process.
	cd.pos = next_pos;

	AnimationData *prev_from = cd.from;
	_animation_process_animation(cd.from, prev_pos, cd.pos, delta, p_blend, &cd == &playback.current, p_seeked, p_started, looped_flag);

	// End detection.
	if (cd.from->animation->get_loop_mode() == Animation::LOOP_NONE) {
		if (prev_from != playback.current.from) {
			return; // Animation has been changed in the process (may be caused by method track), abort process.
		}
		if (!backwards && prev_pos <= len && next_pos == len) {
			// Playback finished.
			end_reached = true;
			end_notify = prev_pos < len; // Notify only if not already at the end.
		}
		if (backwards && prev_pos >= 0 && next_pos == 0) {
			// Playback finished.
			end_reached = true;
			end_notify = prev_pos > 0; // Notify only if not already at the beginning.
		}
	}
}

void AnimationPlayer::_animation_process2(double p_delta, bool p_started) {
	Playback &c = playback;

	accum_pass++;

	bool seeked = c.seeked; // The animation may be changed during process, so it is safer that the state is changed before process.

	if (p_delta != 0) {
		c.seeked = false;
	}

	float blend = 1.0; // First animation we play at 100% blend

	List<Blend>::Element *next = NULL;
	for (List<Blend>::Element *E = c.blend.front(); E; E = next) {
		Blend &b = E->get();
		// Note: There may be issues if an animation event triggers an animation change while this blend is active,
		// so it is best to use "deferred" calls instead of "immediate" for animation events that can trigger new animations.
		_animation_process_data(b.data, p_delta, blend, false, false);
		blend = 1.0 - b.blend_left / b.blend_time; // This is how much to blend the NEXT animation
		b.blend_left -= Math::absf(speed_scale * p_delta);
		next = E->next();
		if (b.blend_left < 0) {
			// If the blend of this has finished, we need to remove ALL the previous blends
			List<Blend>::Element *prev;
			while (E) {
				prev = E->prev();
				c.blend.erase(E);
				E = prev;
			}
		}
	}

	_animation_process_data(c.current, p_delta, blend, seeked, p_started);
}

void AnimationPlayer::_animation_update_transforms() {
	{
		Transform3D t;
		for (int i = 0; i < cache_update_size; i++) {
			TrackNodeCache *nc = cache_update[i];

			ERR_CONTINUE(nc->accum_pass != accum_pass);
#ifndef _3D_DISABLED
			if (nc->skeleton && nc->bone_idx >= 0) {
				if (nc->loc_used) {
					nc->skeleton->set_bone_pose_position(nc->bone_idx, nc->loc_accum);
				}
				if (nc->rot_used) {
					nc->skeleton->set_bone_pose_rotation(nc->bone_idx, nc->rot_accum);
				}
				if (nc->scale_used) {
					nc->skeleton->set_bone_pose_scale(nc->bone_idx, nc->scale_accum);
				}

			} else if (nc->node_blend_shape) {
				nc->node_blend_shape->set_blend_shape_value(nc->blend_shape_idx, nc->blend_shape_accum);
			} else if (nc->node_3d) {
				if (nc->loc_used) {
					nc->node_3d->set_position(nc->loc_accum);
				}
				if (nc->rot_used) {
					nc->node_3d->set_rotation(nc->rot_accum.get_euler());
				}
				if (nc->scale_used) {
					nc->node_3d->set_scale(nc->scale_accum);
				}
			}

#endif // _3D_DISABLED
		}
	}

	for (int i = 0; i < cache_update_prop_size; i++) {
		TrackNodeCache::PropertyAnim *pa = cache_update_prop[i];

		ERR_CONTINUE(pa->accum_pass != accum_pass);

		switch (pa->special) {
			case SP_NONE: {
				bool valid;
				pa->object->set_indexed(pa->subpath, pa->value_accum, &valid); //you are not speshul
#ifdef DEBUG_ENABLED

				if (!valid) {
					// Get subpath as string for printing the error
					// Cannot use `String::join(Vector<String>)` because this is a vector of StringName
					String key_debug;
					if (pa->subpath.size() > 0) {
						key_debug = pa->subpath[0];
						for (int subpath_index = 1; subpath_index < pa->subpath.size(); ++subpath_index) {
							key_debug += ".";
							key_debug += pa->subpath[subpath_index];
						}
					}
					ERR_PRINT("Failed setting key '" + key_debug +
							"' at time " + rtos(playback.current.pos) +
							" in Animation '" + get_current_animation() +
							"' at Node '" + get_path() +
							"', Track '" + String(pa->owner->path) +
							"'. Check if the property exists or the type of key is right for the property.");
				}
#endif

			} break;
			case SP_NODE2D_POS: {
#ifdef DEBUG_ENABLED
				if (pa->value_accum.get_type() != Variant::VECTOR2) {
					ERR_PRINT("Position key at time " + rtos(playback.current.pos) + " in Animation '" + get_current_animation() + "' at Node '" + get_path() + "', Track '" + String(pa->owner->path) + "' not of type Vector2()");
				}
#endif
				static_cast<Node2D *>(pa->object)->set_position(pa->value_accum);
			} break;
			case SP_NODE2D_ROT: {
#ifdef DEBUG_ENABLED
				if (pa->value_accum.is_num()) {
					ERR_PRINT("Rotation key at time " + rtos(playback.current.pos) + " in Animation '" + get_current_animation() + "' at Node '" + get_path() + "', Track '" + String(pa->owner->path) + "' not numerical");
				}
#endif

				static_cast<Node2D *>(pa->object)->set_rotation(Math::deg_to_rad((double)pa->value_accum));
			} break;
			case SP_NODE2D_SCALE: {
#ifdef DEBUG_ENABLED
				if (pa->value_accum.get_type() != Variant::VECTOR2) {
					ERR_PRINT("Scale key at time " + rtos(playback.current.pos) + " in Animation '" + get_current_animation() + "' at Node '" + get_path() + "', Track '" + String(pa->owner->path) + "' not of type Vector2()");
				}
#endif

				static_cast<Node2D *>(pa->object)->set_scale(pa->value_accum);
			} break;
		}
	}

	for (int i = 0; i < cache_update_bezier_size; i++) {
		TrackNodeCache::BezierAnim *ba = cache_update_bezier[i];

		ERR_CONTINUE(ba->accum_pass != accum_pass);
		ba->object->set_indexed(ba->bezier_property, ba->bezier_accum);
	}

	for (int i = 0; i < cache_update_audio_size; i++) {
		TrackNodeCache::AudioAnim *aa = cache_update_audio[i];

		ERR_CONTINUE(aa->accum_pass != accum_pass);

		// Audio ending process.
		LocalVector<int> erase_list;
		for (const KeyValue<int, TrackNodeCache::PlayingAudioStreamInfo> &K : aa->playing_streams) {
			TrackNodeCache::PlayingAudioStreamInfo pasi = K.value;

			bool stop = false;
			if (!aa->audio_stream_playback->is_stream_playing(pasi.index)) {
				stop = true;
			}
			if (!aa->loop) {
				if (!aa->backward) {
					if (aa->time < pasi.start) {
						stop = true;
					}
				} else {
					if (aa->time > pasi.start) {
						stop = true;
					}
				}
			}
			if (pasi.len > 0) {
				double len = 0.0;
				if (!aa->backward) {
					len = pasi.start > aa->time ? (aa->length - pasi.start) + aa->time : aa->time - pasi.start;
				} else {
					len = pasi.start < aa->time ? (aa->length - aa->time) + pasi.start : pasi.start - aa->time;
				}
				if (len > pasi.len) {
					stop = true;
				}
			}
			if (stop) {
				// Time to stop.
				aa->audio_stream_playback->stop_stream(pasi.index);
				erase_list.push_back(K.key);
			}
		}
		for (uint32_t erase_idx = 0; erase_idx < erase_list.size(); erase_idx++) {
			aa->playing_streams.erase(erase_list[erase_idx]);
		}
	}
}

void AnimationPlayer::_animation_process(double p_delta) {
	if (playback.current.from) {
		end_reached = false;
		end_notify = false;

		bool started = playback.started; // The animation may be changed during process, so it is safer that the state is changed before process.
		if (playback.started) {
			playback.started = false;
		}

		cache_update_size = 0;
		cache_update_prop_size = 0;
		cache_update_bezier_size = 0;
		cache_update_audio_size = 0;

		AnimationData *prev_from = playback.current.from;
		_animation_process2(p_delta, started);
		if (prev_from != playback.current.from) {
			return; // Animation has been changed in the process (may be caused by method track), abort process.
		}
		_animation_update_transforms();

		if (end_reached) {
			_clear_audio_streams();
			_stop_playing_caches(false);
			if (queued.size()) {
				String old = playback.assigned;
				play(queued.front()->get());
				String new_name = playback.assigned;
				queued.pop_front();
				if (end_notify) {
					emit_signal(SceneStringNames::get_singleton()->animation_changed, old, new_name);
				}
			} else {
				playing = false;
				_set_process(false);
				if (end_notify) {
					emit_signal(SceneStringNames::get_singleton()->animation_finished, playback.assigned);

					if (movie_quit_on_finish && OS::get_singleton()->has_feature("movie")) {
						print_line(vformat("Movie Maker mode is enabled. Quitting on animation finish as requested by: %s", get_path()));
						get_tree()->quit();
					}
				}
			}
			end_reached = false;
		}

	} else {
		_set_process(false);
	}
}

void AnimationPlayer::_animation_set_cache_update() {
	// Relatively fast function to update all animations.

	animation_set_update_pass++;
	bool clear_cache_needed = false;

	// Update changed and add otherwise
	for (const AnimationLibraryData &lib : animation_libraries) {
		for (const KeyValue<StringName, Ref<Animation>> &K : lib.library->animations) {
			StringName key = lib.name == StringName() ? K.key : StringName(String(lib.name) + "/" + String(K.key));
			if (!animation_set.has(key)) {
				AnimationData ad;
				ad.animation = K.value;
				ad.animation_library = lib.name;
				ad.name = key;
				ad.last_update = animation_set_update_pass;
				animation_set.insert(ad.name, ad);
			} else {
				AnimationData &ad = animation_set[key];
				if (ad.last_update != animation_set_update_pass) {
					// Was not updated, update. If the animation is duplicated, the second one will be ignored.
					if (ad.animation != K.value || ad.animation_library != lib.name) {
						// Animation changed, update and clear caches.
						clear_cache_needed = true;
						ad.animation = K.value;
						ad.animation_library = lib.name;
					}

					ad.last_update = animation_set_update_pass;
				}
			}
		}
	}

	// Check removed
	List<StringName> to_erase;
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.last_update != animation_set_update_pass) {
			// Was not updated, must be erased
			to_erase.push_back(E.key);
			clear_cache_needed = true;
		}
	}

	while (to_erase.size()) {
		animation_set.erase(to_erase.front()->get());
		to_erase.pop_front();
	}

	if (clear_cache_needed) {
		// If something was modified or removed, caches need to be cleared
		clear_caches();
	}

	emit_signal(SNAME("animation_list_changed"));
}

void AnimationPlayer::_animation_added(const StringName &p_name, const StringName &p_library) {
	_animation_set_cache_update();
}

void AnimationPlayer::_animation_removed(const StringName &p_name, const StringName &p_library) {
	StringName name = p_library == StringName() ? p_name : StringName(String(p_library) + "/" + String(p_name));

	if (!animation_set.has(name)) {
		return; // No need to update because not the one from the library being used.
	}

	_animation_set_cache_update();

	// Erase blends if needed
	List<BlendKey> to_erase;
	for (const KeyValue<BlendKey, double> &E : blend_times) {
		BlendKey bk = E.key;
		if (bk.from == name || bk.to == name) {
			to_erase.push_back(bk);
		}
	}

	while (to_erase.size()) {
		blend_times.erase(to_erase.front()->get());
		to_erase.pop_front();
	}
}

void AnimationPlayer::_rename_animation(const StringName &p_from_name, const StringName &p_to_name) {
	// Rename autoplay or blends if needed.
	List<BlendKey> to_erase;
	HashMap<BlendKey, double, BlendKey> to_insert;
	for (const KeyValue<BlendKey, double> &E : blend_times) {
		BlendKey bk = E.key;
		BlendKey new_bk = bk;
		bool erase = false;
		if (bk.from == p_from_name) {
			new_bk.from = p_to_name;
			erase = true;
		}
		if (bk.to == p_from_name) {
			new_bk.to = p_to_name;
			erase = true;
		}

		if (erase) {
			to_erase.push_back(bk);
			to_insert[new_bk] = E.value;
		}
	}

	while (to_erase.size()) {
		blend_times.erase(to_erase.front()->get());
		to_erase.pop_front();
	}

	while (to_insert.size()) {
		blend_times[to_insert.begin()->key] = to_insert.begin()->value;
		to_insert.remove(to_insert.begin());
	}

	if (autoplay == p_from_name) {
		autoplay = p_to_name;
	}
}

void AnimationPlayer::_animation_renamed(const StringName &p_name, const StringName &p_to_name, const StringName &p_library) {
	StringName from_name = p_library == StringName() ? p_name : StringName(String(p_library) + "/" + String(p_name));
	StringName to_name = p_library == StringName() ? p_to_name : StringName(String(p_library) + "/" + String(p_to_name));

	if (!animation_set.has(from_name)) {
		return; // No need to update because not the one from the library being used.
	}
	_animation_set_cache_update();

	_rename_animation(from_name, to_name);
}

Error AnimationPlayer::add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_animation_library) {
	ERR_FAIL_COND_V(p_animation_library.is_null(), ERR_INVALID_PARAMETER);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V_MSG(String(p_name).contains("/") || String(p_name).contains(":") || String(p_name).contains(",") || String(p_name).contains("["), ERR_INVALID_PARAMETER, "Invalid animation name: " + String(p_name) + ".");
#endif

	int insert_pos = 0;

	for (const AnimationLibraryData &lib : animation_libraries) {
		ERR_FAIL_COND_V_MSG(lib.name == p_name, ERR_ALREADY_EXISTS, "Can't add animation library twice with name: " + String(p_name));
		ERR_FAIL_COND_V_MSG(lib.library == p_animation_library, ERR_ALREADY_EXISTS, "Can't add animation library twice (adding as '" + p_name.operator String() + "', exists as '" + lib.name.operator String() + "'.");

		if (lib.name.operator String() >= p_name.operator String()) {
			break;
		}

		insert_pos++;
	}

	AnimationLibraryData ald;
	ald.name = p_name;
	ald.library = p_animation_library;

	animation_libraries.insert(insert_pos, ald);

	ald.library->connect(SNAME("animation_added"), callable_mp(this, &AnimationPlayer::_animation_added).bind(p_name));
	ald.library->connect(SNAME("animation_removed"), callable_mp(this, &AnimationPlayer::_animation_removed).bind(p_name));
	ald.library->connect(SNAME("animation_renamed"), callable_mp(this, &AnimationPlayer::_animation_renamed).bind(p_name));
	ald.library->connect(SNAME("animation_changed"), callable_mp(this, &AnimationPlayer::_animation_changed));

	_animation_set_cache_update();

	notify_property_list_changed();

	return OK;
}

void AnimationPlayer::remove_animation_library(const StringName &p_name) {
	int at_pos = -1;

	for (uint32_t i = 0; i < animation_libraries.size(); i++) {
		if (animation_libraries[i].name == p_name) {
			at_pos = i;
			break;
		}
	}

	ERR_FAIL_COND(at_pos == -1);

	animation_libraries[at_pos].library->disconnect(SNAME("animation_added"), callable_mp(this, &AnimationPlayer::_animation_added));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_removed"), callable_mp(this, &AnimationPlayer::_animation_removed));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_renamed"), callable_mp(this, &AnimationPlayer::_animation_renamed));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_changed"), callable_mp(this, &AnimationPlayer::_animation_changed));

	stop();

	animation_libraries.remove_at(at_pos);
	_animation_set_cache_update();

	notify_property_list_changed();
}

void AnimationPlayer::rename_animation_library(const StringName &p_name, const StringName &p_new_name) {
	if (p_name == p_new_name) {
		return;
	}
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_MSG(String(p_new_name).contains("/") || String(p_new_name).contains(":") || String(p_new_name).contains(",") || String(p_new_name).contains("["), "Invalid animation library name: " + String(p_new_name) + ".");
#endif

	bool found = false;
	for (AnimationLibraryData &lib : animation_libraries) {
		ERR_FAIL_COND_MSG(lib.name == p_new_name, "Can't rename animation library to another existing name: " + String(p_new_name));
		if (lib.name == p_name) {
			found = true;
			lib.name = p_new_name;
			// rename connections
			lib.library->disconnect(SNAME("animation_added"), callable_mp(this, &AnimationPlayer::_animation_added));
			lib.library->disconnect(SNAME("animation_removed"), callable_mp(this, &AnimationPlayer::_animation_removed));
			lib.library->disconnect(SNAME("animation_renamed"), callable_mp(this, &AnimationPlayer::_animation_renamed));

			lib.library->connect(SNAME("animation_added"), callable_mp(this, &AnimationPlayer::_animation_added).bind(p_new_name));
			lib.library->connect(SNAME("animation_removed"), callable_mp(this, &AnimationPlayer::_animation_removed).bind(p_new_name));
			lib.library->connect(SNAME("animation_renamed"), callable_mp(this, &AnimationPlayer::_animation_renamed).bind(p_new_name));

			for (const KeyValue<StringName, Ref<Animation>> &K : lib.library->animations) {
				StringName old_name = p_name == StringName() ? K.key : StringName(String(p_name) + "/" + String(K.key));
				StringName new_name = p_new_name == StringName() ? K.key : StringName(String(p_new_name) + "/" + String(K.key));
				_rename_animation(old_name, new_name);
			}
		}
	}

	ERR_FAIL_COND(!found);

	stop();

	animation_libraries.sort(); // Must keep alphabetical order.

	_animation_set_cache_update(); // Update cache.

	notify_property_list_changed();
}

bool AnimationPlayer::has_animation_library(const StringName &p_name) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		if (lib.name == p_name) {
			return true;
		}
	}

	return false;
}

Ref<AnimationLibrary> AnimationPlayer::get_animation_library(const StringName &p_name) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		if (lib.name == p_name) {
			return lib.library;
		}
	}
	ERR_FAIL_V(Ref<AnimationLibrary>());
}

TypedArray<StringName> AnimationPlayer::_get_animation_library_list() const {
	TypedArray<StringName> ret;
	for (const AnimationLibraryData &lib : animation_libraries) {
		ret.push_back(lib.name);
	}
	return ret;
}

void AnimationPlayer::get_animation_library_list(List<StringName> *p_libraries) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		p_libraries->push_back(lib.name);
	}
}

bool AnimationPlayer::has_animation(const StringName &p_name) const {
	return animation_set.has(p_name);
}

Ref<Animation> AnimationPlayer::get_animation(const StringName &p_name) const {
	ERR_FAIL_COND_V_MSG(!animation_set.has(p_name), Ref<Animation>(), vformat("Animation not found: \"%s\".", p_name));

	const AnimationData &anim_data = animation_set[p_name];

	return anim_data.animation;
}

void AnimationPlayer::get_animation_list(List<StringName> *p_animations) const {
	List<String> anims;

	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		anims.push_back(E.key);
	}

	anims.sort();

	for (const String &E : anims) {
		p_animations->push_back(E);
	}
}

void AnimationPlayer::set_blend_time(const StringName &p_animation1, const StringName &p_animation2, double p_time) {
	ERR_FAIL_COND_MSG(!animation_set.has(p_animation1), vformat("Animation not found: %s.", p_animation1));
	ERR_FAIL_COND_MSG(!animation_set.has(p_animation2), vformat("Animation not found: %s.", p_animation2));
	ERR_FAIL_COND_MSG(p_time < 0, "Blend time cannot be smaller than 0.");

	BlendKey bk;
	bk.from = p_animation1;
	bk.to = p_animation2;
	if (p_time == 0) {
		blend_times.erase(bk);
	} else {
		blend_times[bk] = p_time;
	}
}

double AnimationPlayer::get_blend_time(const StringName &p_animation1, const StringName &p_animation2) const {
	BlendKey bk;
	bk.from = p_animation1;
	bk.to = p_animation2;

	if (blend_times.has(bk)) {
		return blend_times[bk];
	} else {
		return 0;
	}
}

void AnimationPlayer::queue(const StringName &p_name) {
	if (!is_playing()) {
		play(p_name);
	} else {
		queued.push_back(p_name);
	}
}

Vector<String> AnimationPlayer::get_queue() {
	Vector<String> ret;
	for (const StringName &E : queued) {
		ret.push_back(E);
	}

	return ret;
}

void AnimationPlayer::clear_queue() {
	queued.clear();
}

void AnimationPlayer::play_backwards(const StringName &p_name, double p_custom_blend) {
	play(p_name, p_custom_blend, -1, true);
}

void AnimationPlayer::play(const StringName &p_name, double p_custom_blend, float p_custom_scale, bool p_from_end) {
	StringName name = p_name;

	if (String(name) == "") {
		name = playback.assigned;
	}

	ERR_FAIL_COND_MSG(!animation_set.has(name), vformat("Animation not found: %s.", name));

	Playback &c = playback;

	if (c.current.from) {
		double blend_time = 0.0;
		// find if it can blend
		BlendKey bk;
		bk.from = c.current.from->name;
		bk.to = name;

		if (p_custom_blend >= 0) {
			blend_time = p_custom_blend;
		} else if (blend_times.has(bk)) {
			blend_time = blend_times[bk];
		} else {
			bk.from = "*";
			if (blend_times.has(bk)) {
				blend_time = blend_times[bk];
			} else {
				bk.from = c.current.from->name;
				bk.to = "*";

				if (blend_times.has(bk)) {
					blend_time = blend_times[bk];
				}
			}
		}

		if (p_custom_blend < 0 && blend_time == 0 && default_blend_time) {
			blend_time = default_blend_time;
		}
		if (blend_time > 0) {
			Blend b;
			b.data = c.current;
			b.blend_time = b.blend_left = blend_time;
			c.blend.push_back(b);
		} else {
			c.blend.clear();
		}
	}

	if (get_current_animation() != p_name) {
		_clear_audio_streams();
		_stop_playing_caches(false);
	}

	c.current.from = &animation_set[name];

	if (c.assigned != name) { // reset
		c.current.pos = p_from_end ? c.current.from->animation->get_length() : 0;
	} else {
		if (p_from_end && c.current.pos == 0) {
			// Animation reset BUT played backwards, set position to the end
			c.current.pos = c.current.from->animation->get_length();
		} else if (!p_from_end && c.current.pos == c.current.from->animation->get_length()) {
			// Animation resumed but already ended, set position to the beginning
			c.current.pos = 0;
		}
	}

	c.current.speed_scale = p_custom_scale;
	c.assigned = name;
	c.seeked = false;
	c.started = true;

	if (!end_reached) {
		queued.clear();
	}
	_set_process(true); // always process when starting an animation
	playing = true;

	emit_signal(SceneStringNames::get_singleton()->animation_started, c.assigned);

	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint()) {
		return; // no next in this case
	}

	StringName next = animation_get_next(p_name);
	if (next != StringName() && animation_set.has(next)) {
		queue(next);
	}
}

bool AnimationPlayer::is_playing() const {
	return playing;
}

void AnimationPlayer::set_current_animation(const String &p_anim) {
	if (p_anim == "[stop]" || p_anim.is_empty()) {
		stop();
	} else if (!is_playing()) {
		play(p_anim);
	} else if (playback.assigned != p_anim) {
		float speed = playback.current.speed_scale;
		play(p_anim, -1.0, speed, signbit(speed));
	} else {
		// Same animation, do not replay from start
	}
}

String AnimationPlayer::get_current_animation() const {
	return (is_playing() ? playback.assigned : "");
}

void AnimationPlayer::set_assigned_animation(const String &p_anim) {
	if (is_playing()) {
		float speed = playback.current.speed_scale;
		play(p_anim, -1.0, speed, signbit(speed));
	} else {
		ERR_FAIL_COND_MSG(!animation_set.has(p_anim), vformat("Animation not found: %s.", p_anim));
		playback.current.pos = 0;
		playback.current.from = &animation_set[p_anim];
		playback.assigned = p_anim;
	}
}

String AnimationPlayer::get_assigned_animation() const {
	return playback.assigned;
}

void AnimationPlayer::pause() {
	_stop_internal(false, false);
}

void AnimationPlayer::stop(bool p_keep_state) {
	_stop_internal(true, p_keep_state);
}

void AnimationPlayer::set_speed_scale(float p_speed) {
	speed_scale = p_speed;
}

float AnimationPlayer::get_speed_scale() const {
	return speed_scale;
}

float AnimationPlayer::get_playing_speed() const {
	if (!playing) {
		return 0;
	}
	return speed_scale * playback.current.speed_scale;
}

void AnimationPlayer::seek(double p_time, bool p_update) {
	playback.current.pos = p_time;

	if (!playback.current.from) {
		if (playback.assigned) {
			ERR_FAIL_COND_MSG(!animation_set.has(playback.assigned), vformat("Animation not found: %s.", playback.assigned));
			playback.current.from = &animation_set[playback.assigned];
		}
		if (!playback.current.from) {
			return; // There is no animation.
		}
	}

	playback.seeked = true;
	if (p_update) {
		_animation_process(0);
	}
}

void AnimationPlayer::seek_delta(double p_time, double p_delta) {
	playback.current.pos = p_time - p_delta;

	if (!playback.current.from) {
		if (playback.assigned) {
			ERR_FAIL_COND_MSG(!animation_set.has(playback.assigned), vformat("Animation not found: %s.", playback.assigned));
			playback.current.from = &animation_set[playback.assigned];
		}
		if (!playback.current.from) {
			return; // There is no animation.
		}
	}

	if (speed_scale != 0.0) {
		p_delta /= speed_scale;
	}
	_animation_process(p_delta);
}

bool AnimationPlayer::is_valid() const {
	return (playback.current.from);
}

double AnimationPlayer::get_current_animation_position() const {
	ERR_FAIL_COND_V_MSG(!playback.current.from, 0, "AnimationPlayer has no current animation");
	return playback.current.pos;
}

double AnimationPlayer::get_current_animation_length() const {
	ERR_FAIL_COND_V_MSG(!playback.current.from, 0, "AnimationPlayer has no current animation");
	return playback.current.from->animation->get_length();
}

void AnimationPlayer::_animation_changed(const StringName &p_name) {
	clear_caches();
	if (is_playing()) {
		playback.seeked = true; //need to restart stuff, like audio
	}
}

void AnimationPlayer::_stop_playing_caches(bool p_reset) {
	for (TrackNodeCache *E : playing_caches) {
		if (E->node && E->audio_playing) {
			E->node->call(SNAME("stop"));
		}
		if (E->node && E->animation_playing) {
			AnimationPlayer *player = Object::cast_to<AnimationPlayer>(E->node);
			if (!player) {
				continue;
			}

			if (p_reset) {
				player->stop();
			} else {
				player->pause();
			}
		}
	}

	playing_caches.clear();
}

void AnimationPlayer::_node_removed(Node *p_node) {
	clear_caches(); // nodes contained here are being removed, clear the caches
}

void AnimationPlayer::clear_caches() {
	_clear_audio_streams();
	_stop_playing_caches(true);

	node_cache_map.clear();

	for (KeyValue<StringName, AnimationData> &E : animation_set) {
		E.value.node_cache.clear();
	}

	cache_update_size = 0;
	cache_update_prop_size = 0;
	cache_update_bezier_size = 0;
	cache_update_audio_size = 0;

	emit_signal(SNAME("caches_cleared"));
}

void AnimationPlayer::_clear_audio_streams() {
	for (int i = 0; i < playing_audio_stream_players.size(); i++) {
		playing_audio_stream_players[i]->call(SNAME("stop"));
		playing_audio_stream_players[i]->call(SNAME("set_stream"), Ref<AudioStream>());
	}
	playing_audio_stream_players.clear();
}

void AnimationPlayer::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;
	_set_process(processing, true);
}

bool AnimationPlayer::is_active() const {
	return active;
}

StringName AnimationPlayer::find_animation(const Ref<Animation> &p_animation) const {
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.animation == p_animation) {
			return E.key;
		}
	}

	return StringName();
}

StringName AnimationPlayer::find_animation_library(const Ref<Animation> &p_animation) const {
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.animation == p_animation) {
			return E.value.animation_library;
		}
	}
	return StringName();
}

void AnimationPlayer::set_autoplay(const String &p_name) {
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		WARN_PRINT("Setting autoplay after the node has been added to the scene has no effect.");
	}

	autoplay = p_name;
}

String AnimationPlayer::get_autoplay() const {
	return autoplay;
}

void AnimationPlayer::set_reset_on_save_enabled(bool p_enabled) {
	reset_on_save = p_enabled;
}

bool AnimationPlayer::is_reset_on_save_enabled() const {
	return reset_on_save;
}

void AnimationPlayer::set_process_callback(AnimationProcessCallback p_mode) {
	if (process_callback == p_mode) {
		return;
	}

	bool pr = processing;
	if (pr) {
		_set_process(false);
	}
	process_callback = p_mode;
	if (pr) {
		_set_process(true);
	}
}

AnimationPlayer::AnimationProcessCallback AnimationPlayer::get_process_callback() const {
	return process_callback;
}

void AnimationPlayer::set_method_call_mode(AnimationMethodCallMode p_mode) {
	method_call_mode = p_mode;
}

AnimationPlayer::AnimationMethodCallMode AnimationPlayer::get_method_call_mode() const {
	return method_call_mode;
}

void AnimationPlayer::set_audio_max_polyphony(int p_audio_max_polyphony) {
	ERR_FAIL_COND(p_audio_max_polyphony < 0 || p_audio_max_polyphony > 128);
	audio_max_polyphony = p_audio_max_polyphony;
}

int AnimationPlayer::get_audio_max_polyphony() const {
	return audio_max_polyphony;
}

void AnimationPlayer::set_movie_quit_on_finish_enabled(bool p_enabled) {
	movie_quit_on_finish = p_enabled;
}

bool AnimationPlayer::is_movie_quit_on_finish_enabled() const {
	return movie_quit_on_finish;
}

void AnimationPlayer::_set_process(bool p_process, bool p_force) {
	if (processing == p_process && !p_force) {
		return;
	}

	switch (process_callback) {
		case ANIMATION_PROCESS_PHYSICS:
			set_physics_process_internal(p_process && active);
			break;
		case ANIMATION_PROCESS_IDLE:
			set_process_internal(p_process && active);
			break;
		case ANIMATION_PROCESS_MANUAL:
			break;
	}

	processing = p_process;
}

void AnimationPlayer::_stop_internal(bool p_reset, bool p_keep_state) {
	_clear_audio_streams();
	_stop_playing_caches(p_reset);
	Playback &c = playback;
	c.blend.clear();
	if (p_reset) {
		if (p_keep_state) {
			c.current.pos = 0;
		} else {
			is_stopping = true;
			seek(0, true);
			is_stopping = false;
		}
		c.current.from = nullptr;
		c.current.speed_scale = 1;
	}
	_set_process(false);
	queued.clear();
	playing = false;
}

void AnimationPlayer::animation_set_next(const StringName &p_animation, const StringName &p_next) {
	ERR_FAIL_COND_MSG(!animation_set.has(p_animation), vformat("Animation not found: %s.", p_animation));
	animation_set[p_animation].next = p_next;
}

StringName AnimationPlayer::animation_get_next(const StringName &p_animation) const {
	if (!animation_set.has(p_animation)) {
		return StringName();
	}
	return animation_set[p_animation].next;
}

void AnimationPlayer::set_default_blend_time(double p_default) {
	default_blend_time = p_default;
}

double AnimationPlayer::get_default_blend_time() const {
	return default_blend_time;
}

void AnimationPlayer::set_root(const NodePath &p_root) {
	root = p_root;
	clear_caches();
}

NodePath AnimationPlayer::get_root() const {
	return root;
}

void AnimationPlayer::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	String pf = p_function;
	if (p_idx == 0 && (p_function == "play" || p_function == "play_backwards" || p_function == "has_animation" || p_function == "queue")) {
		List<StringName> al;
		get_animation_list(&al);
		for (const StringName &name : al) {
			r_options->push_back(String(name).quote());
		}
	}
	Node::get_argument_options(p_function, p_idx, r_options);
}

#ifdef TOOLS_ENABLED
Ref<AnimatedValuesBackup> AnimationPlayer::backup_animated_values(Node *p_root_override) {
	Ref<AnimatedValuesBackup> backup;
	if (!playback.current.from) {
		return backup;
	}

	_ensure_node_caches(playback.current.from, p_root_override);

	backup.instantiate();
	for (int i = 0; i < playback.current.from->node_cache.size(); i++) {
		TrackNodeCache *nc = playback.current.from->node_cache[i];
		if (!nc) {
			continue;
		}

		if (nc->skeleton) {
			if (nc->bone_idx == -1) {
				continue;
			}

			AnimatedValuesBackup::Entry entry;
			entry.object = nc->skeleton;
			entry.bone_idx = nc->bone_idx;
			Array arr;
			arr.resize(3);
			arr[0] = nc->skeleton->get_bone_pose_position(nc->bone_idx);
			arr[1] = nc->skeleton->get_bone_pose_rotation(nc->bone_idx);
			arr[2] = nc->skeleton->get_bone_pose_scale(nc->bone_idx);
			entry.value = nc;
			backup->entries.push_back(entry);
		} else {
			if (nc->node_3d) {
				AnimatedValuesBackup::Entry entry;
				entry.object = nc->node_3d;
				entry.subpath.push_back("transform");
				entry.value = nc->node_3d->get_transform();
				entry.bone_idx = -1;
				backup->entries.push_back(entry);
			} else {
				for (const KeyValue<StringName, TrackNodeCache::PropertyAnim> &E : nc->property_anim) {
					AnimatedValuesBackup::Entry entry;
					entry.object = E.value.object;
					entry.subpath = E.value.subpath;
					bool valid;
					entry.value = E.value.object->get_indexed(E.value.subpath, &valid);
					entry.bone_idx = -1;
					if (valid) {
						backup->entries.push_back(entry);
					}
				}
			}
		}
	}

	return backup;
}

Ref<AnimatedValuesBackup> AnimationPlayer::apply_reset(bool p_user_initiated) {
	ERR_FAIL_COND_V(!can_apply_reset(), Ref<AnimatedValuesBackup>());

	Ref<Animation> reset_anim = animation_set[SceneStringNames::get_singleton()->RESET].animation;
	ERR_FAIL_COND_V(reset_anim.is_null(), Ref<AnimatedValuesBackup>());

	Node *root_node = get_node_or_null(root);
	ERR_FAIL_NULL_V(root_node, Ref<AnimatedValuesBackup>());

	AnimationPlayer *aux_player = memnew(AnimationPlayer);
	EditorNode::get_singleton()->add_child(aux_player);
	Ref<AnimationLibrary> al;
	al.instantiate();
	al->add_animation(SceneStringNames::get_singleton()->RESET, reset_anim);
	aux_player->add_animation_library("", al);
	aux_player->set_assigned_animation(SceneStringNames::get_singleton()->RESET);
	// Forcing the use of the original root because the scene where original player belongs may be not the active one
	Ref<AnimatedValuesBackup> old_values = aux_player->backup_animated_values(get_node(get_root()));
	aux_player->seek(0.0f, true);
	aux_player->queue_free();

	if (p_user_initiated) {
		Ref<AnimatedValuesBackup> new_values = aux_player->backup_animated_values();
		old_values->restore();

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Animation Apply Reset"));
		ur->add_do_method(new_values.ptr(), "restore");
		ur->add_undo_method(old_values.ptr(), "restore");
		ur->commit_action();
	}

	return old_values;
}

bool AnimationPlayer::can_apply_reset() const {
	return has_animation(SceneStringNames::get_singleton()->RESET) && playback.assigned != SceneStringNames::get_singleton()->RESET;
}
#endif // TOOLS_ENABLED

void AnimationPlayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_animation_library", "name", "library"), &AnimationPlayer::add_animation_library);
	ClassDB::bind_method(D_METHOD("remove_animation_library", "name"), &AnimationPlayer::remove_animation_library);
	ClassDB::bind_method(D_METHOD("rename_animation_library", "name", "newname"), &AnimationPlayer::rename_animation_library);
	ClassDB::bind_method(D_METHOD("has_animation_library", "name"), &AnimationPlayer::has_animation_library);
	ClassDB::bind_method(D_METHOD("get_animation_library", "name"), &AnimationPlayer::get_animation_library);
	ClassDB::bind_method(D_METHOD("get_animation_library_list"), &AnimationPlayer::_get_animation_library_list);

	ClassDB::bind_method(D_METHOD("has_animation", "name"), &AnimationPlayer::has_animation);
	ClassDB::bind_method(D_METHOD("get_animation", "name"), &AnimationPlayer::get_animation);
	ClassDB::bind_method(D_METHOD("get_animation_list"), &AnimationPlayer::_get_animation_list);

	ClassDB::bind_method(D_METHOD("animation_set_next", "anim_from", "anim_to"), &AnimationPlayer::animation_set_next);
	ClassDB::bind_method(D_METHOD("animation_get_next", "anim_from"), &AnimationPlayer::animation_get_next);

	ClassDB::bind_method(D_METHOD("set_blend_time", "anim_from", "anim_to", "sec"), &AnimationPlayer::set_blend_time);
	ClassDB::bind_method(D_METHOD("get_blend_time", "anim_from", "anim_to"), &AnimationPlayer::get_blend_time);

	ClassDB::bind_method(D_METHOD("set_default_blend_time", "sec"), &AnimationPlayer::set_default_blend_time);
	ClassDB::bind_method(D_METHOD("get_default_blend_time"), &AnimationPlayer::get_default_blend_time);

	ClassDB::bind_method(D_METHOD("play", "name", "custom_blend", "custom_speed", "from_end"), &AnimationPlayer::play, DEFVAL(""), DEFVAL(-1), DEFVAL(1.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("play_backwards", "name", "custom_blend"), &AnimationPlayer::play_backwards, DEFVAL(""), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("pause"), &AnimationPlayer::pause);
	ClassDB::bind_method(D_METHOD("stop", "keep_state"), &AnimationPlayer::stop, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_playing"), &AnimationPlayer::is_playing);

	ClassDB::bind_method(D_METHOD("set_current_animation", "anim"), &AnimationPlayer::set_current_animation);
	ClassDB::bind_method(D_METHOD("get_current_animation"), &AnimationPlayer::get_current_animation);
	ClassDB::bind_method(D_METHOD("set_assigned_animation", "anim"), &AnimationPlayer::set_assigned_animation);
	ClassDB::bind_method(D_METHOD("get_assigned_animation"), &AnimationPlayer::get_assigned_animation);
	ClassDB::bind_method(D_METHOD("queue", "name"), &AnimationPlayer::queue);
	ClassDB::bind_method(D_METHOD("get_queue"), &AnimationPlayer::get_queue);
	ClassDB::bind_method(D_METHOD("clear_queue"), &AnimationPlayer::clear_queue);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &AnimationPlayer::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationPlayer::is_active);

	ClassDB::bind_method(D_METHOD("set_speed_scale", "speed"), &AnimationPlayer::set_speed_scale);
	ClassDB::bind_method(D_METHOD("get_speed_scale"), &AnimationPlayer::get_speed_scale);
	ClassDB::bind_method(D_METHOD("get_playing_speed"), &AnimationPlayer::get_playing_speed);

	ClassDB::bind_method(D_METHOD("set_autoplay", "name"), &AnimationPlayer::set_autoplay);
	ClassDB::bind_method(D_METHOD("get_autoplay"), &AnimationPlayer::get_autoplay);

	ClassDB::bind_method(D_METHOD("set_reset_on_save_enabled", "enabled"), &AnimationPlayer::set_reset_on_save_enabled);
	ClassDB::bind_method(D_METHOD("is_reset_on_save_enabled"), &AnimationPlayer::is_reset_on_save_enabled);

	ClassDB::bind_method(D_METHOD("set_root", "path"), &AnimationPlayer::set_root);
	ClassDB::bind_method(D_METHOD("get_root"), &AnimationPlayer::get_root);

	ClassDB::bind_method(D_METHOD("find_animation", "animation"), &AnimationPlayer::find_animation);
	ClassDB::bind_method(D_METHOD("find_animation_library", "animation"), &AnimationPlayer::find_animation_library);

	ClassDB::bind_method(D_METHOD("clear_caches"), &AnimationPlayer::clear_caches);

	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &AnimationPlayer::set_process_callback);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &AnimationPlayer::get_process_callback);

	ClassDB::bind_method(D_METHOD("set_method_call_mode", "mode"), &AnimationPlayer::set_method_call_mode);
	ClassDB::bind_method(D_METHOD("get_method_call_mode"), &AnimationPlayer::get_method_call_mode);

	ClassDB::bind_method(D_METHOD("set_audio_max_polyphony", "max_polyphony"), &AnimationPlayer::set_audio_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_audio_max_polyphony"), &AnimationPlayer::get_audio_max_polyphony);

	ClassDB::bind_method(D_METHOD("set_movie_quit_on_finish_enabled", "enabled"), &AnimationPlayer::set_movie_quit_on_finish_enabled);
	ClassDB::bind_method(D_METHOD("is_movie_quit_on_finish_enabled"), &AnimationPlayer::is_movie_quit_on_finish_enabled);

	ClassDB::bind_method(D_METHOD("get_current_animation_position"), &AnimationPlayer::get_current_animation_position);
	ClassDB::bind_method(D_METHOD("get_current_animation_length"), &AnimationPlayer::get_current_animation_length);

	ClassDB::bind_method(D_METHOD("seek", "seconds", "update"), &AnimationPlayer::seek, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("advance", "delta"), &AnimationPlayer::advance);

	GDVIRTUAL_BIND(_post_process_key_value, "animation", "track", "value", "object", "object_idx");

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_node"), "set_root", "get_root");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "current_animation", PROPERTY_HINT_ENUM, "", PROPERTY_USAGE_EDITOR), "set_current_animation", "get_current_animation");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "assigned_animation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_assigned_animation", "get_assigned_animation");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "autoplay", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_autoplay", "get_autoplay");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "reset_on_save", PROPERTY_HINT_NONE, ""), "set_reset_on_save_enabled", "is_reset_on_save_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "current_animation_length", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_current_animation_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "current_animation_position", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_current_animation_position");

	ADD_GROUP("Playback Options", "playback_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "playback_process_mode", PROPERTY_HINT_ENUM, "Physics,Idle,Manual"), "set_process_callback", "get_process_callback");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "playback_default_blend_time", PROPERTY_HINT_RANGE, "0,4096,0.01,suffix:s"), "set_default_blend_time", "get_default_blend_time");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "playback_active", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed_scale", PROPERTY_HINT_RANGE, "-64,64,0.01"), "set_speed_scale", "get_speed_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "method_call_mode", PROPERTY_HINT_ENUM, "Deferred,Immediate"), "set_method_call_mode", "get_method_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "audio_max_polyphony", PROPERTY_HINT_RANGE, "1,127,1"), "set_audio_max_polyphony", "get_audio_max_polyphony");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "movie_quit_on_finish"), "set_movie_quit_on_finish_enabled", "is_movie_quit_on_finish_enabled");

	ADD_SIGNAL(MethodInfo("animation_finished", PropertyInfo(Variant::STRING_NAME, "anim_name")));
	ADD_SIGNAL(MethodInfo("animation_changed", PropertyInfo(Variant::STRING_NAME, "old_name"), PropertyInfo(Variant::STRING_NAME, "new_name")));
	ADD_SIGNAL(MethodInfo("animation_started", PropertyInfo(Variant::STRING_NAME, "anim_name")));
	ADD_SIGNAL(MethodInfo("animation_list_changed"));
	ADD_SIGNAL(MethodInfo("animation_libraries_updated"));
	ADD_SIGNAL(MethodInfo("caches_cleared"));

	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_MANUAL);

	BIND_ENUM_CONSTANT(ANIMATION_METHOD_CALL_DEFERRED);
	BIND_ENUM_CONSTANT(ANIMATION_METHOD_CALL_IMMEDIATE);
}

AnimationPlayer::AnimationPlayer() {
	root = SceneStringNames::get_singleton()->path_pp;
}

AnimationPlayer::~AnimationPlayer() {
}
