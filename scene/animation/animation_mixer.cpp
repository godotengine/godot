/**************************************************************************/
/*  animation_mixer.cpp                                                   */
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

#include "animation_mixer.h"

#include "core/config/engine.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/scene_string_names.h"
#include "servers/audio/audio_stream.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#endif // TOOLS_ENABLED

bool AnimationMixer::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

#ifndef DISABLE_DEPRECATED
	if (name.begins_with("anims/")) {
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
#else
	if (name.begins_with("libraries")) {
#endif // DISABLE_DEPRECATED
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
		emit_signal(SNAME("animation_libraries_updated"));

	} else {
		return false;
	}

	return true;
}

bool AnimationMixer::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name.begins_with("libraries")) {
		Dictionary d;
		for (const AnimationLibraryData &lib : animation_libraries) {
			d[lib.name] = lib.library;
		}
		r_ret = d;
	} else {
		return false;
	}

	return true;
}

void AnimationMixer::_get_property_list(List<PropertyInfo> *p_list) const {
	List<PropertyInfo> anim_names;
	anim_names.push_back(PropertyInfo(Variant::DICTIONARY, PNAME("libraries")));
	for (const PropertyInfo &E : anim_names) {
		p_list->push_back(E);
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void AnimationMixer::_validate_property(PropertyInfo &p_property) const {
#ifdef TOOLS_ENABLED
	if (editing && (p_property.name == "active" || p_property.name == "deterministic" || p_property.name == "root_motion_track")) {
		p_property.usage |= PROPERTY_USAGE_READ_ONLY;
	}
#endif // TOOLS_ENABLED
}

/* -------------------------------------------- */
/* -- Data lists ------------------------------ */
/* -------------------------------------------- */

void AnimationMixer::_animation_set_cache_update() {
	// Relatively fast function to update all animations.
	animation_set_update_pass++;
	bool clear_cache_needed = false;

	// Update changed and add otherwise.
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

	// Check removed.
	List<StringName> to_erase;
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.last_update != animation_set_update_pass) {
			// Was not updated, must be erased.
			to_erase.push_back(E.key);
			clear_cache_needed = true;
		}
	}

	while (to_erase.size()) {
		animation_set.erase(to_erase.front()->get());
		to_erase.pop_front();
	}

	if (clear_cache_needed) {
		// If something was modified or removed, caches need to be cleared.
		_clear_caches();
	}

	emit_signal(SNAME("animation_list_changed"));
}

void AnimationMixer::_animation_added(const StringName &p_name, const StringName &p_library) {
	_animation_set_cache_update();
}

void AnimationMixer::_animation_removed(const StringName &p_name, const StringName &p_library) {
	StringName name = p_library == StringName() ? p_name : StringName(String(p_library) + "/" + String(p_name));

	if (!animation_set.has(name)) {
		return; // No need to update because not the one from the library being used.
	}

	_animation_set_cache_update();

	_remove_animation(name);
}

void AnimationMixer::_animation_renamed(const StringName &p_name, const StringName &p_to_name, const StringName &p_library) {
	StringName from_name = p_library == StringName() ? p_name : StringName(String(p_library) + "/" + String(p_name));
	StringName to_name = p_library == StringName() ? p_to_name : StringName(String(p_library) + "/" + String(p_to_name));

	if (!animation_set.has(from_name)) {
		return; // No need to update because not the one from the library being used.
	}
	_animation_set_cache_update();

	_rename_animation(from_name, to_name);
}

void AnimationMixer::_animation_changed(const StringName &p_name) {
	_clear_caches();
}

void AnimationMixer::_set_active(bool p_active) {
	//
}

void AnimationMixer::_remove_animation(const StringName &p_name) {
	//
}

void AnimationMixer::_rename_animation(const StringName &p_from_name, const StringName &p_to_name) {
	//
}

TypedArray<StringName> AnimationMixer::_get_animation_library_list() const {
	TypedArray<StringName> ret;
	for (const AnimationLibraryData &lib : animation_libraries) {
		ret.push_back(lib.name);
	}
	return ret;
}

void AnimationMixer::get_animation_library_list(List<StringName> *p_libraries) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		p_libraries->push_back(lib.name);
	}
}

Ref<AnimationLibrary> AnimationMixer::get_animation_library(const StringName &p_name) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		if (lib.name == p_name) {
			return lib.library;
		}
	}
	ERR_FAIL_V(Ref<AnimationLibrary>());
}

bool AnimationMixer::has_animation_library(const StringName &p_name) const {
	for (const AnimationLibraryData &lib : animation_libraries) {
		if (lib.name == p_name) {
			return true;
		}
	}

	return false;
}

StringName AnimationMixer::find_animation_library(const Ref<Animation> &p_animation) const {
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.animation == p_animation) {
			return E.value.animation_library;
		}
	}
	return StringName();
}

Error AnimationMixer::add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_animation_library) {
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

	ald.library->connect(SNAME("animation_added"), callable_mp(this, &AnimationMixer::_animation_added).bind(p_name));
	ald.library->connect(SNAME("animation_removed"), callable_mp(this, &AnimationMixer::_animation_removed).bind(p_name));
	ald.library->connect(SNAME("animation_renamed"), callable_mp(this, &AnimationMixer::_animation_renamed).bind(p_name));
	ald.library->connect(SNAME("animation_changed"), callable_mp(this, &AnimationMixer::_animation_changed));

	_animation_set_cache_update();

	notify_property_list_changed();

	return OK;
}

void AnimationMixer::remove_animation_library(const StringName &p_name) {
	int at_pos = -1;

	for (uint32_t i = 0; i < animation_libraries.size(); i++) {
		if (animation_libraries[i].name == p_name) {
			at_pos = i;
			break;
		}
	}

	ERR_FAIL_COND(at_pos == -1);

	animation_libraries[at_pos].library->disconnect(SNAME("animation_added"), callable_mp(this, &AnimationMixer::_animation_added));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_removed"), callable_mp(this, &AnimationMixer::_animation_removed));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_renamed"), callable_mp(this, &AnimationMixer::_animation_renamed));
	animation_libraries[at_pos].library->disconnect(SNAME("animation_changed"), callable_mp(this, &AnimationMixer::_animation_changed));

	animation_libraries.remove_at(at_pos);
	_animation_set_cache_update();

	notify_property_list_changed();
}

void AnimationMixer::rename_animation_library(const StringName &p_name, const StringName &p_new_name) {
	if (p_name == p_new_name) {
		return;
	}
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_MSG(String(p_new_name).contains("/") || String(p_new_name).contains(":") || String(p_new_name).contains(",") || String(p_new_name).contains("["), "Invalid animation library name: " + String(p_new_name) + ".");
#endif

	bool found = false;
	for (AnimationLibraryData &lib : animation_libraries) {
		ERR_FAIL_COND_MSG(lib.name == p_new_name, "Can't rename animation library to another existing name: " + String(p_new_name) + ".");
		if (lib.name == p_name) {
			found = true;
			lib.name = p_new_name;
			// rename connections
			lib.library->disconnect(SNAME("animation_added"), callable_mp(this, &AnimationMixer::_animation_added));
			lib.library->disconnect(SNAME("animation_removed"), callable_mp(this, &AnimationMixer::_animation_removed));
			lib.library->disconnect(SNAME("animation_renamed"), callable_mp(this, &AnimationMixer::_animation_renamed));

			lib.library->connect(SNAME("animation_added"), callable_mp(this, &AnimationMixer::_animation_added).bind(p_new_name));
			lib.library->connect(SNAME("animation_removed"), callable_mp(this, &AnimationMixer::_animation_removed).bind(p_new_name));
			lib.library->connect(SNAME("animation_renamed"), callable_mp(this, &AnimationMixer::_animation_renamed).bind(p_new_name));

			for (const KeyValue<StringName, Ref<Animation>> &K : lib.library->animations) {
				StringName old_name = p_name == StringName() ? K.key : StringName(String(p_name) + "/" + String(K.key));
				StringName new_name = p_new_name == StringName() ? K.key : StringName(String(p_new_name) + "/" + String(K.key));
				_rename_animation(old_name, new_name);
			}
		}
	}

	ERR_FAIL_COND(!found);

	animation_libraries.sort(); // Must keep alphabetical order.

	_animation_set_cache_update(); // Update cache.

	notify_property_list_changed();
}

void AnimationMixer::get_animation_list(List<StringName> *p_animations) const {
	List<String> anims;
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		anims.push_back(E.key);
	}
	anims.sort();
	for (const String &E : anims) {
		p_animations->push_back(E);
	}
}

Ref<Animation> AnimationMixer::get_animation(const StringName &p_name) const {
	ERR_FAIL_COND_V_MSG(!animation_set.has(p_name), Ref<Animation>(), vformat("Animation not found: \"%s\".", p_name));
	const AnimationData &anim_data = animation_set[p_name];
	return anim_data.animation;
}

bool AnimationMixer::has_animation(const StringName &p_name) const {
	return animation_set.has(p_name);
}

StringName AnimationMixer::find_animation(const Ref<Animation> &p_animation) const {
	for (const KeyValue<StringName, AnimationData> &E : animation_set) {
		if (E.value.animation == p_animation) {
			return E.key;
		}
	}
	return StringName();
}

/* -------------------------------------------- */
/* -- General settings for animation ---------- */
/* -------------------------------------------- */

void AnimationMixer::_set_process(bool p_process, bool p_force) {
	if (processing == p_process && !p_force) {
		return;
	}

	switch (callback_mode_process) {
		case ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS:
#ifdef TOOLS_ENABLED
			set_physics_process_internal(p_process && active && !editing);
#else
			set_physics_process_internal(p_process && active);
#endif // TOOLS_ENABLED
			break;
		case ANIMATION_CALLBACK_MODE_PROCESS_IDLE:
#ifdef TOOLS_ENABLED
			set_process_internal(p_process && active && !editing);
#else
			set_process_internal(p_process && active);
#endif // TOOLS_ENABLED
			break;
		case ANIMATION_CALLBACK_MODE_PROCESS_MANUAL:
			break;
	}

	processing = p_process;
}

void AnimationMixer::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;
	_set_active(active);
	_set_process(processing, true);

	if (!active && is_inside_tree()) {
		_clear_caches();
	}
}

bool AnimationMixer::is_active() const {
	return active;
}

void AnimationMixer::set_root_node(const NodePath &p_path) {
	root_node = p_path;
	_clear_caches();
}

NodePath AnimationMixer::get_root_node() const {
	return root_node;
}

void AnimationMixer::set_deterministic(bool p_deterministic) {
	deterministic = p_deterministic;
	_clear_caches();
}

bool AnimationMixer::is_deterministic() const {
	return deterministic;
}

void AnimationMixer::set_callback_mode_process(AnimationCallbackModeProcess p_mode) {
	if (callback_mode_process == p_mode) {
		return;
	}

	bool was_active = is_active();
	if (was_active) {
		set_active(false);
	}

	callback_mode_process = p_mode;

	if (was_active) {
		set_active(true);
	}

#ifdef TOOLS_ENABLED
	emit_signal(SNAME("mixer_updated"));
#endif // TOOLS_ENABLED
}

AnimationMixer::AnimationCallbackModeProcess AnimationMixer::get_callback_mode_process() const {
	return callback_mode_process;
}

void AnimationMixer::set_callback_mode_method(AnimationCallbackModeMethod p_mode) {
	callback_mode_method = p_mode;
#ifdef TOOLS_ENABLED
	emit_signal(SNAME("mixer_updated"));
#endif // TOOLS_ENABLED
}

AnimationMixer::AnimationCallbackModeMethod AnimationMixer::get_callback_mode_method() const {
	return callback_mode_method;
}

void AnimationMixer::set_audio_max_polyphony(int p_audio_max_polyphony) {
	ERR_FAIL_COND(p_audio_max_polyphony < 0 || p_audio_max_polyphony > 128);
	audio_max_polyphony = p_audio_max_polyphony;
}

int AnimationMixer::get_audio_max_polyphony() const {
	return audio_max_polyphony;
}

#ifdef TOOLS_ENABLED
void AnimationMixer::set_editing(bool p_editing) {
	if (editing == p_editing) {
		return;
	}

	editing = p_editing;
	_set_process(processing, true);

	if (editing && is_inside_tree()) {
		_clear_caches();
	}

	notify_property_list_changed(); // To make active readonly.
}

bool AnimationMixer::is_editing() const {
	return editing;
}

void AnimationMixer::set_dummy(bool p_dummy) {
	dummy = p_dummy;
}

bool AnimationMixer::is_dummy() const {
	return dummy;
}
#endif // TOOLS_ENABLED

/* -------------------------------------------- */
/* -- Caches for blending --------------------- */
/* -------------------------------------------- */

void AnimationMixer::_clear_caches() {
	_init_root_motion_cache();
	_clear_audio_streams();
	_clear_playing_caches();
	for (KeyValue<NodePath, TrackCache *> &K : track_cache) {
		memdelete(K.value);
	}
	track_cache.clear();
	cache_valid = false;

	emit_signal(SNAME("caches_cleared"));
}

void AnimationMixer::_clear_audio_streams() {
	for (int i = 0; i < playing_audio_stream_players.size(); i++) {
		playing_audio_stream_players[i]->call(SNAME("stop"));
		playing_audio_stream_players[i]->call(SNAME("set_stream"), Ref<AudioStream>());
	}
	playing_audio_stream_players.clear();
}

void AnimationMixer::_clear_playing_caches() {
	for (const TrackCache *E : playing_caches) {
		if (ObjectDB::get_instance(E->object_id)) {
			E->object->call(SNAME("stop"));
		}
	}
	playing_caches.clear();
}

void AnimationMixer::_init_root_motion_cache() {
	root_motion_cache.loc = Vector3(0, 0, 0);
	root_motion_cache.rot = Quaternion(0, 0, 0, 1);
	root_motion_cache.scale = Vector3(1, 1, 1);
	root_motion_position = Vector3(0, 0, 0);
	root_motion_rotation = Quaternion(0, 0, 0, 1);
	root_motion_scale = Vector3(0, 0, 0);
	root_motion_position_accumulator = Vector3(0, 0, 0);
	root_motion_rotation_accumulator = Quaternion(0, 0, 0, 1);
	root_motion_scale_accumulator = Vector3(1, 1, 1);
}

bool AnimationMixer::_update_caches() {
	setup_pass++;

	root_motion_cache.loc = Vector3(0, 0, 0);
	root_motion_cache.rot = Quaternion(0, 0, 0, 1);
	root_motion_cache.scale = Vector3(1, 1, 1);

	List<StringName> sname;
	get_animation_list(&sname);

	Node *parent = get_node_or_null(root_node);
	if (!parent) {
		cache_valid = false;
		return false;
	}

	Ref<Animation> reset_anim;
	bool has_reset_anim = has_animation(SceneStringNames::get_singleton()->RESET);
	if (has_reset_anim) {
		reset_anim = get_animation(SceneStringNames::get_singleton()->RESET);
	}
	for (const StringName &E : sname) {
		Ref<Animation> anim = get_animation(E);
		for (int i = 0; i < anim->get_track_count(); i++) {
			NodePath path = anim->track_get_path(i);
			Animation::TrackType track_type = anim->track_get_type(i);

			Animation::TrackType track_cache_type = track_type;
			if (track_cache_type == Animation::TYPE_POSITION_3D || track_cache_type == Animation::TYPE_ROTATION_3D || track_cache_type == Animation::TYPE_SCALE_3D) {
				track_cache_type = Animation::TYPE_POSITION_3D; // Reference them as position3D tracks, even if they modify rotation or scale.
			}

			TrackCache *track = nullptr;
			if (track_cache.has(path)) {
				track = track_cache.get(path);
			}

			// If not valid, delete track.
			if (track && (track->type != track_cache_type || ObjectDB::get_instance(track->object_id) == nullptr)) {
				playing_caches.erase(track);
				memdelete(track);
				track_cache.erase(path);
				track = nullptr;
			}

			if (!track) {
				Ref<Resource> resource;
				Vector<StringName> leftover_path;
				Node *child = parent->get_node_and_resource(path, resource, leftover_path);

				if (!child) {
					ERR_PRINT("AnimationMixer: '" + String(E) + "', couldn't resolve track:  '" + String(path) + "'.");
					continue;
				}

				switch (track_type) {
					case Animation::TYPE_VALUE: {
						TrackCacheValue *track_value = memnew(TrackCacheValue);

						if (resource.is_valid()) {
							track_value->object = resource.ptr();
						} else {
							track_value->object = child;
						}

						track_value->is_continuous = anim->value_track_get_update_mode(i) != Animation::UPDATE_DISCRETE;
						track_value->is_using_angle = anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_LINEAR_ANGLE || anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_CUBIC_ANGLE;

						track_value->subpath = leftover_path;
						track_value->object_id = track_value->object->get_instance_id();

						track = track_value;

						// If a value track without a key is cached first, the initial value cannot be determined.
						// It is a corner case, but which may cause problems with blending.
						ERR_CONTINUE_MSG(anim->track_get_key_count(i) == 0, "AnimationMixer: '" + String(E) + "', Value Track:  '" + String(path) + "' must have at least one key to cache for blending.");
						track_value->init_value = anim->track_get_key_value(i, 0);
						track_value->init_value.zero();

						// If there is a Reset Animation, it takes precedence by overwriting.
						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_value->init_value = reset_anim->track_get_key_value(rt, 0);
							}
						}

					} break;
					case Animation::TYPE_POSITION_3D:
					case Animation::TYPE_ROTATION_3D:
					case Animation::TYPE_SCALE_3D: {
#ifndef _3D_DISABLED
						Node3D *node_3d = Object::cast_to<Node3D>(child);

						if (!node_3d) {
							ERR_PRINT("AnimationMixer: '" + String(E) + "', transform track does not point to Node3D:  '" + String(path) + "'.");
							continue;
						}

						TrackCacheTransform *track_xform = memnew(TrackCacheTransform);
						track_xform->type = Animation::TYPE_POSITION_3D;

						track_xform->node_3d = node_3d;
						track_xform->skeleton = nullptr;
						track_xform->bone_idx = -1;

						bool has_rest = false;
						if (path.get_subname_count() == 1 && Object::cast_to<Skeleton3D>(node_3d)) {
							Skeleton3D *sk = Object::cast_to<Skeleton3D>(node_3d);
							track_xform->skeleton = sk;
							int bone_idx = sk->find_bone(path.get_subname(0));
							if (bone_idx != -1) {
								has_rest = true;
								track_xform->bone_idx = bone_idx;
								Transform3D rest = sk->get_bone_rest(bone_idx);
								track_xform->init_loc = rest.origin;
								track_xform->init_rot = rest.basis.get_rotation_quaternion();
								track_xform->init_scale = rest.basis.get_scale();
							}
						}

						track_xform->object = node_3d;
						track_xform->object_id = track_xform->object->get_instance_id();

						track = track_xform;

						switch (track_type) {
							case Animation::TYPE_POSITION_3D: {
								track_xform->loc_used = true;
							} break;
							case Animation::TYPE_ROTATION_3D: {
								track_xform->rot_used = true;
							} break;
							case Animation::TYPE_SCALE_3D: {
								track_xform->scale_used = true;
							} break;
							default: {
							}
						}

						// For non Skeleton3D bone animation.
						if (has_reset_anim && !has_rest) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								switch (track_type) {
									case Animation::TYPE_POSITION_3D: {
										track_xform->init_loc = reset_anim->track_get_key_value(rt, 0);
									} break;
									case Animation::TYPE_ROTATION_3D: {
										track_xform->init_rot = reset_anim->track_get_key_value(rt, 0);
									} break;
									case Animation::TYPE_SCALE_3D: {
										track_xform->init_scale = reset_anim->track_get_key_value(rt, 0);
									} break;
									default: {
									}
								}
							}
						}
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
						if (path.get_subname_count() != 1) {
							ERR_PRINT("AnimationMixer: '" + String(E) + "', blend shape track does not contain a blend shape subname:  '" + String(path) + "'.");
							continue;
						}
						MeshInstance3D *mesh_3d = Object::cast_to<MeshInstance3D>(child);

						if (!mesh_3d) {
							ERR_PRINT("AnimationMixer: '" + String(E) + "', blend shape track does not point to MeshInstance3D:  '" + String(path) + "'.");
							continue;
						}

						StringName blend_shape_name = path.get_subname(0);
						int blend_shape_idx = mesh_3d->find_blend_shape_by_name(blend_shape_name);
						if (blend_shape_idx == -1) {
							ERR_PRINT("AnimationMixer: '" + String(E) + "', blend shape track points to a non-existing name:  '" + String(blend_shape_name) + "'.");
							continue;
						}

						TrackCacheBlendShape *track_bshape = memnew(TrackCacheBlendShape);

						track_bshape->mesh_3d = mesh_3d;
						track_bshape->shape_index = blend_shape_idx;

						track_bshape->object = mesh_3d;
						track_bshape->object_id = mesh_3d->get_instance_id();
						track = track_bshape;

						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_bshape->init_value = reset_anim->track_get_key_value(rt, 0);
							}
						}
#endif
					} break;
					case Animation::TYPE_METHOD: {
						TrackCacheMethod *track_method = memnew(TrackCacheMethod);

						if (resource.is_valid()) {
							track_method->object = resource.ptr();
						} else {
							track_method->object = child;
						}

						track_method->object_id = track_method->object->get_instance_id();

						track = track_method;

					} break;
					case Animation::TYPE_BEZIER: {
						TrackCacheBezier *track_bezier = memnew(TrackCacheBezier);

						if (resource.is_valid()) {
							track_bezier->object = resource.ptr();
						} else {
							track_bezier->object = child;
						}

						track_bezier->subpath = leftover_path;
						track_bezier->object_id = track_bezier->object->get_instance_id();

						track = track_bezier;

						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_bezier->init_value = (reset_anim->track_get_key_value(rt, 0).operator Array())[0];
							}
						}

					} break;
					case Animation::TYPE_AUDIO: {
						TrackCacheAudio *track_audio = memnew(TrackCacheAudio);

						track_audio->object = child;
						track_audio->object_id = track_audio->object->get_instance_id();
						track_audio->audio_stream.instantiate();
						track_audio->audio_stream->set_polyphony(audio_max_polyphony);

						track = track_audio;

					} break;
					case Animation::TYPE_ANIMATION: {
						TrackCacheAnimation *track_animation = memnew(TrackCacheAnimation);

						track_animation->object = child;
						track_animation->object_id = track_animation->object->get_instance_id();

						track = track_animation;

					} break;
					default: {
						ERR_PRINT("Animation corrupted (invalid track type).");
						continue;
					}
				}

				track_cache[path] = track;
			} else if (track_cache_type == Animation::TYPE_POSITION_3D) {
				TrackCacheTransform *track_xform = static_cast<TrackCacheTransform *>(track);
				if (track->setup_pass != setup_pass) {
					track_xform->loc_used = false;
					track_xform->rot_used = false;
					track_xform->scale_used = false;
				}
				switch (track_type) {
					case Animation::TYPE_POSITION_3D: {
						track_xform->loc_used = true;
					} break;
					case Animation::TYPE_ROTATION_3D: {
						track_xform->rot_used = true;
					} break;
					case Animation::TYPE_SCALE_3D: {
						track_xform->scale_used = true;
					} break;
					default: {
					}
				}
			} else if (track_cache_type == Animation::TYPE_VALUE) {
				// If it has at least one angle interpolation, it also uses angle interpolation for blending.
				TrackCacheValue *track_value = static_cast<TrackCacheValue *>(track);
				bool was_continuous = track_value->is_continuous;
				bool was_using_angle = track_value->is_using_angle;
				track_value->is_continuous |= anim->value_track_get_update_mode(i) != Animation::UPDATE_DISCRETE;
				track_value->is_using_angle |= anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_LINEAR_ANGLE || anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_CUBIC_ANGLE;

				// TODO: Currently, misc type cannot be blended.
				// In the future, it should have a separate blend weight, just as bool is converted to 0 and 1.
				// Then, it should provide the correct precedence value.
				bool skip_update_mode_warning = false;
				if (track_value->is_continuous) {
					if (!Animation::is_variant_interpolatable(track_value->init_value)) {
						WARN_PRINT_ONCE_ED("AnimationMixer: '" + String(E) + "', Value Track: '" + String(path) + "' uses a non-numeric type as key value with UpdateMode.UPDATE_CONTINUOUS. This will not be blended correctly, so it is forced to UpdateMode.UPDATE_DISCRETE.");
						track_value->is_continuous = false;
						skip_update_mode_warning = true;
					}
					if (track_value->init_value.is_string()) {
						WARN_PRINT_ONCE_ED("AnimationMixer: '" + String(E) + "', Value Track: '" + String(path) + "' blends String types. This is an experimental algorithm.");
					}
				}

				if (!skip_update_mode_warning && was_continuous != track_value->is_continuous) {
					WARN_PRINT_ONCE_ED("AnimationMixer: '" + String(E) + "', Value Track: '" + String(path) + "' has different update modes between some animations which may be blended together. Blending prioritizes UpdateMode.UPDATE_CONTINUOUS, so the process treats UpdateMode.UPDATE_DISCRETE as UpdateMode.UPDATE_CONTINUOUS with InterpolationType.INTERPOLATION_NEAREST.");
				}
				if (was_using_angle != track_value->is_using_angle) {
					WARN_PRINT_ONCE_ED("AnimationMixer: '" + String(E) + "', Value Track: '" + String(path) + "' has different interpolation types for rotation between some animations which may be blended together. Blending prioritizes angle interpolation, so the blending result uses the shortest path referenced to the initial (RESET animation) value.");
				}
			}

			track->setup_pass = setup_pass;
		}
	}

	List<NodePath> to_delete;

	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		TrackCache *tc = track_cache[K.key];
		if (tc->setup_pass != setup_pass) {
			to_delete.push_back(K.key);
		}
	}

	while (to_delete.front()) {
		NodePath np = to_delete.front()->get();
		memdelete(track_cache[np]);
		track_cache.erase(np);
		to_delete.pop_front();
	}

	track_map.clear();

	int idx = 0;
	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		track_map[K.key] = idx;
		idx++;
	}

	track_count = idx;

	cache_valid = true;

	return true;
}

/* -------------------------------------------- */
/* -- Blending processor ---------------------- */
/* -------------------------------------------- */

void AnimationMixer::_process_animation(double p_delta, bool p_update_only) {
	_blend_init();
	if (_blend_pre_process(p_delta, track_count, track_map)) {
		_blend_calc_total_weight();
		_blend_process(p_delta, p_update_only);
		_blend_apply();
		_blend_post_process();
	};
	clear_animation_instances();
}

Variant AnimationMixer::post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
	Variant res;
	if (GDVIRTUAL_CALL(_post_process_key_value, p_anim, p_track, p_value, const_cast<Object *>(p_object), p_object_idx, res)) {
		return res;
	}
	return _post_process_key_value(p_anim, p_track, p_value, p_object, p_object_idx);
}

Variant AnimationMixer::_post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
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

void AnimationMixer::_blend_init() {
	// Check all tracks, see if they need modification.
	root_motion_position = Vector3(0, 0, 0);
	root_motion_rotation = Quaternion(0, 0, 0, 1);
	root_motion_scale = Vector3(0, 0, 0);
	root_motion_position_accumulator = Vector3(0, 0, 0);
	root_motion_rotation_accumulator = Quaternion(0, 0, 0, 1);
	root_motion_scale_accumulator = Vector3(1, 1, 1);

	if (!cache_valid) {
		if (!_update_caches()) {
			return;
		}
	}

	// Init all value/transform/blend/bezier tracks that track_cache has.
	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		TrackCache *track = K.value;

		track->total_weight = 0.0;

		switch (track->type) {
			case Animation::TYPE_POSITION_3D: {
				TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
				if (track->root_motion) {
					root_motion_cache.loc = Vector3(0, 0, 0);
					root_motion_cache.rot = Quaternion(0, 0, 0, 1);
					root_motion_cache.scale = Vector3(1, 1, 1);
				}
				t->loc = t->init_loc;
				t->rot = t->init_rot;
				t->scale = t->init_scale;
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
				TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);
				t->value = t->init_value;
			} break;
			case Animation::TYPE_VALUE: {
				TrackCacheValue *t = static_cast<TrackCacheValue *>(track);
				t->value = Animation::cast_to_blendwise(t->init_value);
				t->element_size = t->init_value.is_string() ? (real_t)(t->init_value.operator String()).length() : 0;
			} break;
			case Animation::TYPE_BEZIER: {
				TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);
				t->value = t->init_value;
			} break;
			case Animation::TYPE_AUDIO: {
				TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);
				for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
					PlayingAudioTrackInfo &track_info = L.value;
					track_info.volume = 0.0;
				}
			} break;
			default: {
			} break;
		}
	}
}

bool AnimationMixer::_blend_pre_process(double p_delta, int p_track_count, const HashMap<NodePath, int> &p_track_map) {
	return true;
}

void AnimationMixer::_blend_post_process() {
	//
}

void AnimationMixer::_blend_calc_total_weight() {
	for (const AnimationInstance &ai : animation_instances) {
		Ref<Animation> a = ai.animation_data.animation;
		real_t weight = ai.playback_info.weight;
		Vector<real_t> track_weights = ai.playback_info.track_weights;
		Vector<int> processed_indices;
		for (int i = 0; i < a->get_track_count(); i++) {
			if (!a->track_is_enabled(i)) {
				continue;
			}
			NodePath path = a->track_get_path(i);
			if (!track_cache.has(path)) {
				continue; // No path, but avoid error spamming.
			}
			TrackCache *track = track_cache[path];
			int blend_idx = track_map[path];
			if (processed_indices.has(blend_idx)) {
				continue; // There is the case different track type with same path... Is there more faster iterating way than has()?
			}
			ERR_CONTINUE(blend_idx < 0 || blend_idx >= track_count);
			real_t blend = blend_idx < track_weights.size() ? track_weights[blend_idx] * weight : weight;
			track->total_weight += blend;
			processed_indices.push_back(blend_idx);
		}
	}
}

void AnimationMixer::_blend_process(double p_delta, bool p_update_only) {
	// Apply value/transform/blend/bezier blends to track caches and execute method/audio/animation tracks.
#ifdef TOOLS_ENABLED
	bool can_call = is_inside_tree() && !Engine::get_singleton()->is_editor_hint();
#endif // TOOLS_ENABLED
	for (const AnimationInstance &ai : animation_instances) {
		Ref<Animation> a = ai.animation_data.animation;
		double time = ai.playback_info.time;
		double delta = ai.playback_info.delta;
		bool seeked = ai.playback_info.seeked;
		Animation::LoopedFlag looped_flag = ai.playback_info.looped_flag;
		bool is_external_seeking = ai.playback_info.is_external_seeking;
		real_t weight = ai.playback_info.weight;
		Vector<real_t> track_weights = ai.playback_info.track_weights;
		bool backward = signbit(delta); // This flag is used by the root motion calculates or detecting the end of audio stream.
#ifndef _3D_DISABLED
		bool calc_root = !seeked || is_external_seeking;
#endif // _3D_DISABLED

		for (int i = 0; i < a->get_track_count(); i++) {
			if (!a->track_is_enabled(i)) {
				continue;
			}
			NodePath path = a->track_get_path(i);
			if (!track_cache.has(path)) {
				continue; // No path, but avoid error spamming.
			}
			TrackCache *track = track_cache[path];
			ERR_CONTINUE(!track_map.has(path));
			int blend_idx = track_map[path];
			ERR_CONTINUE(blend_idx < 0 || blend_idx >= track_count);
			real_t blend = blend_idx < track_weights.size() ? track_weights[blend_idx] * weight : weight;
			if (!deterministic) {
				// If non-deterministic, do normalization.
				// It would be better to make this if statement outside the for loop, but come here since too much code...
				if (Math::is_zero_approx(track->total_weight)) {
					continue;
				}
				blend = blend / track->total_weight;
			}
			Animation::TrackType ttype = a->track_get_type(i);
			if (ttype != Animation::TYPE_POSITION_3D && ttype != Animation::TYPE_ROTATION_3D && ttype != Animation::TYPE_SCALE_3D && track->type != ttype) {
				// Broken animation, but avoid error spamming.
				continue;
			}
			track->root_motion = root_motion_track == path;
			switch (ttype) {
				case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
					if (track->root_motion && calc_root) {
						double prev_time = time - delta;
						if (!backward) {
							if (prev_time < 0) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = 0;
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						} else {
							if (prev_time > a->get_length()) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = (double)a->get_length();
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						}
						Vector3 loc[2];
						if (!backward) {
							if (prev_time > time) {
								Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
								if (err != OK) {
									continue;
								}
								loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
								a->try_position_track_interpolate(i, (double)a->get_length(), &loc[1]);
								loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
								root_motion_cache.loc += (loc[1] - loc[0]) * blend;
								prev_time = 0;
							}
						} else {
							if (prev_time < time) {
								Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
								if (err != OK) {
									continue;
								}
								loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
								a->try_position_track_interpolate(i, 0, &loc[1]);
								loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
								root_motion_cache.loc += (loc[1] - loc[0]) * blend;
								prev_time = (double)a->get_length();
							}
						}
						Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
						if (err != OK) {
							continue;
						}
						loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
						a->try_position_track_interpolate(i, time, &loc[1]);
						loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
						root_motion_cache.loc += (loc[1] - loc[0]) * blend;
						prev_time = !backward ? 0 : (double)a->get_length();
					}
					{
						Vector3 loc;
						Error err = a->try_position_track_interpolate(i, time, &loc);
						if (err != OK) {
							continue;
						}
						loc = post_process_key_value(a, i, loc, t->object, t->bone_idx);
						t->loc += (loc - t->init_loc) * blend;
					}
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_ROTATION_3D: {
#ifndef _3D_DISABLED
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
					if (track->root_motion && calc_root) {
						double prev_time = time - delta;
						if (!backward) {
							if (prev_time < 0) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = 0;
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						} else {
							if (prev_time > a->get_length()) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = (double)a->get_length();
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						}
						Quaternion rot[2];
						if (!backward) {
							if (prev_time > time) {
								Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
								if (err != OK) {
									continue;
								}
								rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);
								a->try_rotation_track_interpolate(i, (double)a->get_length(), &rot[1]);
								rot[1] = post_process_key_value(a, i, rot[1], t->object, t->bone_idx);
								root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
								prev_time = 0;
							}
						} else {
							if (prev_time < time) {
								Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
								if (err != OK) {
									continue;
								}
								rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);
								a->try_rotation_track_interpolate(i, 0, &rot[1]);
								root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
								prev_time = (double)a->get_length();
							}
						}
						Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
						if (err != OK) {
							continue;
						}
						rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);
						a->try_rotation_track_interpolate(i, time, &rot[1]);
						rot[1] = post_process_key_value(a, i, rot[1], t->object, t->bone_idx);
						root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
						prev_time = !backward ? 0 : (double)a->get_length();
					}
					{
						Quaternion rot;
						Error err = a->try_rotation_track_interpolate(i, time, &rot);
						if (err != OK) {
							continue;
						}
						rot = post_process_key_value(a, i, rot, t->object, t->bone_idx);
						t->rot = (t->rot * Quaternion().slerp(t->init_rot.inverse() * rot, blend)).normalized();
					}
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_SCALE_3D: {
#ifndef _3D_DISABLED
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
					if (track->root_motion && calc_root) {
						double prev_time = time - delta;
						if (!backward) {
							if (prev_time < 0) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = 0;
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						} else {
							if (prev_time > a->get_length()) {
								switch (a->get_loop_mode()) {
									case Animation::LOOP_NONE: {
										prev_time = (double)a->get_length();
									} break;
									case Animation::LOOP_LINEAR: {
										prev_time = Math::fposmod(prev_time, (double)a->get_length());
									} break;
									case Animation::LOOP_PINGPONG: {
										prev_time = Math::pingpong(prev_time, (double)a->get_length());
									} break;
									default:
										break;
								}
							}
						}
						Vector3 scale[2];
						if (!backward) {
							if (prev_time > time) {
								Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
								if (err != OK) {
									continue;
								}
								scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);
								a->try_scale_track_interpolate(i, (double)a->get_length(), &scale[1]);
								root_motion_cache.scale += (scale[1] - scale[0]) * blend;
								scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
								prev_time = 0;
							}
						} else {
							if (prev_time < time) {
								Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
								if (err != OK) {
									continue;
								}
								scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);
								a->try_scale_track_interpolate(i, 0, &scale[1]);
								scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
								root_motion_cache.scale += (scale[1] - scale[0]) * blend;
								prev_time = (double)a->get_length();
							}
						}
						Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
						if (err != OK) {
							continue;
						}
						scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);
						a->try_scale_track_interpolate(i, time, &scale[1]);
						scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
						root_motion_cache.scale += (scale[1] - scale[0]) * blend;
						prev_time = !backward ? 0 : (double)a->get_length();
					}
					{
						Vector3 scale;
						Error err = a->try_scale_track_interpolate(i, time, &scale);
						if (err != OK) {
							continue;
						}
						scale = post_process_key_value(a, i, scale, t->object, t->bone_idx);
						t->scale += (scale - t->init_scale) * blend;
					}
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);
					float value;
					Error err = a->try_blend_shape_track_interpolate(i, time, &value);
					//ERR_CONTINUE(err!=OK); //used for testing, should be removed
					if (err != OK) {
						continue;
					}
					value = post_process_key_value(a, i, value, t->object, t->shape_index);
					t->value += (value - t->init_value) * blend;
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_VALUE: {
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheValue *t = static_cast<TrackCacheValue *>(track);
					if (t->is_continuous) {
						Variant value = a->value_track_interpolate(i, time);
						value = post_process_key_value(a, i, value, t->object);
						if (value == Variant()) {
							continue;
						}
						// Special case for angle interpolation.
						if (t->is_using_angle) {
							// For blending consistency, it prevents rotation of more than 180 degrees from init_value.
							// This is the same as for Quaternion blends.
							float rot_a = t->value;
							float rot_b = value;
							float rot_init = t->init_value;
							rot_a = Math::fposmod(rot_a, (float)Math_TAU);
							rot_b = Math::fposmod(rot_b, (float)Math_TAU);
							rot_init = Math::fposmod(rot_init, (float)Math_TAU);
							if (rot_init < Math_PI) {
								rot_a = rot_a > rot_init + Math_PI ? rot_a - Math_TAU : rot_a;
								rot_b = rot_b > rot_init + Math_PI ? rot_b - Math_TAU : rot_b;
							} else {
								rot_a = rot_a < rot_init - Math_PI ? rot_a + Math_TAU : rot_a;
								rot_b = rot_b < rot_init - Math_PI ? rot_b + Math_TAU : rot_b;
							}
							t->value = Math::fposmod(rot_a + (rot_b - rot_init) * (float)blend, (float)Math_TAU);
						} else {
							value = Animation::cast_to_blendwise(value);
							if (t->init_value.is_array()) {
								t->element_size = MAX(t->element_size.operator int(), (value.operator Array()).size());
							} else if (t->init_value.is_string()) {
								real_t length = Animation::subtract_variant((real_t)(value.operator Array()).size(), (real_t)(t->init_value.operator String()).length());
								t->element_size = Animation::blend_variant(t->element_size, length, blend);
							}
							value = Animation::subtract_variant(value, Animation::cast_to_blendwise(t->init_value));
							t->value = Animation::blend_variant(t->value, value, blend);
						}
					} else {
						if (seeked) {
							int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
							if (idx < 0) {
								continue;
							}
							Variant value = a->track_get_key_value(i, idx);
							value = post_process_key_value(a, i, value, t->object);
							t->object->set_indexed(t->subpath, value);
						} else {
							List<int> indices;
							a->track_get_key_indices_in_range(i, time, delta, &indices, looped_flag);
							for (int &F : indices) {
								Variant value = a->track_get_key_value(i, F);
								value = post_process_key_value(a, i, value, t->object);
								t->object->set_indexed(t->subpath, value);
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
					if (p_update_only || Math::is_zero_approx(blend)) {
						continue;
					}
					TrackCacheMethod *t = static_cast<TrackCacheMethod *>(track);
					if (seeked) {
						int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
						if (idx < 0) {
							continue;
						}
						StringName method = a->method_track_get_name(i, idx);
						Vector<Variant> params = a->method_track_get_params(i, idx);
						_call_object(t->object, method, params, callback_mode_method == ANIMATION_CALLBACK_MODE_METHOD_DEFERRED);
					} else {
						List<int> indices;
						a->track_get_key_indices_in_range(i, time, delta, &indices, looped_flag);
						for (int &F : indices) {
							StringName method = a->method_track_get_name(i, F);
							Vector<Variant> params = a->method_track_get_params(i, F);
							_call_object(t->object, method, params, callback_mode_method == ANIMATION_CALLBACK_MODE_METHOD_DEFERRED);
						}
					}
				} break;
				case Animation::TYPE_BEZIER: {
					if (Math::is_zero_approx(blend)) {
						continue; // Nothing to blend.
					}
					TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);
					real_t bezier = a->bezier_track_interpolate(i, time);
					bezier = post_process_key_value(a, i, bezier, t->object);
					t->value += (bezier - t->init_value) * blend;
				} break;
				case Animation::TYPE_AUDIO: {
					// The end of audio should be observed even if the blend value is 0, build up the information and store to the cache for that.
					TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);
					Node *asp = Object::cast_to<Node>(t->object);
					if (!asp) {
						t->playing_streams.clear();
						continue;
					}
					ObjectID oid = a->get_instance_id();
					if (!t->playing_streams.has(oid)) {
						t->playing_streams[oid] = PlayingAudioTrackInfo();
					}

					PlayingAudioTrackInfo &track_info = t->playing_streams[oid];
					track_info.length = a->get_length();
					track_info.time = time;
					track_info.volume += blend;
					track_info.loop = a->get_loop_mode() != Animation::LOOP_NONE;
					track_info.backward = backward;
					track_info.use_blend = a->audio_track_is_use_blend(i);
					HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;

					// Main process to fire key is started from here.
					if (p_update_only) {
						continue;
					}
					// Find stream.
					int idx = -1;
					if (seeked) {
						idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
						// Discard previous stream when seeking.
						if (map.has(idx)) {
							t->audio_stream_playback->stop_stream(map[idx].index);
							map.erase(idx);
						}
					} else {
						List<int> to_play;
						a->track_get_key_indices_in_range(i, time, delta, &to_play, looped_flag);
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
						if (seeked) {
							start_ofs += time - a->track_get_key_time(i, idx);
						}
						if (t->object->call(SNAME("get_stream")) != t->audio_stream) {
							t->object->call(SNAME("set_stream"), t->audio_stream);
							t->audio_stream_playback.unref();
							if (!playing_audio_stream_players.has(asp)) {
								playing_audio_stream_players.push_back(asp);
							}
						}
						if (!t->object->call(SNAME("is_playing"))) {
							t->object->call(SNAME("play"));
						}
						if (!t->object->call(SNAME("has_stream_playback"))) {
							t->audio_stream_playback.unref();
							continue;
						}
						if (t->audio_stream_playback.is_null()) {
							t->audio_stream_playback = t->object->call(SNAME("get_stream_playback"));
						}
						PlayingAudioStreamInfo pasi;
						pasi.index = t->audio_stream_playback->play_stream(stream, start_ofs);
						pasi.start = time;
						if (len && end_ofs > 0) { // Force an end at a time.
							pasi.len = len - start_ofs - end_ofs;
						} else {
							pasi.len = 0;
						}
						map[idx] = pasi;
					}
				} break;
				case Animation::TYPE_ANIMATION: {
					if (p_update_only || Math::is_zero_approx(blend)) {
						continue;
					}
					TrackCacheAnimation *t = static_cast<TrackCacheAnimation *>(track);
					AnimationPlayer *player2 = Object::cast_to<AnimationPlayer>(t->object);
					if (!player2) {
						continue;
					}
					if (seeked) {
						// Seek.
						int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
						if (idx < 0) {
							continue;
						}
						double pos = a->track_get_key_time(i, idx);
						StringName anim_name = a->animation_track_get_key_animation(i, idx);
						if (String(anim_name) == "[stop]" || !player2->has_animation(anim_name)) {
							continue;
						}
						Ref<Animation> anim = player2->get_animation(anim_name);
						double at_anim_pos = 0.0;
						switch (anim->get_loop_mode()) {
							case Animation::LOOP_NONE: {
								at_anim_pos = MAX((double)anim->get_length(), time - pos); //seek to end
							} break;
							case Animation::LOOP_LINEAR: {
								at_anim_pos = Math::fposmod(time - pos, (double)anim->get_length()); //seek to loop
							} break;
							case Animation::LOOP_PINGPONG: {
								at_anim_pos = Math::pingpong(time - pos, (double)a->get_length());
							} break;
							default:
								break;
						}
						if (player2->is_playing() || seeked) {
							player2->seek(at_anim_pos);
							player2->play(anim_name);
							t->playing = true;
							playing_caches.insert(t);
						} else {
							player2->set_assigned_animation(anim_name);
							player2->seek(at_anim_pos, true);
						}
					} else {
						// Find stuff to play.
						List<int> to_play;
						a->track_get_key_indices_in_range(i, time, delta, &to_play, looped_flag);
						if (to_play.size()) {
							int idx = to_play.back()->get();
							StringName anim_name = a->animation_track_get_key_animation(i, idx);
							if (String(anim_name) == "[stop]" || !player2->has_animation(anim_name)) {
								if (playing_caches.has(t)) {
									playing_caches.erase(t);
									player2->stop();
									t->playing = false;
								}
							} else {
								player2->play(anim_name);
								t->playing = true;
								playing_caches.insert(t);
							}
						}
					}
				} break;
			}
		}
	}
}

void AnimationMixer::_blend_apply() {
	// Finally, set the tracks.
	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		TrackCache *track = K.value;
		if (!deterministic && Math::is_zero_approx(track->total_weight)) {
			continue;
		}
		switch (track->type) {
			case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
				TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

				if (t->root_motion) {
					root_motion_position = root_motion_cache.loc;
					root_motion_rotation = root_motion_cache.rot;
					root_motion_scale = root_motion_cache.scale - Vector3(1, 1, 1);
					root_motion_position_accumulator = t->loc;
					root_motion_rotation_accumulator = t->rot;
					root_motion_scale_accumulator = t->scale;
				} else if (t->skeleton && t->bone_idx >= 0) {
					if (t->loc_used) {
						t->skeleton->set_bone_pose_position(t->bone_idx, t->loc);
					}
					if (t->rot_used) {
						t->skeleton->set_bone_pose_rotation(t->bone_idx, t->rot);
					}
					if (t->scale_used) {
						t->skeleton->set_bone_pose_scale(t->bone_idx, t->scale);
					}

				} else if (!t->skeleton) {
					if (t->loc_used) {
						t->node_3d->set_position(t->loc);
					}
					if (t->rot_used) {
						t->node_3d->set_rotation(t->rot.get_euler());
					}
					if (t->scale_used) {
						t->node_3d->set_scale(t->scale);
					}
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
				TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);

				if (t->mesh_3d) {
					t->mesh_3d->set_blend_shape_value(t->shape_index, t->value);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_VALUE: {
				TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

				if (!t->is_continuous) {
					break; // Don't overwrite the value set by UPDATE_DISCRETE.
				}

				// Trim unused elements if init array/string is not blended.
				if (t->value.is_array()) {
					int actual_blended_size = (int)Math::round(Math::abs(t->element_size.operator real_t()));
					if (actual_blended_size < (t->value.operator Array()).size()) {
						real_t abs_weight = Math::abs(track->total_weight);
						if (abs_weight >= 1.0) {
							(t->value.operator Array()).resize(actual_blended_size);
						} else if (t->init_value.is_string()) {
							(t->value.operator Array()).resize(Animation::interpolate_variant((t->init_value.operator String()).length(), actual_blended_size, abs_weight));
						}
					}
				}
				t->object->set_indexed(t->subpath, Animation::cast_from_blendwise(t->value, t->init_value.get_type()));

			} break;
			case Animation::TYPE_BEZIER: {
				TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);

				t->object->set_indexed(t->subpath, t->value);

			} break;
			case Animation::TYPE_AUDIO: {
				TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

				// Audio ending process.
				LocalVector<ObjectID> erase_maps;
				for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
					PlayingAudioTrackInfo &track_info = L.value;
					float db = Math::linear_to_db(track_info.use_blend ? track_info.volume : 1.0);
					LocalVector<int> erase_streams;
					HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;
					for (const KeyValue<int, PlayingAudioStreamInfo> &M : map) {
						PlayingAudioStreamInfo pasi = M.value;

						bool stop = false;
						if (!t->audio_stream_playback->is_stream_playing(pasi.index)) {
							stop = true;
						}
						if (!track_info.loop) {
							if (!track_info.backward) {
								if (track_info.time < pasi.start) {
									stop = true;
								}
							} else if (track_info.backward) {
								if (track_info.time > pasi.start) {
									stop = true;
								}
							}
						}
						if (pasi.len > 0) {
							double len = 0.0;
							if (!track_info.backward) {
								len = pasi.start > track_info.time ? (track_info.length - pasi.start) + track_info.time : track_info.time - pasi.start;
							} else {
								len = pasi.start < track_info.time ? (track_info.length - track_info.time) + pasi.start : pasi.start - track_info.time;
							}
							if (len > pasi.len) {
								stop = true;
							}
						}
						if (stop) {
							// Time to stop.
							t->audio_stream_playback->stop_stream(pasi.index);
							erase_streams.push_back(M.key);
						} else {
							t->audio_stream_playback->set_stream_volume(pasi.index, db);
						}
					}
					for (uint32_t erase_idx = 0; erase_idx < erase_streams.size(); erase_idx++) {
						map.erase(erase_streams[erase_idx]);
					}
					if (map.size() == 0) {
						erase_maps.push_back(L.key);
					}
				}
				for (uint32_t erase_idx = 0; erase_idx < erase_maps.size(); erase_idx++) {
					t->playing_streams.erase(erase_maps[erase_idx]);
				}
			} break;
			default: {
			} // The rest don't matter.
		}
	}
}

void AnimationMixer::_call_object(Object *p_object, const StringName &p_method, const Vector<Variant> &p_params, bool p_deferred) {
	// Separate function to use alloca() more efficiently
	const Variant **argptrs = (const Variant **)alloca(sizeof(Variant *) * p_params.size());
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

void AnimationMixer::make_animation_instance(const StringName &p_name, const PlaybackInfo p_playback_info) {
	ERR_FAIL_COND(!has_animation(p_name));

	AnimationData ad;
	ad.name = p_name;
	ad.animation = get_animation(p_name);
	ad.animation_library = find_animation_library(ad.animation);

	AnimationInstance ai;
	ai.animation_data = ad;
	ai.playback_info = p_playback_info;

	animation_instances.push_back(ai);
}

void AnimationMixer::clear_animation_instances() {
	animation_instances.clear();
}

void AnimationMixer::advance(double p_time) {
	_process_animation(p_time);
}

void AnimationMixer::clear_caches() {
	_clear_caches();
}

/* -------------------------------------------- */
/* -- Root motion ----------------------------- */
/* -------------------------------------------- */

void AnimationMixer::set_root_motion_track(const NodePath &p_track) {
	root_motion_track = p_track;
}

NodePath AnimationMixer::get_root_motion_track() const {
	return root_motion_track;
}

Vector3 AnimationMixer::get_root_motion_position() const {
	return root_motion_position;
}

Quaternion AnimationMixer::get_root_motion_rotation() const {
	return root_motion_rotation;
}

Vector3 AnimationMixer::get_root_motion_scale() const {
	return root_motion_scale;
}

Vector3 AnimationMixer::get_root_motion_position_accumulator() const {
	return root_motion_position_accumulator;
}

Quaternion AnimationMixer::get_root_motion_rotation_accumulator() const {
	return root_motion_rotation_accumulator;
}

Vector3 AnimationMixer::get_root_motion_scale_accumulator() const {
	return root_motion_scale_accumulator;
}

void AnimationMixer::set_reset_on_save_enabled(bool p_enabled) {
	reset_on_save = p_enabled;
}

bool AnimationMixer::is_reset_on_save_enabled() const {
	return reset_on_save;
}

bool AnimationMixer::can_apply_reset() const {
	return has_animation(SceneStringNames::get_singleton()->RESET);
}

void AnimationMixer::_build_backup_track_cache() {
	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		TrackCache *track = K.value;
		track->total_weight = 1.0;
		switch (track->type) {
			case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
				TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
				if (t->root_motion) {
					// Do nothing.
				} else if (t->skeleton && t->bone_idx >= 0) {
					if (t->loc_used) {
						t->loc = t->skeleton->get_bone_pose_position(t->bone_idx);
					}
					if (t->rot_used) {
						t->rot = t->skeleton->get_bone_pose_rotation(t->bone_idx);
					}
					if (t->scale_used) {
						t->scale = t->skeleton->get_bone_pose_scale(t->bone_idx);
					}
				} else if (!t->skeleton) {
					if (t->loc_used) {
						t->loc = t->node_3d->get_position();
					}
					if (t->rot_used) {
						t->rot = t->node_3d->get_quaternion();
					}
					if (t->scale_used) {
						t->scale = t->node_3d->get_scale();
					}
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
				TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);
				if (t->mesh_3d) {
					t->value = t->mesh_3d->get_blend_shape_value(t->shape_index);
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_VALUE: {
				TrackCacheValue *t = static_cast<TrackCacheValue *>(track);
				t->value = t->object->get_indexed(t->subpath);
				t->is_continuous = true;
			} break;
			case Animation::TYPE_BEZIER: {
				TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);
				t->value = t->object->get_indexed(t->subpath);
			} break;
			case Animation::TYPE_AUDIO: {
				TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);
				Node *asp = Object::cast_to<Node>(t->object);
				if (asp) {
					t->object->call(SNAME("set_stream"), Ref<AudioStream>());
				}
			} break;
			default: {
			} // The rest don't matter.
		}
	}
}

Ref<AnimatedValuesBackup> AnimationMixer::make_backup() {
	Ref<AnimatedValuesBackup> backup;
	backup.instantiate();

	Ref<Animation> reset_anim = animation_set[SceneStringNames::get_singleton()->RESET].animation;
	ERR_FAIL_COND_V(reset_anim.is_null(), Ref<AnimatedValuesBackup>());

	_blend_init();
	PlaybackInfo pi;
	pi.time = 0;
	pi.delta = 0;
	pi.seeked = true;
	pi.weight = 1.0;
	make_animation_instance(SceneStringNames::get_singleton()->RESET, pi);
	_build_backup_track_cache();

	backup->set_data(track_cache);
	clear_animation_instances();

	return backup;
}

void AnimationMixer::reset() {
	ERR_FAIL_COND(!can_apply_reset());

	Ref<Animation> reset_anim = animation_set[SceneStringNames::get_singleton()->RESET].animation;
	ERR_FAIL_COND(reset_anim.is_null());

	Node *root_node_object = get_node_or_null(root_node);
	ERR_FAIL_NULL(root_node_object);

	AnimationPlayer *aux_player = memnew(AnimationPlayer);
	root_node_object->add_child(aux_player);
	Ref<AnimationLibrary> al;
	al.instantiate();
	al->add_animation(SceneStringNames::get_singleton()->RESET, reset_anim);
	aux_player->set_reset_on_save_enabled(false);
	aux_player->set_root_node(aux_player->get_path_to(root_node_object));
	aux_player->add_animation_library("", al);
	aux_player->set_assigned_animation(SceneStringNames::get_singleton()->RESET);
	aux_player->seek(0.0f, true);
	aux_player->queue_free();
}

void AnimationMixer::restore(const Ref<AnimatedValuesBackup> &p_backup) {
	track_cache = p_backup->get_data();
	_blend_apply();
	track_cache = HashMap<NodePath, AnimationMixer::TrackCache *>();
	cache_valid = false;
}

#ifdef TOOLS_ENABLED
Ref<AnimatedValuesBackup> AnimationMixer::apply_reset(bool p_user_initiated) {
	if (!p_user_initiated && dummy) {
		return Ref<AnimatedValuesBackup>();
	}
	ERR_FAIL_COND_V(!can_apply_reset(), Ref<AnimatedValuesBackup>());

	Ref<Animation> reset_anim = animation_set[SceneStringNames::get_singleton()->RESET].animation;
	ERR_FAIL_COND_V(reset_anim.is_null(), Ref<AnimatedValuesBackup>());

	Ref<AnimatedValuesBackup> backup_current = make_backup();
	if (p_user_initiated) {
		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Animation Apply Reset"));
		ur->add_do_method(this, "_reset");
		ur->add_undo_method(this, "_restore", backup_current);
		ur->commit_action();
	} else {
		reset();
	}

	return backup_current;
}
#endif // TOOLS_ENABLED

/* -------------------------------------------- */
/* -- General functions ----------------------- */
/* -------------------------------------------- */

void AnimationMixer::_node_removed(Node *p_node) {
	_clear_caches();
}

void AnimationMixer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (!processing) {
				set_physics_process_internal(false);
				set_process_internal(false);
			}
			_clear_caches();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (active && callback_mode_process == ANIMATION_CALLBACK_MODE_PROCESS_IDLE) {
				_process_animation(get_process_delta_time());
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (active && callback_mode_process == ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS) {
				_process_animation(get_physics_process_delta_time());
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_clear_caches();
		} break;
	}
}

void AnimationMixer::_bind_methods() {
	/* ---- Data lists ---- */
	ClassDB::bind_method(D_METHOD("add_animation_library", "name", "library"), &AnimationMixer::add_animation_library);
	ClassDB::bind_method(D_METHOD("remove_animation_library", "name"), &AnimationMixer::remove_animation_library);
	ClassDB::bind_method(D_METHOD("rename_animation_library", "name", "newname"), &AnimationMixer::rename_animation_library);
	ClassDB::bind_method(D_METHOD("has_animation_library", "name"), &AnimationMixer::has_animation_library);
	ClassDB::bind_method(D_METHOD("get_animation_library", "name"), &AnimationMixer::get_animation_library);
	ClassDB::bind_method(D_METHOD("get_animation_library_list"), &AnimationMixer::_get_animation_library_list);

	ClassDB::bind_method(D_METHOD("has_animation", "name"), &AnimationMixer::has_animation);
	ClassDB::bind_method(D_METHOD("get_animation", "name"), &AnimationMixer::get_animation);
	ClassDB::bind_method(D_METHOD("get_animation_list"), &AnimationMixer::_get_animation_list);

	/* ---- General settings for animation ---- */
	ClassDB::bind_method(D_METHOD("set_active", "active"), &AnimationMixer::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationMixer::is_active);

	ClassDB::bind_method(D_METHOD("set_deterministic", "deterministic"), &AnimationMixer::set_deterministic);
	ClassDB::bind_method(D_METHOD("is_deterministic"), &AnimationMixer::is_deterministic);

	ClassDB::bind_method(D_METHOD("set_root_node", "path"), &AnimationMixer::set_root_node);
	ClassDB::bind_method(D_METHOD("get_root_node"), &AnimationMixer::get_root_node);

	ClassDB::bind_method(D_METHOD("set_callback_mode_process", "mode"), &AnimationMixer::set_callback_mode_process);
	ClassDB::bind_method(D_METHOD("get_callback_mode_process"), &AnimationMixer::get_callback_mode_process);

	ClassDB::bind_method(D_METHOD("set_callback_mode_method", "mode"), &AnimationMixer::set_callback_mode_method);
	ClassDB::bind_method(D_METHOD("get_callback_mode_method"), &AnimationMixer::get_callback_mode_method);

	ClassDB::bind_method(D_METHOD("set_audio_max_polyphony", "max_polyphony"), &AnimationMixer::set_audio_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_audio_max_polyphony"), &AnimationMixer::get_audio_max_polyphony);

	/* ---- Root motion accumulator for Skeleton3D ---- */
	ClassDB::bind_method(D_METHOD("set_root_motion_track", "path"), &AnimationMixer::set_root_motion_track);
	ClassDB::bind_method(D_METHOD("get_root_motion_track"), &AnimationMixer::get_root_motion_track);

	ClassDB::bind_method(D_METHOD("get_root_motion_position"), &AnimationMixer::get_root_motion_position);
	ClassDB::bind_method(D_METHOD("get_root_motion_rotation"), &AnimationMixer::get_root_motion_rotation);
	ClassDB::bind_method(D_METHOD("get_root_motion_scale"), &AnimationMixer::get_root_motion_scale);
	ClassDB::bind_method(D_METHOD("get_root_motion_position_accumulator"), &AnimationMixer::get_root_motion_position_accumulator);
	ClassDB::bind_method(D_METHOD("get_root_motion_rotation_accumulator"), &AnimationMixer::get_root_motion_rotation_accumulator);
	ClassDB::bind_method(D_METHOD("get_root_motion_scale_accumulator"), &AnimationMixer::get_root_motion_scale_accumulator);

	/* ---- Blending processor ---- */
	ClassDB::bind_method(D_METHOD("clear_caches"), &AnimationMixer::clear_caches);
	ClassDB::bind_method(D_METHOD("advance", "delta"), &AnimationMixer::advance);
	GDVIRTUAL_BIND(_post_process_key_value, "animation", "track", "value", "object", "object_idx");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deterministic"), "set_deterministic", "is_deterministic");

	ClassDB::bind_method(D_METHOD("set_reset_on_save_enabled", "enabled"), &AnimationMixer::set_reset_on_save_enabled);
	ClassDB::bind_method(D_METHOD("is_reset_on_save_enabled"), &AnimationMixer::is_reset_on_save_enabled);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "reset_on_save", PROPERTY_HINT_NONE, ""), "set_reset_on_save_enabled", "is_reset_on_save_enabled");
	ADD_SIGNAL(MethodInfo("mixer_updated")); // For updating dummy player.

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_node"), "set_root_node", "get_root_node");

	ADD_GROUP("Root Motion", "root_motion_");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_motion_track"), "set_root_motion_track", "get_root_motion_track");

	ADD_GROUP("Audio", "audio_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "audio_max_polyphony", PROPERTY_HINT_RANGE, "1,127,1"), "set_audio_max_polyphony", "get_audio_max_polyphony");

	ADD_GROUP("Callback Mode", "callback_mode_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "callback_mode_process", PROPERTY_HINT_ENUM, "Physics,Idle,Manual"), "set_callback_mode_process", "get_callback_mode_process");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "callback_mode_method", PROPERTY_HINT_ENUM, "Deferred,Immediate"), "set_callback_mode_method", "get_callback_mode_method");

	BIND_ENUM_CONSTANT(ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_CALLBACK_MODE_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_CALLBACK_MODE_PROCESS_MANUAL);

	BIND_ENUM_CONSTANT(ANIMATION_CALLBACK_MODE_METHOD_DEFERRED);
	BIND_ENUM_CONSTANT(ANIMATION_CALLBACK_MODE_METHOD_IMMEDIATE);

	ADD_SIGNAL(MethodInfo(SNAME("animation_list_changed")));
	ADD_SIGNAL(MethodInfo(SNAME("animation_libraries_updated")));
	ADD_SIGNAL(MethodInfo(SNAME("animation_finished"), PropertyInfo(Variant::STRING_NAME, "anim_name")));
	ADD_SIGNAL(MethodInfo(SNAME("animation_started"), PropertyInfo(Variant::STRING_NAME, "anim_name")));
	ADD_SIGNAL(MethodInfo(SNAME("caches_cleared")));

	ClassDB::bind_method(D_METHOD("_reset"), &AnimationMixer::reset);
	ClassDB::bind_method(D_METHOD("_restore", "backup"), &AnimationMixer::restore);
}

AnimationMixer::AnimationMixer() {
	root_node = SceneStringNames::get_singleton()->path_pp;
}

AnimationMixer::~AnimationMixer() {
}

void AnimatedValuesBackup::set_data(const HashMap<NodePath, AnimationMixer::TrackCache *> p_data) {
	clear_data();

	for (const KeyValue<NodePath, AnimationMixer::TrackCache *> &E : p_data) {
		AnimationMixer::TrackCache *track = get_cache_copy(E.value);
		if (!track) {
			continue; // Some types of tracks do not get a copy and must be ignored.
		}

		data.insert(E.key, track);
	}
}

HashMap<NodePath, AnimationMixer::TrackCache *> AnimatedValuesBackup::get_data() const {
	HashMap<NodePath, AnimationMixer::TrackCache *> ret;
	for (const KeyValue<NodePath, AnimationMixer::TrackCache *> &E : data) {
		AnimationMixer::TrackCache *track = get_cache_copy(E.value);
		ERR_CONTINUE(!track); // Backup shouldn't contain tracks that cannot be copied, this is a mistake.

		ret.insert(E.key, track);
	}
	return ret;
}

void AnimatedValuesBackup::clear_data() {
	for (KeyValue<NodePath, AnimationMixer::TrackCache *> &K : data) {
		memdelete(K.value);
	}
	data.clear();
}

AnimationMixer::TrackCache *AnimatedValuesBackup::get_cache_copy(AnimationMixer::TrackCache *p_cache) const {
	switch (p_cache->type) {
		case Animation::TYPE_VALUE: {
			AnimationMixer::TrackCacheValue *src = static_cast<AnimationMixer::TrackCacheValue *>(p_cache);
			AnimationMixer::TrackCacheValue *tc = memnew(AnimationMixer::TrackCacheValue(*src));
			return tc;
		}

		case Animation::TYPE_POSITION_3D:
		case Animation::TYPE_ROTATION_3D:
		case Animation::TYPE_SCALE_3D: {
			AnimationMixer::TrackCacheTransform *src = static_cast<AnimationMixer::TrackCacheTransform *>(p_cache);
			AnimationMixer::TrackCacheTransform *tc = memnew(AnimationMixer::TrackCacheTransform(*src));
			return tc;
		}

		case Animation::TYPE_BLEND_SHAPE: {
			AnimationMixer::TrackCacheBlendShape *src = static_cast<AnimationMixer::TrackCacheBlendShape *>(p_cache);
			AnimationMixer::TrackCacheBlendShape *tc = memnew(AnimationMixer::TrackCacheBlendShape(*src));
			return tc;
		}

		case Animation::TYPE_BEZIER: {
			AnimationMixer::TrackCacheBezier *src = static_cast<AnimationMixer::TrackCacheBezier *>(p_cache);
			AnimationMixer::TrackCacheBezier *tc = memnew(AnimationMixer::TrackCacheBezier(*src));
			return tc;
		}

		case Animation::TYPE_AUDIO: {
			AnimationMixer::TrackCacheAudio *src = static_cast<AnimationMixer::TrackCacheAudio *>(p_cache);
			AnimationMixer::TrackCacheAudio *tc = memnew(AnimationMixer::TrackCacheAudio(*src));
			return tc;
		}

		case Animation::TYPE_METHOD:
		case Animation::TYPE_ANIMATION: {
			// Nothing to do here.
		} break;
	}
	return nullptr;
}
