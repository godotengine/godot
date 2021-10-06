/*************************************************************************/
/*  importer_animation_container.cpp                                     */
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

#include "importer_animation_container.h"
#include "core/templates/pair.h"

void ImporterAnimationContainer::add_animation(const StringName &p_name, Ref<ImporterAnimation> p_animation) {
	ERR_FAIL_COND(animations.has(p_name));
	animations[p_name] = p_animation;
}
void ImporterAnimationContainer::set_animation(const StringName &p_name, Ref<ImporterAnimation> p_animation) {
	ERR_FAIL_COND(!animations.has(p_name));
	animations[p_name] = p_animation;
}
void ImporterAnimationContainer::rename_animation(const StringName &p_name, const StringName &p_to_name) {
	ERR_FAIL_COND(animations.has(p_to_name));
	Ref<ImporterAnimation> anim = animations[p_name];
	animations.erase(p_name);
	animations[p_to_name] = p_name;
}
Ref<ImporterAnimation> ImporterAnimationContainer::get_animation(const StringName &p_name) const {
	ERR_FAIL_COND_V(!animations.has(p_name), Ref<ImporterAnimation>());
	return animations[p_name];
}
void ImporterAnimationContainer::remove_animation(const StringName &p_name) {
	animations.erase(p_name);
}
void ImporterAnimationContainer::clear() {
	animations.clear();
}

Vector<StringName> ImporterAnimationContainer::get_animation_list() {
	Vector<StringName> anims;
	for (const KeyValue<StringName, Ref<ImporterAnimation>> &I : animations) {
		anims.push_back(I.key);
	}
	return anims;
}

TypedArray<StringName> ImporterAnimationContainer::_get_animation_list() {
	TypedArray<StringName> anims;
	for (const KeyValue<StringName, Ref<ImporterAnimation>> &I : animations) {
		anims.push_back(I.key);
	}
	return anims;
}

Dictionary ImporterAnimationContainer::_get_animations() const {
	Dictionary d;
	for (const KeyValue<StringName, Ref<ImporterAnimation>> &I : animations) {
		d[I.key] = I.value;
	}
	return d;
}
void ImporterAnimationContainer::_set_animations(const Dictionary &p_animations) {
	List<Variant> keys;
	p_animations.get_key_list(&keys);
	clear();

	for (Variant K : keys) {
		add_animation(K, p_animations[K]);
	}
}

void ImporterAnimationContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_animation", "name", "animation"), &ImporterAnimationContainer::add_animation);
	ClassDB::bind_method(D_METHOD("set_animation", "name", "animation"), &ImporterAnimationContainer::set_animation);
	ClassDB::bind_method(D_METHOD("get_animation", "name"), &ImporterAnimationContainer::get_animation);
	ClassDB::bind_method(D_METHOD("rename_animation", "name", "to_name"), &ImporterAnimationContainer::rename_animation);
	ClassDB::bind_method(D_METHOD("remove_animation", "name"), &ImporterAnimationContainer::remove_animation);
	ClassDB::bind_method(D_METHOD("clear"), &ImporterAnimationContainer::clear);
	ClassDB::bind_method(D_METHOD("get_animation_list"), &ImporterAnimationContainer::_get_animation_list);

	ClassDB::bind_method(D_METHOD("_set_animations", "animations"), &ImporterAnimationContainer::_set_animations);
	ClassDB::bind_method(D_METHOD("_get_animations"), &ImporterAnimationContainer::_get_animations);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_animations", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_animations", "_get_animations");
}

ImporterAnimationContainer::ImporterAnimationContainer() {
}
