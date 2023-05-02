/**************************************************************************/
/* iteration_server.cpp                                                   */
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

#include "iteration_server.h"
#include "core/config/project_settings.h"
#include "custom_iterator.h"

void IterationServer::_bind_methods() {
	BIND_ENUM_CONSTANT(ITERATOR_TYPE_UNSET);
	BIND_ENUM_CONSTANT(ITERATOR_TYPE_MIXED);
	BIND_ENUM_CONSTANT(ITERATOR_TYPE_SEPARATE);

	ClassDB::bind_static_method("IterationServer", D_METHOD("get_iterator_count"), IterationServer::get_iterator_count);
	ClassDB::bind_static_method("IterationServer", D_METHOD("get_iterator", "index"), IterationServer::get_iterator);
	ClassDB::bind_static_method("IterationServer", D_METHOD("register_iterator", "iterator"), IterationServer::register_iterator);
	ClassDB::bind_static_method("IterationServer", D_METHOD("unregister_iterator", "iterator"), IterationServer::unregister_iterator);
}

/* Static members */
CustomIterator *IterationServer::_iterators[MAX_ITERATORS];
int IterationServer::_iterator_count = 0;
CustomIterator *IterationServer::get_iterator(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, _iterator_count, nullptr);

	return _iterators[p_idx];
}
Error IterationServer::register_iterator(CustomIterator *p_iterator) {
	ERR_FAIL_NULL_V(p_iterator, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(_iterator_count >= MAX_ITERATORS, ERR_UNAVAILABLE, "Iterator limit has been reach, cannot register more.");
	for (int i = 0; i < _iterator_count; i++) {
		const CustomIterator *other_iterator = _iterators[i];
		ERR_FAIL_COND_V_MSG(other_iterator->get_name() == p_iterator->get_name(), ERR_ALREADY_EXISTS, "An iterator with name '" + p_iterator->get_name() + "' is already registered.");
	}
	_iterators[_iterator_count++] = p_iterator;
	return OK;
}
Error IterationServer::unregister_iterator(const CustomIterator *p_iterator) {
	for (int i = 0; i < _iterator_count; i++) {
		if (_iterators[i] == p_iterator) {
			_iterator_count--;
			if (i < _iterator_count) {
				SWAP(_iterators[i], _iterators[_iterator_count]);
			}
			return OK;
		}
	}
	return ERR_DOES_NOT_EXIST;
}
bool IterationServer::is_iterator_enabled_for_process(const String &name) {
	PackedStringArray enabled_process_iterators = ProjectSettings::get_singleton()->get_setting("application/config/process_iterators");

	if (enabled_process_iterators.is_empty()) {
		return true;
	}

	return enabled_process_iterators.has(name);
}
bool IterationServer::is_iterator_enabled_for_process(const CustomIterator *p_iterator) {
	ERR_FAIL_COND_V_MSG(p_iterator == nullptr, false, "iterator is null");

	if (!(p_iterator->get_type() & IteratorType::ITERATOR_TYPE_SEPARATE)) {
		return false;
	}
	return is_iterator_enabled_for_process(p_iterator->get_name());
}
bool IterationServer::is_iterator_enabled_for_physics(const String &name) {
	PackedStringArray enabled_physics_iterators = ProjectSettings::get_singleton()->get_setting("application/config/physics_iterators");

	if (enabled_physics_iterators.is_empty()) {
		return true;
	}

	return enabled_physics_iterators.has(name);
}
bool IterationServer::is_iterator_enabled_for_physics(const CustomIterator *p_iterator) {
	ERR_FAIL_COND_V_MSG(p_iterator == nullptr, false, "iterator is null");

	if (!(p_iterator->get_type() & IteratorType::ITERATOR_TYPE_SEPARATE)) {
		return false;
	}
	return is_iterator_enabled_for_physics(p_iterator->get_name());
}
bool IterationServer::is_iterator_enabled_for_audio(const String &name) {
	PackedStringArray enabled_audio_iterators = ProjectSettings::get_singleton()->get_setting("application/config/audio_iterators");

	return enabled_audio_iterators.is_empty() || enabled_audio_iterators.has(name);
}
bool IterationServer::is_iterator_enabled_for_audio(const CustomIterator *p_iterator) {
	ERR_FAIL_COND_V_MSG(p_iterator == nullptr, false, "iterator is null");

	if (!(p_iterator->get_type() & IteratorType::ITERATOR_TYPE_SEPARATE)) {
		return false;
	}
	return is_iterator_enabled_for_audio(p_iterator->get_name());
}
bool IterationServer::is_iterator_enabled_for_mixed(const String &name) {
	PackedStringArray enabled_mixed_iterators = ProjectSettings::get_singleton()->get_setting("application/config/mixed_iterators");
	return enabled_mixed_iterators.is_empty() || enabled_mixed_iterators.has(name);
}
bool IterationServer::is_iterator_enabled_for_mixed(const CustomIterator *p_iterator) {
	ERR_FAIL_COND_V_MSG(p_iterator == nullptr, false, "iterator is null");

	if (!(p_iterator->get_type() & IteratorType::ITERATOR_TYPE_MIXED)) {
		return false;
	}
	return is_iterator_enabled_for_mixed(p_iterator->get_name());
}