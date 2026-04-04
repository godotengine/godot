/**************************************************************************/
/*  packed_scene.cpp                                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/packed_scene.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/scene_state.hpp>

namespace godot {

Error PackedScene::pack(Node *p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PackedScene::get_class_static()._native_ptr(), StringName("pack")._native_ptr(), 2584678054);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_path != nullptr ? &p_path->_owner : nullptr));
}

Node *PackedScene::instantiate(PackedScene::GenEditState p_edit_state) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PackedScene::get_class_static()._native_ptr(), StringName("instantiate")._native_ptr(), 2628778455);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	int64_t p_edit_state_encoded;
	PtrToArg<int64_t>::encode(p_edit_state, &p_edit_state_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, &p_edit_state_encoded);
}

bool PackedScene::can_instantiate() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PackedScene::get_class_static()._native_ptr(), StringName("can_instantiate")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Ref<SceneState> PackedScene::get_state() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(PackedScene::get_class_static()._native_ptr(), StringName("get_state")._native_ptr(), 3479783971);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SceneState>()));
	return Ref<SceneState>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SceneState>(_gde_method_bind, _owner));
}

} // namespace godot
