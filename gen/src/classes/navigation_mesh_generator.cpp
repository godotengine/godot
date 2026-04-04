/**************************************************************************/
/*  navigation_mesh_generator.cpp                                         */
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

#include <godot_cpp/classes/navigation_mesh_generator.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/navigation_mesh.hpp>
#include <godot_cpp/classes/navigation_mesh_source_geometry_data3d.hpp>
#include <godot_cpp/classes/node.hpp>

namespace godot {

NavigationMeshGenerator *NavigationMeshGenerator::singleton = nullptr;

NavigationMeshGenerator *NavigationMeshGenerator::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(NavigationMeshGenerator::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<NavigationMeshGenerator *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &NavigationMeshGenerator::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(NavigationMeshGenerator::get_class_static(), singleton);
		}
	}
	return singleton;
}

NavigationMeshGenerator::~NavigationMeshGenerator() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(NavigationMeshGenerator::get_class_static());
		singleton = nullptr;
	}
}

void NavigationMeshGenerator::bake(const Ref<NavigationMesh> &p_navigation_mesh, Node *p_root_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshGenerator::get_class_static()._native_ptr(), StringName("bake")._native_ptr(), 1401173477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_root_node != nullptr ? &p_root_node->_owner : nullptr));
}

void NavigationMeshGenerator::clear(const Ref<NavigationMesh> &p_navigation_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshGenerator::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 2923361153);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr));
}

void NavigationMeshGenerator::parse_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, Node *p_root_node, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshGenerator::get_class_static()._native_ptr(), StringName("parse_source_geometry_data")._native_ptr(), 3172802542);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), (p_root_node != nullptr ? &p_root_node->_owner : nullptr), &p_callback);
}

void NavigationMeshGenerator::bake_from_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshGenerator::get_class_static()._native_ptr(), StringName("bake_from_source_geometry_data")._native_ptr(), 1286748856);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), &p_callback);
}

} // namespace godot
