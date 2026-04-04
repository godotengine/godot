/**************************************************************************/
/*  graph_edit.cpp                                                        */
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

#include <godot_cpp/classes/graph_edit.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/graph_frame.hpp>
#include <godot_cpp/classes/h_box_container.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/rect2.hpp>

namespace godot {

Error GraphEdit::connect_node(const StringName &p_from_node, int32_t p_from_port, const StringName &p_to_node, int32_t p_to_port, bool p_keep_alive) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("connect_node")._native_ptr(), 1376144231);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	int8_t p_keep_alive_encoded;
	PtrToArg<bool>::encode(p_keep_alive, &p_keep_alive_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_node, &p_from_port_encoded, &p_to_node, &p_to_port_encoded, &p_keep_alive_encoded);
}

bool GraphEdit::is_node_connected(const StringName &p_from_node, int32_t p_from_port, const StringName &p_to_node, int32_t p_to_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_node_connected")._native_ptr(), 4216241294);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_from_node, &p_from_port_encoded, &p_to_node, &p_to_port_encoded);
}

void GraphEdit::disconnect_node(const StringName &p_from_node, int32_t p_from_port, const StringName &p_to_node, int32_t p_to_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("disconnect_node")._native_ptr(), 1933654315);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_node, &p_from_port_encoded, &p_to_node, &p_to_port_encoded);
}

void GraphEdit::set_connection_activity(const StringName &p_from_node, int32_t p_from_port, const StringName &p_to_node, int32_t p_to_port, float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_connection_activity")._native_ptr(), 1141899943);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_node, &p_from_port_encoded, &p_to_node, &p_to_port_encoded, &p_amount_encoded);
}

void GraphEdit::set_connections(const TypedArray<Dictionary> &p_connections) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_connections")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_connections);
}

TypedArray<Dictionary> GraphEdit::get_connection_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_list")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner);
}

int32_t GraphEdit::get_connection_count(const StringName &p_from_node, int32_t p_from_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_count")._native_ptr(), 861718734);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_node, &p_from_port_encoded);
}

Dictionary GraphEdit::get_closest_connection_at_point(const Vector2 &p_point, float p_max_distance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_closest_connection_at_point")._native_ptr(), 453879819);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	double p_max_distance_encoded;
	PtrToArg<double>::encode(p_max_distance, &p_max_distance_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_point, &p_max_distance_encoded);
}

TypedArray<Dictionary> GraphEdit::get_connection_list_from_node(const StringName &p_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_list_from_node")._native_ptr(), 3147814860);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_node);
}

TypedArray<Dictionary> GraphEdit::get_connections_intersecting_with_rect(const Rect2 &p_rect) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connections_intersecting_with_rect")._native_ptr(), 2709748719);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_rect);
}

void GraphEdit::clear_connections() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("clear_connections")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void GraphEdit::force_connection_drag_end() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("force_connection_drag_end")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2 GraphEdit::get_scroll_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_scroll_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void GraphEdit::set_scroll_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_scroll_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

void GraphEdit::add_valid_right_disconnect_type(int32_t p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("add_valid_right_disconnect_type")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

void GraphEdit::remove_valid_right_disconnect_type(int32_t p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("remove_valid_right_disconnect_type")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

void GraphEdit::add_valid_left_disconnect_type(int32_t p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("add_valid_left_disconnect_type")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

void GraphEdit::remove_valid_left_disconnect_type(int32_t p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("remove_valid_left_disconnect_type")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

void GraphEdit::add_valid_connection_type(int32_t p_from_type, int32_t p_to_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("add_valid_connection_type")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_type_encoded;
	PtrToArg<int64_t>::encode(p_from_type, &p_from_type_encoded);
	int64_t p_to_type_encoded;
	PtrToArg<int64_t>::encode(p_to_type, &p_to_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_type_encoded, &p_to_type_encoded);
}

void GraphEdit::remove_valid_connection_type(int32_t p_from_type, int32_t p_to_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("remove_valid_connection_type")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_from_type_encoded;
	PtrToArg<int64_t>::encode(p_from_type, &p_from_type_encoded);
	int64_t p_to_type_encoded;
	PtrToArg<int64_t>::encode(p_to_type, &p_to_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from_type_encoded, &p_to_type_encoded);
}

bool GraphEdit::is_valid_connection_type(int32_t p_from_type, int32_t p_to_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_valid_connection_type")._native_ptr(), 2522259332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_from_type_encoded;
	PtrToArg<int64_t>::encode(p_from_type, &p_from_type_encoded);
	int64_t p_to_type_encoded;
	PtrToArg<int64_t>::encode(p_to_type, &p_to_type_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_from_type_encoded, &p_to_type_encoded);
}

PackedVector2Array GraphEdit::get_connection_line(const Vector2 &p_from_node, const Vector2 &p_to_node) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_line")._native_ptr(), 3932192302);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_from_node, &p_to_node);
}

void GraphEdit::attach_graph_element_to_frame(const StringName &p_element, const StringName &p_frame) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("attach_graph_element_to_frame")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_element, &p_frame);
}

void GraphEdit::detach_graph_element_from_frame(const StringName &p_element) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("detach_graph_element_from_frame")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_element);
}

GraphFrame *GraphEdit::get_element_frame(const StringName &p_element) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_element_frame")._native_ptr(), 988084372);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<GraphFrame>(_gde_method_bind, _owner, &p_element);
}

TypedArray<StringName> GraphEdit::get_attached_nodes_of_frame(const StringName &p_frame) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_attached_nodes_of_frame")._native_ptr(), 689397652);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner, &p_frame);
}

void GraphEdit::set_panning_scheme(GraphEdit::PanningScheme p_scheme) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_panning_scheme")._native_ptr(), 18893313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scheme_encoded;
	PtrToArg<int64_t>::encode(p_scheme, &p_scheme_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scheme_encoded);
}

GraphEdit::PanningScheme GraphEdit::get_panning_scheme() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_panning_scheme")._native_ptr(), 549924446);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GraphEdit::PanningScheme(0)));
	return (GraphEdit::PanningScheme)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_zoom(float p_zoom) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_zoom")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zoom_encoded;
	PtrToArg<double>::encode(p_zoom, &p_zoom_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zoom_encoded);
}

float GraphEdit::get_zoom() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_zoom")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_zoom_min(float p_zoom_min) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_zoom_min")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zoom_min_encoded;
	PtrToArg<double>::encode(p_zoom_min, &p_zoom_min_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zoom_min_encoded);
}

float GraphEdit::get_zoom_min() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_zoom_min")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_zoom_max(float p_zoom_max) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_zoom_max")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zoom_max_encoded;
	PtrToArg<double>::encode(p_zoom_max, &p_zoom_max_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zoom_max_encoded);
}

float GraphEdit::get_zoom_max() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_zoom_max")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_zoom_step(float p_zoom_step) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_zoom_step")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_zoom_step_encoded;
	PtrToArg<double>::encode(p_zoom_step, &p_zoom_step_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_zoom_step_encoded);
}

float GraphEdit::get_zoom_step() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_zoom_step")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_grid(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_grid")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool GraphEdit::is_showing_grid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_grid")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_grid_pattern(GraphEdit::GridPattern p_pattern) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_grid_pattern")._native_ptr(), 1074098205);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_pattern_encoded;
	PtrToArg<int64_t>::encode(p_pattern, &p_pattern_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pattern_encoded);
}

GraphEdit::GridPattern GraphEdit::get_grid_pattern() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_grid_pattern")._native_ptr(), 1286127528);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GraphEdit::GridPattern(0)));
	return (GraphEdit::GridPattern)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_snapping_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_snapping_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool GraphEdit::is_snapping_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_snapping_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_snapping_distance(int32_t p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_snapping_distance")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_pixels_encoded;
	PtrToArg<int64_t>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixels_encoded);
}

int32_t GraphEdit::get_snapping_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_snapping_distance")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_connection_lines_curvature(float p_curvature) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_connection_lines_curvature")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curvature_encoded;
	PtrToArg<double>::encode(p_curvature, &p_curvature_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_curvature_encoded);
}

float GraphEdit::get_connection_lines_curvature() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_lines_curvature")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_connection_lines_thickness(float p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_connection_lines_thickness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixels_encoded;
	PtrToArg<double>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixels_encoded);
}

float GraphEdit::get_connection_lines_thickness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_connection_lines_thickness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_connection_lines_antialiased(bool p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_connection_lines_antialiased")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_pixels_encoded;
	PtrToArg<bool>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixels_encoded);
}

bool GraphEdit::is_connection_lines_antialiased() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_connection_lines_antialiased")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_minimap_size(const Vector2 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_minimap_size")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2 GraphEdit::get_minimap_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_minimap_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void GraphEdit::set_minimap_opacity(float p_opacity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_minimap_opacity")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_opacity_encoded;
	PtrToArg<double>::encode(p_opacity, &p_opacity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_opacity_encoded);
}

float GraphEdit::get_minimap_opacity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_minimap_opacity")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GraphEdit::set_minimap_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_minimap_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool GraphEdit::is_minimap_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_minimap_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_menu(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_menu")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool GraphEdit::is_showing_menu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_menu")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_zoom_label(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_zoom_label")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool GraphEdit::is_showing_zoom_label() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_zoom_label")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_grid_buttons(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_grid_buttons")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool GraphEdit::is_showing_grid_buttons() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_grid_buttons")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_zoom_buttons(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_zoom_buttons")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool GraphEdit::is_showing_zoom_buttons() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_zoom_buttons")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_minimap_button(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_minimap_button")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool GraphEdit::is_showing_minimap_button() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_minimap_button")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_show_arrange_button(bool p_hidden) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_show_arrange_button")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_hidden_encoded;
	PtrToArg<bool>::encode(p_hidden, &p_hidden_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_hidden_encoded);
}

bool GraphEdit::is_showing_arrange_button() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_showing_arrange_button")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_right_disconnects(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_right_disconnects")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool GraphEdit::is_right_disconnects_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("is_right_disconnects_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void GraphEdit::set_type_names(const Dictionary &p_type_names) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_type_names")._native_ptr(), 4155329257);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_names);
}

Dictionary GraphEdit::get_type_names() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_type_names")._native_ptr(), 3102165223);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner);
}

HBoxContainer *GraphEdit::get_menu_hbox() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("get_menu_hbox")._native_ptr(), 3590609951);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<HBoxContainer>(_gde_method_bind, _owner);
}

void GraphEdit::arrange_nodes() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("arrange_nodes")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void GraphEdit::set_selected(Node *p_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GraphEdit::get_class_static()._native_ptr(), StringName("set_selected")._native_ptr(), 1078189570);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr));
}

bool GraphEdit::_is_in_input_hotzone(Object *p_in_node, int32_t p_in_port, const Vector2 &p_mouse_position) {
	return false;
}

bool GraphEdit::_is_in_output_hotzone(Object *p_in_node, int32_t p_in_port, const Vector2 &p_mouse_position) {
	return false;
}

PackedVector2Array GraphEdit::_get_connection_line(const Vector2 &p_from_position, const Vector2 &p_to_position) const {
	return PackedVector2Array();
}

bool GraphEdit::_is_node_hover_valid(const StringName &p_from_node, int32_t p_from_port, const StringName &p_to_node, int32_t p_to_port) {
	return false;
}

} // namespace godot
