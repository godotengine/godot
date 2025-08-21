/**************************************************************************/
/*  packed_arrays.cpp                                                     */
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

// extra functions for packed arrays

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_vector4_array.hpp>

namespace godot {

const uint8_t &PackedByteArray::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_packed_byte_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
}

uint8_t &PackedByteArray::operator[](int64_t p_index) {
	return *internal::gdextension_interface_packed_byte_array_operator_index((GDExtensionTypePtr *)this, p_index);
}

const uint8_t *PackedByteArray::ptr() const {
	return internal::gdextension_interface_packed_byte_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

uint8_t *PackedByteArray::ptrw() {
	return internal::gdextension_interface_packed_byte_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Color &PackedColorArray::operator[](int64_t p_index) const {
	const Color *color = (const Color *)internal::gdextension_interface_packed_color_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *color;
}

Color &PackedColorArray::operator[](int64_t p_index) {
	Color *color = (Color *)internal::gdextension_interface_packed_color_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *color;
}

const Color *PackedColorArray::ptr() const {
	return (const Color *)internal::gdextension_interface_packed_color_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

Color *PackedColorArray::ptrw() {
	return (Color *)internal::gdextension_interface_packed_color_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const float &PackedFloat32Array::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_packed_float32_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
}

float &PackedFloat32Array::operator[](int64_t p_index) {
	return *internal::gdextension_interface_packed_float32_array_operator_index((GDExtensionTypePtr *)this, p_index);
}

const float *PackedFloat32Array::ptr() const {
	return internal::gdextension_interface_packed_float32_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

float *PackedFloat32Array::ptrw() {
	return internal::gdextension_interface_packed_float32_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const double &PackedFloat64Array::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_packed_float64_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
}

double &PackedFloat64Array::operator[](int64_t p_index) {
	return *internal::gdextension_interface_packed_float64_array_operator_index((GDExtensionTypePtr *)this, p_index);
}

const double *PackedFloat64Array::ptr() const {
	return internal::gdextension_interface_packed_float64_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

double *PackedFloat64Array::ptrw() {
	return internal::gdextension_interface_packed_float64_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const int32_t &PackedInt32Array::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_packed_int32_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
}

int32_t &PackedInt32Array::operator[](int64_t p_index) {
	return *internal::gdextension_interface_packed_int32_array_operator_index((GDExtensionTypePtr *)this, p_index);
}

const int32_t *PackedInt32Array::ptr() const {
	return internal::gdextension_interface_packed_int32_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

int32_t *PackedInt32Array::ptrw() {
	return internal::gdextension_interface_packed_int32_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const int64_t &PackedInt64Array::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_packed_int64_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
}

int64_t &PackedInt64Array::operator[](int64_t p_index) {
	return *internal::gdextension_interface_packed_int64_array_operator_index((GDExtensionTypePtr *)this, p_index);
}

const int64_t *PackedInt64Array::ptr() const {
	return internal::gdextension_interface_packed_int64_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

int64_t *PackedInt64Array::ptrw() {
	return internal::gdextension_interface_packed_int64_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const String &PackedStringArray::operator[](int64_t p_index) const {
	const String *string = (const String *)internal::gdextension_interface_packed_string_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *string;
}

String &PackedStringArray::operator[](int64_t p_index) {
	String *string = (String *)internal::gdextension_interface_packed_string_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *string;
}

const String *PackedStringArray::ptr() const {
	return (const String *)internal::gdextension_interface_packed_string_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

String *PackedStringArray::ptrw() {
	return (String *)internal::gdextension_interface_packed_string_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Vector2 &PackedVector2Array::operator[](int64_t p_index) const {
	const Vector2 *vec = (const Vector2 *)internal::gdextension_interface_packed_vector2_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

Vector2 &PackedVector2Array::operator[](int64_t p_index) {
	Vector2 *vec = (Vector2 *)internal::gdextension_interface_packed_vector2_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

const Vector2 *PackedVector2Array::ptr() const {
	return (const Vector2 *)internal::gdextension_interface_packed_vector2_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

Vector2 *PackedVector2Array::ptrw() {
	return (Vector2 *)internal::gdextension_interface_packed_vector2_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Vector3 &PackedVector3Array::operator[](int64_t p_index) const {
	const Vector3 *vec = (const Vector3 *)internal::gdextension_interface_packed_vector3_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

Vector3 &PackedVector3Array::operator[](int64_t p_index) {
	Vector3 *vec = (Vector3 *)internal::gdextension_interface_packed_vector3_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

const Vector3 *PackedVector3Array::ptr() const {
	return (const Vector3 *)internal::gdextension_interface_packed_vector3_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

Vector3 *PackedVector3Array::ptrw() {
	return (Vector3 *)internal::gdextension_interface_packed_vector3_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Vector4 &PackedVector4Array::operator[](int64_t p_index) const {
	const Vector4 *vec = (const Vector4 *)internal::gdextension_interface_packed_vector4_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

Vector4 &PackedVector4Array::operator[](int64_t p_index) {
	Vector4 *vec = (Vector4 *)internal::gdextension_interface_packed_vector4_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *vec;
}

const Vector4 *PackedVector4Array::ptr() const {
	return (const Vector4 *)internal::gdextension_interface_packed_vector4_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

Vector4 *PackedVector4Array::ptrw() {
	return (Vector4 *)internal::gdextension_interface_packed_vector4_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Variant &Array::operator[](int64_t p_index) const {
	const Variant *var = (const Variant *)internal::gdextension_interface_array_operator_index_const((GDExtensionTypePtr *)this, p_index);
	return *var;
}

Variant &Array::operator[](int64_t p_index) {
	Variant *var = (Variant *)internal::gdextension_interface_array_operator_index((GDExtensionTypePtr *)this, p_index);
	return *var;
}

void Array::set_typed(uint32_t p_type, const StringName &p_class_name, const Variant &p_script) {
	// p_type is not Variant::Type so that header doesn't depend on <variant.hpp>.
	internal::gdextension_interface_array_set_typed((GDExtensionTypePtr *)this, (GDExtensionVariantType)p_type, (GDExtensionConstStringNamePtr)&p_class_name, (GDExtensionConstVariantPtr)&p_script);
}

const Variant *Array::ptr() const {
	return (const Variant *)internal::gdextension_interface_array_operator_index_const((GDExtensionTypePtr *)this, 0);
}

Variant *Array::ptrw() {
	return (Variant *)internal::gdextension_interface_array_operator_index((GDExtensionTypePtr *)this, 0);
}

const Variant &Dictionary::operator[](const Variant &p_key) const {
	const Variant *var = (const Variant *)internal::gdextension_interface_dictionary_operator_index_const((GDExtensionTypePtr *)this, (GDExtensionVariantPtr)&p_key);
	return *var;
}

Variant &Dictionary::operator[](const Variant &p_key) {
	Variant *var = (Variant *)internal::gdextension_interface_dictionary_operator_index((GDExtensionTypePtr *)this, (GDExtensionVariantPtr)&p_key);
	return *var;
}

void Dictionary::set_typed(uint32_t p_key_type, const StringName &p_key_class_name, const Variant &p_key_script, uint32_t p_value_type, const StringName &p_value_class_name, const Variant &p_value_script) {
	// p_key_type/p_value_type are not Variant::Type so that header doesn't depend on <variant.hpp>.
	internal::gdextension_interface_dictionary_set_typed((GDExtensionTypePtr *)this, (GDExtensionVariantType)p_key_type, (GDExtensionConstStringNamePtr)&p_key_class_name, (GDExtensionConstVariantPtr)&p_key_script,
			(GDExtensionVariantType)p_value_type, (GDExtensionConstStringNamePtr)&p_value_class_name, (GDExtensionConstVariantPtr)&p_value_script);
}

} // namespace godot
