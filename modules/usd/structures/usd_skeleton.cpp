/**************************************************************************/
/*  usd_skeleton.cpp                                                      */
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

#include "usd_skeleton.h"

void USDSkeleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_joint_paths"), &USDSkeleton::get_joint_paths);
	ClassDB::bind_method(D_METHOD("set_joint_paths", "joint_paths"), &USDSkeleton::set_joint_paths);
	ClassDB::bind_method(D_METHOD("get_joint_parents"), &USDSkeleton::get_joint_parents);
	ClassDB::bind_method(D_METHOD("set_joint_parents", "joint_parents"), &USDSkeleton::set_joint_parents);
	ClassDB::bind_method(D_METHOD("get_bind_transforms"), &USDSkeleton::get_bind_transforms);
	ClassDB::bind_method(D_METHOD("set_bind_transforms", "bind_transforms"), &USDSkeleton::set_bind_transforms);
	ClassDB::bind_method(D_METHOD("get_rest_transforms"), &USDSkeleton::get_rest_transforms);
	ClassDB::bind_method(D_METHOD("set_rest_transforms", "rest_transforms"), &USDSkeleton::set_rest_transforms);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "joint_paths"), "set_joint_paths", "get_joint_paths");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "joint_parents"), "set_joint_parents", "get_joint_parents");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "bind_transforms"), "set_bind_transforms", "get_bind_transforms");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "rest_transforms"), "set_rest_transforms", "get_rest_transforms");
}

Vector<String> USDSkeleton::get_joint_paths() const {
	return joint_paths;
}

void USDSkeleton::set_joint_paths(const Vector<String> &p_joint_paths) {
	joint_paths = p_joint_paths;
}

Vector<int> USDSkeleton::get_joint_parents() const {
	return joint_parents;
}

void USDSkeleton::set_joint_parents(const Vector<int> &p_joint_parents) {
	joint_parents = p_joint_parents;
}

TypedArray<Transform3D> USDSkeleton::get_bind_transforms() const {
	TypedArray<Transform3D> ret;
	for (int i = 0; i < bind_transforms.size(); i++) {
		ret.push_back(bind_transforms[i]);
	}
	return ret;
}

void USDSkeleton::set_bind_transforms(const TypedArray<Transform3D> &p_bind_transforms) {
	bind_transforms.clear();
	for (int i = 0; i < p_bind_transforms.size(); i++) {
		bind_transforms.push_back(p_bind_transforms[i]);
	}
}

TypedArray<Transform3D> USDSkeleton::get_rest_transforms() const {
	TypedArray<Transform3D> ret;
	for (int i = 0; i < rest_transforms.size(); i++) {
		ret.push_back(rest_transforms[i]);
	}
	return ret;
}

void USDSkeleton::set_rest_transforms(const TypedArray<Transform3D> &p_rest_transforms) {
	rest_transforms.clear();
	for (int i = 0; i < p_rest_transforms.size(); i++) {
		rest_transforms.push_back(p_rest_transforms[i]);
	}
}
