/**************************************************************************/
/*  a_star.compat.inc                                                     */
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

#ifndef DISABLE_DEPRECATED

Vector<int64_t> AStar3D::_get_id_path_bind_compat_88047(int64_t p_from_id, int64_t p_to_id) {
	return get_id_path(p_from_id, p_to_id, false);
}

Vector<Vector3> AStar3D::_get_point_path_bind_compat_88047(int64_t p_from_id, int64_t p_to_id) {
	return get_point_path(p_from_id, p_to_id, false);
}

void AStar3D::_add_point_bind_compat_98324(int64_t p_id, const Vector3 &p_pos, real_t p_weight_scale) {
	add_point(p_id, p_pos, p_weight_scale, 1);
}

int64_t AStar3D::_get_closest_point_bind_compat_98324(const Vector3 &p_point, bool p_include_disabled) const {
	return get_closest_point(p_point, p_include_disabled, 1);
}

Vector<int64_t> AStar3D::_get_id_path_bind_compat_98324(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	return get_id_path(p_from_id, p_to_id, p_allow_partial_path, 1);
}

Vector<Vector3> AStar3D::_get_point_path_bind_compat_98324(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	return get_point_path(p_from_id, p_to_id, p_allow_partial_path, 1);
}

void AStar3D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_id_path", "from_id", "to_id"), &AStar3D::_get_id_path_bind_compat_88047);
	ClassDB::bind_compatibility_method(D_METHOD("get_point_path", "from_id", "to_id"), &AStar3D::_get_point_path_bind_compat_88047);
	ClassDB::bind_compatibility_method(D_METHOD("add_point", "id", "position", "weight_scale"), &AStar3D::_add_point_bind_compat_98324, DEFVAL(1.0));
	ClassDB::bind_compatibility_method(D_METHOD("get_closest_point", "to_position", "include_disabled"), &AStar3D::_get_closest_point_bind_compat_98324, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_id_path", "from_id", "to_id", "allow_partial_path"), &AStar3D::_get_id_path_bind_compat_98324, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_point_path", "from_id", "to_id", "allow_partial_path"), &AStar3D::_get_point_path_bind_compat_98324, DEFVAL(false));
}

Vector<int64_t> AStar2D::_get_id_path_bind_compat_88047(int64_t p_from_id, int64_t p_to_id) {
	return get_id_path(p_from_id, p_to_id, false);
}

Vector<Vector2> AStar2D::_get_point_path_bind_compat_88047(int64_t p_from_id, int64_t p_to_id) {
	return get_point_path(p_from_id, p_to_id, false);
}

void AStar2D::_add_point_bind_compat_98324(int64_t p_id, const Vector2 &p_pos, real_t p_weight_scale) {
	add_point(p_id, p_pos, p_weight_scale, 1);
}

int64_t AStar2D::_get_closest_point_bind_compat_98324(const Vector2 &p_point, bool p_include_disabled) const {
	return get_closest_point(p_point, p_include_disabled, 1);
}

Vector<int64_t> AStar2D::_get_id_path_bind_compat_98324(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	return get_id_path(p_from_id, p_to_id, p_allow_partial_path, 1);
}

Vector<Vector2> AStar2D::_get_point_path_bind_compat_98324(int64_t p_from_id, int64_t p_to_id, bool p_allow_partial_path) {
	return get_point_path(p_from_id, p_to_id, p_allow_partial_path, 1);
}

void AStar2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_id_path", "from_id", "to_id"), &AStar2D::_get_id_path_bind_compat_88047);
	ClassDB::bind_compatibility_method(D_METHOD("get_point_path", "from_id", "to_id"), &AStar2D::_get_point_path_bind_compat_88047);
	ClassDB::bind_compatibility_method(D_METHOD("add_point", "id", "position", "weight_scale"), &AStar2D::_add_point_bind_compat_98324, DEFVAL(1.0));
	ClassDB::bind_compatibility_method(D_METHOD("get_closest_point", "to_position", "include_disabled"), &AStar2D::_get_closest_point_bind_compat_98324, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_id_path", "from_id", "to_id", "allow_partial_path"), &AStar2D::_get_id_path_bind_compat_98324, DEFVAL(false));
	ClassDB::bind_compatibility_method(D_METHOD("get_point_path", "from_id", "to_id", "allow_partial_path"), &AStar2D::_get_point_path_bind_compat_98324, DEFVAL(false));
}

#endif // DISABLE_DEPRECATED
