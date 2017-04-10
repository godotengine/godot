/*************************************************************************/
/*  godot_plane.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "godot_plane.h"

#include "math/plane.h"

#ifdef __cplusplus
extern "C" {
#endif

void _plane_api_anchor() {
}

void GDAPI godot_plane_new(godot_plane *p_pl) {
	Plane *pl = (Plane *)p_pl;
	*pl = Plane();
}

void GDAPI godot_plane_new_with_normal(godot_plane *p_pl, const godot_vector3 *p_normal, const godot_real p_d) {
	Plane *pl = (Plane *)p_pl;
	const Vector3 *normal = (const Vector3 *)p_normal;
	*pl = Plane(*normal, p_d);
}

void GDAPI godot_plane_set_normal(godot_plane *p_pl, const godot_vector3 *p_normal) {
	Plane *pl = (Plane *)p_pl;
	const Vector3 *normal = (const Vector3 *)p_normal;
	pl->set_normal(*normal);
}

godot_vector3 godot_plane_get_normal(const godot_plane *p_pl) {
	const Plane *pl = (const Plane *)p_pl;
	const Vector3 normal = pl->get_normal();
	godot_vector3 *v3 = (godot_vector3 *)&normal;
	return *v3;
}

void GDAPI godot_plane_set_d(godot_plane *p_pl, const godot_real p_d) {
	Plane *pl = (Plane *)p_pl;
	pl->d = p_d;
}

godot_real GDAPI godot_plane_get_d(const godot_plane *p_pl) {
	const Plane *pl = (const Plane *)p_pl;
	return pl->d;
}

#ifdef __cplusplus
}
#endif
