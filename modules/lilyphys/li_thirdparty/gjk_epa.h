//
// Created by amara on 26/11/2021.
//

#ifndef LILYPHYS_GJK_EPA_H
#define LILYPHYS_GJK_EPA_H

// Copy of Godot's copy of Bullet's implementation of GJK+EPA
// A sin against god, but not my god!

/*************************************************************************/
/*  gjk_epa.h                                                            */
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

#include "../l_collision_solver.h"
#include "../internal/li_shape.h"

typedef void (*CallbackResult)(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata);
struct GJKResult {
    Vector3 normal;
    Vector3 position;
    real_t depth;
};

bool gjk_epa_calculate_penetration(RID p_shape_A, const Transform &p_transform_A, RID p_shape_B, const Transform &p_transform_B, GJKResult &p_result, real_t p_margin_A = 0.0, real_t p_margin_B = 0.0);
bool gjk_epa_calculate_distance(RID p_shape_A, const Transform &p_transform_A, RID p_shape_B, const Transform &p_transform_B, Vector3 &r_result_A, Vector3 &r_result_B);

#endif //LILYPHYS_GJK_EPA_H
