/*************************************************************************/
/*  bullet_types_converter.h                                             */
/*  Author: AndreaCatania                                                */
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

#ifndef BULLET_TYPES_CONVERTER_H
#define BULLET_TYPES_CONVERTER_H

#include "LinearMath/btMatrix3x3.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btVector3.h"
#include "core/math/matrix3.h"
#include "core/math/transform.h"
#include "core/math/vector3.h"
#include "core/typedefs.h"

// Bullet to Godot
extern void B_TO_G(btVector3 const &inVal, Vector3 &outVal);
extern void INVERT_B_TO_G(btVector3 const &inVal, Vector3 &outVal);
extern void B_TO_G(btMatrix3x3 const &inVal, Basis &outVal);
extern void INVERT_B_TO_G(btMatrix3x3 const &inVal, Basis &outVal);
extern void B_TO_G(btTransform const &inVal, Transform &outVal);

// Godot TO Bullet
extern void G_TO_B(Vector3 const &inVal, btVector3 &outVal);
extern void INVERT_G_TO_B(Vector3 const &inVal, btVector3 &outVal);
extern void G_TO_B(Basis const &inVal, btMatrix3x3 &outVal);
extern void INVERT_G_TO_B(Basis const &inVal, btMatrix3x3 &outVal);
extern void G_TO_B(Transform const &inVal, btTransform &outVal);

#endif
