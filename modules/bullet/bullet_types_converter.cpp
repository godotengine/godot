/*************************************************************************/
/*  bullet_types_converter.cpp                                           */
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

#pragma once

#include "bullet_types_converter.h"

// ++ BULLET to GODOT ++++++++++
void B_TO_G(btVector3 const &inVal, Vector3 &outVal) {
	outVal[0] = inVal[0];
	outVal[1] = inVal[1];
	outVal[2] = inVal[2];
}

void INVERT_B_TO_G(btVector3 const &inVal, Vector3 &outVal) {
	outVal[0] = inVal[0] != 0. ? 1. / inVal[0] : 0.;
	outVal[1] = inVal[1] != 0. ? 1. / inVal[1] : 0.;
	outVal[2] = inVal[2] != 0. ? 1. / inVal[2] : 0.;
}

void B_TO_G(btMatrix3x3 const &inVal, Basis &outVal) {
	B_TO_G(inVal[0], outVal[0]);
	B_TO_G(inVal[1], outVal[1]);
	B_TO_G(inVal[2], outVal[2]);
}

void INVERT_B_TO_G(btMatrix3x3 const &inVal, Basis &outVal) {
	INVERT_B_TO_G(inVal[0], outVal[0]);
	INVERT_B_TO_G(inVal[1], outVal[1]);
	INVERT_B_TO_G(inVal[2], outVal[2]);
}

void B_TO_G(btTransform const &inVal, Transform &outVal) {
	B_TO_G(inVal.getBasis(), outVal.basis);
	B_TO_G(inVal.getOrigin(), outVal.origin);
}

// ++ GODOT to BULLET ++++++++++
void G_TO_B(Vector3 const &inVal, btVector3 &outVal) {
	outVal[0] = inVal[0];
	outVal[1] = inVal[1];
	outVal[2] = inVal[2];
}

void INVERT_G_TO_B(Vector3 const &inVal, btVector3 &outVal) {
	outVal[0] = inVal[0] != 0. ? 1. / inVal[0] : 0.;
	outVal[1] = inVal[1] != 0. ? 1. / inVal[1] : 0.;
	outVal[2] = inVal[2] != 0. ? 1. / inVal[2] : 0.;
}

void G_TO_B(Basis const &inVal, btMatrix3x3 &outVal) {
	G_TO_B(inVal[0], outVal[0]);
	G_TO_B(inVal[1], outVal[1]);
	G_TO_B(inVal[2], outVal[2]);
}

void INVERT_G_TO_B(Basis const &inVal, btMatrix3x3 &outVal) {
	INVERT_G_TO_B(inVal[0], outVal[0]);
	INVERT_G_TO_B(inVal[1], outVal[1]);
	INVERT_G_TO_B(inVal[2], outVal[2]);
}

void G_TO_B(Transform const &inVal, btTransform &outVal) {
	G_TO_B(inVal.basis, outVal.getBasis());
	G_TO_B(inVal.origin, outVal.getOrigin());
}
