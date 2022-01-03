/*************************************************************************/
/*  bullet_types_converter.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "bullet_types_converter.h"

/**
	@author AndreaCatania
*/

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

void B_TO_G(btTransform const &inVal, Transform3D &outVal) {
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

void G_TO_B(Transform3D const &inVal, btTransform &outVal) {
	G_TO_B(inVal.basis, outVal.getBasis());
	G_TO_B(inVal.origin, outVal.getOrigin());
}

void UNSCALE_BT_BASIS(btTransform &scaledBasis) {
	btMatrix3x3 &basis(scaledBasis.getBasis());
	btVector3 column0 = basis.getColumn(0);
	btVector3 column1 = basis.getColumn(1);
	btVector3 column2 = basis.getColumn(2);

	// Check for zero scaling.
	if (column0.fuzzyZero()) {
		if (column1.fuzzyZero()) {
			if (column2.fuzzyZero()) {
				// All dimensions are fuzzy zero. Create a default basis.
				column0 = btVector3(1, 0, 0);
				column1 = btVector3(0, 1, 0);
				column2 = btVector3(0, 0, 1);
			} else { // Column 2 scale not fuzzy zero.
				// Create two vectors orthogonal to row 2.
				// Ensure that a default basis is created if row 2 = <0, 0, 1>
				column1 = btVector3(0, column2[2], -column2[1]);
				column0 = column1.cross(column2);
			}
		} else { // Column 1 scale not fuzzy zero.
			if (column2.fuzzyZero()) {
				// Create two vectors orthogonal to column 1.
				// Ensure that a default basis is created if column 1 = <0, 1, 0>
				column0 = btVector3(column1[1], -column1[0], 0);
				column2 = column0.cross(column1);
			} else { // Column 1 and column 2 scales not fuzzy zero.
				// Create column 0 orthogonal to column 1 and column 2.
				column0 = column1.cross(column2);
			}
		}
	} else { // Column 0 scale not fuzzy zero.
		if (column1.fuzzyZero()) {
			if (column2.fuzzyZero()) {
				// Create two vectors orthogonal to column 0.
				// Ensure that a default basis is created if column 0 = <1, 0, 0>
				column2 = btVector3(-column0[2], 0, column0[0]);
				column1 = column2.cross(column0);
			} else { // Column 0 and column 2 scales not fuzzy zero.
				// Create column 1 orthogonal to column 0 and column 2.
				column1 = column2.cross(column0);
			}
		} else { // Column 0 and column 1 scales not fuzzy zero.
			if (column2.fuzzyZero()) {
				// Create column 2 orthogonal to column 0 and column 1.
				column2 = column0.cross(column1);
			}
		}
	}

	// Normalize
	column0.normalize();
	column1.normalize();
	column2.normalize();

	basis.setValue(column0[0], column1[0], column2[0],
			column0[1], column1[1], column2[1],
			column0[2], column1[2], column2[2]);
}
