/**************************************************************************/
/*  qcp.cpp                                                               */
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

#include "qcp.h"

QCP::QCP(double p_evec_prec, double p_eval_prec) {
	evec_prec = p_eval_prec;
	eval_prec = p_evec_prec;
}

void QCP::set(PackedVector3Array &r_target, PackedVector3Array &r_moved) {
	target = r_target;
	moved = r_moved;
	rmsd_calculated = false;
	transformation_calculated = false;
	inner_product_calculated = false;
}

double QCP::get_rmsd() {
	if (!rmsd_calculated) {
		calculate_rmsd(moved, target);
		rmsd_calculated = true;
	}
	return rmsd;
}

Quaternion QCP::get_rotation() {
	Quaternion result;
	if (!transformation_calculated) {
		if (!inner_product_calculated) {
			inner_product(target, moved);
		}
		result = calculate_rotation();
		transformation_calculated = true;
	}
	return result;
}

void QCP::calculate_rmsd(double r_length) {
	rmsd = Math::sqrt(Math::abs(2.0f * (e0 - mxEigenV) / r_length));
}

Quaternion QCP::calculate_rotation() {
	// QCP doesn't handle single targets, so if we only have one point and one
	// target, we just rotate by the angular distance between them
	if (moved.size() == 1) {
		Vector3 u = moved[0];
		Vector3 v = target[0];
		double norm_product = u.length() * v.length();
		if (norm_product == 0.0) {
			return Quaternion();
		}
		double dot = u.dot(v);
		if (dot < ((2.0e-15 - 1.0) * norm_product)) {
			// The special case: u = -v,
			// We select a PI angle rotation around
			// an arbitrary vector orthogonal to u.
			Vector3 w = u.normalized();
			return Quaternion(-1 * -w.x, -1 * -w.y, -1 * -w.z, 0.0f).normalized();
		}
		// The general case: (u, v) defines a plane, we select
		// the shortest possible rotation: axis orthogonal to this plane.
		double q0 = Math::sqrt(0.5 * (1.0 + dot / norm_product));
		double coeff = 1.0 / (2.0 * q0 * norm_product);
		Vector3 q = v.cross(u);
		double q1 = coeff * q.x;
		double q2 = coeff * q.y;
		double q3 = coeff * q.z;
		return Quaternion(-1 * q1, -1 * q2, -1 * q3, q0).normalized();
	} else {
		double a11 = SxxpSyy + Szz - mxEigenV;
		double a12 = SyzmSzy;
		double a13 = -SxzmSzx;
		double a14 = SxymSyx;
		double a21 = SyzmSzy;
		double a22 = SxxmSyy - Szz - mxEigenV;
		double a23 = SxypSyx;
		double a24 = SxzpSzx;
		double a31 = a13;
		double a32 = a23;
		double a33 = Syy - Sxx - Szz - mxEigenV;
		double a34 = SyzpSzy;
		double a41 = a14;
		double a42 = a24;
		double a43 = a34;
		double a44 = Szz - SxxpSyy - mxEigenV;
		double a3344_4334 = a33 * a44 - a43 * a34;
		double a3244_4234 = a32 * a44 - a42 * a34;
		double a3243_4233 = a32 * a43 - a42 * a33;
		double a3143_4133 = a31 * a43 - a41 * a33;
		double a3144_4134 = a31 * a44 - a41 * a34;
		double a3142_4132 = a31 * a42 - a41 * a32;
		double q1 = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
		double q2 = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
		double q3 = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
		double q4 = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

		double qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

		/*
		 * The following code tries to calculate another column in the adjoint matrix
		 * when the norm of the current column is too small. Usually this commented
		 * block will never be activated. To be absolutely safe this should be
		 * uncommented, but it is most likely unnecessary.
		 */
		if (qsqr < evec_prec) {
			q1 = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233;
			q2 = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133;
			q3 = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132;
			q4 = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132;
			qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

			if (qsqr < evec_prec) {
				double a1324_1423 = a13 * a24 - a14 * a23, a1224_1422 = a12 * a24 - a14 * a22;
				double a1223_1322 = a12 * a23 - a13 * a22, a1124_1421 = a11 * a24 - a14 * a21;
				double a1123_1321 = a11 * a23 - a13 * a21, a1122_1221 = a11 * a22 - a12 * a21;

				q1 = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
				q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
				q3 = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
				q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
				qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

				if (qsqr < evec_prec) {
					q1 = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
					q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
					q3 = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
					q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
					qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

					if (qsqr < evec_prec) {
						/*
						 * if qsqr is still too small, return the identity rotation
						 */
						return Quaternion();
					}
				}
			}
		}
		q2 *= -1;
		q3 *= -1;
		q4 *= -1;
		real_t min = q1;
		min = q2 < min ? q2 : min;
		min = q3 < min ? q3 : min;
		min = q4 < min ? q4 : min;
		q1 /= min;
		q2 /= min;
		q3 /= min;
		q4 /= min;
		return Quaternion(q2, q3, q4, q1).normalized();
	}
}

double QCP::get_rmsd(PackedVector3Array &r_fixed, PackedVector3Array &r_moved) {
	set(r_fixed, r_moved);
	return get_rmsd();
}

void QCP::translate(Vector3 r_translate, PackedVector3Array &r_x) {
	for (Vector3 &p : r_x) {
		p += r_translate;
	}
}

Vector3 QCP::get_translation() {
	return target_center - moved_center;
}

Vector3 QCP::move_to_weighted_center(PackedVector3Array &r_to_center, Vector<real_t> &r_weight) {
	Vector3 center;
	bool weight_is_empty = r_weight.is_empty();
	if (!weight_is_empty) {
		for (int i = 0; i < r_to_center.size(); i++) {
			w_sum += weight[i];
		}
	}
	if (!weight_is_empty && w_sum > 0) {
		for (int i = 0; i < r_to_center.size(); i++) {
			center += r_to_center[i] * r_weight[i];
		}
		center /= w_sum;
	} else {
		w_sum = 0;
		for (int i = 0; i < r_to_center.size(); i++) {
			center += r_to_center[i];
			w_sum++;
		}
		center /= w_sum;
	}
	return center;
}

void QCP::inner_product(PackedVector3Array &coords1, PackedVector3Array &coords2) {
	double x1, x2, y1, y2, z1, z2;
	double g1 = 0, g2 = 0;

	Sxx = 0;
	Sxy = 0;
	Sxz = 0;
	Syx = 0;
	Syy = 0;
	Syz = 0;
	Szx = 0;
	Szy = 0;
	Szz = 0;

	if (!weight.is_empty()) {
		for (int i = 0; i < coords1.size(); i++) {
			x1 = weight[i] * coords1[i].x;
			y1 = weight[i] * coords1[i].y;
			z1 = weight[i] * coords1[i].z;

			g1 += x1 * coords1[i].x + y1 * coords1[i].y + z1 * coords1[i].z;

			x2 = coords2[i].x;
			y2 = coords2[i].y;
			z2 = coords2[i].z;

			g2 += weight[i] * (x2 * x2 + y2 * y2 + z2 * z2);

			Sxx += (x1 * x2);
			Sxy += (x1 * y2);
			Sxz += (x1 * z2);

			Syx += (y1 * x2);
			Syy += (y1 * y2);
			Syz += (y1 * z2);

			Szx += (z1 * x2);
			Szy += (z1 * y2);
			Szz += (z1 * z2);
		}
	} else {
		for (int i = 0; i < coords1.size(); i++) {
			g1 += coords1[i].x * coords1[i].x + coords1[i].y * coords1[i].y + coords1[i].z * coords1[i].z;
			g2 += coords2[i].x * coords2[i].x + coords2[i].y * coords2[i].y + coords2[i].z * coords2[i].z;

			Sxx += coords1[i].x * coords2[i].x;
			Sxy += coords1[i].x * coords2[i].y;
			Sxz += coords1[i].x * coords2[i].z;

			Syx += coords1[i].y * coords2[i].x;
			Syy += coords1[i].y * coords2[i].y;
			Syz += coords1[i].y * coords2[i].z;

			Szx += coords1[i].z * coords2[i].x;
			Szy += coords1[i].z * coords2[i].y;
			Szz += coords1[i].z * coords2[i].z;
		}
	}

	e0 = (g1 + g2) * 0.5;

	SxzpSzx = Sxz + Szx;
	SyzpSzy = Syz + Szy;
	SxypSyx = Sxy + Syx;
	SyzmSzy = Syz - Szy;
	SxzmSzx = Sxz - Szx;
	SxymSyx = Sxy - Syx;
	SxxpSyy = Sxx + Syy;
	SxxmSyy = Sxx - Syy;
	mxEigenV = e0;

	inner_product_calculated = true;
}

void QCP::calculate_rmsd(PackedVector3Array &x, PackedVector3Array &y) {
	// QCP doesn't handle alignment of single values, so if we only have one point
	// we just compute regular distance.
	if (x.size() == 1) {
		rmsd = x[0].distance_to(y[0]);
		rmsd_calculated = true;
	} else {
		if (!inner_product_calculated) {
			inner_product(y, x);
		}
		calculate_rmsd(w_sum);
	}
}

Quaternion QCP::weighted_superpose(PackedVector3Array &p_moved, PackedVector3Array &p_target, Vector<real_t> &p_weight, bool translate) {
	set(p_moved, p_target, p_weight, translate);
	return get_rotation();
}

void QCP::set(PackedVector3Array &p_moved, PackedVector3Array &p_target, Vector<real_t> &p_weight, bool p_translate) {
	rmsd_calculated = false;
	transformation_calculated = false;
	inner_product_calculated = false;

	moved = p_moved;
	target = p_target;
	weight = p_weight;

	if (p_translate) {
		moved_center = move_to_weighted_center(moved, weight);
		w_sum = 0; // set wsum to 0 so we don't double up.
		target_center = move_to_weighted_center(target, weight);
		translate(moved_center * -1, moved);
		translate(target_center * -1, target);
	} else {
		if (!p_weight.is_empty()) {
			for (int i = 0; i < p_weight.size(); i++) {
				w_sum += p_weight[i];
			}
		} else {
			w_sum = p_moved.size();
		}
	}
}
