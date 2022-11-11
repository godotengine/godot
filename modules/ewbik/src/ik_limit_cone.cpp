/*************************************************************************/
/*  ik_limit_cone.cpp                                                    */
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

#include "ik_limit_cone.h"

#include "core/io/resource.h"
#include "core/object/ref_counted.h"

#include "ik_kusudama.h"

IKLimitCone::IKLimitCone() {
}

void IKLimitCone::update_tangent_handles(Ref<IKLimitCone> p_next) {
	if (p_next.is_valid()) {
		double radA = this->_get_radius();
		double radB = p_next->_get_radius();

		Vector3 A = this->get_control_point();
		Vector3 B = p_next->get_control_point();

		Vector3 arc_normal = A.cross(B);

		/**
		 * There are an infinite number of circles co-tangent with A and B, every other
		 * one of which has a unique radius.
		 *
		 * However, we want the radius of our tangent circles to obey the following properties:
		 *   1) When the radius of A + B == 0, our tangent circle's radius should = 90.
		 *   	In other words, the tangent circle should span a hemisphere.
		 *   2) When the radius of A + B == 180, our tangent circle's radius should = 0.
		 *   	In other words, when A + B combined are capable of spanning the entire sphere,
		 *   	our tangentCircle should be nothing.
		 *
		 * Another way to think of this is -- whatever the maximum distance can be between the
		 * borders of A and B (presuming their centers are free to move about the circle
		 * but their radii remain constant), we want our tangentCircle's diameter to be precisely that distance,
		 * and so, our tangent circles radius should be precisely half of that distance.
		 */
		double tRadius = (Math_PI - (radA + radB)) / 2;

		/**
		 * Once we have the desired radius for our tangent circle, we may find the solution for its
		 * centers (usually, there are two).
		 */
		double boundaryPlusTangentRadiusA = radA + tRadius;
		double boundaryPlusTangentRadiusB = radB + tRadius;

		// the axis of this cone, scaled to minimize its distance to the tangent contact points.
		Vector3 scaledAxisA = A * cos(boundaryPlusTangentRadiusA);
		// a point on the plane running through the tangent contact points
		Quaternion temp_var = quaternion_set_axis_angle(arc_normal, boundaryPlusTangentRadiusA);
		Vector3 planeDir1A = temp_var.xform(A);
		// another point on the same plane
		Quaternion tempVar2 = quaternion_set_axis_angle(A, Math_PI / 2);
		Vector3 planeDir2A = tempVar2.xform(planeDir1A);

		Vector3 scaledAxisB = B * cos(boundaryPlusTangentRadiusB);
		// a point on the plane running through the tangent contact points
		Quaternion tempVar3 = quaternion_set_axis_angle(arc_normal, boundaryPlusTangentRadiusB);
		Vector3 planeDir1B = tempVar3.xform(B);
		// another point on the same plane
		Quaternion tempVar4 = quaternion_set_axis_angle(B, Math_PI / 2);
		Vector3 planeDir2B = tempVar4.xform(planeDir1B);

		// ray from scaled center of next cone to half way point between the circumference of this cone and the next cone.
		Ref<IKRay3D> r1B = Ref<IKRay3D>(memnew(IKRay3D(planeDir1B, scaledAxisB)));
		Ref<IKRay3D> r2B = Ref<IKRay3D>(memnew(IKRay3D(planeDir1B, planeDir2B)));

		r1B->elongate(99);
		r2B->elongate(99);

		Vector3 intersection1 = r1B->intersects_plane(scaledAxisA, planeDir1A, planeDir2A);
		Vector3 intersection2 = r2B->intersects_plane(scaledAxisA, planeDir1A, planeDir2A);

		Ref<IKRay3D> intersectionRay = Ref<IKRay3D>(memnew(IKRay3D(intersection1, intersection2)));
		intersectionRay->elongate(99);

		Vector3 sphereIntersect1;
		Vector3 sphereIntersect2;
		Vector3 sphereCenter;
		intersectionRay->intersects_sphere(sphereCenter, 1.0f, sphereIntersect1, sphereIntersect2);

		this->set_tangent_circle_center_next_1(sphereIntersect1);
		this->set_tangent_circle_center_next_2(sphereIntersect2);
		this->set_tangent_circle_radius_next(tRadius);
	}
	if (this->tangent_circle_center_next_1 == Vector3(NAN, NAN, NAN)) {
		this->tangent_circle_center_next_1 = get_orthogonal(control_point).normalized();
	}
	if (tangent_circle_center_next_2 == Vector3(NAN, NAN, NAN)) {
		tangent_circle_center_next_2 = (tangent_circle_center_next_1 * -1).normalized();
	}
	if (p_next.is_valid()) {
		compute_triangles(p_next);
	}
}

void IKLimitCone::set_tangent_circle_radius_next(double rad) {
	this->tangent_circle_radius_next = rad;
	this->tangent_circle_radius_next_cos = cos(tangent_circle_radius_next);
}

Vector3 IKLimitCone::get_tangent_circle_center_next_1() {
	return tangent_circle_center_next_1;
}

double IKLimitCone::get_tangent_circle_radius_next() {
	return tangent_circle_radius_next;
}

double IKLimitCone::get_tangent_circle_radius_next_cos() {
	return tangent_circle_radius_next_cos;
}

Vector3 IKLimitCone::get_tangent_circle_center_next_2() {
	return tangent_circle_center_next_2;
}

double IKLimitCone::_get_radius() {
	return radius;
}

double IKLimitCone::_get_radius_cosine() {
	return radius_cosine;
}

void IKLimitCone::compute_triangles(Ref<IKLimitCone> p_next) {
	first_triangle_next.write[1] = this->tangent_circle_center_next_1.normalized();
	first_triangle_next.write[0] = this->get_control_point().normalized();
	first_triangle_next.write[2] = p_next->get_control_point().normalized();

	second_triangle_next.write[1] = this->tangent_circle_center_next_2.normalized();
	second_triangle_next.write[0] = this->get_control_point().normalized();
	second_triangle_next.write[2] = p_next->get_control_point().normalized();
}

Vector3 IKLimitCone::get_control_point() const {
	return control_point;
}

inline void IKLimitCone::set_control_point(Vector3 p_control_point) {
	this->control_point = p_control_point;
	this->control_point.normalize();
}

double IKLimitCone::get_radius() const {
	return this->radius;
}

double IKLimitCone::get_radius_cosine() const {
	return this->radius_cosine;
}

void IKLimitCone::set_radius(double p_radius) {
	this->radius = p_radius;
	this->radius_cosine = cos(p_radius);
}

bool IKLimitCone::determine_if_in_bounds(Ref<IKLimitCone> next, Vector3 input) const {
	/**
	 * Procedure : Check if input is contained in this cone, or the next cone
	 * 	if it is, then we're finished and in bounds. otherwise,
	 * check if the point  is contained within the tangent radii,
	 * 	if it is, then we're out of bounds and finished, otherwise
	 * in the tangent triangles while still remaining outside of the tangent radii
	 * if it is, then we're finished and in bounds. otherwise, we're out of bounds.
	 */

	if (control_point.dot(input) >= radius_cosine) {
		return true;
	} else if (next != nullptr && next->control_point.dot(input) >= next->radius_cosine) {
		return true;
	} else {
		if (next == nullptr) {
			return false;
		}
		bool inTan1Rad = tangent_circle_center_next_1.dot(input) > tangent_circle_radius_next_cos;
		if (inTan1Rad) {
			return false;
		}
		bool inTan2Rad = tangent_circle_center_next_2.dot(input) > tangent_circle_radius_next_cos;
		if (inTan2Rad) {
			return false;
		}

		/*if we reach this point in the code, we are either on the path between two limit_cones, or on the path extending out from between them
		 * but outside of their radii.
		 * 	To determine which , we take the cross product of each control point with each tangent center.
		 * 		The direction of each of the resultant vectors will represent the normal of a plane.
		 * 		Each of these four planes define part of a boundary which determines if our point is in bounds.
		 * 		If the dot product of our point with the normal of any of these planes is negative, we must be out
		 * 		of bounds.
		 *
		 *	Older version of this code relied on a triangle intersection algorithm here, which I think is slightly less efficient on average
		 *	as it didn't allow for early termination. .
		 */

		Vector3 c1xc2 = control_point.cross(next->control_point);
		double c1c2dir = input.dot(c1xc2);

		if (c1c2dir < 0.0) {
			Vector3 c1xt1 = control_point.cross(tangent_circle_center_next_1);
			Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->control_point);
			return input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0;
		} else {
			Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
			Vector3 c2xt2 = next->control_point.cross(tangent_circle_center_next_2);
			return input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0;
		}
	}
}

Vector3 IKLimitCone::get_closest_path_point(Ref<IKLimitCone> next, Vector3 input) const {
	Vector3 result = get_on_path_sequence(next, input);
	bool is_number = !(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z));
	if (!is_number) {
		result = closest_cone(next, input);
	}
	return result;
}

Vector3 IKLimitCone::get_closest_collision(Ref<IKLimitCone> next, Vector3 input) const {
	Vector3 result = get_on_great_tangent_triangle(next, input);

	bool is_number = !(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z));
	if (!is_number) {
		Vector<double> in_bounds = { 0.0 };
		result = closest_point_on_closest_cone(next, input, in_bounds);
	}
	return result;
}

Vector3 IKLimitCone::get_orthogonal(Vector3 p_in) {
	Vector3 result;
	float threshold = p_in.length() * 0.6f;
	if (threshold > 0.f) {
		if (Math::abs(p_in.x) <= threshold) {
			float inverse = 1.f / Math::sqrt(p_in.y * p_in.y + p_in.z * p_in.z);
			return result = Vector3(0.f, inverse * p_in.z, -inverse * p_in.y);
		} else if (Math::abs(p_in.y) <= threshold) {
			float inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.z * p_in.z);
			return result = Vector3(-inverse * p_in.z, 0.f, inverse * p_in.x);
		}
		float inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.y * p_in.y);
		return result = Vector3(inverse * p_in.y, -inverse * p_in.x, 0.f);
	}

	return result;
}

IKLimitCone::IKLimitCone(Vector3 &direction, double rad, Ref<IKKusudama> attached_to) {
	parent_kusudama = attached_to;
	tangent_circle_center_next_1 = Vector3(0.0f, -1.0f, 0.0f);
	tangent_circle_center_next_2 = Vector3(0.0f, 1.0f, 0.0f);
	set_radius(MAX(DBL_TRUE_MIN, rad));
	this->control_point = direction.normalized();
}

Vector3 IKLimitCone::get_on_great_tangent_triangle(Ref<IKLimitCone> next, Vector3 input) const {
	Vector3 c1xc2 = control_point.cross(next->control_point);
	double c1c2dir = input.dot(c1xc2);
	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = control_point.cross(tangent_circle_center_next_1);
		Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->control_point);
		if (input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0) {
			double toNextCos = input.dot(tangent_circle_center_next_1);
			if (toNextCos > tangent_circle_radius_next_cos) {
				Vector3 planeNormal = tangent_circle_center_next_1.cross(input);
				planeNormal.normalize();
				Quaternion rotateAboutBy = Quaternion(planeNormal, tangent_circle_radius_next);
				return rotateAboutBy.xform(tangent_circle_center_next_1);
			} else {
				return input;
			}
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	} else {
		Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
		Vector3 c2xt2 = next->control_point.cross(tangent_circle_center_next_2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			if (input.dot(tangent_circle_center_next_2) > tangent_circle_radius_next_cos) {
				Vector3 planeNormal = tangent_circle_center_next_2.cross(input);
				planeNormal.normalize();
				Quaternion rotateAboutBy = Quaternion(planeNormal, tangent_circle_radius_next);
				return rotateAboutBy.xform(tangent_circle_center_next_2);
			} else {
				return input;
			}
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	}
}

Vector3 IKLimitCone::closest_cone(Ref<IKLimitCone> next, Vector3 input) const {
	if (input.dot(control_point) > input.dot(next->control_point)) {
		return this->control_point;
	} else {
		return next->control_point;
	}
}

Vector3 IKLimitCone::closest_point_on_closest_cone(Ref<IKLimitCone> next, Vector3 input, Vector<double> &in_bounds) const {
	Vector3 closestToFirst = this->closest_to_cone(input, in_bounds);
	if (in_bounds[0] > 0.0) {
		return closestToFirst;
	}
	Vector3 closestToSecond = next->closest_to_cone(input, in_bounds);
	if (in_bounds[0] > 0.0) {
		return closestToSecond;
	}
	double cosToFirst = input.dot(closestToFirst);
	double cosToSecond = input.dot(closestToSecond);

	if (cosToFirst > cosToSecond) {
		return closestToFirst;
	} else {
		return closestToSecond;
	}
}

Vector3 IKLimitCone::closest_to_cone(Vector3 input, Vector<double> &in_bounds) const {
	if (input.dot(this->get_control_point()) > this->get_radius_cosine()) {
		in_bounds.write[0] = 1.0;
		return Vector3(NAN, NAN, NAN);
	}
	Vector3 axis = this->get_control_point().cross(input).normalized();
	Quaternion rotTo = Quaternion(axis.normalized(), this->get_radius());
	Vector3 axis_control_point = this->get_control_point();
	Vector3 result = rotTo.xform(axis_control_point);
	in_bounds.write[0] = -1;
	return result;
}

void IKLimitCone::set_tangent_circle_center_next_1(Vector3 point) {
	this->tangent_circle_center_next_1 = point.normalized();
}

void IKLimitCone::set_tangent_circle_center_next_2(Vector3 point) {
	tangent_circle_center_next_2 = point.normalized();
}

Vector3 IKLimitCone::get_on_path_sequence(Ref<IKLimitCone> next, Vector3 input) const {
	Vector3 c1xc2 = get_control_point().cross(next->control_point);
	double c1c2dir = input.dot(c1xc2);
	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = get_control_point().cross(tangent_circle_center_next_1);
		Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->get_control_point());
		if (input.dot(c1xt1) > 0.0f && input.dot(t1xc2) > 0.0f) {
			Ref<IKRay3D> tan1ToInput = Ref<IKRay3D>(memnew(IKRay3D(tangent_circle_center_next_1, input)));
			Vector3 result = tan1ToInput->intersects_plane(Vector3(0.0f, 0.0f, 0.0f), get_control_point(), next->get_control_point());
			return result.normalized();
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	} else {
		Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
		Vector3 c2xt2 = next->get_control_point().cross(tangent_circle_center_next_2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			Ref<IKRay3D> tan2ToInput = Ref<IKRay3D>(memnew(IKRay3D(tangent_circle_center_next_2, input)));
			Vector3 result = tan2ToInput->intersects_plane(Vector3(0.0f, 0.0f, 0.0f), get_control_point(), next->get_control_point());
			return result.normalized();
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	}
	return Vector3(NAN, NAN, NAN);
}
Quaternion IKLimitCone::quaternion_set_axis_angle(Vector3 axis, real_t angle) {
	real_t norm = axis.length();
	if (norm == 0) {
		ERR_PRINT_ONCE("Axis doesn't have a direction.");
		return Quaternion();
	}

	real_t halfAngle = -0.5 * angle;
	real_t coeff = sin(halfAngle) / norm;

	real_t x = coeff * axis.x;
	real_t y = coeff * axis.y;
	real_t z = coeff * axis.z;
	real_t w = cos(halfAngle);
	return Quaternion(x * -1, y * -1, z * -1, w);
}
