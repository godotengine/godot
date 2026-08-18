/**************************************************************************/
/*  spring_bone_collision_capsule_3d.cpp                                  */
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

#include "spring_bone_collision_capsule_3d.h"

#include "core/object/class_db.h"
#include "scene/3d/spring_bone_collision_sphere_3d.h"

void SpringBoneCollisionCapsule3D::set_radius(float p_radius) {
	radius = p_radius;
	if (radius > height * 0.5) {
		height = radius * 2.0;
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float SpringBoneCollisionCapsule3D::get_radius() const {
	return radius;
}

void SpringBoneCollisionCapsule3D::set_height(float p_height) {
	height = p_height;
	if (radius > height * 0.5) {
		radius = height * 0.5;
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float SpringBoneCollisionCapsule3D::get_height() const {
	return height;
}

void SpringBoneCollisionCapsule3D::set_mid_height(real_t p_mid_height) {
	ERR_FAIL_COND_MSG(p_mid_height < 0.0f, "SpringBoneCollisionCapsule3D mid-height cannot be negative.");
	height = p_mid_height + radius * 2.0f;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

real_t SpringBoneCollisionCapsule3D::get_mid_height() const {
	return height - radius * 2.0f;
}

SpringBoneCollision3D::CollideMode SpringBoneCollisionCapsule3D::get_collide_mode() const {
	return collide_mode;
}

void SpringBoneCollisionCapsule3D::set_collide_mode(CollideMode p_collide_mode) {
	collide_mode = p_collide_mode;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Pair<Vector3, Vector3> SpringBoneCollisionCapsule3D::get_head_and_tail(const Transform3D &p_center) const {
	Transform3D tr = get_transform_from_skeleton(p_center);
	return Pair<Vector3, Vector3>(tr.origin + tr.basis.xform(Vector3::UP * (height * 0.5 - radius)), tr.origin + tr.basis.xform(Vector3::DOWN * (height * 0.5 - radius)));
}

void SpringBoneCollisionCapsule3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &SpringBoneCollisionCapsule3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &SpringBoneCollisionCapsule3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &SpringBoneCollisionCapsule3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &SpringBoneCollisionCapsule3D::get_height);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &SpringBoneCollisionCapsule3D::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &SpringBoneCollisionCapsule3D::get_mid_height);
	ClassDB::bind_method(D_METHOD("set_collide_mode", "collide_mode"), &SpringBoneCollisionCapsule3D::set_collide_mode);
	ClassDB::bind_method(D_METHOD("get_collide_mode"), &SpringBoneCollisionCapsule3D::get_collide_mode);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m", PROPERTY_USAGE_NONE), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collide_mode", PROPERTY_HINT_ENUM, "Joint,Inside,Chain"), "set_collide_mode", "get_collide_mode");
}

// The SpringBoneCollisionCapsule3D::_collide() function is to find the deepest point of
// collision between two capsule shaped components, a conical springbone and a cylindrical capsule collider.
// The first step is to select the sphere within the capsule collider that has the deepest ingress into the conical springbone.
// Then we call _collide_sphere_taper() to collide this shape, which is a subset of the collider, into the conical springbone.

// In other words, we need to find a mu between 0 and 1 that minimizes verify_distance_within_taper(lerp(head,tail,mu)).

// Very local calculation verification feature
#define VERIFY_SPRINGBONECAPSULE_CALCULATIONS 1
#if VERIFY_SPRINGBONECAPSULE_CALCULATIONS
#define SB_DEV_ASSERT(m_cond) \
	if ((!(m_cond))) { \
		printf("SB_DEV_ASSERT %s %s %d   %s\n", FUNCTION_STR, __FILE__, __LINE__, _STR(m_cond)); \
	} else \
		((void)0)
#else
#define SB_DEV_ASSERT(m_cond)
#endif

// function to verify calculations
real_t verify_distance_within_taper(const Vector3 &p_origin, float p_bone_radius, float p_bone_length, const Vector3 &p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) {
	// (p_origin) defines the external point we are measuring the distance to (on the axis of the collider)
	// The bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)
	real_t taper_fore = (p_bone_origin_radius - p_bone_radius) / p_bone_length;
	Vector3 diff = p_current - p_origin;
	Vector3 bone_axis = p_current - p_current_origin; // should be length p_bone_radius due to calls to limit_length()
	DEV_ASSERT(Math::is_equal_approx(bone_axis.length(), p_bone_length));
	real_t taper_side = Math::sqrt(1.0 - taper_fore * taper_fore);
	real_t lam = 1.0 - bone_axis.dot(diff) / (p_bone_length * p_bone_length);
	Vector3 vecside = p_origin - (p_current_origin + bone_axis * lam);
	real_t radial_distance = vecside.length();
	real_t bone_axis_length = p_bone_length;

	real_t lamd = radial_distance * taper_fore / taper_side / bone_axis_length;
	real_t lamcone = lam - lamd;
	real_t clamcone = MIN(MAX(lamcone, 0.0), 1.0);
	Vector3 closest_cone_axis_point = p_current_origin + bone_axis * clamcone;
	real_t cone_sphere_rad = p_bone_origin_radius + clamcone * (p_bone_radius - p_bone_origin_radius);
	return (p_origin - closest_cone_axis_point).length() - cone_sphere_rad;
}

static Vector3 _closest_capsule_sphere(const Vector3 &head, const Vector3 &tail, const Vector3 &bone_sphere_center) {
	Vector3 p = tail - head;
	Vector3 q = bone_sphere_center - head;
	real_t dot = p.dot(q);
	if (dot <= 0) {
		return head;
	}
	real_t pls = p.length_squared();
	if ((pls <= dot) || Math::is_zero_approx(pls)) {
		return tail;
	}
	return head + p * (dot / pls);
}

static real_t _closest_capsule_sphere_to_taper(const Vector3 &head, const Vector3 &tail, float radius, float p_bone_radius, float p_bone_length, const Vector3 &p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) {
	// The collision capsule is (head, radius) to (tail, radius) parametrized by mu
	// The bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius) parametrized by lam

	Vector3 bone_axis = p_current - p_current_origin;
	DEV_ASSERT(Math::is_equal_approx(bone_axis.length(), p_bone_length)); // enforced by limit_length()
	Vector3 p = tail - head;

	// The bone_axis and p (the capsule axis) are skew lines,
	// so the cross-product vector is the shortest distance between them.
	Vector3 perp = bone_axis.cross(p);
	real_t perp_sq = perp.dot(perp);
	real_t perp_len = sqrt(perp_sq);
	if (Math::is_zero_approx(perp_len)) { // This case also removes zero length bones and capsules.
		return 0.5; // Axes are parallel, so should actually pick point in overlap, but this is a very rare case.
	}
	real_t perp_bone = perp.dot(p_current_origin);
	real_t perp_capsule = perp.dot(head);
	real_t perp_dist = (perp_capsule - perp_bone) / perp_len;
	if (Math::abs(perp_dist) > radius + MAX(p_bone_origin_radius, p_bone_radius) + CMP_EPSILON) {
		return -1.0; // Geometry too distant for to interactions.
	}

	// Calculate the points of closest approach between these two skew lines
	// by solving: p_current_origin + bone_axis * lam + perp = head + p * mu

	Vector3 hh = p_current_origin - head;
	// dot bone_axis: hh.dot(bone_axis) + bone_axis.dot(bone_axis) * lam = p.dot(bone_axis) * mu
	// dot p: hh.dot(p) + bone_axis.dot(p) * lam = p.dot(p) * mu
	real_t badp = bone_axis.dot(p);
	real_t badba = bone_axis.dot(bone_axis);
	real_t pdp = p.dot(p);
	real_t hhdba = hh.dot(bone_axis);
	real_t hhdp = hh.dot(p);
	// hhdba = -badba * lam + badp * mu
	// hhdp = -badp * lam + pdp * mu
	// ( -badba  badp )   ( lam )   ( hhdba )
	// ( -badp   pdp  ) * (  mu ) = (  hhdp )

	// If T is the angle between ba and p, then the determinant of this matrix is:
	// -badba * pdp + badp * badp = -ba^2*p^2 + ba^2*p^2*cosT^2 = -ba^2*p^2*sinT^2 = -(ba x p)^2
	real_t det = -perp_sq;
	// ( pdp    -badp )   ( hhdba )   ( lam )
	// ( badp  -badba ) * (  hhdp ) = (  mu ) * det
	real_t lam = (pdp * hhdba - badp * hhdp) / det;
	real_t mu = (badp * hhdba - badba * hhdp) / det;

#ifdef VERIFY_SPRINGBONECAPSULE_CALCULATIONS
	SB_DEV_ASSERT(Math::is_equal_approx(det, -badba * pdp + badp * badp));
	real_t Dhhdba = -badba * lam + badp * mu;
	real_t Dhhdp = -badp * lam + pdp * mu;
	Vector3 Dperpvec = perp * (perp_dist / perp_len);
	SB_DEV_ASSERT(Math::is_equal_approx(Dhhdba, hhdba));
	SB_DEV_ASSERT(Math::is_equal_approx(Dhhdp, hhdp));
	Vector3 Dlammuvec = (p_current_origin + bone_axis * lam + Dperpvec) - (head + p * mu);
	SB_DEV_ASSERT(Math::is_zero_approx(Dlammuvec.length()));
#endif

	// Handle cylindrical springbone case.
	// The bone capsule (cylinder) is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius)
	if (p_bone_radius == p_bone_origin_radius) {

		// clamp the collision sphere center point to the bone and recalculate it for the capsule
		if ((lam < 0.0) || (lam > 1.0)) {
			Vector3 bone_sphere_end = (lam < 0.0 ? p_current_origin : p_current);
			mu = (bone_sphere_end - head).dot(p) / pdp;
			lam = (lam < 0.0 ? 0.0 : 1.0);
		}

		mu = (mu < 0.0 ? 0.0 : (mu > 1.0 ? 1.0 : mu)); // clamp(0,1)

#ifdef VERIFY_SPRINGBONECAPSULE_CALCULATIONS
		Vector3 Dcapsule_sphere_center = head * (1.0 - mu) + tail * mu;
		Vector3 Dbone_sphere_center = p_current_origin * (1.0 - lam) + p_current * lam;
		real_t Dvpdist = (Dcapsule_sphere_center - Dbone_sphere_center).length();
		real_t Dvdist = verify_distance_within_taper(Dcapsule_sphere_center, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
		if (fabs((Dvpdist - p_bone_radius) - Dvdist) > 0.01) {
			printf("  %d disag %.3f %.3f mu %.3f lam %.3f\n", SpringBoneCollision3D::Dsegmentindexbeingcalculated, Dvpdist, Dvdist, mu, lam);
		}
#endif

		return mu;
	}

	// Now consider the plane C perpendicular to perp containing the head-tail vector of the capsule
	// Set its origin to be at (head + p * mu) with y-vector along the normalized_bone_axis
	// and x-vector perpendicular to bone_axis and perp.
	// We can apply the radius of the collusion capsule to the radii of the cone and replace the collision capsule with a line.
	// The intersection of this plane with the bone cone will be a hyperbola (conic section).

	// If the capsule intrudes by a distance of intrude_radius into the bone cone, then the cone
	// from (p_current_origin, p_bone_origin_radiusP + radius - intrude_radius) to
	//   to (p_current,        p_bone_radiusP        + radius - intrude_radius)
	// will define a hyperbola that is tangential to the capsule axis.

	// Therefore we need to calculate intrude_radius which makes the hyperbola tangential.

	// But first we need to calculate p_bone_origin_radiusP and p_bone_radiusP which are the
	// radii of the cone in the plane across its axis -- whereas the given definition is
	// in terms of a cone tangential to the spheres around the endpoints of the axis.

	// If cone_side_perp is the unit vector in the cone axis (x component) and perpendicular to the cone axis (y component)
	real_t cone_side_perp_x = (p_bone_origin_radius - p_bone_radius) / p_bone_length;
	real_t cone_side_perp_y = sqrt(1 - cone_side_perp_x * cone_side_perp_x);
	real_t p_bone_origin_radiusP = p_bone_origin_radius / cone_side_perp_y;
	real_t p_bone_radiusP = p_bone_radius / cone_side_perp_y;

	//Vector3 C_plane_origin = head + p * mu;
	Vector3 C_plane_z = perp * (1 / perp_len);
	Vector3 C_plane_y = bone_axis * (1 / p_bone_length);
	Vector3 C_plane_x = C_plane_z.cross(C_plane_y);
	SB_DEV_ASSERT(Math::is_equal_approx(C_plane_z.length(), 1));
	SB_DEV_ASSERT(Math::is_equal_approx(C_plane_y.length(), 1));
	SB_DEV_ASSERT(Math::is_equal_approx(C_plane_x.length(), 1));

	real_t p_length = sqrt(pdp);
	Vector3 C_capsule_vec = p * (1 / p_length);
	SB_DEV_ASSERT(Math::is_equal_approx(C_capsule_vec.length(), 1));
	Vector3 C_capsule_vec_inplane = Vector3(C_capsule_vec.dot(C_plane_x), C_capsule_vec.dot(C_plane_y), C_capsule_vec.dot(C_plane_z));
	SB_DEV_ASSERT(Math::is_zero_approx(C_capsule_vec_inplane.z));
	real_t capsule_vec_slope = C_capsule_vec_inplane.y / C_capsule_vec_inplane.x;

	// The apex of the cone relative to plane C frame is (0, ya, perp_dist)
	// where ya will vary to make different intersections with the C plane as a hyperbola
	// to find the value where it is tangential to C_capsule_vec_inplane.

	real_t cone_slope = (p_bone_radiusP - p_bone_origin_radiusP) / p_bone_length;
	// perp_dist^2 + x^2 = ((y - ya)*cone_slope)^2,
	// Diff by x:  2x = 2(y - ya)cone_slope * dy/dx

	// Simultaneous tangential equations to solve are:
	//   y = x * capsule_vec_slope
	//   perp_dist^2 + x^2 = ((y - ya)*cone_slope)^2
	//   dy/dx = capsule_vec_slope
	//   x = (y - ya) * dy/dx * cone_slope^2

	//   x = (y - ya)*cone_slope * capsule_vec_slope * cone_slope
	//   x/(capsule_vec_slope * cone_slope) = (y - ya)*cone_slope
	//   perp_dist^2 = x^2*((1/(capsule_vec_slope * cone_slope))^2 - 1)
	//   x^2 = perp_dist^2 / ((1/(capsule_vec_slope * cone_slope))^2 - 1)

	// Protect division by zero which occurs with alignment of the capsule beyond the asymtote
	real_t cc = capsule_vec_slope * cone_slope;
	real_t perp_dist_sq = perp_dist * perp_dist;
	real_t xsq_num = perp_dist_sq * cc * cc;
	real_t xsq_den = 1 - cc * cc;
	if (xsq_den <= xsq_num * 0.0000001 + CMP_EPSILON) {
		// this needs to pick the best endpoint of the capsule that will hit the cone
		return -1.0;
	}
	real_t xsq = xsq_num / xsq_den;

	real_t x = sqrt(xsq);
	if ((capsule_vec_slope > 0) == (p_bone_origin_radius > p_bone_radius)) {
		x = -x;
	}
	real_t y = x * capsule_vec_slope;
	//real_t ya = y - x / (capsule_vec_slope * cone_slope * cone_slope);

	real_t emu = y / (C_capsule_vec_inplane.y * p_length);
	real_t mu0 = mu + emu;

	// If the sphere in the bone cone is off and endpoint, then pick a mu in the collision capsule
	// that is closest to that end point
	real_t cone_axis_rad = sqrt(perp_dist_sq + xsq);
	real_t lam_cone_sphere = lam + (y + cone_axis_rad * cone_slope) / p_bone_length;
	if ((lam_cone_sphere < 0) || (lam_cone_sphere > 1)) {
		mu0 = p.dot((lam_cone_sphere < 0 ? p_current_origin : p_current) - head) / pdp;
	}
	return (mu0 < 0.0 ? 0.0 : (mu0 > 1.0 ? 1.0 : mu0)); // clamp(0,1)
}

Vector3 SpringBoneCollisionCapsule3D::_collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3 &p_current_origin, float p_bone_origin_radius, const Vector3 &p_current) const {
	// The tapered bone capsule is from (p_current_origin, p_bone_origin_radius) to (p_current, p_bone_radius).
	Pair<Vector3, Vector3> head_tail = get_head_and_tail(p_center);
	Vector3 head = head_tail.first;
	Vector3 tail = head_tail.second;

	// dispose of the non-capsule bone chains (the capsule collider just hits each bone node).
	if (collide_mode != COLLIDE_MODE_CHAIN) {
		// Pick sphere in collider capsule that best collides with the bone end point (the joint).
		Vector3 capsule_sphere_center = _closest_capsule_sphere(head, tail, p_current);
		return _collide_sphere(capsule_sphere_center, radius, (collide_mode == COLLIDE_MODE_INSIDE), p_bone_radius, p_current);
	}

	real_t capsule_mu = _closest_capsule_sphere_to_taper(head, tail, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	if (capsule_mu == -1.0) {
		return p_current;
	}

	Vector3 capsule_sphere_center = head * (1.0 - capsule_mu) + tail * capsule_mu;

	// Numerically test the claim that we have found the sphere in the collision capsule that enters the bone cone the deepest.
#ifdef VERIFY_SPRINGBONECAPSULE_CALCULATIONS
	real_t Dvdist = verify_distance_within_taper(capsule_sphere_center, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	real_t mulo = MAX(capsule_mu - 0.1, 0.0);
	Vector3 caplo = head * (1.0 - mulo) + tail * mulo;
	real_t muhi = MIN(capsule_mu + 0.1, 1.0);
	Vector3 caphi = head * (1.0 - muhi) + tail * muhi;
	real_t Dvdistlo = verify_distance_within_taper(caplo, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	real_t Dvdisthi = verify_distance_within_taper(caphi, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
	if (Dvdistlo < Dvdist - 0.001) {
		printf("%d Non-minimal mu=%.2f  %.3f < %.3f lo\n", SpringBoneCollision3D::Dsegmentindexbeingcalculated, capsule_mu, Dvdistlo, Dvdist);
	}
	if (Dvdisthi < Dvdist - 0.001) {
		printf("%d Non-minimal mu=%.2f  %.3f < %.3f hi\n", SpringBoneCollision3D::Dsegmentindexbeingcalculated, capsule_mu, Dvdisthi, Dvdist);
	}
#endif

	return _collide_sphere_taper(capsule_sphere_center, radius, p_bone_radius, p_bone_length, p_current_origin, p_bone_origin_radius, p_current);
}
