/**************************************************************************/
/*  joint_limitation_kusudama_3d.cpp                                      */
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

#include "joint_limitation_kusudama_3d.h"

void JointLimitationKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cone_count", "count"), &JointLimitationKusudama3D::set_cone_count);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &JointLimitationKusudama3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_cone_center", "index", "center"), &JointLimitationKusudama3D::set_cone_center);
	ClassDB::bind_method(D_METHOD("get_cone_center", "index"), &JointLimitationKusudama3D::get_cone_center);
	ClassDB::bind_method(D_METHOD("set_cone_radius", "index", "radius"), &JointLimitationKusudama3D::set_cone_radius);
	ClassDB::bind_method(D_METHOD("get_cone_radius", "index"), &JointLimitationKusudama3D::get_cone_radius);
}

// Helper function to find closest point on cone boundary
// Returns NaN if point is inside cone, otherwise returns closest boundary point
static Vector3 closest_to_cone_boundary(const Vector3 &p_input, const Vector3 &p_control_point, real_t p_radius) {
	Vector3 normalized_input = p_input.normalized();
	Vector3 normalized_control_point = p_control_point.normalized();
	real_t radius_cosine = Math::cos(p_radius);

	// If point is inside cone, return NaN
	if (normalized_input.dot(normalized_control_point) > radius_cosine) {
		return Vector3(NAN, NAN, NAN);
	}

	// Find axis for rotation using cross product (robust handling without interval arithmetic)
	Vector3 axis = normalized_control_point.cross(normalized_input);

	// Additional validation for the axis
	if (!axis.is_finite() || Math::is_zero_approx(axis.length_squared())) {
		// Fallback: use any perpendicular vector to the control point
		// But first check that control point is valid
		if (!normalized_control_point.is_finite() || Math::is_zero_approx(normalized_control_point.length_squared())) {
			// Can't compute boundary with zero control point
			return Vector3(NAN, NAN, NAN);
		}
		axis = normalized_control_point.get_any_perpendicular();
		if (!axis.is_finite() || Math::is_zero_approx(axis.length_squared())) {
			axis = Vector3(0, 1, 0);
		}
	}
	axis.normalize();

	// Rotate control point by radius around axis to get boundary point
	Quaternion rot_to = Quaternion(axis.normalized(), p_radius);
	Vector3 axis_control_point = normalized_control_point;
	if (Math::is_zero_approx(axis_control_point.length_squared())) {
		axis_control_point = Vector3(0, 1, 0);
	}
	Vector3 result = rot_to.xform(axis_control_point);
	return result.normalized();
}

// Update tangents for a specific quad based on its cone1 and cone2
void JointLimitationKusudama3D::_update_quad_tangents(int p_quad_index) {
	int quad_count = cones.size() / 3;
	if (p_quad_index < 0 || p_quad_index >= quad_count) {
		return;
	}

	int base_idx = p_quad_index * 3;
	Vector4 cone1_vec = cones[base_idx + 0];
	// cone2 is either the next quad's cone1, or empty for the last quad
	Vector4 cone2_vec;
	if (p_quad_index < quad_count - 1) {
		cone2_vec = cones[(p_quad_index + 1) * 3 + 0]; // Next quad's cone1
	} else {
		// Last triplet: check if there's a stored cone2 (would be at index cones.size() if it exists)
		// For now, we'll need to track this differently or use a sentinel
		// Actually, for the last quad, cone2 should be stored separately or we need to check
		// Let's use a different approach: store cone2 in a separate location for the last triplet
		// Or we can check if cones.size() % 3 == 1, meaning there's an extra cone2
		if (cones.size() % 3 == 1) {
			cone2_vec = cones[cones.size() - 1]; // Extra cone2 stored at the end
		} else {
			cone2_vec = Vector4();
		}
	}

	Vector3 center1 = Vector3(cone1_vec.x, cone1_vec.y, cone1_vec.z);
	Vector3 center2 = Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z);
	real_t radius1 = cone1_vec.w;
	real_t radius2 = cone2_vec.w;

	// Validate that both centers are valid and non-zero
	if (center1.length_squared() < CMP_EPSILON || center2.length_squared() < CMP_EPSILON) {
		// Set tangents to zero/invalid to indicate they're not usable
		cones[base_idx + 1] = Vector4();
		cones[base_idx + 2] = Vector4();
		return;
	}

	center1 = center1.normalized();
	center2 = center2.normalized();

	// Double-check after normalization (shouldn't be needed, but be safe)
	if (!center1.is_finite() || !center2.is_finite() ||
			Math::is_zero_approx(center1.length_squared()) ||
			Math::is_zero_approx(center2.length_squared())) {
		cones[base_idx + 1] = Vector4();
		cones[base_idx + 2] = Vector4();
		return;
	}

	// Compute tangent circle radius
	real_t tan_radius = (Math::PI - (radius1 + radius2)) / 2.0;

	// Find arc normal (axis perpendicular to both cone centers)
	Vector3 arc_normal = center1.cross(center2);
	real_t arc_normal_len = arc_normal.length();

	Vector3 tan1, tan2;

	if (arc_normal_len < CMP_EPSILON) {
		// Handle parallel/opposite cones
		// Ensure center1 is valid before using it
		if (!center1.is_finite() || Math::is_zero_approx(center1.length_squared())) {
			cones[base_idx + 1] = Vector4();
			cones[base_idx + 2] = Vector4();
			return;
		}
		// For opposite cones, any perpendicular to center1 works
		arc_normal = center1.get_any_perpendicular();
		if (arc_normal.is_zero_approx() || !arc_normal.is_finite()) {
			arc_normal = Vector3(0, 1, 0);
		}
		arc_normal.normalize();

		// For opposite cones, tangent circles are at 90 degrees from the cone centers
		// Use a perpendicular vector in the plane perpendicular to center1
		Vector3 perp1 = center1.get_any_perpendicular();
		if (perp1.is_zero_approx() || !perp1.is_finite()) {
			perp1 = Vector3(0, 1, 0);
		}
		perp1.normalize();

		// Rotate around center1 by the tangent radius to get tangent centers
		Quaternion rot1 = Quaternion(center1.normalized(), tan_radius);
		Quaternion rot2 = Quaternion(center1.normalized(), -tan_radius);
		tan1 = rot1.xform(perp1).normalized();
		tan2 = rot2.xform(perp1).normalized();
	} else {
		arc_normal.normalize();

		// Use plane intersection method (simplified version)
		real_t boundary_plus_tan_radius_a = radius1 + tan_radius;
		real_t boundary_plus_tan_radius_b = radius2 + tan_radius;

		Vector3 scaled_axis_a = center1 * Math::cos(boundary_plus_tan_radius_a);
		Vector3 safe_arc_normal = arc_normal;
		if (Math::is_zero_approx(safe_arc_normal.length_squared())) {
			safe_arc_normal = Vector3(0, 1, 0);
		}
		Quaternion rot_arc_a = Quaternion(safe_arc_normal.normalized(), boundary_plus_tan_radius_a);
		Vector3 plane_dir1_a = rot_arc_a.xform(center1);
		Vector3 safe_center1 = center1;
		if (Math::is_zero_approx(safe_center1.length_squared())) {
			safe_center1 = Vector3(0, 0, 1);
		}
		Quaternion rot_perp_a = Quaternion(safe_center1.normalized(), Math::PI / 2);
		Vector3 plane_dir2_a = rot_perp_a.xform(plane_dir1_a);

		Vector3 scaled_axis_b = center2 * Math::cos(boundary_plus_tan_radius_b);
		Quaternion rot_arc_b = Quaternion(safe_arc_normal.normalized(), boundary_plus_tan_radius_b);
		Vector3 plane_dir1_b = rot_arc_b.xform(center2);
		Vector3 safe_center2 = center2;
		if (Math::is_zero_approx(safe_center2.length_squared())) {
			safe_center2 = Vector3(0, 0, 1);
		}
		Quaternion rot_perp_b = Quaternion(safe_center2.normalized(), Math::PI / 2);
		Vector3 plane_dir2_b = rot_perp_b.xform(plane_dir1_b);

		// Extend rays
		Vector3 ray1_b_start = plane_dir1_b;
		Vector3 ray1_b_end = scaled_axis_b;
		Vector3 ray2_b_start = plane_dir1_b;
		Vector3 ray2_b_end = plane_dir2_b;

		Vector3 mid1 = (ray1_b_start + ray1_b_end) * 0.5;
		Vector3 dir1 = (ray1_b_start - mid1).normalized();
		ray1_b_start = mid1 - dir1 * 99.0;
		ray1_b_end = mid1 + dir1 * 99.0;

		Vector3 mid2 = (ray2_b_start + ray2_b_end) * 0.5;
		Vector3 dir2 = (ray2_b_start - mid2).normalized();
		ray2_b_start = mid2 - dir2 * 99.0;
		ray2_b_end = mid2 + dir2 * 99.0;

		// Ray-plane intersection
		Vector3 plane_edge1 = plane_dir1_a - scaled_axis_a;
		Vector3 plane_edge2 = plane_dir2_a - scaled_axis_a;
		Vector3 plane_normal = plane_edge1.cross(plane_edge2).normalized();

		Vector3 ray1_dir = (ray1_b_end - ray1_b_start).normalized();
		Vector3 ray1_to_plane = ray1_b_start - scaled_axis_a;
		real_t plane_dist1 = -plane_normal.dot(ray1_to_plane);
		real_t ray1_dot_normal = plane_normal.dot(ray1_dir);
		Vector3 intersection1 = ray1_b_start + ray1_dir * (plane_dist1 / ray1_dot_normal);

		Vector3 ray2_dir = (ray2_b_end - ray2_b_start).normalized();
		Vector3 ray2_to_plane = ray2_b_start - scaled_axis_a;
		real_t plane_dist2 = -plane_normal.dot(ray2_to_plane);
		real_t ray2_dot_normal = plane_normal.dot(ray2_dir);
		Vector3 intersection2 = ray2_b_start + ray2_dir * (plane_dist2 / ray2_dot_normal);

		// Ray-sphere intersection
		Vector3 intersection_ray_start = intersection1;
		Vector3 intersection_ray_end = intersection2;
		Vector3 intersection_ray_mid = (intersection_ray_start + intersection_ray_end) * 0.5;
		Vector3 intersection_ray_dir_ext = (intersection_ray_start - intersection_ray_mid).normalized();
		intersection_ray_start = intersection_ray_mid - intersection_ray_dir_ext * 99.0;
		intersection_ray_end = intersection_ray_mid + intersection_ray_dir_ext * 99.0;

		Vector3 ray_start_rel = intersection_ray_start;
		Vector3 ray_end_rel = intersection_ray_end;
		Vector3 direction = ray_end_rel - ray_start_rel;
		Vector3 ray_dir_normalized = direction.normalized();
		Vector3 ray_to_center = -ray_start_rel;
		real_t ray_dot_center = ray_dir_normalized.dot(ray_to_center);
		real_t radius_squared = 1.0;
		real_t center_dist_squared = ray_to_center.length_squared();
		real_t ray_dot_squared = ray_dot_center * ray_dot_center;
		real_t discriminant = radius_squared - center_dist_squared + ray_dot_squared;

		if (discriminant >= 0.0) {
			real_t sqrt_discriminant = Math::sqrt(discriminant);
			real_t t1 = ray_dot_center - sqrt_discriminant;
			real_t t2 = ray_dot_center + sqrt_discriminant;
			tan1 = (intersection_ray_start + ray_dir_normalized * t1).normalized();
			tan2 = (intersection_ray_start + ray_dir_normalized * t2).normalized();
		} else {
			// Fallback - ensure center1 is valid before using it
			if (!center1.is_finite() || Math::is_zero_approx(center1.length_squared())) {
				// Can't compute tangents with invalid center
				cones[base_idx + 1] = Vector4();
				cones[base_idx + 2] = Vector4();
				return;
			}
			Vector3 perp1 = center1.get_any_perpendicular();
			if (perp1.is_zero_approx() || !perp1.is_finite()) {
				perp1 = Vector3(0, 1, 0);
			}
			perp1.normalize();
			Quaternion rot1 = Quaternion(center1.normalized(), tan_radius);
			Quaternion rot2 = Quaternion(center1.normalized(), -tan_radius);
			tan1 = rot1.xform(perp1).normalized();
			tan2 = rot2.xform(perp1).normalized();
		}
	}

	// Handle degenerate tangent centers (NaN or zero)
	if (!tan1.is_finite() || Math::is_zero_approx(tan1.length_squared())) {
		// Ensure center1 is valid before using it
		if (!center1.is_finite() || Math::is_zero_approx(center1.length_squared())) {
			cones[base_idx + 1] = Vector4();
			cones[base_idx + 2] = Vector4();
			return;
		}
		tan1 = center1.get_any_perpendicular();
		if (tan1.is_zero_approx() || !tan1.is_finite()) {
			tan1 = Vector3(0, 1, 0);
		}
		tan1.normalize();
	}
	if (!tan2.is_finite() || Math::is_zero_approx(tan2.length_squared())) {
		// Choose a valid base vector
		Vector3 orthogonal_base;
		if (tan1.is_finite() && !Math::is_zero_approx(tan1.length_squared())) {
			orthogonal_base = tan1;
		} else if (center1.is_finite() && !Math::is_zero_approx(center1.length_squared())) {
			orthogonal_base = center1;
		} else {
			cones[base_idx + 1] = Vector4();
			cones[base_idx + 2] = Vector4();
			return;
		}
		tan2 = orthogonal_base.get_any_perpendicular();
		if (tan2.is_zero_approx() || !tan2.is_finite()) {
			tan2 = Vector3(1, 0, 0);
		}
		tan2.normalize();
	}

	// Store tangents in the quad
	// Swap storage to match shader expectations: tan2 in tan1, tan1 in tan2
	cones[base_idx + 1] = Vector4(tan2.x, tan2.y, tan2.z, tan_radius);
	cones[base_idx + 2] = Vector4(tan1.x, tan1.y, tan1.z, tan_radius);

	// Store cone2 for the last quad if needed
	if (p_quad_index == quad_count - 1 && cone2_vec.length_squared() >= CMP_EPSILON) {
		// Ensure we have space for the extra cone2 at the end
		int expected_size = quad_count * 3 + 1;
		if (cones.size() < expected_size) {
			cones.resize(expected_size);
		}
		cones[quad_count * 3] = cone2_vec;
	}
}

Vector3 JointLimitationKusudama3D::_solve(const Vector3 &p_direction) const {
	// If constraints are disabled, return the original direction
	if (cones.is_empty()) {
		return p_direction.normalized();
	}

	Vector3 point = p_direction.normalized();
	real_t closest_cos = -2.0;
	Vector3 closest_collision_point = point;

	// Loop through each limit cone
	// Extract all unique cones from storage: cone0 from group[0].cone1, cone1 from group[0]'s cone2 (which is group[1].cone1 or stored separately for last group), etc.
	int quad_count = cones.size() / 3;
	for (int i = 0; i < quad_count; i++) {
		int base_idx = i * 3;
		// Check cone1 of this group
		Vector4 cone1_vec = cones[base_idx + 0];
		Vector3 control_point = Vector3(cone1_vec.x, cone1_vec.y, cone1_vec.z).normalized();
		real_t radius = cone1_vec.w;

		Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);

		// If the collision point is NaN, return the original point (point is in bounds)
		if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
			return point.normalized();
		}

		// Calculate the cosine of the angle between the collision point and the original point
		real_t this_cos = collision_point.dot(point);

		// If the closest collision point is not set or the cosine is greater than the current closest cosine, update the closest collision point and cosine
		if (closest_collision_point.is_zero_approx() || this_cos > closest_cos) {
			closest_collision_point = collision_point;
			closest_cos = this_cos;
		}
	}
	// Also check the last cone (cone2 of the last group)
	if (quad_count > 0) {
		// cone2 of last group is stored at the end if cones.size() % 3 == 1, otherwise it's empty
		Vector4 cone2_vec;
		if (cones.size() % 3 == 1) {
			cone2_vec = cones[cones.size() - 1]; // Extra cone2 at the end
		} else {
			cone2_vec = Vector4(); // Empty
		}
		Vector3 control_point_raw = Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z);
		// Skip if cone2 is empty (zero vector)
		if (control_point_raw.length_squared() < CMP_EPSILON) {
			// cone2 is empty, skip it
		} else {
			Vector3 control_point = control_point_raw.normalized();
			real_t radius = cone2_vec.w;

			Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);
			if (!Math::is_nan(collision_point.x)) {
				real_t this_cos = collision_point.dot(point);
				if (Math::is_equal_approx(this_cos, real_t(1.0))) {
					return point.normalized();
				}
				if (this_cos > closest_cos) {
					closest_collision_point = collision_point;
					closest_cos = this_cos;
				}
			}
		}
	}

	// If we're out of bounds of all cones, check if we're in the paths between the cones
	// IMPORTANT: We explicitly do NOT check the pair (last_cone, first_cone) to prevent wrap-around
	// For each cone pair, get the group: [cone1, tan2, tan1] (cone2 is implicit: next group's cone1 or stored separately for last group)
	// Note: tangents are swapped in storage (+1 stores tan2, +2 stores tan1)
	if (quad_count > 0) {
		for (int i = 0; i < quad_count; i++) {
			int base_idx = i * 3;
			// Get group: [cone1, tan2, tan1] (swapped storage)
			// cone2 is either next group's cone1 or stored separately for last group (if count >= 2)
			Vector4 cone1_vec = cones[base_idx + 0];
			Vector4 tan1_vec = cones[base_idx + 1]; // Actually stores tan2 (swapped)
			Vector4 cone2_vec;
			if (i < quad_count - 1) {
				cone2_vec = cones[(i + 1) * 3 + 0]; // Next quad's cone1
			} else {
				// Last triplet: check if cone2 is stored at the end
				if (cones.size() % 3 == 1) {
					cone2_vec = cones[cones.size() - 1];
				} else {
					cone2_vec = Vector4();
				}
			}

			Vector3 center1_raw = Vector3(cone1_vec.x, cone1_vec.y, cone1_vec.z);
			Vector3 center2_raw = Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z);

			// Skip this quad if cone2 is empty (zero vector)
			if (center2_raw.length_squared() < CMP_EPSILON) {
				continue;
			}

			Vector3 center1 = center1_raw.normalized();
			Vector3 center2 = center2_raw.normalized();
			real_t tan_radius = tan1_vec.w; // tan1 and tan2 have same radius

			// Check all 4 regions: iterate through [cone1, tan2, tan1, cone2] (tangents are swapped in storage)
			real_t tan_radius_cos = Math::cos(tan_radius);
			Vector4 quad_elements[4] = { cones[base_idx + 0], cones[base_idx + 1], cones[base_idx + 2], cone2_vec };
			for (int quad_idx = 0; quad_idx < 4; quad_idx++) {
				Vector4 elem_vec = quad_elements[quad_idx];
				Vector3 elem_center = Vector3(elem_vec.x, elem_vec.y, elem_vec.z).normalized();
				real_t elem_radius = elem_vec.w;
				bool is_tangent = (quad_idx == 1 || quad_idx == 2);

				Vector3 collision_point;

				if (!is_tangent) {
					// Check cone region
					collision_point = closest_to_cone_boundary(point, elem_center, elem_radius);
				} else {
					// Check tangent region
					// For quad_idx == 1: actually tan2 (swapped storage), check region between center1 and center2
					// For quad_idx == 2: actually tan1 (swapped storage), check region between center1 and center2
					Vector3 c1xt = center1.cross(elem_center);
					Vector3 txc2 = elem_center.cross(center2);
					if (quad_idx == 2) {
						// tan1 (stored at idx 2): reverse the cross products
						c1xt = elem_center.cross(center1);
						txc2 = center2.cross(elem_center);
					}

					if (point.dot(c1xt) > 0 && point.dot(txc2) > 0) {
						real_t to_tan_cos = point.dot(elem_center);
						if (to_tan_cos > tan_radius_cos) {
							// Project onto tangent circle
							Vector3 plane_normal = elem_center.cross(point);
							if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
								plane_normal = Vector3(0, 1, 0);
							}
							plane_normal.normalize();
							Quaternion rotate_about_by = Quaternion(plane_normal.normalized(), elem_radius);
							collision_point = rotate_about_by.xform(elem_center).normalized();
						} else {
							// Point is inside tangent circle, so it's valid
							collision_point = point;
						}
					} else {
						collision_point = Vector3(NAN, NAN, NAN);
					}
				}

				// Process collision point
				if (Math::is_nan(collision_point.x)) {
					continue;
				}

				real_t this_cos = collision_point.dot(point);
				if (Math::is_equal_approx(this_cos, real_t(1.0))) {
					return point.normalized();
				}
				if (this_cos > closest_cos) {
					closest_collision_point = collision_point;
					closest_cos = this_cos;
				}
			}
		}
	}

	// Return the closest boundary point between cones
	return closest_collision_point.normalized();
}

void JointLimitationKusudama3D::set_cone_count(int p_count) {
	if (p_count < 0) {
		p_count = 0;
	}

	// Handle zero case
	if (p_count == 0) {
		cones.clear();
		notify_property_list_changed();
		emit_changed();
		return;
	}

	// Storage format:
	// - 1 cone: 1 group (3 elements: cone1, tan2, tan1), no extra cone2
	// - 2+ cones: n-1 groups (3 elements each) + 1 extra cone2 at the end
	// Note: tangents are swapped in storage (+1 stores tan2, +2 stores tan1) to match shader expectations
	// So for p_count cones, we need: max(1, p_count-1) groups, plus 1 extra element if p_count >= 2

	int old_quad_count = cones.size() / 3;
	bool old_has_cone2 = (cones.size() % 3 == 1);

	// Calculate new storage size
	int new_quad_count = (p_count == 1) ? 1 : (p_count - 1);
	int new_size = new_quad_count * 3;
	if (p_count >= 2) {
		new_size += 1; // Extra space for last quad's cone2
	}

	// Check if we actually need to resize
	if (old_quad_count == new_quad_count && old_has_cone2 == (p_count >= 2)) {
		// Same structure, no change needed
		return;
	}

	int old_size = cones.size();
	cones.resize(new_size);

	// Initialize new groups with default values
	const Vector4 default_cone = Vector4(0, 1, 0, Math::PI * 0.25); // Default: +Y axis, 45 degrees
	for (int i = old_quad_count; i < new_quad_count; i++) {
		int base_idx = i * 3;
		cones[base_idx + 0] = default_cone; // cone1
		// Tangents start empty (storage is swapped: +1 stores tan2, +2 stores tan1)
		cones[base_idx + 1] = Vector4(); // tan2 (swapped storage)
		cones[base_idx + 2] = Vector4(); // tan1 (swapped storage)
	}

	// Initialize cone2 for last group if count is 2+
	if (new_quad_count > 0 && p_count >= 2) {
		int cone2_idx = new_quad_count * 3;
		if (cone2_idx < cones.size()) {
			// Only initialize if this is a new element (wasn't in old size)
			if (cone2_idx >= old_size) {
				cones[cone2_idx] = default_cone;
			} else {
				// Check if existing cone2 is empty and initialize it
				Vector3 cone2_center = Vector3(cones[cone2_idx].x, cones[cone2_idx].y, cones[cone2_idx].z);
				if (cone2_center.length_squared() < CMP_EPSILON) {
					cones[cone2_idx] = default_cone;
				}
			}
		}
	}

	// Recompute tangents for all groups
	for (int i = 0; i < new_quad_count; i++) {
		_update_quad_tangents(i);
	}

	notify_property_list_changed();
	emit_changed();
}

int JointLimitationKusudama3D::get_cone_count() const {
	if (cones.is_empty()) {
		return 0;
	}

	int quad_count = cones.size() / 3;
	// Number of unique cones = number of groups + 1
	// Special case: if we have 1 group but cone2 is empty, we only have 1 cone
	if (quad_count == 1) {
		// Check if cone2 is stored at the end
		if (cones.size() % 3 == 1) {
			Vector3 cone2_center = Vector3(cones[cones.size() - 1].x, cones[cones.size() - 1].y, cones[cones.size() - 1].z);
			if (cone2_center.length_squared() < CMP_EPSILON) {
				return 1;
			}
		} else {
			return 1; // No cone2 stored, so only 1 cone
		}
	}
	return quad_count + 1;
}

void JointLimitationKusudama3D::set_cone_center(int p_index, const Vector3 &p_center) {
	if (p_index < 0) {
		return;
	}

	int quad_count = cones.size() / 3;

	ERR_FAIL_INDEX(p_index, quad_count + 1);

	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	if (p_index < quad_count) {
		// Update cone1 of group at p_index
		int base_idx = p_index * 3;
		cones[base_idx + 0].x = p_center.x;
		cones[base_idx + 0].y = p_center.y;
		cones[base_idx + 0].z = p_center.z;

		// Update previous group's tangents since its cone2 is implicitly the same as this group's cone1
		// (cone2 of group[i-1] = cone1 of group[i])
		if (p_index > 0) {
			Vector4 prev_cone1 = cones[(p_index - 1) * 3 + 0];
			if (Vector3(prev_cone1.x, prev_cone1.y, prev_cone1.z).length_squared() >= CMP_EPSILON) {
				_update_quad_tangents(p_index - 1);
			}
		}

		// Update tangents if cone2 is valid (next group's cone1 or stored separately for last group)
		Vector4 cone2_vec;
		if (p_index < quad_count - 1) {
			cone2_vec = cones[(p_index + 1) * 3 + 0];
		} else {
			// Last group: check if cone2 is stored
			if (cones.size() % 3 == 1) {
				cone2_vec = cones[cones.size() - 1];
			} else {
				cone2_vec = Vector4();
			}
		}
		if (Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z).length_squared() >= CMP_EPSILON) {
			_update_quad_tangents(p_index);
		}
	} else {
		// Update cone2 of last group (stored at the end)
		int cone2_idx = quad_count * 3;
		if (cones.size() <= cone2_idx) {
			cones.resize(cone2_idx + 1);
		}
		Vector3 old_cone2 = Vector3(cones[cone2_idx].x, cones[cone2_idx].y, cones[cone2_idx].z);
		bool was_single_cone = (quad_count == 1 && old_cone2.length_squared() < CMP_EPSILON);

		cones[cone2_idx].x = p_center.x;
		cones[cone2_idx].y = p_center.y;
		cones[cone2_idx].z = p_center.z;
		_update_quad_tangents(quad_count - 1);

		// Notify property list when transitioning from 1 to 2 cones
		if (was_single_cone) {
			notify_property_list_changed();
		}
	}

	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	int quad_count = cones.size() / 3;
	ERR_FAIL_INDEX_V(p_index, quad_count + 1, Vector3(0, 1, 0));

	// If there are no groups, return default value
	if (cones.is_empty()) {
		return Vector3(0, 1, 0);
	}

	Vector4 cone_vec;
	if (p_index < quad_count) {
		// Access cone1 of group at p_index
		cone_vec = cones[p_index * 3 + 0];
	} else {
		// Access cone2 of last group (stored at the end)
		if (cones.size() % 3 == 1) {
			cone_vec = cones[cones.size() - 1];
		} else {
			cone_vec = Vector4(); // Empty
		}
	}
	Vector3 center = Vector3(cone_vec.x, cone_vec.y, cone_vec.z);
	// Normalize when reading to ensure we always return a normalized value
	if (center.length_squared() > CMP_EPSILON) {
		return center.normalized();
	}
	return Vector3(0, 1, 0);
}

void JointLimitationKusudama3D::set_cone_radius(int p_index, real_t p_radius) {
	if (p_index < 0) {
		return;
	}

	int quad_count = cones.size() / 3;

	ERR_FAIL_INDEX(p_index, quad_count + 1);

	if (p_index < quad_count) {
		// Access cone1 of group at p_index
		int base_idx = p_index * 3;
		cones[base_idx + 0].w = p_radius;
		// Update previous group's tangents since its cone2 is implicitly the same as this group's cone1
		// (cone2 of group[i-1] = cone1 of group[i], stored at the same location)
		if (p_index > 0) {
			// Only update tangents if cone1 of previous group is also valid
			Vector4 prev_cone1 = cones[(p_index - 1) * 3 + 0];
			if (Vector3(prev_cone1.x, prev_cone1.y, prev_cone1.z).length_squared() >= CMP_EPSILON) {
				_update_quad_tangents(p_index - 1);
			}
		}
		// Only update tangents if cone2 of this group is also valid
		Vector4 cone2_vec;
		if (p_index < quad_count - 1) {
			// cone2 is the next group's cone1
			cone2_vec = cones[(p_index + 1) * 3 + 0];
		} else {
			// Last group: check if cone2 is stored at the end
			if (cones.size() % 3 == 1) {
				cone2_vec = cones[cones.size() - 1];
			} else {
				cone2_vec = Vector4();
			}
		}
		if (Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z).length_squared() >= CMP_EPSILON) {
			_update_quad_tangents(p_index);
		}
	} else {
		// Access cone2 of last quad (stored at the end)
		int cone2_idx = quad_count * 3;
		if (cones.size() <= cone2_idx) {
			cones.resize(cone2_idx + 1);
		}
		Vector3 old_cone2 = Vector3(cones[cone2_idx].x, cones[cone2_idx].y, cones[cone2_idx].z);
		bool was_single_cone = (quad_count == 1 && old_cone2.length_squared() < CMP_EPSILON);

		cones[cone2_idx].w = p_radius;
		_update_quad_tangents(quad_count - 1);

		// If we just transitioned from 1 to 2 cones, notify property list
		if (was_single_cone) {
			notify_property_list_changed();
		}
	}
	emit_changed();
}

real_t JointLimitationKusudama3D::get_cone_radius(int p_index) const {
	int quad_count = cones.size() / 3;
	ERR_FAIL_INDEX_V(p_index, quad_count + 1, 0.0);

	// If there are no quads, return default value
	if (cones.is_empty()) {
		return Math::PI * 0.25; // Default 45 degrees
	}

	if (p_index < quad_count) {
		// Access cone1 of quad at p_index
		return cones[p_index * 3 + 0].w;
	} else {
		// Access cone2 of last quad (stored at the end)
		if (cones.size() % 3 == 1) {
			return cones[cones.size() - 1].w;
		} else {
			return Math::PI * 0.25; // Default if not stored
		}
	}
}

bool JointLimitationKusudama3D::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;
	if (prop_name == "cone_count") {
		set_cone_count(p_value);
		return true;
	}
	if (prop_name.begins_with("cones/")) {
		int index = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		if (what == "center") {
			// Validate index before proceeding
			if (index < 0) {
				return false; // Invalid index
			}
			// Handle quaternion input from inspector
			if (p_value.get_type() == Variant::QUATERNION) {
				int quad_count = cones.size() / 3;
				// Allow index 0 even when empty (will initialize)
				if (quad_count > 0) {
					ERR_FAIL_INDEX_V(index, quad_count + 1, false);
				}
				// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
				Vector3 default_dir = Vector3(0, 1, 0);
				Vector3 center = Quaternion(p_value).normalized().xform(default_dir);
				set_cone_center(index, center);
			} else {
				set_cone_center(index, p_value);
			}
			return true;
		}
		if (what == "radius") {
			// Validate index before proceeding
			if (index < 0) {
				return false; // Invalid index
			}
			set_cone_radius(index, p_value);
			return true;
		}
	}
	return false;
}

bool JointLimitationKusudama3D::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;
	if (prop_name == "cone_count") {
		r_ret = get_cone_count();
		return true;
	}
	if (prop_name.begins_with("cones/")) {
		int index = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		if (what == "center") {
			// Return as quaternion for inspector display with degrees
			int quad_count = cones.size() / 3;
			ERR_FAIL_INDEX_V(index, quad_count + 1, false);
			Vector3 center = get_cone_center(index); // This already normalizes
			Vector3 default_dir = Vector3(0, 1, 0);
			// Create quaternion representing rotation from default_dir to center
			r_ret = Quaternion(default_dir, center);
			return true;
		}
		if (what == "radius") {
			r_ret = get_cone_radius(index);
			return true;
		}
	}
	return false;
}

void JointLimitationKusudama3D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("cone_count"), PROPERTY_HINT_RANGE, "0,16384,1,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Cones," + String(PNAME("cones")) + "/"));
	for (int i = 0; i < get_cone_count(); i++) {
		const String prefix = vformat("%s/%d/", PNAME("cones"), i);
		// Use quaternion for inspector display with Euler angles in degrees
		p_list->push_back(PropertyInfo(Variant::QUATERNION, prefix + PNAME("center"), PROPERTY_HINT_NONE, ""));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prefix + PNAME("radius"), PROPERTY_HINT_RANGE, "0,180,0.1,radians_as_degrees"));
	}
}

#ifdef TOOLS_ENABLED
LocalVector<JointLimitationKusudama3D::Segment> JointLimitationKusudama3D::get_icosahedron_sphere(int p_subdiv) const {
	// TODO: Define icosahedron statically in the header.
	// Make subdivided icosahedron sphere.
	// All points' length are 1.0 from 0.0.
	LocalVector<Segment> ret;

	if (p_subdiv < 0) {
		p_subdiv = 0;
	}

	// Base icosahedron (unit sphere).
	// Vertex set: (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
	const real_t phi = ((real_t)1.0 + Math::sqrt((real_t)5.0)) * (real_t)0.5;
	Vector3 v[12] = {
		Vector3(-1, phi, 0),
		Vector3(1, phi, 0),
		Vector3(-1, -phi, 0),
		Vector3(1, -phi, 0),
		Vector3(0, -1, phi),
		Vector3(0, 1, phi),
		Vector3(0, -1, -phi),
		Vector3(0, 1, -phi),
		Vector3(phi, 0, -1),
		Vector3(phi, 0, 1),
		Vector3(-phi, 0, -1),
		Vector3(-phi, 0, 1)
	};
	for (int i = 0; i < 12; i++) {
		v[i].normalize();
	}

	// Faces (20 triangles).
	static const int faces[20][3] = {
		{ 0, 11, 5 },
		{ 0, 5, 1 },
		{ 0, 1, 7 },
		{ 0, 7, 10 },
		{ 0, 10, 11 },
		{ 1, 5, 9 },
		{ 5, 11, 4 },
		{ 11, 10, 2 },
		{ 10, 7, 6 },
		{ 7, 1, 8 },
		{ 3, 9, 4 },
		{ 3, 4, 2 },
		{ 3, 2, 6 },
		{ 3, 6, 8 },
		{ 3, 8, 9 },
		{ 4, 9, 5 },
		{ 2, 4, 11 },
		{ 6, 2, 10 },
		{ 8, 6, 7 },
		{ 9, 8, 1 }
	};

	// Helper: subdivide a triangle and push its edges as line segments.
	// NOTE: We intentionally allow duplicated edges; this is a wireframe gizmo.
	auto subdivide = [&](const Vector3 &a, const Vector3 &b, const Vector3 &c, int depth, auto &&subdivide_ref) -> void {
		if (depth <= 0) {
			ret.push_back(Segment{ a, b });
			ret.push_back(Segment{ b, c });
			ret.push_back(Segment{ c, a });
			return;
		}
		Vector3 ab = (a + b) * (real_t)0.5;
		Vector3 bc = (b + c) * (real_t)0.5;
		Vector3 ca = (c + a) * (real_t)0.5;
		ab.normalize();
		bc.normalize();
		ca.normalize();
		subdivide_ref(a, ab, ca, depth - 1, subdivide_ref);
		subdivide_ref(b, bc, ab, depth - 1, subdivide_ref);
		subdivide_ref(c, ca, bc, depth - 1, subdivide_ref);
		subdivide_ref(ab, bc, ca, depth - 1, subdivide_ref);
	};

	for (int f = 0; f < 20; f++) {
		const Vector3 &a = v[faces[f][0]];
		const Vector3 &b = v[faces[f][1]];
		const Vector3 &c = v[faces[f][2]];
		subdivide(a, b, c, p_subdiv, subdivide);
	}

	return ret;
}

LocalVector<JointLimitationKusudama3D::Segment> JointLimitationKusudama3D::cull_lines_by_boundary(const LocalVector<Segment> &p_segments, LocalVector<Vector3> &r_crossed_points) const {
	LocalVector<Segment> ret;
	for (const Segment &seg : p_segments) {
		Vector3 from_solved;
		bool from_is_in_boundary = is_in_boundary(seg.first, from_solved);
		Vector3 to_solved;
		bool to_is_in_boundary = is_in_boundary(seg.second, to_solved);
		if (from_is_in_boundary && to_is_in_boundary) {
			continue;
		} else if (!from_is_in_boundary && !to_is_in_boundary) {
			ret.push_back(seg);
		} else {
			Segment new_seg;
			if (from_is_in_boundary) {
				new_seg.first = seg.second;
				new_seg.second = to_solved;
				r_crossed_points.push_back(to_solved);
			} else {
				new_seg.first = from_solved;
				new_seg.second = seg.first;
				r_crossed_points.push_back(from_solved);
			}
			ret.push_back(new_seg);
		}
	}
	return ret;
}

bool JointLimitationKusudama3D::is_in_boundary(const Vector3 &p_point, Vector3 &r_solved) const {
	// Return whether p_point is in boundary.
	r_solved = _solve(p_point);
	return r_solved.is_equal_approx(p_point);
}

LocalVector<Vector3> JointLimitationKusudama3D::sort_by_nearest_point(const LocalVector<Vector3> &p_points) const {
	LocalVector<Vector3> ret;
	LocalVector<Vector3> points = p_points;
	if (points.size() > 0) {
		ret.push_back(points[0]);
		points.remove_at(0);
		while (points.size() > 0) {
			uint32_t current = ret.size() - 1;
			int nearest_index = -1;
			double nearest = INFINITY;
			for (uint32_t i = 0; i < points.size(); i++) {
				double dist = ret[current].distance_squared_to(points[i]);
				if (dist < nearest) {
					nearest = dist;
					nearest_index = i;
				}
			}
			if (nearest_index >= 0) {
				ret.push_back(points[nearest_index]);
				points.remove_at(nearest_index);
			}
		}
	}
	return ret;
}

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color, int p_bone_index) const {
	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	// Draw subdivided icosahedron sphere.
	LocalVector<Segment> icosahedron_lines = get_icosahedron_sphere(3);
	LocalVector<Vector3> crossed_points;
	icosahedron_lines = cull_lines_by_boundary(icosahedron_lines, crossed_points);
	crossed_points = sort_by_nearest_point(crossed_points);

	p_surface_tool->set_color(p_color);
	for (const Segment &seg : icosahedron_lines) {
		p_surface_tool->add_vertex(p_transform.xform(seg.first * sphere_r));
		p_surface_tool->add_vertex(p_transform.xform(seg.second * sphere_r));
	}
	p_surface_tool->set_color(Color(1.0, 0.0, 0.0, 1.0));
	for (uint32_t i = 0; i < crossed_points.size(); i++) {
		p_surface_tool->add_vertex(p_transform.xform(crossed_points[i] * sphere_r));
		p_surface_tool->add_vertex(p_transform.xform(crossed_points[(i + 1) % crossed_points.size()] * sphere_r));
	}
}
#endif // TOOLS_ENABLED
