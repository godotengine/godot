#pragma once
#include <vector>
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


namespace Mtree { namespace Geometry
{

	void add_circle(std::vector<Vector3>& points, Vector3 position, Vector3 direction, float radius, int n_points);

	Basis get_look_at_rot(Vector3 direction);

	Vector3 random_vec_on_unit_sphere();
	
	Vector3 random_vec(float flatness=0);

	Vector3 get_orthogonal_vector(const Vector3& v);

	void project_on_plane(Vector3& v, const Vector3& plane_normal);

	Vector3 projected_on_plane(const Vector3& v, const Vector3& plane_normal);	

	Vector3 lerp(Vector3 a, Vector3 b, float t);

	float lerp(float a, float b, float t);
}}