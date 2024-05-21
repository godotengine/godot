#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include "GeometryUtilities.hpp"
#include "core/math/vector3.h"
#include "core/math/math_funcs.h"

namespace Mtree {
	namespace Geometry {

		void add_circle(std::vector<Vector3>& points, Vector3 position, Vector3 direction, float radius, int n_points)
		{
			Basis rot;
			// rot = Eigen::AngleAxis<float>( angle, axis );
			rot = get_look_at_rot(direction);

			for (size_t i = 0; i < n_points; i++)
			{
				float circle_angle = M_PI * (float)i / n_points * 2;
				Vector3 position_in_circle = Vector3{ std::cos(circle_angle), std::sin(circle_angle),0 } *radius;
				position_in_circle = position + rot.xform(position_in_circle);
				points.push_back(position_in_circle);
			}
		}

		Basis get_look_at_rot(Vector3 direction)
		{
			Vector3 up{ 0,0,1 };
			Vector3 axis = up.cross(direction);
			float sin = axis.length_squared();
			axis.normalize();
			float cos = up.dot(direction);
			float angle =  std::atan2(sin, cos);
			if (angle < .01)
				axis = up;
			else
				axis /= sin;
			Basis rot;
			rot = Basis(axis,angle );
			return rot;
		}

		Vector3 random_vec_on_unit_sphere()
		{
			auto vec = Vector3{Math::random(-1.0f, 1.0f), Math::random(-1.0f, 1.0f), Math::random(-1.0f, 1.0f)};
			vec.normalize();
			return vec;
		}

		Vector3 random_vec(float flatness)
		{
			auto vec = Vector3{Math::random(-1.0f, 1.0f), Math::random(-1.0f, 1.0f), Math::random(-1.0f, 1.0f)};
			//vec.setRandom();
			vec.z *= (1 - flatness);
			return vec;
		}

		Vector3 lerp(Vector3 a, Vector3 b, float t)
		{
			return t * b + (1 - t) * a;
		}

		float lerp(float a, float b, float t)
		{
			t = CLAMP(t, 0.f, 1.f);
			return t * b + (1 - t) * a;
		}


		Vector3 get_orthogonal_vector(const Vector3& v)
		{
			Vector3 tmp;
			if (abs(v.z) <  0.95)
			{
				tmp = Vector3{1,0,0};
			}
			else
			{
				tmp = Vector3{0,1,0};
			}
			return tmp.cross(v).normalized();
		}

		void project_on_plane(Vector3& v, const Vector3& plane_normal)
		{
			Vector3 offset = v.dot(plane_normal) * plane_normal;
			v -= offset;
		}
		
		Vector3 projected_on_plane(const Vector3& v, const Vector3& plane_normal)
		{
			auto result = v;
			project_on_plane(result, plane_normal);
			return result;
		}
	}
}
