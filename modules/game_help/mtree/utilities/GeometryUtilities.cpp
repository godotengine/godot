#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <Eigen/Geometry>
#include "GeometryUtilities.hpp"

namespace Mtree {
	namespace Geometry {

		void add_circle(std::vector<Eigen::Vector3f>& points, Eigen::Vector3f position, Eigen::Vector3f direction, float radius, int n_points)
		{
			Eigen::Matrix3f rot;
			// rot = Eigen::AngleAxis<float>( angle, axis );
			rot = get_look_at_rot(direction);

			for (size_t i = 0; i < n_points; i++)
			{
				float circle_angle = M_PI * (float)i / n_points * 2;
				Eigen::Vector3f position_in_circle = Eigen::Vector3f{ std::cos(circle_angle), std::sin(circle_angle),0 } *radius;
				position_in_circle = position + rot * position_in_circle;
				points.push_back(position_in_circle);
			}
		}

		Eigen::Matrix3f get_look_at_rot(Eigen::Vector3f direction)
		{
			Eigen::Vector3f up{ 0,0,1 };
			Eigen::Vector3f axis = up.cross(direction);
			float sin = axis.norm();
			float cos = up.dot(direction);
			float angle =  std::atan2(sin, cos);
			if (angle < .01)
				axis = up;
			else
				axis /= sin;
			Eigen::Matrix3f rot;
			rot = Eigen::AngleAxis<float>(angle, axis);
			return rot;
		}

		Eigen::Vector3f random_vec_on_unit_sphere()
		{
			auto vec = Eigen::Vector3f{};
			vec.setRandom();
			vec.normalize();
			return vec;
		}

		Eigen::Vector3f random_vec(float flatness)
		{
			auto vec = Eigen::Vector3f{};
			vec.setRandom();
			vec.z() *= (1 - flatness);
			return vec;
		}

		Eigen::Vector3f lerp(Eigen::Vector3f a, Eigen::Vector3f b, float t)
		{
			return t * b + (1 - t) * a;
		}

		float lerp(float a, float b, float t)
		{
			t = std::clamp(t, 0.f, 1.f);
			return t * b + (1 - t) * a;
		}


		Eigen::Vector3f get_orthogonal_vector(const Eigen::Vector3f& v)
		{
			Eigen::Vector3f tmp;
			if (abs(v.z()) <  .95)
			{
				tmp = Eigen::Vector3f{1,0,0};
			}
			else
			{
				tmp = Eigen::Vector3f{0,1,0};
			}
			return tmp.cross(v).normalized();
		}

		void project_on_plane(Eigen::Vector3f& v, const Eigen::Vector3f& plane_normal)
		{
			Eigen::Vector3f offset = v.dot(plane_normal) * plane_normal;
			v -= offset;
		}
		
		Eigen::Vector3f projected_on_plane(const Eigen::Vector3f& v, const Eigen::Vector3f& plane_normal)
		{
			auto result = v;
			project_on_plane(result, plane_normal);
			return result;
		}
	}
}
