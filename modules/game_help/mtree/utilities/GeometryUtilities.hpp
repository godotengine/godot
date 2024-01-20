#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif


namespace Mtree { namespace Geometry
{

	void add_circle(std::vector<Eigen::Vector3f>& points, Eigen::Vector3f position, Eigen::Vector3f direction, float radius, int n_points);

	Eigen::Matrix3f get_look_at_rot(Eigen::Vector3f direction);

	Eigen::Vector3f random_vec_on_unit_sphere();
	
	Eigen::Vector3f random_vec(float flatness=0);

	Eigen::Vector3f get_orthogonal_vector(const Eigen::Vector3f& v);

	void project_on_plane(Eigen::Vector3f& v, const Eigen::Vector3f& plane_normal);

	Eigen::Vector3f projected_on_plane(const Eigen::Vector3f& v, const Eigen::Vector3f& plane_normal);	

	Eigen::Vector3f lerp(Eigen::Vector3f a, Eigen::Vector3f b, float t);

	float lerp(float a, float b, float t);
}}