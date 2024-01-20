#pragma once
#include <vector>
#include <array>
#include <map>
#include <Eigen/Core>
#include <memory>
#include "Attribute.hpp"

// #include<pybind11/pybind11.h>
// #include<pybind11/numpy.h>

namespace Mtree
{
	
	// namespace py = pybind11;

	class TreeMesh
	{
	public:
		std::vector<Eigen::Vector3f> vertices;
		std::vector<Eigen::Vector3f> normals;
		std::vector<Eigen::Vector2f> uvs;
		std::vector<std::array<int, 4>> polygons;
		std::vector<std::array<int, 4>> uv_loops;
		std::map<std::string, std::shared_ptr<AbstractAttribute>> attributes;

		TreeMesh() {};
		TreeMesh(std::vector<Eigen::Vector3f>&& vertices) { this->vertices = std::move(vertices); }
		std::vector<std::vector<float>> get_vertices();
		std::vector<std::array<int, 4>> get_polygons() { return this->polygons; };
		int add_vertex(const Eigen::Vector3f& position);
		int add_polygon();
		template <class T>
		Attribute<T>& add_attribute(std::string name)
		{
			auto attribute = std::make_shared<Attribute<T>>(name);
			attributes[name] = attribute;
			return *attribute;
		};
	};
}