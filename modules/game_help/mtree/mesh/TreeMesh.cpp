#include "TreeMesh.hpp"

namespace Mtree
{
	std::vector<std::vector<float>> TreeMesh::get_vertices()
	{
		auto result = std::vector<std::vector<float>>();
		for (Eigen::Vector3f& vert : this->vertices)
		{
			result.push_back(std::vector<float>{vert[0], vert[1], vert[2]});
		}
		return result;
	}

	int TreeMesh::add_vertex(const Eigen::Vector3f& position)
	{
		vertices.push_back(position);
		for (auto& attribute : attributes)
		{
			attribute.second->add_data();
		}
		return (int)vertices.size() - 1;
	}
	int TreeMesh::add_polygon()
	{
		polygons.emplace_back();
		uv_loops.emplace_back();
		return  (int)polygons.size() - 1;
	}
}