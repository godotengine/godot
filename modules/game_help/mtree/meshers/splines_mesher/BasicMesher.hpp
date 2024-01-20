#pragma once
#include "../base_types/TreeMesher.hpp"

namespace Mtree
{


	class BasicMesher : public TreeMesher
	{
	private:
		struct SplinePoint 
		{
			Eigen::Vector3f position;
			Eigen::Vector3f direction;
			float radius;
		};
		std::vector<std::vector<SplinePoint>> get_splines(std::vector<Stem>& stems);
		void get_splines_rec(std::vector<std::vector<SplinePoint>>& splines, TreeNode* current_node, Eigen::Vector3f current_position);
		void mesh_spline(TreeMesh& mesh, std::vector<SplinePoint>& spline);

	public:
		int radial_resolution = 8;
		TreeMesh mesh_tree(TreeMesh& tree) override;
	};


}