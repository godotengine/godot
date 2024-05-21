#pragma once
#include "../base_types/TreeMesher.hpp"

namespace Mtree
{


	class BasicMesher : public TreeMesher
	{
	private:
		struct SplinePoint 
		{
			Vector3 position;
			Vector3 direction;
			float radius;
		};
		std::vector<std::vector<SplinePoint>> get_splines(std::vector<Stem>& stems);
		void get_splines_rec(std::vector<std::vector<SplinePoint>>& splines, TreeNode* current_node, Vector3 current_position);
		void mesh_spline(TreeMesh& mesh, std::vector<SplinePoint>& spline);

	public:
		int radial_resolution = 8;
		TreeMesh mesh_tree(Tree& tree) override;
	};


}