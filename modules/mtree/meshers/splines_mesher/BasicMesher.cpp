#include <iostream>
#include <queue>
#include "BasicMesher.hpp"
#include "../../utilities/GeometryUtilities.hpp"


namespace Mtree
{
	std::vector<std::vector<BasicMesher::SplinePoint>> BasicMesher::get_splines(std::vector<Stem>& stems)
	{
		std::vector<std::vector<SplinePoint>> splines;
		
		for (Stem& stem : stems)
		{
			TreeNode* stem_node = &stem.node;
			Vector3  stem_position = stem.position;
			splines.push_back(std::vector<SplinePoint>{});
			get_splines_rec(splines, stem_node, stem_position);
		}
		return splines;
	}

	void BasicMesher::get_splines_rec(std::vector<std::vector<SplinePoint>>& splines, TreeNode* current_node, Vector3 current_position)
	{
		splines.back().push_back(SplinePoint{ current_position, current_node->direction, current_node->radius });
		if (current_node->children.size() == 0)
		{
			splines.back().push_back(SplinePoint{ current_position + current_node->direction * current_node->length, current_node->direction, current_node->radius });
			return;
		}
		for (size_t i = 0; i < current_node->children.size(); i++)
		{
			NodeChild& child = *current_node->children[i];
			Vector3 child_position = current_position + current_node->direction * current_node->length * child.position_in_parent;
			TreeNode* child_node = &child.node;
			if (i > 0)
				splines.push_back(std::vector<SplinePoint>{});
			get_splines_rec(splines, child_node, child_position);
		}
	}

	void BasicMesher::mesh_spline(TreeMesh& mesh, std::vector<SplinePoint>& spline)
	{
		for (SplinePoint& spline_point : spline)
		{
			int n = mesh.vertices.size();
			Geometry::add_circle(mesh.vertices, spline_point.position, spline_point.direction, spline_point.radius, radial_resolution);
			if (&spline_point == &spline.back())
				continue;
			for (int i = 0; i < radial_resolution; i++)
			{
				int polygon_index = mesh.add_polygon();
				mesh.polygons[polygon_index] = { 
					n + i,
					n + radial_resolution + i,
					n + radial_resolution + (i + 1) % radial_resolution,
					n + (i + 1) % radial_resolution };
			}
		}	
	}

	TreeMesh BasicMesher::mesh_tree(Tree& tree)
	{
		std::vector<Stem>& tree_stems = tree.get_stems();

		std::vector<std::vector<SplinePoint>> splines = get_splines(tree_stems);

		TreeMesh mesh;
		for (std::vector<SplinePoint>& spline : splines)
		{
			mesh_spline(mesh, spline);
		}

		return mesh;
	}
}

