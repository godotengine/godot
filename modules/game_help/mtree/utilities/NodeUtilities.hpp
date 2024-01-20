#pragma once
#include "../tree/TreeNode.hpp"


namespace Mtree {
	namespace NodeUtilities {

		struct NodeSelectionElement
		{
			TreeNode* node;
			Eigen::Vector3f node_position;
			NodeSelectionElement(TreeNode& node, const Eigen::Vector3f& position) : node(&node), node_position(position) {};
		};

		using NodeSelection = std::vector<NodeSelectionElement>;
		using BranchSelection = std::vector<NodeSelection>;


		float get_branch_length(TreeNode& branch_origin);
		BranchSelection select_from_tree(std::vector<Stem>& stems, int id);
		Eigen::Vector3f get_position_in_node(const Eigen::Vector3f& node_position, const TreeNode& node, const float factor);

		
	}
}