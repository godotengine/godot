#include <iostream>
#include <queue>
#include "NodeUtilities.hpp"

namespace Mtree
{
	namespace NodeUtilities
	{
		float get_branch_length(TreeNode& branch_origin)
		{
			float length = 0;
			TreeNode* extremity = &branch_origin;
			while (extremity->children.size() > 0)
			{
				length += extremity->length;
				extremity = & extremity->children[0]->node;
			}
			length += extremity->length;
			return length;
		}

		void select_from_tree_rec(BranchSelection& selection, TreeNode& node, const Vector3& node_position, int id)
		{
			if (node.creator_id == id)
			{
				selection[selection.size() - 1].push_back(NodeSelectionElement{node, node_position});
			}
			bool first_child = true;
			for (auto& child : node.children)
			{
				if (!first_child)
				{
					selection.emplace_back();
				}
				first_child = false;
				Vector3 child_position = node_position + child->node.direction * child->position_in_parent * child->node.length;
				select_from_tree_rec(selection, child->node, child_position, id);
			}
		}

		BranchSelection select_from_tree(std::vector<Stem>& stems, int id)
		{
			BranchSelection selection;
			selection.emplace_back();
			for (Stem& stem : stems)
			{
				select_from_tree_rec(selection, stem.node, stem.position, id);
			}
			return selection;
		}


		Vector3 get_position_in_node(const Vector3& node_position, const TreeNode& node, const float factor)
		{
			return node_position + node.direction * node.length;
		};
	}
}
