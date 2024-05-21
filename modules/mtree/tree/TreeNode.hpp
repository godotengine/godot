#pragma once
#include <vector>
#include <memory>
#include "core/math/vector3.h"
#include "GrowthInfo.hpp" 

namespace Mtree
{

	struct NodeChild;

	class TreeNode
	{
	public:
		std::vector<std::shared_ptr<NodeChild>> children;
		Vector3 direction;
		Vector3 tangent;
		float length;
		float radius;
		int creator_id = 0;
		std::unique_ptr<GrowthInfo> growthInfo = nullptr;

		bool is_leaf() const;

		TreeNode(Vector3 direction, Vector3 parent_tangent, float length, float radius, int creator_id);
	};

	struct NodeChild 
	{
		TreeNode node;
		float position_in_parent;
	};

	struct Stem
	{
		TreeNode node;
		Vector3 position;
	};
}