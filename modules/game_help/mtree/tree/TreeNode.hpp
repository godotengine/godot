#pragma once
#include <vector>
#include <memory>
#include <Eigen/Core>
#include "GrowthInfo.hpp"

namespace Mtree
{

	struct NodeChild;

	class TreeNode
	{
	public:
		std::vector<std::shared_ptr<NodeChild>> children;
		Eigen::Vector3f direction;
		Eigen::Vector3f tangent;
		float length;
		float radius;
		int creator_id = 0;
		std::unique_ptr<GrowthInfo> growthInfo = nullptr;

		bool is_leaf() const;

		TreeNode(Eigen::Vector3f direction, Eigen::Vector3f parent_tangent, float length, float radius, int creator_id);
	};

	struct NodeChild 
	{
		TreeNode node;
		float position_in_parent;
	};

	struct Stem
	{
		TreeNode node;
		Eigen::Vector3f position;
	};
}