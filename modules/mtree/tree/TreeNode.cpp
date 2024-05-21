#include "TreeNode.hpp"
#include "../utilities/GeometryUtilities.hpp"


bool Mtree::TreeNode::is_leaf() const
{
	return children.size() == 0;
}

Mtree::TreeNode::TreeNode(Vector3 direction, Vector3 parent_tangent, float length, float radius, int creator_id)
{
	this->direction = direction;
	this->tangent = Geometry::projected_on_plane(parent_tangent, direction).normalized();
	this->length = length;
	this->radius = radius;
	this->creator_id = creator_id;
}
