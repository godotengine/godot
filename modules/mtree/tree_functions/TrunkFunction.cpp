#include "TrunkFunction.hpp"
#include "../utilities/GeometryUtilities.hpp"
namespace Mtree
{
	void TrunkFunction::execute(std::vector<Stem>& stems, int id, int parent_id)
	{
		rand_gen.set_seed(seed);

		float segment_length = 1 / (resolution + .001);
		Vector3 stem_direction = Vector3{ 0,0,1 };
		TreeNode firstNode{ stem_direction, Geometry::get_orthogonal_vector(stem_direction), segment_length, start_radius, id };
		TreeNode* current_node = &firstNode;
		int node_count = resolution * length;
		for (size_t i = 0; i < node_count; i++)
		{
			if (i == node_count - 1)
				segment_length = length - i*segment_length;
			
			float factor = std::pow((float)i / (node_count), shape);
			float radius = Geometry::lerp(start_radius, end_radius, factor);
			Vector3 direction = current_node->direction + Geometry::random_vec() * (randomness / (resolution+.001f));
			direction += Vector3(0,0,up_attraction / (resolution + .001f));
			direction.normalize();
			float position_in_parent = 1;
			NodeChild child{ TreeNode{direction, firstNode.tangent, segment_length, radius, id}, position_in_parent };
			current_node->children.push_back(std::make_shared<NodeChild>(std::move(child)));
			current_node = &current_node->children.back()->node;
		}

		Vector3 position{ 0,0,0 };
		Stem stem{ std::move(firstNode), position };
		stems.push_back(std::move(stem));
		execute_children(stems, id);
	}

}
