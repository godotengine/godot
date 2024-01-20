#include "TreeFunction.hpp"

namespace Mtree
{
	void TreeFunction::execute_children(std::vector<Stem>& stems, int id)
	{
		int child_id = id;
		for (std::shared_ptr<TreeFunction>& child : children)
		{
			child_id++;
			child->execute(stems, child_id, id);
		}
	}
	void TreeFunction::add_child(std::shared_ptr<TreeFunction> child)
	{
		children.push_back(child);
	}
	
}