#pragma once
#include "../../mesh/TreeMesh.hpp"
#include "../../tree/Tree.hpp"

namespace Mtree
{
	class TreeMesher
	{
	public:
		virtual TreeMesh mesh_tree(Tree& tree) = 0;
	};
}
