#pragma once
#include "modules/game_help/mtree/mesh/TreeMesh.hpp"
#include "modules/game_help/mtree/tree/Tree.hpp"

namespace Mtree
{
	class TreeMesher
	{
	public:
		virtual TreeMesh mesh_tree(Tree& tree) = 0;
	};
}
