#pragma once
#include<vector>
#include "TreeNode.hpp"
#include "modules/game_help/mtree/tree_functions/base_types/TreeFunction.hpp"

namespace Mtree
{
	class Tree
	{
	private:
		std::vector<Stem> stems;
		std::shared_ptr<TreeFunction> firstFunction;
	public:
		Tree(std::shared_ptr<TreeFunction> trunkFunction);
		Tree() { firstFunction = nullptr; };
		void set_first_function(std::shared_ptr<TreeFunction> function);
		void execute_functions();
		void print_tree();
		TreeFunction& get_first_function();
		std::vector<Stem>& get_stems();
	};
}