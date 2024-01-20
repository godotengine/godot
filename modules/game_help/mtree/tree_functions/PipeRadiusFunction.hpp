#pragma once
#include <vector>
// #include <queue>
#include "./base_types/TreeFunction.hpp"
// #include "modules/game_help/mtree/utilities/NodeUtilities.hpp"
// #include "modules/game_help/mtree/utilities/GeometryUtilities.hpp"
#include "modules/game_help/mtree/tree_functions/base_types/Property.hpp"


namespace Mtree
{
	class PipeRadiusFunction : public TreeFunction
	{
	private:

        void update_radius_rec(TreeNode& node);

	public:
		float power = 2.f;
		float end_radius = .01f;
		float constant_growth = .01f;
		void execute(std::vector<Stem>& stems, int id, int parent_id) override;
	};

}
