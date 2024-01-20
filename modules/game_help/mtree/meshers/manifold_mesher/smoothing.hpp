#pragma once
#include "modules/game_help/mtree/mesh/TreeMesh.hpp"

namespace Mtree::MeshProcessing::Smoothing
{
    void smooth_mesh(TreeMesh& mesh, const int iterations, const float factor, std::vector<float>* weights = nullptr);
}