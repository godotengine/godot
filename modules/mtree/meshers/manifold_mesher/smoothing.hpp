#pragma once
#include "../../mesh/TreeMesh.hpp"

namespace Mtree::MeshProcessing::Smoothing
{
    void smooth_mesh(TreeMesh& mesh, const int iterations, const float factor, std::vector<float>* weights = nullptr);
}