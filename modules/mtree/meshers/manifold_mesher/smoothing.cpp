#include "smoothing.hpp"
#include "../../utilities/GeometryUtilities.hpp"

namespace Mtree::MeshProcessing::Smoothing
{
    void add_index_no_duplicates(std::vector<int>& indices, const int index)
    {
        for (auto i : indices)
        {
            if (index == i)
            {
                return;
            }
        }
        indices.push_back(index);
    }


    std::vector<std::vector<int>> get_neighbourhoods(Mtree::TreeMesh& mesh)
    {
        std::vector<std::vector<int>> vertex_neighbourhood;
        vertex_neighbourhood.resize(mesh.vertices.size());
        for (auto& polygon : mesh.polygons)
        {
            for (int i = 1; i < polygon.size(); i+=1)
            {
                int i1 = polygon[i];
                int i2 = polygon[(i + 1) % polygon.size()];
                add_index_no_duplicates(vertex_neighbourhood[i1], i2);
                add_index_no_duplicates(vertex_neighbourhood[i2], i1);
            }
        }
        return vertex_neighbourhood;
    }

    void smooth_mesh_once(std::vector<Vector3>* result, const std::vector<Vector3>* previous_iteration, const std::vector<std::vector<int>>& neighbourhoods, float factor, std::vector<float>* weights=nullptr)
    {
        for (size_t i = 0; i < result->size(); i++)
        {
            if (neighbourhoods[i].size() <= 1)
            {
                continue;
            }
            Vector3 barycenter{ 0,0,0 };
            for (auto neighbour : neighbourhoods[i])
            {
                barycenter += previous_iteration->at(neighbour);
            }
            barycenter /= neighbourhoods[i].size();
            float true_factor = factor;
            if (weights != nullptr)
            {
                true_factor *= (*weights)[i];
            }
            result->at(i) = Geometry::lerp(previous_iteration->at(i), barycenter, true_factor);
        }
    }

    void smooth_mesh(TreeMesh& mesh, const int iterations, const float factor, std::vector<float>* weights)
    {
        auto neighbourhoods = get_neighbourhoods(mesh);
        std::vector<Vector3>* previous_iteration = &mesh.vertices;
        std::vector<Vector3> buffer = mesh.vertices;
        std::vector<Vector3>* result = &buffer;

        for (size_t i = 0; i < iterations; i++)
        {
            smooth_mesh_once(result, previous_iteration, neighbourhoods, factor, weights);
            auto tmp = result;
            result = previous_iteration;
            previous_iteration = tmp;
        }
        if (result != &mesh.vertices)
        {
            mesh.vertices = std::move(buffer);
        }
    }
}