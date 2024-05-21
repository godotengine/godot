#include <iostream>
#include <algorithm>
#include "../../utilities/GeometryUtilities.hpp"
#include "../../utilities/NodeUtilities.hpp"
#include "ManifoldMesher.hpp"
#include "smoothing.hpp"
#include "core/math/vector2.h"

using namespace Mtree;
using namespace Mtree::NodeUtilities;

using AttributeNames = ManifoldMesher::AttributeNames;

namespace 
{
    struct CircleDesignator
    {
        int vertex_index;
        int uv_index;
        int radial_n;
    };
    struct IndexRange
    {
        int min_index;
        int max_index;
    };

    static float get_smooth_amount(const float radius, const float node_length)
    {
        return std::min(1.f, radius / node_length);
    }

    static CircleDesignator add_circle(const Vector3& node_position, const TreeNode& node, float factor, const int radial_n_points, TreeMesh& mesh, const float uv_y)
    {
        const Vector3& right = node.tangent;
        int vertex_index = mesh.vertices.size();
        int uv_index = mesh.uvs.size();
        Vector3 up = node.tangent.cross(node.direction);
        Vector3 circle_position = node_position + node.length * factor * node.direction;
        float radius = node.is_leaf() ? node.radius : Geometry::lerp(node.radius, node.children[0]->node.radius, factor);
        auto& smooth_attr = *static_cast<Attribute<float>*> (mesh.attributes[AttributeNames::smooth_amount].get());
        auto& radius_attr = *static_cast<Attribute<float>*> (mesh.attributes[AttributeNames::radius].get());
        float smooth_amount = get_smooth_amount(radius, node.length);
        auto& direction_attr = *static_cast<Attribute<Vector3>*> (mesh.attributes[AttributeNames::direction].get());
        for (size_t i = 0; i < radial_n_points; i++)
        {
            float angle = (float)i / radial_n_points * 2 * M_PI;
            Vector3 point = cos(angle) * right + sin(angle) * up;
            point = point * radius + circle_position;
            int index = mesh.add_vertex(point);
            smooth_attr.data[index] = smooth_amount;
            radius_attr.data[index] = radius;
            direction_attr.data[index] = node.direction;
            mesh.uvs.emplace_back((float)i / radial_n_points, uv_y);
        }
        mesh.uvs.emplace_back(1, uv_y);
        return CircleDesignator{vertex_index, uv_index, radial_n_points};
    }
    
    static bool is_index_in_branch_mask(const std::vector<IndexRange>& mask, const int index, const int radial_n_points)
    {
        int offset = radial_n_points / 2;
        for (auto range : mask)
        {
            int i = index;
            if (range.max_index < range.min_index)
            {
                i = (i + offset) % radial_n_points;
                range.min_index = (range.min_index + offset) % radial_n_points;
                range.max_index = (range.max_index + offset) % radial_n_points;
            }
            if (i >= range.min_index && i < range.max_index)
            {
                return true;
            }
        }
        return false;
    }
    
    static void bridge_circles(const CircleDesignator& first_circle, const CircleDesignator& second_circle, const int radial_n_points, TreeMesh& mesh, std::vector<IndexRange>* mask = nullptr)
    {
        for (int i = 0; i < radial_n_points; i++)
        {
            if (mask != nullptr && is_index_in_branch_mask(*mask, i, radial_n_points))
            {
                continue;
            }
            int polygon_index = mesh.add_polygon();
            mesh.polygons[polygon_index] = 
            {
                first_circle.vertex_index + i,
                first_circle.vertex_index + (i + 1) % radial_n_points,
                second_circle.vertex_index + (i + 1) % radial_n_points,
                second_circle.vertex_index + i
            };
            mesh.uv_loops[polygon_index] = 
            { // no need for modulo since a circle with n points has n differnt 3d coordinates but n+1 different uv coordinates
                first_circle.uv_index + i,
                first_circle.uv_index + (i + 1),
                second_circle.uv_index + (i + 1),
                second_circle.uv_index + i 
            };
        }
    }

    static float get_branch_angle_around_parent(const TreeNode& parent, const TreeNode& branch)
    {
        Vector3 projected_branch_dir = Geometry::projected_on_plane(branch.direction, parent.direction).normalized();
        auto& right = parent.tangent;
        Vector3 up = right.cross(parent.direction);
        float cos_angle = projected_branch_dir.dot(right);
        float sin_angle = projected_branch_dir.dot(up);
        return std::fmod(std::atan2(sin_angle, cos_angle) + 2*M_PI, 2*M_PI);
    }
    
    static IndexRange get_branch_indices_on_circle(const int radial_n_points, const float circle_radius, const float branch_radius, const float branch_angle)
    {
        float angle_delta = std::asin(std::clamp(branch_radius / circle_radius, -1.f, 1.f));
        float increment = 2 * M_PI / radial_n_points;
        int min_index = (int)(std::fmod(branch_angle - angle_delta + 2*M_PI, 2 * M_PI) / increment);
        int max_index = (int)(std::fmod(branch_angle + angle_delta + increment + 2*M_PI, 2 * M_PI) / increment);
        return IndexRange{ min_index, max_index };
    }
    
    static std::vector<IndexRange> get_children_ranges(const TreeNode& node, const int radial_n_points)
    {
        std::vector<IndexRange> ranges;
        for (size_t i = 1; i < node.children.size(); i++)
        {
            auto& child = node.children[i];
            float angle = get_branch_angle_around_parent(node, child->node);
            IndexRange range = get_branch_indices_on_circle(radial_n_points, node.radius, child->node.radius, angle);

            ranges.push_back(range);
        }
        return ranges;
    }
    
    static std::vector<int> get_child_index_order(const CircleDesignator& parent_base, const int child_radial_n, const IndexRange child_range, const NodeChild& child, const TreeNode& parent, const TreeMesh& mesh)
    {
        int start = child_range.min_index + parent_base.vertex_index;
        std::vector<int> child_base_indices;
        child_base_indices.resize((size_t)child_radial_n);

        for (int i = 0; i < child_radial_n / 2; i++)
        {
            int lower_index = (child_range.min_index + i) % parent_base.radial_n + parent_base.vertex_index;
            int upper_index = lower_index + parent_base.radial_n;
            int vertex_index = start + (i % (child_radial_n / 2));

            child_base_indices[i] = lower_index;
            child_base_indices[(size_t)child_radial_n - i - 1] = upper_index;
        }
        return child_base_indices;
    }
    
    static void add_child_base_geometry(const std::vector<int>& child_base_indices, const CircleDesignator& child_base, const float child_radius, const Vector3& child_pos, const int offset, const float smooth_amount, TreeMesh& mesh)
    {
        auto& smooth_attr = *static_cast<Attribute<float>*> (mesh.attributes[ManifoldMesher::AttributeNames::smooth_amount].get());
        auto& radius_attr = *static_cast<Attribute<float>*> (mesh.attributes[ManifoldMesher::AttributeNames::radius].get());
        auto& direction_attr = *static_cast<Attribute<Vector3>*> (mesh.attributes[ManifoldMesher::AttributeNames::direction].get());

        Vector3 direction = (mesh.vertices[child_base_indices[2]] - mesh.vertices[child_base_indices[0]]).cross(mesh.vertices[child_base_indices[1]] - mesh.vertices[child_base_indices[0]]).normalized();
        
        Vector3 child_base_center{ 0,0,0 };
        for (auto& i : child_base_indices)
            child_base_center += mesh.vertices[(size_t)i];
        child_base_center /= child_base_indices.size();
        
        for (int i = 0; i < child_base.radial_n; i++)
        {
            int index = (i + offset) % child_base.radial_n;
            Vector3 vertex = mesh.vertices[child_base_indices[(size_t)index]];
            vertex = (vertex - child_base_center).normalized() * child_radius + child_pos;
            int added_vertex_index = mesh.add_vertex(vertex);
            smooth_attr.data[added_vertex_index] = smooth_amount;
            radius_attr.data[added_vertex_index] = child_radius;
            direction_attr.data[added_vertex_index] = direction;

            int polygon_index = mesh.add_polygon();
            mesh.polygons[polygon_index] =
            {
                child_base_indices[index],
                child_base_indices[(index + 1) % child_base.radial_n],
                child_base.vertex_index + (i + 1) % child_base.radial_n,
                child_base.vertex_index + i
            };
            int uv_start = child_base.uv_index - child_base.radial_n*2;
            mesh.uv_loops[polygon_index] = 
            { 
                uv_start + index,
                uv_start + (index + 1)%child_base.radial_n,
                uv_start + child_base.radial_n + (index + 1)% child_base.radial_n,
                uv_start + child_base.radial_n + index
            };
        }
    }
    
    static float get_child_twist(const TreeNode& child, const TreeNode& parent)
    {
        Vector3 projected_parent_dir = Geometry::projected_on_plane(parent.direction, child.direction).normalized();
        auto& right = projected_parent_dir;
        Vector3 up = right.cross(child.direction);
        float cos_angle = child.tangent.dot(right);
        float sin_angle = child.tangent.dot(up);
        return std::fmod(std::atan2(sin_angle, cos_angle) + 2 * M_PI, 2 * M_PI);
    }

    static int add_child_base_uvs(float parent_uv_y, const TreeNode& parent, const NodeChild& child, const IndexRange child_range, const int child_radial_n, const int parent_radial_n, TreeMesh& mesh)
    {
        float uv_growth = parent.length / (parent.radius + .001f) / (2 * M_PI);
        for (size_t i = 0; i < 2; i++) // recreating outer uvs (but without continuous (no looping back to x=0)
        {
            float uv_y = parent_uv_y + i * uv_growth;
            float x_start = child_range.min_index + i * (child_radial_n / 2.f - 1);
            float step = i == 0 ? 1 : -1;
            for (size_t j = 0; j < child_radial_n / 2; j++)
            {
                float uv_x = (x_start + j*step) / parent_radial_n;
                mesh.uvs.emplace_back(uv_x, uv_y);
            }
        }

        Vector2 uv_circle_center{ (child_range.min_index + (child_radial_n / 4.f - .5f))/parent_radial_n, parent_uv_y + uv_growth / 2 };
        float uv_circle_radius = std::min((float)child_radial_n / parent_radial_n, uv_growth/2) * .6f;
        for (size_t i = 0; i < child_radial_n; i++) // inner uvs
        {
            float angle = (float)i / (child_radial_n -1) * 2 * M_PI + M_PI;
            Vector2 uv_position = Vector2{ cos(angle), sin(angle) } *uv_circle_radius + uv_circle_center;
            mesh.uvs.push_back(uv_position);
        }
        int circle_uv_start_index = mesh.uvs.size();

        for (int i = 0; i < child_radial_n; i++)
        {
            mesh.uvs.emplace_back((float)i / child_radial_n, parent_uv_y);
        }
        mesh.uvs.emplace_back(1, parent_uv_y);

        return circle_uv_start_index;
    }

    static CircleDesignator add_child_circle(const TreeNode& parent, const NodeChild& child, const Vector3& child_pos, const Vector3& parent_pos, const CircleDesignator& parent_base, const IndexRange child_range, const float uv_y, TreeMesh& mesh)
    {
        float smooth_amount = get_smooth_amount(child.node.radius, parent.length);
        
        int child_radial_n = 2 * ((child_range.max_index - child_range.min_index + parent_base.radial_n) % parent_base.radial_n + 1); // number of vertices in child circle
        std::vector<int> child_base_indices = get_child_index_order(parent_base, child_radial_n, child_range, child, parent, mesh);
        
        float child_twist = get_child_twist(child.node, parent);
        int offset = (int)(child_twist / (2 * M_PI) * child_radial_n - child_radial_n / 4 + child_radial_n) % child_radial_n;

        CircleDesignator child_base{ (int)mesh.vertices.size(), (int)mesh.uvs.size(), child_radial_n };
        child_base.uv_index = add_child_base_uvs(uv_y, parent, child, child_range, child_radial_n, parent_base.radial_n, mesh);
        add_child_base_geometry(child_base_indices, child_base, child.node.radius, child_pos, offset, smooth_amount, mesh);
        return child_base;
    }
    
    static Vector3 get_side_child_position(const TreeNode& parent, const NodeChild& child, const Vector3& node_position)
    {
        Vector3 tangent = Geometry::projected_on_plane(child.node.direction, parent.direction).normalized();
        return node_position + parent.direction * parent.length * child.position_in_parent + tangent * parent.radius;
    }

    static bool has_side_branches(const TreeNode& node)
    {
        if (node.children.size() < 2)
            return false;

        for (int i = 1; i < node.children.size(); i++)
        {
            if (node.children[i]->node.children.size() > 0)
                return true;
        }
    }
    
    static void mesh_node_rec(const TreeNode& node, const Vector3& node_position, const CircleDesignator& base, TreeMesh& mesh, const float uv_y)
    {
        if (node.children.size() < 2)
        {
            float uv_growth = node.length / (node.radius+.001f) / (2*M_PI);
            auto child_circle = add_circle(node_position, node, 1 , base.radial_n, mesh, uv_y + uv_growth);
            bridge_circles(base, child_circle, base.radial_n, mesh);
            Vector3 child_pos = NodeUtilities::get_position_in_node(node_position, node, 1);

            if (!node.is_leaf())
            {
                mesh_node_rec(node.children[0]->node, child_pos, child_circle, mesh, uv_y + uv_growth);
            }
        }
        else
        {
            float uv_growth = node.length / (node.radius + .001f) / (2*M_PI);
            auto end_circle = add_circle(node_position, node, 1, base.radial_n, mesh, uv_y + uv_growth);
            std::vector<IndexRange> children_ranges = get_children_ranges(node, base.radial_n);
            bridge_circles(base, end_circle, base.radial_n, mesh, &children_ranges);
            for (int i = 0; i < node.children.size(); i++)
            {
                if (i == 0) // first child is the continuity of the branch
                {
                    Vector3 child_pos = NodeUtilities::get_position_in_node(node_position, node, 1);
                    if (node.children.size() > 0)
                    {
                        mesh_node_rec(node.children[0]->node, child_pos, end_circle, mesh, uv_y + uv_growth);
                    }
                }
                else
                {
                    auto& child = *node.children[i];
                    Vector3 child_pos = get_side_child_position(node, child, node_position);

                    auto child_base = add_child_circle(node, child, child_pos, node_position, base, children_ranges[i - 1], uv_y, mesh);
                    mesh_node_rec(node.children[i]->node, child_pos, child_base, mesh, uv_y + uv_growth);
                }
            }
        }
    }
}


namespace Mtree
{

	TreeMesh ManifoldMesher::mesh_tree(Tree& tree)
    {
        TreeMesh mesh;
        auto& smooth_attr = mesh.add_attribute<float>(AttributeNames::smooth_amount);
        auto& radius_attr = mesh.add_attribute<float>(AttributeNames::radius);
        auto& direction_attr = mesh.add_attribute<Vector3>(AttributeNames::direction);
        for (auto& stem : tree.get_stems())
        {
            //__debugbreak();

            if (stem.node.children.size() == 0)
                continue;
            CircleDesignator start_circle{ (int)mesh.vertices.size(), (int)mesh.uvs.size(), radial_resolution };
            add_circle(stem.position, stem.node, 0, radial_resolution, mesh, 0);
            mesh_node_rec(stem.node, stem.position, start_circle, mesh, 0);
        }
        if (smooth_iterations > 0)
            MeshProcessing::Smoothing::smooth_mesh(mesh, smooth_iterations, 1, &smooth_attr.data);
        return mesh;
    }
    
}