#ifdef TOOLS_ENABLED

#include "gaussian_splat_gizmo_plugin.h"
#include "core/config/project_settings.h"
#include "core/math/math_defs.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/3d/camera_3d.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"
#include "scene/resources/texture.h"

#include "../nodes/gaussian_splat_node_3d.h"

GaussianSplatGizmoPlugin::GaussianSplatGizmoPlugin() {
    create_materials();
}

void GaussianSplatGizmoPlugin::create_materials() {
    // Bounds material (wireframe box)
    bounds_material.instantiate();
    bounds_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    bounds_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
    bounds_material->set_albedo(Color(0.5, 0.7, 1.0, 0.3));
    bounds_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    // Selected bounds material
    bounds_material_selected.instantiate();
    bounds_material_selected->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    bounds_material_selected->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
    bounds_material_selected->set_albedo(Color(1.0, 0.8, 0.2, 0.6));
    bounds_material_selected->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    // LOD visualization materials
    const Color lod_colors[3] = {
        Color(0.2, 1.0, 0.2, 0.25),
        Color(1.0, 0.8, 0.2, 0.2),
        Color(1.0, 0.3, 0.2, 0.15)
    };
    for (int i = 0; i < 3; i++) {
        lod_sphere_materials[i].instantiate();
        lod_sphere_materials[i]->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
        lod_sphere_materials[i]->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
        lod_sphere_materials[i]->set_albedo(lod_colors[i]);
        lod_sphere_materials[i]->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
    }

    // Statistics overlay material
    statistics_material.instantiate();
    statistics_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    statistics_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
    statistics_material->set_albedo(Color(1.0, 1.0, 1.0, 0.9));

    // Performance overlay materials
    performance_material_good.instantiate();
    performance_material_good->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    performance_material_good->set_albedo(Color(0.2, 0.9, 0.2, 0.8));
    performance_material_good->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    performance_material_warning.instantiate();
    performance_material_warning->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    performance_material_warning->set_albedo(Color(0.95, 0.75, 0.2, 0.85));
    performance_material_warning->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    performance_material_critical.instantiate();
    performance_material_critical->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    performance_material_critical->set_albedo(Color(0.95, 0.25, 0.15, 0.9));
    performance_material_critical->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    // Preview materials
    preview_wire_material.instantiate();
    preview_wire_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    preview_wire_material->set_albedo(Color(0.7, 0.9, 1.0, 0.8));
    preview_wire_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    preview_point_material.instantiate();
    preview_point_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
    preview_point_material->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
    preview_point_material->set_albedo(Color(0.9, 0.9, 1.0));
    preview_point_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

    const Color heatmap_colors[3] = {
        Color(0.0, 0.6, 1.0, 0.9),
        Color(1.0, 0.8, 0.2, 0.9),
        Color(1.0, 0.2, 0.1, 0.9)
    };
    for (int i = 0; i < 3; i++) {
        preview_heatmap_materials[i].instantiate();
        preview_heatmap_materials[i]->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
        preview_heatmap_materials[i]->set_albedo(heatmap_colors[i]);
        preview_heatmap_materials[i]->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
    }
}

bool GaussianSplatGizmoPlugin::has_gizmo(Node3D *p_spatial) {
    return Object::cast_to<GaussianSplatNode3D>(p_spatial) != nullptr;
}

String GaussianSplatGizmoPlugin::get_gizmo_name() const {
    return "GaussianSplat";
}

int GaussianSplatGizmoPlugin::get_priority() const {
    return -1; // Default priority
}

void GaussianSplatGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
    p_gizmo->clear();

    GaussianSplatNode3D *splat_node = Object::cast_to<GaussianSplatNode3D>(p_gizmo->get_node_3d());
    if (!splat_node) {
        return;
    }

    // Always draw bounds if enabled or selected
    if (splat_node->is_showing_bounds() || p_gizmo->is_selected()) {
        draw_bounds(p_gizmo, splat_node);
    }

    // Draw LOD radius indicator
    if (splat_node->is_showing_lod_spheres() || p_gizmo->is_selected()) {
        draw_lod_radius(p_gizmo, splat_node);
    }

    // Draw statistics overlay and performance metrics
    if (splat_node->is_showing_statistics() || splat_node->is_showing_performance_overlay()) {
        draw_statistics(p_gizmo, splat_node);
    }

    // Draw splat preview (simplified representation)
    if (splat_node->is_preview_enabled() && splat_node->get_splat_asset().is_valid() &&
            splat_node->get_debug_draw_mode() != GaussianSplatNode3D::DEBUG_DRAW_OFF) {
        draw_splat_preview(p_gizmo, splat_node);
    }
}

void GaussianSplatGizmoPlugin::draw_bounds(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node) {
    AABB aabb = p_node->get_aabb();
    if (aabb.size == Vector3()) {
        return;
    }

    Vector<Vector3> lines;
    Vector3 min_pos = aabb.position;
    Vector3 max_pos = aabb.position + aabb.size;

    // Draw box edges
    // Bottom face
    lines.push_back(Vector3(min_pos.x, min_pos.y, min_pos.z));
    lines.push_back(Vector3(max_pos.x, min_pos.y, min_pos.z));

    lines.push_back(Vector3(max_pos.x, min_pos.y, min_pos.z));
    lines.push_back(Vector3(max_pos.x, min_pos.y, max_pos.z));

    lines.push_back(Vector3(max_pos.x, min_pos.y, max_pos.z));
    lines.push_back(Vector3(min_pos.x, min_pos.y, max_pos.z));

    lines.push_back(Vector3(min_pos.x, min_pos.y, max_pos.z));
    lines.push_back(Vector3(min_pos.x, min_pos.y, min_pos.z));

    // Top face
    lines.push_back(Vector3(min_pos.x, max_pos.y, min_pos.z));
    lines.push_back(Vector3(max_pos.x, max_pos.y, min_pos.z));

    lines.push_back(Vector3(max_pos.x, max_pos.y, min_pos.z));
    lines.push_back(Vector3(max_pos.x, max_pos.y, max_pos.z));

    lines.push_back(Vector3(max_pos.x, max_pos.y, max_pos.z));
    lines.push_back(Vector3(min_pos.x, max_pos.y, max_pos.z));

    lines.push_back(Vector3(min_pos.x, max_pos.y, max_pos.z));
    lines.push_back(Vector3(min_pos.x, max_pos.y, min_pos.z));

    // Vertical edges
    lines.push_back(Vector3(min_pos.x, min_pos.y, min_pos.z));
    lines.push_back(Vector3(min_pos.x, max_pos.y, min_pos.z));

    lines.push_back(Vector3(max_pos.x, min_pos.y, min_pos.z));
    lines.push_back(Vector3(max_pos.x, max_pos.y, min_pos.z));

    lines.push_back(Vector3(max_pos.x, min_pos.y, max_pos.z));
    lines.push_back(Vector3(max_pos.x, max_pos.y, max_pos.z));

    lines.push_back(Vector3(min_pos.x, min_pos.y, max_pos.z));
    lines.push_back(Vector3(min_pos.x, max_pos.y, max_pos.z));

    Ref<Material> material = p_gizmo->is_selected() ? bounds_material_selected : bounds_material;
    p_gizmo->add_lines(lines, material, false);

    // Add collision for selection
    Vector<Vector3> collision_segments;
    collision_segments.append_array(lines);
    p_gizmo->add_collision_segments(collision_segments);
}

void GaussianSplatGizmoPlugin::draw_lod_radius(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node) {
    float max_distance = p_node->get_max_render_distance();
    if (max_distance <= 0.0f) {
        return;
    }

    // Draw circles at different LOD distances
    const int segments = 48;
    const float lod_distances[] = { max_distance * 0.25f, max_distance * 0.5f, max_distance };
    const Vector3 center = p_node->get_aabb().get_center();

    for (int lod = 0; lod < 3; lod++) {
        Vector<Vector3> circle_points;
        float radius = MAX(0.01f, lod_distances[lod]);

        for (int i = 1; i <= segments; i++) {
            float prev_angle = ((i - 1) / float(segments)) * Math::TAU;
            float angle = (i / float(segments)) * Math::TAU;
            Vector3 from(center.x + radius * cos(prev_angle), center.y, center.z + radius * sin(prev_angle));
            Vector3 to(center.x + radius * cos(angle), center.y, center.z + radius * sin(angle));
            circle_points.push_back(from);
            circle_points.push_back(to);
        }

        if (lod_sphere_materials[lod].is_valid()) {
            p_gizmo->add_lines(circle_points, lod_sphere_materials[lod], false);
        } else {
            p_gizmo->add_lines(circle_points, bounds_material, false);
        }
    }
}

void GaussianSplatGizmoPlugin::draw_statistics(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node) {
    Dictionary stats = p_node->get_statistics();
    Vector3 center = p_node->get_aabb().get_center();

    if (p_node->is_showing_statistics() && statistics_material.is_valid()) {
        Vector<Vector3> cross_lines;
        const float cross_extent = MAX(0.1f, p_node->get_aabb().get_longest_axis_size() * 0.05f);
        cross_lines.push_back(center + Vector3(0, cross_extent, 0));
        cross_lines.push_back(center - Vector3(0, cross_extent, 0));
        cross_lines.push_back(center + Vector3(cross_extent, 0, 0));
        cross_lines.push_back(center - Vector3(cross_extent, 0, 0));
        cross_lines.push_back(center + Vector3(0, 0, cross_extent));
        cross_lines.push_back(center - Vector3(0, 0, cross_extent));
        p_gizmo->add_lines(cross_lines, statistics_material, true);
    }

    if (p_node->is_showing_performance_overlay()) {
        auto get_material_for_time = [this](float p_ms) -> Ref<StandardMaterial3D> {
            if (p_ms <= 16.0f) {
                return performance_material_good;
            }
            if (p_ms <= 33.0f) {
                return performance_material_warning;
            }
            return performance_material_critical;
        };

        float update_ms = stats.has(StringName("update_time_ms")) ? float(stats[StringName("update_time_ms")]) : 0.0f;
        float render_ms = stats.has(StringName("render_time_ms")) ? float(stats[StringName("render_time_ms")]) : 0.0f;
        float sort_ms = stats.has(StringName("sort_time_ms")) ? float(stats[StringName("sort_time_ms")]) : 0.0f;

        Vector3 aabb_size = p_node->get_aabb().get_size();
        const float max_height = MAX(0.25f, aabb_size.y * 0.6f);
        const float base_height = MAX(0.05f, aabb_size.y * 0.3f);
        const float spacing = MAX(0.1f, aabb_size.x * 0.2f);

        struct MetricBar {
            float ms;
            float offset;
        } bars[3] = {
            { update_ms, -spacing },
            { render_ms, 0.0f },
            { sort_ms, spacing }
        };

        for (int i = 0; i < 3; i++) {
            float normalized = CLAMP(bars[i].ms / 50.0f, 0.0f, 1.0f);
            float bar_height = base_height + normalized * max_height;
            Vector<Vector3> bar_lines;
            Vector3 base = center + Vector3(bars[i].offset, aabb_size.y * 0.5f, 0);
            bar_lines.push_back(base);
            bar_lines.push_back(base + Vector3(0, bar_height, 0));
            bar_lines.push_back(base + Vector3(-spacing * 0.2f, bar_height * 0.2f, 0));
            bar_lines.push_back(base + Vector3(spacing * 0.2f, bar_height * 0.2f, 0));

            Ref<StandardMaterial3D> mat = get_material_for_time(bars[i].ms);
            if (mat.is_valid()) {
                p_gizmo->add_lines(bar_lines, mat, false);
            }
        }
    }
}

void GaussianSplatGizmoPlugin::draw_splat_preview(EditorNode3DGizmo *p_gizmo, GaussianSplatNode3D *p_node) {
    GaussianSplatNode3D::DebugDrawMode mode = p_node->get_debug_draw_mode();

    if (mode == GaussianSplatNode3D::DEBUG_DRAW_WIREFRAME) {
        AABB aabb = p_node->get_aabb();
        Vector<Vector3> wire_lines;
        Vector3 min_pos = aabb.position;
        Vector3 max_pos = aabb.position + aabb.size;

        // Reuse bounds drawing logic but with preview material to avoid selection tint.
        const Vector3 corners[8] = {
            Vector3(min_pos.x, min_pos.y, min_pos.z),
            Vector3(max_pos.x, min_pos.y, min_pos.z),
            Vector3(max_pos.x, max_pos.y, min_pos.z),
            Vector3(min_pos.x, max_pos.y, min_pos.z),
            Vector3(min_pos.x, min_pos.y, max_pos.z),
            Vector3(max_pos.x, min_pos.y, max_pos.z),
            Vector3(max_pos.x, max_pos.y, max_pos.z),
            Vector3(min_pos.x, max_pos.y, max_pos.z)
        };

        const int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };

        for (int i = 0; i < 12; i++) {
            wire_lines.push_back(corners[edges[i][0]]);
            wire_lines.push_back(corners[edges[i][1]]);
        }

        if (preview_wire_material.is_valid()) {
            p_gizmo->add_lines(wire_lines, preview_wire_material, false);
        } else {
            p_gizmo->add_lines(wire_lines, bounds_material, false);
        }
        return;
    }

    Ref<GaussianSplatAsset> asset = p_node->get_splat_asset();
    if (!asset.is_valid() || asset->get_splat_count() == 0) {
        return;
    }

    const int max_preview_points = 1000;
    int splat_count = asset->get_splat_count();
    int step = MAX(1, splat_count / max_preview_points);

    PackedFloat32Array positions = asset->get_positions();
    PackedColorArray colors = asset->get_colors();

    if (positions.size() < 3) {
        return;
    }

    const float cross_size = MAX(0.01f, p_node->get_aabb().get_longest_axis_size() * 0.01f);

    if (mode == GaussianSplatNode3D::DEBUG_DRAW_POINTS) {
        Vector<Vector3> point_lines;
        for (int i = 0; i < splat_count && point_lines.size() / 6 < max_preview_points; i += step) {
            int idx = i * 3;
            if (idx + 2 >= positions.size()) {
                break;
            }
            Vector3 point(positions[idx], positions[idx + 1], positions[idx + 2]);
            point_lines.push_back(point - Vector3(cross_size, 0, 0));
            point_lines.push_back(point + Vector3(cross_size, 0, 0));
            point_lines.push_back(point - Vector3(0, cross_size, 0));
            point_lines.push_back(point + Vector3(0, cross_size, 0));
            point_lines.push_back(point - Vector3(0, 0, cross_size));
            point_lines.push_back(point + Vector3(0, 0, cross_size));
        }

        if (!point_lines.is_empty()) {
            Ref<StandardMaterial3D> mat = preview_point_material.is_valid() ? preview_point_material : bounds_material;
            p_gizmo->add_lines(point_lines, mat, false);
        }
        return;
    }

    if (mode == GaussianSplatNode3D::DEBUG_DRAW_HEATMAP) {
        Vector<Vector3> heatmap_lines[3];
        for (int i = 0; i < splat_count && (heatmap_lines[0].size() + heatmap_lines[1].size() + heatmap_lines[2].size()) / 6 < max_preview_points; i += step) {
            int idx = i * 3;
            if (idx + 2 >= positions.size()) {
                break;
            }

            Vector3 point(positions[idx], positions[idx + 1], positions[idx + 2]);
            float intensity = 0.0f;
            if (i < colors.size()) {
                intensity = colors[i].get_luminance();
            } else {
                intensity = float(i) / float(splat_count);
            }
            int bucket = intensity < 0.33f ? 0 : (intensity < 0.66f ? 1 : 2);

            Vector<Vector3> &target = heatmap_lines[bucket];
            target.push_back(point - Vector3(cross_size, 0, 0));
            target.push_back(point + Vector3(cross_size, 0, 0));
            target.push_back(point - Vector3(0, cross_size, 0));
            target.push_back(point + Vector3(0, cross_size, 0));
            target.push_back(point - Vector3(0, 0, cross_size));
            target.push_back(point + Vector3(0, 0, cross_size));
        }

        for (int i = 0; i < 3; i++) {
            if (!heatmap_lines[i].is_empty()) {
                Ref<StandardMaterial3D> mat = preview_heatmap_materials[i].is_valid() ? preview_heatmap_materials[i] : bounds_material;
                p_gizmo->add_lines(heatmap_lines[i], mat, false);
            }
        }
    }
}

int GaussianSplatGizmoPlugin::subgizmos_intersect_ray(const EditorNode3DGizmo *p_gizmo, Camera3D *p_camera, const Vector2 &p_point) const {
    // Handle sub-gizmo selection if needed
    return -1;
}

Vector<int> GaussianSplatGizmoPlugin::subgizmos_intersect_frustum(const EditorNode3DGizmo *p_gizmo, const Camera3D *p_camera, const Vector<Plane> &p_frustum) const {
    // Handle frustum selection if needed
    return Vector<int>();
}

Transform3D GaussianSplatGizmoPlugin::get_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id) const {
    return Transform3D();
}

void GaussianSplatGizmoPlugin::set_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id, Transform3D p_transform) {
    // Handle sub-gizmo transformation if needed
}

void GaussianSplatGizmoPlugin::commit_subgizmos(const EditorNode3DGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform3D> &p_restore, bool p_cancel) {
    // Handle sub-gizmo commit if needed
}

#endif // TOOLS_ENABLED
