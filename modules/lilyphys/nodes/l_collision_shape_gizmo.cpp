//
// Created by amara on 25/11/2021.
//

#include "l_collision_shape_gizmo.h"

#include "l_collision_shape.h"
#include "scene/3d/camera.h"

LCollisionShapeGizmoPlugin::LCollisionShapeGizmoPlugin() {
    const Color gizmo_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/l_shape", Color(0.5, 0.7, 1));
    create_material("l_shape_material", gizmo_color);
    const float gizmo_value = gizmo_color.get_v();
    const Color gizmo_color_disabled = Color(gizmo_value, gizmo_value, gizmo_value, 0.65);
    create_material("l_shape_material_disabled", gizmo_color_disabled);
    create_handle_material("handles");
}

bool LCollisionShapeGizmoPlugin::has_gizmo(Spatial *p_spatial) {
    return Object::cast_to<LCollisionShape>(p_spatial) != NULL;
}

String LCollisionShapeGizmoPlugin::get_name() const {
    return "LCollisionShape";
}

int LCollisionShapeGizmoPlugin::get_priority() const {
    return -1;
}

String LCollisionShapeGizmoPlugin::get_handle_name(const EditorSpatialGizmo *p_gizmo, int p_idx) const {

    const LCollisionShape *cs = Object::cast_to<LCollisionShape>(p_gizmo->get_spatial_node());

    Ref<LShape> s = cs->get_shape();
    if (Object::cast_to<LBoxShape>(*s)) {

        return "Extents";
    }

    if (Object::cast_to<LSphereShape>(*s)) {

        return "Radius";
    }

    return "";
}

Variant LCollisionShapeGizmoPlugin::get_handle_value(EditorSpatialGizmo *p_gizmo, int p_idx) const {

    LCollisionShape *cs = Object::cast_to<LCollisionShape>(p_gizmo->get_spatial_node());

    Ref<LShape> s = cs->get_shape();
    if (s.is_null())
        return Variant();

    if (Object::cast_to<LBoxShape>(*s)) {

        Ref<LBoxShape> bs = s;
        return bs->get_extents();
    }

    if (Object::cast_to<LSphereShape>(*s)) {

        Ref<LSphereShape> ss = s;
        return ss->get_radius();
    }

    return Variant();
}
void LCollisionShapeGizmoPlugin::set_handle(EditorSpatialGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point) {

    LCollisionShape *cs = Object::cast_to<LCollisionShape>(p_gizmo->get_spatial_node());

    Ref<LShape> s = cs->get_shape();
    if (s.is_null())
        return;

    Transform gt = cs->get_global_transform();
    Transform gi = gt.affine_inverse();

    Vector3 ray_from = p_camera->project_ray_origin(p_point);
    Vector3 ray_dir = p_camera->project_ray_normal(p_point);

    Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 4096) };

    if (Object::cast_to<LBoxShape>(*s)) {

        Vector3 axis;
        axis[p_idx] = 1.0;
        Ref<LBoxShape> bs = s;
        Vector3 ra, rb;
        Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
        float d = ra[p_idx];
        if (SpatialEditor::get_singleton()->is_snap_enabled()) {
            d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
        }

        if (d < 0.001)
            d = 0.001;

        Vector3 he = bs->get_extents();
        he[p_idx] = d;
        bs->set_extents(he);
    }

    if (Object::cast_to<LSphereShape>(*s)) {

        Ref<LSphereShape> ss = s;
        Vector3 ra, rb;
        Geometry::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
        float d = ra.x;
        if (SpatialEditor::get_singleton()->is_snap_enabled()) {
            d = Math::stepify(d, SpatialEditor::get_singleton()->get_translate_snap());
        }

        if (d < 0.001)
            d = 0.001;

        ss->set_radius(d);
    }
}
void LCollisionShapeGizmoPlugin::commit_handle(EditorSpatialGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) {

    LCollisionShape *cs = Object::cast_to<LCollisionShape>(p_gizmo->get_spatial_node());

    Ref<LShape> s = cs->get_shape();
    if (s.is_null())
        return;

    if (Object::cast_to<LBoxShape>(*s)) {

        Ref<LBoxShape> ss = s;
        if (p_cancel) {
            ss->set_extents(p_restore);
            return;
        }

        UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
        ur->create_action(TTR("Change Box Shape Extents"));
        ur->add_do_method(ss.ptr(), "set_extents", ss->get_extents());
        ur->add_undo_method(ss.ptr(), "set_extents", p_restore);
        ur->commit_action();
    }

    if (Object::cast_to<LSphereShape>(*s)) {

        Ref<LSphereShape> ss = s;
        if (p_cancel) {
            ss->set_radius(p_restore);
            return;
        }

        UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
        ur->create_action(TTR("Change Sphere Shape Radius"));
        ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
        ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
        ur->commit_action();
    }
}
void LCollisionShapeGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {

    LCollisionShape *cs = Object::cast_to<LCollisionShape>(p_gizmo->get_spatial_node());

    p_gizmo->clear();

    Ref<LShape> s = cs->get_shape();
    if (s.is_null())
        return;

    const Ref<Material> material =
            get_material(!cs->is_disabled() ? "l_shape_material" : "l_shape_material_disabled", p_gizmo);
    Ref<Material> handles_material = get_material("handles");

    if (Object::cast_to<LBoxShape>(*s)) {

        Ref<LBoxShape> bs = s;
        Vector<Vector3> lines;
        AABB aabb;
        aabb.position = -bs->get_extents();
        aabb.size = aabb.position * -2;

        for (int i = 0; i < 12; i++) {
            Vector3 a, b;
            aabb.get_edge(i, a, b);
            lines.push_back(a);
            lines.push_back(b);
        }

        Vector<Vector3> handles;

        for (int i = 0; i < 3; i++) {

            Vector3 ax;
            ax[i] = bs->get_extents()[i];
            handles.push_back(ax);
        }

        p_gizmo->add_lines(lines, material);
        p_gizmo->add_collision_segments(lines);
        p_gizmo->add_handles(handles, handles_material);
    }

    if (Object::cast_to<LSphereShape>(*s)) {

        Ref<LSphereShape> sp = s;
        float r = sp->get_radius();

        Vector<Vector3> points;

        for (int i = 0; i <= 360; i++) {

            float ra = Math::deg2rad((float)i);
            float rb = Math::deg2rad((float)i + 1);
            Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
            Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

            points.push_back(Vector3(a.x, 0, a.y));
            points.push_back(Vector3(b.x, 0, b.y));
            points.push_back(Vector3(0, a.x, a.y));
            points.push_back(Vector3(0, b.x, b.y));
            points.push_back(Vector3(a.x, a.y, 0));
            points.push_back(Vector3(b.x, b.y, 0));
        }

        Vector<Vector3> collision_segments;

        for (int i = 0; i < 64; i++) {

            float ra = i * Math_PI * 2.0 / 64.0;
            float rb = (i + 1) * Math_PI * 2.0 / 64.0;
            Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r;
            Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r;

            collision_segments.push_back(Vector3(a.x, 0, a.y));
            collision_segments.push_back(Vector3(b.x, 0, b.y));
            collision_segments.push_back(Vector3(0, a.x, a.y));
            collision_segments.push_back(Vector3(0, b.x, b.y));
            collision_segments.push_back(Vector3(a.x, a.y, 0));
            collision_segments.push_back(Vector3(b.x, b.y, 0));
        }

        p_gizmo->add_lines(points, material);
        p_gizmo->add_collision_segments(collision_segments);
        Vector<Vector3> handles;
        handles.push_back(Vector3(r, 0, 0));
        p_gizmo->add_handles(handles, handles_material);
    }
}