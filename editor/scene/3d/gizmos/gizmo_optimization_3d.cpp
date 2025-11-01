#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/resources/mesh.h"
#include "scene/resources/material.h"

class SimplePerfGizmo : public EditorNode3DGizmo {
    GDCLASS(SimplePerfGizmo, EditorNode3DGizmo);

private:
    Ref<Mesh> batched_mesh;         
    Ref<Material> simple_material;  
    bool handles_visible = false;   

public:
    SimplePerfGizmo() {
        // Create a simple green material
        simple_material.instantiate();
        simple_material->set("albedo_color", Color(0, 1, 0, 1));

        // Create an empty mesh to batch all geometry
        batched_mesh.instantiate();
    }

    void create() override {
        if (!get_node_3d()) return;

        batched_mesh->clear();

        for (int i = 0; i < 12; i++) {
            Vector3 start = Vector3(i*0.1, 0, 0);
            Vector3 end   = Vector3(i*0.1, 1, 0);
            batched_mesh->add_vertex(start);
            batched_mesh->add_vertex(end);
        }

        if (handles_visible) {
            for (int i = 0; i < 8; i++) {
                Vector3 handle = Vector3(i*0.1, i*0.1, i*0.1);
                batched_mesh->add_vertex(handle);
                batched_mesh->add_vertex(handle + Vector3(0.05, 0, 0));
            }
        }

        add_mesh(batched_mesh, Transform3D());
    }

    void transform() override {
        // Apply the node's transform to the batched mesh
        Transform3D t = get_node_3d()->get_global_transform();
        set_mesh_transform(batched_mesh, t);
    }

    void set_selected(bool selected) {
        handles_visible = selected;
        create();
    }
};

class SimplePerfGizmoPlugin : public EditorNode3DGizmoPlugin {
    GDCLASS(SimplePerfGizmoPlugin, EditorNode3DGizmoPlugin);

public:
    Ref<EditorNode3DGizmo> create_gizmo(Node3D *node) override {
        if (!node) return nullptr;
        SimplePerfGizmo *gizmo = memnew(SimplePerfGizmo);
        gizmo->set_node_3d(node);
        gizmo->create();
        return Ref<EditorNode3DGizmo>(gizmo);
    }

    bool has_gizmo(Node3D *node) const override {
        return node != nullptr;
    }
};