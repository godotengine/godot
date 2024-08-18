#ifndef _MCURVEMESH
#define _MCURVEMESH

#define SLICE_EPSILONE 0.001

#include "core/object/worker_thread_pool.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "core/templates/vector.h"
#include "core/templates/vmap.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

#include "mpath.h"
#include "mintersection.h"
#include "../octmesh/mmesh_lod.h"
#include "mcurve_mesh_override.h"

#include <mutex>

using namespace godot;


class MeshSlicedInfo : public RefCounted
{
    public:
    float lenght;
    RID mesh_rid = RID(); // This used to detect the uniqueness
    Ref<Material> material;
    PackedVector3Array vertex;
    PackedVector3Array normal;
    PackedFloat32Array tangent;
    PackedColorArray color;
    PackedVector2Array uv;
    PackedVector2Array uv2;
    PackedInt32Array index;
    Vector<float> sliced_pos;
    Vector<Vector<int32_t>> sliced_info;
    void merge_vertex_by_distance(float merge_distance=0.0001f); // after calling this sliced_pos and slice_info are invalide and should be recalculate
    void clear();
    int slice_count() const;
    void get_color(int mesh_count,PackedColorArray& input);
    void get_uv(int mesh_count,PackedVector2Array& input);
    void get_uv2(int mesh_count,PackedVector2Array& input);
    void get_index(int mesh_count,PackedInt32Array& input);
};

class MCurveMesh : public Node {
    GDCLASS(MCurveMesh,Node);
    protected:
    static void _bind_methods();

    private:
    bool merge_vertex_by_distance = false;
    bool is_thread_updating = false;
    int32_t curve_user_id;
    WorkerThreadPool::TaskID thread_task_id;
    MPath* path=nullptr;
    Ref<MCurve> curve;
    Array meshes;
    Array intersections;
    Array materials;

    struct MeshSlicedInfoArray
    {
        // order of each mesh lod
        Vector<Ref<MeshSlicedInfo>> meshes;
        inline void resize(int new_size){ meshes.resize(new_size); }
        inline void set(int index,Ref<MeshSlicedInfo> input){ meshes.set(index,input); }
        inline Ref<MeshSlicedInfo> get(int index) const { return meshes[index]; }
        inline int size() const { return meshes.size(); }
    };

    struct Instance {
        RID original_mesh_rid;
        RID instance;
        RID mesh;
    };

    private:
    Vector<MeshSlicedInfoArray> meshlod_sliced_info;// Order of meshes
    HashMap<int64_t,Instance> curve_mesh_instances;
    Ref<MCurveMeshOverride> ov;

    Vector<Pair<int64_t,RID>> mesh_updated_list; // int64t -> connection ID, RID -> can be mesh or multimesh
    HashMap<int64_t,Pair<float,float>> conn_ratio_limits;

    public:
    MCurveMesh()=default;
    ~MCurveMesh();
    std::recursive_mutex update_mutex;
    void _on_connections_updated();

    static void thread_update(void* input);

    void _generate_all_mesh_sliced_info();
    private:
    Ref<MeshSlicedInfo> _generate_mesh_sliced_info(Ref<Mesh> mesh);

    public:
    void _update_visibilty();
    void _apply_update();
    void _remove_instance(int64_t id,bool is_intersection=false);
    void _remove_mesh(int64_t id,bool is_intersection=false);
    void clear();
    void restart();
    void reload();
    void recreate();
    void _swap_point_id(int64_t p_a,int64_t p_b);
    void _id_force_update(int64_t id);
    void _point_force_update(int32_t point_id);
    void _connection_force_update(int64_t conn_id);
    void _point_remove(int32_t point_id);
    void _connection_remove(int64_t conn_id);
    void _generate_connection(const MCurve::ConnUpdateInfo& update_info,bool immediate_update=false);
    void _generate_intersection(const MCurve::PointUpdateInfo& update_info,bool immediate_update=false);
    void _process_tick();

    _FORCE_INLINE_ Pair<float,float> get_conn_ratio_limits(int64_t conn_id);
    _FORCE_INLINE_ void set_conn_ratio_limits(int64_t conn_id,float limit , bool is_end);
    _FORCE_INLINE_ void clear_point_conn_ratio_limits(int32_t point_id);

    public:
    void set_overrides(Ref<MCurveMeshOverride> input);
    Ref<MCurveMeshOverride> get_overrides();
    void set_meshes(Array input);
    Array get_meshes();

    void set_intersections(Array input);
    Array get_intersections();

    void set_materials(Array input);
    Array get_materials();

    void _on_curve_changed();
    void _generate_all_intersections_info();
    void _notification(int p_what);
    PackedStringArray _get_configuration_warnings() const;
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

    void dumy_set_restart(bool input);
    bool dumy_get_true();
};
#endif