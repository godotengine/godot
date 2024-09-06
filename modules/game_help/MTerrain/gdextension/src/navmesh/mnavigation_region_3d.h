#ifndef MNAVIGATIONREGION3D
#define MNAVIGATIONREGION3D

#define BUFFER_STRID_FLOAT 12
#include "servers/navigation_server_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/resources/navigation_mesh.h"
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"
#include "scene/resources/mesh.h"
#include "scene/main/timer.h"
#include "core/templates/vector.h"
#include "core/templates/vset.h"
#include "core/templates/hash_map.h"
#include "core/math/vector4.h"

#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"
#include "../mterrain.h"
#include "mnavigation_mesh_data.h"
#include "../grass/mgrass_chunk.h"
#include "mobstacle.h"

#include <thread>
#include <future>
#include <chrono>




class MNavigationRegion3D : public NavigationRegion3D{
    GDCLASS(MNavigationRegion3D,NavigationRegion3D);
    Ref<MNavigationMeshData> nav_data;
    bool follow_camera = true;
    MGrid* grid = nullptr;
    MTerrain* terrain = nullptr;
    float navigation_radius = 256;
    std::future<void> update_thread;
    bool is_updating = false;
    bool _force_update = false;
    bool active = true;
    Vector3 cam_pos;
    Vector3 g_pos;
    Vector3 last_update_pos;
    Node3D* custom_camera = nullptr;
    Timer* update_timer;
    float distance_update_threshold=64;
    void _update_navmesh(Vector3 cam_pos);
    MeshInstance3D* debug_mesh_instance;
    Ref<ArrayMesh> debug_mesh;
    Ref<NavigationMesh> tmp_nav;
    Vector<Vector4> obs_info;
    PackedFloat32Array obst_info;

    RID scenario;
    uint32_t base_grid_size_in_pixel;
    uint32_t region_pixel_width;
    uint32_t region_pixel_size;
    std::mutex npoint_mutex;
    HashMap<int64_t,MGrassChunk*> grid_to_npoint;
    Vector<MGrassChunk*> to_be_visible;
    VSet<int>* dirty_points_id;
    uint64_t update_id=0;
    Ref<ArrayMesh> paint_mesh;
    Ref<StandardMaterial3D> paint_material;
    uint32_t region_grid_width;
    uint32_t width;
    uint32_t height;
    float h_scale;
    bool is_npoints_visible = false;
    int max_shown_lod=2;
    static VSet<MObstacle*> obstacles;
    static Vector<MNavigationRegion3D*> all_navigation_nodes;




    protected:
    static void _bind_methods();

    public:
    static TypedArray<MNavigationRegion3D> get_all_navigation_nodes();
    struct ObstacleInfo
    {
        float width;
        float depth;
        Transform3D transform;
    };
    Vector<ObstacleInfo> obstacles_infos;
    bool is_nav_init = false;
    MNavigationRegion3D();
    ~MNavigationRegion3D();
    
    void init(MTerrain* _terrain, MGrid* _grid);
    void clear();
    void _update_loop();
    void update_navmesh(Vector3 cam_pos);
    void _finish_update(Ref<NavigationMesh> nvm);
    void _set_is_updating(bool input);
    void get_cam_pos();
    void force_update();
    bool has_data();

    void set_force_update(bool input);
    bool get_force_update();

    void set_nav_data(Ref<MNavigationMeshData> input);
    Ref<MNavigationMeshData> get_nav_data();

    void set_follow_camera(bool input);
    bool get_follow_camera();

    void set_active(bool input);
    bool get_active();

    void set_distance_update_threshold(float input);
    float get_distance_update_threshold();

    void set_navigation_radius(float input);
    float get_navigation_radius();

    void set_max_shown_lod(int input);
    int get_max_shown_lod();



    void update_npoints();
    void update_dirty_npoints();
    void apply_update_npoints();
    void create_npoints(int grid_index,MGrassChunk* grass_chunk=nullptr);
    void set_npoint_by_pixel(uint32_t px, uint32_t py, bool p_value);
    bool get_npoint_by_pixel(uint32_t px, uint32_t py);
    Vector2i get_closest_pixel(Vector3 pos);
    void draw_npoints(Vector3 brush_pos,real_t radius,bool add);
    void set_npoints_visible(bool val);

    Error save_nav_data();

    static void add_obstacle(MObstacle* input);
    static void remove_obstacle(MObstacle* input);
};
#endif