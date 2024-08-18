#ifndef __MOctree
#define __MOctree

#define MIN_OCTANT_EDGE_LENGTH 0.2
#define EXTRA_BOUND_MARGIN 200 //This should be alway some number bigger than zero, otherwise cause some points to not be inserted
#define MAX_CAPACITY 10000
#define MIN_CAPACITY 10
#define INVALID_OCT_POINT_ID -1
#define MAX_LOD 125

#include "core/object/object.h"
#include "scene/main/viewport.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"
#include "core/templates/vector.h"
#include "core/templates/vset.h"
#include "core/templates/hash_map.h"
#include "core/templates/rb_set.h"


#include <mutex>
#include <thread>
#include <chrono>


using namespace godot;


class MOctree : public Node3D {
    GDCLASS(MOctree,Node3D);
    protected:
    static void _bind_methods();

    private:
    /*
    Octant number according to coordinate bellow (y is toward screen)
    ->x
    |z
    V
    First Floor or lower y
    |0|1|
    |2|3|
    Second Floor or upper y
    |4|5|
    |6|7|
    */
    struct OctPoint
    {
        int8_t lod = -1;
        uint8_t update_id = 0;
        uint16_t oct_id = 0;
        int32_t id;
        Vector3 position;
        OctPoint(){}
        OctPoint(const int32_t _id,const Vector3& _position,uint8_t _oct_id):
        id(_id),position(_position),oct_id(_oct_id){}
    };
    public:

    struct PointUpdate
    {
        int8_t lod;
        int8_t last_lod;
        int32_t id;
    };

    struct PointMoveReq
    {
        int32_t p_id;
        uint16_t oct_id;
        Vector3 old_pos;
        Vector3 new_pos;
        PointMoveReq();
        PointMoveReq(int32_t _p_id,uint16_t _oct_id,Vector3 _old_pos,Vector3 _new_pos);
        _FORCE_INLINE_ uint64_t hash() const;
        bool operator<(const PointMoveReq& other) const;
    };

    private:
    struct OctUpdateInfo {
        uint8_t update_id;
        int8_t lod;
        Pair<Vector3,Vector3> bound;
        Pair<Vector3,Vector3> exclude_bound;
    };
    
    
    struct Octant
    {
        Vector<OctPoint> points;
        Vector3 start;
        Vector3 end;
        Octant* octs = nullptr; // pointer to the fist child oct other octs will be in row next to this
        Octant* parent = nullptr;

        Octant();
        ~Octant();
        bool insert_point(const Vector3& p_point,int32_t p_id,uint16_t oct_id , const uint16_t capacity);
        bool insert_point(const OctPoint& p_point, const uint16_t capacity);
        OctPoint* insert_point_ret_octpoint(const OctPoint& p_point, const uint16_t capacity);
        Octant* find_octant_by_point(const int32_t id,const uint16_t oct_id,Vector3 pos,int& point_index);
        Octant* find_octant_by_point_classic(const int32_t id,const uint16_t oct_id,int& point_index);

        void get_ids(const Pair<Vector3,Vector3>& bound, PackedInt32Array& _ids,uint16_t oct_id);
        void get_ids_exclude(const Pair<Vector3,Vector3>& bound,const Pair<Vector3,Vector3>& exclude_bound, PackedInt32Array& _ids,uint16_t oct_id);
        //Bellow only for lod zero which has no exclude
        void update_lod_zero(const OctUpdateInfo& update_info,HashMap<uint16_t,Vector<PointUpdate>>& u_info);
        //Bellow function must be called by LOD order from One to max lod number
        void update_lod(const OctUpdateInfo& update_info,HashMap<uint16_t,Vector<PointUpdate>>& u_info);
        void get_all_bounds(Vector<Pair<Vector3,Vector3>>& bounds);
        void get_all_data(Vector<OctPoint>& data);
        void append_all_ids_to(PackedInt32Array& _ids);
        void get_points_count(int& count);
        void get_oct_id_points_count(uint16_t oct_id,int& count);
        Octant* get_mergeable(int capacity);
        Octant* remove_point(int32_t id,const Vector3& pos,uint16_t oct_id);
        void clear();
        void remove_points_with_oct_id(uint16_t oct_id);
        _FORCE_INLINE_ static bool has_point(const Pair<Vector3,Vector3>& bound, const Vector3& point);
        _FORCE_INLINE_ bool has_point(const Vector3& point) const; //if the point is in ourt bound
        //For Debug
        bool check_id_exist_classic(int32_t id);
        void get_tree_lines(PackedVector3Array& lines);
        void merge_octs();
        private:
        _FORCE_INLINE_ bool divide();
        _FORCE_INLINE_ bool intersects(const Pair<Vector3,Vector3>& bound) const;
        _FORCE_INLINE_ bool encloses_by(const Pair<Vector3,Vector3>& bound) const;
        _FORCE_INLINE_ bool encloses_between(const Pair<Vector3,Vector3>& include, const Pair<Vector3,Vector3>& exclude) const;

    };
    bool is_world_boundary_set = false;
    Vector3 world_start;
    Vector3 world_end;
    uint8_t update_id=0; //Update id zero always is for init state
    uint16_t custom_capacity = 0;
    uint16_t last_oct_id = 0;
    uint32_t point_count = 0;
    Octant root;
    Node3D* camera_node = nullptr;
    Node3D* auto_camera_found = nullptr;
    RID scenario;
    Vector3 camera_position;
    PackedFloat32Array lod_setting;
    HashMap<uint16_t,Vector<PointUpdate>> update_change_info;
    Pair<Vector3,Vector3> last_update_boundary;
    VSet<uint16_t> oct_ids;
    VSet<uint16_t> waiting_oct_ids;
    RBSet<PointMoveReq> moves_req_cache;
    RBSet<PointMoveReq> moves_req;

    bool update_lod_include_root_bound = true;
    bool is_updating = false;
    bool is_point_process_wait = false;
    bool is_first_update = true;

    std::mutex oct_mutex; // work with octtree data
    std::mutex move_req_mutex; // work with this data -> moves_req_cache
    WorkerThreadPool::TaskID tid;

    bool is_camera_warn_print = false;
    bool is_ready = false;
    bool is_octmesh_updater = false;
    bool is_path_updater = false;
    bool disable_octree = false;

    bool debug_draw = false;
    uint64_t last_draw_time=0;
    RID debug_instance;
    Ref<ArrayMesh> debug_mesh;
    Ref<StandardMaterial3D> debug_material;

    public:
    MOctree();
    ~MOctree();
    int get_oct_id();
    void clear_oct_id(int oct_id);
    void remove_oct_id(int oct_id);
    bool remove_point(int32_t id, const Vector3& pos,uint16_t oct_id);
    bool check_for_mergeable(Octant* start_point);
    void set_camera_node(Node3D* camera);
    void set_world_boundary(const Vector3& start,const Vector3& end);
    void enable_as_octmesh_updater();
    void enable_as_curve_updater();
    void update_camera_position();
    uint32_t get_capacity(int p_count);
    //Insert point and id is point index
    void insert_points(const PackedVector3Array& points,const PackedInt32Array ids, int oct_id);
    // bellow insert a single point and update it lod
    // good for adding point after initilazation
    bool insert_point(const Vector3& pos,const int32_t id, int oct_id);
    void change_point_id(int16_t oct_id,const Vector3& point_pos,int32_t old_id,int32_t new_id);
    // Move point and return its confirm OctPoint
    _FORCE_INLINE_ void move_point(const PointMoveReq& mp,int8_t updated_lod,uint8_t update_id); // Must be called only in update_lod
    
    void add_move_req(const PointMoveReq& mv_data);
    void release_move_req_cache();
    
    int8_t get_pos_lod_classic(const Vector3& pos);
    PackedInt32Array get_ids(const AABB& search_bound,int oct_id);
    PackedInt32Array get_ids_exclude(const AABB& search_bound, const AABB& exclude_bound,int oct_id);
    void update_lod(bool include_root_bound);
    _FORCE_INLINE_ void clear_update_change_info();

    int get_points_count();
    int get_oct_id_points_count(int oct_id);
    PackedVector3Array get_tree_lines();

    void set_lod_setting(const PackedFloat32Array _lod_setting);
    void set_custom_capacity(int input);

    void set_debug_draw(bool input);
    bool get_debug_draw();
    
    void update_debug_gizmo();

    bool is_valid_octmesh_updater();
    static void thread_update(void* instance);
    void _notification(int p_what);
    void process_tick();
    void point_process_finished(int oct_id);
    void check_point_process_finished();
    void send_update_signal();
    Vector<PointUpdate> get_point_update(uint16_t oct_id);
    Array get_point_update_dictionary_array(int oct_id);

    void update_scenario();
    RID get_scenario();


};
#endif