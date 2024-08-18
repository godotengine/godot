#ifndef _MCURVE
#define _MCURVE

// These two bellow should match in number
// Max conn should not be bigger than 127
#define MAX_CONN 4
#define conn_DEFAULT_VALUE {0,0,0,0}

#define INIT_POINTS_BUFFER_SIZE 10
#define INC_POINTS_BUFFER_SIZE 10
#define INVALID_POINT_INDEX 0
#define LENGTH_POINT_SAMPLE_COUNT_BASIC 1 // exclude start and end point only middle
#define INVALID_POINT_LOD -1 // Must be bigger than MAX_LOD in octree -- Only for internal reason outside lod=-1 is invalid

// RULE ---> LENGTH_POINT_SAMPLE_COUNT % DISTANCE_BAKE_INTERVAL = 0 ... Please change two bellow base on prev rule 
#define LENGTH_POINT_SAMPLE_COUNT 128
// Each 3 point which get sampled one bake lenght will be added
#define DISTANCE_BAKE_INTERVAL 4 // Total_point_interrval = LENGTH_POINT_SAMPLE_COUNT / DISTANCE_BAKE_INTERVAL
#define DISTANCE_BAKE_TOTAL (LENGTH_POINT_SAMPLE_COUNT/DISTANCE_BAKE_INTERVAL)
#define RATIO_BAKE_INTERVAL (1.0f/DISTANCE_BAKE_TOTAL)

#include "core/io/resource.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/variant/variant_utility.h"

#include "../moctree.h"

using namespace godot;

class MPath;
/*
    Each point has int32_t id
    Each has an array of next conn in conn!
    -------------- conn ----------------------
    Each conection can be interpolated in various ways
    In each bezier line interpolation between two point we use two position of Point A and B
    Then inside struct point A if in conn the B id is negated we use vector3 in in this interpolation!
    And if the B id is positive we use Vector3 out for this conn interpolation
    The same rule will apply inside B struct!
    -------------- conn unique ID ----------------------------
    Each conn will have unique ID which is define in Conn Union
    The way this Union defined it will generate one unique ID for each conn
    Each unique ID is defined as int64_t to also be consistate with Variant integer
    //////////////////////  INVALIDE ID INDEX RULE ///////////////////////////////////
    Due to nature of using negetive number we can not use point with id of 0
    So we use the point with id of 0 to be invalide point index
    In total id of 0 are not usable and they be left empty in points_buffer
*/  
class MCurve : public Resource{
    GDCLASS(MCurve,Resource);
    protected:
    static void _bind_methods();

    public:
    struct PointSave;
    struct Point // Index zero is considerd to be null
    {
        int8_t lod = INVALID_POINT_LOD;
        int32_t conn[MAX_CONN] = conn_DEFAULT_VALUE;
        float tilt = 0.0;
        float scale = 1.0;
        Vector3 up;
        Vector3 in;
        Vector3 out;
        Vector3 position;
        Point() = default;
        Point(Vector3 _position,Vector3 _in,Vector3 _out);
        PointSave get_point_save();
    };

    union Conn
    {
        struct {
            int32_t a;
            int32_t b;
        } p;
        int64_t id = 0;
        // p0 and p1 should be always positive and can't be equale
        Conn(int32_t p0, int32_t p1);
        Conn(int64_t _id);
        Conn() = default;
        ~Conn() = default;
        inline bool is_connection(){
            return p.b!=0;
        }
    };

    enum ConnType {
        CONN_NONE = 0,
        OUT_IN = 1,
        IN_OUT = 2,
        IN_IN = 3,
        OUT_OUT = 4
    };

    struct ConnUpdateInfo {
        int8_t last_lod;
        int8_t current_lod;
        int64_t conn_id;
    };

    struct PointUpdateInfo {
        int8_t last_lod;
        int8_t current_lod;
        int32_t point_id;
    };

    struct ConnDistances {
        float dis[DISTANCE_BAKE_TOTAL];
    };

    struct PointSave
    {
        int32_t prev;
        int32_t conn[MAX_CONN];
        float tilt;
        float scale;
        Vector3 in;
        Vector3 out;
        Vector3 position;
        Point get_point(){
            Point p;
            for(int8_t i=0; i < MAX_CONN; i++){
                p.conn[i] = conn[i];
            }
            p.tilt = tilt;
            p.scale = scale;
            p.in = in;
            p.out = out;
            p.position = position;
            return p;
        }
    };

    private:
    bool is_init_insert = false;
    bool is_waiting_for_user = false;
    int8_t active_lod_limit = 2;
    uint16_t oct_id = 0;
    PackedInt32Array free_buffer_indicies;
    void _increase_points_buffer_size(size_t q);
    //PackedInt32Array root_ids;
    static MOctree* octree;
    int32_t last_curve_id = 0;
    Vector<int32_t> curve_users;
    VSet<int32_t> processing_users;
    VSet<int32_t> active_points;
    VSet<int64_t> active_conn;
    //Vector<int32_t> force_reupdate_points;
    HashMap<int64_t,int8_t> conn_list; // Key -> conn, Value -> LOD
    HashMap<int64_t,ConnDistances> conn_distances;

    float bake_interval = 0.2;

    // Only editor and in debug version
    //#ifdef DEBUG_ENABLED
    HashMap<int64_t,PackedVector3Array> baked_lines;
    //#endif

    public:
    Vector<ConnUpdateInfo> conn_update;
    Vector<PointUpdateInfo> point_update;
    static void set_octree(MOctree* input);
    static MOctree* get_octree();
    Vector<MCurve::Point> points_buffer;
    MCurve();
    ~MCurve();

    int get_points_count();
    // Users
    int32_t get_curve_users_id();
    void remove_curve_user_id(int32_t user_id);

    // In case prev_conn = -1 this will insert as root node and if a root node exist this will give an error
    int32_t add_point(const Vector3& position,const Vector3& in,const Vector3& out, const int32_t prev_conn);
    int32_t add_point_conn_point(const Vector3& position,const Vector3& in,const Vector3& out,const Array& conn_types,const PackedInt32Array& conn_points);
    bool connect_points(int32_t p0,int32_t p1,ConnType con_type=CONN_NONE);
    bool disconnect_conn(int64_t conn_id);
    bool disconnect_points(int32_t p0,int32_t p1);
    void remove_point(const int32_t point_index);
    void clear_points();
    void init_insert();
    void _octree_update_finish();
    void user_finish_process(int32_t user_id);


    public:
    int64_t get_conn_id(int32_t p0, int32_t p1);
    PackedInt64Array get_conn_ids_exist(const PackedInt32Array points);
    int8_t get_conn_lod(int64_t conn_id);
    int8_t get_point_lod(int64_t p_id);
    PackedInt32Array get_active_points();
    PackedVector3Array get_active_points_positions();
    PackedInt64Array get_active_conns();
    PackedVector3Array get_conn_baked_points(int64_t input_conn);
    PackedVector3Array get_conn_baked_line(int64_t input_conn);

    public:
    void toggle_conn_type(int32_t point, int64_t conn_id);
    void validate_conn(int64_t conn_id,bool send_signal=true);
    void swap_points(const int32_t p_a,const int32_t p_b);
    void swap_points_with_validation(const int32_t p_a,const int32_t p_b);
    int32_t sort_from(int32_t root_point,bool increasing);
    void move_point(int p_index,const Vector3& pos);
    void move_point_in(int p_index,const Vector3& pos);
    void move_point_out(int p_index,const Vector3& pos);

    bool has_point(int p_index) const;
    bool has_conn(int64_t conn_id);
    ConnType get_conn_type(int64_t conn_id) const;
    Array get_point_conn_types(int32_t p_index) const;
    int get_point_conn_count(int32_t p_index) const;
    PackedInt32Array get_point_conn_points_exist(int32_t p_index) const;
    PackedInt32Array get_point_conn_points(int32_t p_index) const;
    PackedInt32Array get_point_conn_points_recursive(int32_t p_index) const;
    PackedInt64Array get_point_conns(int32_t p_index) const;
    PackedInt64Array get_point_conns_inc_neighbor_points(int32_t p_index) const;
    PackedInt64Array growed_conn(PackedInt64Array conn_ids) const;
    Vector3 get_point_position(int p_index);
    Vector3 get_point_in(int p_index);
    Vector3 get_point_out(int p_index);
    float get_point_tilt(int p_index);
    void set_point_tilt(int p_index,float input);
    float get_point_scale(int p_index);
    void set_point_scale(int p_index,float input);
    void commit_point_update(int p_index);
    void commit_conn_update(int64_t conn_id);


    public:
    /// Function bellow should be thread safe in case is called from another thread
    Vector3 get_conn_position(int64_t conn_id,float t);
    AABB get_conn_aabb(int64_t conn_id);
    AABB get_conns_aabb(const PackedInt64Array& conn_ids);
    float get_closest_ratio_to_point(int64_t conn_id,Vector3 pos);
    Vector3 get_point_order_tangent(int32_t point_a,int32_t point_b,float t);
    Vector3 get_conn_tangent(int64_t conn_id,float t);
    Transform3D get_point_order_transform(int32_t point_a,int32_t point_b,float t,bool tilt=true,bool scale=true);
    Transform3D get_conn_transform(int64_t conn_id,float t,bool apply_tilt=true,bool apply_scale=true);
    void get_conn_transforms(int64_t conn_id,const Vector<float>& t,Vector<Transform3D>& transforms,bool apply_tilt=true,bool apply_scale=true);
    float get_conn_lenght(int64_t conn_id);
    Pair<float,float> conn_ratio_limit_to_dis_limit(int64_t conn_id,const Pair<float,float>& limits);
    float get_point_order_distance_ratio(int32_t point_a,int32_t point_b,float distance);
    float get_conn_distance_ratio(int64_t conn_id,float distance);
    Pair<int,int> get_conn_distances_ratios(int64_t conn_id,const Vector<float>& distances,Vector<float>& t);
    private:
    _FORCE_INLINE_ float _get_conn_ratio_distance(const float* baked_dis,const float ratio) const;
    _FORCE_INLINE_ float _get_conn_distance_ratios(const float* baked_dis,const float distance) const;
    _FORCE_INLINE_ float* _bake_conn_distance(int64_t conn_id);
    // End of thread safe
    public:
    int32_t ray_active_point_collision(const Vector3& org,Vector3 dir,float threshold); // Maybe later optmize this
    void _set_data(const PackedByteArray& input);
    PackedByteArray _get_data();

    void set_bake_interval(float input);
    float get_bake_interval();
    void set_active_lod_limit(int input);
    int get_active_lod_limit();
    
    private:
    _FORCE_INLINE_ float  get_length_between_basic(const Point* a, const Point* b,const Vector3& a_control, const Vector3& b_control);
    
    #define BEZIER_EPSILON 0.1f
    _FORCE_INLINE_ Vector3 _get_bezier_extreme_t(const Vector3& a,const Vector3& b,const Vector3& a_control, const Vector3& b_control){
        return (2*a_control - (b_control + a))/(b - a + 3*(a_control - b_control));
    }
    _FORCE_INLINE_ Vector3 _get_bezier_tangent(const Vector3& a,const Vector3& b,const Vector3& a_control, const Vector3& b_control,const float t){

        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;
        float ut = u * t;

        Vector3 tangent;
        // Handling tangent zero points
        if( t < BEZIER_EPSILON && a.is_equal_approx(a_control)){
            Vector3 pos2 = a.bezier_interpolate(a_control,b_control,b,BEZIER_EPSILON);
            Vector3 pos = uu*u*a + 3*ut*u*a_control + 3*ut*t*b_control + tt*t*b;
            tangent = pos2 - pos;
        } else if( 1.0f - t < BEZIER_EPSILON && b.is_equal_approx(b_control)){
            Vector3 pos = uu*u*a + 3*ut*u*a_control + 3*ut*t*b_control + tt*t*b;
            Vector3 pos2 = a.bezier_interpolate(a_control,b_control,b,1.0f - BEZIER_EPSILON);
            tangent = pos - pos2;
        } else {
            tangent = 3*uu*(a_control - a) + 6*ut*(b_control - a_control) + 3*tt*(b - b_control);
        }
        tangent.normalize();
        return tangent;
    }
    // This function must not be called with completly straight Up_Vector line
    // It can handle if small part of line has a perpendiculare part
    _FORCE_INLINE_ Transform3D _get_bezier_transform(const Vector3& a,const Vector3& b,const Vector3& a_control, const Vector3& b_control,const Vector3& init_up_vec,float t){
        t = CLAMP(t, 0.0f, 1.0f);
        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;
        float ut = u * t;

        Vector3 pos = uu*u*a + 3*ut*u*a_control + 3*ut*t*b_control + tt*t*b;
        Vector3 tangent;
        Vector3 normal;
        // normal by derivative which does not work unfortunatly
        // normal = 6*u*(b_control - 2*a_control + a) + 6*t*(b - 2*b_control + a_control);

        // Handling tangent zero points
        if(unlikely(t < BEZIER_EPSILON && a.is_equal_approx(a_control))){
            Vector3 pos2 = a.bezier_interpolate(a_control,b_control,b,BEZIER_EPSILON);
            tangent = pos2 - pos;
        } else if(unlikely(1.0f - t < BEZIER_EPSILON && b.is_equal_approx(b_control))){
            Vector3 pos2 = a.bezier_interpolate(a_control,b_control,b,1.0f - BEZIER_EPSILON);
            tangent = pos - pos2;
        } else {
            tangent = 3*uu*(a_control - a) + 6*ut*(b_control - a_control) + 3*tt*(b - b_control);
        }
        tangent.normalize();
        // Handling small section Up_vector tangent
        if(unlikely(abs(tangent.y) > 0.999)){
            if(t > BEZIER_EPSILON * 10){
                Vector3 etangent = _get_bezier_tangent(a,b,a_control,b_control,t - BEZIER_EPSILON);
                normal = etangent.cross(init_up_vec);
            } else {
                Vector3 etangent = _get_bezier_tangent(a,b,a_control,b_control,t + BEZIER_EPSILON);
                normal = etangent.cross(init_up_vec);
            }
        } else {
            normal = tangent.cross(init_up_vec);
        }
        normal.normalize();

        Vector3 binormal = normal.cross(tangent);
        return Transform3D(tangent,binormal,normal,pos);
    }


};

VARIANT_ENUM_CAST(MCurve::ConnType);

#endif