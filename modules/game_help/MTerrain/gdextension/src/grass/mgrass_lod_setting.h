#ifndef MGRASSLODSETTING
#define MGRASSLODSETTING


#include "core/io/resource.h"
#include "core/math/vector3.h"

#include "scene/3d/multimesh_instance_3d.h"





class MGrassLodSetting : public Resource {
    GDCLASS(MGrassLodSetting,Resource);
    private:
    uint32_t _buffer_strid_float = 12;
    uint32_t _buffer_strid_byte = 48;
    bool _process_color_data = false;
    bool _process_custom_data = false;
    // ONLY DETERMINE IF IN ANY OF 
    bool _has_color_img;
    bool _has_custom_img;

    _FORCE_INLINE_ double rand_float(double a,double b,int seed);

    protected:
    static void _bind_methods();

    public:
    bool is_dirty=false;
    int seed=1001;
    int divide=1;
    int grass_in_cell=1;
    int force_lod_count=-1;
    Vector3 offset = Vector3(0,0,0);
    Vector3 rot_offset = Vector3(0,0,0);
    Vector3 rand_pos_start = Vector3(0,0,0);
    Vector3 rand_pos_end = Vector3(1,0,1);
    Vector3 rand_rot_start = Vector3(0,0,0);
    Vector3 rand_rot_end = Vector3(0,0,0);
    float unifrom_rand_scale_start=1.0;
    float unifrom_rand_scale_end=1.0;
    Vector3 rand_scale_start = Vector3(1,1,1);
    Vector3 rand_scale_end = Vector3(1,1,1);

    RenderingServer::ShadowCastingSetting shadow_setting = RenderingServer::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF;
    GeometryInstance3D::GIMode gi_mode = GeometryInstance3D::GIMode::GI_MODE_DISABLED;

    enum CUSTOM {
        RANDOM=0,
        IMAGE=1,
        CREATION_TIME=2
    };
    bool active_color_data=false;
    String color_img;
    int color_img_index=-1;
    Vector4 color_rand_start = Vector4(0,0,0,0);
    Vector4 color_rand_end = Vector4(1,1,1,1);
    CUSTOM color_r = RANDOM;
    CUSTOM color_g = RANDOM;
    CUSTOM color_b = RANDOM;
    CUSTOM color_a = RANDOM;
    bool active_custom_data=false;
    String custom_img;
    int custom_img_index=-1;
    Vector4 custom_rand_start = Vector4(0,0,0,0);
    Vector4 custom_rand_end = Vector4(1,1,1,1);
    CUSTOM custom_r = RANDOM;
    CUSTOM custom_g = RANDOM;
    CUSTOM custom_b = RANDOM;
    CUSTOM custom_a = RANDOM;

    uint32_t get_buffer_strid_float();
    uint32_t get_buffer_strid_byte();
    bool process_color_data();
    bool process_custom_data();
    bool has_color_img();
    bool has_custom_img();

    void set_seed(int input);
    int get_seed();

    void set_divide(int input);
    int get_divide();

    void set_grass_in_cell(int input);
    int get_grass_in_cell();

    void set_force_lod_count(int input);
    int get_force_lod_count();

    void set_offset(Vector3 input);
    Vector3 get_offset();

    void set_rot_offset(Vector3 input);
    Vector3 get_rot_offset();

    void set_rand_pos_start(Vector3 input);
    Vector3 get_rand_pos_start();

    void set_rand_pos_end(Vector3 input);
    Vector3 get_rand_pos_end();

    void set_rand_rot_start(Vector3 input);
    Vector3 get_rand_rot_start();

    void set_rand_rot_end(Vector3 input);
    Vector3 get_rand_rot_end();

    void set_uniform_rand_scale_start(float input);
    float get_uniform_rand_scale_start();

    void set_uniform_rand_scale_end(float input);
    float get_uniform_rand_scale_end();

    void set_rand_scale_start(Vector3 input);
    Vector3 get_rand_scale_start();

    void set_rand_scale_end(Vector3 input);
    Vector3 get_rand_scale_end();

    PackedFloat32Array generate_random_number(float density,int amount);


    //Geometry setting
    void set_shadow_setting(RenderingServer::ShadowCastingSetting input);
    RenderingServer::ShadowCastingSetting get_shadow_setting();
    void set_gi_mode(GeometryInstance3D::GIMode input);
    GeometryInstance3D::GIMode get_gi_mode();

    void set_active_color_data(bool input);
    bool get_active_color_data();
    void set_color_img(String input);
    String get_color_img();
    void set_color_rand_start(Vector4 input);
    Vector4 get_color_rand_start();
    void set_color_rand_end(Vector4 input);
    Vector4 get_color_rand_end();
    void set_color_r(CUSTOM input);
    CUSTOM get_color_r();
    void set_color_g(CUSTOM input);
    CUSTOM get_color_g();
    void set_color_b(CUSTOM input);
    CUSTOM get_color_b();
    void set_color_a(CUSTOM input);
    CUSTOM get_color_a();

    void set_active_custom_data(bool input);
    bool get_active_custom_data();
    void set_custom_img(String input);
    String get_custom_img();
    void set_custom_rand_start(Vector4 input);
    Vector4 get_custom_rand_start();
    void set_custom_rand_end(Vector4 input);
    Vector4 get_custom_rand_end();
    void set_custom_r(CUSTOM input);
    CUSTOM get_custom_r();
    void set_custom_g(CUSTOM input);
    CUSTOM get_custom_g();
    void set_custom_b(CUSTOM input);
    CUSTOM get_custom_b();
    void set_custom_a(CUSTOM input);
    CUSTOM get_custom_a();
};
VARIANT_ENUM_CAST(MGrassLodSetting::CUSTOM);
#endif