#include "mgrass_lod_setting.h"

#include "core/math/transform_3d.h"
#include "core/math/basis.h"
#include <cstdlib>
#include "core/math/random_number_generator.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"


void MGrassLodSetting::_bind_methods() {
    ADD_SIGNAL(MethodInfo("lod_setting_changed"));
    ClassDB::bind_method(D_METHOD("set_seed","input"), &MGrassLodSetting::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &MGrassLodSetting::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"seed"),"set_seed","get_seed");
    ClassDB::bind_method(D_METHOD("set_grid_lod","input"), &MGrassLodSetting::set_grid_lod);
    ClassDB::bind_method(D_METHOD("get_grid_lod"), &MGrassLodSetting::get_grid_lod);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"grid_lod"),"set_grid_lod","get_grid_lod");
    ClassDB::bind_method(D_METHOD("set_multimesh_subdivisions","input"), &MGrassLodSetting::set_multimesh_subdivisions);
    ClassDB::bind_method(D_METHOD("get_multimesh_subdivisions"), &MGrassLodSetting::get_multimesh_subdivisions);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"multimesh_subdivisions"),"set_multimesh_subdivisions","get_multimesh_subdivisions");

    ADD_GROUP("Grass Setting","");
    ClassDB::bind_method(D_METHOD("set_cell_instance_count","input"), &MGrassLodSetting::set_cell_instance_count);
    ClassDB::bind_method(D_METHOD("get_cell_instance_count"), &MGrassLodSetting::get_cell_instance_count);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"cell_instance_count"),"set_cell_instance_count","get_cell_instance_count");
    ClassDB::bind_method(D_METHOD("set_offset","input"), &MGrassLodSetting::set_offset);
    ClassDB::bind_method(D_METHOD("get_offset"), &MGrassLodSetting::get_offset);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"offset"),"set_offset","get_offset");
    ClassDB::bind_method(D_METHOD("set_rot_offset","input"), &MGrassLodSetting::set_rot_offset);
    ClassDB::bind_method(D_METHOD("get_rot_offset"), &MGrassLodSetting::get_rot_offset);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rot_offset"),"set_rot_offset","get_rot_offset");
    ClassDB::bind_method(D_METHOD("set_rand_pos_start","input"), &MGrassLodSetting::set_rand_pos_start);
    ClassDB::bind_method(D_METHOD("get_rand_pos_start"), &MGrassLodSetting::get_rand_pos_start);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_pos_start"),"set_rand_pos_start","get_rand_pos_start");
    ClassDB::bind_method(D_METHOD("set_rand_pos_end","input"), &MGrassLodSetting::set_rand_pos_end);
    ClassDB::bind_method(D_METHOD("get_rand_pos_end"), &MGrassLodSetting::get_rand_pos_end);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_pos_end"),"set_rand_pos_end","get_rand_pos_end");
    ClassDB::bind_method(D_METHOD("set_rand_rot_start","input"), &MGrassLodSetting::set_rand_rot_start);
    ClassDB::bind_method(D_METHOD("get_rand_rot_start"), &MGrassLodSetting::get_rand_rot_start);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_rot_start"),"set_rand_rot_start","get_rand_rot_start");
    ClassDB::bind_method(D_METHOD("set_rand_rot_end","input"), &MGrassLodSetting::set_rand_rot_end);
    ClassDB::bind_method(D_METHOD("get_rand_rot_end"), &MGrassLodSetting::get_rand_rot_end);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_rot_end"),"set_rand_rot_end","get_rand_rot_end");
    ClassDB::bind_method(D_METHOD("set_uniform_rand_scale_start","input"), &MGrassLodSetting::set_uniform_rand_scale_start);
    ClassDB::bind_method(D_METHOD("get_uniform_rand_scale_start"), &MGrassLodSetting::get_uniform_rand_scale_start);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"uniform_rand_scale_start"),"set_uniform_rand_scale_start","get_uniform_rand_scale_start");
    ClassDB::bind_method(D_METHOD("set_uniform_rand_scale_end","input"), &MGrassLodSetting::set_uniform_rand_scale_end);
    ClassDB::bind_method(D_METHOD("get_uniform_rand_scale_end"), &MGrassLodSetting::get_uniform_rand_scale_end);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"uniform_rand_scale_end"),"set_uniform_rand_scale_end","get_uniform_rand_scale_end");
    ClassDB::bind_method(D_METHOD("set_rand_scale_start","input"), &MGrassLodSetting::set_rand_scale_start);
    ClassDB::bind_method(D_METHOD("get_rand_scale_start"), &MGrassLodSetting::get_rand_scale_start);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_scale_start"),"set_rand_scale_start","get_rand_scale_start");
    ClassDB::bind_method(D_METHOD("set_rand_scale_end","input"), &MGrassLodSetting::set_rand_scale_end);
    ClassDB::bind_method(D_METHOD("get_rand_scale_end"), &MGrassLodSetting::get_rand_scale_end);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3,"rand_scale_end"),"set_rand_scale_end","get_rand_scale_end");

    ADD_GROUP("Geometry Setting","");
    ClassDB::bind_method(D_METHOD("set_shadow_setting","input"), &MGrassLodSetting::set_shadow_setting);
    ClassDB::bind_method(D_METHOD("get_shadow_setting"), &MGrassLodSetting::get_shadow_setting);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"shadow_setting",PropertyHint::PROPERTY_HINT_ENUM,"OFF,ON,DOUBLE_SIDED,SHADOWS_ONLY"),"set_shadow_setting","get_shadow_setting");

    ClassDB::bind_method(D_METHOD("set_gi_mode","input"), &MGrassLodSetting::set_gi_mode);
    ClassDB::bind_method(D_METHOD("get_gi_mode"), &MGrassLodSetting::get_gi_mode);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"gi_mode",PropertyHint::PROPERTY_HINT_ENUM,"Disabled,Static (VoxelGI/SDFGI/LightmapGI),Dynamic (VoxelGI only)"),"set_gi_mode","get_gi_mode");

    BIND_ENUM_CONSTANT(RANDOM);
    BIND_ENUM_CONSTANT(IMAGE);
    BIND_ENUM_CONSTANT(CREATION_TIME);

    ADD_GROUP("color data","");
    ClassDB::bind_method(D_METHOD("set_active_color_data","input"), &MGrassLodSetting::set_active_color_data);
    ClassDB::bind_method(D_METHOD("get_active_color_data"), &MGrassLodSetting::get_active_color_data);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"active_color_data"),"set_active_color_data","get_active_color_data");

    ClassDB::bind_method(D_METHOD("set_color_img","input"), &MGrassLodSetting::set_color_img);
    ClassDB::bind_method(D_METHOD("get_color_img"), &MGrassLodSetting::get_color_img);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"color_img"),"set_color_img","get_color_img");

    ClassDB::bind_method(D_METHOD("set_color_rand_start","input"), &MGrassLodSetting::set_color_rand_start);
    ClassDB::bind_method(D_METHOD("get_color_rand_start"), &MGrassLodSetting::get_color_rand_start);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR4,"color_rand_start"),"set_color_rand_start","get_color_rand_start");
    ClassDB::bind_method(D_METHOD("set_color_rand_end","input"), &MGrassLodSetting::set_color_rand_end);
    ClassDB::bind_method(D_METHOD("get_color_rand_end"), &MGrassLodSetting::get_color_rand_end);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR4,"color_rand_end"),"set_color_rand_end","get_color_rand_end");

    ClassDB::bind_method(D_METHOD("set_color_r","input"), &MGrassLodSetting::set_color_r);
    ClassDB::bind_method(D_METHOD("get_color_r"), &MGrassLodSetting::get_color_r);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"color_r",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_R,CREATION_TIME"),"set_color_r","get_color_r");
    ClassDB::bind_method(D_METHOD("set_color_g","input"), &MGrassLodSetting::set_color_g);
    ClassDB::bind_method(D_METHOD("get_color_g"), &MGrassLodSetting::get_color_g);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"color_g",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_G,CREATION_TIME"),"set_color_g","get_color_g");
    ClassDB::bind_method(D_METHOD("set_color_b","input"), &MGrassLodSetting::set_color_b);
    ClassDB::bind_method(D_METHOD("get_color_b"), &MGrassLodSetting::get_color_b);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"color_b",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_B,CREATION_TIME"),"set_color_b","get_color_b");
    ClassDB::bind_method(D_METHOD("set_color_a","input"), &MGrassLodSetting::set_color_a);
    ClassDB::bind_method(D_METHOD("get_color_a"), &MGrassLodSetting::get_color_a);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"color_a",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_A,CREATION_TIME"),"set_color_a","get_color_a");

    ADD_GROUP("custom data","");
    ClassDB::bind_method(D_METHOD("set_active_custom_data","input"), &MGrassLodSetting::set_active_custom_data);
    ClassDB::bind_method(D_METHOD("get_active_custom_data"), &MGrassLodSetting::get_active_custom_data);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"active_custom_data"),"set_active_custom_data","get_active_custom_data");

    ClassDB::bind_method(D_METHOD("set_custom_img","input"), &MGrassLodSetting::set_custom_img);
    ClassDB::bind_method(D_METHOD("get_custom_img"), &MGrassLodSetting::get_custom_img);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"custom_img"),"set_custom_img","get_custom_img");

    ClassDB::bind_method(D_METHOD("set_custom_rand_start","input"), &MGrassLodSetting::set_custom_rand_start);
    ClassDB::bind_method(D_METHOD("get_custom_rand_start"), &MGrassLodSetting::get_custom_rand_start);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR4,"custom_rand_start"),"set_custom_rand_start","get_custom_rand_start");
    ClassDB::bind_method(D_METHOD("set_custom_rand_end","input"), &MGrassLodSetting::set_custom_rand_end);
    ClassDB::bind_method(D_METHOD("get_custom_rand_end"), &MGrassLodSetting::get_custom_rand_end);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR4,"custom_rand_end"),"set_custom_rand_end","get_custom_rand_end");


    ClassDB::bind_method(D_METHOD("set_custom_r","input"), &MGrassLodSetting::set_custom_r);
    ClassDB::bind_method(D_METHOD("get_custom_r"), &MGrassLodSetting::get_custom_r);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"custom_r",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_R,CREATION_TIME"),"set_custom_r","get_custom_r");
    ClassDB::bind_method(D_METHOD("set_custom_g","input"), &MGrassLodSetting::set_custom_g);
    ClassDB::bind_method(D_METHOD("get_custom_g"), &MGrassLodSetting::get_custom_g);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"custom_g",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_G,CREATION_TIME"),"set_custom_g","get_custom_g");
    ClassDB::bind_method(D_METHOD("set_custom_b","input"), &MGrassLodSetting::set_custom_b);
    ClassDB::bind_method(D_METHOD("get_custom_b"), &MGrassLodSetting::get_custom_b);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"custom_b",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_B,CREATION_TIME"),"set_custom_b","get_custom_b");
    ClassDB::bind_method(D_METHOD("set_custom_a","input"), &MGrassLodSetting::set_custom_a);
    ClassDB::bind_method(D_METHOD("get_custom_a"), &MGrassLodSetting::get_custom_a);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"custom_a",PROPERTY_HINT_ENUM,"RANDOM,IMAGE_A,CREATION_TIME"),"set_custom_a","get_custom_a");

}

uint32_t MGrassLodSetting::get_buffer_strid_float(){
    return _buffer_strid_float;
}
uint32_t MGrassLodSetting::get_buffer_strid_byte(){
    return _buffer_strid_byte;
}

bool MGrassLodSetting::process_color_data(){
    return _process_color_data;
}

bool MGrassLodSetting::process_custom_data(){
    return _process_custom_data;
}

bool MGrassLodSetting::has_color_img(){
    return _has_color_img;
}
bool MGrassLodSetting::has_custom_img(){
    return _has_custom_img;
}

void MGrassLodSetting::set_seed(int input){
    ERR_FAIL_COND(input<0);
    seed = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
int MGrassLodSetting::get_seed(){
    return seed;
}

void MGrassLodSetting::set_multimesh_subdivisions(int input){
    ERR_FAIL_COND(input<1);
    multimesh_subdivisions = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
int MGrassLodSetting::get_multimesh_subdivisions(){
    return multimesh_subdivisions;
}

void MGrassLodSetting::set_cell_instance_count(int input){
    ERR_FAIL_COND(input<1);
    cell_instance_count = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
int MGrassLodSetting::get_cell_instance_count(){
    return cell_instance_count;
}

void MGrassLodSetting::set_grid_lod(int input){
    grid_lod = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
int MGrassLodSetting::get_grid_lod(){
    return grid_lod;
}

void MGrassLodSetting::set_offset(Vector3 input){
    offset = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_offset(){
    return offset;
}

void MGrassLodSetting::set_rot_offset(Vector3 input){
    rot_offset = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rot_offset(){
    return rot_offset;
}

void MGrassLodSetting::set_rand_pos_start(Vector3 input){
    rand_pos_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_pos_start(){
    return rand_pos_start;
}

void MGrassLodSetting::set_rand_pos_end(Vector3 input){
    rand_pos_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_pos_end(){
    return rand_pos_end;
}

void MGrassLodSetting::set_rand_rot_start(Vector3 input){
    rand_rot_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_rot_start(){
    return rand_rot_start;
}

void MGrassLodSetting::set_rand_rot_end(Vector3 input){
    rand_rot_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_rot_end(){
    return rand_rot_end;
}

void MGrassLodSetting::set_uniform_rand_scale_start(float input){
    unifrom_rand_scale_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
float MGrassLodSetting::get_uniform_rand_scale_start(){
    return unifrom_rand_scale_start;
}

void MGrassLodSetting::set_uniform_rand_scale_end(float input){
    unifrom_rand_scale_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}

float MGrassLodSetting::get_uniform_rand_scale_end(){
    return unifrom_rand_scale_end;
}

void MGrassLodSetting::set_rand_scale_start(Vector3 input){
    rand_scale_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_scale_start(){
    return rand_scale_start;
}

void MGrassLodSetting::set_rand_scale_end(Vector3 input){
    rand_scale_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector3 MGrassLodSetting::get_rand_scale_end(){
    return rand_scale_end;
}

double MGrassLodSetting::rand_float(double a,double b,int seed){
   Ref<RandomNumberGenerator> rand;
   rand.instantiate();
   rand->set_seed(seed);
   return rand->randf_range(a,b);
}



PackedFloat32Array MGrassLodSetting::generate_random_number(float density,int amount){
    // if data contain only random number grass should not worry about that 
    _process_color_data = color_r!=RANDOM || color_g!=RANDOM || color_b!=RANDOM || color_a!=RANDOM;
    _process_custom_data = custom_r!=RANDOM || custom_g!=RANDOM || custom_b!=RANDOM || custom_a!=RANDOM;
    _process_color_data = _process_color_data && active_color_data;
     _process_custom_data = _process_custom_data && active_custom_data;
    _has_color_img = (color_r==IMAGE || color_g==IMAGE || color_b==IMAGE || color_a==IMAGE) && !color_img.is_empty();
    _has_custom_img = (custom_r==IMAGE || custom_g==IMAGE || custom_b==IMAGE || custom_a==IMAGE) && !custom_img.is_empty();


    _buffer_strid_float = 12 + int(active_color_data)*4 + int(active_custom_data)*4;
    _buffer_strid_byte = _buffer_strid_float*4;

    PackedFloat32Array out;
    out.resize(amount * _buffer_strid_float);
    for(int i=0;i<amount;i++){
        Vector3 _rand_pos;
        Vector3 _rand_rot;
        Vector3 _rand_scale;
        float uniform_scale;
        int iseed = i*10+seed;
        _rand_pos.x = rand_float(rand_pos_start.x*density,rand_pos_end.x*density,iseed);
        iseed++;
        _rand_pos.y = rand_float(rand_pos_start.y*density,rand_pos_end.y*density,iseed);
        iseed++;
        _rand_pos.z = rand_float(rand_pos_start.z*density,rand_pos_end.z*density,iseed);
        iseed++;
        _rand_rot.x = rand_float(rand_rot_start.x,rand_rot_end.x,iseed);
        iseed++;
        _rand_rot.y = rand_float(rand_rot_start.y,rand_rot_end.y,iseed);
        iseed++;
        _rand_rot.z = rand_float(rand_rot_start.z,rand_rot_end.z,iseed);
        iseed++;
        _rand_scale.x = rand_float(rand_scale_start.x,rand_scale_end.x,iseed);
        iseed++;
        _rand_scale.y = rand_float(rand_scale_start.y,rand_scale_end.y,iseed);
        iseed++;
        _rand_scale.z = rand_float(rand_scale_start.z,rand_scale_end.z,iseed);
        iseed++;
        uniform_scale = rand_float(unifrom_rand_scale_start,unifrom_rand_scale_end,iseed);
        // stage 2
        Basis b;
        b.scale(Vector3(uniform_scale,uniform_scale,uniform_scale));
        b.scale(_rand_scale);
        b.rotate(Vector3(0,1,0),VariantUtilityFunctions::deg_to_rad(rot_offset.y));
        b.rotate(Vector3(1,0,0),VariantUtilityFunctions::deg_to_rad(rot_offset.x));
        b.rotate(Vector3(0,0,1),VariantUtilityFunctions::deg_to_rad(rot_offset.z));
        // Rotation order YXZ
        b.rotate(Vector3(0,1,0),VariantUtilityFunctions::deg_to_rad(_rand_rot.y));
        b.rotate(Vector3(1,0,0),VariantUtilityFunctions::deg_to_rad(_rand_rot.x));
        b.rotate(Vector3(0,0,1),VariantUtilityFunctions::deg_to_rad(_rand_rot.z));


        Vector3 org = offset + _rand_pos;
        Transform3D t(b,org);

        
        int index = i*get_buffer_strid_float();

        out.set(index,t.basis[0][0]);
        out.set(index+1,t.basis[0][1]);
        out.set(index+2,t.basis[0][2]);
        out.set(index+3,t.origin[0]);

        out.set(index+4,t.basis[1][0]);
        out.set(index+5,t.basis[1][1]);
        out.set(index+6,t.basis[1][2]);
        out.set(index+7,t.origin[1]);

        out.set(index+8,t.basis[2][0]);
        out.set(index+9,t.basis[2][1]);
        out.set(index+10,t.basis[2][2]);
        out.set(index+11,t.origin[2]);

        index += 12;

        //Setting Color data
        if(active_color_data){
            int cseed = i*4 + seed + seed;
            out.set(index,rand_float(color_rand_start.x,color_rand_end.x,cseed));
            index++; cseed++;
            out.set(index,rand_float(color_rand_start.y,color_rand_end.y,cseed));
            index++; cseed++;
            out.set(index,rand_float(color_rand_start.z,color_rand_end.z,cseed));
            index++; cseed++;
            out.set(index,rand_float(color_rand_start.w,color_rand_end.w,cseed));
            index++;
        }
        if(active_custom_data){
            int cseed = i*4 + seed + seed + seed;
            out.set(index,rand_float(custom_rand_start.x,custom_rand_end.x,cseed));
            index++; cseed++;
            out.set(index,rand_float(custom_rand_start.y,custom_rand_end.y,cseed));
            index++; cseed++;
            out.set(index,rand_float(custom_rand_start.z,custom_rand_end.z,cseed));
            index++; cseed++;
            out.set(index,rand_float(custom_rand_start.w,custom_rand_end.w,cseed));
        }
    }
    return out;
}


void MGrassLodSetting::set_shadow_setting(RenderingServer::ShadowCastingSetting input){
    shadow_setting = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
RenderingServer::ShadowCastingSetting MGrassLodSetting::get_shadow_setting(){
    return shadow_setting;
}

void MGrassLodSetting::set_gi_mode(GeometryInstance3D::GIMode input){
    gi_mode = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}

GeometryInstance3D::GIMode MGrassLodSetting::get_gi_mode(){
    return gi_mode;
}

void MGrassLodSetting::set_active_color_data(bool input){
    active_color_data = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
bool MGrassLodSetting::get_active_color_data(){
    return active_color_data;
}

void MGrassLodSetting::set_color_img(String input){
    color_img = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
String MGrassLodSetting::get_color_img(){
    return color_img;
}

void MGrassLodSetting::set_color_rand_start(Vector4 input){
    color_rand_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector4 MGrassLodSetting::get_color_rand_start(){
    return color_rand_start;
}
void MGrassLodSetting::set_color_rand_end(Vector4 input){
    color_rand_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector4 MGrassLodSetting::get_color_rand_end(){
    return color_rand_end;
}

void MGrassLodSetting::set_color_r(MGrassLodSetting::CUSTOM input){
    color_r = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_color_r(){
    return color_r;
}
void MGrassLodSetting::set_color_g(MGrassLodSetting::CUSTOM input){
    color_g = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_color_g(){
    return color_g;
}
void MGrassLodSetting::set_color_b(MGrassLodSetting::CUSTOM input){
    color_b = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_color_b(){
    return color_b;
}
void MGrassLodSetting::set_color_a(MGrassLodSetting::CUSTOM input){
    color_a = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_color_a(){
    return color_a;
}


void MGrassLodSetting::set_active_custom_data(bool input){
    active_custom_data = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
bool MGrassLodSetting::get_active_custom_data(){
    return active_custom_data;
}

void MGrassLodSetting::set_custom_img(String input){
    custom_img = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
String MGrassLodSetting::get_custom_img(){
    return custom_img;
}

void MGrassLodSetting::set_custom_rand_start(Vector4 input){
    custom_rand_start = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector4 MGrassLodSetting::get_custom_rand_start(){
    return custom_rand_start;
}
void MGrassLodSetting::set_custom_rand_end(Vector4 input){
    custom_rand_end = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
Vector4 MGrassLodSetting::get_custom_rand_end(){
    return custom_rand_end;
}

void MGrassLodSetting::set_custom_r(MGrassLodSetting::CUSTOM input){
    custom_r = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_custom_r(){
    return custom_r;
}
void MGrassLodSetting::set_custom_g(MGrassLodSetting::CUSTOM input){
    custom_g = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_custom_g(){
    return custom_g;
}
void MGrassLodSetting::set_custom_b(MGrassLodSetting::CUSTOM input){
    custom_b = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_custom_b(){
    return custom_b;
}
void MGrassLodSetting::set_custom_a(MGrassLodSetting::CUSTOM input){
    custom_a = input;
    is_dirty = true;
    emit_signal("lod_setting_changed");
}
MGrassLodSetting::CUSTOM MGrassLodSetting::get_custom_a(){
    return custom_a;
}

bool MGrassLodSetting::_set(const StringName &p_name, const Variant &p_value){
    /// Old name compatibity stuff
    if(p_name==String("grass_in_cell")){
        cell_instance_count = p_value;
        return true;
    }
    if (p_name==String("force_lod_count")){
        grid_lod = p_value;
        return true;
    }
    if (p_name==String("divide")){
        multimesh_subdivisions = p_value;
        return true;
    }
    return false;
}