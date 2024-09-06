#include "mtool.h"
#include "core/io/marshalls.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "editor/editor_script.h"
#include "scene/main/viewport.h"
#include "scene/3d/camera_3d.h"


#include "editor/editor_interface.h"

bool MTool::editor_plugin_active = false;
Node3D* MTool::cached_editor_camera = nullptr;

void MTool::_bind_methods() {
   ClassDB::bind_static_method("MTool", D_METHOD("get_r16_image","file_path","width","height","min_height","max_height","is_half"), &MTool::get_r16_image);
   ClassDB::bind_static_method("MTool", D_METHOD("write_r16","file_path","data","min_height","max_height"), &MTool::write_r16);
   ClassDB::bind_static_method("MTool", D_METHOD("normalize_rf_data","data","min_height","max_height"), &MTool::normalize_rf_data);
   ClassDB::bind_static_method("MTool", D_METHOD("find_camera","changed_camera"), &MTool::find_editor_camera);
   ClassDB::bind_static_method("MTool", D_METHOD("enable_editor_plugin"), &MTool::enable_editor_plugin);
   ClassDB::bind_static_method("MTool", D_METHOD("ray_collision_y_zero_plane","ray_origin","ray"), &MTool::ray_collision_y_zero_plane);
}


MTool::MTool()
{
}

MTool::~MTool()
{
}


	static void _PackedByteArray_encode_u16(PackedByteArray *p_instance, int64_t p_offset, int64_t p_value) {
		uint64_t size = p_instance->size();
		ERR_FAIL_COND(p_offset < 0 || p_offset > int64_t(size) - 2);
		uint8_t *w = p_instance->ptrw();
		encode_uint16((uint16_t)p_value, &w[p_offset]);
	}

Ref<Image> MTool::get_r16_image(const String& file_path, const uint64_t width, const uint64_t height,double min_height, double max_height,const bool is_half) {
    Ref<Image> img;
    VariantUtilityFunctions::_print("open: ", file_path);
    if(!FileAccess::exists(file_path)){
        ERR_FAIL_COND_V("File does not exist check your file path again", img);
    }
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if(file->get_error() != OK){
        ERR_FAIL_COND_V("Can not open the file", img);
    }
    uint64_t final_width = width;
    uint64_t final_height = height;
    uint64_t size = file->get_length();
    uint64_t size16 = size/2;
    if(width==0 || height==0){
        final_width = sqrt(size16);
        if(final_width*final_width*2 != size){
            ERR_FAIL_COND_V("Image width is not valid please set width and height", img);
        }
        final_height = final_width;
    } else {
        if(width*height != size16){
            ERR_FAIL_COND_V("Image width or height is not valid", img);
        }
    }
    if(is_half){
        PackedByteArray data;
        data.resize(size);
        uint64_t offset = 0;
        for(int i = 0; i<size16; i++){
            double p = (double)file->get_16()/65535;
            p *= (max_height - min_height);
            p += min_height;
            _PackedByteArray_encode_u16(&data,offset, p);
            offset += 2;
        }
        img = Image::create_from_data(final_width,final_height,false, Image::FORMAT_RH, data);
    } else {
        PackedFloat32Array dataf;
        for(int i = 0; i<size16; i++){
            double p = (double)file->get_16()/65535;
            p *= (max_height - min_height);
            p += min_height;
            dataf.append(p);
        }
        img = Image::create_from_data(final_width,final_height,false, Image::FORMAT_RF, dataf.to_byte_array());
    }
    return img;
}

void MTool::write_r16(const String& file_path,const PackedByteArray& data,double min_height,double max_height){
    ERR_FAIL_COND(data.size()%4!=0);
    double dh = max_height - min_height;
    ERR_FAIL_COND(dh<0);
    Ref<FileAccess> file = FileAccess::open(file_path,FileAccess::ModeFlags::WRITE);
    const float* ptr = (float*)data.ptr();
    uint32_t size = data.size()/4;

    for(uint32_t i=0;i<size;i++){
        double val = (ptr[i] - min_height)/dh;
        if (val < 0){
            val = 0;
        }
        val *= 65535;
        if(val > 65535){
            val = 65535;
        }
        uint16_t u16 = val;
        file->store_16(u16);
    }
    file->close();
}

PackedByteArray MTool::normalize_rf_data(const PackedByteArray& data,double min_height,double max_height){
    ERR_FAIL_COND_V(data.size()%4!=0,data);
    double dh = max_height - min_height;
    ERR_FAIL_COND_V(dh<0,data);
    const float* ptr = (const float*)data.ptr();
    PackedByteArray out;
    out.resize(data.size());
    float* ptrw = (float*)out.ptrw();
    uint32_t size = data.size()/4;
    for(uint32_t i=0;i<size;i++){
        double val = (ptr[i] - min_height)/dh;
        if (val < 0){
            val = 0;
        }
        if(val>1.0){
            val = 1.0;
        }
        ptrw[i] = val;
    }
    return out;
}

Node3D* MTool::find_editor_camera(bool changed_camera){
    if(cached_editor_camera!=nullptr){
        return cached_editor_camera;
    }
	if (Engine::get_singleton()->is_editor_hint()) {
        Ref<EditorScript> script;
        script.instantiate();
        SubViewport* sub_viewport = script->get_editor_interface()->get_editor_viewport_3d(0);
        if(sub_viewport==nullptr){
            return nullptr;
        }
        cached_editor_camera = sub_viewport->get_camera_3d();
        return cached_editor_camera;
	}
    return nullptr;
}

void MTool::enable_editor_plugin(){
    editor_plugin_active = true;
}

bool MTool::is_editor_plugin_active(){
    return editor_plugin_active;
}

Ref<MCollision> MTool::ray_collision_y_zero_plane(const Vector3& ray_origin,const Vector3& ray){
    Ref<MCollision> col;
    col.instantiate();
    if(ray.y > -0.001){
        return col;
    }
    col->collision_position = ray_origin - (ray_origin.y/ray.y)*ray;
    col->collided = true;
    return col;
}
