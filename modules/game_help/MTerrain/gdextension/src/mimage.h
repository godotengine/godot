#ifndef MIMAGE
#define MIMAGE

#include "mconfig.h"

#include <mutex>
#include <thread>
#include <chrono>

#include "core/io/resource_loader.h"
#include "core/templates/vector.h"
#include "core/math/color.h"
#include "core/io/image.h"
#include "scene/resources/image_texture.h"
#include "core/templates/hash_map.h"

#include "mbound.h"
#include "mresource.h"


using namespace godot;

class MRegion;

struct MImageUndoData {
    int layer;
    bool empty=false;// In case the layer is empty at this data
    uint8_t* data;

    void free(){
        if(!empty)
            memdelete_arr(data);
    }
};

struct MImage {
public:
    int index=-1;
    MRegion* region=nullptr;
    StringName name;
    String uniform_name;
    int compression=-1;
    uint32_t width;
    uint32_t height;
    uint32_t current_size;
    uint32_t current_scale = 1;
    uint32_t pixel_size;
    uint32_t total_pixel_amount;
    Image::Format format = Image::Format::FORMAT_MAX; //Setting an invalid format so in case it is not set we can generate error
    PackedByteArray data;
    #ifdef M_IMAGE_LAYER_ON
    int active_layer=0;
    int holes_layer=-1;
    PackedStringArray layer_names;
    Vector<PackedByteArray*> image_layers;
    Vector<bool> is_saved_layers;
    #endif
    RID old_tex_rid;
    RID new_tex_rid;
    bool has_texture_to_apply = false;
    bool is_dirty = false;
    bool is_save = false;
    MGridPos grid_pos;
    std::mutex update_mutex;
    std::recursive_mutex load_mutex;//Any method which read/write the data or layer data exept for pixel modification as that would be expensive, for pixel modifcation we should do some higher level lock 
    bool active_undo=false;
    int current_undo_id;
    // Key is undo redo id
    HashMap<int,MImageUndoData> undo_data;
    bool is_init=false;
    bool is_corrupt_file = false;
    bool is_null_image=true;
    bool is_ram_image=false; // in case the image exist only on RAM not VRAM
    
    MImage();
    MImage(const String& _name,const String& _uniform_name,MGridPos _grid_pos,MRegion* r);
    ~MImage();
    void load(Ref<MResource> mres);
    void unload(Ref<MResource> mres);
    void set_active_layer(int l);
    void add_layer(String lname);
    void rename_layer(int layer_index,String new_name);
    void merge_layer();
    void remove_layer(bool is_visible);
    void layer_visible(bool input);
    void create(uint32_t _size, Image::Format _format);
    // This create bellow should not be used for terrain, It is for other stuff
    void create(uint32_t _width,uint32_t _height, Image::Format _format);
    // get data with custom scale
    void get_data(PackedByteArray* out,int scale);
    void update_texture(int scale,bool apply_update);
    void apply_update();
    // This works only for Format_RF
    real_t get_pixel_RF(const uint32_t x, const uint32_t  y) const;
    void set_pixel_RF(const uint32_t x, const uint32_t  y,const real_t value);
    real_t get_pixel_RF_in_layer(const uint32_t x, const uint32_t  y);
    Color get_pixel(const uint32_t x, const uint32_t  y) const;
    void set_pixel(const uint32_t x, const uint32_t  y,const Color& color);
    void set_pixel_by_data_pointer(uint32_t x,uint32_t y,uint8_t* ptr);
    const uint8_t* get_pixel_by_data_pointer(uint32_t x,uint32_t y);
    bool save(Ref<MResource> mres,bool force_save);
    void check_undo(); // Register the state of image before the draw
    void remove_undo_data(int ur_id);
    void remove_undo_data_in_layer(int layer_index);
    bool go_to_undo(int ur_id);
    bool has_undo(int ur_id);

    // This functions exist in godot source code
	_FORCE_INLINE_ Color _get_color_at_ofs(const uint8_t *ptr, uint32_t ofs) const;
	_FORCE_INLINE_ void _set_color_at_ofs(uint8_t *ptr, uint32_t ofs, const Color &p_color);
    static int get_format_pixel_size(Image::Format p_format);
    static int get_format_uint_channel_count(Image::Format p_format);

    void set_pixel_in_channel(const uint32_t x, const uint32_t  y,int8_t channel,const float value);
    float get_pixel_in_channel(const uint32_t x, const uint32_t  y,int8_t channel);


    private:
    void load_layer(String lname);
    _FORCE_INLINE_ String get_layer_data_dir();
};

#endif