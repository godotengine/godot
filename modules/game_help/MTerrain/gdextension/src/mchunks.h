#ifndef MCHUNKS
#define MCHUNKS

#include "mconfig.h"

#include "core/object/object.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"
#include "scene/resources/mesh.h"
#include "core/object/ref_counted.h"
#include "core/variant/array.h"
#include "scene/resources/material.h"




struct MLod{
    real_t h_scale;
    int8_t lod;
    Vector<RID> meshes;
};

struct MSize
{
    int8_t size;
    int32_t size_meter;
    Vector<MLod> lods;
};




class MChunks : public Object {
    GDCLASS(MChunks, Object);
    private:
    static int number_of_user;
    static void remove_user();
    static Vector<Ref<Mesh>> meshes;
    static HashMap<int64_t,int> mesh_hash;
    static RID get_mesh(int32_t size_meter, real_t h_scale, int8_t edge,const Ref<Material>& _material);
    


    protected:
    static void _bind_methods();

    public:
    #ifdef M_DEBUG
    Vector<Ref<Material>> debug_material;
    #endif
    real_t h_scale;
    int32_t base_size_meter;
    int8_t max_lod;
    int8_t max_size;
    Vector<MSize> sizes;
    MChunks();
    ~MChunks();
    void create_chunks(int32_t _min_size, int32_t _max_size, real_t _min_h_scale, real_t _max_h_scale, Array _info);
};
#endif