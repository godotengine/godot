#ifndef __MCURVE_OVERRIDE
#define __MCURVE_OVERRIDE

#include "core/io/resource.h"



using namespace godot;

/*
    -1 -> means default mesh or material, whatever is that
    -2 -> means the mesh should be removed
*/

class MCurveMeshOverride : public Resource {
    GDCLASS(MCurveMeshOverride,Resource);
    protected:
    static void _bind_methods();
    public:
    struct Override
    {
        int material = -1;
        int mesh = -1;
        Override()=default;
        Override(int _material,int _mesh){
            material = _material;
            mesh = _mesh;
        }
    };
    

    private:
    HashMap<int64_t,Override> data;

    public:
    void set_mesh_override(int64_t id,int mesh);
    void set_material_override(int64_t id,int material);
    int get_mesh_override(int64_t id);
    int get_material_override(int64_t id);
    Override get_override(int64_t id);
    void clear_mesh_override(int64_t id);
    void clear_material_override(int64_t id);
    void clear_override(int64_t id);
    void clear();


    void set_data(const PackedByteArray& input);
    PackedByteArray get_data();
    
};
#endif