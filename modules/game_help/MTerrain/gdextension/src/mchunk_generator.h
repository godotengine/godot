#ifndef MCHUNK_GENERATOR
#define MCHUNK_GENERATOR


#include "core/object/object.h"
#include "scene/resources/mesh.h"
#include "core/object/ref_counted.h"



class MChunkGenerator : public Object{
    GDCLASS(MChunkGenerator, Object);

    protected:
    static void _bind_methods();

    public:
    static Ref<ArrayMesh> generate(real_t size, real_t h_scale, bool el, bool er, bool et, bool eb);

};




#endif