#ifndef FOLIAGE_MANAGER_H
#define FOLIAGE_MANAGER_H

#include "foliage_engine.h"

#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"


namespace Foliage
{
    class FoliageManager : public Node3D
    {
        GDCLASS(FoliageManager, Node3D);
        static void _bind_methods()
        {

        }
    public:
        FoliageManager();
        ~FoliageManager();
        void init(TypedArray<FoliageMapChunkConfig> map_config);
        void clear();
        void set_camera(Camera3D* p_camera);
        void _notification(int p_what);
        void update();
    private:
    };
}

#endif