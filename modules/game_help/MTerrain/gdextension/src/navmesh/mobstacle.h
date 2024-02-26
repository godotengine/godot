#ifndef MOBSTACLE
#define MOBSTACLE

#include "scene/3d/node_3d.h"





class MObstacle : public Node3D{
    GDCLASS(MObstacle,Node3D);

    protected:
    static void _bind_methods();

    public:
    float width=1.0;
    float depth=1.0;
    MObstacle();
    ~MObstacle();
    
    float get_width();
    void set_width(float input);
    float get_depth();
    void set_depth(float input);
};
#endif