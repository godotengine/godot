//
// Created by amara on 23/11/2021.
//

#ifndef LILYPHYS_L_COLLISION_SHAPE_H
#define LILYPHYS_L_COLLISION_SHAPE_H

#include "scene/3d/spatial.h"
#include "l_shape.h"

class LCollisionObject;
class LCollisionShape : public Spatial {
    GDCLASS(LCollisionShape, Spatial);
    LCollisionObject* parent = nullptr;
    size_t owner_id = 0;
    Ref<LShape> shape;
    bool disabled = false;
    void resource_changed(RES res);
protected:
    void _notification(int p_what);
    static void _bind_methods();
public:
    void set_shape(const Ref<LShape> &p_shape);
    Ref<LShape> get_shape() const;

    void set_disabled(bool p_disabled);
    bool is_disabled() const;

    void update_parent(bool p_transform_only = false);
    LCollisionShape();
    ~LCollisionShape();
};

#endif //LILYPHYS_L_COLLISION_SHAPE_H
