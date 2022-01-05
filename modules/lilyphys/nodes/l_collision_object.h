//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_L_COLLISION_OBJECT_H
#define LILYPHYS_L_COLLISION_OBJECT_H

#include "scene/3d/spatial.h"
#include "l_shape.h"

class LCollisionShape;
class LCollisionObject : public Spatial {
    GDCLASS(LCollisionObject, Spatial);
public:
    enum Type {
        TYPE_BODY,
        TYPE_TRIGGER
    };
private:
    struct ShapeData {
        LCollisionShape* owner;
        Ref<LShape> shape;
        size_t id;
        Transform xform;
        bool disabled = false;
    };
    Map<size_t, ShapeData> shapes;
    size_t last_idx = 0;
    Type type;
protected:
    RID rid;
    static void _bind_methods();
    void _notification(int p_what);
public:
    RID get_rid();
    size_t create_shape_owner(LCollisionShape* p_owner);
    void remove_shape_owner(size_t p_id);
    void shape_owner_set_shape(size_t p_id, const Ref<LShape>& p_shape);
    void shape_owner_remove_shape(size_t p_id);
    void shape_owner_set_disabled(size_t p_id, bool p_disabled);
    void shape_owner_set_transform(size_t p_id, const Transform& p_transform);
    LCollisionObject(Type p_type) : type(p_type) {}
    ~LCollisionObject() override;
};


#endif //LILYPHYS_L_COLLISION_OBJECT_H
