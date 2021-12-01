//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LI_COLLISION_OBJECT_H
#define LILYPHYS_LI_COLLISION_OBJECT_H

#include "core/map.h"
#include "core/rid.h"
#include "core/math/transform.h"

class LICollisionObject : public RID_Data {
public:
    enum Type {
        TYPE_BODY,
        TYPE_AREA // Not implemented yet.
    };
    struct ShapeData {
        RID shape;
        bool disabled = false;
        Transform transform;
    };

private:
    RID self;
    Type type;
    Map<size_t, ShapeData> shapes;
    size_t last_idx = 0;

protected:
    Transform transform;

public:
    LICollisionObject(Type p_type);
    void set_self(RID p_rid);
    RID get_self();
    void set_transform(const Transform& p_transform) { transform = p_transform; }
    const Transform &get_transform() const { return transform; }
    size_t add_shape(RID p_rid);
    void remove_shape(size_t p_id);
    void remove_shape(RID p_rid);
    void clear_shapes();
    void set_shape_disabled(size_t p_id, bool p_disabled);
    void set_shape_transform(size_t p_id, const Transform& p_transform);
    bool get_shape_exists(size_t p_id);
    const Map<size_t, ShapeData>& get_shapes() { return shapes; }
};


#endif //LILYPHYS_LI_COLLISION_OBJECT_H
