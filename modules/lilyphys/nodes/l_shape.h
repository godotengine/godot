//
// Created by amara on 25/11/2021.
//

#ifndef LILYPHYS_L_SHAPE_H
#define LILYPHYS_L_SHAPE_H

#include "core/resource.h"
#include "../lilyphys_server.h"

class LShape : public Resource {
    GDCLASS(LShape, Resource);
    OBJ_SAVE_TYPE(LShape);
    RES_BASE_EXTENSION("lshape");
protected:
    RID self;
    virtual void update_shape();
public:
    LShape(LShapeType p_type);
    RID get_self() const;
};

class LBoxShape : public LShape {
    GDCLASS(LBoxShape, LShape);
private:
    Vector3 extents;
    void update_shape() override;
protected:
    static void _bind_methods();
public:
    void set_extents(const Vector3& p_extends);
    Vector3 get_extents() const;
    LBoxShape();
};


#endif //LILYPHYS_L_SHAPE_H
