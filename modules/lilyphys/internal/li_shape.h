//
// Created by amara on 25/11/2021.
//

#ifndef LILYPHYS_LI_SHAPE_H
#define LILYPHYS_LI_SHAPE_H

#include "core/rid.h"
#include "core/math/vector3.h"
#include "core/map.h"

class LICollisionObject;
class LIShape : public RID_Data {
private:
    RID self;
    Map<LICollisionObject*, int> owners;
public:
    RID get_self() { return self; }
    void set_self(RID p_self) { self = p_self; }
    virtual void set_data(const Variant& p_data) = 0;
    virtual Variant get_data() const = 0;
    virtual Vector3 get_support(Vector3 p_direction) const = 0;
    void add_owner(LICollisionObject* p_object);
    void remove_owner(LICollisionObject* p_object);
    bool is_owner(LICollisionObject* p_object) const;
    const Map<LICollisionObject*, int>& get_owners() const;
};

class LIBoxShape : public LIShape {
private:
    Vector3 half_extents;
    void set_half_extends(Vector3 p_extends) { half_extents = p_extends.abs(); }
public:
    void set_data(const Variant &p_data) override;
    Variant get_data() const override;
    Vector3 get_support(Vector3 p_direction) const override;
};

class LISphereShape : public LIShape {
private:
    real_t radius;
    void set_radius(real_t p_radius) { radius = abs(p_radius); }
public:
    void set_data(const Variant &p_data) override;
    Variant get_data() const override;
    Vector3 get_support(Vector3 p_direction) const override;
};


#endif //LILYPHYS_LI_SHAPE_H
