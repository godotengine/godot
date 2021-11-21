//
// Created by amara on 21/11/2021.
//

#ifndef LILYPHYS_LI_SPRING_H
#define LILYPHYS_LI_SPRING_H

#include "li_force_generator.h"

class LISpring : public LIForceGenerator {
private:
    Vector3 connection_point;
    Vector3 other_connection_point;
    RID other;
    real_t spring_constant;
    real_t rest_length;
public:
    LISpring(const Vector3& p_local_connection_point, RID p_other, const Vector3& p_other_connection_point, real_t p_spring_constant, real_t p_rest_length)
        : connection_point(p_local_connection_point), other(p_other), other_connection_point(p_other_connection_point), spring_constant(p_spring_constant), rest_length(p_rest_length) {}
    void update_force(LIPhysicsBody *p_body, real_t p_delta) override;
};


#endif //LILYPHYS_LI_SPRING_H
