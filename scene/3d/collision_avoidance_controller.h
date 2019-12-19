/*************************************************************************/
/*  collision_avoidance_controller.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef COLLISION_AVOIDANCE_CONTROLLER_H
#define COLLISION_AVOIDANCE_CONTROLLER_H

#include "scene/main/node.h"

class CollisionAvoidanceController : public Node {
    GDCLASS(CollisionAvoidanceController, Node);

    RID agent;

    real_t neighbor_dist;
    int max_neighbors;
    real_t time_horizon;
    real_t time_horizon_obs;
    real_t radius;
    real_t max_speed;

    bool velocity_submitted;
    Vector2 prev_safe_velocity;
    /// The submitted target velocity
    Vector3 target_velocity;

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    CollisionAvoidanceController();

    RID get_rid() const {
        return agent;
    }

    void set_neighbor_dist(real_t p_dist);
    real_t get_neighbor_dist() const {
        return neighbor_dist;
    }

    void set_max_neighbors(int p_count);
    int get_max_neighbors() const {
        return max_neighbors;
    }

    void set_time_horizon(real_t p_time);
    real_t get_time_horizon() const {
        return time_horizon;
    }
    void set_time_horizon_obs(real_t p_time);
    real_t get_time_horizon_obs() const {
        return time_horizon_obs;
    }
    void set_radius(real_t p_radius);
    real_t get_radius() const {
        return radius;
    }

    void set_max_speed(real_t p_max_speed);
    real_t get_max_speed() const {
        return max_speed;
    }

    void set_velocity(Vector3 p_velocity);

    void _avoidance_done(Vector2 p_new_velocity);

    virtual String get_configuration_warning() const;
};

#endif // COLLISION_AVOIDANCE_CONTROLLER_H
