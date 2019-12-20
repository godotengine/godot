/*************************************************************************/
/*  rvo_agent.cpp                                                        */
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

#include "rvo_agent.h"

#include "rvo_space.h"

RvoAgent::RvoAgent(RvoSpace *p_space) :
        space(p_space) {
    callback.id = ObjectID(0);
}

void RvoAgent::set_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {
    callback.id = p_id;
    callback.method = p_method;
    callback.udata = p_udata;

    if (p_id == 0) {
        space->remove_agent_as_controlled(this);
    } else {
        space->set_agent_as_controlled(this);
    }
}

void RvoAgent::dispatch_callback() {
    if (callback.id == 0) {
        return;
    }
    Object *obj = ObjectDB::get_instance(callback.id);
    if (obj == NULL) {
        callback.id = ObjectID(0);
    }

    Variant::CallError responseCallError;

    callback.new_velocity = Vector2(agent.newVelocity_.x(), agent.newVelocity_.y());

    const Variant *vp[2] = { &callback.new_velocity, &callback.udata };
    int argc = (callback.udata.get_type() == Variant::NIL) ? 1 : 2;
    obj->call(callback.method, vp, argc, responseCallError);
}
