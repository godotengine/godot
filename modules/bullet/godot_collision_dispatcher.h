/*************************************************************************/
/*  godot_collision_dispatcher.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_COLLISION_DISPATCHER_H
#define GODOT_COLLISION_DISPATCHER_H

#include "core/int_types.h"

#if defined(__clang__) && (__clang_major__ >= 13)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#endif

#include <btBulletDynamicsCommon.h>

#if defined(__clang__) && (__clang_major__ >= 13)
#pragma clang diagnostic pop
#endif

/**
	@author AndreaCatania
*/

/// This class is required to implement custom collision behaviour in the narrowphase
class GodotCollisionDispatcher : public btCollisionDispatcher {
private:
	static const int CASTED_TYPE_AREA;

public:
	GodotCollisionDispatcher(btCollisionConfiguration *collisionConfiguration);
	virtual bool needsCollision(const btCollisionObject *body0, const btCollisionObject *body1);
	virtual bool needsResponse(const btCollisionObject *body0, const btCollisionObject *body1);
};

#endif // GODOT_COLLISION_DISPATCHER_H
