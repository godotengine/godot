/**************************************************************************/
/*  jolt_group_filter.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef JOLT_GROUP_FILTER_H
#define JOLT_GROUP_FILTER_H

#include "Jolt/Jolt.h"

#include "Jolt/Physics/Collision/CollisionGroup.h"
#include "Jolt/Physics/Collision/GroupFilter.h"

class JoltObject3D;

class JoltGroupFilter final : public JPH::GroupFilter {
	virtual bool CanCollide(const JPH::CollisionGroup &p_group1, const JPH::CollisionGroup &p_group2) const override;

public:
	inline static JoltGroupFilter *instance = nullptr;

	static void encode_object(const JoltObject3D *p_object, JPH::CollisionGroup::GroupID &r_group_id, JPH::CollisionGroup::SubGroupID &r_sub_group_id);
	static const JoltObject3D *decode_object(JPH::CollisionGroup::GroupID p_group_id, JPH::CollisionGroup::SubGroupID p_sub_group_id);
};

#endif // JOLT_GROUP_FILTER_H
