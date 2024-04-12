/**************************************************************************/
/*  godot_broad_phase_2d_bvh.h                                            */
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

#ifndef GODOT_BROAD_PHASE_2D_BVH_H
#define GODOT_BROAD_PHASE_2D_BVH_H

#include "godot_broad_phase_2d.h"

#include "core/math/bvh.h"
#include "core/math/rect2.h"
#include "core/math/vector2.h"

class GodotBroadPhase2DBVH : public GodotBroadPhase2D {
	template <typename T>
	class UserPairTestFunction {
	public:
		static bool user_pair_check(const T *p_a, const T *p_b) {
			// return false if no collision, decided by masks etc
			return p_a->interacts_with(p_b);
		}
	};

	template <typename T>
	class UserCullTestFunction {
	public:
		static bool user_cull_check(const T *p_a, const T *p_b) {
			return true;
		}
	};

	enum Tree {
		TREE_STATIC = 0,
		TREE_DYNAMIC = 1,
	};

	enum TreeFlag {
		TREE_FLAG_STATIC = 1 << TREE_STATIC,
		TREE_FLAG_DYNAMIC = 1 << TREE_DYNAMIC,
	};

	BVH_Manager<GodotCollisionObject2D, 2, true, 128, UserPairTestFunction<GodotCollisionObject2D>, UserCullTestFunction<GodotCollisionObject2D>, Rect2, Vector2> bvh;

	static void *_pair_callback(void *, uint32_t, GodotCollisionObject2D *, int, uint32_t, GodotCollisionObject2D *, int);
	static void _unpair_callback(void *, uint32_t, GodotCollisionObject2D *, int, uint32_t, GodotCollisionObject2D *, int, void *);

	PairCallback pair_callback = nullptr;
	void *pair_userdata = nullptr;
	UnpairCallback unpair_callback = nullptr;
	void *unpair_userdata = nullptr;

public:
	// 0 is an invalid ID
	virtual ID create(GodotCollisionObject2D *p_object, int p_subindex = 0, const Rect2 &p_aabb = Rect2(), bool p_static = false) override;
	virtual void move(ID p_id, const Rect2 &p_aabb) override;
	virtual void set_static(ID p_id, bool p_static) override;
	virtual void remove(ID p_id) override;

	virtual GodotCollisionObject2D *get_object(ID p_id) const override;
	virtual bool is_static(ID p_id) const override;
	virtual int get_subindex(ID p_id) const override;

	virtual int cull_segment(const Vector2 &p_from, const Vector2 &p_to, GodotCollisionObject2D **p_results, int p_max_results, int *p_result_indices = nullptr) override;
	virtual int cull_aabb(const Rect2 &p_aabb, GodotCollisionObject2D **p_results, int p_max_results, int *p_result_indices = nullptr) override;

	virtual void set_pair_callback(PairCallback p_pair_callback, void *p_userdata) override;
	virtual void set_unpair_callback(UnpairCallback p_unpair_callback, void *p_userdata) override;

	virtual void update() override;

	static GodotBroadPhase2D *_create();
	GodotBroadPhase2DBVH();
};

#endif // GODOT_BROAD_PHASE_2D_BVH_H
