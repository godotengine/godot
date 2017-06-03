/*************************************************************************/
/*  broad_phase_octree.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef BROAD_PHASE_OCTREE_H
#define BROAD_PHASE_OCTREE_H

#include "broad_phase_sw.h"
#include "octree.h"

class BroadPhaseOctree : public BroadPhaseSW {

	Octree<CollisionObjectSW, true> octree;

	static void *_pair_callback(void *, OctreeElementID, CollisionObjectSW *, int, OctreeElementID, CollisionObjectSW *, int);
	static void _unpair_callback(void *, OctreeElementID, CollisionObjectSW *, int, OctreeElementID, CollisionObjectSW *, int, void *);

	PairCallback pair_callback;
	void *pair_userdata;
	UnpairCallback unpair_callback;
	void *unpair_userdata;

public:
	// 0 is an invalid ID
	virtual ID create(CollisionObjectSW *p_object_, int p_subindex = 0);
	virtual void move(ID p_id, const Rect3 &p_aabb);
	virtual void set_static(ID p_id, bool p_static);
	virtual void remove(ID p_id);

	virtual CollisionObjectSW *get_object(ID p_id) const;
	virtual bool is_static(ID p_id) const;
	virtual int get_subindex(ID p_id) const;

	virtual int cull_segment(const Vector3 &p_from, const Vector3 &p_to, CollisionObjectSW **p_results, int p_max_results, int *p_result_indices = NULL);
	virtual int cull_aabb(const Rect3 &p_aabb, CollisionObjectSW **p_results, int p_max_results, int *p_result_indices = NULL);

	virtual void set_pair_callback(PairCallback p_pair_callback, void *p_userdata);
	virtual void set_unpair_callback(UnpairCallback p_unpair_callback, void *p_userdata);

	virtual void update();

	static BroadPhaseSW *_create();
	BroadPhaseOctree();
};

#endif // BROAD_PHASE_OCTREE_H
