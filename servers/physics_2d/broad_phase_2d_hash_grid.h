/*************************************************************************/
/*  broad_phase_2d_hash_grid.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef BROAD_PHASE_2D_HASH_GRID_H
#define BROAD_PHASE_2D_HASH_GRID_H

#include "broad_phase_2d_sw.h"
#include "core/map.h"

class BroadPhase2DHashGrid : public BroadPhase2DSW {
	struct PairData {
		bool colliding;
		int rc;
		void *ud;
		PairData() {
			colliding = false;
			rc = 1;
			ud = nullptr;
		}
	};

	struct Element {
		ID self;
		CollisionObject2DSW *owner;
		bool _static;
		Rect2 aabb;
		int subindex;
		uint64_t pass;
		Map<Element *, PairData *> paired;
	};

	struct RC {
		int ref;

		_FORCE_INLINE_ int inc() {
			ref++;
			return ref;
		}
		_FORCE_INLINE_ int dec() {
			ref--;
			return ref;
		}

		_FORCE_INLINE_ RC() {
			ref = 0;
		}
	};

	Map<ID, Element> element_map;
	Map<Element *, RC> large_elements;

	ID current;

	uint64_t pass;

	struct PairKey {
		union {
			struct {
				ID a;
				ID b;
			};
			uint64_t key;
		};

		_FORCE_INLINE_ bool operator<(const PairKey &p_key) const {
			return key < p_key.key;
		}

		PairKey() { key = 0; }
		PairKey(ID p_a, ID p_b) {
			if (p_a > p_b) {
				a = p_b;
				b = p_a;
			} else {
				a = p_a;
				b = p_b;
			}
		}
	};

	Map<PairKey, PairData> pair_map;

	int cell_size;
	int large_object_min_surface;

	PairCallback pair_callback;
	void *pair_userdata;
	UnpairCallback unpair_callback;
	void *unpair_userdata;

	void _enter_grid(Element *p_elem, const Rect2 &p_rect, bool p_static);
	void _exit_grid(Element *p_elem, const Rect2 &p_rect, bool p_static);
	template <bool use_aabb, bool use_segment>
	_FORCE_INLINE_ void _cull(const Point2i p_cell, const Rect2 &p_aabb, const Point2 &p_from, const Point2 &p_to, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices, int &index);

	struct PosKey {
		union {
			struct {
				int32_t x;
				int32_t y;
			};
			uint64_t key;
		};

		_FORCE_INLINE_ uint32_t hash() const {
			uint64_t k = key;
			k = (~k) + (k << 18); // k = (k << 18) - k - 1;
			k = k ^ (k >> 31);
			k = k * 21; // k = (k + (k << 2)) + (k << 4);
			k = k ^ (k >> 11);
			k = k + (k << 6);
			k = k ^ (k >> 22);
			return k;
		}

		bool operator==(const PosKey &p_key) const { return key == p_key.key; }
		_FORCE_INLINE_ bool operator<(const PosKey &p_key) const {
			return key < p_key.key;
		}
	};

	struct PosBin {
		PosKey key;
		Map<Element *, RC> object_set;
		Map<Element *, RC> static_object_set;
		PosBin *next;
	};

	uint32_t hash_table_size;
	PosBin **hash_table;

	void _pair_attempt(Element *p_elem, Element *p_with);
	void _unpair_attempt(Element *p_elem, Element *p_with);
	void _check_motion(Element *p_elem);

public:
	virtual ID create(CollisionObject2DSW *p_object, int p_subindex = 0);
	virtual void move(ID p_id, const Rect2 &p_aabb);
	virtual void set_static(ID p_id, bool p_static);
	virtual void remove(ID p_id);

	virtual CollisionObject2DSW *get_object(ID p_id) const;
	virtual bool is_static(ID p_id) const;
	virtual int get_subindex(ID p_id) const;

	virtual int cull_segment(const Vector2 &p_from, const Vector2 &p_to, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices = nullptr);
	virtual int cull_aabb(const Rect2 &p_aabb, CollisionObject2DSW **p_results, int p_max_results, int *p_result_indices = nullptr);

	virtual void set_pair_callback(PairCallback p_pair_callback, void *p_userdata);
	virtual void set_unpair_callback(UnpairCallback p_unpair_callback, void *p_userdata);

	virtual void update();

	static BroadPhase2DSW *_create();

	BroadPhase2DHashGrid();
	~BroadPhase2DHashGrid();
};

#endif // BROAD_PHASE_2D_HASH_GRID_H
