/*************************************************************************/
/*  octree.h                                                             */
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
#ifndef OCTREE_H
#define OCTREE_H

#include "list.h"
#include "map.h"
#include "print_string.h"
#include "rect3.h"
#include "variant.h"
#include "vector3.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

typedef uint32_t OctreeElementID;

#define OCTREE_ELEMENT_INVALID_ID 0
#define OCTREE_SIZE_LIMIT 1e15

template <class T, bool use_pairs = false, class AL = DefaultAllocator>
class Octree {
public:
	typedef void *(*PairCallback)(void *, OctreeElementID, T *, int, OctreeElementID, T *, int);
	typedef void (*UnpairCallback)(void *, OctreeElementID, T *, int, OctreeElementID, T *, int, void *);

private:
	enum {

		NEG = 0,
		POS = 1,
	};

	enum {
		OCTANT_NX_NY_NZ,
		OCTANT_PX_NY_NZ,
		OCTANT_NX_PY_NZ,
		OCTANT_PX_PY_NZ,
		OCTANT_NX_NY_PZ,
		OCTANT_PX_NY_PZ,
		OCTANT_NX_PY_PZ,
		OCTANT_PX_PY_PZ
	};

	struct PairKey {

		union {
			struct {
				OctreeElementID A;
				OctreeElementID B;
			};
			uint64_t key;
		};

		_FORCE_INLINE_ bool operator<(const PairKey &p_pair) const {

			return key < p_pair.key;
		}

		_FORCE_INLINE_ PairKey(OctreeElementID p_A, OctreeElementID p_B) {

			if (p_A < p_B) {

				A = p_A;
				B = p_B;
			} else {

				B = p_A;
				A = p_B;
			}
		}

		_FORCE_INLINE_ PairKey() {}
	};

	struct Element;

	struct Octant {

		// cached for FAST plane check
		Rect3 aabb;

		uint64_t last_pass;
		Octant *parent;
		Octant *children[8];

		int children_count; // cache for amount of childrens (fast check for removal)
		int parent_index; // cache for parent index (fast check for removal)

		List<Element *, AL> pairable_elements;
		List<Element *, AL> elements;

		Octant() {
			children_count = 0;
			parent_index = -1;
			last_pass = 0;
			parent = NULL;
			for (int i = 0; i < 8; i++)
				children[i] = NULL;
		}

		~Octant() {

			/*
			for (int i=0;i<8;i++)
				memdelete_notnull(children[i]);
			*/
		}
	};

	struct PairData;

	struct Element {

		Octree *octree;

		T *userdata;
		int subindex;
		bool pairable;
		uint32_t pairable_mask;
		uint32_t pairable_type;

		uint64_t last_pass;
		OctreeElementID _id;
		Octant *common_parent;

		Rect3 aabb;
		Rect3 container_aabb;

		List<PairData *, AL> pair_list;

		struct OctantOwner {

			Octant *octant;
			typename List<Element *, AL>::Element *E;
		}; // an element can be in max 8 octants

		List<OctantOwner, AL> octant_owners;

		Element() {
			last_pass = 0;
			_id = 0;
			pairable = false;
			subindex = 0;
			userdata = 0;
			octree = 0;
			pairable_mask = 0;
			pairable_type = 0;
			common_parent = NULL;
		}
	};

	struct PairData {

		int refcount;
		bool intersect;
		Element *A, *B;
		void *ud;
		typename List<PairData *, AL>::Element *eA, *eB;
	};

	typedef Map<OctreeElementID, Element, Comparator<OctreeElementID>, AL> ElementMap;
	typedef Map<PairKey, PairData, Comparator<PairKey>, AL> PairMap;
	ElementMap element_map;
	PairMap pair_map;

	PairCallback pair_callback;
	UnpairCallback unpair_callback;
	void *pair_callback_userdata;
	void *unpair_callback_userdata;

	OctreeElementID last_element_id;
	uint64_t pass;

	real_t unit_size;
	Octant *root;
	int octant_count;
	int pair_count;

	_FORCE_INLINE_ void _pair_check(PairData *p_pair) {

		bool intersect = p_pair->A->aabb.intersects_inclusive(p_pair->B->aabb);

		if (intersect != p_pair->intersect) {

			if (intersect) {

				if (pair_callback) {
					p_pair->ud = pair_callback(pair_callback_userdata, p_pair->A->_id, p_pair->A->userdata, p_pair->A->subindex, p_pair->B->_id, p_pair->B->userdata, p_pair->B->subindex);
				}
				pair_count++;
			} else {

				if (unpair_callback) {
					unpair_callback(pair_callback_userdata, p_pair->A->_id, p_pair->A->userdata, p_pair->A->subindex, p_pair->B->_id, p_pair->B->userdata, p_pair->B->subindex, p_pair->ud);
				}
				pair_count--;
			}

			p_pair->intersect = intersect;
		}
	}

	_FORCE_INLINE_ void _pair_reference(Element *p_A, Element *p_B) {

		if (p_A == p_B || (p_A->userdata == p_B->userdata && p_A->userdata))
			return;

		if (!(p_A->pairable_type & p_B->pairable_mask) &&
				!(p_B->pairable_type & p_A->pairable_mask))
			return; // none can pair with none

		PairKey key(p_A->_id, p_B->_id);
		typename PairMap::Element *E = pair_map.find(key);

		if (!E) {

			PairData pdata;
			pdata.refcount = 1;
			pdata.A = p_A;
			pdata.B = p_B;
			pdata.intersect = false;
			E = pair_map.insert(key, pdata);
			E->get().eA = p_A->pair_list.push_back(&E->get());
			E->get().eB = p_B->pair_list.push_back(&E->get());

			/*
			if (pair_callback)
				pair_callback(pair_callback_userdata,p_A->userdata,p_B->userdata);
			*/
		} else {

			E->get().refcount++;
		}
	}

	_FORCE_INLINE_ void _pair_unreference(Element *p_A, Element *p_B) {

		if (p_A == p_B)
			return;

		PairKey key(p_A->_id, p_B->_id);
		typename PairMap::Element *E = pair_map.find(key);
		if (!E) {
			return; // no pair
		}

		E->get().refcount--;

		if (E->get().refcount == 0) {
			// bye pair

			if (E->get().intersect) {
				if (unpair_callback) {
					unpair_callback(pair_callback_userdata, p_A->_id, p_A->userdata, p_A->subindex, p_B->_id, p_B->userdata, p_B->subindex, E->get().ud);
				}

				pair_count--;
			}

			if (p_A == E->get().B) {
				//may be reaching inverted
				SWAP(p_A, p_B);
			}

			p_A->pair_list.erase(E->get().eA);
			p_B->pair_list.erase(E->get().eB);
			pair_map.erase(E);
		}
	}

	_FORCE_INLINE_ void _element_check_pairs(Element *p_element) {

		typename List<PairData *, AL>::Element *E = p_element->pair_list.front();
		while (E) {

			_pair_check(E->get());
			E = E->next();
		}
	}

	_FORCE_INLINE_ void _optimize() {

		while (root && root->children_count < 2 && !root->elements.size() && !(use_pairs && root->pairable_elements.size())) {

			Octant *new_root = NULL;
			if (root->children_count == 1) {

				for (int i = 0; i < 8; i++) {

					if (root->children[i]) {
						new_root = root->children[i];
						root->children[i] = NULL;
						break;
					}
				}
				ERR_FAIL_COND(!new_root);
				new_root->parent = NULL;
				new_root->parent_index = -1;
			}

			memdelete_allocator<Octant, AL>(root);
			octant_count--;
			root = new_root;
		}
	}

	void _insert_element(Element *p_element, Octant *p_octant);
	void _ensure_valid_root(const Rect3 &p_aabb);
	bool _remove_element_from_octant(Element *p_element, Octant *p_octant, Octant *p_limit = NULL);
	void _remove_element(Element *p_element);
	void _pair_element(Element *p_element, Octant *p_octant);
	void _unpair_element(Element *p_element, Octant *p_octant);

	struct _CullConvexData {

		const Plane *planes;
		int plane_count;
		T **result_array;
		int *result_idx;
		int result_max;
		uint32_t mask;
	};

	void _cull_convex(Octant *p_octant, _CullConvexData *p_cull);
	void _cull_AABB(Octant *p_octant, const Rect3 &p_aabb, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask);
	void _cull_segment(Octant *p_octant, const Vector3 &p_from, const Vector3 &p_to, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask);
	void _cull_point(Octant *p_octant, const Vector3 &p_point, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask);

	void _remove_tree(Octant *p_octant) {

		if (!p_octant)
			return;

		for (int i = 0; i < 8; i++) {

			if (p_octant->children[i])
				_remove_tree(p_octant->children[i]);
		}

		memdelete_allocator<Octant, AL>(p_octant);
	}

public:
	OctreeElementID create(T *p_userdata, const Rect3 &p_aabb = Rect3(), int p_subindex = 0, bool p_pairable = false, uint32_t p_pairable_type = 0, uint32_t pairable_mask = 1);
	void move(OctreeElementID p_id, const Rect3 &p_aabb);
	void set_pairable(OctreeElementID p_id, bool p_pairable = false, uint32_t p_pairable_type = 0, uint32_t pairable_mask = 1);
	void erase(OctreeElementID p_id);

	bool is_pairable(OctreeElementID p_id) const;
	T *get(OctreeElementID p_id) const;
	int get_subindex(OctreeElementID p_id) const;

	int cull_convex(const Vector<Plane> &p_convex, T **p_result_array, int p_result_max, uint32_t p_mask = 0xFFFFFFFF);
	int cull_AABB(const Rect3 &p_aabb, T **p_result_array, int p_result_max, int *p_subindex_array = NULL, uint32_t p_mask = 0xFFFFFFFF);
	int cull_segment(const Vector3 &p_from, const Vector3 &p_to, T **p_result_array, int p_result_max, int *p_subindex_array = NULL, uint32_t p_mask = 0xFFFFFFFF);

	int cull_point(const Vector3 &p_point, T **p_result_array, int p_result_max, int *p_subindex_array = NULL, uint32_t p_mask = 0xFFFFFFFF);

	void set_pair_callback(PairCallback p_callback, void *p_userdata);
	void set_unpair_callback(UnpairCallback p_callback, void *p_userdata);

	int get_octant_count() const { return octant_count; }
	int get_pair_count() const { return pair_count; }
	Octree(real_t p_unit_size = 1.0);
	~Octree() { _remove_tree(root); }
};

/* PRIVATE FUNCTIONS */

template <class T, bool use_pairs, class AL>
T *Octree<T, use_pairs, AL>::get(OctreeElementID p_id) const {
	const typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, NULL);
	return E->get().userdata;
}

template <class T, bool use_pairs, class AL>
bool Octree<T, use_pairs, AL>::is_pairable(OctreeElementID p_id) const {

	const typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, false);
	return E->get().pairable;
}

template <class T, bool use_pairs, class AL>
int Octree<T, use_pairs, AL>::get_subindex(OctreeElementID p_id) const {

	const typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND_V(!E, -1);
	return E->get().subindex;
}

#define OCTREE_DIVISOR 4

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_insert_element(Element *p_element, Octant *p_octant) {

	real_t element_size = p_element->aabb.get_longest_axis_size() * 1.01; // avoid precision issues

	if (p_octant->aabb.size.x / OCTREE_DIVISOR < element_size) {
		//if (p_octant->aabb.size.x*0.5 < element_size) {

		/* at smallest possible size for the element  */
		typename Element::OctantOwner owner;
		owner.octant = p_octant;

		if (use_pairs && p_element->pairable) {

			p_octant->pairable_elements.push_back(p_element);
			owner.E = p_octant->pairable_elements.back();
		} else {

			p_octant->elements.push_back(p_element);
			owner.E = p_octant->elements.back();
		}

		p_element->octant_owners.push_back(owner);

		if (p_element->common_parent == NULL) {
			p_element->common_parent = p_octant;
			p_element->container_aabb = p_octant->aabb;
		} else {
			p_element->container_aabb.merge_with(p_octant->aabb);
		}

		if (use_pairs && p_octant->children_count > 0) {

			pass++; //elements below this only get ONE reference added

			for (int i = 0; i < 8; i++) {

				if (p_octant->children[i]) {
					_pair_element(p_element, p_octant->children[i]);
				}
			}
		}
	} else {
		/* not big enough, send it to subitems */
		int splits = 0;
		bool candidate = p_element->common_parent == NULL;

		for (int i = 0; i < 8; i++) {

			if (p_octant->children[i]) {
				/* element exists, go straight to it */
				if (p_octant->children[i]->aabb.intersects_inclusive(p_element->aabb)) {
					_insert_element(p_element, p_octant->children[i]);
					splits++;
				}
			} else {
				/* check againt AABB where child should be */

				Rect3 aabb = p_octant->aabb;
				aabb.size *= 0.5;

				if (i & 1)
					aabb.pos.x += aabb.size.x;
				if (i & 2)
					aabb.pos.y += aabb.size.y;
				if (i & 4)
					aabb.pos.z += aabb.size.z;

				if (aabb.intersects_inclusive(p_element->aabb)) {
					/* if actually intersects, create the child */

					Octant *child = memnew_allocator(Octant, AL);
					p_octant->children[i] = child;
					child->parent = p_octant;
					child->parent_index = i;

					child->aabb = aabb;

					p_octant->children_count++;

					_insert_element(p_element, child);
					octant_count++;
					splits++;
				}
			}
		}

		if (candidate && splits > 1) {

			p_element->common_parent = p_octant;
		}
	}

	if (use_pairs) {

		typename List<Element *, AL>::Element *E = p_octant->pairable_elements.front();

		while (E) {
			_pair_reference(p_element, E->get());
			E = E->next();
		}

		if (p_element->pairable) {
			// and always test non-pairable if element is pairable
			E = p_octant->elements.front();
			while (E) {
				_pair_reference(p_element, E->get());
				E = E->next();
			}
		}
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_ensure_valid_root(const Rect3 &p_aabb) {

	if (!root) {
		// octre is empty

		Rect3 base(Vector3(), Vector3(1.0, 1.0, 1.0) * unit_size);

		while (!base.encloses(p_aabb)) {

			if (ABS(base.pos.x + base.size.x) <= ABS(base.pos.x)) {
				/* grow towards positive */
				base.size *= 2.0;
			} else {
				base.pos -= base.size;
				base.size *= 2.0;
			}
		}

		root = memnew_allocator(Octant, AL);

		root->parent = NULL;
		root->parent_index = -1;
		root->aabb = base;

		octant_count++;

	} else {

		Rect3 base = root->aabb;

		while (!base.encloses(p_aabb)) {

			if (base.size.x > OCTREE_SIZE_LIMIT) {
				ERR_EXPLAIN("Octree upper size limit reeached, does the AABB supplied contain NAN?");
				ERR_FAIL();
			}

			Octant *gp = memnew_allocator(Octant, AL);
			octant_count++;
			root->parent = gp;

			if (ABS(base.pos.x + base.size.x) <= ABS(base.pos.x)) {
				/* grow towards positive */
				base.size *= 2.0;
				gp->aabb = base;
				gp->children[0] = root;
				root->parent_index = 0;
			} else {
				base.pos -= base.size;
				base.size *= 2.0;
				gp->aabb = base;
				gp->children[(1 << 0) | (1 << 1) | (1 << 2)] = root; // add at all-positive
				root->parent_index = (1 << 0) | (1 << 1) | (1 << 2);
			}

			gp->children_count = 1;
			root = gp;
		}
	}
}

template <class T, bool use_pairs, class AL>
bool Octree<T, use_pairs, AL>::_remove_element_from_octant(Element *p_element, Octant *p_octant, Octant *p_limit) {

	bool octant_removed = false;

	while (true) {

		// check all exit conditions

		if (p_octant == p_limit) // reached limit, nothing to erase, exit
			return octant_removed;

		bool unpaired = false;

		if (use_pairs && p_octant->last_pass != pass) {
			// check wether we should unpair stuff
			// always test pairable
			typename List<Element *, AL>::Element *E = p_octant->pairable_elements.front();
			while (E) {
				_pair_unreference(p_element, E->get());
				E = E->next();
			}
			if (p_element->pairable) {
				// and always test non-pairable if element is pairable
				E = p_octant->elements.front();
				while (E) {
					_pair_unreference(p_element, E->get());
					E = E->next();
				}
			}
			p_octant->last_pass = pass;
			unpaired = true;
		}

		bool removed = false;

		Octant *parent = p_octant->parent;

		if (p_octant->children_count == 0 && p_octant->elements.empty() && p_octant->pairable_elements.empty()) {

			// erase octant

			if (p_octant == root) { // won't have a parent, just erase

				root = NULL;
			} else {
				ERR_FAIL_INDEX_V(p_octant->parent_index, 8, octant_removed);

				parent->children[p_octant->parent_index] = NULL;
				parent->children_count--;
			}

			memdelete_allocator<Octant, AL>(p_octant);
			octant_count--;
			removed = true;
			octant_removed = true;
		}

		if (!removed && !unpaired)
			return octant_removed; // no reason to keep going up anymore! was already visited and was not removed

		p_octant = parent;
	}

	return octant_removed;
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_unpair_element(Element *p_element, Octant *p_octant) {

	// always test pairable
	typename List<Element *, AL>::Element *E = p_octant->pairable_elements.front();
	while (E) {
		if (E->get()->last_pass != pass) { // only remove ONE reference
			_pair_unreference(p_element, E->get());
			E->get()->last_pass = pass;
		}
		E = E->next();
	}

	if (p_element->pairable) {
		// and always test non-pairable if element is pairable
		E = p_octant->elements.front();
		while (E) {
			if (E->get()->last_pass != pass) { // only remove ONE reference
				_pair_unreference(p_element, E->get());
				E->get()->last_pass = pass;
			}
			E = E->next();
		}
	}

	p_octant->last_pass = pass;

	if (p_octant->children_count == 0)
		return; // small optimization for leafs

	for (int i = 0; i < 8; i++) {

		if (p_octant->children[i])
			_unpair_element(p_element, p_octant->children[i]);
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_pair_element(Element *p_element, Octant *p_octant) {

	// always test pairable

	typename List<Element *, AL>::Element *E = p_octant->pairable_elements.front();

	while (E) {

		if (E->get()->last_pass != pass) { // only get ONE reference
			_pair_reference(p_element, E->get());
			E->get()->last_pass = pass;
		}
		E = E->next();
	}

	if (p_element->pairable) {
		// and always test non-pairable if element is pairable
		E = p_octant->elements.front();
		while (E) {
			if (E->get()->last_pass != pass) { // only get ONE reference
				_pair_reference(p_element, E->get());
				E->get()->last_pass = pass;
			}
			E = E->next();
		}
	}
	p_octant->last_pass = pass;

	if (p_octant->children_count == 0)
		return; // small optimization for leafs

	for (int i = 0; i < 8; i++) {

		if (p_octant->children[i])
			_pair_element(p_element, p_octant->children[i]);
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_remove_element(Element *p_element) {

	pass++; // will do a new pass for this

	typename List<typename Element::OctantOwner, AL>::Element *I = p_element->octant_owners.front();

	/* FIRST remove going up normally */
	for (; I; I = I->next()) {

		Octant *o = I->get().octant;

		if (!use_pairs) // small speedup
			o->elements.erase(I->get().E);

		_remove_element_from_octant(p_element, o);
	}

	/* THEN remove going down */

	I = p_element->octant_owners.front();

	if (use_pairs) {

		for (; I; I = I->next()) {

			Octant *o = I->get().octant;

			// erase children pairs, they are erased ONCE even if repeated
			pass++;
			for (int i = 0; i < 8; i++) {

				if (o->children[i])
					_unpair_element(p_element, o->children[i]);
			}

			if (p_element->pairable)
				o->pairable_elements.erase(I->get().E);
			else
				o->elements.erase(I->get().E);
		}
	}

	p_element->octant_owners.clear();

	if (use_pairs) {

		int remaining = p_element->pair_list.size();
		//p_element->pair_list.clear();
		ERR_FAIL_COND(remaining);
	}
}

template <class T, bool use_pairs, class AL>
OctreeElementID Octree<T, use_pairs, AL>::create(T *p_userdata, const Rect3 &p_aabb, int p_subindex, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask) {

// check for AABB validity
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(p_aabb.pos.x > 1e15 || p_aabb.pos.x < -1e15, 0);
	ERR_FAIL_COND_V(p_aabb.pos.y > 1e15 || p_aabb.pos.y < -1e15, 0);
	ERR_FAIL_COND_V(p_aabb.pos.z > 1e15 || p_aabb.pos.z < -1e15, 0);
	ERR_FAIL_COND_V(p_aabb.size.x > 1e15 || p_aabb.size.x < 0.0, 0);
	ERR_FAIL_COND_V(p_aabb.size.y > 1e15 || p_aabb.size.y < 0.0, 0);
	ERR_FAIL_COND_V(p_aabb.size.z > 1e15 || p_aabb.size.z < 0.0, 0);
	ERR_FAIL_COND_V(Math::is_nan(p_aabb.size.x), 0);
	ERR_FAIL_COND_V(Math::is_nan(p_aabb.size.y), 0);
	ERR_FAIL_COND_V(Math::is_nan(p_aabb.size.z), 0);

#endif
	typename ElementMap::Element *E = element_map.insert(last_element_id++,
			Element());
	Element &e = E->get();

	e.aabb = p_aabb;
	e.userdata = p_userdata;
	e.subindex = p_subindex;
	e.last_pass = 0;
	e.octree = this;
	e.pairable = p_pairable;
	e.pairable_type = p_pairable_type;
	e.pairable_mask = p_pairable_mask;
	e._id = last_element_id - 1;

	if (!e.aabb.has_no_surface()) {
		_ensure_valid_root(p_aabb);
		_insert_element(&e, root);
		if (use_pairs)
			_element_check_pairs(&e);
	}

	return last_element_id - 1;
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::move(OctreeElementID p_id, const Rect3 &p_aabb) {

#ifdef DEBUG_ENABLED
	// check for AABB validity
	ERR_FAIL_COND(p_aabb.pos.x > 1e15 || p_aabb.pos.x < -1e15);
	ERR_FAIL_COND(p_aabb.pos.y > 1e15 || p_aabb.pos.y < -1e15);
	ERR_FAIL_COND(p_aabb.pos.z > 1e15 || p_aabb.pos.z < -1e15);
	ERR_FAIL_COND(p_aabb.size.x > 1e15 || p_aabb.size.x < 0.0);
	ERR_FAIL_COND(p_aabb.size.y > 1e15 || p_aabb.size.y < 0.0);
	ERR_FAIL_COND(p_aabb.size.z > 1e15 || p_aabb.size.z < 0.0);
	ERR_FAIL_COND(Math::is_nan(p_aabb.size.x));
	ERR_FAIL_COND(Math::is_nan(p_aabb.size.y));
	ERR_FAIL_COND(Math::is_nan(p_aabb.size.z));
#endif
	typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);
	Element &e = E->get();

#if 0

	pass++;
	if (!e.aabb.has_no_surface()) {
		_remove_element(&e);
	}

	e.aabb=p_aabb;

	if (!e.aabb.has_no_surface()) {
		_ensure_valid_root(p_aabb);

		_insert_element(&e,root);
		if (use_pairs)
			_element_check_pairs(&e);

	}

	_optimize();

#else

	bool old_has_surf = !e.aabb.has_no_surface();
	bool new_has_surf = !p_aabb.has_no_surface();

	if (old_has_surf != new_has_surf) {

		if (old_has_surf) {
			_remove_element(&e); // removing
			e.common_parent = NULL;
			e.aabb = Rect3();
			_optimize();
		} else {
			_ensure_valid_root(p_aabb); // inserting
			e.common_parent = NULL;
			e.aabb = p_aabb;
			_insert_element(&e, root);
			if (use_pairs)
				_element_check_pairs(&e);
		}

		return;
	}

	if (!old_has_surf) // doing nothing
		return;

	// it still is enclosed in the same AABB it was assigned to
	if (e.container_aabb.encloses(p_aabb)) {

		e.aabb = p_aabb;
		if (use_pairs)
			_element_check_pairs(&e); // must check pairs anyway

		return;
	}

	Rect3 combined = e.aabb;
	combined.merge_with(p_aabb);
	_ensure_valid_root(combined);

	ERR_FAIL_COND(e.octant_owners.front() == NULL);

	/* FIND COMMON PARENT */

	List<typename Element::OctantOwner, AL> owners = e.octant_owners; // save the octant owners
	Octant *common_parent = e.common_parent;
	ERR_FAIL_COND(!common_parent);

	//src is now the place towards where insertion is going to happen
	pass++;

	while (common_parent && !common_parent->aabb.encloses(p_aabb))
		common_parent = common_parent->parent;

	ERR_FAIL_COND(!common_parent);

	//prepare for reinsert
	e.octant_owners.clear();
	e.common_parent = NULL;
	e.aabb = p_aabb;

	_insert_element(&e, common_parent); // reinsert from this point

	pass++;

	for (typename List<typename Element::OctantOwner, AL>::Element *E = owners.front(); E;) {

		Octant *o = E->get().octant;
		typename List<typename Element::OctantOwner, AL>::Element *N = E->next();

		/*
		if (!use_pairs)
			o->elements.erase( E->get().E );
		*/

		if (use_pairs && e.pairable)
			o->pairable_elements.erase(E->get().E);
		else
			o->elements.erase(E->get().E);

		if (_remove_element_from_octant(&e, o, common_parent->parent)) {

			owners.erase(E);
		}

		E = N;
	}

	if (use_pairs) {
		//unpair child elements in anything that survived
		for (typename List<typename Element::OctantOwner, AL>::Element *E = owners.front(); E; E = E->next()) {

			Octant *o = E->get().octant;

			// erase children pairs, unref ONCE
			pass++;
			for (int i = 0; i < 8; i++) {

				if (o->children[i])
					_unpair_element(&e, o->children[i]);
			}
		}

		_element_check_pairs(&e);
	}

	_optimize();
#endif
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::set_pairable(OctreeElementID p_id, bool p_pairable, uint32_t p_pairable_type, uint32_t p_pairable_mask) {

	typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();

	if (p_pairable == e.pairable && e.pairable_type == p_pairable_type && e.pairable_mask == p_pairable_mask)
		return; // no changes, return

	if (!e.aabb.has_no_surface()) {
		_remove_element(&e);
	}

	e.pairable = p_pairable;
	e.pairable_type = p_pairable_type;
	e.pairable_mask = p_pairable_mask;
	e.common_parent = NULL;

	if (!e.aabb.has_no_surface()) {
		_ensure_valid_root(e.aabb);
		_insert_element(&e, root);
		if (use_pairs)
			_element_check_pairs(&e);
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::erase(OctreeElementID p_id) {

	typename ElementMap::Element *E = element_map.find(p_id);
	ERR_FAIL_COND(!E);

	Element &e = E->get();

	if (!e.aabb.has_no_surface()) {

		_remove_element(&e);
	}

	element_map.erase(p_id);
	_optimize();
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_cull_convex(Octant *p_octant, _CullConvexData *p_cull) {

	if (*p_cull->result_idx == p_cull->result_max)
		return; //pointless

	if (!p_octant->elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->elements.front();

		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_cull->mask)))
				continue;
			e->last_pass = pass;

			if (e->aabb.intersects_convex_shape(p_cull->planes, p_cull->plane_count)) {

				if (*p_cull->result_idx < p_cull->result_max) {
					p_cull->result_array[*p_cull->result_idx] = e->userdata;
					(*p_cull->result_idx)++;
				} else {

					return; // pointless to continue
				}
			}
		}
	}

	if (use_pairs && !p_octant->pairable_elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->pairable_elements.front();

		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_cull->mask)))
				continue;
			e->last_pass = pass;

			if (e->aabb.intersects_convex_shape(p_cull->planes, p_cull->plane_count)) {

				if (*p_cull->result_idx < p_cull->result_max) {

					p_cull->result_array[*p_cull->result_idx] = e->userdata;
					(*p_cull->result_idx)++;
				} else {

					return; // pointless to continue
				}
			}
		}
	}

	for (int i = 0; i < 8; i++) {

		if (p_octant->children[i] && p_octant->children[i]->aabb.intersects_convex_shape(p_cull->planes, p_cull->plane_count)) {
			_cull_convex(p_octant->children[i], p_cull);
		}
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_cull_AABB(Octant *p_octant, const Rect3 &p_aabb, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (*p_result_idx == p_result_max)
		return; //pointless

	if (!p_octant->elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;
			e->last_pass = pass;

			if (p_aabb.intersects_inclusive(e->aabb)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;

					(*p_result_idx)++;
				} else {

					return; // pointless to continue
				}
			}
		}
	}

	if (use_pairs && !p_octant->pairable_elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->pairable_elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;
			e->last_pass = pass;

			if (p_aabb.intersects_inclusive(e->aabb)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;
					(*p_result_idx)++;
				} else {

					return; // pointless to continue
				}
			}
		}
	}

	for (int i = 0; i < 8; i++) {

		if (p_octant->children[i] && p_octant->children[i]->aabb.intersects_inclusive(p_aabb)) {
			_cull_AABB(p_octant->children[i], p_aabb, p_result_array, p_result_idx, p_result_max, p_subindex_array, p_mask);
		}
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_cull_segment(Octant *p_octant, const Vector3 &p_from, const Vector3 &p_to, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (*p_result_idx == p_result_max)
		return; //pointless

	if (!p_octant->elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;
			e->last_pass = pass;

			if (e->aabb.intersects_segment(p_from, p_to)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;
					(*p_result_idx)++;

				} else {

					return; // pointless to continue
				}
			}
		}
	}

	if (use_pairs && !p_octant->pairable_elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->pairable_elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;

			e->last_pass = pass;

			if (e->aabb.intersects_segment(p_from, p_to)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;

					(*p_result_idx)++;

				} else {

					return; // pointless to continue
				}
			}
		}
	}

	for (int i = 0; i < 8; i++) {

		if (p_octant->children[i] && p_octant->children[i]->aabb.intersects_segment(p_from, p_to)) {
			_cull_segment(p_octant->children[i], p_from, p_to, p_result_array, p_result_idx, p_result_max, p_subindex_array, p_mask);
		}
	}
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::_cull_point(Octant *p_octant, const Vector3 &p_point, T **p_result_array, int *p_result_idx, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (*p_result_idx == p_result_max)
		return; //pointless

	if (!p_octant->elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;
			e->last_pass = pass;

			if (e->aabb.has_point(p_point)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;
					(*p_result_idx)++;

				} else {

					return; // pointless to continue
				}
			}
		}
	}

	if (use_pairs && !p_octant->pairable_elements.empty()) {

		typename List<Element *, AL>::Element *I;
		I = p_octant->pairable_elements.front();
		for (; I; I = I->next()) {

			Element *e = I->get();

			if (e->last_pass == pass || (use_pairs && !(e->pairable_type & p_mask)))
				continue;

			e->last_pass = pass;

			if (e->aabb.has_point(p_point)) {

				if (*p_result_idx < p_result_max) {

					p_result_array[*p_result_idx] = e->userdata;
					if (p_subindex_array)
						p_subindex_array[*p_result_idx] = e->subindex;

					(*p_result_idx)++;

				} else {

					return; // pointless to continue
				}
			}
		}
	}

	for (int i = 0; i < 8; i++) {

		//could be optimized..
		if (p_octant->children[i] && p_octant->children[i]->aabb.has_point(p_point)) {
			_cull_point(p_octant->children[i], p_point, p_result_array, p_result_idx, p_result_max, p_subindex_array, p_mask);
		}
	}
}

template <class T, bool use_pairs, class AL>
int Octree<T, use_pairs, AL>::cull_convex(const Vector<Plane> &p_convex, T **p_result_array, int p_result_max, uint32_t p_mask) {

	if (!root)
		return 0;

	int result_count = 0;
	pass++;
	_CullConvexData cdata;
	cdata.planes = &p_convex[0];
	cdata.plane_count = p_convex.size();
	cdata.result_array = p_result_array;
	cdata.result_max = p_result_max;
	cdata.result_idx = &result_count;
	cdata.mask = p_mask;

	_cull_convex(root, &cdata);

	return result_count;
}

template <class T, bool use_pairs, class AL>
int Octree<T, use_pairs, AL>::cull_AABB(const Rect3 &p_aabb, T **p_result_array, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (!root)
		return 0;

	int result_count = 0;
	pass++;
	_cull_AABB(root, p_aabb, p_result_array, &result_count, p_result_max, p_subindex_array, p_mask);

	return result_count;
}

template <class T, bool use_pairs, class AL>
int Octree<T, use_pairs, AL>::cull_segment(const Vector3 &p_from, const Vector3 &p_to, T **p_result_array, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (!root)
		return 0;

	int result_count = 0;
	pass++;
	_cull_segment(root, p_from, p_to, p_result_array, &result_count, p_result_max, p_subindex_array, p_mask);

	return result_count;
}

template <class T, bool use_pairs, class AL>
int Octree<T, use_pairs, AL>::cull_point(const Vector3 &p_point, T **p_result_array, int p_result_max, int *p_subindex_array, uint32_t p_mask) {

	if (!root)
		return 0;

	int result_count = 0;
	pass++;
	_cull_point(root, p_point, p_result_array, &result_count, p_result_max, p_subindex_array, p_mask);

	return result_count;
}

template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::set_pair_callback(PairCallback p_callback, void *p_userdata) {

	pair_callback = p_callback;
	pair_callback_userdata = p_userdata;
}
template <class T, bool use_pairs, class AL>
void Octree<T, use_pairs, AL>::set_unpair_callback(UnpairCallback p_callback, void *p_userdata) {

	unpair_callback = p_callback;
	unpair_callback_userdata = p_userdata;
}

template <class T, bool use_pairs, class AL>
Octree<T, use_pairs, AL>::Octree(real_t p_unit_size) {

	last_element_id = 1;
	pass = 1;
	unit_size = p_unit_size;
	root = NULL;

	octant_count = 0;
	pair_count = 0;

	pair_callback = NULL;
	unpair_callback = NULL;
	pair_callback_userdata = NULL;
	unpair_callback_userdata = NULL;
}

#endif
