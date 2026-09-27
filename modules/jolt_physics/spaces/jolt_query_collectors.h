/**************************************************************************/
/*  jolt_query_collectors.h                                               */
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

#pragma once

#include "core/math/vector3.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/STLLocalAllocator.h>
#include <Jolt/Physics/Collision/InternalEdgeRemovingCollector.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

template <typename TBase, int TDefaultCapacity>
class JoltQueryCollectorAll : public TBase {
public:
	typedef typename TBase::ResultType Hit;
	typedef JPH::Array<Hit, JPH::STLLocalAllocator<Hit, TDefaultCapacity>> HitArray;

private:
	HitArray hits;

public:
	JoltQueryCollectorAll() {
		hits.reserve(TDefaultCapacity);
	}

	bool had_hit() const {
		return !hits.is_empty();
	}

	int get_hit_count() const {
		return hits.size();
	}

	const Hit &get_hit(int p_index) const {
		return hits[p_index];
	}

	void reset() { Reset(); }

	virtual void Reset() override {
		TBase::Reset();
		hits.clear();
	}

	virtual void AddHit(const Hit &p_hit) override {
		hits.push_back(p_hit);
	}
};

template <typename TBase>
class JoltQueryCollectorAny : public TBase {
public:
	typedef typename TBase::ResultType Hit;

private:
	Hit hit;
	bool valid = false;

public:
	bool had_hit() const { return valid; }

	const Hit &get_hit() const { return hit; }

	void reset() {
		Reset();
	}

	virtual void Reset() override {
		TBase::Reset();
		valid = false;
	}

	virtual void AddHit(const Hit &p_hit) override {
		hit = p_hit;
		valid = true;

		TBase::ForceEarlyOut();
	}
};

template <typename TBase, int TDefaultCapacity>
class JoltQueryCollectorAnyMulti : public TBase {
public:
	typedef typename TBase::ResultType Hit;
	typedef JPH::Array<Hit, JPH::STLLocalAllocator<Hit, TDefaultCapacity>> HitArray;

private:
	HitArray hits;
	int max_hits = 0;

public:
	explicit JoltQueryCollectorAnyMulti(int p_max_hits = TDefaultCapacity) :
			max_hits(p_max_hits) {
		hits.reserve(TDefaultCapacity);
	}

	bool had_hit() const {
		return hits.size() > 0;
	}

	int get_hit_count() const {
		return hits.size();
	}

	const Hit &get_hit(int p_index) const {
		return hits[p_index];
	}

	void reset() {
		Reset();
	}

	virtual void Reset() override {
		TBase::Reset();
		hits.clear();
	}

	virtual void AddHit(const Hit &p_hit) override {
		if ((int)hits.size() < max_hits) {
			hits.push_back(p_hit);
		}

		if ((int)hits.size() == max_hits) {
			TBase::ForceEarlyOut();
		}
	}
};

template <typename TBase>
class JoltQueryCollectorClosest : public TBase {
public:
	typedef typename TBase::ResultType Hit;

private:
	Hit hit;
	bool valid = false;

public:
	bool had_hit() const { return valid; }

	const Hit &get_hit() const { return hit; }

	void reset() {
		Reset();
	}

	virtual void Reset() override {
		TBase::Reset();
		valid = false;
	}

	virtual void AddHit(const Hit &p_hit) override {
		const float early_out = p_hit.GetEarlyOutFraction();

		if (!valid || early_out < hit.GetEarlyOutFraction()) {
			TBase::UpdateEarlyOutFraction(early_out);

			hit = p_hit;
			valid = true;
		}
	}
};

template <typename TBase, int TDefaultCapacity>
class JoltQueryCollectorClosestMulti : public TBase {
public:
	typedef typename TBase::ResultType Hit;
	typedef JPH::Array<Hit, JPH::STLLocalAllocator<Hit, TDefaultCapacity + 1>> HitArray;

private:
	HitArray hits;
	int max_hits = 0;

public:
	explicit JoltQueryCollectorClosestMulti(int p_max_hits = TDefaultCapacity) :
			max_hits(p_max_hits) {
		hits.reserve(TDefaultCapacity + 1);
	}

	bool had_hit() const {
		return hits.size() > 0;
	}

	int get_hit_count() const {
		return hits.size();
	}

	const Hit &get_hit(int p_index) const {
		return hits[p_index];
	}

	void reset() {
		Reset();
	}

	virtual void Reset() override {
		TBase::Reset();
		hits.clear();
	}

	virtual void AddHit(const Hit &p_hit) override {
		typename HitArray::const_iterator E = hits.cbegin();
		for (; E != hits.cend(); ++E) {
			if (p_hit.GetEarlyOutFraction() < E->GetEarlyOutFraction()) {
				break;
			}
		}

		hits.insert(E, p_hit);

		if ((int)hits.size() > max_hits) {
			hits.resize(max_hits);
		}
	}
};

template <typename TCollector, int TDefaultCapacity>
class JoltQueryCollectorMotion : public JoltQueryCollectorClosestMulti<TCollector, TDefaultCapacity> {
public:
	typedef JoltQueryCollectorClosestMulti<TCollector, TDefaultCapacity> Base;
	typedef typename Base::Hit Hit;

private:
	Vector3 direction;
	float distance_sq = 0.0f;
	float margin = 0.0f;

public:
	JoltQueryCollectorMotion(const Vector3 &p_motion, float p_margin, int p_max_hits = TDefaultCapacity) :
			Base(p_max_hits),
			direction(p_motion.normalized()),
			distance_sq(p_motion.length_squared()),
			margin(p_margin) {}

	virtual void AddHit(const Hit &p_hit) override {
		// Ignore hits that are outside of the margin.
		const float penetration_depth = p_hit.mPenetrationDepth + margin;
		if (penetration_depth <= 0.0f) {
			return;
		}

		// Ignore hits that don't oppose the motion direction.
		//
		// This is a deliberate divergence from the Godot Physics reference implementation (which
		// does not do this type of filtering) and is known to cause issues. However, not having
		// this results in a problematic amount of ghost collisions with `move_and_slide`, for
		// reasons that are still unclear as of writing this.
		if (distance_sq > 0) {
			const Vector3 normal = to_godot(-p_hit.mPenetrationAxis.Normalized());
			if (direction.dot(normal) >= -CMP_EPSILON) {
				return;
			}
		}

		Base::AddHit(p_hit);
	}
};
