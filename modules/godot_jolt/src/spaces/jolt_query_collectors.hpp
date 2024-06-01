#pragma once
#include "../common.h"
#include "servers/jolt_project_settings.hpp"
#include "containers/inline_vector.hpp"

template<typename TBase, int32_t TInlineCapacity>
class JoltQueryCollectorAll : public TBase {
public:
	using Hit = typename TBase::ResultType;

	bool had_hit() const { return !hits.is_empty(); }

	int32_t get_hit_count() const { return hits.size(); }

	const Hit& get_hit(int32_t p_index) const { return hits[p_index]; }

	void reset() { Reset(); }

	void Reset() override {
		TBase::Reset();

		hits.clear();
	}

	void AddHit(const Hit& p_hit) override { hits.push_back(p_hit); }

private:
	InlineVector<Hit, TInlineCapacity> hits;
};

template<typename TBase>
class JoltQueryCollectorAny final : public TBase {
public:
	using Hit = typename TBase::ResultType;

	bool had_hit() const { return valid; }

	const Hit& get_hit() const { return hit; }

	void reset() { Reset(); }

	void Reset() override {
		TBase::Reset();

		valid = false;
	}

	void AddHit(const Hit& p_hit) override {
		hit = p_hit;
		valid = true;

		TBase::ForceEarlyOut();
	}

private:
	Hit hit;

	bool valid = false;
};

template<typename TBase, int32_t TInlineCapacity>
class JoltQueryCollectorAnyMulti final : public TBase {
public:
	using Hit = typename TBase::ResultType;

	explicit JoltQueryCollectorAnyMulti(int32_t p_max_hits = TInlineCapacity)
		: max_hits(p_max_hits) { }

	bool had_hit() const { return hits.size() > 0; }

	int32_t get_hit_count() const { return hits.size(); }

	const Hit& get_hit(int32_t p_index) const { return hits[p_index]; }

	void reset() { Reset(); }

	void Reset() override {
		TBase::Reset();

		hits.clear();
	}

	void AddHit(const Hit& p_hit) override {
		if (hits.size() < max_hits) {
			hits.push_back(p_hit);
		}

		if (hits.size() == max_hits) {
			TBase::ForceEarlyOut();
		}
	}

private:
	InlineVector<Hit, TInlineCapacity> hits;

	int32_t max_hits = 0;
};

template<typename TBase>
class JoltQueryCollectorClosest final : public TBase {
public:
	using Hit = typename TBase::ResultType;

	bool had_hit() const { return valid; }

	const Hit& get_hit() const { return hit; }

	void reset() { Reset(); }

	void Reset() override {
		TBase::Reset();

		valid = false;
	}

	void AddHit(const Hit& p_hit) override {
		const float early_out = p_hit.GetEarlyOutFraction();

		if (!valid || early_out < hit.GetEarlyOutFraction()) {
			TBase::UpdateEarlyOutFraction(early_out);

			hit = p_hit;
			valid = true;
		}
	}

private:
	Hit hit;

	bool valid = false;
};

template<typename TBase, int32_t TInlineCapacity>
class JoltQueryCollectorClosestMulti final : public TBase {
public:
	using Hit = typename TBase::ResultType;

	explicit JoltQueryCollectorClosestMulti(int32_t p_max_hits = TInlineCapacity)
		: max_hits(p_max_hits) { }

	bool had_hit() const { return hits.size() > 0; }

	int32_t get_hit_count() const { return hits.size(); }

	const Hit& get_hit(int32_t p_index) const { return hits[p_index]; }

	void reset() { Reset(); }

	void Reset() override {
		TBase::Reset();

		hits.clear();
	}

	void AddHit(const Hit& p_hit) override {
		hits.ordered_insert(p_hit, [](const Hit& p_lhs, const Hit& p_rhs) {
			return p_lhs.GetEarlyOutFraction() < p_rhs.GetEarlyOutFraction();
		});

		if (hits.size() > max_hits) {
			hits.resize(max_hits);
		}
	}

private:
	InlineVector<Hit, TInlineCapacity + 1> hits;

	int32_t max_hits = 0;
};

template<typename TInnerCollector>
class JoltQueryCollectorNoEdges final : public JPH::InternalEdgeRemovingCollector {
	using Base = JPH::InternalEdgeRemovingCollector;

public:
	using Hit = typename TInnerCollector::ResultType;

	template<typename... TArgs>
	explicit JoltQueryCollectorNoEdges(TArgs&&... p_args)
		: Base(inner_collector)
		, inner_collector(std::forward<TArgs>(p_args)...) { }

	int32_t get_hit_count() const { return inner_collector.get_hit_count(); }

	const Hit& get_hit() const { return inner_collector.get_hit(); }

	const Hit& get_hit(int32_t p_index) const { return inner_collector.get_hit(p_index); }

	bool finish() {
		if (use_enhanced_edge_removal) {
			Base::Flush();
		}

		return inner_collector.had_hit();
	}

	void reset() { Reset(); }

	void OnBody(const JPH::Body& p_body) override {
		if (use_enhanced_edge_removal) {
			Base::OnBody(p_body);
		} else {
			inner_collector.OnBody(p_body);
		}
	}

	void AddHit(const Hit& p_hit) override {
		if (use_enhanced_edge_removal) {
			Base::AddHit(p_hit);
		} else {
			inner_collector.AddHit(p_hit);
		}
	}

	void Reset() override {
		if (use_enhanced_edge_removal) {
			Base::Reset();
		} else {
			inner_collector.Reset();
		}
	}

private:
	TInnerCollector inner_collector;

	bool use_enhanced_edge_removal = JoltProjectSettings::use_enhanced_edge_removal();
};

template<int32_t TInlineCapacity>
using JoltQueryCollectorAllNoEdges = JoltQueryCollectorNoEdges<
	JoltQueryCollectorAll<JPH::CollideShapeCollector, TInlineCapacity>>;

using JoltQueryCollectorAnyNoEdges = JoltQueryCollectorNoEdges<
	JoltQueryCollectorAny<JPH::CollideShapeCollector>>;

template<int32_t TInlineCapacity>
using JoltQueryCollectorAnyMultiNoEdges = JoltQueryCollectorNoEdges<
	JoltQueryCollectorAnyMulti<JPH::CollideShapeCollector, TInlineCapacity>>;

using JoltQueryCollectorClosestNoEdges = JoltQueryCollectorNoEdges<
	JoltQueryCollectorClosest<JPH::CollideShapeCollector>>;

template<int32_t TInlineCapacity>
using JoltQueryCollectorClosestMultiNoEdges = JoltQueryCollectorNoEdges<
	JoltQueryCollectorClosestMulti<JPH::CollideShapeCollector, TInlineCapacity>>;
