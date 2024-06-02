#pragma once

template<
	typename TKey,
	typename THasher = HashMapHasherDefault,
	typename TComparator = HashMapComparatorDefault<TKey>>
class JHashSet {
	struct Hasher {
		_FORCE_INLINE_ size_t operator()(const TKey& p_key) const {
			return (size_t)THasher::hash(p_key);
		}
	};

	struct Comparator {
		_FORCE_INLINE_ bool operator()(const TKey& p_lhs, const TKey& p_rhs) const {
			return TComparator::compare(p_lhs, p_rhs);
		}
	};

	using Implementation = std::unordered_set<TKey, Hasher, Comparator>;

public:
	using Iterator = typename Implementation::iterator;
	using ConstIterator = typename Implementation::const_iterator;

	JHashSet() = default;

	explicit JHashSet(int32_t p_capacity) { impl.reserve((size_t)p_capacity); }

	_FORCE_INLINE_ int32_t get_capacity() const {
		return int32_t(impl.max_load_factor() * impl.bucket_count());
	}

	_FORCE_INLINE_ int32_t size() const { return (int32_t)impl.size(); }

	_FORCE_INLINE_ bool is_empty() const { return impl.empty(); }

	_FORCE_INLINE_ void clear() { impl.clear(); }

	_FORCE_INLINE_ bool has(const TKey& p_key) const { return impl.find(p_key) != end(); }

	_FORCE_INLINE_ bool erase(const TKey& p_key) { return impl.erase(p_key) != 0; }

	template<typename TPredicate>
	_FORCE_INLINE_ int32_t erase_if(TPredicate&& p_pred) {
		int32_t count = 0;

		for (auto iter = begin(); iter != end();) {
			if (p_pred(*iter)) {
				iter = impl.erase(iter);
				count++;
			} else {
				iter++;
			}
		}

		return count;
	}

	_FORCE_INLINE_ void reserve(int32_t p_capacity) { impl.reserve((size_t)p_capacity); }

	_FORCE_INLINE_ Iterator find(const TKey& p_key) const { return impl.find(p_key); }

	_FORCE_INLINE_ void remove(ConstIterator p_iter) { impl.erase(p_iter); }

	_FORCE_INLINE_ Iterator insert(const TKey& p_key) { return emplace(p_key); }

	_FORCE_INLINE_ Iterator insert(TKey&& p_key) { return emplace(std::move(p_key)); }

	template<typename... TArgs>
	_FORCE_INLINE_ Iterator emplace(TArgs&&... p_args) {
		return impl.emplace(std::forward<TArgs>(p_args)...).first;
	}

	_FORCE_INLINE_ Iterator begin() { return impl.begin(); }

	_FORCE_INLINE_ Iterator end() { return impl.end(); }

	_FORCE_INLINE_ ConstIterator begin() const { return impl.begin(); }

	_FORCE_INLINE_ ConstIterator end() const { return impl.end(); }

	_FORCE_INLINE_ ConstIterator cbegin() const { return impl.cbegin(); }

	_FORCE_INLINE_ ConstIterator cend() const { return impl.cend(); }

private:
	Implementation impl;
};
