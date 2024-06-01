// #pragma once
// #include "core/templates/hash_map.h"

// template<typename TResource>
// // NOLINTNEXTLINE(readability-identifier-naming)
// class RID_PtrOwner {
// public:
// 	RID_PtrOwner() = default;

// 	RID_PtrOwner(const RID_PtrOwner& p_other) = default;

// 	RID_PtrOwner(RID_PtrOwner&& p_other) noexcept = default;

// 	~RID_PtrOwner() {
// 		if (ptrs_by_id.size() > 0) {
// 			WARN_PRINT(vformat(
// 				"%d RIDs in Godot Jolt were found to not have been freed. "
// 				"This is likely caused by orphaned nodes. "
// 				"If not, consider reporting this issue.",
// 				ptrs_by_id.size()
// 			));
// 		}
// 	}

// 	_FORCE_INLINE_ RID make_rid(TResource* p_ptr) {
// 		const int64_t id = UtilityFunctions::rid_allocate_id();
// 		ptrs_by_id[id] = p_ptr;
// 		return UtilityFunctions::rid_from_int64(id);
// 	}

// 	_FORCE_INLINE_ TResource* get_or_null(const RID& p_rid) const {
// 		auto iter = ptrs_by_id.find(p_rid.get_id());
// 		return iter != ptrs_by_id.end() ? iter->second : nullptr;
// 	}

// 	_FORCE_INLINE_ void replace(const RID& p_rid, TResource* p_new_ptr) {
// 		auto iter = ptrs_by_id.find(p_rid.get_id());
// 		ERR_FAIL_COND(iter == ptrs_by_id.end());
// 		iter->second = p_new_ptr;
// 	}

// 	_FORCE_INLINE_ bool owns(const RID& p_rid) const { return ptrs_by_id.has(p_rid.get_id()); }

// 	_FORCE_INLINE_ void free(const RID& p_rid) { ptrs_by_id.erase(p_rid.get_id()); }

// 	RID_PtrOwner& operator=(const RID_PtrOwner& p_other) = default;

// 	RID_PtrOwner& operator=(RID_PtrOwner&& p_other) noexcept = default;

// private:
// 	HashMap<int64_t, TResource*> ptrs_by_id;
// };
