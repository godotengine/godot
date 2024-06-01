#pragma once

#include "objects/jolt_object_impl_3d.hpp"
#include <variant>

class JoltAreaImpl3D;
class JoltBodyImpl3D;
class JoltShapedObjectImpl3D;
class JoltSpace3D;

class JoltBodyAccessor3D {
	struct BodyIDSpan {
		BodyIDSpan(const JPH::BodyID* p_ptr, int32_t p_count)
			: ptr(p_ptr)
			, count(p_count) { }

		const JPH::BodyID* ptr;

		int32_t count;
	};

public:
	explicit JoltBodyAccessor3D(const JoltSpace3D* p_space);

	virtual ~JoltBodyAccessor3D() = 0;

	void acquire(const JPH::BodyID* p_ids, int32_t p_id_count);

	void acquire(const JPH::BodyID& p_id);

	void acquire_active();

	void acquire_all();

	void release();

	bool is_acquired() const { return lock_iface != nullptr; }

	bool not_acquired() const { return lock_iface == nullptr; }

	const JoltSpace3D& get_space() const { return *space; }

	const JPH::BodyID* get_ids() const;

	int32_t get_count() const;

	const JPH::BodyID& get_at(int32_t p_index) const;

protected:
	virtual void _acquire_internal(const JPH::BodyID* p_ids, int32_t p_id_count) = 0;

	virtual void _release_internal() = 0;

	const JoltSpace3D* space = nullptr;

	const JPH::BodyLockInterface* lock_iface = nullptr;

	std::variant<JPH::BodyID, JPH::BodyIDVector, BodyIDSpan> ids;
};

class JoltBodyReader3D final : public JoltBodyAccessor3D {
public:
	explicit JoltBodyReader3D(const JoltSpace3D* p_space);

	const JPH::Body* try_get(const JPH::BodyID& p_id) const;

	const JPH::Body* try_get(int32_t p_index) const;

	const JPH::Body* try_get() const;

private:
	void _acquire_internal(const JPH::BodyID* p_ids, int32_t p_id_count) override;

	void _release_internal() override;

	JPH::BodyLockInterface::MutexMask mutex_mask = 0;
};

class JoltBodyWriter3D final : public JoltBodyAccessor3D {
public:
	explicit JoltBodyWriter3D(const JoltSpace3D* p_space);

	JPH::Body* try_get(const JPH::BodyID& p_id) const;

	JPH::Body* try_get(int32_t p_index) const;

	JPH::Body* try_get() const;

private:
	void _acquire_internal(const JPH::BodyID* p_ids, int32_t p_id_count) override;

	void _release_internal() override;

	JPH::BodyLockInterface::MutexMask mutex_mask = 0;
};

template<typename TBodyAccessor>
class JoltScopedBodyAccessor3D {
public:
	JoltScopedBodyAccessor3D(
		const JoltSpace3D& p_space,
		const JPH::BodyID* p_ids,
		int32_t p_id_count
	)
		: inner(&p_space) {
		inner.acquire(p_ids, p_id_count);
	}

	JoltScopedBodyAccessor3D(const JoltSpace3D& p_space, const JPH::BodyID& p_id)
		: inner(&p_space) {
		inner.acquire(p_id);
	}

	JoltScopedBodyAccessor3D(const JoltScopedBodyAccessor3D& p_other) = delete;

	JoltScopedBodyAccessor3D(JoltScopedBodyAccessor3D&& p_other) noexcept = default;

	const JoltSpace3D& get_space() const { return inner.get_space(); }

	int32_t get_count() const { return inner.get_count(); }

	const JPH::BodyID& get_at(int32_t p_index) const { return inner.get_at(p_index); }

	~JoltScopedBodyAccessor3D() { inner.release(); }

	JoltScopedBodyAccessor3D& operator=(const JoltScopedBodyAccessor3D& p_other) = delete;

	JoltScopedBodyAccessor3D& operator=(JoltScopedBodyAccessor3D&& p_other) noexcept = default;

	decltype(auto) try_get(const JPH::BodyID& p_id) const { return inner.try_get(p_id); }

	decltype(auto) try_get(int32_t p_index) const { return inner.try_get(p_index); }

	decltype(auto) try_get() const { return inner.try_get(); }

private:
	TBodyAccessor inner;
};

template<typename TAccessor, typename TBody>
class JoltAccessibleBody3D {
public:
	JoltAccessibleBody3D(const JoltSpace3D& p_space, const JPH::BodyID& p_id)
		: accessor(p_space, p_id)
		, body(accessor.try_get()) { }

	bool is_valid() const { return body != nullptr; }

	bool is_invalid() const { return body == nullptr; }

	JoltObjectImpl3D* as_object() const {
		if (body != nullptr) {
			return reinterpret_cast<JoltObjectImpl3D*>(body->GetUserData());
		} else {
			return nullptr;
		}
	}

	JoltShapedObjectImpl3D* as_shaped() const {
		if (JoltObjectImpl3D* object = as_object(); object != nullptr && object->is_shaped()) {
			return reinterpret_cast<JoltShapedObjectImpl3D*>(body->GetUserData());
		} else {
			return nullptr;
		}
	}

	JoltBodyImpl3D* as_body() const {
		if (JoltObjectImpl3D* object = as_object(); object != nullptr && object->is_body()) {
			return reinterpret_cast<JoltBodyImpl3D*>(body->GetUserData());
		} else {
			return nullptr;
		}
	}

	JoltAreaImpl3D* as_area() const {
		if (JoltObjectImpl3D* object = as_object(); object != nullptr && object->is_area()) {
			return reinterpret_cast<JoltAreaImpl3D*>(body->GetUserData());
		} else {
			return nullptr;
		}
	}

	TBody* operator->() const { return body; }

	TBody& operator*() const { return *body; }

	explicit operator TBody*() const { return body; }

private:
	TAccessor accessor;

	TBody* body = nullptr;
};

template<typename TAccessor, typename TBody>
class JoltAccessibleBodies3D {
public:
	JoltAccessibleBodies3D(const JoltSpace3D& p_space, const JPH::BodyID* p_ids, int32_t p_id_count)
		: accessor(p_space, p_ids, p_id_count) { }

	JoltAccessibleBody3D<TAccessor, TBody> operator[](int32_t p_index) const {
		const JPH::BodyID& body_id = p_index < accessor.get_count()
			? accessor.get_at(p_index)
			: JPH::BodyID();

		return {accessor.get_space(), body_id};
	}

private:
	TAccessor accessor;
};

using JoltScopedBodyReader3D = JoltScopedBodyAccessor3D<JoltBodyReader3D>;
using JoltScopedBodyWriter3D = JoltScopedBodyAccessor3D<JoltBodyWriter3D>;

using JoltReadableBody3D = JoltAccessibleBody3D<JoltScopedBodyReader3D, const JPH::Body>;
using JoltWritableBody3D = JoltAccessibleBody3D<JoltScopedBodyWriter3D, JPH::Body>;

using JoltReadableBodies3D = JoltAccessibleBodies3D<JoltScopedBodyReader3D, const JPH::Body>;
using JoltWritableBodies3D = JoltAccessibleBodies3D<JoltScopedBodyWriter3D, JPH::Body>;
