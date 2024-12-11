#include "jolt_body_accessor_3d.hpp"

#include "spaces/jolt_space_3d.hpp"

namespace {

template<class... TTypes>
struct VariantVisitors : TTypes... {
	using TTypes::operator()...;
};

template<class... TTypes>
VariantVisitors(TTypes...) -> VariantVisitors<TTypes...>;

} // namespace

JoltBodyAccessor3D::JoltBodyAccessor3D(const JoltSpace3D* p_space)
	: space(p_space) { }

JoltBodyAccessor3D::~JoltBodyAccessor3D() = default;

void JoltBodyAccessor3D::acquire(const JPH::BodyID* p_ids, int32_t p_id_count) {
	ERR_FAIL_NULL(space);

	lock_iface = &space->get_lock_iface();
	ids = BodyIDSpan(p_ids, p_id_count);
	_acquire_internal(p_ids, p_id_count);
}

void JoltBodyAccessor3D::acquire(const JPH::BodyID& p_id) {
	ERR_FAIL_NULL(space);

	lock_iface = &space->get_lock_iface();
	ids = p_id;
	_acquire_internal(&p_id, 1);
}

void JoltBodyAccessor3D::acquire_active() {
	const JPH::PhysicsSystem& physics_system = space->get_physics_system();

	acquire(
		physics_system.GetActiveBodiesUnsafe(JPH::EBodyType::RigidBody),
		(int32_t)physics_system.GetNumActiveBodies(JPH::EBodyType::RigidBody)
	);
}

void JoltBodyAccessor3D::acquire_all() {
	ERR_FAIL_NULL(space);

	lock_iface = &space->get_lock_iface();

	auto* vector = std::get_if<JPH::BodyIDVector>(&ids);

	if (vector == nullptr) {
		ids = JPH::BodyIDVector();
		vector = std::get_if<JPH::BodyIDVector>(&ids);
	}

	space->get_physics_system().GetBodies(*vector);

	_acquire_internal(vector->data(), (int32_t)vector->size());
}

void JoltBodyAccessor3D::release() {
	_release_internal();
	lock_iface = nullptr;
}

const JPH::BodyID* JoltBodyAccessor3D::get_ids() const {
	ERR_FAIL_COND_D(not_acquired());

	return std::visit(
		VariantVisitors{
			[](const JPH::BodyID& p_id) {
				return &p_id;
			},
			[](const JPH::BodyIDVector& p_vector) {
				return p_vector.data();
			},
			[](const BodyIDSpan& p_span) {
				return p_span.ptr;
			}},
		ids
	);
}

int32_t JoltBodyAccessor3D::get_count() const {
	ERR_FAIL_COND_D(not_acquired());

	return std::visit(
		VariantVisitors{
			[]([[maybe_unused]] const JPH::BodyID& p_id) {
				return 1;
			},
			[](const JPH::BodyIDVector& p_vector) {
				return (int32_t)p_vector.size();
			},
			[](const BodyIDSpan& p_span) {
				return p_span.count;
			}},
		ids
	);
}

const JPH::BodyID& JoltBodyAccessor3D::get_at(int32_t p_index) const {
	CRASH_BAD_INDEX(p_index, get_count());
	return get_ids()[p_index];
}

JoltBodyReader3D::JoltBodyReader3D(const JoltSpace3D* p_space)
	: JoltBodyAccessor3D(p_space) { }

const JPH::Body* JoltBodyReader3D::try_get(const JPH::BodyID& p_id) const {
	QUIET_FAIL_COND_D(p_id.IsInvalid());
	ERR_FAIL_COND_D(not_acquired());
	return lock_iface->TryGetBody(p_id);
}

const JPH::Body* JoltBodyReader3D::try_get(int32_t p_index) const {
	QUIET_FAIL_INDEX_D(p_index, get_count());
	return try_get(get_at(p_index));
}

const JPH::Body* JoltBodyReader3D::try_get() const {
	return try_get(0);
}

void JoltBodyReader3D::_acquire_internal(const JPH::BodyID* p_ids, int32_t p_id_count) {
	mutex_mask = lock_iface->GetMutexMask(p_ids, p_id_count);
	lock_iface->LockRead(mutex_mask);
}

void JoltBodyReader3D::_release_internal() {
	ERR_FAIL_COND(not_acquired());
	lock_iface->UnlockRead(mutex_mask);
}

JoltBodyWriter3D::JoltBodyWriter3D(const JoltSpace3D* p_space)
	: JoltBodyAccessor3D(p_space) { }

JPH::Body* JoltBodyWriter3D::try_get(const JPH::BodyID& p_id) const {
	QUIET_FAIL_COND_D(p_id.IsInvalid());
	ERR_FAIL_COND_D(not_acquired());
	return lock_iface->TryGetBody(p_id);
}

JPH::Body* JoltBodyWriter3D::try_get(int32_t p_index) const {
	QUIET_FAIL_INDEX_D(p_index, get_count());
	return try_get(get_at(p_index));
}

JPH::Body* JoltBodyWriter3D::try_get() const {
	return try_get(0);
}

void JoltBodyWriter3D::_acquire_internal(const JPH::BodyID* p_ids, int32_t p_id_count) {
	mutex_mask = lock_iface->GetMutexMask(p_ids, p_id_count);
	lock_iface->LockWrite(mutex_mask);
}

void JoltBodyWriter3D::_release_internal() {
	ERR_FAIL_COND(not_acquired());
	lock_iface->UnlockWrite(mutex_mask);
}
