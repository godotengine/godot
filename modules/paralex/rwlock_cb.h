#ifndef RWLOCK_CB_H
#define RWLOCK_CB_H

#include "core/os/rw_lock.h"
#include "core/reference.h"

class _RWLock : public Reference {
	GDCLASS(_RWLock, Reference);
private:
	mutable RWLock lock;
protected:
	static _FORCE_INLINE_ void _bind_methods(){
		ClassDB::bind_method(D_METHOD("read_lock"), &_RWLock::read_lock);
		ClassDB::bind_method(D_METHOD("read_unlock"), &_RWLock::read_unlock);
		ClassDB::bind_method(D_METHOD("read_try_lock"), &_RWLock::read_try_lock);

		ClassDB::bind_method(D_METHOD("write_lock"), &_RWLock::write_lock);
		ClassDB::bind_method(D_METHOD("write_unlock"), &_RWLock::write_unlock);
		ClassDB::bind_method(D_METHOD("write_try_lock"), &_RWLock::write_try_lock);
	}
public:
	_RWLock()  = default;
	~_RWLock() = default;

	_FORCE_INLINE_ void read_lock() const { lock.read_lock(); }
	_FORCE_INLINE_ void read_unlock() const { lock.read_unlock(); }
	_FORCE_INLINE_ Error read_try_lock() const { return lock.read_try_lock(); }

	_FORCE_INLINE_ void write_lock() const { lock.write_lock(); }
	_FORCE_INLINE_ void write_unlock() const { lock.write_unlock(); }
	_FORCE_INLINE_ Error write_try_lock() const { return lock.write_try_lock(); }
};

#endif
