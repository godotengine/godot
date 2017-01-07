#ifndef RWLOCK_H
#define RWLOCK_H

#include "error_list.h"

class RWLock {
protected:
	static RWLock* (*create_func)();

public:

	virtual void read_lock()=0; ///< Lock the rwlock, block if locked by someone else
	virtual void read_unlock()=0; ///< Unlock the rwlock, let other threads continue
	virtual Error read_try_lock()=0; ///< Attempt to lock the rwlock, OK on success, ERROR means it can't lock.

	virtual void write_lock()=0; ///< Lock the rwlock, block if locked by someone else
	virtual void write_unlock()=0; ///< Unlock the rwlock, let other thwrites continue
	virtual Error write_try_lock()=0; ///< Attempt to lock the rwlock, OK on success, ERROR means it can't lock.

	static RWLock * create(); ///< Create a rwlock

	virtual ~RWLock();
};


class RWLockRead {

	RWLock *lock;
public:

	RWLockRead(RWLock* p_lock) { lock=p_lock; if (lock) lock->read_lock(); }
	~RWLockRead() { if (lock) lock->read_unlock(); }

};

class RWLockWrite {

	RWLock *lock;
public:

	RWLockWrite(RWLock* p_lock) { lock=p_lock; if (lock) lock->write_lock(); }
	~RWLockWrite() { if (lock) lock->write_unlock(); }

};

#endif // RWLOCK_H
