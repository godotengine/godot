#ifndef RWLOCKPOSIX_H
#define RWLOCKPOSIX_H

#if defined(UNIX_ENABLED) || defined(PTHREAD_ENABLED)

#include <pthread.h>
#include "os/rw_lock.h"

class RWLockPosix : public RWLock {


	pthread_rwlock_t rwlock;

	static RWLock *create_func_posix();

public:

	virtual void read_lock();
	virtual void read_unlock();
	virtual Error read_try_lock();

	virtual void write_lock();
	virtual void write_unlock();
	virtual Error write_try_lock();

	static void make_default();

	RWLockPosix();

	~RWLockPosix();

};

#endif


#endif // RWLOCKPOSIX_H
