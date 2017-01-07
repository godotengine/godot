
#if defined(UNIX_ENABLED) || defined(PTHREAD_ENABLED)

#include "os/memory.h"
#include "rw_lock_posix.h"
#include "error_macros.h"
#include <stdio.h>

void RWLockPosix::read_lock() {

	int err =pthread_rwlock_rdlock(&rwlock);
	if (err!=0) {
		perror("wtf: ");
	}
	ERR_FAIL_COND(err!=0);
}

void RWLockPosix::read_unlock() {

	pthread_rwlock_unlock(&rwlock);
}

Error RWLockPosix::read_try_lock() {

	if (pthread_rwlock_tryrdlock(&rwlock)!=0) {
		return ERR_BUSY;
	} else {
		return OK;
	}

}

void RWLockPosix::write_lock() {

	int err = pthread_rwlock_wrlock(&rwlock);
	ERR_FAIL_COND(err!=0);
}

void RWLockPosix::write_unlock() {

	pthread_rwlock_unlock(&rwlock);
}

Error RWLockPosix::write_try_lock() {
	if (pthread_rwlock_trywrlock(&rwlock)!=0) {
		return ERR_BUSY;
	} else {
		return OK;
	}
}


RWLock *RWLockPosix::create_func_posix() {

	return memnew( RWLockPosix );
}

void RWLockPosix::make_default() {

	create_func=create_func_posix;
}


RWLockPosix::RWLockPosix() {

	rwlock=PTHREAD_RWLOCK_INITIALIZER;
}


RWLockPosix::~RWLockPosix() {

	pthread_rwlock_destroy(&rwlock);

}

#endif
