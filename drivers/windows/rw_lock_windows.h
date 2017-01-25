#ifndef RWLOCKWINDOWS_H
#define RWLOCKWINDOWS_H

#if defined(WINDOWS_ENABLED)

#include <windows.h>
#include "os/rw_lock.h"

class RWLockWindows : public RWLock {


	SRWLOCK lock;

	static RWLock *create_func_windows();

public:

	virtual void read_lock();
	virtual void read_unlock();
	virtual Error read_try_lock();

	virtual void write_lock();
	virtual void write_unlock();
	virtual Error write_try_lock();

	static void make_default();

	RWLockWindows();

	~RWLockWindows();

};

#endif


#endif // RWLOCKWINDOWS_H
