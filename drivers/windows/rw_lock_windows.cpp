
#if defined(WINDOWS_ENABLED)

#include "os/memory.h"
#include "rw_lock_windows.h"
#include "error_macros.h"
#include <stdio.h>

void RWLockWindows::read_lock() {

	AcquireSRWLockShared(&lock);

}

void RWLockWindows::read_unlock() {

	ReleaseSRWLockShared(&lock);
}

Error RWLockWindows::read_try_lock() {

	if (TryAcquireSRWLockShared(&lock)==0) {
		return ERR_BUSY;
	} else {
		return OK;
	}

}

void RWLockWindows::write_lock() {

	AcquireSRWLockExclusive(&lock);

}

void RWLockWindows::write_unlock() {

	ReleaseSRWLockExclusive(&lock);
}

Error RWLockWindows::write_try_lock() {
	if (TryAcquireSRWLockExclusive(&lock)==0) {
		return ERR_BUSY;
	} else {
		return OK;
	}
}


RWLock *RWLockWindows::create_func_windows() {

	return memnew( RWLockWindows );
}

void RWLockWindows::make_default() {

	create_func=create_func_windows;
}


RWLockWindows::RWLockWindows() {

	InitializeSRWLock(&lock);
}


RWLockWindows::~RWLockWindows() {


}

#endif
