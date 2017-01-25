#include "rw_lock.h"
#include "error_macros.h"
#include <stddef.h>



RWLock* (*RWLock::create_func)()=0;

RWLock *RWLock::create() {

	ERR_FAIL_COND_V( !create_func, 0 );

	return create_func();
}


RWLock::~RWLock() {


}

