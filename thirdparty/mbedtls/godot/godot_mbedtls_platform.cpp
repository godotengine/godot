/**************************************************************************/
/*  godot_mbedtls_platform.cpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "core/os/condition_variable.h"
#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/os/os.h"

#include <mbedtls/build_info.h>
#include <mbedtls/threading.h>
#include <tf-psa-crypto/build_info.h>

extern "C" {
#ifdef MBEDTLS_PLATFORM_ZEROIZE_ALT
static void *(*const volatile memset_func)(void *, int, size_t) = memset;

void mbedtls_platform_zeroize(void *buf, size_t len) {
	memset_func(buf, 0, len);
}

void mbedtls_zeroize_and_free(void *buf, size_t len) {
	if (buf != NULL) {
		mbedtls_platform_zeroize(buf, len);
	}
	free(buf);
}
#endif
#if defined(MBEDTLS_PSA_DRIVER_GET_ENTROPY)
int mbedtls_platform_get_entropy(uint32_t p_flags, size_t *r_estimate_bits, unsigned char *r_output, size_t p_output_size) {
	ERR_FAIL_NULL_V(OS::get_singleton(), -1);
	Error err = OS::get_singleton()->get_entropy(r_output, p_output_size);
	ERR_FAIL_COND_V(err != OK, -1);
	*r_estimate_bits = p_output_size * 8;
	return 0;
}
#endif
#if defined(MBEDTLS_THREADING_ALT)
int godot_mbedtls_mutex_init(mbedtls_platform_mutex_t *p_mutex) {
	ERR_FAIL_NULL_V(p_mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	p_mutex->mutex = memnew(BinaryMutex);
	return 0;
}

void godot_mbedtls_mutex_free(mbedtls_platform_mutex_t *p_mutex) {
	ERR_FAIL_NULL(p_mutex);
	ERR_FAIL_NULL(p_mutex->mutex);
	memdelete((BinaryMutex *)p_mutex->mutex);
}

int godot_mbedtls_mutex_lock(mbedtls_platform_mutex_t *p_mutex) {
	ERR_FAIL_NULL_V(p_mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_mutex->mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	((BinaryMutex *)p_mutex->mutex)->lock();
	return 0;
}

int godot_mbedtls_mutex_unlock(mbedtls_platform_mutex_t *p_mutex) {
	ERR_FAIL_NULL_V(p_mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_mutex->mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	((BinaryMutex *)p_mutex->mutex)->unlock();
	return 0;
}

int godot_mbedtls_cond_init(mbedtls_platform_condition_variable_t *p_cv) {
	ERR_FAIL_NULL_V(p_cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	p_cv->cv = memnew(ConditionVariable);
	return 0;
}
void godot_mbedtls_cond_destroy(mbedtls_platform_condition_variable_t *p_cv) {
	ERR_FAIL_NULL(p_cv);
	ERR_FAIL_NULL(p_cv->cv);
	memdelete((ConditionVariable *)p_cv->cv);
}
int godot_mbedtls_cond_signal(mbedtls_platform_condition_variable_t *p_cv) {
	ERR_FAIL_NULL_V(p_cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_cv->cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	((ConditionVariable *)p_cv->cv)->notify_one();
	return 0;
}
int godot_mbedtls_cond_broadcast(mbedtls_platform_condition_variable_t *p_cv) {
	ERR_FAIL_NULL_V(p_cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_cv->cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	((ConditionVariable *)p_cv->cv)->notify_all();
	return 0;
}
int godot_mbedtls_cond_wait(mbedtls_platform_condition_variable_t *p_cv, mbedtls_platform_mutex_t *p_mutex) {
	// XXX: This function is not currently implemented but also **not used** by mbedTLS.
	// The library documentation states that this function is expected to be called with the mutex already locked.
	// This feels a bit strange (and not possible without changing Godot core API).
	// Use a verbose error so we can catch it if a future mbedTLS version starts using it.
	ERR_FAIL_V(MBEDTLS_ERR_THREADING_USAGE_ERROR);
#if 0
	ERR_FAIL_NULL_V(p_mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_mutex->mutex, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	ERR_FAIL_NULL_V(p_cv->cv, MBEDTLS_ERR_THREADING_USAGE_ERROR);
	MutexLock lock(*((BinaryMutex *)p_mutex->mutex));
	((ConditionVariable *)p_cv->cv)->wait(lock);
	return 0;
#endif
}
#endif
void godot_mbedtls_platform_init() {
#if defined(MBEDTLS_THREADING_ALT)
	mbedtls_threading_set_alt(
			godot_mbedtls_mutex_init,
			godot_mbedtls_mutex_free,
			godot_mbedtls_mutex_lock,
			godot_mbedtls_mutex_unlock,
			godot_mbedtls_cond_init,
			godot_mbedtls_cond_destroy,
			godot_mbedtls_cond_signal,
			godot_mbedtls_cond_broadcast,
			godot_mbedtls_cond_wait);
#endif
}
void godot_mbedtls_platform_free() {
#if defined(MBEDTLS_THREADING_ALT)
	mbedtls_threading_free_alt();
#endif
}
};
