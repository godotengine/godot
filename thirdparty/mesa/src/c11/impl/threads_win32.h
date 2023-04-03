/*
 * Copyright 2022 Yonggang Luo
 * SPDX-License-Identifier: MIT
 */

#ifndef C11_IMPL_THREADS_WIN32_H_
#define C11_IMPL_THREADS_WIN32_H_


#ifdef __cplusplus
extern "C" {
#endif

void __threads_win32_tls_callback(void);

#ifdef __cplusplus
}
#endif

#endif
