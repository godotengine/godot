/*
 * Copyright 2019 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Socket operations helpers
 */

#ifndef _OS_SOCKET_H_
#define _OS_SOCKET_H_

#include <stdio.h>
#include <stdbool.h>
#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int os_socket_accept(int s);

int os_socket_listen_abstract(const char *path, int count);

ssize_t os_socket_recv(int socket, void *buffer, size_t length, int flags);
ssize_t os_socket_send(int socket, const void *buffer, size_t length, int flags);

void os_socket_block(int s, bool block);
void os_socket_close(int s);

#ifdef __cplusplus
}
#endif

#endif /* _OS_SOCKET_H_ */
