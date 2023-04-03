/*
 * Copyright 2019 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * File operations helpers
 */

#ifndef _OS_FILE_H_
#define _OS_FILE_H_

#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Create a new file and opens it for writing-only.
 * If the given filename already exists, nothing is done and NULL is returned.
 * `errno` gets set to the failure reason; if that is not EEXIST, the caller
 * might want to do something other than trying again.
 */
FILE *
os_file_create_unique(const char *filename, int filemode);

/*
 * Duplicate a file descriptor, making sure not to keep it open after an exec*()
 */
int
os_dupfd_cloexec(int fd);

/*
 * Read a file.
 * Returns a char* that the caller must free(), or NULL and sets errno.
 * If size is not null and no error occurred it's set to the size of the
 * file.
 * Reads files as binary and includes a NUL terminator after the end of the
 * returned buffer.
 */
char *
os_read_file(const char *filename, size_t *size);

/*
 * Try to determine if two file descriptors reference the same file description
 *
 * Return values:
 * - 0:   They reference the same file description
 * - > 0: They do not reference the same file description
 * - < 0: Unable to determine whether they reference the same file description
 */
int
os_same_file_description(int fd1, int fd2);

#ifdef __cplusplus
}
#endif

#endif /* _OS_FILE_H_ */
