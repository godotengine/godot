/*
 * Copyright © 2003 Felix Kuehling
 * Copyright © 2018 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS, AUTHORS
 * AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 */

#ifndef PROCESS_H
#define PROCESS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *
util_get_process_name(void);

size_t
util_get_process_exec_path(char* process_path, size_t len);

/**
 * Return the name of the current process.
 * \param env_name  the environment variable name used to override
 * \param procname  returns the process name
 * \param size  size of the procname buffer
 * \return  true or false for success, failure
 */
bool
util_get_process_name_may_override(const char *env_name, char *procname, size_t size);

/**
 * Return the command line for the calling process.  This is basically
 * the argv[] array with the arguments separated by spaces.
 * \param cmdline  returns the command line string
 * \param size  size of the cmdline buffer
 * \return  true or false for success, failure
 */
bool
util_get_command_line(char *cmdline, size_t size);

#ifdef __cplusplus
} /* extern C */
#endif

#endif
