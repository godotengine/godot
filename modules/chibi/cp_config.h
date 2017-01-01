/*************************************************************************/
/*  cp_config.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef CP_CONFIG_H
#define CP_CONFIG_H


#include "typedefs.h"
#include "error_macros.h"
#include "math_funcs.h"
#include "os/memory.h"
#include "os/copymem.h"

#define CP_PRINTERR(m_err) ERR_PRINT(m_err)
#define CP_ERR_COND(m_cond) ERR_FAIL_COND(m_cond)
#define CP_ERR_COND_V(m_cond,m_ret) ERR_FAIL_COND_V(m_cond,m_ret)
#define CP_FAIL_INDEX(m_index,m_size) ERR_FAIL_INDEX(m_index,m_size)
#define CP_FAIL_INDEX_V(m_index,m_size,m_ret) ERR_FAIL_INDEX_V(m_index,m_size,m_ret)
#define cp_intabs(m_val) ABS(m_val)

#define CP_ALLOC(m_mem) memalloc(m_mem)
#define CP_REALLOC(m_mem,m_size) memrealloc(m_mem,m_size)
#define CP_FREE(m_mem) memfree(m_mem)

#define cp_memzero(m_mem,m_size) zeromem(m_mem,m_size)

#endif // CP_CONFIG_H
