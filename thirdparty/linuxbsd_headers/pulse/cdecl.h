#ifndef foopulsecdeclhfoo
#define foopulsecdeclhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

/** \file
 * C++ compatibility support */

#ifdef __cplusplus
/** If using C++ this macro enables C mode, otherwise does nothing */
#define PA_C_DECL_BEGIN extern "C" {
/** If using C++ this macros switches back to C++ mode, otherwise does nothing */
#define PA_C_DECL_END }

#else
/** If using C++ this macro enables C mode, otherwise does nothing */
#define PA_C_DECL_BEGIN
/** If using C++ this macros switches back to C++ mode, otherwise does nothing */
#define PA_C_DECL_END

#endif

#endif
