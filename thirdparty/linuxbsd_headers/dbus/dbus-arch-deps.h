/* -*- mode: C; c-file-style: "gnu" -*- */
/* dbus-arch-deps.h Header with architecture/compiler specific information, installed to libdir
 *
 * Copyright (C) 2003 Red Hat, Inc.
 *
 * Licensed under the Academic Free License version 2.0
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#if !defined (DBUS_INSIDE_DBUS_H) && !defined (DBUS_COMPILATION)
#error "Only <dbus/dbus.h> can be included directly, this file may disappear or change contents."
#endif

#ifndef DBUS_ARCH_DEPS_H
#define DBUS_ARCH_DEPS_H

#include <dbus/dbus-macros.h>

DBUS_BEGIN_DECLS

/* D-Bus no longer supports platforms with no 64-bit integer type. */
#define DBUS_HAVE_INT64 1
_DBUS_GNUC_EXTENSION typedef long dbus_int64_t;
_DBUS_GNUC_EXTENSION typedef unsigned long dbus_uint64_t;

#define DBUS_INT64_CONSTANT(val)  (_DBUS_GNUC_EXTENSION (val##L))
#define DBUS_UINT64_CONSTANT(val) (_DBUS_GNUC_EXTENSION (val##UL))

typedef int dbus_int32_t;
typedef unsigned int dbus_uint32_t;

typedef short dbus_int16_t;
typedef unsigned short dbus_uint16_t;

/* This is not really arch-dependent, but it's not worth
 * creating an additional generated header just for this
 */
#define DBUS_MAJOR_VERSION 1
#define DBUS_MINOR_VERSION 12
#define DBUS_MICRO_VERSION 2

#define DBUS_VERSION_STRING "1.12.2"

#define DBUS_VERSION ((1 << 16) | (12 << 8) | (2)) 

DBUS_END_DECLS

#endif /* DBUS_ARCH_DEPS_H */
