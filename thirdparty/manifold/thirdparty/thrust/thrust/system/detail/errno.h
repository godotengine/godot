/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <thrust/detail/config.h>

// The rationale for the existence of these apparently redundant definitions is
// to provide them portably and to avoid bringing in system headers which might
// pollute the global namespace. These identifiers are in lowercase to avoid
// colliding with the real macros in errno.h.

THRUST_NAMESPACE_BEGIN

namespace system
{

namespace detail
{

static const int eafnosupport    = 9901;
static const int eaddrinuse      = 9902;
static const int eaddrnotavail   = 9903;
static const int eisconn         = 9904;
static const int ebadmsg         = 9905;
static const int econnaborted    = 9906;
static const int ealready        = 9907;
static const int econnrefused    = 9908;
static const int econnreset      = 9909;
static const int edestaddrreq    = 9910;
static const int ehostunreach    = 9911;
static const int eidrm           = 9912;
static const int emsgsize        = 9913;
static const int enetdown        = 9914;
static const int enetreset       = 9915;
static const int enetunreach     = 9916;
static const int enobufs         = 9917;
static const int enolink         = 9918;
static const int enodata         = 9919;
static const int enomsg          = 9920;
static const int enoprotoopt     = 9921;
static const int enosr           = 9922;
static const int enotsock        = 9923;
static const int enostr          = 9924;
static const int enotconn        = 9925;
static const int enotsup         = 9926;
static const int ecanceled       = 9927;
static const int einprogress     = 9928;
static const int eopnotsupp      = 9929;
static const int ewouldblock     = 9930;
static const int eownerdead      = 9931;
static const int eproto          = 9932;
static const int eprotonosupport = 9933;
static const int enotrecoverable = 9934;
static const int etime           = 9935;
static const int etxtbsy         = 9936;
static const int etimedout       = 9938;
static const int eloop           = 9939;
static const int eoverflow       = 9940;
static const int eprototype      = 9941;
static const int enosys          = 9942;
static const int einval          = 9943;
static const int erange          = 9944;
static const int eilseq          = 9945;
static const int e2big           = 9946;
static const int edom            = 9947;
static const int efault          = 9948;
static const int ebadf           = 9949;
static const int epipe           = 9950;
static const int exdev           = 9951;
static const int ebusy           = 9952;
static const int enotempty       = 9953;
static const int enoexec         = 9954;
static const int eexist          = 9955;
static const int efbig           = 9956;
static const int enametoolong    = 9957;
static const int enotty          = 9958;
static const int eintr           = 9959;
static const int espipe          = 9960;
static const int eio             = 9961;
static const int eisdir          = 9962;
static const int echild          = 9963;
static const int enolck          = 9964;
static const int enospc          = 9965;
static const int enxio           = 9966;
static const int enodev          = 9967;
static const int enoent          = 9968;
static const int esrch           = 9969;
static const int enotdir         = 9970;
static const int enomem          = 9971;
static const int eperm           = 9972;
static const int eacces          = 9973;
static const int erofs           = 9974;
static const int edeadlk         = 9975;
static const int eagain          = 9976;
static const int enfile          = 9977;
static const int emfile          = 9978;
static const int emlink          = 9979;

} // end detail

} // end system

THRUST_NAMESPACE_END

