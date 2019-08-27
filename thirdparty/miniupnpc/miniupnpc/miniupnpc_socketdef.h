/* $Id: miniupnpc_socketdef.h,v 1.1 2018/03/13 23:44:10 nanard Exp $ */
/* Miniupnp project : http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org/
 * Author : Thomas Bernard
 * Copyright (c) 2018 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided within this distribution */
#ifndef MINIUPNPC_SOCKETDEF_H_INCLUDED
#define MINIUPNPC_SOCKETDEF_H_INCLUDED

#ifdef _MSC_VER

#define ISINVALID(s) (INVALID_SOCKET==(s))

#else

#ifndef SOCKET
#define SOCKET int
#endif
#ifndef SSIZE_T
#define SSIZE_T ssize_t
#endif
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#ifndef ISINVALID
#define ISINVALID(s) ((s)<0)
#endif

#endif

#ifdef _MSC_VER
#define MSC_CAST_INT (int)
#else
#define MSC_CAST_INT
#endif

/* definition of PRINT_SOCKET_ERROR */
#ifdef _WIN32
#define PRINT_SOCKET_ERROR(x)    fprintf(stderr, "Socket error: %s, %d\n", x, WSAGetLastError());
#else
#define PRINT_SOCKET_ERROR(x) perror(x)
#endif

#endif /* MINIUPNPC_SOCKETDEF_H_INCLUDED */
