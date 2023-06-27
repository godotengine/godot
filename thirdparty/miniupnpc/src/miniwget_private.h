/* $Id: miniwget_private.h,v 1.1 2018/04/06 10:17:58 nanard Exp $ */
/* Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2018 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution.
 * */
#ifndef MINIWGET_INTERNAL_H_INCLUDED
#define MINIWGET_INTERNAL_H_INCLUDED

#include "miniupnpc_socketdef.h"

void * getHTTPResponse(SOCKET s, int * size, int * status_code);

#endif
