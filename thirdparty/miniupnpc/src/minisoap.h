/* $Id: minisoap.h,v 1.4 2010/04/12 20:39:41 nanard Exp $ */
/* Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2005-2018 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution. */
#ifndef MINISOAP_H_INCLUDED
#define MINISOAP_H_INCLUDED

#include "miniupnpc_socketdef.h"

/*int httpWrite(int, const char *, int, const char *);*/
int soapPostSubmit(SOCKET, const char *, const char *, unsigned short,
		   const char *, const char *, const char *);

#endif

