/* $Id: miniupnpctypes.h,v 1.4 2025/02/08 23:15:16 nanard Exp $ */
/* Project: miniupnp
 * http://miniupnp.free.fr/ or https://miniupnp.tuxfamily.org
 * Author: Thomas Bernard
 * Copyright (c) 2021-2025 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided within this distribution */
#ifndef MINIUPNPCTYPES_H_INCLUDED
#define MINIUPNPCTYPES_H_INCLUDED

/*! \file miniupnpctypes.h
 * \brief type definitions
 *
 * Use unsigned long long when available :
 * strtoull is C99
 *
 * \def UNSIGNED_INTEGER
 * \brief `unsigned long long` or `unsigned int`
 * \todo int can be 16 bits, so it should be `unsigned long`
 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define UNSIGNED_INTEGER unsigned long long
#else
#define UNSIGNED_INTEGER unsigned int
#endif

#endif
