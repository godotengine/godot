/* $Id: miniwget_private.h,v 1.1 2018/04/06 10:17:58 nanard Exp $ */
/* Project : miniupnp
 * Author : Thomas Bernard
 * Copyright (c) 2018-2025 Thomas Bernard
 * This software is subject to the conditions detailed in the
 * LICENCE file provided in this distribution.
 * */
#ifndef MINIWGET_INTERNAL_H_INCLUDED
#define MINIWGET_INTERNAL_H_INCLUDED

/*! \file miniwget_private.h
 * \brief Lightweight HTTP client private API
 */
#include "miniupnpc_socketdef.h"

/*! \brief Read a HTTP response from a socket
 *
 * Processed HTTP headers :
 * - `Content-Length`
 * - `Transfer-encoding`
 * return a pointer to the content buffer, which length is saved
 * to the length parameter.
 * \param[in] s socket
 * \param[out] size returned content buffer size
 * \param[out] status_code HTTP Status code
 * \return malloc'ed content buffer
 */
void * getHTTPResponse(SOCKET s, int * size, int * status_code);

/*! \brief parse a HTTP URL
 *
 * URL formats supported :
 * - `http://192.168.1.1/path/xxx`
 * - `http://192.168.1.1:8080/path/xxx`
 * - `http://[2a00:1234:5678:90ab::123]/path/xxx`
 * - `http://[2a00:1234:5678:90ab::123]:8080/path/xxx`
 * - `http://[fe80::1234:5678:90ab%%eth0]/path/xxx`
 * - `http://[fe80::1234:5678:90ab%%eth0]:8080/path/xxx`
 *
 * `%` may be encoded as `%25`
 *
 * \param[in] url URL to parse
 * \param[out] hostname hostname part of the URL (size of MAXHOSTNAMELEN+1)
 * \param[out] port set to the port specified in the URL or 80
 * \param[out] path set to the begining of the path part of the URL
 * \param[out] scope_id set to the interface id if specified in the
 *             link-local IPv6 address
 * \return 0 for failure, 1 for success
 */
int parseURL(const char * url,
             char * hostname, unsigned short * port, char * * path,
             unsigned int * scope_id);

#endif
