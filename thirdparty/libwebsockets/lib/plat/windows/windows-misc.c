/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 */

#ifndef _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif
#include "core/private.h"


uint64_t
lws_time_in_microseconds()
{
#ifndef DELTA_EPOCH_IN_MICROSECS
#define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
#endif
	FILETIME filetime;
	ULARGE_INTEGER datetime;

#ifdef _WIN32_WCE
	GetCurrentFT(&filetime);
#else
	GetSystemTimeAsFileTime(&filetime);
#endif

	/*
	 * As per Windows documentation for FILETIME, copy the resulting
	 * FILETIME structure to a ULARGE_INTEGER structure using memcpy
	 * (using memcpy instead of direct assignment can prevent alignment
	 * faults on 64-bit Windows).
	 */
	memcpy(&datetime, &filetime, sizeof(datetime));

	/* Windows file times are in 100s of nanoseconds. */
	return (datetime.QuadPart / 10) - DELTA_EPOCH_IN_MICROSECS;
}


#ifdef _WIN32_WCE
time_t time(time_t *t)
{
	time_t ret = lws_time_in_microseconds() / 1000000;

	if(t != NULL)
		*t = ret;

	return ret;
}
#endif

LWS_VISIBLE int
lws_get_random(struct lws_context *context, void *buf, int len)
{
	int n;
	char *p = (char *)buf;

	for (n = 0; n < len; n++)
		p[n] = (unsigned char)rand();

	return n;
}


LWS_VISIBLE void
lwsl_emit_syslog(int level, const char *line)
{
	lwsl_emit_stderr(level, line);
}


int kill(int pid, int sig)
{
	lwsl_err("Sorry Windows doesn't support kill().");
	exit(0);
}

int fork(void)
{
	lwsl_err("Sorry Windows doesn't support fork().");
	exit(0);
}


int
lws_plat_recommended_rsa_bits(void)
{
	return 4096;
}



