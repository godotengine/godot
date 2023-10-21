/*-
 * Copyright (c) 2011-2012 Irene Ruengeler
 * Copyright (c) 2011-2012 Michael Tuexen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */


#ifdef _WIN32
#include <netinet/sctp_pcb.h>
#include <sys/timeb.h>
#include <iphlpapi.h>
#if !defined(__MINGW32__)
#pragma comment(lib, "iphlpapi.lib")
#endif
#endif
#include <netinet/sctp_os_userspace.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(_WIN32)
/* Adapter to translate Unix thread start routines to Windows thread start
 * routines.
 */
#if defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
static DWORD WINAPI
sctp_create_thread_adapter(void *arg) {
	start_routine_t start_routine = (start_routine_t)arg;
	return start_routine(NULL) == NULL;
}

int
sctp_userspace_thread_create(userland_thread_t *thread, start_routine_t start_routine)
{
	*thread = CreateThread(NULL, 0, sctp_create_thread_adapter,
			       (void *)start_routine, 0, NULL);
	if (*thread == NULL)
		return GetLastError();
	return 0;
}

#if defined(__MINGW32__)
#pragma GCC diagnostic pop
#endif

#else
int
sctp_userspace_thread_create(userland_thread_t *thread, start_routine_t start_routine)
{
	return pthread_create(thread, NULL, start_routine, NULL);
}
#endif

void
sctp_userspace_set_threadname(const char *name)
{
#if defined(__APPLE__)
	pthread_setname_np(name);
#endif
#if defined(__linux__)
	prctl(PR_SET_NAME, name);
#endif
#if defined(__FreeBSD__)
	pthread_set_name_np(pthread_self(), name);
#endif
}

#if !defined(_WIN32) && !defined(__native_client__)
int
sctp_userspace_get_mtu_from_ifn(uint32_t if_index)
{
#if defined(INET) || defined(INET6)
	struct ifreq ifr;
	int fd;
#endif
	int mtu;

	if (if_index == 0xffffffff) {
		mtu = 1280;
	} else {
		mtu = 0;
#if defined(INET) || defined(INET6)
		memset(&ifr, 0, sizeof(struct ifreq));
		if (if_indextoname(if_index, ifr.ifr_name) != NULL) {
			/* TODO can I use the raw socket here and not have to open a new one with each query? */
#if defined(INET)
			if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) >= 0) {
#else
			if ((fd = socket(AF_INET6, SOCK_DGRAM, 0)) >= 0) {
#endif
				if (ioctl(fd, SIOCGIFMTU, &ifr) >= 0) {
					mtu = ifr.ifr_mtu;
				}
				close(fd);
			}
		}
#endif
	}
	return (mtu);
}
#endif

#if defined(__native_client__)
int
sctp_userspace_get_mtu_from_ifn(uint32_t if_index)
{
	return 1280;
}
#endif

#if defined(__APPLE__) || defined(__DragonFly__) || defined(__linux__) || defined(__native_client__) || defined(__NetBSD__) || defined(__QNX__) || defined(_WIN32) || defined(__Fuchsia__) || defined(__EMSCRIPTEN__)
int
timingsafe_bcmp(const void *b1, const void *b2, size_t n)
{
	const unsigned char *p1 = b1, *p2 = b2;
	int ret = 0;

	for (; n > 0; n--)
		ret |= *p1++ ^ *p2++;
	return (ret != 0);
}
#endif

#ifdef _WIN32
int
sctp_userspace_get_mtu_from_ifn(uint32_t if_index)
{
#if defined(INET) || defined(INET6)
	PIP_ADAPTER_ADDRESSES pAdapterAddrs, pAdapt;
	DWORD AdapterAddrsSize, Err;
#endif
	int mtu;

	if (if_index == 0xffffffff) {
		mtu = 1280;
	} else {
		mtu = 0;
#if defined(INET) || defined(INET6)
		AdapterAddrsSize = 0;
		pAdapterAddrs = NULL;
		if ((Err = GetAdaptersAddresses(AF_UNSPEC, 0, NULL, NULL, &AdapterAddrsSize)) != 0) {
			if ((Err != ERROR_BUFFER_OVERFLOW) && (Err != ERROR_INSUFFICIENT_BUFFER)) {
				SCTPDBG(SCTP_DEBUG_USR, "GetAdaptersAddresses() sizing failed with error code %d, AdapterAddrsSize = %d\n", Err, AdapterAddrsSize);
				mtu = -1;
				goto cleanup;
			}
		}
		if ((pAdapterAddrs = (PIP_ADAPTER_ADDRESSES) GlobalAlloc(GPTR, AdapterAddrsSize)) == NULL) {
			SCTPDBG(SCTP_DEBUG_USR, "Memory allocation error!\n");
			mtu = -1;
			goto cleanup;
		}
		if ((Err = GetAdaptersAddresses(AF_UNSPEC, 0, NULL, pAdapterAddrs, &AdapterAddrsSize)) != ERROR_SUCCESS) {
			SCTPDBG(SCTP_DEBUG_USR, "GetAdaptersAddresses() failed with error code %d\n", Err);
			mtu = -1;
			goto cleanup;
		}
		for (pAdapt = pAdapterAddrs; pAdapt; pAdapt = pAdapt->Next) {
			if (pAdapt->IfIndex == if_index) {
				mtu = pAdapt->Mtu;
				break;
			}
		}
	cleanup:
		if (pAdapterAddrs != NULL) {
			GlobalFree(pAdapterAddrs);
		}
#endif
	}
	return (mtu);
}

void
getwintimeofday(struct timeval *tv)
{
	FILETIME filetime;
	ULARGE_INTEGER ularge;

	GetSystemTimeAsFileTime(&filetime);
	ularge.LowPart = filetime.dwLowDateTime;
	ularge.HighPart = filetime.dwHighDateTime;
	/* Change base from Jan 1 1601 00:00:00 to Jan 1 1970 00:00:00 */
#if defined(__MINGW32__)
	ularge.QuadPart -= 116444736000000000ULL;
#else
	ularge.QuadPart -= 116444736000000000UI64;
#endif
	/*
	 * ularge.QuadPart is now the number of 100-nanosecond intervals
	 * since Jan 1 1970 00:00:00.
	 */
#if defined(__MINGW32__)
	tv->tv_sec = (long)(ularge.QuadPart / 10000000ULL);
	tv->tv_usec = (long)((ularge.QuadPart % 10000000ULL) / 10ULL);
#else
	tv->tv_sec = (long)(ularge.QuadPart / 10000000UI64);
	tv->tv_usec = (long)((ularge.QuadPart % 10000000UI64) / 10UI64);
#endif
}

int
win_if_nametoindex(const char *ifname)
{
	IP_ADAPTER_ADDRESSES *addresses, *addr;
	ULONG status, size;
	int index = 0;

	if (!ifname) {
		return 0;
	}

	size = 0;
	status = GetAdaptersAddresses(AF_UNSPEC, 0, NULL, NULL, &size);
	if (status != ERROR_BUFFER_OVERFLOW) {
		return 0;
	}
	addresses = malloc(size);
	status = GetAdaptersAddresses(AF_UNSPEC, 0, NULL, addresses, &size);
	if (status == ERROR_SUCCESS) {
		for (addr = addresses; addr; addr = addr->Next) {
			if (addr->AdapterName && !strcmp(ifname, addr->AdapterName)) {
				index = addr->IfIndex;
				break;
			}
		}
	}

	free(addresses);
	return index;
}

#if WINVER < 0x0600
/* These functions are written based on the code at
 * http://www.cs.wustl.edu/~schmidt/win32-cv-1.html
 * Therefore, for the rest of the file the following applies:
 *
 *
 * Copyright and Licensing Information for ACE(TM), TAO(TM), CIAO(TM),
 * DAnCE(TM), and CoSMIC(TM)
 *
 * [1]ACE(TM), [2]TAO(TM), [3]CIAO(TM), DAnCE(TM), and [4]CoSMIC(TM)
 * (henceforth referred to as "DOC software") are copyrighted by
 * [5]Douglas C. Schmidt and his [6]research group at [7]Washington
 * University, [8]University of California, Irvine, and [9]Vanderbilt
 * University, Copyright (c) 1993-2012, all rights reserved. Since DOC
 * software is open-source, freely available software, you are free to
 * use, modify, copy, and distribute--perpetually and irrevocably--the
 * DOC software source code and object code produced from the source, as
 * well as copy and distribute modified versions of this software. You
 * must, however, include this copyright statement along with any code
 * built using DOC software that you release. No copyright statement
 * needs to be provided if you just ship binary executables of your
 * software products.
 *
 * You can use DOC software in commercial and/or binary software releases
 * and are under no obligation to redistribute any of your source code
 * that is built using DOC software. Note, however, that you may not
 * misappropriate the DOC software code, such as copyrighting it yourself
 * or claiming authorship of the DOC software code, in a way that will
 * prevent DOC software from being distributed freely using an
 * open-source development model. You needn't inform anyone that you're
 * using DOC software in your software, though we encourage you to let
 * [10]us know so we can promote your project in the [11]DOC software
 * success stories.
 *
 * The [12]ACE, [13]TAO, [14]CIAO, [15]DAnCE, and [16]CoSMIC web sites
 * are maintained by the [17]DOC Group at the [18]Institute for Software
 * Integrated Systems (ISIS) and the [19]Center for Distributed Object
 * Computing of Washington University, St. Louis for the development of
 * open-source software as part of the open-source software community.
 * Submissions are provided by the submitter ``as is'' with no warranties
 * whatsoever, including any warranty of merchantability, noninfringement
 * of third party intellectual property, or fitness for any particular
 * purpose. In no event shall the submitter be liable for any direct,
 * indirect, special, exemplary, punitive, or consequential damages,
 * including without limitation, lost profits, even if advised of the
 * possibility of such damages. Likewise, DOC software is provided as is
 * with no warranties of any kind, including the warranties of design,
 * merchantability, and fitness for a particular purpose,
 * noninfringement, or arising from a course of dealing, usage or trade
 * practice. Washington University, UC Irvine, Vanderbilt University,
 * their employees, and students shall have no liability with respect to
 * the infringement of copyrights, trade secrets or any patents by DOC
 * software or any part thereof. Moreover, in no event will Washington
 * University, UC Irvine, or Vanderbilt University, their employees, or
 * students be liable for any lost revenue or profits or other special,
 * indirect and consequential damages.
 *
 * DOC software is provided with no support and without any obligation on
 * the part of Washington University, UC Irvine, Vanderbilt University,
 * their employees, or students to assist in its use, correction,
 * modification, or enhancement. A [20]number of companies around the
 * world provide commercial support for DOC software, however. DOC
 * software is Y2K-compliant, as long as the underlying OS platform is
 * Y2K-compliant. Likewise, DOC software is compliant with the new US
 * daylight savings rule passed by Congress as "The Energy Policy Act of
 * 2005," which established new daylight savings times (DST) rules for
 * the United States that expand DST as of March 2007. Since DOC software
 * obtains time/date and calendaring information from operating systems
 * users will not be affected by the new DST rules as long as they
 * upgrade their operating systems accordingly.
 *
 * The names ACE(TM), TAO(TM), CIAO(TM), DAnCE(TM), CoSMIC(TM),
 * Washington University, UC Irvine, and Vanderbilt University, may not
 * be used to endorse or promote products or services derived from this
 * source without express written permission from Washington University,
 * UC Irvine, or Vanderbilt University. This license grants no permission
 * to call products or services derived from this source ACE(TM),
 * TAO(TM), CIAO(TM), DAnCE(TM), or CoSMIC(TM), nor does it grant
 * permission for the name Washington University, UC Irvine, or
 * Vanderbilt University to appear in their names.
 *
 * If you have any suggestions, additions, comments, or questions, please
 * let [21]me know.
 *
 * [22]Douglas C. Schmidt
 *
 * References
 *
 *  1. http://www.cs.wustl.edu/~schmidt/ACE.html
 *  2. http://www.cs.wustl.edu/~schmidt/TAO.html
 *  3. http://www.dre.vanderbilt.edu/CIAO/
 *  4. http://www.dre.vanderbilt.edu/cosmic/
 *  5. http://www.dre.vanderbilt.edu/~schmidt/
 *  6. http://www.cs.wustl.edu/~schmidt/ACE-members.html
 *  7. http://www.wustl.edu/
 *  8. http://www.uci.edu/
 *  9. http://www.vanderbilt.edu/
 * 10. mailto:doc_group@cs.wustl.edu
 * 11. http://www.cs.wustl.edu/~schmidt/ACE-users.html
 * 12. http://www.cs.wustl.edu/~schmidt/ACE.html
 * 13. http://www.cs.wustl.edu/~schmidt/TAO.html
 * 14. http://www.dre.vanderbilt.edu/CIAO/
 * 15. http://www.dre.vanderbilt.edu/~schmidt/DOC_ROOT/DAnCE/
 * 16. http://www.dre.vanderbilt.edu/cosmic/
 * 17. http://www.dre.vanderbilt.edu/
 * 18. http://www.isis.vanderbilt.edu/
 * 19. http://www.cs.wustl.edu/~schmidt/doc-center.html
 * 20. http://www.cs.wustl.edu/~schmidt/commercial-support.html
 * 21. mailto:d.schmidt@vanderbilt.edu
 * 22. http://www.dre.vanderbilt.edu/~schmidt/
 * 23. http://www.cs.wustl.edu/ACE.html
 */

void
InitializeXPConditionVariable(userland_cond_t *cv)
{
	cv->waiters_count = 0;
	InitializeCriticalSection(&(cv->waiters_count_lock));
	cv->events_[C_SIGNAL] = CreateEvent (NULL, FALSE, FALSE, NULL);
	cv->events_[C_BROADCAST] = CreateEvent (NULL, TRUE, FALSE, NULL);
}

void
DeleteXPConditionVariable(userland_cond_t *cv)
{
	CloseHandle(cv->events_[C_BROADCAST]);
	CloseHandle(cv->events_[C_SIGNAL]);
	DeleteCriticalSection(&(cv->waiters_count_lock));
}

int
SleepXPConditionVariable(userland_cond_t *cv, userland_mutex_t *mtx)
{
	int result, last_waiter;

	EnterCriticalSection(&cv->waiters_count_lock);
	cv->waiters_count++;
	LeaveCriticalSection(&cv->waiters_count_lock);
	LeaveCriticalSection (mtx);
	result = WaitForMultipleObjects(2, cv->events_, FALSE, INFINITE);
	if (result==-1) {
		result = GetLastError();
	}
	EnterCriticalSection(&cv->waiters_count_lock);
	cv->waiters_count--;
	last_waiter = result == (C_SIGNAL + C_BROADCAST && (cv->waiters_count == 0));
	LeaveCriticalSection(&cv->waiters_count_lock);
	if (last_waiter)
		ResetEvent(cv->events_[C_BROADCAST]);
	EnterCriticalSection (mtx);
	return result;
}

void
WakeAllXPConditionVariable(userland_cond_t *cv)
{
	int have_waiters;
	EnterCriticalSection(&cv->waiters_count_lock);
	have_waiters = cv->waiters_count > 0;
	LeaveCriticalSection(&cv->waiters_count_lock);
	if (have_waiters)
		SetEvent (cv->events_[C_BROADCAST]);
}
#endif
#endif
