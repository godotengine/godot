/*-
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2001-2007, by Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2008-2012, by Randall Stewart. All rights reserved.
 * Copyright (c) 2008-2012, by Michael Tuexen. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * a) Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * b) Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the distribution.
 *
 * c) Neither the name of Cisco Systems, Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");
#endif

#include <netinet/sctp_os.h>
#include <netinet/sctp_var.h>
#include <netinet/sctp_pcb.h>
#include <netinet/sctp_header.h>
#include <netinet/sctputil.h>
#include <netinet/sctp_output.h>
#include <netinet/sctp_bsd_addr.h>
#include <netinet/sctp_uio.h>
#include <netinet/sctputil.h>
#include <netinet/sctp_timer.h>
#include <netinet/sctp_asconf.h>
#include <netinet/sctp_sysctl.h>
#include <netinet/sctp_indata.h>
#if defined(__FreeBSD__) && !defined(__Userspace__)
#include <sys/unistd.h>
#endif

/* Declare all of our malloc named types */
MALLOC_DEFINE(SCTP_M_MAP, "sctp_map", "sctp asoc map descriptor");
MALLOC_DEFINE(SCTP_M_STRMI, "sctp_stri", "sctp stream in array");
MALLOC_DEFINE(SCTP_M_STRMO, "sctp_stro", "sctp stream out array");
MALLOC_DEFINE(SCTP_M_ASC_ADDR, "sctp_aadr", "sctp asconf address");
MALLOC_DEFINE(SCTP_M_ASC_IT, "sctp_a_it", "sctp asconf iterator");
MALLOC_DEFINE(SCTP_M_AUTH_CL, "sctp_atcl", "sctp auth chunklist");
MALLOC_DEFINE(SCTP_M_AUTH_KY, "sctp_atky", "sctp auth key");
MALLOC_DEFINE(SCTP_M_AUTH_HL, "sctp_athm", "sctp auth hmac list");
MALLOC_DEFINE(SCTP_M_AUTH_IF, "sctp_athi", "sctp auth info");
MALLOC_DEFINE(SCTP_M_STRESET, "sctp_stre", "sctp stream reset");
MALLOC_DEFINE(SCTP_M_CMSG, "sctp_cmsg", "sctp CMSG buffer");
MALLOC_DEFINE(SCTP_M_COPYAL, "sctp_cpal", "sctp copy all");
MALLOC_DEFINE(SCTP_M_VRF, "sctp_vrf", "sctp vrf struct");
MALLOC_DEFINE(SCTP_M_IFA, "sctp_ifa", "sctp ifa struct");
MALLOC_DEFINE(SCTP_M_IFN, "sctp_ifn", "sctp ifn struct");
MALLOC_DEFINE(SCTP_M_TIMW, "sctp_timw", "sctp time block");
MALLOC_DEFINE(SCTP_M_MVRF, "sctp_mvrf", "sctp mvrf pcb list");
MALLOC_DEFINE(SCTP_M_ITER, "sctp_iter", "sctp iterator control");
MALLOC_DEFINE(SCTP_M_SOCKOPT, "sctp_socko", "sctp socket option");
MALLOC_DEFINE(SCTP_M_MCORE, "sctp_mcore", "sctp mcore queue");

/* Global NON-VNET structure that controls the iterator */
struct iterator_control sctp_it_ctl;
#if !(defined(__FreeBSD__) && !defined(__Userspace__))

static void
sctp_cleanup_itqueue(void)
{
	struct sctp_iterator *it, *nit;

	TAILQ_FOREACH_SAFE(it, &sctp_it_ctl.iteratorhead, sctp_nxt_itr, nit) {
		if (it->function_atend != NULL) {
			(*it->function_atend) (it->pointer, it->val);
		}
		TAILQ_REMOVE(&sctp_it_ctl.iteratorhead, it, sctp_nxt_itr);
		SCTP_FREE(it, SCTP_M_ITER);
	}
}
#endif
#if defined(__Userspace__)
/*__Userspace__ TODO if we use thread based iterator
 * then the implementation of wakeup will need to change.
 * Currently we are using timeo_cond for ident so_timeo
 * but that is not sufficient if we need to use another ident
 * like wakeup(&sctppcbinfo.iterator_running);
 */
#endif

void
sctp_wakeup_iterator(void)
{
#if defined(SCTP_PROCESS_LEVEL_LOCKS)
#if defined(_WIN32)
	WakeAllConditionVariable(&sctp_it_ctl.iterator_wakeup);
#else
	pthread_cond_broadcast(&sctp_it_ctl.iterator_wakeup);
#endif
#else
	wakeup(&sctp_it_ctl.iterator_running);
#endif
}

#if defined(__Userspace__)
static void *
#else
static void
#endif
sctp_iterator_thread(void *v SCTP_UNUSED)
{
#if defined(__Userspace__)
	sctp_userspace_set_threadname("SCTP iterator");
#endif
	SCTP_IPI_ITERATOR_WQ_LOCK();
	/* In FreeBSD this thread never terminates. */
#if defined(__FreeBSD__) && !defined(__Userspace__)
	for (;;) {
#else
	while ((sctp_it_ctl.iterator_flags & SCTP_ITERATOR_MUST_EXIT) == 0) {
#endif
#if !defined(__Userspace__)
		msleep(&sctp_it_ctl.iterator_running,
#if defined(__FreeBSD__)
		       &sctp_it_ctl.ipi_iterator_wq_mtx,
#elif defined(__APPLE__)
		       sctp_it_ctl.ipi_iterator_wq_mtx,
#endif
		       0, "waiting_for_work", 0);
#else
#if defined(_WIN32)
		SleepConditionVariableCS(&sctp_it_ctl.iterator_wakeup, &sctp_it_ctl.ipi_iterator_wq_mtx, INFINITE);
#else
		pthread_cond_wait(&sctp_it_ctl.iterator_wakeup, &sctp_it_ctl.ipi_iterator_wq_mtx);
#endif
#endif
#if !(defined(__FreeBSD__) && !defined(__Userspace__))
		if (sctp_it_ctl.iterator_flags & SCTP_ITERATOR_MUST_EXIT) {
			break;
		}
#endif
		sctp_iterator_worker();
	}
#if !(defined(__FreeBSD__) && !defined(__Userspace__))
	/* Now this thread needs to be terminated */
	sctp_cleanup_itqueue();
	sctp_it_ctl.iterator_flags |= SCTP_ITERATOR_EXITED;
	SCTP_IPI_ITERATOR_WQ_UNLOCK();
#if defined(__Userspace__)
	sctp_wakeup_iterator();
	return (NULL);
#else
	wakeup(&sctp_it_ctl.iterator_flags);
	thread_terminate(current_thread());
#ifdef INVARIANTS
	panic("Hmm. thread_terminate() continues...");
#endif
#endif
#endif
}

void
sctp_startup_iterator(void)
{
	if (sctp_it_ctl.thread_proc) {
		/* You only get one */
		return;
	}
	/* Initialize global locks here, thus only once. */
	SCTP_ITERATOR_LOCK_INIT();
	SCTP_IPI_ITERATOR_WQ_INIT();
	TAILQ_INIT(&sctp_it_ctl.iteratorhead);
#if defined(__Userspace__)
	if (sctp_userspace_thread_create(&sctp_it_ctl.thread_proc, &sctp_iterator_thread)) {
		SCTP_PRINTF("ERROR: Creating sctp_iterator_thread failed.\n");
	} else {
		SCTP_BASE_VAR(iterator_thread_started) = 1;
	}
#elif defined(__FreeBSD__)
	kproc_create(sctp_iterator_thread,
	             (void *)NULL,
	             &sctp_it_ctl.thread_proc,
	             0,
	             SCTP_KTHREAD_PAGES,
	             SCTP_KTRHEAD_NAME);
#elif defined(__APPLE__)
	kernel_thread_start((thread_continue_t)sctp_iterator_thread, NULL, &sctp_it_ctl.thread_proc);
#endif
}

#ifdef INET6

#if defined(__Userspace__)
/* __Userspace__ TODO. struct in6_ifaddr is defined in sys/netinet6/in6_var.h
   ip6_use_deprecated is defined as  int ip6_use_deprecated = 1; in /src/sys/netinet6/in6_proto.c
 */
void
sctp_gather_internal_ifa_flags(struct sctp_ifa *ifa)
{
    return; /* stub */
}
#else
void
sctp_gather_internal_ifa_flags(struct sctp_ifa *ifa)
{
	struct in6_ifaddr *ifa6;

	ifa6 = (struct in6_ifaddr *)ifa->ifa;
	ifa->flags = ifa6->ia6_flags;
	if (!MODULE_GLOBAL(ip6_use_deprecated)) {
		if (ifa->flags &
		    IN6_IFF_DEPRECATED) {
			ifa->localifa_flags |= SCTP_ADDR_IFA_UNUSEABLE;
		} else {
			ifa->localifa_flags &= ~SCTP_ADDR_IFA_UNUSEABLE;
		}
	} else {
		ifa->localifa_flags &= ~SCTP_ADDR_IFA_UNUSEABLE;
	}
	if (ifa->flags &
	    (IN6_IFF_DETACHED |
	     IN6_IFF_ANYCAST |
	     IN6_IFF_NOTREADY)) {
		ifa->localifa_flags |= SCTP_ADDR_IFA_UNUSEABLE;
	} else {
		ifa->localifa_flags &= ~SCTP_ADDR_IFA_UNUSEABLE;
	}
}
#endif /* __Userspace__ */
#endif /* INET6 */

#if !defined(__Userspace__)
static uint32_t
sctp_is_desired_interface_type(struct ifnet *ifn)
{
	int result;

	/* check the interface type to see if it's one we care about */
#if defined(__APPLE__) && !defined(__Userspace__)
	switch(ifnet_type(ifn)) {
#else
	switch (ifn->if_type) {
#endif
	case IFT_ETHER:
	case IFT_ISO88023:
	case IFT_ISO88024:
	case IFT_ISO88025:
	case IFT_ISO88026:
	case IFT_STARLAN:
	case IFT_P10:
	case IFT_P80:
	case IFT_HY:
	case IFT_FDDI:
	case IFT_XETHER:
	case IFT_ISDNBASIC:
	case IFT_ISDNPRIMARY:
	case IFT_PTPSERIAL:
	case IFT_OTHER:
	case IFT_PPP:
	case IFT_LOOP:
	case IFT_SLIP:
	case IFT_GIF:
	case IFT_L2VLAN:
	case IFT_STF:
#if !(defined(__APPLE__) && !defined(__Userspace__))
	case IFT_IP:
	case IFT_IPOVERCDLC:
	case IFT_IPOVERCLAW:
	case IFT_PROPVIRTUAL: /* NetGraph Virtual too */
	case IFT_VIRTUALIPADDRESS:
#endif
		result = 1;
		break;
	default:
		result = 0;
	}

	return (result);
}
#endif
#if defined(__APPLE__) && !defined(__Userspace__)

int
sctp_is_vmware_interface(struct ifnet *ifn)
{
	return (strncmp(ifnet_name(ifn), "vmnet", 5) == 0);
}

#endif

#if defined(_WIN32) && defined(__Userspace__)
#ifdef MALLOC
#undef MALLOC
#define MALLOC(x) HeapAlloc(GetProcessHeap(), 0, (x))
#endif
#ifdef FREE
#undef FREE
#define FREE(x) HeapFree(GetProcessHeap(), 0, (x))
#endif
static void
sctp_init_ifns_for_vrf(int vrfid)
{
#if defined(INET) || defined(INET6)
	struct sctp_ifa *sctp_ifa;
	DWORD Err, AdapterAddrsSize;
	PIP_ADAPTER_ADDRESSES pAdapterAddrs, pAdapt;
	PIP_ADAPTER_UNICAST_ADDRESS pUnicast;
#endif

#ifdef INET
	AdapterAddrsSize = 0;

	if ((Err = GetAdaptersAddresses(AF_INET, 0, NULL, NULL, &AdapterAddrsSize)) != 0) {
		if ((Err != ERROR_BUFFER_OVERFLOW) && (Err != ERROR_INSUFFICIENT_BUFFER)) {
			SCTP_PRINTF("GetAdaptersV4Addresses() sizing failed with error code %d\n", Err);
			SCTP_PRINTF("err = %d; AdapterAddrsSize = %d\n", Err, AdapterAddrsSize);
			return;
		}
	}

	/* Allocate memory from sizing information */
	if ((pAdapterAddrs = (PIP_ADAPTER_ADDRESSES) GlobalAlloc(GPTR, AdapterAddrsSize)) == NULL) {
		SCTP_PRINTF("Memory allocation error!\n");
		return;
	}
	/* Get actual adapter information */
	if ((Err = GetAdaptersAddresses(AF_INET, 0, NULL, pAdapterAddrs, &AdapterAddrsSize)) != ERROR_SUCCESS) {
		SCTP_PRINTF("GetAdaptersV4Addresses() failed with error code %d\n", Err);
		FREE(pAdapterAddrs);
		return;
	}
	/* Enumerate through each returned adapter and save its information */
	for (pAdapt = pAdapterAddrs; pAdapt; pAdapt = pAdapt->Next) {
		if (pAdapt->IfType == IF_TYPE_IEEE80211 || pAdapt->IfType == IF_TYPE_ETHERNET_CSMACD) {
			for (pUnicast = pAdapt->FirstUnicastAddress; pUnicast; pUnicast = pUnicast->Next) {
				if (IN4_ISLINKLOCAL_ADDRESS(&(((struct sockaddr_in *)(pUnicast->Address.lpSockaddr))->sin_addr))) {
					continue;
				}
				sctp_ifa = sctp_add_addr_to_vrf(0,
				                                NULL,
				                                pAdapt->IfIndex,
				                                (pAdapt->IfType == IF_TYPE_IEEE80211)?MIB_IF_TYPE_ETHERNET:pAdapt->IfType,
				                                pAdapt->AdapterName,
				                                NULL,
				                                pUnicast->Address.lpSockaddr,
				                                pAdapt->Flags,
				                                0);
				if (sctp_ifa) {
					sctp_ifa->localifa_flags &= ~SCTP_ADDR_DEFER_USE;
				}
			}
		}
	}
	FREE(pAdapterAddrs);
#endif
#ifdef INET6
	AdapterAddrsSize = 0;

	if ((Err = GetAdaptersAddresses(AF_INET6, 0, NULL, NULL, &AdapterAddrsSize)) != 0) {
		if ((Err != ERROR_BUFFER_OVERFLOW) && (Err != ERROR_INSUFFICIENT_BUFFER)) {
			SCTP_PRINTF("GetAdaptersV6Addresses() sizing failed with error code %d\n", Err);
			SCTP_PRINTF("err = %d; AdapterAddrsSize = %d\n", Err, AdapterAddrsSize);
			return;
		}
	}
	/* Allocate memory from sizing information */
	if ((pAdapterAddrs = (PIP_ADAPTER_ADDRESSES) GlobalAlloc(GPTR, AdapterAddrsSize)) == NULL) {
		SCTP_PRINTF("Memory allocation error!\n");
		return;
	}
	/* Get actual adapter information */
	if ((Err = GetAdaptersAddresses(AF_INET6, 0, NULL, pAdapterAddrs, &AdapterAddrsSize)) != ERROR_SUCCESS) {
		SCTP_PRINTF("GetAdaptersV6Addresses() failed with error code %d\n", Err);
		FREE(pAdapterAddrs);
		return;
	}
	/* Enumerate through each returned adapter and save its information */
	for (pAdapt = pAdapterAddrs; pAdapt; pAdapt = pAdapt->Next) {
		if (pAdapt->IfType == IF_TYPE_IEEE80211 || pAdapt->IfType == IF_TYPE_ETHERNET_CSMACD) {
			for (pUnicast = pAdapt->FirstUnicastAddress; pUnicast; pUnicast = pUnicast->Next) {
				sctp_ifa = sctp_add_addr_to_vrf(0,
				                                NULL,
				                                pAdapt->Ipv6IfIndex,
				                                (pAdapt->IfType == IF_TYPE_IEEE80211)?MIB_IF_TYPE_ETHERNET:pAdapt->IfType,
				                                pAdapt->AdapterName,
				                                NULL,
				                                pUnicast->Address.lpSockaddr,
				                                pAdapt->Flags,
				                                0);
				if (sctp_ifa) {
					sctp_ifa->localifa_flags &= ~SCTP_ADDR_DEFER_USE;
				}
			}
		}
	}
	FREE(pAdapterAddrs);
#endif
}
#elif defined(__Userspace__)
static void
sctp_init_ifns_for_vrf(int vrfid)
{
#if defined(INET) || defined(INET6)
	int rc;
	struct ifaddrs *ifa, *ifas;
	struct sctp_ifa *sctp_ifa;
	uint32_t ifa_flags;

	rc = getifaddrs(&ifas);
	if (rc != 0) {
		return;
	}
	for (ifa = ifas; ifa; ifa = ifa->ifa_next) {
		if (ifa->ifa_addr == NULL) {
			continue;
		}
#if !defined(INET)
		if (ifa->ifa_addr->sa_family != AF_INET6) {
			/* non inet6 skip */
			continue;
		}
#elif !defined(INET6)
		if (ifa->ifa_addr->sa_family != AF_INET) {
			/* non inet skip */
			continue;
		}
#else
		if ((ifa->ifa_addr->sa_family != AF_INET) && (ifa->ifa_addr->sa_family != AF_INET6)) {
			/* non inet/inet6 skip */
			continue;
		}
#endif
#if defined(INET6)
		if ((ifa->ifa_addr->sa_family == AF_INET6) &&
		    IN6_IS_ADDR_UNSPECIFIED(&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr)) {
			/* skip unspecified addresses */
			continue;
		}
#endif
#if defined(INET)
		if (ifa->ifa_addr->sa_family == AF_INET &&
		    ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr == 0) {
			continue;
		}
#endif
		ifa_flags = 0;
		sctp_ifa = sctp_add_addr_to_vrf(vrfid,
		                                NULL,
		                                if_nametoindex(ifa->ifa_name),
		                                0,
		                                ifa->ifa_name,
		                                NULL,
		                                ifa->ifa_addr,
		                                ifa_flags,
		                                0);
		if (sctp_ifa) {
			sctp_ifa->localifa_flags &= ~SCTP_ADDR_DEFER_USE;
		}
	}
	freeifaddrs(ifas);
#endif
}
#endif
#if defined(__APPLE__) && !defined(__Userspace__)
static void
sctp_init_ifns_for_vrf(int vrfid)
{
	/* Here we must apply ANY locks needed by the
	 * IFN we access and also make sure we lock
	 * any IFA that exists as we float through the
	 * list of IFA's
	 */
	struct ifnet **ifnetlist;
	uint32_t i, j, count;
	char name[SCTP_IFNAMSIZ];
	struct ifnet *ifn;
	struct ifaddr **ifaddrlist;
	struct ifaddr *ifa;
	struct in6_ifaddr *ifa6;
	struct sctp_ifa *sctp_ifa;
	uint32_t ifa_flags;

	if (ifnet_list_get(IFNET_FAMILY_ANY, &ifnetlist, &count) != 0) {
		return;
	}
	for (i = 0; i < count; i++) {
		ifn = ifnetlist[i];
		if (SCTP_BASE_SYSCTL(sctp_ignore_vmware_interfaces) && sctp_is_vmware_interface(ifn)) {
			continue;
		}
		if (sctp_is_desired_interface_type(ifn) == 0) {
			/* non desired type */
			continue;
		}
		if (ifnet_get_address_list(ifn, &ifaddrlist) != 0) {
			continue;
		}
		for (j = 0; ifaddrlist[j] != NULL; j++) {
			ifa = ifaddrlist[j];
			if (ifa->ifa_addr == NULL) {
				continue;
			}
			if ((ifa->ifa_addr->sa_family != AF_INET) && (ifa->ifa_addr->sa_family != AF_INET6)) {
				/* non inet/inet6 skip */
				continue;
			}
			if (ifa->ifa_addr->sa_family == AF_INET6) {
				if (IN6_IS_ADDR_UNSPECIFIED(&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr)) {
					/* skip unspecified addresses */
					continue;
				}
			} else {
				if (((struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr == INADDR_ANY) {
					continue;
				}
			}
			if (ifa->ifa_addr->sa_family == AF_INET6) {
				ifa6 = (struct in6_ifaddr *)ifa;
				ifa_flags = ifa6->ia6_flags;
			} else {
				ifa_flags = 0;
			}
			SCTP_SNPRINTF(name, SCTP_IFNAMSIZ, "%s%d", ifnet_name(ifn), ifnet_unit(ifn));
			sctp_ifa = sctp_add_addr_to_vrf(vrfid,
			                                (void *)ifn, /* XXX */
			                                ifnet_index(ifn),
			                                ifnet_type(ifn),
			                                name,
			                                (void *)ifa, /* XXX */
			                                ifa->ifa_addr,
			                                ifa_flags,
			                                0);
			if (sctp_ifa) {
				sctp_ifa->localifa_flags &= ~SCTP_ADDR_DEFER_USE;
			}
		}
		ifnet_free_address_list(ifaddrlist);
	}
	ifnet_list_free(ifnetlist);
}
#endif
#if defined(__FreeBSD__) && !defined(__Userspace__)
static void
sctp_init_ifns_for_vrf(int vrfid)
{
	/* Here we must apply ANY locks needed by the
	 * IFN we access and also make sure we lock
	 * any IFA that exists as we float through the
	 * list of IFA's
	 */
	struct epoch_tracker et;
	struct ifnet *ifn;
	struct ifaddr *ifa;
	struct sctp_ifa *sctp_ifa;
	uint32_t ifa_flags;
#ifdef INET6
	struct in6_ifaddr *ifa6;
#endif

	IFNET_RLOCK();
	NET_EPOCH_ENTER(et);
	CK_STAILQ_FOREACH(ifn, &MODULE_GLOBAL(ifnet), if_link) {
		if (sctp_is_desired_interface_type(ifn) == 0) {
			/* non desired type */
			continue;
		}
		CK_STAILQ_FOREACH(ifa, &ifn->if_addrhead, ifa_link) {
			if (ifa->ifa_addr == NULL) {
				continue;
			}
			switch (ifa->ifa_addr->sa_family) {
#ifdef INET
			case AF_INET:
				if (((struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr == 0) {
					continue;
				}
				break;
#endif
#ifdef INET6
			case AF_INET6:
				if (IN6_IS_ADDR_UNSPECIFIED(&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr)) {
					/* skip unspecified addresses */
					continue;
				}
				break;
#endif
			default:
				continue;
			}
			switch (ifa->ifa_addr->sa_family) {
#ifdef INET
			case AF_INET:
				ifa_flags = 0;
				break;
#endif
#ifdef INET6
			case AF_INET6:
				ifa6 = (struct in6_ifaddr *)ifa;
				ifa_flags = ifa6->ia6_flags;
				break;
#endif
			default:
				ifa_flags = 0;
				break;
			}
			sctp_ifa = sctp_add_addr_to_vrf(vrfid,
			                                (void *)ifn,
			                                ifn->if_index,
			                                ifn->if_type,
			                                ifn->if_xname,
			                                (void *)ifa,
			                                ifa->ifa_addr,
			                                ifa_flags,
			                                0);
			if (sctp_ifa) {
				sctp_ifa->localifa_flags &= ~SCTP_ADDR_DEFER_USE;
			}
		}
	}
	NET_EPOCH_EXIT(et);
	IFNET_RUNLOCK();
}
#endif

void
sctp_init_vrf_list(int vrfid)
{
	if (vrfid > SCTP_MAX_VRF_ID)
		/* can't do that */
		return;

	/* Don't care about return here */
	(void)sctp_allocate_vrf(vrfid);

	/* Now we need to build all the ifn's
	 * for this vrf and there addresses
	 */
	sctp_init_ifns_for_vrf(vrfid);
}

void
sctp_addr_change(struct ifaddr *ifa, int cmd)
{
#if defined(__Userspace__)
        return;
#else
	uint32_t ifa_flags = 0;

	if (SCTP_BASE_VAR(sctp_pcb_initialized) == 0) {
		return;
	}
	/* BSD only has one VRF, if this changes
	 * we will need to hook in the right
	 * things here to get the id to pass to
	 * the address management routine.
	 */
	if (SCTP_BASE_VAR(first_time) == 0) {
		/* Special test to see if my ::1 will showup with this */
		SCTP_BASE_VAR(first_time) = 1;
		sctp_init_ifns_for_vrf(SCTP_DEFAULT_VRFID);
	}

	if ((cmd != RTM_ADD) && (cmd != RTM_DELETE)) {
		/* don't know what to do with this */
		return;
	}

	if (ifa->ifa_addr == NULL) {
		return;
	}
	if (sctp_is_desired_interface_type(ifa->ifa_ifp) == 0) {
		/* non desired type */
		return;
	}
	switch (ifa->ifa_addr->sa_family) {
#ifdef INET
	case AF_INET:
		if (((struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr == 0) {
			return;
		}
		break;
#endif
#ifdef INET6
	case AF_INET6:
		ifa_flags = ((struct in6_ifaddr *)ifa)->ia6_flags;
		if (IN6_IS_ADDR_UNSPECIFIED(&((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr)) {
			/* skip unspecified addresses */
			return;
		}
		break;
#endif
	default:
		/* non inet/inet6 skip */
		return;
	}
	if (cmd == RTM_ADD) {
		(void)sctp_add_addr_to_vrf(SCTP_DEFAULT_VRFID, (void *)ifa->ifa_ifp,
#if defined(__APPLE__) && !defined(__Userspace__)
		                           ifnet_index(ifa->ifa_ifp), ifnet_type(ifa->ifa_ifp), ifnet_name(ifa->ifa_ifp),
#else
		                           ifa->ifa_ifp->if_index, ifa->ifa_ifp->if_type, ifa->ifa_ifp->if_xname,
#endif
		                           (void *)ifa, ifa->ifa_addr, ifa_flags, 1);
	} else {
		sctp_del_addr_from_vrf(SCTP_DEFAULT_VRFID, ifa->ifa_addr,
#if defined(__APPLE__) && !defined(__Userspace__)
		                       ifnet_index(ifa->ifa_ifp),
		                       ifnet_name(ifa->ifa_ifp));
#else
		                       ifa->ifa_ifp->if_index,
		                       ifa->ifa_ifp->if_xname);
#endif

		/* We don't bump refcount here so when it completes
		 * the final delete will happen.
		 */
	}
#endif
}

#if defined(__FreeBSD__) && !defined(__Userspace__)
void
sctp_addr_change_event_handler(void *arg __unused, struct ifaddr *ifa, int cmd) {
	sctp_addr_change(ifa, cmd);
}
#endif
#if defined(__APPLE__) && !defined(__Userspace__)
void
sctp_add_or_del_interfaces(int (*pred)(struct ifnet *), int add)
{
	struct ifnet **ifnetlist;
	struct ifaddr **ifaddrlist;
	uint32_t i, j, count;

	if (ifnet_list_get(IFNET_FAMILY_ANY, &ifnetlist, &count) != 0) {
		return;
	}
	for (i = 0; i < count; i++) {
		if (!(*pred)(ifnetlist[i])) {
			continue;
		}
		if (ifnet_get_address_list(ifnetlist[i], &ifaddrlist) != 0) {
			continue;
		}
		for (j = 0; ifaddrlist[j] != NULL; j++) {
			sctp_addr_change(ifaddrlist[j], add ? RTM_ADD : RTM_DELETE);
		}
		ifnet_free_address_list(ifaddrlist);
	}
	ifnet_list_free(ifnetlist);
	return;
}
#endif

struct mbuf *
sctp_get_mbuf_for_msg(unsigned int space_needed, int want_header,
		      int how, int allonebuf, int type)
{
	struct mbuf *m = NULL;
#if defined(__FreeBSD__) || defined(__Userspace__)
#if defined(__Userspace__)
	m = m_getm2(NULL, space_needed, how, type, want_header ? M_PKTHDR : 0, allonebuf);
#else
	m = m_getm2(NULL, space_needed, how, type, want_header ? M_PKTHDR : 0);
#endif
	if (m == NULL) {
		/* bad, no memory */
		return (m);
	}
#if !defined(__Userspace__)
	if (allonebuf) {
		if (SCTP_BUF_SIZE(m) < space_needed) {
			m_freem(m);
			return (NULL);
		}
		KASSERT(SCTP_BUF_NEXT(m) == NULL, ("%s: no chain allowed", __func__));
	}
#endif
#ifdef SCTP_MBUF_LOGGING
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_MBUF_LOGGING_ENABLE) {
		sctp_log_mb(m, SCTP_MBUF_IALLOC);
	}
#endif
#else
	int mbuf_threshold;
	unsigned int size;

	if (want_header) {
		MGETHDR(m, how, type);
		size = MHLEN;
	} else {
		MGET(m, how, type);
		size = MLEN;
	}
	if (m == NULL) {
		return (NULL);
	}
	if (allonebuf == 0) {
		mbuf_threshold = SCTP_BASE_SYSCTL(sctp_mbuf_threshold_count);
	} else {
		mbuf_threshold = 1;
	}

	if (space_needed > (unsigned int)(((mbuf_threshold - 1) * MLEN) + MHLEN)) {
		MCLGET(m, how);
		if (m == NULL) {
			return (NULL);
		}
		if (SCTP_BUF_IS_EXTENDED(m) == 0) {
			sctp_m_freem(m);
			return (NULL);
		}
		size = SCTP_BUF_EXTEND_SIZE(m);
	}
	if (allonebuf != 0 && size < space_needed) {
		m_freem(m);
		return (NULL);
	}
	SCTP_BUF_LEN(m) = 0;
	SCTP_BUF_NEXT(m) = SCTP_BUF_NEXT_PKT(m) = NULL;
#ifdef SCTP_MBUF_LOGGING
	if (SCTP_BASE_SYSCTL(sctp_logging_level) & SCTP_MBUF_LOGGING_ENABLE) {
		sctp_log_mb(m, SCTP_MBUF_IALLOC);
	}
#endif
#endif
	return (m);
}

#ifdef SCTP_PACKET_LOGGING
void
sctp_packet_log(struct mbuf *m)
{
	int *lenat, thisone;
	void *copyto;
	uint32_t *tick_tock;
	int length;
	int total_len;
	int grabbed_lock = 0;
	int value, newval, thisend, thisbegin;
	/*
	 * Buffer layout.
	 * -sizeof this entry (total_len)
	 * -previous end      (value)
	 * -ticks of log      (ticks)
	 * o -ip packet
	 * o -as logged
	 * - where this started (thisbegin)
	 * x <--end points here
	 */
	length = SCTP_HEADER_LEN(m);
	total_len = SCTP_SIZE32((length + (4 * sizeof(int))));
	/* Log a packet to the buffer. */
	if (total_len> SCTP_PACKET_LOG_SIZE) {
		/* Can't log this packet I have not a buffer big enough */
		return;
	}
	if (length < (int)(SCTP_MIN_V4_OVERHEAD + sizeof(struct sctp_cookie_ack_chunk))) {
		return;
	}
	atomic_add_int(&SCTP_BASE_VAR(packet_log_writers), 1);
 try_again:
	if (SCTP_BASE_VAR(packet_log_writers) > SCTP_PKTLOG_WRITERS_NEED_LOCK) {
		SCTP_IP_PKTLOG_LOCK();
		grabbed_lock = 1;
	again_locked:
		value = SCTP_BASE_VAR(packet_log_end);
		newval = SCTP_BASE_VAR(packet_log_end) + total_len;
		if (newval >= SCTP_PACKET_LOG_SIZE) {
			/* we wrapped */
			thisbegin = 0;
			thisend = total_len;
		} else {
			thisbegin = SCTP_BASE_VAR(packet_log_end);
			thisend = newval;
		}
		if (!(atomic_cmpset_int(&SCTP_BASE_VAR(packet_log_end), value, thisend))) {
			goto again_locked;
		}
	} else {
		value = SCTP_BASE_VAR(packet_log_end);
		newval = SCTP_BASE_VAR(packet_log_end) + total_len;
		if (newval >= SCTP_PACKET_LOG_SIZE) {
			/* we wrapped */
			thisbegin = 0;
			thisend = total_len;
		} else {
			thisbegin = SCTP_BASE_VAR(packet_log_end);
			thisend = newval;
		}
		if (!(atomic_cmpset_int(&SCTP_BASE_VAR(packet_log_end), value, thisend))) {
			goto try_again;
		}
	}
	/* Sanity check */
	if (thisend >= SCTP_PACKET_LOG_SIZE) {
		SCTP_PRINTF("Insanity stops a log thisbegin:%d thisend:%d writers:%d lock:%d end:%d\n",
		            thisbegin,
		            thisend,
		            SCTP_BASE_VAR(packet_log_writers),
		            grabbed_lock,
		            SCTP_BASE_VAR(packet_log_end));
		SCTP_BASE_VAR(packet_log_end) = 0;
		goto no_log;
	}
	lenat = (int *)&SCTP_BASE_VAR(packet_log_buffer)[thisbegin];
	*lenat = total_len;
	lenat++;
	*lenat = value;
	lenat++;
	tick_tock = (uint32_t *)lenat;
	lenat++;
	*tick_tock = sctp_get_tick_count();
	copyto = (void *)lenat;
	thisone = thisend - sizeof(int);
	lenat = (int *)&SCTP_BASE_VAR(packet_log_buffer)[thisone];
	*lenat = thisbegin;
	if (grabbed_lock) {
		SCTP_IP_PKTLOG_UNLOCK();
		grabbed_lock = 0;
	}
	m_copydata(m, 0, length, (caddr_t)copyto);
 no_log:
	if (grabbed_lock) {
		SCTP_IP_PKTLOG_UNLOCK();
	}
	atomic_subtract_int(&SCTP_BASE_VAR(packet_log_writers), 1);
}

int
sctp_copy_out_packet_log(uint8_t *target, int length)
{
	/* We wind through the packet log starting at
	 * start copying up to length bytes out.
	 * We return the number of bytes copied.
	 */
	int this_copy;
	int *lenat;
	int did_delay = 0;

	if (length < (int)(2 * sizeof(int))) {
		/* not enough room */
		return (0);
	}
	if (SCTP_PKTLOG_WRITERS_NEED_LOCK) {
		atomic_add_int(&SCTP_BASE_VAR(packet_log_writers), SCTP_PKTLOG_WRITERS_NEED_LOCK);
	again:
		if ((did_delay == 0) && (SCTP_BASE_VAR(packet_log_writers) != SCTP_PKTLOG_WRITERS_NEED_LOCK)) {
			/* we delay here for just a moment hoping the writer(s) that were
			 * present when we entered will have left and we only have
			 * locking ones that will contend with us for the lock. This
			 * does not assure 100% access, but its good enough for
			 * a logging facility like this.
			 */
			did_delay = 1;
			DELAY(10);
			goto again;
		}
	}
	SCTP_IP_PKTLOG_LOCK();
	lenat = (int *)target;
	*lenat = SCTP_BASE_VAR(packet_log_end);
	lenat++;
	this_copy = min((length - sizeof(int)), SCTP_PACKET_LOG_SIZE);
	memcpy((void *)lenat, (void *)SCTP_BASE_VAR(packet_log_buffer), this_copy);
	if (SCTP_PKTLOG_WRITERS_NEED_LOCK) {
		atomic_subtract_int(&SCTP_BASE_VAR(packet_log_writers),
		                    SCTP_PKTLOG_WRITERS_NEED_LOCK);
	}
	SCTP_IP_PKTLOG_UNLOCK();
	return (this_copy + sizeof(int));
}

#endif
