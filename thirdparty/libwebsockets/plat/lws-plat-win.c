#ifndef _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif
#include "private-libwebsockets.h"

unsigned long long
time_in_microseconds()
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
	 * As per Windows documentation for FILETIME, copy the resulting FILETIME structure to a
	 * ULARGE_INTEGER structure using memcpy (using memcpy instead of direct assignment can
	 * prevent alignment faults on 64-bit Windows).
	 */
	memcpy(&datetime, &filetime, sizeof(datetime));

	/* Windows file times are in 100s of nanoseconds. */
	return (datetime.QuadPart - DELTA_EPOCH_IN_MICROSECS) / 10;
}

#ifdef _WIN32_WCE
time_t time(time_t *t)
{
	time_t ret = time_in_microseconds() / 1000000;

	if(t != NULL)
		*t = ret;

	return ret;
}
#endif

/* file descriptor hash management */

struct lws *
wsi_from_fd(const struct lws_context *context, lws_sockfd_type fd)
{
	int h = LWS_FD_HASH(fd);
	int n = 0;

	for (n = 0; n < context->fd_hashtable[h].length; n++)
		if (context->fd_hashtable[h].wsi[n]->desc.sockfd == fd)
			return context->fd_hashtable[h].wsi[n];

	return NULL;
}

int
insert_wsi(struct lws_context *context, struct lws *wsi)
{
	int h = LWS_FD_HASH(wsi->desc.sockfd);

	if (context->fd_hashtable[h].length == (getdtablesize() - 1)) {
		lwsl_err("hash table overflow\n");
		return 1;
	}

	context->fd_hashtable[h].wsi[context->fd_hashtable[h].length++] = wsi;

	return 0;
}

int
delete_from_fd(struct lws_context *context, lws_sockfd_type fd)
{
	int h = LWS_FD_HASH(fd);
	int n = 0;

	for (n = 0; n < context->fd_hashtable[h].length; n++)
		if (context->fd_hashtable[h].wsi[n]->desc.sockfd == fd) {
			while (n < context->fd_hashtable[h].length) {
				context->fd_hashtable[h].wsi[n] =
						context->fd_hashtable[h].wsi[n + 1];
				n++;
			}
			context->fd_hashtable[h].length--;

			return 0;
		}

	lwsl_err("Failed to find fd %d requested for "
		 "delete in hashtable\n", fd);
	return 1;
}

LWS_VISIBLE int lws_get_random(struct lws_context *context,
								 void *buf, int len)
{
	int n;
	char *p = (char *)buf;

	for (n = 0; n < len; n++)
		p[n] = (unsigned char)rand();

	return n;
}

LWS_VISIBLE int lws_send_pipe_choked(struct lws *wsi)
{
	/* treat the fact we got a truncated send pending as if we're choked */
	if (wsi->trunc_len)
		return 1;

	return (int)wsi->sock_send_blocking;
}

LWS_VISIBLE int lws_poll_listen_fd(struct lws_pollfd *fd)
{
	fd_set readfds;
	struct timeval tv = { 0, 0 };

	assert((fd->events & LWS_POLLIN) == LWS_POLLIN);

	FD_ZERO(&readfds);
	FD_SET(fd->fd, &readfds);

	return select(fd->fd + 1, &readfds, NULL, NULL, &tv);
}

LWS_VISIBLE void
lws_cancel_service(struct lws_context *context)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	int n = context->count_threads;

	while (n--) {
		WSASetEvent(pt->events[0]);
		pt++;
	}
}

LWS_VISIBLE void
lws_cancel_service_pt(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	WSASetEvent(pt->events[0]);
}

LWS_VISIBLE void lwsl_emit_syslog(int level, const char *line)
{
	lwsl_emit_stderr(level, line);
}

LWS_VISIBLE LWS_EXTERN int
_lws_plat_service_tsi(struct lws_context *context, int timeout_ms, int tsi)
{
	struct lws_context_per_thread *pt;
	WSANETWORKEVENTS networkevents;
	struct lws_pollfd *pfd;
	struct lws *wsi;
	unsigned int i;
	DWORD ev;
	int n, m;

	/* stay dead once we are dead */
	if (context == NULL || !context->vhost_list)
		return 1;

	pt = &context->pt[tsi];

	if (!context->service_tid_detected) {
		struct lws _lws;

		memset(&_lws, 0, sizeof(_lws));
		_lws.context = context;

		context->service_tid_detected = context->vhost_list->
			protocols[0].callback(&_lws, LWS_CALLBACK_GET_THREAD_ID,
						  NULL, NULL, 0);
		context->service_tid = context->service_tid_detected;
		context->service_tid_detected = 1;
	}

	if (timeout_ms < 0)
	{
			if (lws_service_flag_pending(context, tsi)) {
			/* any socket with events to service? */
			for (n = 0; n < (int)pt->fds_count; n++) {
				if (!pt->fds[n].revents)
					continue;

				m = lws_service_fd_tsi(context, &pt->fds[n], tsi);
				if (m < 0)
					return -1;
				/* if something closed, retry this slot */
				if (m)
					n--;
			}
		}
		return 0;
	}

	for (i = 0; i < pt->fds_count; ++i) {
		pfd = &pt->fds[i];

		if (!(pfd->events & LWS_POLLOUT))
			continue;

		wsi = wsi_from_fd(context, pfd->fd);
		if (wsi->listener)
			continue;
		if (!wsi || wsi->sock_send_blocking)
			continue;
		pfd->revents = LWS_POLLOUT;
		n = lws_service_fd(context, pfd);
		if (n < 0)
			return -1;
		/* if something closed, retry this slot */
		if (n)
			i--;

		if (wsi->trunc_len)
			WSASetEvent(pt->events[0]);
	}

	/*
	 * is there anybody with pending stuff that needs service forcing?
	 */
	if (!lws_service_adjust_timeout(context, 1, tsi)) {
		/* -1 timeout means just do forced service */
		_lws_plat_service_tsi(context, -1, pt->tid);
		/* still somebody left who wants forced service? */
		if (!lws_service_adjust_timeout(context, 1, pt->tid))
			/* yes... come back again quickly */
			timeout_ms = 0;
	}

	ev = WSAWaitForMultipleEvents( 1,  pt->events , FALSE, timeout_ms, FALSE);
	if (ev == WSA_WAIT_EVENT_0) {
		unsigned int eIdx;

		WSAResetEvent(pt->events[0]);

		for (eIdx = 0; eIdx < pt->fds_count; ++eIdx) {
			if (WSAEnumNetworkEvents(pt->fds[eIdx].fd, 0, &networkevents) == SOCKET_ERROR) {
				lwsl_err("WSAEnumNetworkEvents() failed with error %d\n", LWS_ERRNO);
				return -1;
			}

			pfd = &pt->fds[eIdx];
			pfd->revents = (short)networkevents.lNetworkEvents;

			if ((networkevents.lNetworkEvents & FD_CONNECT) &&
				 networkevents.iErrorCode[FD_CONNECT_BIT] &&
				 networkevents.iErrorCode[FD_CONNECT_BIT] != LWS_EALREADY &&
				 networkevents.iErrorCode[FD_CONNECT_BIT] != LWS_EINPROGRESS &&
				 networkevents.iErrorCode[FD_CONNECT_BIT] != LWS_EWOULDBLOCK &&
				 networkevents.iErrorCode[FD_CONNECT_BIT] != WSAEINVAL) {
				lwsl_debug("Unable to connect errno=%d\n",
					   networkevents.iErrorCode[FD_CONNECT_BIT]);
				pfd->revents |= LWS_POLLHUP;
			}

			if (pfd->revents & LWS_POLLOUT) {
				wsi = wsi_from_fd(context, pfd->fd);
				if (wsi)
					wsi->sock_send_blocking = 0;
			}
			 /* if something closed, retry this slot */
			if (pfd->revents & LWS_POLLHUP)
					--eIdx;

			if( pfd->revents != 0 ) {
				lws_service_fd_tsi(context, pfd, tsi);

			}
		}
	}

	context->service_tid = 0;

	if (ev == WSA_WAIT_TIMEOUT) {
		lws_service_fd(context, NULL);
	}
	return 0;;
}

LWS_VISIBLE int
lws_plat_service(struct lws_context *context, int timeout_ms)
{
	return _lws_plat_service_tsi(context, timeout_ms, 0);
}

LWS_VISIBLE int
lws_plat_set_socket_options(struct lws_vhost *vhost, lws_sockfd_type fd)
{
	int optval = 1;
	int optlen = sizeof(optval);
	u_long optl = 1;
	DWORD dwBytesRet;
	struct tcp_keepalive alive;
	int protonbr;
#ifndef _WIN32_WCE
	struct protoent *tcp_proto;
#endif

	if (vhost->ka_time) {
		/* enable keepalive on this socket */
		optval = 1;
		if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE,
						 (const char *)&optval, optlen) < 0)
			return 1;

		alive.onoff = TRUE;
		alive.keepalivetime = vhost->ka_time;
		alive.keepaliveinterval = vhost->ka_interval;

		if (WSAIoctl(fd, SIO_KEEPALIVE_VALS, &alive, sizeof(alive),
						  NULL, 0, &dwBytesRet, NULL, NULL))
			return 1;
	}

	/* Disable Nagle */
	optval = 1;
#ifndef _WIN32_WCE
	tcp_proto = getprotobyname("TCP");
	if (!tcp_proto) {
		lwsl_err("getprotobyname() failed with error %d\n", LWS_ERRNO);
		return 1;
	}
	protonbr = tcp_proto->p_proto;
#else
	protonbr = 6;
#endif

	setsockopt(fd, protonbr, TCP_NODELAY, (const char *)&optval, optlen);

	/* We are nonblocking... */
	ioctlsocket(fd, FIONBIO, &optl);

	return 0;
}

LWS_VISIBLE void
lws_plat_drop_app_privileges(struct lws_context_creation_info *info)
{
}

LWS_VISIBLE int
lws_plat_context_early_init(void)
{
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;

	/* Use the MAKEWORD(lowbyte, highbyte) macro from Windef.h */
	wVersionRequested = MAKEWORD(2, 2);

	err = WSAStartup(wVersionRequested, &wsaData);
	if (!err)
		return 0;
	/*
	 * Tell the user that we could not find a usable
	 * Winsock DLL
	 */
	lwsl_err("WSAStartup failed with error: %d\n", err);

	return 1;
}

LWS_VISIBLE void
lws_plat_context_early_destroy(struct lws_context *context)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	int n = context->count_threads;

	while (n--) {
		if (pt->events) {
			WSACloseEvent(pt->events[0]);
			lws_free(pt->events);
		}
		pt++;
	}
}

LWS_VISIBLE void
lws_plat_context_late_destroy(struct lws_context *context)
{
	int n;

	for (n = 0; n < FD_HASHTABLE_MODULUS; n++) {
		if (context->fd_hashtable[n].wsi)
			lws_free(context->fd_hashtable[n].wsi);
	}

	WSACleanup();
}

LWS_VISIBLE LWS_EXTERN int
lws_interface_to_sa(int ipv6,
		const char *ifname, struct sockaddr_in *addr, size_t addrlen)
{
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)addr;

	if (ipv6) {
		if (lws_plat_inet_pton(AF_INET6, ifname, &addr6->sin6_addr) == 1) {
			return 0;
		}
	}
#endif

	long long address = inet_addr(ifname);

	if (address == INADDR_NONE) {
		struct hostent *entry = gethostbyname(ifname);
		if (entry)
			address = ((struct in_addr *)entry->h_addr_list[0])->s_addr;
	}

	if (address == INADDR_NONE)
		return -1;

	addr->sin_addr.s_addr = (lws_intptr_t)address;

	return 0;
}

LWS_VISIBLE void
lws_plat_insert_socket_into_fds(struct lws_context *context, struct lws *wsi)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];

	pt->fds[pt->fds_count++].revents = 0;
	pt->events[pt->fds_count] = pt->events[0];
	WSAEventSelect(wsi->desc.sockfd, pt->events[0],
			   LWS_POLLIN | LWS_POLLHUP | FD_CONNECT);
}

LWS_VISIBLE void
lws_plat_delete_socket_from_fds(struct lws_context *context,
						struct lws *wsi, int m)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];

	pt->events[m + 1] = pt->events[pt->fds_count--];
}

LWS_VISIBLE void
lws_plat_service_periodic(struct lws_context *context)
{
}

LWS_VISIBLE int
lws_plat_check_connection_error(struct lws *wsi)
{
	int optVal;
	int optLen = sizeof(int);

	if (getsockopt(wsi->desc.sockfd, SOL_SOCKET, SO_ERROR,
			   (char*)&optVal, &optLen) != SOCKET_ERROR && optVal &&
		optVal != LWS_EALREADY && optVal != LWS_EINPROGRESS &&
		optVal != LWS_EWOULDBLOCK && optVal != WSAEINVAL) {
		   lwsl_debug("Connect failed SO_ERROR=%d\n", optVal);
		   return 1;
	}

	return 0;
}

LWS_VISIBLE int
lws_plat_change_pollfd(struct lws_context *context,
			  struct lws *wsi, struct lws_pollfd *pfd)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	long networkevents = LWS_POLLHUP | FD_CONNECT;

	if ((pfd->events & LWS_POLLIN))
		networkevents |= LWS_POLLIN;

	if ((pfd->events & LWS_POLLOUT))
		networkevents |= LWS_POLLOUT;

	if (WSAEventSelect(wsi->desc.sockfd,
			pt->events[0],
						   networkevents) != SOCKET_ERROR)
		return 0;

	lwsl_err("WSAEventSelect() failed with error %d\n", LWS_ERRNO);

	return 1;
}

LWS_VISIBLE const char *
lws_plat_inet_ntop(int af, const void *src, char *dst, int cnt)
{
	WCHAR *buffer;
	DWORD bufferlen = cnt;
	BOOL ok = FALSE;

	buffer = lws_malloc(bufferlen * 2, "inet_ntop");
	if (!buffer) {
		lwsl_err("Out of memory\n");
		return NULL;
	}

	if (af == AF_INET) {
		struct sockaddr_in srcaddr;
		bzero(&srcaddr, sizeof(srcaddr));
		srcaddr.sin_family = AF_INET;
		memcpy(&(srcaddr.sin_addr), src, sizeof(srcaddr.sin_addr));

		if (!WSAAddressToStringW((struct sockaddr*)&srcaddr, sizeof(srcaddr), 0, buffer, &bufferlen))
			ok = TRUE;
#ifdef LWS_WITH_IPV6
	} else if (af == AF_INET6) {
		struct sockaddr_in6 srcaddr;
		bzero(&srcaddr, sizeof(srcaddr));
		srcaddr.sin6_family = AF_INET6;
		memcpy(&(srcaddr.sin6_addr), src, sizeof(srcaddr.sin6_addr));

		if (!WSAAddressToStringW((struct sockaddr*)&srcaddr, sizeof(srcaddr), 0, buffer, &bufferlen))
			ok = TRUE;
#endif
	} else
		lwsl_err("Unsupported type\n");

	if (!ok) {
		int rv = WSAGetLastError();
		lwsl_err("WSAAddressToString() : %d\n", rv);
	} else {
		if (WideCharToMultiByte(CP_ACP, 0, buffer, bufferlen, dst, cnt, 0, NULL) <= 0)
			ok = FALSE;
	}

	lws_free(buffer);
	return ok ? dst : NULL;
}

LWS_VISIBLE int
lws_plat_inet_pton(int af, const char *src, void *dst)
{
	WCHAR *buffer;
	DWORD bufferlen = strlen(src) + 1;
	BOOL ok = FALSE;

	buffer = lws_malloc(bufferlen * 2, "inet_pton");
	if (!buffer) {
		lwsl_err("Out of memory\n");
		return -1;
	}

	if (MultiByteToWideChar(CP_ACP, 0, src, bufferlen, buffer, bufferlen) <= 0) {
		lwsl_err("Failed to convert multi byte to wide char\n");
		lws_free(buffer);
		return -1;
	}

	if (af == AF_INET) {
		struct sockaddr_in dstaddr;
		int dstaddrlen = sizeof(dstaddr);
		bzero(&dstaddr, sizeof(dstaddr));
		dstaddr.sin_family = AF_INET;

		if (!WSAStringToAddressW(buffer, af, 0, (struct sockaddr *) &dstaddr, &dstaddrlen)) {
			ok = TRUE;
			memcpy(dst, &dstaddr.sin_addr, sizeof(dstaddr.sin_addr));
		}
#ifdef LWS_WITH_IPV6
	} else if (af == AF_INET6) {
		struct sockaddr_in6 dstaddr;
		int dstaddrlen = sizeof(dstaddr);
		bzero(&dstaddr, sizeof(dstaddr));
		dstaddr.sin6_family = AF_INET6;

		if (!WSAStringToAddressW(buffer, af, 0, (struct sockaddr *) &dstaddr, &dstaddrlen)) {
			ok = TRUE;
			memcpy(dst, &dstaddr.sin6_addr, sizeof(dstaddr.sin6_addr));
		}
#endif
	} else
		lwsl_err("Unsupported type\n");

	if (!ok) {
		int rv = WSAGetLastError();
		lwsl_err("WSAAddressToString() : %d\n", rv);
	}

	lws_free(buffer);
	return ok ? 1 : -1;
}

LWS_VISIBLE lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags)
{
	HANDLE ret;
	WCHAR buf[MAX_PATH];
	lws_fop_fd_t fop_fd;
	LARGE_INTEGER llFileSize = {0};

	MultiByteToWideChar(CP_UTF8, 0, filename, -1, buf, ARRAY_SIZE(buf));
	if (((*flags) & 7) == _O_RDONLY) {
		ret = CreateFileW(buf, GENERIC_READ, FILE_SHARE_READ,
			  NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	} else {
		ret = CreateFileW(buf, GENERIC_WRITE, 0, NULL,
			  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	}

	if (ret == LWS_INVALID_FILE)
		goto bail;

	fop_fd = malloc(sizeof(*fop_fd));
	if (!fop_fd)
		goto bail;

	fop_fd->fops = fops;
	fop_fd->fd = ret;
	fop_fd->filesystem_priv = NULL; /* we don't use it */
	fop_fd->flags = *flags;
	fop_fd->len = GetFileSize(ret, NULL);
	if(GetFileSizeEx(ret, &llFileSize))
		fop_fd->len = llFileSize.QuadPart;

	fop_fd->pos = 0;

	return fop_fd;

bail:
	return NULL;
}

LWS_VISIBLE int
_lws_plat_file_close(lws_fop_fd_t *fop_fd)
{
	HANDLE fd = (*fop_fd)->fd;

	free(*fop_fd);
	*fop_fd = NULL;

	CloseHandle((HANDLE)fd);

	return 0;
}

LWS_VISIBLE lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	LARGE_INTEGER l;

	l.QuadPart = offset;
	return SetFilePointerEx((HANDLE)fop_fd->fd, l, NULL, FILE_CURRENT);
}

LWS_VISIBLE int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	DWORD _amount;

	if (!ReadFile((HANDLE)fop_fd->fd, buf, (DWORD)len, &_amount, NULL)) {
		*amount = 0;

		return 1;
	}

	fop_fd->pos += _amount;
	*amount = (unsigned long)_amount;

	return 0;
}

LWS_VISIBLE int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			 uint8_t* buf, lws_filepos_t len)
{
	DWORD _amount;

	if (!WriteFile((HANDLE)fop_fd->fd, buf, (DWORD)len, &_amount, NULL)) {
		*amount = 0;

		return 1;
	}

	fop_fd->pos += _amount;
	*amount = (unsigned long)_amount;

	return 0;
}

LWS_VISIBLE int
lws_plat_init(struct lws_context *context,
		  struct lws_context_creation_info *info)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	int i, n = context->count_threads;

	for (i = 0; i < FD_HASHTABLE_MODULUS; i++) {
		context->fd_hashtable[i].wsi =
			lws_zalloc(sizeof(struct lws*) * context->max_fds, "win hashtable");

		if (!context->fd_hashtable[i].wsi)
			return -1;
	}

	while (n--) {
		pt->events = lws_malloc(sizeof(WSAEVENT) *
					(context->fd_limit_per_thread + 1), "event table");
		if (pt->events == NULL) {
			lwsl_err("Unable to allocate events array for %d connections\n",
					context->fd_limit_per_thread + 1);
			return 1;
		}

		pt->fds_count = 0;
		pt->events[0] = WSACreateEvent();

		pt++;
	}

	context->fd_random = 0;

#ifdef LWS_WITH_PLUGINS
	if (info->plugin_dirs)
		lws_plat_plugins_init(context, info->plugin_dirs);
#endif

	return 0;
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

