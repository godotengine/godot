/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2017 Andy Green <andy@warmcat.com>
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

#include "private-libwebsockets.h"

#include <pwd.h>
#include <grp.h>

#ifdef LWS_WITH_PLUGINS
#include <dlfcn.h>
#endif
#include <dirent.h>

unsigned long long time_in_microseconds(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return ((unsigned long long)tv.tv_sec * 1000000LL) + tv.tv_usec;
}

LWS_VISIBLE int
lws_get_random(struct lws_context *context, void *buf, int len)
{
	return read(context->fd_random, (char *)buf, len);
}

LWS_VISIBLE int
lws_send_pipe_choked(struct lws *wsi)
{
	struct lws_pollfd fds;
	struct lws *wsi_eff = wsi;

#if defined(LWS_WITH_HTTP2)
	wsi_eff = lws_get_network_wsi(wsi);
#endif
	/* treat the fact we got a truncated send pending as if we're choked */
	if (wsi_eff->trunc_len)
		return 1;

	fds.fd = wsi_eff->desc.sockfd;
	fds.events = POLLOUT;
	fds.revents = 0;

	if (poll(&fds, 1, 0) != 1)
		return 1;

	if ((fds.revents & POLLOUT) == 0)
		return 1;

	/* okay to send another packet without blocking */

	return 0;
}

LWS_VISIBLE int
lws_poll_listen_fd(struct lws_pollfd *fd)
{
	return poll(fd, 1, 0);
}

LWS_VISIBLE void
lws_cancel_service_pt(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	char buf = 0;

	if (write(pt->dummy_pipe_fds[1], &buf, sizeof(buf)) != 1)
		lwsl_err("Cannot write to dummy pipe");
}

LWS_VISIBLE void
lws_cancel_service(struct lws_context *context)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	char buf = 0, m = context->count_threads;

	while (m--) {
		if (write(pt->dummy_pipe_fds[1], &buf, sizeof(buf)) != 1)
			lwsl_err("Cannot write to dummy pipe");
		pt++;
	}
}

LWS_VISIBLE void lwsl_emit_syslog(int level, const char *line)
{
	int syslog_level = LOG_DEBUG;

	switch (level) {
	case LLL_ERR:
		syslog_level = LOG_ERR;
		break;
	case LLL_WARN:
		syslog_level = LOG_WARNING;
		break;
	case LLL_NOTICE:
		syslog_level = LOG_NOTICE;
		break;
	case LLL_INFO:
		syslog_level = LOG_INFO;
		break;
	}
	syslog(syslog_level, "%s", line);
}

LWS_VISIBLE LWS_EXTERN int
_lws_plat_service_tsi(struct lws_context *context, int timeout_ms, int tsi)
{
	struct lws_context_per_thread *pt;
	int n = -1, m, c;
	char buf;

	/* stay dead once we are dead */

	if (!context || !context->vhost_list)
		return 1;

	pt = &context->pt[tsi];

	lws_stats_atomic_bump(context, pt, LWSSTATS_C_SERVICE_ENTRY, 1);

	if (timeout_ms < 0)
		goto faked_service;

	lws_libev_run(context, tsi);
	lws_libuv_run(context, tsi);
	lws_libevent_run(context, tsi);

	if (!context->service_tid_detected) {
		struct lws _lws;

		memset(&_lws, 0, sizeof(_lws));
		_lws.context = context;

		context->service_tid_detected =
			context->vhost_list->protocols[0].callback(
			&_lws, LWS_CALLBACK_GET_THREAD_ID, NULL, NULL, 0);
		context->service_tid = context->service_tid_detected;
		context->service_tid_detected = 1;
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

	n = poll(pt->fds, pt->fds_count, timeout_ms);

#ifdef LWS_OPENSSL_SUPPORT
	if (!n && !pt->rx_draining_ext_list &&
	    !lws_ssl_anybody_has_buffered_read_tsi(context, tsi)) {
#else
	if (!pt->rx_draining_ext_list && !n) /* poll timeout */ {
#endif
		lws_service_fd_tsi(context, NULL, tsi);
		return 0;
	}

faked_service:
	m = lws_service_flag_pending(context, tsi);
	if (m)
		c = -1; /* unknown limit */
	else
		if (n < 0) {
			if (LWS_ERRNO != LWS_EINTR)
				return -1;
			return 0;
		} else
			c = n;

	/* any socket with events to service? */
	for (n = 0; n < pt->fds_count && c; n++) {
		if (!pt->fds[n].revents)
			continue;

		c--;

		if (pt->fds[n].fd == pt->dummy_pipe_fds[0]) {
			if (read(pt->fds[n].fd, &buf, 1) != 1)
				lwsl_err("Cannot read from dummy pipe.");
			continue;
		}

		m = lws_service_fd_tsi(context, &pt->fds[n], tsi);
		if (m < 0)
			return -1;
		/* if something closed, retry this slot */
		if (m)
			n--;
	}

	return 0;
}

LWS_VISIBLE int
lws_plat_check_connection_error(struct lws *wsi)
{
	return 0;
}

LWS_VISIBLE int
lws_plat_service(struct lws_context *context, int timeout_ms)
{
	return _lws_plat_service_tsi(context, timeout_ms, 0);
}

LWS_VISIBLE int
lws_plat_set_socket_options(struct lws_vhost *vhost, int fd)
{
	int optval = 1;
	socklen_t optlen = sizeof(optval);

#if defined(__APPLE__) || \
    defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || \
    defined(__NetBSD__) || \
    defined(__OpenBSD__) || \
    defined(__HAIKU__)
	struct protoent *tcp_proto;
#endif

	if (vhost->ka_time) {
		/* enable keepalive on this socket */
		optval = 1;
		if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE,
			       (const void *)&optval, optlen) < 0)
			return 1;

#if defined(__APPLE__) || \
    defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || \
    defined(__NetBSD__) || \
    defined(__CYGWIN__) || defined(__OpenBSD__) || defined (__sun) || \
    defined(__HAIKU__)

		/*
		 * didn't find a way to set these per-socket, need to
		 * tune kernel systemwide values
		 */
#else
		/* set the keepalive conditions we want on it too */
		optval = vhost->ka_time;
		if (setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,
			       (const void *)&optval, optlen) < 0)
			return 1;

		optval = vhost->ka_interval;
		if (setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL,
			       (const void *)&optval, optlen) < 0)
			return 1;

		optval = vhost->ka_probes;
		if (setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,
			       (const void *)&optval, optlen) < 0)
			return 1;
#endif
	}

#if defined(SO_BINDTODEVICE)
	if (vhost->bind_iface && vhost->iface) {
		lwsl_info("binding listen skt to %s using SO_BINDTODEVICE\n", vhost->iface);
		if (setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, vhost->iface,
				strlen(vhost->iface)) < 0) {
			lwsl_warn("Failed to bind to device %s\n", vhost->iface);
			return 1;
		}
	}
#endif

	/* Disable Nagle */
	optval = 1;
#if defined (__sun)
	if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const void *)&optval, optlen) < 0)
		return 1;
#elif !defined(__APPLE__) && \
      !defined(__FreeBSD__) && !defined(__FreeBSD_kernel__) &&        \
      !defined(__NetBSD__) && \
      !defined(__OpenBSD__) && \
      !defined(__HAIKU__)
	if (setsockopt(fd, SOL_TCP, TCP_NODELAY, (const void *)&optval, optlen) < 0)
		return 1;
#else
	tcp_proto = getprotobyname("TCP");
	if (setsockopt(fd, tcp_proto->p_proto, TCP_NODELAY, &optval, optlen) < 0)
		return 1;
#endif

	/* We are nonblocking... */
	if (fcntl(fd, F_SETFL, O_NONBLOCK) < 0)
		return 1;

	return 0;
}

#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
static void
_lws_plat_apply_caps(int mode, cap_value_t *cv, int count)
{
	cap_t caps;

	if (!count)
		return;

	caps = cap_get_proc();

	cap_set_flag(caps, mode, count, cv, CAP_SET);
	cap_set_proc(caps);
	prctl(PR_SET_KEEPCAPS, 1, 0, 0, 0);
	cap_free(caps);
}
#endif

LWS_VISIBLE void
lws_plat_drop_app_privileges(struct lws_context_creation_info *info)
{
#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
	int n;
#endif

	if (info->gid && info->gid != -1)
		if (setgid(info->gid))
			lwsl_warn("setgid: %s\n", strerror(LWS_ERRNO));

	if (info->uid && info->uid != -1) {
		struct passwd *p = getpwuid(info->uid);

		if (p) {

#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
			_lws_plat_apply_caps(CAP_PERMITTED, info->caps, info->count_caps);
#endif

			initgroups(p->pw_name, info->gid);
			if (setuid(info->uid))
				lwsl_warn("setuid: %s\n", strerror(LWS_ERRNO));
			else
				lwsl_notice("Set privs to user '%s'\n", p->pw_name);

#if defined(LWS_HAVE_SYS_CAPABILITY_H) && defined(LWS_HAVE_LIBCAP)
			_lws_plat_apply_caps(CAP_EFFECTIVE, info->caps, info->count_caps);

			if (info->count_caps)
				for (n = 0; n < info->count_caps; n++)
					lwsl_notice("   RETAINING CAPABILITY %d\n", (int)info->caps[n]);
#endif

		} else
			lwsl_warn("getpwuid: unable to find uid %d", info->uid);
	}
}

#ifdef LWS_WITH_PLUGINS

#if defined(LWS_WITH_LIBUV) && UV_VERSION_MAJOR > 0

/* libuv.c implements these in a cross-platform way */

#else

static int filter(const struct dirent *ent)
{
	if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
		return 0;

	return 1;
}

LWS_VISIBLE int
lws_plat_plugins_init(struct lws_context * context, const char * const *d)
{
	struct lws_plugin_capability lcaps;
	struct lws_plugin *plugin;
	lws_plugin_init_func initfunc;
	struct dirent **namelist;
	int n, i, m, ret = 0;
	char path[256];
	void *l;

	lwsl_notice("  Plugins:\n");

	while (d && *d) {
		n = scandir(*d, &namelist, filter, alphasort);
		if (n < 0) {
			lwsl_err("Scandir on %s failed\n", *d);
			return 1;
		}

		for (i = 0; i < n; i++) {
			if (strlen(namelist[i]->d_name) < 7)
				goto inval;

			lwsl_notice("   %s\n", namelist[i]->d_name);

			lws_snprintf(path, sizeof(path) - 1, "%s/%s", *d,
				 namelist[i]->d_name);
			l = dlopen(path, RTLD_NOW);
			if (!l) {
				lwsl_err("Error loading DSO: %s\n", dlerror());
				while (i++ < n)
					free(namelist[i]);
				goto bail;
			}
			/* we could open it, can we get his init function? */
			m = lws_snprintf(path, sizeof(path) - 1, "init_%s",
				     namelist[i]->d_name + 3 /* snip lib... */);
			path[m - 3] = '\0'; /* snip the .so */
			initfunc = dlsym(l, path);
			if (!initfunc) {
				lwsl_err("Failed to get init on %s: %s",
						namelist[i]->d_name, dlerror());
				dlclose(l);
			}
			lcaps.api_magic = LWS_PLUGIN_API_MAGIC;
			m = initfunc(context, &lcaps);
			if (m) {
				lwsl_err("Initializing %s failed %d\n",
					namelist[i]->d_name, m);
				dlclose(l);
				goto skip;
			}

			plugin = lws_malloc(sizeof(*plugin), "plugin");
			if (!plugin) {
				lwsl_err("OOM\n");
				goto bail;
			}
			plugin->list = context->plugin_list;
			context->plugin_list = plugin;
			strncpy(plugin->name, namelist[i]->d_name, sizeof(plugin->name) - 1);
			plugin->name[sizeof(plugin->name) - 1] = '\0';
			plugin->l = l;
			plugin->caps = lcaps;
			context->plugin_protocol_count += lcaps.count_protocols;
			context->plugin_extension_count += lcaps.count_extensions;

			free(namelist[i]);
			continue;

	skip:
			dlclose(l);
	inval:
			free(namelist[i]);
		}
		free(namelist);
		d++;
	}

bail:
	free(namelist);

	return ret;
}

LWS_VISIBLE int
lws_plat_plugins_destroy(struct lws_context * context)
{
	struct lws_plugin *plugin = context->plugin_list, *p;
	lws_plugin_destroy_func func;
	char path[256];
	int m;

	if (!plugin)
		return 0;

	lwsl_notice("%s\n", __func__);

	while (plugin) {
		p = plugin;
		m = lws_snprintf(path, sizeof(path) - 1, "destroy_%s", plugin->name + 3);
		path[m - 3] = '\0';
		func = dlsym(plugin->l, path);
		if (!func) {
			lwsl_err("Failed to get destroy on %s: %s",
					plugin->name, dlerror());
			goto next;
		}
		m = func(context);
		if (m)
			lwsl_err("Initializing %s failed %d\n",
				plugin->name, m);
next:
		dlclose(p->l);
		plugin = p->list;
		p->list = NULL;
		free(p);
	}

	context->plugin_list = NULL;

	return 0;
}

#endif
#endif


#if 0
static void
sigabrt_handler(int x)
{
	printf("%s\n", __func__);
}
#endif

LWS_VISIBLE int
lws_plat_context_early_init(void)
{
#if !defined(LWS_AVOID_SIGPIPE_IGN)
	signal(SIGPIPE, SIG_IGN);
#endif

	return 0;
}

LWS_VISIBLE void
lws_plat_context_early_destroy(struct lws_context *context)
{
}

LWS_VISIBLE void
lws_plat_context_late_destroy(struct lws_context *context)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	int m = context->count_threads;

#ifdef LWS_WITH_PLUGINS
	if (context->plugin_list)
		lws_plat_plugins_destroy(context);
#endif

	if (context->lws_lookup)
		lws_free(context->lws_lookup);

	while (m--) {
		if (pt->dummy_pipe_fds[0])
			close(pt->dummy_pipe_fds[0]);
		if (pt->dummy_pipe_fds[1])
			close(pt->dummy_pipe_fds[1]);
		pt++;
	}
	if (!context->fd_random)
		lwsl_err("ZERO RANDOM FD\n");
	if (context->fd_random != LWS_INVALID_FILE)
		close(context->fd_random);
}

/* cast a struct sockaddr_in6 * into addr for ipv6 */

LWS_VISIBLE int
lws_interface_to_sa(int ipv6, const char *ifname, struct sockaddr_in *addr,
		    size_t addrlen)
{
	int rc = -1;

	struct ifaddrs *ifr;
	struct ifaddrs *ifc;
#ifdef LWS_WITH_IPV6
	struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)addr;
#endif

	getifaddrs(&ifr);
	for (ifc = ifr; ifc != NULL && rc; ifc = ifc->ifa_next) {
		if (!ifc->ifa_addr)
			continue;

		lwsl_info(" interface %s vs %s\n", ifc->ifa_name, ifname);

		if (strcmp(ifc->ifa_name, ifname))
			continue;

		switch (ifc->ifa_addr->sa_family) {
		case AF_INET:
#ifdef LWS_WITH_IPV6
			if (ipv6) {
				/* map IPv4 to IPv6 */
				bzero((char *)&addr6->sin6_addr,
						sizeof(struct in6_addr));
				addr6->sin6_addr.s6_addr[10] = 0xff;
				addr6->sin6_addr.s6_addr[11] = 0xff;
				memcpy(&addr6->sin6_addr.s6_addr[12],
					&((struct sockaddr_in *)ifc->ifa_addr)->sin_addr,
							sizeof(struct in_addr));
			} else
#endif
				memcpy(addr,
					(struct sockaddr_in *)ifc->ifa_addr,
						    sizeof(struct sockaddr_in));
			break;
#ifdef LWS_WITH_IPV6
		case AF_INET6:
			memcpy(&addr6->sin6_addr,
			  &((struct sockaddr_in6 *)ifc->ifa_addr)->sin6_addr,
						       sizeof(struct in6_addr));
			break;
#endif
		default:
			continue;
		}
		rc = 0;
	}

	freeifaddrs(ifr);

	if (rc == -1) {
		/* check if bind to IP address */
#ifdef LWS_WITH_IPV6
		if (inet_pton(AF_INET6, ifname, &addr6->sin6_addr) == 1)
			rc = 0;
		else
#endif
		if (inet_pton(AF_INET, ifname, &addr->sin_addr) == 1)
			rc = 0;
	}

	return rc;
}

LWS_VISIBLE void
lws_plat_insert_socket_into_fds(struct lws_context *context, struct lws *wsi)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];

	lws_libev_io(wsi, LWS_EV_START | LWS_EV_READ);
	lws_libuv_io(wsi, LWS_EV_START | LWS_EV_READ);
	lws_libevent_io(wsi, LWS_EV_START | LWS_EV_READ);

	pt->fds[pt->fds_count++].revents = 0;
}

LWS_VISIBLE void
lws_plat_delete_socket_from_fds(struct lws_context *context,
						struct lws *wsi, int m)
{
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];

	lws_libev_io(wsi, LWS_EV_STOP | LWS_EV_READ | LWS_EV_WRITE);
	lws_libuv_io(wsi, LWS_EV_STOP | LWS_EV_READ | LWS_EV_WRITE);
	lws_libevent_io(wsi, LWS_EV_STOP | LWS_EV_READ | LWS_EV_WRITE);

	pt->fds_count--;
}

LWS_VISIBLE void
lws_plat_service_periodic(struct lws_context *context)
{
	/* if our parent went down, don't linger around */
	if (context->started_with_parent &&
	    kill(context->started_with_parent, 0) < 0)
		kill(getpid(), SIGTERM);
}

LWS_VISIBLE int
lws_plat_change_pollfd(struct lws_context *context,
		      struct lws *wsi, struct lws_pollfd *pfd)
{
	return 0;
}

LWS_VISIBLE const char *
lws_plat_inet_ntop(int af, const void *src, char *dst, int cnt)
{
	return inet_ntop(af, src, dst, cnt);
}

LWS_VISIBLE int
lws_plat_inet_pton(int af, const char *src, void *dst)
{
	return inet_pton(af, src, dst);
}

LWS_VISIBLE lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags)
{
	struct stat stat_buf;
	int ret = open(filename, (*flags) & LWS_FOP_FLAGS_MASK, 0664);
	lws_fop_fd_t fop_fd;

	if (ret < 0)
		return NULL;

	if (fstat(ret, &stat_buf) < 0)
		goto bail;

	fop_fd = malloc(sizeof(*fop_fd));
	if (!fop_fd)
		goto bail;

	fop_fd->fops = fops;
	fop_fd->flags = *flags;
	fop_fd->fd = ret;
	fop_fd->filesystem_priv = NULL; /* we don't use it */
	fop_fd->len = stat_buf.st_size;
	fop_fd->pos = 0;

	return fop_fd;

bail:
	close(ret);
	return NULL;
}

LWS_VISIBLE int
_lws_plat_file_close(lws_fop_fd_t *fop_fd)
{
	int fd = (*fop_fd)->fd;

	free(*fop_fd);
	*fop_fd = NULL;

	return close(fd);
}

LWS_VISIBLE lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	lws_fileofs_t r;

	if (offset > 0 && offset > fop_fd->len - fop_fd->pos)
		offset = fop_fd->len - fop_fd->pos;

	if ((lws_fileofs_t)fop_fd->pos + offset < 0)
		offset = -fop_fd->pos;

	r = lseek(fop_fd->fd, offset, SEEK_CUR);

	if (r >= 0)
		fop_fd->pos = r;
	else
		lwsl_err("error seeking from cur %ld, offset %ld\n",
                        (long)fop_fd->pos, (long)offset);

	return r;
}

LWS_VISIBLE int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	long n;

	n = read((int)fop_fd->fd, buf, len);
	if (n == -1) {
		*amount = 0;
		return -1;
	}
	fop_fd->pos += n;
	lwsl_debug("%s: read %ld of req %ld, pos %ld, len %ld\n", __func__, n,
                  (long)len, (long)fop_fd->pos, (long)fop_fd->len);
	*amount = n;

	return 0;
}

LWS_VISIBLE int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		     uint8_t *buf, lws_filepos_t len)
{
	long n;

	n = write((int)fop_fd->fd, buf, len);
	if (n == -1) {
		*amount = 0;
		return -1;
	}

	fop_fd->pos += n;
	*amount = n;

	return 0;
}


LWS_VISIBLE int
lws_plat_init(struct lws_context *context,
	      struct lws_context_creation_info *info)
{
	struct lws_context_per_thread *pt = &context->pt[0];
	int n = context->count_threads, fd;

	/* master context has the global fd lookup array */
	context->lws_lookup = lws_zalloc(sizeof(struct lws *) *
					 context->max_fds, "lws_lookup");
	if (context->lws_lookup == NULL) {
		lwsl_err("OOM on lws_lookup array for %d connections\n",
			 context->max_fds);
		return 1;
	}

	lwsl_info(" mem: platform fd map: %5lu bytes\n",
		    (unsigned long)(sizeof(struct lws *) * context->max_fds));
	fd = open(SYSTEM_RANDOM_FILEPATH, O_RDONLY);

	context->fd_random = fd;
	if (context->fd_random < 0) {
		lwsl_err("Unable to open random device %s %d\n",
			 SYSTEM_RANDOM_FILEPATH, context->fd_random);
		return 1;
	}

	if (!lws_libev_init_fd_table(context) &&
	    !lws_libuv_init_fd_table(context) &&
	    !lws_libevent_init_fd_table(context)) {
		/* otherwise libev/uv/event handled it instead */

		while (n--) {
			if (pipe(pt->dummy_pipe_fds)) {
				lwsl_err("Unable to create pipe\n");
				return 1;
			}

			/* use the read end of pipe as first item */
			pt->fds[0].fd = pt->dummy_pipe_fds[0];
			pt->fds[0].events = LWS_POLLIN;
			pt->fds[0].revents = 0;
			pt->fds_count = 1;
			pt++;
		}
	}

#ifdef LWS_WITH_PLUGINS
	if (info->plugin_dirs)
		lws_plat_plugins_init(context, info->plugin_dirs);
#endif

	return 0;
}
