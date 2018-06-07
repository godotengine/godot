/*
 * libwebsockets web server application
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
#include "../misc/lejp.h"

#ifndef _WIN32
/* this is needed for Travis CI */
#include <dirent.h>
#endif

#define ESC_INSTALL_DATADIR "_lws_ddir_"

static const char * const paths_global[] = {
	"global.uid",
	"global.gid",
	"global.count-threads",
	"global.init-ssl",
	"global.server-string",
	"global.plugin-dir",
	"global.ws-pingpong-secs",
	"global.timeout-secs",
	"global.reject-service-keywords[].*",
	"global.reject-service-keywords[]",
};

enum lejp_global_paths {
	LEJPGP_UID,
	LEJPGP_GID,
	LEJPGP_COUNT_THREADS,
	LWJPGP_INIT_SSL,
	LEJPGP_SERVER_STRING,
	LEJPGP_PLUGIN_DIR,
	LWJPGP_PINGPONG_SECS,
	LWJPGP_TIMEOUT_SECS,
	LWJPGP_REJECT_SERVICE_KEYWORDS_NAME,
	LWJPGP_REJECT_SERVICE_KEYWORDS
};

static const char * const paths_vhosts[] = {
	"vhosts[]",
	"vhosts[].mounts[]",
	"vhosts[].name",
	"vhosts[].port",
	"vhosts[].interface",
	"vhosts[].unix-socket",
	"vhosts[].sts",
	"vhosts[].host-ssl-key",
	"vhosts[].host-ssl-cert",
	"vhosts[].host-ssl-ca",
	"vhosts[].access-log",
	"vhosts[].mounts[].mountpoint",
	"vhosts[].mounts[].origin",
	"vhosts[].mounts[].protocol",
	"vhosts[].mounts[].default",
	"vhosts[].mounts[].auth-mask",
	"vhosts[].mounts[].cgi-timeout",
	"vhosts[].mounts[].cgi-env[].*",
	"vhosts[].mounts[].cache-max-age",
	"vhosts[].mounts[].cache-reuse",
	"vhosts[].mounts[].cache-revalidate",
	"vhosts[].mounts[].basic-auth",
	"vhosts[].mounts[].cache-intermediaries",
	"vhosts[].mounts[].extra-mimetypes.*",
	"vhosts[].mounts[].interpret.*",
	"vhosts[].ws-protocols[].*.*",
	"vhosts[].ws-protocols[].*",
	"vhosts[].ws-protocols[]",
	"vhosts[].keepalive_timeout",
	"vhosts[].enable-client-ssl",
	"vhosts[].ciphers",
	"vhosts[].ecdh-curve",
	"vhosts[].noipv6",
	"vhosts[].ipv6only",
	"vhosts[].ssl-option-set",
	"vhosts[].ssl-option-clear",
	"vhosts[].mounts[].pmo[].*",
	"vhosts[].headers[].*",
	"vhosts[].headers[]",
	"vhosts[].client-ssl-key",
	"vhosts[].client-ssl-cert",
	"vhosts[].client-ssl-ca",
	"vhosts[].client-ssl-ciphers",
	"vhosts[].onlyraw",
};

enum lejp_vhost_paths {
	LEJPVP,
	LEJPVP_MOUNTS,
	LEJPVP_NAME,
	LEJPVP_PORT,
	LEJPVP_INTERFACE,
	LEJPVP_UNIXSKT,
	LEJPVP_STS,
	LEJPVP_HOST_SSL_KEY,
	LEJPVP_HOST_SSL_CERT,
	LEJPVP_HOST_SSL_CA,
	LEJPVP_ACCESS_LOG,
	LEJPVP_MOUNTPOINT,
	LEJPVP_ORIGIN,
	LEJPVP_MOUNT_PROTOCOL,
	LEJPVP_DEFAULT,
	LEJPVP_DEFAULT_AUTH_MASK,
	LEJPVP_CGI_TIMEOUT,
	LEJPVP_CGI_ENV,
	LEJPVP_MOUNT_CACHE_MAX_AGE,
	LEJPVP_MOUNT_CACHE_REUSE,
	LEJPVP_MOUNT_CACHE_REVALIDATE,
	LEJPVP_MOUNT_BASIC_AUTH,
	LEJPVP_MOUNT_CACHE_INTERMEDIARIES,
	LEJPVP_MOUNT_EXTRA_MIMETYPES,
	LEJPVP_MOUNT_INTERPRET,
	LEJPVP_PROTOCOL_NAME_OPT,
	LEJPVP_PROTOCOL_NAME,
	LEJPVP_PROTOCOL,
	LEJPVP_KEEPALIVE_TIMEOUT,
	LEJPVP_ENABLE_CLIENT_SSL,
	LEJPVP_CIPHERS,
	LEJPVP_ECDH_CURVE,
	LEJPVP_NOIPV6,
	LEJPVP_IPV6ONLY,
	LEJPVP_SSL_OPTION_SET,
	LEJPVP_SSL_OPTION_CLEAR,
	LEJPVP_PMO,
	LEJPVP_HEADERS_NAME,
	LEJPVP_HEADERS,
	LEJPVP_CLIENT_SSL_KEY,
	LEJPVP_CLIENT_SSL_CERT,
	LEJPVP_CLIENT_SSL_CA,
	LEJPVP_CLIENT_CIPHERS,
	LEJPVP_FLAG_ONLYRAW,
};

static const char * const parser_errs[] = {
	"",
	"",
	"No opening '{'",
	"Expected closing '}'",
	"Expected '\"'",
	"String underrun",
	"Illegal unescaped control char",
	"Illegal escape format",
	"Illegal hex number",
	"Expected ':'",
	"Illegal value start",
	"Digit required after decimal point",
	"Bad number format",
	"Bad exponent format",
	"Unknown token",
	"Too many ']'",
	"Mismatched ']'",
	"Expected ']'",
	"JSON nesting limit exceeded",
	"Nesting tracking used up",
	"Number too long",
	"Comma or block end expected",
	"Unknown",
	"Parser callback errored (see earlier error)",
};

#define MAX_PLUGIN_DIRS 10

struct jpargs {
	struct lws_context_creation_info *info;
	struct lws_context *context;
	const struct lws_protocols *protocols;
	const struct lws_extension *extensions;
	char *p, *end, valid;
	struct lws_http_mount *head, *last;

	struct lws_protocol_vhost_options *pvo;
	struct lws_protocol_vhost_options *pvo_em;
	struct lws_protocol_vhost_options *pvo_int;
	struct lws_http_mount m;
	const char **plugin_dirs;
	int count_plugin_dirs;

	unsigned int enable_client_ssl:1;
	unsigned int fresh_mount:1;
	unsigned int any_vhosts:1;
};

static void *
lwsws_align(struct jpargs *a)
{
	if ((lws_intptr_t)(a->p) & 15)
		a->p += 16 - ((lws_intptr_t)(a->p) & 15);

	return a->p;
}

static int
arg_to_bool(const char *s)
{
	static const char * const on[] = { "on", "yes", "true" };
	int n = atoi(s);

	if (n)
		return 1;

	for (n = 0; n < ARRAY_SIZE(on); n++)
		if (!strcasecmp(s, on[n]))
			return 1;

	return 0;
}

static signed char
lejp_globals_cb(struct lejp_ctx *ctx, char reason)
{
	struct jpargs *a = (struct jpargs *)ctx->user;
	struct lws_protocol_vhost_options *rej;
	int n;

	/* we only match on the prepared path strings */
	if (!(reason & LEJP_FLAG_CB_IS_VALUE) || !ctx->path_match)
		return 0;

	/* this catches, eg, vhosts[].headers[].xxx */
	if (reason == LEJPCB_VAL_STR_END &&
	    ctx->path_match == LWJPGP_REJECT_SERVICE_KEYWORDS_NAME + 1) {
		rej = lwsws_align(a);
		a->p += sizeof(*rej);

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		rej->next = a->info->reject_service_keywords;
		a->info->reject_service_keywords = rej;
		rej->name = a->p;
		 lwsl_notice("  adding rej %s=%s\n", a->p, ctx->buf);
		a->p += n - 1;
		*(a->p++) = '\0';
		rej->value = a->p;
		rej->options = NULL;
		goto dostring;
	}

	switch (ctx->path_match - 1) {
	case LEJPGP_UID:
		a->info->uid = atoi(ctx->buf);
		return 0;
	case LEJPGP_GID:
		a->info->gid = atoi(ctx->buf);
		return 0;
	case LEJPGP_COUNT_THREADS:
		a->info->count_threads = atoi(ctx->buf);
		return 0;
	case LWJPGP_INIT_SSL:
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
		return 0;
	case LEJPGP_SERVER_STRING:
		a->info->server_string = a->p;
		break;
	case LEJPGP_PLUGIN_DIR:
		if (a->count_plugin_dirs == MAX_PLUGIN_DIRS - 1) {
			lwsl_err("Too many plugin dirs\n");
			return -1;
		}
		a->plugin_dirs[a->count_plugin_dirs++] = a->p;
		break;

	case LWJPGP_PINGPONG_SECS:
		a->info->ws_ping_pong_interval = atoi(ctx->buf);
		return 0;

	case LWJPGP_TIMEOUT_SECS:
		a->info->timeout_secs = atoi(ctx->buf);
		return 0;

	default:
		return 0;
	}

dostring:
	a->p += lws_snprintf(a->p, a->end - a->p, "%s", ctx->buf);
	*(a->p)++ = '\0';

	return 0;
}

static signed char
lejp_vhosts_cb(struct lejp_ctx *ctx, char reason)
{
	struct jpargs *a = (struct jpargs *)ctx->user;
	struct lws_protocol_vhost_options *pvo, *mp_cgienv, *headers;
	struct lws_http_mount *m;
	char *p, *p1;
	int n;

#if 0
	lwsl_notice(" %d: %s (%d)\n", reason, ctx->path, ctx->path_match);
	for (n = 0; n < ctx->wildcount; n++)
		lwsl_notice("    %d\n", ctx->wild[n]);
#endif

	if (reason == LEJPCB_OBJECT_START && ctx->path_match == LEJPVP + 1) {
		/* set the defaults for this vhost */
		a->valid = 1;
		a->head = NULL;
		a->last = NULL;
		a->info->port = 0;
		a->info->iface = NULL;
		a->info->protocols = a->protocols;
		a->info->extensions = a->extensions;
		a->info->ssl_cert_filepath = NULL;
		a->info->ssl_private_key_filepath = NULL;
		a->info->ssl_ca_filepath = NULL;
		a->info->client_ssl_cert_filepath = NULL;
		a->info->client_ssl_private_key_filepath = NULL;
		a->info->client_ssl_ca_filepath = NULL;
		a->info->client_ssl_cipher_list = "ECDHE-ECDSA-AES256-GCM-SHA384:"
			"ECDHE-RSA-AES256-GCM-SHA384:"
			"DHE-RSA-AES256-GCM-SHA384:"
			"ECDHE-RSA-AES256-SHA384:"
			"HIGH:!aNULL:!eNULL:!EXPORT:"
			"!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:"
			"!SHA1:!DHE-RSA-AES128-GCM-SHA256:"
			"!DHE-RSA-AES128-SHA256:"
			"!AES128-GCM-SHA256:"
			"!AES128-SHA256:"
			"!DHE-RSA-AES256-SHA256:"
			"!AES256-GCM-SHA384:"
			"!AES256-SHA256";
		a->info->timeout_secs = 5;
		a->info->ssl_cipher_list = "ECDHE-ECDSA-AES256-GCM-SHA384:"
				       "ECDHE-RSA-AES256-GCM-SHA384:"
				       "DHE-RSA-AES256-GCM-SHA384:"
				       "ECDHE-RSA-AES256-SHA384:"
				       "HIGH:!aNULL:!eNULL:!EXPORT:"
				       "!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:"
				       "!SHA1:!DHE-RSA-AES128-GCM-SHA256:"
				       "!DHE-RSA-AES128-SHA256:"
				       "!AES128-GCM-SHA256:"
				       "!AES128-SHA256:"
				       "!DHE-RSA-AES256-SHA256:"
				       "!AES256-GCM-SHA384:"
				       "!AES256-SHA256";
		a->info->pvo = NULL;
		a->info->headers = NULL;
		a->info->keepalive_timeout = 5;
		a->info->log_filepath = NULL;
		a->info->options &= ~(LWS_SERVER_OPTION_UNIX_SOCK |
				      LWS_SERVER_OPTION_STS | LWS_SERVER_OPTION_ONLY_RAW);
		a->enable_client_ssl = 0;
	}

	if (reason == LEJPCB_OBJECT_START &&
	    ctx->path_match == LEJPVP_MOUNTS + 1) {
		a->fresh_mount = 1;
		memset(&a->m, 0, sizeof(a->m));
	}

	/* this catches, eg, vhosts[].ws-protocols[].xxx-protocol */
	if (reason == LEJPCB_OBJECT_START &&
	    ctx->path_match == LEJPVP_PROTOCOL_NAME + 1) {
		a->pvo = lwsws_align(a);
		a->p += sizeof(*a->pvo);

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		/* ie, enable this protocol, no options yet */
		a->pvo->next = a->info->pvo;
		a->info->pvo = a->pvo;
		a->pvo->name = a->p;
		lwsl_notice("  adding protocol %s\n", a->p);
		a->p += n;
		a->pvo->value = a->p;
		a->pvo->options = NULL;
		goto dostring;
	}

	/* this catches, eg, vhosts[].headers[].xxx */
	if (reason == LEJPCB_VAL_STR_END &&
	    ctx->path_match == LEJPVP_HEADERS_NAME + 1) {
		headers = lwsws_align(a);
		a->p += sizeof(*headers);

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		/* ie, enable this protocol, no options yet */
		headers->next = a->info->headers;
		a->info->headers = headers;
		headers->name = a->p;
		// lwsl_notice("  adding header %s=%s\n", a->p, ctx->buf);
		a->p += n - 1;
		*(a->p++) = ':';
		if (a->p < a->end)
			*(a->p++) = '\0';
		else
			*(a->p - 1) = '\0';
		headers->value = a->p;
		headers->options = NULL;
		goto dostring;
	}

	if (reason == LEJPCB_OBJECT_END &&
	    (ctx->path_match == LEJPVP + 1 || !ctx->path[0]) &&
	    a->valid) {

		struct lws_vhost *vhost;

		//lwsl_notice("%s\n", ctx->path);
		if (!a->info->port) {
			lwsl_err("Port required (eg, 443)");
			return 1;
		}
		a->valid = 0;
		a->info->mounts = a->head;

		vhost = lws_create_vhost(a->context, a->info);
		if (!vhost) {
			lwsl_err("Failed to create vhost %s\n",
				 a->info->vhost_name);
			return 1;
		}
		a->any_vhosts = 1;

		if (a->enable_client_ssl) {
			const char *cert_filepath = a->info->client_ssl_cert_filepath;
			const char *private_key_filepath = a->info->client_ssl_private_key_filepath;
			const char *ca_filepath = a->info->client_ssl_ca_filepath;
			const char *cipher_list = a->info->client_ssl_cipher_list;
			memset(a->info, 0, sizeof(*a->info));
			a->info->client_ssl_cert_filepath = cert_filepath;
			a->info->client_ssl_private_key_filepath = private_key_filepath;
			a->info->client_ssl_ca_filepath = ca_filepath;
			a->info->client_ssl_cipher_list = cipher_list;
			a->info->options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
			lws_init_vhost_client_ssl(a->info, vhost);
		}

		return 0;
	}

	if (reason == LEJPCB_OBJECT_END &&
	    ctx->path_match == LEJPVP_MOUNTS + 1) {
		static const char * const mount_protocols[] = {
			"http://",
			"https://",
			"file://",
			"cgi://",
			">http://",
			">https://",
			"callback://",
			"gzip://",
		};

		if (!a->fresh_mount)
			return 0;

		if (!a->m.mountpoint || !a->m.origin) {
			lwsl_err("mountpoint and origin required\n");
			return 1;
		}
		lwsl_debug("adding mount %s\n", a->m.mountpoint);
		m = lwsws_align(a);
		memcpy(m, &a->m, sizeof(*m));
		if (a->last)
			a->last->mount_next = m;

		for (n = 0; n < ARRAY_SIZE(mount_protocols); n++)
			if (!strncmp(a->m.origin, mount_protocols[n],
			     strlen(mount_protocols[n]))) {
				lwsl_info("----%s\n", a->m.origin);
				m->origin_protocol = n;
				m->origin = a->m.origin +
					    strlen(mount_protocols[n]);
				break;
			}

		if (n == ARRAY_SIZE(mount_protocols)) {
			lwsl_err("unsupported protocol:// %s\n", a->m.origin);
			return 1;
		}

		a->p += sizeof(*m);
		if (!a->head)
			a->head = m;

		a->last = m;
		a->fresh_mount = 0;
	}

	/* we only match on the prepared path strings */
	if (!(reason & LEJP_FLAG_CB_IS_VALUE) || !ctx->path_match)
		return 0;

	switch (ctx->path_match - 1) {
	case LEJPVP_NAME:
		a->info->vhost_name = a->p;
		break;
	case LEJPVP_PORT:
		a->info->port = atoi(ctx->buf);
		return 0;
	case LEJPVP_INTERFACE:
		a->info->iface = a->p;
		break;
	case LEJPVP_UNIXSKT:
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_UNIX_SOCK;
		else
			a->info->options &= ~(LWS_SERVER_OPTION_UNIX_SOCK);
		return 0;
	case LEJPVP_STS:
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_STS;
		else
			a->info->options &= ~(LWS_SERVER_OPTION_STS);
		return 0;
	case LEJPVP_HOST_SSL_KEY:
		a->info->ssl_private_key_filepath = a->p;
		break;
	case LEJPVP_HOST_SSL_CERT:
		a->info->ssl_cert_filepath = a->p;
		break;
	case LEJPVP_HOST_SSL_CA:
		a->info->ssl_ca_filepath = a->p;
		break;
	case LEJPVP_ACCESS_LOG:
		a->info->log_filepath = a->p;
		break;
	case LEJPVP_MOUNTPOINT:
		a->m.mountpoint = a->p;
		a->m.mountpoint_len = (unsigned char)strlen(ctx->buf);
		break;
	case LEJPVP_ORIGIN:
		if (!strncmp(ctx->buf, "callback://", 11))
			a->m.protocol = a->p + 11;

		if (!a->m.origin)
			a->m.origin = a->p;
		break;
	case LEJPVP_DEFAULT:
		a->m.def = a->p;
		break;
	case LEJPVP_DEFAULT_AUTH_MASK:
		a->m.auth_mask = atoi(ctx->buf);
		return 0;
	case LEJPVP_MOUNT_CACHE_MAX_AGE:
		a->m.cache_max_age = atoi(ctx->buf);
		return 0;
	case LEJPVP_MOUNT_CACHE_REUSE:
		a->m.cache_reusable = arg_to_bool(ctx->buf);
		return 0;
	case LEJPVP_MOUNT_CACHE_REVALIDATE:
		a->m.cache_revalidate = arg_to_bool(ctx->buf);
		return 0;
	case LEJPVP_MOUNT_CACHE_INTERMEDIARIES:
		a->m.cache_intermediaries = arg_to_bool(ctx->buf);;
		return 0;
	case LEJPVP_MOUNT_BASIC_AUTH:
		a->m.basic_auth_login_file = a->p;
		break;
	case LEJPVP_CGI_TIMEOUT:
		a->m.cgi_timeout = atoi(ctx->buf);
		return 0;
	case LEJPVP_KEEPALIVE_TIMEOUT:
		a->info->keepalive_timeout = atoi(ctx->buf);
		return 0;
	case LEJPVP_CLIENT_CIPHERS:
		a->info->client_ssl_cipher_list = a->p;
		break;
	case LEJPVP_CIPHERS:
		a->info->ssl_cipher_list = a->p;
		break;
	case LEJPVP_ECDH_CURVE:
		a->info->ecdh_curve = a->p;
		break;
	case LEJPVP_PMO:
	case LEJPVP_CGI_ENV:
		mp_cgienv = lwsws_align(a);
		a->p += sizeof(*a->m.cgienv);

		mp_cgienv->next = a->m.cgienv;
		a->m.cgienv = mp_cgienv;

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		mp_cgienv->name = a->p;
		a->p += n;
		mp_cgienv->value = a->p;
		mp_cgienv->options = NULL;
		//lwsl_notice("    adding pmo / cgi-env '%s' = '%s'\n", mp_cgienv->name,
		//		mp_cgienv->value);
		goto dostring;

	case LEJPVP_PROTOCOL_NAME_OPT:
		/* this catches, eg,
		 * vhosts[].ws-protocols[].xxx-protocol.yyy-option
		 * ie, these are options attached to a protocol with { }
		 */
		pvo = lwsws_align(a);
		a->p += sizeof(*a->pvo);

		n = lejp_get_wildcard(ctx, 1, a->p, a->end - a->p);
		/* ie, enable this protocol, no options yet */
		pvo->next = a->pvo->options;
		a->pvo->options = pvo;
		pvo->name = a->p;
		a->p += n;
		pvo->value = a->p;
		pvo->options = NULL;
		break;

	case LEJPVP_MOUNT_EXTRA_MIMETYPES:
		a->pvo_em = lwsws_align(a);
		a->p += sizeof(*a->pvo_em);

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		/* ie, enable this protocol, no options yet */
		a->pvo_em->next = a->m.extra_mimetypes;
		a->m.extra_mimetypes = a->pvo_em;
		a->pvo_em->name = a->p;
		lwsl_notice("  adding extra-mimetypes %s -> %s\n", a->p, ctx->buf);
		a->p += n;
		a->pvo_em->value = a->p;
		a->pvo_em->options = NULL;
		break;

	case LEJPVP_MOUNT_INTERPRET:
		a->pvo_int = lwsws_align(a);
		a->p += sizeof(*a->pvo_int);

		n = lejp_get_wildcard(ctx, 0, a->p, a->end - a->p);
		/* ie, enable this protocol, no options yet */
		a->pvo_int->next = a->m.interpret;
		a->m.interpret = a->pvo_int;
		a->pvo_int->name = a->p;
		lwsl_notice("  adding interpret %s -> %s\n", a->p,
			    ctx->buf);
		a->p += n;
		a->pvo_int->value = a->p;
		a->pvo_int->options = NULL;
		break;

	case LEJPVP_ENABLE_CLIENT_SSL:
		a->enable_client_ssl = arg_to_bool(ctx->buf);
		return 0;
	case LEJPVP_CLIENT_SSL_KEY:
		a->info->client_ssl_private_key_filepath = a->p;
		break;
	case LEJPVP_CLIENT_SSL_CERT:
		a->info->client_ssl_cert_filepath = a->p;
		break;
	case LEJPVP_CLIENT_SSL_CA:
		a->info->client_ssl_ca_filepath = a->p;
		break;

	case LEJPVP_NOIPV6:
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_DISABLE_IPV6;
		else
			a->info->options &= ~(LWS_SERVER_OPTION_DISABLE_IPV6);
		return 0;

	case LEJPVP_FLAG_ONLYRAW:
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_ONLY_RAW;
		else
			a->info->options &= ~(LWS_SERVER_OPTION_ONLY_RAW);
		return 0;

	case LEJPVP_IPV6ONLY:
		a->info->options |= LWS_SERVER_OPTION_IPV6_V6ONLY_MODIFY;
		if (arg_to_bool(ctx->buf))
			a->info->options |= LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE;
		else
			a->info->options &= ~(LWS_SERVER_OPTION_IPV6_V6ONLY_VALUE);
		return 0;

	case LEJPVP_SSL_OPTION_SET:
		a->info->ssl_options_set |= atol(ctx->buf);
		return 0;
	case LEJPVP_SSL_OPTION_CLEAR:
		a->info->ssl_options_clear |= atol(ctx->buf);
		return 0;

	default:
		return 0;
	}

dostring:
	p = ctx->buf;
	p1 = strstr(p, ESC_INSTALL_DATADIR);
	if (p1) {
		n = p1 - p;
		if (n > a->end - a->p)
			n = a->end - a->p;
		strncpy(a->p, p, n);
		a->p += n;
		a->p += lws_snprintf(a->p, a->end - a->p, "%s", LWS_INSTALL_DATADIR);
		p += n + strlen(ESC_INSTALL_DATADIR);
	}

	a->p += lws_snprintf(a->p, a->end - a->p, "%s", p);
	*(a->p)++ = '\0';

	return 0;
}

/*
 * returns 0 = OK, 1 = can't open, 2 = parsing error
 */

static int
lwsws_get_config(void *user, const char *f, const char * const *paths,
		 int count_paths, lejp_callback cb)
{
	unsigned char buf[128];
	struct lejp_ctx ctx;
	int n, m, fd;

	fd = open(f, O_RDONLY);
	if (fd < 0) {
		lwsl_err("Cannot open %s\n", f);
		return 2;
	}
	lwsl_info("%s: %s\n", __func__, f);
	lejp_construct(&ctx, cb, user, paths, count_paths);

	do {
		n = read(fd, buf, sizeof(buf));
		if (!n)
			break;

		m = (int)(signed char)lejp_parse(&ctx, buf, n);
	} while (m == LEJP_CONTINUE);

	close(fd);
	n = ctx.line;
	lejp_destruct(&ctx);

	if (m < 0) {
		lwsl_err("%s(%u): parsing error %d: %s\n", f, n, m,
			 parser_errs[-m]);
		return 2;
	}

	return 0;
}

#if defined(LWS_WITH_LIBUV) && UV_VERSION_MAJOR > 0

static int
lwsws_get_config_d(void *user, const char *d, const char * const *paths,
		   int count_paths, lejp_callback cb)
{
	uv_dirent_t dent;
	uv_fs_t req;
	char path[256];
	int ret = 0, ir;
	uv_loop_t loop;

	ir = uv_loop_init(&loop);
	if (ir) {
		lwsl_err("%s: loop init failed %d\n", __func__, ir);
	}

	if (!uv_fs_scandir(&loop, &req, d, 0, NULL)) {
		lwsl_err("Scandir on %s failed\n", d);
		return 2;
	}

	while (uv_fs_scandir_next(&req, &dent) != UV_EOF) {
		lws_snprintf(path, sizeof(path) - 1, "%s/%s", d, dent.name);
		ret = lwsws_get_config(user, path, paths, count_paths, cb);
		if (ret)
			goto bail;
	}

bail:
	uv_fs_req_cleanup(&req);
	while (uv_loop_close(&loop))
		;

	return ret;
}

#else

#ifndef _WIN32
static int filter(const struct dirent *ent)
{
	if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
		return 0;

	return 1;
}
#endif

static int
lwsws_get_config_d(void *user, const char *d, const char * const *paths,
		   int count_paths, lejp_callback cb)
{
#ifndef _WIN32
	struct dirent **namelist;
	char path[256];
	int n, i, ret = 0;

	n = scandir(d, &namelist, filter, alphasort);
	if (n < 0) {
		lwsl_err("Scandir on %s failed\n", d);
		return 1;
	}

	for (i = 0; i < n; i++) {
		if (strchr(namelist[i]->d_name, '~'))
			goto skip;
		lws_snprintf(path, sizeof(path) - 1, "%s/%s", d,
			 namelist[i]->d_name);
		ret = lwsws_get_config(user, path, paths, count_paths, cb);
		if (ret) {
			while (i++ < n)
				free(namelist[i]);
			goto bail;
		}
skip:
		free(namelist[i]);
	}

bail:
	free(namelist);

	return ret;
#else
	return 0;
#endif
}

#endif

int
lwsws_get_config_globals(struct lws_context_creation_info *info, const char *d,
			 char **cs, int *len)
{
	struct jpargs a;
	const char * const *old = info->plugin_dirs;
	char dd[128];

	memset(&a, 0, sizeof(a));

	a.info = info;
	a.p = *cs;
	a.end = (a.p + *len) - 1;
	a.valid = 0;

	lwsws_align(&a);
	info->plugin_dirs = (void *)a.p;
	a.plugin_dirs = (void *)a.p; /* writeable version */
	a.p += MAX_PLUGIN_DIRS * sizeof(void *);

	/* copy any default paths */

	while (old && *old) {
		a.plugin_dirs[a.count_plugin_dirs++] = *old;
		old++;
	}

	lws_snprintf(dd, sizeof(dd) - 1, "%s/conf", d);
	if (lwsws_get_config(&a, dd, paths_global,
			     ARRAY_SIZE(paths_global), lejp_globals_cb) > 1)
		return 1;
	lws_snprintf(dd, sizeof(dd) - 1, "%s/conf.d", d);
	if (lwsws_get_config_d(&a, dd, paths_global,
			       ARRAY_SIZE(paths_global), lejp_globals_cb) > 1)
		return 1;

	a.plugin_dirs[a.count_plugin_dirs] = NULL;

	*cs = a.p;
	*len = a.end - a.p;

	return 0;
}

int
lwsws_get_config_vhosts(struct lws_context *context,
			struct lws_context_creation_info *info, const char *d,
			char **cs, int *len)
{
	struct jpargs a;
	char dd[128];

	memset(&a, 0, sizeof(a));

	a.info = info;
	a.p = *cs;
	a.end = a.p + *len;
	a.valid = 0;
	a.context = context;
	a.protocols = info->protocols;
	a.extensions = info->extensions;

	lws_snprintf(dd, sizeof(dd) - 1, "%s/conf", d);
	if (lwsws_get_config(&a, dd, paths_vhosts,
			     ARRAY_SIZE(paths_vhosts), lejp_vhosts_cb) > 1)
		return 1;
	lws_snprintf(dd, sizeof(dd) - 1, "%s/conf.d", d);
	if (lwsws_get_config_d(&a, dd, paths_vhosts,
			       ARRAY_SIZE(paths_vhosts), lejp_vhosts_cb) > 1)
		return 1;

	*cs = a.p;
	*len = a.end - a.p;

	if (!a.any_vhosts) {
		lwsl_err("Need at least one vhost\n");
		return 1;
	}

//	lws_finalize_startup(context);

	return 0;
}
