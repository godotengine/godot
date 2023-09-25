/*-
 * Copyright (c) 2009-2010 Brad Penoff
 * Copyright (c) 2009-2010 Humaira Kamal
 * Copyright (c) 2011-2012 Irene Ruengeler
 * Copyright (c) 2011-2012 Michael Tuexen
 *
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
 */

#ifndef __USRSCTP_H__
#define __USRSCTP_H__

#ifdef  __cplusplus
extern "C" {
#endif

#include <errno.h>
#include <sys/types.h>
#ifdef _WIN32
#ifdef _MSC_VER
#pragma warning(disable: 4200)
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#endif

#ifndef MSG_NOTIFICATION
/* This definition MUST be in sync with usrsctplib/user_socketvar.h */
#define MSG_NOTIFICATION 0x2000
#endif

#ifndef IPPROTO_SCTP
/* This is the IANA assigned protocol number of SCTP. */
#define IPPROTO_SCTP 132
#endif

#ifdef _WIN32
#if defined(_MSC_VER) && _MSC_VER >= 1600
#include <stdint.h>
#elif defined(SCTP_STDINT_INCLUDE)
#include SCTP_STDINT_INCLUDE
#else
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef          __int16 int16_t;
typedef          __int32 int32_t;
#endif

#ifndef ssize_t
#ifdef _WIN64
typedef __int64 ssize_t;
#elif defined _WIN32
typedef int ssize_t;
#else
#error "Unknown platform!"
#endif
#endif

#define MSG_EOR   0x8
#ifndef EWOULDBLOCK
#define EWOULDBLOCK  WSAEWOULDBLOCK
#endif
#ifndef EINPROGRESS
#define EINPROGRESS  WSAEINPROGRESS
#endif
#define SHUT_RD    1
#define SHUT_WR    2
#define SHUT_RDWR  3
#endif

typedef uint32_t sctp_assoc_t;

#if defined(_WIN32) && defined(_MSC_VER)
#pragma pack (push, 1)
#define SCTP_PACKED
#else
#define SCTP_PACKED __attribute__((packed))
#endif

struct sctp_common_header {
	uint16_t source_port;
	uint16_t destination_port;
	uint32_t verification_tag;
	uint32_t crc32c;
} SCTP_PACKED;

#if defined(_WIN32) && defined(_MSC_VER)
#pragma pack(pop)
#endif
#undef SCTP_PACKED

#define AF_CONN 123
/* The definition of struct sockaddr_conn MUST be in
 * tune with other sockaddr_* structures.
 */
#if defined(__APPLE__) || defined(__Bitrig__) || defined(__DragonFly__) || \
    defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
struct sockaddr_conn {
	uint8_t sconn_len;
	uint8_t sconn_family;
	uint16_t sconn_port;
	void *sconn_addr;
};
#else
struct sockaddr_conn {
	uint16_t sconn_family;
	uint16_t sconn_port;
	void *sconn_addr;
};
#endif

union sctp_sockstore {
	struct sockaddr_in sin;
	struct sockaddr_in6 sin6;
	struct sockaddr_conn sconn;
	struct sockaddr sa;
};

#define SCTP_FUTURE_ASSOC  0
#define SCTP_CURRENT_ASSOC 1
#define SCTP_ALL_ASSOC     2

#define SCTP_EVENT_READ    0x0001
#define SCTP_EVENT_WRITE   0x0002
#define SCTP_EVENT_ERROR   0x0004

/***  Structures and definitions to use the socket API  ***/

#define SCTP_ALIGN_RESV_PAD 92
#define SCTP_ALIGN_RESV_PAD_SHORT 76

struct sctp_rcvinfo {
	uint16_t rcv_sid;
	uint16_t rcv_ssn;
	uint16_t rcv_flags;
	uint32_t rcv_ppid;
	uint32_t rcv_tsn;
	uint32_t rcv_cumtsn;
	uint32_t rcv_context;
	sctp_assoc_t rcv_assoc_id;
};

struct sctp_nxtinfo {
	uint16_t nxt_sid;
	uint16_t nxt_flags;
	uint32_t nxt_ppid;
	uint32_t nxt_length;
	sctp_assoc_t nxt_assoc_id;
};

#define SCTP_NO_NEXT_MSG           0x0000
#define SCTP_NEXT_MSG_AVAIL        0x0001
#define SCTP_NEXT_MSG_ISCOMPLETE   0x0002
#define SCTP_NEXT_MSG_IS_UNORDERED 0x0004
#define SCTP_NEXT_MSG_IS_NOTIFICATION 0x0008

struct sctp_recvv_rn {
	struct sctp_rcvinfo recvv_rcvinfo;
	struct sctp_nxtinfo recvv_nxtinfo;
};

#define SCTP_RECVV_NOINFO  0
#define SCTP_RECVV_RCVINFO 1
#define SCTP_RECVV_NXTINFO 2
#define SCTP_RECVV_RN      3

#define SCTP_SENDV_NOINFO   0
#define SCTP_SENDV_SNDINFO  1
#define SCTP_SENDV_PRINFO   2
#define SCTP_SENDV_AUTHINFO 3
#define SCTP_SENDV_SPA      4

#define SCTP_SEND_SNDINFO_VALID  0x00000001
#define SCTP_SEND_PRINFO_VALID   0x00000002
#define SCTP_SEND_AUTHINFO_VALID 0x00000004

struct sctp_snd_all_completes {
	uint16_t sall_stream;
	uint16_t sall_flags;
	uint32_t sall_ppid;
	uint32_t sall_context;
	uint32_t sall_num_sent;
	uint32_t sall_num_failed;
};

struct sctp_sndinfo {
	uint16_t snd_sid;
	uint16_t snd_flags;
	uint32_t snd_ppid;
	uint32_t snd_context;
	sctp_assoc_t snd_assoc_id;
};

struct sctp_prinfo {
	uint16_t pr_policy;
	uint32_t pr_value;
};

struct sctp_authinfo {
	uint16_t auth_keynumber;
};

struct sctp_sendv_spa {
	uint32_t sendv_flags;
	struct sctp_sndinfo sendv_sndinfo;
	struct sctp_prinfo sendv_prinfo;
	struct sctp_authinfo sendv_authinfo;
};

struct sctp_udpencaps {
	struct sockaddr_storage sue_address;
	uint32_t sue_assoc_id;
	uint16_t sue_port;
};

/********  Notifications  **************/

/* notification types */
#define SCTP_ASSOC_CHANGE                 0x0001
#define SCTP_PEER_ADDR_CHANGE             0x0002
#define SCTP_REMOTE_ERROR                 0x0003
#define SCTP_SEND_FAILED                  0x0004
#define SCTP_SHUTDOWN_EVENT               0x0005
#define SCTP_ADAPTATION_INDICATION        0x0006
#define SCTP_PARTIAL_DELIVERY_EVENT       0x0007
#define SCTP_AUTHENTICATION_EVENT         0x0008
#define SCTP_STREAM_RESET_EVENT           0x0009
#define SCTP_SENDER_DRY_EVENT             0x000a
#define SCTP_NOTIFICATIONS_STOPPED_EVENT  0x000b
#define SCTP_ASSOC_RESET_EVENT            0x000c
#define SCTP_STREAM_CHANGE_EVENT          0x000d
#define SCTP_SEND_FAILED_EVENT            0x000e

/* notification event structures */


/* association change event */
struct sctp_assoc_change {
	uint16_t sac_type;
	uint16_t sac_flags;
	uint32_t sac_length;
	uint16_t sac_state;
	uint16_t sac_error;
	uint16_t sac_outbound_streams;
	uint16_t sac_inbound_streams;
	sctp_assoc_t sac_assoc_id;
	uint8_t sac_info[]; /* not available yet */
};

/* sac_state values */
#define SCTP_COMM_UP        0x0001
#define SCTP_COMM_LOST      0x0002
#define SCTP_RESTART        0x0003
#define SCTP_SHUTDOWN_COMP  0x0004
#define SCTP_CANT_STR_ASSOC 0x0005

/* sac_info values */
#define SCTP_ASSOC_SUPPORTS_PR           0x01
#define SCTP_ASSOC_SUPPORTS_AUTH         0x02
#define SCTP_ASSOC_SUPPORTS_ASCONF       0x03
#define SCTP_ASSOC_SUPPORTS_MULTIBUF     0x04
#define SCTP_ASSOC_SUPPORTS_RE_CONFIG    0x05
#define SCTP_ASSOC_SUPPORTS_INTERLEAVING 0x06
#define SCTP_ASSOC_SUPPORTS_MAX          0x06

/* Address event */
struct sctp_paddr_change {
	uint16_t spc_type;
	uint16_t spc_flags;
	uint32_t spc_length;
	struct sockaddr_storage spc_aaddr;
	uint32_t spc_state;
	uint32_t spc_error;
	sctp_assoc_t spc_assoc_id;
	uint8_t spc_padding[4];
};

/* paddr state values */
#define SCTP_ADDR_AVAILABLE   0x0001
#define SCTP_ADDR_UNREACHABLE 0x0002
#define SCTP_ADDR_REMOVED     0x0003
#define SCTP_ADDR_ADDED       0x0004
#define SCTP_ADDR_MADE_PRIM   0x0005
#define SCTP_ADDR_CONFIRMED   0x0006

/* remote error events */
struct sctp_remote_error {
	uint16_t sre_type;
	uint16_t sre_flags;
	uint32_t sre_length;
	uint16_t sre_error;
	sctp_assoc_t sre_assoc_id;
	uint8_t sre_data[];
};

/* shutdown event */
struct sctp_shutdown_event {
	uint16_t sse_type;
	uint16_t sse_flags;
	uint32_t sse_length;
	sctp_assoc_t sse_assoc_id;
};

/* Adaptation layer indication */
struct sctp_adaptation_event {
	uint16_t sai_type;
	uint16_t sai_flags;
	uint32_t sai_length;
	uint32_t sai_adaptation_ind;
	sctp_assoc_t sai_assoc_id;
};

/* Partial delivery event */
struct sctp_pdapi_event {
	uint16_t pdapi_type;
	uint16_t pdapi_flags;
	uint32_t pdapi_length;
	uint32_t pdapi_indication;
	uint32_t pdapi_stream;
	uint32_t pdapi_seq;
	sctp_assoc_t pdapi_assoc_id;
};

/* indication values */
#define SCTP_PARTIAL_DELIVERY_ABORTED  0x0001

/* SCTP authentication event */
struct sctp_authkey_event {
	uint16_t auth_type;
	uint16_t auth_flags;
	uint32_t auth_length;
	uint16_t auth_keynumber;
	uint32_t auth_indication;
	sctp_assoc_t auth_assoc_id;
};

/* indication values */
#define SCTP_AUTH_NEW_KEY   0x0001
#define SCTP_AUTH_NO_AUTH   0x0002
#define SCTP_AUTH_FREE_KEY  0x0003

/* SCTP sender dry event */
struct sctp_sender_dry_event {
	uint16_t sender_dry_type;
	uint16_t sender_dry_flags;
	uint32_t sender_dry_length;
	sctp_assoc_t sender_dry_assoc_id;
};


/* Stream reset event - subscribe to SCTP_STREAM_RESET_EVENT */
struct sctp_stream_reset_event {
	uint16_t strreset_type;
	uint16_t strreset_flags;
	uint32_t strreset_length;
	sctp_assoc_t strreset_assoc_id;
	uint16_t strreset_stream_list[];
};

/* flags in stream_reset_event (strreset_flags) */
#define SCTP_STREAM_RESET_INCOMING_SSN  0x0001
#define SCTP_STREAM_RESET_OUTGOING_SSN  0x0002
#define SCTP_STREAM_RESET_DENIED        0x0004 /* SCTP_STRRESET_FAILED */
#define SCTP_STREAM_RESET_FAILED        0x0008 /* SCTP_STRRESET_FAILED */
#define SCTP_STREAM_CHANGED_DENIED      0x0010

#define SCTP_STREAM_RESET_INCOMING      0x00000001
#define SCTP_STREAM_RESET_OUTGOING      0x00000002


/* Assoc reset event - subscribe to SCTP_ASSOC_RESET_EVENT */
struct sctp_assoc_reset_event {
	uint16_t assocreset_type;
	uint16_t assocreset_flags;
	uint32_t assocreset_length;
	sctp_assoc_t assocreset_assoc_id;
	uint32_t assocreset_local_tsn;
	uint32_t assocreset_remote_tsn;
};

#define SCTP_ASSOC_RESET_DENIED        0x0004
#define SCTP_ASSOC_RESET_FAILED        0x0008


/* Stream change event - subscribe to SCTP_STREAM_CHANGE_EVENT */
struct sctp_stream_change_event {
	uint16_t strchange_type;
	uint16_t strchange_flags;
	uint32_t strchange_length;
	sctp_assoc_t strchange_assoc_id;
	uint16_t strchange_instrms;
	uint16_t strchange_outstrms;
};

#define SCTP_STREAM_CHANGE_DENIED	0x0004
#define SCTP_STREAM_CHANGE_FAILED	0x0008


/* SCTP send failed event */
struct sctp_send_failed_event {
	uint16_t ssfe_type;
	uint16_t ssfe_flags;
	uint32_t ssfe_length;
	uint32_t ssfe_error;
	struct sctp_sndinfo ssfe_info;
	sctp_assoc_t ssfe_assoc_id;
	uint8_t  ssfe_data[];
};

/* flag that indicates state of data */
#define SCTP_DATA_UNSENT  0x0001	/* inqueue never on wire */
#define SCTP_DATA_SENT    0x0002	/* on wire at failure */

/* SCTP event option */
struct sctp_event {
	sctp_assoc_t   se_assoc_id;
	uint16_t       se_type;
	uint8_t        se_on;
};

union sctp_notification {
	struct sctp_tlv {
		uint16_t sn_type;
		uint16_t sn_flags;
		uint32_t sn_length;
	} sn_header;
	struct sctp_assoc_change sn_assoc_change;
	struct sctp_paddr_change sn_paddr_change;
	struct sctp_remote_error sn_remote_error;
	struct sctp_shutdown_event sn_shutdown_event;
	struct sctp_adaptation_event sn_adaptation_event;
	struct sctp_pdapi_event sn_pdapi_event;
	struct sctp_authkey_event sn_auth_event;
	struct sctp_sender_dry_event sn_sender_dry_event;
	struct sctp_send_failed_event sn_send_failed_event;
	struct sctp_stream_reset_event sn_strreset_event;
	struct sctp_assoc_reset_event  sn_assocreset_event;
	struct sctp_stream_change_event sn_strchange_event;
};

struct sctp_event_subscribe {
	uint8_t sctp_data_io_event;
	uint8_t sctp_association_event;
	uint8_t sctp_address_event;
	uint8_t sctp_send_failure_event;
	uint8_t sctp_peer_error_event;
	uint8_t sctp_shutdown_event;
	uint8_t sctp_partial_delivery_event;
	uint8_t sctp_adaptation_layer_event;
	uint8_t sctp_authentication_event;
	uint8_t sctp_sender_dry_event;
	uint8_t sctp_stream_reset_event;
};



/* Flags that go into the sinfo->sinfo_flags field */
#define SCTP_DATA_LAST_FRAG   0x0001 /* tail part of the message could not be sent */
#define SCTP_DATA_NOT_FRAG    0x0003 /* complete message could not be sent */
#define SCTP_NOTIFICATION     0x0010 /* next message is a notification */
#define SCTP_COMPLETE         0x0020 /* next message is complete */
#define SCTP_EOF              0x0100 /* Start shutdown procedures */
#define SCTP_ABORT            0x0200 /* Send an ABORT to peer */
#define SCTP_UNORDERED        0x0400 /* Message is un-ordered */
#define SCTP_ADDR_OVER        0x0800 /* Override the primary-address */
#define SCTP_SENDALL          0x1000 /* Send this on all associations */
#define SCTP_EOR              0x2000 /* end of message signal */
#define SCTP_SACK_IMMEDIATELY 0x4000 /* Set I-Bit */

#define INVALID_SINFO_FLAG(x) (((x) & 0xfffffff0 \
                                    & ~(SCTP_EOF | SCTP_ABORT | SCTP_UNORDERED |\
				        SCTP_ADDR_OVER | SCTP_SENDALL | SCTP_EOR |\
					SCTP_SACK_IMMEDIATELY)) != 0)
/* for the endpoint */

/* The lower byte is an enumeration of PR-SCTP policies */
#define SCTP_PR_SCTP_NONE 0x0000 /* Reliable transfer */
#define SCTP_PR_SCTP_TTL  0x0001 /* Time based PR-SCTP */
#define SCTP_PR_SCTP_BUF  0x0002 /* Buffer based PR-SCTP */
#define SCTP_PR_SCTP_RTX  0x0003 /* Number of retransmissions based PR-SCTP */

#define PR_SCTP_POLICY(x)         ((x) & 0x0f)
#define PR_SCTP_ENABLED(x)        (PR_SCTP_POLICY(x) != SCTP_PR_SCTP_NONE)
#define PR_SCTP_TTL_ENABLED(x)    (PR_SCTP_POLICY(x) == SCTP_PR_SCTP_TTL)
#define PR_SCTP_BUF_ENABLED(x)    (PR_SCTP_POLICY(x) == SCTP_PR_SCTP_BUF)
#define PR_SCTP_RTX_ENABLED(x)    (PR_SCTP_POLICY(x) == SCTP_PR_SCTP_RTX)
#define PR_SCTP_INVALID_POLICY(x) (PR_SCTP_POLICY(x) > SCTP_PR_SCTP_RTX)


/*
 * user socket options: socket API defined
 */
/*
 * read-write options
 */
#define SCTP_RTOINFO                    0x00000001
#define SCTP_ASSOCINFO                  0x00000002
#define SCTP_INITMSG                    0x00000003
#define SCTP_NODELAY                    0x00000004
#define SCTP_AUTOCLOSE                  0x00000005
#define SCTP_PRIMARY_ADDR               0x00000007
#define SCTP_ADAPTATION_LAYER           0x00000008
#define SCTP_DISABLE_FRAGMENTS          0x00000009
#define SCTP_PEER_ADDR_PARAMS           0x0000000a
/* ancillary data/notification interest options */
/* Without this applied we will give V4 and V6 addresses on a V6 socket */
#define SCTP_I_WANT_MAPPED_V4_ADDR      0x0000000d
#define SCTP_MAXSEG                     0x0000000e
#define SCTP_DELAYED_SACK               0x0000000f
#define SCTP_FRAGMENT_INTERLEAVE        0x00000010
#define SCTP_PARTIAL_DELIVERY_POINT     0x00000011
/* authentication support */
#define SCTP_HMAC_IDENT                 0x00000014
#define SCTP_AUTH_ACTIVE_KEY            0x00000015
#define SCTP_AUTO_ASCONF                0x00000018
#define SCTP_MAX_BURST                  0x00000019
/* assoc level context */
#define SCTP_CONTEXT                    0x0000001a
/* explicit EOR signalling */
#define SCTP_EXPLICIT_EOR               0x0000001b
#define SCTP_REUSE_PORT                 0x0000001c

#define SCTP_EVENT                      0x0000001e
#define SCTP_RECVRCVINFO                0x0000001f
#define SCTP_RECVNXTINFO                0x00000020
#define SCTP_DEFAULT_SNDINFO            0x00000021
#define SCTP_DEFAULT_PRINFO             0x00000022
#define SCTP_REMOTE_UDP_ENCAPS_PORT     0x00000024
#define SCTP_ECN_SUPPORTED              0x00000025
#define SCTP_PR_SUPPORTED               0x00000026
#define SCTP_AUTH_SUPPORTED             0x00000027
#define SCTP_ASCONF_SUPPORTED           0x00000028
#define SCTP_RECONFIG_SUPPORTED         0x00000029
#define SCTP_NRSACK_SUPPORTED           0x00000030
#define SCTP_PKTDROP_SUPPORTED          0x00000031
#define SCTP_MAX_CWND                   0x00000032
#define SCTP_ACCEPT_ZERO_CHECKSUM       0x00000033

#define SCTP_ENABLE_STREAM_RESET        0x00000900 /* struct sctp_assoc_value */

/* Pluggable Stream Scheduling Socket option */
#define SCTP_PLUGGABLE_SS               0x00001203
#define SCTP_SS_VALUE                   0x00001204

/*
 * read-only options
 */
#define SCTP_STATUS                     0x00000100
#define SCTP_GET_PEER_ADDR_INFO         0x00000101
/* authentication support */
#define SCTP_PEER_AUTH_CHUNKS           0x00000102
#define SCTP_LOCAL_AUTH_CHUNKS          0x00000103
#define SCTP_GET_ASSOC_NUMBER           0x00000104
#define SCTP_GET_ASSOC_ID_LIST          0x00000105
#define SCTP_TIMEOUTS                   0x00000106
#define SCTP_PR_STREAM_STATUS           0x00000107
#define SCTP_PR_ASSOC_STATUS            0x00000108

/*
 * write-only options
 */
#define SCTP_SET_PEER_PRIMARY_ADDR      0x00000006
#define SCTP_AUTH_CHUNK                 0x00000012
#define SCTP_AUTH_KEY                   0x00000013
#define SCTP_AUTH_DEACTIVATE_KEY        0x0000001d
#define SCTP_AUTH_DELETE_KEY            0x00000016
#define SCTP_RESET_STREAMS              0x00000901 /* struct sctp_reset_streams */
#define SCTP_RESET_ASSOC                0x00000902 /* sctp_assoc_t */
#define SCTP_ADD_STREAMS                0x00000903 /* struct sctp_add_streams */

struct sctp_initmsg {
	uint16_t sinit_num_ostreams;
	uint16_t sinit_max_instreams;
	uint16_t sinit_max_attempts;
	uint16_t sinit_max_init_timeo;
};

struct sctp_rtoinfo {
	sctp_assoc_t srto_assoc_id;
	uint32_t srto_initial;
	uint32_t srto_max;
	uint32_t srto_min;
};

struct sctp_assocparams {
	sctp_assoc_t sasoc_assoc_id;
	uint32_t sasoc_peer_rwnd;
	uint32_t sasoc_local_rwnd;
	uint32_t sasoc_cookie_life;
	uint16_t sasoc_asocmaxrxt;
	uint16_t sasoc_number_peer_destinations;
};

struct sctp_setprim {
	struct sockaddr_storage ssp_addr;
	sctp_assoc_t ssp_assoc_id;
	uint8_t ssp_padding[4];
};

struct sctp_setadaptation {
	uint32_t   ssb_adaptation_ind;
};

struct sctp_paddrparams {
	struct sockaddr_storage spp_address;
	sctp_assoc_t spp_assoc_id;
	uint32_t spp_hbinterval;
	uint32_t spp_pathmtu;
	uint32_t spp_flags;
	uint32_t spp_ipv6_flowlabel;
	uint16_t spp_pathmaxrxt;
	uint8_t spp_dscp;
};

#define SPP_HB_ENABLE       0x00000001
#define SPP_HB_DISABLE      0x00000002
#define SPP_HB_DEMAND       0x00000004
#define SPP_PMTUD_ENABLE    0x00000008
#define SPP_PMTUD_DISABLE   0x00000010
#define SPP_HB_TIME_IS_ZERO 0x00000080
#define SPP_IPV6_FLOWLABEL  0x00000100
#define SPP_DSCP            0x00000200

/* Used for SCTP_MAXSEG, SCTP_MAX_BURST, SCTP_ENABLE_STREAM_RESET, and SCTP_CONTEXT */
struct sctp_assoc_value {
	sctp_assoc_t assoc_id;
	uint32_t assoc_value;
};

/* To enable stream reset */
#define SCTP_ENABLE_RESET_STREAM_REQ  0x00000001
#define SCTP_ENABLE_RESET_ASSOC_REQ   0x00000002
#define SCTP_ENABLE_CHANGE_ASSOC_REQ  0x00000004
#define SCTP_ENABLE_VALUE_MASK        0x00000007

struct sctp_reset_streams {
	sctp_assoc_t srs_assoc_id;
	uint16_t srs_flags;
	uint16_t srs_number_streams;  /* 0 == ALL */
	uint16_t srs_stream_list[];   /* list if strrst_num_streams is not 0 */
};

struct sctp_add_streams {
	sctp_assoc_t	sas_assoc_id;
	uint16_t	sas_instrms;
	uint16_t	sas_outstrms;
};

struct sctp_hmacalgo {
	uint32_t shmac_number_of_idents;
	uint16_t shmac_idents[];
};

/* AUTH hmac_id */
#define SCTP_AUTH_HMAC_ID_RSVD    0x0000
#define SCTP_AUTH_HMAC_ID_SHA1    0x0001	/* default, mandatory */
#define SCTP_AUTH_HMAC_ID_SHA256  0x0003
#define SCTP_AUTH_HMAC_ID_SHA224  0x0004
#define SCTP_AUTH_HMAC_ID_SHA384  0x0005
#define SCTP_AUTH_HMAC_ID_SHA512  0x0006


struct sctp_sack_info {
	sctp_assoc_t sack_assoc_id;
	uint32_t sack_delay;
	uint32_t sack_freq;
};

struct sctp_default_prinfo {
	uint16_t pr_policy;
	uint32_t pr_value;
	sctp_assoc_t pr_assoc_id;
};

struct sctp_paddrinfo {
	struct sockaddr_storage spinfo_address;
	sctp_assoc_t spinfo_assoc_id;
	int32_t spinfo_state;
	uint32_t spinfo_cwnd;
	uint32_t spinfo_srtt;
	uint32_t spinfo_rto;
	uint32_t spinfo_mtu;
};

struct sctp_status {
	sctp_assoc_t sstat_assoc_id;
	int32_t  sstat_state;
	uint32_t sstat_rwnd;
	uint16_t sstat_unackdata;
	uint16_t sstat_penddata;
	uint16_t sstat_instrms;
	uint16_t sstat_outstrms;
	uint32_t sstat_fragmentation_point;
	struct sctp_paddrinfo sstat_primary;
};

/*
 * user state values
 */
#define SCTP_CLOSED             0x0000
#define SCTP_BOUND              0x1000
#define SCTP_LISTEN             0x2000
#define SCTP_COOKIE_WAIT        0x0002
#define SCTP_COOKIE_ECHOED      0x0004
#define SCTP_ESTABLISHED        0x0008
#define SCTP_SHUTDOWN_SENT      0x0010
#define SCTP_SHUTDOWN_RECEIVED  0x0020
#define SCTP_SHUTDOWN_ACK_SENT  0x0040
#define SCTP_SHUTDOWN_PENDING   0x0080


#define SCTP_ACTIVE       0x0001  /* SCTP_ADDR_REACHABLE */
#define SCTP_INACTIVE     0x0002  /* neither SCTP_ADDR_REACHABLE
                                     nor SCTP_ADDR_UNCONFIRMED */
#define SCTP_UNCONFIRMED  0x0200  /* SCTP_ADDR_UNCONFIRMED */

struct sctp_authchunks {
	sctp_assoc_t gauth_assoc_id;
/*	uint32_t gauth_number_of_chunks; not available */
	uint8_t  gauth_chunks[];
};

struct sctp_assoc_ids {
	uint32_t gaids_number_of_ids;
	sctp_assoc_t gaids_assoc_id[];
};

struct sctp_setpeerprim {
	struct sockaddr_storage sspp_addr;
	sctp_assoc_t sspp_assoc_id;
	uint8_t sspp_padding[4];
};

struct sctp_authchunk {
	uint8_t sauth_chunk;
};


struct sctp_get_nonce_values {
	sctp_assoc_t gn_assoc_id;
	uint32_t gn_peers_tag;
	uint32_t gn_local_tag;
};

/* Values for SCTP_ACCEPT_ZERO_CHECKSUM */
#define SCTP_EDMID_NONE             0
#define SCTP_EDMID_LOWER_LAYER_DTLS 1


/*
 * Main SCTP chunk types
 */
/************0x00 series ***********/
#define SCTP_DATA               0x00
#define SCTP_INITIATION         0x01
#define SCTP_INITIATION_ACK     0x02
#define SCTP_SELECTIVE_ACK      0x03
#define SCTP_HEARTBEAT_REQUEST  0x04
#define SCTP_HEARTBEAT_ACK      0x05
#define SCTP_ABORT_ASSOCIATION  0x06
#define SCTP_SHUTDOWN           0x07
#define SCTP_SHUTDOWN_ACK       0x08
#define SCTP_OPERATION_ERROR    0x09
#define SCTP_COOKIE_ECHO        0x0a
#define SCTP_COOKIE_ACK         0x0b
#define SCTP_ECN_ECHO           0x0c
#define SCTP_ECN_CWR            0x0d
#define SCTP_SHUTDOWN_COMPLETE  0x0e
/* RFC4895 */
#define SCTP_AUTHENTICATION     0x0f
/* EY nr_sack chunk id*/
#define SCTP_NR_SELECTIVE_ACK   0x10
/************0x40 series ***********/
/************0x80 series ***********/
/* RFC5061 */
#define	SCTP_ASCONF_ACK         0x80
/* draft-ietf-stewart-pktdrpsctp */
#define SCTP_PACKET_DROPPED     0x81
/* draft-ietf-stewart-strreset-xxx */
#define SCTP_STREAM_RESET       0x82

/* RFC4820                         */
#define SCTP_PAD_CHUNK          0x84
/************0xc0 series ***********/
/* RFC3758 */
#define SCTP_FORWARD_CUM_TSN    0xc0
/* RFC5061 */
#define SCTP_ASCONF             0xc1

struct sctp_authkey {
	sctp_assoc_t sca_assoc_id;
	uint16_t sca_keynumber;
	uint16_t sca_keylength;
	uint8_t  sca_key[];
};

struct sctp_authkeyid {
	sctp_assoc_t scact_assoc_id;
	uint16_t scact_keynumber;
};

struct sctp_cc_option {
	int option;
	struct sctp_assoc_value aid_value;
};

struct sctp_stream_value {
	sctp_assoc_t assoc_id;
	uint16_t stream_id;
	uint16_t stream_value;
};

struct sctp_timeouts {
	sctp_assoc_t stimo_assoc_id;
	uint32_t stimo_init;
	uint32_t stimo_data;
	uint32_t stimo_sack;
	uint32_t stimo_shutdown;
	uint32_t stimo_heartbeat;
	uint32_t stimo_cookie;
	uint32_t stimo_shutdownack;
};

struct sctp_prstatus {
	sctp_assoc_t sprstat_assoc_id;
	uint16_t sprstat_sid;
	uint16_t sprstat_policy;
	uint64_t sprstat_abandoned_unsent;
	uint64_t sprstat_abandoned_sent;
};

/* Standard TCP Congestion Control */
#define SCTP_CC_RFC2581         0x00000000
/* High Speed TCP Congestion Control (Floyd) */
#define SCTP_CC_HSTCP           0x00000001
/* HTCP Congestion Control */
#define SCTP_CC_HTCP            0x00000002
/* RTCC Congestion Control - RFC2581 plus */
#define SCTP_CC_RTCC            0x00000003

#define SCTP_CC_OPT_RTCC_SETMODE 0x00002000
#define SCTP_CC_OPT_USE_DCCC_EC  0x00002001
#define SCTP_CC_OPT_STEADY_STEP  0x00002002

#define SCTP_CMT_OFF            0
#define SCTP_CMT_BASE           1
#define SCTP_CMT_RPV1           2
#define SCTP_CMT_RPV2           3
#define SCTP_CMT_MPTCP          4
#define SCTP_CMT_MAX            SCTP_CMT_MPTCP

/* RS - Supported stream scheduling modules for pluggable
 * stream scheduling
 */
/* Default simple round-robin */
#define SCTP_SS_DEFAULT             0x00000000
/* Real round-robin */
#define SCTP_SS_ROUND_ROBIN         0x00000001
/* Real round-robin per packet */
#define SCTP_SS_ROUND_ROBIN_PACKET  0x00000002
/* Priority */
#define SCTP_SS_PRIORITY            0x00000003
/* Fair Bandwidth */
#define SCTP_SS_FAIR_BANDWITH       0x00000004
/* First-come, first-serve */
#define SCTP_SS_FIRST_COME          0x00000005

/******************** System calls *************/

struct socket;

void
usrsctp_init(uint16_t,
             int (*)(void *addr, void *buffer, size_t length, uint8_t tos, uint8_t set_df),
             void (*)(const char *format, ...));

void
usrsctp_init_nothreads(uint16_t,
		       int (*)(void *addr, void *buffer, size_t length, uint8_t tos, uint8_t set_df),
		       void (*)(const char *format, ...));

struct socket *
usrsctp_socket(int domain, int type, int protocol,
               int (*receive_cb)(struct socket *sock, union sctp_sockstore addr, void *data,
                                 size_t datalen, struct sctp_rcvinfo, int flags, void *ulp_info),
               int (*send_cb)(struct socket *sock, uint32_t sb_free, void *ulp_info),
               uint32_t sb_threshold,
               void *ulp_info);

int
usrsctp_setsockopt(struct socket *so,
                   int level,
                   int option_name,
                   const void *option_value,
                   socklen_t option_len);

int
usrsctp_getsockopt(struct socket *so,
                   int level,
                   int option_name,
                   void *option_value,
                   socklen_t *option_len);

int
usrsctp_opt_info(struct socket *so,
                 sctp_assoc_t id,
                 int opt,
                 void *arg,
                 socklen_t *size);

int
usrsctp_getpaddrs(struct socket *so,
                  sctp_assoc_t id,
                  struct sockaddr **raddrs);

void
usrsctp_freepaddrs(struct sockaddr *addrs);

int
usrsctp_getladdrs(struct socket *so,
                  sctp_assoc_t id,
                  struct sockaddr **raddrs);

void
usrsctp_freeladdrs(struct sockaddr *addrs);

ssize_t
usrsctp_sendv(struct socket *so,
              const void *data,
              size_t len,
              struct sockaddr *to,
              int addrcnt,
              void *info,
              socklen_t infolen,
              unsigned int infotype,
              int flags);

ssize_t
usrsctp_recvv(struct socket *so,
              void *dbuf,
              size_t len,
              struct sockaddr *from,
              socklen_t * fromlen,
              void *info,
              socklen_t *infolen,
              unsigned int *infotype,
              int *msg_flags);

int
usrsctp_bind(struct socket *so,
             struct sockaddr *name,
             socklen_t namelen);

#define SCTP_BINDX_ADD_ADDR 0x00008001
#define SCTP_BINDX_REM_ADDR 0x00008002

int
usrsctp_bindx(struct socket *so,
              struct sockaddr *addrs,
              int addrcnt,
              int flags);

int
usrsctp_listen(struct socket *so,
               int backlog);

struct socket *
usrsctp_accept(struct socket *so,
               struct sockaddr * aname,
               socklen_t * anamelen);

struct socket *
usrsctp_peeloff(struct socket *, sctp_assoc_t);

int
usrsctp_connect(struct socket *so,
                struct sockaddr *name,
                socklen_t namelen);

int
usrsctp_connectx(struct socket *so,
                 const struct sockaddr *addrs, int addrcnt,
                 sctp_assoc_t *id);

void
usrsctp_close(struct socket *so);

sctp_assoc_t
usrsctp_getassocid(struct socket *, struct sockaddr *);

int
usrsctp_finish(void);

int
usrsctp_shutdown(struct socket *so, int how);

void
usrsctp_conninput(void *, const void *, size_t, uint8_t);

int
usrsctp_set_non_blocking(struct socket *, int);

int
usrsctp_get_non_blocking(struct socket *);

void
usrsctp_register_address(void *);

void
usrsctp_deregister_address(void *);

int
usrsctp_set_ulpinfo(struct socket *, void *);

int
usrsctp_get_ulpinfo(struct socket *, void **);

int
usrsctp_set_upcall(struct socket *so,
                   void (*upcall)(struct socket *, void *, int),
                   void *arg);

int
usrsctp_get_events(struct socket *so);


void
usrsctp_handle_timers(uint32_t elapsed_milliseconds);

#define SCTP_DUMP_OUTBOUND 1
#define SCTP_DUMP_INBOUND  0

char *
usrsctp_dumppacket(const void *, size_t, int);

void
usrsctp_freedumpbuffer(char *);

void
usrsctp_enable_crc32c_offload(void);

void
usrsctp_disable_crc32c_offload(void);

uint32_t
usrsctp_crc32c(void *, size_t);

#define USRSCTP_TUNABLE_DECL(__field)               \
int usrsctp_tunable_set_ ## __field(uint32_t value);\
uint32_t usrsctp_sysctl_get_ ## __field(void);

USRSCTP_TUNABLE_DECL(sctp_hashtblsize)
USRSCTP_TUNABLE_DECL(sctp_pcbtblsize)
USRSCTP_TUNABLE_DECL(sctp_chunkscale)

#define USRSCTP_SYSCTL_DECL(__field)               \
int usrsctp_sysctl_set_ ## __field(uint32_t value);\
uint32_t usrsctp_sysctl_get_ ## __field(void);

USRSCTP_SYSCTL_DECL(sctp_sendspace)
USRSCTP_SYSCTL_DECL(sctp_recvspace)
USRSCTP_SYSCTL_DECL(sctp_auto_asconf)
USRSCTP_SYSCTL_DECL(sctp_multiple_asconfs)
USRSCTP_SYSCTL_DECL(sctp_ecn_enable)
USRSCTP_SYSCTL_DECL(sctp_pr_enable)
USRSCTP_SYSCTL_DECL(sctp_auth_enable)
USRSCTP_SYSCTL_DECL(sctp_asconf_enable)
USRSCTP_SYSCTL_DECL(sctp_reconfig_enable)
USRSCTP_SYSCTL_DECL(sctp_nrsack_enable)
USRSCTP_SYSCTL_DECL(sctp_pktdrop_enable)
USRSCTP_SYSCTL_DECL(sctp_no_csum_on_loopback)
USRSCTP_SYSCTL_DECL(sctp_peer_chunk_oh)
USRSCTP_SYSCTL_DECL(sctp_max_burst_default)
USRSCTP_SYSCTL_DECL(sctp_max_chunks_on_queue)
USRSCTP_SYSCTL_DECL(sctp_min_split_point)
USRSCTP_SYSCTL_DECL(sctp_delayed_sack_time_default)
USRSCTP_SYSCTL_DECL(sctp_sack_freq_default)
USRSCTP_SYSCTL_DECL(sctp_system_free_resc_limit)
USRSCTP_SYSCTL_DECL(sctp_asoc_free_resc_limit)
USRSCTP_SYSCTL_DECL(sctp_heartbeat_interval_default)
USRSCTP_SYSCTL_DECL(sctp_pmtu_raise_time_default)
USRSCTP_SYSCTL_DECL(sctp_shutdown_guard_time_default)
USRSCTP_SYSCTL_DECL(sctp_secret_lifetime_default)
USRSCTP_SYSCTL_DECL(sctp_rto_max_default)
USRSCTP_SYSCTL_DECL(sctp_rto_min_default)
USRSCTP_SYSCTL_DECL(sctp_rto_initial_default)
USRSCTP_SYSCTL_DECL(sctp_init_rto_max_default)
USRSCTP_SYSCTL_DECL(sctp_valid_cookie_life_default)
USRSCTP_SYSCTL_DECL(sctp_init_rtx_max_default)
USRSCTP_SYSCTL_DECL(sctp_assoc_rtx_max_default)
USRSCTP_SYSCTL_DECL(sctp_path_rtx_max_default)
USRSCTP_SYSCTL_DECL(sctp_add_more_threshold)
USRSCTP_SYSCTL_DECL(sctp_nr_incoming_streams_default)
USRSCTP_SYSCTL_DECL(sctp_nr_outgoing_streams_default)
USRSCTP_SYSCTL_DECL(sctp_cmt_on_off)
USRSCTP_SYSCTL_DECL(sctp_cmt_use_dac)
USRSCTP_SYSCTL_DECL(sctp_use_cwnd_based_maxburst)
USRSCTP_SYSCTL_DECL(sctp_nat_friendly)
USRSCTP_SYSCTL_DECL(sctp_L2_abc_variable)
USRSCTP_SYSCTL_DECL(sctp_mbuf_threshold_count)
USRSCTP_SYSCTL_DECL(sctp_do_drain)
USRSCTP_SYSCTL_DECL(sctp_hb_maxburst)
USRSCTP_SYSCTL_DECL(sctp_abort_if_one_2_one_hits_limit)
USRSCTP_SYSCTL_DECL(sctp_min_residual)
USRSCTP_SYSCTL_DECL(sctp_max_retran_chunk)
USRSCTP_SYSCTL_DECL(sctp_logging_level)
USRSCTP_SYSCTL_DECL(sctp_default_cc_module)
USRSCTP_SYSCTL_DECL(sctp_default_frag_interleave)
USRSCTP_SYSCTL_DECL(sctp_mobility_base)
USRSCTP_SYSCTL_DECL(sctp_mobility_fasthandoff)
USRSCTP_SYSCTL_DECL(sctp_inits_include_nat_friendly)
USRSCTP_SYSCTL_DECL(sctp_udp_tunneling_port)
USRSCTP_SYSCTL_DECL(sctp_enable_sack_immediately)
USRSCTP_SYSCTL_DECL(sctp_vtag_time_wait)
USRSCTP_SYSCTL_DECL(sctp_blackhole)
USRSCTP_SYSCTL_DECL(sctp_sendall_limit)
USRSCTP_SYSCTL_DECL(sctp_diag_info_code)
USRSCTP_SYSCTL_DECL(sctp_fr_max_burst_default)
USRSCTP_SYSCTL_DECL(sctp_path_pf_threshold)
USRSCTP_SYSCTL_DECL(sctp_default_ss_module)
USRSCTP_SYSCTL_DECL(sctp_rttvar_bw)
USRSCTP_SYSCTL_DECL(sctp_rttvar_rtt)
USRSCTP_SYSCTL_DECL(sctp_rttvar_eqret)
USRSCTP_SYSCTL_DECL(sctp_steady_step)
USRSCTP_SYSCTL_DECL(sctp_use_dccc_ecn)
USRSCTP_SYSCTL_DECL(sctp_buffer_splitting)
USRSCTP_SYSCTL_DECL(sctp_initial_cwnd)
USRSCTP_SYSCTL_DECL(sctp_ootb_with_zero_cksum)
#ifdef SCTP_DEBUG
USRSCTP_SYSCTL_DECL(sctp_debug_on)
/* More specific values can be found in sctp_constants, but
 * are not considered to be part of the API.
 */
#define SCTP_DEBUG_NONE 0x00000000
#define SCTP_DEBUG_ALL  0xffffffff
#endif
#undef USRSCTP_SYSCTL_DECL
struct sctp_timeval {
	uint32_t tv_sec;
	uint32_t tv_usec;
};

struct sctpstat {
	struct sctp_timeval sctps_discontinuitytime; /* sctpStats 18 (TimeStamp) */
	/* MIB according to RFC 3873 */
	uint32_t  sctps_currestab;           /* sctpStats  1   (Gauge32) */
	uint32_t  sctps_activeestab;         /* sctpStats  2 (Counter32) */
	uint32_t  sctps_restartestab;
	uint32_t  sctps_collisionestab;
	uint32_t  sctps_passiveestab;        /* sctpStats  3 (Counter32) */
	uint32_t  sctps_aborted;             /* sctpStats  4 (Counter32) */
	uint32_t  sctps_shutdown;            /* sctpStats  5 (Counter32) */
	uint32_t  sctps_outoftheblue;        /* sctpStats  6 (Counter32) */
	uint32_t  sctps_checksumerrors;      /* sctpStats  7 (Counter32) */
	uint32_t  sctps_outcontrolchunks;    /* sctpStats  8 (Counter64) */
	uint32_t  sctps_outorderchunks;      /* sctpStats  9 (Counter64) */
	uint32_t  sctps_outunorderchunks;    /* sctpStats 10 (Counter64) */
	uint32_t  sctps_incontrolchunks;     /* sctpStats 11 (Counter64) */
	uint32_t  sctps_inorderchunks;       /* sctpStats 12 (Counter64) */
	uint32_t  sctps_inunorderchunks;     /* sctpStats 13 (Counter64) */
	uint32_t  sctps_fragusrmsgs;         /* sctpStats 14 (Counter64) */
	uint32_t  sctps_reasmusrmsgs;        /* sctpStats 15 (Counter64) */
	uint32_t  sctps_outpackets;          /* sctpStats 16 (Counter64) */
	uint32_t  sctps_inpackets;           /* sctpStats 17 (Counter64) */

	/* input statistics: */
	uint32_t  sctps_recvpackets;         /* total input packets        */
	uint32_t  sctps_recvdatagrams;       /* total input datagrams      */
	uint32_t  sctps_recvpktwithdata;     /* total packets that had data */
	uint32_t  sctps_recvsacks;           /* total input SACK chunks    */
	uint32_t  sctps_recvdata;            /* total input DATA chunks    */
	uint32_t  sctps_recvdupdata;         /* total input duplicate DATA chunks */
	uint32_t  sctps_recvheartbeat;       /* total input HB chunks      */
	uint32_t  sctps_recvheartbeatack;    /* total input HB-ACK chunks  */
	uint32_t  sctps_recvecne;            /* total input ECNE chunks    */
	uint32_t  sctps_recvauth;            /* total input AUTH chunks    */
	uint32_t  sctps_recvauthmissing;     /* total input chunks missing AUTH */
	uint32_t  sctps_recvivalhmacid;      /* total number of invalid HMAC ids received */
	uint32_t  sctps_recvivalkeyid;       /* total number of invalid secret ids received */
	uint32_t  sctps_recvauthfailed;      /* total number of auth failed */
	uint32_t  sctps_recvexpress;         /* total fast path receives all one chunk */
	uint32_t  sctps_recvexpressm;        /* total fast path multi-part data */
	uint32_t  sctps_recv_spare;          /* formerly sctps_recvnocrc */
	uint32_t  sctps_recvswcrc;
	uint32_t  sctps_recvhwcrc;

	/* output statistics: */
	uint32_t  sctps_sendpackets;         /* total output packets       */
	uint32_t  sctps_sendsacks;           /* total output SACKs         */
	uint32_t  sctps_senddata;            /* total output DATA chunks   */
	uint32_t  sctps_sendretransdata;     /* total output retransmitted DATA chunks */
	uint32_t  sctps_sendfastretrans;     /* total output fast retransmitted DATA chunks */
	uint32_t  sctps_sendmultfastretrans; /* total FR's that happened more than once
	                                      * to same chunk (u-del multi-fr algo).
	                                      */
	uint32_t  sctps_sendheartbeat;       /* total output HB chunks     */
	uint32_t  sctps_sendecne;            /* total output ECNE chunks    */
	uint32_t  sctps_sendauth;            /* total output AUTH chunks FIXME   */
	uint32_t  sctps_senderrors;          /* ip_output error counter */
	uint32_t  sctps_send_spare;          /* formerly sctps_sendnocrc */
	uint32_t  sctps_sendswcrc;
	uint32_t  sctps_sendhwcrc;
	/* PCKDROPREP statistics: */
	uint32_t  sctps_pdrpfmbox;           /* Packet drop from middle box */
	uint32_t  sctps_pdrpfehos;           /* P-drop from end host */
	uint32_t  sctps_pdrpmbda;            /* P-drops with data */
	uint32_t  sctps_pdrpmbct;            /* P-drops, non-data, non-endhost */
	uint32_t  sctps_pdrpbwrpt;           /* P-drop, non-endhost, bandwidth rep only */
	uint32_t  sctps_pdrpcrupt;           /* P-drop, not enough for chunk header */
	uint32_t  sctps_pdrpnedat;           /* P-drop, not enough data to confirm */
	uint32_t  sctps_pdrppdbrk;           /* P-drop, where process_chunk_drop said break */
	uint32_t  sctps_pdrptsnnf;           /* P-drop, could not find TSN */
	uint32_t  sctps_pdrpdnfnd;           /* P-drop, attempt reverse TSN lookup */
	uint32_t  sctps_pdrpdiwnp;           /* P-drop, e-host confirms zero-rwnd */
	uint32_t  sctps_pdrpdizrw;           /* P-drop, midbox confirms no space */
	uint32_t  sctps_pdrpbadd;            /* P-drop, data did not match TSN */
	uint32_t  sctps_pdrpmark;            /* P-drop, TSN's marked for Fast Retran */
	/* timeouts */
	uint32_t  sctps_timoiterator;        /* Number of iterator timers that fired */
	uint32_t  sctps_timodata;            /* Number of T3 data time outs */
	uint32_t  sctps_timowindowprobe;     /* Number of window probe (T3) timers that fired */
	uint32_t  sctps_timoinit;            /* Number of INIT timers that fired */
	uint32_t  sctps_timosack;            /* Number of sack timers that fired */
	uint32_t  sctps_timoshutdown;        /* Number of shutdown timers that fired */
	uint32_t  sctps_timoheartbeat;       /* Number of heartbeat timers that fired */
	uint32_t  sctps_timocookie;          /* Number of times a cookie timeout fired */
	uint32_t  sctps_timosecret;          /* Number of times an endpoint changed its cookie secret*/
	uint32_t  sctps_timopathmtu;         /* Number of PMTU timers that fired */
	uint32_t  sctps_timoshutdownack;     /* Number of shutdown ack timers that fired */
	uint32_t  sctps_timoshutdownguard;   /* Number of shutdown guard timers that fired */
	uint32_t  sctps_timostrmrst;         /* Number of stream reset timers that fired */
	uint32_t  sctps_timoearlyfr;         /* Number of early FR timers that fired */
	uint32_t  sctps_timoasconf;          /* Number of times an asconf timer fired */
	uint32_t  sctps_timodelprim;	     /* Number of times a prim_deleted timer fired */
	uint32_t  sctps_timoautoclose;       /* Number of times auto close timer fired */
	uint32_t  sctps_timoassockill;       /* Number of asoc free timers expired */
	uint32_t  sctps_timoinpkill;         /* Number of inp free timers expired */
	/* former early FR counters */
	uint32_t  sctps_spare[11];
	/* others */
	uint32_t  sctps_hdrops;              /* packet shorter than header */
	uint32_t  sctps_badsum;              /* checksum error             */
	uint32_t  sctps_noport;              /* no endpoint for port       */
	uint32_t  sctps_badvtag;             /* bad v-tag                  */
	uint32_t  sctps_badsid;              /* bad SID                    */
	uint32_t  sctps_nomem;               /* no memory                  */
	uint32_t  sctps_fastretransinrtt;    /* number of multiple FR in a RTT window */
	uint32_t  sctps_markedretrans;
	uint32_t  sctps_naglesent;           /* nagle allowed sending      */
	uint32_t  sctps_naglequeued;         /* nagle doesn't allow sending */
	uint32_t  sctps_maxburstqueued;      /* max burst doesn't allow sending */
	uint32_t  sctps_ifnomemqueued;       /* look ahead tells us no memory in
	                                      * interface ring buffer OR we had a
	                                      * send error and are queuing one send.
	                                      */
	uint32_t  sctps_windowprobed;        /* total number of window probes sent */
	uint32_t  sctps_lowlevelerr;         /* total times an output error causes us
	                                      * to clamp down on next user send.
	                                      */
	uint32_t  sctps_lowlevelerrusr;      /* total times sctp_senderrors were caused from
	                                      * a user send from a user invoked send not
	                                      * a sack response
	                                      */
	uint32_t  sctps_datadropchklmt;      /* Number of in data drops due to chunk limit reached */
	uint32_t  sctps_datadroprwnd;        /* Number of in data drops due to rwnd limit reached */
	uint32_t  sctps_ecnereducedcwnd;     /* Number of times a ECN reduced the cwnd */
	uint32_t  sctps_vtagexpress;         /* Used express lookup via vtag */
	uint32_t  sctps_vtagbogus;           /* Collision in express lookup. */
	uint32_t  sctps_primary_randry;      /* Number of times the sender ran dry of user data on primary */
	uint32_t  sctps_cmt_randry;          /* Same for above */
	uint32_t  sctps_slowpath_sack;       /* Sacks the slow way */
	uint32_t  sctps_wu_sacks_sent;       /* Window Update only sacks sent */
	uint32_t  sctps_sends_with_flags;    /* number of sends with sinfo_flags !=0 */
	uint32_t  sctps_sends_with_unord;    /* number of unordered sends */
	uint32_t  sctps_sends_with_eof;      /* number of sends with EOF flag set */
	uint32_t  sctps_sends_with_abort;    /* number of sends with ABORT flag set */
	uint32_t  sctps_protocol_drain_calls;/* number of times protocol drain called */
	uint32_t  sctps_protocol_drains_done;/* number of times we did a protocol drain */
	uint32_t  sctps_read_peeks;          /* Number of times recv was called with peek */
	uint32_t  sctps_cached_chk;          /* Number of cached chunks used */
	uint32_t  sctps_cached_strmoq;       /* Number of cached stream oq's used */
	uint32_t  sctps_left_abandon;        /* Number of unread messages abandoned by close */
	uint32_t  sctps_send_burst_avoid;    /* Unused */
	uint32_t  sctps_send_cwnd_avoid;     /* Send cwnd full  avoidance, already max burst inflight to net */
	uint32_t  sctps_fwdtsn_map_over;     /* number of map array over-runs via fwd-tsn's */
	uint32_t  sctps_queue_upd_ecne;      /* Number of times we queued or updated an ECN chunk on send queue */
	uint32_t  sctps_recvzerocrc;         /* Number of accepted packets with zero CRC */
	uint32_t  sctps_sendzerocrc;         /* Number of packets sent with zero CRC */
	uint32_t  sctps_reserved[29];        /* Future ABI compat - remove int's from here when adding new */
};

void
usrsctp_get_stat(struct sctpstat *);

#ifdef _WIN32
#ifdef _MSC_VER
#pragma warning(default: 4200)
#endif
#endif
#ifdef  __cplusplus
}
#endif
#endif
