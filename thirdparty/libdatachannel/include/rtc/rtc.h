/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_C_API
#define RTC_C_API

#ifdef __cplusplus
extern "C" {
#endif

#ifdef RTC_STATIC
#define RTC_C_EXPORT
#else // dynamic library
#ifdef _WIN32
#ifdef RTC_EXPORTS
#define RTC_C_EXPORT __declspec(dllexport) // building the library
#else
#define RTC_C_EXPORT __declspec(dllimport) // using the library
#endif
#else // not WIN32
#define RTC_C_EXPORT
#endif
#endif

#ifdef _WIN32
#ifdef CAPI_STDCALL
#define RTC_API __stdcall
#else
#define RTC_API
#endif
#else // not WIN32
#define RTC_API
#endif

#ifndef RTC_ENABLE_WEBSOCKET
#define RTC_ENABLE_WEBSOCKET 1
#endif

#ifndef RTC_ENABLE_MEDIA
#define RTC_ENABLE_MEDIA 1
#endif

#define RTC_DEFAULT_MTU 1280 // IPv6 minimum guaranteed MTU

#if RTC_ENABLE_MEDIA
#define RTC_DEFAULT_MAXIMUM_FRAGMENT_SIZE                                                          \
	((uint16_t)(RTC_DEFAULT_MTU - 12 - 8 - 40)) // SRTP/UDP/IPv6
#define RTC_DEFAULT_MAXIMUM_PACKET_COUNT_FOR_NACK_CACHE ((unsigned)512)
#endif

#include <stdbool.h>
#include <stdint.h>

// libdatachannel C API

typedef enum {
	RTC_NEW = 0,
	RTC_CONNECTING = 1,
	RTC_CONNECTED = 2,
	RTC_DISCONNECTED = 3,
	RTC_FAILED = 4,
	RTC_CLOSED = 5
} rtcState;

typedef enum {
	RTC_ICE_NEW = 0,
	RTC_ICE_CHECKING = 1,
	RTC_ICE_CONNECTED = 2,
	RTC_ICE_COMPLETED = 3,
	RTC_ICE_FAILED = 4,
	RTC_ICE_DISCONNECTED = 5,
	RTC_ICE_CLOSED = 6
} rtcIceState;

typedef enum {
	RTC_GATHERING_NEW = 0,
	RTC_GATHERING_INPROGRESS = 1,
	RTC_GATHERING_COMPLETE = 2
} rtcGatheringState;

typedef enum {
	RTC_SIGNALING_STABLE = 0,
	RTC_SIGNALING_HAVE_LOCAL_OFFER = 1,
	RTC_SIGNALING_HAVE_REMOTE_OFFER = 2,
	RTC_SIGNALING_HAVE_LOCAL_PRANSWER = 3,
	RTC_SIGNALING_HAVE_REMOTE_PRANSWER = 4,
} rtcSignalingState;

typedef enum { // Don't change, it must match plog severity
	RTC_LOG_NONE = 0,
	RTC_LOG_FATAL = 1,
	RTC_LOG_ERROR = 2,
	RTC_LOG_WARNING = 3,
	RTC_LOG_INFO = 4,
	RTC_LOG_DEBUG = 5,
	RTC_LOG_VERBOSE = 6
} rtcLogLevel;

typedef enum {
	RTC_CERTIFICATE_DEFAULT = 0, // ECDSA
	RTC_CERTIFICATE_ECDSA = 1,
	RTC_CERTIFICATE_RSA = 2,
} rtcCertificateType;

typedef enum {
	// video
	RTC_CODEC_H264 = 0,
	RTC_CODEC_VP8 = 1,
	RTC_CODEC_VP9 = 2,
	RTC_CODEC_H265 = 3,

	// audio
	RTC_CODEC_OPUS = 128,
	RTC_CODEC_PCMU = 129,
	RTC_CODEC_PCMA = 130,
	RTC_CODEC_AAC = 131,
} rtcCodec;

typedef enum {
	RTC_DIRECTION_UNKNOWN = 0,
	RTC_DIRECTION_SENDONLY = 1,
	RTC_DIRECTION_RECVONLY = 2,
	RTC_DIRECTION_SENDRECV = 3,
	RTC_DIRECTION_INACTIVE = 4
} rtcDirection;

typedef enum { RTC_TRANSPORT_POLICY_ALL = 0, RTC_TRANSPORT_POLICY_RELAY = 1 } rtcTransportPolicy;

#define RTC_ERR_SUCCESS 0
#define RTC_ERR_INVALID -1   // invalid argument
#define RTC_ERR_FAILURE -2   // runtime error
#define RTC_ERR_NOT_AVAIL -3 // element not available
#define RTC_ERR_TOO_SMALL -4 // buffer too small

typedef void(RTC_API *rtcLogCallbackFunc)(rtcLogLevel level, const char *message);
typedef void(RTC_API *rtcDescriptionCallbackFunc)(int pc, const char *sdp, const char *type,
                                                  void *ptr);
typedef void(RTC_API *rtcCandidateCallbackFunc)(int pc, const char *cand, const char *mid,
                                                void *ptr);
typedef void(RTC_API *rtcStateChangeCallbackFunc)(int pc, rtcState state, void *ptr);
typedef void(RTC_API *rtcIceStateChangeCallbackFunc)(int pc, rtcIceState state, void *ptr);
typedef void(RTC_API *rtcGatheringStateCallbackFunc)(int pc, rtcGatheringState state, void *ptr);
typedef void(RTC_API *rtcSignalingStateCallbackFunc)(int pc, rtcSignalingState state, void *ptr);
typedef void(RTC_API *rtcDataChannelCallbackFunc)(int pc, int dc, void *ptr);
typedef void(RTC_API *rtcTrackCallbackFunc)(int pc, int tr, void *ptr);
typedef void(RTC_API *rtcOpenCallbackFunc)(int id, void *ptr);
typedef void(RTC_API *rtcClosedCallbackFunc)(int id, void *ptr);
typedef void(RTC_API *rtcErrorCallbackFunc)(int id, const char *error, void *ptr);
typedef void(RTC_API *rtcMessageCallbackFunc)(int id, const char *message, int size, void *ptr);
typedef void *(RTC_API *rtcInterceptorCallbackFunc)(int pc, const char *message, int size,
                                                    void *ptr);
typedef void(RTC_API *rtcBufferedAmountLowCallbackFunc)(int id, void *ptr);
typedef void(RTC_API *rtcAvailableCallbackFunc)(int id, void *ptr);

// Log

// NULL cb on the first call will log to stdout
RTC_C_EXPORT void rtcInitLogger(rtcLogLevel level, rtcLogCallbackFunc cb);

// User pointer
RTC_C_EXPORT void rtcSetUserPointer(int id, void *ptr);
RTC_C_EXPORT void *rtcGetUserPointer(int i);

// PeerConnection

typedef struct {
	const char **iceServers;
	int iceServersCount;
	const char *proxyServer; // libnice only
	const char *bindAddress; // libjuice only, NULL means any
	rtcCertificateType certificateType;
	rtcTransportPolicy iceTransportPolicy;
	bool enableIceTcp;    // libnice only
	bool enableIceUdpMux; // libjuice only
	bool disableAutoNegotiation;
	bool forceMediaTransport;
	uint16_t portRangeBegin; // 0 means automatic
	uint16_t portRangeEnd;   // 0 means automatic
	int mtu;                 // <= 0 means automatic
	int maxMessageSize;      // <= 0 means default
} rtcConfiguration;

RTC_C_EXPORT int rtcCreatePeerConnection(const rtcConfiguration *config); // returns pc id
RTC_C_EXPORT int rtcClosePeerConnection(int pc);
RTC_C_EXPORT int rtcDeletePeerConnection(int pc);

RTC_C_EXPORT int rtcSetLocalDescriptionCallback(int pc, rtcDescriptionCallbackFunc cb);
RTC_C_EXPORT int rtcSetLocalCandidateCallback(int pc, rtcCandidateCallbackFunc cb);
RTC_C_EXPORT int rtcSetStateChangeCallback(int pc, rtcStateChangeCallbackFunc cb);
RTC_C_EXPORT int rtcSetIceStateChangeCallback(int pc, rtcIceStateChangeCallbackFunc cb);
RTC_C_EXPORT int rtcSetGatheringStateChangeCallback(int pc, rtcGatheringStateCallbackFunc cb);
RTC_C_EXPORT int rtcSetSignalingStateChangeCallback(int pc, rtcSignalingStateCallbackFunc cb);

RTC_C_EXPORT int rtcSetLocalDescription(int pc, const char *type);
RTC_C_EXPORT int rtcSetRemoteDescription(int pc, const char *sdp, const char *type);
RTC_C_EXPORT int rtcAddRemoteCandidate(int pc, const char *cand, const char *mid);

RTC_C_EXPORT int rtcGetLocalDescription(int pc, char *buffer, int size);
RTC_C_EXPORT int rtcGetRemoteDescription(int pc, char *buffer, int size);

RTC_C_EXPORT int rtcGetLocalDescriptionType(int pc, char *buffer, int size);
RTC_C_EXPORT int rtcGetRemoteDescriptionType(int pc, char *buffer, int size);

RTC_C_EXPORT int rtcGetLocalAddress(int pc, char *buffer, int size);
RTC_C_EXPORT int rtcGetRemoteAddress(int pc, char *buffer, int size);

RTC_C_EXPORT int rtcGetSelectedCandidatePair(int pc, char *local, int localSize, char *remote,
                                             int remoteSize);

RTC_C_EXPORT int rtcGetMaxDataChannelStream(int pc);

// DataChannel, Track, and WebSocket common API

RTC_C_EXPORT int rtcSetOpenCallback(int id, rtcOpenCallbackFunc cb);
RTC_C_EXPORT int rtcSetClosedCallback(int id, rtcClosedCallbackFunc cb);
RTC_C_EXPORT int rtcSetErrorCallback(int id, rtcErrorCallbackFunc cb);
RTC_C_EXPORT int rtcSetMessageCallback(int id, rtcMessageCallbackFunc cb);
RTC_C_EXPORT int rtcSendMessage(int id, const char *data, int size);
RTC_C_EXPORT int rtcClose(int id);
RTC_C_EXPORT int rtcDelete(int id);
RTC_C_EXPORT bool rtcIsOpen(int id);
RTC_C_EXPORT bool rtcIsClosed(int id);

RTC_C_EXPORT int rtcGetBufferedAmount(int id); // total size buffered to send
RTC_C_EXPORT int rtcSetBufferedAmountLowThreshold(int id, int amount);
RTC_C_EXPORT int rtcSetBufferedAmountLowCallback(int id, rtcBufferedAmountLowCallbackFunc cb);

// DataChannel, Track, and WebSocket common extended API

RTC_C_EXPORT int rtcGetAvailableAmount(int id); // total size available to receive
RTC_C_EXPORT int rtcSetAvailableCallback(int id, rtcAvailableCallbackFunc cb);
RTC_C_EXPORT int rtcReceiveMessage(int id, char *buffer, int *size);

// DataChannel

typedef struct {
	bool unordered;
	bool unreliable;
	int maxPacketLifeTime; // ignored if reliable
	int maxRetransmits;    // ignored if reliable
} rtcReliability;

typedef struct {
	rtcReliability reliability;
	const char *protocol; // empty string if NULL
	bool negotiated;
	bool manualStream;
	uint16_t stream; // numeric ID 0-65534, ignored if manualStream is false
} rtcDataChannelInit;

RTC_C_EXPORT int rtcSetDataChannelCallback(int pc, rtcDataChannelCallbackFunc cb);
RTC_C_EXPORT int rtcCreateDataChannel(int pc, const char *label); // returns dc id
RTC_C_EXPORT int rtcCreateDataChannelEx(int pc, const char *label,
                                        const rtcDataChannelInit *init); // returns dc id
RTC_C_EXPORT int rtcDeleteDataChannel(int dc);

RTC_C_EXPORT int rtcGetDataChannelStream(int dc);
RTC_C_EXPORT int rtcGetDataChannelLabel(int dc, char *buffer, int size);
RTC_C_EXPORT int rtcGetDataChannelProtocol(int dc, char *buffer, int size);
RTC_C_EXPORT int rtcGetDataChannelReliability(int dc, rtcReliability *reliability);

// Track

typedef struct {
	rtcDirection direction;
	rtcCodec codec;
	int payloadType;
	uint32_t ssrc;
	const char *mid;
	const char *name;    // optional
	const char *msid;    // optional
	const char *trackId; // optional, track ID used in MSID
	const char *profile; // optional, codec profile
} rtcTrackInit;

RTC_C_EXPORT int rtcSetTrackCallback(int pc, rtcTrackCallbackFunc cb);
RTC_C_EXPORT int rtcAddTrack(int pc, const char *mediaDescriptionSdp); // returns tr id
RTC_C_EXPORT int rtcAddTrackEx(int pc, const rtcTrackInit *init);      // returns tr id
RTC_C_EXPORT int rtcDeleteTrack(int tr);

RTC_C_EXPORT int rtcGetTrackDescription(int tr, char *buffer, int size);
RTC_C_EXPORT int rtcGetTrackMid(int tr, char *buffer, int size);
RTC_C_EXPORT int rtcGetTrackDirection(int tr, rtcDirection *direction);

#if RTC_ENABLE_MEDIA

// Media

// Define how OBUs are packetizied in a AV1 Sample
typedef enum {
	RTC_OBU_PACKETIZED_OBU = 0,
	RTC_OBU_PACKETIZED_TEMPORAL_UNIT = 1,
} rtcObuPacketization;

// Define how NAL units are separated in a H264/H265 sample
typedef enum {
	RTC_NAL_SEPARATOR_LENGTH = 0,               // first 4 bytes are NAL unit length
	RTC_NAL_SEPARATOR_LONG_START_SEQUENCE = 1,  // 0x00, 0x00, 0x00, 0x01
	RTC_NAL_SEPARATOR_SHORT_START_SEQUENCE = 2, // 0x00, 0x00, 0x01
	RTC_NAL_SEPARATOR_START_SEQUENCE = 3,       // long or short start sequence
} rtcNalUnitSeparator;

typedef struct {
	uint32_t ssrc;
	const char *cname;
	uint8_t payloadType;
	uint32_t clockRate;
	uint16_t sequenceNumber;
	uint32_t timestamp;

	// H264/H265
	rtcNalUnitSeparator nalSeparator; // NAL unit separator
	uint16_t maxFragmentSize;         // Maximum NAL unit fragment size

} rtcPacketizationHandlerInit;

typedef struct {
	uint32_t ssrc;
	const char *name;    // optional
	const char *msid;    // optional
	const char *trackId; // optional, track ID used in MSID
} rtcSsrcForTypeInit;

// Opaque message

// Opaque type used (via rtcMessage*) to reference an rtc::Message
typedef void *rtcMessage;

// Allocate a new opaque message.
// Must be explicitly freed by rtcDeleteOpaqueMessage() unless
// explicitly returned by a media interceptor callback;
RTC_C_EXPORT rtcMessage *rtcCreateOpaqueMessage(void *data, int size);
RTC_C_EXPORT void rtcDeleteOpaqueMessage(rtcMessage *msg);

// Set MediaInterceptor for peer connection
RTC_C_EXPORT int rtcSetMediaInterceptorCallback(int id, rtcInterceptorCallbackFunc cb);

// Set H264PacketizationHandler for track
RTC_C_EXPORT int rtcSetH264PacketizationHandler(int tr, const rtcPacketizationHandlerInit *init);

// Set H265PacketizationHandler for track
RTC_C_EXPORT int rtcSetH265PacketizationHandler(int tr, const rtcPacketizationHandlerInit *init);

// Set OpusPacketizationHandler for track
RTC_C_EXPORT int rtcSetOpusPacketizationHandler(int tr, const rtcPacketizationHandlerInit *init);

// Set AACPacketizationHandler for track
RTC_C_EXPORT int rtcSetAACPacketizationHandler(int tr, const rtcPacketizationHandlerInit *init);

// Chain RtcpSrReporter to handler chain for given track
RTC_C_EXPORT int rtcChainRtcpSrReporter(int tr);

// Chain RtcpNackResponder to handler chain for given track
RTC_C_EXPORT int rtcChainRtcpNackResponder(int tr, unsigned int maxStoredPacketsCount);

// Transform seconds to timestamp using track's clock rate, result is written to timestamp
RTC_C_EXPORT int rtcTransformSecondsToTimestamp(int id, double seconds, uint32_t *timestamp);

// Transform timestamp to seconds using track's clock rate, result is written to seconds
RTC_C_EXPORT int rtcTransformTimestampToSeconds(int id, uint32_t timestamp, double *seconds);

// Get current timestamp, result is written to timestamp
RTC_C_EXPORT int rtcGetCurrentTrackTimestamp(int id, uint32_t *timestamp);

// Set RTP timestamp for track identified by given id
RTC_C_EXPORT int rtcSetTrackRtpTimestamp(int id, uint32_t timestamp);

// Get timestamp of last RTCP SR, result is written to timestamp
RTC_C_EXPORT int rtcGetLastTrackSenderReportTimestamp(int id, uint32_t *timestamp);

// Set NeedsToReport flag in RtcpSrReporter handler identified by given track id
RTC_C_EXPORT int rtcSetNeedsToSendRtcpSr(int id);

// Get all available payload types for given codec and stores them in buffer, does nothing if
// buffer is NULL
int rtcGetTrackPayloadTypesForCodec(int tr, const char *ccodec, int *buffer, int size);

// Get all SSRCs for given track
int rtcGetSsrcsForTrack(int tr, uint32_t *buffer, int count);

// Get CName for SSRC
int rtcGetCNameForSsrc(int tr, uint32_t ssrc, char *cname, int cnameSize);

// Get all SSRCs for given media type in given SDP
int rtcGetSsrcsForType(const char *mediaType, const char *sdp, uint32_t *buffer, int bufferSize);

// Set SSRC for given media type in given SDP
int rtcSetSsrcForType(const char *mediaType, const char *sdp, char *buffer, const int bufferSize,
                      rtcSsrcForTypeInit *init);

#endif // RTC_ENABLE_MEDIA

#if RTC_ENABLE_WEBSOCKET

// WebSocket

typedef struct {
	bool disableTlsVerification; // if true, don't verify the TLS certificate
	const char *proxyServer;     // only non-authenticated http supported for now
	const char **protocols;
	int protocolsCount;
	int connectionTimeoutMs; // in milliseconds, 0 means default, < 0 means disabled
	int pingIntervalMs;      // in milliseconds, 0 means default, < 0 means disabled
	int maxOutstandingPings; // 0 means default, < 0 means disabled
} rtcWsConfiguration;

RTC_C_EXPORT int rtcCreateWebSocket(const char *url); // returns ws id
RTC_C_EXPORT int rtcCreateWebSocketEx(const char *url, const rtcWsConfiguration *config);
RTC_C_EXPORT int rtcDeleteWebSocket(int ws);

RTC_C_EXPORT int rtcGetWebSocketRemoteAddress(int ws, char *buffer, int size);
RTC_C_EXPORT int rtcGetWebSocketPath(int ws, char *buffer, int size);

// WebSocketServer

typedef void(RTC_API *rtcWebSocketClientCallbackFunc)(int wsserver, int ws, void *ptr);

typedef struct {
	uint16_t port;                  // 0 means automatic selection
	bool enableTls;                 // if true, enable TLS (WSS)
	const char *certificatePemFile; // NULL for autogenerated certificate
	const char *keyPemFile;         // NULL for autogenerated certificate
	const char *keyPemPass;         // NULL if no pass
	const char *bindAddress;        // NULL for IP_ANY_ADDR
	int connectionTimeoutMs;        // in milliseconds, 0 means default, < 0 means disabled
} rtcWsServerConfiguration;

RTC_C_EXPORT int rtcCreateWebSocketServer(const rtcWsServerConfiguration *config,
                                          rtcWebSocketClientCallbackFunc cb); // returns wsserver id
RTC_C_EXPORT int rtcDeleteWebSocketServer(int wsserver);

RTC_C_EXPORT int rtcGetWebSocketServerPort(int wsserver);

#endif

// Optional global preload and cleanup

RTC_C_EXPORT void rtcPreload(void);
RTC_C_EXPORT void rtcCleanup(void);

// SCTP global settings

typedef struct {
	int recvBufferSize;          // in bytes, <= 0 means optimized default
	int sendBufferSize;          // in bytes, <= 0 means optimized default
	int maxChunksOnQueue;        // in chunks, <= 0 means optimized default
	int initialCongestionWindow; // in MTUs, <= 0 means optimized default
	int maxBurst;                // in MTUs, 0 means optimized default, < 0 means disabled
	int congestionControlModule; // 0: RFC2581 (default), 1: HSTCP, 2: H-TCP, 3: RTCC
	int delayedSackTimeMs;       // in milliseconds, 0 means optimized default, < 0 means disabled
	int minRetransmitTimeoutMs;  // in milliseconds, <= 0 means optimized default
	int maxRetransmitTimeoutMs;  // in milliseconds, <= 0 means optimized default
	int initialRetransmitTimeoutMs; // in milliseconds, <= 0 means optimized default
	int maxRetransmitAttempts;      // number of retransmissions, <= 0 means optimized default
	int heartbeatIntervalMs;        // in milliseconds, <= 0 means optimized default
} rtcSctpSettings;

// Note: SCTP settings apply to newly-created PeerConnections only
RTC_C_EXPORT int rtcSetSctpSettings(const rtcSctpSettings *settings);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
