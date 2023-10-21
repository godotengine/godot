/**
 * Copyright (c) 2020 Staz Modrzynski
 * Copyright (c) 2020 Paul-Louis Ageneau
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_RTP_HPP
#define RTC_RTP_HPP

#include "common.hpp"

#include <vector>

namespace rtc {

typedef uint32_t SSRC;

RTC_CPP_EXPORT bool IsRtcp(const binary &data);

#pragma pack(push, 1)

struct RTC_CPP_EXPORT RtpExtensionHeader {
	uint16_t _profileSpecificId;
	uint16_t _headerLength;

	[[nodiscard]] uint16_t profileSpecificId() const;
	[[nodiscard]] uint16_t headerLength() const;

	[[nodiscard]] size_t getSize() const;
	[[nodiscard]] const char *getBody() const;
	[[nodiscard]] char *getBody();

	void setProfileSpecificId(uint16_t profileSpecificId);
	void setHeaderLength(uint16_t headerLength);

	void clearBody();
	void writeCurrentVideoOrientation(size_t offset, uint8_t id, uint8_t value);
	void writeOneByteHeader(size_t offset, uint8_t id, const byte *value, size_t size);
};

struct RTC_CPP_EXPORT RtpHeader {
	uint8_t _first;
	uint8_t _payloadType;
	uint16_t _seqNumber;
	uint32_t _timestamp;
	SSRC _ssrc;
	// The following field is SSRC _csrc[]

	[[nodiscard]] uint8_t version() const;
	[[nodiscard]] bool padding() const;
	[[nodiscard]] bool extension() const;
	[[nodiscard]] uint8_t csrcCount() const;
	[[nodiscard]] uint8_t marker() const;
	[[nodiscard]] uint8_t payloadType() const;
	[[nodiscard]] uint16_t seqNumber() const;
	[[nodiscard]] uint32_t timestamp() const;
	[[nodiscard]] uint32_t ssrc() const;

	[[nodiscard]] size_t getSize() const;
	[[nodiscard]] size_t getExtensionHeaderSize() const;
	[[nodiscard]] const RtpExtensionHeader *getExtensionHeader() const;
	[[nodiscard]] RtpExtensionHeader *getExtensionHeader();
	[[nodiscard]] const char *getBody() const;
	[[nodiscard]] char *getBody();

	void log() const;

	void preparePacket();
	void setSeqNumber(uint16_t newSeqNo);
	void setPayloadType(uint8_t newPayloadType);
	void setSsrc(uint32_t in_ssrc);
	void setMarker(bool marker);
	void setTimestamp(uint32_t i);
	void setExtension(bool extension);
};

struct RTC_CPP_EXPORT RtcpReportBlock {
	SSRC _ssrc;
	uint32_t _fractionLostAndPacketsLost; // fraction lost is 8-bit, packets lost is 24-bit
	uint16_t _seqNoCycles;
	uint16_t _highestSeqNo;
	uint32_t _jitter;
	uint32_t _lastReport;
	uint32_t _delaySinceLastReport;

	[[nodiscard]] uint16_t seqNoCycles() const;
	[[nodiscard]] uint16_t highestSeqNo() const;
	[[nodiscard]] uint32_t jitter() const;
	[[nodiscard]] uint32_t delaySinceSR() const;

	[[nodiscard]] SSRC getSSRC() const;
	[[nodiscard]] uint32_t getNTPOfSR() const;
	[[nodiscard]] unsigned int getLossPercentage() const;
	[[nodiscard]] unsigned int getPacketLostCount() const;

	void preparePacket(SSRC in_ssrc, unsigned int packetsLost, unsigned int totalPackets,
	                   uint16_t highestSeqNo, uint16_t seqNoCycles, uint32_t jitter,
	                   uint64_t lastSR_NTP, uint64_t lastSR_DELAY);
	void setSSRC(SSRC in_ssrc);
	void setPacketsLost(unsigned int packetsLost, unsigned int totalPackets);
	void setSeqNo(uint16_t highestSeqNo, uint16_t seqNoCycles);
	void setJitter(uint32_t jitter);
	void setNTPOfSR(uint64_t ntp);
	void setDelaySinceSR(uint32_t sr);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpHeader {
	uint8_t _first;
	uint8_t _payloadType;
	uint16_t _length;

	[[nodiscard]] uint8_t version() const;
	[[nodiscard]] bool padding() const;
	[[nodiscard]] uint8_t reportCount() const;
	[[nodiscard]] uint8_t payloadType() const;
	[[nodiscard]] uint16_t length() const;
	[[nodiscard]] size_t lengthInBytes() const;

	void prepareHeader(uint8_t payloadType, uint8_t reportCount, uint16_t length);
	void setPayloadType(uint8_t type);
	void setReportCount(uint8_t count);
	void setLength(uint16_t length);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpFbHeader {
	RtcpHeader header;

	SSRC _packetSender;
	SSRC _mediaSource;

	[[nodiscard]] SSRC packetSenderSSRC() const;
	[[nodiscard]] SSRC mediaSourceSSRC() const;

	void setPacketSenderSSRC(SSRC ssrc);
	void setMediaSourceSSRC(SSRC ssrc);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpSr {
	RtcpHeader header;

	SSRC _senderSSRC;
	uint64_t _ntpTimestamp;
	uint32_t _rtpTimestamp;
	uint32_t _packetCount;
	uint32_t _octetCount;

	RtcpReportBlock _reportBlocks;

	[[nodiscard]] static unsigned int Size(unsigned int reportCount);

	[[nodiscard]] uint64_t ntpTimestamp() const;
	[[nodiscard]] uint32_t rtpTimestamp() const;
	[[nodiscard]] uint32_t packetCount() const;
	[[nodiscard]] uint32_t octetCount() const;
	[[nodiscard]] uint32_t senderSSRC() const;

	[[nodiscard]] const RtcpReportBlock *getReportBlock(int num) const;
	[[nodiscard]] RtcpReportBlock *getReportBlock(int num);
	[[nodiscard]] unsigned int size(unsigned int reportCount);
	[[nodiscard]] size_t getSize() const;

	void preparePacket(SSRC senderSSRC, uint8_t reportCount);
	void setNtpTimestamp(uint64_t ts);
	void setRtpTimestamp(uint32_t ts);
	void setOctetCount(uint32_t ts);
	void setPacketCount(uint32_t ts);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpSdesItem {
	uint8_t type;

	uint8_t _length;
	char _text[1];

	[[nodiscard]] static unsigned int Size(uint8_t textLength);

	[[nodiscard]] string text() const;
	[[nodiscard]] uint8_t length() const;

	void setText(string text);
};

struct RTC_CPP_EXPORT RtcpSdesChunk {
	SSRC _ssrc;
	RtcpSdesItem _items;

	[[nodiscard]] static unsigned int Size(const std::vector<uint8_t> textLengths);

	[[nodiscard]] SSRC ssrc() const;

	void setSSRC(SSRC ssrc);

	// Get item at given index
	// All items with index < num must be valid, otherwise this function has undefined behaviour
	// (use safelyCountChunkSize() to check if chunk is valid).
	[[nodiscard]] const RtcpSdesItem *getItem(int num) const;
	[[nodiscard]] RtcpSdesItem *getItem(int num);

	// Get size of chunk
	// All items must be valid, otherwise this function has undefined behaviour (use
	// safelyCountChunkSize() to check if chunk is valid)
	[[nodiscard]] unsigned int getSize() const;

	long safelyCountChunkSize(size_t maxChunkSize) const;
};

struct RTC_CPP_EXPORT RtcpSdes {
	RtcpHeader header;
	RtcpSdesChunk _chunks;

	[[nodiscard]] static unsigned int Size(const std::vector<std::vector<uint8_t>> lengths);

	bool isValid() const;

	// Returns number of chunks in this packet
	// Returns 0 if packet is invalid
	unsigned int chunksCount() const;

	// Get chunk at given index
	// All chunks (and their items) with index < `num` must be valid, otherwise this function has
	// undefined behaviour (use `isValid` to check if chunk is valid).
	const RtcpSdesChunk *getChunk(int num) const;
	RtcpSdesChunk *getChunk(int num);

	void preparePacket(uint8_t chunkCount);
};

struct RTC_CPP_EXPORT RtcpRr {
	RtcpHeader header;

	SSRC _senderSSRC;
	RtcpReportBlock _reportBlocks;

	[[nodiscard]] static size_t SizeWithReportBlocks(uint8_t reportCount);

	SSRC senderSSRC() const;
	bool isSenderReport();
	bool isReceiverReport();

	[[nodiscard]] RtcpReportBlock *getReportBlock(int num);
	[[nodiscard]] const RtcpReportBlock *getReportBlock(int num) const;
	[[nodiscard]] size_t getSize() const;

	void preparePacket(SSRC senderSSRC, uint8_t reportCount);
	void setSenderSSRC(SSRC ssrc);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpRemb {
	RtcpFbHeader header;

	char _id[4];       // Unique identifier ('R' 'E' 'M' 'B')
	uint32_t _bitrate; // Num SSRC, Br Exp, Br Mantissa (bit mask)
	SSRC _ssrc[1];

	[[nodiscard]] static size_t SizeWithSSRCs(int count);

	[[nodiscard]] unsigned int getSize() const;

	void preparePacket(SSRC senderSSRC, unsigned int numSSRC, unsigned int in_bitrate);
	void setBitrate(unsigned int numSSRC, unsigned int in_bitrate);
	void setSsrc(int iterator, SSRC newSssrc);
};

struct RTC_CPP_EXPORT RtcpPli {
	RtcpFbHeader header;

	[[nodiscard]] static unsigned int Size();

	void preparePacket(SSRC messageSSRC);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpFirPart {
	uint32_t ssrc;
	uint8_t seqNo;
	uint8_t dummy1;
	uint16_t dummy2;
};

struct RTC_CPP_EXPORT RtcpFir {
	RtcpFbHeader header;
	RtcpFirPart parts[1];

	static unsigned int Size();

	void preparePacket(SSRC messageSSRC, uint8_t seqNo);

	void log() const;
};

struct RTC_CPP_EXPORT RtcpNackPart {
	uint16_t _pid;
	uint16_t _blp;

	uint16_t pid();
	uint16_t blp();

	void setPid(uint16_t pid);
	void setBlp(uint16_t blp);

	std::vector<uint16_t> getSequenceNumbers();
};

struct RTC_CPP_EXPORT RtcpNack {
	RtcpFbHeader header;
	RtcpNackPart parts[1];

	[[nodiscard]] static unsigned int Size(unsigned int discreteSeqNoCount);

	[[nodiscard]] unsigned int getSeqNoCount();

	void preparePacket(SSRC ssrc, unsigned int discreteSeqNoCount);

	/**
	 * Add a packet to the list of missing packets.
	 * @param fciCount The number of FCI fields that are present in this packet.
	 *                  Let the number start at zero and let this function grow the number.
	 * @param fciPID The seq no of the active FCI. It will be initialized automatically, and will
	 * change automatically.
	 * @param missingPacket The seq no of the missing packet. This will be added to the queue.
	 * @return true if the packet has grown, false otherwise.
	 */
	bool addMissingPacket(unsigned int *fciCount, uint16_t *fciPID, uint16_t missingPacket);
};

struct RTC_CPP_EXPORT RtpRtx {
	RtpHeader header;

	[[nodiscard]] const char *getBody() const;
	[[nodiscard]] char *getBody();
	[[nodiscard]] size_t getBodySize(size_t totalSize) const;
	[[nodiscard]] size_t getSize() const;
	[[nodiscard]] uint16_t getOriginalSeqNo() const;

	// Returns the new size of the packet
	size_t normalizePacket(size_t totalSize, SSRC originalSSRC, uint8_t originalPayloadType);

	size_t copyTo(RtpHeader *dest, size_t totalSize, uint8_t originalPayloadType);
};

// For backward compatibility, do not use
using RTP_ExtensionHeader = RtpExtensionHeader;
using RTP = RtpHeader;
using RTCP_ReportBlock = RtcpReportBlock;
using RTCP_HEADER = RtcpHeader;
using RTCP_FB_HEADER = RtcpFbHeader;
using RTCP_SR = RtcpSr;
using RTCP_SDES_ITEM = RtcpSdesItem;
using RTCP_SDES_CHUNK = RtcpSdesChunk;
using RTCP_SDES = RtcpSdes;
using RTCP_RR = RtcpRr;
using RTCP_REMB = RtcpRemb;
using RTCP_PLI = RtcpPli;
using RTCP_FIR_PART = RtcpFirPart;
using RTCP_FIR = RtcpFir;
using RTCP_NACK_PART = RtcpNackPart;
using RTCP_NACK = RtcpNack;
using RTP_RTX = RtpRtx;

#pragma pack(pop)

} // namespace rtc

#endif
