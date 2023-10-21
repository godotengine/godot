/**
 * Copyright (c) 2020 Staz Modrzynski
 * Copyright (c) 2020 Paul-Louis Ageneau
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "rtp.hpp"

#include "impl/internals.hpp"

#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

#ifndef htonll
#define htonll(x)                                                                                  \
	((uint64_t)(((uint64_t)htonl((uint32_t)(x))) << 32) | (uint64_t)htonl((uint32_t)((x) >> 32)))
#endif
#ifndef ntohll
#define ntohll(x) htonll(x)
#endif

namespace rtc {

bool IsRtcp(const binary &data) {
	if (data.size() < 8)
		return false;

	uint8_t payloadType = std::to_integer<uint8_t>(data[1]) & 0x7F;
	PLOG_VERBOSE << "Demultiplexing RTCP and RTP with payload type, value=" << int(payloadType);

	// RFC 5761 Multiplexing RTP and RTCP 4. Distinguishable RTP and RTCP Packets
	// https://www.rfc-editor.org/rfc/rfc5761.html#section-4
	// It is RECOMMENDED to follow the guidelines in the RTP/AVP profile for the choice of RTP
	// payload type values, with the additional restriction that payload type values in the
	// range 64-95 MUST NOT be used. Specifically, dynamic RTP payload types SHOULD be chosen in
	// the range 96-127 where possible. Values below 64 MAY be used if that is insufficient
	// [...]
	return (payloadType >= 64 && payloadType <= 95); // Range 64-95 (inclusive) MUST be RTCP
}

uint8_t RtpHeader::version() const { return _first >> 6; }
bool RtpHeader::padding() const { return (_first >> 5) & 0x01; }
bool RtpHeader::extension() const { return (_first >> 4) & 0x01; }
uint8_t RtpHeader::csrcCount() const { return _first & 0x0F; }
uint8_t RtpHeader::marker() const { return _payloadType & 0b10000000; }
uint8_t RtpHeader::payloadType() const { return _payloadType & 0b01111111; }
uint16_t RtpHeader::seqNumber() const { return ntohs(_seqNumber); }
uint32_t RtpHeader::timestamp() const { return ntohl(_timestamp); }
uint32_t RtpHeader::ssrc() const { return ntohl(_ssrc); }

size_t RtpHeader::getSize() const {
	return reinterpret_cast<const char *>(&_ssrc + 1 + csrcCount()) -
	       reinterpret_cast<const char *>(this);
}

size_t RtpHeader::getExtensionHeaderSize() const {
	auto header = getExtensionHeader();
	return header ? header->getSize() + sizeof(RtpExtensionHeader) : 0;
}

const RtpExtensionHeader *RtpHeader::getExtensionHeader() const {
	return extension() ? reinterpret_cast<const RtpExtensionHeader *>(&_ssrc + 1 + csrcCount())
	                   : nullptr;
}

RtpExtensionHeader *RtpHeader::getExtensionHeader() {
	return extension() ? reinterpret_cast<RtpExtensionHeader *>(&_ssrc + 1 + csrcCount()) : nullptr;
}

const char *RtpHeader::getBody() const {
	return reinterpret_cast<const char *>(&_ssrc + 1 + csrcCount()) + getExtensionHeaderSize();
}

char *RtpHeader::getBody() {
	return reinterpret_cast<char *>(&_ssrc + 1 + csrcCount()) + getExtensionHeaderSize();
}

void RtpHeader::preparePacket() { _first |= (1 << 7); }

void RtpHeader::setSeqNumber(uint16_t newSeqNo) { _seqNumber = htons(newSeqNo); }

void RtpHeader::setPayloadType(uint8_t newPayloadType) {
	_payloadType = (_payloadType & 0b10000000u) | (0b01111111u & newPayloadType);
}

void RtpHeader::setSsrc(uint32_t in_ssrc) { _ssrc = htonl(in_ssrc); }

void RtpHeader::setMarker(bool marker) { _payloadType = (_payloadType & 0x7F) | (marker << 7); };

void RtpHeader::setTimestamp(uint32_t i) { _timestamp = htonl(i); }

void RtpHeader::setExtension(bool extension) { _first = (_first & ~0x10) | ((extension & 1) << 4); }

void RtpHeader::log() const {
	PLOG_VERBOSE << "RtpHeader V: " << (int)version() << " P: " << (padding() ? "P" : " ")
	             << " X: " << (extension() ? "X" : " ") << " CC: " << (int)csrcCount()
	             << " M: " << (marker() ? "M" : " ") << " PT: " << (int)payloadType()
	             << " SEQNO: " << seqNumber() << " TS: " << timestamp();
}

uint16_t RtpExtensionHeader::profileSpecificId() const { return ntohs(_profileSpecificId); }

uint16_t RtpExtensionHeader::headerLength() const { return ntohs(_headerLength); }

size_t RtpExtensionHeader::getSize() const { return headerLength() * 4; }

const char *RtpExtensionHeader::getBody() const {
	return reinterpret_cast<const char *>((&_headerLength) + 1);
}

char *RtpExtensionHeader::getBody() { return reinterpret_cast<char *>((&_headerLength) + 1); }

void RtpExtensionHeader::setProfileSpecificId(uint16_t profileSpecificId) {
	_profileSpecificId = htons(profileSpecificId);
}

void RtpExtensionHeader::setHeaderLength(uint16_t headerLength) {
	_headerLength = htons(headerLength);
}

void RtpExtensionHeader::clearBody() { std::memset(getBody(), 0, getSize()); }

void RtpExtensionHeader::writeOneByteHeader(size_t offset, uint8_t id, const byte *value,
                                            size_t size) {
	if ((id == 0) || (id > 14) || (size == 0) || (size > 16) || ((offset + 1 + size) > getSize()))
		return;
	auto buf = getBody() + offset;
	buf[0] = id << 4;
	if (size != 1) {
		buf[0] |= (uint8_t(size) - 1);
	}
	std::memcpy(buf + 1, value, size);
}

void RtpExtensionHeader::writeCurrentVideoOrientation(size_t offset, const uint8_t id,
                                                      uint8_t value) {
	auto v = std::byte{value};
	writeOneByteHeader(offset, id, &v, 1);
}

SSRC RtcpReportBlock::getSSRC() const { return ntohl(_ssrc); }

void RtcpReportBlock::preparePacket(SSRC in_ssrc, [[maybe_unused]] unsigned int packetsLost,
                                    [[maybe_unused]] unsigned int totalPackets,
                                    uint16_t highestSeqNo, uint16_t seqNoCycles, uint32_t jitter,
                                    uint64_t lastSR_NTP, uint64_t lastSR_DELAY) {
	setSeqNo(highestSeqNo, seqNoCycles);
	setJitter(jitter);
	setSSRC(in_ssrc);

	// Middle 32 bits of NTP Timestamp
	// _lastReport = lastSR_NTP >> 16u;
	setNTPOfSR(uint64_t(lastSR_NTP));
	setDelaySinceSR(uint32_t(lastSR_DELAY));

	// The delay, expressed in units of 1/65536 seconds
	// _delaySinceLastReport = lastSR_DELAY;
}

void RtcpReportBlock::setSSRC(SSRC in_ssrc) { _ssrc = htonl(in_ssrc); }

void RtcpReportBlock::setPacketsLost([[maybe_unused]] unsigned int packetsLost,
                                     [[maybe_unused]] unsigned int totalPackets) {
	// TODO Implement loss percentages.
	_fractionLostAndPacketsLost = 0;
}

unsigned int RtcpReportBlock::getLossPercentage() const {
	// TODO Implement loss percentages.
	return 0;
}

unsigned int RtcpReportBlock::getPacketLostCount() const {
	// TODO Implement total packets lost.
	return 0;
}

uint16_t RtcpReportBlock::seqNoCycles() const { return ntohs(_seqNoCycles); }

uint16_t RtcpReportBlock::highestSeqNo() const { return ntohs(_highestSeqNo); }

uint32_t RtcpReportBlock::jitter() const { return ntohl(_jitter); }

uint32_t RtcpReportBlock::delaySinceSR() const { return ntohl(_delaySinceLastReport); }

void RtcpReportBlock::setSeqNo(uint16_t highestSeqNo, uint16_t seqNoCycles) {
	_highestSeqNo = htons(highestSeqNo);
	_seqNoCycles = htons(seqNoCycles);
}

void RtcpReportBlock::setJitter(uint32_t jitter) { _jitter = htonl(jitter); }

void RtcpReportBlock::setNTPOfSR(uint64_t ntp) { _lastReport = htonll(ntp >> 16u); }

uint32_t RtcpReportBlock::getNTPOfSR() const { return ntohl(_lastReport) << 16u; }

void RtcpReportBlock::setDelaySinceSR(uint32_t sr) {
	// The delay, expressed in units of 1/65536 seconds
	_delaySinceLastReport = htonl(sr);
}

void RtcpReportBlock::log() const {
	PLOG_VERBOSE << "RTCP report block: "
	             << "ssrc="
	             << ntohl(_ssrc)
	             // TODO: Implement these reports
	             //	<< ", fractionLost=" << fractionLost
	             //	<< ", packetsLost=" << packetsLost
	             << ", highestSeqNo=" << highestSeqNo() << ", seqNoCycles=" << seqNoCycles()
	             << ", jitter=" << jitter() << ", lastSR=" << getNTPOfSR()
	             << ", lastSRDelay=" << delaySinceSR();
}

uint8_t RtcpHeader::version() const { return _first >> 6; }

bool RtcpHeader::padding() const { return (_first >> 5) & 0x01; }

uint8_t RtcpHeader::reportCount() const { return _first & 0x1F; }

uint8_t RtcpHeader::payloadType() const { return _payloadType; }

uint16_t RtcpHeader::length() const { return ntohs(_length); }

size_t RtcpHeader::lengthInBytes() const { return (1 + length()) * 4; }

void RtcpHeader::setPayloadType(uint8_t type) { _payloadType = type; }

void RtcpHeader::setReportCount(uint8_t count) {
	_first = (_first & 0b11100000u) | (count & 0b00011111u);
}

void RtcpHeader::setLength(uint16_t length) { _length = htons(length); }

void RtcpHeader::prepareHeader(uint8_t payloadType, uint8_t reportCount, uint16_t length) {
	_first = 0b10000000; // version 2, no padding
	setReportCount(reportCount);
	setPayloadType(payloadType);
	setLength(length);
}

void RtcpHeader::log() const {
	PLOG_VERBOSE << "RTCP header: "
	             << "version=" << unsigned(version()) << ", padding=" << padding()
	             << ", reportCount=" << unsigned(reportCount())
	             << ", payloadType=" << unsigned(payloadType()) << ", length=" << length();
}

SSRC RtcpFbHeader::packetSenderSSRC() const { return ntohl(_packetSender); }

SSRC RtcpFbHeader::mediaSourceSSRC() const { return ntohl(_mediaSource); }

void RtcpFbHeader::setPacketSenderSSRC(SSRC ssrc) { _packetSender = htonl(ssrc); }

void RtcpFbHeader::setMediaSourceSSRC(SSRC ssrc) { _mediaSource = htonl(ssrc); }

void RtcpFbHeader::log() const {
	header.log();
	PLOG_VERBOSE << "FB: "
	             << " packet sender: " << packetSenderSSRC()
	             << " media source: " << mediaSourceSSRC();
}

unsigned int RtcpSr::Size(unsigned int reportCount) {
	return sizeof(RtcpHeader) + 24 + reportCount * sizeof(RtcpReportBlock);
}

void RtcpSr::preparePacket(SSRC senderSSRC, uint8_t reportCount) {
	unsigned int length = ((sizeof(header) + 24 + reportCount * sizeof(RtcpReportBlock)) / 4) - 1;
	header.prepareHeader(200, reportCount, uint16_t(length));
	this->_senderSSRC = htonl(senderSSRC);
}

const RtcpReportBlock *RtcpSr::getReportBlock(int num) const { return &_reportBlocks + num; }

RtcpReportBlock *RtcpSr::getReportBlock(int num) { return &_reportBlocks + num; }

size_t RtcpSr::getSize() const {
	// "length" in packet is one less than the number of 32 bit words in the packet.
	return sizeof(uint32_t) * (1 + size_t(header.length()));
}

uint64_t RtcpSr::ntpTimestamp() const { return ntohll(_ntpTimestamp); }
uint32_t RtcpSr::rtpTimestamp() const { return ntohl(_rtpTimestamp); }
uint32_t RtcpSr::packetCount() const { return ntohl(_packetCount); }
uint32_t RtcpSr::octetCount() const { return ntohl(_octetCount); }
uint32_t RtcpSr::senderSSRC() const { return ntohl(_senderSSRC); }

void RtcpSr::setNtpTimestamp(uint64_t ts) { _ntpTimestamp = htonll(ts); }
void RtcpSr::setRtpTimestamp(uint32_t ts) { _rtpTimestamp = htonl(ts); }
void RtcpSr::setOctetCount(uint32_t ts) { _octetCount = htonl(ts); }
void RtcpSr::setPacketCount(uint32_t ts) { _packetCount = htonl(ts); }

void RtcpSr::log() const {
	header.log();
	PLOG_VERBOSE << "RTCP SR: "
	             << " SSRC=" << senderSSRC() << ", NTP_TS=" << ntpTimestamp()
	             << ", RtpTS=" << rtpTimestamp() << ", packetCount=" << packetCount()
	             << ", octetCount=" << octetCount();

	for (unsigned i = 0; i < unsigned(header.reportCount()); i++) {
		getReportBlock(i)->log();
	}
}

unsigned int RtcpSdesItem::Size(uint8_t textLength) { return textLength + 2; }

std::string RtcpSdesItem::text() const { return std::string(_text, _length); }

void RtcpSdesItem::setText(std::string text) {
	if (text.size() > 0xFF)
		throw std::invalid_argument("text is too long");

	_length = uint8_t(text.size());
	memcpy(_text, text.data(), text.size());
}

uint8_t RtcpSdesItem::length() const { return _length; }

unsigned int RtcpSdesChunk::Size(const std::vector<uint8_t> textLengths) {
	unsigned int itemsSize = 0;
	for (auto length : textLengths) {
		itemsSize += RtcpSdesItem::Size(length);
	}
	auto nullTerminatedItemsSize = itemsSize + 1;
	auto words = uint8_t(std::ceil(double(nullTerminatedItemsSize) / 4)) + 1;
	return words * 4;
}

SSRC RtcpSdesChunk::ssrc() const { return ntohl(_ssrc); }

void RtcpSdesChunk::setSSRC(SSRC ssrc) { _ssrc = htonl(ssrc); }

const RtcpSdesItem *RtcpSdesChunk::getItem(int num) const {
	auto base = &_items;
	while (num-- > 0) {
		auto itemSize = RtcpSdesItem::Size(base->length());
		base = reinterpret_cast<const RtcpSdesItem *>(reinterpret_cast<const uint8_t *>(base) +
		                                              itemSize);
	}
	return reinterpret_cast<const RtcpSdesItem *>(base);
}

RtcpSdesItem *RtcpSdesChunk::getItem(int num) {
	auto base = &_items;
	while (num-- > 0) {
		auto itemSize = RtcpSdesItem::Size(base->length());
		base = reinterpret_cast<RtcpSdesItem *>(reinterpret_cast<uint8_t *>(base) + itemSize);
	}
	return reinterpret_cast<RtcpSdesItem *>(base);
}

unsigned int RtcpSdesChunk::getSize() const {
	std::vector<uint8_t> textLengths{};
	unsigned int i = 0;
	auto item = getItem(i);
	while (item->type != 0) {
		textLengths.push_back(item->length());
		item = getItem(++i);
	}
	return Size(textLengths);
}

long RtcpSdesChunk::safelyCountChunkSize(size_t maxChunkSize) const {
	if (maxChunkSize < RtcpSdesChunk::Size({})) {
		// chunk is truncated
		return -1;
	}

	size_t size = sizeof(SSRC);
	unsigned int i = 0;
	// We can always access first 4 bytes of first item (in case of no items there will be 4
	// null bytes)
	auto item = getItem(i);
	std::vector<uint8_t> textsLength{};
	while (item->type != 0) {
		if (size + RtcpSdesItem::Size(0) > maxChunkSize) {
			// item is too short
			return -1;
		}
		auto itemLength = item->length();
		if (size + RtcpSdesItem::Size(itemLength) >= maxChunkSize) {
			// item is too large (it can't be equal to chunk size because after item there
			// must be 1-4 null bytes as padding)
			return -1;
		}
		textsLength.push_back(itemLength);
		// safely to access next item
		item = getItem(++i);
	}
	auto realSize = RtcpSdesChunk::Size(textsLength);
	if (realSize > maxChunkSize) {
		// Chunk is too large
		return -1;
	}
	return realSize;
}

unsigned int RtcpSdes::Size(const std::vector<std::vector<uint8_t>> lengths) {
	unsigned int chunks_size = 0;
	for (auto length : lengths)
		chunks_size += RtcpSdesChunk::Size(length);

	return 4 + chunks_size;
}

bool RtcpSdes::isValid() const {
	auto chunksSize = header.lengthInBytes() - sizeof(header);
	if (chunksSize == 0) {
		return true;
	}
	// there is at least one chunk
	unsigned int i = 0;
	unsigned int size = 0;
	while (size < chunksSize) {
		if (chunksSize < size + RtcpSdesChunk::Size({})) {
			// chunk is truncated
			return false;
		}
		auto chunk = getChunk(i++);
		auto chunkSize = chunk->safelyCountChunkSize(chunksSize - size);
		if (chunkSize < 0) {
			// chunk is invalid
			return false;
		}
		size += chunkSize;
	}
	return size == chunksSize;
}

unsigned int RtcpSdes::chunksCount() const {
	if (!isValid()) {
		return 0;
	}
	uint16_t chunksSize = 4 * (header.length() + 1) - sizeof(header);
	unsigned int size = 0;
	unsigned int i = 0;
	while (size < chunksSize) {
		size += getChunk(i++)->getSize();
	}
	return i;
}

const RtcpSdesChunk *RtcpSdes::getChunk(int num) const {
	auto base = &_chunks;
	while (num-- > 0) {
		auto chunkSize = base->getSize();
		base = reinterpret_cast<const RtcpSdesChunk *>(reinterpret_cast<const uint8_t *>(base) +
		                                               chunkSize);
	}
	return reinterpret_cast<const RtcpSdesChunk *>(base);
}

RtcpSdesChunk *RtcpSdes::getChunk(int num) {
	auto base = &_chunks;
	while (num-- > 0) {
		auto chunkSize = base->getSize();
		base = reinterpret_cast<RtcpSdesChunk *>(reinterpret_cast<uint8_t *>(base) + chunkSize);
	}
	return reinterpret_cast<RtcpSdesChunk *>(base);
}

void RtcpSdes::preparePacket(uint8_t chunkCount) {
	unsigned int chunkSize = 0;
	for (uint8_t i = 0; i < chunkCount; i++) {
		auto chunk = getChunk(i);
		chunkSize += chunk->getSize();
	}
	uint16_t length = uint16_t((sizeof(header) + chunkSize) / 4 - 1);
	header.prepareHeader(202, chunkCount, length);
}

const RtcpReportBlock *RtcpRr::getReportBlock(int num) const { return &_reportBlocks + num; }

RtcpReportBlock *RtcpRr::getReportBlock(int num) { return &_reportBlocks + num; }

size_t RtcpRr::SizeWithReportBlocks(uint8_t reportCount) {
	return sizeof(header) + 4 + size_t(reportCount) * sizeof(RtcpReportBlock);
}

SSRC RtcpRr::senderSSRC() const { return ntohl(_senderSSRC); }

bool RtcpRr::isSenderReport() { return header.payloadType() == 200; }

bool RtcpRr::isReceiverReport() { return header.payloadType() == 201; }

size_t RtcpRr::getSize() const {
	// "length" in packet is one less than the number of 32 bit words in the packet.
	return sizeof(uint32_t) * (1 + size_t(header.length()));
}

void RtcpRr::preparePacket(SSRC senderSSRC, uint8_t reportCount) {
	// "length" in packet is one less than the number of 32 bit words in the packet.
	size_t length = (SizeWithReportBlocks(reportCount) / 4) - 1;
	header.prepareHeader(201, reportCount, uint16_t(length));
	this->_senderSSRC = htonl(senderSSRC);
}

void RtcpRr::setSenderSSRC(SSRC ssrc) { this->_senderSSRC = htonl(ssrc); }

void RtcpRr::log() const {
	header.log();
	PLOG_VERBOSE << "RTCP RR: "
	             << " SSRC=" << ntohl(_senderSSRC);

	for (unsigned i = 0; i < unsigned(header.reportCount()); i++) {
		getReportBlock(i)->log();
	}
}

size_t RtcpRemb::SizeWithSSRCs(int count) { return sizeof(RtcpRemb) + (count - 1) * sizeof(SSRC); }

unsigned int RtcpRemb::getSize() const {
	// "length" in packet is one less than the number of 32 bit words in the packet.
	return sizeof(uint32_t) * (1 + header.header.length());
}

void RtcpRemb::preparePacket(SSRC senderSSRC, unsigned int numSSRC, unsigned int in_bitrate) {

	// Report Count becomes the format here.
	header.header.prepareHeader(206, 15, 0);

	// Always zero.
	header.setMediaSourceSSRC(0);

	header.setPacketSenderSSRC(senderSSRC);

	_id[0] = 'R';
	_id[1] = 'E';
	_id[2] = 'M';
	_id[3] = 'B';

	setBitrate(numSSRC, in_bitrate);
}

void RtcpRemb::setBitrate(unsigned int numSSRC, unsigned int in_bitrate) {
	unsigned int exp = 0;
	while (in_bitrate > pow(2, 18) - 1) {
		exp++;
		in_bitrate /= 2;
	}

	// "length" in packet is one less than the number of 32 bit words in the packet.
	header.header.setLength(uint16_t((offsetof(RtcpRemb, _ssrc) / sizeof(uint32_t)) - 1 + numSSRC));

	_bitrate = htonl((numSSRC << (32u - 8u)) | (exp << (32u - 8u - 6u)) | in_bitrate);
}

void RtcpRemb::setSsrc(int iterator, SSRC newSssrc) { _ssrc[iterator] = htonl(newSssrc); }

unsigned int RtcpPli::Size() { return sizeof(RtcpFbHeader); }

void RtcpPli::preparePacket(SSRC messageSSRC) {
	header.header.prepareHeader(206, 1, 2);
	header.setPacketSenderSSRC(messageSSRC);
	header.setMediaSourceSSRC(messageSSRC);
}

void RtcpPli::log() const { header.log(); }

unsigned int RtcpFir::Size() { return sizeof(RtcpFbHeader) + sizeof(RtcpFirPart); }

void RtcpFir::preparePacket(SSRC messageSSRC, uint8_t seqNo) {
	header.header.prepareHeader(206, 4, 2 + 2 * 1);
	header.setPacketSenderSSRC(messageSSRC);
	header.setMediaSourceSSRC(messageSSRC);
	parts[0].ssrc = htonl(messageSSRC);
	parts[0].seqNo = seqNo;
}

void RtcpFir::log() const { header.log(); }

uint16_t RtcpNackPart::pid() { return ntohs(_pid); }
uint16_t RtcpNackPart::blp() { return ntohs(_blp); }

void RtcpNackPart::setPid(uint16_t pid) { _pid = htons(pid); }
void RtcpNackPart::setBlp(uint16_t blp) { _blp = htons(blp); }

std::vector<uint16_t> RtcpNackPart::getSequenceNumbers() {
	std::vector<uint16_t> result{};
	result.reserve(17);
	uint16_t p = pid();
	result.push_back(p);
	uint16_t bitmask = blp();
	uint16_t i = p + 1;
	while (bitmask > 0) {
		if (bitmask & 0x1) {
			result.push_back(i);
		}
		i += 1;
		bitmask >>= 1;
	}
	return result;
}

unsigned int RtcpNack::Size(unsigned int discreteSeqNoCount) {
	return offsetof(RtcpNack, parts) + sizeof(RtcpNackPart) * discreteSeqNoCount;
}

unsigned int RtcpNack::getSeqNoCount() { return header.header.length() - 2; }

void RtcpNack::preparePacket(SSRC ssrc, unsigned int discreteSeqNoCount) {
	header.header.prepareHeader(205, 1, 2 + uint16_t(discreteSeqNoCount));
	header.setMediaSourceSSRC(ssrc);
	header.setPacketSenderSSRC(ssrc);
}

bool RtcpNack::addMissingPacket(unsigned int *fciCount, uint16_t *fciPID, uint16_t missingPacket) {
	if (*fciCount == 0 || missingPacket < *fciPID || missingPacket > (*fciPID + 16)) {
		parts[*fciCount].setPid(missingPacket);
		parts[*fciCount].setBlp(0);
		*fciPID = missingPacket;
		(*fciCount)++;
		return true;
	} else {
		// TODO SPEED!
		uint16_t blp = parts[(*fciCount) - 1].blp();
		uint16_t newBit = uint16_t(1u << (missingPacket - (1 + *fciPID)));
		parts[(*fciCount) - 1].setBlp(blp | newBit);
		return false;
	}
}

uint16_t RtpRtx::getOriginalSeqNo() const { return ntohs(*(uint16_t *)(header.getBody())); }

const char *RtpRtx::getBody() const { return header.getBody() + sizeof(uint16_t); }

char *RtpRtx::getBody() { return header.getBody() + sizeof(uint16_t); }

size_t RtpRtx::getBodySize(size_t totalSize) const {
	return totalSize - (getBody() - reinterpret_cast<const char *>(this));
}

size_t RtpRtx::getSize() const { return header.getSize() + sizeof(uint16_t); }

size_t RtpRtx::normalizePacket(size_t totalSize, SSRC originalSSRC, uint8_t originalPayloadType) {
	header.setSeqNumber(getOriginalSeqNo());
	header.setSsrc(originalSSRC);
	header.setPayloadType(originalPayloadType);
	// TODO, the -12 is the size of the header (which is variable!)
	memmove(header.getBody(), getBody(), totalSize - getSize());
	return totalSize - 2;
}

size_t RtpRtx::copyTo(RtpHeader *dest, size_t totalSize, uint8_t originalPayloadType) {
	memmove((char *)dest, (char *)this, header.getSize());
	dest->setSeqNumber(getOriginalSeqNo());
	dest->setPayloadType(originalPayloadType);
	memmove(dest->getBody(), getBody(), getBodySize(totalSize));
	return totalSize;
}

}; // namespace rtc
