/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "datachannel.hpp"
#include "common.hpp"
#include "internals.hpp"
#include "logcounter.hpp"
#include "peerconnection.hpp"
#include "sctptransport.hpp"

#include "rtc/datachannel.hpp"
#include "rtc/track.hpp"

#include <algorithm>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

using std::chrono::milliseconds;

namespace rtc::impl {

// Messages for the DataChannel establishment protocol (RFC 8832)
// See https://www.rfc-editor.org/rfc/rfc8832.html

enum MessageType : uint8_t {
	MESSAGE_OPEN_REQUEST = 0x00,
	MESSAGE_OPEN_RESPONSE = 0x01,
	MESSAGE_ACK = 0x02,
	MESSAGE_OPEN = 0x03
};

enum ChannelType : uint8_t {
	CHANNEL_RELIABLE = 0x00,
	CHANNEL_PARTIAL_RELIABLE_REXMIT = 0x01,
	CHANNEL_PARTIAL_RELIABLE_TIMED = 0x02
};

#pragma pack(push, 1)
struct OpenMessage {
	uint8_t type = MESSAGE_OPEN;
	uint8_t channelType;
	uint16_t priority;
	uint32_t reliabilityParameter;
	uint16_t labelLength;
	uint16_t protocolLength;
	// The following fields are:
	// uint8_t[labelLength] label
	// uint8_t[protocolLength] protocol
};

struct AckMessage {
	uint8_t type = MESSAGE_ACK;
};

#pragma pack(pop)

bool DataChannel::IsOpenMessage(message_ptr message) {
	if (message->type != Message::Control)
		return false;

	auto raw = reinterpret_cast<const uint8_t *>(message->data());
	return !message->empty() && raw[0] == MESSAGE_OPEN;
}

DataChannel::DataChannel(weak_ptr<PeerConnection> pc, string label, string protocol,
                         Reliability reliability)
    : mPeerConnection(pc), mLabel(std::move(label)), mProtocol(std::move(protocol)),
      mReliability(std::make_shared<Reliability>(std::move(reliability))),
      mRecvQueue(RECV_QUEUE_LIMIT, message_size_func) {}

DataChannel::~DataChannel() {
	PLOG_VERBOSE << "Destroying DataChannel";
	try {
		close();
	} catch (const std::exception &e) {
		PLOG_ERROR << e.what();
	}
}

void DataChannel::close() {
	PLOG_VERBOSE << "Closing DataChannel";

	shared_ptr<SctpTransport> transport;
	{
		std::shared_lock lock(mMutex);
		transport = mSctpTransport.lock();
	}

	if (!mIsClosed.exchange(true)) {
		if (transport && mStream.has_value())
			transport->closeStream(mStream.value());

		triggerClosed();
	}

	resetCallbacks();
}

void DataChannel::remoteClose() { close(); }

optional<message_variant> DataChannel::receive() {
	auto next = mRecvQueue.pop();
	return next ? std::make_optional(to_variant(std::move(**next))) : nullopt;
}

optional<message_variant> DataChannel::peek() {
	auto next = mRecvQueue.peek();
	return next ? std::make_optional(to_variant(**next)) : nullopt;
}

size_t DataChannel::availableAmount() const { return mRecvQueue.amount(); }

optional<uint16_t> DataChannel::stream() const {
	std::shared_lock lock(mMutex);
	return mStream;
}

string DataChannel::label() const {
	std::shared_lock lock(mMutex);
	return mLabel;
}

string DataChannel::protocol() const {
	std::shared_lock lock(mMutex);
	return mProtocol;
}

Reliability DataChannel::reliability() const {
	std::shared_lock lock(mMutex);
	return *mReliability;
}

bool DataChannel::isOpen(void) const { return !mIsClosed && mIsOpen; }

bool DataChannel::isClosed(void) const { return mIsClosed; }

size_t DataChannel::maxMessageSize() const {
	auto pc = mPeerConnection.lock();
	return pc ? pc->remoteMaxMessageSize() : DEFAULT_MAX_MESSAGE_SIZE;
}

void DataChannel::assignStream(uint16_t stream) {
	std::unique_lock lock(mMutex);

	if (mStream.has_value())
		throw std::logic_error("DataChannel already has a stream assigned");

	mStream = stream;
}

void DataChannel::open(shared_ptr<SctpTransport> transport) {
	{
		std::unique_lock lock(mMutex);
		mSctpTransport = transport;
	}

	if (!mIsClosed && !mIsOpen.exchange(true))
		triggerOpen();
}

void DataChannel::processOpenMessage(message_ptr) {
	PLOG_WARNING << "Received an open message for a user-negotiated DataChannel, ignoring";
}

bool DataChannel::outgoing(message_ptr message) {
	shared_ptr<SctpTransport> transport;
	{
		std::shared_lock lock(mMutex);
		transport = mSctpTransport.lock();

		if (!transport || mIsClosed)
			throw std::runtime_error("DataChannel is closed");

		if (!mStream.has_value())
			throw std::logic_error("DataChannel has no stream assigned");

		if (message->size() > maxMessageSize())
			throw std::invalid_argument("Message size exceeds limit");

		// Before the ACK has been received on a DataChannel, all messages must be sent ordered
		message->reliability = mIsOpen ? mReliability : nullptr;
		message->stream = mStream.value();
	}

	return transport->send(message);
}

void DataChannel::incoming(message_ptr message) {
	if (!message || mIsClosed)
		return;

	switch (message->type) {
	case Message::Control: {
		if (message->size() == 0)
			break; // Ignore
		auto raw = reinterpret_cast<const uint8_t *>(message->data());
		switch (raw[0]) {
		case MESSAGE_OPEN:
			processOpenMessage(message);
			break;
		case MESSAGE_ACK:
			if (!mIsOpen.exchange(true)) {
				triggerOpen();
			}
			break;
		default:
			// Ignore
			break;
		}
		break;
	}
	case Message::Reset:
		remoteClose();
		break;
	case Message::String:
	case Message::Binary:
		mRecvQueue.push(message);
		triggerAvailable(mRecvQueue.size());
		break;
	default:
		// Ignore
		break;
	}
}

OutgoingDataChannel::OutgoingDataChannel(weak_ptr<PeerConnection> pc, string label, string protocol,
                                         Reliability reliability)
    : DataChannel(pc, std::move(label), std::move(protocol), std::move(reliability)) {}

OutgoingDataChannel::~OutgoingDataChannel() {}

void OutgoingDataChannel::open(shared_ptr<SctpTransport> transport) {
	std::unique_lock lock(mMutex);
	mSctpTransport = transport;

	if (!mStream.has_value())
		throw std::runtime_error("DataChannel has no stream assigned");

	uint8_t channelType;
	uint32_t reliabilityParameter;
	switch (mReliability->type) {
	case Reliability::Type::Rexmit:
		channelType = CHANNEL_PARTIAL_RELIABLE_REXMIT;
		reliabilityParameter = uint32_t(std::max(std::get<int>(mReliability->rexmit), 0));
		break;

	case Reliability::Type::Timed:
		channelType = CHANNEL_PARTIAL_RELIABLE_TIMED;
		reliabilityParameter = uint32_t(std::get<milliseconds>(mReliability->rexmit).count());
		break;

	default:
		channelType = CHANNEL_RELIABLE;
		reliabilityParameter = 0;
		break;
	}

	if (mReliability->unordered)
		channelType |= 0x80;

	const size_t len = sizeof(OpenMessage) + mLabel.size() + mProtocol.size();
	binary buffer(len, byte(0));
	auto &open = *reinterpret_cast<OpenMessage *>(buffer.data());
	open.type = MESSAGE_OPEN;
	open.channelType = channelType;
	open.priority = htons(0);
	open.reliabilityParameter = htonl(reliabilityParameter);
	open.labelLength = htons(uint16_t(mLabel.size()));
	open.protocolLength = htons(uint16_t(mProtocol.size()));

	auto end = reinterpret_cast<char *>(buffer.data() + sizeof(OpenMessage));
	std::copy(mLabel.begin(), mLabel.end(), end);
	std::copy(mProtocol.begin(), mProtocol.end(), end + mLabel.size());

	lock.unlock();

	transport->send(make_message(buffer.begin(), buffer.end(), Message::Control, mStream.value()));
}

void OutgoingDataChannel::processOpenMessage(message_ptr) {
	PLOG_WARNING << "Received an open message for a locally-created DataChannel, ignoring";
}

IncomingDataChannel::IncomingDataChannel(weak_ptr<PeerConnection> pc,
                                         weak_ptr<SctpTransport> transport)
    : DataChannel(pc, "", "", {}) {

	mSctpTransport = transport;
}

IncomingDataChannel::~IncomingDataChannel() {}

void IncomingDataChannel::open(shared_ptr<SctpTransport>) {
	// Ignore
}

void IncomingDataChannel::processOpenMessage(message_ptr message) {
	std::unique_lock lock(mMutex);
	auto transport = mSctpTransport.lock();
	if (!transport)
		throw std::logic_error("DataChannel has no transport");

	if (!mStream.has_value())
		throw std::logic_error("DataChannel has no stream assigned");

	if (message->size() < sizeof(OpenMessage))
		throw std::invalid_argument("DataChannel open message too small");

	OpenMessage open = *reinterpret_cast<const OpenMessage *>(message->data());
	open.priority = ntohs(open.priority);
	open.reliabilityParameter = ntohl(open.reliabilityParameter);
	open.labelLength = ntohs(open.labelLength);
	open.protocolLength = ntohs(open.protocolLength);

	if (message->size() < sizeof(OpenMessage) + size_t(open.labelLength + open.protocolLength))
		throw std::invalid_argument("DataChannel open message truncated");

	auto end = reinterpret_cast<const char *>(message->data() + sizeof(OpenMessage));
	mLabel.assign(end, open.labelLength);
	mProtocol.assign(end + open.labelLength, open.protocolLength);

	mReliability->unordered = (open.channelType & 0x80) != 0;
	switch (open.channelType & 0x7F) {
	case CHANNEL_PARTIAL_RELIABLE_REXMIT:
		mReliability->type = Reliability::Type::Rexmit;
		mReliability->rexmit = int(open.reliabilityParameter);
		break;
	case CHANNEL_PARTIAL_RELIABLE_TIMED:
		mReliability->type = Reliability::Type::Timed;
		mReliability->rexmit = milliseconds(open.reliabilityParameter);
		break;
	default:
		mReliability->type = Reliability::Type::Reliable;
		mReliability->rexmit = int(0);
	}

	lock.unlock();

	binary buffer(sizeof(AckMessage), byte(0));
	auto &ack = *reinterpret_cast<AckMessage *>(buffer.data());
	ack.type = MESSAGE_ACK;

	transport->send(make_message(buffer.begin(), buffer.end(), Message::Control, mStream.value()));

	if (!mIsOpen.exchange(true))
		triggerOpen();
}

} // namespace rtc::impl
