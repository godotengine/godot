/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_DATA_CHANNEL_H
#define RTC_IMPL_DATA_CHANNEL_H

#include "channel.hpp"
#include "common.hpp"
#include "message.hpp"
#include "peerconnection.hpp"
#include "queue.hpp"
#include "reliability.hpp"
#include "sctptransport.hpp"

#include <atomic>
#include <shared_mutex>

namespace rtc::impl {

struct PeerConnection;

struct DataChannel : Channel, std::enable_shared_from_this<DataChannel> {
	static bool IsOpenMessage(message_ptr message);

	DataChannel(weak_ptr<PeerConnection> pc, string label, string protocol,
	            Reliability reliability);
	virtual ~DataChannel();

	void close();
	void remoteClose();
	bool outgoing(message_ptr message);
	void incoming(message_ptr message);

	optional<message_variant> receive() override;
	optional<message_variant> peek() override;
	size_t availableAmount() const override;

	optional<uint16_t> stream() const;
	string label() const;
	string protocol() const;
	Reliability reliability() const;

	bool isOpen(void) const;
	bool isClosed(void) const;
	size_t maxMessageSize() const;

	virtual void assignStream(uint16_t stream);
	virtual void open(shared_ptr<SctpTransport> transport);
	virtual void processOpenMessage(message_ptr);

protected:
	const weak_ptr<impl::PeerConnection> mPeerConnection;
	weak_ptr<SctpTransport> mSctpTransport;

	optional<uint16_t> mStream;
	string mLabel;
	string mProtocol;
	shared_ptr<Reliability> mReliability;

	mutable std::shared_mutex mMutex;

	std::atomic<bool> mIsOpen = false;
	std::atomic<bool> mIsClosed = false;

private:
	Queue<message_ptr> mRecvQueue;
};

struct OutgoingDataChannel final : public DataChannel {
	OutgoingDataChannel(weak_ptr<PeerConnection> pc, string label, string protocol,
	                    Reliability reliability);
	~OutgoingDataChannel();

	void open(shared_ptr<SctpTransport> transport) override;
	void processOpenMessage(message_ptr message) override;
};

struct IncomingDataChannel final : public DataChannel {
	IncomingDataChannel(weak_ptr<PeerConnection> pc, weak_ptr<SctpTransport> transport);
	~IncomingDataChannel();

	void open(shared_ptr<SctpTransport> transport) override;
	void processOpenMessage(message_ptr message) override;
};

} // namespace rtc::impl

#endif
