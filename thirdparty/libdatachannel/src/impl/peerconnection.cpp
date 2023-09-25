/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "peerconnection.hpp"
#include "certificate.hpp"
#include "dtlstransport.hpp"
#include "icetransport.hpp"
#include "internals.hpp"
#include "logcounter.hpp"
#include "peerconnection.hpp"
#include "processor.hpp"
#include "rtp.hpp"
#include "sctptransport.hpp"
#include "utils.hpp"

#if RTC_ENABLE_MEDIA
#include "dtlssrtptransport.hpp"
#endif

#include <algorithm>
#include <array>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>

using namespace std::placeholders;

namespace rtc::impl {

static LogCounter COUNTER_MEDIA_TRUNCATED(plog::warning,
                                          "Number of truncated RTP packets over past second");
static LogCounter COUNTER_SRTP_DECRYPT_ERROR(plog::warning,
                                             "Number of SRTP decryption errors over past second");
static LogCounter COUNTER_SRTP_ENCRYPT_ERROR(plog::warning,
                                             "Number of SRTP encryption errors over past second");
static LogCounter
    COUNTER_UNKNOWN_PACKET_TYPE(plog::warning,
                                "Number of unknown RTCP packet types over past second");

PeerConnection::PeerConnection(Configuration config_)
    : config(std::move(config_)), mCertificate(make_certificate(config.certificateType)) {
	PLOG_VERBOSE << "Creating PeerConnection";

	if (config.portRangeEnd && config.portRangeBegin > config.portRangeEnd)
		throw std::invalid_argument("Invalid port range");

	if (config.mtu) {
		if (*config.mtu < 576) // Min MTU for IPv4
			throw std::invalid_argument("Invalid MTU value");

		if (*config.mtu > 1500) { // Standard Ethernet
			PLOG_WARNING << "MTU set to " << *config.mtu;
		} else {
			PLOG_VERBOSE << "MTU set to " << *config.mtu;
		}
	}
}

PeerConnection::~PeerConnection() {
	PLOG_VERBOSE << "Destroying PeerConnection";
	mProcessor.join();
}

void PeerConnection::close() {
	negotiationNeeded = false;
	if (!closing.exchange(true)) {
		PLOG_VERBOSE << "Closing PeerConnection";
		if (auto transport = std::atomic_load(&mSctpTransport))
			transport->stop();
		else
			remoteClose();
	}
}

void PeerConnection::remoteClose() {
	close();
	if (state.load() != State::Closed) {
		// Close data channels and tracks asynchronously
		mProcessor.enqueue(&PeerConnection::closeDataChannels, shared_from_this());
		mProcessor.enqueue(&PeerConnection::closeTracks, shared_from_this());

		closeTransports();
	}
}

optional<Description> PeerConnection::localDescription() const {
	std::lock_guard lock(mLocalDescriptionMutex);
	return mLocalDescription;
}

optional<Description> PeerConnection::remoteDescription() const {
	std::lock_guard lock(mRemoteDescriptionMutex);
	return mRemoteDescription;
}

size_t PeerConnection::remoteMaxMessageSize() const {
	const size_t localMax = config.maxMessageSize.value_or(DEFAULT_LOCAL_MAX_MESSAGE_SIZE);

	size_t remoteMax = DEFAULT_MAX_MESSAGE_SIZE;
	std::lock_guard lock(mRemoteDescriptionMutex);
	if (mRemoteDescription)
		if (auto *application = mRemoteDescription->application())
			if (auto max = application->maxMessageSize()) {
				// RFC 8841: If the SDP "max-message-size" attribute contains a maximum message
				// size value of zero, it indicates that the SCTP endpoint will handle messages
				// of any size, subject to memory capacity, etc.
				remoteMax = *max > 0 ? *max : std::numeric_limits<size_t>::max();
			}

	return std::min(remoteMax, localMax);
}

// Helper for PeerConnection::initXTransport methods: start and emplace the transport
template <typename T>
shared_ptr<T> emplaceTransport(PeerConnection *pc, shared_ptr<T> *member, shared_ptr<T> transport) {
	std::atomic_store(member, transport);
	try {
		transport->start();
	} catch (...) {
		std::atomic_store(member, decltype(transport)(nullptr));
		throw;
	}

	if (pc->closing.load() || pc->state.load() == PeerConnection::State::Closed) {
		std::atomic_store(member, decltype(transport)(nullptr));
		transport->stop();
		return nullptr;
	}

	return transport;
}

shared_ptr<IceTransport> PeerConnection::initIceTransport() {
	try {
		if (auto transport = std::atomic_load(&mIceTransport))
			return transport;

		PLOG_VERBOSE << "Starting ICE transport";

		auto transport = std::make_shared<IceTransport>(
		    config, weak_bind(&PeerConnection::processLocalCandidate, this, _1),
		    [this, weak_this = weak_from_this()](IceTransport::State transportState) {
			    auto shared_this = weak_this.lock();
			    if (!shared_this)
				    return;
			    switch (transportState) {
			    case IceTransport::State::Connecting:
				    changeIceState(IceState::Checking);
				    changeState(State::Connecting);
				    break;
			    case IceTransport::State::Connected:
				    changeIceState(IceState::Connected);
				    initDtlsTransport();
				    break;
			    case IceTransport::State::Completed:
				    changeIceState(IceState::Completed);
				    break;
			    case IceTransport::State::Failed:
				    changeIceState(IceState::Failed);
				    changeState(State::Failed);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    case IceTransport::State::Disconnected:
				    changeIceState(IceState::Disconnected);
				    changeState(State::Disconnected);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    default:
				    // Ignore
				    break;
			    }
		    },
		    [this, weak_this = weak_from_this()](IceTransport::GatheringState gatheringState) {
			    auto shared_this = weak_this.lock();
			    if (!shared_this)
				    return;
			    switch (gatheringState) {
			    case IceTransport::GatheringState::InProgress:
				    changeGatheringState(GatheringState::InProgress);
				    break;
			    case IceTransport::GatheringState::Complete:
				    endLocalCandidates();
				    changeGatheringState(GatheringState::Complete);
				    break;
			    default:
				    // Ignore
				    break;
			    }
		    });

		return emplaceTransport(this, &mIceTransport, std::move(transport));

	} catch (const std::exception &e) {
		PLOG_ERROR << e.what();
		changeState(State::Failed);
		throw std::runtime_error("ICE transport initialization failed");
	}
}

shared_ptr<DtlsTransport> PeerConnection::initDtlsTransport() {
	try {
		if (auto transport = std::atomic_load(&mDtlsTransport))
			return transport;

		PLOG_VERBOSE << "Starting DTLS transport";

		auto lower = std::atomic_load(&mIceTransport);
		if (!lower)
			throw std::logic_error("No underlying ICE transport for DTLS transport");

		auto certificate = mCertificate.get();
		auto verifierCallback = weak_bind(&PeerConnection::checkFingerprint, this, _1);
		auto dtlsStateChangeCallback =
		    [this, weak_this = weak_from_this()](DtlsTransport::State transportState) {
			    auto shared_this = weak_this.lock();
			    if (!shared_this)
				    return;

			    switch (transportState) {
			    case DtlsTransport::State::Connected:
				    if (auto remote = remoteDescription(); remote && remote->hasApplication())
					    initSctpTransport();
				    else
					    changeState(State::Connected);

				    mProcessor.enqueue(&PeerConnection::openTracks, shared_from_this());
				    break;
			    case DtlsTransport::State::Failed:
				    changeState(State::Failed);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    case DtlsTransport::State::Disconnected:
				    changeState(State::Disconnected);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    default:
				    // Ignore
				    break;
			    }
		    };

		shared_ptr<DtlsTransport> transport;
		auto local = localDescription();
		if (config.forceMediaTransport || (local && local->hasAudioOrVideo())) {
#if RTC_ENABLE_MEDIA
			PLOG_INFO << "This connection requires media support";

			// DTLS-SRTP
			transport = std::make_shared<DtlsSrtpTransport>(
			    lower, certificate, config.mtu, verifierCallback,
			    weak_bind(&PeerConnection::forwardMedia, this, _1), dtlsStateChangeCallback);
#else
			PLOG_WARNING << "Ignoring media support (not compiled with media support)";
#endif
		}

		if (!transport) {
			// DTLS only
			transport = std::make_shared<DtlsTransport>(lower, certificate, config.mtu,
			                                            verifierCallback, dtlsStateChangeCallback);
		}

		return emplaceTransport(this, &mDtlsTransport, std::move(transport));

	} catch (const std::exception &e) {
		PLOG_ERROR << e.what();
		changeState(State::Failed);
		throw std::runtime_error("DTLS transport initialization failed");
	}
}

shared_ptr<SctpTransport> PeerConnection::initSctpTransport() {
	try {
		if (auto transport = std::atomic_load(&mSctpTransport))
			return transport;

		PLOG_VERBOSE << "Starting SCTP transport";

		auto lower = std::atomic_load(&mDtlsTransport);
		if (!lower)
			throw std::logic_error("No underlying DTLS transport for SCTP transport");

		auto local = localDescription();
		if (!local || !local->application())
			throw std::logic_error("Starting SCTP transport without local application description");

		auto remote = remoteDescription();
		if (!remote || !remote->application())
			throw std::logic_error(
			    "Starting SCTP transport without remote application description");

		SctpTransport::Ports ports = {};
		ports.local = local->application()->sctpPort().value_or(DEFAULT_SCTP_PORT);
		ports.remote = remote->application()->sctpPort().value_or(DEFAULT_SCTP_PORT);

		auto transport = std::make_shared<SctpTransport>(
		    lower, config, std::move(ports), weak_bind(&PeerConnection::forwardMessage, this, _1),
		    weak_bind(&PeerConnection::forwardBufferedAmount, this, _1, _2),
		    [this, weak_this = weak_from_this()](SctpTransport::State transportState) {
			    auto shared_this = weak_this.lock();
			    if (!shared_this)
				    return;

			    switch (transportState) {
			    case SctpTransport::State::Connected:
				    changeState(State::Connected);
				    assignDataChannels();
				    mProcessor.enqueue(&PeerConnection::openDataChannels, shared_from_this());
				    break;
			    case SctpTransport::State::Failed:
				    changeState(State::Failed);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    case SctpTransport::State::Disconnected:
				    changeState(State::Disconnected);
				    mProcessor.enqueue(&PeerConnection::remoteClose, shared_from_this());
				    break;
			    default:
				    // Ignore
				    break;
			    }
		    });

		return emplaceTransport(this, &mSctpTransport, std::move(transport));

	} catch (const std::exception &e) {
		PLOG_ERROR << e.what();
		changeState(State::Failed);
		throw std::runtime_error("SCTP transport initialization failed");
	}
}

shared_ptr<IceTransport> PeerConnection::getIceTransport() const {
	return std::atomic_load(&mIceTransport);
}

shared_ptr<DtlsTransport> PeerConnection::getDtlsTransport() const {
	return std::atomic_load(&mDtlsTransport);
}

shared_ptr<SctpTransport> PeerConnection::getSctpTransport() const {
	return std::atomic_load(&mSctpTransport);
}

void PeerConnection::closeTransports() {
	PLOG_VERBOSE << "Closing transports";

	// Change ICE state to sink state Closed
	changeIceState(IceState::Closed);

	// Change state to sink state Closed
	if (!changeState(State::Closed))
		return; // already closed

	// Reset interceptor and callbacks now that state is changed
	setMediaHandler(nullptr);
	resetCallbacks();

	// Pass the pointers to a thread, allowing to terminate a transport from its own thread
	auto sctp = std::atomic_exchange(&mSctpTransport, decltype(mSctpTransport)(nullptr));
	auto dtls = std::atomic_exchange(&mDtlsTransport, decltype(mDtlsTransport)(nullptr));
	auto ice = std::atomic_exchange(&mIceTransport, decltype(mIceTransport)(nullptr));

	if (sctp) {
		sctp->onRecv(nullptr);
		sctp->onBufferedAmount(nullptr);
	}

	using array = std::array<shared_ptr<Transport>, 3>;
	array transports{std::move(sctp), std::move(dtls), std::move(ice)};

	for (const auto &t : transports)
		if (t)
			t->onStateChange(nullptr);

	TearDownProcessor::Instance().enqueue(
	    [transports = std::move(transports), token = Init::Instance().token()]() mutable {
		    for (const auto &t : transports) {
			    if (t) {
				    t->stop();
				    break;
			    }
		    }

		    for (auto &t : transports)
			    t.reset();
	    });
}

void PeerConnection::endLocalCandidates() {
	std::lock_guard lock(mLocalDescriptionMutex);
	if (mLocalDescription)
		mLocalDescription->endCandidates();
}

void PeerConnection::rollbackLocalDescription() {
	PLOG_DEBUG << "Rolling back pending local description";

	std::unique_lock lock(mLocalDescriptionMutex);
	if (mCurrentLocalDescription) {
		std::vector<Candidate> existingCandidates;
		if (mLocalDescription)
			existingCandidates = mLocalDescription->extractCandidates();

		mLocalDescription.emplace(std::move(*mCurrentLocalDescription));
		mLocalDescription->addCandidates(std::move(existingCandidates));
		mCurrentLocalDescription.reset();
	}
}

bool PeerConnection::checkFingerprint(const std::string &fingerprint) const {
	std::lock_guard lock(mRemoteDescriptionMutex);
	auto expectedFingerprint = mRemoteDescription ? mRemoteDescription->fingerprint() : nullopt;
	if (expectedFingerprint && *expectedFingerprint == fingerprint) {
		PLOG_VERBOSE << "Valid fingerprint \"" << fingerprint << "\"";
		return true;
	}

	PLOG_ERROR << "Invalid fingerprint \"" << fingerprint << "\", expected \""
	           << expectedFingerprint.value_or("[none]") << "\"";
	return false;
}

void PeerConnection::forwardMessage(message_ptr message) {
	if (!message) {
		remoteCloseDataChannels();
		return;
	}

	auto iceTransport = std::atomic_load(&mIceTransport);
	auto sctpTransport = std::atomic_load(&mSctpTransport);
	if (!iceTransport || !sctpTransport)
		return;

	const uint16_t stream = uint16_t(message->stream);
	auto [channel, found] = findDataChannel(stream);

	if (DataChannel::IsOpenMessage(message)) {
		if (found) {
			// The stream is already used, the receiver must close the DataChannel
			PLOG_WARNING << "Got open message on already used stream " << stream;
			if (channel && !channel->isClosed())
				channel->close();
			else
				sctpTransport->closeStream(message->stream);

			return;
		}

		const uint16_t remoteParity = (iceTransport->role() == Description::Role::Active) ? 1 : 0;
		if (stream % 2 != remoteParity) {
			// The odd/even rule is violated, the receiver must close the DataChannel
			PLOG_WARNING << "Got open message violating the odd/even rule on stream " << stream;
			sctpTransport->closeStream(message->stream);
			return;
		}

		channel = std::make_shared<IncomingDataChannel>(weak_from_this(), sctpTransport);
		channel->assignStream(stream);
		channel->openCallback =
		    weak_bind(&PeerConnection::triggerDataChannel, this, weak_ptr<DataChannel>{channel});

		std::unique_lock lock(mDataChannelsMutex); // we are going to emplace
		mDataChannels.emplace(stream, channel);
	} else if (!found) {
		if (message->type == Message::Reset)
			return; // ignore

		// Invalid, close the DataChannel
		PLOG_WARNING << "Got unexpected message on stream " << stream;
		sctpTransport->closeStream(message->stream);
		return;
	}

	if (message->type == Message::Reset) {
		// Incoming stream is reset, unregister it
		removeDataChannel(stream);
	}

	if (channel) {
		// Forward the message
		channel->incoming(message);
	} else {
		// DataChannel was destroyed, ignore
		PLOG_DEBUG << "Ignored message on stream " << stream << ", DataChannel is destroyed";
	}
}

void PeerConnection::forwardMedia(message_ptr message) {
	if (!message)
		return;

	auto handler = getMediaHandler();

	if (handler) {
		message = handler->incoming(message);
		if (!message)
			return;
	}

	// Browsers like to compound their packets with a random SSRC.
	// we have to do this monstrosity to distribute the report blocks
	if (message->type == Message::Control) {
		std::set<uint32_t> ssrcs;
		size_t offset = 0;
		while ((sizeof(RtcpHeader) + offset) <= message->size()) {
			auto header = reinterpret_cast<RtcpHeader *>(message->data() + offset);
			if (header->lengthInBytes() > message->size() - offset) {
				COUNTER_MEDIA_TRUNCATED++;
				break;
			}
			offset += header->lengthInBytes();
			if (header->payloadType() == 205 || header->payloadType() == 206) {
				auto rtcpfb = reinterpret_cast<RtcpFbHeader *>(header);
				ssrcs.insert(rtcpfb->packetSenderSSRC());
				ssrcs.insert(rtcpfb->mediaSourceSSRC());

			} else if (header->payloadType() == 200) {
				auto rtcpsr = reinterpret_cast<RtcpSr *>(header);
				ssrcs.insert(rtcpsr->senderSSRC());
				for (int i = 0; i < rtcpsr->header.reportCount(); ++i)
					ssrcs.insert(rtcpsr->getReportBlock(i)->getSSRC());
			} else if (header->payloadType() == 201) {
				auto rtcprr = reinterpret_cast<RtcpRr *>(header);
				ssrcs.insert(rtcprr->senderSSRC());
				for (int i = 0; i < rtcprr->header.reportCount(); ++i)
					ssrcs.insert(rtcprr->getReportBlock(i)->getSSRC());
			} else if (header->payloadType() == 202) {
				auto sdes = reinterpret_cast<RtcpSdes *>(header);
				if (!sdes->isValid()) {
					PLOG_WARNING << "RTCP SDES packet is invalid";
					continue;
				}
				for (unsigned int i = 0; i < sdes->chunksCount(); i++) {
					auto chunk = sdes->getChunk(i);
					ssrcs.insert(chunk->ssrc());
				}
			} else {
				// PT=203 == Goodbye
				// PT=204 == Application Specific
				// PT=207 == Extended Report
				if (header->payloadType() != 203 && header->payloadType() != 204 &&
				    header->payloadType() != 207) {
					COUNTER_UNKNOWN_PACKET_TYPE++;
				}
			}
		}

		if (!ssrcs.empty()) {
			std::shared_lock lock(mTracksMutex); // read-only
			for (uint32_t ssrc : ssrcs) {
				if (auto it = mTracksBySsrc.find(ssrc); it != mTracksBySsrc.end()) {
					if (auto track = it->second.lock())
						track->incoming(message);
				}
			}
			return;
		}
	}

	uint32_t ssrc = uint32_t(message->stream);

	std::shared_lock lock(mTracksMutex); // read-only
	if (auto it = mTracksBySsrc.find(ssrc); it != mTracksBySsrc.end()) {
		if (auto track = it->second.lock())
			track->incoming(message);
	} else {
		/*
		 * TODO: So the problem is that when stop sending streams, we stop getting report blocks for
		 * those streams Therefore when we get compound RTCP packets, they are empty, and we can't
		 * forward them. Therefore, it is expected that we don't know where to forward packets. Is
		 * this ideal? No! Do I know how to fix it? No!
		 */
		// PLOG_WARNING << "Track not found for SSRC " << ssrc << ", dropping";
		return;
	}
}

void PeerConnection::forwardBufferedAmount(uint16_t stream, size_t amount) {
	[[maybe_unused]] auto [channel, found] = findDataChannel(stream);
	if (channel)
		channel->triggerBufferedAmount(amount);
}

shared_ptr<DataChannel> PeerConnection::emplaceDataChannel(string label, DataChannelInit init) {
	std::unique_lock lock(mDataChannelsMutex); // we are going to emplace

	// If the DataChannel is user-negotiated, do not negotiate it in-band
	auto channel =
	    init.negotiated
	        ? std::make_shared<DataChannel>(weak_from_this(), std::move(label),
	                                        std::move(init.protocol), std::move(init.reliability))
	        : std::make_shared<OutgoingDataChannel>(weak_from_this(), std::move(label),
	                                                std::move(init.protocol),
	                                                std::move(init.reliability));

	// If the user supplied a stream id, use it, otherwise assign it later
	if (init.id) {
		uint16_t stream = *init.id;
		if (stream > maxDataChannelStream())
			throw std::invalid_argument("DataChannel stream id is too high");

		channel->assignStream(stream);
		mDataChannels.emplace(std::make_pair(stream, channel));

	} else {
		mUnassignedDataChannels.push_back(channel);
	}

	lock.unlock(); // we are going to call assignDataChannels()

	// If SCTP is connected, assign and open now
	auto sctpTransport = std::atomic_load(&mSctpTransport);
	if (sctpTransport && sctpTransport->state() == SctpTransport::State::Connected) {
		assignDataChannels();
		channel->open(sctpTransport);
	}

	return channel;
}

std::pair<shared_ptr<DataChannel>, bool> PeerConnection::findDataChannel(uint16_t stream) {
	std::shared_lock lock(mDataChannelsMutex); // read-only
	if (auto it = mDataChannels.find(stream); it != mDataChannels.end())
		return std::make_pair(it->second.lock(), true);
	else
		return std::make_pair(nullptr, false);
}

bool PeerConnection::removeDataChannel(uint16_t stream) {
	std::unique_lock lock(mDataChannelsMutex); // we are going to erase
	return mDataChannels.erase(stream) != 0;
}

uint16_t PeerConnection::maxDataChannelStream() const {
	auto sctpTransport = std::atomic_load(&mSctpTransport);
	return sctpTransport ? sctpTransport->maxStream() : (MAX_SCTP_STREAMS_COUNT - 1);
}

void PeerConnection::assignDataChannels() {
	std::unique_lock lock(mDataChannelsMutex); // we are going to emplace

	auto iceTransport = std::atomic_load(&mIceTransport);
	if (!iceTransport)
		throw std::logic_error("Attempted to assign DataChannels without ICE transport");

	const uint16_t maxStream = maxDataChannelStream();
	for (auto it = mUnassignedDataChannels.begin(); it != mUnassignedDataChannels.end(); ++it) {
		auto channel = it->lock();
		if (!channel)
			continue;

		// RFC 8832: The peer that initiates opening a data channel selects a stream identifier
		// for which the corresponding incoming and outgoing streams are unused.  If the side is
		// acting as the DTLS client, it MUST choose an even stream identifier; if the side is
		// acting as the DTLS server, it MUST choose an odd one. See
		// https://www.rfc-editor.org/rfc/rfc8832.html#section-6
		uint16_t stream = (iceTransport->role() == Description::Role::Active) ? 0 : 1;
		while (true) {
			if (stream > maxStream)
				throw std::runtime_error("Too many DataChannels");

			if (mDataChannels.find(stream) == mDataChannels.end())
				break;

			stream += 2;
		}

		PLOG_DEBUG << "Assigning stream " << stream << " to DataChannel";

		channel->assignStream(stream);
		mDataChannels.emplace(std::make_pair(stream, channel));
	}

	mUnassignedDataChannels.clear();
}

void PeerConnection::iterateDataChannels(
    std::function<void(shared_ptr<DataChannel> channel)> func) {
	std::vector<shared_ptr<DataChannel>> locked;
	{
		std::shared_lock lock(mDataChannelsMutex); // read-only
		locked.reserve(mDataChannels.size());
		auto it = mDataChannels.begin();
		while (it != mDataChannels.end()) {
			auto channel = it->second.lock();
			if (channel && !channel->isClosed())
				locked.push_back(std::move(channel));

			++it;
		}
	}

	for (auto &channel : locked) {
		try {
			func(std::move(channel));
		} catch (const std::exception &e) {
			PLOG_WARNING << e.what();
		}
	}
}

void PeerConnection::openDataChannels() {
	if (auto transport = std::atomic_load(&mSctpTransport))
		iterateDataChannels([&](shared_ptr<DataChannel> channel) {
			if (!channel->isOpen())
				channel->open(transport);
		});
}

void PeerConnection::closeDataChannels() {
	iterateDataChannels([&](shared_ptr<DataChannel> channel) { channel->close(); });
}

void PeerConnection::remoteCloseDataChannels() {
	iterateDataChannels([&](shared_ptr<DataChannel> channel) { channel->remoteClose(); });
}

shared_ptr<Track> PeerConnection::emplaceTrack(Description::Media description) {
#if !RTC_ENABLE_MEDIA
	// No media support, mark as removed
	PLOG_WARNING << "Tracks are disabled (not compiled with media support)";
	description.markRemoved();
#endif

	shared_ptr<Track> track;
	if (auto it = mTracks.find(description.mid()); it != mTracks.end())
		if (track = it->second.lock(); track)
			track->setDescription(std::move(description));

	if (!track) {
		track = std::make_shared<Track>(weak_from_this(), std::move(description));
		mTracks.emplace(std::make_pair(track->mid(), track));
		mTrackLines.emplace_back(track);
	}

	if (track->description().isRemoved())
		track->close();

	return track;
}

void PeerConnection::iterateTracks(std::function<void(shared_ptr<Track> track)> func) {
	std::shared_lock lock(mTracksMutex); // read-only
	for (auto it = mTrackLines.begin(); it != mTrackLines.end(); ++it) {
		auto track = it->lock();
		if (track && !track->isClosed()) {
			try {
				func(std::move(track));
			} catch (const std::exception &e) {
				PLOG_WARNING << e.what();
			}
		}
	}
}

void PeerConnection::openTracks() {
#if RTC_ENABLE_MEDIA
	if (auto transport = std::atomic_load(&mDtlsTransport)) {
		auto srtpTransport = std::dynamic_pointer_cast<DtlsSrtpTransport>(transport);

		iterateTracks([&](const shared_ptr<Track> &track) {
			if (!track->isOpen()) {
				if (srtpTransport) {
					track->open(srtpTransport);
				} else {
					// A track was added during a latter renegotiation, whereas SRTP transport was
					// not initialized. This is an optimization to use the library with data
					// channels only. Set forceMediaTransport to true to initialize the transport
					// before dynamically adding tracks.
					auto errorMsg = "The connection has no media transport";
					PLOG_ERROR << errorMsg;
					track->triggerError(errorMsg);
				}
			}
		});
	}
#endif
}

void PeerConnection::closeTracks() {
	std::shared_lock lock(mTracksMutex); // read-only
	iterateTracks([&](shared_ptr<Track> track) { track->close(); });
}

void PeerConnection::validateRemoteDescription(const Description &description) {
	if (!description.iceUfrag())
		throw std::invalid_argument("Remote description has no ICE user fragment");

	if (!description.icePwd())
		throw std::invalid_argument("Remote description has no ICE password");

	if (!description.fingerprint())
		throw std::invalid_argument("Remote description has no valid fingerprint");

	if (description.mediaCount() == 0)
		throw std::invalid_argument("Remote description has no media line");

	int activeMediaCount = 0;
	for (unsigned int i = 0; i < description.mediaCount(); ++i)
		std::visit(rtc::overloaded{[&](const Description::Application *application) {
			                           if (!application->isRemoved())
				                           ++activeMediaCount;
		                           },
		                           [&](const Description::Media *media) {
			                           if (!media->isRemoved() ||
			                               media->direction() != Description::Direction::Inactive)
				                           ++activeMediaCount;
		                           }},
		           description.media(i));

	if (activeMediaCount == 0)
		throw std::invalid_argument("Remote description has no active media");

	if (auto local = localDescription(); local && local->iceUfrag() && local->icePwd())
		if (*description.iceUfrag() == *local->iceUfrag() &&
		    *description.icePwd() == *local->icePwd())
			throw std::logic_error("Got the local description as remote description");

	PLOG_VERBOSE << "Remote description looks valid";
}

void PeerConnection::processLocalDescription(Description description) {
	const uint16_t localSctpPort = DEFAULT_SCTP_PORT;
	const size_t localMaxMessageSize =
	    config.maxMessageSize.value_or(DEFAULT_LOCAL_MAX_MESSAGE_SIZE);

	// Clean up the application entry the ICE transport might have added already (libnice)
	description.clearMedia();

	if (auto remote = remoteDescription()) {
		// Reciprocate remote description
		for (unsigned int i = 0; i < remote->mediaCount(); ++i)
			std::visit( // reciprocate each media
			    rtc::overloaded{
			        [&](Description::Application *remoteApp) {
				        std::shared_lock lock(mDataChannelsMutex);
				        if (!mDataChannels.empty() || !mUnassignedDataChannels.empty()) {
					        // Prefer local description
					        Description::Application app(remoteApp->mid());
					        app.setSctpPort(localSctpPort);
					        app.setMaxMessageSize(localMaxMessageSize);

					        PLOG_DEBUG << "Adding application to local description, mid=\""
					                   << app.mid() << "\"";

					        description.addMedia(std::move(app));
					        return;
				        }

				        auto reciprocated = remoteApp->reciprocate();
				        reciprocated.hintSctpPort(localSctpPort);
				        reciprocated.setMaxMessageSize(localMaxMessageSize);

				        PLOG_DEBUG << "Reciprocating application in local description, mid=\""
				                   << reciprocated.mid() << "\"";

				        description.addMedia(std::move(reciprocated));
			        },
			        [&](Description::Media *remoteMedia) {
				        std::shared_lock lock(mTracksMutex);
				        if (auto it = mTracks.find(remoteMedia->mid()); it != mTracks.end()) {
					        // Prefer local description
					        if (auto track = it->second.lock()) {
						        auto media = track->description();

						        PLOG_DEBUG << "Adding media to local description, mid=\""
						                   << media.mid() << "\", removed=" << std::boolalpha
						                   << media.isRemoved();

						        description.addMedia(std::move(media));

					        } else {
						        auto reciprocated = remoteMedia->reciprocate();
						        reciprocated.markRemoved();

						        PLOG_DEBUG << "Adding media to local description, mid=\""
						                   << reciprocated.mid()
						                   << "\", removed=true (track is destroyed)";

						        description.addMedia(std::move(reciprocated));
					        }
					        return;
				        }

				        auto reciprocated = remoteMedia->reciprocate();
#if !RTC_ENABLE_MEDIA
				        if (!reciprocated.isRemoved()) {
					        // No media support, mark as removed
					        PLOG_WARNING << "Rejecting track (not compiled with media support)";
					        reciprocated.markRemoved();
				        }
#endif

				        PLOG_DEBUG << "Reciprocating media in local description, mid=\""
				                   << reciprocated.mid() << "\", removed=" << std::boolalpha
				                   << reciprocated.isRemoved();

				        // Create incoming track
				        auto track =
				            std::make_shared<Track>(weak_from_this(), std::move(reciprocated));
				        mTracks.emplace(std::make_pair(track->mid(), track));
				        mTrackLines.emplace_back(track);
				        triggerTrack(track); // The user may modify the track description

				        if (track->description().isRemoved())
					        track->close();

				        description.addMedia(track->description());
			        },
			    },
			    remote->media(i));

		// We need to update the SSRC cache for newly-created incoming tracks
		updateTrackSsrcCache(*remote);
	}

	if (description.type() == Description::Type::Offer) {
		// This is an offer, add locally created data channels and tracks
		// Add media for local tracks
		std::shared_lock lock(mTracksMutex);
		for (auto it = mTrackLines.begin(); it != mTrackLines.end(); ++it) {
			if (auto track = it->lock()) {
				if (description.hasMid(track->mid()))
					continue;

				auto media = track->description();

				PLOG_DEBUG << "Adding media to local description, mid=\"" << media.mid()
				           << "\", removed=" << std::boolalpha << media.isRemoved();

				description.addMedia(std::move(media));
			}
		}

		// Add application for data channels
		if (!description.hasApplication()) {
			std::shared_lock lock(mDataChannelsMutex);
			if (!mDataChannels.empty() || !mUnassignedDataChannels.empty()) {
				// Prevents mid collision with remote or local tracks
				unsigned int m = 0;
				while (description.hasMid(std::to_string(m)))
					++m;

				Description::Application app(std::to_string(m));
				app.setSctpPort(localSctpPort);
				app.setMaxMessageSize(localMaxMessageSize);

				PLOG_DEBUG << "Adding application to local description, mid=\"" << app.mid()
				           << "\"";

				description.addMedia(std::move(app));
			}
		}

		// There might be no media at this point if the user created a Track, deleted it,
		// then called setLocalDescription().
		if (description.mediaCount() == 0)
			throw std::runtime_error("No DataChannel or Track to negotiate");
	}

	// Set local fingerprint (wait for certificate if necessary)
	description.setFingerprint(mCertificate.get()->fingerprint());

	PLOG_VERBOSE << "Issuing local description: " << description;

	if (description.mediaCount() == 0)
		throw std::logic_error("Local description has no media line");

	updateTrackSsrcCache(description);

	{
		// Set as local description
		std::lock_guard lock(mLocalDescriptionMutex);

		std::vector<Candidate> existingCandidates;
		if (mLocalDescription) {
			existingCandidates = mLocalDescription->extractCandidates();
			mCurrentLocalDescription.emplace(std::move(*mLocalDescription));
		}

		mLocalDescription.emplace(description);
		mLocalDescription->addCandidates(std::move(existingCandidates));
	}

	mProcessor.enqueue(&PeerConnection::trigger<Description>, shared_from_this(),
	                   &localDescriptionCallback, std::move(description));

	// Reciprocated tracks might need to be open
	if (auto dtlsTransport = std::atomic_load(&mDtlsTransport);
	    dtlsTransport && dtlsTransport->state() == Transport::State::Connected)
		mProcessor.enqueue(&PeerConnection::openTracks, shared_from_this());
}

void PeerConnection::processLocalCandidate(Candidate candidate) {
	std::lock_guard lock(mLocalDescriptionMutex);
	if (!mLocalDescription)
		throw std::logic_error("Got a local candidate without local description");

	if (config.iceTransportPolicy == TransportPolicy::Relay &&
	    candidate.type() != Candidate::Type::Relayed) {
		PLOG_VERBOSE << "Not issuing local candidate because of transport policy: " << candidate;
		return;
	}

	PLOG_VERBOSE << "Issuing local candidate: " << candidate;

	candidate.resolve(Candidate::ResolveMode::Simple);
	mLocalDescription->addCandidate(candidate);

	mProcessor.enqueue(&PeerConnection::trigger<Candidate>, shared_from_this(),
	                   &localCandidateCallback, std::move(candidate));
}

void PeerConnection::processRemoteDescription(Description description) {
	// Update the SSRC cache for existing tracks
	updateTrackSsrcCache(description);

	{
		// Set as remote description
		std::lock_guard lock(mRemoteDescriptionMutex);

		std::vector<Candidate> existingCandidates;
		if (mRemoteDescription)
			existingCandidates = mRemoteDescription->extractCandidates();

		mRemoteDescription.emplace(description);
		mRemoteDescription->addCandidates(std::move(existingCandidates));
	}

	if (description.hasApplication()) {
		auto dtlsTransport = std::atomic_load(&mDtlsTransport);
		auto sctpTransport = std::atomic_load(&mSctpTransport);
		if (!sctpTransport && dtlsTransport &&
		    dtlsTransport->state() == Transport::State::Connected)
			initSctpTransport();
	} else {
		mProcessor.enqueue(&PeerConnection::remoteCloseDataChannels, shared_from_this());
	}
}

void PeerConnection::processRemoteCandidate(Candidate candidate) {
	auto iceTransport = std::atomic_load(&mIceTransport);
	{
		// Set as remote candidate
		std::lock_guard lock(mRemoteDescriptionMutex);
		if (!mRemoteDescription)
			throw std::logic_error("Got a remote candidate without remote description");

		if (!iceTransport)
			throw std::logic_error("Got a remote candidate without ICE transport");

		candidate.hintMid(mRemoteDescription->bundleMid());

		if (mRemoteDescription->hasCandidate(candidate))
			return; // already in description, ignore

		candidate.resolve(Candidate::ResolveMode::Simple);
		mRemoteDescription->addCandidate(candidate);
	}

	if (candidate.isResolved()) {
		iceTransport->addRemoteCandidate(std::move(candidate));
	} else {
		// We might need a lookup, do it asynchronously
		// We don't use the thread pool because we have no control on the timeout
		if ((iceTransport = std::atomic_load(&mIceTransport))) {
			weak_ptr<IceTransport> weakIceTransport{iceTransport};
			std::thread t([weakIceTransport, candidate = std::move(candidate)]() mutable {
				utils::this_thread::set_name("RTC resolver");
				if (candidate.resolve(Candidate::ResolveMode::Lookup))
					if (auto iceTransport = weakIceTransport.lock())
						iceTransport->addRemoteCandidate(std::move(candidate));
			});
			t.detach();
		}
	}
}

string PeerConnection::localBundleMid() const {
	std::lock_guard lock(mLocalDescriptionMutex);
	return mLocalDescription ? mLocalDescription->bundleMid() : "0";
}

void PeerConnection::setMediaHandler(shared_ptr<MediaHandler> handler) {
	std::unique_lock lock(mMediaHandlerMutex);
	if (mMediaHandler)
		mMediaHandler->onOutgoing(nullptr);
	mMediaHandler = handler;
}

shared_ptr<MediaHandler> PeerConnection::getMediaHandler() {
	std::shared_lock lock(mMediaHandlerMutex);
	return mMediaHandler;
}

void PeerConnection::triggerDataChannel(weak_ptr<DataChannel> weakDataChannel) {
	auto dataChannel = weakDataChannel.lock();
	if (dataChannel) {
		dataChannel->resetOpenCallback(); // might be set internally
		mPendingDataChannels.push(std::move(dataChannel));
	}
	triggerPendingDataChannels();
}

void PeerConnection::triggerTrack(weak_ptr<Track> weakTrack) {
	auto track = weakTrack.lock();
	if (track) {
		track->resetOpenCallback(); // might be set internally
		mPendingTracks.push(std::move(track));
	}
	triggerPendingTracks();
}

void PeerConnection::triggerPendingDataChannels() {
	while (dataChannelCallback) {
		auto next = mPendingDataChannels.pop();
		if (!next)
			break;

		auto impl = std::move(*next);

		try {
			dataChannelCallback(std::make_shared<rtc::DataChannel>(impl));
		} catch (const std::exception &e) {
			PLOG_WARNING << "Uncaught exception in callback: " << e.what();
		}

		impl->triggerOpen();
	}
}

void PeerConnection::triggerPendingTracks() {
	while (trackCallback) {
		auto next = mPendingTracks.pop();
		if (!next)
			break;

		auto impl = std::move(*next);

		try {
			trackCallback(std::make_shared<rtc::Track>(impl));
		} catch (const std::exception &e) {
			PLOG_WARNING << "Uncaught exception in callback: " << e.what();
		}

		// Do not trigger open immediately for tracks as it'll be done later
	}
}

void PeerConnection::flushPendingDataChannels() {
	mProcessor.enqueue(&PeerConnection::triggerPendingDataChannels, shared_from_this());
}

void PeerConnection::flushPendingTracks() {
	mProcessor.enqueue(&PeerConnection::triggerPendingTracks, shared_from_this());
}

bool PeerConnection::changeState(State newState) {
	State current;
	do {
		current = state.load();
		if (current == State::Closed)
			return false;
		if (current == newState)
			return false;

	} while (!state.compare_exchange_weak(current, newState));

	std::ostringstream s;
	s << newState;
	PLOG_INFO << "Changed state to " << s.str();

	if (newState == State::Closed) {
		auto callback = std::move(stateChangeCallback); // steal the callback
		callback(State::Closed);                        // call it synchronously
	} else {
		mProcessor.enqueue(&PeerConnection::trigger<State>, shared_from_this(),
		                   &stateChangeCallback, newState);
	}
	return true;
}

bool PeerConnection::changeIceState(IceState newState) {
	if (iceState.exchange(newState) == newState)
		return false;

	std::ostringstream s;
	s << newState;
	PLOG_INFO << "Changed ICE state to " << s.str();

	if (newState == IceState::Closed) {
		auto callback = std::move(iceStateChangeCallback); // steal the callback
		callback(IceState::Closed);                        // call it synchronously
	} else {
		mProcessor.enqueue(&PeerConnection::trigger<IceState>, shared_from_this(),
		                   &iceStateChangeCallback, newState);
	}
	return true;
}

bool PeerConnection::changeGatheringState(GatheringState newState) {
	if (gatheringState.exchange(newState) == newState)
		return false;

	std::ostringstream s;
	s << newState;
	PLOG_INFO << "Changed gathering state to " << s.str();
	mProcessor.enqueue(&PeerConnection::trigger<GatheringState>, shared_from_this(),
	                   &gatheringStateChangeCallback, newState);

	return true;
}

bool PeerConnection::changeSignalingState(SignalingState newState) {
	if (signalingState.exchange(newState) == newState)
		return false;

	std::ostringstream s;
	s << newState;
	PLOG_INFO << "Changed signaling state to " << s.str();
	mProcessor.enqueue(&PeerConnection::trigger<SignalingState>, shared_from_this(),
	                   &signalingStateChangeCallback, newState);

	return true;
}

void PeerConnection::resetCallbacks() {
	// Unregister all callbacks
	dataChannelCallback = nullptr;
	localDescriptionCallback = nullptr;
	localCandidateCallback = nullptr;
	stateChangeCallback = nullptr;
	iceStateChangeCallback = nullptr;
	gatheringStateChangeCallback = nullptr;
	signalingStateChangeCallback = nullptr;
	trackCallback = nullptr;
}

void PeerConnection::updateTrackSsrcCache(const Description &description) {
	std::unique_lock lock(mTracksMutex); // for safely writing to mTracksBySsrc

	// Setup SSRC -> Track mapping
	for (unsigned int i = 0; i < description.mediaCount(); ++i)
		std::visit( // ssrc -> track mapping
		    rtc::overloaded{
		        [&](Description::Application const *) { return; },
		        [&](Description::Media const *media) {
			        const auto ssrcs = media->getSSRCs();

			        // Note: We don't want to lock (or do any other lookups), if we
			        // already know there's no SSRCs to loop over.
			        if (ssrcs.size() <= 0) {
				        return;
			        }

			        std::shared_ptr<Track> track{nullptr};
			        if (auto it = mTracks.find(media->mid()); it != mTracks.end())
				        if (auto track_for_mid = it->second.lock())
					        track = track_for_mid;

			        if (!track) {
				        // Unable to find track for MID
				        return;
			        }

			        for (auto ssrc : ssrcs) {
				        mTracksBySsrc.insert_or_assign(ssrc, track);
			        }
		        },
		    },
		    description.media(i));
}

} // namespace rtc::impl
