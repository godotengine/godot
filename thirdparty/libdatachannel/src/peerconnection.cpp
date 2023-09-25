/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "peerconnection.hpp"
#include "common.hpp"
#include "rtp.hpp"

#include "impl/certificate.hpp"
#include "impl/dtlstransport.hpp"
#include "impl/icetransport.hpp"
#include "impl/internals.hpp"
#include "impl/peerconnection.hpp"
#include "impl/sctptransport.hpp"
#include "impl/threadpool.hpp"
#include "impl/track.hpp"

#if RTC_ENABLE_MEDIA
#include "impl/dtlssrtptransport.hpp"
#endif

#include <iomanip>
#include <set>
#include <thread>

using namespace std::placeholders;

namespace rtc {

PeerConnection::PeerConnection() : PeerConnection(Configuration()) {}

PeerConnection::PeerConnection(Configuration config)
    : CheshireCat<impl::PeerConnection>(std::move(config)) {}

PeerConnection::~PeerConnection() {
	try {
		impl()->remoteClose();
	} catch (const std::exception &e) {
		PLOG_ERROR << e.what();
	}
}

void PeerConnection::close() { impl()->close(); }

const Configuration *PeerConnection::config() const { return &impl()->config; }

PeerConnection::State PeerConnection::state() const { return impl()->state; }

PeerConnection::IceState PeerConnection::iceState() const { return impl()->iceState; }

PeerConnection::GatheringState PeerConnection::gatheringState() const {
	return impl()->gatheringState;
}

PeerConnection::SignalingState PeerConnection::signalingState() const {
	return impl()->signalingState;
}

optional<Description> PeerConnection::localDescription() const {
	return impl()->localDescription();
}

optional<Description> PeerConnection::remoteDescription() const {
	return impl()->remoteDescription();
}

bool PeerConnection::hasMedia() const {
	auto local = localDescription();
	return local && local->hasAudioOrVideo();
}

void PeerConnection::setLocalDescription(Description::Type type) {
	std::unique_lock signalingLock(impl()->signalingMutex);
	PLOG_VERBOSE << "Setting local description, type=" << Description::typeToString(type);

	SignalingState signalingState = impl()->signalingState.load();
	if (type == Description::Type::Rollback) {
		if (signalingState == SignalingState::HaveLocalOffer ||
		    signalingState == SignalingState::HaveLocalPranswer) {
			impl()->rollbackLocalDescription();
			impl()->changeSignalingState(SignalingState::Stable);
		}
		return;
	}

	// Guess the description type if unspecified
	if (type == Description::Type::Unspec) {
		if (signalingState == SignalingState::HaveRemoteOffer)
			type = Description::Type::Answer;
		else
			type = Description::Type::Offer;
	}

	// Only a local offer resets the negotiation needed flag
	if (type == Description::Type::Offer && !impl()->negotiationNeeded.exchange(false)) {
		PLOG_DEBUG << "No negotiation needed";
		return;
	}

	// Get the new signaling state
	SignalingState newSignalingState;
	switch (signalingState) {
	case SignalingState::Stable:
		if (type != Description::Type::Offer) {
			std::ostringstream oss;
			oss << "Unexpected local desciption type " << type << " in signaling state "
			    << signalingState;
			throw std::logic_error(oss.str());
		}
		newSignalingState = SignalingState::HaveLocalOffer;
		break;

	case SignalingState::HaveRemoteOffer:
	case SignalingState::HaveLocalPranswer:
		if (type != Description::Type::Answer && type != Description::Type::Pranswer) {
			std::ostringstream oss;
			oss << "Unexpected local description type " << type
			    << " description in signaling state " << signalingState;
			throw std::logic_error(oss.str());
		}
		newSignalingState = SignalingState::Stable;
		break;

	default: {
		std::ostringstream oss;
		oss << "Unexpected local description in signaling state " << signalingState << ", ignoring";
		LOG_WARNING << oss.str();
		return;
	}
	}

	auto iceTransport = impl()->initIceTransport();
	if (!iceTransport)
		return; // closed

	Description local = iceTransport->getLocalDescription(type);
	impl()->processLocalDescription(std::move(local));

	impl()->changeSignalingState(newSignalingState);
	signalingLock.unlock();

	if (impl()->gatheringState == GatheringState::New) {
		iceTransport->gatherLocalCandidates(impl()->localBundleMid());
	}
}

void PeerConnection::setRemoteDescription(Description description) {
	std::unique_lock signalingLock(impl()->signalingMutex);
	PLOG_VERBOSE << "Setting remote description: " << string(description);

	if (description.type() == Description::Type::Rollback) {
		// This is mostly useless because we accept any offer
		PLOG_VERBOSE << "Rolling back pending remote description";
		impl()->changeSignalingState(SignalingState::Stable);
		return;
	}

	impl()->validateRemoteDescription(description);

	// Get the new signaling state
	SignalingState signalingState = impl()->signalingState.load();
	SignalingState newSignalingState;
	switch (signalingState) {
	case SignalingState::Stable:
		description.hintType(Description::Type::Offer);
		if (description.type() != Description::Type::Offer) {
			std::ostringstream oss;
			oss << "Unexpected remote " << description.type() << " description in signaling state "
			    << signalingState;
			throw std::logic_error(oss.str());
		}
		newSignalingState = SignalingState::HaveRemoteOffer;
		break;

	case SignalingState::HaveLocalOffer:
		description.hintType(Description::Type::Answer);
		if (description.type() == Description::Type::Offer) {
			// The ICE agent will automatically initiate a rollback when a peer that had previously
			// created an offer receives an offer from the remote peer
			impl()->rollbackLocalDescription();
			impl()->changeSignalingState(SignalingState::Stable);
			signalingState = SignalingState::Stable;
			newSignalingState = SignalingState::HaveRemoteOffer;
			break;
		}
		if (description.type() != Description::Type::Answer &&
		    description.type() != Description::Type::Pranswer) {
			std::ostringstream oss;
			oss << "Unexpected remote " << description.type() << " description in signaling state "
			    << signalingState;
			throw std::logic_error(oss.str());
		}
		newSignalingState = SignalingState::Stable;
		break;

	case SignalingState::HaveRemotePranswer:
		description.hintType(Description::Type::Answer);
		if (description.type() != Description::Type::Answer &&
		    description.type() != Description::Type::Pranswer) {
			std::ostringstream oss;
			oss << "Unexpected remote " << description.type() << " description in signaling state "
			    << signalingState;
			throw std::logic_error(oss.str());
		}
		newSignalingState = SignalingState::Stable;
		break;

	default: {
		std::ostringstream oss;
		oss << "Unexpected remote description in signaling state " << signalingState;
		throw std::logic_error(oss.str());
	}
	}

	// Candidates will be added at the end, extract them for now
	auto remoteCandidates = description.extractCandidates();
	auto type = description.type();

	auto iceTransport = impl()->initIceTransport();
	if (!iceTransport)
		return; // closed

	iceTransport->setRemoteDescription(description); // ICE transport might reject the description

	impl()->processRemoteDescription(std::move(description));
	impl()->changeSignalingState(newSignalingState);
	signalingLock.unlock();

	if (type == Description::Type::Offer) {
		// This is an offer, we need to answer
		if (!impl()->config.disableAutoNegotiation)
			setLocalDescription(Description::Type::Answer);
	}

	for (const auto &candidate : remoteCandidates)
		addRemoteCandidate(candidate);
}

void PeerConnection::addRemoteCandidate(Candidate candidate) {
	std::unique_lock signalingLock(impl()->signalingMutex);
	PLOG_VERBOSE << "Adding remote candidate: " << string(candidate);
	impl()->processRemoteCandidate(std::move(candidate));
}

void PeerConnection::setMediaHandler(shared_ptr<MediaHandler> handler) {
	impl()->setMediaHandler(std::move(handler));
};

shared_ptr<MediaHandler> PeerConnection::getMediaHandler() { return impl()->getMediaHandler(); };

optional<string> PeerConnection::localAddress() const {
	auto iceTransport = impl()->getIceTransport();
	return iceTransport ? iceTransport->getLocalAddress() : nullopt;
}

optional<string> PeerConnection::remoteAddress() const {
	auto iceTransport = impl()->getIceTransport();
	return iceTransport ? iceTransport->getRemoteAddress() : nullopt;
}

uint16_t PeerConnection::maxDataChannelId() const { return impl()->maxDataChannelStream(); }

shared_ptr<DataChannel> PeerConnection::createDataChannel(string label, DataChannelInit init) {
	auto channelImpl = impl()->emplaceDataChannel(std::move(label), std::move(init));
	auto channel = std::make_shared<DataChannel>(channelImpl);

	// Renegotiation is needed iff the current local description does not have application
	auto local = impl()->localDescription();
	if (!local || !local->hasApplication())
		impl()->negotiationNeeded = true;

	if (!impl()->config.disableAutoNegotiation)
		setLocalDescription();

	return channel;
}

void PeerConnection::onDataChannel(
    std::function<void(shared_ptr<DataChannel> dataChannel)> callback) {
	impl()->dataChannelCallback = callback;
	impl()->flushPendingDataChannels();
}

std::shared_ptr<Track> PeerConnection::addTrack(Description::Media description) {
	auto trackImpl = impl()->emplaceTrack(std::move(description));
	auto track = std::make_shared<Track>(trackImpl);

	// Renegotiation is needed for the new or updated track
	impl()->negotiationNeeded = true;

	return track;
}

void PeerConnection::onTrack(std::function<void(std::shared_ptr<Track>)> callback) {
	impl()->trackCallback = callback;
	impl()->flushPendingTracks();
}

void PeerConnection::onLocalDescription(std::function<void(Description description)> callback) {
	impl()->localDescriptionCallback = callback;
}

void PeerConnection::onLocalCandidate(std::function<void(Candidate candidate)> callback) {
	impl()->localCandidateCallback = callback;
}

void PeerConnection::onStateChange(std::function<void(State state)> callback) {
	impl()->stateChangeCallback = callback;
}

void PeerConnection::onIceStateChange(std::function<void(IceState state)> callback) {
	impl()->iceStateChangeCallback = callback;
}

void PeerConnection::onGatheringStateChange(std::function<void(GatheringState state)> callback) {
	impl()->gatheringStateChangeCallback = callback;
}

void PeerConnection::onSignalingStateChange(std::function<void(SignalingState state)> callback) {
	impl()->signalingStateChangeCallback = callback;
}

void PeerConnection::resetCallbacks() { impl()->resetCallbacks(); }

bool PeerConnection::getSelectedCandidatePair(Candidate *local, Candidate *remote) {
	auto iceTransport = impl()->getIceTransport();
	return iceTransport ? iceTransport->getSelectedCandidatePair(local, remote) : false;
}

void PeerConnection::clearStats() {
	if (auto sctpTransport = impl()->getSctpTransport())
		return sctpTransport->clearStats();
}

size_t PeerConnection::bytesSent() {
	auto sctpTransport = impl()->getSctpTransport();
	return sctpTransport ? sctpTransport->bytesSent() : 0;
}

size_t PeerConnection::bytesReceived() {
	auto sctpTransport = impl()->getSctpTransport();
	return sctpTransport ? sctpTransport->bytesReceived() : 0;
}

optional<std::chrono::milliseconds> PeerConnection::rtt() {
	auto sctpTransport = impl()->getSctpTransport();
	return sctpTransport ? sctpTransport->rtt() : nullopt;
}

} // namespace rtc

std::ostream &operator<<(std::ostream &out, rtc::PeerConnection::State state) {
	using State = rtc::PeerConnection::State;
	const char *str;
	switch (state) {
	case State::New:
		str = "new";
		break;
	case State::Connecting:
		str = "connecting";
		break;
	case State::Connected:
		str = "connected";
		break;
	case State::Disconnected:
		str = "disconnected";
		break;
	case State::Failed:
		str = "failed";
		break;
	case State::Closed:
		str = "closed";
		break;
	default:
		str = "unknown";
		break;
	}
	return out << str;
}

std::ostream &operator<<(std::ostream &out, rtc::PeerConnection::IceState state) {
	using IceState = rtc::PeerConnection::IceState;
	const char *str;
	switch (state) {
	case IceState::New:
		str = "new";
		break;
	case IceState::Checking:
		str = "checking";
		break;
	case IceState::Connected:
		str = "connected";
		break;
	case IceState::Completed:
		str = "completed";
		break;
	case IceState::Failed:
		str = "failed";
		break;
	case IceState::Disconnected:
		str = "disconnected";
		break;
	case IceState::Closed:
		str = "closed";
		break;
	default:
		str = "unknown";
		break;
	}
	return out << str;
}

std::ostream &operator<<(std::ostream &out, rtc::PeerConnection::GatheringState state) {
	using GatheringState = rtc::PeerConnection::GatheringState;
	const char *str;
	switch (state) {
	case GatheringState::New:
		str = "new";
		break;
	case GatheringState::InProgress:
		str = "in-progress";
		break;
	case GatheringState::Complete:
		str = "complete";
		break;
	default:
		str = "unknown";
		break;
	}
	return out << str;
}

std::ostream &operator<<(std::ostream &out, rtc::PeerConnection::SignalingState state) {
	using SignalingState = rtc::PeerConnection::SignalingState;
	const char *str;
	switch (state) {
	case SignalingState::Stable:
		str = "stable";
		break;
	case SignalingState::HaveLocalOffer:
		str = "have-local-offer";
		break;
	case SignalingState::HaveRemoteOffer:
		str = "have-remote-offer";
		break;
	case SignalingState::HaveLocalPranswer:
		str = "have-local-pranswer";
		break;
	case SignalingState::HaveRemotePranswer:
		str = "have-remote-pranswer";
		break;
	default:
		str = "unknown";
		break;
	}
	return out << str;
}
