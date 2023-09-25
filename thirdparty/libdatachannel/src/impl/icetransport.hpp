/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_ICE_TRANSPORT_H
#define RTC_IMPL_ICE_TRANSPORT_H

#include "candidate.hpp"
#include "common.hpp"
#include "configuration.hpp"
#include "description.hpp"
#include "global.hpp"
#include "peerconnection.hpp"
#include "transport.hpp"

#if !USE_NICE
#include <juice/juice.h>
#else
#include <nice/agent.h>
#endif

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace rtc::impl {

class IceTransport : public Transport {
public:
	static void Init();
	static void Cleanup();

	enum class GatheringState { New = 0, InProgress = 1, Complete = 2 };

	using candidate_callback = std::function<void(const Candidate &candidate)>;
	using gathering_state_callback = std::function<void(GatheringState state)>;

	IceTransport(const Configuration &config, candidate_callback candidateCallback,
	             state_callback stateChangeCallback,
	             gathering_state_callback gatheringStateChangeCallback);
	~IceTransport();

	Description::Role role() const;
	GatheringState gatheringState() const;
	Description getLocalDescription(Description::Type type) const;
	void setRemoteDescription(const Description &description);
	bool addRemoteCandidate(const Candidate &candidate);
	void gatherLocalCandidates(string mid);

	optional<string> getLocalAddress() const;
	optional<string> getRemoteAddress() const;

	bool send(message_ptr message) override; // false if dropped

	bool getSelectedCandidatePair(Candidate *local, Candidate *remote);

private:
	bool outgoing(message_ptr message) override;

	void changeGatheringState(GatheringState state);

	void processStateChange(unsigned int state);
	void processCandidate(const string &candidate);
	void processGatheringDone();
	void processTimeout();

	Description::Role mRole;
	string mMid;
	std::chrono::milliseconds mTrickleTimeout;
	std::atomic<GatheringState> mGatheringState;

	candidate_callback mCandidateCallback;
	gathering_state_callback mGatheringStateChangeCallback;

#if !USE_NICE
	unique_ptr<juice_agent_t, void (*)(juice_agent_t *)> mAgent;

	static void StateChangeCallback(juice_agent_t *agent, juice_state_t state, void *user_ptr);
	static void CandidateCallback(juice_agent_t *agent, const char *sdp, void *user_ptr);
	static void GatheringDoneCallback(juice_agent_t *agent, void *user_ptr);
	static void RecvCallback(juice_agent_t *agent, const char *data, size_t size, void *user_ptr);
	static void LogCallback(juice_log_level_t level, const char *message);
#else
	static unique_ptr<GMainLoop, void (*)(GMainLoop *)> MainLoop;
	static std::thread MainLoopThread;

	unique_ptr<NiceAgent, void (*)(NiceAgent *)> mNiceAgent;
	uint32_t mStreamId = 0;
	guint mTimeoutId = 0;
	std::mutex mOutgoingMutex;
	unsigned int mOutgoingDscp;

	static string AddressToString(const NiceAddress &addr);

	static void CandidateCallback(NiceAgent *agent, NiceCandidate *candidate, gpointer userData);
	static void GatheringDoneCallback(NiceAgent *agent, guint streamId, gpointer userData);
	static void StateChangeCallback(NiceAgent *agent, guint streamId, guint componentId,
	                                guint state, gpointer userData);
	static void RecvCallback(NiceAgent *agent, guint stream_id, guint component_id, guint len,
	                         gchar *buf, gpointer userData);
	static gboolean TimeoutCallback(gpointer userData);
	static void LogCallback(const gchar *log_domain, GLogLevelFlags log_level, const gchar *message,
	                        gpointer user_data);
#endif
};

} // namespace rtc::impl

#endif
