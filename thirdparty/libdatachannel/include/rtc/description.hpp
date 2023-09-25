/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 * Copyright (c) 2020 Staz Modrzynski
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_DESCRIPTION_H
#define RTC_DESCRIPTION_H

#include "candidate.hpp"
#include "common.hpp"

#include <iostream>
#include <map>
#include <vector>

namespace rtc {

const string DEFAULT_OPUS_AUDIO_PROFILE =
    "minptime=10;maxaveragebitrate=96000;stereo=1;sprop-stereo=1;useinbandfec=1";

// Use Constrained Baseline profile Level 3.1 (necessary for Firefox)
// https://developer.mozilla.org/en-US/docs/Web/Media/Formats/WebRTC_codecs#Supported_video_codecs
// TODO: Should be 42E0 but 42C0 appears to be more compatible. Investigate this.
const string DEFAULT_H264_VIDEO_PROFILE =
    "profile-level-id=42e01f;packetization-mode=1;level-asymmetry-allowed=1";

class RTC_CPP_EXPORT Description {
public:
	enum class Type { Unspec, Offer, Answer, Pranswer, Rollback };
	enum class Role { ActPass, Passive, Active };

	enum class Direction {
		SendOnly = RTC_DIRECTION_SENDONLY,
		RecvOnly = RTC_DIRECTION_RECVONLY,
		SendRecv = RTC_DIRECTION_SENDRECV,
		Inactive = RTC_DIRECTION_INACTIVE,
		Unknown = RTC_DIRECTION_UNKNOWN
	};

	Description(const string &sdp, Type type = Type::Unspec, Role role = Role::ActPass);
	Description(const string &sdp, string typeString);

	Type type() const;
	string typeString() const;
	Role role() const;
	string bundleMid() const;
	std::vector<string> iceOptions() const;
	optional<string> iceUfrag() const;
	optional<string> icePwd() const;
	optional<string> fingerprint() const;
	bool ended() const;

	void hintType(Type type);
	void setFingerprint(string fingerprint);
	void addIceOption(string option);
	void removeIceOption(const string &option);

	std::vector<string> attributes() const;
	void addAttribute(string attr);
	void removeAttribute(const string &attr);

	std::vector<Candidate> candidates() const;
	std::vector<Candidate> extractCandidates();
	bool hasCandidate(const Candidate &candidate) const;
	void addCandidate(Candidate candidate);
	void addCandidates(std::vector<Candidate> candidates);
	void endCandidates();

	operator string() const;
	string generateSdp(string_view eol = "\r\n") const;
	string generateApplicationSdp(string_view eol = "\r\n") const;

	class RTC_CPP_EXPORT Entry {
	public:
		virtual ~Entry() = default;

		virtual string type() const;
		virtual string description() const;
		virtual string mid() const;

		Direction direction() const;
		void setDirection(Direction dir);

		bool isRemoved() const;
		void markRemoved();

		std::vector<string> attributes() const;
		void addAttribute(string attr);
		void removeAttribute(const string &attr);
		void addRid(string rid);

		struct RTC_CPP_EXPORT ExtMap {
			static int parseId(string_view description);

			ExtMap(int id, string uri, Direction direction = Direction::Unknown);
			ExtMap(string_view description);

			void setDescription(string_view description);

			int id;
			string uri;
			string attributes;
			Direction direction = Direction::Unknown;
		};

		std::vector<int> extIds();
		ExtMap *extMap(int id);
		void addExtMap(ExtMap map);
		void removeExtMap(int id);

		operator string() const;
		string generateSdp(string_view eol = "\r\n", string_view addr = "0.0.0.0",
		                   uint16_t port = 9) const;

		virtual void parseSdpLine(string_view line);

	protected:
		Entry(const string &mline, string mid, Direction dir = Direction::Unknown);

		virtual string generateSdpLines(string_view eol) const;

		std::vector<string> mAttributes;
		std::map<int, ExtMap> mExtMaps;

	private:
		string mType;
		string mDescription;
		string mMid;
		std::vector<string> mRids;
		Direction mDirection;
		bool mIsRemoved;
	};

	struct RTC_CPP_EXPORT Application : public Entry {
	public:
		Application(string mid = "data");
		Application(const string &mline, string mid);
		virtual ~Application() = default;

		string description() const override;
		Application reciprocate() const;

		void setSctpPort(uint16_t port);
		void hintSctpPort(uint16_t port);
		void setMaxMessageSize(size_t size);

		optional<uint16_t> sctpPort() const;
		optional<size_t> maxMessageSize() const;

		virtual void parseSdpLine(string_view line) override;

	private:
		virtual string generateSdpLines(string_view eol) const override;

		optional<uint16_t> mSctpPort;
		optional<size_t> mMaxMessageSize;
	};

	// Media (non-data)
	class RTC_CPP_EXPORT Media : public Entry {
	public:
		Media(const string &sdp);
		Media(const string &mline, string mid, Direction dir = Direction::SendOnly);
		virtual ~Media() = default;

		string description() const override;
		Media reciprocate() const;

		void addSSRC(uint32_t ssrc, optional<string> name, optional<string> msid = nullopt,
		             optional<string> trackId = nullopt);
		void removeSSRC(uint32_t ssrc);
		void replaceSSRC(uint32_t old, uint32_t ssrc, optional<string> name,
		                 optional<string> msid = nullopt, optional<string> trackID = nullopt);
		bool hasSSRC(uint32_t ssrc) const;
		void clearSSRCs();
		std::vector<uint32_t> getSSRCs() const;
		optional<std::string> getCNameForSsrc(uint32_t ssrc) const;

		int bitrate() const;
		void setBitrate(int bitrate);

		struct RTC_CPP_EXPORT RtpMap {
			static int parsePayloadType(string_view description);

			explicit RtpMap(int payloadType);
			RtpMap(string_view description);

			void setDescription(string_view description);

			void addFeedback(string fb);
			void removeFeedback(const string &str);
			void addParameter(string p);
			void removeParameter(const string &str);

			int payloadType;
			string format;
			int clockRate;
			string encParams;

			std::vector<string> rtcpFbs;
			std::vector<string> fmtps;
		};

		bool hasPayloadType(int payloadType) const;
		std::vector<int> payloadTypes() const;
		RtpMap *rtpMap(int payloadType);
		void addRtpMap(RtpMap map);
		void removeRtpMap(int payloadType);
		void removeFormat(const string &format);

		void addRtxCodec(int payloadType, int origPayloadType, unsigned int clockRate);

		virtual void parseSdpLine(string_view line) override;

	private:
		virtual string generateSdpLines(string_view eol) const override;

		int mBas = -1;

		std::map<int, RtpMap> mRtpMaps;
		std::vector<uint32_t> mSsrcs;
		std::map<uint32_t, string> mCNameMap;
	};

	class RTC_CPP_EXPORT Audio : public Media {
	public:
		Audio(string mid = "audio", Direction dir = Direction::SendOnly);

		void addAudioCodec(int payloadType, string codec, optional<string> profile = std::nullopt);

		void addOpusCodec(int payloadType, optional<string> profile = DEFAULT_OPUS_AUDIO_PROFILE);
		void addPCMACodec(int payloadType, optional<string> profile = std::nullopt);
		void addPCMUCodec(int payloadType, optional<string> profile = std::nullopt);
		void addAacCodec(int payloadType, optional<string> profile = std::nullopt);
	};

	class RTC_CPP_EXPORT Video : public Media {
	public:
		Video(string mid = "video", Direction dir = Direction::SendOnly);

		void addVideoCodec(int payloadType, string codec, optional<string> profile = std::nullopt);

		void addH264Codec(int payloadType, optional<string> profile = DEFAULT_H264_VIDEO_PROFILE);
		void addH265Codec(int payloadType, optional<string> profile = std::nullopt);
		void addVP8Codec(int payloadType, optional<string> profile = std::nullopt);
		void addVP9Codec(int payloadType, optional<string> profile = std::nullopt);
		void addAV1Codec(int payloadType, optional<string> profile = std::nullopt);
	};

	bool hasApplication() const;
	bool hasAudioOrVideo() const;
	bool hasMid(string_view mid) const;

	int addMedia(Media media);
	int addMedia(Application application);
	int addApplication(string mid = "data");
	int addVideo(string mid = "video", Direction dir = Direction::SendOnly);
	int addAudio(string mid = "audio", Direction dir = Direction::SendOnly);
	void clearMedia();

	variant<Media *, Application *> media(unsigned int index);
	variant<const Media *, const Application *> media(unsigned int index) const;
	unsigned int mediaCount() const;

	const Application *application() const;
	Application *application();

	static Type stringToType(const string &typeString);
	static string typeToString(Type type);

private:
	optional<Candidate> defaultCandidate() const;
	shared_ptr<Entry> createEntry(string mline, string mid, Direction dir);
	void removeApplication();

	Type mType;

	// Session-level attributes
	Role mRole;
	string mUsername;
	string mSessionId;
	std::vector<string> mIceOptions;
	optional<string> mIceUfrag, mIcePwd;
	optional<string> mFingerprint;
	std::vector<string> mAttributes; // other attributes

	// Entries
	std::vector<shared_ptr<Entry>> mEntries;
	shared_ptr<Application> mApplication;

	// Candidates
	std::vector<Candidate> mCandidates;
	bool mEnded = false;
};

} // namespace rtc

RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, const rtc::Description &description);
RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, rtc::Description::Type type);
RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, rtc::Description::Role role);
RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, const rtc::Description::Direction &direction);

#endif
