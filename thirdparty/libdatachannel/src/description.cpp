/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 * Copyright (c) 2020 Staz Modrzynski
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "description.hpp"

#include "impl/internals.hpp"
#include "impl/utils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>

using std::chrono::system_clock;

namespace {

using std::string;
using std::string_view;

inline bool match_prefix(string_view str, string_view prefix) {
	return str.size() >= prefix.size() &&
	       std::mismatch(prefix.begin(), prefix.end(), str.begin()).first == prefix.end();
}

inline void trim_begin(string &str) {
	str.erase(str.begin(),
	          std::find_if(str.begin(), str.end(), [](char c) { return !std::isspace(c); }));
}

inline void trim_end(string &str) {
	str.erase(
	    std::find_if(str.rbegin(), str.rend(), [](char c) { return !std::isspace(c); }).base(),
	    str.end());
}

inline std::pair<string_view, string_view> parse_pair(string_view attr) {
	string_view key, value;
	if (size_t separator = attr.find(':'); separator != string::npos) {
		key = attr.substr(0, separator);
		value = attr.substr(separator + 1);
	} else {
		key = attr;
	}
	return std::make_pair(std::move(key), std::move(value));
}

template <typename T> T to_integer(string_view s) {
	const string str(s);
	try {
		return std::is_signed<T>::value ? T(std::stol(str)) : T(std::stoul(str));
	} catch (...) {
		throw std::invalid_argument("Invalid integer \"" + str + "\" in description");
	}
}

inline bool is_sha256_fingerprint(string_view f) {
	if (f.size() != 32 * 3 - 1)
		return false;

	for (size_t i = 0; i < f.size(); ++i) {
		if (i % 3 == 2) {
			if (f[i] != ':')
				return false;
		} else {
			if (!std::isxdigit(f[i]))
				return false;
		}
	}
	return true;
}

} // namespace

namespace rtc {

namespace utils = impl::utils;

Description::Description(const string &sdp, Type type, Role role)
    : mType(Type::Unspec), mRole(role) {
	hintType(type);

	int index = -1;
	shared_ptr<Entry> current;
	std::istringstream ss(sdp);
	while (ss) {
		string line;
		std::getline(ss, line);
		trim_end(line);
		if (line.empty())
			continue;

		if (match_prefix(line, "m=")) { // Media description line (aka m-line)
			current = createEntry(line.substr(2), std::to_string(++index), Direction::Unknown);

		} else if (match_prefix(line, "o=")) { // Origin line
			std::istringstream origin(line.substr(2));
			origin >> mUsername >> mSessionId;

		} else if (match_prefix(line, "a=")) { // Attribute line
			string attr = line.substr(2);
			auto [key, value] = parse_pair(attr);

			if (key == "setup") {
				if (value == "active")
					mRole = Role::Active;
				else if (value == "passive")
					mRole = Role::Passive;
				else
					mRole = Role::ActPass;

			} else if (key == "fingerprint") {
				if (match_prefix(value, "sha-256 ") || match_prefix(value, "SHA-256 ")) {
					string fingerprint{value.substr(8)};
					trim_begin(fingerprint);
					setFingerprint(std::move(fingerprint));
				} else {
					PLOG_WARNING << "Unknown SDP fingerprint format: " << value;
				}
			} else if (key == "ice-ufrag") {
				mIceUfrag = value;
			} else if (key == "ice-pwd") {
				mIcePwd = value;
			} else if (key == "ice-options") {
				mIceOptions = utils::explode(string(value), ',');
			} else if (key == "candidate") {
				addCandidate(Candidate(attr, bundleMid()));
			} else if (key == "end-of-candidates") {
				mEnded = true;
			} else if (current) {
				current->parseSdpLine(std::move(line));
			} else {
				mAttributes.emplace_back(attr);
			}

		} else if (current) {
			current->parseSdpLine(std::move(line));
		}
	}

	if (mUsername.empty())
		mUsername = "rtc";

	if (mSessionId.empty()) {
		auto uniform = std::bind(std::uniform_int_distribution<uint32_t>(), utils::random_engine());
		mSessionId = std::to_string(uniform());
	}
}

Description::Description(const string &sdp, string typeString)
    : Description(sdp, !typeString.empty() ? stringToType(typeString) : Type::Unspec,
                  Role::ActPass) {}

Description::Type Description::type() const { return mType; }

string Description::typeString() const { return typeToString(mType); }

Description::Role Description::role() const { return mRole; }

string Description::bundleMid() const {
	// Get the mid of the first non-removed media
	for (const auto &entry : mEntries)
		if (!entry->isRemoved())
			return entry->mid();

	return "0";
}

optional<string> Description::iceUfrag() const { return mIceUfrag; }

std::vector<string> Description::iceOptions() const { return mIceOptions; }

optional<string> Description::icePwd() const { return mIcePwd; }

optional<string> Description::fingerprint() const { return mFingerprint; }

bool Description::ended() const { return mEnded; }

void Description::hintType(Type type) {
	if (mType == Type::Unspec)
		mType = type;
}

void Description::setFingerprint(string fingerprint) {
	if (!is_sha256_fingerprint(fingerprint))
		throw std::invalid_argument("Invalid SHA256 fingerprint \"" + fingerprint + "\"");

	std::transform(fingerprint.begin(), fingerprint.end(), fingerprint.begin(),
	               [](char c) { return char(std::toupper(c)); });
	mFingerprint.emplace(std::move(fingerprint));
}

void Description::addIceOption(string option) {
	if (std::find(mIceOptions.begin(), mIceOptions.end(), option) == mIceOptions.end())
		mIceOptions.emplace_back(std::move(option));
}

void Description::removeIceOption(const string &option) {
	mIceOptions.erase(std::remove(mIceOptions.begin(), mIceOptions.end(), option),
	                  mIceOptions.end());
}

std::vector<string> Description::Entry::attributes() const { return mAttributes; }

void Description::Entry::addAttribute(string attr) {
	if (std::find(mAttributes.begin(), mAttributes.end(), attr) == mAttributes.end())
		mAttributes.emplace_back(std::move(attr));
}

void Description::Entry::removeAttribute(const string &attr) {
	mAttributes.erase(
	    std::remove_if(mAttributes.begin(), mAttributes.end(),
	                   [&](const auto &a) { return a == attr || parse_pair(a).first == attr; }),
	    mAttributes.end());
}

std::vector<Candidate> Description::candidates() const { return mCandidates; }

std::vector<Candidate> Description::extractCandidates() {
	std::vector<Candidate> result;
	std::swap(mCandidates, result);
	mEnded = false;
	return result;
}

bool Description::hasCandidate(const Candidate &candidate) const {
	return std::find(mCandidates.begin(), mCandidates.end(), candidate) != mCandidates.end();
}

void Description::addCandidate(Candidate candidate) {
	candidate.hintMid(bundleMid());

	if (!hasCandidate(candidate))
		mCandidates.emplace_back(std::move(candidate));
}

void Description::addCandidates(std::vector<Candidate> candidates) {
	for (Candidate candidate : candidates)
		addCandidate(std::move(candidate));
}

void Description::endCandidates() { mEnded = true; }

Description::operator string() const { return generateSdp("\r\n"); }

string Description::generateSdp(string_view eol) const {
	std::ostringstream sdp;

	// Header
	sdp << "v=0" << eol;
	sdp << "o=" << mUsername << " " << mSessionId << " 0 IN IP4 127.0.0.1" << eol;
	sdp << "s=-" << eol;
	sdp << "t=0 0" << eol;

	// BUNDLE (RFC 8843 Negotiating Media Multiplexing Using the Session Description Protocol)
	// https://www.rfc-editor.org/rfc/rfc8843.html
	std::ostringstream bundleGroup;
	for (const auto &entry : mEntries)
		if (!entry->isRemoved())
			bundleGroup << ' ' << entry->mid();

	if (!bundleGroup.str().empty())
		sdp << "a=group:BUNDLE" << bundleGroup.str() << eol;

	// Lip-sync
	std::ostringstream lsGroup;
	for (const auto &entry : mEntries)
		if (!entry->isRemoved() && entry != mApplication)
			lsGroup << ' ' << entry->mid();

	if (!lsGroup.str().empty())
		sdp << "a=group:LS" << lsGroup.str() << eol;

	// Session-level attributes
	sdp << "a=msid-semantic:WMS *" << eol;
	sdp << "a=setup:" << mRole << eol;

	if (mIceUfrag)
		sdp << "a=ice-ufrag:" << *mIceUfrag << eol;
	if (mIcePwd)
		sdp << "a=ice-pwd:" << *mIcePwd << eol;
	if (!mIceOptions.empty())
		sdp << "a=ice-options:" << utils::implode(mIceOptions, ',') << eol;
	if (mFingerprint)
		sdp << "a=fingerprint:sha-256 " << *mFingerprint << eol;

	for (const auto &attr : mAttributes)
		sdp << "a=" << attr << eol;

	auto cand = defaultCandidate();
	const string addr = cand && cand->isResolved()
	                        ? (string(cand->family() == Candidate::Family::Ipv6 ? "IP6" : "IP4") +
	                           " " + *cand->address())
	                        : "IP4 0.0.0.0";
	const uint16_t port =
	    cand && cand->isResolved() ? *cand->port() : 9; // Port 9 is the discard protocol

	// Entries
	bool first = true;
	for (const auto &entry : mEntries) {
		sdp << entry->generateSdp(eol, addr, port);

		if (!entry->isRemoved() && std::exchange(first, false)) {
			// Candidates
			for (const auto &candidate : mCandidates)
				sdp << string(candidate) << eol;

			if (mEnded)
				sdp << "a=end-of-candidates" << eol;
		}
	}

	return sdp.str();
}

string Description::generateApplicationSdp(string_view eol) const {
	std::ostringstream sdp;

	// Header
	sdp << "v=0" << eol;
	sdp << "o=" << mUsername << " " << mSessionId << " 0 IN IP4 127.0.0.1" << eol;
	sdp << "s=-" << eol;
	sdp << "t=0 0" << eol;

	auto cand = defaultCandidate();
	const string addr = cand && cand->isResolved()
	                        ? (string(cand->family() == Candidate::Family::Ipv6 ? "IP6" : "IP4") +
	                           " " + *cand->address())
	                        : "IP4 0.0.0.0";
	const uint16_t port =
	    cand && cand->isResolved() ? *cand->port() : 9; // Port 9 is the discard protocol

	// Application
	auto app = mApplication ? mApplication : std::make_shared<Application>();
	sdp << app->generateSdp(eol, addr, port);

	// Session-level attributes
	sdp << "a=msid-semantic:WMS *" << eol;
	sdp << "a=setup:" << mRole << eol;

	if (mIceUfrag)
		sdp << "a=ice-ufrag:" << *mIceUfrag << eol;
	if (mIcePwd)
		sdp << "a=ice-pwd:" << *mIcePwd << eol;
	if (!mIceOptions.empty())
		sdp << "a=ice-options:" << utils::implode(mIceOptions, ',') << eol;
	if (mFingerprint)
		sdp << "a=fingerprint:sha-256 " << *mFingerprint << eol;

	for (const auto &attr : mAttributes)
		sdp << "a=" << attr << eol;

	// Candidates
	for (const auto &candidate : mCandidates)
		sdp << string(candidate) << eol;

	if (mEnded)
		sdp << "a=end-of-candidates" << eol;

	return sdp.str();
}

optional<Candidate> Description::defaultCandidate() const {
	// Return the first host candidate with highest priority, favoring IPv4
	optional<Candidate> result;
	for (const auto &c : mCandidates) {
		if (c.type() == Candidate::Type::Host) {
			if (!result ||
			    (result->family() == Candidate::Family::Ipv6 &&
			     c.family() == Candidate::Family::Ipv4) ||
			    (result->family() == c.family() && result->priority() < c.priority()))
				result.emplace(c);
		}
	}
	return result;
}

shared_ptr<Description::Entry> Description::createEntry(string mline, string mid, Direction dir) {
	string type = mline.substr(0, mline.find(' '));
	if (type == "application") {
		removeApplication();
		mApplication = std::make_shared<Application>(mline, std::move(mid));
		mEntries.emplace_back(mApplication);
		return mApplication;
	} else {
		auto media = std::make_shared<Media>(std::move(mline), std::move(mid), dir);
		mEntries.emplace_back(media);
		return media;
	}
}

void Description::removeApplication() {
	if (!mApplication)
		return;

	auto it = std::find(mEntries.begin(), mEntries.end(), mApplication);
	if (it != mEntries.end())
		mEntries.erase(it);

	mApplication.reset();
}

bool Description::hasApplication() const { return mApplication && !mApplication->isRemoved(); }

bool Description::hasAudioOrVideo() const {
	for (auto entry : mEntries)
		if (entry != mApplication && !entry->isRemoved())
			return true;

	return false;
}

bool Description::hasMid(string_view mid) const {
	for (const auto &entry : mEntries)
		if (entry->mid() == mid)
			return true;

	return false;
}

int Description::addMedia(Media media) {
	mEntries.emplace_back(std::make_shared<Media>(std::move(media)));
	return int(mEntries.size()) - 1;
}

int Description::addMedia(Application application) {
	removeApplication();
	mApplication = std::make_shared<Application>(std::move(application));
	mEntries.emplace_back(mApplication);
	return int(mEntries.size()) - 1;
}

int Description::addApplication(string mid) { return addMedia(Application(std::move(mid))); }

const Description::Application *Description::application() const { return mApplication.get(); }

Description::Application *Description::application() { return mApplication.get(); }

int Description::addVideo(string mid, Direction dir) {
	return addMedia(Video(std::move(mid), dir));
}

int Description::addAudio(string mid, Direction dir) {
	return addMedia(Audio(std::move(mid), dir));
}

void Description::clearMedia() {
	mEntries.clear();
	mApplication.reset();
}

variant<Description::Media *, Description::Application *> Description::media(unsigned int index) {
	if (index >= mEntries.size())
		throw std::out_of_range("Media index out of range");

	const auto &entry = mEntries[index];
	if (entry == mApplication) {
		auto result = dynamic_cast<Application *>(entry.get());
		if (!result)
			throw std::logic_error("Bad type of application in description");

		return result;

	} else {
		auto result = dynamic_cast<Media *>(entry.get());
		if (!result)
			throw std::logic_error("Bad type of media in description");

		return result;
	}
}

variant<const Description::Media *, const Description::Application *>
Description::media(unsigned int index) const {
	if (index >= mEntries.size())
		throw std::out_of_range("Media index out of range");

	const auto &entry = mEntries[index];
	if (entry == mApplication) {
		auto result = dynamic_cast<Application *>(entry.get());
		if (!result)
			throw std::logic_error("Bad type of application in description");

		return result;

	} else {
		auto result = dynamic_cast<Media *>(entry.get());
		if (!result)
			throw std::logic_error("Bad type of media in description");

		return result;
	}
}

unsigned int Description::mediaCount() const { return unsigned(mEntries.size()); }

Description::Entry::Entry(const string &mline, string mid, Direction dir)
    : mMid(std::move(mid)), mDirection(dir) {

	uint16_t port;
	std::istringstream ss(mline);
	ss >> mType;
	ss >> port;
	ss >> mDescription;

	// RFC 3264: Existing media streams are removed by creating a new SDP with the port number for
	// that stream set to zero.
	// RFC 8843: If the offerer assigns a zero port value to a bundled "m=" section, but does not
	// include an SDP 'bundle-only' attribute in the "m=" section, it is an indication that the
	// offerer wants to disable the "m=" section.
	mIsRemoved = (port == 0);
}

string Description::Entry::type() const { return mType; }

string Description::Entry::description() const { return mDescription; }

string Description::Entry::mid() const { return mMid; }

Description::Direction Description::Entry::direction() const { return mDirection; }

void Description::Entry::setDirection(Direction dir) { mDirection = dir; }

bool Description::Entry::isRemoved() const { return mIsRemoved; }

void Description::Entry::markRemoved() { mIsRemoved = true; }

std::vector<string> Description::attributes() const { return mAttributes; }

void Description::addAttribute(string attr) {
	if (std::find(mAttributes.begin(), mAttributes.end(), attr) == mAttributes.end())
		mAttributes.emplace_back(std::move(attr));
}

void Description::Entry::addRid(string rid) { mRids.emplace_back(rid); }

void Description::removeAttribute(const string &attr) {
	mAttributes.erase(
	    std::remove_if(mAttributes.begin(), mAttributes.end(),
	                   [&](const auto &a) { return a == attr || parse_pair(a).first == attr; }),
	    mAttributes.end());
}

std::vector<int> Description::Entry::extIds() {
	std::vector<int> result;
	for (auto it = mExtMaps.begin(); it != mExtMaps.end(); ++it)
		result.push_back(it->first);

	return result;
}

Description::Entry::ExtMap *Description::Entry::extMap(int id) {
	auto it = mExtMaps.find(id);
	if (it == mExtMaps.end())
		throw std::invalid_argument("extmap not found");

	return &it->second;
}

void Description::Entry::addExtMap(ExtMap map) {
	auto id = map.id;
	mExtMaps.emplace(id, std::move(map));
}

void Description::Entry::removeExtMap(int id) { mExtMaps.erase(id); }

Description::Entry::operator string() const { return generateSdp("\r\n", "IP4 0.0.0.0", 9); }

string Description::Entry::generateSdp(string_view eol, string_view addr, uint16_t port) const {
	std::ostringstream sdp;
	// RFC 3264: Existing media streams are removed by creating a new SDP with the port number for
	// that stream set to zero. [...] A stream that is offered with a port of zero MUST be marked
	// with port zero in the answer.
	sdp << "m=" << type() << ' ' << (mIsRemoved ? 0 : port) << ' ' << description() << eol;
	sdp << "c=IN " << addr << eol;
	sdp << generateSdpLines(eol);

	return sdp.str();
}

string Description::Entry::generateSdpLines(string_view eol) const {
	std::ostringstream sdp;
	sdp << "a=mid:" << mMid << eol;

	for (auto it = mExtMaps.begin(); it != mExtMaps.end(); ++it) {
		auto &map = it->second;

		sdp << "a=extmap:" << map.id;
		if (map.direction != Direction::Unknown)
			sdp << '/' << map.direction;

		sdp << ' ' << map.uri;
		if (!map.attributes.empty())
			sdp << ' ' << map.attributes;

		sdp << eol;
	}

	if (mDirection != Direction::Unknown)
		sdp << "a=" << mDirection << eol;

	for (const auto &attr : mAttributes) {
		if (mRids.size() != 0 && match_prefix(attr, "ssrc:")) {
			continue;
		}

		sdp << "a=" << attr << eol;
	}

	for (const auto &rid : mRids) {
		sdp << "a=rid:" << rid << " send" << eol;
	}

	if (mRids.size() != 0) {
		sdp << "a=simulcast:send ";

		bool first = true;
		for (const auto &rid : mRids) {
			if (first) {
				first = false;
			} else {
				sdp << ";";
			}

			sdp << rid;
		}

		sdp << eol;
	}

	return sdp.str();
}

void Description::Entry::parseSdpLine(string_view line) {
	if (match_prefix(line, "a=")) {
		string_view attr = line.substr(2);
		auto [key, value] = parse_pair(attr);

		if (key == "mid") {
			mMid = value;
		} else if (key == "extmap") {
			auto id = Description::Media::ExtMap::parseId(value);
			auto it = mExtMaps.find(id);
			if (it == mExtMaps.end())
				it = mExtMaps.insert(std::make_pair(id, Description::Media::ExtMap(value))).first;
			else
				it->second.setDescription(value);

		} else if (attr == "sendonly")
			mDirection = Direction::SendOnly;
		else if (attr == "recvonly")
			mDirection = Direction::RecvOnly;
		else if (key == "sendrecv")
			mDirection = Direction::SendRecv;
		else if (key == "inactive")
			mDirection = Direction::Inactive;
		else if (key == "bundle-only") {
			// RFC 8843: When an offerer generates a subsequent offer, in which it wants to disable
			// a bundled "m=" section from a BUNDLE group, the offerer [...] MUST NOT assign an SDP
			// 'bundle-only' attribute to the "m=" section.
			mIsRemoved = false;
		} else {
			mAttributes.emplace_back(attr);
		}
	}
}

int Description::Entry::ExtMap::parseId(string_view description) {
	size_t p = description.find(' ');
	return to_integer<int>(description.substr(0, p));
}

Description::Entry::ExtMap::ExtMap(int id, string uri, Direction direction) {
	this->id = id;
	this->uri = std::move(uri);
	this->direction = direction;
}

Description::Entry::ExtMap::ExtMap(string_view description) { setDescription(description); }

void Description::Entry::ExtMap::setDescription(string_view description) {
	const size_t uriStart = description.find(' ');
	if (uriStart == string::npos)
		throw std::invalid_argument("Invalid description for extmap");

	const string_view idAndDirection = description.substr(0, uriStart);
	const size_t idSplit = idAndDirection.find('/');
	if (idSplit == string::npos) {
		this->id = to_integer<int>(idAndDirection);
	} else {
		this->id = to_integer<int>(idAndDirection.substr(0, idSplit));

		const string_view directionStr = idAndDirection.substr(idSplit + 1);
		if (directionStr == "sendonly")
			this->direction = Direction::SendOnly;
		else if (directionStr == "recvonly")
			this->direction = Direction::RecvOnly;
		else if (directionStr == "sendrecv")
			this->direction = Direction::SendRecv;
		else if (directionStr == "inactive")
			this->direction = Direction::Inactive;
		else
			throw std::invalid_argument("Invalid direction for extmap");
	}

	const string_view uriAndAttributes = description.substr(uriStart + 1);
	const size_t attributeSplit = uriAndAttributes.find(' ');

	if (attributeSplit == string::npos)
		this->uri = uriAndAttributes;
	else {
		this->uri = uriAndAttributes.substr(0, attributeSplit);
		this->attributes = uriAndAttributes.substr(attributeSplit + 1);
	}
}

void Description::Media::addSSRC(uint32_t ssrc, optional<string> name, optional<string> msid,
                                 optional<string> trackId) {
	if (name) {
		mAttributes.emplace_back("ssrc:" + std::to_string(ssrc) + " cname:" + *name);
		mCNameMap.emplace(ssrc, *name);
	} else {
		mAttributes.emplace_back("ssrc:" + std::to_string(ssrc));
	}

	if (msid) {
		mAttributes.emplace_back("ssrc:" + std::to_string(ssrc) + " msid:" + *msid + " " +
		                         trackId.value_or(*msid));
		mAttributes.emplace_back("msid:" + *msid + " " + trackId.value_or(*msid));
	}

	mSsrcs.emplace_back(ssrc);
}

void Description::Media::removeSSRC(uint32_t ssrc) {
	string prefix = "ssrc:" + std::to_string(ssrc);
	mAttributes.erase(std::remove_if(mAttributes.begin(), mAttributes.end(),
	                                 [&](const auto &a) { return match_prefix(a, prefix); }),
	                  mAttributes.end());

	mSsrcs.erase(std::remove(mSsrcs.begin(), mSsrcs.end(), ssrc), mSsrcs.end());
}

void Description::Media::replaceSSRC(uint32_t old, uint32_t ssrc, optional<string> name,
                                     optional<string> msid, optional<string> trackID) {
	removeSSRC(old);
	addSSRC(ssrc, std::move(name), std::move(msid), std::move(trackID));
}

bool Description::Media::hasSSRC(uint32_t ssrc) const {
	return std::find(mSsrcs.begin(), mSsrcs.end(), ssrc) != mSsrcs.end();
}

void Description::Media::clearSSRCs() {
	auto it = mAttributes.begin();
	while (it != mAttributes.end()) {
		if (match_prefix(*it, "ssrc:"))
			it = mAttributes.erase(it);
		else
			++it;
	}

	mSsrcs.clear();
	mCNameMap.clear();
}

std::vector<uint32_t> Description::Media::getSSRCs() const { return mSsrcs; }

optional<string> Description::Media::getCNameForSsrc(uint32_t ssrc) const {
	auto it = mCNameMap.find(ssrc);
	if (it != mCNameMap.end()) {
		return it->second;
	}
	return nullopt;
}

Description::Application::Application(string mid)
    : Entry("application 9 UDP/DTLS/SCTP", std::move(mid), Direction::SendRecv) {}

Description::Application::Application(const string &mline, string mid)
    : Entry(mline, std::move(mid), Direction::SendRecv) {}

string Description::Application::description() const {
	return Entry::description() + " webrtc-datachannel";
}

Description::Application Description::Application::reciprocate() const {
	Application reciprocated(*this);

	reciprocated.mMaxMessageSize.reset();

	return reciprocated;
}

void Description::Application::setSctpPort(uint16_t port) { mSctpPort = port; }

void Description::Application::hintSctpPort(uint16_t port) { mSctpPort = mSctpPort.value_or(port); }

void Description::Application::setMaxMessageSize(size_t size) { mMaxMessageSize = size; }

optional<uint16_t> Description::Application::sctpPort() const { return mSctpPort; }

optional<size_t> Description::Application::maxMessageSize() const { return mMaxMessageSize; }

string Description::Application::generateSdpLines(string_view eol) const {
	std::ostringstream sdp;
	sdp << Entry::generateSdpLines(eol);

	if (mSctpPort)
		sdp << "a=sctp-port:" << *mSctpPort << eol;

	if (mMaxMessageSize)
		sdp << "a=max-message-size:" << *mMaxMessageSize << eol;

	return sdp.str();
}

void Description::Application::parseSdpLine(string_view line) {
	if (match_prefix(line, "a=")) {
		string_view attr = line.substr(2);
		auto [key, value] = parse_pair(attr);

		if (key == "sctp-port") {
			mSctpPort = to_integer<uint16_t>(value);
		} else if (key == "max-message-size") {
			mMaxMessageSize = to_integer<size_t>(value);
		} else {
			Entry::parseSdpLine(line);
		}
	} else {
		Entry::parseSdpLine(line);
	}
}

Description::Media::Media(const string &sdp) : Entry(sdp, "", Direction::Unknown) {
	std::istringstream ss(sdp);
	while (ss) {
		string line;
		std::getline(ss, line);
		trim_end(line);
		if (line.empty())
			continue;

		parseSdpLine(line);
	}

	if (mid().empty())
		throw std::invalid_argument("Missing mid in media description");
}

Description::Media::Media(const string &mline, string mid, Direction dir)
    : Entry(mline, std::move(mid), dir) {}

string Description::Media::description() const {
	std::ostringstream desc;
	desc << Entry::description();
	for (auto it = mRtpMaps.begin(); it != mRtpMaps.end(); ++it)
		desc << ' ' << it->first;

	return desc.str();
}

Description::Media Description::Media::reciprocate() const {
	Media reciprocated(*this);

	// Invert direction
	switch (reciprocated.direction()) {
	case Direction::RecvOnly:
		reciprocated.setDirection(Direction::SendOnly);
		break;
	case Direction::SendOnly:
		reciprocated.setDirection(Direction::RecvOnly);
		break;
	default:
		// We are good
		break;
	}

	// Invert directions of extmap
	auto &extMaps = reciprocated.mExtMaps;
	for (auto it = extMaps.begin(); it != extMaps.end(); ++it) {
		auto &map = it->second;
		switch (map.direction) {
		case Direction::RecvOnly:
			map.direction = Direction::SendOnly;
			break;
		case Direction::SendOnly:
			map.direction = Direction::RecvOnly;
			break;
		default:
			// We are good
			break;
		}
	}

	// Clear sent SSRCs
	reciprocated.clearSSRCs();

	// Remove rtcp-rsize attribute as Reduced-Size RTCP is not supported (see RFC 5506)
	reciprocated.removeAttribute("rtcp-rsize");

	return reciprocated;
}

int Description::Media::bitrate() const { return mBas; }

void Description::Media::setBitrate(int bitrate) { mBas = bitrate; }

bool Description::Media::hasPayloadType(int payloadType) const {
	return mRtpMaps.find(payloadType) != mRtpMaps.end();
}

std::vector<int> Description::Media::payloadTypes() const {
	std::vector<int> result;
	result.reserve(mRtpMaps.size());
	for (auto it = mRtpMaps.begin(); it != mRtpMaps.end(); ++it)
		result.push_back(it->first);

	return result;
}

Description::Media::RtpMap *Description::Media::rtpMap(int payloadType) {
	auto it = mRtpMaps.find(payloadType);
	if (it == mRtpMaps.end())
		throw std::invalid_argument("rtpmap not found");

	return &it->second;
}

void Description::Media::addRtpMap(RtpMap map) {
	auto payloadType = map.payloadType;
	mRtpMaps.emplace(payloadType, std::move(map));
}

void Description::Media::removeRtpMap(int payloadType) {
	// Remove the actual format
	mRtpMaps.erase(payloadType);

	// Remove any other rtpmaps that depend on the format we just removed
	auto it = mRtpMaps.begin();
	while (it != mRtpMaps.end()) {
		const auto &fmtps = it->second.fmtps;
		if (std::find(fmtps.begin(), fmtps.end(), "apt=" + std::to_string(payloadType)) !=
		    fmtps.end())
			it = mRtpMaps.erase(it);
		else
			++it;
	}
}

void Description::Media::removeFormat(const string &format) {
	std::vector<int> payloadTypes;
	for (const auto &it : mRtpMaps) {
		if (it.second.format == format)
			payloadTypes.push_back(it.first);
	}
	for (int pt : payloadTypes)
		removeRtpMap(pt);
}

void Description::Media::addRtxCodec(int payloadType, int origPayloadType, unsigned int clockRate) {
	RtpMap rtp(std::to_string(payloadType) + " RTX/" + std::to_string(clockRate));
	rtp.fmtps.emplace_back("apt=" + std::to_string(origPayloadType));
	addRtpMap(rtp);
}

string Description::Media::generateSdpLines(string_view eol) const {
	std::ostringstream sdp;
	if (mBas >= 0)
		sdp << "b=AS:" << mBas << eol;

	sdp << Entry::generateSdpLines(eol);
	sdp << "a=rtcp-mux" << eol;

	for (auto it = mRtpMaps.begin(); it != mRtpMaps.end(); ++it) {
		auto &map = it->second;

		// Create the a=rtpmap
		sdp << "a=rtpmap:" << map.payloadType << ' ' << map.format << '/' << map.clockRate;
		if (!map.encParams.empty())
			sdp << '/' << map.encParams;

		sdp << eol;

		for (const auto &val : map.rtcpFbs)
			sdp << "a=rtcp-fb:" << map.payloadType << ' ' << val << eol;

		for (const auto &val : map.fmtps)
			sdp << "a=fmtp:" << map.payloadType << ' ' << val << eol;
	}

	return sdp.str();
}

void Description::Media::parseSdpLine(string_view line) {
	if (match_prefix(line, "a=")) {
		string_view attr = line.substr(2);
		auto [key, value] = parse_pair(attr);

		if (key == "rtpmap") {
			auto pt = Description::Media::RtpMap::parsePayloadType(value);
			auto it = mRtpMaps.find(pt);
			if (it == mRtpMaps.end())
				it = mRtpMaps.insert(std::make_pair(pt, Description::Media::RtpMap(value))).first;
			else
				it->second.setDescription(value);

		} else if (key == "rtcp-fb") {
			size_t p = value.find(' ');
			int pt = to_integer<int>(value.substr(0, p));
			auto it = mRtpMaps.find(pt);
			if (it == mRtpMaps.end())
				it = mRtpMaps.insert(std::make_pair(pt, Description::Media::RtpMap(pt))).first;

			it->second.rtcpFbs.emplace_back(value.substr(p + 1));

		} else if (key == "fmtp") {
			size_t p = value.find(' ');
			int pt = to_integer<int>(value.substr(0, p));
			auto it = mRtpMaps.find(pt);
			if (it == mRtpMaps.end())
				it = mRtpMaps.insert(std::make_pair(pt, Description::Media::RtpMap(pt))).first;

			it->second.fmtps.emplace_back(value.substr(p + 1));

		} else if (key == "rtcp-mux") {
			// always added

		} else if (key == "ssrc") {
			auto ssrc = to_integer<uint32_t>(value);
			if (!hasSSRC(ssrc))
				mSsrcs.emplace_back(ssrc);

			auto cnamePos = value.find("cname:");
			if (cnamePos != string::npos) {
				auto cname = value.substr(cnamePos + 6);
				mCNameMap.emplace(ssrc, cname);
			}
			mAttributes.emplace_back(attr);

		} else {
			Entry::parseSdpLine(line);
		}

	} else if (match_prefix(line, "b=AS")) {
		mBas = to_integer<int>(line.substr(line.find(':') + 1));
	} else {
		Entry::parseSdpLine(line);
	}
}

Description::Media::RtpMap::RtpMap(int payloadType) {
	this->payloadType = payloadType;
	this->clockRate = 0;
}

int Description::Media::RtpMap::parsePayloadType(string_view mline) {
	size_t p = mline.find(' ');
	return to_integer<int>(mline.substr(0, p));
}

Description::Media::RtpMap::RtpMap(string_view description) { setDescription(description); }

void Description::Media::RtpMap::setDescription(string_view description) {
	size_t p = description.find(' ');
	if (p == string::npos)
		throw std::invalid_argument("Invalid format description for rtpmap");

	this->payloadType = to_integer<int>(description.substr(0, p));

	string_view line = description.substr(p + 1);
	size_t spl = line.find('/');
	if (spl == string::npos)
		throw std::invalid_argument("Invalid format description for rtpmap");

	this->format = line.substr(0, spl);

	line = line.substr(spl + 1);
	spl = line.find('/');
	if (spl == string::npos) {
		spl = line.find(' ');
	}
	if (spl == string::npos)
		this->clockRate = to_integer<int>(line);
	else {
		this->clockRate = to_integer<int>(line.substr(0, spl));
		this->encParams = line.substr(spl + 1);
	}
}

void Description::Media::RtpMap::addFeedback(string fb) {
	if (std::find(rtcpFbs.begin(), rtcpFbs.end(), fb) == rtcpFbs.end())
		rtcpFbs.emplace_back(std::move(fb));
}

void Description::Media::RtpMap::removeFeedback(const string &str) {
	auto it = rtcpFbs.begin();
	while (it != rtcpFbs.end()) {
		if (it->find(str) != string::npos)
			it = rtcpFbs.erase(it);
		else
			it++;
	}
}

void Description::Media::RtpMap::addParameter(string p) {
	if (std::find(fmtps.begin(), fmtps.end(), p) == fmtps.end())
		fmtps.emplace_back(std::move(p));
}

void Description::Media::RtpMap::removeParameter(const string &str) {
	fmtps.erase(std::remove_if(fmtps.begin(), fmtps.end(),
	                           [&](const auto &p) { return p.find(str) != string::npos; }),
	            fmtps.end());
}

Description::Audio::Audio(string mid, Direction dir)
    : Media("audio 9 UDP/TLS/RTP/SAVPF", std::move(mid), dir) {}

void Description::Audio::addAudioCodec(int payloadType, string codec, optional<string> profile) {
	if (codec.find('/') == string::npos) {
		if (codec == "PCMA" || codec == "PCMU")
			codec += "/8000/1";
		else
			codec += "/48000/2";
	}

	RtpMap map(std::to_string(payloadType) + ' ' + codec);

	if (profile)
		map.fmtps.emplace_back(*profile);

	addRtpMap(map);
}

void Description::Audio::addOpusCodec(int payloadType, optional<string> profile) {
	addAudioCodec(payloadType, "opus", profile);
}

void Description::Audio::addPCMACodec(int payloadType, optional<string> profile) {
	addAudioCodec(payloadType, "PCMA", profile);
}

void Description::Audio::addPCMUCodec(int payloadType, optional<string> profile) {
	addAudioCodec(payloadType, "PCMU", profile);
}

void Description::Audio::addAacCodec(int payloadType, optional<string> profile) {
	if (profile) {
		addAudioCodec(payloadType, "MP4A-LATM", profile);
	} else {
		addAudioCodec(payloadType, "MP4A-LATM", "cpresent=1");
	}
}

Description::Video::Video(string mid, Direction dir)
    : Media("video 9 UDP/TLS/RTP/SAVPF", std::move(mid), dir) {}

void Description::Video::addVideoCodec(int payloadType, string codec, optional<string> profile) {
	if (codec.find('/') == string::npos)
		codec += "/90000";

	RtpMap map(std::to_string(payloadType) + ' ' + codec);

	map.addFeedback("nack");
	map.addFeedback("nack pli");
	// map.addFB("ccm fir");
	map.addFeedback("goog-remb");

	if (profile)
		map.fmtps.emplace_back(*profile);

	addRtpMap(map);

	/* TODO
	 *  TIL that Firefox does not properly support the negotiation of RTX! It works, but doesn't
	 * negotiate the SSRC so we have no idea what SSRC is RTX going to be. Three solutions: One) we
	 * don't negotitate it and (maybe) break RTX support with Edge. Two) we do negotiate it and
	 * rebuild the original packet before we send it distribute it to each track. Three) we complain
	 * to mozilla. This one probably won't do much.
	 */
	// RTX Packets
	// Format rtx(std::to_string(payloadType+1) + " rtx/90000");
	// // TODO rtx-time is how long can a request be stashed for before needing to resend it.
	// Needs to be parameterized rtx.addAttribute("apt=" + std::to_string(payloadType) +
	// ";rtx-time=3000"); addFormat(rtx);
}

void Description::Video::addH264Codec(int payloadType, optional<string> profile) {
	addVideoCodec(payloadType, "H264", profile);
}

void Description::Video::addH265Codec(int payloadType, optional<string> profile) {
	addVideoCodec(payloadType, "H265", profile);
}

void Description::Video::addVP8Codec(int payloadType, optional<string> profile) {
	addVideoCodec(payloadType, "VP8", profile);
}

void Description::Video::addVP9Codec(int payloadType, optional<string> profile) {
	addVideoCodec(payloadType, "VP9", profile);
}

void Description::Video::addAV1Codec(int payloadType, optional<string> profile) {
	addVideoCodec(payloadType, "AV1", profile);
}

Description::Type Description::stringToType(const string &typeString) {
	using TypeMap_t = std::unordered_map<string, Type>;
	static const TypeMap_t TypeMap = {{"unspec", Type::Unspec},
	                                  {"offer", Type::Offer},
	                                  {"answer", Type::Answer},
	                                  {"pranswer", Type::Pranswer},
	                                  {"rollback", Type::Rollback}};
	auto it = TypeMap.find(typeString);
	return it != TypeMap.end() ? it->second : Type::Unspec;
}

string Description::typeToString(Type type) {
	switch (type) {
	case Type::Unspec:
		return "unspec";
	case Type::Offer:
		return "offer";
	case Type::Answer:
		return "answer";
	case Type::Pranswer:
		return "pranswer";
	case Type::Rollback:
		return "rollback";
	default:
		return "unknown";
	}
}

} // namespace rtc

std::ostream &operator<<(std::ostream &out, const rtc::Description &description) {
	return out << std::string(description);
}

std::ostream &operator<<(std::ostream &out, rtc::Description::Type type) {
	return out << rtc::Description::typeToString(type);
}

std::ostream &operator<<(std::ostream &out, rtc::Description::Role role) {
	using Role = rtc::Description::Role;
	// Used for SDP generation, do not change
	switch (role) {
	case Role::Active:
		out << "active";
		break;
	case Role::Passive:
		out << "passive";
		break;
	default:
		out << "actpass";
		break;
	}
	return out;
}

std::ostream &operator<<(std::ostream &out, const rtc::Description::Direction &direction) {
	// Used for SDP generation, do not change
	switch (direction) {
	case rtc::Description::Direction::RecvOnly:
		out << "recvonly";
		break;
	case rtc::Description::Direction::SendOnly:
		out << "sendonly";
		break;
	case rtc::Description::Direction::SendRecv:
		out << "sendrecv";
		break;
	case rtc::Description::Direction::Inactive:
		out << "inactive";
		break;
	case rtc::Description::Direction::Unknown:
	default:
		out << "unknown";
		break;
	}
	return out;
}
