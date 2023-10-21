/**
 * Copyright (c) 2023 Zita Liao (Dolby)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_H265_NAL_UNIT_H
#define RTC_H265_NAL_UNIT_H

#if RTC_ENABLE_MEDIA

#include "common.hpp"
#include "nalunit.hpp"

#include <cassert>

namespace rtc {

#pragma pack(push, 1)

#define H265_FU_HEADER_SIZE 1
/// Nalu header
struct RTC_CPP_EXPORT H265NalUnitHeader {
	/*
	* nal_unit_header( ) {
	* forbidden_zero_bit	f(1)
	* nal_unit_type			u(6)
	* nuh_layer_id			u(6)
	* nuh_temporal_id_plus1	u(3)
	}
	*/
	uint8_t _first = 0;  // high byte of header
	uint8_t _second = 0; // low byte of header

	bool forbiddenBit() const { return _first >> 7; }
	uint8_t unitType() const { return (_first & 0b0111'1110) >> 1; }
	uint8_t nuhLayerId() const { return ((_first & 0x1) << 5) | ((_second & 0b1111'1000) >> 3); }
	uint8_t nuhTempIdPlus1() const { return _second & 0b111; }

	void setForbiddenBit(bool isSet) { _first = (_first & 0x7F) | (isSet << 7); }
	void setUnitType(uint8_t type) { _first = (_first & 0b1000'0001) | ((type & 0b11'1111) << 1); }
	void setNuhLayerId(uint8_t nuhLayerId) {
		_first = (_first & 0b1111'1110) | ((nuhLayerId & 0b10'0000) >> 5);
		_second = (_second & 0b0000'0111) | ((nuhLayerId & 0b01'1111) << 3);
	}
	void setNuhTempIdPlus1(uint8_t nuhTempIdPlus1) {
		_second = (_second & 0b1111'1000) | (nuhTempIdPlus1 & 0b111);
	}
};

/// Nalu fragment header
struct RTC_CPP_EXPORT H265NalUnitFragmentHeader {
	/*
	 * +---------------+
	 * |0|1|2|3|4|5|6|7|
	 * +-+-+-+-+-+-+-+-+
	 * |S|E|  FuType   |
	 * +---------------+
	 */
	uint8_t _first = 0;

	bool isStart() const { return _first >> 7; }
	bool isEnd() const { return (_first >> 6) & 0x01; }
	uint8_t unitType() const { return _first & 0b11'1111; }

	void setStart(bool isSet) { _first = (_first & 0x7F) | (isSet << 7); }
	void setEnd(bool isSet) { _first = (_first & 0b1011'1111) | (isSet << 6); }
	void setUnitType(uint8_t type) { _first = (_first & 0b1100'0000) | (type & 0b11'1111); }
};

#pragma pack(pop)

/// Nal unit
struct RTC_CPP_EXPORT H265NalUnit : NalUnit {
	H265NalUnit(const H265NalUnit &unit) = default;
	H265NalUnit(size_t size, bool includingHeader = true)
	    : NalUnit(size, includingHeader, NalUnit::Type::H265) {}
	H265NalUnit(binary &&data) : NalUnit(std::move(data)) {}
	H265NalUnit() : NalUnit(NalUnit::Type::H265) {}

	template <typename Iterator>
	H265NalUnit(Iterator begin_, Iterator end_) : NalUnit(begin_, end_) {}

	bool forbiddenBit() const { return header()->forbiddenBit(); }
	uint8_t unitType() const { return header()->unitType(); }
	uint8_t nuhLayerId() const { return header()->nuhLayerId(); }
	uint8_t nuhTempIdPlus1() const { return header()->nuhTempIdPlus1(); }

	binary payload() const {
		assert(size() >= H265_NAL_HEADER_SIZE);
		return {begin() + H265_NAL_HEADER_SIZE, end()};
	}

	void setForbiddenBit(bool isSet) { header()->setForbiddenBit(isSet); }
	void setUnitType(uint8_t type) { header()->setUnitType(type); }
	void setNuhLayerId(uint8_t nuhLayerId) { header()->setNuhLayerId(nuhLayerId); }
	void setNuhTempIdPlus1(uint8_t nuhTempIdPlus1) { header()->setNuhTempIdPlus1(nuhTempIdPlus1); }

	void setPayload(binary payload) {
		assert(size() >= H265_NAL_HEADER_SIZE);
		erase(begin() + H265_NAL_HEADER_SIZE, end());
		insert(end(), payload.begin(), payload.end());
	}

protected:
	const H265NalUnitHeader *header() const {
		assert(size() >= H265_NAL_HEADER_SIZE);
		return reinterpret_cast<const H265NalUnitHeader *>(data());
	}

	H265NalUnitHeader *header() {
		assert(size() >= H265_NAL_HEADER_SIZE);
		return reinterpret_cast<H265NalUnitHeader *>(data());
	}
};

/// Nal unit fragment A
struct RTC_CPP_EXPORT H265NalUnitFragment : H265NalUnit {
	static std::vector<shared_ptr<H265NalUnitFragment>> fragmentsFrom(shared_ptr<H265NalUnit> nalu,
	                                                                  uint16_t maximumFragmentSize);

	enum class FragmentType { Start, Middle, End };

	H265NalUnitFragment(FragmentType type, bool forbiddenBit, uint8_t nuhLayerId,
	                    uint8_t nuhTempIdPlus1, uint8_t unitType, binary data);

	uint8_t unitType() const { return fragmentHeader()->unitType(); }

	binary payload() const {
		assert(size() >= H265_NAL_HEADER_SIZE + H265_FU_HEADER_SIZE);
		return {begin() + H265_NAL_HEADER_SIZE + H265_FU_HEADER_SIZE, end()};
	}

	FragmentType type() const {
		if (fragmentHeader()->isStart()) {
			return FragmentType::Start;
		} else if (fragmentHeader()->isEnd()) {
			return FragmentType::End;
		} else {
			return FragmentType::Middle;
		}
	}

	void setUnitType(uint8_t type) { fragmentHeader()->setUnitType(type); }

	void setPayload(binary payload) {
		assert(size() >= H265_NAL_HEADER_SIZE + H265_FU_HEADER_SIZE);
		erase(begin() + H265_NAL_HEADER_SIZE + H265_FU_HEADER_SIZE, end());
		insert(end(), payload.begin(), payload.end());
	}

	void setFragmentType(FragmentType type);

protected:
	const uint8_t nal_type_fu = 49;

	H265NalUnitHeader *fragmentIndicator() { return reinterpret_cast<H265NalUnitHeader *>(data()); }

	const H265NalUnitHeader *fragmentIndicator() const {
		return reinterpret_cast<const H265NalUnitHeader *>(data());
	}

	H265NalUnitFragmentHeader *fragmentHeader() {
		return reinterpret_cast<H265NalUnitFragmentHeader *>(fragmentIndicator() +
		                                                     H265_NAL_HEADER_SIZE);
	}

	const H265NalUnitFragmentHeader *fragmentHeader() const {
		return reinterpret_cast<const H265NalUnitFragmentHeader *>(fragmentIndicator() +
		                                                           H265_NAL_HEADER_SIZE);
	}
};

class RTC_CPP_EXPORT H265NalUnits : public std::vector<shared_ptr<H265NalUnit>> {
public:
	static const uint16_t defaultMaximumFragmentSize =
	    uint16_t(RTC_DEFAULT_MTU - 12 - 8 - 40); // SRTP/UDP/IPv6

	std::vector<shared_ptr<binary>> generateFragments(uint16_t maximumFragmentSize);
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_NAL_UNIT_H */
