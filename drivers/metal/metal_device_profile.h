/**************************************************************************/
/*  metal_device_profile.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/typedefs.h"

class MinOsVersion {
	uint32_t version;

public:
	String to_compiler_os_version() const;
	bool is_null() const { return version == UINT32_MAX; }
	bool is_valid() const { return version != UINT32_MAX; }

	MinOsVersion(const String &p_version);
	constexpr explicit MinOsVersion(uint32_t p_version) :
			version(p_version) {}
	constexpr MinOsVersion(uint32_t p_major, uint32_t p_minor, uint32_t p_patch = 0) :
			version(p_major * 10000 + p_minor * 100 + p_patch) {}
	constexpr MinOsVersion() :
			version(UINT32_MAX) {}

	bool operator>(uint32_t p_other) {
		return version > p_other;
	}
	constexpr operator uint32_t() const { return version; }
};

namespace os_version {

constexpr MinOsVersion MACOS_26_0(26'00'00);
constexpr MinOsVersion MACOS_15_0(15'00'00);
constexpr MinOsVersion MACOS_14_0(14'00'00);
constexpr MinOsVersion MACOS_13_0(13'00'00);
constexpr MinOsVersion MACOS_12_0(12'00'00);
constexpr MinOsVersion MACOS_11_0(11'00'00);

constexpr MinOsVersion IOS_26_0(26'00'00);
constexpr MinOsVersion IOS_18_0(18'00'00);
constexpr MinOsVersion IOS_17_0(17'00'00);
constexpr MinOsVersion IOS_16_0(16'00'00);
constexpr MinOsVersion IOS_15_0(15'00'00);

constexpr MinOsVersion VISIONOS_26_0(26'00'00);
constexpr MinOsVersion VISIONOS_02_4(02'04'00);

} //namespace os_version

/// @brief A minimal structure that defines a device profile for Metal.
///
/// This structure is used by the `RenderingShaderContainerMetal` class to
/// determine options for compiling SPIR-V to Metal source. It currently only
/// contains the minimum properties required to transform shaders from SPIR-V to Metal
/// and potentially compile to a `.metallib`.
struct MetalDeviceProfile {
	enum class Platform : uint32_t {
		macOS = 0,
		iOS = 1,
		visionOS = 2,
	};

	/*! @brief The GPU family.
	 *
	 * NOTE: These values match Apple's MTLGPUFamily
	 */
	enum class GPU : uint32_t {
		Apple1 = 1001,
		Apple2 = 1002,
		Apple3 = 1003,
		Apple4 = 1004,
		Apple5 = 1005,
		Apple6 = 1006,
		Apple7 = 1007,
		Apple8 = 1008,
		Apple9 = 1009,
	};

	enum class ArgumentBuffersTier : uint32_t {
		Tier1 = 0,
		Tier2 = 1,
	};

	struct Features {
		uint32_t msl_version = 0;
		bool use_argument_buffers = false;
		bool simdPermute = false;
	};

	Platform platform = Platform::macOS;
	GPU gpu = GPU::Apple4;
	MinOsVersion min_os_version;
	Features features;

	static const MetalDeviceProfile *get_profile(Platform p_platform, GPU p_gpu, MinOsVersion p_min_os_version);

	MetalDeviceProfile() = default;

private:
	static Mutex profiles_lock; ///< Mutex to protect access to the profiles map.

	struct ProfileKey {
		friend struct HashMapHasherDefaultImpl<ProfileKey>;
		union {
			struct {
				uint32_t min_os_version;
				uint16_t platform;
				uint16_t gpu;
			};
			uint64_t value = 0;
		};

		ProfileKey() = default;
		ProfileKey(MinOsVersion p_min_os_version, Platform p_platform, GPU p_gpu) :
				min_os_version(p_min_os_version), platform((uint16_t)p_platform), gpu((uint16_t)p_gpu) {}

		_FORCE_INLINE_ uint32_t hash() const {
			return hash_one_uint64(value);
		}

		bool operator==(const ProfileKey &p_other) const {
			return value == p_other.value;
		}
	};

	static HashMap<ProfileKey, MetalDeviceProfile> profiles;
};
