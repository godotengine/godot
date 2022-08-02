// +----------------------------------------------------------------------
// | Project : ray.
// | All rights reserved.
// +----------------------------------------------------------------------
// | Copyright (c) 2013-2017.
// +----------------------------------------------------------------------
// | * Redistribution and use of this software in source and binary forms,
// |   with or without modification, are permitted provided that the following
// |   conditions are met:
// |
// | * Redistributions of source code must retain the above
// |   copyright notice, this list of conditions and the
// |   following disclaimer.
// |
// | * Redistributions in binary form must reproduce the above
// |   copyright notice, this list of conditions and the
// |   following disclaimer in the documentation and/or other
// |   materials provided with the distribution.
// |
// | * Neither the name of the ray team, nor the names of its
// |   contributors may be used to endorse or promote products
// |   derived from this software without specific prior
// |   written permission of the ray team.
// |
// | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// | "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// | LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// | A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// | OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// | SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// | LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// | DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// | THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// | (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// | OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// +----------------------------------------------------------------------
#ifndef _H_IES_LOADER_H_
#define _H_IES_LOADER_H_

#include <vector>
#include <string>

// https://knowledge.autodesk.com/support/3ds-max/learn-explore/caas/CloudHelp/cloudhelp/2016/ENU/3DSMax/files/GUID-EA0E3DE0-275C-42F7-83EC-429A37B2D501-htm.html
class IESFileInfo
{
public:
	IESFileInfo();

	bool valid() const;

	const std::string& error() const;

public:
	float totalLights;
	float totalLumens;

	float candalaMult;

	std::int32_t typeOfPhotometric;
	std::int32_t typeOfUnit;

	std::int32_t anglesNumH;
	std::int32_t anglesNumV;

	float width;
	float length;
	float height;

	float ballastFactor;
	float futureUse;
	float inputWatts;

private:
	friend class IESLoadHelper;

	float _cachedIntegral;

	std::string _error;
	std::string _version;

	std::vector<float> _anglesH;
	std::vector<float> _anglesV;
	std::vector<float> _candalaValues;
};

class IESLoadHelper final
{
public:
	IESLoadHelper();
	~IESLoadHelper();

	bool load(const std::string& data, IESFileInfo& info);
	bool load(const char* data, std::size_t dataLength, IESFileInfo& info);

	bool saveAs1D(const IESFileInfo& info, float* data, std::uint32_t width = 256, std::uint8_t channel = 3) noexcept;
	bool saveAs2D(const IESFileInfo& info, float* data, std::uint32_t width = 256, std::uint32_t height = 256, std::uint8_t channel = 3) noexcept;
	bool saveAsPreview(const IESFileInfo& info, std::uint8_t* data, std::uint32_t width = 64, std::uint32_t height = 64, std::uint8_t channel = 3) noexcept;

private:
	float computeInvMax(const std::vector<float>& candalaValues) const;
	float computeFilterPos(float value, const std::vector<float>& angle) const;

	float interpolate1D(const IESFileInfo& info, float angle) const;
	float interpolate2D(const IESFileInfo& info, float angleV, float angleH) const;
	float interpolatePoint(const IESFileInfo& info, std::uint32_t x, std::uint32_t y) const;
	float interpolateBilinear(const IESFileInfo& info, float x, float y) const;

private:
	static void skipSpaceAndLineEnd(const std::string& data, std::string& out, bool stopOnComma = false);

	static void getLineContent(const std::string& data, std::string& next, std::string& line, bool stopOnWhiteSpace, bool stopOnComma);
	static void getFloat(const std::string& data, std::string& next, float& ret, bool stopOnWhiteSpace = true, bool stopOnComma = false);
	static void getInt(const std::string& data, std::string& next, std::int32_t& ret, bool stopOnWhiteSpace = true, bool stopOnComma = false);
};

#endif