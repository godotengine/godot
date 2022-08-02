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
#include "ies_loader.h"
#include <assert.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath> 

IESFileInfo::IESFileInfo()
	: _cachedIntegral(std::numeric_limits<float>::max())
	, _error("No data loaded")
{
}

bool
IESFileInfo::valid() const
{
	return _error.empty();
}

const std::string&
IESFileInfo::error() const
{
	return _error;
}

IESLoadHelper::IESLoadHelper()
{
}

IESLoadHelper::~IESLoadHelper()
{
}

bool
IESLoadHelper::load(const char* data, std::size_t dataLength, IESFileInfo& info)
{
	assert(!info.valid());
	return this->load(std::string(data, dataLength), info);
}

bool
IESLoadHelper::load(const std::string& data, IESFileInfo& info)
{
	assert(!info.valid());

	std::string dataPos;

	std::string version;
	this->getLineContent(data, dataPos, version, false, false);

	if (version.empty())
	{
		info._error = "Unknown IES version";
		return false;
	}
	else if (version == "IESNA:LM-63-1995")
		info._version = "IESNA:LM-63-1995";
	else if (version == "IESNA91")
		info._version = "IESNA91";
	else if (version == "IESNA:LM-63-2002")
		info._version = "IESNA:LM-63-2002";
	else
		info._version = version;

	while (!dataPos.empty())
	{
		std::string line;
		this->getLineContent(dataPos, dataPos, line, false, false);

		if (line.compare(0, 9, "TILT=NONE", 9) == 0 ||
			line.compare(0, 10, "TILT= NONE", 10) == 0 ||
			line.compare(0, 10, "TILT =NONE", 10) == 0 ||
			line.compare(0, 11, "TILT = NONE", 11) == 0)
		{
			break;
		}
		else if (line.compare(0, 5, "TILT=", 5) == 0 ||
			line.compare(0, 5, "TILT =", 5) == 0)
		{
			info._error = "Not supported yet.";
			return false;
		}
	}

	this->getFloat(dataPos, dataPos, info.totalLights);
	if (info.totalLights < 0 || info.totalLights > std::numeric_limits<short>::max())
	{
		info._error = "Light Count is not valid";
		return false;
	}

	this->getFloat(dataPos, dataPos, info.totalLumens);
	// TODO: Godot Engine: fire 2022-08-02 
	// if (info.totalLumens < 0)
	// {
	// 	info._error = "TotalLumens is not positive number";
	// 	return false;
	// }

	this->getFloat(dataPos, dataPos, info.candalaMult);
	if (info.candalaMult < 0)
	{
		info._error = "CandalaMult is not positive number";
		return false;
	}

	this->getInt(dataPos, dataPos, info.anglesNumV);
	if (info.anglesNumV < 0 || info.anglesNumV > std::numeric_limits<short>::max())
	{
		info._error = "VAnglesNum is not valid";
		return false;
	}

	this->getInt(dataPos, dataPos, info.anglesNumH);
	if (info.anglesNumH < 0 || info.anglesNumH > std::numeric_limits<short>::max())
	{
		info._error = "HAnglesNum is not valid";
		return false;
	}

	this->getInt(dataPos, dataPos, info.typeOfPhotometric);
	this->getInt(dataPos, dataPos, info.typeOfUnit);

	this->getFloat(dataPos, dataPos, info.width);
	this->getFloat(dataPos, dataPos, info.length);
	this->getFloat(dataPos, dataPos, info.height);

	this->getFloat(dataPos, dataPos, info.ballastFactor);
	this->getFloat(dataPos, dataPos, info.futureUse);
	this->getFloat(dataPos, dataPos, info.inputWatts);

	float minSoFarV = std::numeric_limits<float>::lowest();
	float minSoFarH = std::numeric_limits<float>::lowest();

	info._anglesV.reserve(info.anglesNumV);
	info._anglesH.reserve(info.anglesNumH);

	for (std::int32_t y = 0; y < info.anglesNumV; ++y)
	{
		float value;
		this->getFloat(dataPos, dataPos, value, true, true);

		if (value < minSoFarV)
		{
			info._error = "V Values is not valid";
			return false;
		}

		minSoFarV = value;
		info._anglesV.push_back(value);
	}

	for (std::int32_t x = 0; x < info.anglesNumH; ++x)
	{
		float value;
		this->getFloat(dataPos, dataPos, value, true, true);

		if (value < minSoFarH)
		{
			info._error = "H Values is not valid";
			return false;
		}

		minSoFarH = value;
		info._anglesH.push_back(value);
	}

	info._candalaValues.reserve(info.anglesNumH * info.anglesNumV);

	for (std::int32_t y = 0; y < info.anglesNumH; ++y)
	{
		for (std::int32_t x = 0; x < info.anglesNumV; ++x)
		{
			float value;
			this->getFloat(dataPos, dataPos, value, true, true);
			info._candalaValues.push_back(value * info.candalaMult);
		}
	}

	skipSpaceAndLineEnd(dataPos, dataPos);

	if (!dataPos.empty())
	{
		std::string line;
		this->getLineContent(dataPos, dataPos, line, true, false);

		if (line == "END")
			skipSpaceAndLineEnd(dataPos, dataPos);

		if (!dataPos.empty())
		{
			info._error = "Unexpected content after END.";
			return false;
		}
	}

	info._error.clear();

	return true;
}

bool
IESLoadHelper::saveAs1D(const IESFileInfo& info, float* data, std::uint32_t width, std::uint8_t channel) noexcept
{
	assert(data);
	assert(width > 0);
	assert(channel == 1 || channel == 3 || channel == 4);
	assert(info.valid());

	float invW = 1.0f / width;
	float invMaxValue = this->computeInvMax(info._candalaValues);

	for (std::uint32_t x = 0; x < width; ++x)
	{
		float fraction = x * invW;
		float value = invMaxValue * interpolate1D(info, fraction * 180.0f);

		switch (channel)
		{
		case 1:
			*data++ = value;
			break;
		case 3:
			*data++ = value;
			*data++ = value;
			*data++ = value;
			break;
		case 4:
			*data++ = value;
			*data++ = value;
			*data++ = value;
			*data++ = 1.0f;
			break;
		default:
			return false;
		}
	}

	return true;
}

bool
IESLoadHelper::saveAs2D(const IESFileInfo& info, float* data, std::uint32_t width, std::uint32_t height, std::uint8_t channel) noexcept
{
	assert(data);
	assert(width > 0 && height > 0);
	assert(channel == 1 || channel == 3 || channel == 4);
	assert(info.valid());

	float invW = 1.0f / width;
	float invH = 1.0f / height;
	float invMaxValue = this->computeInvMax(info._candalaValues);

	for (std::uint32_t y = 0; y < height; ++y)
	{
		for (std::uint32_t x = 0; x < width; ++x)
		{
			float fractionV = x * invW * 180.0f;
			float fractionH = y * invH * 180.0f;
			float value = invMaxValue * interpolate2D(info, fractionV, fractionH);

			switch (channel)
			{
			case 1:
				*data++ = value;
				break;
			case 3:
				*data++ = value;
				*data++ = value;
				*data++ = value;
				break;
			case 4:
				*data++ = value;
				*data++ = value;
				*data++ = value;
				*data++ = 1.0;
				break;
			default:
				return false;
			}
		}
	}

	return true;
}

bool
IESLoadHelper::saveAsPreview(const IESFileInfo& info, std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint8_t channel) noexcept
{
	assert(data);
	assert(width > 0 && height > 0);
	assert(channel == 1 || channel == 3 || channel == 4);
	assert(info.valid());

	std::vector<float> ies(256);
	if (!this->saveAs1D(info, ies.data(), ies.size(), 1))
		return false;

	float maxValue = this->computeInvMax(info._candalaValues);

	auto TonemapHable = [](float x)
	{
		const float A = 0.22f;
		const float B = 0.30f;
		const float C = 0.10f;
		const float D = 0.20f;
		const float E = 0.01f;
		const float F = 0.30f;
		return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E / F;
	};

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float u = ((float)x / width) * 2.0f - 1.0f;
			float v = 1.0f - ((float)y / height) * 2.0f - 1.0f;

			u *= 2.2f;
			v *= 2.4f;

			// float3(0.0f, 0.0f, -0.5f) - ray::float3(u, v, 0.0f)
			float lx = +0.0f - u;
			float ly = +0.0f - v;
			float lz = -0.5f - 0.0f;

			// normalize
			float length = std::sqrt(lx * lx + ly * ly + lz * lz);
			lx /= length;
			ly /= length;
			lz /= length;

			float angle = 1.0 - std::acos(lx * 0.0 + ly * -1.0 + lz * 0.0f) / 3.141592654;

			float intensity = ies[angle * 255] * maxValue / length;

			std::uint8_t value = std::min(std::max((int)std::floor(TonemapHable(intensity) / TonemapHable(maxValue) * 255.0f), 0), 255);

			switch (channel)
			{
			case 1:
				*data++ = value;
				break;
			case 3:
				*data++ = value;
				*data++ = value;
				*data++ = value;
				break;
			case 4:
				*data++ = value;
				*data++ = value;
				*data++ = value;
				*data++ = 1.0;
				break;
			default:
				return false;
			}
		}
	}

	return true;
}

float
IESLoadHelper::computeInvMax(const std::vector<float>& candalaValues) const
{
	assert(candalaValues.size());

	float candala = *std::max_element(candalaValues.begin(), candalaValues.end());
	return 1.0f / candala;
}

float
IESLoadHelper::computeFilterPos(float value, const std::vector<float>& angles) const
{
	assert(angles.size());

	std::size_t start = 0;
	std::size_t end = angles.size() - 1;

	if (value < angles[start]) return 0.0f;
	if (value > angles[end]) return (float)end;

	while (start < end)
	{
		std::size_t index = (start + end + 1) / 2;

		float angle = angles[index];
		if (value >= angle)
		{
			assert(start != index);
			start = index;
		}
		else
		{
			assert(end != index - 1);
			end = index - 1;
		}
	}

	float leftValue = angles[start];
	float fraction = 0.0f;

	if (start + 1 < (std::uint32_t)angles.size())
	{
		float rightValue = angles[start + 1];
		float deltaValue = rightValue - leftValue;

		if (deltaValue > 0.0001f)
		{
			fraction = (value - leftValue) / deltaValue;
		}
	}

	return start + fraction;
}

float
IESLoadHelper::interpolate1D(const IESFileInfo& info, float angle) const
{
	float angleV = this->computeFilterPos(angle, info._anglesV);
	float anglesNum = (float)info._anglesH.size();
	float angleTotal = 0.0f;

	for (float x = 0; x < anglesNum; x++)
		angleTotal += this->interpolateBilinear(info, x, angleV);

	return angleTotal / anglesNum;
}

float
IESLoadHelper::interpolate2D(const IESFileInfo& info, float angleV, float angleH) const
{
	float u = this->computeFilterPos(angleH, info._anglesH);
	float v = this->computeFilterPos(angleV, info._anglesV);
	return this->interpolateBilinear(info, u, v);
}

float
IESLoadHelper::interpolatePoint(const IESFileInfo& info, std::uint32_t x, std::uint32_t y) const
{
	assert(x >= 0);
	assert(y >= 0);

	std::size_t anglesNumH = info._anglesH.size();
	std::size_t anglesNumV = info._anglesV.size();

	x %= anglesNumH;
	y %= anglesNumV;

	assert(x < anglesNumH);
	assert(y < anglesNumV);

	return info._candalaValues[y + anglesNumV * x];
}

float
IESLoadHelper::interpolateBilinear(const IESFileInfo& info, float x, float y) const
{
	int ix = (int)std::floor(x);
	int iy = (int)std::floor(y);

	float fracX = x - ix;
	float fracY = y - iy;

	float p00 = this->interpolatePoint(info, ix + 0, iy + 0);
	float p10 = this->interpolatePoint(info, ix + 1, iy + 0);
	float p01 = this->interpolatePoint(info, ix + 0, iy + 1);
	float p11 = this->interpolatePoint(info, ix + 1, iy + 1);

	auto lerp = [](float t1, float t2, float t3) -> float { return t1 + (t2 - t1) * t3; };

	float p0 = lerp(p00, p01, fracY);
	float p1 = lerp(p10, p11, fracY);

	return lerp(p0, p1, fracX);
}

void
IESLoadHelper::skipSpaceAndLineEnd(const std::string& data, std::string& out, bool stopOnComma)
{
	std::size_t dataBegin = 0;
	std::size_t dataEnd = data.size();

	while (dataBegin < dataEnd)
	{
		if (data[dataBegin] != '\r' && data[dataBegin] != '\n' && data[dataBegin] > ' ')
			break;
		dataBegin++;
	}

	if (stopOnComma)
	{
		while (dataBegin < dataEnd)
		{
			if (data[dataBegin] != ',')
				break;
			dataBegin++;
		}
	}

	out = data.substr(dataBegin, data.size() - dataBegin);
}

void
IESLoadHelper::getLineContent(const std::string& data, std::string& next, std::string& line, bool stopOnWhiteSpace, bool stopOnComma)
{
	skipSpaceAndLineEnd(data, next);

	auto it = data.begin();
	auto end = data.end();

	for (; it < end; ++it)
	{
		if ((*it == '\r') ||
			(*it == '\n') ||
			(*it <= ' ' && stopOnWhiteSpace) ||
			(*it == ',' && stopOnComma))
		{
			break;
		}
	}

	line.assign(data, 0, it - data.begin());
	next.assign(data, it - data.begin(), end - it);

	skipSpaceAndLineEnd(next, next, stopOnComma);
}

void
IESLoadHelper::getFloat(const std::string& data, std::string& next, float& ret, bool stopOnWhiteSpace, bool stopOnComma)
{
	std::string line;
	getLineContent(data, next, line, stopOnWhiteSpace, stopOnComma);
	assert(!line.empty());
	ret = (float)std::atof(line.c_str());
}

void
IESLoadHelper::getInt(const std::string& data, std::string& next, std::int32_t& ret, bool stopOnWhiteSpace, bool stopOnComma)
{
	std::string line;
	getLineContent(data, next, line, stopOnWhiteSpace, stopOnComma);
	assert(!line.empty());
	ret = std::atoi(line.c_str());
}