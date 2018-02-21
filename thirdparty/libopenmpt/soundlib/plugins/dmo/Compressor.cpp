/*
 * Compressor.cpp
 * ---------------
 * Purpose: Implementation of the DMO Compressor DSP (for non-Windows platforms)
 * Notes  : The original plugin's integer and floating point code paths only
 *          behave identically when feeding floating point numbers in range
 *          [-32768, +32768] rather than the usual [-1, +1] into the plugin.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "Compressor.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

// See Distortion.cpp
float logGain(float x, int32 shiftL, int32 shiftR);

IMixPlugin* Compressor::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) Compressor(factory, sndFile, mixStruct);
}


Compressor::Compressor(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
{
	m_param[kCompGain] = 0.5f;
	m_param[kCompAttack] = 0.02f;
	m_param[kCompRelease] = 150.0f / 2950.0f;
	m_param[kCompThreshold] = 2.0f / 3.0f;
	m_param[kCompRatio] = 0.02f;
	m_param[kCompPredelay] = 1.0f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


void Compressor::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_bufSize || !m_mixBuffer.Ok())
		return;

	const float *in[2] = { m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1) };
	float *out[2] = { m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1) };

	for(uint32 i = numFrames; i != 0; i--)
	{
		float leftIn  = *(in[0])++;
		float rightIn = *(in[1])++;
			
		m_buffer[m_bufPos * 2] = leftIn;
		m_buffer[m_bufPos * 2 + 1] = rightIn;

		leftIn = mpt::abs(leftIn);
		rightIn = mpt::abs(rightIn);

		float mono = (leftIn + rightIn) * (0.5f * 32768.0f * 32768.0f);
		float monoLog = mpt::abs(logGain(mono, 31, 5)) * (1.0f / float(1u << 31));

		float newPeak = monoLog + (m_peak - monoLog) * ((m_peak <= monoLog) ? m_attack : m_release);
		m_peak = newPeak;

		if(newPeak < m_threshold)
			newPeak = m_threshold;

		float compGain = (m_threshold - newPeak) * m_ratio + 0.9999999f;

		// Computes 2 ^ (2 ^ (log2(x) - 26) - 1) (x = 0...2^31)
		uint32 compGainInt = static_cast<uint32>(compGain * 2147483648.0f);
		uint32 compGainPow = compGainInt << 5;
		compGainInt >>= 26;
		if(compGainInt)	// compGainInt >= 2^26
		{
			compGainPow |= 0x80000000u;
			compGainInt--;
		}
		compGainPow >>= (31 - compGainInt);
		
		int32 readOffset = m_predelay + m_bufPos * 4096 + m_bufSize - 1;
		readOffset /= 4096;
		readOffset %= m_bufSize;
		
		float outGain = (compGainPow * (1.0f / 2147483648.0f)) * m_gain;
		*(out[0])++ = m_buffer[readOffset * 2] * outGain;
		*(out[1])++ = m_buffer[readOffset * 2 + 1] * outGain;
		
		if(m_bufPos-- == 0)
			m_bufPos += m_bufSize;
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue Compressor::GetParameter(PlugParamIndex index)
{
	if(index < kCompNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void Compressor::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kCompNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		m_param[index] = value;
		RecalculateCompressorParams();
	}
}


void Compressor::Resume()
{
	m_isResumed = true;
	PositionChanged();
	RecalculateCompressorParams();
}


void Compressor::PositionChanged()
{
	m_bufSize = Util::muldiv(m_SndFile.GetSampleRate(), 200, 1000);
	try
	{
		m_buffer.assign(m_bufSize * 2, 0.0f);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		m_bufSize = 0;
	}
	m_bufPos = 0;
	m_peak = 0.0f;
}


#ifdef MODPLUG_TRACKER

CString Compressor::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kCompGain: return _T("Gain");
	case kCompAttack: return _T("Attack");
	case kCompRelease: return _T("Release");
	case kCompThreshold: return _T("Threshold");
	case kCompRatio: return _T("Ratio");
	case kCompPredelay: return _T("Predelay");
	}
	return CString();
}


CString Compressor::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kCompGain:
	case kCompThreshold:
		return _T("dB");
	case kCompAttack:
	case kCompRelease:
	case kCompPredelay:
		return _T("ms");
	}
	return CString();
}


CString Compressor::GetParamDisplay(PlugParamIndex param)
{
	float value = m_param[param];
	switch(param)
	{
	case kCompGain:
		value = GainInDecibel();
		break;
	case kCompAttack:
		value = AttackTime();
		break;
	case kCompRelease:
		value = ReleaseTime();
		break;
	case kCompThreshold:
		value = ThresholdInDecibel();
		break;
	case kCompRatio:
		value = CompressorRatio();
		break;
	case kCompPredelay:
		value = PreDelay();
		break;
	}
	CString s;
	s.Format(_T("%.2f"), value);
	return s;
}

#endif // MODPLUG_TRACKER


void Compressor::RecalculateCompressorParams()
{
	const float sampleRate = m_SndFile.GetSampleRate() / 1000.0f;
	m_gain = std::pow(10.0f, GainInDecibel() / 20.0f);
	m_attack = std::pow(10.0f, -1.0f / (AttackTime() * sampleRate));
	m_release = std::pow(10.0f, -1.0f / (ReleaseTime() * sampleRate));
	const float _2e31 = float(1u << 31);
	const float _2e26 = float(1u << 26);
	m_threshold = std::min((_2e31 - 1.0f), (std::log(std::pow(10.0f, ThresholdInDecibel() / 20.0f) * _2e31) * _2e26) / static_cast<float>(M_LN2) + _2e26) * (1.0f / _2e31);
	m_ratio = 1.0f - (1.0f / CompressorRatio());
	m_predelay = static_cast<int32>((PreDelay() * sampleRate) + 2.0f);
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(Compressor)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
