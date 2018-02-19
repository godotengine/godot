/*
 * DMOPlugin.h
 * -----------
 * Purpose: DirectX Media Object plugin handling / processing.
 * Notes  : Some default plugins only have the same output characteristics in the floating point code path (compared to integer PCM)
 *          if we feed them input in the range [-32768, +32768] rather than the more usual [-1, +1].
 *          Hence, OpenMPT uses this range for both the floating-point and integer path.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_DMO
#include "../../Sndfile.h"
#include "../../../common/mptUUID.h"
#include "DMOPlugin.h"
#include "../PluginManager.h"
#include <uuids.h>
#include <medparam.h>
#include <mmsystem.h>
#endif // !NO_DMO

OPENMPT_NAMESPACE_BEGIN


#ifndef NO_DMO


#define DMO_LOG

IMixPlugin* DMOPlugin::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	CLSID clsid;
	if (Util::VerifyStringToCLSID(factory.dllPath.ToWide(), clsid))
	{
		IMediaObject *pMO = nullptr;
		IMediaObjectInPlace *pMOIP = nullptr;
		if ((CoCreateInstance(clsid, nullptr, CLSCTX_INPROC_SERVER, IID_IMediaObject, (VOID **)&pMO) == S_OK) && (pMO))
		{
			if (pMO->QueryInterface(IID_IMediaObjectInPlace, (void **)&pMOIP) != S_OK) pMOIP = nullptr;
		} else pMO = nullptr;
		if ((pMO) && (pMOIP))
		{
			DWORD dwInputs = 0, dwOutputs = 0;
			pMO->GetStreamCount(&dwInputs, &dwOutputs);
			if (dwInputs == 1 && dwOutputs == 1)
			{
				DMOPlugin *p = new (std::nothrow) DMOPlugin(factory, sndFile, mixStruct, pMO, pMOIP, clsid.Data1);
				return p;
			}
#ifdef DMO_LOG
			Log(factory.libraryName.ToUnicode() + MPT_USTRING(": Unable to use this DMO"));
#endif
		}
#ifdef DMO_LOG
		else Log(factory.libraryName.ToUnicode() + MPT_USTRING(": Failed to get IMediaObject & IMediaObjectInPlace interfaces"));
#endif
		if (pMO) pMO->Release();
		if (pMOIP) pMOIP->Release();
	}
	return nullptr;
}


DMOPlugin::DMOPlugin(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct, IMediaObject *pMO, IMediaObjectInPlace *pMOIP, uint32 uid)
	: IMixPlugin(factory, sndFile, mixStruct)
	, m_pMediaObject(pMO)
	, m_pMediaProcess(pMOIP)
	, m_pParamInfo(nullptr)
	, m_pMediaParams(nullptr)
	, m_nSamplesPerSec(sndFile.GetSampleRate())
	, m_uid(uid)
{
	if(FAILED(m_pMediaObject->QueryInterface(IID_IMediaParamInfo, (void **)&m_pParamInfo)))
		m_pParamInfo = nullptr;
	if (FAILED(m_pMediaObject->QueryInterface(IID_IMediaParams, (void **)&m_pMediaParams)))
		m_pMediaParams = nullptr;
	m_alignedBuffer.f32 = (float *)((((intptr_t)m_interleavedBuffer.f32) + 15) & ~15);

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();

}


DMOPlugin::~DMOPlugin()
{
	if(m_pMediaParams)
	{
		m_pMediaParams->Release();
		m_pMediaParams = nullptr;
	}
	if(m_pParamInfo)
	{
		m_pParamInfo->Release();
		m_pParamInfo = nullptr;
	}
	if(m_pMediaProcess)
	{
		m_pMediaProcess->Release();
		m_pMediaProcess = nullptr;
	}
	if(m_pMediaObject)
	{
		m_pMediaObject->Release();
		m_pMediaObject = nullptr;
	}
}


uint32 DMOPlugin::GetLatency() const
{
	REFERENCE_TIME time;	// Unit 100-nanoseconds
	if(m_pMediaProcess->GetLatency(&time) == S_OK)
	{
		return static_cast<uint32>(time * m_nSamplesPerSec / (10 * 1000 * 1000));
	}
	return 0;
}


static const float _f2si = 32768.0f;
static const float _si2f = 1.0f / 32768.0f;


static void InterleaveStereo(const float * MPT_RESTRICT inputL, const float * MPT_RESTRICT inputR, float * MPT_RESTRICT output, uint32 numFrames)
{
#if (defined(ENABLE_SSE) || defined(ENABLE_SSE2))
	if(GetProcSupport() & PROCSUPPORT_SSE)
	{
		// We may read beyond the wanted length... this works because we know that we will always work on our buffers of size MIXBUFFERSIZE
		STATIC_ASSERT((MIXBUFFERSIZE & 7) == 0);
		__m128 factor = _mm_set_ps1(_f2si);
		numFrames = (numFrames + 3) / 4;
		do
		{
			__m128 fl = _mm_loadu_ps(inputL);		// Load four float values, LLLL
			__m128 fr = _mm_loadu_ps(inputR);		// Load four float values, RRRR
			fl = _mm_mul_ps(fl, factor);			// Scale them
			fr = _mm_mul_ps(fr, factor);			// Scale them
			inputL += 4;
			inputR += 4;
			__m128 f1 = _mm_unpacklo_ps(fl, fr);	// LL__+RR__ => LRLR
			__m128 f2 = _mm_unpackhi_ps(fl, fr);	// __LL+__RR => LRLR
			_mm_store_ps(output, f1);				// Store four int values, LRLR
			_mm_store_ps(output + 4, f2);			// Store four int values, LRLR
			output += 8;
		} while(--numFrames);
		return;
	}
#endif
	while(numFrames--)
	{
		*(output++) = *(inputL++) * _f2si;
		*(output++) = *(inputR++) * _f2si;
	}
}


static void DeinterleaveStereo(const float * MPT_RESTRICT input, float * MPT_RESTRICT outputL, float * MPT_RESTRICT outputR, uint32 numFrames)
{
#if (defined(ENABLE_SSE) || defined(ENABLE_SSE2))
	if(GetProcSupport() & PROCSUPPORT_SSE)
	{
		// We may read beyond the wanted length... this works because we know that we will always work on our buffers of size MIXBUFFERSIZE
		STATIC_ASSERT((MIXBUFFERSIZE & 7) == 0);
		__m128 factor = _mm_set_ps1(_si2f);
		numFrames = (numFrames + 3) / 4;
		do
		{
			__m128 f1 = _mm_load_ps(input);		// Load four float values, LRLR
			__m128 f2 = _mm_load_ps(input + 4);	// Load four float values, LRLR
			f1 = _mm_mul_ps(f1, factor);		// Scale them
			f2 = _mm_mul_ps(f2, factor);		// Scale them
			input += 8;
			__m128 fl = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(2, 0, 2, 0));	// LRLR+LRLR => LLLL
			__m128 fr = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(3, 1, 3, 1));	// LRLR+LRLR => RRRR
			_mm_storeu_ps(outputL, fl);				// Store four float values, LLLL
			_mm_storeu_ps(outputR, fr);				// Store four float values, RRRR
			outputL += 4;
			outputR += 4;
		} while(--numFrames);
		return;
	}
#endif
	while(numFrames--)
	{
		*(outputL++) = *(input++) * _si2f;
		*(outputR++) = *(input++) * _si2f;
	}
}


// Interleave two float streams into one int16 stereo stream.
static void InterleaveFloatToInt16(const float * MPT_RESTRICT inputL, const float * MPT_RESTRICT inputR, int16 * MPT_RESTRICT output, uint32 numFrames)
{
#ifdef ENABLE_SSE
	// This uses __m64, so it's not available on the MSVC 64-bit compiler.
	// But if the user runs a 64-bit operating system, they will go the floating-point path anyway.
	if(GetProcSupport() & PROCSUPPORT_SSE)
	{
		// We may read beyond the wanted length... this works because we know that we will always work on our buffers of size MIXBUFFERSIZE
		STATIC_ASSERT((MIXBUFFERSIZE & 7) == 0);
		__m64 *out = reinterpret_cast<__m64 *>(output);
		__m128 factor = _mm_set_ps1(_f2si);
		numFrames = (numFrames + 3) / 4;
		do
		{
			__m128 fl = _mm_loadu_ps(inputL);		// Load four float values, L1L2L3L4
			__m128 fr = _mm_loadu_ps(inputR);		// Load four float values, R1R2R3R4
			fl = _mm_mul_ps(fl, factor);			// Scale them
			fr = _mm_mul_ps(fr, factor);			// Scale them
			inputL += 4;
			inputR += 4;

			// First two stereo pairs
			__m128 f12 = _mm_shuffle_ps(fl, fr, _MM_SHUFFLE(1, 0, 1, 0));	// L1 L2 R1 R2
			f12 = _mm_shuffle_ps(f12 , f12, _MM_SHUFFLE(3, 1, 2, 0));		// L1 R1 L2 R2
			__m64 i1 = _mm_cvtps_pi32(f12);									// Convert to two ints, L1R1
			f12 = _mm_shuffle_ps(f12 , f12, _MM_SHUFFLE(1, 0, 3, 2));		// L2 R2 L1 R1
			__m64 i2 = _mm_cvtps_pi32(f12);									// Convert to two ints, L2R2
			__m64 sat12 = _mm_packs_pi32(i1, i2);							// Pack and saturate them to 16-bit
			*(out++) = sat12;												// Store L1R1L2R2

			// Second two stereo pairs
			__m128 f34 = _mm_shuffle_ps(fl, fr, _MM_SHUFFLE(3, 1, 3, 1));	// L3 L4 R3 R4
			f34 = _mm_shuffle_ps(f34 , f34, _MM_SHUFFLE(3, 1, 2, 0));		// L3 R3 L4 R4
			__m64 i3 = _mm_cvtps_pi32(f34);									// Convert to two ints, L3R3
			f34 = _mm_shuffle_ps(f34 , f34, _MM_SHUFFLE(1, 0, 3, 2));		// L4 R4 L3 R3
			__m64 i4 = _mm_cvtps_pi32(f34);									// Convert to two ints, L4R4
			__m64 sat34 = _mm_packs_pi32(i3, i4);							// Pack and saturate them to 16-bit
			*(out++) = sat34;												// Store L3R3L4R4
		} while(--numFrames);
		_mm_empty();
		return;
	}
#endif
	while(numFrames--)
	{
		*(output++) = static_cast<int16>(Clamp(*(inputL++) * _f2si, static_cast<float>(int16_min), static_cast<float>(int16_max)));
		*(output++) = static_cast<int16>(Clamp(*(inputR++) * _f2si, static_cast<float>(int16_min), static_cast<float>(int16_max)));
	}
}


// Deinterleave an int16 stereo stream into two float streams.
static void DeinterleaveInt16ToFloat(const int16 * MPT_RESTRICT input, float * MPT_RESTRICT outputL, float * MPT_RESTRICT outputR, uint32 numFrames)
{
#ifdef ENABLE_SSE
	// This uses __m64, so it's not available on the MSVC 64-bit compiler.
	// But if the user runs a 64-bit operating system, they will go the floating-point path anyway.
	if(GetProcSupport() & PROCSUPPORT_SSE)
	{
		// We may read beyond the wanted length... this works because we know that we will always work on our buffers of size MIXBUFFERSIZE
		STATIC_ASSERT((MIXBUFFERSIZE & 7) == 0);
		const __m128i *in = reinterpret_cast<const __m128i *>(input);
		__m128 factor = _mm_set_ps1(_si2f);
		numFrames = (numFrames + 3) / 4;
		do
		{
			__m128i in16 = _mm_load_si128(in);		// Load eight int16 values, LRLRLRLR
			in++;
			__m128i lo = _mm_unpacklo_epi16(_mm_setzero_si128(), in16);	// 0L0R0L0R (1)
			__m128i hi = _mm_unpackhi_epi16(_mm_setzero_si128(), in16);	// 0L0R0L0R (2)
			lo = _mm_srai_epi32(lo, 16);			// LsRsLsRs, s = sign (1)
			hi = _mm_srai_epi32(hi, 16);			// LsRsLsRs, s = sign (2)

			__m64 lo1, lo2, hi1, hi2;
			_mm_storel_pi(&lo1, _mm_castsi128_ps(lo));				// L1R1
			_mm_storeh_pi(&lo2, _mm_castsi128_ps(lo));				// L2R2
			_mm_storel_pi(&hi1, _mm_castsi128_ps(hi));				// L3R3
			_mm_storeh_pi(&hi2, _mm_castsi128_ps(hi));				// L4R4
			__m128 f1 = _mm_cvt_pi2ps(_mm_setzero_ps(), lo1);		// Convert to two floats, L1R1
			__m128 f2 = _mm_cvt_pi2ps(_mm_setzero_ps(), lo2);		// Convert to two floats, L2R2
			f1 = _mm_shuffle_ps(f1, f1, _MM_SHUFFLE(1, 0, 1, 0));	// Move to upper
			f2 = _mm_shuffle_ps(f2, f2, _MM_SHUFFLE(1, 0, 1, 0));	// Move to upper
			f1 = _mm_cvt_pi2ps(f1, hi1);							// Convert to two floats, L3R3 | L1R1
			f2 = _mm_cvt_pi2ps(f2, hi2);							// Convert to two floats, L4R4 | L2R2

			__m128 fl = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(0, 2, 0, 2));	// => L1L3L2L4
			__m128 fr = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(1, 3, 1, 3));	// => R1R3R2R4
			fl = _mm_shuffle_ps(fl, fl, _MM_SHUFFLE(3, 1, 2, 0));			// => L1L2L3L4
			fr = _mm_shuffle_ps(fr, fr, _MM_SHUFFLE(3, 1, 2, 0));			// => R1R2R3R4
			fl = _mm_mul_ps(fl, factor);			// Scale them
			fr = _mm_mul_ps(fr, factor);			// Scale them
			_mm_storeu_ps(outputL, fl);				// Store four float values, LLLL
			_mm_storeu_ps(outputR, fr);				// Store four float values, RRRR
			outputL += 4;
			outputR += 4;
		} while(--numFrames);
		_mm_empty();
		return;
	}
#endif
	while(numFrames--)
	{
		*outputL++ += _si2f * static_cast<float>(*input++);
		*outputR++ += _si2f * static_cast<float>(*input++);
	}
}


void DMOPlugin::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!numFrames || !m_mixBuffer.Ok())
		return;
	m_mixBuffer.ClearOutputBuffers(numFrames);
	REFERENCE_TIME startTime = Util::muldiv(m_SndFile.GetTotalSampleCount(), 10000000, m_nSamplesPerSec);
	
	if(m_useFloat)
	{
		InterleaveStereo(m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1), m_alignedBuffer.f32, numFrames);
		m_pMediaProcess->Process(numFrames * 2 * sizeof(float), reinterpret_cast<BYTE *>(m_alignedBuffer.f32), startTime, DMO_INPLACE_NORMAL);
		DeinterleaveStereo(m_alignedBuffer.f32, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
	} else
	{
		InterleaveFloatToInt16(m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1), m_alignedBuffer.i16, numFrames);
		m_pMediaProcess->Process(numFrames * 2 * sizeof(int16), reinterpret_cast<BYTE *>(m_alignedBuffer.i16), startTime, DMO_INPLACE_NORMAL);
		DeinterleaveInt16ToFloat(m_alignedBuffer.i16, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamIndex DMOPlugin::GetNumParameters() const
{
	DWORD dwParamCount = 0;
	m_pParamInfo->GetParamCount(&dwParamCount);
	return dwParamCount;
}


PlugParamValue DMOPlugin::GetParameter(PlugParamIndex index)
{
	if(index < GetNumParameters() && m_pParamInfo != nullptr && m_pMediaParams != nullptr)
	{
		MP_PARAMINFO mpi;
		MP_DATA md;

		MemsetZero(mpi);
		md = 0;
		if (m_pParamInfo->GetParamInfo(index, &mpi) == S_OK
			&& m_pMediaParams->GetParam(index, &md) == S_OK)
		{
			float fValue, fMin, fMax, fDefault;

			fValue = md;
			fMin = mpi.mpdMinValue;
			fMax = mpi.mpdMaxValue;
			fDefault = mpi.mpdNeutralValue;
			if (mpi.mpType == MPT_BOOL)
			{
				fMin = 0;
				fMax = 1;
			}
			fValue -= fMin;
			if (fMax > fMin) fValue /= (fMax - fMin);
			return fValue;
		}
	}
	return 0;
}


void DMOPlugin::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < GetNumParameters() && m_pParamInfo != nullptr && m_pMediaParams != nullptr)
	{
		MP_PARAMINFO mpi;
		MemsetZero(mpi);
		if (m_pParamInfo->GetParamInfo(index, &mpi) == S_OK)
		{
			float fMin = mpi.mpdMinValue;
			float fMax = mpi.mpdMaxValue;

			if (mpi.mpType == MPT_BOOL)
			{
				fMin = 0;
				fMax = 1;
				value = (value > 0.5f) ? 1.0f : 0.0f;
			}
			if (fMax > fMin) value *= (fMax - fMin);
			value += fMin;
			Limit(value, fMin, fMax);
			if (mpi.mpType != MPT_FLOAT) value = Util::Round(value);
			m_pMediaParams->SetParam(index, value);
		}
	}
}


void DMOPlugin::Resume()
{
	m_nSamplesPerSec = m_SndFile.GetSampleRate();
	m_isResumed = true;

	DMO_MEDIA_TYPE mt;
	WAVEFORMATEX wfx;

	mt.majortype = MEDIATYPE_Audio;
	mt.subtype = MEDIASUBTYPE_PCM;
	mt.bFixedSizeSamples = TRUE;
	mt.bTemporalCompression = FALSE;
	mt.formattype = FORMAT_WaveFormatEx;
	mt.pUnk = nullptr;
	mt.pbFormat = (LPBYTE)&wfx;
	mt.cbFormat = sizeof(WAVEFORMATEX);
	mt.lSampleSize = 2 * sizeof(float);
	wfx.wFormatTag = 3; // WAVE_FORMAT_IEEE_FLOAT;
	wfx.nChannels = 2;
	wfx.nSamplesPerSec = m_nSamplesPerSec;
	wfx.wBitsPerSample = sizeof(float) * 8;
	wfx.nBlockAlign = wfx.nChannels * (wfx.wBitsPerSample / 8);
	wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
	wfx.cbSize = 0;

	// First try 32-bit float (DirectX 9+)
	m_useFloat = true;
	if(FAILED(m_pMediaObject->SetInputType(0, &mt, 0))
		|| FAILED(m_pMediaObject->SetOutputType(0, &mt, 0)))
	{
		m_useFloat = false;
		// Try again with 16-bit PCM
		mt.lSampleSize = 2 * sizeof(int16);
		wfx.wFormatTag = WAVE_FORMAT_PCM;
		wfx.wBitsPerSample = sizeof(int16) * 8;
		wfx.nBlockAlign = wfx.nChannels * (wfx.wBitsPerSample / 8);
		wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
		if(FAILED(m_pMediaObject->SetInputType(0, &mt, 0))
			|| FAILED(m_pMediaObject->SetOutputType(0, &mt, 0)))
		{
#ifdef DMO_LOG
		Log(MPT_USTRING("DMO: Failed to set I/O media type"));
#endif
		}
	}
}


void DMOPlugin::PositionChanged()
{
	m_pMediaObject->Discontinuity(0);
	m_pMediaObject->Flush();
}


void DMOPlugin::Suspend()
{
	m_isResumed = false;
	m_pMediaObject->Flush();
	m_pMediaObject->SetInputType(0, nullptr, DMO_SET_TYPEF_CLEAR);
	m_pMediaObject->SetOutputType(0, nullptr, DMO_SET_TYPEF_CLEAR);
}


#ifdef MODPLUG_TRACKER

CString DMOPlugin::GetParamName(PlugParamIndex param)
{
	if(param < GetNumParameters() && m_pParamInfo != nullptr)
	{
		MP_PARAMINFO mpi;
		mpi.mpType = MPT_INT;
		mpi.szUnitText[0] = 0;
		mpi.szLabel[0] = 0;
		if(m_pParamInfo->GetParamInfo(param, &mpi) == S_OK)
		{
			return mpi.szLabel;
		}
	}
	return CString();

}


CString DMOPlugin::GetParamLabel(PlugParamIndex param)
{
	if(param < GetNumParameters() && m_pParamInfo != nullptr)
	{
		MP_PARAMINFO mpi;
		mpi.mpType = MPT_INT;
		mpi.szUnitText[0] = 0;
		mpi.szLabel[0] = 0;
		if(m_pParamInfo->GetParamInfo(param, &mpi) == S_OK)
		{
			return mpi.szUnitText;
		}
	}
	return CString();
}


CString DMOPlugin::GetParamDisplay(PlugParamIndex param)
{
	if(param < GetNumParameters() && m_pParamInfo != nullptr && m_pMediaParams != nullptr)
	{
		MP_PARAMINFO mpi;
		mpi.mpType = MPT_INT;
		mpi.szUnitText[0] = 0;
		mpi.szLabel[0] = 0;
		if (m_pParamInfo->GetParamInfo(param, &mpi) == S_OK)
		{
			MP_DATA md;
			if(m_pMediaParams->GetParam(param, &md) == S_OK)
			{
				switch(mpi.mpType)
				{
				case MPT_FLOAT:
					{
						CString s;
						s.Format(_T("%.2f"), md);
						return s;
					}
					break;

				case MPT_BOOL:
					return ((int)md) ? _T("Yes") : _T("No");
					break;

				case MPT_ENUM:
					{
						WCHAR *text = nullptr;
						m_pParamInfo->GetParamText(param, &text);

						const int nValue = Util::Round<int>(md * (mpi.mpdMaxValue - mpi.mpdMinValue));
						// Always skip first two strings (param name, unit name)
						for(int i = 0; i < nValue + 2; i++)
						{
							text += wcslen(text) + 1;
						}
						return CString(text);
					}
					break;

				case MPT_INT:
				default:
					{
						CString s;
						s.Format(_T("%d"), Util::Round<int>(md));
						return s;
					}
					break;
				}
			}
		}
	}
	return CString();
}

#endif // MODPLUG_TRACKER

#else // NO_DMO

MPT_MSVC_WORKAROUND_LNK4221(DMOPlugin)

#endif // !NO_DMO

OPENMPT_NAMESPACE_END

