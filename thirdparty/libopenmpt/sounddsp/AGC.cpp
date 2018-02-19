/*
 * AGC.cpp
 * -------
 * Purpose: Automatic Gain Control
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "../soundlib/Sndfile.h"
#include "../sounddsp/AGC.h"


OPENMPT_NAMESPACE_BEGIN

	
//////////////////////////////////////////////////////////////////////////////////
// Automatic Gain Control

#ifndef NO_AGC

#define AGC_PRECISION		10
#define AGC_UNITY			(1 << AGC_PRECISION)

// Limiter
#define MIXING_LIMITMAX		(0x08100000)
#define MIXING_LIMITMIN		(-MIXING_LIMITMAX)


static UINT ProcessAGC(int *pBuffer, int *pRearBuffer, std::size_t nSamples, std::size_t nChannels, int nAGC)
{
	if(nChannels == 1)
	{
		while(nSamples--)
		{
			int val = (int)(((int64)*pBuffer * (int32)nAGC) >> AGC_PRECISION);
			if(val < MIXING_LIMITMIN || val > MIXING_LIMITMAX) nAGC--;
			*pBuffer = val;
			pBuffer++;
		}
	} else
	{
		if(nChannels == 2)
		{
			while(nSamples--)
			{
				int fl = (int)(((int64)pBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
				int fr = (int)(((int64)pBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
				bool dec = false;
				dec = dec || (fl < MIXING_LIMITMIN || fl > MIXING_LIMITMAX);
				dec = dec || (fr < MIXING_LIMITMIN || fr > MIXING_LIMITMAX);
				if(dec) nAGC--;
				pBuffer[0] = fl;
				pBuffer[1] = fr;
				pBuffer += 2;
			}
		} else if(nChannels == 4)
		{
			while(nSamples--)
			{
				int fl = (int)(((int64)pBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
				int fr = (int)(((int64)pBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
				int rl = (int)(((int64)pRearBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
				int rr = (int)(((int64)pRearBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
				bool dec = false;
				dec = dec || (fl < MIXING_LIMITMIN || fl > MIXING_LIMITMAX);
				dec = dec || (fr < MIXING_LIMITMIN || fr > MIXING_LIMITMAX);
				dec = dec || (rl < MIXING_LIMITMIN || rl > MIXING_LIMITMAX);
				dec = dec || (rr < MIXING_LIMITMIN || rr > MIXING_LIMITMAX);
				if(dec) nAGC--;
				pBuffer[0] = fl;
				pBuffer[1] = fr;
				pRearBuffer[0] = rl;
				pRearBuffer[1] = rr;
				pBuffer += 2;
				pRearBuffer += 2;
			}
		}
	}
	return nAGC;
}


CAGC::CAGC()
{
	Initialize(true, 44100);
}


void CAGC::Process(int *MixSoundBuffer, int *RearSoundBuffer, std::size_t count, std::size_t nChannels)
{
	UINT agc = ProcessAGC(MixSoundBuffer, RearSoundBuffer, count, nChannels, m_nAGC);
	// Some kind custom law, so that the AGC stays quite stable, but slowly
	// goes back up if the sound level stays below a level inversely proportional
	// to the AGC level. (J'me comprends)
	if((agc >= m_nAGC) && (m_nAGC < AGC_UNITY))
	{
		m_nAGCRecoverCount += count;
		if(m_nAGCRecoverCount >= m_Timeout)
		{
			m_nAGCRecoverCount = 0;
			m_nAGC++;
		}
	} else
	{
		m_nAGC = agc;
		m_nAGCRecoverCount = 0;
	}
}


void CAGC::Adjust(UINT oldVol, UINT newVol)
{
	m_nAGC = m_nAGC * oldVol / newVol;
	if (m_nAGC > AGC_UNITY) m_nAGC = AGC_UNITY;
}


void CAGC::Initialize(bool bReset, DWORD MixingFreq)
{
	if(bReset)
	{
		m_nAGC = AGC_UNITY;
		m_nAGCRecoverCount = 0;
	}
	m_Timeout = (MixingFreq >> (AGC_PRECISION-8)) >> 1;
}


#else


MPT_MSVC_WORKAROUND_LNK4221(AGC)


#endif // NO_AGC


OPENMPT_NAMESPACE_END
