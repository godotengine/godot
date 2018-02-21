/*
 * AGC.h
 * -----
 * Purpose: Automatic Gain Control
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once


OPENMPT_NAMESPACE_BEGIN


#ifndef NO_AGC

class CAGC
{
private:
	UINT m_nAGC;
	std::size_t m_nAGCRecoverCount;
	UINT m_Timeout;
public:
	CAGC();
	void Initialize(bool bReset, DWORD MixingFreq);
public:
	void Process(int *MixSoundBuffer, int *RearSoundBuffer, std::size_t count, std::size_t nChannels);
	void Adjust(UINT oldVol, UINT newVol);
};

#endif // NO_AGC


OPENMPT_NAMESPACE_END
