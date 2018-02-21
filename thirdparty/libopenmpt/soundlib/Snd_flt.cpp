/*
 * snd_flt.cpp
 * -----------
 * Purpose: Calculation of resonant filter coefficients.
 * Notes  : Extended filter range was introduced in MPT 1.12 and went up to 8652 Hz.
 *          MPT 1.16 upped this to the current 10670 Hz.
 *          We have no way of telling whether a file was made with MPT 1.12 or 1.16 though.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "Tables.h"
#include "../common/misc_util.h"


OPENMPT_NAMESPACE_BEGIN


// AWE32: cutoff = reg[0-255] * 31.25 + 100 -> [100Hz-8060Hz]
// EMU10K1 docs: cutoff = reg[0-127]*62+100


uint8 CSoundFile::FrequencyToCutOff(double frequency) const
{
	// IT Cutoff is computed as cutoff = 110 * 2 ^ (0.25 + x/y), where x is the cutoff and y defines the filter range.
	// Reversed, this gives us x = (log2(cutoff / 110) - 0.25) * y.
	// <==========> Rewrite as x = (log2(cutoff) - log2(110) - 0.25) * y.
	// <==========> Rewrite as x = (ln(cutoff) - ln(110) - 0.25*ln(2)) * y/ln(2).
	//                                           <4.8737671609324025>
	double cutoff = (std::log(frequency) - 4.8737671609324025) * (m_SongFlags[SONG_EXFILTERRANGE] ? (20.0 / M_LN2) : (24.0 / M_LN2));
	Limit(cutoff, 0.0, 127.0);
	return Util::Round<uint8>(cutoff);
}


uint32 CSoundFile::CutOffToFrequency(uint32 nCutOff, int flt_modifier) const
{
	MPT_ASSERT(nCutOff < 128);
	float Fc = 110.0f * std::pow(2.0f, 0.25f + ((float)(nCutOff * (flt_modifier + 256))) / (m_SongFlags[SONG_EXFILTERRANGE] ? 20.0f * 512.0f : 24.0f * 512.0f));
	int freq = Util::Round<int>(Fc);
	Limit(freq, 120, 20000);
	if (freq * 2 > (int)m_MixerSettings.gdwMixingFreq) freq = m_MixerSettings.gdwMixingFreq / 2;
	return static_cast<uint32>(freq);
}


// Simple 2-poles resonant filter
void CSoundFile::SetupChannelFilter(ModChannel *pChn, bool bReset, int flt_modifier) const
{
	int cutoff = (int)pChn->nCutOff + (int)pChn->nCutSwing;
	int resonance = (int)(pChn->nResonance & 0x7F) + (int)pChn->nResSwing;

	Limit(cutoff, 0, 127);
	Limit(resonance, 0, 127);

	if(!m_playBehaviour[kMPTOldSwingBehaviour])
	{
		pChn->nCutOff = (uint8)cutoff;
		pChn->nCutSwing = 0;
		pChn->nResonance = (uint8)resonance;
		pChn->nResSwing = 0;
	}

	// flt_modifier is in [-256, 256], so cutoff is in [0, 127 * 2] after this calculation.
	const int computedCutoff = cutoff * (flt_modifier + 256) / 256;

	// Filtering is only ever done in IT if either cutoff is not full or if resonance is set.
	if(m_playBehaviour[kITFilterBehaviour] && resonance == 0 && computedCutoff >= 254)
	{
		if(pChn->rowCommand.IsNote() && !pChn->rowCommand.IsPortamento() && !pChn->nMasterChn && m_SongFlags[SONG_FIRSTTICK])
		{
			// Z7F next to a note disables the filter, however in other cases this should not happen.
			// Test cases: filter-reset.it, filter-reset-carry.it, filter-nna.it
			pChn->dwFlags.reset(CHN_FILTER);
		}
		return;
	}

	pChn->dwFlags.set(CHN_FILTER);

	// 2 * damping factor
	const float dmpfac = std::pow(10.0f, -resonance * ((24.0f / 128.0f) / 20.0f));
	const float fc = CutOffToFrequency(cutoff, flt_modifier) * (2.0f * (float)M_PI);
	float d, e;
	if(m_playBehaviour[kITFilterBehaviour] && !m_SongFlags[SONG_EXFILTERRANGE])
	{
		const float r = m_MixerSettings.gdwMixingFreq / fc;

		d = dmpfac * r + dmpfac - 1.0f;
		e = r * r;
	} else
	{
		const float r = fc / m_MixerSettings.gdwMixingFreq;

		d = (1.0f - 2.0f * dmpfac) * r;
		LimitMax(d, 2.0f);
		d = (2.0f * dmpfac - d) / r;
		e = 1.0f / (r * r);
	}

	float fg = 1.0f / (1.0f + d + e);
	float fb0 = (d + e + e) / (1 + d + e);
	float fb1 = -e / (1.0f + d + e);

#if defined(MPT_INTMIXER)
#define FILTER_CONVERT(x) Util::Round<mixsample_t>((x) * (1 << MIXING_FILTER_PRECISION))
#else
#define FILTER_CONVERT(x) (x)
#endif

	switch(pChn->nFilterMode)
	{
	case FLTMODE_HIGHPASS:
		pChn->nFilter_A0 = FILTER_CONVERT(1.0f - fg);
		pChn->nFilter_B0 = FILTER_CONVERT(fb0);
		pChn->nFilter_B1 = FILTER_CONVERT(fb1);
#ifdef MPT_INTMIXER
		pChn->nFilter_HP = -1;
#else
		pChn->nFilter_HP = 1.0f;
#endif // MPT_INTMIXER
		break;

	default:
		pChn->nFilter_A0 = FILTER_CONVERT(fg);
		pChn->nFilter_B0 = FILTER_CONVERT(fb0);
		pChn->nFilter_B1 = FILTER_CONVERT(fb1);
#ifdef MPT_INTMIXER
		if(pChn->nFilter_A0 == 0)
			pChn->nFilter_A0 = 1;	// Prevent silence at low filter cutoff and very high sampling rate
		pChn->nFilter_HP = 0;
#else
		pChn->nFilter_HP = 0;
#endif // MPT_INTMIXER
		break;
	}
#undef FILTER_CONVERT

	if (bReset)
	{
		pChn->nFilter_Y[0][0] = pChn->nFilter_Y[0][1] = 0;
		pChn->nFilter_Y[1][0] = pChn->nFilter_Y[1][1] = 0;
	}

}


OPENMPT_NAMESPACE_END
