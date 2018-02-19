/*
 * IntMixer.h
 * ----------
 * Purpose: Fixed point mixer classes
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "Resampler.h"
#include "MixerInterface.h"
#include "Paula.h"

OPENMPT_NAMESPACE_BEGIN

template<int channelsOut, int channelsIn, typename out, typename in, size_t mixPrecision>
struct IntToIntTraits : public MixerTraits<channelsOut, channelsIn, out, in>
{
	typedef MixerTraits<channelsOut, channelsIn, out, in> base_t;
	typedef typename base_t::input_t input_t;
	typedef typename base_t::output_t output_t;

	static MPT_CONSTEXPR11_FUN output_t Convert(const input_t x)
	{
		static_assert(std::numeric_limits<input_t>::is_integer, "Input must be integer");
		static_assert(std::numeric_limits<output_t>::is_integer, "Output must be integer");
		static_assert(sizeof(out) * 8 >= mixPrecision, "Mix precision is higher than output type can handle");
		static_assert(sizeof(in) * 8 <= mixPrecision, "Mix precision is lower than input type");
		return static_cast<output_t>(x) * (1<<(mixPrecision - sizeof(in) * 8));
	}
};

typedef IntToIntTraits<2, 1, mixsample_t, int8,  16> Int8MToIntS;
typedef IntToIntTraits<2, 1, mixsample_t, int16, 16> Int16MToIntS;
typedef IntToIntTraits<2, 2, mixsample_t, int8,  16> Int8SToIntS;
typedef IntToIntTraits<2, 2, mixsample_t, int16, 16> Int16SToIntS;


//////////////////////////////////////////////////////////////////////////
// Interpolation templates


template<class Traits>
struct AmigaBlepInterpolation
{
	SamplePosition subIncrement;
	Paula::State *paula;
	int numSteps;
	bool filter;

	MPT_FORCEINLINE void Start(ModChannel &chn, const CResampler &)
	{
		paula = &chn.paulaState;
		numSteps = paula->numSteps;
		filter = chn.dwFlags[CHN_AMIGAFILTER];
		if(numSteps)
			subIncrement = chn.increment / paula->numSteps;
	}

	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		SamplePosition pos(0, posLo);
		// First, process steps of full length (one Amiga clock interval)
		for(int step = numSteps; step > 0; step--)
		{
			typename Traits::output_t inSample = 0;
			int32 posInt = pos.GetInt() * Traits::numChannelsIn;
			for(int32 i = 0; i < Traits::numChannelsIn; i++)
				inSample += Traits::Convert(inBuffer[posInt + i]);
			paula->InputSample(static_cast<int16>(inSample / (4 * Traits::numChannelsIn)));
			paula->Clock(Paula::MINIMUM_INTERVAL);
			pos += subIncrement;
		}
		paula->remainder += paula->stepRemainder;

		// Now, process any remaining integer clock amount < MINIMUM_INTERVAL
		uint32 remainClocks = paula->remainder.GetInt();
		if(remainClocks)
		{
			typename Traits::output_t inSample = 0;
			int32 posInt = pos.GetInt() * Traits::numChannelsIn;
			for(int32 i = 0; i < Traits::numChannelsIn; i++)
				inSample += Traits::Convert(inBuffer[posInt + i]);
			paula->InputSample(static_cast<int16>(inSample / (4 * Traits::numChannelsIn)));
			paula->Clock(remainClocks);
			paula->remainder.RemoveInt();
		}

		auto out = paula->OutputSample(filter);
		for(unsigned int i = 0; i < Traits::numChannelsOut; i++)
			outSample[i] = out;
	}
};


template<class Traits>
struct LinearInterpolation
{
	MPT_FORCEINLINE void Start(const ModChannel &, const CResampler &) { }

	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		static_assert(Traits::numChannelsIn <= Traits::numChannelsOut, "Too many input channels");
		const typename Traits::output_t fract = posLo >> 18u;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t srcVol = Traits::Convert(inBuffer[i]);
			typename Traits::output_t destVol = Traits::Convert(inBuffer[i + Traits::numChannelsIn]);

			outSample[i] = srcVol + ((fract * (destVol - srcVol)) / 16384);
		}
	}
};


template<class Traits>
struct FastSincInterpolation
{
	MPT_FORCEINLINE void Start(const ModChannel &, const CResampler &) { }
	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		static_assert(Traits::numChannelsIn <= Traits::numChannelsOut, "Too many input channels");
		const int16 *lut = CResampler::FastSincTable + ((posLo >> 22) & 0x3FC);

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			outSample[i] =
				 (lut[0] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
				+ lut[1] * Traits::Convert(inBuffer[i])
				+ lut[2] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
				+ lut[3] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])) / 16384;
		}
	}
};


template<class Traits>
struct PolyphaseInterpolation
{
	const SINC_TYPE *sinc;

	MPT_FORCEINLINE void Start(const ModChannel &chn, const CResampler &resampler)
	{
		#ifdef MODPLUG_TRACKER
			// Otherwise causes "warning C4100: 'resampler' : unreferenced formal parameter"
			// because all 3 tables are static members.
			// #pragma warning fails with this templated case for unknown reasons.
			MPT_UNREFERENCED_PARAMETER(resampler);
		#endif // MODPLUG_TRACKER
		sinc = (((chn.increment > SamplePosition(0x130000000ll)) || (chn.increment < SamplePosition(-0x130000000ll))) ?
			(((chn.increment > SamplePosition(0x180000000ll)) || (chn.increment < SamplePosition(-0x180000000ll))) ? resampler.gDownsample2x : resampler.gDownsample13x) : resampler.gKaiserSinc);
	}

	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		static_assert(Traits::numChannelsIn <= Traits::numChannelsOut, "Too many input channels");
		const SINC_TYPE *lut = sinc + ((posLo >> (32 - SINC_PHASES_BITS)) & SINC_MASK) * SINC_WIDTH;

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			outSample[i] =
				 (lut[0] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn])
				+ lut[1] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn])
				+ lut[2] * Traits::Convert(inBuffer[i - Traits::numChannelsIn])
				+ lut[3] * Traits::Convert(inBuffer[i])
				+ lut[4] * Traits::Convert(inBuffer[i + Traits::numChannelsIn])
				+ lut[5] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn])
				+ lut[6] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn])
				+ lut[7] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn])) / (1 << SINC_QUANTSHIFT);
		}
	}
};


template<class Traits>
struct FIRFilterInterpolation
{
	const int16 *WFIRlut;

	MPT_FORCEINLINE void Start(const ModChannel &, const CResampler &resampler)
	{
		WFIRlut = resampler.m_WindowedFIR.lut;
	}

	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const typename Traits::input_t * const MPT_RESTRICT inBuffer, const uint32 posLo)
	{
		static_assert(Traits::numChannelsIn <= Traits::numChannelsOut, "Too many input channels");
		const int16 * const lut = WFIRlut + ((((posLo >> 16) + WFIR_FRACHALVE) >> WFIR_FRACSHIFT) & WFIR_FRACMASK);

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t vol1 =
				  (lut[0] * Traits::Convert(inBuffer[i - 3 * Traits::numChannelsIn]))
				+ (lut[1] * Traits::Convert(inBuffer[i - 2 * Traits::numChannelsIn]))
				+ (lut[2] * Traits::Convert(inBuffer[i - Traits::numChannelsIn]))
				+ (lut[3] * Traits::Convert(inBuffer[i]));
			typename Traits::output_t vol2 =
				  (lut[4] * Traits::Convert(inBuffer[i + 1 * Traits::numChannelsIn]))
				+ (lut[5] * Traits::Convert(inBuffer[i + 2 * Traits::numChannelsIn]))
				+ (lut[6] * Traits::Convert(inBuffer[i + 3 * Traits::numChannelsIn]))
				+ (lut[7] * Traits::Convert(inBuffer[i + 4 * Traits::numChannelsIn]));
			outSample[i] = ((vol1 / 2) + (vol2 / 2)) / (1 << (WFIR_16BITSHIFT - 1));
		}
	}
};


//////////////////////////////////////////////////////////////////////////
// Mixing templates (add sample to stereo mix)

template<class Traits>
struct NoRamp
{
	typename Traits::output_t lVol, rVol;

	MPT_FORCEINLINE void Start(const ModChannel &chn)
	{
		lVol = chn.leftVol;
		rVol = chn.rightVol;
	}

	MPT_FORCEINLINE void End(const ModChannel &) { }
};


struct Ramp
{
	int32 lRamp, rRamp;

	MPT_FORCEINLINE void Start(const ModChannel &chn)
	{
		lRamp = chn.rampLeftVol;
		rRamp = chn.rampRightVol;
	}

	MPT_FORCEINLINE void End(ModChannel &chn)
	{
		chn.rampLeftVol = lRamp; chn.leftVol = lRamp >> VOLUMERAMPPRECISION;
		chn.rampRightVol = rRamp; chn.rightVol = rRamp >> VOLUMERAMPPRECISION;
	}
};


// Legacy optimization: If chn.nLeftVol == chn.nRightVol, save one multiplication instruction
template<class Traits>
struct MixMonoFastNoRamp : public NoRamp<Traits>
{
	typedef NoRamp<Traits> base_t;
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const MPT_RESTRICT outBuffer)
	{
		typename Traits::output_t vol = outSample[0] * base_t::lVol;
		for(int i = 0; i < Traits::numChannelsOut; i++)
		{
			outBuffer[i] += vol;
		}
	}
};


template<class Traits>
struct MixMonoNoRamp : public NoRamp<Traits>
{
	typedef NoRamp<Traits> base_t;
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const MPT_RESTRICT outBuffer)
	{
		outBuffer[0] += outSample[0] * base_t::lVol;
		outBuffer[1] += outSample[0] * base_t::rVol;
	}
};


template<class Traits>
struct MixMonoRamp : public Ramp
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &chn, typename Traits::output_t * const MPT_RESTRICT outBuffer)
	{
		lRamp += chn.leftRamp;
		rRamp += chn.rightRamp;
		outBuffer[0] += outSample[0] * (lRamp >> VOLUMERAMPPRECISION);
		outBuffer[1] += outSample[0] * (rRamp >> VOLUMERAMPPRECISION);
	}
};


template<class Traits>
struct MixStereoNoRamp : public NoRamp<Traits>
{
	typedef NoRamp<Traits> base_t;
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &, typename Traits::output_t * const MPT_RESTRICT outBuffer)
	{
		outBuffer[0] += outSample[0] * base_t::lVol;
		outBuffer[1] += outSample[1] * base_t::rVol;
	}
};


template<class Traits>
struct MixStereoRamp : public Ramp
{
	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &outSample, const ModChannel &chn, typename Traits::output_t * const MPT_RESTRICT outBuffer)
	{
		lRamp += chn.leftRamp;
		rRamp += chn.rightRamp;
		outBuffer[0] += outSample[0] * (lRamp >> VOLUMERAMPPRECISION);
		outBuffer[1] += outSample[1] * (rRamp >> VOLUMERAMPPRECISION);
	}
};


//////////////////////////////////////////////////////////////////////////
// Filter templates


template<class Traits>
struct NoFilter
{
	MPT_FORCEINLINE void Start(const ModChannel &) { }
	MPT_FORCEINLINE void End(const ModChannel &) { }

	MPT_FORCEINLINE void operator() (const typename Traits::outbuf_t &, const ModChannel &) { }
};


// Resonant filter
template<class Traits>
struct ResonantFilter
{
	// Filter history
	typename Traits::output_t fy[Traits::numChannelsIn][2];

	MPT_FORCEINLINE void Start(const ModChannel &chn)
	{
		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			fy[i][0] = chn.nFilter_Y[i][0];
			fy[i][1] = chn.nFilter_Y[i][1];
		}
	}

	MPT_FORCEINLINE void End(ModChannel &chn)
	{
		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			chn.nFilter_Y[i][0] = fy[i][0];
			chn.nFilter_Y[i][1] = fy[i][1];
		}
	}

	// Filter values are clipped to double the input range
#define ClipFilter(x) Clamp<typename Traits::output_t, typename Traits::output_t>(x, int16_min * 2, int16_max * 2)

	MPT_FORCEINLINE void operator() (typename Traits::outbuf_t &outSample, const ModChannel &chn)
	{
		static_assert(Traits::numChannelsIn <= Traits::numChannelsOut, "Too many input channels");

		for(int i = 0; i < Traits::numChannelsIn; i++)
		{
			typename Traits::output_t val = static_cast<typename Traits::output_t>((
				Util::mul32to64(outSample[i], chn.nFilter_A0) +
				Util::mul32to64(ClipFilter(fy[i][0]), chn.nFilter_B0) +
				Util::mul32to64(ClipFilter(fy[i][1]), chn.nFilter_B1) +
				(1 << (MIXING_FILTER_PRECISION - 1))) / (1 << MIXING_FILTER_PRECISION));
			fy[i][1] = fy[i][0];
			fy[i][0] = val - (outSample[i] & chn.nFilter_HP);
			outSample[i] = val;
		}
	}

#undef ClipFilter
};


OPENMPT_NAMESPACE_END
