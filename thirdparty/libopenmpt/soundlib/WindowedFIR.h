/*
 * WindowedFIR.h
 * -------------
 * Purpose: FIR resampling code
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 *          ModPlug-XMMS Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "Mixer.h"

OPENMPT_NAMESPACE_BEGIN

/*
  ------------------------------------------------------------------------------------------------
   fir interpolation doc,
	(derived from "an engineer's guide to fir digital filters", n.j. loy)

	calculate coefficients for ideal lowpass filter (with cutoff = fc in 0..1 (mapped to 0..nyquist))
	  c[-N..N] = (i==0) ? fc : sin(fc*pi*i)/(pi*i)

	then apply selected window to coefficients
	  c[-N..N] *= w(0..N)
	with n in 2*N and w(n) being a window function (see loy)

	then calculate gain and scale filter coefs to have unity gain.
  ------------------------------------------------------------------------------------------------
*/

#ifdef MPT_INTMIXER
// quantizer scale of window coefs - only required for integer mixing
#define WFIR_QUANTBITS		15
#define WFIR_QUANTSCALE		float(1L<<WFIR_QUANTBITS)
#define WFIR_8SHIFT			(WFIR_QUANTBITS-8)
#define WFIR_16BITSHIFT		(WFIR_QUANTBITS)
typedef int16 WFIR_TYPE;
#else
typedef mixsample_t WFIR_TYPE;
#endif // INTMIXER
// log2(number)-1 of precalculated taps range is [4..12]
#define WFIR_FRACBITS		12 //10
#define WFIR_LUTLEN			((1L<<(WFIR_FRACBITS+1))+1)
// number of samples in window
#define WFIR_LOG2WIDTH		3
#define WFIR_WIDTH			(1L<<WFIR_LOG2WIDTH)
// cutoff (1.0 == pi/2)
//float WFIR_CUTOFF			= 0.5f;//0.75f;	//0.90f;
// wfir type
enum WFIRType
{
	WFIR_HANN			= 0,
	WFIR_HAMMING		= 1,
	WFIR_BLACKMANEXACT	= 2,
	WFIR_BLACKMAN3T61	= 3,
	WFIR_BLACKMAN3T67	= 4,
	WFIR_BLACKMAN4T92	= 5,
	WFIR_BLACKMAN4T74	= 6,
	WFIR_KAISER4T		= 7,
};
//int WFIR_TYPE				= WFIR_KAISER4T;//WFIR_BLACKMANEXACT;
//int WFIR_TYPE				= TrackerSettings::Instance().gbWFIRType;
// wfir help
#ifndef M_zPI
#define M_zPI				3.1415926535897932384626433832795
#endif
#define M_zEPS				1e-8


// fir interpolation
#define WFIR_FRACSHIFT	(16-(WFIR_FRACBITS+1+WFIR_LOG2WIDTH))
#define WFIR_FRACMASK	((((1L<<(17-WFIR_FRACSHIFT))-1)&~((1L<<WFIR_LOG2WIDTH)-1)))
#define WFIR_FRACHALVE	(1L<<(16-(WFIR_FRACBITS+2)))

class CWindowedFIR
{
public:
	float coef(int,float,float,int,int);
	void InitTable(double WFIRCutoff, uint8 WFIRType);
	WFIR_TYPE lut[WFIR_LUTLEN*WFIR_WIDTH];
};

OPENMPT_NAMESPACE_END
