/*
 * WindowedFIR.cpp
 * ---------------
 * Purpose: FIR resampling code
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 *          ModPlug-XMMS Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "WindowedFIR.h"
#include <math.h>

OPENMPT_NAMESPACE_BEGIN

// -> CODE#0025
// -> DESC="enable polyphase resampling on stereo samples"
// -> !!! stolen from modplug-xmms sourceforge project !!!


float CWindowedFIR::coef( int _PCnr, float _POfs, float _PCut, int _PWidth, int _PType ) //float _PPos, float _PFc, int _PLen )
{	
	double	_LWidthM1       = _PWidth-1;
	double	_LWidthM1Half   = 0.5*_LWidthM1;
	double	_LPosU          = ((double)_PCnr - _POfs);
	double	_LPos           = _LPosU-_LWidthM1Half;
	double	_LPIdl          = 2.0*M_zPI/_LWidthM1;
	double	_LWc,_LSi;
	if(fabs(_LPos) < M_zEPS)
	{
		_LWc	= 1.0;
		_LSi	= _PCut;
	} else
	{
		switch( _PType )
		{	
		case WFIR_HANN:
			_LWc = 0.50 - 0.50 * cos(_LPIdl*_LPosU);
			break;
		case WFIR_HAMMING:
			_LWc = 0.54 - 0.46 * cos(_LPIdl*_LPosU);
			break;
		case WFIR_BLACKMANEXACT:
			_LWc = 0.42 - 0.50 * cos(_LPIdl*_LPosU) + 0.08 * cos(2.0*_LPIdl*_LPosU);
			break;
		case WFIR_BLACKMAN3T61:
			_LWc = 0.44959 - 0.49364 * cos(_LPIdl*_LPosU) + 0.05677 * cos(2.0*_LPIdl*_LPosU);
			break;
		case WFIR_BLACKMAN3T67:
			_LWc = 0.42323 - 0.49755 * cos(_LPIdl*_LPosU) + 0.07922 * cos(2.0*_LPIdl*_LPosU);
			break;
		case WFIR_BLACKMAN4T92: // blackman harris
			_LWc = 0.35875 - 0.48829 * cos(_LPIdl*_LPosU) + 0.14128 * cos(2.0*_LPIdl*_LPosU) - 0.01168 * cos(3.0*_LPIdl*_LPosU);
			break;
		case WFIR_BLACKMAN4T74:
			_LWc = 0.40217 - 0.49703 * cos(_LPIdl*_LPosU) + 0.09392 * cos(2.0*_LPIdl*_LPosU) - 0.00183 * cos(3.0*_LPIdl*_LPosU);
			break;
		case WFIR_KAISER4T: // kaiser-bessel, alpha~7.5
			_LWc = 0.40243 - 0.49804 * cos(_LPIdl*_LPosU) + 0.09831 * cos(2.0*_LPIdl*_LPosU) - 0.00122 * cos(3.0*_LPIdl*_LPosU);
			break;
		default:
			_LWc = 1.0;
			break;
		}
		_LPos	 *= M_zPI;
		_LSi	 = sin(_PCut*_LPos)/_LPos;
	}
	return (float)(_LWc*_LSi);
}

void CWindowedFIR::InitTable(double WFIRCutoff, uint8 WFIRType)
{
	int _LPcl;
	float _LPcllen	= (float)(1L<<WFIR_FRACBITS);	// number of precalculated lines for 0..1 (-1..0)
	float _LNorm	= 1.0f / (float)(2.0f * _LPcllen);
	float _LCut		= static_cast<float>(WFIRCutoff);
	for( _LPcl=0;_LPcl<WFIR_LUTLEN;_LPcl++ )
	{	
		float _LGain,_LCoefs[WFIR_WIDTH];
		float _LOfs		= ((float)_LPcl-_LPcllen)*_LNorm;
		int _LCc,_LIdx	= _LPcl<<WFIR_LOG2WIDTH;
		for( _LCc=0,_LGain=0.0f;_LCc<WFIR_WIDTH;_LCc++ )
		{	_LGain	+= (_LCoefs[_LCc] = coef( _LCc, _LOfs, _LCut, WFIR_WIDTH, WFIRType));
		}
		_LGain = 1.0f/_LGain;
		for( _LCc=0;_LCc<WFIR_WIDTH;_LCc++ )
		{
#ifdef MPT_INTMIXER
			float _LCoef = floorf( 0.5f + WFIR_QUANTSCALE*_LCoefs[_LCc]*_LGain );
			lut[_LIdx+_LCc] = (signed short)( (_LCoef<-WFIR_QUANTSCALE)?-WFIR_QUANTSCALE:((_LCoef>WFIR_QUANTSCALE)?WFIR_QUANTSCALE:_LCoef) );
#else
			float _LCoef = _LCoefs[_LCc] * _LGain;
			lut[_LIdx+_LCc] = _LCoef;
#endif // MPT_INTMIXER
		}
	}
}


OPENMPT_NAMESPACE_END
