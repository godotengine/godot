#ifndef AUDIOFRAME_H
#define AUDIOFRAME_H

#include "typedefs.h"

struct AudioFrame {

	float l,r;

	_ALWAYS_INLINE_ const float& operator[](int idx) const { return idx==0?l:r; }
	_ALWAYS_INLINE_ float& operator[](int idx) { return idx==0?l:r; }

	_ALWAYS_INLINE_ AudioFrame operator+(const AudioFrame& p_frame) const { return AudioFrame(l+p_frame.l,r+p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator-(const AudioFrame& p_frame) const { return AudioFrame(l-p_frame.l,r-p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator*(const AudioFrame& p_frame) const { return AudioFrame(l*p_frame.l,r*p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator/(const AudioFrame& p_frame) const { return AudioFrame(l/p_frame.l,r/p_frame.r); }

	_ALWAYS_INLINE_ void operator+=(const AudioFrame& p_frame) { l+=p_frame.l; r+=p_frame.r; }
	_ALWAYS_INLINE_ void operator-=(const AudioFrame& p_frame) { l-=p_frame.l; r-=p_frame.r; }
	_ALWAYS_INLINE_ void operator*=(const AudioFrame& p_frame) { l*=p_frame.l; r*=p_frame.r; }
	_ALWAYS_INLINE_ void operator/=(const AudioFrame& p_frame) { l/=p_frame.l; r/=p_frame.r; }

	_ALWAYS_INLINE_ AudioFrame(float p_l, float p_r) {l=p_l; r=p_r;}
	_ALWAYS_INLINE_ AudioFrame(const AudioFrame& p_frame) {l=p_frame.l; r=p_frame.r;}

};

#endif
