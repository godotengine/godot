#ifndef AUDIOFRAME_H
#define AUDIOFRAME_H

#include "typedefs.h"


static inline float undenormalise(volatile float f)
{
	union {
		uint32_t i;
		float f;
	} v;

	v.f = f;

	// original: return (v.i & 0x7f800000) == 0 ? 0.0f : f;
	// version from Tim Blechmann:
	return (v.i & 0x7f800000) < 0x08000000 ? 0.0f : f;
}


struct AudioFrame {

	//left and right samples
	float l,r;

	_ALWAYS_INLINE_ const float& operator[](int idx) const { return idx==0?l:r; }
	_ALWAYS_INLINE_ float& operator[](int idx) { return idx==0?l:r; }

	_ALWAYS_INLINE_ AudioFrame operator+(const AudioFrame& p_frame) const { return AudioFrame(l+p_frame.l,r+p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator-(const AudioFrame& p_frame) const { return AudioFrame(l-p_frame.l,r-p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator*(const AudioFrame& p_frame) const { return AudioFrame(l*p_frame.l,r*p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator/(const AudioFrame& p_frame) const { return AudioFrame(l/p_frame.l,r/p_frame.r); }

	_ALWAYS_INLINE_ AudioFrame operator+(float p_sample) const { return AudioFrame(l+p_sample,r+p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator-(float p_sample) const { return AudioFrame(l-p_sample,r-p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator*(float p_sample) const { return AudioFrame(l*p_sample,r*p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator/(float p_sample) const { return AudioFrame(l/p_sample,r/p_sample); }

	_ALWAYS_INLINE_ void operator+=(const AudioFrame& p_frame) { l+=p_frame.l; r+=p_frame.r; }
	_ALWAYS_INLINE_ void operator-=(const AudioFrame& p_frame) { l-=p_frame.l; r-=p_frame.r; }
	_ALWAYS_INLINE_ void operator*=(const AudioFrame& p_frame) { l*=p_frame.l; r*=p_frame.r; }
	_ALWAYS_INLINE_ void operator/=(const AudioFrame& p_frame) { l/=p_frame.l; r/=p_frame.r; }

	_ALWAYS_INLINE_ void operator+=(float p_sample) { l+=p_sample; r+=p_sample; }
	_ALWAYS_INLINE_ void operator-=(float p_sample) { l-=p_sample; r-=p_sample; }
	_ALWAYS_INLINE_ void operator*=(float p_sample) { l*=p_sample; r*=p_sample; }
	_ALWAYS_INLINE_ void operator/=(float p_sample) { l/=p_sample; r/=p_sample; }

	_ALWAYS_INLINE_ void undenormalise() {
		l = ::undenormalise(l);
		r = ::undenormalise(r);
	}

	_ALWAYS_INLINE_ AudioFrame(float p_l, float p_r) {l=p_l; r=p_r;}
	_ALWAYS_INLINE_ AudioFrame(const AudioFrame& p_frame) {l=p_frame.l; r=p_frame.r;}

	_ALWAYS_INLINE_ AudioFrame() {}
};

#endif
