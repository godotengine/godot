/*
 * SampleFormat.h
 * ---------------
 * Purpose: Utility enum and funcion to describe sample formats.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once


OPENMPT_NAMESPACE_BEGIN


enum SampleFormatEnum
{
	SampleFormatUnsigned8 =  8,       // do not change value (for compatibility with old configuration settings)
	SampleFormatInt16     = 16,       // do not change value (for compatibility with old configuration settings)
	SampleFormatInt24     = 24,       // do not change value (for compatibility with old configuration settings)
	SampleFormatInt32     = 32,       // do not change value (for compatibility with old configuration settings)
	SampleFormatFloat32   = 32 + 128, // do not change value (for compatibility with old configuration settings)
	SampleFormatInvalid   =  0
};

template<typename Tsample> struct SampleFormatTraits;
template<> struct SampleFormatTraits<uint8>     { static MPT_CONSTEXPR11_VAR SampleFormatEnum sampleFormat = SampleFormatUnsigned8; };
template<> struct SampleFormatTraits<int16>     { static MPT_CONSTEXPR11_VAR SampleFormatEnum sampleFormat = SampleFormatInt16;     };
template<> struct SampleFormatTraits<int24>     { static MPT_CONSTEXPR11_VAR SampleFormatEnum sampleFormat = SampleFormatInt24;     };
template<> struct SampleFormatTraits<int32>     { static MPT_CONSTEXPR11_VAR SampleFormatEnum sampleFormat = SampleFormatInt32;     };
template<> struct SampleFormatTraits<float>     { static MPT_CONSTEXPR11_VAR SampleFormatEnum sampleFormat = SampleFormatFloat32;   };

template<SampleFormatEnum sampleFormat> struct SampleFormatToType;
template<> struct SampleFormatToType<SampleFormatUnsigned8> { typedef uint8     type; };
template<> struct SampleFormatToType<SampleFormatInt16>     { typedef int16     type; };
template<> struct SampleFormatToType<SampleFormatInt24>     { typedef int24     type; };
template<> struct SampleFormatToType<SampleFormatInt32>     { typedef int32     type; };
template<> struct SampleFormatToType<SampleFormatFloat32>   { typedef float     type; };


struct SampleFormat
{
	SampleFormatEnum value;
	MPT_CONSTEXPR11_FUN SampleFormat(SampleFormatEnum v = SampleFormatInvalid) : value(v) { }
	MPT_CONSTEXPR11_FUN bool operator == (SampleFormat other) const { return value == other.value; }
	MPT_CONSTEXPR11_FUN bool operator != (SampleFormat other) const { return value != other.value; }
	MPT_CONSTEXPR11_FUN bool operator == (SampleFormatEnum other) const { return value == other; }
	MPT_CONSTEXPR11_FUN bool operator != (SampleFormatEnum other) const { return value != other; }
	MPT_CONSTEXPR11_FUN operator SampleFormatEnum () const
	{
		return value;
	}
	MPT_CONSTEXPR11_FUN bool IsValid() const
	{
		return value != SampleFormatInvalid;
	}
	MPT_CONSTEXPR11_FUN bool IsUnsigned() const
	{
		return IsValid() && (value == SampleFormatUnsigned8);
	}
	MPT_CONSTEXPR11_FUN bool IsFloat() const
	{
		return IsValid() && (value == SampleFormatFloat32);
	}
	MPT_CONSTEXPR11_FUN bool IsInt() const
	{
		return IsValid() && (value != SampleFormatFloat32);
	}
	MPT_CONSTEXPR11_FUN uint8 GetBitsPerSample() const
	{
		return
			!IsValid() ? 0 :
			(value == SampleFormatUnsigned8) ?  8 :
			(value == SampleFormatInt16)     ? 16 :
			(value == SampleFormatInt24)     ? 24 :
			(value == SampleFormatInt32)     ? 32 :
			(value == SampleFormatFloat32)   ? 32 :
			0;
	}

	// backward compatibility, conversion to/from integers
	MPT_CONSTEXPR11_FUN operator int () const { return value; }
	MPT_CONSTEXPR11_FUN SampleFormat(int v) : value(SampleFormatEnum(v)) { }
	MPT_CONSTEXPR11_FUN operator long () const { return value; }
	MPT_CONSTEXPR11_FUN SampleFormat(long v) : value(SampleFormatEnum(v)) { }
	MPT_CONSTEXPR11_FUN operator unsigned int () const { return value; }
	MPT_CONSTEXPR11_FUN SampleFormat(unsigned int v) : value(SampleFormatEnum(v)) { }
	MPT_CONSTEXPR11_FUN operator unsigned long () const { return value; }
	MPT_CONSTEXPR11_FUN SampleFormat(unsigned long v) : value(SampleFormatEnum(v)) { }
};


OPENMPT_NAMESPACE_END
