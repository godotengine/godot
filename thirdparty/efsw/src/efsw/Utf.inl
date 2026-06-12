// References :
// http://www.unicode.org/
// http://www.unicode.org/Public/PROGRAMS/CVTUTF/ConvertUTF.c
// http://www.unicode.org/Public/PROGRAMS/CVTUTF/ConvertUTF.h
// http://people.w3.org/rishida/scripts/uniview/conversion
////////////////////////////////////////////////////////////

template <typename In> In Utf<8>::Decode( In begin, In end, Uint32& output, Uint32 replacement ) {
	// Some useful precomputed data
	static const int trailing[256] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
		2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5 };
	static const Uint32 offsets[6] = { 0x00000000, 0x00003080, 0x000E2080,
									   0x03C82080, 0xFA082080, 0x82082080 };

	// Decode the character
	int trailingBytes = trailing[static_cast<Uint8>( *begin )];
	if ( begin + trailingBytes < end ) {
		output = 0;
		switch ( trailingBytes ) {
			case 5:
				output += static_cast<Uint8>( *begin++ );
				output <<= 6;
			case 4:
				output += static_cast<Uint8>( *begin++ );
				output <<= 6;
			case 3:
				output += static_cast<Uint8>( *begin++ );
				output <<= 6;
			case 2:
				output += static_cast<Uint8>( *begin++ );
				output <<= 6;
			case 1:
				output += static_cast<Uint8>( *begin++ );
				output <<= 6;
			case 0:
				output += static_cast<Uint8>( *begin++ );
		}
		output -= offsets[trailingBytes];
	} else {
		// Incomplete character
		begin = end;
		output = replacement;
	}

	return begin;
}

template <typename Out> Out Utf<8>::Encode( Uint32 input, Out output, Uint8 replacement ) {
	// Some useful precomputed data
	static const Uint8 firstBytes[7] = { 0x00, 0x00, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC };

	// Encode the character
	if ( ( input > 0x0010FFFF ) || ( ( input >= 0xD800 ) && ( input <= 0xDBFF ) ) ) {
		// Invalid character
		if ( replacement )
			*output++ = replacement;
	} else {
		// Valid character

		// Get the number of bytes to write
		int bytesToWrite = 1;
		if ( input < 0x80 )
			bytesToWrite = 1;
		else if ( input < 0x800 )
			bytesToWrite = 2;
		else if ( input < 0x10000 )
			bytesToWrite = 3;
		else if ( input <= 0x0010FFFF )
			bytesToWrite = 4;

		// Extract the bytes to write
		Uint8 bytes[4];
		switch ( bytesToWrite ) {
			case 4:
				bytes[3] = static_cast<Uint8>( ( input | 0x80 ) & 0xBF );
				input >>= 6;
			case 3:
				bytes[2] = static_cast<Uint8>( ( input | 0x80 ) & 0xBF );
				input >>= 6;
			case 2:
				bytes[1] = static_cast<Uint8>( ( input | 0x80 ) & 0xBF );
				input >>= 6;
			case 1:
				bytes[0] = static_cast<Uint8>( input | firstBytes[bytesToWrite] );
		}

		// Add them to the output
		const Uint8* currentByte = bytes;
		switch ( bytesToWrite ) {
			case 4:
				*output++ = *currentByte++;
			case 3:
				*output++ = *currentByte++;
			case 2:
				*output++ = *currentByte++;
			case 1:
				*output++ = *currentByte++;
		}
	}

	return output;
}

template <typename In> In Utf<8>::Next( In begin, In end ) {
	Uint32 codepoint;
	return Decode( begin, end, codepoint );
}

template <typename In> std::size_t Utf<8>::Count( In begin, In end ) {
	std::size_t length = 0;
	while ( begin < end ) {
		begin = Next( begin, end );
		++length;
	}

	return length;
}

template <typename In, typename Out>
Out Utf<8>::FromAnsi( In begin, In end, Out output, const std::locale& locale ) {
	while ( begin < end ) {
		Uint32 codepoint = Utf<32>::DecodeAnsi( *begin++, locale );
		output = Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<8>::FromWide( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint = Utf<32>::DecodeWide( *begin++ );
		output = Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<8>::FromLatin1( In begin, In end, Out output ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end )
		output = Encode( *begin++, output );

	return output;
}

template <typename In, typename Out>
Out Utf<8>::ToAnsi( In begin, In end, Out output, char replacement, const std::locale& locale ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<32>::EncodeAnsi( codepoint, output, replacement, locale );
	}

	return output;
}

#ifndef EFSW_NO_WIDECHAR
template <typename In, typename Out>
Out Utf<8>::ToWide( In begin, In end, Out output, wchar_t replacement ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<32>::EncodeWide( codepoint, output, replacement );
	}

	return output;
}
#endif

template <typename In, typename Out>
Out Utf<8>::ToLatin1( In begin, In end, Out output, char replacement ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		*output++ = codepoint < 256 ? static_cast<char>( codepoint ) : replacement;
	}

	return output;
}

template <typename In, typename Out> Out Utf<8>::toUtf8( In begin, In end, Out output ) {
	while ( begin < end )
		*output++ = *begin++;

	return output;
}

template <typename In, typename Out> Out Utf<8>::ToUtf16( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<16>::Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<8>::ToUtf32( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		*output++ = codepoint;
	}

	return output;
}

template <typename In> In Utf<16>::Decode( In begin, In end, Uint32& output, Uint32 replacement ) {
	Uint16 first = *begin++;

	// If it's a surrogate pair, first convert to a single UTF-32 character
	if ( ( first >= 0xD800 ) && ( first <= 0xDBFF ) ) {
		if ( begin < end ) {
			Uint32 second = *begin++;
			if ( ( second >= 0xDC00 ) && ( second <= 0xDFFF ) ) {
				// The second element is valid: convert the two elements to a UTF-32 character
				output = static_cast<Uint32>( ( ( first - 0xD800 ) << 10 ) + ( second - 0xDC00 ) +
											  0x0010000 );
			} else {
				// Invalid character
				output = replacement;
			}
		} else {
			// Invalid character
			begin = end;
			output = replacement;
		}
	} else {
		// We can make a direct copy
		output = first;
	}

	return begin;
}

template <typename Out> Out Utf<16>::Encode( Uint32 input, Out output, Uint16 replacement ) {
	if ( input < 0xFFFF ) {
		// The character can be copied directly, we just need to check if it's in the valid range
		if ( ( input >= 0xD800 ) && ( input <= 0xDFFF ) ) {
			// Invalid character (this range is reserved)
			if ( replacement )
				*output++ = replacement;
		} else {
			// Valid character directly convertible to a single UTF-16 character
			*output++ = static_cast<Uint16>( input );
		}
	} else if ( input > 0x0010FFFF ) {
		// Invalid character (greater than the maximum unicode value)
		if ( replacement )
			*output++ = replacement;
	} else {
		// The input character will be converted to two UTF-16 elements
		input -= 0x0010000;
		*output++ = static_cast<Uint16>( ( input >> 10 ) + 0xD800 );
		*output++ = static_cast<Uint16>( ( input & 0x3FFUL ) + 0xDC00 );
	}

	return output;
}

template <typename In> In Utf<16>::Next( In begin, In end ) {
	Uint32 codepoint;
	return Decode( begin, end, codepoint );
}

template <typename In> std::size_t Utf<16>::Count( In begin, In end ) {
	std::size_t length = 0;
	while ( begin < end ) {
		begin = Next( begin, end );
		++length;
	}

	return length;
}

template <typename In, typename Out>
Out Utf<16>::FromAnsi( In begin, In end, Out output, const std::locale& locale ) {
	while ( begin < end ) {
		Uint32 codepoint = Utf<32>::DecodeAnsi( *begin++, locale );
		output = Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<16>::FromWide( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint = Utf<32>::DecodeWide( *begin++ );
		output = Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<16>::FromLatin1( In begin, In end, Out output ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end )
		*output++ = *begin++;

	return output;
}

template <typename In, typename Out>
Out Utf<16>::ToAnsi( In begin, In end, Out output, char replacement, const std::locale& locale ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<32>::EncodeAnsi( codepoint, output, replacement, locale );
	}

	return output;
}

#ifndef EFSW_NO_WIDECHAR
template <typename In, typename Out>
Out Utf<16>::ToWide( In begin, In end, Out output, wchar_t replacement ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<32>::EncodeWide( codepoint, output, replacement );
	}

	return output;
}
#endif

template <typename In, typename Out>
Out Utf<16>::ToLatin1( In begin, In end, Out output, char replacement ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end ) {
		*output++ = *begin < 256 ? static_cast<char>( *begin ) : replacement;
		begin++;
	}

	return output;
}

template <typename In, typename Out> Out Utf<16>::toUtf8( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		output = Utf<8>::Encode( codepoint, output );
	}

	return output;
}

template <typename In, typename Out> Out Utf<16>::ToUtf16( In begin, In end, Out output ) {
	while ( begin < end )
		*output++ = *begin++;

	return output;
}

template <typename In, typename Out> Out Utf<16>::ToUtf32( In begin, In end, Out output ) {
	while ( begin < end ) {
		Uint32 codepoint;
		begin = Decode( begin, end, codepoint );
		*output++ = codepoint;
	}

	return output;
}

template <typename In> In Utf<32>::Decode( In begin, In end, Uint32& output, Uint32 ) {
	output = *begin++;
	return begin;
}

template <typename Out> Out Utf<32>::Encode( Uint32 input, Out output, Uint32 replacement ) {
	*output++ = input;
	return output;
}

template <typename In> In Utf<32>::Next( In begin, In end ) {
	return ++begin;
}

template <typename In> std::size_t Utf<32>::Count( In begin, In end ) {
	return begin - end;
}

template <typename In, typename Out>
Out Utf<32>::FromAnsi( In begin, In end, Out output, const std::locale& locale ) {
	while ( begin < end )
		*output++ = DecodeAnsi( *begin++, locale );

	return output;
}

template <typename In, typename Out> Out Utf<32>::FromWide( In begin, In end, Out output ) {
	while ( begin < end )
		*output++ = DecodeWide( *begin++ );

	return output;
}

template <typename In, typename Out> Out Utf<32>::FromLatin1( In begin, In end, Out output ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end )
		*output++ = *begin++;

	return output;
}

template <typename In, typename Out>
Out Utf<32>::ToAnsi( In begin, In end, Out output, char replacement, const std::locale& locale ) {
	while ( begin < end )
		output = EncodeAnsi( *begin++, output, replacement, locale );

	return output;
}

#ifndef EFSW_NO_WIDECHAR
template <typename In, typename Out>
Out Utf<32>::ToWide( In begin, In end, Out output, wchar_t replacement ) {
	while ( begin < end )
		output = EncodeWide( *begin++, output, replacement );

	return output;
}
#endif

template <typename In, typename Out>
Out Utf<32>::ToLatin1( In begin, In end, Out output, char replacement ) {
	// Latin-1 is directly compatible with Unicode encodings,
	// and can thus be treated as (a sub-range of) UTF-32
	while ( begin < end ) {
		*output++ = *begin < 256 ? static_cast<char>( *begin ) : replacement;
		begin++;
	}

	return output;
}

template <typename In, typename Out> Out Utf<32>::toUtf8( In begin, In end, Out output ) {
	while ( begin < end )
		output = Utf<8>::Encode( *begin++, output );

	return output;
}

template <typename In, typename Out> Out Utf<32>::ToUtf16( In begin, In end, Out output ) {
	while ( begin < end )
		output = Utf<16>::Encode( *begin++, output );

	return output;
}

template <typename In, typename Out> Out Utf<32>::ToUtf32( In begin, In end, Out output ) {
	while ( begin < end )
		*output++ = *begin++;

	return output;
}

template <typename In> Uint32 Utf<32>::DecodeAnsi( In input, const std::locale& locale ) {
	// On Windows, gcc's standard library (glibc++) has almost
	// no support for Unicode stuff. As a consequence, in this
	// context we can only use the default locale and ignore
	// the one passed as parameter.

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN && /* if Windows ... */                  \
	( defined( __GLIBCPP__ ) ||                                                 \
	  defined( __GLIBCXX__ ) ) && /* ... and standard library is glibc++ ... */ \
	!( defined( __SGI_STL_PORT ) ||                                             \
	   defined( _STLPORT_VERSION ) ) /* ... and STLPort is not used on top of it */

	wchar_t character = 0;
	mbtowc( &character, &input, 1 );
	return static_cast<Uint32>( character );

#else
// Get the facet of the locale which deals with character conversion
#ifndef EFSW_NO_WIDECHAR
	const std::ctype<wchar_t>& facet = std::use_facet<std::ctype<wchar_t>>( locale );
#else
	const std::ctype<char>& facet = std::use_facet<std::ctype<char>>( locale );
#endif

	// Use the facet to convert each character of the input string
	return static_cast<Uint32>( facet.widen( input ) );

#endif
}

template <typename In> Uint32 Utf<32>::DecodeWide( In input ) {
	// The encoding of wide characters is not well defined and is left to the system;
	// however we can safely assume that it is UCS-2 on Windows and
	// UCS-4 on Unix systems.
	// In both cases, a simple copy is enough (UCS-2 is a subset of UCS-4,
	// and UCS-4 *is* UTF-32).

	return input;
}

template <typename Out>
Out Utf<32>::EncodeAnsi( Uint32 codepoint, Out output, char replacement,
						 const std::locale& locale ) {
	// On Windows, gcc's standard library (glibc++) has almost
	// no support for Unicode stuff. As a consequence, in this
	// context we can only use the default locale and ignore
	// the one passed as parameter.

#if EFSW_PLATFORM == EFSW_PLATFORM_WIN && /* if Windows ... */                  \
	( defined( __GLIBCPP__ ) ||                                                 \
	  defined( __GLIBCXX__ ) ) && /* ... and standard library is glibc++ ... */ \
	!( defined( __SGI_STL_PORT ) ||                                             \
	   defined( _STLPORT_VERSION ) ) /* ... and STLPort is not used on top of it */

	char character = 0;
	if ( wctomb( &character, static_cast<wchar_t>( codepoint ) ) >= 0 )
		*output++ = character;
	else if ( replacement )
		*output++ = replacement;

	return output;

#else
// Get the facet of the locale which deals with character conversion
#ifndef EFSW_NO_WIDECHAR
	const std::ctype<wchar_t>& facet = std::use_facet<std::ctype<wchar_t>>( locale );
#else
	const std::ctype<char>& facet = std::use_facet<std::ctype<char>>( locale );
#endif

	// Use the facet to convert each character of the input string
	*output++ = facet.narrow( static_cast<wchar_t>( codepoint ), replacement );

	return output;

#endif
}

#ifndef EFSW_NO_WIDECHAR
template <typename Out>
Out Utf<32>::EncodeWide( Uint32 codepoint, Out output, wchar_t replacement ) {
	// The encoding of wide characters is not well defined and is left to the system;
	// however we can safely assume that it is UCS-2 on Windows and
	// UCS-4 on Unix systems.
	// For UCS-2 we need to check if the source characters fits in (UCS-2 is a subset of UCS-4).
	// For UCS-4 we can do a direct copy (UCS-4 *is* UTF-32).

	switch ( sizeof( wchar_t ) ) {
		case 4: {
			*output++ = static_cast<wchar_t>( codepoint );
			break;
		}

		default: {
			if ( ( codepoint <= 0xFFFF ) && ( ( codepoint < 0xD800 ) || ( codepoint > 0xDFFF ) ) ) {
				*output++ = static_cast<wchar_t>( codepoint );
			} else if ( replacement ) {
				*output++ = replacement;
			}
			break;
		}
	}

	return output;
}
#endif
