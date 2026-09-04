#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS
{

using StringTransform = NS::String*;
using StringEncodingDetectionOptionsKey = NS::String*;
extern StringTransform const StringTransformLatinToKatakana __asm__("_NSStringTransformLatinToKatakana");
extern StringTransform const StringTransformLatinToHiragana __asm__("_NSStringTransformLatinToHiragana");
extern StringTransform const StringTransformLatinToHangul __asm__("_NSStringTransformLatinToHangul");
extern StringTransform const StringTransformLatinToArabic __asm__("_NSStringTransformLatinToArabic");
extern StringTransform const StringTransformLatinToHebrew __asm__("_NSStringTransformLatinToHebrew");
extern StringTransform const StringTransformLatinToThai __asm__("_NSStringTransformLatinToThai");
extern StringTransform const StringTransformLatinToCyrillic __asm__("_NSStringTransformLatinToCyrillic");
extern StringTransform const StringTransformLatinToGreek __asm__("_NSStringTransformLatinToGreek");
extern StringTransform const StringTransformToLatin __asm__("_NSStringTransformToLatin");
extern StringTransform const StringTransformMandarinToLatin __asm__("_NSStringTransformMandarinToLatin");
extern StringTransform const StringTransformHiraganaToKatakana __asm__("_NSStringTransformHiraganaToKatakana");
extern StringTransform const StringTransformFullwidthToHalfwidth __asm__("_NSStringTransformFullwidthToHalfwidth");
extern StringTransform const StringTransformToXMLHex __asm__("_NSStringTransformToXMLHex");
extern StringTransform const StringTransformToUnicodeName __asm__("_NSStringTransformToUnicodeName");
extern StringTransform const StringTransformStripCombiningMarks __asm__("_NSStringTransformStripCombiningMarks");
extern StringTransform const StringTransformStripDiacritics __asm__("_NSStringTransformStripDiacritics");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionSuggestedEncodingsKey __asm__("_NSStringEncodingDetectionSuggestedEncodingsKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionDisallowedEncodingsKey __asm__("_NSStringEncodingDetectionDisallowedEncodingsKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionUseOnlySuggestedEncodingsKey __asm__("_NSStringEncodingDetectionUseOnlySuggestedEncodingsKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionAllowLossyKey __asm__("_NSStringEncodingDetectionAllowLossyKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionFromWindowsKey __asm__("_NSStringEncodingDetectionFromWindowsKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionLossySubstitutionKey __asm__("_NSStringEncodingDetectionLossySubstitutionKey");
extern StringEncodingDetectionOptionsKey const StringEncodingDetectionLikelyLanguageKey __asm__("_NSStringEncodingDetectionLikelyLanguageKey");
extern NS::String* const CharacterConversionException __asm__("_NSCharacterConversionException");
extern NS::String* const ParseErrorException __asm__("_NSParseErrorException");
_NS_OPTIONS(NS::UInteger, StringCompareOptions) {
    CaseInsensitiveSearch = 1,
    LiteralSearch = 2,
    BackwardsSearch = 4,
    AnchoredSearch = 8,
    NumericSearch = 64,
    DiacriticInsensitiveSearch = 128,
    WidthInsensitiveSearch = 256,
    ForcedOrderingSearch = 512,
    RegularExpressionSearch = 1024,
};

_NS_ENUM(NS::UInteger, StringEncoding) {
    ASCIIStringEncoding = 1,
    NEXTSTEPStringEncoding = 2,
    JapaneseEUCStringEncoding = 3,
    UTF8StringEncoding = 4,
    ISOLatin1StringEncoding = 5,
    SymbolStringEncoding = 6,
    NonLossyASCIIStringEncoding = 7,
    ShiftJISStringEncoding = 8,
    ISOLatin2StringEncoding = 9,
    UnicodeStringEncoding = 10,
    WindowsCP1251StringEncoding = 11,
    WindowsCP1252StringEncoding = 12,
    WindowsCP1253StringEncoding = 13,
    WindowsCP1254StringEncoding = 14,
    WindowsCP1250StringEncoding = 15,
    ISO2022JPStringEncoding = 21,
    MacOSRomanStringEncoding = 30,
    UTF16StringEncoding = UnicodeStringEncoding,
    UTF16BigEndianStringEncoding = 0x90000100,
    UTF16LittleEndianStringEncoding = 0x94000100,
    UTF32StringEncoding = 0x8c000100,
    UTF32BigEndianStringEncoding = 0x98000100,
    UTF32LittleEndianStringEncoding = 0x9c000100,
};

_NS_OPTIONS(NS::UInteger, StringEncodingConversionOptions) {
    StringEncodingConversionAllowLossy = 1,
    StringEncodingConversionExternalRepresentation = 2,
};

_NS_OPTIONS(NS::UInteger, StringEnumerationOptions) {
    StringEnumerationByLines = 0,
    StringEnumerationByParagraphs = 1,
    StringEnumerationByComposedCharacterSequences = 2,
    StringEnumerationByWords = 3,
    StringEnumerationBySentences = 4,
    StringEnumerationByCaretPositions = 5,
    StringEnumerationByDeletionClusters = 6,
    StringEnumerationReverse = 1UL << 8,
    StringEnumerationSubstringNotRequired = 1UL << 9,
    StringEnumerationLocalized = 1UL << 10,
};


class String : public NS::SecureCoding<String>
{
public:
    static String* alloc();
    String*        init() const;

    static NS::String* string();
    static NS::String* string(NS::String* string);
    static NS::String* string(const char * cString, NS::StringEncoding enc);

    const char * cString(NS::StringEncoding encoding);
    long         caseInsensitiveCompare(NS::String* string);
    NS::unichar  character(NS::UInteger index);
    NS::String*  init(NS::String* aString);
    NS::String*  init(const void * bytes, NS::UInteger len, NS::StringEncoding encoding);
    NS::String*  init(void * bytes, NS::UInteger len, NS::StringEncoding encoding, bool freeBuffer);
    NS::String*  init(const char * nullTerminatedCString, NS::StringEncoding encoding);
    bool         isEqualToString(NS::String* aString);
    NS::UInteger length() const;
    NS::UInteger lengthOfBytes(NS::StringEncoding enc);
    NS::UInteger maximumLengthOfBytes(NS::StringEncoding enc);
    NS::Range    rangeOfString(NS::String* searchString, NS::StringCompareOptions mask);
    NS::String*  stringByAppendingString(NS::String* aString);
    const char * utf8String() const;

};

/// Create an NS::String* from a string literal.
#define MTLSTR(literal) (NS::String*)__builtin___CFStringMakeConstantString("" literal "")

template <std::size_t _StringLen>
[[deprecated("please use MTLSTR(str)")]] constexpr const String* MakeConstantString(const char (&str)[_StringLen])
{
    return reinterpret_cast<const String*>(__CFStringMakeConstantString(str));
}

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSString;

_NS_INLINE NS::String* NS::String::alloc()
{
    return _NS_msg_NS__Stringp_alloc((const void*)&OBJC_CLASS_$_NSString, nullptr);
}

_NS_INLINE NS::String* NS::String::init() const
{
    return _NS_msg_NS__Stringp_init((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::String::string()
{
    return _NS_msg_NS__Stringp_string((const void*)&OBJC_CLASS_$_NSString, nullptr);
}

_NS_INLINE NS::String* NS::String::string(NS::String* string)
{
    return _NS_msg_NS__Stringp_stringWithString__NS__Stringp((const void*)&OBJC_CLASS_$_NSString, nullptr, string);
}

_NS_INLINE NS::String* NS::String::string(const char * cString, NS::StringEncoding enc)
{
    return _NS_msg_NS__Stringp_stringWithCString_encoding__constcharp_NS__StringEncoding((const void*)&OBJC_CLASS_$_NSString, nullptr, cString, enc);
}

_NS_INLINE NS::UInteger NS::String::length() const
{
    return _NS_msg_NS__UInteger_length((const void*)this, nullptr);
}

_NS_INLINE const char * NS::String::utf8String() const
{
    return _NS_msg_constcharp_UTF8String((const void*)this, nullptr);
}

_NS_INLINE NS::unichar NS::String::character(NS::UInteger index)
{
    return _NS_msg_unsignedshort_characterAtIndex__NS__UInteger((const void*)this, nullptr, index);
}

_NS_INLINE long NS::String::caseInsensitiveCompare(NS::String* string)
{
    return _NS_msg_long_caseInsensitiveCompare__NS__Stringp((const void*)this, nullptr, string);
}

_NS_INLINE bool NS::String::isEqualToString(NS::String* aString)
{
    return _NS_msg_bool_isEqualToString__NS__Stringp((const void*)this, nullptr, aString);
}

_NS_INLINE NS::Range NS::String::rangeOfString(NS::String* searchString, NS::StringCompareOptions mask)
{
    return _NS_msg_NS__Range_rangeOfString_options__NS__Stringp_NS__StringCompareOptions((const void*)this, nullptr, searchString, mask);
}

_NS_INLINE NS::String* NS::String::stringByAppendingString(NS::String* aString)
{
    return _NS_msg_NS__Stringp_stringByAppendingString__NS__Stringp((const void*)this, nullptr, aString);
}

_NS_INLINE const char * NS::String::cString(NS::StringEncoding encoding)
{
    return _NS_msg_constcharp_cStringUsingEncoding__NS__StringEncoding((const void*)this, nullptr, encoding);
}

_NS_INLINE NS::UInteger NS::String::maximumLengthOfBytes(NS::StringEncoding enc)
{
    return _NS_msg_NS__UInteger_maximumLengthOfBytesUsingEncoding__NS__StringEncoding((const void*)this, nullptr, enc);
}

_NS_INLINE NS::UInteger NS::String::lengthOfBytes(NS::StringEncoding enc)
{
    return _NS_msg_NS__UInteger_lengthOfBytesUsingEncoding__NS__StringEncoding((const void*)this, nullptr, enc);
}

_NS_INLINE NS::String* NS::String::init(NS::String* aString)
{
    return _NS_msg_NS__Stringp_initWithString__NS__Stringp((const void*)this, nullptr, aString);
}

_NS_INLINE NS::String* NS::String::init(const void * bytes, NS::UInteger len, NS::StringEncoding encoding)
{
    return _NS_msg_NS__Stringp_initWithBytes_length_encoding__constvoidp_NS__UInteger_NS__StringEncoding((const void*)this, nullptr, bytes, len, encoding);
}

_NS_INLINE NS::String* NS::String::init(void * bytes, NS::UInteger len, NS::StringEncoding encoding, bool freeBuffer)
{
    return _NS_msg_NS__Stringp_initWithBytesNoCopy_length_encoding_freeWhenDone__voidp_NS__UInteger_NS__StringEncoding_bool((const void*)this, nullptr, bytes, len, encoding, freeBuffer);
}

_NS_INLINE NS::String* NS::String::init(const char * nullTerminatedCString, NS::StringEncoding encoding)
{
    return _NS_msg_NS__Stringp_initWithCString_encoding__constcharp_NS__StringEncoding((const void*)this, nullptr, nullTerminatedCString, encoding);
}
