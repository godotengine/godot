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

_NS_OPTIONS(NS::UInteger, DataReadingOptions) {
    DataReadingMappedIfSafe = 1UL << 0,
    DataReadingUncached = 1UL << 1,
    DataReadingMappedAlways = 1UL << 3,
    DataReadingMapped = DataReadingMappedIfSafe,
    MappedRead = DataReadingMapped,
    UncachedRead = DataReadingUncached,
};

_NS_OPTIONS(NS::UInteger, DataWritingOptions) {
    DataWritingAtomic = 1UL << 0,
    DataWritingWithoutOverwriting = 1UL << 1,
    DataWritingFileProtectionNone = 0x10000000,
    DataWritingFileProtectionComplete = 0x20000000,
    DataWritingFileProtectionCompleteUnlessOpen = 0x30000000,
    DataWritingFileProtectionCompleteUntilFirstUserAuthentication = 0x40000000,
    DataWritingFileProtectionCompleteWhenUserInactive = 0x50000000,
    DataWritingFileProtectionMask = 0xf0000000,
    AtomicWrite = DataWritingAtomic,
};

_NS_OPTIONS(NS::UInteger, DataSearchOptions) {
    DataSearchBackwards = 1UL << 0,
    DataSearchAnchored = 1UL << 1,
};

_NS_OPTIONS(NS::UInteger, DataBase64EncodingOptions) {
    DataBase64Encoding64CharacterLineLength = 1UL << 0,
    DataBase64Encoding76CharacterLineLength = 1UL << 1,
    DataBase64EncodingEndLineWithCarriageReturn = 1UL << 4,
    DataBase64EncodingEndLineWithLineFeed = 1UL << 5,
};

_NS_OPTIONS(NS::UInteger, DataBase64DecodingOptions) {
    DataBase64DecodingIgnoreUnknownCharacters = 1UL << 0,
};

_NS_ENUM(NS::Integer, DataCompressionAlgorithm) {
    DataCompressionAlgorithmLZFSE = 0,
    DataCompressionAlgorithmLZ4 = 1,
    DataCompressionAlgorithmLZMA = 2,
    DataCompressionAlgorithmZlib = 3,
};


class Data : public NS::SecureCoding<Data>
{
public:
    static Data* alloc();
    Data*        init() const;

    const void * bytes() const;
    NS::UInteger length() const;

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSData;

_NS_INLINE NS::Data* NS::Data::alloc()
{
    return _NS_msg_NS__Datap_alloc((const void*)&OBJC_CLASS_$_NSData, nullptr);
}

_NS_INLINE NS::Data* NS::Data::init() const
{
    return _NS_msg_NS__Datap_init((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::Data::length() const
{
    return _NS_msg_NS__UInteger_length((const void*)this, nullptr);
}

_NS_INLINE const void * NS::Data::bytes() const
{
    return _NS_msg_constvoidp_bytes((const void*)this, nullptr);
}
