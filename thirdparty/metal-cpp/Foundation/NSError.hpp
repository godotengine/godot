#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Array;
    class Dictionary;
    class String;
}

namespace NS
{

using ErrorDomain = NS::String*;
using ErrorUserInfoKey = NS::String*;
extern ErrorDomain const CocoaErrorDomain __asm__("_NSCocoaErrorDomain");
extern ErrorDomain const POSIXErrorDomain __asm__("_NSPOSIXErrorDomain");
extern ErrorDomain const OSStatusErrorDomain __asm__("_NSOSStatusErrorDomain");
extern ErrorDomain const MachErrorDomain __asm__("_NSMachErrorDomain");
extern ErrorUserInfoKey const UnderlyingErrorKey __asm__("_NSUnderlyingErrorKey");
extern ErrorUserInfoKey const MultipleUnderlyingErrorsKey __asm__("_NSMultipleUnderlyingErrorsKey");
extern ErrorUserInfoKey const LocalizedDescriptionKey __asm__("_NSLocalizedDescriptionKey");
extern ErrorUserInfoKey const LocalizedFailureReasonErrorKey __asm__("_NSLocalizedFailureReasonErrorKey");
extern ErrorUserInfoKey const LocalizedRecoverySuggestionErrorKey __asm__("_NSLocalizedRecoverySuggestionErrorKey");
extern ErrorUserInfoKey const LocalizedRecoveryOptionsErrorKey __asm__("_NSLocalizedRecoveryOptionsErrorKey");
extern ErrorUserInfoKey const RecoveryAttempterErrorKey __asm__("_NSRecoveryAttempterErrorKey");
extern ErrorUserInfoKey const HelpAnchorErrorKey __asm__("_NSHelpAnchorErrorKey");
extern ErrorUserInfoKey const DebugDescriptionErrorKey __asm__("_NSDebugDescriptionErrorKey");
extern ErrorUserInfoKey const LocalizedFailureErrorKey __asm__("_NSLocalizedFailureErrorKey");
extern ErrorUserInfoKey const StringEncodingErrorKey __asm__("_NSStringEncodingErrorKey");
extern ErrorUserInfoKey const URLErrorKey __asm__("_NSURLErrorKey");
extern ErrorUserInfoKey const FilePathErrorKey __asm__("_NSFilePathErrorKey");

class Error : public NS::SecureCoding<Error>
{
public:
    static Error* alloc();
    Error*        init() const;

    static NS::Error* error(NS::ErrorDomain domain, NS::Integer code, NS::Dictionary* dict);

    NS::Integer     code() const;
    NS::ErrorDomain domain() const;
    NS::Error*      init(NS::ErrorDomain domain, NS::Integer code, NS::Dictionary* dict);
    NS::String*     localizedDescription() const;
    NS::String*     localizedFailureReason() const;
    NS::Array*      localizedRecoveryOptions() const;
    NS::String*     localizedRecoverySuggestion() const;
    NS::Dictionary* userInfo() const;

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSError;

_NS_INLINE NS::Error* NS::Error::alloc()
{
    return _NS_msg_NS__Errorp_alloc((const void*)&OBJC_CLASS_$_NSError, nullptr);
}

_NS_INLINE NS::Error* NS::Error::init() const
{
    return _NS_msg_NS__Errorp_init((const void*)this, nullptr);
}

_NS_INLINE NS::Error* NS::Error::error(NS::ErrorDomain domain, NS::Integer code, NS::Dictionary* dict)
{
    return _NS_msg_NS__Errorp_errorWithDomain_code_userInfo__NS__Stringp_NS__Integer_NS__Dictionaryp((const void*)&OBJC_CLASS_$_NSError, nullptr, domain, code, dict);
}

_NS_INLINE NS::ErrorDomain NS::Error::domain() const
{
    return _NS_msg_NS__Stringp_domain((const void*)this, nullptr);
}

_NS_INLINE NS::Integer NS::Error::code() const
{
    return _NS_msg_NS__Integer_code((const void*)this, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Error::userInfo() const
{
    return _NS_msg_NS__Dictionaryp_userInfo((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Error::localizedDescription() const
{
    return _NS_msg_NS__Stringp_localizedDescription((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Error::localizedFailureReason() const
{
    return _NS_msg_NS__Stringp_localizedFailureReason((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Error::localizedRecoverySuggestion() const
{
    return _NS_msg_NS__Stringp_localizedRecoverySuggestion((const void*)this, nullptr);
}

_NS_INLINE NS::Array* NS::Error::localizedRecoveryOptions() const
{
    return _NS_msg_NS__Arrayp_localizedRecoveryOptions((const void*)this, nullptr);
}

_NS_INLINE NS::Error* NS::Error::init(NS::ErrorDomain domain, NS::Integer code, NS::Dictionary* dict)
{
    return _NS_msg_NS__Errorp_initWithDomain_code_userInfo__NS__Stringp_NS__Integer_NS__Dictionaryp((const void*)this, nullptr, domain, code, dict);
}
