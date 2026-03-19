//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSError.hpp
//
// Copyright 2020-2024 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSObject.hpp"
#include "NSPrivate.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
using ErrorDomain = class String*;

_NS_CONST(ErrorDomain, CocoaErrorDomain);
_NS_CONST(ErrorDomain, POSIXErrorDomain);
_NS_CONST(ErrorDomain, OSStatusErrorDomain);
_NS_CONST(ErrorDomain, MachErrorDomain);

using ErrorUserInfoKey = class String*;

_NS_CONST(ErrorUserInfoKey, UnderlyingErrorKey);
_NS_CONST(ErrorUserInfoKey, LocalizedDescriptionKey);
_NS_CONST(ErrorUserInfoKey, LocalizedFailureReasonErrorKey);
_NS_CONST(ErrorUserInfoKey, LocalizedRecoverySuggestionErrorKey);
_NS_CONST(ErrorUserInfoKey, LocalizedRecoveryOptionsErrorKey);
_NS_CONST(ErrorUserInfoKey, RecoveryAttempterErrorKey);
_NS_CONST(ErrorUserInfoKey, HelpAnchorErrorKey);
_NS_CONST(ErrorUserInfoKey, DebugDescriptionErrorKey);
_NS_CONST(ErrorUserInfoKey, LocalizedFailureErrorKey);
_NS_CONST(ErrorUserInfoKey, StringEncodingErrorKey);
_NS_CONST(ErrorUserInfoKey, URLErrorKey);
_NS_CONST(ErrorUserInfoKey, FilePathErrorKey);

class Error : public Copying<Error>
{
public:
    static Error*     error(ErrorDomain domain, Integer code, class Dictionary* pDictionary);

    static Error*     alloc();
    Error*            init();
    Error*            init(ErrorDomain domain, Integer code, class Dictionary* pDictionary);

    Integer           code() const;
    ErrorDomain       domain() const;
    class Dictionary* userInfo() const;

    class String*     localizedDescription() const;
    class Array*      localizedRecoveryOptions() const;
    class String*     localizedRecoverySuggestion() const;
    class String*     localizedFailureReason() const;
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_PRIVATE_DEF_CONST(NS::ErrorDomain, CocoaErrorDomain);
_NS_PRIVATE_DEF_CONST(NS::ErrorDomain, POSIXErrorDomain);
_NS_PRIVATE_DEF_CONST(NS::ErrorDomain, OSStatusErrorDomain);
_NS_PRIVATE_DEF_CONST(NS::ErrorDomain, MachErrorDomain);

_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, UnderlyingErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, LocalizedDescriptionKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, LocalizedFailureReasonErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, LocalizedRecoverySuggestionErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, LocalizedRecoveryOptionsErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, RecoveryAttempterErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, HelpAnchorErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, DebugDescriptionErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, LocalizedFailureErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, StringEncodingErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, URLErrorKey);
_NS_PRIVATE_DEF_CONST(NS::ErrorUserInfoKey, FilePathErrorKey);

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Error* NS::Error::error(ErrorDomain domain, Integer code, class Dictionary* pDictionary)
{
    return Object::sendMessage<Error*>(_NS_PRIVATE_CLS(NSError), _NS_PRIVATE_SEL(errorWithDomain_code_userInfo_), domain, code, pDictionary);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Error* NS::Error::alloc()
{
    return Object::alloc<Error>(_NS_PRIVATE_CLS(NSError));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Error* NS::Error::init()
{
    return Object::init<Error>();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Error* NS::Error::init(ErrorDomain domain, Integer code, class Dictionary* pDictionary)
{
    return Object::sendMessage<Error*>(this, _NS_PRIVATE_SEL(initWithDomain_code_userInfo_), domain, code, pDictionary);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Integer NS::Error::code() const
{
    return Object::sendMessage<Integer>(this, _NS_PRIVATE_SEL(code));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::ErrorDomain NS::Error::domain() const
{
    return Object::sendMessage<ErrorDomain>(this, _NS_PRIVATE_SEL(domain));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Dictionary* NS::Error::userInfo() const
{
    return Object::sendMessage<Dictionary*>(this, _NS_PRIVATE_SEL(userInfo));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Error::localizedDescription() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(localizedDescription));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Array* NS::Error::localizedRecoveryOptions() const
{
    return Object::sendMessage<Array*>(this, _NS_PRIVATE_SEL(localizedRecoveryOptions));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Error::localizedRecoverySuggestion() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(localizedRecoverySuggestion));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Error::localizedFailureReason() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(localizedFailureReason));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
