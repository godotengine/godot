//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSPrivate.hpp
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

#include <objc/runtime.h>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#define _NS_PRIVATE_CLS(symbol) (Private::Class::s_k##symbol)
#define _NS_PRIVATE_SEL(accessor) (Private::Selector::s_k##accessor)

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#if defined(NS_PRIVATE_IMPLEMENTATION)

#include <dlfcn.h>

namespace NS::Private
{
    template <typename _Type>
    inline _Type const LoadSymbol(const char* pSymbol)
    {
        const _Type* pAddress = static_cast<_Type*>(dlsym(RTLD_DEFAULT, pSymbol));

        return pAddress ? *pAddress : _Type();
    }
} // NS::Private

#ifdef METALCPP_SYMBOL_VISIBILITY_HIDDEN
#define _NS_PRIVATE_VISIBILITY __attribute__((visibility("hidden")))
#else
#define _NS_PRIVATE_VISIBILITY __attribute__((visibility("default")))
#endif // METALCPP_SYMBOL_VISIBILITY_HIDDEN

#define _NS_PRIVATE_IMPORT __attribute__((weak_import))

#ifdef __OBJC__
#define _NS_PRIVATE_OBJC_LOOKUP_CLASS(symbol) ((__bridge void*)objc_lookUpClass(#symbol))
#define _NS_PRIVATE_OBJC_GET_PROTOCOL(symbol) ((__bridge void*)objc_getProtocol(#symbol))
#else
#define _NS_PRIVATE_OBJC_LOOKUP_CLASS(symbol) objc_lookUpClass(#symbol)
#define _NS_PRIVATE_OBJC_GET_PROTOCOL(symbol) objc_getProtocol(#symbol)
#endif // __OBJC__

#define _NS_PRIVATE_DEF_CLS(symbol) void* s_k##symbol _NS_PRIVATE_VISIBILITY = _NS_PRIVATE_OBJC_LOOKUP_CLASS(symbol)
#define _NS_PRIVATE_DEF_PRO(symbol) void* s_k##symbol _NS_PRIVATE_VISIBILITY = _NS_PRIVATE_OBJC_GET_PROTOCOL(symbol)
#define _NS_PRIVATE_DEF_SEL(accessor, symbol) SEL s_k##accessor _NS_PRIVATE_VISIBILITY = sel_registerName(symbol)

#if defined(__MAC_26_0) || defined(__IPHONE_26_0) || defined(__TVOS_26_0)
#define _NS_PRIVATE_DEF_CONST(type, symbol)              \
    _NS_EXTERN type const NS##symbol _NS_PRIVATE_IMPORT; \
    type const                       NS::symbol = (nullptr != &NS##symbol) ? NS##symbol : type()
#else
#define _NS_PRIVATE_DEF_CONST(type, symbol) \
    _NS_EXTERN type const MTL##symbol _NS_PRIVATE_IMPORT; \
    type const             NS::symbol = Private::LoadSymbol<type>("NS" #symbol)
#endif

#else

#define _NS_PRIVATE_DEF_CLS(symbol) extern void* s_k##symbol
#define _NS_PRIVATE_DEF_PRO(symbol) extern void* s_k##symbol
#define _NS_PRIVATE_DEF_SEL(accessor, symbol) extern SEL s_k##accessor
#define _NS_PRIVATE_DEF_CONST(type, symbol) extern type const NS::symbol

#endif // NS_PRIVATE_IMPLEMENTATION

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
namespace Private
{
    namespace Class
    {

        _NS_PRIVATE_DEF_CLS(NSArray);
        _NS_PRIVATE_DEF_CLS(NSAutoreleasePool);
        _NS_PRIVATE_DEF_CLS(NSBundle);
        _NS_PRIVATE_DEF_CLS(NSCondition);
        _NS_PRIVATE_DEF_CLS(NSDate);
        _NS_PRIVATE_DEF_CLS(NSDictionary);
        _NS_PRIVATE_DEF_CLS(NSError);
        _NS_PRIVATE_DEF_CLS(NSNotificationCenter);
        _NS_PRIVATE_DEF_CLS(NSNumber);
        _NS_PRIVATE_DEF_CLS(NSObject);
        _NS_PRIVATE_DEF_CLS(NSProcessInfo);
        _NS_PRIVATE_DEF_CLS(NSSet);
        _NS_PRIVATE_DEF_CLS(NSString);
        _NS_PRIVATE_DEF_CLS(NSURL);
        _NS_PRIVATE_DEF_CLS(NSValue);

    } // Class
} // Private
} // MTL

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
namespace Private
{
    namespace Protocol
    {

    } // Protocol
} // Private
} // NS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
namespace Private
{
    namespace Selector
    {

        _NS_PRIVATE_DEF_SEL(addObject_,
            "addObject:");
        _NS_PRIVATE_DEF_SEL(addObserverName_object_queue_block_,
            "addObserverForName:object:queue:usingBlock:");
        _NS_PRIVATE_DEF_SEL(activeProcessorCount,
            "activeProcessorCount");
        _NS_PRIVATE_DEF_SEL(allBundles,
            "allBundles");
        _NS_PRIVATE_DEF_SEL(allFrameworks,
            "allFrameworks");
        _NS_PRIVATE_DEF_SEL(allObjects,
            "allObjects");
        _NS_PRIVATE_DEF_SEL(alloc,
            "alloc");
        _NS_PRIVATE_DEF_SEL(appStoreReceiptURL,
            "appStoreReceiptURL");
        _NS_PRIVATE_DEF_SEL(arguments,
            "arguments");
        _NS_PRIVATE_DEF_SEL(array,
            "array");
        _NS_PRIVATE_DEF_SEL(arrayWithObject_,
            "arrayWithObject:");
        _NS_PRIVATE_DEF_SEL(arrayWithObjects_count_,
            "arrayWithObjects:count:");
        _NS_PRIVATE_DEF_SEL(automaticTerminationSupportEnabled,
            "automaticTerminationSupportEnabled");
        _NS_PRIVATE_DEF_SEL(autorelease,
            "autorelease");
        _NS_PRIVATE_DEF_SEL(beginActivityWithOptions_reason_,
            "beginActivityWithOptions:reason:");
        _NS_PRIVATE_DEF_SEL(boolValue,
            "boolValue");
        _NS_PRIVATE_DEF_SEL(broadcast,
            "broadcast");
        _NS_PRIVATE_DEF_SEL(builtInPlugInsPath,
            "builtInPlugInsPath");
        _NS_PRIVATE_DEF_SEL(builtInPlugInsURL,
            "builtInPlugInsURL");
        _NS_PRIVATE_DEF_SEL(bundleIdentifier,
            "bundleIdentifier");
        _NS_PRIVATE_DEF_SEL(bundlePath,
            "bundlePath");
        _NS_PRIVATE_DEF_SEL(bundleURL,
            "bundleURL");
        _NS_PRIVATE_DEF_SEL(bundleWithPath_,
            "bundleWithPath:");
        _NS_PRIVATE_DEF_SEL(bundleWithURL_,
            "bundleWithURL:");
        _NS_PRIVATE_DEF_SEL(caseInsensitiveCompare_,
            "caseInsensitiveCompare:");
        _NS_PRIVATE_DEF_SEL(characterAtIndex_,
            "characterAtIndex:");
        _NS_PRIVATE_DEF_SEL(charValue,
            "charValue");
        _NS_PRIVATE_DEF_SEL(countByEnumeratingWithState_objects_count_,
            "countByEnumeratingWithState:objects:count:");
        _NS_PRIVATE_DEF_SEL(cStringUsingEncoding_,
            "cStringUsingEncoding:");
        _NS_PRIVATE_DEF_SEL(code,
            "code");
        _NS_PRIVATE_DEF_SEL(compare_,
            "compare:");
        _NS_PRIVATE_DEF_SEL(copy,
            "copy");
        _NS_PRIVATE_DEF_SEL(count,
            "count");
        _NS_PRIVATE_DEF_SEL(dateWithTimeIntervalSinceNow_,
            "dateWithTimeIntervalSinceNow:");
        _NS_PRIVATE_DEF_SEL(defaultCenter,
            "defaultCenter");
        _NS_PRIVATE_DEF_SEL(descriptionWithLocale_,
            "descriptionWithLocale:");
        _NS_PRIVATE_DEF_SEL(disableAutomaticTermination_,
            "disableAutomaticTermination:");
        _NS_PRIVATE_DEF_SEL(disableSuddenTermination,
            "disableSuddenTermination");
        _NS_PRIVATE_DEF_SEL(debugDescription,
            "debugDescription");
        _NS_PRIVATE_DEF_SEL(description,
            "description");
        _NS_PRIVATE_DEF_SEL(dictionary,
            "dictionary");
        _NS_PRIVATE_DEF_SEL(dictionaryWithObject_forKey_,
            "dictionaryWithObject:forKey:");
        _NS_PRIVATE_DEF_SEL(dictionaryWithObjects_forKeys_count_,
            "dictionaryWithObjects:forKeys:count:");
        _NS_PRIVATE_DEF_SEL(domain,
            "domain");
        _NS_PRIVATE_DEF_SEL(doubleValue,
            "doubleValue");
        _NS_PRIVATE_DEF_SEL(drain,
            "drain");
        _NS_PRIVATE_DEF_SEL(enableAutomaticTermination_,
            "enableAutomaticTermination:");
        _NS_PRIVATE_DEF_SEL(enableSuddenTermination,
            "enableSuddenTermination");
        _NS_PRIVATE_DEF_SEL(endActivity_,
            "endActivity:");
        _NS_PRIVATE_DEF_SEL(environment,
            "environment");
        _NS_PRIVATE_DEF_SEL(errorWithDomain_code_userInfo_,
            "errorWithDomain:code:userInfo:");
        _NS_PRIVATE_DEF_SEL(executablePath,
            "executablePath");
        _NS_PRIVATE_DEF_SEL(executableURL,
            "executableURL");
        _NS_PRIVATE_DEF_SEL(fileSystemRepresentation,
            "fileSystemRepresentation");
        _NS_PRIVATE_DEF_SEL(fileURLWithPath_,
            "fileURLWithPath:");
        _NS_PRIVATE_DEF_SEL(floatValue,
            "floatValue");
        _NS_PRIVATE_DEF_SEL(fullUserName,
            "fullUserName");
        _NS_PRIVATE_DEF_SEL(getValue_size_,
            "getValue:size:");
        _NS_PRIVATE_DEF_SEL(globallyUniqueString,
            "globallyUniqueString");
        _NS_PRIVATE_DEF_SEL(hash,
            "hash");
        _NS_PRIVATE_DEF_SEL(hasPerformanceProfile_,
            "hasPerformanceProfile:");
        _NS_PRIVATE_DEF_SEL(hostName,
            "hostName");
        _NS_PRIVATE_DEF_SEL(infoDictionary,
            "infoDictionary");
        _NS_PRIVATE_DEF_SEL(init,
            "init");
        _NS_PRIVATE_DEF_SEL(initFileURLWithPath_,
            "initFileURLWithPath:");
        _NS_PRIVATE_DEF_SEL(initWithBool_,
            "initWithBool:");
        _NS_PRIVATE_DEF_SEL(initWithBytes_objCType_,
            "initWithBytes:objCType:");
        _NS_PRIVATE_DEF_SEL(initWithBytesNoCopy_length_encoding_freeWhenDone_,
            "initWithBytesNoCopy:length:encoding:freeWhenDone:");
        _NS_PRIVATE_DEF_SEL(initWithChar_,
            "initWithChar:");
        _NS_PRIVATE_DEF_SEL(initWithCoder_,
            "initWithCoder:");
        _NS_PRIVATE_DEF_SEL(initWithCString_encoding_,
            "initWithCString:encoding:");
        _NS_PRIVATE_DEF_SEL(initWithDomain_code_userInfo_,
            "initWithDomain:code:userInfo:");
        _NS_PRIVATE_DEF_SEL(initWithDouble_,
            "initWithDouble:");
        _NS_PRIVATE_DEF_SEL(initWithFloat_,
            "initWithFloat:");
        _NS_PRIVATE_DEF_SEL(initWithInt_,
            "initWithInt:");
        _NS_PRIVATE_DEF_SEL(initWithLong_,
            "initWithLong:");
        _NS_PRIVATE_DEF_SEL(initWithLongLong_,
            "initWithLongLong:");
        _NS_PRIVATE_DEF_SEL(initWithObjects_count_,
            "initWithObjects:count:");
        _NS_PRIVATE_DEF_SEL(initWithObjects_forKeys_count_,
            "initWithObjects:forKeys:count:");
        _NS_PRIVATE_DEF_SEL(initWithPath_,
            "initWithPath:");
        _NS_PRIVATE_DEF_SEL(initWithShort_,
            "initWithShort:");
        _NS_PRIVATE_DEF_SEL(initWithString_,
            "initWithString:");
        _NS_PRIVATE_DEF_SEL(initWithUnsignedChar_,
            "initWithUnsignedChar:");
        _NS_PRIVATE_DEF_SEL(initWithUnsignedInt_,
            "initWithUnsignedInt:");
        _NS_PRIVATE_DEF_SEL(initWithUnsignedLong_,
            "initWithUnsignedLong:");
        _NS_PRIVATE_DEF_SEL(initWithUnsignedLongLong_,
            "initWithUnsignedLongLong:");
        _NS_PRIVATE_DEF_SEL(initWithUnsignedShort_,
            "initWithUnsignedShort:");
        _NS_PRIVATE_DEF_SEL(initWithURL_,
            "initWithURL:");
        _NS_PRIVATE_DEF_SEL(integerValue,
            "integerValue");
        _NS_PRIVATE_DEF_SEL(intValue,
            "intValue");
        _NS_PRIVATE_DEF_SEL(isDeviceCertified_,
            "isDeviceCertifiedFor:");
        _NS_PRIVATE_DEF_SEL(isEqual_,
            "isEqual:");
        _NS_PRIVATE_DEF_SEL(isEqualToNumber_,
            "isEqualToNumber:");
        _NS_PRIVATE_DEF_SEL(isEqualToString_,
            "isEqualToString:");
        _NS_PRIVATE_DEF_SEL(isEqualToValue_,
            "isEqualToValue:");
        _NS_PRIVATE_DEF_SEL(isiOSAppOnMac,
            "isiOSAppOnMac");
        _NS_PRIVATE_DEF_SEL(isLoaded,
            "isLoaded");
        _NS_PRIVATE_DEF_SEL(isLowPowerModeEnabled,
            "isLowPowerModeEnabled");
        _NS_PRIVATE_DEF_SEL(isMacCatalystApp,
            "isMacCatalystApp");
        _NS_PRIVATE_DEF_SEL(isOperatingSystemAtLeastVersion_,
            "isOperatingSystemAtLeastVersion:");
        _NS_PRIVATE_DEF_SEL(keyEnumerator,
            "keyEnumerator");
        _NS_PRIVATE_DEF_SEL(length,
            "length");
        _NS_PRIVATE_DEF_SEL(lengthOfBytesUsingEncoding_,
            "lengthOfBytesUsingEncoding:");
        _NS_PRIVATE_DEF_SEL(load,
            "load");
        _NS_PRIVATE_DEF_SEL(loadAndReturnError_,
            "loadAndReturnError:");
        _NS_PRIVATE_DEF_SEL(localizedDescription,
            "localizedDescription");
        _NS_PRIVATE_DEF_SEL(localizedFailureReason,
            "localizedFailureReason");
        _NS_PRIVATE_DEF_SEL(localizedInfoDictionary,
            "localizedInfoDictionary");
        _NS_PRIVATE_DEF_SEL(localizedRecoveryOptions,
            "localizedRecoveryOptions");
        _NS_PRIVATE_DEF_SEL(localizedRecoverySuggestion,
            "localizedRecoverySuggestion");
        _NS_PRIVATE_DEF_SEL(localizedStringForKey_value_table_,
            "localizedStringForKey:value:table:");
        _NS_PRIVATE_DEF_SEL(lock,
            "lock");
        _NS_PRIVATE_DEF_SEL(longValue,
            "longValue");
        _NS_PRIVATE_DEF_SEL(longLongValue,
            "longLongValue");
        _NS_PRIVATE_DEF_SEL(mainBundle,
            "mainBundle");
        _NS_PRIVATE_DEF_SEL(maximumLengthOfBytesUsingEncoding_,
            "maximumLengthOfBytesUsingEncoding:");
        _NS_PRIVATE_DEF_SEL(methodSignatureForSelector_,
            "methodSignatureForSelector:");
        _NS_PRIVATE_DEF_SEL(mutableBytes,
            "mutableBytes");
        _NS_PRIVATE_DEF_SEL(name,
            "name");
        _NS_PRIVATE_DEF_SEL(nextObject,
            "nextObject");
        _NS_PRIVATE_DEF_SEL(numberWithBool_,
            "numberWithBool:");
        _NS_PRIVATE_DEF_SEL(numberWithChar_,
            "numberWithChar:");
        _NS_PRIVATE_DEF_SEL(numberWithDouble_,
            "numberWithDouble:");
        _NS_PRIVATE_DEF_SEL(numberWithFloat_,
            "numberWithFloat:");
        _NS_PRIVATE_DEF_SEL(numberWithInt_,
            "numberWithInt:");
        _NS_PRIVATE_DEF_SEL(numberWithLong_,
            "numberWithLong:");
        _NS_PRIVATE_DEF_SEL(numberWithLongLong_,
            "numberWithLongLong:");
        _NS_PRIVATE_DEF_SEL(numberWithShort_,
            "numberWithShort:");
        _NS_PRIVATE_DEF_SEL(numberWithUnsignedChar_,
            "numberWithUnsignedChar:");
        _NS_PRIVATE_DEF_SEL(numberWithUnsignedInt_,
            "numberWithUnsignedInt:");
        _NS_PRIVATE_DEF_SEL(numberWithUnsignedLong_,
            "numberWithUnsignedLong:");
        _NS_PRIVATE_DEF_SEL(numberWithUnsignedLongLong_,
            "numberWithUnsignedLongLong:");
        _NS_PRIVATE_DEF_SEL(numberWithUnsignedShort_,
            "numberWithUnsignedShort:");
        _NS_PRIVATE_DEF_SEL(objCType,
            "objCType");
        _NS_PRIVATE_DEF_SEL(object,
            "object");
        _NS_PRIVATE_DEF_SEL(objectAtIndex_,
            "objectAtIndex:");
        _NS_PRIVATE_DEF_SEL(objectEnumerator,
            "objectEnumerator");
        _NS_PRIVATE_DEF_SEL(objectForInfoDictionaryKey_,
            "objectForInfoDictionaryKey:");
        _NS_PRIVATE_DEF_SEL(objectForKey_,
            "objectForKey:");
        _NS_PRIVATE_DEF_SEL(operatingSystem,
            "operatingSystem");
        _NS_PRIVATE_DEF_SEL(operatingSystemVersion,
            "operatingSystemVersion");
        _NS_PRIVATE_DEF_SEL(operatingSystemVersionString,
            "operatingSystemVersionString");
        _NS_PRIVATE_DEF_SEL(pathForAuxiliaryExecutable_,
            "pathForAuxiliaryExecutable:");
        _NS_PRIVATE_DEF_SEL(performActivityWithOptions_reason_usingBlock_,
            "performActivityWithOptions:reason:usingBlock:");
        _NS_PRIVATE_DEF_SEL(performExpiringActivityWithReason_usingBlock_,
            "performExpiringActivityWithReason:usingBlock:");
        _NS_PRIVATE_DEF_SEL(physicalMemory,
            "physicalMemory");
        _NS_PRIVATE_DEF_SEL(pointerValue,
            "pointerValue");
        _NS_PRIVATE_DEF_SEL(preflightAndReturnError_,
            "preflightAndReturnError:");
        _NS_PRIVATE_DEF_SEL(privateFrameworksPath,
            "privateFrameworksPath");
        _NS_PRIVATE_DEF_SEL(privateFrameworksURL,
            "privateFrameworksURL");
        _NS_PRIVATE_DEF_SEL(processIdentifier,
            "processIdentifier");
        _NS_PRIVATE_DEF_SEL(processInfo,
            "processInfo");
        _NS_PRIVATE_DEF_SEL(processName,
            "processName");
        _NS_PRIVATE_DEF_SEL(processorCount,
            "processorCount");
        _NS_PRIVATE_DEF_SEL(rangeOfString_options_,
            "rangeOfString:options:");
        _NS_PRIVATE_DEF_SEL(release,
            "release");
        _NS_PRIVATE_DEF_SEL(removeObserver_,
            "removeObserver:");
        _NS_PRIVATE_DEF_SEL(resourcePath,
            "resourcePath");
        _NS_PRIVATE_DEF_SEL(resourceURL,
            "resourceURL");
        _NS_PRIVATE_DEF_SEL(respondsToSelector_,
            "respondsToSelector:");
        _NS_PRIVATE_DEF_SEL(retain,
            "retain");
        _NS_PRIVATE_DEF_SEL(retainCount,
            "retainCount");
        _NS_PRIVATE_DEF_SEL(setAutomaticTerminationSupportEnabled_,
            "setAutomaticTerminationSupportEnabled:");
        _NS_PRIVATE_DEF_SEL(setProcessName_,
            "setProcessName:");
        _NS_PRIVATE_DEF_SEL(sharedFrameworksPath,
            "sharedFrameworksPath");
        _NS_PRIVATE_DEF_SEL(sharedFrameworksURL,
            "sharedFrameworksURL");
        _NS_PRIVATE_DEF_SEL(sharedSupportPath,
            "sharedSupportPath");
        _NS_PRIVATE_DEF_SEL(sharedSupportURL,
            "sharedSupportURL");
        _NS_PRIVATE_DEF_SEL(shortValue,
            "shortValue");
        _NS_PRIVATE_DEF_SEL(showPools,
            "showPools");
        _NS_PRIVATE_DEF_SEL(signal,
            "signal");
        _NS_PRIVATE_DEF_SEL(string,
            "string");
        _NS_PRIVATE_DEF_SEL(stringValue,
            "stringValue");
        _NS_PRIVATE_DEF_SEL(stringWithString_,
            "stringWithString:");
        _NS_PRIVATE_DEF_SEL(stringWithCString_encoding_,
            "stringWithCString:encoding:");
        _NS_PRIVATE_DEF_SEL(stringByAppendingString_,
            "stringByAppendingString:");
        _NS_PRIVATE_DEF_SEL(systemUptime,
            "systemUptime");
        _NS_PRIVATE_DEF_SEL(thermalState,
            "thermalState");
        _NS_PRIVATE_DEF_SEL(unload,
            "unload");
        _NS_PRIVATE_DEF_SEL(unlock,
            "unlock");
        _NS_PRIVATE_DEF_SEL(unsignedCharValue,
            "unsignedCharValue");
        _NS_PRIVATE_DEF_SEL(unsignedIntegerValue,
            "unsignedIntegerValue");
        _NS_PRIVATE_DEF_SEL(unsignedIntValue,
            "unsignedIntValue");
        _NS_PRIVATE_DEF_SEL(unsignedLongValue,
            "unsignedLongValue");
        _NS_PRIVATE_DEF_SEL(unsignedLongLongValue,
            "unsignedLongLongValue");
        _NS_PRIVATE_DEF_SEL(unsignedShortValue,
            "unsignedShortValue");
        _NS_PRIVATE_DEF_SEL(URLForAuxiliaryExecutable_,
            "URLForAuxiliaryExecutable:");
        _NS_PRIVATE_DEF_SEL(userInfo,
            "userInfo");
        _NS_PRIVATE_DEF_SEL(userName,
            "userName");
        _NS_PRIVATE_DEF_SEL(UTF8String,
            "UTF8String");
        _NS_PRIVATE_DEF_SEL(valueWithBytes_objCType_,
            "valueWithBytes:objCType:");
        _NS_PRIVATE_DEF_SEL(valueWithPointer_,
            "valueWithPointer:");
        _NS_PRIVATE_DEF_SEL(wait,
            "wait");
        _NS_PRIVATE_DEF_SEL(waitUntilDate_,
            "waitUntilDate:");
    } // Class
} // Private
} // MTL

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
