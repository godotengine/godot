## About

**metal-cpp** is a low overhead and header only C++ interface for Metal that helps developers add Metal functionality to graphics applications that are written in C++ (such as game engines). **metal-cpp** removes the need to create a shim and allows developers to call Metal functions directly from anywhere in their existing C++ code.


## Highlights

- Drop in C++ alternative interface to the Metal Objective-C headers.
- Direct mapping of all Metal Objective-C classes, constants and enums to C++ in the MTL C++ namespace.
- No measurable overhead compared to calling Metal Objective-C headers, due to inlining of C++ function calls.
- No usage of wrapper containers that require additional allocations.
- Requires C++17 due to the usage of `constexpr` in `NS::Object`.
- Identical header files and function/constant/enum availability for iOS, macOS and tvOS.
- Backwards compatibility: All `bool MTL::Device::supports...()` functions check if their required selectors exist and automatically return `false` if not.
- String (`ErrorDomain`) constants are weak linked and automatically set to `nullptr` if not available.

## Changelog

| Version | Changes |
|-|-|
| macOS 26, iOS 26 | Add all the Metal APIs in macOS 26, iOS 26, including support for the **Apple10** GPU family. <br/>Add support for Metal 4 and new denoiser and temporal scalers in MetalFX.|
| macOS 15, iOS 18 | Add all the Metal APIs in macOS 15 and iOS 18. |
| macOS 14, iOS 17 | Add support for the **MetalFX** framework. <br/>Add all the APIs in macOS 14 and iOS 17. |
| macOS 13.3, iOS 16.4 | Add all the APIs in macOS 13.3 and iOS 16.4. |
| macOS 13, iOS 16| Add all the APIs in macOS 13 and iOS 16.<br />New optional `NS::SharedPtr<T>` type to assist with memory management.<br/>New convenience function to create a `CA::MetalLayer`.<br/>New `MTLSTR(str)` macro allows faster string creation from literals.<br/>Fix a problem with the signature of functions that take an array of pointers as input.<br/>Fix a problem with the signature of the `setGroups()` function in `MTL::LinkedFunctions`.|
| macOS 12, iOS 15 | Initial release. |

## Memory Allocation Policy

**metal-cpp** follows the object allocation policies of Cocoa, Cocoa Touch, and CoreFoundation. Understanding these rules is especially important when using metal-cpp, as C++ objects are not eligible for automatic reference counting (ARC).

**metal-cpp** objects are reference counted. To help convey and manage object lifecycles, the following conventions are observed:

1. *You own any object returned by methods whose name begins with* `alloc` *,* `new` *,* `copy` *,* `mutableCopy` *, or* `Create`. The method returns these objects with `retainCount` equals to `1`.
2. *You can take ownership of an object by calling its* ```retain()``` *method*. A received object is normally guaranteed to remain valid within the method it was received in. You use `retain` in two situations: (1) In the implementation of an accessor method (a setter) or to take ownership of an object; and (2) To prevent an object from being deallocated as a side-effect of some other operation.
3. *When you no longer need it, you must relinquish ownership of an object you own*. You relinquish ownership by calling its `release()` or `autorelease()` method.
4. *You must not relinquish ownership of an object you do not own*.

When an object's `retainCount` reaches `0`, the object is immediately deallocated. It is illegal to call methods on a deallocated object and it may lead to an application crash.

### AutoreleasePools and Objects

Several methods that create temporary objects in **metal-cpp** add them to an `AutoreleasePool` to help manage their lifetimes. In these situations, after **metal-cpp** creates the object, it adds it to an `AutoreleasePool`, which will release its objects when you release (or drain) it.

By adding temporary objects to an AutoreleasePool, you do not need to explicitly call `release()` to deallocate them. Instead, you can rely on the `AutoreleasePool` to implicitly manage those lifetimes.

If you create an object with a method that does not begin with `alloc`, `new`, `copy`, `mutableCopy`, or `Create`, the creating method adds the object to an autorelease pool.

The typical scope of an `AutoreleasePool` is one frame of rendering for the main thread of the program. When the thread returns control to the RunLoop (an object responsible for receiving input and events from the windowing system), the pool is *drained*, releasing its objects.

You can create and manage additional `AutoreleasePool`s at smaller scopes to reduce your program's working set, and you are required to do so for any additional threads your program creates.

If an object's lifecycle needs to be extended beyond the scope of an `AutoreleasePool` instance, you can claim ownership of it by calling its `retain()` method before the pool is drained. In these cases, you are responsible for making the appropriate `release()` call on the object after you no longer need it.

You can find a more-detailed introduction to the memory management rules here: https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmRules.html, and here: https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFMemoryMgmt/Concepts/Ownership.html

For more details about the application's RunLoop, please find its documentation here: https://developer.apple.com/documentation/foundation/nsrunloop

### Use and debug AutoreleasePools

When you create an autoreleased object and there is no enclosing `AutoreleasePool`, the object is leaked.

To prevent this, you normally create an `AutoreleasePool` in your program's `main` function, and in the entry function for every thread you create. You may also create additional `AutoreleasePool`s to avoid growing your program's high memory watermark when you create several autoreleased objects, such as when rendering.

Use the Environment Variable `OBJC_DEBUG_MISSING_POOLS=YES` to print a runtime warning when an autoreleased object is leaked because no enclosing `AutoreleasePool` is available for its thread.

You can also run `leaks --autoreleasePools` on a memgraph file or a process ID (macOS only) to view a listing of your program's `AutoreleasePool`s and all objects they contain.

### NS::SharedPtr

The **metal-cpp** headers include an optional `NS::SharedPtr<>` (shared pointer) template that can help you manually manage memory in your apps.

Shared pointers in **metal-cpp** are different from `std::shared_ptr<>` in that they implement specific optimizations for its memory model. For example, **metal-cpp**'s shared pointers avoid the overhead of the standard library's version by leveraging the reference counting implementation of the `NS::Object` type.

#### Note

The **metal-cpp** shared pointer’s destructor method always calls the `release()` method of the pointer that it wraps.

You can create an `NS::SharedPtr<>` by calling the metal-cpp's factory method that's appropriate for your application's intent:

* You can **transfer** ownership of a pointer to a new shared pointer instance by calling the  `NS::TransferPtr()` factory function, which is the correct function for Resource Acquisition is Initialization (RAII) implementations because it doesn't increase the pointee's retain count.

* You can **share** ownership of a pointer with another entity by calling the `NS::RetainPtr()` factory function. This function can also extend an object's lifecycle beyond an `AutoreleasePool` instance's scope because it creates a strong reference to the pointee and increases its retain count.

Usage of `NS::SharedPtr<>` is optional.

### nullptr

Similar to Objective-C, it is legal to call any method, including `retain()` and `release()`, on `nullptr` "objects". While calling methods on `nullptr` still does incur in function call overhead, the effective result is equivalent of a NOP.

Conversely, do not assume that because calling a method on a pointer did not result in a crash, that the pointed-to object is valid.

## Adding metal-cpp to a Project

Simply include `Metal/Metal.hpp`. To ensure that the selector and class symbols are linked, add to one of your cpp files:

```cpp
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
```

If you want to use the QuartzCore wrapper, add:

```cpp
#define CA_PRIVATE_IMPLEMENTATION

#include "QuartzCore/QuartzCore.hpp"
```

## Generating a Single Header File

Purely optional: You can generate a single header file that contains all **metal-cpp** headers via:

```shell
./SingleHeader/MakeSingleHeader.py Foundation/Foundation.hpp QuartzCore/QuartzCore.hpp Metal/Metal.hpp MetalFX/MetalFX.hpp
```

By default the generator script writes its output to `./SingleHeader/Metal.hpp`. Use the `-o` option to customize output filename.

## Global Symbol Visibility

metal-cpp marks all its symbols with `default` visibility. Define the macro: `METALCPP_SYMBOL_VISIBILITY_HIDDEN` to override this behavior and hide its symbols.

## Examples

#### Creating the device

###### Objective-C (with automatic reference counting)

```objc
id< MTLDevice > device = MTLCreateSystemDefaultDevice();

// ...
```

###### Objective-C

```objc
id< MTLDevice > device = MTLCreateSystemDefaultDevice();

// ...

[device release];
```

###### C++

```cpp
MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

// ...

pDevice->release();
```

###### C++ (using NS::SharedPtr)

```cpp
NS::SharedPtr< MTL::Device > pDevice = NS::TransferPtr( MTL::CreateSystemDefaultDevice() );

// ...
```

#### Metal function calls map directly to C++

###### Objective-C (with automatic reference counting)

```objc
MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];

[samplerDescriptor setSAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setTAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setRAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setMagFilter: MTLSamplerMinMagFilterLinear];
[samplerDescriptor setMinFilter: MTLSamplerMinMagFilterLinear];
[samplerDescriptor setMipFilter: MTLSamplerMipFilterLinear];
[samplerDescriptor setSupportArgumentBuffers: YES];

id< MTLSamplerState > samplerState = [device newSamplerStateWithDescriptor:samplerDescriptor];
```

###### Objective-C

```objc
MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];

[samplerDescriptor setSAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setTAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setRAddressMode: MTLSamplerAddressModeRepeat];
[samplerDescriptor setMagFilter: MTLSamplerMinMagFilterLinear];
[samplerDescriptor setMinFilter: MTLSamplerMinMagFilterLinear];
[samplerDescriptor setMipFilter: MTLSamplerMipFilterLinear];
[samplerDescriptor setSupportArgumentBuffers: YES];

id< MTLSamplerState > samplerState = [device newSamplerStateWithDescriptor:samplerDescriptor];

[samplerDescriptor release];

// ...

[samplerState release];
```

###### C++

```cpp
MTL::SamplerDescriptor* pSamplerDescriptor = MTL::SamplerDescriptor::alloc()->init();

pSamplerDescriptor->setSAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setTAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setRAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setMagFilter( MTL::SamplerMinMagFilterLinear );
pSamplerDescriptor->setMinFilter( MTL::SamplerMinMagFilterLinear );
pSamplerDescriptor->setMipFilter( MTL::SamplerMipFilterLinear );
pSamplerDescriptor->setSupportArgumentBuffers( true );

MTL::SamplerState* pSamplerState = pDevice->newSamplerState( pSamplerDescriptor );

pSamplerDescriptor->release();

// ...

pSamplerState->release();
```

###### C++ (using NS::SharedPtr)

```cpp
NS::SharedPtr< MTL::SamplerDescriptor > pSamplerDescriptor = NS::TransferPtr( MTL::SamplerDescriptor::alloc()->init() );

pSamplerDescriptor->setSAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setTAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setRAddressMode( MTL::SamplerAddressModeRepeat );
pSamplerDescriptor->setMagFilter( MTL::SamplerMinMagFilterLinear );
pSamplerDescriptor->setMinFilter( MTL::SamplerMinMagFilterLinear );
pSamplerDescriptor->setMipFilter( MTL::SamplerMipFilterLinear );
pSamplerDescriptor->setSupportArgumentBuffers( true );

NS::SharedPtr< MTL::SamplerState > pSamplerState( pDevice->newSamplerState( pSamplerDescriptor ) );
```

#### A subset of bindings for Foundation classes is provided for seamless integration

###### Objective-C (with automatic reference counting)

```objc
NSAutoreleasePool*  pool   = [[NSAutoreleasePool alloc] init];
NSString*           string = [NSString stringWithCString: "Hello World" encoding: NSASCIIStringEncoding];

printf( "string = \"%s\"\n", [string cStringUsingEncoding: NSASCIIStringEncoding] );
```

###### Objective-C

```objc
NSAutoreleasePool* pool   = [[NSAutoreleasePool alloc] init];
NSString*          string = [NSString stringWithCString: "Hello World" encoding: NSASCIIStringEncoding];
								
printf( "string = \"%s\"\n", [string cStringUsingEncoding: NSASCIIStringEncoding] );

[pool release];
```

###### C++

```cpp
NS::AutoreleasePool* pPool   = NS::AutoreleasePool::alloc()->init();
NS::String*          pString = NS::String::string( "Hello World", NS::ASCIIStringEncoding );

printf( "pString = \"%s\"\n", pString->cString( NS::ASCIIStringEncoding ) );

pPool->release();
```

###### C++ (using NS::SharedPtr)

```cpp
NS::SharedPtr< NS::AutoreleasePool > pPool   = NS::TransferPtr( NS::AutoreleasePool::alloc()->init() );
NS::String*                          pString = NS::String::string( "Hello World", NS::ASCIIStringEncoding );

printf( "pString = \"%s\"\n", pString->cString( NS::ASCIIStringEncoding ) );
```

####  Containers

Use the CoreFoundation framework to create `NS::Array` and `NS::Dictionary` instances.

```cpp
MTL::AccelerationStructureTriangleGeometryDescriptor* pGeoDescriptor  = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
CFTypeRef                                             descriptors[]   = { ( CFTypeRef )( pGeoDescriptor ) };
NS::Array*                                            pGeoDescriptors = ( NS::Array* )( CFArrayCreate( kCFAllocatorDefault, descriptors, SIZEOF_ARRAY( descriptors), &kCFTypeArrayCallBacks ) );

// ...

pGeoDescriptors->release();
```

Containers, such as `NS::Array` and `NS::Dictionary`, retain the objects they hold and release them when the container is deallocated.

#### Accessing the Metal Drawable

```cpp
#import <QuartzCore/QuartzCore.hpp>

// ...

CA::MetalLayer*    pMetalLayer    = /* layer associated with the view */;
CA::MetalDrawable* pMetalDrawable = pMetalLayer->nextDrawable();

// ...
```
