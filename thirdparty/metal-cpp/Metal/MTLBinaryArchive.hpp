//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLBinaryArchive.hpp
//
// Copyright 2020-2025 Apple Inc.
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

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
class BinaryArchiveDescriptor;
class ComputePipelineDescriptor;
class Device;
class FunctionDescriptor;
class Library;
class MeshRenderPipelineDescriptor;
class RenderPipelineDescriptor;
class StitchedLibraryDescriptor;
class TileRenderPipelineDescriptor;
_MTL_ENUM(NS::UInteger, BinaryArchiveError) {
    BinaryArchiveErrorNone = 0,
    BinaryArchiveErrorInvalidFile = 1,
    BinaryArchiveErrorUnexpectedElement = 2,
    BinaryArchiveErrorCompilationFailure = 3,
    BinaryArchiveErrorInternalError = 4,
};

_MTL_CONST(NS::ErrorDomain, BinaryArchiveDomain);
class BinaryArchiveDescriptor : public NS::Copying<BinaryArchiveDescriptor>
{
public:
    static BinaryArchiveDescriptor* alloc();

    BinaryArchiveDescriptor*        init();

    void                            setUrl(const NS::URL* url);
    NS::URL*                        url() const;
};
class BinaryArchive : public NS::Referencing<BinaryArchive>
{
public:
    bool        addComputePipelineFunctions(const MTL::ComputePipelineDescriptor* descriptor, NS::Error** error);

    bool        addFunction(const MTL::FunctionDescriptor* descriptor, const MTL::Library* library, NS::Error** error);

    bool        addLibrary(const MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error);

    bool        addMeshRenderPipelineFunctions(const MTL::MeshRenderPipelineDescriptor* descriptor, NS::Error** error);

    bool        addRenderPipelineFunctions(const MTL::RenderPipelineDescriptor* descriptor, NS::Error** error);

    bool        addTileRenderPipelineFunctions(const MTL::TileRenderPipelineDescriptor* descriptor, NS::Error** error);

    Device*     device() const;

    NS::String* label() const;

    bool        serializeToURL(const NS::URL* url, NS::Error** error);

    void        setLabel(const NS::String* label);
};

}
_MTL_PRIVATE_DEF_CONST(NS::ErrorDomain, BinaryArchiveDomain);
_MTL_INLINE MTL::BinaryArchiveDescriptor* MTL::BinaryArchiveDescriptor::alloc()
{
    return NS::Object::alloc<MTL::BinaryArchiveDescriptor>(_MTL_PRIVATE_CLS(MTLBinaryArchiveDescriptor));
}

_MTL_INLINE MTL::BinaryArchiveDescriptor* MTL::BinaryArchiveDescriptor::init()
{
    return NS::Object::init<MTL::BinaryArchiveDescriptor>();
}

_MTL_INLINE void MTL::BinaryArchiveDescriptor::setUrl(const NS::URL* url)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setUrl_), url);
}

_MTL_INLINE NS::URL* MTL::BinaryArchiveDescriptor::url() const
{
    return Object::sendMessage<NS::URL*>(this, _MTL_PRIVATE_SEL(url));
}

_MTL_INLINE bool MTL::BinaryArchive::addComputePipelineFunctions(const MTL::ComputePipelineDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addComputePipelineFunctionsWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addFunction(const MTL::FunctionDescriptor* descriptor, const MTL::Library* library, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addFunctionWithDescriptor_library_error_), descriptor, library, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addLibrary(const MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addLibraryWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addMeshRenderPipelineFunctions(const MTL::MeshRenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addMeshRenderPipelineFunctionsWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addRenderPipelineFunctions(const MTL::RenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addRenderPipelineFunctionsWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addTileRenderPipelineFunctions(const MTL::TileRenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(addTileRenderPipelineFunctionsWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE MTL::Device* MTL::BinaryArchive::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::BinaryArchive::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE bool MTL::BinaryArchive::serializeToURL(const NS::URL* url, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(serializeToURL_error_), url, error);
}

_MTL_INLINE void MTL::BinaryArchive::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}
