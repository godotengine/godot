//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ResourceManager.cpp: Implements the the ResourceManager classes, which handle allocation and
// lifetime of GL objects.

#include "libANGLE/ResourceManager.h"

#include "libANGLE/Buffer.h"
#include "libANGLE/Context.h"
#include "libANGLE/Fence.h"
#include "libANGLE/MemoryObject.h"
#include "libANGLE/Path.h"
#include "libANGLE/Program.h"
#include "libANGLE/ProgramPipeline.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Sampler.h"
#include "libANGLE/Semaphore.h"
#include "libANGLE/Shader.h"
#include "libANGLE/Texture.h"
#include "libANGLE/renderer/ContextImpl.h"

namespace gl
{

namespace
{

template <typename ResourceType, typename IDType>
IDType AllocateEmptyObject(HandleAllocator *handleAllocator,
                           ResourceMap<ResourceType, IDType> *objectMap)
{
    IDType handle = FromGL<IDType>(handleAllocator->allocate());
    objectMap->assign(handle, nullptr);
    return handle;
}

}  // anonymous namespace

template <typename HandleAllocatorType>
ResourceManagerBase<HandleAllocatorType>::ResourceManagerBase() : mRefCount(1)
{}

template <typename HandleAllocatorType>
void ResourceManagerBase<HandleAllocatorType>::addRef()
{
    mRefCount++;
}

template <typename HandleAllocatorType>
void ResourceManagerBase<HandleAllocatorType>::release(const Context *context)
{
    if (--mRefCount == 0)
    {
        reset(context);
        delete this;
    }
}

template <typename ResourceType, typename HandleAllocatorType, typename ImplT, typename IDType>
TypedResourceManager<ResourceType, HandleAllocatorType, ImplT, IDType>::~TypedResourceManager()
{
    ASSERT(mObjectMap.empty());
}

template <typename ResourceType, typename HandleAllocatorType, typename ImplT, typename IDType>
void TypedResourceManager<ResourceType, HandleAllocatorType, ImplT, IDType>::reset(
    const Context *context)
{
    this->mHandleAllocator.reset();
    for (const auto &resource : mObjectMap)
    {
        if (resource.second)
        {
            ImplT::DeleteObject(context, resource.second);
        }
    }
    mObjectMap.clear();
}

template <typename ResourceType, typename HandleAllocatorType, typename ImplT, typename IDType>
void TypedResourceManager<ResourceType, HandleAllocatorType, ImplT, IDType>::deleteObject(
    const Context *context,
    IDType handle)
{
    ResourceType *resource = nullptr;
    if (!mObjectMap.erase(handle, &resource))
    {
        return;
    }

    // Requires an explicit this-> because of C++ template rules.
    this->mHandleAllocator.release(GetIDValue(handle));

    if (resource)
    {
        ImplT::DeleteObject(context, resource);
    }
}

template class ResourceManagerBase<HandleAllocator>;
template class ResourceManagerBase<HandleRangeAllocator>;
template class TypedResourceManager<Buffer, HandleAllocator, BufferManager, BufferID>;
template class TypedResourceManager<Texture, HandleAllocator, TextureManager, TextureID>;
template class TypedResourceManager<Renderbuffer,
                                    HandleAllocator,
                                    RenderbufferManager,
                                    RenderbufferID>;
template class TypedResourceManager<Sampler, HandleAllocator, SamplerManager, SamplerID>;
template class TypedResourceManager<Sync, HandleAllocator, SyncManager, GLuint>;
template class TypedResourceManager<Framebuffer,
                                    HandleAllocator,
                                    FramebufferManager,
                                    FramebufferID>;
template class TypedResourceManager<ProgramPipeline,
                                    HandleAllocator,
                                    ProgramPipelineManager,
                                    ProgramPipelineID>;

// BufferManager Implementation.

// static
Buffer *BufferManager::AllocateNewObject(rx::GLImplFactory *factory, BufferID handle)
{
    Buffer *buffer = new Buffer(factory, handle);
    buffer->addRef();
    return buffer;
}

// static
void BufferManager::DeleteObject(const Context *context, Buffer *buffer)
{
    buffer->release(context);
}

BufferID BufferManager::createBuffer()
{
    return AllocateEmptyObject(&mHandleAllocator, &mObjectMap);
}

Buffer *BufferManager::getBuffer(BufferID handle) const
{
    return mObjectMap.query(handle);
}

// ShaderProgramManager Implementation.

ShaderProgramManager::ShaderProgramManager() {}

ShaderProgramManager::~ShaderProgramManager()
{
    ASSERT(mPrograms.empty());
    ASSERT(mShaders.empty());
}

void ShaderProgramManager::reset(const Context *context)
{
    while (!mPrograms.empty())
    {
        deleteProgram(context, {mPrograms.begin()->first});
    }
    mPrograms.clear();
    while (!mShaders.empty())
    {
        deleteShader(context, {mShaders.begin()->first});
    }
    mShaders.clear();
}

ShaderProgramID ShaderProgramManager::createShader(rx::GLImplFactory *factory,
                                                   const gl::Limitations &rendererLimitations,
                                                   ShaderType type)
{
    ASSERT(type != ShaderType::InvalidEnum);
    ShaderProgramID handle = ShaderProgramID{mHandleAllocator.allocate()};
    mShaders.assign(handle, new Shader(this, factory, rendererLimitations, type, handle));
    return handle;
}

void ShaderProgramManager::deleteShader(const Context *context, ShaderProgramID shader)
{
    deleteObject(context, &mShaders, shader);
}

Shader *ShaderProgramManager::getShader(ShaderProgramID handle) const
{
    return mShaders.query(handle);
}

ShaderProgramID ShaderProgramManager::createProgram(rx::GLImplFactory *factory)
{
    ShaderProgramID handle = ShaderProgramID{mHandleAllocator.allocate()};
    mPrograms.assign(handle, new Program(factory, this, handle));
    return handle;
}

void ShaderProgramManager::deleteProgram(const gl::Context *context, ShaderProgramID program)
{
    deleteObject(context, &mPrograms, program);
}

template <typename ObjectType, typename IDType>
void ShaderProgramManager::deleteObject(const Context *context,
                                        ResourceMap<ObjectType, IDType> *objectMap,
                                        IDType id)
{
    ObjectType *object = objectMap->query(id);
    if (!object)
    {
        return;
    }

    if (object->getRefCount() == 0)
    {
        mHandleAllocator.release(id.value);
        object->onDestroy(context);
        objectMap->erase(id, &object);
    }
    else
    {
        object->flagForDeletion();
    }
}

// TextureManager Implementation.

// static
Texture *TextureManager::AllocateNewObject(rx::GLImplFactory *factory,
                                           TextureID handle,
                                           TextureType type)
{
    Texture *texture = new Texture(factory, handle, type);
    texture->addRef();
    return texture;
}

// static
void TextureManager::DeleteObject(const Context *context, Texture *texture)
{
    texture->release(context);
}

TextureID TextureManager::createTexture()
{
    return AllocateEmptyObject(&mHandleAllocator, &mObjectMap);
}

void TextureManager::signalAllTexturesDirty() const
{
    for (const auto &texture : mObjectMap)
    {
        if (texture.second)
        {
            // We don't know if the Texture needs init, but that's ok, since it will only force
            // a re-check, and will not initialize the pixels if it's not needed.
            texture.second->signalDirtyStorage(InitState::MayNeedInit);
        }
    }
}

void TextureManager::enableHandleAllocatorLogging()
{
    mHandleAllocator.enableLogging(true);
}

// RenderbufferManager Implementation.

// static
Renderbuffer *RenderbufferManager::AllocateNewObject(rx::GLImplFactory *factory,
                                                     RenderbufferID handle)
{
    Renderbuffer *renderbuffer = new Renderbuffer(factory, handle);
    renderbuffer->addRef();
    return renderbuffer;
}

// static
void RenderbufferManager::DeleteObject(const Context *context, Renderbuffer *renderbuffer)
{
    renderbuffer->release(context);
}

RenderbufferID RenderbufferManager::createRenderbuffer()
{
    return {AllocateEmptyObject(&mHandleAllocator, &mObjectMap)};
}

Renderbuffer *RenderbufferManager::getRenderbuffer(RenderbufferID handle) const
{
    return mObjectMap.query(handle);
}

// SamplerManager Implementation.

// static
Sampler *SamplerManager::AllocateNewObject(rx::GLImplFactory *factory, SamplerID handle)
{
    Sampler *sampler = new Sampler(factory, handle);
    sampler->addRef();
    return sampler;
}

// static
void SamplerManager::DeleteObject(const Context *context, Sampler *sampler)
{
    sampler->release(context);
}

SamplerID SamplerManager::createSampler()
{
    return AllocateEmptyObject(&mHandleAllocator, &mObjectMap);
}

Sampler *SamplerManager::getSampler(SamplerID handle) const
{
    return mObjectMap.query(handle);
}

bool SamplerManager::isSampler(SamplerID sampler) const
{
    return mObjectMap.contains(sampler);
}

// SyncManager Implementation.

// static
void SyncManager::DeleteObject(const Context *context, Sync *sync)
{
    sync->release(context);
}

GLuint SyncManager::createSync(rx::GLImplFactory *factory)
{
    GLuint handle = mHandleAllocator.allocate();
    Sync *sync    = new Sync(factory->createSync(), handle);
    sync->addRef();
    mObjectMap.assign(handle, sync);
    return handle;
}

Sync *SyncManager::getSync(GLuint handle) const
{
    return mObjectMap.query(handle);
}

// PathManager Implementation.

PathManager::PathManager() = default;

angle::Result PathManager::createPaths(Context *context, GLsizei range, PathID *createdOut)
{
    *createdOut = {0};

    // Allocate client side handles.
    const GLuint client = mHandleAllocator.allocateRange(static_cast<GLuint>(range));
    if (client == HandleRangeAllocator::kInvalidHandle)
    {
        context->handleError(GL_OUT_OF_MEMORY, "Failed to allocate path handle range.", __FILE__,
                             ANGLE_FUNCTION, __LINE__);
        return angle::Result::Stop;
    }

    const auto &paths = context->getImplementation()->createPaths(range);
    if (paths.empty())
    {
        mHandleAllocator.releaseRange(client, range);
        context->handleError(GL_OUT_OF_MEMORY, "Failed to allocate path objects.", __FILE__,
                             ANGLE_FUNCTION, __LINE__);
        return angle::Result::Stop;
    }

    for (GLsizei i = 0; i < range; ++i)
    {
        rx::PathImpl *impl = paths[static_cast<unsigned>(i)];
        PathID id          = PathID{client + i};
        mPaths.assign(id, new Path(impl));
    }
    *createdOut = PathID{client};
    return angle::Result::Continue;
}

void PathManager::deletePaths(PathID first, GLsizei range)
{
    GLuint firstHandle = first.value;
    for (GLsizei i = 0; i < range; ++i)
    {
        GLuint id = firstHandle + i;
        Path *p   = nullptr;
        if (!mPaths.erase({id}, &p))
            continue;
        delete p;
    }
    mHandleAllocator.releaseRange(firstHandle, static_cast<GLuint>(range));
}

Path *PathManager::getPath(PathID handle) const
{
    return mPaths.query(handle);
}

bool PathManager::hasPath(PathID handle) const
{
    return mHandleAllocator.isUsed(GetIDValue(handle));
}

PathManager::~PathManager()
{
    ASSERT(mPaths.empty());
}

void PathManager::reset(const Context *context)
{
    for (auto path : mPaths)
    {
        SafeDelete(path.second);
    }
    mPaths.clear();
}

// FramebufferManager Implementation.

// static
Framebuffer *FramebufferManager::AllocateNewObject(rx::GLImplFactory *factory,
                                                   FramebufferID handle,
                                                   const Caps &caps)
{
    // Make sure the caller isn't using a reserved handle.
    ASSERT(handle != Framebuffer::kDefaultDrawFramebufferHandle);
    return new Framebuffer(caps, factory, handle);
}

// static
void FramebufferManager::DeleteObject(const Context *context, Framebuffer *framebuffer)
{
    framebuffer->onDestroy(context);
    delete framebuffer;
}

FramebufferID FramebufferManager::createFramebuffer()
{
    return AllocateEmptyObject(&mHandleAllocator, &mObjectMap);
}

Framebuffer *FramebufferManager::getFramebuffer(FramebufferID handle) const
{
    return mObjectMap.query(handle);
}

void FramebufferManager::setDefaultFramebuffer(Framebuffer *framebuffer)
{
    ASSERT(framebuffer == nullptr || framebuffer->isDefault());
    mObjectMap.assign(Framebuffer::kDefaultDrawFramebufferHandle, framebuffer);
}

void FramebufferManager::invalidateFramebufferCompletenessCache() const
{
    for (const auto &framebuffer : mObjectMap)
    {
        if (framebuffer.second)
        {
            framebuffer.second->invalidateCompletenessCache();
        }
    }
}

// ProgramPipelineManager Implementation.

// static
ProgramPipeline *ProgramPipelineManager::AllocateNewObject(rx::GLImplFactory *factory,
                                                           ProgramPipelineID handle)
{
    ProgramPipeline *pipeline = new ProgramPipeline(factory, handle);
    pipeline->addRef();
    return pipeline;
}

// static
void ProgramPipelineManager::DeleteObject(const Context *context, ProgramPipeline *pipeline)
{
    pipeline->release(context);
}

ProgramPipelineID ProgramPipelineManager::createProgramPipeline()
{
    return AllocateEmptyObject(&mHandleAllocator, &mObjectMap);
}

ProgramPipeline *ProgramPipelineManager::getProgramPipeline(ProgramPipelineID handle) const
{
    return mObjectMap.query(handle);
}

// MemoryObjectManager Implementation.

MemoryObjectManager::MemoryObjectManager() {}

MemoryObjectManager::~MemoryObjectManager()
{
    ASSERT(mMemoryObjects.empty());
}

void MemoryObjectManager::reset(const Context *context)
{
    while (!mMemoryObjects.empty())
    {
        deleteMemoryObject(context, {mMemoryObjects.begin()->first});
    }
    mMemoryObjects.clear();
}

MemoryObjectID MemoryObjectManager::createMemoryObject(rx::GLImplFactory *factory)
{
    MemoryObjectID handle      = MemoryObjectID{mHandleAllocator.allocate()};
    MemoryObject *memoryObject = new MemoryObject(factory, handle);
    memoryObject->addRef();
    mMemoryObjects.assign(handle, memoryObject);
    return handle;
}

void MemoryObjectManager::deleteMemoryObject(const Context *context, MemoryObjectID handle)
{
    MemoryObject *memoryObject = nullptr;
    if (!mMemoryObjects.erase(handle, &memoryObject))
    {
        return;
    }

    // Requires an explicit this-> because of C++ template rules.
    this->mHandleAllocator.release(handle.value);

    if (memoryObject)
    {
        memoryObject->release(context);
    }
}

MemoryObject *MemoryObjectManager::getMemoryObject(MemoryObjectID handle) const
{
    return mMemoryObjects.query(handle);
}

// SemaphoreManager Implementation.

SemaphoreManager::SemaphoreManager() {}

SemaphoreManager::~SemaphoreManager()
{
    ASSERT(mSemaphores.empty());
}

void SemaphoreManager::reset(const Context *context)
{
    while (!mSemaphores.empty())
    {
        deleteSemaphore(context, {mSemaphores.begin()->first});
    }
    mSemaphores.clear();
}

SemaphoreID SemaphoreManager::createSemaphore(rx::GLImplFactory *factory)
{
    SemaphoreID handle   = SemaphoreID{mHandleAllocator.allocate()};
    Semaphore *semaphore = new Semaphore(factory, handle);
    semaphore->addRef();
    mSemaphores.assign(handle, semaphore);
    return handle;
}

void SemaphoreManager::deleteSemaphore(const Context *context, SemaphoreID handle)
{
    Semaphore *semaphore = nullptr;
    if (!mSemaphores.erase(handle, &semaphore))
    {
        return;
    }

    // Requires an explicit this-> because of C++ template rules.
    this->mHandleAllocator.release(handle.value);

    if (semaphore)
    {
        semaphore->release(context);
    }
}

Semaphore *SemaphoreManager::getSemaphore(SemaphoreID handle) const
{
    return mSemaphores.query(handle);
}

}  // namespace gl
