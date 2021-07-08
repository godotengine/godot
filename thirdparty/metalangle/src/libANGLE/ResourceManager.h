//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ResourceManager.h : Defines the ResourceManager classes, which handle allocation and lifetime of
// GL objects.

#ifndef LIBANGLE_RESOURCEMANAGER_H_
#define LIBANGLE_RESOURCEMANAGER_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/HandleAllocator.h"
#include "libANGLE/HandleRangeAllocator.h"
#include "libANGLE/ResourceMap.h"

namespace rx
{
class GLImplFactory;
}

namespace gl
{
class Buffer;
struct Caps;
class Context;
class Sync;
class Framebuffer;
struct Limitations;
class MemoryObject;
class Path;
class Program;
class ProgramPipeline;
class Renderbuffer;
class Sampler;
class Shader;
class Semaphore;
class Texture;

template <typename HandleAllocatorType>
class ResourceManagerBase : angle::NonCopyable
{
  public:
    ResourceManagerBase();

    void addRef();
    void release(const Context *context);

  protected:
    virtual void reset(const Context *context) = 0;
    virtual ~ResourceManagerBase() {}

    HandleAllocatorType mHandleAllocator;

  private:
    size_t mRefCount;
};

template <typename ResourceType, typename HandleAllocatorType, typename ImplT, typename IDType>
class TypedResourceManager : public ResourceManagerBase<HandleAllocatorType>
{
  public:
    TypedResourceManager() {}

    void deleteObject(const Context *context, IDType handle);
    ANGLE_INLINE bool isHandleGenerated(IDType handle) const
    {
        // Zero is always assumed to have been generated implicitly.
        return GetIDValue(handle) == 0 || mObjectMap.contains(handle);
    }

  protected:
    ~TypedResourceManager() override;

    // Inlined in the header for performance.
    template <typename... ArgTypes>
    ANGLE_INLINE ResourceType *checkObjectAllocation(rx::GLImplFactory *factory,
                                                     IDType handle,
                                                     ArgTypes... args)
    {
        ResourceType *value = mObjectMap.query(handle);
        if (value)
        {
            return value;
        }

        if (GetIDValue(handle) == 0)
        {
            return nullptr;
        }

        return checkObjectAllocationImpl(factory, handle, args...);
    }

    void reset(const Context *context) override;

    ResourceMap<ResourceType, IDType> mObjectMap;

  private:
    template <typename... ArgTypes>
    ResourceType *checkObjectAllocationImpl(rx::GLImplFactory *factory,
                                            IDType handle,
                                            ArgTypes... args)
    {
        ResourceType *object = ImplT::AllocateNewObject(factory, handle, args...);

        if (!mObjectMap.contains(handle))
        {
            this->mHandleAllocator.reserve(GetIDValue(handle));
        }
        mObjectMap.assign(handle, object);

        return object;
    }
};

class BufferManager : public TypedResourceManager<Buffer, HandleAllocator, BufferManager, BufferID>
{
  public:
    BufferID createBuffer();
    Buffer *getBuffer(BufferID handle) const;

    ANGLE_INLINE Buffer *checkBufferAllocation(rx::GLImplFactory *factory, BufferID handle)
    {
        return checkObjectAllocation(factory, handle);
    }

    // TODO(jmadill): Investigate design which doesn't expose these methods publicly.
    static Buffer *AllocateNewObject(rx::GLImplFactory *factory, BufferID handle);
    static void DeleteObject(const Context *context, Buffer *buffer);

  protected:
    ~BufferManager() override {}
};

class ShaderProgramManager : public ResourceManagerBase<HandleAllocator>
{
  public:
    ShaderProgramManager();

    ShaderProgramID createShader(rx::GLImplFactory *factory,
                                 const Limitations &rendererLimitations,
                                 ShaderType type);
    void deleteShader(const Context *context, ShaderProgramID shader);
    Shader *getShader(ShaderProgramID handle) const;

    ShaderProgramID createProgram(rx::GLImplFactory *factory);
    void deleteProgram(const Context *context, ShaderProgramID program);

    ANGLE_INLINE Program *getProgram(ShaderProgramID handle) const
    {
        return mPrograms.query(handle);
    }

  protected:
    ~ShaderProgramManager() override;

  private:
    template <typename ObjectType, typename IDType>
    void deleteObject(const Context *context,
                      ResourceMap<ObjectType, IDType> *objectMap,
                      IDType id);

    void reset(const Context *context) override;

    ResourceMap<Shader, ShaderProgramID> mShaders;
    ResourceMap<Program, ShaderProgramID> mPrograms;
};

class TextureManager
    : public TypedResourceManager<Texture, HandleAllocator, TextureManager, TextureID>
{
  public:
    TextureID createTexture();
    ANGLE_INLINE Texture *getTexture(TextureID handle) const
    {
        ASSERT(mObjectMap.query({0}) == nullptr);
        return mObjectMap.query(handle);
    }

    void signalAllTexturesDirty() const;

    ANGLE_INLINE Texture *checkTextureAllocation(rx::GLImplFactory *factory,
                                                 TextureID handle,
                                                 TextureType type)
    {
        return checkObjectAllocation(factory, handle, type);
    }

    static Texture *AllocateNewObject(rx::GLImplFactory *factory,
                                      TextureID handle,
                                      TextureType type);
    static void DeleteObject(const Context *context, Texture *texture);

    void enableHandleAllocatorLogging();

  protected:
    ~TextureManager() override {}
};

class RenderbufferManager : public TypedResourceManager<Renderbuffer,
                                                        HandleAllocator,
                                                        RenderbufferManager,
                                                        RenderbufferID>
{
  public:
    RenderbufferID createRenderbuffer();
    Renderbuffer *getRenderbuffer(RenderbufferID handle) const;

    Renderbuffer *checkRenderbufferAllocation(rx::GLImplFactory *factory, RenderbufferID handle)
    {
        return checkObjectAllocation(factory, handle);
    }

    static Renderbuffer *AllocateNewObject(rx::GLImplFactory *factory, RenderbufferID handle);
    static void DeleteObject(const Context *context, Renderbuffer *renderbuffer);

  protected:
    ~RenderbufferManager() override {}
};

class SamplerManager
    : public TypedResourceManager<Sampler, HandleAllocator, SamplerManager, SamplerID>
{
  public:
    SamplerID createSampler();
    Sampler *getSampler(SamplerID handle) const;
    bool isSampler(SamplerID sampler) const;

    Sampler *checkSamplerAllocation(rx::GLImplFactory *factory, SamplerID handle)
    {
        return checkObjectAllocation(factory, handle);
    }

    static Sampler *AllocateNewObject(rx::GLImplFactory *factory, SamplerID handle);
    static void DeleteObject(const Context *context, Sampler *sampler);

  protected:
    ~SamplerManager() override {}
};

class SyncManager : public TypedResourceManager<Sync, HandleAllocator, SyncManager, GLuint>
{
  public:
    GLuint createSync(rx::GLImplFactory *factory);
    Sync *getSync(GLuint handle) const;

    static void DeleteObject(const Context *context, Sync *sync);

  protected:
    ~SyncManager() override {}
};

class PathManager : public ResourceManagerBase<HandleRangeAllocator>
{
  public:
    PathManager();

    angle::Result createPaths(Context *context, GLsizei range, PathID *numCreated);
    void deletePaths(PathID first, GLsizei range);
    Path *getPath(PathID handle) const;
    bool hasPath(PathID handle) const;

  protected:
    ~PathManager() override;
    void reset(const Context *context) override;

  private:
    ResourceMap<Path, PathID> mPaths;
};

class FramebufferManager
    : public TypedResourceManager<Framebuffer, HandleAllocator, FramebufferManager, FramebufferID>
{
  public:
    FramebufferID createFramebuffer();
    Framebuffer *getFramebuffer(FramebufferID handle) const;
    void setDefaultFramebuffer(Framebuffer *framebuffer);

    void invalidateFramebufferCompletenessCache() const;

    Framebuffer *checkFramebufferAllocation(rx::GLImplFactory *factory,
                                            const Caps &caps,
                                            FramebufferID handle)
    {
        return checkObjectAllocation<const Caps &>(factory, handle, caps);
    }

    static Framebuffer *AllocateNewObject(rx::GLImplFactory *factory,
                                          FramebufferID handle,
                                          const Caps &caps);
    static void DeleteObject(const Context *context, Framebuffer *framebuffer);

  protected:
    ~FramebufferManager() override {}
};

class ProgramPipelineManager : public TypedResourceManager<ProgramPipeline,
                                                           HandleAllocator,
                                                           ProgramPipelineManager,
                                                           ProgramPipelineID>
{
  public:
    ProgramPipelineID createProgramPipeline();
    ProgramPipeline *getProgramPipeline(ProgramPipelineID handle) const;

    ProgramPipeline *checkProgramPipelineAllocation(rx::GLImplFactory *factory,
                                                    ProgramPipelineID handle)
    {
        return checkObjectAllocation(factory, handle);
    }

    static ProgramPipeline *AllocateNewObject(rx::GLImplFactory *factory, ProgramPipelineID handle);
    static void DeleteObject(const Context *context, ProgramPipeline *pipeline);

  protected:
    ~ProgramPipelineManager() override {}
};

class MemoryObjectManager : public ResourceManagerBase<HandleAllocator>
{
  public:
    MemoryObjectManager();

    MemoryObjectID createMemoryObject(rx::GLImplFactory *factory);
    void deleteMemoryObject(const Context *context, MemoryObjectID handle);
    MemoryObject *getMemoryObject(MemoryObjectID handle) const;

  protected:
    ~MemoryObjectManager() override;

  private:
    void reset(const Context *context) override;

    ResourceMap<MemoryObject, MemoryObjectID> mMemoryObjects;
};

class SemaphoreManager : public ResourceManagerBase<HandleAllocator>
{
  public:
    SemaphoreManager();

    SemaphoreID createSemaphore(rx::GLImplFactory *factory);
    void deleteSemaphore(const Context *context, SemaphoreID handle);
    Semaphore *getSemaphore(SemaphoreID handle) const;

  protected:
    ~SemaphoreManager() override;

  private:
    void reset(const Context *context) override;

    ResourceMap<Semaphore, SemaphoreID> mSemaphores;
};

}  // namespace gl

#endif  // LIBANGLE_RESOURCEMANAGER_H_
