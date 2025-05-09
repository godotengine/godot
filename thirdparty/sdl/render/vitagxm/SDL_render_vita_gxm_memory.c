/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#ifdef SDL_VIDEO_RENDER_VITA_GXM

#include "SDL_render_vita_gxm_memory.h"

void *vita_mem_alloc(unsigned int type, unsigned int size, unsigned int alignment, unsigned int attribs, SceUID *uid)
{
    void *mem;

    if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW) {
        size = ALIGN(size, 256 * 1024);
    } else if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW) {
        size = ALIGN(size, 1024 * 1024);
    } else {
        size = ALIGN(size, 4 * 1024);
    }

    *uid = sceKernelAllocMemBlock("gpu_mem", type, size, NULL);

    if (*uid < 0) {
        return NULL;
    }

    if (sceKernelGetMemBlockBase(*uid, &mem) < 0) {
        return NULL;
    }

    if (sceGxmMapMemory(mem, size, attribs) < 0) {
        return NULL;
    }

    return mem;
}

void vita_mem_free(SceUID uid)
{
    void *mem = NULL;
    if (sceKernelGetMemBlockBase(uid, &mem) < 0) {
        return;
    }
    sceGxmUnmapMemory(mem);
    sceKernelFreeMemBlock(uid);
}

void *vita_gpu_mem_alloc(VITA_GXM_RenderData *data, unsigned int size)
{
    void *mem;

    if (!data->texturePool) {
        int poolsize;
        int ret;
        SceKernelFreeMemorySizeInfo info;
        info.size = sizeof(SceKernelFreeMemorySizeInfo);
        sceKernelGetFreeMemorySize(&info);

        poolsize = ALIGN(info.size_cdram, 256 * 1024);
        if (poolsize > info.size_cdram) {
            poolsize = ALIGN(info.size_cdram - 256 * 1024, 256 * 1024);
        }
        data->texturePoolUID = sceKernelAllocMemBlock("gpu_texture_pool", SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, poolsize, NULL);
        if (data->texturePoolUID < 0) {
            return NULL;
        }

        ret = sceKernelGetMemBlockBase(data->texturePoolUID, &mem);
        if (ret < 0) {
            return NULL;
        }
        data->texturePool = sceClibMspaceCreate(mem, poolsize);

        if (!data->texturePool) {
            return NULL;
        }
        ret = sceGxmMapMemory(mem, poolsize, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
        if (ret < 0) {
            return NULL;
        }
    }
    return sceClibMspaceMemalign(data->texturePool, SCE_GXM_TEXTURE_ALIGNMENT, size);
}

void vita_gpu_mem_free(VITA_GXM_RenderData *data, void *ptr)
{
    if (data->texturePool) {
        sceClibMspaceFree(data->texturePool, ptr);
    }
}

void vita_gpu_mem_destroy(VITA_GXM_RenderData *data)
{
    void *mem = NULL;
    if (data->texturePool) {
        sceClibMspaceDestroy(data->texturePool);
        data->texturePool = NULL;
        if (sceKernelGetMemBlockBase(data->texturePoolUID, &mem) < 0) {
            return;
        }
        sceGxmUnmapMemory(mem);
        sceKernelFreeMemBlock(data->texturePoolUID);
    }
}

void *vita_mem_vertex_usse_alloc(unsigned int size, SceUID *uid, unsigned int *usse_offset)
{
    void *mem = NULL;

    size = ALIGN(size, 4096);
    *uid = sceKernelAllocMemBlock("vertex_usse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, NULL);

    if (sceKernelGetMemBlockBase(*uid, &mem) < 0) {
        return NULL;
    }
    if (sceGxmMapVertexUsseMemory(mem, size, usse_offset) < 0) {
        return NULL;
    }

    return mem;
}

void vita_mem_vertex_usse_free(SceUID uid)
{
    void *mem = NULL;
    if (sceKernelGetMemBlockBase(uid, &mem) < 0) {
        return;
    }
    sceGxmUnmapVertexUsseMemory(mem);
    sceKernelFreeMemBlock(uid);
}

void *vita_mem_fragment_usse_alloc(unsigned int size, SceUID *uid, unsigned int *usse_offset)
{
    void *mem = NULL;

    size = ALIGN(size, 4096);
    *uid = sceKernelAllocMemBlock("fragment_usse", SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, size, NULL);

    if (sceKernelGetMemBlockBase(*uid, &mem) < 0) {
        return NULL;
    }
    if (sceGxmMapFragmentUsseMemory(mem, size, usse_offset) < 0) {
        return NULL;
    }

    return mem;
}

void vita_mem_fragment_usse_free(SceUID uid)
{
    void *mem = NULL;
    if (sceKernelGetMemBlockBase(uid, &mem) < 0) {
        return;
    }
    sceGxmUnmapFragmentUsseMemory(mem);
    sceKernelFreeMemBlock(uid);
}

#endif // SDL_VIDEO_RENDER_VITA_GXM
