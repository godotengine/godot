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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include "SDL_waylandshmbuffer.h"
#include "SDL_waylandvideo.h"

static bool SetTempFileSize(int fd, off_t size)
{
#ifdef HAVE_POSIX_FALLOCATE
    sigset_t set, old_set;
    int ret;

    /* SIGALRM can potentially block a large posix_fallocate() operation
     * from succeeding, so block it.
     */
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    sigprocmask(SIG_BLOCK, &set, &old_set);

    do {
        ret = posix_fallocate(fd, 0, size);
    } while (ret == EINTR);

    sigprocmask(SIG_SETMASK, &old_set, NULL);

    if (ret == 0) {
        return true;
    } else if (ret != EINVAL && errno != EOPNOTSUPP) {
        return false;
    }
#endif

    if (ftruncate(fd, size) < 0) {
        return false;
    }
    return true;
}

static int CreateTempFD(off_t size)
{
    int fd;

#ifdef HAVE_MEMFD_CREATE
    fd = memfd_create("SDL", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd >= 0) {
        fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_SEAL);
    } else
#endif
    {
        static const char template[] = "/sdl-shared-XXXXXX";
        const char *xdg_path;
        char tmp_path[PATH_MAX];

        xdg_path = SDL_getenv("XDG_RUNTIME_DIR");
        if (!xdg_path) {
            return -1;
        }

        SDL_strlcpy(tmp_path, xdg_path, PATH_MAX);
        SDL_strlcat(tmp_path, template, PATH_MAX);

        fd = mkostemp(tmp_path, O_CLOEXEC);
        if (fd < 0) {
            return -1;
        }

        // Need to manually unlink the temp files, or they can persist after close and fill up the temp storage.
        unlink(tmp_path);
    }

    if (!SetTempFileSize(fd, size)) {
        close(fd);
        return -1;
    }

    return fd;
}

static void buffer_handle_release(void *data, struct wl_buffer *wl_buffer)
{
    // NOP
}

static struct wl_buffer_listener buffer_listener = {
    buffer_handle_release
};

bool Wayland_AllocSHMBuffer(int width, int height, struct Wayland_SHMBuffer *shmBuffer)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *data = vd->internal;
    struct wl_shm_pool *shm_pool;
    const Uint32 SHM_FMT = WL_SHM_FORMAT_ARGB8888;

    if (!shmBuffer) {
        return SDL_InvalidParamError("shmBuffer");
    }

    const int stride = width * 4;
    shmBuffer->shm_data_size = stride * height;

    const int shm_fd = CreateTempFD(shmBuffer->shm_data_size);
    if (shm_fd < 0) {
        return SDL_SetError("Creating SHM buffer failed.");
    }

    shmBuffer->shm_data = mmap(NULL, shmBuffer->shm_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shmBuffer->shm_data == MAP_FAILED) {
        shmBuffer->shm_data = NULL;
        close(shm_fd);
        return SDL_SetError("mmap() failed.");
    }

    SDL_assert(shmBuffer->shm_data != NULL);

    shm_pool = wl_shm_create_pool(data->shm, shm_fd, shmBuffer->shm_data_size);
    shmBuffer->wl_buffer = wl_shm_pool_create_buffer(shm_pool, 0, width, height, stride, SHM_FMT);
    wl_buffer_add_listener(shmBuffer->wl_buffer, &buffer_listener, shmBuffer);

    wl_shm_pool_destroy(shm_pool);
    close(shm_fd);

    return true;
}

void Wayland_ReleaseSHMBuffer(struct Wayland_SHMBuffer *shmBuffer)
{
    if (shmBuffer) {
        if (shmBuffer->wl_buffer) {
            wl_buffer_destroy(shmBuffer->wl_buffer);
            shmBuffer->wl_buffer = NULL;
        }
        if (shmBuffer->shm_data) {
            munmap(shmBuffer->shm_data, shmBuffer->shm_data_size);
            shmBuffer->shm_data = NULL;
        }
        shmBuffer->shm_data_size = 0;
    }
}

#endif
