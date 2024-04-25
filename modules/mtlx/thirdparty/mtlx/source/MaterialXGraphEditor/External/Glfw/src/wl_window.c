//========================================================================
// GLFW 3.4 Wayland - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2014 Jonas Ådahl <jadahl@gmail.com>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#define _GNU_SOURCE

#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/timerfd.h>
#include <poll.h>


static int createTmpfileCloexec(char* tmpname)
{
    int fd;

    fd = mkostemp(tmpname, O_CLOEXEC);
    if (fd >= 0)
        unlink(tmpname);

    return fd;
}

/*
 * Create a new, unique, anonymous file of the given size, and
 * return the file descriptor for it. The file descriptor is set
 * CLOEXEC. The file is immediately suitable for mmap()'ing
 * the given size at offset zero.
 *
 * The file should not have a permanent backing store like a disk,
 * but may have if XDG_RUNTIME_DIR is not properly implemented in OS.
 *
 * The file name is deleted from the file system.
 *
 * The file is suitable for buffer sharing between processes by
 * transmitting the file descriptor over Unix sockets using the
 * SCM_RIGHTS methods.
 *
 * posix_fallocate() is used to guarantee that disk space is available
 * for the file at the given size. If disk space is insufficient, errno
 * is set to ENOSPC. If posix_fallocate() is not supported, program may
 * receive SIGBUS on accessing mmap()'ed file contents instead.
 */
static int createAnonymousFile(off_t size)
{
    static const char template[] = "/glfw-shared-XXXXXX";
    const char* path;
    char* name;
    int fd;
    int ret;

#ifdef HAVE_MEMFD_CREATE
    fd = memfd_create("glfw-shared", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd >= 0)
    {
        // We can add this seal before calling posix_fallocate(), as the file
        // is currently zero-sized anyway.
        //
        // There is also no need to check for the return value, we couldn’t do
        // anything with it anyway.
        fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_SEAL);
    }
    else
#elif defined(SHM_ANON)
    fd = shm_open(SHM_ANON, O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0)
#endif
    {
        path = getenv("XDG_RUNTIME_DIR");
        if (!path)
        {
            errno = ENOENT;
            return -1;
        }

        name = calloc(strlen(path) + sizeof(template), 1);
        strcpy(name, path);
        strcat(name, template);

        fd = createTmpfileCloexec(name);
        free(name);
        if (fd < 0)
            return -1;
    }

#if defined(SHM_ANON)
    // posix_fallocate does not work on SHM descriptors
    ret = ftruncate(fd, size);
#else
    ret = posix_fallocate(fd, 0, size);
#endif
    if (ret != 0)
    {
        close(fd);
        errno = ret;
        return -1;
    }
    return fd;
}

static struct wl_buffer* createShmBuffer(const GLFWimage* image)
{
    struct wl_shm_pool* pool;
    struct wl_buffer* buffer;
    int stride = image->width * 4;
    int length = image->width * image->height * 4;
    void* data;
    int fd, i;

    fd = createAnonymousFile(length);
    if (fd < 0)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Creating a buffer file for %d B failed: %m",
                        length);
        return NULL;
    }

    data = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: mmap failed: %m");
        close(fd);
        return NULL;
    }

    pool = wl_shm_create_pool(_glfw.wl.shm, fd, length);

    close(fd);
    unsigned char* source = (unsigned char*) image->pixels;
    unsigned char* target = data;
    for (i = 0;  i < image->width * image->height;  i++, source += 4)
    {
        unsigned int alpha = source[3];

        *target++ = (unsigned char) ((source[2] * alpha) / 255);
        *target++ = (unsigned char) ((source[1] * alpha) / 255);
        *target++ = (unsigned char) ((source[0] * alpha) / 255);
        *target++ = (unsigned char) alpha;
    }

    buffer =
        wl_shm_pool_create_buffer(pool, 0,
                                  image->width,
                                  image->height,
                                  stride, WL_SHM_FORMAT_ARGB8888);
    munmap(data, length);
    wl_shm_pool_destroy(pool);

    return buffer;
}

static void createDecoration(_GLFWdecorationWayland* decoration,
                             struct wl_surface* parent,
                             struct wl_buffer* buffer, GLFWbool opaque,
                             int x, int y,
                             int width, int height)
{
    struct wl_region* region;

    decoration->surface = wl_compositor_create_surface(_glfw.wl.compositor);
    decoration->subsurface =
        wl_subcompositor_get_subsurface(_glfw.wl.subcompositor,
                                        decoration->surface, parent);
    wl_subsurface_set_position(decoration->subsurface, x, y);
    decoration->viewport = wp_viewporter_get_viewport(_glfw.wl.viewporter,
                                                      decoration->surface);
    wp_viewport_set_destination(decoration->viewport, width, height);
    wl_surface_attach(decoration->surface, buffer, 0, 0);

    if (opaque)
    {
        region = wl_compositor_create_region(_glfw.wl.compositor);
        wl_region_add(region, 0, 0, width, height);
        wl_surface_set_opaque_region(decoration->surface, region);
        wl_surface_commit(decoration->surface);
        wl_region_destroy(region);
    }
    else
        wl_surface_commit(decoration->surface);
}

static void createDecorations(_GLFWwindow* window)
{
    unsigned char data[] = { 224, 224, 224, 255 };
    const GLFWimage image = { 1, 1, data };
    GLFWbool opaque = (data[3] == 255);

    if (!_glfw.wl.viewporter || !window->decorated || window->wl.decorations.serverSide)
        return;

    if (!window->wl.decorations.buffer)
        window->wl.decorations.buffer = createShmBuffer(&image);
    if (!window->wl.decorations.buffer)
        return;

    createDecoration(&window->wl.decorations.top, window->wl.surface,
                     window->wl.decorations.buffer, opaque,
                     0, -_GLFW_DECORATION_TOP,
                     window->wl.width, _GLFW_DECORATION_TOP);
    createDecoration(&window->wl.decorations.left, window->wl.surface,
                     window->wl.decorations.buffer, opaque,
                     -_GLFW_DECORATION_WIDTH, -_GLFW_DECORATION_TOP,
                     _GLFW_DECORATION_WIDTH, window->wl.height + _GLFW_DECORATION_TOP);
    createDecoration(&window->wl.decorations.right, window->wl.surface,
                     window->wl.decorations.buffer, opaque,
                     window->wl.width, -_GLFW_DECORATION_TOP,
                     _GLFW_DECORATION_WIDTH, window->wl.height + _GLFW_DECORATION_TOP);
    createDecoration(&window->wl.decorations.bottom, window->wl.surface,
                     window->wl.decorations.buffer, opaque,
                     -_GLFW_DECORATION_WIDTH, window->wl.height,
                     window->wl.width + _GLFW_DECORATION_HORIZONTAL, _GLFW_DECORATION_WIDTH);
}

static void destroyDecoration(_GLFWdecorationWayland* decoration)
{
    if (decoration->surface)
        wl_surface_destroy(decoration->surface);
    if (decoration->subsurface)
        wl_subsurface_destroy(decoration->subsurface);
    if (decoration->viewport)
        wp_viewport_destroy(decoration->viewport);
    decoration->surface = NULL;
    decoration->subsurface = NULL;
    decoration->viewport = NULL;
}

static void destroyDecorations(_GLFWwindow* window)
{
    destroyDecoration(&window->wl.decorations.top);
    destroyDecoration(&window->wl.decorations.left);
    destroyDecoration(&window->wl.decorations.right);
    destroyDecoration(&window->wl.decorations.bottom);
}

static void xdgDecorationHandleConfigure(void* data,
                                         struct zxdg_toplevel_decoration_v1* decoration,
                                         uint32_t mode)
{
    _GLFWwindow* window = data;

    window->wl.decorations.serverSide = (mode == ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);

    if (!window->wl.decorations.serverSide)
        createDecorations(window);
}

static const struct zxdg_toplevel_decoration_v1_listener xdgDecorationListener = {
    xdgDecorationHandleConfigure,
};

// Makes the surface considered as XRGB instead of ARGB.
static void setOpaqueRegion(_GLFWwindow* window)
{
    struct wl_region* region;

    region = wl_compositor_create_region(_glfw.wl.compositor);
    if (!region)
        return;

    wl_region_add(region, 0, 0, window->wl.width, window->wl.height);
    wl_surface_set_opaque_region(window->wl.surface, region);
    wl_surface_commit(window->wl.surface);
    wl_region_destroy(region);
}


static void resizeWindow(_GLFWwindow* window)
{
    int scale = window->wl.scale;
    int scaledWidth = window->wl.width * scale;
    int scaledHeight = window->wl.height * scale;
    wl_egl_window_resize(window->wl.native, scaledWidth, scaledHeight, 0, 0);
    if (!window->wl.transparent)
        setOpaqueRegion(window);
    _glfwInputFramebufferSize(window, scaledWidth, scaledHeight);
    _glfwInputWindowContentScale(window, scale, scale);

    if (!window->wl.decorations.top.surface)
        return;

    // Top decoration.
    wp_viewport_set_destination(window->wl.decorations.top.viewport,
                                window->wl.width, _GLFW_DECORATION_TOP);
    wl_surface_commit(window->wl.decorations.top.surface);

    // Left decoration.
    wp_viewport_set_destination(window->wl.decorations.left.viewport,
                                _GLFW_DECORATION_WIDTH, window->wl.height + _GLFW_DECORATION_TOP);
    wl_surface_commit(window->wl.decorations.left.surface);

    // Right decoration.
    wl_subsurface_set_position(window->wl.decorations.right.subsurface,
                               window->wl.width, -_GLFW_DECORATION_TOP);
    wp_viewport_set_destination(window->wl.decorations.right.viewport,
                                _GLFW_DECORATION_WIDTH, window->wl.height + _GLFW_DECORATION_TOP);
    wl_surface_commit(window->wl.decorations.right.surface);

    // Bottom decoration.
    wl_subsurface_set_position(window->wl.decorations.bottom.subsurface,
                               -_GLFW_DECORATION_WIDTH, window->wl.height);
    wp_viewport_set_destination(window->wl.decorations.bottom.viewport,
                                window->wl.width + _GLFW_DECORATION_HORIZONTAL, _GLFW_DECORATION_WIDTH);
    wl_surface_commit(window->wl.decorations.bottom.surface);
}

static void checkScaleChange(_GLFWwindow* window)
{
    int scale = 1;
    int i;
    int monitorScale;

    // Check if we will be able to set the buffer scale or not.
    if (_glfw.wl.compositorVersion < 3)
        return;

    // Get the scale factor from the highest scale monitor.
    for (i = 0; i < window->wl.monitorsCount; ++i)
    {
        monitorScale = window->wl.monitors[i]->wl.scale;
        if (scale < monitorScale)
            scale = monitorScale;
    }

    // Only change the framebuffer size if the scale changed.
    if (scale != window->wl.scale)
    {
        window->wl.scale = scale;
        wl_surface_set_buffer_scale(window->wl.surface, scale);
        resizeWindow(window);
    }
}

static void surfaceHandleEnter(void *data,
                               struct wl_surface *surface,
                               struct wl_output *output)
{
    _GLFWwindow* window = data;
    _GLFWmonitor* monitor = wl_output_get_user_data(output);

    if (window->wl.monitorsCount + 1 > window->wl.monitorsSize)
    {
        ++window->wl.monitorsSize;
        window->wl.monitors =
            realloc(window->wl.monitors,
                    window->wl.monitorsSize * sizeof(_GLFWmonitor*));
    }

    window->wl.monitors[window->wl.monitorsCount++] = monitor;

    checkScaleChange(window);
}

static void surfaceHandleLeave(void *data,
                               struct wl_surface *surface,
                               struct wl_output *output)
{
    _GLFWwindow* window = data;
    _GLFWmonitor* monitor = wl_output_get_user_data(output);
    GLFWbool found;
    int i;

    for (i = 0, found = GLFW_FALSE; i < window->wl.monitorsCount - 1; ++i)
    {
        if (monitor == window->wl.monitors[i])
            found = GLFW_TRUE;
        if (found)
            window->wl.monitors[i] = window->wl.monitors[i + 1];
    }
    window->wl.monitors[--window->wl.monitorsCount] = NULL;

    checkScaleChange(window);
}

static const struct wl_surface_listener surfaceListener = {
    surfaceHandleEnter,
    surfaceHandleLeave
};

static void setIdleInhibitor(_GLFWwindow* window, GLFWbool enable)
{
    if (enable && !window->wl.idleInhibitor && _glfw.wl.idleInhibitManager)
    {
        window->wl.idleInhibitor =
            zwp_idle_inhibit_manager_v1_create_inhibitor(
                _glfw.wl.idleInhibitManager, window->wl.surface);
        if (!window->wl.idleInhibitor)
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Idle inhibitor creation failed");
    }
    else if (!enable && window->wl.idleInhibitor)
    {
        zwp_idle_inhibitor_v1_destroy(window->wl.idleInhibitor);
        window->wl.idleInhibitor = NULL;
    }
}

static GLFWbool createSurface(_GLFWwindow* window,
                              const _GLFWwndconfig* wndconfig)
{
    window->wl.surface = wl_compositor_create_surface(_glfw.wl.compositor);
    if (!window->wl.surface)
        return GLFW_FALSE;

    wl_surface_add_listener(window->wl.surface,
                            &surfaceListener,
                            window);

    wl_surface_set_user_data(window->wl.surface, window);

    window->wl.native = wl_egl_window_create(window->wl.surface,
                                             wndconfig->width,
                                             wndconfig->height);
    if (!window->wl.native)
        return GLFW_FALSE;

    window->wl.width = wndconfig->width;
    window->wl.height = wndconfig->height;
    window->wl.scale = 1;

    if (!window->wl.transparent)
        setOpaqueRegion(window);

    return GLFW_TRUE;
}

static void setFullscreen(_GLFWwindow* window, _GLFWmonitor* monitor,
                          int refreshRate)
{
    if (window->wl.xdg.toplevel)
    {
        xdg_toplevel_set_fullscreen(
            window->wl.xdg.toplevel,
            monitor->wl.output);
    }
    setIdleInhibitor(window, GLFW_TRUE);
    if (!window->wl.decorations.serverSide)
        destroyDecorations(window);
}

static void xdgToplevelHandleConfigure(void* data,
                                       struct xdg_toplevel* toplevel,
                                       int32_t width,
                                       int32_t height,
                                       struct wl_array* states)
{
    _GLFWwindow* window = data;
    float aspectRatio;
    float targetRatio;
    uint32_t* state;
    GLFWbool maximized = GLFW_FALSE;
    GLFWbool fullscreen = GLFW_FALSE;
    GLFWbool activated = GLFW_FALSE;

    wl_array_for_each(state, states)
    {
        switch (*state)
        {
            case XDG_TOPLEVEL_STATE_MAXIMIZED:
                maximized = GLFW_TRUE;
                break;
            case XDG_TOPLEVEL_STATE_FULLSCREEN:
                fullscreen = GLFW_TRUE;
                break;
            case XDG_TOPLEVEL_STATE_RESIZING:
                break;
            case XDG_TOPLEVEL_STATE_ACTIVATED:
                activated = GLFW_TRUE;
                break;
        }
    }

    if (width != 0 && height != 0)
    {
        if (!maximized && !fullscreen)
        {
            if (window->numer != GLFW_DONT_CARE && window->denom != GLFW_DONT_CARE)
            {
                aspectRatio = (float)width / (float)height;
                targetRatio = (float)window->numer / (float)window->denom;
                if (aspectRatio < targetRatio)
                    height = width / targetRatio;
                else if (aspectRatio > targetRatio)
                    width = height * targetRatio;
            }
        }

        _glfwInputWindowSize(window, width, height);
        _glfwPlatformSetWindowSize(window, width, height);
        _glfwInputWindowDamage(window);
    }

    if (window->wl.wasFullscreen && window->autoIconify)
    {
        if (!activated || !fullscreen)
        {
            _glfwPlatformIconifyWindow(window);
            window->wl.wasFullscreen = GLFW_FALSE;
        }
    }
    if (fullscreen && activated)
        window->wl.wasFullscreen = GLFW_TRUE;
    _glfwInputWindowFocus(window, activated);
}

static void xdgToplevelHandleClose(void* data,
                                   struct xdg_toplevel* toplevel)
{
    _GLFWwindow* window = data;
    _glfwInputWindowCloseRequest(window);
}

static const struct xdg_toplevel_listener xdgToplevelListener = {
    xdgToplevelHandleConfigure,
    xdgToplevelHandleClose
};

static void xdgSurfaceHandleConfigure(void* data,
                                      struct xdg_surface* surface,
                                      uint32_t serial)
{
    xdg_surface_ack_configure(surface, serial);
}

static const struct xdg_surface_listener xdgSurfaceListener = {
    xdgSurfaceHandleConfigure
};

static void setXdgDecorations(_GLFWwindow* window)
{
    if (_glfw.wl.decorationManager)
    {
        window->wl.xdg.decoration =
            zxdg_decoration_manager_v1_get_toplevel_decoration(
                _glfw.wl.decorationManager, window->wl.xdg.toplevel);
        zxdg_toplevel_decoration_v1_add_listener(window->wl.xdg.decoration,
                                                 &xdgDecorationListener,
                                                 window);
        zxdg_toplevel_decoration_v1_set_mode(
            window->wl.xdg.decoration,
            ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
    }
    else
    {
        window->wl.decorations.serverSide = GLFW_FALSE;
        createDecorations(window);
    }
}

static GLFWbool createXdgSurface(_GLFWwindow* window)
{
    window->wl.xdg.surface = xdg_wm_base_get_xdg_surface(_glfw.wl.wmBase,
                                                         window->wl.surface);
    if (!window->wl.xdg.surface)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: xdg-surface creation failed");
        return GLFW_FALSE;
    }

    xdg_surface_add_listener(window->wl.xdg.surface,
                             &xdgSurfaceListener,
                             window);

    window->wl.xdg.toplevel = xdg_surface_get_toplevel(window->wl.xdg.surface);
    if (!window->wl.xdg.toplevel)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: xdg-toplevel creation failed");
        return GLFW_FALSE;
    }

    xdg_toplevel_add_listener(window->wl.xdg.toplevel,
                              &xdgToplevelListener,
                              window);

    if (window->wl.title)
        xdg_toplevel_set_title(window->wl.xdg.toplevel, window->wl.title);

    if (window->minwidth != GLFW_DONT_CARE && window->minheight != GLFW_DONT_CARE)
        xdg_toplevel_set_min_size(window->wl.xdg.toplevel,
                                  window->minwidth, window->minheight);
    if (window->maxwidth != GLFW_DONT_CARE && window->maxheight != GLFW_DONT_CARE)
        xdg_toplevel_set_max_size(window->wl.xdg.toplevel,
                                  window->maxwidth, window->maxheight);

    if (window->monitor)
    {
        xdg_toplevel_set_fullscreen(window->wl.xdg.toplevel,
                                    window->monitor->wl.output);
        setIdleInhibitor(window, GLFW_TRUE);
    }
    else if (window->wl.maximized)
    {
        xdg_toplevel_set_maximized(window->wl.xdg.toplevel);
        setIdleInhibitor(window, GLFW_FALSE);
        setXdgDecorations(window);
    }
    else
    {
        setIdleInhibitor(window, GLFW_FALSE);
        setXdgDecorations(window);
    }

    wl_surface_commit(window->wl.surface);
    wl_display_roundtrip(_glfw.wl.display);

    return GLFW_TRUE;
}

static void setCursorImage(_GLFWwindow* window,
                           _GLFWcursorWayland* cursorWayland)
{
    struct itimerspec timer = {};
    struct wl_cursor* wlCursor = cursorWayland->cursor;
    struct wl_cursor_image* image;
    struct wl_buffer* buffer;
    struct wl_surface* surface = _glfw.wl.cursorSurface;
    int scale = 1;

    if (!wlCursor)
        buffer = cursorWayland->buffer;
    else
    {
        if (window->wl.scale > 1 && cursorWayland->cursorHiDPI)
        {
            wlCursor = cursorWayland->cursorHiDPI;
            scale = 2;
        }

        image = wlCursor->images[cursorWayland->currentImage];
        buffer = wl_cursor_image_get_buffer(image);
        if (!buffer)
            return;

        timer.it_value.tv_sec = image->delay / 1000;
        timer.it_value.tv_nsec = (image->delay % 1000) * 1000000;
        timerfd_settime(_glfw.wl.cursorTimerfd, 0, &timer, NULL);

        cursorWayland->width = image->width;
        cursorWayland->height = image->height;
        cursorWayland->xhot = image->hotspot_x;
        cursorWayland->yhot = image->hotspot_y;
    }

    wl_pointer_set_cursor(_glfw.wl.pointer, _glfw.wl.serial,
                          surface,
                          cursorWayland->xhot / scale,
                          cursorWayland->yhot / scale);
    wl_surface_set_buffer_scale(surface, scale);
    wl_surface_attach(surface, buffer, 0, 0);
    wl_surface_damage(surface, 0, 0,
                      cursorWayland->width, cursorWayland->height);
    wl_surface_commit(surface);
}

static void incrementCursorImage(_GLFWwindow* window)
{
    _GLFWcursor* cursor;

    if (!window || window->wl.decorations.focus != mainWindow)
        return;

    cursor = window->wl.currentCursor;
    if (cursor && cursor->wl.cursor)
    {
        cursor->wl.currentImage += 1;
        cursor->wl.currentImage %= cursor->wl.cursor->image_count;
        setCursorImage(window, &cursor->wl);
    }
}

static void handleEvents(int timeout)
{
    struct wl_display* display = _glfw.wl.display;
    struct pollfd fds[] = {
        { wl_display_get_fd(display), POLLIN },
        { _glfw.wl.timerfd, POLLIN },
        { _glfw.wl.cursorTimerfd, POLLIN },
    };
    ssize_t read_ret;
    uint64_t repeats, i;

    while (wl_display_prepare_read(display) != 0)
        wl_display_dispatch_pending(display);

    // If an error different from EAGAIN happens, we have likely been
    // disconnected from the Wayland session, try to handle that the best we
    // can.
    if (wl_display_flush(display) < 0 && errno != EAGAIN)
    {
        _GLFWwindow* window = _glfw.windowListHead;
        while (window)
        {
            _glfwInputWindowCloseRequest(window);
            window = window->next;
        }
        wl_display_cancel_read(display);
        return;
    }

    if (poll(fds, 3, timeout) > 0)
    {
        if (fds[0].revents & POLLIN)
        {
            wl_display_read_events(display);
            wl_display_dispatch_pending(display);
        }
        else
        {
            wl_display_cancel_read(display);
        }

        if (fds[1].revents & POLLIN)
        {
            read_ret = read(_glfw.wl.timerfd, &repeats, sizeof(repeats));
            if (read_ret != 8)
                return;

            for (i = 0; i < repeats; ++i)
                _glfwInputKey(_glfw.wl.keyboardFocus, _glfw.wl.keyboardLastKey,
                              _glfw.wl.keyboardLastScancode, GLFW_REPEAT,
                              _glfw.wl.xkb.modifiers);
        }

        if (fds[2].revents & POLLIN)
        {
            read_ret = read(_glfw.wl.cursorTimerfd, &repeats, sizeof(repeats));
            if (read_ret != 8)
                return;

            incrementCursorImage(_glfw.wl.pointerFocus);
        }
    }
    else
    {
        wl_display_cancel_read(display);
    }
}

//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

int _glfwPlatformCreateWindow(_GLFWwindow* window,
                              const _GLFWwndconfig* wndconfig,
                              const _GLFWctxconfig* ctxconfig,
                              const _GLFWfbconfig* fbconfig)
{
    window->wl.transparent = fbconfig->transparent;

    if (!createSurface(window, wndconfig))
        return GLFW_FALSE;

    if (ctxconfig->client != GLFW_NO_API)
    {
        if (ctxconfig->source == GLFW_EGL_CONTEXT_API ||
            ctxconfig->source == GLFW_NATIVE_CONTEXT_API)
        {
            if (!_glfwInitEGL())
                return GLFW_FALSE;
            if (!_glfwCreateContextEGL(window, ctxconfig, fbconfig))
                return GLFW_FALSE;
        }
        else if (ctxconfig->source == GLFW_OSMESA_CONTEXT_API)
        {
            if (!_glfwInitOSMesa())
                return GLFW_FALSE;
            if (!_glfwCreateContextOSMesa(window, ctxconfig, fbconfig))
                return GLFW_FALSE;
        }
    }

    if (wndconfig->title)
        window->wl.title = _glfw_strdup(wndconfig->title);

    if (wndconfig->visible)
    {
        if (!createXdgSurface(window))
            return GLFW_FALSE;

        window->wl.visible = GLFW_TRUE;
    }
    else
    {
        window->wl.xdg.surface = NULL;
        window->wl.xdg.toplevel = NULL;
        window->wl.visible = GLFW_FALSE;
    }

    window->wl.currentCursor = NULL;

    window->wl.monitors = calloc(1, sizeof(_GLFWmonitor*));
    window->wl.monitorsCount = 0;
    window->wl.monitorsSize = 1;

    return GLFW_TRUE;
}

void _glfwPlatformDestroyWindow(_GLFWwindow* window)
{
    if (window == _glfw.wl.pointerFocus)
    {
        _glfw.wl.pointerFocus = NULL;
        _glfwInputCursorEnter(window, GLFW_FALSE);
    }
    if (window == _glfw.wl.keyboardFocus)
    {
        _glfw.wl.keyboardFocus = NULL;
        _glfwInputWindowFocus(window, GLFW_FALSE);
    }

    if (window->wl.idleInhibitor)
        zwp_idle_inhibitor_v1_destroy(window->wl.idleInhibitor);

    if (window->context.destroy)
        window->context.destroy(window);

    destroyDecorations(window);
    if (window->wl.xdg.decoration)
        zxdg_toplevel_decoration_v1_destroy(window->wl.xdg.decoration);

    if (window->wl.decorations.buffer)
        wl_buffer_destroy(window->wl.decorations.buffer);

    if (window->wl.native)
        wl_egl_window_destroy(window->wl.native);

    if (window->wl.xdg.toplevel)
        xdg_toplevel_destroy(window->wl.xdg.toplevel);

    if (window->wl.xdg.surface)
        xdg_surface_destroy(window->wl.xdg.surface);

    if (window->wl.surface)
        wl_surface_destroy(window->wl.surface);

    free(window->wl.title);
    free(window->wl.monitors);
}

void _glfwPlatformSetWindowTitle(_GLFWwindow* window, const char* title)
{
    if (window->wl.title)
        free(window->wl.title);
    window->wl.title = _glfw_strdup(title);
    if (window->wl.xdg.toplevel)
        xdg_toplevel_set_title(window->wl.xdg.toplevel, title);
}

void _glfwPlatformSetWindowIcon(_GLFWwindow* window,
                                int count, const GLFWimage* images)
{
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Setting window icon not supported");
}

void _glfwPlatformGetWindowPos(_GLFWwindow* window, int* xpos, int* ypos)
{
    // A Wayland client is not aware of its position, so just warn and leave it
    // as (0, 0)

    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Window position retrieval not supported");
}

void _glfwPlatformSetWindowPos(_GLFWwindow* window, int xpos, int ypos)
{
    // A Wayland client can not set its position, so just warn

    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Window position setting not supported");
}

void _glfwPlatformGetWindowSize(_GLFWwindow* window, int* width, int* height)
{
    if (width)
        *width = window->wl.width;
    if (height)
        *height = window->wl.height;
}

void _glfwPlatformSetWindowSize(_GLFWwindow* window, int width, int height)
{
    window->wl.width = width;
    window->wl.height = height;
    resizeWindow(window);
}

void _glfwPlatformSetWindowSizeLimits(_GLFWwindow* window,
                                      int minwidth, int minheight,
                                      int maxwidth, int maxheight)
{
    if (window->wl.xdg.toplevel)
    {
        if (minwidth == GLFW_DONT_CARE || minheight == GLFW_DONT_CARE)
            minwidth = minheight = 0;
        if (maxwidth == GLFW_DONT_CARE || maxheight == GLFW_DONT_CARE)
            maxwidth = maxheight = 0;
        xdg_toplevel_set_min_size(window->wl.xdg.toplevel, minwidth, minheight);
        xdg_toplevel_set_max_size(window->wl.xdg.toplevel, maxwidth, maxheight);
        wl_surface_commit(window->wl.surface);
    }
}

void _glfwPlatformSetWindowAspectRatio(_GLFWwindow* window,
                                       int numer, int denom)
{
    // TODO: find out how to trigger a resize.
    // The actual limits are checked in the xdg_toplevel::configure handler.
}

void _glfwPlatformGetFramebufferSize(_GLFWwindow* window,
                                     int* width, int* height)
{
    _glfwPlatformGetWindowSize(window, width, height);
    *width *= window->wl.scale;
    *height *= window->wl.scale;
}

void _glfwPlatformGetWindowFrameSize(_GLFWwindow* window,
                                     int* left, int* top,
                                     int* right, int* bottom)
{
    if (window->decorated && !window->monitor && !window->wl.decorations.serverSide)
    {
        if (top)
            *top = _GLFW_DECORATION_TOP;
        if (left)
            *left = _GLFW_DECORATION_WIDTH;
        if (right)
            *right = _GLFW_DECORATION_WIDTH;
        if (bottom)
            *bottom = _GLFW_DECORATION_WIDTH;
    }
}

void _glfwPlatformGetWindowContentScale(_GLFWwindow* window,
                                        float* xscale, float* yscale)
{
    if (xscale)
        *xscale = (float) window->wl.scale;
    if (yscale)
        *yscale = (float) window->wl.scale;
}

void _glfwPlatformIconifyWindow(_GLFWwindow* window)
{
    if (window->wl.xdg.toplevel)
        xdg_toplevel_set_minimized(window->wl.xdg.toplevel);
}

void _glfwPlatformRestoreWindow(_GLFWwindow* window)
{
    if (window->wl.xdg.toplevel)
    {
        if (window->monitor)
            xdg_toplevel_unset_fullscreen(window->wl.xdg.toplevel);
        if (window->wl.maximized)
            xdg_toplevel_unset_maximized(window->wl.xdg.toplevel);
        // There is no way to unset minimized, or even to know if we are
        // minimized, so there is nothing to do in this case.
    }
    _glfwInputWindowMonitor(window, NULL);
    window->wl.maximized = GLFW_FALSE;
}

void _glfwPlatformMaximizeWindow(_GLFWwindow* window)
{
    if (window->wl.xdg.toplevel)
    {
        xdg_toplevel_set_maximized(window->wl.xdg.toplevel);
    }
    window->wl.maximized = GLFW_TRUE;
}

void _glfwPlatformShowWindow(_GLFWwindow* window)
{
    if (!window->wl.visible)
    {
        createXdgSurface(window);
        window->wl.visible = GLFW_TRUE;
    }
}

void _glfwPlatformHideWindow(_GLFWwindow* window)
{
    if (window->wl.xdg.toplevel)
    {
        xdg_toplevel_destroy(window->wl.xdg.toplevel);
        xdg_surface_destroy(window->wl.xdg.surface);
        window->wl.xdg.toplevel = NULL;
        window->wl.xdg.surface = NULL;
    }
    window->wl.visible = GLFW_FALSE;
}

void _glfwPlatformRequestWindowAttention(_GLFWwindow* window)
{
    // TODO
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Window attention request not implemented yet");
}

void _glfwPlatformFocusWindow(_GLFWwindow* window)
{
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Focusing a window requires user interaction");
}

void _glfwPlatformSetWindowMonitor(_GLFWwindow* window,
                                   _GLFWmonitor* monitor,
                                   int xpos, int ypos,
                                   int width, int height,
                                   int refreshRate)
{
    if (monitor)
    {
        setFullscreen(window, monitor, refreshRate);
    }
    else
    {
        if (window->wl.xdg.toplevel)
            xdg_toplevel_unset_fullscreen(window->wl.xdg.toplevel);
        setIdleInhibitor(window, GLFW_FALSE);
        if (!_glfw.wl.decorationManager)
            createDecorations(window);
    }
    _glfwInputWindowMonitor(window, monitor);
}

int _glfwPlatformWindowFocused(_GLFWwindow* window)
{
    return _glfw.wl.keyboardFocus == window;
}

int _glfwPlatformWindowIconified(_GLFWwindow* window)
{
    // xdg-shell doesn’t give any way to request whether a surface is
    // iconified.
    return GLFW_FALSE;
}

int _glfwPlatformWindowVisible(_GLFWwindow* window)
{
    return window->wl.visible;
}

int _glfwPlatformWindowMaximized(_GLFWwindow* window)
{
    return window->wl.maximized;
}

int _glfwPlatformWindowHovered(_GLFWwindow* window)
{
    return window->wl.hovered;
}

int _glfwPlatformFramebufferTransparent(_GLFWwindow* window)
{
    return window->wl.transparent;
}

void _glfwPlatformSetWindowResizable(_GLFWwindow* window, GLFWbool enabled)
{
    // TODO
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Window attribute setting not implemented yet");
}

void _glfwPlatformSetWindowDecorated(_GLFWwindow* window, GLFWbool enabled)
{
    if (!window->monitor)
    {
        if (enabled)
            createDecorations(window);
        else
            destroyDecorations(window);
    }
}

void _glfwPlatformSetWindowFloating(_GLFWwindow* window, GLFWbool enabled)
{
    // TODO
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Window attribute setting not implemented yet");
}

float _glfwPlatformGetWindowOpacity(_GLFWwindow* window)
{
    return 1.f;
}

void _glfwPlatformSetWindowOpacity(_GLFWwindow* window, float opacity)
{
}

void _glfwPlatformSetRawMouseMotion(_GLFWwindow *window, GLFWbool enabled)
{
    // This is handled in relativePointerHandleRelativeMotion
}

GLFWbool _glfwPlatformRawMouseMotionSupported(void)
{
    return GLFW_TRUE;
}

void _glfwPlatformPollEvents(void)
{
    handleEvents(0);
}

void _glfwPlatformWaitEvents(void)
{
    handleEvents(-1);
}

void _glfwPlatformWaitEventsTimeout(double timeout)
{
    handleEvents((int) (timeout * 1e3));
}

void _glfwPlatformPostEmptyEvent(void)
{
    wl_display_sync(_glfw.wl.display);
}

void _glfwPlatformGetCursorPos(_GLFWwindow* window, double* xpos, double* ypos)
{
    if (xpos)
        *xpos = window->wl.cursorPosX;
    if (ypos)
        *ypos = window->wl.cursorPosY;
}

static GLFWbool isPointerLocked(_GLFWwindow* window);

void _glfwPlatformSetCursorPos(_GLFWwindow* window, double x, double y)
{
    if (isPointerLocked(window))
    {
        zwp_locked_pointer_v1_set_cursor_position_hint(
            window->wl.pointerLock.lockedPointer,
            wl_fixed_from_double(x), wl_fixed_from_double(y));
        wl_surface_commit(window->wl.surface);
    }
}

void _glfwPlatformSetCursorMode(_GLFWwindow* window, int mode)
{
    _glfwPlatformSetCursor(window, window->wl.currentCursor);
}

const char* _glfwPlatformGetScancodeName(int scancode)
{
    // TODO
    return NULL;
}

int _glfwPlatformGetKeyScancode(int key)
{
    return _glfw.wl.scancodes[key];
}

int _glfwPlatformCreateCursor(_GLFWcursor* cursor,
                              const GLFWimage* image,
                              int xhot, int yhot)
{
    cursor->wl.buffer = createShmBuffer(image);
    if (!cursor->wl.buffer)
        return GLFW_FALSE;

    cursor->wl.width = image->width;
    cursor->wl.height = image->height;
    cursor->wl.xhot = xhot;
    cursor->wl.yhot = yhot;
    return GLFW_TRUE;
}

int _glfwPlatformCreateStandardCursor(_GLFWcursor* cursor, int shape)
{
    const char* name = NULL;

    // Try the XDG names first
    if (shape == GLFW_ARROW_CURSOR)
        name = "default";
    else if (shape == GLFW_IBEAM_CURSOR)
        name = "text";
    else if (shape == GLFW_CROSSHAIR_CURSOR)
        name = "crosshair";
    else if (shape == GLFW_POINTING_HAND_CURSOR)
        name = "pointer";
    else if (shape == GLFW_RESIZE_EW_CURSOR)
        name = "ew-resize";
    else if (shape == GLFW_RESIZE_NS_CURSOR)
        name = "ns-resize";
    else if (shape == GLFW_RESIZE_NWSE_CURSOR)
        name = "nwse-resize";
    else if (shape == GLFW_RESIZE_NESW_CURSOR)
        name = "nesw-resize";
    else if (shape == GLFW_RESIZE_ALL_CURSOR)
        name = "all-scroll";
    else if (shape == GLFW_NOT_ALLOWED_CURSOR)
        name = "not-allowed";

    cursor->wl.cursor = wl_cursor_theme_get_cursor(_glfw.wl.cursorTheme, name);

    if (_glfw.wl.cursorThemeHiDPI)
    {
        cursor->wl.cursorHiDPI =
            wl_cursor_theme_get_cursor(_glfw.wl.cursorThemeHiDPI, name);
    }

    if (!cursor->wl.cursor)
    {
        // Fall back to the core X11 names
        if (shape == GLFW_ARROW_CURSOR)
            name = "left_ptr";
        else if (shape == GLFW_IBEAM_CURSOR)
            name = "xterm";
        else if (shape == GLFW_CROSSHAIR_CURSOR)
            name = "crosshair";
        else if (shape == GLFW_POINTING_HAND_CURSOR)
            name = "hand2";
        else if (shape == GLFW_RESIZE_EW_CURSOR)
            name = "sb_h_double_arrow";
        else if (shape == GLFW_RESIZE_NS_CURSOR)
            name = "sb_v_double_arrow";
        else if (shape == GLFW_RESIZE_ALL_CURSOR)
            name = "fleur";
        else
        {
            _glfwInputError(GLFW_CURSOR_UNAVAILABLE,
                            "Wayland: Standard cursor shape unavailable");
            return GLFW_FALSE;
        }

        cursor->wl.cursor = wl_cursor_theme_get_cursor(_glfw.wl.cursorTheme, name);
        if (!cursor->wl.cursor)
        {
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Failed to create standard cursor \"%s\"",
                            name);
            return GLFW_FALSE;
        }

        if (_glfw.wl.cursorThemeHiDPI)
        {
            if (!cursor->wl.cursorHiDPI)
            {
                cursor->wl.cursorHiDPI =
                    wl_cursor_theme_get_cursor(_glfw.wl.cursorThemeHiDPI, name);
            }
        }
    }

    return GLFW_TRUE;
}

void _glfwPlatformDestroyCursor(_GLFWcursor* cursor)
{
    // If it's a standard cursor we don't need to do anything here
    if (cursor->wl.cursor)
        return;

    if (cursor->wl.buffer)
        wl_buffer_destroy(cursor->wl.buffer);
}

static void relativePointerHandleRelativeMotion(void* data,
                                                struct zwp_relative_pointer_v1* pointer,
                                                uint32_t timeHi,
                                                uint32_t timeLo,
                                                wl_fixed_t dx,
                                                wl_fixed_t dy,
                                                wl_fixed_t dxUnaccel,
                                                wl_fixed_t dyUnaccel)
{
    _GLFWwindow* window = data;
    double xpos = window->virtualCursorPosX;
    double ypos = window->virtualCursorPosY;

    if (window->cursorMode != GLFW_CURSOR_DISABLED)
        return;

    if (window->rawMouseMotion)
    {
        xpos += wl_fixed_to_double(dxUnaccel);
        ypos += wl_fixed_to_double(dyUnaccel);
    }
    else
    {
        xpos += wl_fixed_to_double(dx);
        ypos += wl_fixed_to_double(dy);
    }

    _glfwInputCursorPos(window, xpos, ypos);
}

static const struct zwp_relative_pointer_v1_listener relativePointerListener = {
    relativePointerHandleRelativeMotion
};

static void lockedPointerHandleLocked(void* data,
                                      struct zwp_locked_pointer_v1* lockedPointer)
{
}

static void unlockPointer(_GLFWwindow* window)
{
    struct zwp_relative_pointer_v1* relativePointer =
        window->wl.pointerLock.relativePointer;
    struct zwp_locked_pointer_v1* lockedPointer =
        window->wl.pointerLock.lockedPointer;

    zwp_relative_pointer_v1_destroy(relativePointer);
    zwp_locked_pointer_v1_destroy(lockedPointer);

    window->wl.pointerLock.relativePointer = NULL;
    window->wl.pointerLock.lockedPointer = NULL;
}

static void lockPointer(_GLFWwindow* window);

static void lockedPointerHandleUnlocked(void* data,
                                        struct zwp_locked_pointer_v1* lockedPointer)
{
}

static const struct zwp_locked_pointer_v1_listener lockedPointerListener = {
    lockedPointerHandleLocked,
    lockedPointerHandleUnlocked
};

static void lockPointer(_GLFWwindow* window)
{
    struct zwp_relative_pointer_v1* relativePointer;
    struct zwp_locked_pointer_v1* lockedPointer;

    if (!_glfw.wl.relativePointerManager)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: no relative pointer manager");
        return;
    }

    relativePointer =
        zwp_relative_pointer_manager_v1_get_relative_pointer(
            _glfw.wl.relativePointerManager,
            _glfw.wl.pointer);
    zwp_relative_pointer_v1_add_listener(relativePointer,
                                         &relativePointerListener,
                                         window);

    lockedPointer =
        zwp_pointer_constraints_v1_lock_pointer(
            _glfw.wl.pointerConstraints,
            window->wl.surface,
            _glfw.wl.pointer,
            NULL,
            ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
    zwp_locked_pointer_v1_add_listener(lockedPointer,
                                       &lockedPointerListener,
                                       window);

    window->wl.pointerLock.relativePointer = relativePointer;
    window->wl.pointerLock.lockedPointer = lockedPointer;

    wl_pointer_set_cursor(_glfw.wl.pointer, _glfw.wl.serial,
                          NULL, 0, 0);
}

static GLFWbool isPointerLocked(_GLFWwindow* window)
{
    return window->wl.pointerLock.lockedPointer != NULL;
}

void _glfwPlatformSetCursor(_GLFWwindow* window, _GLFWcursor* cursor)
{
    struct wl_cursor* defaultCursor;
    struct wl_cursor* defaultCursorHiDPI = NULL;

    if (!_glfw.wl.pointer)
        return;

    window->wl.currentCursor = cursor;

    // If we're not in the correct window just save the cursor
    // the next time the pointer enters the window the cursor will change
    if (window != _glfw.wl.pointerFocus || window->wl.decorations.focus != mainWindow)
        return;

    // Unlock possible pointer lock if no longer disabled.
    if (window->cursorMode != GLFW_CURSOR_DISABLED && isPointerLocked(window))
        unlockPointer(window);

    if (window->cursorMode == GLFW_CURSOR_NORMAL)
    {
        if (cursor)
            setCursorImage(window, &cursor->wl);
        else
        {
            defaultCursor = wl_cursor_theme_get_cursor(_glfw.wl.cursorTheme,
                                                       "left_ptr");
            if (!defaultCursor)
            {
                _glfwInputError(GLFW_PLATFORM_ERROR,
                                "Wayland: Standard cursor not found");
                return;
            }
            if (_glfw.wl.cursorThemeHiDPI)
                defaultCursorHiDPI =
                    wl_cursor_theme_get_cursor(_glfw.wl.cursorThemeHiDPI,
                                               "left_ptr");
            _GLFWcursorWayland cursorWayland = {
                defaultCursor,
                defaultCursorHiDPI,
                NULL,
                0, 0,
                0, 0,
                0
            };
            setCursorImage(window, &cursorWayland);
        }
    }
    else if (window->cursorMode == GLFW_CURSOR_DISABLED)
    {
        if (!isPointerLocked(window))
            lockPointer(window);
    }
    else if (window->cursorMode == GLFW_CURSOR_HIDDEN)
    {
        wl_pointer_set_cursor(_glfw.wl.pointer, _glfw.wl.serial, NULL, 0, 0);
    }
}

static void dataSourceHandleTarget(void* data,
                                   struct wl_data_source* dataSource,
                                   const char* mimeType)
{
    if (_glfw.wl.dataSource != dataSource)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Unknown clipboard data source");
        return;
    }
}

static void dataSourceHandleSend(void* data,
                                 struct wl_data_source* dataSource,
                                 const char* mimeType,
                                 int fd)
{
    const char* string = _glfw.wl.clipboardSendString;
    size_t len = _glfw.wl.clipboardSendSize;
    int ret;

    if (_glfw.wl.dataSource != dataSource)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Unknown clipboard data source");
        return;
    }

    if (!string)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Copy requested from an invalid string");
        return;
    }

    if (strcmp(mimeType, "text/plain;charset=utf-8") != 0)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Wrong MIME type asked from clipboard");
        close(fd);
        return;
    }

    while (len > 0)
    {
        ret = write(fd, string, len);
        if (ret == -1 && errno == EINTR)
            continue;
        if (ret == -1)
        {
            // TODO: also report errno maybe.
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Error while writing the clipboard");
            close(fd);
            return;
        }
        len -= ret;
    }
    close(fd);
}

static void dataSourceHandleCancelled(void* data,
                                      struct wl_data_source* dataSource)
{
    wl_data_source_destroy(dataSource);

    if (_glfw.wl.dataSource != dataSource)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Unknown clipboard data source");
        return;
    }

    _glfw.wl.dataSource = NULL;
}

static const struct wl_data_source_listener dataSourceListener = {
    dataSourceHandleTarget,
    dataSourceHandleSend,
    dataSourceHandleCancelled,
};

void _glfwPlatformSetClipboardString(const char* string)
{
    if (_glfw.wl.dataSource)
    {
        wl_data_source_destroy(_glfw.wl.dataSource);
        _glfw.wl.dataSource = NULL;
    }

    if (_glfw.wl.clipboardSendString)
    {
        free(_glfw.wl.clipboardSendString);
        _glfw.wl.clipboardSendString = NULL;
    }

    _glfw.wl.clipboardSendString = strdup(string);
    if (!_glfw.wl.clipboardSendString)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Impossible to allocate clipboard string");
        return;
    }
    _glfw.wl.clipboardSendSize = strlen(string);
    _glfw.wl.dataSource =
        wl_data_device_manager_create_data_source(_glfw.wl.dataDeviceManager);
    if (!_glfw.wl.dataSource)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Impossible to create clipboard source");
        free(_glfw.wl.clipboardSendString);
        return;
    }
    wl_data_source_add_listener(_glfw.wl.dataSource,
                                &dataSourceListener,
                                NULL);
    wl_data_source_offer(_glfw.wl.dataSource, "text/plain;charset=utf-8");
    wl_data_device_set_selection(_glfw.wl.dataDevice,
                                 _glfw.wl.dataSource,
                                 _glfw.wl.serial);
}

static GLFWbool growClipboardString(void)
{
    char* clipboard = _glfw.wl.clipboardString;

    clipboard = realloc(clipboard, _glfw.wl.clipboardSize * 2);
    if (!clipboard)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Impossible to grow clipboard string");
        return GLFW_FALSE;
    }
    _glfw.wl.clipboardString = clipboard;
    _glfw.wl.clipboardSize = _glfw.wl.clipboardSize * 2;
    return GLFW_TRUE;
}

const char* _glfwPlatformGetClipboardString(void)
{
    int fds[2];
    int ret;
    size_t len = 0;

    if (!_glfw.wl.dataOffer)
    {
        _glfwInputError(GLFW_FORMAT_UNAVAILABLE,
                        "No clipboard data has been sent yet");
        return NULL;
    }

    ret = pipe2(fds, O_CLOEXEC);
    if (ret < 0)
    {
        // TODO: also report errno maybe?
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Impossible to create clipboard pipe fds");
        return NULL;
    }

    wl_data_offer_receive(_glfw.wl.dataOffer, "text/plain;charset=utf-8", fds[1]);
    close(fds[1]);

    // XXX: this is a huge hack, this function shouldn’t be synchronous!
    handleEvents(-1);

    while (1)
    {
        // Grow the clipboard if we need to paste something bigger, there is no
        // shrink operation yet.
        if (len + 4096 > _glfw.wl.clipboardSize)
        {
            if (!growClipboardString())
            {
                close(fds[0]);
                return NULL;
            }
        }

        // Then read from the fd to the clipboard, handling all known errors.
        ret = read(fds[0], _glfw.wl.clipboardString + len, 4096);
        if (ret == 0)
            break;
        if (ret == -1 && errno == EINTR)
            continue;
        if (ret == -1)
        {
            // TODO: also report errno maybe.
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Impossible to read from clipboard fd");
            close(fds[0]);
            return NULL;
        }
        len += ret;
    }
    close(fds[0]);
    if (len + 1 > _glfw.wl.clipboardSize)
    {
        if (!growClipboardString())
            return NULL;
    }
    _glfw.wl.clipboardString[len] = '\0';
    return _glfw.wl.clipboardString;
}

void _glfwPlatformGetRequiredInstanceExtensions(char** extensions)
{
    if (!_glfw.vk.KHR_surface || !_glfw.vk.KHR_wayland_surface)
        return;

    extensions[0] = "VK_KHR_surface";
    extensions[1] = "VK_KHR_wayland_surface";
}

int _glfwPlatformGetPhysicalDevicePresentationSupport(VkInstance instance,
                                                      VkPhysicalDevice device,
                                                      uint32_t queuefamily)
{
    PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR
        vkGetPhysicalDeviceWaylandPresentationSupportKHR =
        (PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
    if (!vkGetPhysicalDeviceWaylandPresentationSupportKHR)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "Wayland: Vulkan instance missing VK_KHR_wayland_surface extension");
        return VK_NULL_HANDLE;
    }

    return vkGetPhysicalDeviceWaylandPresentationSupportKHR(device,
                                                            queuefamily,
                                                            _glfw.wl.display);
}

VkResult _glfwPlatformCreateWindowSurface(VkInstance instance,
                                          _GLFWwindow* window,
                                          const VkAllocationCallbacks* allocator,
                                          VkSurfaceKHR* surface)
{
    VkResult err;
    VkWaylandSurfaceCreateInfoKHR sci;
    PFN_vkCreateWaylandSurfaceKHR vkCreateWaylandSurfaceKHR;

    vkCreateWaylandSurfaceKHR = (PFN_vkCreateWaylandSurfaceKHR)
        vkGetInstanceProcAddr(instance, "vkCreateWaylandSurfaceKHR");
    if (!vkCreateWaylandSurfaceKHR)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "Wayland: Vulkan instance missing VK_KHR_wayland_surface extension");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
    sci.display = _glfw.wl.display;
    sci.surface = window->wl.surface;

    err = vkCreateWaylandSurfaceKHR(instance, &sci, allocator, surface);
    if (err)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to create Vulkan surface: %s",
                        _glfwGetVulkanResultString(err));
    }

    return err;
}


//////////////////////////////////////////////////////////////////////////
//////                        GLFW native API                       //////
//////////////////////////////////////////////////////////////////////////

GLFWAPI struct wl_display* glfwGetWaylandDisplay(void)
{
    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    return _glfw.wl.display;
}

GLFWAPI struct wl_surface* glfwGetWaylandWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    return window->wl.surface;
}

