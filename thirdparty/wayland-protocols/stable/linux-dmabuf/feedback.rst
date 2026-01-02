.. Copyright 2021 Simon Ser

.. contents::


linux-dmabuf feedback introduction
==================================

linux-dmabuf feedback allows compositors and clients to negotiate optimal buffer
allocation parameters. This document will assume that the compositor is using a
rendering API such as OpenGL or Vulkan and KMS as the presentation API: even if
linux-dmabuf feedback isn't restricted to this use-case, it's the most common.

linux-dmabuf feedback introduces the following concepts:

1. A main device. This is the render device that the compositor is using to
   perform composition. Compositors should always be able to display a buffer
   submitted by a client, so this device can be used as a fallback in case none
   of the more optimized code-paths work. Clients should allocate buffers such
   that they can be imported and textured from the main device.

2. One or more tranches. Each tranche consists of a target device, allocation
   flags and a set of format/modifier pairs. A tranche can be seen as a set of
   formats/modifier pairs that are compatible with the target device.

   A tranche can have the ``scanout`` flag. It means that the target device is
   a KMS device, and that buffers allocated with one of the format/modifier
   pairs in the tranche are eligible for direct scanout.

   Clients should use the tranches in order to allocate buffers with the most
   appropriate format/modifier and also to avoid allocating in private device
   memory when cross-device operations are going to happen.

linux-dmabuf feedback implementation notes
==========================================

This section contains recommendations for client and compositor implementations.

For clients
-----------

Clients are expected to either pick a fixed DRM format beforehand, or
perform the following steps repeatedly until they find a suitable format.

Basic clients may only support static buffer allocation on startup. These
clients should do the following:

1. Send a ``get_default_feedback`` request to get global feedback.
2. Select the device indicated by ``main_device`` for allocation.
3. For each tranche:

   1. If ``tranche_target_device`` doesn't match the allocation device, ignore
      the tranche.
   2. Accumulate allocation flags from ``tranche_flags``.
   3. Accumulate format/modifier pairs received via ``tranche_formats`` in a
      list.
   4. When the ``tranche_done`` event is received, try to allocate the buffer
      with the accumulated list of modifiers and allocation flags. If that
      fails, proceed with the next tranche. If that succeeds, stop the loop.

4. Destroy the feedback object.

Tranches are ordered by preference: the more optimized tranches come first. As
such, clients should use the first tranche that happens to work.

Some clients may have already selected the device they want to use beforehand.
These clients can ignore the ``main_device`` event, and ignore tranches whose
``tranche_target_device`` doesn't match the selected device. Such clients need
to be prepared for the ``wp_linux_buffer_params.create`` request to potentially
fail.

If the client allocates a buffer without specifying explicit modifiers on a
device different from the one indicated by ``main_device``, then the client
must force a linear layout.

Some clients might support re-negotiating the buffer format/modifier on the
fly. These clients should send a ``get_surface_feedback`` request and keep the
feedback object alive after the initial allocation. Each time a new set of
feedback parameters is received (ended by the ``done`` event), they should
perform the same steps as basic clients described above. They should detect
when the optimal allocation parameters didn't change (same
format/modifier/flags) to avoid needlessly re-allocating their buffers.

Some clients might additionally support switching the device used for
allocations on the fly. Such clients should send a ``get_surface_feedback``
request. For each tranche, select the device indicated by
``tranche_target_device`` for allocation. Accumulate allocation flags (received
via ``tranche_flags``) and format/modifier pairs (received via
``tranche_formats``) as usual. When the ``tranche_done`` event is received, try
to allocate the buffer with the accumulated list of modifiers and the
allocation flags. Try to import the resulting buffer by sending a
``wp_linux_buffer_params.create`` request (this might fail). Repeat with each
tranche until an allocation and import succeeds. Each time a new set of
feedback parameters is received, they should perform these steps again. They
should detect when the optimal allocation parameters didn't change (same
device/format/modifier/flags) to avoid needlessly re-allocating their buffers.

For compositors
---------------

Basic compositors may only support texturing the DMA-BUFs via a rendering API
such as OpenGL or Vulkan. Such compositors can send a single tranche as a reply
to both ``get_default_feedback`` and ``get_surface_feedback``. Set the
``main_device`` to the rendering device. Send the tranche with
``tranche_target_device`` set to the rendering device and all of the DRM
format/modifier pairs supported by the rendering API. Do not set the
``scanout`` flag in the ``tranche_flags`` event.

Some compositors may support direct scan-out for full-screen surfaces. These
compositors can re-send the feedback parameters when a surface becomes
full-screen or leaves full-screen mode if the client has used the
``get_surface_feedback`` request. The non-full-screen feedback parameters are
the same as basic compositors described above. The full-screen feedback
parameters have two tranches: one with the format/modifier pairs supported by
the KMS plane, with the ``scanout`` flag set in the ``tranche_flags`` event and
with ``tranche_target_device`` set to the KMS scan-out device; the other with
the rest of the format/modifier pairs (supported for texturing, but not for
scan-out), without the ``scanout`` flag set in the ``tranche_flags`` event, and
with the ``tranche_target_device`` set to the rendering device.

Some compositors may support direct scan-out for all surfaces. These
compositors can send two tranches for surfaces that become candidates for
direct scan-out, similarly to compositors supporting direct scan-out for
fullscreen surfaces. When a surface stops being a candidate for direct
scan-out, compositors should re-send the feedback parameters optimized for
texturing only.  The way candidates for direct scan-out are selected is
compositor policy, a possible implementation is to select as many surfaces as
there are available hardware planes, starting from surfaces closer to the eye.

Some compositors may support multiple devices at the same time. If the
compositor supports rendering with a fixed device and direct scan-out on a
secondary device, it may send a separate tranche for surfaces displayed on
the secondary device that are candidates for direct scan-out. The
``tranche_target_device`` for this tranche will be the secondary device and
will not match the ``main_device``.

Some compositors may support switching their rendering device at runtime or
changing their rendering device depending on the surface. When the rendering
device changes for a surface, such compositors may re-send the feedback
parameters with a different ``main_device``. However there is a risk that
clients don't support switching their device at runtime and continue using the
previous device. For this reason, compositors should always have a fallback
rendering device that they initially send as ``main_device``, such that these
clients use said fallback device.

Compositors should not change the ``main_device`` on-the-fly when explicit
modifiers are not supported, because there's a risk of importing buffers
with an implicit non-linear modifier as a linear buffer, resulting in
misinterpreted buffer contents.

Compositors should not send feedback parameters if they don't have a fallback
path. For instance, compositors shouldn't send a format/modifier supported for
direct scan-out but not supported by the rendering API for texturing.

Compositors can decide to use multiple tranches to describe the allocation
parameters optimized for texturing. For example, if there are formats which
have a fast texturing path and formats which have a slower texturing path, the
compositor can decide to expose two separate tranches.

Compositors can decide to use intermediate tranches to describe code-paths
slower than direct scan-out but faster than texturing. For instance, a
compositor could insert an intermediate tranche if it's possible to use a
mem2mem device to convert buffers to be able to use scan-out.

``dev_t`` encoding
==================

The protocol carries ``dev_t`` values on the wire using arrays. A compositor
written in C can encode the values as follows:

.. code-block:: c

    struct stat drm_node_stat;
    struct wl_array dev_array = {
        .size = sizeof(drm_node_stat.st_rdev),
        .data = &drm_node_stat.st_rdev,
    };

A client can decode the values as follows:

.. code-block:: c

    dev_t dev;
    assert(dev_array->size == sizeof(dev));
    memcpy(&dev, dev_array->data, sizeof(dev));

Because two DRM nodes can refer to the same DRM device while having different
``dev_t`` values, clients should use ``drmDevicesEqual`` to compare two
devices.

``format_table`` encoding
=========================

The ``format_table`` event carries a file descriptor containing a list of
format + modifier pairs. The list is an array of pairs which can be accessed
with this C structure definition:

.. code-block:: c

    struct dmabuf_format_modifier {
        uint32_t format;
        uint32_t pad; /* unused */
        uint64_t modifier;
    };

Integration with other APIs
===========================

- libdrm: ``drmGetDeviceFromDevId`` returns a ``drmDevice`` from a device ID.
- EGL: the `EGL_EXT_device_drm_render_node`_ extension may be used to query the
  DRM device render node used by a given EGL display. When unavailable, the
  older `EGL_EXT_device_drm`_ extension may be used as a fallback.
- Vulkan: the `VK_EXT_physical_device_drm`_ extension may be used to query the
  DRM device used by a given ``VkPhysicalDevice``.

.. _EGL_EXT_device_drm: https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_device_drm.txt
.. _EGL_EXT_device_drm_render_node: https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_device_drm_render_node.txt
.. _VK_EXT_physical_device_drm: https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_physical_device_drm.html
