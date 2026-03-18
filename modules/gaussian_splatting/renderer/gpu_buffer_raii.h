#ifndef GPU_BUFFER_RAII_H
#define GPU_BUFFER_RAII_H

#include "servers/rendering/rendering_device.h"

class GPUBuffer {
    RID rid;
    RenderingDevice *device = nullptr;

public:
    GPUBuffer() = default;
    GPUBuffer(RenderingDevice *p_device, const RID &p_rid) : rid(p_rid), device(p_device) {}

    ~GPUBuffer() {
        if (rid.is_valid() && device) {
            device->free(rid);
        }
    }

    GPUBuffer(GPUBuffer &&other) noexcept : rid(other.rid), device(other.device) {
        other.rid = RID();
        other.device = nullptr;
    }

    GPUBuffer &operator=(GPUBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            rid = other.rid;
            device = other.device;
            other.rid = RID();
            other.device = nullptr;
        }
        return *this;
    }

    GPUBuffer(const GPUBuffer &) = delete;
    GPUBuffer &operator=(const GPUBuffer &) = delete;

    RID get() const { return rid; }
    bool is_valid() const { return rid.is_valid(); }

    void reset() {
        if (rid.is_valid() && device) {
            device->free(rid);
        }
        rid = RID();
        device = nullptr;
    }

    RID release() {
        RID result = rid;
        rid = RID();
        device = nullptr;
        return result;
    }
};

#endif // GPU_BUFFER_RAII_H
