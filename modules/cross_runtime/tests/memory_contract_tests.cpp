/*
This tests the memory layout for correctness.
It is crucial that you run this test after the full generation
*/
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "../memory_layout.h"

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "CHECK failed: " << #expr << "\n"; \
        return 1; \
    } \
} while (0)

static int test_layout_contract() {
    static_assert(sizeof(float) == 4, "schema assumes 32-bit float");
    static_assert(sizeof(Entity) == ENTITY_STRIDE, "Entity size must match stride");

    static_assert(offsetof(Entity, x)  == ENTITY_FIELD_X_OFFSET,  "x offset mismatch");
    static_assert(offsetof(Entity, y)  == ENTITY_FIELD_Y_OFFSET,  "y offset mismatch");
    static_assert(offsetof(Entity, vx) == ENTITY_FIELD_VX_OFFSET, "vx offset mismatch");
    static_assert(offsetof(Entity, vy) == ENTITY_FIELD_VY_OFFSET, "vy offset mismatch");

    static_assert(ENTITIES_OFFSET % alignof(Entity) == 0, "entities must be aligned");
    static_assert(WORKER_READY_OFFSET % alignof(std::uint32_t) == 0, "worker flag must be 32-bit aligned");
    static_assert(WORKER_READY_OFFSET >= ENTITIES_OFFSET + ENTITY_COUNT * ENTITY_STRIDE, "worker flag overlaps entity storage");
    return 0;
}

static int test_field_roundtrip() {
    alignas(Entity) std::uint8_t raw[sizeof(Entity)] = {};
    auto* e = reinterpret_cast<Entity*>(raw);

    e->x  = 1.25f;
    e->y  = 2.5f;
    e->vx = 3.75f;
    e->vy = 4.5f;

    Entity copy{};
    std::memcpy(&copy, raw, sizeof(copy));

    CHECK(copy.x  == e->x);
    CHECK(copy.y  == e->y);
    CHECK(copy.vx == e->vx);
    CHECK(copy.vy == e->vy);
    return 0;
}

static int test_index_math() {
    for (std::uint32_t i = 0; i < ENTITY_COUNT; ++i) {
        const std::uint32_t expected = ENTITIES_OFFSET + i * ENTITY_STRIDE;
        const std::uint32_t actual    = ENTITIES_OFFSET + i * static_cast<std::uint32_t>(sizeof(Entity));
        CHECK(expected == actual);
    }
    return 0;
}

int main() {
    if (int r = test_layout_contract()) return r;
    if (int r = test_field_roundtrip()) return r;
    if (int r = test_index_math()) return r;
    std::cout << "memory contract tests passed\n";
    return 0;
}