/*
exposes C++ functions to the outside world
manages entity memory directly
calls js/csharp logic each frame
*/

#include "memory_layout.h" //imports the generated memory layout contract
#include <emscripten.h>
#include <cstdint> //for fixed width integers
#include <cstdio>  //printf logging


//create a C++ callable wrapper around JS - This will be callable in C++ like a noormal function
EM_JS(void, js_bridge_tick, (int offset, int count, float dt), {
    //if it exists as a global function in js, call it
    if (typeof window.godot_js_bridge_tick === "function") {
        window.godot_js_bridge_tick(offset, count, dt);
    }
});


//we use this to avoid name mangling
extern "C" {

    //returns a pointer to the entity array -- it treats enitites offset as a memory adress pointing to godot wasm linear memory
static inline Entity *entities() {
    return reinterpret_cast<Entity *>(ENTITIES_OFFSET);
}

//pseudo randm generator that mutates the state. Its used in initializing the starting point of the entities - this is fast and simple using xorshift RNG
static inline std::uint32_t next_u32(std::uint32_t &state) {
    state ^= state << 13; //shift left by 13 bits
    state ^= state >> 17; //right by 17
    state ^= state << 5; //left by 5
    return state;
}


//called once at startup
void init_entities() {
    // This may run too early and get overwritten but that is fine

    Entity *arr = entities(); //gets poointer to entity array
    std::uint32_t rng = 0x12345678u; //seed
    for (std::uint32_t i = 0; i < ENTITY_COUNT; ++i) { //iterates all entities
        arr[i].x  = static_cast<float>(next_u32(rng) % 1024u);
        arr[i].y  = static_cast<float>(next_u32(rng) % 768u);
        arr[i].vx = static_cast<float>(static_cast<int>(next_u32(rng) % 301u) - 150);
        arr[i].vy = static_cast<float>(static_cast<int>(next_u32(rng) % 301u) - 150);
    }
}

//runs every frame - calls JS, passing where entities are, how many and frametime - it lets JS/C# update entities
void tick_entities(float dt) {
    js_bridge_tick(ENTITIES_OFFSET, ENTITY_COUNT, dt);
        
}

}