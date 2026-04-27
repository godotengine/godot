/*
(Critical)
This is the memory-layout contract;
It tells C++ code where data lives in memory and how it is shaped
*/
#pragma once

#include <cstdint> //imports fixed width integer types


//this was automatically generated from the declared schema in tools/schema.json . Do not edit directly
constexpr std::uint32_t ENTITY_COUNT    = 1000u;
constexpr std::uint32_t ENTITIES_OFFSET = 0x1000u;
constexpr std::uint32_t ENTITY_STRIDE   = 16u; //distance in bytes between consecutive entities


//stores the snchronization flag used to synchronise the simulation worker in host.html with the renderer 
// it lets the renderer know that the dotnet worker has finished initializing
constexpr std::uint32_t WORKER_READY_OFFSET = 0x8000u;


struct Entity {
    float x;
    float y;
    float vx;
    float vy;
};

// Byte offsets for each field
constexpr std::uint32_t ENTITY_FIELD_X_OFFSET = 0u;
constexpr std::uint32_t ENTITY_FIELD_Y_OFFSET = 4u;
constexpr std::uint32_t ENTITY_FIELD_VX_OFFSET = 8u;
constexpr std::uint32_t ENTITY_FIELD_VY_OFFSET = 12u;

// Field order:
//   0 : x
//   1 : y
//   2 : vx
//   3 : vy