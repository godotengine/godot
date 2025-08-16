#[compute]

#version 450

#VERSION_DEFINES

#define BLOCK_SIZE 8

#include "_included.glsl"

void main() {
	uint t = BLOCK_SIZE + 1;
}
