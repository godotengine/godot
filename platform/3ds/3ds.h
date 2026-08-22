#ifndef GODOT_3DS_H
#define GODOT_3DS_H

#include <3ds.h>

namespace Godot3DS {

bool initialize();
void shutdown();
bool should_quit();
void process_frame();

} // namespace Godot3DS

#endif
