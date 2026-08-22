#include "godot_3ds.h"

#include "3ds.h"

int godot_3ds_main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    if (!Godot3DS::initialize()) {
        return 1;
    }

    while (!Godot3DS::should_quit()) {
        Godot3DS::process_frame();
    }

    Godot3DS::shutdown();

    return 0;
}
