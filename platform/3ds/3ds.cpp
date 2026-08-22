#include "3ds.h"

namespace Godot3DS {

bool initialize() {
    gfxInitDefault();
    consoleInit(GFX_TOP, nullptr);

    return true;
}

void shutdown() {
    gfxExit();
}

bool should_quit() {
    hidScanInput();

    const u32 kDown = hidKeysDown();

    return (kDown & KEY_START) != 0;
}

void process_frame() {
    hidScanInput();

    gfxFlushBuffers();
    gfxSwapBuffers();

    gspWaitForVBlank();
}

} // namespace Godot3DS
