#include <3ds.h>

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;

	gfxInitDefault();

	consoleInit(GFX_TOP, nullptr);

	printf("Godot 4.7 3DS\n");
	printf("Export template funcionando!\n");

	while (aptMainLoop()) {
		hidScanInput();

		u32 kDown = hidKeysDown();

		if (kDown & KEY_START) {
			break;
		}

		gfxFlushBuffers();
		gfxSwapBuffers();

		gspWaitForVBlank();
	}

	gfxExit();

	return 0;
}
