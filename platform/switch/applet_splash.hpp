
#include <switch_wrapper.h>

#define WIDTH 1280
#define HEIGHT 720

// applet_splash.rgba.gz (available in romfs)
// will be display instead of the actual godot app if the app is not
// started properly from the switch
// see ref ()
int display_applet_splash() {
	romfsInit();

	NWindow *win = nwindowGetDefault();
	Framebuffer fb;
	framebufferCreate(&fb, win, WIDTH, HEIGHT, PIXEL_FORMAT_RGBA_8888, 1);
	framebufferMakeLinear(&fb);

	u32 stride;
	u32 *framebuf = (u32 *)framebufferBegin(&fb, &stride);

	FILE *splash = fopen("romfs:/applet_splash.rgba.gz", "rb");

	fseek(splash, 0, SEEK_END);
	size_t splash_size = ftell(splash);

	u8 *compressed_splash = (u8 *)malloc(splash_size);
	memset(compressed_splash, 0, splash_size);
	fseek(splash, 0, SEEK_SET);

	fread(compressed_splash, 1, splash_size, splash);
	fclose(splash);

	memset(framebuf, 0, stride * HEIGHT);

	struct z_stream_s stream;
	memset(&stream, 0, sizeof(stream));
	stream.zalloc = NULL;

	stream.zfree = NULL;
	stream.next_in = compressed_splash;
	stream.avail_in = splash_size;
	stream.next_out = (u8 *)framebuf;
	stream.avail_out = stride * HEIGHT;

	inflateInit2(&stream, 16 + MAX_WBITS);
	inflate(&stream, 0);
	inflateEnd(&stream);

	framebufferEnd(&fb);

	// set up input
	PadState pad;
	padConfigureInput(1, HidNpadStyleSet_NpadStandard);
	padInitializeDefault(&pad);
	while (appletMainLoop()) {
		padUpdate(&pad);
		if (padGetButtonsDown(&pad) != 0) {
			break;
		}
	}

	framebufferClose(&fb);
	romfsExit();
	return 0;
}
