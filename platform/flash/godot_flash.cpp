
#include "main/main.h"
#include "os_flash.h"
#include <AS3/AS3.h>
#include <Flash++.h>

#include <stdio.h>

using namespace AS3::ui;

static int frames = 0;

static OSFlash* os_flash = NULL;
static uint64_t last_time = 0;

static var frame(void* data, var args) {

	if (frames < 3) {
		uint64_t now = os_flash->get_ticks_usec();
		printf("enter frame %i, %f\n", frames, (now -  last_time) / 1000000.0);
		last_time = now;
	};
	switch (frames) {

	case 0: {
		char* argv[] = {"-test", "gui", NULL};
		Main::setup("", 2, argv, false);
		++frames;
	} break;
	case 1:
		Main::setup2();
		++frames;
		break;
	case 2:
		Main::start();
		++frames;
		break;
	default:
		os_flash->iterate();
		inline_as3("import GLS3D.*;\n"
					"GLAPI.instance.context.present();\n"
				   : :
		);
		/*
		flash::display::Stage stage = internal::get_Stage();
		flash::display::Stage3D s3d = var(var(stage->stage3Ds)[0]);
		flash::display3D::Context3D ctx3d = s3d->context3D;
		ctx3d->present();
		*/
		break;
	};

	return internal::_undefined;
};

static var context_error(void *arg, var as3Args) {

	printf("stage 3d error!\n");

	return internal::_undefined;
};

int main(int argc, char* argv[]) {

	printf("godot flash\n");
	os_flash = new OSFlash;
	printf("os\n");

	last_time = os_flash->get_ticks_usec();

	flash::display::Stage stage = internal::get_Stage();
	stage->scaleMode = flash::display::StageScaleMode::NO_SCALE;
	stage->align = flash::display::StageAlign::TOP_LEFT;
	stage->frameRate = 60;

	stage->addEventListener(flash::events::Event::ENTER_FRAME, Function::_new(frame, NULL));

	AS3_GoAsync();
};

