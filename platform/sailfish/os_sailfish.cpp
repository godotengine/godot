#include "os_sailfish.h"
#include "os/keyboard.h"
#include "key_mapping_sailfish.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/physics_2d/physics_2d_server_wrap_mt.h"
#include "main/main.h"

#include <sailfishapp.h>
#include <QScreen>
#include <QTouchEvent>


static void on_audio_resource_acquired(audioresource_t*, bool, void*);

OS_Sailfish::OS_Sailfish() {

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

	if (AudioDriverManager::get_driver_count() == 0) {
		WARN_PRINT("No sound driver found... Defaulting to dummy driver");
		AudioDriverManager::add_driver(&driver_dummy);
	}

	last_mouse_position_valid = false;
	is_audio_resource_acquired = false;
}

String OS_Sailfish::get_name() {
	return "Sailfish OS";
}

int OS_Sailfish::get_video_driver_count() const {
	return 1;
}

const char* OS_Sailfish::get_video_driver_name(int p_driver) const {
	return "GLES3";
}

OS::VideoMode OS_Sailfish::get_default_video_mode() const {
	return VideoMode(
		window->screen()->size().width(),
		window->screen()->size().height(),
		true,	// fullscreen
		false,	// resizable
		true	// borderless
	);
}

int OS_Sailfish::get_audio_driver_count() const {
	return AudioDriverManager::get_driver_count();
}

const char* OS_Sailfish::get_audio_driver_name(int p_driver) const {
	AudioDriver* driver = AudioDriverManager::get_driver(p_driver);
	ERR_FAIL_COND_V(!driver, "");
	return AudioDriverManager::get_driver(p_driver)->get_name();
}

void OS_Sailfish::initialize(const VideoMode& p_desired, int p_video_driver, int p_audio_driver) {
	main_loop = NULL;
	current_video_mode = p_desired;

	window->installEventFilter(this);
	window->showFullScreen();

#if defined(OPENGL_ENABLED)
	// wait until we have an OpenGL context
	while (window->context() == NULL) {
		application->processEvents();
	}

	openglContext = window->context();

	RasterizerGLES3::register_config();
	RasterizerGLES3::make_current();
#endif

	visual_server = memnew(VisualServerRaster());
	ERR_FAIL_COND(!visual_server);
	visual_server->init();

	AudioDriverManager::get_driver(p_audio_driver)->set_singleton();

	audio_resource = audioresource_init(
		AUDIO_RESOURCE_GAME,
		on_audio_resource_acquired,
		//AudioDriverManager::get_driver(p_audio_driver)
		this
	);
	audioresource_acquire(audio_resource);

	while (!this->is_audio_resource_acquired) {
		application->processEvents();
	}

	//if (AudioDriverManager::get_driver(p_audio_driver)->init() != OK) {
		//ERR_PRINT("Initializing audio failed.");
	//}

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = Physics2DServerWrapMT::init_server<Physics2DServerSW>();
	physics_2d_server->init();

	input = memnew(InputDefault);

	application->processEvents();
}

void OS_Sailfish::finalize() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;

	for (int i = 0; i < get_audio_driver_count(); i++) {
		AudioDriverManager::get_driver(i)->finish();
	}

	audioresource_release(audio_resource);
	audioresource_free(audio_resource);

	memdelete(input);

	visual_server->finish();
	memdelete(visual_server);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);
}

void OS_Sailfish::set_main_loop(MainLoop* p_main_loop) {
	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

/* Processes events for the main view. */
bool OS_Sailfish::eventFilter(QObject* object, QEvent* event) {
	switch (event->type()) {
		case QEvent::Expose:
		case QEvent::Paint:
			Main::force_redraw();
			break;

		case QEvent::KeyPress:
		case QEvent::KeyRelease: {
			QKeyEvent* qkey_event = static_cast<QKeyEvent*>(event);

			Ref<InputEventKey> key_event;
			key_event.instance();

			get_key_modifier_state(qkey_event->modifiers(), key_event);
			key_event->set_pressed(event->type() == QEvent::KeyPress);
			key_event->set_scancode(KeyMappingSailfish::get_keycode(qkey_event->key()));
			key_event->set_unicode(qkey_event->text().unicode()->unicode());
			key_event->set_echo(qkey_event->isAutoRepeat());

			//make it consistent across platforms.
			if (key_event->get_scancode() == KEY_BACKTAB) {
				key_event->set_scancode(KEY_TAB);
				key_event->set_shift(true);
			}

			//don't set mod state if modifier keys are released by themselves
			//else event.is_action() will not work correctly here
			if (!key_event->is_pressed()) {
				if (key_event->get_scancode() == KEY_SHIFT)
					key_event->set_shift(false);
				else if (key_event->get_scancode() == KEY_CONTROL)
					key_event->set_control(false);
				else if (key_event->get_scancode() == KEY_ALT)
					key_event->set_alt(false);
				else if (key_event->get_scancode() == KEY_META)
					key_event->set_metakey(false);
			}

			input->parse_input_event(key_event);
			break;
		}

		//case QEvent::TouchBegin:
			//break;
		//case QEvent::TouchUpdate:
			//break;
		//case QEvent::TouchEnd:
			//break;

		case QEvent::MouseButtonDblClick:
		case QEvent::MouseButtonPress:
		case QEvent::MouseButtonRelease: {
			QMouseEvent* qmouse_event = static_cast<QMouseEvent*>(event);

			Ref<InputEventMouseButton> mouse_event;
			mouse_event.instance();

			get_key_modifier_state(QGuiApplication::keyboardModifiers(), mouse_event);
			mouse_event->set_button_mask(get_mouse_button_state(qmouse_event->buttons()));
			mouse_event->set_position(Vector2(qmouse_event->x(), qmouse_event->y()));
			mouse_event->set_global_position(mouse_event->get_position());
			mouse_event->set_pressed(event->type() != QEvent::MouseButtonRelease);
			mouse_event->set_doubleclick(event->type() == QEvent::MouseButtonDblClick);

			switch (qmouse_event->button()) {
				default:
				case Qt::LeftButton:
					mouse_event->set_button_index(1);
					break;

				case Qt::RightButton:
					mouse_event->set_button_index(2);
					break;

				case Qt::MidButton:
					mouse_event->set_button_index(3);
					break;
			}

			input->parse_input_event(mouse_event);
			break;
		}

		case QEvent::MouseMove: {
			QMouseEvent* qmouse_event = static_cast<QMouseEvent*>(event);

			Point2i position(qmouse_event->x(), qmouse_event->y());
			Point2i global_position(qmouse_event->globalX(), qmouse_event->globalY());

			if (!last_mouse_position_valid) {
				last_mouse_position = position;
				last_mouse_position_valid = true;
			}

			Point2i relative = position - last_mouse_position;

			Ref<InputEventMouseMotion> motion_event;
			motion_event.instance();

			get_key_modifier_state(QGuiApplication::keyboardModifiers(), motion_event);
			motion_event->set_button_mask(get_mouse_button_state(qmouse_event->buttons()));
			motion_event->set_position(position);
			input->set_mouse_position(position);
			motion_event->set_global_position(global_position);
			motion_event->set_speed(input->get_last_mouse_speed());
			motion_event->set_relative(relative);

			last_mouse_position = position;

			input->parse_input_event(motion_event);
			break;
		}

		case QEvent::Close:
			qDebug("close event");
			main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
			break;

		default:
			break;
	}

	qDebug("event %d", event->type());

	return false;
}

void OS_Sailfish::set_cursor_shape(CursorShape p_shape) {
	// TODO: implement
}

Point2 OS_Sailfish::get_mouse_position() const {
	// DODO: implement
	return Point2();
}

int OS_Sailfish::get_mouse_button_state() const {
	return last_button_state;
}

void OS_Sailfish::set_window_title(const String& p_title) {
	// TODO: implement
}

MainLoop* OS_Sailfish::get_main_loop() const {
	return main_loop;
}

void OS_Sailfish::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;
}

int OS_Sailfish::get_mouse_button_state(Qt::MouseButtons p_buttons) {
	int state = 0;

	if (p_buttons.testFlag(Qt::LeftButton)) {
		state |= 1 << 0;
	}

	if (p_buttons.testFlag(Qt::RightButton)) {
		state |= 1 << 1;
	}

	if (p_buttons.testFlag(Qt::MidButton)) {
		state |= 1 << 2;
	}

	last_button_state = state;
	return state;
}

void OS_Sailfish::get_key_modifier_state(Qt::KeyboardModifiers p_modifiers, Ref<InputEventWithModifiers> state) {
	state->set_shift(p_modifiers.testFlag(Qt::ShiftModifier));
	state->set_control(p_modifiers.testFlag(Qt::ControlModifier));
	state->set_alt(p_modifiers.testFlag(Qt::AltModifier));
	state->set_metakey(p_modifiers.testFlag(Qt::MetaModifier));
}

bool OS_Sailfish::can_draw() const {
	return window->isVisible();
}

void OS_Sailfish::release_rendering_thread() {
	window->doneCurrent();
}

void OS_Sailfish::make_rendering_thread() {
	window->makeCurrent();
}

void OS_Sailfish::swap_buffers() {
	openglContext->swapBuffers(window);
}

void OS_Sailfish::set_video_mode(const VideoMode& p_video_mode, int p_screen) {

}

OS::VideoMode OS_Sailfish::get_video_mode(int p_screen) const {
	return current_video_mode;
}

void OS_Sailfish::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
	ERR_PRINT("get_fullscreen_mode_list() NOT IMPLEMENTED");
}

int OS_Sailfish::get_screen_count() const {
	// TODO: implement OS_Sailfish::get_screen_count()
	return 1;
}

int OS_Sailfish::get_current_screen() const {
	// TODO: implement OS_Sailfish::get_current_screen
	return 0;
}

Size2 OS_Sailfish::get_window_size() const {
	return Size2i(window->width(), window->height());
}

void OS_Sailfish::run() {
	if (!main_loop) {
		return;
	}

	main_loop->init();

	while (true) {

		// Process pending events
		application->processEvents();

		if (Main::iteration() == true) {
			break;
		}
	}

	main_loop->finish();
}

void OS_Sailfish::start_audio_driver() {
	//if (AudioDriverManager::get_driver(0)->init() != OK) {
		//ERR_PRINT("Initializing audio failed.");
	//}

	if (AudioDriver::get_singleton()->init() != OK) {
		ERR_PRINT("Initializing audio failed.");
	}
}

void OS_Sailfish::stop_audio_driver() {
	for (int i = 0; i < get_audio_driver_count(); i++) {
		AudioDriverManager::get_driver(i)->finish();
	}
}

static void on_audio_resource_acquired(audioresource_t* audio_resource, bool acquired, void* user_data) {
	//AudioDriver* driver = (AudioDriver*) user_data;
	OS_Sailfish* os = (OS_Sailfish*) user_data;

	if (acquired) {
		qDebug("starting audio driver");
		// start playback
		os->is_audio_resource_acquired = true;
		os->start_audio_driver();
	} else {
		qDebug("stopping audio driver");
		// stop playback
		//driver->finish();
		os->stop_audio_driver();
	}
}
