#ifndef JOYSTICK_LINUX_H
#define JOYSTICK_LINUX_H
#ifdef __linux__
#include "main/input_default.h"
#include "os/thread.h"
#include "os/mutex.h"

struct input_absinfo;

class joystick_linux
{
public:
    joystick_linux(InputDefault *in);
    ~joystick_linux();
    uint32_t process_joysticks(uint32_t p_event_id);
private:

    enum {
        JOYSTICKS_MAX = 16,
        MAX_ABS = 63,
        MAX_KEY = 767,   // Hack because <linux/input.h> can't be included here
        BT_MISC = 256,
        HAT_MAX = 4,
    };

    struct Joystick {
        int key_map[MAX_KEY - BT_MISC];
        int abs_map[MAX_ABS];
        int num_buttons;
        int num_axes;
        int fd;
        String devpath;
        struct libevdev *dev;

        Joystick();
        void reset();
    };

    int dpad_last[2];
    bool exit_udev;
    Mutex *joy_mutex;
    Thread *joy_thread;
    InputDefault *input;
    Joystick joysticks[JOYSTICKS_MAX];

    static void joy_thread_func(void *p_user);

    int handle_hat() const;
    int get_joy_from_path(String path) const;
    int get_free_joy_slot() const;

    void setup_joystick_properties(int p_id);
    void close_joystick(int p_id = -1);
    void enumerate_joysticks(struct udev *_udev);
    void monitor_joysticks(struct udev *_udev);
    void run_joystick_thread();
    void open_joystick(const char* path);

    float axis_correct(const input_absinfo *abs, int value) const;


};

#endif
#endif // JOYSTICK_LINUX_H
