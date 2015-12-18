/*************************************************************************/
/*  joystick_linux.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

//author: Andreas Haas <hondres,  liugam3@gmail.com>
#ifdef __linux__

#include "joystick_linux.h"
#include "print_string.h"

#include <libevdev-1.0/libevdev/libevdev.h>
#include <libudev.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>

static const char* ignore_str = "/dev/input/js";

joystick_linux::Joystick::Joystick() {
    fd = -1;
    dpad = 0;
}

void joystick_linux::Joystick::reset() {
    num_buttons = 0;
    num_axes    = 0;
    dpad        = 0;
    fd          = -1;
    for (int i=0; i < MAX_ABS; i++) {
        abs_map[i] = -1;
    }
}

joystick_linux::joystick_linux(InputDefault *in)
{
    exit_udev = false;
    input = in;
    joy_mutex = Mutex::create();
    joy_thread = Thread::create(joy_thread_func, this);
}

joystick_linux::~joystick_linux() {
    exit_udev = true;
    Thread::wait_to_finish(joy_thread);
    close_joystick();
}

void joystick_linux::joy_thread_func(void *p_user) {

    if (p_user) {
        joystick_linux* joy = (joystick_linux*) p_user;
        joy->run_joystick_thread();
    }
    return;
}

void joystick_linux::run_joystick_thread() {

    udev *_udev = udev_new();
    ERR_FAIL_COND(!_udev);
    enumerate_joysticks(_udev);
    monitor_joysticks(_udev);
    udev_unref(_udev);
}

void joystick_linux::enumerate_joysticks(udev *p_udev) {

    udev_enumerate *enumerate;
    udev_list_entry *devices, *dev_list_entry;
    udev_device *dev;

    enumerate = udev_enumerate_new(p_udev);
    udev_enumerate_add_match_subsystem(enumerate,"input");
    udev_enumerate_add_match_property(enumerate, "ID_INPUT_JOYSTICK", "1");

    udev_enumerate_scan_devices(enumerate);
    devices = udev_enumerate_get_list_entry(enumerate);
    udev_list_entry_foreach(dev_list_entry, devices) {

        const char* path = udev_list_entry_get_name(dev_list_entry);
        dev = udev_device_new_from_syspath(p_udev, path);
        const char* devnode = udev_device_get_devnode(dev);

        if (devnode != NULL && strstr(devnode, ignore_str) == NULL) {
            joy_mutex->lock();
            open_joystick(devnode);
            joy_mutex->unlock();
        }
        udev_device_unref(dev);
    }
    udev_enumerate_unref(enumerate);
}

void joystick_linux::monitor_joysticks(udev *p_udev) {

    udev_device *dev = NULL;
    udev_monitor *mon = udev_monitor_new_from_netlink(p_udev, "udev");
    udev_monitor_filter_add_match_subsystem_devtype(mon, "input", NULL);
    udev_monitor_enable_receiving(mon);
    int fd = udev_monitor_get_fd(mon);

    while (!exit_udev) {

        fd_set fds;
        struct timeval tv;
        int ret;

        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        tv.tv_sec = 0;
        tv.tv_usec = 0;

        ret = select(fd+1, &fds, NULL, NULL, &tv);

        /* Check if our file descriptor has received data. */
        if (ret > 0 && FD_ISSET(fd, &fds)) {
            /* Make the call to receive the device.
                       select() ensured that this will not block. */
            dev = udev_monitor_receive_device(mon);

            if (dev && udev_device_get_devnode(dev) != 0) {

                joy_mutex->lock();
                const char* action = udev_device_get_action(dev);
                const char* devnode = udev_device_get_devnode(dev);

                if (strstr(devnode, ignore_str) == NULL) {

                    if (strcmp(action, "add") == 0)
                        open_joystick(devnode);

                    else if (strcmp(action, "remove") == 0)
                        close_joystick(get_joy_from_path(devnode));
                }

                udev_device_unref(dev);
                joy_mutex->unlock();
            }
        }
        usleep(50000);
    }
    //printf("exit udev\n");
    udev_monitor_unref(mon);
}

int joystick_linux::get_free_joy_slot() const {

    for (int i = 0; i < JOYSTICKS_MAX; i++) {

        if (joysticks[i].fd == -1) return i;
    }
    return -1;
}

int joystick_linux::get_joy_from_path(String p_path) const {

    for (int i = 0; i < JOYSTICKS_MAX; i++) {

        if (joysticks[i].devpath == p_path) {
            return i;
        }
    }
    return -2;
}

void joystick_linux::close_joystick(int p_id) {
    if (p_id == -1) {
        for (int i=0; i<JOYSTICKS_MAX; i++) {

            close_joystick(i);
        };
        return;
    }
    else if (p_id < 0) return;

    Joystick &joy = joysticks[p_id];

    if (joy.fd != -1) {

        libevdev_free(joy.dev);
        close(joy.fd);
        joy.fd = -1;
        input->joy_connection_changed(p_id, false, "");
    };
};

static String _hex_str(uint8_t p_byte) {

    static const char* dict = "0123456789abcdef";
    char ret[3];
    ret[2] = 0;

    ret[0] = dict[p_byte>>4];
    ret[1] = dict[p_byte & 0xF];

    return ret;
};

void joystick_linux::setup_joystick_properties(int p_id) {

    Joystick* joy = &joysticks[p_id];

    libevdev* dev = joy->dev;
    for (int i = BTN_JOYSTICK; i < KEY_MAX; ++i) {

        if (libevdev_has_event_code(dev, EV_KEY, i)) {

            joy->key_map[i] = joy->num_buttons++;
        }
    }
    for (int i = BTN_MISC; i < BTN_JOYSTICK; ++i) {

        if (libevdev_has_event_code(dev, EV_KEY, i)) {

            joy->key_map[i] = joy->num_buttons++;
        }
    }
    for (int i = 0; i < ABS_MISC; ++i) {
        /* Skip hats */
        if (i == ABS_HAT0X) {
            i = ABS_HAT3Y;
            continue;
        }
        if (libevdev_has_event_code(dev, EV_ABS, i)) {

            joy->abs_map[i] = joy->num_axes++;
        }
    }
}

void joystick_linux::open_joystick(const char *p_path) {

    int joy_num = get_free_joy_slot();
    int fd = open(p_path, O_RDONLY | O_NONBLOCK);
    if (fd != -1 && joy_num != -1) {

        int rc = libevdev_new_from_fd(fd, &joysticks[joy_num].dev);
        if (rc < 0) {

            fprintf(stderr, "Failed to init libevdev (%s)\n", strerror(-rc));
            return;
        }

        libevdev *dev = joysticks[joy_num].dev;

        //check if the device supports basic gamepad events, prevents certain keyboards from
        //being detected as joysticks
        if (libevdev_has_event_type(dev, EV_ABS) && libevdev_has_event_type(dev, EV_KEY) &&
                (libevdev_has_event_code(dev, EV_KEY, BTN_A) || libevdev_has_event_code(dev, EV_KEY, BTN_THUMBL) || libevdev_has_event_code(dev, EV_KEY, BTN_TOP))) {

            char uid[128];
            String name = libevdev_get_name(dev);
            uint16_t bus = __bswap_16(libevdev_get_id_bustype(dev));
            uint16_t vendor = __bswap_16(libevdev_get_id_vendor(dev));
            uint16_t product = __bswap_16(libevdev_get_id_product(dev));
            uint16_t version = __bswap_16(libevdev_get_id_version(dev));

            joysticks[joy_num].reset();

            Joystick &joy = joysticks[joy_num];
            joy.fd = fd;
            joy.devpath = String(p_path);
            setup_joystick_properties(joy_num);
            sprintf(uid, "%04x%04x", bus, 0);
            if (vendor && product && version) {

                sprintf(uid + String(uid).length(), "%04x%04x%04x%04x%04x%04x", vendor,0,product,0,version,0);
                input->joy_connection_changed(joy_num, true, name, uid);
            }
            else {
                String uidname = uid;
                int uidlen = MIN(name.length(), 11);
                for (int i=0; i<uidlen; i++) {

                    uidname = uidname + _hex_str(name[i]);
                }
                uidname += "00";
                input->joy_connection_changed(joy_num, true, name, uidname);

            }
        }
        else {
            //device is not a gamepad, clean up
            libevdev_free(dev);
            close(fd);
        }
    }
}

InputDefault::JoyAxis joystick_linux::axis_correct(const input_absinfo *p_abs, int p_value) const {

    int min = p_abs->minimum;
    int max = p_abs->maximum;
    InputDefault::JoyAxis jx;

    if (min < 0) {
        jx.min = -1;
        if (p_value < 0) {
            jx.value = (float) -p_value / min;
        }
        jx.value = (float) p_value / max;
    }
    if (min == 0) {
        jx.min = 0;
        jx.value = 0.0f + (float) p_value / max;
    }
    return jx;
}

uint32_t joystick_linux::process_joysticks(uint32_t p_event_id) {

    if (joy_mutex->try_lock() != OK) {
        return p_event_id;
    }
    for (int i=0; i<JOYSTICKS_MAX; i++) {

        if (joysticks[i].fd == -1) continue;

        input_event ev;
        Joystick* joy = &joysticks[i];
        libevdev* dev = joy->dev;
        int rc = 1;

        rc = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);

        if (rc < 0 && rc != -EAGAIN) {
            continue;
        }

        while (rc == LIBEVDEV_READ_STATUS_SYNC || rc == LIBEVDEV_READ_STATUS_SUCCESS) {

            switch (ev.type) {
            case EV_KEY:
                p_event_id = input->joy_button(p_event_id, i, joy->key_map[ev.code], ev.value);
                break;

            case EV_ABS:

                switch (ev.code) {
                case ABS_HAT0X:
                    if (ev.value != 0) {
                        if (ev.value < 0) joy->dpad |= InputDefault::HAT_MASK_LEFT;
                        else              joy->dpad |= InputDefault::HAT_MASK_RIGHT;

                    }
                    else joy->dpad &= ~(InputDefault::HAT_MASK_LEFT | InputDefault::HAT_MASK_RIGHT);

                    p_event_id = input->joy_hat(p_event_id, i, joy->dpad);
                    break;

                case ABS_HAT0Y:
                    if (ev.value != 0) {
                        if (ev.value < 0) joy->dpad |= InputDefault::HAT_MASK_UP;
                        else              joy->dpad |= InputDefault::HAT_MASK_DOWN;
                    }
                    else joy->dpad &= ~(InputDefault::HAT_MASK_UP | InputDefault::HAT_MASK_DOWN);

                    p_event_id = input->joy_hat(p_event_id, i, joy->dpad);
                    break;

                default:
                    if (joy->abs_map[ev.code] != -1) {
                        InputDefault::JoyAxis value = axis_correct(libevdev_get_abs_info(dev, ev.code), ev.value);
                        p_event_id = input->joy_axis(p_event_id, i, joy->abs_map[ev.code], value);
                    }
                    break;
                }
                break;
            }
            rc = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
        }
    }
    joy_mutex->unlock();
    return p_event_id;
}
#endif
