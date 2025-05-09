/*
  Simple DirectMedia Layer
  Copyright (C) 2025 Simon Wood <simon@mungewell.org>
  Copyright (C) 2025 Michal Malý <madcatxster@devoid-pointer.net>
  Copyright (C) 2025 Bernat Arlandis <berarma@hotmail.com>
  Copyright (C) 2025 Katharine Chui <katharine.chui@gmail.com>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "SDL_hidapihaptic_c.h"

#ifdef SDL_HAPTIC_HIDAPI_LG4FF

#include "SDL3/SDL_thread.h"
#include "SDL3/SDL_mutex.h"
#include "SDL3/SDL_timer.h"

#include <math.h>

#define USB_VENDOR_ID_LOGITECH 0x046d
#define USB_DEVICE_ID_LOGITECH_G29_WHEEL 0xc24f
#define USB_DEVICE_ID_LOGITECH_G27_WHEEL 0xc29b
#define USB_DEVICE_ID_LOGITECH_G25_WHEEL 0xc299
#define USB_DEVICE_ID_LOGITECH_DFGT_WHEEL 0xc29a
#define USB_DEVICE_ID_LOGITECH_DFP_WHEEL 0xc298
#define USB_DEVICE_ID_LOGITECH_WHEEL 0xc294

static Uint32 supported_device_ids[] = {
    USB_DEVICE_ID_LOGITECH_G29_WHEEL,
    USB_DEVICE_ID_LOGITECH_G27_WHEEL,
    USB_DEVICE_ID_LOGITECH_G25_WHEEL,
    USB_DEVICE_ID_LOGITECH_DFGT_WHEEL,
    USB_DEVICE_ID_LOGITECH_DFP_WHEEL,
    USB_DEVICE_ID_LOGITECH_WHEEL
};



#define LG4FF_MAX_EFFECTS 16

#define FF_EFFECT_STARTED 0
#define FF_EFFECT_ALLSET 1
#define FF_EFFECT_PLAYING 2
#define FF_EFFECT_UPDATING 3

struct lg4ff_effect_state {
    SDL_HapticEffect effect;
    Uint64 start_at;
    Uint64 play_at;
    Uint64 stop_at;
    Uint32 flags;
    Uint64 time_playing;
    Uint64 updated_at;
    Uint32 phase;
    Uint32 phase_adj;
    Uint32 count;

    double direction_gain;
    Sint32 slope;

    bool allocated;
};

struct lg4ff_effect_parameters {
    Sint32 level;
    Sint32 d1;
    Sint32 d2;
    Sint32 k1;
    Sint32 k2;
    Uint32 clip;
};

struct lg4ff_slot {
    Sint32 id;
    struct lg4ff_effect_parameters parameters;
    Uint8 current_cmd[7];
    Uint32 cmd_op;
    bool is_updated;
    Uint32 effect_type;
};

typedef struct lg4ff_device {
    Uint16 product_id;
    Uint16 release_number;
    struct lg4ff_effect_state states[LG4FF_MAX_EFFECTS];
    struct lg4ff_slot slots[4];
    Sint32 effects_used;

    Sint32 gain;
    Sint32 app_gain;

    Sint32 spring_level;
    Sint32 damper_level;
    Sint32 friction_level;

    Sint32 peak_ffb_level;

    SDL_Joystick *hid_handle;

    bool stop_thread;
    SDL_Thread *thread;
    char thread_name_buf[256];

    SDL_Mutex *mutex;

    bool is_ffex;
} lg4ff_device;

static SDL_INLINE Uint64 get_time_ms(void) {
    return SDL_GetTicks();
}

#define test_bit(bit, field) (*(field) & (1 << bit))
#define __set_bit(bit, field) {*(field) = *(field) | (1 << bit);}
#define __clear_bit(bit, field) {*(field) = *(field) & ~(1 << bit);}
#define sin_deg(in) (double)(SDL_sin((double)(in) * SDL_PI_D / 180.0))

#define time_after_eq(a, b) (a >= b)
#define time_before(a, b) (a < b)
#define time_diff(a, b) (a - b)

#define STOP_EFFECT(state) ((state)->flags = 0)

#define CLAMP_VALUE_U16(x) ((Uint16)((x) > 0xffff ? 0xffff : (x)))
#define SCALE_VALUE_U16(x, bits) (CLAMP_VALUE_U16(x) >> (16 - bits))
#define CLAMP_VALUE_S16(x) ((Uint16)((x) <= -0x8000 ? -0x8000 : ((x) > 0x7fff ? 0x7fff : (x))))
#define TRANSLATE_FORCE(x) ((CLAMP_VALUE_S16(x) + 0x8000) >> 8)
#define SCALE_COEFF(x, bits) SCALE_VALUE_U16(abs32(x) * 2, bits)

static SDL_INLINE Sint32 abs32(Sint32 x) {
    return x < 0 ? -x : x;
}
static SDL_INLINE Sint64 abs64(Sint64 x) {
    return x < 0 ? -x : x;
}

static SDL_INLINE bool effect_is_periodic(const SDL_HapticEffect *effect)
{

    return effect->type == SDL_HAPTIC_SINE ||
           effect->type == SDL_HAPTIC_TRIANGLE ||
           effect->type == SDL_HAPTIC_SAWTOOTHUP ||
           effect->type == SDL_HAPTIC_SAWTOOTHDOWN ||
           effect->type == SDL_HAPTIC_SQUARE;
}

static SDL_INLINE bool effect_is_condition(const SDL_HapticEffect *effect)
{
    return effect->type == SDL_HAPTIC_SPRING ||
           effect->type == SDL_HAPTIC_DAMPER ||
           effect->type == SDL_HAPTIC_FRICTION;
}

// linux SDL_syshaptic.c SDL_SYS_ToDirection
static Uint16 to_linux_direction(SDL_HapticDirection *src)
{
    Uint32 tmp;

    switch (src->type) {
    case SDL_HAPTIC_POLAR:
        tmp = ((src->dir[0] % 36000) * 0x8000) / 18000; /* convert to range [0,0xFFFF] */
        return (Uint16)tmp;

    case SDL_HAPTIC_SPHERICAL:
        /*
            We convert to polar, because that's the only supported direction on Linux.
            The first value of a spherical direction is practically the same as a
            Polar direction, except that we have to add 90 degrees. It is the angle
            from EAST {1,0} towards SOUTH {0,1}.
            --> add 9000
            --> finally convert to [0,0xFFFF] as in case SDL_HAPTIC_POLAR.
        */
        tmp = ((src->dir[0]) + 9000) % 36000; /* Convert to polars */
        tmp = (tmp * 0x8000) / 18000;         /* convert to range [0,0xFFFF] */
        return (Uint16)tmp;

    case SDL_HAPTIC_CARTESIAN:
        if (!src->dir[1]) {
            return (Uint16) (src->dir[0] >= 0 ? 0x4000 : 0xC000);
        } else if (!src->dir[0]) {
            return (Uint16) (src->dir[1] >= 0 ? 0x8000 : 0);
        } else {
            float f = (float)SDL_atan2(src->dir[1], src->dir[0]);    /* Ideally we'd use fixed point math instead of floats... */
                    /*
                    SDL_atan2 takes the parameters: Y-axis-value and X-axis-value (in that order)
                    - Y-axis-value is the second coordinate (from center to SOUTH)
                    - X-axis-value is the first coordinate (from center to EAST)
                        We add 36000, because SDL_atan2 also returns negative values. Then we practically
                        have the first spherical value. Therefore we proceed as in case
                        SDL_HAPTIC_SPHERICAL and add another 9000 to get the polar value.
                    --> add 45000 in total
                    --> finally convert to [0,0xFFFF] as in case SDL_HAPTIC_POLAR.
                    */
                tmp = (((Sint32) (f * 18000. / SDL_PI_D)) + 45000) % 36000;
            tmp = (tmp * 0x8000) / 18000; /* convert to range [0,0xFFFF] */
            return (Uint16)tmp;
        }
    case SDL_HAPTIC_STEERING_AXIS:
        return 0x4000;
    default:
        SDL_assert(0);
    }

    return 0;
}

static Uint16 get_effect_direction(SDL_HapticEffect *effect)
{
    Uint16 direction = 0;
    if (effect_is_periodic(effect)) {
        direction = to_linux_direction(&effect->periodic.direction);
    } else if (effect_is_condition(effect)) {
        direction = to_linux_direction(&effect->condition.direction);
    } else {
        switch(effect->type) {
            case SDL_HAPTIC_CONSTANT:
                direction = to_linux_direction(&effect->constant.direction);
                break;
            case SDL_HAPTIC_RAMP:
                direction = to_linux_direction(&effect->ramp.direction);
                break;
            default:
                SDL_assert(0);
        }
    }
    
    return direction;
}

static Uint32 get_effect_replay_length(SDL_HapticEffect *effect)
{
    Uint32 length = 0;
    if (effect_is_periodic(effect)) {
        length = effect->periodic.length;
    } else if (effect_is_condition(effect)) {
        length = effect->condition.length;
    } else {
        switch(effect->type) {
            case SDL_HAPTIC_CONSTANT:
                length = effect->constant.length;
                break;
            case SDL_HAPTIC_RAMP:
                length = effect->ramp.length;
                break;
            default:
                SDL_assert(0);
        }
    }

    if (length == SDL_HAPTIC_INFINITY) {
        length = 0;
    }

    return length;
}

static Uint16 get_effect_replay_delay(SDL_HapticEffect *effect)
{
    Uint16 delay = 0;
    if (effect_is_periodic(effect)) {
        delay = effect->periodic.delay;
    } else if (effect_is_condition(effect)) {
        delay = effect->condition.delay;
    } else {
        switch(effect->type) {
            case SDL_HAPTIC_CONSTANT:
                delay = effect->constant.delay;
                break;
            case SDL_HAPTIC_RAMP:
                delay = effect->ramp.delay;
                break;
            default:
                SDL_assert(0);
        }
    }

    return delay;
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static int lg4ff_play_effect(struct lg4ff_device *device, SDL_HapticEffectID effect_id, int value)
{
    struct lg4ff_effect_state *state;
    Uint64 now = get_time_ms();

    state = &device->states[effect_id];

    if (value > 0) {
        if (test_bit(FF_EFFECT_STARTED, &state->flags)) {
            STOP_EFFECT(state);
        } else {
            device->effects_used++;
        }
        __set_bit(FF_EFFECT_STARTED, &state->flags);
        state->start_at = now;
        state->count = value;
    } else {
        if (test_bit(FF_EFFECT_STARTED, &state->flags)) {
            STOP_EFFECT(state);
            device->effects_used--;
        }
    }

    return 0;
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static int lg4ff_upload_effect(struct lg4ff_device *device, const SDL_HapticEffect *effect, SDL_HapticEffectID id)
{
    struct lg4ff_effect_state *state;
    Uint64 now = get_time_ms();

    if (effect_is_periodic(effect) && effect->periodic.period == 0) {
        return -1;
    }

    state = &device->states[id];

    if (test_bit(FF_EFFECT_STARTED, &state->flags) && effect->type != state->effect.type) {
        return -1;
    }

    state->effect = *effect;

    if (test_bit(FF_EFFECT_STARTED, &state->flags)) {
        __set_bit(FF_EFFECT_UPDATING, &state->flags);
        state->updated_at = now;
    }

    return 0;
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static void lg4ff_update_state(struct lg4ff_effect_state *state, const Uint64 now)
{
    SDL_HapticEffect *effect = &state->effect;
    Uint64 phase_time;
    Uint16 effect_direction = get_effect_direction(effect);

    if (!test_bit(FF_EFFECT_ALLSET, &state->flags)) {
        state->play_at = state->start_at + get_effect_replay_delay(effect);
        if (!test_bit(FF_EFFECT_UPDATING, &state->flags)) {
            state->updated_at = state->play_at;
        }
        state->direction_gain = sin_deg(effect_direction * 360 / 0x10000);
        if (effect_is_periodic(effect)) {
            state->phase_adj = effect->periodic.phase * 360 / effect->periodic.period;
        }
        if (get_effect_replay_length(effect)) {
            state->stop_at = state->play_at + get_effect_replay_length(effect);
        }
    }
    __set_bit(FF_EFFECT_ALLSET, &state->flags);

    if (test_bit(FF_EFFECT_UPDATING, &state->flags)) {
        __clear_bit(FF_EFFECT_PLAYING, &state->flags);
        state->play_at = state->updated_at + get_effect_replay_delay(effect);
        state->direction_gain = sin_deg(effect_direction * 360 / 0x10000);
        if (get_effect_replay_length(effect)) {
            state->stop_at = state->updated_at + get_effect_replay_length(effect);
        }
        if (effect_is_periodic(effect)) {
            state->phase_adj = state->phase;
        }
    }
    __clear_bit(FF_EFFECT_UPDATING, &state->flags);

    state->slope = 0;
    if (effect->type == SDL_HAPTIC_RAMP && effect->ramp.length && (effect->ramp.length - effect->ramp.attack_length - effect->ramp.fade_length) != 0) {
        state->slope = ((effect->ramp.end - effect->ramp.start) << 16) / (effect->ramp.length - effect->ramp.attack_length - effect->ramp.fade_length);
    }

    if (!test_bit(FF_EFFECT_PLAYING, &state->flags) && time_after_eq(now,
                state->play_at) && (get_effect_replay_length(effect) == 0 ||
                    time_before(now, state->stop_at))) {
        __set_bit(FF_EFFECT_PLAYING, &state->flags);
    }

    if (test_bit(FF_EFFECT_PLAYING, &state->flags)) {
        state->time_playing = time_diff(now, state->play_at);
        if (effect_is_periodic(effect)) {
            phase_time = time_diff(now, state->updated_at);
            state->phase = (phase_time % effect->periodic.period) * 360 / effect->periodic.period;
            state->phase += state->phase_adj % 360;
        }
    }
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static Sint32 lg4ff_calculate_constant(struct lg4ff_effect_state *state)
{
    SDL_HapticConstant *constant = (SDL_HapticConstant *)&state->effect;
    Sint32 level_sign;
    Sint32 level = constant->level;
    Sint32 d, t;

    if (state->time_playing < constant->attack_length) {
        level_sign = level < 0 ? -1 : 1;
        d = level - level_sign * constant->attack_level;
        level = (Sint32) (level_sign * constant->attack_level + d * state->time_playing / constant->attack_length);
    } else if (constant->length && constant->fade_length) {
        t = (Sint32) (state->time_playing - constant->length + constant->fade_length);
        if (t > 0) {
            level_sign = level < 0 ? -1 : 1;
            d = level - level_sign * constant->fade_level;
            level = level - d * t / constant->fade_length;
        }
    }

    return (Sint32)(state->direction_gain * level);
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static Sint32 lg4ff_calculate_ramp(struct lg4ff_effect_state *state)
{
    SDL_HapticRamp *ramp = (SDL_HapticRamp *)&state->effect;
    Sint32 level_sign;
    Sint32 level;
    Sint32 d, t;

    if (state->time_playing < ramp->attack_length) {
        level = ramp->start;
        level_sign =  level < 0 ? -1 : 1;
        t = (Sint32) (ramp->attack_length - state->time_playing);
        d = level - level_sign * ramp->attack_level;
        level = level_sign * ramp->attack_level + d * t / ramp->attack_length;
    } else if (ramp->length && state->time_playing >= ramp->length - ramp->fade_length && ramp->fade_length) {
        level = ramp->end;
        level_sign = level < 0 ? -1 : 1;
        t = (Sint32) (state->time_playing - ramp->length + ramp->fade_length);
        d = level_sign * ramp->fade_level - level;
        level = level - d * t / ramp->fade_length;
    } else {
        t = (Sint32) (state->time_playing - ramp->attack_length);
        level = ramp->start + ((t * state->slope) >> 16);
    }

    return (Sint32)(state->direction_gain * level);
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static Sint32 lg4ff_calculate_periodic(struct lg4ff_effect_state *state)
{
    SDL_HapticPeriodic *periodic = (SDL_HapticPeriodic *)&state->effect;
    Sint32 magnitude = periodic->magnitude;
    Sint32 magnitude_sign = magnitude < 0 ? -1 : 1;
    Sint32 level = periodic->offset;
    Sint32 d, t;

    if (state->time_playing < periodic->attack_length) {
        d = magnitude - magnitude_sign * periodic->attack_level;
        magnitude = (Sint32) (magnitude_sign * periodic->attack_level + d * state->time_playing / periodic->attack_length);
    } else if (periodic->length && periodic->fade_length) {
        t = (Sint32) (state->time_playing - get_effect_replay_length(&state->effect) + periodic->fade_length);
        if (t > 0) {
            d = magnitude - magnitude_sign * periodic->fade_level;
            magnitude = magnitude - d * t / periodic->fade_length;
        }
    }

    switch (periodic->type) {
        case SDL_HAPTIC_SINE:
            level += (Sint32)(sin_deg(state->phase) * magnitude);
            break;
        case SDL_HAPTIC_SQUARE:
            level += (state->phase < 180 ? 1 : -1) * magnitude;
            break;
        case SDL_HAPTIC_TRIANGLE:
            level += (Sint32) (abs64((Sint64)state->phase * magnitude * 2 / 360 - magnitude) * 2 - magnitude);
            break;
        case SDL_HAPTIC_SAWTOOTHUP:
            level += state->phase * magnitude * 2 / 360 - magnitude;
            break;
        case SDL_HAPTIC_SAWTOOTHDOWN:
            level += magnitude - state->phase * magnitude * 2 / 360;
            break;
        default:
            SDL_assert(0);
    }

    return (Sint32)(state->direction_gain * level);
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static void lg4ff_calculate_spring(struct lg4ff_effect_state *state, struct lg4ff_effect_parameters *parameters)
{
    SDL_HapticCondition *condition = (SDL_HapticCondition *)&state->effect;

    parameters->d1 = ((Sint32)condition->center[0]) - condition->deadband[0] / 2;
    parameters->d2 = ((Sint32)condition->center[0]) + condition->deadband[0] / 2;
    parameters->k1 = condition->left_coeff[0];
    parameters->k2 = condition->right_coeff[0];
    parameters->clip = (Uint16)condition->right_sat[0];
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static void lg4ff_calculate_resistance(struct lg4ff_effect_state *state, struct lg4ff_effect_parameters *parameters)
{
    SDL_HapticCondition *condition = (SDL_HapticCondition *)&state->effect;

    parameters->k1 = condition->left_coeff[0];
    parameters->k2 = condition->right_coeff[0];
    parameters->clip = (Uint16)condition->right_sat[0];
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static void lg4ff_update_slot(struct lg4ff_slot *slot, struct lg4ff_effect_parameters *parameters)
{
    Uint8 original_cmd[7];
    Sint32 d1;
    Sint32 d2; 
    Sint32 k1;
    Sint32 k2;
    Sint32 s1;
    Sint32 s2;

    SDL_memcpy(original_cmd, slot->current_cmd, sizeof(original_cmd));

    if ((original_cmd[0] & 0xf) == 1) {
        original_cmd[0] = (original_cmd[0] & 0xf0) + 0xc;
    }

    if (slot->effect_type == SDL_HAPTIC_CONSTANT) {
        if (slot->cmd_op == 0) {
            slot->cmd_op = 1;
        } else {
            slot->cmd_op = 0xc;
        }
    } else {
        if (parameters->clip == 0) {
            slot->cmd_op = 3;
        } else if (slot->cmd_op == 3) {
            slot->cmd_op = 1;
        } else {
            slot->cmd_op = 0xc;
        }
    }

    slot->current_cmd[0] = (Uint8)((0x10 << slot->id) + slot->cmd_op);

    if (slot->cmd_op == 3) {
        slot->current_cmd[1] = 0;
        slot->current_cmd[2] = 0;
        slot->current_cmd[3] = 0;
        slot->current_cmd[4] = 0;
        slot->current_cmd[5] = 0;
        slot->current_cmd[6] = 0;
    } else {
        switch (slot->effect_type) {
            case SDL_HAPTIC_CONSTANT:
                slot->current_cmd[1] = 0x00;
                slot->current_cmd[2] = 0;
                slot->current_cmd[3] = 0;
                slot->current_cmd[4] = 0;
                slot->current_cmd[5] = 0;
                slot->current_cmd[6] = 0;
                slot->current_cmd[2 + slot->id] = TRANSLATE_FORCE(parameters->level);
                break;
            case SDL_HAPTIC_SPRING:
                d1 = SCALE_VALUE_U16(((parameters->d1) + 0x8000) & 0xffff, 11);
                d2 = SCALE_VALUE_U16(((parameters->d2) + 0x8000) & 0xffff, 11);
                s1 = parameters->k1 < 0;
                s2 = parameters->k2 < 0;
                k1 = abs32(parameters->k1);
                k2 = abs32(parameters->k2);
                if (k1 < 2048) {
                    d1 = 0;
                } else {
                    k1 -= 2048;
                }
                if (k2 < 2048) {
                    d2 = 2047;
                } else {
                    k2 -= 2048;
                }
                slot->current_cmd[1] = 0x0b;
                slot->current_cmd[2] = (Uint8)(d1 >> 3);
                slot->current_cmd[3] = (Uint8)(d2 >> 3);
                slot->current_cmd[4] = (SCALE_COEFF(k2, 4) << 4) + SCALE_COEFF(k1, 4);
                slot->current_cmd[5] = (Uint8)(((d2 & 7) << 5) + ((d1 & 7) << 1) + (s2 << 4) + s1);
                slot->current_cmd[6] = SCALE_VALUE_U16(parameters->clip, 8);
                break;
            case SDL_HAPTIC_DAMPER:
                s1 = parameters->k1 < 0;
                s2 = parameters->k2 < 0;
                slot->current_cmd[1] = 0x0c;
                slot->current_cmd[2] = SCALE_COEFF(parameters->k1, 4);
                slot->current_cmd[3] = (Uint8)s1;
                slot->current_cmd[4] = SCALE_COEFF(parameters->k2, 4);
                slot->current_cmd[5] = (Uint8)s2;
                slot->current_cmd[6] = SCALE_VALUE_U16(parameters->clip, 8);
                break;
            case SDL_HAPTIC_FRICTION:
                s1 = parameters->k1 < 0;
                s2 = parameters->k2 < 0;
                slot->current_cmd[1] = 0x0e;
                slot->current_cmd[2] = SCALE_COEFF(parameters->k1, 8);
                slot->current_cmd[3] = SCALE_COEFF(parameters->k2, 8);
                slot->current_cmd[4] = SCALE_VALUE_U16(parameters->clip, 8);
                slot->current_cmd[5] = (Uint8)((s2 << 4) + s1);
                slot->current_cmd[6] = 0;
                break;
        }
    }

    if (SDL_memcmp(original_cmd, slot->current_cmd, sizeof(original_cmd))) {
        slot->is_updated = 1;
    }
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static int lg4ff_init_slots(struct lg4ff_device *device)
{
    struct lg4ff_effect_parameters parameters;
    Uint8 cmd[7] = {0};
    int i;
    bool ret;

    // Set/unset fixed loop mode
    cmd[0] = 0x0d;
    //cmd[1] = fixed_loop ? 1 : 0;
    cmd[1] = 0;
    ret = SDL_SendJoystickEffect(device->hid_handle, cmd, 7);
    if (!ret) {
        return -1;
    }

    SDL_memset(&device->states, 0, sizeof(device->states));
    SDL_memset(&device->slots, 0, sizeof(device->slots));
    SDL_memset(&parameters, 0, sizeof(parameters));

    device->slots[0].effect_type = SDL_HAPTIC_CONSTANT;
    device->slots[1].effect_type = SDL_HAPTIC_SPRING;
    device->slots[2].effect_type = SDL_HAPTIC_DAMPER;
    device->slots[3].effect_type = SDL_HAPTIC_FRICTION;

    for (i = 0; i < 4; i++) {
        device->slots[i].id = i;
        lg4ff_update_slot(&device->slots[i], &parameters);
        ret = SDL_SendJoystickEffect(device->hid_handle, cmd, 7);
        if (!ret) {
            return -1;
        }
        device->slots[i].is_updated = 0;
    }

    return 0;
}

/*
  *Ported*
  Original function by:
  Bernat Arlandis <berarma@hotmail.com>
  `git blame 1a2d5727876dd7befce23d9695924e9446b31c4b hid-lg4ff.c`, https://github.com/berarma/new-lg4ff.git
*/
static int lg4ff_timer(struct lg4ff_device *device)
{
    struct lg4ff_slot *slot;
    struct lg4ff_effect_state *state;
    struct lg4ff_effect_parameters parameters[4];
    Uint64 now = get_time_ms();
    Uint16 gain;
    Sint32 count;
    Sint32 effect_id;
    int i;
    Sint32 ffb_level;
    int status = 0;

    // XXX how to detect stacked up effects here?

    SDL_memset(parameters, 0, sizeof(parameters));

    gain = (Uint16)((Uint32)device->gain * device->app_gain / 0xffff);

    count = device->effects_used;

    for (effect_id = 0; effect_id < LG4FF_MAX_EFFECTS; effect_id++) {

        if (!count) {
            break;
        }

        state = &device->states[effect_id];

        if (!test_bit(FF_EFFECT_STARTED, &state->flags)) {
            continue;
        }

        count--;

        if (test_bit(FF_EFFECT_ALLSET, &state->flags)) {
            if (get_effect_replay_length(&state->effect) && time_after_eq(now, state->stop_at)) {
            STOP_EFFECT(state);
            if (!--state->count) {
                device->effects_used--;
                continue;
            }
            __set_bit(FF_EFFECT_STARTED, &state->flags);
            state->start_at = state->stop_at;
            }
        }

        lg4ff_update_state(state, now);

        if (!test_bit(FF_EFFECT_PLAYING, &state->flags)) {
            continue;
        }

        if (effect_is_periodic(&state->effect)) {
            parameters[0].level += lg4ff_calculate_periodic(state);
        } else {
            switch (state->effect.type) {
            case SDL_HAPTIC_CONSTANT:
                parameters[0].level += lg4ff_calculate_constant(state);
                break;
            case SDL_HAPTIC_RAMP:
                parameters[0].level += lg4ff_calculate_ramp(state);
                break;
            case SDL_HAPTIC_SPRING:
                lg4ff_calculate_spring(state, &parameters[1]);
                break;
            case SDL_HAPTIC_DAMPER:
                lg4ff_calculate_resistance(state, &parameters[2]);
                break;
            case SDL_HAPTIC_FRICTION:
                lg4ff_calculate_resistance(state, &parameters[3]);
                break;
            }
        }
    }

    parameters[0].level = (Sint32)((Sint64)parameters[0].level * gain / 0xffff);
    parameters[1].clip = parameters[1].clip * device->spring_level / 100;
    parameters[2].clip = parameters[2].clip * device->damper_level / 100;
    parameters[3].clip = parameters[3].clip * device->friction_level / 100;

    ffb_level = abs32(parameters[0].level);
    for (i = 1; i < 4; i++) {
        parameters[i].k1 = (Sint32)((Sint64)parameters[i].k1 * gain / 0xffff);
        parameters[i].k2 = (Sint32)((Sint64)parameters[i].k2 * gain / 0xffff);
        parameters[i].clip = parameters[i].clip * gain / 0xffff;
        ffb_level = (Sint32)(ffb_level + parameters[i].clip * 0x7fff / 0xffff);
    }
    if (ffb_level > device->peak_ffb_level) {
        device->peak_ffb_level = ffb_level;
    }

    for (i = 0; i < 4; i++) {
        slot = &device->slots[i];
        lg4ff_update_slot(slot, &parameters[i]);
        if (slot->is_updated) {
            bool ret = SDL_SendJoystickEffect(device->hid_handle, slot->current_cmd, 7);
            if (!ret) {
                status = -1;
            }
            slot->is_updated = 0;
        }
    }

    return status;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_JoystickSupported(SDL_Joystick *joystick)
{
    Uint16 vendor_id = SDL_GetJoystickVendor(joystick);
    Uint16 product_id = SDL_GetJoystickProduct(joystick);
    if (vendor_id != USB_VENDOR_ID_LOGITECH) {
        return false;
    }
    for (int i = 0;i < sizeof(supported_device_ids) / sizeof(Uint32);i++) {
        if (supported_device_ids[i] == product_id) {
            return true;
        }
    }
    return false;
}

static int SDLCALL SDL_HIDAPI_HapticDriverLg4ff_ThreadFunction(void *ctx_in)
{
    lg4ff_device *ctx = (lg4ff_device *)ctx_in;
    while (true) {
        if (ctx->stop_thread) {
            return 0;
        }
        SDL_LockMutex(ctx->mutex);
        lg4ff_timer(ctx);
        SDL_UnlockMutex(ctx->mutex);
        SDL_Delay(2);
    }
}

static int SDL_HIDAPI_HapticDriverLg4ff_GetEnvInt(const char *env_name, int min, int max, int def)
{
    const char *env = SDL_getenv(env_name);
    int value = 0;
    if (env == NULL) {
        return def;
    }
    value = SDL_atoi(env);
    if (value < min) {
        value = min;
    }
    if (value > max) {
        value = max;
    }
    return value;
}

/*
  ffex identification method by:
  Simon Wood <simon@mungewell.org>
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_init
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static void *SDL_HIDAPI_HapticDriverLg4ff_Open(SDL_Joystick *joystick)
{
    lg4ff_device *ctx;
    if (!SDL_HIDAPI_HapticDriverLg4ff_JoystickSupported(joystick)) {
        SDL_SetError("Device not supported by the lg4ff hidapi haptic driver");
        return NULL;
    }

    ctx = SDL_malloc(sizeof(lg4ff_device));
    if (ctx == NULL) {
        SDL_OutOfMemory();
        return NULL;
    }
    SDL_memset(ctx, 0, sizeof(lg4ff_device));

    ctx->hid_handle = joystick;
    if (lg4ff_init_slots(ctx) != 0) {
        SDL_SetError("lg4ff hidapi driver failed initializing effect slots");
        SDL_free(ctx);
        return NULL;
    }

    ctx->mutex = SDL_CreateMutex();
    if (ctx->mutex == NULL) {
        SDL_free(ctx);
        return NULL;
    }

    ctx->spring_level = SDL_HIDAPI_HapticDriverLg4ff_GetEnvInt("SDL_HAPTIC_LG4FF_SPRING", 0, 100, 30);
    ctx->damper_level = SDL_HIDAPI_HapticDriverLg4ff_GetEnvInt("SDL_HAPTIC_LG4FF_DAMPER", 0, 100, 30);
    ctx->friction_level = SDL_HIDAPI_HapticDriverLg4ff_GetEnvInt("SDL_HAPTIC_LG4FF_FRICTION", 0, 100, 30);
    ctx->gain = SDL_HIDAPI_HapticDriverLg4ff_GetEnvInt("SDL_HAPTIC_LG4FF_GAIN", 0, 65535, 65535);
    ctx->app_gain = 65535;

    ctx->product_id = SDL_GetJoystickProduct(joystick);
    ctx->release_number = SDL_GetJoystickProductVersion(joystick);

    SDL_snprintf(ctx->thread_name_buf, sizeof(ctx->thread_name_buf), "SDL_hidapihaptic_lg4ff %d %04x:%04x", SDL_GetJoystickID(joystick), USB_VENDOR_ID_LOGITECH, ctx->product_id);
    ctx->stop_thread = false;
    ctx->thread = SDL_CreateThread(SDL_HIDAPI_HapticDriverLg4ff_ThreadFunction, ctx->thread_name_buf, ctx);

    if (ctx->product_id == USB_DEVICE_ID_LOGITECH_WHEEL &&
            (ctx->release_number >> 8) == 0x21 &&
            (ctx->release_number & 0xff) == 0x00) {
        ctx->is_ffex = true;
    } else {
        ctx->is_ffex = false;
    }

    return ctx;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_StopEffects(SDL_HIDAPI_HapticDevice *device)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    int i;

    SDL_LockMutex(ctx->mutex);
    for (i = 0;i < LG4FF_MAX_EFFECTS;i++) {
        struct lg4ff_effect_state *state = &ctx->states[i];
        STOP_EFFECT(state);
    }
    SDL_UnlockMutex(ctx->mutex);

    return true;
}

static void SDL_HIDAPI_HapticDriverLg4ff_Close(SDL_HIDAPI_HapticDevice *device)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;

    SDL_HIDAPI_HapticDriverLg4ff_StopEffects(device);

    // let effects finish in lg4ff_timer
    SDL_Delay(50);

    ctx->stop_thread = true;
    SDL_WaitThread(ctx->thread, NULL);
    SDL_DestroyMutex(ctx->mutex);
}

static int SDL_HIDAPI_HapticDriverLg4ff_NumEffects(SDL_HIDAPI_HapticDevice *device)
{
    return LG4FF_MAX_EFFECTS;
}

static Uint32 SDL_HIDAPI_HapticDriverLg4ff_GetFeatures(SDL_HIDAPI_HapticDevice *device)
{
    return SDL_HAPTIC_CONSTANT |
        SDL_HAPTIC_SPRING |
        SDL_HAPTIC_DAMPER |
        SDL_HAPTIC_AUTOCENTER |
        SDL_HAPTIC_SINE |
        SDL_HAPTIC_SQUARE |
        SDL_HAPTIC_TRIANGLE |
        SDL_HAPTIC_SAWTOOTHUP |
        SDL_HAPTIC_SAWTOOTHDOWN |
        SDL_HAPTIC_RAMP |
        SDL_HAPTIC_FRICTION |
        SDL_HAPTIC_STATUS |
        SDL_HAPTIC_GAIN;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_EffectSupported(SDL_HIDAPI_HapticDevice *device, const SDL_HapticEffect *effect) {
    Uint32 features = SDL_HIDAPI_HapticDriverLg4ff_GetFeatures(device);
    return (features & effect->type)? true : false;
}

static int SDL_HIDAPI_HapticDriverLg4ff_NumAxes(SDL_HIDAPI_HapticDevice *device)
{
    return 1;
}

static SDL_HapticEffectID SDL_HIDAPI_HapticDriverLg4ff_CreateEffect(SDL_HIDAPI_HapticDevice *device, const SDL_HapticEffect *data)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    SDL_HapticEffectID i;
    SDL_HapticEffectID state_slot = -1;
    int ret;
    if (!SDL_HIDAPI_HapticDriverLg4ff_EffectSupported(device, data)) {
        SDL_SetError("Unsupported effect");
        return -1;
    }

    SDL_LockMutex(ctx->mutex);
    for (i = 0;i < LG4FF_MAX_EFFECTS;i++) {
        if (!ctx->states[i].allocated) {
            state_slot = i;
            break;
        }
    }
    if (state_slot == -1) {
        SDL_UnlockMutex(ctx->mutex);
        SDL_SetError("All effect slots in-use");
        return -1;
    }

    ret = lg4ff_upload_effect(ctx, data, state_slot);
    SDL_UnlockMutex(ctx->mutex);
    if (ret == 0) {
        ctx->states[state_slot].allocated = true;
        return state_slot;
    } else {
        SDL_SetError("Bad effect parameters");
        return -1;
    }
}

// assumes ctx->mutex locked
static bool lg4ff_effect_slot_valid_active(lg4ff_device *ctx, SDL_HapticEffectID id)
{
    if (id >= LG4FF_MAX_EFFECTS || id < 0) {
        return false;
    }
    if (!ctx->states[id].allocated) {
        return false;
    }
    return true;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_UpdateEffect(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id, const SDL_HapticEffect *data)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    int ret;

    SDL_LockMutex(ctx->mutex);
    if (!lg4ff_effect_slot_valid_active(ctx, id)) {
        SDL_UnlockMutex(ctx->mutex);
        SDL_SetError("Bad effect id");
        return false;
    }

    ret = lg4ff_upload_effect(ctx, data, id);
    SDL_UnlockMutex(ctx->mutex);

    return ret == 0;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_RunEffect(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id, Uint32 iterations)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    int ret;

    SDL_LockMutex(ctx->mutex);
    if (!lg4ff_effect_slot_valid_active(ctx, id)) {
        SDL_UnlockMutex(ctx->mutex);
        SDL_SetError("Bad effect id");
        return false;
    }

    ret = lg4ff_play_effect(ctx, id, iterations);
    SDL_UnlockMutex(ctx->mutex);

    return ret == 0;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_StopEffect(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id)
{
    return SDL_HIDAPI_HapticDriverLg4ff_RunEffect(device, id, 0);
}

static void SDL_HIDAPI_HapticDriverLg4ff_DestroyEffect(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    struct lg4ff_effect_state *state;

    SDL_LockMutex(ctx->mutex);
    if (!lg4ff_effect_slot_valid_active(ctx, id)) {
        SDL_UnlockMutex(ctx->mutex);
        return;
    }

    state = &ctx->states[id];
    STOP_EFFECT(state);
    state->allocated = false;

    SDL_UnlockMutex(ctx->mutex);
}

static bool SDL_HIDAPI_HapticDriverLg4ff_GetEffectStatus(SDL_HIDAPI_HapticDevice *device, SDL_HapticEffectID id)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    bool ret = false;

    SDL_LockMutex(ctx->mutex);
    if (!lg4ff_effect_slot_valid_active(ctx, id)) {
        SDL_UnlockMutex(ctx->mutex);
        return false;
    }

    if (test_bit(FF_EFFECT_STARTED, &ctx->states[id].flags)) {
        ret = true;
    }
    SDL_UnlockMutex(ctx->mutex);

    return ret;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_SetGain(SDL_HIDAPI_HapticDevice *device, int gain)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    if (gain > 100) {
        gain = 100;
    }
    if (gain < 0) {
        gain = 0;
    }
    ctx->app_gain = (65535 * gain) / 100;
    return true;
}

/*
  *Ported*
  Original functions by:
  Simon Wood <simon@mungewell.org>
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_set_autocenter_default lg4ff_set_autocenter_ffex
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool SDL_HIDAPI_HapticDriverLg4ff_SetAutocenter(SDL_HIDAPI_HapticDevice *device, int autocenter)
{
    lg4ff_device *ctx = (lg4ff_device *)device->ctx;
    Uint8 cmd[7] = {0};
    bool ret;

    if (autocenter < 0) {
        autocenter = 0;
    }
    if (autocenter > 100) {
        autocenter = 100;
    }

    SDL_LockMutex(ctx->mutex);
    if (ctx->is_ffex) {
        int magnitude = (90 * autocenter) / 100;

        cmd[0] = 0xfe;
        cmd[1] = 0x03;
        cmd[2] = (Uint8)((Uint16)magnitude >> 14);
        cmd[3] = (Uint8)((Uint16)magnitude >> 14);
        cmd[4] = (Uint8)magnitude;

        ret = SDL_SendJoystickEffect(ctx->hid_handle, cmd, sizeof(cmd));
        if (!ret) {
            SDL_UnlockMutex(ctx->mutex);
            SDL_SetError("Failed sending autocenter command");
            return false;
        }
    } else {
        Uint32 expand_a;
        Uint32 expand_b;
        int magnitude = (65535 * autocenter) / 100;

        // first disable
        cmd[0] = 0xf5;

        ret = SDL_SendJoystickEffect(ctx->hid_handle, cmd, sizeof(cmd));
        if (!ret) {
            SDL_UnlockMutex(ctx->mutex);
            SDL_SetError("Failed sending autocenter disable command");
            return false;
        }

        if (magnitude == 0) {
            SDL_UnlockMutex(ctx->mutex);
            return true;
        }

        // set strength
        if (magnitude <= 0xaaaa) {
            expand_a = 0x0c * magnitude;
            expand_b = 0x80 * magnitude;
        } else {
            expand_a = (0x0c * 0xaaaa) + 0x06 * (magnitude - 0xaaaa);
            expand_b = (0x80 * 0xaaaa) + 0xff * (magnitude - 0xaaaa);
        }
        expand_a = expand_a >> 1;

        SDL_memset(cmd, 0x00, 7);
        cmd[0] = 0xfe;
        cmd[1] = 0x0d;
        cmd[2] = (Uint8)(expand_a / 0xaaaa);
        cmd[3] = (Uint8)(expand_a / 0xaaaa);
        cmd[4] = (Uint8)(expand_b / 0xaaaa);

        ret = SDL_SendJoystickEffect(ctx->hid_handle, cmd, sizeof(cmd));
        if (!ret) {
            SDL_UnlockMutex(ctx->mutex);
            SDL_SetError("Failed sending autocenter magnitude command");
            return false;
        }

        // enable
        SDL_memset(cmd, 0x00, 7);
        cmd[0] = 0x14;

        ret = SDL_SendJoystickEffect(ctx->hid_handle, cmd, sizeof(cmd));
        if (!ret) {
            SDL_UnlockMutex(ctx->mutex);
            SDL_SetError("Failed sending autocenter enable command");
            return false;
        }
    }
    SDL_UnlockMutex(ctx->mutex);
    return true;
}

static bool SDL_HIDAPI_HapticDriverLg4ff_Pause(SDL_HIDAPI_HapticDevice *device)
{
    return SDL_Unsupported();
}

static bool SDL_HIDAPI_HapticDriverLg4ff_Resume(SDL_HIDAPI_HapticDevice *device)
{
    return SDL_Unsupported();
}

SDL_HIDAPI_HapticDriver SDL_HIDAPI_HapticDriverLg4ff = {
    SDL_HIDAPI_HapticDriverLg4ff_JoystickSupported,
    SDL_HIDAPI_HapticDriverLg4ff_Open,
    SDL_HIDAPI_HapticDriverLg4ff_Close,
    SDL_HIDAPI_HapticDriverLg4ff_NumEffects,
    SDL_HIDAPI_HapticDriverLg4ff_NumEffects,
    SDL_HIDAPI_HapticDriverLg4ff_GetFeatures,
    SDL_HIDAPI_HapticDriverLg4ff_NumAxes,
    SDL_HIDAPI_HapticDriverLg4ff_CreateEffect,
    SDL_HIDAPI_HapticDriverLg4ff_UpdateEffect,
    SDL_HIDAPI_HapticDriverLg4ff_RunEffect,
    SDL_HIDAPI_HapticDriverLg4ff_StopEffect,
    SDL_HIDAPI_HapticDriverLg4ff_DestroyEffect,
    SDL_HIDAPI_HapticDriverLg4ff_GetEffectStatus,
    SDL_HIDAPI_HapticDriverLg4ff_SetGain,
    SDL_HIDAPI_HapticDriverLg4ff_SetAutocenter,
    SDL_HIDAPI_HapticDriverLg4ff_Pause,
    SDL_HIDAPI_HapticDriverLg4ff_Resume,
    SDL_HIDAPI_HapticDriverLg4ff_StopEffects,
};

#endif //SDL_HAPTIC_HIDAPI_LG4FF
#endif //SDL_JOYSTICK_HIDAPI
