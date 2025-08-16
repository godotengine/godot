/*
 * Copyright © 2012 Daniel Stone
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author: Daniel Stone <daniel@fooishbar.org>
 */

#ifndef _XKBCOMMON_COMPAT_H
#define _XKBCOMMON_COMPAT_H

/**
 * Renamed keymap API.
 */
#define xkb_group_index_t xkb_layout_index_t
#define xkb_group_mask_t xkb_layout_mask_t
#define xkb_map_compile_flags xkb_keymap_compile_flags
#define XKB_GROUP_INVALID XKB_LAYOUT_INVALID

#define XKB_STATE_DEPRESSED \
    (XKB_STATE_MODS_DEPRESSED | XKB_STATE_LAYOUT_DEPRESSED)
#define XKB_STATE_LATCHED \
    (XKB_STATE_MODS_LATCHED | XKB_STATE_LAYOUT_LATCHED)
#define XKB_STATE_LOCKED \
    (XKB_STATE_MODS_LOCKED | XKB_STATE_LAYOUT_LOCKED)
#define XKB_STATE_EFFECTIVE \
    (XKB_STATE_DEPRESSED | XKB_STATE_LATCHED | XKB_STATE_LOCKED | \
     XKB_STATE_MODS_EFFECTIVE | XKB_STATE_LAYOUT_EFFECTIVE)

#define xkb_map_new_from_names(context, names, flags) \
        xkb_keymap_new_from_names(context, names, flags)
#define xkb_map_new_from_file(context, file, format, flags) \
        xkb_keymap_new_from_file(context, file, format, flags)
#define xkb_map_new_from_string(context, string, format, flags) \
        xkb_keymap_new_from_string(context, string, format, flags)
#define xkb_map_get_as_string(keymap) \
        xkb_keymap_get_as_string(keymap, XKB_KEYMAP_FORMAT_TEXT_V1)
#define xkb_map_ref(keymap) xkb_keymap_ref(keymap)
#define xkb_map_unref(keymap) xkb_keymap_unref(keymap)

#define xkb_map_num_mods(keymap) xkb_keymap_num_mods(keymap)
#define xkb_map_mod_get_name(keymap, idx) xkb_keymap_mod_get_name(keymap, idx)
#define xkb_map_mod_get_index(keymap, str) xkb_keymap_mod_get_index(keymap, str)
#define xkb_key_mod_index_is_consumed(state, key, mod) \
        xkb_state_mod_index_is_consumed(state, key, mod)
#define xkb_key_mod_mask_remove_consumed(state, key, modmask) \
        xkb_state_mod_mask_remove_consumed(state, key, modmask)

#define xkb_map_num_groups(keymap) xkb_keymap_num_layouts(keymap)
#define xkb_key_num_groups(keymap, key) \
        xkb_keymap_num_layouts_for_key(keymap, key)
#define xkb_map_group_get_name(keymap, idx) \
        xkb_keymap_layout_get_name(keymap, idx)
#define xkb_map_group_get_index(keymap, str) \
        xkb_keymap_layout_get_index(keymap, str)

#define xkb_map_num_leds(keymap) xkb_keymap_num_leds(keymap)
#define xkb_map_led_get_name(keymap, idx) xkb_keymap_led_get_name(keymap, idx)
#define xkb_map_led_get_index(keymap, str) \
        xkb_keymap_led_get_index(keymap, str)

#define xkb_key_repeats(keymap, key) xkb_keymap_key_repeats(keymap, key)

#define xkb_key_get_syms(state, key, syms_out) \
        xkb_state_key_get_syms(state, key, syms_out)

#define xkb_state_group_name_is_active(state, name, type) \
        xkb_state_layout_name_is_active(state, name, type)
#define xkb_state_group_index_is_active(state, idx, type) \
        xkb_state_layout_index_is_active(state, idx, type)

#define xkb_state_serialize_group(state, component) \
        xkb_state_serialize_layout(state, component)

#define xkb_state_get_map(state) xkb_state_get_keymap(state)

/* Not needed anymore, since there's NO_FLAGS. */
#define XKB_MAP_COMPILE_PLACEHOLDER XKB_KEYMAP_COMPILE_NO_FLAGS
#define XKB_MAP_COMPILE_NO_FLAGS XKB_KEYMAP_COMPILE_NO_FLAGS

#endif
