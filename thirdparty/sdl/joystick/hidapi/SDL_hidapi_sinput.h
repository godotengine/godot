/*
  Simple DirectMedia Layer
  Copyright (C) 2025 Mitchell Cairns <mitch.cairns@handheldlegend.com>

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

typedef enum
{
    SINPUT_ANALOGSTYLE_NONE,
    SINPUT_ANALOGSTYLE_LEFTONLY,
    SINPUT_ANALOGSTYLE_RIGHTONLY,
    SINPUT_ANALOGSTYLE_LEFTRIGHT,
    SINPUT_ANALOGSTYLE_MAX,
} SInput_AnalogStyleType;

typedef enum
{
    SINPUT_BUMPERSTYLE_NONE,
    SINPUT_BUMPERSTYLE_ONE,
    SINPUT_BUMPERSTYLE_TWO,
    SINPUT_BUMPERSTYLE_MAX,
} SInput_BumperStyleType;

typedef enum
{
    SINPUT_TRIGGERSTYLE_NONE,
    SINPUT_TRIGGERSTYLE_ANALOG,
    SINPUT_TRIGGERSTYLE_DIGITAL,
    SINPUT_TRIGGERSTYLE_DUALSTAGE,
    SINPUT_TRIGGERSTYLE_MAX,
} SInput_TriggerStyleType;

typedef enum
{
    SINPUT_PADDLESTYLE_NONE,
    SINPUT_PADDLESTYLE_TWO,
    SINPUT_PADDLESTYLE_FOUR,
    SINPUT_PADDLESTYLE_MAX,
} SInput_PaddleStyleType;

typedef enum
{
    SINPUT_METASTYLE_NONE,
    SINPUT_METASTYLE_BACK,
    SINPUT_METASTYLE_BACKGUIDE,
    SINPUT_METASTYLE_BACKGUIDESHARE,
    SINPUT_METASTYLE_MAX,
} SInput_MetaStyleType;

typedef enum
{
    SINPUT_TOUCHSTYLE_NONE,
    SINPUT_TOUCHSTYLE_SINGLE,
    SINPUT_TOUCHSTYLE_DOUBLE,
    SINPUT_TOUCHSTYLE_MAX,
} SInput_TouchStyleType;

typedef enum
{
    SINPUT_MISCSTYLE_NONE,
    SINPUT_MISCSTYLE_1,
    SINPUT_MISCSTYLE_2,
    SINPUT_MISCSTYLE_3,
    SINPUT_MISCSTYLE_4,
    SINPUT_MISCSTYLE_MAX,
} SInput_MiscStyleType;

typedef struct
{
    Uint16 analog_style;
    Uint16 bumper_style;
    Uint16 trigger_style;
    Uint16 paddle_style;
    Uint16 meta_style;
    Uint16 touch_style;
    Uint16 misc_style;
} SDL_SInputStyles_t;
