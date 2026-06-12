/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#ifndef usb_ids_h_
#define usb_ids_h_

// Definitions of useful USB VID/PID values

#define USB_VENDOR_8BITDO       0x2dc8
#define USB_VENDOR_AMAZON       0x1949
#define USB_VENDOR_APPLE        0x05ac
#define USB_VENDOR_ASTRO        0x9886
#define USB_VENDOR_ASUS         0x0b05
#define USB_VENDOR_BACKBONE     0x358a
#define USB_VENDOR_CRKD         0x3651
#define USB_VENDOR_GAMESIR      0x3537
#define USB_VENDOR_DRAGONRISE   0x0079
#define USB_VENDOR_FLYDIGI_V1   0x04b4
#define USB_VENDOR_FLYDIGI_V2   0x37d7
#define USB_VENDOR_GOOGLE       0x18d1
#define USB_VENDOR_HARMONIX     0x1bad
#define USB_VENDOR_HORI         0x0f0d
#define USB_VENDOR_HP           0x03f0
#define USB_VENDOR_HYPERKIN     0x2e24
#define USB_VENDOR_LOGITECH     0x046d
#define USB_VENDOR_MADCATZ      0x0738
#define USB_VENDOR_MAYFLASH     0x33df
#define USB_VENDOR_MICROSOFT    0x045e
#define USB_VENDOR_NACON        0x146b
#define USB_VENDOR_NACON_ALT    0x3285
#define USB_VENDOR_NINTENDO     0x057e
#define USB_VENDOR_NVIDIA       0x0955
#define USB_VENDOR_PDP          0x0e6f
#define USB_VENDOR_POWERA       0x24c6
#define USB_VENDOR_POWERA_ALT   0x20d6
#define USB_VENDOR_QANBA        0x2c22
#define USB_VENDOR_RAZER        0x1532
#define USB_VENDOR_RED_OCTANE   0x1430
#define USB_VENDOR_SAITEK       0x06a3
#define USB_VENDOR_SCEA         0x12ba
#define USB_VENDOR_SHANWAN      0x2563
#define USB_VENDOR_SHANWAN_ALT  0x20bc
#define USB_VENDOR_SONY         0x054c
#define USB_VENDOR_THRUSTMASTER 0x044f
#define USB_VENDOR_TURTLE_BEACH 0x10f5
#define USB_VENDOR_SWITCH       0x2563
#define USB_VENDOR_VALVE        0x28de
#define USB_VENDOR_ZEROPLUS     0x0c12
#define USB_VENDOR_RASPBERRYPI  0x2e8a // Commercial hardware from various companies are registered under this VID
#define USB_VENDOR_ZUIKI        0x33dd

#define USB_PRODUCT_8BITDO_SF30_PRO                       0x6000    // B + START
#define USB_PRODUCT_8BITDO_SF30_PRO_BT                    0x6100    // B + START
#define USB_PRODUCT_8BITDO_SN30_PRO                       0x6001    // B + START
#define USB_PRODUCT_8BITDO_SN30_PRO_BT                    0x6101    // B + START
#define USB_PRODUCT_8BITDO_PRO_2                          0x6003    // mode switch to D
#define USB_PRODUCT_8BITDO_PRO_2_BT                       0x6006    // mode switch to D
#define USB_PRODUCT_8BITDO_PRO_3                          0x6009    // mode switch to D
#define USB_PRODUCT_8BITDO_ULTIMATE2_WIRELESS             0x6012    // mode switch to BT
#define USB_PRODUCT_AMAZON_LUNA_CONTROLLER                0x0419
#define USB_PRODUCT_ASTRO_C40_XBOX360                     0x0024
#define USB_PRODUCT_BACKBONE_ONE_IOS                      0x0103
#define USB_PRODUCT_BACKBONE_ONE_IOS_PS5                  0x0104
#define USB_PRODUCT_BDA_XB1_CLASSIC                       0x581a
#define USB_PRODUCT_BDA_XB1_FIGHTPAD                      0x791a
#define USB_PRODUCT_BDA_XB1_SPECTRA_PRO                   0x592a
#define USB_PRODUCT_GOOGLE_STADIA_CONTROLLER              0x9400
#define USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER1            0x1843
#define USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER2            0x1844
#define USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER3            0x1846
#define USB_PRODUCT_FLYDIGI_V1_GAMEPAD                    0x2412
#define USB_PRODUCT_FLYDIGI_V2_APEX                       0x2501
#define USB_PRODUCT_FLYDIGI_V2_VADER                      0x2401
#define USB_PRODUCT_HARMONIX_WII_RB1_GUITAR               0x0004
#define USB_PRODUCT_HARMONIX_WII_RB1_DRUMS                0x0005
#define USB_PRODUCT_HARMONIX_WII_RB2_GUITAR               0x3010
#define USB_PRODUCT_HARMONIX_WII_RB2_DRUMS                0x3110
#define USB_PRODUCT_HARMONIX_WII_RB3_MPA_DRUMS_MODE       0x3138
#define USB_PRODUCT_HORI_FIGHTING_STICK_ALPHA_PS4         0x011c
#define USB_PRODUCT_HORI_FIGHTING_STICK_ALPHA_PS5         0x0184
#define USB_PRODUCT_HORI_FIGHTING_STICK_ALPHA_PS5         0x0184
#define USB_PRODUCT_HORI_STEAM_CONTROLLER                 0x01AB
#define USB_PRODUCT_HORI_STEAM_CONTROLLER_BT              0x0196
#define USB_PRODUCT_LOGITECH_F310                         0xc216
#define USB_PRODUCT_LOGITECH_CHILLSTREAM                  0xcad1
#define USB_PRODUCT_MADCATZ_SAITEK_SIDE_PANEL_CONTROL_DECK 0x2218
#define USB_PRODUCT_MADCATZ_XB1_STRATOCASTER_GUITAR       0x4161
#define USB_PRODUCT_MADCATZ_XB1_DRUM_KIT                  0x4262
#define USB_PRODUCT_MADCATZ_XB1_LEGACY_ADAPTER            0x4164
#define USB_PRODUCT_NACON_REVOLUTION_5_PRO_PS4_WIRELESS   0x0d16
#define USB_PRODUCT_NACON_REVOLUTION_5_PRO_PS4_WIRED      0x0d17
#define USB_PRODUCT_NACON_REVOLUTION_5_PRO_PS5_WIRELESS   0x0d18
#define USB_PRODUCT_NACON_REVOLUTION_5_PRO_PS5_WIRED      0x0d19
#define USB_PRODUCT_NACON_REVOLUTION_X_UNLIMITED_BT       0x0689
#define USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER             0x0337
#define USB_PRODUCT_NINTENDO_N64_CONTROLLER               0x2019
#define USB_PRODUCT_NINTENDO_SEGA_GENESIS_CONTROLLER      0x201e
#define USB_PRODUCT_NINTENDO_SNES_CONTROLLER              0x2017
#define USB_PRODUCT_NINTENDO_SWITCH_JOYCON_GRIP           0x200e
#define USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT           0x2006
#define USB_PRODUCT_NINTENDO_SWITCH_JOYCON_PAIR           0x2008 // Used by joycond
#define USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT          0x2007
#define USB_PRODUCT_NINTENDO_SWITCH_PRO                   0x2009
#define USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER  0x2073
#define USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT          0x2067
#define USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_PAIR          0x2068
#define USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT         0x2066
#define USB_PRODUCT_NINTENDO_SWITCH2_PRO                  0x2069
#define USB_PRODUCT_NINTENDO_WII_REMOTE                   0x0306
#define USB_PRODUCT_NINTENDO_WII_REMOTE2                  0x0330
#define USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V103         0x7210
#define USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V104         0x7214
#define USB_PRODUCT_PDP_ROCK_CANDY                        0x0246
#define USB_PRODUCT_PDP_REALMZ_WIRELESS                   0x018c
#define USB_PRODUCT_PDP_XB1_DRUM_KIT                      0x0171
#define USB_PRODUCT_PDP_XB1_JAGUAR_GUITAR                 0x0170
#define USB_PRODUCT_PDP_XB1_RIFFMASTER_GUITAR             0x0248
#define USB_PRODUCT_POWERA_MINI                           0x541a
#define USB_PRODUCT_RAZER_ATROX                           0x0a00
#define USB_PRODUCT_RAZER_KITSUNE                         0x1012
#define USB_PRODUCT_RAZER_PANTHERA                        0x0401
#define USB_PRODUCT_RAZER_PANTHERA_EVO                    0x1008
#define USB_PRODUCT_RAZER_RAIJU                           0x1000
#define USB_PRODUCT_RAZER_RAIJU_V3_PRO_PS5_WIRED          0x1024
#define USB_PRODUCT_RAZER_RAIJU_V3_PRO_PS5_WIRELESS       0x1026
#define USB_PRODUCT_RAZER_TOURNAMENT_EDITION_USB          0x1007
#define USB_PRODUCT_RAZER_TOURNAMENT_EDITION_BLUETOOTH    0x100a
#define USB_PRODUCT_RAZER_ULTIMATE_EDITION_USB            0x1004
#define USB_PRODUCT_RAZER_ULTIMATE_EDITION_BLUETOOTH      0x1009
#define USB_PRODUCT_RAZER_WOLVERINE_V2                    0x0a29
#define USB_PRODUCT_RAZER_WOLVERINE_V2_CHROMA             0x0a2e
#define USB_PRODUCT_RAZER_WOLVERINE_V2_PRO_PS5_WIRED      0x100b
#define USB_PRODUCT_RAZER_WOLVERINE_V2_PRO_PS5_WIRELESS   0x100c
#define USB_PRODUCT_RAZER_WOLVERINE_V2_PRO_XBOX_WIRED     0x1010
#define USB_PRODUCT_RAZER_WOLVERINE_V2_PRO_XBOX_WIRELESS  0x1011
#define USB_PRODUCT_RAZER_WOLVERINE_V3_PRO                0x0a3f
#define USB_PRODUCT_RED_OCTANE_XB1_GUITAR_HERO_LIVE_GUITAR 0x0170
#define USB_PRODUCT_SAITEK_CYBORG_V3                      0xf622
#define USB_PRODUCT_SCEA_PS3_GH_GUITAR                    0x0100
#define USB_PRODUCT_SCEA_PS3_GH_DRUMS                     0x0120
#define USB_PRODUCT_SCEA_PS3_RB_GUITAR                    0x0200
#define USB_PRODUCT_SCEA_PS3_RB_DRUMS                     0x0210
#define USB_PRODUCT_SCEA_PS3_RB3_MPA_DRUMS_MODE           0x0218
#define USB_PRODUCT_SCEA_PS3WIIU_GHLIVE                   0x074b
#define USB_PRODUCT_SHANWAN_DS3                           0x0523
#define USB_PRODUCT_SONY_DS3                              0x0268
#define USB_PRODUCT_SONY_DS4                              0x05c4
#define USB_PRODUCT_SONY_DS4_DONGLE                       0x0ba0
#define USB_PRODUCT_SONY_DS4_SLIM                         0x09cc
#define USB_PRODUCT_SONY_DS4_STRIKEPAD                    0x05c5
#define USB_PRODUCT_SONY_DS5                              0x0ce6
#define USB_PRODUCT_SONY_DS5_EDGE                         0x0df2
#define USB_PRODUCT_STEALTH_ULTRA_WIRED                   0x7073
#define USB_PRODUCT_SWITCH_RETROBIT_CONTROLLER            0x0575
#define USB_PRODUCT_THRUSTMASTER_ESWAPX_PRO_PS4           0xd00e
#define USB_PRODUCT_THRUSTMASTER_T_FLIGHT_HOTAS_ONE       0xb68c
#define USB_PRODUCT_VALVE_STEAM_CONTROLLER_DONGLE         0x1142
#define USB_PRODUCT_VALVE_STEAM_PROTEUS_DONGLE            0x1304
#define USB_PRODUCT_VALVE_STEAM_NEREID_DONGLE             0x1305
#define USB_PRODUCT_VICTRIX_FS_PRO                        0x0203
#define USB_PRODUCT_VICTRIX_FS_PRO_V2                     0x0207
#define USB_PRODUCT_XBOX360_XUSB_CONTROLLER               0x02a1 // XUSB driver software PID
#define USB_PRODUCT_XBOX360_WIRED_CONTROLLER              0x028e
#define USB_PRODUCT_XBOX360_WIRELESS_RECEIVER             0x0719
#define USB_PRODUCT_XBOX360_WIRELESS_RECEIVER_THIRDPARTY1 0x02a9
#define USB_PRODUCT_XBOX360_WIRELESS_RECEIVER_THIRDPARTY2 0x0291
#define USB_PRODUCT_XBOX_ONE_ADAPTIVE                     0x0b0a
#define USB_PRODUCT_XBOX_ONE_ADAPTIVE_BLUETOOTH           0x0b0c
#define USB_PRODUCT_XBOX_ONE_ADAPTIVE_BLE                 0x0b21
#define USB_PRODUCT_XBOX_ONE_ELITE_SERIES_1               0x02e3
#define USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2               0x0b00
#define USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2_BLUETOOTH     0x0b05
#define USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2_BLE           0x0b22
#define USB_PRODUCT_XBOX_ONE_S                            0x02ea
#define USB_PRODUCT_XBOX_ONE_S_REV1_BLUETOOTH             0x02e0
#define USB_PRODUCT_XBOX_ONE_S_REV2_BLUETOOTH             0x02fd
#define USB_PRODUCT_XBOX_ONE_S_REV2_BLE                   0x0b20
#define USB_PRODUCT_XBOX_SERIES_X                         0x0b12
#define USB_PRODUCT_XBOX_SERIES_X_BLE                     0x0b13
#define USB_PRODUCT_XBOX_ONE_XBOXGIP_CONTROLLER           0x02ff // XBOXGIP driver software PID
#define USB_PRODUCT_STEAM_VIRTUAL_GAMEPAD                 0x11ff
#define USB_PRODUCT_HANDHELDLEGEND_SINPUT_GENERIC         0x10c6
#define USB_PRODUCT_HANDHELDLEGEND_PROGCC                 0x10df
#define USB_PRODUCT_HANDHELDLEGEND_GCULTIMATE             0x10dd
#define USB_PRODUCT_BONZIRICHANNEL_FIREBIRD               0x10e0
#define USB_PRODUCT_ZUIKI_MASCON_PRO                      0x0006
#define USB_PRODUCT_ZUIKI_EVOTOP_UWB_DINPUT               0X001c
#define USB_PRODUCT_ZUIKI_EVOTOP_PC_DINPUT                0X001d
#define USB_PRODUCT_ZUIKI_EVOTOP_PC_BT                    0X0017
#define USB_PRODUCT_VOIDGAMING_PS4FIREBIRD                0x10e5

// USB usage pages
#define USB_USAGEPAGE_GENERIC_DESKTOP 0x0001
#define USB_USAGEPAGE_SIMULATION      0x0002
#define USB_USAGEPAGE_DEVICE_CONTROLS 0x0006
#define USB_USAGEPAGE_BUTTON          0x0009
#define USB_USAGEPAGE_CONSUMER        0x000C
#define USB_USAGEPAGE_VENDOR_FLYDIGI  0xFFA0

// USB usages for USAGE_PAGE_GENERIC_DESKTOP
#define USB_USAGE_GENERIC_POINTER             0x0001
#define USB_USAGE_GENERIC_MOUSE               0x0002
#define USB_USAGE_GENERIC_JOYSTICK            0x0004
#define USB_USAGE_GENERIC_GAMEPAD             0x0005
#define USB_USAGE_GENERIC_KEYBOARD            0x0006
#define USB_USAGE_GENERIC_KEYPAD              0x0007
#define USB_USAGE_GENERIC_MULTIAXISCONTROLLER 0x0008
#define USB_USAGE_GENERIC_X                   0x0030
#define USB_USAGE_GENERIC_Y                   0x0031
#define USB_USAGE_GENERIC_Z                   0x0032
#define USB_USAGE_GENERIC_RX                  0x0033
#define USB_USAGE_GENERIC_RY                  0x0034
#define USB_USAGE_GENERIC_RZ                  0x0035
#define USB_USAGE_GENERIC_SLIDER              0x0036
#define USB_USAGE_GENERIC_DIAL                0x0037
#define USB_USAGE_GENERIC_WHEEL               0x0038
#define USB_USAGE_GENERIC_HAT                 0x0039

// USB usages for USB_USAGEPAGE_SIMULATION
#define USB_USAGE_SIMULATION_ACCELERATOR      0x00C4
#define USB_USAGE_SIMULATION_BRAKE            0x00C5

// USB usages for USB_USAGEPAGE_DEVICE_CONTROLS
#define USB_USAGE_DEVICE_CONTROLS_BATTERY_STRENGTH 0x0020

// USB usages for USB_USAGEPAGE_CONSUMER
#define USB_USAGE_CONSUMER_ASSIGN_SELECTION   0x0081
#define USB_USAGE_CONSUMER_ORDER_MOVIE        0x0085
#define USB_USAGE_CONSUMER_RECORD             0x00B2
#define USB_USAGE_CONSUMER_AC_HOME            0x0223
#define USB_USAGE_CONSUMER_AC_BACK            0x0224

#define MAKE_USAGE(PAGE, USAGE) (((Uint32)PAGE) << 16 | USAGE)

/* Bluetooth SIG assigned Company Identifiers
   https://www.bluetooth.com/specifications/assigned-numbers/company-identifiers/ */
#define BLUETOOTH_VENDOR_AMAZON 0x0171

#define BLUETOOTH_PRODUCT_LUNA_CONTROLLER 0x0419

#endif // usb_ids_h_
