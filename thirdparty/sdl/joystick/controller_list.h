/*
  Copyright (C) Valve Corporation

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
#define MAKE_CONTROLLER_ID( nVID, nPID )	(unsigned int)( (unsigned int)nVID << 16 | (unsigned int)nPID )

static const ControllerDescription_t arrControllers[] = {
	{ MAKE_CONTROLLER_ID( 0x0079, 0x181a ), k_eControllerType_PS3Controller, NULL },	// Venom Arcade Stick
	{ MAKE_CONTROLLER_ID( 0x0079, 0x1844 ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x044f, 0xb315 ), k_eControllerType_PS3Controller, NULL },	// Firestorm Dual Analog 3
	{ MAKE_CONTROLLER_ID( 0x044f, 0xd007 ), k_eControllerType_PS3Controller, NULL },	// Thrustmaster wireless 3-1
	{ MAKE_CONTROLLER_ID( 0x046d, 0xcad1 ), k_eControllerType_PS3Controller, NULL },	// Logitech Chillstream
	//{ MAKE_CONTROLLER_ID( 0x046d, 0xc24f ), k_eControllerType_PS3Controller, NULL },	// Logitech G29 (PS3)
	{ MAKE_CONTROLLER_ID( 0x054c, 0x0268 ), k_eControllerType_PS3Controller, NULL },	// Sony PS3 Controller
	{ MAKE_CONTROLLER_ID( 0x056e, 0x200f ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x056e, 0x2013 ), k_eControllerType_PS3Controller, NULL },	// JC-U4113SBK
	{ MAKE_CONTROLLER_ID( 0x05b8, 0x1004 ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x05b8, 0x1006 ), k_eControllerType_PS3Controller, NULL },	// JC-U3412SBK
	{ MAKE_CONTROLLER_ID( 0x06a3, 0xf622 ), k_eControllerType_PS3Controller, NULL },	// Cyborg V3
	{ MAKE_CONTROLLER_ID( 0x0738, 0x3180 ), k_eControllerType_PS3Controller, NULL },	// Mad Catz Alpha PS3 mode
	{ MAKE_CONTROLLER_ID( 0x0738, 0x3250 ), k_eControllerType_PS3Controller, NULL },	// madcats fightpad pro ps3
	{ MAKE_CONTROLLER_ID( 0x0738, 0x3481 ), k_eControllerType_PS3Controller, NULL },	// Mad Catz FightStick TE 2+ PS3
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8180 ), k_eControllerType_PS3Controller, NULL },	// Mad Catz Alpha PS4 mode (no touchpad on device)
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8838 ), k_eControllerType_PS3Controller, NULL },	// Madcatz Fightstick Pro
	{ MAKE_CONTROLLER_ID( 0x0810, 0x0001 ), k_eControllerType_PS3Controller, NULL },	// actually ps2 - maybe break out later
	{ MAKE_CONTROLLER_ID( 0x0810, 0x0003 ), k_eControllerType_PS3Controller, NULL },	// actually ps2 - maybe break out later
	{ MAKE_CONTROLLER_ID( 0x0925, 0x0005 ), k_eControllerType_PS3Controller, NULL },	// Sony PS3 Controller
	{ MAKE_CONTROLLER_ID( 0x0925, 0x8866 ), k_eControllerType_PS3Controller, NULL },	// PS2 maybe break out later
	{ MAKE_CONTROLLER_ID( 0x0925, 0x8888 ), k_eControllerType_PS3Controller, NULL },	// Actually ps2 -maybe break out later Lakeview Research WiseGroup Ltd, MP-8866 Dual Joypad
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0109 ), k_eControllerType_PS3Controller, NULL },	// PDP Versus Fighting Pad
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x011e ), k_eControllerType_PS3Controller, NULL },	// Rock Candy PS4
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0128 ), k_eControllerType_PS3Controller, NULL },	// Rock Candy PS3
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0214 ), k_eControllerType_PS3Controller, NULL },	// afterglow ps3
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x1314 ), k_eControllerType_PS3Controller, NULL },	// PDP Afterglow Wireless PS3 controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x6302 ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x0e8f, 0x0008 ), k_eControllerType_PS3Controller, NULL },	// Green Asia
	{ MAKE_CONTROLLER_ID( 0x0e8f, 0x3075 ), k_eControllerType_PS3Controller, NULL },	// SpeedLink Strike FX
	{ MAKE_CONTROLLER_ID( 0x0e8f, 0x310d ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0009 ), k_eControllerType_PS3Controller, NULL },	// HORI BDA GP1
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x004d ), k_eControllerType_PS3Controller, NULL },	// Horipad 3
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x005f ), k_eControllerType_PS3Controller, NULL },	// HORI Fighting Commander 4 PS3
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x006a ), k_eControllerType_PS3Controller, NULL },	// Real Arcade Pro 4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x006e ), k_eControllerType_PS3Controller, NULL },	// HORI horipad4 ps3
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0085 ), k_eControllerType_PS3Controller, NULL },	// HORI Fighting Commander PS3
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0086 ), k_eControllerType_PS3Controller, NULL },	// HORI Fighting Commander PC (Uses the Xbox 360 protocol, but has PS3 buttons)
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0088 ), k_eControllerType_PS3Controller, NULL },	// HORI Fighting Stick mini 4
	{ MAKE_CONTROLLER_ID( 0x0f30, 0x1100 ), k_eControllerType_PS3Controller, NULL },	// Qanba Q1 fight stick
	{ MAKE_CONTROLLER_ID( 0x11ff, 0x3331 ), k_eControllerType_PS3Controller, NULL },	// SRXJ-PH2400
	{ MAKE_CONTROLLER_ID( 0x1345, 0x1000 ), k_eControllerType_PS3Controller, NULL },	// PS2 ACME GA-D5
	{ MAKE_CONTROLLER_ID( 0x1345, 0x6005 ), k_eControllerType_PS3Controller, NULL },	// ps2 maybe break out later
	{ MAKE_CONTROLLER_ID( 0x146b, 0x5500 ), k_eControllerType_PS3Controller, NULL },	// From SDL
	{ MAKE_CONTROLLER_ID( 0x1a34, 0x0836 ), k_eControllerType_PS3Controller, NULL },	// Afterglow PS3
	{ MAKE_CONTROLLER_ID( 0x20bc, 0x5500 ), k_eControllerType_PS3Controller, NULL },	// ShanWan PS3
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x576d ), k_eControllerType_PS3Controller, NULL },	// Power A PS3
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xca6d ), k_eControllerType_PS3Controller, NULL },	// BDA Pro Ex
	{ MAKE_CONTROLLER_ID( 0x2563, 0x0523 ), k_eControllerType_PS3Controller, NULL },	// Digiflip GP006
	{ MAKE_CONTROLLER_ID( 0x2563, 0x0575 ), k_eControllerType_PS3Controller, "Retro-bit Controller" },	// SWITCH CO., LTD. Retro-bit Controller
	{ MAKE_CONTROLLER_ID( 0x25f0, 0x83c3 ), k_eControllerType_PS3Controller, NULL },	// gioteck vx2
	{ MAKE_CONTROLLER_ID( 0x25f0, 0xc121 ), k_eControllerType_PS3Controller, NULL },	//
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2003 ), k_eControllerType_PS3Controller, NULL },	// Qanba Drone
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2302 ), k_eControllerType_PS3Controller, NULL },	// Qanba Obsidian
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2502 ), k_eControllerType_PS3Controller, NULL },	// Qanba Dragon
	{ MAKE_CONTROLLER_ID( 0x8380, 0x0003 ), k_eControllerType_PS3Controller, NULL },	// BTP 2163
	{ MAKE_CONTROLLER_ID( 0x8888, 0x0308 ), k_eControllerType_PS3Controller, NULL },	// Sony PS3 Controller

	{ MAKE_CONTROLLER_ID( 0x0079, 0x181b ), k_eControllerType_PS4Controller, NULL },	// Venom Arcade Stick - XXX:this may not work and may need to be called a ps3 controller
	//{ MAKE_CONTROLLER_ID( 0x046d, 0xc260 ), k_eControllerType_PS4Controller, NULL },	// Logitech G29 (PS4)
	{ MAKE_CONTROLLER_ID( 0x044f, 0xd00e ), k_eControllerType_PS4Controller, NULL },	// Thrustmaster Eswap Pro - No gyro and lightbar doesn't change color. Works otherwise
	{ MAKE_CONTROLLER_ID( 0x054c, 0x05c4 ), k_eControllerType_PS4Controller, NULL },	// Sony PS4 Controller
	{ MAKE_CONTROLLER_ID( 0x054c, 0x05c5 ), k_eControllerType_PS4Controller, NULL },	// STRIKEPAD PS4 Grip Add-on
	{ MAKE_CONTROLLER_ID( 0x054c, 0x09cc ), k_eControllerType_PS4Controller, NULL },	// Sony PS4 Slim Controller
	{ MAKE_CONTROLLER_ID( 0x054c, 0x0ba0 ), k_eControllerType_PS4Controller, NULL },	// Sony PS4 Controller (Wireless dongle)
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8250 ), k_eControllerType_PS4Controller, NULL },	// Mad Catz FightPad Pro PS4
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8384 ), k_eControllerType_PS4Controller, NULL },	// Mad Catz FightStick TE S+ PS4
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8480 ), k_eControllerType_PS4Controller, NULL },	// Mad Catz FightStick TE 2 PS4
	{ MAKE_CONTROLLER_ID( 0x0738, 0x8481 ), k_eControllerType_PS4Controller, NULL },	// Mad Catz FightStick TE 2+ PS4
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0e10 ), k_eControllerType_PS4Controller, NULL },	// Armor Armor 3 Pad PS4
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0e13 ), k_eControllerType_PS4Controller, NULL },	// ZEROPLUS P4 Wired Gamepad
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0e15 ), k_eControllerType_PS4Controller, NULL },	// Game:Pad 4
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0e20 ), k_eControllerType_PS4Controller, NULL },	// Brook Mars Controller - needs FW update to show up as Ps4 controller on PC. Has Gyro but touchpad is a single button.
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0ef6 ), k_eControllerType_PS4Controller, NULL },	// Hitbox Arcade Stick
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x1cf6 ), k_eControllerType_PS4Controller, NULL },	// EMIO PS4 Elite Controller
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x1e10 ), k_eControllerType_PS4Controller, NULL },	// P4 Wired Gamepad generic knock off - lightbar but not trackpad or gyro
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0203 ), k_eControllerType_PS4Controller, NULL },	// Victrix Pro FS (PS4 peripheral but no trackpad/lightbar)
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0207 ), k_eControllerType_PS4Controller, NULL },	// Victrix Pro FS V2 w/ Touchpad for PS4
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x020a ), k_eControllerType_PS4Controller, NULL },	// Victrix Pro FS PS4/PS5 (PS4 mode)
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0055 ), k_eControllerType_PS4Controller, NULL },	// HORIPAD 4 FPS
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x005e ), k_eControllerType_PS4Controller, NULL },	// HORI Fighting Commander 4 PS4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0066 ), k_eControllerType_PS4Controller, NULL },	// HORIPAD 4 FPS Plus
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0084 ), k_eControllerType_PS4Controller, NULL },	// HORI Fighting Commander PS4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0087 ), k_eControllerType_PS4Controller, NULL },	// HORI Fighting Stick mini 4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x008a ), k_eControllerType_PS4Controller, NULL },	// HORI Real Arcade Pro 4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x009c ), k_eControllerType_PS4Controller, NULL },	// HORI TAC PRO mousething
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00a0 ), k_eControllerType_PS4Controller, NULL },	// HORI TAC4 mousething
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00ed ), k_eControllerType_XInputPS4Controller, NULL },	// Hori Fighting Stick mini 4 kai - becomes an Xbox 360 controller on PC
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00ee ), k_eControllerType_PS4Controller, NULL },	// Hori mini wired https://www.playstation.com/en-us/explore/accessories/gaming-controllers/mini-wired-gamepad/
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x011c ), k_eControllerType_PS4Controller, NULL },	// Hori Fighting Stick α
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0123 ), k_eControllerType_PS4Controller, NULL },	// HORI Wireless Controller Light (Japan only) - only over bt- over usb is xbox and pid 0x0124
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0162 ), k_eControllerType_PS4Controller, NULL },	// HORI Fighting Commander OCTA
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0164 ), k_eControllerType_XInputPS4Controller, NULL },	// HORI Fighting Commander OCTA
	{ MAKE_CONTROLLER_ID( 0x11c0, 0x4001 ), k_eControllerType_PS4Controller, NULL },	// "PS4 Fun Controller" added from user log
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0603 ), k_eControllerType_XInputPS4Controller, NULL },	// Nacon PS4 Compact Controller
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0604 ), k_eControllerType_XInputPS4Controller, NULL },	// NACON Daija Arcade Stick
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0605 ), k_eControllerType_XInputPS4Controller, NULL },	// NACON PS4 controller in Xbox mode - might also be other bigben brand xbox controllers
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0606 ), k_eControllerType_XInputPS4Controller, NULL },	// NACON Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0609 ), k_eControllerType_XInputPS4Controller, NULL },	// NACON Wireless Controller for PS4
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d01 ), k_eControllerType_PS4Controller, NULL },	// Nacon Revolution Pro Controller - has gyro
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d02 ), k_eControllerType_PS4Controller, NULL },	// Nacon Revolution Pro Controller v2 - has gyro
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d06 ), k_eControllerType_PS4Controller, NULL },	// NACON Asymmetric Controller Wireless Dongle -- show up as ps4 until you connect controller to it then it reboots into Xbox controller with different vvid/pid
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d08 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution Unlimited Wireless Dongle
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d09 ), k_eControllerType_PS4Controller, NULL },	// NACON Daija Fight Stick - touchpad but no gyro/rumble
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d10 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution Infinite - has gyro
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d10 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution Unlimited
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0d13 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution Pro Controller 3
	{ MAKE_CONTROLLER_ID( 0x146b, 0x1103 ), k_eControllerType_PS4Controller, NULL },	// NACON Asymmetric Controller -- on windows this doesn't enumerate
	{ MAKE_CONTROLLER_ID( 0x1532, 0X0401 ), k_eControllerType_PS4Controller, NULL },	// Razer Panthera PS4 Controller
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1000 ), k_eControllerType_PS4Controller, NULL },	// Razer Raiju PS4 Controller
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1004 ), k_eControllerType_PS4Controller, NULL },	// Razer Raiju 2 Ultimate USB
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1007 ), k_eControllerType_PS4Controller, NULL },	// Razer Raiju 2 Tournament edition USB
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1008 ), k_eControllerType_PS4Controller, NULL },	// Razer Panthera Evo Fightstick
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1009 ), k_eControllerType_PS4Controller, NULL },	// Razer Raiju 2 Ultimate BT
	{ MAKE_CONTROLLER_ID( 0x1532, 0x100A ), k_eControllerType_PS4Controller, NULL },	// Razer Raiju 2 Tournament edition BT
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1100 ), k_eControllerType_PS4Controller, NULL },	// Razer RAION Fightpad - Trackpad, no gyro, lightbar hardcoded to green
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x792a ), k_eControllerType_PS4Controller, NULL },	// PowerA Fusion Fight Pad
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2000 ), k_eControllerType_PS4Controller, NULL },	// Qanba Drone
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2300 ), k_eControllerType_PS4Controller, NULL },	// Qanba Obsidian
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2303 ), k_eControllerType_XInputPS4Controller, NULL },	// Qanba Obsidian Arcade Joystick
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2500 ), k_eControllerType_PS4Controller, NULL },	// Qanba Dragon
	{ MAKE_CONTROLLER_ID( 0x2c22, 0x2503 ), k_eControllerType_XInputPS4Controller, NULL },	// Qanba Dragon Arcade Joystick
	{ MAKE_CONTROLLER_ID( 0x3285, 0x0d16 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution 5 Pro (PS4 mode with dongle)
	{ MAKE_CONTROLLER_ID( 0x3285, 0x0d17 ), k_eControllerType_PS4Controller, NULL },	// NACON Revolution 5 Pro (PS4 mode wired)
	{ MAKE_CONTROLLER_ID( 0x7545, 0x0104 ), k_eControllerType_PS4Controller, NULL },	// Armor 3 or Level Up Cobra - At least one variant has gyro
    { MAKE_CONTROLLER_ID (0x9886, 0x0024 ), k_eControllerType_XInputPS4Controller, NULL },  // Astro C40 in Xbox 360 mode
	{ MAKE_CONTROLLER_ID( 0x9886, 0x0025 ), k_eControllerType_PS4Controller, NULL },	// Astro C40
	// Removing the Giotek because there were a bunch of help tickets from users w/ issues including from non-PS4 controller users. This VID/PID is probably used in different FW's
//	{ MAKE_CONTROLLER_ID( 0x7545, 0x1122 ), k_eControllerType_PS4Controller, NULL },	// Giotek VX4 - trackpad/gyro don't work. Had to not filter on interface info. Light bar is flaky, but works.

	{ MAKE_CONTROLLER_ID( 0x054c, 0x0ce6 ), k_eControllerType_PS5Controller, NULL },	// Sony DualSense Controller
	{ MAKE_CONTROLLER_ID( 0x054c, 0x0df2 ), k_eControllerType_PS5Controller, NULL },	// Sony DualSense Edge Controller
	{ MAKE_CONTROLLER_ID( 0x054c, 0x0e5f ), k_eControllerType_PS5Controller, NULL },	// Access Controller for PS5
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0209 ), k_eControllerType_PS5Controller, NULL },	// Victrix Pro FS PS4/PS5 (PS5 mode)
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0163 ), k_eControllerType_PS5Controller, NULL },	// HORI Fighting Commander OCTA
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0184 ), k_eControllerType_PS5Controller, NULL },	// Hori Fighting Stick α
	{ MAKE_CONTROLLER_ID( 0x1532, 0x100b ), k_eControllerType_PS5Controller, NULL },	// Razer Wolverine V2 Pro (Wired)
	{ MAKE_CONTROLLER_ID( 0x1532, 0x100c ), k_eControllerType_PS5Controller, NULL },	// Razer Wolverine V2 Pro (Wireless)
	{ MAKE_CONTROLLER_ID( 0x1532, 0x1012 ), k_eControllerType_PS5Controller, NULL },	// Razer Kitsune
	{ MAKE_CONTROLLER_ID( 0x3285, 0x0d18 ), k_eControllerType_PS5Controller, NULL },	// NACON Revolution 5 Pro (PS5 mode with dongle)
	{ MAKE_CONTROLLER_ID( 0x3285, 0x0d19 ), k_eControllerType_PS5Controller, NULL },	// NACON Revolution 5 Pro (PS5 mode wired)
	{ MAKE_CONTROLLER_ID( 0x358a, 0x0104 ), k_eControllerType_PS5Controller, NULL },	// Backbone One PlayStation Edition for iOS

	{ MAKE_CONTROLLER_ID( 0x0079, 0x0006 ), k_eControllerType_UnknownNonSteamController, NULL },	// DragonRise Generic USB PCB, sometimes configured as a PC Twin Shock Controller - looks like a DS3 but the face buttons are 1-4 instead of symbols

	{ MAKE_CONTROLLER_ID( 0x0079, 0x18d4 ), k_eControllerType_XBox360Controller, NULL },	// GPD Win 2 X-Box Controller
	{ MAKE_CONTROLLER_ID( 0x03eb, 0xff02 ), k_eControllerType_XBox360Controller, NULL },	// Wooting Two
	{ MAKE_CONTROLLER_ID( 0x044f, 0xb326 ), k_eControllerType_XBox360Controller, NULL },	// Thrustmaster Gamepad GP XID
	{ MAKE_CONTROLLER_ID( 0x045e, 0x028e ), k_eControllerType_XBox360Controller, "Xbox 360 Controller" },          // Microsoft Xbox 360 Wired Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x028f ), k_eControllerType_XBox360Controller, "Xbox 360 Controller" },          // Microsoft Xbox 360 Play and Charge Cable
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0291 ), k_eControllerType_XBox360Controller, "Xbox 360 Wireless Controller" }, // X-box 360 Wireless Receiver (third party knockoff)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02a0 ), k_eControllerType_XBox360Controller, NULL },                           // Microsoft Xbox 360 Big Button IR
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02a1 ), k_eControllerType_XBox360Controller, "Xbox 360 Wireless Controller" }, // Microsoft Xbox 360 Wireless Controller with XUSB driver on Windows
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02a9 ), k_eControllerType_XBox360Controller, "Xbox 360 Wireless Controller" }, // X-box 360 Wireless Receiver (third party knockoff)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0719 ), k_eControllerType_XBox360Controller, "Xbox 360 Wireless Controller" }, // Microsoft Xbox 360 Wireless Receiver
	{ MAKE_CONTROLLER_ID( 0x046d, 0xc21d ), k_eControllerType_XBox360Controller, NULL },	// Logitech Gamepad F310
	{ MAKE_CONTROLLER_ID( 0x046d, 0xc21e ), k_eControllerType_XBox360Controller, NULL },	// Logitech Gamepad F510
	{ MAKE_CONTROLLER_ID( 0x046d, 0xc21f ), k_eControllerType_XBox360Controller, NULL },	// Logitech Gamepad F710
	{ MAKE_CONTROLLER_ID( 0x046d, 0xc242 ), k_eControllerType_XBox360Controller, NULL },	// Logitech Chillstream Controller
	{ MAKE_CONTROLLER_ID( 0x056e, 0x2004 ), k_eControllerType_XBox360Controller, NULL },	// Elecom JC-U3613M
// This isn't actually an Xbox 360 controller, it just looks like one
//	{ MAKE_CONTROLLER_ID( 0x06a3, 0xf51a ), k_eControllerType_XBox360Controller, NULL },	// Saitek P3600
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4716 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Wired Xbox 360 Controller
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4718 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Street Fighter IV FightStick SE
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4726 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Xbox 360 Controller
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4728 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Street Fighter IV FightPad
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4736 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz MicroCon Gamepad
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4738 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Wired Xbox 360 Controller (SFIV)
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4740 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Beat Pad
	{ MAKE_CONTROLLER_ID( 0x0738, 0xb726 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Xbox controller - MW2
	{ MAKE_CONTROLLER_ID( 0x0738, 0xbeef ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz JOYTECH NEO SE Advanced GamePad
	{ MAKE_CONTROLLER_ID( 0x0738, 0xcb02 ), k_eControllerType_XBox360Controller, NULL },	// Saitek Cyborg Rumble Pad - PC/Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0738, 0xcb03 ), k_eControllerType_XBox360Controller, NULL },	// Saitek P3200 Rumble Pad - PC/Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0738, 0xf738 ), k_eControllerType_XBox360Controller, NULL },	// Super SFIV FightStick TE S
	{ MAKE_CONTROLLER_ID( 0x0955, 0x7210 ), k_eControllerType_XBox360Controller, NULL },	// Nvidia Shield local controller
	{ MAKE_CONTROLLER_ID( 0x0955, 0xb400 ), k_eControllerType_XBox360Controller, NULL },	// NVIDIA Shield streaming controller
	{ MAKE_CONTROLLER_ID( 0x0b05, 0x1b4c ), k_eControllerType_XBox360Controller, NULL },	// ASUS ROG Ally X built-in controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0105 ), k_eControllerType_XBox360Controller, NULL },	// HSM3 Xbox360 dancepad
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0113 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Afterglow" },	// PDP Afterglow Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x011f ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Rock Candy" },	// PDP Rock Candy Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0125 ), k_eControllerType_XBox360Controller, "PDP INJUSTICE FightStick" },	// PDP INJUSTICE FightStick for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0127 ), k_eControllerType_XBox360Controller, "PDP INJUSTICE FightPad" },	// PDP INJUSTICE FightPad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0131 ), k_eControllerType_XBox360Controller, "PDP EA Soccer Controller" },	// PDP EA Soccer Gamepad
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0133 ), k_eControllerType_XBox360Controller, "PDP Battlefield 4 Controller" },	// PDP Battlefield 4 Gamepad
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0143 ), k_eControllerType_XBox360Controller, "PDP MK X Fight Stick" },	// PDP MK X Fight Stick for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0147 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Marvel Controller" },	// PDP Marvel Controller for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0201 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Controller" },	// PDP Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0213 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Afterglow" },	// PDP Afterglow Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x021f ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Rock Candy" },	// PDP Rock Candy Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0301 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Controller" },	// PDP Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0313 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Afterglow" },	// PDP Afterglow Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0314 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Afterglow" },	// PDP Afterglow Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0401 ), k_eControllerType_XBox360Controller, "PDP Xbox 360 Controller" },	// PDP Gamepad for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0413 ), k_eControllerType_XBox360Controller, NULL },	// PDP Afterglow AX.1 (unlisted)
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0501 ), k_eControllerType_XBox360Controller, NULL },	// PDP Xbox 360 Controller (unlisted)
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0xf900 ), k_eControllerType_XBox360Controller, NULL },	// PDP Afterglow AX.1 (unlisted)
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x000a ), k_eControllerType_XBox360Controller, NULL },	// Hori Co. DOA4 FightStick
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x000c ), k_eControllerType_XBox360Controller, NULL },	// Hori PadEX Turbo
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x000d ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Stick EX2
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0016 ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro.EX
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x001b ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro VX
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x008c ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro 4
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00db ), k_eControllerType_XBox360Controller, "HORI Slime Controller" },	// Hori Dragon Quest Slime Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x011e ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Stick α
	{ MAKE_CONTROLLER_ID( 0x1038, 0x1430 ), k_eControllerType_XBox360Controller, "SteelSeries Stratus Duo" },	// SteelSeries Stratus Duo
	{ MAKE_CONTROLLER_ID( 0x1038, 0x1431 ), k_eControllerType_XBox360Controller, "SteelSeries Stratus Duo" },	// SteelSeries Stratus Duo
	{ MAKE_CONTROLLER_ID( 0x1038, 0xb360 ), k_eControllerType_XBox360Controller, NULL },	// SteelSeries Nimbus/Stratus XL
	{ MAKE_CONTROLLER_ID( 0x11c9, 0x55f0 ), k_eControllerType_XBox360Controller, NULL },	// Nacon GC-100XF
	{ MAKE_CONTROLLER_ID( 0x12ab, 0x0004 ), k_eControllerType_XBox360Controller, NULL },	// Honey Bee Xbox360 dancepad
	{ MAKE_CONTROLLER_ID( 0x12ab, 0x0301 ), k_eControllerType_XBox360Controller, NULL },	// PDP AFTERGLOW AX.1
	{ MAKE_CONTROLLER_ID( 0x12ab, 0x0303 ), k_eControllerType_XBox360Controller, NULL },	// Mortal Kombat Klassic FightStick
	{ MAKE_CONTROLLER_ID( 0x1430, 0x02a0 ), k_eControllerType_XBox360Controller, NULL },	// RedOctane Controller Adapter
	{ MAKE_CONTROLLER_ID( 0x1430, 0x4748 ), k_eControllerType_XBox360Controller, NULL },	// RedOctane Guitar Hero X-plorer
	{ MAKE_CONTROLLER_ID( 0x1430, 0xf801 ), k_eControllerType_XBox360Controller, NULL },	// RedOctane Controller
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0601 ), k_eControllerType_XBox360Controller, NULL },	// BigBen Interactive XBOX 360 Controller
//	{ MAKE_CONTROLLER_ID( 0x1532, 0x0037 ), k_eControllerType_XBox360Controller, NULL },	// Razer Sabertooth
	{ MAKE_CONTROLLER_ID( 0x15e4, 0x3f00 ), k_eControllerType_XBox360Controller, NULL },	// Power A Mini Pro Elite
	{ MAKE_CONTROLLER_ID( 0x15e4, 0x3f0a ), k_eControllerType_XBox360Controller, NULL },	// Xbox Airflo wired controller
	{ MAKE_CONTROLLER_ID( 0x15e4, 0x3f10 ), k_eControllerType_XBox360Controller, NULL },	// Batarang Xbox 360 controller
	{ MAKE_CONTROLLER_ID( 0x162e, 0xbeef ), k_eControllerType_XBox360Controller, NULL },	// Joytech Neo-Se Take2
	{ MAKE_CONTROLLER_ID( 0x1689, 0xfd00 ), k_eControllerType_XBox360Controller, NULL },	// Razer Onza Tournament Edition
	{ MAKE_CONTROLLER_ID( 0x1689, 0xfd01 ), k_eControllerType_XBox360Controller, NULL },	// Razer Onza Classic Edition
	{ MAKE_CONTROLLER_ID( 0x1689, 0xfe00 ), k_eControllerType_XBox360Controller, NULL },	// Razer Sabertooth
	{ MAKE_CONTROLLER_ID( 0x1949, 0x041a ), k_eControllerType_XBox360Controller, "Amazon Luna Controller" },	// Amazon Luna Controller
	{ MAKE_CONTROLLER_ID( 0x1bad, 0x0002 ), k_eControllerType_XBox360Controller, NULL },	// Harmonix Rock Band Guitar
	{ MAKE_CONTROLLER_ID( 0x1bad, 0x0003 ), k_eControllerType_XBox360Controller, NULL },	// Harmonix Rock Band Drumkit
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf016 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Xbox 360 Controller
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf018 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Street Fighter IV SE Fighting Stick
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf019 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Brawlstick for Xbox 360
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf021 ), k_eControllerType_XBox360Controller, NULL },	// Mad Cats Ghost Recon FS GamePad
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf023 ), k_eControllerType_XBox360Controller, NULL },	// MLG Pro Circuit Controller (Xbox)
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf025 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Call Of Duty
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf027 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz FPS Pro
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf028 ), k_eControllerType_XBox360Controller, NULL },	// Street Fighter IV FightPad
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf02e ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Fightpad
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf036 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz MicroCon GamePad Pro
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf038 ), k_eControllerType_XBox360Controller, NULL },	// Street Fighter IV FightStick TE
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf039 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz MvC2 TE
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf03a ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz SFxT Fightstick Pro
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf03d ), k_eControllerType_XBox360Controller, NULL },	// Street Fighter IV Arcade Stick TE - Chun Li
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf03e ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz MLG FightStick TE
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf03f ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz FightStick SoulCaliber
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf042 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz FightStick TES+
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf080 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz FightStick TE2
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf501 ), k_eControllerType_XBox360Controller, NULL },	// HoriPad EX2 Turbo
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf502 ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro.VX SA
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf503 ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Stick VX
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf504 ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro. EX
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf505 ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Stick EX2B
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf506 ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro.EX Premium VLX
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf900 ), k_eControllerType_XBox360Controller, NULL },	// Harmonix Xbox 360 Controller
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf901 ), k_eControllerType_XBox360Controller, NULL },	// Gamestop Xbox 360 Controller
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf902 ), k_eControllerType_XBox360Controller, NULL },	// Mad Catz Gamepad2
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf903 ), k_eControllerType_XBox360Controller, NULL },	// Tron Xbox 360 controller
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf904 ), k_eControllerType_XBox360Controller, NULL },	// PDP Versus Fighting Pad
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xf906 ), k_eControllerType_XBox360Controller, NULL },	// MortalKombat FightStick
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xfa01 ), k_eControllerType_XBox360Controller, NULL },	// MadCatz GamePad
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xfd00 ), k_eControllerType_XBox360Controller, NULL },	// Razer Onza TE
	{ MAKE_CONTROLLER_ID( 0x1bad, 0xfd01 ), k_eControllerType_XBox360Controller, NULL },	// Razer Onza
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5000 ), k_eControllerType_XBox360Controller, NULL },	// Razer Atrox Arcade Stick
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5300 ), k_eControllerType_XBox360Controller, NULL },	// PowerA MINI PROEX Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5303 ), k_eControllerType_XBox360Controller, NULL },	// Xbox Airflo wired controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x530a ), k_eControllerType_XBox360Controller, NULL },	// Xbox 360 Pro EX Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x531a ), k_eControllerType_XBox360Controller, NULL },	// PowerA Pro Ex
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5397 ), k_eControllerType_XBox360Controller, NULL },	// FUS1ON Tournament Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5500 ), k_eControllerType_XBox360Controller, NULL },	// Hori XBOX 360 EX 2 with Turbo
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5501 ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro VX-SA
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5502 ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Stick VX Alt
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5503 ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Edge
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5506 ), k_eControllerType_XBox360Controller, NULL },	// Hori SOULCALIBUR V Stick
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x550d ), k_eControllerType_XBox360Controller, NULL },	// Hori GEM Xbox controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x550e ), k_eControllerType_XBox360Controller, NULL },	// Hori Real Arcade Pro V Kai 360
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5508 ), k_eControllerType_XBox360Controller, NULL },	// Hori PAD A
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5510 ), k_eControllerType_XBox360Controller, NULL },	// Hori Fighting Commander ONE
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5b00 ), k_eControllerType_XBox360Controller, NULL },	// ThrustMaster Ferrari Italia 458 Racing Wheel
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5b02 ), k_eControllerType_XBox360Controller, NULL },	// Thrustmaster, Inc. GPX Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5b03 ), k_eControllerType_XBox360Controller, NULL },	// Thrustmaster Ferrari 458 Racing Wheel
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x5d04 ), k_eControllerType_XBox360Controller, NULL },	// Razer Sabertooth
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfafa ), k_eControllerType_XBox360Controller, NULL },	// Aplay Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfafb ), k_eControllerType_XBox360Controller, NULL },	// Aplay Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfafc ), k_eControllerType_XBox360Controller, NULL },	// Afterglow Gamepad 1
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfafd ), k_eControllerType_XBox360Controller, NULL },	// Afterglow Gamepad 3
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfafe ), k_eControllerType_XBox360Controller, NULL },	// Rock Candy Gamepad for Xbox 360

	{ MAKE_CONTROLLER_ID( 0x03f0, 0x0495 ), k_eControllerType_XBoxOneController, NULL },	// HP HyperX Clutch Gladiate
	{ MAKE_CONTROLLER_ID( 0x044f, 0xd012 ), k_eControllerType_XBoxOneController, NULL },	// ThrustMaster eSwap PRO Controller Xbox
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02d1 ), k_eControllerType_XBoxOneController, "Xbox One Controller" },         // Microsoft Xbox One Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02dd ), k_eControllerType_XBoxOneController, "Xbox One Controller" },         // Microsoft Xbox One Controller (Firmware 2015)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02e0 ), k_eControllerType_XBoxOneController, "Xbox One S Controller" },       // Microsoft Xbox One S Controller (Bluetooth)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02e3 ), k_eControllerType_XBoxOneController, "Xbox One Elite Controller" },   // Microsoft Xbox One Elite Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02ea ), k_eControllerType_XBoxOneController, "Xbox One S Controller" },       // Microsoft Xbox One S Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02fd ), k_eControllerType_XBoxOneController, "Xbox One S Controller" },       // Microsoft Xbox One S Controller (Bluetooth)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02ff ), k_eControllerType_XBoxOneController, "Xbox One Controller" },         // Microsoft Xbox One Controller with XBOXGIP driver on Windows
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b00 ), k_eControllerType_XBoxOneController, "Xbox One Elite 2 Controller" }, // Microsoft Xbox One Elite Series 2 Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b05 ), k_eControllerType_XBoxOneController, "Xbox One Elite 2 Controller" }, // Microsoft Xbox One Elite Series 2 Controller (Bluetooth)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b0a ), k_eControllerType_XBoxOneController, "Xbox Adaptive Controller" },    // Microsoft Xbox Adaptive Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b0c ), k_eControllerType_XBoxOneController, "Xbox Adaptive Controller" },    // Microsoft Xbox Adaptive Controller (Bluetooth)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b12 ), k_eControllerType_XBoxOneController, "Xbox Series X Controller" },    // Microsoft Xbox Series X Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b13 ), k_eControllerType_XBoxOneController, "Xbox Series X Controller" },    // Microsoft Xbox Series X Controller (BLE)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b20 ), k_eControllerType_XBoxOneController, "Xbox One S Controller" },       // Microsoft Xbox One S Controller (BLE)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b21 ), k_eControllerType_XBoxOneController, "Xbox Adaptive Controller" },    // Microsoft Xbox Adaptive Controller (BLE)
	{ MAKE_CONTROLLER_ID( 0x045e, 0x0b22 ), k_eControllerType_XBoxOneController, "Xbox One Elite 2 Controller" }, // Microsoft Xbox One Elite Series 2 Controller (BLE)
	{ MAKE_CONTROLLER_ID( 0x0738, 0x4a01 ), k_eControllerType_XBoxOneController, NULL },	// Mad Catz FightStick TE 2
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0139 ), k_eControllerType_XBoxOneController, "PDP Xbox One Afterglow" },	// PDP Afterglow Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x013B ), k_eControllerType_XBoxOneController, "PDP Xbox One Face-Off Controller" },	// PDP Face-Off Gamepad for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x013a ), k_eControllerType_XBoxOneController, NULL },	// PDP Xbox One Controller (unlisted)
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0145 ), k_eControllerType_XBoxOneController, "PDP MK X Fight Pad" },	// PDP MK X Fight Pad for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0146 ), k_eControllerType_XBoxOneController, "PDP Xbox One Rock Candy" },	// PDP Rock Candy Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x015b ), k_eControllerType_XBoxOneController, "PDP Fallout 4 Vault Boy Controller" },	// PDP Fallout 4 Vault Boy Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x015c ), k_eControllerType_XBoxOneController, "PDP Xbox One @Play Controller" },	// PDP @Play Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x015d ), k_eControllerType_XBoxOneController, "PDP Mirror's Edge Controller" },	// PDP Mirror's Edge Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x015f ), k_eControllerType_XBoxOneController, "PDP Metallic Controller" },	// PDP Metallic Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0160 ), k_eControllerType_XBoxOneController, "PDP NFL Face-Off Controller" },	// PDP NFL Official Face-Off Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0161 ), k_eControllerType_XBoxOneController, "PDP Xbox One Camo" },	// PDP Camo Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0162 ), k_eControllerType_XBoxOneController, "PDP Xbox One Controller" },	// PDP Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0163 ), k_eControllerType_XBoxOneController, "PDP Deliverer of Truth" },	// PDP Legendary Collection: Deliverer of Truth
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0164 ), k_eControllerType_XBoxOneController, "PDP Battlefield 1 Controller" },	// PDP Battlefield 1 Official Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0165 ), k_eControllerType_XBoxOneController, "PDP Titanfall 2 Controller" },	// PDP Titanfall 2 Official Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0166 ), k_eControllerType_XBoxOneController, "PDP Mass Effect: Andromeda Controller" },	// PDP Mass Effect: Andromeda Official Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0167 ), k_eControllerType_XBoxOneController, "PDP Halo Wars 2 Face-Off Controller" },	// PDP Halo Wars 2 Official Face-Off Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0205 ), k_eControllerType_XBoxOneController, "PDP Victrix Pro Fight Stick" },	// PDP Victrix Pro Fight Stick
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0206 ), k_eControllerType_XBoxOneController, "PDP Mortal Kombat Controller" },	// PDP Mortal Kombat 25 Anniversary Edition Stick (Xbox One)
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0246 ), k_eControllerType_XBoxOneController, "PDP Xbox One Rock Candy" },	// PDP Rock Candy Wired Controller for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0261 ), k_eControllerType_XBoxOneController, "PDP Xbox One Camo" },	// PDP Camo Wired Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0262 ), k_eControllerType_XBoxOneController, "PDP Xbox One Controller" },	// PDP Wired Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a0 ), k_eControllerType_XBoxOneController, "PDP Xbox One Midnight Blue" },	// PDP Wired Controller for Xbox One - Midnight Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a1 ), k_eControllerType_XBoxOneController, "PDP Xbox One Verdant Green" },	// PDP Wired Controller for Xbox One - Verdant Green
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a2 ), k_eControllerType_XBoxOneController, "PDP Xbox One Crimson Red" },	// PDP Wired Controller for Xbox One - Crimson Red
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a3 ), k_eControllerType_XBoxOneController, "PDP Xbox One Arctic White" },	// PDP Wired Controller for Xbox One - Arctic White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a4 ), k_eControllerType_XBoxOneController, "PDP Xbox One Phantom Black" },	// PDP Wired Controller for Xbox One - Stealth Series | Phantom Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a5 ), k_eControllerType_XBoxOneController, "PDP Xbox One Ghost White" },	// PDP Wired Controller for Xbox One - Stealth Series | Ghost White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a6 ), k_eControllerType_XBoxOneController, "PDP Xbox One Revenant Blue" },	// PDP Wired Controller for Xbox One - Stealth Series | Revenant Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a7 ), k_eControllerType_XBoxOneController, "PDP Xbox One Raven Black" },	// PDP Wired Controller for Xbox One - Raven Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a8 ), k_eControllerType_XBoxOneController, "PDP Xbox One Arctic White" },	// PDP Wired Controller for Xbox One - Arctic White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02a9 ), k_eControllerType_XBoxOneController, "PDP Xbox One Midnight Blue" },	// PDP Wired Controller for Xbox One - Midnight Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02aa ), k_eControllerType_XBoxOneController, "PDP Xbox One Verdant Green" },	// PDP Wired Controller for Xbox One - Verdant Green
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ab ), k_eControllerType_XBoxOneController, "PDP Xbox One Crimson Red" },	// PDP Wired Controller for Xbox One - Crimson Red
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ac ), k_eControllerType_XBoxOneController, "PDP Xbox One Ember Orange" },	// PDP Wired Controller for Xbox One - Ember Orange
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ad ), k_eControllerType_XBoxOneController, "PDP Xbox One Phantom Black" },	// PDP Wired Controller for Xbox One - Stealth Series | Phantom Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ae ), k_eControllerType_XBoxOneController, "PDP Xbox One Ghost White" },	// PDP Wired Controller for Xbox One - Stealth Series | Ghost White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02af ), k_eControllerType_XBoxOneController, "PDP Xbox One Revenant Blue" },	// PDP Wired Controller for Xbox One - Stealth Series | Revenant Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02b0 ), k_eControllerType_XBoxOneController, "PDP Xbox One Raven Black" },	// PDP Wired Controller for Xbox One - Raven Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02b1 ), k_eControllerType_XBoxOneController, "PDP Xbox One Arctic White" },	// PDP Wired Controller for Xbox One - Arctic White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02b3 ), k_eControllerType_XBoxOneController, "PDP Xbox One Afterglow" },	// PDP Afterglow Prismatic Wired Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02b5 ), k_eControllerType_XBoxOneController, "PDP Xbox One GAMEware Controller" },	// PDP GAMEware Wired Controller Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02b6 ), k_eControllerType_XBoxOneController, NULL },	// PDP One-Handed Joystick Adaptive Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02bd ), k_eControllerType_XBoxOneController, "PDP Xbox One Royal Purple" },	// PDP Wired Controller for Xbox One - Royal Purple
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02be ), k_eControllerType_XBoxOneController, "PDP Xbox One Raven Black" },	// PDP Deluxe Wired Controller for Xbox One - Raven Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02bf ), k_eControllerType_XBoxOneController, "PDP Xbox One Midnight Blue" },	// PDP Deluxe Wired Controller for Xbox One - Midnight Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c0 ), k_eControllerType_XBoxOneController, "PDP Xbox One Phantom Black" },	// PDP Deluxe Wired Controller for Xbox One - Stealth Series | Phantom Black
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c1 ), k_eControllerType_XBoxOneController, "PDP Xbox One Ghost White" },	// PDP Deluxe Wired Controller for Xbox One - Stealth Series | Ghost White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c2 ), k_eControllerType_XBoxOneController, "PDP Xbox One Revenant Blue" },	// PDP Deluxe Wired Controller for Xbox One - Stealth Series | Revenant Blue
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c3 ), k_eControllerType_XBoxOneController, "PDP Xbox One Verdant Green" },	// PDP Deluxe Wired Controller for Xbox One - Verdant Green
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c4 ), k_eControllerType_XBoxOneController, "PDP Xbox One Ember Orange" },	// PDP Deluxe Wired Controller for Xbox One - Ember Orange
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c5 ), k_eControllerType_XBoxOneController, "PDP Xbox One Royal Purple" },	// PDP Deluxe Wired Controller for Xbox One - Royal Purple
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c6 ), k_eControllerType_XBoxOneController, "PDP Xbox One Crimson Red" },	// PDP Deluxe Wired Controller for Xbox One - Crimson Red
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c7 ), k_eControllerType_XBoxOneController, "PDP Xbox One Arctic White" },	// PDP Deluxe Wired Controller for Xbox One - Arctic White
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c8 ), k_eControllerType_XBoxOneController, "PDP Kingdom Hearts Controller" },	// PDP Kingdom Hearts Wired Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02c9 ), k_eControllerType_XBoxOneController, "PDP Xbox One Phantasm Red" },	// PDP Deluxe Wired Controller for Xbox One - Stealth Series | Phantasm Red
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ca ), k_eControllerType_XBoxOneController, "PDP Xbox One Specter Violet" },	// PDP Deluxe Wired Controller for Xbox One - Stealth Series | Specter Violet
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02cb ), k_eControllerType_XBoxOneController, "PDP Xbox One Specter Violet" },	// PDP Wired Controller for Xbox One - Stealth Series | Specter Violet
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02cd ), k_eControllerType_XBoxOneController, "PDP Xbox One Blu-merang" },	// PDP Rock Candy Wired Controller for Xbox One - Blu-merang
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02ce ), k_eControllerType_XBoxOneController, "PDP Xbox One Cranblast" },	// PDP Rock Candy Wired Controller for Xbox One - Cranblast
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02cf ), k_eControllerType_XBoxOneController, "PDP Xbox One Aqualime" },	// PDP Rock Candy Wired Controller for Xbox One - Aqualime
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02d5 ), k_eControllerType_XBoxOneController, "PDP Xbox One Red Camo" },	// PDP Wired Controller for Xbox One - Red Camo
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0346 ), k_eControllerType_XBoxOneController, "PDP Xbox One RC Gamepad" },	// PDP RC Gamepad for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0446 ), k_eControllerType_XBoxOneController, "PDP Xbox One RC Gamepad" },	// PDP RC Gamepad for Xbox One
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02da ), k_eControllerType_XBoxOneController, "PDP Xbox Series X Afterglow" },	// PDP Xbox Series X Afterglow
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02d6 ), k_eControllerType_XBoxOneController, "Victrix Gambit Tournament Controller" },	// Victrix Gambit Tournament Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x02d9 ), k_eControllerType_XBoxOneController, "PDP Xbox Series X Midnight Blue" },	// PDP Xbox Series X Midnight Blue
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0063 ), k_eControllerType_XBoxOneController, NULL },	// Hori Real Arcade Pro Hayabusa (USA) Xbox One
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0067 ), k_eControllerType_XBoxOneController, NULL },	// HORIPAD ONE
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0078 ), k_eControllerType_XBoxOneController, NULL },	// Hori Real Arcade Pro V Kai Xbox One
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00c5 ), k_eControllerType_XBoxOneController, NULL },	// HORI Fighting Commander
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0150 ), k_eControllerType_XBoxOneController, NULL },	// HORI Fighting Commander OCTA for Xbox Series X
	{ MAKE_CONTROLLER_ID( 0x10f5, 0x7009 ), k_eControllerType_XBoxOneController, NULL },	// Turtle Beach Recon Controller
	{ MAKE_CONTROLLER_ID( 0x10f5, 0x7013 ), k_eControllerType_XBoxOneController, NULL },	// Turtle Beach REACT-R
	{ MAKE_CONTROLLER_ID( 0x1532, 0x0a00 ), k_eControllerType_XBoxOneController, NULL },	// Razer Atrox Arcade Stick
	{ MAKE_CONTROLLER_ID( 0x1532, 0x0a03 ), k_eControllerType_XBoxOneController, NULL },	// Razer Wildcat
	{ MAKE_CONTROLLER_ID( 0x1532, 0x0a14 ), k_eControllerType_XBoxOneController, NULL },	// Razer Wolverine Ultimate
	{ MAKE_CONTROLLER_ID( 0x1532, 0x0a15 ), k_eControllerType_XBoxOneController, NULL },	// Razer Wolverine Tournament Edition
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2001 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller - Black Inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2002 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Gray/White Inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2003 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Green Inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2004 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Pink inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2005 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X Wired Controller Core - Black
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2006 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X Wired Controller Core - White
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2009 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Red inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200a ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Blue inline
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200b ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Camo Metallic Red
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200c ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Camo Metallic Blue
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200d ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Seafoam Fade
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200e ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Midnight Blue
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x200f ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Soldier Green
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2011 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired - Metallic Ice
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2012 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X Cuphead EnWired Controller - Mugman
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2015 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller - Blue Hint
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2016 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller - Green Hint
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2017 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Cntroller - Arctic Camo
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2018 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Arc Lightning
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x2019 ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Royal Purple
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x201a ), k_eControllerType_XBoxOneController, "PowerA Xbox Series X Controller" },       // PowerA Xbox Series X EnWired Controller Nebula
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x4001 ), k_eControllerType_XBoxOneController, "PowerA Fusion Pro 2 Controller" },	// PowerA Fusion Pro 2 Wired Controller (Xbox Series X style)
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x4002 ), k_eControllerType_XBoxOneController, "PowerA Spectra Infinity Controller" },	// PowerA Spectra Infinity Wired Controller (Xbox Series X style)
	{ MAKE_CONTROLLER_ID( 0x20d6, 0x890b ), k_eControllerType_XBoxOneController, NULL },	// PowerA MOGA XP-Ultra Controller (Xbox Series X style)
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x541a ), k_eControllerType_XBoxOneController, NULL },	// PowerA Xbox One Mini Wired Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x542a ), k_eControllerType_XBoxOneController, NULL },	// Xbox ONE spectra
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x543a ), k_eControllerType_XBoxOneController, "PowerA Xbox One Controller" },	// PowerA Xbox ONE liquid metal controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x551a ), k_eControllerType_XBoxOneController, NULL },	// PowerA FUSION Pro Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x561a ), k_eControllerType_XBoxOneController, NULL },	// PowerA FUSION Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x581a ), k_eControllerType_XBoxOneController, NULL },	// BDA XB1 Classic Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x591a ), k_eControllerType_XBoxOneController, NULL },	// PowerA FUSION Pro Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x592a ), k_eControllerType_XBoxOneController, NULL },	// BDA XB1 Spectra Pro
	{ MAKE_CONTROLLER_ID( 0x24c6, 0x791a ), k_eControllerType_XBoxOneController, NULL },	// PowerA Fusion Fight Pad
	{ MAKE_CONTROLLER_ID( 0x2dc8, 0x2002 ), k_eControllerType_XBoxOneController, NULL },	// 8BitDo Ultimate Wired Controller for Xbox
	{ MAKE_CONTROLLER_ID( 0x2dc8, 0x3106 ), k_eControllerType_XBoxOneController, NULL },	// 8Bitdo Ultimate Wired Controller. Windows, Android, Switch.
	{ MAKE_CONTROLLER_ID( 0x2e24, 0x0652 ), k_eControllerType_XBoxOneController, NULL },	// Hyperkin Duke
	{ MAKE_CONTROLLER_ID( 0x2e24, 0x1618 ), k_eControllerType_XBoxOneController, NULL },	// Hyperkin Duke
	{ MAKE_CONTROLLER_ID( 0x2e24, 0x1688 ), k_eControllerType_XBoxOneController, NULL },	// Hyperkin X91
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0611 ), k_eControllerType_XBoxOneController, NULL },	// Xbox Controller Mode for NACON Revolution 3

	// These have been added via Minidump for unrecognized Xinput controller assert
	{ MAKE_CONTROLLER_ID( 0x0000, 0x0000 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x045e, 0x02a2 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller - Microsoft VID
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x1414 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0159 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x24c6, 0xfaff ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x006d ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00a4 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x1832 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x187f ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x1883 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x03eb, 0xff01 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0c12, 0x0ef8 ), k_eControllerType_XBox360Controller, NULL },	// Homemade fightstick based on brook pcb (with XInput driver??)
	{ MAKE_CONTROLLER_ID( 0x046d, 0x1000 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x11ff, 0x0511 ), k_eControllerType_XBox360Controller, NULL },	// PXN V900
	{ MAKE_CONTROLLER_ID( 0x1345, 0x6006 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller

	{ MAKE_CONTROLLER_ID( 0x056e, 0x2012 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x146b, 0x0602 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00ae ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x046d, 0x0401 ), k_eControllerType_XBox360Controller, NULL },	// logitech xinput
	{ MAKE_CONTROLLER_ID( 0x046d, 0x0301 ), k_eControllerType_XBox360Controller, NULL },	// logitech xinput
	{ MAKE_CONTROLLER_ID( 0x046d, 0xcaa3 ), k_eControllerType_XBox360Controller, NULL },	// logitech xinput
	{ MAKE_CONTROLLER_ID( 0x046d, 0xc261 ), k_eControllerType_XBox360Controller, NULL },	// logitech xinput
	{ MAKE_CONTROLLER_ID( 0x046d, 0x0291 ), k_eControllerType_XBox360Controller, NULL },	// logitech xinput
	{ MAKE_CONTROLLER_ID( 0x0079, 0x18d3 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00b1 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0001, 0x0001 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x188e ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x187c ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x189c ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x0079, 0x1874 ), k_eControllerType_XBox360Controller, NULL },	// Unknown Controller

	{ MAKE_CONTROLLER_ID( 0x2f24, 0x0050 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24, 0x2e ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24, 0x91 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1430, 0x719 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xf0d, 0xed ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xf0d, 0xc0 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f, 0x152 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f, 0x2a7 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x46d, 0x1007 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f, 0x2b8 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f, 0x2a8 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x79, 0x18a1 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller

	// Added from Minidumps 10-9-19
	{ MAKE_CONTROLLER_ID( 0x0,		0x6686 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x12ab,	0x304 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1430,	0x291 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1430,	0x2a9 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1430,	0x70b ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1bad,	0x28e ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1bad,	0x2a0 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x1bad,	0x5500 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x20ab,	0x55ef ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x24c6,	0x5509 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2516,	0x69 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x25b1,	0x360 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2c22,	0x2203 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24,	0x11 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24,	0x53 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24,	0xb7 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x46d,	0x0 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x46d,	0x1004 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x46d,	0x1008 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x46d,	0xf301 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x738,	0x2a0 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x738,	0x7263 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x738,	0xb738 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x738,	0xcb29 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x738,	0xf401 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x79,		0x18c2 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x79,		0x18c8 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x79,		0x18cf ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xc12,	0xe17 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xc12,	0xe1c ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xc12,	0xe22 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xc12,	0xe30 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xd2d2,	0xd2d2 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xd62,	0x9a1a ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xd62,	0x9a1b ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe00,	0xe00 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x12a ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2a1 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2a2 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2a5 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2b2 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2bd ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2bf ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2c0 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0x2c6 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xf0d,	0x97 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xf0d,	0xba ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xf0d,	0xd8 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xfff,	0x2a1 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x45e,	0x867 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	// Added 12-17-2020
	{ MAKE_CONTROLLER_ID( 0x16d0,	0xf3f ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0x2f24,	0x8f ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller
	{ MAKE_CONTROLLER_ID( 0xe6f,	0xf501 ), k_eControllerType_XBoxOneController, NULL },	// Unknown Controller

	//{ MAKE_CONTROLLER_ID( 0x1949, 0x0402 ), /*android*/, NULL },	// Unknown Controller

	{ MAKE_CONTROLLER_ID( 0x05ac, 0x0001 ), k_eControllerType_AppleController, NULL },	// MFI Extended Gamepad (generic entry for iOS/tvOS)
	{ MAKE_CONTROLLER_ID( 0x05ac, 0x0002 ), k_eControllerType_AppleController, NULL },	// MFI Standard Gamepad (generic entry for iOS/tvOS)

    { MAKE_CONTROLLER_ID( 0x057e, 0x2006 ), k_eControllerType_SwitchJoyConLeft, NULL },    // Nintendo Switch Joy-Con (Left)
    { MAKE_CONTROLLER_ID( 0x057e, 0x2007 ), k_eControllerType_SwitchJoyConRight, NULL },   // Nintendo Switch Joy-Con (Right)
    { MAKE_CONTROLLER_ID( 0x057e, 0x2008 ), k_eControllerType_SwitchJoyConPair, NULL },    // Nintendo Switch Joy-Con (Left+Right Combined)

    // This same controller ID is spoofed by many 3rd-party Switch controllers.
    // The ones we currently know of are:
    // * Any 8bitdo controller with Switch support
    // * ORTZ Gaming Wireless Pro Controller
    // * ZhiXu Gamepad Wireless
    // * Sunwaytek Wireless Motion Controller for Nintendo Switch
	{ MAKE_CONTROLLER_ID( 0x057e, 0x2009 ), k_eControllerType_SwitchProController, NULL },        // Nintendo Switch Pro Controller
    //{ MAKE_CONTROLLER_ID( 0x057e, 0x2017 ), k_eControllerType_SwitchProController, NULL },        // Nintendo Online SNES Controller
    //{ MAKE_CONTROLLER_ID( 0x057e, 0x2019 ), k_eControllerType_SwitchProController, NULL },        // Nintendo Online N64 Controller
    //{ MAKE_CONTROLLER_ID( 0x057e, 0x201e ), k_eControllerType_SwitchProController, NULL },        // Nintendo Online SEGA Genesis Controller

	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00c1 ), k_eControllerType_SwitchInputOnlyController, NULL },  // HORIPAD for Nintendo Switch
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x0092 ), k_eControllerType_SwitchInputOnlyController, NULL },  // HORI Pokken Tournament DX Pro Pad
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00f6 ), k_eControllerType_SwitchProController, NULL },		// HORI Wireless Switch Pad
	// The HORIPAD S, which comes in multiple styles:
	// - NSW-108, classic GameCube controller
	// - NSW-244, Fighting Commander arcade pad
	// - NSW-278, Hori Pad Mini gamepad
	// - NSW-326, HORIPAD FPS for Nintendo Switch
	//
	// The first two, at least, shouldn't have their buttons remapped, and since we
	// can't tell which model we're actually using, we won't do any button remapping
	// for any of them.
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00dc ), k_eControllerType_XInputSwitchController, NULL },	 // HORIPAD S - Looks like a Switch controller but uses the Xbox 360 controller protocol, there is also a version of this that looks like a GameCube controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0180 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Faceoff Wired Pro Controller for Nintendo Switch
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0181 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Faceoff Deluxe Wired Pro Controller for Nintendo Switch
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0184 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Faceoff Wired Deluxe+ Audio Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0185 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Wired Fight Pad Pro for Nintendo Switch
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0186 ), k_eControllerType_SwitchProController, NULL },        // PDP Afterglow Wireless Switch Controller - working gyro. USB is for charging only. Many later "Wireless" line devices w/ gyro also use this vid/pid
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0187 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Rockcandy Wired Controller
	{ MAKE_CONTROLLER_ID( 0x0e6f, 0x0188 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PDP Afterglow Wired Deluxe+ Audio Controller
	{ MAKE_CONTROLLER_ID( 0x0f0d, 0x00aa ), k_eControllerType_SwitchInputOnlyController, NULL },  // HORI Real Arcade Pro V Hayabusa in Switch Mode
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa711 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Wired Controller Plus/PowerA Wired Controller Nintendo GameCube Style
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa712 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Nintendo Switch Fusion Fight Pad
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa713 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Super Mario Controller
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa714 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Nintendo Switch Spectra Controller
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa715 ), k_eControllerType_SwitchInputOnlyController, NULL },  // Power A Fusion Wireless Arcade Stick (USB Mode) Over BT is shows up as 057e 2009
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa716 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Nintendo Switch Fusion Pro Controller - USB requires toggling switch on back of device
	{ MAKE_CONTROLLER_ID( 0x20d6, 0xa718 ), k_eControllerType_SwitchInputOnlyController, NULL },  // PowerA Nintendo Switch Nano Wired Controller
    { MAKE_CONTROLLER_ID( 0x33dd, 0x0001 ), k_eControllerType_SwitchInputOnlyController, NULL },  // ZUIKI MasCon for Nintendo Switch Black
    { MAKE_CONTROLLER_ID( 0x33dd, 0x0002 ), k_eControllerType_SwitchInputOnlyController, NULL },  // ZUIKI MasCon for Nintendo Switch ??
    { MAKE_CONTROLLER_ID( 0x33dd, 0x0003 ), k_eControllerType_SwitchInputOnlyController, NULL },  // ZUIKI MasCon for Nintendo Switch Red

	// Valve products
	{ MAKE_CONTROLLER_ID( 0x0000, 0x11fb ), k_eControllerType_MobileTouch, NULL },	// Streaming mobile touch virtual controls
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1101 ), k_eControllerType_SteamController, NULL },	// Valve Legacy Steam Controller (CHELL)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1102 ), k_eControllerType_SteamController, NULL },	// Valve wired Steam Controller (D0G)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1105 ), k_eControllerType_SteamController, NULL },	// Valve Bluetooth Steam Controller (D0G)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1106 ), k_eControllerType_SteamController, NULL },	// Valve Bluetooth Steam Controller (D0G)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x11ff ), k_eControllerType_UnknownNonSteamController, NULL },	// Steam Virtual Gamepad
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1142 ), k_eControllerType_SteamController, NULL },	// Valve wireless Steam Controller
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1201 ), k_eControllerType_SteamControllerV2, NULL },	// Valve wired Steam Controller (HEADCRAB)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1202 ), k_eControllerType_SteamControllerV2, NULL },	// Valve Bluetooth Steam Controller (HEADCRAB)
	{ MAKE_CONTROLLER_ID( 0x28de, 0x1205 ), k_eControllerType_SteamControllerNeptune, NULL },	// Valve Steam Deck Builtin Controller
};
