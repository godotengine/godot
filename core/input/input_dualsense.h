/**************************************************************************/
/*  input_dualsense.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <stdint.h>

// https://controllers.fandom.com/wiki/Sony_DualSense#Output_Reports
struct DS5SetStateData { // 47
	/*    */ // Report Set Flags
	/*    */ // These flags are used to indicate what contents from this report should be processed
	/* 0.0*/ uint8_t EnableRumbleEmulation : 1; // Suggest halving rumble strength
	/* 0.1*/ uint8_t UseRumbleNotHaptics : 1; //
	/*    */
	/* 0.2*/ uint8_t AllowRightTriggerFFB : 1; // Enable setting RightTriggerFFB
	/* 0.3*/ uint8_t AllowLeftTriggerFFB : 1; // Enable setting LeftTriggerFFB
	/*    */
	/* 0.4*/ uint8_t AllowHeadphoneVolume : 1; // Enable setting VolumeHeadphones
	/* 0.5*/ uint8_t AllowSpeakerVolume : 1; // Enable setting VolumeSpeaker
	/* 0.6*/ uint8_t AllowMicVolume : 1; // Enable setting VolumeMic
	/*    */
	/* 0.7*/ uint8_t AllowAudioControl : 1; // Enable setting AudioControl section
	/* 1.0*/ uint8_t AllowMuteLight : 1; // Enable setting MuteLightMode
	/* 1.1*/ uint8_t AllowAudioMute : 1; // Enable setting MuteControl section
	/*    */
	/* 1.2*/ uint8_t AllowLedColor : 1; // Enable RGB LED section
	/*    */
	/* 1.3*/ uint8_t ResetLights : 1; // Release the LEDs from Wireless firmware control
	/*    */ // When in wireless mode this must be signaled to control LEDs
	/*    */ // This cannot be applied during the BT pair animation.
	/*    */ // SDL2 waits until the SensorTimestamp value is >= 10200000
	/*    */ // before pulsing this bit once.
	/*    */
	/* 1.4*/ uint8_t AllowPlayerIndicators : 1; // Enable setting PlayerIndicators section
	/* 1.5*/ uint8_t AllowHapticLowPassFilter : 1; // Enable HapticLowPassFilter
	/* 1.6*/ uint8_t AllowMotorPowerLevel : 1; // MotorPowerLevel reductions for trigger/haptic
	/* 1.7*/ uint8_t AllowAudioControl2 : 1; // Enable setting AudioControl2 section
	/*    */
	/* 2  */ uint8_t RumbleEmulationRight; // emulates the light weight
	/* 3  */ uint8_t RumbleEmulationLeft; // emulated the heavy weight
	/*    */
	/* 4  */ uint8_t VolumeHeadphones; // max 0x7f
	/* 5  */ uint8_t VolumeSpeaker; // PS5 appears to only use the range 0x3d-0x64
	/* 6  */ uint8_t VolumeMic; // not linear, seems to max at 64, 0 is fully muted only in chat mode
	/*    */
	/*    */ // AudioControl
	/* 7.0*/ uint8_t MicSelect : 2; // 0 Auto
	/*    */ // 1 Internal Only
	/*    */ // 2 External Only
	/*    */ // 3 Unclear, sets external mic flag but might use internal mic, do test
	/* 7.2*/ uint8_t EchoCancelEnable : 1;
	/* 7.3*/ uint8_t NoiseCancelEnable : 1;
	/* 7.4*/ uint8_t OutputPathSelect : 2; // 0 L_R_X
	/*    */ // 1 L_L_X
	/*    */ // 2 L_L_R
	/*    */ // 3 X_X_R
	/* 7.6*/ uint8_t InputPathSelect : 2; // 0 CHAT_ASR
	/*    */ // 1 CHAT_CHAT
	/*    */ // 2 ASR_ASR
	/*    */ // 3 Does Nothing, invalid
	/*    */
	/* 8  */ uint8_t MuteLightMode;
	/*    */
	/*    */ // MuteControl
	/* 9.0*/ uint8_t TouchPowerSave : 1;
	/* 9.1*/ uint8_t MotionPowerSave : 1;
	/* 9.2*/ uint8_t HapticPowerSave : 1; // AKA BulletPowerSave
	/* 9.3*/ uint8_t AudioPowerSave : 1;
	/* 9.4*/ uint8_t MicMute : 1;
	/* 9.5*/ uint8_t SpeakerMute : 1;
	/* 9.6*/ uint8_t HeadphoneMute : 1;
	/* 9.7*/ uint8_t HapticMute : 1; // AKA BulletMute
	/*    */
	/*10  */ uint8_t RightTriggerFFB[11];
	/*21  */ uint8_t LeftTriggerFFB[11];
	/*32  */ uint32_t HostTimestamp; // mirrored into report read
	/*    */
	/*    */ // MotorPowerLevel
	/*36.0*/ uint8_t TriggerMotorPowerReduction : 4; // 0x0-0x7 (no 0x8?) Applied in 12.5% reductions
	/*36.4*/ uint8_t RumbleMotorPowerReduction : 4; // 0x0-0x7 (no 0x8?) Applied in 12.5% reductions
	/*    */
	/*    */ // AudioControl2
	/*37.0*/ uint8_t SpeakerCompPreGain : 3; // additional speaker volume boost
	/*37.3*/ uint8_t BeamformingEnable : 1; // Probably for MIC given there's 2, might be more bits, can't find what it does
	/*37.4*/ uint8_t UnkAudioControl2 : 4; // some of these bits might apply to the above
	/*    */
	/*38.0*/ uint8_t AllowLightBrightnessChange : 1; // LED_BRIHTNESS_CONTROL
	/*38.1*/ uint8_t AllowColorLightFadeAnimation : 1; // LIGHTBAR_SETUP_CONTROL
	/*38.2*/ uint8_t EnableImprovedRumbleEmulation : 1; // Use instead of EnableRumbleEmulation
														// requires FW >= 0x0224
														// No need to halve rumble strength
	/*38.3*/ uint8_t UNKBITC : 5; // unused
	/*    */
	/*39.0*/ uint8_t HapticLowPassFilter : 1;
	/*39.1*/ uint8_t UNKBIT : 7;
	/*    */
	/*40  */ uint8_t UNKBYTE; // previous notes suggested this was HLPF, was probably off by 1
	/*    */
	/*41  */ uint8_t LightFadeAnimation;
	/*42  */ uint8_t LightBrightness;
	/*    */
	/*    */ // PlayerIndicators
	/*    */ // These bits control the white LEDs under the touch pad.
	/*    */ // Note the reduction in functionality for later revisions.
	/*    */ // Generation 0x03 - Full Functionality
	/*    */ // Generation 0x04 - Mirrored Only
	/*    */ // Suggested detection: (HardwareInfo & 0x00FFFF00) == 0X00000400
	/*    */ //
	/*    */ // Layout used by PS5:
	/*    */ // 0x04 - -x- -  Player 1
	/*    */ // 0x06 - x-x -  Player 2
	/*    */ // 0x15 x -x- x  Player 3
	/*    */ // 0x1B x x-x x  Player 4
	/*    */ // 0x1F x xxx x  Player 5* (Unconfirmed)
	/*    */ //
	/*    */ //                        // HW 0x03 // HW 0x04
	/*43.0*/ uint8_t PlayerLight1 : 1; // x --- - // x --- x
	/*43.1*/ uint8_t PlayerLight2 : 1; // - x-- - // - x-x -
	/*43.2*/ uint8_t PlayerLight3 : 1; // - -x- - // - -x- -
	/*43.3*/ uint8_t PlayerLight4 : 1; // - --x - // - x-x -
	/*43.4*/ uint8_t PlayerLight5 : 1; // - --- x // x --- x
	/*43.5*/ uint8_t PlayerLightFade : 1; // if low player lights fade in, if high player lights instantly change
	/*43.6*/ uint8_t PlayerLightUNK : 2;
	/*    */
	/*    */ // RGB LED
	/*44  */ uint8_t LedRed;
	/*45  */ uint8_t LedGreen;
	/*46  */ uint8_t LedBlue;
	// Structure ends here though on BT there is padding and a CRC, see ReportOut31
};
