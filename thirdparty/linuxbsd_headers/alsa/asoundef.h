/**
 * \file include/asoundef.h
 * \brief Application interface library for the ALSA driver
 * \author Jaroslav Kysela <perex@perex.cz>
 * \author Abramo Bagnara <abramo@alsa-project.org>
 * \author Takashi Iwai <tiwai@suse.de>
 * \date 1998-2001
 *
 * Definitions of constants for the ALSA driver
 */
/*
 *   This library is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as
 *   published by the Free Software Foundation; either version 2.1 of
 *   the License, or (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public
 *   License along with this library; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 */

#ifndef __ALSA_ASOUNDEF_H
#define __ALSA_ASOUNDEF_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup Digital_Audio_Interface Constants for Digital Audio Interfaces
 * AES/IEC958 channel status bits.
 * \{
 */

#define IEC958_AES0_PROFESSIONAL	(1<<0)	/**< 0 = consumer, 1 = professional */
#define IEC958_AES0_NONAUDIO		(1<<1)	/**< 0 = audio, 1 = non-audio */
#define IEC958_AES0_PRO_EMPHASIS	(7<<2)	/**< mask - emphasis */
#define IEC958_AES0_PRO_EMPHASIS_NOTID	(0<<2)	/**< emphasis not indicated */
#define IEC958_AES0_PRO_EMPHASIS_NONE	(1<<2)	/**< no emphasis */
#define IEC958_AES0_PRO_EMPHASIS_5015	(3<<2)	/**< 50/15us emphasis */
#define IEC958_AES0_PRO_EMPHASIS_CCITT	(7<<2)	/**< CCITT J.17 emphasis */
#define IEC958_AES0_PRO_FREQ_UNLOCKED	(1<<5)	/**< source sample frequency: 0 = locked, 1 = unlocked */
#define IEC958_AES0_PRO_FS		(3<<6)	/**< mask - sample frequency */
#define IEC958_AES0_PRO_FS_NOTID	(0<<6)	/**< fs not indicated */
#define IEC958_AES0_PRO_FS_44100	(1<<6)	/**< 44.1kHz */
#define IEC958_AES0_PRO_FS_48000	(2<<6)	/**< 48kHz */
#define IEC958_AES0_PRO_FS_32000	(3<<6)	/**< 32kHz */
#define IEC958_AES0_CON_NOT_COPYRIGHT	(1<<2)	/**< 0 = copyright, 1 = not copyright */
#define IEC958_AES0_CON_EMPHASIS	(7<<3)	/**< mask - emphasis */
#define IEC958_AES0_CON_EMPHASIS_NONE	(0<<3)	/**< no emphasis */
#define IEC958_AES0_CON_EMPHASIS_5015	(1<<3)	/**< 50/15us emphasis */
#define IEC958_AES0_CON_MODE		(3<<6)	/**< mask - mode */
#define IEC958_AES1_PRO_MODE		(15<<0)	/**< mask - channel mode */
#define IEC958_AES1_PRO_MODE_NOTID	(0<<0)	/**< mode not indicated */
#define IEC958_AES1_PRO_MODE_STEREOPHONIC (2<<0) /**< stereophonic - ch A is left */
#define IEC958_AES1_PRO_MODE_SINGLE	(4<<0)	/**< single channel */
#define IEC958_AES1_PRO_MODE_TWO	(8<<0)	/**< two channels */
#define IEC958_AES1_PRO_MODE_PRIMARY	(12<<0)	/**< primary/secondary */
#define IEC958_AES1_PRO_MODE_BYTE3	(15<<0)	/**< vector to byte 3 */
#define IEC958_AES1_PRO_USERBITS	(15<<4)	/**< mask - user bits */
#define IEC958_AES1_PRO_USERBITS_NOTID	(0<<4)	/**< user bits not indicated */
#define IEC958_AES1_PRO_USERBITS_192	(8<<4)	/**< 192-bit structure */
#define IEC958_AES1_PRO_USERBITS_UDEF	(12<<4)	/**< user defined application */
#define IEC958_AES1_CON_CATEGORY	0x7f	/**< consumer category */
#define IEC958_AES1_CON_GENERAL		0x00	/**< general category */
#define IEC958_AES1_CON_LASEROPT_MASK	0x07	/**< Laser-optical mask */
#define IEC958_AES1_CON_LASEROPT_ID	0x01	/**< Laser-optical ID */
#define IEC958_AES1_CON_IEC908_CD	(IEC958_AES1_CON_LASEROPT_ID|0x00)	/**< IEC958 CD compatible device */
#define IEC958_AES1_CON_NON_IEC908_CD	(IEC958_AES1_CON_LASEROPT_ID|0x08)	/**< non-IEC958 CD compatible device */
#define IEC958_AES1_CON_MINI_DISC	(IEC958_AES1_CON_LASEROPT_ID|0x48)	/**< Mini-Disc device */
#define IEC958_AES1_CON_DVD		(IEC958_AES1_CON_LASEROPT_ID|0x18)	/**< DVD device */
#define IEC958_AES1_CON_LASTEROPT_OTHER	(IEC958_AES1_CON_LASEROPT_ID|0x78)	/**< Other laser-optical product */
#define IEC958_AES1_CON_DIGDIGCONV_MASK 0x07	/**< digital<->digital converter mask */
#define IEC958_AES1_CON_DIGDIGCONV_ID	0x02	/**< digital<->digital converter id */
#define IEC958_AES1_CON_PCM_CODER	(IEC958_AES1_CON_DIGDIGCONV_ID|0x00)	/**< PCM coder */
#define IEC958_AES1_CON_MIXER		(IEC958_AES1_CON_DIGDIGCONV_ID|0x10)	/**< Digital signal mixer */
#define IEC958_AES1_CON_RATE_CONVERTER	(IEC958_AES1_CON_DIGDIGCONV_ID|0x18)	/**< Rate converter */
#define IEC958_AES1_CON_SAMPLER		(IEC958_AES1_CON_DIGDIGCONV_ID|0x20)	/**< PCM sampler */
#define IEC958_AES1_CON_DSP		(IEC958_AES1_CON_DIGDIGCONV_ID|0x28)	/**< Digital sound processor */
#define IEC958_AES1_CON_DIGDIGCONV_OTHER (IEC958_AES1_CON_DIGDIGCONV_ID|0x78)	/**< Other digital<->digital product */
#define IEC958_AES1_CON_MAGNETIC_MASK	0x07	/**< Magnetic device mask */
#define IEC958_AES1_CON_MAGNETIC_ID	0x03	/**< Magnetic device ID */
#define IEC958_AES1_CON_DAT		(IEC958_AES1_CON_MAGNETIC_ID|0x00)	/**< Digital Audio Tape */
#define IEC958_AES1_CON_VCR		(IEC958_AES1_CON_MAGNETIC_ID|0x08)	/**< Video recorder */
#define IEC958_AES1_CON_DCC		(IEC958_AES1_CON_MAGNETIC_ID|0x40)	/**< Digital compact cassette */
#define IEC958_AES1_CON_MAGNETIC_DISC	(IEC958_AES1_CON_MAGNETIC_ID|0x18)	/**< Magnetic disc digital audio device */
#define IEC958_AES1_CON_MAGNETIC_OTHER	(IEC958_AES1_CON_MAGNETIC_ID|0x78)	/**< Other magnetic device */
#define IEC958_AES1_CON_BROADCAST1_MASK 0x07	/**< Broadcast mask */
#define IEC958_AES1_CON_BROADCAST1_ID	0x04	/**< Broadcast ID */
#define IEC958_AES1_CON_DAB_JAPAN	(IEC958_AES1_CON_BROADCAST1_ID|0x00)	/**< Digital audio broadcast (Japan) */
#define IEC958_AES1_CON_DAB_EUROPE	(IEC958_AES1_CON_BROADCAST1_ID|0x08)	/**< Digital audio broadcast (Europe) */
#define IEC958_AES1_CON_DAB_USA		(IEC958_AES1_CON_BROADCAST1_ID|0x60)	/**< Digital audio broadcast (USA) */
#define IEC958_AES1_CON_SOFTWARE	(IEC958_AES1_CON_BROADCAST1_ID|0x40)	/**< Electronic software delivery */
#define IEC958_AES1_CON_IEC62105	(IEC958_AES1_CON_BROADCAST1_ID|0x20)	/**< Used by another standard (IEC 62105) */
#define IEC958_AES1_CON_BROADCAST1_OTHER (IEC958_AES1_CON_BROADCAST1_ID|0x78)	/**< Other broadcast product */
#define IEC958_AES1_CON_BROADCAST2_MASK 0x0f	/**< Broadcast alternative mask */
#define IEC958_AES1_CON_BROADCAST2_ID	0x0e	/**< Broadcast alternative ID */
#define IEC958_AES1_CON_MUSICAL_MASK	0x07	/**< Musical device mask */
#define IEC958_AES1_CON_MUSICAL_ID	0x05	/**< Musical device ID */
#define IEC958_AES1_CON_SYNTHESIZER	(IEC958_AES1_CON_MUSICAL_ID|0x00)	/**< Synthesizer */
#define IEC958_AES1_CON_MICROPHONE	(IEC958_AES1_CON_MUSICAL_ID|0x08)	/**< Microphone */
#define IEC958_AES1_CON_MUSICAL_OTHER	(IEC958_AES1_CON_MUSICAL_ID|0x78)	/**< Other musical device */
#define IEC958_AES1_CON_ADC_MASK	0x1f	/**< ADC Mask */
#define IEC958_AES1_CON_ADC_ID		0x06	/**< ADC ID */
#define IEC958_AES1_CON_ADC		(IEC958_AES1_CON_ADC_ID|0x00)	/**< ADC without copyright information */
#define IEC958_AES1_CON_ADC_OTHER	(IEC958_AES1_CON_ADC_ID|0x60)	/**< Other ADC product (with no copyright information) */
#define IEC958_AES1_CON_ADC_COPYRIGHT_MASK 0x1f	/**< ADC Copyright mask */
#define IEC958_AES1_CON_ADC_COPYRIGHT_ID 0x16	/**< ADC Copyright ID */
#define IEC958_AES1_CON_ADC_COPYRIGHT	(IEC958_AES1_CON_ADC_COPYRIGHT_ID|0x00)	/**< ADC with copyright information */
#define IEC958_AES1_CON_ADC_COPYRIGHT_OTHER (IEC958_AES1_CON_ADC_COPYRIGHT_ID|0x60)	/**< Other ADC with copyright information product */
#define IEC958_AES1_CON_SOLIDMEM_MASK	0x0f	/**< Solid memory based products mask */
#define IEC958_AES1_CON_SOLIDMEM_ID	0x08	/**< Solid memory based products ID */
#define IEC958_AES1_CON_SOLIDMEM_DIGITAL_RECORDER_PLAYER (IEC958_AES1_CON_SOLIDMEM_ID|0x00)	/**< Digital audio recorder and player using solid state memory */
#define IEC958_AES1_CON_SOLIDMEM_OTHER	(IEC958_AES1_CON_SOLIDMEM_ID|0x70)	/**< Other solid state memory based product */
#define IEC958_AES1_CON_EXPERIMENTAL	0x40	/**< experimental category */
#define IEC958_AES1_CON_ORIGINAL	(1<<7)	/**< this bits depends on the category code */
#define IEC958_AES2_PRO_SBITS		(7<<0)	/**< mask - sample bits */
#define IEC958_AES2_PRO_SBITS_20	(2<<0)	/**< 20-bit - coordination */
#define IEC958_AES2_PRO_SBITS_24	(4<<0)	/**< 24-bit - main audio */
#define IEC958_AES2_PRO_SBITS_UDEF	(6<<0)	/**< user defined application */
#define IEC958_AES2_PRO_WORDLEN		(7<<3)	/**< mask - source word length */
#define IEC958_AES2_PRO_WORDLEN_NOTID	(0<<3)	/**< source word length not indicated */
#define IEC958_AES2_PRO_WORDLEN_22_18	(2<<3)	/**< 22-bit or 18-bit */
#define IEC958_AES2_PRO_WORDLEN_23_19	(4<<3)	/**< 23-bit or 19-bit */
#define IEC958_AES2_PRO_WORDLEN_24_20	(5<<3)	/**< 24-bit or 20-bit */
#define IEC958_AES2_PRO_WORDLEN_20_16	(6<<3)	/**< 20-bit or 16-bit */
#define IEC958_AES2_CON_SOURCE		(15<<0)	/**< mask - source number */
#define IEC958_AES2_CON_SOURCE_UNSPEC	(0<<0)	/**< source number unspecified */
#define IEC958_AES2_CON_CHANNEL		(15<<4)	/**< mask - channel number */
#define IEC958_AES2_CON_CHANNEL_UNSPEC	(0<<4)	/**< channel number unspecified */
#define IEC958_AES3_CON_FS		(15<<0)	/**< mask - sample frequency */
#define IEC958_AES3_CON_FS_44100	(0<<0)	/**< 44.1kHz */
#define IEC958_AES3_CON_FS_NOTID	(1<<0)	/**< sample frequency non indicated */
#define IEC958_AES3_CON_FS_48000	(2<<0)	/**< 48kHz */
#define IEC958_AES3_CON_FS_32000	(3<<0)	/**< 32kHz */
#define IEC958_AES3_CON_FS_22050	(4<<0)	/**< 22.05kHz */
#define IEC958_AES3_CON_FS_24000	(6<<0)	/**< 24kHz */
#define IEC958_AES3_CON_FS_88200	(8<<0)	/**< 88.2kHz */
#define IEC958_AES3_CON_FS_768000	(9<<0)	/**< 768kHz */
#define IEC958_AES3_CON_FS_96000	(10<<0)	/**< 96kHz */
#define IEC958_AES3_CON_FS_176400	(12<<0)	/**< 176.4kHz */
#define IEC958_AES3_CON_FS_192000	(14<<0)	/**< 192kHz */
#define IEC958_AES3_CON_CLOCK		(3<<4)	/**< mask - clock accuracy */
#define IEC958_AES3_CON_CLOCK_1000PPM	(0<<4)	/**< 1000 ppm */
#define IEC958_AES3_CON_CLOCK_50PPM	(1<<4)	/**< 50 ppm */
#define IEC958_AES3_CON_CLOCK_VARIABLE	(2<<4)	/**< variable pitch */
#define IEC958_AES4_CON_MAX_WORDLEN_24	(1<<0)	/**< 0 = 20-bit, 1 = 24-bit */
#define IEC958_AES4_CON_WORDLEN		(7<<1)	/**< mask - sample word length */
#define IEC958_AES4_CON_WORDLEN_NOTID	(0<<1)	/**< not indicated */
#define IEC958_AES4_CON_WORDLEN_20_16	(1<<1)	/**< 20-bit or 16-bit */
#define IEC958_AES4_CON_WORDLEN_22_18	(2<<1)	/**< 22-bit or 18-bit */
#define IEC958_AES4_CON_WORDLEN_23_19	(4<<1)	/**< 23-bit or 19-bit */
#define IEC958_AES4_CON_WORDLEN_24_20	(5<<1)	/**< 24-bit or 20-bit */
#define IEC958_AES4_CON_WORDLEN_21_17	(6<<1)	/**< 21-bit or 17-bit */
#define IEC958_AES4_CON_ORIGFS		(15<<4)	/**< mask - original sample frequency */
#define IEC958_AES4_CON_ORIGFS_NOTID	(0<<4)	/**< original sample frequency not indicated */
#define IEC958_AES4_CON_ORIGFS_192000	(1<<4)	/**< 192kHz */
#define IEC958_AES4_CON_ORIGFS_12000	(2<<4)	/**< 12kHz */
#define IEC958_AES4_CON_ORIGFS_176400	(3<<4)	/**< 176.4kHz */
#define IEC958_AES4_CON_ORIGFS_96000	(5<<4)	/**< 96kHz */
#define IEC958_AES4_CON_ORIGFS_8000	(6<<4)	/**< 8kHz */
#define IEC958_AES4_CON_ORIGFS_88200	(7<<4)	/**< 88.2kHz */
#define IEC958_AES4_CON_ORIGFS_16000	(8<<4)	/**< 16kHz */
#define IEC958_AES4_CON_ORIGFS_24000	(9<<4)	/**< 24kHz */
#define IEC958_AES4_CON_ORIGFS_11025	(10<<4)	/**< 11.025kHz */
#define IEC958_AES4_CON_ORIGFS_22050	(11<<4)	/**< 22.05kHz */
#define IEC958_AES4_CON_ORIGFS_32000	(12<<4)	/**< 32kHz */
#define IEC958_AES4_CON_ORIGFS_48000	(13<<4)	/**< 48kHz */
#define IEC958_AES4_CON_ORIGFS_44100	(15<<4)	/**< 44.1kHz */
#define IEC958_AES5_CON_CGMSA		(3<<0)	/**< mask - CGMS-A */
#define IEC958_AES5_CON_CGMSA_COPYFREELY (0<<0)	/**< copying is permitted without restriction */
#define IEC958_AES5_CON_CGMSA_COPYONCE	(1<<0)	/**< one generation of copies may be made */
#define IEC958_AES5_CON_CGMSA_COPYNOMORE (2<<0)	/**< condition not be used */
#define IEC958_AES5_CON_CGMSA_COPYNEVER	(3<<0)	/**< no copying is permitted */

/** \} */

/**
 * \defgroup MIDI_Interface Constants for MIDI v1.0
 * Constants for MIDI v1.0.
 * \{
 */

#define MIDI_CHANNELS			16	/**< Number of channels per port/cable. */
#define MIDI_GM_DRUM_CHANNEL		(10-1)	/**< Channel number for GM drums. */

/**
 * \defgroup MIDI_Commands MIDI Commands
 * MIDI command codes.
 * \{
 */

#define MIDI_CMD_NOTE_OFF		0x80	/**< note off */
#define MIDI_CMD_NOTE_ON		0x90	/**< note on */
#define MIDI_CMD_NOTE_PRESSURE		0xa0	/**< key pressure */
#define MIDI_CMD_CONTROL		0xb0	/**< control change */
#define MIDI_CMD_PGM_CHANGE		0xc0	/**< program change */
#define MIDI_CMD_CHANNEL_PRESSURE	0xd0	/**< channel pressure */
#define MIDI_CMD_BENDER			0xe0	/**< pitch bender */

#define MIDI_CMD_COMMON_SYSEX		0xf0	/**< sysex (system exclusive) begin */
#define MIDI_CMD_COMMON_MTC_QUARTER	0xf1	/**< MTC quarter frame */
#define MIDI_CMD_COMMON_SONG_POS	0xf2	/**< song position */
#define MIDI_CMD_COMMON_SONG_SELECT	0xf3	/**< song select */
#define MIDI_CMD_COMMON_TUNE_REQUEST	0xf6	/**< tune request */
#define MIDI_CMD_COMMON_SYSEX_END	0xf7	/**< end of sysex */
#define MIDI_CMD_COMMON_CLOCK		0xf8	/**< clock */
#define MIDI_CMD_COMMON_START		0xfa	/**< start */
#define MIDI_CMD_COMMON_CONTINUE	0xfb	/**< continue */
#define MIDI_CMD_COMMON_STOP		0xfc	/**< stop */
#define MIDI_CMD_COMMON_SENSING		0xfe	/**< active sensing */
#define MIDI_CMD_COMMON_RESET		0xff	/**< reset */

/** \} */

/**
 * \defgroup MIDI_Controllers MIDI Controllers
 * MIDI controller numbers.
 * \{
 */

#define MIDI_CTL_MSB_BANK		0x00	/**< Bank selection */
#define MIDI_CTL_MSB_MODWHEEL         	0x01	/**< Modulation */
#define MIDI_CTL_MSB_BREATH           	0x02	/**< Breath */
#define MIDI_CTL_MSB_FOOT             	0x04	/**< Foot */
#define MIDI_CTL_MSB_PORTAMENTO_TIME 	0x05	/**< Portamento time */
#define MIDI_CTL_MSB_DATA_ENTRY		0x06	/**< Data entry */
#define MIDI_CTL_MSB_MAIN_VOLUME      	0x07	/**< Main volume */
#define MIDI_CTL_MSB_BALANCE          	0x08	/**< Balance */
#define MIDI_CTL_MSB_PAN              	0x0a	/**< Panpot */
#define MIDI_CTL_MSB_EXPRESSION       	0x0b	/**< Expression */
#define MIDI_CTL_MSB_EFFECT1		0x0c	/**< Effect1 */
#define MIDI_CTL_MSB_EFFECT2		0x0d	/**< Effect2 */
#define MIDI_CTL_MSB_GENERAL_PURPOSE1 	0x10	/**< General purpose 1 */
#define MIDI_CTL_MSB_GENERAL_PURPOSE2 	0x11	/**< General purpose 2 */
#define MIDI_CTL_MSB_GENERAL_PURPOSE3 	0x12	/**< General purpose 3 */
#define MIDI_CTL_MSB_GENERAL_PURPOSE4 	0x13	/**< General purpose 4 */
#define MIDI_CTL_LSB_BANK		0x20	/**< Bank selection */
#define MIDI_CTL_LSB_MODWHEEL        	0x21	/**< Modulation */
#define MIDI_CTL_LSB_BREATH           	0x22	/**< Breath */
#define MIDI_CTL_LSB_FOOT             	0x24	/**< Foot */
#define MIDI_CTL_LSB_PORTAMENTO_TIME 	0x25	/**< Portamento time */
#define MIDI_CTL_LSB_DATA_ENTRY		0x26	/**< Data entry */
#define MIDI_CTL_LSB_MAIN_VOLUME      	0x27	/**< Main volume */
#define MIDI_CTL_LSB_BALANCE          	0x28	/**< Balance */
#define MIDI_CTL_LSB_PAN              	0x2a	/**< Panpot */
#define MIDI_CTL_LSB_EXPRESSION       	0x2b	/**< Expression */
#define MIDI_CTL_LSB_EFFECT1		0x2c	/**< Effect1 */
#define MIDI_CTL_LSB_EFFECT2		0x2d	/**< Effect2 */
#define MIDI_CTL_LSB_GENERAL_PURPOSE1 	0x30	/**< General purpose 1 */
#define MIDI_CTL_LSB_GENERAL_PURPOSE2 	0x31	/**< General purpose 2 */
#define MIDI_CTL_LSB_GENERAL_PURPOSE3 	0x32	/**< General purpose 3 */
#define MIDI_CTL_LSB_GENERAL_PURPOSE4 	0x33	/**< General purpose 4 */
#define MIDI_CTL_SUSTAIN              	0x40	/**< Sustain pedal */
#define MIDI_CTL_PORTAMENTO           	0x41	/**< Portamento */
#define MIDI_CTL_SOSTENUTO            	0x42	/**< Sostenuto */
#define MIDI_CTL_SUSTENUTO            	0x42	/**< Sostenuto (a typo in the older version) */
#define MIDI_CTL_SOFT_PEDAL           	0x43	/**< Soft pedal */
#define MIDI_CTL_LEGATO_FOOTSWITCH	0x44	/**< Legato foot switch */
#define MIDI_CTL_HOLD2                	0x45	/**< Hold2 */
#define MIDI_CTL_SC1_SOUND_VARIATION	0x46	/**< SC1 Sound Variation */
#define MIDI_CTL_SC2_TIMBRE		0x47	/**< SC2 Timbre */
#define MIDI_CTL_SC3_RELEASE_TIME	0x48	/**< SC3 Release Time */
#define MIDI_CTL_SC4_ATTACK_TIME	0x49	/**< SC4 Attack Time */
#define MIDI_CTL_SC5_BRIGHTNESS		0x4a	/**< SC5 Brightness */
#define MIDI_CTL_SC6			0x4b	/**< SC6 */
#define MIDI_CTL_SC7			0x4c	/**< SC7 */
#define MIDI_CTL_SC8			0x4d	/**< SC8 */
#define MIDI_CTL_SC9			0x4e	/**< SC9 */
#define MIDI_CTL_SC10			0x4f	/**< SC10 */
#define MIDI_CTL_GENERAL_PURPOSE5     	0x50	/**< General purpose 5 */
#define MIDI_CTL_GENERAL_PURPOSE6     	0x51	/**< General purpose 6 */
#define MIDI_CTL_GENERAL_PURPOSE7     	0x52	/**< General purpose 7 */
#define MIDI_CTL_GENERAL_PURPOSE8     	0x53	/**< General purpose 8 */
#define MIDI_CTL_PORTAMENTO_CONTROL	0x54	/**< Portamento control */
#define MIDI_CTL_E1_REVERB_DEPTH	0x5b	/**< E1 Reverb Depth */
#define MIDI_CTL_E2_TREMOLO_DEPTH	0x5c	/**< E2 Tremolo Depth */
#define MIDI_CTL_E3_CHORUS_DEPTH	0x5d	/**< E3 Chorus Depth */
#define MIDI_CTL_E4_DETUNE_DEPTH	0x5e	/**< E4 Detune Depth */
#define MIDI_CTL_E5_PHASER_DEPTH	0x5f	/**< E5 Phaser Depth */
#define MIDI_CTL_DATA_INCREMENT       	0x60	/**< Data Increment */
#define MIDI_CTL_DATA_DECREMENT       	0x61	/**< Data Decrement */
#define MIDI_CTL_NONREG_PARM_NUM_LSB  	0x62	/**< Non-registered parameter number */
#define MIDI_CTL_NONREG_PARM_NUM_MSB  	0x63	/**< Non-registered parameter number */
#define MIDI_CTL_REGIST_PARM_NUM_LSB  	0x64	/**< Registered parameter number */
#define MIDI_CTL_REGIST_PARM_NUM_MSB	0x65	/**< Registered parameter number */
#define MIDI_CTL_ALL_SOUNDS_OFF		0x78	/**< All sounds off */
#define MIDI_CTL_RESET_CONTROLLERS	0x79	/**< Reset Controllers */
#define MIDI_CTL_LOCAL_CONTROL_SWITCH	0x7a	/**< Local control switch */
#define MIDI_CTL_ALL_NOTES_OFF		0x7b	/**< All notes off */
#define MIDI_CTL_OMNI_OFF		0x7c	/**< Omni off */
#define MIDI_CTL_OMNI_ON		0x7d	/**< Omni on */
#define MIDI_CTL_MONO1			0x7e	/**< Mono1 */
#define MIDI_CTL_MONO2			0x7f	/**< Mono2 */

/** \} */

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* __ALSA_ASOUNDEF_H */
