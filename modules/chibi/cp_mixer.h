/*************************************************************************/
/*  cp_mixer.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#ifndef CP_MIXER_H
#define CP_MIXER_H

#include "cp_sample_defs.h"

/**Abstract base class representing a mixer
  *@author Juan Linietsky
  */


/******************************
 mixer.h
 ----------

Abstract base class for the mixer.
This is what the player uses to setup
voices and stuff.. this way
it can be abstracted to hardware
devices or other stuff..
********************************/

class CPSample_ID; /* need this */

class CPMixer {
public: 
	
	enum {
		
		FREQUENCY_BITS=8
		
	};
	
	enum ReverbMode {
		REVERB_MODE_ROOM,
		REVERB_MODE_STUDIO_SMALL,
		REVERB_MODE_STUDIO_MEDIUM,
		REVERB_MODE_STUDIO_LARGE,
		REVERB_MODE_HALL,
		REVERB_MODE_SPACE_ECHO,
		REVERB_MODE_ECHO,
		REVERB_MODE_DELAY,
		REVERB_MODE_HALF_ECHO
	};
		
	/* Callback */	
		
	virtual void set_callback_interval(int p_interval_us)=0; //in usecs, for tracker it's 2500000/tempo
	virtual void set_callback(void (*p_callback)(void*),void *p_userdata)=0;
	
	/* Voice Control */
			
	virtual void setup_voice(int p_voice_index,CPSample_ID p_sample_id,int32_t p_start_index) =0;
	virtual void stop_voice(int p_voice_index) =0;
	virtual void set_voice_frequency(int p_voice_index,int32_t p_freq) =0; //in freq*FREQUENCY_BITS
	virtual void set_voice_panning(int p_voice_index,int p_pan) =0;
	virtual void set_voice_volume(int p_voice_index,int p_vol) =0;
	virtual void set_voice_filter(int p_filter,bool p_enabled,uint8_t p_cutoff, uint8_t p_resonance )=0;
        virtual void set_voice_reverb_send(int p_voice_index,int p_reverb)=0;
	virtual void set_voice_chorus_send(int p_voice_index,int p_chorus)=0; /* 0 - 255 */
	
	virtual void set_reverb_mode(ReverbMode p_mode)=0;
	virtual void set_chorus_params(unsigned int p_delay_ms,unsigned int p_separation_ms,unsigned int p_depth_ms10,unsigned int p_speed_hz10)=0;
	
	
	/* Info retrieving */	
	
	virtual int32_t get_voice_sample_pos_index(int p_voice_index) =0;
	virtual int get_voice_panning(int p_voice_index) =0;
	virtual int get_voice_volume(int p_voice_index) =0;
	virtual CPSample_ID get_voice_sample_id(int p_voice_index) =0;
	virtual bool is_voice_active(int p_voice_index) =0;
	virtual int get_active_voice_count()=0;
	virtual int get_total_voice_count()=0;
	
	
	virtual uint32_t get_mix_frequency()=0; //if mixer is not software, return 0

	/* Methods below only work with software mixers, meant for software-based sound drivers, hardware mixers ignore them */
	virtual int32_t process(int32_t p_frames)=0; /* Call this to process N frames, returns how much it was processed */
	virtual int32_t *get_mixdown_buffer_ptr()=0; /* retrieve what was mixed */
	virtual void set_mix_frequency(int32_t p_mix_frequency)=0;
		
	virtual ~CPMixer() {}
};

#endif
