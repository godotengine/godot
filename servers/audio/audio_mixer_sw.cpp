/*************************************************************************/
/*  audio_mixer_sw.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_mixer_sw.h"
#include "print_string.h"
#include "os/os.h"
//TODO implement FAST_AUDIO macro

#ifdef FAST_AUDIO
#define NO_REVERB
#endif

template<class Depth,bool is_stereo,bool is_ima_adpcm,bool use_filter,bool use_fx,AudioMixerSW::InterpolationType type,AudioMixerSW::MixChannels mix_mode>
void AudioMixerSW::do_resample(const Depth* p_src, int32_t *p_dst, ResamplerState *p_state) {

	// this function will be compiled branchless by any decent compiler

	int32_t final,final_r,next,next_r;
	int32_t *reverb_dst = p_state->reverb_buffer;
	while (p_state->amount--) {

		int32_t pos=p_state->pos >> MIX_FRAC_BITS;
		if (is_stereo && !is_ima_adpcm)
			pos<<=1;

		if (is_ima_adpcm) {

			int sample_pos = pos + p_state->ima_adpcm[0].window_ofs;

			while(sample_pos>p_state->ima_adpcm[0].last_nibble) {


				static const int16_t _ima_adpcm_step_table[89] = {
					7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
					19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
					50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
					130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
					337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
					876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
					2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
					5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
					15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
				};

				static const int8_t _ima_adpcm_index_table[16] = {
					-1, -1, -1, -1, 2, 4, 6, 8,
					-1, -1, -1, -1, 2, 4, 6, 8
				};

				for(int i=0;i<(is_stereo?2:1);i++) {


					int16_t nibble,diff,step;

					p_state->ima_adpcm[i].last_nibble++;
					const uint8_t *src_ptr=p_state->ima_adpcm[i].ptr;


					uint8_t nbb = src_ptr[ (p_state->ima_adpcm[i].last_nibble>>1) *  (is_stereo?2:1) + i ];
					nibble = (p_state->ima_adpcm[i].last_nibble&1)?(nbb>>4):(nbb&0xF);
					step=_ima_adpcm_step_table[p_state->ima_adpcm[i].step_index];


					p_state->ima_adpcm[i].step_index += _ima_adpcm_index_table[nibble];
					if (p_state->ima_adpcm[i].step_index<0)
						p_state->ima_adpcm[i].step_index=0;
					if (p_state->ima_adpcm[i].step_index>88)
						p_state->ima_adpcm[i].step_index=88;

					diff = step >> 3 ;
					if (nibble & 1)
						diff += step >> 2 ;
					if (nibble & 2)
						diff += step >> 1 ;
					if (nibble & 4)
						diff += step ;
					if (nibble & 8)
						diff = -diff ;

					p_state->ima_adpcm[i].predictor+=diff;
					if (p_state->ima_adpcm[i].predictor<-0x8000)
						p_state->ima_adpcm[i].predictor=-0x8000;
					else if (p_state->ima_adpcm[i].predictor>0x7FFF)
						p_state->ima_adpcm[i].predictor=0x7FFF;


					/* store loop if there */
					if (p_state->ima_adpcm[i].last_nibble==p_state->ima_adpcm[i].loop_pos) {

						p_state->ima_adpcm[i].loop_step_index = p_state->ima_adpcm[i].step_index;
						p_state->ima_adpcm[i].loop_predictor = p_state->ima_adpcm[i].predictor;
					}

					//printf("%i - %i - pred %i\n",int(p_state->ima_adpcm[i].last_nibble),int(nibble),int(p_state->ima_adpcm[i].predictor));

				}

			}

			final=p_state->ima_adpcm[0].predictor;
			if (is_stereo) {
				final_r=p_state->ima_adpcm[1].predictor;
			}

		} else {
			final=p_src[pos];
			if (is_stereo)
				final_r=p_src[pos+1];

			if (sizeof(Depth)==1) { /* conditions will not exist anymore when compiled! */
				final<<=8;
				if (is_stereo)
					final_r<<=8;
			}

			if (type==INTERPOLATION_LINEAR) {

				if (is_stereo) {

					next=p_src[pos+2];
					next_r=p_src[pos+3];
				} else {
					next=p_src[pos+1];
				}

				if (sizeof(Depth)==1) {
					next<<=8;
					if (is_stereo)
						next_r<<=8;
				}

				int32_t frac=int32_t(p_state->pos&MIX_FRAC_MASK);

				final=final+((next-final)*frac >> MIX_FRAC_BITS);
				if (is_stereo)
					final_r=final_r+((next_r-final_r)*frac >> MIX_FRAC_BITS);
			}
		}

		if (use_filter) {

			Channel::Mix::Filter *f = p_state->filter_l;
			float finalf=final;
			float pre = finalf;
			finalf = ((finalf*p_state->coefs.b0) + (f->hb[0]*p_state->coefs.b1)  + (f->hb[1]*p_state->coefs.b2) + (f->ha[0]*p_state->coefs.a1) + (f->ha[1]*p_state->coefs.a2)
				  );

			f->ha[1]=f->ha[0];
			f->hb[1]=f->hb[0];
			f->hb[0]=pre;
			f->ha[0]=finalf;

			final=Math::fast_ftoi(finalf);

			if (is_stereo) {

				f = p_state->filter_r;
				finalf=final_r;
				pre = finalf;
				finalf = ((finalf*p_state->coefs.b0) + (f->hb[0]*p_state->coefs.b1)  + (f->hb[1]*p_state->coefs.b2) + (f->ha[0]*p_state->coefs.a1) + (f->ha[1]*p_state->coefs.a2)
					  );
				f->ha[1]=f->ha[0];
				f->hb[1]=f->hb[0];
				f->hb[0]=pre;
				f->ha[0]=finalf;

				final_r=Math::fast_ftoi(finalf);

			}

			p_state->coefs.b0+=p_state->coefs_inc.b0;
			p_state->coefs.b1+=p_state->coefs_inc.b1;
			p_state->coefs.b2+=p_state->coefs_inc.b2;
			p_state->coefs.a1+=p_state->coefs_inc.a1;
			p_state->coefs.a2+=p_state->coefs_inc.a2;
		}

		if (!is_stereo) {
			final_r=final; //copy to right channel if stereo
		}

		//convert back to 24 bits and mix to buffers

		if (mix_mode==MIX_STEREO) {
			*p_dst++ +=(final*(p_state->vol[0]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
			*p_dst++ +=(final_r*(p_state->vol[1]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;

			p_state->vol[0]+=p_state->vol_inc[0];
			p_state->vol[1]+=p_state->vol_inc[1];

			if (use_fx) {
				*reverb_dst++ +=(final*(p_state->reverb_vol[0]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				*reverb_dst++ +=(final_r*(p_state->reverb_vol[1]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				p_state->reverb_vol[0]+=p_state->reverb_vol_inc[0];
				p_state->reverb_vol[1]+=p_state->reverb_vol_inc[1];
			}


		} else if (mix_mode==MIX_QUAD) {

			*p_dst++ +=(final*(p_state->vol[0]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
			*p_dst++ +=(final_r*(p_state->vol[1]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;

			*p_dst++ +=(final*(p_state->vol[2]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
			*p_dst++ +=(final_r*(p_state->vol[3]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;

			p_state->vol[0]+=p_state->vol_inc[0];
			p_state->vol[1]+=p_state->vol_inc[1];
			p_state->vol[2]+=p_state->vol_inc[2];
			p_state->vol[3]+=p_state->vol_inc[3];

			if (use_fx) {
				*reverb_dst++ +=(final*(p_state->reverb_vol[0]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				*reverb_dst++ +=(final_r*(p_state->reverb_vol[1]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				*reverb_dst++ +=(final*(p_state->reverb_vol[2]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				*reverb_dst++ +=(final_r*(p_state->reverb_vol[3]>>MIX_VOLRAMP_FRAC_BITS))>>MIX_VOL_MOVE_TO_24;
				p_state->reverb_vol[0]+=p_state->reverb_vol_inc[0];
				p_state->reverb_vol[1]+=p_state->reverb_vol_inc[1];
				p_state->reverb_vol[2]+=p_state->reverb_vol_inc[2];
				p_state->reverb_vol[3]+=p_state->reverb_vol_inc[3];
			}
		}

		p_state->pos+=p_state->increment;
	}
}


void AudioMixerSW::mix_channel(Channel& c) {


	if (!sample_manager->is_sample(c.sample)) {
		// sample is gone!
		c.active=false;
		return;
	}


	/* some 64-bit fixed point precaches */

	int64_t loop_begin_fp=((int64_t)sample_manager->sample_get_loop_begin(c.sample) << MIX_FRAC_BITS);
	int64_t loop_end_fp=((int64_t)sample_manager->sample_get_loop_end(c.sample) << MIX_FRAC_BITS);
	int64_t length_fp=((int64_t)sample_manager->sample_get_length(c.sample) << MIX_FRAC_BITS);
	int64_t begin_limit=(sample_manager->sample_get_loop_format(c.sample)!=AS::SAMPLE_LOOP_NONE)?loop_begin_fp:0;
	int64_t end_limit=(sample_manager->sample_get_loop_format(c.sample)!=AS::SAMPLE_LOOP_NONE)?loop_end_fp:length_fp;
	bool is_stereo=sample_manager->sample_is_stereo(c.sample);

	int32_t todo=mix_chunk_size;
//	int mixed=0;
	bool use_filter=false;

	ResamplerState rstate;

	/* compute voume ramps, increment, etc */



	for(int i=0;i<mix_channels;i++) {
		c.mix.old_vol[i]=c.mix.vol[i];
		c.mix.old_reverb_vol[i]=c.mix.reverb_vol[i];
		c.mix.old_chorus_vol[i]=c.mix.chorus_vol[i];
	}

	float vol = c.vol*channel_nrg;

	float reverb_vol = c.reverb_send*channel_nrg;
	float chorus_vol = c.chorus_send*channel_nrg;

	if (mix_channels==2) {
		//stereo pan
		float pan = c.pan * 0.5 + 0.5;
		float panv[2]={
			(1.0 - pan)*(1<<MIX_VOL_FRAC_BITS),
			(pan)*(1<<MIX_VOL_FRAC_BITS)
		};

		for(int i=0;i<2;i++) {

			c.mix.vol[i]=Math::fast_ftoi(vol*panv[i]);
			c.mix.reverb_vol[i]=Math::fast_ftoi(reverb_vol*panv[i]);
			c.mix.chorus_vol[i]=Math::fast_ftoi(chorus_vol*panv[i]);
		}

	} else {
		//qudra pan
		float panx = c.pan * 0.5 + 0.5;
		float pany = c.depth * 0.5 + 0.5;
		// with this model every speaker plays at 0.25 energy at the center.. i'm not sure if it's correct but it seems to be balanced
		float panv[4]={
			(1.0-pany)*(1.0-panx)*(1<<MIX_VOL_FRAC_BITS),
			(1.0-pany)*(    panx)*(1<<MIX_VOL_FRAC_BITS),
			(    pany)*(1.0-panx)*(1<<MIX_VOL_FRAC_BITS),
			(    pany)*(    panx)*(1<<MIX_VOL_FRAC_BITS)
		};

		for(int i=0;i<4;i++) {

			c.mix.vol[i]=Math::fast_ftoi(vol*panv[i]);
			c.mix.reverb_vol[i]=Math::fast_ftoi(reverb_vol*panv[i]);
			c.mix.chorus_vol[i]=Math::fast_ftoi(chorus_vol*panv[i]);
		}

	}

	if (c.first_mix) { // avoid ramp up

		for(int i=0;i<mix_channels;i++) {
			c.mix.old_vol[i]=c.mix.vol[i];
			c.mix.old_reverb_vol[i]=c.mix.reverb_vol[i];
			c.mix.old_chorus_vol[i]=c.mix.chorus_vol[i];
		}

		c.first_mix=false;
	}



	Channel::Filter::Coefs filter_coefs;
	Channel::Filter::Coefs filter_inc;

	if (c.filter.type!=AudioMixer::FILTER_NONE) {

		filter_coefs=c.filter.old_coefs;
		filter_inc.b0=(c.filter.coefs.b0-filter_coefs.b0)/(1<<mix_chunk_bits);
		filter_inc.b1=(c.filter.coefs.b1-filter_coefs.b1)/(1<<mix_chunk_bits);
		filter_inc.b2=(c.filter.coefs.b2-filter_coefs.b2)/(1<<mix_chunk_bits);
		filter_inc.a1=(c.filter.coefs.a1-filter_coefs.a1)/(1<<mix_chunk_bits);
		filter_inc.a2=(c.filter.coefs.a2-filter_coefs.a2)/(1<<mix_chunk_bits);
		use_filter=true;
	}

	if (c.mix.increment>0)
		c.mix.increment=((int64_t)c.speed<<MIX_FRAC_BITS)/mix_rate;
	else
		c.mix.increment=-((int64_t)c.speed<<MIX_FRAC_BITS)/mix_rate;

	//volume ramp


	for(int i=0;i<mix_channels;i++) {
		rstate.vol_inc[i]=((c.mix.vol[i]-c.mix.old_vol[i])<<MIX_VOLRAMP_FRAC_BITS)>>mix_chunk_bits;
		rstate.vol[i]=c.mix.old_vol[i]<<MIX_VOLRAMP_FRAC_BITS;
		rstate.reverb_vol_inc[i]=((c.mix.reverb_vol[i]-c.mix.old_reverb_vol[i])<<MIX_VOLRAMP_FRAC_BITS)>>mix_chunk_bits;
		rstate.reverb_vol[i]=c.mix.old_reverb_vol[i]<<MIX_VOLRAMP_FRAC_BITS;
		rstate.chorus_vol_inc[i]=((c.mix.chorus_vol[i]-c.mix.old_chorus_vol[i])<<MIX_VOLRAMP_FRAC_BITS)>>mix_chunk_bits;
		rstate.chorus_vol[i]=c.mix.old_chorus_vol[i]<<MIX_VOLRAMP_FRAC_BITS;
	}


	//looping

	AS::SampleLoopFormat loop_format=sample_manager->sample_get_loop_format(c.sample);
	AS::SampleFormat format=sample_manager->sample_get_format(c.sample);

	bool use_fx=false;

	if (fx_enabled) {

		for(int i=0;i<mix_channels;i++) {
			if (c.mix.old_reverb_vol[i] || c.mix.reverb_vol[i] || c.mix.old_chorus_vol[i] || c.mix.chorus_vol[i] ) {
				use_fx=true;
				break;
			}
		}
	}

	/* audio data */

	const void *data=sample_manager->sample_get_data_ptr(c.sample);
	int32_t *dst_buff=mix_buffer;

#ifndef NO_REVERB
	rstate.reverb_buffer=reverb_state[c.reverb_room].buffer;
#endif

	/* @TODO validar loops al registrar? */

	rstate.coefs=filter_coefs;
	rstate.coefs_inc=filter_inc;
	rstate.filter_l=&c.mix.filter_l;
	rstate.filter_r=&c.mix.filter_r;

	if (format==AS::SAMPLE_FORMAT_IMA_ADPCM) {

		rstate.ima_adpcm=c.mix.ima_adpcm;
		if (loop_format!=AS::SAMPLE_LOOP_NONE) {
			c.mix.ima_adpcm[0].loop_pos=loop_begin_fp>>MIX_FRAC_BITS;
			c.mix.ima_adpcm[1].loop_pos=loop_begin_fp>>MIX_FRAC_BITS;
			loop_format=AS::SAMPLE_LOOP_FORWARD;
		}
	}

	while (todo>0) {

		int64_t limit=0;
		int32_t target=0,aux=0;

		/** LOOP CHECKING **/

		if ( c.mix.increment < 0 ) {
			/* going backwards */

			if (  loop_format!=AS::SAMPLE_LOOP_NONE && c.mix.offset < loop_begin_fp ) {
				/* loopstart reached */
				if ( loop_format==AS::SAMPLE_LOOP_PING_PONG ) {
					/* bounce ping pong */
					c.mix.offset= loop_begin_fp + ( loop_begin_fp-c.mix.offset );
					c.mix.increment=-c.mix.increment;
				} else {
					/* go to loop-end */
					c.mix.offset=loop_end_fp-(loop_begin_fp-c.mix.offset);
				}
			} else {
				/* check for sample not reaching begining */
				if(c.mix.offset < 0) {

					c.active=false;
					break;
				}
			}
		} else {
			/* going forward */
			if(  loop_format!=AS::SAMPLE_LOOP_NONE && c.mix.offset >= loop_end_fp ) {
				/* loopend reached */

				if ( loop_format==AS::SAMPLE_LOOP_PING_PONG ) {
					/* bounce ping pong */
					c.mix.offset=loop_end_fp-(c.mix.offset-loop_end_fp);
					c.mix.increment=-c.mix.increment;
				} else {
					/* go to loop-begin */

					if (format==AS::SAMPLE_FORMAT_IMA_ADPCM) {
						for(int i=0;i<2;i++) {
							c.mix.ima_adpcm[i].step_index=c.mix.ima_adpcm[i].loop_step_index;
							c.mix.ima_adpcm[i].predictor=c.mix.ima_adpcm[i].loop_predictor;
							c.mix.ima_adpcm[i].last_nibble=loop_begin_fp>>MIX_FRAC_BITS;
						}
						c.mix.offset=loop_begin_fp;
					} else {
						c.mix.offset=loop_begin_fp+(c.mix.offset-loop_end_fp);
					}

				}
			} else {
				/* no loop, check for end of sample */
				if(c.mix.offset >= length_fp) {

					c.active=false;
					break;
				}
			}
		}

		/** MIXCOUNT COMPUTING **/

		/* next possible limit (looppoints or sample begin/end */
		limit=(c.mix.increment < 0) ?begin_limit:end_limit;

		/* compute what is shorter, the todo or the limit? */
		aux=(limit-c.mix.offset)/c.mix.increment+1;
		target=(aux<todo)?aux:todo; /* mix target is the shorter buffer */

		/* check just in case */
		if ( target<=0 ) {
			c.active=false;
			break;
		}

		todo-=target;

		int32_t offset=c.mix.offset&mix_chunk_mask; /* strip integer */
		c.mix.offset-=offset;

		rstate.increment=c.mix.increment;
		rstate.amount=target;
		rstate.pos=offset;

/* Macros to call the resample function for all possibilities, creating a dedicated-non branchy function call for each thanks to template magic*/

#define CALL_RESAMPLE_FUNC( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	do_resample<m_depth,m_stereo,m_ima_adpcm, m_use_filter,m_use_fx,m_interp, m_mode>(\
		src_ptr,\
		dst_buff,&rstate);


#define CALL_RESAMPLE_INTERP( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	if(m_interp==INTERPOLATION_RAW) {\
		CALL_RESAMPLE_FUNC(m_depth,m_stereo, m_ima_adpcm,m_use_filter,m_use_fx,INTERPOLATION_RAW,m_mode);\
	} else if(m_interp==INTERPOLATION_LINEAR) {\
		CALL_RESAMPLE_FUNC(m_depth,m_stereo, m_ima_adpcm,m_use_filter,m_use_fx,INTERPOLATION_LINEAR,m_mode);\
	} else if(m_interp==INTERPOLATION_CUBIC) {\
		CALL_RESAMPLE_FUNC(m_depth,m_stereo, m_ima_adpcm,m_use_filter,m_use_fx,INTERPOLATION_CUBIC,m_mode);\
	}\

#define CALL_RESAMPLE_FX( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	if(m_use_fx) {\
		CALL_RESAMPLE_INTERP(m_depth,m_stereo, m_ima_adpcm,m_use_filter,true,m_interp, m_mode);\
	} else {\
		CALL_RESAMPLE_INTERP(m_depth,m_stereo, m_ima_adpcm,m_use_filter,false,m_interp, m_mode);\
	}\


#define CALL_RESAMPLE_FILTER( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	if(m_use_filter) {\
		CALL_RESAMPLE_FX(m_depth,m_stereo, m_ima_adpcm,true,m_use_fx,m_interp, m_mode);\
	} else {\
		CALL_RESAMPLE_FX(m_depth,m_stereo, m_ima_adpcm,false,m_use_fx,m_interp, m_mode);\
	}\

#define CALL_RESAMPLE_STEREO( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	if(m_stereo) {\
		CALL_RESAMPLE_FILTER(m_depth,true,m_ima_adpcm, m_use_filter,m_use_fx,m_interp, m_mode);\
	} else {\
		CALL_RESAMPLE_FILTER(m_depth,false,m_ima_adpcm,m_use_filter,m_use_fx,m_interp, m_mode);\
	}\

#define CALL_RESAMPLE_MODE( m_depth, m_stereo, m_ima_adpcm, m_use_filter,  m_use_fx, m_interp, m_mode)\
	if(m_mode==MIX_STEREO) {\
		CALL_RESAMPLE_STEREO(m_depth,m_stereo, m_ima_adpcm,m_use_filter,m_use_fx,m_interp, MIX_STEREO);\
	} else {\
		CALL_RESAMPLE_STEREO(m_depth,m_stereo, m_ima_adpcm,m_use_filter,m_use_fx,m_interp, MIX_QUAD);\
	}\




		if (format==AS::SAMPLE_FORMAT_PCM8) {

			int8_t *src_ptr =  &((int8_t*)data)[(c.mix.offset >> MIX_FRAC_BITS)<<(is_stereo?1:0) ];
			CALL_RESAMPLE_MODE(int8_t,is_stereo,false,use_filter,use_fx,interpolation_type,mix_channels);

		} else if (format==AS::SAMPLE_FORMAT_PCM16) {
			int16_t *src_ptr =  &((int16_t*)data)[(c.mix.offset >> MIX_FRAC_BITS)<<(is_stereo?1:0) ];
			CALL_RESAMPLE_MODE(int16_t,is_stereo,false,use_filter,use_fx,interpolation_type,mix_channels);

		} else if (format==AS::SAMPLE_FORMAT_IMA_ADPCM) {
			for(int i=0;i<2;i++) {
				c.mix.ima_adpcm[i].window_ofs=c.mix.offset>>MIX_FRAC_BITS;
				c.mix.ima_adpcm[i].ptr=(const uint8_t*)data;
			}
			int8_t *src_ptr =  NULL;
			CALL_RESAMPLE_MODE(int8_t,is_stereo,true,use_filter,use_fx,interpolation_type,mix_channels);

		}

		c.mix.offset+=rstate.pos;
		dst_buff+=target*mix_channels;
		rstate.reverb_buffer+=target*mix_channels;
	}

	c.filter.old_coefs=c.filter.coefs;
}

void AudioMixerSW::mix_chunk() {

	ERR_FAIL_COND(mix_chunk_left);

	inside_mix=true;

	// emit tick in usecs
	for (int i=0;i<mix_chunk_size*mix_channels;i++) {

		mix_buffer[i]=0;
	}
#ifndef NO_REVERB
	for(int i=0;i<max_reverbs;i++)
		reverb_state[i].used_in_chunk=false;
#endif


	audio_mixer_chunk_call(mix_chunk_size);

	int ac=0;
	for (int i=0;i<MAX_CHANNELS;i++) {

		if (!channels[i].active)
			continue;
		ac++;

		/* process volume */
		Channel&c=channels[i];
#ifndef NO_REVERB
		bool has_reverb = c.reverb_send>CMP_EPSILON && fx_enabled;
		if (has_reverb || c.had_prev_reverb) {

			if (!reverb_state[c.reverb_room].used_in_chunk) {
				//zero the room
				int32_t *buff = reverb_state[c.reverb_room].buffer;
				int len = mix_chunk_size*mix_channels;
				for (int j=0;j<len;j++) {

					buff[j]=0; // buffer in use, clear it for appending
				}
				reverb_state[c.reverb_room].used_in_chunk=true;
			}
		}
#else
		bool has_reverb = false;
#endif
		bool has_chorus = c.chorus_send>CMP_EPSILON && fx_enabled;


		mix_channel(c);

		c.had_prev_reverb=has_reverb;
		c.had_prev_chorus=has_chorus;

	}

	//process reverb
#ifndef NO_REVERB
	if (fx_enabled) {


		for(int i=0;i<max_reverbs;i++) {

			if (!reverb_state[i].enabled && !reverb_state[i].used_in_chunk)
				continue; //this reverb is not in use

			int32_t *src=NULL;

			if (reverb_state[i].used_in_chunk)
				src=reverb_state[i].buffer;
			else
				src=zero_buffer;

			bool in_use=false;

			int passes=mix_channels/2;

			for(int j=0;j<passes;j++) {

				if (reverb_state[i].reverb[j].process((int*)&src[j*2],(int*)&mix_buffer[j*2],mix_chunk_size,passes))
					in_use=true;
			}

			if (in_use) {
				reverb_state[i].enabled=true;
				reverb_state[i].frames_idle=0;
				//copy data over

			} else {
				reverb_state[i].frames_idle+=mix_chunk_size;
				if (false) { // go idle because too many frames passed
					//disable this reverb, as nothing important happened on it
					reverb_state[i].enabled=false;
					reverb_state[i].frames_idle=0;
				}
			}

		}
	}
#endif
	mix_chunk_left=mix_chunk_size;
	inside_mix=false;
}

int AudioMixerSW::mix(int32_t *p_buffer,int p_frames) {

	int todo=p_frames;
	int mixes=0;

	while(todo) {


		if (!mix_chunk_left) {

			if (step_callback)
				step_callback(step_udata);
			mix_chunk();
			mixes++;
		}

		int to_mix=MIN(mix_chunk_left,todo);
		int from=mix_chunk_size-mix_chunk_left;

		for (int i=0;i<to_mix*2;i++) {

			(*p_buffer++)=mix_buffer[from*2+i];
		}

		mix_chunk_left-=to_mix;
		todo-=to_mix;
	}

	return mixes;
}

uint64_t AudioMixerSW::get_step_usecs() const {

	double mct = (1<<mix_chunk_bits)/double(mix_rate);
	return mct*1000000.0;
}

int AudioMixerSW::_get_channel(ChannelID p_channel) const {

	if (p_channel<0) {
		return -1;
	}

	int idx=p_channel%MAX_CHANNELS;
	int check=p_channel/MAX_CHANNELS;
	ERR_FAIL_INDEX_V(idx,MAX_CHANNELS,-1);
	if (channels[idx].check!=check) {
		return -1;
	}
	if (!channels[idx].active) {
		return -1;
	}

	return idx;
}

AudioMixer::ChannelID AudioMixerSW::channel_alloc(RID p_sample) {

	ERR_FAIL_COND_V( !sample_manager->is_sample(p_sample), INVALID_CHANNEL );


	int index=-1;
	for (int i=0;i<MAX_CHANNELS;i++) {

		if (!channels[i].active) {
			index=i;
			break;
		}
	}

	if (index==-1)
		return INVALID_CHANNEL;

	Channel &c=channels[index];

	// init variables
	c.sample=p_sample;
	c.vol=1;
	c.pan=0;
	c.depth=0;
	c.height=0;
	c.chorus_send=0;
	c.reverb_send=0;
	c.reverb_room=REVERB_HALL;
	c.positional=false;
	c.filter.type=FILTER_NONE;
	c.speed=sample_manager->sample_get_mix_rate(p_sample);
	c.active=true;
	c.check=channel_id_count++;
	c.first_mix=true;

	// init mix variables

	c.mix.offset=0;
	c.mix.increment=1;
	//zero everything when this errors
	for(int i=0;i<4;i++) {
		c.mix.vol[i]=0;
		c.mix.reverb_vol[i]=0;
		c.mix.chorus_vol[i]=0;

		c.mix.old_vol[i]=0;
		c.mix.old_reverb_vol[i]=0;
		c.mix.old_chorus_vol[i]=0;
	}

	c.had_prev_chorus=false;
	c.had_prev_reverb=false;
	c.had_prev_vol=false;


	if (sample_manager->sample_get_format(c.sample)==AudioServer::SAMPLE_FORMAT_IMA_ADPCM) {

		for(int i=0;i<2;i++) {
			c.mix.ima_adpcm[i].step_index=0;
			c.mix.ima_adpcm[i].predictor=0;
			c.mix.ima_adpcm[i].loop_step_index=0;
			c.mix.ima_adpcm[i].loop_predictor=0;
			c.mix.ima_adpcm[i].last_nibble=-1;
			c.mix.ima_adpcm[i].loop_pos=0x7FFFFFFF;
			c.mix.ima_adpcm[i].window_ofs=0;
			c.mix.ima_adpcm[i].ptr=NULL;
		}
	}

	ChannelID ret_id = index+c.check*MAX_CHANNELS;

	return ret_id;

}

void AudioMixerSW::channel_set_volume(ChannelID p_channel, float p_gain) {

	if (p_gain>3) // avoid gain going too high
		p_gain=3;
	if (p_gain<0)
		p_gain=0;

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;
	Channel &c = channels[chan];

	//Math::exp( p_db * 0.11512925464970228420089957273422 );
	c.vol=p_gain;

}

void AudioMixerSW::channel_set_pan(ChannelID p_channel, float p_pan, float p_depth,float p_height) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;
	Channel &c = channels[chan];

	c.pan=p_pan;
	c.depth=p_depth;
	c.height=p_height;

}
void AudioMixerSW::channel_set_filter(ChannelID p_channel, FilterType p_type, float p_cutoff, float p_resonance, float p_gain) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c = channels[chan];

	if (c.filter.type==p_type && c.filter.cutoff==p_cutoff && c.filter.resonance==p_resonance && c.filter.gain==p_gain)
		return; //bye


	bool type_changed = p_type!=c.filter.type;

	c.filter.type=p_type;
	c.filter.cutoff=p_cutoff;
	c.filter.resonance=p_resonance;
	c.filter.gain=p_gain;


	AudioFilterSW filter;
	switch(p_type) {
		case FILTER_NONE: {

			return; //do nothing else
		} break;
		case FILTER_LOWPASS: {
			filter.set_mode(AudioFilterSW::LOWPASS);
		} break;
		case FILTER_BANDPASS: {
			filter.set_mode(AudioFilterSW::BANDPASS);
		} break;
		case FILTER_HIPASS: {
			filter.set_mode(AudioFilterSW::HIGHPASS);
		} break;
		case FILTER_NOTCH: {
			filter.set_mode(AudioFilterSW::NOTCH);
		} break;
		case FILTER_PEAK: {
			filter.set_mode(AudioFilterSW::PEAK);
		} break;
		case FILTER_BANDLIMIT: {
			filter.set_mode(AudioFilterSW::BANDLIMIT);
		} break;
		case FILTER_LOW_SHELF: {
			filter.set_mode(AudioFilterSW::LOWSHELF);
		} break;
		case FILTER_HIGH_SHELF: {
			filter.set_mode(AudioFilterSW::HIGHSHELF);
		} break;
	}

	filter.set_cutoff(p_cutoff);
	filter.set_resonance(p_resonance);
	filter.set_gain(p_gain);
	filter.set_sampling_rate(mix_rate);
	filter.set_stages(1);

	AudioFilterSW::Coeffs coefs;
	filter.prepare_coefficients(&coefs);

	if (!type_changed)
		c.filter.old_coefs=c.filter.coefs;

	c.filter.coefs.b0=coefs.b0;
	c.filter.coefs.b1=coefs.b1;
	c.filter.coefs.b2=coefs.b2;
	c.filter.coefs.a1=coefs.a1;
	c.filter.coefs.a2=coefs.a2;


	if (type_changed) {
		//type changed reset filter
		c.filter.old_coefs=c.filter.coefs;
		c.mix.filter_l.ha[0]=0;
		c.mix.filter_l.ha[1]=0;
		c.mix.filter_l.hb[0]=0;
		c.mix.filter_l.hb[1]=0;
		c.mix.filter_r.ha[0]=0;
		c.mix.filter_r.ha[1]=0;
		c.mix.filter_r.hb[0]=0;
		c.mix.filter_r.hb[1]=0;
	}


}
void AudioMixerSW::channel_set_chorus(ChannelID p_channel, float p_chorus ) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c = channels[chan];
	c.chorus_send=p_chorus;

}
void AudioMixerSW::channel_set_reverb(ChannelID p_channel, ReverbRoomType p_room_type, float p_reverb) {

	ERR_FAIL_INDEX(p_room_type,MAX_REVERBS);
	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c = channels[chan];
	c.reverb_room=p_room_type;
	c.reverb_send=p_reverb;

}

void AudioMixerSW::channel_set_mix_rate(ChannelID p_channel, int p_mix_rate) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c = channels[chan];
	c.speed=p_mix_rate;

}
void AudioMixerSW::channel_set_positional(ChannelID p_channel, bool p_positional) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c = channels[chan];
	c.positional=p_positional;
}

float AudioMixerSW::channel_get_volume(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	//Math::log( c.vol ) * 8.6858896380650365530225783783321;
	return c.vol;
}

float AudioMixerSW::channel_get_pan(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.pan;
}
float AudioMixerSW::channel_get_pan_depth(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.depth;
}
float AudioMixerSW::channel_get_pan_height(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.height;

}
AudioMixer::FilterType AudioMixerSW::channel_get_filter_type(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return FILTER_NONE;

	const Channel &c = channels[chan];
	return c.filter.type;
}
float AudioMixerSW::channel_get_filter_cutoff(ChannelID p_channel) const {


	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.filter.cutoff;

}
float AudioMixerSW::channel_get_filter_resonance(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.filter.resonance;

}

float AudioMixerSW::channel_get_filter_gain(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.filter.gain;
}


float AudioMixerSW::channel_get_chorus(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.chorus_send;

}
AudioMixer::ReverbRoomType AudioMixerSW::channel_get_reverb_type(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return REVERB_HALL;

	const Channel &c = channels[chan];
	return c.reverb_room;

}
float AudioMixerSW::channel_get_reverb(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.reverb_send;
}

int AudioMixerSW::channel_get_mix_rate(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return 0;

	const Channel &c = channels[chan];
	return c.speed;
}
bool AudioMixerSW::channel_is_positional(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return false;

	const Channel &c = channels[chan];
	return c.positional;
}

bool AudioMixerSW::channel_is_valid(ChannelID p_channel) const {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return false;
	return channels[chan].active;
}


void AudioMixerSW::channel_free(ChannelID p_channel) {

	int chan = _get_channel(p_channel);
	if (chan<0 || chan >=MAX_CHANNELS)
		return;

	Channel &c=channels[chan];

	if (!c.active)
		return;

	bool has_vol=false;

	for(int i=0;i<mix_channels;i++) {

		if (c.mix.vol[i])
			has_vol=true;
		if (c.mix.reverb_vol[i])
			has_vol=true;
		if (c.mix.chorus_vol[i])
			has_vol=true;
	}
	if (c.active && has_vol && inside_mix) {
		// drive voice to zero, and run a chunk, the VRAMP will fade it good
		c.vol=0;
		c.reverb_send=0;
		c.chorus_send=0;
		mix_channel(c);
	}
	/* @TODO RAMP DOWN ON STOP */
	c.active=false;
}



AudioMixerSW::AudioMixerSW(SampleManagerSW *p_sample_manager,int p_desired_latency_ms,int p_mix_rate,MixChannels p_mix_channels,bool p_use_fx,InterpolationType p_interp,MixStepCallback p_step_callback,void *p_step_udata) {

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("AudioServerSW Params: ");
		print_line(" -mix chans: "+itos(p_mix_channels));
		print_line(" -mix rate: "+itos(p_mix_rate));
		print_line(" -latency: "+itos(p_desired_latency_ms));
		print_line(" -fx: "+itos(p_use_fx));
		print_line(" -interp: "+itos(p_interp));
	}
	sample_manager=p_sample_manager;
	mix_channels=p_mix_channels;
	mix_rate=p_mix_rate;
	step_callback=p_step_callback;
	step_udata=p_step_udata;


	mix_chunk_bits=nearest_shift( p_desired_latency_ms * p_mix_rate / 1000 );

	mix_chunk_size=(1<<mix_chunk_bits);
	mix_chunk_mask=mix_chunk_size-1;
	mix_buffer = memnew_arr(int32_t,mix_chunk_size*mix_channels);
#ifndef NO_REVERB
	zero_buffer = memnew_arr(int32_t,mix_chunk_size*mix_channels);
	for(int i=0;i<mix_chunk_size*mix_channels;i++)
		zero_buffer[i]=0; //zero buffer is zero...

	max_reverbs=MAX_REVERBS;
	int reverberators=mix_channels/2;

	reverb_state = memnew_arr(ReverbState,max_reverbs);
	for(int i=0;i<max_reverbs;i++) {
		reverb_state[i].enabled=false;
		reverb_state[i].reverb = memnew_arr(ReverbSW,reverberators);
		reverb_state[i].buffer = memnew_arr(int32_t,mix_chunk_size*mix_channels);
		reverb_state[i].frames_idle=0;
		for(int j=0;j<reverberators;j++) {
			static ReverbSW::ReverbMode modes[MAX_REVERBS]={ReverbSW::REVERB_MODE_STUDIO_SMALL,ReverbSW::REVERB_MODE_STUDIO_MEDIUM,ReverbSW::REVERB_MODE_STUDIO_LARGE,ReverbSW::REVERB_MODE_HALL};
			reverb_state[i].reverb[j].set_mix_rate(p_mix_rate);
			reverb_state[i].reverb[j].set_mode(modes[i]);
		}

	}
	fx_enabled=p_use_fx;
#else
	fx_enabled=false;
#endif
	mix_chunk_left=0;

	interpolation_type=p_interp;
	channel_id_count=1;
	inside_mix=false;
	channel_nrg=1.0;

}

void AudioMixerSW::set_mixer_volume(float p_volume) {

	channel_nrg=p_volume;
}

AudioMixerSW::~AudioMixerSW() {

	memdelete_arr(mix_buffer);

#ifndef NO_REVERB
	memdelete_arr(zero_buffer);
	for(int i=0;i<max_reverbs;i++) {
		memdelete_arr(reverb_state[i].reverb);
		memdelete_arr(reverb_state[i].buffer);
	}
	memdelete_arr(reverb_state);
#endif


}
