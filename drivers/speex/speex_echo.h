/* Copyright (C) Jean-Marc Valin */
/**
   @file speex_echo.h
   @brief Echo cancellation
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef SPEEX_ECHO_H
#define SPEEX_ECHO_H
/** @defgroup SpeexEchoState SpeexEchoState: Acoustic echo canceller
 *  This is the acoustic echo canceller module.
 *  @{
 */
#include "speex/speex_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Obtain frame size used by the AEC */
#define SPEEX_ECHO_GET_FRAME_SIZE 3

/** Set sampling rate */
#define SPEEX_ECHO_SET_SAMPLING_RATE 24
/** Get sampling rate */
#define SPEEX_ECHO_GET_SAMPLING_RATE 25

/* Can't set window sizes */
/** Get size of impulse response (int32) */
#define SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE 27

/* Can't set window content */
/** Get impulse response (int32[]) */
#define SPEEX_ECHO_GET_IMPULSE_RESPONSE 29

/** Internal echo canceller state. Should never be accessed directly. */
struct SpeexEchoState_;

/** @class SpeexEchoState
 * This holds the state of the echo canceller. You need one per channel. 
*/

/** Internal echo canceller state. Should never be accessed directly. */
typedef struct SpeexEchoState_ SpeexEchoState;

/** Creates a new echo canceller state
 * @param frame_size Number of samples to process at one time (should correspond to 10-20 ms)
 * @param filter_length Number of samples of echo to cancel (should generally correspond to 100-500 ms)
 * @return Newly-created echo canceller state
 */
SpeexEchoState *speex_echo_state_init(int frame_size, int filter_length);

/** Creates a new multi-channel echo canceller state
 * @param frame_size Number of samples to process at one time (should correspond to 10-20 ms)
 * @param filter_length Number of samples of echo to cancel (should generally correspond to 100-500 ms)
 * @param nb_mic Number of microphone channels
 * @param nb_speakers Number of speaker channels
 * @return Newly-created echo canceller state
 */
SpeexEchoState *speex_echo_state_init_mc(int frame_size, int filter_length, int nb_mic, int nb_speakers);

/** Destroys an echo canceller state 
 * @param st Echo canceller state
*/
void speex_echo_state_destroy(SpeexEchoState *st);

/** Performs echo cancellation a frame, based on the audio sent to the speaker (no delay is added
 * to playback in this form)
 *
 * @param st Echo canceller state
 * @param rec Signal from the microphone (near end + far end echo)
 * @param play Signal played to the speaker (received from far end)
 * @param out Returns near-end signal with echo removed
 */
void speex_echo_cancellation(SpeexEchoState *st, const spx_int16_t *rec, const spx_int16_t *play, spx_int16_t *out);

/** Performs echo cancellation a frame (deprecated) */
void speex_echo_cancel(SpeexEchoState *st, const spx_int16_t *rec, const spx_int16_t *play, spx_int16_t *out, spx_int32_t *Yout);

/** Perform echo cancellation using internal playback buffer, which is delayed by two frames
 * to account for the delay introduced by most soundcards (but it could be off!)
 * @param st Echo canceller state
 * @param rec Signal from the microphone (near end + far end echo)
 * @param out Returns near-end signal with echo removed
*/
void speex_echo_capture(SpeexEchoState *st, const spx_int16_t *rec, spx_int16_t *out);

/** Let the echo canceller know that a frame was just queued to the soundcard
 * @param st Echo canceller state
 * @param play Signal played to the speaker (received from far end)
*/
void speex_echo_playback(SpeexEchoState *st, const spx_int16_t *play);

/** Reset the echo canceller to its original state 
 * @param st Echo canceller state
 */
void speex_echo_state_reset(SpeexEchoState *st);

/** Used like the ioctl function to control the echo canceller parameters
 *
 * @param st Echo canceller state
 * @param request ioctl-type request (one of the SPEEX_ECHO_* macros)
 * @param ptr Data exchanged to-from function
 * @return 0 if no error, -1 if request in unknown
 */
int speex_echo_ctl(SpeexEchoState *st, int request, void *ptr);



struct SpeexDecorrState_;

typedef struct SpeexDecorrState_ SpeexDecorrState;


/** Create a state for the channel decorrelation algorithm
    This is useful for multi-channel echo cancellation only 
 * @param rate Sampling rate
 * @param channels Number of channels (it's a bit pointless if you don't have at least 2)
 * @param frame_size Size of the frame to process at ones (counting samples *per* channel)
*/
SpeexDecorrState *speex_decorrelate_new(int rate, int channels, int frame_size);

/** Remove correlation between the channels by modifying the phase and possibly
    adding noise in a way that is not (or little) perceptible.
 * @param st Decorrelator state
 * @param in Input audio in interleaved format
 * @param out Result of the decorrelation (out *may* alias in)
 * @param strength How much alteration of the audio to apply from 0 to 100.
*/
void speex_decorrelate(SpeexDecorrState *st, const spx_int16_t *in, spx_int16_t *out, int strength);

/** Destroy a Decorrelation state 
 * @param st State to destroy
*/
void speex_decorrelate_destroy(SpeexDecorrState *st);


#ifdef __cplusplus
}
#endif


/** @}*/
#endif
