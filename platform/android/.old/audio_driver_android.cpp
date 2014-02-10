#include "audio_driver_android.h"

AudioDriverAndroid* AudioDriverAndroid::s_ad=NULL;

const char* AudioDriverAndroid::get_name() const {

	return "Android";
}

#if 0
int AudioDriverAndroid::thread_func(SceSize args, void *argp) {

	AudioDriverAndroid* ad = s_ad;
	sceAudioOutput2Reserve(AUDIO_OUTPUT_SAMPLE);

	int half=0;
	while(!ad->exit_thread) {

		int16_t *ptr = &ad->outbuff[AUDIO_OUTPUT_SAMPLE*2*half];



		if (!ad->active) {

			for(int i=0;i<AUDIO_OUTPUT_SAMPLE*2;i++) {
				ptr[i]=0;
			}

		} else {

		//printf("samples: %i\n",AUDIO_OUTPUT_SAMPLE);
			ad->lock();

			ad->audio_server_process(AUDIO_OUTPUT_SAMPLE,ad->outbuff_32);

			ad->unlock();

			const int32_t* src_buff=ad->outbuff_32;

			for(int i=0;i<AUDIO_OUTPUT_SAMPLE*2;i++) {

				ptr[i]=src_buff[i]>>16;
			}
		}


		/* Output 16-bit PCM STEREO data that is in pcmBuf without changing the volume */
		sceAudioOutput2OutputBlocking(
			   SCE_AUDIO_VOLUME_0dB*3, //0db at 0x8000, that's obvious
			 ptr
		);

		if (half)
			half=0;
		else
			half=1;

	}

	sceAudioOutput2Release();

	sceKernelExitThread(SCE_KERNEL_EXIT_SUCCESS);
	ad->thread_exited=true;
	return SCE_KERNEL_EXIT_SUCCESS;

}

#endif
Error AudioDriverAndroid::init(){

	return OK;


}
void AudioDriverAndroid::start(){


}
int AudioDriverAndroid::get_mix_rate() const {

	return 44100;
}
AudioDriverSW::OutputFormat AudioDriverAndroid::get_output_format() const{

	return OUTPUT_STEREO;
}
void AudioDriverAndroid::lock(){


}
void AudioDriverAndroid::unlock() {


}
void AudioDriverAndroid::finish(){

	}


AudioDriverAndroid::AudioDriverAndroid()
{
	s_ad=this;
}

