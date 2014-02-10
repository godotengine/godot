#ifndef AUDIO_DRIVER_ANDROID_H
#define AUDIO_DRIVER_ANDROID_H

#include "servers/audio/audio_server_sw.h"

class AudioDriverAndroid : public AudioDriverSW {


	static AudioDriverAndroid* s_ad;

public:

	void set_singleton();

	virtual const char* get_name() const;

	virtual Error init();
	virtual void start();
	virtual int get_mix_rate() const ;
	virtual OutputFormat get_output_format() const;
	virtual void lock();
	virtual void unlock();
	virtual void finish();


	AudioDriverAndroid();
};

#endif // AUDIO_DRIVER_ANDROID_H
