
#include "audio_stream_ogg_vorbis.h"
#include "thirdparty/stb_vorbis/stb_vorbis.c"
#include "os/file_access.h"


void AudioStreamPlaybackOGGVorbis::_mix_internal(AudioFrame* p_buffer,int p_frames) {

	ERR_FAIL_COND(!active);

	int todo=p_frames;

	while(todo) {

		int mixed = stb_vorbis_get_samples_float_interleaved(ogg_stream,2,(float*)p_buffer,todo*2);
		todo-=mixed;

		if (todo) {
			//end of file!
			if (false) {
				//loop
				seek_pos(0);
				loops++;
			} else {
				for(int i=mixed;i<p_frames;i++) {
					p_buffer[i]=AudioFrame(0,0);
				}
				active=false;
			}
		}
	}


}

float AudioStreamPlaybackOGGVorbis::get_stream_sampling_rate() {

	return vorbis_stream->sample_rate;
}


void AudioStreamPlaybackOGGVorbis::start(float p_from_pos) {

	seek_pos(p_from_pos);
	active=true;
	loops=0;
	_begin_resample();


}

void AudioStreamPlaybackOGGVorbis::stop() {

	active=false;
}
bool AudioStreamPlaybackOGGVorbis::is_playing() const {

	return active;
}

int AudioStreamPlaybackOGGVorbis::get_loop_count() const {

	return loops;
}

float AudioStreamPlaybackOGGVorbis::get_pos() const {

	return float(frames_mixed)/vorbis_stream->sample_rate;
}
void AudioStreamPlaybackOGGVorbis::seek_pos(float p_time) {

	if (!active)
		return;

	stb_vorbis_seek(ogg_stream, uint32_t(p_time*vorbis_stream->sample_rate));
}

float AudioStreamPlaybackOGGVorbis::get_length() const {

	return vorbis_stream->length;
}

AudioStreamPlaybackOGGVorbis::~AudioStreamPlaybackOGGVorbis() {
	if (ogg_alloc.alloc_buffer) {
		AudioServer::get_singleton()->audio_data_free(ogg_alloc.alloc_buffer);
		stb_vorbis_close(ogg_stream);
	}
}

Ref<AudioStreamPlayback> AudioStreamOGGVorbis::instance_playback() {



	Ref<AudioStreamPlaybackOGGVorbis> ovs;
	printf("instance at %p, data %p\n",this,data);

	ERR_FAIL_COND_V(data==NULL,ovs);

	ovs.instance();
	ovs->vorbis_stream=Ref<AudioStreamOGGVorbis>(this);
	ovs->ogg_alloc.alloc_buffer=(char*)AudioServer::get_singleton()->audio_data_alloc(decode_mem_size);
	ovs->ogg_alloc.alloc_buffer_length_in_bytes=decode_mem_size;
	ovs->frames_mixed=0;
	ovs->active=false;
	ovs->loops=0;
	int error ;
	ovs->ogg_stream = stb_vorbis_open_memory( (const unsigned char*)data, data_len, &error, &ovs->ogg_alloc );
	if (!ovs->ogg_stream) {

		AudioServer::get_singleton()->audio_data_free(ovs->ogg_alloc.alloc_buffer);
		ovs->ogg_alloc.alloc_buffer=NULL;
		ERR_FAIL_COND_V(!ovs->ogg_stream,Ref<AudioStreamPlaybackOGGVorbis>());
	}

	return ovs;
}

String AudioStreamOGGVorbis::get_stream_name() const {

	return "";//return stream_name;
}

Error AudioStreamOGGVorbis::setup(const uint8_t *p_data,uint32_t p_data_len) {


#define MAX_TEST_MEM (1<<20)

	uint32_t alloc_try=1024;
	PoolVector<char> alloc_mem;
	PoolVector<char>::Write w;
	stb_vorbis * ogg_stream=NULL;
	stb_vorbis_alloc ogg_alloc;

	while(alloc_try<MAX_TEST_MEM) {

		alloc_mem.resize(alloc_try);
		w = alloc_mem.write();

		ogg_alloc.alloc_buffer=w.ptr();
		ogg_alloc.alloc_buffer_length_in_bytes=alloc_try;

		int error;
		ogg_stream = stb_vorbis_open_memory( (const unsigned char*)p_data, p_data_len, &error, &ogg_alloc );

		if (!ogg_stream && error==VORBIS_outofmem) {
			w = PoolVector<char>::Write();
			alloc_try*=2;
		} else {
			break;
		}
	}
	ERR_FAIL_COND_V(alloc_try==MAX_TEST_MEM,ERR_OUT_OF_MEMORY);
	ERR_FAIL_COND_V(ogg_stream==NULL,ERR_FILE_CORRUPT);

	stb_vorbis_info info = stb_vorbis_get_info(ogg_stream);

	channels = info.channels;
	sample_rate = info.sample_rate;
	decode_mem_size = alloc_try;
	//does this work? (it's less mem..)
	//decode_mem_size = ogg_alloc.alloc_buffer_length_in_bytes + info.setup_memory_required + info.temp_memory_required + info.max_frame_size;

	//print_line("succeded "+itos(ogg_alloc.alloc_buffer_length_in_bytes)+" setup "+itos(info.setup_memory_required)+" setup temp "+itos(info.setup_temp_memory_required)+" temp "+itos(info.temp_memory_required)+" maxframe"+itos(info.max_frame_size));

	length=stb_vorbis_stream_length_in_seconds(ogg_stream);
	stb_vorbis_close(ogg_stream);

	data = AudioServer::get_singleton()->audio_data_alloc(p_data_len,p_data);
	data_len=p_data_len;

	printf("create at %p, data %p\n",this,data);
	return OK;
}

AudioStreamOGGVorbis::AudioStreamOGGVorbis() {


	data=NULL;
	length=0;
	sample_rate=1;
	channels=1;
	decode_mem_size=0;
}




RES ResourceFormatLoaderAudioStreamOGGVorbis::load(const String &p_path, const String& p_original_path, Error *r_error) {
	if (r_error)
		*r_error=OK;

	FileAccess *f = FileAccess::open(p_path,FileAccess::READ);
	if (!f) {
		*r_error=ERR_CANT_OPEN;
		ERR_FAIL_COND_V(!f,RES());
	}

	size_t len = f->get_len();

	PoolVector<uint8_t> data;
	data.resize(len);
	PoolVector<uint8_t>::Write w = data.write();

	f->get_buffer(w.ptr(),len);

	memdelete(f);

	Ref<AudioStreamOGGVorbis> ogg_stream;
	ogg_stream.instance();

	Error err = ogg_stream->setup(w.ptr(),len);

	if (err!=OK) {
		*r_error=err;
		ogg_stream.unref();
		ERR_FAIL_V(RES());
	}

	return ogg_stream;
}

void ResourceFormatLoaderAudioStreamOGGVorbis::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ogg");
}
String ResourceFormatLoaderAudioStreamOGGVorbis::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower()=="ogg")
		return "AudioStreamOGGVorbis";
	return "";
}

bool ResourceFormatLoaderAudioStreamOGGVorbis::handles_type(const String& p_type) const {
	return (p_type=="AudioStream" || p_type=="AudioStreamOGG" || p_type=="AudioStreamOGGVorbis");
}

