#include "audio_stream_mpc.h"


Error AudioStreamMPC::_open_file() {

	if (f) {
		memdelete(f);
		f=NULL;
	}
	Error err;
	//printf("mpc open file %ls\n", file.c_str());
	f=FileAccess::open(file,FileAccess::READ,&err);

	if (err) {
		f=NULL;
		ERR_FAIL_V(err);
		return err;
	}

	//printf("file size is %i\n", f->get_len());
	//f->seek_end(0);
	streamlen=f->get_len();
	//f->seek(0);
	if (streamlen<=0) {
		memdelete(f);
		f=NULL;
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	data_ofs=0;
	if (preload) {

		data.resize(streamlen);
		DVector<uint8_t>::Write w = data.write();
		f->get_buffer(&w[0],streamlen);
		memdelete(f);
		f=NULL;

	}

	return OK;
}

void AudioStreamMPC::_close_file() {

	if (f) {
		memdelete(f);
		f=NULL;
	}
	data.resize(0);
	streamlen=0;
	data_ofs=0;
}

int AudioStreamMPC::_read_file(void *p_dst,int p_bytes) {

	if (f)
		return f->get_buffer((uint8_t*)p_dst,p_bytes);

	DVector<uint8_t>::Read r = data.read();
	if (p_bytes+data_ofs > streamlen) {
		p_bytes=streamlen-data_ofs;
	}

	copymem(p_dst,&r[data_ofs],p_bytes);
	//print_line("read file: "+itos(p_bytes));
	data_ofs+=p_bytes;
	return p_bytes;
}

bool AudioStreamMPC::_seek_file(int p_pos){

	if (p_pos<0 || p_pos>streamlen)
		return false;

	if (f) {
		f->seek(p_pos);
		return true;
	}

	//print_line("read file to: "+itos(p_pos));
	data_ofs=p_pos;
	return true;

}
int AudioStreamMPC::_tell_file()  const{

	if (f)
		return f->get_pos();

	//print_line("tell file, get: "+itos(data_ofs));
	return data_ofs;

}

int AudioStreamMPC::_sizeof_file() const{

	//print_line("sizeof file, get: "+itos(streamlen));
	return streamlen;
}

bool AudioStreamMPC::_canseek_file() const{

	//print_line("canseek file, get true");
	return true;
}

/////////////////////

mpc_int32_t AudioStreamMPC::_mpc_read(mpc_reader *p_reader,void *p_dst, mpc_int32_t p_bytes) {

	AudioStreamMPC *smpc=(AudioStreamMPC *)p_reader->data;
	return smpc->_read_file(p_dst,p_bytes);
}

mpc_bool_t AudioStreamMPC::_mpc_seek(mpc_reader *p_reader,mpc_int32_t p_offset) {

	AudioStreamMPC *smpc=(AudioStreamMPC *)p_reader->data;
	return smpc->_seek_file(p_offset);

}
mpc_int32_t AudioStreamMPC::_mpc_tell(mpc_reader *p_reader) {

	AudioStreamMPC *smpc=(AudioStreamMPC *)p_reader->data;
	return smpc->_tell_file();

}
mpc_int32_t AudioStreamMPC::_mpc_get_size(mpc_reader *p_reader) {

	AudioStreamMPC *smpc=(AudioStreamMPC *)p_reader->data;
	return smpc->_sizeof_file();


}
mpc_bool_t AudioStreamMPC::_mpc_canseek(mpc_reader *p_reader) {

	AudioStreamMPC *smpc=(AudioStreamMPC *)p_reader->data;
	return smpc->_canseek_file();
}



bool AudioStreamMPC::_can_mix() const {

	return /*active &&*/ !paused;
}


void AudioStreamMPC::update() {

	if (!active || paused)
		return;

	int todo=get_todo();

	while(todo>MPC_DECODER_BUFFER_LENGTH/si.channels) {

		mpc_frame_info frame;

		frame.buffer=sample_buffer;

		mpc_status err = mpc_demux_decode(demux, &frame);
		if (frame.bits!=-1) {

			int16_t *dst_buff = get_write_buffer();

#ifdef MPC_FIXED_POINT

			for( int i = 0; i < frame.samples * si.channels; i++) {
				int tmp = sample_buffer[i] >> MPC_FIXED_POINT_FRACTPART;
				if (tmp > ((1 << 15) - 1)) tmp = ((1 << 15) - 1);
				if (tmp < -(1 << 15)) tmp = -(1 << 15);
				dst_buff[i] = tmp;
			}
#else
			for( int i = 0; i < frame.samples * si.channels; i++) {

				int tmp = Math::fast_ftoi(sample_buffer[i]*32767.0);
				if (tmp > ((1 << 15) - 1)) tmp = ((1 << 15) - 1);
				if (tmp < -(1 << 15)) tmp = -(1 << 15);
				dst_buff[i] = tmp;

			}

#endif

			int frames = frame.samples;
			write(frames);
			todo-=frames;
		} else {

			if (err != MPC_STATUS_OK) {

				stop();
				ERR_EXPLAIN("Error decoding MPC");
				ERR_FAIL();
			} else {

				//finished
				if (!loop) {
					stop();
					return;
				} else {


					loops++;
					mpc_demux_exit(demux);
					_seek_file(0);
					demux = mpc_demux_init(&reader);
					//do loop somehow

				}
			}
		}
	}
}

Error AudioStreamMPC::_reload() {

	ERR_FAIL_COND_V(demux!=NULL, ERR_FILE_ALREADY_IN_USE);

	Error err = _open_file();
	ERR_FAIL_COND_V(err!=OK,ERR_CANT_OPEN);

	demux = mpc_demux_init(&reader);
	ERR_FAIL_COND_V(!demux,ERR_CANT_CREATE);

	mpc_demux_get_info(demux,  &si);
	_setup(si.channels,si.sample_freq,MPC_DECODER_BUFFER_LENGTH*2/si.channels);

	return OK;
}

void AudioStreamMPC::set_file(const String& p_file) {

	file=p_file;

}


String AudioStreamMPC::get_file() const {

	return file;
}


void AudioStreamMPC::play() {


	_THREAD_SAFE_METHOD_

	if (active)
		stop();
	active=false;

	Error err = _open_file();
	ERR_FAIL_COND(err!=OK);
	if (_reload()!=OK)
		return;
	active=true;
	loops=0;

}

void AudioStreamMPC::stop()  {

	_THREAD_SAFE_METHOD_
	if (!active)
		return;
	if (demux) {
		mpc_demux_exit(demux);
		demux=NULL;
	}
	_close_file();
	active=false;

}
bool AudioStreamMPC::is_playing() const  {

	return active || (get_total() - get_todo() -1 > 0);
}

void AudioStreamMPC::set_paused(bool p_paused)  {

	paused=p_paused;
}
bool AudioStreamMPC::is_paused(bool p_paused) const  {

	return paused;
}

void AudioStreamMPC::set_loop(bool p_enable)  {

	loop=p_enable;
}
bool AudioStreamMPC::has_loop() const  {

	return loop;
}

float AudioStreamMPC::get_length() const {

	return 0;
}

String AudioStreamMPC::get_stream_name() const {

	return "";
}

int AudioStreamMPC::get_loop_count() const {

	return 0;
}

float AudioStreamMPC::get_pos() const {

	return 0;
}
void AudioStreamMPC::seek_pos(float p_time) {


}

AudioStream::UpdateMode AudioStreamMPC::get_update_mode() const {

	return UPDATE_THREAD;
}

void AudioStreamMPC::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_file","name"),&AudioStreamMPC::set_file);
	ObjectTypeDB::bind_method(_MD("get_file"),&AudioStreamMPC::get_file);

	ADD_PROPERTYNZ( PropertyInfo(Variant::STRING,"file",PROPERTY_HINT_FILE,"mpc"), _SCS("set_file"), _SCS("get_file"));

}

AudioStreamMPC::AudioStreamMPC() {

	preload=true;
	f=NULL;
	streamlen=0;
	data_ofs=0;
	active=false;
	paused=false;
	loop=false;
	demux=NULL;
	reader.data=this;
	reader.read=_mpc_read;
	reader.seek=_mpc_seek;
	reader.tell=_mpc_tell;
	reader.get_size=_mpc_get_size;
	reader.canseek=_mpc_canseek;
	loops=0;

}

AudioStreamMPC::~AudioStreamMPC() {

	stop();

	if (f)
		memdelete(f);
}



RES ResourceFormatLoaderAudioStreamMPC::load(const String &p_path,const String& p_original_path) {

	AudioStreamMPC *mpc_stream = memnew(AudioStreamMPC);
	mpc_stream->set_file(p_path);
	return Ref<AudioStreamMPC>(mpc_stream);
}

void ResourceFormatLoaderAudioStreamMPC::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("mpc");
}
bool ResourceFormatLoaderAudioStreamMPC::handles_type(const String& p_type) const {

	return (p_type=="AudioStream") || (p_type=="AudioStreamMPC");
}

String ResourceFormatLoaderAudioStreamMPC::get_resource_type(const String &p_path) const {

	if (p_path.extension().to_lower()=="mpc")
		return "AudioStreamMPC";
	return "";
}

