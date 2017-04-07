/*************************************************************************/
/*  sample_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "sample_editor_plugin.h"

#if 0
#include "editor/editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"




void SampleEditor::_gui_input(InputEvent p_event) {


}

void SampleEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_FIXED_PROCESS) {

	}

	if (p_what==NOTIFICATION_ENTER_TREE) {
		play->set_icon( get_icon("Play","EditorIcons") );
		stop->set_icon( get_icon("Stop","EditorIcons") );
	}

	if (p_what==NOTIFICATION_READY) {

		//get_scene()->connect("node_removed",this,"_node_removed");

	}

	if (p_what==NOTIFICATION_DRAW) {

	}
}

void SampleEditor::_play_pressed() {

	player->play("default",true);
	stop->set_pressed(false);
	play->set_pressed(true);
}
void SampleEditor::_stop_pressed() {

	player->stop_all();
	play->set_pressed(false);
}

void SampleEditor::generate_preview_texture(const Ref<Sample>& p_sample,Ref<ImageTexture> &p_texture) {


	PoolVector<uint8_t> data = p_sample->get_data();

	PoolVector<uint8_t> img;
	int w = p_texture->get_width();
	int h = p_texture->get_height();
	img.resize(w*h*3);
	PoolVector<uint8_t>::Write imgdata = img.write();
	uint8_t * imgw = imgdata.ptr();
	PoolVector<uint8_t>::Read sampledata = data.read();
	const uint8_t *sdata=sampledata.ptr();

	bool stereo = p_sample->is_stereo();
	bool _16=p_sample->get_format()==Sample::FORMAT_PCM16;
	int len = p_sample->get_length();

	if (len<1)
		return;

	if (p_sample->get_format()==Sample::FORMAT_IMA_ADPCM) {


		struct IMA_ADPCM_State {

			int16_t step_index;
			int32_t predictor;
			/* values at loop point */
			int16_t loop_step_index;
			int32_t loop_predictor;
			int32_t last_nibble;
			int32_t loop_pos;
			int32_t window_ofs;
			const uint8_t *ptr;
		} ima_adpcm;

		ima_adpcm.step_index=0;
		ima_adpcm.predictor=0;
		ima_adpcm.loop_step_index=0;
		ima_adpcm.loop_predictor=0;
		ima_adpcm.last_nibble=-1;
		ima_adpcm.loop_pos=0x7FFFFFFF;
		ima_adpcm.window_ofs=0;
		ima_adpcm.ptr=NULL;


		for(int i=0;i<w;i++) {

			float max[2]={-1e10,-1e10};
			float min[2]={1e10,1e10};
			int from = i*len/w;
			int to = (i+1)*len/w;
			if (to>=len)
				to=len-1;

			for(int j=from;j<to;j++) {

				while(j>ima_adpcm.last_nibble) {

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

					int16_t nibble,diff,step;

					ima_adpcm.last_nibble++;
					const uint8_t *src_ptr=sdata;

					int ofs = ima_adpcm.last_nibble>>1;

					if (stereo)
						ofs*=2;

					nibble = (ima_adpcm.last_nibble&1)?
							(src_ptr[ofs]>>4):(src_ptr[ofs]&0xF);

					step=_ima_adpcm_step_table[ima_adpcm.step_index];

					ima_adpcm.step_index += _ima_adpcm_index_table[nibble];
					if (ima_adpcm.step_index<0)
						ima_adpcm.step_index=0;
					if (ima_adpcm.step_index>88)
						ima_adpcm.step_index=88;

					diff = step >> 3 ;
					if (nibble & 1)
						diff += step >> 2 ;
					if (nibble & 2)
						diff += step >> 1 ;
					if (nibble & 4)
						diff += step ;
					if (nibble & 8)
						diff = -diff ;

					ima_adpcm.predictor+=diff;
					if (ima_adpcm.predictor<-0x8000)
						ima_adpcm.predictor=-0x8000;
					else if (ima_adpcm.predictor>0x7FFF)
						ima_adpcm.predictor=0x7FFF;


					/* store loop if there */
					if (ima_adpcm.last_nibble==ima_adpcm.loop_pos) {

						ima_adpcm.loop_step_index = ima_adpcm.step_index;
						ima_adpcm.loop_predictor = ima_adpcm.predictor;
					}

				}

				float v=ima_adpcm.predictor/32767.0;
				if (v>max[0])
					max[0]=v;
				if (v<min[0])
					min[0]=v;
			}

			for(int j=0;j<h;j++) {
				float v = (j/(float)h) * 2.0 - 1.0;
				uint8_t* imgofs = &imgw[(uint64_t(j)*w+i)*3];
				if (v>min[0] && v<max[0]) {
					imgofs[0]=255;
					imgofs[1]=150;
					imgofs[2]=80;
				} else {
					imgofs[0]=0;
					imgofs[1]=0;
					imgofs[2]=0;
				}
			}
		}
	} else {
		for(int i=0;i<w;i++) {
			// i trust gcc will optimize this loop
			float max[2]={-1e10,-1e10};
			float min[2]={1e10,1e10};
			int c=stereo?2:1;
			int from = uint64_t(i)*len/w;
			int to = (uint64_t(i)+1)*len/w;
			if (to>=len)
				to=len-1;

			if (_16) {
				const int16_t*src =(const int16_t*)sdata;

				for(int j=0;j<c;j++) {

					for(int k=from;k<=to;k++) {

						float v = src[uint64_t(k)*c+j]/32768.0;
						if (v>max[j])
							max[j]=v;
						if (v<min[j])
							min[j]=v;
					}

				}
			} else {

				const int8_t*src =(const int8_t*)sdata;

				for(int j=0;j<c;j++) {

					for(int k=from;k<=to;k++) {

						float v = src[uint64_t(k)*c+j]/128.0;
						if (v>max[j])
							max[j]=v;
						if (v<min[j])
							min[j]=v;
					}

				}
			}

			if (!stereo) {
				for(int j=0;j<h;j++) {
					float v = (j/(float)h) * 2.0 - 1.0;
					uint8_t* imgofs = &imgw[(uint64_t(j)*w+i)*3];
					if (v>min[0] && v<max[0]) {
						imgofs[0]=255;
						imgofs[1]=150;
						imgofs[2]=80;
					} else {
						imgofs[0]=0;
						imgofs[1]=0;
						imgofs[2]=0;
					}
				}
			} else {

				for(int j=0;j<h;j++) {

					int half;
					float v;
					if (j<(h/2)) {
						half=0;
						v = (j/(float)(h/2)) * 2.0 - 1.0;
					} else {
						half=1;
						v = ((j-(h/2))/(float)(h/2)) * 2.0 - 1.0;
					}

					uint8_t* imgofs = &imgw[(uint64_t(j)*w+i)*3];
					if (v>min[half] && v<max[half]) {
						imgofs[0]=255;
						imgofs[1]=150;
						imgofs[2]=80;
					} else {
						imgofs[0]=0;
						imgofs[1]=0;
						imgofs[2]=0;
					}
				}

			}

		}
	}

	imgdata = PoolVector<uint8_t>::Write();


	p_texture->set_data(Image(w,h,0,Image::FORMAT_RGB8,img));

}

void SampleEditor::_update_sample() {

	player->stop_all();

	generate_preview_texture(sample,peakdisplay);
	info_label->set_text(TTR("Length:")+" "+String::num(sample->get_length()/(float)sample->get_mix_rate(),2)+"s");

	if (library->has_sample("default"))
		library->remove_sample("default");

	library->add_sample("default",sample);
}



void SampleEditor::edit(Ref<Sample> p_sample) {

	sample=p_sample;

	if (!sample.is_null())
		_update_sample();
	else {

		hide();
		set_fixed_process(false);
	}

}



void SampleEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"),&SampleEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_play_pressed"),&SampleEditor::_play_pressed);
	ClassDB::bind_method(D_METHOD("_stop_pressed"),&SampleEditor::_stop_pressed);

}

SampleEditor::SampleEditor() {

	player = memnew(SamplePlayer);
	add_child(player);
	add_style_override("panel", get_stylebox("panel","Panel"));
	library = Ref<SampleLibrary>(memnew(SampleLibrary));
	player->set_sample_library(library);
	sample_texframe = memnew( TextureRect );
	add_child(sample_texframe);
	sample_texframe->set_anchor_and_margin(MARGIN_LEFT,ANCHOR_BEGIN,5);
	sample_texframe->set_anchor_and_margin(MARGIN_RIGHT,ANCHOR_END,5);
	sample_texframe->set_anchor_and_margin(MARGIN_TOP,ANCHOR_BEGIN,30);
	sample_texframe->set_anchor_and_margin(MARGIN_BOTTOM,ANCHOR_END,5);

	info_label = memnew( Label );
	sample_texframe->add_child(info_label);
	info_label->set_area_as_parent_rect();
	info_label->set_anchor_and_margin(MARGIN_TOP,ANCHOR_END,15);
	info_label->set_margin(MARGIN_BOTTOM,4);
	info_label->set_margin(MARGIN_RIGHT,4);
	info_label->set_align(Label::ALIGN_RIGHT);


	play = memnew( Button );

	play->set_pos(Point2( 5, 5 ));
	play->set_size( Size2(1,1 ) );
	play->set_toggle_mode(true);
	add_child(play);

	stop = memnew( Button );

	stop->set_pos(Point2( 35, 5 ));
	stop->set_size( Size2(1,1 ) );
	stop->set_toggle_mode(true);
	add_child(stop);

	peakdisplay=Ref<ImageTexture>( memnew( ImageTexture) );
	peakdisplay->create( EDITOR_DEF("editors/sample_editor/preview_width",512),EDITOR_DEF("editors/sample_editor/preview_height",128),Image::FORMAT_RGB8);
	sample_texframe->set_expand(true);
	sample_texframe->set_texture(peakdisplay);

	play->connect("pressed", this,"_play_pressed");
	stop->connect("pressed", this,"_stop_pressed");

	set_custom_minimum_size(Size2(1,150)*EDSCALE);

}


void SampleEditorPlugin::edit(Object *p_object) {

	Sample * s = p_object->cast_to<Sample>();
	if (!s)
		return;

	sample_editor->edit(Ref<Sample>(s));
}

bool SampleEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Sample");
}

void SampleEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		sample_editor->show();
		//sample_editor->set_process(true);
	} else {

		sample_editor->hide();
		//sample_editor->set_process(false);
	}

}

SampleEditorPlugin::SampleEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	sample_editor = memnew( SampleEditor );
	add_control_to_container(CONTAINER_PROPERTY_EDITOR_BOTTOM,sample_editor);
	sample_editor->hide();



}


SampleEditorPlugin::~SampleEditorPlugin()
{
}

#endif
