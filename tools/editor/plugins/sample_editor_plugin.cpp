/*************************************************************************/
/*  sample_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "io/resource_loader.h"
#include "globals.h"
#include "tools/editor/editor_settings.h"




void SampleEditor::_input_event(InputEvent p_event) {


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


	DVector<uint8_t> data = p_sample->get_data();

	DVector<uint8_t> img;
	int w = p_texture->get_width();
	int h = p_texture->get_height();
	img.resize(w*h*3);
	DVector<uint8_t>::Write imgdata = img.write();
	uint8_t * imgw = imgdata.ptr();
	DVector<uint8_t>::Read sampledata = data.read();
	const uint8_t *sdata=sampledata.ptr();

	bool stereo = p_sample->is_stereo();
	bool _16=p_sample->get_format()==Sample::FORMAT_PCM16;
	int len = p_sample->get_length();

	if (len<1)
		return;

	if (p_sample->get_format()==Sample::FORMAT_IMA_ADPCM) {


	} else {
		for(int i=0;i<w;i++) {
			// i trust gcc will optimize this loop
			float max[2]={-1e10,-1e10};
			float min[2]={1e10,1e10};
			int c=stereo?2:1;
			int from = i*len/w;
			int to = (i+1)*len/w;
			if (to>=len)
				to=len-1;

			if (_16) {
				const int16_t*src =(const int16_t*)sdata;

				for(int j=0;j<c;j++) {

					for(int k=from;k<=to;k++) {

						float v = src[k*c+j]/32768.0;
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

						float v = src[k*c+j]/128.0;
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
					uint8_t* imgofs = &imgw[(j*w+i)*3];
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

					int half,ofs;
					float v;
					if (j<(h/2)) {
						half=0;
						ofs=0;
						v = (j/(float)(h/2)) * 2.0 - 1.0;
					} else {
						half=1;
						ofs=h/2;
						v = ((j-(h/2))/(float)(h/2)) * 2.0 - 1.0;
					}

					uint8_t* imgofs = &imgw[(j*w+i)*3];
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

	imgdata = DVector<uint8_t>::Write();


	p_texture->set_data(Image(w,h,0,Image::FORMAT_RGB,img));

}

void SampleEditor::_update_sample() {

	player->stop_all();
	if (sample->get_format()==Sample::FORMAT_IMA_ADPCM)
		return; //bye or unsupported

	generate_preview_texture(sample,peakdisplay);
	info_label->set_text("Length: "+itos(sample->get_length())+" frames ("+String::num(sample->get_length()/(float)sample->get_mix_rate(),2)+" s), "+(sample->get_format()==Sample::FORMAT_PCM16?"16 Bits, ":"8 bits, ")+(sample->is_stereo()?"Stereo.":"Mono."));

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

	ObjectTypeDB::bind_method(_MD("_input_event"),&SampleEditor::_input_event);
	ObjectTypeDB::bind_method(_MD("_play_pressed"),&SampleEditor::_play_pressed);
	ObjectTypeDB::bind_method(_MD("_stop_pressed"),&SampleEditor::_stop_pressed);

}

SampleEditor::SampleEditor() {

	player = memnew(SamplePlayer);
	add_child(player);
	add_style_override("panel", get_stylebox("panel","Panel"));
	library = Ref<SampleLibrary>(memnew(SampleLibrary));
	player->set_sample_library(library);
	sample_texframe = memnew( TextureFrame );
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
	peakdisplay->create( EDITOR_DEF("audio/sample_editor_preview_width",512),EDITOR_DEF("audio/sample_editor_preview_height",128),Image::FORMAT_RGB);
	sample_texframe->set_expand(true);
	sample_texframe->set_texture(peakdisplay);

	play->connect("pressed", this,"_play_pressed");
	stop->connect("pressed", this,"_stop_pressed");

}


void SampleEditorPlugin::edit(Object *p_object) {

	Sample * s = p_object->cast_to<Sample>();
	if (!s)
		return;

	sample_editor->edit(Ref<Sample>(s));
}

bool SampleEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("Sample");
}

void SampleEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		sample_editor->show();
//		sample_editor->set_process(true);
	} else {

		sample_editor->hide();
//		sample_editor->set_process(false);
	}

}

SampleEditorPlugin::SampleEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	sample_editor = memnew( SampleEditor );
	editor->get_viewport()->add_child(sample_editor);
	sample_editor->set_area_as_parent_rect();
	sample_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
	sample_editor->set_margin( MARGIN_TOP, 120 );
	sample_editor->hide();



}


SampleEditorPlugin::~SampleEditorPlugin()
{
}


