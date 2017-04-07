/*************************************************************************/
/*  test_gui.cpp                                                         */
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
#ifndef _3D_DISABLED

#include "test_gui.h"

#include "io/image_loader.h"
#include "os/os.h"
#include "print_string.h"
#include "scene/2d/sprite.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"
#include "scene/main/scene_main_loop.h"

#include "scene/3d/camera.h"
#include "scene/3d/test_cube.h"
#include "scene/main/viewport.h"

namespace TestGUI {

class TestMainLoop : public SceneTree {

	Control *control;

public:
	virtual void request_quit() {

		quit();
	}
	virtual void init() {

		SceneTree::init();

#if 0


		Viewport *vp = memnew( Viewport );
		vp->set_world( Ref<World>( memnew( World )));
		get_root()->add_child(vp);

		vp->set_rect(Rect2(0,0,256,256));
		vp->set_as_render_target(true);
		vp->set_render_target_update_mode(Viewport::RENDER_TARGET_UPDATE_ALWAYS);


		Camera *camera = memnew( Camera );
		vp->add_child(camera);
		camera->make_current();

		TestCube *testcube = memnew( TestCube );
		vp->add_child(testcube);
		testcube->set_transform(Transform( Basis().rotated(Vector3(0,1,0),Math_PI*0.25), Vector3(0,0,-8)));

		Sprite *sp = memnew( Sprite );
		sp->set_texture( vp->get_render_target_texture() );
		//sp->set_texture( ResourceLoader::load("res://ball.png") );
		sp->set_pos(Point2(300,300));
		get_root()->add_child(sp);


		return;
#endif

		Panel *frame = memnew(Panel);
		frame->set_anchor(MARGIN_RIGHT, Control::ANCHOR_END);
		frame->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_END);
		frame->set_end(Point2(0, 0));

		Ref<Theme> t = memnew(Theme);
		frame->set_theme(t);

		get_root()->add_child(frame);

		Label *label = memnew(Label);

		label->set_pos(Point2(80, 90));
		label->set_size(Point2(170, 80));
		label->set_align(Label::ALIGN_FILL);
		//label->set_text("There");
		label->set_text("There was once upon a time a beautiful unicorn that loved to play with little girls...");

		frame->add_child(label);

		Button *button = memnew(Button);

		button->set_pos(Point2(20, 20));
		button->set_size(Point2(1, 1));
		button->set_text("This is a biggie button");

		frame->add_child(button);

#if 0
		Sprite *tf = memnew( Sprite );
		frame->add_child(tf);
		Image img;
		ImageLoader::load_image("LarvoClub.png",&img);

		img.resize(512,512);
		img.generate_mipmaps();
		img.compress(Image::COMPRESS_PVRTC4);
		Ref<ImageTexture> tt = memnew( ImageTexture );
		tt->create_from_image(img);
		tf->set_texture(tt);
		tf->set_pos(Point2(50,50));
		//tf->set_scale(Point2(0.3,0.3));


		return;
#endif

		Tree *tree = memnew(Tree);
		tree->set_columns(2);

		tree->set_pos(Point2(230, 210));
		tree->set_size(Point2(150, 250));

		TreeItem *item = tree->create_item();
		item->set_editable(0, true);
		item->set_text(0, "root");
		item = tree->create_item(tree->get_root());
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_editable(0, true);
		item->set_text(0, "check");
		item->set_cell_mode(1, TreeItem::CELL_MODE_CHECK);
		item->set_editable(1, true);
		item->set_text(1, "check2");
		item = tree->create_item(tree->get_root());
		item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
		item->set_editable(0, true);
		item->set_range_config(0, 0, 20, 0.1);
		item->set_range(0, 2);
		item->add_button(0, Theme::get_default()->get_icon("folder", "FileDialog"));
		item->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
		item->set_editable(1, true);
		item->set_range_config(1, 0, 20, 0.1);
		item->set_range(1, 3);

		item = tree->create_item(tree->get_root());
		item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
		item->set_editable(0, true);
		item->set_text(0, "Have,Many,Several,Options!");
		item->set_range(0, 2);

		item = tree->create_item(item);
		item->set_editable(0, true);
		item->set_text(0, "Gershwin!");

		frame->add_child(tree);

		//control = memnew( Control );
		//root->add_child( control );

		LineEdit *line_edit = memnew(LineEdit);

		line_edit->set_pos(Point2(30, 190));
		line_edit->set_size(Point2(180, 1));

		frame->add_child(line_edit);

		HScrollBar *hscroll = memnew(HScrollBar);

		hscroll->set_pos(Point2(30, 290));
		hscroll->set_size(Point2(180, 1));
		hscroll->set_max(10);
		hscroll->set_page(4);

		frame->add_child(hscroll);

		SpinBox *spin = memnew(SpinBox);

		spin->set_pos(Point2(30, 260));
		spin->set_size(Point2(120, 1));

		frame->add_child(spin);
		hscroll->share(spin);

		ProgressBar *progress = memnew(ProgressBar);

		progress->set_pos(Point2(30, 330));
		progress->set_size(Point2(120, 1));

		frame->add_child(progress);
		hscroll->share(progress);

		MenuButton *menu_button = memnew(MenuButton);

		menu_button->set_text("I'm a menu!");
		menu_button->set_pos(Point2(30, 380));
		menu_button->set_size(Point2(1, 1));

		frame->add_child(menu_button);

		PopupMenu *popup = menu_button->get_popup();

		popup->add_item("Hello, testing");
		popup->add_item("My Dearest");
		popup->add_separator();
		popup->add_item("Popup");
		popup->add_check_item("Check Popup");
		popup->set_item_checked(4, true);

		OptionButton *options = memnew(OptionButton);

		options->add_item("Hello, testing");
		options->add_item("My Dearest");

		options->set_pos(Point2(230, 180));
		options->set_size(Point2(1, 1));

		frame->add_child(options);

		/*
		Tree * tree = memnew( Tree );
		tree->set_columns(2);

		tree->set_pos( Point2( 230,210 ) );
		tree->set_size( Point2( 150,250 ) );


		TreeItem *item = tree->create_item();
		item->set_editable(0,true);
		item->set_text(0,"root");
		item = tree->create_item( tree->get_root() );
		item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		item->set_editable(0,true);
		item->set_text(0,"check");
		item = tree->create_item( tree->get_root() );
		item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
		item->set_editable(0,true);
		item->set_range_config(0,0,20,0.1);
		item->set_range(0,2);
		item->add_button(0,Theme::get_default()->get_icon("folder","FileDialog"));
		item = tree->create_item( tree->get_root() );
		item->set_cell_mode(0, TreeItem::CELL_MODE_RANGE);
		item->set_editable(0,true);
		item->set_text(0,"Have,Many,Several,Options!");
		item->set_range(0,2);

		frame->add_child(tree);
*/

		RichTextLabel *richtext = memnew(RichTextLabel);

		richtext->set_pos(Point2(600, 210));
		richtext->set_size(Point2(180, 250));
		richtext->set_anchor_and_margin(MARGIN_RIGHT, Control::ANCHOR_END, 20);

		frame->add_child(richtext);

		richtext->add_text("Hello, My Friends!\n\nWelcome to the amazing world of ");

		richtext->add_newline();
		richtext->add_newline();

		richtext->push_color(Color(1, 0.5, 0.5));
		richtext->add_text("leprechauns");
		richtext->pop();

		richtext->add_text(" and ");
		richtext->push_color(Color(0, 1.0, 0.5));
		richtext->add_text("faeries.\n");
		richtext->pop();
		richtext->add_text("In this new episode, we will attempt to ");
		richtext->push_font(richtext->get_font("mono_font", "Fonts"));
		richtext->push_color(Color(0.7, 0.5, 1.0));
		richtext->add_text("deliver something nice");
		richtext->pop();
		richtext->pop();
		richtext->add_text(" to all the viewers! Unfortunately, I need to ");
		richtext->push_underline();
		richtext->add_text("keep writing a lot of text");
		richtext->pop();
		richtext->add_text(" so the label control overflows and the scrollbar appears.\n");
		//richtext->push_indent(1);
		//richtext->add_text("By the way, testing indent levels! Yohohoho! Everything should appear to the right sightly here!\n");
		//richtext->pop();
		richtext->push_meta("http://www.scrollingcapabilities.xz");
		richtext->add_text("This allows to test for the scrolling capabilities ");
		richtext->pop();
		richtext->add_text("of the rich text label for huge text (not like this text will really be huge but, you know).\nAs long as it is so long that it will work nicely for a test/demo, then it's welcomed in my book...\nChanging subject, the day is cloudy today and I'm wondering if I'll get che chance to travel somewhere nice. Sometimes, watching the clouds from satellite images may give a nice insight about how pressure zones in our planet work, althogh it also makes it pretty obvious to see why most weather forecasts get it wrong so often.\nClouds are so difficult to predict!\nBut it's pretty cool how our civilization has adapted to having water falling from the sky each time it rains...");
		//richtext->add_text("Hello!\nGorgeous..");

		//richtext->push_meta("http://www.scrollingcapabilities.xz");
		///richtext->add_text("Hello!\n");
		//richtext->pop();

		richtext->set_anchor(MARGIN_RIGHT, Control::ANCHOR_END);

		TabContainer *tabc = memnew(TabContainer);

		Control *ctl = memnew(Control);
		ctl->set_name("tab 1");
		tabc->add_child(ctl);

		ctl = memnew(Control);
		ctl->set_name("tab 2");
		tabc->add_child(ctl);
		label = memnew(Label);
		label->set_text("Some Label");
		label->set_pos(Point2(20, 20));
		ctl->add_child(label);

		ctl = memnew(Control);
		ctl->set_name("tab 3");
		button = memnew(Button);
		button->set_text("Some Button");
		button->set_pos(Point2(30, 50));
		ctl->add_child(button);

		tabc->add_child(ctl);

		frame->add_child(tabc);

		tabc->set_pos(Point2(400, 210));
		tabc->set_size(Point2(180, 250));

		/*Ref<ImageTexture> text = memnew( ImageTexture );
		text->load("test_data/concave.png");

		Sprite* sprite = memnew(Sprite);
		sprite->set_texture(text);
		sprite->set_pos(Point2(300, 300));
		frame->add_child(sprite);
		sprite->show();

		Sprite* sprite2 = memnew(Sprite);
		sprite->set_texture(text);
		sprite->add_child(sprite2);
		sprite2->set_pos(Point2(50, 50));
		sprite2->show();*/
	}
};

MainLoop *test() {

	return memnew(TestMainLoop);
}
}

#endif
