/*************************************************************************/
/*  create_dialog.h                                                      */
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
#ifndef CREATE_DIALOG_H
#define CREATE_DIALOG_H

#include "scene/gui/dialogs.h"
#include "scene/gui/button.h"
#include "scene/gui/tree.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/label.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#if 1


class CreateDialog : public ConfirmationDialog {

	OBJ_TYPE(CreateDialog,ConfirmationDialog )

	LineEdit *search_box;
	Tree *search_options;
	String base_type;

	void _update_search();

	void _sbox_input(const InputEvent& p_ie);

	void _confirmed();
	void _text_changed(const String& p_newtext);

	void add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem *p_root,TreeItem **to_select);


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	Object *instance_selected();

	void set_base_type(const String& p_base);
	String get_base_type() const;

	void popup(bool p_dontclear);


	CreateDialog();
};


#else

//old create dialog, disabled

class CreateDialog : public ConfirmationDialog {
	
	OBJ_TYPE( CreateDialog, ConfirmationDialog );
	
	Tree *tree;
	Button *create;
	Button *cancel;
	LineEdit *filter;

	
	void update_tree();
	void _create();
	void _cancel();
	void add_type(const String& p_type,HashMap<String,TreeItem*>& p_types,TreeItem 
	*p_root);
		
	String base;
	void _text_changed(String p_text);
	virtual void _post_popup() { tree->grab_focus();}

protected:
	static void _bind_methods();	
	void _notification(int p_what);
public:
	


	Object *instance_selected();
	
	void set_base_type(const String& p_base);
	String get_base_type() const;
	CreateDialog();	
	~CreateDialog();

};

#endif

#endif
