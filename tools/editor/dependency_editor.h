#ifndef DEPENDENCY_EDITOR_H
#define DEPENDENCY_EDITOR_H

#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/gui/tab_container.h"
#include "editor_file_dialog.h"

class EditorFileSystemDirectory;

class DependencyEditor : public AcceptDialog {
	OBJ_TYPE(DependencyEditor,AcceptDialog);


	Tree *tree;
	Button *fixdeps;

	EditorFileDialog *search;

	String replacing;
	String editing;
	List<String> missing;


	void _fix_and_find(EditorFileSystemDirectory *efsd, Map<String,Map<String,String> >& candidates);

	void _searched(const String& p_path);
	void _load_pressed(Object* p_item,int p_cell,int p_button);
	void _fix_all();
	void _update_list();

	void _update_file();

protected:

	static void _bind_methods();
	void _notification(int p_what);
public:


	void edit(const String& p_path);
	DependencyEditor();
};

class DependencyEditorOwners : public AcceptDialog {
	OBJ_TYPE(DependencyEditorOwners,AcceptDialog);

	ItemList *owners;
	String editing;
	void _fill_owners(EditorFileSystemDirectory *efsd);

public:

	void show(const String& p_path);
	DependencyEditorOwners();
};

class DependencyRemoveDialog : public ConfirmationDialog {
	OBJ_TYPE(DependencyRemoveDialog,ConfirmationDialog);


	Label *text;
	Tree *owners;
	bool exist;
	Map<String,TreeItem*> files;
	void _fill_owners(EditorFileSystemDirectory *efsd);

	void ok_pressed();

public:

	void show(const Vector<String> &to_erase);
	DependencyRemoveDialog();
};


class DependencyErrorDialog : public ConfirmationDialog {
	OBJ_TYPE(DependencyErrorDialog,ConfirmationDialog);


	String for_file;
	Button *fdep;
	Label *text;
	Tree *files;
	void ok_pressed();
	void custom_action(const String&);

public:

	void show(const String& p_for,const Vector<String> &report);
	DependencyErrorDialog();
};



class OrphanResourcesDialog : public ConfirmationDialog {
	OBJ_TYPE(OrphanResourcesDialog,ConfirmationDialog);

	DependencyEditor *dep_edit;
	Tree *files;
	ConfirmationDialog *delete_confirm;
	void ok_pressed();

	bool _fill_owners(EditorFileSystemDirectory *efsd, HashMap<String,int>& refs, TreeItem *p_parent);

	List<String> paths;
	void _find_to_delete(TreeItem* p_item,List<String>& paths);
	void _delete_confirm();
	void _button_pressed(Object *p_item,int p_column, int p_id);

	void refresh();
	static void _bind_methods();
public:

	void show();
	OrphanResourcesDialog();
};

#endif // DEPENDENCY_EDITOR_H
