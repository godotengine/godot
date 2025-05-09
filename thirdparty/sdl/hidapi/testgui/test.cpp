/*******************************************************
 Demo Program for HIDAPI
 
 Alan Ott
 Signal 11 Software

 2010-07-20

 Copyright 2010, All Rights Reserved
 
 This contents of this file may be used by anyone
 for any reason without any conditions and may be
 used as a starting point for your own applications
 which use HIDAPI.
********************************************************/


#include <fx.h>

#include "hidapi.h"
#include "mac_support.h"
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#ifdef _WIN32
	// Thanks Microsoft, but I know how to use strncpy().
	#pragma warning(disable:4996)
#endif

class MainWindow : public FXMainWindow {
	FXDECLARE(MainWindow)
	
public:
	enum {
		ID_FIRST = FXMainWindow::ID_LAST,
		ID_CONNECT,
		ID_DISCONNECT,
		ID_RESCAN,
		ID_SEND_OUTPUT_REPORT,
		ID_SEND_FEATURE_REPORT,
		ID_GET_FEATURE_REPORT,
		ID_CLEAR,
		ID_TIMER,
		ID_MAC_TIMER,
		ID_LAST,
	};
	
private:
	FXList *device_list;
	FXButton *connect_button;
	FXButton *disconnect_button;
	FXButton *rescan_button;
	FXButton *output_button;
	FXLabel *connected_label;
	FXTextField *output_text;
	FXTextField *output_len;
	FXButton *feature_button;
	FXButton *get_feature_button;
	FXTextField *feature_text;
	FXTextField *feature_len;
	FXTextField *get_feature_text;
	FXText *input_text;
	FXFont *title_font;
	
	struct hid_device_info *devices;
	hid_device *connected_device;
	size_t getDataFromTextField(FXTextField *tf, char *buf, size_t len);
	int getLengthFromTextField(FXTextField *tf);


protected:
	MainWindow() {};
public:
	MainWindow(FXApp *a);
	~MainWindow();
	virtual void create();
	
	long onConnect(FXObject *sender, FXSelector sel, void *ptr);
	long onDisconnect(FXObject *sender, FXSelector sel, void *ptr);
	long onRescan(FXObject *sender, FXSelector sel, void *ptr);
	long onSendOutputReport(FXObject *sender, FXSelector sel, void *ptr);
	long onSendFeatureReport(FXObject *sender, FXSelector sel, void *ptr);
	long onGetFeatureReport(FXObject *sender, FXSelector sel, void *ptr);
	long onClear(FXObject *sender, FXSelector sel, void *ptr);
	long onTimeout(FXObject *sender, FXSelector sel, void *ptr);
	long onMacTimeout(FXObject *sender, FXSelector sel, void *ptr);
};

// FOX 1.7 changes the timeouts to all be nanoseconds.
// Fox 1.6 had all timeouts as milliseconds.
#if (FOX_MINOR >= 7)
	const int timeout_scalar = 1000*1000;
#else
	const int timeout_scalar = 1;
#endif

FXMainWindow *g_main_window;


FXDEFMAP(MainWindow) MainWindowMap [] = {
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_CONNECT, MainWindow::onConnect ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_DISCONNECT, MainWindow::onDisconnect ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_RESCAN, MainWindow::onRescan ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_SEND_OUTPUT_REPORT, MainWindow::onSendOutputReport ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_SEND_FEATURE_REPORT, MainWindow::onSendFeatureReport ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_GET_FEATURE_REPORT, MainWindow::onGetFeatureReport ),
	FXMAPFUNC(SEL_COMMAND, MainWindow::ID_CLEAR, MainWindow::onClear ),
	FXMAPFUNC(SEL_TIMEOUT, MainWindow::ID_TIMER, MainWindow::onTimeout ),
	FXMAPFUNC(SEL_TIMEOUT, MainWindow::ID_MAC_TIMER, MainWindow::onMacTimeout ),
};

FXIMPLEMENT(MainWindow, FXMainWindow, MainWindowMap, ARRAYNUMBER(MainWindowMap));

MainWindow::MainWindow(FXApp *app)
	: FXMainWindow(app, "HIDAPI Test Application", NULL, NULL, DECOR_ALL, 200,100, 425,700)
{
	devices = NULL;
	connected_device = NULL;

	FXVerticalFrame *vf = new FXVerticalFrame(this, LAYOUT_FILL_Y|LAYOUT_FILL_X);

	FXLabel *label = new FXLabel(vf, "HIDAPI Test Tool");
	title_font = new FXFont(getApp(), "Arial", 14, FXFont::Bold);
	label->setFont(title_font);
	
	new FXLabel(vf,
		"Select a device and press Connect.", NULL, JUSTIFY_LEFT);
	new FXLabel(vf,
		"Output data bytes can be entered in the Output section, \n"
		"separated by space, comma or brackets. Data starting with 0x\n"
		"is treated as hex. Data beginning with a 0 is treated as \n"
		"octal. All other data is treated as decimal.", NULL, JUSTIFY_LEFT);
	new FXLabel(vf,
		"Data received from the device appears in the Input section.",
		NULL, JUSTIFY_LEFT);
	new FXLabel(vf,
		"Optionally, a report length may be specified. Extra bytes are\n"
		"padded with zeros. If no length is specified, the length is \n"
		"inferred from the data.",
		NULL, JUSTIFY_LEFT);
	new FXLabel(vf, "");

	// Device List and Connect/Disconnect buttons
	FXHorizontalFrame *hf = new FXHorizontalFrame(vf, LAYOUT_FILL_X);
	//device_list = new FXList(new FXHorizontalFrame(hf,FRAME_SUNKEN|FRAME_THICK, 0,0,0,0, 0,0,0,0), NULL, 0, LISTBOX_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_Y|LAYOUT_FIX_WIDTH|LAYOUT_FIX_HEIGHT, 0,0,300,200);
	device_list = new FXList(new FXHorizontalFrame(hf,FRAME_SUNKEN|FRAME_THICK|LAYOUT_FILL_X|LAYOUT_FILL_Y, 0,0,0,0, 0,0,0,0), NULL, 0, LISTBOX_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_Y, 0,0,300,200);
	FXVerticalFrame *buttonVF = new FXVerticalFrame(hf);
	connect_button = new FXButton(buttonVF, "Connect", NULL, this, ID_CONNECT, BUTTON_NORMAL|LAYOUT_FILL_X);
	disconnect_button = new FXButton(buttonVF, "Disconnect", NULL, this, ID_DISCONNECT, BUTTON_NORMAL|LAYOUT_FILL_X);
	disconnect_button->disable();
	rescan_button = new FXButton(buttonVF, "Re-Scan devices", NULL, this, ID_RESCAN, BUTTON_NORMAL|LAYOUT_FILL_X);
	new FXHorizontalFrame(buttonVF, 0, 0,0,0,0, 0,0,50,0);

	connected_label = new FXLabel(vf, "Disconnected");
	
	new FXHorizontalFrame(vf);
	
	// Output Group Box
	FXGroupBox *gb = new FXGroupBox(vf, "Output", FRAME_GROOVE|LAYOUT_FILL_X);
	FXMatrix *matrix = new FXMatrix(gb, 3, MATRIX_BY_COLUMNS|LAYOUT_FILL_X);
	new FXLabel(matrix, "Data");
	new FXLabel(matrix, "Length");
	new FXLabel(matrix, "");

	//hf = new FXHorizontalFrame(gb, LAYOUT_FILL_X);
	output_text = new FXTextField(matrix, 30, NULL, 0, TEXTFIELD_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
	output_text->setText("1 0x81 0");
	output_len = new FXTextField(matrix, 5, NULL, 0, TEXTFIELD_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
	output_button = new FXButton(matrix, "Send Output Report", NULL, this, ID_SEND_OUTPUT_REPORT, BUTTON_NORMAL|LAYOUT_FILL_X);
	output_button->disable();
	//new FXHorizontalFrame(matrix, LAYOUT_FILL_X);

	//hf = new FXHorizontalFrame(gb, LAYOUT_FILL_X);
	feature_text = new FXTextField(matrix, 30, NULL, 0, TEXTFIELD_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
	feature_len = new FXTextField(matrix, 5, NULL, 0, TEXTFIELD_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
	feature_button = new FXButton(matrix, "Send Feature Report", NULL, this, ID_SEND_FEATURE_REPORT, BUTTON_NORMAL|LAYOUT_FILL_X);
	feature_button->disable();

	get_feature_text = new FXTextField(matrix, 30, NULL, 0, TEXTFIELD_NORMAL|LAYOUT_FILL_X|LAYOUT_FILL_COLUMN);
	new FXWindow(matrix);
	get_feature_button = new FXButton(matrix, "Get Feature Report", NULL, this, ID_GET_FEATURE_REPORT, BUTTON_NORMAL|LAYOUT_FILL_X);
	get_feature_button->disable();


	// Input Group Box
	gb = new FXGroupBox(vf, "Input", FRAME_GROOVE|LAYOUT_FILL_X|LAYOUT_FILL_Y);
	FXVerticalFrame *innerVF = new FXVerticalFrame(gb, LAYOUT_FILL_X|LAYOUT_FILL_Y);
	input_text = new FXText(new FXHorizontalFrame(innerVF,LAYOUT_FILL_X|LAYOUT_FILL_Y|FRAME_SUNKEN|FRAME_THICK, 0,0,0,0, 0,0,0,0), NULL, 0, LAYOUT_FILL_X|LAYOUT_FILL_Y);
	input_text->setEditable(false);
	new FXButton(innerVF, "Clear", NULL, this, ID_CLEAR, BUTTON_NORMAL|LAYOUT_RIGHT);
	

}

MainWindow::~MainWindow()
{
	if (connected_device)
		hid_close(connected_device);
	hid_exit();
	delete title_font;
}

void
MainWindow::create()
{
	FXMainWindow::create();
	show();

	onRescan(NULL, 0, NULL);
	

#ifdef __APPLE__
	init_apple_message_system();
#endif
	
	getApp()->addTimeout(this, ID_MAC_TIMER,
		50 * timeout_scalar /*50ms*/);
}

long
MainWindow::onConnect(FXObject *sender, FXSelector sel, void *ptr)
{
	if (connected_device != NULL)
		return 1;
	
	FXint cur_item = device_list->getCurrentItem();
	if (cur_item < 0)
		return -1;
	FXListItem *item = device_list->getItem(cur_item);
	if (!item)
		return -1;
	struct hid_device_info *device_info = (struct hid_device_info*) item->getData();
	if (!device_info)
		return -1;
	
	connected_device =  hid_open_path(device_info->path);
	
	if (!connected_device) {
		FXMessageBox::error(this, MBOX_OK, "Device Error", "Unable To Connect to Device");
		return -1;
	}
	
	hid_set_nonblocking(connected_device, 1);

	getApp()->addTimeout(this, ID_TIMER,
		5 * timeout_scalar /*5ms*/);
	
	FXString s;
	s.format("Connected to: %04hx:%04hx -", device_info->vendor_id, device_info->product_id);
	s += FXString(" ") + device_info->manufacturer_string;
	s += FXString(" ") + device_info->product_string;
	connected_label->setText(s);
	output_button->enable();
	feature_button->enable();
	get_feature_button->enable();
	connect_button->disable();
	disconnect_button->enable();
	input_text->setText("");


	return 1;
}

long
MainWindow::onDisconnect(FXObject *sender, FXSelector sel, void *ptr)
{
	hid_close(connected_device);
	connected_device = NULL;
	connected_label->setText("Disconnected");
	output_button->disable();
	feature_button->disable();
	get_feature_button->disable();
	connect_button->enable();
	disconnect_button->disable();

	getApp()->removeTimeout(this, ID_TIMER);
	
	return 1;
}

long
MainWindow::onRescan(FXObject *sender, FXSelector sel, void *ptr)
{
	struct hid_device_info *cur_dev;

	device_list->clearItems();
	
	// List the Devices
	hid_free_enumeration(devices);
	devices = hid_enumerate(0x0, 0x0);
	cur_dev = devices;	
	while (cur_dev) {
		// Add it to the List Box.
		FXString s;
		FXString usage_str;
		s.format("%04hx:%04hx -", cur_dev->vendor_id, cur_dev->product_id);
		s += FXString(" ") + cur_dev->manufacturer_string;
		s += FXString(" ") + cur_dev->product_string;
		usage_str.format(" (usage: %04hx:%04hx) ", cur_dev->usage_page, cur_dev->usage);
		s += usage_str;
		FXListItem *li = new FXListItem(s, NULL, cur_dev);
		device_list->appendItem(li);
		
		cur_dev = cur_dev->next;
	}

	if (device_list->getNumItems() == 0)
		device_list->appendItem("*** No Devices Connected ***");
	else {
		device_list->selectItem(0);
	}

	return 1;
}

size_t
MainWindow::getDataFromTextField(FXTextField *tf, char *buf, size_t len)
{
	const char *delim = " ,{}\t\r\n";
	FXString data = tf->getText();
	const FXchar *d = data.text();
	size_t i = 0;
	
	// Copy the string from the GUI.
	size_t sz = strlen(d);
	char *str = (char*) malloc(sz+1);
	strcpy(str, d);
	
	// For each token in the string, parse and store in buf[].
	char *token = strtok(str, delim);
	while (token) {
		char *endptr;
		long int val = strtol(token, &endptr, 0);
		buf[i++] = val;
		token = strtok(NULL, delim);
	}
	
	free(str);
	return i;
}

/* getLengthFromTextField()
   Returns length:
	 0: empty text field
	>0: valid length
	-1: invalid length */
int
MainWindow::getLengthFromTextField(FXTextField *tf)
{
	long int len;
	FXString str = tf->getText();
	size_t sz = str.length();

	if (sz > 0) {
		char *endptr;
		len = strtol(str.text(), &endptr, 0);
		if (endptr != str.text() && *endptr == '\0') {
			if (len <= 0) {
				FXMessageBox::error(this, MBOX_OK, "Invalid length", "Enter a length greater than zero.");
				return -1;
			}
			return len;
		}
		else
			return -1;
	}

	return 0;
}

long
MainWindow::onSendOutputReport(FXObject *sender, FXSelector sel, void *ptr)
{
	char buf[256];
	size_t data_len, len;
	int textfield_len;

	memset(buf, 0x0, sizeof(buf));
	textfield_len = getLengthFromTextField(output_len);
	data_len = getDataFromTextField(output_text, buf, sizeof(buf));

	if (textfield_len < 0) {
		FXMessageBox::error(this, MBOX_OK, "Invalid length", "Length field is invalid. Please enter a number in hex, octal, or decimal.");
		return 1;
	}

	if (textfield_len > sizeof(buf)) {
		FXMessageBox::error(this, MBOX_OK, "Invalid length", "Length field is too long.");
		return 1;
	}

	len = (textfield_len)? textfield_len: data_len;

	int res = hid_write(connected_device, (const unsigned char*)buf, len);
	if (res < 0) {
		FXMessageBox::error(this, MBOX_OK, "Error Writing", "Could not write to device. Error reported was: %ls", hid_error(connected_device));
	}
	
	return 1;
}

long
MainWindow::onSendFeatureReport(FXObject *sender, FXSelector sel, void *ptr)
{
	char buf[256];
	size_t data_len, len;
	int textfield_len;

	memset(buf, 0x0, sizeof(buf));
	textfield_len = getLengthFromTextField(feature_len);
	data_len = getDataFromTextField(feature_text, buf, sizeof(buf));

	if (textfield_len < 0) {
		FXMessageBox::error(this, MBOX_OK, "Invalid length", "Length field is invalid. Please enter a number in hex, octal, or decimal.");
		return 1;
	}

	if (textfield_len > sizeof(buf)) {
		FXMessageBox::error(this, MBOX_OK, "Invalid length", "Length field is too long.");
		return 1;
	}

	len = (textfield_len)? textfield_len: data_len;

	int res = hid_send_feature_report(connected_device, (const unsigned char*)buf, len); 
	if (res < 0) {
		FXMessageBox::error(this, MBOX_OK, "Error Writing", "Could not send feature report to device. Error reported was: %ls", hid_error(connected_device));
	}

	return 1;
}

long
MainWindow::onGetFeatureReport(FXObject *sender, FXSelector sel, void *ptr)
{
	char buf[256];
	size_t len;

	memset(buf, 0x0, sizeof(buf));
	len = getDataFromTextField(get_feature_text, buf, sizeof(buf));

	if (len != 1) {
		FXMessageBox::error(this, MBOX_OK, "Too many numbers", "Enter only a single report number in the text field");
	}

	int res = hid_get_feature_report(connected_device, (unsigned char*)buf, sizeof(buf));
	if (res < 0) {
		FXMessageBox::error(this, MBOX_OK, "Error Getting Report", "Could not get feature report from device. Error reported was: %ls", hid_error(connected_device));
	}

	if (res > 0) {
		FXString s;
		s.format("Returned Feature Report. %d bytes:\n", res);
		for (int i = 0; i < res; i++) {
			FXString t;
			t.format("%02hhx ", buf[i]);
			s += t;
			if ((i+1) % 4 == 0)
				s += " ";
			if ((i+1) % 16 == 0)
				s += "\n";
		}
		s += "\n";
		input_text->appendText(s);
		input_text->setBottomLine(INT_MAX);
	}
	
	return 1;
}

long
MainWindow::onClear(FXObject *sender, FXSelector sel, void *ptr)
{
	input_text->setText("");
	return 1;
}

long
MainWindow::onTimeout(FXObject *sender, FXSelector sel, void *ptr)
{
	unsigned char buf[256];
	int res = hid_read(connected_device, buf, sizeof(buf));
	
	if (res > 0) {
		FXString s;
		s.format("Received %d bytes:\n", res);
		for (int i = 0; i < res; i++) {
			FXString t;
			t.format("%02hhx ", buf[i]);
			s += t;
			if ((i+1) % 4 == 0)
				s += " ";
			if ((i+1) % 16 == 0)
				s += "\n";
		}
		s += "\n";
		input_text->appendText(s);
		input_text->setBottomLine(INT_MAX);
	}
	if (res < 0) {
		input_text->appendText("hid_read() returned error\n");
		input_text->setBottomLine(INT_MAX);
	}

	getApp()->addTimeout(this, ID_TIMER,
		5 * timeout_scalar /*5ms*/);
	return 1;
}

long
MainWindow::onMacTimeout(FXObject *sender, FXSelector sel, void *ptr)
{
#ifdef __APPLE__
	check_apple_events();
	
	getApp()->addTimeout(this, ID_MAC_TIMER,
		50 * timeout_scalar /*50ms*/);
#endif

	return 1;
}

int main(int argc, char **argv)
{
	FXApp app("HIDAPI Test Application", "Signal 11 Software");
	app.init(argc, argv);
	g_main_window = new MainWindow(&app);
	app.create();
	app.run();
	return 0;
}
