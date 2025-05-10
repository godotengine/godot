/*******************************
 Mac support for HID Test GUI
 
 Alan Ott
 Signal 11 Software
 
*******************************/

#ifndef MAC_SUPPORT_H__
#define MAC_SUPPORT_H__

extern "C" {
	void init_apple_message_system();
	void check_apple_events();
}

#endif
