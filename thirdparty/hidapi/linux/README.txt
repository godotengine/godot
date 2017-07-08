
There are two implementations of HIDAPI for Linux. One (linux/hid.c) uses the
Linux hidraw driver, and the other (libusb/hid.c) uses libusb. Which one you
use depends on your application. Complete functionality of the hidraw
version depends on patches to the Linux kernel which are not currently in
the mainline. These patches have to do with sending and receiving feature
reports. The libusb implementation uses libusb to talk directly to the
device, bypassing any Linux HID driver. The disadvantage of the libusb
version is that it will only work with USB devices, while the hidraw
implementation will work with Bluetooth devices as well.

To use HIDAPI, simply drop either linux/hid.c or libusb/hid.c into your
application and build using the build parameters in the Makefile.


Libusb Implementation notes
----------------------------
For the libusb implementation, libusb-1.0 must be installed. Libusb 1.0 is
different than the legacy libusb 0.1 which is installed on many systems. To
install libusb-1.0 on Ubuntu and other Debian-based systems, run:
	sudo apt-get install libusb-1.0-0-dev


Hidraw Implementation notes
----------------------------
For the hidraw implementation, libudev headers and libraries are required to
build hidapi programs.  To install libudev libraries on Ubuntu,
and other Debian-based systems, run:
	sudo apt-get install libudev-dev

On Redhat-based systems, run the following as root:
	yum install libudev-devel

Unfortunately, the hidraw driver, which the linux version of hidapi is based
on, contains bugs in kernel versions < 2.6.36, which the client application
should be aware of.

Bugs (hidraw implementation only):
-----------------------------------
On Kernel versions < 2.6.34, if your device uses numbered reports, an extra
byte will be returned at the beginning of all reports returned from read()
for hidraw devices. This is worked around in the libary. No action should be
necessary in the client library.

On Kernel versions < 2.6.35, reports will only be sent using a Set_Report
transfer on the CONTROL endpoint. No data will ever be sent on an Interrupt
Out endpoint if one exists. This is fixed in 2.6.35. In 2.6.35, OUTPUT
reports will be sent to the device on the first INTERRUPT OUT endpoint if it
exists; If it does not exist, OUTPUT reports will be sent on the CONTROL
endpoint.

On Kernel versions < 2.6.36, add an extra byte containing the report number
to sent reports if numbered reports are used, and the device does not
contain an INTERRPUT OUT endpoint for OUTPUT transfers.  For example, if
your device uses numbered reports and wants to send {0x2 0xff 0xff 0xff} to
the device (0x2 is the report number), you must send {0x2 0x2 0xff 0xff
0xff}. If your device has the optional Interrupt OUT endpoint, this does not
apply (but really on 2.6.35 only, because 2.6.34 won't use the interrupt
out endpoint).
