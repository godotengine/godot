#include "rendering_device.h"


RenderingDevice *RenderingDevice::singleton=NULL;

RenderingDevice *RenderingDevice::get_singleton() {
	return singleton;
}

RenderingDevice::RenderingDevice() {

	singleton=this;
}
