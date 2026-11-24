package org.godotengine.godot.input;

import android.hardware.usb.UsbDevice;

interface HIDDevice
{
    public int getId();
    public int getVendorId();
    public int getProductId();
    public String getSerialNumber();
    public int getVersion();
    public String getManufacturerName();
    public String getProductName();
    public UsbDevice getDevice();
    public boolean open();
    public int writeReport(byte[] report, boolean feature);
    public boolean readReport(byte[] report, boolean feature);
    public void setFrozen(boolean frozen);
    public void close();
    public void shutdown();
}
