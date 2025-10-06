package org.godotengine.godot.input;

import android.hardware.usb.*;
import android.os.Build;
import android.util.Log;
import java.util.Arrays;

class HIDDeviceUSB implements HIDDevice {

    private static final String TAG = "hidapi";

    protected HIDDeviceManager mManager;
    protected UsbDevice mDevice;
    protected int mInterfaceIndex;
    protected int mInterface;
    protected int mDeviceId;
    protected UsbDeviceConnection mConnection;
    protected UsbEndpoint mInputEndpoint;
    protected UsbEndpoint mOutputEndpoint;
    protected InputThread mInputThread;
    protected boolean mRunning;
    protected boolean mFrozen;

    public HIDDeviceUSB(HIDDeviceManager manager, UsbDevice usbDevice, int interface_index) {
        mManager = manager;
        mDevice = usbDevice;
        mInterfaceIndex = interface_index;
        mInterface = mDevice.getInterface(mInterfaceIndex).getId();
        mDeviceId = manager.getDeviceIDForIdentifier(getIdentifier());
        mRunning = false;
    }

    public String getIdentifier() {
        return String.format("%s/%x/%x/%d", mDevice.getDeviceName(), mDevice.getVendorId(), mDevice.getProductId(), mInterfaceIndex);
    }

    @Override
    public int getId() {
        return mDeviceId;
    }

    @Override
    public int getVendorId() {
        return mDevice.getVendorId();
    }

    @Override
    public int getProductId() {
        return mDevice.getProductId();
    }

    @Override
    public String getSerialNumber() {
        String result = null;
        if (Build.VERSION.SDK_INT >= 21 /* Android 5.0 (LOLLIPOP) */) {
            try {
                result = mDevice.getSerialNumber();
            }
            catch (SecurityException exception) {
                //Log.w(TAG, "App permissions mean we cannot get serial number for device " + getDeviceName() + " message: " + exception.getMessage());
            }
        }
        if (result == null) {
            result = "";
        }
        return result;
    }

    @Override
    public int getVersion() {
        return 0;
    }

    @Override
    public String getManufacturerName() {
        String result = null;
        if (Build.VERSION.SDK_INT >= 21 /* Android 5.0 (LOLLIPOP) */) {
            result = mDevice.getManufacturerName();
        }
        if (result == null) {
            result = String.format("%x", getVendorId());
        }
        return result;
    }

    @Override
    public String getProductName() {
        String result = null;
        if (Build.VERSION.SDK_INT >= 21 /* Android 5.0 (LOLLIPOP) */) {
            result = mDevice.getProductName();
        }
        if (result == null) {
            result = String.format("%x", getProductId());
        }
        return result;
    }

    @Override
    public UsbDevice getDevice() {
        return mDevice;
    }

    public String getDeviceName() {
        return getManufacturerName() + " " + getProductName() + "(0x" + String.format("%x", getVendorId()) + "/0x" + String.format("%x", getProductId()) + ")";
    }

    @Override
    public boolean open() {
        mConnection = mManager.getUSBManager().openDevice(mDevice);
        if (mConnection == null) {
            Log.w(TAG, "Unable to open USB device " + getDeviceName());
            return false;
        }

        // Force claim our interface
        UsbInterface iface = mDevice.getInterface(mInterfaceIndex);
        if (!mConnection.claimInterface(iface, true)) {
            Log.w(TAG, "Failed to claim interfaces on USB device " + getDeviceName());
            close();
            return false;
        }

        // Find the endpoints
        for (int j = 0; j < iface.getEndpointCount(); j++) {
            UsbEndpoint endpt = iface.getEndpoint(j);
            switch (endpt.getDirection()) {
            case UsbConstants.USB_DIR_IN:
                if (mInputEndpoint == null) {
                    mInputEndpoint = endpt;
                }
                break;
            case UsbConstants.USB_DIR_OUT:
                if (mOutputEndpoint == null) {
                    mOutputEndpoint = endpt;
                }
                break;
            }
        }

        // Make sure the required endpoints were present
        if (mInputEndpoint == null || mOutputEndpoint == null) {
            Log.w(TAG, "Missing required endpoint on USB device " + getDeviceName());
            close();
            return false;
        }

        // Start listening for input
        mRunning = true;
        mInputThread = new InputThread();
        mInputThread.start();

        return true;
    }

    @Override
    public int writeReport(byte[] report, boolean feature) {
        if (mConnection == null) {
            Log.w(TAG, "writeReport() called with no device connection");
            return -1;
        }

        if (feature) {
            int res = -1;
            int offset = 0;
            int length = report.length;
            boolean skipped_report_id = false;
            byte report_number = report[0];

            if (report_number == 0x0) {
                ++offset;
                --length;
                skipped_report_id = true;
            }

            res = mConnection.controlTransfer(
                UsbConstants.USB_TYPE_CLASS | 0x01 /*RECIPIENT_INTERFACE*/ | UsbConstants.USB_DIR_OUT,
                0x09/*HID set_report*/,
                (3/*HID feature*/ << 8) | report_number,
                mInterface,
                report, offset, length,
                1000/*timeout millis*/);

            if (res < 0) {
                Log.w(TAG, "writeFeatureReport() returned " + res + " on device " + getDeviceName());
                return -1;
            }

            if (skipped_report_id) {
                ++length;
            }
            return length;
        } else {
            int res = mConnection.bulkTransfer(mOutputEndpoint, report, report.length, 1000);
            if (res != report.length) {
                Log.w(TAG, "writeOutputReport() returned " + res + " on device " + getDeviceName());
            }
            return res;
        }
    }

    @Override
    public boolean readReport(byte[] report, boolean feature) {
        int res = -1;
        int offset = 0;
        int length = report.length;
        boolean skipped_report_id = false;
        byte report_number = report[0];

        if (mConnection == null) {
            Log.w(TAG, "readReport() called with no device connection");
            return false;
        }

        if (report_number == 0x0) {
            /* Offset the return buffer by 1, so that the report ID
               will remain in byte 0. */
            ++offset;
            --length;
            skipped_report_id = true;
        }

        res = mConnection.controlTransfer(
            UsbConstants.USB_TYPE_CLASS | 0x01 /*RECIPIENT_INTERFACE*/ | UsbConstants.USB_DIR_IN,
            0x01/*HID get_report*/,
            ((feature ? 3/*HID feature*/ : 1/*HID Input*/) << 8) | report_number,
            mInterface,
            report, offset, length,
            1000/*timeout millis*/);

        if (res < 0) {
            Log.w(TAG, "getFeatureReport() returned " + res + " on device " + getDeviceName());
            return false;
        }

        if (skipped_report_id) {
            ++res;
            ++length;
        }

        byte[] data;
        if (res == length) {
            data = report;
        } else {
            data = Arrays.copyOfRange(report, 0, res);
        }
        mManager.HIDDeviceReportResponse(mDeviceId, data);

        return true;
    }

    @Override
    public void close() {
        mRunning = false;
        if (mInputThread != null) {
            while (mInputThread.isAlive()) {
                mInputThread.interrupt();
                try {
                    mInputThread.join();
                } catch (InterruptedException e) {
                    // Keep trying until we're done
                }
            }
            mInputThread = null;
        }
        if (mConnection != null) {
            UsbInterface iface = mDevice.getInterface(mInterfaceIndex);
            mConnection.releaseInterface(iface);
            mConnection.close();
            mConnection = null;
        }
    }

    @Override
    public void shutdown() {
        close();
        mManager = null;
    }

    @Override
    public void setFrozen(boolean frozen) {
        mFrozen = frozen;
    }

    protected class InputThread extends Thread {
        @Override
        public void run() {
            int packetSize = mInputEndpoint.getMaxPacketSize();
            byte[] packet = new byte[packetSize];
            while (mRunning) {
                int r;
                try
                {
                    r = mConnection.bulkTransfer(mInputEndpoint, packet, packetSize, 1000);
                }
                catch (Exception e)
                {
                    Log.v(TAG, "Exception in UsbDeviceConnection bulktransfer: " + e);
                    break;
                }
                if (r < 0) {
                    // Could be a timeout or an I/O error
                }
                if (r > 0) {
                    byte[] data;
                    if (r == packetSize) {
                        data = packet;
                    } else {
                        data = Arrays.copyOfRange(packet, 0, r);
                    }

                    if (!mFrozen) {
                        mManager.HIDDeviceInputReport(mDeviceId, data);
                    }
                }
            }
        }
    }
}
