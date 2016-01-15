/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Original file: aidl/ILicenseResultListener.aidl
 */
package com.google.android.vending.licensing;
import java.lang.String;
import android.os.RemoteException;
import android.os.IBinder;
import android.os.IInterface;
import android.os.Binder;
import android.os.Parcel;
public interface ILicenseResultListener extends android.os.IInterface
{
/** Local-side IPC implementation stub class. */
public static abstract class Stub extends android.os.Binder implements com.google.android.vending.licensing.ILicenseResultListener
{
private static final java.lang.String DESCRIPTOR = "com.android.vending.licensing.ILicenseResultListener";
/** Construct the stub at attach it to the interface. */
public Stub()
{
this.attachInterface(this, DESCRIPTOR);
}
/**
 * Cast an IBinder object into an ILicenseResultListener interface,
 * generating a proxy if needed.
 */
public static com.google.android.vending.licensing.ILicenseResultListener asInterface(android.os.IBinder obj)
{
if ((obj==null)) {
return null;
}
android.os.IInterface iin = (android.os.IInterface)obj.queryLocalInterface(DESCRIPTOR);
if (((iin!=null)&&(iin instanceof com.google.android.vending.licensing.ILicenseResultListener))) {
return ((com.google.android.vending.licensing.ILicenseResultListener)iin);
}
return new com.google.android.vending.licensing.ILicenseResultListener.Stub.Proxy(obj);
}
public android.os.IBinder asBinder()
{
return this;
}
public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
{
switch (code)
{
case INTERFACE_TRANSACTION:
{
reply.writeString(DESCRIPTOR);
return true;
}
case TRANSACTION_verifyLicense:
{
data.enforceInterface(DESCRIPTOR);
int _arg0;
_arg0 = data.readInt();
java.lang.String _arg1;
_arg1 = data.readString();
java.lang.String _arg2;
_arg2 = data.readString();
this.verifyLicense(_arg0, _arg1, _arg2);
return true;
}
}
return super.onTransact(code, data, reply, flags);
}
private static class Proxy implements com.google.android.vending.licensing.ILicenseResultListener
{
private android.os.IBinder mRemote;
Proxy(android.os.IBinder remote)
{
mRemote = remote;
}
public android.os.IBinder asBinder()
{
return mRemote;
}
public java.lang.String getInterfaceDescriptor()
{
return DESCRIPTOR;
}
public void verifyLicense(int responseCode, java.lang.String signedData, java.lang.String signature) throws android.os.RemoteException
{
android.os.Parcel _data = android.os.Parcel.obtain();
try {
_data.writeInterfaceToken(DESCRIPTOR);
_data.writeInt(responseCode);
_data.writeString(signedData);
_data.writeString(signature);
mRemote.transact(Stub.TRANSACTION_verifyLicense, _data, null, IBinder.FLAG_ONEWAY);
}
finally {
_data.recycle();
}
}
}
static final int TRANSACTION_verifyLicense = (IBinder.FIRST_CALL_TRANSACTION + 0);
}
public void verifyLicense(int responseCode, java.lang.String signedData, java.lang.String signature) throws android.os.RemoteException;
}
