Implementation Notes
--------------------
NetBSD maps every `uhidev` device to one or more `uhid`
devices. Each `uhid` device only supports one report ID.
The parent device `uhidev` creates one `uhid` device per
report ID found in the hardware's report descriptor.

In the event there are no report ID(s) found within the
report descriptor, only one `uhid` device with a report ID
of `0` is created.

In order to remain compatible with existing `hidapi` APIs,
all the `uhid` devices created by the parent `uhidev` device
must be opened under the same `hid_device` instance to ensure
that we can route reports to their appropriate `uhid` device.

Internally the `uhid` driver will insert the report ID as
needed so we must also omit the report ID in any situation
where the `hidapi` API expects it to be included in the
report data stream.

Given the design of `uhid`, it must be augmented with extra
platform specific APIs to ensure that the exact relationship
between `uhidev` devices and `uhid` devices can be determined.

The NetBSD implementation does this via the `drvctl` kernel
driver. At present there is no known way to do this on OpenBSD
for a `uhid` implementation to be at the same level as the
NetBSD one.
