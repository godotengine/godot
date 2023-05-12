/************************************************************
Copyright (c) 1993 by Silicon Graphics Computer Systems, Inc.

Permission to use, copy, modify, and distribute this
software and its documentation for any purpose and without
fee is hereby granted, provided that the above copyright
notice appear in all copies and that both that copyright
notice and this permission notice appear in supporting
documentation, and that the name of Silicon Graphics not be
used in advertising or publicity pertaining to distribution
of the software without specific prior written permission.
Silicon Graphics makes no representation about the suitability
of this software for any purpose. It is provided "as is"
without any express or implied warranty.

SILICON GRAPHICS DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL SILICON
GRAPHICS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION  WITH
THE USE OR PERFORMANCE OF THIS SOFTWARE.

********************************************************/

#ifndef _XKB_H_
#define	_XKB_H_

    /*
     * XKB request codes, used in:
     *  -  xkbReqType field of all requests
     *  -  requestMinor field of some events
     */
#define	X_kbUseExtension		 0
#define	X_kbSelectEvents	 	 1
#define	X_kbBell			 3
#define	X_kbGetState			 4
#define	X_kbLatchLockState		 5
#define	X_kbGetControls			 6
#define	X_kbSetControls			 7
#define	X_kbGetMap			 8
#define	X_kbSetMap			 9
#define	X_kbGetCompatMap		10
#define	X_kbSetCompatMap		11
#define	X_kbGetIndicatorState		12
#define	X_kbGetIndicatorMap		13
#define	X_kbSetIndicatorMap		14
#define	X_kbGetNamedIndicator		15
#define	X_kbSetNamedIndicator		16
#define	X_kbGetNames			17
#define	X_kbSetNames			18
#define	X_kbGetGeometry			19
#define	X_kbSetGeometry			20
#define	X_kbPerClientFlags		21
#define	X_kbListComponents		22
#define	X_kbGetKbdByName		23
#define	X_kbGetDeviceInfo		24
#define	X_kbSetDeviceInfo		25
#define	X_kbSetDebuggingFlags		101

    /*
     * In the X sense, XKB reports only one event.
     * The type field of all XKB events is XkbEventCode
     */
#define	XkbEventCode			0
#define	XkbNumberEvents			(XkbEventCode+1)

    /*
     * XKB has a minor event code so it can use one X event code for
     * multiple purposes.
     *  - reported in the xkbType field of all XKB events.
     *  - XkbSelectEventDetails: Indicates the event for which event details
     *    are being changed
     */
#define	XkbNewKeyboardNotify		0
#define XkbMapNotify			1
#define	XkbStateNotify			2
#define XkbControlsNotify		3
#define	XkbIndicatorStateNotify		4
#define	XkbIndicatorMapNotify		5
#define	XkbNamesNotify			6
#define XkbCompatMapNotify		7
#define	XkbBellNotify			8
#define	XkbActionMessage		9
#define	XkbAccessXNotify		10
#define	XkbExtensionDeviceNotify	11

    /*
     * Event Mask:
     *  - XkbSelectEvents:  Specifies event interest.
     */
#define	XkbNewKeyboardNotifyMask	(1L << 0)
#define XkbMapNotifyMask		(1L << 1)
#define	XkbStateNotifyMask		(1L << 2)
#define XkbControlsNotifyMask		(1L << 3)
#define	XkbIndicatorStateNotifyMask	(1L << 4)
#define	XkbIndicatorMapNotifyMask	(1L << 5)
#define	XkbNamesNotifyMask		(1L << 6)
#define XkbCompatMapNotifyMask		(1L << 7)
#define	XkbBellNotifyMask		(1L << 8)
#define	XkbActionMessageMask		(1L << 9)
#define	XkbAccessXNotifyMask		(1L << 10)
#define	XkbExtensionDeviceNotifyMask	(1L << 11)
#define	XkbAllEventsMask		(0xFFF)

    /*
     * NewKeyboardNotify event details:
     */
#define	XkbNKN_KeycodesMask		(1L << 0)
#define	XkbNKN_GeometryMask		(1L << 1)
#define	XkbNKN_DeviceIDMask		(1L << 2)
#define	XkbAllNewKeyboardEventsMask	(0x7)

    /*
     * AccessXNotify event types:
     *  - The 'what' field of AccessXNotify events reports the
     *    reason that the event was generated.
     */
#define	XkbAXN_SKPress			0
#define	XkbAXN_SKAccept			1
#define	XkbAXN_SKReject			2
#define	XkbAXN_SKRelease		3
#define	XkbAXN_BKAccept			4
#define	XkbAXN_BKReject			5
#define	XkbAXN_AXKWarning		6

    /*
     * AccessXNotify details:
     * - Used as an event detail mask to limit the conditions under which
     *   AccessXNotify events are reported
     */
#define	XkbAXN_SKPressMask		(1L << 0)
#define	XkbAXN_SKAcceptMask		(1L << 1)
#define	XkbAXN_SKRejectMask		(1L << 2)
#define	XkbAXN_SKReleaseMask		(1L << 3)
#define	XkbAXN_BKAcceptMask		(1L << 4)
#define	XkbAXN_BKRejectMask		(1L << 5)
#define	XkbAXN_AXKWarningMask		(1L << 6)
#define	XkbAllAccessXEventsMask		(0x7f)

    /*
     * Miscellaneous event details:
     * - event detail masks for assorted events that don't really
     *   have any details.
     */
#define	XkbAllStateEventsMask		XkbAllStateComponentsMask
#define	XkbAllMapEventsMask		XkbAllMapComponentsMask
#define	XkbAllControlEventsMask		XkbAllControlsMask
#define	XkbAllIndicatorEventsMask	XkbAllIndicatorsMask
#define	XkbAllNameEventsMask		XkbAllNamesMask
#define	XkbAllCompatMapEventsMask	XkbAllCompatMask
#define	XkbAllBellEventsMask		(1L << 0)
#define	XkbAllActionMessagesMask	(1L << 0)

    /*
     * XKB reports one error:  BadKeyboard
     * A further reason for the error is encoded into to most significant
     * byte of the resourceID for the error:
     *    XkbErr_BadDevice - the device in question was not found
     *    XkbErr_BadClass  - the device was found but it doesn't belong to
     *                       the appropriate class.
     *    XkbErr_BadId     - the device was found and belongs to the right
     *                       class, but not feedback with a matching id was
     *                       found.
     * The low byte of the resourceID for this error contains the device
     * id, class specifier or feedback id that failed.
     */
#define	XkbKeyboard			0
#define	XkbNumberErrors			1

#define	XkbErr_BadDevice	0xff
#define	XkbErr_BadClass		0xfe
#define	XkbErr_BadId		0xfd

    /*
     * Keyboard Components Mask:
     * - Specifies the components that follow a GetKeyboardByNameReply
     */
#define	XkbClientMapMask		(1L << 0)
#define	XkbServerMapMask		(1L << 1)
#define	XkbCompatMapMask		(1L << 2)
#define	XkbIndicatorMapMask		(1L << 3)
#define	XkbNamesMask			(1L << 4)
#define	XkbGeometryMask			(1L << 5)
#define	XkbControlsMask			(1L << 6)
#define	XkbAllComponentsMask		(0x7f)

    /*
     * State detail mask:
     *  - The 'changed' field of StateNotify events reports which of
     *    the keyboard state components have changed.
     *  - Used as an event detail mask to limit the conditions under
     *    which StateNotify events are reported.
     */
#define	XkbModifierStateMask		(1L << 0)
#define	XkbModifierBaseMask		(1L << 1)
#define	XkbModifierLatchMask		(1L << 2)
#define	XkbModifierLockMask		(1L << 3)
#define	XkbGroupStateMask		(1L << 4)
#define	XkbGroupBaseMask		(1L << 5)
#define	XkbGroupLatchMask		(1L << 6)
#define XkbGroupLockMask		(1L << 7)
#define	XkbCompatStateMask		(1L << 8)
#define	XkbGrabModsMask			(1L << 9)
#define	XkbCompatGrabModsMask		(1L << 10)
#define	XkbLookupModsMask		(1L << 11)
#define	XkbCompatLookupModsMask		(1L << 12)
#define	XkbPointerButtonMask		(1L << 13)
#define	XkbAllStateComponentsMask	(0x3fff)

    /*
     * Controls detail masks:
     *  The controls specified in XkbAllControlsMask:
     *  - The 'changed' field of ControlsNotify events reports which of
     *    the keyboard controls have changed.
     *  - The 'changeControls' field of the SetControls request specifies
     *    the controls for which values are to be changed.
     *  - Used as an event detail mask to limit the conditions under
     *    which ControlsNotify events are reported.
     *
     *  The controls specified in the XkbAllBooleanCtrlsMask:
     *  - The 'enabledControls' field of ControlsNotify events reports the
     *    current status of the boolean controls.
     *  - The 'enabledControlsChanges' field of ControlsNotify events reports
     *    any boolean controls that have been turned on or off.
     *  - The 'affectEnabledControls' and 'enabledControls' fields of the
     *    kbSetControls request change the set of enabled controls.
     *  - The 'accessXTimeoutMask' and 'accessXTimeoutValues' fields of
     *    an XkbControlsRec specify the controls to be changed if the keyboard
     *    times out and the values to which they should be changed.
     *  - The 'autoCtrls' and 'autoCtrlsValues' fields of the PerClientFlags
     *    request specifies the specify the controls to be reset when the
     *    client exits and the values to which they should be reset.
     *  - The 'ctrls' field of an indicator map specifies the controls
     *    that drive the indicator.
     *  - Specifies the boolean controls affected by the SetControls and
     *    LockControls key actions.
     */
#define	XkbRepeatKeysMask	 (1L << 0)
#define	XkbSlowKeysMask		 (1L << 1)
#define	XkbBounceKeysMask	 (1L << 2)
#define	XkbStickyKeysMask	 (1L << 3)
#define	XkbMouseKeysMask	 (1L << 4)
#define	XkbMouseKeysAccelMask	 (1L << 5)
#define	XkbAccessXKeysMask	 (1L << 6)
#define	XkbAccessXTimeoutMask	 (1L << 7)
#define	XkbAccessXFeedbackMask	 (1L << 8)
#define	XkbAudibleBellMask	 (1L << 9)
#define	XkbOverlay1Mask		 (1L << 10)
#define	XkbOverlay2Mask		 (1L << 11)
#define	XkbIgnoreGroupLockMask	 (1L << 12)
#define	XkbGroupsWrapMask	 (1L << 27)
#define	XkbInternalModsMask	 (1L << 28)
#define	XkbIgnoreLockModsMask	 (1L << 29)
#define	XkbPerKeyRepeatMask	 (1L << 30)
#define	XkbControlsEnabledMask	 (1L << 31)

#define	XkbAccessXOptionsMask    (XkbStickyKeysMask|XkbAccessXFeedbackMask)

#define	XkbAllBooleanCtrlsMask	 (0x00001FFF)
#define	XkbAllControlsMask	 (0xF8001FFF)
#define	XkbAllControlEventsMask	 XkbAllControlsMask

    /*
     * AccessX Options Mask
     *  - The 'accessXOptions' field of an XkbControlsRec specifies the
     *    AccessX options that are currently in effect.
     *  - The 'accessXTimeoutOptionsMask' and 'accessXTimeoutOptionsValues'
     *    fields of an XkbControlsRec specify the Access X options to be
     *    changed if the keyboard times out and the values to which they
     *    should be changed.
     */
#define	XkbAX_SKPressFBMask	(1L << 0)
#define	XkbAX_SKAcceptFBMask	(1L << 1)
#define	XkbAX_FeatureFBMask	(1L << 2)
#define	XkbAX_SlowWarnFBMask	(1L << 3)
#define	XkbAX_IndicatorFBMask	(1L << 4)
#define	XkbAX_StickyKeysFBMask	(1L << 5)
#define	XkbAX_TwoKeysMask	(1L << 6)
#define	XkbAX_LatchToLockMask	(1L << 7)
#define	XkbAX_SKReleaseFBMask	(1L << 8)
#define	XkbAX_SKRejectFBMask	(1L << 9)
#define	XkbAX_BKRejectFBMask	(1L << 10)
#define	XkbAX_DumbBellFBMask	(1L << 11)
#define	XkbAX_FBOptionsMask	(0xF3F)
#define	XkbAX_SKOptionsMask	(0x0C0)
#define	XkbAX_AllOptionsMask	(0xFFF)

    /*
     * XkbUseCoreKbd is used to specify the core keyboard without having
     * 			to look up its X input extension identifier.
     * XkbUseCorePtr is used to specify the core pointer without having
     *			to look up its X input extension identifier.
     * XkbDfltXIClass is used to specify "don't care" any place that the
     *			XKB protocol is looking for an X Input Extension
     *			device class.
     * XkbDfltXIId is used to specify "don't care" any place that the
     *			XKB protocol is looking for an X Input Extension
     *			feedback identifier.
     * XkbAllXIClasses is used to get information about all device indicators,
     *			whether they're part of the indicator feedback class
     *			or the keyboard feedback class.
     * XkbAllXIIds is used to get information about all device indicator
     *			feedbacks without having to list them.
     * XkbXINone is used to indicate that no class or id has been specified.
     * XkbLegalXILedClass(c)  True if 'c' specifies a legal class with LEDs
     * XkbLegalXIBellClass(c) True if 'c' specifies a legal class with bells
     * XkbExplicitXIDevice(d) True if 'd' explicitly specifies a device
     * XkbExplicitXIClass(c)  True if 'c' explicitly specifies a device class
     * XkbExplicitXIId(c)     True if 'i' explicitly specifies a device id
     * XkbSingleXIClass(c)    True if 'c' specifies exactly one device class,
     *                        including the default.
     * XkbSingleXIId(i)       True if 'i' specifies exactly one device
     *	                      identifier, including the default.
     */
#define	XkbUseCoreKbd		0x0100
#define	XkbUseCorePtr		0x0200
#define	XkbDfltXIClass		0x0300
#define	XkbDfltXIId		0x0400
#define	XkbAllXIClasses		0x0500
#define	XkbAllXIIds		0x0600
#define	XkbXINone		0xff00

#define	XkbLegalXILedClass(c)	(((c)==KbdFeedbackClass)||\
					((c)==LedFeedbackClass)||\
					((c)==XkbDfltXIClass)||\
					((c)==XkbAllXIClasses))
#define	XkbLegalXIBellClass(c)	(((c)==KbdFeedbackClass)||\
					((c)==BellFeedbackClass)||\
					((c)==XkbDfltXIClass)||\
					((c)==XkbAllXIClasses))
#define	XkbExplicitXIDevice(c)	(((c)&(~0xff))==0)
#define	XkbExplicitXIClass(c)	(((c)&(~0xff))==0)
#define	XkbExplicitXIId(c)	(((c)&(~0xff))==0)
#define	XkbSingleXIClass(c)	((((c)&(~0xff))==0)||((c)==XkbDfltXIClass))
#define	XkbSingleXIId(c)	((((c)&(~0xff))==0)||((c)==XkbDfltXIId))

#define	XkbNoModifier		0xff
#define	XkbNoShiftLevel		0xff
#define	XkbNoShape		0xff
#define	XkbNoIndicator		0xff

#define	XkbNoModifierMask	0
#define	XkbAllModifiersMask	0xff
#define	XkbAllVirtualModsMask	0xffff

#define	XkbNumKbdGroups		4
#define	XkbMaxKbdGroup		(XkbNumKbdGroups-1)

#define	XkbMaxMouseKeysBtn	4

    /*
     * Group Index and Mask:
     *  - Indices into the kt_index array of a key type.
     *  - Mask specifies types to be changed for XkbChangeTypesOfKey
     */
#define	XkbGroup1Index		0
#define	XkbGroup2Index		1
#define	XkbGroup3Index		2
#define	XkbGroup4Index		3
#define	XkbAnyGroup		254
#define	XkbAllGroups		255

#define	XkbGroup1Mask		(1<<0)
#define	XkbGroup2Mask		(1<<1)
#define	XkbGroup3Mask		(1<<2)
#define	XkbGroup4Mask		(1<<3)
#define	XkbAnyGroupMask		(1<<7)
#define	XkbAllGroupsMask	(0xf)

    /*
     * BuildCoreState: Given a keyboard group and a modifier state,
     *                 construct the value to be reported an event.
     * GroupForCoreState:  Given the state reported in an event,
     *                 determine the keyboard group.
     * IsLegalGroup:   Returns TRUE if 'g' is a valid group index.
     */
#define	XkbBuildCoreState(m,g)	((((g)&0x3)<<13)|((m)&0xff))
#define XkbGroupForCoreState(s)	(((s)>>13)&0x3)
#define	XkbIsLegalGroup(g)	(((g)>=0)&&((g)<XkbNumKbdGroups))

    /*
     * GroupsWrap values:
     *  - The 'groupsWrap' field of an XkbControlsRec specifies the
     *    treatment of out of range groups.
     *  - Bits 6 and 7 of the group info field of a key symbol map
     *    specify the interpretation of out of range groups for the
     *    corresponding key.
     */
#define	XkbWrapIntoRange	(0x00)
#define	XkbClampIntoRange	(0x40)
#define	XkbRedirectIntoRange	(0x80)

    /*
     * Action flags:  Reported in the 'flags' field of most key actions.
     * Interpretation depends on the type of the action; not all actions
     * accept all flags.
     *
     * Option			Used for Actions
     * ------			----------------
     * ClearLocks		SetMods, LatchMods, SetGroup, LatchGroup
     * LatchToLock		SetMods, LatchMods, SetGroup, LatchGroup
     * LockNoLock		LockMods, ISOLock, LockPtrBtn, LockDeviceBtn
     * LockNoUnlock		LockMods, ISOLock, LockPtrBtn, LockDeviceBtn
     * UseModMapMods		SetMods, LatchMods, LockMods, ISOLock
     * GroupAbsolute		SetGroup, LatchGroup, LockGroup, ISOLock
     * UseDfltButton		PtrBtn, LockPtrBtn
     * NoAcceleration		MovePtr
     * MoveAbsoluteX		MovePtr
     * MoveAbsoluteY		MovePtr
     * ISODfltIsGroup		ISOLock
     * ISONoAffectMods		ISOLock
     * ISONoAffectGroup		ISOLock
     * ISONoAffectPtr		ISOLock
     * ISONoAffectCtrls		ISOLock
     * MessageOnPress		ActionMessage
     * MessageOnRelease		ActionMessage
     * MessageGenKeyEvent	ActionMessage
     * AffectDfltBtn		SetPtrDflt
     * DfltBtnAbsolute		SetPtrDflt
     * SwitchApplication	SwitchScreen
     * SwitchAbsolute		SwitchScreen
     */

#define	XkbSA_ClearLocks	(1L << 0)
#define	XkbSA_LatchToLock	(1L << 1)

#define	XkbSA_LockNoLock	(1L << 0)
#define	XkbSA_LockNoUnlock	(1L << 1)

#define	XkbSA_UseModMapMods	(1L << 2)

#define	XkbSA_GroupAbsolute	(1L << 2)
#define	XkbSA_UseDfltButton	0

#define	XkbSA_NoAcceleration	(1L << 0)
#define	XkbSA_MoveAbsoluteX	(1L << 1)
#define	XkbSA_MoveAbsoluteY	(1L << 2)

#define	XkbSA_ISODfltIsGroup 	 (1L << 7)
#define	XkbSA_ISONoAffectMods	 (1L << 6)
#define	XkbSA_ISONoAffectGroup	 (1L << 5)
#define	XkbSA_ISONoAffectPtr	 (1L << 4)
#define	XkbSA_ISONoAffectCtrls	 (1L << 3)
#define	XkbSA_ISOAffectMask	 (0x78)

#define	XkbSA_MessageOnPress	 (1L << 0)
#define	XkbSA_MessageOnRelease	 (1L << 1)
#define	XkbSA_MessageGenKeyEvent (1L << 2)

#define	XkbSA_AffectDfltBtn	1
#define	XkbSA_DfltBtnAbsolute	(1L << 2)

#define	XkbSA_SwitchApplication	(1L << 0)
#define	XkbSA_SwitchAbsolute	(1L << 2)

    /*
     * The following values apply to the SA_DeviceValuator
     * action only.  Valuator operations specify the action
     * to be taken.   Values specified in the action are
     * multiplied by 2^scale before they are applied.
     */
#define	XkbSA_IgnoreVal		(0x00)
#define	XkbSA_SetValMin		(0x10)
#define	XkbSA_SetValCenter	(0x20)
#define	XkbSA_SetValMax		(0x30)
#define	XkbSA_SetValRelative	(0x40)
#define	XkbSA_SetValAbsolute	(0x50)
#define	XkbSA_ValOpMask		(0x70)
#define	XkbSA_ValScaleMask	(0x07)
#define	XkbSA_ValOp(a)		((a)&XkbSA_ValOpMask)
#define	XkbSA_ValScale(a)	((a)&XkbSA_ValScaleMask)

    /*
     * Action types: specifies the type of a key action.  Reported in the
     * type field of all key actions.
     */
#define	XkbSA_NoAction		0x00
#define	XkbSA_SetMods		0x01
#define	XkbSA_LatchMods		0x02
#define	XkbSA_LockMods		0x03
#define	XkbSA_SetGroup		0x04
#define	XkbSA_LatchGroup	0x05
#define	XkbSA_LockGroup		0x06
#define	XkbSA_MovePtr		0x07
#define	XkbSA_PtrBtn		0x08
#define	XkbSA_LockPtrBtn	0x09
#define	XkbSA_SetPtrDflt	0x0a
#define	XkbSA_ISOLock		0x0b
#define	XkbSA_Terminate		0x0c
#define	XkbSA_SwitchScreen	0x0d
#define	XkbSA_SetControls	0x0e
#define	XkbSA_LockControls	0x0f
#define	XkbSA_ActionMessage	0x10
#define	XkbSA_RedirectKey	0x11
#define	XkbSA_DeviceBtn		0x12
#define	XkbSA_LockDeviceBtn	0x13
#define	XkbSA_DeviceValuator	0x14
#define	XkbSA_LastAction	XkbSA_DeviceValuator
#define	XkbSA_NumActions	(XkbSA_LastAction+1)

#define	XkbSA_XFree86Private	0x86

    /*
     * Specifies the key actions that clear latched groups or modifiers.
     */
#define	XkbSA_BreakLatch \
	((1<<XkbSA_NoAction)|(1<<XkbSA_PtrBtn)|(1<<XkbSA_LockPtrBtn)|\
	(1<<XkbSA_Terminate)|(1<<XkbSA_SwitchScreen)|(1<<XkbSA_SetControls)|\
	(1<<XkbSA_LockControls)|(1<<XkbSA_ActionMessage)|\
	(1<<XkbSA_RedirectKey)|(1<<XkbSA_DeviceBtn)|(1<<XkbSA_LockDeviceBtn))

    /*
     * Macros to classify key actions
     */
#define	XkbIsModAction(a)	(((a)->type>=Xkb_SASetMods)&&((a)->type<=XkbSA_LockMods))
#define	XkbIsGroupAction(a)	(((a)->type>=XkbSA_SetGroup)&&((a)->type<=XkbSA_LockGroup))
#define	XkbIsPtrAction(a)	(((a)->type>=XkbSA_MovePtr)&&((a)->type<=XkbSA_SetPtrDflt))


    /*
     * Key Behavior Qualifier:
     *    KB_Permanent indicates that the behavior describes an unalterable
     *    characteristic of the keyboard, not an XKB software-simulation of
     *    the listed behavior.
     * Key Behavior Types:
     *    Specifies the behavior of the underlying key.
     */
#define	XkbKB_Permanent		0x80
#define	XkbKB_OpMask		0x7f

#define	XkbKB_Default		0x00
#define	XkbKB_Lock		0x01
#define	XkbKB_RadioGroup	0x02
#define	XkbKB_Overlay1		0x03
#define	XkbKB_Overlay2		0x04

#define	XkbKB_RGAllowNone	0x80

    /*
     * Various macros which describe the range of legal keycodes.
     */
#define	XkbMinLegalKeyCode	8
#define	XkbMaxLegalKeyCode	255
#define	XkbMaxKeyCount		(XkbMaxLegalKeyCode-XkbMinLegalKeyCode+1)
#define	XkbPerKeyBitArraySize	((XkbMaxLegalKeyCode+1)/8)
/* Seems kinda silly to check that an unsigned char is <= 255... */
#define	XkbIsLegalKeycode(k)	((k)>=XkbMinLegalKeyCode)

    /*
     * Assorted constants and limits.
     */
#define	XkbNumModifiers		8
#define	XkbNumVirtualMods	16
#define	XkbNumIndicators	32
#define	XkbAllIndicatorsMask	(0xffffffff)
#define	XkbMaxRadioGroups	32
#define	XkbAllRadioGroupsMask	(0xffffffff)
#define	XkbMaxShiftLevel	63
#define	XkbMaxSymsPerKey	(XkbMaxShiftLevel*XkbNumKbdGroups)
#define	XkbRGMaxMembers		12
#define	XkbActionMessageLength	6
#define	XkbKeyNameLength	4
#define	XkbMaxRedirectCount	8

#define	XkbGeomPtsPerMM		10
#define	XkbGeomMaxColors	32
#define	XkbGeomMaxLabelColors	3
#define	XkbGeomMaxPriority	255

    /*
     * Key Type index and mask for the four standard key types.
     */
#define	XkbOneLevelIndex	0
#define	XkbTwoLevelIndex	1
#define	XkbAlphabeticIndex	2
#define	XkbKeypadIndex		3
#define	XkbLastRequiredType	XkbKeypadIndex
#define	XkbNumRequiredTypes	(XkbLastRequiredType+1)
#define	XkbMaxKeyTypes		255

#define	XkbOneLevelMask		(1<<0)
#define	XkbTwoLevelMask		(1<<1)
#define	XkbAlphabeticMask	(1<<2)
#define	XkbKeypadMask		(1<<3)
#define	XkbAllRequiredTypes	(0xf)

#define	XkbShiftLevel(n)	((n)-1)
#define	XkbShiftLevelMask(n)	(1<<((n)-1))

    /*
     * Extension name and version information
     */
#define	XkbName "XKEYBOARD"
#define	XkbMajorVersion	1
#define	XkbMinorVersion	0

    /*
     * Explicit map components:
     *  - Used in the 'explicit' field of an XkbServerMap.  Specifies
     *    the keyboard components that should _not_ be updated automatically
     *    in response to core protocol keyboard mapping requests.
     */
#define	XkbExplicitKeyTypesMask	  (0x0f)
#define	XkbExplicitKeyType1Mask	  (1<<0)
#define	XkbExplicitKeyType2Mask	  (1<<1)
#define	XkbExplicitKeyType3Mask	  (1<<2)
#define	XkbExplicitKeyType4Mask	  (1<<3)
#define	XkbExplicitInterpretMask  (1<<4)
#define	XkbExplicitAutoRepeatMask (1<<5)
#define	XkbExplicitBehaviorMask	  (1<<6)
#define	XkbExplicitVModMapMask	  (1<<7)
#define	XkbAllExplicitMask	  (0xff)

    /*
     * Map components masks:
     * Those in AllMapComponentsMask:
     *  - Specifies the individual fields to be loaded or changed for the
     *    GetMap and SetMap requests.
     * Those in ClientInfoMask:
     *  - Specifies the components to be allocated by XkbAllocClientMap.
     * Those in ServerInfoMask:
     *  - Specifies the components to be allocated by XkbAllocServerMap.
     */
#define	XkbKeyTypesMask		(1<<0)
#define	XkbKeySymsMask		(1<<1)
#define	XkbModifierMapMask	(1<<2)
#define	XkbExplicitComponentsMask (1<<3)
#define XkbKeyActionsMask	(1<<4)
#define	XkbKeyBehaviorsMask	(1<<5)
#define	XkbVirtualModsMask	(1<<6)
#define	XkbVirtualModMapMask	(1<<7)

#define	XkbAllClientInfoMask	(XkbKeyTypesMask|XkbKeySymsMask|XkbModifierMapMask)
#define	XkbAllServerInfoMask	(XkbExplicitComponentsMask|XkbKeyActionsMask|XkbKeyBehaviorsMask|XkbVirtualModsMask|XkbVirtualModMapMask)
#define	XkbAllMapComponentsMask	(XkbAllClientInfoMask|XkbAllServerInfoMask)

    /*
     * Symbol interpretations flags:
     *  - Used in the flags field of a symbol interpretation
     */
#define	XkbSI_AutoRepeat	(1<<0)
#define	XkbSI_LockingKey	(1<<1)

    /*
     * Symbol interpretations match specification:
     *  - Used in the match field of a symbol interpretation to specify
     *    the conditions under which an interpretation is used.
     */
#define	XkbSI_LevelOneOnly	(0x80)
#define	XkbSI_OpMask		(0x7f)
#define	XkbSI_NoneOf		(0)
#define	XkbSI_AnyOfOrNone	(1)
#define	XkbSI_AnyOf		(2)
#define	XkbSI_AllOf		(3)
#define	XkbSI_Exactly		(4)

    /*
     * Indicator map flags:
     *  - Used in the flags field of an indicator map to indicate the
     *    conditions under which and indicator can be changed and the
     *    effects of changing the indicator.
     */
#define	XkbIM_NoExplicit	(1L << 7)
#define	XkbIM_NoAutomatic	(1L << 6)
#define	XkbIM_LEDDrivesKB	(1L << 5)

    /*
     * Indicator map component specifications:
     *  - Used by the 'which_groups' and 'which_mods' fields of an indicator
     *    map to specify which keyboard components should be used to drive
     *    the indicator.
     */
#define	XkbIM_UseBase		(1L << 0)
#define	XkbIM_UseLatched	(1L << 1)
#define	XkbIM_UseLocked		(1L << 2)
#define	XkbIM_UseEffective	(1L << 3)
#define	XkbIM_UseCompat		(1L << 4)

#define	XkbIM_UseNone	  0
#define	XkbIM_UseAnyGroup (XkbIM_UseBase|XkbIM_UseLatched|XkbIM_UseLocked\
                           |XkbIM_UseEffective)
#define	XkbIM_UseAnyMods  (XkbIM_UseAnyGroup|XkbIM_UseCompat)

    /*
     * Compatibility Map Components:
     *  - Specifies the components to be allocated in XkbAllocCompatMap.
     */
#define	XkbSymInterpMask	(1<<0)
#define	XkbGroupCompatMask	(1<<1)
#define	XkbAllCompatMask	(0x3)

    /*
     * Names component mask:
     *  - Specifies the names to be loaded or changed for the GetNames and
     *    SetNames requests.
     *  - Specifies the names that have changed in a NamesNotify event.
     *  - Specifies the names components to be allocated by XkbAllocNames.
     */
#define	XkbKeycodesNameMask	(1<<0)
#define	XkbGeometryNameMask	(1<<1)
#define	XkbSymbolsNameMask	(1<<2)
#define	XkbPhysSymbolsNameMask	(1<<3)
#define	XkbTypesNameMask	(1<<4)
#define	XkbCompatNameMask 	(1<<5)
#define	XkbKeyTypeNamesMask	(1<<6)
#define	XkbKTLevelNamesMask	(1<<7)
#define	XkbIndicatorNamesMask	(1<<8)
#define	XkbKeyNamesMask		(1<<9)
#define	XkbKeyAliasesMask	(1<<10)
#define	XkbVirtualModNamesMask	(1<<11)
#define	XkbGroupNamesMask	(1<<12)
#define	XkbRGNamesMask		(1<<13)
#define	XkbComponentNamesMask	(0x3f)
#define	XkbAllNamesMask		(0x3fff)

    /*
     * GetByName components:
     *  - Specifies desired or necessary components to GetKbdByName request.
     *  - Reports the components that were found in a GetKbdByNameReply
     */
#define	XkbGBN_TypesMask		(1L << 0)
#define	XkbGBN_CompatMapMask		(1L << 1)
#define	XkbGBN_ClientSymbolsMask	(1L << 2)
#define	XkbGBN_ServerSymbolsMask	(1L << 3)
#define	XkbGBN_SymbolsMask (XkbGBN_ClientSymbolsMask|XkbGBN_ServerSymbolsMask)
#define	XkbGBN_IndicatorMapMask		(1L << 4)
#define	XkbGBN_KeyNamesMask		(1L << 5)
#define	XkbGBN_GeometryMask		(1L << 6)
#define	XkbGBN_OtherNamesMask		(1L << 7)
#define	XkbGBN_AllComponentsMask	(0xff)

     /*
      * ListComponents flags
      */
#define	XkbLC_Hidden			(1L <<  0)
#define	XkbLC_Default			(1L <<  1)
#define	XkbLC_Partial			(1L <<  2)

#define	XkbLC_AlphanumericKeys		(1L <<  8)
#define	XkbLC_ModifierKeys		(1L <<  9)
#define	XkbLC_KeypadKeys		(1L << 10)
#define	XkbLC_FunctionKeys		(1L << 11)
#define	XkbLC_AlternateGroup		(1L << 12)

    /*
     * X Input Extension Interactions
     * - Specifies the possible interactions between XKB and the X input
     *   extension
     * - Used to request (XkbGetDeviceInfo) or change (XKbSetDeviceInfo)
     *   XKB information about an extension device.
     * - Reports the list of supported optional features in the reply to
     *   XkbGetDeviceInfo or in an XkbExtensionDeviceNotify event.
     * XkbXI_UnsupportedFeature is reported in XkbExtensionDeviceNotify
     * events to indicate an attempt to use an unsupported feature.
     */
#define	XkbXI_KeyboardsMask		(1L << 0)
#define	XkbXI_ButtonActionsMask		(1L << 1)
#define	XkbXI_IndicatorNamesMask	(1L << 2)
#define	XkbXI_IndicatorMapsMask		(1L << 3)
#define	XkbXI_IndicatorStateMask	(1L << 4)
#define	XkbXI_UnsupportedFeatureMask	(1L << 15)
#define	XkbXI_AllFeaturesMask		(0x001f)
#define	XkbXI_AllDeviceFeaturesMask	(0x001e)

#define	XkbXI_IndicatorsMask		(0x001c)
#define	XkbAllExtensionDeviceEventsMask (0x801f)

    /*
     * Per-Client Flags:
     *  - Specifies flags to be changed by the PerClientFlags request.
     */
#define	XkbPCF_DetectableAutoRepeatMask	(1L << 0)
#define	XkbPCF_GrabsUseXKBStateMask	(1L << 1)
#define	XkbPCF_AutoResetControlsMask	(1L << 2)
#define	XkbPCF_LookupStateWhenGrabbed	(1L << 3)
#define	XkbPCF_SendEventUsesXKBState	(1L << 4)
#define	XkbPCF_AllFlagsMask		(0x1F)

    /*
     * Debugging flags and controls
     */
#define	XkbDF_DisableLocks	(1<<0)

#endif /* _XKB_H_ */
