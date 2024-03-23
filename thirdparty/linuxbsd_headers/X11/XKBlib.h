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

#ifndef _X11_XKBLIB_H_
#define _X11_XKBLIB_H_

#include <X11/Xlib.h>
#include <X11/extensions/XKBstr.h>

typedef struct _XkbAnyEvent {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* # of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XKB event minor code */
	unsigned int 	device;		/* device ID */
} XkbAnyEvent;

typedef struct _XkbNewKeyboardNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbNewKeyboardNotify */
	int	 	device;		/* device ID */
	int	 	old_device;	/* device ID of previous keyboard */
	int	 	min_key_code;	/* minimum key code */
	int		max_key_code;	/* maximum key code */
	int	 	old_min_key_code;/* min key code of previous kbd */
	int		old_max_key_code;/* max key code of previous kbd */
	unsigned int	changed;	/* changed aspects of the keyboard */
	char	 	req_major;	/* major and minor opcode of req */
	char	 	req_minor;	/* that caused change, if applicable */
} XkbNewKeyboardNotifyEvent;

typedef struct _XkbMapNotifyEvent {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbMapNotify */
	int 		device;		/* device ID */
	unsigned int 	changed;	/* fields which have been changed */
	unsigned int 	flags;		/* reserved */
	int 		first_type;	/* first changed key type */
	int 		num_types;	/* number of changed key types */
	KeyCode		min_key_code;
	KeyCode		max_key_code;
	KeyCode		first_key_sym;
	KeyCode		first_key_act;
	KeyCode		first_key_behavior;
	KeyCode		first_key_explicit;
	KeyCode		first_modmap_key;
	KeyCode		first_vmodmap_key;
	int		num_key_syms;
	int		num_key_acts;
	int		num_key_behaviors;
	int		num_key_explicit;
	int 		num_modmap_keys;
	int 		num_vmodmap_keys;
	unsigned int 	vmods;		/* mask of changed virtual mods */
} XkbMapNotifyEvent;

typedef struct _XkbStateNotifyEvent {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* # of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbStateNotify */
	int 		device;		/* device ID */
	unsigned int 	changed;	/* mask of changed state components */
	int 		group;		/* keyboard group */
	int 		base_group;	/* base keyboard group */
	int 		latched_group;	/* latched keyboard group */
	int 		locked_group;	/* locked keyboard group */
	unsigned int	mods;		/* modifier state */
	unsigned int 	base_mods;	/* base modifier state */
	unsigned int	latched_mods;	/* latched modifiers */
	unsigned int	locked_mods;	/* locked modifiers */
	int 		compat_state;	/* compatibility state */
	unsigned char	grab_mods;	/* mods used for grabs */
	unsigned char	compat_grab_mods;/* grab mods for non-XKB clients */
	unsigned char	lookup_mods;	/* mods sent to clients */
	unsigned char	compat_lookup_mods; /* mods sent to non-XKB clients */
	int 		ptr_buttons;	/* pointer button state */
	KeyCode		keycode;	/* keycode that caused the change */
	char 		event_type;	/* KeyPress or KeyRelease */
	char 		req_major;	/* Major opcode of request */
	char 		req_minor;	/* Minor opcode of request */
} XkbStateNotifyEvent;

typedef struct _XkbControlsNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbControlsNotify */
	int 		device;		/* device ID */
	unsigned int	changed_ctrls;	/* controls with changed sub-values */
	unsigned int 	enabled_ctrls;	/* controls currently enabled */
	unsigned int	enabled_ctrl_changes;/* controls just {en,dis}abled */
	int 		num_groups;	/* total groups on keyboard */
	KeyCode		keycode;	/* key that caused change or 0 */
	char 		event_type;	/* type of event that caused change */
	char 		req_major;	/* if keycode==0, major and minor */
	char 		req_minor;	/* opcode of req that caused change */
} XkbControlsNotifyEvent;

typedef struct _XkbIndicatorNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbIndicatorNotify */
	int 		device;		/* device ID */
	unsigned int	changed;	/* indicators with new state or map */
	unsigned int	state;	 	/* current state of all indicators */
} XkbIndicatorNotifyEvent;

typedef struct _XkbNamesNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbNamesNotify */
	int	 	device;		/* device ID */
	unsigned int 	changed;	/* names that have changed */
	int	 	first_type;	/* first key type with new name */
	int	 	num_types;	/* number of key types with new names */
	int	 	first_lvl;	/* first key type new new level names */
	int	 	num_lvls;	/* # of key types w/new level names */
	int	 	num_aliases;	/* total number of key aliases*/
	int	 	num_radio_groups;/* total number of radio groups */
	unsigned int 	changed_vmods;	/* virtual modifiers with new names */
	unsigned int 	changed_groups;	/* groups with new names */
	unsigned int 	changed_indicators;/* indicators with new names */
	int		first_key;	/* first key with new name */
	int		num_keys;	/* number of keys with new names */
} XkbNamesNotifyEvent;

typedef struct _XkbCompatMapNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbCompatMapNotify */
	int	 	device;		/* device ID */
	unsigned int 	changed_groups; /* groups with new compat maps */
	int	 	first_si;	/* first new symbol interp */
	int	 	num_si;		/* number of new symbol interps */
	int	 	num_total_si;	/* total # of symbol interps */
} XkbCompatMapNotifyEvent;

typedef struct _XkbBellNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbBellNotify */
	int	 	device;		/* device ID */
	int	 	percent;	/* requested volume as a % of maximum */
	int	 	pitch;		/* requested pitch in Hz */
	int	 	duration;	/* requested duration in useconds */
	int	 	bell_class;	/* (input extension) feedback class */
	int	 	bell_id;	/* (input extension) ID of feedback */
	Atom 		name;		/* "name" of requested bell */
	Window 		window;		/* window associated with event */
	Bool		event_only;	/* "event only" requested */
} XkbBellNotifyEvent;

typedef struct _XkbActionMessage {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbActionMessage */
	int	 	device;		/* device ID */
	KeyCode		keycode;	/* key that generated the event */
	Bool 		press;		/* true if act caused by key press */
	Bool 		key_event_follows;/* true if key event also generated */
	int		group;		/* effective group */
	unsigned int	mods;		/* effective mods */
	char 		message[XkbActionMessageLength+1];
					/* message -- leave space for NUL */
} XkbActionMessageEvent;

typedef struct _XkbAccessXNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbAccessXNotify */
	int	 	device;		/* device ID */
	int	 	detail;		/* XkbAXN_* */
	int	 	keycode;	/* key of event */
	int	 	sk_delay;	/* current slow keys delay */
	int		debounce_delay;	/* current debounce delay */
} XkbAccessXNotifyEvent;

typedef struct _XkbExtensionDeviceNotify {
	int 		type;		/* XkbAnyEvent */
	unsigned long 	serial;		/* of last req processed by server */
	Bool 		send_event;	/* is this from a SendEvent request? */
	Display *	display;	/* Display the event was read from */
	Time 		time;		/* milliseconds */
	int 		xkb_type;	/* XkbExtensionDeviceNotify */
	int	 	device;		/* device ID */
	unsigned int	reason;		/* reason for the event */
	unsigned int	supported;	/* mask of supported features */
	unsigned int	unsupported;	/* mask of unsupported features */
					/* that some app tried to use */
	int	 	first_btn;	/* first button that changed */
	int	 	num_btns;	/* range of buttons changed */
	unsigned int	leds_defined;   /* indicators with names or maps */
	unsigned int	led_state;	/* current state of the indicators */
	int		led_class;	/* feedback class for led changes */
	int		led_id;   	/* feedback id for led changes */
} XkbExtensionDeviceNotifyEvent;

typedef union _XkbEvent {
	int				type;
	XkbAnyEvent			any;
	XkbNewKeyboardNotifyEvent	new_kbd;
	XkbMapNotifyEvent		map;
	XkbStateNotifyEvent		state;
	XkbControlsNotifyEvent		ctrls;
	XkbIndicatorNotifyEvent 	indicators;
	XkbNamesNotifyEvent		names;
	XkbCompatMapNotifyEvent		compat;
	XkbBellNotifyEvent		bell;
	XkbActionMessageEvent		message;
	XkbAccessXNotifyEvent		accessx;
	XkbExtensionDeviceNotifyEvent 	device;
	XEvent				core;
} XkbEvent;

typedef struct	_XkbKbdDpyState	XkbKbdDpyStateRec,*XkbKbdDpyStatePtr;

	/* XkbOpenDisplay error codes */
#define	XkbOD_Success		0
#define	XkbOD_BadLibraryVersion	1
#define	XkbOD_ConnectionRefused	2
#define	XkbOD_NonXkbServer	3
#define	XkbOD_BadServerVersion	4

	/* Values for XlibFlags */
#define	XkbLC_ForceLatin1Lookup		(1<<0)
#define	XkbLC_ConsumeLookupMods		(1<<1)
#define	XkbLC_AlwaysConsumeShiftAndLock (1<<2)
#define	XkbLC_IgnoreNewKeyboards	(1<<3)
#define	XkbLC_ControlFallback		(1<<4)
#define	XkbLC_ConsumeKeysOnComposeFail	(1<<29)
#define	XkbLC_ComposeLED		(1<<30)
#define	XkbLC_BeepOnComposeFail		(1<<31)

#define	XkbLC_AllComposeControls	(0xc0000000)
#define	XkbLC_AllControls		(0xc000001f)

_XFUNCPROTOBEGIN

extern	Bool	XkbIgnoreExtension(
	Bool			/* ignore */
);

extern	Display *XkbOpenDisplay(
	char *			/* name */,
	int *			/* ev_rtrn */,
	int *			/* err_rtrn */,
	int *			/* major_rtrn */,
	int *			/* minor_rtrn */,
	int *			/* reason */
);

extern	Bool	XkbQueryExtension(
	Display *		/* dpy */,
	int *			/* opcodeReturn */,
	int *			/* eventBaseReturn */,
	int *			/* errorBaseReturn */,
	int *			/* majorRtrn */,
	int *			/* minorRtrn */
);

extern	Bool	XkbUseExtension(
	Display *		/* dpy */,
	int *			/* major_rtrn */,
	int *			/* minor_rtrn */
);

extern	Bool	XkbLibraryVersion(
	int *			/* libMajorRtrn */,
	int *			/* libMinorRtrn */
);

extern	unsigned int	XkbSetXlibControls(
	Display*		/* dpy */,
	unsigned int		/* affect */,
	unsigned int		/* values */
);

extern	unsigned int	XkbGetXlibControls(
	Display*		/* dpy */
);

extern	unsigned int	XkbXlibControlsImplemented(void);

typedef	Atom	(*XkbInternAtomFunc)(
	Display *		/* dpy */,
	_Xconst char *		/* name */,
	Bool			/* only_if_exists */
);

typedef char *	(*XkbGetAtomNameFunc)(
	Display *		/* dpy */,
	Atom			/* atom */
);

extern void		XkbSetAtomFuncs(
	XkbInternAtomFunc	/* getAtom */,
	XkbGetAtomNameFunc	/* getName */
);

extern	KeySym XkbKeycodeToKeysym(
		Display *	/* dpy */,
#if NeedWidePrototypes
		 unsigned int 	/* kc */,
#else
		 KeyCode 	/* kc */,
#endif
		 int 		/* group */,
		 int		/* level */
);

extern	unsigned int	XkbKeysymToModifiers(
    Display *			/* dpy */,
    KeySym 			/* ks */
);

extern	Bool		XkbLookupKeySym(
    Display *			/* dpy */,
    KeyCode 			/* keycode */,
    unsigned int 		/* modifiers */,
    unsigned int *		/* modifiers_return */,
    KeySym *			/* keysym_return */
);

extern	int		XkbLookupKeyBinding(
    Display *			/* dpy */,
    KeySym 			/* sym_rtrn */,
    unsigned int 		/* mods */,
    char *			/* buffer */,
    int 			/* nbytes */,
    int * 			/* extra_rtrn */
);

extern	Bool		XkbTranslateKeyCode(
    XkbDescPtr			/* xkb */,
    KeyCode 			/* keycode */,
    unsigned int 		/* modifiers */,
    unsigned int *		/* modifiers_return */,
    KeySym *			/* keysym_return */
);

extern	int		XkbTranslateKeySym(
    Display *			/* dpy */,
    KeySym *			/* sym_return */,
    unsigned int 		/* modifiers */,
    char *			/* buffer */,
    int 			/* nbytes */,
    int *			/* extra_rtrn */
);

extern	Bool	XkbSetAutoRepeatRate(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* delay */,
	unsigned int		/* interval */
);

extern	Bool	XkbGetAutoRepeatRate(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int *		/* delayRtrn */,
	unsigned int *		/* intervalRtrn */
);

extern	Bool	XkbChangeEnabledControls(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* affect */,
	unsigned int		/* values */
);

extern	Bool	XkbDeviceBell(
	Display *		/* dpy */,
	Window			/* win */,
	int			/* deviceSpec */,
	int			/* bellClass */,
	int			/* bellID */,
	int			/* percent */,
	Atom			/* name */
);

extern	Bool	XkbForceDeviceBell(
	Display *		/* dpy */,
	int			/* deviceSpec */,
	int			/* bellClass */,
	int			/* bellID */,
	int			/* percent */
);

extern	Bool	XkbDeviceBellEvent(
	Display *		/* dpy */,
	Window			/* win */,
	int			/* deviceSpec */,
	int			/* bellClass */,
	int			/* bellID */,
	int			/* percent */,
	Atom			/* name */
);

extern	Bool	XkbBell(
	Display *		/* dpy */,
	Window			/* win */,
	int			/* percent */,
	Atom			/* name */
);

extern	Bool	XkbForceBell(
	Display *		/* dpy */,
	int			/* percent */
);

extern	Bool	XkbBellEvent(
	Display *		/* dpy */,
	Window			/* win */,
	int			/* percent */,
	Atom			/* name */
);

extern	Bool	XkbSelectEvents(
	Display *		/* dpy */,
	unsigned int		/* deviceID */,
	unsigned int 		/* affect */,
	unsigned int 		/* values */
);

extern	Bool	XkbSelectEventDetails(
	Display *		/* dpy */,
	unsigned int 		/* deviceID */,
	unsigned int 		/* eventType */,
	unsigned long 		/* affect */,
	unsigned long 		/* details */
);

extern	void	XkbNoteMapChanges(
    XkbMapChangesPtr		/* old */,
    XkbMapNotifyEvent	*	/* new */,
    unsigned int	 	/* wanted */
);

extern	void	XkbNoteNameChanges(
    XkbNameChangesPtr		/* old */,
    XkbNamesNotifyEvent	*	/* new */,
    unsigned int	 	/* wanted */
);

extern	Status	XkbGetIndicatorState(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int *		/* pStateRtrn */
);

extern	Status	XkbGetDeviceIndicatorState(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* ledClass */,
	unsigned int		/* ledID */,
	unsigned int *		/* pStateRtrn */
);

extern	Status	 XkbGetIndicatorMap(
	Display *		/* dpy */,
	unsigned long		/* which */,
	XkbDescPtr		/* desc */
);

extern	Bool	 XkbSetIndicatorMap(
	Display *		/* dpy */,
	unsigned long 		/* which */,
	XkbDescPtr		/* desc */
);

#define	XkbNoteIndicatorMapChanges(o,n,w) \
				((o)->map_changes|=((n)->map_changes&(w)))
#define	XkbNoteIndicatorStateChanges(o,n,w)\
				((o)->state_changes|=((n)->state_changes&(w)))
#define	XkbGetIndicatorMapChanges(d,x,c) \
				(XkbGetIndicatorMap((d),(c)->map_changes,x))
#define	XkbChangeIndicatorMaps(d,x,c) \
				(XkbSetIndicatorMap((d),(c)->map_changes,x))

extern	Bool	XkbGetNamedIndicator(
	Display *		/* dpy */,
	Atom			/* name */,
	int *			/* pNdxRtrn */,
	Bool *			/* pStateRtrn */,
	XkbIndicatorMapPtr	/* pMapRtrn */,
	Bool *			/* pRealRtrn */
);

extern	Bool	XkbGetNamedDeviceIndicator(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* ledClass */,
	unsigned int		/* ledID */,
	Atom			/* name */,
	int *			/* pNdxRtrn */,
	Bool *			/* pStateRtrn */,
	XkbIndicatorMapPtr	/* pMapRtrn */,
	Bool *			/* pRealRtrn */
);

extern	Bool	XkbSetNamedIndicator(
	Display *		/* dpy */,
	Atom			/* name */,
	Bool			/* changeState */,
	Bool 			/* state */,
	Bool			/* createNewMap */,
	XkbIndicatorMapPtr	/* pMap */
);

extern	Bool	XkbSetNamedDeviceIndicator(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* ledClass */,
	unsigned int		/* ledID */,
	Atom			/* name */,
	Bool			/* changeState */,
	Bool 			/* state */,
	Bool			/* createNewMap */,
	XkbIndicatorMapPtr	/* pMap */
);

extern	Bool	XkbLockModifiers(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* affect */,
	unsigned int 		/* values */
);

extern	Bool	XkbLatchModifiers(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* affect */,
	unsigned int 		/* values */
);

extern	Bool	XkbLockGroup(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* group */
);

extern	Bool	XkbLatchGroup(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* group */
);

extern	Bool	XkbSetServerInternalMods(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* affectReal */,
	unsigned int 		/* realValues */,
	unsigned int		/* affectVirtual */,
	unsigned int		/* virtualValues */
);

extern	Bool	XkbSetIgnoreLockMods(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	unsigned int 		/* affectReal */,
	unsigned int 		/* realValues */,
	unsigned int		/* affectVirtual */,
	unsigned int		/* virtualValues */
);


extern	Bool	XkbVirtualModsToReal(
	XkbDescPtr		/* xkb */,
	unsigned int		/* virtual_mask */,
	unsigned int *		/* mask_rtrn */
);

extern	Bool	XkbComputeEffectiveMap(
	XkbDescPtr 		/* xkb */,
	XkbKeyTypePtr		/* type */,
	unsigned char *		/* map_rtrn */
);

extern	Status XkbInitCanonicalKeyTypes(
    XkbDescPtr			/* xkb */,
    unsigned int		/* which */,
    int				/* keypadVMod */
);

extern	XkbDescPtr XkbAllocKeyboard(
	void
);

extern	void	XkbFreeKeyboard(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	Bool			/* freeDesc */
);

extern	Status XkbAllocClientMap(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	unsigned int		/* nTypes */
);

extern	Status XkbAllocServerMap(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	unsigned int		/* nActions */
);

extern	void	XkbFreeClientMap(
    XkbDescPtr			/* xkb */,
    unsigned int		/* what */,
    Bool			/* freeMap */
);

extern	void	XkbFreeServerMap(
    XkbDescPtr			/* xkb */,
    unsigned int		/* what */,
    Bool			/* freeMap */
);

extern	XkbKeyTypePtr	XkbAddKeyType(
    XkbDescPtr			/* xkb */,
    Atom			/* name */,
    int				/* map_count */,
    Bool			/* want_preserve */,
    int				/* num_lvls */
);

extern	Status XkbAllocIndicatorMaps(
	XkbDescPtr		/* xkb */
);

extern	void XkbFreeIndicatorMaps(
    XkbDescPtr			/* xkb */
);

extern	XkbDescPtr XkbGetMap(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	unsigned int 		/* deviceSpec */
);

extern	Status	XkbGetUpdatedMap(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	XkbDescPtr		/* desc */
);

extern	Status	XkbGetMapChanges(
    Display *			/* dpy */,
    XkbDescPtr			/* xkb */,
    XkbMapChangesPtr		/* changes */
);


extern	Status	XkbRefreshKeyboardMapping(
    XkbMapNotifyEvent *		/* event */
);

extern	Status	XkbGetKeyTypes(
    Display *			/* dpy */,
    unsigned int		/* first */,
    unsigned int 		/* num */,
    XkbDescPtr			/* xkb */
);

extern	Status	XkbGetKeySyms(
    Display *			/* dpy */,
    unsigned int		/* first */,
    unsigned int		/* num */,
    XkbDescPtr			/* xkb */
);

extern	Status	XkbGetKeyActions(
    Display *			/* dpy */,
    unsigned int 		/* first */,
    unsigned int 		/* num */,
    XkbDescPtr			/* xkb */
);

extern	Status	XkbGetKeyBehaviors(
	Display *		/* dpy */,
	unsigned int 		/* firstKey */,
	unsigned int		/* nKeys */,
	XkbDescPtr		/* desc */
);

extern	Status	XkbGetVirtualMods(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	XkbDescPtr		/* desc */
);

extern	Status	XkbGetKeyExplicitComponents(
	Display *		/* dpy */,
	unsigned int 		/* firstKey */,
	unsigned int		/* nKeys */,
	XkbDescPtr		/* desc */
);

extern	Status	XkbGetKeyModifierMap(
	Display *		/* dpy */,
	unsigned int 		/* firstKey */,
	unsigned int		/* nKeys */,
	XkbDescPtr		/* desc */
);

extern	Status	XkbGetKeyVirtualModMap(
	Display *		/* dpy */,
	unsigned int		/* first */,
	unsigned int		/* num */,
	XkbDescPtr		/* xkb */
);

extern	Status	XkbAllocControls(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which*/
);

extern	void	XkbFreeControls(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	Bool			/* freeMap */
);

extern	Status	XkbGetControls(
	Display *		/* dpy */,
	unsigned long		/* which */,
	XkbDescPtr		/* desc */
);

extern	Bool	XkbSetControls(
	Display *		/* dpy */,
	unsigned long		/* which */,
	XkbDescPtr		/* desc */
);

extern	void	XkbNoteControlsChanges(
    XkbControlsChangesPtr	/* old */,
    XkbControlsNotifyEvent *	/* new */,
    unsigned int	 	/* wanted */
);

#define	XkbGetControlsChanges(d,x,c)	XkbGetControls(d,(c)->changed_ctrls,x)
#define	XkbChangeControls(d,x,c)	XkbSetControls(d,(c)->changed_ctrls,x)

extern	Status	XkbAllocCompatMap(
    XkbDescPtr			/* xkb */,
    unsigned int		/* which */,
    unsigned int		/* nInterpret */
);

extern	void	XkbFreeCompatMap(
    XkbDescPtr			/* xkb */,
    unsigned int		/* which */,
    Bool			/* freeMap */
);

extern Status XkbGetCompatMap(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	XkbDescPtr 		/* xkb */
);

extern Bool XkbSetCompatMap(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	XkbDescPtr 		/* xkb */,
	Bool			/* updateActions */
);

extern	XkbSymInterpretPtr XkbAddSymInterpret(
	XkbDescPtr		/* xkb */,
	XkbSymInterpretPtr	/* si */,
	Bool			/* updateMap */,
	XkbChangesPtr		/* changes */
);

extern	Status XkbAllocNames(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	int			/* nTotalRG */,
	int			/* nTotalAliases */
);

extern	Status	XkbGetNames(
	Display *		/* dpy */,
	unsigned int		/* which */,
	XkbDescPtr		/* desc */
);

extern	Bool	XkbSetNames(
	Display *		/* dpy */,
	unsigned int		/* which */,
	unsigned int		/* firstType */,
	unsigned int		/* nTypes */,
	XkbDescPtr		/* desc */
);

extern	Bool	XkbChangeNames(
	Display *		/* dpy */,
	XkbDescPtr		/* xkb */,
	XkbNameChangesPtr	/* changes */
);

extern	void XkbFreeNames(
	XkbDescPtr		/* xkb */,
	unsigned int		/* which */,
	Bool			/* freeMap */
);


extern	Status	XkbGetState(
	Display *		/* dpy */,
	unsigned int 		/* deviceSpec */,
	XkbStatePtr		/* rtrnState */
);

extern	Bool	XkbSetMap(
	Display *		/* dpy */,
	unsigned int		/* which */,
	XkbDescPtr		/* desc */
);

extern	Bool	XkbChangeMap(
	Display*		/* dpy */,
	XkbDescPtr		/* desc */,
	XkbMapChangesPtr	/* changes */
);

extern	Bool	XkbSetDetectableAutoRepeat(
	Display *		/* dpy */,
	Bool			/* detectable */,
	Bool *			/* supported */
);

extern	Bool	XkbGetDetectableAutoRepeat(
	Display *		/* dpy */,
	Bool *			/* supported */
);

extern	Bool	XkbSetAutoResetControls(
    Display *			/* dpy */,
    unsigned int 		/* changes */,
    unsigned int *		/* auto_ctrls */,
    unsigned int *		/* auto_values */
);

extern	Bool	XkbGetAutoResetControls(
    Display *			/* dpy */,
    unsigned int *		/* auto_ctrls */,
    unsigned int *		/* auto_ctrl_values */
);

extern	Bool	XkbSetPerClientControls(
    Display *			/* dpy */,
    unsigned int		/* change */,
    unsigned int *		/* values */
);

extern	Bool	XkbGetPerClientControls(
    Display *			/* dpy */,
    unsigned int *		/* ctrls */
);

extern Status XkbCopyKeyType(
    XkbKeyTypePtr	/* from */,
    XkbKeyTypePtr	/* into */
);

extern Status XkbCopyKeyTypes(
    XkbKeyTypePtr	/* from */,
    XkbKeyTypePtr	/* into */,
    int			/* num_types */
);

extern	Status	XkbResizeKeyType(
    XkbDescPtr		/* xkb */,
    int			/* type_ndx */,
    int			/* map_count */,
    Bool		/* want_preserve */,
    int			/* new_num_lvls */
);

extern	KeySym *XkbResizeKeySyms(
	XkbDescPtr		/* desc */,
	int 			/* forKey */,
	int 			/* symsNeeded */
);

extern	XkbAction *XkbResizeKeyActions(
	XkbDescPtr		/* desc */,
	int 			/* forKey */,
	int 			/* actsNeeded */
);

extern	Status XkbChangeTypesOfKey(
	XkbDescPtr		/* xkb */,
	int 			/* key */,
	int			/* num_groups */,
	unsigned int		/* groups */,
	int *			/* newTypes */,
	XkbMapChangesPtr	/* pChanges */
);

extern  Status   XkbChangeKeycodeRange(
	XkbDescPtr		/* xkb */,
	int			/* minKC */,
	int			/* maxKC */,
	XkbChangesPtr		/* changes */
);

/***====================================================================***/

extern	XkbComponentListPtr	XkbListComponents(
	Display *		/* dpy */,
	unsigned int		/* deviceSpec */,
	XkbComponentNamesPtr	/* ptrns */,
	int *			/* max_inout */
);

extern	void XkbFreeComponentList(
	XkbComponentListPtr	/* list */
);

extern	XkbDescPtr XkbGetKeyboard(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	unsigned int 		/* deviceSpec */
);

extern XkbDescPtr XkbGetKeyboardByName(
    Display *			/* dpy */,
    unsigned int		/* deviceSpec */,
    XkbComponentNamesPtr	/* names */,
    unsigned int 		/* want */,
    unsigned int 		/* need */,
    Bool			/* load */
);

/***====================================================================***/

extern	int	XkbKeyTypesForCoreSymbols(	/* returns # of groups */
    XkbDescPtr	/* xkb */,			/* keyboard device */
    int		/* map_width */,		/* width of core KeySym array */
    KeySym *	/* core_syms */,		/* always mapWidth symbols */
    unsigned int	/* protected */,	/* explicit key types */
    int *	/* types_inout */,		/* always four type indices */
    KeySym * 	/* xkb_syms_rtrn */		/* must have enough space */
);

extern	Bool	XkbApplyCompatMapToKey(	/* False only on error */
    XkbDescPtr		/* xkb */,		/* keymap to be edited */
    KeyCode		/* key */,		/* key to be updated */
    XkbChangesPtr	/* changes */		/* resulting changes to map */
);

extern	Bool	XkbUpdateMapFromCore( /* False only on error */
    XkbDescPtr		/* xkb */,		/* XKB keyboard to be edited */
    KeyCode		/* first_key */,	/* first changed key */
    int			/* num_keys */, 	/* number of changed keys */
    int			/* map_width */,	/* width of core keymap */
    KeySym *		/* core_keysyms */,	/* symbols from core keymap */
    XkbChangesPtr	/* changes */		/* resulting changes */
);

/***====================================================================***/

extern	XkbDeviceLedInfoPtr	XkbAddDeviceLedInfo(
	XkbDeviceInfoPtr	/* devi */,
	unsigned int		/* ledClass */,
	unsigned int		/* ledId */
);

extern	Status			XkbResizeDeviceButtonActions(
	XkbDeviceInfoPtr	/* devi */,
	unsigned int		/* newTotal */
);

extern	XkbDeviceInfoPtr	XkbAllocDeviceInfo(
	unsigned int		/* deviceSpec */,
	unsigned int		/* nButtons */,
	unsigned int		/* szLeds */
);

extern	void XkbFreeDeviceInfo(
	XkbDeviceInfoPtr	/* devi */,
	unsigned int		/* which */,
	Bool			/* freeDevI */
);

extern	void	XkbNoteDeviceChanges(
    XkbDeviceChangesPtr			/* old */,
    XkbExtensionDeviceNotifyEvent *	/* new */,
    unsigned int	 		/* wanted */
);

extern	XkbDeviceInfoPtr XkbGetDeviceInfo(
	Display *		/* dpy */,
	unsigned int 		/* which */,
	unsigned int		/* deviceSpec */,
	unsigned int		/* ledClass */,
	unsigned int		/* ledID */
);

extern	Status	XkbGetDeviceInfoChanges(
	Display *		/* dpy */,
	XkbDeviceInfoPtr	/* devi */,
	XkbDeviceChangesPtr 	/* changes */
);

extern	Status	XkbGetDeviceButtonActions(
	Display *		/* dpy */,
	XkbDeviceInfoPtr	/* devi */,
	Bool			/* all */,
	unsigned int		/* first */,
	unsigned int		/* nBtns */
);

extern	Status	XkbGetDeviceLedInfo(
	Display *		/* dpy */,
	XkbDeviceInfoPtr	/* devi */,
	unsigned int		/* ledClass (class, XIDflt, XIAll) */,
	unsigned int		/* ledId (id, XIDflt, XIAll) */,
	unsigned int		/* which (XkbXI_Indicator{Names,Map}Mask */
);

extern	Bool	XkbSetDeviceInfo(
	Display *		/* dpy */,
	unsigned int		/* which */,
	XkbDeviceInfoPtr	/* devi */
);

extern	Bool	XkbChangeDeviceInfo(
	Display*		/* dpy */,
	XkbDeviceInfoPtr	/* desc */,
	XkbDeviceChangesPtr	/* changes */
);

extern  Bool XkbSetDeviceLedInfo(
	Display *		/* dpy */,
	XkbDeviceInfoPtr	/* devi */,
	unsigned int 		/* ledClass */,
	unsigned int		/* ledID */,
	unsigned int		/* which */
);

extern	Bool XkbSetDeviceButtonActions(
	Display *		/* dpy */,
	XkbDeviceInfoPtr	/* devi */,
	unsigned int		/* first */,
	unsigned int		/* nBtns */
);

/***====================================================================***/

extern	char	XkbToControl(
	char		/* c */
);

/***====================================================================***/

extern	Bool XkbSetDebuggingFlags(
    Display *		/* dpy */,
    unsigned int	/* mask */,
    unsigned int	/* flags */,
    char *		/* msg */,
    unsigned int	/* ctrls_mask */,
    unsigned int	/* ctrls */,
    unsigned int *	/* rtrn_flags */,
    unsigned int *	/* rtrn_ctrls */
);

extern	Bool XkbApplyVirtualModChanges(
   XkbDescPtr		/* xkb */,
   unsigned int		/* changed */,
   XkbChangesPtr	/* changes */
);

extern Bool XkbUpdateActionVirtualMods(
	XkbDescPtr		/* xkb */,
	XkbAction *		/* act */,
	unsigned int		/* changed */
);

extern void XkbUpdateKeyTypeVirtualMods(
	XkbDescPtr		/* xkb */,
	XkbKeyTypePtr		/* type */,
	unsigned int		/* changed */,
	XkbChangesPtr		/* changes */
);

_XFUNCPROTOEND

#endif /* _X11_XKBLIB_H_ */
