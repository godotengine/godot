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

#ifndef _XKBSTR_H_
#define	_XKBSTR_H_

#include <X11/Xfuncproto.h>
#include <X11/extensions/XKB.h>

#define	XkbCharToInt(v)		((v)&0x80?(int)((v)|(~0xff)):(int)((v)&0x7f))
#define	XkbIntTo2Chars(i,h,l)	(((h)=((i>>8)&0xff)),((l)=((i)&0xff)))
#define	Xkb2CharsToInt(h,l)	((short)(((h)<<8)|(l)))

/*
 * The Xkb structs are full of implicit padding to properly align members.
 * We can't clean that up without breaking ABI, so tell clang not to bother
 * complaining about it.
 */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

	/*
	 * Common data structures and access macros
	 */

typedef struct _XkbStateRec {
	unsigned char	group;
	unsigned char   locked_group;
	unsigned short	base_group;
	unsigned short	latched_group;
	unsigned char	mods;
	unsigned char	base_mods;
	unsigned char	latched_mods;
	unsigned char	locked_mods;
	unsigned char	compat_state;
	unsigned char	grab_mods;
	unsigned char	compat_grab_mods;
	unsigned char	lookup_mods;
	unsigned char	compat_lookup_mods;
	unsigned short	ptr_buttons;
} XkbStateRec,*XkbStatePtr;
#define	XkbModLocks(s)	 ((s)->locked_mods)
#define	XkbStateMods(s)	 ((s)->base_mods|(s)->latched_mods|XkbModLocks(s))
#define	XkbGroupLock(s)	 ((s)->locked_group)
#define	XkbStateGroup(s) ((s)->base_group+(s)->latched_group+XkbGroupLock(s))
#define	XkbStateFieldFromRec(s)	XkbBuildCoreState((s)->lookup_mods,(s)->group)
#define	XkbGrabStateFromRec(s)	XkbBuildCoreState((s)->grab_mods,(s)->group)

typedef struct _XkbMods {
	unsigned char	mask;	/* effective mods */
	unsigned char	real_mods;
	unsigned short	vmods;
} XkbModsRec,*XkbModsPtr;

typedef struct _XkbKTMapEntry {
	Bool		active;
	unsigned char	level;
	XkbModsRec	mods;
} XkbKTMapEntryRec,*XkbKTMapEntryPtr;

typedef struct _XkbKeyType {
	XkbModsRec		mods;
	unsigned char	  	num_levels;
	unsigned char	  	map_count;
	/* map is an array of map_count XkbKTMapEntryRec structs */
	XkbKTMapEntryPtr  	map;
	/* preserve is an array of map_count XkbModsRec structs */
	XkbModsPtr  		preserve;
	Atom		  	name;
	/* level_names is an array of num_levels Atoms */
	Atom *			level_names;
} XkbKeyTypeRec, *XkbKeyTypePtr;

#define	XkbNumGroups(g)			((g)&0x0f)
#define	XkbOutOfRangeGroupInfo(g)	((g)&0xf0)
#define	XkbOutOfRangeGroupAction(g)	((g)&0xc0)
#define	XkbOutOfRangeGroupNumber(g)	(((g)&0x30)>>4)
#define	XkbSetGroupInfo(g,w,n)	(((w)&0xc0)|(((n)&3)<<4)|((g)&0x0f))
#define	XkbSetNumGroups(g,n)	(((g)&0xf0)|((n)&0x0f))

	/*
	 * Structures and access macros used primarily by the server
	 */

typedef struct _XkbBehavior {
	unsigned char	type;
	unsigned char	data;
} XkbBehavior;

#define	XkbAnyActionDataSize 7
typedef	struct _XkbAnyAction {
	unsigned char	type;
	unsigned char	data[XkbAnyActionDataSize];
} XkbAnyAction;

typedef struct _XkbModAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	mask;
	unsigned char	real_mods;
	unsigned char	vmods1;
	unsigned char	vmods2;
} XkbModAction;
#define	XkbModActionVMods(a)      \
	((short)(((a)->vmods1<<8)|((a)->vmods2)))
#define	XkbSetModActionVMods(a,v) \
	(((a)->vmods1=(((v)>>8)&0xff)),(a)->vmods2=((v)&0xff))

typedef struct _XkbGroupAction {
	unsigned char	type;
	unsigned char	flags;
	char		group_XXX;
} XkbGroupAction;
#define	XkbSAGroup(a)		(XkbCharToInt((a)->group_XXX))
#define	XkbSASetGroup(a,g)	((a)->group_XXX=(g))

typedef struct _XkbISOAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	mask;
	unsigned char	real_mods;
	char		group_XXX;
	unsigned char	affect;
	unsigned char	vmods1;
	unsigned char	vmods2;
} XkbISOAction;

typedef struct _XkbPtrAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	high_XXX;
	unsigned char	low_XXX;
	unsigned char	high_YYY;
	unsigned char	low_YYY;
} XkbPtrAction;
#define	XkbPtrActionX(a)      (Xkb2CharsToInt((a)->high_XXX,(a)->low_XXX))
#define	XkbPtrActionY(a)      (Xkb2CharsToInt((a)->high_YYY,(a)->low_YYY))
#define	XkbSetPtrActionX(a,x) (XkbIntTo2Chars(x,(a)->high_XXX,(a)->low_XXX))
#define	XkbSetPtrActionY(a,y) (XkbIntTo2Chars(y,(a)->high_YYY,(a)->low_YYY))

typedef struct _XkbPtrBtnAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	count;
	unsigned char	button;
} XkbPtrBtnAction;

typedef struct _XkbPtrDfltAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	affect;
	char		valueXXX;
} XkbPtrDfltAction;
#define	XkbSAPtrDfltValue(a)		(XkbCharToInt((a)->valueXXX))
#define	XkbSASetPtrDfltValue(a,c)	((a)->valueXXX= ((c)&0xff))

typedef struct _XkbSwitchScreenAction {
	unsigned char	type;
	unsigned char	flags;
	char		screenXXX;
} XkbSwitchScreenAction;
#define	XkbSAScreen(a)			(XkbCharToInt((a)->screenXXX))
#define	XkbSASetScreen(a,s)		((a)->screenXXX= ((s)&0xff))

typedef struct _XkbCtrlsAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	ctrls3;
	unsigned char	ctrls2;
	unsigned char	ctrls1;
	unsigned char	ctrls0;
} XkbCtrlsAction;
#define	XkbActionSetCtrls(a,c)	(((a)->ctrls3=(((c)>>24)&0xff)),\
					((a)->ctrls2=(((c)>>16)&0xff)),\
					((a)->ctrls1=(((c)>>8)&0xff)),\
					((a)->ctrls0=((c)&0xff)))
#define	XkbActionCtrls(a) ((((unsigned int)(a)->ctrls3)<<24)|\
			   (((unsigned int)(a)->ctrls2)<<16)|\
			   (((unsigned int)(a)->ctrls1)<<8)|\
			   ((unsigned int)((a)->ctrls0)))

typedef struct _XkbMessageAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	message[6];
} XkbMessageAction;

typedef struct	_XkbRedirectKeyAction {
	unsigned char	type;
	unsigned char	new_key;
	unsigned char	mods_mask;
	unsigned char	mods;
	unsigned char	vmods_mask0;
	unsigned char	vmods_mask1;
	unsigned char	vmods0;
	unsigned char	vmods1;
} XkbRedirectKeyAction;

#define	XkbSARedirectVMods(a)		((((unsigned int)(a)->vmods1)<<8)|\
					((unsigned int)(a)->vmods0))
#define	XkbSARedirectSetVMods(a,m)	(((a)->vmods1=(((m)>>8)&0xff)),\
					 ((a)->vmods0=((m)&0xff)))
#define	XkbSARedirectVModsMask(a)	((((unsigned int)(a)->vmods_mask1)<<8)|\
					((unsigned int)(a)->vmods_mask0))
#define	XkbSARedirectSetVModsMask(a,m)	(((a)->vmods_mask1=(((m)>>8)&0xff)),\
					 ((a)->vmods_mask0=((m)&0xff)))

typedef struct _XkbDeviceBtnAction {
	unsigned char	type;
	unsigned char	flags;
	unsigned char	count;
	unsigned char	button;
	unsigned char	device;
} XkbDeviceBtnAction;

typedef struct _XkbDeviceValuatorAction {
	unsigned char	type;
	unsigned char	device;
	unsigned char	v1_what;
	unsigned char	v1_ndx;
	unsigned char	v1_value;
	unsigned char	v2_what;
	unsigned char	v2_ndx;
	unsigned char	v2_value;
} XkbDeviceValuatorAction;

typedef	union _XkbAction {
	XkbAnyAction		any;
	XkbModAction		mods;
	XkbGroupAction		group;
	XkbISOAction		iso;
	XkbPtrAction		ptr;
	XkbPtrBtnAction		btn;
	XkbPtrDfltAction	dflt;
	XkbSwitchScreenAction	screen;
	XkbCtrlsAction		ctrls;
	XkbMessageAction	msg;
	XkbRedirectKeyAction	redirect;
	XkbDeviceBtnAction	devbtn;
	XkbDeviceValuatorAction	devval;
	unsigned char 		type;
} XkbAction;

typedef	struct _XkbControls {
	unsigned char	mk_dflt_btn;
	unsigned char	num_groups;
	unsigned char	groups_wrap;
	XkbModsRec	internal;
	XkbModsRec	ignore_lock;
	unsigned int	enabled_ctrls;
	unsigned short	repeat_delay;
	unsigned short	repeat_interval;
	unsigned short	slow_keys_delay;
	unsigned short	debounce_delay;
	unsigned short	mk_delay;
	unsigned short	mk_interval;
	unsigned short	mk_time_to_max;
	unsigned short	mk_max_speed;
		 short	mk_curve;
	unsigned short	ax_options;
	unsigned short	ax_timeout;
	unsigned short	axt_opts_mask;
	unsigned short	axt_opts_values;
	unsigned int	axt_ctrls_mask;
	unsigned int	axt_ctrls_values;
	unsigned char	per_key_repeat[XkbPerKeyBitArraySize];
} XkbControlsRec, *XkbControlsPtr;

#define	XkbAX_AnyFeedback(c)	((c)->enabled_ctrls&XkbAccessXFeedbackMask)
#define	XkbAX_NeedOption(c,w)	((c)->ax_options&(w))
#define	XkbAX_NeedFeedback(c,w)	(XkbAX_AnyFeedback(c)&&XkbAX_NeedOption(c,w))

typedef struct _XkbServerMapRec {
	/* acts is an array of XkbActions structs, with size_acts entries
	   allocated, and num_acts entries used. */
	unsigned short		 num_acts;
	unsigned short		 size_acts;
	XkbAction		*acts;

	/* behaviors, key_acts, explicit, & vmodmap are all arrays with
	   (xkb->max_key_code + 1) entries allocated for each. */
	XkbBehavior		*behaviors;
	unsigned short		*key_acts;
#if defined(__cplusplus) || defined(c_plusplus)
	/* explicit is a C++ reserved word */
	unsigned char		*c_explicit;
#else
	unsigned char		*explicit;
#endif
	unsigned char		 vmods[XkbNumVirtualMods];
	unsigned short		*vmodmap;
} XkbServerMapRec, *XkbServerMapPtr;

#define	XkbSMKeyActionsPtr(m,k) (&(m)->acts[(m)->key_acts[k]])

	/*
	 * Structures and access macros used primarily by clients
	 */

typedef	struct _XkbSymMapRec {
	unsigned char	 kt_index[XkbNumKbdGroups];
	unsigned char	 group_info;
	unsigned char	 width;
	unsigned short	 offset;
} XkbSymMapRec, *XkbSymMapPtr;

typedef struct _XkbClientMapRec {
	/* types is an array of XkbKeyTypeRec structs, with size_types entries
	   allocated, and num_types entries used. */
	unsigned char		 size_types;
	unsigned char		 num_types;
	XkbKeyTypePtr		 types;

	/* syms is an array of size_syms KeySyms, in which num_syms are used */
	unsigned short		 size_syms;
	unsigned short		 num_syms;
	KeySym			*syms;
	/* key_sym_map is an array of (max_key_code + 1) XkbSymMapRec structs */
	XkbSymMapPtr		 key_sym_map;

	/* modmap is an array of (max_key_code + 1) unsigned chars */
	unsigned char		*modmap;
} XkbClientMapRec, *XkbClientMapPtr;

#define	XkbCMKeyGroupInfo(m,k)  ((m)->key_sym_map[k].group_info)
#define	XkbCMKeyNumGroups(m,k)	 (XkbNumGroups((m)->key_sym_map[k].group_info))
#define	XkbCMKeyGroupWidth(m,k,g) (XkbCMKeyType(m,k,g)->num_levels)
#define	XkbCMKeyGroupsWidth(m,k) ((m)->key_sym_map[k].width)
#define	XkbCMKeyTypeIndex(m,k,g) ((m)->key_sym_map[k].kt_index[g&0x3])
#define	XkbCMKeyType(m,k,g)	 (&(m)->types[XkbCMKeyTypeIndex(m,k,g)])
#define	XkbCMKeyNumSyms(m,k) (XkbCMKeyGroupsWidth(m,k)*XkbCMKeyNumGroups(m,k))
#define	XkbCMKeySymsOffset(m,k)	((m)->key_sym_map[k].offset)
#define	XkbCMKeySymsPtr(m,k)	(&(m)->syms[XkbCMKeySymsOffset(m,k)])

	/*
	 * Compatibility structures and access macros
	 */

typedef struct _XkbSymInterpretRec {
	KeySym		sym;
	unsigned char	flags;
	unsigned char	match;
	unsigned char	mods;
	unsigned char	virtual_mod;
	XkbAnyAction	act;
} XkbSymInterpretRec,*XkbSymInterpretPtr;

typedef struct _XkbCompatMapRec {
	/* sym_interpret is an array of XkbSymInterpretRec structs,
	   in which size_si are allocated & num_si are used. */
	XkbSymInterpretPtr	 sym_interpret;
	XkbModsRec		 groups[XkbNumKbdGroups];
	unsigned short		 num_si;
	unsigned short		 size_si;
} XkbCompatMapRec, *XkbCompatMapPtr;

typedef struct _XkbIndicatorMapRec {
	unsigned char	flags;
	unsigned char	which_groups;
	unsigned char	groups;
	unsigned char	which_mods;
	XkbModsRec	mods;
	unsigned int	ctrls;
} XkbIndicatorMapRec, *XkbIndicatorMapPtr;

#define	XkbIM_IsAuto(i)	((((i)->flags&XkbIM_NoAutomatic)==0)&&\
			    (((i)->which_groups&&(i)->groups)||\
			     ((i)->which_mods&&(i)->mods.mask)||\
			     ((i)->ctrls)))
#define	XkbIM_InUse(i)	(((i)->flags)||((i)->which_groups)||\
					((i)->which_mods)||((i)->ctrls))


typedef struct _XkbIndicatorRec {
	unsigned long	  	phys_indicators;
	XkbIndicatorMapRec	maps[XkbNumIndicators];
} XkbIndicatorRec,*XkbIndicatorPtr;

typedef	struct _XkbKeyNameRec {
	char	name[XkbKeyNameLength]	_X_NONSTRING;
} XkbKeyNameRec,*XkbKeyNamePtr;

typedef struct _XkbKeyAliasRec {
	char	real[XkbKeyNameLength]	_X_NONSTRING;
	char	alias[XkbKeyNameLength]	_X_NONSTRING;
} XkbKeyAliasRec,*XkbKeyAliasPtr;

	/*
	 * Names for everything
	 */
typedef struct _XkbNamesRec {
	Atom		  keycodes;
	Atom		  geometry;
	Atom		  symbols;
	Atom              types;
	Atom		  compat;
	Atom		  vmods[XkbNumVirtualMods];
	Atom		  indicators[XkbNumIndicators];
	Atom		  groups[XkbNumKbdGroups];
	/* keys is an array of (xkb->max_key_code + 1) XkbKeyNameRec entries */
	XkbKeyNamePtr	  keys;
	/* key_aliases is an array of num_key_aliases XkbKeyAliasRec entries */
	XkbKeyAliasPtr	  key_aliases;
	/* radio_groups is an array of num_rg Atoms */
	Atom		 *radio_groups;
	Atom		  phys_symbols;

	/* num_keys seems to be unused in libX11 */
	unsigned char	  num_keys;
	unsigned char	  num_key_aliases;
	unsigned short	  num_rg;
} XkbNamesRec,*XkbNamesPtr;

typedef	struct _XkbGeometry	*XkbGeometryPtr;
	/*
	 * Tie it all together into one big keyboard description
	 */
typedef	struct _XkbDesc {
	struct _XDisplay *	dpy;
	unsigned short	 	flags;
	unsigned short		device_spec;
	KeyCode			min_key_code;
	KeyCode			max_key_code;

	XkbControlsPtr		ctrls;
	XkbServerMapPtr		server;
	XkbClientMapPtr		map;
	XkbIndicatorPtr		indicators;
	XkbNamesPtr		names;
	XkbCompatMapPtr		compat;
	XkbGeometryPtr		geom;
} XkbDescRec, *XkbDescPtr;
#define	XkbKeyKeyTypeIndex(d,k,g)	(XkbCMKeyTypeIndex((d)->map,k,g))
#define	XkbKeyKeyType(d,k,g)		(XkbCMKeyType((d)->map,k,g))
#define	XkbKeyGroupWidth(d,k,g)		(XkbCMKeyGroupWidth((d)->map,k,g))
#define	XkbKeyGroupsWidth(d,k)		(XkbCMKeyGroupsWidth((d)->map,k))
#define	XkbKeyGroupInfo(d,k)		(XkbCMKeyGroupInfo((d)->map,(k)))
#define	XkbKeyNumGroups(d,k)		(XkbCMKeyNumGroups((d)->map,(k)))
#define	XkbKeyNumSyms(d,k)		(XkbCMKeyNumSyms((d)->map,(k)))
#define	XkbKeySymsPtr(d,k)		(XkbCMKeySymsPtr((d)->map,(k)))
#define	XkbKeySym(d,k,n)		(XkbKeySymsPtr(d,k)[n])
#define	XkbKeySymEntry(d,k,sl,g) \
	(XkbKeySym(d,k,((XkbKeyGroupsWidth(d,k)*(g))+(sl))))
#define	XkbKeyAction(d,k,n) \
	(XkbKeyHasActions(d,k)?&XkbKeyActionsPtr(d,k)[n]:NULL)
#define	XkbKeyActionEntry(d,k,sl,g) \
	(XkbKeyHasActions(d,k)?\
		XkbKeyAction(d,k,((XkbKeyGroupsWidth(d,k)*(g))+(sl))):NULL)

#define	XkbKeyHasActions(d,k)	((d)->server->key_acts[k]!=0)
#define	XkbKeyNumActions(d,k)	(XkbKeyHasActions(d,k)?XkbKeyNumSyms(d,k):1)
#define	XkbKeyActionsPtr(d,k)	(XkbSMKeyActionsPtr((d)->server,k))
#define	XkbKeycodeInRange(d,k)	(((k)>=(d)->min_key_code)&&\
				 ((k)<=(d)->max_key_code))
#define	XkbNumKeys(d)		((d)->max_key_code-(d)->min_key_code+1)


	/*
	 * The following structures can be used to track changes
	 * to a keyboard device
	 */
typedef struct _XkbMapChanges {
	unsigned short		 changed;
	KeyCode			 min_key_code;
	KeyCode			 max_key_code;
	unsigned char		 first_type;
	unsigned char		 num_types;
	KeyCode			 first_key_sym;
	unsigned char		 num_key_syms;
	KeyCode			 first_key_act;
	unsigned char		 num_key_acts;
	KeyCode			 first_key_behavior;
	unsigned char		 num_key_behaviors;
	KeyCode 		 first_key_explicit;
	unsigned char		 num_key_explicit;
	KeyCode			 first_modmap_key;
	unsigned char		 num_modmap_keys;
	KeyCode			 first_vmodmap_key;
	unsigned char		 num_vmodmap_keys;
	unsigned char		 pad;
	unsigned short		 vmods;
} XkbMapChangesRec,*XkbMapChangesPtr;

typedef struct _XkbControlsChanges {
	unsigned int 		 changed_ctrls;
	unsigned int		 enabled_ctrls_changes;
	Bool			 num_groups_changed;
} XkbControlsChangesRec,*XkbControlsChangesPtr;

typedef struct _XkbIndicatorChanges {
	unsigned int		 state_changes;
	unsigned int		 map_changes;
} XkbIndicatorChangesRec,*XkbIndicatorChangesPtr;

typedef struct _XkbNameChanges {
	unsigned int 		changed;
	unsigned char		first_type;
	unsigned char		num_types;
	unsigned char		first_lvl;
	unsigned char		num_lvls;
	unsigned char		num_aliases;
	unsigned char		num_rg;
	unsigned char		first_key;
	unsigned char		num_keys;
	unsigned short		changed_vmods;
	unsigned long		changed_indicators;
	unsigned char		changed_groups;
} XkbNameChangesRec,*XkbNameChangesPtr;

typedef struct _XkbCompatChanges {
	unsigned char		changed_groups;
	unsigned short		first_si;
	unsigned short		num_si;
} XkbCompatChangesRec,*XkbCompatChangesPtr;

typedef struct _XkbChanges {
	unsigned short		 device_spec;
	unsigned short		 state_changes;
	XkbMapChangesRec	 map;
	XkbControlsChangesRec	 ctrls;
	XkbIndicatorChangesRec	 indicators;
	XkbNameChangesRec	 names;
	XkbCompatChangesRec	 compat;
} XkbChangesRec, *XkbChangesPtr;

	/*
	 * These data structures are used to construct a keymap from
	 * a set of components or to list components in the server
	 * database.
	 */
typedef struct _XkbComponentNames {
	char *			 keymap;
	char *			 keycodes;
	char *			 types;
	char *			 compat;
	char *			 symbols;
	char *			 geometry;
} XkbComponentNamesRec, *XkbComponentNamesPtr;

typedef struct _XkbComponentName {
	unsigned short		flags;
	char *			name;
} XkbComponentNameRec,*XkbComponentNamePtr;

typedef struct _XkbComponentList {
	int			num_keymaps;
	int			num_keycodes;
	int			num_types;
	int			num_compat;
	int			num_symbols;
	int			num_geometry;
	XkbComponentNamePtr	keymaps;
	XkbComponentNamePtr 	keycodes;
	XkbComponentNamePtr	types;
	XkbComponentNamePtr	compat;
	XkbComponentNamePtr	symbols;
	XkbComponentNamePtr	geometry;
} XkbComponentListRec, *XkbComponentListPtr;

	/*
	 * The following data structures describe and track changes to a
	 * non-keyboard extension device
	 */
typedef struct _XkbDeviceLedInfo {
	unsigned short			led_class;
	unsigned short			led_id;
	unsigned int			phys_indicators;
	unsigned int			maps_present;
	unsigned int			names_present;
	unsigned int			state;
	Atom 				names[XkbNumIndicators];
	XkbIndicatorMapRec		maps[XkbNumIndicators];
} XkbDeviceLedInfoRec,*XkbDeviceLedInfoPtr;

typedef struct _XkbDeviceInfo {
	char *			name;
	Atom			type;
	unsigned short		device_spec;
	Bool			has_own_state;
	unsigned short		supported;
	unsigned short		unsupported;

	/* btn_acts is an array of num_btn XkbAction entries */
	unsigned short		num_btns;
	XkbAction *		btn_acts;

	unsigned short		sz_leds;
	unsigned short		num_leds;
	unsigned short		dflt_kbd_fb;
	unsigned short		dflt_led_fb;
	/* leds is an array of XkbDeviceLedInfoRec in which
	   sz_leds entries are allocated and num_leds entries are used */
	XkbDeviceLedInfoPtr	leds;
} XkbDeviceInfoRec,*XkbDeviceInfoPtr;

#define	XkbXI_DevHasBtnActs(d)	(((d)->num_btns>0)&&((d)->btn_acts!=NULL))
#define	XkbXI_LegalDevBtn(d,b)	(XkbXI_DevHasBtnActs(d)&&((b)<(d)->num_btns))
#define	XkbXI_DevHasLeds(d)	(((d)->num_leds>0)&&((d)->leds!=NULL))

typedef struct _XkbDeviceLedChanges {
	unsigned short		led_class;
	unsigned short		led_id;
	unsigned int		defined; /* names or maps changed */
	struct _XkbDeviceLedChanges *next;
} XkbDeviceLedChangesRec,*XkbDeviceLedChangesPtr;

typedef struct _XkbDeviceChanges {
	unsigned int		changed;
	unsigned short		first_btn;
	unsigned short		num_btns;
	XkbDeviceLedChangesRec 	leds;
} XkbDeviceChangesRec,*XkbDeviceChangesPtr;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif /* _XKBSTR_H_ */
