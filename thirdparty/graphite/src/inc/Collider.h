// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once

#include "inc/List.h"
#include "inc/Position.h"
#include "inc/Intervals.h"
#include "inc/debug.h"

namespace graphite2 {

class json;
class Slot;
class Segment;

#define SLOTCOLSETUINTPROP(x, y) uint16 x() const { return _ ##x; } void y (uint16 v) { _ ##x = v; }
#define SLOTCOLSETINTPROP(x, y) int16 x() const { return _ ##x; } void y (int16 v) { _ ##x = v; }
#define SLOTCOLSETPOSITIONPROP(x, y) const Position &x() const { return _ ##x; } void y (const Position &v) { _ ##x = v; }

// Slot attributes related to collision-fixing
class SlotCollision
{
public:
    enum {
    //  COLL_TESTONLY = 0,  // default - test other glyphs for collision with this one, but don't move this one
        COLL_FIX = 1,       // fix collisions involving this glyph
        COLL_IGNORE = 2,    // ignore this glyph altogether
        COLL_START = 4,     // start of range of possible collisions
        COLL_END = 8,       // end of range of possible collisions
        COLL_KERN = 16,     // collisions with this glyph are fixed by adding kerning space after it
        COLL_ISCOL = 32,    // this glyph has a collision
        COLL_KNOWN = 64,    // we've figured out what's happening with this glyph
        COLL_ISSPACE = 128,		// treat this glyph as a space with regard to kerning
        COLL_TEMPLOCK = 256,    // Lock glyphs that have been given priority positioning
        ////COLL_JUMPABLE = 128,    // moving glyphs may jump this stationary glyph in any direction - DELETE
        ////COLL_OVERLAP = 256,    // use maxoverlap to restrict - DELETE
    };

    // Behavior for the collision.order attribute. To GDL this is an enum, to us it's a bitfield, with only 1 bit set
    // Allows for easier inversion.
    enum {
        SEQ_ORDER_LEFTDOWN = 1,
        SEQ_ORDER_RIGHTUP = 2,
        SEQ_ORDER_NOABOVE = 4,
        SEQ_ORDER_NOBELOW = 8,
        SEQ_ORDER_NOLEFT = 16,
        SEQ_ORDER_NORIGHT = 32
    };

    SlotCollision(Segment *seg, Slot *slot);
    void initFromSlot(Segment *seg, Slot *slot);

    const Rect &limit() const { return _limit; }
    void setLimit(const Rect &r) { _limit = r; }
    SLOTCOLSETPOSITIONPROP(shift, setShift)
    SLOTCOLSETPOSITIONPROP(offset, setOffset)
    SLOTCOLSETPOSITIONPROP(exclOffset, setExclOffset)
    SLOTCOLSETUINTPROP(margin, setMargin)
    SLOTCOLSETUINTPROP(marginWt, setMarginWt)
    SLOTCOLSETUINTPROP(flags, setFlags)
    SLOTCOLSETUINTPROP(exclGlyph, setExclGlyph)
    SLOTCOLSETUINTPROP(seqClass, setSeqClass)
    SLOTCOLSETUINTPROP(seqProxClass, setSeqProxClass)
    SLOTCOLSETUINTPROP(seqOrder, setSeqOrder)
    SLOTCOLSETINTPROP(seqAboveXoff, setSeqAboveXoff)
    SLOTCOLSETUINTPROP(seqAboveWt, setSeqAboveWt)
    SLOTCOLSETINTPROP(seqBelowXlim, setSeqBelowXlim)
    SLOTCOLSETUINTPROP(seqBelowWt, setSeqBelowWt)
    SLOTCOLSETUINTPROP(seqValignHt, setSeqValignHt)
    SLOTCOLSETUINTPROP(seqValignWt, setSeqValignWt)

    float getKern(int dir) const;
    bool ignore() const;

private:
    Rect        _limit;
    Position    _shift;     // adjustment within the given pass
    Position    _offset;    // total adjustment for collisions
    Position    _exclOffset;
    uint16		_margin;
    uint16		_marginWt;
    uint16		_flags;
    uint16		_exclGlyph;
    uint16		_seqClass;
	uint16		_seqProxClass;
    uint16		_seqOrder;
    int16		_seqAboveXoff;
    uint16		_seqAboveWt;
    int16		_seqBelowXlim;
    uint16		_seqBelowWt;
    uint16		_seqValignHt;
    uint16		_seqValignWt;

};  // end of class SlotColllision

struct BBox;
struct SlantBox;

class ShiftCollider
{
public:
    typedef std::pair<float, float> fpair;
    typedef Vector<fpair> vfpairs;
    typedef vfpairs::iterator ivfpairs;

    ShiftCollider(json *dbgout);
    ~ShiftCollider() throw() { };

    bool initSlot(Segment *seg, Slot *aSlot, const Rect &constraint,
                float margin, float marginMin, const Position &currShift,
                const Position &currOffset, int dir, GR_MAYBE_UNUSED json * const dbgout);
    bool mergeSlot(Segment *seg, Slot *slot, const SlotCollision *cinfo, const Position &currShift, bool isAfter,
                bool sameCluster, bool &hasCol, bool isExclusion, GR_MAYBE_UNUSED json * const dbgout);
    Position resolve(Segment *seg, bool &isCol, GR_MAYBE_UNUSED json * const dbgout);
    void addBox_slope(bool isx, const Rect &box, const BBox &bb, const SlantBox &sb, const Position &org, float weight, float m, bool minright, int mode);
    void removeBox(const Rect &box, const BBox &bb, const SlantBox &sb, const Position &org, int mode);
    const Position &origin() const { return _origin; }

#if !defined GRAPHITE2_NTRACING
	void outputJsonDbg(json * const dbgout, Segment *seg, int axis);
	void outputJsonDbgStartSlot(json * const dbgout, Segment *seg);
	void outputJsonDbgEndSlot(json * const dbgout, Position resultPos, int bestAxis, bool isCol);
	void outputJsonDbgOneVector(json * const dbgout, Segment *seg, int axis, float tleft, float bestCost, float bestVal);
	void outputJsonDbgRawRanges(json * const dbgout, int axis);
	void outputJsonDbgRemovals(json * const dbgout, int axis, Segment *seg);
#endif

    CLASS_NEW_DELETE;

protected:
    Zones _ranges[4];   // possible movements in 4 directions (horizontally, vertically, diagonally);
    Slot *  _target;    // the glyph to fix
    Rect    _limit;
    Position _currShift;
    Position _currOffset;
    Position _origin;   // Base for all relative calculations
    float   _margin;
	float	_marginWt;
    float   _len[4];
    uint16  _seqClass;
	uint16	_seqProxClass;
    uint16  _seqOrder;

	//bool _scraping[4];

};	// end of class ShiftCollider

inline
ShiftCollider::ShiftCollider(GR_MAYBE_UNUSED json *dbgout)
: _target(0),
  _margin(0.0),
  _marginWt(0.0),
  _seqClass(0),
  _seqProxClass(0),
  _seqOrder(0)
{
#if !defined GRAPHITE2_NTRACING
    for (int i = 0; i < 4; ++i)
        _ranges[i].setdebug(dbgout);
#endif
}

class KernCollider
{
public:
    KernCollider(json *dbg);
    ~KernCollider() throw() { };
    bool initSlot(Segment *seg, Slot *aSlot, const Rect &constraint, float margin,
            const Position &currShift, const Position &offsetPrev, int dir,
            float ymin, float ymax, json * const dbgout);
    bool mergeSlot(Segment *seg, Slot *slot, const Position &currShift, float currSpace, int dir, json * const dbgout);
    Position resolve(Segment *seg, Slot *slot, int dir, json * const dbgout);
    void shift(const Position &mv, int dir);

    CLASS_NEW_DELETE;

private:
    Slot *  _target;        // the glyph to fix
    Rect    _limit;
    float   _margin;
    Position _offsetPrev;   // kern from a previous pass
    Position _currShift;    // NOT USED??
    float _miny;	        // y-coordinates offset by global slot position
    float _maxy;
    Vector<float> _edges;   // edges of horizontal slices
    float _sliceWidth;      // width of each slice
    float _mingap;
    float _xbound;        // max or min edge
    bool  _hit;

#if !defined GRAPHITE2_NTRACING
    // Debugging
    Segment * _seg;
    Vector<float> _nearEdges; // closest potential collision in each slice
    Vector<Slot*> _slotNear;
#endif
};	// end of class KernCollider


inline
float sqr(float x) {
    return x * x;
}

inline
KernCollider::KernCollider(GR_MAYBE_UNUSED json *dbg)
: _target(0),
  _margin(0.0f),
  _miny(-1e38f),
  _maxy(1e38f),
  _sliceWidth(0.0f),
  _mingap(0.0f),
  _xbound(0.0),
  _hit(false)
{
#if !defined GRAPHITE2_NTRACING
    _seg = 0;
#endif
};

};  // end of namespace graphite2
