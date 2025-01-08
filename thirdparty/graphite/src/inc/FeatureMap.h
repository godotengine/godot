// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2010, SIL International, All rights reserved.

#pragma once
#include "inc/Main.h"
#include "inc/FeatureVal.h"

namespace graphite2 {

// Forward declarations for implmentation types
class FeatureMap;
class Face;


class FeatureSetting
{
public:
    FeatureSetting(int16 theValue, uint16 labelId) : m_label(labelId), m_value(theValue) {};
    uint16 label() const { return m_label; }
    int16 value() const { return m_value; }

    CLASS_NEW_DELETE;
private:
    FeatureSetting(const FeatureSetting & fs) : m_label(fs.m_label), m_value(fs.m_value) {};

    uint16 m_label;
    int16 m_value;
};

class FeatureRef
{
    typedef uint32      chunk_t;
    static const uint8  SIZEOF_CHUNK = sizeof(chunk_t)*8;

public:
    enum flags_t : uint16 {
        HIDDEN = 0x0800
    };
    FeatureRef() throw();
    FeatureRef(const Face & face, unsigned short & bits_offset, uint32 max_val,
               uint32 name, uint16 uiName, flags_t flags,
               FeatureSetting *settings, uint16 num_set) throw();
    ~FeatureRef() throw();

    bool applyValToFeature(uint32 val, Features& pDest) const; //defined in GrFaceImp.h
    void maskFeature(Features & pDest) const {
    if (m_index < pDest.size())                 //defensive
        pDest[m_index] |= m_mask;
    }

    uint32 getFeatureVal(const Features& feats) const; //defined in GrFaceImp.h

    uint32 getId() const { return m_id; }
    uint16 getNameId() const { return m_nameid; }
    uint16 getNumSettings() const { return m_numSet; }
    uint16 getSettingName(uint16 index) const { return m_nameValues[index].label(); }
    int16  getSettingValue(uint16 index) const { return m_nameValues[index].value(); }
    flags_t getFlags() const { return m_flags; }
    uint32 maxVal() const { return m_max; }
    const Face & getFace() const { assert(m_face); return *m_face;}
    const FeatureMap* getFeatureMap() const;// { return m_pFace;}

    CLASS_NEW_DELETE;
private:
    FeatureRef(const FeatureRef & rhs);

    const Face     * m_face;
    FeatureSetting * m_nameValues; // array of name table ids for feature values
    chunk_t m_mask,             // bit mask to get the value from the vector
            m_max;              // max value the value can take
    uint32  m_id;               // feature identifier/name
    uint16  m_nameid,           // Name table id for feature name
            m_numSet;           // number of values (number of entries in m_nameValues)
    flags_t m_flags;            // feature flags see FeatureRef::flags_t.
    byte    m_bits,             // how many bits to shift the value into place
            m_index;            // index into the array to find the ulong to mask

private:        //unimplemented
    FeatureRef& operator=(const FeatureRef&);
};

inline
FeatureRef::FeatureRef() throw()
: m_face(0),
  m_nameValues(0),
  m_mask(0), m_max(0),
  m_id(0), m_nameid(0), m_numSet(0), 
  m_flags(flags_t(0)),
  m_bits(0), m_index(0)
{
}


class NameAndFeatureRef
{
  public:
    NameAndFeatureRef(uint32 name = 0) : m_name(name) , m_pFRef(NULL){}
    NameAndFeatureRef(FeatureRef const & p) : m_name(p.getId()), m_pFRef(&p) {}

    bool operator<(const NameAndFeatureRef& rhs) const //orders by m_name
        {   return m_name<rhs.m_name; }

    CLASS_NEW_DELETE

    uint32 m_name;
    const FeatureRef* m_pFRef;
};

class FeatureMap
{
public:
    FeatureMap() : m_numFeats(0), m_feats(NULL), m_pNamedFeats(NULL) {}
    ~FeatureMap() { delete[] m_feats; delete[] m_pNamedFeats; }

    bool readFeats(const Face & face);
    const FeatureRef *findFeatureRef(uint32 name) const;
    FeatureRef *feature(uint16 index) const { return m_feats + index; }
    //GrFeatureRef *featureRef(byte index) { return index < m_numFeats ? m_feats + index : NULL; }
    const FeatureRef *featureRef(byte index) const { return index < m_numFeats ? m_feats + index : NULL; }
    FeatureVal* cloneFeatures(uint32 langname/*0 means default*/) const;      //call destroy_Features when done.
    uint16 numFeats() const { return m_numFeats; };
    CLASS_NEW_DELETE
private:
friend class SillMap;
    uint16 m_numFeats;

    FeatureRef *m_feats;
    NameAndFeatureRef* m_pNamedFeats;   //owned
    FeatureVal m_defaultFeatures;        //owned

private:        //defensive on m_feats, m_pNamedFeats, and m_defaultFeatures
    FeatureMap(const FeatureMap&);
    FeatureMap& operator=(const FeatureMap&);
};


class SillMap
{
private:
    class LangFeaturePair
    {
        LangFeaturePair(const LangFeaturePair &);
        LangFeaturePair & operator = (const LangFeaturePair &);

    public:
        LangFeaturePair() :  m_lang(0), m_pFeatures(0) {}
        ~LangFeaturePair() { delete m_pFeatures; }

        uint32 m_lang;
        Features* m_pFeatures;      //owns
        CLASS_NEW_DELETE
    };
public:
    SillMap() : m_langFeats(NULL), m_numLanguages(0) {}
    ~SillMap() { delete[] m_langFeats; }
    bool readFace(const Face & face);
    bool readSill(const Face & face);
    FeatureVal* cloneFeatures(uint32 langname/*0 means default*/) const;      //call destroy_Features when done.
    uint16 numLanguages() const { return m_numLanguages; };
    uint32 getLangName(uint16 index) const { return (index < m_numLanguages)? m_langFeats[index].m_lang : 0; };

    const FeatureMap & theFeatureMap() const { return m_FeatureMap; };
private:
    FeatureMap m_FeatureMap;        //of face
    LangFeaturePair * m_langFeats;
    uint16 m_numLanguages;

private:        //defensive on m_langFeats
    SillMap(const SillMap&);
    SillMap& operator=(const SillMap&);
};

} // namespace graphite2

struct gr_feature_ref : public graphite2::FeatureRef {};
