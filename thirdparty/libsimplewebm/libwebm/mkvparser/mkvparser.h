// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef MKVPARSER_MKVPARSER_H_
#define MKVPARSER_MKVPARSER_H_

#include <cstddef>

namespace mkvparser {

const int E_PARSE_FAILED = -1;
const int E_FILE_FORMAT_INVALID = -2;
const int E_BUFFER_NOT_FULL = -3;

class IMkvReader {
 public:
  virtual int Read(long long pos, long len, unsigned char* buf) = 0;
  virtual int Length(long long* total, long long* available) = 0;

 public:
  virtual ~IMkvReader();
};

template <typename Type>
Type* SafeArrayAlloc(unsigned long long num_elements,
                     unsigned long long element_size);
long long GetUIntLength(IMkvReader*, long long, long&);
long long ReadUInt(IMkvReader*, long long, long&);
long long ReadID(IMkvReader* pReader, long long pos, long& len);
long long UnserializeUInt(IMkvReader*, long long pos, long long size);

long UnserializeFloat(IMkvReader*, long long pos, long long size, double&);
long UnserializeInt(IMkvReader*, long long pos, long long size,
                    long long& result);

long UnserializeString(IMkvReader*, long long pos, long long size, char*& str);

long ParseElementHeader(IMkvReader* pReader,
                        long long& pos,  // consume id and size fields
                        long long stop,  // if you know size of element's parent
                        long long& id, long long& size);

bool Match(IMkvReader*, long long&, unsigned long, long long&);
bool Match(IMkvReader*, long long&, unsigned long, unsigned char*&, size_t&);

void GetVersion(int& major, int& minor, int& build, int& revision);

struct EBMLHeader {
  EBMLHeader();
  ~EBMLHeader();
  long long m_version;
  long long m_readVersion;
  long long m_maxIdLength;
  long long m_maxSizeLength;
  char* m_docType;
  long long m_docTypeVersion;
  long long m_docTypeReadVersion;

  long long Parse(IMkvReader*, long long&);
  void Init();
};

class Segment;
class Track;
class Cluster;

class Block {
  Block(const Block&);
  Block& operator=(const Block&);

 public:
  const long long m_start;
  const long long m_size;

  Block(long long start, long long size, long long discard_padding);
  ~Block();

  long Parse(const Cluster*);

  long long GetTrackNumber() const;
  long long GetTimeCode(const Cluster*) const;  // absolute, but not scaled
  long long GetTime(const Cluster*) const;  // absolute, and scaled (ns)
  bool IsKey() const;
  void SetKey(bool);
  bool IsInvisible() const;

  enum Lacing { kLacingNone, kLacingXiph, kLacingFixed, kLacingEbml };
  Lacing GetLacing() const;

  int GetFrameCount() const;  // to index frames: [0, count)

  struct Frame {
    long long pos;  // absolute offset
    long len;

    long Read(IMkvReader*, unsigned char*) const;
  };

  const Frame& GetFrame(int frame_index) const;

  long long GetDiscardPadding() const;

 private:
  long long m_track;  // Track::Number()
  short m_timecode;  // relative to cluster
  unsigned char m_flags;

  Frame* m_frames;
  int m_frame_count;

 protected:
  const long long m_discard_padding;
};

class BlockEntry {
  BlockEntry(const BlockEntry&);
  BlockEntry& operator=(const BlockEntry&);

 protected:
  BlockEntry(Cluster*, long index);

 public:
  virtual ~BlockEntry();

  bool EOS() const { return (GetKind() == kBlockEOS); }
  const Cluster* GetCluster() const;
  long GetIndex() const;
  virtual const Block* GetBlock() const = 0;

  enum Kind { kBlockEOS, kBlockSimple, kBlockGroup };
  virtual Kind GetKind() const = 0;

 protected:
  Cluster* const m_pCluster;
  const long m_index;
};

class SimpleBlock : public BlockEntry {
  SimpleBlock(const SimpleBlock&);
  SimpleBlock& operator=(const SimpleBlock&);

 public:
  SimpleBlock(Cluster*, long index, long long start, long long size);
  long Parse();

  Kind GetKind() const;
  const Block* GetBlock() const;

 protected:
  Block m_block;
};

class BlockGroup : public BlockEntry {
  BlockGroup(const BlockGroup&);
  BlockGroup& operator=(const BlockGroup&);

 public:
  BlockGroup(Cluster*, long index,
             long long block_start,  // absolute pos of block's payload
             long long block_size,  // size of block's payload
             long long prev, long long next, long long duration,
             long long discard_padding);

  long Parse();

  Kind GetKind() const;
  const Block* GetBlock() const;

  long long GetPrevTimeCode() const;  // relative to block's time
  long long GetNextTimeCode() const;  // as above
  long long GetDurationTimeCode() const;

 private:
  Block m_block;
  const long long m_prev;
  const long long m_next;
  const long long m_duration;
};

///////////////////////////////////////////////////////////////
// ContentEncoding element
// Elements used to describe if the track data has been encrypted or
// compressed with zlib or header stripping.
class ContentEncoding {
 public:
  enum { kCTR = 1 };

  ContentEncoding();
  ~ContentEncoding();

  // ContentCompression element names
  struct ContentCompression {
    ContentCompression();
    ~ContentCompression();

    unsigned long long algo;
    unsigned char* settings;
    long long settings_len;
  };

  // ContentEncAESSettings element names
  struct ContentEncAESSettings {
    ContentEncAESSettings() : cipher_mode(kCTR) {}
    ~ContentEncAESSettings() {}

    unsigned long long cipher_mode;
  };

  // ContentEncryption element names
  struct ContentEncryption {
    ContentEncryption();
    ~ContentEncryption();

    unsigned long long algo;
    unsigned char* key_id;
    long long key_id_len;
    unsigned char* signature;
    long long signature_len;
    unsigned char* sig_key_id;
    long long sig_key_id_len;
    unsigned long long sig_algo;
    unsigned long long sig_hash_algo;

    ContentEncAESSettings aes_settings;
  };

  // Returns ContentCompression represented by |idx|. Returns NULL if |idx|
  // is out of bounds.
  const ContentCompression* GetCompressionByIndex(unsigned long idx) const;

  // Returns number of ContentCompression elements in this ContentEncoding
  // element.
  unsigned long GetCompressionCount() const;

  // Parses the ContentCompression element from |pReader|. |start| is the
  // starting offset of the ContentCompression payload. |size| is the size in
  // bytes of the ContentCompression payload. |compression| is where the parsed
  // values will be stored.
  long ParseCompressionEntry(long long start, long long size,
                             IMkvReader* pReader,
                             ContentCompression* compression);

  // Returns ContentEncryption represented by |idx|. Returns NULL if |idx|
  // is out of bounds.
  const ContentEncryption* GetEncryptionByIndex(unsigned long idx) const;

  // Returns number of ContentEncryption elements in this ContentEncoding
  // element.
  unsigned long GetEncryptionCount() const;

  // Parses the ContentEncAESSettings element from |pReader|. |start| is the
  // starting offset of the ContentEncAESSettings payload. |size| is the
  // size in bytes of the ContentEncAESSettings payload. |encryption| is
  // where the parsed values will be stored.
  long ParseContentEncAESSettingsEntry(long long start, long long size,
                                       IMkvReader* pReader,
                                       ContentEncAESSettings* aes);

  // Parses the ContentEncoding element from |pReader|. |start| is the
  // starting offset of the ContentEncoding payload. |size| is the size in
  // bytes of the ContentEncoding payload. Returns true on success.
  long ParseContentEncodingEntry(long long start, long long size,
                                 IMkvReader* pReader);

  // Parses the ContentEncryption element from |pReader|. |start| is the
  // starting offset of the ContentEncryption payload. |size| is the size in
  // bytes of the ContentEncryption payload. |encryption| is where the parsed
  // values will be stored.
  long ParseEncryptionEntry(long long start, long long size,
                            IMkvReader* pReader, ContentEncryption* encryption);

  unsigned long long encoding_order() const { return encoding_order_; }
  unsigned long long encoding_scope() const { return encoding_scope_; }
  unsigned long long encoding_type() const { return encoding_type_; }

 private:
  // Member variables for list of ContentCompression elements.
  ContentCompression** compression_entries_;
  ContentCompression** compression_entries_end_;

  // Member variables for list of ContentEncryption elements.
  ContentEncryption** encryption_entries_;
  ContentEncryption** encryption_entries_end_;

  // ContentEncoding element names
  unsigned long long encoding_order_;
  unsigned long long encoding_scope_;
  unsigned long long encoding_type_;

  // LIBWEBM_DISALLOW_COPY_AND_ASSIGN(ContentEncoding);
  ContentEncoding(const ContentEncoding&);
  ContentEncoding& operator=(const ContentEncoding&);
};

class Track {
  Track(const Track&);
  Track& operator=(const Track&);

 public:
  class Info;
  static long Create(Segment*, const Info&, long long element_start,
                     long long element_size, Track*&);

  enum Type { kVideo = 1, kAudio = 2, kSubtitle = 0x11, kMetadata = 0x21 };

  Segment* const m_pSegment;
  const long long m_element_start;
  const long long m_element_size;
  virtual ~Track();

  long GetType() const;
  long GetNumber() const;
  unsigned long long GetUid() const;
  const char* GetNameAsUTF8() const;
  const char* GetLanguage() const;
  const char* GetCodecNameAsUTF8() const;
  const char* GetCodecId() const;
  const unsigned char* GetCodecPrivate(size_t&) const;
  bool GetLacing() const;
  unsigned long long GetDefaultDuration() const;
  unsigned long long GetCodecDelay() const;
  unsigned long long GetSeekPreRoll() const;

  const BlockEntry* GetEOS() const;

  struct Settings {
    long long start;
    long long size;
  };

  class Info {
   public:
    Info();
    ~Info();
    int Copy(Info&) const;
    void Clear();
    long type;
    long number;
    unsigned long long uid;
    unsigned long long defaultDuration;
    unsigned long long codecDelay;
    unsigned long long seekPreRoll;
    char* nameAsUTF8;
    char* language;
    char* codecId;
    char* codecNameAsUTF8;
    unsigned char* codecPrivate;
    size_t codecPrivateSize;
    bool lacing;
    Settings settings;

   private:
    Info(const Info&);
    Info& operator=(const Info&);
    int CopyStr(char* Info::*str, Info&) const;
  };

  long GetFirst(const BlockEntry*&) const;
  long GetNext(const BlockEntry* pCurr, const BlockEntry*& pNext) const;
  virtual bool VetEntry(const BlockEntry*) const;
  virtual long Seek(long long time_ns, const BlockEntry*&) const;

  const ContentEncoding* GetContentEncodingByIndex(unsigned long idx) const;
  unsigned long GetContentEncodingCount() const;

  long ParseContentEncodingsEntry(long long start, long long size);

 protected:
  Track(Segment*, long long element_start, long long element_size);

  Info m_info;

  class EOSBlock : public BlockEntry {
   public:
    EOSBlock();

    Kind GetKind() const;
    const Block* GetBlock() const;
  };

  EOSBlock m_eos;

 private:
  ContentEncoding** content_encoding_entries_;
  ContentEncoding** content_encoding_entries_end_;
};

struct PrimaryChromaticity {
  PrimaryChromaticity() : x(0), y(0) {}
  ~PrimaryChromaticity() {}
  static bool Parse(IMkvReader* reader, long long read_pos,
                    long long value_size, bool is_x,
                    PrimaryChromaticity** chromaticity);
  float x;
  float y;
};

struct MasteringMetadata {
  static const float kValueNotPresent;

  MasteringMetadata()
      : r(NULL),
        g(NULL),
        b(NULL),
        white_point(NULL),
        luminance_max(kValueNotPresent),
        luminance_min(kValueNotPresent) {}
  ~MasteringMetadata() {
    delete r;
    delete g;
    delete b;
    delete white_point;
  }

  static bool Parse(IMkvReader* reader, long long element_start,
                    long long element_size,
                    MasteringMetadata** mastering_metadata);

  PrimaryChromaticity* r;
  PrimaryChromaticity* g;
  PrimaryChromaticity* b;
  PrimaryChromaticity* white_point;
  float luminance_max;
  float luminance_min;
};

struct Colour {
  static const long long kValueNotPresent;

  // Unless otherwise noted all values assigned upon construction are the
  // equivalent of unspecified/default.
  Colour()
      : matrix_coefficients(kValueNotPresent),
        bits_per_channel(kValueNotPresent),
        chroma_subsampling_horz(kValueNotPresent),
        chroma_subsampling_vert(kValueNotPresent),
        cb_subsampling_horz(kValueNotPresent),
        cb_subsampling_vert(kValueNotPresent),
        chroma_siting_horz(kValueNotPresent),
        chroma_siting_vert(kValueNotPresent),
        range(kValueNotPresent),
        transfer_characteristics(kValueNotPresent),
        primaries(kValueNotPresent),
        max_cll(kValueNotPresent),
        max_fall(kValueNotPresent),
        mastering_metadata(NULL) {}
  ~Colour() {
    delete mastering_metadata;
    mastering_metadata = NULL;
  }

  static bool Parse(IMkvReader* reader, long long element_start,
                    long long element_size, Colour** colour);

  long long matrix_coefficients;
  long long bits_per_channel;
  long long chroma_subsampling_horz;
  long long chroma_subsampling_vert;
  long long cb_subsampling_horz;
  long long cb_subsampling_vert;
  long long chroma_siting_horz;
  long long chroma_siting_vert;
  long long range;
  long long transfer_characteristics;
  long long primaries;
  long long max_cll;
  long long max_fall;

  MasteringMetadata* mastering_metadata;
};

struct Projection {
  enum ProjectionType {
    kTypeNotPresent = -1,
    kRectangular = 0,
    kEquirectangular = 1,
    kCubeMap = 2,
    kMesh = 3,
  };
  static const float kValueNotPresent;
  Projection()
      : type(kTypeNotPresent),
        private_data(NULL),
        private_data_length(0),
        pose_yaw(kValueNotPresent),
        pose_pitch(kValueNotPresent),
        pose_roll(kValueNotPresent) {}
  ~Projection() { delete[] private_data; }
  static bool Parse(IMkvReader* reader, long long element_start,
                    long long element_size, Projection** projection);

  ProjectionType type;
  unsigned char* private_data;
  size_t private_data_length;
  float pose_yaw;
  float pose_pitch;
  float pose_roll;
};

class VideoTrack : public Track {
  VideoTrack(const VideoTrack&);
  VideoTrack& operator=(const VideoTrack&);

  VideoTrack(Segment*, long long element_start, long long element_size);

 public:
  virtual ~VideoTrack();
  static long Parse(Segment*, const Info&, long long element_start,
                    long long element_size, VideoTrack*&);

  long long GetWidth() const;
  long long GetHeight() const;
  long long GetDisplayWidth() const;
  long long GetDisplayHeight() const;
  long long GetDisplayUnit() const;
  long long GetStereoMode() const;
  double GetFrameRate() const;

  bool VetEntry(const BlockEntry*) const;
  long Seek(long long time_ns, const BlockEntry*&) const;

  Colour* GetColour() const;

  Projection* GetProjection() const;

 private:
  long long m_width;
  long long m_height;
  long long m_display_width;
  long long m_display_height;
  long long m_display_unit;
  long long m_stereo_mode;

  double m_rate;

  Colour* m_colour;
  Projection* m_projection;
};

class AudioTrack : public Track {
  AudioTrack(const AudioTrack&);
  AudioTrack& operator=(const AudioTrack&);

  AudioTrack(Segment*, long long element_start, long long element_size);

 public:
  static long Parse(Segment*, const Info&, long long element_start,
                    long long element_size, AudioTrack*&);

  double GetSamplingRate() const;
  long long GetChannels() const;
  long long GetBitDepth() const;

 private:
  double m_rate;
  long long m_channels;
  long long m_bitDepth;
};

class Tracks {
  Tracks(const Tracks&);
  Tracks& operator=(const Tracks&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  Tracks(Segment*, long long start, long long size, long long element_start,
         long long element_size);

  ~Tracks();

  long Parse();

  unsigned long GetTracksCount() const;

  const Track* GetTrackByNumber(long tn) const;
  const Track* GetTrackByIndex(unsigned long idx) const;

 private:
  Track** m_trackEntries;
  Track** m_trackEntriesEnd;

  long ParseTrackEntry(long long payload_start, long long payload_size,
                       long long element_start, long long element_size,
                       Track*&) const;
};

class Chapters {
  Chapters(const Chapters&);
  Chapters& operator=(const Chapters&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  Chapters(Segment*, long long payload_start, long long payload_size,
           long long element_start, long long element_size);

  ~Chapters();

  long Parse();

  class Atom;
  class Edition;

  class Display {
    friend class Atom;
    Display();
    Display(const Display&);
    ~Display();
    Display& operator=(const Display&);

   public:
    const char* GetString() const;
    const char* GetLanguage() const;
    const char* GetCountry() const;

   private:
    void Init();
    void ShallowCopy(Display&) const;
    void Clear();
    long Parse(IMkvReader*, long long pos, long long size);

    char* m_string;
    char* m_language;
    char* m_country;
  };

  class Atom {
    friend class Edition;
    Atom();
    Atom(const Atom&);
    ~Atom();
    Atom& operator=(const Atom&);

   public:
    unsigned long long GetUID() const;
    const char* GetStringUID() const;

    long long GetStartTimecode() const;
    long long GetStopTimecode() const;

    long long GetStartTime(const Chapters*) const;
    long long GetStopTime(const Chapters*) const;

    int GetDisplayCount() const;
    const Display* GetDisplay(int index) const;

   private:
    void Init();
    void ShallowCopy(Atom&) const;
    void Clear();
    long Parse(IMkvReader*, long long pos, long long size);
    static long long GetTime(const Chapters*, long long timecode);

    long ParseDisplay(IMkvReader*, long long pos, long long size);
    bool ExpandDisplaysArray();

    char* m_string_uid;
    unsigned long long m_uid;
    long long m_start_timecode;
    long long m_stop_timecode;

    Display* m_displays;
    int m_displays_size;
    int m_displays_count;
  };

  class Edition {
    friend class Chapters;
    Edition();
    Edition(const Edition&);
    ~Edition();
    Edition& operator=(const Edition&);

   public:
    int GetAtomCount() const;
    const Atom* GetAtom(int index) const;

   private:
    void Init();
    void ShallowCopy(Edition&) const;
    void Clear();
    long Parse(IMkvReader*, long long pos, long long size);

    long ParseAtom(IMkvReader*, long long pos, long long size);
    bool ExpandAtomsArray();

    Atom* m_atoms;
    int m_atoms_size;
    int m_atoms_count;
  };

  int GetEditionCount() const;
  const Edition* GetEdition(int index) const;

 private:
  long ParseEdition(long long pos, long long size);
  bool ExpandEditionsArray();

  Edition* m_editions;
  int m_editions_size;
  int m_editions_count;
};

class Tags {
  Tags(const Tags&);
  Tags& operator=(const Tags&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  Tags(Segment*, long long payload_start, long long payload_size,
       long long element_start, long long element_size);

  ~Tags();

  long Parse();

  class Tag;
  class SimpleTag;

  class SimpleTag {
    friend class Tag;
    SimpleTag();
    SimpleTag(const SimpleTag&);
    ~SimpleTag();
    SimpleTag& operator=(const SimpleTag&);

   public:
    const char* GetTagName() const;
    const char* GetTagString() const;

   private:
    void Init();
    void ShallowCopy(SimpleTag&) const;
    void Clear();
    long Parse(IMkvReader*, long long pos, long long size);

    char* m_tag_name;
    char* m_tag_string;
  };

  class Tag {
    friend class Tags;
    Tag();
    Tag(const Tag&);
    ~Tag();
    Tag& operator=(const Tag&);

   public:
    int GetSimpleTagCount() const;
    const SimpleTag* GetSimpleTag(int index) const;

   private:
    void Init();
    void ShallowCopy(Tag&) const;
    void Clear();
    long Parse(IMkvReader*, long long pos, long long size);

    long ParseSimpleTag(IMkvReader*, long long pos, long long size);
    bool ExpandSimpleTagsArray();

    SimpleTag* m_simple_tags;
    int m_simple_tags_size;
    int m_simple_tags_count;
  };

  int GetTagCount() const;
  const Tag* GetTag(int index) const;

 private:
  long ParseTag(long long pos, long long size);
  bool ExpandTagsArray();

  Tag* m_tags;
  int m_tags_size;
  int m_tags_count;
};

class SegmentInfo {
  SegmentInfo(const SegmentInfo&);
  SegmentInfo& operator=(const SegmentInfo&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  SegmentInfo(Segment*, long long start, long long size,
              long long element_start, long long element_size);

  ~SegmentInfo();

  long Parse();

  long long GetTimeCodeScale() const;
  long long GetDuration() const;  // scaled
  const char* GetMuxingAppAsUTF8() const;
  const char* GetWritingAppAsUTF8() const;
  const char* GetTitleAsUTF8() const;

 private:
  long long m_timecodeScale;
  double m_duration;
  char* m_pMuxingAppAsUTF8;
  char* m_pWritingAppAsUTF8;
  char* m_pTitleAsUTF8;
};

class SeekHead {
  SeekHead(const SeekHead&);
  SeekHead& operator=(const SeekHead&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  SeekHead(Segment*, long long start, long long size, long long element_start,
           long long element_size);

  ~SeekHead();

  long Parse();

  struct Entry {
    Entry();

    // the SeekHead entry payload
    long long id;
    long long pos;

    // absolute pos of SeekEntry ID
    long long element_start;

    // SeekEntry ID size + size size + payload
    long long element_size;
  };

  int GetCount() const;
  const Entry* GetEntry(int idx) const;

  struct VoidElement {
    // absolute pos of Void ID
    long long element_start;

    // ID size + size size + payload size
    long long element_size;
  };

  int GetVoidElementCount() const;
  const VoidElement* GetVoidElement(int idx) const;

 private:
  Entry* m_entries;
  int m_entry_count;

  VoidElement* m_void_elements;
  int m_void_element_count;

  static bool ParseEntry(IMkvReader*,
                         long long pos,  // payload
                         long long size, Entry*);
};

class Cues;
class CuePoint {
  friend class Cues;

  CuePoint(long, long long);
  ~CuePoint();

  CuePoint(const CuePoint&);
  CuePoint& operator=(const CuePoint&);

 public:
  long long m_element_start;
  long long m_element_size;

  bool Load(IMkvReader*);

  long long GetTimeCode() const;  // absolute but unscaled
  long long GetTime(const Segment*) const;  // absolute and scaled (ns units)

  struct TrackPosition {
    long long m_track;
    long long m_pos;  // of cluster
    long long m_block;
    // codec_state  //defaults to 0
    // reference = clusters containing req'd referenced blocks
    //  reftime = timecode of the referenced block

    bool Parse(IMkvReader*, long long, long long);
  };

  const TrackPosition* Find(const Track*) const;

 private:
  const long m_index;
  long long m_timecode;
  TrackPosition* m_track_positions;
  size_t m_track_positions_count;
};

class Cues {
  friend class Segment;

  Cues(Segment*, long long start, long long size, long long element_start,
       long long element_size);
  ~Cues();

  Cues(const Cues&);
  Cues& operator=(const Cues&);

 public:
  Segment* const m_pSegment;
  const long long m_start;
  const long long m_size;
  const long long m_element_start;
  const long long m_element_size;

  bool Find(  // lower bound of time_ns
      long long time_ns, const Track*, const CuePoint*&,
      const CuePoint::TrackPosition*&) const;

  const CuePoint* GetFirst() const;
  const CuePoint* GetLast() const;
  const CuePoint* GetNext(const CuePoint*) const;

  const BlockEntry* GetBlock(const CuePoint*,
                             const CuePoint::TrackPosition*) const;

  bool LoadCuePoint() const;
  long GetCount() const;  // loaded only
  // long GetTotal() const;  //loaded + preloaded
  bool DoneParsing() const;

 private:
  bool Init() const;
  bool PreloadCuePoint(long&, long long) const;

  mutable CuePoint** m_cue_points;
  mutable long m_count;
  mutable long m_preload_count;
  mutable long long m_pos;
};

class Cluster {
  friend class Segment;

  Cluster(const Cluster&);
  Cluster& operator=(const Cluster&);

 public:
  Segment* const m_pSegment;

 public:
  static Cluster* Create(Segment*,
                         long index,  // index in segment
                         long long off);  // offset relative to segment
  // long long element_size);

  Cluster();  // EndOfStream
  ~Cluster();

  bool EOS() const;

  long long GetTimeCode() const;  // absolute, but not scaled
  long long GetTime() const;  // absolute, and scaled (nanosecond units)
  long long GetFirstTime() const;  // time (ns) of first (earliest) block
  long long GetLastTime() const;  // time (ns) of last (latest) block

  long GetFirst(const BlockEntry*&) const;
  long GetLast(const BlockEntry*&) const;
  long GetNext(const BlockEntry* curr, const BlockEntry*& next) const;

  const BlockEntry* GetEntry(const Track*, long long ns = -1) const;
  const BlockEntry* GetEntry(const CuePoint&,
                             const CuePoint::TrackPosition&) const;
  // const BlockEntry* GetMaxKey(const VideoTrack*) const;

  //    static bool HasBlockEntries(const Segment*, long long);

  static long HasBlockEntries(const Segment*, long long idoff, long long& pos,
                              long& size);

  long GetEntryCount() const;

  long Load(long long& pos, long& size) const;

  long Parse(long long& pos, long& size) const;
  long GetEntry(long index, const mkvparser::BlockEntry*&) const;

 protected:
  Cluster(Segment*, long index, long long element_start);
  // long long element_size);

 public:
  const long long m_element_start;
  long long GetPosition() const;  // offset relative to segment

  long GetIndex() const;
  long long GetElementSize() const;
  // long long GetPayloadSize() const;

  // long long Unparsed() const;

 private:
  long m_index;
  mutable long long m_pos;
  // mutable long long m_size;
  mutable long long m_element_size;
  mutable long long m_timecode;
  mutable BlockEntry** m_entries;
  mutable long m_entries_size;
  mutable long m_entries_count;

  long ParseSimpleBlock(long long, long long&, long&);
  long ParseBlockGroup(long long, long long&, long&);

  long CreateBlock(long long id, long long pos, long long size,
                   long long discard_padding);
  long CreateBlockGroup(long long start_offset, long long size,
                        long long discard_padding);
  long CreateSimpleBlock(long long, long long);
};

class Segment {
  friend class Cues;
  friend class Track;
  friend class VideoTrack;

  Segment(const Segment&);
  Segment& operator=(const Segment&);

 private:
  Segment(IMkvReader*, long long elem_start,
          // long long elem_size,
          long long pos, long long size);

 public:
  IMkvReader* const m_pReader;
  const long long m_element_start;
  // const long long m_element_size;
  const long long m_start;  // posn of segment payload
  const long long m_size;  // size of segment payload
  Cluster m_eos;  // TODO: make private?

  static long long CreateInstance(IMkvReader*, long long, Segment*&);
  ~Segment();

  long Load();  // loads headers and all clusters

  // for incremental loading
  // long long Unparsed() const;
  bool DoneParsing() const;
  long long ParseHeaders();  // stops when first cluster is found
  // long FindNextCluster(long long& pos, long& size) const;
  long LoadCluster(long long& pos, long& size);  // load one cluster
  long LoadCluster();

  long ParseNext(const Cluster* pCurr, const Cluster*& pNext, long long& pos,
                 long& size);

  const SeekHead* GetSeekHead() const;
  const Tracks* GetTracks() const;
  const SegmentInfo* GetInfo() const;
  const Cues* GetCues() const;
  const Chapters* GetChapters() const;
  const Tags* GetTags() const;

  long long GetDuration() const;

  unsigned long GetCount() const;
  const Cluster* GetFirst() const;
  const Cluster* GetLast() const;
  const Cluster* GetNext(const Cluster*);

  const Cluster* FindCluster(long long time_nanoseconds) const;
  // const BlockEntry* Seek(long long time_nanoseconds, const Track*) const;

  const Cluster* FindOrPreloadCluster(long long pos);

  long ParseCues(long long cues_off,  // offset relative to start of segment
                 long long& parse_pos, long& parse_len);

 private:
  long long m_pos;  // absolute file posn; what has been consumed so far
  Cluster* m_pUnknownSize;

  SeekHead* m_pSeekHead;
  SegmentInfo* m_pInfo;
  Tracks* m_pTracks;
  Cues* m_pCues;
  Chapters* m_pChapters;
  Tags* m_pTags;
  Cluster** m_clusters;
  long m_clusterCount;  // number of entries for which m_index >= 0
  long m_clusterPreloadCount;  // number of entries for which m_index < 0
  long m_clusterSize;  // array size

  long DoLoadCluster(long long&, long&);
  long DoLoadClusterUnknownSize(long long&, long&);
  long DoParseNext(const Cluster*&, long long&, long&);

  bool AppendCluster(Cluster*);
  bool PreloadCluster(Cluster*, ptrdiff_t);

  // void ParseSeekHead(long long pos, long long size);
  // void ParseSeekEntry(long long pos, long long size);
  // void ParseCues(long long);

  const BlockEntry* GetBlock(const CuePoint&, const CuePoint::TrackPosition&);
};

}  // namespace mkvparser

inline long mkvparser::Segment::LoadCluster() {
  long long pos;
  long size;

  return LoadCluster(pos, size);
}

#endif  // MKVPARSER_MKVPARSER_H_
