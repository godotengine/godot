// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"
#include "webvtt/webvttparser.h"

using std::string;

namespace libwebm {
namespace vttdemux {

typedef long long mkvtime_t;  // NOLINT
typedef long long mkvpos_t;  // NOLINT
typedef std::unique_ptr<mkvparser::Segment> segment_ptr_t;

// WebVTT metadata tracks have a type (encoded in the CodecID for the track).
// We use |type| to synthesize a filename for the out-of-band WebVTT |file|.
struct MetadataInfo {
  enum Type { kSubtitles, kCaptions, kDescriptions, kMetadata, kChapters } type;
  FILE* file;
};

// We use a map, indexed by track number, to collect information about
// each track in the input file.
typedef std::map<long, MetadataInfo> metadata_map_t;  // NOLINT

// The distinguished key value we use to store the chapters
// information in the metadata map.
enum { kChaptersKey = 0 };

// The data from the original WebVTT Cue is stored as a WebM block.
// The FrameParser is used to parse the lines of text out from the
// block, in order to reconstruct the original WebVTT Cue.
class FrameParser : public libwebvtt::LineReader {
 public:
  //  Bind the FrameParser instance to a WebM block.
  explicit FrameParser(const mkvparser::BlockGroup* block_group);
  virtual ~FrameParser();

  // The Webm block (group) to which this instance is bound.  We
  // treat the payload of the block as a stream of characters.
  const mkvparser::BlockGroup* const block_group_;

 protected:
  // Read the next character from the character stream (the payload
  // of the WebM block).  We increment the stream pointer |pos_| as
  // each character from the stream is consumed.
  virtual int GetChar(char* c);

  // End-of-line handling requires that we put a character back into
  // the stream.  Here we need only decrement the stream pointer |pos_|
  // to unconsume the character.
  virtual void UngetChar(char c);

  // The current position in the character stream (the payload of the block).
  mkvpos_t pos_;

  // The position of the end of the character stream. When the current
  // position |pos_| equals the end position |pos_end_|, the entire
  // stream (block payload) has been consumed and end-of-stream is indicated.
  mkvpos_t pos_end_;

 private:
  // Disable copy ctor and copy assign
  FrameParser(const FrameParser&);
  FrameParser& operator=(const FrameParser&);
};

// The data from the original WebVTT Cue is stored as an MKV Chapters
// Atom element (the cue payload is stored as a Display sub-element).
// The ChapterAtomParser is used to parse the lines of text out from
// the String sub-element of the Display element (though it would be
// admittedly odd if there were more than one line).
class ChapterAtomParser : public libwebvtt::LineReader {
 public:
  explicit ChapterAtomParser(const mkvparser::Chapters::Display* display);
  virtual ~ChapterAtomParser();

  const mkvparser::Chapters::Display* const display_;

 protected:
  // Read the next character from the character stream (the title
  // member of the atom's display).  We increment the stream pointer
  // |str_| as each character from the stream is consumed.
  virtual int GetChar(char* c);

  // End-of-line handling requires that we put a character back into
  // the stream.  Here we need only decrement the stream pointer |str_|
  // to unconsume the character.
  virtual void UngetChar(char c);

  // The current position in the character stream (the title of the
  // atom's display).
  const char* str_;

  // The position of the end of the character stream. When the current
  // position |str_| equals the end position |str_end_|, the entire
  // stream (title of the display) has been consumed and end-of-stream
  // is indicated.
  const char* str_end_;

 private:
  ChapterAtomParser(const ChapterAtomParser&);
  ChapterAtomParser& operator=(const ChapterAtomParser&);
};

// Parse the EBML header of the WebM input file, to determine whether we
// actually have a WebM file.  Returns false if this is not a WebM file.
bool ParseHeader(mkvparser::IMkvReader* reader, mkvpos_t* pos);

// Parse the Segment of the input file and load all of its clusters.
// Returns false if there was an error parsing the file.
bool ParseSegment(mkvparser::IMkvReader* reader, mkvpos_t pos,
                  segment_ptr_t* segment);

// If |segment| has a Chapters element (in which case, there will be a
// corresponding entry in |metadata_map|), convert the MKV chapters to
// WebVTT chapter cues and write them to the output file.  Returns
// false on error.
bool WriteChaptersFile(const metadata_map_t& metadata_map,
                       const mkvparser::Segment* segment);

// Convert an MKV Chapters Atom to a WebVTT cue and write it to the
// output |file|.  Returns false on error.
bool WriteChaptersCue(FILE* file, const mkvparser::Chapters* chapters,
                      const mkvparser::Chapters::Atom* atom,
                      const mkvparser::Chapters::Display* display);

// Write the Cue Identifier line of the WebVTT cue, if it's present.
// Returns false on error.
bool WriteChaptersCueIdentifier(FILE* file,
                                const mkvparser::Chapters::Atom* atom);

// Use the timecodes from the chapters |atom| to write just the
// timings line of the WebVTT cue.  Returns false on error.
bool WriteChaptersCueTimings(FILE* file, const mkvparser::Chapters* chapters,
                             const mkvparser::Chapters::Atom* atom);

// Parse the String sub-element of the |display| and write the payload
// of the WebVTT cue.  Returns false on error.
bool WriteChaptersCuePayload(FILE* file,
                             const mkvparser::Chapters::Display* display);

// Iterate over the tracks of the input file (and any chapters
// element) and cache information about each metadata track.
void BuildMap(const mkvparser::Segment* segment, metadata_map_t* metadata_map);

// For each track listed in the cache, synthesize its output filename
// and open a file handle that designates the out-of-band file.
// Returns false if we were unable to open an output file for a track.
bool OpenFiles(metadata_map_t* metadata_map, const char* filename);

// Close the file handle for each track in the cache.
void CloseFiles(metadata_map_t* metadata_map);

// Iterate over the clusters of the input file, and write a WebVTT cue
// for each metadata block.  Returns false if processing of a cluster
// failed.
bool WriteFiles(const metadata_map_t& m, mkvparser::Segment* s);

// Write the WebVTT header for each track in the cache.  We do this
// immediately before writing the actual WebVTT cues.  Returns false
// if the write failed.
bool InitializeFiles(const metadata_map_t& metadata_map);

// Iterate over the blocks of the |cluster|, writing a WebVTT cue to
// its associated output file for each block of metadata.  Returns
// false if processing a block failed, or there was a parse error.
bool ProcessCluster(const metadata_map_t& metadata_map,
                    const mkvparser::Cluster* cluster);

// Look up this track number in the cache, and if found (meaning this
// is a metadata track), write a WebVTT cue to the associated output
// file.  Returns false if writing the WebVTT cue failed.
bool ProcessBlockEntry(const metadata_map_t& metadata_map,
                       const mkvparser::BlockEntry* block_entry);

// Parse the lines of text from the |block_group| to reconstruct the
// original WebVTT cue, and write it to the associated output |file|.
// Returns false if there was an error writing to the output file.
bool WriteCue(FILE* file, const mkvparser::BlockGroup* block_group);

// Consume a line of text from the character stream, and if the line
// is not empty write the cue identifier to the associated output
// file.  Returns false if there was an error writing to the file.
bool WriteCueIdentifier(FILE* f, FrameParser* parser);

// Consume a line of text from the character stream (which holds any
// cue settings) and write the cue timings line for this cue to the
// associated output file.  Returns false if there was an error
// writing to the file.
bool WriteCueTimings(FILE* f, FrameParser* parser);

// Write the timestamp (representating either the start time or stop
// time of the cue) to the output file.  Returns false if there was an
// error writing to the file.
bool WriteCueTime(FILE* f, mkvtime_t time_ns);

// Consume the remaining lines of text from the character stream
// (these lines are the actual payload of the WebVTT cue), and write
// them to the associated output file.  Returns false if there was an
// error writing to the file.
bool WriteCuePayload(FILE* f, FrameParser* parser);
}  // namespace vttdemux

namespace vttdemux {

FrameParser::FrameParser(const mkvparser::BlockGroup* block_group)
    : block_group_(block_group) {
  const mkvparser::Block* const block = block_group->GetBlock();
  const mkvparser::Block::Frame& f = block->GetFrame(0);

  // The beginning and end of the character stream corresponds to the
  // position of this block's frame within the WebM input file.

  pos_ = f.pos;
  pos_end_ = f.pos + f.len;
}

FrameParser::~FrameParser() {}

int FrameParser::GetChar(char* c) {
  if (pos_ >= pos_end_)  // end-of-stream
    return 1;  // per the semantics of libwebvtt::Reader::GetChar

  const mkvparser::Cluster* const cluster = block_group_->GetCluster();
  const mkvparser::Segment* const segment = cluster->m_pSegment;
  mkvparser::IMkvReader* const reader = segment->m_pReader;

  unsigned char* const buf = reinterpret_cast<unsigned char*>(c);
  const int result = reader->Read(pos_, 1, buf);

  if (result < 0)  // error
    return -1;

  ++pos_;  // consume this character in the stream
  return 0;
}

void FrameParser::UngetChar(char /* c */) {
  // All we need to do here is decrement the position in the stream.
  // The next time GetChar is called the same character will be
  // re-read from the input file.
  --pos_;
}

ChapterAtomParser::ChapterAtomParser(
    const mkvparser::Chapters::Display* display)
    : display_(display) {
  str_ = display->GetString();
  if (str_ == NULL)
    return;
  const size_t len = strlen(str_);
  str_end_ = str_ + len;
}

ChapterAtomParser::~ChapterAtomParser() {}

int ChapterAtomParser::GetChar(char* c) {
  if (str_ == NULL || str_ >= str_end_)  // end-of-stream
    return 1;  // per the semantics of libwebvtt::Reader::GetChar

  *c = *str_++;  // consume this character in the stream
  return 0;
}

void ChapterAtomParser::UngetChar(char /* c */) {
  // All we need to do here is decrement the position in the stream.
  // The next time GetChar is called the same character will be
  // re-read from the input file.
  --str_;
}

}  // namespace vttdemux

bool vttdemux::ParseHeader(mkvparser::IMkvReader* reader, mkvpos_t* pos) {
  mkvparser::EBMLHeader h;
  const mkvpos_t status = h.Parse(reader, *pos);

  if (status) {
    printf("error parsing EBML header\n");
    return false;
  }

  if (h.m_docType == NULL || strcmp(h.m_docType, "webm") != 0) {
    printf("bad doctype\n");
    return false;
  }

  return true;  // success
}

bool vttdemux::ParseSegment(mkvparser::IMkvReader* reader, mkvpos_t pos,
                            segment_ptr_t* segment_ptr) {
  // We first create the segment object.

  mkvparser::Segment* p;
  const mkvpos_t create = mkvparser::Segment::CreateInstance(reader, pos, p);

  if (create) {
    printf("error parsing segment element\n");
    return false;
  }

  segment_ptr->reset(p);

  // Now parse all of the segment's sub-elements, in toto.

  const long status = p->Load();  // NOLINT

  if (status < 0) {
    printf("error loading segment\n");
    return false;
  }

  return true;
}

void vttdemux::BuildMap(const mkvparser::Segment* segment,
                        metadata_map_t* map_ptr) {
  metadata_map_t& m = *map_ptr;
  m.clear();

  if (segment->GetChapters()) {
    MetadataInfo info;
    info.file = NULL;
    info.type = MetadataInfo::kChapters;

    m[kChaptersKey] = info;
  }

  const mkvparser::Tracks* const tt = segment->GetTracks();
  if (tt == NULL)
    return;

  const long tc = tt->GetTracksCount();  // NOLINT
  if (tc <= 0)
    return;

  // Iterate over the tracks in the intput file.  We determine whether
  // a track holds metadata by inspecting its CodecID.

  for (long idx = 0; idx < tc; ++idx) {  // NOLINT
    const mkvparser::Track* const t = tt->GetTrackByIndex(idx);

    if (t == NULL)  // weird
      continue;

    const long tn = t->GetNumber();  // NOLINT

    if (tn <= 0)  // weird
      continue;

    const char* const codec_id = t->GetCodecId();

    if (codec_id == NULL)  // weird
      continue;

    MetadataInfo info;
    info.file = NULL;

    if (strcmp(codec_id, "D_WEBVTT/SUBTITLES") == 0) {
      info.type = MetadataInfo::kSubtitles;
    } else if (strcmp(codec_id, "D_WEBVTT/CAPTIONS") == 0) {
      info.type = MetadataInfo::kCaptions;
    } else if (strcmp(codec_id, "D_WEBVTT/DESCRIPTIONS") == 0) {
      info.type = MetadataInfo::kDescriptions;
    } else if (strcmp(codec_id, "D_WEBVTT/METADATA") == 0) {
      info.type = MetadataInfo::kMetadata;
    } else {
      continue;
    }

    m[tn] = info;  // create an entry in the cache for this track
  }
}

bool vttdemux::OpenFiles(metadata_map_t* metadata_map, const char* filename) {
  if (metadata_map == NULL || metadata_map->empty())
    return false;

  if (filename == NULL)
    return false;

  // Find the position of the filename extension.  We synthesize the
  // output filename from the directory path and basename of the input
  // filename.

  const char* const ext = strrchr(filename, '.');

  if (ext == NULL)  // TODO(matthewjheaney): liberalize?
    return false;

  // Remember whether a track of this type has already been seen (the
  // map key) by keeping a count (the map item).  We quality the
  // output filename with the track number if there is more than one
  // track having a given type.

  std::map<MetadataInfo::Type, int> exists;

  typedef metadata_map_t::iterator iter_t;

  metadata_map_t& m = *metadata_map;
  const iter_t ii = m.begin();
  const iter_t j = m.end();

  // Make a first pass over the cache to determine whether there is
  // more than one track corresponding to a given metadata type.

  iter_t i = ii;
  while (i != j) {
    const metadata_map_t::value_type& v = *i++;
    const MetadataInfo& info = v.second;
    const MetadataInfo::Type type = info.type;
    ++exists[type];
  }

  // Make a second pass over the cache, synthesizing the filename of
  // each output file (from the input file basename, the input track
  // metadata type, and its track number if necessary), and then
  // opening a WebVTT output file having that filename.

  i = ii;
  while (i != j) {
    metadata_map_t::value_type& v = *i++;
    MetadataInfo& info = v.second;
    const MetadataInfo::Type type = info.type;

    // Start with the basename of the input file.

    string name(filename, ext);

    // Next append the metadata kind.

    switch (type) {
      case MetadataInfo::kSubtitles:
        name += "_SUBTITLES";
        break;

      case MetadataInfo::kCaptions:
        name += "_CAPTIONS";
        break;

      case MetadataInfo::kDescriptions:
        name += "_DESCRIPTIONS";
        break;

      case MetadataInfo::kMetadata:
        name += "_METADATA";
        break;

      case MetadataInfo::kChapters:
        name += "_CHAPTERS";
        break;

      default:
        return false;
    }

    // If there is more than one metadata track having a given type
    // (the WebVTT-in-WebM spec doesn't preclude this), then qualify
    // the output filename with the input track number.

    if (exists[type] > 1) {
      enum { kLen = 33 };
      char str[kLen];  // max 126 tracks, so only 4 chars really needed
#ifndef _MSC_VER
      snprintf(str, kLen, "%ld", v.first);  // track number
#else
      _snprintf_s(str, sizeof(str), kLen, "%ld", v.first);  // track number
#endif
      name += str;
    }

    // Finally append the output filename extension.

    name += ".vtt";

    // We have synthesized the full output filename, so attempt to
    // open the WebVTT output file.

    info.file = fopen(name.c_str(), "wb");
    const bool success = (info.file != NULL);

    if (!success) {
      printf("unable to open output file %s\n", name.c_str());
      return false;
    }
  }

  return true;
}

void vttdemux::CloseFiles(metadata_map_t* metadata_map) {
  if (metadata_map == NULL)
    return;

  metadata_map_t& m = *metadata_map;

  typedef metadata_map_t::iterator iter_t;

  iter_t i = m.begin();
  const iter_t j = m.end();

  // Gracefully close each output file, to ensure all output gets
  // propertly flushed.

  while (i != j) {
    metadata_map_t::value_type& v = *i++;
    MetadataInfo& info = v.second;

    if (info.file != NULL) {
      fclose(info.file);
      info.file = NULL;
    }
  }
}

bool vttdemux::WriteFiles(const metadata_map_t& m, mkvparser::Segment* s) {
  // First write the WebVTT header.

  InitializeFiles(m);

  if (!WriteChaptersFile(m, s))
    return false;

  // Now iterate over the clusters, writing the WebVTT cue as we parse
  // each metadata block.

  const mkvparser::Cluster* cluster = s->GetFirst();

  while (cluster != NULL && !cluster->EOS()) {
    if (!ProcessCluster(m, cluster))
      return false;

    cluster = s->GetNext(cluster);
  }

  return true;
}

bool vttdemux::InitializeFiles(const metadata_map_t& m) {
  // Write the WebVTT header for each output file in the cache.

  typedef metadata_map_t::const_iterator iter_t;
  iter_t i = m.begin();
  const iter_t j = m.end();

  while (i != j) {
    const metadata_map_t::value_type& v = *i++;
    const MetadataInfo& info = v.second;
    FILE* const f = info.file;

    if (fputs("WEBVTT\n", f) < 0) {
      printf("unable to initialize output file\n");
      return false;
    }
  }

  return true;
}

bool vttdemux::WriteChaptersFile(const metadata_map_t& m,
                                 const mkvparser::Segment* s) {
  const metadata_map_t::const_iterator info_iter = m.find(kChaptersKey);
  if (info_iter == m.end())  // no chapters, so nothing to do
    return true;

  const mkvparser::Chapters* const chapters = s->GetChapters();
  if (chapters == NULL)  // weird
    return true;

  const MetadataInfo& info = info_iter->second;
  FILE* const file = info.file;

  const int edition_count = chapters->GetEditionCount();

  if (edition_count <= 0)  // weird
    return true;  // nothing to do

  if (edition_count > 1) {
    // TODO(matthewjheaney): figure what to do here
    printf("more than one chapter edition detected\n");
    return false;
  }

  const mkvparser::Chapters::Edition* const edition = chapters->GetEdition(0);

  const int atom_count = edition->GetAtomCount();

  for (int idx = 0; idx < atom_count; ++idx) {
    const mkvparser::Chapters::Atom* const atom = edition->GetAtom(idx);
    const int display_count = atom->GetDisplayCount();

    if (display_count <= 0)
      continue;

    if (display_count > 1) {
      // TODO(matthewjheaney): handle case of multiple languages
      printf("more than 1 display in atom detected\n");
      return false;
    }

    const mkvparser::Chapters::Display* const display = atom->GetDisplay(0);

    if (const char* language = display->GetLanguage()) {
      if (strcmp(language, "eng") != 0) {
        // TODO(matthewjheaney): handle case of multiple languages.

        // We must create a separate webvtt file for each language.
        // This isn't a simple problem (which is why we defer it for
        // now), because there's nothing in the header that tells us
        // what languages we have as cues.  We must parse the displays
        // of each atom to determine that.

        // One solution is to make two passes over the input data.
        // First parse the displays, creating an in-memory cache of
        // all the chapter cues, sorted according to their language.
        // After we have read all of the chapter atoms from the input
        // file, we can then write separate output files for each
        // language.

        printf("only English-language chapter cues are supported\n");
        return false;
      }
    }

    if (!WriteChaptersCue(file, chapters, atom, display))
      return false;
  }

  return true;
}

bool vttdemux::WriteChaptersCue(FILE* f, const mkvparser::Chapters* chapters,
                                const mkvparser::Chapters::Atom* atom,
                                const mkvparser::Chapters::Display* display) {
  // We start a new cue by writing a cue separator (an empty line)
  // into the stream.

  if (fputc('\n', f) < 0)
    return false;

  // A WebVTT Cue comprises 3 things: a cue identifier, followed by
  // the cue timings, followed by the payload of the cue.  We write
  // each part of the cue in sequence.

  if (!WriteChaptersCueIdentifier(f, atom))
    return false;

  if (!WriteChaptersCueTimings(f, chapters, atom))
    return false;

  if (!WriteChaptersCuePayload(f, display))
    return false;

  return true;
}

bool vttdemux::WriteChaptersCueIdentifier(
    FILE* f, const mkvparser::Chapters::Atom* atom) {
  const char* const identifier = atom->GetStringUID();

  if (identifier == NULL)
    return true;  // nothing else to do

  if (fprintf(f, "%s\n", identifier) < 0)
    return false;

  return true;
}

bool vttdemux::WriteChaptersCueTimings(FILE* f,
                                       const mkvparser::Chapters* chapters,
                                       const mkvparser::Chapters::Atom* atom) {
  const mkvtime_t start_ns = atom->GetStartTime(chapters);

  if (start_ns < 0)
    return false;

  const mkvtime_t stop_ns = atom->GetStopTime(chapters);

  if (stop_ns < 0)
    return false;

  if (!WriteCueTime(f, start_ns))
    return false;

  if (fputs(" --> ", f) < 0)
    return false;

  if (!WriteCueTime(f, stop_ns))
    return false;

  if (fputc('\n', f) < 0)
    return false;

  return true;
}

bool vttdemux::WriteChaptersCuePayload(
    FILE* f, const mkvparser::Chapters::Display* display) {
  // Bind a Chapter parser object to the display, which allows us to
  // extract each line of text from the title-part of the display.
  ChapterAtomParser parser(display);

  int count = 0;  // count of lines of payload text written to output file
  for (string line;;) {
    const int e = parser.GetLine(&line);

    if (e < 0)  // error (only -- we allow EOS here)
      return false;

    if (line.empty())  // TODO(matthewjheaney): retain this check?
      break;

    if (fprintf(f, "%s\n", line.c_str()) < 0)
      return false;

    ++count;
  }

  if (count <= 0)  // WebVTT cue requires non-empty payload
    return false;

  return true;
}

bool vttdemux::ProcessCluster(const metadata_map_t& m,
                              const mkvparser::Cluster* c) {
  // Visit the blocks in this cluster, writing a WebVTT cue for each
  // metadata block.

  const mkvparser::BlockEntry* block_entry;

  long result = c->GetFirst(block_entry);  // NOLINT
  if (result < 0) {
    printf("bad cluster (unable to get first block)\n");
    return false;
  }

  while (block_entry != NULL && !block_entry->EOS()) {
    if (!ProcessBlockEntry(m, block_entry))
      return false;

    result = c->GetNext(block_entry, block_entry);
    if (result < 0) {  // error
      printf("bad cluster (unable to get next block)\n");
      return false;
    }
  }

  return true;
}

bool vttdemux::ProcessBlockEntry(const metadata_map_t& m,
                                 const mkvparser::BlockEntry* block_entry) {
  // If the track number for this block is in the cache, then we have
  // a metadata block, so write the WebVTT cue to the output file.

  const mkvparser::Block* const block = block_entry->GetBlock();
  const long long tn = block->GetTrackNumber();  // NOLINT

  typedef metadata_map_t::const_iterator iter_t;
  const iter_t i = m.find(static_cast<metadata_map_t::key_type>(tn));

  if (i == m.end())  // not a metadata track
    return true;  // nothing else to do

  if (block_entry->GetKind() != mkvparser::BlockEntry::kBlockGroup)
    return false;  // weird

  typedef mkvparser::BlockGroup BG;
  const BG* const block_group = static_cast<const BG*>(block_entry);

  const MetadataInfo& info = i->second;
  FILE* const f = info.file;

  return WriteCue(f, block_group);
}

bool vttdemux::WriteCue(FILE* f, const mkvparser::BlockGroup* block_group) {
  // Bind a FrameParser object to the block, which allows us to
  // extract each line of text from the payload of the block.
  FrameParser parser(block_group);

  // We start a new cue by writing a cue separator (an empty line)
  // into the stream.

  if (fputc('\n', f) < 0)
    return false;

  // A WebVTT Cue comprises 3 things: a cue identifier, followed by
  // the cue timings, followed by the payload of the cue.  We write
  // each part of the cue in sequence.

  if (!WriteCueIdentifier(f, &parser))
    return false;

  if (!WriteCueTimings(f, &parser))
    return false;

  if (!WriteCuePayload(f, &parser))
    return false;

  return true;
}

bool vttdemux::WriteCueIdentifier(FILE* f, FrameParser* parser) {
  string line;
  int e = parser->GetLine(&line);

  if (e)  // error or EOS
    return false;

  // If the cue identifier line is empty, this means that the original
  // WebVTT cue did not have a cue identifier, so we don't bother
  // writing an extra line terminator to the output file (though doing
  // so would be harmless).

  if (!line.empty()) {
    if (fputs(line.c_str(), f) < 0)
      return false;

    if (fputc('\n', f) < 0)
      return false;
  }

  return true;
}

bool vttdemux::WriteCueTimings(FILE* f, FrameParser* parser) {
  const mkvparser::BlockGroup* const block_group = parser->block_group_;
  const mkvparser::Cluster* const cluster = block_group->GetCluster();
  const mkvparser::Block* const block = block_group->GetBlock();

  // A WebVTT Cue "timings" line comprises two parts: the start and
  // stop time for this cue, followed by the (optional) cue settings,
  // such as orientation of the rendered text or its size.  Only the
  // settings part of the cue timings line is stored in the WebM
  // block.  We reconstruct the start and stop times of the WebVTT cue
  // from the timestamp and duration of the WebM block.

  const mkvtime_t start_ns = block->GetTime(cluster);

  if (!WriteCueTime(f, start_ns))
    return false;

  if (fputs(" --> ", f) < 0)
    return false;

  const mkvtime_t duration_timecode = block_group->GetDurationTimeCode();

  if (duration_timecode < 0)
    return false;

  const mkvparser::Segment* const segment = cluster->m_pSegment;
  const mkvparser::SegmentInfo* const info = segment->GetInfo();

  if (info == NULL)
    return false;

  const mkvtime_t timecode_scale = info->GetTimeCodeScale();

  if (timecode_scale <= 0)
    return false;

  const mkvtime_t duration_ns = duration_timecode * timecode_scale;
  const mkvtime_t stop_ns = start_ns + duration_ns;

  if (!WriteCueTime(f, stop_ns))
    return false;

  string line;
  int e = parser->GetLine(&line);

  if (e)  // error or EOS
    return false;

  if (!line.empty()) {
    if (fputc(' ', f) < 0)
      return false;

    if (fputs(line.c_str(), f) < 0)
      return false;
  }

  if (fputc('\n', f) < 0)
    return false;

  return true;
}

bool vttdemux::WriteCueTime(FILE* f, mkvtime_t time_ns) {
  mkvtime_t ms = time_ns / 1000000;  // WebVTT time has millisecond resolution

  mkvtime_t sec = ms / 1000;
  ms -= sec * 1000;

  mkvtime_t min = sec / 60;
  sec -= 60 * min;

  mkvtime_t hr = min / 60;
  min -= 60 * hr;

  if (hr > 0) {
    if (fprintf(f, "%02lld:", hr) < 0)
      return false;
  }

  if (fprintf(f, "%02lld:%02lld.%03lld", min, sec, ms) < 0)
    return false;

  return true;
}

bool vttdemux::WriteCuePayload(FILE* f, FrameParser* parser) {
  int count = 0;  // count of lines of payload text written to output file
  for (string line;;) {
    const int e = parser->GetLine(&line);

    if (e < 0)  // error (only -- we allow EOS here)
      return false;

    if (line.empty())  // TODO(matthewjheaney): retain this check?
      break;

    if (fprintf(f, "%s\n", line.c_str()) < 0)
      return false;

    ++count;
  }

  if (count <= 0)  // WebVTT cue requires non-empty payload
    return false;

  return true;
}

}  // namespace libwebm

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    printf("usage: vttdemux <webmfile>\n");
    return EXIT_SUCCESS;
  }

  const char* const filename = argv[1];
  mkvparser::MkvReader reader;

  int e = reader.Open(filename);

  if (e) {  // error
    printf("unable to open file\n");
    return EXIT_FAILURE;
  }

  libwebm::vttdemux::mkvpos_t pos;

  if (!libwebm::vttdemux::ParseHeader(&reader, &pos))
    return EXIT_FAILURE;

  libwebm::vttdemux::segment_ptr_t segment_ptr;

  if (!libwebm::vttdemux::ParseSegment(&reader, pos, &segment_ptr))
    return EXIT_FAILURE;

  libwebm::vttdemux::metadata_map_t metadata_map;

  BuildMap(segment_ptr.get(), &metadata_map);

  if (metadata_map.empty()) {
    printf("no WebVTT metadata found\n");
    return EXIT_FAILURE;
  }

  if (!OpenFiles(&metadata_map, filename)) {
    CloseFiles(&metadata_map);  // nothing to flush, so not strictly necessary
    return EXIT_FAILURE;
  }

  if (!WriteFiles(metadata_map, segment_ptr.get())) {
    CloseFiles(&metadata_map);  // might as well flush what we do have
    return EXIT_FAILURE;
  }

  CloseFiles(&metadata_map);

  return EXIT_SUCCESS;
}
