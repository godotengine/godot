/*
 * Copyright © 2015-2019  Ebrahim Byagowi
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#include "hb.hh"

#ifdef HAVE_DIRECTWRITE

#include "hb-shaper-impl.hh"

#include <dwrite_1.h>

#include "hb-directwrite.h"

#include "hb-ms-feature-ranges.hh"

/**
 * SECTION:hb-directwrite
 * @title: hb-directwrite
 * @short_description: DirectWrite integration
 * @include: hb-directwrite.h
 *
 * Functions for using HarfBuzz with DirectWrite fonts.
 **/

/* Declare object creator for dynamic support of DWRITE */
typedef HRESULT (* WINAPI t_DWriteCreateFactory)(
  DWRITE_FACTORY_TYPE factoryType,
  REFIID              iid,
  IUnknown            **factory
);


/*
 * DirectWrite font stream helpers
 */

// This is a font loader which provides only one font (unlike its original design).
// For a better implementation which was also source of this
// and DWriteFontFileStream, have a look at to NativeFontResourceDWrite.cpp in Mozilla
class DWriteFontFileLoader : public IDWriteFontFileLoader
{
private:
  IDWriteFontFileStream *mFontFileStream;
public:
  DWriteFontFileLoader (IDWriteFontFileStream *fontFileStream)
  { mFontFileStream = fontFileStream; }

  // IUnknown interface
  IFACEMETHOD (QueryInterface) (IID const& iid, OUT void** ppObject)
  { return S_OK; }
  IFACEMETHOD_ (ULONG, AddRef) ()  { return 1; }
  IFACEMETHOD_ (ULONG, Release) () { return 1; }

  // IDWriteFontFileLoader methods
  virtual HRESULT STDMETHODCALLTYPE
  CreateStreamFromKey (void const* fontFileReferenceKey,
		       uint32_t fontFileReferenceKeySize,
		       OUT IDWriteFontFileStream** fontFileStream)
  {
    *fontFileStream = mFontFileStream;
    return S_OK;
  }

  virtual ~DWriteFontFileLoader() {}
};

class DWriteFontFileStream : public IDWriteFontFileStream
{
private:
  uint8_t *mData;
  uint32_t mSize;
public:
  DWriteFontFileStream (uint8_t *aData, uint32_t aSize)
  {
    mData = aData;
    mSize = aSize;
  }

  // IUnknown interface
  IFACEMETHOD (QueryInterface) (IID const& iid, OUT void** ppObject)
  { return S_OK; }
  IFACEMETHOD_ (ULONG, AddRef) ()  { return 1; }
  IFACEMETHOD_ (ULONG, Release) () { return 1; }

  // IDWriteFontFileStream methods
  virtual HRESULT STDMETHODCALLTYPE
  ReadFileFragment (void const** fragmentStart,
		    UINT64 fileOffset,
		    UINT64 fragmentSize,
		    OUT void** fragmentContext)
  {
    // We are required to do bounds checking.
    if (fileOffset + fragmentSize > mSize) return E_FAIL;

    // truncate the 64 bit fileOffset to size_t sized index into mData
    size_t index = static_cast<size_t> (fileOffset);

    // We should be alive for the duration of this.
    *fragmentStart = &mData[index];
    *fragmentContext = nullptr;
    return S_OK;
  }

  virtual void STDMETHODCALLTYPE
  ReleaseFileFragment (void* fragmentContext) {}

  virtual HRESULT STDMETHODCALLTYPE
  GetFileSize (OUT UINT64* fileSize)
  {
    *fileSize = mSize;
    return S_OK;
  }

  virtual HRESULT STDMETHODCALLTYPE
  GetLastWriteTime (OUT UINT64* lastWriteTime) { return E_NOTIMPL; }

  virtual ~DWriteFontFileStream() {}
};


/*
* shaper face data
*/

struct hb_directwrite_face_data_t
{
  HMODULE dwrite_dll;
  IDWriteFactory *dwriteFactory;
  IDWriteFontFile *fontFile;
  DWriteFontFileStream *fontFileStream;
  DWriteFontFileLoader *fontFileLoader;
  IDWriteFontFace *fontFace;
  hb_blob_t *faceBlob;
};

hb_directwrite_face_data_t *
_hb_directwrite_shaper_face_data_create (hb_face_t *face)
{
  hb_directwrite_face_data_t *data = new hb_directwrite_face_data_t;
  if (unlikely (!data))
    return nullptr;

#define FAIL(...) \
  HB_STMT_START { \
    DEBUG_MSG (DIRECTWRITE, nullptr, __VA_ARGS__); \
    return nullptr; \
  } HB_STMT_END

  data->dwrite_dll = LoadLibrary (TEXT ("DWRITE"));
  if (unlikely (!data->dwrite_dll))
    FAIL ("Cannot find DWrite.DLL");

  t_DWriteCreateFactory p_DWriteCreateFactory;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

  p_DWriteCreateFactory = (t_DWriteCreateFactory)
			  GetProcAddress (data->dwrite_dll, "DWriteCreateFactory");

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  if (unlikely (!p_DWriteCreateFactory))
    FAIL ("Cannot find DWriteCreateFactory().");

  HRESULT hr;

  // TODO: factory and fontFileLoader should be cached separately
  IDWriteFactory* dwriteFactory;
  hr = p_DWriteCreateFactory (DWRITE_FACTORY_TYPE_SHARED, __uuidof (IDWriteFactory),
			      (IUnknown**) &dwriteFactory);

  if (unlikely (hr != S_OK))
    FAIL ("Failed to run DWriteCreateFactory().");

  hb_blob_t *blob = hb_face_reference_blob (face);
  DWriteFontFileStream *fontFileStream;
  fontFileStream = new DWriteFontFileStream ((uint8_t *) hb_blob_get_data (blob, nullptr),
					     hb_blob_get_length (blob));

  DWriteFontFileLoader *fontFileLoader = new DWriteFontFileLoader (fontFileStream);
  dwriteFactory->RegisterFontFileLoader (fontFileLoader);

  IDWriteFontFile *fontFile;
  uint64_t fontFileKey = 0;
  hr = dwriteFactory->CreateCustomFontFileReference (&fontFileKey, sizeof (fontFileKey),
						     fontFileLoader, &fontFile);

  if (FAILED (hr))
    FAIL ("Failed to load font file from data!");

  BOOL isSupported;
  DWRITE_FONT_FILE_TYPE fileType;
  DWRITE_FONT_FACE_TYPE faceType;
  uint32_t numberOfFaces;
  hr = fontFile->Analyze (&isSupported, &fileType, &faceType, &numberOfFaces);
  if (FAILED (hr) || !isSupported)
    FAIL ("Font file is not supported.");

#undef FAIL

  IDWriteFontFace *fontFace;
  dwriteFactory->CreateFontFace (faceType, 1, &fontFile, 0,
				 DWRITE_FONT_SIMULATIONS_NONE, &fontFace);

  data->dwriteFactory = dwriteFactory;
  data->fontFile = fontFile;
  data->fontFileStream = fontFileStream;
  data->fontFileLoader = fontFileLoader;
  data->fontFace = fontFace;
  data->faceBlob = blob;

  return data;
}

void
_hb_directwrite_shaper_face_data_destroy (hb_directwrite_face_data_t *data)
{
  if (data->fontFace)
    data->fontFace->Release ();
  if (data->fontFile)
    data->fontFile->Release ();
  if (data->dwriteFactory)
  {
    if (data->fontFileLoader)
      data->dwriteFactory->UnregisterFontFileLoader (data->fontFileLoader);
    data->dwriteFactory->Release ();
  }
  if (data->fontFileLoader)
    delete data->fontFileLoader;
  if (data->fontFileStream)
    delete data->fontFileStream;
  if (data->faceBlob)
    hb_blob_destroy (data->faceBlob);
  if (data->dwrite_dll)
    FreeLibrary (data->dwrite_dll);
  if (data)
    delete data;
}


/*
 * shaper font data
 */

struct hb_directwrite_font_data_t {};

hb_directwrite_font_data_t *
_hb_directwrite_shaper_font_data_create (hb_font_t *font)
{
  hb_directwrite_font_data_t *data = new hb_directwrite_font_data_t;
  if (unlikely (!data))
    return nullptr;

  return data;
}

void
_hb_directwrite_shaper_font_data_destroy (hb_directwrite_font_data_t *data)
{
  delete data;
}


// Most of TextAnalysis is originally written by Bas Schouten for Mozilla project
// but now is relicensed to MIT for HarfBuzz use
class TextAnalysis : public IDWriteTextAnalysisSource, public IDWriteTextAnalysisSink
{
public:

  IFACEMETHOD (QueryInterface) (IID const& iid, OUT void** ppObject)
  { return S_OK; }
  IFACEMETHOD_ (ULONG, AddRef) () { return 1; }
  IFACEMETHOD_ (ULONG, Release) () { return 1; }

  // A single contiguous run of characters containing the same analysis
  // results.
  struct Run
  {
    uint32_t mTextStart;   // starting text position of this run
    uint32_t mTextLength;  // number of contiguous code units covered
    uint32_t mGlyphStart;  // starting glyph in the glyphs array
    uint32_t mGlyphCount;  // number of glyphs associated with this run
    // text
    DWRITE_SCRIPT_ANALYSIS mScript;
    uint8_t mBidiLevel;
    bool mIsSideways;

    bool ContainsTextPosition (uint32_t aTextPosition) const
    {
      return aTextPosition >= mTextStart &&
	     aTextPosition <  mTextStart + mTextLength;
    }

    Run *nextRun;
  };

public:
  TextAnalysis (const wchar_t* text, uint32_t textLength,
		const wchar_t* localeName, DWRITE_READING_DIRECTION readingDirection)
	       : mTextLength (textLength), mText (text), mLocaleName (localeName),
		 mReadingDirection (readingDirection), mCurrentRun (nullptr) {}
  ~TextAnalysis ()
  {
    // delete runs, except mRunHead which is part of the TextAnalysis object
    for (Run *run = mRunHead.nextRun; run;)
    {
      Run *origRun = run;
      run = run->nextRun;
      delete origRun;
    }
  }

  STDMETHODIMP
  GenerateResults (IDWriteTextAnalyzer* textAnalyzer, Run **runHead)
  {
    // Analyzes the text using the script analyzer and returns
    // the result as a series of runs.

    HRESULT hr = S_OK;

    // Initially start out with one result that covers the entire range.
    // This result will be subdivided by the analysis processes.
    mRunHead.mTextStart = 0;
    mRunHead.mTextLength = mTextLength;
    mRunHead.mBidiLevel =
      (mReadingDirection == DWRITE_READING_DIRECTION_RIGHT_TO_LEFT);
    mRunHead.nextRun = nullptr;
    mCurrentRun = &mRunHead;

    // Call each of the analyzers in sequence, recording their results.
    if (SUCCEEDED (hr = textAnalyzer->AnalyzeScript (this, 0, mTextLength, this)))
      *runHead = &mRunHead;

    return hr;
  }

  // IDWriteTextAnalysisSource implementation

  IFACEMETHODIMP
  GetTextAtPosition (uint32_t textPosition,
		     OUT wchar_t const** textString,
		     OUT uint32_t* textLength)
  {
    if (textPosition >= mTextLength)
    {
      // No text at this position, valid query though.
      *textString = nullptr;
      *textLength = 0;
    }
    else
    {
      *textString = mText + textPosition;
      *textLength = mTextLength - textPosition;
    }
    return S_OK;
  }

  IFACEMETHODIMP
  GetTextBeforePosition (uint32_t textPosition,
			 OUT wchar_t const** textString,
			 OUT uint32_t* textLength)
  {
    if (textPosition == 0 || textPosition > mTextLength)
    {
      // Either there is no text before here (== 0), or this
      // is an invalid position. The query is considered valid though.
      *textString = nullptr;
      *textLength = 0;
    }
    else
    {
      *textString = mText;
      *textLength = textPosition;
    }
    return S_OK;
  }

  IFACEMETHODIMP_ (DWRITE_READING_DIRECTION)
  GetParagraphReadingDirection () { return mReadingDirection; }

  IFACEMETHODIMP GetLocaleName (uint32_t textPosition, uint32_t* textLength,
				wchar_t const** localeName)
  { return S_OK; }

  IFACEMETHODIMP
  GetNumberSubstitution (uint32_t textPosition,
			 OUT uint32_t* textLength,
			 OUT IDWriteNumberSubstitution** numberSubstitution)
  {
    // We do not support number substitution.
    *numberSubstitution = nullptr;
    *textLength = mTextLength - textPosition;

    return S_OK;
  }

  // IDWriteTextAnalysisSink implementation

  IFACEMETHODIMP
  SetScriptAnalysis (uint32_t textPosition, uint32_t textLength,
		     DWRITE_SCRIPT_ANALYSIS const* scriptAnalysis)
  {
    SetCurrentRun (textPosition);
    SplitCurrentRun (textPosition);
    while (textLength > 0)
    {
      Run *run = FetchNextRun (&textLength);
      run->mScript = *scriptAnalysis;
    }

    return S_OK;
  }

  IFACEMETHODIMP
  SetLineBreakpoints (uint32_t textPosition,
		      uint32_t textLength,
		      const DWRITE_LINE_BREAKPOINT* lineBreakpoints)
  { return S_OK; }

  IFACEMETHODIMP SetBidiLevel (uint32_t textPosition, uint32_t textLength,
			       uint8_t explicitLevel, uint8_t resolvedLevel)
  { return S_OK; }

  IFACEMETHODIMP
  SetNumberSubstitution (uint32_t textPosition, uint32_t textLength,
			 IDWriteNumberSubstitution* numberSubstitution)
  { return S_OK; }

protected:
  Run *FetchNextRun (IN OUT uint32_t* textLength)
  {
    // Used by the sink setters, this returns a reference to the next run.
    // Position and length are adjusted to now point after the current run
    // being returned.

    Run *origRun = mCurrentRun;
    // Split the tail if needed (the length remaining is less than the
    // current run's size).
    if (*textLength < mCurrentRun->mTextLength)
      SplitCurrentRun (mCurrentRun->mTextStart + *textLength);
    else
      // Just advance the current run.
      mCurrentRun = mCurrentRun->nextRun;
    *textLength -= origRun->mTextLength;

    // Return a reference to the run that was just current.
    return origRun;
  }

  void SetCurrentRun (uint32_t textPosition)
  {
    // Move the current run to the given position.
    // Since the analyzers generally return results in a forward manner,
    // this will usually just return early. If not, find the
    // corresponding run for the text position.

    if (mCurrentRun && mCurrentRun->ContainsTextPosition (textPosition))
      return;

    for (Run *run = &mRunHead; run; run = run->nextRun)
      if (run->ContainsTextPosition (textPosition))
      {
	mCurrentRun = run;
	return;
      }
    assert (0); // We should always be able to find the text position in one of our runs
  }

  void SplitCurrentRun (uint32_t splitPosition)
  {
    if (!mCurrentRun)
    {
      assert (0); // SplitCurrentRun called without current run
      // Shouldn't be calling this when no current run is set!
      return;
    }
    // Split the current run.
    if (splitPosition <= mCurrentRun->mTextStart)
    {
      // No need to split, already the start of a run
      // or before it. Usually the first.
      return;
    }
    Run *newRun = new Run;

    *newRun = *mCurrentRun;

    // Insert the new run in our linked list.
    newRun->nextRun = mCurrentRun->nextRun;
    mCurrentRun->nextRun = newRun;

    // Adjust runs' text positions and lengths.
    uint32_t splitPoint = splitPosition - mCurrentRun->mTextStart;
    newRun->mTextStart += splitPoint;
    newRun->mTextLength -= splitPoint;
    mCurrentRun->mTextLength = splitPoint;
    mCurrentRun = newRun;
  }

protected:
  // Input
  // (weak references are fine here, since this class is a transient
  //  stack-based helper that doesn't need to copy data)
  uint32_t mTextLength;
  const wchar_t* mText;
  const wchar_t* mLocaleName;
  DWRITE_READING_DIRECTION mReadingDirection;

  // Current processing state.
  Run *mCurrentRun;

  // Output is a list of runs starting here
  Run  mRunHead;
};

/*
 * shaper
 */

hb_bool_t
_hb_directwrite_shape (hb_shape_plan_t    *shape_plan,
		       hb_font_t          *font,
		       hb_buffer_t        *buffer,
		       const hb_feature_t *features,
		       unsigned int        num_features)
{
  hb_face_t *face = font->face;
  const hb_directwrite_face_data_t *face_data = face->data.directwrite;
  IDWriteFactory *dwriteFactory = face_data->dwriteFactory;
  IDWriteFontFace *fontFace = face_data->fontFace;

  IDWriteTextAnalyzer* analyzer;
  dwriteFactory->CreateTextAnalyzer (&analyzer);

  unsigned int scratch_size;
  hb_buffer_t::scratch_buffer_t *scratch = buffer->get_scratch_buffer (&scratch_size);
#define ALLOCATE_ARRAY(Type, name, len) \
  Type *name = (Type *) scratch; \
  do { \
    unsigned int _consumed = DIV_CEIL ((len) * sizeof (Type), sizeof (*scratch)); \
    assert (_consumed <= scratch_size); \
    scratch += _consumed; \
    scratch_size -= _consumed; \
  } while (0)

#define utf16_index() var1.u32

  ALLOCATE_ARRAY (wchar_t, textString, buffer->len * 2);

  unsigned int chars_len = 0;
  for (unsigned int i = 0; i < buffer->len; i++)
  {
    hb_codepoint_t c = buffer->info[i].codepoint;
    buffer->info[i].utf16_index () = chars_len;
    if (likely (c <= 0xFFFFu))
      textString[chars_len++] = c;
    else if (unlikely (c > 0x10FFFFu))
      textString[chars_len++] = 0xFFFDu;
    else
    {
      textString[chars_len++] = 0xD800u + ((c - 0x10000u) >> 10);
      textString[chars_len++] = 0xDC00u + ((c - 0x10000u) & ((1u << 10) - 1));
    }
  }

  ALLOCATE_ARRAY (WORD, log_clusters, chars_len);
  /* Need log_clusters to assign features. */
  chars_len = 0;
  for (unsigned int i = 0; i < buffer->len; i++)
  {
    hb_codepoint_t c = buffer->info[i].codepoint;
    unsigned int cluster = buffer->info[i].cluster;
    log_clusters[chars_len++] = cluster;
    if (hb_in_range (c, 0x10000u, 0x10FFFFu))
      log_clusters[chars_len++] = cluster; /* Surrogates. */
  }

  DWRITE_READING_DIRECTION readingDirection;
  readingDirection = buffer->props.direction ?
		     DWRITE_READING_DIRECTION_RIGHT_TO_LEFT :
		     DWRITE_READING_DIRECTION_LEFT_TO_RIGHT;

  /*
  * There's an internal 16-bit limit on some things inside the analyzer,
  * but we never attempt to shape a word longer than 64K characters
  * in a single gfxShapedWord, so we cannot exceed that limit.
  */
  uint32_t textLength = chars_len;

  TextAnalysis analysis (textString, textLength, nullptr, readingDirection);
  TextAnalysis::Run *runHead;
  HRESULT hr;
  hr = analysis.GenerateResults (analyzer, &runHead);

#define FAIL(...) \
  HB_STMT_START { \
    DEBUG_MSG (DIRECTWRITE, nullptr, __VA_ARGS__); \
    return false; \
  } HB_STMT_END

  if (FAILED (hr))
    FAIL ("Analyzer failed to generate results.");

  uint32_t maxGlyphCount = 3 * textLength / 2 + 16;
  uint32_t glyphCount;
  bool isRightToLeft = HB_DIRECTION_IS_BACKWARD (buffer->props.direction);

  const wchar_t localeName[20] = {0};
  if (buffer->props.language)
    mbstowcs ((wchar_t*) localeName,
	      hb_language_to_string (buffer->props.language), 20);

  /*
   * Set up features.
   */
  static_assert ((sizeof (DWRITE_TYPOGRAPHIC_FEATURES) == sizeof (hb_ms_features_t)), "");
  static_assert ((sizeof (DWRITE_FONT_FEATURE) == sizeof (hb_ms_feature_t)), "");
  hb_vector_t<hb_ms_features_t *> range_features;
  hb_vector_t<uint32_t> range_char_counts;
  if (num_features)
  {
    hb_vector_t<hb_ms_feature_t> feature_records;
    hb_vector_t<hb_ms_range_record_t> range_records;
    if (hb_ms_setup_features (features, num_features, feature_records, range_records))
      hb_ms_make_feature_ranges (feature_records,
				 range_records,
				 0,
				 chars_len,
				 log_clusters,
				 range_features,
				 range_char_counts);
  }

  uint16_t* clusterMap;
  clusterMap = new uint16_t[textLength];
  DWRITE_SHAPING_TEXT_PROPERTIES* textProperties;
  textProperties = new DWRITE_SHAPING_TEXT_PROPERTIES[textLength];

retry_getglyphs:
  uint16_t* glyphIndices = new uint16_t[maxGlyphCount];
  DWRITE_SHAPING_GLYPH_PROPERTIES* glyphProperties;
  glyphProperties = new DWRITE_SHAPING_GLYPH_PROPERTIES[maxGlyphCount];

  hr = analyzer->GetGlyphs (textString,
			    chars_len,
			    fontFace,
			    false,
			    isRightToLeft,
			    &runHead->mScript,
			    localeName,
			    nullptr,
			    (const DWRITE_TYPOGRAPHIC_FEATURES**) range_features.arrayZ,
			    range_char_counts.arrayZ,
			    range_features.length,
			    maxGlyphCount,
			    clusterMap,
			    textProperties,
			    glyphIndices,
			    glyphProperties,
			    &glyphCount);

  if (unlikely (hr == HRESULT_FROM_WIN32 (ERROR_INSUFFICIENT_BUFFER)))
  {
    delete [] glyphIndices;
    delete [] glyphProperties;

    maxGlyphCount *= 2;

    goto retry_getglyphs;
  }
  if (FAILED (hr))
    FAIL ("Analyzer failed to get glyphs.");

  float* glyphAdvances = new float[maxGlyphCount];
  DWRITE_GLYPH_OFFSET* glyphOffsets = new DWRITE_GLYPH_OFFSET[maxGlyphCount];

  /* The -2 in the following is to compensate for possible
   * alignment needed after the WORD array.  sizeof (WORD) == 2. */
  unsigned int glyphs_size = (scratch_size * sizeof (int) - 2)
			     / (sizeof (WORD) +
				sizeof (DWRITE_SHAPING_GLYPH_PROPERTIES) +
				sizeof (int) +
				sizeof (DWRITE_GLYPH_OFFSET) +
				sizeof (uint32_t));
  ALLOCATE_ARRAY (uint32_t, vis_clusters, glyphs_size);

#undef ALLOCATE_ARRAY

  int fontEmSize = font->face->get_upem ();
  if (fontEmSize < 0) fontEmSize = -fontEmSize;

  if (fontEmSize < 0) fontEmSize = -fontEmSize;
  double x_mult = (double) font->x_scale / fontEmSize;
  double y_mult = (double) font->y_scale / fontEmSize;

  hr = analyzer->GetGlyphPlacements (textString,
				     clusterMap,
				     textProperties,
				     chars_len,
				     glyphIndices,
				     glyphProperties,
				     glyphCount,
				     fontFace,
				     fontEmSize,
				     false,
				     isRightToLeft,
				     &runHead->mScript,
				     localeName,
				     (const DWRITE_TYPOGRAPHIC_FEATURES**) range_features.arrayZ,
				     range_char_counts.arrayZ,
				     range_features.length,
				     glyphAdvances,
				     glyphOffsets);

  if (FAILED (hr))
    FAIL ("Analyzer failed to get glyph placements.");

  /* Ok, we've got everything we need, now compose output buffer,
   * very, *very*, carefully! */

  /* Calculate visual-clusters.  That's what we ship. */
  for (unsigned int i = 0; i < glyphCount; i++)
    vis_clusters[i] = (uint32_t) -1;
  for (unsigned int i = 0; i < buffer->len; i++)
  {
    uint32_t *p =
      &vis_clusters[log_clusters[buffer->info[i].utf16_index ()]];
    *p = hb_min (*p, buffer->info[i].cluster);
  }
  for (unsigned int i = 1; i < glyphCount; i++)
    if (vis_clusters[i] == (uint32_t) -1)
      vis_clusters[i] = vis_clusters[i - 1];

#undef utf16_index

  if (unlikely (!buffer->ensure (glyphCount)))
    FAIL ("Buffer in error");

#undef FAIL

  /* Set glyph infos */
  buffer->len = 0;
  for (unsigned int i = 0; i < glyphCount; i++)
  {
    hb_glyph_info_t *info = &buffer->info[buffer->len++];

    info->codepoint = glyphIndices[i];
    info->cluster = vis_clusters[i];

    /* The rest is crap.  Let's store position info there for now. */
    info->mask = glyphAdvances[i];
    info->var1.i32 = glyphOffsets[i].advanceOffset;
    info->var2.i32 = glyphOffsets[i].ascenderOffset;
  }

  /* Set glyph positions */
  buffer->clear_positions ();
  for (unsigned int i = 0; i < glyphCount; i++)
  {
    hb_glyph_info_t *info = &buffer->info[i];
    hb_glyph_position_t *pos = &buffer->pos[i];

    /* TODO vertical */
    pos->x_advance = x_mult * (int32_t) info->mask;
    pos->x_offset = x_mult * (isRightToLeft ? -info->var1.i32 : info->var1.i32);
    pos->y_offset = y_mult * info->var2.i32;
  }

  if (isRightToLeft) hb_buffer_reverse (buffer);

  delete [] clusterMap;
  delete [] glyphIndices;
  delete [] textProperties;
  delete [] glyphProperties;
  delete [] glyphAdvances;
  delete [] glyphOffsets;

  /* Wow, done! */
  return true;
}

struct _hb_directwrite_font_table_context {
  IDWriteFontFace *face;
  void *table_context;
};

static void
_hb_directwrite_table_data_release (void *data)
{
  _hb_directwrite_font_table_context *context = (_hb_directwrite_font_table_context *) data;
  context->face->ReleaseFontTable (context->table_context);
  hb_free (context);
}

static hb_blob_t *
_hb_directwrite_reference_table (hb_face_t *face HB_UNUSED, hb_tag_t tag, void *user_data)
{
  IDWriteFontFace *dw_face = ((IDWriteFontFace *) user_data);
  const void *data;
  uint32_t length;
  void *table_context;
  BOOL exists;
  if (!dw_face || FAILED (dw_face->TryGetFontTable (hb_uint32_swap (tag), &data,
						    &length, &table_context, &exists)))
    return nullptr;

  if (!data || !exists || !length)
  {
    dw_face->ReleaseFontTable (table_context);
    return nullptr;
  }

  _hb_directwrite_font_table_context *context = (_hb_directwrite_font_table_context *) hb_malloc (sizeof (_hb_directwrite_font_table_context));
  context->face = dw_face;
  context->table_context = table_context;

  return hb_blob_create ((const char *) data, length, HB_MEMORY_MODE_READONLY,
			 context, _hb_directwrite_table_data_release);
}

static void
_hb_directwrite_font_release (void *data)
{
  if (data)
    ((IDWriteFontFace *) data)->Release ();
}

/**
 * hb_directwrite_face_create:
 * @font_face: a DirectWrite IDWriteFontFace object.
 *
 * Constructs a new face object from the specified DirectWrite IDWriteFontFace.
 *
 * Return value: #hb_face_t object corresponding to the given input
 *
 * Since: 2.4.0
 **/
hb_face_t *
hb_directwrite_face_create (IDWriteFontFace *font_face)
{
  if (font_face)
    font_face->AddRef ();
  return hb_face_create_for_tables (_hb_directwrite_reference_table, font_face,
				    _hb_directwrite_font_release);
}

/**
* hb_directwrite_face_get_font_face:
* @face: a #hb_face_t object
*
* Gets the DirectWrite IDWriteFontFace associated with @face.
*
* Return value: DirectWrite IDWriteFontFace object corresponding to the given input
*
* Since: 2.5.0
**/
IDWriteFontFace *
hb_directwrite_face_get_font_face (hb_face_t *face)
{
  return face->data.directwrite->fontFace;
}


#endif
