// SPDX-License-Identifier: Apache 2.0
// Copyright 2021 - 2022, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// USD ASCII parser

#pragma once

// #include <functional>
#include <stdio.h>

#include <stack>

// #include "external/better-enums/enum.h"
#include "composition.hh"
#include "prim-types.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"

//
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// external
#include "nonstd/expected.hpp"

//
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace tinyusdz {

namespace ascii {

// keywords
constexpr auto kUniform = "uniform";
constexpr auto kToken = "token";

// Frequently used attr/meta keywords
constexpr auto kKind = "kind";
constexpr auto kInterpolation = "interpolation";

struct Identifier : std::string {
  // using std::string;
};

// FIXME: Not used? remove.
struct PathIdentifier : std::string {
  // using std::string;
};

// Parser option.
// For strict configuration(e.g. read USDZ on Mobile), should disallow unknown
// items.
struct AsciiParserOption {
  bool allow_unknown_prim{true};
  bool allow_unknown_apiSchema{true};
  bool strict_allowedToken_check{false};
};

///
/// Test if input file is USDA ascii format.
///
bool IsUSDA(const std::string &filename, size_t max_filesize = 0);

class AsciiParser {
 public:
  // TODO: refactor
  struct PrimMetas {
    // Frequently used prim metas
    nonstd::optional<Kind> kind;

    value::dict customData;  // `customData`
    std::vector<value::StringData>
        strings;  // String only unregistered metadata.
  };

  // TODO: Unifity class with StageMetas in prim-types.hh
  struct StageMetas {
    ///
    /// Predefined Stage metas
    ///
    std::vector<value::AssetPath> subLayers;  // 'subLayers'
    value::token defaultPrim;                 // 'defaultPrim'
    value::StringData doc;                    // 'doc' or 'documentation'
    nonstd::optional<Axis> upAxis;            // not specified = nullopt
    nonstd::optional<double> metersPerUnit;
    nonstd::optional<double> kilogramsPerUnit;
    nonstd::optional<double> timeCodesPerSecond;
    nonstd::optional<double> startTimeCode;
    nonstd::optional<double> endTimeCode;
    nonstd::optional<double> framesPerSecond;

    nonstd::optional<bool> autoPlay;
    nonstd::optional<value::token> playbackMode;  // 'none' or 'loop'

    std::map<std::string, MetaVariable> customLayerData;  // `customLayerData`.
    value::StringData comment;  // String only comment string.
  };

  struct ParseState {
    int64_t loc{-1};  // byte location in StreamReder
  };

  struct Cursor {
    int row{0};
    int col{0};
  };

  struct ErrorDiagnostic {
    std::string err;
    Cursor cursor;
  };

  void PushError(const std::string &msg) {
    ErrorDiagnostic diag;
    diag.cursor.row = _curr_cursor.row;
    diag.cursor.col = _curr_cursor.col;
    diag.err = msg;
    err_stack.push(diag);
  }

  // This function is used to cancel recent parsing error.
  void PopError() {
    if (!err_stack.empty()) {
      err_stack.pop();
    }
  }

  void PushWarn(const std::string &msg) {
    ErrorDiagnostic diag;
    diag.cursor.row = _curr_cursor.row;
    diag.cursor.col = _curr_cursor.col;
    diag.err = msg;
    warn_stack.push(diag);
  }

  // This function is used to cancel recent parsing warning.
  void PopWarn() {
    if (!warn_stack.empty()) {
      warn_stack.pop();
    }
  }

  bool IsStageMeta(const std::string &name);
  bool IsRegisteredPrimMeta(const std::string &name);

  class VariableDef {
   public:
    // Handler functor in post parsing stage.
    // e.g. Check input string is a valid one: one of "common", "group",
    // "assembly", "component" or "subcomponent" for "kind" metadata
    using PostParseHandler =
        std::function<nonstd::expected<bool, std::string>(const std::string &)>;

    static nonstd::expected<bool, std::string> DefaultPostParseHandler(
        const std::string &) {
      return true;
    }

    std::string type;  // e.g. token, color3f
    std::string name;
    bool allow_array_type{false};  // when true, we accept `type` and `type[]`

    PostParseHandler post_parse_handler;

    VariableDef() = default;

    VariableDef(const std::string &t, const std::string &n, bool a = false,
                PostParseHandler ph = DefaultPostParseHandler)
        : type(t), name(n), allow_array_type(a), post_parse_handler(ph) {}

    VariableDef(const VariableDef &rhs) = default;
    VariableDef &operator=(const VariableDef &rhs) = default;

    // VariableDef &operator=(const VariableDef &rhs) {
    //   type = rhs.type;
    //   name = rhs.name;
    //   parse_handler = rhs.parse_handler;

    //  return *this;
    //}
  };

  using PrimMetaMap =
      std::map<std::string, std::pair<ListEditQual, MetaVariable>>;

  struct VariantContent {
    PrimMetaMap metas;
    std::vector<int64_t> primIndices;  // primIdx of Reconstrcuted Prim.
    std::map<std::string, Property> props;
    std::vector<value::token> properties;

    // for nested `variantSet` 
    std::map<std::string, std::map<std::string, VariantContent>> variantSets;
  };

  // TODO: Use std::vector instead of std::map?
  using VariantSetList =
      std::map<std::string, std::map<std::string, VariantContent>>;

  AsciiParser();
  AsciiParser(tinyusdz::StreamReader *sr);

  AsciiParser(const AsciiParser &rhs) = delete;
  AsciiParser(AsciiParser &&rhs) = delete;

  ~AsciiParser();

  ///
  /// Assign index to primitive for index-based prim scene graph representation.
  /// -1 = root
  ///
  using PrimIdxAssignFunctin = std::function<int64_t(const int64_t parentIdx)>;
  void RegisterPrimIdxAssignFunction(PrimIdxAssignFunctin fun) {
    _prim_idx_assign_fun = fun;
  }

  ///
  /// Stage Meta construction callback function
  ///
  using StageMetaProcessFunction = std::function<bool(const StageMetas &metas)>;

  ///
  /// Register Stage metadatum processing callback function.
  /// Called when after parsing Stage metadatum.
  ///
  void RegisterStageMetaProcessFunction(StageMetaProcessFunction fun) {
    _stage_meta_process_fun = fun;
  }

  ///
  /// Prim Meta construction callback function
  ///
  // using PrimMetaProcessFunction = std::function<bool(const PrimMetas
  // &metas)>;

  ///
  /// Prim construction callback function
  /// TODO: Refactor arguments
  ///
  /// @param[in] full_path Absolute Prim Path(e.g. "/scope/gmesh0")
  /// @param[in] spec Specifier(`def`, `over` or `class`)
  /// @param[in] primTypeName typeName of this Prim(e.g. "Mesh", "SphereLight")
  /// @param[in] primIdx primitive index
  /// @param[in] parentPrimIdx parent Prim index. -1 for root
  /// @param[in] properties Prim properties
  /// @param[in] in_meta Input Prim metadataum
  /// @param[in] in_variantSetList Input VariantSet contents.
  /// @return true upon success or error message.
  ///
  using PrimConstructFunction =
      std::function<nonstd::expected<bool, std::string>(
          const Path &full_path, const Specifier spec,
          const std::string &primTypeName, const Path &prim_name,
          const int64_t primIdx, const int64_t parentPrimIdx,
          const std::map<std::string, Property> &properties,
          const PrimMetaMap &in_meta, const VariantSetList &in_variantSetList)>;

  ///
  /// Register Prim construction callback function.
  /// Example: "Xform", ReconstrctXform
  ///
  void RegisterPrimConstructFunction(const std::string &prim_type,
                                     PrimConstructFunction fun) {
    _prim_construct_fun_map[prim_type] = fun;
  }

  ///
  /// Callbacks called at closing `def` block.
  ///
  using PostPrimConstructFunction =
      std::function<nonstd::expected<bool, std::string>(
          const Path &path, const int64_t primIdx,
          const int64_t parentPrimIdx)>;
  void RegisterPostPrimConstructFunction(const std::string &prim_type,
                                         PostPrimConstructFunction fun) {
    _post_prim_construct_fun_map[prim_type] = fun;
  }

  ///
  /// For composition(Treat Prim as generic container).
  /// AsciiParser(i.e. USDAReader)
  ///
  using PrimSpecFunction = std::function<nonstd::expected<bool, std::string>(
      const Path &full_path, const Specifier spec,
      const std::string &primTypeName, const Path &prim_name,
      const int64_t primIdx, const int64_t parentPrimIdx,
      const std::map<std::string, Property> &properties,
      const PrimMetaMap &in_meta, const VariantSetList &in_variantSetLists)>;

  void RegisterPrimSpecFunction(PrimSpecFunction fun) { _primspec_fun = fun; }

  ///
  /// Base filesystem directory to search asset files.
  ///
  void SetBaseDir(const std::string &base_dir);

  ///
  /// Set ASCII data stream
  ///
  void SetStream(tinyusdz::StreamReader *sr);

  ///
  /// Check if header data is USDA
  ///
  bool CheckHeader();

  ///
  /// True: create PrimSpec instead of typed Prim.
  /// Set true if you do USD composition.
  ///
  void set_primspec_mode(bool onoff) { _primspec_mode = onoff; }

  ///
  /// Parser entry point
  ///
  /// @param[in] load_states Bit mask of LoadState
  /// @param[in] parser_option Parse option(optional)
  ///
  /// TODO: Move `load_states` to AsciiParserOption?
  ///
  bool Parse(
      const uint32_t load_states = static_cast<uint32_t>(LoadState::Toplevel),
      const AsciiParserOption &parser_option = AsciiParserOption());

  ///
  /// Parse TimeSample value with specified array type of
  /// `type_id`(value::TypeId) (You can obrain type_id from string using
  /// value::GetTypeId())
  ///
  bool ParseTimeSampleValue(const uint32_t type_id, value::Value *result);

  ///
  /// Parse TimeSample value with specified `type_name`(Appears in USDA. .e.g.
  /// "float", "matrix2d")
  ///
  bool ParseTimeSampleValue(const std::string &type_name, value::Value *result);

  ///
  /// Parse TimeSample value with specified base type of
  /// `type_id`(value::TypeId) (You can obrain type_id from string using
  /// value::GetTypeId())
  ///
  bool ParseTimeSampleValueOfArrayType(const uint32_t base_type_id,
                                       value::Value *result);

  ///
  /// Parse TimeSample value with specified array type of `type_name`("[]"
  /// omiotted. .e.g. "float" for "float[]")
  ///
  bool ParseTimeSampleValueOfArrayType(const std::string &type_name,
                                       value::Value *result);

  // TODO: ParseBasicType?
  bool ParsePurpose(Purpose *result);

  ///
  /// Return true but `value` is set to nullopt for `None`(Attribute Blocked)
  ///
  // template <typename T>
  // bool ReadBasicType(nonstd::optional<T> *value);

  bool ReadBasicType(nonstd::optional<bool> *value);
  bool ReadBasicType(nonstd::optional<value::half> *value);
  bool ReadBasicType(nonstd::optional<value::half2> *value);
  bool ReadBasicType(nonstd::optional<value::half3> *value);
  bool ReadBasicType(nonstd::optional<value::half4> *value);
  bool ReadBasicType(nonstd::optional<int32_t> *value);
  bool ReadBasicType(nonstd::optional<value::int2> *value);
  bool ReadBasicType(nonstd::optional<value::int3> *value);
  bool ReadBasicType(nonstd::optional<value::int4> *value);
  bool ReadBasicType(nonstd::optional<uint32_t> *value);
  bool ReadBasicType(nonstd::optional<value::uint2> *value);
  bool ReadBasicType(nonstd::optional<value::uint3> *value);
  bool ReadBasicType(nonstd::optional<value::uint4> *value);
  bool ReadBasicType(nonstd::optional<int64_t> *value);
  bool ReadBasicType(nonstd::optional<uint64_t> *value);
  bool ReadBasicType(nonstd::optional<float> *value);
  bool ReadBasicType(nonstd::optional<value::float2> *value);
  bool ReadBasicType(nonstd::optional<value::float3> *value);
  bool ReadBasicType(nonstd::optional<value::float4> *value);
  bool ReadBasicType(nonstd::optional<double> *value);
  bool ReadBasicType(nonstd::optional<value::double2> *value);
  bool ReadBasicType(nonstd::optional<value::double3> *value);
  bool ReadBasicType(nonstd::optional<value::double4> *value);
  bool ReadBasicType(nonstd::optional<value::quath> *value);
  bool ReadBasicType(nonstd::optional<value::quatf> *value);
  bool ReadBasicType(nonstd::optional<value::quatd> *value);
  bool ReadBasicType(nonstd::optional<value::point3h> *value);
  bool ReadBasicType(nonstd::optional<value::point3f> *value);
  bool ReadBasicType(nonstd::optional<value::point3d> *value);
  bool ReadBasicType(nonstd::optional<value::vector3h> *value);
  bool ReadBasicType(nonstd::optional<value::vector3f> *value);
  bool ReadBasicType(nonstd::optional<value::vector3d> *value);
  bool ReadBasicType(nonstd::optional<value::normal3h> *value);
  bool ReadBasicType(nonstd::optional<value::normal3f> *value);
  bool ReadBasicType(nonstd::optional<value::normal3d> *value);
  bool ReadBasicType(nonstd::optional<value::color3h> *value);
  bool ReadBasicType(nonstd::optional<value::color3f> *value);
  bool ReadBasicType(nonstd::optional<value::color3d> *value);
  bool ReadBasicType(nonstd::optional<value::color4h> *value);
  bool ReadBasicType(nonstd::optional<value::color4f> *value);
  bool ReadBasicType(nonstd::optional<value::color4d> *value);
  bool ReadBasicType(nonstd::optional<value::matrix2f> *value);
  bool ReadBasicType(nonstd::optional<value::matrix3f> *value);
  bool ReadBasicType(nonstd::optional<value::matrix4f> *value);
  bool ReadBasicType(nonstd::optional<value::matrix2d> *value);
  bool ReadBasicType(nonstd::optional<value::matrix3d> *value);
  bool ReadBasicType(nonstd::optional<value::matrix4d> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord2h> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord2f> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord2d> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord3h> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord3f> *value);
  bool ReadBasicType(nonstd::optional<value::texcoord3d> *value);
  bool ReadBasicType(nonstd::optional<value::StringData> *value);
  bool ReadBasicType(nonstd::optional<std::string> *value);
  bool ReadBasicType(nonstd::optional<value::token> *value);
  bool ReadBasicType(nonstd::optional<Path> *value);
  bool ReadBasicType(nonstd::optional<value::AssetPath> *value);
  bool ReadBasicType(nonstd::optional<Reference> *value);
  bool ReadBasicType(nonstd::optional<Payload> *value);
  bool ReadBasicType(nonstd::optional<Identifier> *value);
  bool ReadBasicType(nonstd::optional<PathIdentifier> *value);

  // template <typename T>
  // bool ReadBasicType(T *value);

  bool ReadBasicType(bool *value);
  bool ReadBasicType(value::half *value);
  bool ReadBasicType(value::half2 *value);
  bool ReadBasicType(value::half3 *value);
  bool ReadBasicType(value::half4 *value);
  bool ReadBasicType(int32_t *value);
  bool ReadBasicType(value::int2 *value);
  bool ReadBasicType(value::int3 *value);
  bool ReadBasicType(value::int4 *value);
  bool ReadBasicType(uint32_t *value);
  bool ReadBasicType(value::uint2 *value);
  bool ReadBasicType(value::uint3 *value);
  bool ReadBasicType(value::uint4 *value);
  bool ReadBasicType(int64_t *value);
  bool ReadBasicType(uint64_t *value);
  bool ReadBasicType(float *value);
  bool ReadBasicType(value::float2 *value);
  bool ReadBasicType(value::float3 *value);
  bool ReadBasicType(value::float4 *value);
  bool ReadBasicType(double *value);
  bool ReadBasicType(value::double2 *value);
  bool ReadBasicType(value::double3 *value);
  bool ReadBasicType(value::double4 *value);
  bool ReadBasicType(value::quath *value);
  bool ReadBasicType(value::quatf *value);
  bool ReadBasicType(value::quatd *value);
  bool ReadBasicType(value::point3h *value);
  bool ReadBasicType(value::point3f *value);
  bool ReadBasicType(value::point3d *value);
  bool ReadBasicType(value::vector3h *value);
  bool ReadBasicType(value::vector3f *value);
  bool ReadBasicType(value::vector3d *value);
  bool ReadBasicType(value::normal3h *value);
  bool ReadBasicType(value::normal3f *value);
  bool ReadBasicType(value::normal3d *value);
  bool ReadBasicType(value::color3h *value);
  bool ReadBasicType(value::color3f *value);
  bool ReadBasicType(value::color3d *value);
  bool ReadBasicType(value::color4h *value);
  bool ReadBasicType(value::color4f *value);
  bool ReadBasicType(value::color4d *value);
  bool ReadBasicType(value::texcoord2h *value);
  bool ReadBasicType(value::texcoord2f *value);
  bool ReadBasicType(value::texcoord2d *value);
  bool ReadBasicType(value::texcoord3h *value);
  bool ReadBasicType(value::texcoord3f *value);
  bool ReadBasicType(value::texcoord3d *value);
  bool ReadBasicType(value::matrix2f *value);
  bool ReadBasicType(value::matrix3f *value);
  bool ReadBasicType(value::matrix4f *value);
  bool ReadBasicType(value::matrix2d *value);
  bool ReadBasicType(value::matrix3d *value);
  bool ReadBasicType(value::matrix4d *value);
  bool ReadBasicType(value::StringData *value);
  bool ReadBasicType(std::string *value);
  bool ReadBasicType(value::token *value);
  bool ReadBasicType(Path *value);
  bool ReadBasicType(value::AssetPath *value);
  bool ReadBasicType(Reference *value);
  bool ReadBasicType(Payload *value);
  bool ReadBasicType(Identifier *value);
  bool ReadBasicType(PathIdentifier *value);

  template <typename T>
  bool ReadBasicType(nonstd::optional<std::vector<T>> *value);

  template <typename T>
  bool ReadBasicType(std::vector<T> *value);

  bool ParseMatrix(value::matrix2f *result);
  bool ParseMatrix(value::matrix3f *result);
  bool ParseMatrix(value::matrix4f *result);

  bool ParseMatrix(value::matrix2d *result);
  bool ParseMatrix(value::matrix3d *result);
  bool ParseMatrix(value::matrix4d *result);

  ///
  /// Parse '(', Sep1By(','), ')'
  ///
  template <typename T, size_t N>
  bool ParseBasicTypeTuple(std::array<T, N> *result);

  ///
  /// Parse '(', Sep1By(','), ')'
  /// Can have `None`
  ///
  template <typename T, size_t N>
  bool ParseBasicTypeTuple(nonstd::optional<std::array<T, N>> *result);

  template <typename T, size_t N>
  bool ParseTupleArray(std::vector<std::array<T, N>> *result);

  ///
  /// Parse the array of tuple. some may be None(e.g. `float3`: [(0, 1, 2),
  /// None, (2, 3, 4), ...] )
  ///
  template <typename T, size_t N>
  bool ParseTupleArray(std::vector<nonstd::optional<std::array<T, N>>> *result);

  template <typename T>
  bool SepBy1BasicType(const char sep, std::vector<T> *result);

  ///
  /// Allow the appearance of `sep` in the last item of array.
  /// (e.g. `[1, 2, 3,]`)
  ///
  template <typename T>
  bool SepBy1BasicType(const char sep, const char end_symbol,
                       std::vector<T> *result);

  ///
  /// Parse '[', Sep1By(','), ']'
  ///
  template <typename T>
  bool ParseBasicTypeArray(std::vector<nonstd::optional<T>> *result);

  ///
  /// Parse '[', Sep1By(','), ']'
  ///
  template <typename T>
  bool ParseBasicTypeArray(std::vector<T> *result);

  ///
  /// Parses 1 or more occurences of value with basic type 'T', separated by
  /// `sep`
  ///
  template <typename T>
  bool SepBy1BasicType(const char sep,
                       std::vector<nonstd::optional<T>> *result);

  ///
  /// Parses 1 or more occurences of tuple values with type 'T', separated by
  /// `sep`. Allows 'None'
  ///
  template <typename T, size_t N>
  bool SepBy1TupleType(const char sep,
                       std::vector<nonstd::optional<std::array<T, N>>> *result);

  ///
  /// Parses N occurences of tuple values with type 'T', separated by
  /// `sep`. Allows 'None'
  ///
  template <typename T, size_t M, size_t N>
  bool SepByNTupleType(
      const char sep,
      std::array<nonstd::optional<std::array<T, M>>, N> *result);

  ///
  /// Parses 1 or more occurences of tuple values with type 'T', separated by
  /// `sep`
  ///
  template <typename T, size_t N>
  bool SepBy1TupleType(const char sep, std::vector<std::array<T, N>> *result);

  bool ParseDictElement(std::string *out_key, MetaVariable *out_var);
  bool ParseDict(std::map<std::string, MetaVariable> *out_dict);

  ///
  /// Parse TimeSample data(scalar type) and store it to type-erased data
  /// structure value::TimeSamples.
  ///
  /// @param[in] type_name Name of TimeSamples type(seen in .usda file. e.g.
  /// "float" for `float var.timeSamples = ..`)
  ///
  bool ParseTimeSamples(const std::string &type_name, value::TimeSamples *ts);

  ///
  /// Parse TimeSample data(array type) and store it to type-erased data
  /// structure value::TimeSamples.
  ///
  /// @param[in] type_name Name of TimeSamples type(seen in .usda file. array
  /// suffix `[]` is omitted. e.g. "float" for `float[] var.timeSamples = ..`)
  ///
  bool ParseTimeSamplesOfArray(const std::string &type_name,
                               value::TimeSamples *ts);

  ///
  /// `variants` in Prim meta.
  ///
  bool ParseVariantsElement(std::string *out_key, std::string *out_var);
  bool ParseVariants(VariantSelectionMap *out_map);

  bool MaybeListEditQual(tinyusdz::ListEditQual *qual);
  bool MaybeVariability(tinyusdz::Variability *variability,
                        bool *varying_authored);

  ///
  /// Try parsing single-quoted(`"`) string
  ///
  bool MaybeString(value::StringData *str);

  ///
  /// Try parsing triple-quited(`"""`) multi-line string.
  ///
  bool MaybeTripleQuotedString(value::StringData *str);

  ///
  /// Parse assset path identifier.
  ///
  bool ParseAssetIdentifier(value::AssetPath *out, bool *triple_deliminated);

#if 0
  ///
  ///
  ///
  std::string GetDefaultPrimName() const;

  ///
  /// Get parsed toplevel "def" nodes(GPrim)
  ///
  std::vector<GPrim> GetGPrims();
#endif
  class PrimIterator;
  using const_iterator = PrimIterator;
  const_iterator begin() const;
  const_iterator end() const;

  ///
  /// Get error message(when `Parse` failed)
  ///
  std::string GetError();

  ///
  /// Get warning message(warnings in `Parse`)
  ///
  std::string GetWarning();

#if 0
  // Return the flag if the .usda is read from `references`
  bool IsReferenced() { return _referenced; }

  // Return the flag if the .usda is read from `subLayers`
  bool IsSubLayered() { return _sub_layered; }

  // Return the flag if the .usda is read from `payload`
  bool IsPayloaded() { return _payloaded; }
#endif

  // Return true if the .udsa is read in the top layer(stage)
  bool IsToplevel() {
    return _toplevel;
    // return !IsReferenced() && !IsSubLayered() && !IsPayloaded();
  }

  bool MaybeNone();
  bool MaybeCustom();

  template <typename T>
  bool MaybeNonFinite(T *out);

  bool LexFloat(std::string *result);

  bool Expect(char expect_c);

  bool ReadStringLiteral(
      std::string *literal);  // identifier wrapped with " or '. result contains
                              // quote chars.
  bool ReadPrimAttrIdentifier(std::string *token);
  bool ReadIdentifier(std::string *token);  // no '"'
  bool ReadPathIdentifier(
      std::string *path_identifier);  // '<' + identifier + '>'

  // read until newline
  bool ReadUntilNewline(std::string *str);


  /// Parse magic
  /// #usda FLOAT
  bool ParseMagicHeader();

  bool SkipWhitespace();

  // skip_semicolon true: ';' can be used as a separator. this flag is for
  // statement block.
  bool SkipWhitespaceAndNewline(const bool allow_semicolon = true);
  bool SkipCommentAndWhitespaceAndNewline(const bool allow_semicolon = true);

  bool SkipUntilNewline();

  // bool ParseAttributeMeta();
  bool ParseAttrMeta(AttrMeta *out_meta);

  bool ParsePrimMetas(PrimMetaMap *out_metamap);

  bool ParseMetaValue(const VariableDef &def, MetaVariable *outvar);

  bool ParseStageMetaOpt();
  // Parsed Stage metadatum is stored in this instance.
  bool ParseStageMetas();

  bool ParseCustomMetaValue();

  bool ParseReference(Reference *out, bool *triple_deliminated);
  bool ParsePayload(Payload *out, bool *triple_deliminated);

  // `#` style comment
  bool ParseSharpComment();

  bool IsSupportedPrimAttrType(const std::string &ty);
  bool IsSupportedPrimType(const std::string &ty);
  bool IsSupportedAPISchema(const std::string &ty);

  bool Eof() {
    // end of buffer, or current char is nullchar('\0')
    return _sr->eof() || _sr->is_nullchar();
  }

  bool ParseRelationship(Relationship *result);
  bool ParseProperties(std::map<std::string, Property> *props,
                       std::vector<value::token> *propNames);

  //
  // Look***() : Fetch chars but do not change input stream position.
  //

  bool LookChar1(char *c);
  bool LookCharN(size_t n, std::vector<char> *nc);

  bool Char1(char *c);
  bool CharN(size_t n, std::vector<char> *nc);

  bool Rewind(size_t offset);
  uint64_t CurrLoc();
  bool SeekTo(uint64_t pos);  // Move to absolute `pos` bytes location

  bool PushParserState();
  bool PopParserState(ParseState *state);

  //
  // Valid after ParseStageMetas() --------------
  //
  StageMetas GetStageMetas() const { return _stage_metas; }

  // primIdx is assigned through `PrimIdxAssignFunctin`
  // parentPrimIdx = -1 => root prim
  // depth = tree level(recursion count)
  // bool ParseClassBlock(const int64_t primIdx, const int64_t parentPrimIdx,
  // const uint32_t depth = 0); bool ParseOverBlock(const int64_t primIdx, const
  // int64_t parentPrimIdx, const uint32_t depth = 0); bool ParseDefBlock(const
  // int64_t primIdx, const int64_t parentPrimIdx, const uint32_t depth = 0);

  // Parse `def`, `over` or `class` block
  // @param[in] in_variantStmt : true when this Block is parsed within
  // `variantSet` statement. Default true.
  bool ParseBlock(const Specifier spec, const int64_t primIdx,
                  const int64_t parentPrimIdx, const uint32_t depth,
                  const bool in_variant = false);

  // Parse `varianntSet` stmt
  bool ParseVariantSet(const int64_t primIdx, const int64_t parentPrimIdx,
                       const uint32_t depth,
                       std::map<std::string, VariantContent> *variantSetMap);

  // --------------------------------------------

 private:
  ///
  /// Do common setups. Assume called in ctor.
  ///
  void Setup();

  nonstd::optional<std::pair<ListEditQual, MetaVariable>> ParsePrimMeta();
  bool ParsePrimProps(std::map<std::string, Property> *props,
                      std::vector<value::token> *propNames);

  template <typename T>
  bool ParseBasicPrimAttr(bool array_qual, const std::string &primattr_name,
                          Attribute *out_attr);

  bool ParseStageMeta(std::pair<ListEditQual, MetaVariable> *out);

  nonstd::optional<VariableDef> GetStageMetaDefinition(const std::string &name);
  nonstd::optional<VariableDef> GetPrimMetaDefinition(const std::string &arg);
  nonstd::optional<VariableDef> GetPropMetaDefinition(const std::string &arg);

  std::string GetCurrentPrimPath();
  bool PrimPathStackDepth() { return _path_stack.size(); }
  void PushPrimPath(const std::string &abs_path) {
    // TODO: validate `abs_path` is really absolute full path.
    _path_stack.push(abs_path);
  }
  void PopPrimPath() {
    if (!_path_stack.empty()) {
      _path_stack.pop();
    }
  }

  const tinyusdz::StreamReader *_sr = nullptr;

  // "class" defs
  // std::map<std::string, Klass> _klasses;
  std::stack<std::string> _path_stack;

  Cursor _curr_cursor;

  // Supported Prim types
  std::set<std::string> _supported_prim_types;
  std::set<std::string> _supported_prim_attr_types;

  // Supported API schemas
  std::set<std::string> _supported_api_schemas;

  // Supported metadataum for Stage
  std::map<std::string, VariableDef> _supported_stage_metas;

  // Supported metadataum for Prim.
  std::map<std::string, VariableDef> _supported_prim_metas;

  // Supported metadataum for Property(Attribute and Relation).
  std::map<std::string, VariableDef> _supported_prop_metas;

  std::stack<ErrorDiagnostic> err_stack;
  std::stack<ErrorDiagnostic> warn_stack;
  std::stack<ParseState> parse_stack;

  float _version{1.0f};

  // load flags
  bool _toplevel{true};
  // TODO: deprecate?
  bool _sub_layered{false};
  bool _referenced{false};
  bool _payloaded{false};

  AsciiParserOption _option;

  std::string _base_dir;

  StageMetas _stage_metas;

  //
  // Callbacks
  //
  PrimIdxAssignFunctin _prim_idx_assign_fun;
  StageMetaProcessFunction _stage_meta_process_fun;
  // PrimMetaProcessFunction _prim_meta_process_fun;
  std::map<std::string, PrimConstructFunction> _prim_construct_fun_map;
  std::map<std::string, PostPrimConstructFunction> _post_prim_construct_fun_map;

  bool _primspec_mode{false};

  // For composition. PrimSpec is typeless so single callback function only.
  PrimSpecFunction _primspec_fun{nullptr};
};

///
/// For USDC.
/// Parse string representation of UnregisteredValue(Attribute value).
/// e.g. "[(0, 1), (2, 3)]" for uint2[] type
///
/// @param[in] typeName typeName(e.g. "uint2")
/// @param[in] str Ascii representation of value.
/// @param[out] value Ascii representation of value.
/// @param[out] err Parse error message when returning false.
///
bool ParseUnregistredValue(const std::string &typeName, const std::string &str,
                           value::Value *value, std::string *err);

}  // namespace ascii

}  // namespace tinyusdz
