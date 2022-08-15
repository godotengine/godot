//===- DIBuilder.h - Debug Information Builder ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DIBuilder that is useful for creating debugging
// information entries in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DIBUILDER_H
#define LLVM_IR_DIBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/TrackingMDRef.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class BasicBlock;
  class Instruction;
  class Function;
  class Module;
  class Value;
  class Constant;
  class LLVMContext;
  class StringRef;

  class DIBuilder {
    Module &M;
    LLVMContext &VMContext;

    DICompileUnit *CUNode;   ///< The one compile unit created by this DIBuiler.
    Function *DeclareFn;     ///< llvm.dbg.declare
    Function *ValueFn;       ///< llvm.dbg.value

    SmallVector<Metadata *, 4> AllEnumTypes;
    /// Track the RetainTypes, since they can be updated later on.
    SmallVector<TrackingMDNodeRef, 4> AllRetainTypes;
    SmallVector<Metadata *, 4> AllSubprograms;
    SmallVector<Metadata *, 4> AllGVs;
    SmallVector<TrackingMDNodeRef, 4> AllImportedModules;

    /// Track nodes that may be unresolved.
    SmallVector<TrackingMDNodeRef, 4> UnresolvedNodes;
    bool AllowUnresolvedNodes;

    /// Each subprogram's preserved local variables.
    DenseMap<MDNode *, std::vector<TrackingMDNodeRef>> PreservedVariables;

    DIBuilder(const DIBuilder &) = delete;
    void operator=(const DIBuilder &) = delete;

    /// Create a temporary.
    ///
    /// Create an \a temporary node and track it in \a UnresolvedNodes.
    void trackIfUnresolved(MDNode *N);

  public:
    /// Construct a builder for a module.
    ///
    /// If \c AllowUnresolved, collect unresolved nodes attached to the module
    /// in order to resolve cycles during \a finalize().
    explicit DIBuilder(Module &M, bool AllowUnresolved = true);
    enum DebugEmissionKind { FullDebug=1, LineTablesOnly };

    /// Construct any deferred debug info descriptors.
    void finalize();

    /// A CompileUnit provides an anchor for all debugging
    /// information generated during this instance of compilation.
    /// \param Lang          Source programming language, eg. dwarf::DW_LANG_C99
    /// \param File          File name
    /// \param Dir           Directory
    /// \param Producer      Identify the producer of debugging information
    ///                      and code.  Usually this is a compiler
    ///                      version string.
    /// \param isOptimized   A boolean flag which indicates whether optimization
    ///                      is enabled or not.
    /// \param Flags         This string lists command line options. This
    ///                      string is directly embedded in debug info
    ///                      output which may be used by a tool
    ///                      analyzing generated debugging information.
    /// \param RV            This indicates runtime version for languages like
    ///                      Objective-C.
    /// \param SplitName     The name of the file that we'll split debug info
    ///                      out into.
    /// \param Kind          The kind of debug information to generate.
    /// \param DWOId         The DWOId if this is a split skeleton compile unit.
    /// \param EmitDebugInfo A boolean flag which indicates whether
    ///                      debug information should be written to
    ///                      the final output or not. When this is
    ///                      false, debug information annotations will
    ///                      be present in the IL but they are not
    ///                      written to the final assembly or object
    ///                      file. This supports tracking source
    ///                      location information in the back end
    ///                      without actually changing the output
    ///                      (e.g., when using optimization remarks).
    DICompileUnit *
    createCompileUnit(unsigned Lang, StringRef File, StringRef Dir,
                      StringRef Producer, bool isOptimized, StringRef Flags,
                      unsigned RV, StringRef SplitName = StringRef(),
                      DebugEmissionKind Kind = FullDebug, uint64_t DWOId = 0,
                      bool EmitDebugInfo = true);

    /// Create a file descriptor to hold debugging information
    /// for a file.
    DIFile *createFile(StringRef Filename, StringRef Directory);

    /// Create a single enumerator value.
    DIEnumerator *createEnumerator(StringRef Name, int64_t Val);

    /// Create a DWARF unspecified type.
    DIBasicType *createUnspecifiedType(StringRef Name);

    /// Create C++11 nullptr type.
    DIBasicType *createNullPtrType();

    /// Create debugging information entry for a basic
    /// type.
    /// \param Name        Type name.
    /// \param SizeInBits  Size of the type.
    /// \param AlignInBits Type alignment.
    /// \param Encoding    DWARF encoding code, e.g. dwarf::DW_ATE_float.
    DIBasicType *createBasicType(StringRef Name, uint64_t SizeInBits,
                                 uint64_t AlignInBits, unsigned Encoding);

    /// Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    /// \param Tag         Tag identifing type, e.g. dwarf::TAG_volatile_type
    /// \param FromTy      Base Type.
    DIDerivedType *createQualifiedType(unsigned Tag, DIType *FromTy);

    /// Create debugging information entry for a pointer.
    /// \param PointeeTy   Type pointed by this pointer.
    /// \param SizeInBits  Size.
    /// \param AlignInBits Alignment. (optional)
    /// \param Name        Pointer type name. (optional)
    DIDerivedType *createPointerType(DIType *PointeeTy, uint64_t SizeInBits,
                                     uint64_t AlignInBits = 0,
                                     StringRef Name = "");

    /// Create debugging information entry for a pointer to member.
    /// \param PointeeTy Type pointed to by this pointer.
    /// \param SizeInBits  Size.
    /// \param AlignInBits Alignment. (optional)
    /// \param Class Type for which this pointer points to members of.
    DIDerivedType *createMemberPointerType(DIType *PointeeTy, DIType *Class,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits = 0);

    /// Create debugging information entry for a c++
    /// style reference or rvalue reference type.
    DIDerivedType *createReferenceType(unsigned Tag, DIType *RTy);

    /// Create debugging information entry for a typedef.
    /// \param Ty          Original type.
    /// \param Name        Typedef name.
    /// \param File        File where this type is defined.
    /// \param LineNo      Line number.
    /// \param Context     The surrounding context for the typedef.
    DIDerivedType *createTypedef(DIType *Ty, StringRef Name, DIFile *File,
                                 unsigned LineNo, DIScope *Context);

    /// Create debugging information entry for a 'friend'.
    DIDerivedType *createFriend(DIType *Ty, DIType *FriendTy);

    /// Create debugging information entry to establish
    /// inheritance relationship between two types.
    /// \param Ty           Original type.
    /// \param BaseTy       Base type. Ty is inherits from base.
    /// \param BaseOffset   Base offset.
    /// \param Flags        Flags to describe inheritance attribute,
    ///                     e.g. private
    DIDerivedType *createInheritance(DIType *Ty, DIType *BaseTy,
                                     uint64_t BaseOffset, unsigned Flags);

    /// Create debugging information entry for a member.
    /// \param Scope        Member scope.
    /// \param Name         Member name.
    /// \param File         File where this member is defined.
    /// \param LineNo       Line number.
    /// \param SizeInBits   Member size.
    /// \param AlignInBits  Member alignment.
    /// \param OffsetInBits Member offset.
    /// \param Flags        Flags to encode member attribute, e.g. private
    /// \param Ty           Parent type.
    DIDerivedType *createMemberType(DIScope *Scope, StringRef Name,
                                    DIFile *File, unsigned LineNo,
                                    uint64_t SizeInBits, uint64_t AlignInBits,
                                    uint64_t OffsetInBits, unsigned Flags,
                                    DIType *Ty);

    /// Create debugging information entry for a
    /// C++ static data member.
    /// \param Scope      Member scope.
    /// \param Name       Member name.
    /// \param File       File where this member is declared.
    /// \param LineNo     Line number.
    /// \param Ty         Type of the static member.
    /// \param Flags      Flags to encode member attribute, e.g. private.
    /// \param Val        Const initializer of the member.
    DIDerivedType *createStaticMemberType(DIScope *Scope, StringRef Name,
                                          DIFile *File, unsigned LineNo,
                                          DIType *Ty, unsigned Flags,
                                          llvm::Constant *Val);

    /// Create debugging information entry for Objective-C
    /// instance variable.
    /// \param Name         Member name.
    /// \param File         File where this member is defined.
    /// \param LineNo       Line number.
    /// \param SizeInBits   Member size.
    /// \param AlignInBits  Member alignment.
    /// \param OffsetInBits Member offset.
    /// \param Flags        Flags to encode member attribute, e.g. private
    /// \param Ty           Parent type.
    /// \param PropertyNode Property associated with this ivar.
    DIDerivedType *createObjCIVar(StringRef Name, DIFile *File, unsigned LineNo,
                                  uint64_t SizeInBits, uint64_t AlignInBits,
                                  uint64_t OffsetInBits, unsigned Flags,
                                  DIType *Ty, MDNode *PropertyNode);

    /// Create debugging information entry for Objective-C
    /// property.
    /// \param Name         Property name.
    /// \param File         File where this property is defined.
    /// \param LineNumber   Line number.
    /// \param GetterName   Name of the Objective C property getter selector.
    /// \param SetterName   Name of the Objective C property setter selector.
    /// \param PropertyAttributes Objective C property attributes.
    /// \param Ty           Type.
    DIObjCProperty *createObjCProperty(StringRef Name, DIFile *File,
                                       unsigned LineNumber,
                                       StringRef GetterName,
                                       StringRef SetterName,
                                       unsigned PropertyAttributes, DIType *Ty);

    /// Create debugging information entry for a class.
    /// \param Scope        Scope in which this class is defined.
    /// \param Name         class name.
    /// \param File         File where this member is defined.
    /// \param LineNumber   Line number.
    /// \param SizeInBits   Member size.
    /// \param AlignInBits  Member alignment.
    /// \param OffsetInBits Member offset.
    /// \param Flags        Flags to encode member attribute, e.g. private
    /// \param Elements     class members.
    /// \param VTableHolder Debug info of the base class that contains vtable
    ///                     for this type. This is used in
    ///                     DW_AT_containing_type. See DWARF documentation
    ///                     for more info.
    /// \param TemplateParms Template type parameters.
    /// \param UniqueIdentifier A unique identifier for the class.
    DICompositeType *createClassType(DIScope *Scope, StringRef Name,
                                     DIFile *File, unsigned LineNumber,
                                     uint64_t SizeInBits, uint64_t AlignInBits,
                                     uint64_t OffsetInBits, unsigned Flags,
                                     DIType *DerivedFrom, DINodeArray Elements,
                                     DIType *VTableHolder = nullptr,
                                     MDNode *TemplateParms = nullptr,
                                     StringRef UniqueIdentifier = "");

    /// Create debugging information entry for a struct.
    /// \param Scope        Scope in which this struct is defined.
    /// \param Name         Struct name.
    /// \param File         File where this member is defined.
    /// \param LineNumber   Line number.
    /// \param SizeInBits   Member size.
    /// \param AlignInBits  Member alignment.
    /// \param Flags        Flags to encode member attribute, e.g. private
    /// \param Elements     Struct elements.
    /// \param RunTimeLang  Optional parameter, Objective-C runtime version.
    /// \param UniqueIdentifier A unique identifier for the struct.
    DICompositeType *createStructType(
        DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
        uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
        DIType *DerivedFrom, DINodeArray Elements, unsigned RunTimeLang = 0,
        DIType *VTableHolder = nullptr, StringRef UniqueIdentifier = "");

    /// Create debugging information entry for an union.
    /// \param Scope        Scope in which this union is defined.
    /// \param Name         Union name.
    /// \param File         File where this member is defined.
    /// \param LineNumber   Line number.
    /// \param SizeInBits   Member size.
    /// \param AlignInBits  Member alignment.
    /// \param Flags        Flags to encode member attribute, e.g. private
    /// \param Elements     Union elements.
    /// \param RunTimeLang  Optional parameter, Objective-C runtime version.
    /// \param UniqueIdentifier A unique identifier for the union.
    DICompositeType *createUnionType(DIScope *Scope, StringRef Name,
                                     DIFile *File, unsigned LineNumber,
                                     uint64_t SizeInBits, uint64_t AlignInBits,
                                     unsigned Flags, DINodeArray Elements,
                                     unsigned RunTimeLang = 0,
                                     StringRef UniqueIdentifier = "");

    /// Create debugging information for template
    /// type parameter.
    /// \param Scope        Scope in which this type is defined.
    /// \param Name         Type parameter name.
    /// \param Ty           Parameter type.
    DITemplateTypeParameter *
    createTemplateTypeParameter(DIScope *Scope, StringRef Name, DIType *Ty);

    /// Create debugging information for template
    /// value parameter.
    /// \param Scope        Scope in which this type is defined.
    /// \param Name         Value parameter name.
    /// \param Ty           Parameter type.
    /// \param Val          Constant parameter value.
    DITemplateValueParameter *createTemplateValueParameter(DIScope *Scope,
                                                           StringRef Name,
                                                           DIType *Ty,
                                                           Constant *Val);

    /// Create debugging information for a template template parameter.
    /// \param Scope        Scope in which this type is defined.
    /// \param Name         Value parameter name.
    /// \param Ty           Parameter type.
    /// \param Val          The fully qualified name of the template.
    DITemplateValueParameter *createTemplateTemplateParameter(DIScope *Scope,
                                                              StringRef Name,
                                                              DIType *Ty,
                                                              StringRef Val);

    /// Create debugging information for a template parameter pack.
    /// \param Scope        Scope in which this type is defined.
    /// \param Name         Value parameter name.
    /// \param Ty           Parameter type.
    /// \param Val          An array of types in the pack.
    DITemplateValueParameter *createTemplateParameterPack(DIScope *Scope,
                                                          StringRef Name,
                                                          DIType *Ty,
                                                          DINodeArray Val);

    /// Create debugging information entry for an array.
    /// \param Size         Array size.
    /// \param AlignInBits  Alignment.
    /// \param Ty           Element type.
    /// \param Subscripts   Subscripts.
    DICompositeType *createArrayType(uint64_t Size, uint64_t AlignInBits,
                                     DIType *Ty, DINodeArray Subscripts);

    /// Create debugging information entry for a vector type.
    /// \param Size         Array size.
    /// \param AlignInBits  Alignment.
    /// \param Ty           Element type.
    /// \param Subscripts   Subscripts.
    DICompositeType *createVectorType(uint64_t Size, uint64_t AlignInBits,
                                      DIType *Ty, DINodeArray Subscripts);

    /// Create debugging information entry for an
    /// enumeration.
    /// \param Scope          Scope in which this enumeration is defined.
    /// \param Name           Union name.
    /// \param File           File where this member is defined.
    /// \param LineNumber     Line number.
    /// \param SizeInBits     Member size.
    /// \param AlignInBits    Member alignment.
    /// \param Elements       Enumeration elements.
    /// \param UnderlyingType Underlying type of a C++11/ObjC fixed enum.
    /// \param UniqueIdentifier A unique identifier for the enum.
    DICompositeType *createEnumerationType(
        DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
        uint64_t SizeInBits, uint64_t AlignInBits, DINodeArray Elements,
        DIType *UnderlyingType, StringRef UniqueIdentifier = "");

    /// Create subroutine type.
    /// \param File            File in which this subroutine is defined.
    /// \param ParameterTypes  An array of subroutine parameter types. This
    ///                        includes return type at 0th index.
    /// \param Flags           E.g.: LValueReference.
    ///                        These flags are used to emit dwarf attributes.
    DISubroutineType *createSubroutineType(DIFile *File,
                                           DITypeRefArray ParameterTypes,
                                           unsigned Flags = 0);

    /// Create a new DIType* with "artificial" flag set.
    DIType *createArtificialType(DIType *Ty);

    /// Create a new DIType* with the "object pointer"
    /// flag set.
    DIType *createObjectPointerType(DIType *Ty);

    /// Create a permanent forward-declared type.
    DICompositeType *createForwardDecl(unsigned Tag, StringRef Name,
                                       DIScope *Scope, DIFile *F, unsigned Line,
                                       unsigned RuntimeLang = 0,
                                       uint64_t SizeInBits = 0,
                                       uint64_t AlignInBits = 0,
                                       StringRef UniqueIdentifier = "");

    /// Create a temporary forward-declared type.
    DICompositeType *createReplaceableCompositeType(
        unsigned Tag, StringRef Name, DIScope *Scope, DIFile *F, unsigned Line,
        unsigned RuntimeLang = 0, uint64_t SizeInBits = 0,
        uint64_t AlignInBits = 0, unsigned Flags = DINode::FlagFwdDecl,
        StringRef UniqueIdentifier = "");

    /// Retain DIType* in a module even if it is not referenced
    /// through debug info anchors.
    void retainType(DIType *T);

    /// Create unspecified parameter type
    /// for a subroutine type.
    DIBasicType *createUnspecifiedParameter();

    /// Get a DINodeArray, create one if required.
    DINodeArray getOrCreateArray(ArrayRef<Metadata *> Elements);

    /// Get a DITypeRefArray, create one if required.
    DITypeRefArray getOrCreateTypeArray(ArrayRef<Metadata *> Elements);

    /// Create a descriptor for a value range.  This
    /// implicitly uniques the values returned.
    DISubrange *getOrCreateSubrange(int64_t Lo, int64_t Count);

    /// Create a new descriptor for the specified
    /// variable.
    /// \param Context     Variable scope.
    /// \param Name        Name of the variable.
    /// \param LinkageName Mangled  name of the variable.
    /// \param File        File where this variable is defined.
    /// \param LineNo      Line number.
    /// \param Ty          Variable Type.
    /// \param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// \param Val         llvm::Value of the variable.
    /// \param Decl        Reference to the corresponding declaration.
    DIGlobalVariable *createGlobalVariable(DIScope *Context, StringRef Name,
                                           StringRef LinkageName, DIFile *File,
                                           unsigned LineNo, DIType *Ty,
                                           bool isLocalToUnit,
                                           llvm::Constant *Val,
                                           MDNode *Decl = nullptr);

    /// Identical to createGlobalVariable
    /// except that the resulting DbgNode is temporary and meant to be RAUWed.
    DIGlobalVariable *createTempGlobalVariableFwdDecl(
        DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *File,
        unsigned LineNo, DIType *Ty, bool isLocalToUnit, llvm::Constant *Val,
        MDNode *Decl = nullptr);

    /// Create a new descriptor for the specified
    /// local variable.
    /// \param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// \param Scope       Variable scope.
    /// \param Name        Variable name.
    /// \param File        File where this variable is defined.
    /// \param LineNo      Line number.
    /// \param Ty          Variable Type
    /// \param AlwaysPreserve Boolean. Set to true if debug info for this
    ///                       variable should be preserved in optimized build.
    /// \param Flags       Flags, e.g. artificial variable.
    /// \param ArgNo       If this variable is an argument then this argument's
    ///                    number. 1 indicates 1st argument.
    DILocalVariable *createLocalVariable(unsigned Tag, DIScope *Scope,
                                         StringRef Name, DIFile *File,
                                         unsigned LineNo, DIType *Ty,
                                         bool AlwaysPreserve = false,
                                         unsigned Flags = 0,
                                         unsigned ArgNo = 0);

    /// Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    /// \param Addr        An array of complex address operations.
    DIExpression *createExpression(ArrayRef<uint64_t> Addr = None);
    DIExpression *createExpression(ArrayRef<int64_t> Addr);

    /// Create a descriptor to describe one part
    /// of aggregate variable that is fragmented across multiple Values.
    ///
    /// \param OffsetInBits Offset of the piece in bits.
    /// \param SizeInBits   Size of the piece in bits.
    DIExpression *createBitPieceExpression(unsigned OffsetInBits,
                                           unsigned SizeInBits);

    /// Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram* for descriptions of these fields.
    /// \param Scope         Function scope.
    /// \param Name          Function name.
    /// \param LinkageName   Mangled function name.
    /// \param File          File where this variable is defined.
    /// \param LineNo        Line number.
    /// \param Ty            Function type.
    /// \param isLocalToUnit True if this function is not externally visible.
    /// \param isDefinition  True if this is a function definition.
    /// \param ScopeLine     Set to the beginning of the scope this starts
    /// \param Flags         e.g. is this function prototyped or not.
    ///                      These flags are used to emit dwarf attributes.
    /// \param isOptimized   True if optimization is ON.
    /// \param Fn            llvm::Function pointer.
    /// \param TParam        Function template parameters.
    DISubprogram *
    createFunction(DIScope *Scope, StringRef Name, StringRef LinkageName,
                   DIFile *File, unsigned LineNo, DISubroutineType *Ty,
                   bool isLocalToUnit, bool isDefinition, unsigned ScopeLine,
                   unsigned Flags = 0, bool isOptimized = false,
                   Function *Fn = nullptr, MDNode *TParam = nullptr,
                   MDNode *Decl = nullptr);

    /// Identical to createFunction,
    /// except that the resulting DbgNode is meant to be RAUWed.
    DISubprogram *createTempFunctionFwdDecl(
        DIScope *Scope, StringRef Name, StringRef LinkageName, DIFile *File,
        unsigned LineNo, DISubroutineType *Ty, bool isLocalToUnit,
        bool isDefinition, unsigned ScopeLine, unsigned Flags = 0,
        bool isOptimized = false, Function *Fn = nullptr,
        MDNode *TParam = nullptr, MDNode *Decl = nullptr);

    /// FIXME: this is added for dragonegg. Once we update dragonegg
    /// to call resolve function, this will be removed.
    DISubprogram *
    createFunction(DIScopeRef Scope, StringRef Name, StringRef LinkageName,
                   DIFile *File, unsigned LineNo, DISubroutineType *Ty,
                   bool isLocalToUnit, bool isDefinition, unsigned ScopeLine,
                   unsigned Flags = 0, bool isOptimized = false,
                   Function *Fn = nullptr, MDNode *TParam = nullptr,
                   MDNode *Decl = nullptr);

    /// Create a new descriptor for the specified C++ method.
    /// See comments in \a DISubprogram* for descriptions of these fields.
    /// \param Scope         Function scope.
    /// \param Name          Function name.
    /// \param LinkageName   Mangled function name.
    /// \param File          File where this variable is defined.
    /// \param LineNo        Line number.
    /// \param Ty            Function type.
    /// \param isLocalToUnit True if this function is not externally visible..
    /// \param isDefinition  True if this is a function definition.
    /// \param Virtuality    Attributes describing virtualness. e.g. pure
    ///                      virtual function.
    /// \param VTableIndex   Index no of this method in virtual table.
    /// \param VTableHolder  Type that holds vtable.
    /// \param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// \param isOptimized   True if optimization is ON.
    /// \param Fn            llvm::Function pointer.
    /// \param TParam        Function template parameters.
    DISubprogram *
    createMethod(DIScope *Scope, StringRef Name, StringRef LinkageName,
                 DIFile *File, unsigned LineNo, DISubroutineType *Ty,
                 bool isLocalToUnit, bool isDefinition, unsigned Virtuality = 0,
                 unsigned VTableIndex = 0, DIType *VTableHolder = nullptr,
                 unsigned Flags = 0, bool isOptimized = false,
                 Function *Fn = nullptr, MDNode *TParam = nullptr);

    /// This creates new descriptor for a namespace with the specified
    /// parent scope.
    /// \param Scope       Namespace scope
    /// \param Name        Name of this namespace
    /// \param File        Source file
    /// \param LineNo      Line number
    DINamespace *createNameSpace(DIScope *Scope, StringRef Name, DIFile *File,
                                 unsigned LineNo);

    /// This creates new descriptor for a module with the specified
    /// parent scope.
    /// \param Scope       Parent scope
    /// \param Name        Name of this module
    /// \param ConfigurationMacros
    ///                    A space-separated shell-quoted list of -D macro
    ///                    definitions as they would appear on a command line.
    /// \param IncludePath The path to the module map file.
    /// \param ISysRoot    The clang system root (value of -isysroot).
    DIModule *createModule(DIScope *Scope, StringRef Name,
                           StringRef ConfigurationMacros,
                           StringRef IncludePath,
                           StringRef ISysRoot);

    /// This creates a descriptor for a lexical block with a new file
    /// attached. This merely extends the existing
    /// lexical block as it crosses a file.
    /// \param Scope       Lexical block.
    /// \param File        Source file.
    /// \param Discriminator DWARF path discriminator value.
    DILexicalBlockFile *createLexicalBlockFile(DIScope *Scope, DIFile *File,
                                               unsigned Discriminator = 0);

    /// This creates a descriptor for a lexical block with the
    /// specified parent context.
    /// \param Scope         Parent lexical scope.
    /// \param File          Source file.
    /// \param Line          Line number.
    /// \param Col           Column number.
    DILexicalBlock *createLexicalBlock(DIScope *Scope, DIFile *File,
                                       unsigned Line, unsigned Col);

    /// Create a descriptor for an imported module.
    /// \param Context The scope this module is imported into
    /// \param NS The namespace being imported here
    /// \param Line Line number
    DIImportedEntity *createImportedModule(DIScope *Context, DINamespace *NS,
                                           unsigned Line);

    /// Create a descriptor for an imported module.
    /// \param Context The scope this module is imported into
    /// \param NS An aliased namespace
    /// \param Line Line number
    DIImportedEntity *createImportedModule(DIScope *Context,
                                           DIImportedEntity *NS, unsigned Line);

    /// Create a descriptor for an imported module.
    /// \param Context The scope this module is imported into
    /// \param M The module being imported here
    /// \param Line Line number
    DIImportedEntity *createImportedModule(DIScope *Context, DIModule *M,
                                           unsigned Line);

    /// Create a descriptor for an imported function.
    /// \param Context The scope this module is imported into
    /// \param Decl The declaration (or definition) of a function, type, or
    ///             variable
    /// \param Line Line number
    DIImportedEntity *createImportedDeclaration(DIScope *Context, DINode *Decl,
                                                unsigned Line,
                                                StringRef Name = "");

    /// Insert a new llvm.dbg.declare intrinsic call.
    /// \param Storage     llvm::Value of the variable
    /// \param VarInfo     Variable's debug info descriptor.
    /// \param Expr        A complex location expression.
    /// \param DL          Debug info location.
    /// \param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, DILocalVariable *VarInfo,
                               DIExpression *Expr, const DILocation *DL,
                               BasicBlock *InsertAtEnd);

    /// Insert a new llvm.dbg.declare intrinsic call.
    /// \param Storage      llvm::Value of the variable
    /// \param VarInfo      Variable's debug info descriptor.
    /// \param Expr         A complex location expression.
    /// \param DL           Debug info location.
    /// \param InsertBefore Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, DILocalVariable *VarInfo,
                               DIExpression *Expr, const DILocation *DL,
                               Instruction *InsertBefore);

    /// Insert a new llvm.dbg.value intrinsic call.
    /// \param Val          llvm::Value of the variable
    /// \param Offset       Offset
    /// \param VarInfo      Variable's debug info descriptor.
    /// \param Expr         A complex location expression.
    /// \param DL           Debug info location.
    /// \param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DILocalVariable *VarInfo,
                                         DIExpression *Expr,
                                         const DILocation *DL,
                                         BasicBlock *InsertAtEnd);

    /// Insert a new llvm.dbg.value intrinsic call.
    /// \param Val          llvm::Value of the variable
    /// \param Offset       Offset
    /// \param VarInfo      Variable's debug info descriptor.
    /// \param Expr         A complex location expression.
    /// \param DL           Debug info location.
    /// \param InsertBefore Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DILocalVariable *VarInfo,
                                         DIExpression *Expr,
                                         const DILocation *DL,
                                         Instruction *InsertBefore);

    /// Replace the vtable holder in the given composite type.
    ///
    /// If this creates a self reference, it may orphan some unresolved cycles
    /// in the operands of \c T, so \a DIBuilder needs to track that.
    void replaceVTableHolder(DICompositeType *&T,
                             DICompositeType *VTableHolder);

    /// Replace arrays on a composite type.
    ///
    /// If \c T is resolved, but the arrays aren't -- which can happen if \c T
    /// has a self-reference -- \a DIBuilder needs to track the array to
    /// resolve cycles.
    void replaceArrays(DICompositeType *&T, DINodeArray Elements,
                       DINodeArray TParems = DINodeArray());

    /// Replace a temporary node.
    ///
    /// Call \a MDNode::replaceAllUsesWith() on \c N, replacing it with \c
    /// Replacement.
    ///
    /// If \c Replacement is the same as \c N.get(), instead call \a
    /// MDNode::replaceWithUniqued().  In this case, the uniqued node could
    /// have a different address, so we return the final address.
    template <class NodeTy>
    NodeTy *replaceTemporary(TempMDNode &&N, NodeTy *Replacement) {
      if (N.get() == Replacement)
        return cast<NodeTy>(MDNode::replaceWithUniqued(std::move(N)));

      N->replaceAllUsesWith(Replacement);
      return Replacement;
    }
  };
} // end namespace llvm

#endif
