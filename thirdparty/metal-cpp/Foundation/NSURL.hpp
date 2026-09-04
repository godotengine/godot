#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class String;
}

namespace NS
{

using URLResourceKey = NS::String*;
using URLFileResourceType = NS::String*;
using URLThumbnailDictionaryItem = NS::String*;
using URLFileProtectionType = NS::String*;
using URLUbiquitousItemDownloadingStatus = NS::String*;
using URLUbiquitousSharedItemRole = NS::String*;
using URLUbiquitousSharedItemPermissions = NS::String*;
extern NS::String* const URLFileScheme __asm__("_NSURLFileScheme");
extern URLResourceKey const URLKeysOfUnsetValuesKey __asm__("_NSURLKeysOfUnsetValuesKey");
extern URLResourceKey const URLNameKey __asm__("_NSURLNameKey");
extern URLResourceKey const URLLocalizedNameKey __asm__("_NSURLLocalizedNameKey");
extern URLResourceKey const URLIsRegularFileKey __asm__("_NSURLIsRegularFileKey");
extern URLResourceKey const URLIsDirectoryKey __asm__("_NSURLIsDirectoryKey");
extern URLResourceKey const URLIsSymbolicLinkKey __asm__("_NSURLIsSymbolicLinkKey");
extern URLResourceKey const URLIsVolumeKey __asm__("_NSURLIsVolumeKey");
extern URLResourceKey const URLIsPackageKey __asm__("_NSURLIsPackageKey");
extern URLResourceKey const URLIsApplicationKey __asm__("_NSURLIsApplicationKey");
extern URLResourceKey const URLApplicationIsScriptableKey __asm__("_NSURLApplicationIsScriptableKey");
extern URLResourceKey const URLIsSystemImmutableKey __asm__("_NSURLIsSystemImmutableKey");
extern URLResourceKey const URLIsUserImmutableKey __asm__("_NSURLIsUserImmutableKey");
extern URLResourceKey const URLIsHiddenKey __asm__("_NSURLIsHiddenKey");
extern URLResourceKey const URLHasHiddenExtensionKey __asm__("_NSURLHasHiddenExtensionKey");
extern URLResourceKey const URLCreationDateKey __asm__("_NSURLCreationDateKey");
extern URLResourceKey const URLContentAccessDateKey __asm__("_NSURLContentAccessDateKey");
extern URLResourceKey const URLContentModificationDateKey __asm__("_NSURLContentModificationDateKey");
extern URLResourceKey const URLAttributeModificationDateKey __asm__("_NSURLAttributeModificationDateKey");
extern URLResourceKey const URLLinkCountKey __asm__("_NSURLLinkCountKey");
extern URLResourceKey const URLParentDirectoryURLKey __asm__("_NSURLParentDirectoryURLKey");
extern URLResourceKey const URLVolumeURLKey __asm__("_NSURLVolumeURLKey");
extern URLResourceKey const URLTypeIdentifierKey __asm__("_NSURLTypeIdentifierKey");
extern URLResourceKey const URLContentTypeKey __asm__("_NSURLContentTypeKey");
extern URLResourceKey const URLLocalizedTypeDescriptionKey __asm__("_NSURLLocalizedTypeDescriptionKey");
extern URLResourceKey const URLLabelNumberKey __asm__("_NSURLLabelNumberKey");
extern URLResourceKey const URLLabelColorKey __asm__("_NSURLLabelColorKey");
extern URLResourceKey const URLLocalizedLabelKey __asm__("_NSURLLocalizedLabelKey");
extern URLResourceKey const URLEffectiveIconKey __asm__("_NSURLEffectiveIconKey");
extern URLResourceKey const URLCustomIconKey __asm__("_NSURLCustomIconKey");
extern URLResourceKey const URLFileResourceIdentifierKey __asm__("_NSURLFileResourceIdentifierKey");
extern URLResourceKey const URLVolumeIdentifierKey __asm__("_NSURLVolumeIdentifierKey");
extern URLResourceKey const URLPreferredIOBlockSizeKey __asm__("_NSURLPreferredIOBlockSizeKey");
extern URLResourceKey const URLIsReadableKey __asm__("_NSURLIsReadableKey");
extern URLResourceKey const URLIsWritableKey __asm__("_NSURLIsWritableKey");
extern URLResourceKey const URLIsExecutableKey __asm__("_NSURLIsExecutableKey");
extern URLResourceKey const URLFileSecurityKey __asm__("_NSURLFileSecurityKey");
extern URLResourceKey const URLIsExcludedFromBackupKey __asm__("_NSURLIsExcludedFromBackupKey");
extern URLResourceKey const URLTagNamesKey __asm__("_NSURLTagNamesKey");
extern URLResourceKey const URLPathKey __asm__("_NSURLPathKey");
extern URLResourceKey const URLCanonicalPathKey __asm__("_NSURLCanonicalPathKey");
extern URLResourceKey const URLIsMountTriggerKey __asm__("_NSURLIsMountTriggerKey");
extern URLResourceKey const URLGenerationIdentifierKey __asm__("_NSURLGenerationIdentifierKey");
extern URLResourceKey const URLDocumentIdentifierKey __asm__("_NSURLDocumentIdentifierKey");
extern URLResourceKey const URLAddedToDirectoryDateKey __asm__("_NSURLAddedToDirectoryDateKey");
extern URLResourceKey const URLQuarantinePropertiesKey __asm__("_NSURLQuarantinePropertiesKey");
extern URLResourceKey const URLFileResourceTypeKey __asm__("_NSURLFileResourceTypeKey");
extern URLResourceKey const URLFileIdentifierKey __asm__("_NSURLFileIdentifierKey");
extern URLResourceKey const URLFileContentIdentifierKey __asm__("_NSURLFileContentIdentifierKey");
extern URLResourceKey const URLMayShareFileContentKey __asm__("_NSURLMayShareFileContentKey");
extern URLResourceKey const URLMayHaveExtendedAttributesKey __asm__("_NSURLMayHaveExtendedAttributesKey");
extern URLResourceKey const URLIsPurgeableKey __asm__("_NSURLIsPurgeableKey");
extern URLResourceKey const URLIsSparseKey __asm__("_NSURLIsSparseKey");
extern URLFileResourceType const URLFileResourceTypeNamedPipe __asm__("_NSURLFileResourceTypeNamedPipe");
extern URLFileResourceType const URLFileResourceTypeCharacterSpecial __asm__("_NSURLFileResourceTypeCharacterSpecial");
extern URLFileResourceType const URLFileResourceTypeDirectory __asm__("_NSURLFileResourceTypeDirectory");
extern URLFileResourceType const URLFileResourceTypeBlockSpecial __asm__("_NSURLFileResourceTypeBlockSpecial");
extern URLFileResourceType const URLFileResourceTypeRegular __asm__("_NSURLFileResourceTypeRegular");
extern URLFileResourceType const URLFileResourceTypeSymbolicLink __asm__("_NSURLFileResourceTypeSymbolicLink");
extern URLFileResourceType const URLFileResourceTypeSocket __asm__("_NSURLFileResourceTypeSocket");
extern URLFileResourceType const URLFileResourceTypeUnknown __asm__("_NSURLFileResourceTypeUnknown");
extern URLResourceKey const URLThumbnailDictionaryKey __asm__("_NSURLThumbnailDictionaryKey");
extern URLResourceKey const URLThumbnailKey __asm__("_NSURLThumbnailKey");
extern URLThumbnailDictionaryItem const Thumbnail1024x1024SizeKey __asm__("_NSThumbnail1024x1024SizeKey");
extern URLResourceKey const URLFileSizeKey __asm__("_NSURLFileSizeKey");
extern URLResourceKey const URLFileAllocatedSizeKey __asm__("_NSURLFileAllocatedSizeKey");
extern URLResourceKey const URLTotalFileSizeKey __asm__("_NSURLTotalFileSizeKey");
extern URLResourceKey const URLTotalFileAllocatedSizeKey __asm__("_NSURLTotalFileAllocatedSizeKey");
extern URLResourceKey const URLIsAliasFileKey __asm__("_NSURLIsAliasFileKey");
extern URLResourceKey const URLFileProtectionKey __asm__("_NSURLFileProtectionKey");
extern URLFileProtectionType const URLFileProtectionNone __asm__("_NSURLFileProtectionNone");
extern URLFileProtectionType const URLFileProtectionComplete __asm__("_NSURLFileProtectionComplete");
extern URLFileProtectionType const URLFileProtectionCompleteUnlessOpen __asm__("_NSURLFileProtectionCompleteUnlessOpen");
extern URLFileProtectionType const URLFileProtectionCompleteUntilFirstUserAuthentication __asm__("_NSURLFileProtectionCompleteUntilFirstUserAuthentication");
extern URLFileProtectionType const URLFileProtectionCompleteWhenUserInactive __asm__("_NSURLFileProtectionCompleteWhenUserInactive");
extern URLResourceKey const URLDirectoryEntryCountKey __asm__("_NSURLDirectoryEntryCountKey");
extern URLResourceKey const URLVolumeLocalizedFormatDescriptionKey __asm__("_NSURLVolumeLocalizedFormatDescriptionKey");
extern URLResourceKey const URLVolumeTotalCapacityKey __asm__("_NSURLVolumeTotalCapacityKey");
extern URLResourceKey const URLVolumeAvailableCapacityKey __asm__("_NSURLVolumeAvailableCapacityKey");
extern URLResourceKey const URLVolumeResourceCountKey __asm__("_NSURLVolumeResourceCountKey");
extern URLResourceKey const URLVolumeSupportsPersistentIDsKey __asm__("_NSURLVolumeSupportsPersistentIDsKey");
extern URLResourceKey const URLVolumeSupportsSymbolicLinksKey __asm__("_NSURLVolumeSupportsSymbolicLinksKey");
extern URLResourceKey const URLVolumeSupportsHardLinksKey __asm__("_NSURLVolumeSupportsHardLinksKey");
extern URLResourceKey const URLVolumeSupportsJournalingKey __asm__("_NSURLVolumeSupportsJournalingKey");
extern URLResourceKey const URLVolumeIsJournalingKey __asm__("_NSURLVolumeIsJournalingKey");
extern URLResourceKey const URLVolumeSupportsSparseFilesKey __asm__("_NSURLVolumeSupportsSparseFilesKey");
extern URLResourceKey const URLVolumeSupportsZeroRunsKey __asm__("_NSURLVolumeSupportsZeroRunsKey");
extern URLResourceKey const URLVolumeSupportsCaseSensitiveNamesKey __asm__("_NSURLVolumeSupportsCaseSensitiveNamesKey");
extern URLResourceKey const URLVolumeSupportsCasePreservedNamesKey __asm__("_NSURLVolumeSupportsCasePreservedNamesKey");
extern URLResourceKey const URLVolumeSupportsRootDirectoryDatesKey __asm__("_NSURLVolumeSupportsRootDirectoryDatesKey");
extern URLResourceKey const URLVolumeSupportsVolumeSizesKey __asm__("_NSURLVolumeSupportsVolumeSizesKey");
extern URLResourceKey const URLVolumeSupportsRenamingKey __asm__("_NSURLVolumeSupportsRenamingKey");
extern URLResourceKey const URLVolumeSupportsAdvisoryFileLockingKey __asm__("_NSURLVolumeSupportsAdvisoryFileLockingKey");
extern URLResourceKey const URLVolumeSupportsExtendedSecurityKey __asm__("_NSURLVolumeSupportsExtendedSecurityKey");
extern URLResourceKey const URLVolumeIsBrowsableKey __asm__("_NSURLVolumeIsBrowsableKey");
extern URLResourceKey const URLVolumeMaximumFileSizeKey __asm__("_NSURLVolumeMaximumFileSizeKey");
extern URLResourceKey const URLVolumeIsEjectableKey __asm__("_NSURLVolumeIsEjectableKey");
extern URLResourceKey const URLVolumeIsRemovableKey __asm__("_NSURLVolumeIsRemovableKey");
extern URLResourceKey const URLVolumeIsInternalKey __asm__("_NSURLVolumeIsInternalKey");
extern URLResourceKey const URLVolumeIsAutomountedKey __asm__("_NSURLVolumeIsAutomountedKey");
extern URLResourceKey const URLVolumeIsLocalKey __asm__("_NSURLVolumeIsLocalKey");
extern URLResourceKey const URLVolumeIsReadOnlyKey __asm__("_NSURLVolumeIsReadOnlyKey");
extern URLResourceKey const URLVolumeCreationDateKey __asm__("_NSURLVolumeCreationDateKey");
extern URLResourceKey const URLVolumeURLForRemountingKey __asm__("_NSURLVolumeURLForRemountingKey");
extern URLResourceKey const URLVolumeUUIDStringKey __asm__("_NSURLVolumeUUIDStringKey");
extern URLResourceKey const URLVolumeNameKey __asm__("_NSURLVolumeNameKey");
extern URLResourceKey const URLVolumeLocalizedNameKey __asm__("_NSURLVolumeLocalizedNameKey");
extern URLResourceKey const URLVolumeIsEncryptedKey __asm__("_NSURLVolumeIsEncryptedKey");
extern URLResourceKey const URLVolumeIsRootFileSystemKey __asm__("_NSURLVolumeIsRootFileSystemKey");
extern URLResourceKey const URLVolumeSupportsCompressionKey __asm__("_NSURLVolumeSupportsCompressionKey");
extern URLResourceKey const URLVolumeSupportsFileCloningKey __asm__("_NSURLVolumeSupportsFileCloningKey");
extern URLResourceKey const URLVolumeSupportsSwapRenamingKey __asm__("_NSURLVolumeSupportsSwapRenamingKey");
extern URLResourceKey const URLVolumeSupportsExclusiveRenamingKey __asm__("_NSURLVolumeSupportsExclusiveRenamingKey");
extern URLResourceKey const URLVolumeSupportsImmutableFilesKey __asm__("_NSURLVolumeSupportsImmutableFilesKey");
extern URLResourceKey const URLVolumeSupportsAccessPermissionsKey __asm__("_NSURLVolumeSupportsAccessPermissionsKey");
extern URLResourceKey const URLVolumeSupportsFileProtectionKey __asm__("_NSURLVolumeSupportsFileProtectionKey");
extern URLResourceKey const URLVolumeAvailableCapacityForImportantUsageKey __asm__("_NSURLVolumeAvailableCapacityForImportantUsageKey");
extern URLResourceKey const URLVolumeAvailableCapacityForOpportunisticUsageKey __asm__("_NSURLVolumeAvailableCapacityForOpportunisticUsageKey");
extern URLResourceKey const URLVolumeTypeNameKey __asm__("_NSURLVolumeTypeNameKey");
extern URLResourceKey const URLVolumeSubtypeKey __asm__("_NSURLVolumeSubtypeKey");
extern URLResourceKey const URLVolumeMountFromLocationKey __asm__("_NSURLVolumeMountFromLocationKey");
extern URLResourceKey const URLIsUbiquitousItemKey __asm__("_NSURLIsUbiquitousItemKey");
extern URLResourceKey const URLUbiquitousItemHasUnresolvedConflictsKey __asm__("_NSURLUbiquitousItemHasUnresolvedConflictsKey");
extern URLResourceKey const URLUbiquitousItemIsDownloadedKey __asm__("_NSURLUbiquitousItemIsDownloadedKey");
extern URLResourceKey const URLUbiquitousItemIsDownloadingKey __asm__("_NSURLUbiquitousItemIsDownloadingKey");
extern URLResourceKey const URLUbiquitousItemIsUploadedKey __asm__("_NSURLUbiquitousItemIsUploadedKey");
extern URLResourceKey const URLUbiquitousItemIsUploadingKey __asm__("_NSURLUbiquitousItemIsUploadingKey");
extern URLResourceKey const URLUbiquitousItemPercentDownloadedKey __asm__("_NSURLUbiquitousItemPercentDownloadedKey");
extern URLResourceKey const URLUbiquitousItemPercentUploadedKey __asm__("_NSURLUbiquitousItemPercentUploadedKey");
extern URLResourceKey const URLUbiquitousItemDownloadingStatusKey __asm__("_NSURLUbiquitousItemDownloadingStatusKey");
extern URLResourceKey const URLUbiquitousItemDownloadingErrorKey __asm__("_NSURLUbiquitousItemDownloadingErrorKey");
extern URLResourceKey const URLUbiquitousItemUploadingErrorKey __asm__("_NSURLUbiquitousItemUploadingErrorKey");
extern URLResourceKey const URLUbiquitousItemDownloadRequestedKey __asm__("_NSURLUbiquitousItemDownloadRequestedKey");
extern URLResourceKey const URLUbiquitousItemContainerDisplayNameKey __asm__("_NSURLUbiquitousItemContainerDisplayNameKey");
extern URLResourceKey const URLUbiquitousItemIsExcludedFromSyncKey __asm__("_NSURLUbiquitousItemIsExcludedFromSyncKey");
extern URLResourceKey const URLUbiquitousItemIsSharedKey __asm__("_NSURLUbiquitousItemIsSharedKey");
extern URLResourceKey const URLUbiquitousSharedItemCurrentUserRoleKey __asm__("_NSURLUbiquitousSharedItemCurrentUserRoleKey");
extern URLResourceKey const URLUbiquitousSharedItemCurrentUserPermissionsKey __asm__("_NSURLUbiquitousSharedItemCurrentUserPermissionsKey");
extern URLResourceKey const URLUbiquitousSharedItemOwnerNameComponentsKey __asm__("_NSURLUbiquitousSharedItemOwnerNameComponentsKey");
extern URLResourceKey const URLUbiquitousSharedItemMostRecentEditorNameComponentsKey __asm__("_NSURLUbiquitousSharedItemMostRecentEditorNameComponentsKey");
extern URLUbiquitousItemDownloadingStatus const URLUbiquitousItemDownloadingStatusNotDownloaded __asm__("_NSURLUbiquitousItemDownloadingStatusNotDownloaded");
extern URLUbiquitousItemDownloadingStatus const URLUbiquitousItemDownloadingStatusDownloaded __asm__("_NSURLUbiquitousItemDownloadingStatusDownloaded");
extern URLUbiquitousItemDownloadingStatus const URLUbiquitousItemDownloadingStatusCurrent __asm__("_NSURLUbiquitousItemDownloadingStatusCurrent");
extern URLUbiquitousSharedItemRole const URLUbiquitousSharedItemRoleOwner __asm__("_NSURLUbiquitousSharedItemRoleOwner");
extern URLUbiquitousSharedItemRole const URLUbiquitousSharedItemRoleParticipant __asm__("_NSURLUbiquitousSharedItemRoleParticipant");
extern URLUbiquitousSharedItemPermissions const URLUbiquitousSharedItemPermissionsReadOnly __asm__("_NSURLUbiquitousSharedItemPermissionsReadOnly");
extern URLUbiquitousSharedItemPermissions const URLUbiquitousSharedItemPermissionsReadWrite __asm__("_NSURLUbiquitousSharedItemPermissionsReadWrite");
extern URLResourceKey const URLUbiquitousItemSupportedSyncControlsKey __asm__("_NSURLUbiquitousItemSupportedSyncControlsKey");
extern URLResourceKey const URLUbiquitousItemIsSyncPausedKey __asm__("_NSURLUbiquitousItemIsSyncPausedKey");
_NS_OPTIONS(NS::UInteger, URLBookmarkCreationOptions) {
    URLBookmarkCreationPreferFileIDResolution = ( 1UL << 8 ),
    URLBookmarkCreationMinimalBookmark = ( 1UL << 9 ),
    URLBookmarkCreationSuitableForBookmarkFile = ( 1UL << 10 ),
    URLBookmarkCreationWithSecurityScope = ( 1 << 11 ),
    URLBookmarkCreationSecurityScopeAllowOnlyReadAccess = ( 1 << 12 ),
    URLBookmarkCreationWithoutImplicitSecurityScope = (1 << 29),
};

_NS_OPTIONS(NS::UInteger, URLBookmarkResolutionOptions) {
    URLBookmarkResolutionWithoutUI = ( 1UL << 8 ),
    URLBookmarkResolutionWithoutMounting = ( 1UL << 9 ),
    URLBookmarkResolutionWithSecurityScope = ( 1 << 10 ),
    URLBookmarkResolutionWithoutImplicitStartAccessing = ( 1 << 15 ),
};


class URL : public NS::SecureCoding<URL>
{
public:
    static URL* alloc();
    URL*        init() const;

    static NS::URL* fileURL(NS::String* path);

    const char * fileSystemRepresentation() const;
    NS::URL*     init(NS::String* URLString);
    NS::URL*     initFileURLWithPath(NS::String* path);

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSURL;

_NS_INLINE NS::URL* NS::URL::alloc()
{
    return _NS_msg_NS__URLp_alloc((const void*)&OBJC_CLASS_$_NSURL, nullptr);
}

_NS_INLINE NS::URL* NS::URL::init() const
{
    return _NS_msg_NS__URLp_init((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::URL::fileURL(NS::String* path)
{
    return _NS_msg_NS__URLp_fileURLWithPath__NS__Stringp((const void*)&OBJC_CLASS_$_NSURL, nullptr, path);
}

_NS_INLINE const char * NS::URL::fileSystemRepresentation() const
{
    return _NS_msg_constcharp_fileSystemRepresentation((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::URL::initFileURLWithPath(NS::String* path)
{
    return _NS_msg_NS__URLp_initFileURLWithPath__NS__Stringp((const void*)this, nullptr, path);
}

_NS_INLINE NS::URL* NS::URL::init(NS::String* URLString)
{
    return _NS_msg_NS__URLp_initWithString__NS__Stringp((const void*)this, nullptr, URLString);
}
