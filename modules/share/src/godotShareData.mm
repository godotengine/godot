#include "godotShareData.h"


#import "app_delegate.h"


GodotShareData::GodotShareData() {
    ERR_FAIL_COND(instance != NULL);
    instance = this;
}

GodotShareData::~GodotShareData() {
    instance = NULL;
}



void GodotShareData::shareText(const String &title, const String &subject, const String &text) {
    
    UIViewController *root_controller = [[UIApplication sharedApplication] delegate].window.rootViewController;
    
    NSString * ns_text = [NSString stringWithCString:text.utf8().get_data() encoding:NSUTF8StringEncoding];
    NSString * ns_subject = [NSString stringWithCString:subject.utf8().get_data() encoding:NSUTF8StringEncoding];
    
    NSArray * shareItems = @[ns_text];
    
    UIActivityViewController * avc = [[UIActivityViewController alloc] initWithActivityItems:shareItems applicationActivities:nil];
    [avc setValue:ns_subject forKey:@"subject"];
    //if iPhone
    if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone) {
        [root_controller presentViewController:avc animated:YES completion:nil];
    }
    //if iPad
    else {
        // Change Rect to position Popover
        avc.modalPresentationStyle = UIModalPresentationPopover;
        avc.popoverPresentationController.sourceView = root_controller.view;
        avc.popoverPresentationController.sourceRect = CGRectMake(CGRectGetMidX(root_controller.view.bounds), CGRectGetMidY(root_controller.view.bounds),0,0);
        avc.popoverPresentationController.permittedArrowDirections = UIPopoverArrowDirection(0);
        [root_controller presentViewController:avc animated:YES completion:nil];
    }
}

void GodotShareData::shareImage(const String &path, const String &title, const String &subject, const String &text) {
    UIViewController *root_controller = [[UIApplication sharedApplication] delegate].window.rootViewController;
    
    NSString * ns_text = [NSString stringWithCString:text.utf8().get_data() encoding:NSUTF8StringEncoding];
    NSString * ns_subject = [NSString stringWithCString:subject.utf8().get_data() encoding:NSUTF8StringEncoding];
    NSString * imagePath = [NSString stringWithCString:path.utf8().get_data() encoding:NSUTF8StringEncoding];
    
    UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
    
    NSArray * shareItems = @[ns_text, image];
    
    UIActivityViewController * avc = [[UIActivityViewController alloc] initWithActivityItems:shareItems applicationActivities:nil];
    [avc setValue:ns_subject forKey:@"subject"];
     //if iPhone
    if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiomPhone) {
        [root_controller presentViewController:avc animated:YES completion:nil];
    }
    //if iPad
    else {
        // Change Rect to position Popover
        avc.modalPresentationStyle = UIModalPresentationPopover;
        avc.popoverPresentationController.sourceView = root_controller.view;
        avc.popoverPresentationController.sourceRect = CGRectMake(CGRectGetMidX(root_controller.view.bounds), CGRectGetMidY(root_controller.view.bounds),0,0);
        avc.popoverPresentationController.permittedArrowDirections = UIPopoverArrowDirection(0);
        [root_controller presentViewController:avc animated:YES completion:nil];
    }
}



void GodotShareData::_bind_methods() {
    ClassDB::bind_method(D_METHOD("shareText"), &GodotShareData::shareText);
    ClassDB::bind_method(D_METHOD("shareImage"), &GodotShareData::shareImage);
}
