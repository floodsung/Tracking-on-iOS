//
//  ViewController.m
//  VisionSeg
//
//  Created by FloodSurge on 15/7/14.
//  Copyright (c) 2015年 FloodSurge. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <opencv2/opencv.hpp>

#import <opencv2/imgproc/types_c.h>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>

#import "CompressiveTracker.h"
#import "CMT.h"
#import "TLD.h"
#import "color_tracker.h"
#import "Camshift.h"

#import "StruckTracker.h"
#import "Config.h"


#define RATIO  640.0/568.0

using namespace cv;
using namespace cv::colortracker;
using namespace std;
using namespace tld;


typedef enum {
    CT_TRACKER,
    TLD_TRACKER,
    CMT_TRACKER,
    COLOR_TRACKER,
    CAMSHIFT_TRACKER,
    STRUCK_TRACKER,
}TrackType;

@interface ViewController ()<CvVideoCameraDelegate>
{
    CGPoint rectLeftTopPoint;
    CGPoint rectRightDownPoint
    ;
    
    TrackType trackType;
    
    // CT Tracker
    CompressiveTracker *ctTracker;
    cv::Rect selectBox;
    cv::Rect initCTBox;
    cv::Rect box;
    bool beginInit;
    bool startTracking;
    
    // CMT Tracker
    cmt::CMT *cmtTracker;
    
    // TLD Tracker
    tld::TLD *tldTracker;
    
    // Color Tracker
    ColorTracker *colorTracker;
    
    // Camshift Tracker
    Camshift *camshift;
    
    // Struck Tracker
    StruckTracker *struckTracker;
    
    
}
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (nonatomic,strong) CvVideoCamera *videoCamera;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    self.imageView.frame = CGRectMake(0, 0, 568, 480 * 568/640.0);
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    self.videoCamera.defaultFPS = 30;
    [self.videoCamera start];
    
    rectLeftTopPoint = CGPointZero;
    rectRightDownPoint = CGPointZero;
    
    beginInit = false;
    startTracking = false;
    
    trackType = STRUCK_TRACKER;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)reset
{
    startTracking = false;
    beginInit = false;
    
    rectLeftTopPoint.x = 0;
    rectRightDownPoint.x = 0;
}

- (IBAction)COLOR:(id)sender
{
    trackType = COLOR_TRACKER;
    [self reset];
}

- (IBAction)CT:(id)sender
{
    trackType = CT_TRACKER;
    [self reset];
}
- (IBAction)TLD:(id)sender
{
    trackType = TLD_TRACKER;
    [self reset];
}

- (IBAction)CMT:(id)sender
{
    trackType = CMT_TRACKER;
    [self reset];
}
- (IBAction)Camshift:(id)sender {
    trackType = CAMSHIFT_TRACKER;
    [self reset];
}

- (IBAction)Struck:(id)sender
{
    trackType = STRUCK_TRACKER;
    [self reset];
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    startTracking = false;
    beginInit = false;
    UITouch *aTouch  = [touches anyObject];
    rectLeftTopPoint = [aTouch locationInView:self.imageView];
    
    NSLog(@"touch in :%f,%f",rectLeftTopPoint.x,rectLeftTopPoint.y);
    rectRightDownPoint = CGPointZero;
    selectBox = cv::Rect(rectLeftTopPoint.x * RATIO,rectLeftTopPoint.y * RATIO,0,0);
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
    UITouch *aTouch  = [touches anyObject];
    rectRightDownPoint = [aTouch locationInView:self.imageView];
    
    //NSLog(@"touch move :%f,%f",rectRightDownPoint.x,rectRightDownPoint.y);
    
    
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
    UITouch *aTouch  = [touches anyObject];
    rectRightDownPoint = [aTouch locationInView:self.imageView];
    
    NSLog(@"touch end :%f,%f",rectRightDownPoint.x,rectRightDownPoint.y);
    selectBox.width = abs(rectRightDownPoint.x * RATIO - selectBox.x);
    selectBox.height = abs(rectRightDownPoint.y * RATIO - selectBox.y);
    beginInit = true;
    initCTBox = selectBox;
    
}

- (void)processImage:(cv::Mat &)image
{
    if (rectLeftTopPoint.x != 0 && rectLeftTopPoint.y != 0 && rectRightDownPoint.x != 0 && rectRightDownPoint.y != 0 && !beginInit && !startTracking) {
        
        rectangle(image, cv::Point(rectLeftTopPoint.x * RATIO,rectLeftTopPoint.y * RATIO), cv::Point(rectRightDownPoint.x * RATIO,rectRightDownPoint.y * RATIO), Scalar(0,0,255));
    }
    
    switch (trackType) {
        case CT_TRACKER:
            [self compressiveTracking:image];
            break;
        case CMT_TRACKER:
            [self cmtTracking:image];
            break;
        case TLD_TRACKER:
            [self tldTracking:image];
            break;
        case COLOR_TRACKER:
            [self colorTracking:image];
        case CAMSHIFT_TRACKER:
            [self camshiftTracking:image];
        case STRUCK_TRACKER:
            [self struckTracking:image];
        default:
            break;
    }

}

- (void)struckTracking:(cv::Mat &)image
{
    Mat img_gray;
    cvtColor(image,img_gray,CV_BGR2GRAY);
    //cv::flip(img_gray, img_gray, 1);
    if (beginInit) {
        startTracking = true;
        beginInit = false;
        
        NSString *path = [[NSBundle mainBundle] pathForResource:@"config" ofType:@"txt"];
        string configPath = [path UTF8String];
        Config conf(configPath);
        cout << conf << endl;
        srand(conf.seed);
        if (struckTracker != NULL) {
            delete struckTracker;
        }
        
        struckTracker = new StruckTracker(conf);
        
        FloatStruckRect floatRect;
        floatRect.Set(initCTBox.x, initCTBox.y, initCTBox.width, initCTBox.height);
        struckTracker->Initialise(img_gray, floatRect);
        NSLog(@"Struck tracker init");
    }
    
    if (startTracking) {
        
        NSLog(@"Struck Tracker process...");
        struckTracker->Track(img_gray);
        FloatStruckRect bb = struckTracker->GetBB();
        cv::Rect bb_rot;
        bb_rot.x = (int)bb.XMin();
        bb_rot.y = (int)bb.YMin();
        bb_rot.width = (int)bb.Width();
        bb_rot.height = (int)bb.Height();
        
        rectangle(image, bb_rot, Scalar(125,255,0),1);
    }
}

- (void)camshiftTracking:(cv::Mat &)image
{
    if (beginInit) {
        if (camshift != NULL) {
            delete camshift;
        };
        camshift = new Camshift();
        camshift->initialize(image, initCTBox);
        NSLog(@"Camshift track init!");
        startTracking = true;
        beginInit = false;
    }
    
    if (startTracking) {
        NSLog(@"Camshift Track process...");
        camshift->processFrame(image);
        
        RotatedRect rect = camshift->objectBox;
        Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            line(image, vertices[i], vertices[(i+1)%4], Scalar(255,0,255));
        }
    }
}

- (void)colorTracking:(cv::Mat &)image
{
    if (beginInit) {
        ColorTrackerParameters params;
        params.visualization = 0;
        params.init_pos.x = initCTBox.x + initCTBox.width/2;
        params.init_pos.y = initCTBox.y + initCTBox.height/2;
        params.wsize = initCTBox.size();
        
        if (colorTracker != NULL) {
            delete colorTracker;
        }
        
        colorTracker = new ColorTracker(params);
        colorTracker->init_tracking();

        NSLog(@"color track init!");
        startTracking = true;
        beginInit = false;
    }
    
    if (startTracking) {
        NSLog(@"color track process...");
        cv::Rect rect = colorTracker->track_frame(image);
        
        rectangle(image, rect, Scalar(125,255,0),1);
    }
    
    
}


- (void)tldTracking:(cv::Mat &)image
{
    Mat img_gray;
    cvtColor(image,img_gray,CV_BGR2GRAY);
    if (beginInit) {
        if (tldTracker != NULL) {
            tldTracker->release();
            delete tldTracker;
        }
        
        tldTracker = new tld::TLD();
        tldTracker->detectorCascade->imgWidth = img_gray.cols;
        tldTracker->detectorCascade->imgHeight = img_gray.rows;
        tldTracker->detectorCascade->imgWidthStep = img_gray.step;
        tldTracker->selectObject(img_gray, &initCTBox);
        NSLog(@"tld track init!");
        startTracking = true;
        beginInit = false;
    }
    
    if (startTracking) {
        NSLog(@"tld process...");
        tldTracker->processImage(image);
        
        if (tldTracker->currBB != NULL) {
            
            rectangle(image, *tldTracker->currBB, Scalar(0,255,0),1);
        }

    }
}


- (void)cmtTracking:(cv::Mat &)image
{
    Mat img_gray;
    cvtColor(image,img_gray,CV_RGB2GRAY);
    
    if (beginInit) {
        if (cmtTracker != NULL) {
            delete cmtTracker;
        }
        cmtTracker = new cmt::CMT();
        cmtTracker->initialize(img_gray,initCTBox);
        NSLog(@"cmt track init!");
        startTracking = true;
        beginInit = false;
    }
    
    if (startTracking) {
        NSLog(@"cmt process...");
        cmtTracker->processFrame(img_gray);
        
        for(size_t i = 0; i < cmtTracker->points_active.size(); i++)
        {
            circle(image, cmtTracker->points_active[i], 2, Scalar(255,0,0));
        }
        
        
        RotatedRect rect = cmtTracker->bb_rot;
        Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            line(image, vertices[i], vertices[(i+1)%4], Scalar(255,0,255));
        }
        
        
    }
}

- (void)compressiveTracking:(cv::Mat &)image
{
    Mat img_gray;
    cvtColor(image,img_gray,CV_RGB2GRAY);
    // 初始化选择框，初始化ct算法
    if (beginInit)
    {
        startTracking=true;
        if (ctTracker != NULL) {
            delete ctTracker;
        }
        ctTracker = new CompressiveTracker;
        ctTracker->init(img_gray,initCTBox);
        NSLog(@"init CT Box: %d,%d,%d,%d",initCTBox.x,initCTBox.y,initCTBox.width,initCTBox.height);
        box = initCTBox;
        rectangle(image, initCTBox, Scalar(0,0,255),1);
        beginInit =false;
    }
    
    //  采用ct算法进行跟踪
    if (startTracking)
    {
        /** if (box.size())
         {
         
         }*/
        ctTracker->processFrame(img_gray, box);
        
        rectangle(image, box, Scalar(0,0,255),1);
        
    }
}

@end
