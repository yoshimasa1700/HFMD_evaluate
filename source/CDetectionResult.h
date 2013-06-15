#ifndef DETECTIONRESULT_H
#define DETECTIONRESULT_H

#include <string>
#include <opencv2/opencv.hpp>

class CDetectedClass
{
public:
    CDetectedClass(){}
    virtual ~CDetectedClass(){}

    std::string name;
    cv::Rect bbox;
    cv::Point centerPoint;

    double error;
    float score;
    std::string nearestClass;
};

class CDetectionResult
{
public:
    CDetectionResult(){
        voteImage.clear();
        detectedClass.clear();
    }
    virtual ~CDetectionResult(){
    }

    std::vector<cv::Mat> voteImage;

    std::vector<CDetectedClass> detectedClass;
};

#endif // DETECTIONRESULT_H
