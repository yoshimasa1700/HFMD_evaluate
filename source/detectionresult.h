#ifndef DETECTIONRESULT_H
#define DETECTIONRESULT_H

#include <string>
#include <opencv2/opencv.hpp>

class detectionResult
{
public:
    detectionResult();
    std::string className;
    cv::Rect bbox;
};

#endif // DETECTIONRESULT_H
