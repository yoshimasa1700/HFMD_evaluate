#include "CDataset.h"

CDataset::CDataset()
    :imgFlag(0),
      featureFlag(0)
{
    img.clear();
    feature.clear();
}

virtual CDataset::~CDataset(){
    if(imgFlag)
        releaseImage();

    if(featureFlag)
        releaseFeatures();
}

int CDataset::loadImage(){
    cv::Mat *rgbImg, *depthImg;

    rgbImg = new cv::imread(rgb,3);
    if(rgbImg == NULL){
        std::cout << "error! rgb image file " << rgb << " not found!" << std::endl;
        return -1;
    }
    depthImg = new cv::imread(depth, CV_LOAD_IMAGE_ANYDEPTH);
    if(depthImg == NULL){
        std::cout << "error! depth image file " << depth << " not found!" << std::endl;
        return -1;
    }

    img.push_back(rgbImg, depthImg);

    imgFlag  = 1;

    return 0;
}

int CDataset::releaseImage(){
    if(imgFlag == 0){
        std::cout << "image is already released! foolish!" << std::endl;
        return -1;
    }

    for(int i = 0; i < img.size(); ++i)
        delete img.at(i);

    return 0;
}

int CDataset::extractFeatures(){
    feature.clear();
    feature.resize(32);
    for(int i = 0; i < 32; ++i)
      feature.at(i) = new cv::Mat(img.at(0)->rows, img.at(0)->cols, CV_8UC1);

    cv::cvtColor(*img.at(0), *(feature.at(0)), CV_RGB2GRAY);


    cv::Mat I_x(img.at(0)->rows, img.at(0)->cols, CV_16SC1);
    cv::Mat I_y(img.at(0)->rows, img.at(0)->cols, CV_16SC1);


    cv::Sobel(*(feature.at(0)), I_x, CV_16S, 1, 0);
    cv::Sobel(*(feature.at(0)), I_y, CV_16S, 0, 1);

    cv::convertScaleAbs(I_x, *(feature[3]), 0.25);
    cv::convertScaleAbs(I_y, *(feature[4]), 0.25);

    //std::cout << "feature[3]" << std::endl;

     /* cv::namedWindow("test"); */
     /* cv::imshow("test",*(feature[3])); */
     /* cv::waitKey(0); */
     /* cv::destroyWindow("test"); */

    // Orientation of gradients
    for(int  y = 0; y < img->rows; y++)
      for(int  x = 0; x < img->cols; x++) {
        // Avoid division by zero
        float tx = (float)I_x.at<short>(y, x) + (float)copysign(0.000001f, I_x.at<short>(y, x));
        // Scaling [-pi/2 pi/2] -> [0 80*pi]
        feature.at(1)->at<uchar>(y, x) = (uchar)(( atan((float)I_y.at<short>(y, x) / tx) + 3.14159265f / 2.0f ) * 80);
        //std::cout << "scaling" << std::endl;
        feature.at(2)->at<uchar>(y, x) = (uchar)sqrt((float)I_x.at<short>(y, x)* (float)I_x.at<short>(y, x) + (float)I_y.at<short>(y, x) * (float)I_y.at<short>(y, x));
      }

    // Magunitude of gradients
    for(int y = 0; y < img->rows; y++)
        for(int x = 0; x < img->cols; x++ ) {
      feature.at(2)->at<uchar>(y, x) = (uchar)sqrt(I_x.at<short>(y, x)*I_x.at<short>(y, x) + I_y.at<short>(y, x) * I_y.at<short>(y, x));
        }

    hog.extractOBin(feature[1], feature[2], feature, 7);

    // calc I_xx I_yy
    cv::Sobel(*(feature.at(0)), I_x, CV_16S, 2, 0);
    cv::Sobel(*(feature.at(0)), I_y, CV_16S, 0, 2);

    cv::convertScaleAbs(I_x, *(feature[5]), 0.25);
    cv::convertScaleAbs(I_y, *(feature[6]), 0.25);

    cv::Mat img_Lab;
    cv::cvtColor(*img.at(0), img_Lab, CV_RGB2Lab);
    cv::vector<cv::Mat> tempfeature(3);

    cv::split(img_Lab, tempfeature);

    for(int i = 0; i < 3; ++i)
      tempfeature.at(i).copyTo(*(feature.at(i)));

    // min filter
    for(int c = 0; c < 16; ++c)
      minFilter(feature[c], feature[c + 16], 5);

  //  for(int i = 0; i < 32; ++i){
  //      cv::namedWindow("test");
  //      cv::imshow("test",*feature.at(i));
  //      cv::waitKey(0);
  //      cv::destroyWindow("test");
  //  }

    for(int c = 0; c < 16; ++c)
      maxFilter(feature[c], feature[c], 5);

    cv::Mat *tempDepth = new *img.at(1);
    feature.push_back(tempDepth);

    featureFlag = 1;

    return 0;
}

int CDataset::releaseFeatures(){
    if(featureFlag == 0){
        std::cout << "image is already released! foolish!" << std::endl;
        return -1;
    }

    for(int i = 0; i < feature.size(); ++i)
        delete feature.at(i);

    return 0;
}

void CDataset::minFilter(cv::Mat* src, cv::Mat* des, int fWind) {
    int d = (fWind - 1) / 2;
    cv::Rect roi;
    cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

    for(int y = 0; y < src->rows - fWind; ++y){ //for image height
        if(y < fWind)
            roi = cv::Rect(0, 0, src->cols, fWind - y);
        else
            roi = cv::Rect(0, y, src->cols, fWind);

        cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MIN);

        roi = cv::Rect(0, y + d, src->cols, 1);
        //cv::Mat roiDesTemp(desTemp, roi);
        vTemp.copyTo(desTemp(roi));
    }// For image height

    for(int x = 0; x < src->cols - fWind; ++x){ // for image width
        if(x < d)
            roi = cv::Rect(0, 0, fWind - x, src->rows);
        else
            roi = cv::Rect(x, 0, fWind, src->rows);

        cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MIN);

        roi = cv::Rect(x + d, 0, 1, src->rows);
        cv::Mat roiDesTemp((*des), roi);
        vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
    } // for image width
}

void CDataset::maxFilter(cv::Mat* src, cv::Mat* des, int fWind) {
  int d = (fWind - 1) / 2;
  cv::Rect roi;
  cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

    for(int y = 0; y < src->rows - fWind; ++y){ //for image height
      if(y < fWind)
    roi = cv::Rect(0, 0, src->cols, fWind - y);
      else
    roi = cv::Rect(0, y, src->cols, fWind);

      cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MAX);

      roi = cv::Rect(0, y + d, src->cols, 1);
      cv::Mat roiDesTemp(desTemp, roi);
      vTemp.copyTo(desTemp(roi));
    }// For image height

  for(int x = 0; x < src->cols - fWind; ++x){ // for image width
    if(x < d)
      roi = cv::Rect(0, 0, fWind - x, src->rows);
    else
      roi = cv::Rect(x, 0, fWind, src->rows);

    cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MAX);

    roi = cv::Rect(x + d, 0, 1, src->rows);
    cv::Mat roiDesTemp(*des, roi);
    vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
  } // for image width

}
