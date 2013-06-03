#include "CRForest.h"
#include <opencv2/opencv.hpp>
#include <boost/timer.hpp>
//#include "CRTree.h"

#include "util.h"

using namespace std;

void loadImage(CDataset &dataSet, std::vector<cv::Mat*> &image){
    cv::Mat* rgb, *depth;//, *mask;

    rgb = new cv::Mat();
    depth = new cv::Mat();
    //mask = new cv::Mat();

    //std::vector<cv::Mat> rgbSplited;

//    *mask = cv::imread(dataSet.imageFilePath
//                       + dataSet.maskImageName,
//                       CV_LOAD_IMAGE_ANYCOLOR).clone();

    // load RGB image
    *rgb = cv::imread(dataSet.imageFilePath
                      + dataSet.rgbImageName,
                      CV_LOAD_IMAGE_ANYCOLOR).clone();

    // load Depth image
    *depth = cv::imread(dataSet.imageFilePath
                        + dataSet.depthImageName,
                        CV_LOAD_IMAGE_ANYDEPTH).clone();

//    cv::Point tempPoint;
//    tempPoint.x = (*rgb).cols / 2;
//    tempPoint.y = (*rgb).rows / 2;

    //dataSet.centerPoint.push_back(tempPoint);

    // masking image
    // for(int k = 0;k < rgb->cols; ++k)
    //   for(int l = 0;l < rgb->rows; ++l){
    //     if(!(bool)mask->at<char>(l, k))
    // 	depth->at<short>(l, k) = 0;
    //     for(int j = 0;j < 3; ++j)
    // 	if(!(bool)mask->at<char>(l, k))
    // 	  rgb->at<cv::Vec3b>(l, k)[j] = 0;
    //   }

    //rgbSplited.resize(rgb.channels());

    //cv::split(rgb, rgbSplited);

    //for(int i = 0; i < rgbSplited.size(); ++i)
    image.push_back(rgb);//rgbSplited.at(i));
    image.push_back(depth);

    //delete mask;
}

void loadTestFile(CConfig conf, std::vector<CDataset> &dataSet){
    std::string testfilepath = conf.testPath + PATH_SEP +  conf.testData;
    int n_folders;
    int n_files;
    std::vector<std::string> testimagefolder;
    CDataset temp;
    std::vector<CDataset> tempDataSet;
    std::string testDataListPath;
    int dataSetNum;

    cv::Point tempPoint;

    std::ifstream in(testfilepath.c_str());
    if(!in.is_open()){
        std::cout << "test data floder list is not found!" << std::endl;
        exit(1);
    }
    in >> n_folders;

    testimagefolder.resize(n_folders);
    for(int i = 0;i < n_folders; ++i)
        in >> testimagefolder.at(i);
    in.close();

    //read train file name and grand truth from file
    tempDataSet.resize(0);
    for(int i = 0;i < n_folders; ++i){
        testDataListPath
                = conf.testPath + PATH_SEP + testimagefolder.at(i)
                + PATH_SEP + conf.testdatalist;
        temp.imageFilePath
                = conf.testPath + PATH_SEP + testimagefolder.at(i) + PATH_SEP;
        //std::cout << trainDataListPath << std::endl;
        std::ifstream testDataList(testDataListPath.c_str());
        if(testDataList.is_open()){

            testDataList >> n_files;
            //std::cout << "number of file: " << n_files << std::endl;
            for(int j = 0;j < n_files; ++j){
                //read file names
                testDataList >> temp.rgbImageName;
                testDataList >> temp.depthImageName;
                testDataList >> temp.maskImageName;

                //read bounding box
                // testDataList >> temp.bBox.x;
                // testDataList >> temp.bBox.y;
                // testDataList >> temp.bBox.width;
                // testDataList >> temp.bBox.height;

                temp.centerPoint.resize(0);

                //read center point
                //testDataList >> tempPoint.x;//temp.centerPoint.x;
                //testDataList >> tempPoint.y;

                //temp.centerPoint.push_back(tempPoint);

                //read class name
                std::string tempClassName;
                testDataList >> tempClassName;
                temp.className.push_back(tempClassName);

                //read angle grand truth
                double tempAngle;
                testDataList >> tempAngle;

                temp.angles.push_back(tempAngle);

                //show read data *for debug
                //temp.showDataset();

                tempDataSet.push_back(temp);
            }
            testDataList.close();
        }
    }
    dataSetNum = tempDataSet.size();
    for(int j = 0;j < dataSetNum; ++j)
        dataSet.push_back(tempDataSet.at(j));
    std::cout << "test data number: " << dataSet.size() << std::endl;
}

void loadTestFileMultiObject(CConfig conf, std::vector<CDataset> &dataSet){
    std::string testfilepath = conf.testPath + PATH_SEP +  conf.testData;
    int n_folders;
    int n_files;
    std::vector<std::string> testimagefolder;
    CDataset temp;
    std::vector<CDataset> tempDataSet;
    std::string testDataListPath;
    int dataSetNum;

    cv::Point tempPoint;

    std::ifstream in(testfilepath.c_str());
    if(!in.is_open()){
        std::cout << "test data floder list is not found!" << std::endl;
        exit(1);
    }
    in >> n_folders;

    testimagefolder.resize(n_folders);
    for(int i = 0;i < n_folders; ++i)
        in >> testimagefolder.at(i);
    in.close();

    //read train file name and grand truth from file
    tempDataSet.resize(0);
    for(int i = 0;i < n_folders; ++i){
        testDataListPath
                = conf.testPath + PATH_SEP + testimagefolder.at(i)
                + PATH_SEP + conf.testdatalist;
        temp.imageFilePath
                = conf.testPath + PATH_SEP + testimagefolder.at(i) + PATH_SEP;
        //std::cout << trainDataListPath << std::endl;
        std::ifstream testDataList(testDataListPath.c_str());
        if(testDataList.is_open()){
            testDataList >> n_files;
            //std::cout << "number of file: " << n_files << std::endl;
            for(int j = 0;j < n_files; ++j){
                //read file names
                testDataList >> temp.rgbImageName;
                testDataList >> temp.depthImageName;
                testDataList >> temp.maskImageName;

                //read bounding box
                // testDataList >> temp.bBox.x;
                // testDataList >> temp.bBox.y;
                // testDataList >> temp.bBox.width;
                // testDataList >> temp.bBox.height;

                temp.centerPoint.resize(0);

                //read center point
                //testDataList >> tempPoint.x;//temp.centerPoint.x;
                //testDataList >> tempPoint.y;

                //temp.centerPoint.push_back(tempPoint);

                std::string tempClassName;
                cv::Point tempPoint;
                double tempAngle;
                do{
                    //read class name
                    testDataList >> tempClassName;

                    if(tempClassName != "EOL"){
                        temp.className.push_back(tempClassName);
                        testDataList >> tempPoint.x;
                        testDataList >> tempPoint.y;
                        temp.centerPoint.push_back(tempPoint);
                        testDataList >> tempAngle;
                        temp.angles.push_back(tempAngle);
                    }
                }while(tempClassName != "EOL");

                tempDataSet.push_back(temp);
                temp.centerPoint.clear();
                temp.className.clear();
            }
            testDataList.close();
        }
    }
    dataSetNum = tempDataSet.size();
    for(int j = 0;j < dataSetNum; ++j)
        dataSet.push_back(tempDataSet.at(j));
    std::cout << "test data number: " << dataSet.size() << std::endl;
}

void detect(const CRForest &forest, CConfig conf){
    std::vector<CDataset> dataSet;
    vector<cv::Mat*> image;

    //std::fstream result("detectionResult.txt", std::ios::out);

    //set dataset
    dataSet.clear();
    loadTestFileMultiObject(conf,dataSet);


    for(int i = 0; i < dataSet.size(); ++i){
        dataSet.at(i).showDataset();

        image.clear();
        loadImage(dataSet.at(i),image);
//         cv::namedWindow("test");
//         cv::imshow("test",*(image.at(0)));
//         cv::waitKey(0);
//         cv::destroyWindow("test");
        forest.detection(dataSet.at(i),image);
    }

    //result.close();
}



int main(int argc, char* argv[]){

    CConfig		conf;	 // setting
    std::vector<CDataset> dataSet; // training data name list and grand truth

    //read argument
    //check argument
    if(argc < 2) {
        cout << "Usage: ./learning [config.xml]"<< endl;
        conf.loadConfig("config.xml");
    } else
        conf.loadConfig(argv[1]);

    if(argc < 3)
        conf.off_tree = 0;
    else
        conf.off_tree = atoi(argv[2]);

    // create random forest class
    CRForest forest(conf);

    forest.loadForest();

    //create tree directory
    string opath(conf.outpath);
    opath.erase(opath.find_last_of(PATH_SEP));
    string execstr = "mkdir ";
    execstr += opath;
    system( execstr.c_str() );

    // learning
    //forest.learning();
    detect(forest, conf);

    return 0;
}

