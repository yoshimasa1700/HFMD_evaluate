#include "util.h"

boost::lagged_fibonacci1279 nCk::gen = boost::lagged_fibonacci1279();

float calcSumOfDepth(cv::Mat &depth, const CConfig &conf){
    cv::Mat convertedDepth = cv::Mat(depth.rows, depth.cols, CV_8U);
    cv::Mat integralMat = cv::Mat(depth.rows + 1, depth.cols+1, CV_32F);
    cv::Mat temp1,temp2;
    depth.convertTo(convertedDepth,CV_8U,255.0/(double)(conf.maxdist - conf.mindist));

//    cv::namedWindow("test");
//    cv::imshow("test",convertedDepth);
//    cv::waitKey(0);
//    cv::destroyWindow("test");

    cv::integral(convertedDepth,integralMat,temp1,temp2,CV_32F);
    return integralMat.at<int>(depth.rows, depth.cols);
}


void loadTrainPosFile(CConfig conf, std::vector<CPosDataset> &posSet)
{
    std::string traindatafilepath = conf.trainpath + PATH_SEP +  conf.traindatafile;
    int n_folders;
    int n_files;
    std::vector<std::string> trainimagefolder;
    std::vector<CPosDataset> tempDataSet;
    std::string trainDataListPath;
    int dataSetNum;
    CClassDatabase database;
    cv::Point tempPoint;
    nCk nck;

    posSet.clear();

    //read train data folder list
    std::ifstream in(traindatafilepath.c_str());
    if(!in.is_open()){
        std::cout << "train data floder list is not found!" << std::endl;
        exit(1);
    }

    // read train pos folder
    in >> n_folders;
    std::cout << "number of training data folders : " << n_folders << std::endl;
    trainimagefolder.resize(n_folders);
    for(int i = 0;i < n_folders; ++i)
        in >> trainimagefolder.at(i);
    std::cout << trainimagefolder.at(0) << std::endl;

    // close train pos data folder list
    in.close();

    std::cout << "train folders lists : " << std::endl;
    //read train file name and grand truth from file
    tempDataSet.clear();
    for(int i = 0;i < n_folders; ++i){
        trainDataListPath
                = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP + conf.traindatalist;

        std::cout << trainDataListPath << std::endl;
        std::string imageFilePath
                = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP;

        std::ifstream trainDataList(trainDataListPath.c_str());
        if(!trainDataList.is_open()){
            std::cout << "can't read " << trainDataListPath << std::endl;
            exit(1);
        }

        trainDataList >> n_files;

        for(int j = 0;j < n_files; ++j){
            CPosDataset posTemp;
            std::string nameTemp;


            //            posTemp.angles.clear();
            //            posTemp.centerPoint.clear();

            //read file names
            trainDataList >> nameTemp;
            posTemp.setRgbImagePath(imageFilePath + nameTemp);
	    

            trainDataList >> nameTemp;
            posTemp.setDepthImagePath(imageFilePath + nameTemp);

            trainDataList >> nameTemp;// dummy

            //read class name
            std::string tempClassName;
            trainDataList >> tempClassName;
            //temp.className.push_back(tempClassName);
            posTemp.setClassName(tempClassName);

            // read image size
            cv::Mat tempImage = cv::imread(posTemp.getRgbImagePath(),3);
            cv::Size tempSize = cv::Size(tempImage.cols, tempImage.rows);

            // set center point
            tempPoint = cv::Point(tempImage.cols / 2, tempImage.rows / 2);
            posTemp.setCenterPoint(tempPoint);

            // registre class param to class database
            database.add(posTemp.getParam()->getClassName(), tempSize, 0);

            //read angle grand truth
            double tempAngle;
            trainDataList >> tempAngle;
            posTemp.setAngle(tempAngle);

            tempDataSet.push_back(posTemp);
        }

        trainDataList.close();
    }

    dataSetNum = tempDataSet.size();
    int dataOffset = 0;
    database.show();
    for(int j = 0;j < database.vNode.size(); j++){
        std::set<int> chosenData = nck.generate(database.vNode.at(j).instances, conf.imagePerTree);

        std::set<int>::iterator ite = chosenData.begin();

        while(ite != chosenData.end()){
            posSet.push_back(tempDataSet.at(*ite + dataOffset));
            //std::cout << tempDataSet.at(*ite + dataOffset).getRgbImagePath() << std::endl;
            ite++;
        }
        dataOffset += database.vNode.at(j).instances;
    }


    //    std::cout << "show chosen dataset" << std::endl;
    //    for(int i = 0; i < dataSet.size(); ++i){
    //        dataSet.at(i).showDataset();
    //    }
}

void loadTrainNegFile(CConfig conf, std::vector<CNegDataset> &negSet)
{
    //std::string traindatafilepath
    int n_folders;
    std::ifstream in_F((conf.negDataPath + PATH_SEP + conf.negFolderList).c_str());
    in_F >> n_folders;
    std::cout << conf.negDataPath + PATH_SEP + conf.negFolderList << std::endl;

    for(int j = 0; j < n_folders; ++j){
        int n_files;
        CDataset temp;
        //std::string trainDataListPath = conf.negDataPath + PATH_SEP +  conf.negDataList;

        std::string negDataFolderName;
        in_F >> negDataFolderName;
        std::string negDataFilePath = conf.negDataPath + PATH_SEP + negDataFolderName +  PATH_SEP;

        //read train data folder list
        std::ifstream in((negDataFilePath + conf.negDataList).c_str());
        if(!in.is_open()){
            std::cout << negDataFilePath << " train negative data floder list is not found!" << std::endl;
            exit(1);
        }

        std::cout << negDataFilePath << " loaded!" << std::endl;
        in >> n_files;

        //negSet.clear();

        for(int i = 0; i < n_files; ++i){
            CNegDataset negTemp;
            std::string tempName;

            in >> tempName;
            negTemp.setRgbImagePath(negDataFilePath + tempName);

            in >> tempName;
            negTemp.setDepthImagePath(negDataFilePath + tempName);

            negSet.push_back(negTemp);
        }

        in.close();
    }

    in_F.close();
    //    for(int i = 0; i < dataSet.size(); ++i)
    //        dataSet.at(i).showDataset();
}

// extract patch from images
// !!!!!!coution!!!!!!!
// this function is use for negatime mode!!!!!
void extractPosPatches(std::vector<CPosDataset> &posSet,
                       std::vector<CPosPatch> &posPatch,
                       CConfig conf,
                       const int treeNum,
                       const CClassDatabase &classDatabase){
    cv::Rect tempRect;

    std::vector<CPosPatch> tPosPatch(0);//, posPatch(0);
    std::vector<std::vector<CPosPatch> > patchPerClass(classDatabase.vNode.size());
    int pixNum;
    nCk nck;
    int classNum = 0;
    cv::Mat roi;

    posPatch.clear();

    tempRect.width  = conf.p_width;
    tempRect.height = conf.p_height;

    std::cout << "image num is " << posSet.size();

    std::cout << "extracting patch from image" << std::endl;
    for(int l = 0;l < posSet.size(); ++l){
        std::cout << posSet.at(l).getRgbImagePath() << std::endl;
//                    cv::namedWindow("test");
//                    cv::imshow("test", *posSet.at(l).img.at(0));
//                    cv::waitKey(0);
//                    cv::destroyWindow("test");
        for(int j = 0; j < posSet.at(l).img.at(0)->cols - conf.p_width; j += conf.stride){
            for(int k = 0; k < posSet.at(l).img.at(0)->rows - conf.p_height; k += conf.stride){
                tempRect.x = j;
                tempRect.y = k;
                pixNum = 0;
                int centerDepthFlag = 1;

                // detect negative patch
//                if(conf.learningMode != 2 && posSet.at(l).img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0){

//                    roi = (*posSet.at(l).img.at(1))(cv::Rect(j, k, conf.p_width, conf.p_height));
//                    pixNum = calcSumOfDepth(roi,conf);
//                }
//                    for(int m = j; m < j + conf.p_width; ++m){
//                        for(int n = k; n < k + conf.p_height; ++n){
//                            if(posSet.at(l).img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0)
//                                pixNum += (int)(posSet.at(l).img.at(1)->at<ushort>(n, m));
//                        }
//                    }

                // set patch class
                classNum = classDatabase.search(posSet.at(l).getParam()->getClassName());//dataSet.at(l).className.at(0));
                if(classNum == -1){
                    std::cout << "class not found!" << std::endl;
                    exit(-1);
                }

                //tPatch.setPatch(temp, image.at(l), dataSet.at(l), classNum);


                CPosPatch posTemp(&posSet.at(l),tempRect);
                //if (conf.learningMode == 2){// || pixNum > 0){
                    tPosPatch.push_back(posTemp);
                    patchPerClass.at(classNum).push_back(posTemp);

                //} // if
            }//x
        }//y
    }//allimages

    for(int w = 0; w < patchPerClass.size(); ++w){
        std::cout << patchPerClass.at(w).size() << " ";
    }
    std::cout << std::endl;

    std::vector<int> patchNum(patchPerClass.size(),conf.patchRatio);

    for(int i = 0; i < patchPerClass.size(); ++i){
        if(i == treeNum % patchPerClass.size())
            patchNum.at(i) = conf.patchRatio;
        else
            patchNum.at(i) = conf.patchRatio * conf.acPatchRatio;
    }

    for(int w = 0; w < patchPerClass.size(); ++w){
        std::cout << patchPerClass.at(w).size() << " ";
    }

    // choose patch from each image
    for(int i = 0; i < patchPerClass.size(); ++i){
        if(patchPerClass.at(i).size() > conf.patchRatio){

            std::set<int> chosenPatch = nck.generate(patchPerClass.at(i).size(),patchNum.at(i));// conf.patchRatio);//totalPatchNum * conf.patchRatio);
            std::set<int>::iterator ite = chosenPatch.begin();

            while(ite != chosenPatch.end()){
                //std::cout << "this is for debug ite is " << tPosPatch.at(*ite).center << std::endl;
                //std::cout <<posPatch.at(i)
                //std::cout << patchPerClass.at(i).at(*ite).getRgbImageFilePath() << std::endl;
                posPatch.push_back(patchPerClass.at(i).at(*ite));
                ite++;
            }
        }else{
//                std::cout << classNum << std::endl;
////            cv::namedWindow("test");
////            cv::imshow("test", *posSet.at(0).img.at(0));
////            cv::waitKey(0);
////            cv::destroyWindow("test");
//            //std::cout << *posSet.at(1).img.at(1) << std::endl;
//                std::cout << patchPerClass.size() << std::endl;

            std::cout << "can't extruct enough patch" << std::endl;
            //exit(-1);
        }
    }
    return;
}

void extractNegPatches(std::vector<CNegDataset> &negSet,
                       std::vector<CNegPatch> &negPatch,
                       CConfig conf){
    cv::Rect tempRect;

    std::vector<CNegPatch> tNegPatch(0);//, posPatch(0);
    nCk nck;
    int negPatchNum;

    negPatch.clear();

    tempRect.width  = conf.p_width;
    tempRect.height = conf.p_height;

    // extract negative patch
    std::cout << negSet.size() << std::endl;
    std::cout << negSet.at(0).img.at(0)->cols << std::endl;
    std::cout << negSet.at(0).img.at(0)->rows << std::endl;
    for(int i = 0; i < negSet.size(); ++i){
        for(int j = 0; j < negSet.at(i).img.at(0)->cols - conf.p_width; j += conf.stride){
            for(int k = 0; k < negSet.at(i).img.at(0)->rows - conf.p_height; k += conf.stride){

                tempRect.x = j;
                tempRect.y = k;
                int pix = 0;

                // detect negative patch
//                if(conf.learningMode != 2 && negSet.at(i).img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0){
//                    cv::Mat roi;
//                    roi = (*negSet.at(i).img.at(1))(cv::Rect(j, k, conf.p_width, conf.p_height));
//                    pix = calcSumOfDepth(roi,conf);
//                }

                //tPatch.setPatch(temp, negImage.at(i));
                //if(conf.learningMode == 2 || pix > 0){
                    CNegPatch negTemp(&negSet.at(i),tempRect);
                    tNegPatch.push_back(negTemp);
                //}
            }//x
        }//y
    } // every image

    // choose negative patch randamly
    negPatchNum = conf.patchRatio * conf.pnRatio;
    //std::cout << "pos patch num : " << posPatch.size() << " neg patch num : " << negPatchNum << std::endl;

    if(negPatchNum < tNegPatch.size()){
        std::set<int> chosenPatch = nck.generate(tNegPatch.size(), negPatchNum);//totalPatchNum * conf.patchRatio);
        std::set<int>::iterator ite = chosenPatch.begin();

        while(ite != chosenPatch.end()){
            negPatch.push_back(tNegPatch.at(*ite));
            ite++;
        }
    }else{
        std::cout << "only " << tNegPatch.size() << " pathes extracted from negative images" << std::endl;
        std::cout << "can't extract enogh negative patch please set pnratio more low!" << std::endl;
        exit(-1);
    }

//    patches.push_back(posPatch);
//    patches.push_back(negPatch);
}

void extractTestPatches(CTestDataset &testSet,std::vector<CTestPatch> &testPatch, CConfig conf){

    cv::Rect tempRect;
    //CPatch tPatch;

    tempRect.width = conf.p_width;
    tempRect.height = conf.p_height;

    testPatch.clear();
    //std::cout << "extraction patches!" << std::endl;
    for(int j = 0; j < testSet.img.at(0)->cols - conf.p_width; j += conf.stride){
        for(int k = 0; k < testSet.img.at(0)->rows - conf.p_height; k += conf.stride){
            tempRect.x = j;
            tempRect.y = k;

            int depthSum;

//            if(conf.learningMode != 2 && testSet.img.at(1)->at<ushort>(k + (conf.p_height / 2) + 1, j + (conf.p_width / 2) + 1 ) != 0){
//                cv::Mat roi;
//                roi = (*testSet.img.at(1))(cv::Rect(j, k, conf.p_width, conf.p_height));
//                depthSum = calcSumOfDepth(roi,conf);
//            }
//                for(int p = 0; p < testSet.feature.at(32)->rows; ++p)
//                    for(int q = 0; q < testSet.feature.at(32)->cols; ++q)
//                        depthSum += testSet.feature.at(32)->at<ushort>(p, q);

            //std::cout << dataSet.className << std::endl;

            //int classNum = classDatabase.search(dataSet.className.at(0));

            //std::cout << "class num is " << classNum << std::endl;
            //classDatabase.show();
            //if(classNum == -1){
                //std::cout << "This tree not contain this class data" << std::endl;
                //exit(-1);
            //}

            //if(conf.learningMode == 2){// || depthSum > 0){
                CTestPatch testTemp(&testSet,tempRect);
                //tesetPatch.setPatch(temp, image, dataSet, classNum);

                //tPatch.setPosition(j,k);
                testPatch.push_back(testTemp);
            //}
        }
    }
}

void pBar(int p,int maxNum, int width){
    int i;
    std::cout << "[";// << std::flush;
    for(i = 0;i < (int)((double)(p + 1)/(double)maxNum*(double)width);++i)
        std::cout << "*";

    for(int j = 0;j < width - i;++j)
        std::cout << " ";

    std::cout << "]" << (int)((double)(p + 1)/(double)maxNum*100) << "%"  << "\r" << std::flush;
}

void CClassDatabase::add(std::string str, cv::Size size, uchar depth){
    for(int i = 0; i < vNode.size(); ++i){
        if(str == vNode.at(i).name){
            vNode.at(i).instances++;
            return;
        }
    }
    vNode.push_back(databaseNode(str,size,depth));
    return;
}

void CClassDatabase::write(const char* str){

    std::ofstream out(str);
    if(!out.is_open()){
        std::cout << "can't open " << str << std::endl;
        return;
    }

    for(int i = 0; i < vNode.size(); ++i){
        out << i << " " << vNode.at(i).name
            << " " << vNode.at(i).classSize.width << " " << vNode.at(i).classSize.height
            << " " << vNode.at(i).classDepth
            << std::endl;
    }
    out.close();

    std::cout << "out ha shimemashita" << std::endl;
}

void CClassDatabase::read(const char* str){
    std::string tempStr;
    std::stringstream tempStream;
    int tempClassNum;
    std::string tempClassName;
    cv::Size tempSize;
    uchar tempDepth;


    std::ifstream in(str);
    if(!in.is_open()){
        std::cout << "can't open " << str << std::endl;
        return;
    }

    //std::cout << str << std::endl;

    //vNode.clear();

    do{
        in >> tempClassNum;
        in >> tempClassName;
        in >> tempSize.width;
        in >> tempSize.height;
        in >> tempDepth;
        in.ignore();
        if(!in.eof())
            this->add(tempClassName,tempSize,tempDepth);

    }while(!in.eof());

    in.close();
}

void CClassDatabase::show() const{
    if(vNode.size() == 0){
        std::cout << "No class registerd" << std::endl;
        return;
    }

    for(int i = 0; i < vNode.size(); ++i){
        std::cout << "class:" << i << " name:" << vNode.at(i).name << " has " << vNode.at(i).instances << " instances" << std::endl;
    }
}

int CClassDatabase::search(std::string str) const{
    for(int i = 0; i < vNode.size(); i++){
        //std::cout << i << " " << str << " " << vNode.at(i).name << std::endl;
        if(str == vNode.at(i).name)return i;
    }
    return -1;
}
