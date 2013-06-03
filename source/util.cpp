#include "util.h"

boost::lagged_fibonacci1279 nCk::gen = boost::lagged_fibonacci1279();

CDataset::CDataset(){
}

void CDataset::showDataset(){
    std::cout << "RGB image name:" << rgbImageName << std::endl;
    std::cout << "Depth image name:" << depthImageName << std::endl;
    std::cout << "mask image name:" << maskImageName << std::endl;

    std::cout << "bouding box: " << bBox << std::endl;

    for(int i = 0; i < centerPoint.size(); ++i){
        std::cout << "class: " << className.at(i) << std::endl;
        std::cout << "\t" << "center point: " << centerPoint.at(i) << std::endl;

        std::cout << "\t" << "angle grand truth:" << angles.at(i) << std::endl;
    }
}

CConfig::CConfig()
{
}


int CConfig::loadConfig(const char* filename)
{
    read_xml(filename, pt);

    // load tree path
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.treepath")) {
        std::cout << str.get() << std::endl;
        treepath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load number of tree
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.ntree")) {
        std::cout << integer << std::endl;
        ntrees = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load patch width
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.pwidth")) {
        std::cout << integer << std::endl;
        p_width = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load patch height
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.pheight")) {
        std::cout << integer << std::endl;
        p_height = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load test image path
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.imgpath")) {
        std::cout << str.get() << std::endl;
        impath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load image name list
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.imgnamelist")) {
        std::cout << str.get() << std::endl;
        imfiles = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load extruct feature flag
    if (boost::optional<bool> boolean
            = pt.get_optional<bool>("root.efeatures")) {
        std::cout << boolean << std::endl;
        xtrFeature = *boolean;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load image scales
    std::cout << "kokomade" << std::endl;
    scales.resize(0);
    BOOST_FOREACH (const boost::property_tree::ptree::value_type& child,
                   pt.get_child("root.scales")) {
        const float value = boost::lexical_cast<float>(child.second.data());
        scales.push_back(value);

        std::cout << value << std::endl;
    }
    for (int i;i < scales.size(); ++i)
        std::cout << i << ": " << scales.at(i) << std::endl;
    float value_temp = 1;
    scales.clear();
    scales.push_back(value_temp);

    ratios.clear();
    ratios.push_back(value_temp);
    ratios.push_back(value_temp);

    // // load image ratios
    ratios.resize(0);
    BOOST_FOREACH (const boost::property_tree::ptree::value_type& child, pt.get_child("root.ratio")) {
        const float value = boost::lexical_cast<float>(child.second.data());
        ratios.push_back(value);

        std::cout << value << std::endl;
    }
    for (int i;i < ratios.size(); ++i)
        std::cout << i << ": " << ratios.at(i) << std::endl;

    // load output path
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.outpath")) {
        std::cout << str.get() << std::endl;
        outpath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.sfactor")) {
        std::cout << integer << std::endl;
        out_scale = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }


    // load training image name list
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.traindataname")) {
        std::cout << str.get() << std::endl;
        traindatafile = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load training image folder
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.trainimgpath")) {
        std::cout << str.get() << std::endl;
        trainpath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.trainimagepertree")) {
        std::cout << integer << std::endl;
        imagePerTree = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.minsample")) {
        std::cout << integer << std::endl;
        min_sample = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.maxdepth")) {
        std::cout << integer << std::endl;
        max_depth = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.featurechannel")) {
        std::cout << integer << std::endl;
        featureChannel = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<double> variableDouble
            = pt.get_optional<double>("root.patchratio")) {
        std::cout << variableDouble << std::endl;
        patchRatio = *variableDouble;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load scale factor for output imae
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.stride")) {
        std::cout << integer << std::endl;
        stride = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load offset of tree name
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.offTree")) {
        std::cout << integer << std::endl;
        off_tree = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load test data folder
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.testpath")) {
        std::cout << str.get() << std::endl;
        testPath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load test data file name
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.testdataname")) {
        std::cout << str.get() << std::endl;
        testData = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load test data file name
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.classdatabasename")) {
        std::cout << str.get() << std::endl;
        classDatabaseName = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load offset of tree name
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.learningmode")) {
        std::cout << "learning mode is " << integer << std::endl;
        learningMode = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load training image name list
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.traindatalistname")) {
        std::cout << str.get() << std::endl;
        traindatalist = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load testing image name list
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.testdatalistname")) {
        std::cout << str.get() << std::endl;
        testdatalist = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load offset of tree name
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.testmode")) {
        std::cout << integer << std::endl;
        testMode = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load testing image name list
    if (boost::optional<double> dou
        = pt.get_optional<double>("root.detectthreshold")) {
      std::cout << dou.get() << std::endl;
      detectThreshold = *dou;
    }
    else {
      std::cout << "root.str is nothing" << std::endl;
    }

    // load offset of tree name
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.showgrandtruth")) {
        std::cout << integer << std::endl;
        showGT = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load training image folder
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.negativedatapath")) {
        std::cout << str.get() << std::endl;
        negDataPath = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load training image folder
    if (boost::optional<std::string> str
            = pt.get_optional<std::string>("root.negativedatalist")) {
        std::cout << str.get() << std::endl;
        negDataList = *str;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load offset of tree name
    if (boost::optional<int> integer
            = pt.get_optional<int>("root.negativemode")) {
        std::cout << integer << std::endl;
        negMode = *integer;
    }
    else {
        std::cout << "root.str is nothing" << std::endl;
    }

    // load testing image name list
    if (boost::optional<double> dou
        = pt.get_optional<double>("root.posnegpatchratio")) {
      std::cout << dou.get() << std::endl;
      pnRatio = *dou;
    }
    else {
      std::cout << "root.str is nothing" << std::endl;
    }



    return 0;
}

void loadTrainPosFile(CConfig conf, std::vector<CPosDataset> &posSet)
{
    std::string traindatafilepath = conf.trainpath + PATH_SEP +  conf.traindatafile;
    int n_folders;
    int n_files;
    std::vector<std::string> trainimagefolder;
    std::vector<CDataset> tempDataSet;
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
            posTemp.setRgbImagePath(nameTemp);

            trainDataList >> nameTemp;


            trainDataList >> posTemp.maskImageName;

            //read bounding box
            //trainDataList >> posTemp.bBox.x;
            //trainDataList >> posTemp.bBox.y;
            //trainDataList >> posTemp.bBox.width;
            //trainDataList >> posTemp.bBox.height;

            posTemp.centerPoint.clear();

            //read center point
            //trainDataList >> posTempPoint.x;//posTemp.centerPoint.x;
            //trainDataList >> posTempPoint.y;

            //posTemp.centerPoint.push_back(posTempPoint);

            //read class name
            std::string tempClassName;
            trainDataList >> tempClassName;
            temp.className.push_back(tempClassName);

            cv::Mat tempImage = cv::imread(temp.imageFilePath
                                           + temp.rgbImageName,3);

            cv::Size tempSize = cv::Size(tempImage.cols, tempImage.rows);

            //std::cout << temp.className.at(0) << std::endl;
            database.add(temp.className.at(0), tempSize, 0);

            //read angle grand truth
            double tempAngle;
            trainDataList >> tempAngle;
            temp.angles.push_back(tempAngle);

            //show read data *for debug
            //temp.showDataset();

            tempDataSet.push_back(temp);
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
            dataSet.push_back(tempDataSet.at(*ite + dataOffset));
            ite++;
        }
        dataOffset += database.vNode.at(j).instances;
    }


//    std::cout << "show chosen dataset" << std::endl;
//    for(int i = 0; i < dataSet.size(); ++i){
//        dataSet.at(i).showDataset();
//    }

    //in.close();
}

void loadTrainNegFile(CConfig conf, std::vector<CDataset> &dataSet)
{
    //std::string traindatafilepath
    int n_files;
    CDataset temp;
    std::string trainDataListPath = conf.negDataPath + PATH_SEP +  conf.negDataList;
    int dataSetNum;
    CClassDatabase database;
    cv::Point tempPoint;
    nCk nck;


    //read train data folder list
    std::ifstream in(trainDataListPath.c_str());
    if(!in.is_open()){
        std::cout << "train negative data floder list is not found!" << std::endl;
        exit(1);
    }

    in >> n_files;

    dataSet.clear();

    for(int i = 0; i < n_files; ++i){
        temp.className.clear();
        temp.angles.clear();
        temp.centerPoint.clear();

        temp.imageFilePath = conf.negDataPath + PATH_SEP;
        std::string negName = "neg";
        temp.className.push_back(negName);

        in >> temp.rgbImageName;
        in >> temp.depthImageName;

        dataSet.push_back(temp);
    }

    in.close();

    for(int i = 0; i < dataSet.size(); ++i)
        dataSet.at(i).showDataset();
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

    std::cout << str << std::endl;

    vNode.clear();

    do{
        in >> tempClassNum;
        in >> tempClassName;
        in >> tempSize.width;
        in >> tempSize.height;
        in >> tempDepth;
        in.ignore();
        if(!in.eof())
            vNode.push_back(databaseNode(tempClassName,tempSize,tempDepth));
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
