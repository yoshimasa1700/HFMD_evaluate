#include "CRForest.h"
//#include "CRTree.h"

#include "util.h"

using namespace std;

void detect(CConfig conf){
  boost::mt19937    gen(static_cast<unsigned long>(time(NULL)) );
  
  

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
  forest.learning();

  return 0; 
}
