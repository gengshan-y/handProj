#include "parameters.hpp"
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
  string configFileName = string(argv[1]);
  cout << "reading " << configFileName  << endl;

  Parameters hp(configFileName);

  string tmp = hp.readStringParameter("Tracker.Name");  

  cout << tmp << endl;

  return 0;
}
