#include "Tracker.hpp"
#include "cvLib.hpp"

unsigned int Tracker::getFrameNum() {
  return currFrameNum;
}


void Tracker::getBBox(vector<Mat>& imgVec, vector<Rect>& rectVec, 
                      vector<int>& idVec, vector<unsigned int>& frameNumVec) {
  for (auto it = trkObjs.begin(); it != trkObjs.end(); it++) {
    imgVec.push_back(currFrame);
    rectVec.push_back(it->getBBox());
    idVec.push_back(it->getID());
    frameNumVec.push_back(currFrameNum);
  }
}

void Tracker::update(vector<Rect> found, Mat& targImg) {
    targImg.copyTo(currFrame);  // get current frame
    currFrameNum = frameCount;
    Mat oriImg;
    targImg.copyTo(oriImg);  // save original image

    /* Upgrade old tracking objects */
    for (auto it = trkObjs.begin(); it != trkObjs.end(); it++) {
        (*it).incAge();
        (*it).predKalmanFilter();
        (*it).showInfo();
    }

    /* Build measured objects */
    vector<TrackingObj> meaObjs;
    for (auto it = found.begin(); it != found.end(); it++)
        meaObjs.push_back(measureObj(targImg, *it));  // measured object
    // testStateParsing(meaObjs[0]);  // test the parsing interface

    /* Update/Add tracking objects */
    cout << "@@data association" << endl;
    vector<TrackingObj> lastTracker = trkObjs;  // save for association
    for (auto it = meaObjs.begin(); it != meaObjs.end(); it++) {
        /* get measured state */
        cout << "measured..." << endl;
        (*it).showState();
        vector<float> meaArray = (*it).getStateVec();

        vector<float> scoreArr;  // to store the comparison scores 
        for (auto itt = lastTracker.begin(); itt != lastTracker.end(); itt++) {
            /* get tracker predicted state */
            cout << "predicted..." << endl;
            (*itt).showState();
            vector<float> predArray = (*itt).getStateVec();

            float score;
            // compare states
            float stateScore = meaStateDis(meaArray, predArray);
            // float alpha = 0.001; 
            // stateScore = exp(-stateScore * alpha);  // normalize to 0-1, tune a
            cout << "iou metric:\t" << stateScore << endl;;
            // get SVM score for measurement
            float SVMScore = (*itt).testSVM( (*it).getAppearance() );
            cout << "SVM score: \t" << SVMScore << endl;
            // waitKey(0);
            score = (stateScore + SVMScore) / 2.;

            scoreArr.push_back(score);  // add score to an array
        }

        unsigned int targIdx = distance(scoreArr.begin(),
                              max_element(scoreArr.begin(), scoreArr.end()));
        // unsigned int targIdx = distance(scoreArr.begin(),
        //                       min_element(scoreArr.begin(), scoreArr.end()));
        // if the highest score is higher than a th
        // if (scoreArr.size() != 0 && scoreArr[targIdx] < 1000) {
        if (scoreArr.size() != 0 && scoreArr[targIdx] > 0.5) {
            cout << "**ID " << trkObjs[targIdx].getID() << " updated" << endl;
            /* update the according tracker */
            // update tracklet
            trkObjs[targIdx].updateTracklet( (*it).getPos() );
            drawObj.drawTracklet(targImg, trkObjs[targIdx].getID(),
                                 trkObjs[targIdx].getTracklet());
            // update SVM
            trkObjs[targIdx].updateSVM( oriImg, (*it).getAppearance() );
            trkObjs[targIdx].updateKalmanFilter( (*it).getMeaState() );
            trkObjs[targIdx].state2Attr();

            // reset age
            trkObjs[targIdx].resetAge();
            continue;
        }

        /* Else push detection results to tracker */
        // initialize tracklet
        (*it).initTracklet();
        // initialize SVM for *it
        (*it).initSVM(targImg);
        trkObjs.push_back(*it);
        currID++;  // update ID
        cout << "**ID " << trkObjs.back().getID() << " added." << endl;
        // trkObjs.back().showInfo();
    }

    /* Remove outdated objects */
    for (int it = trkObjs.size() - 1; it >= 0; it--) {
        if ( (trkObjs[it]).getAge() > 20 ) {  // set age as 20
            cout << "$$ID " << trkObjs[it].getID() << " to be deleted." << endl;

            // delete SVM for it
            (*(trkObjs.begin() + it)).rmSVM();

            // save appearance for future reference
            // (*(trkObjs.begin() + it)).svAppearance();
            destroyWindow("object " + \
               to_string((trkObjs.begin() + it)->getID()));  // destory window
            destroyWindow("pose_" + \
               to_string((trkObjs.begin() + it)->getID()));
            trkObjs.erase(trkObjs.begin() + it);
        }
    }
}

unsigned int TrackingObj::getAge() {
  return age;
}   

unsigned int TrackingObj::getID() {
  return ID;
}

Mat TrackingObj::getAppearance() {
  return appearance;
}

Mat TrackingObj::getFrame() {
  return oriFrame;
}

Rect TrackingObj::getBBox() {
  return Rect(pos.first - size.first/2, pos.second - size.second/2,
              size.first, size.second);
}

pair<float, float> TrackingObj::getPos() {
  return pos;
}

Mat TrackingObj::getState() {
  return state;
}

Mat TrackingObj::getMeaState() {
  Mat meaState(4, 1, CV_32F);
  meaState.at<float>(0, 0) = state.at<float>(0, 0);
  meaState.at<float>(1, 0) = state.at<float>(1, 0);
  meaState.at<float>(2, 0) = state.at<float>(4, 0);
  meaState.at<float>(3, 0) = state.at<float>(5, 0);
  return meaState;
}

vector<float> TrackingObj::getStateVec() {
  vector<float> arr;
  arr.assign((float*)state.datastart, (float*)state.dataend);
  return arr;
}

vector<pair<unsigned int, unsigned int>> TrackingObj::getTracklet() {
  return tracklet;
}

void TrackingObj::incAge() {
  age++;
  // cout << "ID " << ID << " age increased." << endl; 
}

void TrackingObj::resetAge() {
  age = 1;
}

void TrackingObj::showInfo() {
  cout << "||------------------------------------" << endl;
  cout << "Tracker information" << endl;
  cout << "ID\t" << ID << endl;
  cout << "age\t" << age << endl;
  cout << "pos\t(" << pos.first << "," << pos.second << ")" << endl;
  cout << "vel\t(" << vel.first << "," << vel.second << ")" << endl;
  cout << "size\t(" << size.first << "," << size.second << ")" << endl;
  imshow("object " + to_string(ID), appearance);
  // pauseFrame(0);
  showState();
  
/* SVM too long
  if (trackerSVM)
    trackerSVM->showInfo();
*/

  cout << "------------------------------------||" << endl;
}

void TrackingObj::attr2State() {
  int count = 0;

  state.at<float>(count++, 0) = pos.first;
  state.at<float>(count++, 0) = pos.second;
  state.at<float>(count++, 0) = vel.first;
  state.at<float>(count++, 0) = vel.second;
  state.at<float>(count++, 0) = size.first;
  state.at<float>(count++, 0) = size.second;

  /* Assert */
  if(count != state.rows) {
    cout << "Size check failed." << endl;
    cout << count << ", " << state.rows << endl;
    exit(-1);
  }
}

void TrackingObj::state2Attr() {
  pos = make_pair(state.at<float>(0, 0), state.at<float>(1, 0));
  vel = make_pair(state.at<float>(2, 0), state.at<float>(3, 0));
  size = make_pair(state.at<float>(4, 0), state.at<float>(5, 0));
}

bool TrackingObj::operator==(const TrackingObj& other) {

  if(pos != other.pos) {
    cout << "position not matched" << endl;
    return false;
  }

  if(vel != other.vel) {
    cout << "velocity not matched" << endl;
    return false;
  }

  if(size != other.size) {
    cout << "size not matched" << endl;
    return false;
  }

  return true;
}

void TrackingObj::showState() {
  cout << "state\t" << state << endl;
  /*
  for (int it = 0; it < state.rows; it++) {
    cout << state.at<float>(it, 0) << endl;
  }
  */  
}

void TrackingObj::initKalmanFilter() {
  // showState();
  KF.transitionMatrix = *( Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,  // fps-1 
                                                0, 1, 0, 1, 0, 0,
                                                0, 0, 1, 0, 0, 0,
                                                0, 0, 0, 1, 0, 0,
                                                0, 0, 0, 0, 1, 0,
                                                0, 0, 0, 0, 0, 1 );

  KF.measurementMatrix = *( Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
                                                 0, 1, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 1, 0,
                                                 0, 0, 0, 0, 0, 1 );

  setIdentity(KF.processNoiseCov, Scalar::all(1e-5));   // to be tuned
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));  // to be tuned

  setIdentity(KF.errorCovPost, Scalar::all(1));
  KF.statePost = state;

  /*
  cout << "transition matrix:\n" << KF.transitionMatrix << endl;
  cout << "measurement matrix:\n" << KF.measurementMatrix << endl;
  cout << "process noise:\n" << KF.processNoiseCov << endl;
  cout << "measurement noise:\n" << KF.measurementNoiseCov << endl;
  cout << "initial state covariance:\n" << KF.errorCovPost << endl;
  cout << "initial state:\n" << KF.statePost << endl;
  */
}

void TrackingObj::predKalmanFilter() {
  state = KF.predict();
  /*
  cout << "predicted state:\n" << state << endl;
  cout << "prior state:\n" << KF.statePre << endl;
  cout << "prior state covariance:\n" << KF.errorCovPre << endl;
  */
}

void TrackingObj::updateKalmanFilter(Mat measuredState) {
  /* correct using new measurement */
  state = KF.correct(measuredState);
  // showState();
  // Mat processNoise(5, 1, CV_32F);
  // randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
  // state = KF.transitionMatrix*state + processNoise;
}

vector<Mat> TrackingObj::sampleBgImg(Mat bgImg) {
  vector<Mat> negImgVec;  // a vector of negtive training images

  /* Generate random numbers */
  RNG rng;
  vector<unsigned int> rnXVec;
  vector<unsigned int> rnYVec;
  for (unsigned int it = 0; it < negNum; it++) {
    rnXVec.push_back( rng.uniform(0, bgImg.cols - winSize.width) );
    rnYVec.push_back( rng.uniform(0, bgImg.rows - winSize.height) );
  }

/*
  for (unsigned int it = 0; it < negNum; it++) {
    cout << "posx:\t" << rnXVec[it] << endl;
    cout << "posy:\t" << rnYVec[it] << endl;
  }
*/
  
  /* Form negative image vectors */
  for (unsigned int it = 0; it < negNum; it++) {
    vector<float> posPos;
    vector<float> negPos;    
    posPos.push_back(pos.first);
    posPos.push_back(pos.second);
    negPos.push_back(rnXVec[it]);
    negPos.push_back(rnYVec[it]);

    float dis = norm(posPos, negPos, NORM_L2);
    if ( dis < 50 ) {  // remove boxes near head
        continue;
    }
    Mat croppedImg;  // a tmp continer of negative images
    Rect tmpBox(rnXVec[it], rnYVec[it], winSize.width, winSize.height);
    bgImg(tmpBox).copyTo(croppedImg);
    negImgVec.push_back(croppedImg);
  }

  return negImgVec;
}

void TrackingObj::initSVM(Mat bgImg) {
  trackerSVM = new imgSVM();

  /* Prepare positive features */ 
  vector<Mat> posImgVec;
  posImgVec.push_back(appearance);
  Mat posFeat = trackerSVM->img2feat(posImgVec);
  // imshow("", appearance);
  // waitKey(0);

  /* Preapare negative features */
  vector<Mat> negImgVec = sampleBgImg(bgImg);
  // for (auto it = negImgVec.begin(); it != negImgVec.end(); it++) {
  //   imshow("", *it);
  //   waitKey(0); 
  // }
  Mat negFeat = trackerSVM->img2feat(negImgVec);

  trackerSVM->fillData(posFeat, negFeat);

  trackerSVM->SVMConfig();

  trackerSVM->SVMTrain();
}

float TrackingObj::testSVM(Mat inAppearance) {
  vector<Mat> imgVec;
  imgVec.push_back(inAppearance);
  
  Mat inFeat = trackerSVM->img2feat(imgVec);
 
  float res = trackerSVM->SVMPredict(inFeat);
  Mat cmbedImg = combImgs(inAppearance, appearance);
  imshow("", cmbedImg);
  // waitKey(0);
  
  /*
  imshow("img1", inAppearance);
  imshow("img2", appearance);
  pauseFrame(0);
  */

  return res;
}

void TrackingObj::updateSVM(Mat bgImg, Mat inAppearance) { 
  oriFrame = bgImg;// update frame
  appearance = inAppearance;

  vector<Mat> newImgPos;
  newImgPos.push_back(inAppearance);

  /* get features from new images */
  Mat newPosFeat = trackerSVM->img2feat(newImgPos);
  Mat newNegFeat = trackerSVM->img2feat( sampleBgImg(bgImg) );

  trackerSVM->fillData(newPosFeat, newNegFeat);
  trackerSVM->SVMTrain();
}

void TrackingObj::rmSVM() {
  delete trackerSVM;
}

void TrackingObj::initTracklet() {
  pair<unsigned int, unsigned int> iniPos= make_pair(pos.first,
                                                     pos.second);
  tracklet.push_back(iniPos);
}

void TrackingObj::updateTracklet(pair<float, float> inPos) {
  pair<unsigned int, unsigned int> newPos= make_pair(inPos.first,
                                                   inPos.second);
  tracklet.push_back(newPos);
}

bool TrackingObj::getDirection() {
  int accum = 0;
  for (unsigned int it = 1; it < tracklet.size(); it++) {
    accum += tracklet[it].second - tracklet[it - 1].second;
  }

  return accum > 0;
}

void TrackingObj::svAppearance() {
  imwrite(appearancePath + to_string(ID) + ".jpg", appearance);
}
