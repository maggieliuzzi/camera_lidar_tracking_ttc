
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "sensorFusion.hpp"

using namespace std;


int main(int argc, const char *argv[])
{
    // Data location
    string dataPath = "../";

    // Camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // Object Detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // LiDAR
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // Calibration data for camera and lidar provided by respective sensor providers
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for LiDAR and camera
    int dataBufferSize = 2;                // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; 		   // list of data frames which are held in memory at the same time
    bool bVis = true;            	 	   // visualise results

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth) {
        
        /* LOAD IMAGE INTO BUFFER */

        // assembling filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
      
        // loading image from file onto frame and adding frame to buffer
        cv::Mat img = cv::imread(imgFullFilename);
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        cout << "Image loaded into buffer" << endl;

      
        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold, yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        cout << "Objects detected and classified" << endl;


        /* CROP LIDAR POINTS */

        // loading 3D LiDAR points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // focusing on ego lane
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1;  // removing LiDAR points based on distance properties
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;
        cout << "LiDAR points cropped" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associating LiDAR points with camera-based ROI
        float shrinkFactor = 0.10;  // shrinking each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // visualising 3D objects
        //bVis = false;
        if (bVis) {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        //bVis = false;
        cout << "LiDAR point cloud clustered" << endl;
        
        // continue;  // skips directly to the next image without processing what comes beneath

      
        /* DETECT IMAGE KEYPOINTS */

        // converting current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extracting 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;  // creating empty feature list for current image
        
      	// selecting keypoint detector
        string detectorType = "FAST";  // "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"


        if (detectorType.compare("SHITOMASI") == 0) {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0) {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST")  == 0 ||
                 detectorType.compare("BRISK") == 0 ||
                 detectorType.compare("ORB")   == 0 ||
                 detectorType.compare("AKAZE") == 0 ||
                 detectorType.compare("SIFT")  == 0) {  // modern detector types
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        else
        	throw invalid_argument(detectorType + ": invalid detector type");

        // limiting number of keypoints (only for debugging and learning purposes)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "Note: only using a limited number of keypoints" << endl;
        }

        (dataBuffer.end() - 1)->keypoints = keypoints;  // adding keypoints and descriptor for current frame to end of data buffer
        cout << "Image keypoints detected" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        
        // selecting keypoint descriptor
        string descriptorType = "FREAK";  // "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"  // AKAZE requires AKAZE detector  // ORB is incompatible with SIFT detector

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        
        (dataBuffer.end() - 1)->descriptors = descriptors;  // adding descriptors for current frame to end of data buffer
        cout << "Keypoint descriptors extracted" << endl;


      	/* MATCH KEYPOINT DESCRIPTORS */
      
        if (dataBuffer.size() > 1) {  // at least two images need to be process for object tracking
            
            vector<cv::DMatch> matches;

            // selecting keypoint matcher
            string matcherType = "MAT_BF";  // Brute force ("MAT_BF") or Fast Library for Approximate Nearest Neighbors ("MAT_FLANN")
            
			// selecting appropriate descriptor type for matching
            /* some BINARY descriptors: BRISK, BRIEF, ORB, FREAK, and (A)KAZE. */
            /* some HOG descriptors: SIFT (and SURF and GLOH, all patented). */
            string descriptorCategory {};  // Binary ("DES_BINARY") or Histogram of Gradients ("DES_HOG")
            if (0 == descriptorType.compare("SIFT")) {
                descriptorCategory = "DES_HOG";
            }
            else {
                descriptorCategory = "DES_BINARY";
            }

          	// selecting appropriate selector type for matching
            string selectorType = "SEL_KNN";  // Nearest Neighbour ("SEL_NN") or K-Nearest Neighbours ("SEL_KNN")

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);

            (dataBuffer.end() - 1)->kptMatches = matches;  // adding matches to current dataframe
            cout << "Keypoint descriptors matched" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            // matching list of 3D objects (vector<BoundingBox>) between current and previous frame
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1));  // associating bounding boxes between current and previous frame using keypoint matches

            (dataBuffer.end()-1)->bbMatches = bbBestMatches;  // adding matches in current data frame
            cout << "3D-object bounding boxes tracked" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // looping over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
                // finding bounding boxes associated with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
                    if (it1->second == it2->boxID) {  // checking whether current match partner corresponds to this BB
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
                    if (it1->first == it2->boxID) {  // checking whether current match partner corresponds to this BB
                        prevBB = &(*it2);
                    }
                }

                // computing TTC for current match
                if (currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0) // only computing TTC if we have LiDAR points
                {
                    // computing time-to-collision (TTC) based on LiDAR data alone
                    double ttcLidar {0.0}; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);

                    // assigning enclosed keypoint matches to bounding box
                    // computing time-to-collision based on camera data alone
                    double ttcCamera {0.0};
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);

                    //bVis = true;
                    if (bVis) {  // visualisation
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarTopview(currBB->lidarPoints, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC (LiDAR): %fs, TTC (Camera): %fs", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "TTC";
                        cv::namedWindow(windowName, 1);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    //bVis = false;

                }  // eof TTC computation
            }  // eof loop over all BB matches            
        }  // eof check if two frames to match
    }  // eof loop over all images

    return 0;
}
