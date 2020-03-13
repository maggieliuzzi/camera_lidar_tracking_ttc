
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "sensorFusion.hpp"
#include "dataStructures.h"

using namespace std;


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    /* Creating groups of LiDAR points whose projection onto the camera image falls within the same bounding box */
    
    // looping over all LiDAR points and associating them to a 2D bounding box
   
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assembling vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // projecting LiDAR point into camera image
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);  // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes;  // pointers to all bounding boxes which enclose the current LiDAR point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            
            // shrinking current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // checking whether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        }  // eof loop over all bounding boxes

        // checking if point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) { 
            // adding/associating LiDAR point to/with bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
    }  // eof loop over all LiDAR points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // creating top-view image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1) {
        
        // creating randomised colour for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plotting LiDAR points into top-view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            
            // world coordinates
            float xw = (*it2).x;  // world position in m with x facing forward from sensor
            float yw = (*it2).y;  // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // finding enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // drawing individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // drawing enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0,0,0), 2);

        // augmenting object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plotting distance markers
    float lineSpacing = 2.0;  // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // displaying image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if (bWait)
        cv::waitKey(0);  // waiting for key to be pressed
}


void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    /* Associates a given bounding box with the keypoints it contains */
    
    // looping over all matches in the current frame
    for (cv::DMatch match : kptMatches) {  //// or: for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            boundingBox.kptMatches.push_back(match);
        }
    }
}


void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    /* Computes time-to-collision (TTC) based on keypoint correspondences in successive images */
    
    // computing distance ratios on every pair of keypoints, O(n^2) on the number of matches contained within the ROI
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);  		// kptsCurr is indexed by trainIdx
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);  		// kptsPrev is indexed by queryIdx

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);  	// kptsCurr is indexed by trainIdx
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);  	// kptsPrev is indexed by queryIdx

            // using cv::norm to calculate the current and previous Euclidean distances between each keypoint in the pair
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            double minDist = 100.0;  // thresholding the calculated distRatios by requiring a min current distance between keypoints 
            // avoiding division by zero and applying the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    if (distRatios.size() == 0)  // only continuing if the vector of distRatios is not empty
    {
        TTC = std::numeric_limits<double>::quiet_NaN();  //// or: TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());  // as with computeTTCLidar, using the median as a reasonable method for excluding outliers
    double medianDistRatio = distRatios[distRatios.size() / 2];

    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);  // estimating TTC based on 2D camera features
}


void sortLidarPoints(std::vector<LidarPoint> &lidarPoints)
{
    /* Helper function: sorts LiDAR points based on their X (longitudinal) coordinate */

    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint a, LidarPoint b) {  // std::sort with a lambda mutates lidarPoints, a vector of LidarPoint
        return a.x < b.x;  // Sort ascending on the x coordinate only
    });
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    /* Computes time-to-collision (TTC) based on relevant LiDAR points */
    
    
    sortLidarPoints(lidarPointsPrev);
    sortLidarPoints(lidarPointsCurr);
    // taking the median x-distance on each frame to increase robustness of estimation. If performance is suffering, try taking the median of a random subset of the points
    double medXPrev = lidarPointsPrev[lidarPointsPrev.size() / 2].x;
    double medXCurr = lidarPointsCurr[lidarPointsCurr.size() / 2].x;

    /* TTC = d1 * delta_t / (d0 - d1)  // note: using the constant-velocity model  // TODO: try constant-acceleration model
	  		d0: previous frame's closing distance (front-to-rear bumper)
	        d1: current frame's closing distance (front-to-rear bumper)
            delta_t: time elapsed between images, i.e. 1 / frameRate */
    // Note: this function does not take into account the distance from the LiDAR origin to the front bumper of our vehicle, nor does it account for the curvature or protrusions from the rear bumper of the preceding vehicle
    TTC = medXCurr * (1.0 / frameRate) / (medXPrev - medXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Note: after calling a cv::DescriptorMatcher::match function, each DMatch contains two keypoint indices, queryIdx and trainIdx, based on the order of image arguments to match  // https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba
    // prevFrame.keypoints is indexed by queryIdx; currFrame.keypoints is indexed by trainIdx

    std::multimap<int, int> mmap {};
    int maxPrevBoxID = 0;

    for (auto match : matches) {
        cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];
        
        int prevBoxID = -1;
        int currBoxID = -1;

        // for each bounding box in previous frame
        for (auto bbox : prevFrame.boundingBoxes) {
            if (bbox.roi.contains(prevKp.pt)) prevBoxID = bbox.boxID;
        }

        // for each bounding box in current frame
        for (auto bbox : currFrame.boundingBoxes) {
            if (bbox.roi.contains(currKp.pt)) currBoxID = bbox.boxID;
        }
        
        // adding the containing boxID for each match to a multimap
        mmap.insert({currBoxID, prevBoxID});

        maxPrevBoxID = std::max(maxPrevBoxID, prevBoxID);
    }

    // setting up a list of boxID int values to iterate over in the current frame
    vector<int> currFrameBoxIDs {};
    for (auto box : currFrame.boundingBoxes) currFrameBoxIDs.push_back(box.boxID);

    // looping through each boxID in the current frame and getting the mode (most frequent value) of associated boxID for the previous frame
    for (int k : currFrameBoxIDs) {
        // counting the greatest number of matches in the multimap, where each element is {key=currBoxID, val=prevBoxID}
        auto rangePrevBoxIDs = mmap.equal_range(k);  // std::multimap::equal_range(k) returns the range of all elements matching key = k

        // creating a vector of counts (per current bbox) of prevBoxIDs
        std::vector<int> counts(maxPrevBoxID + 1, 0);

        // accumulator loop
        for (auto it = rangePrevBoxIDs.first; it != rangePrevBoxIDs.second; ++it) {
            if (-1 != (*it).second) counts[(*it).second] += 1;
        }

        // getting the index of the maximum count (the mode) of the previous frame's boxID
        int modeIndex = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));

        // setting the best matching bounding box map with key: previous frame's most likely matching boxID, and value: current frame's boxID, k
        bbBestMatches.insert({modeIndex, k});
    }
}
