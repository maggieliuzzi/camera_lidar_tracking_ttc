#include <numeric>
#include "matching2D.hpp"

using namespace std;


void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType)
{
    /* Finds best matches for keypoints in two camera images based on several matching methods */
    
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

	// creating matcher
    
    if (matcherType.compare("MAT_BF") == 0) {  // Brute Force
        int normType;
        if (descriptorCategory.compare("DES_HOG") == 0) {  // with SIFT
            normType = cv::NORM_L2;
        }
        else if (descriptorCategory.compare("DES_BINARY") == 0) {  // with binary descriptors except for SIFT
            normType = cv::NORM_HAMMING;
        }
        else
            throw invalid_argument(descriptorCategory + ": invalid descriptor category");
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0) {  // FLANN matching
        if (descriptorCategory.compare("DES_HOG") == 0) {  // with SIFT
            matcher = cv::FlannBasedMatcher::create();
        }
        else if (descriptorCategory.compare("DES_BINARY") == 0) {  // with binary descriptors except for SIFT
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }
        else
            throw invalid_argument(descriptorCategory + ": invalid descriptor category");
    }
    else
        throw invalid_argument(matcherType + ": invalid matcher type");

    // performing matching
  
    if (selectorType.compare("SEL_NN") == 0) {  // nearest neighbor matching (best match)
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0) {  // K nearest neighbours, k=2 (best two)
        vector<vector<cv::DMatch>> knn_matches;
        int k = 2;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it : knn_matches) {  //// or:  for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
            if ((it[0].distance < minDescDistRatio * it[1].distance) && 2 == it.size()) {  // the returned knn_matches vector contains some nested vectors with size < 2
                matches.push_back(it[0]);
            }
        }
    }
    else
        throw invalid_argument(selectorType + ": invalid selector type");
}


void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    /* Uses state-of-art descriptors to uniquely identify keypoints */
    
    // selecting appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;          // FAST/AGAST detection threshold score
        int octaves = 3;             // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f;   // applying this scale to the pattern used for sampling the neighbourhood of a keypoint

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
        throw invalid_argument(descriptorType + ": invalid descriptor type");

    // double t = (double)cv::getTickCount();
  
    extractor->compute(img, keypoints, descriptors);
  
    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}


void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    /* Detecting keypoints in image using the traditional Shi-Thomasi detector */
    
    // computing detector parameters based on image size
    int blockSize = 4;        // size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0;  // max acceptable overlap between two features, %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance);  // max num of keypoints
    double qualityLevel = 0.01;  // min accepted quality for image corners
    double k = 0.04;

    // applying corner detection
    // double t = (double)cv::getTickCount();
  
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // adding corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
  
    // t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis) {  // visualisation
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    /* Detects keypoints in an image using the Harris keypoint detection algorithm */  
  
    int blockSize = 2;     // a blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for the Sobel operator
    assert(1 == apertureSize % 2);  // aperture size must be odd
    int minResponse = 100; // min value for a corner in the scaled (0...255) response matrix
    double k = 0.04;       // Harris parameter
    double maxOverlap = 0.0;  // maximum overlap between two features, %  // non-maximum suppression (NMS) setting

    // detecting Harris corners and normalising output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    if (bVis) {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }

    // Aapplying non-maximum suppression (NMS)
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
          
            int response = (int)dst_norm.at<float>(j, i);
          
            // applying the minimum threshold for Harris cornerness response
            if (response < minResponse) continue;
            // else, creating a tentative new keypoint
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // perform non-maximum suppression (NMS) in local neighbourhood around the new keypoint
            bool bOverlap = false;
            // looping over all existing keypoints
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                // testing if overlap exceeds the maximum percentage allowable
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;
                    // if overlapping, test if new response is the local maximum
                    if (newKeyPoint.response > (*it).response) {
                        *it = newKeyPoint;  // replacing the old keypoint
                        break;
                    }
                }
            }

            if (!bOverlap) {  // if above response threshold and not overlapping any other keypoint
                keypoints.push_back(newKeyPoint);  // adding to keypoints list
            }
        }
    }

    if (bVis)
    {
        string windowName = "Harris corner detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    /* Detecting keypoints in an image using more recent methods available in OpenCV */
  
    if (detectorType.compare("FAST") == 0) {
        // int threshold_FAST = 150;
        auto fast = cv::FastFeatureDetector::create();
        fast->detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0) {
        // int threshold_BRISK = 200;
        auto brisk = cv::BRISK::create();
        brisk->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) {
        auto orb = cv::ORB::create();
        orb->detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0) {
        auto akaze = cv::AKAZE::create();
        akaze->detect(img, keypoints);
    }
    else if (detectorType.compare("SIFT") == 0) {
        auto sift = cv::xfeatures2d::SIFT::create();
        sift->detect(img, keypoints);
    }
    else
        throw invalid_argument(detectorType + ": invalid detector type");

    if (bVis) {
        string windowName = detectorType + " keypoint detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
