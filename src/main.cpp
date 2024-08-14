#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include "Util.h"

using namespace cv;

extern const int num_images = 2;

int main(int argc, char** argv) {
    Util util;

    // std::string im_names[num_images] = {"../data/incline_L.png" };
    std::string im_names[num_images] = {"../data/incline_L.png", "../data/incline_R.png"};


    std::vector<Mat> images;
    images.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        Mat im;
        if (!util.readImage(im_names[i], im)) {
            return -1;
        }
        convertImg2Float(im);
        images.push_back(im);
    }

    // read in test pattern points to compute BRIEF
    Point* compareA = NULL;
    Point* compareB = NULL;
    std::string test_pattern_filename = "../data/testPattern.txt";
    util.readTestPattern(compareA, compareB, test_pattern_filename);

    // compute BRIEF for keypoints
    std::vector<BriefResult> brief_results;
    brief_results.reserve(num_images);
    for (int i = 0; i < num_images; i++) {
        BriefResult brief_result = util.BriefLite(im_names[i], compareA, compareB);
        brief_results.push_back(brief_result);
    }

    for (int i = 1; i < num_images; i++) {

        // compute keypoint matching between image(i) and image(i-1)
        util.matching(im_names[i-1], im_names[i],
                                       brief_results[i-1], brief_results[i]);
    }

    util.printTiming();

    return 0;
}
