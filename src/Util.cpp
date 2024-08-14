#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include "Util.h"
#include "filter.h"
#include "keyPointDetector.h"

using namespace cv;

extern const int num_images;

Util::Util() {
    gaussian_pyramid_elapsed = 0.0;
    dog_pyramid_elapsed = 0.0;
    keypoint_detection_elapsed = 0.0;
    compute_brief_elapsed = 0.0;
    find_match_elapsed = 0.0;
}

Util::~Util() {}

void Util::printTiming() const {
    printf("Compute Gaussian Pyramid: %.2f\n", gaussian_pyramid_elapsed);
    printf("Compute DoG Pyramid: %.2f\n", dog_pyramid_elapsed);
    printf("Detect Keypoints: %.2f\n", keypoint_detection_elapsed);
    printf("Compute BRIEF Descriptor: %.2f\n", compute_brief_elapsed);
    printf("Match keypoint descriptors: %.12f\n", find_match_elapsed);
}

inline double Util::get_time_elapsed(clock_t& start)
{
    return (clock() - start) / (double)CLOCKS_PER_SEC;
}

bool Util::readImage(std::string im_name, Mat& im) {
    im = imread(im_name, IMREAD_COLOR);
    if(!im.data ) {
        std::cout <<  "Could not open or find the image " +
                        im_name << std::endl ;
        return false;
    }
    return true;
}

void Util::cleanPointerArray(float** arr, int num_levels) {
    for (int i = 0; i < num_levels; i++) {
        delete[] arr[i];
    }
    delete[] arr;
}

void Util::matching(std::string im1_name, std::string im2_name,
                      BriefResult brief_result1, BriefResult brief_result2) {

    std::cout << "Matching key points: " + im1_name + ", " + im2_name << std::endl;
    find_match_start = clock();
    MatchResult match = cudaBriefMatch(brief_result1.descriptors, brief_result2.descriptors);

    find_match_elapsed += get_time_elapsed(find_match_start);
     // Count the number of matches
    int num_matches = match.indices1.size();
    std::cout << "Number of matching keypoints: " << num_matches << std::endl;

    plotMatches(im1_name, im2_name, brief_result1.keypoints, 
                brief_result2.keypoints, match);
}


BriefResult Util::BriefLite(std::string im_name, Point* compareA,
                            Point* compareB) {

    std::cout << "Computing BRIEF for image " + im_name << std::endl;

    Mat im_color = imread(im_name, IMREAD_COLOR);

    // Load as grayscale and convert to float
    Mat im_gray = imread(im_name, IMREAD_GRAYSCALE);
    Mat im;
    im_gray.convertTo(im, CV_32F);
    int h = im.rows;
    int w = im.cols;
    float *im1_ptr = (float*) im.ptr<float>();
    normalize_img(im1_ptr, h, w);

    // parameters for generating Gaussian Pyramid
    float sigma0 = 1.0;
    float k = sqrt(2);
    int num_levels = 7;
    int levels[7] = {-1, 0, 1, 2, 3, 4, 5};

    CudaFilterer cudaFilterer;
    cudaFilterer.setup(im1_ptr, h, w);

    // float** gaussian_pyramid_cuda = 
    //     cudaFilterer.createGaussianPyramid(sigma0, k, levels, num_levels);

    gaussian_pyramid_start = clock();
    float** gaussian_pyramid = 
        cudaFilterer.createGaussianPyramid(sigma0, k, levels, num_levels);
    
    //float** gaussian_pyramid = createGaussianPyramid(im1_ptr, h, w, sigma0, k,
                                                     // levels, num_levels);                    
    gaussian_pyramid_elapsed += get_time_elapsed(gaussian_pyramid_start);


    dog_pyramid_start = clock();
    float** dog_pyramid = createDoGPyramid(gaussian_pyramid, h, w, num_levels);

    dog_pyramid_elapsed += get_time_elapsed(dog_pyramid_start);

    keypoint_detection_start = clock();
    // Detect key points
    float th_contrast = 0.03;
    float th_r = 12;
    std::vector<Point> keypoints = getLocalExtrema(dog_pyramid, num_levels - 1,
                                                   h, w, th_contrast, th_r);
    printf("Detected %lu key points\n", keypoints.size());
    keypoint_detection_elapsed += get_time_elapsed(keypoint_detection_start);
    
    outputGaussianImages(gaussian_pyramid, h, w, num_levels);
    outputDoGImages(dog_pyramid, h, w, num_levels);
    outputImageWithKeypoints(im_name, im_color, keypoints);

    compute_brief_start = clock();
    BriefResult brief_result = computeBrief(gaussian_pyramid[0], h, w, 
                                            keypoints, compareA, compareB);
    compute_brief_elapsed += get_time_elapsed(compute_brief_start);

    // clean up
    cleanPointerArray(gaussian_pyramid, num_levels);
    cleanPointerArray(dog_pyramid, num_levels - 1);

    return brief_result;
}

void Util::plotMatches(std::string im1_name, std::string im2_name,
                       std::vector<Point>& pts1, std::vector<Point>& pts2,
                       MatchResult& match) {

    Mat im1 = imread(im1_name, IMREAD_COLOR);
    Mat im2 = imread(im2_name, IMREAD_COLOR);
    int h1 = im1.rows;
    int w1 = im1.cols;
    int h2 = im2.rows;
    int w2 = im2.cols;
    int width = w1 + w2;
    int height = max(h1, h2);
    Mat grid(height, width, CV_8UC3, Scalar(0,0,0));
    im1.copyTo(grid(Rect(0,0,w1,h1)));
    im2.copyTo(grid(Rect(w1,0,w2,h2)));

    for (int i = 0; i < match.indices1.size(); i++) {
        Point p1 = pts1[match.indices1[i]];
        Point p2 = pts2[match.indices2[i]];
        line(grid, p1, Point(p2.x+w1, p2.y), Scalar(0, 0, 255));
    }

    std::cout << "Output Match Image" << std::endl;
    imwrite("../output/match.jpg", grid);
}


int Util::readTestPattern(Point*& compareA, Point*& compareB,
                          std::string test_pattern_filename) {

    std::ifstream infile(test_pattern_filename);
    int num_test_pairs;
    infile >> num_test_pairs;
    compareA = new Point[num_test_pairs];
    compareB = new Point[num_test_pairs];
    int x1, y1, x2, y2;
    for (int i = 0; i < num_test_pairs; i++) {
        infile >> x1 >> y1 >> x2 >> y2;
        compareA[i] = Point(x1, y1);
        compareB[i] = Point(x2, y2);
    }
    return num_test_pairs;
}

void Util::printImage(float* img, int h, int w) const {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            std::cout << img[i * w + j] << " ";
        }
        std::cout << std::endl;
    }
}

inline void Util::displayImg(Mat& im) const {
    // Create a window for display.
    namedWindow( "Display window", 0);
    resizeWindow("Display window", im.cols, im.rows);
    imshow( "Display window", im);           // Show our image inside it.
    waitKey(0);
}
