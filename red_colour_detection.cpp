#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap(1); 

    if (!cap.isOpened()) 
    {
        cout << "Cannot open the webcam" << endl;
        return -1;
    }

    namedWindow("Color Detect", WINDOW_AUTOSIZE); 

    // HSV values for red 
    int lowH1 = 0, highH1 = 10;  // Lower red range
    int lowH2 = 160, highH2 = 179; // Higher red range
    int lowS = 100, highS = 255;   // Saturation
    int lowV = 100, highV = 255;   // Value

    while (true)
    {
        Mat imgOriginal;

        // Read new frame
        bool bSuccess = cap.read(imgOriginal);
        if (!bSuccess)
        {
            cout << "Cannot read a frame" << endl;
            break;
        }

        Mat imgHSV, imgThresholded1, imgThresholded2, imgThresholded;

        // BGR to HSV
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

        // Threshold to detect red
        inRange(imgHSV, Scalar(lowH1, lowS, lowV), Scalar(highH1, highS, highV), imgThresholded1);
        inRange(imgHSV, Scalar(lowH2, lowS, lowV), Scalar(highH2, highS, highV), imgThresholded2);

        // Combine masks
        imgThresholded = imgThresholded1 | imgThresholded2;

        // Noise reduction
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        // Find contours to detect 
        vector<vector<Point>> contours;
        findContours(imgThresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Draw rectangles 
        for (size_t i = 0; i < contours.size(); i++)
        {
            Rect boundingBox = boundingRect(contours[i]);

            // consider large objects
            if (boundingBox.area() > 500)
            {
                rectangle(imgOriginal, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2); // Draw green rectangle
            }
        }

        // Display 
        imshow("Thresholded Image", imgThresholded);
        imshow("Original with Tracking", imgOriginal);

        // break loop (esc key)
        if (waitKey(30) == 27)
        {
            cout << "Exiting..." << endl;
            break;
        }
    }

    return 0;
}
