#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

extern "C" int* imageToMat(char* filename,int* dims){
    Mat image;
    image=imread(filename,CV_LOAD_IMAGE_COLOR);

    Mat gray_image;
    cvtColor(image,gray_image,CV_BGR2GRAY);

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        int* temp;
        return temp;
    }
    int width=gray_image.size().width;
    int height=gray_image.size().height;
    dims[0]=height;
    dims[1]=width;

    cout << "Height: " << dims[0] << endl;
    cout << "Width : " << dims[1] << endl;
    //namedWindow( "Original Image", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Original Image", gray_image );                   // Show our image inside it.
 
 
    //allocate 2d array
    int *matrix;
    matrix=(int*) malloc(height*width*sizeof(*matrix));
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            int intensity=gray_image.at<uchar>(i,j);
            if(intensity>254){
                intensity=254;
            }
            if(intensity<0){
                intensity=0;
            }
            matrix[i*width+j]=intensity;
        }
    }

    return matrix;
}


extern "C" void matToImageColor(char* filename, int* mat, int* dims){
    int height=dims[0];
    int width=dims[1];
    Mat image(height, width, CV_8UC3, Scalar(0,0,0));

    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            int r=0;
            int b=0;
            int g=0;
            int c=mat[i*width+j]%16;

            if(mat[i*width+j]<255 && mat[i*width+j]>0){
                switch(c){
                    case(0):
                        r=66;
                        g=30;
                        b=15;
                        break;
                   case(1):
                        r=25;
                        g=7;
                        b=26;
                        break;
                   case(2):
                        r=9;
                        g=1;
                        b=47;
                        break;
                   case(3):
                        r=4;
                        g=4;
                        b=73;
                        break;
                   case(4):
                        r=0;
                        g=7;
                        b=100;
                        break;
                   case(5):
                        r=12;
                        g=44;
                        b=138;
                        break;
                   case(6):
                        r=24;
                        g=82;
                        b=177;
                        break;
                   case(7):
                        r=57;
                        g=125;
                        b=209;
                        break;
                   case(8):
                        r=134;
                        g=181;
                        b=229;
                        break;
                   case(9):
                        r=211;
                        g=236;
                        r=248;
                        break;
                   case(10):
                        r=241;
                        g=233;
                        b=191;
                        break;
                   case(11):
                        r=248;
                        g=201;
                        b=95;
                        break;
                   case(12):
                        r=255;
                        g=170;
                        b=0;
                        break;
                   case(13):
                        r=204;
                        g=128;
                        b=0;
                        break;
                   case(14):
                        r=153;
                        g=87;
                        b=0;
                        break;
                   case(15):
                        r=106;
                        g=52;
                        b=3;
                        break;
                   default:
                        break;
                }
            }



            image.at<Vec3b>(i,j)[0] = b;
            image.at<Vec3b>(i,j)[1] = g;
            image.at<Vec3b>(i,j)[2] = r;
        }
    } 
    //namedWindow( "Processed Image", WINDOW_AUTOSIZE );// Create a window for display.

    //imshow( "Processed Image", image );                   // Show our image inside it.
    imwrite(filename,image);
    //waitKey(0);                                          // Wait for a keystroke in the window
    return;

}
extern "C" void matToImage(char* filename, int* mat, int* dims){
     int height=dims[0];
    int width=dims[1];
    Mat image(height, width, CV_8UC1, Scalar(0,0,0));

    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            image.at<uchar>(i,j) = (int)mat[i*width+j];
        }
    } 


    imwrite(filename,image);
}
