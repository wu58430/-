
#include"ty.h"
#include "file.h"
#include "all.h"
#include "distorted.h"
using namespace cv;
using namespace std;
#define alpha 1



int main() {
    string dir = "../img7";
    int conor_width = 8;
    int conor_high = 6;

    //读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
    cout << "开始提取角点………………" << endl;



/*******读取文件夹下所有图片************/
    vector<string> allFileList = getFilesList(dir);
    int pic_num = allFileList.size();
    string pic_name[pic_num]; //储存所有图片路径
    pic_num = get_picname(dir, pic_name);

/********获取去畸变参数************/
    string pic_name_ok[pic_num];
    double distorted_parameter[11];

/* 0:ImgWidth
 * 1:ImgHeight
 * 2:fx =K.at<double>(0, 0)
 * 3:fy = K.at<double>(1, 1)
 * 4: ux = K.at<double>(0, 2)
 * 5:  uy = K.at<double>(1, 2);
 * 6: k1 = .at<double>(0, 0),
 * 7:  k2 = D.at<double>(0, 1),
 * 8: k3=  D.at<double>(0, 4),
 * 9: p1 = D.at<double>(0, 2),
 * 10:   p2 = D.at<double>(0, 3);
 */
    pic_num = get_distorted_mat(pic_name, pic_name_ok, pic_num, conor_width, conor_high, distorted_parameter);




/*********************还原******************************/
    int ImgWidth = (int) distorted_parameter[0];
    int ImgHeight = (int) distorted_parameter[1];
    double fx = distorted_parameter[2]
    , fy = distorted_parameter[3]
    , ux = distorted_parameter[4]
    , uy = distorted_parameter[5]
    , k1 = distorted_parameter[6]
    , k2 = distorted_parameter[7]
    , k3 = distorted_parameter[8]
    , p1 = distorted_parameter[9]
    , p2 = distorted_parameter[10];


    string InputPath = pic_name[0];
    cv::Mat img_tmp = cv::imread(InputPath);
    cvtColor(img_tmp, img_tmp, COLOR_RGB2GRAY);
//
//
    ///**********自动计算结果图大小**********/

    int max_x = -9999999, max_y = -9999999, min_x = 9999999, min_y = 9999999;
    for (int i = 0; i < ImgHeight; i++) {
        for (int j = 0; j < ImgWidth; j++) {
            double xDistortion = (j - ux) / fx;
            double yDistortion = (i - uy) / fy;

            double xCorrected, yCorrected;

            double x0 = xDistortion;
            double y0 = yDistortion;
            for (int j = 0; j < 10; j++) {
                double r2 = xDistortion * xDistortion + yDistortion * yDistortion;

                double distRadialA = 1 / (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
                double distRadialB = 1.;

                double deltaX = 2. * p1 * xDistortion * yDistortion + p2 * (r2 + 2. * xDistortion * xDistortion);
                double deltaY = p1 * (r2 + 2. * yDistortion * yDistortion) + 2. * p2 * xDistortion * yDistortion;

                xCorrected = (x0 - deltaX) * distRadialA * distRadialB;
                yCorrected = (y0 - deltaY) * distRadialA * distRadialB;

                xDistortion = xCorrected;
                yDistortion = yCorrected;
            }
            xCorrected = xCorrected * fx + ux;
            yCorrected = yCorrected * fy + uy;

            if (i == 0 && j == 0) {

                min_x = min_int(min_x, xCorrected);
                min_y = min_int(min_y, yCorrected);//左上
            }
            if (i == 0 && j == ImgWidth - 1) {
                //右上
                min_y = min_int(min_y, yCorrected);
                max_x = max_int(min_x, xCorrected);

            }
            if (i == ImgHeight - 1 && j == 0) {
                //左下
                min_x = min_int(min_x, xCorrected);
                max_y = max_int(min_y, yCorrected);

            }
            if (i == ImgHeight - 1 && j == ImgWidth - 1) {
                //右下
                max_x = max_int(min_x, xCorrected);

                max_y = max_int(min_y, yCorrected);

            }

        }


    }


    int ImgWidth_out;
    int ImgHeight_out;
    if (alpha) {
        ImgWidth_out = max_x - min_x;
        ImgHeight_out = max_y - min_y;

    } else {
        ImgWidth_out = ImgWidth;
        ImgHeight_out = ImgHeight;
    }
    /**********定义映射矩阵**************/
    unsigned char back_color = 255;
    unsigned char *distorted_img[ImgHeight_out][ImgWidth_out];
    unsigned char now_img[ImgHeight][ImgWidth];
    for (int i = 0; i < ImgHeight; i++)
        for (int j = 0; j < ImgWidth; j++)
            now_img[i][j] = img_tmp.at<uchar>(i, j);

    Mat out1 = Mat(ImgHeight_out, ImgWidth_out, CV_8UC1);


    int max_width = max_x;
    int max_high = max_y;
    int move_x, move_y;
    if (alpha) {
        move_x = (ImgWidth_out - max_width);
        move_y = (ImgHeight_out - max_high);
    } else {
        move_x = 0;
        move_y = 100;
    }
    cout << "move:" << move_x << "\t" << move_y << endl;

    cout << "size:" << ImgHeight_out << "\t" << ImgWidth_out << endl;


    for (int i = -move_y; i < ImgHeight_out; i++) {
        for (int j = -move_x; j < ImgWidth_out; j++) {
            double xCorrected = (j - ux) / fx;
            double yCorrected = (i - uy) / fy;

            double xDistortion, yDistortion;

            //我们已知的是经过畸变矫正或理想点的坐标；
            double r2 = xCorrected * xCorrected + yCorrected * yCorrected;

            double deltaRa = 1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            double deltaRb = 1 / (1.);
            double deltaTx = 2. * p1 * xCorrected * yCorrected + p2 * (r2 + 2. * xCorrected * xCorrected);
            double deltaTy = p1 * (r2 + 2. * yCorrected * yCorrected) + 2. * p2 * xCorrected * yCorrected;

            //下面为畸变模型；
            xDistortion = xCorrected * deltaRa * deltaRb + deltaTx;
            yDistortion = yCorrected * deltaRa * deltaRb + deltaTy;

            //最后再次通过相机模型将归一化的坐标转换到像素坐标系下；
            xDistortion = xDistortion * fx + ux;
            yDistortion = yDistortion * fy + uy;
            if (i + move_y >= 0 && i + move_y < ImgHeight_out && j + move_x >= 0 && j + move_x <= ImgWidth_out) {
                if (yDistortion >= 0 && yDistortion < ImgHeight && xDistortion >= 0 && xDistortion < ImgWidth) {
                    distorted_img[i + move_y][j + move_x] = &now_img[(int) yDistortion][(int) xDistortion];
                } else distorted_img[i + move_y][j + move_x] = &back_color;
            }
        }
    }
    cout<<"show"<<endl;
    for (int p = 1; p < pic_num; p++) {
        cv::Mat img_tmp = cv::imread(pic_name[p]);
        cvtColor(img_tmp, img_tmp, COLOR_RGB2GRAY);

        for (int i = 0; i < ImgHeight; i++)
            for (int j = 0; j <ImgWidth; j++)
                now_img[i][j] =    img_tmp.at<uchar>(i, j) ;

        for (int i = 0; i < ImgHeight_out; i++)
            for (int j = 0; j < ImgWidth_out; j++)
                out1.at<uchar>(i, j) = 255;

        for (int i = 0; i < ImgHeight_out; i++)
            for (int j = 0; j < ImgWidth_out; j++)
                out1.at<uchar>(i, j) = *distorted_img[i][j];

        cv::imshow("ou1", out1);
        cv::imshow("RawImage", img_tmp);
        cv::waitKey(0);
    }


}


