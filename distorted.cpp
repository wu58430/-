//
// Created by RUBO on 2022/10/6.
//
#include "distorted.h"
int get_distorted_mat(string pic_name[], string pic_name_ok[], int pic_num, int conor_width, int conor_high,
                      double parameter[]) {
    int ok_pic = 0;
    Size image_size;  /* 图像的尺寸 */
    Size board_size = Size(conor_width, conor_high);   //6 4 10 7 9 6   69 68 68/* 标定板上每行、列的角点数 */
    vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
    vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
    for (int image_count = 0; image_count < pic_num; image_count++) {
        string filename = pic_name[image_count];

        Mat imageInput = imread(filename);
        if (image_count == 0)  //读入第一张图片时获取图像宽高信息
        {
            image_size.width = imageInput.cols;
            image_size.height = imageInput.rows;
            cout << "image_size.width = " << image_size.width << endl;
            cout << "image_size.height = " << image_size.height << endl;
        }

        /* 提取角点 */
        if (0 == findChessboardCorners(imageInput, board_size, image_points_buf)) {
            cout << filename << " 找不到角点\n"; //找不到角点
        } else if (image_size.width != imageInput.cols) {
            cout << "图片尺寸不同" << endl;
            exit(0);
        } else {
            Mat view_gray;
            cvtColor(imageInput, view_gray, COLOR_RGB2GRAY);
            /* 亚像素精确化 */
            find4QuadCornerSubpix(view_gray, image_points_buf, Size(conor_width, conor_high)); //对粗提取的角点进行精确化
            image_points_seq.push_back(image_points_buf);  //保存亚像素角点
            /* 在图像上显示角点位置 */
            drawChessboardCorners(view_gray, board_size, image_points_buf, true); //用于在图片中标记角点

            pic_name_ok[ok_pic] = filename;
            ok_pic++;

        }
    }
    pic_num = ok_pic;
//    pic_num =  find_corner( pic_name, pic_name_ok, pic_num,  image_size, board_size ,   image_points_buf,  image_points_seq, conor_width, conor_high);
    memcpy(pic_name, pic_name_ok, sizeof(pic_name_ok));
    int total = image_points_seq.size();
    cout << "total = " << total << endl;
    cout << "角点提取完成！\n";
    //以下是摄像机标定
    cout << "开始标定………………";
    /*棋盘三维信息*/
    Size square_size = Size(10, 10);  /* 实际测量得到的标定板上每个棋盘格的大小 */
    vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
    /*内外参数*/
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
    vector<int> point_counts;  // 每幅图像中角点的数量
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
    vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
    /* 初始化标定板上角点的三维坐标 */
    for (int t = 0; t < pic_num; t++) {
        vector<Point3f> tempPointSet;
        for (int i = 0; i < board_size.height; i++) {
            for (int j = 0; j < board_size.width; j++) {
                Point3f realPoint;
                /* 假设标定板放在世界坐标系中z=0的平面上 */
                realPoint.x = i * square_size.width;
                realPoint.y = j * square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);
    }
    /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
    for (int i = 0; i < pic_num; i++) {
        point_counts.push_back(board_size.width * board_size.height);
    }
    /* 开始标定 */
    calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    cout << "标定完成！\n";
    //对标定结果进行评价
    cout << "开始评价标定结果………………\n";
    double total_err = 0.0; /* 所有图像的平均误差的总和 */
    double err = 0.0; /* 每幅图像的平均误差 */
    vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
    for (int i = 0; i < pic_num; i++) {
        vector<Point3f> tempPointSet = object_points[i];
        /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
        projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
        /* 计算新的投影点和旧的投影点之间的误差*/
        vector<Point2f> tempImagePoint = image_points_seq[i];
        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
        Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
        for (int j = 0; j < tempImagePoint.size(); j++) {
            image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
            tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err /= point_counts[i];
    }
    std::cout << "总体平均误差：" << total_err / pic_num << "像素" << endl;
    std::cout << "评价完成！" << endl;
    Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
    cout << "相机内参数矩阵：" << endl;
    cout << cameraMatrix << endl << endl;
    cout << "畸变系数：\n";
    cout << distCoeffs << endl << endl << endl;


    parameter[0] = image_size.width;
    parameter[1] = image_size.height;
    parameter[2] = cameraMatrix.at<double>(0, 0);
    parameter[3] = cameraMatrix.at<double>(1, 1);
    parameter[4] = cameraMatrix.at<double>(0, 2);
    parameter[5] = cameraMatrix.at<double>(1, 2);
    parameter[6] = distCoeffs.at<double>(0, 0);
    parameter[7] = distCoeffs.at<double>(0, 1);
    parameter[8] = distCoeffs.at<double>(0, 4);
    parameter[9] = distCoeffs.at<double>(0, 2);
    parameter[10] = distCoeffs.at<double>(0, 3);
    return ok_pic;


}