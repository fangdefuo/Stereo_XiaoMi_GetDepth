#include <stdio.h>
#include <iostream>  
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mynteye/api/api.h"
#include "mynteye/logger.h"
//#include "util/pc_viewer.h"
#include <opencv2/opencv.hpp>
#include <string>
#include "util/cv_painter.h"

#include "mynteye/device/device.h"
#include "mynteye/device/utils.h"
#include "mynteye/util/times.h"


using namespace std;
using namespace cv;

MYNTEYE_USE_NAMESPACE

///////////////////////////***********************新加的//////////////////////////////////

const int imageWidth = 752;                             //摄像头的分辨率  
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Vec3f  point3, point3_depth, point3_line;
Vec2f  point2,  point2_1;
float d;
float depth_value = 0;
float small_depth_value  = 0;
float depth_value_line;

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 8, uniquenessRatio =20, numDisparities=16;
Ptr<StereoBM> bm = StereoBM::create(16, 9);//（uniquenessRatio，blockSize）

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 414.489, 0, 416.122,
    0, 387.555, 229.863,
    0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.323097, 0.0879537, -3.77881e-06, -0.000145117, 0.00000);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 420.15, 0, 421.411,
    0, 371.704, 237.379,
    0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.330712, 0.0903598, 0.00068584, 0.000149492, 0.00000);

Mat T = (Mat_<double>(3, 1) << -115.35694, -0.27769, 3.244748);//T平移向量right to left
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
//Mat R;//R 旋转矩阵
Mat R = (Mat_<double>(3, 3) << 0.99984, -0.00659, 0.01673,
       0.00663, 0.99997, -0.00255, 
       -0.01672, 0.00267, 0.99985);

/////////////////////////////////根据两点画延长线////////////////////////////////////////////////////////

void OnDrawDotline(CvPoint s, CvPoint e,Mat disp8)
{
	CvPoint pa,pb;
	
	double k=(s.y-e.y)/(s.x-e.x+0.000001);//不加0.000001 会变成曲线，斜率可能为0，即e.x-s.x可能为0
	
//	double h=disp8.rows, 
  int w=disp8.cols;	
	
	pa.x=w;
	pa.y=s.y+k*(w-s.x);	
	
	line(disp8,e,pa,Scalar(255,0,0), 3, 8);	//向右画线
	
	
	pb.y=e.y-k*e.x;
	pb.x=0;
	
	line(disp8,pb,s,Scalar(255,0,0), 3, 8 );	//向左画线
 
}



/////////////////////////////////////////根据两点画延长线结束////////////////////////////////////////////////

/*****立体匹配*****/
void stereo_match(int,void*)
{
    bm->setBlockSize(2*blockSize+5);     //SAD窗口大小，5~21之间为宜
    bm->setROI1(validROIL);//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
    bm->setROI2(validROIR);//在裁剪区域外视差为0
    bm->setPreFilterCap(31);//预处理滤波器的截断值，预处理的输出值仅保留【-PreFilterCap，PreFilterCap】
    bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    bm->setNumDisparities(numDisparities*16+16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    bm->setTextureThreshold(10); //低纹理区域的判断阈值，如果当前SAD窗口内所有邻域像素点的x的导数绝对值之和小于阈值，则该窗口
    //对应的像素视差为0
    bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配，视差唯一性百分比，视差窗口范围内最低代价是次低代价
    //的（1+uniquenessRatio/100）倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为0；
    bm->setSpeckleWindowSize(100);//检查视差区域连通区域变化度的窗口大小，，0为取消检查
    bm->setSpeckleRange(32);//视差变化阈值，当窗口内视差大于阈值时，该窗口内的视差清零
    bm->setDisp12MaxDiff(-1);///左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间最大容许差异，超过该差异的视差值
    //将清零，-1不执行视差检查
    Mat disp, disp8;
    bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    
    ////////////////////////////////////找到最小值和次小值///////////////////////////////////////////
    int width = xyz.cols;
	  int height = xyz.rows;
    point3_depth = xyz.at<Vec3f>(240, 376);//求最小值

    point2[0] = 376;
    point2[1] = 240;

    small_depth_value = point3_depth[2];

    point3_line = xyz.at<Vec3f>(250, 371);
    depth_value_line = small_depth_value + 1000.0;//point3_line[2];//求次小的值

    for (int row = 20; row < height-20; row++) {
		   for (int col = 20; col < width-20; col++) {
			    point3_depth = xyz.at<Vec3f>(row, col);
          depth_value = point3_depth[2];
			    if (small_depth_value > depth_value){
               
               depth_value_line = small_depth_value;
               small_depth_value = depth_value;
               
               point2_1[0] = point2[0];
               point2_1[1] = point2[1];

               point2[0] = col;
               point2[1] = row;
          }  
          else if(depth_value < depth_value_line && depth_value != small_depth_value){
               
               depth_value_line =  depth_value;
               point2_1[0] = col;
               point2_1[1] = row;
          }    
	    }
  	}
  /////////////////////////////////////拟合直线//////////////////////////////////////

      // for (int row = 20; row < height-20; row++) {
		  //   for (int col = 20; col < width-20; col++) {
			//     point3_line = xyz.at<Vec3f>(row, col);
      //     depth_value_line = point3_line[2];
        
			   // if (std::abs(small_depth_value - depth_value_line) < 200.0){
  line(disp8, Point( point2_1[0], point2_1[1]), Point(point2[0], point2[1]), Scalar(255, 0, 0), 3, 8);  
              //  XX = col;
              //  YY = row; 
  OnDrawDotline(Point( point2_1[0],  point2_1[1]),Point(point2[0],  point2[1] ),disp8);
      //     }      
	    //   }
    	// }
     


  ///////////////////////////////////拟合直线结束/////////////////////////////////////////////
  cout << "in world coordinate small depth is: " << small_depth_value << "mm" << endl;
  cout << "the point is :" << "(" << point2[0] << "," << point2[1] << ")" << endl;
  
  ////////////////////////////////////遍历结束///////////////////////////////////////////////////
  //line(disp8, Point(376, 240), Point(point2[0], point2[1]), Scalar(255, 0, 0), 3, 8);  
  imshow("disparity", disp8);
}

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin <<"in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;

        point3 = xyz.at<Vec3f>(origin);
	
		    // //cout << "point3[0]:" << point3[0] << "point3[1]:" << point3[1] << "point3[2]:" << point3[2]<<endl;
	      cout << "世界座标系下深度为：" << endl;
        cout <<  "  z: " << point3[2] << "mm" << endl;
		    // cout << "x: " << point3[0] << "  y: " << point3[1] << "  z: " << point3[2] << endl;
		    // d = point3[0] * point3[0]+ point3[1] * point3[1]+ point3[2] * point3[2];
		    // d = sqrt(d);   //mm
		    // // cout << "距离是:" << d << "mm" << endl;
		
		    // d = d / 10.0;   //cm
        // cout << "距离是:" << d << "cm" << endl;

        // // d = d/1000.0;   //m
	    	// // cout << "距离是:" << d << "m" << endl;

    

        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
        break;
    }
}

// //检测距离
// void detectDistance(Mat&pointCloud)
// {
//     if (pointCloud.empty())
//     {
//         return;
//     }

//     // 提取深度图像
//     vector<cv::Mat> xyzSet;
//     split(pointCloud, xyzSet);
//     cv::Mat depth;
//     xyzSet[2].copyTo(depth);

//     // 根据深度阈值进行二值化处理
//     double maxVal = 0, minVal = 0;
//     cv::Mat depthThresh = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
//     cv::minMaxLoc(depth, &minVal, &maxVal);
//     double thrVal = minVal * 1.5;
//     threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
//     depthThresh.convertTo(depthThresh, CV_8UC1);
//     //imageDenoising(depthThresh, 3);

//     double  distance = depth.at<float>(pic_info[0], pic_info[1]);
//     cout << "distance:" << distance << endl;


// }


//////////////////////////////****************************结束///////////////////////////

string convertToString(double d);

int main(int argc, char *argv[]) {
  auto &&api = API::Create(argc, argv);
  if (!api) return 1;

  bool ok;
  auto &&request = api->SelectStreamRequest(&ok);
  if (!ok) return 1;
  api->ConfigStreamRequest(request);

  double fps;
  double t = 0.01;
  std::cout << "depth fps:" << std::endl;

  api->EnableStreamData(Stream::DEPTH);
  api->EnableStreamData(Stream::DISPARITY_NORMALIZED);

  api->Start(Source::VIDEO_STREAMING);

  ////////////////////////////新加的程序///////////////////////////////////////
  api->SetDisparityComputingMethodType(DisparityComputingMethod::SGBM);
  api->EnableStreamData(Stream::DISPARITY_NORMALIZED);
  
  //api->EnableStreamData(Stream::POINTS);
  // CVPainter painter;
 // PCViewer pcviewer;

  if (argc == 2) {
    std::string config_path(argv[1]);
    if (api->ConfigDisparityFromFile(config_path)) {
      LOG(INFO) << "load disparity file: "
                << config_path
                << " success."
                << std::endl;
    } else {
      LOG(INFO) << "load disparity file: "
                << config_path
                << " failed."
                << std::endl;
    }
  }

  api->Start(Source::VIDEO_STREAMING);


  //////////////////////////////////////////////////////////////////////////

  cv::namedWindow("frame");
  cv::namedWindow("depth_real");
  cv::namedWindow("depth_normalized");
  cv::namedWindow("disparity_normalized");
  cv::namedWindow("min_depth_value");
 // cv::namedWindow("points");

  
  while (true) {
    api->WaitForStreams();
/////////////////////////////////////////////左右两张图连接在一起
    auto &&left_data = api->GetStreamData(Stream::LEFT);
    auto &&right_data = api->GetStreamData(Stream::RIGHT);

    if (!left_data.frame.empty() && !right_data.frame.empty()) {
      cv::Mat img;
      cv::hconcat(left_data.frame, right_data.frame, img);
      cv::imshow("frame", img);
    }

//////////////////////////////////////////////深度图像图 　　　　　　　　　　　　　　　　　　　　　　　
    auto &&depth_data = api->GetStreamData(Stream::DEPTH);
    if (!depth_data.frame.empty()) {
      double t_c = cv::getTickCount() / cv::getTickFrequency();
      fps = 1.0/(t_c - t);
      printf("\b\b\b\b\b\b\b\b\b%.2f", fps);
      t = t_c;
      cv::imshow("depth_real", depth_data.frame);
    }  // CV_16UC1
      
/////////////////////////////////新加的//////*****************//////////////////////////////

  // Rodrigues(rec, R); //Rodrigues变换
  //执行双目校正
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
  //分别生成两个图像校正所需的像素映射矩阵
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    grayImageL = left_data.frame.clone();
    //cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    grayImageR = right_data.frame.clone();
    //cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

    imshow("ImageL Before Rectify", grayImageL);
    imshow("ImageR Before Rectify", grayImageR);
    //分别对两个图像进行校正
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    imshow("ImageL After Rectify", rgbRectifyImageL);
    imshow("ImageR After Rectify", rgbRectifyImageR);
    
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道


    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
   // imshow("rectified", canvas);

    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0,0);
    line(canvas, Point(376 * sf, 240 * sf), Point(point2[0] * sf, point2[1] * sf), Scalar(255, 0, 0), 3, 8);
    line(canvas, Point( point2_1[0] * sf, point2_1[1] * sf), Point(point2[0] * sf, point2[1] * sf), Scalar(255, 0, 0), 3, 8);
    OnDrawDotline(Point( point2_1[0] * sf,  point2_1[1] * sf),Point(point2[0] * sf,  point2[1] * sf),canvas);
    imshow("rectified", canvas);
    //detectDistance(Mat&pointCloud)；



  //cout << min_value << endl;
  //putText(src, ss_depth , Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);

 // cv::imshow("min_depth_value", src);
  //waitKey(50);
    //cv::imshow("min_depth_value", src);
//////////////////////////////////////////结束/***********************///////////////////////////////////////////


    

/////////////////////////////////////////////// 深度归一化图
    auto &&disp_norm_data = api->GetStreamData(Stream::DISPARITY_NORMALIZED);
    if (!disp_norm_data.frame.empty()) {
      cv::imshow("depth_normalized", disp_norm_data.frame);  // CV_8UC1
    }
/////////////////////////////新加的//////////////////////////////////////////
  //  auto &&disp_norm_data = api->GetStreamData(Stream::DISPARITY_NORMALIZED);
  //   if (!disp_norm_data.frame.empty()) {
  //     // double t_c = cv::getTickCount() / cv::getTickFrequency();
  //     // fps = 1.0/(t_c - t);
  //     // printf("\b\b\b\b\b\b\b\b\b%.2f", fps);
  //     // t = t_c;
  //     cv::imshow("disparity_normalized", disp_norm_data.frame);  // CV_8UC1
  //   }

//Mat src, dst;
    // int row, col;
     //string ss_depth;
     //double min_value = 1;
     //src = disp_norm_data.frame.clone();

     int width = disp_norm_data.frame.cols;
     cout << width << "宽度" << endl;
     waitKey(500);
	   int height = left_data.frame.rows;
     cout << height << "高度" << endl;
     waitKey(500);
	  // for (int row = 50; row < height-50; row++) {
		  // for (int col = 50; col < width-50; col++) {
			 //  char depth_value = disp_norm_data.frame.at<char>(row, col);
			 // if (depth_value > min_value){
        //  min_value = depth_value;
         // cout << "最小值："　<< min_value << endl;
         // cout << depth_value << endl;
           //cv::imshow("points", min_value);
         // LOG(INFO) << "深度值为"　<< min_value << endl;
         // ss_depth = std::to_string(min_value);
         // putText(src, ss_depth , Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2, LINE_AA);
       //  double* pint = &min_value;
         // painter.DrawImgData(src,pint);
         
          //cv::imshow("min_depth_value", src);
           //painter.DrawImgData(img, *left_data.img);
        //}
			  //cv::imshow("min_depth_value", src);
		//}
	//}






  auto &&disp_data = api->GetStreamData(Stream::DISPARITY);
    if (!disp_data.frame.empty()) {
      cv::imshow("disparity_normalized", disp_data.frame);
    }

   // auto &&points_data = api->GetStreamData(Stream::POINTS);
    //if (!points_data.frame.empty()) {
      // double t_c = cv::getTickCount() / cv::getTickFrequency();
      // fps = 1.0/(t_c - t);
      // printf("\b\b\b\b\b\b\b\b\b%.2f", fps);
      // t = t_c;
      
     // cv::imshow("points", points_data.frame);

      //pcviewer.Update(points_data.frame);
   // }
    

/////////////////////////////////////结束////////////////////////////////////////////////
    char key = static_cast<char>(cv::waitKey(1));
    if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
      break;
    }
  }

  api->Stop(Source::VIDEO_STREAMING);
  return 0;
}

string convertToString(double d) {
	ostringstream os;
	if (os << d)
		return os.str();
	return "invalid conversion";
}
