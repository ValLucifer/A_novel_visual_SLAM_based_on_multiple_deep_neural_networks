//
//  Created by Lucifer on 2022/7/11.
//

#ifndef _SG_SLAM_CAMERA_H_
#define _SG_SLAM_CAMERA_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW, 参见sg_slam/frame.h
#include <memory>  // 智能指针

#include "sg_slam/common_include.h"

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace sg_slam {

enum class CameraType { MONO, SETERO, RGBD };

/**
 * @brief Pinhole stereo camera model and RGB-D model
 * 
 */
class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见sg_slam/frame.h
    using this_type = Camera;  // 本类别名
    using Ptr = std::shared_ptr<Camera>;  // 本类共享/弱指针别名

    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0, baseline_ = 0.0;  // Camera intrinsics.
    double K_baseline_ = 0; // 来检查读取的 “calib.txt” 是否正确. // N.B.: 可以不填充
    // 2022100501 加 (与opencv 关键点数据类型float匹配)
    float depth_scale_ = 5000.0;  // RGB-D 相机深度尺度因子 (TUM)

    SE3 pose_;  // extrinsic(外参), from stereo camera to single camera. (stereo 归一化坐标，变换)(变换到左相机，若相机为左相机则是单位变化),  以cam0为参考的世界坐标位姿.
    Mat34 K_pose_;  // 仅用来检查读取的 “calib.txt” 是否正确. N.B.: 可以不填充 // 不是位姿，不满足位姿约束条件, K 不是旋转矩阵.
    SE3 pose_inv_;  // inverse of extrinsic

    // N.B.: 在具体数据集或输入数据流中填充.
    cv::Mat m_distcoef_;  // 相机图像的去畸变参数 // N.B.: My Add 为了与KITTI之外的数据集兼容

    double fx_baseline_ = 0;  // N.B.: My Add 为了g2o stereo优化

    // 2022100501 加
    CameraType m_camera_type_ = CameraType::SETERO;

    // 2022101101 加
    bool m_image_cut_flag = false;
    cv::Rect m_image_cut_select;

    int m_image_width;
    int m_image_height;
    int m_image_cut_width;
    int m_image_cut_height;

    
    // 用与直接读取文件中的相机参数(目前仅RGB-D使用)
    Camera();

    // Stereo 数据集构造函数
    Camera(double fx, double fy, double cx, double cy, double baseline, double k_baseline, const SE3 &pose, const Mat34 &k_pose);

    ~Camera() = default;

    SE3 pose() const { 
        return pose_; 
    }

    // return intrinsic matrix (左乘)
    Mat33 K() const {
        Mat33 k;
        k << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
        return k;
    }

    // coordinate transform: world, camera, pixel.
    Vec3 world2camera(const Vec3 &p_w, const SE3 &T_c_w);

    Vec3 camera2world(const Vec3 &p_c, const SE3 &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);

    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1.0);

    Vec2 world2pixel(const Vec3 &p_w, const SE3 &T_c_w);

    Vec3 pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth = 1.0);
};  // class Camera

}  // namespace sg_slam

#endif  // _SG_SLAM_CAMERA_H_
