//
//  Created by Lucifer on 2022/7/11.
//

#include "sg_slam/camera.h"
#include "sg_slam/config.h"  // 2022100501 加

namespace sg_slam {

Camera::Camera() {
    int camera_type = Config::Get<int>(static_cast<std::string>("camera.type"));

    if( camera_type == 0 ) {
        m_camera_type_ = CameraType::MONO;
    } else if( camera_type == 1 ) {
        m_camera_type_ = CameraType::SETERO;
    } else if( camera_type == 2 ) {
        m_camera_type_ = CameraType::RGBD;
    }

    // 2022101101 加
    int cut_flag = Config::Get<int>(static_cast<std::string>("camera.cut"));
    if( cut_flag == 0 ) {
        m_image_cut_flag = false;
    } else {
        m_image_cut_flag = true;
    }

    int m_image_width = Config::Get<int>(static_cast<std::string>("camera.width"));
    int m_image_height = Config::Get<int>(static_cast<std::string>("camera.height"));
    int m_image_cut_width = Config::Get<int>(static_cast<std::string>("camera.cut_width"));
    int m_image_cut_height = Config::Get<int>(static_cast<std::string>("camera.cut_height"));

    std::cout << std::endl;
    std::cout << "(Camera::Camera()): m_camera_type_ = " << static_cast<int>(m_camera_type_) << std::endl;
    std::cout << std::endl;

    if(m_camera_type_ == CameraType::RGBD) {
        fx_ = Config::Get<double>(static_cast<std::string>("camera.fx"));
        fy_ = Config::Get<double>(static_cast<std::string>("camera.fy"));
        cx_ = Config::Get<double>(static_cast<std::string>("camera.cx"));
        cy_ = Config::Get<double>(static_cast<std::string>("camera.cy"));
        depth_scale_ = Config::Get<float>(static_cast<std::string>("camera.depth_scale"));
        fx_baseline_ = Config::Get<double>(static_cast<std::string>("camera.bf"));
        baseline_ = fx_baseline_ / fx_;
        K_baseline_ = fx_baseline_;

        Vec6 se3_zero;
        se3_zero.setZero();
        pose_ = SE3::exp(se3_zero);
        pose_inv_ = pose_.inverse();
    }

    // 2022101101 加
    if(m_image_cut_flag) {
        int start_x = (m_image_width - m_image_cut_width) / 2;
        int start_y = (m_image_height - m_image_cut_height) / 2;
        cx_ = cx_ - start_x;
        cy_ = cy_ - start_y;
        m_image_cut_select = cv::Rect(start_x, start_y, m_image_cut_width, m_image_cut_height);
    }

    std::cout << std::endl;
    std::cout << "(Camera::Camera()): m_camera_type_ = " << static_cast<int>(m_camera_type_) << std::endl;
    std::cout << "(Camera::Camera()): fx_ = " << fx_ << std::endl;
    std::cout << "(Camera::Camera()): fy_ = " << fy_ << std::endl;
    std::cout << "(Camera::Camera()): cx_ = " << cx_ << std::endl;
    std::cout << "(Camera::Camera()): cy_ = " << cy_ << std::endl;
    std::cout << "(Camera::Camera()): depth_scale_ = " << depth_scale_ << std::endl;
    std::cout << "(Camera::Camera()): fx_baseline_ = " << fx_baseline_ << std::endl;
    std::cout << "(Camera::Camera()): baseline_ = " << baseline_ << std::endl;
    std::cout << "(Camera::Camera()): K_baseline_ = " << K_baseline_ << std::endl;
    std::cout << "(Camera::Camera()): pose_ = "<< std::endl;
    std::cout << pose_.matrix() << std::endl;
    std::cout << "(Camera::Camera()): m_image_cut_flag = " << m_image_cut_flag << std::endl;
    std::cout << "(Camera::Camera()): m_image_width = " << m_image_width << std::endl;
    std::cout << "(Camera::Camera()): m_image_height = " << m_image_height << std::endl;
    std::cout << "(Camera::Camera()): m_image_cut_width = " << m_image_cut_width << std::endl;
    std::cout << "(Camera::Camera()): m_image_cut_height = " << m_image_cut_height << std::endl;
    std::cout << std::endl;
}

Camera::Camera(double fx, double fy, double cx, double cy, double baseline, double k_baseline, const SE3 &pose, const Mat34 &k_pose)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), K_baseline_(k_baseline), pose_(pose), K_pose_(k_pose) {
    
    int camera_type = Config::Get<int>(static_cast<std::string>("camera.type"));

    if( camera_type == 0 ) {
        m_camera_type_ = CameraType::MONO;
    } else if( camera_type == 1 ) {
        m_camera_type_ = CameraType::SETERO;
    } else if( camera_type == 2 ) {
        m_camera_type_ = CameraType::RGBD;
    }
    
    pose_inv_ = pose.inverse();
    fx_baseline_ = fx_ * baseline_;


    // 2022101101 加
    int cut_flag = Config::Get<int>(static_cast<std::string>("camera.cut"));
    if( cut_flag == 0 ) {
        m_image_cut_flag = false;
    } else {
        m_image_cut_flag = true;
    }

    int m_image_width = Config::Get<int>(static_cast<std::string>("camera.width"));
    int m_image_height = Config::Get<int>(static_cast<std::string>("camera.height"));
    int m_image_cut_width = Config::Get<int>(static_cast<std::string>("camera.cut_width"));
    int m_image_cut_height = Config::Get<int>(static_cast<std::string>("camera.cut_height"));

    // 2022101101 加
    if(m_image_cut_flag) {
        int start_x, start_y;
        int image_cut_width = m_image_width - m_image_cut_width;
        int image_cut_height = m_image_height - m_image_cut_height;
        if(image_cut_width%2 == 0) {
            start_x = image_cut_width / 2;
            std::cout << "start_x: 1" << std::endl;
        } else {
            start_x = (image_cut_width + 1) / 2;
            std::cout << "start_x: 2" << std::endl;
        }

        if(image_cut_height%2 == 0) {
            start_y = image_cut_height / 2;
            std::cout << "start_y: 1" << std::endl;
        } else {
            start_y = (image_cut_height + 1) / 2;
            std::cout << "start_y: 2" << std::endl;
        }
        cx_ = cx_ - start_x;
        cy_ = cy_ - start_y;
        m_image_cut_select = cv::Rect(start_x, start_y, m_image_cut_width, m_image_cut_height);
        auto K_cut = K();
        Mat34 KT = K_cut * pose_.matrix3x4();
        K_pose_ = KT;
        auto k_t = KT.col(3);
        K_baseline_ = k_t.norm();
    }

    std::cout << std::endl;
    std::cout << "(Camera::Camera(Stereo)): m_camera_type_ = " << static_cast<int>(m_camera_type_) << std::endl;
    std::cout << "(Camera::Camera(Stereo)): fx_ = " << fx_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): fy_ = " << fy_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): cx_ = " << cx_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): cx_ = " << cy_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): depth_scale_ = " << depth_scale_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): fx_baseline_ = " << fx_baseline_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): baseline_ = " << baseline_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): K_baseline_ = " << K_baseline_ << std::endl;
    std::cout << "(Camera::Camera(Stereo)): pose_ = "<< std::endl;
    std::cout << pose_.matrix() << std::endl;
    std::cout << std::endl;
}

Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) {
    return pose_ * T_c_w * p_w;
}

Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) {
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Vec2 Camera::camera2pixel(const Vec3 &p_c) {
    return Vec2(
        fx_*p_c(0, 0)/p_c(2, 0) + cx_, 
        fy_*p_c(1, 0)/p_c(2, 0) + cy_
    );
}

Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) {
    return Vec3(
        (p_p(0, 0) - cx_) * depth / fx_, 
        (p_p(1, 0) - cy_) * depth / fy_, 
        depth
    );
}

Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) {
    return camera2pixel(world2camera(p_w, T_c_w));
}

Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) {
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

}  // namespace sg_slam

