//
//  Created by Lucifer on 2022/10/8.
//

#ifndef _SG_SLAM_TUM_DATASET_H_
#define _SG_SLAM_TUM_DATASET_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW, 参见sg_slam/frame.h
#include <memory>  // 智能指针
#include <string>
#include <vector>

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "sg_slam/frame.h" 
#include "sg_slam/camera.h"
#include "sg_slam/KITTI_dataset.h"

namespace sg_slam {

/**
 * @brief TUM数据集读取
 * 构造时传入数据集路径, 配置文件的dataset_dir为数据集路径
 * Init执行之后，才可获得相机和下一帧图像
 * 
 */
class TUM_dataset : public Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见sg_slam/frame.h
    using this_type = TUM_dataset;  // 本类别名
    using Ptr = std::shared_ptr<TUM_dataset>;  // 本类共享/弱指针别名

    int current_image_index_ = 0;

    std::vector<double> vTimestamps_;
    std::vector<double> vDepthTimestamps_;

    std::vector<std::string> vRGB_files_;
    std::vector<std::string> vDepth_files_;

    TUM_dataset(const std::string &dataset_path);

    ~TUM_dataset() = default;

    // 初始化, 输入图像缩放比例， 返回是否初始化成功. N.B.: 自改
    bool Init(double scale = 1.0);

    // create and return the next frame containing RGBD images
    // 仅填充图像数据(非引用, 但是是浅拷贝)和帧ID没有提取特征
    Frame::Ptr NextFrame();

    // get camera
    Camera::Ptr getCamera() const {
        return m_camera_;
    }

private:
    std::string dataset_path_;
    double m_scale_ = 1.0;  // 目前没有使用

    Camera::Ptr m_camera_;
};  // class TUM_dataset


}  // namespace sg_slam

#endif  // _SG_SLAM_TUM_DATASET_H_
