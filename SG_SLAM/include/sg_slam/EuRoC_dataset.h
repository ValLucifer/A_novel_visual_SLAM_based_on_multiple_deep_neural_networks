//
//  Created by Lucifer on 2022/10/27.
//

#ifndef _SG_SLAM_EUROC_DATASET_H_
#define _SG_SLAM_EUROC_DATASET_H_

#include <Eigen/Core>  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW, 参见sg_slam/frame.h
#include <memory>  // 智能指针
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "sg_slam/frame.h" 
#include "sg_slam/camera.h"
#include "sg_slam/KITTI_dataset.h"

namespace sg_slam {

/**
 * @brief EuRoC 数据集读取
 * 构造时传入数据集路径, 配置文件的dataset_dir为数据集路径
 * Init执行之后，才可获得相机和下一帧图像
 * 
 */
class EuRoC_dataset : public Dataset 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  // 参见sg_slam/frame.h
    using this_type = EuRoC_dataset;  // 本类别名
    using Ptr = std::shared_ptr<EuRoC_dataset>;  // 本类共享/弱指针别名

    int current_image_index_ = 0;

    std::vector<double> vTimestamps_;
    std::vector<std::string> vTimestamps_str_;

    EuRoC_dataset(const std::string &dataset_path);

    ~EuRoC_dataset() = default;

    // 初始化, 输入图像缩放比例， 返回是否初始化成功. N.B.: 自改
    bool Init(double scale = 1.0);

    // create and return the next frame containing stereo images
    // 仅填充图像数据(非引用, 但是是浅拷贝)和帧ID没有提取特征
    Frame::Ptr NextFrame();

    // get camera by id
    Camera::Ptr getCamera(int camera_id) const {
        // https://blog.csdn.net/zpznba/article/details/89669888
        return cameras_.at(camera_id);
    }

private:
    std::string dataset_path_;
    double m_scale_ = 1.0;  // 目前没有使用 

    std::vector< Camera::Ptr > cameras_;

    // OpenCV 计算的到的 EuRoC 矫正参数.
    cv::Mat m_M1l_;
    cv::Mat m_M2l_;
    cv::Mat m_M1r_;
    cv::Mat m_M2r_;

};  // class EuRoC_dataset


}  // namespace sg_slam

#endif  // _SG_SLAM_EUROC_DATASET_H_
