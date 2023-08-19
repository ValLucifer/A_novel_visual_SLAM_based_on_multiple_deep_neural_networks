//
//  Created by Lucifer on 2022/7/12.
//

#ifndef _SG_SLAM_SG_DETECTMATCHER_H_
#define _SG_SLAM_SG_DETECTMATCHER_H_

#include "sg_slam/SG_superpoint.h"
#include "sg_slam/superglue.h"

#include <memory>  // 智能指针
#include <string>
#include <mutex>  // 参见 sg_slam/frame.h
#include <tuple>

// #include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>

namespace sg_slam {

/**
 * @brief superglue(superpoint) 检测及匹配器。
 * 整个系统共享一组模型参数，故都使用static, 因为需要多线程使用匹配器所以匹配器需要上锁.
 * 使用单例模式设计： 参见 sg_slam/config.h
 */
class sg_DetectMatcher
{
private:
    // N.B.: 只在主函数或主线程设置一次不用加锁
    static std::shared_ptr< sg_DetectMatcher > m_instance_;
    // static std::mutex m_instance_mutex_;

    sg_DetectMatcher() {};
public:
    ~sg_DetectMatcher() = default;

    // set detector and Matcher
    static void set_DetectorAndMatcher(std::string &sp_weight_file, std::string &sg_weight_file, int sp_gpu_id, int sg_gpu_id);

    // detect, input: image(cv::Mat), output: 

    static SG_SuperPoint m_detector;
    static SuperGlue m_matcher;
    static std::mutex m_matcher_mutex;

    static std::tuple<std::vector< cv::KeyPoint >, torch::Tensor> detect(const cv::Mat &im, int max_keypoints = 1000);
    static std::map<std::string, torch::Tensor> descMatch(torch::Tensor desc0, torch::Tensor desc1, float match_threshold=0.2);
    
};

/**
 * @brief RANSAC 处理函数
 * output:
 *          std::vector<cv::DMatch>   // vMatches_RANSAC_01
 * 
 */
std::vector<cv::DMatch> sg_RANSAC(
    const std::vector<cv::KeyPoint> &vKeys0,
    const std::vector<cv::KeyPoint> &vKeys1,
    const std::vector<cv::DMatch> &vMatches01,
    int min_number
);

/**
 * @brief 将desc(Tensor ([b, d, n]/[c, h, w], b=1)) 转换为 cv::Mat进行存储
 * 
 */
cv::Mat convertDesTensor2cvMat(torch::Tensor des_tensor);

/**
 * @brief 将存储的 desc(cv::Mat) 转换为 desc(Tensor ([b, d, n]/[c, h, w], b=1))
 * 
 */
torch::Tensor convertDescvMat2Tensor(cv::Mat des_mat);

/**
 * @brief 将desc(Tensor ([b, d, n]/[c, h, w], b=1)) 转换为 std::vector 进行存储
 * 
 */
std::vector<float> convertDesTensor2Vector(torch::Tensor des_tensor);

/**
 * @brief 将存储的 desc(std::vector) 转换为 desc(Tensor ([b, d, n]/[c, h, w], b=1))
 * 
 */
torch::Tensor convertDesVector2Tensor(const std::vector<float> &des_vector, int64_t b, int64_t d, int64_t n);


}  // namespace sg_slam

#endif  // _SG_SLAM_SG_DETECTMATCHER_H_
