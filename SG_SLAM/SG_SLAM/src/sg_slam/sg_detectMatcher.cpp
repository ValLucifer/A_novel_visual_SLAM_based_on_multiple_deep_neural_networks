//
//  Created by Lucifer on 2022/7/12.
//

#include "sg_slam/sg_detectMatcher.h"
//#include <ATen/ATen.h>

namespace sg_slam {

std::shared_ptr< sg_DetectMatcher > sg_DetectMatcher::m_instance_ = nullptr;
// std::mutex sg_DetectMatcher::m_instance_mutex_;

SG_SuperPoint sg_DetectMatcher::m_detector = nullptr;
SuperGlue sg_DetectMatcher::m_matcher = nullptr;
std::mutex sg_DetectMatcher::m_matcher_mutex;

// --------------------------------------------------------------------------------------------------------------
void sg_DetectMatcher::set_DetectorAndMatcher(std::string &sp_weight_file, std::string &sg_weight_file, int sp_gpu_id, int sg_gpu_id) {
    if(m_instance_ == nullptr) {
        m_instance_ = std::shared_ptr< sg_DetectMatcher >(new sg_DetectMatcher);
    }

    m_instance_->m_detector = SG_superpointModel(sp_weight_file, sp_gpu_id);
    m_instance_->m_matcher = superglueModel(sg_weight_file, sg_gpu_id);
    m_instance_->m_detector->eval();
    m_instance_->m_matcher->eval();
}

// --------------------------------------------------------------------------------------------------------------
std::tuple<std::vector< cv::KeyPoint >, torch::Tensor> sg_DetectMatcher::detect(const cv::Mat &im, int max_keypoints) {
    m_detector->m_config["max_keypoints"] = max_keypoints;
    torch::Tensor x = torch::from_blob(im.clone().data, {1, 1, im.rows, im.cols}, torch::kByte).to(m_detector->m_device);
    x = x.to(torch::kFloat) / 255.0;
    x = x.set_requires_grad(false);
    auto out_list = m_detector->forward(x);
    auto out_dict = m_detector->forwardDetect(out_list);

    auto keypoints_v = m_detector->convertKpts2cvKeypoints(out_dict["keypoints"], out_dict["scores"]);

    std::unique_lock<std::mutex> lck(m_matcher_mutex);
    x = x.to(m_matcher->m_device);
    auto descriptors0 = out_dict["descriptors"][0].unsqueeze(0).to(m_matcher->m_device); 
    auto keypoints0 = out_dict["keypoints"][0].unsqueeze(0).to(m_matcher->m_device);
    auto scores0 = out_dict["scores"][0].unsqueeze(0).to(m_matcher->m_device);
    auto desc0 = m_matcher->normalize_keypoints_encoder(keypoints0, x.sizes(), descriptors0, scores0);

    std::tuple<std::vector< cv::KeyPoint >, torch::Tensor> out_tuple{keypoints_v, desc0};

    return out_tuple;
}

// --------------------------------------------------------------------------------------------------------------
std::map<std::string, torch::Tensor> sg_DetectMatcher::descMatch(torch::Tensor desc0, torch::Tensor desc1, float match_threshold) {
    desc0 = desc0.to(m_matcher->m_device);
    desc1 = desc1.to(m_matcher->m_device);
    std::unique_lock<std::mutex> lck(m_matcher_mutex);
    m_matcher->m_match_threshold = match_threshold;
    auto matcher_out_dict = m_matcher->descMatch(desc0, desc1);
    return matcher_out_dict;
}


// --------------------------------------------------------------------------------------------------------------
// function
std::vector<cv::DMatch> sg_RANSAC(
    const std::vector<cv::KeyPoint> &vKeys0,
    const std::vector<cv::KeyPoint> &vKeys1,
    const std::vector<cv::DMatch> &vMatches01,
    int min_number
) {
    // RANSAC匹配过程

    int ptCount = vMatches01.size();

    // if(ptCount < 100){
    // if(ptCount < 50){
    if(ptCount < min_number){
        std::cout << "Don't find enough match points" << std::endl;
        return vMatches01;
    }

    // // 坐标转换为float类型
    std::vector<cv::KeyPoint> RAN_kp0_v, RAN_kp1_v;
    // bool tensor_is_first = true;

    // vMatches01还保存着原来未缩小的 vKeys对应关键点的索引,
    for(size_t i=0; i< vMatches01.size(); i++) {
        RAN_kp0_v.push_back(vKeys0[vMatches01[i].queryIdx]);
        RAN_kp1_v.push_back(vKeys1[vMatches01[i].trainIdx]);
    }

    // // 坐标变换
    std::vector<cv::Point2f> p0_v, p1_v;
    for (size_t i = 0; i < vMatches01.size(); i++) {
        p0_v.push_back(RAN_kp0_v[i].pt);
        p1_v.push_back(RAN_kp1_v[i].pt);
    }

    // // 求基础矩阵 Fundamental,3*3的基础矩阵
    std::vector<uchar> RansacStatus;
    cv::Mat Fundametal = cv::findFundamentalMat(p0_v, p1_v, RansacStatus, cv::FM_RANSAC);

    // // 重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
    std::vector<cv::DMatch> RR_matchers_v;
    for(size_t i = 0; i < vMatches01.size(); i++) {
        if(RansacStatus[i] != 0) {
            RR_matchers_v.push_back(vMatches01[i]);
        }
    }

    return RR_matchers_v;
}

// --------------------------------------------------------------------------------------------------------------
// function
cv::Mat convertDesTensor2cvMat(torch::Tensor des_tensor) {
    torch::Tensor desc = des_tensor.to(torch::kCPU).contiguous();
    cv::Mat des_mat(cv::Size(desc.size(2), desc.size(1)), CV_32FC1, desc.data_ptr<float>());
    return des_mat.clone();
}

// --------------------------------------------------------------------------------------------------------------
// function
torch::Tensor convertDescvMat2Tensor(cv::Mat des_mat) {
    // torch::Tensor des_tensor = torch::from_blob(des_mat.clone().data, {des_mat.channels(), des_mat.rows, des_mat.cols}, torch::kFloat);
    torch::Tensor des_tensor = torch::from_blob(des_mat.clone().data, {des_mat.channels(), des_mat.rows, des_mat.cols}, torch::kFloat).contiguous();
    return des_tensor.clone();
}

// --------------------------------------------------------------------------------------------------------------
// function
std::vector<float> convertDesTensor2Vector(torch::Tensor des_tensor) {
    // std::cout << "1.0" << std::endl;
    std::vector<float> des_vector(des_tensor.data_ptr<float>(), des_tensor.data_ptr<float>()+des_tensor.numel());
    // std::cout << "1.1" << std::endl;
    return des_vector;
}

torch::Tensor convertDesVector2Tensor(const std::vector<float> &des_vector, int64_t b, int64_t d, int64_t n) {
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat);
    // c10::IntArrayRef s={b, d, n};
    // torch::Tensor des_tensor = torch::from_blob(des_tensor.data(), {b, d, n}, torch::kFloat);
    auto des = des_vector;
    torch::Tensor des_tensor = torch::from_blob(des.data(), {int64_t(des_vector.size())}, opts);
    des_tensor = des_tensor.reshape({b, d, n});
    return des_tensor.clone();
}

}  // namespace sg_slam 
