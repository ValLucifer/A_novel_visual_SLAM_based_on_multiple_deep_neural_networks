#ifndef _SUPERPOINT_H_INCLUDED_
#define _SUPERPOINT_H_INCLUDED_

#include <torch/torch.h>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace sg_slam {

inline torch::nn::Conv2dOptions conv_options(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
                                             int64_t stride=1, int64_t padding=0, bool with_bias = true) {
    torch::nn::Conv2dOptions conv_opt = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size);
    conv_opt.stride(stride);
    conv_opt.padding(padding);
    conv_opt.bias(with_bias);
    return conv_opt;
}

inline torch::nn::MaxPool2dOptions maxpool_options(int64_t kernel_size, int64_t stride) {
    torch::nn::MaxPool2dOptions maxpool_opt(kernel_size);
    maxpool_opt.stride(stride);
    return maxpool_opt;
}

// SuperPoint Declaration
class SG_SuperPointImpl final : public torch::nn::Module
{
public:
    torch::Device m_device = torch::Device(torch::kCPU);
    std::map<std::string, int> m_config{
        {"descriptor_dim", 256},
        {"nms_radius", 4},
        {"max_keypoints", -1},
        {"remove_borders", 4}
    };
    float m_keypoint_threshold = 0.005;

    SG_SuperPointImpl(std::map<std::string, int> &cfg, float keypoint_threshold);
    ~SG_SuperPointImpl() = default;

    std::vector<torch::Tensor> forward(torch::Tensor x);

    std::map<std::string, std::vector<torch::Tensor>> forwardDetect(std::vector<torch::Tensor> &input);

    // N.B.: 仅适用于B(batch) = 1 的情况, 其他batch情况未考虑.
    std::vector<cv::KeyPoint> convertKpts2cvKeypoints(std::vector<torch::Tensor> kpts_v, std::vector<torch::Tensor> scores_v);

    // N.B.: 仅适用于B(batch) = 1 的情况, 其他batch情况未考虑.
    cv::Mat convertDes2cvMat(std::vector<torch::Tensor> des_v);

private:
    // Declare layers
    torch::nn::Conv2d conv1a{nullptr};
    torch::nn::Conv2d conv1b{nullptr};
    torch::nn::Conv2d conv2a{nullptr};
    torch::nn::Conv2d conv2b{nullptr};
    torch::nn::Conv2d conv3a{nullptr};
    torch::nn::Conv2d conv3b{nullptr};
    torch::nn::Conv2d conv4a{nullptr};
    torch::nn::Conv2d conv4b{nullptr};
    torch::nn::Conv2d convPa{nullptr};
    torch::nn::Conv2d convPb{nullptr};
    torch::nn::Conv2d convDa{nullptr};
    torch::nn::Conv2d convDb{nullptr};

    torch::nn::MaxPool2d _m_pool{nullptr};
    torch::nn::ReLU _m_relu{nullptr};
};
TORCH_MODULE(SG_SuperPoint);

SG_SuperPoint SG_superpointModel(std::string &weight_file, int gpu_id);

torch::Tensor simple_nms(torch::Tensor &scores, int nms_radius);
std::vector<torch::Tensor> remove_borders(torch::Tensor &keypoints, torch::Tensor &scores, int border, int height, int width);
std::vector<torch::Tensor> top_k_keypoints(torch::Tensor &keypoints, torch::Tensor &scores, int k);

}  // namespace sg_slam 

#endif  // _SUPERPOINT_H_INCLUDED_
