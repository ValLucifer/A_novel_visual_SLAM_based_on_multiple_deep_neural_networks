#include "sg_slam/SG_superpoint.h"
#include <iostream>
#include <tuple>

namespace sg_slam {

// N.B.: 这个nms不是很好，但原论文使用的是这个
// N.B.: scores.shape = [B, H*8, W*8]
torch::Tensor simple_nms(torch::Tensor &scores, int nms_radius) {
    // Fast Non-maximum suppression to remove nearby points
    torch::Tensor zeros = torch::zeros_like(scores);
    torch::Tensor max_mask = torch::zeros_like(scores);
    torch::Tensor new_max_mask = torch::zeros_like(scores);
    torch::Tensor supp_mask = torch::zeros_like(scores);
    torch::Tensor supp_scores = torch::zeros_like(scores);
    // F::MaxPool2dFuncOptions(3).stride(2)
    torch::nn::functional::MaxPool2dFuncOptions maxPool2Opt = torch::nn::functional::MaxPool2dFuncOptions(nms_radius*2+1).stride(1).padding(nms_radius);
    torch::Tensor max_pool = torch::nn::functional::max_pool2d(scores, maxPool2Opt);
    max_mask = scores == max_pool;
    for(int i=0; i<2; i++) {
        max_pool = torch::nn::functional::max_pool2d(max_mask.to(torch::kF32), maxPool2Opt);
        supp_mask = max_mask > 0;
        supp_scores = torch::where(supp_mask, zeros, scores);
        max_pool = torch::nn::functional::max_pool2d(supp_scores, maxPool2Opt);
        new_max_mask = supp_scores == max_pool;
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }

    return torch::where(max_mask, scores, zeros); // 输出与scores形状相同
}

// N.B.: keypoints = [N, 2] 即没有批次B
std::vector<torch::Tensor> remove_borders(torch::Tensor &keypoints, torch::Tensor &scores, int border, int height, int width) {
    torch::Tensor mask_h = (keypoints.select(1, 0) >= border) & (keypoints.select(1, 0) < (height - border));
    torch::Tensor mask_w = (keypoints.select(1, 1) >= border) & (keypoints.select(1, 1) < (width - border));
    auto mask = mask_h & mask_w;
    std::vector<torch::Tensor> out_list{keypoints.index({mask.to(torch::kBool)}), scores.index({mask.to(torch::kBool)})};
    return out_list;
}

std::vector<torch::Tensor> top_k_keypoints(torch::Tensor &keypoints, torch::Tensor &scores, int k) {
    std::vector<torch::Tensor> out_list{keypoints, scores};
    
    if(k >= keypoints.size(0)) {
        k = keypoints.size(0);
    }
    if(k < 0) {
        k = keypoints.size(0);
    }
    
    auto out_top_k = torch::topk(scores, k, 0);
    torch::Tensor scores_topk, indices;
    std::tie(scores_topk, indices) = out_top_k;
    out_list[0] = keypoints.index({indices});
    out_list[1] = scores_topk;
    return out_list;
}

// N.B.: 这里的特征点坐标为(x, y)不是(y, x)且已经float化, 详见使用.
torch::Tensor sample_descriptors(torch::Tensor &keypoints, torch::Tensor &descriptors, int s) {
    // Interpolate descriptors at keypoint locations
    auto des_shape = descriptors.sizes();  // [c, h, w]
    int c = des_shape[0];
    int h = des_shape[1];
    int w = des_shape[2];
    auto keypoints_temp = keypoints - s / 2.0 + 0.5;
    keypoints_temp /= torch::tensor({(w*s - s/2.0 -0.5), (h*s - s/2.0 -0.5)}).to(keypoints_temp.device());
    keypoints_temp = keypoints_temp*2 - 1;
    torch::nn::functional::GridSampleFuncOptions gridSampleOpt = torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true);

    auto descs_temp = torch::nn::functional::grid_sample(descriptors.unsqueeze(0), keypoints_temp.view({1, 1, -1, 2}), gridSampleOpt);
    descs_temp = torch::nn::functional::normalize(descs_temp.reshape({c, -1}), torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)); // 注意不同点 dim(0)不是dim(1)

    return descs_temp;
}


SG_SuperPointImpl::SG_SuperPointImpl(std::map<std::string, int> &cfg, float keypoint_threshold) {
    m_config["descriptor_dim"] = cfg["descriptor_dim"];
    m_config["nms_radius"] = cfg["nms_radius"];
    m_config["max_keypoints"] = cfg["max_keypoints"];
    m_config["remove_borders"] = cfg["remove_borders"];
    m_keypoint_threshold = keypoint_threshold;

    _m_relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));
    _m_pool = torch::nn::MaxPool2d(maxpool_options(2, 2));

    int c1 = 64;
    int c2 = 64;
    int c3 = 128;
    int c4 = 128;
    int c5 = 256;

    conv1a = torch::nn::Conv2d(conv_options(1 , c1, 3, 1, 1));
    conv1b = torch::nn::Conv2d(conv_options(c1, c1, 3, 1, 1));
    conv2a = torch::nn::Conv2d(conv_options(c1, c2, 3, 1, 1));
    conv2b = torch::nn::Conv2d(conv_options(c2, c2, 3, 1, 1));
    conv3a = torch::nn::Conv2d(conv_options(c2, c3, 3, 1, 1));
    conv3b = torch::nn::Conv2d(conv_options(c3, c3, 3, 1, 1));
    conv4a = torch::nn::Conv2d(conv_options(c3, c4, 3, 1, 1));
    conv4b = torch::nn::Conv2d(conv_options(c4, c4, 3, 1, 1));

    convPa = torch::nn::Conv2d(conv_options(c4, c5, 3, 1, 1));
    convPb = torch::nn::Conv2d(conv_options(c5, 65, 1, 1, 0));

    convDa = torch::nn::Conv2d(conv_options(c4, c5, 3, 1, 1));
    convDb = torch::nn::Conv2d(conv_options(c5, m_config["descriptor_dim"], 1, 1, 0));

    conv1a = register_module("conv1a", conv1a);
    conv1b = register_module("conv1b", conv1b);
    conv2a = register_module("conv2a", conv2a);
    conv2b = register_module("conv2b", conv2b);
    conv3a = register_module("conv3a", conv3a);
    conv3b = register_module("conv3b", conv3b);
    conv4a = register_module("conv4a", conv4a);
    conv4b = register_module("conv4b", conv4b);
    convPa = register_module("convPa", convPa);
    convPb = register_module("convPb", convPb);
    convDa = register_module("convDa", convDa);
    convDb = register_module("convDb", convDb);
}

std::vector<torch::Tensor> SG_SuperPointImpl::forward(torch::Tensor x) {
    // Shared Encoder
    x = _m_relu->forward(conv1a->forward(x));
    x = _m_relu->forward(conv1b->forward(x));
    x = _m_pool->forward(x);
    x = _m_relu->forward(conv2a->forward(x));
    x = _m_relu->forward(conv2b->forward(x));
    x = _m_pool->forward(x);
    x = _m_relu->forward(conv3a->forward(x));
    x = _m_relu->forward(conv3b->forward(x));
    x = _m_pool->forward(x);
    x = _m_relu->forward(conv4a->forward(x));
    x = _m_relu->forward(conv4b->forward(x));

    // Compute the dense keypoint scores
    torch::Tensor cPa = _m_relu->forward(convPa->forward(x));
    torch::Tensor scores = convPb->forward(cPa);
    scores = torch::nn::functional::softmax(scores, 1);
    scores = scores.narrow(1, 0, 64);

    torch::Tensor cDa = _m_relu->forward(convDa->forward(x));
    torch::Tensor descriptors = convDb->forward(cDa);
    descriptors = torch::nn::functional::normalize(descriptors, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // std::cout<< "scores: " << scores.sizes() <<std::endl;
    // std::cout<< "descriptors: " << descriptors.sizes() <<std::endl;
    std::vector<torch::Tensor> out_list{scores, descriptors};

    return out_list;
}

std::map<std::string, std::vector<torch::Tensor>> SG_SuperPointImpl::forwardDetect(std::vector<torch::Tensor> &input) {
    auto scores = input[0];
    auto descriptors = input[1]; // N.B.: 可以用引用吗?
    auto scores_shape = scores.sizes();  //  [b, c, h, w]
    // int c = des_shape[0];
    int b = scores_shape[0];
    int h = scores_shape[2];
    int w = scores_shape[3];
    scores = scores.permute({0, 2, 3, 1}).reshape({b, h, w, 8, 8});
    // std::cout<<"scores.shape: "<< scores.sizes() <<std::endl;
    scores = scores.permute({0 , 1, 3, 2, 4}).reshape({b, h*8, w*8});
    // std::cout<<"scores.shape: "<< scores.sizes() <<std::endl;
    scores = simple_nms(scores, m_config["nms_radius"]);
    // std::cout<<"scores.shape: "<< scores.sizes() <<std::endl;
    // std::cout<<"scores B: "<<scores.size(0)<<std::endl;

    // Extract keypoints
    std::vector<torch::Tensor> v_keypoints;
    for(int i=0 ; i < scores.size(0); i++) {
        v_keypoints.push_back(torch::nonzero(scores[i] > m_keypoint_threshold));
    }
    // N.B.: len 对应批次
    // std::cout<<"keypoints len: "<<v_keypoints.size()<<std::endl;
    // std::cout<<"keypoints[0].shape: "<<v_keypoints[0].sizes()<<std::endl;
    // std::cout<<"keypoints[1].shape: "<<v_keypoints[1].sizes()<<std::endl;
    // std::cout<<"keypoints[2].shape: "<<v_keypoints[2].sizes()<<std::endl;
    std::vector<torch::Tensor> v_scores;
    // N.B.: v_keypoints.size()的输出类型为size_t
    for(size_t i=0; i<v_keypoints.size(); i++) {
        v_scores.push_back(
            scores[i].index(
                {v_keypoints[i].index_select(1, torch::tensor({0}).to(m_device)), 
                 v_keypoints[i].index_select(1, torch::tensor({1}).to(m_device))}).squeeze());
    }
    // N.B.: 得到的v_scores的元素比python的一个大小为"1"的维度, 如v_scores[0].shape = [90, 1](c++), [90](python). 故要加.squeeze()使形状一致.

    // Discard keypoints near the image borders
    for(size_t i=0; i<v_keypoints.size(); i++) {
        auto out_list = remove_borders(v_keypoints[i], v_scores[i], m_config["remove_borders"], h*8, w*8);
        v_keypoints[i] = out_list[0];
        v_scores[i] = out_list[1];
    }

    // Keep the k keypoints with highest score
    // N.B.: 此处对top_k_keypoints()函数做了一些改动，为的是保留在所有点全取的情况下，依然保留类排序功能(降序)
    for(size_t i=0; i<v_keypoints.size(); i++) {
        auto out_list = top_k_keypoints(v_keypoints[i], v_scores[i], m_config["max_keypoints"]);
        v_keypoints[i] = out_list[0];
        v_scores[i] = out_list[1];
    }

    // Convert (h, w)/(y, x)/(v, u) to (x, y)
    // std::cout << v_keypoints[0][7] << std::endl;
    for(size_t i=0; i<v_keypoints.size(); i++) {
        v_keypoints[i] = torch::flip(v_keypoints[i], 1).to({torch::kFloat});
    }
    // std::cout << v_keypoints[0][7] << std::endl;
    
    // Extract descriptors
    std::vector<torch::Tensor> v_descriptors;
    for(size_t i=0; i<v_keypoints.size(); i++) {
        auto des = descriptors[i];
        v_descriptors.push_back(sample_descriptors(v_keypoints[i], des, 8));
    }

    std::map<std::string, std::vector<torch::Tensor>> out_dict{
        {"keypoints", v_keypoints},
        {"scores", v_scores},
        {"descriptors", v_descriptors}
    };

    return out_dict;
}

// 详细见 superpoint.h
std::vector<cv::KeyPoint> SG_SuperPointImpl::convertKpts2cvKeypoints(std::vector<torch::Tensor> kpts_v, std::vector<torch::Tensor> scores_v) {
    std::vector<cv::KeyPoint> keypoints_v;
    torch::Tensor kpts = kpts_v[0].to(torch::kCPU);
    torch::Tensor prob = scores_v[0].to(torch::kCPU);

    for(int64_t i=0; i<kpts.size(0); i++) {
        float response = prob[i].item<float>();
        keypoints_v.push_back(cv::KeyPoint(kpts[i][0].item<float>(), kpts[i][1].item<float>(), 8, -1, response));
    }

    return keypoints_v;
}

// 详细见 SG_superpoint.h
cv::Mat SG_SuperPointImpl::convertDes2cvMat(std::vector<torch::Tensor> des_v) {
    torch::Tensor desc = des_v[0].to(torch::kCPU);  // [256, n_keypoints]
    desc = desc.transpose(0, 1).contiguous();  // [n_keypoints, 256]
    // std::cout << desc.sizes() << std::endl;
    // cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data<float>());
    cv::Mat desc_mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr<float>());  // N.B.: desc_mat为[256 x 3617], Des_mat.col(0)为[1, 3617], Des_mat.row(0)为[256 x 1].

    return desc_mat;
}


SG_SuperPoint SG_superpointModel(std::string &weight_file, int gpu_id) 
{
    
    std::map<std::string, int> config{
        {"descriptor_dim", 256},
        {"nms_radius", 4},
        {"max_keypoints", 1000},
        {"remove_borders", 4}
    };
    

    /**
    std::map<std::string, int> config{
        {"descriptor_dim", 256},
        {"nms_radius", 4},
        {"max_keypoints", 2000},
        {"remove_borders", 4}
    };
    **/

    float keypoint_threshold = 0.005;

    SG_SuperPoint model = SG_SuperPoint(config, keypoint_threshold);
    torch::Device device = torch::Device(torch::kCPU);
    if(gpu_id >= 0) {
        device = torch::Device(torch::kCUDA, gpu_id);
    } else {
        device = torch::Device(torch::kCPU);
    }
    model->m_device = device;

    model->to(torch::Device(torch::kCPU));
    torch::load(model, weight_file);
    model->to(device);
    return model;
}

}  // namespace sg_slam 
