#ifndef _SUPERGLUE_H_INCLUDED_
#define _SUPERGLUE_H_INCLUDED_

#include <torch/torch.h>
#include <vector>

namespace sg_slam {

// N.B.: 加前缀表示superglue(SG_)专用模块.
// Keypoint encoder
// ==============================
inline torch::nn::Conv1dOptions SG_conv1d_options(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                               int64_t stride=1, int64_t padding=0, bool with_bias = true) {
    torch::nn::Conv1dOptions conv1d_opt = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size);
    conv1d_opt.stride(stride);
    conv1d_opt.padding(padding);
    conv1d_opt.bias(with_bias);
    return conv1d_opt;
}

torch::nn::Sequential SG_mlp(std::vector<int> &channels, bool do_bn=true);

torch::Tensor SG_normalize_keypoints(torch::Tensor kpts, c10::IntArrayRef image_shape);

class SG_KeypointEncoderImpl final : public torch::nn::Module
{
public:
    SG_KeypointEncoderImpl(int feature_dim, std::vector<int> layers);
    ~SG_KeypointEncoderImpl() = default;

    torch::Tensor forward(torch::Tensor kpts, torch::Tensor scores);

private:
    torch::nn::Sequential encoder_{nullptr};
};
TORCH_MODULE(SG_KeypointEncoder);


// GNN (transformer)/attention
// ==============================
std::vector<torch::Tensor> SG_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value);

class SG_MultiHeadedAttentionImpl final : public torch::nn::Module
{
public:
    SG_MultiHeadedAttentionImpl(int num_heads, int d_model);
    ~SG_MultiHeadedAttentionImpl() = default;

    torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value);
    
private:
    int m_dim_;
    int m_num_heads_;

    torch::nn::Conv1d m_merge_{nullptr};
    torch::nn::ModuleList m_proj_{nullptr};
};
TORCH_MODULE(SG_MultiHeadedAttention);

class SG_AttentionalPropagationImpl final : public torch::nn::Module
{
public:
    SG_AttentionalPropagationImpl(int feature_dim, int num_heads);
    ~SG_AttentionalPropagationImpl() = default;

    torch::Tensor forward(torch::Tensor x, torch::Tensor source);

private:
    SG_MultiHeadedAttention m_attn_{nullptr};
    torch::nn::Sequential m_mlp_{nullptr};
};
TORCH_MODULE(SG_AttentionalPropagation);

class SG_AttentionalGNNImpl final : public torch::nn::Module
{
public:
    // N.B.: 与python不同将layer_names改为std::vector<int>, 1: self, 2: cross
    // SG_AttentionalGNNImpl(int feature_dim, std::vector<std::string> layer_names);
    SG_AttentionalGNNImpl(int feature_dim, std::vector<int> &layer_names);
    ~SG_AttentionalGNNImpl() = default;

    std::vector<torch::Tensor> forward(torch::Tensor desc0, torch::Tensor desc1);

private:
    torch::nn::ModuleList m_layers_{nullptr};
    std::vector<int> m_layers_names_;
};
TORCH_MODULE(SG_AttentionalGNN);


// Optimal transport
// ============================== 
torch::Tensor SG_log_sinkhorn_iterations(torch::Tensor Z, torch::Tensor log_mu, torch::Tensor log_nu, int iters);

torch::Tensor SG_log_optimal_transport(torch::Tensor scores, torch::Tensor alpha, int iters);

torch::Tensor SG_arange_like(torch::Tensor x, int dim);


// SuperGlue Declaration
// ============================== 
class SuperGlueImpl final : public torch::nn::Module
{
public:
    torch::Device m_device = torch::Device(torch::kCPU);
    // N.B.: 与python实现略有不同，将输入的配置参数分为了三部分主要是因为数据类型不同!
    std::map<std::string, int> m_config_int{
        {"descriptor_dim", 256},
        {"sinkhorn_iterations", 100}
    };

    std::map<std::string, std::vector<int>> m_config_int_v{
        {"keypoint_encoder", {32, 64, 128, 256}},
        {"GNN_layers", {1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2}}
    };

    float m_match_threshold = 0.2;

    /**
    static std::map<std::string, torch::Tensor> m_data_test{
        {"image0", torch::tensor(0.0)},
        {"image1", torch::tensor(0.0)},
        {"descriptors0", torch::tensor(0.0)},
        {"descriptors1", torch::tensor(0.0)},
        {"keypoints0", torch::tensor(0.0)},
        {"keypoints1", torch::tensor(0.0)},
        {"scores0", torch::tensor(0.0)},
        {"scores1", torch::tensor(0.0)}
    };
    **/

    SuperGlueImpl(std::map<std::string, int> &cfg_int, std::map<std::string, std::vector<int>> &cfg_int_v, float match_threshold);
    ~SuperGlueImpl() = default;

    std::vector<torch::Tensor> forward(std::map<std::string, torch::Tensor> &data);

    std::map<std::string, torch::Tensor> pre_forward(std::map<std::string, torch::Tensor> &data);

    torch::Tensor normalize_keypoints_encoder(torch::Tensor kpts, c10::IntArrayRef image_shape, torch::Tensor desc, torch::Tensor scores);
    std::map<std::string, torch::Tensor> descMatch(torch::Tensor desc0, torch::Tensor desc1);

    std::map<std::string, torch::Tensor> pre_forward2(std::map<std::string, torch::Tensor> &data);

private:
    torch::Tensor m_bin_score_ = torch::tensor(1.0);  // N.B.: Parameter

    // Declare Module
    SG_KeypointEncoder m_kenc_{nullptr};
    SG_AttentionalGNN m_gnn_{nullptr};

    torch::nn::Conv1d m_final_proj_{nullptr};
};
TORCH_MODULE(SuperGlue);

SuperGlue superglueModel(std::string &weight_file, int gpu_id);

}  // namespace sg_slam 

#endif  // _SUPERGLUE_H_INCLUDED_
