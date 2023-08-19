#include "sg_slam/superglue.h"
#include <tuple>
#include <math.h>

namespace sg_slam {
// N.B.: 加前缀表示superglue(SG_)专用模块.
// Keypoint encoder
// ==============================
torch::nn::Sequential SG_mlp(std::vector<int> &channels, bool do_bn) {
    torch::nn::Sequential layers;
    size_t n = channels.size();
    for(size_t i=1; i<n; i++)
    {
        auto conv1d = torch::nn::Conv1d(SG_conv1d_options(channels[i-1], channels[i], 1, 1, 0, true));
        if(i == (n-1)) {
            torch::nn::init::constant_(conv1d->bias, 0.0);
        }
        
        layers->push_back(conv1d);
        if(i < (n-1)) {
            if(do_bn) {
                layers->push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(channels[i])));
            }
            layers->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(false)));
        }
    }

    return layers;
}

torch::Tensor SG_normalize_keypoints(torch::Tensor kpts, c10::IntArrayRef image_shape) {
    // Normalize keypoints locations based on image image_shape
    auto height = image_shape[2];
    auto width = image_shape[3];
    auto one = torch::tensor(1.0).to(kpts.device()); // N.B.: 没有找到与python对应的函数, 故根据功能实现.
    auto size = torch::stack({one*width, one*height}, 0).unsqueeze(0); // (x/w, y/h)
    auto center = size / 2.0;

    auto scaling = std::get<0>(size.max(1, true)) * 0.7;
    center = center.unsqueeze(1);
    scaling = scaling.unsqueeze(1);

    /**
    std::cout << "+++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "one: " << one.sizes() << std::endl;
    std::cout << "one: " << one<< std::endl;
    std::cout << "kpts: " << kpts.sizes() << std::endl;
    std::cout << "size: " << size.sizes() << std::endl;
    std::cout << "size: " << size<< std::endl;
    std::cout << "center: " << center.sizes() << std::endl;
    std::cout << "center: " << center<< std::endl;
    std::cout << "scaling: " << scaling.sizes() << std::endl;
    std::cout << "scaling: " << scaling<< std::endl;
    std::cout << "+++++++++++++++++++++++++++++++++++" << std::endl;
    **/

    return (kpts - center) / scaling;
}

SG_KeypointEncoderImpl::SG_KeypointEncoderImpl(int feature_dim, std::vector<int> layers) {
    layers.insert(layers.begin(), 3);
    layers.push_back(feature_dim);
    encoder_ = SG_mlp(layers, true); // N.B.: 与python的实现略有不同，将最后一层的bias初始化写在了SG_mlp函数内部.

    encoder_ = register_module("encoder", encoder_);
    
}

torch::Tensor SG_KeypointEncoderImpl::forward(torch::Tensor kpts, torch::Tensor scores) {
    auto kpts_T = kpts.transpose(1, 2);
    auto scores_un = scores.unsqueeze(1);
    auto cat = torch::cat({kpts_T, scores_un}, 1);

    // std::cout << "kpts_T: " << kpts_T.sizes() << std::endl;
    // std::cout << "scores_un: " << scores_un.sizes() << std::endl;
    // std::cout << "cat: " << cat.sizes() << std::endl;
    return encoder_->forward(cat);
}


// GNN (transformer)/attention
// ==============================
std::vector<torch::Tensor> SG_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    int64_t dim = query.size(1);
    torch::Tensor scores = torch::einsum("bdhn,bdhm->bhnm", {query, key}) / pow(dim, 0.5);
    auto softmax_opt = torch::nn::functional::SoftmaxFuncOptions(-1);
    torch::Tensor prob = torch::nn::functional::softmax(scores, softmax_opt);

    /**
    std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"dim: "<<dim<<std::endl;
    std::cout<<"query: "<<query.sizes()<<std::endl;
    std::cout<<"key: "<<key.sizes()<<std::endl;
    std::cout<<"scores: "<<scores.sizes()<<std::endl;
    std::cout<<"prob: "<<prob.sizes()<<std::endl;
    std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
    **/
    
    auto weight_value = torch::einsum("bhnm,bdhm->bdhn", {prob, value});

    std::vector<torch::Tensor> out_list{weight_value, prob};
    return out_list;
}

SG_MultiHeadedAttentionImpl::SG_MultiHeadedAttentionImpl(int num_heads, int d_model) {
    m_dim_ = d_model / num_heads;
    m_num_heads_ = num_heads;

    // std::cout << "d_model: " << d_model << std::endl;
    m_merge_ = torch::nn::Conv1d(SG_conv1d_options(d_model, d_model, 1, 1, 0, true));

    torch::nn::ModuleList temp_proj;
    for(int i=0; i<3; i++) {
        temp_proj->push_back(torch::nn::Conv1d(SG_conv1d_options(d_model, d_model, 1, 1, 0, true)));
    }
    m_proj_ = temp_proj;
    m_merge_ = register_module("merge", m_merge_);
    m_proj_ = register_module("proj", m_proj_);
}

torch::Tensor SG_MultiHeadedAttentionImpl::forward(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    int64_t batch_dim = query.size(0);
    query = m_proj_[0]->as<torch::nn::Conv1dImpl>()->forward(query).view({batch_dim, m_dim_, m_num_heads_, -1});
    key = m_proj_[1]->as<torch::nn::Conv1dImpl>()->forward(key).view({batch_dim, m_dim_, m_num_heads_, -1});
    value = m_proj_[2]->as<torch::nn::Conv1dImpl>()->forward(value).view({batch_dim, m_dim_, m_num_heads_, -1});

    auto temp_outlist = SG_attention(query, key, value);
    torch::Tensor x = temp_outlist[0];
    auto out = m_merge_->forward(x.contiguous().view({batch_dim, m_dim_*m_num_heads_, -1}));

    return out;
}

SG_AttentionalPropagationImpl::SG_AttentionalPropagationImpl(int feature_dim, int num_heads) {
    std::vector<int> in_layers{feature_dim*2, feature_dim*2, feature_dim};

    SG_MultiHeadedAttention temp_attn(num_heads, feature_dim);
    m_attn_ = temp_attn;
    m_mlp_ = SG_mlp(in_layers, true);

    m_attn_ = register_module("attn", m_attn_);
    m_mlp_ = register_module("mlp", m_mlp_);
    
}

torch::Tensor SG_AttentionalPropagationImpl::forward(torch::Tensor x, torch::Tensor source) {
    // std::cout<<"***********1*************"<<std::endl;
    torch::Tensor message = m_attn_->forward(x, source, source);
    // std::cout<<"***********2*************"<<std::endl;
    torch::Tensor cat = torch::cat({x, message}, 1);
    // std::cout<<"***********3*************"<<std::endl;
    // std::cout<<"cat: "<< cat.sizes() << std::endl;
    torch::Tensor out = m_mlp_->forward(cat);
    // std::cout<<"***********4*************"<<std::endl;

    return out;
}

SG_AttentionalGNNImpl::SG_AttentionalGNNImpl(int feature_dim, std::vector<int> &layer_names) {
    // std::cout<< "********0*******" << std::endl;
    size_t n = layer_names.size();
    // std::cout<< "********0.1*******" << std::endl;
    torch::nn::ModuleList temp_layers;
    for(size_t i=0; i<n; i++) {
        temp_layers->push_back(SG_AttentionalPropagation(feature_dim, 4));
    }
    // std::cout<< "********0.2*******" << std::endl;
    // temp_layers[0]->as<SG_AttentionalPropagationImpl>()->forward()
    m_layers_ = temp_layers;
    // std::cout<< "********0.3*******" << std::endl;
    m_layers_ = register_module("layers", m_layers_);
    // std::cout<< "********0.4*******" << std::endl;
    m_layers_names_ = layer_names;
    // std::cout<< "********0.5*******" << std::endl;
}

std::vector<torch::Tensor> SG_AttentionalGNNImpl::forward(torch::Tensor desc0, torch::Tensor desc1) {
    // std::string Cross_str = "cross";
    size_t n = m_layers_->size();

    torch::Tensor src0 = desc0;
    torch::Tensor src1 = desc1;
    for(size_t i=0; i<n; i++) 
    {
        if(m_layers_names_[i] == 2) {
            src0 = desc1;
            src1 = desc0;
            // std::string Cross_str = "cross";
            // std::cout << Cross_str << std::endl;
        } else {
            src0 = desc0;
            src1 = desc1;
            // std::string Self_str = "self";
            // std::cout << Self_str << std::endl;
        }

        torch::Tensor delta0 = m_layers_[i]->as<SG_AttentionalPropagationImpl>()->forward(desc0, src0);
        torch::Tensor delta1 = m_layers_[i]->as<SG_AttentionalPropagationImpl>()->forward(desc1, src1);

        desc0 = desc0 + delta0;
        desc1 = desc1 + delta1;
    }

    std::vector<torch::Tensor> out_list{desc0, desc1};

    return out_list;
}


// Optimal transport
// ============================== 
torch::Tensor SG_log_sinkhorn_iterations(torch::Tensor Z, torch::Tensor log_mu, torch::Tensor log_nu, int iters) {
    torch::Tensor u = torch::zeros_like(log_mu);
    torch::Tensor v = torch::zeros_like(log_nu);

    for(int i=0; i<iters; i++) {
        u = log_mu - torch::logsumexp(Z + v.unsqueeze(1), 2, false);
        v = log_nu - torch::logsumexp(Z + u.unsqueeze(2), 1, false);
    }

    torch::Tensor out = Z + u.unsqueeze(2) + v.unsqueeze(1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "iters = " << iters << std::endl;
    std::cout << "Z = " << Z.sizes() << std::endl;
    std::cout << "log_mu = " << log_mu.sizes() << std::endl;
    std::cout << "log_nu = " << log_nu.sizes() << std::endl;
    std::cout << "u = " << u.sizes() << std::endl;
    std::cout << "v = " << v.sizes() << std::endl;
    std::cout << "u.unsqueeze(2) = " << u.unsqueeze(2).sizes() << std::endl;
    std::cout << "v.unsqueeze(1) = " << v.unsqueeze(1).sizes() << std::endl;
    std::cout << "out = " << out.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    return out;
}

torch::Tensor SG_log_optimal_transport(torch::Tensor scores, torch::Tensor alpha, int iters) {
    int64_t b = scores.size(0);
    int64_t m = scores.size(1);
    int64_t n = scores.size(2);
    torch::Tensor one = torch::tensor(1.0).to(scores.device()); // N.B.: 没有找到与python对应的函数, 故根据功能实现.
    // auto one = torch::tensor(1.0).to(scores); // N.B.: 没有找到与python对应的函数, 故根据功能实现.
    torch::Tensor ms = (m*one).to(scores);
    torch::Tensor ns = (n*one).to(scores);

    torch::Tensor bins0 = alpha.expand({b, m, 1}, false);
    torch::Tensor bins1 = alpha.expand({b, 1, n}, false);
    alpha = alpha.expand({b, 1, 1}, false);

    torch::Tensor couplings = torch::cat({torch::cat({scores, bins0}, -1), 
                                          torch::cat({bins1, alpha}, -1)}, 1);  // N.B.: [b, m+1, n+1] (加了dustbin score)
    
    torch::Tensor norm = - (ms + ns).log();
    torch::Tensor log_mu = torch::cat({norm.expand(m), ns.log().unsqueeze(0) + norm}, 0);  // N.B.: 是维度为m+1(加了dustbin score)的向量 m-ns
    torch::Tensor log_nu = torch::cat({norm.expand(n), ms.log().unsqueeze(0) + norm}, 0);  // n-ms (N.B.: 交叉了)

    log_mu = log_mu.unsqueeze(0).expand({b, -1}, false);
    log_nu = log_nu.unsqueeze(0).expand({b, -1}, false);

    torch::Tensor Z = SG_log_sinkhorn_iterations(couplings, log_mu, log_nu, iters);
    Z = Z - norm;  // multiply probabilities by M+N

    return Z; // in: scores.shape = (B, M, N), out: Z.shape = (B, M+1, N+1)
}

torch::Tensor SG_arange_like(torch::Tensor x, int dim) {
    return x.new_ones({x.size(dim)}).cumsum(0) - 1;
}


// SuperGlue Declaration
// ============================== 
SuperGlueImpl::SuperGlueImpl(std::map<std::string, int> &cfg_int, std::map<std::string, std::vector<int>> &cfg_int_v, float match_threshold) {
    m_config_int["descriptor_dim"] = cfg_int["descriptor_dim"];
    m_config_int["sinkhorn_iterations"] = cfg_int["sinkhorn_iterations"];

    m_config_int_v["keypoint_encoder"] = cfg_int_v["keypoint_encoder"];
    m_config_int_v["GNN_layers"] = cfg_int_v["GNN_layers"];

    m_match_threshold = match_threshold;

    m_kenc_ = SG_KeypointEncoder(m_config_int["descriptor_dim"], m_config_int_v["keypoint_encoder"]);
    m_gnn_ = SG_AttentionalGNN(m_config_int["descriptor_dim"], m_config_int_v["GNN_layers"]);
    m_final_proj_ = torch::nn::Conv1d(SG_conv1d_options(m_config_int["descriptor_dim"], m_config_int["descriptor_dim"], 1, 1, 0, true));

    // std::cout << "*******************1**********************" << std::endl;

    // N.B.: 与python不同, 因为没有完全对应的函数
    m_bin_score_ = torch::tensor(1.0);
    m_bin_score_ = register_parameter("bin_score", m_bin_score_);
    // std::cout << "*******************2**********************" << std::endl;
    
    m_kenc_ = register_module("kenc", m_kenc_);
    m_gnn_ = register_module("gnn", m_gnn_);
    m_final_proj_ = register_module("final_proj", m_final_proj_);
    // std::cout << "*******************3**********************" << std::endl;
}

std::vector<torch::Tensor> SuperGlueImpl::forward(std::map<std::string, torch::Tensor> &data) {
    // std::cout << "*******************5.0**********************" << std::endl;
    torch::Tensor desc0 = data["descriptors0"];
    torch::Tensor desc1 = data["descriptors1"];
    torch::Tensor kpts0 = data["keypoints0"];
    torch::Tensor kpts1 = data["keypoints1"];

    // std::cout << "*******************5.1**********************" << std::endl;
    // Keypoint normalization.
    kpts0 = SG_normalize_keypoints(kpts0, data["image0"].sizes());
    kpts1 = SG_normalize_keypoints(kpts1, data["image1"].sizes());

    // std::cout << "*******************5.2**********************" << std::endl;
    // Keypoint MLP encoder.
    desc0 = desc0 + m_kenc_->forward(kpts0, data["scores0"]);
    desc1 = desc1 + m_kenc_->forward(kpts1, data["scores1"]);

    // std::cout << "*******************5.3**********************" << std::endl;
    // Multi-layer Transformer network.
    auto out_temp = m_gnn_->forward(desc0, desc1);
    desc0 = out_temp[0];
    desc1 = out_temp[1];

    // std::cout << "*******************5.4**********************" << std::endl;
    // Final MLP projection.
    torch::Tensor mdesc0 = m_final_proj_->forward(desc0);
    torch::Tensor mdesc1 = m_final_proj_->forward(desc1);

    // Compute matching descriptor distance.
    // torch::Tensor scores = torch::einsum("bdhn,bdhm->bhnm", {query, key}) / pow(dim, 0.5);
    int64_t temp_n = mdesc0.size(2);
    int64_t temp_m = mdesc1.size(2);
    torch::Tensor scores = torch::einsum("bdn,bdm->bnm", {mdesc0, mdesc1});
    scores = scores / pow(m_config_int["descriptor_dim"], 0.5);

    // Run the optimal transport.
    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "bin_score: " << m_bin_score_ << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/
    scores = SG_log_optimal_transport(scores, m_bin_score_, m_config_int["sinkhorn_iterations"]);

    // Get the matches with score above "match_threshold".
    torch::Tensor scores_temp = scores.narrow(1, 0, temp_n).narrow(2, 0, temp_m);
    auto max0 = scores_temp.max(2);
    auto max1 = scores_temp.max(1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "scores: " << scores.sizes() << std::endl;
    std::cout << "scores_temp: " << scores_temp.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/


    // auto scaling = std::get<0>(size.max(1, true)) * 0.7;
    torch::Tensor indices0 = std::get<1>(max0);
    torch::Tensor indices1 = std::get<1>(max1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "indices0: " << indices0.sizes() << std::endl;
    std::cout << "indices1: " << indices1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    
    torch::Tensor mutual0 = SG_arange_like(indices0, 1).unsqueeze(0) == indices1.gather(1, indices0);
    torch::Tensor mutual1 = SG_arange_like(indices1, 1).unsqueeze(0) == indices0.gather(1, indices1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "mutual0: " << mutual0.sizes() << std::endl;
    std::cout << "mutual1: " << mutual1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    // torch::Tensor one = torch::tensor(1.0).to(scores.device()); // N.B.: 没有找到与python对应的函数, 故根据功能实现.
    torch::Tensor zero = torch::tensor(0.0).to(scores);
    torch::Tensor mscores0 = torch::where(mutual0, std::get<0>(max0).exp(), zero);
    torch::Tensor mscores1 = torch::where(mutual1, mscores0.gather(1, indices1), zero);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "mscores0: " << mscores0.sizes() << std::endl;
    std::cout << "mscores1: " << mscores1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    torch::Tensor valid0 = mutual0 & (mscores0 > m_match_threshold);
    torch::Tensor valid1 = mutual1 & valid0.gather(1, indices1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "valid0: " << valid0.sizes() << std::endl;
    std::cout << "valid1: " << valid1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    indices0 = torch::where(valid0, indices0, torch::tensor(-1.0).to(indices0));
    indices1 = torch::where(valid1, indices1, torch::tensor(-1.0).to(indices1));

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "indices0: " << indices0.sizes() << std::endl;
    std::cout << "indices1: " << indices1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    std::vector<torch::Tensor> out_list{indices0, indices1, mscores0, mscores1};

    return out_list;
}

std::map<std::string, torch::Tensor> SuperGlueImpl::pre_forward(std::map<std::string, torch::Tensor> &data) {
    torch::Tensor kpts0 = data["keypoints0"];
    torch::Tensor kpts1 = data["keypoints1"];

    if((kpts0.size(1) == 0) || (kpts1.size(1) == 0)) {
        int64_t shape00 = kpts0.size(0);
        int64_t shape01 = kpts0.size(1);
        int64_t shape10 = kpts1.size(0);
        int64_t shape11 = kpts1.size(1);

        std::map<std::string, torch::Tensor> out_map{
            {"matches0", kpts0.new_full({shape00, shape01}, -1, torch::TensorOptions(torch::kInt))},
            {"matches1", kpts1.new_full({shape10, shape11}, -1, torch::TensorOptions(torch::kInt))},
            {"matching_scores0", kpts0.new_zeros({shape00, shape01})},
            {"matching_scores1", kpts1.new_zeros({shape10, shape11})}
        };

        return out_map;
    }

    auto out_list = forward(data);
    torch::Tensor indices0 = out_list[0];
    torch::Tensor indices1 = out_list[1];
    torch::Tensor mscores0 = out_list[2];
    torch::Tensor mscores1 = out_list[3];

    std::map<std::string, torch::Tensor> out_map{
            {"matches0", indices0},  // use -1 for invalid match
            {"matches1", indices1},  // use -1 for invalid match
            {"matching_scores0", mscores0},
            {"matching_scores1", mscores1}
    };

    return out_map;
}

torch::Tensor SuperGlueImpl::normalize_keypoints_encoder(torch::Tensor kpts, c10::IntArrayRef image_shape, torch::Tensor desc, torch::Tensor scores) {
    torch::Tensor kpts_normal = SG_normalize_keypoints(kpts, image_shape);

    // std::cout << "*******************5.2**********************" << std::endl;
    // Keypoint MLP encoder.
    torch::Tensor desc_encoder = desc + m_kenc_->forward(kpts_normal, scores);

    return desc_encoder;
}

std::map<std::string, torch::Tensor> SuperGlueImpl::descMatch(torch::Tensor desc0, torch::Tensor desc1) {
    // Multi-layer Transformer network.
    auto out_temp = m_gnn_->forward(desc0, desc1);
    desc0 = out_temp[0];
    desc1 = out_temp[1];

    // std::cout << "*******************5.4**********************" << std::endl;
    // Final MLP projection.
    torch::Tensor mdesc0 = m_final_proj_->forward(desc0);
    torch::Tensor mdesc1 = m_final_proj_->forward(desc1);

    // Compute matching descriptor distance.
    // torch::Tensor scores = torch::einsum("bdhn,bdhm->bhnm", {query, key}) / pow(dim, 0.5);
    int64_t temp_n = mdesc0.size(2);
    int64_t temp_m = mdesc1.size(2);
    torch::Tensor scores = torch::einsum("bdn,bdm->bnm", {mdesc0, mdesc1});
    scores = scores / pow(m_config_int["descriptor_dim"], 0.5);

    // Run the optimal transport.
    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "bin_score: " << m_bin_score_ << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/
    scores = SG_log_optimal_transport(scores, m_bin_score_, m_config_int["sinkhorn_iterations"]);

    // Get the matches with score above "match_threshold".
    torch::Tensor scores_temp = scores.narrow(1, 0, temp_n).narrow(2, 0, temp_m);
    auto max0 = scores_temp.max(2);
    auto max1 = scores_temp.max(1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "scores: " << scores.sizes() << std::endl;
    std::cout << "scores_temp: " << scores_temp.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/


    // auto scaling = std::get<0>(size.max(1, true)) * 0.7;
    torch::Tensor indices0 = std::get<1>(max0);
    torch::Tensor indices1 = std::get<1>(max1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "indices0: " << indices0.sizes() << std::endl;
    std::cout << "indices1: " << indices1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    // N.B.: 这里已经交叉匹配检验过了
    torch::Tensor mutual0 = SG_arange_like(indices0, 1).unsqueeze(0) == indices1.gather(1, indices0);
    torch::Tensor mutual1 = SG_arange_like(indices1, 1).unsqueeze(0) == indices0.gather(1, indices1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "mutual0: " << mutual0.sizes() << std::endl;
    std::cout << "mutual1: " << mutual1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    // torch::Tensor one = torch::tensor(1.0).to(scores.device()); // N.B.: 没有找到与python对应的函数, 故根据功能实现.
    torch::Tensor zero = torch::tensor(0.0).to(scores);
    torch::Tensor mscores0 = torch::where(mutual0, std::get<0>(max0).exp(), zero);
    torch::Tensor mscores1 = torch::where(mutual1, mscores0.gather(1, indices1), zero);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "mscores0: " << mscores0.sizes() << std::endl;
    std::cout << "mscores1: " << mscores1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    torch::Tensor valid0 = mutual0 & (mscores0 > m_match_threshold);
    torch::Tensor valid1 = mutual1 & valid0.gather(1, indices1);

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "valid0: " << valid0 << std::endl;
    std::cout << "valid1: " << valid1 << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "valid0: " << valid0.sizes() << std::endl;
    std::cout << "valid1: " << valid1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    indices0 = torch::where(valid0, indices0, torch::tensor(-1.0).to(indices0));
    indices1 = torch::where(valid1, indices1, torch::tensor(-1.0).to(indices1));

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "indices0: " << indices0.sizes() << std::endl;
    std::cout << "indices1: " << indices1.sizes() << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    /**
    std::vector<torch::Tensor> out_list{indices0, indices1, mscores0, mscores1};

    indices0 = out_list[0];
    indices1 = out_list[1];
    mscores0 = out_list[2];
    mscores1 = out_list[3];
    **/

    /**
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    std::cout << "mscores0: " << mscores0 << std::endl;
    std::cout << "mscores1: " << mscores1 << std::endl;
    std::cout << "+++++++++++++++++++++++++++" << std::endl;
    **/

    std::map<std::string, torch::Tensor> out_map{
            {"matches0", indices0},  // use -1 for invalid match
            {"matches1", indices1},  // use -1 for invalid match
            {"matching_scores0", mscores0},
            {"matching_scores1", mscores1}
    };

    return out_map;
}

std::map<std::string, torch::Tensor> SuperGlueImpl::pre_forward2(std::map<std::string, torch::Tensor> &data) {
    torch::Tensor kpts0 = data["keypoints0"];
    torch::Tensor kpts1 = data["keypoints1"];

    if((kpts0.size(1) == 0) || (kpts1.size(1) == 0)) {
        int64_t shape00 = kpts0.size(0);
        int64_t shape01 = kpts0.size(1);
        int64_t shape10 = kpts1.size(0);
        int64_t shape11 = kpts1.size(1);

        std::map<std::string, torch::Tensor> out_map{
            {"matches0", kpts0.new_full({shape00, shape01}, -1, torch::TensorOptions(torch::kInt))},
            {"matches1", kpts1.new_full({shape10, shape11}, -1, torch::TensorOptions(torch::kInt))},
            {"matching_scores0", kpts0.new_zeros({shape00, shape01})},
            {"matching_scores1", kpts1.new_zeros({shape10, shape11})}
        };

        return out_map;
    }

    torch::Tensor desc0 = data["descriptors0"];
    torch::Tensor desc1 = data["descriptors1"];

    desc0 = normalize_keypoints_encoder(kpts0, data["image0"].sizes(), desc0, data["scores0"]);
    desc1 = normalize_keypoints_encoder(kpts1, data["image1"].sizes(), desc1, data["scores1"]);

    std::map<std::string, torch::Tensor> out_map = descMatch(desc0, desc1);

    return out_map;
}




SuperGlue superglueModel(std::string &weight_file, int gpu_id)
{
    std::map<std::string, int> config_int{
        {"descriptor_dim", 256},
        {"sinkhorn_iterations", 20}
    };

    std::map<std::string, std::vector<int>> config_int_v{
        {"keypoint_encoder", {32, 64, 128, 256}},
        {"GNN_layers", {1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2}}
    };

    float match_threshold = 0.2;

    SuperGlue model = SuperGlue(config_int, config_int_v, match_threshold);
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
