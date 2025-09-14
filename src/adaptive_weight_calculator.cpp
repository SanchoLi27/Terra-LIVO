/*
 * TERRA-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry
 * Copyright (C) 2025 SanchoLi27 <hdalhd1104@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "adaptive_weight_calculator.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <chrono>

AdaptiveWeightCalculator::AdaptiveWeightCalculator()
    : initialized_(false)
    , last_calculation_time_(0.0)
{
    // 使用默认参数初始化
    params_ = CalculationParams();
    reset();
}

AdaptiveWeightCalculator::AdaptiveWeightCalculator(const CalculationParams& params)
    : params_(params)
    , initialized_(false)
    , last_calculation_time_(0.0)
{
    reset();
}

AdaptiveWeightCalculator::~AdaptiveWeightCalculator()
{
}

void AdaptiveWeightCalculator::setCalculationParams(const CalculationParams& params)
{
    params_ = params;
}

void AdaptiveWeightCalculator::reset()
{
    while (!weight_history_.empty()) {
        weight_history_.pop();
    }
    
    stats_ = Statistics();
    initialized_ = false;
    last_weights_ = SensorFusionWeights();
    last_calculation_time_ = 0.0;
    
    if (debug_info_.enable_debug) {
        debug_info_.debug_log.clear();
        logDebugInfo("AdaptiveWeightCalculator reset");
    }
}

// =============================================================================
// 主要权重计算接口
// =============================================================================

SensorFusionWeights AdaptiveWeightCalculator::calculateAdaptiveWeights(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality,
    WeightStrategy strategy)
{
    // 获取当前时间戳
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
    
    SensorFusionWeights weights;
    weights.timestamp = timestamp;
    
    // 根据策略选择计算方法
    switch (strategy) {
        case QUALITY_BASED:
            weights = calculateQualityBasedWeights(lidar_quality, visual_quality, imu_quality);
            break;
        case CONFIDENCE_BASED:
            weights = calculateConfidenceBasedWeights(lidar_quality, visual_quality, imu_quality);
            break;
        case HYBRID:
            weights = calculateHybridWeights(lidar_quality, visual_quality, imu_quality);
            break;
        case CONSERVATIVE:
            weights = calculateConservativeWeights(lidar_quality, visual_quality, imu_quality);
            break;
        default:
            weights = calculateHybridWeights(lidar_quality, visual_quality, imu_quality);
            break;
    }
    
    // 应用约束条件
    weights = applyConstraints(weights);
    
    // 应用时间平滑
    weights = applyTemporalSmoothing(weights);
    
    // 应用稳定性控制
    weights = applyStabilityControl(weights);
    
    // 确保权重有效性
    if (!weights.isValid()) {
        weights.normalize();
    }
    
    // 计算总体置信度
    weights.total_confidence = calculateTotalConfidence(lidar_quality, visual_quality, imu_quality);
    
    // 检测异常值
    if (params_.enable_outlier_rejection && isOutlier(weights)) {
        if (debug_info_.enable_debug) {
            logDebugInfo("Outlier detected, using previous weights");
        }
        weights = last_weights_;
    }
    
    // 更新统计信息
    updateStatistics(weights);
    
    // 更新权重历史
    weight_history_.push(weights);
    if (weight_history_.size() > MAX_HISTORY_SIZE) {
        weight_history_.pop();
    }
    
    // 调用回调函数
    if (weight_change_callback_) {
        weight_change_callback_(weights);
    }
    
    // 更新内部状态
    last_weights_ = weights;
    last_calculation_time_ = timestamp;
    initialized_ = true;
    
    if (debug_info_.enable_debug) {
        char debug_msg[256];
        snprintf(debug_msg, sizeof(debug_msg), 
                "Calculated weights: L=%.3f, V=%.3f, I=%.3f, Conf=%.3f",
                weights.lidar_weight, weights.visual_weight, weights.imu_weight, weights.total_confidence);
        logDebugInfo(debug_msg);
    }
    
    return weights;
}

// =============================================================================
// 不同策略的权重计算实现
// =============================================================================

SensorFusionWeights AdaptiveWeightCalculator::calculateQualityBasedWeights(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    SensorFusionWeights weights;
    
    // 直接基于质量评分计算权重
    double raw_lidar_weight = qualityToWeight(lidar_quality.overall_score, params_.quality_sensitivity);
    double raw_visual_weight = qualityToWeight(visual_quality.overall_score, params_.quality_sensitivity);
    double raw_imu_weight = qualityToWeight(imu_quality.overall_score, params_.quality_sensitivity);
    
    // 相对质量调整
    double total_quality = lidar_quality.overall_score + visual_quality.overall_score + imu_quality.overall_score;
    if (total_quality > 1e-6) {
        raw_lidar_weight *= (lidar_quality.overall_score / total_quality) * 3.0;
        raw_visual_weight *= (visual_quality.overall_score / total_quality) * 3.0;
        raw_imu_weight *= (imu_quality.overall_score / total_quality) * 3.0;
    }
    
    weights.lidar_weight = raw_lidar_weight;
    weights.visual_weight = raw_visual_weight;
    weights.imu_weight = raw_imu_weight;
    
    return weights;
}

SensorFusionWeights AdaptiveWeightCalculator::calculateConfidenceBasedWeights(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    SensorFusionWeights weights;
    
    // 基于置信度的权重计算，考虑数据的不确定性
    double lidar_confidence = sigmoid(lidar_quality.overall_score, params_.confidence_alpha, 0.5);
    double visual_confidence = sigmoid(visual_quality.overall_score, params_.confidence_alpha, 0.5);
    double imu_confidence = sigmoid(imu_quality.overall_score, params_.confidence_alpha, 0.5);
    
    // 基于置信度计算权重
    weights.lidar_weight = params_.confidence_beta + (1.0 - params_.confidence_beta) * lidar_confidence;
    weights.visual_weight = params_.confidence_beta + (1.0 - params_.confidence_beta) * visual_confidence;
    weights.imu_weight = params_.confidence_beta + (1.0 - params_.confidence_beta) * imu_confidence;
    
    return weights;
}

SensorFusionWeights AdaptiveWeightCalculator::calculateHybridWeights(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    // 混合策略：结合质量评估和置信度
    SensorFusionWeights quality_weights = calculateQualityBasedWeights(lidar_quality, visual_quality, imu_quality);
    SensorFusionWeights confidence_weights = calculateConfidenceBasedWeights(lidar_quality, visual_quality, imu_quality);
    
    SensorFusionWeights weights;
    
    // 动态调整混合比例，基于整体质量水平
    double overall_quality = (lidar_quality.overall_score + visual_quality.overall_score + imu_quality.overall_score) / 3.0;
    double alpha = 0.5 + 0.3 * overall_quality; // 质量越高，越依赖质量评估
    
    weights.lidar_weight = alpha * quality_weights.lidar_weight + (1.0 - alpha) * confidence_weights.lidar_weight;
    weights.visual_weight = alpha * quality_weights.visual_weight + (1.0 - alpha) * confidence_weights.visual_weight;
    weights.imu_weight = alpha * quality_weights.imu_weight + (1.0 - alpha) * confidence_weights.imu_weight;
    
    // 应用质量相关的连续调整函数，而非离散的场景判断
    weights = applyQualityBasedAdjustments(weights, lidar_quality, visual_quality, imu_quality);
    
    return weights;
}

SensorFusionWeights AdaptiveWeightCalculator::calculateConservativeWeights(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    SensorFusionWeights weights;
    
    if (!initialized_) {
        // 初始情况使用默认权重
        weights.lidar_weight = 0.4;
        weights.visual_weight = 0.4;
        weights.imu_weight = 0.2;
    } else {
        // 基于上一次权重进行小幅调整
        SensorFusionWeights target_weights = calculateHybridWeights(lidar_quality, visual_quality, imu_quality);
        
        double smoothing = 0.95; // 高平滑因子，变化更保守
        weights.lidar_weight = smoothing * last_weights_.lidar_weight + (1.0 - smoothing) * target_weights.lidar_weight;
        weights.visual_weight = smoothing * last_weights_.visual_weight + (1.0 - smoothing) * target_weights.visual_weight;
        weights.imu_weight = smoothing * last_weights_.imu_weight + (1.0 - smoothing) * target_weights.imu_weight;
    }
    
    return weights;
}

// =============================================================================
// 权重处理和约束方法
// =============================================================================

SensorFusionWeights AdaptiveWeightCalculator::applyConstraints(const SensorFusionWeights& weights) const
{
    SensorFusionWeights constrained = weights;
    
    // 应用最小和最大权重限制
    constrained.lidar_weight = std::max(params_.min_weight_threshold, 
                                       std::min(params_.max_weight_threshold, constrained.lidar_weight));
    constrained.visual_weight = std::max(params_.min_weight_threshold, 
                                        std::min(params_.max_weight_threshold, constrained.visual_weight));
    constrained.imu_weight = std::max(params_.imu_base_weight, 
                                     std::min(params_.imu_max_weight, constrained.imu_weight));
    
    // 确保IMU始终有基础权重
    if (constrained.imu_weight < params_.imu_base_weight) {
        constrained.imu_weight = params_.imu_base_weight;
    }
    
    return constrained;
}

SensorFusionWeights AdaptiveWeightCalculator::applyTemporalSmoothing(const SensorFusionWeights& weights)
{
    if (!initialized_) {
        return weights; // 第一次计算不进行平滑
    }
    
    SensorFusionWeights smoothed = weights;
    
    // 指数平滑
    double alpha = 1.0 - params_.smoothing_factor;
    smoothed.lidar_weight = exponentialSmoothing(weights.lidar_weight, last_weights_.lidar_weight, alpha);
    smoothed.visual_weight = exponentialSmoothing(weights.visual_weight, last_weights_.visual_weight, alpha);
    smoothed.imu_weight = exponentialSmoothing(weights.imu_weight, last_weights_.imu_weight, alpha);
    
    return smoothed;
}

SensorFusionWeights AdaptiveWeightCalculator::applyStabilityControl(const SensorFusionWeights& weights)
{
    if (!initialized_) {
        return weights;
    }
    
    SensorFusionWeights controlled = weights;
    
    // 限制权重变化率
    double lidar_change = std::abs(weights.lidar_weight - last_weights_.lidar_weight);
    double visual_change = std::abs(weights.visual_weight - last_weights_.visual_weight);
    double imu_change = std::abs(weights.imu_weight - last_weights_.imu_weight);
    
    if (lidar_change > params_.max_weight_change_rate) {
        double direction = (weights.lidar_weight > last_weights_.lidar_weight) ? 1.0 : -1.0;
        controlled.lidar_weight = last_weights_.lidar_weight + direction * params_.max_weight_change_rate;
    }
    
    if (visual_change > params_.max_weight_change_rate) {
        double direction = (weights.visual_weight > last_weights_.visual_weight) ? 1.0 : -1.0;
        controlled.visual_weight = last_weights_.visual_weight + direction * params_.max_weight_change_rate;
    }
    
    if (imu_change > params_.max_weight_change_rate) {
        double direction = (weights.imu_weight > last_weights_.imu_weight) ? 1.0 : -1.0;
        controlled.imu_weight = last_weights_.imu_weight + direction * params_.max_weight_change_rate;
    }
    
    // 记录变化率
    controlled.lidar_change_rate = lidar_change;
    controlled.visual_change_rate = visual_change;
    controlled.imu_change_rate = imu_change;
    
    return controlled;
}

SensorFusionWeights AdaptiveWeightCalculator::applyQualityBasedAdjustments(
    const SensorFusionWeights& weights,
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    SensorFusionWeights adjusted = weights;
    
    // 使用连续函数进行质量调整，而非离散的场景判断
    
    // 1. LiDAR质量调整因子：基于质量分数的连续函数
    double lidar_adjustment = sigmoid(lidar_quality.overall_score, 4.0, 0.5); // 陡峭的sigmoid
    
    // 2. 视觉质量调整因子
    double visual_adjustment = sigmoid(visual_quality.overall_score, 4.0, 0.5);
    
    // 3. IMU质量调整因子
    double imu_adjustment = sigmoid(imu_quality.overall_score, 3.0, 0.5);
    
    // 4. 应用相对质量重新分配
    double total_raw_quality = lidar_quality.overall_score + visual_quality.overall_score + imu_quality.overall_score;
    if (total_raw_quality > 1e-6) {
        // 基于相对质量进行权重重新分配
        double lidar_relative_quality = lidar_quality.overall_score / total_raw_quality;
        double visual_relative_quality = visual_quality.overall_score / total_raw_quality;
        double imu_relative_quality = imu_quality.overall_score / total_raw_quality;
        
        // 平滑的质量相关调整
        adjusted.lidar_weight = adjusted.lidar_weight * (0.7 + 0.3 * lidar_relative_quality);
        adjusted.visual_weight = adjusted.visual_weight * (0.7 + 0.3 * visual_relative_quality);
        adjusted.imu_weight = adjusted.imu_weight * (0.7 + 0.3 * imu_relative_quality);
    }
    
    // 5. 应用质量置信度调整
    double quality_confidence = calculateTotalConfidence(lidar_quality, visual_quality, imu_quality);
    double confidence_factor = sigmoid(quality_confidence, 2.0, 0.6);
    
    // 当整体质量置信度低时，增加IMU权重作为稳定基准
    if (quality_confidence < 0.5) {
        double imu_boost = (0.5 - quality_confidence) * 0.5; // 最多增加0.25的权重
        adjusted.imu_weight += imu_boost;
    }
    
    return adjusted;
}

// =============================================================================
// 工具函数实现
// =============================================================================

double AdaptiveWeightCalculator::qualityToWeight(double quality_score, double sensitivity)
{
    // 使用sigmoid函数将质量评分映射到权重
    return sigmoid(quality_score, sensitivity, 0.5);
}

double AdaptiveWeightCalculator::calculateTotalConfidence(
    const LidarQualityMetrics& lidar_quality,
    const VisualQualityMetrics& visual_quality,
    const ImuQualityMetrics& imu_quality)
{
    // 计算综合置信度
    double weighted_confidence = 0.4 * lidar_quality.overall_score +
                                0.4 * visual_quality.overall_score +
                                0.2 * imu_quality.overall_score;
    
    // 考虑数据一致性
    double quality_variance = 0.0;
    double mean_quality = (lidar_quality.overall_score + visual_quality.overall_score + imu_quality.overall_score) / 3.0;
    quality_variance += (lidar_quality.overall_score - mean_quality) * (lidar_quality.overall_score - mean_quality);
    quality_variance += (visual_quality.overall_score - mean_quality) * (visual_quality.overall_score - mean_quality);
    quality_variance += (imu_quality.overall_score - mean_quality) * (imu_quality.overall_score - mean_quality);
    quality_variance /= 3.0;
    
    // 方差越小，置信度越高
    double consistency_factor = 1.0 / (1.0 + quality_variance * 10.0);
    
    return weighted_confidence * consistency_factor;
}

bool AdaptiveWeightCalculator::isOutlier(const SensorFusionWeights& weights) const
{
    if (stats_.sample_count < 5) {
        return false; // 样本不足时不进行异常检测
    }
    
    // 基于统计信息检测异常值
    double lidar_z_score = std::abs(weights.lidar_weight - stats_.mean_lidar_weight) / 
                          (stats_.std_lidar_weight + 1e-6);
    double visual_z_score = std::abs(weights.visual_weight - stats_.mean_visual_weight) / 
                           (stats_.std_visual_weight + 1e-6);
    double imu_z_score = std::abs(weights.imu_weight - stats_.mean_imu_weight) / 
                        (stats_.std_imu_weight + 1e-6);
    
    // 如果任何一个传感器的权重偏离超过阈值，认为是异常值
    return (lidar_z_score > params_.outlier_detection_threshold ||
            visual_z_score > params_.outlier_detection_threshold ||
            imu_z_score > params_.outlier_detection_threshold);
}

void AdaptiveWeightCalculator::updateStatistics(const SensorFusionWeights& weights)
{
    // 更新均值（增量更新）
    double alpha = 1.0 / (stats_.sample_count + 1);
    
    double old_mean_lidar = stats_.mean_lidar_weight;
    double old_mean_visual = stats_.mean_visual_weight;
    double old_mean_imu = stats_.mean_imu_weight;
    
    stats_.mean_lidar_weight += alpha * (weights.lidar_weight - stats_.mean_lidar_weight);
    stats_.mean_visual_weight += alpha * (weights.visual_weight - stats_.mean_visual_weight);
    stats_.mean_imu_weight += alpha * (weights.imu_weight - stats_.mean_imu_weight);
    
    // 更新标准差（增量更新）
    if (stats_.sample_count > 0) {
        double lidar_diff = weights.lidar_weight - old_mean_lidar;
        double visual_diff = weights.visual_weight - old_mean_visual;
        double imu_diff = weights.imu_weight - old_mean_imu;
        
        stats_.std_lidar_weight = std::sqrt((stats_.sample_count * stats_.std_lidar_weight * stats_.std_lidar_weight + 
                                           lidar_diff * lidar_diff) / (stats_.sample_count + 1));
        stats_.std_visual_weight = std::sqrt((stats_.sample_count * stats_.std_visual_weight * stats_.std_visual_weight + 
                                            visual_diff * visual_diff) / (stats_.sample_count + 1));
        stats_.std_imu_weight = std::sqrt((stats_.sample_count * stats_.std_imu_weight * stats_.std_imu_weight + 
                                         imu_diff * imu_diff) / (stats_.sample_count + 1));
    }
    
    stats_.sample_count++;
}

double AdaptiveWeightCalculator::sigmoid(double x, double alpha, double center)
{
    return 1.0 / (1.0 + std::exp(-alpha * (x - center)));
}

double AdaptiveWeightCalculator::exponentialSmoothing(double current, double previous, double alpha)
{
    return alpha * current + (1.0 - alpha) * previous;
}

Eigen::Vector3d AdaptiveWeightCalculator::normalizeWeights(const Eigen::Vector3d& weights)
{
    double sum = weights.sum();
    if (sum > 1e-6) {
        return weights / sum;
    } else {
        return Eigen::Vector3d(1.0/3.0, 1.0/3.0, 1.0/3.0);
    }
}

// =============================================================================
// 查询和监控接口
// =============================================================================

std::vector<SensorFusionWeights> AdaptiveWeightCalculator::getWeightHistory() const
{
    std::vector<SensorFusionWeights> history;
    std::queue<SensorFusionWeights> temp_queue = weight_history_;
    
    while (!temp_queue.empty()) {
        history.push_back(temp_queue.front());
        temp_queue.pop();
    }
    
    return history;
}

double AdaptiveWeightCalculator::getWeightStability() const
{
    if (weight_history_.size() < 5) {
        return 0.5; // 数据不足时返回中等稳定性
    }
    
    std::vector<SensorFusionWeights> history = getWeightHistory();
    
    // 计算权重变化的方差
    double mean_lidar_change = 0.0, mean_visual_change = 0.0, mean_imu_change = 0.0;
    int change_count = 0;
    
    for (size_t i = 1; i < history.size(); ++i) {
        mean_lidar_change += std::abs(history[i].lidar_weight - history[i-1].lidar_weight);
        mean_visual_change += std::abs(history[i].visual_weight - history[i-1].visual_weight);
        mean_imu_change += std::abs(history[i].imu_weight - history[i-1].imu_weight);
        change_count++;
    }
    
    if (change_count > 0) {
        mean_lidar_change /= change_count;
        mean_visual_change /= change_count;
        mean_imu_change /= change_count;
    }
    
    double total_change = mean_lidar_change + mean_visual_change + mean_imu_change;
    
    // 变化越小，稳定性越高
    return 1.0 / (1.0 + total_change * 20.0);
}

SensorFusionWeights AdaptiveWeightCalculator::predictNextWeights(double prediction_horizon) const
{
    if (!initialized_ || weight_history_.size() < 3) {
        return last_weights_; // 数据不足时返回当前权重
    }
    
    // 简单的线性预测
    std::vector<SensorFusionWeights> history = getWeightHistory();
    
    // 计算最近的变化趋势
    SensorFusionWeights trend;
    trend.lidar_weight = (history.back().lidar_weight - history[history.size()-2].lidar_weight);
    trend.visual_weight = (history.back().visual_weight - history[history.size()-2].visual_weight);
    trend.imu_weight = (history.back().imu_weight - history[history.size()-2].imu_weight);
    
    // 预测权重
    SensorFusionWeights predicted = last_weights_;
    predicted.lidar_weight += trend.lidar_weight * prediction_horizon;
    predicted.visual_weight += trend.visual_weight * prediction_horizon;
    predicted.imu_weight += trend.imu_weight * prediction_horizon;
    
    // 应用约束
    predicted = applyConstraints(predicted);
    predicted.normalize();
    
    return predicted;
}

void AdaptiveWeightCalculator::setWeightChangeCallback(std::function<void(const SensorFusionWeights&)> callback)
{
    weight_change_callback_ = callback;
}

// =============================================================================
// 调试接口
// =============================================================================

void AdaptiveWeightCalculator::enableDebugMode(bool enable, const std::string& output_path)
{
    debug_info_.enable_debug = enable;
    debug_info_.debug_output_path = output_path;
    if (enable) {
        logDebugInfo("Debug mode enabled");
    }
}

void AdaptiveWeightCalculator::logDebugInfo(const std::string& info)
{
    if (debug_info_.enable_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        auto timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        
        char timestamped_info[512];
        snprintf(timestamped_info, sizeof(timestamped_info), "[%.6f] %s", timestamp, info.c_str());
        
        debug_info_.debug_log.push_back(std::string(timestamped_info));
        
        // 限制日志大小
        if (debug_info_.debug_log.size() > 1000) {
            debug_info_.debug_log.erase(debug_info_.debug_log.begin());
        }
    }
}
