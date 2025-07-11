"""
Train a diffusion model on images with Random Memory Assignment (Optimized Version).
"""

import os
import time
import types
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.transforms as transforms
import cv2
import csv

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from NetworkModels.MemoryUnitFramework_ import *
from NetworkModels.MemoryUnitGraphFramework_ import *
from NetworkModels.Balance_TeacherStudent_NoMPI_ import *
from NetworkModels.Teacher_Model_NoMPI_ import *
from NetworkModels.TFCL_TeacherStudent_ import *
from NetworkModels.DynamicDiffusionMixture_ import *

from Task_Split.Task_utilizes import *
from cv2_imageProcess import *
from datasets.Data_Loading import *
from datasets.Fid_evaluation import *
from Task_Split.TaskFree_Split import *
from datasets.MNIST32 import *

import torch.distributions as td
from torch.distributions.multivariate_normal import MultivariateNormal

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

# 全局阈值参数
MAX_CLUSTERS = int(os.getenv('MAX_CLUSTERS', '20'))
MAX_SAMPLES_PER_CLUSTER = int(os.getenv('MAX_SAMPLES_PER_CLUSTER', '100'))
VERBOSE = bool(int(os.getenv('VERBOSE', '0')))  # 控制详细输出
TRAIN_FREQUENCY = int(os.getenv('TRAIN_FREQUENCY', '10'))  # 每N步训练一次模型

def Transfer_To_Numpy(sample):
    """将张量转换为numpy数组格式（优化版）"""
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous()
    # 避免不必要的GPU/CPU传输
    if sample.is_cuda:
        sample = sample.cpu()
    return sample.numpy()

def Save_Image(name, image):
    """保存图片"""
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/" + name, image)

def add_fast_random_clustering_support():
    """
    为记忆集群框架添加快速随机聚类支持
    """
    
    def fast_random_add_data_batch(self, x1):
        """
        快速随机聚类数据批处理函数
        
        优化点：
        1. 减少print语句
        2. 批量处理样本
        3. 预计算可用集群
        """
        
        myarrScore = []
        n = np.shape(x1)[0]
        
        if VERBOSE:
            print(f"\n开始处理批次，包含 {n} 个样本")
        
        # 预计算可用集群信息（避免重复计算）
        def get_available_clusters():
            available = []
            for j, cluster in enumerate(self.MemoryClusterArr):
                if cluster.GiveCount() < cluster.maxMemorySize:
                    available.append(j)
            return available
        
        for i in range(n):
            data1 = x1[i]
            
            # 如果没有集群，创建第一个
            if len(self.MemoryClusterArr) == 0:
                if VERBOSE:
                    print("创建第一个集群")
                arr = [data1]
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                        self.input_size, self, "Random")
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                myarrScore.append(0.0)
                continue
            
            # 获取可用集群
            available_clusters = get_available_clusters()
            
            # 情况1: 有可用的集群空间
            if available_clusters:
                selected_cluster_idx = random.choice(available_clusters)
                self.MemoryClusterArr[selected_cluster_idx].AddSingleSample(data1)
                myarrScore.append(random.random() * 0.5)
                
            # 情况2: 所有集群都满了
            else:
                if len(self.MemoryClusterArr) < self.MaxMemoryCluster:
                    # 简化决策：直接创建新集群
                    if random.random() < 0.3:  # 30%概率创建新集群
                        arr = [data1]
                        newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                                self.input_size, self, "Random")
                        self.MemoryClusterArr.append(newMemory)
                        self.memoryUnits = self.memoryUnits + 1
                        myarrScore.append(0.6)
                    else:
                        # 随机替换现有集群中的样本
                        selected_cluster_idx = random.randint(0, len(self.MemoryClusterArr) - 1)
                        cluster = self.MemoryClusterArr[selected_cluster_idx]
                        if len(cluster.arr) > 0:
                            replace_idx = random.randint(0, len(cluster.arr) - 1)
                            cluster.arr[replace_idx] = data1
                        myarrScore.append(0.7)
                else:
                    # 达到最大集群数量，直接随机替换
                    selected_cluster_idx = random.randint(0, len(self.MemoryClusterArr) - 1)
                    cluster = self.MemoryClusterArr[selected_cluster_idx]
                    if len(cluster.arr) > 0:
                        replace_idx = random.randint(0, len(cluster.arr) - 1)
                        cluster.arr[replace_idx] = data1
                    myarrScore.append(0.8)
        
        # 批量计算完成后打印信息
        if VERBOSE and n > 0:
            print(f"批次处理完成，当前集群数: {len(self.MemoryClusterArr)}/{self.MaxMemoryCluster}")
        
        return np.max(myarrScore) if len(myarrScore) > 0 else 0
    
    return fast_random_add_data_batch

def add_fast_random_sample_support():
    """
    为MemoryCluster类添加快速随机样本添加支持
    """
    
    def fast_random_distance_calculation(self, sample1, sample2):
        """快速返回随机距离值"""
        return random.random()
    
    # 为MemoryCluster类添加新方法
    MemoryCluster.fast_random_distance_calculation = fast_random_distance_calculation
    
    # 覆盖距离计算方法
    original_method = MemoryCluster.CalculateWDistance_Individual
    
    def new_distance_method(self, sample1, sample2):
        if self.distance_type == "Random":
            return self.fast_random_distance_calculation(sample1, sample2)
        else:
            return original_method(self, sample1, sample2)
            
    MemoryCluster.CalculateWDistance_Individual = new_distance_method

def Calculate_JS(TSFramework, batch, batchReco):
    """计算JS散度（快速版本）"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    miniBatch = 64
    
    # 避免重复的reshape操作
    if batch.ndim > 2:
        batch = batch.reshape(batch.shape[0], -1)
    if batchReco.ndim > 2:
        batchReco = batchReco.reshape(batchReco.shape[0], -1)
    
    # 使用更高效的标准差创建方法
    std = torch.full_like(batch, 0.01)
    
    t = 100
    diffusion = TSFramework.teacherArray[0].diffusion
    schedule_sampler = UniformSampler(diffusion)
    times, weights = schedule_sampler.sample(batch.shape[0], dist_util.dev())
    times.fill_(t)  # 更高效的填充
    
    beta = _extract_into_tensor(TSFramework.teacherArray[0].diffusion.sqrt_alphas_cumprod, times, batch.shape)
    
    batch = batch * beta
    batchReco = batchReco * beta
    
    q_z1 = td.normal.Normal(batch, std)
    q_z2 = td.normal.Normal(batchReco, std)
    score11 = td.kl_divergence(q_z1, q_z2).mean()
    score12 = td.kl_divergence(q_z2, q_z1).mean()
    
    score = (score11 + score12) / (2.0 * miniBatch)
    return score

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def main():
    """主函数（优化版）"""
    # 设置基本参数
    dataNmae = "mnist"
    modelName = "GraphMemory_Random_Fast"
    distanceType = "Random"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    dataStramX, dataStramY, totalTestX, totalSetY = Give_DataStream_Supervised(dataNmae)
    
    defaultTest = totalTestX
    miniBatch = 64
    totalTrainingTime = int(dataStramX.shape[0] / miniBatch)
    
    inputSize = 32
    
    # 创建框架
    print("创建框架...")
    start = time.time()
    TSFramework = MemoryUnitGraphFramework("RandomMemory", device, inputSize)
    
    # 设置参数
    TSFramework.MaxMemoryCluster = MAX_CLUSTERS
    TSFramework.maxSizeForEachMemory = MAX_SAMPLES_PER_CLUSTER
    TSFramework.distance_type = distanceType
    TSFramework.OriginalInputSize = 32
    TSFramework.batch_size = 64
    
    # 应用快速随机聚类支持
    add_fast_random_sample_support()
    fast_clustering_method = add_fast_random_clustering_support()
    TSFramework.AddDataBatch = types.MethodType(fast_clustering_method, TSFramework)
    
    # 准备测试数据
    test_data = torch.tensor(defaultTest).to(device, dtype=torch.float)
    
    # 创建第一个组件
    newComponent = TSFramework.Create_NewComponent()
    TSFramework.currentComponent = newComponent
    initial_batch = dataStramX[0:miniBatch]
    newComponent.memoryBuffer = initial_batch
    
    TSFramework.currentMemory = initial_batch
    TSFramework.maxMemorySize = 2000
    
    # 训练参数
    threshold = 30
    TSFramework.threshold = threshold
    train_epoch = int(os.getenv('TRAIN_EPOCHS', '2'))  # 减少训练epoch数以提高速度
    
    # 训练统计
    currentClass = 1
    componentArr = []
    classArr = []
    arr = []
    
    # 优化数据转换：避免不必要的GPU/CPU传输
    print("转换数据格式...")
    if dataStramX.is_cuda:
        dataStramX = dataStramX.cpu()
    dataStramX = dataStramX.numpy()
    
    print("开始训练...")
    print(f"总训练步数: {totalTrainingTime}")
    print(f"最大集群数: {MAX_CLUSTERS}")
    print(f"每个集群最大样本数: {MAX_SAMPLES_PER_CLUSTER}")
    print(f"训练epoch数: {train_epoch}")
    
    # 主训练循环（优化版）
    for step in range(totalTrainingTime):
        batch = dataStramX[step*miniBatch:(step + 1)*miniBatch]
        
        if len(TSFramework.MemoryClusterArr) == 0:
            TSFramework.MemoryBegin(batch)
        
        # 优化标签处理
        y = dataStramY[step*miniBatch:(step + 1)*miniBatch]
        if y.is_cuda:
            y = y.cpu()
        y_numpy = y.numpy()
        
        # 更新类别统计
        maxin = np.max(y_numpy)
        if maxin > currentClass:
            currentClass = maxin
        
        classArr.append(currentClass)
        componentArr.append(len(TSFramework.teacherArray))
        
        # 减少打印频率
        if step % 10 == 0 or step == totalTrainingTime - 1:
            print(f"步骤 {step+1}/{totalTrainingTime}, 集群数: {len(TSFramework.MemoryClusterArr)}, 当前类别: {currentClass}")
        
        # 每步都训练模型（这是算法要求）
        # 获取记忆化样本并优化内存使用
        memoryBuffer = TSFramework.GiveMemorizedSamples()
        
        # 优化：避免重复的tensor创建
        if not isinstance(memoryBuffer, torch.Tensor):
            memoryBuffer = torch.tensor(memoryBuffer, device=device, dtype=torch.float)
        else:
            memoryBuffer = memoryBuffer.to(device, dtype=torch.float)
        
        batch_tensor = torch.tensor(batch, device=device, dtype=torch.float)
        memoryBuffer_ = torch.cat([memoryBuffer, batch_tensor], 0)
        
        # 训练模型
        TSFramework.currentComponent.Train(train_epoch, memoryBuffer_)
        
        # 随机添加数据批次
        expansion_score = TSFramework.AddDataBatch(batch)
        arr.append(expansion_score)
        
        # 清理临时变量和GPU内存
        del memoryBuffer, batch_tensor, memoryBuffer_
        if step % 20 == 0:  # 更频繁的内存清理
            torch.cuda.empty_cache()
    
    # 计算训练时间
    end = time.time()
    training_time = end - start
    print(f"训练完成！训练时间: {training_time:.2f} 秒")
    
    # 打印记忆信息
    TSFramework.PrintMemoryInformation()
    
    # 保存训练结果（批量操作）
    print("保存训练结果...")
    os.makedirs("results", exist_ok=True)
    
    # 保存所有统计数据
    stats_data = {
        "scores": arr,
        "classes": classArr,
        "components": componentArr
    }
    
    for name, data in stats_data.items():
        filename = f"results/MNIST_{name}_Random_Fast_{threshold}.txt"
        with open(filename, "w") as f:
            for item in data:
                f.write(f"{item}\n")
    
    # 生成和保存样本
    print("生成样本...")
    try:
        gen = TSFramework.Give_GenerationFromTeacher(100)
        generatedImages = Transfer_To_Numpy(gen)
        name_generation = f"{dataNmae}_{modelName}_{threshold}.png"
        Save_Image(name_generation, merge2(generatedImages[0:64], [8, 8]))
    except Exception as e:
        print(f"生成样本时出错: {e}")
    
    # 重建测试
    print("重建测试...")
    try:
        batch = test_data[0:64]
        reco = TSFramework.student.Give_ReconstructionSingle(batch)
        
        # 保存真实和重建图像
        realBatch = Transfer_To_Numpy(batch)
        reco_numpy = Transfer_To_Numpy(reco)
        
        name_real = f"{dataNmae}_{modelName}_Real_{threshold}.png"
        name_reco = f"{dataNmae}_{modelName}_Reco_{threshold}.png"
        
        Save_Image(name_real, merge2(realBatch, [8, 8]))
        Save_Image(name_reco, merge2(reco_numpy, [8, 8]))
    except Exception as e:
        print(f"重建测试时出错: {e}")
    
    # FID评估
    print("FID评估...")
    try:
        generated = TSFramework.Give_GenerationFromTeacher(1000)
        mytest = test_data[0:generated.shape[0]]
        fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
        print(f"FID分数: {fid1:.4f}")
    except Exception as e:
        print(f"FID评估时出错: {e}")
        fid1 = -1
    
    # 保存模型
    print("保存模型...")
    try:
        myModelName = f"{modelName}_{threshold}_{dataNmae}.pkl"
        os.makedirs("data", exist_ok=True)
        torch.save(TSFramework, f'./data/{myModelName}')
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # 分析随机聚类结果
    print("\n随机聚类分析:")
    total_samples = 0
    cluster_info = []
    
    for i, cluster in enumerate(TSFramework.MemoryClusterArr):
        cluster_size = cluster.GiveCount()
        total_samples += cluster_size
        cluster_info.append((i, cluster_size))
    
    # 批量打印集群信息
    for i, size in cluster_info:
        print(f"  集群 {i}: {size} 个样本")
    
    avg_samples = total_samples/len(TSFramework.MemoryClusterArr) if len(TSFramework.MemoryClusterArr) > 0 else 0
    print(f"总样本数: {total_samples}")
    print(f"平均每个集群样本数: {avg_samples:.2f}")
    
    # 保存验证指标
    print("保存验证指标...")
    try:
        os.makedirs("metrics", exist_ok=True)
        metrics = {
            "dataset": dataNmae,
            "model": modelName,
            "threshold": threshold,
            "memory_size": TSFramework.GiveMemorySize(),
            "memory_clusters": len(TSFramework.MemoryClusterArr),
            "distance_type": distanceType,
            "clustering_method": "Random_Fast",
            "fid_generation": fid1,
            "training_time": training_time,
            "total_samples": total_samples,
            "avg_samples_per_cluster": avg_samples,
            "train_epochs": train_epoch
        }
        
        filename = f"metrics/{modelName}_metrics.csv"
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='') as file:
            fieldnames = list(metrics.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        print(f"验证指标已保存到 {filename}")
    except Exception as e:
        print(f"保存验证指标时出错: {e}")
    
    print("所有任务完成！")
    print(f"总耗时: {training_time:.2f} 秒")
    
    # 清理GPU内存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 