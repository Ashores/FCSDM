"""
Train a diffusion model on images.
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import types  # Add this import for method binding

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import  structural_similarity
from NetworkModels.MemoryUnitFramework_ import *
from NetworkModels.MemoryUnitGraphFramework_ import *

#
import argparse
import torch
from datasets.MyCIFAR10 import *
from NetworkModels.Balance_TeacherStudent_NoMPI_ import *
from NetworkModels.Teacher_Model_NoMPI_ import *

from Task_Split.Task_utilizes import *
import cv2
from cv2_imageProcess import *
from datasets.Data_Loading import *
from datasets.Fid_evaluation import *
from Task_Split.TaskFree_Split import *
from datasets.MNIST32 import *
import torchvision.transforms as transforms
import torch.utils.data as Data
from NetworkModels.TFCL_TeacherStudent_ import *
from NetworkModels.DynamicDiffusionMixture_ import *

# 导入CLIP模型相关库
import clip
from PIL import Image
import torch.nn.functional as F

# 导入因果关系相关库
import numpy as np
import random

# 导入高级因果分析相关库
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#dad
import numpy as np

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
#
import torch.nn.functional as F         # 函数包

import torch.distributions as td
from torch.distributions.multivariate_normal import MultivariateNormal

# 全局CLIP模型变量
clip_model = None
clip_preprocess = None

def initialize_clip_model(device):
    """初始化CLIP模型"""
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()  # 设置为评估模式
    return clip_model, clip_preprocess

#
def Transfer_To_Numpy(sample):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    mySamples = sample.unsqueeze(0).cuda().cpu()
    mySamples = np.array(mySamples)
    mySamples = mySamples[0]
    return mySamples

def Save_Image(name,image):
    cv2.imwrite("results/" + name, image)
    #cv2.waitKey(0)

def TransferNumpyToTensor(totalSetX,device):
    newSet = []
    for i in range(np.shape(totalSetX)[0]):
        arr1 = totalSetX[i]
        arr1 = torch.tensor(arr1).cuda().to(device=device, dtype=torch.float)
        newSet.append(arr1)
    return newSet

def RandomSelectionArr(memory,newXList):
    for i in range(np.shape(newXList)[0]):
        memory = RandomSelection(memory,newXList[i])
    return memory

def RandomSelection(memory,newX):
    N = np.shape(memory)[0] + 1
    j = int(random.random() * N)
    if j > 0 and j < N-2:
        memory[j] = newX
    return memory

def GiveMSE(data,reco):
    mse = nn.functional.mse_loss(data,reco)
    mse.unsqueeze(0).cuda().cpu()
    return mse
#

def Calculate_JS(TSFramework,batch,batchReco):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    miniBatch = 64

    batch = batch.reshape(np.shape(batch)[0],32*32*3)
    batchReco = batchReco.reshape(np.shape(batchReco)[0],32*32*3)
    std = np.zeros((np.shape(batch)))
    std[:,:] = 0.01
    std = torch.tensor(std).cuda().to(device=device, dtype=torch.float)

    t = 100
    diffusion = TSFramework.teacherArray[0].diffusion
    schedule_sampler = UniformSampler(diffusion)
    times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())
    for i in range(np.shape(times)[0]):
        times[i] = t

    beta = _extract_into_tensor(TSFramework.teacherArray[0].diffusion.sqrt_alphas_cumprod, times, batch.shape)

    batch = batch * beta
    batchReco = batchReco * beta

    q_z1 = td.normal.Normal(batch, std)
    q_z2 = td.normal.Normal(batchReco, std)
    score11 = td.kl_divergence(q_z1, q_z2).mean()
    score12 = td.kl_divergence(q_z2, q_z1).mean()

    score11 = score11 / miniBatch
    score12 = score12 / miniBatch
    score = (score11 + score12) / 2.0
    return score

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas
#
def q_x(x_0, t, noise=None):
    num_steps = 100
    betas = make_beta_schedule(schedule='sigmoid', n_timesteps=num_steps, start=1e-5, end=0.5e-2)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod = alphas_prod.to(x_0.device)
    #alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

#
def Calculate_ExpansionScore(TSFramework,batch):
    arr = []
    #t = torch.tensor([50])
    t= 50
    for i in range(np.shape(TSFramework.teacherArray)[0]):
        currentComponent = TSFramework.teacherArray[i]

        buffer = currentComponent.memoryBuffer
        #reco1 = currentComponent.q_sample(batch,t) #q_x(buffer,t)
        #reco2 = currentComponent.q_sample(buffer,t)#q_x(batch,t)
        reco1 = batch
        reco2 = buffer

        score = Calculate_JS(TSFramework,reco1,reco2)
        score = score.cpu().detach().numpy()
        #score = score[0]
        arr.append(score)

    #arr = arr.cpu().numpy()
    arr = np.array(arr)
    maxScore = np.min(arr)
    index = np.argmin(arr)
    return maxScore,index

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def LoadModel():
    dataNmae = "mnist"
    modelName = "GraphMemory"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataStramX, dataStramY, totalTestX, totalSetY = Give_DataStream_Supervised(dataNmae)
    #dist_util.setup_dist()

    defaultTest = totalTestX
    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)

    threshold = 2000
    myModelName = modelName + str(threshold) + "_" + dataNmae + ".pkl"
    TSFramework = torch.load('./data/' + myModelName)

    # Evaluation
    print("Generation")
    generated = TSFramework.Give_GenerationFromTeacher(1000)
    mytest = test_data[0:np.shape(generated)[0]]
    fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
    print(fid1)

#
def main():
    #
    dataNmae = "cifar10"
    modelName = "GraphMemory"
    distanceType = "CLIP"  # 更改为使用CLIP特征
    modelName = modelName + "_" + distanceType

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化CLIP模型
    initialize_clip_model(device)

    dataStramX, dataStramY, totalTestX, totalSetY = Give_DataStream_Supervised(dataNmae)
    defaultTest = totalTestX

    miniBatch = 64
    totalTrainingTime = int(dataStramX.shape[0] / miniBatch)

    inputSize = 32
    epoch = 1

    Tepoch = 100
    Sepoch = 100

    start = time.time()
    inputSize = 32
    TSFramework = MemoryUnitGraphFramework("myName",device,inputSize)
    TSFramework.distance_type = distanceType  # 设置为CLIP
    TSFramework.MaxMemoryCluster = 20
    TSFramework.OriginalInputSize = 32
    TSFramework.batch_size = 24

    # 应用CLIP和因果关系支持
    add_clip_support_to_memory_cluster()
    
    # 添加软聚类支持 - 确保在使用相关功能前调用
    add_soft_clustering_support()

    # 应用性能优化
    optimized_clustering_method = optimize_memory_framework()
    TSFramework.AddDataBatch = types.MethodType(optimized_clustering_method, TSFramework)

    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)
    batch = test_data[0:64]
    batch2 = dataStramX[0:64]

    #build the first one
    newComponent = TSFramework.Create_NewComponent()
    TSFramework.currentComponent = newComponent
    batch = dataStramX[0:miniBatch]
    newComponent.memoryBuffer = batch

    TSFramework.currentMemory = batch
    TSFramework.maxMemorySize = 2000

    memoryBuffer = []
    maxMemorySize = 2000
    maxMemorySizeDefault = 2000

    dataloader = []

    threshold = 30

    TSFramework.threshold = threshold
    epoch = 6
    runCount = 0

    currentValue = 0
    arr = []
    currentClass = 1
    componentArr = []
    classArr = []
    runStep = 0

    dataStramX = dataStramX.unsqueeze(0).cuda().cpu()
    dataStramX = np.array(dataStramX)
    dataStramX = dataStramX[0]

    for step in range(totalTrainingTime):
        batch = dataStramX[step*miniBatch:(step + 1)*miniBatch]

        if np.shape(TSFramework.MemoryClusterArr)[0] == 0:
            TSFramework.MemoryBegin(batch)

        y = dataStramY[step*miniBatch:(step + 1)*miniBatch]
        batch_cpu = y.unsqueeze(0).cuda().cpu()
        batch_cpu = np.array(batch_cpu)
        runStep = runStep + 1

        #TSFramework.currentMemory = batch

        maxin = np.max(batch_cpu)
        if maxin > currentClass:
            currentClass = maxin

        classArr.append(currentClass)
        componentArr.append(np.shape(TSFramework.teacherArray)[0])

        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
              .format(step, totalTrainingTime, np.shape(TSFramework.MemoryClusterArr)[0], 0, 1, 0, 1))

        #TSFramework.currentMemory = TSFramework.SampleSelection(batch)

        memoryBuffer = TSFramework.GiveMemorizedSamples()
        memoryBuffer = torch.tensor(memoryBuffer).cuda().to(device=device, dtype=torch.float)

        batch2 = torch.tensor(batch).cuda().to(device=device, dtype=torch.float)

        memoryBuffer_ = torch.cat([memoryBuffer,batch2],0)

        TSFramework.currentComponent.Train(epoch,memoryBuffer_)

        TSFramework.AddDataBatch(batch)

        arr.append(currentValue)

    #Knolwedge transfer
    KD_epoch = 10
    #TSFramework.KnowledgeTransferForStudent(KD_epoch,memoryBuffer2)

    print("information")
    TSFramework.PrintMemoryInformation()

    arr1 = np.array(arr).astype('str')
    myThirdName = "results/ScoreCurve_MNIST" + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    arr1 = np.array(classArr).astype('str')
    myThirdName = "results/MNIST_Class"  + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    arr1 = np.array(componentArr).astype('str')
    myThirdName = "results/MNIST_Component" + "_" + str(threshold) + ".txt"
    #myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
    f = open(myThirdName, "w", encoding="utf-8")
    for i in range(np.shape(arr1)[0]):
        f.writelines(arr1[i])
        f.writelines('\n')
    f.flush()
    f.close()

    end = time.time()
    print("Training times")
    print((end - start))
    print("Finish the training")

    gen = TSFramework.Give_GenerationFromTeacher(100)
    generatedImages = Transfer_To_Numpy(gen)
    name_generation = dataNmae + "_" + modelName + str(threshold) + ".png"
    Save_Image(name_generation,merge2(generatedImages[0:64], [8, 8]))

    #Evaluation
    test_data = torch.tensor(defaultTest).cuda().to(device=device, dtype=torch.float)

    batch = test_data[0:64]
    reco = TSFramework.student.Give_ReconstructionSingle(batch)
    myReco = Transfer_To_Numpy(reco)
    #myReco = merge2(myReco, [8, 8])

    realBatch = Transfer_To_Numpy(batch)
    #realBatch = merge2(realBatch, [8, 8])
    name = dataNmae + "_" + modelName + "_" + "Real_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Real_small_" + str(0) + ".png"

    Save_Image(name,merge2(realBatch, [8, 8]))
    Save_Image(name_small,merge2(realBatch[0:16], [2, 8]))

    reco = Transfer_To_Numpy(reco)
    # realBatch = merge2(realBatch, [8, 8])
    name = dataNmae + "_" + modelName + "_" + "Reco_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Reco_small_" + str(0) + ".png"

    Save_Image(name, merge2(reco, [8, 8]))
    Save_Image(name_small, merge2(reco[0:16], [2, 8]))

    # 只保留FID评估    print("\nFID Evaluation:")
    generated = TSFramework.Give_GenerationFromTeacher(1000)
    mytest = test_data[0:np.shape(generated)[0]]
    fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
    print(f"FID Score: {fid1:.4f}")
    
    # 保存模型
    myModelName = modelName + str(threshold) + "_" + dataNmae + ".pkl"
    torch.save(TSFramework, './data/' + myModelName)

    # 添加记录每个集群的因果特征统计信息的功能
    print("记录因果特征统计信息:")
    for i, cluster in enumerate(TSFramework.MemoryClusterArr):
        print(f"集群 {i}:")
        if hasattr(cluster, 'causal_graph'):
            # 分析该集群的样本，查看各个因果特征组的分布情况
            samples_features = []
            for sample in cluster.arr[:min(10, len(cluster.arr))]:  # 只取前10个样本分析
                sample_tensor = torch.tensor(sample).to(device)
                feature = extract_clip_features(sample_tensor.unsqueeze(0), device)
                samples_features.append(feature)
            
            if samples_features:
                samples_features = torch.cat(samples_features, dim=0)
                for group, indices in cluster.causal_graph.feature_groups.items():
                    group_features = samples_features[:, indices]
                    group_mean = torch.mean(group_features, dim=0)
                    
                    # 修复标准差计算，防止出现nan
                    if len(samples_features) > 1:
                        group_std = torch.std(group_features, dim=0)
                        std_value = torch.mean(group_std).item()
                    else:
                        std_value = 0.0  # 单个样本无法计算标准差
                        
                    print(f"  {group} 特征组平均值: {torch.mean(group_mean).item():.4f}, 标准差: {std_value:.4f}")
        else:
            print("  没有因果图信息")

    # 保存验证指标到CSV文件
    import csv
    import os
    
    # 创建metrics目录（如果不存在）
    os.makedirs("metrics", exist_ok=True)
    
    # 组织所有指标到字典中
    metrics = {
        "dataset": dataNmae,
        "model": modelName,
        "threshold": threshold,
        "memory_size": TSFramework.GiveMemorySize(),
        "memory_clusters": len(TSFramework.MemoryClusterArr),
        "distance_type": distanceType,  # 添加距离类型
        "causal_enabled": "True",       # 标记使用了因果分析
        "fid_generation": fid1
    }
    
    # 使用模型名称创建文件名
    filename = f"metrics/{modelName}_metrics.csv"
    
    # 检查文件是否存在，决定是否写入标题行
    file_exists = os.path.isfile(filename)
    
    # 写入CSV文件
    with open(filename, mode='a', newline='') as file:
        fieldnames = list(metrics.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)
    
    print(f"验证指标已保存到 {filename}")

    # 添加一个简单的分析函数来显示共享样本的统计信息
    def analyze_shared_samples():
        total_shared = 0
        cluster_shares = []
        total_samples_count = 0
        sample_hashes = set()  # 跟踪所有集群中的所有样本
        
        print("\n共享样本分析:")
        for i, cluster in enumerate(TSFramework.MemoryClusterArr):
            if hasattr(cluster, 'shared_samples'):
                shared_count = len(cluster.shared_samples)
                total_count = cluster.GiveCount()
                share_percent = (shared_count / total_count * 100) if total_count > 0 else 0
                print(f"  簇 {i}: {shared_count}/{total_count} 样本共享 ({share_percent:.2f}%)")
                total_shared += shared_count
                cluster_shares.append(shared_count)
                
                # 添加到总样本集
                for sample in cluster.arr:
                    sample_hash = hash(sample.tobytes())
                    sample_hashes.add(sample_hash)
                    total_samples_count += 1
            else:
                print(f"  簇 {i}: 未初始化共享样本跟踪")
        
        if len(cluster_shares) > 0:
            print(f"总共有 {total_shared} 个共享样本实例")
            print(f"平均每个簇有 {sum(cluster_shares)/len(cluster_shares):.2f} 个共享样本")
        else:
            print("没有找到共享样本")
            
        # 分析训练样本覆盖情况
        total_training_samples = len(dataStramX)
        samples_in_clusters = len(sample_hashes)
        coverage_percent = (samples_in_clusters / total_training_samples * 100) if total_training_samples > 0 else 0
        
        print(f"\n样本参与分析:")
        print(f"  训练集总样本数: {total_training_samples}")
        print(f"  包含在集群中的样本数: {samples_in_clusters}")
        print(f"  样本覆盖率: {coverage_percent:.2f}%")
        if samples_in_clusters < total_training_samples:
            print(f"  警告: 有 {total_training_samples - samples_in_clusters} 个样本未被添加到任何集群中!")

    # 在训练完成后调用这个分析函数来查看共享样本的情况
    analyze_shared_samples()

def extract_clip_features(images, device):
    """
    从图像中提取CLIP特征
    
    Args:
        images: 图像数据，格式为 [batch_size, channels, height, width]
        device: 设备
    
    Returns:
        提取的特征向量 [batch_size, feature_dim]
    """
    global clip_model, clip_preprocess
    
    if clip_model is None:
        clip_model, clip_preprocess = initialize_clip_model(device)
    
    features = []
    batch_size = images.shape[0]
    
    # 转换图像格式以适应CLIP模型
    with torch.no_grad():
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
            img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            img_pil = Image.fromarray(img)
            img_preprocessed = clip_preprocess(img_pil).unsqueeze(0).to(device)
            feature = clip_model.encode_image(img_preprocessed)
            features.append(feature)
    
    return torch.cat(features, dim=0)

def extract_clip_features_with_cache(images, device):
    """使用限制大小的缓存提取CLIP特征"""
    features = []
    
    for img in images:
        # 生成缓存键 - 确保张量在CPU上
        if img.is_cuda:
            img_cpu = img.cpu()
        else:
            img_cpu = img
        cache_key = hash(img_cpu.numpy().tobytes())
        
        if cache_key in feature_cache:
            features.append(feature_cache[cache_key])
        else:
            try:
                # 限制缓存大小
                if len(feature_cache) >= max_cache_size:
                    # 删除最早添加的20%缓存项
                    old_keys = list(feature_cache.keys())[:int(max_cache_size*0.2)]
                    for k in old_keys:
                        del feature_cache[k]
                    
                    # 强制执行垃圾收集
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # 直接处理图像，避免递归调用
                img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                img_pil = Image.fromarray(img)
                img_preprocessed = clip_preprocess(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = clip_model.encode_image(img_preprocessed)
                    # 使用clone并分离确保不保留计算图
                    feature = feature.clone().detach()
                feature_cache[cache_key] = feature
                features.append(feature)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                feature = torch.zeros(1, 512, device=device)
                features.append(feature)
    
    # 确保features不为空，避免torch.cat错误
    if len(features) == 0:
        return torch.zeros(0, 512, device=device)
        
    result = torch.cat(features, dim=0)
    return result

# 定义简化版的因果图结构
class CausalGraph:
    def __init__(self, n_features=512):
        """
        初始化一个通用的因果图结构
        
        Args:
            n_features: 特征维度
        """
        self.n_features = n_features
        
        # 定义通用的特征组，适用于不同的数据集和任务
        self.feature_groups = {
            'identity': list(range(0, n_features//4)),                      # 身份特征：包含对象识别信息
            'appearance': list(range(n_features//4, n_features//2)),         # 外观特征：包含颜色、纹理等
            'pose': list(range(n_features//2, 3*n_features//4)),            # 姿势特征：包含形状、结构信息
            'background': list(range(3*n_features//4, n_features))          # 背景特征：包含上下文信息
        }
        
        # 定义通用的因果关系
        self.causal_relationships = {
            'identity': ['appearance', 'pose'],      # 身份影响外观和姿势
            'appearance': ['background'],            # 外观影响背景
            'pose': ['background'],                  # 姿势影响背景
            'background': []                         # 背景是最终表现
        }
        
        # 反向映射: 获取每个节点的父节点
        self.parent_map = self._build_parent_map()
        
        # 特征组权重，用于计算相似度时的加权
        self.feature_weights = {
            'identity': 0.4,    # 身份特征最重要
            'appearance': 0.3,  # 外观次之
            'pose': 0.2,        # 姿势再次之
            'background': 0.1   # 背景最不重要
        }
    
    def _build_parent_map(self):
        """构建每个特征组的父节点映射"""
        parent_map = {}
        for child in self.feature_groups.keys():
            parent_map[child] = []
            for parent, children in self.causal_relationships.items():
                if child in children:
                    parent_map[child].append(parent)
        return parent_map
    
    def get_parents(self, feature_group):
        """获取特定特征组的父节点"""
        return self.parent_map.get(feature_group, [])
    
    def get_feature_indices(self, feature_group):
        """获取特定特征组的索引"""
        return self.feature_groups.get(feature_group, [])
    
    def get_feature_weight(self, feature_group):
        """获取特征组的权重"""
        return self.feature_weights.get(feature_group, 1.0/len(self.feature_groups))
    
    def get_causal_mask(self, source_group, target_group):
        """
        生成因果掩码，表示source_group对target_group的因果影响
        
        如果source_group是target_group的父节点，则允许影响
        """
        mask = torch.zeros(self.n_features)
        
        # 如果source是target的父节点，或者就是target本身
        if source_group == target_group or source_group in self.get_parents(target_group):
            # 允许source_group中的特征影响target_group
            source_indices = self.get_feature_indices(source_group)
            for idx in source_indices:
                mask[idx] = 1.0
                
        return mask

def calculate_causal_similarity(feature1, feature2, causal_graph=None):
    """
    计算两个特征向量之间的因果相似度
    
    基于预定义的因果图结构评估相似度
    
    Args:
        feature1, feature2: 特征向量 [1, feature_dim]
        causal_graph: 可选的因果图结构
    
    Returns:
        因果关系强度分数 (0-1之间，越高表示越相似)
    """
    # 确保特征形状正确
    if feature1.ndim == 3:
        feature1 = feature1.squeeze(0)
    if feature2.ndim == 3:
        feature2 = feature2.squeeze(0)
    
    # 如果没有提供因果图，则创建一个
    if causal_graph is None:
        feature_dim = feature1.shape[1] if feature1.ndim > 1 else feature1.shape[0]
        causal_graph = CausalGraph(n_features=feature_dim)
    
    # 计算每个特征组的相似度
    group_similarities = {}
    total_weight = 0
    weighted_sum = 0
    
    # 为不同的特征组分配权重
    weights = {
        'identity': 0.4,    # 身份特征最重要
        'appearance': 0.3,  # 外观次之
        'pose': 0.2,        # 姿势再次之
        'background': 0.1   # 背景最不重要
    }
    
    for group, indices in causal_graph.feature_groups.items():
        # 提取当前特征组的特征
        if feature1.ndim > 1:
            f1_group = feature1[:, indices]
            f2_group = feature2[:, indices]
        else:
            f1_group = feature1[indices]
            f2_group = feature2[indices]
        
        # 计算余弦相似度
        f1_norm = F.normalize(f1_group.unsqueeze(0) if f1_group.ndim == 1 else f1_group, p=2, dim=1)
        f2_norm = F.normalize(f2_group.unsqueeze(0) if f2_group.ndim == 1 else f2_group, p=2, dim=1)
        
        # 计算当前组的相似度
        group_sim = torch.mm(f1_norm, f2_norm.t()).item()
        group_similarities[group] = group_sim
        
        # 累加加权相似度
        weight = weights.get(group, 1.0/len(causal_graph.feature_groups))
        weighted_sum += group_sim * weight
        total_weight += weight
    
    # 计算加权平均相似度
    if total_weight > 0:
        final_similarity = weighted_sum / total_weight
    else:
        # 如果没有权重，则使用平均值
        final_similarity = sum(group_similarities.values()) / len(group_similarities)
    
    return final_similarity

def calculate_causal_similarity_batch(feature1, features_list):
    """
    计算一个特征与一组特征之间的因果相似度
    
    Args:
        feature1: 单个特征向量 [1, feature_dim]
        features_list: 特征向量列表 [n, feature_dim]
    
    Returns:
        相似度分数列表
    """
    # 创建一个因果图以重用
    feature_dim = feature1.shape[1] if feature1.ndim > 1 else feature1.shape[0]
    causal_graph = CausalGraph(n_features=feature_dim)
    
    similarities = []
    
    for i in range(features_list.shape[0]):
        similarity = calculate_causal_similarity(
            feature1, 
            features_list[i:i+1], 
            causal_graph=causal_graph
        )
        similarities.append(similarity)
    
    return np.array(similarities)

def add_clip_support_to_memory_cluster():
    """
    为MemoryCluster类添加CLIP特征相关的方法和因果图属性
    
    添加的功能:
    1. 初始化因果图结构
    2. 计算基于CLIP特征的样本间距离
    3. 基于距离约束添加样本到簇中
    """
    # 给 MemoryCluster 类添加CLIP特征相关的方法和因果图属性
    def init_causal_graph(self):
        self.causal_graph = CausalGraph(n_features=512)  # CLIP ViT-B/32的特征维度是512
    
    MemoryCluster.init_causal_graph = init_causal_graph
    
    def calculate_clip_distance(self, sample1, sample2):
        device = self.device
        # 首次使用时初始化因果图
        if not hasattr(self, 'causal_graph'):
            self.init_causal_graph()
            
        # 确保输入是PyTorch张量
        if isinstance(sample1, np.ndarray):
            sample1 = torch.from_numpy(sample1).to(device)
        if isinstance(sample2, np.ndarray):
            sample2 = torch.from_numpy(sample2).to(device)
        
        # 提取CLIP特征
        # 将NumPy数组转换为PyTorch张量
        sample1_tensor = torch.tensor(sample1).to(device)
        feature1 = extract_clip_features(sample1_tensor.unsqueeze(0), device)
        # 将NumPy数组转换为PyTorch张量
        sample2_tensor = torch.tensor(sample2).to(device)
        feature2 = extract_clip_features(sample2_tensor.unsqueeze(0), device)
        
        # 使用因果相似度计算，转换为距离度量
        similarity = calculate_causal_similarity(feature1, feature2, self.causal_graph)
        distance = 1.0 - similarity
        return distance
            
    # 给MemoryCluster类添加新方法
    MemoryCluster.calculate_clip_distance = calculate_clip_distance
    
    # 覆盖CalculateWDistance_Individual方法
    original_method = MemoryCluster.CalculateWDistance_Individual
    
    def new_method(self, sample1, sample2):
        if self.distance_type == "CLIP":
            return self.calculate_clip_distance(sample1, sample2)
        else:
            return original_method(self, sample1, sample2)
            
    MemoryCluster.CalculateWDistance_Individual = new_method
    
    # 添加基于因果关系的样本添加方法
    original_add_sample = MemoryCluster.AddSingleSample
    
    def add_sample_with_causal_constraint(self, x1):
        # 首次使用时初始化因果图
        if not hasattr(self, 'causal_graph'):
            self.init_causal_graph()
            
        # 如果集群是空的，直接添加样本
        if len(self.arr) == 0:
            original_add_sample(self, x1)
            return True
        
        # 使用统一的距离计算方法
        distance = self.calculate_clip_distance(self.centreSample, x1)
        
        # 统一的距离阈值 - 所有方法共用
        distance_threshold = 0.28
        
        # 如果距离小于阈值，直接添加
        if distance < distance_threshold:
            original_add_sample(self, x1)
            return True  # 添加成功
        else:
            # 即使不满足约束，也添加样本，但打印警告
            print(f"警告: 样本与集群中心的距离 ({distance:.4f}) 高于阈值 {distance_threshold}，但仍添加到集群中")
            original_add_sample(self, x1)
            return False  # 添加成功但不满足约束
    
    MemoryCluster.add_sample_with_causal_constraint = add_sample_with_causal_constraint

def add_soft_clustering_support():
    """
    为记忆集群框架添加软聚类支持，允许样本同时属于多个集群
    
    添加的功能:
    1. 添加共享样本追踪能力
    2. 提供将样本添加为共享样本的方法
    3. 实现基础的软聚类批处理逻辑
    """
    # 添加标记共享样本的能力到MemoryCluster类
    def init_shared_samples_tracking(self):
        self.shared_samples = set()  # 存储共享样本的路径
        
    MemoryCluster.init_shared_samples_tracking = init_shared_samples_tracking
    
    # 扩展添加样本方法，添加标记共享样本的功能
    original_add_sample = MemoryCluster.AddSingleSample
    
    def add_sample_as_shared(self, x1, is_shared=False):
        # 确保已初始化共享样本跟踪
        if not hasattr(self, 'shared_samples'):
            self.init_shared_samples_tracking()
            
        # 如果集群为空，直接添加
        if len(self.arr) == 0:
            self.AddSingleSample(x1)
            return
            
        # 检查样本是否已存在 - 使用numpy的array_equal进行比较
        def is_sample_in_array(sample, array):
            for arr_sample in array:
                if np.array_equal(sample, arr_sample):
                    return True
            return False
        
        if is_sample_in_array(x1, self.arr):
            # 如果已存在并且标记为共享，只需更新标记
            if is_shared:
                self.shared_samples.add(hash(x1.tobytes()))
            return
            
        # 添加样本，并标记为共享样本
        self.AddSingleSample(x1)
        if is_shared:
            self.shared_samples.add(hash(x1.tobytes()))
    
    MemoryCluster.add_sample_as_shared = add_sample_as_shared
    
    # 基础版的软聚类批处理函数 - 主要实现基本逻辑，不含性能优化
    def add_data_batch_soft_clustering(self, x1):
        """
        基础版的软聚类支持数据批处理函数
        
        实现逻辑：
        1. 每个样本都加入到与其最相似的簇
        2. 如果样本与多个簇的距离都小于0.28，则加入所有这些簇
        3. 如果样本与所有簇的距离都大于0.36，则创建新簇
        """
        # 统一距离阈值
        add_threshold = 0.28  # 距离小于此值添加到多个簇中
        new_cluster_threshold = 0.36  # 距离大于此值创建新簇
        
        myarrScore = []
        
        n = np.shape(x1)[0]
        for i in range(n):
            data1 = x1[i]
            distances = []
            clusters = []
            
            # 计算与所有簇的距离
            for j in range(np.shape(self.MemoryClusterArr)[0]):
                m1 = self.MemoryClusterArr[j]
                # 确保簇已初始化共享样本跟踪
                if not hasattr(m1, 'shared_samples'):
                    m1.init_shared_samples_tracking()
                # 计算距离
                dis1 = m1.calculate_clip_distance(m1.centreSample, data1)
                distances.append(dis1)
                clusters.append(m1)
            
            # 如果没有簇，创建第一个
            if len(distances) == 0:
                arr = []
                arr.append(data1)
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                          self.input_size, self, self.distance_type)
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                newMemory.init_causal_graph()
                newMemory.init_shared_samples_tracking()
                myarrScore.append(1.0)  # 假设最大距离为1
                continue
            
            minDistance = np.min(distances)
            minIndex = np.argmin(distances)
            myarrScore.append(minDistance)
            
            # 找出所有距离小于阈值的簇
            close_clusters = [j for j, dist in enumerate(distances) if dist < add_threshold]
            
            # 情况1: 与所有簇距离都大于new_cluster_threshold -> 创建新簇
            if minDistance > new_cluster_threshold and np.shape(self.MemoryClusterArr)[0] < self.MaxMemoryCluster + 1:
                arr = []
                arr.append(data1)
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                         self.input_size, self, self.distance_type)
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                newMemory.init_causal_graph()
                newMemory.init_shared_samples_tracking()
            else:
                # 情况2: 距离小于add_threshold的簇，添加到所有这些簇
                if len(close_clusters) > 0:
                    # 检查是否有足够的空间
                    if len(close_clusters) > 1:
                        # 如果有任何簇接近容量上限，考虑删除一些记忆
                        counts = [self.MemoryClusterArr[idx].GiveCount() for idx in close_clusters]
                        if max(counts) >= self.MemoryClusterArr[0].maxMemorySize:
                            self.RemoveMemory()
                    
                    # 添加到所有足够近的簇中
                    for idx in close_clusters:
                        # 添加为共享样本，除了第一个
                        is_shared = len(close_clusters) > 1
                        self.MemoryClusterArr[idx].add_sample_as_shared(data1, is_shared)
                # 情况3: 没有距离小于阈值的簇，添加到最近的簇
                else:
                    self.MemoryClusterArr[minIndex].add_sample_with_causal_constraint(data1)
        
        # 返回最大距离作为扩展得分
        max_distance = np.max(myarrScore) if len(myarrScore) > 0 else 0
        return max_distance
    
    return add_data_batch_soft_clustering

def optimize_memory_framework():
    """
    优化记忆框架的性能和功能
    
    添加的功能:
    1. 实现特征缓存以提高性能
    2. 优化后的软聚类批处理算法
    3. 详细的日志输出
    """
    # 限制大小的特征缓存
    feature_cache = {}
    max_cache_size = 1000  # 限制缓存最大项数
    
    def extract_clip_features_with_cache(images, device):
        """使用限制大小的缓存提取CLIP特征"""
        features = []
        
        for img in images:
            # 生成缓存键 - 确保张量在CPU上
            if img.is_cuda:
                img_cpu = img.cpu()
            else:
                img_cpu = img
            cache_key = hash(img_cpu.numpy().tobytes())
            
            if cache_key in feature_cache:
                features.append(feature_cache[cache_key])
            else:
                try:
                    # 限制缓存大小
                    if len(feature_cache) >= max_cache_size:
                        # 删除最早添加的20%缓存项
                        old_keys = list(feature_cache.keys())[:int(max_cache_size*0.2)]
                        for k in old_keys:
                            del feature_cache[k]
                        
                        # 强制执行垃圾收集
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # 直接处理图像，避免递归调用
                    img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                    img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                    img_pil = Image.fromarray(img)
                    img_preprocessed = clip_preprocess(img_pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feature = clip_model.encode_image(img_preprocessed)
                        # 使用clone并分离确保不保留计算图
                        feature = feature.clone().detach()
                    feature_cache[cache_key] = feature
                    features.append(feature)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    feature = torch.zeros(1, 512, device=device)
                    features.append(feature)
        
        # 确保features不为空，避免torch.cat错误
        if len(features) == 0:
            return torch.zeros(0, 512, device=device)
            
        result = torch.cat(features, dim=0)
        return result
    
    global extract_clip_features
    extract_clip_features = extract_clip_features_with_cache
    
    def optimized_add_data_batch_soft_clustering(self, x1):
        """
        优化版的软聚类支持数据批处理函数
        
        实现逻辑：
        1. 每个样本都加入到与其最相似的簇
        2. 如果样本与多个簇的距离都小于0.28，则加入所有这些簇
        3. 如果样本与所有簇的距离都大于0.36，则创建新簇
        
        优化点:
        - 批量预提取特征，避免重复提取
        - 缓存中心点特征
        - 详细的日志输出
        """
        # 统一的距离阈值
        add_threshold = 0.28  # 距离小于此值添加到多个簇中
        new_cluster_threshold = 0.36  # 距离大于此值创建新簇
        
        batch_features = {}  # 缓存当前批次的样本特征
        myarrScore = []
        
        # 预提取所有集群中心点特征
        def update_center_features():
            center_features = {}
            for j, cluster in enumerate(self.MemoryClusterArr):
                if not hasattr(cluster, 'shared_samples'):
                    cluster.init_shared_samples_tracking()
                if not hasattr(cluster, 'causal_graph'):
                    cluster.init_causal_graph()
                # 确保centreSample是PyTorch张量
                if isinstance(cluster.centreSample, np.ndarray):
                    center_sample = torch.from_numpy(cluster.centreSample).to(self.device)
                else:
                    center_sample = cluster.centreSample
                # 确保center_sample是PyTorch张量
                if not isinstance(center_sample, torch.Tensor):
                    center_sample = torch.tensor(center_sample).to(self.device)
                center_features[j] = extract_clip_features(center_sample.unsqueeze(0), self.device)
            return center_features
            
        # 添加特征缓存保护
        def safe_extract_features(tensor_or_array, device):
            """安全地从张量或数组提取特征，确保输入是张量"""
            if not isinstance(tensor_or_array, torch.Tensor):
                tensor_or_array = torch.tensor(tensor_or_array).to(device)
            return extract_clip_features(tensor_or_array.unsqueeze(0), device)
        
        # 初始获取所有中心点特征
        center_features = update_center_features() if len(self.MemoryClusterArr) > 0 else {}
        
        n = np.shape(x1)[0]
        for i in range(n):
            data1 = x1[i]
            distances = []
            clusters = []
            
            print(f"\n处理样本 {i+1}/{n}")
            
            # 计算与所有簇的距离
            for j in range(np.shape(self.MemoryClusterArr)[0]):
                m1 = self.MemoryClusterArr[j]
                
                # 预先提取当前样本特征 - 避免重复提取
                # 使用样本数据的哈希值作为缓存键
                cache_key = hash(data1.tobytes())
                if cache_key not in batch_features:
                    # 确保数据是PyTorch张量
                    if isinstance(data1, np.ndarray):
                        data_tensor = torch.from_numpy(data1).to(self.device)
                    else:
                        data_tensor = data1
                    batch_features[cache_key] = extract_clip_features(data_tensor.unsqueeze(0), self.device)
                
                # 使用缓存的特征计算相似度
                similarity = calculate_causal_similarity(
                    batch_features[cache_key], 
                    center_features[j], 
                    m1.causal_graph
                )
                distance = 1.0 - similarity
                distances.append(distance)
                clusters.append(m1)
            
            # 如果没有簇，创建第一个
            if len(distances) == 0:
                print("● 没有现有簇，创建第一个簇")
                arr = []
                arr.append(data1)
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                         self.input_size, self, self.distance_type)
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                newMemory.init_causal_graph()
                newMemory.init_shared_samples_tracking()
                
                # 更新中心点特征缓存
                center_features = update_center_features()
                myarrScore.append(1.0)  # 假设最大距离为1
                continue
            
            minDistance = np.min(distances)
            minIndex = np.argmin(distances)
            print(f"● 最小距离: {minDistance:.4f} (簇{minIndex})")
            myarrScore.append(minDistance)
            
            # 找出所有距离小于阈值的簇
            close_clusters = [j for j, dist in enumerate(distances) if dist < add_threshold]
            
            # 检查簇的数量，确保不超过限制
            actual_cluster_count = min(len(self.MemoryClusterArr), self.MaxMemoryCluster)
            
            # 过滤掉超出实际簇数量的索引
            close_clusters = [idx for idx in close_clusters if idx < actual_cluster_count]
            
            # 判断是否创建新簇
            create_new_cluster = minDistance > new_cluster_threshold and len(self.MemoryClusterArr) < self.MaxMemoryCluster
            
            # 显示决策信息
            print(f"● 当前簇数: {len(self.MemoryClusterArr)}/{self.MaxMemoryCluster}")
            print(f"● 阈值: 添加={add_threshold}, 新建={new_cluster_threshold}")
            
            if close_clusters:
                print(f"● 距离小于{add_threshold}的簇: ", end="")
                for j in close_clusters:
                    print(f"簇{j}: {distances[j]:.4f}", end=" | ")
                print()
            else:
                print(f"● 没有簇距离小于{add_threshold}")
            
            # 确定样本应该添加到哪些簇
            # 情况1: 与所有簇距离都大于new_cluster_threshold -> 创建新簇
            if create_new_cluster:
                print(f"✓ 决策: 创建新簇 (最小距离{minDistance:.4f} > {new_cluster_threshold})")
                arr = []
                arr.append(data1)
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                         self.input_size, self, self.distance_type)
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                newMemory.init_causal_graph()
                newMemory.init_shared_samples_tracking()
                
                # 更新中心点特征缓存
                center_features = update_center_features()
            else:
                # 情况2: 距离小于add_threshold的簇，添加到所有这些簇
                if len(close_clusters) > 0:
                    print(f"✓ 决策: 添加到{len(close_clusters)}个簇 {close_clusters}")
                    
                    # 检查是否有足够的空间
                    if len(close_clusters) > 1:
                        # 如果有任何簇接近容量上限，考虑删除一些记忆
                        counts = [self.MemoryClusterArr[idx].GiveCount() for idx in close_clusters]
                        if max(counts) >= self.MemoryClusterArr[0].maxMemorySize:
                            self.RemoveMemory()
                    
                    # 添加到所有足够近的簇中
                    for idx in close_clusters:
                        # 添加为共享样本，除了第一个
                        is_shared = len(close_clusters) > 1
                        self.MemoryClusterArr[idx].add_sample_as_shared(data1, is_shared)
                # 情况3: 没有距离小于阈值的簇，添加到最近的簇
                else:
                    print(f"✓ 决策: 添加到最近的簇 {minIndex} (距离={minDistance:.4f})")
                    # 确保最近的簇索引在有效范围内
                    if minIndex < len(self.MemoryClusterArr):
                        self.MemoryClusterArr[minIndex].add_sample_with_causal_constraint(data1)
                    else:
                        # 如果索引无效，添加到第一个簇
                        print(f"警告: 最近的簇索引{minIndex}超出范围，添加到簇0")
                        self.MemoryClusterArr[0].add_sample_with_causal_constraint(data1)
        
        # 返回最大距离作为扩展得分
        max_distance = np.max(myarrScore) if len(myarrScore) > 0 else 0
        return max_distance
    
    return optimized_add_data_batch_soft_clustering

if __name__ == "__main__":
    main()
    #LoadModel()