"""
Train a diffusion model on images.
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
from skimage import io,data
import types
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
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import  structural_similarity
from NetworkModels.MemoryUnitFramework_ import *
from NetworkModels.MemoryUnitGraphFramework_ import *

# 导入CLIP模型相关库
import clip
from PIL import Image
import torch.nn.functional as F

# 导入因果关系相关库
import numpy as np
import random
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# 全局CLIP模型变量
clip_model = None
clip_preprocess = None

def Transfer_To_Numpy(sample):
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    mySamples = sample.unsqueeze(0).cuda().cpu()
    mySamples = np.array(mySamples)
    mySamples = mySamples[0]
    return mySamples

def Save_Image(name,image):
    image = image.astype(np.float32)
    cv2.imwrite("results/" + name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# CLIP相关和因果分析相关函数（迁移自CACD128）
def initialize_clip_model(device):
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    return clip_model, clip_preprocess

def extract_clip_features(images, device):
    global clip_model, clip_preprocess
    if clip_model is None:
        clip_model, clip_preprocess = initialize_clip_model(device)
    features = []
    batch_size = images.shape[0]
    with torch.no_grad():
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0)
            img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            img_pil = Image.fromarray(img)
            img_preprocessed = clip_preprocess(img_pil).unsqueeze(0).to(device)
            feature = clip_model.encode_image(img_preprocessed)
            features.append(feature)
    return torch.cat(features, dim=0)

def extract_clip_features_from_path(data, device):
    global clip_model, clip_preprocess
    if clip_model is None:
        clip_model, clip_preprocess = initialize_clip_model(device)
    features = []
    
    # 如果输入是单个数据项，转换为列表
    if not isinstance(data, (list, np.ndarray)) or (isinstance(data, np.ndarray) and data.ndim <= 3):
        data = [data]
    
    with torch.no_grad():
        for item in data:
            try:
                if isinstance(item, str):
                    # 处理路径
                    img = Image.open(item).convert("RGB")
                    img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                elif isinstance(item, np.ndarray):
                    # 处理numpy数组图像数据
                    if item.ndim == 3:
                        # 单张图片 [C, H, W] 格式（从数据加载函数来的）
                        if item.shape[0] == 3:  # [C, H, W]
                            img_array = np.transpose(item, (1, 2, 0))  # 转换为 [H, W, C]
                        elif item.shape[-1] == 3:  # [H, W, C]
                            img_array = item
                        else:
                            raise ValueError(f"Unexpected image array shape: {item.shape}")
                    else:
                        raise ValueError(f"Unexpected image array shape: {item.shape}")
                    
                    # 确保数据在正确范围内
                    # 数据可能在[-1, 1]范围内，需要转换到[0, 255]
                    if img_array.min() < 0:
                        # 从[-1, 1]转换到[0, 255]
                        img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                    elif img_array.max() <= 1.0:
                        # 从[0, 1]转换到[0, 255]
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        # 已经在[0, 255]范围内
                        img_array = img_array.astype(np.uint8)
                    
                    img = Image.fromarray(img_array)
                    img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                else:
                    raise ValueError(f"Unsupported data type: {type(item)}")
                
                feature = clip_model.encode_image(img_preprocessed)
                features.append(feature)
            except Exception as e:
                print(f"Error processing image data: {e}")
                feature = torch.zeros(1, 512, device=device)
                features.append(feature)
    return torch.cat(features, dim=0)

class CausalGraph:
    def __init__(self, n_features=512):
        self.n_features = n_features
        self.feature_groups = {
            'semantic': list(range(0, n_features//3)),
            'structural': list(range(n_features//3, 2*n_features//3)),
            'style': list(range(2*n_features//3, n_features))
        }
        self.causal_relationships = {
            'semantic': ['structural', 'style'],
            'structural': ['style'],
            'style': []
        }
        self.parent_map = self._build_parent_map()
        self.feature_weights = {
            'semantic': 0.5,
            'structural': 0.3,
            'style': 0.2
        }
    def _build_parent_map(self):
        parent_map = {}
        for child in self.feature_groups.keys():
            parent_map[child] = []
            for parent, children in self.causal_relationships.items():
                if child in children:
                    parent_map[child].append(parent)
        return parent_map
    def get_parents(self, feature_group):
        return self.parent_map.get(feature_group, [])
    def get_feature_indices(self, feature_group):
        return self.feature_groups.get(feature_group, [])
    def get_feature_weight(self, feature_group):
        return self.feature_weights.get(feature_group, 1.0/len(self.feature_groups))
    def get_causal_mask(self, source_group, target_group):
        mask = torch.zeros(self.n_features)
        if source_group == target_group or source_group in self.get_parents(target_group):
            source_indices = self.get_feature_indices(source_group)
            for idx in source_indices:
                mask[idx] = 1.0
        return mask

def calculate_causal_similarity(feature1, feature2, causal_graph=None):
    if feature1.ndim == 3:
        feature1 = feature1.squeeze(0)
    if feature2.ndim == 3:
        feature2 = feature2.squeeze(0)
    if causal_graph is None:
        feature_dim = feature1.shape[1] if feature1.ndim > 1 else feature1.shape[0]
        causal_graph = CausalGraph(n_features=feature_dim)
    group_similarities = {}
    total_weight = 0
    weighted_sum = 0
    weights = {
        'identity': 0.4,
        'appearance': 0.3,
        'pose': 0.2,
        'background': 0.1
    }
    for group, indices in causal_graph.feature_groups.items():
        if feature1.ndim > 1:
            f1_group = feature1[:, indices]
            f2_group = feature2[:, indices]
        else:
            f1_group = feature1[indices]
            f2_group = feature2[indices]
        f1_norm = F.normalize(f1_group.unsqueeze(0) if f1_group.ndim == 1 else f1_group, p=2, dim=1)
        f2_norm = F.normalize(f2_group.unsqueeze(0) if f2_group.ndim == 1 else f2_group, p=2, dim=1)
        group_sim = torch.mm(f1_norm, f2_norm.t()).item()
        group_similarities[group] = group_sim
        weight = weights.get(group, 1.0/len(causal_graph.feature_groups))
        weighted_sum += group_sim * weight
        total_weight += weight
    if total_weight > 0:
        final_similarity = weighted_sum / total_weight
    else:
        final_similarity = sum(group_similarities.values()) / len(group_similarities)
    return final_similarity

def calculate_causal_similarity_batch(feature1, features_list):
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

# MemoryCluster 扩展

def add_clip_support_to_memory_cluster():
    def init_causal_graph(self):
        self.causal_graph = CausalGraph(n_features=512)
    MemoryCluster.init_causal_graph = init_causal_graph
    def calculate_clip_distance(self, data1, data2):
        device = self.device
        if not hasattr(self, 'causal_graph'):
            self.init_causal_graph()
        feature1 = extract_clip_features_from_path([data1], device)
        feature2 = extract_clip_features_from_path([data2], device)
        similarity = calculate_causal_similarity(feature1, feature2, self.causal_graph)
        distance = 1.0 - similarity
        return distance
    MemoryCluster.calculate_clip_distance = calculate_clip_distance
    original_method = MemoryCluster.CalculateWDistance_Individual_Files
    def new_method(self, data1, data2):
        if self.distance_type == "CLIP":
            return self.calculate_clip_distance(data1, data2)
        else:
            return original_method(self, data1, data2)
    MemoryCluster.CalculateWDistance_Individual_Files = new_method
    original_add_sample = MemoryCluster.AddSingleSample_Files
    def add_sample_with_causal_constraint(self, x1):
        if not hasattr(self, 'causal_graph'):
            self.init_causal_graph()
        if len(self.arr) == 0:
            original_add_sample(self, x1)
            return True
        distance = self.calculate_clip_distance(self.centreSample, x1)
        distance_threshold = 0.55
        if distance < distance_threshold:
            original_add_sample(self, x1)
            return True
        else:
            print(f"警告: 样本与集群中心的距离 ({distance:.4f}) 高于阈值 {distance_threshold}，但仍添加到集群中")
            original_add_sample(self, x1)
            return False
    MemoryCluster.add_sample_with_causal_constraint = add_sample_with_causal_constraint

def add_soft_clustering_support():
    def init_shared_samples_tracking(self):
        self.shared_samples = set()
    MemoryCluster.init_shared_samples_tracking = init_shared_samples_tracking
    original_add_sample = MemoryCluster.AddSingleSample_Files
    def add_sample_as_shared(self, x1, is_shared=False):
        if not hasattr(self, 'shared_samples'):
            self.init_shared_samples_tracking()
        if len(self.arr) == 0:
            original_add_sample(self, x1)
            return
        
        # 检查x1是否已经在arr中（处理numpy数组的比较）
        x1_exists = False
        if isinstance(x1, np.ndarray):
            for existing_item in self.arr:
                if isinstance(existing_item, np.ndarray) and np.array_equal(x1, existing_item):
                    x1_exists = True
                    break
        else:
            x1_exists = x1 in self.arr
            
        if x1_exists:
            if is_shared:
                # 使用字符串表示作为集合的键
                self.shared_samples.add(str(x1))
            return
        
        original_add_sample(self, x1)
        if is_shared:
            # 使用字符串表示作为集合的键
            self.shared_samples.add(str(x1))
    MemoryCluster.add_sample_as_shared = add_sample_as_shared
    def add_data_batch_soft_clustering(self, x1):
        add_threshold = 0.45
        new_cluster_threshold = 0.5
        myarrScore = []
        n = np.shape(x1)[0]
        for i in range(n):
            data1 = x1[i]
            distances = []
            clusters = []
            for j in range(np.shape(self.MemoryClusterArr)[0]):
                m1 = self.MemoryClusterArr[j]
                if not hasattr(m1, 'shared_samples'):
                    m1.init_shared_samples_tracking()
                dis1 = m1.calculate_clip_distance(m1.centreSample, data1)
                distances.append(dis1)
                clusters.append(m1)
            if len(distances) == 0:
                arr = []
                arr.append(data1)
                newMemory = MemoryCluster(self.device, arr, self.maxSizeForEachMemory, 
                                          self.input_size, self, self.distance_type)
                self.MemoryClusterArr.append(newMemory)
                self.memoryUnits = self.memoryUnits + 1
                newMemory.init_causal_graph()
                newMemory.init_shared_samples_tracking()
                myarrScore.append(1.0)
                continue
            minDistance = np.min(distances)
            minIndex = np.argmin(distances)
            myarrScore.append(minDistance)
            close_clusters = [j for j, dist in enumerate(distances) if dist < add_threshold]
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
                if len(close_clusters) > 0:
                    if len(close_clusters) > 1:
                        counts = [self.MemoryClusterArr[idx].GiveCount() for idx in close_clusters]
                        if max(counts) >= self.MemoryClusterArr[0].maxMemorySize:
                            self.RemoveMemory_Files()
                    for idx in close_clusters:
                        is_shared = len(close_clusters) > 1
                        self.MemoryClusterArr[idx].add_sample_as_shared(data1, is_shared)
                else:
                    self.MemoryClusterArr[minIndex].add_sample_with_causal_constraint(data1)
        max_distance = np.max(myarrScore) if len(myarrScore) > 0 else 0
        return max_distance
    return add_data_batch_soft_clustering

def optimize_memory_framework():
    feature_cache = {}
    max_cache_size = 1000
    def extract_clip_features_with_cache(data, device):
        features = []
        
        # 如果输入是单个数据项，转换为列表
        if not isinstance(data, (list, np.ndarray)) or (isinstance(data, np.ndarray) and data.ndim <= 3):
            data = [data]
        
        for item in data:
            # 为每个数据项创建一个唯一的缓存键
            if isinstance(item, str):
                # 如果是路径字符串
                cache_key = item
            elif isinstance(item, np.ndarray):
                # 如果是numpy数组，使用数组的哈希作为键
                cache_key = hash(item.tobytes())
            else:
                # 其他类型，转换为字符串
                cache_key = str(item)
            
            if cache_key in feature_cache:
                features.append(feature_cache[cache_key])
            else:
                try:
                    if len(feature_cache) >= max_cache_size:
                        old_keys = list(feature_cache.keys())[:int(max_cache_size*0.2)]
                        for k in old_keys:
                            del feature_cache[k]
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    if isinstance(item, str):
                        # 处理路径
                        img = Image.open(item).convert("RGB")
                        img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                    elif isinstance(item, np.ndarray):
                        # 处理numpy数组图像数据
                        if item.ndim == 3:
                            # 单张图片 [C, H, W] 格式（从数据加载函数来的）
                            if item.shape[0] == 3:  # [C, H, W]
                                img_array = np.transpose(item, (1, 2, 0))  # 转换为 [H, W, C]
                            elif item.shape[-1] == 3:  # [H, W, C]
                                img_array = item
                            else:
                                raise ValueError(f"Unexpected image array shape: {item.shape}")
                        else:
                            raise ValueError(f"Unexpected image array shape: {item.shape}")
                        
                        # 确保数据在正确范围内
                        # 数据可能在[-1, 1]范围内，需要转换到[0, 255]
                        if img_array.min() < 0:
                            # 从[-1, 1]转换到[0, 255]
                            img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                        elif img_array.max() <= 1.0:
                            # 从[0, 1]转换到[0, 255]
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            # 已经在[0, 255]范围内
                            img_array = img_array.astype(np.uint8)
                        
                        img = Image.fromarray(img_array)
                        img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                    else:
                        raise ValueError(f"Unsupported data type: {type(item)}")
                    
                    with torch.no_grad():
                        feature = clip_model.encode_image(img_preprocessed)
                        feature = feature.clone().detach()
                    feature_cache[cache_key] = feature
                    features.append(feature)
                    del img_preprocessed
                    if 'img' in locals():
                        del img
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    feature = torch.zeros(1, 512, device=device)
                    features.append(feature)
        
        if len(features) == 0:
            return torch.zeros(0, 512, device=device)
        result = torch.cat(features, dim=0)
        return result
    global extract_clip_features_from_path
    extract_clip_features_from_path = extract_clip_features_with_cache
    def optimized_add_data_batch_soft_clustering(self, x1):
        add_threshold = 0.4
        new_cluster_threshold = 0.45
        batch_features = {}
        myarrScore = []
        def update_center_features():
            center_features = {}
            for j, cluster in enumerate(self.MemoryClusterArr):
                if not hasattr(cluster, 'shared_samples'):
                    cluster.init_shared_samples_tracking()
                if not hasattr(cluster, 'causal_graph'):
                    cluster.init_causal_graph()
                center_features[j] = extract_clip_features_from_path([cluster.centreSample], self.device)
            return center_features
        center_features = update_center_features() if len(self.MemoryClusterArr) > 0 else {}
        n = np.shape(x1)[0]
        for i in range(n):
            data1 = x1[i]
            distances = []
            clusters = []
            print(f"\n处理样本 {i+1}/{n}: 数据项{i}")
            for j in range(np.shape(self.MemoryClusterArr)[0]):
                m1 = self.MemoryClusterArr[j]
                # 为numpy数组创建一个更好的缓存键
                if isinstance(data1, np.ndarray):
                    cache_key = hash(data1.tobytes())
                else:
                    cache_key = str(data1)
                
                if cache_key not in batch_features:
                    batch_features[cache_key] = extract_clip_features_from_path([data1], self.device)
                similarity = calculate_causal_similarity(
                    batch_features[cache_key], 
                    center_features[j], 
                    m1.causal_graph
                )
                distance = 1.0 - similarity
                distances.append(distance)
                clusters.append(m1)
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
                center_features = update_center_features()
                myarrScore.append(1.0)
                continue
            minDistance = np.min(distances)
            minIndex = np.argmin(distances)
            print(f"● 最小距离: {minDistance:.4f} (簇{minIndex})")
            myarrScore.append(minDistance)
            close_clusters = [j for j, dist in enumerate(distances) if dist < add_threshold]
            actual_cluster_count = min(len(self.MemoryClusterArr), self.MaxMemoryCluster)
            close_clusters = [idx for idx in close_clusters if idx < actual_cluster_count]
            create_new_cluster = minDistance > new_cluster_threshold and len(self.MemoryClusterArr) < self.MaxMemoryCluster
            print(f"● 当前簇数: {len(self.MemoryClusterArr)}/{self.MaxMemoryCluster}")
            print(f"● 阈值: 添加={add_threshold}, 新建={new_cluster_threshold}")
            if close_clusters:
                print(f"● 距离小于{add_threshold}的簇: ", end="")
                for j in close_clusters:
                    print(f"簇{j}: {distances[j]:.4f}", end=" | ")
                print()
            else:
                print(f"● 没有簇距离小于{add_threshold}")
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
                center_features = update_center_features()
            else:
                if len(close_clusters) > 0:
                    print(f"✓ 决策: 添加到{len(close_clusters)}个簇 {close_clusters}")
                    if len(close_clusters) > 1:
                        counts = [self.MemoryClusterArr[idx].GiveCount() for idx in close_clusters]
                        if max(counts) >= self.MemoryClusterArr[0].maxMemorySize:
                            self.RemoveMemory_Files()
                    for idx in close_clusters:
                        is_shared = len(close_clusters) > 1
                        self.MemoryClusterArr[idx].add_sample_as_shared(data1, is_shared)
                else:
                    print(f"✓ 决策: 添加到最近的簇 {minIndex} (距离={minDistance:.4f})")
                    if minIndex < len(self.MemoryClusterArr):
                        self.MemoryClusterArr[minIndex].add_sample_with_causal_constraint(data1)
                    else:
                        print(f"警告: 最近的簇索引{minIndex}超出范围，添加到簇0")
                        self.MemoryClusterArr[0].add_sample_with_causal_constraint(data1)
        max_distance = np.max(myarrScore) if len(myarrScore) > 0 else 0
        return max_distance
    return optimized_add_data_batch_soft_clustering

def LoadFFHQFromPath(data):
    """
    从图像数据加载并转换为tensor
    data: numpy数组，形状为 [batch_size, channels, height, width] 或路径列表
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # print(f"LoadFFHQFromPath input type: {type(data)}")
    # if isinstance(data, np.ndarray):
    #     print(f"LoadFFHQFromPath input shape: {data.shape}")
    
    if isinstance(data, np.ndarray):
        # 如果是numpy数组，直接转换为tensor
        if data.ndim == 4:
            # 数据应该已经是 [batch, channels, height, width] 格式
            # print(f"Converting 4D numpy array with shape {data.shape}")
            # 确保通道数是3
            if data.shape[1] == 3:
                batch = torch.from_numpy(data).float().to(device)
            else:
                raise ValueError(f"Expected 3 channels, got {data.shape[1]} channels")
        elif data.ndim == 3:
            # 单张图片，添加batch维度
            # print(f"Converting 3D numpy array with shape {data.shape}")
            if data.shape[0] == 3:
                batch = torch.from_numpy(data).unsqueeze(0).float().to(device)
            else:
                raise ValueError(f"Expected 3 channels, got {data.shape[0]} channels")
        elif data.ndim == 1:
            # 如果是一维数组，可能是路径列表
            return LoadFFHQFromPath(data.tolist())
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    elif isinstance(data, torch.Tensor):
        # print(f"Input is already a tensor with shape {data.shape}")
        batch = data.to(device)
    elif isinstance(data, (list, tuple)):
        # 如果是路径列表或包含图像数据的列表
        if len(data) == 0:
            return torch.zeros(0, 3, 64, 64, device=device)
        
        # 检查第一个元素的类型
        first_item = data[0]
        # print(f"List/tuple with first item type: {type(first_item)}")
        # if isinstance(first_item, np.ndarray):
        #     print(f"First item shape: {first_item.shape}")
        
        if isinstance(first_item, str):
            # 路径列表
            images = []
            for path in data:
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize((64, 64))
                    img_array = np.array(img)
                    # 归一化到[-1, 1]
                    img_array = (img_array / 127.5) - 1.0
                    images.append(img_array)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    default_img = np.zeros((64, 64, 3))
                    images.append(default_img)
            
            if len(images) == 0:
                return torch.zeros(0, 3, 64, 64, device=device)
            
            batch = np.array(images)
            # 转换为 [batch, channels, height, width]
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
        elif isinstance(first_item, np.ndarray):
            # numpy数组列表，每个元素是单张图片 [channels, height, width]
            if first_item.ndim == 3 and first_item.shape[0] == 3:
                # 每个元素是单张图片 [3, 64, 64]
                batch = np.stack(data, axis=0)  # 变成 [batch, 3, 64, 64]
                batch = torch.from_numpy(batch).float().to(device)
            else:
                raise ValueError(f"Unexpected item shape in list: {first_item.shape}")
        else:
            raise ValueError(f"Unsupported data type in list: {type(first_item)}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # print(f"LoadFFHQFromPath output shape: {batch.shape}")
    return batch

# 主流程 main() 调整

def main():
    distanceType = "CLIP"
    dataNmae = "CelebAtoImageNet"
    modelName = "GraphMemory"
    modelName = modelName + "_" + distanceType
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initialize_clip_model(device)
    trainX1,testX1 = Load_CelebA()
    trainX2, testX2 = Load_ImageNet()

    defaultTest = np.concatenate((testX1,testX2),0)
    n_examples = np.shape(defaultTest)[0]
    index2 = [i for i in range(n_examples)]
    np.random.shuffle(index2)
    defaultTest = defaultTest[index2]
    defaultTest = defaultTest[0:1000]

    trainX1 = np.concatenate((trainX1,trainX2),0)

    #
    print(trainX1[0])
    print(np.max(trainX1))

    #trainX1 = ((trainX1 + 1) * 127.5)
    #testX1 = ((testX1 + 1) * 127.5)
    dataStream = trainX1
    totalTestX = testX1
    miniBatch = 64
    totalTrainingTime = int(dataStream.shape[0] / miniBatch)
    inputSize = 32
    epoch = 1
    Tepoch = 100
    Sepoch = 100
    start = time.time()
    inputSize = 64
    TSFramework = MemoryUnitGraphFramework("myName",device,inputSize)
    TSFramework.distance_type = distanceType
    TSFramework.MaxMemoryCluster = 50
    TSFramework.OriginalInputSize = 64
    TSFramework.batch_size = 64
    add_clip_support_to_memory_cluster()
    add_soft_clustering_support()
    optimized_clustering_method = optimize_memory_framework()
    TSFramework.AddDataBatch_Files = types.MethodType(optimized_clustering_method, TSFramework)
    newComponent = TSFramework.Create_NewComponent()
    TSFramework.currentComponent = newComponent
    batchFiles = dataStream[0:miniBatch]
    print(f"Initial batchFiles shape: {batchFiles.shape}")
    print(f"Initial batchFiles type: {type(batchFiles)}")
    batch = LoadFFHQFromPath(batchFiles)
    print(f"Initial batch shape: {batch.shape}")
    newComponent.memoryBuffer = batch
    memoryBuffer = []
    maxMemorySize = 2000
    maxMemorySizeDefault = 2000
    dataloader = []
    threshold = 0.6
    TSFramework.threshold = threshold
    epoch = 3
    runStep = 0
    for step in range(TSFramework.currentTraningTime,totalTrainingTime):
        batchFiles = dataStream[step*miniBatch:(step + 1)*miniBatch]
        
        if np.shape(TSFramework.MemoryClusterArr)[0] == 0:
            TSFramework.MemoryBegin_Files(batchFiles)
            # 为第一个记忆集群初始化因果图
            TSFramework.MemoryClusterArr[0].init_causal_graph()

        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
              .format(step, totalTrainingTime, np.shape(TSFramework.MemoryClusterArr)[0], 0, 1, 0, 1))

        # TSFramework.currentMemory = batch
        print(np.shape(TSFramework.currentMemory))

        memoryBuffer = TSFramework.GiveMemorizedSamples()
        memoryBuffer = torch.tensor(memoryBuffer).cuda().to(device=device, dtype=torch.float)
        batch = LoadFFHQFromPath(batchFiles)

        memoryBuffer_ = torch.cat([memoryBuffer,batch],0)

        TSFramework.currentComponent.Train(epoch,memoryBuffer_)

        TSFramework.AddDataBatch_Files(batchFiles)

        # 在主循环中每10步执行一次清理
        if step % 10 == 0 and step > 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    print("记录因果特征统计信息:")
    for i, cluster in enumerate(TSFramework.MemoryClusterArr):
        print(f"集群 {i}:")
        if hasattr(cluster, 'causal_graph'):
            samples_features = []
            for sample_data in cluster.arr[:min(10, len(cluster.arr))]:
                feature = extract_clip_features_from_path([sample_data], device)
                samples_features.append(feature)
            if samples_features:
                samples_features = torch.cat(samples_features, dim=0)
                for group, indices in cluster.causal_graph.feature_groups.items():
                    group_features = samples_features[:, indices]
                    group_mean = torch.mean(group_features, dim=0)
                    if len(samples_features) > 1:
                        group_std = torch.std(group_features, dim=0)
                        std_value = torch.mean(group_std).item()
                    else:
                        std_value = 0.0
                    print(f"  {group} 特征组平均值: {torch.mean(group_mean).item():.4f}, 标准差: {std_value:.4f}")
        else:
            print("  没有因果图信息")
    myModelName = modelName + "_" + dataNmae + ".pkl"
    torch.save(TSFramework, './data/' + myModelName)
    end = time.time()
    print("Training times")
    print((end - start))
    print("Finish the training")
    gen = TSFramework.Give_GenerationFromTeacher(1000)
    generatedImages = Transfer_To_Numpy(gen)
    name_generation = dataNmae + "_" + modelName + str(threshold) + ".png"
    Save_Image(name_generation,merge2(generatedImages[0:1000], [10, 100]))
    # Evaluation
    # 假设有LoadFFHQFromPath函数，若没有请补充
    test_data = LoadFFHQFromPath(defaultTest)
    batch = test_data[0:64]
    reco = TSFramework.student.Give_Reconstruction(batch)
    myReco = Transfer_To_Numpy(reco)
    realBatch = Transfer_To_Numpy(batch)
    name = dataNmae + "_" + modelName + str(threshold) + "_" + "Real_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Real_small_" + str(0) + ".png"
    Save_Image(name,merge2(realBatch, [8, 8]))
    Save_Image(name_small,merge2(realBatch[0:16], [2, 8]))
    reco = Transfer_To_Numpy(reco)
    name = dataNmae + "_" + modelName + "_" + "Reco_" + str(0) + ".png"
    name_small = dataNmae + "_" + modelName + "_" + "Reco_small_" + str(0) + ".png"
    Save_Image(name, merge2(reco, [8, 8]))
    Save_Image(name_small, merge2(reco[0:16], [2, 8]))
    print("Generation")
    generated = TSFramework.Give_GenerationFromTeacher(1000)
    mytest = test_data[0:np.shape(generated)[0]]
    fid1 = calculate_fid_given_paths_Byimages(mytest, generated, 50, device, 2048)
    print(f"FID score: {fid1}")
    import csv
    import os
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "dataset": dataNmae,
        "model": modelName,
        "threshold": threshold,
        "memory_size": TSFramework.GiveMemorySize(),
        "memory_clusters": len(TSFramework.MemoryClusterArr),
        "distance_type": distanceType,
        "causal_enabled": "True",
        "fid_generation": fid1
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
    def analyze_shared_samples():
        total_shared = 0
        cluster_shares = []
        total_samples = set()
        print("\n共享样本分析:")
        for i, cluster in enumerate(TSFramework.MemoryClusterArr):
            if hasattr(cluster, 'shared_samples'):
                shared_count = len(cluster.shared_samples)
                total_count = cluster.GiveCount()
                share_percent = (shared_count / total_count * 100) if total_count > 0 else 0
                print(f"  簇 {i}: {shared_count}/{total_count} 样本共享 ({share_percent:.2f}%)")
                total_shared += shared_count
                cluster_shares.append(shared_count)
                # 对于numpy数组，使用字符串表示来避免集合操作问题
                for item in cluster.arr:
                    total_samples.add(str(item))
            else:
                print(f"  簇 {i}: 未初始化共享样本跟踪")
        if len(cluster_shares) > 0:
            print(f"总共有 {total_shared} 个共享样本实例")
            print(f"平均每个簇有 {sum(cluster_shares)/len(cluster_shares):.2f} 个共享样本")
        else:
            print("没有找到共享样本")
        total_training_samples = len(dataStream)
        samples_in_clusters = len(total_samples)
        coverage_percent = (samples_in_clusters / total_training_samples * 100) if total_training_samples > 0 else 0
        print(f"\n样本参与分析:")
        print(f"  训练集总样本数: {total_training_samples}")
        print(f"  包含在集群中的唯一样本数: {samples_in_clusters}")
        print(f"  样本覆盖率: {coverage_percent:.2f}%")
        # 注意：由于我们使用字符串表示，这个比较可能不完全准确
        print(f"  注意: 由于数据类型转换，覆盖率计算可能不完全准确")
    
    analyze_shared_samples()

if __name__ == "__main__":
    main()