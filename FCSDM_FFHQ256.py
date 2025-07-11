"""
Train a diffusion model on images.
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'
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

def extract_clip_features_from_path(paths, device):
    global clip_model, clip_preprocess
    if clip_model is None:
        clip_model, clip_preprocess = initialize_clip_model(device)
    features = []
    with torch.no_grad():
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                feature = clip_model.encode_image(img_preprocessed)
                features.append(feature)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
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
    def calculate_clip_distance(self, path1, path2):
        device = self.device
        if not hasattr(self, 'causal_graph'):
            self.init_causal_graph()
        feature1 = extract_clip_features_from_path([path1], device)
        feature2 = extract_clip_features_from_path([path2], device)
        similarity = calculate_causal_similarity(feature1, feature2, self.causal_graph)
        distance = 1.0 - similarity
        return distance
    MemoryCluster.calculate_clip_distance = calculate_clip_distance
    original_method = MemoryCluster.CalculateWDistance_Individual_Files
    def new_method(self, path1, path2):
        if self.distance_type == "CLIP":
            return self.calculate_clip_distance(path1, path2)
        else:
            return original_method(self, path1, path2)
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
            print(f"警告: 样本 {os.path.basename(x1)} 与集群中心的距离 ({distance:.4f}) 高于阈值 {distance_threshold}，但仍添加到集群中")
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
        if x1 in self.arr:
            if is_shared:
                self.shared_samples.add(x1)
            return
        original_add_sample(self, x1)
        if is_shared:
            self.shared_samples.add(x1)
    MemoryCluster.add_sample_as_shared = add_sample_as_shared
    def add_data_batch_soft_clustering(self, x1):
        add_threshold = 0.55
        new_cluster_threshold = 0.6
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
    def extract_clip_features_with_cache(paths, device):
        features = []
        for path in paths:
            if path in feature_cache:
                features.append(feature_cache[path])
            else:
                try:
                    if len(feature_cache) >= max_cache_size:
                        old_keys = list(feature_cache.keys())[:int(max_cache_size*0.2)]
                        for k in old_keys:
                            del feature_cache[k]
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    img = Image.open(path).convert("RGB")
                    img_preprocessed = clip_preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feature = clip_model.encode_image(img_preprocessed)
                        feature = feature.clone().detach()
                    feature_cache[path] = feature
                    features.append(feature)
                    del img, img_preprocessed
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
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
            print(f"\n处理样本 {i+1}/{n}: {os.path.basename(data1)}")
            for j in range(np.shape(self.MemoryClusterArr)[0]):
                m1 = self.MemoryClusterArr[j]
                if data1 not in batch_features:
                    batch_features[data1] = extract_clip_features_from_path([data1], self.device)
                similarity = calculate_causal_similarity(
                    batch_features[data1], 
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

def LoadFFHQFromPath(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = [GetImage_cv(
            sample_file,
            input_height=256,
            input_width=256,
            resize_height=256,
            resize_width=256,
            crop=False)
            for sample_file in path]
    batch = np.array(batch)
    batch = batch.transpose(0, 3, 1, 2)
    batch = torch.tensor(batch).cuda().to(device=device, dtype=torch.float)
    return batch

# 主流程 main() 调整

def main():
    distanceType = "CLIP"
    dataNmae = "FFHQ256"
    modelName = "GraphMemory"
    modelName = modelName + "_" + distanceType
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initialize_clip_model(device)
    trainX1, testX1 = Load_FFHQ()  # 需保证有此函数
    defaultTest = testX1
    dataStream = trainX1
    totalTestX = testX1
    miniBatch = 64
    totalTrainingTime = int(dataStream.shape[0] / miniBatch)
    inputSize = 32
    epoch = 1
    Tepoch = 100
    Sepoch = 100
    start = time.time()
    inputSize = 256
    TSFramework = MemoryUnitGraphFramework("myName",device,inputSize)
    TSFramework.distance_type = distanceType
    TSFramework.MaxMemoryCluster = 20
    TSFramework.OriginalInputSize = 256
    TSFramework.batch_size = 24
    add_clip_support_to_memory_cluster()
    add_soft_clustering_support()
    optimized_clustering_method = optimize_memory_framework()
    TSFramework.AddDataBatch_Files = types.MethodType(optimized_clustering_method, TSFramework)
    newComponent = TSFramework.Create_NewComponent()
    TSFramework.currentComponent = newComponent
    batchFiles = dataStream[0:miniBatch]
    batch = LoadFFHQFromPath(batchFiles)
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
            TSFramework.MemoryClusterArr[0].init_causal_graph()
        print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}".format(step, totalTrainingTime, np.shape(TSFramework.MemoryClusterArr)[0], 0, 1, 0, 1))
        print(np.shape(TSFramework.currentMemory))
        memoryBuffer = TSFramework.GiveMemorizedSamples_ByFiles()
        memoryBuffer = torch.tensor(memoryBuffer).cuda().to(device=device, dtype=torch.float)
        batch = LoadFFHQFromPath(batchFiles)
        memoryBuffer_ = torch.cat([memoryBuffer,batch],0)
        TSFramework.currentComponent.Train(epoch,memoryBuffer_)
        TSFramework.AddDataBatch_Files(batchFiles)
        if step % 10 == 0 and step > 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    print("记录因果特征统计信息:")
    for i, cluster in enumerate(TSFramework.MemoryClusterArr):
        print(f"集群 {i}:")
        if hasattr(cluster, 'causal_graph'):
            samples_features = []
            for sample_path in cluster.arr[:min(10, len(cluster.arr))]:
                feature = extract_clip_features_from_path([sample_path], device)
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
    gen = TSFramework.Give_GenerationFromTeacher(100)
    generatedImages = Transfer_To_Numpy(gen)
    name_generation = dataNmae + "_" + modelName + str(threshold) + ".png"
    Save_Image(name_generation,merge2(generatedImages[0:64], [8, 8]))
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
                total_samples.update(cluster.arr)
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
        print(f"  包含在集群中的样本数: {samples_in_clusters}")
        print(f"  样本覆盖率: {coverage_percent:.2f}%")
        if samples_in_clusters < total_training_samples:
            print(f"  警告: 有 {total_training_samples - samples_in_clusters} 个样本未被添加到任何集群中!")
    analyze_shared_samples()

if __name__ == "__main__":
    main()