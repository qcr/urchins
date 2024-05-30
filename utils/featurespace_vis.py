from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from time import time
import yaml
from ultralytics import YOLO
import cv2 as cv
import os
import glob
from ultralytics.engine.results import Results, Boxes
from tqdm import tqdm
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.getcwd()))
#from src.img2vec_resnet18 import Img2VecResnet18
from img2vec_pytorch import Img2Vec

with open("tracker/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)

# Extract configuration parameters
video_path = config["video_path_in"]
output_dir = config["save_dir"]
weights_path = config["detection_model_path"]

class Img2Vec2():
    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512, gpu=0):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        #self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def normalize2(self, tensor):
        """Normalize a tensor image with mean and standard deviation."""
        # Normalize tensor to [0, 1] range
        tensor = tensor / 255.0
        # You can apply additional normalization here if needed
        return tensor


    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
    
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
        #image = self.normalize2(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        if self.model_name in ['alexnet', 'vgg']:
            my_embedding = torch.zeros(1, self.layer_output_size)
        elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
            my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
        elif self.model_name == 'yolo':
            my_embedding = torch.zeros(1, self.layer_output_size, 4, 1029)
        else:
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)
        
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(image)
        h.remove()

        # import code
        # code.interact(local=dict(globals(), **locals()))

        if tensor:
            return my_embedding
        else:
            if self.model_name in ['alexnet', 'vgg']:
                return my_embedding.numpy()[0, :]
            elif self.model_name == 'densenet':
                return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
            else:
                return my_embedding.numpy()[0, :, 0, 0]
        

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)
            # import code
            # code.interact(local=dict(globals(), **locals()))
            return model, layer
        
        elif model_name == 'yolo':
            model = YOLO(weights_path)
            if layer == 'default':
                last_layer = None
                for module in model.modules():
                    if len(list(module.children())) == 0:
                        last_layer = module
                layer = last_layer
            else:
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

def crop_images(img_list, label_list, save_path):
    label_dict = {os.path.splitext(os.path.basename(label))[0]: label for label in label_list}
    cls_label = []
    for i, img_path in enumerate(img_list):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if img_name not in label_dict:
            continue
        img = cv.imread(img_path)
        img_width, img_height = img.shape[1], img.shape[0]
        label = label_dict[img_name]
        print(f" image basename {os.path.basename(img_path)}, labek {os.path.basename(label)}")
        label = label_dict[img_name]
        with open(label, 'r') as f:
            label_data = f.readlines()
        for j, line in enumerate(label_data):
            line = line.strip().split()
            cls, x, y, w, h = line
            x1 = int(float(x)*img_width - float(w)*img_width/2)
            x2 = int(float(x)*img_width + float(w)*img_width/2)
            y1 = int(float(y)*img_height - float(h)*img_height/2)
            y2 = int(float(y)*img_height + float(h)*img_height/2)
            crop_img = img[y1:y2, x1:x2]
            cls_label.append(int(cls))
            save_loc = os.path.join(save_path, f"{os.path.basename(img_path[:-4])}_{j}.jpg")
            cv.imwrite(save_loc, crop_img)
        # import code
        # code.interact(local=dict(globals(), **locals()))
    return cls_label

img_dir = "/home/java/Java/data/20231201_urchin/images/test"
label_dir = "/home/java/Java/data/20231201_urchin/labels/test"
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
label_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
save_path = "/home/java/Java/data/20231201_urchin/crops_from_images_test"

os.makedirs(save_path, exist_ok=True)
cls_label = crop_images(img_list, label_list, save_path)
print("done crop")
# import code
# code.interact(local=dict(globals(), **locals()))

crop_path = '/home/java/Java/data/20231201_urchin/crops_from_images_test'
crop_list= sorted(glob.glob(os.path.join(crop_path, '*.jpg')))
img_path = crop_path

# Function to display images on the scatter plot
def show_images(x, y, imagenes, cls_label, ax):
    for i in range(len(imagenes)):
        # Create an image box for each image using OffsetImage
        # image_box = OffsetImage(imagenes[i], zoom=0.6)
        # # Create an annotation box for each image at the corresponding coordinates
        # ab = AnnotationBbox(image_box, (x[i], y[i]), frameon=False)
        # # Add the annotation box to the plot
        # ax.add_artist(ab)
        if cls_label[i] == 0:
            ax.text(x[i], y[i], cls_label[i], fontsize=12, ha='right', va='bottom', color='blue')
        else:
            ax.text(x[i], y[i], cls_label[i], fontsize=12, ha='right', va='bottom', color='red')


#img2vec = Img2VecResnet18()
img2vec = Img2Vec2(model='yolo')
allVectors = {}
print("Converting imges to feature vectors:")
list_imgs = sorted(glob.glob(os.path.join(img_path, '*.jpg')))

for image in tqdm(list_imgs):
    I = Image.open(image)
    #vec = img2vec.getVec(I)
    vec = img2vec.get_vec(I)
    allVectors[image] = vec
    I.close()

embeddings = np.array(list(allVectors.values()))
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

images = []

# Iterate over all files in the directory specified by PATH
for image in tqdm(list_imgs):
    I = Image.open(image)
    I.thumbnail([100, 100], Image.Resampling.LANCZOS)
    images.append(I)


fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
show_images(embeddings_tsne[:, 0], embeddings_tsne[:, 1], images, cls_label, ax)
ax.set_title('t-SNE')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
plt.show()

import code
code.interact(local=dict(globals(), **locals()))



# with open("tracker/pipeline_config.yaml", "r") as f:
#         config = yaml.safe_load(f)

# # Extract configuration parameters
# video_path = config["video_path_in"]
# output_dir = config["save_dir"]
# weights_path = config["detection_model_path"]
# conf_threshold = config.get("detection_confidence_threshold")

# def extract_features_from_yolo_output(results):


#     return results

# # Load the YOLOv8 model
# model = YOLO(weights_path)
# img_loc = '/home/java/Java/data/20231201_urchin/images/val'
# img_list = sorted(glob.glob(os.path.join(img_loc, '*.jpg')))
# features = []
# for i, img_name in enumerate(img_list):
#     if i > 10:
#           break
#     img = cv.imread(img_name)
#     output = model.track(img)
#     pred = extract_features_from_yolo_output(output)
#     if len(pred) != 0:
#         features.append(pred)
# import code
# code.interact(local=dict(globals(), **locals()))

# # Replace this with your actual features and labels
# #features = np.random.rand(100, 100)  # Example random features
# labels = np.random.randint(2, size=100)  # Example random labels (0 or 1)

# def plot_embedding(X, title, labels):
#     _, ax = plt.subplots()
#     X = MinMaxScaler().fit_transform(X)

#     for label in np.unique(labels):
#         ax.scatter(
#             *X[labels == label].T,
#             label=f"Class {label}",
#             s=60,
#             alpha=0.425,
#             zorder=2,
#         )

#     ax.set_title(title)
#     ax.legend()
#     ax.axis("off")

# transformer = TSNE(
#     n_components=2,
#     n_iter=500,
#     n_iter_without_progress=150,
#     n_jobs=2,
#     random_state=0,
# )

# print("Computing t-SNE embedding...")
# start_time = time()
# projections = transformer.fit_transform(features)
# timing = time() - start_time

# title = f"t-SNE embedding (time {timing:.3f}s)"
# plot_embedding(projections, title, labels)

# plt.show()














########## DO TNSE for Digits


# digits = load_digits(n_class=6)
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape
# n_neighbors = 30

# fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
# for idx, ax in enumerate(axs.ravel()):
#     ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
#     ax.axis("off")
# _ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)

# plt.show()

# def plot_embedding(X, title):
#     _, ax = plt.subplots()
#     X = MinMaxScaler().fit_transform(X)

#     for digit in digits.target_names:
#         ax.scatter(
#             *X[y == digit].T,
#             marker=f"${digit}$",
#             s=60,
#             color=plt.cm.Dark2(digit),
#             alpha=0.425,
#             zorder=2,
#         )
#     shown_images = np.array([[1.0, 1.0]])  # just something big
#     for i in range(X.shape[0]):
#         # plot every digit on the embedding
#         # show an annotation box for a group of digits
#         dist = np.sum((X[i] - shown_images) ** 2, 1)
#         if np.min(dist) < 4e-3:
#             # don't show points that are too close
#             continue
#         shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
#         imagebox = offsetbox.AnnotationBbox(
#             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
#         )
#         imagebox.set(zorder=1)
#         ax.add_artist(imagebox)

#     ax.set_title(title)
#     ax.axis("off")


# projections, timing = {}, {}
# transformer = TSNE(
#         n_components=2,
#         n_iter=500,
#         n_iter_without_progress=150,
#         n_jobs=2,
#         random_state=0,
#     )
# data = X

# print("Computing t-SNE embedding...")
# start_time = time()
# projections = transformer.fit_transform(data, y)
# timing = time() - start_time

# title = f"t-SNE embedding (time {timing:.3f}s)"
# plot_embedding(projections, title)

# plt.show()
