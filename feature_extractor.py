import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class CocoDataset(Dataset):
    def __init__(self, file_names):
        self.image_names = file_names
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformation = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(), normalize])
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transformation is not None:
            image = self.transformation(image)
        return image

    def __len__(self, ):
        return len(self.image_names)

class FeatureExtractor(object):
    """Extract features of images from dataset at a specified layer of a specified model

    Args:
        model_name (str): name of the model you want to use for extracting features (default: resnet101).
            Supported models: vgg16, vgg19, resnet50, resnet101, resnet152.

        layer (int): reversed index of the layer that you want to extract the feature from.
    """
    def __init__(self, model_name='resnet101', layer=3):
        if model_name.lower() == 'vgg16':
            orig_model = models.vgg16(pretrained=True)
        elif model_name.lower() == 'vgg19':
            orig_model = models.vgg19(pretrained=True)
        elif model_name.lower() == 'resnet50':
            orig_model = models.resnet50(pretrained=True)
        elif model_name.lower() == 'resnet152':
            orig_model = models.resnet152(pretrained=True)
        else:
            orig_model = models.resnet101(pretrained=True)

        self.model = torch.nn.Sequential(*list(orig_model.children())[:-layer])

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.cuda()

        self.model.eval()

    def __call__(self, images):
        images = torch.autograd.Variable(images.cuda())
        features = self.model(images)
        
        return features.permute(0, 2, 3, 1)
