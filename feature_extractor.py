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
        return int(image_name.phase('/')[-1].split('_')[-1].split('.')[0].lstrip('0')), image

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

class TensorFlowCocoDataset(object):
    def __init__(self, phases):
        def _parse_fn(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_image(image_string, channels=3)
            image_decoded.set_shape([None, None, 3])
            image_preprocessed = image_preprocessing_fn(image_decoded, 224, 224)
            return image_preprocessed

        self.datasets = {}
        for phase in phases:
            anno_path = './data/%s/%s.annotations.pkl' % (phase, phase)
            annotations = load_pickle(anno_path)
            file_names = list(annotations['file_name'].unique())
            image_ids = list(annotations['image_id'].unique())
            file_names_tensor = tf.constant(file_names)
            dataset = tf.data.Dataset.from_tensor_slices(file_names_tensor)
            dataset = dataset.map(_parse_fn).batch(batch_size)
            self.datasets[phase] = [dataset, image_ids]

        self.iterator = tf.data.Iterator.from_structure(datasets[datasets.keys()[0]][0].output_types, 
                                                        datasets[datasets.keys()[0]][0].output_shapes)

    def get_iter(self):
        return self.iterator

class TensorFlowFeatureExtracter(object):
    def __init__(self, model_name, model_num_layers, model_ckpt):
        # If resnet 101: use unit 22
        # If resnet 152: use unit 35
        unit = 22 if resnet_arch == '101' else 35 if resnet_arch == '152'
        model_name = 'resnet_v2_%s' % (resnet_arch)
        layer_name = model_name + '/block%d/unit_%d/bottleneck_v%d' % (3, unit, 2)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception', is_training=False)
        network_fn = nets_factory.get_network_fn(name=model_name, num_classes=None, is_training=False)

    def get_features(self, dataset_iterator): 
        dataset_batch = dataset_iterator.get_next()
        _, end_points = network_fn(dataset_batch)
        features = end_points[layer_name]
        return features