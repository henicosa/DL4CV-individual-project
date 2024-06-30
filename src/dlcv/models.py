import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes, chosen_backbone, aspect_ratios):
    # load a pre-trained model for classification and return
    # only the features

    if chosen_backbone == "squeezenet1_1":
        # Load the pretrained SqueezeNet1_0 backbone.
        backbone = torchvision.models.squeezenet1_1(pretrained=True).features
        # We need the output channels of the last convolutional layers from
        # the features for the Faster RCNN model.
        # It is 512 for SqueezeNet1_0.
        backbone.out_channels = 512
        
        # Generate anchors using the RPN. Here, we are using 5x3 anchors.
        # Meaning, anchors with 5 different sizes and 3 different aspect 
        # ratios.
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=(aspect_ratios,)
        )
        # Feature maps to perform RoI cropping.
        # If backbone returns a Tensor, `featmap_names` is expected to
        # be [0]. We can choose which feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )


    elif chosen_backbone == "resnet101":
        resnet_net = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 2048
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=(aspect_ratios,)
        )
        # Feature maps to perform RoI cropping.
        # If backbone returns a Tensor, `featmap_names` is expected to
        # be [0]. We can choose which feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

    elif chosen_backbone == "mobilenet_v2":
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        # ``FasterRCNN`` needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280
            
        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=(aspect_ratios,)
        )
        
        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )


    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
