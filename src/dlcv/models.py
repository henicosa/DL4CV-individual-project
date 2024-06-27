import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes, chosen_backbone):
    # load a pre-trained model for classification and return
    # only the features

    if chosen_backbone == "resnet50":
        backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").backbone
        backbone.out_channels = 256

    elif chosen_backbone == "resnet101":
        backbone = torchvision.models.resnet101(pretrained=True)
        backbone.out_channels = 2048  # ResNet-101 has 2048 output channels in its final layer

    else:
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        # ``FasterRCNN`` needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280
    
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
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